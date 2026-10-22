"""Image mitigation strategies – v2 overhaul.

Key improvements over v1:
  • QR:  size-adaptive multi-pass Gaussian blur (no colored blob)
  • Face: full opaque replacement via seamlessClone (no bleed-through)
  • PII:  erase-first, then render synthetic text at matching size/alignment
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    from faker import Faker
except ImportError:
    Faker = None  # type: ignore

from ..utils.config import ImageConfig
from ..utils.types import BoundingBox, DetectedEntity


class ImageMitigator:
    """Apply blurring or synthetic replacement to sensitive regions."""
# Labels that should always be blurred (not replaced with synthetic text)
BLUR_LABELS = frozenset({
    "face", "document_card", "id_number", "person_name",
    "institution_name", "sensitive_text",
    "PERSON", "ORG", "GPE", "CARDINAL", "MONEY",
    "email", "phone", "credit_card",
    "scene_text",  # In privacy context, blur all OCR text by default
})


class ImageMitigator:
    """Apply blurring to faces, text, and document regions for privacy protection."""

    def __init__(self, config: ImageConfig) -> None:
        self.config = config
        self.faker = Faker() if Faker else None

        # Load face assets
        assets_dir = Path(__file__).parent.parent / "utils" / "assets"
        self._face_assets = {
            "generic": self._load_asset(assets_dir / "generic_face.png"),
            "male": self._load_asset(assets_dir / "male_face.png"),
            "female": self._load_asset(assets_dir / "female_face.png"),
        }

    @staticmethod
    def _load_asset(path: Path) -> np.ndarray | None:
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            return img
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def mitigate(
        self,
        path: Path,
        entities: List[DetectedEntity],
        output_path: Path | None,
    ) -> Tuple[Path, List[DetectedEntity]]:
        original = cv2.imread(str(path))
        if original is None:
            return path, []

        if output_path is None:
            output_path = path.with_name(f"{path.stem}.sanitized{path.suffix}")

        sanitized_frame, updated_entities = self.mitigate_frame(original, entities)
        cv2.imwrite(str(output_path), sanitized_frame)
        return output_path, updated_entities

    def mitigate_frame(
        self,
        original: np.ndarray,
        entities: List[DetectedEntity],
    ) -> Tuple[np.ndarray, List[DetectedEntity]]:
        if original is None or original.size == 0:
            return original, []

        height, width = original.shape[:2]

        if self.config.only_mitigated_info:
            image = np.zeros((height, width, 3), dtype=np.uint8) + 30
        else:
            image = original.copy()

        updated_entities: list[DetectedEntity] = []
        image_identity_asset = None

        # Sort: process faces first, then QR, then text PII
        # This ensures the face is placed before text overwrites nearby areas
        label_order = {"face": 0, "qr_code": 1}
        sorted_entities = sorted(entities, key=lambda e: label_order.get(e.label, 2))

        # Deduplicate text PII entities that share the same bounding box.
        # OCR often returns one text line, but the text detector splits it into
        # multiple entities (PERSON + numbers + GPE + address) all pointing at
        # the same bbox.  Rendering all of them would cause overlapping text at
        # wrong sizes.  We keep only the highest-priority entity per bbox.
        _label_priority = {
            "address": 0, "STREET_ADDRESS": 0,
            "PERSON": 1, "NAME": 1, "name": 1,
            "license": 2, "roll_number": 2,
            "GPE": 3, "LOCATION": 3,
            "ORG": 4,
            "dob": 5,
            "aadhar": 6,
            "numbers": 7,
            "CARDINAL": 8,
        }

        def _bbox_key(e: DetectedEntity) -> tuple:
            b = e.bbox
            return (b.x, b.y, b.width, b.height) if b else (0, 0, 0, 0)

        def _bboxes_overlap(a: DetectedEntity, b: DetectedEntity, threshold: float = 0.3) -> bool:
            """Return True if two bboxes overlap or are vertically adjacent at the same X.

            Uses intersection-over-smaller-area with a low threshold, plus a
            vertical-proximity heuristic for multi-line address fields.
            """
            if not a.bbox or not b.bbox:
                return False
            ax1, ay1 = a.bbox.x, a.bbox.y
            ax2, ay2 = ax1 + a.bbox.width, ay1 + a.bbox.height
            bx1, by1 = b.bbox.x, b.bbox.y
            bx2, by2 = bx1 + b.bbox.width, by1 + b.bbox.height

            # Check geometric intersection
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = a.bbox.width * a.bbox.height
                area_b = b.bbox.width * b.bbox.height
                smaller = min(area_a, area_b)
                if smaller > 0 and (inter / smaller) > threshold:
                    return True

            # Vertical adjacency: only for address fields of the same field class,
            # to avoid merging separate fields like name, dob, and roll number
            is_address_a = a.label in {"address", "STREET_ADDRESS", "GPE", "LOCATION"}
            is_address_b = b.label in {"address", "STREET_ADDRESS", "GPE", "LOCATION"}
            if is_address_a and is_address_b:
                x_overlap = min(ax2, bx2) - max(ax1, bx1)
                min_w = min(a.bbox.width, b.bbox.width)
                if min_w > 0 and x_overlap / min_w > 0.5:
                    gap = max(0, max(ay1, by1) - min(ay2, by2))
                    line_h = max(a.bbox.height, b.bbox.height)
                    if gap < line_h * 0.5:
                        return True
 
            return False

        # Separate face/qr (always processed) from text PII (needs dedup)
        non_text = [e for e in sorted_entities if e.label in {"face", "qr_code"}]
        text_pii = [e for e in sorted_entities if e.label not in {"face", "qr_code"}]

        # Group text PII by overlapping bboxes; keep highest priority per group
        # CRITICAL: merge all bboxes in the group into one unified bbox so
        # the ENTIRE region gets erased (no original text bleed-through).
        deduped_text: list[DetectedEntity] = []
        used: set[int] = set()
        for i, ent_i in enumerate(text_pii):
            if i in used:
                continue
            # Find all entities overlapping with ent_i
            group = [ent_i]
            used.add(i)
            for j, ent_j in enumerate(text_pii):
                if j in used:
                    continue
                # Check overlap against ALL members of the group (transitive)
                if any(_bboxes_overlap(member, ent_j) for member in group):
                    group.append(ent_j)
                    used.add(j)
            # Pick the best entity from the group
            best = min(group, key=lambda e: _label_priority.get(e.label, 99))

            # Merge all bboxes into one unified bbox covering the full region
            if len(group) > 1 and best.bbox:
                all_bboxes = [e.bbox for e in group if e.bbox]
                if all_bboxes:
                    min_x = min(b.x for b in all_bboxes)
                    min_y = min(b.y for b in all_bboxes)
                    max_x2 = max(b.x + b.width for b in all_bboxes)
                    max_y2 = max(b.y + b.height for b in all_bboxes)
                    merged_bbox = BoundingBox(
                        x=min_x, y=min_y,
                        width=max_x2 - min_x, height=max_y2 - min_y,
                    )
                    # Concatenate original texts for proper font-scale sizing
                    all_texts = [e.text for e in group if e.text]
                    longest_text = max(all_texts, key=len) if all_texts else best.text
                    best = best.model_copy(update={
                        "bbox": merged_bbox,
                        "text": longest_text,
                    })
            deduped_text.append(best)

        sorted_entities = non_text + deduped_text

        for entity in sorted_entities:
            if not entity.bbox:
                continue

            # Label filtering
            if self.config.mitigated_labels and entity.label not in self.config.mitigated_labels:
                updated_entities.append(entity)
                continue

            # Compute padded region
            is_text_pii = entity.label not in {"face", "qr_code", "person"}
            pad_pct = 0.0 if is_text_pii else self.config.padding_percent
            padding = int(max(entity.bbox.width, entity.bbox.height) * pad_pct)
            x = max(0, int(entity.bbox.x) - padding)
            y = max(0, int(entity.bbox.y) - padding)
            w = min(width - x, int(entity.bbox.width) + 2 * padding)
            h = min(height - y, int(entity.bbox.height) + 2 * padding)

            roi_original = original[y : y + h, x : x + w]
            roi_current = image[y : y + h, x : x + w]
            if roi_original.size == 0:
                continue

            style = entity.mitigation if entity.mitigation != "none" else self.config.default_style
            if entity.label in {"qr_code", "aadhar", "mrz_zone"} and style not in {"synthetic", "replace"}:
                style = "aggressive_blur"

            if entity.label == "face" and (entity.mitigation in {"synthetic", "replace"} or self.config.default_style in {"synthetic", "replace"}):
                if style in {"blur", "synthetic"}:
                    style = "replace"

            mitigated_text = entity.text
            mapped_style = "synthetic" if style == "replace" else style

            if mapped_style == "synthetic":
                if entity.label == "face":
                    # Skip face replacement for tiny faces (watermarks, ghost images)
                    if w < 40 or h < 40:
                        image[y : y + h, x : x + w] = cv2.GaussianBlur(roi_original, (31, 31), 10)
                        mitigated_text = "[BLURRED_SMALL_FACE]"
                    else:
                        image, mitigated_text, image_identity_asset = self._handle_face(
                            image, original, roi_original, entity, x, y, w, h, image_identity_asset
                        )
                elif entity.label == "qr_code":
                    image[y : y + h, x : x + w] = self._blur_qr(roi_original)
                    mitigated_text = "[QR_BLURRED]"
                else:
                    # Use roi_current so previous erasures are visible (no double text)
                    roi_result, mitigated_text = self._replace_pii_text(roi_current, entity)
                    image[y : y + h, x : x + w] = roi_result
            elif mapped_style == "aggressive_blur":
                image[y : y + h, x : x + w] = self._blur_qr(roi_original) if entity.label == "qr_code" else self._aggressive_blur(roi_original)
                mitigated_text = "[AGGRESSIVE_BLUR]"
            elif mapped_style == "pixelate":
                image[y : y + h, x : x + w] = self._pixelate_region(roi_original)
                mitigated_text = "[PIXELATED]"
            elif mapped_style == "redact":
                image[y : y + h, x : x + w] = self._redact_region(roi_original)
                mitigated_text = "[REDACTED]"
            else:
                image[y : y + h, x : x + w] = self._blur_region(roi_original)
                mitigated_text = "[BLURRED]"

            updated_entities.append(
                entity.model_copy(
                    update={
                        "mitigation": style,
                        "text": mitigated_text,
                        "explanation": entity.text,
                    }
                )
            )

        return image, updated_entities

    # ------------------------------------------------------------------
    # 1. QR CODE – size-adaptive multi-pass Gaussian blur
    # ------------------------------------------------------------------
    def _blur_qr(self, roi: np.ndarray) -> np.ndarray:
        """Blur QR code region with kernel proportional to its size.

        Uses multi-pass Gaussian blur to ensure the pattern is completely
        unreadable while preserving the surrounding color naturally
        (no colored-blob artifact).
        """
        h, w = roi.shape[:2]
        # Kernel = ~40% of the smaller dimension, clamped and forced odd
        base_k = max(h, w) * 4 // 10
        kernel = max(31, base_k)
        if kernel % 2 == 0:
            kernel += 1
        kernel = min(kernel, 255)

        result = roi.copy()
        # 3 passes for thorough destruction of the QR pattern
        for _ in range(3):
            result = cv2.GaussianBlur(result, (kernel, kernel), 0)
        return result

    # ------------------------------------------------------------------
    # 2. FACE – identity-altering perturbation (primary) or asset swap (fallback)
    # ------------------------------------------------------------------
    def _handle_face(
        self,
        image: np.ndarray,
        original: np.ndarray,
        roi: np.ndarray,
        entity: DetectedEntity,
        x: int, y: int, w: int, h: int,
        image_identity_asset: np.ndarray | None,
    ) -> Tuple[np.ndarray, str, np.ndarray | None]:
        """Alter face identity while preserving natural appearance.

        Primary strategy: geometrically warp the original face to change
        identity while keeping skin tone, lighting, hair, clothing, and
        background intact.

        Fallback: paste a stock asset with improved color adaptation when
        warping cannot produce a convincing result.
        """

        img_h, img_w = image.shape[:2]

        # --- Try perturbation first (keeps everything natural) ---
        perturbed = self._perturb_face(original, x, y, w, h)
        if perturbed is not None:
            # Successfully perturbed — paste it back with feathered edges
            # Expand slightly beyond the face bbox to cover forehead/chin
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.20)
            px = max(0, x - pad_x)
            py = max(0, y - pad_y)
            pw = min(img_w - px, w + 2 * pad_x)
            ph = min(img_h - py, h + 2 * pad_y)

            perturbed_region = perturbed[py : py + ph, px : px + pw]
            if perturbed_region.size == 0 or perturbed_region.shape[:2] != (ph, pw):
                perturbed_region = cv2.resize(perturbed[y:y+h, x:x+w], (pw, ph))

            # Create feathered mask for seamless blending
            mask = np.zeros((ph, pw), dtype=np.uint8)
            margin = max(2, min(pw, ph) // 10)
            mask[margin:ph - margin, margin:pw - margin] = 255
            feather_k = max(3, margin * 2 + 1)
            if feather_k % 2 == 0:
                feather_k += 1
            mask = cv2.GaussianBlur(mask, (feather_k, feather_k), margin * 0.6)

            alpha = mask.astype(np.float32) / 255.0
            alpha_3ch = cv2.merge([alpha, alpha, alpha])
            current_roi = image[py : py + ph, px : px + pw]
            blended = (perturbed_region.astype(np.float32) * alpha_3ch +
                       current_roi.astype(np.float32) * (1.0 - alpha_3ch))
            image[py : py + ph, px : px + pw] = blended.clip(0, 255).astype(np.uint8)

            return image, "[PERTURBED_FACE]", image_identity_asset

        # --- Fallback: stock asset replacement with improved blending ---
        gender = entity.metadata.get("gender", "generic")
        if image_identity_asset is None:
            image_identity_asset = self._face_assets.get(gender)
            if image_identity_asset is None:
                image_identity_asset = self._face_assets.get("generic")

        asset = image_identity_asset
        if asset is None:
            image[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (99, 99), 30)
            return image, "[BLURRED_FACE]", None

        # Tighter region expansion (head+neck only, not full shoulders)
        photo_x = max(0, x - int(w * 0.35))
        photo_y = max(0, y - int(h * 0.25))
        photo_w = min(img_w - photo_x, int(w * 1.7))
        photo_h = min(img_h - photo_y, int(h * 2.0))

        photo_roi = original[photo_y : photo_y + photo_h, photo_x : photo_x + photo_w]
        if photo_roi.size == 0:
            photo_roi = roi
            photo_x, photo_y, photo_w, photo_h = x, y, w, h

        # Resize asset to fill the photo region

        ah, aw = asset.shape[:2]
        crop_top = int(ah * 0.02)
        crop_bottom = int(ah * 0.85)
        crop_left = int(aw * 0.05)
        crop_right = int(aw * 0.95)
        asset_crop = asset[crop_top:crop_bottom, crop_left:crop_right]
        face_resized = cv2.resize(asset_crop, (photo_w, photo_h), interpolation=cv2.INTER_LANCZOS4)

        # --- Stronger LAB color adaptation (85% blend to target) ---
        face_lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi_lab = cv2.cvtColor(photo_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            m_roi, s_roi = cv2.meanStdDev(roi_lab[:, :, ch])
            m_face, s_face = cv2.meanStdDev(face_lab[:, :, ch])
            scale = (s_roi[0][0] / (s_face[0][0] + 1e-6))
            scale = 0.15 + 0.85 * min(scale, 2.0)
            face_lab[:, :, ch] = (
                (face_lab[:, :, ch] - m_face[0][0]) * scale +
                m_face[0][0] * 0.15 + m_roi[0][0] * 0.85
            )

        face_lab = face_lab.clip(0, 255).astype(np.uint8)
        face_matched = cv2.cvtColor(face_lab, cv2.COLOR_LAB2BGR)

        # --- Background color matching ---
        # Extract dominant bg color from photo_roi borders and apply to asset bg
        border_pixels = np.concatenate([
            photo_roi[0, :], photo_roi[-1, :],
            photo_roi[:, 0], photo_roi[:, -1],
        ])
        bg_color = np.median(border_pixels, axis=0).astype(np.uint8)

        # Detect asset background via simple thresholding in LAB
        face_gray = cv2.cvtColor(face_matched, cv2.COLOR_BGR2GRAY)
        # Background tends to be relatively uniform; detect via edges
        edges = cv2.Canny(face_matched, 30, 100)
        bg_mask = cv2.dilate(edges, None, iterations=3)
        bg_mask = cv2.bitwise_not(bg_mask)
        # Erode to keep only confident background pixels
        bg_mask = cv2.erode(bg_mask, None, iterations=5)

        # Tint the background region toward the original photo background color
        bg_color_img = np.full_like(face_matched, bg_color)
        bg_alpha = (bg_mask.astype(np.float32) / 255.0)
        bg_alpha_3ch = cv2.merge([bg_alpha, bg_alpha, bg_alpha])
        # 50% blend to tint background without making it fully flat
        face_matched = (
            face_matched.astype(np.float32) * (1.0 - bg_alpha_3ch * 0.5) +
            bg_color_img.astype(np.float32) * (bg_alpha_3ch * 0.5)
        ).clip(0, 255).astype(np.uint8)

        # Match sharpness
        roi_gray = cv2.cvtColor(photo_roi, cv2.COLOR_BGR2GRAY)
        roi_sharpness = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        if roi_sharpness < 300:
            sigma = max(0.3, 2.0 * (1.0 - (roi_sharpness / 300.0)))
            ks = int(round(sigma * 3.0)) * 2 + 1
            face_matched = cv2.GaussianBlur(face_matched, (ks, ks), sigma)

        # Feathered mask
        mask = np.zeros((photo_h, photo_w), dtype=np.uint8)
        margin = max(3, min(photo_w, photo_h) // 12)
        mask[margin:photo_h - margin, margin:photo_w - margin] = 255
        feather_k = margin * 2 + 1
        if feather_k % 2 == 0:
            feather_k += 1
        mask = cv2.GaussianBlur(mask, (feather_k, feather_k), margin * 0.7)

        # JPEG noise matching
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, enc = cv2.imencode('.jpg', face_matched, encode_param)
        face_matched = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if face_matched.shape[:2] != (photo_h, photo_w):
            face_matched = cv2.resize(face_matched, (photo_w, photo_h))

        # Blend into image
        if not self.config.only_mitigated_info:
            alpha = mask.astype(np.float32) / 255.0
            alpha_3ch = cv2.merge([alpha, alpha, alpha])
            current_roi = image[photo_y : photo_y + photo_h, photo_x : photo_x + photo_w]
            blended = (face_matched.astype(np.float32) * alpha_3ch +
                       current_roi.astype(np.float32) * (1.0 - alpha_3ch))
            image[photo_y : photo_y + photo_h, photo_x : photo_x + photo_w] = blended.clip(0, 255).astype(np.uint8)
        else:
            image[photo_y : photo_y + photo_h, photo_x : photo_x + photo_w] = face_matched

        return image, "[REPLACED_FACE]", image_identity_asset

    # ------------------------------------------------------------------
    # 2b. Face perturbation — warp original face to alter identity
    # ------------------------------------------------------------------
    def _perturb_face(
        self,
        original: np.ndarray,
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray | None:
        """Geometrically warp the face to alter identity while keeping it natural.

        Uses a grid-based piecewise affine warp to shift facial features:
        - Eyes are shifted/resized
        - Nose region is widened/narrowed
        - Jawline is altered
        - Face proportions (width/height) change subtly

        The result looks like a real person but is NOT the original identity.
        Skin tone, lighting, background, hair, and clothing are all preserved.

        Returns the full image with the face region warped, or None if
        the face is too small for convincing perturbation.
        """
        img_h, img_w = original.shape[:2]

        # Need a reasonable face size for grid warping to look good
        if w < 40 or h < 40:
            return None

        # Work on an expanded region around the face for smooth transitions
        expand = 0.6
        ex = max(0, int(x - w * expand))
        ey = max(0, int(y - h * expand))
        ew = min(img_w - ex, int(w * (1 + 2 * expand)))
        eh = min(img_h - ey, int(h * (1 + 2 * expand)))

        if ew < 20 or eh < 20:
            return None

        roi = original[ey : ey + eh, ex : ex + ew].copy()
        rh, rw = roi.shape[:2]

        # Face center relative to the ROI
        cx = (x - ex) + w // 2
        cy = (y - ey) + h // 2

        # Use deterministic seed based on face position for consistency
        rng = np.random.RandomState(seed=(x * 7 + y * 13 + w * 3) % (2**31))

        # --- Build source and destination control points for warping ---
        # Dense grid for fine-grained control over facial feature distortion
        grid_rows, grid_cols = 8, 8

        def _build_warp_points(rng_state, roi_w, roi_h, face_cx, face_cy, face_w, face_h):
            """Build src/dst point pairs for one warp pass."""
            src_pts_list = []
            dst_pts_list = []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    px = int(c * roi_w / (grid_cols - 1))
                    py_pt = int(r * roi_h / (grid_rows - 1))
                    src_pts_list.append([px, py_pt])

                    # Distance from face center (normalized to face size)
                    ddx = (px - face_cx) / (face_w / 2 + 1e-6)
                    ddy = (py_pt - face_cy) / (face_h / 2 + 1e-6)
                    dist = np.sqrt(ddx * ddx + ddy * ddy)

                    # Perturb points within 1.5 face radii (wide coverage)
                    if dist < 1.5:
                        # Smooth falloff from center to edge
                        strength = max(0, 1.0 - dist / 1.5) ** 0.8
                        # Moderate shift: 11% of face size
                        max_shift = min(face_w, face_h) * 0.11

                        shift_x = 0.0
                        shift_y = 0.0

                        # Eye region (upper face): shift to change eye spacing/shape
                        if -0.9 < ddy < -0.1 and abs(ddx) < 1.0:
                            shift_x = rng_state.uniform(-1, 1) * max_shift * 0.9
                            shift_y = rng_state.uniform(-0.6, 0.6) * max_shift * 0.6

                        # Nose region (center): widen or narrow strongly
                        elif -0.3 < ddy < 0.4 and abs(ddx) < 0.6:
                            widen = rng_state.choice([-1, 1])
                            shift_x = widen * (np.sign(ddx) if abs(ddx) > 0.1 else rng_state.choice([-1, 1])) * max_shift * 0.8
                            shift_y = rng_state.uniform(-0.5, 0.5) * max_shift * 0.6

                        # Mouth region: stretch or compress
                        elif 0.3 < ddy < 0.7 and abs(ddx) < 0.7:
                            shift_x = rng_state.uniform(-1, 1) * max_shift * 0.7
                            shift_y = rng_state.uniform(-0.6, 0.6) * max_shift * 0.5

                        # Jaw/chin region (lower face): reshape jawline
                        elif 0.5 < ddy < 1.3 and abs(ddx) < 1.2:
                            shift_x = rng_state.uniform(-1, 1) * max_shift * 0.8
                            shift_y = rng_state.uniform(-0.3, 0.6) * max_shift * 0.5

                        # Forehead region
                        elif ddy < -0.7 and abs(ddx) < 1.0:
                            shift_x = rng_state.uniform(-1, 1) * max_shift * 0.6
                            shift_y = rng_state.uniform(-1, 0.5) * max_shift * 0.5

                        # Cheek/general face area
                        else:
                            shift_x = rng_state.uniform(-1, 1) * max_shift * 0.7
                            shift_y = rng_state.uniform(-1, 1) * max_shift * 0.5

                        shift_x *= strength
                        shift_y *= strength

                        dst_pts_list.append([
                            np.clip(px + shift_x, 0, roi_w - 1),
                            np.clip(py_pt + shift_y, 0, roi_h - 1),
                        ])
                    else:
                        dst_pts_list.append([px, py_pt])
            return np.float32(src_pts_list), np.float32(dst_pts_list)

        # --- Apply single-pass warping for identity change ---
        src_pts, dst_pts = _build_warp_points(rng, rw, rh, cx, cy, w, h)
        warped = self._piecewise_affine_warp(roi, src_pts, dst_pts, grid_rows, grid_cols)
        if warped is None:
            return None

        # --- Smooth out triangle seam artifacts with bilateral filter ---
        # Bilateral filter preserves edges (face features stay sharp) while
        # smoothing out the subtle seam lines from piecewise affine warping
        face_rx = max(0, x - ex)
        face_ry = max(0, y - ey)
        face_rw_clip = min(rw - face_rx, w)
        face_rh_clip = min(rh - face_ry, h)
        # Only smooth the face region, leave surroundings untouched
        face_sub = warped[face_ry:face_ry + face_rh_clip, face_rx:face_rx + face_rw_clip].copy()
        face_sub_smooth = cv2.bilateralFilter(face_sub, d=7, sigmaColor=40, sigmaSpace=40)
        # Blend smoothed into warped with a soft mask
        smooth_mask = np.zeros((rh, rw), dtype=np.float32)
        inner_m = max(3, min(face_rw_clip, face_rh_clip) // 8)
        smooth_mask[face_ry + inner_m:face_ry + face_rh_clip - inner_m,
                    face_rx + inner_m:face_rx + face_rw_clip - inner_m] = 1.0
        smooth_mask = cv2.GaussianBlur(smooth_mask, (15, 15), 0)
        smooth_3ch = cv2.merge([smooth_mask, smooth_mask, smooth_mask])
        # Reconstruct: smoothed face + original surrounding
        warped_smooth = warped.copy()
        warped_smooth[face_ry:face_ry + face_rh_clip, face_rx:face_rx + face_rw_clip] = face_sub_smooth
        warped = (warped_smooth.astype(np.float32) * smooth_3ch +
                  warped.astype(np.float32) * (1.0 - smooth_3ch)).clip(0, 255).astype(np.uint8)

        # --- Skin texture perturbation for anti-biometric ---
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, (0, 15, 40), (40, 255, 255))
        skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

        if np.sum(skin_mask > 0) > 100:
            noise = rng.normal(0, 4, warped.shape).astype(np.float32)
            skin_alpha = (skin_mask.astype(np.float32) / 255.0)
            skin_alpha_3ch = cv2.merge([skin_alpha, skin_alpha, skin_alpha])
            warped = (warped.astype(np.float32) + noise * skin_alpha_3ch * 0.7).clip(0, 255).astype(np.uint8)

        # --- Color shift on the face to further alter appearance ---
        face_mask_in_roi = np.zeros((rh, rw), dtype=np.float32)
        face_rx2 = max(0, x - ex)
        face_ry2 = max(0, y - ey)
        face_rw2 = min(rw - face_rx2, w)
        face_rh2 = min(rh - face_ry2, h)
        face_mask_in_roi[face_ry2:face_ry2 + face_rh2, face_rx2:face_rx2 + face_rw2] = 1.0
        face_mask_in_roi = cv2.GaussianBlur(face_mask_in_roi, (21, 21), 0)

        # Moderate hue/saturation shift
        hue_shift = rng.uniform(-8, 8)
        sat_scale = rng.uniform(0.8, 1.2)
        warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
        face_3ch = cv2.merge([face_mask_in_roi, face_mask_in_roi, face_mask_in_roi])
        shifted_hsv = warped_hsv.copy()
        shifted_hsv[:, :, 0] = (shifted_hsv[:, :, 0] + hue_shift) % 180
        shifted_hsv[:, :, 1] = shifted_hsv[:, :, 1] * sat_scale
        shifted_hsv = shifted_hsv.clip([0, 0, 0], [179, 255, 255])
        # Blend shifted version into the face region only
        blended_hsv = (shifted_hsv * face_3ch + warped_hsv * (1.0 - face_3ch))
        warped = cv2.cvtColor(blended_hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Paste warped region back into the full image
        result = original.copy()
        result[ey : ey + eh, ex : ex + ew] = warped

        return result

    @staticmethod
    def _piecewise_affine_warp(
        image: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        grid_rows: int,
        grid_cols: int,
    ) -> np.ndarray | None:
        """Apply piecewise affine warp using Delaunay triangulation.

        Each triangle in the source grid is mapped to its corresponding
        triangle in the destination grid via an affine transform.
        """
        h, w = image.shape[:2]
        output = image.copy()

        # Build triangles from the grid structure
        triangles = []
        for r in range(grid_rows - 1):
            for c in range(grid_cols - 1):
                idx_tl = r * grid_cols + c
                idx_tr = r * grid_cols + c + 1
                idx_bl = (r + 1) * grid_cols + c
                idx_br = (r + 1) * grid_cols + c + 1
                # Two triangles per grid cell
                triangles.append((idx_tl, idx_tr, idx_bl))
                triangles.append((idx_tr, idx_br, idx_bl))

        for tri_indices in triangles:
            i0, i1, i2 = tri_indices
            src_tri = np.float32([src_points[i0], src_points[i1], src_points[i2]])
            dst_tri = np.float32([dst_points[i0], dst_points[i1], dst_points[i2]])

            # Bounding rects
            sr = cv2.boundingRect(src_tri.reshape(1, -1, 2).astype(np.int32))
            dr = cv2.boundingRect(dst_tri.reshape(1, -1, 2).astype(np.int32))

            # Clip to image bounds
            sr_x2 = min(sr[0] + sr[2], w)
            sr_y2 = min(sr[1] + sr[3], h)
            sr = (max(0, sr[0]), max(0, sr[1]), sr_x2 - max(0, sr[0]), sr_y2 - max(0, sr[1]))

            dr_x2 = min(dr[0] + dr[2], w)
            dr_y2 = min(dr[1] + dr[3], h)
            dr = (max(0, dr[0]), max(0, dr[1]), dr_x2 - max(0, dr[0]), dr_y2 - max(0, dr[1]))

            if sr[2] <= 0 or sr[3] <= 0 or dr[2] <= 0 or dr[3] <= 0:
                continue

            # Offset triangle coords to bounding rect origins
            src_tri_offset = src_tri - np.float32([sr[0], sr[1]])
            dst_tri_offset = dst_tri - np.float32([dr[0], dr[1]])

            # Crop the source region
            src_crop = image[sr[1]:sr[1] + sr[3], sr[0]:sr[0] + sr[2]]
            if src_crop.size == 0:
                continue

            # Compute affine transform
            try:
                mat = cv2.getAffineTransform(src_tri_offset, dst_tri_offset)
            except cv2.error:
                continue

            warped_tri = cv2.warpAffine(
                src_crop, mat, (dr[2], dr[3]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
            )

            # Create mask for the destination triangle
            tri_mask = np.zeros((dr[3], dr[2]), dtype=np.uint8)
            pts = dst_tri_offset.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillConvexPoly(tri_mask, pts, 255)

            # Blend into output
            dst_crop = output[dr[1]:dr[1] + dr[3], dr[0]:dr[0] + dr[2]]
            if dst_crop.shape != warped_tri.shape:
                continue
            mask_3ch = cv2.merge([tri_mask, tri_mask, tri_mask]).astype(np.float32) / 255.0
            blended = (warped_tri.astype(np.float32) * mask_3ch +
                       dst_crop.astype(np.float32) * (1.0 - mask_3ch))
            output[dr[1]:dr[1] + dr[3], dr[0]:dr[0] + dr[2]] = blended.clip(0, 255).astype(np.uint8)

        return output

    # ------------------------------------------------------------------
    # 3. PII TEXT – erase original, then render synthetic at matching size
    # ------------------------------------------------------------------
    def _replace_pii_text(
        self, roi: np.ndarray, entity: DetectedEntity,
    ) -> Tuple[np.ndarray, str]:
        """Fully opaque text replacement that looks like a new card.

        Strategy:
        1. Detect the background color from ROI borders
        2. Detect the text color from the darkest/lightest pixels
        3. Inpaint to erase original text cleanly
        4. Render synthetic text at exact matching size
        5. NO alpha blending — fully opaque to prevent original showing through
        """
        h, w = roi.shape[:2]
        if h < 5 or w < 5:
            return roi, entity.text or ""

        # ---- Step 1: Detect background & text colors from the ROI ----
        border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
        bg_color = np.median(border, axis=0).astype(np.uint8).tolist()
        is_light_bg = np.mean(bg_color) > 127

        # Detect original text color from non-background pixels
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if is_light_bg:
            # Dark text on light bg — find the darkest cluster
            text_mask = gray < (np.mean(gray) - 20)
            text_color = (20, 20, 20)  # near-black for clean look
        else:
            text_mask = gray > (np.mean(gray) + 20)
            text_color = (240, 240, 240)

        # ---- Step 2: Erase original text — two-stage for guaranteed removal ----
        # Stage A: Otsu threshold to find text pixels, then inpaint
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        if is_light_bg:
            _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Dilate mask generously to ensure full coverage of text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=3)

        erased = cv2.inpaint(roi, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        # Stage B: Fill interior with solid background color to guarantee
        # no residual text, especially in multi-line merged bounding boxes.
        # Keep a thin border so the fill blends with surroundings.
        margin = max(2, min(h, w) // 15)
        fill_region = np.full_like(erased, bg_color, dtype=np.uint8)
        # Blend: keep original edges, solid fill in center
        blend_mask = np.zeros((h, w), dtype=np.float32)
        blend_mask[margin:h - margin, margin:w - margin] = 1.0
        # Feather the transition
        blend_mask = cv2.GaussianBlur(blend_mask, (margin * 2 + 1, margin * 2 + 1), 0)
        blend_3ch = cv2.merge([blend_mask, blend_mask, blend_mask])
        erased = (fill_region.astype(np.float32) * blend_3ch +
                  erased.astype(np.float32) * (1.0 - blend_3ch)).clip(0, 255).astype(np.uint8)

        # ---- Step 3: Generate synthetic text ----
        fake_text = self._generate_fake_text(entity)

        # ---- Step 4: Render text at size matching the ORIGINAL text ----
        # The key insight: we first compute the font scale that the ORIGINAL
        # text would need to fit the ROI, then apply that same scale to the
        # replacement text.  If the replacement is longer, we shrink to fit;
        # if shorter, we keep the same scale (never blow up).
        font = cv2.FONT_HERSHEY_DUPLEX  # cleaner, more document-like

        target_w = int(w * 0.90)
        target_h = int(h * 0.70)
        orig_text = entity.text or fake_text

        # Binary search: what scale would the ORIGINAL text render at?
        lo, hi = 0.1, 8.0
        orig_scale = 0.5
        for _ in range(25):
            mid = (lo + hi) / 2.0
            ts = cv2.getTextSize(orig_text, font, mid, max(1, int(mid * 1.0)))[0]
            if ts[0] <= target_w and ts[1] <= target_h:
                orig_scale = mid
                lo = mid
            else:
                hi = mid

        # Start with the original text's scale
        font_scale = orig_scale
        thickness = max(1, int(font_scale * 1.0))

        # Check if replacement text fits at this scale
        text_size = cv2.getTextSize(fake_text, font, font_scale, thickness)[0]
        if text_size[0] > target_w or text_size[1] > target_h:
            # Replacement is longer — shrink just enough to fit
            lo, hi = 0.1, font_scale
            for _ in range(25):
                mid = (lo + hi) / 2.0
                ts = cv2.getTextSize(fake_text, font, mid, max(1, int(mid * 1.0)))[0]
                if ts[0] <= target_w and ts[1] <= target_h:
                    font_scale = mid
                    lo = mid
                else:
                    hi = mid
            thickness = max(1, int(font_scale * 1.0))
            text_size = cv2.getTextSize(fake_text, font, font_scale, thickness)[0]

        # ---- Step 5: Render text with anti-aliased TrueType font via Pillow ----
        rendered_with_pil = False
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont
            target_pt = max(9, int(h * 0.72))
            font_candidates = [
                "arial.ttf", "segoeui.ttf", "calibri.ttf", "tahoma.ttf", "DejaVuSans.ttf",
                "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            pil_font = None
            for fc in font_candidates:
                try:
                    pil_font = PILImageFont.truetype(fc, target_pt)
                    break
                except Exception:
                    continue
            if pil_font is None:
                pil_font = PILImageFont.load_default()

            if pil_font:
                bbox_text = pil_font.getbbox(fake_text)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
                if tw > target_w and target_w > 0:
                    scale_factor = target_w / float(tw)
                    new_pt = max(8, int(target_pt * scale_factor))
                    for fc in font_candidates:
                        try:
                            pil_font = PILImageFont.truetype(fc, new_pt)
                            break
                        except Exception:
                            continue
                    bbox_text = pil_font.getbbox(fake_text)
                    tw = bbox_text[2] - bbox_text[0]
                    th = bbox_text[3] - bbox_text[1]

                tx = max(2, (w - tw) // 2)
                ty = max(0, (h - th) // 2 - bbox_text[1])

                pil_img = PILImage.fromarray(cv2.cvtColor(erased, cv2.COLOR_BGR2RGB))
                draw = PILImageDraw.Draw(pil_img)
                rgb_color = (int(text_color[2]), int(text_color[1]), int(text_color[0]))
                draw.text((tx, ty), fake_text, fill=rgb_color, font=pil_font)
                erased = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                rendered_with_pil = True
        except Exception as exc:
            LOGGER.debug("Pillow font rendering fallback: %s", exc)
            rendered_with_pil = False

        if not rendered_with_pil:
            text_y = (h + text_size[1]) // 2
            text_x = max(2, (w - text_size[0]) // 2)
            cv2.putText(erased, fake_text, (text_x, text_y), font,
                        font_scale, text_color, thickness, cv2.LINE_AA)

        # ---- Step 6: Add subtle noise to match document texture ----
        noise = np.random.normal(0, 1.5, erased.shape).astype(np.float32)
        result = cv2.add(erased.astype(np.float32), noise).clip(0, 255).astype(np.uint8)

        # FULLY OPAQUE — return the result directly, no alpha blending
        return result, fake_text

    # ------------------------------------------------------------------
    # Fake text generation
    # ------------------------------------------------------------------
    def _generate_fake_text(self, entity: DetectedEntity) -> str:
        """Generate contextually appropriate fake text for a PII entity of the exact same size.

        Gender-aware: uses entity.metadata['gender'] to pick male/female
        names so that male names are replaced with male names and vice versa.
        """
        label = entity.label.upper()
        orig = entity.text or ""
        L = len(orig)

        if L == 0:
            return f"<{label}>"

        # Helper to preserve character casing and numeric digits
        def format_preserve(text: str) -> str:
            chars = []
            for c in text:
                if c.isupper():
                    chars.append(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
                elif c.islower():
                    chars.append(np.random.choice(list("abcdefghijklmnopqrstuvwxyz")))
                elif c.isdigit():
                    chars.append(np.random.choice(list("0123456789")))
                else:
                    chars.append(c)
            return "".join(chars)

        if not self.faker:
            return format_preserve(orig)

        if label == "AADHAR":
            return f"{self.faker.random_number(4, True)} {self.faker.random_number(4, True)} {self.faker.random_number(4, True)}"[:L]

        if label in {"PERSON", "NAME"}:
            gender = entity.metadata.get("gender")
            for _ in range(50):
                if gender == "male":
                    candidate = self.faker.name_male()
                elif gender == "female":
                    candidate = self.faker.name_female()
                else:
                    candidate = self.faker.name()
                
                if orig.isupper():
                    candidate = candidate.upper()
                
                if len(candidate) == L:
                    return candidate
            
            # Fallback: truncate or pad
            candidate = self.faker.name()
            if orig.isupper():
                candidate = candidate.upper()
            if len(candidate) > L:
                return candidate[:L]
            else:
                needed = L - len(candidate)
                padding = "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for _ in range(needed))
                return candidate + (padding.upper() if orig.isupper() else padding.lower())

        if label == "ORG":
            for _ in range(50):
                candidate = self.faker.company()
                if orig.isupper():
                    candidate = candidate.upper()
                if len(candidate) == L:
                    return candidate
            if L <= 5:
                return "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for _ in range(L))
            
            candidate = self.faker.company()
            if orig.isupper():
                candidate = candidate.upper()
            if len(candidate) > L:
                return candidate[:L]
            else:
                return candidate.ljust(L, "A")

        if label == "DOB":
            return self.faker.date(pattern="%d/%m/%Y")[:L]

        if label in {"ADDRESS", "GPE", "STREET_ADDRESS", "LOCATION"}:
            # Alphanumeric codes or short strings detected as GPE should be format-preserved
            has_letters = any(c.isalpha() for c in orig)
            has_digits = any(c.isdigit() for c in orig)
            if (has_letters and has_digits) or L < 20:
                return format_preserve(orig)
            
            for _ in range(30):
                candidate = self.faker.address().replace("\n", ", ")
                if len(candidate) == L:
                    return candidate
            candidate = self.faker.address().replace("\n", ", ")
            if len(candidate) > L:
                return candidate[:L]
            else:
                return candidate.ljust(L, " ")

        if label == "LICENSE":
            return self.faker.bothify("??## #### #######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:L]

        if label == "ROLL_NUMBER":
            return format_preserve(orig)

        if label in {"NUMBERS", "CARDINAL"}:
            return "".join(str(np.random.randint(0, 10)) for _ in range(L))

        if label in {"EMAIL", "EMAIL_ADDRESS"}:
            for _ in range(30):
                candidate = self.faker.email()
                if len(candidate) == L:
                    return candidate
            return self.faker.email()[:L]

        if label in {"PHONE", "PHONE_NUMBER"}:
            for _ in range(30):
                candidate = self.faker.phone_number()
                if len(candidate) == L:
                    return candidate
            return self.faker.phone_number()[:L]

        return format_preserve(orig)

    # ------------------------------------------------------------------
    # Color matching helper
    # ------------------------------------------------------------------
    @staticmethod
    def _color_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match source's color distribution to target's (channel-wise)."""
        src_f = source.astype(np.float32)
        tgt_f = target.astype(np.float32)

        result_channels = []
        for i in range(3):
            m_tgt, s_tgt = cv2.meanStdDev(tgt_f[:, :, i])
            m_src, s_src = cv2.meanStdDev(src_f[:, :, i])

            scale = s_tgt[0][0] / (s_src[0][0] + 1e-6)
            normalized = (src_f[:, :, i] - m_src[0][0]) * scale + m_tgt[0][0]
            result_channels.append(normalized.clip(0, 255))

        return cv2.merge(result_channels).astype(np.uint8)

    # ------------------------------------------------------------------
    # Utility blur methods
    # ------------------------------------------------------------------
    def _blur_region(self, roi: np.ndarray) -> np.ndarray:
        kernel_size = max(51, self.config.face_blur_kernel * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

    def _aggressive_blur(self, roi: np.ndarray) -> np.ndarray:
        """Near-total destruction of details for high-risk PII."""
        smallest_dim = min(roi.shape[0], roi.shape[1])
        kernel = smallest_dim // 2
        if kernel % 2 == 0:
            kernel += 1
        kernel = max(15, min(kernel, 255))
        return cv2.medianBlur(roi, kernel)

    def _pixelate_region(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        grid_w, grid_h = max(2, w // 10), max(2, h // 10)
        small = cv2.resize(roi, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _redact_region(self, roi: np.ndarray) -> np.ndarray:
        return np.full_like(roi, 50)

    def _blur_region(self, roi: np.ndarray) -> np.ndarray:
        """Standard Gaussian blur for faces."""
        kernel = (self.config.face_blur_kernel, self.config.face_blur_kernel)
        return cv2.GaussianBlur(roi, kernel, 0)

    def _heavy_blur_region(self, roi: np.ndarray) -> np.ndarray:
        """Heavy blur for entire document/card regions — must be unreadable."""
        # Use a larger kernel proportional to the region size
        min_dim = min(roi.shape[0], roi.shape[1])
        kernel_size = max(self.config.face_blur_kernel, min_dim // 3)
        # Ensure kernel is odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(3, min(kernel_size, 251))  # Clamp to valid range

        # Apply multiple rounds of blur for thorough obscuration
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
        return blurred

    def _text_blur_region(self, roi: np.ndarray) -> np.ndarray:
        """Blur specifically sized for text regions — strong enough to make text unreadable."""
        min_dim = min(roi.shape[0], roi.shape[1])
        # Text needs a kernel relative to its height to become unreadable
        kernel_size = max(self.config.face_blur_kernel, int(min_dim * 0.8))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel_size = max(3, min(kernel_size, 251))

        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
        return blurred
