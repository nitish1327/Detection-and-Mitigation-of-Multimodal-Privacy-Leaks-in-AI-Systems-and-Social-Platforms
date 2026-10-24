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
    """Apply natural synthetic replacement, perturbation, or blurring to sensitive regions."""

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
            "AADHAR": 0, "aadhar": 0,
            "id_number": 1, "LICENSE": 1, "license": 1, "roll_number": 1, "pan": 1, "PAN": 1,
            "dob": 2, "DOB": 2, "DATE": 2,
            "address": 3, "STREET_ADDRESS": 3,
            "PERSON": 4, "NAME": 4, "name": 4, "person_name": 4,
            "GPE": 5, "LOCATION": 5,
            "ORG": 6, "ORGANIZATION": 6,
            "numbers": 7, "CARDINAL": 7,
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

            # CRITICAL SAFETY: fundamentally different field types must NEVER merge!
            # An ID number, Aadhaar number, or DOB should NEVER absorb an Address or Name.
            def _field_type(lbl: str) -> str:
                u = lbl.upper()
                if u in {"AADHAR", "ID_NUMBER", "LICENSE", "PAN", "ROLL_NUMBER"}:
                    return "id"
                if u in {"DOB", "DATE"}:
                    return "dob"
                if u in {"ADDRESS", "STREET_ADDRESS", "GPE", "LOCATION"}:
                    return "address"
                if u in {"PERSON", "NAME", "PERSON_NAME"}:
                    return "name"
                return "other"

            fa, fb = _field_type(a.label), _field_type(b.label)
            if fa != fb:
                return False

            ax1, ay1 = a.bbox.x, a.bbox.y
            ax2, ay2 = ax1 + a.bbox.width, ay1 + a.bbox.height
            bx1, by1 = b.bbox.x, b.bbox.y
            bx2, by2 = bx1 + b.bbox.width, by1 + b.bbox.height

            # Check geometric intersection (strictly same line only)
            center_ya = ay1 + a.bbox.height / 2
            center_yb = by1 + b.bbox.height / 2
            avg_h = (a.bbox.height + b.bbox.height) / 2
            if abs(center_ya - center_yb) > avg_h * 0.65:
                # Different vertical lines: NEVER merge vertically
                return False

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = a.bbox.width * a.bbox.height
                area_b = b.bbox.width * b.bbox.height
                smaller = min(area_a, area_b)
                if smaller > 0 and (inter / smaller) > threshold:
                    return True

            # Vertical adjacency between separate lines is strictly disabled:
            # Merging separate lines vertically collapses multi-line blocks into
            # giant boxes and inflates synthetic text font sizes unnaturally.

            # Horizontal same-line adjacency: merge adjacent text words on the same line
            is_name_a = fa == "name"
            is_name_b = fb == "name"
            is_address_a = fa == "address"
            is_address_b = fb == "address"
            is_dob_or_id_a = a.label in {"dob", "DOB", "AADHAR", "aadhar", "id_number", "LICENSE", "license"}
            is_dob_or_id_b = b.label in {"dob", "DOB", "AADHAR", "aadhar", "id_number", "LICENSE", "license"}
            can_merge_horizontal = False
            if is_name_a and is_name_b:
                can_merge_horizontal = True
            elif is_address_a and is_address_b:
                can_merge_horizontal = True
            elif not is_dob_or_id_a and not is_dob_or_id_b:
                can_merge_horizontal = True

            if can_merge_horizontal:
                y_overlap = min(ay2, by2) - max(ay1, by1)
                min_h = min(a.bbox.height, b.bbox.height)
                if min_h > 0 and y_overlap / min_h > 0.35:
                    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
                    line_h = max(a.bbox.height, b.bbox.height)
                    if x_gap < line_h * 2.0:
                        return True

            return False

        # Separate face/qr/signature/barcode from text PII
        non_text_labels = {"face", "qr_code", "barcode", "signature", "stamp", "document_card", "license_plate"}
        non_text = [e for e in sorted_entities if e.label.lower() in non_text_labels]
        text_pii = [e for e in sorted_entities if e.label.lower() not in non_text_labels]

        # Group text PII by overlapping bboxes; keep highest priority per group
        # CRITICAL: merge all bboxes in the group into one unified bbox so
        # the ENTIRE region gets erased (no original text bleed-through).
        deduped_text: list[DetectedEntity] = []
        used: set[int] = set()
        for i, ent_i in enumerate(text_pii):
            if i in used:
                continue
            # Find all entities overlapping with ent_i (full transitive closure)
            group = [ent_i]
            used.add(i)
            while True:
                added = False
                for j, ent_j in enumerate(text_pii):
                    if j in used:
                        continue
                    if any(_bboxes_overlap(member, ent_j) for member in group):
                        group.append(ent_j)
                        used.add(j)
                        added = True
                if not added:
                    break
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
                    # Sort group members top-to-bottom, left-to-right so multi-line text reads naturally
                    sorted_group = sorted(
                        [e for e in group if e.bbox],
                        key=lambda e: (round(e.bbox.y / max(1, e.bbox.height * 0.5)), e.bbox.x)
                    )
                    words_in_order = []
                    for e in sorted_group:
                        if e.text:
                            for w in e.text.split():
                                if not words_in_order or w.lower() != words_in_order[-1].lower():
                                    words_in_order.append(w)
                    joined_text = " ".join(words_in_order)
                    best = best.model_copy(update={
                        "bbox": merged_bbox,
                        "text": joined_text if joined_text else best.text,
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

            style = entity.mitigation if entity.mitigation != "none" else self.config.default_style
            if entity.label in {"qr_code", "mrz_zone"}:
                style = "aggressive_blur"

            if entity.label.lower() in {"signature", "stamp"}:
                style = "blur"

            if entity.label == "face" and style in {"blur", "synthetic"}:
                style = "replace"

            # Skip tiny false-positive faces (emblems, logos, watermarks) for synthetic/replace
            if (
                entity.label == "face"
                and style in {"synthetic", "replace"}
                and entity.confidence < 0.6
                and (entity.bbox.width < 45 or entity.bbox.height < 45)
            ):
                continue

            # Compute padded region
            is_text_pii = entity.label.lower() not in non_text_labels
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

            if (self.config.default_style in {"synthetic", "replace"} or style in {"synthetic", "replace"}) and entity.label.lower() not in non_text_labels:
                style = "replace"

            mitigated_text = entity.text
            mapped_style = "synthetic" if (style in {"replace", "synthetic"} and entity.label.lower() not in non_text_labels) else style

            if mapped_style in {"synthetic", "replace"}:
                if entity.label == "face":
                    # Skip face replacement for tiny faces (watermarks, ghost images)
                    if w < 20 or h < 20:
                        image[y : y + h, x : x + w] = cv2.GaussianBlur(roi_original, (31, 31), 10)
                        mitigated_text = "[BLURRED_SMALL_FACE]"
                    else:
                        image, mitigated_text, image_identity_asset = self._handle_face(
                            image, original, roi_original, entity, x, y, w, h, image_identity_asset
                        )
                elif entity.label == "qr_code":
                    image[y : y + h, x : x + w] = self._blur_qr(roi_original)
                    mitigated_text = "[QR_BLURRED]"
                elif entity.label.lower() in {"signature", "stamp"}:
                    image[y : y + h, x : x + w] = self._blur_region(roi_original)
                    mitigated_text = "[SIGNATURE_BLURRED]"
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
        if w < 20 or h < 20:
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
        """Natural text replacement that cleanly erases original ink strokes and renders authentic synthetic text.

        Key features:
        1. Non-destructive inpainting: Inpaints ONLY the dilated ink mask using Telea inpainting,
           preserving 100% of underlying card texture, guilloche patterns, background gradients, and paper grain.
           Eliminates blocky flat rectangular patches completely.
        2. Accurate TrueType typography: Uses Arial Bold / Segoe UI Bold with Pillow for realistic print appearance.
        3. Precise cap-height calibration and alignment (centered for Aadhaar/badge names, left-aligned for fields).
        4. Camera MTF anti-aliasing blur to seamlessly blend rendered glyphs into photo noise.
        """
        h, w = roi.shape[:2]
        if h < 5 or w < 5:
            return roi, entity.text or ""

        # ---- Step 1: Robust background & text color detection ----
        # Sample border pixels (top, bottom, left, right rows)
        border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
        bg_color = np.percentile(border, 50, axis=0).astype(np.uint8)
        bg_mean = float(np.mean(bg_color))
        is_light_bg = bg_mean > 127

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Text color: sample pixels that deviate most from background
        if is_light_bg:
            dark_val = np.percentile(gray, 5)
            text_pixels = roi[gray <= dark_val + 15]
            if len(text_pixels) > 0:
                text_color_bgr = np.median(text_pixels, axis=0).astype(np.uint8)
            else:
                text_color_bgr = np.array([20, 20, 20], dtype=np.uint8)
            text_color_bgr = np.minimum(text_color_bgr, np.array([75, 75, 75], dtype=np.uint8))
        else:
            bright_val = np.percentile(gray, 95)
            text_pixels = roi[gray >= bright_val - 15]
            if len(text_pixels) > 0:
                text_color_bgr = np.median(text_pixels, axis=0).astype(np.uint8)
            else:
                text_color_bgr = np.array([235, 235, 235], dtype=np.uint8)
            text_color_bgr = np.maximum(text_color_bgr, np.array([180, 180, 180], dtype=np.uint8))

        # ---- Step 2: Build inpaint mask around text pixels ----
        if is_light_bg:
            _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate to cover ink strokes and anti-aliased edges
        dilation_iters = 2 if h >= 26 else 1
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inpaint_mask = cv2.dilate(inpaint_mask, k, iterations=dilation_iters)

        # ---- Step 3: Clean Inpainting (NO Pass B / NO flat patch) ----
        # Telea inpainting replaces only the ink stroke pixels, keeping all surrounding texture intact
        erased = cv2.inpaint(roi, inpaint_mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)

        # ---- Step 4: Generate synthetic replacement text ----
        fake_text = self._generate_fake_text(entity)

        # Multi-line wrapping for tall boxes (multi-line address blocks)
        num_lines = max(1, fake_text.count("\n") + 1)
        if h >= 28 and len(fake_text) > 18 and "\n" not in fake_text:
            import textwrap
            chars_per_line = max(16, len(fake_text) // max(2, round(h / 18)))
            wrapped = textwrap.wrap(fake_text, width=chars_per_line)
            if len(wrapped) > 1:
                fake_text = "\n".join(wrapped)
                num_lines = len(wrapped)

        # ---- Step 5: Size font and render with Pillow ----
        # In printed identity documents, body text (names, addresses, DOB, ID)
        # has a natural font size of 10-12pt regular.
        # Only the 12-digit Aadhaar number has larger bold digits (~18-20pt).
        line_h = h / num_lines
        is_large_aadhaar_number = (
            entity.label.upper() == "AADHAR"
            and bool(re.search(r"\d{4}\s\d{4}\s\d{4}", fake_text))
            and h >= 24
        )

        target_w_px = max(int(w * 0.95), w - 4)

        if is_large_aadhaar_number:
            target_pt = min(20, max(16, int(line_h * 0.58)))
            is_bold = True
        else:
            target_pt = min(11, max(9, int(line_h * 0.65)))
            is_bold = False

        rendered_with_pil = False
        try:
            from PIL import Image as PILImage, ImageDraw as PILImageDraw, ImageFont as PILImageFont

            bold_candidates = [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
            regular_candidates = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            font_candidates = bold_candidates if is_bold else regular_candidates

            def _load_pil_font(size: int):
                for fc in font_candidates:
                    try:
                        return PILImageFont.truetype(fc, size)
                    except Exception:
                        continue
                return PILImageFont.load_default()

            pil_img_dummy = PILImage.fromarray(cv2.cvtColor(erased, cv2.COLOR_BGR2RGB))
            dummy_draw = PILImageDraw.Draw(pil_img_dummy)

            pil_font = _load_pil_font(target_pt)
            if "\n" in fake_text:
                bb = dummy_draw.multiline_textbbox((0, 0), fake_text, font=pil_font, spacing=3)
            else:
                bb = pil_font.getbbox(fake_text)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]

            if tw > target_w_px and target_w_px > 20:
                scale = target_w_px / float(tw)
                target_pt = max(7, int(target_pt * scale))
                pil_font = _load_pil_font(target_pt)
                if "\n" in fake_text:
                    bb = dummy_draw.multiline_textbbox((0, 0), fake_text, font=pil_font, spacing=3)
                else:
                    bb = pil_font.getbbox(fake_text)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]

            if is_large_aadhaar_number:
                tx = max(2, (w - tw) // 2 - bb[0])
            else:
                tx = max(2, 2 - bb[0])

            if tx + bb[0] + tw > w:
                tx = max(0, w - tw - 2 - bb[0])

            ty = max(0, (h - th) // 2 - bb[1])

            pil_img = PILImage.fromarray(cv2.cvtColor(erased, cv2.COLOR_BGR2RGB))
            draw = PILImageDraw.Draw(pil_img)
            ink_color = np.clip(text_color_bgr, 30, 70)
            rgb_color = (int(ink_color[2]), int(ink_color[1]), int(ink_color[0]))
            if "\n" in fake_text:
                draw.multiline_text((tx, ty), fake_text, fill=rgb_color, font=pil_font, spacing=3)
            else:
                draw.text((tx, ty), fake_text, fill=rgb_color, font=pil_font)
            erased = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Subtle camera blur (anti-aliasing rendered glyphs into image texture)
            erased = cv2.GaussianBlur(erased, (3, 3), 0.25)
            rendered_with_pil = True
        except Exception:
            rendered_with_pil = False

        if not rendered_with_pil:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8 if is_large_aadhaar_number else 0.42
            thickness = 2 if is_large_aadhaar_number else 1
            text_size = cv2.getTextSize(fake_text, font, font_scale, thickness)[0]
            text_x = max(2, (w - text_size[0]) // 2) if is_large_aadhaar_number else max(2, 2)
            text_y = (h + text_size[1]) // 2
            ink_color = np.clip(text_color_bgr, 30, 70)
            text_color_tuple = (int(ink_color[0]), int(ink_color[1]), int(ink_color[2]))
            cv2.putText(erased, fake_text, (text_x, text_y), font,
                        font_scale, text_color_tuple, thickness, cv2.LINE_AA)

        return erased, fake_text

    # ------------------------------------------------------------------
    # Fake text generation
    # ------------------------------------------------------------------
    def _generate_fake_text(self, entity: DetectedEntity) -> str:
        """Generate contextually appropriate, natural, complete fake text for a PII entity.

        Ensures zero truncated words, zero 'AAAA' padding, and authentic locale formatting.
        """
        label = entity.label.upper()
        orig = (entity.text or "").strip()
        L = len(orig)

        if L == 0:
            return f"<{label}>"

        # Aadhaar: 12 digits (or 16-digit VID), grouped 4 4 4 if original had spaces
        if label == "AADHAR":
            d1 = np.random.randint(2000, 9999)
            d2 = np.random.randint(1000, 9999)
            d3 = np.random.randint(1000, 9999)
            digits_only = re.sub(r"\D", "", orig)
            if len(digits_only) == 16:
                d4 = np.random.randint(1000, 9999)
                prefix = "VID : " if "VID" in orig.upper() or "MID" in orig.upper() else ""
                return f"{prefix}{d1} {d2} {d3} {d4}"
            if " " in orig or L > 12:
                return f"{d1} {d2} {d3}"
            return f"{d1}{d2}{d3}"

        # Person name
        if label in {"PERSON", "NAME", "PERSON_NAME"}:
            gender = entity.metadata.get("gender")
            indian_male = [
                "Arjun R", "Rahul Kumar", "Vivek Patel", "Aditya Sharma",
                "Sanjay Verma", "Chinta Sai", "Rohit Verma", "Manish N", "Abhishek K",
                "Pranav G", "Nikhil Shetty", "Suresh Babu", "Vikram Joshi",
            ]
            indian_female = [
                "Priya Sharma", "Ananya Rao", "Pooja Patel", "Sneha Reddy", "Kavya Nair",
                "Meera Iyer", "Divya Menon", "Shreya Deshmukh", "Aishwarya G", "Deepika Joshi",
            ]
            western_male = [
                "Marcelo Silva", "David Miller", "Alexander Cross", "Robert Chen", "Lucas Moreira",
            ]
            western_female = [
                "Emma Watson", "Sarah Connor", "Laura Palmer", "Clara Oswald", "Julia Stiles",
            ]

            is_western = any(w in orig.upper() for w in ["RONALDO", "CRISTIANO", "JOHN", "MICHAEL", "SMITH", "SILVA"])
            candidates = (western_female if gender == "female" else western_male) if is_western else (indian_female if gender == "female" else indian_male)

            orig_word_count = len(orig.split())
            matching = [c for c in candidates if len(c.split()) == orig_word_count]
            if not matching:
                matching = sorted(candidates, key=lambda c: abs(len(c) - L))
            chosen = str(np.random.choice(matching)) if matching else candidates[0]

            # Preserve S/O, D/O, W/O, C/O prefix if present
            so_match = re.match(r"^(s/o|d/o|w/o|c/o|son of|daughter of|wife of)[:\s]*", orig, re.IGNORECASE)
            prefix = ""
            if so_match:
                prefix = so_match.group(0).rstrip() + " "

            if orig.isupper():
                return prefix + chosen.upper()
            return prefix + chosen

        # Date of Birth
        if label == "DOB":
            sep = "/" if "/" in orig else ("-" if "-" in orig else ("." if "." in orig else "/"))
            year = np.random.randint(1982, 2005)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            if re.match(r"^\d{4}", orig):
                return f"{year:04d}{sep}{month:02d}{sep}{day:02d}"
            return f"{day:02d}{sep}{month:02d}{sep}{year:04d}"

        # Driving license / Roll number / ID number / PAN card / Enrollment Number
        if label in {"LICENSE", "ID_NUMBER", "ROLL_NUMBER", "PAN"}:
            clean_orig = orig.replace(" ", "")

            # Aadhaar enrollment number (e.g. 2189/42082/02898 or Enrollment No 2189/42082/02898)
            enr_match = re.search(r"(\d{4})/(\d{5})/(\d{5})", orig)
            if enr_match or any(k in orig.lower() for k in ["enroll", "enrol"]):
                n1 = np.random.randint(2100, 2999)
                n2 = np.random.randint(10000, 99999)
                n3 = np.random.randint(10000, 99999)
                prefix = "Enrollment No " if any(k in orig.lower() for k in ["enroll", "enrol"]) else ""
                return f"{prefix}{n1}/{n2}/{n3}"

            # College Roll Number (e.g. PES2UG23CS189)
            pes_match = re.search(r"([A-Z]{3,4}\dUG\d{2}[A-Z]{2})(\d{3})", clean_orig)
            if pes_match:
                prefix = pes_match.group(1)
                num = np.random.randint(100, 599)
                return f"{prefix}{num:03d}"

            # PAN Card (5 letters + 4 digits + 1 letter, e.g. BKZPR4928K)
            if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", clean_orig):
                letters1 = "".join(np.random.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 5))
                digits = np.random.randint(1000, 9999)
                letter2 = np.random.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"))
                return f"{letters1}{digits}{letter2}"

            # Driving license (e.g. AP00820210048192 or KA01 20180004521)
            dl_match = re.search(r"^([A-Z]{2}\d{2,3})\s*(\d{4})\s*(\d{5,7})$", orig)
            if dl_match:
                rto, yr, ser = dl_match.groups()
                new_yr = np.random.randint(2015, 2024)
                new_ser = np.random.randint(10000, 99999)
                return f"{rto} {new_yr}{new_ser:07d}" if " " in orig else f"{rto}{new_yr}{new_ser:07d}"

            # General alphanumeric ID: preserve length and character classes
            chars = []
            for c in orig:
                if c.isupper():
                    chars.append(np.random.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ")))
                elif c.islower():
                    chars.append(np.random.choice(list("abcdefghijkmnopqrstuvwxyz")))
                elif c.isdigit():
                    chars.append(str(np.random.randint(0, 10)))
                else:
                    chars.append(c)
            return "".join(chars)

        # Organization
        if label == "ORG":
            companies = [
                "INFOSYS TECH", "TATA CONSULTANCY", "WIPRO ENTERPRISES", "RELIANCE DIGITAL",
                "BHARTI AIRTEL", "HDFC HOLDINGS", "ICICI SECURITIES", "MAHINDRA TECH",
            ]
            orig_word_count = len(orig.split())
            matching = [c for c in companies if len(c.split()) == orig_word_count]
            chosen = str(np.random.choice(matching)) if matching else companies[0]
            return chosen.upper() if orig.isupper() else chosen

        # Address / Location: line-by-line realistic replacement
        if label in {"ADDRESS", "GPE", "STREET_ADDRESS", "LOCATION"}:
            orig_lower = orig.lower()

            # 1. House / Flat / Door number (e.g. #19, 019, Flat 204)
            if re.search(r"^[#\d/]|(flat|door|plot|no\.)", orig_lower) and L <= 12:
                nums = ["#42", "#28", "#15", "Flat 104", "No. 36", "Plot 18"]
                return sorted(nums, key=lambda n: abs(len(n) - L))[0]

            # 2. State + PIN code line (e.g. Karnataka 560075)
            if re.search(r"\b\d{6}\b", orig):
                state_pins = [
                    "Karnataka 560038", "Karnataka 560102", "Karnataka 560003",
                    "Maharashtra 400050", "Telangana 500034", "Tamil Nadu 600028"
                ]
                chosen = sorted(state_pins, key=lambda p: abs(len(p) - L))[0]
                return chosen.upper() if orig.isupper() else chosen

            # 3. Street / Road line (e.g. 10th Main 13th Cross)
            if any(k in orig_lower for k in ["main", "cross", "road", "street", "rd", "st"]):
                streets = [
                    "4th Cross, 8th Main Road", "12th Main, 5th Cross",
                    "7th Cross, 2nd Stage", "3rd Main, 4th Block",
                    "MG Road, 6th Cross", "15th Cross Road"
                ]
                chosen = sorted(streets, key=lambda s: abs(len(s) - L))[0]
                return chosen.upper() if orig.isupper() else chosen

            # 4. Locality / Sub-district (e.g. Maruthi Nagar, Bangalore North)
            if any(k in orig_lower for k in ["nagar", "layout", "colony", "enclave", "north", "south", "east", "west"]):
                localities = [
                    "Indiranagar", "HSR Layout", "Koramangala", "Jayanagar",
                    "Bengaluru East", "Bengaluru South", "Whitefield", "BTM Layout"
                ]
                chosen = sorted(localities, key=lambda loc: abs(len(loc) - L))[0]
                return chosen.upper() if orig.isupper() else chosen

            # 5. Short city / place name (<= 15 chars)
            if L <= 15:
                cities = ["Bengaluru", "Hyderabad", "Mysuru", "Chennai", "Pune", "Mumbai", "Mangaluru"]
                chosen = sorted(cities, key=lambda c: abs(len(c) - L))[0]
                return chosen.upper() if orig.isupper() else chosen

            # 6. Full area / longer address line
            longer_areas = [
                "Old Airport Road, Kodihalli, Bengaluru",
                "Outer Ring Road, Bellandur, Bengaluru",
                "Intermediate Ring Road, Domlur, Bengaluru",
                "Bannerghatta Main Road, Bengaluru"
            ]
            chosen = sorted(longer_areas, key=lambda a: abs(len(a) - L))[0]
            return chosen.upper() if orig.isupper() else chosen

        # Phone number
        if label in {"PHONE", "PHONE_NUMBER"}:
            p1 = np.random.randint(7, 10)
            p2 = np.random.randint(100000000, 999999999)
            d = f"{p1}{p2}"
            if orig.startswith("+91"):
                return f"+91 {d[:5]} {d[5:]}"
            return d

        # Email
        if label in {"EMAIL", "EMAIL_ADDRESS"}:
            domains = ["gmail.com", "outlook.com", "yahoo.com"]
            dom = np.random.choice(domains)
            return f"user.{np.random.randint(100, 999)}@{dom}"

        # General numbers
        if label in {"NUMBERS", "CARDINAL"}:
            return "".join(str(np.random.randint(0, 10)) for _ in range(L))

        # Default fallback: preserve character format
        chars = []
        for c in orig:
            if c.isupper():
                chars.append(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
            elif c.islower():
                chars.append(np.random.choice(list("abcdefghijklmnopqrstuvwxyz")))
            elif c.isdigit():
                chars.append(str(np.random.randint(0, 10)))
            else:
                chars.append(c)
        return "".join(chars)


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
