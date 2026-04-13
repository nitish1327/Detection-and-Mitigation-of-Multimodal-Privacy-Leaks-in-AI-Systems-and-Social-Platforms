"""Image mitigation strategies – v2 overhaul.

Key improvements over v1:
  • QR:  size-adaptive multi-pass Gaussian blur (no colored blob)
  • Face: full opaque replacement via seamlessClone (no bleed-through)
  • PII:  erase-first, then render synthetic text at matching size/alignment
"""

from __future__ import annotations

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

        height, width = original.shape[:2]

        if self.config.only_mitigated_info:
            image = np.zeros((height, width, 3), dtype=np.uint8) + 30
        else:
            image = original.copy()

        if output_path is None:
            output_path = path.with_name(f"{path.stem}.sanitized{path.suffix}")

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
            "PERSON": 1, "NAME": 1,
            "license": 2,
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

            # Vertical adjacency: same X region, within one line-height gap
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
            is_text_pii = entity.label not in {"face", "qr_code"}
            pad_pct = 0.05 if is_text_pii else self.config.padding_percent
            padding = int(max(entity.bbox.width, entity.bbox.height) * pad_pct)
            x = max(0, int(entity.bbox.x) - padding)
            y = max(0, int(entity.bbox.y) - padding)
            w = min(width - x, int(entity.bbox.width) + 2 * padding)
            h = min(height - y, int(entity.bbox.height) + 2 * padding)

            roi_original = original[y : y + h, x : x + w]
            roi_current = image[y : y + h, x : x + w]
            if roi_original.size == 0:
                continue

            style = self.config.default_style
            if entity.label in {"qr_code", "aadhar", "mrz_zone"} and style != "synthetic":
                style = "aggressive_blur"

            mitigated_text = entity.text

            if style == "synthetic":
                if entity.label == "face":
                    # Skip face replacement for tiny faces (watermarks, ghost images)
                    if w < 60 or h < 60:
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
            elif style == "aggressive_blur":
                image[y : y + h, x : x + w] = self._blur_qr(roi_original) if entity.label == "qr_code" else self._aggressive_blur(roi_original)
                mitigated_text = "[AGGRESSIVE_BLUR]"
            elif style == "pixelate":
                image[y : y + h, x : x + w] = self._pixelate_region(roi_original)
                mitigated_text = "[PIXELATED]"
            elif style == "redact":
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

        cv2.imwrite(str(output_path), image)
        return output_path, updated_entities

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
    # 2. FACE – natural ID-photo replacement
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
        """Replace face with gender-matched asset that looks like a natural ID photo."""

        img_h, img_w = image.shape[:2]

        # Pick asset based on detected gender
        gender = entity.metadata.get("gender", "generic")
        if image_identity_asset is None:
            image_identity_asset = self._face_assets.get(gender)
            if image_identity_asset is None:
                image_identity_asset = self._face_assets.get("generic")

        asset = image_identity_asset
        if asset is None:
            # Fallback: heavy blur
            image[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (99, 99), 30)
            return image, "[BLURRED_FACE]", None

        # ------- Expand the replacement region to cover the full ID photo area -------
        # ID photos typically show head+shoulders, so expand down and slightly wider
        # than just the detected face bbox
        photo_x = max(0, x - int(w * 0.5))
        photo_y = max(0, y - int(h * 0.3))
        photo_w = min(img_w - photo_x, int(w * 2.0))
        photo_h = min(img_h - photo_y, int(h * 2.8))

        # Get the photo region from original for color reference
        photo_roi = original[photo_y : photo_y + photo_h, photo_x : photo_x + photo_w]
        if photo_roi.size == 0:
            photo_roi = roi
            photo_x, photo_y, photo_w, photo_h = x, y, w, h

        # ------- Resize full asset to fill the photo region -------
        # Use the full asset (head + shoulders) for a natural ID photo look
        ah, aw = asset.shape[:2]
        # Crop top portion of asset (head and upper shoulders, skip bottom)
        crop_top = int(ah * 0.02)
        crop_bottom = int(ah * 0.85)
        crop_left = int(aw * 0.05)
        crop_right = int(aw * 0.95)
        asset_crop = asset[crop_top:crop_bottom, crop_left:crop_right]

        # Resize to fill the photo region
        face_resized = cv2.resize(asset_crop, (photo_w, photo_h), interpolation=cv2.INTER_LANCZOS4)

        # ------- Skin tone adaptation via LAB color space -------
        # LAB gives perceptually uniform color matching (much better than BGR)
        face_lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        roi_lab = cv2.cvtColor(photo_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            m_roi, s_roi = cv2.meanStdDev(roi_lab[:, :, ch])
            m_face, s_face = cv2.meanStdDev(face_lab[:, :, ch])
            # Blend: 60% adapt to document colors, 40% keep original asset colors
            scale = (s_roi[0][0] / (s_face[0][0] + 1e-6))
            scale = 0.4 + 0.6 * min(scale, 2.0)  # clamp to prevent over-adaptation
            face_lab[:, :, ch] = (
                (face_lab[:, :, ch] - m_face[0][0]) * scale + 
                m_face[0][0] * 0.4 + m_roi[0][0] * 0.6
            )

        face_lab = face_lab.clip(0, 255).astype(np.uint8)
        face_matched = cv2.cvtColor(face_lab, cv2.COLOR_LAB2BGR)

        # ------- Match sharpness/blur level to document -------
        roi_gray = cv2.cvtColor(photo_roi, cv2.COLOR_BGR2GRAY)
        roi_sharpness = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        if roi_sharpness < 300:
            sigma = max(0.3, 2.0 * (1.0 - (roi_sharpness / 300.0)))
            ks = int(round(sigma * 3.0)) * 2 + 1
            face_matched = cv2.GaussianBlur(face_matched, (ks, ks), sigma)

        # ------- Create a full rectangular mask with feathered edges -------
        mask = np.zeros((photo_h, photo_w), dtype=np.uint8)
        margin = max(3, min(photo_w, photo_h) // 12)
        mask[margin:photo_h - margin, margin:photo_w - margin] = 255
        # Heavier feathering for natural edge blend
        feather_k = margin * 2 + 1
        if feather_k % 2 == 0:
            feather_k += 1
        mask = cv2.GaussianBlur(mask, (feather_k, feather_k), margin * 0.7)

        # ------- Add subtle JPEG compression noise to match document quality -------
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, enc = cv2.imencode('.jpg', face_matched, encode_param)
        face_matched = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if face_matched.shape[:2] != (photo_h, photo_w):
            face_matched = cv2.resize(face_matched, (photo_w, photo_h))

        # ------- Blend into the image -------
        if not self.config.only_mitigated_info:
            # Alpha blend with feathered mask for clean edges
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

        # ---- Step 5: Center text vertically, left-align horizontally ----
        text_y = (h + text_size[1]) // 2
        text_x = max(2, (w - text_size[0]) // 2)  # center horizontally

        # Draw text directly on the inpainted (erased) image
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
        """Generate contextually appropriate fake text for a PII entity.

        Gender-aware: uses entity.metadata['gender'] to pick male/female
        names so that male names are replaced with male names and vice versa.
        Addresses are replaced with realistic full addresses.
        """
        label = entity.label.upper()

        if not self.faker:
            return f"<{label}>"

        if label == "AADHAR":
            return f"{self.faker.random_number(4, True)} {self.faker.random_number(4, True)} {self.faker.random_number(4, True)}"
        if label in {"PERSON", "NAME"}:
            # Gender-aware name replacement
            gender = entity.metadata.get("gender")
            if gender == "male":
                return self.faker.name_male()
            elif gender == "female":
                return self.faker.name_female()
            return self.faker.name()
        if label == "ORG":
            return self.faker.company()
        if label == "DOB":
            return self.faker.date(pattern="%d/%m/%Y")
        if label in {"ADDRESS", "GPE", "STREET_ADDRESS", "LOCATION"}:
            # Replace address with a realistic full address
            return self.faker.address().replace("\n", ", ")
        if label == "LICENSE":
            return self.faker.bothify("??## #### #######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if label == "NUMBERS":
            orig_len = len(entity.text) if entity.text else 6
            return "".join(str(np.random.randint(0, 10)) for _ in range(orig_len))
        if label == "CARDINAL":
            return str(self.faker.random_int(min=100, max=9999))
        if label in {"EMAIL", "EMAIL_ADDRESS"}:
            return self.faker.email()
        if label in {"PHONE", "PHONE_NUMBER"}:
            return self.faker.phone_number()

        # Fallback: generate text matching original length
        orig = entity.text or label
        return self.faker.pystr(min_chars=len(orig), max_chars=len(orig) + 2)

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
