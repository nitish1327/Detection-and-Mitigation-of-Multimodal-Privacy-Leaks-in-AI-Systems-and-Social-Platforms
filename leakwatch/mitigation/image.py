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
from ..utils.types import DetectedEntity


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
    # 2. FACE – full opaque replacement with seamlessClone
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
        """Replace face with gender-matched asset, fully opaque."""

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

        # ------- Crop center 70% of asset (face-only, no suit/background) -------
        ah, aw = asset.shape[:2]
        crop_x = int(aw * 0.15)
        crop_y = int(ah * 0.10)  # keep more forehead, less chin
        crop_w = int(aw * 0.70)
        crop_h = int(ah * 0.75)
        face_crop = asset[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        # Resize cropped face to match detected face bbox
        face_resized = cv2.resize(face_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # ------- Color-match asset to document lighting -------
        face_matched = self._color_match(face_resized, roi)

        # ------- Sharpness matching -------
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_sharpness = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        if roi_sharpness < 200:
            sigma = max(0.5, 2.5 * (1.0 - (roi_sharpness / 200.0)))
            ks = int(round(sigma * 3.0)) * 2 + 1
            face_matched = cv2.GaussianBlur(face_matched, (ks, ks), sigma)

        # ------- Create tight elliptical mask -------
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        # Ellipse covering ~85% of the face region
        axes = (int(w * 0.42), int(h * 0.45))
        cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        # Feather edges for natural blend
        mask = cv2.GaussianBlur(mask, (21, 21), 7)

        if not self.config.only_mitigated_info:
            # Use seamlessClone for perfectly natural edge blending
            mask_3ch = cv2.merge([mask, mask, mask])
            center = (x + w // 2, y + h // 2)
            try:
                image = cv2.seamlessClone(face_matched, image, mask_3ch, center, cv2.NORMAL_CLONE)
            except cv2.error:
                # Fallback: manual alpha blend
                alpha = mask.astype(np.float32) / 255.0
                alpha_3ch = cv2.merge([alpha, alpha, alpha])
                blended = (face_matched.astype(np.float32) * alpha_3ch +
                           roi.astype(np.float32) * (1.0 - alpha_3ch))
                image[y : y + h, x : x + w] = blended.clip(0, 255).astype(np.uint8)
        else:
            image[y : y + h, x : x + w] = face_matched

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

        # ---- Step 2: Erase original text via inpainting ----
        # Create a mask of where the text pixels are
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        if is_light_bg:
            _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Dilate mask to ensure full coverage of text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)

        # Inpaint the text away — this fills with surrounding background naturally
        erased = cv2.inpaint(roi, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        # ---- Step 3: Generate synthetic text ----
        fake_text = self._generate_fake_text(entity)

        # ---- Step 4: Render text at exact size match ----
        font = cv2.FONT_HERSHEY_DUPLEX  # cleaner, more document-like
        thickness = 1

        # Binary search for best font_scale to fill ~90% of ROI width
        target_w = int(w * 0.90)
        target_h = int(h * 0.70)

        lo, hi = 0.1, 8.0
        best_scale = 0.5
        for _ in range(25):
            mid = (lo + hi) / 2.0
            ts = cv2.getTextSize(fake_text, font, mid, max(1, int(mid * 1.0)))[0]
            if ts[0] <= target_w and ts[1] <= target_h:
                best_scale = mid
                lo = mid
            else:
                hi = mid

        font_scale = best_scale
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
        """Generate contextually appropriate fake text for a PII entity."""
        label = entity.label.upper()

        if not self.faker:
            return f"<{label}>"

        if label == "AADHAR":
            return f"{self.faker.random_number(4, True)} {self.faker.random_number(4, True)} {self.faker.random_number(4, True)}"
        if label in {"PERSON", "NAME"}:
            return self.faker.name()
        if label == "ORG":
            return self.faker.company()
        if label == "DOB":
            return self.faker.date(pattern="%d/%m/%Y")
        if label in {"ADDRESS", "GPE"}:
            return self.faker.city()
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
