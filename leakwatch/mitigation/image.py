"""Image mitigation strategies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ..utils.config import ImageConfig
from ..utils.types import DetectedEntity


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

    def mitigate(
        self,
        path: Path,
        entities: List[DetectedEntity],
        output_path: Path | None,
    ) -> Tuple[Path, List[DetectedEntity]]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        if output_path is None:
            output_path = path.with_name(f"{path.stem}.sanitized{path.suffix}")

        updated_entities: list[DetectedEntity] = []
        for entity in entities:
            if not entity.bbox:
                continue
            x, y, w, h = (
                int(entity.bbox.x),
                int(entity.bbox.y),
                int(entity.bbox.width),
                int(entity.bbox.height),
            )

            # Clamp to image boundaries
            img_h, img_w = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            roi = image[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            # Determine blur strategy based on entity type
            if entity.label.lower() == "document_card":
                # Strong blur for the entire card region
                image[y : y + h, x : x + w] = self._heavy_blur_region(roi)
                action = "blur"
            elif entity.label.lower() == "face":
                # Standard face blur
                image[y : y + h, x : x + w] = self._blur_region(roi)
                action = "blur"
            elif entity.mitigation == "blur" or entity.label in BLUR_LABELS:
                # Blur for sensitive text regions (names, IDs, etc.)
                image[y : y + h, x : x + w] = self._text_blur_region(roi)
                action = "blur"
            else:
                # Fallback: still blur for safety in privacy context
                image[y : y + h, x : x + w] = self._blur_region(roi)
                action = "blur"

            updated_entities.append(entity.model_copy(update={"mitigation": action}))

        cv2.imwrite(str(output_path), image)
        return output_path, updated_entities

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
