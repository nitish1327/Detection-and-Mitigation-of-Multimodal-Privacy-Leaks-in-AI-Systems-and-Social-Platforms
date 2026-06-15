"""Image privacy detection module."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

try:  # EasyOCR is optional; the code degrades gracefully without it.
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

from ..utils.config import ImageConfig
from ..utils.logging import get_logger
from ..utils.types import BoundingBox, DetectedEntity, Modality

if TYPE_CHECKING:  # pragma: no cover
    from .text import TextDetector

if easyocr:
    EasyOcrReader = easyocr.Reader
else:  # pragma: no cover
    EasyOcrReader = None  # type: ignore


LOGGER = get_logger(__name__)


class ImageDetector:
    """Detect faces and sensitive text regions inside images."""

    def __init__(
        self,
        config: ImageConfig,
        text_labels: Sequence[str] | None = None,
        text_detector: "TextDetector | None" = None,
    ) -> None:
        self.config = config
        self.text_labels = tuple(text_labels or ("scene_text",))
        self._text_detector = text_detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        self._ocr_reader = None

    def detect(self, path: Path) -> List[DetectedEntity]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        entities: list[DetectedEntity] = []
        entities.extend(self._detect_faces(image))
        entities.extend(self._detect_text(image))
        return entities

    # --- Internal helpers --------------------------------------------
    def _detect_faces(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            yield DetectedEntity(
                modality=Modality.IMAGE,
                label="face",
                confidence=0.9,
                bbox=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                mitigation="blur",
                detection_method="heuristic",
            )

    def _detect_text(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        if not self.config.enable_ocr or EasyOcrReader is None:
            return []
        if self._ocr_reader is None:
            try:
                self._ocr_reader = EasyOcrReader(
                    list(self.config.text_detection_langs),
                    gpu=bool(self.config.ocr_use_gpu),
                )
            except Exception as exc:  # noqa: BLE001
                if self.config.ocr_use_gpu:
                    LOGGER.warning(
                        "EasyOCR GPU initialization failed (%s); retrying on CPU.",
                        exc,
                    )
                    self._ocr_reader = EasyOcrReader(list(self.config.text_detection_langs), gpu=False)
                else:
                    raise

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self._ocr_reader.readtext(rgb_image)
        results: list[DetectedEntity] = []
        for bbox_points, text, confidence in detections:
            if confidence < self.config.min_confidence:
                continue
            bbox = self._bbox_from_points(bbox_points)
            nested_entities = self._text_detector.detect(text) if self._text_detector else []
            if nested_entities:
                for nested in nested_entities:
                    results.append(
                        nested.model_copy(
                            update={
                                "modality": Modality.IMAGE,
                                "bbox": bbox,
                                "confidence": float(confidence),
                            }
                        )
                    )
            else:
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label=self.text_labels[0],
                        confidence=float(confidence),
                        text=text,
                        bbox=bbox,
                        mitigation="blur",
                        detection_method="heuristic",
                    )
                )
        return results

    @staticmethod
    def _bbox_from_points(points: Sequence[Tuple[float, float]]) -> BoundingBox:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return BoundingBox(
            x=int(x_min),
            y=int(y_min),
            width=int(x_max - x_min),
            height=int(y_max - y_min),
        )
