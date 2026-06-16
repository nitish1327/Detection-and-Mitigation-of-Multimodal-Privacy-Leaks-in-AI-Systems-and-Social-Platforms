"""Image privacy detection module."""

from __future__ import annotations

import re
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

# Regex patterns for ID-like strings (enrollment numbers, student IDs, etc.)
ID_NUMBER_RE = re.compile(
    r"[A-Z]{2,}\d{1,2}[A-Z]*\d{2,}",  # e.g. PES2UG23CS426
    re.IGNORECASE,
)

# Common institution/org keywords found on ID cards
INSTITUTION_KEYWORDS = {"university", "college", "institute", "school", "academy", "pes"}

# Geographic/map text to EXCLUDE from sensitive detection
GEOGRAPHIC_WORDS = {
    "ocean", "atlantic", "pacific", "indian", "arctic", "south", "north",
    "east", "west", "africa", "asia", "europe", "america",
    "sudan", "chad", "egypt", "algeria", "iran", "iraq", "mali", "niger",
    "congo", "kenya", "ethiopia", "somalia", "libya", "morocco", "tunisia",
    "mauritania", "senegal", "guinea", "cameroon", "nigeria", "angola",
    "mozambique", "madagascar", "tanzania", "zambia", "zimbabwe", "botswana",
    "namibia", "pakistan", "afghanistan", "india", "china", "russia",
    "brazil", "argentina", "chile", "peru", "colombia", "venezuela",
    "mexico", "canada", "australia", "japan", "korea", "thailand",
    "vietnam", "myanmar", "malaysia", "indonesia", "philippines",
    "saudi", "arabia", "yemen", "oman", "qatar", "kuwait", "bahrain",
    "turkey", "syria", "jordan", "israel", "palestine", "lebanon",
    "cape", "verde", "sierra", "leone", "burkina", "faso", "ivory", "coast",
    "republic", "democratic", "central", "island", "islands", "sea",
    "gulf", "bay", "strait", "channel", "lake", "river", "mountain",
    "desert", "sahara", "gobi", "arabian", "mediterranean", "caribbean",
    "tristan", "cunha", "gough", "sandwich", "georgia", "falkland",
    "maldives", "comoros", "reunion", "mauritius", "seychelles",
    "eritrea", "djibouti", "uganda", "rwanda", "burundi", "malawi",
    "lesotho", "swaziland", "gabon", "benin", "togo", "ghana", "liberia",
    "gambia", "guinea-bissau", "equatorial",
    # Common map labels
    "lat", "lon", "gmt", "utc", "equator", "tropic", "meridian",
    "formerly", "coordinated", "universal", "time", "french", "greenwich",
}


class ImageDetector:
    """Detect faces, sensitive text, and ID-card regions inside images."""

    def __init__(
        self,
        config: ImageConfig,
        text_labels: Sequence[str] | None = None,
        text_detector: "TextDetector | None" = None,
    ) -> None:
        self.config = config
        self.text_labels = tuple(text_labels or ("scene_text",))
        self._text_detector = text_detector

        # Frontal face cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

        # Profile face cascade for angled faces
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self._profile_cascade = cv2.CascadeClassifier(profile_path)

        self._ocr_reader = None

    def detect(self, path: Path) -> List[DetectedEntity]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        entities: list[DetectedEntity] = []
        entities.extend(self._detect_id_card_by_color(image))
        entities.extend(self._detect_faces(image))
        entities.extend(self._detect_text_multirotation(image))
        return entities

    # --- ID Card Detection via Color Segmentation -------------------------
    def _detect_id_card_by_color(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        """Detect light-colored rectangular cards against darker backgrounds."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # Detect light/white regions — broadened range to include off-white cards
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 100, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Clean up the mask with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_card_area = img_area * 0.02
        max_card_area = img_area * 0.50

        results: list[DetectedEntity] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_card_area or area > max_card_area:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (w_rect, h_rect), angle = rect
            if w_rect == 0 or h_rect == 0:
                continue

            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
            if aspect < 1.2 or aspect > 2.8:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = (area / hull_area) if hull_area > 0 else 0
            if solidity < 0.6:
                continue

            box = cv2.boxPoints(rect)
            box = np.int32(box)
            pad = 20
            x_min = max(0, int(min(box[:, 0])) - pad)
            y_min = max(0, int(min(box[:, 1])) - pad)
            x_max = min(img_w, int(max(box[:, 0])) + pad)
            y_max = min(img_h, int(max(box[:, 1])) + pad)
            w = x_max - x_min
            h = y_max - y_min

            if w > 30 and h > 30:
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="document_card",
                        confidence=0.88,
                        bbox=BoundingBox(x=x_min, y=y_min, width=w, height=h),
                        mitigation="blur",
                        detection_method="heuristic",
                    )
                )

        return results

    # --- Face Detection ---------------------------------------------------
    def _detect_faces(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        img_h, img_w = image.shape[:2]

        rotations = [
            (0, image),
            (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(image, cv2.ROTATE_180)),
            (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        seen_faces_global: list[BoundingBox] = []

        for angle, rot_img in rotations:
            gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)

            # Detect frontal faces
            frontal_faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
            )
            for (x, y, w, h) in frontal_faces:
                # Transform bbox back to original image coordinates
                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                bbox = self._transform_bbox_rotated(points, angle, img_h, img_w) if angle != 0 else self._bbox_from_points(points)

                # Check for overlap with already seen faces in original coordinates
                if any(self._bbox_overlap(bbox, seen) for seen in seen_faces_global):
                    continue
                seen_faces_global.append(bbox)
                yield DetectedEntity(
                    modality=Modality.IMAGE,
                    label="face",
                    confidence=0.9,
                    bbox=bbox,
                    mitigation="blur",
                    detection_method="heuristic",
                )

            # Detect profile faces
            profile_faces = self._profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20),
            )
            for (x, y, w, h) in profile_faces:
                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                bbox = self._transform_bbox_rotated(points, angle, img_h, img_w) if angle != 0 else self._bbox_from_points(points)

                if any(self._bbox_overlap(bbox, seen) for seen in seen_faces_global):
                    continue
                seen_faces_global.append(bbox)
                yield DetectedEntity(
                    modality=Modality.IMAGE,
                    label="face",
                    confidence=0.85,
                    bbox=bbox,
                    mitigation="blur",
                    detection_method="heuristic",
                )

    # --- Multi-Rotation OCR Text Detection --------------------------------
    def _detect_text_multirotation(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        """Run OCR at 0° and 90° to catch text at any orientation."""
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
                        "EasyOCR GPU init failed (%s); retrying CPU.", exc,
                    )
                    self._ocr_reader = EasyOcrReader(
                        list(self.config.text_detection_langs), gpu=False,
                    )
                else:
                    raise

        img_h, img_w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sensitive_entities: list[DetectedEntity] = []

        # Run OCR at 0°, 90°, 180°, and 270° (covers all orientations)
        rotations = [
            (0, rgb_image),
            (90, cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(rgb_image, cv2.ROTATE_180)),
            (270, cv2.rotate(rgb_image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        for angle, rotated_img in rotations:
            detections = self._ocr_reader.readtext(rotated_img)

            for bbox_points, text, confidence in detections:
                if confidence < self.config.min_confidence:
                    continue

                # Skip geographic/map text
                if self._is_geographic_text(text):
                    continue

                # Transform bbox back to original image coordinates
                if angle != 0:
                    bbox = self._transform_bbox_rotated(bbox_points, angle, img_h, img_w)
                else:
                    bbox = self._bbox_from_points(bbox_points)

                # Only keep text classified as sensitive
                is_sensitive, label = self._classify_text_sensitivity(text)
                if is_sensitive:
                    sensitive_entities.append(
                        DetectedEntity(
                            modality=Modality.IMAGE,
                            label=label,
                            confidence=float(confidence),
                            text=text,
                            bbox=bbox,
                            mitigation="blur",
                            detection_method="heuristic",
                        )
                    )

        return sensitive_entities

    @staticmethod
    def _is_geographic_text(text: str) -> bool:
        """Check if text is a geographic/map label that should be ignored."""
        if not text:
            return False
        cleaned = text.strip().lower()
        # Remove punctuation for matching
        cleaned_alpha = re.sub(r"[^a-z\s]", "", cleaned)
        words = cleaned_alpha.split()
        if not words:
            return False
        # If ALL words are geographic, skip it
        geo_count = sum(1 for w in words if w in GEOGRAPHIC_WORDS)
        return geo_count == len(words)

    def _transform_bbox_rotated(
        self, points: Sequence[Tuple[float, float]], angle: int, orig_h: int, orig_w: int,
    ) -> BoundingBox:
        """Transform bbox from rotated image coordinates back to original image coordinates."""
        if angle == 90:
            # 90 degrees Clockwise
            transformed = [(p[1], orig_h - 1 - p[0]) for p in points]
        elif angle == 180:
            # 180 degrees
            transformed = [(orig_w - 1 - p[0], orig_h - 1 - p[1]) for p in points]
        elif angle == 270:
            # 270 degrees Clockwise (90 CCW)
            transformed = [(orig_w - 1 - p[1], p[0]) for p in points]
        else:
            transformed = points
        return self._bbox_from_points(transformed)

    def _classify_text_sensitivity(self, text: str) -> tuple[bool, str]:
        """Determine if OCR-detected text is sensitive PII."""
        if not text or not text.strip():
            return False, "scene_text"

        cleaned = text.strip()

        # Check for ID number patterns (e.g. PES2UG23CS426)
        if ID_NUMBER_RE.search(cleaned):
            return True, "id_number"

        # Check for institution/org keywords
        lower = cleaned.lower()
        for keyword in INSTITUTION_KEYWORDS:
            if keyword in lower:
                return True, "institution_name"

        # Check if looks like a person's name (2+ capitalized words)
        words = cleaned.split()
        if 2 <= len(words) <= 4:
            alpha_words = [w for w in words if w.isalpha()]
            if len(alpha_words) >= 2:
                capitalized = sum(1 for w in alpha_words if w[0].isupper())
                if capitalized == len(alpha_words) and len(cleaned) >= 6:
                    # Extra check: not geographic
                    if not self._is_geographic_text(cleaned):
                        return True, "person_name"

        # Run through NER if available
        if self._text_detector and len(cleaned) >= 3:
            entities = self._text_detector.detect(cleaned)
            for ent in entities:
                if ent.label == "PERSON":
                    return True, "PERSON"
                if ent.label == "ORG" and not self._is_geographic_text(cleaned):
                    return True, "ORG"

        return False, "scene_text"

    @staticmethod
    def _bbox_from_points(points: Sequence[Tuple[float, float]]) -> BoundingBox:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return BoundingBox(
            x=max(0, int(x_min)),
            y=max(0, int(y_min)),
            width=max(1, int(x_max - x_min)),
            height=max(1, int(y_max - y_min)),
        )

    @staticmethod
    def _bbox_overlap(box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two bounding boxes overlap significantly."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        if x1 >= x2 or y1 >= y2:
            return False
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        return intersection / min(area1, area2) > 0.3
