"""Image privacy detection module."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

try:  # EasyOCR is optional; the code degrades gracefully without it.
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None

from ..utils.config import ImageConfig
from ..utils.types import BoundingBox, DetectedEntity, Modality

if TYPE_CHECKING:  # pragma: no cover
    from .text import TextDetector

if easyocr:
    EasyOcrReader = easyocr.Reader
else:  # pragma: no cover
    EasyOcrReader = None  # type: ignore


# Document field labels that are structural text, NOT PII values.
# These should never be mitigated — only the values beside them are sensitive.
_DOCUMENT_FIELD_LABELS = {
    # Common ID / driving license fields
    "name", "full name", "surname", "first name", "last name",
    "license no", "license no.", "licence no", "licence no.",
    "license number", "licence number", "dl no", "dl no.",
    "authorization to drive", "auth to drive",
    "date of issue", "doi", "issue date",
    "date of expiry", "doe", "expiry date", "valid till", "valid upto",
    "dob", "date of birth", "d.o.b", "d.o.b.",
    "s/w/d", "s/d/w", "s/o", "d/o", "w/o", "c/o",
    "blood group", "blood grp", "bg",
    "sex", "gender",
    "permanent address", "present address", "address",
    "place of issue", "issuing authority", "issued by",
    "nationality", "country",
    "signature", "photograph", "photo",
    "father's name", "fathers name", "husband's name", "husbands name",
    "mother's name", "mothers name",
    # Aadhaar specific
    "aadhaar", "aadhaar no", "aadhaar number", "uid", "vid",
    "government of india", "govt of india",
    "unique identification authority",
    # Passport fields
    "passport no", "passport number",
    "place of birth", "place of issue",
    # PAN
    "pan", "pan no", "pan number",
    "income tax department",
    # Generic document structure
    "sr no", "sr no.", "serial no", "serial number",
    "reg no", "reg no.", "registration no",
    "class of vehicle", "cov", "type of vehicle",
    "transport", "non-transport", "non transport",
}


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
        self._qr_detector = cv2.QRCodeDetector()
        self._ocr_reader = None
        # MRZ regex: usually 30-44 chars of uppercase letters, digits, and '<'
        self._mrz_pattern = re.compile(r"[A-Z0-9<]{30,44}")

    def detect(self, path: Path) -> List[DetectedEntity]:
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            return []

        entities: list[DetectedEntity] = []
        
        # QR Detection
        if self.config.enable_qr:
            entities.extend(self._detect_qr(image))

        # OCR + MRZ
        all_ocr_text = ""
        if self.config.enable_ocr:
            ocr_results = self._get_ocr_results(image)
            all_ocr_text = " ".join([r[1] for r in ocr_results])
            
            detected_gender = None
            ocr_lower = all_ocr_text.lower()
            # Explicit gender fields
            if "Female" in all_ocr_text or "FEMALE" in all_ocr_text or "Gender: F" in all_ocr_text or " sex: f" in ocr_lower:
                detected_gender = "female"
            elif "Male" in all_ocr_text or "MALE" in all_ocr_text or "Gender: M" in all_ocr_text or " sex: m" in ocr_lower:
                detected_gender = "male"
            # Indian document S/O, D/O, W/O patterns (Son/Daughter/Wife of)
            elif "s/o" in ocr_lower or "s/w/d" in ocr_lower or "siwid" in ocr_lower or "son of" in ocr_lower:
                detected_gender = "male"
            elif "d/o" in ocr_lower or "w/o" in ocr_lower or "daughter of" in ocr_lower or "wife of" in ocr_lower:
                detected_gender = "female"
                
            entities.extend(self._process_ocr_results(ocr_results, detected_gender))
            
            if self.config.enable_mrz:
                entities.extend(self._detect_mrz(ocr_results))
        
        # Face Detection (last, so we can use OCR context if available)
        face_entities = list(self._detect_faces(image))
        if detected_gender:
            for face in face_entities:
                face.metadata["gender"] = detected_gender
        entities.extend(face_entities)
                
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

    def _get_ocr_results(self, image: np.ndarray) -> List[Tuple[Sequence[Tuple[float, float]], str, float]]:
        """Run OCR and return raw detections (bbox, text, confidence)."""
        if not self.config.enable_ocr or EasyOcrReader is None:
            return []
        if self._ocr_reader is None:
            self._ocr_reader = EasyOcrReader(list(self.config.text_detection_langs), gpu=False)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._ocr_reader.readtext(rgb_image)

    @staticmethod
    def _is_field_label(text: str) -> bool:
        """Return True if the OCR text is a document field label, not a PII value."""
        cleaned = text.strip().lower()
        # Remove trailing colons/dashes that are common on labels
        cleaned = cleaned.rstrip(":- .")
        return cleaned in _DOCUMENT_FIELD_LABELS

    def _process_ocr_results(self, ocr_results: List[Tuple[Sequence[Tuple[float, float]], str, float]], detected_gender: str | None = None) -> List[DetectedEntity]:
        """Convert OCR results into DetectedEntity objects, routing via TextDetector.

        Field labels (e.g. "NAME", "License No.", "DOB") are skipped so that
        only the actual PII values beside them are mitigated.
        """
        results: list[DetectedEntity] = []
        for bbox_points, text, confidence in ocr_results:
            if confidence < self.config.min_confidence:
                continue

            # Skip structural document field labels — they are NOT PII
            if self._is_field_label(text):
                continue

            bbox = self._bbox_from_points(bbox_points)
            nested_entities = self._text_detector.detect(text) if self._text_detector else []
            if nested_entities:
                for nested in nested_entities:
                    # Also skip if the nested entity text itself is a label
                    if nested.text and self._is_field_label(nested.text):
                        continue

                    metadata = nested.metadata.copy()
                    if detected_gender and nested.label in {"PERSON", "NAME"}:
                        metadata["gender"] = detected_gender

                    results.append(
                        nested.model_copy(
                            update={
                                "modality": Modality.IMAGE,
                                "bbox": bbox,
                                "confidence": float(confidence),
                                "metadata": metadata,
                            }
                        )
                    )
            # Removed fallback for 'scene_text' to avoid blurring non-sensitive text
        return results

    def _detect_qr(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        """Detect and decode QR codes with aggressive multi-stage fallbacks."""
        detectors = [self._qr_detector]
        # Include WeChat detector if available
        if hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            try:
                detectors.append(cv2.wechat_qrcode_WeChatQRCode())
            except Exception:
                pass

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different image versions: Original, Sharpened, Thresholded
        versions = [gray]
        
        # Sharpened version
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        versions.append(cv2.filter2D(gray, -1, kernel))
        
        # Binary versions (Adaptive Thresholding)
        versions.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
        
        processed_points: set[tuple[int, int, int, int]] = set()

        for version in versions:
            for scale in [1.0, 1.5, 0.75, 2.0]:
                if scale == 1.0:
                    v = version
                else:
                    v = cv2.resize(version, (0, 0), fx=scale, fy=scale)
                
                for detector in detectors:
                    try:
                        if hasattr(detector, "detectAndDecodeMulti"):
                            ok, decoded_info, points, _ = detector.detectAndDecodeMulti(v)
                        else: # WeChat or single
                            decoded_info, points = detector.detectAndDecode(v)
                            ok = bool(decoded_info)
                            if ok and isinstance(decoded_info, str):
                                decoded_info = [decoded_info]
                                points = [points]
                    except Exception:
                        continue

                    if ok and points is not None:
                        for i, info in enumerate(decoded_info):
                            if not info and not (points is not None and i < len(points)):
                                continue
                            
                            p_raw = points[i] / scale
                            bbox = self._bbox_from_points(p_raw)
                            
                            # Expand QR bbox by 25% to ensure full coverage (including quiet zones)
                            expand_w = int(bbox.width * 0.25)
                            expand_h = int(bbox.height * 0.25)
                            bbox.x = max(0, bbox.x - expand_w // 2)
                            bbox.y = max(0, bbox.y - expand_h // 2)
                            bbox.width = min(image.shape[1] - bbox.x, bbox.width + expand_w)
                            bbox.height = min(image.shape[0] - bbox.y, bbox.height + expand_h)

                            # Deduplicate by bbox
                            bbox_key = (bbox.x, bbox.y, bbox.width, bbox.height)
                            if bbox_key in processed_points:
                                continue
                            processed_points.add(bbox_key)

                            yield DetectedEntity(
                                modality=Modality.IMAGE,
                                label="qr_code",
                                confidence=0.9 if info else 0.7,
                                text=info or "QR_CODE",
                                bbox=bbox,
                                mitigation="blur",
                            )
                        # If we found something at this scale, keep looking for others but maybe skip this version

    def _detect_mrz(self, ocr_results: List[Tuple[Sequence[Tuple[float, float]], str, float]]) -> Iterable[DetectedEntity]:
        """Detect MRZ zones in identity documents."""
        for bbox_points, text, confidence in ocr_results:
            clean_text = text.replace(" ", "").upper()
            if self._mrz_pattern.fullmatch(clean_text) or "<<" in clean_text:
                yield DetectedEntity(
                    modality=Modality.IMAGE,
                    label="mrz_zone",
                    confidence=float(confidence),
                    text=text,
                    bbox=self._bbox_from_points(bbox_points),
                    mitigation="blur",
                )
        

    def _detect_layout(
        self,
        image: np.ndarray,
        ocr_results: List[Tuple[Sequence[Tuple[float, float]], str, float]]
    ) -> Iterable[DetectedEntity]:
        """Detect layout blocks by grouping text regions."""
        if not ocr_results:
            return
        
        all_xs = []
        all_ys = []
        for bbox_points, _, _ in ocr_results:
            for p in bbox_points:
                all_xs.append(p[0])
                all_ys.append(p[1])
                
        if not all_xs or not all_ys:
            return
            
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        
        yield DetectedEntity(
            modality=Modality.IMAGE,
            label="layout_block",
            confidence=0.9,
            bbox=BoundingBox(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min),
                height=int(y_max - y_min),
            ),
            mitigation="blur",
        )


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
