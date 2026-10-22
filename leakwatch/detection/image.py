"""Image privacy detection module."""

from __future__ import annotations

import re
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
from ..utils.logging import get_logger
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

        self._qr_detector = cv2.QRCodeDetector()
        self._barcode_detector = None
        if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
            try:
                self._barcode_detector = cv2.barcode.BarcodeDetector()
            except Exception:
                self._barcode_detector = None

        # OpenCV 5 YuNet face detector
        self._model_path = Path(__file__).resolve().parent.parent / "utils" / "assets" / "face_detection_yunet_2023mar.onnx"
        self._face_detector: cv2.FaceDetectorYN | None = None
        self._face_cascade = None
        self._profile_cascade = None

        if not self._model_path.exists():
            try:
                from ..video.tracker import ensure_yunet_model
                ensure_yunet_model(self._model_path)
            except Exception as exc:
                LOGGER.warning("Could not auto-download YuNet model: %s", exc)

        if self._model_path.exists():
            try:
                self._face_detector = cv2.FaceDetectorYN_create(
                    model=str(self._model_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.4,
                    nms_threshold=0.3,
                    top_k=5000,
                )
            except Exception as exc:
                LOGGER.warning("Failed to initialize FaceDetectorYN: %s", exc)

        self._ocr_reader = None
        # MRZ regex: usually 30-44 chars of uppercase letters, digits, and '<'
        self._mrz_pattern = re.compile(r"[A-Z0-9<]{30,44}")

    def detect(self, path: Path) -> List[DetectedEntity]:
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            return []
        return self.detect_frame(image)

    def detect_frame(self, image: np.ndarray) -> List[DetectedEntity]:
        if image is None or image.size == 0:
            return []

        entities: list[DetectedEntity] = []
        detected_gender = None
        
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
                
        entities.extend(self._detect_id_card_by_color(image))
        entities.extend(self._detect_text_multirotation(image))

        # Barcode Detection (1D and 2D barcodes)
        if getattr(self.config, "enable_barcode", True):
            entities.extend(self._detect_barcodes(image))

        # License Plate Detection
        if getattr(self.config, "enable_license_plate", True):
            ocr_res = ocr_results if self.config.enable_ocr else None
            entities.extend(self._detect_license_plates(image, ocr_results=ocr_res))

        # Signature & Stamp Detection
        if getattr(self.config, "enable_signature_stamp", True):
            ocr_res = ocr_results if self.config.enable_ocr else None
            entities.extend(self._detect_signatures_and_stamps(image, ocr_results=ocr_res))

        return entities

    # --- ID Card Detection via Color Segmentation -------------------------
    def _detect_id_card_by_color(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        """Detect light-colored rectangular cards against darker backgrounds."""
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
                        pass

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

    # --- Face Detection ---------------------------------------------------
    def _detect_faces(self, image: np.ndarray) -> Iterable[DetectedEntity]:
        if self._face_detector is None and self._model_path.exists():
            try:
                self._face_detector = cv2.FaceDetectorYN_create(
                    model=str(self._model_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.4,
                    nms_threshold=0.3,
                    top_k=5000,
                )
            except Exception as exc:
                LOGGER.warning("Failed to initialize FaceDetectorYN: %s", exc)
                return []

        if self._face_detector is None:
            return []

        img_h, img_w = image.shape[:2]

        rotations = [
            (0, image),
            (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(image, cv2.ROTATE_180)),
            (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        seen_faces_global: list[BoundingBox] = []

        for angle, rot_img in rotations:
            rh, rw = rot_img.shape[:2]
            self._face_detector.setInputSize((rw, rh))
            try:
                _, raw = self._face_detector.detect(rot_img)
            except Exception as exc:
                LOGGER.warning("YuNet face detection failed at angle %d: %s", angle, exc)
                continue

            if raw is None:
                continue

            for face in raw:
                fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                score = float(face[14]) if len(face) > 14 else 0.9

                # Clamp bounding box coordinates to rotated image dimensions
                x = max(0, fx)
                y = max(0, fy)
                w = min(rw - x, fw)
                h = min(rh - y, fh)
                if w <= 0 or h <= 0:
                    continue

                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                bbox = self._transform_bbox_rotated(points, angle, img_h, img_w) if angle != 0 else self._bbox_from_points(points)

                if any(self._bbox_overlap(bbox, seen) for seen in seen_faces_global):
                    continue
                seen_faces_global.append(bbox)
                face_mitigation = "replace" if self.config.default_style in {"synthetic", "replace"} else ("blur" if self.config.default_style == "blur" else "none")
                yield DetectedEntity(
                    modality=Modality.IMAGE,
                    label="face",
                    confidence=score,
                    bbox=bbox,
                    mitigation=face_mitigation,
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

    def _detect_barcodes(self, image: np.ndarray) -> list[DetectedEntity]:
        """Detect 1D barcodes and 2D barcodes/data matrices."""
        results: list[DetectedEntity] = []
        img_h, img_w = image.shape[:2]

        if hasattr(self, "_barcode_detector") and self._barcode_detector is not None:
            try:
                retval, decoded_info, decoded_type, points = self._barcode_detector.detectAndDecode(image)
                if retval and points is not None:
                    for i, poly in enumerate(points):
                        pts = poly.astype(np.int32)
                        x, y, w, h = cv2.boundingRect(pts)
                        if w > 10 and h > 10:
                            b_text = decoded_info[i] if i < len(decoded_info) else ""
                            b_type = decoded_type[i] if i < len(decoded_type) else "barcode"
                            results.append(
                                DetectedEntity(
                                    modality=Modality.IMAGE,
                                    label="BARCODE",
                                    text=b_text or b_type,
                                    confidence=0.95,
                                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                                    detection_method="cv",
                                    mitigation="mask",
                                )
                            )
            except Exception as exc:
                LOGGER.debug("OpenCV barcode detector failed: %s", exc)

        # Fallback 1D barcode gradient pattern detection
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)
            blurred = cv2.blur(gradient, (9, 9))
            _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            closed = cv2.erode(closed, None, iterations=4)
            closed = cv2.dilate(closed, None, iterations=4)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h) if h > 0 else 0
                area = w * h
                if 1.5 <= aspect <= 7.0 and (img_w * img_h * 0.005) <= area <= (img_w * img_h * 0.25):
                    if not any(abs(e.bbox.x - x) < 20 and abs(e.bbox.y - y) < 20 for e in results if e.bbox):
                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label="BARCODE",
                                text="BARCODE_1D",
                                confidence=0.80,
                                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                                detection_method="heuristic",
                                mitigation="mask",
                            )
                        )
        except Exception as exc:
            LOGGER.debug("1D barcode gradient detection error: %s", exc)

        return results

    def _detect_license_plates(self, image: np.ndarray, ocr_results: list | None = None) -> list[DetectedEntity]:
        """Detect vehicle license plates using regex on OCR and visual aspect ratio/edge profiling."""
        results: list[DetectedEntity] = []
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        if ocr_results:
            plate_re = re.compile(r"\b[A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{3,4}\b", re.IGNORECASE)
            for bbox_poly, text, conf in ocr_results:
                if plate_re.search(text):
                    pts = np.array(bbox_poly, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    results.append(
                        DetectedEntity(
                            modality=Modality.IMAGE,
                            label="LICENSE_PLATE",
                            text=text,
                            confidence=float(conf),
                            bbox=BoundingBox(x=x, y=y, width=w, height=h),
                            detection_method="pattern",
                            mitigation="mask",
                        )
                    )

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
            _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h) if h > 0 else 0
                area = w * h
                if 2.2 <= aspect <= 5.5 and (img_area * 0.003) <= area <= (img_area * 0.15):
                    roi_thresh = thresh[y:y+h, x:x+w]
                    white_ratio = cv2.countNonZero(roi_thresh) / float(area) if area > 0 else 0
                    if 0.15 <= white_ratio <= 0.65:
                        if not any(abs(e.bbox.x - x) < 20 and abs(e.bbox.y - y) < 20 for e in results if e.bbox):
                            results.append(
                                DetectedEntity(
                                    modality=Modality.IMAGE,
                                    label="LICENSE_PLATE",
                                    text="VEHICLE_PLATE",
                                    confidence=0.75,
                                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                                    detection_method="heuristic",
                                    mitigation="mask",
                                )
                            )
        except Exception as exc:
            LOGGER.debug("License plate visual contour error: %s", exc)

        return results

    def _detect_signatures_and_stamps(self, image: np.ndarray, ocr_results: list | None = None) -> list[DetectedEntity]:
        """Detect handwritten ink signatures and official stamps using HSV ink segmentation."""
        results: list[DetectedEntity] = []
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # OCR signature labels
        if ocr_results:
            sig_re = re.compile(r"\b(signature|signed|signatory|specimen\s+sign)\b", re.IGNORECASE)
            for bbox_poly, text, conf in ocr_results:
                if sig_re.search(text):
                    pts = np.array(bbox_poly, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    # Expand signature zone below or around label
                    sig_y = min(img_h - 1, y + h)
                    sig_h = min(img_h - sig_y, int(h * 2.5))
                    sig_w = min(img_w - x, int(w * 1.5))
                    if sig_h > 10 and sig_w > 20:
                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label="SIGNATURE",
                                text="SIGNATURE_FIELD",
                                confidence=float(conf),
                                bbox=BoundingBox(x=x, y=sig_y, width=sig_w, height=sig_h),
                                detection_method="pattern",
                                mitigation="mask",
                            )
                        )

        # Ink color segmentation (blue, violet, red ink)
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask_blue = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([135, 255, 240]))
            mask_purple = cv2.inRange(hsv, np.array([135, 50, 50]), np.array([165, 255, 240]))
            mask_red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 240]))
            mask_red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 240]))
            mask_ink = mask_blue | mask_purple | mask_red1 | mask_red2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_ink = cv2.morphologyEx(mask_ink, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if (img_area * 0.0005) <= area <= (img_area * 0.12) and w >= 15 and h >= 10:
                    # Require that ink appears on a light document/paper background (mean gray > 140)
                    roi_gray = gray[y:y+h, x:x+w]
                    if float(np.mean(roi_gray)) < 140.0:
                        continue
                    # Genuine ink signatures/stamps on paper are ink strokes, not solid saturated fabric (density < 70%)
                    roi_ink = mask_ink[y:y+h, x:x+w]
                    ink_density = float(cv2.countNonZero(roi_ink)) / float(area)
                    if ink_density > 0.70 or ink_density < 0.03:
                        continue

                    aspect = w / float(h) if h > 0 else 0
                    is_stamp = (0.7 <= aspect <= 1.5) and (cv2.contourArea(c) / float(area) > 0.35)
                    label = "STAMP" if is_stamp else "SIGNATURE"
                    if not any(abs(e.bbox.x - x) < 15 and abs(e.bbox.y - y) < 15 for e in results if e.bbox):
                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label=label,
                                text=label,
                                confidence=0.82,
                                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                                detection_method="heuristic",
                                mitigation="mask",
                            )
                        )
        except Exception as exc:
            LOGGER.debug("Ink stamp/signature detection error: %s", exc)

        return results
