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


# Document field labels, headers, and issuers that are structural text, NOT PII values.
# These should never be mitigated — only the values beside them are sensitive.
_DOCUMENT_FIELD_LABELS = {
    # Issuing authorities & document headers
    "republic of india", "republic of", "republic", "india", "bharat ganarajya", "bharat",
    "government of india", "govt of india", "govt. of india", "gov of india", "government",
    "goverment of india", "goverment",
    "unique identification authority of india", "unique identification authority", "uidai",
    "identification authority of india", "identification authority",
    "government of fictional territory", "fictional territory",
    "income tax department", "income tax dept", "dept of income tax", "department of income tax",
    "aaykar vibhag", "aaykar", "incometax",
    "election commission of india", "election commission",
    "transport department", "motor vehicles department",
    "indian union driving license", "union driving license",
    "driving license", "driver license", "driving licence", "driver licence",
    "permanent account number card", "permanent account number", "pan card",
    "elector photo identity card", "elactor photo identity card", "photo identity card", "identity card", "elector", "elector photo",
    "voter id", "voter identity card",
    "pes university", "pes institute of technology", "pes", "university",
    "bharat sarkar",
    
    # Common ID / driving license / passport card field labels & values
    "type", "type/type", "type / type", "code", "code/code", "code / code", "country code", "ind", "p",
    "nationality", "national", "indian", "hrat indian", "hrt turt", "hrat", "turt",
    "name", "full name", "surname", "first name", "last name", "given name", "given names",
    "license no", "license no.", "license no_", "licence no", "licence no.", "license number", "licence number", "dl no", "dl no.",
    "authorization to drive", "auth to drive", "to drive", "class of vehicle", "cov",
    "date of issue", "doi", "issue date",
    "date of expiry", "doe", "expiry date", "valid till", "valid upto", "validity",
    "dob", "date of birth", "d.o.b", "d.o.b.", "birth date",
    "s/w/d", "s/d/w", "s/o", "d/o", "w/o", "c/o", "son of", "daughter of", "wife of", "care of", "siwid",
    "father's name", "fathers name", "mother's name", "mothers name", "husband's name", "husbands name",
    "blood group", "blood grp", "bg",
    "sex", "gender", "male", "female", "transgender",
    "permanent address", "present address", "address", "res. address", "residential address",
    "place of issue", "issuing authority", "issued by",
    "place of birth", "country",
    "signature", "specimen signature", "holder's signature", "photograph", "photo", "sign",
    "enrolment no", "enrolment number", "enrollment no", "enrollment number",
    "enrollront no", "enrollront number", "enrollront",
    "aadhaar no", "aadhaar number", "your aadhaar no", "your aadhaar number", "your aadhaar no.", "aadhaar",
    "uid", "vid",
    "passport no", "passport number",
    "place of birth",
    "pan", "pan no", "pan number",
    "sr no", "sr no.", "serial no", "serial number",
    "reg no", "reg no.", "registration no",
    "roll no", "roll number", "prn", "usn",
    "transport", "non-transport", "non transport",
    "to", "to,", "to:", "ref", "ref:", "rel", "rel:",

    # Common Document Boilerplate / Instructions
    "this card is for identification purposes only",
    "it is not a proof of citizenship",
    "for identification purposes only",
    "not a proof of citizenship",
    "identification purposes only",
    "terms of use", "instructions",
    "tap to zoom", "to zoom",
    "signature valid", "digitally signed", "valid",
    "mera aadhaar meri pehchan", "nanna aadhaar nanna gurutu",
    "lmv,mcwg", "mcwg", "lmv", "nt", "tr", "unknown",
    # Single structural words commonly isolated by OCR in document headers/fields
    "government", "goverment", "fictional", "territory", "authority", "unique", "identification",
    "republic", "state", "department", "transport", "card", "permanent", "account",
    "driving", "license", "licence", "union", "expiry", "issue", "birth", "elector", "elactor",
    "election", "commission", "income", "tax", "pes", "university", "technology", "institute", "ministry",
    "citizenship", "citizen", "citizens", "proof", "purposes",
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
        
        # Face Detection (run early so photo location is available for ID card signature detection)
        raw_faces = list(self._detect_faces(image))

        # Ghost portrait / secondary photo detection (e.g. blue watermark silhouette on right side of passports)
        ghost_entities = list(self._detect_ghost_portraits(image, face_entities=raw_faces))

        # Filter out any face entity on the right half that overlaps with ghost portrait
        face_entities: list[DetectedEntity] = []
        for face in raw_faces:
            if face.bbox and face.bbox.x > image.shape[1] * 0.45:
                if any(self._bbox_overlap(face.bbox, g.bbox) for g in ghost_entities if g.bbox):
                    continue
            face_entities.append(face)

        # QR Detection
        if self.config.enable_qr:
            entities.extend(self._detect_qr(image))

        # Signature & Stamp Detection (using photo locations)
        sig_entities: list[DetectedEntity] = []
        if getattr(self.config, "enable_signature_stamp", True):
            sig_entities.extend(self._detect_signatures_and_stamps(image, ocr_results=None, face_entities=face_entities))

        # OCR + MRZ
        all_ocr_text = ""
        ocr_results = []
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

            # Check OCR signature labels
            if getattr(self.config, "enable_signature_stamp", True) and ocr_results:
                more_sigs = self._detect_signatures_and_stamps(image, ocr_results=ocr_results, face_entities=None)
                for s in more_sigs:
                    if not any(s.bbox and ex.bbox and abs(s.bbox.x - ex.bbox.x) < 20 and abs(s.bbox.y - ex.bbox.y) < 20 for ex in sig_entities):
                        sig_entities.append(s)

                # Remove false-positive face-bottom heuristic signatures if the region overlaps printed OCR text (e.g. Address)
                cleaned_sigs = []
                for s in sig_entities:
                    if s.detection_method == "heuristic" and s.bbox:
                        sb = s.bbox
                        overlaps_ocr = False
                        for r in ocr_results:
                            if not r[1] or not r[1].strip():
                                continue
                            r_poly = np.array(r[0], dtype=np.int32)
                            rx, ry, rw, rh = cv2.boundingRect(r_poly)
                            if max(sb.x, rx) < min(sb.x + sb.width, rx + rw) and max(sb.y, ry) < min(sb.y + sb.height, ry + rh):
                                overlaps_ocr = True
                                break
                        if overlaps_ocr:
                            continue
                    cleaned_sigs.append(s)
                sig_entities = cleaned_sigs

            sig_boxes = [s.bbox for s in sig_entities if s.bbox]
            mrz_names, mrz_passport_no = self._extract_mrz_metadata(ocr_results)
                
            entities.extend(self._process_ocr_results(
                ocr_results,
                detected_gender,
                image_shape=image.shape,
                sig_boxes=sig_boxes,
                mrz_names=mrz_names,
                mrz_passport_no=mrz_passport_no,
            ))
            
            if self.config.enable_mrz:
                entities.extend(self._detect_mrz(ocr_results, image_shape=image.shape))
        
        entities.extend(sig_entities)

        if detected_gender:
            for face in face_entities:
                face.metadata["gender"] = detected_gender
        entities.extend(face_entities)
        entities.extend(ghost_entities)
                
        # Only run id_card_by_color and multi-rotation OCR if normal OCR found little to no text
        has_ocr_text = bool(ocr_results and len(ocr_results) >= 2)
        if not has_ocr_text:
            entities.extend(self._detect_id_card_by_color(image))
            entities.extend(self._detect_text_multirotation(image))

        # Barcode Detection (1D and 2D barcodes)
        if getattr(self.config, "enable_barcode", True):
            ocr_res = ocr_results if self.config.enable_ocr else None
            entities.extend(self._detect_barcodes(image, ocr_results=ocr_res))

        # License Plate Detection
        if getattr(self.config, "enable_license_plate", True):
            ocr_res = ocr_results if self.config.enable_ocr else None
            entities.extend(self._detect_license_plates(image, ocr_results=ocr_res))

        return entities

    # --- Ghost Portrait Detection -----------------------------------------
    def _detect_ghost_portraits(
        self,
        image: np.ndarray,
        face_entities: list[DetectedEntity] | None = None,
    ) -> Iterable[DetectedEntity]:
        """Detect secondary ghost portraits / holographic watermarked silhouettes on ID cards/passports."""
        if image is None or image.size == 0:
            return
        img_h, img_w = image.shape[:2]
        if img_w < 100 or img_h < 100:
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Blue/cyan ghost watermark on right side of biometric documents
        blue_mask = cv2.inRange(hsv, np.array([90, 40, 50]), np.array([135, 255, 255]))

        # Look on the right half between 15% and 75% height
        y1, y2 = int(img_h * 0.15), int(img_h * 0.75)
        x1 = int(img_w * 0.55)
        right_mask = np.zeros_like(blue_mask)
        right_mask[y1:y2, x1:] = blue_mask[y1:y2, x1:]

        cnts, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > (img_w * img_h * 0.02):
                bx, by, bw, bh = cv2.boundingRect(c)
                # Refine to dense columns to trim sparse margin text/markings
                c_mask = np.zeros_like(blue_mask)
                cv2.drawContours(c_mask, [c], -1, 255, -1)
                col_counts = np.sum(c_mask > 0, axis=0)
                dense_cols = np.where(col_counts > (bh * 0.15))[0]
                if len(dense_cols) > 0:
                    bx = int(dense_cols[0])
                    bw = int(dense_cols[-1] - dense_cols[0])

                if 0.5 <= (bw / max(1, bh)) <= 1.6 and bw >= int(img_w * 0.12) and bh >= int(img_h * 0.20):
                    yield DetectedEntity(
                        modality=Modality.IMAGE,
                        label="ghost_portrait",
                        confidence=0.92,
                        bbox=BoundingBox(x=bx, y=by, width=bw, height=bh),
                        mitigation="blur",
                        detection_method="heuristic",
                    )

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

        img_h, img_w = image.shape[:2]
        # For low-resolution or thumbnail document scans, scale up adaptively so EasyOCR can detect small font characters
        scale = max(1.0, min(3.5, 800.0 / max(img_w, 1))) if (img_w < 500 or img_h < 350) else 1.0
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if scale > 1.0:
            resized = cv2.resize(rgb_image, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_CUBIC)
            raw_res = self._ocr_reader.readtext(resized)
            inv_scale = 1.0 / scale
            scaled_res = []
            for pts, txt, conf in raw_res:
                scaled_pts = [(p[0] * inv_scale, p[1] * inv_scale) for p in pts]
                scaled_res.append((scaled_pts, txt, conf))
            return scaled_res

        return self._ocr_reader.readtext(rgb_image)

    @staticmethod
    def _is_field_label(text: str) -> bool:
        """Return True if the OCR text is document structure, header, issuer, or template label."""
        if not text:
            return True
        cleaned = text.strip().lower()
        cleaned = cleaned.rstrip(":- .")
        if not cleaned:
            return True
        # If the text contains an actual ID number (like 2189/42082/02898) or Aadhaar number, it has sensitive value
        if re.search(r"\b\d{4}/\d{5}/\d{5}\b", cleaned) or re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", cleaned):
            return False
        # Document header, country code, and nationality protections
        if any(term in cleaned for term in ["republic of india", "government of india", "govt of india", "nationality", "indian", "rashtriyata", "bharat ganarajya"]):
            return True
        if cleaned in {"ind", "p", "type", "code", "ind.", "ind / ind", "type / type", "code / code", "hrt turt", "hrat", "turt", "type/type", "code/code"}:
            return True
        if re.search(r"\b(type|code|nationality|republic)\b", cleaned):
            return True
        if cleaned in _DOCUMENT_FIELD_LABELS:
            return True
        if any(cleaned.startswith(p) for p in ["signatur", "signature", "photograph", "photo", "sign.", "sign:", "sign "]) or bool(re.match(r"^sign\b", cleaned)):
            return True
        if cleaned in {"vid", "mid", "uid", "vid:", "mid:", "uid:"}:
            return True
        for term in _DOCUMENT_FIELD_LABELS:
            if " " in term and term in cleaned:
                return True
        words = cleaned.split()
        if len(words) <= 4 and all(w in _DOCUMENT_FIELD_LABELS for w in words):
            return True
        return False

    @staticmethod
    def _extract_mrz_metadata(ocr_results: list) -> tuple[set[str], str | None]:
        """Extract surname, given names, and passport number from MRZ lines."""
        mrz_names: set[str] = set()
        passport_no: str | None = None
        for _, text, _ in ocr_results:
            clean = text.replace(" ", "").upper()
            clean_norm = clean
            clean_norm = re.sub(r"^[P{][A-Z0-9<]{1,3}N[PD]", "P<IND", clean_norm)
            clean_norm = re.sub(r"^[P{][<K{(S]{0,2}", "P<", clean_norm)
            clean_norm = re.sub(r"[K{(<]{2,}", "<<", clean_norm)
            clean_norm = re.sub(r"[K{(]<", "<<", clean_norm)
            if "<" in clean_norm and ("P<" in clean_norm or "<<" in clean_norm or re.match(r"[A-Z0-9<]{25,44}", clean_norm)):
                m1 = re.search(r"P<[A-Z]{3}([A-Z]+)<<([A-Z<]+)", clean_norm)
                if m1:
                    surname = m1.group(1).replace("<", "").strip()
                    if surname and len(surname) >= 2:
                        mrz_names.add(surname)
                    given_names = [n.strip() for n in m1.group(2).split("<") if n.strip() and len(n.strip()) >= 2]
                    for gn in given_names:
                        mrz_names.add(gn)
                m2 = re.search(r"^([A-Z0-9]{7,9})<\d", clean_norm)
                if m2:
                    candidate = m2.group(1).strip()
                    if re.match(r"^[A-Z0-9]{7,9}$", candidate):
                        passport_no = candidate
        return mrz_names, passport_no

    def _process_ocr_results(
        self,
        ocr_results: List[Tuple[Sequence[Tuple[float, float]], str, float]],
        detected_gender: str | None = None,
        image_shape: tuple[int, ...] | None = None,
        sig_boxes: list[BoundingBox] | None = None,
        mrz_names: set[str] | None = None,
        mrz_passport_no: str | None = None,
    ) -> List[DetectedEntity]:
        """Convert OCR results into DetectedEntity objects, routing via TextDetector.

        Field labels and document headers are skipped so that only actual
        sensitive PII values beside them are mitigated.
        """
        results: list[DetectedEntity] = []
        valid_items: list[tuple[Sequence[Tuple[float, float]], str, float, BoundingBox]] = []
        img_h = image_shape[0] if image_shape else None

        # Filter out MRZ lines from text processing so they are not treated as id_number
        if ocr_results:
            filtered_ocr = []
            for pts, txt, conf in ocr_results:
                clean = txt.replace(" ", "").upper()
                ys = [p[1] for p in pts]
                is_mrz_line = (
                    "<" in clean and ("P<" in clean or "<<" in clean or re.match(r"[A-Z0-9<]{25,44}", clean))
                )
                if image_shape and min(ys) >= image_shape[0] * 0.70 and "<" in clean:
                    is_mrz_line = True
                if is_mrz_line:
                    continue
                filtered_ocr.append((pts, txt, conf))
            ocr_results = filtered_ocr

        # Horizontal same-line token grouping: join words on the same line that OCR split apart
        if ocr_results:
            raw_entries = []
            for pts, txt, conf in ocr_results:
                cleaned = txt.strip()
                if not cleaned:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                # If token falls inside an identified signature box, exclude it from text lines
                if sig_boxes and any(sb.x <= cx <= sb.x + sb.width and sb.y <= cy <= sb.y + sb.height for sb in sig_boxes):
                    continue
                raw_entries.append({
                    "pts": pts, "txt": cleaned, "conf": float(conf),
                    "x": min(xs), "y": min(ys),
                    "w": max(xs) - min(xs), "h": max(ys) - min(ys),
                })
            raw_entries.sort(key=lambda e: (e["y"], e["x"]))
            merged_ocr = []
            used_raw = set()
            for i, e1 in enumerate(raw_entries):
                if i in used_raw:
                    continue
                if e1["h"] > e1["w"] * 2.5:
                    merged_ocr.append((e1["pts"], e1["txt"], e1["conf"]))
                    used_raw.add(i)
                    continue

                line_tokens = [e1]
                used_raw.add(i)
                for j, e2 in enumerate(raw_entries):
                    if j in used_raw or e2["h"] > e2["w"] * 2.5:
                        continue
                    if max(e1["h"], e2["h"]) / max(1, min(e1["h"], e2["h"])) > 1.6:
                        continue
                    y_overlap = min(e1["y"] + e1["h"], e2["y"] + e2["h"]) - max(e1["y"], e2["y"])
                    min_h = min(e1["h"], e2["h"])
                    if min_h > 0 and (y_overlap / min_h) > 0.45:
                        x_gap = max(0, max(e1["x"], e2["x"]) - min(e1["x"] + e1["w"], e2["x"] + e2["w"]))
                        if x_gap <= max(e1["h"], e2["h"]) * 2.0:
                            line_tokens.append(e2)
                            used_raw.add(j)

                line_tokens.sort(key=lambda t: t["x"])
                merged_txt = " ".join(t["txt"] for t in line_tokens)
                min_x = min(t["x"] for t in line_tokens)
                max_x = max(t["x"] + t["w"] for t in line_tokens)
                min_y = min(t["y"] for t in line_tokens)
                max_y = max(t["y"] + t["h"] for t in line_tokens)
                avg_conf = sum(t["conf"] for t in line_tokens) / len(line_tokens)
                merged_pts = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
                merged_ocr.append((merged_pts, merged_txt, avg_conf))

            ocr_results = merged_ocr

        for bbox_points, text, confidence in ocr_results:
            cleaned = text.strip()
            if not cleaned or len(cleaned) < 2:
                continue
            if self._is_field_label(cleaned):
                continue

            # Check if this token matches passport number or is in the passport data column
            is_p_no = False
            clean_alnum = cleaned.replace(" ", "").strip()
            if (
                bool(re.fullmatch(r"[A-Z]{1,2}\d{6,8}", clean_alnum, re.IGNORECASE))
                or (mrz_passport_no and (clean_alnum.upper() == mrz_passport_no.upper() or (len(clean_alnum) >= 6 and clean_alnum[1:7] in mrz_passport_no)))
            ):
                is_p_no = True

            is_passport_col = False
            if image_shape:
                ih, iw = image_shape[:2]
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                bx, by = min(xs), min(ys)
                if (iw * 0.25 <= bx <= iw * 0.65) and (ih * 0.18 <= by <= ih * 0.75):
                    is_passport_col = True

            if confidence < self.config.min_confidence and not is_p_no and not is_passport_col:
                continue

            bbox = self._bbox_from_points(bbox_points)
            valid_items.append((bbox_points, cleaned, float(confidence), bbox))

        mit_style = "replace" if self.config.default_style in {"synthetic", "replace"} else ("blur" if self.config.default_style == "blur" else "mask")

        # Collect address and signature labels from all OCR results
        addr_labels = [
            self._bbox_from_points(pts)
            for pts, txt, conf in ocr_results
            if any(al in txt.lower().rstrip(":- .") for al in ["address", "permanent address", "present address", "res. address", "to,", "to"])
        ]
        sig_label_boxes = [
            self._bbox_from_points(pts)
            for pts, txt, conf in ocr_results
            if any(st in txt.lower() for st in ["signature", "हस्ताक्षर", "sign"])
        ]

        # Multi-token Aadhaar numbers: e.g. "9876", "5432", "1011"
        used_indices: set[int] = set()
        four_digit_items = [
            (idx, item) for idx, item in enumerate(valid_items)
            if re.fullmatch(r"\d{4}", item[1].strip())
        ]
        for i in range(len(four_digit_items)):
            if four_digit_items[i][0] in used_indices:
                continue
            idx1, item1 = four_digit_items[i]
            box1 = item1[3]
            group = [(idx1, item1)]
            for j in range(i + 1, len(four_digit_items)):
                idx2, item2 = four_digit_items[j]
                if idx2 in used_indices:
                    continue
                box2 = item2[3]
                y_diff = abs((box1.y + box1.height / 2) - (box2.y + box2.height / 2))
                x_gap = abs(box2.x - (box1.x + box1.width))
                if y_diff < max(box1.height, box2.height) * 1.5 and x_gap < max(box1.width, box2.width) * 3.0:
                    group.append((idx2, item2))
            if len(group) == 3:
                group.sort(key=lambda g: g[1][3].x)
                merged_text = f"{group[0][1][1]} {group[1][1][1]} {group[2][1][1]}"
                min_x = min(g[1][3].x for g in group)
                min_y = min(g[1][3].y for g in group)
                max_x2 = max(g[1][3].x + g[1][3].width for g in group)
                max_y2 = max(g[1][3].y + g[1][3].height for g in group)
                merged_bbox = BoundingBox(x=min_x, y=min_y, width=max_x2 - min_x, height=max_y2 - min_y)
                avg_conf = sum(g[1][2] for g in group) / 3.0
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="AADHAR",
                        text=merged_text,
                        confidence=avg_conf,
                        bbox=merged_bbox,
                        mitigation=mit_style,
                        detection_method="pattern",
                    )
                )
                for g_idx, _ in group:
                    used_indices.add(g_idx)

        # Process remaining items
        for idx, (bbox_points, text, confidence, bbox) in enumerate(valid_items):
            if idx in used_indices:
                continue

            clean_lower = text.lower().strip()

            # Filter out orphan single-digit noise or short fragmented numbers
            if len(text.strip()) <= 2 and not text.strip().isalpha():
                continue
            if text.strip().isdigit() and len(text.strip()) <= 3 and not re.search(r"\b(19|20)\d{2}\b", text):
                continue

            # Direct Aadhaar match: "9006 2534 7320"
            if re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text):
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="AADHAR",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="regex",
                    )
                )
                continue

            # Date of Birth match: "05/02/1985", "2004-07-15", "25/05/2004"
            if re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2})\b", text):
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="DOB",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="regex",
                    )
                )
                continue

            # 10-digit Phone number match (e.g. 9945145618)
            if re.search(r"\b[6-9]\d{9}\b", text):
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="PHONE",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="regex",
                    )
                )
                continue

            # Enrollment number match: "2189/42082/02898" or "Enrollment No ..."
            if re.search(r"\b\d{4}/\d{5}/\d{5}\b", text) or any(k in clean_lower for k in ["enroll", "enrol", "enrollront"]):
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="id_number",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="regex",
                    )
                )
                continue

            # ID / License / Roll / PAN / Passport number match
            clean_alnum = text.replace(" ", "").strip()
            is_passport_num = (
                bool(re.fullmatch(r"[A-Z]{1,2}\d{6,8}", clean_alnum, re.IGNORECASE))
                or bool(re.fullmatch(r"[A-PR-WYa-pr-wy][1-9]\d{6,7}", clean_alnum, re.IGNORECASE))
                or (mrz_passport_no and (clean_alnum.upper() == mrz_passport_no.upper() or (len(clean_alnum) >= 6 and clean_alnum[1:7] in mrz_passport_no)))
            )
            if ID_NUMBER_RE.search(text) or is_passport_num or re.search(r"\b[A-Z]{2}[0-9O]{12,16}\b", clean_alnum) or re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text):
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="id_number",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="regex",
                    )
                )
                continue

            # Passport column personal data mapping (Surname, Given Name, DOB, Place of Birth, Place of Issue, Date of Issue, Date of Expiry)
            if image_shape:
                ih, iw = image_shape[:2]
                if (iw * 0.25 <= bbox.x <= iw * 0.65) and (ih * 0.18 <= bbox.y <= ih * 0.75):
                    rel_y = bbox.y / float(ih)
                    lbl = None
                    if rel_y < 0.28:
                        lbl = "PERSON"  # Surname
                    elif rel_y < 0.35:
                        lbl = "PERSON"  # Given Names
                    elif rel_y < 0.43:
                        lbl = "DOB"     # Date of Birth
                    elif rel_y < 0.50:
                        lbl = "address" # Place of Birth
                    elif rel_y < 0.58:
                        lbl = "address" # Place of Issue
                    elif rel_y < 0.65:
                        lbl = "DOB"     # Date of Issue
                    else:
                        lbl = "DOB"     # Date of Expiry

                    if lbl:
                        metadata = {}
                        if detected_gender and lbl == "PERSON":
                            metadata["gender"] = detected_gender
                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label=lbl,
                                text=text,
                                confidence=confidence,
                                bbox=bbox,
                                mitigation=mit_style,
                                detection_method="pattern",
                                metadata=metadata,
                            )
                        )
                        continue

            # Person surname / name match from MRZ
            if mrz_names and text.strip().upper() in mrz_names:
                metadata = {}
                if detected_gender:
                    metadata["gender"] = detected_gender
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="PERSON",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="pattern",
                        metadata=metadata,
                    )
                )
                continue

            # Address match: contains address keywords or PIN code
            addr_keywords = {
                "street", "road", "nagar", "apartments", "apartment", "flat", "door", "plot",
                "cross", "layout", "district", "pradesh", "karnataka", "dubai", "emirates",
                "near", "sector", "phase", "main", "stage", "block", "colony", "enclave",
                "north", "south", "east", "west", "bengaluru", "bangalore", "mumbai", "delhi",
                "hyderabad", "chennai", "pune", "mysuru", "thippasandra", "taluk", "post", "opp", "behind"
            }
            if any(ak in clean_lower for ak in addr_keywords) or re.search(r"\b\d{6}\b", text):
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="address",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="pattern",
                    )
                )
                continue

            # Care of / Guardian / Relative prefix: S/O, D/O, W/O, C/O
            if re.search(r"\b(s/o|d/o|w/o|c/o|son of|daughter of|wife of)\b", clean_lower):
                metadata = {}
                if detected_gender:
                    metadata["gender"] = detected_gender
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label="person_name",
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="pattern",
                        metadata=metadata,
                    )
                )
                continue

            # Address field zone match: all lines directly below/beside an address or "To," label
            in_address_zone = False
            for alb in addr_labels:
                if alb.y - 10 <= bbox.y <= alb.y + 220 and bbox.x >= alb.x - 40:
                    if not re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text) and not re.fullmatch(r"\d{4}", text) and not re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text):
                        if not any(discl in text.lower() for discl in ["citizen", "proof", "terms", "instruction", "card is", "use", "only"]):
                            if not self._is_field_label(text):
                                in_address_zone = True
                                break
            if in_address_zone:
                # If first line after To, and looks like a person name, mark as person_name; otherwise address
                is_name = False
                for alb in addr_labels:
                    if 15 <= (bbox.y - alb.y) <= 60 and len(text.split()) <= 4:
                        is_name = True
                        break
                lbl = "person_name" if is_name else "address"
                metadata = {}
                if detected_gender and lbl == "person_name":
                    metadata["gender"] = detected_gender
                results.append(
                    DetectedEntity(
                        modality=Modality.IMAGE,
                        label=lbl,
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        mitigation=mit_style,
                        detection_method="pattern",
                        metadata=metadata,
                    )
                )
                continue

            # Next, route through TextDetector (normalizing common OCR letter confusions in names e.g. 5 -> S)
            text_norm = re.sub(r"\b5\b", "S", re.sub(r"\$+", "S", text))
            nested_entities = self._text_detector.detect(text_norm) if self._text_detector else []
            if nested_entities:
                for nested in nested_entities:
                    if nested.text and (self._is_field_label(nested.text) or any(k in nested.text.lower() for k in ["republic of india", "indian", "nationality"]) or nested.text.strip().upper() in {"IND", "P"}):
                        continue
                    metadata = nested.metadata.copy()
                    if detected_gender and nested.label in {"PERSON", "NAME"}:
                        metadata["gender"] = detected_gender

                    label = nested.label
                    if label in {"ORG", "ORGANIZATION"} and (text.isupper() or len(text.split()) <= 3):
                        if not any(k in clean_lower for k in INSTITUTION_KEYWORDS):
                            label = "PERSON"

                    # Top header shielding: text in the top 8% of an ID document is an institutional header banner
                    if img_h and bbox.y < img_h * 0.08 and label in {"PERSON", "NAME", "person_name"}:
                        continue

                    # Signature zone shielding: text right above or near a signature label is handwritten signature, not a printed person name
                    is_in_sig_zone = any(
                        abs((bbox.x + bbox.width / 2) - (sb.x + sb.width / 2)) < max(bbox.width, sb.width) * 1.5
                        and -90 <= (sb.y - (bbox.y + bbox.height)) <= 25
                        for sb in sig_label_boxes
                    )
                    if not is_in_sig_zone and img_h and image_shape:
                        img_w = image_shape[1]
                        if bbox.y > img_h * 0.70 and bbox.x > img_w * 0.40 and (bbox.x + bbox.width) < img_w * 0.75:
                            if not re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text):
                                is_in_sig_zone = True

                    if is_in_sig_zone and label in {"PERSON", "NAME", "person_name"}:
                        continue

                    # Low confidence filter for NLP entity routing to prevent background leaflet noise
                    if confidence < 0.45 and label in {"PERSON", "NAME", "person_name", "ORG"}:
                        continue

                    # Skip standalone short numbers (1-4 digits) that are not part of an ID, PIN, or date
                    if label in {"CARDINAL", "numbers"} and len(nested.text.strip()) <= 4:
                        continue

                    results.append(
                        nested.model_copy(
                            update={
                                "modality": Modality.IMAGE,
                                "label": label,
                                "bbox": bbox,
                                "confidence": float(confidence),
                                "metadata": metadata,
                                "mitigation": mit_style,
                            }
                        )
                    )
            elif 2 <= len(text.split()) <= 5:
                # Top header shielding: text in the top 8% of an ID document is an institutional header banner
                if img_h and bbox.y < img_h * 0.08:
                    continue
                # Signature zone shielding: text right above a signature label is handwritten signature, not a printed name
                is_in_sig_zone = any(
                    abs((bbox.x + bbox.width / 2) - (sb.x + sb.width / 2)) < max(bbox.width, sb.width) * 1.5
                    and -90 <= (sb.y - (bbox.y + bbox.height)) <= 25
                    for sb in sig_label_boxes
                )
                if not is_in_sig_zone and img_h and image_shape:
                    img_w = image_shape[1]
                    if bbox.y > img_h * 0.70 and bbox.x > img_w * 0.40 and (bbox.x + bbox.width) < img_w * 0.75:
                        if not re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text):
                            is_in_sig_zone = True

                if is_in_sig_zone:
                    continue
                # Low confidence filter for pattern person names
                if confidence < 0.45:
                    continue
                # Person name pattern: 2-5 words, capitalized, not geographic, not document structure
                words = text.split()
                alpha_words = [w for w in words if w.isalpha()]
                if len(alpha_words) >= 2 and all(w[0].isupper() for w in alpha_words):
                    if not self._is_geographic_text(text) and not self._is_field_label(text) and not any(k in text.lower() for k in ["republic of india", "indian", "nationality"]):
                        metadata = {}
                        if detected_gender:
                            metadata["gender"] = detected_gender
                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label="person_name",
                                text=text,
                                confidence=confidence,
                                bbox=bbox,
                                mitigation=mit_style,
                                detection_method="pattern",
                                metadata=metadata,
                            )
                        )
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

        # Fallback: High-density 2D matrix / secure Aadhaar QR code detection via texture & contour analysis
        try:
            img_h, img_w = gray.shape[:2]
            img_area = img_h * img_w
            val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = w / float(h) if h > 0 else 0
                area = w * h
                # QR codes are square (0.85-1.15), area between 0.8% and 20% of image, min dimension 40px
                if 0.85 <= aspect <= 1.15 and (img_area * 0.008) <= area <= (img_area * 0.20) and w >= 40:
                    roi = gray[y : y + h, x : x + w]
                    lap = cv2.Laplacian(roi, cv2.CV_64F).var()
                    # High-density 2D matrix / QR code has massive Laplacian variance (> 3500)
                    if lap > 3500:
                        if any(abs(p[0] - x) < 30 and abs(p[1] - y) < 30 for p in processed_points):
                            continue
                        expand_w = int(w * 0.15)
                        expand_h = int(h * 0.15)
                        bx = max(0, x - expand_w // 2)
                        by = max(0, y - expand_h // 2)
                        bw = min(image.shape[1] - bx, w + expand_w)
                        bh = min(image.shape[0] - by, h + expand_h)
                        processed_points.add((bx, by, bw, bh))

                        yield DetectedEntity(
                            modality=Modality.IMAGE,
                            label="qr_code",
                            confidence=0.88,
                            text="QR_CODE",
                            bbox=BoundingBox(x=bx, y=by, width=bw, height=bh),
                            mitigation="blur",
                        )
        except Exception as exc:
            LOGGER.debug("Visual QR detection fallback error: %s", exc)

    def _detect_mrz(
        self,
        ocr_results: List[Tuple[Sequence[Tuple[float, float]], str, float]],
        image_shape: tuple[int, ...] | None = None,
    ) -> Iterable[DetectedEntity]:
        """Detect MRZ zones in identity documents and yield unified erase region."""
        mrz_boxes = []
        mrz_texts = []
        for bbox_points, text, confidence in ocr_results:
            clean_text = text.replace(" ", "").upper()
            is_mrz = False
            if (self._mrz_pattern.fullmatch(clean_text) and "<" in clean_text) or "<<" in clean_text:
                is_mrz = True
            elif "<" in clean_text and (len(clean_text) >= 25 or re.search(r"<[A-Z0-9<]{10,}", clean_text)):
                if image_shape:
                    img_h = image_shape[0]
                    ys = [p[1] for p in bbox_points]
                    if min(ys) >= img_h * 0.70:
                        is_mrz = True
                else:
                    is_mrz = True
            if is_mrz:
                mrz_boxes.append(self._bbox_from_points(bbox_points))
                mrz_texts.append(text)

        if not mrz_boxes:
            return

        min_x = 0
        min_y = max(0, min(b.y for b in mrz_boxes) - 5)
        if image_shape:
            img_h, img_w = image_shape[:2]
            max_y2 = img_h
            max_x2 = img_w
        else:
            max_x2 = max(b.x + b.width for b in mrz_boxes) + 5
            max_y2 = max(b.y + b.height for b in mrz_boxes) + 10

        yield DetectedEntity(
            modality=Modality.IMAGE,
            label="mrz_zone",
            confidence=0.95,
            text="\n".join(mrz_texts),
            bbox=BoundingBox(
                x=int(min_x),
                y=int(min_y),
                width=int(max_x2 - min_x),
                height=int(max_y2 - min_y),
            ),
            mitigation="mask",
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
        if self._is_field_label(cleaned):
            return False, "document_structure"

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

    def _detect_barcodes(self, image: np.ndarray, ocr_results: list | None = None) -> list[DetectedEntity]:
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

        # Skip visual Sobel heuristic if OCR text is already detected on the document
        if ocr_results and len(ocr_results) >= 2:
            return results

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
            return results

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

    def _detect_signatures_and_stamps(
        self,
        image: np.ndarray,
        ocr_results: list | None = None,
        face_entities: list | None = None,
    ) -> list[DetectedEntity]:
        """Detect handwritten ink signatures and official stamps using HSV and spatial heuristic segmentation."""
        results: list[DetectedEntity] = []
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # 1. Signature directly below photo IDs (e.g. passport or identity card holder signature)
        if face_entities:
            for face in face_entities:
                if not face.bbox or face.confidence < 0.5:
                    continue
                fb = face.bbox
                if fb.x > img_w * 0.40:
                    continue
                sig_y = int(fb.y + fb.height)
                sig_max_y = int(img_h * 0.77)
                sig_h = min(sig_max_y - sig_y, int(fb.height * 1.5))
                sig_x = max(0, int(fb.x - 30))
                sig_w = min(img_w - sig_x, int(fb.width + 60))
                # For passports and left-aligned photo IDs, keep signature strictly within left photo column (x <= 195)
                if fb.x + fb.width < img_w * 0.4:
                    sig_w = min(sig_w, max(20, int(img_w * 0.32) - sig_x))
                if sig_h > 20 and sig_w > 30:
                    roi = image[sig_y : sig_y + sig_h, sig_x : sig_x + sig_w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    bg_val = float(np.percentile(roi_gray, 75))
                    dark_thresh = min(150, bg_val - 25)
                    stroke_mask = roi_gray < dark_thresh
                    stroke_count = int(np.count_nonzero(stroke_mask))
                    area = sig_w * sig_h
                    if stroke_count > 150 and 0.01 <= (stroke_count / area) <= 0.40:
                        stroke_pts = np.argwhere(stroke_mask)
                        sy1 = int(stroke_pts[:, 0].min() + sig_y)
                        sy2 = int(stroke_pts[:, 0].max() + sig_y)
                        sx1 = int(stroke_pts[:, 1].min() + sig_x)
                        sx2 = int(stroke_pts[:, 1].max() + sig_x)
                        pad = 6
                        fx = max(0, sx1 - pad)
                        fy = max(0, sy1 - pad)
                        fw = min(img_w - fx, (sx2 - sx1) + 2 * pad)
                        fh = min(img_h - fy, (sy2 - sy1) + 2 * pad)
                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label="SIGNATURE",
                                text="SIGNATURE",
                                confidence=0.90,
                                bbox=BoundingBox(x=fx, y=fy, width=fw, height=fh),
                                detection_method="heuristic",
                                mitigation="blur",
                            )
                        )

        # 2. OCR signature labels
        if ocr_results:
            sig_re = re.compile(
                r"\b(signature|signed|signatory|specimen\s+sign|हस्ताक्षर)\b|(?:\bsign\b[.:\s])",
                re.IGNORECASE,
            )
            for bbox_poly, text, conf in ocr_results:
                if sig_re.search(text):
                    pts = np.array(bbox_poly, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    # On identity cards and official documents, the handwritten signature
                    # is placed directly ABOVE the signature label/line.
                    sig_h = int(h * 2.8)
                    sig_y = max(0, y - sig_h)
                    sig_x = max(0, x - int(w * 0.10))
                    sig_w = min(img_w - sig_x, int(w * 1.2))
                    if sig_h > 10 and sig_w > 20:
                        # Refine bbox by finding actual ink strokes in the search region above the label
                        roi = image[sig_y:y, sig_x:sig_x + sig_w]
                        if roi.size > 0:
                            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            bg_val = float(np.percentile(roi_gray, 75))
                            dark_thresh = min(150, bg_val - 25)
                            stroke_mask = roi_gray < dark_thresh
                            stroke_pts = np.argwhere(stroke_mask)
                            if len(stroke_pts) > 40:
                                sy1 = int(stroke_pts[:, 0].min() + sig_y)
                                sy2 = int(stroke_pts[:, 0].max() + sig_y)
                                sx1 = int(stroke_pts[:, 1].min() + sig_x)
                                sx2 = int(stroke_pts[:, 1].max() + sig_x)
                                pad = 8
                                final_x = max(0, sx1 - pad)
                                final_y = max(0, sy1 - pad)
                                final_w = min(img_w - final_x, (sx2 - sx1) + 2 * pad)
                                final_h = min(img_h - final_y, (sy2 - sy1) + 2 * pad)
                            else:
                                final_x, final_y, final_w, final_h = sig_x, sig_y, sig_w, sig_h
                        else:
                            final_x, final_y, final_w, final_h = sig_x, sig_y, sig_w, sig_h

                        results.append(
                            DetectedEntity(
                                modality=Modality.IMAGE,
                                label="SIGNATURE",
                                text="SIGNATURE",
                                confidence=float(conf) if conf > 0.4 else 0.85,
                                bbox=BoundingBox(x=final_x, y=final_y, width=final_w, height=final_h),
                                detection_method="pattern",
                                mitigation="blur",
                            )
                        )

        # Skip HSV ink color segmentation on documents where face signature or text has been detected,
        # to avoid misidentifying blue garments, university logos, or banners as signatures/stamps
        if face_entities is not None or (ocr_results and len(ocr_results) >= 2):
            return results

        # Ink color segmentation (blue, violet ink for official stamps and signatures)
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask_blue = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([135, 255, 240]))
            mask_purple = cv2.inRange(hsv, np.array([135, 50, 50]), np.array([165, 255, 240]))
            mask_ink = mask_blue | mask_purple

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
