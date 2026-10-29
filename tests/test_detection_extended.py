import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from leakwatch.detection.image import ImageDetector
from leakwatch.utils.config import ImageConfig
from leakwatch.utils.types import BoundingBox, Modality

@pytest.fixture
def mock_config():
    return ImageConfig(
        enable_ocr=True,
        enable_qr=True,
        enable_layout=True,
        enable_mrz=True
    )

@pytest.fixture
def detector(mock_config):
    # We pass a mock text_detector if needed
    return ImageDetector(config=mock_config, text_detector=None)

def test_qr_detection(detector):
    # Mock image (just a blank array)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Mock the internal QR detector
    detector._qr_detector = MagicMock()
    # Mock QR detector output: (ok, decoded_info, points, _)
    mock_points = np.array([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=np.float32)
    detector._qr_detector.detectAndDecodeMulti.side_effect = (
        [(True, ["secret_data"], mock_points, None)] + [(False, [], None, None)] * 20
    )
    
    entities = list(detector._detect_qr(image))
    
    assert len(entities) >= 1
    assert entities[0].label == "qr_code"
    assert entities[0].text == "secret_data"
    assert entities[0].bbox.x == 5
    assert entities[0].bbox.width == 50

def test_mrz_detection(detector):
    # Mock OCR results: List[Tuple[points, text, confidence]]
    ocr_results = [
        ([(0, 0), (10, 0), (10, 5), (0, 5)], "P<USAHAPPY<<PEOPLE<<<<<<<<<<<<<<<<<<<<<<", 0.95),
        ([(0, 10), (10, 10), (10, 15), (0, 15)], "1234567890USA8001011M2512312<<<<<<<<<<<", 0.95)
    ]
    
    entities = list(detector._detect_mrz(ocr_results))
    
    # Both lines form a unified MRZ zone
    assert len(entities) == 1
    assert all(e.label == "mrz_zone" for e in entities)

@patch("cv2.imread")
def test_full_detection_pipeline(mock_imread, detector):
    # Mock imread to return a blank image
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Mock internal methods to avoid real OCR/Cascades
    detector._detect_faces = MagicMock(return_value=[])
    detector._detect_qr = MagicMock(return_value=[])
    detector._get_ocr_results = MagicMock(return_value=[])
    
    entities = detector.detect(Path("fake_image.jpg"))
    
    # Should call all 4 main branches (Face, QR, OCR/MRZ/Layout)
    assert detector._detect_faces.called
    assert detector._detect_qr.called
    assert detector._get_ocr_results.called
