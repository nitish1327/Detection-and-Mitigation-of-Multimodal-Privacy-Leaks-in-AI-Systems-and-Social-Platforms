import pytest
import numpy as np
import cv2
from pathlib import Path
from leakwatch.mitigation.image import ImageMitigator
from leakwatch.utils.config import ImageConfig
from leakwatch.utils.types import DetectedEntity, BoundingBox, Modality

@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "test.png"
    # Create a 100x100 white image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(path), img)
    return path

def test_padding_application(sample_image):
    config = ImageConfig(padding_percent=0.1, default_style="redact")
    mitigator = ImageMitigator(config)
    
    entity = DetectedEntity(
        modality=Modality.IMAGE,
        label="face",
        confidence=1.0,
        bbox=BoundingBox(x=40, y=40, width=20, height=20)
    )
    
    output_path = sample_image.with_name("padded.png")
    mitigator.mitigate(sample_image, [entity], output_path)
    
    img = cv2.imread(str(output_path))
    
    # Redact color is 50. PNG is lossless.
    assert np.all(img[38, 38] == 50), f"Expected [50,50,50] at (38,38), got {img[38,38]}"
    assert np.all(img[37, 37] == 255), f"Expected [255,255,255] at (37,37), got {img[37,37]}"

def test_pixelate_style(sample_image):
    config = ImageConfig(default_style="pixelate", padding_percent=0)
    mitigator = ImageMitigator(config)
    
    entity = DetectedEntity(
        modality=Modality.IMAGE,
        label="person",
        confidence=1.0,
        bbox=BoundingBox(x=10, y=10, width=50, height=50)
    )
    
    # Create a gradient image
    img_multi = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img_multi[:, i] = i
    cv2.imwrite(str(sample_image), img_multi)
    
    output_path = sample_image.with_name("pixelated.png")
    mitigator.mitigate(sample_image, [entity], output_path)
    
    img_out = cv2.imread(str(output_path))
    roi_out = img_out[10:60, 10:60]
    
    # In a 5x5 grid for 50x50 ROI, pixels 0 and 1 should be identical in a lossless format
    assert np.all(roi_out[0, 0] == roi_out[0, 1]), f"Pixels at (0,0) and (0,1) differ: {roi_out[0,0]} vs {roi_out[0,1]}"

def test_redact_style(sample_image):
    config = ImageConfig(default_style="redact", padding_percent=0)
    mitigator = ImageMitigator(config)
    
    entity = DetectedEntity(
        modality=Modality.IMAGE,
        label="person",
        confidence=1.0,
        bbox=BoundingBox(x=10, y=10, width=10, height=10)
    )
    
    output_path = sample_image.with_name("redacted.jpg")
    mitigator.mitigate(sample_image, [entity], output_path)
    
    img = cv2.imread(str(output_path))
    assert np.all(img[15, 15] == 50)

def test_aggressive_blur_forced(sample_image):
    config = ImageConfig(default_style="pixelate") # Normal is pixelate
    mitigator = ImageMitigator(config)
    
    # QR Code should be blurred regardless of default_style
    entity = DetectedEntity(
        modality=Modality.IMAGE,
        label="qr_code",
        confidence=1.0,
        bbox=BoundingBox(x=10, y=10, width=20, height=20)
    )
    
    output_path = sample_image.with_name("qr_blurred.jpg")
    _, updated = mitigator.mitigate(sample_image, [entity], output_path)
    
    assert updated[0].mitigation == "aggressive_blur"


def test_face_tweaking_perturbation(tmp_path):
    path = tmp_path / "face_test.png"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, (200, 180, 150), -1)
    cv2.circle(img, (85, 90), 5, (50, 50, 50), -1)
    cv2.circle(img, (115, 90), 5, (50, 50, 50), -1)
    cv2.imwrite(str(path), img)

    config = ImageConfig(default_style="synthetic")
    mitigator = ImageMitigator(config)

    entity = DetectedEntity(
        modality=Modality.IMAGE,
        label="face",
        confidence=0.95,
        bbox=BoundingBox(x=60, y=60, width=80, height=80),
    )

    output_path = tmp_path / "face_perturbed.png"
    out, updated = mitigator.mitigate(path, [entity], output_path)

    assert updated[0].text == "[PERTURBED_FACE]"
    assert updated[0].mitigation == "replace"
    assert output_path.exists()

