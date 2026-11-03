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


def test_sensitive_numbers_partial_special_char_masking(tmp_path):
    path = tmp_path / "card_test.png"
    img = np.ones((300, 500, 3), dtype=np.uint8) * 240
    cv2.imwrite(str(path), img)

    config = ImageConfig(default_style="synthetic")
    mitigator = ImageMitigator(config)

    test_cases = [
        # (label, original_text, expected_mask, min_special, max_special)
        ("AADHAR", "9006 2534 7320", "9006 **** 7320", 4, 4),
        ("AADHAR", "900625347320", "9006****7320", 4, 4),
        ("PAN", "BKZPR4928K", "BKZPR****K", 4, 4),
        ("PASSPORT_NUMBER", "U6624402", "U66****2", 3, 4),
        ("PASSPORT", "SP003362", "SP0****2", 3, 4),
        ("LICENSE", "KA01 20180004521", "KA01 ****0004521", 4, 4),
        ("LICENSE", "AP00820210048192", "AP008****0048192", 4, 4),
        ("credit_card", "4532 1234 5678 9012", "4532 1234 **** 9012", 4, 4),
        ("CREDIT_CARD", "4532-1234-5678-9012", "4532-1234-****-9012", 4, 4),
        ("credit_card", "4532123456789012", "45321234****9012", 4, 4),
    ]

    for idx, (label, orig_text, expected_mask, min_spec, max_spec) in enumerate(test_cases):
        entity = DetectedEntity(
            modality=Modality.IMAGE,
            label=label,
            confidence=0.98,
            text=orig_text,
            bbox=BoundingBox(x=20, y=20 + idx * 25, width=200, height=20),
        )
        out_path = tmp_path / f"out_{idx}.png"
        _, updated = mitigator.mitigate(path, [entity], out_path)

        mitigated_str = updated[0].text
        # Must not be completely replaced with a fake one
        assert mitigated_str != orig_text, f"Failed for {label}: text was not mitigated"
        assert mitigated_str == expected_mask, f"Failed for {label}: expected '{expected_mask}', got '{mitigated_str}'"

        # Count how many characters differ between original and mitigated
        diffs = [i for i in range(min(len(orig_text), len(mitigated_str))) if orig_text[i] != mitigated_str[i]]
        diff_count = len(diffs)
        assert min_spec <= diff_count <= max_spec, f"Expected {min_spec} to {max_spec} characters replaced, got {diff_count}"

        # Replaced characters must be special characters
        for i in diffs:
            assert not mitigated_str[i].isalnum(), f"Expected special character at position {i}, got '{mitigated_str[i]}'"


def test_person_name_still_uses_synthetic_name(tmp_path):
    path = tmp_path / "name_test.png"
    img = np.ones((100, 300, 3), dtype=np.uint8) * 240
    cv2.imwrite(str(path), img)

    config = ImageConfig(default_style="synthetic")
    mitigator = ImageMitigator(config)

    entity = DetectedEntity(
        modality=Modality.IMAGE,
        label="PERSON",
        confidence=0.95,
        text="Arjun Sharma",
        bbox=BoundingBox(x=20, y=20, width=150, height=25),
    )
    out_path = tmp_path / "name_out.png"
    _, updated = mitigator.mitigate(path, [entity], out_path)

    mitigated_str = updated[0].text
    # Name should be replaced with synthetic name (not asterisks)
    assert "*" not in mitigated_str
    assert mitigated_str != "Arjun Sharma"


