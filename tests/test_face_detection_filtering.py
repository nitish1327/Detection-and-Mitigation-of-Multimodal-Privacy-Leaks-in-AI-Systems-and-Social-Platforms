from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path

from leakwatch.video.tracker import create_face_detector, detect_faces


def test_detect_faces_filters_low_confidence_and_noise() -> None:
    detector = create_face_detector(320, 320, score_threshold=0.60)
    
    # Blank frame
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    faces = detect_faces(blank, detector)
    assert faces == []

    # Frame with uniform texture / noise that could produce weak false positives
    noisy = np.random.randint(50, 180, (320, 320, 3), dtype=np.uint8)
    faces = detect_faces(noisy, detector)
    assert faces == []


def test_detect_faces_rejects_hands_and_chest_on_sample_videos() -> None:
    """Verify that user sample videos with hands and chest now detect ONLY genuine faces."""
    detector = create_face_detector(320, 320, score_threshold=0.60)

    # 1. VID_20261027_105952399.mp4 (previously had 4 detections: 2 faces, 1 waving hand, 1 man's chest)
    v1_path = Path("samples/video/VID_20261027_105952399.mp4")
    if v1_path.exists():
        cap = cv2.VideoCapture(str(v1_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            faces = detect_faces(frame, detector)
            # Must detect exactly the 2 human faces (woman and man), rejecting hand and chest
            assert len(faces) == 2, f"Expected 2 faces, got {len(faces)}: {faces}"

    # 2. WhatsApp Video 2026-10-27 at 11.23.33 AM.mp4 (previously had 3 detections: 2 faces, 1 moving hand)
    v2_path = Path("samples/video/WhatsApp Video 2026-10-27 at 11.23.33 AM.mp4")
    if v2_path.exists():
        cap = cv2.VideoCapture(str(v2_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            faces = detect_faces(frame, detector)
            # Must detect exactly the 2 human faces (man and woman), rejecting the moving hand
            assert len(faces) == 2, f"Expected 2 faces, got {len(faces)}: {faces}"

