from __future__ import annotations

import cv2
import numpy as np
import pytest

from leakwatch.video.tracker import blur_region


def test_video_face_blur_strength_obscures_identity() -> None:
    """Verify that blur_region substantially reduces high-frequency detail so faces are difficult to identify."""
    # Create a synthetic face-like pattern with sharp high-frequency features (eyes, nose, mouth lines)
    face_img = np.full((120, 100, 3), 180, dtype=np.uint8)
    # Eyes
    cv2.circle(face_img, (35, 45), 8, (20, 20, 20), -1)
    cv2.circle(face_img, (65, 45), 8, (20, 20, 20), -1)
    # Eyebrows
    cv2.line(face_img, (25, 32), (45, 32), (10, 10, 10), 3)
    cv2.line(face_img, (55, 32), (75, 32), (10, 10, 10), 3)
    # Nose
    cv2.line(face_img, (50, 48), (50, 68), (40, 40, 40), 2)
    # Mouth
    cv2.ellipse(face_img, (50, 85), (20, 8), 0, 0, 180, (30, 30, 120), -1)

    frame = np.full((240, 200, 3), 220, dtype=np.uint8)
    # Place face in the center
    frame[60:180, 50:150] = face_img
    bbox = (50, 60, 100, 120)

    # Compute pre-blur detail
    gray_pre = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    var_pre = cv2.Laplacian(gray_pre, cv2.CV_64F).var()
    sobelx_pre = cv2.Sobel(gray_pre, cv2.CV_64F, 1, 0, ksize=3)
    sobely_pre = cv2.Sobel(gray_pre, cv2.CV_64F, 0, 1, ksize=3)
    grad_pre = float(np.mean(np.sqrt(sobelx_pre**2 + sobely_pre**2)))

    assert var_pre > 100.0, "Pre-blur synthetic face should have high Laplacian variance"
    assert grad_pre > 10.0, "Pre-blur synthetic face should have high gradient magnitude"

    # Apply default heavy face blur
    blurred_frame = blur_region(frame.copy(), bbox)

    blurred_face = blurred_frame[60:180, 50:150]
    gray_post = cv2.cvtColor(blurred_face, cv2.COLOR_BGR2GRAY)
    var_post = cv2.Laplacian(gray_post, cv2.CV_64F).var()
    sobelx_post = cv2.Sobel(gray_post, cv2.CV_64F, 1, 0, ksize=3)
    sobely_post = cv2.Sobel(gray_post, cv2.CV_64F, 0, 1, ksize=3)
    grad_post = float(np.mean(np.sqrt(sobelx_post**2 + sobely_post**2)))

    # Detail reduction must be >= 85% to ensure facial identity cannot be recognized
    assert var_post < var_pre * 0.15, f"Expected >85% variance reduction, got pre={var_pre}, post={var_post}"
    assert grad_post < grad_pre * 0.25, f"Expected >75% gradient magnitude reduction, got pre={grad_pre}, post={grad_post}"


def test_video_face_blur_handles_boundary_and_invalid_boxes() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Fully out of bounds bbox
    out = blur_region(frame, (-50, -50, 10, 10))
    assert out.shape == frame.shape

    # Partially out of bounds bbox (border touching)
    out2 = blur_region(frame, (0, 0, 50, 50))
    assert out2.shape == frame.shape
    assert not np.isnan(out2).any()

