"""Video processing and face tracking utilities for LeakWatch."""

from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# Model location and download URL
_ASSET_DIR = Path(__file__).resolve().parent.parent / "utils" / "assets"
_MODEL_PATH = _ASSET_DIR / "face_detection_yunet_2023mar.onnx"
_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_yunet_model(target_path: Path | None = None) -> str:
    """Ensure the YuNet ONNX face detection model is available locally."""
    path = target_path or _MODEL_PATH
    if path.exists():
        return str(path)

    LOGGER.info("Downloading YuNet face detection model to %s...", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(_MODEL_URL, str(path))
    LOGGER.info("YuNet model saved to %s", path)
    return str(path)


def create_face_detector(
    width: int = 320,
    height: int = 320,
    score_threshold: float = 0.22,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> cv2.FaceDetectorYN:
    """Create an OpenCV FaceDetectorYN instance using the YuNet ONNX model."""
    model_path = ensure_yunet_model()
    detector = cv2.FaceDetectorYN_create(
        model=model_path,
        config="",
        input_size=(max(1, width), max(1, height)),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )
    return detector


def detect_faces(
    frame: np.ndarray,
    detector: cv2.FaceDetectorYN,
) -> List[Tuple[int, int, int, int]]:
    """Return a list of (x, y, w, h) face bounding boxes in the frame."""
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, raw = detector.detect(frame)
    if raw is None:
        return []
    return [
        (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
        for face in raw
    ]


class FaceTracker:
    """Track a single face across video frames using per-frame detection + association.

    Uses YuNet face detection on every frame and associates the target face
    with its closest match using spatial proximity + size similarity, with
    a velocity prediction model for smoother tracking.

    Handles temporary occlusion (e.g. another person walking in front) by
    remembering the anchor position and searching with an expanding radius
    for up to ~1.2 seconds (35 frames at 30fps).

    When a person exits the frame edges, extrapolates the bounding box
    outward using velocity for up to 8 frames so the blur continues smoothly
    until the head is completely off-screen, preventing unblurred face slices.
    """

    MAX_LOST_ANCHOR = 35  # ~1.2s at 30fps -- search near anchor for occlusion
    MAX_EDGE_COAST = 8    # frames to extrapolate while exiting frame edge

    def __init__(
        self,
        initial_bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.last_bbox = initial_bbox
        self.initial_size = (initial_bbox[2], initial_bbox[3])
        self.anchor_bbox = initial_bbox
        self.vx = 0.0  # velocity estimate (pixels/frame)
        self.vy = 0.0
        self.lost_count = 0
        self.active = True
        self.frame_width = frame_width
        self.frame_height = frame_height

    def update(
        self,
        frame: np.ndarray,
        detector: cv2.FaceDetectorYN,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return the tracked bbox for this frame, or None if temporarily lost / exited."""
        if not self.active:
            return None

        faces = detect_faces(frame, detector)
        lx, ly, lw, lh = self.last_bbox

        # Proximity to frame boundary
        margin = 50
        near_edge = (
            lx <= margin
            or ly <= margin
            or (lx + lw) >= self.frame_width - margin
            or (ly + lh) >= self.frame_height - margin
        )

        # Expected center position
        if self.lost_count > 0:
            ax, ay, aw, ah = self.anchor_bbox
            search_cx = ax + aw / 2
            search_cy = ay + ah / 2
            ref_w, ref_h = self.initial_size
        else:
            search_cx = lx + lw / 2 + self.vx
            search_cy = ly + lh / 2 + self.vy
            ref_w, ref_h = lw, lh

        # Search radius expands during occlusion
        search_radius = max(140, 90 + self.lost_count * 4)

        best_box: Optional[Tuple[int, int, int, int]] = None
        best_dist = float("inf")

        for (fx, fy, fw, fh) in faces:
            fcx, fcy = fx + fw / 2, fy + fh / 2
            dist = ((fcx - search_cx) ** 2 + (fcy - search_cy) ** 2) ** 0.5
            target_area = ref_w * ref_h
            cand_area = fw * fh
            size_ratio = max(target_area, cand_area) / (min(target_area, cand_area) + 1e-5)

            if dist < search_radius and size_ratio < 2.5 and dist < best_dist:
                best_dist = dist
                best_box = (fx, fy, fw, fh)

        if best_box is not None:
            # Re-acquire / update tracking
            if self.lost_count == 0:
                self.vx = 0.6 * (best_box[0] - lx) + 0.4 * self.vx
                self.vy = 0.6 * (best_box[1] - ly) + 0.4 * self.vy
            else:
                self.vx = 0.0
                self.vy = 0.0
            self.last_bbox = best_box
            self.anchor_bbox = best_box
            self.lost_count = 0
            self.active = True
            return best_box

        # Face was not detected on this frame
        self.lost_count += 1

        if near_edge:
            if self.lost_count <= self.MAX_EDGE_COAST:
                cur_x = int(lx + self.vx)
                cur_y = int(ly + self.vy)
                extrapolated = (cur_x, cur_y, lw, lh)
                self.last_bbox = extrapolated
                if (
                    cur_x >= self.frame_width
                    or (cur_x + lw) <= 0
                    or cur_y >= self.frame_height
                    or (cur_y + lh) <= 0
                ):
                    self.active = False
                    return None
                return extrapolated
            else:
                self.active = False
                return None

        # Lost in interior (e.g. hand/object occlusion): coast at anchor position for up to 90 frames (~3 sec)
        if self.lost_count <= 90:
            return self.anchor_bbox

        return None


def blur_region(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    strength: int = 51,
) -> np.ndarray:
    """Apply elliptical Gaussian blur to the bbox region of the frame (in-place).

    Expands the bounding box slightly to cover forehead/chin, and uses an
    elliptical mask with soft edges for a natural-looking blur.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = bbox

    # Expand slightly beyond face bbox to cover forehead/chin
    pad_w = int(w * 0.15)
    pad_h = int(h * 0.20)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(fw, x + w + pad_w)
    y2 = min(fh, y + h + pad_h)

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2].copy()
    k = strength if strength % 2 == 1 else strength + 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    # Create elliptical mask for smooth blending
    rh, rw = roi.shape[:2]
    mask = np.zeros((rh, rw), dtype=np.float32)
    center = (rw // 2, rh // 2)
    axes = (rw // 2, rh // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # If the region touches image borders, ensure full blur coverage at boundary
    if x1 == 0:
        mask[:, 0:max(1, pad_w)] = 1.0
    if x2 == fw:
        mask[:, -max(1, pad_w):] = 1.0
    if y1 == 0:
        mask[0:max(1, pad_h), :] = 1.0
    if y2 == fh:
        mask[-max(1, pad_h):, :] = 1.0

    mask_3ch = np.expand_dims(mask, axis=2)
    blended = (blurred * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    return frame


def generate_synthetic_replacement(label: str, orig_text: str | None = None) -> str:
    """Generate realistic fake text for a sensitive entity label."""
    lbl = (label or "").lower()
    orig = (orig_text or "").strip()

    if any(k in lbl for k in ("id", "roll", "license", "cardinal", "number")):
        if orig and any(c.isalpha() for c in orig) and any(c.isdigit() for c in orig):
            prefix = "".join(c for c in orig if c.isalpha())[:4] or "ID"
            digits = "".join(c for c in orig if c.isdigit())[:6] or "202401"
            mutated = "".join(str((int(d) + 3) % 10) for d in digits)
            return f"{prefix}{mutated}"
        if orig and orig.isdigit():
            return "".join(str((int(d) + 5) % 10) for d in orig)
        return "PES2UG22CS999"

    if "aadhar" in lbl:
        return "XXXX XXXX 4821"

    if any(k in lbl for k in ("person", "name")):
        return "ADITYA SHARMA"

    if any(k in lbl for k in ("institution", "org")):
        if "univ" in orig.lower():
            return "NATIONAL UNIVERSITY"
        if "pes" in orig.lower():
            return "APEX"
        return "INSTITUTE"

    if "phone" in lbl:
        return "+91 98765 43210"

    if any(k in lbl for k in ("credit", "bank", "account", "money")):
        return "4532 **** **** 8821"

    if any(k in lbl for k in ("address", "street", "location", "gpe")):
        return "123 MG Road, Bengaluru"

    if "email" in lbl:
        return "contact@example.org"

    return "CONFIDENTIAL"


def render_synthetic_text(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    orig_text: str,
    fake_text: str,
) -> np.ndarray:
    """Inpaint original sensitive text and render crisp synthetic replacement text."""
    fh, fw = frame.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(fw, int(x + w))
    y2 = min(fh, int(y + h))
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 4 or (y2 - y1) < 4:
        return frame

    roi = frame[y1:y2, x1:x2]
    is_vertical = (y2 - y1) > (x2 - x1) * 1.35
    if is_vertical:
        # Rotate 90 counter-clockwise so text can be inpainted and rendered horizontally
        roi_to_process = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        roi_to_process = roi

    rh, rw = roi_to_process.shape[:2]

    # Background color detection from border pixels
    border = np.concatenate([roi_to_process[0, :], roi_to_process[-1, :], roi_to_process[:, 0], roi_to_process[:, -1]])
    bg_color = np.median(border, axis=0).astype(np.uint8).tolist()
    is_light_bg = float(np.mean(bg_color)) > 127.0
    text_color = (25, 25, 25) if is_light_bg else (235, 235, 235)

    # Inpaint original text
    gray = cv2.cvtColor(roi_to_process, cv2.COLOR_BGR2GRAY)
    if is_light_bg:
        _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)
    erased = cv2.inpaint(roi_to_process, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Feathered background fill in center
    margin = max(2, min(rh, rw) // 10)
    fill_region = np.full_like(erased, bg_color, dtype=np.uint8)
    blend_mask = np.zeros((rh, rw), dtype=np.float32)
    blend_mask[margin:rh - margin, margin:rw - margin] = 1.0
    ksize = margin * 2 + 1
    blend_mask = cv2.GaussianBlur(blend_mask, (ksize, ksize), 0)
    blend_3ch = cv2.merge([blend_mask, blend_mask, blend_mask])
    erased = (
        fill_region.astype(np.float32) * blend_3ch
        + erased.astype(np.float32) * (1.0 - blend_3ch)
    ).clip(0, 255).astype(np.uint8)

    # Render synthetic text at matching scale
    font = cv2.FONT_HERSHEY_DUPLEX
    target_w = int(rw * 0.90)
    target_h = int(rh * 0.72)
    ref_text = orig_text if orig_text else fake_text

    lo, hi = 0.1, 4.0
    scale = 0.5
    for _ in range(15):
        mid = (lo + hi) / 2.0
        ts = cv2.getTextSize(ref_text, font, mid, max(1, int(mid)))[0]
        if ts[0] <= target_w and ts[1] <= target_h:
            scale = mid
            lo = mid
        else:
            hi = mid

    thickness = max(1, int(scale))
    text_size = cv2.getTextSize(fake_text, font, scale, thickness)[0]
    if text_size[0] > target_w:
        scale = max(0.2, scale * (target_w / max(1, text_size[0])))
        thickness = max(1, int(scale))
        text_size = cv2.getTextSize(fake_text, font, scale, thickness)[0]

    text_x = max(2, (rw - text_size[0]) // 2)
    text_y = max(rh - 2, (rh + text_size[1]) // 2)
    cv2.putText(erased, fake_text, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)

    if is_vertical:
        # Rotate back 90 clockwise
        erased = cv2.rotate(erased, cv2.ROTATE_90_CLOCKWISE)

    frame[y1:y2, x1:x2] = erased
    return frame


def find_card_region(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect bright document / ID card region against darker background."""
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if 20000 < area < 700000:
            x, y, w, h = cv2.boundingRect(c)
            if y > 80 and area > best_area:
                best_area = area
                best_box = (x, y, w, h)
    return best_box


class SensitiveTextTracker:
    """Tracks a sensitive text region across frames and applies synthetic Faker text.

    Combines:
    1. Card-relative normalized coordinates for rigid document cards with perspective scaling.
    2. Lucas-Kanade sparse optical flow on Shi-Tomasi feature points for sub-pixel motion tracking.
    3. Normalized cross-correlation template matching with wide adaptive search for drift correction.
    4. Multi-entity group displacement propagation for rigid cards/documents during occlusions.
    5. Velocity extrapolation during momentary blur/swipes.
    6. Zero-leak guarantee: Never drops redaction while on screen.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        orig_text: str,
        fake_text: str,
        initial_frame: np.ndarray,
        anchor_box: Optional[Tuple[int, int, int, int]] = None,
        card_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.label = label
        self.orig_text = orig_text
        self.fake_text = fake_text
        self.lost_frames = 0
        self.vx = 0.0
        self.vy = 0.0
        self.last_shift: tuple[float, float] = (0.0, 0.0)

        # Card-relative normalized coordinates
        if card_box is not None and card_box[2] > 0 and card_box[3] > 0:
            cx, cy, cw, ch = card_box
            self.card_rel: Optional[Tuple[float, float, float, float]] = (
                float(self.x - cx) / float(cw),
                float(self.y - cy) / float(ch),
                max(float(self.w), float(cw * 0.045)) / float(cw),
                float(self.h) / float(ch),
            )
            self.last_card_box: Optional[Tuple[int, int, int, int]] = card_box
        else:
            self.card_rel = None
            self.last_card_box = None

        # Relative offset from face anchor box
        if anchor_box is not None:
            self.anchor_offset: Optional[Tuple[float, float]] = (
                float(self.x - anchor_box[0]),
                float(self.y - anchor_box[1]),
            )
        else:
            self.anchor_offset = None

        fh, fw = initial_frame.shape[:2]
        cx1 = max(0, int(self.x))
        cy1 = max(0, int(self.y))
        cx2 = min(fw, int(self.x + self.w))
        cy2 = min(fh, int(self.y + self.h))
        self.template = (
            initial_frame[cy1:cy2, cx1:cx2].copy()
            if (cx2 > cx1 and cy2 > cy1)
            else None
        )

        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[cy1:cy2, cx1:cx2] = 255
        self.prev_gray = gray
        self.p0 = cv2.goodFeaturesToTrack(
            gray, mask=mask, maxCorners=35, qualityLevel=0.01, minDistance=4
        )

    def update(
        self,
        frame: np.ndarray,
        group_dx: float = 0.0,
        group_dy: float = 0.0,
        current_anchor: Optional[Tuple[int, int, int, int]] = None,
        curr_card_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> tuple[int, int, int, int] | None:
        fh, fw = frame.shape[:2]
        # Check if permanently off-screen
        if (
            self.x + self.w <= -50
            or self.x >= fw + 50
            or self.y + self.h <= -50
            or self.y >= fh + 50
        ):
            return None

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = 0.0, 0.0
        found = False

        # 0. Card-relative tracking (highest precision for document / ID card text)
        if self.card_rel is not None:
            c_box = curr_card_box or self.last_card_box
            if c_box is not None:
                self.last_card_box = c_box
                cx, cy, cw, ch = c_box
                self.x = float(cx + self.card_rel[0] * cw)
                self.y = float(cy + self.card_rel[1] * ch)
                self.w = float(self.card_rel[2] * cw)
                self.h = float(self.card_rel[3] * ch)
                self.lost_frames = 0
                found = True

        # 0. Anchor-assisted tracking for document / ID card text
        if self.anchor_offset is not None:
            eff_anchor = current_anchor or getattr(self, "last_anchor", None)
            if eff_anchor is not None:
                if current_anchor is not None:
                    self.last_anchor = current_anchor
                base_x = float(eff_anchor[0] + self.anchor_offset[0])
                base_y = float(eff_anchor[1] + self.anchor_offset[1])
                search_pad = 55
                sx1 = max(0, int(base_x - search_pad))
                sy1 = max(0, int(base_y - search_pad))
                sx2 = min(fw, int(base_x + self.w + search_pad))
                sy2 = min(fh, int(base_y + self.h + search_pad))

                if self.template is not None and self.template.size > 0:
                    th, tw = self.template.shape[:2]
                    if (sx2 - sx1) >= tw and (sy2 - sy1) >= th:
                        roi = frame[sy1:sy2, sx1:sx2]
                        res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val > 0.35:
                            self.x = float(sx1 + max_loc[0])
                            self.y = float(sy1 + max_loc[1])
                            self.lost_frames = 0
                            found = True

                if not found:
                    # Occluded or low correlation: snap to anchor position to guarantee zero leakage
                    self.x = base_x
                    self.y = base_y
                    self.lost_frames = 0
                    found = True

        # 1. Optical flow tracking (fallback when no anchor)
        if not found and self.p0 is not None and len(self.p0) >= 3:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, curr_gray, self.p0, None, winSize=(21, 21), maxLevel=3
            )
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            if len(good_new) >= 3:
                shifts = good_new - good_old
                dx = float(np.median(shifts[:, 0]))
                dy = float(np.median(shifts[:, 1]))
                self.x += dx
                self.y += dy
                self.p0 = good_new.reshape(-1, 1, 2)
                found = True

        # 2. Template matching refinement / recovery
        if not found and self.template is not None and self.template.size > 0:
            th, tw = self.template.shape[:2]
            search_pad = 85
            sx1 = max(0, int(self.x - search_pad))
            sy1 = max(0, int(self.y - search_pad))
            sx2 = min(fw, int(self.x + self.w + search_pad))
            sy2 = min(fh, int(self.y + self.h + search_pad))

            if (sx2 - sx1) >= tw and (sy2 - sy1) >= th:
                roi = frame[sy1:sy2, sx1:sx2]
                res = cv2.matchTemplate(roi, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.40:
                    matched_x = sx1 + max_loc[0]
                    matched_y = sy1 + max_loc[1]
                    self.x = float(matched_x)
                    self.y = float(matched_y)
                    self.lost_frames = 0
                    found = True

        if not found:
            self.lost_frames += 1
            if abs(group_dx) > 0.1 or abs(group_dy) > 0.1:
                self.x += group_dx
                self.y += group_dy
                dx, dy = group_dx, group_dy
            else:
                self.x += self.vx
                self.y += self.vy
                dx, dy = self.vx, self.vy

        self.last_shift = (dx, dy)
        self.prev_gray = curr_gray

        # Constrain within frame bounds
        self.x = max(-10.0, min(float(fw - 5), self.x))
        self.y = max(-10.0, min(float(fh - 5), self.y))
        return (int(self.x), int(self.y), int(self.w), int(self.h))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return render_synthetic_text(
            frame, int(self.x), int(self.y), int(self.w), int(self.h), self.orig_text, self.fake_text
        )


def find_ffmpeg() -> str | None:
    """Locate ffmpeg executable across system PATH, imageio_ffmpeg, and local shims."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        fb = get_ffmpeg_exe()
        if fb and Path(fb).exists():
            return fb
    except ImportError:
        pass
    shim = Path(__file__).resolve().parent.parent.parent / ".ffmpeg_shim" / "ffmpeg.exe"
    if shim.exists():
        return str(shim)
    return None


def has_audio_stream(video_path: Path | str) -> bool:
    """Check if the video container includes an audio stream."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-i", str(video_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr.splitlines():
        if "Stream #" in line and "Audio:" in line:
            return True
    return False


def extract_audio(video_path: Path | str, output_wav: Path | str) -> bool:
    """Extract 16kHz mono PCM WAV for Whisper transcription and audio mitigation."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0


def reconstruct_with_audio(
    raw_video: Path,
    source_video: Path,
    output_path: Path,
    sanitized_audio_path: Path | None = None,
) -> Path:
    """Transcode raw_video to H.264 and mux audio (sanitized if provided, else source) using ffmpeg."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        LOGGER.warning("ffmpeg not found -- output will lack audio and use raw container.")
        if raw_video.resolve() != output_path.resolve():
            shutil.copy2(raw_video, output_path)
            raw_video.unlink(missing_ok=True)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_source = (
        sanitized_audio_path
        if (sanitized_audio_path and Path(sanitized_audio_path).exists())
        else source_video
    )
    cmd = [
        ffmpeg, "-y",
        "-i", str(raw_video),
        "-i", str(audio_source),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-map_metadata", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.warning("ffmpeg transcode failed: %s", result.stderr[:500])
        if raw_video.resolve() != output_path.resolve():
            shutil.copy2(raw_video, output_path)
        return output_path

    raw_video.unlink(missing_ok=True)
    return output_path


_VIDEO_SENSITIVE_ENTITIES_CACHE: dict[str, list[dict[str, Any]]] = {}


def _video_fingerprint(video_path: Path | str) -> str:
    p = Path(video_path)
    if not p.exists():
        return ""
    try:
        import hashlib
        size = p.stat().st_size
        with open(p, "rb") as f:
            head = f.read(8192)
        h = hashlib.sha256(head).hexdigest()[:16]
        return f"{size}_{h}"
    except Exception:
        return str(p)


def generate_video_face_preview(video_path: Path | str) -> dict[str, Any]:
    """Inspect first valid frame, detect faces with YuNet and sensitive text, and generate preview payload."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        raise RuntimeError("Could not decode any valid frame from video.")

    if width <= 0 or height <= 0:
        height, width = first_frame.shape[:2]

    detector = create_face_detector(width, height)
    faces = detect_faces(first_frame, detector)

    # Annotate copy for visual preview
    annotated = first_frame.copy()
    face_items: list[dict[str, Any]] = []

    for i, (fx, fy, fw, fh) in enumerate(faces):
        color = (0, 255, 0)
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, 2)
        label = f"#{i + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(annotated, (fx, fy - th - 10), (fx + tw + 6, fy), color, -1)
        cv2.putText(annotated, label, (fx + 3, fy - 5), font, 0.7, (0, 0, 0), 2)

        # Extract face crop/thumbnail
        y1 = max(0, fy)
        y2 = min(height, fy + fh)
        x1 = max(0, fx)
        x2 = min(width, fx + fw)
        crop = first_frame[y1:y2, x1:x2]
        thumb_b64 = ""
        if crop.size > 0:
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                thumb_b64 = base64.b64encode(buf).decode("ascii")

        face_items.append({
            "id": i + 1,
            "x": fx,
            "y": fy,
            "w": fw,
            "h": fh,
            "bbox": [fx, fy, fw, fh],
            "thumbnail_base64": thumb_b64,
        })

    # Detect sensitive text & document entities in first frame
    sensitive_items: list[dict[str, Any]] = []
    try:
        from ..detection.image import ImageDetector
        from ..utils.config import ImageConfig

        img_cfg = ImageConfig()
        img_detector = ImageDetector(img_cfg)
        detected_entities = img_detector.detect_frame(first_frame)

        s_idx = 1
        for ent in detected_entities:
            if not ent.bbox:
                continue
            lbl = ent.label.lower()
            txt = (ent.text or "").strip()
            # Skip pure face entities since they are in faces list
            if lbl in {"face", "person"} and not txt:
                continue
            # Exclude known geographic words from desk mats / map backgrounds
            if txt.lower() in {
                "ocean", "atlantic", "pacific", "indian", "arctic", "south", "north",
                "east", "west", "africa", "asia", "europe", "america", "sea", "bay",
                "lat", "lon", "gmt", "utc", "equator", "tropic", "meridian",
            }:
                continue

            bx, by, bw, bh = int(ent.bbox.x), int(ent.bbox.y), int(ent.bbox.width), int(ent.bbox.height)
            if bw < 8 or bh < 6:
                continue

            fake_txt = generate_synthetic_replacement(ent.label, txt)
            sensitive_items.append({
                "id": s_idx,
                "label": ent.label,
                "text": txt,
                "fake_text": fake_txt,
                "x": bx,
                "y": by,
                "w": bw,
                "h": bh,
                "bbox": [bx, by, bw, bh],
            })
            s_idx += 1

            # Draw preview box on annotated image
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
            tag = f"[{ent.label.upper()}]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (bx, max(0, by - th - 6)), (bx + tw + 4, by), (255, 200, 0), -1)
            cv2.putText(annotated, tag, (bx + 2, max(th, by - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    except Exception as exc:
        LOGGER.warning("Sensitive entity detection on video preview: %s", exc)

    fp = _video_fingerprint(path)
    if fp:
        _VIDEO_SENSITIVE_ENTITIES_CACHE[fp] = sensitive_items

    # Encode annotated frame preview
    ok_prev, buf_prev = cv2.imencode(".jpg", annotated)
    preview_b64 = base64.b64encode(buf_prev).decode("ascii") if ok_prev else ""

    return {
        "face_count": len(face_items),
        "has_audio": has_audio_stream(path),
        "preview_image_base64": preview_b64,
        "annotated_frame_base64": preview_b64,
        "faces": face_items,
        "sensitive_entities": sensitive_items,
        "sensitive_entity_count": len(sensitive_items),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        },
    }


def process_selective_video(
    source_video_path: Path | str,
    output_path: Path | str,
    selected_face_ids: Sequence[int],
    mode: str = "blur_selected",
    blur_strength: int = 51,
    redact_sensitive_info: bool = True,
    mitigate_audio: bool = False,
    pipeline_manager: Any = None,
    end_frame: int | None = None,
) -> dict[str, Any]:
    """Track and blur selected faces across all frames and replace sensitive text with Faker."""
    video_path = Path(source_video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Cannot decode first frame of video.")

    if width <= 0 or height <= 0:
        height, width = first_frame.shape[:2]

    detector = create_face_detector(width, height)
    faces = detect_faces(first_frame, detector)

    # Determine faces to blur based on 1-indexed IDs and mode
    selected_set = {int(x) for x in selected_face_ids}
    if mode == "blur_all":
        target_faces = list(faces)
        target_ids = list(range(1, len(faces) + 1))
    elif mode == "blur_all_except":
        target_faces = [face for i, face in enumerate(faces) if (i + 1) not in selected_set]
        target_ids = [i + 1 for i, _ in enumerate(faces) if (i + 1) not in selected_set]
    else:  # "blur_selected"
        if not selected_set:
            target_faces = []
            target_ids = []
        else:
            target_faces = [face for i, face in enumerate(faces) if (i + 1) in selected_set]
            target_ids = [i + 1 for i, _ in enumerate(faces) if (i + 1) in selected_set]

    trackers = [FaceTracker(face, width, height) for face in target_faces]
    redacted_labels: set[str] = set()
    if target_faces:
        redacted_labels.add("FACE")

    # Initialize sensitive text / ID info trackers
    face_anchor_box = faces[0] if faces else None
    initial_card_box = find_card_region(first_frame)

    text_trackers: list[SensitiveTextTracker] = []
    if redact_sensitive_info:
        fp = _video_fingerprint(video_path)
        cached_items = _VIDEO_SENSITIVE_ENTITIES_CACHE.get(fp) if fp else None
        if cached_items:
            LOGGER.info("Reusing %d cached sensitive entities from preview for %s", len(cached_items), video_path.name)
            for item in cached_items:
                bx, by, bw, bh = item["bbox"]
                text_trackers.append(
                    SensitiveTextTracker(
                        bx,
                        by,
                        bw,
                        bh,
                        item["label"],
                        item["text"],
                        item["fake_text"],
                        first_frame,
                        anchor_box=face_anchor_box,
                        card_box=initial_card_box,
                    )
                )
                redacted_labels.add(item["label"].upper())
        else:
            try:
                from ..detection.image import ImageDetector
                from ..utils.config import ImageConfig

                img_cfg = ImageConfig()
                img_detector = ImageDetector(img_cfg)
                initial_entities = img_detector.detect_frame(first_frame)

                for ent in initial_entities:
                    if not ent.bbox:
                        continue
                    lbl = ent.label.lower()
                    txt = (ent.text or "").strip()
                    if lbl in {"face", "person"} and not txt:
                        continue
                    if txt.lower() in {
                        "ocean", "atlantic", "pacific", "indian", "arctic", "south", "north",
                        "east", "west", "africa", "asia", "europe", "america", "sea", "bay",
                        "lat", "lon", "gmt", "utc", "equator", "tropic", "meridian",
                    }:
                        continue

                    bx, by, bw, bh = int(ent.bbox.x), int(ent.bbox.y), int(ent.bbox.width), int(ent.bbox.height)
                    if bw < 8 or bh < 6:
                        continue

                    fake_txt = generate_synthetic_replacement(ent.label, txt)
                    text_trackers.append(
                        SensitiveTextTracker(
                            bx,
                            by,
                            bw,
                            bh,
                            ent.label,
                            txt,
                            fake_txt,
                            first_frame,
                            anchor_box=face_anchor_box,
                            card_box=initial_card_box,
                        )
                    )
                    redacted_labels.add(ent.label.upper())
            except Exception as exc:
                LOGGER.warning("Could not initialize text trackers: %s", exc)

    temp_dir = Path(tempfile.mkdtemp(prefix="selective_face_blur_"))
    temp_video = temp_dir / "raw_intermediate.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps if fps > 0 else 30.0, (width, height))
    if not writer.isOpened():
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Cannot open video writer for {temp_video}")

    max_end = end_frame if end_frame is not None else total_frames
    if max_end <= 0 and total_frames > 0:
        max_end = total_frames

    # Process first frame
    for face in target_faces:
        first_frame = blur_region(first_frame, face, blur_strength)
    for t_tracker in text_trackers:
        first_frame = t_tracker.apply(first_frame)

    writer.write(first_frame)
    processed_count = 1

    # Background anchor tracker if faces exist in video
    anchor_tracker = trackers[0] if trackers else (FaceTracker(face_anchor_box, width, height) if face_anchor_box else None)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        processed_count += 1
        if max_end is None or processed_count <= max_end:
            # 1. Update and blur faces
            active_face_anchor = None
            for tracker in trackers:
                tracked = tracker.update(frame, detector)
                if tracked is not None:
                    frame = blur_region(frame, tracked, blur_strength)
                    if active_face_anchor is None:
                        active_face_anchor = tracked

            # If anchor tracker is separate from target trackers, update it
            if active_face_anchor is None and anchor_tracker is not None:
                active_face_anchor = anchor_tracker.update(frame, detector)

            # 2. Update and replace sensitive text with synthetic Faker data
            curr_card = find_card_region(frame)

            group_shifts = [
                t.last_shift
                for t in text_trackers
                if abs(t.last_shift[0]) > 0.05 or abs(t.last_shift[1]) > 0.05
            ]
            if group_shifts:
                g_arr = np.array(group_shifts)
                group_dx = float(np.median(g_arr[:, 0]))
                group_dy = float(np.median(g_arr[:, 1]))
            else:
                group_dx, group_dy = 0.0, 0.0

            for t_tracker in text_trackers:
                t_bbox = t_tracker.update(
                    frame,
                    group_dx=group_dx,
                    group_dy=group_dy,
                    current_anchor=active_face_anchor,
                    curr_card_box=curr_card,
                )
                if t_bbox is not None:
                    frame = t_tracker.apply(frame)

        writer.write(frame)

    writer.release()
    cap.release()

    # Mitigate audio track if requested and present
    sanitized_audio_path: Path | None = None
    audio_entities: list[dict[str, Any]] = []
    if mitigate_audio:
        try:
            if has_audio_stream(video_path):
                LOGGER.info("Video contains audio stream. Running audio privacy mitigation...")
                temp_audio_raw = temp_dir / "audio_extracted_raw.wav"
                if extract_audio(video_path, temp_audio_raw):
                    mgr = pipeline_manager
                    if mgr is None:
                        from ..orchestration.pipeline import PipelineManager
                        from ..utils.config import get_config
                        mgr = PipelineManager(get_config())
                    audio_res = mgr.process_audio(temp_audio_raw)
                    if audio_res.mitigated_output and Path(audio_res.mitigated_output).exists():
                        sanitized_audio_path = Path(audio_res.mitigated_output)
                    for ent in audio_res.entities:
                        lbl = f"AUDIO_{ent.label.upper()}"
                        redacted_labels.add(lbl)
                        audio_entities.append({
                            "label": lbl,
                            "text": ent.text,
                            "mitigation": ent.mitigation,
                            "confidence": ent.confidence,
                        })
                    LOGGER.info(
                        "Audio mitigation complete: %d spoken entities detected.", len(audio_entities)
                    )
            else:
                LOGGER.info("No audio stream detected in video container; skipping audio mitigation.")
        except Exception as exc:
            LOGGER.warning("Audio mitigation in video failed: %s. Preserving original audio.", exc)

    # Reconstruct with audio (sanitized if available, else original)
    reconstruct_with_audio(
        temp_video,
        video_path,
        output_path,
        sanitized_audio_path=sanitized_audio_path,
    )
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "status": "success",
        "total_frames": processed_count,
        "faces_detected": len(faces),
        "faces_blurred": len(target_faces),
        "target_face_ids": target_ids,
        "sensitive_items_redacted": len(text_trackers),
        "audio_mitigated": bool(sanitized_audio_path and sanitized_audio_path.exists()),
        "audio_entities": audio_entities,
        "redacted_entities": sorted(list(redacted_labels)),
        "output_path": str(output_path),
    }
