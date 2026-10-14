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
    score_threshold: float = 0.4,
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
        search_radius = 85 + self.lost_count * 3

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
            return best_box

        # Face was not detected on this frame
        self.lost_count += 1

        if near_edge:
            if self.lost_count <= self.MAX_EDGE_COAST:
                step_x = self.vx
                step_y = self.vy
                # Ensure velocity continues pointing outward
                if (lx + lw) >= self.frame_width - margin and step_x < 5.0:
                    step_x = max(step_x, 10.0)
                elif lx <= margin and step_x > -5.0:
                    step_x = min(step_x, -10.0)
                if (ly + lh) >= self.frame_height - margin and step_y < 5.0:
                    step_y = max(step_y, 10.0)
                elif ly <= margin and step_y > -5.0:
                    step_y = min(step_y, -10.0)

                cur_x = int(lx + step_x)
                cur_y = int(ly + step_y)
                extrapolated = (cur_x, cur_y, lw, lh)
                self.last_bbox = extrapolated

                # If completely outside frame boundaries, deactivate
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

        # Lost in the interior of the frame (e.g. occlusion)
        if self.lost_count > self.MAX_LOST_ANCHOR:
            self.active = False
            return None

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
    rh, rw = roi.shape[:2]

    # Background color detection from border pixels
    border = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    bg_color = np.median(border, axis=0).astype(np.uint8).tolist()
    is_light_bg = float(np.mean(bg_color)) > 127.0
    text_color = (25, 25, 25) if is_light_bg else (235, 235, 235)

    # Inpaint original text
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if is_light_bg:
        _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, inpaint_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)
    erased = cv2.inpaint(roi, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Feathered background fill in center
    margin = max(2, min(rh, rw) // 12)
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
    target_h = int(rh * 0.70)
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

    frame[y1:y2, x1:x2] = erased
    return frame


class SensitiveTextTracker:
    """Tracks a sensitive text region across frames and applies synthetic Faker text."""

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
    ):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.label = label
        self.orig_text = orig_text
        self.fake_text = fake_text
        self.lost_frames = 0

        fh, fw = initial_frame.shape[:2]
        cx1 = max(0, self.x)
        cy1 = max(0, self.y)
        cx2 = min(fw, self.x + self.w)
        cy2 = min(fh, self.y + self.h)
        self.template = (
            initial_frame[cy1:cy2, cx1:cx2].copy()
            if (cx2 > cx1 and cy2 > cy1)
            else None
        )

    def update(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        fh, fw = frame.shape[:2]
        if self.template is None or self.template.size == 0:
            return (self.x, self.y, self.w, self.h)

        search_pad = 35
        sx1 = max(0, self.x - search_pad)
        sy1 = max(0, self.y - search_pad)
        sx2 = min(fw, self.x + self.w + search_pad)
        sy2 = min(fh, self.y + self.h + search_pad)

        th, tw = self.template.shape[:2]
        if (sx2 - sx1) < tw or (sy2 - sy1) < th:
            return (self.x, self.y, self.w, self.h)

        search_roi = frame[sy1:sy2, sx1:sx2]
        res = cv2.matchTemplate(search_roi, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.45:
            self.x = sx1 + max_loc[0]
            self.y = sy1 + max_loc[1]
            self.lost_frames = 0
        else:
            self.lost_frames += 1

        if self.lost_frames > 45:
            return None

        return (self.x, self.y, self.w, self.h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return render_synthetic_text(
            frame, self.x, self.y, self.w, self.h, self.orig_text, self.fake_text
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


def reconstruct_with_audio(
    raw_video: Path,
    source_video: Path,
    output_path: Path,
) -> Path:
    """Transcode raw_video to H.264 and mux audio from source_video using ffmpeg."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        LOGGER.warning("ffmpeg not found -- output will lack audio and use raw container.")
        if raw_video.resolve() != output_path.resolve():
            shutil.copy2(raw_video, output_path)
            raw_video.unlink(missing_ok=True)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-i", str(raw_video),
        "-i", str(source_video),
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
    text_trackers: list[SensitiveTextTracker] = []
    if redact_sensitive_info:
        fp = _video_fingerprint(video_path)
        cached_items = _VIDEO_SENSITIVE_ENTITIES_CACHE.get(fp) if fp else None
        if cached_items:
            LOGGER.info("Reusing %d cached sensitive entities from preview for %s", len(cached_items), video_path.name)
            for item in cached_items:
                bx, by, bw, bh = item["bbox"]
                text_trackers.append(
                    SensitiveTextTracker(bx, by, bw, bh, item["label"], item["text"], item["fake_text"], first_frame)
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
                        SensitiveTextTracker(bx, by, bw, bh, ent.label, txt, fake_txt, first_frame)
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

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        processed_count += 1
        if max_end is None or processed_count <= max_end:
            # 1. Update and blur faces
            for tracker in trackers:
                tracked = tracker.update(frame, detector)
                if tracked is not None:
                    frame = blur_region(frame, tracked, blur_strength)
            # 2. Update and replace sensitive text with synthetic Faker data
            for t_tracker in text_trackers:
                t_bbox = t_tracker.update(frame)
                if t_bbox is not None:
                    frame = t_tracker.apply(frame)

        writer.write(frame)

    writer.release()
    cap.release()

    # Reconstruct with audio
    reconstruct_with_audio(temp_video, video_path, output_path)
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "status": "success",
        "total_frames": processed_count,
        "faces_detected": len(faces),
        "faces_blurred": len(target_faces),
        "target_face_ids": target_ids,
        "sensitive_items_redacted": len(text_trackers),
        "redacted_entities": sorted(list(redacted_labels)),
        "output_path": str(output_path),
    }
