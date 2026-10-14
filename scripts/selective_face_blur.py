"""Selective Face Blur -- blur one or more faces in a video.

Interactive workflow:
  1. The first frame is displayed with numbered face bounding boxes.
  2. Click on the face(s) you want blurred -> the script tracks those people.
  3. Gaussian blur is applied to each tracked face from frame 1 to --end-frame.
  4. The output video preserves audio via ffmpeg.

Multi-face & invert modes:
  --face 1,3       Blur faces #1 and #3 (comma-separated, 1-indexed).
  --blur-all-except 2   Blur EVERY detected face EXCEPT #2.

Usage:
    python scripts/selective_face_blur.py --video samples/video/Video1.mp4
    python scripts/selective_face_blur.py --video samples/video/Video1.mp4 --face 1,3
    python scripts/selective_face_blur.py --video samples/video/Video1.mp4 --blur-all-except 2
    python scripts/selective_face_blur.py --video samples/video/Video1.mp4 --blur-strength 71 --output out.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core tracking engine imported from leakwatch.video.tracker
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leakwatch.video.tracker import (
    FaceTracker,
    blur_region,
    create_face_detector,
    detect_faces,
    ensure_yunet_model as _ensure_yunet_model,
    find_ffmpeg as _find_ffmpeg,
    reconstruct_with_audio,
)


# ---------------------------------------------------------------------------
# Interactive face selection
# ---------------------------------------------------------------------------

class FaceSelector:
    """Display a frame with numbered face boxes; supports multi-face selection.

    Click faces to toggle them on/off (highlighted in red).
    Press ENTER to confirm, or 'q' to cancel.
    """

    def __init__(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> None:
        self.frame = frame.copy()
        self.faces = faces
        self.selected_indices: set[int] = set()
        self._window_name = "Select face(s) to blur (click to toggle, ENTER to confirm, 'q' to cancel)"

    def _find_face_at(self, x: int, y: int) -> int | None:
        """Return index of face at (x, y), or nearest face if click is close."""
        for i, (fx, fy, fw, fh) in enumerate(self.faces):
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                return i
        # Fallback: nearest face within 80px
        if self.faces:
            best_i, best_dist = 0, float("inf")
            for i, (fx, fy, fw, fh) in enumerate(self.faces):
                cx, cy = fx + fw // 2, fy + fh // 2
                dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            if best_dist < 80:
                return best_i
        return None

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_face_at(x, y)
            if idx is not None:
                # Toggle selection
                if idx in self.selected_indices:
                    self.selected_indices.discard(idx)
                else:
                    self.selected_indices.add(idx)

    def _draw(self) -> np.ndarray:
        """Render the frame with face boxes and selection state."""
        display = self.frame.copy()
        for i, (x, y, w, h) in enumerate(self.faces):
            selected = i in self.selected_indices
            color = (0, 0, 255) if selected else (0, 255, 0)
            thickness = 3 if selected else 2
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            label = f"#{i + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            lbl_thick = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, lbl_thick)
            cv2.rectangle(display, (x, y - th - 10), (x + tw + 6, y), color, -1)
            cv2.putText(display, label, (x + 3, y - 5), font, font_scale, (255, 255, 255) if selected else (0, 0, 0), lbl_thick)

        # Instructions
        instr = "CLICK faces to toggle (red=selected). ENTER=confirm, 'q'=cancel."
        cv2.putText(display, instr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        if self.selected_indices:
            sel_str = ", ".join(f"#{i+1}" for i in sorted(self.selected_indices))
            cv2.putText(
                display,
                f"Selected: {sel_str}",
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
        return display

    def run(self) -> List[Tuple[int, int, int, int]] | None:
        """Show the GUI and return list of selected face bboxes, or None if cancelled."""
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        display = self._draw()
        cv2.resizeWindow(self._window_name, min(display.shape[1], 900), min(display.shape[0], 900))
        cv2.setMouseCallback(self._window_name, self._mouse_callback)
        cv2.imshow(self._window_name, display)

        prev_selection = set()
        while True:
            key = cv2.waitKey(50) & 0xFF
            # Refresh display if selection changed
            if self.selected_indices != prev_selection:
                prev_selection = set(self.selected_indices)
                cv2.imshow(self._window_name, self._draw())
            # ENTER (13) or SPACE (32) to confirm
            if key in (13, 32):
                if self.selected_indices:
                    result = [self.faces[i] for i in sorted(self.selected_indices)]
                    cv2.destroyAllWindows()
                    return result
            if key == ord("q"):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return None





# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _parse_face_list(value: str) -> List[int]:
    """Parse a comma-separated list of 1-indexed face numbers into 0-indexed indices."""
    indices = []
    for part in value.split(","):
        part = part.strip()
        if part:
            indices.append(int(part) - 1)  # convert to 0-indexed
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selectively blur one or more faces in a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", required=True, type=str, help="Path to input video file")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last frame to blur (1-indexed). Default: blur through the entire video.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path. Default: <input>.face_blurred.mp4")
    parser.add_argument("--blur-strength", type=int, default=51,
                        help="Gaussian blur kernel size (odd number, higher = stronger). Default: 51")
    parser.add_argument("--face", type=str, default=None,
                        help="Face number(s) to blur (1-indexed, comma-separated). "
                             "Examples: --face 1  or  --face 1,3,5. "
                             "Skips the GUI. Run once without this flag to see the numbered preview.")
    parser.add_argument("--blur-all-except", type=str, default=None, dest="blur_all_except",
                        help="Blur ALL detected faces EXCEPT the specified one(s) (1-indexed, comma-separated). "
                             "Example: --blur-all-except 2  keeps face #2 clear and blurs everyone else.")
    args = parser.parse_args()

    if args.face is not None and args.blur_all_except is not None:
        print("[error] Cannot use both --face and --blur-all-except at the same time.")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[error] Video not found: {video_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else video_path.with_name(
        f"{video_path.stem}.face_blurred.mp4"
    )

    # --- Step 1: Read first frame and detect faces ---
    print(f"[info] Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[error] Cannot open video.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    end_frame = args.end_frame if args.end_frame is not None else total_frames
    end_frame = min(end_frame, total_frames)
    print(f"[info] Will blur from frame 1 to frame {end_frame}")

    ret, first_frame = cap.read()
    if not ret:
        print("[error] Cannot read first frame.")
        sys.exit(1)

    detector = create_face_detector(width, height)
    faces = detect_faces(first_frame, detector)
    print(f"[info] Detected {len(faces)} face(s) in the first frame.")

    if not faces:
        print("[error] No faces detected in the first frame. Exiting.")
        sys.exit(1)

    # Save a preview image with numbered face boxes (always useful)
    preview = first_frame.copy()
    for i, (fx, fy, fw, fh) in enumerate(faces):
        color = (0, 255, 0)
        cv2.rectangle(preview, (fx, fy), (fx + fw, fy + fh), color, 2)
        label = f"#{i + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(preview, (fx, fy - th - 10), (fx + tw + 6, fy), color, -1)
        cv2.putText(preview, label, (fx + 3, fy - 5), font, 0.7, (0, 0, 0), 2)
    preview_path = output_path.parent / f"{video_path.stem}.faces_preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    print(f"[info] Face preview saved to: {preview_path}")
    for i, (fx, fy, fw, fh) in enumerate(faces):
        print(f"       Face #{i + 1}: position=({fx}, {fy}), size={fw}x{fh}")

    # --- Step 2: Determine which faces to blur ---
    selected_faces: List[Tuple[int, int, int, int]] = []

    if args.blur_all_except is not None:
        # --- Invert mode: blur ALL except the listed faces ---
        except_indices = _parse_face_list(args.blur_all_except)
        for idx in except_indices:
            if idx < 0 or idx >= len(faces):
                print(f"[error] --blur-all-except index {idx + 1} is out of range. Valid: 1-{len(faces)}")
                sys.exit(1)
        keep_labels = ", ".join(f"#{i+1}" for i in except_indices)
        selected_faces = [f for i, f in enumerate(faces) if i not in except_indices]
        blur_labels = ", ".join(f"#{i+1}" for i in range(len(faces)) if i not in except_indices)
        print(f"[info] Keeping face(s) {keep_labels} clear; blurring face(s) {blur_labels}")

    elif args.face is not None:
        # --- Non-interactive: use --face N or --face 1,3,5 ---
        face_indices = _parse_face_list(args.face)
        for idx in face_indices:
            if idx < 0 or idx >= len(faces):
                print(f"[error] --face index {idx + 1} is out of range. Valid: 1-{len(faces)}")
                sys.exit(1)
        selected_faces = [faces[i] for i in face_indices]
        labels = ", ".join(f"#{i+1}" for i in face_indices)
        print(f"[info] Using face(s) {labels} (from --face argument)")

    else:
        # --- Interactive: GUI multi-face selection ---
        try:
            selector = FaceSelector(first_frame, faces)
            result = selector.run()
            if result is None or len(result) == 0:
                print("[info] Selection cancelled.")
                sys.exit(0)
            selected_faces = result
        except cv2.error:
            print("[error] GUI window could not be opened (headless environment?).")
            print(f"        Use --face N to select a face by number (see preview: {preview_path})")
            sys.exit(1)

    n = len(selected_faces)
    print(f"[info] Blurring {n} face(s):")
    for i, (sx, sy, sw, sh) in enumerate(selected_faces):
        print(f"       [{i+1}/{n}] ({sx}, {sy}, {sw}x{sh})")

    # --- Step 3: Track and blur ---
    trackers = [FaceTracker(face, width, height) for face in selected_faces]
    blur_strength = args.blur_strength

    temp_dir = Path(tempfile.mkdtemp(prefix="face_blur_"))
    temp_video = temp_dir / "raw_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("[error] Cannot create video writer.")
        sys.exit(1)

    # Process first frame (already read)
    for face in selected_faces:
        first_frame = blur_region(first_frame, face, blur_strength)
    writer.write(first_frame)
    processed = 1

    print(f"[info] Processing frames... (blurring frames 1-{end_frame})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1

        if processed <= end_frame:
            for tracker in trackers:
                tracked = tracker.update(frame, detector)
                if tracked is not None:
                    frame = blur_region(frame, tracked, blur_strength)

        writer.write(frame)

        if processed % 50 == 0 or processed == total_frames:
            pct = int(processed / total_frames * 100)
            active_count = sum(1 for t in trackers if t.active)
            print(f"  [{pct:3d}%] Frame {processed}/{total_frames} ({active_count}/{n} faces tracked)", end="\r")

    writer.release()
    cap.release()
    print(f"\n[info] Processed {processed} frames.")

    # --- Step 4: Reconstruct with audio ---
    print("[info] Reconstructing output with audio...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruct_with_audio(temp_video, video_path, output_path)

    # Cleanup temp
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    print(f"[done] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
