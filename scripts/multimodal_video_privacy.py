"""Multimodal Audio + Video Privacy Mitigation Pipeline.

Applies joint privacy mitigation to any input video:
1. Audio Track Mitigation:
   - Extracts audio track via FFmpeg.
   - Transcribes speech with OpenAI Whisper (with word-level timestamps).
   - Detects spoken PII (names, phone numbers, numeric IDs, emails, locations).
   - Generates tone-masked beeps over sensitive speech intervals (*.sanitized.wav).
2. Video Track Mitigation:
   - Detects faces with OpenCV YuNet ONNX deep learning detector.
   - Tracks target faces across all frames with spatial velocity extrapolation.
   - Applies Gaussian blur to faces and detects/rewrites sensitive document text.
3. Multimodal Assembly:
   - Muxes the sanitized video stream with the sanitized tone-masked audio stream.
   - Exports the final privacy-preserving MP4 with H.264 video and AAC audio.
   - Saves a complete unified multimodal audit trail (JSON).

Usage:
    python scripts/multimodal_video_privacy.py --video samples/video/Video2.mp4
    python scripts/multimodal_video_privacy.py --video path/to/any_video.mp4 --face 1
    python scripts/multimodal_video_privacy.py --video path/to/any_video.mp4 --mode blur_all
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leakwatch.adapters import VideoAdapter
from leakwatch.orchestration.pipeline import PipelineManager
from leakwatch.utils.config import get_config
from leakwatch.video.tracker import (
    find_ffmpeg,
    generate_video_face_preview,
    process_selective_video,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("leakwatch.multimodal")


def has_audio_stream(video_path: Path, ffmpeg_bin: str) -> bool:
    """Check if the video container includes an audio stream."""
    cmd = [ffmpeg_bin, "-i", str(video_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stderr.splitlines():
        if "Stream #" in line and "Audio:" in line:
            return True
    return False


def extract_audio(video_path: Path, output_wav: Path, ffmpeg_bin: str) -> bool:
    """Extract 16kHz mono PCM WAV for Whisper transcription and audio mitigation."""
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0


def mux_video_and_audio(
    sanitized_video_path: Path,
    sanitized_audio_path: Path,
    final_output_path: Path,
    ffmpeg_bin: str,
) -> Path:
    """Combine sanitized video stream with sanitized tone-masked audio stream."""
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(sanitized_video_path),
        "-i", str(sanitized_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(final_output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.warning("FFmpeg muxing failed: %s. Falling back to video stream.", result.stderr[:400])
        shutil.copy2(sanitized_video_path, final_output_path)
    return final_output_path


def run_multimodal_privacy(
    video_path: Path,
    output_path: Path | None = None,
    face_ids: list[int] | None = None,
    mode: str = "blur_all",
    blur_strength: int = 51,
    end_frame: int | None = None,
) -> dict[str, Any]:
    """Execute complete multimodal (audio + video) privacy pipeline."""
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    ffmpeg_bin = find_ffmpeg()
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg executable not found. Install ffmpeg or imageio-ffmpeg.")

    output_dir = PROJECT_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = output_dir / f"{video_path.stem}.multimodal_sanitized.mp4"
    else:
        output_path = output_path.resolve()

    print("=" * 70)
    print("  LEAKWATCH MULTIMODAL PRIVACY ENGINE (AUDIO + VIDEO)")
    print("=" * 70)
    print(f"  Input Video   : {video_path}")
    print(f"  Target Output : {output_path}")
    print(f"  FFmpeg Binary : {ffmpeg_bin}")
    print("=" * 70)

    start_time = time.time()
    config = get_config()
    pipeline = PipelineManager(config)

    # ---------------------------------------------------------
    # STAGE 1: Audio Privacy Detection & Tone Mitigation
    # ---------------------------------------------------------
    print("\n[Stage 1/3] Processing Audio Stream...")
    video_has_audio = has_audio_stream(video_path, ffmpeg_bin)
    audio_entities: list[dict[str, Any]] = []
    sanitized_audio_wav: Path | None = None

    if video_has_audio:
        temp_audio_raw = output_dir / f"{video_path.stem}_extracted_raw.wav"
        print(f"  -> Extracting audio track to {temp_audio_raw.name}...")
        extracted = extract_audio(video_path, temp_audio_raw, ffmpeg_bin)

        if extracted:
            print("  -> Transcribing speech with Whisper & detecting spoken PII...")
            audio_result = pipeline.process_audio(temp_audio_raw)
            sanitized_audio_wav = Path(audio_result.mitigated_output) if audio_result.mitigated_output else None

            for e in audio_result.entities:
                audio_entities.append({
                    "label": e.label,
                    "text": e.text,
                    "mitigation": e.mitigation,
                    "replacement_or_explanation": e.explanation,
                    "confidence": e.confidence,
                })
                print(f"     * Spoken PII: [{e.label}] '{e.text}' -> {e.mitigation} ({e.explanation})")

            print(f"  -> Sanitized audio emitted: {sanitized_audio_wav}")
        else:
            print("  -> Warning: Failed to extract audio stream.")
    else:
        print("  -> No audio stream detected in video container.")

    # ---------------------------------------------------------
    # STAGE 2: Video Face & Visual Privacy Mitigation
    # ---------------------------------------------------------
    print("\n[Stage 2/3] Processing Video Stream (Faces & Visual PII)...")
    preview_info = generate_video_face_preview(video_path)
    total_detected_faces = preview_info.get("face_count", 0)
    print(f"  -> Detected {total_detected_faces} face(s) in reference frame.")

    # Determine face selection
    target_face_ids = face_ids
    effective_mode = mode
    if target_face_ids is None or len(target_face_ids) == 0:
        if total_detected_faces > 0:
            target_face_ids = list(range(1, total_detected_faces + 1))
            effective_mode = "blur_all"
        else:
            target_face_ids = []

    print(f"  -> Mode: '{effective_mode}', Target Face IDs: {target_face_ids}, Blur Strength: {blur_strength}")
    print("  -> Tracking faces & applying Gaussian privacy blur across frames...")

    temp_video_sanitized = output_dir / f"{video_path.stem}_visual_sanitized.mp4"
    video_result = process_selective_video(
        source_video_path=video_path,
        output_path=temp_video_sanitized,
        selected_face_ids=target_face_ids,
        mode=effective_mode,
        blur_strength=blur_strength,
        redact_sensitive_info=True,
        end_frame=end_frame,
    )
    print(f"  -> Video visual frames processed: {video_result.get('total_frames')} frames.")

    # ---------------------------------------------------------
    # STAGE 3: Multimodal Assembly & Muxing
    # ---------------------------------------------------------
    print("\n[Stage 3/3] Assembling Multimodal Output...")
    if sanitized_audio_wav and sanitized_audio_wav.exists():
        print("  -> Muxing sanitized video track + tone-masked sanitized audio track...")
        final_video = mux_video_and_audio(
            sanitized_video_path=temp_video_sanitized,
            sanitized_audio_path=sanitized_audio_wav,
            final_output_path=output_path,
            ffmpeg_bin=ffmpeg_bin,
        )
    else:
        print("  -> Audio was silent or absent; preserving video output.")
        if temp_video_sanitized.resolve() != output_path.resolve():
            shutil.copy2(temp_video_sanitized, output_path)
        final_video = output_path

    elapsed = time.time() - start_time

    # ---------------------------------------------------------
    # Audit Trail Generation
    # ---------------------------------------------------------
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_video": str(video_path),
        "sanitized_video": str(final_video),
        "elapsed_seconds": round(elapsed, 2),
        "audio_stream": {
            "present": video_has_audio,
            "sanitized_audio_wav": str(sanitized_audio_wav) if sanitized_audio_wav else None,
            "entities_detected": audio_entities,
        },
        "video_stream": {
            "faces_detected": video_result.get("faces_detected", 0),
            "faces_blurred": video_result.get("faces_blurred", 0),
            "target_face_ids": video_result.get("target_face_ids", []),
            "total_frames": video_result.get("total_frames", 0),
            "redacted_entities": video_result.get("redacted_entities", []),
        },
    }

    audit_path = output_dir / f"{video_path.stem}.multimodal_audit.json"
    audit_path.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  MULTIMODAL PRIVACY MITIGATION COMPLETE!")
    print("=" * 70)
    print(f"  Sanitized Video : {final_video}")
    print(f"  Audit Report    : {audit_path}")
    print(f"  Time Elapsed    : {elapsed:.2f}s")
    print(f"  Spoken PII Beeped: {len(audio_entities)} item(s)")
    print(f"  Faces Blurred   : {video_result.get('faces_blurred', 0)} face(s)")
    print("=" * 70 + "\n")

    return audit_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeakWatch Multimodal Audio + Video Privacy Mitigation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default="samples/video/Video2.mp4",
        help="Path to input video file (defaults to samples/video/Video2.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output video path (defaults to artifacts/<name>.multimodal_sanitized.mp4)",
    )
    parser.add_argument(
        "--face",
        type=str,
        default=None,
        help="Comma-separated face IDs to blur (e.g. '1' or '1,2'). Defaults to blurring all detected faces.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["blur_all", "blur_selected", "blur_all_except"],
        default="blur_all",
        help="Face blur mode (default: blur_all)",
    )
    parser.add_argument(
        "--blur-strength",
        type=int,
        default=51,
        help="Gaussian blur strength (odd integer, default: 51)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional: last frame to process for quick preview testing.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()

    face_ids = None
    if args.face:
        face_ids = [int(x.strip()) for x in args.face.split(",") if x.strip()]

    output_path = Path(args.output) if args.output else None

    run_multimodal_privacy(
        video_path=video_path,
        output_path=output_path,
        face_ids=face_ids,
        mode=args.mode,
        blur_strength=args.blur_strength,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
