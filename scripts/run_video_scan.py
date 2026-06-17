"""Run the LeakWatch video privacy pipeline on my_id_test.mp4.

Usage:
    python scripts/run_video_scan.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add ffmpeg shim to PATH if present
ffmpeg_shim = PROJECT_ROOT / ".ffmpeg_shim"
if ffmpeg_shim.exists():
    os.environ["PATH"] = str(ffmpeg_shim) + os.pathsep + os.environ.get("PATH", "")
    print(f"[setup] Added ffmpeg shim to PATH: {ffmpeg_shim}")


def main() -> None:
    from leakwatch.orchestration import PipelineManager
    from leakwatch.utils.config import LeakWatchConfig

    video_path = PROJECT_ROOT / "samples" / "video" / "my_id_test.mp4"
    config_path = PROJECT_ROOT / "config" / "leakwatch.yaml"

    if not video_path.exists():
        print(f"[error] Video not found: {video_path}")
        sys.exit(1)

    print(f"[info] Loading config from: {config_path}")
    config = LeakWatchConfig.from_path(config_path)
    config.app.output_dir = Path(PROJECT_ROOT / "artifacts")

    print(f"[info] Video: {video_path}")
    print(f"[info] Output dir: {config.app.output_dir}")
    print(f"[info] Redaction mode: {config.video.redaction_mode}")
    print(f"[info] Frame stride: {config.video.frame_stride}")
    print(f"[info] Carryover frames: {config.video.carryover_frames}")
    print(f"[info] OCR enabled: {config.image.enable_ocr}")
    print(f"[info] OCR min confidence: {config.image.min_confidence}")
    print()

    manager = PipelineManager(config)

    print("=" * 70)
    print("  STARTING VIDEO PRIVACY SCAN")
    print("=" * 70)
    start_time = time.time()

    try:
        result = manager.process_video(video_path)
    except Exception as exc:
        print(f"\n[FAILED] Pipeline error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("  SCAN COMPLETE")
    print("=" * 70)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Entities detected: {len(result.entities)}")
    print(f"  Mitigated output: {result.mitigated_output}")
    print(f"  Artifacts: {len(result.artifacts)}")
    print()

    # Summarize detected entity types
    label_counts: dict[str, int] = {}
    for entity in result.entities:
        label_counts[entity.label] = label_counts.get(entity.label, 0) + 1
    if label_counts:
        print("  Entity breakdown:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")
    print()

    # Print runtime metadata
    if result.runtime_metadata:
        print("  Runtime metadata:")
        for key, value in result.runtime_metadata.items():
            print(f"    {key}: {value}")
    print()

    # Save full results JSON
    results_json_path = config.app.output_dir / "my_id_test_results.json"
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_data = result.model_dump()
    results_json_path.write_text(
        json.dumps(results_data, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Full results saved to: {results_json_path}")


if __name__ == "__main__":
    main()
