from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from leakwatch.orchestration import PipelineManager, VideoProcessingError
from leakwatch.utils.config import AppConfig, ExplainabilityConfig, ImageConfig, LeakWatchConfig, TextConfig, VideoConfig
from leakwatch.utils.types import BoundingBox, DetectionResult, DetectedEntity, Modality


def _write_synthetic_video(path: Path, *, frame_count: int = 12, fps: int = 6, width: int = 160, height: int = 120) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Unable to create synthetic test video")

    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = ((i * 17) % 255, (i * 29) % 255, (i * 41) % 255)
        cv2.putText(frame, f"F{i:02d}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


@pytest.fixture
def video_test_config(tmp_path: Path) -> LeakWatchConfig:
    return LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(enable_spacy=False),
        image=ImageConfig(enable_ocr=False),
        video=VideoConfig(enabled=True, redaction_mode="per_frame"),
        explainability=ExplainabilityConfig(
            save_text_spans=True,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
    )


def test_frame_extraction_expected_count(tmp_path: Path, video_test_config: LeakWatchConfig) -> None:
    video_path = tmp_path / "demo.mp4"
    _write_synthetic_video(video_path, frame_count=12, fps=6)

    manager = PipelineManager(video_test_config)
    manager.video_adapter.frame_stride = 2
    manager.video_adapter.max_frames = 5

    frame_dir = tmp_path / "frames"
    frames = manager.video_adapter.extract_frames(video_path, frame_dir)

    assert len(frames) == 5
    for frame in frames:
        assert frame.exists()


def test_video_pipeline_processes_each_frame_and_reconstructs_output(tmp_path: Path, video_test_config: LeakWatchConfig) -> None:
    video_path = tmp_path / "call.mp4"
    _write_synthetic_video(video_path, frame_count=10, fps=5)

    source_cap = cv2.VideoCapture(str(video_path))
    assert source_cap.isOpened()
    source_frame_count = int(source_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_cap.release()

    manager = PipelineManager(video_test_config)
    manager.video_adapter.frame_stride = 1
    manager.video_adapter.max_frames = None

    processed_calls: list[Path] = []

    def _fake_process_image(
        frame_path: Path,
        output_path: Path | None = None,
        entities_override: list[DetectedEntity] | None = None,
    ) -> DetectionResult:
        processed_calls.append(frame_path)
        image = cv2.imread(str(frame_path))
        assert image is not None
        # Emit a visually distinct processed frame (solid red in BGR) to verify reconstruction uses adapter output.
        image[:, :] = (0, 0, 255)
        sanitized_path = (tmp_path / "artifacts" / f"{frame_path.stem}.sanitized.png")
        sanitized_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sanitized_path), image)
        return DetectionResult(
            source_path=frame_path,
            modality=Modality.IMAGE,
            entities=[],
            mitigated_output=sanitized_path,
            artifacts=[],
            runtime_metadata={"detection_mode": manager.text_detector.detection_mode},
        )

    manager.process_image = _fake_process_image  # type: ignore[method-assign]

    result = manager.process_video(video_path)

    assert len(processed_calls) == source_frame_count
    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    assert result.mitigated_output.stat().st_size > 0

    cap = cv2.VideoCapture(str(result.mitigated_output))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
    ok_late, late_frame = cap.read()
    cap.release()

    # Reconstructed output should preserve source duration (same total frame count).
    assert frame_count == source_frame_count
    assert ok and first_frame is not None
    assert ok_late and late_frame is not None

    # Allow codec compression tolerance but require output to be dominantly red.
    mean_bgr = first_frame.mean(axis=(0, 1))
    assert mean_bgr[2] > 180  # red channel
    assert mean_bgr[1] < 80   # green channel
    assert mean_bgr[0] < 80   # blue channel

    # Every output frame must pass through mitigation.
    late_mean_bgr = late_frame.mean(axis=(0, 1))
    assert late_mean_bgr[2] > 180
    assert late_mean_bgr[1] < 80
    assert late_mean_bgr[0] < 80


@pytest.mark.parametrize(
    ("frame_stride", "max_frames", "message"),
    [
        (2, None, "frame_stride=1"),
        (1, 4, "does not allow max_frames"),
    ],
)
def test_video_pipeline_rejects_incomplete_frame_coverage(
    tmp_path: Path,
    video_test_config: LeakWatchConfig,
    frame_stride: int,
    max_frames: int | None,
    message: str,
) -> None:
    video_path = tmp_path / "coverage.mp4"
    _write_synthetic_video(video_path, frame_count=6, fps=3)

    manager = PipelineManager(video_test_config)
    manager.video_adapter.frame_stride = frame_stride
    manager.video_adapter.max_frames = max_frames

    with pytest.raises(VideoProcessingError, match=message):
        manager.process_video(video_path)


def test_video_extraction_failure_records_failure_reason(tmp_path: Path, video_test_config: LeakWatchConfig) -> None:
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_bytes(b"not-a-real-video")

    manager = PipelineManager(video_test_config)

    def _raise_corrupt(*args, **kwargs):
        raise RuntimeError("corrupt input stream")

    manager.video_adapter.extract_frames = _raise_corrupt  # type: ignore[assignment]

    with pytest.raises(VideoProcessingError, match="corrupt input stream"):
        manager.process_video(video_path)

    audit_paths = sorted(tmp_path.glob("corrupt.*.audit.json"))
    assert len(audit_paths) == 1
    payload = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    assert payload.get("failure_reason")
    assert "corrupt" in payload["failure_reason"].lower()


def test_video_per_frame_temporal_fill_keeps_redaction_continuous(tmp_path: Path) -> None:
    video_path = tmp_path / "moving_subject.mp4"
    _write_synthetic_video(video_path, frame_count=8, fps=4)

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(enable_spacy=False),
        image=ImageConfig(enable_ocr=False),
        video=VideoConfig(enabled=True, redaction_mode="per_frame", carryover_frames=2),
        explainability=ExplainabilityConfig(
            save_text_spans=False,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
    )

    manager = PipelineManager(config)
    manager.video_adapter.frame_stride = 1
    manager.video_adapter.max_frames = None

    def _sparse_detect(frame_path: Path) -> list[DetectedEntity]:
        # Simulate intermittent misses: detection only on two keyframes.
        idx = int(frame_path.stem.split("_frame_")[1])
        if idx in {2, 5}:
            return [
                DetectedEntity(
                    modality=Modality.IMAGE,
                    label="ID_NUMBER",
                    confidence=0.95,
                    text="A1234567",
                    bbox=BoundingBox(x=30, y=25, width=50, height=35),
                    mitigation="replace",
                    detection_method="heuristic",
                )
            ]
        return []

    manager.image_detector.detect = _sparse_detect  # type: ignore[assignment]

    entity_counts: dict[int, int] = {}

    def _recording_process_image(
        frame_path: Path,
        output_path: Path | None = None,
        entities_override: list[DetectedEntity] | None = None,
    ) -> DetectionResult:
        idx = int(frame_path.stem.split("_frame_")[1])
        entity_counts[idx] = len(entities_override or [])
        image = cv2.imread(str(frame_path))
        assert image is not None
        sanitized_path = (tmp_path / "artifacts" / f"{frame_path.stem}.sanitized.png")
        sanitized_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sanitized_path), image)
        return DetectionResult(
            source_path=frame_path,
            modality=Modality.IMAGE,
            entities=entities_override or [],
            mitigated_output=sanitized_path,
            artifacts=[],
            runtime_metadata={"detection_mode": manager.text_detector.detection_mode},
        )

    manager.process_image = _recording_process_image  # type: ignore[method-assign]

    result = manager.process_video(video_path)

    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    # Carryover should populate neighboring missed frames.
    assert entity_counts[1] > 0
    assert entity_counts[3] > 0
    assert entity_counts[4] > 0
    assert entity_counts[6] > 0
