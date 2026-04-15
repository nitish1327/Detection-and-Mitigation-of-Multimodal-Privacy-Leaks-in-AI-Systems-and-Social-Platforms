from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import AppConfig, ExplainabilityConfig, ImageConfig, LeakWatchConfig, ModalityToggle, TextConfig
from leakwatch.utils.types import DetectionResult, Modality


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
        video=ModalityToggle(enabled=True),
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

    manager = PipelineManager(video_test_config)
    manager.video_adapter.frame_stride = 1
    manager.video_adapter.max_frames = 4

    processed_calls: list[Path] = []

    def _fake_process_image(frame_path: Path) -> DetectionResult:
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

    assert len(processed_calls) == 4
    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    assert result.mitigated_output.stat().st_size > 0

    cap = cv2.VideoCapture(str(result.mitigated_output))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = cap.read()
    cap.release()
    assert frame_count > 0
    assert ok and first_frame is not None

    # Allow codec compression tolerance but require output to be dominantly red.
    mean_bgr = first_frame.mean(axis=(0, 1))
    assert mean_bgr[2] > 180  # red channel
    assert mean_bgr[1] < 80   # green channel
    assert mean_bgr[0] < 80   # blue channel


def test_video_extraction_failure_records_failure_reason(tmp_path: Path, video_test_config: LeakWatchConfig) -> None:
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_bytes(b"not-a-real-video")

    manager = PipelineManager(video_test_config)

    def _raise_corrupt(*args, **kwargs):
        raise RuntimeError("corrupt input stream")

    manager.video_adapter.extract_frames = _raise_corrupt  # type: ignore[assignment]

    result = manager.process_video(video_path)

    assert result.mitigated_output is None
    assert result.audit_log is not None
    payload = json.loads(result.audit_log.read_text(encoding="utf-8"))
    assert payload.get("failure_reason")
    assert "corrupt" in payload["failure_reason"].lower()
