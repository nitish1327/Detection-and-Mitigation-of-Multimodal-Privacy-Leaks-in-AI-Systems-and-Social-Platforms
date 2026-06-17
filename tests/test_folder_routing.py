from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import AppConfig, AudioConfig, ImageConfig, LeakWatchConfig, TextConfig, VideoConfig
from leakwatch.utils.types import DetectionResult, Modality


def test_process_folder_routes_audio_and_video_files(tmp_path: Path) -> None:
    scan_dir = tmp_path / "inputs"
    scan_dir.mkdir(parents=True, exist_ok=True)

    text_path = scan_dir / "note.txt"
    text_path.write_text("hello", encoding="utf-8")

    image_path = scan_dir / "card.png"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))

    audio_path = scan_dir / "call.m4a"
    audio_path.write_bytes(b"fake-audio")

    video_path = scan_dir / "clip.mp4"
    video_path.write_bytes(b"fake-video")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(enable_spacy=False),
        image=ImageConfig(enable_ocr=False),
        audio=AudioConfig(enabled=True),
        video=VideoConfig(enabled=True),
    )
    manager = PipelineManager(config)

    routed_suffixes: list[str] = []

    def _result(path: Path, modality: Modality) -> DetectionResult:
        return DetectionResult(source_path=path, modality=modality, entities=[], artifacts=[])

    manager.process_text = lambda p: (routed_suffixes.append(Path(p).suffix.lower()) or _result(Path(p), Modality.TEXT))  # type: ignore[method-assign]
    manager.process_image = lambda p: (routed_suffixes.append(Path(p).suffix.lower()) or _result(Path(p), Modality.IMAGE))  # type: ignore[method-assign]
    manager.process_audio = lambda p: (routed_suffixes.append(Path(p).suffix.lower()) or _result(Path(p), Modality.AUDIO))  # type: ignore[method-assign]
    manager.process_video = lambda p: (routed_suffixes.append(Path(p).suffix.lower()) or _result(Path(p), Modality.VIDEO))  # type: ignore[method-assign]

    results = manager.process_folder(scan_dir, recursive=False)

    assert len(results) == 4
    assert sorted(routed_suffixes) == [".m4a", ".mp4", ".png", ".txt"]
