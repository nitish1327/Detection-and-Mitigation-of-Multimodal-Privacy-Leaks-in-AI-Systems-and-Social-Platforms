from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import (
    AppConfig,
    AudioConfig,
    ExplainabilityConfig,
    ImageConfig,
    LeakWatchConfig,
    RegexEntityConfig,
    TextConfig,
)


def _write_sine_wave(path: Path, *, frequency: float = 440.0, seconds: float = 2.0, sample_rate: int = 16000) -> None:
    total_frames = int(seconds * sample_rate)
    amplitude = 9000
    frames = bytearray()
    for i in range(total_frames):
        value = int(amplitude * math.sin(2 * math.pi * frequency * (i / sample_rate)))
        frames += value.to_bytes(2, byteorder="little", signed=True)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


def test_audio_pipeline_beeps_sensitive_segments(tmp_path: Path) -> None:
    audio_path = tmp_path / "call.wav"
    transcript_path = tmp_path / "call.txt"
    _write_sine_wave(audio_path)

    transcript = "Hello, my passport number is A1234567 and my phone is 425-555-1212."
    transcript_path.write_text(transcript, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=False,
            regex_entities=[
                RegexEntityConfig(name="passport", pattern=r"\b[A-Z]\d{7}\b", action="mask"),
                RegexEntityConfig(name="phone", pattern=r"\b\d{3}-\d{3}-\d{4}\b", action="mask"),
            ],
        ),
        image=ImageConfig(enable_ocr=False),
        explainability=ExplainabilityConfig(
            save_text_spans=True,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
        audio=AudioConfig(enabled=True),
    )

    manager = PipelineManager(config)
    result = manager.process_audio(audio_path)

    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    assert result.mitigated_output.suffix == ".wav"

    sanitized_text_path = tmp_path / "artifacts" / "call.audio.sanitized.txt"
    assert sanitized_text_path.exists()
    assert "A1234567" not in sanitized_text_path.read_text(encoding="utf-8")
    assert "425-555-1212" not in sanitized_text_path.read_text(encoding="utf-8")

    beep_map_path = tmp_path / "artifacts" / "call.audio.beeps.json"
    assert beep_map_path.exists()
    beep_map_text = beep_map_path.read_text(encoding="utf-8")
    assert "start_sec" in beep_map_text
    assert "end_sec" in beep_map_text

    with wave.open(str(audio_path), "rb") as original_wav:
        original_frames = original_wav.readframes(original_wav.getnframes())
    with wave.open(str(result.mitigated_output), "rb") as sanitized_wav:
        sanitized_frames = sanitized_wav.readframes(sanitized_wav.getnframes())

    assert original_frames != sanitized_frames


def test_audio_pipeline_requires_companion_transcript(tmp_path: Path) -> None:
    audio_path = tmp_path / "missing_transcript.wav"
    _write_sine_wave(audio_path)

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(enable_spacy=False),
        image=ImageConfig(enable_ocr=False),
        audio=AudioConfig(enabled=True),
    )

    manager = PipelineManager(config)
    with pytest.raises(RuntimeError, match="No transcript found"):
        manager.process_audio(audio_path)
