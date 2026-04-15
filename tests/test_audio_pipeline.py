from __future__ import annotations

import json
import math
import subprocess
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from leakwatch.adapters import AudioAdapter
from leakwatch.explainability.audit import record_audit
from leakwatch.mitigation.audio import AudioMitigator
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
from leakwatch.utils.types import DetectionResult, DetectedEntity, Modality, Span


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


def _write_u8_wave(path: Path, *, seconds: float = 1.0, sample_rate: int = 8000) -> None:
    total_frames = int(seconds * sample_rate)
    frames = bytes([128] * total_frames)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
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
        audio=AudioConfig(enabled=True, enable_transcription=False),
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
        audio=AudioConfig(enabled=True, enable_transcription=False),
    )

    manager = PipelineManager(config)
    with pytest.raises(RuntimeError, match="No transcript found"):
        manager.process_audio(audio_path)


def test_audio_adapter_transcribes_when_no_companion_transcript(tmp_path: Path) -> None:
    audio_path = tmp_path / "phone_recording.m4a"
    audio_path.write_bytes(b"placeholder")

    adapter = AudioAdapter(
        transcript_extension=".txt",
        enable_transcription=True,
        transcriber=lambda _: "call me at 425-555-1212",
    )

    text = adapter.to_text(audio_path)
    assert text == "call me at 425-555-1212"


def test_audio_window_padding_logic() -> None:
    transcript = "x" * 100
    entity = DetectedEntity(
        modality=Modality.AUDIO,
        label="PHONE",
        confidence=0.95,
        span=Span(start=10, end=20),
        mitigation="mask",
        detection_method="regex",
    )
    mitigator = AudioMitigator(AudioConfig(enabled=True, padding_ms=150, merge_gap_ms=0, min_beep_ms=0))

    windows = mitigator._ranges_from_entities(transcript, [entity], duration_seconds=10.0)

    assert len(windows) == 1
    assert windows[0]["start_sec"] == pytest.approx(0.85, abs=1e-6)
    assert windows[0]["end_sec"] == pytest.approx(2.15, abs=1e-6)


def test_audio_window_merge_logic() -> None:
    transcript = "x" * 100
    entities = [
        DetectedEntity(
            modality=Modality.AUDIO,
            label="PHONE",
            confidence=0.95,
            span=Span(start=10, end=20),
            mitigation="mask",
            detection_method="regex",
        ),
        DetectedEntity(
            modality=Modality.AUDIO,
            label="ID_NUMBER",
            confidence=0.95,
            span=Span(start=22, end=30),
            mitigation="mask",
            detection_method="regex",
        ),
    ]
    mitigator = AudioMitigator(AudioConfig(enabled=True, padding_ms=0, merge_gap_ms=300, min_beep_ms=0))

    windows = mitigator._ranges_from_entities(transcript, entities, duration_seconds=10.0)

    assert len(windows) == 1
    assert windows[0]["start_sec"] == pytest.approx(1.0, abs=1e-6)
    assert windows[0]["end_sec"] == pytest.approx(3.0, abs=1e-6)
    assert windows[0]["entity_indexes"] == [0, 1]


def test_low_confidence_audio_mapping_sets_null_timeline_and_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    entity = DetectedEntity(
        modality=Modality.AUDIO,
        label="PHONE",
        confidence=0.3,
        text="425-555-1212",
        span=Span(start=0, end=12),
        mitigation="mask",
        detection_method="regex",
        explanation="425-555-1212",
    )
    mitigator = AudioMitigator(
        AudioConfig(enabled=True, mapping_confidence_threshold=0.8, padding_ms=150, merge_gap_ms=300)
    )

    caplog.set_level("WARNING")
    windows = mitigator._ranges_from_entities("425-555-1212", [entity], duration_seconds=2.0)
    assert windows == []
    assert "could not be confidently mapped" in caplog.text

    beep_map_path = tmp_path / "beeps.audio.beeps.json"
    beep_map_path.write_text("[]", encoding="utf-8")
    result = DetectionResult(
        source_path=tmp_path / "call.wav",
        modality=Modality.AUDIO,
        entities=[entity],
        artifacts=[beep_map_path],
    )
    audit_path = record_audit(result, ExplainabilityConfig(audit_log_path=tmp_path / "audit.log"))
    payload = audit_path.read_text(encoding="utf-8")
    assert '"timeline_window": null' in payload


def test_audio_timeline_maps_correctly_with_non_insertion_entity_order(tmp_path: Path) -> None:
    # Beep map windows are keyed by entity_ids from detection order.
    beep_map = [
        {"label": "PHONE", "start_sec": 1.0, "end_sec": 1.5, "entity_ids": ["e1"]},
        {"label": "EMAIL", "start_sec": 2.0, "end_sec": 2.6, "entity_ids": ["e2"]},
    ]
    beep_map_path = tmp_path / "call.audio.beeps.json"
    beep_map_path.write_text(json.dumps(beep_map), encoding="utf-8")

    # Deliberately reverse entity order to simulate sorting/reordering between stages.
    entities = [
        DetectedEntity(
            entity_id="e2",
            modality=Modality.AUDIO,
            label="EMAIL",
            confidence=0.95,
            text="person@example.com",
            mitigation="mask",
            detection_method="regex",
        ),
        DetectedEntity(
            entity_id="e1",
            modality=Modality.AUDIO,
            label="PHONE",
            confidence=0.95,
            text="425-555-1212",
            mitigation="mask",
            detection_method="regex",
        ),
    ]
    result = DetectionResult(
        source_path=tmp_path / "call.wav",
        modality=Modality.AUDIO,
        entities=entities,
        artifacts=[beep_map_path],
    )

    audit_path = record_audit(result, ExplainabilityConfig(audit_log_path=tmp_path / "audit.log"))
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    spans = payload["spans"]

    assert spans[0]["entity_type"] == "EMAIL"
    assert spans[0]["timeline_window"] == [2000, 2600]
    assert spans[1]["entity_type"] == "PHONE"
    assert spans[1]["timeline_window"] == [1000, 1500]


@pytest.mark.parametrize(
    ("system_ffmpeg", "system_raises"),
    [
        ("system_ffmpeg", True),  # binary discovered, but invocation fails
        (None, False),             # binary absent from PATH
    ],
)
def test_ffmpeg_fallback_is_used_when_system_binary_fails(
    tmp_path: Path,
    system_ffmpeg: str | None,
    system_raises: bool,
) -> None:
    audio_path = tmp_path / "call.wav"
    transcript_path = tmp_path / "call.txt"
    _write_u8_wave(audio_path)
    transcript_path.write_text("Call me at 425-555-1212", encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=False,
            regex_entities=[RegexEntityConfig(name="phone", pattern=r"\b\d{3}-\d{3}-\d{4}\b", action="mask")],
        ),
        image=ImageConfig(enable_ocr=False),
        explainability=ExplainabilityConfig(
            save_text_spans=True,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
        audio=AudioConfig(enabled=True, enable_transcription=False),
    )
    manager = PipelineManager(config)

    commands: list[str] = []

    def _fake_run(cmd: list[str], capture_output: bool, text: bool):
        commands.append(cmd[0])
        if system_raises and cmd[0] == "system_ffmpeg":
            raise FileNotFoundError("system ffmpeg not found")

        # Simulate fallback ffmpeg output conversion into 16-bit PCM wav.
        output = Path(cmd[-1])
        _write_sine_wave(output, seconds=1.0)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with (
        patch("leakwatch.mitigation.audio.shutil.which", return_value=system_ffmpeg),
        patch.object(manager.audio_mitigator, "_bundled_ffmpeg", return_value="fallback_ffmpeg"),
        patch("leakwatch.mitigation.audio.subprocess.run", side_effect=_fake_run),
    ):
        result = manager.process_audio(audio_path)

    if system_ffmpeg:
        assert commands[0] == "system_ffmpeg"
    else:
        assert commands[0] == "fallback_ffmpeg"
    assert "fallback_ffmpeg" in commands
    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    assert result.runtime_metadata.get("ffmpeg_fallback") is True

    assert result.audit_log is not None
    payload = json.loads(result.audit_log.read_text(encoding="utf-8"))
    assert payload["ffmpeg_fallback"] is True
