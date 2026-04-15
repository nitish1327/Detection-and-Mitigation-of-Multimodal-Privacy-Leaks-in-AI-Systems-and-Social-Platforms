"""Audio mitigation by replacing sensitive regions with a generated beep tone."""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

from ..utils.config import AudioConfig
from ..utils.types import DetectedEntity


class AudioMitigator:
    """Mitigate sensitive audio spans by overlaying a tone in WAV waveforms."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def mitigate(
        self,
        audio_path: Path,
        transcript: str,
        entities: list[DetectedEntity],
        output_path: Path,
    ) -> tuple[Path, list[dict[str, float]]]:
        source_path = Path(audio_path)
        if source_path.suffix.lower() != ".wav":
            raise RuntimeError("Audio beeping currently supports only .wav inputs.")

        with wave.open(str(source_path), "rb") as in_wav:
            channels = in_wav.getnchannels()
            sample_width = in_wav.getsampwidth()
            frame_rate = in_wav.getframerate()
            frame_count = in_wav.getnframes()
            raw_frames = in_wav.readframes(frame_count)

        if sample_width != 2:
            raise RuntimeError("Audio beeping currently supports 16-bit PCM WAV files only.")

        duration_seconds = frame_count / frame_rate if frame_rate else 0
        time_ranges = self._ranges_from_entities(transcript, entities, duration_seconds)

        samples = array("h")
        samples.frombytes(raw_frames)

        # Convert entity-aligned time windows into sample indices and paint a tone.
        for window in time_ranges:
            start_index = max(0, int(window["start_sec"] * frame_rate))
            end_index = min(frame_count, int(window["end_sec"] * frame_rate))
            for frame_idx in range(start_index, end_index):
                tone_value = int(
                    32767
                    * self.config.beep_amplitude
                    * math.sin(2.0 * math.pi * self.config.beep_frequency_hz * (frame_idx / frame_rate))
                )
                base = frame_idx * channels
                for ch in range(channels):
                    samples[base + ch] = tone_value

        with wave.open(str(output_path), "wb") as out_wav:
            out_wav.setnchannels(channels)
            out_wav.setsampwidth(sample_width)
            out_wav.setframerate(frame_rate)
            out_wav.writeframes(samples.tobytes())

        return output_path, time_ranges

    def _ranges_from_entities(
        self,
        transcript: str,
        entities: list[DetectedEntity],
        duration_seconds: float,
    ) -> list[dict[str, float]]:
        if not transcript or not entities or duration_seconds <= 0:
            return []

        windows: list[dict[str, float]] = []
        transcript_len = max(1, len(transcript))
        padding = max(0.0, self.config.padding_ms / 1000.0)
        min_beep = max(0.0, self.config.min_beep_ms / 1000.0)

        for entity in entities:
            if not entity.span:
                continue
            start_char = max(0, min(entity.span.start, transcript_len))
            end_char = max(start_char, min(entity.span.end, transcript_len))
            start_sec = (start_char / transcript_len) * duration_seconds
            end_sec = (end_char / transcript_len) * duration_seconds
            if end_sec - start_sec < min_beep:
                end_sec = min(duration_seconds, start_sec + min_beep)

            windows.append(
                {
                    "label": entity.label,
                    "start_sec": max(0.0, start_sec - padding),
                    "end_sec": min(duration_seconds, end_sec + padding),
                }
            )

        if not windows:
            return []

        windows.sort(key=lambda item: item["start_sec"])
        merged: list[dict[str, float]] = [windows[0].copy()]

        for current in windows[1:]:
            previous = merged[-1]
            if current["start_sec"] <= previous["end_sec"]:
                previous["end_sec"] = max(previous["end_sec"], current["end_sec"])
                if current["label"] != previous["label"]:
                    previous["label"] = f"{previous['label']}|{current['label']}"
            else:
                merged.append(current.copy())

        return merged
