"""Audio mitigation by replacing sensitive regions with a generated beep tone."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import wave
from array import array
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..utils.config import AudioConfig
from ..utils.logging import get_logger
from ..utils.types import DetectedEntity


LOGGER = get_logger(__name__)


def _resolve_bundled_ffmpeg() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    ffmpeg_exe = get_ffmpeg_exe()
    return ffmpeg_exe if ffmpeg_exe and Path(ffmpeg_exe).exists() else None


def _startup_validate_ffmpeg() -> None:
    system_ffmpeg = shutil.which("ffmpeg")
    fallback_ffmpeg = _resolve_bundled_ffmpeg()
    if system_ffmpeg or fallback_ffmpeg:
        return
    raise RuntimeError(
        "LeakWatch audio mitigation requires ffmpeg. Install ffmpeg in PATH or install imageio-ffmpeg "
        "(pip install imageio-ffmpeg) so bundled fallback can be used."
    )


_startup_validate_ffmpeg()


class AudioMitigator:
    """Mitigate sensitive audio spans by overlaying a tone in WAV waveforms."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.last_ffmpeg_fallback = False

    def mitigate(
        self,
        audio_path: Path,
        transcript: str,
        entities: list[DetectedEntity],
        output_path: Path,
        word_timestamps: list[dict[str, Any]] | None = None,
    ) -> tuple[Path, list[dict[str, float]]]:
        source_path = Path(audio_path)
        normalized_path, cleanup_path, fallback_used = self._normalize_for_beeping(source_path)
        self.last_ffmpeg_fallback = fallback_used
        with wave.open(str(normalized_path), "rb") as in_wav:
            channels = in_wav.getnchannels()
            sample_width = in_wav.getsampwidth()
            frame_rate = in_wav.getframerate()
            frame_count = in_wav.getnframes()
            raw_frames = in_wav.readframes(frame_count)

        if sample_width != 2:
            raise RuntimeError("Audio normalization failed to produce 16-bit PCM WAV frames.")

        duration_seconds = frame_count / frame_rate if frame_rate else 0
        time_ranges = self._ranges_from_entities(
            transcript, entities, duration_seconds, word_timestamps=word_timestamps
        )

        samples = array("h")
        samples.frombytes(raw_frames)

        mask_style = getattr(self.config, "mask_style", "beep").lower()
        amplitude = self.config.beep_amplitude
        import random

        # Pink noise state variables if needed
        b0, b1, b2, b3, b4, b5, b6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Convert entity-aligned time windows into sample indices and paint audio masking.
        for window in time_ranges:
            window["ffmpeg_fallback"] = fallback_used
            start_index = max(0, int(window["start_sec"] * frame_rate))
            end_index = min(frame_count, int(window["end_sec"] * frame_rate))
            for frame_idx in range(start_index, end_index):
                t = (frame_idx / frame_rate) if frame_rate else 0.0
                if mask_style == "silence":
                    tone_value = 0
                elif mask_style == "white_noise":
                    tone_value = int(random.uniform(-1.0, 1.0) * 32767 * amplitude)
                elif mask_style == "pink_noise":
                    white = random.uniform(-1.0, 1.0)
                    b0 = 0.99886 * b0 + white * 0.0555179
                    b1 = 0.99332 * b1 + white * 0.0750759
                    b2 = 0.96900 * b2 + white * 0.1538520
                    b3 = 0.86650 * b3 + white * 0.3104856
                    b4 = 0.55000 * b4 + white * 0.5329522
                    b5 = -0.7616 * b5 - white * 0.0168980
                    pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
                    b6 = white * 0.115926
                    tone_value = int(max(-32767, min(32767, pink * 0.2 * 32767 * amplitude)))
                elif mask_style == "garble":
                    mod = math.sin(2.0 * math.pi * 5.0 * t)
                    carrier = math.sin(2.0 * math.pi * (250.0 + 180.0 * mod) * t)
                    base = frame_idx * channels
                    orig = samples[base] if base < len(samples) else 0
                    tone_value = int(max(-32767, min(32767, (orig * 0.3) + (carrier * 32767 * amplitude * 0.6))))
                else:  # default "beep"
                    tone_value = int(
                        32767
                        * amplitude
                        * math.sin(2.0 * math.pi * self.config.beep_frequency_hz * t)
                    )
                base = frame_idx * channels
                for ch in range(channels):
                    samples[base + ch] = tone_value

        target_ext = output_path.suffix.lower()
        if target_ext in {".mp3", ".m4a", ".aac", ".ogg", ".webm"}:
            wav_temp = output_path.with_suffix(".tmp.wav")
            with wave.open(str(wav_temp), "wb") as out_wav:
                out_wav.setnchannels(channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(frame_rate)
                out_wav.writeframes(samples.tobytes())

            ffmpeg = shutil.which(self.config.ffmpeg_binary) or self._bundled_ffmpeg()
            if ffmpeg:
                cmd = [ffmpeg, "-y", "-i", str(wav_temp), str(output_path)]
                res = subprocess.run(cmd, capture_output=True, text=True)
                wav_temp.unlink(missing_ok=True)
                if res.returncode == 0:
                    if cleanup_path is not None:
                        with suppress(FileNotFoundError):
                            cleanup_path.unlink()
                    return output_path, time_ranges
            wav_temp.unlink(missing_ok=True)

        with wave.open(str(output_path), "wb") as out_wav:
            out_wav.setnchannels(channels)
            out_wav.setsampwidth(sample_width)
            out_wav.setframerate(frame_rate)
            out_wav.writeframes(samples.tobytes())

        if cleanup_path is not None:
            with suppress(FileNotFoundError):
                cleanup_path.unlink()

        return output_path, time_ranges

    def _normalize_for_beeping(self, source_path: Path) -> tuple[Path, Path | None, bool]:
        if source_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(source_path), "rb") as wav_file:
                    if wav_file.getsampwidth() == 2:
                        return source_path, None, False
            except wave.Error:
                pass

        system_ffmpeg = shutil.which(self.config.ffmpeg_binary)
        fallback_ffmpeg = self._bundled_ffmpeg()
        candidates: list[tuple[str, bool]] = []
        if system_ffmpeg:
            candidates.append((system_ffmpeg, False))
        if fallback_ffmpeg and fallback_ffmpeg != system_ffmpeg:
            candidates.append((fallback_ffmpeg, True))
        if not candidates:
            raise RuntimeError("ffmpeg is required to process non-WAV audio input.")

        tmp_file = tempfile.NamedTemporaryFile(prefix="leakwatch_audio_", suffix=".wav", delete=False)
        tmp_file_path = Path(tmp_file.name)
        tmp_file.close()

        failures: list[str] = []
        for ffmpeg_binary, is_fallback in candidates:
            command = [
                ffmpeg_binary,
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(tmp_file_path),
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True)
            except FileNotFoundError as exc:
                failures.append(f"{ffmpeg_binary}: {exc}")
                continue
            if result.returncode == 0:
                return tmp_file_path, tmp_file_path, is_fallback
            failures.append(f"{ffmpeg_binary}: {result.stderr.strip()}")

        with suppress(FileNotFoundError):
            tmp_file_path.unlink()
        joined = " | ".join(failures)
        raise RuntimeError(f"Failed to normalize audio using ffmpeg for {source_path.name}: {joined}")

    def _bundled_ffmpeg(self) -> str | None:
        return _resolve_bundled_ffmpeg()

    def _ranges_from_entities(
        self,
        transcript: str,
        entities: list[DetectedEntity],
        duration_seconds: float,
        word_timestamps: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not transcript or not entities or duration_seconds <= 0:
            return []

        windows: list[dict[str, Any]] = []
        transcript_len = max(1, len(transcript))
        padding = max(0.0, self.config.padding_ms / 1000.0)
        min_beep = max(0.0, self.config.min_beep_ms / 1000.0)
        merge_gap = max(0.0, self.config.merge_gap_ms / 1000.0)
        mapping_threshold = float(self.config.mapping_confidence_threshold)

        for index, entity in enumerate(entities):
            if not entity.span:
                continue
            if entity.confidence < mapping_threshold:
                LOGGER.warning(
                    "Sensitive span could not be confidently mapped: label=%s confidence=%.3f threshold=%.3f",
                    entity.label,
                    entity.confidence,
                    mapping_threshold,
                )
                continue
            start_char = max(0, min(entity.span.start, transcript_len))
            end_char = max(start_char, min(entity.span.end, transcript_len))

            start_sec: float | None = None
            end_sec: float | None = None

            if word_timestamps:
                matched_words = [
                    w for w in word_timestamps
                    if max(int(w.get("char_start", 0)), start_char) < min(int(w.get("char_end", 0)), end_char)
                ]
                if not matched_words:
                    matched_words = [
                        w for w in word_timestamps
                        if (start_char <= int(w.get("char_start", 0)) < end_char)
                        or (start_char < int(w.get("char_end", 0)) <= end_char)
                    ]

                if matched_words:
                    start_sec = min(float(w["start"]) for w in matched_words)
                    end_sec = max(float(w["end"]) for w in matched_words)

            if start_sec is None or end_sec is None:
                start_sec = (start_char / transcript_len) * duration_seconds
                end_sec = (end_char / transcript_len) * duration_seconds

            if end_sec - start_sec < min_beep:
                end_sec = min(duration_seconds, start_sec + min_beep)

            windows.append(
                {
                    "label": entity.label,
                    "start_sec": max(0.0, start_sec - padding),
                    "end_sec": min(duration_seconds, end_sec + padding),
                    "entity_indexes": [index],
                    "entity_ids": [entity.entity_id] if entity.entity_id else [],
                }
            )

        if not windows:
            return []

        windows.sort(key=lambda item: item["start_sec"])
        merged: list[dict[str, Any]] = [windows[0].copy()]

        for current in windows[1:]:
            previous = merged[-1]
            if current["start_sec"] <= previous["end_sec"] + merge_gap:
                previous["end_sec"] = max(previous["end_sec"], current["end_sec"])
                if current["label"] != previous["label"]:
                    previous["label"] = f"{previous['label']}|{current['label']}"
                previous["entity_indexes"].extend(current.get("entity_indexes", []))
                previous["entity_indexes"] = sorted(set(previous["entity_indexes"]))
                previous["entity_ids"].extend(current.get("entity_ids", []))
                previous["entity_ids"] = sorted(set(previous["entity_ids"]))
            else:
                merged.append(current.copy())

        return merged
