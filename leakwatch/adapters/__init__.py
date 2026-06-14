"""Adapters for audio/video placeholders and future integrations."""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping

import cv2

from ..utils.logging import ensure_dir, get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class VideoFrame:
	"""Extracted frame with explicit source timing and identity."""

	index: int
	timestamp_seconds: float
	path: Path


class AudioAdapter:
	"""Thin wrapper that maps audio inputs to transcripts for text reuse."""

	def __init__(
		self,
		transcript_extension: str = ".txt",
		enable_transcription: bool = True,
		transcription_backend: str = "whisper",
		whisper_model: str = "base",
		app_mode: str = "cli",
		transcriber: Callable[[Path], str] | None = None,
	) -> None:
		self.transcript_extension = transcript_extension
		self.enable_transcription = enable_transcription
		self.transcription_backend = transcription_backend
		self.whisper_model = whisper_model
		self.app_mode = app_mode
		self._transcriber = transcriber
		self._whisper_model_obj: Any | None = None
		self.last_multi_speaker_detected = False
		self.last_redaction_confidence = "normal"

	def to_text(self, audio_path: Path) -> str:
		self.last_multi_speaker_detected = False
		self.last_redaction_confidence = "normal"
		path = Path(audio_path)
		if path.suffix.lower() == self.transcript_extension:
			return path.read_text(encoding="utf-8")

		transcript_candidate = path.with_suffix(self.transcript_extension)
		if transcript_candidate.exists():
			LOGGER.info("Using companion transcript for audio: %s", transcript_candidate.name)
			return transcript_candidate.read_text(encoding="utf-8")

		if self.enable_transcription:
			LOGGER.info("No companion transcript found for %s. Running built-in transcription.", path.name)
			return self._transcribe(path)

		raise RuntimeError(
			f"No transcript found for {path.name}. Expected companion file: {transcript_candidate.name}"
		)

	def _transcribe(self, audio_path: Path) -> str:
		if self._transcriber is not None:
			result = self._transcriber(audio_path)
			if isinstance(result, dict):
				self._update_speaker_flags(result)
				return str(result.get("text", "")).strip()
			return str(result).strip()

		backend = self.transcription_backend.lower().strip()
		if backend != "whisper":
			raise RuntimeError(f"Unsupported transcription backend: {self.transcription_backend}")

		try:
			import whisper  # type: ignore
		except ImportError as exc:  # pragma: no cover
			raise RuntimeError(
				"Built-in transcription requires the optional Whisper package. "
				"Install it with: pip install openai-whisper"
			) from exc

		if self._whisper_model_obj is None:
			self._whisper_model_obj = whisper.load_model(self.whisper_model)

		transcript = self._whisper_model_obj.transcribe(str(audio_path), fp16=False, word_timestamps=True)
		self._update_speaker_flags(transcript)
		text = str(transcript.get("text", "")).strip()
		if not text:
			raise RuntimeError(f"Transcription produced empty text for {audio_path.name}")
		return text

	def _update_speaker_flags(self, transcript: dict[str, Any]) -> None:
		if not transcript:
			return
		segments = transcript.get("segments") or []
		if not isinstance(segments, list) or not segments:
			return

		words: list[tuple[float, float]] = []
		high_no_speech = 0
		for segment in segments:
			if not isinstance(segment, dict):
				continue
			no_speech_prob = float(segment.get("no_speech_prob", 0.0) or 0.0)
			if no_speech_prob >= 0.6:
				high_no_speech += 1
			for word in segment.get("words", []) or []:
				if not isinstance(word, dict):
					continue
				start = word.get("start")
				end = word.get("end")
				if start is None or end is None:
					continue
				try:
					words.append((float(start), float(end)))
				except (TypeError, ValueError):
					continue

		rapid_gap_count = 0
		overlap_count = 0
		pair_count = 0
		for idx in range(1, len(words)):
			gap = words[idx][0] - words[idx - 1][1]
			pair_count += 1
			if gap < 0:
				overlap_count += 1
			if gap < 0.05:
				rapid_gap_count += 1

		rapid_ratio = (rapid_gap_count / pair_count) if pair_count else 0.0
		multi_speaker_likely = (
			overlap_count > 0
			or rapid_ratio > 0.2
			or high_no_speech >= 2
		)

		if multi_speaker_likely:
			self.last_multi_speaker_detected = True
			self.last_redaction_confidence = "reduced"
			message = (
				"[LeakWatch] Multi-speaker overlap likely detected in audio transcript; "
				"redaction precision may be lower for tightly overlapping speech."
			)
			LOGGER.warning(message)
			if self.app_mode == "cli":
				print(message)


class VideoAdapter:
	"""Frame sampler that funnels videos back into the image pipeline."""

	def __init__(self, frame_stride: int = 1, max_frames: int | None = None) -> None:
		self.frame_stride = max(1, frame_stride)
		self.max_frames = max_frames if max_frames is None else max(1, max_frames)
		self.last_reconstruction_metadata: dict[str, Any] = {}

	def extract_frames(self, video_path: Path, frame_dir: Path) -> List[VideoFrame]:
		frame_dir = Path(frame_dir)
		frame_dir.mkdir(parents=True, exist_ok=True)
		capture = cv2.VideoCapture(str(video_path))
		if not capture.isOpened():
			LOGGER.warning("Unable to open video: %s", video_path)
			return []

		fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
		extracted: list[VideoFrame] = []
		frame_index = 0
		saved = 0
		while True:
			ret, frame = capture.read()
			if not ret:
				break
			if frame_index % self.frame_stride == 0:
				frame_path = frame_dir / f"{video_path.stem}_frame_{frame_index:05d}.png"
				ensure_dir(frame_path)
				if not cv2.imwrite(str(frame_path), frame):
					capture.release()
					raise RuntimeError(f"Unable to write extracted video frame: {frame_path}")
				timestamp_seconds = (frame_index / fps) if fps > 0 else 0.0
				extracted.append(
					VideoFrame(
						index=frame_index,
						timestamp_seconds=timestamp_seconds,
						path=frame_path,
					)
				)
				saved += 1
				if self.max_frames is not None and saved >= self.max_frames:
					break
			frame_index += 1

		capture.release()
		if not extracted:
			LOGGER.warning("Video contained no readable frames: %s", video_path)
		return extracted

	def reconstruct_video(
		self,
		source_video_path: Path,
		processed_frames: Mapping[int, Path],
		output_video_path: Path,
	) -> Path:
		self.last_reconstruction_metadata = {}
		capture = cv2.VideoCapture(str(source_video_path))
		if not capture.isOpened():
			raise RuntimeError(f"Unable to open source video for reconstruction: {source_video_path}")

		fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
		if not fps or fps <= 0:
			capture.release()
			raise RuntimeError("Source video does not expose a valid FPS value")
		expected_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		if width <= 0 or height <= 0:
			ok, first_source = capture.read()
			if not ok or first_source is None:
				capture.release()
				raise RuntimeError("Unable to read source frames for reconstruction")
			height, width = first_source.shape[:2]
			capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

		replacements: dict[int, Any] = {}
		for frame_index, frame_path in processed_frames.items():
			frame = cv2.imread(str(frame_path))
			if frame is None:
				capture.release()
				raise RuntimeError(f"Unable to read processed video frame: {frame_path}")
			if frame.shape[0] != height or frame.shape[1] != width:
				frame = cv2.resize(frame, (width, height))
			replacements[frame_index] = frame

		temp_output_path = output_video_path.with_name(f"{output_video_path.stem}.tmp.mp4")
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))
		if not writer.isOpened():
			capture.release()
			raise RuntimeError(f"Unable to open video writer for: {temp_output_path}")

		frame_idx = 0
		while True:
			ok, source_frame = capture.read()
			if not ok or source_frame is None:
				break
			output_frame = replacements.get(frame_idx, source_frame)
			writer.write(output_frame)
			frame_idx += 1

		writer.release()
		capture.release()
		if expected_frame_count > 0 and frame_idx != expected_frame_count:
			temp_output_path.unlink(missing_ok=True)
			raise RuntimeError(
				f"Source decode ended early: expected {expected_frame_count} frames, decoded {frame_idx}"
			)
		if len(replacements) != frame_idx:
			temp_output_path.unlink(missing_ok=True)
			raise RuntimeError(
				f"Processed frame coverage mismatch: expected {frame_idx} replacements, got {len(replacements)}"
			)

		self._transcode_h264(temp_output_path, source_video_path, output_video_path)
		output_frame_count, output_fps = self._video_metrics(output_video_path)
		if output_frame_count != frame_idx:
			raise RuntimeError(
				f"Output frame count mismatch: expected {frame_idx}, got {output_frame_count}"
			)
		fps_tolerance = max(0.01, fps * 0.001)
		if abs(output_fps - fps) > fps_tolerance:
			raise RuntimeError(f"Output FPS mismatch: expected {fps:.6f}, got {output_fps:.6f}")

		source_has_audio = self._has_audio_stream(source_video_path)
		output_has_audio = self._has_audio_stream(output_video_path)
		if source_has_audio and not output_has_audio:
			raise RuntimeError("Source audio stream was not preserved in reconstructed video")

		self.last_reconstruction_metadata = {
			"source_frame_count": frame_idx,
			"output_frame_count": output_frame_count,
			"source_fps": fps,
			"output_fps": output_fps,
			"source_duration_seconds": frame_idx / fps,
			"output_duration_seconds": output_frame_count / output_fps,
			"source_has_audio": source_has_audio,
			"output_has_audio": output_has_audio,
		}
		return output_video_path

	def _transcode_h264(self, input_path: Path, source_path: Path, output_path: Path) -> None:
		ffmpeg_binary = self._ffmpeg_executable()
		if not ffmpeg_binary:
			input_path.unlink(missing_ok=True)
			raise RuntimeError("ffmpeg is required to preserve video audio and container timing")

		command = [
			ffmpeg_binary,
			"-y",
			"-i",
			str(input_path),
			"-i",
			str(source_path),
			"-map",
			"0:v:0",
			"-map",
			"1:a?",
			"-c:v",
			"libx264",
			"-c:a",
			"aac",
			"-pix_fmt",
			"yuv420p",
			"-movflags",
			"+faststart",
			"-map_metadata",
			"1",
			str(output_path),
		]
		result = subprocess.run(command, capture_output=True, text=True)
		if result.returncode != 0:
			input_path.unlink(missing_ok=True)
			raise RuntimeError(
				f"H.264 transcode failed for {output_path.name}: {result.stderr.strip()}"
			)
		if input_path.exists():
			input_path.unlink()

	def _video_metrics(self, path: Path) -> tuple[int, float]:
		capture = cv2.VideoCapture(str(path))
		if not capture.isOpened():
			raise RuntimeError(f"Unable to open reconstructed video for validation: {path}")
		frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
		capture.release()
		if frame_count <= 0 or fps <= 0:
			raise RuntimeError(f"Reconstructed video has invalid metrics: frames={frame_count}, fps={fps}")
		return frame_count, fps

	def _has_audio_stream(self, path: Path) -> bool:
		ffmpeg_binary = self._ffmpeg_executable()
		if not ffmpeg_binary:
			raise RuntimeError("ffmpeg is required to validate video audio streams")
		result = subprocess.run(
			[ffmpeg_binary, "-hide_banner", "-i", str(path)],
			capture_output=True,
			text=True,
		)
		probe_text = f"{result.stdout}\n{result.stderr}"
		return "Audio:" in probe_text

	def _ffmpeg_executable(self) -> str | None:
		system_ffmpeg = shutil.which("ffmpeg")
		if system_ffmpeg:
			return system_ffmpeg
		try:
			from imageio_ffmpeg import get_ffmpeg_exe
		except ImportError:
			return None
		fallback = get_ffmpeg_exe()
		if fallback and Path(fallback).exists():
			return fallback
		return None


# GAN/GraphSAGE powered adapters may be slotted in later; Phase 2 intentionally stays deterministic.
