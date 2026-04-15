"""Adapters for audio/video placeholders and future integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List

import cv2

from ..utils.logging import ensure_dir, get_logger

LOGGER = get_logger(__name__)


class AudioAdapter:
	"""Thin wrapper that maps audio inputs to transcripts for text reuse."""

	def __init__(
		self,
		transcript_extension: str = ".txt",
		enable_transcription: bool = True,
		transcription_backend: str = "whisper",
		whisper_model: str = "base",
		transcriber: Callable[[Path], str] | None = None,
	) -> None:
		self.transcript_extension = transcript_extension
		self.enable_transcription = enable_transcription
		self.transcription_backend = transcription_backend
		self.whisper_model = whisper_model
		self._transcriber = transcriber
		self._whisper_model_obj: Any | None = None

	def to_text(self, audio_path: Path) -> str:
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
			return self._transcriber(audio_path)

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

		transcript = self._whisper_model_obj.transcribe(str(audio_path), fp16=False)
		text = str(transcript.get("text", "")).strip()
		if not text:
			raise RuntimeError(f"Transcription produced empty text for {audio_path.name}")
		return text


class VideoAdapter:
	"""Frame sampler that funnels videos back into the image pipeline."""

	def __init__(self, frame_stride: int = 15, max_frames: int = 5) -> None:
		self.frame_stride = max(1, frame_stride)
		self.max_frames = max(1, max_frames)

	def extract_frames(self, video_path: Path, frame_dir: Path) -> List[Path]:
		frame_dir = Path(frame_dir)
		frame_dir.mkdir(parents=True, exist_ok=True)
		capture = cv2.VideoCapture(str(video_path))
		if not capture.isOpened():
			LOGGER.warning("Unable to open video: %s", video_path)
			return []

		extracted: list[Path] = []
		frame_index = 0
		saved = 0
		while saved < self.max_frames:
			ret, frame = capture.read()
			if not ret:
				break
			if frame_index % self.frame_stride == 0:
				frame_path = frame_dir / f"{video_path.stem}_frame_{frame_index:05d}.png"
				ensure_dir(frame_path)
				cv2.imwrite(str(frame_path), frame)
				extracted.append(frame_path)
				saved += 1
			frame_index += 1

		capture.release()
		if not extracted:
			LOGGER.warning("Video contained no readable frames: %s", video_path)
		return extracted

	def reconstruct_video(
		self,
		source_video_path: Path,
		processed_frames: List[Path],
		output_video_path: Path,
	) -> Path:
		if not processed_frames:
			raise RuntimeError("No processed frames available to reconstruct video")

		first_frame = cv2.imread(str(processed_frames[0]))
		if first_frame is None:
			raise RuntimeError(f"Unable to read processed frame: {processed_frames[0]}")

		height, width = first_frame.shape[:2]
		fps = 10.0
		capture = cv2.VideoCapture(str(source_video_path))
		if capture.isOpened():
			detected_fps = capture.get(cv2.CAP_PROP_FPS)
			if detected_fps and detected_fps > 0:
				fps = float(detected_fps)
		capture.release()

		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
		if not writer.isOpened():
			raise RuntimeError(f"Unable to open video writer for: {output_video_path}")

		for frame_path in processed_frames:
			frame = cv2.imread(str(frame_path))
			if frame is None:
				continue
			if frame.shape[0] != height or frame.shape[1] != width:
				frame = cv2.resize(frame, (width, height))
			writer.write(frame)

		writer.release()
		return output_video_path


# GAN/GraphSAGE powered adapters may be slotted in later; Phase 2 intentionally stays deterministic.
