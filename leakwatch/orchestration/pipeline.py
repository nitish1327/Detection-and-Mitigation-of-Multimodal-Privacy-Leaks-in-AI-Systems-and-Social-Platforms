"""Pipeline orchestration for LeakWatch."""

from __future__ import annotations

import json
from pathlib import Path

from ..adapters import AudioAdapter, VideoAdapter
from ..detection import ImageDetector, TextDetector
from ..explainability.audit import record_audit
from ..explainability.image import render_image_overlay
from ..explainability.text import render_text_spans
from ..mitigation import AudioMitigator, ImageMitigator, TextMitigator
from ..utils.config import LeakWatchConfig, get_config
from ..utils.logging import ensure_dir, get_logger
from ..utils.types import DetectedEntity, DetectionResult, Modality

LOGGER = get_logger(__name__)
TEXT_EXTENSIONS = {".txt", ".md", ".json"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
AUDIO_EXTENSIONS = {".wav"}


class PipelineManager:
    """Coordinate modality-specific detectors and mitigators."""

    def __init__(self, config: LeakWatchConfig | None = None) -> None:
        self.config = config or get_config()
        self.text_detector = TextDetector(self.config.text)
        self.text_mitigator = TextMitigator(self.config.text)
        self.image_detector = ImageDetector(self.config.image, text_detector=self.text_detector)
        self.image_mitigator = ImageMitigator(self.config.image)
        self.audio_adapter = AudioAdapter(
            transcript_extension=self.config.audio.transcript_extension,
            enable_transcription=self.config.audio.enable_transcription,
            transcription_backend=self.config.audio.transcription_backend,
            whisper_model=self.config.audio.whisper_model,
        )
        self.audio_mitigator = AudioMitigator(self.config.audio)
        self.video_adapter = VideoAdapter()

    # --- Text Modality -------------------------------------------------
    def process_text(self, path: Path) -> DetectionResult:
        source_path = Path(path)
        raw_text = source_path.read_text(encoding="utf-8")
        LOGGER.info("Processing text: %s", source_path.name)
        return self._process_text_payload(
            source_path,
            raw_text,
            suffix=".sanitized.txt",
            modality=Modality.TEXT,
        )

    # --- Audio Modality ----------------------------------------------
    def process_audio(self, path: Path) -> DetectionResult:
        if not self.config.audio.enabled:
            raise RuntimeError("Audio modality disabled in configuration.")
        source_path = Path(path)
        LOGGER.info("Processing audio: %s", source_path.name)
        transcript = self.audio_adapter.to_text(source_path)

        detected_entities = self.text_detector.detect(transcript)
        sanitized_text, mitigated_entities = self.text_mitigator.mitigate(transcript, detected_entities)

        transcript_path = self._output_path(source_path, suffix=".transcript.txt")
        ensure_dir(transcript_path)
        transcript_path.write_text(transcript, encoding="utf-8")

        sanitized_text_path = self._output_path(source_path, suffix=".audio.sanitized.txt")
        ensure_dir(sanitized_text_path)
        sanitized_text_path.write_text(sanitized_text, encoding="utf-8")

        beeped_audio_path = self._output_path(source_path, suffix=".sanitized.wav")
        ensure_dir(beeped_audio_path)
        beeped_audio_path, beep_ranges = self.audio_mitigator.mitigate(
            source_path,
            transcript,
            detected_entities,
            beeped_audio_path,
        )

        artifacts: list[Path] = [transcript_path, sanitized_text_path]
        if self.config.explainability.save_text_spans and mitigated_entities:
            span_path = sanitized_text_path.with_suffix(".spans.txt")
            artifacts.append(render_text_spans(sanitized_text, mitigated_entities, span_path))

        beep_map_path = self._output_path(source_path, suffix=".audio.beeps.json")
        ensure_dir(beep_map_path)
        beep_map_path.write_text(json.dumps(beep_ranges, indent=2), encoding="utf-8")
        artifacts.append(beep_map_path)

        result = DetectionResult(
            source_path=source_path,
            modality=Modality.AUDIO,
            entities=mitigated_entities,
            mitigated_output=beeped_audio_path,
            artifacts=artifacts,
        )
        self._attach_audit(result)
        return result

    # --- Image Modality -----------------------------------------------
    def process_image(self, path: Path, output_path: Path | None = None) -> DetectionResult:
        source_path = Path(path)
        LOGGER.info("Processing image: %s", source_path.name)
        entities = self.image_detector.detect(source_path)
        final_output = output_path or self._output_path(source_path, suffix=f".sanitized{source_path.suffix}")
        ensure_dir(final_output)
        mitigated_path, mitigated_entities = self.image_mitigator.mitigate(source_path, entities, final_output)
        artifacts = []
        if self.config.explainability.save_image_overlays and mitigated_entities:
            overlay_path = final_output.with_suffix(".overlay.png")
            artifacts.append(render_image_overlay(mitigated_path, mitigated_entities, overlay_path))

        result = DetectionResult(
            source_path=source_path,
            modality=Modality.IMAGE,
            entities=mitigated_entities,
            mitigated_output=mitigated_path,
            artifacts=artifacts,
        )
        self._attach_audit(result)
        return result

    # --- Video Modality -----------------------------------------------
    def process_video(self, path: Path) -> DetectionResult:
        if not self.config.video.enabled:
            raise RuntimeError("Video modality disabled in configuration.")
        source_path = Path(path)
        LOGGER.info("Processing video: %s", source_path.name)
        frame_dir = Path(self.config.app.output_dir) / f"{source_path.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = self.video_adapter.extract_frames(source_path, frame_dir)
        aggregated_entities: list[DetectedEntity] = []
        artifacts: list[Path] = []
        sanitized_manifest_entries: list[str] = []

        for frame_path in frame_paths:
            frame_result = self.process_image(frame_path)
            aggregated_entities.extend(frame_result.entities)
            artifacts.extend(frame_result.artifacts)
            artifacts.append(frame_path)
            if frame_result.mitigated_output:
                artifacts.append(frame_result.mitigated_output)
                sanitized_manifest_entries.append(str(frame_result.mitigated_output))

        manifest_path = self._output_path(source_path, suffix=".video.manifest.txt")
        ensure_dir(manifest_path)
        manifest_path.write_text("\n".join(sanitized_manifest_entries), encoding="utf-8")
        result = DetectionResult(
            source_path=source_path,
            modality=Modality.VIDEO,
            entities=aggregated_entities,
            mitigated_output=manifest_path,
            artifacts=artifacts,
        )
        self._attach_audit(result)
        return result

    # --- Helpers -------------------------------------------------------
    def _output_path(self, source_path: Path, suffix: str) -> Path:
        base_name = f"{source_path.stem}{suffix}"
        return Path(self.config.app.output_dir) / base_name

    def process_folder(self, folder: Path, recursive: bool = True) -> list[DetectionResult]:
        folder = Path(folder)
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        results: list[DetectionResult] = []
        for path in iterator:
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in TEXT_EXTENSIONS:
                results.append(self.process_text(path))
            elif suffix in IMAGE_EXTENSIONS:
                try:
                    results.append(self.process_image(path))
                except FileNotFoundError:
                    LOGGER.warning("Skipping unreadable image: %s", path)
            elif suffix in AUDIO_EXTENSIONS:
                try:
                    results.append(self.process_audio(path))
                except RuntimeError as exc:
                    LOGGER.warning("Skipping audio %s: %s", path, exc)
        return results

    def _attach_audit(self, result: DetectionResult) -> None:
        if not self.config.explainability.audit_log_path:
            return
        audit_path = record_audit(result, self.config.explainability)
        result.audit_log = audit_path

    def _process_text_payload(
        self,
        source_reference: Path,
        raw_text: str,
        suffix: str,
        modality: Modality,
    ) -> DetectionResult:
        entities = self.text_detector.detect(raw_text)
        sanitized_text, mitigated_entities = self.text_mitigator.mitigate(raw_text, entities)
        output_path = self._output_path(source_reference, suffix=suffix)
        ensure_dir(output_path)
        output_path.write_text(sanitized_text, encoding="utf-8")
        artifacts = []
        if self.config.explainability.save_text_spans and mitigated_entities:
            span_path = output_path.with_suffix(".spans.txt")
            artifacts.append(render_text_spans(sanitized_text, mitigated_entities, span_path))

        result = DetectionResult(
            source_path=source_reference,
            modality=modality,
            entities=mitigated_entities,
            mitigated_output=output_path,
            artifacts=artifacts,
        )
        self._attach_audit(result)
        return result
