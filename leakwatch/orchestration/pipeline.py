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


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be inspected and sanitized completely."""


LOGGER = get_logger(__name__)
TEXT_EXTENSIONS = {".txt", ".md", ".json"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


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
            app_mode=self.config.app.mode,
            ffmpeg_binary=self.config.audio.ffmpeg_binary,
        )
        self.audio_mitigator = AudioMitigator(self.config.audio)
        self.video_adapter = VideoAdapter(
            frame_stride=1,  # Always extract all frames for continuous temporal mitigation coverage
            max_frames=self.config.video.max_frames,
        )

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

    def detect_text_entities(self, text: str) -> list[DetectedEntity]:
        """Detect text entities and attach stable span-based IDs for API/UI use."""
        return self._stamp_entity_ids(self.text_detector.detect(text))

    def mitigate_text_entities(
        self,
        text: str,
        entities: list[DetectedEntity],
        selected_entity_types: set[str],
    ) -> tuple[str, list[DetectedEntity]]:
        """Mitigate only the selected entity types from the unchanged source text."""
        selected_entities = [
            entity for entity in entities if entity.label.upper() in selected_entity_types
        ]
        if not selected_entities:
            return text, []
        return self.text_mitigator.mitigate(
            text,
            selected_entities,
            force_synthetic=True,
        )

    # --- Audio Modality ----------------------------------------------
    def process_audio(self, path: Path) -> DetectionResult:
        if not self.config.audio.enabled:
            raise RuntimeError("Audio modality disabled in configuration.")
        source_path = Path(path)
        LOGGER.info("Processing audio: %s", source_path.name)
        transcript = self.audio_adapter.to_text(source_path)

        detected_entities = self._stamp_entity_ids(self.text_detector.detect(transcript))
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
            word_timestamps=getattr(self.audio_adapter, "last_word_timestamps", None),
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
            runtime_metadata={
                "ffmpeg_fallback": self.audio_mitigator.last_ffmpeg_fallback,
                "detection_mode": self.text_detector.detection_mode,
                "multi_speaker_detected": self.audio_adapter.last_multi_speaker_detected,
                "redaction_confidence": self.audio_adapter.last_redaction_confidence,
            },
        )
        self._attach_audit(result)
        return result

    def _stamp_entity_ids(self, entities: list[DetectedEntity]) -> list[DetectedEntity]:
        stamped: list[DetectedEntity] = []
        for index, entity in enumerate(entities):
            start = entity.span.start if entity.span else -1
            end = entity.span.end if entity.span else -1
            token = f"{index}:{entity.label}:{start}:{end}:{entity.text or ''}"
            stamped.append(entity.model_copy(update={"entity_id": token}))
        return stamped

    # --- Image Modality -----------------------------------------------
    def process_image(
        self,
        path: Path,
        output_path: Path | None = None,
        entities_override: list[DetectedEntity] | None = None,
    ) -> DetectionResult:
        source_path = Path(path)
        LOGGER.info("Processing image: %s", source_path.name)
        entities = entities_override if entities_override is not None else self.image_detector.detect(source_path)
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
            runtime_metadata={"detection_mode": self.text_detector.detection_mode},
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

        try:
            # NOTE: frame_stride > 1 is supported via temporal gap-filling
            # (_fill_temporal_detection_gaps) which propagates detections to
            # intermediate frames during reconstruction.
            if self.video_adapter.frame_stride != 1:
                raise RuntimeError(
                    "Video sanitization requires frame_stride=1 until tracked propagation "
                    "can guarantee coverage for intermediate frames."
                )
            if self.video_adapter.max_frames is not None:
                raise RuntimeError(
                    "Video sanitization does not allow max_frames because it would leave "
                    "the remaining output frames uninspected."
                )
            frames = self.video_adapter.extract_frames(source_path, frame_dir)
            if not frames:
                raise RuntimeError("Video contained no decodable frames.")
            frame_paths = [frame.path for frame in frames]
            frame_index_by_path = {frame.path: frame.index for frame in frames}
            aggregated_entities: list[DetectedEntity] = []
            artifacts: list[Path] = []
            processed_frames: dict[int, Path] = {}

            redaction_mode = (self.config.video.redaction_mode or "per_frame").lower().strip()
            detection_stride = max(1, int(self.config.video.frame_stride or 1))

            # Run detection on a strided basis to save computation time
            detection_map: dict[Path, list[DetectedEntity]] = {}
            for idx, frame_path in enumerate(frame_paths):
                if idx % detection_stride == 0:
                    detection_map[frame_path] = self.image_detector.detect(frame_path)
                else:
                    detection_map[frame_path] = []

            if redaction_mode == "per_frame":
                effective_map = self._fill_temporal_detection_gaps(frame_paths, detection_map)
                for frame_path in frame_paths:
                    frame_result = self.process_image(
                        frame_path,
                        entities_override=effective_map.get(frame_path, []),
                    )
                    aggregated_entities.extend(frame_result.entities)
                    artifacts.extend(frame_result.artifacts)
                    artifacts.append(frame_path)
                    if frame_result.mitigated_output:
                        artifacts.append(frame_result.mitigated_output)
                        processed_frames[frame_index_by_path[frame_path]] = frame_result.mitigated_output
            else:
                union_entities: list[DetectedEntity] = []
                for frame_path in frame_paths:
                    detected = detection_map.get(frame_path, [])
                    if redaction_mode == "strict" and detected:
                        union_entities.extend(detected)

                strict_entities = self._dedupe_entities(union_entities) if union_entities else []
                carryover_entities: list[DetectedEntity] = []
                carryover_remaining = max(0, int(self.config.video.carryover_frames or 0))

                for frame_path in frame_paths:
                    detected = detection_map.get(frame_path, [])
                    if redaction_mode == "strict" and strict_entities:
                        use_entities = strict_entities
                    elif redaction_mode == "carryover":
                        if detected:
                            use_entities = detected
                            carryover_entities = detected
                            carryover_remaining = max(0, int(self.config.video.carryover_frames or 0))
                        elif carryover_entities and carryover_remaining > 0:
                            use_entities = carryover_entities
                            carryover_remaining -= 1
                        else:
                            use_entities = []
                    else:
                        use_entities = detected

                    frame_result = self.process_image(frame_path, entities_override=use_entities)
                    aggregated_entities.extend(frame_result.entities)
                    artifacts.extend(frame_result.artifacts)
                    artifacts.append(frame_path)
                    if frame_result.mitigated_output:
                        artifacts.append(frame_result.mitigated_output)
                        processed_frames[frame_index_by_path[frame_path]] = frame_result.mitigated_output

            output_video_path = self._output_path(source_path, suffix=".sanitized.mp4")
            ensure_dir(output_video_path)
            reconstructed_video = self.video_adapter.reconstruct_video(source_path, processed_frames, output_video_path)
            artifacts.append(reconstructed_video)

            result = DetectionResult(
                source_path=source_path,
                modality=Modality.VIDEO,
                entities=aggregated_entities,
                mitigated_output=reconstructed_video,
                artifacts=artifacts,
                runtime_metadata={
                    "detection_mode": self.text_detector.detection_mode,
                    "redaction_mode": redaction_mode,
                    **self.video_adapter.last_reconstruction_metadata,
                },
            )
            self._attach_audit(result)
            return result

        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Video processing failed for %s: %s", source_path, exc)
            result = DetectionResult(
                source_path=source_path,
                modality=Modality.VIDEO,
                entities=[],
                mitigated_output=None,
                artifacts=[],
                runtime_metadata={
                    "detection_mode": self.text_detector.detection_mode,
                    "failure_reason": str(exc),
                },
            )
            self._attach_audit(result)
            raise VideoProcessingError(f"Video processing failed for {source_path.name}: {exc}") from exc

    @staticmethod
    def _dedupe_entities(entities: list[DetectedEntity]) -> list[DetectedEntity]:
        seen: set[tuple[str, int, int, int, int]] = set()
        unique: list[DetectedEntity] = []
        for entity in entities:
            if not entity.bbox:
                continue
            key = (
                entity.label,
                int(entity.bbox.x),
                int(entity.bbox.y),
                int(entity.bbox.width),
                int(entity.bbox.height),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(entity)
        return unique

    def _fill_temporal_detection_gaps(
        self,
        frame_paths: list[Path],
        detection_map: dict[Path, list[DetectedEntity]],
    ) -> dict[Path, list[DetectedEntity]]:
        carry_window = max(0, int(self.config.video.carryover_frames or 0))
        if carry_window <= 0:
            return detection_map

        filled: dict[Path, list[DetectedEntity]] = {}
        path_by_index = {idx: path for idx, path in enumerate(frame_paths)}
        non_empty_indexes = [idx for idx, path in path_by_index.items() if detection_map.get(path)]

        for idx, path in path_by_index.items():
            current_entities = detection_map.get(path, [])
            if current_entities:
                filled[path] = current_entities
                continue

            previous_index = next(
                (candidate for candidate in range(idx - 1, max(-1, idx - carry_window - 1), -1) if candidate in non_empty_indexes),
                None,
            )
            next_index = next(
                (candidate for candidate in range(idx + 1, min(len(frame_paths), idx + carry_window + 1)) if candidate in non_empty_indexes),
                None,
            )

            propagated: list[DetectedEntity] = []
            if previous_index is not None:
                propagated.extend(self._expand_entities_for_gap(detection_map[path_by_index[previous_index]], idx - previous_index))
            if next_index is not None:
                propagated.extend(self._expand_entities_for_gap(detection_map[path_by_index[next_index]], next_index - idx))

            filled[path] = self._dedupe_entities(propagated)

        return filled

    @staticmethod
    def _expand_entities_for_gap(entities: list[DetectedEntity], frame_gap: int) -> list[DetectedEntity]:
        if frame_gap <= 0:
            return entities

        expanded: list[DetectedEntity] = []
        # Slightly enlarge propagated boxes so moving targets stay redacted across missed detections.
        grow_pixels = min(24, 4 * frame_gap)
        for entity in entities:
            if not entity.bbox:
                expanded.append(entity)
                continue
            x = max(0, int(entity.bbox.x) - grow_pixels)
            y = max(0, int(entity.bbox.y) - grow_pixels)
            width = max(1, int(entity.bbox.width) + (2 * grow_pixels))
            height = max(1, int(entity.bbox.height) + (2 * grow_pixels))
            expanded.append(
                entity.model_copy(
                    update={
                        "bbox": entity.bbox.model_copy(
                            update={"x": x, "y": y, "width": width, "height": height}
                        )
                    }
                )
            )
        return expanded

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
            elif suffix in VIDEO_EXTENSIONS:
                try:
                    results.append(self.process_video(path))
                except RuntimeError as exc:
                    LOGGER.warning("Skipping video %s: %s", path, exc)
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

        # Apply line wrapping if configured
        final_text = sanitized_text
        if self.config.text.wrap_lines:
            import textwrap
            paragraphs = sanitized_text.split("\n")
            wrapped_paragraphs = [
                textwrap.fill(p, width=self.config.text.max_line_length) if p.strip() else p
                for p in paragraphs
            ]
            final_text = "\n".join(wrapped_paragraphs)

        output_path.write_text(final_text, encoding="utf-8")
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
            runtime_metadata={"detection_mode": self.text_detector.detection_mode},
        )
        self._attach_audit(result)
        return result
