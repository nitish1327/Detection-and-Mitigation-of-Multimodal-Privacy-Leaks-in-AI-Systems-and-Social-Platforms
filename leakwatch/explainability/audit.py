"""Audit logging helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.types import Modality

from ..utils.config import ExplainabilityConfig
from ..utils.logging import ensure_dir
from ..utils.types import DetectionResult


def record_audit(result: DetectionResult, config: ExplainabilityConfig) -> Path:
    spans = _build_spans(result, config)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modality": result.modality.value,
        "source": str(result.source_path),
        "output": str(result.mitigated_output) if result.mitigated_output else None,
        "artifacts": [str(path) for path in result.artifacts],
        "entity_count": len(result.entities),
        "spans": spans,
        "entities": [
            {
                "label": ent.label,
                "confidence": ent.confidence,
                "text": ent.text,
                "span": ent.span.model_dump() if ent.span else None,
                "bbox": ent.bbox.model_dump() if ent.bbox else None,
                "mitigation": ent.mitigation,
                "detection_method": ent.detection_method,
            }
            for ent in result.entities
        ],
    }

    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    configured_path = Path(config.audit_log_path)
    audit_dir = configured_path if not configured_path.suffix else configured_path.parent
    audit_path = audit_dir / f"{result.source_path.stem}.{timestamp_slug}.audit.json"
    ensure_dir(audit_path)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(entry, handle, indent=2)
        handle.write("\n")
    return audit_path


def _build_spans(result: DetectionResult, config: ExplainabilityConfig) -> list[dict[str, Any]]:
    audio_windows = _audio_windows_for_entities(result)
    spans: list[dict[str, Any]] = []
    for index, entity in enumerate(result.entities):
        span_text = entity.explanation or entity.text
        if config.audit_span_text_policy.lower() == "hash" and span_text:
            span_text = hashlib.sha256(span_text.encode("utf-8")).hexdigest()

        if result.modality == Modality.IMAGE and entity.bbox:
            timeline_window: list[int] | None = [
                entity.bbox.x,
                entity.bbox.y,
                entity.bbox.width,
                entity.bbox.height,
            ]
        elif result.modality == Modality.AUDIO:
            timeline_window = audio_windows[index] if index < len(audio_windows) else None
        else:
            timeline_window = None

        spans.append(
            {
                "span_text": span_text,
                "entity_type": entity.label,
                "detection_method": entity.detection_method,
                "mitigation_applied": entity.mitigation,
                "confidence": entity.confidence if entity.detection_method == "nlp" else None,
                "timeline_window": timeline_window,
            }
        )
    return spans


def _audio_windows_for_entities(result: DetectionResult) -> list[list[int] | None]:
    if result.modality != Modality.AUDIO:
        return []

    beep_map = _read_beep_map(result)
    if not beep_map:
        return []

    windows: list[list[int] | None] = []
    for entity in result.entities:
        matched = _pop_matching_audio_window(entity.label, beep_map)
        if matched is None:
            windows.append(None)
            continue
        windows.append([
            int(float(matched.get("start_sec", 0.0)) * 1000),
            int(float(matched.get("end_sec", 0.0)) * 1000),
        ])
    return windows


def _read_beep_map(result: DetectionResult) -> list[dict[str, Any]]:
    for artifact in result.artifacts:
        if artifact.name.endswith(".audio.beeps.json") and artifact.exists():
            with artifact.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
    return []


def _pop_matching_audio_window(label: str, windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = label.lower()
    for index, window in enumerate(windows):
        candidate = str(window.get("label", "")).lower()
        if target in candidate:
            return windows.pop(index)
    if windows:
        return windows.pop(0)
    return None
