"""Text privacy detection module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

try:  # spaCy is optional at runtime
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover
    spacy = None
    Language = None  # type: ignore

from ..utils.config import TextConfig
from ..utils.logging import get_logger
from ..utils.types import DetectedEntity, Modality, Span

LOGGER = get_logger(__name__)


@dataclass
class _RegexPattern:
    name: str
    pattern: re.Pattern[str]
    action: str


class TextDetector:
    """Detect sensitive entities in text via spaCy + regex."""

    def __init__(self, config: TextConfig, model: str = "en_core_web_sm") -> None:
        self.config = config
        self._nlp: Language | None = None
        self._nlp_configured = bool(config.enable_spacy)
        self._nlp_loaded = False
        self._nlp_failed = False
        self._patterns = [
            _RegexPattern(item.name, re.compile(item.pattern, re.IGNORECASE), item.action)
            for item in config.regex_entities
        ]
        if self._nlp_configured:
            if spacy is None:
                self._nlp_failed = True
                LOGGER.warning(
                    "[LeakWatch] NLP configured but spaCy is unavailable; falling back to regex-only detection."
                )
            else:
                try:
                    self._nlp = spacy.load(model)
                    self._nlp_loaded = self._nlp is not None
                except (ImportError, OSError, Exception) as exc:  # noqa: BLE001
                    self._nlp_failed = True
                    self._nlp = None
                    LOGGER.warning(
                        "[LeakWatch] NLP configured but failed to load model '%s' (%s); falling back to regex-only detection.",
                        model,
                        exc,
                    )

        LOGGER.info("[LeakWatch] Detection mode: %s", self._startup_mode_label())

    @property
    def detection_mode(self) -> str:
        """Return audit-friendly detection mode."""
        regex_active = bool(self._patterns)
        nlp_active = self._nlp_loaded
        if self._nlp_configured and self._nlp_failed:
            return "regex_fallback"
        if regex_active and nlp_active:
            return "hybrid"
        if nlp_active:
            return "nlp"
        return "regex"

    def _startup_mode_label(self) -> str:
        if self.detection_mode == "hybrid":
            return "hybrid (regex + NLP)"
        if self.detection_mode == "nlp":
            return "nlp_only"
        if self.detection_mode == "regex_fallback":
            return "regex_fallback"
        return "regex_only"

    def detect(self, text: str) -> list[DetectedEntity]:
        if not text:
            return []

        entities: list[DetectedEntity] = []
        entities.extend(self._spacy_entities(text))
        entities.extend(self._regex_entities(text))
        return entities

    def _spacy_entities(self, text: str) -> Iterable[DetectedEntity]:
        if not self._nlp:
            return []
        doc = self._nlp(text[: self.config.max_doc_length])
        results: list[DetectedEntity] = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "ORG", "CARDINAL", "MONEY"}:
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label=ent.label_,
                        confidence=0.8,
                        text=ent.text,
                        span=Span(start=ent.start_char, end=ent.end_char),
                        detection_method="nlp",
                    )
                )
        return results

    def _regex_entities(self, text: str) -> Iterable[DetectedEntity]:
        results: list[DetectedEntity] = []
        for pattern in self._patterns:
            for match in pattern.pattern.finditer(text):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label=pattern.name,
                        confidence=0.95,
                        text=match.group(0),
                        span=Span(start=match.start(), end=match.end()),
                        mitigation=pattern.action,  # pre-select desired action
                        detection_method="regex",
                    )
                )
        return results
