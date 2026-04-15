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
from ..utils.types import DetectedEntity, Modality, Span


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
        self._patterns = [
            _RegexPattern(item.name, re.compile(item.pattern, re.IGNORECASE), item.action)
            for item in config.regex_entities
        ]
        if config.enable_spacy and spacy is not None:
            try:
                self._nlp = spacy.load(model)
            except OSError:
                self._nlp = None

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
