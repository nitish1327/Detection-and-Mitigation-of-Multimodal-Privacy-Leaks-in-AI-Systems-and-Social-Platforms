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
        self._patterns = []
        for item in config.regex_entities:
            flags = 0 if item.case_sensitive else re.IGNORECASE
            self._patterns.append(
                _RegexPattern(item.name, re.compile(item.pattern, flags), item.action)
            )
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

        # Document field labels that are structural text, NOT PII values
        self._field_labels = {
            "name", "full name", "surname", "first name", "last name",
            "license no", "license no.", "licence no", "licence no.",
            "license number", "licence number", "dl no", "dl no.",
            "authorization to drive", "auth to drive",
            "date of issue", "doi", "issue date",
            "date of expiry", "doe", "expiry date", "valid till", "valid upto",
            "dob", "date of birth", "d.o.b", "d.o.b.",
            "s/w/d", "s/d/w", "s/o", "d/o", "w/o", "c/o",
            "blood group", "blood grp", "bg",
            "sex", "gender",
            "permanent address", "present address", "address",
            "place of issue", "issuing authority", "issued by",
            "nationality", "country",
            "signature", "photograph", "photo",
            "father's name", "fathers name", "husband's name", "husbands name",
            "mother's name", "mothers name",
            "aadhaar", "aadhaar no", "aadhaar number", "uid", "vid",
            "government of india", "govt of india",
            "passport no", "passport number",
            "place of birth", "place of issue",
            "pan", "pan no", "pan number",
            "income tax department",
            "sr no", "sr no.", "serial no", "serial number",
            "reg no", "reg no.", "registration no",
            "class of vehicle", "cov", "type of vehicle",
            "transport", "non-transport", "non transport",
        }

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

    def _is_field_label(self, text: str) -> bool:
        """Return True if the text is a document field label, not a PII value."""
        cleaned = text.strip().lower().rstrip(":- .")
        return cleaned in self._field_labels

    def detect(self, text: str) -> list[DetectedEntity]:
        if not text:
            return []

        entities: list[DetectedEntity] = []
        entities.extend(self._spacy_entities(text))
        entities.extend(self._regex_entities(text))

        # Filter out entities that are part of an allowlisted term (case-insensitive)
        allowlist_lower = [term.lower() for term in self.config.allowlist]
        filtered_entities = [
            ent for ent in entities 
            if not any(ent.text.lower() in term for term in allowlist_lower)
        ]

        # Filter out document field labels — only PII values should be mitigated
        filtered_entities = [
            ent for ent in filtered_entities
            if not (ent.text and self._is_field_label(ent.text))
        ]

        # Filter out false positive names/persons that contain document keywords
        non_name_words = {
            "university", "license", "driving", "government", "india", 
            "blood", "group", "card", "issue", "expiry", "date", 
            "birth", "class", "vehicle", "signature", "photo", "thumb",
            "impression", "holder", "authority", "union", "territory",
            "state", "transport"
        }
        filtered_entities = [
            ent for ent in filtered_entities
            if not (
                ent.label.upper() in {"PERSON", "NAME"} and 
                ent.text and 
                any(w in ent.text.lower().split() for w in non_name_words)
            )
        ]

        # Primitive document-level gender detection for name replacement
        text_lower = text.lower()
        male_score = sum(text_lower.count(w) for w in [" he ", " his ", " him ", " male "])
        female_score = sum(text_lower.count(w) for w in [" she ", " her ", " hers ", " female "])
        
        doc_gender = None
        if male_score > female_score and male_score > 0:
            doc_gender = "male"
        elif female_score > male_score and female_score > 0:
            doc_gender = "female"

        if doc_gender:
            for ent in filtered_entities:
                if ent.label in {"PERSON", "NAME"}:
                    ent.metadata["gender"] = doc_gender

        return filtered_entities

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
