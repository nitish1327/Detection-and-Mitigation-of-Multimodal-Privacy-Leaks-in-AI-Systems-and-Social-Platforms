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

# Common non-person words that can follow conversational introductions (e.g., "i am ready")
NON_PERSON_INTRO_WORDS = {
    "here", "there", "back", "done", "fine", "good", "sorry", "happy", "writing",
    "looking", "working", "interested", "sure", "not", "also", "just", "now",
    "available", "using", "online", "testing", "trying", "ready", "new", "glad",
    "home", "away", "busy", "well", "okay", "ok", "pleased", "excited", "proud"
}

# Known given names commonly misclassified as GPE or missed by Western NLP models
FEMALE_GIVEN_NAMES = {
    "ananya", "sonika", "priya", "sneha", "aditi", "neha", "pooja", "anjali",
    "kavita", "meera", "swati", "divya", "tanvi", "riya", "shruti", "shreya",
    "anushka", "diya", "radha", "simran", "kriti", "nidhi", "pallavi", "rashmi",
    "bhavna", "rekha", "geetha", "lakshmi", "anu", "gowri", "tanu", "varsha"
}

MALE_GIVEN_NAMES = {
    "pranav", "nitish", "rahul", "aarav", "rohit", "amit", "deepak", "vikram",
    "suresh", "ramesh", "manish", "sunil", "raj", "arjun", "varun", "karan",
    "ishaan", "krishna", "madhav", "sai", "sanjay", "vijay", "ajay", "alok",
    "ashok", "vinay", "harsh", "ayush", "kartik", "chirag", "rohan"
}

KNOWN_GIVEN_NAMES = FEMALE_GIVEN_NAMES | MALE_GIVEN_NAMES


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

        spoken_emails = self._detect_spoken_emails(text)
        if spoken_emails:
            retained_entities: list[DetectedEntity] = []
            for ent in entities:
                if ent.span:
                    is_subsumed = any(
                        se.span and se.span.start <= ent.span.start and ent.span.end <= se.span.end
                        for se in spoken_emails
                    )
                    if is_subsumed:
                        continue
                retained_entities.append(ent)
            entities = retained_entities + spoken_emails

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

        # 1. Reclassify entities tagged as GPE/ORG/MISC that are known given names to PERSON
        for ent in filtered_entities:
            clean_token = ent.text.strip().lower()
            if clean_token in KNOWN_GIVEN_NAMES and ent.label.upper() in {"GPE", "ORG", "MISC", "NORP"}:
                ent.label = "PERSON"

        # 2. Reclassify or detect names from conversational self-introductions:
        # e.g., "hello i am ananya", "my name is ananya", "this is pranav", "call me sonika"
        intro_regex = re.compile(
            r"\b(?:i\s+am|i'm|im|my\s+name\s+is|myself|this\s+is|call\s+me)\s+([A-Za-z][A-Za-z'\-]{1,25})\b",
            re.IGNORECASE,
        )
        for match in intro_regex.finditer(text):
            candidate = match.group(1)
            cand_lower = candidate.lower()
            if cand_lower in NON_PERSON_INTRO_WORDS or self._is_field_label(cand_lower):
                continue
            cand_start = match.start(1)
            cand_end = match.end(1)

            matched_ent = None
            for ent in filtered_entities:
                if ent.span and not (ent.span.end <= cand_start or ent.span.start >= cand_end):
                    matched_ent = ent
                    break

            if matched_ent:
                if matched_ent.label.upper() in {"GPE", "ORG", "MISC", "NORP"}:
                    matched_ent.label = "PERSON"
            else:
                filtered_entities.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="PERSON",
                        confidence=0.85,
                        text=candidate,
                        span=Span(start=cand_start, end=cand_end),
                        detection_method="heuristic",
                    )
                )

        # 3. Catch standalone known given names not yet covered by an entity span
        for name in KNOWN_GIVEN_NAMES:
            for m in re.finditer(rf"\b({re.escape(name)})\b", text, re.IGNORECASE):
                n_start, n_end = m.start(1), m.end(1)
                is_covered = any(
                    ent.span and not (ent.span.end <= n_start or ent.span.start >= n_end)
                    for ent in filtered_entities
                )
                if not is_covered:
                    filtered_entities.append(
                        DetectedEntity(
                            modality=Modality.TEXT,
                            label="PERSON",
                            confidence=0.85,
                            text=m.group(1),
                            span=Span(start=n_start, end=n_end),
                            detection_method="heuristic",
                            metadata={"locale": "en_IN"},
                        )
                    )

        # 4. Hinglish, regional vernacular, and Indian PII patterns
        hinglish_entities = self._detect_hinglish_entities(text, filtered_entities)
        filtered_entities.extend(hinglish_entities)

        # Primitive document-level gender detection for name replacement
        text_lower = text.lower()
        male_score = sum(text_lower.count(w) for w in [" he ", " his ", " him ", " male "])
        female_score = sum(text_lower.count(w) for w in [" she ", " her ", " hers ", " female "])
        
        doc_gender = None
        if male_score > female_score and male_score > 0:
            doc_gender = "male"
        elif female_score > male_score and female_score > 0:
            doc_gender = "female"

        # Apply gender metadata (specific name gender takes precedence over document context)
        for ent in filtered_entities:
            if ent.label in {"PERSON", "NAME"}:
                token_lower = ent.text.strip().lower()
                if token_lower in FEMALE_GIVEN_NAMES:
                    ent.metadata["gender"] = "female"
                elif token_lower in MALE_GIVEN_NAMES:
                    ent.metadata["gender"] = "male"
                elif doc_gender:
                    ent.metadata["gender"] = doc_gender

        return filtered_entities

    def _detect_hinglish_entities(
        self, text: str, existing_entities: list[DetectedEntity]
    ) -> list[DetectedEntity]:
        """Detect Hinglish, transliterated vernacular, and regional Indian PII patterns."""
        results: list[DetectedEntity] = []

        def _is_covered(start: int, end: int) -> bool:
            return any(
                ent.span and not (ent.span.end <= start or ent.span.start >= end)
                for ent in existing_entities + results
            )

        # 1. Transliterated / Hinglish Introductions: "Mera naam Rahul Verma hai" / "Mera name ..."
        hinglish_name_pattern = re.compile(
            r"\b(?:mera\s+naam|meri\s+naam|humara\s+naam|naam\s+hai|name\s+hai|shri|smt|kumari)\s+([A-Z][a-zA-Z'\-]{1,25}(?:\s+[A-Z][a-zA-Z'\-]{1,25})?)\b",
            re.IGNORECASE,
        )
        for m in hinglish_name_pattern.finditer(text):
            cand = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if cand.lower() not in NON_PERSON_INTRO_WORDS and not _is_covered(start, end):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="PERSON",
                        confidence=0.90,
                        text=cand,
                        span=Span(start=start, end=end),
                        detection_method="heuristic",
                        metadata={"locale": "en_IN"},
                    )
                )

        # 2. Hinglish Phone: "Mera phone number 9876543210 hai" / "Sampark sankhya: ..."
        hinglish_phone_pattern = re.compile(
            r"\b(?:phone\s+number|mobile\s+number|mera\s+number|sampark\s+karein|durvash)\s*(?:hai|no|number)?\s*[:=-]?\s*(\+?91[\s-]?[6-9]\d{9}|[6-9]\d{9})\b",
            re.IGNORECASE,
        )
        for m in hinglish_phone_pattern.finditer(text):
            num = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if not _is_covered(start, end):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="phone_numbers",
                        confidence=0.95,
                        text=num,
                        span=Span(start=start, end=end),
                        mitigation="mask",
                        detection_method="regex",
                        metadata={"locale": "en_IN"},
                    )
                )

        # 3. Hinglish Address / Residence: "Mera pata 123 MG Road, Indiranagar hai"
        hinglish_addr_pattern = re.compile(
            r"\b(?:mera\s+pata|ghar\s+ka\s+pata|sthit\s+hai|rehte\s+hain|nivas)\s*(?:hai)?\s*[:=-]?\s*([A-Za-z0-9\s,/\-]{8,60})\s+(?:hai|mein|me|sthit)\b",
            re.IGNORECASE,
        )
        for m in hinglish_addr_pattern.finditer(text):
            addr = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if not _is_covered(start, end):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="GPE",
                        confidence=0.85,
                        text=addr,
                        span=Span(start=start, end=end),
                        detection_method="heuristic",
                        metadata={"locale": "en_IN"},
                    )
                )

        # 4. Devanagari script detection (Hindi vernacular)
        devanagari_name_pattern = re.compile(
            r"(?:मेरा\s+नाम|नाम\s+है)\s+([\u0900-\u097F]+(?:\s+[\u0900-\u097F]+)?)"
        )
        for m in devanagari_name_pattern.finditer(text):
            val = m.group(1).strip()
            start, end = m.start(1), m.end(1)
            if not _is_covered(start, end):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="PERSON",
                        confidence=0.92,
                        text=val,
                        span=Span(start=start, end=end),
                        detection_method="heuristic",
                        metadata={"locale": "hi_IN"},
                    )
                )

        # 5. Indian PAN card pattern: 5 letters, 4 digits, 1 letter
        pan_pattern = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
        for m in pan_pattern.finditer(text):
            val = m.group(1)
            start, end = m.start(1), m.end(1)
            if not _is_covered(start, end):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="PAN",
                        confidence=0.98,
                        text=val,
                        span=Span(start=start, end=end),
                        mitigation="mask",
                        detection_method="regex",
                        metadata={"locale": "en_IN"},
                    )
                )

        # 6. Indian Vehicle Registration: e.g. KA 01 AB 1234, DL-3C-AB-1234
        vehicle_pattern = re.compile(r"\b([A-Z]{2}[ -]?[0-9]{1,2}[ -]?[A-Z]{1,3}[ -]?[0-9]{4})\b")
        for m in vehicle_pattern.finditer(text):
            val = m.group(1)
            start, end = m.start(1), m.end(1)
            if any(c.isdigit() for c in val) and any(c.isalpha() for c in val) and not _is_covered(start, end):
                results.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="LICENSE_PLATE",
                        confidence=0.92,
                        text=val,
                        span=Span(start=start, end=end),
                        mitigation="mask",
                        detection_method="regex",
                        metadata={"locale": "en_IN"},
                    )
                )

        return results

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

    def _detect_spoken_emails(self, text: str) -> list[DetectedEntity]:
        """Detect spoken, phonetically transcribed, or conversational email addresses."""
        if not text:
            return []

        common_prepositions = {
            "out", "at", "to", "for", "in", "on", "by", "with", "from",
            "reach", "call", "contact", "write", "send", "me", "us", "him", "her", "them", "is", "and", "or",
        }

        raw_matches: list[tuple[int, int, str]] = []

        # 1. Contextual email intro phrase: "email id is ...", "email is ...", "mail id is ...", "email: ..."
        # Matches: "my email ID is me on uradyadheradegmail.com", "email is user@domain.com", etc.
        p_intro = re.compile(
            r'\b(?:my\s+)?(?:email|e-mail|mail)\s*(?:id|address)?\s*(?:is|:|-)\s+'
            r'([a-zA-Z0-9._%+-]+(?:\s+[a-zA-Z0-9._%+-]+)*?'
            r'(?:@[a-zA-Z0-9.-]+|\.[a-zA-Z]{2,}|adherade|attherate|at\s*the\s*rate|\bat\b\s+[a-zA-Z0-9.-]+|gmail|yahoo|outlook|hotmail)'
            r'[^\s,.]*(?:\.[a-zA-Z]{2,})?)',
            re.IGNORECASE,
        )
        for m in p_intro.finditer(text):
            val = m.group(1).strip()
            if val:
                raw_matches.append((m.start(1), m.end(1), val))

        # 2. Spoken 'at the rate' or phonetic 'adherade' / 'attherate'
        # e.g., "john at the rate gmail dot com", "alice adherade yahoo.com"
        p_spoken_rate = re.compile(
            r'\b([a-zA-Z0-9._%+-]+)\s+(?:at\s+the\s+rate|at\s+rate|adherade|attherate)\s+'
            r'([a-zA-Z0-9.-]+(?:\s*(?:\.|\s+dot\s+)\s*)[a-zA-Z]{2,})\b',
            re.IGNORECASE,
        )
        for m in p_spoken_rate.finditer(text):
            if m.group(1).lower() not in common_prepositions:
                raw_matches.append((m.start(0), m.end(0), m.group(0)))

        # 3. Spoken 'at' with a known mail provider or domain
        # e.g., "john.doe at gmail dot com", "alice at yahoo.com"
        p_spoken_at = re.compile(
            r'\b([a-zA-Z0-9._%+-]+)\s+at\s+'
            r'((?:gmail|yahoo|outlook|hotmail|icloud|protonmail|rediffmail)(?:\s*(?:\.|\s*dot\s*)\s*[a-zA-Z]{2,})?)\b',
            re.IGNORECASE,
        )
        for m in p_spoken_at.finditer(text):
            if m.group(1).lower() not in common_prepositions:
                raw_matches.append((m.start(0), m.end(0), m.group(0)))

        # 4. Phonetic / domain-concatenated token ending in domain
        # e.g., "uradyadheradegmail.com", "testadheradeyahoo.com", "userattherategmail.com"
        p_phonetic_tok = re.compile(
            r'\b[a-zA-Z0-9._%+-]*(?:adherade|attherate)[a-zA-Z0-9._%+-]*'
            r'(?:gmail|yahoo|outlook|hotmail|icloud|protonmail|rediffmail|[a-zA-Z0-9-]+)\.[a-zA-Z]{2,}\b',
            re.IGNORECASE,
        )
        for m in p_phonetic_tok.finditer(text):
            raw_matches.append((m.start(0), m.end(0), m.group(0)))

        # 5. Token ending in email provider and TLD without @ (Whisper transcription artifact)
        # e.g. "uradygmail.com"
        p_provider_tok = re.compile(
            r'\b[a-zA-Z0-9._%+-]+(?:gmail|yahoo|outlook|hotmail|icloud|protonmail|rediffmail)\.(?:com|org|net|in|co|edu)\b',
            re.IGNORECASE,
        )
        for m in p_provider_tok.finditer(text):
            raw_matches.append((m.start(0), m.end(0), m.group(0)))

        if not raw_matches:
            return []

        # Merge overlapping matches preferring the encompassing (longest) span
        raw_matches.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged: list[tuple[int, int, str]] = []
        for start, end, val in raw_matches:
            if not merged:
                merged.append((start, end, val))
                continue
            prev_start, prev_end, prev_val = merged[-1]
            if start >= prev_start and end <= prev_end:
                # Fully contained in previous longer match; skip
                continue
            if start < prev_end:
                # Partially overlapping: take outer envelope
                new_end = max(prev_end, end)
                merged[-1] = (prev_start, new_end, text[prev_start:new_end])
            else:
                merged.append((start, end, val))

        entities: list[DetectedEntity] = []
        for start, end, val in merged:
            entities.append(
                DetectedEntity(
                    modality=Modality.TEXT,
                    label="email",
                    confidence=0.95,
                    text=val,
                    span=Span(start=start, end=end),
                    mitigation="mask",
                    detection_method="regex",
                )
            )
        return entities

