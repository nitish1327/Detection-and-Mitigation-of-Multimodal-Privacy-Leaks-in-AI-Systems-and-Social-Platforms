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
    "home", "away", "busy", "well", "okay", "ok", "pleased", "excited", "proud",
    "at", "on", "in", "by", "for", "from", "with", "to", "a", "an", "the", "my", "your", "his", "her",
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

        spoken_pii = self._detect_spoken_pii(text)
        if spoken_pii:
            filtered_spoken: list[DetectedEntity] = []
            for sp in spoken_pii:
                if not sp.span:
                    continue
                exact_match = next(
                    (
                        ent for ent in entities
                        if ent.span and ent.span.start == sp.span.start and ent.span.end == sp.span.end
                        and ent.label.upper() not in {"CARDINAL", "NUMBERS", "ORG", "GPE", "DATE", "MISC", "NORP"}
                    ),
                    None,
                )
                if exact_match is None:
                    filtered_spoken.append(sp)

            retained_entities: list[DetectedEntity] = []
            for ent in entities:
                if ent.span:
                    is_subsumed = any(
                        sp.span and (
                            ((sp.span.start < ent.span.start and ent.span.end <= sp.span.end) or (sp.span.start <= ent.span.start and ent.span.end < sp.span.end))
                            or (
                                max(sp.span.start, ent.span.start) < min(sp.span.end, ent.span.end)
                                and ent.label.upper() in {"CARDINAL", "NUMBERS", "ORG", "GPE", "DATE", "MISC", "NORP"}
                            )
                        )
                        for sp in filtered_spoken
                    )
                    if is_subsumed:
                        continue
                retained_entities.append(ent)
            entities = retained_entities + filtered_spoken

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

    def _detect_pan_cards(self, text: str) -> list[DetectedEntity]:
        matches: list[DetectedEntity] = []
        p_pan = re.compile(
            r"\b([A-Za-z](?:\s*[A-Za-z]){4}\s*\d(?:\s*\d){3}\s*[A-Za-z])\b"
        )
        for m in p_pan.finditer(text):
            val = m.group(1)
            cleaned = re.sub(r"\s+", "", val)
            if len(cleaned) == 10 and cleaned[:5].isalpha() and cleaned[5:9].isdigit() and cleaned[9].isalpha():
                matches.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="pan",
                        confidence=0.98,
                        text=val,
                        span=Span(start=m.start(1), end=m.end(1)),
                        mitigation="mask",
                        detection_method="regex",
                        metadata={"locale": "en_IN"},
                    )
                )
        return matches

    def _detect_spoken_licenses(self, text: str) -> list[DetectedEntity]:
        matches: list[DetectedEntity] = []
        p_context = re.compile(
            r"\b(?:driving\s+licen[sc]e|licen[sc]e|dl)\b[^\w\n]*?(?:number|no|id|is|:|-|\s)*?\s*([A-Za-z](?:\s*[A-Za-z])(?:\s*[A-Za-z0-9]){10,20})\b",
            re.IGNORECASE,
        )
        for m in p_context.finditer(text):
            val = m.group(1)
            cleaned = re.sub(r"[\s.-]+", "", val)
            if any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
                matches.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="license",
                        confidence=0.95,
                        text=val,
                        span=Span(start=m.start(1), end=m.end(1)),
                        mitigation="replace",
                        detection_method="regex",
                        metadata={"locale": "en_IN"},
                    )
                )

        p_standalone = re.compile(
            r"\b([A-Z]{2}[-\s]?[0-9]{2}[-\s]?[0-9]{4}[-\s]?[0-9]{7})\b|"
            r"\b([A-Z]\s+[A-Z]\s+[0-9]\s+[0-9]\s+(?:[0-9]\s+){10,13}[0-9])\b",
            re.IGNORECASE,
        )
        for m in p_standalone.finditer(text):
            val = m.group(0)
            cleaned = re.sub(r"[\s.-]+", "", val)
            if 13 <= len(cleaned) <= 17 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
                if not any(
                    prev.span and m.start(0) >= prev.span.start and m.end(0) <= prev.span.end
                    for prev in matches
                ):
                    matches.append(
                        DetectedEntity(
                            modality=Modality.TEXT,
                            label="license",
                            confidence=0.95,
                            text=val,
                            span=Span(start=m.start(0), end=m.end(0)),
                            mitigation="replace",
                            detection_method="regex",
                            metadata={"locale": "en_IN"},
                        )
                    )
        return matches

    def _detect_spoken_passports(self, text: str, occupied: list[tuple[int, int]]) -> list[DetectedEntity]:
        matches: list[DetectedEntity] = []
        stopwords = {"is", "as", "us", "in", "it", "at", "to", "or", "so", "no", "if", "by", "my", "he", "we", "be"}

        # Contextual: "passport [no/number/id/is] U 6 6 2 4 4 0 2"
        p_context = re.compile(
            r"\b(?:passport)\b[^\w\n]*?(?:number|no|id|code|is|:|-|\s)*?\s*([A-Za-z](?:\s*[A-Za-z])?(?:\s*[-.]?\s*\d){6,8})\b",
            re.IGNORECASE,
        )
        for m in p_context.finditer(text):
            val = m.group(1)
            cleaned = re.sub(r"[\s.-]+", "", val)
            if re.match(r"^[A-Za-z]{1,2}\d{6,8}$", cleaned):
                matches.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="passport_number",
                        confidence=0.98,
                        text=val,
                        span=Span(start=m.start(1), end=m.end(1)),
                        mitigation="replace",
                        detection_method="regex",
                    )
                )

        # Standalone: 1-letter or 2-letter uppercase passports
        p_standalone = re.compile(
            r"\b([A-PR-WY]\s*\d(?:\s*\d){6,7})\b|\b([A-Z]\s*[A-Z]\s*\d(?:\s*\d){5,6})\b"
        )
        for m in p_standalone.finditer(text):
            s, e = m.start(0), m.end(0)
            if any(o_s <= s and e <= o_e for o_s, o_e in occupied):
                continue
            val = m.group(0)
            first_word = val.split()[0].lower()
            if first_word in stopwords:
                continue
            cleaned = re.sub(r"\s+", "", val)
            if len(cleaned) in {7, 8, 9} and not any(
                prev.span and s >= prev.span.start and e <= prev.span.end for prev in matches
            ):
                matches.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label="passport_number",
                        confidence=0.95,
                        text=val,
                        span=Span(start=s, end=e),
                        mitigation="replace",
                        detection_method="regex",
                    )
                )
        return matches

    def _detect_contextual_pii(self, text: str, occupied: list[tuple[int, int]]) -> list[DetectedEntity]:
        matches: list[DetectedEntity] = []
        patterns = [
            (
                r"\b(?:phone\s+number|mobile\s+number|contact\s+number|whatsapp\s+number|call\s+(?:me\s+)?at|reach\s+(?:me\s+)?at)\b[^\w\n]*?(?:is|:|-|\s)*?\s*(\+?[\d\s-]{7,18}\b)",
                "phone",
                "mask",
            ),
            (
                r"\b(?:aadhaar|aadhar|uid|vid)\s*(?:number|no|card|id)?\b[^\w\n]*?(?:is|:|-|\s)*?\s*([\d\s-]{10,18}\b)",
                "aadhar",
                "blur",
            ),
            (
                r"\b(?:pan\s+card|pan)\b[^\w\n]*?(?:number|no|id|code|is|:|-|\s)*?\s*([A-Za-z0-9\s-]{10,15}\b)",
                "pan",
                "mask",
            ),
            (
                r"\b(?:credit\s+card|debit\s+card|card\s+number)\s*(?:no|number)?\b[^\w\n]*?(?:is|:|-|\s)*?\s*([\d\s-]{13,23}\b)",
                "credit_card",
                "mask",
            ),
            (
                r"\b(?:account|bank\s+account|acc\s+no|a/c)\s*(?:number|no)?\b[^\w\n]*?(?:is|:|-|\s)*?\s*([\d\s-]{9,20}\b)",
                "numbers",
                "replace",
            ),
        ]
        for pattern_str, label, mitigation in patterns:
            for m in re.finditer(pattern_str, text, re.IGNORECASE):
                val = m.group(1).strip()
                s, e = m.start(1), m.end(1)
                if any(o_s <= s and e <= o_e for o_s, o_e in occupied):
                    continue
                cleaned = re.sub(r"[\s.-]+", "", val)
                if len(cleaned) < 6:
                    continue
                if label == "pan" and not (any(c.isalpha() for c in cleaned) and any(c.isdigit() for c in cleaned)):
                    continue
                if label == "phone" and sum(c.isdigit() for c in cleaned) < 7:
                    continue
                if label == "aadhar" and sum(c.isdigit() for c in cleaned) < 10:
                    continue
                if label == "credit_card" and sum(c.isdigit() for c in cleaned) < 13:
                    continue
                if label == "numbers" and sum(c.isdigit() for c in cleaned) < 7:
                    continue

                matches.append(
                    DetectedEntity(
                        modality=Modality.TEXT,
                        label=label,
                        confidence=0.96,
                        text=val,
                        span=Span(start=s, end=e),
                        mitigation=mitigation,
                        detection_method="regex",
                    )
                )
        return matches

    def _detect_spoken_numbers(self, text: str, occupied: list[tuple[int, int]]) -> list[DetectedEntity]:
        matches: list[DetectedEntity] = []
        p_spaced = re.compile(r"\b(?:\+?\d{1,3}[ -])?(?:\d[ -]?){6,19}\d\b")
        for m in p_spaced.finditer(text):
            s, e = m.start(0), m.end(0)
            if any(o_s <= s and e <= o_e for o_s, o_e in occupied):
                continue
            val = m.group(0)
            digits_only = re.sub(r"\D", "", val)
            n = len(digits_only)
            if n < 7:
                continue
            if n == 12 and (val.startswith("+91") or val.startswith("91")):
                label = "phone"
                mitigation = "mask"
            elif n == 10:
                label = "phone"
                mitigation = "mask"
            elif n == 12:
                label = "aadhar"
                mitigation = "blur"
            elif 13 <= n <= 19:
                label = "credit_card"
                mitigation = "mask"
            elif n >= 7:
                label = "numbers"
                mitigation = "replace"
            else:
                continue
            matches.append(
                DetectedEntity(
                    modality=Modality.TEXT,
                    label=label,
                    confidence=0.95,
                    text=val,
                    span=Span(start=s, end=e),
                    mitigation=mitigation,
                    detection_method="regex",
                )
            )
        return matches

    def _detect_spoken_digit_words(self, text: str) -> list[DetectedEntity]:
        matches: list[DetectedEntity] = []
        digit_map = {
            "zero": "0", "oh": "0",
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        }
        word_token_pattern = r"(?:(?:double|triple)\s+)?(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
        seq_pattern = re.compile(
            rf"\b({word_token_pattern}(?:\s+{word_token_pattern}){{6,}})\b",
            re.IGNORECASE,
        )
        for m in seq_pattern.finditer(text):
            matched_str = m.group(1)
            tokens = matched_str.lower().split()
            digit_count = 0
            multiplier = 1
            for tok in tokens:
                if tok == "double":
                    multiplier = 2
                elif tok == "triple":
                    multiplier = 3
                elif tok in digit_map:
                    digit_count += multiplier
                    multiplier = 1

            if digit_count == 10:
                label = "phone"
                mitigation = "mask"
            elif digit_count == 12:
                label = "aadhar"
                mitigation = "blur"
            elif 13 <= digit_count <= 19:
                label = "credit_card"
                mitigation = "mask"
            else:
                label = "numbers"
                mitigation = "replace"

            matches.append(
                DetectedEntity(
                    modality=Modality.TEXT,
                    label=label,
                    confidence=0.92,
                    text=matched_str,
                    span=Span(start=m.start(1), end=m.end(1)),
                    mitigation=mitigation,
                    detection_method="heuristic",
                )
            )
        return matches

    def _detect_spoken_pii(self, text: str) -> list[DetectedEntity]:
        if not text:
            return []

        all_spoken: list[DetectedEntity] = []

        # 1. Spoken emails
        all_spoken.extend(self._detect_spoken_emails(text))

        # 2. Indian PAN cards
        pan_entities = self._detect_pan_cards(text)
        all_spoken.extend(pan_entities)

        # 3. Driving licenses
        license_entities = self._detect_spoken_licenses(text)
        all_spoken.extend(license_entities)

        occupied = [
            (e.span.start, e.span.end) for e in all_spoken if e.span
        ]

        # 4. Passports (aware of already occupied DL/PAN spans)
        pass_entities = self._detect_spoken_passports(text, occupied)
        all_spoken.extend(pass_entities)

        occupied = [
            (e.span.start, e.span.end) for e in all_spoken if e.span
        ]

        # 5. Contextual PII (aware of occupied spans)
        context_entities = self._detect_contextual_pii(text, occupied)
        all_spoken.extend(context_entities)

        occupied = [
            (e.span.start, e.span.end) for e in all_spoken if e.span
        ]

        # 6. Spaced numeric sequences (phones, aadhaar, credit cards, numbers)
        num_entities = self._detect_spoken_numbers(text, occupied)
        all_spoken.extend(num_entities)

        # 7. Spoken digit words ("nine eight seven...")
        word_entities = self._detect_spoken_digit_words(text)
        all_spoken.extend(word_entities)

        if not all_spoken:
            return []

        # Merge overlapping spoken matches preferring the longest span
        valid = [e for e in all_spoken if e.span]
        valid.sort(key=lambda e: (e.span.start, -(e.span.end - e.span.start)))
        merged: list[DetectedEntity] = []
        for ent in valid:
            assert ent.span
            if not merged:
                merged.append(ent)
                continue
            prev = merged[-1]
            assert prev.span
            if ent.span.start >= prev.span.start and ent.span.end <= prev.span.end:
                # Fully inside previous; skip
                continue
            if ent.span.start < prev.span.end:
                # Partially overlapping: prefer higher confidence
                if ent.confidence > prev.confidence:
                    merged[-1] = ent
                continue
            merged.append(ent)

        return merged

