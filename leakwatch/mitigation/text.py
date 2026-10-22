from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple

try:
    from faker import Faker
except ImportError:
    Faker = None  # type: ignore

from ..utils.config import TextConfig
from ..utils.types import DetectedEntity, Span


class TextMitigator:
    """Apply masking or replacement for sensitive text spans."""

    def __init__(self, config: TextConfig) -> None:
        self.config = config
        self._fakers: Dict[str, Any] = {}
        if Faker:
            default_loc = config.faker_locale or "en_US"
            try:
                self._fakers[default_loc] = Faker(default_loc)
            except Exception:
                self._fakers[default_loc] = Faker()
            self.faker = self._fakers[default_loc]
        else:
            self.faker = None
        self._faker_cache: Dict[str, str] = {}

    def _get_faker_for_entity(self, entity: DetectedEntity) -> Any:
        if not Faker:
            return None
        target_locale = entity.metadata.get("locale")
        if not target_locale:
            text_str = str(entity.text or "")
            label_upper = entity.label.upper()
            if (
                label_upper in {"AADHAR", "AADHAAR", "PAN", "VOTER_ID", "LICENSE_PLATE"}
                or text_str.startswith("+91")
                or any(ind in text_str.lower() for ind in [
                    "bangalore", "bengaluru", "delhi", "mumbai", "india", "pune",
                    "karnataka", "hyderabad", "chennai", "kolkata"
                ])
                or re.search(r"\b[6-9]\d{9}\b", text_str)
            ):
                target_locale = "en_IN"
            else:
                target_locale = self.config.faker_locale or "en_US"

        if target_locale not in self._fakers:
            try:
                self._fakers[target_locale] = Faker(target_locale)
            except Exception:
                self._fakers[target_locale] = self.faker or Faker()
        return self._fakers[target_locale]

    def mitigate(
        self,
        text: str,
        entities: list[DetectedEntity],
        *,
        force_synthetic: bool = False,
    ) -> Tuple[str, List[DetectedEntity]]:
        if not text or not entities:
            return text, entities

        sanitized = text
        
        # Merge overlapping spans to prevent corrupted output
        merged_entities = self._merge_overlapping_spans(entities)
        
        # Sort spans ASCENDING to accurately track index shifts
        sortable = [e for e in merged_entities if e.span]
        sortable.sort(key=lambda ent: ent.span.start if ent.span else 0)

        updated_entities: list[DetectedEntity] = []
        total_shift = 0

        for entity in sortable:
            assert entity.span  # for type checkers
            orig_start, orig_end = entity.span.start, entity.span.end
            
            current_start = orig_start + total_shift
            current_end = orig_end + total_shift
            
            original_snippet = text[orig_start:orig_end]
            replacement = self._replacement_for(entity, force_synthetic=force_synthetic)
            
            # Update the evolving sanitized string
            sanitized = sanitized[:current_start] + replacement + sanitized[current_end:]
            
            new_entity = entity.model_copy(
                update={
                    "text": replacement,
                    "span": Span(start=current_start, end=current_start + len(replacement)),
                    "mitigation": self._action_for(entity),
                    "explanation": original_snippet,
                }
            )
            updated_entities.append(new_entity)
            
            # Track the shift caused by length difference
            total_shift += len(replacement) - (orig_end - orig_start)

        # Append non-span entities without modification
        updated_entities.extend(ent for ent in entities if not ent.span)
        return sanitized, updated_entities

    def _action_for(self, entity: DetectedEntity) -> str:
        if entity.mitigation != "none":
            return entity.mitigation
        return "mask" if self.config.mask_style not in {"synthetic", "faker"} else "replace"

    def _replacement_for(self, entity: DetectedEntity, *, force_synthetic: bool = False) -> str:
        # Check static replacements first
        if not force_synthetic and entity.text and entity.text in self.config.static_replacements:
            return self.config.static_replacements[entity.text]

        style = self.config.mask_style
        label = entity.label.upper()

        if (force_synthetic or style in {"faker", "synthetic"}) and entity.text:
            if entity.text not in self._faker_cache:
                self._faker_cache[entity.text] = self._generate_faker_value(entity)
            return self._faker_cache[entity.text]

        if style == "asterisks":
            return "*" * (len(entity.text or "REDACTED"))
        if style == "brackets":
            return f"[REDACTED:{label}]"
        return self._generate_faker_value(entity)

    def _generate_faker_value(self, entity: DetectedEntity) -> str:
        """Generate a realistic fake value based on the entity label and dynamically routed locale."""
        label = entity.label.upper()
        faker = self._get_faker_for_entity(entity) or self.faker

        # Try dynamic Faker generation if available
        if faker:
            try:
                # Identity & Contact
                if "PERSON" in label or "NAME" in label:
                    gender = entity.metadata.get("gender")
                    if gender == "male":
                        return faker.name_male()
                    elif gender == "female":
                        return faker.name_female()
                    return faker.name()
                if "EMAIL" in label:
                    return faker.email()
                if "PHONE" in label or "TELEPHONE" in label:
                    return faker.phone_number()
                if "USER_NAME" in label:
                    return faker.user_name()
                if "PASSWORD" in label:
                    return faker.password()
                    
                # Location & Network
                if "ORG" in label or "COMPANY" in label:
                    return faker.company()
                if "GPE" in label or "LOCATION" in label or "CITY" in label:
                    return faker.city()
                if "ADDRESS" in label or "STREET" in label:
                    return faker.address().replace("\n", ", ")
                if "ZIP" in label or "PIN" in label or "POSTCODE" in label:
                    return faker.postcode() if hasattr(faker, "postcode") else faker.zipcode()
                if "IP" in label or "IPV4" in label or "IPV6" in label:
                    return faker.ipv4()
                if "MAC" in label:
                    return faker.mac_address()
                if "URL" in label:
                    return faker.url()

                # Financial & IDs
                if "CREDIT" in label or "CARD_NUMBER" in label:
                    return faker.credit_card_number()
                if "IBAN" in label:
                    return faker.iban()
                if "SWIFT" in label:
                    return faker.swift()
                if "SSN" in label or "AADHAR" in label or "PAN" in label:
                    return faker.ssn() if hasattr(faker, "ssn") else "4521 8963 1124"
                if "PASSPORT" in label:
                    return faker.passport_number() if hasattr(faker, "passport_number") else "A1234567"
                    
                # Misc
                if "DATE" in label:
                    return str(faker.date())
                if "CARDINAL" in label or "NUMBER" in label:
                    return str(faker.random_int(min=100000, max=999999))
                if "LICENSE" in label:
                    if "PLATE" in label:
                        return faker.license_plate() if hasattr(faker, "license_plate") else "KA 01 AB 1234"
            except Exception:
                pass

        # Robust realistic fallback pool if faker is missing or raises an exception
        is_indian = entity.metadata.get("locale") == "en_IN" or "AADHAR" in label or "PAN" in label
        if "PERSON" in label or "NAME" in label:
            return "Raj Kumar" if is_indian else "John Smith"
        if "EMAIL" in label:
            return "raj.kumar@example.in" if is_indian else "john.smith@example.com"
        if "PHONE" in label or "TELEPHONE" in label:
            return "+91 98765 43210" if is_indian else "+1-415-555-0199"
        if "ADDRESS" in label or "STREET" in label:
            return "123 MG Road, Bengaluru" if is_indian else "742 Evergreen Terrace, Springfield"
        if "ORG" in label or "COMPANY" in label:
            return "TechCorp Global"
        if "CREDIT" in label or "CARD" in label:
            return "4532 **** **** 8821"
        if "ID" in label or "NUMBER" in label:
            return "ID556759"
        return "CONFIDENTIAL"

    def _merge_overlapping_spans(self, entities: list[DetectedEntity]) -> list[DetectedEntity]:
        """Merge overlapping or nested entity spans to prevent mitigation corruption."""
        if not entities:
            return []

        # Separate items with and without spans
        with_span = sorted(
            [e for e in entities if e.span], 
            key=lambda e: (e.span.start if e.span else 0, -(e.span.end if e.span else 0))
        )
        without_span = [e for e in entities if not e.span]

        if not with_span:
            return without_span

        merged: list[DetectedEntity] = []
        current = with_span[0]

        for next_entity in with_span[1:]:
            c_span = current.span
            n_span = next_entity.span
            assert c_span and n_span

            # If they overlap
            if n_span.start < c_span.end:
                # Merge: take the union of the spans
                new_end = max(c_span.end, n_span.end)
                current.span = Span(start=c_span.start, end=new_end)
                # optionally combine labels
                if current.label != next_entity.label:
                    current.label = f"{current.label}_{next_entity.label}"
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged + without_span
