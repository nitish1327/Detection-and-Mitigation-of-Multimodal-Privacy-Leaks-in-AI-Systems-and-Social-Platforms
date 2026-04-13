from typing import Dict, List, Tuple

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
        self.faker = Faker(config.faker_locale) if Faker else None
        self._faker_cache: Dict[str, str] = {}

    def mitigate(self, text: str, entities: list[DetectedEntity]) -> Tuple[str, List[DetectedEntity]]:
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
            replacement = self._replacement_for(entity)
            
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

    def _replacement_for(self, entity: DetectedEntity) -> str:
        # Check static replacements first
        if entity.text and entity.text in self.config.static_replacements:
            return self.config.static_replacements[entity.text]

        style = self.config.mask_style
        label = entity.label.upper()

        if style == "faker" and self.faker and entity.text:
            if entity.text not in self._faker_cache:
                self._faker_cache[entity.text] = self._generate_faker_value(entity)
            return self._faker_cache[entity.text]

        if style == "asterisks":
            return "*" * (len(entity.text or "REDACTED"))
        if style == "synthetic":
            return f"<{label}>"
        return f"[REDACTED:{label}]"

    def _generate_faker_value(self, entity: DetectedEntity) -> str:
        """Generate a realistic fake value based on the entity label."""
        if not self.faker:
            return f"<{entity.label.upper()}>"

        label = entity.label.upper()
        
        # Identity & Contact
        if label in {"PERSON", "NAME"}:
            gender = entity.metadata.get("gender")
            if gender == "male":
                return self.faker.name_male()
            elif gender == "female":
                return self.faker.name_female()
            return self.faker.name()
        if label in {"EMAIL", "EMAIL_ADDRESS"}:
            return self.faker.email()
        if label in {"PHONE", "PHONE_NUMBER", "TELEPHONE"}:
            return self.faker.phone_number()
        if label == "USER_NAME":
            return self.faker.user_name()
        if label == "PASSWORD":
            return self.faker.password()
            
        # Location & Network
        if label in {"ORG", "ORGANIZATION", "COMPANY"}:
            return self.faker.company()
        if label in {"GPE", "LOCATION", "CITY"}:
            return self.faker.city()
        if label in {"ADDRESS", "STREET_ADDRESS"}:
            return self.faker.address().replace("\n", ", ")
        if label == "ZIP_CODE":
            return self.faker.zipcode()
        if label in {"IP_ADDRESS", "IPV4", "IPV6"}:
            return self.faker.ipv4()
        if label == "MAC_ADDRESS":
            return self.faker.mac_address()
        if label == "URL":
            return self.faker.url()

        # Financial & IDs
        if label in {"CREDIT_CARD", "CREDIT_CARD_NUMBER"}:
            return self.faker.credit_card_number()
        if label == "IBAN":
            return self.faker.iban()
        if label == "SWIFT":
            return self.faker.swift()
        if label == "SSN":
            return self.faker.ssn()
        if label == "PASSPORT":
            return self.faker.passport_number()
            
        # Misc
        if label == "DATE":
            return str(self.faker.date())
        if label == "CARDINAL":
            return str(self.faker.random_int(min=0, max=999999))
        if label == "LICENSE_PLATE":
            return self.faker.license_plate()
        if label == "LICENSE":
            return self.faker.bothify("??## #### #######", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        return f"<{label}>"

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
