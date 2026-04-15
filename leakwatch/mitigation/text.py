"""Text mitigation strategies."""

from __future__ import annotations

from typing import Dict, List, Tuple, Callable
import faker

from ..utils.config import TextConfig
from ..utils.types import DetectedEntity, Span


class TextMitigator:
    """Apply masking or replacement for sensitive text spans."""

    def __init__(self, config: TextConfig) -> None:
        self.config = config
        self.fake = faker.Faker()
        self._faker_map: Dict[str, Callable] = {
            "PERSON": self.fake.name,
            "ORG": self.fake.company,
            "GPE": self.fake.city,
            "DATE": lambda: self.fake.date_between(start_date="-1y", end_date="today").strftime("%B %d, %Y"),
            "ADDRESS": self.fake.address,
            "PHONE": self.fake.phone_number,
            "EMAIL": self.fake.email,
        }

    def mitigate(self, text: str, entities: list[DetectedEntity]) -> Tuple[str, List[DetectedEntity]]:
        if not text or not entities:
            return text, entities

        sanitized = text
        updated_entities: list[DetectedEntity] = []

        # Sort spans descending to avoid offset issues when replacing substrings.
        sortable = [e for e in entities if e.span]
        sortable.sort(key=lambda ent: ent.span.start if ent.span else -1, reverse=True)

        for entity in sortable:
            assert entity.span  # for type checkers
            start, end = entity.span.start, entity.span.end
            original_snippet = sanitized[start:end]
            replacement = self._replacement_for(entity)
            sanitized = sanitized[:start] + replacement + sanitized[end:]
            new_entity = entity.model_copy(
                update={
                    "text": replacement,
                    "span": Span(start=start, end=start + len(replacement)),
                    "mitigation": self._action_for(entity),
                    "explanation": original_snippet,
                }
            )
            updated_entities.append(new_entity)

        # Append non-span entities without modification
        updated_entities.extend(ent for ent in entities if not ent.span)
        return sanitized, list(reversed(updated_entities))

    def _action_for(self, entity: DetectedEntity) -> str:
        if entity.mitigation != "none":
            return entity.mitigation
        return "mask" if self.config.mask_style != "synthetic" else "replace"

    def _replacement_for(self, entity: DetectedEntity) -> str:
        style = self.config.mask_style
        label = entity.label.upper()
        if style == "asterisks":
            return "*" * (len(entity.text or "REDACTED"))
        if style == "synthetic":
            return f"<{label}>"
        if style == "faker":
            if label in self._faker_map:
                return self._faker_map[label]()
            return f"[FAKE:{label}]"
        return f"[REDACTED:{label}]"
