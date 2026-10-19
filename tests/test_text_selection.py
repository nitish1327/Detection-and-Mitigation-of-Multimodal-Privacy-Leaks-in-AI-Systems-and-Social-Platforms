from __future__ import annotations

from typing import Iterable

import pytest

import service.app as service_app
from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import (
    AppConfig,
    LeakWatchConfig,
    RegexEntityConfig,
    TextConfig,
)


SAMPLE = "hello i am gowni ananya phone number is 7676696644"
PERSON = r"\bgowni ananya\b"
PHONE = r"\b\d{10}\b"


def make_pipeline(tmp_path, *, mask_style: str = "faker") -> PipelineManager:
    return PipelineManager(
        LeakWatchConfig(
            app=AppConfig(output_dir=tmp_path / "artifacts"),
            text=TextConfig(
                enable_spacy=False,
                mask_style=mask_style,
                regex_entities=[
                    RegexEntityConfig(name="PERSON", pattern=PERSON),
                    RegexEntityConfig(name="PHONE_NUMBER", pattern=PHONE),
                ],
            ),
        )
    )


@pytest.fixture
def selection_pipeline(tmp_path, monkeypatch) -> PipelineManager:
    pipeline = make_pipeline(tmp_path)
    monkeypatch.setattr(service_app, "PIPELINE", pipeline)
    return pipeline


def scan(text: str, *, detect_only: bool = False, selected: Iterable[str] | None = None):
    fields = {"text": text, "detect_only": detect_only}
    if selected is not None:
        fields["selected_entity_types"] = list(selected)
    return service_app.scan_text(service_app.TextScanRequest(**fields))


def test_detect_only_returns_spans_and_leaves_original_unchanged(selection_pipeline):
    response = scan(SAMPLE, detect_only=True)

    assert response.sanitized_text == SAMPLE
    assert {entity["label"] for entity in response.entities} == {"PERSON", "PHONE_NUMBER"}
    assert all(entity["span"] and entity["entity_id"] for entity in response.entities)


def test_person_only_preserves_phone(selection_pipeline):
    result = scan(SAMPLE, selected=["PERSON"])

    assert "gowni ananya" not in result.sanitized_text.lower()
    assert "7676696644" in result.sanitized_text
    assert "[REDACTED" not in result.sanitized_text


def test_phone_only_preserves_person(selection_pipeline):
    result = scan(SAMPLE, selected=["PHONE_NUMBER"])

    assert "gowni ananya" in result.sanitized_text.lower()
    assert "7676696644" not in result.sanitized_text
    assert "[REDACTED" not in result.sanitized_text


def test_both_types_are_mitigated(selection_pipeline):
    result = scan(SAMPLE, selected=["PERSON", "PHONE_NUMBER"])

    assert "gowni ananya" not in result.sanitized_text.lower()
    assert "7676696644" not in result.sanitized_text
    assert "[REDACTED" not in result.sanitized_text


def test_selected_entities_force_synthetic_values_even_if_general_text_style_is_masking(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(service_app, "PIPELINE", make_pipeline(tmp_path, mask_style="brackets"))
    result = scan(SAMPLE, selected=["PERSON"])

    assert "gowni ananya" not in result.sanitized_text.lower()
    assert "[REDACTED" not in result.sanitized_text


def test_empty_selection_preserves_original_exactly(selection_pipeline):
    result = scan(SAMPLE, selected=[])

    assert result.sanitized_text == SAMPLE
    assert result.redacted_entities == []


def test_select_all_and_clear_all_correspond_to_all_and_no_labels(selection_pipeline):
    detected = scan(SAMPLE, detect_only=True)
    all_types = {entity["label"] for entity in detected.entities}

    selected_all = scan(SAMPLE, selected=all_types)
    cleared = scan(SAMPLE, selected=[])

    assert selected_all.sanitized_text != SAMPLE
    assert cleared.sanitized_text == SAMPLE


def test_duplicate_occurrences_are_replaced_from_original_and_consistently(selection_pipeline):
    text = "gowni ananya called gowni ananya; numbers 7676696644 and 7676696644"
    first = scan(text, selected=["PERSON", "PHONE_NUMBER"])
    repeated_apply = scan(text, selected=["PERSON", "PHONE_NUMBER"])

    assert "gowni ananya" not in first.sanitized_text.lower()
    assert "7676696644" not in first.sanitized_text
    assert first.sanitized_text == repeated_apply.sanitized_text


def test_new_scan_returns_only_current_entities_and_does_not_reuse_selection_state(selection_pipeline):
    first = scan(SAMPLE, detect_only=True)
    second_text = "Phone contact: 1234567890"
    second = scan(second_text, detect_only=True)

    assert {entity["label"] for entity in first.entities} == {"PERSON", "PHONE_NUMBER"}
    assert {entity["label"] for entity in second.entities} == {"PHONE_NUMBER"}
    assert second.sanitized_text == second_text
