import json
from pathlib import Path
from unittest.mock import patch

import leakwatch.detection.text as text_detection_module
from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import (
    AppConfig,
    ExplainabilityConfig,
    ImageConfig,
    LeakWatchConfig,
    RegexEntityConfig,
    TextConfig,
)


def test_text_pipeline_creates_artifacts(tmp_path):
    input_text = "Reach me at 123-456-7890 and email me@sample.com"
    sample = tmp_path / "input.txt"
    sample.write_text(input_text, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=False,
            regex_entities=[
                RegexEntityConfig(name="phone", pattern=r"\d{3}-\d{3}-\d{4}", action="mask"),
                RegexEntityConfig(name="email", pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", action="mask"),
            ],
        ),
        image=ImageConfig(enable_ocr=False),
        explainability=ExplainabilityConfig(
            save_text_spans=True,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
    )

    manager = PipelineManager(config)
    result = manager.process_text(sample)

    assert result.mitigated_output is not None
    sanitized_text = Path(result.mitigated_output).read_text(encoding="utf-8")
    assert "[REDACTED" in sanitized_text

    assert result.artifacts, "expected explainability artifact"
    for artifact in result.artifacts:
        assert artifact.exists()

    assert result.audit_log is not None
    assert Path(result.audit_log).exists()

    audit_payload = json.loads(Path(result.audit_log).read_text(encoding="utf-8"))
    assert "spans" in audit_payload
    assert isinstance(audit_payload["spans"], list)
    assert audit_payload["spans"], "expected non-empty spans in audit payload"
    first_span = audit_payload["spans"][0]
    assert "span_text" in first_span
    assert "entity_type" in first_span
    assert "detection_method" in first_span
    assert "mitigation_applied" in first_span
    assert "confidence" in first_span
    assert "timeline_window" in first_span


def test_nlp_importerror_emits_warning_and_uses_regex_fallback_mode(tmp_path, caplog):
    input_text = "Call me at 123-456-7890"
    sample = tmp_path / "input.txt"
    sample.write_text(input_text, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=True,
            regex_entities=[
                RegexEntityConfig(name="phone", pattern=r"\d{3}-\d{3}-\d{4}", action="mask"),
            ],
        ),
        image=ImageConfig(enable_ocr=False),
        explainability=ExplainabilityConfig(
            save_text_spans=True,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
    )

    caplog.set_level("WARNING")
    with patch.object(text_detection_module, "spacy") as mock_spacy:
        mock_spacy.load.side_effect = ImportError("mocked import failure")
        manager = PipelineManager(config)
        result = manager.process_text(sample)

    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    assert "falling back to regex-only" in caplog.text

    assert result.audit_log is not None
    payload = json.loads(Path(result.audit_log).read_text(encoding="utf-8"))
    assert payload["detection_mode"] == "regex_fallback"


def test_text_wrapping_behavior(tmp_path):
    input_text = "This is a very long sentence that has more than fifty characters to test if the wrapping logic wraps correctly."
    sample = tmp_path / "long_input.txt"
    sample.write_text(input_text, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=False,
            wrap_lines=True,
            max_line_length=40,
        ),
        image=ImageConfig(enable_ocr=False),
        explainability=ExplainabilityConfig(
            save_text_spans=False,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
    )

    manager = PipelineManager(config)
    result = manager.process_text(sample)

    assert result.mitigated_output is not None
    sanitized_text = Path(result.mitigated_output).read_text(encoding="utf-8")
    lines = sanitized_text.splitlines()
    assert len(lines) > 1
    assert all(len(line) <= 40 for line in lines)


def test_spoken_and_phonetic_email_detection():
    config = TextConfig(enable_spacy=False)
    detector = text_detection_module.TextDetector(config)

    # 1. Video2 transcript with phonetic Whisper transcription of email
    transcript = (
        "Hi I am Ghani Ananya and my other number is 9.0625347320 and my phone number is "
        "7.67669.624 and my email ID is me on uradyadheradegmail.com"
    )
    entities = detector.detect(transcript)
    email_entities = [e for e in entities if e.label == "email"]
    assert len(email_entities) >= 1
    assert "uradyadheradegmail.com" in email_entities[0].text

    # 2. Standalone spoken 'at the rate'
    t2 = "Send an email to user at the rate yahoo dot com please"
    e2 = [e for e in detector.detect(t2) if e.label == "email"]
    assert len(e2) == 1
    assert e2[0].text == "user at the rate yahoo dot com"

    # 3. Standalone spoken 'at' with mail provider
    t3 = "Contact me at alice at gmail dot com for updates"
    e3 = [e for e in detector.detect(t3) if e.label == "email"]
    assert len(e3) == 1
    assert e3[0].text == "alice at gmail dot com"