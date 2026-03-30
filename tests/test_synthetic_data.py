from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import (
    AppConfig,
    LeakWatchConfig,
    TextConfig,
    RegexEntityConfig,
)

def test_faker_synthetic_replacement(tmp_path):
    input_text = "My name is John Doe and my email is john.doe@example.com. Reach me at 555-0199."
    sample = tmp_path / "faker_input.txt"
    sample.write_text(input_text, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=True,
            mask_style="faker",
            regex_entities=[
                RegexEntityConfig(name="phone", pattern=r"\d{3}-\d{4}", action="mask"),
                RegexEntityConfig(name="email", pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", action="mask"),
            ],
        ),
    )

    manager = PipelineManager(config)
    result = manager.process_text(sample)

    mitigated_text = (tmp_path / "artifacts" / "faker_input.sanitized.txt").read_text(encoding="utf-8")
    
    print(f"Original: {input_text}")
    print(f"Mitigated: {mitigated_text}")

    # Check that original PII is gone
    assert "John Doe" not in mitigated_text
    assert "john.doe@example.com" not in mitigated_text
    assert "555-0199" not in mitigated_text

    # Check that we have some replacements (not just [REDACTED])
    assert "[REDACTED" not in mitigated_text
    assert "<" not in mitigated_text # Faker should have provided real-looking strings

def test_faker_consistency(tmp_path):
    input_text = "John Doe went to the park. John Doe bought an ice cream."
    sample = tmp_path / "consistency_input.txt"
    sample.write_text(input_text, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(mask_style="faker"),
    )

    manager = PipelineManager(config)
    result = manager.process_text(sample)
    mitigated_text = (tmp_path / "artifacts" / "consistency_input.sanitized.txt").read_text(encoding="utf-8")
    
    # Split by spaces and find replacements for John Doe
    # If consistency works, both "John Doe" spans should be replaced by the same name.
    # Note: Spacy might detect "John Doe" as one entity.
    
    print(f"Mitigated Consistency: {mitigated_text}")
    
    # Find all entities in result
    mitigated_names = [e.text for e in result.entities if e.label == "PERSON"]
    assert len(mitigated_names) >= 2
    assert mitigated_names[0] == mitigated_names[1], "Consistent replacement failed"

if __name__ == "__main__":
    import pytest
    import sys
    from pathlib import Path
    
    # Add project root to sys.path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    pytest.main([__file__])
