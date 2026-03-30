from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import (
    AppConfig,
    LeakWatchConfig,
    TextConfig,
)

def test_allowlist_filtering(tmp_path):
    # "TechCorp" is often detected as an ORG by SpaCy
    input_text = "Jane Smith works at TechCorp Solutions Inc. Her email is jane@example.com."
    sample = tmp_path / "allowlist_input.txt"
    sample.write_text(input_text, encoding="utf-8")

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(
            enable_spacy=True,
            mask_style="brackets",
            allowlist=["TechCorp Solutions Inc."]
        ),
    )

    manager = PipelineManager(config)
    result = manager.process_text(sample)
    mitigated_text = (tmp_path / "artifacts" / "allowlist_input.sanitized.txt").read_text(encoding="utf-8")
    
    print(f"Mitigated: {mitigated_text}")

    # TechCorp should NOT be redacted
    assert "TechCorp Solutions Inc." in mitigated_text
    
    # Jane Smith should be redacted (mapped to PERSON by SpaCy)
    assert "Jane Smith" not in mitigated_text
    assert "[REDACTED:PERSON]" in mitigated_text

if __name__ == "__main__":
    import pytest
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    pytest.main([__file__])
