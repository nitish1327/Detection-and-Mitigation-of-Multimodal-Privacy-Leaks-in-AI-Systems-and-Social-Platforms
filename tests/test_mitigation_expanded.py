import pytest
from leakwatch.mitigation.text import TextMitigator
from leakwatch.utils.config import TextConfig
from leakwatch.utils.types import DetectedEntity, Span, Modality

print("TextConfig Schema:", TextConfig.model_json_schema())

def test_expanded_faker_mappings():
    config = TextConfig(mask_style="faker", faker_locale="en_US")
    mitigator = TextMitigator(config)
    
    entities = [
        DetectedEntity(modality=Modality.TEXT, label="SSN", text="123-45-6789", confidence=1.0, span=Span(start=10, end=21)),
        DetectedEntity(modality=Modality.TEXT, label="IBAN", text="DE1234567890", confidence=1.0, span=Span(start=29, end=41)),
        DetectedEntity(modality=Modality.TEXT, label="IP_ADDRESS", text="192.168.1.1", confidence=1.0, span=Span(start=46, end=57)),
    ]
    text = "My SSN is 123-45-6789, IBAN DE1234567890, IP 192.168.1.1"
    
    sanitized, updated = mitigator.mitigate(text, entities)
    
    print(f"\nOriginal: {text}")
    print(f"Sanitized: {sanitized}")
    
    assert "123-45-6789" not in sanitized
    assert "DE1234567890" not in sanitized
    assert "192.168.1.1" not in sanitized
    assert len(updated) == 3

def test_localization():
    # Test with a different locale (e.g., hi_IN for India)
    config = TextConfig(mask_style="faker", faker_locale="hi_IN")
    mitigator = TextMitigator(config)
    
    entities = [
        DetectedEntity(modality=Modality.TEXT, label="PERSON", text="John Doe", confidence=1.0, span=Span(start=0, end=8)),
    ]
    text = "John Doe is here."
    
    sanitized, updated = mitigator.mitigate(text, entities)
    print(f"\nLocale hi_IN Sanitized: {sanitized}")
    
    # Simple check: it should be different from "John Doe"
    assert "John Doe" not in sanitized

def test_static_replacements():
    config = TextConfig(
        mask_style="faker", 
        static_replacements={"Internal Project Alpha": "Public Project Beta"}
    )
    mitigator = TextMitigator(config)
    
    entities = [
        DetectedEntity(modality=Modality.TEXT, label="PROJECT", text="Internal Project Alpha", confidence=1.0, span=Span(start=0, end=22)),
    ]
    text = "Internal Project Alpha is secret."
    
    sanitized, updated = mitigator.mitigate(text, entities)
    print(f"\nStatic Replacement Sanitized: {sanitized}")
    
    assert "Public Project Beta" in sanitized
    assert "Internal Project Alpha" not in sanitized

if __name__ == "__main__":
    # For manual running
    test_expanded_faker_mappings()
    test_localization()
    test_static_replacements()
    print("\nAll tests passed!")
