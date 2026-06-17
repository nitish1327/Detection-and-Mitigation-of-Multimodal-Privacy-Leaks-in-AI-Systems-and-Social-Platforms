"""Process ronaldo_aadhar.jpg for privacy mitigation."""

from pathlib import Path
from leakwatch.orchestration.pipeline import PipelineManager
from leakwatch.utils.config import LeakWatchConfig, ImageConfig, TextConfig, AppConfig, ExplainabilityConfig, RegexEntityConfig

def main():
    # Define paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    image_path = base_dir / "samples" / "image" / "ronaldo_aadhar.jpg"
    
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return

    output_dir = base_dir / "artifacts"
    output_dir.mkdir(exist_ok=True)

    # Configure LeakWatch
    config = LeakWatchConfig(
        app=AppConfig(
            mode="cli",
            output_dir=output_dir
        ),
        text=TextConfig(
            enable_spacy=True,
            mask_style="faker",
            faker_locale="en_US",
            regex_entities=[
                RegexEntityConfig(name="aadhar", pattern=r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b|\b\d{12}\b|\b\d{4}\b", action="blur")
            ]
        ),
        image=ImageConfig(
            face_blur_kernel=35,
            enable_ocr=True,
            enable_qr=True,
            enable_mrz=True,
            enable_layout=True,
            min_confidence=0.3
        ),
        explainability=ExplainabilityConfig(
            save_text_spans=True,
            save_image_overlays=True,
            audit_log_path=output_dir / "ronaldo_aadhar_mitigation_report.log"
        )
    )

    # Initialize Pipeline
    manager = PipelineManager(config=config)

    # Process Image
    print(f"Processing {image_path}...")
    result = manager.process_image(image_path)

    print("\nMitigation Complete!")
    print(f"Sanitized image: {result.mitigated_output}")
    print(f"Audit report: {result.audit_log}")
    
    if result.entities:
        print("\nDetected Entities:")
        for entity in result.entities:
            print(f" - {entity.label}: {entity.mitigation} (Confidence: {entity.confidence:.2f})")
    else:
        print("\nNo sensitive entities detected.")

if __name__ == "__main__":
    main()
