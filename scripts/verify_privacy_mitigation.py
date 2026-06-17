"""Verify the enhanced image privacy mitigation."""

from pathlib import Path
from leakwatch.orchestration.pipeline import PipelineManager
from leakwatch.utils.config import get_config
import cv2

def main():
    # Define paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    image_path = base_dir / "samples" / "image" / "ronaldo_aadhar.jpg"
    
    if not image_path.exists():
        # Fallback to artifacts if not in samples
        image_path = base_dir / "artifacts" / "ronaldo_aadhar.sanitized.jpg"

    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return

    output_dir = base_dir / "artifacts"
    output_dir.mkdir(exist_ok=True)

    # Load config from the yaml file we just modified
    config = get_config(base_dir / "config" / "leakwatch.yaml")
    
    # Ensure default_style is synthetic for testing
    config.image.default_style = "synthetic"
    config.explainability.audit_log_path = output_dir / "verification_audit.log"

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
            # Check if replacements are happening correctly
            replacement = entity.text if entity.mitigation == "replace" else entity.mitigation
            print(f" - {entity.label}: {replacement} (Conf: {entity.confidence:.2f})")
    else:
        print("\nNo sensitive entities detected.")

if __name__ == "__main__":
    main()
