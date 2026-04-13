import sys
from pathlib import Path
from leakwatch.orchestration.pipeline import PipelineManager
from leakwatch.utils.config import LeakWatchConfig, ImageConfig, TextConfig, AppConfig, ExplainabilityConfig, RegexEntityConfig

def main(image_path_str):
    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return

    output_dir = Path("c:/Capstone/code/-Detection-and-Mitigation-of-Multimodal-Privacy-Leaks-in-AI-Systems-and-Social-Platforms/artifacts")
    output_dir.mkdir(exist_ok=True, parents=True)

    from leakwatch.utils.config import get_config
    config = get_config(Path("c:/Capstone/code/-Detection-and-Mitigation-of-Multimodal-Privacy-Leaks-in-AI-Systems-and-Social-Platforms/config/leakwatch.yaml"))
    
    # Optional runtime overrides:
    config.app.output_dir = output_dir
    config.image.default_style = "synthetic"
    config.explainability.audit_log_path = output_dir / "user_image_mitigation_report.log"

    manager = PipelineManager(config=config)
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
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide an image path.")
