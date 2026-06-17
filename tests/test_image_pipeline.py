from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from leakwatch.orchestration import PipelineManager
from leakwatch.utils.config import AppConfig, ExplainabilityConfig, ImageConfig, LeakWatchConfig, TextConfig
from leakwatch.utils.types import BoundingBox, DetectedEntity, Modality


def test_image_pipeline_rewrites_sensitive_text_region(tmp_path: Path) -> None:
    image_path = tmp_path / "id_card.png"
    image = np.full((140, 280, 3), 255, dtype=np.uint8)
    cv2.putText(image, "ID A1234567", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(image_path), image)

    config = LeakWatchConfig(
        app=AppConfig(output_dir=tmp_path / "artifacts"),
        text=TextConfig(enable_spacy=False),
        image=ImageConfig(enable_ocr=False),
        explainability=ExplainabilityConfig(
            save_text_spans=False,
            save_image_overlays=False,
            audit_log_path=tmp_path / "audit.log",
        ),
    )

    manager = PipelineManager(config)
    entities = [
        DetectedEntity(
            modality=Modality.IMAGE,
            label="ID_NUMBER",
            confidence=0.95,
            text="A1234567",
            bbox=BoundingBox(x=15, y=45, width=230, height=55),
            mitigation="replace",
            detection_method="regex",
        )
    ]

    result = manager.process_image(image_path, entities_override=entities)

    assert result.mitigated_output is not None
    assert result.mitigated_output.exists()
    assert result.entities
    assert result.entities[0].mitigation == "replace"

    original = cv2.imread(str(image_path))
    sanitized = cv2.imread(str(result.mitigated_output))
    assert original is not None and sanitized is not None
    assert original.shape == sanitized.shape

    changed_pixels = np.count_nonzero(np.any(original != sanitized, axis=2))
    assert changed_pixels > 0
    assert result.audit_log is not None
    assert result.audit_log.exists()
