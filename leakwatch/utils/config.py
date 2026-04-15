"""Configuration utilities for LeakWatch."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

CONFIG_PATH = Path("config/leakwatch.yaml")


class RegexEntityConfig(BaseModel):
    """Regex detector configuration for text modality."""

    name: str
    pattern: str
    action: str = "mask"


class TextConfig(BaseModel):
    enable_spacy: bool = True
    max_doc_length: int = 10000
    mask_style: str = "brackets"
    confidence_threshold: float = 0.5
    regex_entities: list[RegexEntityConfig] = Field(default_factory=list)


class ImageConfig(BaseModel):
    face_blur_kernel: int = 35
    text_detection_langs: list[str] = Field(default_factory=lambda: ["en"])
    min_confidence: float = 0.3
    enable_ocr: bool = True

    @field_validator("face_blur_kernel")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:  # noqa: D401
        """Ensure kernel size is odd to satisfy OpenCV requirements."""

        return value if value % 2 == 1 else value + 1


class ExplainabilityConfig(BaseModel):
    save_text_spans: bool = True
    save_image_overlays: bool = True
    audit_log_path: Path = Path("artifacts/audit.log")


class ModalityToggle(BaseModel):
    enabled: bool = False


class AudioConfig(ModalityToggle):
    transcript_extension: str = ".txt"
    beep_frequency_hz: float = 1000.0
    beep_amplitude: float = 0.85
    min_beep_ms: int = 120
    padding_ms: int = 80


class AppConfig(BaseModel):
    mode: str = "cli"
    output_dir: Path = Path("artifacts")


class LeakWatchConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    video: ModalityToggle = Field(default_factory=ModalityToggle)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LeakWatchConfig":
        return cls(**data)

    @classmethod
    def from_path(cls, path: Path | str | None = None) -> "LeakWatchConfig":
        cfg_path = Path(path) if path else CONFIG_PATH
        with cfg_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_mapping(raw)


@lru_cache(maxsize=1)
def get_config(path: Path | str | None = None) -> LeakWatchConfig:
    """Return cached configuration object."""

    return LeakWatchConfig.from_path(path)
