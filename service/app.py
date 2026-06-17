"""FastAPI wrapper around the LeakWatch pipeline."""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, HttpUrl, model_validator

from leakwatch.orchestration import PipelineManager, VideoProcessingError
from leakwatch.utils.config import get_config
from leakwatch.utils.types import DetectionResult

app = FastAPI(
    title="LeakWatch Middleware API",
    version="1.0.0",
    description=(
        "REST interface for the LeakWatch multimodal privacy leak detection "
        "pipeline. Reuses the CLI components without modification."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG = get_config()
PIPELINE = PipelineManager(CONFIG)


class PolicyConfig(BaseModel):
    """Simple policy knobs for allow/block decisions."""

    block_on: list[str] = Field(default_factory=list, description="Entity labels that trigger a block decision.")
    mask_on: list[str] = Field(
        default_factory=list,
        description="Entity labels that should always be masked (handled by core pipeline).",
    )


class TextScanRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw text content to scan.")
    path: str | None = Field(default=None, description="Filesystem path to a text document.")
    url: HttpUrl | None = Field(default=None, description="Optional URL pointing to a text file.")
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @model_validator(mode="after")
    def validate_source(cls, values: "TextScanRequest") -> "TextScanRequest":
        if not values.text and not values.path and not values.url:
            raise ValueError("Provide at least one of: text, path, url")
        return values


class FileScanRequest(BaseModel):
    path: str | None = Field(default=None, description="Filesystem path to the asset.")
    url: HttpUrl | None = Field(default=None, description="Optional URL pointing to the asset.")
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @model_validator(mode="after")
    def validate_source(cls, values: "FileScanRequest") -> "FileScanRequest":
        if not values.path and not values.url:
            raise ValueError("Provide either path or url for non-text modalities")
        return values


class ScanResponse(BaseModel):
    decision: Literal["allow", "block"]
    reason: str | None = None
    entities: list[dict] = Field(default_factory=list)
    mitigated_output: str | None = None
    audit_log: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    sanitized_text: str | None = None
    redacted_entities: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "leakwatch"}


@app.post("/scan/text", response_model=ScanResponse)
def scan_text(request: TextScanRequest) -> ScanResponse:
    source_path: Path | None = None
    scratch_dir: Path | None = None
    try:
        if request.text:
            scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_text_"))
            source_path = scratch_dir / "payload.txt"
            source_path.write_text(request.text, encoding="utf-8")
        else:
            source_path, scratch_dir = _materialize_file(request.path, request.url, suffix=".txt")
        result = PIPELINE.process_text(source_path)
        response = _build_response(result, request.policy)
        if result.mitigated_output and result.mitigated_output.exists():
            response.sanitized_text = result.mitigated_output.read_text(encoding="utf-8")
        response.redacted_entities = _extract_redacted_labels(result)
        return response
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/image", response_model=ScanResponse)
def scan_image(request: FileScanRequest) -> ScanResponse:
    source_path, scratch_dir = _materialize_file(request.path, request.url, suffix=".png")
    try:
        result = PIPELINE.process_image(source_path)
        return _build_response(result, request.policy)
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/audio", response_model=ScanResponse)
def scan_audio(request: FileScanRequest) -> ScanResponse:
    source_path, scratch_dir = _materialize_file(request.path, request.url, suffix=".wav")
    try:
        result = PIPELINE.process_audio(source_path)
        return _build_response(result, request.policy)
    except RuntimeError as exc:  # e.g., modality disabled
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/video", response_model=ScanResponse)
def scan_video(request: FileScanRequest) -> ScanResponse:
    source_path, scratch_dir = _materialize_file(request.path, request.url, suffix=".mp4")
    try:
        result = PIPELINE.process_video(source_path)
        return _build_response(result, request.policy)
    except VideoProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup(scratch_dir)


@app.get("/artifacts/download")
def download_artifact(path: str) -> FileResponse:
    """Serve sanitized artifacts generated by the pipeline."""

    artifact_path = _resolve_artifact(path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    return FileResponse(artifact_path, filename=artifact_path.name)


def _materialize_file(
    path_value: str | None,
    url_value: HttpUrl | None,
    *,
    suffix: str,
) -> tuple[Path, Path | None]:
    if path_value:
        source_path = Path(path_value)
        if not source_path.exists():
            raise HTTPException(status_code=400, detail=f"File not found: {path_value}")
        return source_path, None
    if url_value:
        scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_dl_"))
        destination = scratch_dir / f"payload{suffix}"
        try:
            urllib.request.urlretrieve(str(url_value), destination)
        except Exception as exc:  # noqa: BLE001
            _cleanup(scratch_dir)
            raise HTTPException(status_code=400, detail=f"Failed to download asset: {exc}") from exc
        return destination, scratch_dir
    raise HTTPException(status_code=400, detail="A local path or URL is required")


def _cleanup(directory: Path | None) -> None:
    if directory and directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


def _resolve_artifact(relative_or_absolute_path: str) -> Path:
    base_dir = Path(CONFIG.app.output_dir).resolve()
    candidate = Path(relative_or_absolute_path)
    candidate = candidate if candidate.is_absolute() else base_dir / candidate
    candidate = candidate.resolve()
    if candidate == base_dir or base_dir not in candidate.parents:
        raise HTTPException(status_code=400, detail="Artifact path must reside inside the configured output directory")
    return candidate


def _build_response(result: DetectionResult, policy: PolicyConfig) -> ScanResponse:
    decision, reason = _evaluate_policy(policy, result)
    return ScanResponse(
        decision=decision,
        reason=reason,
        entities=[entity.model_dump() for entity in result.entities],
        mitigated_output=str(result.mitigated_output) if result.mitigated_output else None,
        audit_log=str(result.audit_log) if result.audit_log else None,
        artifacts=[str(path) for path in result.artifacts],
    )


def _extract_redacted_labels(result: DetectionResult) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for entity in result.entities:
        if entity.mitigation == "none":
            continue
        label = entity.label.upper()
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _evaluate_policy(policy: PolicyConfig, result: DetectionResult) -> tuple[Literal["allow", "block"], str | None]:
    if not policy.block_on:
        return "allow", None
    block_labels = {label.lower() for label in policy.block_on}
    for entity in result.entities:
        if entity.label.lower() in block_labels:
            return "block", f"Policy blocked entity label: {entity.label}"
    return "allow", None
