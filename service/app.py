"""FastAPI wrapper around the LeakWatch pipeline."""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, HttpUrl, model_validator

from leakwatch.orchestration import PipelineManager, VideoProcessingError
from leakwatch.utils.config import get_config
from leakwatch.utils.types import DetectionResult, Modality
from leakwatch.video.tracker import (
    generate_video_face_preview,
    process_selective_video,
)

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

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/webp",
    "image/pjpeg",
    "image/tiff",
}

ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


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
    detect_only: bool = Field(default=False, description="Return detections without applying mitigation.")
    selected_entity_types: list[str] | None = Field(
        default=None,
        description="Entity labels to mitigate. Omit for the legacy behavior of mitigating all detected entities.",
    )
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @model_validator(mode="after")
    def validate_source(self) -> "TextScanRequest":
        if not self.text and not self.path and not self.url:
            raise ValueError("Provide at least one of: text, path, url")
        return self


class FileScanRequest(BaseModel):
    path: str | None = Field(default=None, description="Filesystem path to the asset.")
    url: HttpUrl | None = Field(default=None, description="Optional URL pointing to the asset.")
    policy: PolicyConfig = Field(default_factory=PolicyConfig)

    @model_validator(mode="after")
    def validate_source(self) -> "FileScanRequest":
        if not self.path and not self.url:
            raise ValueError("Provide either path or url for non-text modalities")
        return self


class ScanResponse(BaseModel):
    decision: Literal["allow", "block"]
    reason: str | None = None
    entities: list[dict] = Field(default_factory=list)
    mitigated_output: str | None = None
    audit_log: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    sanitized_text: str | None = None
    redacted_entities: list[str] = Field(default_factory=list)
    sanitized_image_base64: str | None = None
    download_url: str | None = None


class SelectiveVideoScanResponse(ScanResponse):
    status: str = "success"
    faces_detected: int = 0
    faces_blurred: int = 0
    selected_faces: list[int] = Field(default_factory=list)
    mode: str = "blur_selected"
    audio_mitigated: bool = False
    audio_entities_detected: int = 0


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

        if request.detect_only or request.selected_entity_types is not None:
            original_text = source_path.read_text(encoding="utf-8")
            detected = PIPELINE.detect_text_entities(original_text)
            detection_result = DetectionResult(
                source_path=source_path,
                modality=Modality.TEXT,
                entities=detected,
            )
            response = _build_response(detection_result, request.policy)

            if request.detect_only:
                response.sanitized_text = original_text
                response.redacted_entities = sorted({entity.label.upper() for entity in detected})
                return response

            selected_types = {label.strip().upper() for label in request.selected_entity_types or []}
            sanitized_text, mitigated_entities = PIPELINE.mitigate_text_entities(
                original_text,
                detected,
                selected_types,
            )
            response.sanitized_text = sanitized_text
            response.entities = [entity.model_dump() for entity in mitigated_entities]
            response.redacted_entities = sorted({entity.label.upper() for entity in mitigated_entities})
            return response

        # Requests without the new optional fields retain the original automatic pipeline.
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
        response = _build_response(result, request.policy)
        _attach_image_payload(response, result)
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {exc}") from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/image/upload", response_model=ScanResponse)
async def scan_image_upload(file: UploadFile = File(...)) -> ScanResponse:
    """Accept browser-uploaded images via multipart/form-data (field name: file)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Image filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{content_type}'.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_img_upload_"))
    source_path = scratch_dir / f"upload{suffix}"
    source_path.write_bytes(raw)

    try:
        result = PIPELINE.process_image(source_path)
        response = _build_response(result, PolicyConfig())
        _attach_image_payload(response, result)
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {exc}") from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/audio", response_model=ScanResponse)
def scan_audio(request: FileScanRequest) -> ScanResponse:
    source_path, scratch_dir = _materialize_file(request.path, request.url, suffix=".wav")
    try:
        result = PIPELINE.process_audio(source_path)
        response = _build_response(result, request.policy)
        _attach_download_payload(response, result)
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {exc}") from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/audio/upload", response_model=ScanResponse)
async def scan_audio_upload(file: UploadFile = File(...)) -> ScanResponse:
    """Accept browser-uploaded audio via multipart/form-data (field name: file)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_audio_upload_"))
    source_path = scratch_dir / f"upload{suffix}"
    source_path.write_bytes(raw)

    try:
        result = PIPELINE.process_audio(source_path)
        response = _build_response(result, PolicyConfig())
        _attach_download_payload(response, result)
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {exc}") from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/video", response_model=ScanResponse)
def scan_video(request: FileScanRequest) -> ScanResponse:
    source_path, scratch_dir = _materialize_file(request.path, request.url, suffix=".mp4")
    try:
        result = PIPELINE.process_video(source_path)
        response = _build_response(result, request.policy)
        _attach_download_payload(response, result)
        return response
    except VideoProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/video/upload", response_model=ScanResponse)
async def scan_video_upload(file: UploadFile = File(...)) -> ScanResponse:
    """Accept browser-uploaded video via multipart/form-data (field name: file)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Video filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_video_upload_"))
    source_path = scratch_dir / f"upload{suffix}"
    source_path.write_bytes(raw)

    try:
        output_dir = Path(CONFIG.app.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{source_path.stem}.sanitized_{uuid.uuid4().hex[:8]}.mp4"
        output_path = output_dir / unique_name

        res = process_selective_video(
            source_video_path=source_path,
            output_path=output_path,
            selected_face_ids=[],
            mode="blur_all",
            blur_strength=51,
            redact_sensitive_info=True,
            mitigate_audio=True,
            pipeline_manager=PIPELINE,
        )

        relative_download_url = f"/artifacts/download?path={output_path.name}"
        redacted_entities = res.get("redacted_entities", ["FACE"])

        return ScanResponse(
            status="success",
            decision="allow",
            reason=None,
            sanitized_text=None,
            redacted_entities=redacted_entities,
            artifacts=[str(output_path)],
            download_url=relative_download_url,
            audit_log=f"Full video sanitization applied. Frames: {res.get('total_frames', 0)}. Entities: {redacted_entities}",
            mitigated_output=str(output_path),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Video processing failed: {exc}") from exc
    finally:
        _cleanup(scratch_dir)


@app.post("/scan/video/preview-faces")
async def preview_video_faces(request: Request) -> dict[str, Any]:
    """Inspect first valid video frame, detect all faces using YuNet, and return previews."""
    content_type = (request.headers.get("content-type") or "").lower()
    source_path: Path | None = None
    scratch_dir: Path | None = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            if not uploaded_file:
                # Fallback to path in form
                path_val = form.get("path")
                if path_val:
                    source_path, scratch_dir = _materialize_file(str(path_val), None, suffix=".mp4")
                else:
                    raise HTTPException(status_code=400, detail="Video file is required in 'file' field.")
            else:
                filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_VIDEO_SUFFIXES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported video format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}",
                    )
                raw = await uploaded_file.read()
                if not raw:
                    raise HTTPException(status_code=400, detail="Uploaded video is empty.")
                scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_video_prev_"))
                source_path = scratch_dir / f"upload{suffix}"
                source_path.write_bytes(raw)
        elif "application/json" in content_type:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body.")
            path_val = body.get("path")
            url_val = body.get("url")
            source_path, scratch_dir = _materialize_file(path_val, url_val, suffix=".mp4")
        else:
            form = await request.form()
            uploaded_file = form.get("file")
            if uploaded_file:
                filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
                suffix = Path(filename).suffix.lower()
                raw = await uploaded_file.read()
                scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_video_prev_"))
                source_path = scratch_dir / f"upload{suffix}"
                source_path.write_bytes(raw)
            else:
                raise HTTPException(status_code=400, detail="Provide a video file upload or JSON with path/url.")

        preview_data = generate_video_face_preview(source_path)
        return preview_data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup(scratch_dir)


def _parse_face_selection(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        val = raw.strip()
        if not val:
            return []
        if val.startswith("[") and val.endswith("]"):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
            except Exception:
                pass
        return [int(p.strip()) for p in val.split(",") if p.strip()]
    return []


@app.post("/scan/video/selective", response_model=SelectiveVideoScanResponse)
async def scan_video_selective(request: Request) -> dict[str, Any]:
    """Selectively blur target faces in a video, preserving audio with FFmpeg."""
    content_type = (request.headers.get("content-type") or "").lower()
    source_path: Path | None = None
    scratch_dir: Path | None = None
    selected_faces: list[int] = []
    mode: str = "blur_selected"
    blur_strength: int = 51
    redact_sensitive_info: bool = True
    mitigate_audio: bool = True

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            if not uploaded_file:
                path_val = form.get("path")
                if path_val:
                    source_path, scratch_dir = _materialize_file(str(path_val), None, suffix=".mp4")
                else:
                    raise HTTPException(status_code=400, detail="Video file is required in 'file' field.")
            else:
                filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
                suffix = Path(filename).suffix.lower()
                if suffix not in ALLOWED_VIDEO_SUFFIXES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported video format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}",
                    )
                raw = await uploaded_file.read()
                if not raw:
                    raise HTTPException(status_code=400, detail="Uploaded video is empty.")
                scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_video_sel_"))
                source_path = scratch_dir / f"upload{suffix}"
                source_path.write_bytes(raw)

            selected_faces = _parse_face_selection(form.get("selected_faces"))
            mode = str(form.get("mode", "blur_selected")).strip()
            if "redact_sensitive_info" in form:
                redact_sensitive_info = str(form.get("redact_sensitive_info", "true")).lower() in ("true", "1", "yes")
            if "mitigate_audio" in form:
                mitigate_audio = str(form.get("mitigate_audio", "true")).lower() in ("true", "1", "yes")
            try:
                blur_strength = int(form.get("blur_strength", 51))
            except (ValueError, TypeError):
                blur_strength = 51

        elif "application/json" in content_type:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body.")
            path_val = body.get("path")
            url_val = body.get("url")
            source_path, scratch_dir = _materialize_file(path_val, url_val, suffix=".mp4")
            selected_faces = _parse_face_selection(body.get("selected_faces"))
            mode = str(body.get("mode", "blur_selected")).strip()
            if "redact_sensitive_info" in body:
                redact_sensitive_info = bool(body.get("redact_sensitive_info"))
            if "mitigate_audio" in body:
                mitigate_audio = bool(body.get("mitigate_audio"))
            try:
                blur_strength = int(body.get("blur_strength", 51))
            except (ValueError, TypeError):
                blur_strength = 51

        else:
            form = await request.form()
            uploaded_file = form.get("file")
            if uploaded_file:
                filename = getattr(uploaded_file, "filename", "video.mp4") or "video.mp4"
                suffix = Path(filename).suffix.lower()
                raw = await uploaded_file.read()
                scratch_dir = Path(tempfile.mkdtemp(prefix="leakwatch_video_sel_"))
                source_path = scratch_dir / f"upload{suffix}"
                source_path.write_bytes(raw)
                selected_faces = _parse_face_selection(form.get("selected_faces"))
                mode = str(form.get("mode", "blur_selected")).strip()
                blur_strength = int(form.get("blur_strength", 51))
                if "redact_sensitive_info" in form:
                    redact_sensitive_info = str(form.get("redact_sensitive_info", "true")).lower() in ("true", "1", "yes")
                if "mitigate_audio" in form:
                    mitigate_audio = str(form.get("mitigate_audio", "true")).lower() in ("true", "1", "yes")
            else:
                raise HTTPException(status_code=400, detail="Provide a video file upload or JSON with path/url.")

        if mode not in ("blur_selected", "blur_all_except", "blur_all"):
            mode = "blur_selected"

        output_dir = Path(CONFIG.app.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_name = f"{source_path.stem}.sanitized_{uuid.uuid4().hex[:8]}.mp4"
        output_path = output_dir / unique_name

        res = process_selective_video(
            source_video_path=source_path,
            output_path=output_path,
            selected_face_ids=selected_faces,
            mode=mode,
            blur_strength=blur_strength,
            redact_sensitive_info=redact_sensitive_info,
            mitigate_audio=mitigate_audio,
            pipeline_manager=PIPELINE,
        )

        relative_download_url = f"/artifacts/download?path={output_path.name}"
        entities = [
            {
                "modality": "video",
                "label": "face",
                "confidence": 0.9,
                "mitigation": "blur",
                "metadata": {"face_id": fid},
            }
            for fid in res.get("target_face_ids", [])
        ]
        for a_ent in res.get("audio_entities", []):
            entities.append({
                "modality": "audio",
                "label": a_ent.get("label", "audio_pii"),
                "confidence": a_ent.get("confidence", 0.8),
                "mitigation": a_ent.get("mitigation", "tone_mask"),
                "text": a_ent.get("text"),
            })
        redacted_entities = res.get("redacted_entities", ["FACE"] if entities else [])
        audio_info = f"Audio mitigated: {res.get('audio_mitigated', False)} ({len(res.get('audio_entities', []))} spoken entities)."
        audit_log = (
            f"Selective face blur applied. Mode: {mode}. Selected IDs: {selected_faces}. "
            f"Blurred face IDs: {res.get('target_face_ids', [])}. Frames processed: {res.get('total_frames', 0)}. {audio_info}"
        )

        return {
            "status": "success",
            "decision": "allow",
            "reason": None,
            "download_url": relative_download_url,
            "mitigated_output": str(output_path),
            "faces_detected": res.get("faces_detected", 0),
            "faces_blurred": res.get("faces_blurred", 0),
            "selected_faces": selected_faces,
            "mode": mode,
            "audio_mitigated": res.get("audio_mitigated", False),
            "audio_entities_detected": len(res.get("audio_entities", [])),
            "audit_log": audit_log,
            "entities": entities,
            "redacted_entities": redacted_entities,
            "artifacts": [str(output_path)],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _cleanup(scratch_dir)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


@app.get("/artifacts/download")
def download_artifact(path: str) -> FileResponse:
    """Serve sanitized artifacts generated by the pipeline."""

    artifact_path = _resolve_artifact(path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    return FileResponse(artifact_path, filename=artifact_path.name)


@app.get("/audit/history")
def get_audit_history(limit: int = 50) -> list[dict[str, Any]]:
    """List recent audit logs from artifacts directory."""
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        return []

    logs = []
    for log_path in sorted(artifacts_dir.glob("*.audit.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logs.append({
                "filename": log_path.name,
                "timestamp": data.get("timestamp"),
                "modality": data.get("modality"),
                "source": Path(data.get("source", "")).name,
                "entity_count": data.get("entity_count", len(data.get("entities", []))),
                "detection_mode": data.get("detection_mode"),
                "output": Path(data.get("output", "")).name if data.get("output") else None,
            })
            if len(logs) >= limit:
                break
        except Exception:
            continue
    return logs


@app.get("/audit/history/{filename}")
def get_audit_log_file(filename: str) -> dict[str, Any]:
    """Retrieve full audit log JSON content by filename."""
    artifacts_dir = Path("artifacts")
    file_path = artifacts_dir / filename
    if not file_path.exists() or not filename.endswith(".audit.json"):
        raise HTTPException(status_code=404, detail="Audit log file not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read audit log: {exc}")


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


def _attach_download_payload(response: ScanResponse, result: DetectionResult) -> None:
    response.redacted_entities = _extract_redacted_labels(result)
    if not result.mitigated_output or not result.mitigated_output.exists():
        return

    mitigated_path = result.mitigated_output.resolve()
    base_dir = Path(CONFIG.app.output_dir).resolve()
    try:
        relative = mitigated_path.relative_to(base_dir)
        response.download_url = f"/artifacts/download?path={relative.as_posix()}"
    except ValueError:
        response.download_url = f"/artifacts/download?path={result.mitigated_output.name}"


def _attach_image_payload(response: ScanResponse, result: DetectionResult) -> None:
    _attach_download_payload(response, result)
    if result.mitigated_output and result.mitigated_output.exists():
        mitigated_path = result.mitigated_output.resolve()
        response.sanitized_image_base64 = base64.b64encode(mitigated_path.read_bytes()).decode("ascii")


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
