from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from leakwatch.orchestration import VideoProcessingError
from service import app as service_app


def test_video_endpoint_fails_closed_when_processing_fails(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_bytes(b"not-a-real-video")

    def _fail_video_processing(path: Path):
        raise VideoProcessingError(f"Video processing failed for {Path(path).name}: corrupt stream")

    monkeypatch.setattr(service_app.PIPELINE, "process_video", _fail_video_processing)

    response = TestClient(service_app.app).post(
        "/scan/video",
        json={"path": str(video_path), "policy": {"block_on": []}},
    )

    assert response.status_code == 422
    assert "corrupt stream" in response.json()["detail"]
    assert "decision" not in response.json()


def test_preview_faces_endpoint_json() -> None:
    video_path = Path("samples/video/Video1.mp4")
    if not video_path.exists():
        return

    client = TestClient(service_app.app)
    response = client.post(
        "/scan/video/preview-faces",
        json={"path": str(video_path)},
    )

    assert response.status_code == 200
    data = response.json()
    assert "face_count" in data
    assert data["face_count"] > 0
    assert "preview_image_base64" in data
    assert len(data["preview_image_base64"]) > 0
    assert "faces" in data
    assert len(data["faces"]) == data["face_count"]
    first_face = data["faces"][0]
    assert first_face["id"] == 1
    assert "bbox" in first_face
    assert len(first_face["bbox"]) == 4
    assert "thumbnail_base64" in first_face
    assert len(first_face["thumbnail_base64"]) > 0


def test_preview_faces_endpoint_multipart() -> None:
    video_path = Path("samples/video/Video1.mp4")
    if not video_path.exists():
        return

    client = TestClient(service_app.app)
    with open(video_path, "rb") as f:
        response = client.post(
            "/scan/video/preview-faces",
            files={"file": ("Video1.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["face_count"] > 0
    assert bool(data["preview_image_base64"])


def test_selective_video_endpoint_json() -> None:
    video_path = Path("samples/video/Video1.mp4")
    if not video_path.exists():
        return

    client = TestClient(service_app.app)
    response = client.post(
        "/scan/video/selective",
        json={
            "path": str(video_path),
            "selected_faces": [1],
            "mode": "blur_selected",
            "blur_strength": 51,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["decision"] == "allow"
    assert "download_url" in data
    assert "mitigated_output" in data
    assert Path(data["mitigated_output"]).exists()
    assert data["faces_blurred"] == 1
    assert data["selected_faces"] == [1]

    # Verify download endpoint serves the video
    dl_resp = client.get(data["download_url"])
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 0


def test_selective_video_endpoint_multipart() -> None:
    video_path = Path("samples/video/Video1.mp4")
    if not video_path.exists():
        return

    client = TestClient(service_app.app)
    with open(video_path, "rb") as f:
        response = client.post(
            "/scan/video/selective",
            files={"file": ("Video1.mp4", f, "video/mp4")},
            data={"selected_faces": "[1]", "mode": "blur_selected"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["faces_blurred"] == 1
