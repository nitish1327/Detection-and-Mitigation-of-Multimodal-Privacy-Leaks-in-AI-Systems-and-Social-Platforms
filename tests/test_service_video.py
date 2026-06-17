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
