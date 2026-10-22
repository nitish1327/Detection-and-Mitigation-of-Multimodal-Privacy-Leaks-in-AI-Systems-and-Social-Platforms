#!/usr/bin/env python3
"""Automated tests for Privacy Sandbox extension + LeakWatch backend."""

from __future__ import annotations

import json
import mimetypes
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

BACKEND = "http://127.0.0.1:8000"
EXTENSION_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXTENSION_DIR.parent
TEXT_TIMEOUT = 20
IMAGE_TIMEOUT = 180

SAMPLE_IMAGE_CANDIDATES = [
    REPO_ROOT / "samples_anu_original" / "adhaar_image.jpeg",
    REPO_ROOT / "samples_anu" / "adhaar_image.jpeg",
]

REQUIRED_FILES = [
    "manifest.json",
    "popup.html",
    "popup.css",
    "popup.js",
    "content.js",
    "background.js",
    "icons/icon16.png",
    "icons/icon48.png",
    "icons/icon128.png",
]

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def http_get(path: str, timeout: int = TEXT_TIMEOUT) -> tuple[int, dict | str | bytes]:
    req = urllib.request.Request(f"{BACKEND}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return resp.status, json.loads(body.decode("utf-8"))
        return resp.status, body


def http_post_json(path: str, payload: dict, timeout: int = TEXT_TIMEOUT) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_post_multipart_image(image_path: Path, timeout: int = IMAGE_TIMEOUT) -> tuple[int, dict]:
    boundary = "----LeakWatchBoundary7e9f2a"
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    file_bytes = image_path.read_bytes()

    body = BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode()
    )
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BACKEND}/scan/image/upload",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def resolve_sample_image() -> Path | None:
    for candidate in SAMPLE_IMAGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def test_extension_structure() -> None:
    for rel in REQUIRED_FILES:
        path = EXTENSION_DIR / rel
        record(f"File exists: {rel}", path.is_file(), str(path) if not path.is_file() else "")

    manifest = json.loads((EXTENSION_DIR / "manifest.json").read_text(encoding="utf-8"))
    record("manifest_version is 3", manifest.get("manifest_version") == 3)
    record("side panel registered", manifest.get("side_panel", {}).get("default_path") == "popup.html")
    record("content scripts defined", len(manifest.get("content_scripts", [])) >= 3)
    record(
        "localhost host_permissions",
        any("localhost:8000" in p for p in manifest.get("host_permissions", [])),
    )


def test_ui_tabs() -> None:
    html = (EXTENSION_DIR / "popup.html").read_text(encoding="utf-8")
    js = (EXTENSION_DIR / "popup.js").read_text(encoding="utf-8")
    record("Text tab present", 'data-tab="text"' in html and "tab-text" in html)
    record("Image tab present", 'data-tab="image"' in html and "tab-image" in html)
    record("Video tab present", 'data-tab="video"' in html and "tab-video" in html)
    record("Policy tab present", 'data-tab="policy"' in html and "tab-policy" in html)
    record("Audit tab present", 'data-tab="audit"' in html and "tab-audit" in html)
    record("Image upload control", 'id="image-input"' in html and 'type="file"' in html)
    record("Video upload control", 'id="video-input"' in html and 'type="file"' in html)
    record("Video face selector wired", "renderFaceGrid" in js and "video-face-grid" in html)
    record("Video selective controls", "handleSelectiveSanitize" in js and "video-apply-selective-btn" in html)
    record("Download button wired", "handleDownloadImage" in js and "download-btn" in html)
    record("Text sanitization flow wired", "requestTextScan" in js and "handleSendToLlm" in js)
    record(
        "Text entity selection controls",
        all(control in html for control in (
            'id="entity-selection-controls"',
            'id="select-all-entities-btn"',
            'id="clear-all-entities-btn"',
            'id="apply-mitigation-btn"',
        )),
    )
    record(
        "Text scan resets selection state",
        "detectedEntities = [];" in js and "selectedEntityTypes = new Set();" in js,
    )


def test_backend_health() -> None:
    try:
        status, body = http_get("/health")
        record("Backend connection", status == 200 and body.get("status") == "ok", str(body))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        record("Backend connection", False, f"offline: {exc}")


def test_text_sanitization() -> None:
    sample = "My email is abc@gmail.com"
    try:
        status, body = http_post_json("/scan/text", {"text": sample})
        sanitized = body.get("sanitized_text", "")
        entities = body.get("redacted_entities", [])
        has_sanitized = isinstance(sanitized, str) and sanitized != sample
        has_entities = isinstance(entities, list) and len(entities) > 0
        record(
            "Text sanitization works",
            status == 200 and has_sanitized and has_entities,
            f"sanitized={sanitized!r} entities={entities}",
        )
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        record("Text sanitization works", False, f"offline: {exc}")
    except Exception as exc:  # noqa: BLE001
        record("Text sanitization works", False, str(exc))


def test_image_sanitization() -> None:
    image_path = resolve_sample_image()
    if not image_path:
        record("Upload image sample available", False, "no sample image found")
        record("Image sanitization works", False, "skipped — no sample")
        record("Sanitized preview data", False, "skipped — no sample")
        record("Download image endpoint", False, "skipped — no sample")
        return

    record("Upload image sample available", True, str(image_path))

    try:
        status, body = http_post_multipart_image(image_path)
        entities = body.get("redacted_entities", [])
        has_base64 = bool(body.get("sanitized_image_base64"))
        has_download = bool(body.get("download_url") or body.get("mitigated_output"))
        record(
            "Image sanitization works",
            status == 200 and (has_base64 or has_download) and len(entities) > 0,
            f"entities={entities}",
        )
        record("Sanitized preview data", has_base64, "base64" if has_base64 else "missing base64")

        if body.get("download_url"):
            dl_status, dl_body = http_get(body["download_url"], timeout=IMAGE_TIMEOUT)
            record(
                "Download image endpoint",
                dl_status == 200 and isinstance(dl_body, bytes) and len(dl_body) > 0,
                f"bytes={len(dl_body) if isinstance(dl_body, bytes) else 0}",
            )
        else:
            record("Download image endpoint", has_base64, "via base64 only")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        record("Image sanitization works", False, f"offline: {exc}")
        record("Sanitized preview data", False, str(exc))
        record("Download image endpoint", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        record("Image sanitization works", False, str(exc))
        record("Sanitized preview data", False, str(exc))
        record("Download image endpoint", False, str(exc))


def test_copy_button_logic() -> None:
    popup_js = (EXTENSION_DIR / "popup.js").read_text(encoding="utf-8")
    record(
        "Copy button wired",
        "handleCopy" in popup_js and "navigator.clipboard.writeText" in popup_js,
    )


def test_injection_adapters() -> None:
    content_js = (EXTENSION_DIR / "content.js").read_text(encoding="utf-8")
    record("ChatGPT adapter", "chat.openai.com" in content_js and "chatgpt.com" in content_js)
    record("Gemini adapter", "gemini.google.com" in content_js)
    record("Claude adapter", "claude.ai" in content_js)
    record("Perplexity adapter", "perplexity.ai" in content_js)
    record("Copilot adapter", "copilot.microsoft.com" in content_js)
    record("HuggingChat adapter", "huggingface.co" in content_js)
    record("Poe adapter", "poe.com" in content_js)
    record("Twitter/X adapter", "twitter.com" in content_js and "x.com" in content_js)
    record("LinkedIn adapter", "linkedin.com" in content_js)
    record(
        "Injection message handler",
        "INJECT_SANITIZED_TEXT" in content_js and "injectSanitizedText" in content_js,
    )


def test_error_handling_in_popup() -> None:
    popup_js = (EXTENSION_DIR / "popup.js").read_text(encoding="utf-8")
    checks = [
        ("Backend offline handling", "Cannot reach LeakWatch backend" in popup_js),
        ("Empty response handling", "empty response" in popup_js.lower()),
        ("Invalid JSON handling", "invalid JSON" in popup_js),
        ("Network timeout handling", "timed out" in popup_js.lower()),
        ("No image selected handling", "Select an image" in popup_js),
        ("Unsupported format handling", "Unsupported format" in popup_js),
    ]
    for name, ok in checks:
        record(name, ok)


def main() -> int:
    print("Privacy Sandbox — automated test run\n" + "=" * 40)
    test_extension_structure()
    test_ui_tabs()
    test_backend_health()
    test_text_sanitization()
    test_image_sanitization()
    test_copy_button_logic()
    test_injection_adapters()
    test_error_handling_in_popup()

    print("\n" + "=" * 40)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"Summary: {passed}/{total} passed\n")

    manual = [
        "Popup opens (Text + Image tabs)",
        "Upload image preview in browser",
        "ChatGPT injection works",
        "Gemini injection works",
        "Claude injection works",
    ]
    print("Manual browser tests (see extension/TESTING.md):")
    for name in manual:
        print(f"  [MANUAL] {name}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
