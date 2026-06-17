#!/usr/bin/env python3
"""Automated tests for Privacy Sandbox extension + LeakWatch backend."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = "http://127.0.0.1:8000"
EXTENSION_DIR = Path(__file__).resolve().parent
TIMEOUT = 20

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


def http_get(path: str) -> tuple[int, dict | str]:
    req = urllib.request.Request(f"{BACKEND}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
        try:
            return resp.status, json.loads(body)
        except json.JSONDecodeError:
            return resp.status, body


def http_post(path: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_extension_structure() -> None:
    for rel in REQUIRED_FILES:
        path = EXTENSION_DIR / rel
        record(f"File exists: {rel}", path.is_file(), str(path) if not path.is_file() else "")

    manifest = json.loads((EXTENSION_DIR / "manifest.json").read_text(encoding="utf-8"))
    record("manifest_version is 3", manifest.get("manifest_version") == 3)
    record("popup registered", manifest.get("action", {}).get("default_popup") == "popup.html")
    record("content scripts defined", len(manifest.get("content_scripts", [])) >= 3)
    record(
        "localhost host_permissions",
        any("localhost:8000" in p for p in manifest.get("host_permissions", [])),
    )


def test_backend_health() -> None:
    try:
        status, body = http_get("/health")
        record("Backend connection", status == 200 and body.get("status") == "ok", str(body))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        record("Backend connection", False, f"offline: {exc}")


def test_sanitization() -> None:
    sample = "My email is abc@gmail.com"
    try:
        status, body = http_post("/scan/text", {"text": sample})
        sanitized = body.get("sanitized_text", "")
        entities = body.get("redacted_entities", [])
        has_sanitized = isinstance(sanitized, str) and sanitized != sample
        has_entities = isinstance(entities, list) and len(entities) > 0
        record(
            "Sanitization works",
            status == 200 and has_sanitized and has_entities,
            f"sanitized={sanitized!r} entities={entities}",
        )
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        record("Sanitization works", False, f"offline: {exc}")
    except Exception as exc:  # noqa: BLE001
        record("Sanitization works", False, str(exc))


def test_copy_button_logic() -> None:
    """Popup exposes copy handler and clipboard API usage in source."""
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
    ]
    for name, ok in checks:
        record(name, ok)


def main() -> int:
    print("Privacy Sandbox — automated test run\n" + "=" * 40)
    test_extension_structure()
    test_backend_health()
    test_sanitization()
    test_copy_button_logic()
    test_injection_adapters()
    test_error_handling_in_popup()

    print("\n" + "=" * 40)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"Summary: {passed}/{total} passed\n")

    browser_tests = [
        "Popup opens",
        "ChatGPT injection works",
        "Gemini injection works",
        "Claude injection works",
    ]
    print("Manual browser tests (see extension/TESTING.md):")
    for name in browser_tests:
        print(f"  [MANUAL] {name}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
