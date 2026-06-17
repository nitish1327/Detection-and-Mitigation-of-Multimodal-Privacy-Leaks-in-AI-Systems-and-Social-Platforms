# Privacy Sandbox — Installation

Privacy Sandbox is a Chrome Extension (Manifest V3) that sanitizes prompts through your local LeakWatch FastAPI backend before you paste them into ChatGPT, Gemini, or Claude.

## Prerequisites

1. **Google Chrome** (or Chromium-based browser with MV3 support)
2. **Python 3.10+** with LeakWatch dependencies installed
3. **spaCy model**: `python -m spacy download en_core_web_sm`

## Step 1 — Start the LeakWatch backend

From the repository root:

```powershell
cd "path\to\-Detection-and-Mitigation-of-Multimodal-Privacy-Leaks-in-AI-Systems-and-Social-Platforms"
python -m pip install -r requirements.txt
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Verify the server is running:

```powershell
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok","service":"leakwatch"}
```

## Step 2 — Generate extension icons (first time only)

```powershell
cd extension
python generate_icons.py
```

This creates `icons/icon16.png`, `icon48.png`, and `icon128.png`.

## Step 3 — Load the extension in Chrome

1. Open `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder in this repository
5. Pin **Privacy Sandbox** to the toolbar for quick access

## Step 4 — Use the extension

1. Click the Privacy Sandbox icon
2. Type or paste your prompt in the textarea
3. Click **Sanitize** — the extension calls `POST http://localhost:8000/scan/text`
4. Review the sanitized preview and detected entities
5. **Copy Sanitized Text** — copies to clipboard
6. **Send To LLM** — injects sanitized text into the active ChatGPT, Gemini, or Claude tab

## Security model

| Property | Behavior |
|----------|----------|
| Network | `localhost:8000` only — no external cloud calls |
| Storage | No `chrome.storage`, no cookies, no telemetry |
| Permissions | `activeTab`, `scripting`, `clipboardWrite` + host permissions for LLM sites |
| Data flow | Text leaves the browser only to your local LeakWatch instance |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Red status dot | Start `uvicorn service.app:app --port 8000` |
| Request timed out | Check firewall; ensure port 8000 is free |
| Content script not loaded | Refresh the LLM page after installing the extension |
| Chat input not found | Click inside the message box, then retry **Send To LLM** |
| Invalid JSON | Restart the API server after pulling latest code |

## Updating

1. Pull latest code
2. Go to `chrome://extensions/` → click **Reload** on Privacy Sandbox
3. Restart the backend if `service/app.py` changed
