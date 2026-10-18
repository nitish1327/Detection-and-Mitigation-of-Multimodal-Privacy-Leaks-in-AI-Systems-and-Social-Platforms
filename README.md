# LeakWatch Middleware

LeakWatch is a privacy-preserving middleware designed during the UE23CS320A Capstone project to detect and mitigate multimodal privacy leaks **before** user content reaches downstream AI systems or social platforms. This repository implements the Phase 2 scope described in the project PDFs: full support for text + image modalities, adapters for audio/video, explainability artifacts, and an orchestration API/CLI layer.

## Features
- **Text Privacy Detection**: spaCy NER, regex heuristics, and rule graphs.
- **Text Mitigation Strategies**: Realistic Faker and synthetic identity generation (replacing names with realistic counterparts like "Raj", realistic emails, phones, and addresses) with deterministic caching and audit trails.
- **Image Privacy Detection**: OpenCV 5.0-compatible ONNX YuNet deep face detector (`cv2.FaceDetectorYN`) with multi-angle rotation scanning, HSV color contour detection for ID cards, and OCR-assisted text scanning.
- **Image Mitigation**: Facial blurring, synthetic text rewrites with typography/color matching, and explainability overlay generation.
- **Video Privacy & Tracking Engine**: Reusable tracking engine (`leakwatch.video`) with velocity extrapolation, occlusion recovery, edge coasting, multi-face selection, inverted blur mode, and audio-preserving FFmpeg remuxing.
- **Zero-Leak Video PII Redaction**: Dynamic ID card contour detection (`find_card_region`), card-relative normalized coordinate tracking, face anchor persistence (zero-dropout guarantee), and 90° CCW inpainting for vertical text orientations.
- **Audio Privacy & Mitigation**: Speech-to-text via OpenAI Whisper with word-level timestamps and speaker overlap warnings, tone-masked beep redaction for sensitive windows, and auto-discovery of FFmpeg from venv/shim.
- **FastAPI REST Service**: Modular endpoints for text, image, audio, and video sanitization, including video face preview (`/scan/video/preview-faces`), selective video sanitization (`/scan/video/selective`), and audio upload (`/scan/audio/upload`).
- **Manifest V3 Chrome Extension**: Interactive browser extension supporting:
  - **Text**: PII sanitization + 1-click prompt injection into ChatGPT, Gemini, and Claude.
  - **Image**: Face and document redaction with side-by-side preview and downloads.
  - **Audio**: Device file upload + live in-browser voice recording, pre-sanitization playback, and on-the-spot tone masking.
  - **Video**: Interactive numbered face preview grid, selective face blurring toggle, blur slider, and in-popup player.
- **Audit & Explainability**: JSONL audit logs, span reports, and annotated preview overlays.

## Repository Layout
```
leakwatch/
  detection/         # modality detectors (text, image with OpenCV 5 YuNet ONNX)
  mitigation/        # mitigation strategies per modality
  video/             # video face tracking engine, preview, & audio remuxing
  explainability/    # highlighting + audit artifacts
  orchestration/     # pipeline manager + CLI wiring
  adapters/          # audio/video adapters
  utils/             # shared helpers, schemas, and ONNX model assets
service/
  app.py             # FastAPI REST middleware (text, image, audio, video)
extension/           # Chrome Extension (Manifest V3) for browser privacy
  popup.html         # multi-tab UI (Text, Image, Audio, Interactive Video)
  popup.js           # dynamic preview, face selection grid, API integration
  content.js         # injection adapters for ChatGPT, Gemini, and Claude
  run_tests.py       # automated extension integration test suite
config/
  leakwatch.yaml     # default runtime configuration
samples/
  text/              # demo text inputs
  images/            # demo image inputs
  audio/             # demo audio inputs
  video/             # demo video inputs
scripts/
  selective_face_blur.py # standalone CLI face tracking tool
tests/               # pytest suite (42 unit and integration tests)
```

## Getting Started
1. **Install dependencies**
   ```powershell
   D:/Capstone/.venv/Scripts/python.exe -m pip install -U pip
   D:/Capstone/.venv/Scripts/python.exe -m pip install -r requirements.txt
   ```
  The default requirements now include the audio transcription stack, so phone recordings can be processed without extra manual setup.
2. **Download NLP/OCR models** (after `pip install`):
   ```powershell
   D:/Capstone/.venv/Scripts/python.exe -m spacy download en_core_web_sm
   ```
3. **Run the CLI**
   ```powershell
   D:/Capstone/.venv/Scripts/python.exe -m leakwatch scan-text samples/text/demo_email.txt
   ```
  Other useful commands:
  ```powershell
  # Scan an image
  D:/Capstone/.venv/Scripts/python.exe -m leakwatch scan-image path/to/photo.jpg

  # Scan audio (works with phone recordings like .m4a/.mp3/.wav)
  D:/Capstone/.venv/Scripts/python.exe -m leakwatch scan-audio path/to/recording.wav

  # Scan video (extracts sparse frames and sanitizes them via the image pipeline)
  D:/Capstone/.venv/Scripts/python.exe -m leakwatch scan-video path/to/clip.mp4

  # Recursively scan a folder (text + images)
  D:/Capstone/.venv/Scripts/python.exe -m leakwatch scan-folder data/uploads
  ```

## Interactive Video Face Blur
LeakWatch includes a high-performance selective video anonymization engine with spatial proximity tracking, occlusion recovery, and audio preservation.

### Standalone CLI Tool
```powershell
# Interactive GUI mode (displays first frame; click faces to blur)
python scripts/selective_face_blur.py --video samples/video/Video1.mp4

# Headless / CLI selection: blur faces #1 and #3
python scripts/selective_face_blur.py --video samples/video/Video1.mp4 --face 1,3

# Inverted mode: blur everyone EXCEPT face #2
python scripts/selective_face_blur.py --video samples/video/Video1.mp4 --blur-all-except 2 --blur-strength 51
```

**Key Capabilities:**
- **OpenCV 5.0 YuNet ONNX**: Uses deep learning face detection (`face_detection_yunet_2023mar.onnx`) with multi-angle rotation invariance.
- **Multi-face Tracking**: Tracks multiple individuals simultaneously with velocity extrapolation across frames.
- **Occlusion Recovery**: Retains spatial anchors during occlusions and re-identifies targets when they reappear.
- **Edge Coasting**: Smoothly predicts trajectories when subjects move off-screen or near frame borders.
- **Audio Preservation**: Automatically remuxes the original audio track into the output using FFmpeg or bundled `imageio-ffmpeg`.

---

## Chrome Browser Extension (Privacy Sandbox)
LeakWatch includes a full Manifest V3 browser extension under [`extension/`](file:///D:/LEAKWATCH_CAPSTONE/-Detection-and-Mitigation-of-Multimodal-Privacy-Leaks-in-AI-Systems-and-Social-Platforms/extension/) for protecting sensitive multimodal data directly inside web browsers.

### Multimodal Tabs:
- **Text Tab**: Scans text for PII/secrets, replaces with synthetic entities, and features **one-click direct prompt injection** into ChatGPT (`chatgpt.com`), Google Gemini (`gemini.google.com`), and Anthropic Claude (`claude.ai`).
- **Image Tab**: Redacts faces, Aadhaar cards, and document PII with side-by-side previews and direct download.
- **Audio Tab**:
  - **Device Upload**: Upload `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`, or `.webm` files.
  - **Live Voice Recording**: Record voice directly inside the extension (`MediaRecorder`), listen to the recording before sanitizing, and sanitize on the spot with audible beep tones.
  *(Note: Allow microphone permission once via `chrome://settings/content/microphone` or by visiting `chrome-extension://<id>/popup.html` in a tab).*
- **Video Tab (Interactive Multi-Face Workflow)**:
  1. **Upload Video**: Automatically invokes `/scan/video/preview-faces` upon file selection.
  2. **Face Selector Grid**: Renders numbered face crop cards with checkboxes for visual selection.
  3. **Mitigation Controls**: Radio toggle between *Blur selected faces* and *Blur all except selected*, plus an adjustable blur strength slider.
  4. **Action & Player**: Triggers selective tracking or blanket fallback blur, loading the output into an in-popup HTML5 video player with download support.

To install: Open Chrome -> navigate to `chrome://extensions` -> enable **Developer Mode** -> click **Load unpacked** -> select the `extension/` directory.

---

### Sample Media Assets
- Audio: `samples/audio/demo_call.wav` (plus transcript `demo_call.txt`) demonstrates the `scan-audio` flow. For real phone recordings, LeakWatch can:
  - use companion transcript if present (`<name>.txt`), or
  - auto-transcribe using Whisper when transcript is missing.
  Non-WAV inputs are normalized with ffmpeg before mitigation. The pipeline emits:
  - `*.audio.sanitized.txt` (sanitized transcript)
  - `*.audio.beeps.json` (beep windows)
  - `*.sanitized.wav` (audio with sensitive ranges replaced by tone)
- Video: `samples/video/Video1.mp4` and `samples/video/demo_clip.mp4` provide real-world and synthetic clips for selective tracking and OCR walkthroughs.

---

## REST API Middleware
LeakWatch ships with a FastAPI backend service (`service/app.py`).

1. **Start the service**
  ```powershell
  python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
  ```
2. **Core Endpoints**:
  - `POST /scan/text`: Scan and redact text with customizable policies.
  - `POST /scan/image` & `POST /scan/image/upload`: Sanitize document and face PII in images.
  - `POST /scan/audio` & `POST /scan/audio/upload`: Sanitize speech with tone-masked sensitive terms.
  - `POST /scan/video`: Blanket video sanitization via sparse frame processing.
  - `POST /scan/video/preview-faces`: Upload video (multipart or JSON) to detect faces in frame 1; returns face count, annotated frame preview, and individual face crop thumbnails.
  - `POST /scan/video/selective`: Selectively track and blur target faces (or invert selection) with audio preservation.
  - `GET /artifacts/download`: Download generated sanitized media.

3. **Example Request (`/scan/video/preview-faces`)**
  ```powershell
  curl -X POST http://localhost:8000/scan/video/preview-faces ^
    -F "file=@samples/video/Video1.mp4"
  ```

---

## Testing
Run the automated test suites:
```powershell
# 1. Run core Pytest unit and integration suite (43 tests)
python -m pytest tests/ -q

# 2. Run Chrome Extension integrity and API integration suite (39 tests)
python extension/run_tests.py
```

## Phone Audio Quickstart
1. Install dependencies with `requirements.txt`; it now includes the audio stack by default.
2. If system `ffmpeg` is not available, LeakWatch falls back to the bundled ffmpeg shipped with `imageio-ffmpeg`.
3. Copy your recording (for example `my_call.m4a`) into the workspace.
4. Run:
```powershell
D:/Capstone/.venv/Scripts/python.exe -m leakwatch scan-audio samples/audio/my_call.m4a
```
5. Check artifacts in `artifacts/` (`*.sanitized.wav`, `*.audio.sanitized.txt`, `*.audio.beeps.json`).

## Documentation
- `docs/IMPLEMENTATION_PLAN.md`: bridge between Phase 1 report and this implementation.
- Additional docs (`ARCHITECTURE.md`, pipeline diagrams, etc.) will be added as modules are completed.

## Status
Project is under active development. Remaining stretch goals include richer regex coverage, image test automation, audio/video UX polish, and (future) GraphSAGE/GAN experiments once the deterministic baseline is fully signed off.
