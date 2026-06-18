# Privacy Sandbox — Testing Checklist

## Automated tests (no browser required)

Run from the repository root:

```powershell
python extension/run_tests.py
```

This validates backend health, text + image sanitization APIs, extension file structure, tab UI, and content-script adapter logic.

Sample images are read from `samples_anu_original/` if present, otherwise `samples_anu/`.

## Manual browser tests

### Setup

- [ ] LeakWatch backend running at `http://localhost:8000`
- [ ] Extension loaded unpacked from `extension/`
- [ ] Extension pinned to toolbar

### Popup & backend

- [ ] Click extension icon — popup opens without errors
- [ ] Status dot is **green** when backend is running
- [ ] Status dot is **red** when backend is stopped
### Text tab

- [ ] **Text** tab — enter `My email is abc@gmail.com` and click **Sanitize**
- [ ] Preview shows original and sanitized text side by side
- [ ] Detected entities list shows labels (e.g. EMAIL)
- [ ] **Copy Sanitized Text** copies to clipboard

### Image tab

- [ ] Switch to **Image** tab
- [ ] Upload `samples_anu/adhaar_image.jpeg` (or `samples_anu_original/`)
- [ ] Original preview appears immediately
- [ ] Click **Sanitize Image** — status dot green, no errors
- [ ] Sanitized preview appears (face blur, Aadhaar/name redaction)
- [ ] Detected entities include labels such as FACE, AADHAR, PERSON
- [ ] **Download Sanitized Image** saves a file locally

### Error handling

- [ ] Stop backend → Sanitize shows clear offline error
- [ ] Empty textarea → Sanitize shows validation error
- [ ] Image tab with no file → Sanitize shows validation error
- [ ] Upload non-image file → unsupported format error
- [ ] Non-LLM tab active → **Send To LLM** shows unsupported-site error

### ChatGPT injection

- [ ] Open https://chatgpt.com or https://chat.openai.com
- [ ] Sanitize a prompt with PII in the extension popup
- [ ] Click **Send To LLM**
- [ ] Sanitized text appears in the chat input box

### Gemini injection

- [ ] Open https://gemini.google.com
- [ ] Sanitize a prompt with PII
- [ ] Click **Send To LLM**
- [ ] Sanitized text appears in the Gemini input area

### Claude injection

- [ ] Open https://claude.ai
- [ ] Sanitize a prompt with PII
- [ ] Click **Send To LLM**
- [ ] Sanitized text appears in the Claude message box

### Security verification

- [ ] DevTools → Network tab during Sanitize — requests go only to `localhost:8000`
- [ ] No requests to third-party analytics domains
- [ ] `chrome://extensions/` → Privacy Sandbox → no `storage` permission

## Expected sanitization example

**Input:**

```
My email is abc@gmail.com and my phone is 555-123-4567
```

**Preview (approximate):**

```
My email is [EMAIL] and my phone is [PHONE]
```

**Detected:**

- EMAIL
- PHONE

(Exact mask style depends on `config/leakwatch.yaml` — faker, brackets, etc.)
