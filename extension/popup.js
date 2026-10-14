"use strict";

const BACKEND_BASE = "http://localhost:8000";
const TEXT_SCAN_ENDPOINT = `${BACKEND_BASE}/scan/text`;
const IMAGE_UPLOAD_ENDPOINT = `${BACKEND_BASE}/scan/image/upload`;
const AUDIO_UPLOAD_ENDPOINT = `${BACKEND_BASE}/scan/audio/upload`;
const VIDEO_UPLOAD_ENDPOINT = `${BACKEND_BASE}/scan/video/upload`;
const VIDEO_PREVIEW_FACES_ENDPOINT = `${BACKEND_BASE}/scan/video/preview-faces`;
const VIDEO_SELECTIVE_ENDPOINT = `${BACKEND_BASE}/scan/video/selective`;
const HEALTH_ENDPOINT = `${BACKEND_BASE}/health`;
const TEXT_TIMEOUT_MS = 15000;
const IMAGE_TIMEOUT_MS = 120000;
const AUDIO_TIMEOUT_MS = 120000;
const VIDEO_TIMEOUT_MS = 300000;

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/bmp",
  "image/webp",
  "image/pjpeg",
]);

const SUPPORTED_AUDIO_TYPES = new Set([
  "audio/wav",
  "audio/x-wav",
  "audio/mp3",
  "audio/mpeg",
  "audio/m4a",
  "audio/mp4",
  "audio/aac",
  "audio/flac",
  "audio/ogg",
  "audio/opus",
]);

const SUPPORTED_VIDEO_TYPES = new Set([
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
  "video/webm",
  "video/m4v",
]);

// Shared UI
const errorPanel = document.getElementById("error-panel");
const backendStatus = document.getElementById("backend-status");

// Text tab
const inputText = document.getElementById("input-text");
const sanitizeBtn = document.getElementById("sanitize-btn");
const copyBtn = document.getElementById("copy-btn");
const sendBtn = document.getElementById("send-btn");
const previewPanel = document.getElementById("preview-panel");
const originalPreview = document.getElementById("original-preview");
const sanitizedPreview = document.getElementById("sanitized-preview");
const entitiesList = document.getElementById("entities-list");
const noEntities = document.getElementById("no-entities");

// Image tab
const imageInput = document.getElementById("image-input");
const imageSanitizeBtn = document.getElementById("image-sanitize-btn");
const imagePreviewPanel = document.getElementById("image-preview-panel");
const originalImageEl = document.getElementById("original-image");
const sanitizedImageEl = document.getElementById("sanitized-image");
const imageEntitiesList = document.getElementById("image-entities-list");
const imageNoEntities = document.getElementById("image-no-entities");
const downloadBtn = document.getElementById("download-btn");

// Audio tab
const audioInput = document.getElementById("audio-input");
const audioSanitizeBtn = document.getElementById("audio-sanitize-btn");
const audioPreviewPanel = document.getElementById("audio-preview-panel");
const sanitizedAudioEl = document.getElementById("sanitized-audio");
const audioEntitiesList = document.getElementById("audio-entities-list");
const audioNoEntities = document.getElementById("audio-no-entities");
const audioDownloadBtn = document.getElementById("audio-download-btn");

// Video tab
const videoInput = document.getElementById("video-input");
const videoScanStatus = document.getElementById("video-scan-status");
const videoFaceSelectionPanel = document.getElementById("video-face-selection-panel");
const videoFaceCountText = document.getElementById("video-face-count-text");
const videoAnnotatedFrameContainer = document.getElementById("video-annotated-frame-container");
const videoAnnotatedFrame = document.getElementById("video-annotated-frame");
const videoFaceGrid = document.getElementById("video-face-grid");
const videoBlurStrengthSlider = document.getElementById("video-blur-strength");
const videoBlurStrengthVal = document.getElementById("video-blur-strength-val");
const videoApplySelectiveBtn = document.getElementById("video-apply-selective-btn");
const videoSanitizeBtn = document.getElementById("video-sanitize-btn");
const videoPreviewPanel = document.getElementById("video-preview-panel");
const sanitizedVideoEl = document.getElementById("sanitized-video");
const videoEntitiesList = document.getElementById("video-entities-list");
const videoNoEntities = document.getElementById("video-no-entities");
const videoDownloadBtn = document.getElementById("video-download-btn");

// Tabs
const tabButtons = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");

let sanitizedText = "";
let selectedImageFile = null;
let originalImageUrl = null;
let sanitizedImageUrl = null;
let sanitizedImageBlob = null;
let sanitizedDownloadName = "sanitized.png";

let selectedAudioFile = null;
let sanitizedAudioUrl = null;
let sanitizedAudioBlob = null;
let sanitizedAudioDownloadName = "sanitized.wav";

let selectedVideoFile = null;
let sanitizedVideoUrl = null;
let sanitizedVideoBlob = null;
let sanitizedVideoDownloadName = "sanitized.mp4";
let detectedVideoFaces = [];
let selectedFaceIds = new Set();

document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  initTabs();
  initTextTab();
  initImageTab();
  initAudioTab();
  initVideoTab();
});

window.addEventListener("unload", () => {
  revokeImageUrls();
  revokeAudioUrls();
  revokeVideoUrls();
});

// =============================================================================
// Shared helpers
// =============================================================================

function initTabs() {
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
}

function switchTab(tabName) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
    panel.classList.toggle("hidden", panel.id !== `tab-${tabName}`);
  });
  clearError();
}

async function checkBackendHealth() {
  try {
    const response = await fetchWithTimeout(HEALTH_ENDPOINT, { method: "GET" }, 5000);
    if (!response.ok) {
      setBackendStatus("error");
      return;
    }
    const data = await response.json();
    setBackendStatus(data.status === "ok" ? "ok" : "error");
  } catch {
    setBackendStatus("error");
  }
}

function setBackendStatus(state) {
  backendStatus.className = "status-badge";
  if (state === "ok") {
    backendStatus.classList.add("status-ok");
    backendStatus.title = "LeakWatch backend connected";
  } else if (state === "error") {
    backendStatus.classList.add("status-error");
    backendStatus.title = "LeakWatch backend offline";
  } else {
    backendStatus.classList.add("status-unknown");
    backendStatus.title = "Checking backend…";
  }
}

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.classList.remove("hidden");
}

function clearError() {
  errorPanel.textContent = "";
  errorPanel.classList.add("hidden");
}

function renderEntityList(listElement, emptyElement, entities) {
  listElement.innerHTML = "";
  const unique = [...new Set(entities.map((e) => String(e).toUpperCase()))];
  if (unique.length === 0) {
    emptyElement.classList.remove("hidden");
  } else {
    emptyElement.classList.add("hidden");
    unique.forEach((label) => {
      const li = document.createElement("li");
      li.textContent = `• ${label}`;
      listElement.appendChild(li);
    });
  }
}

function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  return fetch(url, { ...options, signal: controller.signal })
    .catch((error) => {
      if (error.name === "AbortError") {
        throw new Error(
          `Request timed out after ${Math.round(timeoutMs / 1000)}s. Is the backend running?`
        );
      }
      throw new Error(
        "Cannot reach LeakWatch backend at localhost:8000. Start it with: uvicorn service.app:app --port 8000"
      );
    })
    .finally(() => clearTimeout(timer));
}

async function parseJsonResponse(response) {
  const raw = await response.text();
  if (!raw) {
    throw new Error("Backend returned an empty response.");
  }
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error("Backend returned invalid JSON.");
  }
}

// =============================================================================
// Text tab (existing functionality preserved)
// =============================================================================

function initTextTab() {
  sanitizeBtn.addEventListener("click", handleSanitize);
  copyBtn.addEventListener("click", handleCopy);
  sendBtn.addEventListener("click", handleSendToLlm);
}

async function handleSanitize() {
  const text = inputText.value.trim();
  clearError();

  if (!text) {
    showError("Enter some text to sanitize.");
    return;
  }

  sanitizeBtn.classList.add("loading");
  sanitizeBtn.textContent = "Sanitizing…";
  copyBtn.disabled = true;
  sendBtn.disabled = true;

  try {
    const result = await sanitizeText(text);
    sanitizedText = result.sanitized_text || text;
    renderTextPreview(text, sanitizedText, result.redacted_entities || []);
    setBackendStatus("ok");
  } catch (error) {
    showError(error.message);
    previewPanel.classList.add("hidden");
    setBackendStatus("error");
  } finally {
    sanitizeBtn.classList.remove("loading");
    sanitizeBtn.textContent = "Sanitize";
  }
}

async function sanitizeText(text) {
  const response = await fetchWithTimeout(
    TEXT_SCAN_ENDPOINT,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    },
    TEXT_TIMEOUT_MS
  );

  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    const detail = payload.detail || payload.message || response.statusText;
    throw new Error(`Sanitization failed (${response.status}): ${detail}`);
  }

  if (!payload.sanitized_text && payload.sanitized_text !== "") {
    if (payload.mitigated_output) {
      throw new Error(
        "Backend did not return sanitized_text. Restart the LeakWatch API server."
      );
    }
    throw new Error("Backend response missing sanitized_text.");
  }

  return payload;
}

function renderTextPreview(original, sanitized, entities) {
  originalPreview.textContent = original;
  sanitizedPreview.textContent = sanitized;
  renderEntityList(entitiesList, noEntities, entities);
  previewPanel.classList.remove("hidden");
  copyBtn.disabled = false;
  sendBtn.disabled = false;
}

async function handleCopy() {
  if (!sanitizedText) {
    showError("Nothing to copy. Sanitize text first.");
    return;
  }

  try {
    await navigator.clipboard.writeText(sanitizedText);
    const originalLabel = copyBtn.textContent;
    copyBtn.textContent = "Copied!";
    setTimeout(() => {
      copyBtn.textContent = originalLabel;
    }, 1500);
    clearError();
  } catch {
    showError("Clipboard access denied. Check extension permissions.");
  }
}

async function handleSendToLlm() {
  if (!sanitizedText) {
    showError("Nothing to send. Sanitize text first.");
    return;
  }

  sendBtn.disabled = true;
  sendBtn.textContent = "Injecting…";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) {
      throw new Error("No active tab found. Open ChatGPT, Gemini, or Claude first.");
    }

    const url = tab.url || "";
    if (!isSupportedLlmUrl(url)) {
      throw new Error(
        "Active tab is not a supported LLM site. Open ChatGPT, Gemini, or Claude."
      );
    }

    const response = await chrome.tabs.sendMessage(tab.id, {
      type: "INJECT_SANITIZED_TEXT",
      text: sanitizedText,
    });

    if (!response || !response.success) {
      throw new Error(response?.error || "Failed to inject text into the chat input.");
    }

    clearError();
    window.close();
  } catch (error) {
    const message = error.message || String(error);
    if (message.includes("Receiving end does not exist")) {
      showError("Content script not loaded. Refresh the LLM page and try again.");
    } else {
      showError(message);
    }
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send To LLM";
  }
}

function isSupportedLlmUrl(url) {
  const supported = [
    "chat.openai.com",
    "chatgpt.com",
    "gemini.google.com",
    "claude.ai",
  ];
  try {
    const hostname = new URL(url).hostname;
    return supported.some((host) => hostname === host || hostname.endsWith(`.${host}`));
  } catch {
    return false;
  }
}

// =============================================================================
// Image tab
// =============================================================================

function initImageTab() {
  imageInput.addEventListener("change", handleImageSelected);
  imageSanitizeBtn.addEventListener("click", handleImageSanitize);
  downloadBtn.addEventListener("click", handleDownloadImage);
}

function revokeImageUrls() {
  if (originalImageUrl) {
    URL.revokeObjectURL(originalImageUrl);
    originalImageUrl = null;
  }
  if (sanitizedImageUrl) {
    URL.revokeObjectURL(sanitizedImageUrl);
    sanitizedImageUrl = null;
  }
  sanitizedImageBlob = null;
}

function handleImageSelected(event) {
  clearError();
  revokeImageUrls();
  downloadBtn.disabled = true;
  sanitizedImageEl.removeAttribute("src");

  const file = event.target.files?.[0];
  if (!file) {
    selectedImageFile = null;
    imageSanitizeBtn.disabled = true;
    imagePreviewPanel.classList.add("hidden");
    return;
  }

  if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
    selectedImageFile = null;
    imageSanitizeBtn.disabled = true;
    imageInput.value = "";
    showError(`Unsupported format "${file.type || "unknown"}". Use PNG, JPEG, BMP, or WebP.`);
    return;
  }

  selectedImageFile = file;
  originalImageUrl = URL.createObjectURL(file);
  originalImageEl.src = originalImageUrl;
  imageSanitizeBtn.disabled = false;
  imagePreviewPanel.classList.remove("hidden");
}

async function handleImageSanitize() {
  clearError();

  if (!selectedImageFile) {
    showError("Select an image to sanitize.");
    return;
  }

  imageSanitizeBtn.classList.add("loading");
  imageSanitizeBtn.textContent = "Sanitizing…";
  imageSanitizeBtn.disabled = true;
  downloadBtn.disabled = true;

  try {
    const result = await sanitizeImage(selectedImageFile);
    const blob = await resolveSanitizedImageBlob(result);
    if (sanitizedImageUrl) {
      URL.revokeObjectURL(sanitizedImageUrl);
    }
    sanitizedImageBlob = blob;
    sanitizedImageUrl = URL.createObjectURL(blob);
    sanitizedImageEl.src = sanitizedImageUrl;

    const stem = selectedImageFile.name.replace(/\.[^.]+$/, "");
    const ext = guessExtensionFromBlob(blob) || ".png";
    sanitizedDownloadName = `${stem}.sanitized${ext}`;

    renderEntityList(
      imageEntitiesList,
      imageNoEntities,
      result.redacted_entities || extractEntityLabels(result.entities)
    );

    imagePreviewPanel.classList.remove("hidden");
    downloadBtn.disabled = false;
    setBackendStatus("ok");
  } catch (error) {
    showError(error.message);
    setBackendStatus("error");
  } finally {
    imageSanitizeBtn.classList.remove("loading");
    imageSanitizeBtn.textContent = "Sanitize Image";
    imageSanitizeBtn.disabled = !selectedImageFile;
  }
}

async function sanitizeImage(file) {
  const formData = new FormData();
  formData.append("file", file, file.name);

  const response = await fetchWithTimeout(
    IMAGE_UPLOAD_ENDPOINT,
    { method: "POST", body: formData },
    IMAGE_TIMEOUT_MS
  );

  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    const detail = payload.detail || payload.message || response.statusText;
    throw new Error(`Image sanitization failed (${response.status}): ${detail}`);
  }

  const hasBase64 = Boolean(payload.sanitized_image_base64);
  const hasDownload = Boolean(payload.download_url || payload.mitigated_output);
  if (!hasBase64 && !hasDownload) {
    throw new Error("Backend response missing sanitized image data.");
  }

  return payload;
}

async function resolveSanitizedImageBlob(payload) {
  if (payload.sanitized_image_base64) {
    return base64ToBlob(payload.sanitized_image_base64, guessMimeFromPath(payload.mitigated_output));
  }

  const downloadPath = payload.download_url || artifactDownloadPath(payload.mitigated_output);
  if (!downloadPath) {
    throw new Error("No download URL available for sanitized image.");
  }

  const url = downloadPath.startsWith("http")
    ? downloadPath
    : `${BACKEND_BASE}${downloadPath.startsWith("/") ? "" : "/"}${downloadPath}`;

  const response = await fetchWithTimeout(url, { method: "GET" }, IMAGE_TIMEOUT_MS);
  if (!response.ok) {
    throw new Error(`Failed to download sanitized image (${response.status}).`);
  }
  const blob = await response.blob();
  if (!blob || blob.size === 0) {
    throw new Error("Downloaded sanitized image is empty.");
  }
  return blob;
}

function artifactDownloadPath(mitigatedOutput) {
  if (!mitigatedOutput) return null;
  const normalized = String(mitigatedOutput).replace(/\\/g, "/");
  const filename = normalized.split("/").pop();
  return filename ? `/artifacts/download?path=${encodeURIComponent(filename)}` : null;
}

function extractEntityLabels(entities) {
  if (!Array.isArray(entities)) return [];
  return entities
    .filter((e) => e && e.mitigation !== "none")
    .map((e) => String(e.label || "").toUpperCase())
    .filter(Boolean);
}

function base64ToBlob(base64, mimeType) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mimeType || "image/png" });
}

function guessMimeFromPath(pathValue) {
  if (!pathValue) return "image/png";
  const lower = String(pathValue).toLowerCase();
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".webp")) return "image/webp";
  if (lower.endsWith(".bmp")) return "image/bmp";
  return "image/png";
}

function guessExtensionFromBlob(blob) {
  const type = blob.type || "image/png";
  if (type.includes("jpeg") || type.includes("jpg")) return ".jpeg";
  if (type.includes("webp")) return ".webp";
  if (type.includes("bmp")) return ".bmp";
  return ".png";
}

function handleDownloadImage() {
  if (!sanitizedImageBlob) {
    showError("Nothing to download. Sanitize an image first.");
    return;
  }

  const url = URL.createObjectURL(sanitizedImageBlob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = sanitizedDownloadName;
  anchor.click();
  URL.revokeObjectURL(url);
  clearError();
}

// =============================================================================
// Audio tab
// =============================================================================

function initAudioTab() {
  audioInput.addEventListener("change", handleAudioSelected);
  audioSanitizeBtn.addEventListener("click", handleAudioSanitize);
  audioDownloadBtn.addEventListener("click", handleDownloadAudio);
}

function revokeAudioUrls() {
  if (sanitizedAudioUrl) {
    URL.revokeObjectURL(sanitizedAudioUrl);
    sanitizedAudioUrl = null;
  }
  sanitizedAudioBlob = null;
}

function handleAudioSelected(event) {
  clearError();
  revokeAudioUrls();
  audioDownloadBtn.disabled = true;
  sanitizedAudioEl.removeAttribute("src");

  const file = event.target.files?.[0];
  if (!file) {
    selectedAudioFile = null;
    audioSanitizeBtn.disabled = true;
    audioPreviewPanel.classList.add("hidden");
    return;
  }

  selectedAudioFile = file;
  audioSanitizeBtn.disabled = false;
  audioPreviewPanel.classList.add("hidden");
}

async function handleAudioSanitize() {
  clearError();

  if (!selectedAudioFile) {
    showError("Select an audio file to sanitize.");
    return;
  }

  audioSanitizeBtn.classList.add("loading");
  audioSanitizeBtn.textContent = "Sanitizing…";
  audioSanitizeBtn.disabled = true;
  audioDownloadBtn.disabled = true;

  try {
    const result = await sanitizeMedia(selectedAudioFile, AUDIO_UPLOAD_ENDPOINT, AUDIO_TIMEOUT_MS);
    const blob = await resolveSanitizedMediaBlob(result, "audio/wav", AUDIO_TIMEOUT_MS);
    if (sanitizedAudioUrl) {
      URL.revokeObjectURL(sanitizedAudioUrl);
    }
    sanitizedAudioBlob = blob;
    sanitizedAudioUrl = URL.createObjectURL(blob);
    sanitizedAudioEl.src = sanitizedAudioUrl;

    const stem = selectedAudioFile.name.replace(/\.[^.]+$/, "");
    sanitizedAudioDownloadName = `${stem}.sanitized.wav`;

    renderEntityList(
      audioEntitiesList,
      audioNoEntities,
      result.redacted_entities || extractEntityLabels(result.entities)
    );

    audioPreviewPanel.classList.remove("hidden");
    audioDownloadBtn.disabled = false;
    setBackendStatus("ok");
  } catch (error) {
    showError(error.message);
    setBackendStatus("error");
  } finally {
    audioSanitizeBtn.classList.remove("loading");
    audioSanitizeBtn.textContent = "Sanitize Audio";
    audioSanitizeBtn.disabled = !selectedAudioFile;
  }
}

function handleDownloadAudio() {
  if (!sanitizedAudioBlob) {
    showError("Nothing to download. Sanitize an audio file first.");
    return;
  }

  const url = URL.createObjectURL(sanitizedAudioBlob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = sanitizedAudioDownloadName;
  anchor.click();
  URL.revokeObjectURL(url);
  clearError();
}

// =============================================================================
// Video tab
// =============================================================================

function initVideoTab() {
  videoInput.addEventListener("change", handleVideoSelected);
  videoSanitizeBtn.addEventListener("click", handleVideoSanitize);
  videoDownloadBtn.addEventListener("click", handleDownloadVideo);
  if (videoApplySelectiveBtn) {
    videoApplySelectiveBtn.addEventListener("click", handleSelectiveSanitize);
  }
  if (videoBlurStrengthSlider && videoBlurStrengthVal) {
    videoBlurStrengthSlider.addEventListener("input", () => {
      videoBlurStrengthVal.textContent = videoBlurStrengthSlider.value;
    });
  }
}

function revokeVideoUrls() {
  if (sanitizedVideoUrl) {
    URL.revokeObjectURL(sanitizedVideoUrl);
    sanitizedVideoUrl = null;
  }
  sanitizedVideoBlob = null;
}

async function handleVideoSelected(event) {
  clearError();
  revokeVideoUrls();
  videoDownloadBtn.disabled = true;
  sanitizedVideoEl.removeAttribute("src");
  if (videoFaceSelectionPanel) videoFaceSelectionPanel.classList.add("hidden");
  if (videoPreviewPanel) videoPreviewPanel.classList.add("hidden");
  if (videoAnnotatedFrameContainer) videoAnnotatedFrameContainer.classList.add("hidden");
  if (videoFaceGrid) videoFaceGrid.innerHTML = "";
  detectedVideoFaces = [];
  selectedFaceIds.clear();

  const file = event.target.files?.[0];
  if (!file) {
    selectedVideoFile = null;
    videoSanitizeBtn.disabled = true;
    if (videoApplySelectiveBtn) videoApplySelectiveBtn.disabled = true;
    return;
  }

  selectedVideoFile = file;
  if (videoScanStatus) videoScanStatus.classList.remove("hidden");

  try {
    const formData = new FormData();
    formData.append("file", file, file.name);

    const response = await fetchWithTimeout(
      VIDEO_PREVIEW_FACES_ENDPOINT,
      { method: "POST", body: formData },
      VIDEO_TIMEOUT_MS
    );
    const data = await parseJsonResponse(response);

    if (!response.ok) {
      const detail = data.detail || data.message || response.statusText;
      throw new Error(`Face preview failed (${response.status}): ${detail}`);
    }

    detectedVideoFaces = data.faces || [];
    const count = data.face_count !== undefined ? data.face_count : detectedVideoFaces.length;

    if (videoFaceCountText) {
      videoFaceCountText.textContent = count > 0
        ? `Detected ${count} face(s) in video. Select faces to mitigate:`
        : "No faces detected in the first frame. You can still use 'Blur Everything'.";
    }

    const previewB64 = data.annotated_frame_base64 || data.preview_image_base64;
    if (previewB64 && videoAnnotatedFrame && videoAnnotatedFrameContainer) {
      videoAnnotatedFrame.src = `data:image/jpeg;base64,${previewB64}`;
      videoAnnotatedFrameContainer.classList.remove("hidden");
    }

    renderFaceGrid(detectedVideoFaces);

    const sensitiveEntities = data.sensitive_entities || [];
    const sensitiveBanner = document.getElementById("video-sensitive-info-banner");
    const sensitiveText = document.getElementById("video-sensitive-info-text");
    if (sensitiveBanner && sensitiveText) {
      if (sensitiveEntities.length > 0) {
        const uniqueLabels = Array.from(new Set(sensitiveEntities.map((e) => (e.label || "").toUpperCase()))).join(", ");
        sensitiveText.textContent = `Detected ${sensitiveEntities.length} sensitive item(s) (${uniqueLabels}). Redacting with synthetic Faker data.`;
        sensitiveBanner.classList.remove("hidden");
      } else {
        sensitiveBanner.classList.add("hidden");
      }
    }

    if (videoFaceSelectionPanel) {
      videoFaceSelectionPanel.classList.remove("hidden");
    }
    videoSanitizeBtn.disabled = false;
    if (videoApplySelectiveBtn) {
      videoApplySelectiveBtn.disabled = false;
    }
    setBackendStatus("ok");
  } catch (err) {
    showError(err.message);
    setBackendStatus("error");
    // Fallback: still show panel for blanket blur option
    if (videoFaceSelectionPanel) videoFaceSelectionPanel.classList.remove("hidden");
    videoSanitizeBtn.disabled = false;
  } finally {
    if (videoScanStatus) videoScanStatus.classList.add("hidden");
  }
}

function renderFaceGrid(faces) {
  if (!videoFaceGrid) return;
  videoFaceGrid.innerHTML = "";
  selectedFaceIds.clear();

  if (!faces || faces.length === 0) {
    const emptyP = document.createElement("p");
    emptyP.className = "muted";
    emptyP.textContent = "No face thumbnails available.";
    videoFaceGrid.appendChild(emptyP);
    return;
  }

  // Pre-select first face by default
  selectedFaceIds.add(faces[0].id);

  faces.forEach((face) => {
    const card = document.createElement("div");
    card.className = "face-card";
    if (selectedFaceIds.has(face.id)) {
      card.classList.add("selected");
    }
    card.dataset.faceId = String(face.id);

    const thumbWrap = document.createElement("div");
    thumbWrap.className = "face-thumb-wrapper";

    if (face.thumbnail_base64) {
      const img = document.createElement("img");
      img.src = `data:image/jpeg;base64,${face.thumbnail_base64}`;
      img.alt = `Face #${face.id}`;
      thumbWrap.appendChild(img);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "face-thumb-placeholder";
      placeholder.textContent = `Face #${face.id}`;
      thumbWrap.appendChild(placeholder);
    }

    const badge = document.createElement("span");
    badge.className = "face-id-badge";
    badge.textContent = `#${face.id}`;
    thumbWrap.appendChild(badge);

    const checkWrap = document.createElement("label");
    checkWrap.className = "face-checkbox-label";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "face-checkbox";
    checkbox.value = String(face.id);
    checkbox.checked = selectedFaceIds.has(face.id);

    const checkText = document.createElement("span");
    checkText.textContent = " Select";

    checkWrap.appendChild(checkbox);
    checkWrap.appendChild(checkText);

    card.appendChild(thumbWrap);
    card.appendChild(checkWrap);

    function toggleFace() {
      if (selectedFaceIds.has(face.id)) {
        selectedFaceIds.delete(face.id);
        card.classList.remove("selected");
        checkbox.checked = false;
      } else {
        selectedFaceIds.add(face.id);
        card.classList.add("selected");
        checkbox.checked = true;
      }
    }

    card.addEventListener("click", (e) => {
      if (e.target !== checkbox) {
        toggleFace();
      }
    });

    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        selectedFaceIds.add(face.id);
        card.classList.add("selected");
      } else {
        selectedFaceIds.delete(face.id);
        card.classList.remove("selected");
      }
    });

    videoFaceGrid.appendChild(card);
  });
}

async function handleSelectiveSanitize(options = {}) {
  clearError();

  if (!selectedVideoFile) {
    showError("Select a video file to sanitize.");
    return;
  }

  const isBlurAll = Boolean(options && options.blurAll);
  const modeRadio = document.querySelector('input[name="video-mode"]:checked');
  const mode = isBlurAll ? "blur_all" : (modeRadio ? modeRadio.value : "blur_selected");
  const blurStrength = videoBlurStrengthSlider ? videoBlurStrengthSlider.value : 51;
  const faceIdsArray = isBlurAll ? [] : Array.from(selectedFaceIds);

  if (mode === "blur_selected" && faceIdsArray.length === 0) {
    showError("Please select at least one face, or choose 'Blur all except selected', or click 'Blur Everything'.");
    return;
  }

  const redactToggle = document.getElementById("video-redact-sensitive-toggle");
  const redactSensitive = redactToggle ? redactToggle.checked : true;

  const activeBtn = isBlurAll ? videoSanitizeBtn : videoApplySelectiveBtn;
  activeBtn.classList.add("loading");
  activeBtn.textContent = "Sanitizing…";
  videoApplySelectiveBtn.disabled = true;
  videoSanitizeBtn.disabled = true;
  videoDownloadBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedVideoFile, selectedVideoFile.name);
    formData.append("selected_faces", JSON.stringify(faceIdsArray));
    formData.append("mode", mode);
    formData.append("blur_strength", String(blurStrength));
    formData.append("redact_sensitive_info", redactSensitive ? "true" : "false");

    const response = await fetchWithTimeout(
      VIDEO_SELECTIVE_ENDPOINT,
      { method: "POST", body: formData },
      VIDEO_TIMEOUT_MS
    );
    const result = await parseJsonResponse(response);

    if (!response.ok) {
      const detail = result.detail || result.message || response.statusText;
      throw new Error(`Sanitization failed (${response.status}): ${detail}`);
    }

    const blob = await resolveSanitizedMediaBlob(result, "video/mp4", VIDEO_TIMEOUT_MS);
    if (sanitizedVideoUrl) {
      URL.revokeObjectURL(sanitizedVideoUrl);
    }
    sanitizedVideoBlob = blob;
    sanitizedVideoUrl = URL.createObjectURL(blob);
    sanitizedVideoEl.src = sanitizedVideoUrl;

    const stem = selectedVideoFile.name.replace(/\.[^.]+$/, "");
    sanitizedVideoDownloadName = `${stem}.sanitized.mp4`;

    renderEntityList(
      videoEntitiesList,
      videoNoEntities,
      result.redacted_entities || extractEntityLabels(result.entities)
    );

    videoPreviewPanel.classList.remove("hidden");
    videoDownloadBtn.disabled = false;
    setBackendStatus("ok");
  } catch (error) {
    showError(error.message);
    setBackendStatus("error");
  } finally {
    videoApplySelectiveBtn.classList.remove("loading");
    videoApplySelectiveBtn.textContent = "Apply Selective Sanitization";
    videoSanitizeBtn.classList.remove("loading");
    videoSanitizeBtn.textContent = "Blur Everything (Blanket Blur)";
    videoApplySelectiveBtn.disabled = !selectedVideoFile;
    videoSanitizeBtn.disabled = !selectedVideoFile;
  }
}

async function handleVideoSanitize() {
  return handleSelectiveSanitize({ blurAll: true });
}

function handleDownloadVideo() {
  if (!sanitizedVideoBlob) {
    showError("Nothing to download. Sanitize a video file first.");
    return;
  }

  const url = URL.createObjectURL(sanitizedVideoBlob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = sanitizedVideoDownloadName;
  anchor.click();
  URL.revokeObjectURL(url);
  clearError();
}

// Shared media helper functions
async function sanitizeMedia(file, endpoint, timeoutMs) {
  const formData = new FormData();
  formData.append("file", file, file.name);

  const response = await fetchWithTimeout(
    endpoint,
    { method: "POST", body: formData },
    timeoutMs
  );

  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    const detail = payload.detail || payload.message || response.statusText;
    throw new Error(`Sanitization failed (${response.status}): ${detail}`);
  }

  return payload;
}

async function resolveSanitizedMediaBlob(payload, defaultMimeType, timeoutMs) {
  const downloadPath = payload.download_url || artifactDownloadPath(payload.mitigated_output);
  if (!downloadPath) {
    throw new Error("No download URL available for sanitized output.");
  }

  const url = downloadPath.startsWith("http")
    ? downloadPath
    : `${BACKEND_BASE}${downloadPath.startsWith("/") ? "" : "/"}${downloadPath}`;

  const response = await fetchWithTimeout(url, { method: "GET" }, timeoutMs);
  if (!response.ok) {
    throw new Error(`Failed to download sanitized file (${response.status}).`);
  }
  const blob = await response.blob();
  if (!blob || blob.size === 0) {
    throw new Error("Downloaded sanitized file is empty.");
  }
  return new Blob([blob], { type: defaultMimeType });
}

