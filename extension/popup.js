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
  "audio/webm",
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
const maximizeBtn = document.getElementById("maximize-btn");


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
const entitySelectionControls = document.getElementById("entity-selection-controls");
const selectAllEntitiesBtn = document.getElementById("select-all-entities-btn");
const clearAllEntitiesBtn = document.getElementById("clear-all-entities-btn");
const applyMitigationBtn = document.getElementById("apply-mitigation-btn");

// Image tab
const imageInput = document.getElementById("image-input");
const imageSanitizeBtn = document.getElementById("image-sanitize-btn");
const imagePreviewPanel = document.getElementById("image-preview-panel");
const originalImageEl = document.getElementById("original-image");
const sanitizedImageEl = document.getElementById("sanitized-image");
const sanitizedPlaceholder = document.getElementById("sanitized-placeholder");
const originalImageFrame = document.getElementById("original-image-frame");
const sanitizedImageFrame = document.getElementById("sanitized-image-frame");
const imageEntitiesList = document.getElementById("image-entities-list");
const imageNoEntities = document.getElementById("image-no-entities");
const downloadBtn = document.getElementById("download-btn");
const imageClearBtn = document.getElementById("image-clear-btn");
const imageExpandBtn = document.getElementById("image-expand-btn");

// Lightbox modal elements
const mediaLightboxModal = document.getElementById("media-lightbox-modal");
const lightboxBackdrop = document.getElementById("lightbox-backdrop");
const lightboxModeBadge = document.getElementById("lightbox-mode-badge");
const lightboxFilename = document.getElementById("lightbox-filename");
const lightboxToggleBtn = document.getElementById("lightbox-toggle-btn");
const lightboxOpenTabBtn = document.getElementById("lightbox-open-tab-btn");
const lightboxDownloadBtn = document.getElementById("lightbox-download-btn");
const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxVideo = document.getElementById("lightbox-video");


// Audio tab
const audioInput = document.getElementById("audio-input");
const audioSanitizeBtn = document.getElementById("audio-sanitize-btn");
const audioPreviewPanel = document.getElementById("audio-preview-panel");
const sanitizedAudioEl = document.getElementById("sanitized-audio");
const audioEntitiesList = document.getElementById("audio-entities-list");
const audioNoEntities = document.getElementById("audio-no-entities");
const audioDownloadBtn = document.getElementById("audio-download-btn");
const audioRecordBtn = document.getElementById("audio-record-btn");
const audioRecordIcon = document.getElementById("audio-record-icon");
const audioRecordLabel = document.getElementById("audio-record-label");
const audioRecordTimer = document.getElementById("audio-record-timer");
const audioRecordedContainer = document.getElementById("audio-recorded-container");
const audioRecordedPlayer = document.getElementById("audio-recorded-player");

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
const videoExpandBtn = document.getElementById("video-expand-btn");

const videoMitigateAudioToggle = document.getElementById("video-mitigate-audio-toggle");
const videoAudioHint = document.getElementById("video-audio-hint");
const videoAudioMitigatedBadge = document.getElementById("video-audio-mitigated-badge");

// Tabs
const tabButtons = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");

let sanitizedText = "";
let originalText = "";
let detectedEntities = [];
let selectedEntityTypes = new Set();
let selectedImageFile = null;
let originalImageUrl = null;
let sanitizedImageUrl = null;
let sanitizedImageBlob = null;
let sanitizedDownloadName = "sanitized.png";

let selectedAudioFile = null;
let sanitizedAudioUrl = null;
let sanitizedAudioBlob = null;
let sanitizedAudioDownloadName = "sanitized.wav";
let mediaRecorder = null;
let audioRecordStream = null;
let audioRecordChunks = [];
let audioRecordTimerInterval = null;
let audioRecordSeconds = 0;
let recordedAudioUrl = null;
let isRecordingAudio = false;

let selectedVideoFile = null;
let sanitizedVideoUrl = null;
let sanitizedVideoBlob = null;
let sanitizedVideoDownloadName = "sanitized.mp4";
let detectedVideoFaces = [];
let selectedFaceIds = new Set();

document.addEventListener("DOMContentLoaded", () => {
  initMaximizeUi();
  initLightbox();
  checkBackendHealth();
  initTabs();
  initTextTab();
  initImageTab();
  initAudioTab();
  initVideoTab();
  initPolicyTab();
  initAuditTab();
});

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type !== "MIC_PERMISSION_COMPLETED") return false;
  if (message.granted) {
    void startAudioRecording();
  } else {
    showError("Microphone permission denied. Allow Privacy Sandbox in Chrome's microphone settings (chrome://settings/content/microphone), then click Record Voice again.");
  }
  return false;
});

window.addEventListener("unload", () => {
  revokeImageUrls();
  revokeAudioUrls();
  revokeVideoUrls();
});

// =============================================================================
// Shared helpers & Maximize / Tab Mode
// =============================================================================

function getActiveTabName() {
  const activeBtn = document.querySelector(".tab.active");
  return activeBtn?.dataset?.tab || "text";
}

function updateMaximizeUi() {
  const urlParams = new URLSearchParams(window.location.search);
  const isExplicitTabMode = urlParams.get("mode") === "tab";
  const isWide = window.innerWidth >= 620;
  const isFullscreen = Boolean(document.fullscreenElement);

  if (isExplicitTabMode || isWide) {
    document.body.classList.add("tab-mode");
    document.body.classList.add("maximized-mode");
  } else {
    document.body.classList.remove("tab-mode");
    document.body.classList.remove("maximized-mode");
  }

  if (!maximizeBtn) return;

  if (isExplicitTabMode || isWide) {
    maximizeBtn.title = isFullscreen ? "Exit Fullscreen (Esc)" : "Toggle Fullscreen";
    maximizeBtn.setAttribute("aria-label", maximizeBtn.title);
    maximizeBtn.innerHTML = `
      <svg id="maximize-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="4 14 10 14 10 20"></polyline>
        <polyline points="20 10 14 10 14 4"></polyline>
        <line x1="14" y1="10" x2="21" y2="3"></line>
        <line x1="3" y1="21" x2="10" y2="14"></line>
      </svg>
    `;
  } else {
    maximizeBtn.title = "Maximize (Open in Full Tab)";
    maximizeBtn.setAttribute("aria-label", maximizeBtn.title);
    maximizeBtn.innerHTML = `
      <svg id="maximize-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="15 3 21 3 21 9"></polyline>
        <polyline points="9 21 3 21 3 15"></polyline>
        <line x1="21" y1="3" x2="14" y2="10"></line>
        <line x1="3" y1="21" x2="10" y2="14"></line>
      </svg>
    `;
  }
}

function handleMaximize() {
  const urlParams = new URLSearchParams(window.location.search);
  const isExplicitTabMode = urlParams.get("mode") === "tab";
  const isWide = window.innerWidth >= 620;

  if (!isExplicitTabMode && !isWide) {
    // Open full browser tab
    const activeTab = getActiveTabName();
    const targetUrl = chrome?.runtime?.getURL
      ? chrome.runtime.getURL(`popup.html?mode=tab&tab=${encodeURIComponent(activeTab)}`)
      : `popup.html?mode=tab&tab=${encodeURIComponent(activeTab)}`;
    if (chrome?.tabs?.create) {
      chrome.tabs.create({ url: targetUrl });
    } else {
      window.open(targetUrl, "_blank");
    }
  } else {
    // Already in a wide window or browser tab -> toggle fullscreen
    if (!document.fullscreenElement) {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(() => {});
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
      }
    }
  }
}

function initMaximizeUi() {
  const urlParams = new URLSearchParams(window.location.search);
  const requestedTab = urlParams.get("tab");
  if (requestedTab && ["text", "image", "audio", "video", "policy", "audit"].includes(requestedTab)) {
    switchTab(requestedTab);
  }

  updateMaximizeUi();

  if (maximizeBtn) {
    maximizeBtn.addEventListener("click", handleMaximize);
  }
  window.addEventListener("resize", updateMaximizeUi);
  document.addEventListener("fullscreenchange", updateMaximizeUi);
}


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
  if (tabName === "audit") {
    loadAuditHistory();
  }
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
      if (label.startsWith("AUDIO_")) {
        li.textContent = `• 🔊 Spoken ${label.replace("AUDIO_", "")} (Audio Tone-Masked)`;
      } else {
        li.textContent = `• ${label}`;
      }
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
  selectAllEntitiesBtn.addEventListener("click", () => setAllEntityTypesSelected(true));
  clearAllEntitiesBtn.addEventListener("click", () => setAllEntityTypesSelected(false));
  applyMitigationBtn.addEventListener("click", handleApplyTextMitigation);
  copyBtn.addEventListener("click", handleCopy);
  sendBtn.addEventListener("click", handleSendToLlm);
}

async function handleSanitize() {
  const text = inputText.value;
  clearError();

  if (!text.trim()) {
    showError("Enter some text to sanitize.");
    return;
  }

  originalText = text;
  detectedEntities = [];
  selectedEntityTypes = new Set();
  sanitizedText = text;
  sanitizeBtn.classList.add("loading");
  sanitizeBtn.textContent = "Sanitizing…";
  copyBtn.disabled = true;
  sendBtn.disabled = true;
  applyMitigationBtn.disabled = true;

  try {
    const result = await requestTextScan(text, { detect_only: true });
    detectedEntities = Array.isArray(result.entities) ? result.entities : [];
    selectedEntityTypes = new Set(detectedEntities.map((entity) => String(entity.label).toUpperCase()));
    renderTextPreview(originalText, originalText);
    renderSelectableTextEntities();
    setBackendStatus("ok");
  } catch (error) {
    showError(error.message);
    previewPanel.classList.add("hidden");
    setBackendStatus("error");
  } finally {
    sanitizeBtn.classList.remove("loading");
    sanitizeBtn.textContent = "Sanitize";
    applyMitigationBtn.disabled = detectedEntities.length === 0;
  }
}

async function handleApplyTextMitigation() {
  if (!originalText) {
    showError("Run a text scan before applying mitigation.");
    return;
  }

  clearError();
  applyMitigationBtn.disabled = true;
  applyMitigationBtn.classList.add("loading");
  applyMitigationBtn.textContent = "Applying…";
  try {
    const result = await requestTextScan(originalText, {
      selected_entity_types: [...selectedEntityTypes],
    });
    sanitizedText = result.sanitized_text ?? originalText;
    renderTextPreview(originalText, sanitizedText);
    renderSelectableTextEntities();
    setBackendStatus("ok");
  } catch (error) {
    showError(error.message);
    setBackendStatus("error");
  } finally {
    applyMitigationBtn.disabled = detectedEntities.length === 0;
    applyMitigationBtn.classList.remove("loading");
    applyMitigationBtn.textContent = "Apply Mitigation";
  }
}

function setAllEntityTypesSelected(selected) {
  const types = new Set(detectedEntities.map((entity) => String(entity.label).toUpperCase()));
  selectedEntityTypes = selected ? types : new Set();
  renderSelectableTextEntities();
}

function renderSelectableTextEntities() {
  entitiesList.innerHTML = "";
  const types = [...new Set(detectedEntities.map((entity) => String(entity.label).toUpperCase()))];
  if (types.length === 0) {
    noEntities.classList.remove("hidden");
    entitySelectionControls.classList.add("hidden");
    return;
  }

  noEntities.classList.add("hidden");
  entitySelectionControls.classList.remove("hidden");
  types.forEach((type) => {
    const item = document.createElement("li");
    item.className = "entity-choice-item";
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedEntityTypes.has(type);
    checkbox.dataset.entityType = type;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selectedEntityTypes.add(type);
      else selectedEntityTypes.delete(type);
    });
    const name = document.createElement("span");
    name.textContent = displayEntityType(type);
    label.className = "entity-option";
    label.append(checkbox, name);
    item.appendChild(label);
    entitiesList.appendChild(item);
  });
}

function displayEntityType(type) {
  if (type.includes("PHONE")) return "PHONE NUMBERS";
  return type.replaceAll("_", " ");
}

async function requestTextScan(text, options = {}) {
  const response = await fetchWithTimeout(
    TEXT_SCAN_ENDPOINT,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, ...options }),
    },
    TEXT_TIMEOUT_MS
  );

  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    const detail = payload.detail || payload.message || response.statusText;
    throw new Error(`Sanitization failed (${response.status}): ${detail}`);
  }

  if (typeof payload.sanitized_text !== "string") {
    if (payload.mitigated_output) {
      throw new Error(
        "Backend did not return sanitized_text. Restart the LeakWatch API server."
      );
    }
    throw new Error("Backend response missing sanitized_text.");
  }

  return payload;
}

function renderTextPreview(original, sanitized) {
  originalPreview.textContent = original;
  sanitizedPreview.textContent = sanitized;
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

// =============================================================================
// IndexedDB Session Persistence (Restores state when popup reopens)
// =============================================================================

const DB_NAME = "leakwatch_session_db";
const DB_VERSION = 1;
const STORE_NAME = "sessions";

function openSessionDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      resolve(null);
      return;
    }
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      console.warn("IndexedDB error:", request.error);
      resolve(null);
    };
  });
}

async function saveImageSession(data) {
  try {
    const db = await openSessionDb();
    if (!db) return;
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.put({ id: "current_image", timestamp: Date.now(), ...data });
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch (err) {
    console.warn("Failed to save image session:", err);
  }
}

async function loadImageSession() {
  try {
    const db = await openSessionDb();
    if (!db) return null;
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.get("current_image");
      req.onsuccess = () => {
        const result = req.result;
        // Keep valid for up to 4 hours
        if (result && Date.now() - result.timestamp < 4 * 3600 * 1000) {
          resolve(result);
        } else {
          resolve(null);
        }
      };
      req.onerror = () => resolve(null);
    });
  } catch (err) {
    console.warn("Failed to load image session:", err);
    return null;
  }
}

async function clearImageSession() {
  try {
    const db = await openSessionDb();
    if (!db) return;
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.delete("current_image");
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch (err) {
    console.warn("Failed to clear image session:", err);
  }
}

// =============================================================================
// Lightbox / Full View Modal (Stays in popup to prevent auto-close)
// =============================================================================

let currentLightboxData = null;

function openLightbox(options) {
  if (!mediaLightboxModal) return;
  currentLightboxData = options;

  if (lightboxFilename) {
    lightboxFilename.textContent = options.filename || "Preview";
    lightboxFilename.title = options.filename || "";
  }

  updateLightboxBadge(options.isSanitized);

  if (options.type === "image") {
    if (lightboxVideo) {
      lightboxVideo.classList.add("hidden");
      lightboxVideo.pause();
      lightboxVideo.removeAttribute("src");
    }

    if (lightboxImage) {
      lightboxImage.src = options.activeUrl;
      lightboxImage.classList.remove("hidden");
    }

    if (lightboxToggleBtn) {
      if (options.originalUrl && options.sanitizedUrl) {
        lightboxToggleBtn.classList.remove("hidden");
        lightboxToggleBtn.textContent = options.isSanitized ? "Show Original" : "Show Sanitized";
      } else {
        lightboxToggleBtn.classList.add("hidden");
      }
    }
  } else if (options.type === "video") {
    if (lightboxImage) {
      lightboxImage.classList.add("hidden");
      lightboxImage.removeAttribute("src");
    }
    if (lightboxToggleBtn) {
      lightboxToggleBtn.classList.add("hidden");
    }

    if (lightboxVideo) {
      lightboxVideo.src = options.activeUrl;
      lightboxVideo.classList.remove("hidden");
    }
  }

  mediaLightboxModal.classList.remove("hidden");
}

function updateLightboxBadge(isSanitized) {
  if (!lightboxModeBadge) return;
  if (isSanitized) {
    lightboxModeBadge.textContent = "SANITIZED";
    lightboxModeBadge.className = "lightbox-badge badge-sanitized";
  } else {
    lightboxModeBadge.textContent = "ORIGINAL";
    lightboxModeBadge.className = "lightbox-badge badge-original";
  }
}

function toggleLightboxCompare() {
  if (!currentLightboxData || currentLightboxData.type !== "image") return;
  const nextSanitized = !currentLightboxData.isSanitized;
  const nextUrl = nextSanitized ? currentLightboxData.sanitizedUrl : currentLightboxData.originalUrl;

  if (nextUrl && lightboxImage) {
    lightboxImage.src = nextUrl;
    currentLightboxData.activeUrl = nextUrl;
    currentLightboxData.isSanitized = nextSanitized;
    updateLightboxBadge(nextSanitized);
    if (lightboxToggleBtn) {
      lightboxToggleBtn.textContent = nextSanitized ? "Show Original" : "Show Sanitized";
    }
  }
}

function closeLightbox() {
  if (!mediaLightboxModal) return;
  mediaLightboxModal.classList.add("hidden");
  if (lightboxVideo) {
    lightboxVideo.pause();
    lightboxVideo.removeAttribute("src");
  }
  if (lightboxImage) {
    lightboxImage.removeAttribute("src");
  }
  currentLightboxData = null;
}

function handleLightboxDownload() {
  if (!currentLightboxData || !currentLightboxData.downloadBlob) {
    if (currentLightboxData?.type === "video") {
      handleDownloadVideo();
    } else {
      handleDownloadImage();
    }
    return;
  }
  const url = URL.createObjectURL(currentLightboxData.downloadBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = currentLightboxData.filename || "sanitized_media";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function handleLightboxOpenTab() {
  const activeTab = currentLightboxData?.type === "video" ? "video" : "image";
  const targetUrl = chrome?.runtime?.getURL
    ? chrome.runtime.getURL(`popup.html?mode=tab&tab=${activeTab}`)
    : `popup.html?mode=tab&tab=${activeTab}`;
  if (chrome?.tabs?.create) {
    chrome.tabs.create({ url: targetUrl });
  } else {
    window.open(targetUrl, "_blank");
  }
}

function initLightbox() {
  if (lightboxCloseBtn) lightboxCloseBtn.addEventListener("click", closeLightbox);
  if (lightboxBackdrop) lightboxBackdrop.addEventListener("click", closeLightbox);
  if (lightboxToggleBtn) lightboxToggleBtn.addEventListener("click", toggleLightboxCompare);
  if (lightboxDownloadBtn) lightboxDownloadBtn.addEventListener("click", handleLightboxDownload);
  if (lightboxOpenTabBtn) lightboxOpenTabBtn.addEventListener("click", handleLightboxOpenTab);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mediaLightboxModal && !mediaLightboxModal.classList.contains("hidden")) {
      closeLightbox();
    }
  });
}

// =============================================================================
// Image tab
// =============================================================================

function initImageTab() {
  imageInput.addEventListener("change", handleImageSelected);
  imageSanitizeBtn.addEventListener("click", handleImageSanitize);
  downloadBtn.addEventListener("click", handleDownloadImage);
  if (imageClearBtn) {
    imageClearBtn.addEventListener("click", handleImageClear);
  }
  if (imageExpandBtn) {
    imageExpandBtn.addEventListener("click", handleImageExpand);
  }
  if (originalImageFrame) {
    originalImageFrame.addEventListener("click", () => {
      if (originalImageUrl) {
        openLightbox({
          type: "image",
          activeUrl: originalImageUrl,
          originalUrl: originalImageUrl,
          sanitizedUrl: sanitizedImageUrl,
          isSanitized: false,
          filename: selectedImageFile?.name || "original_image",
          downloadBlob: selectedImageFile,
        });
      }
    });
  }
  if (sanitizedImageFrame) {
    sanitizedImageFrame.addEventListener("click", () => {
      if (sanitizedImageUrl) {
        openLightbox({
          type: "image",
          activeUrl: sanitizedImageUrl,
          originalUrl: originalImageUrl,
          sanitizedUrl: sanitizedImageUrl,
          isSanitized: true,
          filename: sanitizedDownloadName,
          downloadBlob: sanitizedImageBlob,
        });
      }
    });
  }

  // Restore saved session if user reopened popup
  void (async () => {
    const saved = await loadImageSession();
    if (saved && saved.sanitizedBlob) {
      sanitizedImageBlob = saved.sanitizedBlob;
      sanitizedDownloadName = saved.downloadName || "sanitized.png";
      sanitizedImageUrl = URL.createObjectURL(sanitizedImageBlob);
      sanitizedImageEl.src = sanitizedImageUrl;
      sanitizedImageEl.classList.remove("hidden");
      if (sanitizedPlaceholder) {
        sanitizedPlaceholder.classList.add("hidden");
      }
      downloadBtn.disabled = false;

      if (saved.originalBlob) {
        originalImageUrl = URL.createObjectURL(saved.originalBlob);
        originalImageEl.src = originalImageUrl;
        try {
          selectedImageFile = new File([saved.originalBlob], saved.filename || "image.png", {
            type: saved.fileType || saved.originalBlob.type || "image/png",
          });
          imageSanitizeBtn.disabled = false;
        } catch {
          selectedImageFile = saved.originalBlob;
        }
      }

      if (saved.entities && saved.entities.length) {
        renderEntityList(imageEntitiesList, imageNoEntities, saved.entities);
      }

      imagePreviewPanel.classList.remove("hidden");
    }
  })();
}

function handleImageExpand() {
  const target = sanitizedImageUrl || originalImageUrl;
  if (!target) return;
  openLightbox({
    type: "image",
    activeUrl: target,
    originalUrl: originalImageUrl,
    sanitizedUrl: sanitizedImageUrl,
    isSanitized: Boolean(sanitizedImageUrl),
    filename: sanitizedDownloadName || selectedImageFile?.name || "image.png",
    downloadBlob: sanitizedImageBlob || selectedImageFile,
  });
}

function handleImageClear() {
  revokeImageUrls();
  void clearImageSession();
  selectedImageFile = null;
  imageInput.value = "";
  imageSanitizeBtn.disabled = true;
  downloadBtn.disabled = true;
  imagePreviewPanel.classList.add("hidden");
  if (sanitizedPlaceholder) {
    sanitizedPlaceholder.classList.remove("hidden");
  }
  sanitizedImageEl.classList.add("hidden");
  originalImageEl.removeAttribute("src");
  sanitizedImageEl.removeAttribute("src");
  clearError();
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
  void clearImageSession();
  downloadBtn.disabled = true;
  sanitizedImageEl.removeAttribute("src");
  sanitizedImageEl.classList.add("hidden");
  if (sanitizedPlaceholder) {
    sanitizedPlaceholder.classList.remove("hidden");
  }

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
    sanitizedImageEl.classList.remove("hidden");
    if (sanitizedPlaceholder) {
      sanitizedPlaceholder.classList.add("hidden");
    }

    const stem = selectedImageFile.name.replace(/\.[^.]+$/, "");
    const ext = guessExtensionFromBlob(blob) || ".png";
    sanitizedDownloadName = `${stem}.sanitized${ext}`;

    const labels = result.redacted_entities || extractEntityLabels(result.entities);
    renderEntityList(
      imageEntitiesList,
      imageNoEntities,
      labels
    );

    // Save state to IndexedDB so reopening extension preserves image + download
    void saveImageSession({
      filename: selectedImageFile.name,
      fileType: selectedImageFile.type,
      downloadName: sanitizedDownloadName,
      originalBlob: selectedImageFile,
      sanitizedBlob: blob,
      entities: labels,
    });

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
  if (audioRecordBtn) {
    audioRecordBtn.addEventListener("click", handleToggleAudioRecording);
  }
  window.addEventListener("pagehide", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    releaseAudioRecordStream();
    resetAudioRecordButton();
  }, { once: true });
}

function revokeAudioUrls() {
  if (sanitizedAudioUrl) {
    URL.revokeObjectURL(sanitizedAudioUrl);
    sanitizedAudioUrl = null;
  }
  if (recordedAudioUrl) {
    URL.revokeObjectURL(recordedAudioUrl);
    recordedAudioUrl = null;
  }
  sanitizedAudioBlob = null;
}

function handleAudioSelected(event) {
  clearError();
  revokeAudioUrls();
  if (isRecordingAudio) {
    stopAudioRecording();
  }
  resetAudioRecordButton();
  if (audioRecordedContainer) {
    audioRecordedContainer.classList.add("hidden");
  }
  if (audioRecordedPlayer) {
    audioRecordedPlayer.removeAttribute("src");
  }
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

async function handleToggleAudioRecording() {
  clearError();
  if (isRecordingAudio) {
    stopAudioRecording();
  } else {
    openMicrophonePermissionPage();
  }
}

function openMicrophonePermissionPage() {
  chrome.tabs.create({ url: chrome.runtime.getURL("permission.html") }, (tab) => {
    if (chrome.runtime.lastError || !tab) {
      showError("Could not open the microphone permission page. Please try again.");
    }
  });
}

async function startAudioRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Microphone recording is not supported in this browser.");
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    showError("MediaRecorder is not supported in this browser. You can still upload an audio file.");
    return;
  }

  try {
    audioRecordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    releaseAudioRecordStream();
    showError(getAudioRecordingErrorMessage(err));
    return;
  }

  // Clear file input if a file was selected previously
  if (audioInput) {
    audioInput.value = "";
  }
  if (recordedAudioUrl) {
    URL.revokeObjectURL(recordedAudioUrl);
    recordedAudioUrl = null;
  }
  if (audioRecordedPlayer) {
    audioRecordedPlayer.removeAttribute("src");
  }
  if (audioRecordedContainer) {
    audioRecordedContainer.classList.add("hidden");
  }
  audioPreviewPanel.classList.add("hidden");
  selectedAudioFile = null;
  audioSanitizeBtn.disabled = true;

  audioRecordChunks = [];
  try {
    const preferredMimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    mediaRecorder = new MediaRecorder(
      audioRecordStream,
      preferredMimeType ? { mimeType: preferredMimeType } : undefined
    );
  } catch (err) {
    releaseAudioRecordStream();
    showError("Could not start audio recording. You can still upload an audio file.");
    return;
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      audioRecordChunks.push(e.data);
    }
  };

  mediaRecorder.onstop = () => {
    releaseAudioRecordStream();

    if (audioRecordTimerInterval) {
      clearInterval(audioRecordTimerInterval);
      audioRecordTimerInterval = null;
    }

    const recordedMime = mediaRecorder.mimeType || audioRecordChunks[0]?.type || "audio/webm";
    const extension = recordedMime.includes("ogg") ? "ogg" : "webm";
    const audioBlob = new Blob(audioRecordChunks, { type: recordedMime });

    if (audioBlob.size === 0) {
      showError("Recording was empty. Try speaking into the microphone.");
      resetAudioRecordButton();
      return;
    }

    const filename = `voice_recording_${Date.now()}.${extension}`;
    selectedAudioFile = new File([audioBlob], filename, { type: recordedMime });

    // Show playback so user can listen to the recorded audio
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl);
    }
    recordedAudioUrl = URL.createObjectURL(audioBlob);
    if (audioRecordedPlayer) {
      audioRecordedPlayer.src = recordedAudioUrl;
    }
    if (audioRecordedContainer) {
      audioRecordedContainer.classList.remove("hidden");
    }

    // Enable sanitization button
    audioSanitizeBtn.disabled = false;
    resetAudioRecordButton();
  };

  try {
    mediaRecorder.start(250);
  } catch (err) {
    mediaRecorder = null;
    releaseAudioRecordStream();
    showError("Could not start audio recording. You can still upload an audio file.");
    return;
  }
  isRecordingAudio = true;
  if (audioRecordBtn) {
    audioRecordBtn.classList.add("recording");
  }
  if (audioRecordLabel) {
    audioRecordLabel.textContent = "Stop Recording";
  }
  if (audioRecordIcon) {
    audioRecordIcon.textContent = "⏹️";
  }
  if (audioRecordTimer) {
    audioRecordTimer.classList.remove("hidden");
    audioRecordTimer.textContent = "00:00";
  }
  audioRecordSeconds = 0;

  audioRecordTimerInterval = setInterval(() => {
    audioRecordSeconds += 1;
    const mins = String(Math.floor(audioRecordSeconds / 60)).padStart(2, "0");
    const secs = String(audioRecordSeconds % 60).padStart(2, "0");
    if (audioRecordTimer) {
      audioRecordTimer.textContent = `${mins}:${secs}`;
    }
  }, 1000);
}

function stopAudioRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  } else {
    releaseAudioRecordStream();
    resetAudioRecordButton();
  }
  isRecordingAudio = false;
}

function releaseAudioRecordStream() {
  if (audioRecordStream) {
    audioRecordStream.getTracks().forEach((track) => track.stop());
    audioRecordStream = null;
  }
}

function getAudioRecordingErrorMessage(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Microphone permission denied. Allow Privacy Sandbox in Chrome's microphone settings (chrome://settings/content/microphone), then click Record Voice again.";
    case "NotFoundError":
      return "No microphone was found. Connect a microphone or upload an audio file.";
    case "NotReadableError":
      return "The microphone is unavailable or being used by another app. You can still upload an audio file.";
    case "SecurityError":
      return "Microphone access is blocked by browser security settings. Allow microphone access or upload an audio file.";
    default:
      return `Could not access the microphone${error?.name ? ` (${error.name})` : ""}. You can still upload an audio file.`;
  }
}

function resetAudioRecordButton() {
  isRecordingAudio = false;
  if (audioRecordBtn) {
    audioRecordBtn.classList.remove("recording");
  }
  if (audioRecordLabel) {
    audioRecordLabel.textContent = "Record Voice";
  }
  if (audioRecordIcon) {
    audioRecordIcon.textContent = "🎙️";
  }
  if (audioRecordTimer) {
    audioRecordTimer.classList.add("hidden");
  }
  if (audioRecordTimerInterval) {
    clearInterval(audioRecordTimerInterval);
    audioRecordTimerInterval = null;
  }
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
  if (sanitizedVideoEl) {
    sanitizedVideoEl.setAttribute("controlsList", "nofullscreen");
  }
  videoInput.addEventListener("change", handleVideoSelected);
  videoSanitizeBtn.addEventListener("click", handleVideoSanitize);
  videoDownloadBtn.addEventListener("click", handleDownloadVideo);
  if (videoExpandBtn) {
    videoExpandBtn.addEventListener("click", handleVideoExpand);
  }
  if (videoApplySelectiveBtn) {
    videoApplySelectiveBtn.addEventListener("click", handleSelectiveSanitize);
  }
  if (videoBlurStrengthSlider && videoBlurStrengthVal) {
    videoBlurStrengthSlider.addEventListener("input", () => {
      videoBlurStrengthVal.textContent = videoBlurStrengthSlider.value;
    });
  }
}

function handleVideoExpand() {
  if (sanitizedVideoUrl) {
    openLightbox({
      type: "video",
      activeUrl: sanitizedVideoUrl,
      sanitizedUrl: sanitizedVideoUrl,
      isSanitized: true,
      filename: sanitizedVideoDownloadName || "sanitized.mp4",
      downloadBlob: sanitizedVideoBlob,
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
  if (videoAudioMitigatedBadge) videoAudioMitigatedBadge.classList.add("hidden");
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

    if (data.has_audio === false) {
      if (videoMitigateAudioToggle) {
        videoMitigateAudioToggle.checked = false;
        videoMitigateAudioToggle.disabled = true;
      }
      if (videoAudioHint) {
        videoAudioHint.textContent = "No audio stream detected in this video.";
      }
    } else {
      if (videoMitigateAudioToggle) {
        videoMitigateAudioToggle.disabled = false;
        videoMitigateAudioToggle.checked = true;
      }
      if (videoAudioHint) {
        videoAudioHint.textContent = "Transcribe speech & beep/mute spoken PII (names, phone numbers, IDs)";
      }
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
  const audioToggle = document.getElementById("video-mitigate-audio-toggle");
  const mitigateAudio = audioToggle && !audioToggle.disabled ? audioToggle.checked : false;

  const activeBtn = isBlurAll ? videoSanitizeBtn : videoApplySelectiveBtn;
  activeBtn.classList.add("loading");
  activeBtn.textContent = mitigateAudio ? "Sanitizing (Video + Audio)…" : "Sanitizing…";
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
    formData.append("mitigate_audio", mitigateAudio ? "true" : "false");

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

    if (videoAudioMitigatedBadge) {
      if (result.audio_mitigated) {
        const spokenCount = result.audio_entities_detected || 0;
        videoAudioMitigatedBadge.innerHTML = `<span style="font-size: 11px; font-weight: 600; color: #22c55e;">
          🔊 Audio Privacy Protected — ${spokenCount > 0 ? spokenCount + " spoken PII item(s) tone-masked with audio beeps" : "Speech transcribed & verified clean"}
        </span>`;
        videoAudioMitigatedBadge.classList.remove("hidden");
      } else {
        videoAudioMitigatedBadge.classList.add("hidden");
      }
    }

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

// ==================== POLICY TAB ====================
const policyAllowlist = document.getElementById("policy-allowlist");
const policyRegex = document.getElementById("policy-regex");
const policyBlurStrength = document.getElementById("policy-blur-strength");
const blurStrengthVal = document.getElementById("blur-strength-val");
const policyAudioMask = document.getElementById("policy-audio-mask");
const policySaveBtn = document.getElementById("policy-save-btn");
const policyResetBtn = document.getElementById("policy-reset-btn");
const policyStatusMsg = document.getElementById("policy-status-msg");

function getStoredPolicy(callback) {
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["leakwatch_policy"], (res) => {
      callback(res.leakwatch_policy || {});
    });
  } else {
    try {
      const stored = localStorage.getItem("leakwatch_policy");
      callback(stored ? JSON.parse(stored) : {});
    } catch {
      callback({});
    }
  }
}

function saveStoredPolicy(policy, callback) {
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.set({ leakwatch_policy: policy }, callback);
  } else {
    try {
      localStorage.setItem("leakwatch_policy", JSON.stringify(policy));
      if (callback) callback();
    } catch {
      if (callback) callback();
    }
  }
}

function initPolicyTab() {
  if (policyBlurStrength && blurStrengthVal) {
    policyBlurStrength.addEventListener("input", () => {
      blurStrengthVal.textContent = policyBlurStrength.value;
    });
  }

  getStoredPolicy((policy) => {
    if (policyAllowlist && policy.allowlist) {
      policyAllowlist.value = policy.allowlist.join("\n");
    }
    if (policyRegex && policy.custom_regex) {
      policyRegex.value = policy.custom_regex.join("\n");
    }
    if (policyBlurStrength && policy.blur_strength) {
      policyBlurStrength.value = policy.blur_strength;
      if (blurStrengthVal) blurStrengthVal.textContent = policy.blur_strength;
    }
    if (policyAudioMask && policy.audio_mask_style) {
      policyAudioMask.value = policy.audio_mask_style;
    }
  });

  if (policySaveBtn) {
    policySaveBtn.addEventListener("click", () => {
      const allowlist = policyAllowlist ? policyAllowlist.value.split("\n").map(s => s.trim()).filter(Boolean) : [];
      const regexRules = policyRegex ? policyRegex.value.split("\n").map(s => s.trim()).filter(Boolean) : [];
      const blurStrength = policyBlurStrength ? parseInt(policyBlurStrength.value, 10) : 51;
      const audioMask = policyAudioMask ? policyAudioMask.value : "beep";

      const policy = {
        allowlist,
        custom_regex: regexRules,
        blur_strength: blurStrength,
        audio_mask_style: audioMask,
      };

      saveStoredPolicy(policy, () => {
        if (policyStatusMsg) {
          policyStatusMsg.textContent = "Settings saved successfully!";
          policyStatusMsg.classList.remove("hidden");
          setTimeout(() => policyStatusMsg.classList.add("hidden"), 3000);
        }
      });
    });
  }

  if (policyResetBtn) {
    policyResetBtn.addEventListener("click", () => {
      if (policyAllowlist) policyAllowlist.value = "";
      if (policyRegex) policyRegex.value = "";
      if (policyBlurStrength) {
        policyBlurStrength.value = 51;
        if (blurStrengthVal) blurStrengthVal.textContent = "51";
      }
      if (policyAudioMask) policyAudioMask.value = "beep";
      saveStoredPolicy({}, () => {
        if (policyStatusMsg) {
          policyStatusMsg.textContent = "Reset to default settings.";
          policyStatusMsg.classList.remove("hidden");
          setTimeout(() => {
            policyStatusMsg.textContent = "Settings saved successfully!";
            policyStatusMsg.classList.add("hidden");
          }, 2500);
        }
      });
    });
  }
}

// ==================== AUDIT TAB ====================
const auditRefreshBtn = document.getElementById("audit-refresh-btn");
const auditList = document.getElementById("audit-list");

function initAuditTab() {
  if (auditRefreshBtn) {
    auditRefreshBtn.addEventListener("click", loadAuditHistory);
  }
}

async function loadAuditHistory() {
  if (!auditList) return;
  auditList.innerHTML = `<p class="muted">Loading audit records...</p>`;
  try {
    const res = await fetch(`${BACKEND_BASE}/audit/history?limit=30`);
    if (!res.ok) {
      auditList.innerHTML = `<p class="error-text">Failed to fetch audit records (${res.status}). Ensure backend is running.</p>`;
      return;
    }
    const logs = await res.json();
    if (!logs || logs.length === 0) {
      auditList.innerHTML = `<p class="muted">No audit logs found in artifacts.</p>`;
      return;
    }

    auditList.innerHTML = "";
    logs.forEach((log) => {
      const item = document.createElement("div");
      item.className = "audit-item";
      item.style.cssText = "padding: 10px; margin-bottom: 8px; background: rgba(255,255,255,0.04); border-radius: 6px; border: 1px solid rgba(255,255,255,0.08); font-size: 12px;";

      const timeStr = log.timestamp ? new Date(log.timestamp).toLocaleString() : "Unknown date";
      const modalityUpper = (log.modality || "TEXT").toUpperCase();

      item.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
          <strong style="color: #60a5fa;">[${modalityUpper}] ${log.source || "Unknown Source"}</strong>
          <span style="color: #94a3b8; font-size: 11px;">${timeStr}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; color: #cbd5e1;">
          <span>Entities detected: <strong>${log.entity_count}</strong> (${log.detection_mode || "regex"})</span>
          <a href="${BACKEND_BASE}/audit/history/${log.filename}" target="_blank" download="${log.filename}" class="btn-lightbox" style="padding: 2px 8px; font-size: 11px; text-decoration: none; display: inline-block;">JSON</a>
        </div>
      `;
      auditList.appendChild(item);
    });
  } catch (err) {
    auditList.innerHTML = `<p class="error-text">Could not connect to audit history: ${err.message}</p>`;
  }
}

