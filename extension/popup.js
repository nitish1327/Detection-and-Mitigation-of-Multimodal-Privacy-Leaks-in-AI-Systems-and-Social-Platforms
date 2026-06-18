"use strict";

const BACKEND_BASE = "http://localhost:8000";
const TEXT_SCAN_ENDPOINT = `${BACKEND_BASE}/scan/text`;
const IMAGE_UPLOAD_ENDPOINT = `${BACKEND_BASE}/scan/image/upload`;
const HEALTH_ENDPOINT = `${BACKEND_BASE}/health`;
const TEXT_TIMEOUT_MS = 15000;
const IMAGE_TIMEOUT_MS = 120000;

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/bmp",
  "image/webp",
  "image/pjpeg",
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

// Tabs
const tabButtons = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");

let sanitizedText = "";
let selectedImageFile = null;
let originalImageUrl = null;
let sanitizedImageUrl = null;
let sanitizedImageBlob = null;
let sanitizedDownloadName = "sanitized.png";

document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  initTabs();
  initTextTab();
  initImageTab();
});

window.addEventListener("unload", () => {
  revokeImageUrls();
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
