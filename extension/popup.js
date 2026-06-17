"use strict";

const BACKEND_BASE = "http://localhost:8000";
const SCAN_ENDPOINT = `${BACKEND_BASE}/scan/text`;
const HEALTH_ENDPOINT = `${BACKEND_BASE}/health`;
const REQUEST_TIMEOUT_MS = 15000;

const inputText = document.getElementById("input-text");
const sanitizeBtn = document.getElementById("sanitize-btn");
const copyBtn = document.getElementById("copy-btn");
const sendBtn = document.getElementById("send-btn");
const previewPanel = document.getElementById("preview-panel");
const errorPanel = document.getElementById("error-panel");
const originalPreview = document.getElementById("original-preview");
const sanitizedPreview = document.getElementById("sanitized-preview");
const entitiesList = document.getElementById("entities-list");
const noEntities = document.getElementById("no-entities");
const backendStatus = document.getElementById("backend-status");

let sanitizedText = "";

document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  sanitizeBtn.addEventListener("click", handleSanitize);
  copyBtn.addEventListener("click", handleCopy);
  sendBtn.addEventListener("click", handleSendToLlm);
});

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
    renderPreview(text, sanitizedText, result.redacted_entities || []);
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
    SCAN_ENDPOINT,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    },
    REQUEST_TIMEOUT_MS
  );

  let payload;
  const raw = await response.text();

  if (!raw) {
    throw new Error("Backend returned an empty response.");
  }

  try {
    payload = JSON.parse(raw);
  } catch {
    throw new Error("Backend returned invalid JSON.");
  }

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

function renderPreview(original, sanitized, entities) {
  originalPreview.textContent = original;
  sanitizedPreview.textContent = sanitized;
  entitiesList.innerHTML = "";

  const unique = [...new Set(entities.map((e) => String(e).toUpperCase()))];
  if (unique.length === 0) {
    noEntities.classList.remove("hidden");
  } else {
    noEntities.classList.add("hidden");
    unique.forEach((label) => {
      const li = document.createElement("li");
      li.textContent = `• ${label}`;
      entitiesList.appendChild(li);
    });
  }

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
      showError(
        "Content script not loaded. Refresh the LLM page and try again."
      );
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
