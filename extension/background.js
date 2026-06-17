"use strict";

/**
 * Privacy Sandbox service worker.
 * No persistent storage, no external network calls — only message routing.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.info("[Privacy Sandbox] Extension installed. Backend: http://localhost:8000");
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "PING") {
    sendResponse({ ok: true, service: "privacy-sandbox" });
    return false;
  }
  return false;
});
