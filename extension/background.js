"use strict";

/**
 * Privacy Sandbox service worker.
 * No persistent storage, no external network calls — only message routing.
 */

chrome.runtime.onInstalled.addListener(() => {
  console.info("[Privacy Sandbox] Extension installed. Backend: http://localhost:8000");
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "PING") {
    sendResponse({ ok: true, service: "privacy-sandbox" });
    return false;
  }

  if (message?.type === "MIC_PERMISSION_RESULT") {
    if (_sender.tab?.id) {
      chrome.tabs.remove(_sender.tab.id).catch(() => undefined);
    }
    chrome.runtime.sendMessage({
      type: "MIC_PERMISSION_COMPLETED",
      granted: Boolean(message.granted),
    }).catch(() => undefined);
    sendResponse({ ok: true });
    return true;
  }
  return false;
});
