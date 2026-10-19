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
    const expectedPermissionUrl = chrome.runtime.getURL("permission.html");
    if (_sender.url !== expectedPermissionUrl || !_sender.tab?.id) {
      sendResponse({ ok: false });
      return false;
    }

    chrome.runtime.sendMessage({
      type: "MIC_PERMISSION_COMPLETED",
      granted: Boolean(message.granted),
    }).catch(() => undefined).then(() => chrome.tabs.remove(_sender.tab.id))
      .then(() => sendResponse({ ok: true }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
  return false;
});
