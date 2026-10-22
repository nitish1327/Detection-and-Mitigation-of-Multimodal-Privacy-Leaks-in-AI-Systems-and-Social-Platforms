"use strict";

async function requestMicrophonePermission() {
  const statusEl = document.getElementById("status-text");
  const closeBtn = document.getElementById("close-btn");
  let stream = null;
  let granted = false;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    granted = true;
    if (statusEl) {
      statusEl.innerHTML = "Microphone access <strong style='color:#3fb950;'>granted</strong>! Closing this tab…";
    }
  } catch (_error) {
    granted = false;
    if (statusEl) {
      statusEl.innerHTML = "Microphone access <strong style='color:#f85149;'>denied</strong>. Please allow microphone access in your browser settings (click the lock or camera/mic icon in the address bar) and try again.";
    }
    if (closeBtn) {
      closeBtn.style.display = "inline-block";
    }
  } finally {
    if (stream) {
      stream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {}
      });
    }
  }

  try {
    await chrome.runtime.sendMessage({
      type: "MIC_PERMISSION_RESULT",
      granted,
    });
  } catch (_error) {
    // Service worker may already be asleep or popup closed
  }

  if (granted) {
    setTimeout(() => {
      try {
        window.close();
      } catch (_) {}
    }, 600);
  }
}

void requestMicrophonePermission();
