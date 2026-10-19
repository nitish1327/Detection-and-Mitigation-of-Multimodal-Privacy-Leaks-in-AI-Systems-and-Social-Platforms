"use strict";

async function requestMicrophonePermission() {
  let stream;
  let granted = false;

  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    granted = true;
  } catch (_error) {
    granted = false;
  } finally {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  }

  try {
    await chrome.runtime.sendMessage({
      type: "MIC_PERMISSION_RESULT",
      granted,
    });
  } catch (_error) {
    // The permission page is temporary; the service worker may close it after routing.
  }
}

void requestMicrophonePermission();
