"use strict";

/**
 * Privacy Sandbox content script — site adapters for LLM chat injection.
 * Loaded only on supported hosts (see manifest.json).
 */

const SITE_ADAPTERS = {
  chatgpt: {
    hosts: ["chat.openai.com", "chatgpt.com"],
    selectors: [
      "#prompt-textarea",
      'textarea[data-testid="composer-text-input"]',
      'div[contenteditable="true"]#prompt-textarea',
      'div[contenteditable="true"][data-testid]',
      'div.ProseMirror[contenteditable="true"]',
      'textarea[placeholder*="Message"]',
      'textarea[placeholder*="Ask"]',
      'div[contenteditable="true"].ProseMirror',
    ],
  },
  gemini: {
    hosts: ["gemini.google.com"],
    selectors: [
      "rich-textarea .ql-editor",
      "rich-textarea div[contenteditable='true']",
      ".text-input-field textarea",
      ".text-input-field [contenteditable='true']",
      "div[contenteditable='true'][aria-label*='Enter']",
      "div[contenteditable='true'][data-placeholder]",
      "textarea",
      "div[contenteditable='true']",
    ],
  },
  claude: {
    hosts: ["claude.ai"],
    selectors: [
      'div[contenteditable="true"].ProseMirror',
      'div.ProseMirror[contenteditable="true"]',
      'div[contenteditable="true"][data-testid]',
      'textarea[placeholder*="Reply"]',
      'textarea[placeholder*="Message"]',
      'div[contenteditable="true"]',
      "textarea",
    ],
  },
};

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "INJECT_SANITIZED_TEXT") {
    const result = injectSanitizedText(message.text);
    sendResponse(result);
    return true;
  }
  return false;
});

function injectSanitizedText(text) {
  if (!text || typeof text !== "string") {
    return { success: false, error: "No text provided for injection." };
  }

  const adapter = resolveAdapter();
  if (!adapter) {
    return {
      success: false,
      error: "Unsupported site. Open ChatGPT, Gemini, or Claude.",
    };
  }

  const element = findVisibleInput(adapter.selectors);
  if (!element) {
    return {
      success: false,
      error: `${adapter.name} chat input not found. Click the message box and try again.`,
    };
  }

  try {
    setElementText(element, text);
    element.focus();
    return { success: true, site: adapter.name };
  } catch (error) {
    return {
      success: false,
      error: `Injection failed: ${error.message || error}`,
    };
  }
}

function resolveAdapter() {
  const hostname = window.location.hostname;
  for (const [name, config] of Object.entries(SITE_ADAPTERS)) {
    if (config.hosts.some((host) => hostname === host || hostname.endsWith(`.${host}`))) {
      return { name, selectors: config.selectors };
    }
  }
  return null;
}

function findVisibleInput(selectors) {
  for (const selector of selectors) {
    const nodes = document.querySelectorAll(selector);
    for (const node of nodes) {
      if (isVisible(node) && !isDisabled(node)) {
        return node;
      }
    }
  }
  return null;
}

function isVisible(element) {
  if (!element) return false;
  const style = window.getComputedStyle(element);
  if (style.display === "none" || style.visibility === "hidden") return false;
  const rect = element.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function isDisabled(element) {
  return element.disabled === true || element.getAttribute("aria-disabled") === "true";
}

function setElementText(element, text) {
  element.focus();

  if (element.tagName === "TEXTAREA" || element.tagName === "INPUT") {
    setNativeValue(element, text);
    dispatchInputEvents(element);
    return;
  }

  if (element.isContentEditable || element.getAttribute("contenteditable") === "true") {
    setContentEditableText(element, text);
    dispatchInputEvents(element);
    return;
  }

  throw new Error("Unsupported input element type.");
}

function setNativeValue(element, text) {
  const prototype =
    element.tagName === "TEXTAREA"
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
  const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
  if (descriptor?.set) {
    descriptor.set.call(element, text);
  } else {
    element.value = text;
  }
}

function setContentEditableText(element, text) {
  element.focus();

  const selection = window.getSelection();
  if (selection) {
    const range = document.createRange();
    range.selectNodeContents(element);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  const inserted = document.execCommand("insertText", false, text);
  if (!inserted) {
    element.textContent = text;
    element.innerHTML = "";
    const paragraph = document.createElement("p");
    paragraph.textContent = text;
    element.appendChild(paragraph);
  }
}

function dispatchInputEvents(element) {
  const events = ["input", "change", "keyup", "keydown"];
  for (const type of events) {
    element.dispatchEvent(new Event(type, { bubbles: true, cancelable: true }));
  }
  element.dispatchEvent(
    new InputEvent("input", {
      bubbles: true,
      cancelable: true,
      inputType: "insertText",
      data: element.value || element.textContent || "",
    })
  );
}
