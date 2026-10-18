/**
 * Sidebar script — chat UI restyled to the approved mockups (#2570).
 * Clicking the Penny brand or the external-link button opens the full page.
 * Communicates with the background script via browser.runtime messaging.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  MAX_STORED_MESSAGES,
  type MessageSender,
  MessageSender as MS,
  type RuntimeMessage,
  RuntimeMessageType,
  STORAGE_KEY_CHAT_HISTORY,
  STORAGE_KEY_DEVICE_LABEL,
  type StoredMessage,
  TEXTAREA_LINE_HEIGHT,
  TEXTAREA_MAX_ROWS,
} from "../protocol.js";
import { hostnameOf, type TrailRow, parseTrail } from "./trail.js";

// DOM refs (resolved after init)
let messagesEl: HTMLElement;
let inputEl: HTMLTextAreaElement;
let sendBtn: HTMLButtonElement;
let statusEl: HTMLElement;

const TYPING_ID = "typing";
const PERM_WAIT_ID = "perm-wait";
const EMPTY_STATE_ID = "empty-state";

// Page context for decorating the next response
let pendingPageRef: { title: string; url: string; image: string } | null = null;
let currentPageImage = "";

// --- Registration ---

async function init(): Promise<void> {
  statusEl = document.getElementById("status")!;
  const stored = await browser.storage.local.get(STORAGE_KEY_DEVICE_LABEL);
  if (stored[STORAGE_KEY_DEVICE_LABEL]) {
    showChat();
  } else {
    showRegister();
  }
}

function showView(view: "register" | "chat"): void {
  document.getElementById("register")!.classList.toggle("hidden", view !== "register");
  document.getElementById("chat")!.classList.toggle("hidden", view !== "chat");
}

function showRegister(): void {
  showView("register");

  const labelInput = document.getElementById("device-label") as HTMLInputElement;
  const registerBtn = document.getElementById("register-btn")!;

  registerBtn.addEventListener("click", () => saveLabel(labelInput));
  labelInput.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") saveLabel(labelInput);
  });

  labelInput.focus();
}

async function saveLabel(labelInput: HTMLInputElement): Promise<void> {
  const label = labelInput.value.trim();
  if (!label) return;

  await browser.storage.local.set({ [STORAGE_KEY_DEVICE_LABEL]: label });
  showChat();
}

async function showChat(): Promise<void> {
  showView("chat");

  messagesEl = document.getElementById("messages")!;
  inputEl = document.getElementById("input") as HTMLTextAreaElement;
  sendBtn = document.getElementById("send") as HTMLButtonElement;

  sendBtn.addEventListener("click", send);
  inputEl.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autoResize);

  // Both the brand and the external-link button open the full page
  document.getElementById("nav-logo")!.addEventListener("click", openFullPage);
  document.getElementById("open-page")!.addEventListener("click", openFullPage);

  await rehydrateHistory();
  listenToBackground();
}

function openFullPage(): void {
  browser.tabs.create({ url: browser.runtime.getURL("page/page.html") });
}

// --- Background communication ---

function listenToBackground(): void {
  const port = browser.runtime.connect({ name: "sidebar" });
  port.onMessage.addListener((message: object) => {
    handleBackgroundMessage(message as RuntimeMessage);
  });
  port.onDisconnect.addListener(() => {
    setStatus(CS.Disconnected);
  });

  browser.runtime.onMessage.addListener(handleBackgroundMessage);
}

function handleBackgroundMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.ConnectionState) {
    setStatus(message.state);
  } else if (message.type === RuntimeMessageType.ChatMessage) {
    setTyping(false);
    removePermissionWaiting();
    let content = message.content;
    if (pendingPageRef) {
      content = buildPageHeader(pendingPageRef) + content.replace(/<img[^>]*><br>/g, "");
      pendingPageRef = null;
    }
    addMessage(content, MS.Penny);
    setInputEnabled(true);
  } else if (message.type === RuntimeMessageType.Typing) {
    setTyping(message.active, message.content);
  } else if (message.type === RuntimeMessageType.PermissionRequest) {
    showPermissionDialog(message.request_id, message.domain, message.url);
  } else if (message.type === RuntimeMessageType.PermissionDismiss) {
    document.getElementById("permission-dialog")?.classList.add("hidden");
    removePermissionWaiting();
  } else if (message.type === RuntimeMessageType.ToolUseState) {
    document.getElementById("tool-use-icon")?.classList.toggle("hidden", !message.enabled);
  } else if (message.type === RuntimeMessageType.PageInfo) {
    updatePageContextBar(message.title, message.url, message.favicon, message.image, message.available);
  }
}

// --- Page context bar ---

function updatePageContextBar(
  title: string, url: string, favicon: string, image: string, available: boolean,
): void {
  const bar = document.getElementById("page-context-bar")!;
  const titleEl = document.getElementById("page-context-title")!;
  const subEl = document.getElementById("page-context-sub")!;
  const faviconEl = document.getElementById("page-context-favicon") as HTMLImageElement;
  const toggle = document.getElementById("page-context-toggle") as HTMLInputElement;

  currentPageImage = image;

  if (!available || !title) {
    bar.classList.add("hidden");
    if (!available || !title) toggle.checked = false;
    return;
  }

  const urlChanged = bar.dataset.url !== url;
  titleEl.textContent = title;
  titleEl.title = title;
  faviconEl.src = favicon || "";
  bar.dataset.url = url;
  const hostname = hostnameOf(url);
  subEl.textContent = hostname;
  subEl.classList.toggle("hidden", hostname === "");
  if (urlChanged) {
    toggle.checked = false;
  }
  bar.classList.remove("hidden");
}

// --- Permission dialog ---

function showPermissionDialog(requestId: string, domain: string, url: string): void {
  const dialog = document.getElementById("permission-dialog")!;
  document.getElementById("permission-domain")!.textContent = domain;
  document.getElementById("permission-url")!.textContent = url;
  dialog.classList.remove("hidden");
  showPermissionWaiting(domain);

  const allowBtn = document.getElementById("permission-allow")!;
  const denyBtn = document.getElementById("permission-deny")!;

  function respond(allowed: boolean): void {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.PermissionResponse,
      request_id: requestId,
      allowed,
    });
    dialog.classList.add("hidden");
    removePermissionWaiting();
    allowBtn.removeEventListener("click", onAllow);
    denyBtn.removeEventListener("click", onDeny);
  }

  function onAllow(): void { respond(true); }
  function onDeny(): void { respond(false); }

  allowBtn.addEventListener("click", onAllow);
  denyBtn.addEventListener("click", onDeny);
}

// Paused-on-permission status row (sidebar-permission.html): same rail,
// no pulse — Penny is blocked on the user's answer, not streaming.

function showPermissionWaiting(domain: string): void {
  const trail = document.createElement("div");
  trail.id = PERM_WAIT_ID;
  trail.className = "live-status";
  const row = document.createElement("div");
  row.className = "status-row";
  const rail = document.createElement("div");
  rail.className = "rail";
  rail.innerHTML = `<span class="dot"></span>`;
  const line = document.createElement("div");
  line.className = "status-line waiting";
  line.textContent = `Waiting on permission to browse ${domain}`;
  row.append(rail, line);
  trail.append(row);
  messagesEl.appendChild(trail);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removePermissionWaiting(): void {
  document.getElementById(PERM_WAIT_ID)?.remove();
}

function send(): void {
  const text = inputEl.value.trim();
  if (!text) return;

  const toggle = document.getElementById("page-context-toggle") as HTMLInputElement;
  const includePage = toggle?.checked ?? false;
  const bar = document.getElementById("page-context-bar")!;
  const titleEl = document.getElementById("page-context-title")!;

  if (includePage && titleEl.textContent) {
    pendingPageRef = {
      title: titleEl.textContent,
      url: bar.dataset.url ?? "",
      image: currentPageImage,
    };
  } else {
    pendingPageRef = null;
  }

  addMessage(text, MS.User);
  setInputEnabled(false);
  browser.runtime.sendMessage({
    type: RuntimeMessageType.SendChat,
    content: text,
    include_page: includePage,
  });
  inputEl.value = "";
  inputEl.rows = 1;
}

function setInputEnabled(enabled: boolean): void {
  inputEl.disabled = !enabled;
  sendBtn.disabled = !enabled;
  const toggle = document.getElementById("page-context-toggle") as HTMLInputElement;
  if (toggle) toggle.disabled = !enabled;
}

function buildPageHeader(ref: { title: string; url: string; image: string }): string {
  const img = ref.image
    ? `<img src="${ref.image}" alt="${ref.title}">`
    : "";
  const link = `<a href="${ref.url}" target="_blank">${ref.title}</a>`;
  return `<div class="page-header">${img}<div class="page-header-label">In response to ${link}</div></div>`;
}

// --- Chat history persistence ---

async function rehydrateHistory(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_CHAT_HISTORY);
  const messages: StoredMessage[] = stored[STORAGE_KEY_CHAT_HISTORY] ?? [];
  for (const msg of messages) {
    renderMessage(msg.text, msg.sender, false);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
  updateEmptyState();
}

async function persistMessage(text: string, sender: MessageSender): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_CHAT_HISTORY);
  const messages: StoredMessage[] = stored[STORAGE_KEY_CHAT_HISTORY] ?? [];
  messages.push({ text, sender });
  if (messages.length > MAX_STORED_MESSAGES) {
    messages.splice(0, messages.length - MAX_STORED_MESSAGES);
  }
  await browser.storage.local.set({ [STORAGE_KEY_CHAT_HISTORY]: messages });
}

// --- Chat UI ---

function renderMessage(text: string, sender: MessageSender, animate = true): void {
  const row = document.createElement("div");
  row.className = `msg-row ${sender}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (sender === MS.Penny) {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    bubble.innerHTML = text;
    row.append(avatar, bubble);
  } else {
    bubble.textContent = text;
    row.append(bubble);
  }
  messagesEl.appendChild(row);
  updateEmptyState();
  if (!animate) return;

  scrollToMessage(row);

  const images = Array.from(row.querySelectorAll("img"));
  for (const img of images) {
    img.addEventListener("load", () => scrollToMessage(row), { once: true });
  }
}

function updateEmptyState(): void {
  const hasMessages = messagesEl.querySelector(".msg-row") !== null;
  document.getElementById(EMPTY_STATE_ID)?.classList.toggle("hidden", hasMessages);
}

function scrollToMessage(div: HTMLElement): void {
  const fitsInView = div.offsetHeight <= messagesEl.clientHeight;
  div.scrollIntoView({ block: fitsInView ? "end" : "start", behavior: "smooth" });
}

function addMessage(text: string, sender: MessageSender): void {
  renderMessage(text, sender);
  persistMessage(text, sender);
}

function setStatus(state: ConnectionState): void {
  statusEl.className = state;
  statusEl.title = state === CS.Connected ? "Connected to Penny" : "Disconnected — reconnecting";
  document.getElementById("conn-banner")?.classList.toggle("hidden", state === CS.Connected);
  if (sendBtn) sendBtn.disabled = state !== CS.Connected;
}

// --- Live tool-status trail (the Prompts tab's rail-dot language) ---

function typingDotsHTML(): string {
  return `<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>`;
}

/** One trail row: rail (dot + connector line unless the last row) and label. */
function trailRowEl(row: TrailRow, index: number, total: number): HTMLElement {
  const wrap = document.createElement("div");
  wrap.className = "status-row";
  const rail = document.createElement("div");
  rail.className = "rail";
  const dot = document.createElement("span");
  dot.className = row.done ? "dot" : "dot live";
  rail.appendChild(dot);
  if (index < total - 1) {
    const line = document.createElement("span");
    line.className = "line";
    rail.appendChild(line);
  }
  const text = document.createElement("div");
  text.className = row.done ? "status-line done" : "status-line live";
  const isLastLive = !row.done && index === total - 1;
  const dots = isLastLive ? typingDotsHTML() : "";
  text.innerHTML = row.done
    ? `<svg class="icon ic-sys"><use href="#i-check"/></svg><span>${row.html}</span>`
    : `<svg class="icon spin ic-model"><use href="#i-spinner"/></svg><span>${row.html}${dots}</span>`;
  wrap.append(rail, text);
  return wrap;
}

function setTyping(active: boolean, content?: string): void {
  const existing = document.getElementById(TYPING_ID);
  if (!active) {
    existing?.remove();
    return;
  }
  const text = content && content.trim() !== "" ? content : "Penny is thinking";
  const rows = parseTrail(text);
  const trail = document.createElement("div");
  trail.id = TYPING_ID;
  trail.className = "live-status";
  rows.forEach((row, index) => trail.appendChild(trailRowEl(row, index, rows.length)));
  if (existing) {
    existing.replaceWith(trail);
  } else {
    messagesEl.appendChild(trail);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function autoResize(): void {
  inputEl.rows = 1;
  const lines = Math.ceil(inputEl.scrollHeight / TEXTAREA_LINE_HEIGHT);
  inputEl.rows = Math.min(lines, TEXTAREA_MAX_ROWS);
}

// --- Boot ---

init();