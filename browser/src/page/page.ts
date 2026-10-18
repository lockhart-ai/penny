/**
 * Full page — wires the four tab modules (prompts, collectors/memories,
 * domains, config) together: init, top-level tab switching, the runtime
 * message dispatch, and connection state.  Each tab's own logic lives in its
 * module (#2565); everything here except the two extracted response branches
 * moved unchanged from the pre-split page.ts.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  type RuntimeMessage,
  RuntimeMessageType,
} from "../protocol.js";
import {
  applyPromptLogsResponse,
  handleCallStarted,
  handlePromptUpdate,
  handleRunOutcome,
  handleToolFinished,
  handleToolStarted,
  isPromptsLoaded,
  markPromptsLoaded,
  requestPromptLogs,
  resetAndFetchPrompts,
  setupPrompts,
} from "./prompts.js";
import {
  handleCollectionTriggerResult,
  handleCollectorChanged,
  handleCollectorDetailResponse,
  handleCollectorPageResponse,
  handleCollectorSaveError,
  handleCollectorsResponse,
  requestCollectors,
  setupCollectors,
} from "./collectors.js";
import { loadDomainsFromCache, renderDomains, setupDomains } from "./domains.js";
import { applyConfigResponse, loadToolUseState, setupConfig } from "./config.js";

type Tab =
  | "prompts"
  | "memories"
  | "domains"
  | "config";

// ============================================================
// Init
// ============================================================

function init(): void {
  browser.runtime.onMessage.addListener(handleMessage);

  // Top-level tab switching
  for (const btn of Array.from(document.querySelectorAll(".tab"))) {
    btn.addEventListener("click", () => switchTab(btn.getAttribute("data-tab") as Tab));
  }

  // Load initial data for the prompts tab (default)
  requestPromptLogs(0);
  markPromptsLoaded();

  // Set up all panel interactions
  setupPrompts();
  setupCollectors();
  setupDomains();
  setupConfig();
}

function switchTab(tab: Tab): void {
  for (const btn of Array.from(document.querySelectorAll(".tab"))) {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tab);
  }
  for (const panel of Array.from(document.querySelectorAll(".panel"))) {
    panel.classList.toggle("hidden", panel.id !== `panel-${tab}`);
  }

  // Request data for the activated tab
  if (tab === "prompts" && !isPromptsLoaded()) {
    requestPromptLogs(0);
    markPromptsLoaded();
  } else if (tab === "memories") {
    requestCollectors();
  } else if (tab === "domains") {
    loadDomainsFromCache();
  } else if (tab === "config") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ConfigRequest });
    loadToolUseState();
  }
}

// ============================================================
// Message handler
// ============================================================

function handleMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.PromptLogsResponse) {
    applyPromptLogsResponse(message);
  } else if (message.type === RuntimeMessageType.PromptLogUpdate) {
    handlePromptUpdate(message.prompt);
  } else if (message.type === RuntimeMessageType.RunOutcomeUpdate) {
    handleRunOutcome(message.run_id, message.outcome, message.reason);
  } else if (message.type === RuntimeMessageType.CallStarted) {
    handleCallStarted(message);
  } else if (message.type === RuntimeMessageType.ToolStarted) {
    handleToolStarted(message);
  } else if (message.type === RuntimeMessageType.ToolFinished) {
    handleToolFinished(message);
  } else if (message.type === RuntimeMessageType.ConfigResponse) {
    applyConfigResponse(message);
  } else if (message.type === RuntimeMessageType.ToolUseState) {
    const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement | null;
    if (toggle) toggle.checked = message.enabled;
  } else if (message.type === RuntimeMessageType.DomainPermissionsSync) {
    renderDomains(message.permissions);
  } else if (message.type === RuntimeMessageType.MemoriesResponse) {
    handleCollectorsResponse(message.memories);
  } else if (message.type === RuntimeMessageType.MemoryDetailResponse) {
    handleCollectorDetailResponse(message);
  } else if (message.type === RuntimeMessageType.MemoryPageResponse) {
    handleCollectorPageResponse(message);
  } else if (message.type === RuntimeMessageType.MemoryChanged) {
    handleCollectorChanged(message.name);
  } else if (message.type === RuntimeMessageType.CollectionTriggerResult) {
    handleCollectionTriggerResult(message);
  } else if (message.type === RuntimeMessageType.MemorySaveError) {
    handleCollectorSaveError(message);
  } else if (message.type === RuntimeMessageType.ConnectionState) {
    handleConnectionState(message.state);
  }
}

function handleConnectionState(state: ConnectionState): void {
  // background.ts only pushes the current state on a *change* — the page
  // gets no initial push on load (that's a sidebar-port-only courtesy) — so
  // every Connected the page ever hears here is a state change, never the
  // page's own starting state. That covers both cases: a reconnect after a
  // drop (init()'s own fetch is stale) and a load that raced the socket
  // coming up (init()'s own fetch went nowhere). Either way, catch up now.
  if (state !== CS.Connected) return;
  resetAndFetchPrompts();
  const memoriesPanel = document.getElementById("panel-memories");
  if (memoriesPanel && !memoriesPanel.classList.contains("hidden")) {
    requestCollectors();
  }
}

// ============================================================
// Boot
// ============================================================

init();