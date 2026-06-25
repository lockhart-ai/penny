/**
 * Background script — owns the WebSocket connection to Penny server.
 * Handles tool requests, permission prompts, and active tab tracking.
 * The sidebar communicates with this script via browser.runtime messaging.
 */

import {
  type ConnectionState,
  ConnectionState as CS,
  type DomainAllowlist,
  type DomainPermissionEntry,
  HEARTBEAT_INTERVAL_MS,
  type MemorySection,
  type PageContext,
  RECONNECT_DELAY_MS,
  type RuntimeMessage,
  RuntimeMessageType,
  SERVER_URL,
  STORAGE_KEY_DEVICE_LABEL,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
  STORAGE_KEY_TOOL_USE,
  type WsIncomingPayload,
  WsIncomingType,
  type WsIncomingToolRequestPayload,
  WsIncomingType as WsIn,
  WsOutgoingType,
} from "../protocol.js";
import { browseUrl } from "./tools/browse_url.js";
import { logStep, setWsStateProvider } from "./ws_log.js";

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
let deviceLabel: string | null = null;
let connectionState: ConnectionState = CS.Disconnected;
let currentPageContext: PageContext | null = null;

// Let the trace logger read the live socket state without importing it back.
setWsStateProvider(() => (ws ? ws.readyState : undefined));

// URLs we should never try to extract from
const SKIP_URL_PREFIXES = ["about:", "moz-extension:", "chrome:", "data:", "file:"];

// --- Lifecycle ---

async function init(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DEVICE_LABEL);
  deviceLabel = (stored[STORAGE_KEY_DEVICE_LABEL] as string) ?? null;

  browser.runtime.onMessage.addListener(handleRuntimeMessage);
  browser.storage.onChanged.addListener(handleStorageChange);
  browser.tabs.onActivated.addListener(handleTabActivated);
  browser.tabs.onUpdated.addListener(handleTabUpdated);

  if (deviceLabel) {
    connect();
  }

  extractFromActiveTab();
}

function handleStorageChange(
  changes: Record<string, browser.storage.StorageChange>,
): void {
  if (changes[STORAGE_KEY_DEVICE_LABEL]?.newValue) {
    deviceLabel = changes[STORAGE_KEY_DEVICE_LABEL].newValue as string;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connect();
    }
  }
}

// --- Active tab tracking ---

function handleTabActivated(): void {
  logStep("tab", "onActivated → extractFromActiveTab");
  extractFromActiveTab();
}

function handleTabUpdated(
  tabId: number,
  changeInfo: browser.tabs._OnUpdatedChangeInfo,
): void {
  if (changeInfo.status === "complete") {
    logStep("tab", `onUpdated complete tab=${tabId} → extractFromActiveTab`);
    extractFromActiveTab();
  }
}

async function extractFromActiveTab(): Promise<void> {
  let favicon = "";
  try {
    logStep("extractActive", "query active tab");
    const tabs = await browser.tabs.query({ active: true, currentWindow: true });
    const tab = tabs[0];
    if (!tab?.id || !tab.url || SKIP_URL_PREFIXES.some((p) => tab.url!.startsWith(p))) {
      currentPageContext = null;
      broadcastPageInfo("", "", "", "", false);
      return;
    }

    favicon = tab.favIconUrl ?? "";

    logStep("extractActive", `executeScript on active tab=${tab.id}`);
    const results = await browser.tabs.executeScript(tab.id, {
      file: "/dist/content/extract_text.js",
      runAt: "document_idle",
    });
    logStep("extractActive", `executeScript returned tab=${tab.id}`);

    if (results?.[0]) {
      const data = results[0] as {
        title: string;
        url: string;
        text: string;
        image: string;
        ready: boolean;
        extracted: boolean;
      };
      if (!data.ready || !data.extracted) {
        currentPageContext = null;
        broadcastPageInfo("", "", "", "", false);
        return;
      }
      currentPageContext = {
        title: data.title,
        url: data.url,
        text: data.text,
        image: data.image,
      };
      broadcastPageInfo(data.title, data.url, favicon, data.image, true);
      sendHeartbeat();
    } else {
      currentPageContext = null;
      broadcastPageInfo("", "", "", "", false);
    }
  } catch {
    currentPageContext = null;
    broadcastPageInfo("", "", "", "", false);
  }
}



function broadcastPageInfo(
  title: string, url: string, favicon: string, image: string, available: boolean,
): void {
  broadcastToSidebar({
    type: RuntimeMessageType.PageInfo,
    title,
    url,
    favicon,
    image,
    available,
  });
}

// --- Runtime messaging (sidebar ↔ background) ---

function handleRuntimeMessage(message: RuntimeMessage): void {
  if (message.type === RuntimeMessageType.SendChat) {
    sendChatToServer(message.content, message.include_page);
  } else if (message.type === RuntimeMessageType.ConfigRequest) {
    requestConfig();
  } else if (message.type === RuntimeMessageType.ConfigUpdate) {
    sendConfigUpdate(message.key, message.value);
  } else if (message.type === RuntimeMessageType.ToolUseToggle) {
    setToolUse(message.enabled);
  } else if (message.type === RuntimeMessageType.DomainUpdate) {
    sendDomainUpdate(message.domain, message.permission);
  } else if (message.type === RuntimeMessageType.DomainDelete) {
    sendDomainDelete(message.domain);
  } else if (message.type === RuntimeMessageType.PermissionResponse) {
    sendPermissionDecision(message.request_id, message.allowed);
  } else if (message.type === RuntimeMessageType.SchedulesRequest) {
    requestSchedules();
  } else if (message.type === RuntimeMessageType.ScheduleAdd) {
    sendScheduleAdd(message.command);
  } else if (message.type === RuntimeMessageType.ScheduleUpdate) {
    sendScheduleUpdate(message.schedule_id, message.prompt_text);
  } else if (message.type === RuntimeMessageType.ScheduleDelete) {
    sendScheduleDelete(message.schedule_id);
  } else if (message.type === RuntimeMessageType.PromptLogsRequest) {
    requestPromptLogs(message.agent_name, message.offset, message.query, message.flagged_only);
  } else if (message.type === RuntimeMessageType.MemoriesRequest) {
    requestMemories(message.query);
  } else if (message.type === RuntimeMessageType.MemoryDetailRequest) {
    requestMemoryDetail(message.name, message.query);
  } else if (message.type === RuntimeMessageType.MemoryPageRequest) {
    requestMemoryPage(message.name, message.section, message.offset, message.query);
  } else if (message.type === RuntimeMessageType.CollectionTrigger) {
    triggerCollection(message.name);
  } else if (message.type === RuntimeMessageType.MemoryCreate) {
    sendMemoryCreate(message);
  } else if (message.type === RuntimeMessageType.MemoryUpdate) {
    sendMemoryUpdate(message);
  } else if (message.type === RuntimeMessageType.MemoryArchive) {
    sendMemoryArchive(message.name);
  } else if (message.type === RuntimeMessageType.EntryCreate) {
    sendEntryCreate(message.memory, message.key, message.content);
  } else if (message.type === RuntimeMessageType.EntryUpdate) {
    sendEntryUpdate(message.memory, message.key, message.content);
  } else if (message.type === RuntimeMessageType.EntryDelete) {
    sendEntryDelete(message.memory, message.key);
  } else if (message.type === RuntimeMessageType.CursorSet) {
    sendCursorSet(message.name, message.log_name, message.last_read_at);
  } else if (message.type === RuntimeMessageType.CursorClear) {
    sendCursorClear(message.name, message.log_name);
  }
}

function broadcastToSidebar(message: RuntimeMessage): void {
  browser.runtime.sendMessage(message).catch(() => {
    // Sidebar not open — ignore
  });
}

function setConnectionState(state: ConnectionState): void {
  connectionState = state;
  broadcastToSidebar({ type: RuntimeMessageType.ConnectionState, state });
}

// --- WebSocket ---

function connect(): void {
  if (!deviceLabel) return;
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }

  setConnectionState(CS.Reconnecting);
  logStep("ws", "connect(): new WebSocket()");
  ws = new WebSocket(SERVER_URL);

  ws.addEventListener("open", () => {
    console.log("Background: connected to Penny server");
    logStep("ws", "open");
    startHeartbeat();
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    const data: WsIncomingPayload = JSON.parse(event.data);

    if (data.type === WsIncomingType.Status && data.connected) {
      setConnectionState(CS.Connected);
      sendRegister();
      sendCapabilities();
    } else if (data.type === WsIncomingType.Message) {
      broadcastToSidebar({ type: RuntimeMessageType.ChatMessage, content: data.content });
    } else if (data.type === WsIncomingType.Typing) {
      broadcastToSidebar({ type: RuntimeMessageType.Typing, active: data.active, content: data.content });
    } else if (data.type === WsIncomingType.ToolRequest) {
      handleToolRequest(data);
    } else if (data.type === WsIn.ConfigResponse) {
      broadcastToSidebar({ type: RuntimeMessageType.ConfigResponse, params: data.params });
    } else if (data.type === WsIn.DomainPermissionsSync) {
      syncDomainPermissionsToLocal(data.permissions);
      broadcastToSidebar({
        type: RuntimeMessageType.DomainPermissionsSync,
        permissions: data.permissions,
      });
    } else if (data.type === WsIn.PermissionPrompt) {
      broadcastToSidebar({
        type: RuntimeMessageType.PermissionRequest,
        request_id: data.request_id,
        domain: data.domain,
        url: data.url,
      });
    } else if (data.type === WsIn.PermissionDismiss) {
      broadcastToSidebar({ type: RuntimeMessageType.PermissionDismiss });
    } else if (data.type === WsIn.SchedulesResponse) {
      broadcastToSidebar({
        type: RuntimeMessageType.SchedulesResponse,
        schedules: data.schedules,
        error: data.error,
      });
    } else if (data.type === WsIn.PromptLogUpdate) {
      broadcastToSidebar({
        type: RuntimeMessageType.PromptLogUpdate,
        prompt: data.prompt,
      });
    } else if (data.type === WsIn.PromptLogsResponse) {
      broadcastToSidebar({
        type: RuntimeMessageType.PromptLogsResponse,
        runs: data.runs,
        has_more: data.has_more,
      });
    } else if (data.type === WsIn.RunOutcomeUpdate) {
      broadcastToSidebar({
        type: RuntimeMessageType.RunOutcomeUpdate,
        run_id: data.run_id,
        outcome: data.outcome,
        reason: data.reason,
      });
    } else if (data.type === WsIn.MemoriesResponse) {
      broadcastToSidebar({
        type: RuntimeMessageType.MemoriesResponse,
        memories: data.memories,
      });
    } else if (data.type === WsIn.MemoryDetailResponse) {
      broadcastToSidebar({
        type: RuntimeMessageType.MemoryDetailResponse,
        memory: data.memory,
        entries: data.entries,
        entries_has_more: data.entries_has_more,
        collector_runs: data.collector_runs,
        collector_runs_has_more: data.collector_runs_has_more,
        cursors: data.cursors,
      });
    } else if (data.type === WsIn.MemoryPageResponse) {
      broadcastToSidebar({
        type: RuntimeMessageType.MemoryPageResponse,
        name: data.name,
        section: data.section,
        entries: data.entries,
        runs: data.runs,
        has_more: data.has_more,
      });
    } else if (data.type === WsIn.MemoryChanged) {
      broadcastToSidebar({
        type: RuntimeMessageType.MemoryChanged,
        name: data.name,
      });
    } else if (data.type === WsIn.CollectionTriggerResult) {
      broadcastToSidebar({
        type: RuntimeMessageType.CollectionTriggerResult,
        name: data.name,
        success: data.success,
        message: data.message,
      });
    }
  });

  ws.addEventListener("close", (event: CloseEvent) => {
    logStep("ws", `close: code=${event.code} reason=${event.reason || "(none)"} clean=${event.wasClean}`);
    stopHeartbeat();
    setConnectionState(CS.Reconnecting);
    scheduleReconnect();
  });

  ws.addEventListener("error", () => {
    // Error fires before close — close handler will reconnect
    logStep("ws", "error");
  });
}

function scheduleReconnect(): void {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, RECONNECT_DELAY_MS);
}

function startHeartbeat(): void {
  stopHeartbeat();
  heartbeatTimer = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);
}

function stopHeartbeat(): void {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = null;
  }
}

function sendChatToServer(content: string, includePage: boolean): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || !deviceLabel) return;
  const payload: Record<string, unknown> = {
    type: WsOutgoingType.Message,
    content,
    sender: deviceLabel,
  };
  if (includePage && currentPageContext) {
    payload.page_context = currentPageContext;
  }
  ws.send(JSON.stringify(payload));
}

function sendRegister(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN || !deviceLabel) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.Register, sender: deviceLabel }));
}

function sendHeartbeat(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    logStep("heartbeat", "skipped — socket not open");
    return;
  }
  logStep("heartbeat", "send");
  ws.send(JSON.stringify({ type: WsOutgoingType.Heartbeat }));
}

function requestConfig(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ConfigRequest }));
}

function sendConfigUpdate(key: string, value: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ConfigUpdate, key, value }));
}

function sendDomainUpdate(domain: string, permission: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.DomainUpdate, domain, permission }));
}

function sendPermissionDecision(requestId: string, allowed: boolean): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.PermissionDecision, request_id: requestId, allowed }));
}

function sendDomainDelete(domain: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.DomainDelete, domain }));
}

function requestSchedules(): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.SchedulesRequest }));
}

function sendScheduleAdd(command: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ScheduleAdd, command }));
}

function sendScheduleUpdate(scheduleId: number, promptText: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: WsOutgoingType.ScheduleUpdate,
    schedule_id: scheduleId,
    prompt_text: promptText,
  }));
}

function sendScheduleDelete(scheduleId: number): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.ScheduleDelete, schedule_id: scheduleId }));
}

function requestPromptLogs(
  agentName?: string,
  offset?: number,
  query?: string,
  flaggedOnly?: boolean,
): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload: Record<string, unknown> = { type: WsOutgoingType.PromptLogsRequest };
  if (agentName) payload.agent_name = agentName;
  if (offset) payload.offset = offset;
  if (query) payload.query = query;
  if (flaggedOnly) payload.flagged_only = true;
  ws.send(JSON.stringify(payload));
}

function requestMemories(query?: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload: Record<string, unknown> = { type: WsOutgoingType.MemoriesRequest };
  if (query) payload.query = query;
  ws.send(JSON.stringify(payload));
}

function requestMemoryDetail(name: string, query?: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload: Record<string, unknown> = { type: WsOutgoingType.MemoryDetailRequest, name };
  if (query) payload.query = query;
  ws.send(JSON.stringify(payload));
}

function requestMemoryPage(
  name: string,
  section: MemorySection,
  offset: number,
  query?: string,
): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const payload: Record<string, unknown> = {
    type: WsOutgoingType.MemoryPageRequest,
    name,
    section,
    offset,
  };
  if (query) payload.query = query;
  ws.send(JSON.stringify(payload));
}

function triggerCollection(name: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.CollectionTrigger, name }));
}

function sendMemoryCreate(message: {
  name: string;
  description: string;
  intent: string;
  inclusion: string;
  recall: string;
  published?: boolean;
  extraction_prompt?: string | null;
  collector_interval_seconds?: number | null;
}): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: WsOutgoingType.MemoryCreate,
    name: message.name,
    description: message.description,
    intent: message.intent,
    inclusion: message.inclusion,
    recall: message.recall,
    published: message.published ?? false,
    extraction_prompt: message.extraction_prompt ?? null,
    collector_interval_seconds: message.collector_interval_seconds ?? null,
  }));
}

function sendMemoryUpdate(message: {
  name: string;
  description?: string | null;
  intent?: string | null;
  inclusion?: string | null;
  recall?: string | null;
  published?: boolean | null;
  extraction_prompt?: string | null;
  collector_interval_seconds?: number | null;
}): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: WsOutgoingType.MemoryUpdate,
    name: message.name,
    description: message.description ?? null,
    intent: message.intent ?? null,
    inclusion: message.inclusion ?? null,
    recall: message.recall ?? null,
    published: message.published ?? null,
    extraction_prompt: message.extraction_prompt ?? null,
    collector_interval_seconds: message.collector_interval_seconds ?? null,
  }));
}

function sendCursorSet(name: string, logName: string, lastReadAt: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: WsOutgoingType.CursorSet,
    name,
    log_name: logName,
    last_read_at: lastReadAt,
  }));
}

function sendCursorClear(name: string, logName: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.CursorClear, name, log_name: logName }));
}

function sendMemoryArchive(name: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.MemoryArchive, name }));
}

function sendEntryCreate(memory: string, key: string, content: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.EntryCreate, memory, key, content }));
}

function sendEntryUpdate(memory: string, key: string, content: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.EntryUpdate, memory, key, content }));
}

function sendEntryDelete(memory: string, key: string): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: WsOutgoingType.EntryDelete, memory, key }));
}

function syncDomainPermissionsToLocal(permissions: DomainPermissionEntry[]): void {
  const allowlist: DomainAllowlist = {};
  for (const { domain, permission } of permissions) {
    allowlist[domain] = permission;
  }
  browser.storage.local.set({ [STORAGE_KEY_DOMAIN_ALLOWLIST]: allowlist });
}

async function sendCapabilities(): Promise<void> {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
  const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
  ws.send(JSON.stringify({ type: WsOutgoingType.CapabilitiesUpdate, tool_use_enabled: enabled }));
}

async function setToolUse(enabled: boolean): Promise<void> {
  await browser.storage.local.set({ [STORAGE_KEY_TOOL_USE]: enabled });
  await sendCapabilities();
  broadcastToSidebar({ type: RuntimeMessageType.ToolUseState, enabled });
}

function sendToolResponse(
  requestId: string, result?: string, error?: string, image?: string,
): void {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    logStep("tool", `sendToolResponse DROPPED (socket not open) id=${requestId}`);
    return;
  }
  logStep(
    "tool",
    `sendToolResponse id=${requestId} ${error ? `error=${error}` : `result=${(result ?? "").length}ch`}`,
  );
  ws.send(JSON.stringify({
    type: WsOutgoingType.ToolResponse,
    request_id: requestId,
    result,
    error,
    image,
  }));
}

// --- Tool request handling ---

async function handleToolRequest(request: WsIncomingToolRequestPayload): Promise<void> {
  const { request_id, tool, arguments: args } = request;
  logStep("tool", `handleToolRequest start tool=${tool} id=${request_id}`);

  try {
    if (tool === "browse_url") {
      const result = await executeBrowseUrl(request_id, args);
      logStep("tool", `browse_url returned id=${request_id}, sending response`);
      sendToolResponse(request_id, result.text, undefined, result.image);
    } else {
      sendToolResponse(request_id, undefined, `Unknown tool: ${tool}`);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    logStep("tool", `handleToolRequest threw id=${request_id}: ${message}`);
    sendToolResponse(request_id, undefined, message);
  }
}

async function executeBrowseUrl(
  _requestId: string,
  args: Record<string, unknown>,
): Promise<{ text: string; image: string }> {
  const url = args.url as string;
  if (!url) throw new Error("Missing required argument: url");
  return await browseUrl(url);
}

// --- Sidebar state sync ---

browser.runtime.onConnect.addListener(async (port) => {
  if (port.name === "sidebar") {
    port.postMessage({ type: RuntimeMessageType.ConnectionState, state: connectionState });
    const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
    const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
    port.postMessage({ type: RuntimeMessageType.ToolUseState, enabled });
    if (currentPageContext) {
      port.postMessage({
        type: RuntimeMessageType.PageInfo,
        title: currentPageContext.title,
        url: currentPageContext.url,
        favicon: "",
        image: currentPageContext.image,
        available: true,
      });
    }
  }
});

// --- Debug: test browse_url from background console ---
// Usage: debugBrowseUrl("https://example.com")

// @ts-expect-error -- exposed for console debugging
globalThis.debugBrowseUrl = (url: string): void => {
  browseUrl(url).then(
    (result) => {
      console.log(`[debug] ${result.text.length} chars, image: ${result.image || "none"}`);
      console.log(result.text);
    },
    (err) => console.error("[debug] ERROR:", err),
  );
};

// --- Boot ---

init();
