/**
 * WebSocket protocol types shared between sidebar and background scripts.
 *
 * Wire-format identifiers (message ``type`` strings, etc.) MUST match the
 * Python side's expectations in penny/penny/channels/browser/models.py since
 * both sides need to encode/decode the same wire format. Anything beyond
 * those identifiers — page sizes, timeouts, thresholds — should NOT be
 * mirrored: derive it from server payloads or pick a single owner.
 */

// --- Connection ---

export const SERVER_URL = "ws://localhost:9090";
export const RECONNECT_DELAY_MS = 3000;
// Cadence of the keepalive heartbeat the background script sends while
// connected.  Gives the server a liveness signal (and resets Penny's idle
// timer); kept well under the server's ~20s ping interval so the socket
// stays warm rather than relying on tab-activity events alone.
export const HEARTBEAT_INTERVAL_MS = 15000;

export type ConnectionState = "connected" | "disconnected" | "reconnecting";
export const ConnectionState = {
  Connected: "connected",
  Disconnected: "disconnected",
  Reconnecting: "reconnecting",
} as const satisfies Record<string, ConnectionState>;

// --- WebSocket: outgoing (browser → server) ---

export type WsOutgoingType =
  | "message"
  | "tool_response"
  | "heartbeat"
  | "config_request"
  | "config_update"
  | "register"
  | "capabilities_update"
  | "domain_update"
  | "domain_delete"
  | "permission_decision"
  | "schedules_request"
  | "schedule_add"
  | "schedule_update"
  | "schedule_delete"
  | "prompt_logs_request"
  | "memories_request"
  | "memory_detail_request"
  | "memory_page_request"
  | "memory_create"
  | "memory_update"
  | "memory_archive"
  | "entry_create"
  | "entry_update"
  | "entry_delete"
  | "collection_trigger"
  | "cursor_set"
  | "cursor_clear";
export const WsOutgoingType = {
  Message: "message",
  ToolResponse: "tool_response",
  Heartbeat: "heartbeat",
  ConfigRequest: "config_request",
  ConfigUpdate: "config_update",
  Register: "register",
  CapabilitiesUpdate: "capabilities_update",
  DomainUpdate: "domain_update",
  DomainDelete: "domain_delete",
  PermissionDecision: "permission_decision",
  SchedulesRequest: "schedules_request",
  ScheduleAdd: "schedule_add",
  ScheduleUpdate: "schedule_update",
  ScheduleDelete: "schedule_delete",
  PromptLogsRequest: "prompt_logs_request",
  MemoriesRequest: "memories_request",
  MemoryDetailRequest: "memory_detail_request",
  MemoryPageRequest: "memory_page_request",
  MemoryCreate: "memory_create",
  MemoryUpdate: "memory_update",
  MemoryArchive: "memory_archive",
  EntryCreate: "entry_create",
  EntryUpdate: "entry_update",
  EntryDelete: "entry_delete",
  CollectionTrigger: "collection_trigger",
  CursorSet: "cursor_set",
  CursorClear: "cursor_clear",
} as const satisfies Record<string, WsOutgoingType>;

export interface WsOutgoingMessage {
  type: typeof WsOutgoingType.Message;
  content: string;
  sender: string;
  page_context?: PageContext;
}

export interface WsOutgoingToolResponse {
  type: typeof WsOutgoingType.ToolResponse;
  request_id: string;
  result?: string;
  error?: string;
}

export interface WsOutgoingHeartbeat {
  type: typeof WsOutgoingType.Heartbeat;
}

export interface WsOutgoingCapabilitiesUpdate {
  type: typeof WsOutgoingType.CapabilitiesUpdate;
  tool_use_enabled: boolean;
}

export interface WsOutgoingSchedulesRequest {
  type: typeof WsOutgoingType.SchedulesRequest;
}

export interface WsOutgoingScheduleAdd {
  type: typeof WsOutgoingType.ScheduleAdd;
  command: string;
}

export interface WsOutgoingScheduleUpdate {
  type: typeof WsOutgoingType.ScheduleUpdate;
  schedule_id: number;
  prompt_text: string;
}

export interface WsOutgoingScheduleDelete {
  type: typeof WsOutgoingType.ScheduleDelete;
  schedule_id: number;
}

export type WsOutgoing =
  | WsOutgoingMessage
  | WsOutgoingToolResponse
  | WsOutgoingHeartbeat
  | WsOutgoingCapabilitiesUpdate
  | WsOutgoingSchedulesRequest
  | WsOutgoingScheduleAdd
  | WsOutgoingScheduleUpdate
  | WsOutgoingScheduleDelete;

// --- WebSocket: incoming (server → browser) ---

export type WsIncomingType =
  | "message"
  | "typing"
  | "status"
  | "tool_request"
  | "config_response"
  | "domain_permissions_sync"
  | "permission_prompt"
  | "permission_dismiss"
  | "schedules_response"
  | "prompt_logs_response"
  | "prompt_log_update"
  | "run_outcome_update"
  | "memories_response"
  | "memory_detail_response"
  | "memory_page_response"
  | "memory_changed"
  | "collection_trigger_result";
export const WsIncomingType = {
  Message: "message",
  Typing: "typing",
  Status: "status",
  ToolRequest: "tool_request",
  ConfigResponse: "config_response",
  DomainPermissionsSync: "domain_permissions_sync",
  PermissionPrompt: "permission_prompt",
  PermissionDismiss: "permission_dismiss",
  SchedulesResponse: "schedules_response",
  PromptLogsResponse: "prompt_logs_response",
  PromptLogUpdate: "prompt_log_update",
  RunOutcomeUpdate: "run_outcome_update",
  MemoriesResponse: "memories_response",
  MemoryDetailResponse: "memory_detail_response",
  MemoryPageResponse: "memory_page_response",
  MemoryChanged: "memory_changed",
  CollectionTriggerResult: "collection_trigger_result",
} as const satisfies Record<string, WsIncomingType>;

export interface WsIncomingMessagePayload {
  type: typeof WsIncomingType.Message;
  content: string;
}

export interface WsIncomingTypingPayload {
  type: typeof WsIncomingType.Typing;
  active: boolean;
  content?: string;
}

export interface WsIncomingStatusPayload {
  type: typeof WsIncomingType.Status;
  connected: boolean;
}

export interface WsIncomingToolRequestPayload {
  type: typeof WsIncomingType.ToolRequest;
  request_id: string;
  tool: string;
  arguments: Record<string, unknown>;
}

export interface RuntimeConfigParam {
  key: string;
  value: string;
  default: string;
  description: string;
  type: "int" | "float" | "str";
  group: string;
}

export interface WsIncomingConfigPayload {
  type: typeof WsIncomingType.ConfigResponse;
  params: RuntimeConfigParam[];
}

export interface DomainPermissionEntry {
  domain: string;
  permission: DomainPermission;
}

export interface WsIncomingDomainPermissionsPayload {
  type: typeof WsIncomingType.DomainPermissionsSync;
  permissions: DomainPermissionEntry[];
}

export interface WsIncomingPermissionPromptPayload {
  type: typeof WsIncomingType.PermissionPrompt;
  request_id: string;
  domain: string;
  url: string;
}

export interface WsIncomingPermissionDismissPayload {
  type: typeof WsIncomingType.PermissionDismiss;
  request_id: string;
}

export interface ScheduleItem {
  id: number;
  timing_description: string;
  prompt_text: string;
  cron_expression: string;
}

export interface WsIncomingSchedulesPayload {
  type: typeof WsIncomingType.SchedulesResponse;
  schedules: ScheduleItem[];
  error: string | null;
}

export interface PromptLogEntry {
  id: number;
  timestamp: string;
  model: string;
  agent_name: string;
  prompt_type: string;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  // The bound collection (collector cycles) / null (chat, schedule), stamped at
  // write time so a live run is labelled from its first prompt.
  run_target: string | null;
  messages: Record<string, unknown>[];
  response: Record<string, unknown>;
  thinking: string;
  has_tools: boolean;
}

export interface PromptLogRun {
  run_id: string;
  agent_name: string;
  prompt_count: number;
  started_at: string;
  ended_at: string;
  total_duration_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  run_outcome: RunOutcome | null;
  run_reason: string | null;
  run_target: string | null;
  prompts: PromptLogEntry[];
}

/** First-class outcome of a collector cycle (mirrors penny's RunOutcome). */
export type RunOutcome = "failed" | "no_work" | "worked" | "incomplete" | "cancelled";

export interface WsIncomingPromptLogsPayload {
  type: typeof WsIncomingType.PromptLogsResponse;
  runs: PromptLogRun[];
  has_more: boolean;
}

export interface WsIncomingPromptLogUpdatePayload {
  type: typeof WsIncomingType.PromptLogUpdate;
  prompt: PromptLogEntry & { run_id: string };
}

export interface WsIncomingRunOutcomePayload {
  type: typeof WsIncomingType.RunOutcomeUpdate;
  run_id: string;
  outcome: RunOutcome;
  reason: string;
}

export interface MemoryRecord {
  name: string;
  type: "collection" | "log";
  description: string;
  /** The user's stated goal at creation — editable only via the user (UI)
   * path, never by the agent's collection_update tool. */
  intent: string | null;
  /** Stage-1 routing: does this memory participate in recall at all. */
  inclusion: "always" | "relevant" | "never";
  /** Stage-2 entry rendering once included. */
  recall: "all" | "relevant" | "recent";
  /** Pub/sub: when true the notifier delivers new entries to the user. Orthogonal to recall. */
  published: boolean;
  archived: boolean;
  extraction_prompt: string | null;
  collector_interval_seconds: number | null;
  last_collected_at: string | null;
  entry_count: number;
}

/** One read-cursor a collection holds over a log it reads. */
export interface CursorRecord {
  log_name: string;
  /** ISO-8601 UTC high-water mark of what the collection has read. */
  last_read_at: string;
}

export interface MemoryEntryRecord {
  id: number;
  key: string | null;
  content: string;
  author: string;
  created_at: string;
}

/** Independently-paginated sections of a memory's detail view. */
export type MemorySection = "entries" | "collector_runs";

export interface WsIncomingMemoriesPayload {
  type: typeof WsIncomingType.MemoriesResponse;
  memories: MemoryRecord[];
}

export interface WsIncomingMemoryDetailPayload {
  type: typeof WsIncomingType.MemoryDetailResponse;
  memory: MemoryRecord;
  entries: MemoryEntryRecord[];
  entries_has_more: boolean;
  collector_runs: MemoryEntryRecord[];
  collector_runs_has_more: boolean;
  cursors: CursorRecord[];
}

export interface WsIncomingMemoryPagePayload {
  type: typeof WsIncomingType.MemoryPageResponse;
  name: string;
  section: MemorySection;
  entries: MemoryEntryRecord[];
  has_more: boolean;
}

export interface WsIncomingMemoryChangedPayload {
  type: typeof WsIncomingType.MemoryChanged;
  name: string | null;
}

export interface WsIncomingCollectionTriggerResultPayload {
  type: typeof WsIncomingType.CollectionTriggerResult;
  name: string;
  success: boolean;
  message: string;
}

export type WsIncomingPayload =
  | WsIncomingMessagePayload
  | WsIncomingTypingPayload
  | WsIncomingStatusPayload
  | WsIncomingToolRequestPayload
  | WsIncomingConfigPayload
  | WsIncomingDomainPermissionsPayload
  | WsIncomingPermissionPromptPayload
  | WsIncomingPermissionDismissPayload
  | WsIncomingSchedulesPayload
  | WsIncomingPromptLogsPayload
  | WsIncomingPromptLogUpdatePayload
  | WsIncomingRunOutcomePayload
  | WsIncomingMemoriesPayload
  | WsIncomingMemoryDetailPayload
  | WsIncomingMemoryPagePayload
  | WsIncomingMemoryChangedPayload
  | WsIncomingCollectionTriggerResultPayload;

// --- Runtime messages (sidebar ↔ background) ---

export type RuntimeMessageType =
  | "send_chat"
  | "chat_message"
  | "typing"
  | "connection_state"
  | "permission_request"
  | "permission_response"
  | "permission_dismiss"
  | "page_info"
  | "config_request"
  | "config_response"
  | "config_update"
  | "tool_use_toggle"
  | "tool_use_state"
  | "domain_update"
  | "domain_delete"
  | "domain_permissions_sync"
  | "schedules_request"
  | "schedules_response"
  | "schedule_add"
  | "schedule_update"
  | "schedule_delete"
  | "prompt_logs_request"
  | "prompt_logs_response"
  | "prompt_log_update"
  | "run_outcome_update"
  | "memories_request"
  | "memories_response"
  | "memory_detail_request"
  | "memory_detail_response"
  | "memory_page_request"
  | "memory_page_response"
  | "memory_changed"
  | "memory_create"
  | "memory_update"
  | "memory_archive"
  | "entry_create"
  | "entry_update"
  | "entry_delete"
  | "collection_trigger"
  | "collection_trigger_result"
  | "cursor_set"
  | "cursor_clear";

export const RuntimeMessageType = {
  SendChat: "send_chat",
  ChatMessage: "chat_message",
  Typing: "typing",
  ConnectionState: "connection_state",
  PermissionRequest: "permission_request",
  PermissionResponse: "permission_response",
  PermissionDismiss: "permission_dismiss",
  PageInfo: "page_info",
  ConfigRequest: "config_request",
  ConfigResponse: "config_response",
  ConfigUpdate: "config_update",
  ToolUseToggle: "tool_use_toggle",
  ToolUseState: "tool_use_state",
  DomainUpdate: "domain_update",
  DomainDelete: "domain_delete",
  DomainPermissionsSync: "domain_permissions_sync",
  SchedulesRequest: "schedules_request",
  SchedulesResponse: "schedules_response",
  ScheduleAdd: "schedule_add",
  ScheduleUpdate: "schedule_update",
  ScheduleDelete: "schedule_delete",
  PromptLogsRequest: "prompt_logs_request",
  PromptLogsResponse: "prompt_logs_response",
  PromptLogUpdate: "prompt_log_update",
  RunOutcomeUpdate: "run_outcome_update",
  MemoriesRequest: "memories_request",
  MemoriesResponse: "memories_response",
  MemoryDetailRequest: "memory_detail_request",
  MemoryDetailResponse: "memory_detail_response",
  MemoryPageRequest: "memory_page_request",
  MemoryPageResponse: "memory_page_response",
  MemoryChanged: "memory_changed",
  MemoryCreate: "memory_create",
  MemoryUpdate: "memory_update",
  MemoryArchive: "memory_archive",
  EntryCreate: "entry_create",
  EntryUpdate: "entry_update",
  EntryDelete: "entry_delete",
  CollectionTrigger: "collection_trigger",
  CollectionTriggerResult: "collection_trigger_result",
  CursorSet: "cursor_set",
  CursorClear: "cursor_clear",
} as const satisfies Record<string, RuntimeMessageType>;

/** Sidebar → background: user typed a chat message */
export interface RuntimeSendChat {
  type: typeof RuntimeMessageType.SendChat;
  content: string;
  include_page: boolean;
}

/** Background → sidebar: incoming message from Penny */
export interface RuntimeChatMessage {
  type: typeof RuntimeMessageType.ChatMessage;
  content: string;
}

/** Background → sidebar: typing indicator */
export interface RuntimeTyping {
  type: typeof RuntimeMessageType.Typing;
  active: boolean;
  content?: string;
}

/** Background → sidebar: connection state changed */
export interface RuntimeConnectionState {
  type: typeof RuntimeMessageType.ConnectionState;
  state: ConnectionState;
}

/** Background → sidebar: ask user to allow/deny a domain */
export interface RuntimePermissionRequest {
  type: typeof RuntimeMessageType.PermissionRequest;
  request_id: string;
  domain: string;
  url: string;
}

/** Sidebar → background: user's permission decision */
export interface RuntimePermissionResponse {
  type: typeof RuntimeMessageType.PermissionResponse;
  request_id: string;
  allowed: boolean;
}

/** Background → sidebar: dismiss any open permission dialog */
export interface RuntimePermissionDismiss {
  type: typeof RuntimeMessageType.PermissionDismiss;
}

/** Background → sidebar: current page info for the context toggle */
export interface RuntimePageInfo {
  type: typeof RuntimeMessageType.PageInfo;
  title: string;
  url: string;
  favicon: string;
  image: string;      // og:image or similar meta image
  available: boolean;  // false if extraction failed or on a privileged page
}

/** Sidebar → background: request all config params */
export interface RuntimeConfigRequest {
  type: typeof RuntimeMessageType.ConfigRequest;
}

/** Background → sidebar: all config params with current values */
export interface RuntimeConfigResponse {
  type: typeof RuntimeMessageType.ConfigResponse;
  params: RuntimeConfigParam[];
}

/** Sidebar → background: update one config param */
export interface RuntimeConfigUpdate {
  type: typeof RuntimeMessageType.ConfigUpdate;
  key: string;
  value: string;
}

/** Sidebar → background: toggle tool-use capability */
export interface RuntimeToolUseToggle {
  type: typeof RuntimeMessageType.ToolUseToggle;
  enabled: boolean;
}

/** Background → sidebar: current tool-use state */
export interface RuntimeToolUseState {
  type: typeof RuntimeMessageType.ToolUseState;
  enabled: boolean;
}

/** Sidebar → background: add or update a domain permission */
export interface RuntimeDomainUpdate {
  type: typeof RuntimeMessageType.DomainUpdate;
  domain: string;
  permission: DomainPermission;
}

/** Sidebar → background: delete a domain permission */
export interface RuntimeDomainDelete {
  type: typeof RuntimeMessageType.DomainDelete;
  domain: string;
}

/** Background → sidebar: full domain permissions list */
export interface RuntimeDomainPermissionsSync {
  type: typeof RuntimeMessageType.DomainPermissionsSync;
  permissions: DomainPermissionEntry[];
}

/** Sidebar → background: request all schedules */
export interface RuntimeSchedulesRequest {
  type: typeof RuntimeMessageType.SchedulesRequest;
}

/** Background → sidebar: schedules list */
export interface RuntimeSchedulesResponse {
  type: typeof RuntimeMessageType.SchedulesResponse;
  schedules: ScheduleItem[];
  error: string | null;
}

/** Sidebar → background: add a new schedule */
export interface RuntimeScheduleAdd {
  type: typeof RuntimeMessageType.ScheduleAdd;
  command: string;
}

/** Sidebar → background: update a schedule's prompt text */
export interface RuntimeScheduleUpdate {
  type: typeof RuntimeMessageType.ScheduleUpdate;
  schedule_id: number;
  prompt_text: string;
}

/** Sidebar → background: delete a schedule */
export interface RuntimeScheduleDelete {
  type: typeof RuntimeMessageType.ScheduleDelete;
  schedule_id: number;
}

/** Prompts page → background: request prompt logs */
export interface RuntimePromptLogsRequest {
  type: typeof RuntimeMessageType.PromptLogsRequest;
  agent_name?: string;
  offset?: number;
  /** Substring filter over each run's sent messages / response / thinking. */
  query?: string;
}

/** Background → prompts page: prompt logs data */
export interface RuntimePromptLogsResponse {
  type: typeof RuntimeMessageType.PromptLogsResponse;
  runs: PromptLogRun[];
  has_more: boolean;
}

/** Background → prompts page: single prompt logged in real time */
export interface RuntimePromptLogUpdate {
  type: typeof RuntimeMessageType.PromptLogUpdate;
  prompt: PromptLogEntry & { run_id: string };
}

/** Background → prompts page: run outcome set (outcome / reason / target) */
export interface RuntimeRunOutcomeUpdate {
  type: typeof RuntimeMessageType.RunOutcomeUpdate;
  run_id: string;
  outcome: RunOutcome;
  reason: string;
}

/** Memories tab → background: request the memories list */
export interface RuntimeMemoriesRequest {
  type: typeof RuntimeMessageType.MemoriesRequest;
  /** Filter by name / description / intent or matching entry content. */
  query?: string;
}

/** Background → memories tab: memories list */
export interface RuntimeMemoriesResponse {
  type: typeof RuntimeMessageType.MemoriesResponse;
  memories: MemoryRecord[];
}

/** Memories tab → background: drill into one memory */
export interface RuntimeMemoryDetailRequest {
  type: typeof RuntimeMessageType.MemoryDetailRequest;
  name: string;
  /** Active list search — filters the entries section to matching entries. */
  query?: string;
}

/** Background → memories tab: drill-in payload (metadata + first page of
 *  entries + first page of this collection's matching ``collector-runs``
 *  entries — empty for logs).  Each section paginates independently; the
 *  ``*_has_more`` flags drive the per-section "load more" controls. */
export interface RuntimeMemoryDetailResponse {
  type: typeof RuntimeMessageType.MemoryDetailResponse;
  memory: MemoryRecord;
  entries: MemoryEntryRecord[];
  entries_has_more: boolean;
  collector_runs: MemoryEntryRecord[];
  collector_runs_has_more: boolean;
  /** Read positions over the logs this collection reads (empty for logs). */
  cursors: CursorRecord[];
}

/** Memories tab → background: load one more page of a detail section */
export interface RuntimeMemoryPageRequest {
  type: typeof RuntimeMessageType.MemoryPageRequest;
  name: string;
  section: MemorySection;
  offset: number;
  /** Active list search — keeps entry pagination filtered to matches. */
  query?: string;
}

/** Background → memories tab: one more page of a detail section */
export interface RuntimeMemoryPageResponse {
  type: typeof RuntimeMessageType.MemoryPageResponse;
  name: string;
  section: MemorySection;
  entries: MemoryEntryRecord[];
  has_more: boolean;
}

/** Background → memories tab: a memory was mutated, refresh */
export interface RuntimeMemoryChanged {
  type: typeof RuntimeMessageType.MemoryChanged;
  name: string | null;
}

/** Memories tab → background: run a collection's extractor on demand */
export interface RuntimeCollectionTrigger {
  type: typeof RuntimeMessageType.CollectionTrigger;
  name: string;
}

/** Background → memories tab: outcome of an on-demand extractor run */
export interface RuntimeCollectionTriggerResult {
  type: typeof RuntimeMessageType.CollectionTriggerResult;
  name: string;
  success: boolean;
  message: string;
}

/** Memories tab → background: create a new collection */
export interface RuntimeMemoryCreate {
  type: typeof RuntimeMessageType.MemoryCreate;
  name: string;
  description: string;
  intent: string;
  inclusion: "always" | "relevant" | "never";
  recall: "recent" | "relevant" | "all";
  /** Pub/sub: notify the user of new entries (default false). */
  published?: boolean;
  extraction_prompt?: string | null;
  collector_interval_seconds?: number | null;
}

/** Memories tab → background: update collection metadata */
export interface RuntimeMemoryUpdate {
  type: typeof RuntimeMessageType.MemoryUpdate;
  name: string;
  description?: string | null;
  intent?: string | null;
  inclusion?: "always" | "relevant" | "never" | null;
  recall?: "recent" | "relevant" | "all" | null;
  /** Pub/sub: flip notify-on-new; omit/null leaves it unchanged. */
  published?: boolean | null;
  extraction_prompt?: string | null;
  collector_interval_seconds?: number | null;
}

/** Memories tab → background: set a collection's read cursor over one log */
export interface RuntimeCursorSet {
  type: typeof RuntimeMessageType.CursorSet;
  name: string;
  log_name: string;
  /** ISO-8601 datetime to move the cursor to (may be earlier than now). */
  last_read_at: string;
}

/** Memories tab → background: clear a collection's read cursor over one log */
export interface RuntimeCursorClear {
  type: typeof RuntimeMessageType.CursorClear;
  name: string;
  log_name: string;
}

/** Memories tab → background: archive a memory */
export interface RuntimeMemoryArchive {
  type: typeof RuntimeMessageType.MemoryArchive;
  name: string;
}

/** Memories tab → background: add an entry to a collection */
export interface RuntimeEntryCreate {
  type: typeof RuntimeMessageType.EntryCreate;
  memory: string;
  key: string;
  content: string;
}

/** Memories tab → background: edit an entry's content */
export interface RuntimeEntryUpdate {
  type: typeof RuntimeMessageType.EntryUpdate;
  memory: string;
  key: string;
  content: string;
}

/** Memories tab → background: delete an entry */
export interface RuntimeEntryDelete {
  type: typeof RuntimeMessageType.EntryDelete;
  memory: string;
  key: string;
}

export type RuntimeMessage =
  | RuntimeSendChat
  | RuntimeChatMessage
  | RuntimeTyping
  | RuntimeConnectionState
  | RuntimePermissionRequest
  | RuntimePermissionResponse
  | RuntimePermissionDismiss
  | RuntimePageInfo
  | RuntimeConfigRequest
  | RuntimeConfigResponse
  | RuntimeConfigUpdate
  | RuntimeToolUseToggle
  | RuntimeToolUseState
  | RuntimeDomainUpdate
  | RuntimeDomainDelete
  | RuntimeDomainPermissionsSync
  | RuntimeSchedulesRequest
  | RuntimeSchedulesResponse
  | RuntimeScheduleAdd
  | RuntimeScheduleUpdate
  | RuntimeScheduleDelete
  | RuntimePromptLogsRequest
  | RuntimePromptLogsResponse
  | RuntimePromptLogUpdate
  | RuntimeRunOutcomeUpdate
  | RuntimeMemoriesRequest
  | RuntimeMemoriesResponse
  | RuntimeMemoryDetailRequest
  | RuntimeMemoryDetailResponse
  | RuntimeMemoryPageRequest
  | RuntimeMemoryPageResponse
  | RuntimeMemoryChanged
  | RuntimeCollectionTrigger
  | RuntimeCollectionTriggerResult
  | RuntimeMemoryCreate
  | RuntimeMemoryUpdate
  | RuntimeMemoryArchive
  | RuntimeEntryCreate
  | RuntimeEntryUpdate
  | RuntimeEntryDelete
  | RuntimeCursorSet
  | RuntimeCursorClear;

// --- Domain permissions ---

export type DomainPermission = "allowed" | "blocked";
export const DomainPermission = {
  Allowed: "allowed",
  Blocked: "blocked",
} as const satisfies Record<string, DomainPermission>;

/** Map of domain → permission stored in browser.storage.local */
export type DomainAllowlist = Record<string, DomainPermission>;

// --- Page context ---

export interface PageContext {
  title: string;
  url: string;
  text: string;
  image: string;
}

// --- Tool constants ---

export const TAB_LOAD_TIMEOUT_MS = 60_000;

// --- Chat UI ---

export type MessageSender = "user" | "penny";
export const MessageSender = {
  User: "user",
  Penny: "penny",
} as const satisfies Record<string, MessageSender>;

// --- Chat history ---

export interface StoredMessage {
  text: string;
  sender: MessageSender;
}

export const MAX_STORED_MESSAGES = 200;

// --- Storage keys ---

export const STORAGE_KEY_DEVICE_LABEL = "deviceLabel";
export const STORAGE_KEY_CHAT_HISTORY = "chatHistory";
export const STORAGE_KEY_DOMAIN_ALLOWLIST = "domainAllowlist";
export const STORAGE_KEY_TOOL_USE = "toolUseEnabled";

// --- UI constants ---

export const TEXTAREA_LINE_HEIGHT = 20;
export const TEXTAREA_MAX_ROWS = 4;
export const TYPING_INDICATOR_TEXT = "Penny is thinking...";
