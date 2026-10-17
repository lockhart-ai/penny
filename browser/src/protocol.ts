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

// How a browsed page was read (#2388) — mirrors `PageReading` in the Python side's
// `constants.py`.  Set once by the extension (`isLinkIndex`/`richer` in
// `page_text.ts`, the one place that reads the live DOM) and carried unchanged
// through the wire so the server never has to guess it from the text's shape.
export type PageReading = "article" | "link_list";
export const PageReading = {
  Article: "article",
  LinkList: "link_list",
} as const satisfies Record<string, PageReading>;

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
  | "cursor_clear"
  | "run_detail_request";
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
  RunDetailRequest: "run_detail_request",
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
  // The page's own article/link-list mark (#2388) — set once by the extension
  // (`isLinkIndex`/`richer` in `page_text.ts`) and carried unchanged to the
  // extractor's input on the Python side (`PageReading` in `constants.py`).
  // Absent on a tool response that read no page (an error, or a tool other than
  // `browse_url`); the server treats a missing mark as ARTICLE, logged not silent.
  reading?: PageReading;
}

export interface WsOutgoingHeartbeat {
  type: typeof WsOutgoingType.Heartbeat;
}

export interface WsOutgoingCapabilitiesUpdate {
  type: typeof WsOutgoingType.CapabilitiesUpdate;
  tool_use_enabled: boolean;
}

export type WsOutgoing =
  | WsOutgoingMessage
  | WsOutgoingToolResponse
  | WsOutgoingHeartbeat
  | WsOutgoingCapabilitiesUpdate;

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
  | "prompt_logs_response"
  | "prompt_log_update"
  | "run_outcome_update"
  | "memories_response"
  | "memory_detail_response"
  | "memory_page_response"
  | "memory_changed"
  | "collection_trigger_result"
  | "run_detail_response"
  | "memory_save_error";
export const WsIncomingType = {
  Message: "message",
  Typing: "typing",
  Status: "status",
  ToolRequest: "tool_request",
  ConfigResponse: "config_response",
  DomainPermissionsSync: "domain_permissions_sync",
  PermissionPrompt: "permission_prompt",
  PermissionDismiss: "permission_dismiss",
  PromptLogsResponse: "prompt_logs_response",
  PromptLogUpdate: "prompt_log_update",
  RunOutcomeUpdate: "run_outcome_update",
  MemoriesResponse: "memories_response",
  MemoryDetailResponse: "memory_detail_response",
  MemoryPageResponse: "memory_page_response",
  MemoryChanged: "memory_changed",
  CollectionTriggerResult: "collection_trigger_result",
  RunDetailResponse: "run_detail_response",
  MemorySaveError: "memory_save_error",
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

export interface PromptLogEntry {
  id: number;
  // The run this ONE call belongs to — a top-level run's own id for one of its
  // own prompts, or a draw's own (micro-context-minted) id for a draw.  Present
  // on every record (not just a draw's) so a draw's OWN children — a
  // micro-context nested inside another, any depth (#2435) — can be found by
  // matching their `parent_run_id` against THIS id, purely structurally, with
  // no name ever in the loop.
  run_id: string;
  timestamp: string;
  model: string;
  agent_name: string;
  prompt_type: string;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  // The bound collection (collector cycles) / null (chat), stamped at
  // write time so a live run is labelled from its first prompt.
  run_target: string | null;
  messages: Record<string, unknown>[];
  response: Record<string, unknown>;
  thinking: string;
  has_tools: boolean;
  // Whether this call's response actually INVOKED a tool, vs. merely being
  // offered some (has_tools) — the step icon's tool-vs-plain-reply read (#2434).
  made_tool_call: boolean;
  // The enclosing chat/collector run this call was made ON BEHALF OF, when it is
  // a micro-context draw rather than a run's own top-level call; null otherwise.
  parent_run_id: string | null;
  // The specific tool-call id the draw was made for — set only alongside
  // parent_run_id, and only for a draw nested inside a dispatched tool call
  // (e.g. browse's extraction); null for the other micro-context customers,
  // which run on behalf of the run as a whole.
  parent_tool_call_id: string | null;
  // How this CALL ended: "completed" | "cancelled" | "timed_out" | "raised".
  // Null on a row logged before this column existed (read as "completed").
  call_outcome: string | null;
  call_error_class: string | null;
  call_error_message: string | null;
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
  // Structural run-health (the same classifier Penny's quality collector reads)
  // and the concise, copy-pasteable run record.  Present for collector runs.
  health: RunHealth;
  record: string;
  prompts: PromptLogEntry[];
  // The model calls a micro-context made ON BEHALF of this run (#2434) — a flat
  // list in the same per-prompt shape as `prompts`, each entry already carrying
  // its own `parent_tool_call_id` (or null) so the addon nests them itself.
  // Only a FETCHED run carries this; a live single-prompt push builds its own
  // run client-side with none, so it's optional rather than an empty array.
  child_draws?: PromptLogEntry[];
}

/** First-class outcome of a collector cycle (mirrors penny's RunOutcome). */
export type RunOutcome = "failed" | "no_work" | "worked" | "incomplete" | "cancelled";

/** A run-health flag key (mirrors penny's RunHealth.flags). */
export type RunHealthFlag =
  | "no_work_done"
  | "no_writes"
  | "incomplete"
  | "tool_failures"
  | "half_formed_send";

/** Structural failure signals for one collector run (mirrors penny's RunHealth). */
export interface RunHealth {
  bailed: boolean;
  no_writes: boolean;
  incomplete: boolean;
  tool_failures: number;
  degenerate_send: boolean;
  flags: RunHealthFlag[];
  regressive: boolean;
}

export interface WsIncomingPromptLogsPayload {
  type: typeof WsIncomingType.PromptLogsResponse;
  runs: PromptLogRun[];
  has_more: boolean;
}

export interface WsIncomingPromptLogUpdatePayload {
  type: typeof WsIncomingType.PromptLogUpdate;
  prompt: PromptLogEntry;
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
  /** Pub/sub: when true the notifier delivers new entries to the user. */
  published: boolean;
  archived: boolean;
  extraction_prompt: string | null;
  /** The collection's cadence as one RRULE (#1857), or null for a
   * storage-only collection with no schedule. */
  schedule: string | null;
  /** The schedule in plain words, with the exact rule alongside in a code
   * span (#2428's describe_schedule) — null alongside a null schedule. */
  schedule_description: string | null;
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
  // Collector runs render as the prompts tab's run → prompts → turns cards, so
  // each is a full run, not a flattened record.
  collector_runs: PromptLogRun[];
  collector_runs_has_more: boolean;
  cursors: CursorRecord[];
}

export interface WsIncomingMemoryPagePayload {
  type: typeof WsIncomingType.MemoryPageResponse;
  name: string;
  section: MemorySection;
  // Exactly one is populated, keyed by ``section``: entries for "entries",
  // runs for "collector_runs".
  entries: MemoryEntryRecord[];
  runs: PromptLogRun[];
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

/** One run, fully serialized (run → prompts → turns) — the same shape the
 * prompts tab's own list already carries each run in.  A miss is answered,
 * not dropped: `run` is `null` and the requested `run_id` is echoed when no
 * prompt in the ledger carries that id. */
export interface WsIncomingRunDetailPayload {
  type: typeof WsIncomingType.RunDetailResponse;
  run_id: string;
  run: PromptLogRun | null;
}

/** A memory_create/memory_update whose schedule failed server-side
 * validation — delivered back to the requesting connection so a rejected
 * custom RRULE never saves silently (#2442). */
export interface WsIncomingMemorySaveErrorPayload {
  type: typeof WsIncomingType.MemorySaveError;
  action: "create" | "update";
  name: string;
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
  | WsIncomingPromptLogsPayload
  | WsIncomingPromptLogUpdatePayload
  | WsIncomingRunOutcomePayload
  | WsIncomingMemoriesPayload
  | WsIncomingMemoryDetailPayload
  | WsIncomingMemoryPagePayload
  | WsIncomingMemoryChangedPayload
  | WsIncomingCollectionTriggerResultPayload
  | WsIncomingRunDetailPayload
  | WsIncomingMemorySaveErrorPayload;

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
  | "memory_save_error"
  | "cursor_set"
  | "cursor_clear"
  | "run_detail_request"
  | "run_detail_response";

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
  MemorySaveError: "memory_save_error",
  CursorSet: "cursor_set",
  CursorClear: "cursor_clear",
  RunDetailRequest: "run_detail_request",
  RunDetailResponse: "run_detail_response",
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

/** Prompts page → background: request prompt logs */
export interface RuntimePromptLogsRequest {
  type: typeof RuntimeMessageType.PromptLogsRequest;
  agent_name?: string;
  offset?: number;
  /** Substring filter over each run's sent messages / response / thinking. */
  query?: string;
  /** Keep only regressive runs (a bail / incomplete / tool-failure / half-formed send). */
  flagged_only?: boolean;
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
  prompt: PromptLogEntry;
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
  /** Full runs (run → prompts → turns), rendered with the prompts-tab cards. */
  collector_runs: PromptLogRun[];
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

/** Background → memories tab: one more page of a detail section.  Exactly one
 *  of ``entries`` / ``runs`` is populated, keyed by ``section``. */
export interface RuntimeMemoryPageResponse {
  type: typeof RuntimeMessageType.MemoryPageResponse;
  name: string;
  section: MemorySection;
  entries: MemoryEntryRecord[];
  runs: PromptLogRun[];
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
  /** Pub/sub: notify the user of new entries (default false). */
  published?: boolean;
  extraction_prompt?: string | null;
  /** One RRULE (#1857), or null/omitted for a storage-only collection. */
  schedule?: string | null;
}

/** Memories tab → background: update collection metadata */
export interface RuntimeMemoryUpdate {
  type: typeof RuntimeMessageType.MemoryUpdate;
  name: string;
  description?: string | null;
  /** Pub/sub: flip notify-on-new; omit/null leaves it unchanged. */
  published?: boolean | null;
  extraction_prompt?: string | null;
  /** One RRULE (#1857); omitted/null leaves the schedule unchanged. */
  schedule?: string | null;
}

/** Background → memories tab: a memory_create/memory_update's schedule was
 * rejected — the picker's custom box must surface this, never fail quietly. */
export interface RuntimeMemorySaveError {
  type: typeof RuntimeMessageType.MemorySaveError;
  action: "create" | "update";
  name: string;
  message: string;
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

/** Open the run a skill was taught from (or just view a run) */
export interface RuntimeRunDetailRequest {
  type: typeof RuntimeMessageType.RunDetailRequest;
  run_id: string;
}

/** Background → skills tab: one run, fully serialized.  A miss is answered,
 * not dropped: `run` is `null` and `run_id` echoes the request when unknown. */
export interface RuntimeRunDetailResponse {
  type: typeof RuntimeMessageType.RunDetailResponse;
  run_id: string;
  run: PromptLogRun | null;
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
  | RuntimeMemorySaveError
  | RuntimeMemoryCreate
  | RuntimeMemoryUpdate
  | RuntimeMemoryArchive
  | RuntimeEntryCreate
  | RuntimeEntryUpdate
  | RuntimeEntryDelete
  | RuntimeCursorSet
  | RuntimeCursorClear
  | RuntimeRunDetailRequest
  | RuntimeRunDetailResponse;

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
