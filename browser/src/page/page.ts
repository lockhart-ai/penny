/**
 * Full page — consolidates the prompt log, memory explorer, and settings.
 * Tabs: Prompts, Memories, Schedules, Domains, Config.
 */

import {
  type DomainAllowlist,
  DomainPermission as DP,
  type DomainPermissionEntry,
  type MemoryEntryRecord,
  type MemoryRecord,
  type MemorySection,
  type PromptLogEntry,
  type PromptLogRun,
  type RuntimeCollectionTriggerResult,
  type RuntimeConfigParam,
  type RuntimeMemoryDetailResponse,
  type RuntimeMemoryPageResponse,
  type RuntimeMessage,
  RuntimeMessageType,
  type ScheduleItem,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
  STORAGE_KEY_TOOL_USE,
} from "../protocol.js";

// --- Top-level state ---

type Tab =
  | "prompts"
  | "memories"
  | "schedules"
  | "domains"
  | "config";

// --- Toast ---

let toastTimer: ReturnType<typeof setTimeout> | null = null;

function showToast(text: string): void {
  const toast = document.getElementById("toast")!;
  toast.textContent = text;
  toast.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2000);
}

// --- Prompts state ---

const runsContainer = document.getElementById("runs")!;
const promptsLoading = document.getElementById("prompts-loading")!;
const promptsLoadMore = document.getElementById("prompts-load-more")!;
const promptsLoadMoreBtn = document.getElementById("prompts-load-more-btn")!;
let activeAgentFilter = "";

const AGENT_LABELS: Record<string, string> = {
  collector: '<i class="fa-solid fa-database"></i> Collector',
  chat: '<i class="fa-solid fa-comment"></i> Chat',
  history: '<i class="fa-solid fa-clock-rotate-left"></i> History',
  startup: '<i class="fa-solid fa-rocket"></i> Startup',
};

const ACTIVE_TIMEOUT_MS = 60_000;

let allRuns: PromptLogRun[] = [];
let hasMore = false;
const runElements = new Map<string, HTMLElement>();
let activeRunId: string | null = null;
let activeTimer: ReturnType<typeof setTimeout> | null = null;
let promptsLoaded = false;

// --- Memories state ---

const memoriesLoading = document.getElementById("memories-loading")!;
const memoriesList = document.getElementById("memories-list")!;
const memoryDetail = document.getElementById("memory-detail")!;
const memoryDetailContent = document.getElementById("memory-detail-content")!;
const memoryDetailBack = document.getElementById("memory-detail-back")!;

type MemoryTab = "collections" | "logs" | "archived";

let allMemories: MemoryRecord[] = [];
let activeMemoryName: string | null = null;
let activeMemoryTab: MemoryTab = "collections";

// Detail view pagination — each section accumulates pages independently so
// opening a big collection/log never loads its whole history at once.
let activeMemory: MemoryRecord | null = null;
let memoryEntries: MemoryEntryRecord[] = [];
let memoryEntriesHasMore = false;
let memoryRuns: MemoryEntryRecord[] = [];
let memoryRunsHasMore = false;
// Name of the collection whose extractor is currently running on demand
// (drives the "run extractor" button's disabled/spinner state).
let triggeringCollection: string | null = null;

// --- Config state ---

let pendingConfigSave = false;

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
  promptsLoaded = true;

  // Set up all panel interactions
  setupPrompts();
  setupMemories();
  setupSchedules();
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
  if (tab === "prompts" && !promptsLoaded) {
    requestPromptLogs(0);
    promptsLoaded = true;
  } else if (tab === "memories") {
    requestMemories();
  } else if (tab === "schedules") {
    browser.runtime.sendMessage({ type: RuntimeMessageType.SchedulesRequest });
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
    promptsLoaded = true;
    if (message.runs.length > 0 && allRuns.length > 0) {
      appendRuns(message.runs);
    } else {
      allRuns = message.runs;
      renderPrompts();
    }
    hasMore = message.has_more;
    promptsLoadMore.classList.toggle("hidden", !hasMore);
  } else if (message.type === RuntimeMessageType.PromptLogUpdate) {
    handlePromptUpdate(message.prompt);
  } else if (message.type === RuntimeMessageType.RunOutcomeUpdate) {
    handleRunOutcome(
      message.run_id,
      message.success,
      message.reason,
      message.target,
    );
  } else if (message.type === RuntimeMessageType.SchedulesResponse) {
    renderSchedules(message.schedules, message.error);
  } else if (message.type === RuntimeMessageType.ConfigResponse) {
    renderConfig(message.params);
    if (pendingConfigSave) {
      pendingConfigSave = false;
      showToast("Saved");
    }
  } else if (message.type === RuntimeMessageType.ToolUseState) {
    const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement | null;
    if (toggle) toggle.checked = message.enabled;
  } else if (message.type === RuntimeMessageType.DomainPermissionsSync) {
    renderDomains(message.permissions);
  } else if (message.type === RuntimeMessageType.MemoriesResponse) {
    handleMemoriesResponse(message.memories);
  } else if (message.type === RuntimeMessageType.MemoryDetailResponse) {
    handleMemoryDetailResponse(message);
  } else if (message.type === RuntimeMessageType.MemoryPageResponse) {
    handleMemoryPageResponse(message);
  } else if (message.type === RuntimeMessageType.MemoryChanged) {
    handleMemoryChanged(message.name);
  } else if (message.type === RuntimeMessageType.CollectionTriggerResult) {
    handleCollectionTriggerResult(message);
  }
}


// ============================================================
// Prompts
// ============================================================

function setupPrompts(): void {
  for (const btn of Array.from(document.querySelectorAll("#agent-tabs .sub-tab"))) {
    btn.addEventListener("click", () => {
      activeAgentFilter = btn.getAttribute("data-agent") ?? "";
      for (const b of Array.from(document.querySelectorAll("#agent-tabs .sub-tab"))) {
        b.classList.toggle("active", b === btn);
      }
      allRuns = [];
      requestPromptLogs(0);
    });
  }
  promptsLoadMoreBtn.addEventListener("click", () => {
    requestPromptLogs(allRuns.length);
  });
}

function requestPromptLogs(offset: number): void {
  const agentName = activeAgentFilter || undefined;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.PromptLogsRequest,
    agent_name: agentName,
    offset,
  });
}

function handlePromptUpdate(prompt: PromptLogEntry & { run_id: string }): void {
  if (activeAgentFilter && prompt.agent_name !== activeAgentFilter) return;

  const existingRun = allRuns.find((r) => r.run_id === prompt.run_id);
  if (existingRun) {
    updateExistingRun(existingRun, prompt);
  } else {
    insertNewRun(prompt);
  }
}

function updateExistingRun(run: PromptLogRun, prompt: PromptLogEntry): void {
  run.prompts.push(prompt);
  run.prompt_count = run.prompts.length;
  run.ended_at = prompt.timestamp;
  run.total_duration_ms += prompt.duration_ms;
  run.total_input_tokens += prompt.input_tokens;
  run.total_output_tokens += prompt.output_tokens;

  const row = runElements.get(run.run_id);
  if (!row) return;

  const summary = row.querySelector(".run-summary")!;
  const oldHeader = summary.querySelector(".run-header")!;
  const newHeader = createRunHeader(run);
  summary.replaceChild(newHeader, oldHeader);

  const promptsContainer = row.querySelector(".run-prompts")!;
  promptsContainer.appendChild(createPromptRow(prompt, run.prompts.length));

  markRunActive(run.run_id, row);
}

function insertNewRun(prompt: PromptLogEntry & { run_id: string }): void {
  const run: PromptLogRun = {
    run_id: prompt.run_id,
    agent_name: prompt.agent_name,
    prompt_count: 1,
    started_at: prompt.timestamp,
    ended_at: prompt.timestamp,
    total_duration_ms: prompt.duration_ms,
    total_input_tokens: prompt.input_tokens,
    total_output_tokens: prompt.output_tokens,
    run_success: null,
    run_reason: null,
    run_target: null,
    prompts: [prompt],
  };
  allRuns.unshift(run);
  promptsLoading.classList.add("hidden");

  const row = createRunRow(run);
  runsContainer.prepend(row);
  markRunActive(run.run_id, row);
}

function handleRunOutcome(
  runId: string,
  success: boolean,
  reason: string,
  target: string | null,
): void {
  const run = allRuns.find((r) => r.run_id === runId);
  if (!run) return;
  run.run_success = success;
  run.run_reason = reason;
  run.run_target = target;

  const row = runElements.get(runId);
  if (!row) return;

  const summary = row.querySelector(".run-summary");
  if (summary) {
    summary.appendChild(createRunOutcome(success, reason, target));
  }

  // Dismiss spinner — run is complete
  row.classList.remove("active-run");
  if (activeRunId === runId) {
    if (activeTimer) clearTimeout(activeTimer);
    activeRunId = null;
    activeTimer = null;
  }
}

function markRunActive(runId: string, row: HTMLElement): void {
  if (activeRunId && activeRunId !== runId) {
    const previous = runElements.get(activeRunId);
    if (previous) previous.classList.remove("active-run");
  }
  activeRunId = runId;
  row.classList.add("active-run");
  if (activeTimer) clearTimeout(activeTimer);
  activeTimer = setTimeout(() => {
    row.classList.remove("active-run");
    activeRunId = null;
    activeTimer = null;
  }, ACTIVE_TIMEOUT_MS);
}

function renderPrompts(): void {
  promptsLoading.classList.add("hidden");
  runsContainer.innerHTML = "";
  runElements.clear();
  if (activeTimer) clearTimeout(activeTimer);
  activeTimer = null;
  activeRunId = null;

  if (allRuns.length === 0) {
    const label = activeAgentFilter || "any agent";
    promptsLoading.textContent = `No prompt logs for ${label}.`;
    promptsLoading.classList.remove("hidden");
    return;
  }

  for (const run of allRuns) {
    runsContainer.appendChild(createRunRow(run));
  }
}

function appendRuns(newRuns: PromptLogRun[]): void {
  for (const run of newRuns) {
    allRuns.push(run);
    runsContainer.appendChild(createRunRow(run));
  }
}

function createRunRow(run: PromptLogRun): HTMLElement {
  const row = document.createElement("div");
  row.className = "run";
  runElements.set(run.run_id, row);

  const summary = document.createElement("div");
  summary.className = "run-summary";

  const header = createRunHeader(run);
  summary.appendChild(header);

  if (run.run_success !== null || run.run_reason) {
    summary.appendChild(
      createRunOutcome(run.run_success, run.run_reason ?? "", run.run_target),
    );
  }

  row.appendChild(summary);

  const promptsContainer = document.createElement("div");
  promptsContainer.className = "run-prompts";
  for (let i = 0; i < run.prompts.length; i++) {
    promptsContainer.appendChild(createPromptRow(run.prompts[i], i + 1));
  }
  row.appendChild(promptsContainer);

  summary.addEventListener("click", () => {
    row.classList.toggle("expanded");
  });

  return row;
}

function createRunOutcome(
  success: boolean | null,
  reason: string,
  target: string | null,
): HTMLElement {
  const el = document.createElement("div");
  el.className = success
    ? "run-outcome run-outcome-stored"
    : "run-outcome run-outcome-discarded";
  el.textContent = target ? `[${target}] ${reason}` : reason;
  return el;
}

function createRunHeader(run: PromptLogRun): HTMLElement {
  const header = document.createElement("div");
  header.className = "run-header";

  const toggle = document.createElement("span");
  toggle.className = "run-toggle";
  toggle.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
  header.appendChild(toggle);

  const agent = document.createElement("span");
  agent.className = "run-agent";
  agent.innerHTML = AGENT_LABELS[run.agent_name] ?? run.agent_name;
  const spinner = document.createElement("span");
  spinner.className = "run-spinner";
  spinner.innerHTML = ' <i class="fa-solid fa-spinner fa-spin"></i>';
  agent.appendChild(spinner);
  header.appendChild(agent);

  const promptType = extractPromptType(run);
  if (promptType) {
    const typeEl = document.createElement("span");
    typeEl.className = "run-type";
    typeEl.textContent = promptType;
    header.appendChild(typeEl);
  }

  const time = document.createElement("span");
  time.className = "run-time";
  time.textContent = formatDateTime(run.started_at);
  header.appendChild(time);

  const meta = document.createElement("span");
  meta.className = "run-meta";
  const tokPerSec = run.total_duration_ms > 0
    ? ((run.total_output_tokens / run.total_duration_ms) * 1000).toFixed(1)
    : "0";
  meta.innerHTML = `<span><i class="fa-solid fa-layer-group"></i>${run.prompt_count}</span>` +
    `<span><i class="fa-solid fa-arrow-down"></i>${formatTokens(run.total_input_tokens)}</span>` +
    `<span><i class="fa-solid fa-arrow-up"></i>${formatTokens(run.total_output_tokens)}</span>` +
    `<span><i class="fa-solid fa-gauge-high"></i>${tokPerSec} tok/s</span>` +
    `<span><i class="fa-solid fa-clock"></i>${formatDuration(run.total_duration_ms)}</span>`;
  header.appendChild(meta);

  return header;
}

function createPromptRow(prompt: PromptLogEntry, step: number): HTMLElement {
  const row = document.createElement("div");
  row.className = "prompt";

  const header = document.createElement("div");
  header.className = "prompt-header";

  const stepEl = document.createElement("span");
  stepEl.className = "prompt-step";
  stepEl.textContent = String(step);
  header.appendChild(stepEl);

  const iconEl = document.createElement("span");
  iconEl.className = "prompt-tools";
  iconEl.innerHTML = prompt.has_tools
    ? '<i class="fa-solid fa-wrench"></i>'
    : '<i class="fa-solid fa-comment"></i>';
  header.appendChild(iconEl);

  const snippet = extractLastTurnSnippet(prompt);
  if (snippet) {
    const snippetEl = document.createElement("span");
    snippetEl.className = "prompt-snippet";
    snippetEl.textContent = snippet;
    snippetEl.title = snippet;
    header.appendChild(snippetEl);
  }

  const meta = document.createElement("span");
  meta.className = "prompt-meta";
  const promptTokPerSec = prompt.duration_ms > 0
    ? ((prompt.output_tokens / prompt.duration_ms) * 1000).toFixed(1)
    : "0";
  meta.innerHTML =
    `<span></span>` +
    `<span><i class="fa-solid fa-arrow-down"></i>${formatTokens(prompt.input_tokens)}</span>` +
    `<span><i class="fa-solid fa-arrow-up"></i>${formatTokens(prompt.output_tokens)}</span>` +
    `<span><i class="fa-solid fa-gauge-high"></i>${promptTokPerSec} tok/s</span>` +
    `<span><i class="fa-solid fa-clock"></i>${formatDuration(prompt.duration_ms)}</span>`;
  header.appendChild(meta);

  row.appendChild(header);

  const detail = createPromptDetail(prompt);
  row.appendChild(detail);

  header.addEventListener("click", () => {
    row.classList.toggle("expanded");
  });

  return row;
}

function createPromptDetail(prompt: PromptLogEntry): HTMLElement {
  const detail = document.createElement("div");
  detail.className = "prompt-detail";

  for (const message of prompt.messages) {
    const role = String(message.role ?? "unknown");
    const content = extractMessageContent(message);
    detail.appendChild(createPromptSection(role, content));
  }

  if (prompt.thinking) {
    detail.appendChild(createPromptSection("thinking", prompt.thinking));
  }

  detail.appendChild(createPromptSection("response", renderResponse(prompt.response)));

  return detail;
}

function createPromptSection(label: string, content: string): HTMLElement {
  const section = document.createElement("div");
  section.className = "prompt-section";

  const labelEl = document.createElement("div");
  labelEl.className = "prompt-section-label";
  labelEl.dataset.role = label.toLowerCase();
  labelEl.innerHTML = `<i class="fa-solid fa-chevron-right section-toggle-icon"></i> ${label}`;
  section.appendChild(labelEl);

  const contentEl = document.createElement("div");
  contentEl.className = "prompt-section-content";
  contentEl.textContent = content;
  section.appendChild(contentEl);

  labelEl.addEventListener("click", () => {
    section.classList.toggle("expanded");
  });

  return section;
}

function extractMessageContent(message: Record<string, unknown>): string {
  const parts: string[] = [];

  if (typeof message.content === "string" && message.content) {
    parts.push(prettyJson(message.content));
  } else if (Array.isArray(message.content)) {
    const text = message.content.map((part: Record<string, unknown>) => {
      if (part.type === "text") return String(part.text ?? "");
      if (part.type === "image_url") return "[image]";
      return JSON.stringify(part);
    }).join("\n");
    if (text) parts.push(text);
  }

  if (Array.isArray(message.tool_calls)) {
    const calls = message.tool_calls as Record<string, unknown>[];
    for (const call of calls) {
      const fn = call.function as Record<string, unknown> | undefined;
      if (fn) {
        parts.push(`tool_call: ${fn.name}(${prettyJson(String(fn.arguments ?? ""))})`);
      } else {
        parts.push(JSON.stringify(call, null, 2));
      }
    }
  }

  return parts.length > 0 ? parts.join("\n") : prettyJson(JSON.stringify(message.content ?? ""));
}

function renderResponse(response: Record<string, unknown>): string {
  const choices = response.choices as Record<string, unknown>[] | undefined;
  if (!choices || choices.length === 0) {
    return JSON.stringify(response, null, 2);
  }

  const choice = choices[0];
  const message = choice.message as Record<string, unknown> | undefined;
  if (!message) {
    return JSON.stringify(choice, null, 2);
  }

  return extractMessageContent(message);
}


const SNIPPET_MAX_CHARS = 80;

function extractLastTurnSnippet(prompt: PromptLogEntry): string {
  const response = prompt.response as Record<string, unknown>;
  const choices = response.choices as Record<string, unknown>[] | undefined;
  if (!choices || choices.length === 0) return "";
  const message = choices[0].message as Record<string, unknown> | undefined;
  if (!message) return "";

  // Check for tool calls first
  const toolCalls = message.tool_calls as Record<string, unknown>[] | undefined;
  if (toolCalls && toolCalls.length > 0) {
    const names = toolCalls.map((tc) => {
      const fn = tc.function as Record<string, unknown> | undefined;
      return fn?.name ?? "tool";
    });
    const args = toolCalls.map((tc) => {
      const fn = tc.function as Record<string, unknown> | undefined;
      const raw = fn?.arguments;
      if (typeof raw === "string") {
        try {
          const parsed = JSON.parse(raw);
          return parsed.queries ? parsed.queries.join(", ") : raw;
        } catch { return raw; }
      }
      if (typeof raw === "object" && raw !== null) {
        const obj = raw as Record<string, unknown>;
        return obj.queries ? (obj.queries as string[]).join(", ") : JSON.stringify(raw);
      }
      return "";
    });
    return normalizeSnippet(names.map((n, i) => `${n}(${args[i]})`).join(", "));
  }

  return normalizeSnippet(message.content as string | null);
}

function normalizeSnippet(content: string | null | undefined): string {
  if (typeof content !== "string" || content.length === 0) return "";
  const text = content.replace(/\s+/g, " ").trim();
  if (text.length <= SNIPPET_MAX_CHARS) return text;
  return text.slice(0, SNIPPET_MAX_CHARS) + "…";
}

function extractPromptType(run: PromptLogRun): string {
  // For collector cycles, every run has prompt_type="collector" — surface
  // the bound collection name instead, which is the only thing that
  // distinguishes one collector run from another.
  if (run.agent_name === "collector" && run.run_target) {
    return run.run_target;
  }
  for (const prompt of run.prompts) {
    if (!prompt.prompt_type) continue;
    if (prompt.prompt_type === "user_message") {
      const userText = extractLastUserMessage(prompt);
      if (userText) return userText;
    }
    return prompt.prompt_type;
  }
  return "";
}

function extractLastUserMessage(prompt: PromptLogEntry): string {
  for (let i = prompt.messages.length - 1; i >= 0; i--) {
    const message = prompt.messages[i];
    if (message.role !== "user") continue;
    const snippet = normalizeSnippet(message.content as string | null);
    if (snippet) return snippet;
  }
  return "";
}

function formatDateTime(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function prettyJson(value: string): string {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

function formatTokens(count: number): string {
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return String(count);
}

const MS_PER_DAY = 86_400_000;

function formatRelativeDate(iso: string): string {
  try {
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return iso;
    const days = Math.floor((Date.now() - then) / MS_PER_DAY);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;
    return formatDateTime(iso);
  } catch {
    return iso;
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = (ms / 1000).toFixed(1);
  return `${seconds}s`;
}

/** Convert literal ``\n`` escape sequences to real newlines.  Used on
 * extraction prompts before rendering them in the textarea — chat-side
 * tool calls (or the model double-escaping) have stored a few prompts
 * with the two-character escape instead of an actual newline. */
function unescapeNewlines(text: string): string {
  return text.replace(/\\n/g, "\n");
}
// ============================================================
// Schedules
// ============================================================

function setScheduleAddEnabled(enabled: boolean): void {
  const input = document.getElementById("schedules-input") as HTMLInputElement | null;
  const btn = document.getElementById("schedules-add-btn") as HTMLButtonElement | null;
  if (input) input.disabled = !enabled;
  if (btn) btn.disabled = !enabled;
}

function renderSchedules(schedules: ScheduleItem[], error: string | null): void {
  const listEl = document.getElementById("schedules-list")!;
  listEl.innerHTML = "";
  setScheduleAddEnabled(true);

  if (error) {
    const errEl = document.createElement("div");
    errEl.className = "schedule-error";
    errEl.textContent = error;
    listEl.appendChild(errEl);
  }

  if (schedules.length === 0 && !error) {
    const empty = document.createElement("div");
    empty.className = "schedules-empty";
    empty.textContent = "No scheduled tasks yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const schedule of schedules) {
    listEl.appendChild(createScheduleRow(schedule));
  }
}

function createScheduleRow(schedule: ScheduleItem): HTMLElement {
  const row = document.createElement("div");
  row.className = "schedule-row";

  const header = document.createElement("div");
  header.className = "schedule-header";

  const timing = document.createElement("span");
  timing.className = "schedule-timing";
  timing.textContent = schedule.timing_description;

  const prompt = document.createElement("span");
  prompt.className = "schedule-prompt";
  prompt.textContent = schedule.prompt_text;

  const cron = document.createElement("span");
  cron.className = "schedule-cron-inline";
  cron.textContent = schedule.cron_expression;

  const del = document.createElement("button");
  del.className = "schedule-delete";
  del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
  del.setAttribute("aria-label", `Delete schedule: ${schedule.prompt_text}`);
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    browser.runtime.sendMessage({
      type: RuntimeMessageType.ScheduleDelete,
      schedule_id: schedule.id,
    });
  });

  header.appendChild(timing);
  header.appendChild(prompt);
  header.appendChild(cron);
  header.appendChild(del);

  const detail = document.createElement("div");
  detail.className = "schedule-detail";

  const editInput = document.createElement("textarea");
  editInput.className = "schedule-edit-input";
  editInput.value = schedule.prompt_text;
  editInput.rows = 2;

  const saveBtn = document.createElement("button");
  saveBtn.className = "schedule-save";
  saveBtn.textContent = "Save";
  saveBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const newText = editInput.value.trim();
    if (newText && newText !== schedule.prompt_text) {
      browser.runtime.sendMessage({
        type: RuntimeMessageType.ScheduleUpdate,
        schedule_id: schedule.id,
        prompt_text: newText,
      });
    }
  });

  detail.appendChild(editInput);
  detail.appendChild(saveBtn);

  row.appendChild(header);
  row.appendChild(detail);

  header.addEventListener("click", () => {
    row.classList.toggle("expanded");
  });

  return row;
}

function createSkeletonRow(): HTMLElement {
  const row = document.createElement("div");
  row.className = "schedule-row schedule-skeleton";

  const header = document.createElement("div");
  header.className = "schedule-header";

  const timing = document.createElement("span");
  timing.className = "skeleton-block skeleton-timing";

  const prompt = document.createElement("span");
  prompt.className = "skeleton-block skeleton-prompt";

  header.appendChild(timing);
  header.appendChild(prompt);
  row.appendChild(header);
  return row;
}

function setupSchedules(): void {
  const input = document.getElementById("schedules-input") as HTMLInputElement;
  const btn = document.getElementById("schedules-add-btn")!;

  function add(): void {
    const command = input.value.trim();
    if (!command) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.ScheduleAdd, command });
    input.value = "";
    setScheduleAddEnabled(false);
    showToast(`Adding schedule: ${command}`);

    const listEl = document.getElementById("schedules-list")!;
    const empty = listEl.querySelector(".schedules-empty");
    if (empty) empty.remove();
    listEl.appendChild(createSkeletonRow());
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// ============================================================
// Domains
// ============================================================

async function loadDomainsFromCache(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  const permissions = Object.entries(allowlist).map(([domain, permission]) => ({ domain, permission }));
  renderDomains(permissions);
}

function renderDomains(permissions: DomainPermissionEntry[]): void {
  const listEl = document.getElementById("domains-list")!;
  listEl.innerHTML = "";

  const sorted = [...permissions].sort((a, b) => a.domain.localeCompare(b.domain));

  if (sorted.length === 0) {
    const empty = document.createElement("div");
    empty.className = "prefs-empty";
    empty.textContent = "No domains saved yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const { domain, permission } of sorted) {
    const row = document.createElement("div");
    row.className = "domain-row";

    const name = document.createElement("span");
    name.className = "domain-name";
    name.textContent = domain;

    const status = document.createElement("button");
    status.className = `domain-status ${permission}`;
    status.textContent = permission === DP.Allowed ? "Allowed" : "Blocked";
    status.title = "Click to toggle";
    status.addEventListener("click", () => {
      const next = permission === DP.Allowed ? DP.Blocked : DP.Allowed;
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainUpdate, domain, permission: next });
    });

    const del = document.createElement("button");
    del.className = "pref-delete";
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.setAttribute("aria-label", `Remove ${domain}`);
    del.addEventListener("click", () => {
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainDelete, domain });
    });

    row.appendChild(name);
    row.appendChild(status);
    row.appendChild(del);
    listEl.appendChild(row);
  }
}

function setupDomains(): void {
  const input = document.getElementById("domains-input") as HTMLInputElement;
  const select = document.getElementById("domains-permission") as HTMLSelectElement;
  const btn = document.getElementById("domains-add-btn")!;

  function add(): void {
    const raw = input.value.trim().toLowerCase();
    if (!raw) return;
    const domain = raw.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
    if (!domain) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.DomainUpdate,
      domain,
      permission: select.value,
    });
    input.value = "";
    const label = select.value === "allowed" ? "Allowed" : "Blocked";
    showToast(`${label}: ${domain}`);
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}

// ============================================================
// Config
// ============================================================

async function loadToolUseState(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
  const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
  const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement | null;
  if (toggle) toggle.checked = enabled;
}

function setupConfig(): void {
  const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement;
  toggle.addEventListener("change", () => {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ToolUseToggle, enabled: toggle.checked });
  });
}

function renderConfig(params: RuntimeConfigParam[]): void {
  const panel = document.getElementById("config-list")!;
  panel.innerHTML = "";

  const groups = new Map<string, RuntimeConfigParam[]>();
  for (const param of params) {
    if (!groups.has(param.group)) groups.set(param.group, []);
    groups.get(param.group)!.push(param);
  }

  for (const [group, groupParams] of groups) {
    const groupEl = document.createElement("div");
    groupEl.className = "config-group";

    const title = document.createElement("div");
    title.className = "config-group-title";
    title.textContent = group;
    groupEl.appendChild(title);

    for (const param of groupParams) {
      groupEl.appendChild(createConfigItem(param));
    }
    panel.appendChild(groupEl);
  }
}

function createConfigItem(param: RuntimeConfigParam): HTMLElement {
  const item = document.createElement("div");
  item.className = "config-item";

  const header = document.createElement("div");
  header.className = "config-header";

  const label = document.createElement("label");
  label.className = "config-label";
  label.textContent = param.description;
  label.htmlFor = `config-${param.key}`;

  const key = document.createElement("span");
  key.className = "config-key";
  key.textContent = param.key;

  const defaultVal = document.createElement("span");
  defaultVal.className = "config-default";
  defaultVal.textContent = `default: ${param.default}`;

  header.appendChild(label);
  header.appendChild(key);
  header.appendChild(defaultVal);

  const input = document.createElement("input");
  input.id = `config-${param.key}`;
  input.className = "config-input";
  input.type = param.type === "str" ? "text" : "number";
  if (param.type === "int") input.step = "1";
  if (param.type === "float") input.step = "any";
  input.value = param.value;
  input.placeholder = param.default;
  if (param.value !== param.default) input.classList.add("modified");

  input.addEventListener("change", () => {
    pendingConfigSave = true;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.ConfigUpdate,
      key: param.key,
      value: input.value,
    });
  });

  item.appendChild(header);
  item.appendChild(input);
  return item;
}

// ============================================================
// Memories
// ============================================================

function setupMemories(): void {
  memoryDetailBack.addEventListener("click", showMemoriesList);
  for (const btn of Array.from(document.querySelectorAll("#memory-tabs .sub-tab"))) {
    btn.addEventListener("click", () => {
      const tab = btn.getAttribute("data-mtab") as MemoryTab | null;
      if (!tab) return;
      activeMemoryTab = tab;
      for (const b of Array.from(document.querySelectorAll("#memory-tabs .sub-tab"))) {
        b.classList.toggle("active", b === btn);
      }
      // Sub-tab switch returns to the list view.
      activeMemoryName = null;
      memoryDetail.classList.add("hidden");
      memoriesList.classList.remove("hidden");
      renderMemoriesList();
    });
  }
}

function requestMemories(): void {
  memoriesLoading.classList.remove("hidden");
  browser.runtime.sendMessage({ type: RuntimeMessageType.MemoriesRequest });
}

function handleMemoriesResponse(memories: MemoryRecord[]): void {
  allMemories = memories;
  memoriesLoading.classList.add("hidden");
  renderMemoriesList();
}

function handleMemoryDetailResponse(message: RuntimeMemoryDetailResponse): void {
  activeMemoryName = message.memory.name;
  activeMemory = message.memory;
  memoryEntries = message.entries;
  memoryEntriesHasMore = message.entries_has_more;
  memoryRuns = message.collector_runs;
  memoryRunsHasMore = message.collector_runs_has_more;
  showMemoryDetail();
  renderMemoryDetail();
}

function handleMemoryPageResponse(message: RuntimeMemoryPageResponse): void {
  // Drop pages for a memory the user already navigated away from.
  if (!activeMemory || message.name !== activeMemory.name) return;
  if (message.section === "collector_runs") {
    memoryRuns = memoryRuns.concat(message.entries);
    memoryRunsHasMore = message.has_more;
  } else {
    memoryEntries = memoryEntries.concat(message.entries);
    memoryEntriesHasMore = message.has_more;
  }
  renderMemoryDetail();
}

function requestMemoryPage(section: MemorySection, offset: number): void {
  if (!activeMemory) return;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.MemoryPageRequest,
    name: activeMemory.name,
    section,
    offset,
  });
}

function handleMemoryChanged(name: string | null): void {
  // The memories tab might not be visible — refresh data only if it is.
  const memoriesPanel = document.getElementById("panel-memories");
  if (!memoriesPanel || memoriesPanel.classList.contains("hidden")) return;
  if (activeMemoryName && (name === null || name === activeMemoryName)) {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: activeMemoryName,
    });
  } else if (!activeMemoryName) {
    requestMemories();
  }
}

function showMemoriesList(): void {
  activeMemoryName = null;
  memoryDetail.classList.add("hidden");
  memoriesList.classList.remove("hidden");
  requestMemories();
}

function showMemoryDetail(): void {
  memoriesList.classList.add("hidden");
  memoryDetail.classList.remove("hidden");
}

function renderMemoriesList(): void {
  memoriesList.innerHTML = "";
  // The "+" affordance only makes sense on the collections tab — that's
  // the only shape users can create from the addon.
  if (activeMemoryTab === "collections") {
    memoriesList.appendChild(createNewMemoryControl());
  }
  const visible = allMemories.filter(memoryMatchesTab);
  if (visible.length === 0) {
    const empty = document.createElement("div");
    empty.className = "panel-loading";
    empty.textContent = emptyLabel(activeMemoryTab);
    memoriesList.appendChild(empty);
    return;
  }
  for (const memory of visible) {
    memoriesList.appendChild(createMemoryRow(memory));
  }
}

function memoryMatchesTab(memory: MemoryRecord): boolean {
  if (activeMemoryTab === "archived") return memory.archived;
  if (memory.archived) return false;
  return activeMemoryTab === "collections" ? memory.type === "collection" : memory.type === "log";
}

function emptyLabel(tab: MemoryTab): string {
  if (tab === "collections") return "No collections yet.";
  if (tab === "logs") return "No logs yet.";
  return "Nothing archived.";
}

function createNewMemoryControl(): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-new-control";

  const button = document.createElement("button");
  button.className = "memory-new-btn";
  button.innerHTML = '<i class="fa-solid fa-plus"></i> New collection';
  button.addEventListener("click", () => {
    wrapper.replaceWith(createNewMemoryForm());
  });
  wrapper.appendChild(button);
  return wrapper;
}

function createNewMemoryForm(): HTMLElement {
  const form = document.createElement("div");
  form.className = "memory-new-form";

  const fields = createMemoryFormFields({
    description: "",
    recall: "off",
    extraction_prompt: "",
    collector_interval_seconds: null,
  });
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "collection-name";
  nameInput.className = "memory-form-input";

  form.appendChild(labelled("Name", nameInput));
  form.appendChild(labelled("Description", fields.description));
  form.appendChild(labelled("Recall", fields.recall));
  form.appendChild(labelled("Extraction prompt", fields.extractionPrompt));
  form.appendChild(labelled("Collector interval (seconds)", fields.intervalInput));

  const actions = document.createElement("div");
  actions.className = "memory-form-actions";

  const cancel = document.createElement("button");
  cancel.className = "memory-form-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => renderMemoriesList());

  const create = document.createElement("button");
  create.className = "memory-form-save";
  create.textContent = "Create";
  create.addEventListener("click", () => {
    const name = nameInput.value.trim();
    if (!name) {
      showToast("Name is required");
      return;
    }
    const promptValue = fields.extractionPrompt.value.trim();
    const intervalValue = fields.intervalInput.value.trim();
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryCreate,
      name,
      description: fields.description.value.trim(),
      recall: fields.recall.value as "off" | "recent" | "relevant" | "all",
      extraction_prompt: promptValue || null,
      collector_interval_seconds: intervalValue ? Number(intervalValue) : null,
    });
    showToast("Created");
  });

  actions.appendChild(cancel);
  actions.appendChild(create);
  form.appendChild(actions);
  return form;
}

interface MemoryFormFields {
  description: HTMLTextAreaElement;
  recall: HTMLSelectElement;
  extractionPrompt: HTMLTextAreaElement;
  intervalInput: HTMLInputElement;
}

function createMemoryFormFields(initial: {
  description: string;
  recall: string;
  extraction_prompt: string;
  collector_interval_seconds: number | null;
}): MemoryFormFields {
  const description = document.createElement("textarea");
  description.className = "memory-form-input";
  description.rows = 2;
  description.value = initial.description;

  const recall = document.createElement("select");
  recall.className = "memory-form-input";
  for (const value of ["off", "recent", "relevant", "all"]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === initial.recall) option.selected = true;
    recall.appendChild(option);
  }

  const extractionPrompt = document.createElement("textarea");
  extractionPrompt.className = "memory-form-input memory-form-prompt";
  extractionPrompt.rows = 6;
  // Some prompts have been stored with literal "\n" escape sequences
  // instead of real newlines (chat-side tool calls or the model itself
  // double-escaped).  Render those as real newlines so the textarea
  // shows a multi-line prompt; saving from this form normalises the DB
  // value naturally.
  extractionPrompt.value = unescapeNewlines(initial.extraction_prompt);

  const intervalInput = document.createElement("input");
  intervalInput.type = "number";
  intervalInput.className = "memory-form-input";
  intervalInput.min = "30";
  intervalInput.placeholder = "300";
  if (initial.collector_interval_seconds !== null) {
    intervalInput.value = String(initial.collector_interval_seconds);
  }

  return { description, recall, extractionPrompt, intervalInput };
}

function labelled(label: string, control: HTMLElement): HTMLElement {
  const wrapper = document.createElement("label");
  wrapper.className = "memory-form-row";
  const text = document.createElement("span");
  text.className = "memory-form-label";
  text.textContent = label;
  wrapper.appendChild(text);
  wrapper.appendChild(control);
  return wrapper;
}

function createMemoryRow(memory: MemoryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "memory-row";
  if (memory.archived) row.classList.add("memory-row-archived");

  const name = document.createElement("span");
  name.className = "memory-name";
  name.textContent = memory.name;

  const badge = document.createElement("span");
  badge.className = `memory-type-badge ${memory.type}`;
  badge.textContent = memory.type;

  const description = document.createElement("span");
  description.className = "memory-description";
  description.textContent = memory.description;

  const meta = document.createElement("span");
  meta.className = "memory-meta";
  meta.appendChild(metaItem("fa-list", `${memory.entry_count} entries`));
  if (memory.last_collected_at) {
    meta.appendChild(metaItem("fa-clock-rotate-left", formatRelativeDate(memory.last_collected_at)));
  } else if (memory.extraction_prompt) {
    meta.appendChild(metaItem("fa-clock-rotate-left", "never"));
  }

  row.appendChild(name);
  row.appendChild(badge);
  row.appendChild(description);
  row.appendChild(meta);

  row.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: memory.name,
    });
  });

  return row;
}

function metaItem(iconClass: string, text: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.innerHTML = `<i class="fa-solid ${iconClass}"></i>${text}`;
  return span;
}

function renderMemoryDetail(): void {
  if (!activeMemory) return;
  const memory = activeMemory;
  memoryDetailContent.innerHTML = "";

  memoryDetailContent.appendChild(createMemoryHeader(memory));
  memoryDetailContent.appendChild(createMemoryMetadataSection(memory));
  memoryDetailContent.appendChild(
    createMemoryEntriesSection(memory, memoryEntries, memoryEntriesHasMore),
  );
  // Collector activity is per-collection — empty for logs (they aren't
  // driven by a collector cycle).
  if (memoryRuns.length > 0) {
    memoryDetailContent.appendChild(
      createCollectorRunsSection(memory, memoryRuns, memoryRunsHasMore),
    );
  }
}

// "Load more" affordance shared by the detail view's paginated sections.
function createLoadMoreButton(onClick: () => void): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-load-more";
  const button = document.createElement("button");
  button.className = "load-more-btn";
  button.innerHTML = '<i class="fa-solid fa-chevron-down"></i> Load more';
  button.addEventListener("click", onClick);
  wrapper.appendChild(button);
  return wrapper;
}

function createCollectorRunsSection(
  memory: MemoryRecord,
  runs: MemoryEntryRecord[],
  hasMore: boolean,
): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-entries-section";

  const title = document.createElement("h3");
  title.textContent = hasMore
    ? `Collector activity (showing ${runs.length}, newest first)`
    : `Collector activity (${runs.length})`;
  section.appendChild(title);

  for (const run of runs) {
    section.appendChild(createCollectorRunEntry(memory, run));
  }
  if (hasMore) {
    section.appendChild(
      createLoadMoreButton(() => requestMemoryPage("collector_runs", runs.length)),
    );
  }
  return section;
}

function createCollectorRunEntry(memory: MemoryRecord, run: MemoryEntryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "memory-entry";

  // Strip the ``[<collection>] `` prefix the Collector writes — the
  // section is already scoped to one target so the prefix is redundant
  // noise on every line.
  const prefix = `[${memory.name}] `;
  const body = run.content.startsWith(prefix) ? run.content.slice(prefix.length) : run.content;

  const header = document.createElement("div");
  header.className = "memory-entry-header";

  const time = document.createElement("span");
  time.className = "memory-entry-date memory-entry-date-primary";
  time.textContent = formatDateTime(run.created_at);
  header.appendChild(time);

  const author = document.createElement("span");
  author.className = "memory-entry-author";
  author.textContent = run.author;
  header.appendChild(author);

  const content = document.createElement("div");
  content.className = "memory-entry-content";
  content.textContent = body;

  row.appendChild(header);
  row.appendChild(content);
  return row;
}

function createMemoryHeader(memory: MemoryRecord): HTMLElement {
  const header = document.createElement("div");
  header.className = "memory-detail-header";

  const title = document.createElement("h2");
  title.textContent = memory.name;

  const badge = document.createElement("span");
  badge.className = `memory-type-badge ${memory.type}`;
  badge.textContent = memory.type;

  header.appendChild(title);
  header.appendChild(badge);
  if (memory.archived) {
    const archived = document.createElement("span");
    archived.className = "memory-type-badge";
    archived.textContent = "archived";
    header.appendChild(archived);
  }
  return header;
}

function createMemoryMetadataSection(memory: MemoryRecord): HTMLElement {
  // Logs are system-managed (created by migrations, written by agents) — read-only.
  // Collections are user-editable.
  if (memory.type === "log") {
    return createLogMetadataSection(memory);
  }
  return createCollectionMetadataSection(memory);
}

function createLogMetadataSection(memory: MemoryRecord): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-detail-section";

  const title = document.createElement("h3");
  title.textContent = "Metadata";
  section.appendChild(title);

  const grid = document.createElement("dl");
  grid.className = "memory-detail-grid";
  appendDef(grid, "Description", memory.description || "—");
  appendDef(grid, "Recall", memory.recall);
  appendDef(grid, "Entries", String(memory.entry_count));
  section.appendChild(grid);
  return section;
}

function createCollectionMetadataSection(memory: MemoryRecord): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-detail-section";

  const title = document.createElement("h3");
  title.textContent = "Metadata";
  section.appendChild(title);

  const fields = createMemoryFormFields({
    description: memory.description,
    recall: memory.recall,
    extraction_prompt: memory.extraction_prompt ?? "",
    collector_interval_seconds: memory.collector_interval_seconds,
  });

  section.appendChild(labelled("Description", fields.description));
  section.appendChild(labelled("Recall", fields.recall));
  section.appendChild(labelled("Extraction prompt", fields.extractionPrompt));
  section.appendChild(labelled("Collector interval (seconds)", fields.intervalInput));

  const readOnlyGrid = document.createElement("dl");
  readOnlyGrid.className = "memory-detail-grid";
  appendDef(readOnlyGrid, "Entries", String(memory.entry_count));
  if (memory.last_collected_at) {
    appendDef(readOnlyGrid, "Last collected", formatDateTime(memory.last_collected_at));
  }
  section.appendChild(readOnlyGrid);

  const actions = document.createElement("div");
  actions.className = "memory-form-actions";

  // Only collections with an extraction prompt have a collector to run.
  if (memory.extraction_prompt) {
    actions.appendChild(createRunExtractorButton(memory));
  }

  const archive = document.createElement("button");
  archive.className = "memory-form-archive";
  archive.textContent = memory.archived ? "Archived" : "Archive";
  archive.disabled = memory.archived;
  archive.addEventListener("click", () => {
    if (!confirm(`Archive "${memory.name}"? It will disappear from the active list.`)) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.MemoryArchive, name: memory.name });
    showToast("Archived");
    showMemoriesList();
  });

  const save = document.createElement("button");
  save.className = "memory-form-save";
  save.textContent = "Save";
  save.addEventListener("click", () => {
    const intervalValue = fields.intervalInput.value.trim();
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryUpdate,
      name: memory.name,
      description: fields.description.value.trim(),
      recall: fields.recall.value as "off" | "recent" | "relevant" | "all",
      extraction_prompt: fields.extractionPrompt.value.trim() || null,
      collector_interval_seconds: intervalValue ? Number(intervalValue) : null,
    });
    showToast("Saved");
  });

  actions.appendChild(archive);
  actions.appendChild(save);
  section.appendChild(actions);
  return section;
}

function createRunExtractorButton(memory: MemoryRecord): HTMLElement {
  const running = triggeringCollection === memory.name;
  const button = document.createElement("button");
  button.className = "memory-form-run";
  button.disabled = running;
  button.innerHTML = running
    ? '<i class="fa-solid fa-spinner fa-spin"></i> Running…'
    : '<i class="fa-solid fa-play"></i> Run extractor';
  button.addEventListener("click", () => {
    triggeringCollection = memory.name;
    browser.runtime.sendMessage({ type: RuntimeMessageType.CollectionTrigger, name: memory.name });
    renderMemoryDetail(); // reflect the running state immediately
  });
  return button;
}

function handleCollectionTriggerResult(message: RuntimeCollectionTriggerResult): void {
  if (triggeringCollection === message.name) triggeringCollection = null;
  showToast(message.success ? "Extractor finished" : `Extractor failed: ${message.message}`);
  // Refresh the detail so re-enabled button, new entries, and updated
  // "last collected" all reflect the run (also arrives via memory_changed).
  if (activeMemory && activeMemory.name === message.name) renderMemoryDetail();
}

function appendDef(grid: HTMLElement, label: string, value: string, monospace = false): void {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  if (monospace) {
    const pre = document.createElement("pre");
    pre.textContent = value;
    dd.appendChild(pre);
  } else {
    dd.textContent = value;
  }
  grid.appendChild(dt);
  grid.appendChild(dd);
}

function createMemoryEntriesSection(
  memory: MemoryRecord,
  entries: MemoryEntryRecord[],
  hasMore: boolean,
): HTMLElement {
  // The entries section is "flat" — entries themselves are the cards,
  // so wrapping them in another card creates visual nesting noise.
  const section = document.createElement("div");
  section.className = "memory-entries-section";

  const title = document.createElement("h3");
  const shown = entries.length;
  const total = memory.entry_count;
  title.textContent =
    total > shown ? `Entries (showing ${shown} of ${total}, newest first)` : `Entries (${total})`;
  section.appendChild(title);

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "memory-entries-empty";
    empty.textContent = "No entries yet.";
    section.appendChild(empty);
  } else {
    for (const entry of entries) {
      section.appendChild(createMemoryEntry(memory, entry));
    }
  }

  if (hasMore) {
    section.appendChild(createLoadMoreButton(() => requestMemoryPage("entries", entries.length)));
  }

  // Logs are append-only by the system — manual entry add is collection-only.
  if (memory.type === "collection") {
    section.appendChild(createEntryAddForm(memory));
  }
  return section;
}

// Long log entries (esp. ``user-messages`` / ``penny-messages``) get
// collapsed by default so the list stays scannable.  CSS clips after
// ~20 visual lines; the JS heuristic decides whether to clip at all.
const MEMORY_ENTRY_COLLAPSE_LINES = 20;
const MEMORY_ENTRY_COLLAPSE_CHARS = 600;

function createMemoryEntry(memory: MemoryRecord, entry: MemoryEntryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "memory-entry";

  const header = document.createElement("div");
  header.className = "memory-entry-header";

  if (entry.key) {
    // Collections: the key is the title.
    const key = document.createElement("span");
    key.className = "memory-entry-key";
    key.textContent = entry.key;
    header.appendChild(key);
  }

  // For logs the timestamp IS the identifier; rendered prominently
  // either way so the eye lands on it quickly when scanning.
  const time = document.createElement("span");
  time.className = entry.key ? "memory-entry-date" : "memory-entry-date memory-entry-date-primary";
  time.textContent = formatDateTime(entry.created_at);
  header.appendChild(time);

  const author = document.createElement("span");
  author.className = "memory-entry-author";
  author.textContent = entry.author;
  header.appendChild(author);

  // Edit/delete only for collection entries (entry_update / entry_delete are keyed).
  if (memory.type === "collection" && entry.key) {
    header.appendChild(createEntryActions(memory, entry, row));
  }

  const content = document.createElement("div");
  content.className = "memory-entry-content";
  content.textContent = entry.content;

  row.appendChild(header);
  row.appendChild(content);

  if (shouldCollapseEntry(entry.content)) {
    content.classList.add("collapsed");
    row.appendChild(createEntryCollapseToggle(content));
  }

  return row;
}

function shouldCollapseEntry(content: string): boolean {
  return (
    content.split("\n").length > MEMORY_ENTRY_COLLAPSE_LINES ||
    content.length > MEMORY_ENTRY_COLLAPSE_CHARS
  );
}

function createEntryCollapseToggle(content: HTMLElement): HTMLElement {
  const toggle = document.createElement("button");
  toggle.className = "memory-entry-toggle";
  toggle.textContent = "Show more";
  toggle.addEventListener("click", () => {
    const stillCollapsed = content.classList.toggle("collapsed");
    toggle.textContent = stillCollapsed ? "Show more" : "Show less";
  });
  return toggle;
}

function createEntryActions(
  memory: MemoryRecord,
  entry: MemoryEntryRecord,
  row: HTMLElement,
): HTMLElement {
  const actions = document.createElement("span");
  actions.className = "memory-entry-actions";

  const edit = document.createElement("button");
  edit.className = "memory-entry-action";
  edit.title = "Edit";
  edit.innerHTML = '<i class="fa-solid fa-pen"></i>';
  edit.addEventListener("click", (e) => {
    e.stopPropagation();
    enterEditMode(memory, entry, row);
  });

  const del = document.createElement("button");
  del.className = "memory-entry-action";
  del.title = "Delete";
  del.innerHTML = '<i class="fa-solid fa-trash"></i>';
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!entry.key) return;
    if (!confirm(`Delete "${entry.key}"?`)) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.EntryDelete,
      memory: memory.name,
      key: entry.key,
    });
    showToast("Deleted");
  });

  actions.appendChild(edit);
  actions.appendChild(del);
  return actions;
}

function enterEditMode(memory: MemoryRecord, entry: MemoryEntryRecord, row: HTMLElement): void {
  if (!entry.key) return;
  const content = row.querySelector(".memory-entry-content") as HTMLElement | null;
  if (!content) return;

  const textarea = document.createElement("textarea");
  textarea.className = "memory-form-input memory-form-prompt";
  textarea.value = entry.content;
  textarea.rows = Math.max(3, entry.content.split("\n").length);

  const actions = document.createElement("div");
  actions.className = "memory-form-actions";
  const cancel = document.createElement("button");
  cancel.className = "memory-form-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => content.replaceWith(restored));
  const save = document.createElement("button");
  save.className = "memory-form-save";
  save.textContent = "Save";
  save.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.EntryUpdate,
      memory: memory.name,
      key: entry.key as string,
      content: textarea.value,
    });
    showToast("Saved");
  });

  const restored = content.cloneNode(true) as HTMLElement;

  const wrapper = document.createElement("div");
  wrapper.className = "memory-entry-content";
  wrapper.appendChild(textarea);
  actions.appendChild(cancel);
  actions.appendChild(save);
  wrapper.appendChild(actions);

  content.replaceWith(wrapper);
}

function createEntryAddForm(memory: MemoryRecord): HTMLElement {
  const form = document.createElement("div");
  form.className = "memory-entry-add";

  const keyInput = document.createElement("input");
  keyInput.type = "text";
  keyInput.placeholder = "key";
  keyInput.className = "memory-form-input memory-entry-key-input";

  const contentInput = document.createElement("textarea");
  contentInput.placeholder = "content";
  contentInput.className = "memory-form-input";
  contentInput.rows = 2;

  const submit = document.createElement("button");
  submit.className = "memory-form-save";
  submit.innerHTML = '<i class="fa-solid fa-plus"></i> Add entry';
  submit.addEventListener("click", () => {
    const key = keyInput.value.trim();
    const content = contentInput.value.trim();
    if (!key || !content) {
      showToast("Key and content required");
      return;
    }
    browser.runtime.sendMessage({
      type: RuntimeMessageType.EntryCreate,
      memory: memory.name,
      key,
      content,
    });
    keyInput.value = "";
    contentInput.value = "";
    showToast("Added");
  });

  form.appendChild(keyInput);
  form.appendChild(contentInput);
  form.appendChild(submit);
  return form;
}

// ============================================================
// Boot
// ============================================================

init();
