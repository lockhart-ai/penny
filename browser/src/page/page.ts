/**
 * Full page — consolidates the prompt log, memory explorer, and settings.
 * Tabs: Prompts, Memories, Schedules, Domains, Config.
 */

import {
  type CursorRecord,
  type DomainAllowlist,
  DomainPermission as DP,
  type DomainPermissionEntry,
  type MemoryEntryRecord,
  type MemoryRecord,
  type MemorySection,
  type PromptLogEntry,
  type PromptLogRun,
  type RunHealth,
  type RunHealthFlag,
  type RunOutcome,
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
let promptSearch = "";
let flaggedOnly = false;

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
let memorySearch = "";

// Detail view pagination — each section accumulates pages independently so
// opening a big collection/log never loads its whole history at once.
let activeMemory: MemoryRecord | null = null;
let memoryEntries: MemoryEntryRecord[] = [];
let memoryEntriesHasMore = false;
let memoryRuns: PromptLogRun[] = [];
let memoryRunsHasMore = false;
let memoryCursors: CursorRecord[] = [];
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
    handleRunOutcome(message.run_id, message.outcome, message.reason);
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
  // The flagged-only toggle lives in the same row but is NOT an agent filter —
  // exclude it so it doesn't get the agent-switch handler (which would fire a
  // second, unfiltered request and race the flagged one).
  for (const btn of Array.from(
    document.querySelectorAll("#agent-tabs .sub-tab:not(.flagged-toggle)"),
  )) {
    btn.addEventListener("click", () => {
      activeAgentFilter = btn.getAttribute("data-agent") ?? "";
      for (const b of Array.from(
        document.querySelectorAll("#agent-tabs .sub-tab:not(.flagged-toggle)"),
      )) {
        b.classList.toggle("active", b === btn);
      }
      allRuns = [];
      requestPromptLogs(0);
    });
  }
  promptsLoadMoreBtn.addEventListener("click", () => {
    requestPromptLogs(allRuns.length);
  });
  const flaggedToggle = document.getElementById("prompts-flagged-toggle");
  flaggedToggle?.addEventListener("click", () => {
    flaggedOnly = !flaggedOnly;
    flaggedToggle.classList.toggle("active", flaggedOnly);
    allRuns = [];
    requestPromptLogs(0);
  });
  const search = document.getElementById("prompts-search") as HTMLInputElement | null;
  if (search) {
    let timer = 0;
    search.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        promptSearch = search.value.trim();
        // Drop the stale list immediately and show only the loader while the
        // query runs — don't leave the full log visible underneath.
        allRuns = [];
        runsContainer.innerHTML = "";
        runElements.clear();
        promptsLoadMore.classList.add("hidden");
        promptsLoading.textContent = promptSearch ? "Searching…" : "Loading prompt logs…";
        promptsLoading.classList.remove("hidden");
        requestPromptLogs(0);
      }, 250);
    });
  }
  wireSearchClear("prompts-search", "prompts-search-clear");
}

function requestPromptLogs(offset: number): void {
  const agentName = activeAgentFilter || undefined;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.PromptLogsRequest,
    agent_name: agentName,
    offset,
    query: promptSearch || undefined,
    flagged_only: flaggedOnly || undefined,
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

  const promptsPanel = row.querySelector(".run-view-prompts") ?? row.querySelector(".run-prompts")!;
  promptsPanel.appendChild(createPromptRow(prompt, run.prompts.length));

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
    run_outcome: null,
    run_reason: null,
    run_target: prompt.run_target ?? null,
    // A live, mid-flight run has no completed-run health/record yet — they're
    // computed once the run is tagged and re-fetched.  Empty until then.
    health: { bailed: false, no_writes: false, incomplete: false, tool_failures: 0, degenerate_send: false, flags: [], regressive: false },
    record: "",
    prompts: [prompt],
  };
  allRuns.unshift(run);
  promptsLoading.classList.add("hidden");

  const row = createRunRow(run);
  runsContainer.prepend(row);
  markRunActive(run.run_id, row);
}

function handleRunOutcome(runId: string, outcome: RunOutcome, reason: string): void {
  const run = allRuns.find((r) => r.run_id === runId);
  if (!run) return;
  run.run_outcome = outcome;
  run.run_reason = reason;

  const row = runElements.get(runId);
  if (!row) return;

  const summary = row.querySelector(".run-summary");
  if (summary) {
    // run_target is already on the run from its first prompt — stamped at write time.
    summary.appendChild(createRunOutcome(outcome, reason, run.run_target));
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
    promptsLoading.textContent = promptSearch
      ? `No prompts match “${promptSearch}”.`
      : `No prompt logs for ${activeAgentFilter || "any agent"}.`;
    promptsLoading.classList.remove("hidden");
    return;
  }

  if (promptSearch) runsContainer.appendChild(createSearchBanner(promptSearch));
  for (const run of allRuns) {
    runsContainer.appendChild(createRunRow(run));
  }
}

// Firefox doesn't render a native clear (✕) for ``type="search"``, so wire a
// custom one: it shows only when there's text, and clearing dispatches an
// ``input`` event so the existing search handler re-runs (unfiltered).
function wireSearchClear(inputId: string, buttonId: string): void {
  const input = document.getElementById(inputId) as HTMLInputElement | null;
  const button = document.getElementById(buttonId);
  if (!input || !button) return;
  const sync = (): void => {
    button.classList.toggle("visible", input.value.length > 0);
  };
  input.addEventListener("input", sync);
  button.addEventListener("click", () => {
    input.value = "";
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
  sync();
}

// A consistent "Showing matches for X" line above any search-filtered list,
// so it's always obvious the view is filtered rather than complete.
function createSearchBanner(query: string): HTMLElement {
  const banner = document.createElement("div");
  banner.className = "search-banner";
  banner.textContent = `Showing matches for “${query}”`;
  return banner;
}

function appendRuns(newRuns: PromptLogRun[]): void {
  for (const run of newRuns) {
    allRuns.push(run);
    runsContainer.appendChild(createRunRow(run));
  }
}

// The collapsible run card — a clickable `.run-summary` (header + optional
// outcome + health badges) that toggles `.expanded` to reveal the run body.
// Shared by the Prompts tab and the memory Activity tab (run → prompts → turns).
function createRunRow(run: PromptLogRun): HTMLElement {
  const row = document.createElement("div");
  row.className = "run";
  runElements.set(run.run_id, row);

  const summary = document.createElement("div");
  summary.className = "run-summary";
  summary.appendChild(createRunHeader(run));
  if (run.run_outcome !== null || run.run_reason) {
    summary.appendChild(createRunOutcome(run.run_outcome, run.run_reason ?? "", run.run_target));
  }
  const badges = createHealthBadges(run.health);
  if (badges) summary.appendChild(badges);

  row.appendChild(summary);
  row.appendChild(createRunBody(run));
  summary.addEventListener("click", () => row.classList.toggle("expanded"));
  return row;
}

const RUN_OUTCOME_CLASS: Record<RunOutcome, string> = {
  worked: "run-outcome-worked",
  no_work: "run-outcome-no-work",
  failed: "run-outcome-failed",
  incomplete: "run-outcome-incomplete",
  cancelled: "run-outcome-cancelled",
};
const RUN_OUTCOME_LABEL: Record<RunOutcome, string> = {
  worked: "worked",
  no_work: "no work",
  failed: "failed",
  incomplete: "incomplete",
  cancelled: "cancelled",
};

function createRunOutcome(
  outcome: RunOutcome | null,
  reason: string,
  target: string | null,
): HTMLElement {
  const el = document.createElement("div");
  el.className = `run-outcome ${outcome ? RUN_OUTCOME_CLASS[outcome] : "run-outcome-no-work"}`;
  const label = outcome ? RUN_OUTCOME_LABEL[outcome] : "";
  const text = reason ? `${label} — ${reason}` : label;
  el.textContent = target ? `[${target}] ${text}` : text;
  return el;
}

// The expanded body of a run: the interactive turn-based "Prompts" view, plus —
// for a completed collector run — a "Record" tab showing the concise textual
// record, the SAME representation Penny's quality collector reads of this run.
function createRunBody(run: PromptLogRun): HTMLElement {
  const body = document.createElement("div");
  body.className = "run-prompts";

  const prompts = document.createElement("div");
  prompts.className = "run-view run-view-prompts active";
  for (let i = 0; i < run.prompts.length; i++) {
    prompts.appendChild(createPromptRow(run.prompts[i], i + 1));
  }

  // The record is only meaningful once the run is tagged (a completed collector
  // run); a live / chat run has no Penny-facing record, so it shows prompts only.
  if (!run.record || run.run_outcome === null) {
    body.appendChild(prompts);
    return body;
  }

  const record = createRecordView(run.record);
  body.appendChild(createRunViewTabs(prompts, record));
  body.appendChild(prompts);
  body.appendChild(record);
  return body;
}

// A "Prompts | Record" tab bar that swaps which view panel is active.
function createRunViewTabs(promptsPanel: HTMLElement, recordPanel: HTMLElement): HTMLElement {
  const bar = document.createElement("div");
  bar.className = "run-view-tabs";
  const makeTab = (label: string, panel: HTMLElement, active: boolean): HTMLButtonElement => {
    const tab = document.createElement("button");
    tab.className = active ? "sub-tab active" : "sub-tab";
    tab.textContent = label;
    tab.addEventListener("click", () => {
      for (const other of Array.from(bar.querySelectorAll(".sub-tab"))) {
        other.classList.remove("active");
      }
      tab.classList.add("active");
      promptsPanel.classList.toggle("active", panel === promptsPanel);
      recordPanel.classList.toggle("active", panel === recordPanel);
    });
    return tab;
  };
  bar.appendChild(makeTab("Prompts", promptsPanel, true));
  bar.appendChild(makeTab("Record", recordPanel, false));
  return bar;
}

// The concise textual run record, rendered as a syntax-highlighted code listing:
// a light editor background, a line-number gutter, the header + ⚠ flags as
// comment-style lines, and each tool call tokenised like code.  The copy button
// still grabs the raw record verbatim (what Penny's quality collector reads).
// All content goes in via textContent — never innerHTML — since it's model output.
function createRecordView(record: string): HTMLElement {
  const view = document.createElement("div");
  view.className = "run-view run-view-record";
  view.appendChild(createCopyButton(() => record, "Copy the run record"));

  const code = document.createElement("div");
  code.className = "record-code";
  const lines = record.split("\n").filter((line) => line.length > 0);
  lines.forEach((line, index) => code.appendChild(createCodeLine(index + 1, line, index === 0)));
  view.appendChild(code);
  return view;
}

function createCodeLine(lineNumber: number, line: string, isHeader: boolean): HTMLElement {
  const row = document.createElement("div");
  row.className = "record-line";

  const gutter = document.createElement("span");
  gutter.className = "record-ln";
  gutter.textContent = String(lineNumber);
  row.appendChild(gutter);

  const code = document.createElement("span");
  code.className = "record-src";
  if (line.startsWith("⚠")) {
    const capacity = /^⚠\s*(INCOMPLETE|TOOL FAILURES)/.test(line);
    row.classList.add(capacity ? "record-line-capacity" : "record-line-flag");
    code.textContent = line;
  } else if (isHeader) {
    row.classList.add("record-line-header");
    code.textContent = line;
  } else {
    highlightCall(line, code);
  }
  row.appendChild(code);
  return row;
}

// Tokenise one rendered tool call (e.g. ``write(games, "Ark Nova 2")``) and
// append a coloured span per token.  Identifiers are classified by what follows:
// ``name(`` is a function, ``name=`` a keyword arg.
function highlightCall(line: string, into: HTMLElement): void {
  const tokenRe =
    /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|(.)/g;
  type Token = { text: string; type: string };
  const tokens: Token[] = [];
  let match: RegExpExecArray | null;
  while ((match = tokenRe.exec(line)) !== null) {
    if (match[1]) tokens.push({ text: match[1], type: "str" });
    else if (match[2]) tokens.push({ text: match[2], type: "num" });
    else if (match[3]) tokens.push({ text: match[3], type: "ident" });
    else if (match[4]) tokens.push({ text: match[4], type: "space" });
    else tokens.push({ text: match[5], type: "punct" });
  }
  const nextNonSpace = (from: number): string => {
    for (let i = from + 1; i < tokens.length; i++) {
      if (tokens[i].type !== "space") return tokens[i].text;
    }
    return "";
  };
  tokens.forEach((token, index) => {
    if (token.type === "ident") {
      if (token.text === "True" || token.text === "False" || token.text === "None") {
        token.type = "kw";
      } else if (nextNonSpace(index) === "(") token.type = "fn";
      else if (nextNonSpace(index) === "=") token.type = "key";
    }
    const span = document.createElement("span");
    span.className = `tok-${token.type}`;
    span.textContent = token.text;
    into.appendChild(span);
  });
}

const RUN_HEALTH_LABEL: Record<RunHealthFlag, string> = {
  no_work_done: "no work done",
  no_writes: "no writes",
  incomplete: "incomplete",
  tool_failures: "tool failures",
  half_formed_send: "half-formed send",
};

// One chip per set health flag — the compact form of the same RunHealth the
// concise record renders verbatim.  Returns null for a healthy run (no chrome).
function createHealthBadges(health: RunHealth | undefined): HTMLElement | null {
  if (!health || health.flags.length === 0) return null;
  const wrap = document.createElement("div");
  wrap.className = "run-health";
  for (const flag of health.flags) {
    const chip = document.createElement("span");
    chip.className = `run-health-flag run-health-${flag}`;
    const count = flag === "tool_failures" && health.tool_failures > 1 ? ` (${health.tool_failures})` : "";
    chip.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${RUN_HEALTH_LABEL[flag]}${count}`;
    wrap.appendChild(chip);
  }
  return wrap;
}

// A small copy button that copies the JSON returned by ``getText`` to the
// clipboard.  ``getText`` is a thunk so the (sometimes large) payload is only
// serialized on click.  Clicking never toggles the surrounding row (stops
// propagation), and the icon flips to a check briefly on success.
function createCopyButton(getText: () => string, title: string): HTMLButtonElement {
  const copy = document.createElement("button");
  copy.className = "copy-btn";
  copy.title = title;
  copy.innerHTML = '<i class="fa-solid fa-copy"></i>';
  copy.addEventListener("click", (e) => {
    e.stopPropagation();
    void navigator.clipboard.writeText(getText()).then(() => {
      copy.innerHTML = '<i class="fa-solid fa-check"></i>';
      window.setTimeout(() => {
        copy.innerHTML = '<i class="fa-solid fa-copy"></i>';
      }, 1500);
    });
  });
  return copy;
}

const jsonOf = (value: unknown): string => JSON.stringify(value, null, 2);

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

  // Prefer the bound collection a collector run targeted (knowledge,
  // sick-stack, …) over the redundant prompt_type ("collector"), so every
  // collector run is identifiable at a glance even without an outcome line.
  const typeLabel = run.run_target || extractPromptType(run);
  if (typeLabel) {
    const typeEl = document.createElement("span");
    typeEl.className = "run-type";
    typeEl.textContent = typeLabel;
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

  // Copy the whole run — its run_id + every prompt's full JSON (ids included) —
  // so it can be pasted back for a deep look or looked up in the DB by run_id.
  header.appendChild(
    createCopyButton(
      () => jsonOf({ run_id: run.run_id, prompts: run.prompts }),
      `Copy all ${run.prompt_count} prompts as JSON (run_id ${run.run_id})`,
    ),
  );

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

  // Copy just this prompt's full JSON — its ``id`` is the promptlog row id, so
  // it can be looked up directly in the DB.
  header.appendChild(
    createCopyButton(() => jsonOf(prompt), `Copy this prompt as JSON (promptlog id ${prompt.id})`),
  );

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

  // Each turn (one message in this prompt's input) gets a copy button that
  // copies just that turn's JSON, tagged with the promptlog id for lookup.
  for (const message of prompt.messages) {
    const role = String(message.role ?? "unknown");
    const content = extractMessageContent(message);
    detail.appendChild(
      createPromptSection(role, content, () => jsonOf({ promptlog_id: prompt.id, turn: message })),
    );
  }

  if (prompt.thinking) {
    detail.appendChild(
      createPromptSection("thinking", prompt.thinking, () =>
        jsonOf({ promptlog_id: prompt.id, thinking: prompt.thinking }),
      ),
    );
  }

  detail.appendChild(
    createPromptSection("response", renderResponse(prompt.response), () =>
      jsonOf({ promptlog_id: prompt.id, response: prompt.response }),
    ),
  );

  return detail;
}

function createPromptSection(
  label: string,
  content: string,
  copyValue?: () => string,
): HTMLElement {
  const section = document.createElement("div");
  section.className = "prompt-section";

  const labelEl = document.createElement("div");
  labelEl.className = "prompt-section-label";
  labelEl.dataset.role = label.toLowerCase();

  const role = document.createElement("span");
  role.className = "prompt-section-role";
  role.innerHTML = `<i class="fa-solid fa-chevron-right section-toggle-icon"></i> ${label}`;
  labelEl.appendChild(role);

  // A one-line preview of the turn's content, so the whole prompt is scannable
  // while collapsed.  CSS ellipsis trims it to fit — no hardcoded cutoff.  Hidden
  // once the section is expanded (the full content shows below).
  const preview = document.createElement("span");
  preview.className = "prompt-section-preview";
  preview.textContent = sectionPreview(content);
  labelEl.appendChild(preview);

  if (copyValue) {
    labelEl.appendChild(createCopyButton(copyValue, `Copy this ${label} as JSON`));
  }
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

// A single-line preview of a turn's content: whitespace/newlines collapsed to
// spaces; an empty turn shows ``""`` so it's still legible.
function sectionPreview(content: string): string {
  const flat = content.replace(/\s+/g, " ").trim();
  return flat.length > 0 ? flat : '""';
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
  // Collector runs surface their bound collection via run.run_target (stamped
  // at write time, preferred by the caller above) — never reach here with one.
  // The prompt_type fallback below deliberately skips a type that just repeats
  // the agent identity (a collector run's prompt_type IS "collector"): the bold
  // agent label already shows it, so emitting it again as the run-type chip was
  // the bare "collector" with no collection — surface nothing rather than that.
  for (const prompt of run.prompts) {
    if (!prompt.prompt_type) continue;
    if (prompt.prompt_type === run.agent_name) continue;
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
  const search = document.getElementById("memories-search") as HTMLInputElement | null;
  if (search) {
    let timer = 0;
    search.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        // Server-side so the filter can match entry content too, not just the
        // names/descriptions the list view already holds.
        memorySearch = search.value.trim();
        if (activeMemoryName) {
          // A collection is open — re-filter its entries (new query or cleared).
          browser.runtime.sendMessage({
            type: RuntimeMessageType.MemoryDetailRequest,
            name: activeMemoryName,
            query: memorySearch || undefined,
          });
        } else {
          requestMemories();
        }
      }, 250);
    });
  }
  wireSearchClear("memories-search", "memories-search-clear");
}

function requestMemories(): void {
  memoriesLoading.classList.remove("hidden");
  browser.runtime.sendMessage({
    type: RuntimeMessageType.MemoriesRequest,
    query: memorySearch || undefined,
  });
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
  memoryCursors = message.cursors;
  showMemoryDetail();
  renderMemoryDetail();
}

function handleMemoryPageResponse(message: RuntimeMemoryPageResponse): void {
  // Drop pages for a memory the user already navigated away from.
  if (!activeMemory || message.name !== activeMemory.name) return;
  if (message.section === "collector_runs") {
    memoryRuns = memoryRuns.concat(message.runs);
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
    // Keep entry pagination filtered to the active search.
    query: section === "entries" ? memorySearch || undefined : undefined,
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
      query: memorySearch || undefined,
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
  if (memorySearch) memoriesList.appendChild(createSearchBanner(memorySearch));
  const visible = allMemories.filter(memoryMatchesTab);
  if (visible.length === 0) {
    const empty = document.createElement("div");
    empty.className = "panel-loading";
    empty.textContent = memorySearch ? "No memories match." : emptyLabel(activeMemoryTab);
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
    intent: "",
    inclusion: "relevant",
    recall: "relevant",
    published: false,
    extraction_prompt: "",
    collector_interval_seconds: null,
  });
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "collection-name";
  nameInput.className = "memory-form-input";

  form.appendChild(labelled("Name", nameInput));
  form.appendChild(labelled("Description", fields.description));
  form.appendChild(labelled("Intent", fields.intent));
  form.appendChild(labelled("Inclusion", fields.inclusion));
  form.appendChild(labelled("Recall", fields.recall));
  form.appendChild(labelled("Notify on new (published)", fields.published));
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
    const intentValue = fields.intent.value.trim();
    if (!intentValue) {
      showToast("Intent is required — what should this collection do?");
      return;
    }
    const promptValue = fields.extractionPrompt.value.trim();
    const intervalValue = fields.intervalInput.value.trim();
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryCreate,
      name,
      description: fields.description.value.trim(),
      intent: intentValue,
      inclusion: fields.inclusion.value as "always" | "relevant" | "never",
      recall: fields.recall.value as "recent" | "relevant" | "all",
      published: fields.published.checked,
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
  intent: HTMLTextAreaElement;
  inclusion: HTMLSelectElement;
  recall: HTMLSelectElement;
  published: HTMLInputElement;
  extractionPrompt: HTMLTextAreaElement;
  intervalInput: HTMLInputElement;
}

function selectOf(values: string[], selected: string): HTMLSelectElement {
  const select = document.createElement("select");
  select.className = "memory-form-input";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === selected) option.selected = true;
    select.appendChild(option);
  }
  return select;
}

function createMemoryFormFields(initial: {
  description: string;
  intent: string;
  inclusion: string;
  recall: string;
  published: boolean;
  extraction_prompt: string;
  collector_interval_seconds: number | null;
}): MemoryFormFields {
  const description = document.createElement("textarea");
  description.className = "memory-form-input";
  description.rows = 2;
  description.value = initial.description;

  const intent = document.createElement("textarea");
  intent.className = "memory-form-input";
  intent.rows = 2;
  intent.placeholder = "what you asked this collection to do, in your words";
  intent.value = initial.intent;

  // Stage-1 routing and stage-2 entry rendering are independent flags.
  const inclusion = selectOf(["always", "relevant", "never"], initial.inclusion);
  const recall = selectOf(["recent", "relevant", "all"], initial.recall);

  // Pub/sub: when checked, the notifier delivers new entries to the user.
  const published = document.createElement("input");
  published.type = "checkbox";
  published.className = "memory-form-checkbox";
  published.checked = initial.published;

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

  return { description, intent, inclusion, recall, published, extractionPrompt, intervalInput };
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
  if (memory.published) {
    const publishedBadge = document.createElement("span");
    publishedBadge.className = "memory-type-badge published";
    publishedBadge.textContent = "published";
    row.appendChild(publishedBadge);
  }
  row.appendChild(description);
  row.appendChild(meta);

  row.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: memory.name,
      // Carry the active search so the detail view shows only matching entries.
      query: memorySearch || undefined,
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
  memoryDetailContent.appendChild(createMemoryDetailTabs(memory));
}

// Entries / Activity / Config as switchable panels under the header, replacing
// the old single-page vertical stack.  Defaults to Entries.
function createMemoryDetailTabs(memory: MemoryRecord): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-detail-tabs-wrapper";

  const tabs = [
    { label: "Entries", panel: createEntriesPanel(memory) },
    { label: "Activity", panel: createActivityPanel(memory) },
    { label: "Config", panel: createConfigPanel(memory) },
  ];

  const bar = document.createElement("div");
  bar.className = "memory-detail-tabs";
  tabs.forEach((tab, index) => {
    const button = document.createElement("button");
    button.className = index === 0 ? "sub-tab active" : "sub-tab";
    button.textContent = tab.label;
    tab.panel.classList.toggle("active", index === 0);
    button.addEventListener("click", () => activateMemoryDetailTab(bar, tabs, index));
    bar.appendChild(button);
  });

  wrapper.appendChild(bar);
  for (const tab of tabs) wrapper.appendChild(tab.panel);
  return wrapper;
}

function activateMemoryDetailTab(
  bar: HTMLElement,
  tabs: { label: string; panel: HTMLElement }[],
  active: number,
): void {
  const buttons = Array.from(bar.querySelectorAll(".sub-tab"));
  buttons.forEach((button, index) => button.classList.toggle("active", index === active));
  tabs.forEach((tab, index) => tab.panel.classList.toggle("active", index === active));
}

function createMemoryTabPanel(): HTMLElement {
  const panel = document.createElement("div");
  panel.className = "memory-tab-panel";
  return panel;
}

// The collection's stored entries (and, for collections, the add-entry form).
function createEntriesPanel(memory: MemoryRecord): HTMLElement {
  const panel = createMemoryTabPanel();
  panel.appendChild(createMemoryEntriesSection(memory, memoryEntries, memoryEntriesHasMore));
  return panel;
}

// Collector run history plus a collection's read positions over the logs it
// consumes — both empty for logs, which aren't driven by a collector cycle.
function createActivityPanel(memory: MemoryRecord): HTMLElement {
  const panel = createMemoryTabPanel();
  if (memory.type === "collection" && memoryCursors.length > 0) {
    panel.appendChild(createCursorsSection(memory, memoryCursors));
  }
  if (memoryRuns.length > 0) {
    panel.appendChild(createCollectorRunsSection(memoryRuns, memoryRunsHasMore));
  }
  if (!panel.hasChildNodes()) {
    const empty = document.createElement("div");
    empty.className = "memory-entries-empty";
    empty.textContent = "No collector activity yet.";
    panel.appendChild(empty);
  }
  return panel;
}

// Editable configuration (read-only metadata for system-managed logs).
function createConfigPanel(memory: MemoryRecord): HTMLElement {
  const panel = createMemoryTabPanel();
  panel.appendChild(createMemoryMetadataSection(memory));
  return panel;
}

function createCursorsSection(memory: MemoryRecord, cursors: CursorRecord[]): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-detail-section";

  const title = document.createElement("h3");
  title.textContent = "Read cursors";
  section.appendChild(title);

  const hint = document.createElement("p");
  hint.className = "memory-cursor-hint";
  hint.textContent =
    "Where this collection has read up to in each log. Pick an earlier point to " +
    "re-read from there (set it before your data starts to re-read everything); " +
    "Clear starts fresh from the most recent entries.";
  section.appendChild(hint);

  for (const cursor of cursors) {
    section.appendChild(createCursorRow(memory, cursor));
  }
  return section;
}

// datetime-local inputs speak local wall-clock "YYYY-MM-DDTHH:mm"; cursors are
// stored/sent as UTC ISO-8601.  Convert in both directions at the boundary.
function isoToLocalInput(iso: string): string {
  const d = new Date(iso);
  const pad = (n: number): string => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  );
}

function createCursorRow(memory: MemoryRecord, cursor: CursorRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "memory-cursor-row";

  const log = document.createElement("span");
  log.className = "memory-cursor-log";
  log.textContent = cursor.log_name;

  const current = document.createElement("span");
  current.className = "memory-cursor-current";
  current.textContent = `read up to ${formatDateTime(cursor.last_read_at)}`;

  const picker = document.createElement("input");
  picker.type = "datetime-local";
  picker.className = "memory-form-input";
  picker.value = isoToLocalInput(cursor.last_read_at);

  const setBtn = document.createElement("button");
  setBtn.className = "memory-form-save";
  setBtn.textContent = "Set";
  setBtn.addEventListener("click", () => {
    if (!picker.value) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.CursorSet,
      name: memory.name,
      log_name: cursor.log_name,
      last_read_at: new Date(picker.value).toISOString(),
    });
    showToast("Cursor set");
  });

  const clearBtn = document.createElement("button");
  clearBtn.className = "memory-form-archive";
  clearBtn.textContent = "Clear";
  clearBtn.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.CursorClear,
      name: memory.name,
      log_name: cursor.log_name,
    });
    showToast("Cursor cleared");
  });

  row.appendChild(log);
  row.appendChild(current);
  row.appendChild(picker);
  row.appendChild(setBtn);
  row.appendChild(clearBtn);
  return row;
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

// Collector activity is the prompts tab scoped to this collection: each run is
// a full run → prompts → turns card via the shared `createRunRow`.
function createCollectorRunsSection(
  runs: PromptLogRun[],
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
    section.appendChild(createRunRow(run));
  }
  if (hasMore) {
    section.appendChild(
      createLoadMoreButton(() => requestMemoryPage("collector_runs", runs.length)),
    );
  }
  return section;
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
    intent: memory.intent ?? "",
    inclusion: memory.inclusion,
    recall: memory.recall,
    published: memory.published,
    extraction_prompt: memory.extraction_prompt ?? "",
    collector_interval_seconds: memory.collector_interval_seconds,
  });

  section.appendChild(labelled("Description", fields.description));
  section.appendChild(labelled("Intent", fields.intent));
  section.appendChild(labelled("Inclusion", fields.inclusion));
  section.appendChild(labelled("Recall", fields.recall));
  section.appendChild(labelled("Notify on new (published)", fields.published));
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
      intent: fields.intent.value.trim(),
      inclusion: fields.inclusion.value as "always" | "relevant" | "never",
      recall: fields.recall.value as "recent" | "relevant" | "all",
      published: fields.published.checked,
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

  if (memorySearch) section.appendChild(createSearchBanner(memorySearch));

  const title = document.createElement("h3");
  const shown = entries.length;
  const total = memory.entry_count;
  if (memorySearch) {
    title.textContent = `Entries (${shown}${hasMore ? "+" : ""})`;
  } else {
    title.textContent =
      total > shown ? `Entries (showing ${shown} of ${total}, newest first)` : `Entries (${total})`;
  }
  section.appendChild(title);

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "memory-entries-empty";
    empty.textContent = memorySearch ? `No entries match “${memorySearch}”.` : "No entries yet.";
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

  // Logs distinguish authors (user vs penny); a collection's entries are all
  // its collector's, so the author just repeats the collection name — drop it.
  if (memory.type === "log") {
    const author = document.createElement("span");
    author.className = "memory-entry-author";
    author.textContent = entry.author;
    header.appendChild(author);
  }

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
