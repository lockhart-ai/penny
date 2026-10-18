/** Prompts tab — the prompt-log browser: runs, live step markers, nested
 * draws, run records.  Split out of page.ts (#2565); restyled to the
 * approved mockup (#2567) — the DOM-building functions below changed shape,
 * every message-handling / pending-draw / timing function is untouched. */

import {
  type PromptLogEntry,
  type PromptLogRun,
  type RunHealth,
  type RunHealthFlag,
  type RunOutcome,
  type RuntimeCallStarted,
  RuntimeMessageType,
  type RuntimePromptLogsResponse,
  type RuntimeToolFinished,
  type RuntimeToolStarted,
} from "../protocol.js";
import { callOutcomeLabel } from "./call_outcome.js";
import {
  agentLabel,
  attachmentIndex,
  dedupeDraws,
  type DrawTreeEntry,
  type DrawTreeNode,
  directChildNodes,
  dropRunningCalls,
  dropRunningTools,
  expandedIdsAfterRebuild,
  fetchedRunIdsFrom,
  isAnchored,
  isRunningCall,
  isRunningTool,
  moveToFront,
  type RunningCall,
  type RunningTool,
  runningRunIdClosure,
  rowStartedAt,
  runTimingWindow,
  toolCallIds,
} from "./draw_nesting.js";
import { formatPillDuration, formatRunTime, formatStepCount } from "./prompt_format.js";
import { createSearchBanner, wireSearchClear } from "./shared.js";

// --- Prompts state ---

const runsContainer = document.getElementById("runs")!;
const promptsLoading = document.getElementById("prompts-loading")!;
const promptsLoadMore = document.getElementById("prompts-load-more")!;
const promptsLoadMoreBtn = document.getElementById("prompts-load-more-btn")!;
let activeAgentFilter = "";
let promptSearch = "";
let flaggedOnly = false;

const AGENT_LABELS: Record<string, string> = {
  collector: "Collector",
  chat: "Chat",
  history: "History",
  startup: "Startup",
};

const ACTIVE_TIMEOUT_MS = 60_000;

let allRuns: PromptLogRun[] = [];
let hasMore = false;
const runElements = new Map<string, HTMLElement>();
let activeRunId: string | null = null;
let activeTimer: ReturnType<typeof setTimeout> | null = null;
let promptsLoaded = false;

// A live child draw (#2435) that arrived before something it needs: either its
// parent run isn't on screen yet (held here by parent run_id), or it is but the
// step it anchors to hasn't rendered yet (held here by run_id then tool_call id).
// Both are drained the moment what they're waiting for shows up — never dropped,
// never promoted to a top-level run — and both are cleared on every page reset
// (filter switch, flagged toggle, search, reconnect) so a draw whose parent or
// step never arrives (the user switched away) doesn't linger forever.
const pendingDrawsByParentRun = new Map<string, PromptLogEntry[]>();
const pendingDrawsByStep = new Map<string, Map<string, PromptLogEntry[]>>();

// Live step markers (#2439) — running calls and tool calls, pushed ahead of
// any promptlog row.  Flat arrays, not attached into any run's own state the
// way a draw is: `directChildNodes` (draw_nesting.ts) is fed these two arrays
// alongside the logged draws, and recomputes the WHOLE tree fresh at every
// render — any depth, a running call nested under another running call
// included — so there is nothing to hold-and-drain here — a running call
// whose host isn't on screen yet simply isn't rendered until the host's own
// render (for any reason) picks it up.  Resolved three ways, never left
// spinning: the real record arrives (its own run_id is dropped), the run's
// OWN outcome arrives (the whole closure under it is dropped — an abandoned
// nested draw included), or a fetch brings the run in (dropped by
// `fetchedRunIdsFrom`, same as a reconnect).
let runningCalls: RunningCall[] = [];
let runningTools: RunningTool[] = [];
let tickTimer: ReturnType<typeof setInterval> | null = null;

// --- Message-handler entry points (page.ts dispatches here) ---

// `promptsLoaded` is this module's state; page.ts reads and sets it through
// these accessors rather than owning a copy.
export function isPromptsLoaded(): boolean {
  return promptsLoaded;
}

export function markPromptsLoaded(): void {
  promptsLoaded = true;
}

export function applyPromptLogsResponse(message: RuntimePromptLogsResponse): void {
  markPromptsLoaded();
  // A fetched run's OWN child_draws win over anything held live for it
  // (#2435) — reconcile before rendering so the first paint is already
  // correct, and so a live-pending duplicate (arrived both ways) is
  // deduped rather than shown twice.
  for (const run of message.runs) reconcilePendingDraws(run);
  // A fetched run is authoritative for everything it lists — drop any
  // running placeholder for it or any of its child draws, including one the
  // live push never resolved (the call started before this page connected).
  const fetchedRunIds = fetchedRunIdsFrom(message.runs);
  runningCalls = dropRunningCalls(runningCalls, fetchedRunIds);
  runningTools = dropRunningTools(runningTools, fetchedRunIds);
  if (message.runs.length > 0 && allRuns.length > 0) {
    appendRuns(message.runs);
  } else {
    allRuns = message.runs;
    renderPrompts();
  }
  hasMore = message.has_more;
  promptsLoadMore.classList.toggle("hidden", !hasMore);
}

// ============================================================
// Prompts
// ============================================================

export function setupPrompts(): void {
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
      resetAndFetchPrompts();
    });
  }
  promptsLoadMoreBtn.addEventListener("click", () => {
    requestPromptLogs(allRuns.length);
  });
  const flaggedToggle = document.getElementById("prompts-flagged-toggle");
  flaggedToggle?.addEventListener("click", () => {
    flaggedOnly = !flaggedOnly;
    flaggedToggle.classList.toggle("active", flaggedOnly);
    resetAndFetchPrompts();
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
        pendingDrawsByParentRun.clear();
        pendingDrawsByStep.clear();
        runningCalls = [];
        runningTools = [];
        stopTicking();
        promptsLoadMore.classList.add("hidden");
        promptsLoading.textContent = promptSearch ? "Searching…" : "Loading prompt logs…";
        promptsLoading.classList.remove("hidden");
        requestPromptLogs(0);
      }, 250);
    });
  }
  wireSearchClear("prompts-search", "prompts-search-clear");
}

export function resetAndFetchPrompts(): void {
  allRuns = [];
  pendingDrawsByParentRun.clear();
  pendingDrawsByStep.clear();
  // A reconnect/refetch drops every running placeholder — the fetched state
  // wins (#2439); nothing is ever left spinning on a view that moved on.
  runningCalls = [];
  runningTools = [];
  stopTicking();
  requestPromptLogs(0);
}

export function requestPromptLogs(offset: number): void {
  const agentName = activeAgentFilter || undefined;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.PromptLogsRequest,
    agent_name: agentName,
    offset,
    query: promptSearch || undefined,
    flagged_only: flaggedOnly || undefined,
  });
}

// A draw (prompt.parent_run_id !== null) nests under whatever that id names —
// a top-level run OR another draw, any depth (#2435) — purely by id, never by
// agent_name.  A top-level prompt (parent_run_id === null) keeps the ordinary
// top-level insert/update path, which the agent filter still gates; a draw
// rides along with whatever its host run's filter status already resolved,
// so it's never checked against the filter itself.
export function handlePromptUpdate(prompt: PromptLogEntry): void {
  // This call's own running placeholder (#2439), if any, is superseded by
  // the real record that just arrived — drop it before rendering so the two
  // are never shown at once.
  const hadRunningCall = runningCalls.some((call) => call.runId === prompt.run_id);
  if (hadRunningCall) runningCalls = dropRunningCalls(runningCalls, new Set([prompt.run_id]));

  if (prompt.parent_run_id !== null) {
    attachDraw(prompt);
    return;
  }
  if (activeAgentFilter && prompt.agent_name !== activeAgentFilter) return;

  const existingRun = allRuns.find((r) => r.run_id === prompt.run_id);
  if (existingRun) {
    updateExistingRun(existingRun, prompt);
    // updateExistingRun appends incrementally; a resolved running placeholder
    // needs the full rebuild to actually disappear from the DOM.
    if (hadRunningCall) rerenderHostBody(existingRun);
  } else {
    insertNewRun(prompt);
  }
}

// The run's `started_at`/`ended_at`/`wall_duration_ms` recomputed from its own
// prompts + child draws, via the one pure window function the server mirrors
// (#2438) — called after every mutation that can move either boundary (a new
// own prompt, a newly-attached draw, or pending draws merged into a just-
// fetched run), so the live and fetched paths can never disagree.
function applyRunTiming(run: PromptLogRun): void {
  const window = runTimingWindow([...run.prompts, ...(run.child_draws ?? [])]);
  run.started_at = window.startedAt;
  run.ended_at = window.endedAt;
  run.wall_duration_ms = window.wallDurationMs;
}

function updateExistingRun(run: PromptLogRun, prompt: PromptLogEntry): void {
  run.prompts.push(prompt);
  run.prompt_count = run.prompts.length;
  // Model time: this run's OWN prompts only — child draws never add to it
  // (#2438; see wall_duration_ms below for the real elapsed time).
  run.total_duration_ms += prompt.duration_ms;
  run.total_input_tokens += prompt.input_tokens;
  run.total_output_tokens += prompt.output_tokens;
  applyRunTiming(run);

  allRuns = moveToFront(allRuns, run.run_id);
  const row = runElements.get(run.run_id);
  if (row) {
    runsContainer.prepend(row);
    const summary = row.querySelector(".run-summary")!;
    summary.replaceChild(createRunHeader(run), summary.querySelector(".run-header")!);
    const steps = row.querySelector(".steps")!;
    // A brand-new own step has nothing anchored to it yet — anything that
    // will be is drained in by `attachDraw`, which triggers its own full
    // `rerenderHostBody` once it arrives.
    steps.appendChild(createStepRow(prompt, run.prompts.length, []));
    markRunActive(run.run_id, row);
  }
  drainWaitingOn(prompt);
}

function insertNewRun(prompt: PromptLogEntry): void {
  const window = runTimingWindow([prompt]);
  const run: PromptLogRun = {
    run_id: prompt.run_id,
    agent_name: prompt.agent_name,
    prompt_count: 1,
    started_at: window.startedAt,
    ended_at: window.endedAt,
    // Model time: this run's OWN prompts only (#2438; see wall_duration_ms).
    total_duration_ms: prompt.duration_ms,
    wall_duration_ms: window.wallDurationMs,
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
    child_draws: [],
  };
  allRuns.unshift(run);
  promptsLoading.classList.add("hidden");

  const row = createRunRow(run);
  runsContainer.prepend(row);
  markRunActive(run.run_id, row);
  drainWaitingOn(prompt);
}

// --- Nesting a child draw under the round that made it (#2435) ---
//
// A host run carries the FULL flat descendant list (any depth) in its own
// `child_draws`, exactly like a fetched run does — so `findContainer` below
// can locate ANY draw, at any depth, by searching every host's own list for a
// matching run_id, with the tree itself built (for rendering) by
// `directChildNodes` from `draw_nesting.ts`.

/** The host run and "own rows" (the rows sharing `runId`) for whatever
 * `runId` names — the host itself (its own top-level prompts) or one of its
 * descendant draws (every row sharing that draw's run_id) — or `null` when
 * nothing on screen matches yet. */
function findContainer(runId: string): { host: PromptLogRun; ownRows: PromptLogEntry[] } | null {
  for (const run of allRuns) {
    if (run.run_id === runId) return { host: run, ownRows: run.prompts };
    const rows = (run.child_draws ?? []).filter((draw) => draw.run_id === runId);
    if (rows.length > 0) return { host: run, ownRows: rows };
  }
  return null;
}

/** The top-level host whose body-render would (eventually) reach `runId` —
 * real right away via `findContainer`, or by climbing a chain of running
 * calls' own `parentRunId` (#2439), any depth: a running call's parent can
 * itself be another running call, logged or not, exactly as a logged draw's
 * parent can be another draw.  `null` only while the WHOLE chain up to a
 * real run is still pending — nothing is lost even then: `runningCalls` is a
 * flat, recomputed-fresh array, never consumed, so the next render the real
 * ancestor triggers for ANY reason (its own resolution, a sibling arriving)
 * picks up every running descendant still waiting on it, same as this call
 * would have. */
function findHostFor(runId: string): PromptLogRun | null {
  const direct = findContainer(runId);
  if (direct) return direct.host;
  const asRunningCall = runningCalls.find((call) => call.runId === runId);
  if (asRunningCall && asRunningCall.parentRunId !== null) {
    return findHostFor(asRunningCall.parentRunId);
  }
  return null;
}

function holdByParent(parentRunId: string, draw: PromptLogEntry): void {
  const held = pendingDrawsByParentRun.get(parentRunId) ?? [];
  held.push(draw);
  pendingDrawsByParentRun.set(parentRunId, held);
}

function holdByStep(containerRunId: string, toolCallId: string, draw: PromptLogEntry): void {
  const byStep = pendingDrawsByStep.get(containerRunId) ?? new Map<string, PromptLogEntry[]>();
  const held = byStep.get(toolCallId) ?? [];
  held.push(draw);
  byStep.set(toolCallId, held);
  pendingDrawsByStep.set(containerRunId, byStep);
}

/** Merge whatever was held live for `run` or any of its now-fetched
 * descendants into `run.child_draws` (#2435) — the fetched copy wins on a
 * duplicate id (`dedupeDraws` keeps the first, and the fetched ones are
 * listed first).  A step-pending hold merges in too rather than being
 * dropped: the fetched snapshot's own rows are complete, so the normal
 * anchor lookup at render time resolves it properly — or, per point 3, falls
 * back to nesting under the parent as a whole if it truly doesn't match.
 * Walks outward a level at a time (a newly-merged draw's own run_id may ITSELF
 * have something pending), so a multi-level pending chain is caught whole. */
function reconcilePendingDraws(run: PromptLogRun): void {
  let merged = run.child_draws ?? [];
  let frontier = [run.run_id, ...merged.map((draw) => draw.run_id)];
  while (frontier.length > 0) {
    const next: string[] = [];
    for (const runId of frontier) {
      const heldByParent = pendingDrawsByParentRun.get(runId);
      if (heldByParent) {
        pendingDrawsByParentRun.delete(runId);
        merged = [...merged, ...heldByParent];
        next.push(...heldByParent.map((draw) => draw.run_id));
      }
      const heldByStep = pendingDrawsByStep.get(runId);
      if (heldByStep) {
        pendingDrawsByStep.delete(runId);
        for (const held of heldByStep.values()) {
          merged = [...merged, ...held];
          next.push(...held.map((draw) => draw.run_id));
        }
      }
    }
    frontier = next;
  }
  run.child_draws = dedupeDraws(merged);
  // Pending draws merged in above may extend past what the server's own
  // started_at/ended_at/wall_duration_ms already reflected (#2438) — the same
  // window function the server used, so the two can't disagree.
  applyRunTiming(run);
}

/** Attach one draw — live, at any depth: its host run isn't on screen yet
 * (held by parent run_id), its parent IS but the step it anchors to hasn't
 * rendered yet (held by parent run_id + tool_call id), or its parent is ready
 * right now.  Never drops a draw, never promotes one to a top-level run. */
function attachDraw(draw: PromptLogEntry): void {
  const parentId = draw.parent_run_id;
  if (parentId === null) return; // handlePromptUpdate only calls this when set
  const container = findContainer(parentId);
  if (!container) {
    holdByParent(parentId, draw);
    return;
  }
  const index = attachmentIndex(container.ownRows, draw.parent_tool_call_id);
  if (isAnchored(draw.parent_tool_call_id) && index === null) {
    holdByStep(parentId, draw.parent_tool_call_id!, draw);
    return;
  }
  container.host.child_draws = dedupeDraws([...(container.host.child_draws ?? []), draw]);
  // A draw can extend the host's wall time well past its own last own prompt
  // (reading a page takes real time) — recompute so the header's numbers
  // update the same way a new own prompt's do (#2438).
  applyRunTiming(container.host);
  rerenderHostBody(container.host);
  allRuns = moveToFront(allRuns, container.host.run_id);
  const hostRow = runElements.get(container.host.run_id);
  if (hostRow) {
    runsContainer.prepend(hostRow);
    const summary = hostRow.querySelector(".run-summary")!;
    summary.replaceChild(createRunHeader(container.host), summary.querySelector(".run-header")!);
    markRunActive(container.host.run_id, hostRow);
  }
  drainWaitingOn(draw);
}

/** Whatever was waiting on `arrived`'s own run_id: a grandchild draw held on
 * its parent (now `arrived`), and any draw held on one of `arrived`'s own
 * tool calls (now that `arrived` is itself a known row of that container). */
function drainWaitingOn(arrived: PromptLogEntry): void {
  const heldByParent = pendingDrawsByParentRun.get(arrived.run_id);
  if (heldByParent) {
    pendingDrawsByParentRun.delete(arrived.run_id);
    for (const draw of heldByParent) attachDraw(draw);
  }
  const byStep = pendingDrawsByStep.get(arrived.run_id);
  if (!byStep) return;
  for (const callId of toolCallIds(arrived)) {
    const held = byStep.get(callId);
    if (!held) continue;
    byStep.delete(callId);
    for (const draw of held) attachDraw(draw);
  }
  if (byStep.size === 0) pendingDrawsByStep.delete(arrived.run_id);
}

// --- Live step markers (#2439) ---
//
// Unlike a draw, a running call/tool is never "attached" into any host's own
// state (`directChildNodes` in draw_nesting.ts recomputes the match fresh,
// at any depth, every render) — so there is no per-entry hold-and-drain map
// the way `pendingDrawsByParentRun` is for a draw. The one thing a render
// still needs on arrival is WHICH top-level host to rerender, which is what
// `findHostFor` resolves — climbing through running ancestors, any depth,
// when the immediate parent isn't a real row yet. A host that still isn't
// reachable at all (the whole chain pending) simply isn't rendered THIS
// moment; nothing is lost (`runningCalls` is never consumed), and the next
// render the real ancestor triggers for any reason catches every running
// descendant still waiting on it. See `runningCalls`/`runningTools`'s own
// comment for the three resolution paths.

/** A model call is about to be made (#2439).  `parent_run_id === null` is
 * this run's OWN next step — if the run itself isn't on screen yet (the
 * run's very first call), a minimal shell row is created for it now, exactly
 * where `insertNewRun` would create one for its first LOGGED prompt. */
export function handleCallStarted(msg: RuntimeCallStarted): void {
  const call: RunningCall = {
    runId: msg.run_id,
    parentRunId: msg.parent_run_id,
    parentToolCallId: msg.parent_tool_call_id,
    agentName: msg.agent_name,
    promptType: msg.prompt_type,
    model: msg.model,
    runTarget: msg.run_target,
    startedAt: msg.started_at,
  };
  runningCalls = [...runningCalls, call];
  startTicking();

  if (call.parentRunId === null) {
    if (activeAgentFilter && (call.agentName ?? "") !== activeAgentFilter) return;
    const existingRun = allRuns.find((r) => r.run_id === call.runId);
    if (existingRun) {
      bringRunToFront(existingRun);
    } else {
      insertShellRun(call);
    }
    return;
  }

  const host = findHostFor(call.parentRunId);
  if (host) rerenderHostBody(host);
}

/** A dispatched tool call is about to run (#2439) — its host run already
 * exists (dispatch only follows the model's own logged response), so this
 * never needs the hold-and-drain a draw does. */
export function handleToolStarted(msg: RuntimeToolStarted): void {
  runningTools = [
    ...runningTools,
    {
      runId: msg.run_id,
      toolCallId: msg.tool_call_id,
      toolName: msg.tool_name,
      status: msg.status,
      startedAt: msg.started_at,
    },
  ];
  startTicking();
  const container = findContainer(msg.run_id);
  if (container) rerenderHostBody(container.host);
}

/** A dispatched tool call just finished (#2439) — flips its row from a
 * spinner to a checkmark/x in place; it is dropped later, alongside
 * everything else, by whichever resolution path clears its run_id. */
export function handleToolFinished(msg: RuntimeToolFinished): void {
  runningTools = runningTools.map((tool) =>
    tool.runId === msg.run_id && tool.toolCallId === msg.tool_call_id
      ? { ...tool, finishedSuccess: msg.success }
      : tool,
  );
  const container = findContainer(msg.run_id);
  if (container) rerenderHostBody(container.host);
}

/** A brand-new top-level run's minimal shell — created the moment its FIRST
 * call starts, exactly where `insertNewRun` creates a row for its first
 * LOGGED prompt.  `updateExistingRun` (unmodified) is what later turns this
 * into a real run, the instant that first prompt logs: it just pushes onto
 * what is, for now, an empty `prompts` array. */
function insertShellRun(call: RunningCall): void {
  const run: PromptLogRun = {
    run_id: call.runId,
    agent_name: call.agentName ?? "",
    prompt_count: 0,
    started_at: call.startedAt,
    ended_at: call.startedAt,
    total_duration_ms: 0,
    wall_duration_ms: 0,
    total_input_tokens: 0,
    total_output_tokens: 0,
    run_outcome: null,
    run_reason: null,
    run_target: call.runTarget,
    health: {
      bailed: false,
      no_writes: false,
      incomplete: false,
      tool_failures: 0,
      degenerate_send: false,
      flags: [],
      regressive: false,
    },
    record: "",
    prompts: [],
    child_draws: [],
  };
  allRuns.unshift(run);
  promptsLoading.classList.add("hidden");
  const row = createRunRow(run);
  runsContainer.prepend(row);
  markRunActive(run.run_id, row);
}

/** Re-render `run`'s body (picking up whatever running entries now apply),
 * move it to the top of the list, and keep its active-run spinner alive. */
function bringRunToFront(run: PromptLogRun): void {
  rerenderHostBody(run);
  allRuns = moveToFront(allRuns, run.run_id);
  const row = runElements.get(run.run_id);
  if (row) {
    runsContainer.prepend(row);
    markRunActive(run.run_id, row);
  }
}

function rerenderHostBody(host: PromptLogRun): void {
  const row = runElements.get(host.run_id);
  if (!row) return;
  const panel = row.querySelector(".run-view-prompts") ?? row.querySelector(".run-prompts");
  if (!panel) return;
  const snapshot = capturePanelState(panel as HTMLElement);
  panel.innerHTML = "";
  const steps = document.createElement("div");
  steps.className = "steps";
  panel.appendChild(steps);
  appendPromptsAndDraws(steps, host.prompts, host.run_id, host.child_draws ?? []);
  restorePanelState(panel as HTMLElement, snapshot);
}

interface PanelSnapshot {
  expandedIds: Set<number>;
  // The PAGE's own scroll, not the panel's: nothing under `.run-prompts`
  // scrolls internally (the page itself does, via the browser's native
  // document scroll) — capturing `panel.scrollTop` would always read 0.
  scrollY: number;
}

/** Every row the code owner currently has open to read (by promptlog id) and
 * the page's scroll position — what a live rebuild (`rerenderHostBody`) must
 * carry across the wipe, so Penny making another call never collapses what
 * he's watching mid-run or jumps the page out from under him.  Keyed by
 * `.prompt[data-prompt-id]`/`.prompt.expanded` (#2567 condition 3) — the
 * same card class and dataset the restyle kept unchanged for exactly this;
 * only what wraps a card (the rail, the always-visible draws box) changed
 * shape around it. */
function capturePanelState(panel: HTMLElement): PanelSnapshot {
  const expandedIds = new Set<number>();
  for (const el of Array.from(panel.querySelectorAll<HTMLElement>(".prompt.expanded"))) {
    const id = Number(el.dataset.promptId);
    if (!Number.isNaN(id)) expandedIds.add(id);
  }
  return { expandedIds, scrollY: window.scrollY };
}

/** Re-opens whatever `snapshot` captured, scoped to rows the rebuilt panel
 * still carries (`expandedIdsAfterRebuild`), and restores the page's scroll
 * position last — after the rows it might need to scroll to are back. */
function restorePanelState(panel: HTMLElement, snapshot: PanelSnapshot): void {
  const rows = Array.from(panel.querySelectorAll<HTMLElement>(".prompt"));
  const currentIds = rows
    .map((el) => Number(el.dataset.promptId))
    .filter((id) => !Number.isNaN(id));
  const keep = expandedIdsAfterRebuild(snapshot.expandedIds, currentIds);
  for (const el of rows) {
    if (keep.has(Number(el.dataset.promptId))) el.classList.add("expanded");
  }
  window.scrollTo(window.scrollX, snapshot.scrollY);
}

export function handleRunOutcome(runId: string, outcome: RunOutcome, reason: string): void {
  const run = allRuns.find((r) => r.run_id === runId);
  if (!run) return;
  run.run_outcome = outcome;
  run.run_reason = reason;
  allRuns = moveToFront(allRuns, runId);

  // A run's OWN outcome resolves every running placeholder under it (#2439),
  // not only one whose own record happened to arrive first — an abandoned
  // nested draw (a crash mid-call) is still cleared, so nothing is left
  // spinning once the run itself is known to be over.
  const closure = runningRunIdClosure(runId, runningCalls);
  runningCalls = dropRunningCalls(runningCalls, closure);
  runningTools = dropRunningTools(runningTools, closure);

  const row = runElements.get(runId);
  if (!row) return;
  runsContainer.prepend(row);
  rerenderHostBody(run);

  // The outcome (and any health flags already known) render as header chips
  // (#2567) — rebuild the header so it picks them up, the same
  // replace-the-header path `updateExistingRun`/`attachDraw` already use.
  const summary = row.querySelector(".run-summary");
  if (summary) {
    summary.replaceChild(createRunHeader(run), summary.querySelector(".run-header")!);
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

function appendRuns(newRuns: PromptLogRun[]): void {
  for (const run of newRuns) {
    allRuns.push(run);
    runsContainer.appendChild(createRunRow(run));
  }
}

// ============================================================
// Run card (#2567) — header (agent chip, health/outcome chips, snippet,
// fixed metric columns) + body (the step/draw timeline, or the Prompts |
// Record tabs for a tagged collector run). Shared by the Prompts tab and
// the memory Activity tab (run → prompts → turns) via `createRunRow`.
// ============================================================

export function createRunRow(run: PromptLogRun): HTMLElement {
  const row = document.createElement("div");
  row.className = "run";
  runElements.set(run.run_id, row);

  const summary = document.createElement("div");
  summary.className = "run-summary";
  summary.appendChild(createRunHeader(run));
  row.appendChild(summary);
  row.appendChild(createRunBody(run));
  summary.addEventListener("click", () => row.classList.toggle("expanded"));
  return row;
}

const RUN_OUTCOME_LABEL: Record<RunOutcome, string> = {
  worked: "worked",
  no_work: "no work",
  failed: "failed",
  incomplete: "incomplete",
  cancelled: "cancelled",
};

// A chip's colour variant per outcome — worked reads as healthy, a hard
// failure as an error, anything else (no work done / incomplete / cancelled)
// as worth a look without being a hard failure.
const OUTCOME_CHIP_VARIANT: Record<RunOutcome, string> = {
  worked: "chip--ok",
  no_work: "chip--warn",
  failed: "chip--err",
  incomplete: "chip--warn",
  cancelled: "chip--warn",
};

/** The run's outcome as a header chip (#2567) — was a separate block of
 * text under the header ("worked — reason"); the label is now the chip's
 * text and the reason (when present) rides along as its tooltip, the same
 * "say how it ended without expanding the row" shape `createOutcomeBadgeChip`
 * already uses for a single call's outcome. `null` when the run hasn't
 * resolved to anything worth a chip yet. */
function createOutcomeChip(run: PromptLogRun): HTMLElement | null {
  if (run.run_outcome === null && !run.run_reason) return null;
  const chip = document.createElement("span");
  const variant = run.run_outcome ? OUTCOME_CHIP_VARIANT[run.run_outcome] : "chip--warn";
  chip.className = `chip chip-dot ${variant}`;
  chip.textContent = run.run_outcome ? RUN_OUTCOME_LABEL[run.run_outcome] : run.run_reason ?? "";
  if (run.run_reason) chip.title = run.run_reason;
  return chip;
}

const RUN_HEALTH_LABEL: Record<RunHealthFlag, string> = {
  no_work_done: "no work done",
  no_writes: "no writes",
  incomplete: "incomplete",
  tool_failures: "tool failures",
  half_formed_send: "half-formed send",
};

// Incomplete / tool-failures read as capacity (amber, worth a look), the
// rest as a harder failure signal (red) — same split the old block styling
// drew, now as chip variants.
const HEALTH_CHIP_VARIANT: Record<RunHealthFlag, string> = {
  no_work_done: "chip--err",
  no_writes: "chip--err",
  incomplete: "chip--warn",
  tool_failures: "chip--warn",
  half_formed_send: "chip--err",
};

// One chip per set health flag — the compact form of the same RunHealth the
// concise record renders verbatim. Empty array for a healthy run (no chrome).
function createHealthChips(health: RunHealth | undefined): HTMLElement[] {
  if (!health || health.flags.length === 0) return [];
  return health.flags.map((flag) => {
    const chip = document.createElement("span");
    chip.className = `chip chip-dot ${HEALTH_CHIP_VARIANT[flag]}`;
    const count = flag === "tool_failures" && health.tool_failures > 1 ? ` (${health.tool_failures})` : "";
    chip.textContent = `${RUN_HEALTH_LABEL[flag]}${count}`;
    return chip;
  });
}

function createRunHeader(run: PromptLogRun): HTMLElement {
  const header = document.createElement("div");
  header.className = "run-header";
  header.appendChild(createRunLeft(run));
  header.appendChild(createRunRight(run));
  return header;
}

function createRunLeft(run: PromptLogRun): HTMLElement {
  const left = document.createElement("span");
  left.className = "run-left";

  const toggle = document.createElement("span");
  toggle.className = "run-toggle";
  toggle.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
  left.appendChild(toggle);

  left.appendChild(createAgentChip(run));
  for (const chip of createHealthChips(run.health)) left.appendChild(chip);
  const outcomeChip = createOutcomeChip(run);
  if (outcomeChip) left.appendChild(outcomeChip);

  // When the run carries a bound target (a collector run), it's shown
  // inline in the agent chip below instead — the snippet then only ever
  // shows the ELSE branch's derived text, never both at once.
  if (!run.run_target) {
    const typeLabel = extractPromptType(run);
    if (typeLabel) {
      const snippet = document.createElement("span");
      snippet.className = "run-snippet";
      snippet.textContent = typeLabel;
      snippet.title = typeLabel;
      left.appendChild(snippet);
    }
  }

  return left;
}

function createAgentChip(run: PromptLogRun): HTMLElement {
  const chip = document.createElement("span");
  chip.className = `agent-chip ${run.agent_name}`;

  const swatch = document.createElement("span");
  swatch.className = "swatch";
  chip.appendChild(swatch);

  chip.appendChild(document.createTextNode(AGENT_LABELS[run.agent_name] ?? agentLabel(run.agent_name)));

  // Prefer the bound collection a collector run targeted (knowledge,
  // sick-stack, …) — identifiable at a glance even without an outcome line.
  if (run.run_target) {
    const target = document.createElement("span");
    target.className = "agent-target";
    target.textContent = `· ${run.run_target}`;
    chip.appendChild(target);
  }

  // Hidden unless this run is the one `markRunActive` currently marks
  // live — see the `.run.active-run .agent-chip .live-pulse` CSS rule.
  const pulse = document.createElement("span");
  pulse.className = "live-pulse";
  pulse.title = "live";
  chip.appendChild(pulse);

  return chip;
}

function createRunLeftIcon(iconClass: string): string {
  return `<i class="fa-solid ${iconClass}"></i>`;
}

// One metric cell of the shared right-hand grid (#2610): the icon hugs the
// value and the pair is right-aligned (see .run-right/.row-right in
// prompts.css). `unit`, when given, renders in its own span so the narrow
// breakpoints can drop it from the display; the title always carries the
// full "value unit" text so the unit stays reachable.
function createMetricCell(cls: string, iconClass: string, value: string, unit = ""): HTMLElement {
  const cell = document.createElement("span");
  cell.className = `rc ${cls}`;
  cell.innerHTML =
    `${createRunLeftIcon(iconClass)}<span class="v">${value}</span>` +
    (unit ? `<span class="rc-unit">${unit}</span>` : "");
  if (unit) cell.title = `${value} ${unit}`;
  return cell;
}

function createRunRight(run: PromptLogRun): HTMLElement {
  const right = document.createElement("span");
  right.className = "run-right";

  right.appendChild(createMetricCell("rc-steps", "fa-layer-group", formatStepCount(run.prompt_count)));
  right.appendChild(createMetricCell("rc-down", "fa-arrow-down", formatTokens(run.total_input_tokens)));
  right.appendChild(createMetricCell("rc-up", "fa-arrow-up", formatTokens(run.total_output_tokens)));
  // tok/s is paired with "model time" (total_duration_ms, this run's OWN
  // prompts only) deliberately — that's the number that actually measures
  // generation throughput; "wall time" below also counts child draws (a
  // browse extraction reading pages, say), which never generated a token.
  const tokPerSec = run.total_duration_ms > 0
    ? ((run.total_output_tokens / run.total_duration_ms) * 1000).toFixed(1)
    : "0";
  right.appendChild(createMetricCell("rc-toks", "fa-gauge-high", tokPerSec, "tok/s"));

  const time = document.createElement("span");
  time.className = "rc rc-time";
  time.textContent = formatRunTime(run.started_at);
  time.title = new Date(run.started_at).toLocaleString();
  right.appendChild(time);

  right.appendChild(createTimePillCell(run));

  const copyCell = document.createElement("span");
  copyCell.className = "rc rc-copy";
  // Copy the whole run — its run_id + every prompt's full JSON (ids
  // included), plus every draw made on its behalf (#2435) — so it can be
  // pasted back for a deep look or looked up in the DB by run_id.
  copyCell.appendChild(
    createCopyButton(
      () => jsonOf({ run_id: run.run_id, prompts: run.prompts, child_draws: run.child_draws ?? [] }),
      `Copy all ${run.prompt_count} prompts as JSON (run_id ${run.run_id})`,
    ),
  );
  right.appendChild(copyCell);

  return right;
}

function createTimePillCell(run: PromptLogRun): HTMLElement {
  const cell = document.createElement("span");
  cell.className = "rc rc-pill";
  const pill = document.createElement("span");
  pill.className = "time-pair";
  pill.title = "Wall time (real elapsed, draws included) vs. model time (this run's own calls only)";
  pill.innerHTML =
    `<span class="wall">${formatPillDuration(run.wall_duration_ms)}</span>` +
    `<span class="sep">/</span>` +
    `<span class="model">${formatPillDuration(run.total_duration_ms)}</span>`;
  cell.appendChild(pill);
  return cell;
}

// The expanded body of a run: the interactive turn-based "Prompts" view, plus —
// for a completed collector run — a "Record" tab showing the concise textual
// record, the SAME representation Penny's quality collector reads of this run.
function createRunBody(run: PromptLogRun): HTMLElement {
  const body = document.createElement("div");
  body.className = "run-prompts";

  const prompts = document.createElement("div");
  prompts.className = "run-view run-view-prompts active";
  const steps = document.createElement("div");
  steps.className = "steps";
  prompts.appendChild(steps);
  appendPromptsAndDraws(steps, run.prompts, run.run_id, run.child_draws ?? []);

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

// ============================================================
// Step / draw timeline (#2567) — every top-level prompt, interleaved with
// the draws made on this run's behalf — to ANY depth (#2435), nested
// recursively under whichever step (or the run as a whole) each one
// anchors to. `ownRunId`/`ownRows` generalize the same shape
// `rerenderHostBody` reuses for a live update.
// ============================================================

function appendPromptsAndDraws(
  target: HTMLElement,
  ownRows: PromptLogEntry[],
  ownRunId: string,
  allDraws: PromptLogEntry[],
): void {
  // Live step markers (#2439) feed into the SAME tree build a logged draw
  // always went through — never a parallel one-level path — so a running
  // call or tool nests at any depth exactly like a logged draw would,
  // including one whose own parent is itself still running.
  const nodes = directChildNodes(ownRunId, ownRows, allDraws, runningCalls, runningTools);
  const byStep = new Map<number, DrawTreeNode[]>();
  const unanchored: DrawTreeNode[] = [];
  for (const node of nodes) {
    if (node.stepIndex === null) {
      unanchored.push(node);
      continue;
    }
    if (!byStep.has(node.stepIndex)) byStep.set(node.stepIndex, []);
    byStep.get(node.stepIndex)!.push(node);
  }

  // This run's OWN next call in flight is not a "child" in the tree sense
  // (nothing is parented ON it yet) — it's the run's own next row, rendered
  // separately, after its known steps, since it has no step of its own yet.
  const ownRunningCall = runningCalls.find((call) => call.runId === ownRunId);

  for (let i = 0; i < ownRows.length; i++) {
    target.appendChild(createStepRow(ownRows[i], i + 1, byStep.get(i) ?? []));
  }
  // Unanchored draws (no parent_tool_call_id) run on the round's behalf
  // rather than from inside a tool call — the state classifier / notifier /
  // skill-extractor shape — so they render as their own flat row, never
  // nested in a step's draws box, purely by that structural signal (never
  // by name).
  for (const node of unanchored) target.appendChild(createUnanchoredRow(node));
  if (ownRunningCall) target.appendChild(createOwnRunningCallRow(ownRunningCall));
  if (nodes.length > 0 || ownRunningCall) startTicking();
}

/** Whether `entry` reads as failed/erred on the rail (#2567) — a running
 * entry always reads as "live" instead (handled by the caller), so this is
 * only ever checked for a logged row. */
function railDotClass(entry: DrawTreeEntry): "" | "live" | "err" {
  if (isRunningCall(entry) || isRunningTool(entry)) return "live";
  return callOutcomeLabel((entry as PromptLogEntry).call_outcome) ? "err" : "";
}

// Wraps a row's own card (`.prompt`, built by one of the create*Card
// functions below) plus — ALWAYS, never gated by the card's own `.expanded`
// state (#2567 condition 1: a running extract under a collapsed step must
// stay visible) — a nested `.draws` box for its children, into one rail row.
// `kind` only changes the rail's size: "step-row" for a run's own top-level
// timeline entry, "draw-row" for anything nested inside a step's draws box.
function wrapRailRow(
  kind: "step-row" | "draw-row",
  dot: "" | "live" | "err",
  card: HTMLElement,
  children: DrawTreeNode[],
): HTMLElement {
  const row = document.createElement("div");
  row.className = kind;

  const rail = document.createElement("div");
  rail.className = "rail";
  const dotEl = document.createElement("span");
  dotEl.className = dot ? `dot ${dot}` : "dot";
  rail.appendChild(dotEl);
  const line = document.createElement("span");
  line.className = "line";
  rail.appendChild(line);
  row.appendChild(rail);

  const col = document.createElement("div");
  col.className = "step-col";
  col.appendChild(card);
  if (children.length > 0) col.appendChild(createDrawsBox(children));
  row.appendChild(col);

  return row;
}

// Always visible — never collapsed with the step/draw card beside it.
function createDrawsBox(children: DrawTreeNode[]): HTMLElement {
  const box = document.createElement("div");
  box.className = "draws";
  for (const node of children) box.appendChild(createDrawNodeRow(node));
  return box;
}

function createDrawNodeRow(node: DrawTreeNode): HTMLElement {
  const dot = railDotClass(node.draw);
  const card = isRunningTool(node.draw)
    ? createRunningToolCard(node.draw)
    : isRunningCall(node.draw)
      ? createRunningCallCard(node.draw, false)
      : createDrawCard(node.draw, false);
  return wrapRailRow("draw-row", dot, card, node.children);
}

function createStepRow(prompt: PromptLogEntry, step: number, children: DrawTreeNode[]): HTMLElement {
  const dot = callOutcomeLabel(prompt.call_outcome) ? "err" : "";
  return wrapRailRow("step-row", dot, createPromptCard(prompt, step), children);
}

// A draw/running entry with no step anchor — rendered flat, at the same
// rail size as an own step (#2567), never inside a `.draws` box.
function createUnanchoredRow(node: DrawTreeNode): HTMLElement {
  const dot = railDotClass(node.draw);
  const card = isRunningTool(node.draw)
    ? createRunningToolCard(node.draw)
    : isRunningCall(node.draw)
      ? createRunningCallCard(node.draw, false)
      : createDrawCard(node.draw, true);
  return wrapRailRow("step-row", dot, card, node.children);
}

function createOwnRunningCallRow(call: RunningCall): HTMLElement {
  return wrapRailRow("step-row", "live", createRunningCallCard(call, true), []);
}

// --- Cards: the toggleable `.prompt` unit (#2567 condition 3) ---
//
// Keeps the existing class name, `data-prompt-id`, and `.expanded` toggle
// mechanism unchanged — only its internal row-left/row-right layout and
// what surrounds it (the rail, the always-visible draws box) changed shape.
// `capturePanelState`/`restorePanelState` above read `.prompt[data-prompt-
// id]`/`.prompt.expanded` exactly as before a live rebuild.

function buildPromptShell(prompt: PromptLogEntry, rowLeft: HTMLElement, rowRight: HTMLElement): HTMLElement {
  const card = document.createElement("div");
  card.className = "prompt";
  card.dataset.promptId = String(prompt.id);

  const header = document.createElement("div");
  header.className = "prompt-header";
  header.appendChild(rowLeft);
  header.appendChild(rowRight);
  card.appendChild(header);

  card.appendChild(createPromptDetail(prompt));

  header.addEventListener("click", () => card.classList.toggle("expanded"));
  return card;
}

function createRowChev(withIcon: boolean): HTMLElement {
  const wrap = document.createElement("span");
  wrap.className = "row-chev";
  if (withIcon) wrap.innerHTML = '<i class="fa-solid fa-chevron-right chev"></i>';
  return wrap;
}

function createRowIcon(iconClass: string, colorClass: string, spin = false): HTMLElement {
  const wrap = document.createElement("span");
  wrap.className = "row-icon";
  const spinCls = spin ? " fa-spin" : "";
  wrap.innerHTML = `<i class="fa-solid ${iconClass}${spinCls}${colorClass ? ` ${colorClass}` : ""}"></i>`;
  return wrap;
}

// A call that did not simply complete, shown as a chip on its row (#2436) —
// the label itself is generic (`callOutcomeLabel`), and the error class +
// message (when the runtime recorded them) ride along as a tooltip, so
// hovering the chip says HOW it ended without expanding the row.
function createOutcomeBadgeChip(label: string, prompt: PromptLogEntry): HTMLElement {
  const chip = document.createElement("span");
  chip.className = "chip chip-dot chip--err";
  chip.textContent = label;
  const tooltip = [prompt.call_error_class, prompt.call_error_message]
    .filter((part): part is string => Boolean(part))
    .join(": ");
  if (tooltip) chip.title = tooltip;
  return chip;
}

function createStepRowLeft(prompt: PromptLogEntry): HTMLElement {
  const left = document.createElement("span");
  left.className = "row-left";
  left.appendChild(createRowChev(true));
  left.appendChild(
    createRowIcon(prompt.made_tool_call ? "fa-wrench" : "fa-comment", prompt.made_tool_call ? "ic-tool" : ""),
  );

  const snippet = extractLastTurnSnippet(prompt);
  const name = document.createElement("span");
  name.className = prompt.made_tool_call ? "row-name" : "row-name plain-text";
  name.textContent = snippet || (prompt.made_tool_call ? "(tool call)" : "(no response)");
  if (snippet) name.title = snippet;
  left.appendChild(name);

  // The STEP chip lives in the shared right grid's steps column now (#2610)
  // — the run header's step count sits on that same vertical line — so the
  // left badge carries only an outcome chip, and only when there is one.
  const outcomeLabel = callOutcomeLabel(prompt.call_outcome);
  if (outcomeLabel) {
    const badge = document.createElement("span");
    badge.className = "row-badge";
    badge.appendChild(createOutcomeBadgeChip(outcomeLabel, prompt));
    left.appendChild(badge);
  }

  return left;
}

// The shared right grid (#2610) — same tracks as the run header's
// .run-right, so every row's steps / ↓ / ↑ / tok/s / time / duration / copy
// columns land on the run header's vertical lines. `step` is given only for
// a top-level step row: its numbered chip fills the steps column; draws and
// running rows leave that column empty.
function createPromptRowRight(prompt: PromptLogEntry, step?: number): HTMLElement {
  const right = document.createElement("span");
  right.className = "row-right";

  const stepsCell = document.createElement("span");
  stepsCell.className = "cell-steps";
  if (step !== undefined) {
    const chip = document.createElement("span");
    chip.className = "step-badge";
    chip.textContent = `STEP ${step}`;
    stepsCell.appendChild(chip);
  }
  right.appendChild(stepsCell);

  const tokPerSec = prompt.duration_ms > 0
    ? ((prompt.output_tokens / prompt.duration_ms) * 1000).toFixed(1)
    : "0";
  right.appendChild(createCellMetric("cell-down", "fa-arrow-down", formatTokens(prompt.input_tokens)));
  right.appendChild(createCellMetric("cell-up", "fa-arrow-up", formatTokens(prompt.output_tokens)));
  right.appendChild(createCellMetric("cell-toks", "fa-gauge-high", tokPerSec, "tok/s"));

  const time = document.createElement("span");
  time.className = "cell-time";
  // This call's own start — timestamp minus its duration (#2438), the same
  // subtraction the run header's wall time is built from, so a step's shown
  // start can never disagree with it. Start and duration are separate cells
  // on the shared grid now (#2610) — the run header's start time and its
  // wall/model pill sit on these same two columns.
  time.textContent = formatTimeOfDay(rowStartedAt(prompt));
  right.appendChild(time);

  const dur = document.createElement("span");
  dur.className = "cell-dur";
  dur.textContent = formatDuration(prompt.duration_ms);
  right.appendChild(dur);

  const copyCell = document.createElement("span");
  copyCell.className = "cell-copy";
  // Copy just this prompt's full JSON — its ``id`` is the promptlog row id,
  // so it can be looked up directly in the DB.
  copyCell.appendChild(
    createCopyButton(() => jsonOf(prompt), `Copy this prompt as JSON (promptlog id ${prompt.id})`),
  );
  right.appendChild(copyCell);

  return right;
}

// Same shared-grid metric cell shape as createMetricCell above (#2610) —
// icon hugging the value, right-aligned, optional unit span with the full
// text carried in the title.
function createCellMetric(cls: string, iconClass: string, value: string, unit = ""): HTMLElement {
  const cell = document.createElement("span");
  cell.className = cls;
  cell.innerHTML =
    `<i class="fa-solid ${iconClass}"></i><span class="v">${value}</span>` +
    (unit ? `<span class="rc-unit">${unit}</span>` : "");
  if (unit) cell.title = `${value} ${unit}`;
  return cell;
}

function createPromptCard(prompt: PromptLogEntry, step: number): HTMLElement {
  return buildPromptShell(prompt, createStepRowLeft(prompt), createPromptRowRight(prompt, step));
}

// A draw's own row-left — the same prompt-detail renderer every top-level
// step uses, labelled with its OWN agent identity (never a numbered step: a
// draw has no fixed position among its siblings worth counting) rather than
// a name looked up from a closed list, so a micro-context this code has
// never heard of still reads as something (#2435). `unanchored` draws
// (structurally: no parent_tool_call_id, #2567) get the teal/italic
// system-draw treatment + a "draw" tag, same structural signal as above.
function createDrawRowLeft(draw: PromptLogEntry, unanchored: boolean): HTMLElement {
  const left = document.createElement("span");
  left.className = "row-left";
  left.appendChild(createRowChev(true));
  const failed = callOutcomeLabel(draw.call_outcome);
  left.appendChild(failed ? createRowIcon("fa-circle-xmark", "ic-err") : createRowIcon("fa-check", "ic-draw"));

  const wrap = document.createElement("span");
  wrap.className = "row-name-wrap";
  const label = agentLabel(draw.agent_name);
  const name = document.createElement("span");
  name.className = unanchored ? "draw-name sys-name" : "draw-name";
  name.textContent = label;
  name.title = label;
  wrap.appendChild(name);
  if (unanchored) {
    const tag = document.createElement("span");
    tag.className = "draw-tag";
    tag.textContent = "draw";
    wrap.appendChild(tag);
  }
  const result = document.createElement("span");
  result.className = "draw-result";
  const resultText = extractLastTurnSnippet(draw) || failed || "";
  result.textContent = resultText;
  if (resultText) result.title = resultText;
  wrap.appendChild(result);
  left.appendChild(wrap);

  if (failed) {
    const badge = document.createElement("span");
    badge.className = "row-badge";
    badge.appendChild(createOutcomeBadgeChip(failed, draw));
    left.appendChild(badge);
  }

  return left;
}

function createDrawCard(draw: PromptLogEntry, unanchored: boolean): HTMLElement {
  return buildPromptShell(draw, createDrawRowLeft(draw, unanchored), createPromptRowRight(draw));
}

// --- Live step markers (#2439) — never toggleable (no detail exists yet),
// so these build a plain `.prompt.prompt-running` card, never through
// `buildPromptShell` (no `data-prompt-id`, nothing to expand). ---

// Seconds elapsed since `startedAt`, as a short label — never negative (a
// clock a hair ahead of the server's own would otherwise read "-1s").
function formatElapsed(startedAt: string): string {
  const elapsedMs = Date.now() - new Date(startedAt).getTime();
  return `${Math.max(0, Math.floor(elapsedMs / 1000))}s`;
}

function createRunningElapsed(startedAt: string): HTMLElement {
  const elapsed = document.createElement("span");
  elapsed.className = "running-elapsed";
  elapsed.dataset.startedAt = startedAt;
  elapsed.textContent = formatElapsed(startedAt);
  return elapsed;
}

// No cost data exists yet for a running row — blank metric cells (never a
// guessed or zeroed value) and, while still running, a ticking elapsed
// time (`.running-elapsed`, read by the one shared interval `startTicking`
// drives — never a per-row timer); `null` once it's resolved but not yet
// replaced by the real logged row (a tool's checkmark moment).
function createRunningRowRight(elapsedStartedAt: string | null): HTMLElement {
  const right = document.createElement("span");
  right.className = "row-right";
  // Placeholder cells for the shared grid (#2610) — nothing to show in
  // steps / ↓ / ↑ / tok/s / start time while the call is still running.
  right.appendChild(document.createElement("span")).className = "cell-steps";
  right.appendChild(document.createElement("span")).className = "cell-down";
  right.appendChild(document.createElement("span")).className = "cell-up";
  right.appendChild(document.createElement("span")).className = "cell-toks";
  right.appendChild(document.createElement("span")).className = "cell-time";

  // The elapsed timer ticks in the duration column — the slot the resolved
  // row's own duration will occupy.
  const dur = document.createElement("span");
  dur.className = elapsedStartedAt ? "cell-dur live-text" : "cell-dur";
  if (elapsedStartedAt) {
    const spinner = document.createElement("i");
    spinner.className = "fa-solid fa-spinner fa-spin";
    dur.appendChild(spinner);
    dur.appendChild(createRunningElapsed(elapsedStartedAt));
  }
  right.appendChild(dur);

  const copyCell = document.createElement("span");
  copyCell.className = "cell-copy";
  const emptySlot = document.createElement("span");
  emptySlot.className = "copy-slot-empty";
  copyCell.appendChild(emptySlot);
  right.appendChild(copyCell);

  return right;
}

// A running model call's own row. `plain` is true only for the run's OWN
// next step (rendered as "Chat · thinking…" directly in the timeline);
// nested under a step/draw as a child, the same text splits across the
// draw-name-wrap shape instead — no status text is invented either way,
// just `agentLabel(agentName) — model`, exactly as before (#2567).
function createRunningCallCard(call: RunningCall, plain: boolean): HTMLElement {
  const row = document.createElement("div");
  row.className = "prompt prompt-running";
  const header = document.createElement("div");
  header.className = "prompt-header";
  header.appendChild(createRunningCallRowLeft(call, plain));
  header.appendChild(createRunningRowRight(call.startedAt));
  row.appendChild(header);
  return row;
}

function createRunningCallRowLeft(call: RunningCall, plain: boolean): HTMLElement {
  const left = document.createElement("span");
  left.className = "row-left";
  left.appendChild(createRowChev(false));
  left.appendChild(createRowIcon("fa-spinner", "ic-model", true));

  const agent = agentLabel(call.agentName || call.promptType || "model");
  if (plain) {
    const name = document.createElement("span");
    name.className = "row-name plain-text live-text";
    name.textContent = `${agent} · ${call.model}`;
    left.appendChild(name);
  } else {
    const wrap = document.createElement("span");
    wrap.className = "row-name-wrap";
    const name = document.createElement("span");
    name.className = "draw-name";
    name.textContent = agent;
    wrap.appendChild(name);
    const result = document.createElement("span");
    result.className = "draw-result live";
    result.textContent = call.model;
    wrap.appendChild(result);
    left.appendChild(wrap);
  }
  return left;
}

// A running (or just-finished) tool call's own row — spinner while running,
// a checkmark/x once `tool_finished` arrives (`finishedSuccess` set); the
// elapsed timer stops ticking the moment it resolves rather than freezing a
// stale number, by simply not being rendered once `finishedSuccess` is set.
function createRunningToolCard(tool: RunningTool): HTMLElement {
  const pending = tool.finishedSuccess === undefined;
  const row = document.createElement("div");
  row.className = "prompt prompt-running prompt-draw";
  const header = document.createElement("div");
  header.className = "prompt-header";
  header.appendChild(createRunningToolRowLeft(tool, pending));
  header.appendChild(createRunningRowRight(pending ? tool.startedAt : null));
  row.appendChild(header);
  return row;
}

function createRunningToolRowLeft(tool: RunningTool, pending: boolean): HTMLElement {
  const left = document.createElement("span");
  left.className = "row-left";
  left.appendChild(createRowChev(false));
  left.appendChild(
    pending
      ? createRowIcon("fa-spinner", "ic-model", true)
      : createRowIcon(tool.finishedSuccess ? "fa-check" : "fa-xmark", tool.finishedSuccess ? "ic-draw" : "ic-err"),
  );
  const wrap = document.createElement("span");
  wrap.className = "row-name-wrap";
  const result = document.createElement("span");
  result.className = pending ? "draw-result live" : "draw-result";
  result.textContent = tool.status;
  result.title = tool.status;
  wrap.appendChild(result);
  left.appendChild(wrap);
  return left;
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

function createPromptDetail(prompt: PromptLogEntry): HTMLElement {
  const detail = document.createElement("div");
  detail.className = "prompt-detail";

  const turn = document.createElement("div");
  turn.className = "turn-detail";
  detail.appendChild(turn);

  const outcomeLabel = callOutcomeLabel(prompt.call_outcome);
  if (outcomeLabel) {
    const errorText =
      [prompt.call_error_class, prompt.call_error_message].filter((part): part is string => Boolean(part))
        .join(": ") || outcomeLabel;
    turn.appendChild(
      createPromptSection("outcome", errorText, () =>
        jsonOf({
          promptlog_id: prompt.id,
          call_outcome: prompt.call_outcome,
          call_error_class: prompt.call_error_class,
          call_error_message: prompt.call_error_message,
        }),
      ),
    );
  }

  // Each turn (one message in this prompt's input) gets a copy button that
  // copies just that turn's JSON, tagged with the promptlog id for lookup.
  for (const message of prompt.messages) {
    const role = String(message.role ?? "unknown");
    const content = extractMessageContent(message);
    turn.appendChild(
      createPromptSection(role, content, () => jsonOf({ promptlog_id: prompt.id, turn: message })),
    );
  }

  if (prompt.thinking) {
    turn.appendChild(
      createPromptSection("thinking", prompt.thinking, () =>
        jsonOf({ promptlog_id: prompt.id, thinking: prompt.thinking }),
      ),
    );
  }

  turn.appendChild(
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
  // at write time, shown in the agent chip by the caller above) — never
  // reach here with one (#2567: a run_target short-circuits the snippet
  // entirely rather than also running this, which would mostly return ""
  // for a collector run anyway, since its own prompts' prompt_type usually
  // just repeats its agent_name — see the skip below).
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

// A step row's own start (#2438) is within the same round as its run header's
// date, so it only needs to add the finer-grained time-of-day — seconds
// matter here, since steps can land seconds apart.
function formatTimeOfDay(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
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

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = (ms / 1000).toFixed(1);
  return `${seconds}s`;
}

// --- Live step markers' shared ticking interval ---

// One shared ticking interval for every running row on screen at once — never
// a timer per row — started the moment any running row renders, and stopped
// for good once a tick finds none left (every one resolved or was dropped).
function startTicking(): void {
  if (tickTimer !== null) return;
  tickTimer = setInterval(() => {
    const spans = document.querySelectorAll<HTMLElement>(".running-elapsed");
    if (spans.length === 0) {
      stopTicking();
      return;
    }
    for (const span of Array.from(spans)) {
      const startedAt = span.dataset.startedAt;
      if (startedAt) span.textContent = formatElapsed(startedAt);
    }
  }, 1000);
}

function stopTicking(): void {
  if (tickTimer === null) return;
  clearInterval(tickTimer);
  tickTimer = null;
}
