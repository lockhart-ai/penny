/**
 * Nesting a child draw under the round that made it (#2435) — the pure parts:
 * which step a draw anchors to, time ordering, and moving a run to the front
 * of the list on an update.  No DOM here, so these are unit-testable without
 * jsdom; `page.ts` is the only caller.
 */

import type { PromptLogEntry } from "../protocol.js";

/** Every tool_call id this prompt's response actually dispatched — the match
 * key a draw's `parent_tool_call_id` is read against.  Empty when the
 * response made no tool call, or made one with no readable id. */
export function toolCallIds(prompt: PromptLogEntry): string[] {
  const response = prompt.response as Record<string, unknown> | undefined;
  const choices = response?.choices as Record<string, unknown>[] | undefined;
  const message = choices?.[0]?.message as Record<string, unknown> | undefined;
  const calls = message?.tool_calls as Record<string, unknown>[] | undefined;
  if (!calls) return [];
  return calls.map((call) => String(call.id ?? "")).filter((id) => id.length > 0);
}

/** The 0-based index of the step in `prompts` that dispatched `parentToolCallId`
 * — or `null` when the draw carries no anchor at all (the state classifier /
 * notifier / skill-extractor shape, which runs on the round's behalf rather
 * than from inside a tool call) OR when no step's tool calls match it YET
 * (a live run whose anchoring step hasn't rendered — the caller holds the
 * draw rather than treating this the same as "no anchor"; see `isAnchored`). */
export function attachmentIndex(
  prompts: PromptLogEntry[],
  parentToolCallId: string | null,
): number | null {
  if (parentToolCallId === null) return null;
  for (let i = 0; i < prompts.length; i++) {
    if (toolCallIds(prompts[i]).includes(parentToolCallId)) return i;
  }
  return null;
}

/** Whether a draw carries an anchor at all — the caller's way to tell "render
 * now, directly under the run" (`false`) apart from "hold until the step
 * shows up" (`true`, when `attachmentIndex` still came back `null`). */
export function isAnchored(parentToolCallId: string | null): boolean {
  return parentToolCallId !== null;
}

/** A new array, ascending by `timestamp` — the order a run's own prompts
 * already arrive in from the server, and the order its draws render in. */
export function byTimestamp<T extends { timestamp: string }>(items: T[]): T[] {
  return [...items].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

/** One row's own start — `timestamp` minus however long the call took
 * (`duration_ms`) — the same subtraction `runTimingWindow` reduces over many
 * rows, shared so a step row's own displayed start can never disagree with
 * how the run header derives its (#2438). */
export function rowStartedAt(row: { timestamp: string; duration_ms: number }): string {
  return new Date(new Date(row.timestamp).getTime() - row.duration_ms).toISOString();
}

/** The structural timing window of a run: the earliest `timestamp -
 * duration_ms` and the latest `timestamp` across every row handed to it — a
 * run's own prompts AND its child draws, at any depth (#2438).  Mirrors the
 * server's own window (`_run_window` in `message_store.py`), and is the one
 * place either the FETCHED path (reconciling pending draws into a
 * just-fetched run, which may extend its true start/end past what the server
 * already computed) or the LIVE path (a pushed prompt or draw) derives a
 * run's `started_at`/`ended_at`/`wall_duration_ms` — so the two can never
 * disagree. */
export function runTimingWindow(
  rows: { timestamp: string; duration_ms: number }[],
): { startedAt: string; endedAt: string; wallDurationMs: number } {
  const starts = rows.map((r) => new Date(r.timestamp).getTime() - r.duration_ms);
  const ends = rows.map((r) => new Date(r.timestamp).getTime());
  const start = Math.min(...starts);
  const end = Math.max(...ends);
  return {
    startedAt: new Date(start).toISOString(),
    endedAt: new Date(end).toISOString(),
    wallDurationMs: end - start,
  };
}

/** A new array with the entry matching `id` moved to the front — "any update
 * to a run moves it to the top", live.  Returns a shallow copy unchanged when
 * `id` isn't present or is already first. */
export function moveToFront<T extends { run_id: string }>(items: T[], id: string): T[] {
  const index = items.findIndex((item) => item.run_id === id);
  if (index <= 0) return items.slice();
  const copy = items.slice();
  const [moved] = copy.splice(index, 1);
  copy.unshift(moved);
  return copy;
}

/** `draws` with every later occurrence of a promptlog `id` already seen
 * dropped, keeping the FIRST.  Callers put the run's own fetched
 * `child_draws` first and any live-pending draws after, so a draw that
 * arrived both ways keeps the fetched copy and the live duplicate is the one
 * discarded. */
export function dedupeDraws(draws: PromptLogEntry[]): PromptLogEntry[] {
  const seen = new Set<number>();
  const out: PromptLogEntry[] = [];
  for (const draw of draws) {
    if (seen.has(draw.id)) continue;
    seen.add(draw.id);
    out.push(draw);
  }
  return out;
}

/** A readable label for an `agent_name` the caller has no nicer, hand-written
 * entry for (#2435) — the code owner moves, adds and drops micro-contexts at
 * will, so a name this code has never heard of must still read as something,
 * never blank and never the bare hyphenated identifier.  Separators become
 * spaces and the result is sentence-cased: `"browse-extract"` →
 * `"Browse extract"`, `"some_future_microcontext"` → `"Some future
 * microcontext"`. */
export function agentLabel(name: string): string {
  const words = name.replace(/[-_]+/g, " ").trim();
  if (!words) return name;
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** The expanded-row ids to keep across a panel rebuild (#2461 review) — a
 * live update must never collapse a row the code owner has open to read, so
 * `page.ts`'s `rerenderHostBody` captures every expanded id before wiping a
 * run's panel and restores this subset after rebuilding it.  Only the
 * intersection with `currentIds` survives: a rebuild only ever ADDS rows in
 * practice, but a stale id — one the rebuilt panel no longer carries at all —
 * is dropped rather than kept forever, never re-applied to some unrelated
 * row that happens to reuse a DOM position. */
export function expandedIdsAfterRebuild(
  previouslyExpanded: ReadonlySet<number>,
  currentIds: Iterable<number>,
): Set<number> {
  const current = new Set(currentIds);
  const kept = new Set<number>();
  for (const id of previouslyExpanded) {
    if (current.has(id)) kept.add(id);
  }
  return kept;
}

// --- Live step markers (#2439): running calls and tool calls, pushed ahead of
// any promptlog row.  Kept as their OWN client-only types rather than folded
// into `PromptLogEntry` (so a running row can never be mistaken for a logged
// one, or render a "pending" badge through the #2469 outcome styling) — see
// page.ts for the flat arrays these live in and how they resolve.

/** A model call pushed as "about to be made" — no promptlog row exists for it
 * yet.  Anchors exactly like a draw does: `parentRunId`/`parentToolCallId`
 * read the same way `attachmentIndex` reads a draw's. */
export interface RunningCall {
  runId: string;
  parentRunId: string | null;
  parentToolCallId: string | null;
  agentName: string | null;
  promptType: string | null;
  model: string;
  runTarget: string | null;
  startedAt: string;
}

/** A dispatched tool call pushed as running, or just finished —
 * `finishedSuccess` is `undefined` while still running. */
export interface RunningTool {
  runId: string;
  toolCallId: string;
  toolName: string;
  status: string;
  startedAt: string;
  finishedSuccess?: boolean;
}

/** `entry` is a running call, never a logged row or a running tool — the
 * discriminant `directChildNodes` and page.ts both read instead of sniffing
 * structurally at each call site.  A running call carries `runId` (a real
 * `PromptLogEntry` carries `run_id`, and a running tool carries `toolCallId`,
 * never `runId`), so this is unambiguous. */
export function isRunningCall(entry: DrawTreeEntry): entry is RunningCall {
  return "runId" in entry;
}

/** `entry` is a running tool call — see `isRunningCall`. */
export function isRunningTool(entry: DrawTreeEntry): entry is RunningTool {
  return "toolCallId" in entry;
}

function entryTimestamp(entry: DrawTreeEntry): string {
  return isRunningCall(entry) || isRunningTool(entry) ? entry.startedAt : entry.timestamp;
}

/** The run_id a child of `entry` would be parented on, or `null` for a
 * running tool — a tool call never mints a run_id of its own, so nothing
 * chains through it as a PARENT in the run_id sense (it still anchors
 * children visually, by `parent_tool_call_id`, exactly like a draw does —
 * see `directChildNodes`). */
function entryOwnRunId(entry: DrawTreeEntry): string | null {
  if (isRunningTool(entry)) return null;
  return isRunningCall(entry) ? entry.runId : entry.run_id;
}

/** `rootRunId` plus every running call's `runId` transitively parented under
 * it (any depth, via `parentRunId` chains) — the sweep a run's OWN outcome
 * arriving must resolve, not only its own direct entry: an abandoned nested
 * draw that never logs its own terminal row (a crash mid-call) is still
 * cleared once the RUN it belongs to is known to be over, so nothing is left
 * spinning forever. */
export function runningRunIdClosure(rootRunId: string, calls: RunningCall[]): Set<string> {
  const closure = new Set<string>([rootRunId]);
  let added = true;
  while (added) {
    added = false;
    for (const call of calls) {
      if (call.parentRunId !== null && closure.has(call.parentRunId) && !closure.has(call.runId)) {
        closure.add(call.runId);
        added = true;
      }
    }
  }
  return closure;
}

/** Running calls with every entry whose `runId` is in `runIds` dropped — the
 * shared filter behind every resolution path (a real record arriving, a run
 * outcome, a fetch bringing the run in). */
export function dropRunningCalls(calls: RunningCall[], runIds: ReadonlySet<string>): RunningCall[] {
  return calls.filter((call) => !runIds.has(call.runId));
}

/** Running tool calls with every entry whose `runId` is in `runIds` dropped —
 * see `dropRunningCalls`. */
export function dropRunningTools(tools: RunningTool[], runIds: ReadonlySet<string>): RunningTool[] {
  return tools.filter((tool) => !runIds.has(tool.runId));
}

/** Every run_id a fetched page of runs is now authoritative for — each run's
 * own id plus every child draw's — so a running placeholder for anything the
 * fetch just brought in can be dropped, even one the live push never
 * resolved (a reconnect, or a call that started before this page loaded). */
export function fetchedRunIdsFrom(runs: { run_id: string; child_draws?: PromptLogEntry[] }[]): Set<string> {
  const ids = new Set<string>();
  for (const run of runs) {
    ids.add(run.run_id);
    for (const draw of run.child_draws ?? []) ids.add(draw.run_id);
  }
  return ids;
}

/** One entry a `DrawTreeNode` can carry: a logged draw, a running call
 * pushed ahead of its promptlog row, or a running tool call (#2439).  Live
 * step markers are fed into the SAME tree build a logged draw always went
 * through, rather than a parallel one-level path — so a running entry nests
 * at any depth exactly like a logged one would, including when its own
 * parent is itself still running. */
export type DrawTreeEntry = PromptLogEntry | RunningCall | RunningTool;

/** One node of the tree, nested to any depth (#2435, extended to running
 * entries by #2439): the step of the PARENT's own rows it anchors to (by
 * index into whatever `ownRows` the caller resolved for that parent — `null`
 * for "directly under the parent as a whole", whether the entry carries no
 * anchor at all or its anchor never matched a row), and its own children,
 * built the same way, recursively. */
export interface DrawTreeNode {
  draw: DrawTreeEntry;
  stepIndex: number | null;
  children: DrawTreeNode[];
}

/** Every draw, running call, or running tool call directly parented on
 * `containerRunId` — by `parent_run_id`/`parentRunId`/`runId`, never by agent
 * name — each carrying the step of `containerOwnRows` it anchors to and its
 * own children, built recursively from `allDraws` (the full flat descendant
 * list a host run carries, any depth) PLUS whatever `runningCalls`/
 * `runningTools` currently apply.  `runningCalls`/`runningTools` default to
 * empty, so every existing caller that only ever dealt in logged draws is
 * unaffected.
 *
 * A run_id that logged more than one row (a poison-reroll retry) OR has a
 * logged row plus a still-running NEXT attempt contributes one tree node per
 * entry — each renders, ordered by when it started — but only the FIRST
 * carries its children forward; a retry's own node (logged or running) is
 * always a leaf, so a child already reachable through the first attempt is
 * never attached twice.
 *
 * A running call's own run_id is exactly as valid a PARENT id as a logged
 * draw's: a running call nested under another running call recurses the
 * same way, with its container's `ownRows` simply empty (nothing of its own
 * has logged yet) — `attachmentIndex` already returns `null` against an
 * empty list, so such a child renders unanchored, directly under its
 * (also-running) parent, with no special case needed.
 *
 * A running TOOL call is never itself a parent in this sense (a tool call
 * mints no run_id — see `entryOwnRunId`), but it anchors into the SAME step
 * position a draw it goes on to make would, via its own `toolCallId` doubling
 * as the anchor key. */
export function directChildNodes(
  containerRunId: string,
  containerOwnRows: PromptLogEntry[],
  allDraws: PromptLogEntry[],
  runningCalls: RunningCall[] = [],
  runningTools: RunningTool[] = [],
): DrawTreeNode[] {
  const directDraws: DrawTreeEntry[] = allDraws.filter(
    (draw) => draw.parent_run_id === containerRunId,
  );
  const directCalls: DrawTreeEntry[] = runningCalls.filter(
    (call) => call.parentRunId === containerRunId,
  );
  const directTools: DrawTreeEntry[] = runningTools.filter((tool) => tool.runId === containerRunId);
  const direct = [...directDraws, ...directCalls, ...directTools].sort((a, b) =>
    entryTimestamp(a).localeCompare(entryTimestamp(b)),
  );

  const seenRunIds = new Set<string>();
  return direct.map((entry) => {
    const parentToolCallId = isRunningTool(entry)
      ? entry.toolCallId // a tool anchors by its OWN id — it IS the tool call
      : isRunningCall(entry)
        ? entry.parentToolCallId
        : entry.parent_tool_call_id;
    const ownRunId = entryOwnRunId(entry);
    let children: DrawTreeNode[] = [];
    if (ownRunId !== null) {
      const isFirstOfItsRunId = !seenRunIds.has(ownRunId);
      seenRunIds.add(ownRunId);
      if (isFirstOfItsRunId) {
        const ownRows = allDraws.filter((candidate) => candidate.run_id === ownRunId);
        children = directChildNodes(ownRunId, ownRows, allDraws, runningCalls, runningTools);
      }
    }
    return {
      draw: entry,
      stepIndex: attachmentIndex(containerOwnRows, parentToolCallId),
      children,
    };
  });
}
