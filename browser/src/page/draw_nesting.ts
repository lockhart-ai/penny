/**
 * The pure parts of the Prompts tab's run-body list (#2435, #2631): which
 * step a draw anchors to, the run's entries as ONE flat, time-ordered list,
 * where a draw came from (its origin tag, derived purely from ids), and
 * moving a run to the front of the list on an update.  No DOM here, so these
 * are unit-testable without jsdom; `page.ts` is the only caller.
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

/** A dispatched tool call pushed as running — dropped outright the moment
 * `tool_finished` arrives (#2631): no checkmark row is ever rendered, since
 * a finished tool is already shown by the step that called it. */
export interface RunningTool {
  runId: string;
  toolCallId: string;
  toolName: string;
  status: string;
  startedAt: string;
}

/** `entry` is a running call, never a logged row or a running tool — the
 * discriminant `flatRunEntries` and page.ts both read instead of sniffing
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

/** One entry of a run's body list: a logged draw, a running call pushed
 * ahead of its promptlog row, or a running tool call (#2439).  Live step
 * markers are fed into the SAME flat build a logged draw goes through,
 * rather than a parallel one-level path — so a running entry takes its place
 * in time exactly like a logged one, including when its own parent is itself
 * still running. */
export type DrawTreeEntry = PromptLogEntry | RunningCall | RunningTool;

/** One member of `flatRunEntries`' result: the entry plus its sort key —
 * when it STARTED (`timestamp - duration_ms` for a logged row, `startedAt`
 * for a running entry), not when it logged. */
export interface FlatEntry {
  entry: DrawTreeEntry;
  startedAt: string;
}

/** A run's body list as ONE flat, time-ordered sequence (#2631): the run's
 * own prompts, every logged draw in `allDraws` (the full flat descendant
 * list the host carries, any depth — each renders once, in time order), and
 * every live running entry whose parent chain roots at the host run, sorted
 * ascending by when each entry started.
 *
 * The running entries are admitted by a small fixpoint over ids, standing in
 * for `directChildNodes`' former recursion: start from the ids that can name
 * a parent here (the run itself, every logged draw's run_id), then
 * repeatedly admit a running call whose `parentRunId` is already in the set
 * — the run's OWN next call (`parentRunId === null`, `runId` = the host) and
 * calls nested under logged draws or under other running calls, any depth —
 * and a running tool whose `runId` is in the set (a tool of the run's own
 * steps, of a logged draw, or of a still-running nested call).  A run_id
 * that logged more than one row (a poison-reroll retry) contributes one
 * entry per row — each renders, ordered by when it started.
 *
 * The list is recomputed fresh from the caller's arrays at every render, so
 * it is idempotent across rebuilds: the same inputs always produce the same
 * entries, and a live rebuild carries every logged row's promptlog id across
 * unchanged (what `capturePanelState`/`restorePanelState` key on). */
export function flatRunEntries(
  hostRunId: string,
  ownRows: PromptLogEntry[],
  allDraws: PromptLogEntry[],
  runningCalls: RunningCall[] = [],
  runningTools: RunningTool[] = [],
): FlatEntry[] {
  const parentIds = new Set<string>([hostRunId, ...allDraws.map((draw) => draw.run_id)]);
  let added = true;
  while (added) {
    added = false;
    for (const call of runningCalls) {
      if (call.parentRunId !== null && parentIds.has(call.parentRunId) && !parentIds.has(call.runId)) {
        parentIds.add(call.runId);
        added = true;
      }
    }
  }
  const logged: FlatEntry[] = [...ownRows, ...allDraws].map((entry) => ({
    entry,
    startedAt: rowStartedAt(entry),
  }));
  const running: FlatEntry[] = [
    ...runningCalls.filter(
      (call) =>
        (call.parentRunId === null ? call.runId === hostRunId : parentIds.has(call.parentRunId)),
    ),
    ...runningTools.filter((tool) => parentIds.has(tool.runId)),
  ].map((entry) => ({ entry, startedAt: entry.startedAt }));
  return [...logged, ...running].sort((a, b) => a.startedAt.localeCompare(b.startedAt));
}

/** Where a draw came from, derived purely from ids (#2631) — never from a
 * draw's name.  `step`: anchored to one of the host run's own steps by
 * `parent_tool_call_id`; `draw`: parented on another logged draw (its own
 * tag names that draw); `run`: everything else (no anchor at all — the
 * state-classifier/notifier shape — or an anchor that never matched a step,
 * the same structural fallback that renders the draw under the run). */
export type EntryOrigin =
  | { kind: "step"; step: number }
  | { kind: "draw"; label: string }
  | { kind: "run" };

/** `draw`'s origin among `hostRunId`'s own rows (`ownRows`) and its flat
 * descendant list (`allDraws`).  A draw anchored into a step is `step`; the
 * anchor lookup runs against the HOST's own rows, which is also correct for
 * a draw nested inside another draw — its anchor belongs to its parent
 * draw's rows, so it never matches a host step here and falls through to
 * the parent-draw rule below. */
export function drawOrigin(
  draw: PromptLogEntry,
  ownRows: PromptLogEntry[],
  allDraws: PromptLogEntry[],
): EntryOrigin {
  const stepIndex = attachmentIndex(ownRows, draw.parent_tool_call_id);
  if (stepIndex !== null) return { kind: "step", step: stepIndex + 1 };
  const parent = allDraws.find((candidate) => candidate.run_id === draw.parent_run_id);
  if (parent) return { kind: "draw", label: agentLabel(parent.agent_name) };
  return { kind: "run" };
}
