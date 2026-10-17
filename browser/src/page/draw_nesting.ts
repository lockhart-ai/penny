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

/** One draw, nested to any depth (#2435): the step of the PARENT's own rows it
 * anchors to (by index into whatever `ownRows` the caller resolved for that
 * parent — `null` for "directly under the parent as a whole", whether the
 * draw carries no anchor at all or its anchor never matched a row), and its
 * own children, built the same way, recursively. */
export interface DrawTreeNode {
  draw: PromptLogEntry;
  stepIndex: number | null;
  children: DrawTreeNode[];
}

/** Every draw directly parented on `containerRunId` — by `parent_run_id`,
 * never by agent name — each carrying the step of `containerOwnRows` it
 * anchors to and its own children, built recursively from `allDraws` (the
 * full flat descendant list a host run carries, any depth).  A draw run_id
 * that logged more than one row (a micro-context's own poison-reroll
 * attempts keep one run_id across tries) contributes one tree node per row —
 * each renders — but only the FIRST such row carries its children forward;
 * a retry's own node is always a leaf, so a child already reachable through
 * the first attempt is never attached twice. */
export function directChildNodes(
  containerRunId: string,
  containerOwnRows: PromptLogEntry[],
  allDraws: PromptLogEntry[],
): DrawTreeNode[] {
  const direct = byTimestamp(allDraws.filter((draw) => draw.parent_run_id === containerRunId));
  const seenRunIds = new Set<string>();
  return direct.map((draw) => {
    const isFirstOfItsRunId = !seenRunIds.has(draw.run_id);
    seenRunIds.add(draw.run_id);
    const ownRows = allDraws.filter((candidate) => candidate.run_id === draw.run_id);
    return {
      draw,
      stepIndex: attachmentIndex(containerOwnRows, draw.parent_tool_call_id),
      children: isFirstOfItsRunId ? directChildNodes(draw.run_id, ownRows, allDraws) : [],
    };
  });
}
