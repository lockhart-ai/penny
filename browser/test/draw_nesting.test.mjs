/**
 * The Prompts tab run-body list's pure parts (#2435, #2631) — the flat,
 * time-ordered entries of a run, each draw's origin from ids, and the
 * helpers around them.  No jsdom: these are plain data-in, data-out
 * functions.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  agentLabel,
  attachmentIndex,
  byTimestamp,
  dedupeDraws,
  drawOrigin,
  dropRunningCalls,
  dropRunningTools,
  expandedIdsAfterRebuild,
  fetchedRunIdsFrom,
  flatRunEntries,
  isAnchored,
  isRunningCall,
  isRunningTool,
  moveToFront,
  rowStartedAt,
  runningRunIdClosure,
  runTimingWindow,
  toolCallIds,
  upsertPromptEntry,
} from "../dist/test/draw_nesting.mjs";

function prompt(overrides = {}) {
  return {
    id: 1,
    run_id: "run1",
    timestamp: "2026-01-01T00:00:00Z",
    model: "m",
    agent_name: "chat",
    prompt_type: "",
    duration_ms: 0,
    input_tokens: 0,
    output_tokens: 0,
    run_target: null,
    messages: [],
    response: {},
    thinking: "",
    has_tools: false,
    made_tool_call: false,
    parent_run_id: null,
    parent_tool_call_id: null,
    call_outcome: null,
    call_error_class: null,
    call_error_message: null,
    ...overrides,
  };
}

function withToolCall(id) {
  return prompt({
    response: { choices: [{ message: { tool_calls: [{ id, function: { name: "browse" } }] } }] },
  });
}

test("toolCallIds reads every tool call id a response dispatched", () => {
  assert.deepEqual(toolCallIds(withToolCall("b0")), ["b0"]);
});

test("toolCallIds is empty for a plain-text reply", () => {
  const plain = prompt({ response: { choices: [{ message: { content: "hi" } }] } });
  assert.deepEqual(toolCallIds(plain), []);
});

test("toolCallIds is empty for a cancelled/no-response call", () => {
  assert.deepEqual(toolCallIds(prompt({ response: {} })), []);
});

test("attachmentIndex finds the step whose tool call matches", () => {
  const prompts = [withToolCall("a0"), withToolCall("b0")];
  assert.equal(attachmentIndex(prompts, "b0"), 1);
});

test("attachmentIndex is null for a draw with no anchor", () => {
  const prompts = [withToolCall("a0")];
  assert.equal(attachmentIndex(prompts, null), null);
});

test("attachmentIndex is null when the anchor hasn't arrived yet", () => {
  const prompts = [withToolCall("a0")];
  assert.equal(attachmentIndex(prompts, "never-seen"), null);
});

test("isAnchored distinguishes no-anchor from anchor-not-found-yet", () => {
  assert.equal(isAnchored(null), false);
  assert.equal(isAnchored("b0"), true);
});

test("byTimestamp sorts ascending without mutating the input", () => {
  const items = [prompt({ id: 2, timestamp: "2026-01-02T00:00:00Z" }), prompt({ id: 1, timestamp: "2026-01-01T00:00:00Z" })];
  const sorted = byTimestamp(items);
  assert.deepEqual(sorted.map((i) => i.id), [1, 2]);
  assert.deepEqual(items.map((i) => i.id), [2, 1]);
});

test("moveToFront moves a matching entry to the front", () => {
  const runs = [{ run_id: "a" }, { run_id: "b" }, { run_id: "c" }];
  const moved = moveToFront(runs, "c");
  assert.deepEqual(moved.map((r) => r.run_id), ["c", "a", "b"]);
});

test("moveToFront is a no-op copy when the entry is missing", () => {
  const runs = [{ run_id: "a" }, { run_id: "b" }];
  const moved = moveToFront(runs, "nope");
  assert.deepEqual(moved.map((r) => r.run_id), ["a", "b"]);
  assert.notEqual(moved, runs);
});

test("moveToFront is a no-op when the entry is already first", () => {
  const runs = [{ run_id: "a" }, { run_id: "b" }];
  const moved = moveToFront(runs, "a");
  assert.deepEqual(moved.map((r) => r.run_id), ["a", "b"]);
});

test("dedupeDraws keeps the first occurrence of a promptlog id", () => {
  const fetched = prompt({ id: 5, agent_name: "fetched" });
  const livePending = prompt({ id: 5, agent_name: "live" });
  const other = prompt({ id: 6, agent_name: "other" });
  const result = dedupeDraws([fetched, livePending, other]);
  assert.deepEqual(result.map((d) => [d.id, d.agent_name]), [
    [5, "fetched"],
    [6, "other"],
  ]);
});

test("upsertPromptEntry replaces an existing row in place — the revision wins (#2702)", () => {
  const original = prompt({ id: 5, agent_name: "original" });
  const other = prompt({ id: 6, agent_name: "other" });
  const revised = prompt({ id: 5, agent_name: "original", discard_reason: "degenerate_output" });
  const result = upsertPromptEntry([original, other], revised);
  assert.deepEqual(
    result.map((d) => [d.id, d.discard_reason]),
    [
      [5, "degenerate_output"],
      [6, undefined],
    ],
  );
  // The revised row keeps the original's position.
  assert.deepEqual(result.map((d) => d.agent_name), ["original", "other"]);
});

test("upsertPromptEntry appends a row it has never seen", () => {
  const existing = prompt({ id: 5 });
  const fresh = prompt({ id: 7 });
  assert.deepEqual(upsertPromptEntry([existing], fresh).map((d) => d.id), [5, 7]);
  assert.deepEqual(upsertPromptEntry([], fresh).map((d) => d.id), [7]);
});

test("agentLabel derives a readable label for an unknown agent_name", () => {
  assert.equal(agentLabel("browse-extract"), "Browse extract");
  assert.equal(agentLabel("some_future_microcontext"), "Some future microcontext");
});

test("agentLabel never returns blank for a name with only separators", () => {
  assert.equal(agentLabel("---"), "---");
});

test("flatRunEntries lists a direct, unanchored draw under the run as a whole", () => {
  // Today's real production shape (#2458): a browse-extract draw carries only
  // parent_run_id — parent_tool_call_id is always null — so this is the
  // common case, not an edge case: the draw joins the flat list directly,
  // with no step lookup needed at all.
  const ownRows = [withToolCall("b0")];
  const draw = prompt({ id: 2, run_id: "draw-1", parent_run_id: "run1", parent_tool_call_id: null });

  const entries = flatRunEntries("run1", ownRows, [draw]);
  assert.deepEqual(entries.map((e) => e.entry.id), [1, 2]);
  assert.equal(drawOrigin(draw, ownRows, [draw]).kind, "run");
});

test("flatRunEntries orders every entry by when it STARTED, not when it logged", () => {
  // A step logs at T+40 (took 40s, started T0); a draw started at T+10 only
  // logs at T+50 — it sits between the steps in the list, by start time.
  const step1 = withToolCall("a0");
  step1.timestamp = "2026-01-01T00:00:40Z";
  step1.duration_ms = 40_000;
  const step2 = withToolCall("b0");
  step2.id = 2;
  step2.timestamp = "2026-01-01T00:01:00Z";
  step2.duration_ms = 5_000;
  const draw = prompt({
    id: 9,
    run_id: "draw-1",
    parent_run_id: "run1",
    parent_tool_call_id: "a0",
    timestamp: "2026-01-01T00:00:50Z",
    duration_ms: 40_000,
  });

  const entries = flatRunEntries("run1", [step1, step2], [draw]);
  assert.deepEqual(entries.map((e) => e.entry.id), [1, 9, 2]);
});

test("drawOrigin falls back to from-run when the anchor never matches", () => {
  // A draw names a tool_call id, but none of the parent's own rows made it —
  // still never dropped, still rendered under the run as a whole (point 3).
  const ownRows = [withToolCall("b0")];
  const draw = prompt({ id: 2, run_id: "draw-1", parent_run_id: "run1", parent_tool_call_id: "never-seen" });
  assert.deepEqual(drawOrigin(draw, ownRows, [draw]), { kind: "run" });
});

test("drawOrigin names the step whose tool call the draw is anchored to", () => {
  const ownRows = [withToolCall("a0"), withToolCall("b0")];
  const draw = prompt({ id: 2, run_id: "draw-1", parent_run_id: "run1", parent_tool_call_id: "b0" });
  assert.deepEqual(drawOrigin(draw, ownRows, [draw]), { kind: "step", step: 2 });
});

test("drawOrigin names the parent draw when the draw's parent is another draw", () => {
  // run1's own step dispatched tool call "b0"; "draw-child" is anchored to it;
  // "draw-grandchild" is parented on draw-child itself — a micro-context
  // nested inside another, never named by this code (#2435).
  const ownRows = [withToolCall("b0")];
  const child = prompt({
    id: 2,
    run_id: "browse-extract",
    agent_name: "browse-extract",
    parent_run_id: "run1",
    parent_tool_call_id: "b0",
  });
  const grandchild = prompt({ id: 3, run_id: "draw-grandchild", parent_run_id: "browse-extract" });
  assert.deepEqual(drawOrigin(grandchild, ownRows, [child, grandchild]), {
    kind: "draw",
    label: "Browse extract",
  });
  assert.deepEqual(drawOrigin(child, ownRows, [child, grandchild]), { kind: "step", step: 1 });
});

test("expandedIdsAfterRebuild keeps every previously-expanded id still present", () => {
  const kept = expandedIdsAfterRebuild(new Set([1, 2, 3]), [1, 2, 3, 4]);
  assert.deepEqual([...kept].sort(), [1, 2, 3]);
});

test("expandedIdsAfterRebuild drops a previously-expanded id the rebuild no longer carries", () => {
  const kept = expandedIdsAfterRebuild(new Set([1, 2]), [2, 3]);
  assert.deepEqual([...kept], [2]);
});

test("expandedIdsAfterRebuild is empty when nothing was expanded", () => {
  const kept = expandedIdsAfterRebuild(new Set(), [1, 2]);
  assert.equal(kept.size, 0);
});

test("flatRunEntries renders a retry's rows once each and carries its descendant", () => {
  // Two rows share run_id "draw-1" (a poison-reroll retry) — both render,
  // ordered by when they started, and the grandchild (parented on "draw-1")
  // renders exactly once in the same flat list.
  const attempt1 = prompt({ id: 2, run_id: "draw-1", parent_run_id: "run1" });
  const attempt2 = prompt({
    id: 3,
    run_id: "draw-1",
    parent_run_id: "run1",
    timestamp: "2026-01-01T00:00:01Z",
  });
  const grandchild = prompt({ id: 4, run_id: "draw-2", parent_run_id: "draw-1" });
  const allDraws = [attempt1, attempt2, grandchild];

  const entries = flatRunEntries("run1", [], allDraws);
  // attempt1 and the grandchild both start at T0 (attempt2 starts 1s later) —
  // each renders once, in start order.
  assert.deepEqual(entries.map((e) => e.entry.id), [2, 4, 3]);
});

test("flatRunEntries is idempotent across rebuilds, carrying every logged id", () => {
  // A live rebuild recomputes the list from the same arrays — the same
  // entries come back in the same order, with the same promptlog ids, which
  // is what keeps an open row open across the wipe (#2631).
  const ownRows = [withToolCall("b0")];
  const draw = prompt({ id: 2, run_id: "draw-1", parent_run_id: "run1" });
  const calls = [runningCall({ runId: "draw-a", parentRunId: "run1" })];
  const tools = [runningTool({ runId: "run1" })];

  const first = flatRunEntries("run1", ownRows, [draw], calls, tools);
  const second = flatRunEntries("run1", ownRows, [draw], calls, tools);
  const key = (list) => list.map((e) => e.startedAt + (e.entry.id ?? e.entry.runId)).join(",");
  assert.equal(key(first), key(second));
  assert.deepEqual(first.map((e) => e.entry.id).filter((id) => id !== undefined), [1, 2]);
});

// --- Run timing (#2438): started_at is the first call's true start, wall
// time spans a long-running child draw, and a step row shows its own start.

test("rowStartedAt subtracts the call's own duration from its logged timestamp", () => {
  const started = rowStartedAt({ timestamp: "2026-01-01T00:00:05.000Z", duration_ms: 5000 });
  assert.equal(started, "2026-01-01T00:00:00.000Z");
});

test("rowStartedAt is the bare timestamp when duration_ms is 0", () => {
  const started = rowStartedAt({ timestamp: "2026-01-01T00:00:05.000Z", duration_ms: 0 });
  assert.equal(started, "2026-01-01T00:00:05.000Z");
});

test("runTimingWindow with a single row is just its own start/end", () => {
  const row = prompt({ timestamp: "2026-01-01T00:00:05.000Z", duration_ms: 5000 });
  const window = runTimingWindow([row]);
  assert.equal(window.startedAt, "2026-01-01T00:00:00.000Z");
  assert.equal(window.endedAt, "2026-01-01T00:00:05.000Z");
  assert.equal(window.wallDurationMs, 5000);
});

test("runTimingWindow takes the earliest start and the latest end across every row", () => {
  // A quick own call (10ms, starting at T0) dispatches a browse-extract draw
  // right after it ends, which then runs for 233s — the window must reach to
  // the DRAW's own end, not just the own call's, the exact bug #2438 fixes.
  const own = prompt({ id: 1, timestamp: "2026-01-01T00:00:00.010Z", duration_ms: 10 });
  const draw = prompt({ id: 2, timestamp: "2026-01-01T00:03:53.010Z", duration_ms: 233_000 });
  const window = runTimingWindow([own, draw]);
  assert.equal(window.startedAt, "2026-01-01T00:00:00.000Z"); // the OWN call's start (T0)
  assert.equal(window.endedAt, "2026-01-01T00:03:53.010Z"); // the DRAW's own end
  assert.equal(window.wallDurationMs, 233_010);
});

// --- Live step markers (#2439) ---

function runningCall(overrides = {}) {
  return {
    runId: "draw1",
    parentRunId: "run1",
    parentToolCallId: null,
    agentName: "browse-extract",
    promptType: "micro_context",
    model: "m",
    runTarget: null,
    startedAt: "2026-01-01T00:00:00.000Z",
    ...overrides,
  };
}

function runningTool(overrides = {}) {
  return {
    runId: "run1",
    toolCallId: "b0",
    toolName: "browse",
    status: "Reading example.com",
    startedAt: "2026-01-01T00:00:00.000Z",
    ...overrides,
  };
}

test("flatRunEntries includes a running call parented on the run, wherever it started", () => {
  const steps = [withToolCall("b0")];
  const entries = flatRunEntries("run1", steps, [], [runningCall({ parentToolCallId: "b0" })]);
  assert.equal(entries.length, 2);
  assert.equal(isRunningCall(entries[1].entry), true);
  assert.equal(entries[1].entry.runId, "draw1");
});

test("flatRunEntries includes the run's OWN next running call (parentRunId null)", () => {
  const entries = flatRunEntries("run1", [], [], [runningCall({ runId: "run1", parentRunId: null })]);
  assert.equal(entries.length, 1);
  assert.equal(isRunningCall(entries[0].entry), true);
});

test("flatRunEntries ignores a running call parented on a DIFFERENT run", () => {
  const entries = flatRunEntries("run1", [], [], [runningCall({ parentRunId: "some-other-run" })]);
  assert.deepEqual(entries, []);
});

test("flatRunEntries includes a running tool call dispatched by one of the run's steps", () => {
  const steps = [withToolCall("a0"), withToolCall("b0")];
  const entries = flatRunEntries("run1", steps, [], [], [runningTool({ toolCallId: "b0" })]);
  assert.equal(entries.length, 3); // two logged steps + the running tool
  assert.equal(isRunningTool(entries[2].entry), true);
});

test("flatRunEntries reaches a running call under a tool call, and a SECOND under that one — any depth (#2439)", () => {
  // The code owner's motivating shape: a tool call (dispatched by run1's own
  // step 0) is still running a browse-extract draw (A), which is itself
  // still in flight when a FURTHER draw (B) starts, parented on A rather
  // than on run1 directly. Neither A nor B has logged anything yet.
  const steps = [withToolCall("b0")];
  const drawA = runningCall({ runId: "draw-a", parentRunId: "run1", parentToolCallId: "b0" });
  const drawB = runningCall({
    runId: "draw-b",
    parentRunId: "draw-a",
    parentToolCallId: null,
    startedAt: "2026-01-01T00:00:01.000Z",
  });
  const entries = flatRunEntries("run1", steps, [], [drawA, drawB]);

  assert.equal(entries.length, 3); // the logged step + both running calls
  assert.deepEqual(entries.slice(1).map((e) => e.entry.runId), ["draw-a", "draw-b"]); // by startedAt
  assert.equal(isRunningCall(entries[1].entry), true);
  assert.equal(isRunningCall(entries[2].entry), true);
});

test("a running draw nested two levels deep resolves to a LOGGED draw in the same position once its record arrives", () => {
  // Same shape as above, but B's own promptlog row has now landed — it is
  // no longer in runningCalls at all (page.ts drops it by run_id on
  // arrival), and the flat list must still carry it exactly where it
  // rendered while running.
  const steps = [withToolCall("b0")];
  const drawA = runningCall({
    runId: "draw-a",
    parentRunId: "run1",
    parentToolCallId: "b0",
    startedAt: "2026-01-01T00:00:02.000Z",
  });
  const loggedB = prompt({
    id: 7,
    run_id: "draw-b",
    parent_run_id: "draw-a",
    parent_tool_call_id: null,
    timestamp: "2026-01-01T00:00:05.000Z",
  });
  const entries = flatRunEntries("run1", steps, [loggedB], [drawA]);

  assert.equal(entries.length, 3); // the logged step + draw-a (running) + B (logged)
  assert.equal(isRunningCall(entries[1].entry), true);
  assert.equal(isRunningCall(entries[2].entry), false);
  assert.equal(entries[2].entry.id, 7); // now the real PromptLogEntry, not a RunningCall
  // Its parent (draw-a) is still running — no logged draw names it yet, so
  // the origin reads "from run" for this moment; the moment draw-a logs, its
  // rebuild derives the draw-parent tag from ids instead.
  assert.deepEqual(drawOrigin(loggedB, steps, [loggedB]), { kind: "run" });
});

test("runningRunIdClosure is just the root when nothing nests under it", () => {
  const closure = runningRunIdClosure("run1", [runningCall({ runId: "draw1", parentRunId: "some-other-run" })]);
  assert.deepEqual([...closure], ["run1"]);
});

test("runningRunIdClosure reaches a draw nested two levels deep under the root", () => {
  const calls = [
    runningCall({ runId: "draw1", parentRunId: "run1" }),
    runningCall({ runId: "draw2", parentRunId: "draw1" }),
  ];
  const closure = runningRunIdClosure("run1", calls);
  assert.deepEqual([...closure].sort(), ["draw1", "draw2", "run1"]);
});

test("dropRunningCalls filters by runId, leaving everything else", () => {
  const calls = [runningCall({ runId: "draw1" }), runningCall({ runId: "draw2" })];
  assert.deepEqual(
    dropRunningCalls(calls, new Set(["draw1"])).map((c) => c.runId),
    ["draw2"],
  );
});

test("dropRunningTools filters by runId, leaving everything else", () => {
  const tools = [runningTool({ runId: "run1" }), runningTool({ runId: "run2" })];
  assert.deepEqual(
    dropRunningTools(tools, new Set(["run1"])).map((t) => t.runId),
    ["run2"],
  );
});

test("fetchedRunIdsFrom collects a run's own id and every child draw's", () => {
  const runs = [
    { run_id: "run1", child_draws: [prompt({ id: 1, run_id: "draw1" }), prompt({ id: 2, run_id: "draw2" })] },
    { run_id: "run2" },
  ];
  assert.deepEqual([...fetchedRunIdsFrom(runs)].sort(), ["draw1", "draw2", "run1", "run2"]);
});
