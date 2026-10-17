/**
 * Nesting a child draw under the round that made it (#2435) — the pure parts.
 * No jsdom: these are plain data-in, data-out functions.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  agentLabel,
  attachmentIndex,
  byTimestamp,
  dedupeDraws,
  directChildNodes,
  isAnchored,
  moveToFront,
  toolCallIds,
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

test("agentLabel derives a readable label for an unknown agent_name", () => {
  assert.equal(agentLabel("browse-extract"), "Browse extract");
  assert.equal(agentLabel("some_future_microcontext"), "Some future microcontext");
});

test("agentLabel never returns blank for a name with only separators", () => {
  assert.equal(agentLabel("---"), "---");
});

test("directChildNodes attaches a grandchild under its own parent draw", () => {
  // run1's own step dispatched tool call "b0"; "draw-child" is anchored to it;
  // "draw-grandchild" is parented on draw-child itself — a micro-context
  // nested inside another, never named by this code (#2435).
  const ownRows = [withToolCall("b0")];
  const child = prompt({ id: 2, run_id: "draw-child", parent_run_id: "run1", parent_tool_call_id: "b0" });
  const grandchild = prompt({ id: 3, run_id: "draw-grandchild", parent_run_id: "draw-child" });
  const allDraws = [child, grandchild];

  const tree = directChildNodes("run1", ownRows, allDraws);
  assert.equal(tree.length, 1);
  assert.equal(tree[0].draw.id, 2);
  assert.equal(tree[0].stepIndex, 0);
  assert.equal(tree[0].children.length, 1);
  assert.equal(tree[0].children[0].draw.id, 3);
  assert.equal(tree[0].children[0].stepIndex, null);
  assert.deepEqual(tree[0].children[0].children, []);
});

test("directChildNodes nests a retry's children only under its first attempt", () => {
  // Two rows share run_id "draw-1" (a poison-reroll retry) — both render as
  // siblings, but the grandchild (parented on "draw-1") attaches only once.
  const attempt1 = prompt({ id: 2, run_id: "draw-1", parent_run_id: "run1" });
  const attempt2 = prompt({
    id: 3,
    run_id: "draw-1",
    parent_run_id: "run1",
    timestamp: "2026-01-01T00:00:01Z",
  });
  const grandchild = prompt({ id: 4, run_id: "draw-2", parent_run_id: "draw-1" });
  const allDraws = [attempt1, attempt2, grandchild];

  const tree = directChildNodes("run1", [], allDraws);
  assert.equal(tree.length, 2);
  assert.equal(tree[0].children.length, 1);
  assert.equal(tree[0].children[0].draw.id, 4);
  assert.deepEqual(tree[1].children, []);
});
