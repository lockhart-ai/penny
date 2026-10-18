/**
 * A prompt row's outcome label (#2436) — a pure, data-in/data-out function,
 * so no jsdom is needed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { callOutcomeLabel } from "../dist/test/call_outcome.mjs";

test("a null call_outcome (a row logged before the column existed) reads as completed — no label", () => {
  assert.equal(callOutcomeLabel(null), null);
});

test('"completed" reads as completed — no label', () => {
  assert.equal(callOutcomeLabel("completed"), null);
});

test("a known non-completed outcome becomes a readable label", () => {
  assert.equal(callOutcomeLabel("timed_out"), "timed out");
  assert.equal(callOutcomeLabel("cancelled"), "cancelled");
  assert.equal(callOutcomeLabel("raised"), "raised");
});

test("an outcome this code has never heard of still renders a label — no hand-kept list", () => {
  assert.equal(callOutcomeLabel("some_future_outcome"), "some future outcome");
});
