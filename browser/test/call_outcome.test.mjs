/**
 * A prompt row's outcome label (#2436) — a pure, data-in/data-out function,
 * so no jsdom is needed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { callOutcomeLabel, discardLabel } from "../dist/test/call_outcome.mjs";

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

// discardLabel (#2702)

test("a null discard_reason (a used call, or a row logged before the column existed) gets no label", () => {
  assert.equal(discardLabel(null), null);
});

test("a machine reason becomes a readable discarded label", () => {
  assert.equal(discardLabel("degenerate_output"), "discarded · degenerate output");
  assert.equal(discardLabel("xml"), "discarded · xml");
  assert.equal(discardLabel("budget_cut"), "discarded · budget cut");
  assert.equal(discardLabel("hallucinated_urls"), "discarded · hallucinated urls");
});

test("a discard reason this code has never heard of still renders a label — no hand-kept list", () => {
  assert.equal(discardLabel("some_future_reason"), "discarded · some future reason");
});
