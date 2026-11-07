/**
 * Pure helpers for the sidebar's server-backed chat history (#3139/#3148):
 * newest-first → oldest-first ordering, merging an older page with dedup,
 * and the caption-or-not decision history rows share with a live
 * `incoming_message` push. No jsdom needed: all data-in/data-out.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { historyCaption, oldestFirst, prependOlderPage } from "../dist/test/history.mjs";

function outgoing(id, content) {
  return {
    message_id: id,
    created_at: "2026-11-07T12:00:00",
    content,
    direction: "outgoing",
    channel_type: null,
    device_label: null,
  };
}

function incoming(id, content, channelType, deviceLabel) {
  return {
    message_id: id,
    created_at: "2026-11-07T12:00:00",
    content,
    direction: "incoming",
    channel_type: channelType,
    device_label: deviceLabel,
  };
}

test("a newest-first page renders oldest-first, top to bottom", () => {
  const newestFirst = [outgoing(3, "third"), incoming(2, "second", "signal", "Phone"), outgoing(1, "first")];
  assert.deepEqual(
    oldestFirst(newestFirst).map((m) => m.message_id),
    [1, 2, 3],
  );
});

test("an empty page orders to an empty page", () => {
  assert.deepEqual(oldestFirst([]), []);
});

test("an older page prepends ahead of what's already shown, oldest first", () => {
  const shownOldestFirst = [incoming(3, "c", "signal", "Phone"), outgoing(4, "d")];
  const olderPageNewestFirst = [incoming(2, "b", "signal", "Phone"), outgoing(1, "a")];
  assert.deepEqual(
    prependOlderPage(shownOldestFirst, olderPageNewestFirst).map((m) => m.message_id),
    [1, 2, 3, 4],
  );
});

test("a row the shown page already carries is dropped from the older page, not duplicated", () => {
  const shownOldestFirst = [incoming(2, "b", "signal", "Phone"), outgoing(3, "c")];
  // The boundary row (id 2) reappears on the older page, as a cursor page can.
  const olderPageNewestFirst = [incoming(2, "b", "signal", "Phone"), outgoing(1, "a")];
  const merged = prependOlderPage(shownOldestFirst, olderPageNewestFirst);
  assert.deepEqual(merged.map((m) => m.message_id), [1, 2, 3]);
});

test("an outgoing history row never gets a caption", () => {
  assert.equal(historyCaption(outgoing(1, "hi"), "Laptop"), undefined);
});

test("an incoming row from this sidebar's own device renders without a caption", () => {
  assert.equal(historyCaption(incoming(1, "hi", "browser", "Laptop"), "Laptop"), undefined);
});

test("an incoming row from another device gets a caption naming it", () => {
  assert.equal(
    historyCaption(incoming(1, "hi", "signal", "Kitchen Phone"), "Laptop"),
    "from Kitchen Phone · Signal",
  );
});

test("an incoming row from another browser device still gets a caption", () => {
  assert.equal(
    historyCaption(incoming(1, "hi", "browser", "Desktop"), "Laptop"),
    "from Desktop · Browser",
  );
});

test("an incoming row with no resolved device renders plain, not with a guessed caption", () => {
  assert.equal(historyCaption(incoming(1, "hi", null, null), "Laptop"), undefined);
});
