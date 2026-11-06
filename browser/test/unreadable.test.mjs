/**
 * The unreadable-URL classification (#3057) — a pure helper, so no jsdom
 * is needed. Failures retrying can't fix (a PDF that shows the built-in
 * viewer or starts a download) vs retryable ones (timeouts, empty loads).
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { UnreadableUrlError, isUnreadableFailure } from "../dist/test/unreadable.mjs";

test("a missing host permission is unreadable", () => {
  assert.equal(isUnreadableFailure("Missing host permission for the tab"), true);
});

test("a tab-load timeout is retryable", () => {
  assert.equal(isUnreadableFailure("Tab load timed out after 20000ms"), false);
});

test("an empty extraction is retryable", () => {
  assert.equal(isUnreadableFailure("extraction failed after 10 retries"), false);
});

test("UnreadableUrlError carries the plain-words message for its url", () => {
  const err = new UnreadableUrlError("https://example.com/news/report.pdf");
  assert.equal(
    err.message,
    "https://example.com/news/report.pdf is a file (such as a PDF), not a web page, " +
      "so browse can't read it. A page that links to it may have the same content.",
  );
  assert.equal(err.name, "UnreadableUrlError");
});
