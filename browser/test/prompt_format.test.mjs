/**
 * Prompts tab — pure formatting helpers (#2567). No jsdom: plain
 * data-in, data-out functions.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  cacheTooltip,
  earlierMessagesLabel,
  earlierWindow,
  formatCachePercent,
  formatPillDuration,
  formatRunTime,
  formatStepCount,
  formatTokens,
  formatTokensPerSecond,
  sumPairedNonNull,
} from "../dist/test/prompt_format.mjs";

test("a pill duration under 10s keeps one decimal", () => {
  assert.equal(formatPillDuration(1800), "1.8s");
  assert.equal(formatPillDuration(900), "0.9s");
});

test("a pill duration from 10s to under a minute rounds to whole seconds", () => {
  assert.equal(formatPillDuration(59_000), "59s");
  assert.equal(formatPillDuration(2_300), "2.3s");
});

test("a pill duration of a minute or more reads as minutes and seconds", () => {
  assert.equal(formatPillDuration(134_000), "2m 14s");
  assert.equal(formatPillDuration(60_000), "1m 0s");
});

test("a pill duration of an hour or more reads as hours and minutes", () => {
  assert.equal(formatPillDuration(3_780_000), "1h 3m");
});

test("the two numbers in a pill never mismatch units at the same magnitude (#2567 condition 2)", () => {
  // a sub-minute wall time and a sub-minute model time both render in
  // seconds, never one in ms and the other in s.
  const wall = formatPillDuration(900);
  const model = formatPillDuration(2_300);
  assert.match(wall, /s$/);
  assert.match(model, /s$/);
});

test("step count is singular at 1, plural otherwise", () => {
  assert.equal(formatStepCount(1), "1 step");
  assert.equal(formatStepCount(2), "2 steps");
  assert.equal(formatStepCount(0), "0 steps");
});

test("a run time today shows time-of-day with seconds", () => {
  const now = new Date(2026, 9, 18, 14, 0, 0);
  const iso = new Date(2026, 9, 18, 9, 5, 3).toISOString();
  assert.match(formatRunTime(iso, now), /^\d{1,2}:\d{2}:\d{2}\s?[AP]M$/);
});

test("a run time yesterday is date-prefixed and fits the column (no seconds)", () => {
  const now = new Date(2026, 9, 18, 14, 0, 0);
  const iso = new Date(2026, 9, 17, 18, 48, 0).toISOString();
  const text = formatRunTime(iso, now);
  assert.match(text, /^Yest \d{1,2}:\d{2}\s?[AP]M$/);
});

test("a run time from further back carries the date", () => {
  const now = new Date(2026, 9, 18, 14, 0, 0);
  const iso = new Date(2026, 9, 10, 9, 0, 0).toISOString();
  const text = formatRunTime(iso, now);
  assert.match(text, /^[A-Za-z]{3} \d{1,2} \d{1,2}:\d{2}\s?[AP]M$/);
});

test("an unparseable run time falls back to the raw string", () => {
  assert.equal(formatRunTime("not-a-date"), "not-a-date");
});

test("token counts read whole below 1k and one-decimal k below 1M", () => {
  assert.equal(formatTokens(977), "977");
  assert.equal(formatTokens(1000), "1.0k");
  assert.equal(formatTokens(13_600), "13.6k");
  assert.equal(formatTokens(998_000), "998.0k");
});

test("token counts of a million or more read as one-decimal M, never outgrowing the column (#2610)", () => {
  assert.equal(formatTokens(1_000_000), "1.0M");
  assert.equal(formatTokens(1_188_000), "1.2M");
  assert.equal(formatTokens(1_234_567), "1.2M");
});

// --- The earlier/visible split of an opened entry's detail (#2640) ---

// The run's first own prompt's context: system, two earlier turns, then the
// message that started this run.
const FIRST_STEP = [
  { role: "system", content: "sys" },
  { role: "user", content: "earlier question" },
  { role: "assistant", content: "earlier answer" },
  { role: "user", content: "the message that started this run" },
];

test("the run's first step: system rows, collapsed earlier, the starting message visible", () => {
  const window = earlierWindow(FIRST_STEP, FIRST_STEP.length);
  assert.deepEqual(window, { systemCount: 1, earlierCount: 2, visibleFrom: 3 });
});

test("a later step of the same run: the boundary holds at the same index, the run's own steps visible", () => {
  // The later context is the first's, plus the steps the loop has added —
  // the split must land on the SAME starting message, everything after it
  // visible.
  const later = [
    ...FIRST_STEP,
    { role: "assistant", content: "tool call" },
    { role: "tool", content: "tool result" },
  ];
  const window = earlierWindow(FIRST_STEP, later.length);
  assert.deepEqual(window, { systemCount: 1, earlierCount: 2, visibleFrom: 3 });
});

test("a run with no history shows no collapsed row", () => {
  // Just the system prompt and the message that started the run.
  const first = [
    { role: "system", content: "sys" },
    { role: "user", content: "the starting message" },
  ];
  assert.deepEqual(earlierWindow(first, first.length), { systemCount: 1, earlierCount: 0, visibleFrom: 1 });
  // And with no system prompt at all — a single-message context.
  const bare = [{ role: "user", content: "the starting message" }];
  assert.deepEqual(earlierWindow(bare, bare.length), { systemCount: 0, earlierCount: 0, visibleFrom: 0 });
});

test("several leading system messages all render as their own rows", () => {
  const first = [
    { role: "system", content: "sys 1" },
    { role: "system", content: "sys 2" },
    { role: "user", content: "earlier" },
    { role: "assistant", content: "earlier reply" },
    { role: "user", content: "the starting message" },
  ];
  const window = earlierWindow(first, first.length);
  assert.deepEqual(window, { systemCount: 2, earlierCount: 2, visibleFrom: 4 });
});

test("a micro-context draw has no earlier conversation: everything visible, no collapsed row", () => {
  // A draw's own short context is passed as `null` — the caller's decision,
  // so the boundary rule never collapses a micro-context's middle messages.
  assert.deepEqual(earlierWindow(null, 4), { systemCount: 0, earlierCount: 0, visibleFrom: 0 });
});

test("an opened context shorter than the run's first prompt clamps to what it holds", () => {
  // Degenerate (a context can only grow within a run), but never index past
  // the opened context: its final message stays visible, nothing collapses.
  const window = earlierWindow(FIRST_STEP, 2);
  assert.deepEqual(window, { systemCount: 1, earlierCount: 0, visibleFrom: 1 });
});

test("the collapsed row's text reads N earlier messages, singular at 1", () => {
  assert.equal(earlierMessagesLabel(12), "12 earlier messages");
  assert.equal(earlierMessagesLabel(1), "1 earlier message");
});

// --- Prefill speed, generation speed, cache hit rate (#2798) ---

test("tokens per second divides tokens by the duration in seconds", () => {
  assert.equal(formatTokensPerSecond(500, 2000), "250.0");
  assert.equal(formatTokensPerSecond(123, 1000), "123.0");
});

test("tokens per second is — when either input is null, never a guessed 0", () => {
  assert.equal(formatTokensPerSecond(null, 2000), "—");
  assert.equal(formatTokensPerSecond(500, null), "—");
  assert.equal(formatTokensPerSecond(null, null), "—");
});

test("tokens per second is — on a zero duration, not a divide-by-zero or Infinity", () => {
  assert.equal(formatTokensPerSecond(500, 0), "—");
});

test("cache percent rounds cached/prompt tokens to a whole percent", () => {
  assert.equal(formatCachePercent(4096, 5000), "82%");
  assert.equal(formatCachePercent(0, 5000), "0%");
});

test("cache percent is — when either input is null or there are no prompt tokens to cache", () => {
  assert.equal(formatCachePercent(null, 5000), "—");
  assert.equal(formatCachePercent(4096, null), "—");
  assert.equal(formatCachePercent(100, 0), "—");
});

test("the cache tooltip names the counts behind the percentage, on the ↓/↑ columns' own ladder", () => {
  assert.equal(cacheTooltip(4100, 5000), "4.1k of 5.0k input tokens cached");
  assert.equal(cacheTooltip(977, 998_000), "977 of 998.0k input tokens cached");
});

test("the cache tooltip is — wherever the percentage itself would be —", () => {
  assert.equal(cacheTooltip(null, 5000), "—");
  assert.equal(cacheTooltip(100, 0), "—");
});

test("summing paired values only counts a prompt where BOTH sides are non-null", () => {
  const pairs = [
    [100, 200],
    [null, 300], // missing numerator — excluded entirely
    [50, null], // missing denominator — excluded entirely
    [25, 50],
  ];
  assert.deepEqual(sumPairedNonNull(pairs), [125, 250]);
});

test("summing paired values is null when no pair has both sides — the run header shows —", () => {
  assert.equal(sumPairedNonNull([[null, 100], [50, null]]), null);
  assert.equal(sumPairedNonNull([]), null);
});
