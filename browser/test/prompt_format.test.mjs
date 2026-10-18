/**
 * Prompts tab — pure formatting helpers (#2567). No jsdom: plain
 * data-in, data-out functions.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { formatPillDuration, formatRunTime, formatStepCount } from "../dist/test/prompt_format.mjs";

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
