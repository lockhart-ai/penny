/**
 * Collectors list row formatting (#2568) — pure, data-in/data-out helpers,
 * so no jsdom is needed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  collectorsHeading,
  formatRelativeDate,
  logsHeading,
  scheduleAbbrev,
} from "../dist/test/collector_format.mjs";

test("formatRelativeDate reads minutes, then hours, within a day", () => {
  const now = Date.now();
  assert.equal(formatRelativeDate(new Date(now - 5_000).toISOString()), "just now");
  assert.equal(formatRelativeDate(new Date(now - 38 * 60_000).toISOString()), "38m ago");
  assert.equal(formatRelativeDate(new Date(now - 4 * 3_600_000).toISOString()), "4h ago");
});

test("formatRelativeDate reads yesterday, days, and weeks beyond a day", () => {
  const now = Date.now();
  assert.equal(formatRelativeDate(new Date(now - 25 * 3_600_000).toISOString()), "yesterday");
  assert.equal(formatRelativeDate(new Date(now - 2 * 86_400_000).toISOString()), "2d ago");
  assert.equal(formatRelativeDate(new Date(now - 10 * 86_400_000).toISOString()), "1w ago");
});

test("formatRelativeDate falls back to an absolute date beyond a month, and never throws on bad input", () => {
  const now = Date.now();
  const old = formatRelativeDate(new Date(now - 40 * 86_400_000).toISOString());
  assert.ok(old.length > 0);
  assert.equal(/ago$/.test(old), false);
  assert.equal(formatRelativeDate("not a date"), "not a date");
});

test("scheduleAbbrev reads the three presets and custom, off the raw rule alone", () => {
  assert.equal(scheduleAbbrev(null), "—");
  assert.equal(scheduleAbbrev("FREQ=HOURLY"), "hourly");
  assert.equal(scheduleAbbrev("FREQ=DAILY;BYHOUR=7;BYMINUTE=30"), "daily 7:30 AM");
  assert.equal(scheduleAbbrev("FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0"), "Mon 9:00 AM");
  assert.equal(scheduleAbbrev("FREQ=DAILY;BYHOUR=6,18;BYMINUTE=0"), "custom");
});

test("collectorsHeading matches exactly what's shown — never claims a hidden row", () => {
  assert.equal(collectorsHeading(5, 1, false), "Collectors (5)");
  assert.equal(collectorsHeading(5, 1, true), "Collectors (6, 1 archived)");
  assert.equal(collectorsHeading(5, 0, true), "Collectors (5)");
});

test("logsHeading names the count", () => {
  assert.equal(logsHeading(2), "Logs (2 — system-managed, not collectors)");
  assert.equal(logsHeading(0), "Logs (0 — system-managed, not collectors)");
});
