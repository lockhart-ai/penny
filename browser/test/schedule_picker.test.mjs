/**
 * The collection form's schedule picker (#2442) — pure preset <-> RRULE mapping.
 *
 * `presetToRule`/`ruleToPreset` only parse/build the grammar's own NAME=VALUE
 * syntax, never Penny's plain-words formatter (that comes from the server's
 * `schedule_description`, #2428) — so this is plain Node, no jsdom needed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { presetToRule, ruleToPreset, WEEKDAYS } from "../dist/test/schedule_picker.mjs";

test("none preset has no rule and round-trips from a null schedule", () => {
  assert.equal(presetToRule("none", 9, 0, "MO"), null);
  assert.deepEqual(ruleToPreset(null), { preset: "none", hour: 9, minute: 0, day: "MO" });
});

test("hourly preset round-trips", () => {
  const rule = presetToRule("hourly", 9, 0, "MO");
  assert.equal(rule, "FREQ=HOURLY");
  const parsed = ruleToPreset(rule);
  assert.equal(parsed.preset, "hourly");
});

test("daily preset round-trips at several times, including midnight", () => {
  for (const [hour, minute] of [
    [8, 0],
    [13, 30],
    [0, 5],
  ]) {
    const rule = presetToRule("daily", hour, minute, "MO");
    assert.equal(rule, `FREQ=DAILY;BYHOUR=${hour};BYMINUTE=${minute}`);
    const parsed = ruleToPreset(rule);
    assert.deepEqual(parsed, { preset: "daily", hour, minute, day: "MO" });
  }
});

test("weekly preset round-trips on every weekday", () => {
  for (const { code } of WEEKDAYS) {
    const rule = presetToRule("weekly", 9, 0, code);
    assert.equal(rule, `FREQ=WEEKLY;BYDAY=${code};BYHOUR=9;BYMINUTE=0`);
    const parsed = ruleToPreset(rule);
    assert.deepEqual(parsed, { preset: "weekly", hour: 9, minute: 0, day: code });
  }
});

test("a schedule with an end condition falls to custom, not a lossy daily match", () => {
  const parsed = ruleToPreset("FREQ=DAILY;BYHOUR=8;BYMINUTE=0;COUNT=5");
  assert.equal(parsed.preset, "custom");
});

test("a schedule with its own DTSTART falls to custom", () => {
  const parsed = ruleToPreset("DTSTART:20260101T090000Z\nFREQ=HOURLY");
  assert.equal(parsed.preset, "custom");
});

test("a weekly rule naming more than one day falls to custom, not one of them", () => {
  const parsed = ruleToPreset("FREQ=WEEKLY;BYDAY=MO,TU;BYHOUR=9;BYMINUTE=0");
  assert.equal(parsed.preset, "custom");
});

test("unparseable text falls to custom rather than throwing", () => {
  const parsed = ruleToPreset("not a schedule at all");
  assert.equal(parsed.preset, "custom");
});
