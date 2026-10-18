/**
 * The Domains/Config tabs' pure display helpers (#2569) — a pure,
 * data-in/data-out module, so no jsdom is needed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  choiceLabel,
  configGroupLabel,
  domainCounts,
  filterConfigParams,
  filterDomains,
  isModified,
  isSwitchChoices,
  modifiedCount,
} from "../dist/test/tab_filters.mjs";

// --- Domains ---

const domains = [
  { domain: "example.com", permission: "allowed" },
  { domain: "tracker.example", permission: "blocked" },
  { domain: "news.example.org", permission: "allowed" },
];

test("the all filter keeps every domain, sorted by nothing (caller sorts)", () => {
  assert.deepEqual(filterDomains(domains, "all", ""), domains);
});

test("the allowed/blocked filters select by permission", () => {
  assert.deepEqual(filterDomains(domains, "allowed", ""), [domains[0], domains[2]]);
  assert.deepEqual(filterDomains(domains, "blocked", ""), [domains[1]]);
});

test("search matches the domain case-insensitively", () => {
  assert.deepEqual(filterDomains(domains, "all", "TRACKER"), [domains[1]]);
});

test("search and filter combine", () => {
  assert.deepEqual(filterDomains(domains, "blocked", "example"), [domains[1]]);
});

test("counts derive from the full list", () => {
  assert.deepEqual(domainCounts(domains), { all: 3, allowed: 2, blocked: 1 });
});

// --- Config ---

const params = [
  { key: "MAX_STEPS", value: "30", default: "20", description: "Max agent loop steps", type: "int", group: "Chat" },
  { key: "IDLE_SECONDS", value: "60.0", default: "60.0", description: "Idle threshold", type: "float", group: "Background" },
  { key: "SEARCH_URL", value: "https://ddg.gg/?q=", default: "https://duckduckgo.com/?q=", description: "Base URL for text searches", type: "str", group: "Browse" },
];

test("modified means the value differs from the default", () => {
  assert.equal(isModified(params[0]), true);
  assert.equal(isModified(params[1]), false);
});

test("the all filter keeps order; the modified filter keeps only changed params", () => {
  assert.deepEqual(filterConfigParams(params, "all", ""), params);
  assert.deepEqual(filterConfigParams(params, "modified", ""), [params[0], params[2]]);
});

test("config search matches key or description, case-insensitively", () => {
  assert.deepEqual(filterConfigParams(params, "all", "max_st"), [params[0]]);
  assert.deepEqual(filterConfigParams(params, "all", "base url"), [params[2]]);
});

test("config search and filter combine", () => {
  assert.deepEqual(filterConfigParams(params, "modified", "url"), [params[2]]);
});

test("the modified count comes from the full list", () => {
  assert.equal(modifiedCount(params), 2);
});

test("the dedup group renders as Collector — a display mapping only", () => {
  assert.equal(configGroupLabel("Memory"), "Collector");
  assert.equal(configGroupLabel("Chat"), "Chat");
});

// --- Editor choice, keyed to the value set, never to a param name ---

test('a ("0","1") pair is a switch', () => {
  assert.equal(isSwitchChoices(["0", "1"]), true);
  assert.equal(isSwitchChoices(["restrict", "allow_all"]), false);
  assert.equal(isSwitchChoices([]), false);
  assert.equal(isSwitchChoices(undefined), false);
});

test("choice values become control labels", () => {
  assert.equal(choiceLabel("restrict"), "Restrict");
  assert.equal(choiceLabel("allow_all"), "Allow all");
});