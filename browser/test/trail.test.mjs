/**
 * The sidebar's live-status trail parser and the context bar's hostname
 * helper (#2570) — pure, data-in/data-out functions, so no jsdom is needed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { hostnameOf, parseTrail } from "../dist/test/trail.mjs";

test("a plain typing string is a single running row", () => {
  assert.deepEqual(parseTrail("Penny is thinking"), [
    { done: false, html: "Penny is thinking" },
  ]);
});

test("a cumulative checklist splits into done rows then running rows", () => {
  assert.deepEqual(parseTrail("&#x2713; Searched the web<br>Reading example-news.org"), [
    { done: true, html: "Searched the web" },
    { done: false, html: "Reading example-news.org" },
  ]);
});

test("several completed steps keep their order ahead of the running one", () => {
  const rows = parseTrail("&#x2713; Searched the web<br>&#x2713; Read the page<br>Drafting a reply");
  assert.deepEqual(
    rows.map((r) => [r.done, r.html]),
    [
      [true, "Searched the web"],
      [true, "Read the page"],
      [false, "Drafting a reply"],
    ],
  );
});

test("several running rows (parallel tool calls) all read as running", () => {
  const rows = parseTrail("Searching the web<br>Reading the page");
  assert.deepEqual(rows, [
    { done: false, html: "Searching the web" },
    { done: false, html: "Reading the page" },
  ]);
});

test("empty lines between steps are dropped", () => {
  assert.deepEqual(parseTrail("&#x2713; Searched the web<br><br>Reading the page"), [
    { done: true, html: "Searched the web" },
    { done: false, html: "Reading the page" },
  ]);
});

test("an empty string parses to no rows", () => {
  assert.deepEqual(parseTrail(""), []);
});

test("hostnameOf reads the host off a full URL", () => {
  assert.equal(hostnameOf("https://news-aggregator.example/today"), "news-aggregator.example");
});

test("hostnameOf returns the empty string for a URL it cannot parse", () => {
  assert.equal(hostnameOf(""), "");
  assert.equal(hostnameOf("not a url"), "");
});