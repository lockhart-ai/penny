/**
 * The three BUILT artifacts of the harness, exercised as artifacts.
 *
 * `compare_readings.test.mjs` proves the comparator's logic off its source.  This
 * file proves the things that only break after the bundler: the injected build
 * really RETURNS its comparison (an executeScript that silently returns
 * undefined would look like twenty failed pages), the paste-in build really
 * logs and copies, the background bundle parses, and the manifest points at
 * files that exist.
 *
 * Everything here runs in jsdom, so the harness is verified end to end without
 * launching Firefox — which is the point: the browser run is the code owner's.
 *
 * DO NOT MERGE this branch. See tools/compare-readings/README.md.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script } from "node:vm";
import { JSDOM } from "jsdom";

const HOMEPAGE_URL = "https://news-alpha.example/";

const HARNESS = artifact("../tools/compare-readings/extension/build/harness.js");
const INJECTED = artifact("../tools/compare-readings/extension/build/comparator.js");
const SNIPPET = artifact("../dist/compare/compare-snippet.js");
const MANIFEST = JSON.parse(artifact("../tools/compare-readings/extension/manifest.json"));

function artifact(path) {
  return readFileSync(new URL(path, import.meta.url), "utf8");
}

function fixture(name) {
  return readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
}

/** A window the built bundles can be evaluated in, as a page would evaluate
 *  them.  `outside-only` gives the window a working `eval` without letting the
 *  fixture's own scripts run. */
function windowFor(html, url) {
  return new JSDOM(html, { url, runScripts: "outside-only" }).window;
}

test("the injected build returns its comparison to executeScript", () => {
  const window = windowFor(fixture("news-homepage.html"), HOMEPAGE_URL);

  // This is exactly what executeScript does with the file: evaluate it and take
  // the completion value.  undefined here is the failure mode that would make a
  // whole run come back empty with no error anywhere.
  const comparison = window.eval(INJECTED);

  assert.equal(typeof comparison, "object");
  assert.equal(comparison.requestedUrl, HOMEPAGE_URL);
  assert.equal(comparison.arbiter.picked, "index");
  assert.equal(comparison.arbiter.mirrorAgreesWithReadPage, true);
  assert.ok(comparison.pr.words > comparison.production.words);
  assert.ok(comparison.headlineProbes.length > 0);
});

test("the paste-in build logs its JSON and copies it when the console offers copy", () => {
  const window = windowFor(fixture("news-homepage.html"), HOMEPAGE_URL);
  const logged = [];
  window.console.log = (...args) => logged.push(args);
  const copied = [];
  window.copy = (value) => copied.push(value);

  window.eval(SNIPPET);

  assert.equal(copied.length, 1);
  assert.equal(JSON.parse(copied[0]).arbiter.picked, "index");
  assert.deepEqual(logged.at(-1), ["[compare-readings] JSON copied to the clipboard"]);
  assert.equal(logged[0][0], "[compare-readings]");
});

test("the paste-in build prints its JSON where there is no copy", () => {
  const window = windowFor(fixture("news-article.html"), `${HOMEPAGE_URL}world/story`);
  const logged = [];
  window.console.log = (...args) => logged.push(args);

  window.eval(SNIPPET);

  // No DevTools `copy` — the JSON has to be on screen or the spot-check is lost.
  const printed = JSON.parse(logged.at(-1)[0]);
  assert.equal(printed.arbiter.picked, "article");
  assert.equal(printed.pr.chars, printed.production.chars);
});

test("the background bundle parses and drives the file the manifest ships", () => {
  assert.doesNotThrow(() => new Script(HARNESS));

  // The path the harness injects has to be the path the build writes, and the
  // manifest has to load the harness itself — three files, one wiring.
  assert.ok(HARNESS.includes('"/build/comparator.js"'));
  assert.deepEqual(MANIFEST.background.scripts, ["build/harness.js"]);
  assert.ok(INJECTED.length > 0);

  // Everything the harness calls, declared.
  for (const permission of ["tabs", "downloads", "<all_urls>"]) {
    assert.ok(MANIFEST.permissions.includes(permission), `manifest is missing ${permission}`);
  }
  assert.ok(MANIFEST.browser_action, "the progress badge needs a browser action");
  assert.match(MANIFEST.name, /DO NOT SHIP/);
});
