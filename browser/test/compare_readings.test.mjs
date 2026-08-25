/**
 * The reading-comparison harness (tools/compare-readings) — proven over the same
 * jsdom fixtures the PR itself is pinned on, so the metrics and the JSON shape
 * are known-good BEFORE anyone spends ten minutes driving Firefox with it.
 *
 * What these tests are and are not: they prove the harness MEASURES correctly.
 * They cannot prove anything about real news homepages — that is what the run in
 * a rendered browser is for, and it is the whole reason this branch exists.
 *
 * DO NOT MERGE this branch. See tools/compare-readings/README.md.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

import { compareReadings } from "../dist/test/comparator.mjs";
import { SITES, LIST_VERSION } from "../dist/test/sites.mjs";

const SITE = "https://news-alpha.example";
const HOMEPAGE_URL = `${SITE}/`;
const ARTICLE_URL = `${SITE}/world/2036/harbour-bridge-reopens-after-refit`;

/** A sentence that exists only in the article fixture's body. */
const ARTICLE_SENTENCE = "The first vehicles rolled over the harbour bridge shortly after five";

/** An app shell that has rendered its chrome and none of its content. */
const UNRENDERED_SHELL = `<!doctype html><html lang="en"><head><title>News Alpha</title></head>
<body><div id="app"><header><a href="/">News Alpha</a></header><main></main></div></body></html>`;

function fixture(name) {
  return readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
}

function compare(html, url) {
  return compareReadings(new JSDOM(html, { url }).window.document, url);
}

test("an index page is measured as one, and both readings are accounted for", () => {
  const result = compare(fixture("news-homepage.html"), HOMEPAGE_URL);

  // The page is mostly the text of links to other pages — that is what makes it
  // an index, and it is the gate the index reading is even considered behind.
  assert.equal(result.arbiter.isLinkIndex, true);
  assert.ok(result.linkSharePercent > 80, `link share was ${result.linkSharePercent}%`);
  assert.equal(result.linkedWords + (result.visibleWords - result.linkedWords), result.visibleWords);

  // Both readings are present and countable, and the index one was built.
  assert.ok(result.arbiter.articleWords > 0);
  assert.ok(result.arbiter.indexWords > result.arbiter.articleWords);
  assert.equal(result.arbiter.picked, "index");

  // And the mirror that produced those numbers landed where readPage did, which
  // is the only thing that makes them evidence rather than a second opinion.
  assert.equal(result.arbiter.mirrorAgreesWithReadPage, true);
});

test("the production port reproduces the deceptive reading the PR is about", () => {
  const result = compare(fixture("news-homepage.html"), HOMEPAGE_URL);

  // main's reading of this page looks like a read — comfortably over the 200
  // floor, so nothing downstream could tell — and carries almost no words,
  // because what fills those characters is the URLs of links whose headline
  // Defuddle's trailing-heading cleanup deleted.
  assert.equal(result.production.extracted, true);
  assert.ok(result.production.chars > 200);
  assert.ok(result.production.words < 30, `production carried ${result.production.words} words`);

  // The PR's reading of the same page, on every axis the harness reports.
  assert.ok(result.pr.words > result.production.words);
  assert.ok(result.pr.chars > result.production.chars);
});

test("headline survival separates the two readings on an index page", () => {
  const result = compare(fixture("news-homepage.html"), HOMEPAGE_URL);

  assert.ok(result.headlineProbes.length > 0);
  assert.ok(result.headlineProbes.length <= 10, "probes are capped at ten");
  for (const probe of result.headlineProbes) {
    assert.ok(probe.split(/\s+/).length >= 4, `probe too short to be a headline: ${probe}`);
  }

  assert.equal(result.pr.headlinesSurvived, result.headlineProbes.length);
  assert.ok(
    result.production.headlinesSurvived < result.pr.headlinesSurvived,
    `production kept ${result.production.headlinesSurvived} of ${result.headlineProbes.length}`,
  );
});

test("an article page is left exactly as production reads it", () => {
  const result = compare(fixture("news-article.html"), ARTICLE_URL);

  // Not an index, so the index reading is never even built.
  assert.equal(result.arbiter.isLinkIndex, false);
  assert.equal(result.arbiter.indexWords, null);
  assert.equal(result.arbiter.picked, "article");
  assert.equal(result.arbiter.mirrorAgreesWithReadPage, true);

  // Identical on every axis: this is the "article pages untouched" half of the
  // PR's claim, stated as the numbers a results file would show.
  assert.equal(result.pr.chars, result.production.chars);
  assert.equal(result.pr.words, result.production.words);
  assert.equal(result.pr.headlinesSurvived, result.production.headlinesSurvived);
  assert.equal(result.pr.excerpt, result.production.excerpt);
  assert.ok(result.pr.excerpt.length > 0);
  assert.ok(compare(fixture("news-article.html"), ARTICLE_URL).pr.extracted);
});

test("a page that rendered nothing is recorded as read by neither", () => {
  const result = compare(UNRENDERED_SHELL, HOMEPAGE_URL);

  assert.equal(result.production.extracted, false);
  assert.equal(result.pr.extracted, false);
  // Nothing was read, so nothing is claimed: no chars, no words, no headlines.
  assert.deepEqual(
    [result.pr.chars, result.pr.words, result.pr.headlinesSurvived, result.pr.excerpt],
    [0, 0, 0, ""],
  );
  assert.equal(result.arbiter.picked, "none");
  assert.equal(result.arbiter.mirrorAgreesWithReadPage, true);
  // Too few words to be a rendered page — flagged, not silently dropped.
  assert.equal(result.consentWalled, true);
  assert.ok(result.consentSignals.some((signal) => signal.includes("visible words")));
});

test("a consent wall is flagged rather than skipped", () => {
  const walled = fixture("news-homepage.html").replace(
    "<body>",
    '<body><iframe id="sp_message_iframe_1" src="https://cdn.privacy-mgmt.com/index.html"></iframe>',
  );

  const result = compare(walled, HOMEPAGE_URL);

  assert.equal(result.consentWalled, true);
  assert.ok(result.consentSignals.some((signal) => signal.startsWith("consent iframe")));
  // Flagged AND measured — the row is still there to be read or overruled.
  assert.ok(result.pr.chars > 0);
});

test("a record survives the round trip the harness downloads it through", () => {
  const result = compare(fixture("news-homepage.html"), HOMEPAGE_URL);

  const roundTripped = JSON.parse(JSON.stringify({ ...result, kind: "index", pass: 2, loadTimedOut: false }));

  assert.deepEqual(roundTripped, { ...result, kind: "index", pass: 2, loadTimedOut: false });
  assert.equal(roundTripped.excerptLength, undefined);
  assert.ok(roundTripped.pr.excerpt.length <= 500, "excerpts are capped at 500 chars");
  assert.ok(roundTripped.production.excerpt.length <= 500, "excerpts are capped at 500 chars");
  assert.equal(typeof roundTripped.measuredAt, "string");
  assert.equal(roundTripped.title, "News Alpha — Home");
  assert.equal(roundTripped.requestedUrl, HOMEPAGE_URL);
});

/** The Defuddle these numbers were taken on.  Pinned exactly in package.json,
 *  because Defuddle's reading of this fixture changes by an order of magnitude
 *  between versions and a calibration that does not name its version is noise. */
const CALIBRATED_DEFUDDLE = "0.19.3";

test("the fixture numbers this branch is calibrated on have not moved", () => {
  // If this fails, Defuddle changed under the harness and every number in the
  // README's calibration table — and in any results file — needs re-taking.
  assert.equal(
    JSON.parse(readFileSync(new URL("../node_modules/defuddle/package.json", import.meta.url), "utf8")).version,
    CALIBRATED_DEFUDDLE,
  );

  const home = compare(fixture("news-homepage.html"), HOMEPAGE_URL);
  const article = compare(fixture("news-article.html"), ARTICLE_URL);

  // The homepage on 0.19.3: Defuddle returns the LEAD CARD ALONE — 204 chars,
  // barely over the 200 floor, one headline of ten. Not the 0.14 shape (2,026
  // chars of emptied links) but the same loss, and the PR's own docstring names
  // it as the defect's other shape.
  assert.deepEqual(
    [home.production.chars, home.production.words, home.production.headlinesSurvived],
    [204, 22, 1],
  );
  assert.deepEqual([home.pr.chars, home.pr.words, home.pr.headlinesSurvived], [3924, 249, 10]);

  // The article page: both readings identical, as on 0.14.
  assert.deepEqual([article.production.chars, article.production.words], [1435, 258]);
  assert.deepEqual([article.pr.chars, article.pr.words], [1435, 258]);
});

test("the site list is what the results header claims it is", () => {
  assert.equal(SITES.length, 21, "twenty news fronts plus one article control");
  assert.match(LIST_VERSION, /^\d{4}-\d{2}-\d{2}\.\d+$/);

  const hosts = SITES.map((site) => new URL(site.url).host);
  assert.equal(new Set(hosts).size, hosts.length, "one page per site");
  for (const site of SITES) {
    assert.equal(new URL(site.url).protocol, "https:");
  }
  assert.equal(SITES.filter((site) => site.kind === "article-control").length, 1);
});
