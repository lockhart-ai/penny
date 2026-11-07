/**
 * Capturing each link card's own thumbnail (#3068).
 *
 * The regression this pins: a front page is read as its own link index, the
 * reply cites one of its articles, and no image was ever captured for the
 * cited address — the page's meta image is the front page's picture, so the
 * fallback attached an unrelated image or none.  It also pins #3114: a
 * srcset candidate's URL may itself hold commas (some CDNs encode an image
 * id in the path), and a search-style card's only image is often its site's
 * favicon, never its own picture.  The fixture is synthetic (gazette.example,
 * invented stories); what is real is the SHAPE — story cards with plain
 * srcs, srcsets, lazy placeholder pairs, comma-bearing srcset paths, a
 * shared masthead, a tiny icon, a favicon-only card, and a text-only card.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

import { captureLinkThumbnails, largestSrcsetCandidate } from "../dist/test/link_thumbnails.mjs";

const SITE = "https://gazette.example";

function fixture(name) {
  return readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
}

function documentFrom(html, url) {
  const dom = new JSDOM(html, { url });
  return dom.window.document;
}

/** The capture over the cards fixture — one page exercising every shape. */
function captured() {
  return captureLinkThumbnails(documentFrom(fixture("front_page_cards.html"), `${SITE}/`));
}

test("pairs each distinct link with its own card's picture, in document order", () => {
  assert.deepEqual(captured(), [
    {
      url: `${SITE}/stories/first-frost-sets-in`,
      image_url: "https://gazette.example/imgs/first-frost.jpg",
      title: "First frost sets in across the valley",
    },
    {
      url: `${SITE}/stories/tram-works-finish-early`,
      image_url: "https://gazette.example/imgs/tram-large.jpg",
      title: "Tram works finish two weeks early",
    },
    {
      url: `${SITE}/stories/lighthouse-keeper-retires`,
      image_url: "https://gazette.example/imgs/lighthouse.jpg",
      title: "Lighthouse keeper retires after forty winters",
    },
    {
      url: `${SITE}/stories/riverside-market-moves-to-the-quay`,
      image_url: "https://gazette.example/imgs/market-960.jpg",
      title: "Riverside market moves to the old quay",
    },
    {
      url: `${SITE}/stories/northern-lights-reviewed`,
      image_url: "https://gazette.example/imgs/northern-lights.jpg",
      title: "Northern Lights, reviewed",
    },
    {
      url: `${SITE}/stories/harbor-crane-replaced`,
      image_url: "https://gazette.example/imgs/1.100,200/crane-large.jpg",
      title: "Harbor crane replaced ahead of schedule",
    },
    {
      url: `${SITE}/stories/ferry-schedule-extended`,
      image_url: "https://gazette.example/imgs/2.50,60/ferry-960.jpg",
      title: "Ferry schedule extended into the new year",
    },
    {
      url: `${SITE}/stories/author-byline-photo-story`,
      image_url: "https://gazette.example/imgs/byline-story.jpg",
      title: "A story whose byline avatar sits ahead of its own photo",
    },
  ]);
});

test("resolves the srcset's largest candidate, not the src's default", () => {
  const entry = captured().find((e) => e.url === `${SITE}/stories/tram-works-finish-early`);
  assert.equal(entry.image_url, "https://gazette.example/imgs/tram-large.jpg");
});

test("resolves a lazy card's placeholder src to its data-src", () => {
  const entry = captured().find((e) => e.url === `${SITE}/stories/lighthouse-keeper-retires`);
  assert.equal(entry.image_url, "https://gazette.example/imgs/lighthouse.jpg");
});

test("resolves a lazy card's data-srcset to its largest candidate", () => {
  const entry = captured().find((e) => e.url === `${SITE}/stories/riverside-market-moves-to-the-quay`);
  assert.equal(entry.image_url, "https://gazette.example/imgs/market-960.jpg");
});

test("drops the image several different links share — the site's logo", () => {
  const urls = new Set(captured().map((e) => e.url));
  for (const slug of ["tidal-pool-reopens", "night-buses-run-hourly", "chandlery-marks-a-century"]) {
    assert.equal(urls.has(`${SITE}/stories/${slug}`), false, `${slug} shared the masthead — dropped`);
  }
  assert.equal(captured().some((e) => e.image_url.includes("masthead")), false);
});

test("skips a known-tiny icon and a text-only card", () => {
  const urls = new Set(captured().map((e) => e.url));
  assert.equal(urls.has(`${SITE}/stories/weather-brief`), false, "16x16 icon card dropped");
  assert.equal(urls.has(`${SITE}/stories/an-editorial-note`), false, "text-only card dropped");
});

test("one entry per distinct url — the first card's picture wins", () => {
  const frost = captured().filter((e) => e.url === `${SITE}/stories/first-frost-sets-in`);
  assert.equal(frost.length, 1);
  assert.equal(frost[0].image_url, "https://gazette.example/imgs/first-frost.jpg");
});

test("keeps a srcset candidate's URL intact when its own path holds commas", () => {
  const entry = captured().find((e) => e.url === `${SITE}/stories/harbor-crane-replaced`);
  assert.equal(entry.image_url, "https://gazette.example/imgs/1.100,200/crane-large.jpg");
});

test("keeps a lazy data-srcset candidate's URL intact when its own path holds commas", () => {
  const entry = captured().find((e) => e.url === `${SITE}/stories/ferry-schedule-extended`);
  assert.equal(entry.image_url, "https://gazette.example/imgs/2.50,60/ferry-960.jpg");
});

test("drops a card whose only image is a favicon proxy", () => {
  const urls = new Set(captured().map((e) => e.url));
  assert.equal(urls.has(`${SITE}/stories/search-result-favicon-card`), false);
});

test("drops a card whose only image is a legacy .ico", () => {
  const urls = new Set(captured().map((e) => e.url));
  assert.equal(urls.has(`${SITE}/stories/legacy-icon-only-card`), false);
});

test("skips a byline avatar ahead of the card's own photo", () => {
  const entry = captured().find((e) => e.url === `${SITE}/stories/author-byline-photo-story`);
  assert.equal(entry.image_url, "https://gazette.example/imgs/byline-story.jpg");
});

test("largestSrcsetCandidate picks the largest candidate even when a URL holds commas", () => {
  const set =
    "https://img.example-cdn/ais/1.5716272,1793914261251/full/max/0/default.jpg?w=800 800w, " +
    "https://img.example-cdn/ais/small.jpg 320w, " +
    "https://img.example-cdn/ais/medium.jpg 640w";
  assert.equal(
    largestSrcsetCandidate(set),
    "https://img.example-cdn/ais/1.5716272,1793914261251/full/max/0/default.jpg?w=800",
  );
});

test("largestSrcsetCandidate ends a bare candidate at its trailing comma, not at the next candidate's descriptor", () => {
  assert.equal(largestSrcsetCandidate("a.jpg, b.jpg 2x"), "b.jpg");
});

test("largestSrcsetCandidate parses a candidate with no space after its comma the same way", () => {
  assert.equal(largestSrcsetCandidate("a.jpg 1x,b.jpg 2x"), "b.jpg");
});

test("largestSrcsetCandidate reads two bare, comma-joined candidates as one URL — a URL runs to whitespace, per the grammar", () => {
  assert.equal(largestSrcsetCandidate("a.jpg,b.jpg"), "a.jpg,b.jpg");
});

test("an empty page yields no entries", () => {
  const doc = documentFrom("<!doctype html><html><body></body></html>", `${SITE}/`);
  assert.deepEqual(captureLinkThumbnails(doc), []);
});
