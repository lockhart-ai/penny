/**
 * Capturing each link card's own thumbnail (#3068).
 *
 * The regression this pins: a front page is read as its own link index, the
 * reply cites one of its articles, and no image was ever captured for the
 * cited address — the page's meta image is the front page's picture, so the
 * fallback attached an unrelated image or none.  The fixture is synthetic
 * (gazette.example, invented stories); what is real is the SHAPE — story cards
 * with plain srcs, srcsets, lazy placeholder pairs, a shared masthead, a tiny
 * icon, and a text-only card.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

import { captureLinkThumbnails } from "../dist/test/link_thumbnails.mjs";

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

test("an empty page yields no entries", () => {
  const doc = documentFrom("<!doctype html><html><body></body></html>", `${SITE}/`);
  assert.deepEqual(captureLinkThumbnails(doc), []);
});
