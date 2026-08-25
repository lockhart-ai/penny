/**
 * Reading a page (#1942) — the article body, or the page's own link index.
 *
 * The regression this pins: a JS-rendered news homepage was yielding ~1.6K chars
 * to the content script while comparable pages yielded 5-10K, and nothing could
 * tell, because 1.6K clears the length floor and the poller settles on it.  The
 * fixtures are synthetic (news-alpha.example, invented headlines); what is real
 * is the SHAPE — headings inside anchors, inline hydration state, and a page
 * whose text is almost all link text.
 *
 * Run through the same Defuddle + Turndown the extension bundles, over jsdom.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import Defuddle from "defuddle";
import TurndownService from "turndown";

import { readPage, MIN_CONTENT_LENGTH } from "../dist/test/page_text.mjs";

const SITE = "https://news-alpha.example";
const HOMEPAGE_URL = `${SITE}/`;
const ARTICLE_URL = `${SITE}/world/2036/harbour-bridge-reopens-after-refit`;

/** The floor a homepage read has to clear.  Deliberately ABOVE what the broken
 *  reading yields: Defuddle alone returns 2,026 characters of this page — which
 *  is why nothing noticed — and 22 words.  The read carries ~3.9K. */
const MIN_HOMEPAGE_YIELD = 3000;

/** Every headline the homepage fixture offers, with the page it points at — the
 *  content that was being lost.  Listed here rather than scraped from the
 *  fixture, so a fixture that quietly stopped offering one fails this file. */
const HOMEPAGE_HEADLINES = [
  ["Harbour bridge reopens after a two-year refit", "/world/2036/harbour-bridge-reopens-after-refit"],
  ["Dockside market changes hands after eighty years", "/business/2036/dockside-market-changes-hands"],
  ["Lantern festival draws a record crowd to the old quarter", "/world/2036/lantern-festival-draws-record-crowd"],
  ["Museum returns the borrowed mosaics a decade early", "/culture/2036/museum-returns-borrowed-mosaics"],
  ["Rowing club takes the estuary title on a restart", "/sport/2036/rowing-club-wins-on-a-restart"],
  ["Ferry operator adds a night sailing for the winter", "/business/2036/ferry-operator-adds-a-night-sailing"],
  ["Border crossing queues ease after the new lanes open", "/world/2036/border-crossing-queues-ease"],
  ["City orchestra names a conductor from within the ranks", "/culture/2036/city-orchestra-names-a-conductor"],
  ["What the refit really cost us, and who paid it", "/opinion/2036/what-the-refit-cost-us"],
  ["Velodrome hosts a junior series for the first time", "/sport/2036/velodrome-hosts-a-junior-series"],
  ["Warehouse district rezoned for housing after long inquiry", "/business/2036/warehouse-district-rezoned"],
  ["Observatory logs its quietest year for storms on record", "/world/2036/observatory-logs-a-quiet-year"],
  ["Bookshop reopens in the arcade under new owners", "/culture/2036/bookshop-reopens-in-the-arcade"],
  ["Lighthouse keeper's cottage listed after a public campaign", "/world/2036/lighthouse-keepers-cottage-listed"],
  ["Co-op bakery opens a second site on the north shore", "/business/2036/co-op-bakery-opens-a-second-site"],
  ["Harriers take the cross country in heavy going", "/sport/2036/harriers-take-the-cross-country"],
  ["Film festival adds a late strand for local shorts", "/culture/2036/film-festival-adds-a-late-strand"],
  ["The case for a second crossing, twenty years on", "/opinion/2036/the-case-for-a-second-crossing"],
  ["Salt marsh restoration completes ahead of schedule", "/world/2036/salt-marsh-restoration-completes"],
  ["Tram extension clears its final planning hurdle", "/business/2036/tram-extension-clears-final-hurdle"],
  ["Archive digitises the dock records going back to 1870", "/culture/2036/archive-digitises-the-dock-records"],
  ["Swimmers return to the tidal pool after the repairs", "/sport/2036/swimmers-return-to-the-tidal-pool"],
  ["Night buses run hourly again across the eastern routes", "/world/2036/night-buses-run-hourly-again"],
  ["Chandlery on the quay marks a century in the family", "/business/2036/chandlery-marks-a-century"],
  ["Who the quiet year on the water really benefited", "/opinion/2036/who-the-quiet-year-benefited"],
];

/** A sentence that exists only in the article fixture's body. */
const ARTICLE_SENTENCE = "The first vehicles rolled over the harbour bridge shortly after five";

/** An app shell that has rendered its chrome and none of its content — the state
 *  the poller exists to wait out. */
const UNRENDERED_SHELL = `<!doctype html><html lang="en"><head><title>News Alpha</title></head>
<body><div id="app"><div class="sc-9f2a1b"><header><a href="/">News Alpha</a></header>
<main class="sc-7b1e"></main></div></div></body></html>`;

function fixture(name) {
  return readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
}

function documentFrom(html, url) {
  const dom = new JSDOM(html, { url });
  return dom.window.document;
}

/** What Defuddle alone makes of a page — the reading the extension had before
 *  #1942, reproduced here so the diagnosis is asserted rather than asserted-of. */
function articleOnlyReading(doc, url) {
  const result = new Defuddle(doc.cloneNode(true), { url }).parse();
  if (!result.content) return { markdown: "", words: 0 };
  const markdown = new TurndownService({ headingStyle: "atx" }).turndown(result.content);
  return { markdown, words: result.wordCount };
}

test("the homepage fixture is the JS-rendered index shape the yield claims are about", () => {
  const doc = documentFrom(fixture("news-homepage.html"), HOMEPAGE_URL);

  // Every headline is a heading INSIDE an anchor — the shape Defuddle's
  // trailing-heading cleanup deletes.
  const headlineLinks = doc.querySelectorAll("article.sc-card a[href] h2, article.sc-card a[href] h3");
  assert.equal(headlineLinks.length, HOMEPAGE_HEADLINES.length);

  // The page carries inline hydration state, which must not count as its text.
  assert.ok((doc.querySelector("#__hydration__")?.textContent ?? "").length > 200);

  // And it is almost all link text — one standfirst and a rail heading aside.
  const bodyText = (doc.body.textContent ?? "").replace(/\s+/g, " ").trim();
  const linkText = [...doc.querySelectorAll('a[href]:not([href^="#"])')]
    .map((a) => (a.textContent ?? "").replace(/\s+/g, " ").trim())
    .join(" ");
  assert.ok(linkText.length > bodyText.length - linkText.length);
});

test("a JS-rendered homepage yields its headlines and where they point", () => {
  const doc = documentFrom(fixture("news-homepage.html"), HOMEPAGE_URL);

  const text = readPage(doc, HOMEPAGE_URL);

  assert.notEqual(text, null);
  assert.ok(
    text.length >= MIN_HOMEPAGE_YIELD,
    `homepage yielded ${text.length} chars, below the ${MIN_HOMEPAGE_YIELD} floor`,
  );
  for (const [headline, path] of HOMEPAGE_HEADLINES) {
    assert.ok(
      text.includes(`[${headline}](${SITE}${path})`),
      `missing headline link: ${headline}`,
    );
  }
});

test("the article-only reading kept the links and dropped their text", () => {
  const doc = documentFrom(fixture("news-homepage.html"), HOMEPAGE_URL);

  const before = articleOnlyReading(doc, HOMEPAGE_URL);

  // It looks like a read — comfortably over the floor, so nothing downstream
  // could tell — and it carries almost no words: what fills those characters is
  // the URLs of links whose headline Defuddle's trailing-heading cleanup
  // deleted.  This is the deceptive shape #1942 opened on.
  assert.ok(before.markdown.length > MIN_CONTENT_LENGTH);
  assert.ok(before.words < 30, `article-only reading carried ${before.words} words`);
  const kept = HOMEPAGE_HEADLINES.filter(([headline]) => before.markdown.includes(headline));
  assert.ok(kept.length <= 1, `article-only reading kept ${kept.length} headlines`);
  const [, strippedPath] = HOMEPAGE_HEADLINES[HOMEPAGE_HEADLINES.length - 1];
  assert.ok(before.markdown.includes(`](${SITE}${strippedPath})`));
});

test("a short grid, where Defuddle returns the lead card alone, reads whole", () => {
  const doc = documentFrom(fixture("news-homepage.html"), HOMEPAGE_URL);
  // The same defect's other shape: with few enough cards the lead card outscores
  // the grid and Defuddle returns it by itself, so the rest of the page is gone
  // rather than emptied.  Trimming the fixture's grid is what reaches it.
  const cards = [...doc.querySelectorAll(".sc-grid .sc-card")];
  for (const card of cards.slice(3)) card.remove();

  const before = articleOnlyReading(doc, HOMEPAGE_URL);
  const text = readPage(doc, HOMEPAGE_URL);

  assert.ok(before.markdown.length < MIN_CONTENT_LENGTH * 2);
  for (const [headline, path] of HOMEPAGE_HEADLINES.slice(0, 4)) {
    assert.ok(text.includes(`[${headline}](${SITE}${path})`), `missing headline link: ${headline}`);
  }
});

test("an article page still reads as its article, not as its navigation", () => {
  const doc = documentFrom(fixture("news-article.html"), ARTICLE_URL);

  const text = readPage(doc, ARTICLE_URL);

  assert.notEqual(text, null);
  assert.ok(text.includes(ARTICLE_SENTENCE));
  // The page's chrome links are present in force and still lose — the index
  // reading is never built for a page whose text is mostly its body.
  assert.ok(!text.includes(`[About us](${SITE}/about)`));
  assert.ok(!text.includes(`[Tide tables for the week ahead](`));
});

test("a hidden mega-menu decides nothing and is never emitted", () => {
  const doc = documentFrom(fixture("news-article.html"), ARTICLE_URL);
  // A collapsed menu of links nobody can see. Left in, it is enough link text to
  // tip an article page over the majority `isLinkIndex` measures — and the index
  // reading would then hand back the menu as though it were the page.
  const menu = doc.createElement("div");
  menu.setAttribute("hidden", "");
  menu.innerHTML = Array.from(
    { length: 120 },
    (_, i) => `<a href="${SITE}/menu/${i}">Hidden mega-menu entry number ${i}</a>`,
  ).join("");
  doc.body.append(menu);

  const text = readPage(doc, ARTICLE_URL);

  assert.ok(text.includes(ARTICLE_SENTENCE));
  assert.ok(!text.includes("Hidden mega-menu entry"));
});

test("a shell that has rendered no content is not a read", () => {
  const doc = documentFrom(UNRENDERED_SHELL, HOMEPAGE_URL);

  // null is what makes the caller report extracted: false, so the poller waits
  // for the page instead of settling on its chrome.
  assert.equal(readPage(doc, HOMEPAGE_URL), null);
  assert.equal(MIN_CONTENT_LENGTH, 200);
});
