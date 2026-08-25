/**
 * Measuring a page, and mirroring PR #1961's arbitration.
 *
 * `readPage` exports only its verdict — the winning text — so the numbers the
 * verdict was made on (the link share, the two readings' word counts, which one
 * won) are not observable from outside.  The block marked MIRROR below is copied
 * verbatim from `src/content/page_text.ts` on this branch so those numbers can
 * be reported.
 *
 * A copy can drift from what it mirrors, so it is never trusted on its word:
 * the comparator runs the real `readPage` alongside it on every page and records
 * `mirrorAgreesWithReadPage`.  If that is ever false, the mirror's numbers are
 * not evidence and the row says so.
 */

import { defuddleReading, type DefuddleReading } from "./production_reading.js";

// ---------------------------------------------------------------------------
// MIRROR — verbatim from src/content/page_text.ts (see file header)
// ---------------------------------------------------------------------------

const NON_CONTENT_SELECTOR = 'script, style, noscript, template, [hidden], [aria-hidden="true"]';
const INDEX_LINK_SELECTOR = 'a[href]:not([href^="#"])';
const WEB_URL = /^https?:\/\//;
const WHITESPACE_RUN = /\s+/g;
const WHITESPACE = /\s+/;

export interface PageReading {
  text: string;
  words: number;
}

/** The page's body as a reader sees it — code, markup and anything the page
 *  marks hidden removed.  A detached clone, so the live page is untouched. */
export function readableBody(doc: Document): HTMLElement {
  const body = doc.body.cloneNode(true) as HTMLElement;
  for (const element of body.querySelectorAll(NON_CONTENT_SELECTOR)) element.remove();
  return body;
}

/** The page read as an INDEX — one markdown link per distinct page it points at,
 *  in document order. */
export function readLinkIndex(body: HTMLElement): PageReading | null {
  const seen = new Set<string>();
  const lines: string[] = [];
  let words = 0;
  for (const anchor of body.querySelectorAll<HTMLAnchorElement>(INDEX_LINK_SELECTOR)) {
    const label = normalize(anchor.textContent ?? "");
    const href = anchor.href;
    if (!label || !WEB_URL.test(href)) continue;
    const line = `[${label}](${href})`;
    if (seen.has(line)) continue;
    seen.add(line);
    lines.push(line);
    words += countWords(label);
  }
  return lines.length > 0 ? { text: lines.join("\n"), words } : null;
}

/** How many words of plain text an HTML fragment carries. */
export function htmlWords(doc: Document, html: string): number {
  const inert = doc.implementation.createHTMLDocument("");
  inert.body.innerHTML = html;
  return countWords(inert.body.textContent ?? "");
}

export function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(WHITESPACE).length;
}

export function normalize(text: string): string {
  return text.replace(WHITESPACE_RUN, " ").trim();
}

// ---------------------------------------------------------------------------
// Measurement — this harness's own, not part of the PR
// ---------------------------------------------------------------------------

/** A link label this short is navigation ("Sport", "Sign in"), not a headline. */
const HEADLINE_MIN_WORDS = 4;

/** Enough probes to tell "kept them all" from "kept one", few enough to read. */
const HEADLINE_PROBE_LIMIT = 10;

/** The link-share accounting `isLinkIndex` decides on, kept as raw counts so the
 *  majority test can be re-checked by eye rather than believed. */
export interface LinkShare {
  visibleWords: number;
  linkedWords: number;
  linkSharePercent: number;
  isLinkIndex: boolean;
}

export function linkShare(body: HTMLElement): LinkShare {
  const visibleWords = countWords(body.textContent ?? "");
  let linkedWords = 0;
  for (const anchor of body.querySelectorAll<HTMLAnchorElement>(INDEX_LINK_SELECTOR)) {
    linkedWords += countWords(anchor.textContent ?? "");
  }
  return {
    visibleWords,
    linkedWords,
    linkSharePercent: visibleWords === 0 ? 0 : round(100 * (linkedWords / visibleWords)),
    // The mirrored `isLinkIndex`: a majority test, stated as the PR states it.
    isLinkIndex: linkedWords > visibleWords - linkedWords,
  };
}

/** The page's own headlines, structurally: the visible link labels long enough
 *  to be a sentence rather than a nav item, in document order.  Taken off the
 *  readable body, so a collapsed mega-menu contributes none of them. */
export function headlineProbes(body: HTMLElement): string[] {
  const probes: string[] = [];
  const seen = new Set<string>();
  for (const anchor of body.querySelectorAll<HTMLAnchorElement>(INDEX_LINK_SELECTOR)) {
    const label = normalize(anchor.textContent ?? "");
    if (countWords(label) < HEADLINE_MIN_WORDS || seen.has(label)) continue;
    seen.add(label);
    probes.push(label);
    if (probes.length === HEADLINE_PROBE_LIMIT) break;
  }
  return probes;
}

/** How many probes survive into a reading.  Whitespace is collapsed on both
 *  sides first: the two readings wrap lines differently, and a headline broken
 *  over a newline in one of them did survive. */
export function survivingHeadlines(text: string | null, probes: string[]): number {
  if (text === null) return 0;
  const haystack = normalize(text);
  return probes.filter((probe) => haystack.includes(probe)).length;
}

/** The PR's article reading, off the shared Defuddle parse. */
export function articleReading(
  doc: Document,
  reading: DefuddleReading | null,
): PageReading | null {
  if (reading === null) return null;
  return { text: reading.markdown, words: htmlWords(doc, reading.contentHtml) };
}

/** The reading that carries more of the page's own words; either may be absent. */
export function richer(article: PageReading | null, index: PageReading | null): PageReading | null {
  if (article === null) return index;
  if (index === null) return article;
  return index.words > article.words ? index : article;
}

/** Defuddle re-exported under the name the comparator uses it by, so the
 *  comparator has one place to get every reading from. */
export { defuddleReading };

function round(value: number): number {
  return Math.round(value * 10) / 10;
}
