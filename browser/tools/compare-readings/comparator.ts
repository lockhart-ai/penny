/**
 * Reading one page BOTH ways and recording what each way got.
 *
 * The question this exists to answer, on real pages rather than fixtures: does
 * PR #1961's dual reading — the article body, or the page's own link index,
 * whichever carries more of the page's words — beat what ships on main today on
 * a JS-rendered news homepage, and does it leave article pages alone?
 *
 * The PR side is the REAL `readPage` from `src/content/page_text.ts`, not a
 * reimplementation, so the verdict recorded is the verdict that would ship.  The
 * arbitration's inner numbers come from the mirror in `page_metrics.ts`, and the
 * two are checked against each other on every page (`mirrorAgreesWithReadPage`).
 */

import { readPage } from "../../src/content/page_text.js";
import { detectConsentWall } from "./consent.js";
import {
  articleReading,
  defuddleReading,
  headlineProbes,
  linkShare,
  readLinkIndex,
  readableBody,
  richer,
  survivingHeadlines,
  type PageReading,
} from "./page_metrics.js";
import { MAX_CHARS, MIN_CONTENT_LENGTH, readProduction } from "./production_reading.js";
import type { ArbiterAccounting, PageComparison, ReadingMetrics } from "./types.js";

/** Enough of each reading to see what it is at a glance. */
const EXCERPT_CHARS = 500;

/** Read one page both ways and account for the difference. */
export function compareReadings(doc: Document, url: string): PageComparison {
  const body = readableBody(doc);
  const share = linkShare(body);
  const probes = headlineProbes(body);
  const defuddled = defuddleReading(doc, url);
  const article = articleReading(doc, defuddled);
  const index = share.isLinkIndex ? readLinkIndex(body) : null;
  const shipped = cap(readPage(doc, url));

  return {
    requestedUrl: url,
    finalUrl: doc.location?.href ?? url,
    title: doc.title,
    measuredAt: new Date().toISOString(),
    visibleWords: share.visibleWords,
    linkedWords: share.linkedWords,
    linkSharePercent: share.linkSharePercent,
    headlineProbes: probes,
    production: measure(cap(readProduction(defuddled)), wordsOf(article), probes),
    pr: measure(shipped, wordsOf(richer(article, index)), probes),
    arbiter: account(article, index, share.isLinkIndex, shipped),
    ...detectConsentWall(doc, share.visibleWords),
  };
}

/** What a reading got: whether it counted as a read, how much it carried, and
 *  how many of the page's own headlines came through it. */
function measure(text: string | null, words: number, probes: string[]): ReadingMetrics {
  return {
    extracted: text !== null,
    chars: text?.length ?? 0,
    // A reading that did not clear its floor carries nothing — reporting the
    // words its rejected text would have had is the reading's number, not this
    // one's, and would put a word count next to `extracted: false`.
    words: text === null ? 0 : words,
    headlinesSurvived: survivingHeadlines(text, probes),
    excerpt: (text ?? "").slice(0, EXCERPT_CHARS),
  };
}

/** The arbitration, as the mirror computes it — plus whether the mirror landed
 *  where the real `readPage` did, which is what makes the rest of it evidence. */
function account(
  article: PageReading | null,
  index: PageReading | null,
  isLinkIndex: boolean,
  actual: string | null,
): ArbiterAccounting {
  const winner = richer(article, index);
  const mirrored = winner !== null && winner.text.length >= MIN_CONTENT_LENGTH ? winner : null;
  return {
    isLinkIndex,
    articleWords: article?.words ?? null,
    articleChars: article?.text.length ?? null,
    indexWords: index?.words ?? null,
    indexChars: index?.text.length ?? null,
    picked: mirrored === null ? "none" : mirrored === index ? "index" : "article",
    mirrorAgreesWithReadPage: cap(mirrored?.text ?? null) === actual,
  };
}

function wordsOf(reading: PageReading | null): number {
  return reading?.words ?? 0;
}

/** The cap `extract()` applies to whatever a reading returned — applied to both
 *  readings, so the char counts are the ones that would actually ship. */
function cap(text: string | null): string | null {
  return text === null ? null : text.slice(0, MAX_CHARS);
}
