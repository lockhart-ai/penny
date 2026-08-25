/**
 * The reading that ships on `main` TODAY — naive Defuddle, Turndown, a 200-char
 * floor.  The baseline PR #1961 has to beat.
 *
 * Ported from `browser/src/content/extract_text.ts` as of main @ 3fb0adb2 —
 * `extractWithDefuddle` there is this file's `defuddleReading` + the floor in
 * `readProduction`, split only so the article half can be shared (see below).
 * Copied rather than imported because the PR branch has already rewritten that
 * file: importing it would compare the PR against itself.
 *
 * Two things main does that are deliberately NOT here, and neither changes what
 * a news homepage yields:
 *   - the XML/RSS branch (`document.contentType` carrying xml/rss), which an
 *     HTML news page never takes;
 *   - the kagi.com readiness locator, which gates extraction on a hostname none
 *     of the measured sites has.
 * The 50,000-char cap main applies in `extract()` is applied by the comparator,
 * to BOTH readings, so the char counts are the ones that would actually ship.
 */

import Defuddle from "defuddle";
import TurndownService from "turndown";

const turndown = new TurndownService({ headingStyle: "atx" });

/** main's floor: below this, `extracted` is false and the poller tries again. */
export const MIN_CONTENT_LENGTH = 200;

/** main's cap, applied in `extract()` to whatever the reading returned. */
export const MAX_CHARS = 50_000;

/** What Defuddle makes of the page, before any floor.  Returned in both forms
 *  because the two readings need different halves of it: the production reading
 *  measures the markdown, the PR's article reading counts words off the HTML.
 *  One parse serves both — main and the PR clone-then-parse identically, so
 *  parsing twice would only cost time. */
export interface DefuddleReading {
  markdown: string;
  contentHtml: string;
}

/** Defuddle's own extraction of the page, rendered to markdown.  Defuddle is
 *  handed a clone so its cleanup never touches the live page. */
export function defuddleReading(doc: Document, url: string): DefuddleReading | null {
  const clone = doc.cloneNode(true) as Document;
  const result = new Defuddle(clone, { url }).parse();
  if (!result.content) return null;
  return { markdown: turndown.turndown(result.content), contentHtml: result.content };
}

/** main's reading of the page: Defuddle's markdown if it clears the floor. */
export function readProduction(reading: DefuddleReading | null): string | null {
  if (reading === null) return null;
  const text = reading.markdown;
  if (text && text.length >= MIN_CONTENT_LENGTH) return text;
  return null;
}
