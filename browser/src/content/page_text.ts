/**
 * Reading a page's text — as an ARTICLE, or as an INDEX of other pages.
 *
 * Defuddle is an *article* extractor: it scores the document for the one block
 * holding a story and discards everything else, then cleans that block up.  On a
 * page that has no story — a news homepage, a section front, a search-results
 * page — both halves of that work against us, and the failure is silent because
 * what comes back still clears the length floor.
 *
 * Measured on the synthetic homepage in `test/fixtures/news-homepage.html`
 * (25 headline links, each a heading wrapped in an anchor):
 *
 *   - Defuddle picks the right container (`main`) and then its own cleanup step
 *     — "Removed trailing headings: 24" in its debug log — deletes every card's
 *     headline, because each heading is the last thing in its card, which on an
 *     article is a dangling section header and on an index card is the headline
 *     itself.  What survives is two dozen bare `[](url)` links: 2,026 characters
 *     of markdown carrying 22 words.
 *   - Trim the grid and it fails the other way: the lead card outscores the
 *     rest, Defuddle returns that card alone, and the other headlines are gone
 *     entirely rather than emptied.
 *
 * Either way the page's actual content — the headlines and where they point —
 * is lost, and nothing downstream can tell: `extracted` is true, the text is
 * comfortably over the floor, and the poller settles on it.  That is the
 * ~1.6K-char read a national news homepage was yielding while comparable pages
 * yielded 5-10K.
 *
 * So a page gets read BOTH ways and the richer reading wins.  Two structural
 * rules, neither of them a tuned number:
 *
 *   1. The index reading is only *considered* for a page whose own text is
 *      mostly link text (`isLinkIndex`) — that is what an index of other pages
 *      IS, and it is a majority test, not a threshold.  Measured, a homepage
 *      sits at 96-98% and an article page at 11-31%, so a long-nav article page
 *      can never be re-read as its own navigation.
 *   2. Even then it only wins if it carries MORE of the page's own words
 *      (`richer`) — counted the same way for both readings, so the comparison
 *      cannot be skewed by two different definitions of a word.
 *
 * An article page therefore reads exactly as it did before: its index reading is
 * never built, and if it were it would lose.
 */

import Defuddle from "defuddle";
import TurndownService from "turndown";

const turndown = new TurndownService({ headingStyle: "atx" });

/** Below this a reading is not a read of the page at all — the caller reports
 *  `extracted: false` and the poller tries again, which is how a page still
 *  rendering its content gets a second chance. */
export const MIN_CONTENT_LENGTH = 200;

/** Elements whose text is not the page's readable content: code and markup, and
 *  anything the page itself marks hidden.
 *
 *  Both halves are load-bearing.  A JS-heavy page carries kilobytes of inline
 *  hydration state, and counting that as the page's own text would drown the link
 *  share `isLinkIndex` measures.  A collapsed mega-menu carries hundreds of links
 *  nobody can see, and counting those would let a page's *chrome* decide that an
 *  ordinary article is an index of other pages — and then emit those links as if
 *  they were what the page offers.  Defuddle drops hidden elements on the article
 *  path; the index reading does not go through Defuddle, so it drops them here.
 *
 *  Declared hidden, not computed hidden: `getComputedStyle` would catch a
 *  stylesheet-hidden menu too, but it is per-element work over the whole document
 *  on every poll, and what it buys is a wider net on the same class of thing. */
const NON_CONTENT_SELECTOR = 'script, style, noscript, template, [hidden], [aria-hidden="true"]';

/** Links this page offers to other pages.  A same-document fragment goes
 *  nowhere new, so it is not an index entry. */
const INDEX_LINK_SELECTOR = 'a[href]:not([href^="#"])';

const WEB_URL = /^https?:\/\//;
const WHITESPACE_RUN = /\s+/g;
const WHITESPACE = /\s+/;

/** How far from the Unix epoch a timestamp may sit and still be one — a day
 *  each way, because which side of midnight UTC the epoch lands on depends on
 *  the timezone that rendered it. */
const EPOCH_WINDOW_MS = 86_400_000;

/** The date-time string format the language standard defines `Date.parse` for.
 *  Only a string of this shape reaches `Date.parse` — the one parse every
 *  engine agrees on; anything else is engine-specific territory (V8 parses a
 *  bare `0` as the year 2000, SpiderMonkey refuses it), so it never decides. */
const ISO_DATETIME =
  /^\d{4}(-\d{2}(-\d{2})?)?(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?$/;

/** How a reading won — ARTICLE (Defuddle's own extraction) or LINK_LIST (the page's
 *  own index of other pages).  Wire-level name is `PageReading` (`protocol.ts`,
 *  mirrored by `constants.PageReading` on the Python side); this file's internal
 *  type is named `Reading` instead, to tell a page's OWN tagged result apart from
 *  that wire literal without colliding on the name. */
type ReadingKind = "article" | "link_list";

/** One way of reading a page: the text it yields, how much of the page's own
 *  text that text carries, and which reading produced it.  The word count is
 *  what the two readings are compared on — markdown length would count a bare
 *  `[](url)` link's URL as content, which is the exact lie this module exists to
 *  correct; `kind` is what #2388 carries onward past that comparison. */
interface Reading {
  text: string;
  words: number;
  kind: ReadingKind;
}

/** The page's text and which reading won: whichever carries more of the page's
 *  own words.  `reading` is #2388's structural mark — set here, once, where the
 *  live DOM can actually be read — carried unchanged through `extract_text.ts`,
 *  the wire, and the browse tool into the extractor's own input. */
export function readPage(doc: Document, url: string): { text: string; reading: ReadingKind } | null {
  const body = readableBody(doc);
  const index = isLinkIndex(body) ? readLinkIndex(body) : null;
  const reading = richer(readArticle(doc, url), index);
  if (reading === null || reading.text.length < MIN_CONTENT_LENGTH) return null;
  return { text: reading.text, reading: reading.kind };
}

/** The page read as an ARTICLE — Defuddle's own extraction, rendered to
 *  markdown.  Defuddle is handed a clone so its cleanup never touches the live
 *  page under the user. */
function readArticle(doc: Document, url: string): Reading | null {
  const clone = doc.cloneNode(true) as Document;
  stripEpochTimestamps(clone);
  const result = new Defuddle(clone, { url }).parse();
  if (!result.content) return null;
  return {
    text: turndown.turndown(result.content),
    words: htmlWords(doc, result.content),
    kind: "article",
  };
}

/** The page read as an INDEX — one markdown link per distinct page it points at,
 *  in document order.  The line shape is the one a standalone link takes in
 *  Defuddle's own markdown, so nothing downstream has to learn a second form. */
function readLinkIndex(body: HTMLElement): Reading | null {
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
  return lines.length > 0 ? { text: lines.join("\n"), words, kind: "link_list" } : null;
}

/** Is most of this page's text the text of links to other pages?  That is what
 *  an index page is, stated structurally: a homepage or section front is a list
 *  of somewhere-elses, while a page with a body is mostly its body. */
function isLinkIndex(body: HTMLElement): boolean {
  const total = countWords(body.textContent ?? "");
  let linked = 0;
  for (const anchor of body.querySelectorAll<HTMLAnchorElement>(INDEX_LINK_SELECTOR)) {
    linked += countWords(anchor.textContent ?? "");
  }
  return linked > total - linked;
}

/** The reading that carries more of the page's own words; either may be absent. */
function richer(article: Reading | null, index: Reading | null): Reading | null {
  if (article === null) return index;
  if (index === null) return article;
  return index.words > article.words ? index : article;
}

/** The page's body as a reader sees it — code, markup and anything the page marks
 *  hidden removed.  A detached clone, so the live page is untouched. */
function readableBody(doc: Document): HTMLElement {
  const body = doc.body.cloneNode(true) as HTMLElement;
  for (const element of body.querySelectorAll(NON_CONTENT_SELECTOR)) element.remove();
  stripEpochTimestamps(body);
  return body;
}

/** Timestamps the page itself rendered as the Unix epoch are not the page's
 *  content.  An empty or zero timestamp its server or script formatted comes
 *  back as "January 1, 1970" (or "December 31, 1969") — measured on a national
 *  news front page, where every story card is one anchor carrying
 *  `<time class="timeStamp" dateTime="1970-01-01T00:00:00.001Z">January 1,
 *  1970</time>` beside the headline, so the date glued itself onto every link
 *  label.  `<time>` is the element HTML gives timestamps, so it is the only one
 *  dropped, and it is decided machine-readably: by its raw `datetime`
 *  attribute when it is the standardised ISO shape (the one form every engine
 *  parses the same) — or, when the attribute is absent, a zero, or some other
 *  shape, by the element's own text, which a `<time>` holds only its date of.
 *  A `<span>` styled as a timestamp is left alone: its text is free text,
 *  whose parsing is engine-specific, and reading every element's whole subtree
 *  to find dates would be quadratic on a big page.  Runs on the clones; the
 *  live page is untouched. */
function stripEpochTimestamps(root: ParentNode): void {
  for (const element of root.querySelectorAll("time")) {
    if (isEpochTime(element)) element.remove();
  }
}

/** Does this `<time>` contribute nothing but an epoch timestamp?  Empty
 *  elements contribute nothing at all and are left alone. */
function isEpochTime(element: Element): boolean {
  const stamp = element.getAttribute("datetime");
  // A machine-written ISO attribute decides outright; anything else — absent,
  // zero, or not the standardised shape — falls through to the element's own
  // text, which a `<time>` holds only its date of.
  if (stamp !== null && ISO_DATETIME.test(stamp)) return isEpochInstant(Date.parse(stamp));
  const text = (element.textContent ?? "").trim();
  return text !== "" && isEpochInstant(Date.parse(text));
}

/** Is this parsed instant within a day of the Unix epoch?  A page renders its
 *  own empty timestamp as the epoch in the viewer's timezone — "December 31,
 *  1969" west of UTC, "January 1, 1970" east of it — so the window spans a day
 *  each way. */
function isEpochInstant(parsed: number): boolean {
  return !Number.isNaN(parsed) && Math.abs(parsed) <= EPOCH_WINDOW_MS;
}

/** How many words of plain text an HTML fragment carries.  Parsed into an INERT
 *  document (no browsing context, so nothing loads and nothing runs) rather than
 *  counted off the markdown, where a link's URL would read as words. */
function htmlWords(doc: Document, html: string): number {
  const inert = doc.implementation.createHTMLDocument("");
  inert.body.innerHTML = html;
  return countWords(inert.body.textContent ?? "");
}

function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(WHITESPACE).length;
}

function normalize(text: string): string {
  return text.replace(WHITESPACE_RUN, " ").trim();
}
