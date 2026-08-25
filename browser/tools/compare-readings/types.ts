/**
 * The record one page produces — the shape of every object in
 * `compare-results.json`.  Evidence for #1942 / PR #1961, nothing else reads it.
 */

/** What one way of reading the page produced. */
export interface ReadingMetrics {
  /** Did this reading clear its floor and count as a read at all? */
  extracted: boolean;
  chars: number;
  /** Words of the page's own text the reading carries — a link's URL is not a
   *  word, which is the whole point of counting rather than measuring length.
   *  Counted off the HTML the reading came from, so it comes from the mirror
   *  described in `page_metrics.ts` and means what it says only while
   *  `mirrorAgreesWithReadPage` is true. */
  words: number;
  /** How many of `headlineProbes` survive into this reading's text. */
  headlinesSurvived: number;
  /** The first EXCERPT_CHARS characters, so a number can be eyeballed. */
  excerpt: string;
}

/** The PR's own arbitration, recomputed here so the decision is visible rather
 *  than inferred, plus the cross-check that says whether it can be trusted. */
export interface ArbiterAccounting {
  /** Is most of this page's text the text of links to other pages? */
  isLinkIndex: boolean;
  articleWords: number | null;
  articleChars: number | null;
  /** null when the page is not a link index — the reading is never built. */
  indexWords: number | null;
  indexChars: number | null;
  picked: "article" | "index" | "none";
  /** False means this row's `picked` / index numbers are NOT evidence: the
   *  mirror below disagreed with the real `readPage`, so read `pr` only. */
  mirrorAgreesWithReadPage: boolean;
}

/** One reading of one page at one moment. */
export interface PageComparison {
  requestedUrl: string;
  finalUrl: string;
  title: string;
  measuredAt: string;
  /** Words in the page's readable body — script, style and declared-hidden
   *  elements removed.  The denominator for `linkSharePercent`. */
  visibleWords: number;
  linkedWords: number;
  linkSharePercent: number;
  /** Visible link labels of >= HEADLINE_MIN_WORDS words, document order, capped
   *  at HEADLINE_PROBE_LIMIT — a structural stand-in for "the page's own
   *  headlines".  The denominator of both `headlinesSurvived` counts. */
  headlineProbes: string[];
  /** What ships on main today: naive Defuddle -> Turndown -> 200-char floor. */
  production: ReadingMetrics;
  /** What PR #1961 would ship: `readPage` — article or link index, arbitrated. */
  pr: ReadingMetrics;
  arbiter: ArbiterAccounting;
  /** Heuristic, and reported next to the raw numbers that drove it so the
   *  judgement can be re-made by eye rather than taken on faith. */
  consentWalled: boolean;
  consentSignals: string[];
}

/** What the harness knows about a page that the page itself cannot say. */
export interface VisitContext {
  /** "index" for the twenty news fronts, "article-control" for the one page
   *  that is a story — the half of the PR's claim a list of fronts can't test. */
  kind: string;
  /** 1 on settle, 2 ten seconds later — hydration and consent move things. */
  pass: number;
  /** The tab never reported `complete`.  Measured anyway — a page that loads
   *  forever is still a page the extension would try to read — but flagged. */
  loadTimedOut: boolean;
}

/** A page as the harness recorded it. */
export type MeasuredPage = PageComparison & VisitContext;

/** A page the harness could not measure — recorded, never dropped. */
export type FailedPage = VisitContext & {
  requestedUrl: string;
  measuredAt: string;
  error: string;
};

/** The whole run, as downloaded. */
export interface RunReport {
  harness: string;
  branch: string;
  doNotMerge: string;
  comparing: string;
  runStartedAt: string;
  runFinishedAt: string;
  userAgent: string;
  listVersion: string;
  siteCount: number;
  settleMs: number;
  hydrationDelayMs: number;
  pages: (MeasuredPage | FailedPage)[];
}
