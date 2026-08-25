/**
 * Which Defuddle produced a run's numbers.
 *
 * Load-bearing, not bookkeeping: Defuddle's reading of a page changes
 * substantially between versions — on this repo's own homepage fixture, 0.14.0
 * returns 2,026 characters of emptied links and 0.19.3 returns 204 characters of
 * the lead card alone — so a results file that does not name its Defuddle is not
 * comparable with any other results file.
 *
 * Baked in at build time from the INSTALLED package, not from the range in
 * `package.json`: a caret range is a wish, and what read the page is whatever
 * `npm install` actually resolved.  Both are recorded so a mismatch is visible.
 */

declare const __DEFUDDLE_VERSION__: string;
declare const __DEFUDDLE_RANGE__: string;

/** The exact version bundled into this build, from node_modules. */
export const DEFUDDLE_VERSION = __DEFUDDLE_VERSION__;

/** What browser/package.json asked for. */
export const DEFUDDLE_RANGE = __DEFUDDLE_RANGE__;
