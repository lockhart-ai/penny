/**
 * The paste-in build — the ad-hoc spot-check path.
 *
 * Open DevTools on any page, paste the whole of `dist/compare/compare-snippet.js`
 * into the console, press enter: the comparison is logged as an object you can
 * expand, and copied to the clipboard as JSON if the console offers `copy()`.
 * Same comparator the harness runs, so a spot-check and a run are comparable.
 *
 * Firefox's console refuses pasted input until you type `allow pasting` once.
 */

import { compareReadings } from "./comparator.js";

/** DevTools' clipboard helper. Not part of the page, so it may not be there —
 *  a content script or a non-DevTools console has no `copy`. */
declare const copy: unknown;

const comparison = compareReadings(document, location.href);
const json = JSON.stringify(comparison, null, 2);

console.log("[compare-readings]", comparison);

if (typeof copy === "function") {
  (copy as (value: string) => void)(json);
  console.log("[compare-readings] JSON copied to the clipboard");
} else {
  console.log(json);
}
