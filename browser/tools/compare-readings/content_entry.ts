/**
 * What the harness injects into each page.
 *
 * `browser.tabs.executeScript` hands back the value of the last expression the
 * injected script evaluated, so the build wraps this in an IIFE that RETURNS the
 * comparison — the same trick `build-content.mjs` plays for the real content
 * script.  Nothing is posted, nothing is stored: the page is measured and the
 * measurement travels back as the return value.
 */

import { compareReadings } from "./comparator.js";

compareReadings(document, location.href);
