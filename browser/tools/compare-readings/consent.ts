/**
 * Spotting a page that showed a consent wall instead of itself.
 *
 * Entirely a heuristic, and flagged rather than acted on: a consent-walled page
 * is still recorded, still measured, still in the results.  Marking it just says
 * "this row is probably not evidence about news homepages", and every number the
 * mark was made from is in the row next to it, so the call can be overruled by
 * eye.  Missing one is a wrong row in the evidence; SKIPPING one would be a
 * silently missing row, which is worse.
 */

/** Consent-management platforms, by the substring their iframe/container
 *  carries.  Nowhere near exhaustive — a new vendor just goes unflagged, and the
 *  word count below usually catches it anyway. */
const CONSENT_TOKENS = [
  "consent",
  "gdpr",
  "onetrust",
  "optanon",
  "sourcepoint",
  "sp-cc",
  "didomi",
  "quantcast",
  "trustarc",
  "cookielaw",
  "cookiebot",
  "usercentrics",
  "privacy-mgmt",
  "privacymanager",
];

/** A rendered news homepage carries thousands of words.  Under this, whatever
 *  was measured is not the homepage — a wall, an interstitial, or a page that
 *  never rendered. */
const RENDERED_WORD_FLOOR = 150;

export interface ConsentVerdict {
  consentWalled: boolean;
  consentSignals: string[];
}

export function detectConsentWall(doc: Document, visibleWords: number): ConsentVerdict {
  const signals = [
    ...(visibleWords < RENDERED_WORD_FLOOR ? [`only ${visibleWords} visible words`] : []),
    ...consentFrames(doc),
    ...consentDialogs(doc),
  ];
  return { consentWalled: signals.length > 0, consentSignals: signals };
}

/** A CMP that renders in an iframe — the common shape, and the one that hides
 *  the page's own text from every reading at once. */
function consentFrames(doc: Document): string[] {
  const signals: string[] = [];
  for (const frame of doc.querySelectorAll("iframe")) {
    const token = matchedToken(`${frame.getAttribute("src") ?? ""} ${frame.id} ${frame.className}`);
    if (token) signals.push(`consent iframe (${token})`);
  }
  return signals;
}

/** A CMP that renders inline, as a modal over the page. */
function consentDialogs(doc: Document): string[] {
  const signals: string[] = [];
  for (const dialog of doc.querySelectorAll('dialog, [role="dialog"], [role="alertdialog"]')) {
    const token = matchedToken(`${dialog.id} ${dialog.className}`);
    if (token) signals.push(`consent dialog (${token})`);
  }
  return signals;
}

function matchedToken(haystack: string): string | null {
  const lowered = haystack.toLowerCase();
  return CONSENT_TOKENS.find((token) => lowered.includes(token)) ?? null;
}
