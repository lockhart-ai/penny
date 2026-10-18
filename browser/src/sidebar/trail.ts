/**
 * Pure helpers for the sidebar's live-status trail (#2570): parsing the
 * typing content the background already relays into trail rows, and
 * deriving the page-context bar's hostname sub-line.
 */

/**
 * Typing-content marker the server writes ahead of each completed tool step
 * in a cumulative `<br>`-separated checklist (produced by
 * penny/penny/channels/browser/channel.py `on_tool_start` — the one other
 * side that writes this marker). This constant is its single TS definition,
 * so the parser and any future TS consumer can't drift from it.
 */
export const TOOL_STATUS_DONE_PREFIX = "&#x2713; ";

/** One row of the live-status trail: a completed step or the running one. */
export interface TrailRow {
  done: boolean;
  /** Server-authored label HTML, rendered as-is (same trust as today). */
  html: string;
}

/**
 * Parse a typing-indicator content string into trail rows. The server
 * writes a cumulative `<br>`-separated checklist: completed steps carry
 * TOOL_STATUS_DONE_PREFIX, the rest are the running step(s). A plain
 * string without `<br>` (e.g. "Penny is thinking") is a single running row.
 */
export function parseTrail(content: string): TrailRow[] {
  return content
    .split("<br>")
    .filter((line) => line.trim() !== "")
    .map((line) =>
      line.startsWith(TOOL_STATUS_DONE_PREFIX)
        ? { done: true, html: line.slice(TOOL_STATUS_DONE_PREFIX.length) }
        : { done: false, html: line },
    );
}

/** Hostname of a page URL for the context bar's sub-line; "" when unparseable. */
export function hostnameOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return "";
  }
}