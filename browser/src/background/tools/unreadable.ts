/**
 * Unreadable-URL classification — failures retrying a browse can't fix.
 *
 * A fresh tab fails the same way for these (a PDF opens in the built-in
 * viewer or starts a download; the extension can't script either), so they
 * fail at once with one typed error instead of burning MAX_TAB_ATTEMPTS.
 */

export class UnreadableUrlError extends Error {
  constructor(url: string) {
    super(
      `${url} is a file (such as a PDF), not a web page, so browse can't read it. ` +
        `A page that links to it may have the same content.`,
    );
    this.name = "UnreadableUrlError";
  }
}

/** True when a failure message retrying can't fix — a fresh tab fails identically. */
export function isUnreadableFailure(message: string): boolean {
  return message.includes("Missing host permission");
}
