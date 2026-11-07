/**
 * Pure helpers for the sidebar's server-backed chat history (#3139/#3148):
 * ordering a `history_response` page for display, merging an older page in
 * without duplicates, and deciding whether a history row needs the
 * cross-device caption a live `incoming_message` push gets — reusing the
 * same rule (`sidebar/incoming.ts`, #3144) rather than a second one.
 */

import type { BrowserHistoryMessage } from "../protocol.js";
import { deviceCaption, isOwnDevice } from "./incoming.js";

/** The server always answers newest-first; the sidebar renders a
 *  conversation top-to-bottom, oldest first. */
export function oldestFirst(messagesNewestFirst: BrowserHistoryMessage[]): BrowserHistoryMessage[] {
  return [...messagesNewestFirst].reverse();
}

/** Prepend an older page (also newest-first, as the server sends it) ahead
 *  of the oldest-first messages already shown — "Load earlier" pages back
 *  by a cursor, and a row the first page already carried must never
 *  double-appear if the two pages overlap at their boundary. */
export function prependOlderPage(
  shownOldestFirst: BrowserHistoryMessage[],
  olderPageNewestFirst: BrowserHistoryMessage[],
): BrowserHistoryMessage[] {
  const shownIds = new Set(shownOldestFirst.map((m) => m.message_id));
  const olderOldestFirst = oldestFirst(olderPageNewestFirst).filter(
    (m) => !shownIds.has(m.message_id),
  );
  return [...olderOldestFirst, ...shownOldestFirst];
}

/** The caption an incoming history row needs, or `undefined` for a plain
 *  bubble — an outgoing (Penny) row never gets one; an incoming row gets
 *  one unless it's this very sidebar's own device (`isOwnDevice`), matching
 *  the live `incoming_message` push's rule exactly. Also plain when the
 *  row's own device could not be resolved (`channel_type`/`device_label`
 *  null) — a known, rare server-side gap, not a reason to invent a label. */
export function historyCaption(
  message: Pick<BrowserHistoryMessage, "direction" | "channel_type" | "device_label">,
  ownDeviceLabel: string,
): string | undefined {
  if (message.direction !== "incoming") return undefined;
  if (!message.channel_type || !message.device_label) return undefined;
  if (isOwnDevice(message.channel_type, message.device_label, ownDeviceLabel)) return undefined;
  return deviceCaption(message.channel_type, message.device_label);
}
