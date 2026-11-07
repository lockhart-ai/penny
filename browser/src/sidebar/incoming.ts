/**
 * Pure helpers for rendering an `incoming_message` push (#3144) — a user
 * message sent from another device (chat replication, #3137) — as the
 * sidebar's own user bubble, captioned with the device it came from.
 */

import type { IncomingChannelType } from "../protocol.js";

/** Display name for each channel a chat turn can start on. */
const CHANNEL_DISPLAY_NAME: Record<IncomingChannelType, string> = {
  signal: "Signal",
  ios: "iOS",
  browser: "Browser",
};

/** The caption shown under an incoming message's bubble, e.g.
 *  "from Kitchen iPad · iOS" or "from <label> · Signal". */
export function deviceCaption(channelType: IncomingChannelType, deviceLabel: string): string {
  return `from ${deviceLabel} · ${CHANNEL_DISPLAY_NAME[channelType]}`;
}

/** True when an `incoming_message` push is an echo of a message this very
 *  sidebar just sent — its channel is `browser` and its device label is
 *  this sidebar's own stored label — so it is never rendered twice. */
export function isOwnDevice(
  channelType: IncomingChannelType,
  deviceLabel: string,
  ownLabel: string,
): boolean {
  return channelType === "browser" && deviceLabel === ownLabel;
}
