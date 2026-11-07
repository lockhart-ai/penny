/**
 * Attaches each captured link thumbnail's own picture, in the background,
 * after `browse_url`'s read (#3115) — Penny's server no longer downloads
 * images itself (#3113), so the card picture has to arrive on the wire
 * already fetched.
 *
 * Runs at most MAX_CONCURRENT fetches at a time, inside a shared budget of
 * BUDGET_MS from the call: once the budget is spent, whatever has already
 * finished is returned and nothing further is waited on. Every per-entry
 * failure — an HTTP error, a decode error, a result over the size cap, or
 * simply not finishing in time — is a silent drop; one `logStep` line per
 * page says how many of how many succeeded.
 */

import type { LinkThumbnail, LinkThumbnailImage } from "../../protocol.js";
import { logStep } from "../ws_log.js";

const MAX_CONCURRENT = 6;
const BUDGET_MS = 4000;
const MAX_DIMENSION = 640;
const MAX_BYTES = 200 * 1024;
const JPEG_QUALITY = 0.8;
const MAX_ENTRIES = 40;
const BASE64_CHUNK_BYTES = 8192;

/** Fetches and attaches each entry's own picture, within the shared budget —
 *  the entries that got one, in document order, at most MAX_ENTRIES. */
export async function attachThumbnailData(
  entries: LinkThumbnail[],
): Promise<LinkThumbnailImage[]> {
  const deadline = Date.now() + BUDGET_MS;
  const images = await fetchAll(entries, deadline);
  const attached = selectAttached(entries, images);
  logStep("thumbnail", `attached ${attached.length}/${entries.length} thumbnails`);
  return attached;
}

/** Fetches every entry's image under a shared concurrency cap, stopping at
 *  `deadline` without waiting on whatever is still in flight.  Returned in
 *  the same order as `entries`; an index never reached, or whose fetch lost
 *  the race, stays `undefined`. */
async function fetchAll(entries: LinkThumbnail[], deadline: number): Promise<(string | undefined)[]> {
  const images: (string | undefined)[] = new Array(entries.length);
  let cursor = 0;
  async function worker(): Promise<void> {
    while (cursor < entries.length && Date.now() < deadline) {
      const index = cursor++;
      images[index] = await fetchThumbnailImage(entries[index].image_url);
    }
  }
  const workers = Array.from({ length: Math.min(MAX_CONCURRENT, entries.length) }, worker);
  await Promise.race([Promise.all(workers), sleepUntil(deadline)]);
  return images;
}

function sleepUntil(deadline: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, deadline - Date.now())));
}

/** One entry's picture: fetched, decoded, fit to MAX_DIMENSION, re-encoded as
 *  JPEG — or `undefined` on any failure along the way, including over the
 *  size cap. */
async function fetchThumbnailImage(url: string): Promise<string | undefined> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return undefined;
    return await resizeToDataUri(await resp.blob());
  } catch {
    return undefined;
  }
}

/** Decodes, downscales (never up), and re-encodes a fetched image as a JPEG
 *  data URI — `undefined` when the result is still over the size cap. */
async function resizeToDataUri(blob: Blob): Promise<string | undefined> {
  const bitmap = await createImageBitmap(blob);
  const { width, height } = fitWithin(bitmap.width, bitmap.height, MAX_DIMENSION);
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) return undefined;
  ctx.drawImage(bitmap, 0, 0, width, height);
  const jpeg = await canvas.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
  if (exceedsSizeCap(jpeg.size)) return undefined;
  return await blobToDataUri(jpeg);
}

/** The size an image fits within `max` on its longest side, preserving
 *  aspect ratio — unchanged when it's already within `max` on both sides
 *  (never upscaled). Pure: covers landscape, portrait, and square alike. */
export function fitWithin(
  width: number,
  height: number,
  max: number,
): { width: number; height: number } {
  if (width <= max && height <= max) return { width, height };
  const scale = max / Math.max(width, height);
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

/** Whether an encoded image is over the 200KB cap this module drops at. */
export function exceedsSizeCap(bytes: number): boolean {
  return bytes > MAX_BYTES;
}

/** The entries that got an image, in `entries`' own order, capped at `cap` —
 *  an entry whose fetch never produced one (an `undefined` at its index) is
 *  left out, never sent as an empty-string placeholder. */
export function selectAttached(
  entries: LinkThumbnail[],
  images: (string | undefined)[],
  cap: number = MAX_ENTRIES,
): LinkThumbnailImage[] {
  const attached: LinkThumbnailImage[] = [];
  for (let i = 0; i < entries.length && attached.length < cap; i++) {
    const image = images[i];
    if (image) attached.push({ ...entries[i], image });
  }
  return attached;
}

/** Base64-encodes a blob's bytes in fixed-size chunks (never one character
 *  at a time — the old per-byte `+=` loop built a huge string a character at
 *  a time for a 200KB image) and wraps it as a JPEG data URI. */
async function blobToDataUri(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += BASE64_CHUNK_BYTES) {
    chunks.push(String.fromCharCode(...bytes.subarray(i, i + BASE64_CHUNK_BYTES)));
  }
  return `data:image/jpeg;base64,${btoa(chunks.join(""))}`;
}
