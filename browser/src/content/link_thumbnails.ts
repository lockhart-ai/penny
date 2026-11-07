/**
 * Capturing each link card's own thumbnail address (#3068).
 *
 * When a page is read as its own link index (`page_text.ts`), the cited
 * addresses are other pages, and no image is ever captured for them — the
 * page's meta image is the front page's picture, so a reply citing one of its
 * articles attaches an unrelated image or none.  This pass runs over the live
 * document on every read and pairs each link with its own card's picture: the
 * card is the shallow enclosing group that holds no link to a different
 * address, and the picture is whatever the card renders for itself — its
 * `<img>`, a `<picture>` source, or an og:image-style meta — with the
 * lazy-loading placeholder shapes resolved to the address the real picture
 * sits at, an icon skipped by its known size or by its address (a favicon,
 * however served), and what the page repeats across cards (logos) dropped.
 *
 * Pure over the given Document — jsdom in the tests, the live page in the
 * content script.  jsdom has no layout, so a rendered size is only read when
 * it is non-zero: a size of 0 or unknown is never a reason to skip.
 */

import type { LinkThumbnail } from "../protocol.js";
import { INDEX_LINK_SELECTOR, WEB_URL, normalize } from "./page_text.js";

/** Entries per page — a front page offers hundreds of links and the payload
 *  rides the wire on every read; beyond this the later links are rarely the
 *  ones a reply cites. */
const MAX_ENTRIES = 60;

/** An image KNOWN to be smaller than this in width or height is chrome (an
 *  icon, a spacer), not the card's picture.  A size of 0 or unknown is "no
 *  layout here", never "tiny" — jsdom has no layout at all. */
const MIN_KNOWN_SIZE = 80;

/** A src matching this — or missing, or a data URI — is a lazy-loading
 *  placeholder, not the picture; the real address sits in a data-* attribute
 *  beside it. */
const PLACEHOLDER_PATTERN = /1x1|blank|spacer|placeholder/i;

/** How many ancestors above the anchor a card may sit.  A card is a shallow
 *  group; past this the nearest "no other link" scope is the page itself,
 *  whose picture would be the front page's own image again. */
const MAX_CARD_ANCESTORS = 4;

/** The picture selectors tried inside a card, in the order a card renders
 *  them: its `<img>`, a `<picture>` source, an og:image-style meta. */
const META_IMAGE_SELECTOR = 'meta[property="og:image"], meta[itemprop="image"]';

/** An address naming a favicon, not a card's own picture — a convention every
 *  site shares (a search result's only image is often its site icon, served
 *  through a favicon proxy), never a per-site rule. */
const FAVICON_PATTERN = /favicon/i;

/** One thumbnail entry per distinct link, in document order, capped — then
 *  whatever the page repeats across cards is dropped. */
export function captureLinkThumbnails(doc: Document): LinkThumbnail[] {
  const entries: LinkThumbnail[] = [];
  const seen = new Set<string>();
  for (const anchor of doc.body.querySelectorAll<HTMLAnchorElement>(INDEX_LINK_SELECTOR)) {
    if (entries.length >= MAX_ENTRIES) break;
    const entry = entryFor(anchor, doc);
    if (entry === null || seen.has(entry.url)) continue;
    seen.add(entry.url);
    entries.push(entry);
  }
  return dropSharedImages(entries);
}

/** One link's entry: the label the index line carries, the link's address
 *  exactly as the index writes it, and the card's picture resolved absolute —
 *  or null when the link has no label, points nowhere http(s), or its card
 *  holds no usable picture. */
function entryFor(anchor: HTMLAnchorElement, doc: Document): LinkThumbnail | null {
  const title = normalize(anchor.textContent ?? "");
  const url = anchor.href;
  if (!title || !WEB_URL.test(url)) return null;
  const image_url = cardImage(findCard(anchor), doc);
  if (image_url === null) return null;
  return { url, image_url, title };
}

/** The link's card: the highest ancestor, walking up at most four steps, that
 *  contains no link to a different address.  The anchor itself is always in —
 *  it contains only itself — so a card always exists. */
function findCard(anchor: HTMLAnchorElement): Element {
  let card: Element = anchor;
  let up = anchor.parentElement;
  for (let level = 0; level < MAX_CARD_ANCESTORS && up !== null; level++) {
    if (containsOtherLink(up, anchor.href)) break;
    card = up;
    up = up.parentElement;
  }
  return card;
}

/** Does this scope hold any link whose address differs from the given one?
 *  A card is a group around ONE story; a second address means the group is
 *  the page itself, and its picture belongs to no single link. */
function containsOtherLink(scope: Element, href: string): boolean {
  for (const other of scope.querySelectorAll<HTMLAnchorElement>("a[href]")) {
    if (other.href !== href) return true;
  }
  return false;
}

/** The card's own picture, absolute and http(s) only, in the order the card
 *  renders it — each stage may come up empty and hand the next a turn. */
function cardImage(card: Element, doc: Document): string | null {
  return imageFromImg(card, doc) ?? imageFromSource(card, doc) ?? imageFromMeta(card, doc);
}

/** The card's first `<img>` that passes the known-size and favicon checks,
 *  not only its first `<img>` — a card whose first image is an icon or an
 *  avatar still gets its photo from whichever image further down is it. */
function imageFromImg(card: Element, doc: Document): string | null {
  for (const img of card.querySelectorAll<HTMLImageElement>("img")) {
    if (knownSizeBelow(img)) continue;
    const address = resolvedPicture(imgValue(img), doc);
    if (address !== null) return address;
  }
  return null;
}

function imageFromSource(card: Element, doc: Document): string | null {
  const source = card.querySelector("picture source");
  if (source === null) return null;
  const set = source.getAttribute("srcset") ?? source.getAttribute("data-srcset") ?? "";
  return resolvedPicture(largestSrcsetCandidate(set) ?? "", doc);
}

function imageFromMeta(card: Element, doc: Document): string | null {
  const meta = card.querySelector(META_IMAGE_SELECTOR);
  return resolvedPicture(meta?.getAttribute("content") ?? "", doc);
}

/** The `<img>`'s real address.  A src that is missing, a data URI, or a known
 *  blank/spacer pattern is a lazy-loading placeholder — the real address sits
 *  in the data-* attributes beside it, in the order the lazy loaders write
 *  them; otherwise the srcset's largest candidate is the picture itself. */
function imgValue(img: HTMLImageElement): string {
  const direct = img.getAttribute("src") ?? "";
  if (isPlaceholder(direct)) {
    const single = img.getAttribute("data-src") ?? img.getAttribute("data-lazy-src");
    if (single !== null) return single;
    const set = img.getAttribute("data-srcset");
    return set === null ? direct : largestSrcsetCandidate(set) ?? direct;
  }
  const set = img.getAttribute("srcset");
  return set === null ? direct : largestSrcsetCandidate(set) ?? direct;
}

function isPlaceholder(value: string): boolean {
  return value === "" || value.startsWith("data:") || PLACEHOLDER_PATTERN.test(value);
}

/** The largest candidate in a srcset — the one with the highest width (w) or
 *  density (x) descriptor; a bare candidate counts as 1x, the way the browser
 *  reads it. */
export function largestSrcsetCandidate(value: string): string | null {
  let best: { url: string; score: number } | null = null;
  for (const { url, descriptors } of srcsetCandidates(value)) {
    const score = srcsetScore(descriptors);
    if (score !== null && (best === null || score > best.score)) best = { url, score };
  }
  return best?.url ?? null;
}

/** A srcset's candidates, parsed to the HTML grammar rather than split on
 *  every comma: a candidate's URL runs up to the next whitespace — and may
 *  itself hold commas, the way some CDNs encode an image id in the path —
 *  while candidates are separated by a comma after the descriptor.  When the
 *  URL itself ends with one or more commas, those commas end the candidate
 *  right there — stripped, with no descriptor to collect — rather than
 *  being absorbed into the next candidate's descriptor text. */
function srcsetCandidates(value: string): { url: string; descriptors: string[] }[] {
  const candidates: { url: string; descriptors: string[] }[] = [];
  let i = 0;
  while (i < value.length) {
    while (i < value.length && /[\s,]/.test(value[i]!)) i++;
    if (i >= value.length) break;
    const urlStart = i;
    while (i < value.length && !/\s/.test(value[i]!)) i++;
    const rawUrl = value.slice(urlStart, i);
    const url = rawUrl.replace(/,+$/, "");
    if (url === "") continue;
    if (rawUrl !== url) {
      candidates.push({ url, descriptors: [] });
      continue;
    }
    while (i < value.length && /\s/.test(value[i]!)) i++;
    const descriptorsStart = i;
    while (i < value.length && value[i] !== ",") i++;
    const descriptors = value.slice(descriptorsStart, i).trim().split(/\s+/).filter(Boolean);
    candidates.push({ url, descriptors });
  }
  return candidates;
}

/** One candidate's size on the single scale the comparison needs: a width
 *  descriptor counts its pixels, a density descriptor its multiples, and an
 *  unparseable one leaves the candidate out entirely. */
function srcsetScore(descriptors: string[]): number | null {
  if (descriptors.length === 0) return 1;
  const first = descriptors[0]!;
  const tail = first.slice(-1);
  const number = Number(first.slice(0, -1));
  if ((tail !== "w" && tail !== "x") || !Number.isFinite(number) || number <= 0) return null;
  return number;
}

/** The address made absolute against the page's own base, kept only when it
 *  names something fetchable over http(s) and isn't an icon by address — a
 *  data URI, a relative path that fails to resolve, or a favicon all read as
 *  no picture here, never as this candidate's picture. */
function resolvedPicture(value: string, doc: Document): string | null {
  const address = absolute(value, doc);
  return address !== null && !isFaviconAddress(address) ? address : null;
}

/** The address made absolute against the page's own base — kept only when it
 *  names something fetchable over http(s), never a data URI. */
function absolute(value: string, doc: Document): string | null {
  if (value === "") return null;
  try {
    const resolved = new URL(value, doc.baseURI);
    return WEB_URL.test(resolved.href) ? resolved.href : null;
  } catch {
    return null;
  }
}

/** Does this resolved address name a favicon?  The word in the path or the
 *  query, or an ICO extension — the one check every site's icon shares,
 *  however it serves it (a direct file, a favicon proxy). */
function isFaviconAddress(url: string): boolean {
  try {
    const { pathname, search } = new URL(url);
    return FAVICON_PATTERN.test(pathname + search) || pathname.toLowerCase().endsWith(".ico");
  } catch {
    return false;
  }
}

/** Is this image KNOWN to render under the icon threshold?  The width/height
 *  attributes decide when present, then the rendered box when it is non-zero
 *  — a zero box is "no layout here", never "zero-sized" — and last the
 *  loaded image's own natural size, read only once it has loaded. */
function knownSizeBelow(img: HTMLImageElement): boolean {
  for (const attribute of [img.getAttribute("width"), img.getAttribute("height")]) {
    const size = Number(attribute);
    if (Number.isFinite(size) && size > 0 && size < MIN_KNOWN_SIZE) return true;
  }
  const box = img.getBoundingClientRect();
  if ((box.width > 0 && box.width < MIN_KNOWN_SIZE) || (box.height > 0 && box.height < MIN_KNOWN_SIZE)) {
    return true;
  }
  return img.complete && img.naturalWidth > 0
    ? img.naturalWidth < MIN_KNOWN_SIZE || img.naturalHeight < MIN_KNOWN_SIZE
    : false;
}

/** The pass drops what a front page repeats across cards: an address two or
 *  more different links share is the site's logo, not any one card's picture.
 *  Counted over the entries themselves — already one per distinct link. */
function dropSharedImages(entries: LinkThumbnail[]): LinkThumbnail[] {
  const shares = new Map<string, number>();
  for (const entry of entries) shares.set(entry.image_url, (shares.get(entry.image_url) ?? 0) + 1);
  return entries.filter((entry) => shares.get(entry.image_url) === 1);
}
