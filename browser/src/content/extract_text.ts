/**
 * Content script — extracts main content from the current page.
 * Reading the page is `page_text.ts` (article or index, whichever carries more
 * of the page); this module is the part that touches the live environment —
 * readiness, the XML case, the preview image, and the returned page data.
 * Bundled with esbuild (not compiled by tsc) since content scripts can't use imports.
 */

import { readPage } from "./page_text.js";

const MAX_CHARS = 50_000;

interface PageData {
  title: string;
  url: string;
  text: string;
  // How the text was read (#2388) — "article" (Defuddle's own extraction, or the
  // XML/RSS case below, or a page never read at all) vs. "link_list" (the page's
  // own index of other pages won `richer` in `page_text.ts`).  Carried unchanged
  // through the background script and the wire (`browser_url.ts`'s own `PageData`
  // mirrors this shape — content scripts and the background bundle separately, so
  // the shape is duplicated rather than imported, same as every other field here).
  reading: "article" | "link_list";
  image: string;
  ready: boolean;
  extracted: boolean;
}

/** Domain-specific readiness locators. For JS-rendered pages, Defuddle may
 *  extract too early and get page chrome instead of content. These selectors
 *  gate extraction — if the selector isn't present yet, we return ready=false
 *  so pollForContent retries until the real content has rendered. */
const READINESS_LOCATORS: [match: (hostname: string) => boolean, selector: string][] = [
  [(h) => h.includes("kagi.com"), ".search-result"],
];

function findReadinessSelector(): string | null {
  for (const [match, selector] of READINESS_LOCATORS) {
    if (match(location.hostname)) return selector;
  }
  return null;
}

function extractXml(): string | null {
  const contentType = document.contentType;
  if (contentType && (contentType.includes("xml") || contentType.includes("rss"))) {
    const serializer = new XMLSerializer();
    return serializer.serializeToString(document);
  }
  return null;
}

function extractMetaImage(): string {
  const selectors = [
    'meta[property="og:image"]',
    'meta[name="twitter:image"]',
    'meta[property="og:image:url"]',
  ];
  for (const selector of selectors) {
    const el = document.querySelector(selector);
    const content = el?.getAttribute("content");
    if (content) return content;
  }
  // Kagi search results: grab first image from the inline image results
  const kagiImage = document.querySelector("._0_image_item");
  if (kagiImage) {
    const url = kagiImage.getAttribute("data-content_url");
    if (url) return url;
  }
  return "";
}

function extract(): PageData {
  const readinessSelector = findReadinessSelector();
  if (readinessSelector && !document.querySelector(readinessSelector)) {
    return {
      title: document.title,
      url: location.href,
      text: "",
      reading: "article",
      image: "",
      ready: false,
      extracted: false,
    };
  }

  // XML/RSS is read whole, not as an article vs. a link list — "article" is the
  // honest default for it, same as for a page that was never read at all.
  const xml = extractXml();
  const page = xml !== null ? { text: xml, reading: "article" as const } : readPage(document, location.href);

  return {
    title: document.title,
    url: location.href,
    text: (page?.text ?? "").slice(0, MAX_CHARS),
    reading: page?.reading ?? "article",
    image: extractMetaImage(),
    ready: true,
    extracted: page !== null,
  };
}

extract();
