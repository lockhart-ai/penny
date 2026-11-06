/**
 * browse_url tool — opens a URL in a hidden tab, extracts visible text, closes the tab.
 *
 * After the page reports "complete", polls extraction up to EXTRACT_MAX_RETRIES
 * times to handle JS-rendered content (e.g. Kagi search results, SPAs).
 * Failures retrying can't fix (PDFs, downloads, unscriptable tabs) throw
 * UnreadableUrlError at once — see unreadable.ts.
 */

import { PageReading, TAB_LOAD_TIMEOUT_MS } from "../../protocol.js";
import { logStep } from "../ws_log.js";
import { UnreadableUrlError, isUnreadableFailure } from "./unreadable.js";

const EXTRACT_MAX_RETRIES = 10;
const EXTRACT_POLL_MS = 500;
const MIN_CONTENT_LENGTH = 200;

interface PageData {
  title: string;
  url: string;
  text: string;
  reading: PageReading;
  image: string;
  ready: boolean;
  extracted: boolean;
}

const MAX_TAB_ATTEMPTS = 3;

export async function browseUrl(url: string): Promise<BrowseResult> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= MAX_TAB_ATTEMPTS; attempt++) {
    console.log(`[browse_url] opening: ${url} (attempt ${attempt}/${MAX_TAB_ATTEMPTS})`);
    logStep("browse", `attempt ${attempt}/${MAX_TAB_ATTEMPTS} for ${url}`);
    const tab = await openHiddenTab(url);
    const guard = watchForDownload(tab, url);
    try {
      const result = await Promise.race([readTab(tab.id!, url), guard.promise]);
      logStep("browse", `result ready tab=${tab.id} (${result.text.length}ch)`);
      return result;
    } catch (err) {
      console.warn(`[browse_url] attempt ${attempt} failed:`, err);
      logStep("browse", `attempt ${attempt} failed tab=${tab.id}: ${err}`);
      lastError = err;
      if (isUnreadable(err)) {
        logStep("browse", `unreadable ${url} — failing without retry tab=${tab.id}`);
        throw new UnreadableUrlError(url);
      }
    } finally {
      guard.dispose();
      logStep("browse", `closeTab tab=${tab.id}`);
      await closeTab(tab.id!);
    }
  }
  const reason = lastError instanceof Error ? lastError.message : String(lastError);
  throw new Error(`failed to read ${url} after ${MAX_TAB_ATTEMPTS} attempts: ${reason}`);
}

async function readTab(tabId: number, url: string): Promise<BrowseResult> {
  logStep("browse", `waitForTabLoad tab=${tabId}`);
  await waitForTabLoad(tabId);
  logStep("browse", `tab loaded tab=${tabId}, polling content`);
  const pageData = await pollForContent(tabId, url);
  logStep("browse", `content settled tab=${tabId}, formatting result`);
  return await formatResult(pageData);
}

function isUnreadable(err: unknown): boolean {
  return (
    err instanceof UnreadableUrlError ||
    (err instanceof Error && isUnreadableFailure(err.message))
  );
}

interface DownloadGuard {
  /** Rejects with UnreadableUrlError the moment the tab starts a download. */
  promise: Promise<never>;
  dispose: () => void;
}

/** Cancels any download one of the browse's tabs starts, and fails the browse at once. */
function watchForDownload(tab: browser.tabs.Tab, url: string): DownloadGuard {
  let reject: ((err: UnreadableUrlError) => void) | undefined;
  const promise = new Promise<never>((_, rej) => {
    reject = rej;
  });
  let currentUrl = url;

  const onUpdated = (
    updatedTabId: number,
    changeInfo: browser.tabs._OnUpdatedChangeInfo,
  ): void => {
    if (updatedTabId === tab.id && changeInfo.url) currentUrl = changeInfo.url;
  };
  const onCreated = (item: browser.downloads.DownloadItem): void => {
    if (!isOwnDownload(item, url, currentUrl)) return;
    console.warn(`[browse_url] download started from ${url} — cancelling (id=${item.id})`);
    logStep("browse", `download from tab=${tab.id} — cancelling id=${item.id}`);
    void cancelAndErase(item.id);
    reject?.(new UnreadableUrlError(url));
  };

  browser.downloads.onCreated.addListener(onCreated);
  browser.tabs.onUpdated.addListener(onUpdated);
  return {
    promise,
    dispose: () => {
      browser.downloads.onCreated.removeListener(onCreated);
      browser.tabs.onUpdated.removeListener(onUpdated);
    },
  };
}

function isOwnDownload(
  item: browser.downloads.DownloadItem,
  requestedUrl: string,
  currentUrl: string,
): boolean {
  // finalUrl exists at runtime (the post-redirect address) but not in these typings
  const finalUrl = (item as { finalUrl?: string }).finalUrl ?? "";
  return [item.url, finalUrl].some((u) => u === requestedUrl || u === currentUrl);
}

async function cancelAndErase(id: number): Promise<void> {
  try {
    await browser.downloads.cancel(id);
  } catch {
    // Download may already be finished — remove the landed file before erasing
    try {
      await browser.downloads.removeFile(id);
    } catch {
      // File may already be gone too
    }
  }
  try {
    await browser.downloads.erase({ id });
  } catch {
    // Download record may already be gone
  }
}

async function pollForContent(tabId: number, url: string): Promise<PageData> {
  // Baseline extraction — establishes initial content length for growth detection
  const baseline = await extractPageContent(tabId);
  let previousLength = baseline.text.trim().length;
  console.log(
    `[browse_url] ${url}: baseline ${previousLength} chars, ready=${baseline.ready}, extracted=${baseline.extracted}`,
  );

  if (baseline.ready && baseline.extracted && previousLength >= MIN_CONTENT_LENGTH) {
    // Have content — poll once more to check for growth
    await new Promise((r) => setTimeout(r, EXTRACT_POLL_MS));
    const second = await extractPageContent(tabId);
    const secondLength = second.text.trim().length;
    if (second.extracted && secondLength < previousLength * 2) {
      console.log(
        `[browse_url] ${url}: settled at ${secondLength} chars (prev ${previousLength})`,
      );
      return second;
    }
    console.log(
      `[browse_url] ${url}: ${secondLength} chars (prev ${previousLength}), still growing`,
    );
    previousLength = secondLength;
  }

  for (let attempt = 1; attempt <= EXTRACT_MAX_RETRIES; attempt++) {
    await new Promise((r) => setTimeout(r, EXTRACT_POLL_MS));
    const data = await extractPageContent(tabId);
    const textLen = data.text.trim().length;

    if (!data.ready) {
      console.log(
        `[browse_url] ${url}: not ready, waiting (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
      );
      continue;
    }

    if (!data.extracted || textLen < MIN_CONTENT_LENGTH) {
      console.log(
        `[browse_url] ${url}: extracted=${data.extracted}, ${textLen} chars, waiting (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
      );
      previousLength = textLen;
      continue;
    }

    if (previousLength > 0 && textLen < previousLength * 2) {
      console.log(
        `[browse_url] ${url}: settled at ${textLen} chars (prev ${previousLength}, attempt ${attempt})`,
      );
      return data;
    }

    console.log(
      `[browse_url] ${url}: ${textLen} chars (prev ${previousLength}), still growing (attempt ${attempt}/${EXTRACT_MAX_RETRIES})`,
    );
    previousLength = textLen;
  }

  const final = await extractPageContent(tabId);
  if (!final.ready) {
    throw new Error(`page not ready after ${EXTRACT_MAX_RETRIES} retries`);
  }
  if (!final.extracted) {
    throw new Error(`extraction failed after ${EXTRACT_MAX_RETRIES} retries`);
  }
  console.warn(
    `[browse_url] ${url}: settled on ${final.text.trim().length} chars after ${EXTRACT_MAX_RETRIES} retries`,
  );
  return final;
}

async function openHiddenTab(url: string): Promise<browser.tabs.Tab> {
  logStep("browse", "tabs.create start");
  const tab = await browser.tabs.create({ url, active: false });
  logStep("browse", `tabs.create done tab=${tab.id}`);
  if (!tab.id) {
    throw new Error("Failed to create tab");
  }
  try {
    logStep("browse", `tabs.hide tab=${tab.id}`);
    await browser.tabs.hide(tab.id);
    logStep("browse", `tabs.hide done tab=${tab.id}`);
  } catch {
    // tabHide may not be available — tab stays visible but still works
    logStep("browse", `tabs.hide unavailable tab=${tab.id}`);
  }
  return tab;
}

function waitForTabLoad(tabId: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      browser.tabs.onUpdated.removeListener(listener);
      reject(new Error(`Tab load timed out after ${TAB_LOAD_TIMEOUT_MS}ms`));
    }, TAB_LOAD_TIMEOUT_MS);

    function listener(
      updatedTabId: number,
      changeInfo: browser.tabs._OnUpdatedChangeInfo,
    ): void {
      if (updatedTabId === tabId && changeInfo.status === "complete") {
        clearTimeout(timeout);
        browser.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    }

    browser.tabs.onUpdated.addListener(listener);
  });
}

async function extractPageContent(tabId: number): Promise<PageData> {
  logStep("browse", `executeScript (extract) tab=${tabId}`);
  const results = await browser.tabs.executeScript(tabId, {
    file: "/dist/content/extract_text.js",
    runAt: "document_idle",
  });
  logStep("browse", `executeScript returned tab=${tabId}`);

  if (!results || !results[0]) {
    throw new Error("Content script returned no results");
  }

  return results[0] as PageData;
}

async function closeTab(tabId: number): Promise<void> {
  try {
    logStep("browse", `tabs.remove tab=${tabId}`);
    await browser.tabs.remove(tabId);
    logStep("browse", `tabs.remove done tab=${tabId}`);
  } catch {
    // Tab may already be closed
    logStep("browse", `tabs.remove failed tab=${tabId}`);
  }
}

interface BrowseResult {
  text: string;
  image: string;
  reading: PageReading;
}

async function formatResult(data: PageData): Promise<BrowseResult> {
  if (data.image) logStep("browse", "downloadImageAsDataUri start");
  const image = data.image ? await downloadImageAsDataUri(data.image) : "";
  if (data.image) logStep("browse", `downloadImageAsDataUri done (${image.length}ch)`);
  console.log(`[browse_url] image: ${image ? `${image.length} chars` : "none"}`);
  return {
    text: `Title: ${data.title}\nURL: ${data.url}\n\n${data.text}`,
    image,
    reading: data.reading,
  };
}

async function downloadImageAsDataUri(url: string): Promise<string> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return "";
    const blob = await resp.blob();
    const buffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (const b of bytes) binary += String.fromCharCode(b);
    const b64 = btoa(binary);
    return `data:${blob.type};base64,${b64}`;
  } catch {
    console.warn("[browse_url] failed to download image:", url);
    return "";
  }
}
