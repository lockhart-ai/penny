/**
 * The self-driving half: open each site in turn, measure it twice, download the
 * lot as one JSON file.
 *
 * It runs on startup because a temporary extension in a throwaway profile has no
 * user to click anything — `npm run compare` launches Firefox and the run is
 * already going.  Progress shows in the toolbar badge and in the Browser Console
 * (Tools > Browser Tools > Browser Console), which `web-ext` opens for you.
 *
 * Two passes per page, ten seconds apart, because these pages are not finished
 * when they say they are: a consent wall gets dismissed or times out, a lazy
 * grid hydrates, an ad reflows the page.  Recording both is the honest version
 * of picking a settle time.
 */

import { DEFUDDLE_RANGE, DEFUDDLE_VERSION } from "./defuddle_version.js";
import { SITES, LIST_VERSION, type Site } from "./sites.js";
import type { FailedPage, MeasuredPage, RunReport, VisitContext } from "./types.js";

/** Let the page paint and run its startup scripts before the first read. */
const SETTLE_MS = 5_000;

/** Between the two reads — long enough for hydration and a consent wall. */
const HYDRATION_DELAY_MS = 10_000;

/** A page that has not said `complete` by now gets measured as it stands. */
const LOAD_TIMEOUT_MS = 45_000;

const LOAD_POLL_MS = 250;

/** Firefox settling after launch, before the first tab opens. */
const STARTUP_GRACE_MS = 3_000;

const RESULT_FILE = "compare-results.json";

const COMPLETE = "complete";

async function run(): Promise<void> {
  console.log(
    `[compare-readings] ${SITES.length} sites, two passes each, defuddle ${DEFUDDLE_VERSION} — starting`,
  );
  const runStartedAt = new Date().toISOString();
  await delay(STARTUP_GRACE_MS);

  const pages: (MeasuredPage | FailedPage)[] = [];
  for (const [position, site] of SITES.entries()) {
    badge(`${position + 1}/${SITES.length}`);
    pages.push(...(await visit(site)));
  }

  badge("done");
  await downloadReport(report(pages, runStartedAt));
}

/** One site: open it, read it on settle, read it again after hydration, close. */
async function visit(site: Site): Promise<(MeasuredPage | FailedPage)[]> {
  console.log(`[compare-readings] ${site.url}`);
  const tab = await browser.tabs.create({ url: site.url, active: true });
  const tabId = tab.id;
  if (tabId === undefined) return [failure(site, { kind: site.kind, pass: 1, loadTimedOut: false }, "tab had no id")];

  const loadTimedOut = !(await waitForLoad(tabId));
  await delay(SETTLE_MS);
  const first = await measure(tabId, site, { kind: site.kind, pass: 1, loadTimedOut });
  await delay(HYDRATION_DELAY_MS);
  const second = await measure(tabId, site, { kind: site.kind, pass: 2, loadTimedOut });

  await browser.tabs.remove(tabId).catch(() => undefined);
  return [first, second];
}

/** Inject the comparator and take back what it returned. */
async function measure(
  tabId: number,
  site: Site,
  context: VisitContext,
): Promise<MeasuredPage | FailedPage> {
  try {
    const results = await browser.tabs.executeScript(tabId, { file: "/build/comparator.js" });
    const comparison = results?.[0];
    if (!comparison) throw new Error("comparator returned nothing");
    return { ...(comparison as MeasuredPage), ...context };
  } catch (error) {
    console.warn(`[compare-readings] pass ${context.pass} failed on ${site.url}`, error);
    return failure(site, context, String(error));
  }
}

/** Poll rather than listen: the load can complete before a listener attaches,
 *  and a page that never completes has to be measured anyway. */
async function waitForLoad(tabId: number): Promise<boolean> {
  const deadline = Date.now() + LOAD_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const tab = await browser.tabs.get(tabId).catch(() => undefined);
    if (tab?.status === COMPLETE) return true;
    await delay(LOAD_POLL_MS);
  }
  return false;
}

function report(pages: (MeasuredPage | FailedPage)[], runStartedAt: string): RunReport {
  return {
    harness: "browser/tools/compare-readings",
    branch: "compare-readings-harness-1961",
    doNotMerge: "Evidence harness for issue #1942 / PR #1961. Not shippable code.",
    comparing: "production = main's naive Defuddle reading; pr = #1961's readPage",
    runStartedAt,
    runFinishedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    defuddleVersion: DEFUDDLE_VERSION,
    defuddleRange: DEFUDDLE_RANGE,
    listVersion: LIST_VERSION,
    siteCount: SITES.length,
    settleMs: SETTLE_MS,
    hydrationDelayMs: HYDRATION_DELAY_MS,
    pages,
  };
}

/** Save the file, and log it too — a dismissed download prompt or a full disk
 *  should not be the difference between having the run and not having it. */
async function downloadReport(finished: RunReport): Promise<void> {
  const json = JSON.stringify(finished, null, 2);
  console.log(`[compare-readings] run complete — ${finished.pages.length} records`);
  console.log(json);
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  await browser.downloads.download({ url, filename: RESULT_FILE });
  console.log(`[compare-readings] downloaded ${RESULT_FILE}`);
}

function failure(site: Site, context: VisitContext, error: string): FailedPage {
  return { requestedUrl: site.url, measuredAt: new Date().toISOString(), error, ...context };
}

function badge(text: string): void {
  browser.browserAction.setBadgeText({ text });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

run().catch((error) => console.error("[compare-readings] run failed", error));
