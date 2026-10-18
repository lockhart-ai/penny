/** Prompts tab — pure formatting helpers for the restyled run list (#2567).
 * No DOM here, so these are unit-testable without jsdom; `prompts.ts` is the
 * only caller. */

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;

/** The wall/model time-pair pill's own duration format (#2567 condition 2):
 * seconds (one decimal under 10s, whole seconds from 10s up) below a
 * minute, then "Nm Ms", then "Nh Nm" above — one ladder for BOTH numbers in
 * a pill, so they can only differ in unit when they differ in magnitude. A
 * 900ms wall paired with a 2.3s model never reads as "900ms / 2.3s"; both
 * are always expressed in the same unit at the same magnitude ("0.9s /
 * 2.3s"). */
export function formatPillDuration(ms: number): string {
  const totalSeconds = ms / 1000;
  if (totalSeconds < 60) {
    return totalSeconds < 10 ? `${totalSeconds.toFixed(1)}s` : `${Math.round(totalSeconds)}s`;
  }
  if (ms < HOUR_MS) {
    const minutes = Math.floor(ms / MINUTE_MS);
    const seconds = Math.round((ms - minutes * MINUTE_MS) / 1000);
    return `${minutes}m ${seconds}s`;
  }
  const hours = Math.floor(ms / HOUR_MS);
  const minutes = Math.round((ms - hours * HOUR_MS) / MINUTE_MS);
  return `${hours}h ${minutes}m`;
}

/** "1 step" vs "N steps" — the run header's step-count cell. */
export function formatStepCount(count: number): string {
  return count === 1 ? "1 step" : `${count} steps`;
}

/** Token counts for the ↓/↑ columns — whole below 1k, one-decimal k below
 * 1M, one-decimal M above ("977", "13.6k", "1.2M"), the same ladder the
 * pill's duration format uses (#2610): a run total in the millions must
 * never outgrow the ~5-6 character track its column is sized for. */
export function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return String(count);
}

/** #2640 — where an opened Prompts entry's detail splits its message list:
 * `systemCount` leading system messages always render as their own rows,
 * `earlierCount` messages after them hide behind ONE collapsed row ("N
 * earlier messages"), and everything from `visibleFrom` on — the message
 * that started this run, then the run's own tool calls and results — stays
 * visible. */
export interface EarlierWindow {
  systemCount: number;
  earlierCount: number;
  visibleFrom: number;
}

/** The window for an opened prompt of a run, defined STRUCTURALLY by the
 * run's FIRST own prompt (earliest by start time, draws never count): its
 * context minus the leading system message(s) and minus its final message —
 * the one that started the run — is "earlier". Purely positional: never a
 * message's role beyond that leading system run, never its wording.
 *
 * `firstOwnMessages` is that first prompt's context; `null` (a micro-context
 * draw: its own short context, no earlier conversation) means everything
 * visible and no collapsed row. `openedMessageCount` is the opened prompt's
 * own context length, clamped against so a shorter context can never index
 * past it — its final message always stays visible. An empty
 * `firstOwnMessages` (a run with no first prompt on record) reads the same
 * way: no earlier, no collapsed row. */
export function earlierWindow(
  firstOwnMessages: readonly Record<string, unknown>[] | null,
  openedMessageCount: number,
): EarlierWindow {
  if (firstOwnMessages === null) {
    return { systemCount: 0, earlierCount: 0, visibleFrom: 0 };
  }
  let systemLead = 0;
  while (
    systemLead < firstOwnMessages.length &&
    String(firstOwnMessages[systemLead]?.role ?? "") === "system"
  ) {
    systemLead++;
  }
  const runStart = Math.max(0, firstOwnMessages.length - 1);
  const systemCount = Math.min(systemLead, openedMessageCount);
  const visibleFrom = Math.max(
    systemCount,
    Math.min(runStart, Math.max(0, openedMessageCount - 1)),
  );
  return { systemCount, earlierCount: Math.max(0, visibleFrom - systemCount), visibleFrom };
}

/** The collapsed earlier row's text (#2640): "3 earlier messages". */
export function earlierMessagesLabel(count: number): string {
  return count === 1 ? "1 earlier message" : `${count} earlier messages`;
}

function isSameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

/** The run header's own time column (#2567) — seconds matter for a run
 * that just started (steps can land seconds apart), so TODAY shows a full
 * time-of-day with seconds; a date-prefixed time (yesterday, or earlier)
 * drops seconds for width and carries the date instead, so every case
 * still fits the fixed column ("Yest 6:48 PM" is no wider than
 * "11:00:02 AM"). `now` is a parameter (default `new Date()`) purely so
 * this is testable without a real clock. */
export function formatRunTime(iso: string, now: Date = new Date()): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  if (isSameDay(date, now)) {
    return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" });
  }
  const time = date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (isSameDay(date, yesterday)) return `Yest ${time}`;
  const datePart = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${datePart} ${time}`;
}
