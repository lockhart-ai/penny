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
