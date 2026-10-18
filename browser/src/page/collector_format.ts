/** Pure formatting helpers for the Collectors tab's list row (#2568) — the
 * relative "last collected" clock, the row's abbreviated schedule label, and
 * the two section headings, each small enough to be data-in/data-out and
 * tested without jsdom.
 *
 * The row's full plain-words schedule reading still comes from the server's
 * `schedule_description` (describe_schedule, #2428) and is shown verbatim as
 * the row's `title` — `scheduleAbbrev` only picks a short on-row label from
 * the same raw rule `ruleToPreset` already knows how to read. */

import { ruleToPreset, WEEKDAYS } from "./schedule_picker.js";
import { formatDateTime } from "./shared.js";

const MS_PER_MINUTE = 60_000;
const MS_PER_HOUR = 3_600_000;
const MS_PER_DAY = 86_400_000;

/** A relative "last collected" clock: minutes/hours within a day, then
 * yesterday/Nd/Nw, falling back to the shared `formatDateTime` once it's
 * more than a month old. */
export function formatRelativeDate(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const elapsed = Date.now() - then;
  if (elapsed < MS_PER_MINUTE) return "just now";
  if (elapsed < MS_PER_HOUR) return `${Math.floor(elapsed / MS_PER_MINUTE)}m ago`;
  if (elapsed < MS_PER_DAY) return `${Math.floor(elapsed / MS_PER_HOUR)}h ago`;
  const days = Math.floor(elapsed / MS_PER_DAY);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;
  if (days < 30) return `${Math.floor(days / 7)}w ago`;
  return formatDateTime(iso);
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

function formatClock(hour: number, minute: number): string {
  const period = hour < 12 ? "AM" : "PM";
  const displayHour = ((hour + 11) % 12) + 1;
  return `${displayHour}:${pad2(minute)} ${period}`;
}

/** The row's short on-row schedule label — "hourly", "daily 7:30 AM", "Mon
 * 9:00 AM", "custom" — never a reimplementation of `schedule_description`
 * (which carries the full reading, shown in the row's `title`). */
export function scheduleAbbrev(schedule: string | null): string {
  const parsed = ruleToPreset(schedule);
  if (parsed.preset === "none") return "—";
  if (parsed.preset === "hourly") return "hourly";
  if (parsed.preset === "daily") return `daily ${formatClock(parsed.hour, parsed.minute)}`;
  if (parsed.preset === "weekly") {
    const label = WEEKDAYS.find((weekday) => weekday.code === parsed.day)?.label ?? parsed.day;
    return `${label.slice(0, 3)} ${formatClock(parsed.hour, parsed.minute)}`;
  }
  return "custom";
}

/** The "Collectors" section heading's count — matches exactly what's
 * rendered under it: the active count alone when archived collectors are
 * hidden, or the full total plus how many of those are archived once shown
 * (#2568 — the mockup said "hidden" while still showing the archived row). */
export function collectorsHeading(activeCount: number, archivedCount: number, archivedVisible: boolean): string {
  if (archivedVisible && archivedCount > 0) {
    return `Collectors (${activeCount + archivedCount}, ${archivedCount} archived)`;
  }
  return `Collectors (${activeCount})`;
}

/** The "Logs" section heading's count. */
export function logsHeading(count: number): string {
  return `Logs (${count} — system-managed, not collectors)`;
}
