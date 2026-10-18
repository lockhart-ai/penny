// The collection form's schedule control (#2442): three presets (hourly, daily
// at a time, weekly on a day at a time) plus a raw-RRULE box for anything
// else — the code owner's decided shape.
//
// `ruleToPreset`/`presetToRule` only parse and build the RRULE's own
// NAME=VALUE syntax — mechanical splitting, not prose. The PLAIN-WORDS
// reading (what "FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0" actually SAYS) is
// never re-derived here: for the three presets it's the preset's own fixed
// caption (built from the same hour/minute/day the control already holds),
// and for a stored collection's current value it comes verbatim from the
// server's `schedule_description` (`describe_schedule`, #2428), passed in as
// `initial.description`. The custom box gets no live preview while typing —
// there is no safe way to preview arbitrary RRULE text without either a
// server round trip or duplicating that formatter in TypeScript — so its
// caption only reflects the record's last known `schedule_description`; a
// save that changes the schedule re-renders the whole form from the
// refreshed record (the existing `memory_changed` push), which is what
// brings the caption current, not anything this module does on its own.

export type SchedulePreset = "none" | "hourly" | "daily" | "weekly" | "custom";

export const WEEKDAYS: { code: string; label: string }[] = [
  { code: "MO", label: "Monday" },
  { code: "TU", label: "Tuesday" },
  { code: "WE", label: "Wednesday" },
  { code: "TH", label: "Thursday" },
  { code: "FR", label: "Friday" },
  { code: "SA", label: "Saturday" },
  { code: "SU", label: "Sunday" },
];

const DEFAULT_HOUR = 9;
const DEFAULT_MINUTE = 0;
const DEFAULT_DAY = "MO";

interface ParsedSchedule {
  preset: SchedulePreset;
  hour: number;
  minute: number;
  day: string; // a WEEKDAYS code; meaningful only when preset === "weekly"
}

function pad2(n: number): string {
  return String(n).padStart(2, "0");
}

/** A preset's own RRULE text — the inverse of `ruleToPreset`. `null` for "none"
 * (no schedule); "custom" has no built form — its text IS the raw box. */
export function presetToRule(
  preset: SchedulePreset,
  hour: number,
  minute: number,
  day: string,
): string | null {
  if (preset === "hourly") return "FREQ=HOURLY";
  if (preset === "daily") return `FREQ=DAILY;BYHOUR=${hour};BYMINUTE=${minute}`;
  if (preset === "weekly") return `FREQ=WEEKLY;BYDAY=${day};BYHOUR=${hour};BYMINUTE=${minute}`;
  return null;
}

/** One rule line's own `NAME=VALUE` parts, keyed upper-case — a mechanical
 * split of the grammar's syntax, not an interpretation of what it means. */
function ruleParts(ruleLine: string): Record<string, string> {
  const parts: Record<string, string> = {};
  for (const part of ruleLine.split(";")) {
    const idx = part.indexOf("=");
    if (idx < 0) continue;
    parts[part.slice(0, idx).trim().toUpperCase()] = part.slice(idx + 1).trim();
  }
  return parts;
}

const FALLBACK: ParsedSchedule = {
  preset: "custom",
  hour: DEFAULT_HOUR,
  minute: DEFAULT_MINUTE,
  day: DEFAULT_DAY,
};

/** Which preset (if any) a stored schedule matches EXACTLY — the inverse of
 * `presetToRule`. A schedule with a DTSTART line, an end condition
 * (COUNT=/UNTIL=), more than one BYDAY, or any other shape these three
 * presets can't produce reads as "custom", with the raw rule shown verbatim
 * (the code owner's ruling) rather than a lossy best-effort match. */
export function ruleToPreset(schedule: string | null): ParsedSchedule {
  if (schedule === null) {
    return { preset: "none", hour: DEFAULT_HOUR, minute: DEFAULT_MINUTE, day: DEFAULT_DAY };
  }
  const lines = schedule
    .split(/\\n|\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const ruleLines = lines.filter((line) => !line.toUpperCase().startsWith("DTSTART"));
  if (ruleLines.length !== lines.length || ruleLines.length !== 1) return FALLBACK;

  const parts = ruleParts(ruleLines[0].replace(/^RRULE:/i, ""));
  const keys = new Set(Object.keys(parts));

  if (parts.FREQ === "HOURLY" && keys.size === 1) {
    return { preset: "hourly", hour: DEFAULT_HOUR, minute: DEFAULT_MINUTE, day: DEFAULT_DAY };
  }

  if (parts.FREQ === "DAILY" && keys.size === 3 && keys.has("BYHOUR") && keys.has("BYMINUTE")) {
    const hour = Number(parts.BYHOUR);
    const minute = Number(parts.BYMINUTE);
    if (Number.isInteger(hour) && Number.isInteger(minute)) {
      return { preset: "daily", hour, minute, day: DEFAULT_DAY };
    }
  }

  if (
    parts.FREQ === "WEEKLY" &&
    keys.size === 4 &&
    keys.has("BYDAY") &&
    keys.has("BYHOUR") &&
    keys.has("BYMINUTE") &&
    WEEKDAYS.some((weekday) => weekday.code === parts.BYDAY)
  ) {
    const hour = Number(parts.BYHOUR);
    const minute = Number(parts.BYMINUTE);
    if (Number.isInteger(hour) && Number.isInteger(minute)) {
      return { preset: "weekly", hour, minute, day: parts.BYDAY };
    }
  }

  return FALLBACK;
}

function formatClock(hour: number, minute: number): string {
  const period = hour < 12 ? "AM" : "PM";
  const displayHour = ((hour + 11) % 12) + 1;
  return `${displayHour}:${pad2(minute)} ${period}`;
}

/** The fixed caption each preset can produce — never a reimplementation of
 * `describe_schedule`, just these three presets' own words for themselves. */
function presetCaption(preset: SchedulePreset, hour: number, minute: number, day: string): string {
  if (preset === "none") return "";
  if (preset === "hourly") return "every hour";
  if (preset === "daily") return `every day at ${formatClock(hour, minute)}`;
  if (preset === "weekly") {
    const label = WEEKDAYS.find((weekday) => weekday.code === day)?.label ?? day;
    return `every ${label} at ${formatClock(hour, minute)}`;
  }
  return ""; // "custom" — filled by setDescription, never computed here
}

function makeSelect(options: { value: string; label: string }[], selected: string): HTMLSelectElement {
  const select = document.createElement("select");
  select.className = "collector-field-input";
  for (const option of options) {
    const el = document.createElement("option");
    el.value = option.value;
    el.textContent = option.label;
    if (option.value === selected) el.selected = true;
    select.appendChild(el);
  }
  return select;
}

/** The preset choice as a segmented control (the shared `.seg`), one button
 * per preset — the mockup's own shape, in place of a `<select>`. */
function makePresetSeg(selected: SchedulePreset, onChange: (preset: SchedulePreset) => void): HTMLElement {
  const seg = document.createElement("div");
  seg.className = "seg";
  for (const option of PRESET_OPTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = option.label;
    button.classList.toggle("active", option.value === selected);
    button.addEventListener("click", () => {
      for (const sibling of Array.from(seg.querySelectorAll("button"))) {
        sibling.classList.toggle("active", sibling === button);
      }
      onChange(option.value);
    });
    seg.appendChild(button);
  }
  return seg;
}

function timeValue(hour: number, minute: number): string {
  return `${pad2(hour)}:${pad2(minute)}`;
}

/** An `<input type="time">`'s value (always 24-hour "HH:MM" per the HTML spec,
 * regardless of display locale) as `[hour, minute]`, falling back to the
 * default time when the control is momentarily empty. */
function parseTimeInput(value: string): [number, number] {
  const [hour, minute] = value.split(":").map(Number);
  if (Number.isInteger(hour) && Number.isInteger(minute)) return [hour, minute];
  return [DEFAULT_HOUR, DEFAULT_MINUTE];
}

const PRESET_OPTIONS: { value: SchedulePreset; label: string }[] = [
  { value: "none", label: "No schedule" },
  { value: "hourly", label: "Hourly" },
  { value: "daily", label: "Daily at a time" },
  { value: "weekly", label: "Weekly on a day" },
  { value: "custom", label: "Custom (raw rule)" },
];

export interface SchedulePickerHandle {
  element: HTMLElement;
  /** The schedule to send: an RRULE string, or `null` for "no schedule". */
  getValue(): string | null;
}

export function createSchedulePicker(initial: {
  schedule: string | null;
  description: string | null;
}): SchedulePickerHandle {
  const parsed = ruleToPreset(initial.schedule);
  const customDescription = initial.description;
  let currentPreset: SchedulePreset = parsed.preset;

  const wrapper = document.createElement("div");
  wrapper.className = "collector-schedule-picker";

  const timeInput = document.createElement("input");
  timeInput.type = "time";
  timeInput.className = "collector-field-input";
  timeInput.value = timeValue(parsed.hour, parsed.minute);

  const daySelect = makeSelect(
    WEEKDAYS.map((weekday) => ({ value: weekday.code, label: weekday.label })),
    parsed.day,
  );

  const customBox = document.createElement("textarea");
  customBox.className = "collector-field-input mono";
  customBox.rows = 2;
  customBox.placeholder = "FREQ=DAILY;BYHOUR=8";
  customBox.value = parsed.preset === "custom" ? (initial.schedule ?? "") : "";

  const subRow = document.createElement("div");
  subRow.className = "collector-schedule-sub";

  const caption = document.createElement("div");
  caption.className = "collector-plain-words";

  const seg = makePresetSeg(currentPreset, (preset) => {
    currentPreset = preset;
    renderSubControls();
    renderCaption();
  });

  function renderSubControls(): void {
    subRow.innerHTML = "";
    if (currentPreset === "daily") {
      subRow.appendChild(timeInput);
    } else if (currentPreset === "weekly") {
      subRow.appendChild(daySelect);
      subRow.appendChild(timeInput);
    } else if (currentPreset === "custom") {
      subRow.appendChild(customBox);
    }
    subRow.classList.toggle("hidden", currentPreset === "none" || currentPreset === "hourly");
  }

  // The plain-words line always shows the human reading; a preset also gets
  // its own current raw rule alongside it in a `<code>` span. "custom" gets
  // no live preview (the box IS the raw text already); "none" has no
  // schedule to describe at all, so the whole line disappears.
  function renderCaption(): void {
    caption.innerHTML = "";
    caption.classList.toggle("hidden", currentPreset === "none");
    if (currentPreset === "none") return;
    const text = document.createElement("span");
    if (currentPreset === "custom") {
      text.textContent = customDescription ?? "";
      caption.appendChild(text);
      return;
    }
    const [hour, minute] = parseTimeInput(timeInput.value);
    const rule = presetToRule(currentPreset, hour, minute, daySelect.value);
    text.textContent = presetCaption(currentPreset, hour, minute, daySelect.value);
    const code = document.createElement("code");
    code.textContent = rule ?? "";
    caption.appendChild(text);
    caption.appendChild(code);
  }

  timeInput.addEventListener("input", renderCaption);
  daySelect.addEventListener("change", renderCaption);

  renderSubControls();
  renderCaption();

  wrapper.appendChild(seg);
  wrapper.appendChild(subRow);
  wrapper.appendChild(caption);

  return {
    element: wrapper,
    getValue(): string | null {
      if (currentPreset === "none") return null;
      if (currentPreset === "custom") return customBox.value.trim() || null;
      const [hour, minute] = parseTimeInput(timeInput.value);
      return presetToRule(currentPreset, hour, minute, daySelect.value);
    },
  };
}
