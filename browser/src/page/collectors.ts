/** Collectors tab — one list of collections and logs (replacing the old
 * Collections/Logs/Archived sub-tabs), and each collector's detail: entries,
 * activity, read cursors, config. Restyled to the approved design (#2568);
 * the server's own names (`memories_request`, `MemoryRecord`, the
 * `memory_*`/`entry_*`/`cursor_*` message types) are untouched — only this
 * module's own identifiers, markup and CSS say "collector" now. */

import {
  type CursorRecord,
  type MemoryEntryRecord,
  type MemoryRecord,
  type PromptLogRun,
  type RuntimeCollectionTriggerResult,
  RuntimeMessageType,
  type RuntimeMemoryDetailResponse,
  type RuntimeMemoryPageResponse,
  type RuntimeMemorySaveError,
  type MemorySection,
} from "../protocol.js";
import { createRunRow } from "./prompts.js";
import { createSchedulePicker, type SchedulePickerHandle } from "./schedule_picker.js";
import { collectorsHeading, formatRelativeDate, logsHeading, scheduleAbbrev } from "./collector_format.js";
import { createSearchBanner, formatDateTime, showToast, wireSearchClear } from "./shared.js";

// --- Collectors state ---

const collectorsLoading = document.getElementById("collectors-loading")!;
const collectorsList = document.getElementById("collectors-list")!;
const collectorDetail = document.getElementById("collector-detail")!;
const collectorDetailContent = document.getElementById("collector-detail-content")!;
const collectorDetailBack = document.getElementById("collector-detail-back")!;
const collectorNew = document.getElementById("collector-new")!;
const collectorNewContent = document.getElementById("collector-new-content")!;
const collectorNewBack = document.getElementById("collector-new-back")!;
const archivedToggle = document.getElementById("collectors-archived-toggle")!;
const newCollectorBtn = document.getElementById("collectors-new-btn")!;

let allCollectors: MemoryRecord[] = [];
let activeCollectorName: string | null = null;
let archivedVisible = false;
let collectorSearch = "";

// Detail view pagination — each section accumulates pages independently so
// opening a big collection/log never loads its whole history at once.
let activeCollector: MemoryRecord | null = null;
let entries: MemoryEntryRecord[] = [];
let entriesHasMore = false;
let collectorRuns: PromptLogRun[] = [];
let collectorRunsHasMore = false;
let cursors: CursorRecord[] = [];
// Name of the collection whose extractor is currently running on demand
// (drives the "run extractor" button's disabled/spinner state).
let triggeringCollector: string | null = null;

// ============================================================
// Collectors
// ============================================================

export function setupCollectors(): void {
  collectorDetailBack.addEventListener("click", showCollectorsList);
  collectorNewBack.addEventListener("click", showCollectorsList);
  newCollectorBtn.addEventListener("click", showNewCollectorForm);
  archivedToggle.addEventListener("click", () => {
    archivedVisible = !archivedVisible;
    archivedToggle.classList.toggle("active", archivedVisible);
    renderCollectorsList();
  });
  const search = document.getElementById("collectors-search") as HTMLInputElement | null;
  if (search) {
    let timer = 0;
    search.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        // Server-side so the filter can match entry content too, not just the
        // names/descriptions the list view already holds.
        collectorSearch = search.value.trim();
        if (activeCollectorName) {
          // A collector is open — re-filter its entries (new query or cleared).
          browser.runtime.sendMessage({
            type: RuntimeMessageType.MemoryDetailRequest,
            name: activeCollectorName,
            query: collectorSearch || undefined,
          });
        } else {
          requestCollectors();
        }
      }, 250);
    });
  }
  wireSearchClear("collectors-search", "collectors-search-clear");
}

export function requestCollectors(): void {
  collectorsLoading.classList.remove("hidden");
  browser.runtime.sendMessage({
    type: RuntimeMessageType.MemoriesRequest,
    query: collectorSearch || undefined,
  });
}

export function handleCollectorsResponse(memories: MemoryRecord[]): void {
  allCollectors = memories;
  collectorsLoading.classList.add("hidden");
  renderCollectorsList();
}

export function handleCollectorDetailResponse(message: RuntimeMemoryDetailResponse): void {
  activeCollectorName = message.memory.name;
  activeCollector = message.memory;
  entries = message.entries;
  entriesHasMore = message.entries_has_more;
  collectorRuns = message.collector_runs;
  collectorRunsHasMore = message.collector_runs_has_more;
  cursors = message.cursors;
  showCollectorDetail();
  renderCollectorDetail();
}

export function handleCollectorPageResponse(message: RuntimeMemoryPageResponse): void {
  // Drop pages for a collector the user already navigated away from.
  if (!activeCollector || message.name !== activeCollector.name) return;
  if (message.section === "collector_runs") {
    collectorRuns = collectorRuns.concat(message.runs);
    collectorRunsHasMore = message.has_more;
  } else {
    entries = entries.concat(message.entries);
    entriesHasMore = message.has_more;
  }
  renderCollectorDetail();
}

function requestCollectorPage(section: MemorySection, offset: number): void {
  if (!activeCollector) return;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.MemoryPageRequest,
    name: activeCollector.name,
    section,
    offset,
    // Keep entry pagination filtered to the active search.
    query: section === "entries" ? collectorSearch || undefined : undefined,
  });
}

export function handleCollectorChanged(name: string | null): void {
  // The Collectors tab might not be visible — refresh data only if it is.
  const collectorsPanel = document.getElementById("panel-memories");
  if (!collectorsPanel || collectorsPanel.classList.contains("hidden")) return;
  if (activeCollectorName && (name === null || name === activeCollectorName)) {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: activeCollectorName,
      query: collectorSearch || undefined,
    });
  } else if (!activeCollectorName) {
    requestCollectors();
  }
}

function showCollectorsList(): void {
  activeCollectorName = null;
  collectorDetail.classList.add("hidden");
  collectorDetailContent.innerHTML = "";
  collectorNew.classList.add("hidden");
  collectorNewContent.innerHTML = "";
  collectorsList.classList.remove("hidden");
  requestCollectors();
}

function showCollectorDetail(): void {
  collectorsList.classList.add("hidden");
  collectorNew.classList.add("hidden");
  collectorNewContent.innerHTML = "";
  collectorDetail.classList.remove("hidden");
}

function showNewCollectorForm(): void {
  collectorsList.classList.add("hidden");
  collectorDetail.classList.add("hidden");
  collectorDetailContent.innerHTML = "";
  collectorNew.classList.remove("hidden");
  collectorNewContent.innerHTML = "";
  collectorNewContent.appendChild(createNewCollectorForm());
}

// ============================================================
// List
// ============================================================

function renderCollectorsList(): void {
  collectorsList.innerHTML = "";
  if (collectorSearch) collectorsList.appendChild(createSearchBanner(collectorSearch));

  const active = allCollectors.filter((m) => m.type === "collection" && !m.archived);
  const archived = allCollectors.filter((m) => m.archived);
  const logs = allCollectors.filter((m) => m.type === "log" && !m.archived);

  if (active.length === 0 && archived.length === 0 && logs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "panel-loading";
    empty.textContent = collectorSearch ? "No collectors match." : "No collectors yet.";
    collectorsList.appendChild(empty);
    return;
  }

  collectorsList.appendChild(
    createSectionHeading(collectorsHeading(active.length, archived.length, archivedVisible)),
  );
  for (const record of active) collectorsList.appendChild(createCollectorRow(record));
  if (archivedVisible) {
    for (const record of archived) collectorsList.appendChild(createCollectorRow(record));
  }

  if (logs.length > 0) {
    collectorsList.appendChild(createSectionHeading(logsHeading(logs.length)));
    for (const record of logs) collectorsList.appendChild(createCollectorRow(record));
  }
}

function createSectionHeading(text: string): HTMLElement {
  const heading = document.createElement("div");
  heading.className = "collector-section-heading";
  heading.textContent = text;
  return heading;
}

function createCollectorRow(memory: MemoryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "collector-row";
  if (memory.type === "log") row.classList.add("collector-row-log");
  if (memory.archived) row.classList.add("collector-row-archived");

  const left = document.createElement("div");
  left.className = "collector-row-left";

  const name = document.createElement("span");
  name.className = "collector-row-name";
  name.textContent = memory.name;
  name.title = memory.name;
  left.appendChild(name);

  if (memory.archived) {
    left.appendChild(createChip("archived", "collector-chip-muted", "Archived — hidden unless shown"));
  } else if (memory.published) {
    left.appendChild(createChip("published", "chip--accent", "Notifies on new entries"));
  }

  const description = document.createElement("span");
  description.className = "collector-row-desc";
  description.textContent = memory.description;
  description.title = memory.description;
  left.appendChild(description);

  row.appendChild(left);
  row.appendChild(createCollectorRowRight(memory));

  row.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: memory.name,
      // Carry the active search so the detail view shows only matching entries.
      query: collectorSearch || undefined,
    });
  });

  return row;
}

function createChip(label: string, extraClass: string, title: string): HTMLElement {
  const chip = document.createElement("span");
  chip.className = `chip ${extraClass}`;
  chip.textContent = label;
  chip.title = title;
  return chip;
}

function createCollectorRowRight(memory: MemoryRecord): HTMLElement {
  const right = document.createElement("div");
  right.className =
    memory.type === "log" ? "collector-row-right collector-row-right-log" : "collector-row-right";

  right.appendChild(createMetricCell("fa-list", String(memory.entry_count), `${memory.entry_count} entries`));
  if (memory.type === "collection") {
    right.appendChild(createLastCollectedCell(memory));
    right.appendChild(createScheduleCell(memory));
  }

  const chevron = document.createElement("span");
  chevron.className = "collector-row-go";
  chevron.innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
  right.appendChild(chevron);

  return right;
}

function createMetricCell(iconClass: string, value: string, title: string): HTMLElement {
  const cell = document.createElement("span");
  cell.className = "metric";
  cell.title = title;
  cell.innerHTML = `<i class="fa-solid ${iconClass}"></i><span class="v">${value}</span>`;
  return cell;
}

function createMutedCell(title: string): HTMLElement {
  const cell = document.createElement("span");
  cell.className = "metric collector-cell-muted";
  cell.title = title;
  cell.textContent = "—";
  return cell;
}

function createLastCollectedCell(memory: MemoryRecord): HTMLElement {
  if (memory.last_collected_at) {
    return createMetricCell(
      "fa-clock-rotate-left",
      formatRelativeDate(memory.last_collected_at),
      `Last collected ${formatDateTime(memory.last_collected_at)}`,
    );
  }
  if (memory.extraction_prompt) {
    return createMetricCell("fa-clock-rotate-left", "never", "This collector hasn't run yet");
  }
  return createMutedCell("No extraction prompt — entries are added by hand");
}

function createScheduleCell(memory: MemoryRecord): HTMLElement {
  if (!memory.schedule) return createMutedCell("No schedule set");
  const title = memory.schedule_description
    ? `${memory.schedule_description} — ${memory.schedule}`
    : memory.schedule;
  return createMetricCell("fa-calendar", scheduleAbbrev(memory.schedule), title);
}

// ============================================================
// Detail
// ============================================================

function renderCollectorDetail(): void {
  if (!activeCollector) return;
  const memory = activeCollector;
  collectorDetailContent.innerHTML = "";
  collectorDetailContent.appendChild(createCollectorDetailHeader(memory));
  collectorDetailContent.appendChild(createEntriesSection(memory));
  collectorDetailContent.appendChild(createActivitySection());
  collectorDetailContent.appendChild(createConfigSection(memory));
}

function createCollectorDetailHeader(memory: MemoryRecord): HTMLElement {
  const header = document.createElement("div");
  header.className = "collector-detail-header";

  const title = document.createElement("h1");
  title.textContent = memory.name;
  header.appendChild(title);

  if (memory.published) {
    header.appendChild(createChip("published", "chip--accent", "Notifies on new entries"));
  }
  if (memory.archived) {
    header.appendChild(createChip("archived", "collector-chip-muted", "Archived — hidden from the active list"));
  }
  return header;
}

// The collector's stored entries (and, for collections, the add-entry form).
function createEntriesSection(memory: MemoryRecord): HTMLElement {
  const section = document.createElement("section");
  section.className = "collector-section";

  if (collectorSearch) section.appendChild(createSearchBanner(collectorSearch));

  const heading = document.createElement("h2");
  const shown = entries.length;
  const total = memory.entry_count;
  if (collectorSearch) {
    heading.textContent = `Entries (${shown}${entriesHasMore ? "+" : ""})`;
  } else {
    heading.textContent =
      total > shown ? `Entries (showing ${shown} of ${total}, newest first)` : `Entries (${total})`;
  }
  section.appendChild(heading);

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "collector-entries-empty";
    empty.textContent = collectorSearch ? `No entries match “${collectorSearch}”.` : "No entries yet.";
    section.appendChild(empty);
  } else {
    for (const entry of entries) section.appendChild(createEntryRow(memory, entry));
  }

  if (entriesHasMore) {
    section.appendChild(createLoadMoreButton(() => requestCollectorPage("entries", entries.length)));
  }

  // Logs are append-only by the system — manual entry add is collection-only.
  if (memory.type === "collection") {
    section.appendChild(createEntryAddForm(memory));
  }
  return section;
}

// Collector run history — a native <details>, closed by default (the least
// often needed at a glance). Read cursors live in Config now, not here.
function createActivitySection(): HTMLElement {
  const details = document.createElement("details");
  details.className = "collector-activity";

  const summary = document.createElement("summary");
  summary.className = "collector-activity-summary";
  const count = collectorRuns.length;
  summary.innerHTML =
    '<i class="fa-solid fa-chevron-right collector-chev"></i> Activity ' +
    `<span class="collector-activity-sub">— ${count} collector run${count === 1 ? "" : "s"}</span>`;
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "collector-activity-body";
  if (collectorRuns.length === 0) {
    const empty = document.createElement("div");
    empty.className = "collector-entries-empty";
    empty.textContent = "No collector activity yet.";
    body.appendChild(empty);
  } else {
    for (const run of collectorRuns) body.appendChild(createRunRow(run));
    if (collectorRunsHasMore) {
      body.appendChild(createLoadMoreButton(() => requestCollectorPage("collector_runs", collectorRuns.length)));
    }
  }
  details.appendChild(body);
  return details;
}

// Editable configuration (read-only metadata for system-managed logs).
function createConfigSection(memory: MemoryRecord): HTMLElement {
  const section = document.createElement("section");
  section.className = "collector-section";

  const heading = document.createElement("h2");
  heading.textContent = "Config";
  section.appendChild(heading);

  const card = document.createElement("div");
  card.className = "collector-config-card";
  if (memory.type === "log") {
    appendLogConfig(card, memory);
  } else {
    appendCollectionConfig(card, memory);
  }
  section.appendChild(card);
  return section;
}

function appendLogConfig(card: HTMLElement, memory: MemoryRecord): void {
  const grid = document.createElement("dl");
  grid.className = "collector-readonly-grid";
  appendDef(grid, "Description", memory.description || "—");
  appendDef(grid, "Entries", String(memory.entry_count));
  card.appendChild(grid);
}

function appendCollectionConfig(card: HTMLElement, memory: MemoryRecord): void {
  const fields = createCollectorFormFields({
    description: memory.description,
    published: memory.published,
    extraction_prompt: memory.extraction_prompt ?? "",
  });
  const schedulePicker = createSchedulePicker({
    schedule: memory.schedule,
    description: memory.schedule_description,
  });

  card.appendChild(fieldRow("Description", fields.description));
  card.appendChild(
    toggleRow(
      "collector-notify",
      "Notify on new (published)",
      fields.published,
      "When on, Penny delivers each new entry to you as it's collected.",
    ),
  );
  card.appendChild(fieldRow("Extraction prompt", fields.extractionPrompt));
  card.appendChild(fieldRow("Schedule", schedulePicker.element));

  if (cursors.length > 0) card.appendChild(createCursorsBlock(memory));

  const readOnlyGrid = document.createElement("dl");
  readOnlyGrid.className = "collector-readonly-grid";
  appendDef(readOnlyGrid, "Entries", String(memory.entry_count));
  if (memory.last_collected_at) {
    appendDef(readOnlyGrid, "Last collected", formatDateTime(memory.last_collected_at));
  }
  card.appendChild(readOnlyGrid);

  card.appendChild(createConfigActions(memory, fields, schedulePicker));
}

function createCursorsBlock(memory: MemoryRecord): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "collector-cursors-block";

  const heading = document.createElement("h3");
  heading.textContent = "Read cursors";
  wrapper.appendChild(heading);

  const hint = document.createElement("p");
  hint.className = "collector-field-hint";
  hint.textContent =
    "Where this collector has read up to in each log. Pick an earlier point to " +
    "re-read from there (set it before your data starts to re-read everything); " +
    "Clear starts fresh from the most recent entries.";
  wrapper.appendChild(hint);

  for (const cursor of cursors) wrapper.appendChild(createCursorRow(memory, cursor));
  return wrapper;
}

// datetime-local inputs speak local wall-clock "YYYY-MM-DDTHH:mm"; cursors are
// stored/sent as UTC ISO-8601.  Convert in both directions at the boundary.
function isoToLocalInput(iso: string): string {
  const d = new Date(iso);
  const pad = (n: number): string => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}`
  );
}

function createCursorRow(memory: MemoryRecord, cursor: CursorRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "collector-cursor-row";

  const log = document.createElement("span");
  log.className = "collector-cursor-log";
  log.textContent = cursor.log_name;

  const current = document.createElement("span");
  current.className = "collector-cursor-current";
  current.textContent = `read up to ${formatDateTime(cursor.last_read_at)}`;

  const picker = document.createElement("input");
  picker.type = "datetime-local";
  picker.className = "collector-field-input";
  picker.value = isoToLocalInput(cursor.last_read_at);

  const setBtn = document.createElement("button");
  setBtn.className = "btn";
  setBtn.textContent = "Set";
  setBtn.addEventListener("click", () => {
    if (!picker.value) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.CursorSet,
      name: memory.name,
      log_name: cursor.log_name,
      last_read_at: new Date(picker.value).toISOString(),
    });
    showToast("Cursor set");
  });

  const clearBtn = document.createElement("button");
  clearBtn.className = "btn";
  clearBtn.textContent = "Clear";
  clearBtn.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.CursorClear,
      name: memory.name,
      log_name: cursor.log_name,
    });
    showToast("Cursor cleared");
  });

  row.appendChild(log);
  row.appendChild(current);
  row.appendChild(picker);
  row.appendChild(setBtn);
  row.appendChild(clearBtn);
  return row;
}

function createConfigActions(
  memory: MemoryRecord,
  fields: CollectorFormFields,
  schedulePicker: SchedulePickerHandle,
): HTMLElement {
  const actions = document.createElement("div");
  actions.className = "collector-actions-row";

  // Only collectors with an extraction prompt have a collector to run.
  if (memory.extraction_prompt) {
    actions.appendChild(createRunExtractorButton(memory));
  }

  const spacer = document.createElement("span");
  spacer.className = "collector-actions-spacer";
  actions.appendChild(spacer);

  const archive = document.createElement("button");
  archive.className = "btn";
  archive.textContent = memory.archived ? "Archived" : "Archive";
  archive.disabled = memory.archived;
  archive.addEventListener("click", () => {
    if (!confirm(`Archive "${memory.name}"? It will disappear from the active list.`)) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.MemoryArchive, name: memory.name });
    showToast("Archived");
    showCollectorsList();
  });

  const save = document.createElement("button");
  save.className = "btn btn-primary";
  save.textContent = "Save";
  save.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryUpdate,
      name: memory.name,
      description: fields.description.value.trim(),
      published: fields.published.checked,
      extraction_prompt: fields.extractionPrompt.value.trim() || null,
      schedule: schedulePicker.getValue(),
    });
    showToast("Saved");
  });

  actions.appendChild(archive);
  actions.appendChild(save);
  return actions;
}

function createRunExtractorButton(memory: MemoryRecord): HTMLElement {
  const running = triggeringCollector === memory.name;
  const button = document.createElement("button");
  button.className = "btn btn-run";
  button.disabled = running;
  button.innerHTML = running
    ? '<i class="fa-solid fa-spinner fa-spin"></i> Running…'
    : '<i class="fa-solid fa-play"></i> Run extractor';
  button.addEventListener("click", () => {
    triggeringCollector = memory.name;
    browser.runtime.sendMessage({ type: RuntimeMessageType.CollectionTrigger, name: memory.name });
    renderCollectorDetail(); // reflect the running state immediately
  });
  return button;
}

export function handleCollectionTriggerResult(message: RuntimeCollectionTriggerResult): void {
  if (triggeringCollector === message.name) triggeringCollector = null;
  showToast(message.success ? "Extractor finished" : `Extractor failed: ${message.message}`);
  // Refresh the detail so re-enabled button, new entries, and updated
  // "last collected" all reflect the run (also arrives via memory_changed).
  if (activeCollector && activeCollector.name === message.name) renderCollectorDetail();
}

// A memory_create/memory_update whose schedule the server rejected (#2442) —
// the optimistic "Created"/"Saved" toast already fired, so this one follows
// it with the real outcome rather than leaving the rejection unsaid. Nothing
// was persisted, so there is no detail to refresh.
export function handleCollectorSaveError(message: RuntimeMemorySaveError): void {
  const verb = message.action === "create" ? "Create" : "Save";
  showToast(`${verb} failed: ${message.message}`);
}

function appendDef(grid: HTMLElement, label: string, value: string): void {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  grid.appendChild(dt);
  grid.appendChild(dd);
}

// ============================================================
// New collector
// ============================================================

function createNewCollectorForm(): HTMLElement {
  const wrapper = document.createElement("div");

  const heading = document.createElement("h1");
  heading.textContent = "New collector";
  wrapper.appendChild(heading);

  const card = document.createElement("div");
  card.className = "collector-config-card";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "collector-name";
  nameInput.className = "collector-field-input mono";

  const fields = createCollectorFormFields({
    description: "",
    published: false,
    extraction_prompt: "",
  });
  const schedulePicker = createSchedulePicker({ schedule: null, description: null });

  card.appendChild(fieldRow("Name", nameInput));
  card.appendChild(fieldRow("Description", fields.description));
  card.appendChild(
    toggleRow(
      "new-collector-notify",
      "Notify on new (published)",
      fields.published,
      "When on, Penny delivers each new entry to you as it's collected.",
    ),
  );
  card.appendChild(fieldRow("Extraction prompt", fields.extractionPrompt));
  card.appendChild(fieldRow("Schedule", schedulePicker.element));

  const actions = document.createElement("div");
  actions.className = "collector-actions-row";

  const cancel = document.createElement("button");
  cancel.className = "btn";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", showCollectorsList);

  const create = document.createElement("button");
  create.className = "btn btn-primary";
  create.textContent = "Create";
  create.addEventListener("click", () => {
    const name = nameInput.value.trim();
    if (!name) {
      showToast("Name is required");
      return;
    }
    const promptValue = fields.extractionPrompt.value.trim();
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryCreate,
      name,
      description: fields.description.value.trim(),
      published: fields.published.checked,
      extraction_prompt: promptValue || null,
      schedule: schedulePicker.getValue(),
    });
    showToast("Created");
    showCollectorsList();
  });

  actions.appendChild(cancel);
  actions.appendChild(create);
  card.appendChild(actions);

  wrapper.appendChild(card);
  return wrapper;
}

// --- Shared form building blocks (detail's Config section + New collector) ---

interface CollectorFormFields {
  description: HTMLTextAreaElement;
  published: HTMLInputElement;
  extractionPrompt: HTMLTextAreaElement;
}

function createCollectorFormFields(initial: {
  description: string;
  published: boolean;
  extraction_prompt: string;
}): CollectorFormFields {
  const description = document.createElement("textarea");
  description.className = "collector-field-input";
  description.rows = 2;
  description.value = initial.description;

  // Pub/sub: when checked, the notifier delivers new entries to the user.
  const published = document.createElement("input");
  published.type = "checkbox";
  published.checked = initial.published;

  const extractionPrompt = document.createElement("textarea");
  extractionPrompt.className = "collector-field-input mono";
  extractionPrompt.rows = 6;
  // Some prompts have been stored with literal "\n" escape sequences
  // instead of real newlines (chat-side tool calls or the model itself
  // double-escaped).  Render those as real newlines so the textarea
  // shows a multi-line prompt; saving from this form normalises the DB
  // value naturally.
  extractionPrompt.value = unescapeNewlines(initial.extraction_prompt);

  return { description, published, extractionPrompt };
}

function fieldRow(label: string, control: HTMLElement): HTMLElement {
  const row = document.createElement("div");
  row.className = "collector-field-row";
  const text = document.createElement("span");
  text.className = "collector-field-label";
  text.textContent = label;
  row.appendChild(text);
  row.appendChild(control);
  return row;
}

function toggleRow(id: string, label: string, checkbox: HTMLInputElement, hint: string): HTMLElement {
  const row = document.createElement("div");
  row.className = "collector-field-row";

  const inner = document.createElement("div");
  inner.className = "collector-toggle-row";
  checkbox.id = id;
  inner.appendChild(checkbox);
  const text = document.createElement("label");
  text.htmlFor = id;
  text.className = "collector-field-label";
  text.textContent = label;
  inner.appendChild(text);
  row.appendChild(inner);

  const hintEl = document.createElement("span");
  hintEl.className = "collector-field-hint";
  hintEl.textContent = hint;
  row.appendChild(hintEl);
  return row;
}

// "Load more" affordance shared by the detail view's paginated sections.
function createLoadMoreButton(onClick: () => void): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-load-more";
  const button = document.createElement("button");
  button.className = "load-more-btn";
  button.innerHTML = '<i class="fa-solid fa-chevron-down"></i> Load more';
  button.addEventListener("click", onClick);
  wrapper.appendChild(button);
  return wrapper;
}

// --- Entries ---

// Long log entries (esp. ``user-messages`` / ``penny-messages``) get
// collapsed by default so the list stays scannable.  CSS clips after
// ~20 visual lines; the JS heuristic decides whether to clip at all.
const ENTRY_COLLAPSE_LINES = 20;
const ENTRY_COLLAPSE_CHARS = 600;

function createEntryRow(memory: MemoryRecord, entry: MemoryEntryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "collector-entry";

  const header = document.createElement("div");
  header.className = "collector-entry-header";

  if (entry.key) {
    // Collections: the key is the title.
    const key = document.createElement("span");
    key.className = "collector-entry-key";
    key.textContent = entry.key;
    header.appendChild(key);
  }

  // For logs the timestamp IS the identifier; rendered prominently
  // either way so the eye lands on it quickly when scanning.
  const time = document.createElement("span");
  time.className = entry.key ? "collector-entry-date" : "collector-entry-date collector-entry-date-primary";
  time.textContent = formatDateTime(entry.created_at);
  header.appendChild(time);

  // Logs distinguish authors (user vs penny); a collection's entries are all
  // its collector's, so the author just repeats the collector name — drop it.
  if (memory.type === "log") {
    const author = document.createElement("span");
    author.className = "collector-entry-author";
    author.textContent = entry.author;
    header.appendChild(author);
  }

  // Edit/delete only for collection entries (entry_update / entry_delete are keyed).
  if (memory.type === "collection" && entry.key) {
    header.appendChild(createEntryActions(memory, entry, row));
  }

  const content = document.createElement("div");
  content.className = "collector-entry-content";
  content.textContent = entry.content;

  row.appendChild(header);
  row.appendChild(content);

  if (shouldCollapseEntry(entry.content)) {
    content.classList.add("collapsed");
    row.appendChild(createEntryCollapseToggle(content));
  }

  return row;
}

function shouldCollapseEntry(content: string): boolean {
  return content.split("\n").length > ENTRY_COLLAPSE_LINES || content.length > ENTRY_COLLAPSE_CHARS;
}

function createEntryCollapseToggle(content: HTMLElement): HTMLElement {
  const toggle = document.createElement("button");
  toggle.className = "collector-entry-toggle";
  toggle.textContent = "Show more";
  toggle.addEventListener("click", () => {
    const stillCollapsed = content.classList.toggle("collapsed");
    toggle.textContent = stillCollapsed ? "Show more" : "Show less";
  });
  return toggle;
}

function createEntryActions(memory: MemoryRecord, entry: MemoryEntryRecord, row: HTMLElement): HTMLElement {
  const actions = document.createElement("span");
  actions.className = "collector-entry-actions";

  const edit = document.createElement("button");
  edit.className = "icon-btn";
  edit.title = "Edit";
  edit.innerHTML = '<i class="fa-solid fa-pen"></i>';
  edit.addEventListener("click", (e) => {
    e.stopPropagation();
    enterEditMode(memory, entry, row);
  });

  const del = document.createElement("button");
  del.className = "icon-btn";
  del.title = "Delete";
  del.innerHTML = '<i class="fa-solid fa-trash"></i>';
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!entry.key) return;
    if (!confirm(`Delete "${entry.key}"?`)) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.EntryDelete,
      memory: memory.name,
      key: entry.key,
    });
    showToast("Deleted");
  });

  actions.appendChild(edit);
  actions.appendChild(del);
  return actions;
}

function enterEditMode(memory: MemoryRecord, entry: MemoryEntryRecord, row: HTMLElement): void {
  if (!entry.key) return;
  const content = row.querySelector(".collector-entry-content") as HTMLElement | null;
  if (!content) return;

  const textarea = document.createElement("textarea");
  textarea.className = "collector-field-input mono";
  textarea.value = entry.content;
  textarea.rows = Math.max(3, entry.content.split("\n").length);

  const actions = document.createElement("div");
  actions.className = "collector-actions-row";
  const cancel = document.createElement("button");
  cancel.className = "btn";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => content.replaceWith(restored));
  const save = document.createElement("button");
  save.className = "btn btn-primary";
  save.textContent = "Save";
  save.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.EntryUpdate,
      memory: memory.name,
      key: entry.key as string,
      content: textarea.value,
    });
    showToast("Saved");
  });

  const restored = content.cloneNode(true) as HTMLElement;

  const wrapper = document.createElement("div");
  wrapper.className = "collector-entry-content";
  wrapper.appendChild(textarea);
  actions.appendChild(cancel);
  actions.appendChild(save);
  wrapper.appendChild(actions);

  content.replaceWith(wrapper);
}

function createEntryAddForm(memory: MemoryRecord): HTMLElement {
  const form = document.createElement("div");
  form.className = "collector-entry-add";

  const row = document.createElement("div");
  row.className = "collector-entry-add-row";

  const keyInput = document.createElement("input");
  keyInput.type = "text";
  keyInput.placeholder = "key";
  keyInput.className = "collector-field-input mono collector-entry-key-input";

  const contentInput = document.createElement("textarea");
  contentInput.placeholder = "content";
  contentInput.className = "collector-field-input";
  contentInput.rows = 2;

  row.appendChild(keyInput);
  row.appendChild(contentInput);
  form.appendChild(row);

  const actions = document.createElement("div");
  actions.className = "collector-entry-add-actions";
  const submit = document.createElement("button");
  submit.className = "btn btn-primary";
  submit.innerHTML = '<i class="fa-solid fa-plus"></i> Add entry';
  submit.addEventListener("click", () => {
    const key = keyInput.value.trim();
    const content = contentInput.value.trim();
    if (!key || !content) {
      showToast("Key and content required");
      return;
    }
    browser.runtime.sendMessage({
      type: RuntimeMessageType.EntryCreate,
      memory: memory.name,
      key,
      content,
    });
    keyInput.value = "";
    contentInput.value = "";
    showToast("Added");
  });
  actions.appendChild(submit);
  form.appendChild(actions);
  return form;
}

/** Convert literal ``\n`` escape sequences to real newlines.  Used on
 * extraction prompts before rendering them in the textarea — chat-side
 * tool calls (or the model double-escaping) have stored a few prompts
 * with the two-character escape instead of an actual newline. */
function unescapeNewlines(text: string): string {
  return text.replace(/\\n/g, "\n");
}
