/** Memories tab (the collectors view) — collections, logs, archived, and
 * each memory's detail: entries, activity, cursors, metadata.  Split out of
 * page.ts (#2565); every function here moved unchanged, only `export`/`import`
 * lines were added. */

import {
  type CursorRecord,
  type MemoryEntryRecord,
  type MemoryRecord,
  type MemorySection,
  type PromptLogRun,
  type RuntimeCollectionTriggerResult,
  RuntimeMessageType,
  type RuntimeMemoryDetailResponse,
  type RuntimeMemoryPageResponse,
  type RuntimeMemorySaveError,
} from "../protocol.js";
import { createRunRow } from "./prompts.js";
import { createSchedulePicker } from "./schedule_picker.js";
import { createSearchBanner, formatDateTime, showToast, wireSearchClear } from "./shared.js";

// --- Memories state ---

const memoriesLoading = document.getElementById("memories-loading")!;
const memoriesList = document.getElementById("memories-list")!;
const memoryDetail = document.getElementById("memory-detail")!;
const memoryDetailContent = document.getElementById("memory-detail-content")!;
const memoryDetailBack = document.getElementById("memory-detail-back")!;

type MemoryTab = "collections" | "logs" | "archived";

let allMemories: MemoryRecord[] = [];
let activeMemoryName: string | null = null;
let activeMemoryTab: MemoryTab = "collections";
let memorySearch = "";

// Detail view pagination — each section accumulates pages independently so
// opening a big collection/log never loads its whole history at once.
let activeMemory: MemoryRecord | null = null;
let memoryEntries: MemoryEntryRecord[] = [];
let memoryEntriesHasMore = false;
let memoryRuns: PromptLogRun[] = [];
let memoryRunsHasMore = false;
let memoryCursors: CursorRecord[] = [];
// Name of the collection whose extractor is currently running on demand
// (drives the "run extractor" button's disabled/spinner state).
let triggeringCollection: string | null = null;

// ============================================================
// Memories
// ============================================================

// ============================================================
// Memories
// ============================================================

export function setupMemories(): void {
  memoryDetailBack.addEventListener("click", showMemoriesList);
  for (const btn of Array.from(document.querySelectorAll("#memory-tabs .sub-tab"))) {
    btn.addEventListener("click", () => {
      const tab = btn.getAttribute("data-mtab") as MemoryTab | null;
      if (!tab) return;
      activeMemoryTab = tab;
      for (const b of Array.from(document.querySelectorAll("#memory-tabs .sub-tab"))) {
        b.classList.toggle("active", b === btn);
      }
      // Sub-tab switch returns to the list view.
      activeMemoryName = null;
      memoryDetail.classList.add("hidden");
      memoriesList.classList.remove("hidden");
      renderMemoriesList();
    });
  }
  const search = document.getElementById("memories-search") as HTMLInputElement | null;
  if (search) {
    let timer = 0;
    search.addEventListener("input", () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        // Server-side so the filter can match entry content too, not just the
        // names/descriptions the list view already holds.
        memorySearch = search.value.trim();
        if (activeMemoryName) {
          // A collection is open — re-filter its entries (new query or cleared).
          browser.runtime.sendMessage({
            type: RuntimeMessageType.MemoryDetailRequest,
            name: activeMemoryName,
            query: memorySearch || undefined,
          });
        } else {
          requestMemories();
        }
      }, 250);
    });
  }
  wireSearchClear("memories-search", "memories-search-clear");
}

export function requestMemories(): void {
  memoriesLoading.classList.remove("hidden");
  browser.runtime.sendMessage({
    type: RuntimeMessageType.MemoriesRequest,
    query: memorySearch || undefined,
  });
}

export function handleMemoriesResponse(memories: MemoryRecord[]): void {
  allMemories = memories;
  memoriesLoading.classList.add("hidden");
  renderMemoriesList();
}

export function handleMemoryDetailResponse(message: RuntimeMemoryDetailResponse): void {
  activeMemoryName = message.memory.name;
  activeMemory = message.memory;
  memoryEntries = message.entries;
  memoryEntriesHasMore = message.entries_has_more;
  memoryRuns = message.collector_runs;
  memoryRunsHasMore = message.collector_runs_has_more;
  memoryCursors = message.cursors;
  showMemoryDetail();
  renderMemoryDetail();
}

export function handleMemoryPageResponse(message: RuntimeMemoryPageResponse): void {
  // Drop pages for a memory the user already navigated away from.
  if (!activeMemory || message.name !== activeMemory.name) return;
  if (message.section === "collector_runs") {
    memoryRuns = memoryRuns.concat(message.runs);
    memoryRunsHasMore = message.has_more;
  } else {
    memoryEntries = memoryEntries.concat(message.entries);
    memoryEntriesHasMore = message.has_more;
  }
  renderMemoryDetail();
}

function requestMemoryPage(section: MemorySection, offset: number): void {
  if (!activeMemory) return;
  browser.runtime.sendMessage({
    type: RuntimeMessageType.MemoryPageRequest,
    name: activeMemory.name,
    section,
    offset,
    // Keep entry pagination filtered to the active search.
    query: section === "entries" ? memorySearch || undefined : undefined,
  });
}

export function handleMemoryChanged(name: string | null): void {
  // The memories tab might not be visible — refresh data only if it is.
  const memoriesPanel = document.getElementById("panel-memories");
  if (!memoriesPanel || memoriesPanel.classList.contains("hidden")) return;
  if (activeMemoryName && (name === null || name === activeMemoryName)) {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: activeMemoryName,
      query: memorySearch || undefined,
    });
  } else if (!activeMemoryName) {
    requestMemories();
  }
}

function showMemoriesList(): void {
  activeMemoryName = null;
  memoryDetail.classList.add("hidden");
  memoriesList.classList.remove("hidden");
  requestMemories();
}

function showMemoryDetail(): void {
  memoriesList.classList.add("hidden");
  memoryDetail.classList.remove("hidden");
}

function renderMemoriesList(): void {
  memoriesList.innerHTML = "";
  // The "+" affordance only makes sense on the collections tab — that's
  // the only shape users can create from the addon.
  if (activeMemoryTab === "collections") {
    memoriesList.appendChild(createNewMemoryControl());
  }
  if (memorySearch) memoriesList.appendChild(createSearchBanner(memorySearch));
  const visible = allMemories.filter(memoryMatchesTab);
  if (visible.length === 0) {
    const empty = document.createElement("div");
    empty.className = "panel-loading";
    empty.textContent = memorySearch ? "No memories match." : emptyLabel(activeMemoryTab);
    memoriesList.appendChild(empty);
    return;
  }
  for (const memory of visible) {
    memoriesList.appendChild(createMemoryRow(memory));
  }
}

function memoryMatchesTab(memory: MemoryRecord): boolean {
  if (activeMemoryTab === "archived") return memory.archived;
  if (memory.archived) return false;
  return activeMemoryTab === "collections" ? memory.type === "collection" : memory.type === "log";
}

function emptyLabel(tab: MemoryTab): string {
  if (tab === "collections") return "No collections yet.";
  if (tab === "logs") return "No logs yet.";
  return "Nothing archived.";
}

function createNewMemoryControl(): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-new-control";

  const button = document.createElement("button");
  button.className = "memory-new-btn";
  button.innerHTML = '<i class="fa-solid fa-plus"></i> New collection';
  button.addEventListener("click", () => {
    wrapper.replaceWith(createNewMemoryForm());
  });
  wrapper.appendChild(button);
  return wrapper;
}

function createNewMemoryForm(): HTMLElement {
  const form = document.createElement("div");
  form.className = "memory-new-form";

  const fields = createMemoryFormFields({
    description: "",
    published: false,
    extraction_prompt: "",
  });
  const schedulePicker = createSchedulePicker({ schedule: null, description: null });
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "collection-name";
  nameInput.className = "memory-form-input";

  form.appendChild(labelled("Name", nameInput));
  form.appendChild(labelled("Description", fields.description));
  form.appendChild(labelled("Notify on new (published)", fields.published));
  form.appendChild(labelled("Extraction prompt", fields.extractionPrompt));
  form.appendChild(labelled("Schedule", schedulePicker.element));

  const actions = document.createElement("div");
  actions.className = "memory-form-actions";

  const cancel = document.createElement("button");
  cancel.className = "memory-form-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => renderMemoriesList());

  const create = document.createElement("button");
  create.className = "memory-form-save";
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
  });

  actions.appendChild(cancel);
  actions.appendChild(create);
  form.appendChild(actions);
  return form;
}

interface MemoryFormFields {
  description: HTMLTextAreaElement;
  published: HTMLInputElement;
  extractionPrompt: HTMLTextAreaElement;
}

function createMemoryFormFields(initial: {
  description: string;
  published: boolean;
  extraction_prompt: string;
}): MemoryFormFields {
  const description = document.createElement("textarea");
  description.className = "memory-form-input";
  description.rows = 2;
  description.value = initial.description;

  // Pub/sub: when checked, the notifier delivers new entries to the user.
  const published = document.createElement("input");
  published.type = "checkbox";
  published.className = "memory-form-checkbox";
  published.checked = initial.published;

  const extractionPrompt = document.createElement("textarea");
  extractionPrompt.className = "memory-form-input memory-form-prompt";
  extractionPrompt.rows = 6;
  // Some prompts have been stored with literal "\n" escape sequences
  // instead of real newlines (chat-side tool calls or the model itself
  // double-escaped).  Render those as real newlines so the textarea
  // shows a multi-line prompt; saving from this form normalises the DB
  // value naturally.
  extractionPrompt.value = unescapeNewlines(initial.extraction_prompt);

  return { description, published, extractionPrompt };
}

function labelled(label: string, control: HTMLElement): HTMLElement {
  const wrapper = document.createElement("label");
  wrapper.className = "memory-form-row";
  const text = document.createElement("span");
  text.className = "memory-form-label";
  text.textContent = label;
  wrapper.appendChild(text);
  wrapper.appendChild(control);
  return wrapper;
}

function createMemoryRow(memory: MemoryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "memory-row";
  if (memory.archived) row.classList.add("memory-row-archived");

  const name = document.createElement("span");
  name.className = "memory-name";
  name.textContent = memory.name;

  const badge = document.createElement("span");
  badge.className = `memory-type-badge ${memory.type}`;
  badge.textContent = memory.type;

  const description = document.createElement("span");
  description.className = "memory-description";
  description.textContent = memory.description;

  const meta = document.createElement("span");
  meta.className = "memory-meta";
  meta.appendChild(metaItem("fa-list", `${memory.entry_count} entries`));
  if (memory.last_collected_at) {
    meta.appendChild(metaItem("fa-clock-rotate-left", formatRelativeDate(memory.last_collected_at)));
  } else if (memory.extraction_prompt) {
    meta.appendChild(metaItem("fa-clock-rotate-left", "never"));
  }

  row.appendChild(name);
  row.appendChild(badge);
  if (memory.published) {
    const publishedBadge = document.createElement("span");
    publishedBadge.className = "memory-type-badge published";
    publishedBadge.textContent = "published";
    row.appendChild(publishedBadge);
  }
  row.appendChild(description);
  row.appendChild(meta);

  row.addEventListener("click", () => {
    browser.runtime.sendMessage({
      type: RuntimeMessageType.MemoryDetailRequest,
      name: memory.name,
      // Carry the active search so the detail view shows only matching entries.
      query: memorySearch || undefined,
    });
  });

  return row;
}

function metaItem(iconClass: string, text: string): HTMLSpanElement {
  const span = document.createElement("span");
  span.innerHTML = `<i class="fa-solid ${iconClass}"></i>${text}`;
  return span;
}

function renderMemoryDetail(): void {
  if (!activeMemory) return;
  const memory = activeMemory;
  memoryDetailContent.innerHTML = "";

  memoryDetailContent.appendChild(createMemoryHeader(memory));
  memoryDetailContent.appendChild(createMemoryDetailTabs(memory));
}

// Entries / Activity / Config as switchable panels under the header, replacing
// the old single-page vertical stack.  Defaults to Entries.
function createMemoryDetailTabs(memory: MemoryRecord): HTMLElement {
  const wrapper = document.createElement("div");
  wrapper.className = "memory-detail-tabs-wrapper";

  const tabs = [
    { label: "Entries", panel: createEntriesPanel(memory) },
    { label: "Activity", panel: createActivityPanel(memory) },
    { label: "Config", panel: createConfigPanel(memory) },
  ];

  const bar = document.createElement("div");
  bar.className = "memory-detail-tabs";
  tabs.forEach((tab, index) => {
    const button = document.createElement("button");
    button.className = index === 0 ? "sub-tab active" : "sub-tab";
    button.textContent = tab.label;
    tab.panel.classList.toggle("active", index === 0);
    button.addEventListener("click", () => activateMemoryDetailTab(bar, tabs, index));
    bar.appendChild(button);
  });

  wrapper.appendChild(bar);
  for (const tab of tabs) wrapper.appendChild(tab.panel);
  return wrapper;
}

function activateMemoryDetailTab(
  bar: HTMLElement,
  tabs: { label: string; panel: HTMLElement }[],
  active: number,
): void {
  const buttons = Array.from(bar.querySelectorAll(".sub-tab"));
  buttons.forEach((button, index) => button.classList.toggle("active", index === active));
  tabs.forEach((tab, index) => tab.panel.classList.toggle("active", index === active));
}

function createMemoryTabPanel(): HTMLElement {
  const panel = document.createElement("div");
  panel.className = "memory-tab-panel";
  return panel;
}

// The collection's stored entries (and, for collections, the add-entry form).
function createEntriesPanel(memory: MemoryRecord): HTMLElement {
  const panel = createMemoryTabPanel();
  panel.appendChild(createMemoryEntriesSection(memory, memoryEntries, memoryEntriesHasMore));
  return panel;
}

// Collector run history plus a collection's read positions over the logs it
// consumes — both empty for logs, which aren't driven by a collector cycle.
function createActivityPanel(memory: MemoryRecord): HTMLElement {
  const panel = createMemoryTabPanel();
  if (memory.type === "collection" && memoryCursors.length > 0) {
    panel.appendChild(createCursorsSection(memory, memoryCursors));
  }
  if (memoryRuns.length > 0) {
    panel.appendChild(createCollectorRunsSection(memoryRuns, memoryRunsHasMore));
  }
  if (!panel.hasChildNodes()) {
    const empty = document.createElement("div");
    empty.className = "memory-entries-empty";
    empty.textContent = "No collector activity yet.";
    panel.appendChild(empty);
  }
  return panel;
}

// Editable configuration (read-only metadata for system-managed logs).
function createConfigPanel(memory: MemoryRecord): HTMLElement {
  const panel = createMemoryTabPanel();
  panel.appendChild(createMemoryMetadataSection(memory));
  return panel;
}

function createCursorsSection(memory: MemoryRecord, cursors: CursorRecord[]): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-detail-section";

  const title = document.createElement("h3");
  title.textContent = "Read cursors";
  section.appendChild(title);

  const hint = document.createElement("p");
  hint.className = "memory-cursor-hint";
  hint.textContent =
    "Where this collection has read up to in each log. Pick an earlier point to " +
    "re-read from there (set it before your data starts to re-read everything); " +
    "Clear starts fresh from the most recent entries.";
  section.appendChild(hint);

  for (const cursor of cursors) {
    section.appendChild(createCursorRow(memory, cursor));
  }
  return section;
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
  row.className = "memory-cursor-row";

  const log = document.createElement("span");
  log.className = "memory-cursor-log";
  log.textContent = cursor.log_name;

  const current = document.createElement("span");
  current.className = "memory-cursor-current";
  current.textContent = `read up to ${formatDateTime(cursor.last_read_at)}`;

  const picker = document.createElement("input");
  picker.type = "datetime-local";
  picker.className = "memory-form-input";
  picker.value = isoToLocalInput(cursor.last_read_at);

  const setBtn = document.createElement("button");
  setBtn.className = "memory-form-save";
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
  clearBtn.className = "memory-form-archive";
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

// Collector activity is the prompts tab scoped to this collection: each run is
// a full run → prompts → turns card via the shared `createRunRow`.
function createCollectorRunsSection(
  runs: PromptLogRun[],
  hasMore: boolean,
): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-entries-section";

  const title = document.createElement("h3");
  title.textContent = hasMore
    ? `Collector activity (showing ${runs.length}, newest first)`
    : `Collector activity (${runs.length})`;
  section.appendChild(title);

  for (const run of runs) {
    section.appendChild(createRunRow(run));
  }
  if (hasMore) {
    section.appendChild(
      createLoadMoreButton(() => requestMemoryPage("collector_runs", runs.length)),
    );
  }
  return section;
}

function createMemoryHeader(memory: MemoryRecord): HTMLElement {
  const header = document.createElement("div");
  header.className = "memory-detail-header";

  const title = document.createElement("h2");
  title.textContent = memory.name;

  const badge = document.createElement("span");
  badge.className = `memory-type-badge ${memory.type}`;
  badge.textContent = memory.type;

  header.appendChild(title);
  header.appendChild(badge);
  if (memory.archived) {
    const archived = document.createElement("span");
    archived.className = "memory-type-badge";
    archived.textContent = "archived";
    header.appendChild(archived);
  }
  return header;
}

function createMemoryMetadataSection(memory: MemoryRecord): HTMLElement {
  // Logs are system-managed (created by migrations, written by agents) — read-only.
  // Collections are user-editable.
  if (memory.type === "log") {
    return createLogMetadataSection(memory);
  }
  return createCollectionMetadataSection(memory);
}

function createLogMetadataSection(memory: MemoryRecord): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-detail-section";

  const title = document.createElement("h3");
  title.textContent = "Metadata";
  section.appendChild(title);

  const grid = document.createElement("dl");
  grid.className = "memory-detail-grid";
  appendDef(grid, "Description", memory.description || "—");
  appendDef(grid, "Entries", String(memory.entry_count));
  section.appendChild(grid);
  return section;
}

function createCollectionMetadataSection(memory: MemoryRecord): HTMLElement {
  const section = document.createElement("div");
  section.className = "memory-detail-section";

  const title = document.createElement("h3");
  title.textContent = "Metadata";
  section.appendChild(title);

  const fields = createMemoryFormFields({
    description: memory.description,
    published: memory.published,
    extraction_prompt: memory.extraction_prompt ?? "",
  });
  const schedulePicker = createSchedulePicker({
    schedule: memory.schedule,
    description: memory.schedule_description,
  });

  section.appendChild(labelled("Description", fields.description));
  section.appendChild(labelled("Notify on new (published)", fields.published));
  section.appendChild(labelled("Extraction prompt", fields.extractionPrompt));
  section.appendChild(labelled("Schedule", schedulePicker.element));

  const readOnlyGrid = document.createElement("dl");
  readOnlyGrid.className = "memory-detail-grid";
  appendDef(readOnlyGrid, "Entries", String(memory.entry_count));
  if (memory.last_collected_at) {
    appendDef(readOnlyGrid, "Last collected", formatDateTime(memory.last_collected_at));
  }
  section.appendChild(readOnlyGrid);

  const actions = document.createElement("div");
  actions.className = "memory-form-actions";

  // Only collections with an extraction prompt have a collector to run.
  if (memory.extraction_prompt) {
    actions.appendChild(createRunExtractorButton(memory));
  }

  const archive = document.createElement("button");
  archive.className = "memory-form-archive";
  archive.textContent = memory.archived ? "Archived" : "Archive";
  archive.disabled = memory.archived;
  archive.addEventListener("click", () => {
    if (!confirm(`Archive "${memory.name}"? It will disappear from the active list.`)) return;
    browser.runtime.sendMessage({ type: RuntimeMessageType.MemoryArchive, name: memory.name });
    showToast("Archived");
    showMemoriesList();
  });

  const save = document.createElement("button");
  save.className = "memory-form-save";
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
  section.appendChild(actions);
  return section;
}

function createRunExtractorButton(memory: MemoryRecord): HTMLElement {
  const running = triggeringCollection === memory.name;
  const button = document.createElement("button");
  button.className = "memory-form-run";
  button.disabled = running;
  button.innerHTML = running
    ? '<i class="fa-solid fa-spinner fa-spin"></i> Running…'
    : '<i class="fa-solid fa-play"></i> Run extractor';
  button.addEventListener("click", () => {
    triggeringCollection = memory.name;
    browser.runtime.sendMessage({ type: RuntimeMessageType.CollectionTrigger, name: memory.name });
    renderMemoryDetail(); // reflect the running state immediately
  });
  return button;
}

export function handleCollectionTriggerResult(message: RuntimeCollectionTriggerResult): void {
  if (triggeringCollection === message.name) triggeringCollection = null;
  showToast(message.success ? "Extractor finished" : `Extractor failed: ${message.message}`);
  // Refresh the detail so re-enabled button, new entries, and updated
  // "last collected" all reflect the run (also arrives via memory_changed).
  if (activeMemory && activeMemory.name === message.name) renderMemoryDetail();
}

// A memory_create/memory_update whose schedule the server rejected (#2442) —
// the optimistic "Created"/"Saved" toast already fired, so this one follows
// it with the real outcome rather than leaving the rejection unsaid. Nothing
// was persisted, so there is no detail to refresh.
export function handleMemorySaveError(message: RuntimeMemorySaveError): void {
  const verb = message.action === "create" ? "Create" : "Save";
  showToast(`${verb} failed: ${message.message}`);
}

function appendDef(grid: HTMLElement, label: string, value: string, monospace = false): void {
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  if (monospace) {
    const pre = document.createElement("pre");
    pre.textContent = value;
    dd.appendChild(pre);
  } else {
    dd.textContent = value;
  }
  grid.appendChild(dt);
  grid.appendChild(dd);
}

function createMemoryEntriesSection(
  memory: MemoryRecord,
  entries: MemoryEntryRecord[],
  hasMore: boolean,
): HTMLElement {
  // The entries section is "flat" — entries themselves are the cards,
  // so wrapping them in another card creates visual nesting noise.
  const section = document.createElement("div");
  section.className = "memory-entries-section";

  if (memorySearch) section.appendChild(createSearchBanner(memorySearch));

  const title = document.createElement("h3");
  const shown = entries.length;
  const total = memory.entry_count;
  if (memorySearch) {
    title.textContent = `Entries (${shown}${hasMore ? "+" : ""})`;
  } else {
    title.textContent =
      total > shown ? `Entries (showing ${shown} of ${total}, newest first)` : `Entries (${total})`;
  }
  section.appendChild(title);

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "memory-entries-empty";
    empty.textContent = memorySearch ? `No entries match “${memorySearch}”.` : "No entries yet.";
    section.appendChild(empty);
  } else {
    for (const entry of entries) {
      section.appendChild(createMemoryEntry(memory, entry));
    }
  }

  if (hasMore) {
    section.appendChild(createLoadMoreButton(() => requestMemoryPage("entries", entries.length)));
  }

  // Logs are append-only by the system — manual entry add is collection-only.
  if (memory.type === "collection") {
    section.appendChild(createEntryAddForm(memory));
  }
  return section;
}

// Long log entries (esp. ``user-messages`` / ``penny-messages``) get
// collapsed by default so the list stays scannable.  CSS clips after
// ~20 visual lines; the JS heuristic decides whether to clip at all.
const MEMORY_ENTRY_COLLAPSE_LINES = 20;
const MEMORY_ENTRY_COLLAPSE_CHARS = 600;

function createMemoryEntry(memory: MemoryRecord, entry: MemoryEntryRecord): HTMLElement {
  const row = document.createElement("div");
  row.className = "memory-entry";

  const header = document.createElement("div");
  header.className = "memory-entry-header";

  if (entry.key) {
    // Collections: the key is the title.
    const key = document.createElement("span");
    key.className = "memory-entry-key";
    key.textContent = entry.key;
    header.appendChild(key);
  }

  // For logs the timestamp IS the identifier; rendered prominently
  // either way so the eye lands on it quickly when scanning.
  const time = document.createElement("span");
  time.className = entry.key ? "memory-entry-date" : "memory-entry-date memory-entry-date-primary";
  time.textContent = formatDateTime(entry.created_at);
  header.appendChild(time);

  // Logs distinguish authors (user vs penny); a collection's entries are all
  // its collector's, so the author just repeats the collection name — drop it.
  if (memory.type === "log") {
    const author = document.createElement("span");
    author.className = "memory-entry-author";
    author.textContent = entry.author;
    header.appendChild(author);
  }

  // Edit/delete only for collection entries (entry_update / entry_delete are keyed).
  if (memory.type === "collection" && entry.key) {
    header.appendChild(createEntryActions(memory, entry, row));
  }

  const content = document.createElement("div");
  content.className = "memory-entry-content";
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
  return (
    content.split("\n").length > MEMORY_ENTRY_COLLAPSE_LINES ||
    content.length > MEMORY_ENTRY_COLLAPSE_CHARS
  );
}

function createEntryCollapseToggle(content: HTMLElement): HTMLElement {
  const toggle = document.createElement("button");
  toggle.className = "memory-entry-toggle";
  toggle.textContent = "Show more";
  toggle.addEventListener("click", () => {
    const stillCollapsed = content.classList.toggle("collapsed");
    toggle.textContent = stillCollapsed ? "Show more" : "Show less";
  });
  return toggle;
}

function createEntryActions(
  memory: MemoryRecord,
  entry: MemoryEntryRecord,
  row: HTMLElement,
): HTMLElement {
  const actions = document.createElement("span");
  actions.className = "memory-entry-actions";

  const edit = document.createElement("button");
  edit.className = "memory-entry-action";
  edit.title = "Edit";
  edit.innerHTML = '<i class="fa-solid fa-pen"></i>';
  edit.addEventListener("click", (e) => {
    e.stopPropagation();
    enterEditMode(memory, entry, row);
  });

  const del = document.createElement("button");
  del.className = "memory-entry-action";
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
  const content = row.querySelector(".memory-entry-content") as HTMLElement | null;
  if (!content) return;

  const textarea = document.createElement("textarea");
  textarea.className = "memory-form-input memory-form-prompt";
  textarea.value = entry.content;
  textarea.rows = Math.max(3, entry.content.split("\n").length);

  const actions = document.createElement("div");
  actions.className = "memory-form-actions";
  const cancel = document.createElement("button");
  cancel.className = "memory-form-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => content.replaceWith(restored));
  const save = document.createElement("button");
  save.className = "memory-form-save";
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
  wrapper.className = "memory-entry-content";
  wrapper.appendChild(textarea);
  actions.appendChild(cancel);
  actions.appendChild(save);
  wrapper.appendChild(actions);

  content.replaceWith(wrapper);
}

function createEntryAddForm(memory: MemoryRecord): HTMLElement {
  const form = document.createElement("div");
  form.className = "memory-entry-add";

  const keyInput = document.createElement("input");
  keyInput.type = "text";
  keyInput.placeholder = "key";
  keyInput.className = "memory-form-input memory-entry-key-input";

  const contentInput = document.createElement("textarea");
  contentInput.placeholder = "content";
  contentInput.className = "memory-form-input";
  contentInput.rows = 2;

  const submit = document.createElement("button");
  submit.className = "memory-form-save";
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

  form.appendChild(keyInput);
  form.appendChild(contentInput);
  form.appendChild(submit);
  return form;
}

// --- Formatting helpers used only by this tab ---

const MS_PER_DAY = 86_400_000;

function formatRelativeDate(iso: string): string {
  try {
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return iso;
    const days = Math.floor((Date.now() - then) / MS_PER_DAY);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    if (days < 7) return `${days}d ago`;
    if (days < 30) return `${Math.floor(days / 7)}w ago`;
    return formatDateTime(iso);
  } catch {
    return iso;
  }
}

/** Convert literal ``\n`` escape sequences to real newlines.  Used on
 * extraction prompts before rendering them in the textarea — chat-side
 * tool calls (or the model double-escaping) have stored a few prompts
 * with the two-character escape instead of an actual newline. */
function unescapeNewlines(text: string): string {
  return text.replace(/\\n/g, "\n");
}
