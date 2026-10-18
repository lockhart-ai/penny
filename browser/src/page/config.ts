/** Config tab — the tool-use toggle and the runtime-config editor.  Split out
 * of page.ts (#2565); restyled to the approved mockup (#2569): group cards
 * with one param per row and a fixed right-hand editor column (number, text,
 * toggle, or — for a param with a closed value set — a segmented control), a
 * reset-to-default on changed values, an All/Modified filter, and a search
 * box. */

import {
  type RuntimeConfigParam,
  RuntimeMessageType,
  STORAGE_KEY_TOOL_USE,
  type RuntimeConfigResponse,
} from "../protocol.js";
import { showToast, wireSearchClear } from "./shared.js";
import {
  type ConfigFilter,
  choiceLabel,
  configGroupLabel,
  filterConfigParams,
  isModified,
  isSwitchChoices,
  modifiedCount,
} from "./tab_filters.js";

// --- Config state ---

let pendingConfigSave = false;
let allParams: RuntimeConfigParam[] = [];
let activeFilter: ConfigFilter = "all";
let search = "";

// ============================================================
// Config
// ============================================================

export async function loadToolUseState(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_TOOL_USE);
  const enabled = (stored[STORAGE_KEY_TOOL_USE] as boolean) ?? false;
  const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement | null;
  if (toggle) toggle.checked = enabled;
}

export function setupConfig(): void {
  const toggle = document.getElementById("tool-use-toggle") as HTMLInputElement;
  toggle.addEventListener("change", () => {
    browser.runtime.sendMessage({ type: RuntimeMessageType.ToolUseToggle, enabled: toggle.checked });
  });

  for (const btn of Array.from(document.querySelectorAll("#config-filter button"))) {
    btn.addEventListener("click", () => {
      activeFilter = btn.getAttribute("data-filter") as ConfigFilter;
      for (const other of Array.from(document.querySelectorAll("#config-filter button"))) {
        other.classList.toggle("active", other === btn);
      }
      renderGroups();
    });
  }

  const searchInput = document.getElementById("config-search") as HTMLInputElement;
  searchInput.addEventListener("input", () => {
    search = searchInput.value.trim();
    renderGroups();
  });
  wireSearchClear("config-search", "config-search-clear");
}

export function renderConfig(params: RuntimeConfigParam[]): void {
  allParams = params;
  const count = document.getElementById("config-count-modified");
  if (count) count.textContent = String(modifiedCount(params));
  renderGroups();
}

function renderGroups(): void {
  const panel = document.getElementById("config-list")!;
  panel.innerHTML = "";

  const visible = filterConfigParams(allParams, activeFilter, search);
  if (allParams.length === 0) {
    appendEmpty(panel, "Loading config…");
    return;
  }
  if (visible.length === 0) {
    appendEmpty(panel, "No params match.");
    return;
  }

  const groups = new Map<string, RuntimeConfigParam[]>();
  for (const param of visible) {
    if (!groups.has(param.group)) groups.set(param.group, []);
    groups.get(param.group)!.push(param);
  }

  for (const [group, groupParams] of groups) {
    panel.appendChild(createGroupCard(group, groupParams));
  }
}

function appendEmpty(panel: HTMLElement, text: string): void {
  const empty = document.createElement("div");
  empty.className = "config-empty";
  empty.textContent = text;
  panel.appendChild(empty);
}

function createGroupCard(group: string, groupParams: RuntimeConfigParam[]): HTMLElement {
  const card = document.createElement("div");
  card.className = "config-group";

  const head = document.createElement("div");
  head.className = "group-head";
  const title = document.createElement("span");
  title.className = "group-title";
  title.textContent = configGroupLabel(group);
  const count = document.createElement("span");
  count.className = "group-count";
  count.textContent = `${groupParams.length} params`;
  head.appendChild(title);
  head.appendChild(count);
  card.appendChild(head);

  for (const param of groupParams) {
    card.appendChild(createParamRow(param));
  }
  return card;
}

function createParamRow(param: RuntimeConfigParam): HTMLElement {
  const row = document.createElement("div");
  row.className = "param-row";
  row.classList.toggle("modified", isModified(param));

  const left = document.createElement("div");
  left.className = "param-left";
  const key = document.createElement("span");
  key.className = "param-key";
  key.textContent = param.key;
  key.title = param.key;
  const desc = document.createElement("span");
  desc.className = "param-desc";
  desc.textContent = param.description;
  desc.title = param.description;
  left.appendChild(key);
  left.appendChild(desc);

  const col = document.createElement("div");
  col.className = "field-col";
  const def = document.createElement("span");
  def.className = "field-default";
  def.textContent = `default: ${param.default}`;
  const fieldRow = document.createElement("div");
  fieldRow.className = "field-row";
  fieldRow.appendChild(createEditor(param));
  fieldRow.appendChild(createResetSlot(param));
  col.appendChild(def);
  col.appendChild(fieldRow);

  row.appendChild(left);
  row.appendChild(col);
  return row;
}

/** The editor for one param — chosen from the param's own shape (type plus
 * closed value set), never from its name: a future two-valued param gets
 * the segmented control with no code change here. */
function createEditor(param: RuntimeConfigParam): HTMLElement {
  if (isSwitchChoices(param.choices)) return createToggleEditor(param);
  if (param.choices && param.choices.length > 0) return createSegEditor(param);
  return createInputEditor(param);
}

function createToggleEditor(param: RuntimeConfigParam): HTMLElement {
  const on = param.value === "1";
  const toggle = document.createElement("div");
  toggle.className = "toggle";
  toggle.classList.toggle("on", on);
  toggle.setAttribute("role", "switch");
  toggle.setAttribute("aria-checked", String(on));
  toggle.setAttribute("aria-label", param.key);
  toggle.innerHTML = '<span class="knob"></span>';
  toggle.addEventListener("click", () => {
    sendConfigUpdate(param, param.value === "1" ? "0" : "1");
  });
  return toggle;
}

function createSegEditor(param: RuntimeConfigParam): HTMLElement {
  const seg = document.createElement("div");
  seg.className = "field-seg";
  for (const choice of param.choices ?? []) {
    const btn = document.createElement("button");
    btn.textContent = choiceLabel(choice);
    btn.classList.toggle("active", param.value === choice);
    btn.addEventListener("click", () => {
      if (param.value === choice) return;
      sendConfigUpdate(param, choice);
    });
    seg.appendChild(btn);
  }
  return seg;
}

function createInputEditor(param: RuntimeConfigParam): HTMLElement {
  const input = document.createElement("input");
  input.id = `config-${param.key}`;
  input.className = "field-input";
  if (param.type === "str") {
    input.classList.add("wide");
  } else {
    input.type = "number";
    input.step = param.type === "int" ? "1" : "any";
  }
  input.value = param.value;
  input.title = param.value;
  input.addEventListener("change", () => {
    input.title = input.value;
    sendConfigUpdate(param, input.value);
  });
  return input;
}

function createResetSlot(param: RuntimeConfigParam): HTMLElement {
  if (!isModified(param)) {
    const slot = document.createElement("span");
    slot.className = "reset-slot";
    return slot;
  }
  const btn = document.createElement("button");
  btn.className = "reset-btn";
  btn.innerHTML = '<svg class="icon"><use href="#i-undo"/></svg>';
  btn.title = `Reset to default (${param.default})`;
  btn.setAttribute("aria-label", `Reset ${param.key} to default (${param.default})`);
  btn.addEventListener("click", () => {
    sendConfigUpdate(param, param.default);
  });
  return btn;
}

function sendConfigUpdate(param: RuntimeConfigParam, value: string): void {
  pendingConfigSave = true;
  browser.runtime.sendMessage({ type: RuntimeMessageType.ConfigUpdate, key: param.key, value });
}

// --- Message-handler entry point (page.ts dispatches here) ---

export function applyConfigResponse(message: RuntimeConfigResponse): void {
  renderConfig(message.params);
  if (pendingConfigSave) {
    pendingConfigSave = false;
    showToast("Saved");
  }
}