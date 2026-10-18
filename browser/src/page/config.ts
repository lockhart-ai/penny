/** Config tab — the tool-use toggle and the runtime-config editor.  Split out
 * of page.ts (#2565); every function here moved unchanged, only `export`/`import`
 * lines were added. */

import {
  type RuntimeConfigParam,
  RuntimeMessageType,
  STORAGE_KEY_TOOL_USE,
  type RuntimeConfigResponse,
} from "../protocol.js";
import { showToast } from "./shared.js";

// --- Config state ---

let pendingConfigSave = false;

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
}

export function renderConfig(params: RuntimeConfigParam[]): void {
  const panel = document.getElementById("config-list")!;
  panel.innerHTML = "";

  const groups = new Map<string, RuntimeConfigParam[]>();
  for (const param of params) {
    if (!groups.has(param.group)) groups.set(param.group, []);
    groups.get(param.group)!.push(param);
  }

  for (const [group, groupParams] of groups) {
    const groupEl = document.createElement("div");
    groupEl.className = "config-group";

    const title = document.createElement("div");
    title.className = "config-group-title";
    title.textContent = group;
    groupEl.appendChild(title);

    for (const param of groupParams) {
      groupEl.appendChild(createConfigItem(param));
    }
    panel.appendChild(groupEl);
  }
}

function createConfigItem(param: RuntimeConfigParam): HTMLElement {
  const item = document.createElement("div");
  item.className = "config-item";

  const header = document.createElement("div");
  header.className = "config-header";

  const label = document.createElement("label");
  label.className = "config-label";
  label.textContent = param.description;
  label.htmlFor = `config-${param.key}`;

  const key = document.createElement("span");
  key.className = "config-key";
  key.textContent = param.key;

  const defaultVal = document.createElement("span");
  defaultVal.className = "config-default";
  defaultVal.textContent = `default: ${param.default}`;

  header.appendChild(label);
  header.appendChild(key);
  header.appendChild(defaultVal);

  const input = document.createElement("input");
  input.id = `config-${param.key}`;
  input.className = "config-input";
  input.type = param.type === "str" ? "text" : "number";
  if (param.type === "int") input.step = "1";
  if (param.type === "float") input.step = "any";
  input.value = param.value;
  input.placeholder = param.default;
  if (param.value !== param.default) input.classList.add("modified");

  input.addEventListener("change", () => {
    pendingConfigSave = true;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.ConfigUpdate,
      key: param.key,
      value: input.value,
    });
  });

  item.appendChild(header);
  item.appendChild(input);
  return item;
}

// --- Message-handler entry point (page.ts dispatches here) ---

export function applyConfigResponse(message: RuntimeConfigResponse): void {
  renderConfig(message.params);
  if (pendingConfigSave) {
    pendingConfigSave = false;
    showToast("Saved");
  }
}
