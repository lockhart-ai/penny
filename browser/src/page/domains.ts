/** Domains tab — the allow/block list.  Split out of page.ts (#2565);
 * every function here moved unchanged, only `export`/`import` lines were added. */

import {
  type DomainAllowlist,
  DomainPermission as DP,
  type DomainPermissionEntry,
  RuntimeMessageType,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
} from "../protocol.js";
import { showToast } from "./shared.js";

// ============================================================
// Domains
// ============================================================

export async function loadDomainsFromCache(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  const permissions = Object.entries(allowlist).map(([domain, permission]) => ({ domain, permission }));
  renderDomains(permissions);
}

export function renderDomains(permissions: DomainPermissionEntry[]): void {
  const listEl = document.getElementById("domains-list")!;
  listEl.innerHTML = "";

  const sorted = [...permissions].sort((a, b) => a.domain.localeCompare(b.domain));

  if (sorted.length === 0) {
    const empty = document.createElement("div");
    empty.className = "prefs-empty";
    empty.textContent = "No domains saved yet.";
    listEl.appendChild(empty);
    return;
  }

  for (const { domain, permission } of sorted) {
    const row = document.createElement("div");
    row.className = "domain-row";

    const name = document.createElement("span");
    name.className = "domain-name";
    name.textContent = domain;

    const status = document.createElement("button");
    status.className = `domain-status ${permission}`;
    status.textContent = permission === DP.Allowed ? "Allowed" : "Blocked";
    status.title = "Click to toggle";
    status.addEventListener("click", () => {
      const next = permission === DP.Allowed ? DP.Blocked : DP.Allowed;
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainUpdate, domain, permission: next });
    });

    const del = document.createElement("button");
    del.className = "pref-delete";
    del.innerHTML = '<i class="fa-solid fa-xmark"></i>';
    del.setAttribute("aria-label", `Remove ${domain}`);
    del.addEventListener("click", () => {
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainDelete, domain });
    });

    row.appendChild(name);
    row.appendChild(status);
    row.appendChild(del);
    listEl.appendChild(row);
  }
}

export function setupDomains(): void {
  const input = document.getElementById("domains-input") as HTMLInputElement;
  const select = document.getElementById("domains-permission") as HTMLSelectElement;
  const btn = document.getElementById("domains-add-btn")!;

  function add(): void {
    const raw = input.value.trim().toLowerCase();
    if (!raw) return;
    const domain = raw.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
    if (!domain) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.DomainUpdate,
      domain,
      permission: select.value,
    });
    input.value = "";
    const label = select.value === "allowed" ? "Allowed" : "Blocked";
    showToast(`${label}: ${domain}`);
  }

  btn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });
}
