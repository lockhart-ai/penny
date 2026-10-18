/** Domains tab — the allow/block list.  Split out of page.ts (#2565);
 * restyled to the approved mockup (#2569): an All/Allowed/Blocked filter
 * with counts, a search box, and a per-row Allow/Block segmented toggle
 * instead of the old click-to-toggle status pill. */

import {
  type DomainAllowlist,
  DomainPermission as DP,
  type DomainPermissionEntry,
  RuntimeMessageType,
  STORAGE_KEY_DOMAIN_ALLOWLIST,
} from "../protocol.js";
import { showToast, wireSearchClear } from "./shared.js";
import { type DomainFilter, domainCounts, filterDomains } from "./tab_filters.js";

// --- Domains state ---

let allPermissions: DomainPermissionEntry[] = [];
let activeFilter: DomainFilter = "all";
let search = "";
let addPermission: DomainPermissionEntry["permission"] = DP.Allowed;

// ============================================================
// Domains
// ============================================================

export async function loadDomainsFromCache(): Promise<void> {
  const stored = await browser.storage.local.get(STORAGE_KEY_DOMAIN_ALLOWLIST);
  const allowlist: DomainAllowlist = (stored[STORAGE_KEY_DOMAIN_ALLOWLIST] as DomainAllowlist) ?? {};
  renderDomains(Object.entries(allowlist).map(([domain, permission]) => ({ domain, permission })));
}

export function renderDomains(permissions: DomainPermissionEntry[]): void {
  allPermissions = permissions;
  renderCounts();
  renderRows();
}

function renderCounts(): void {
  const counts = domainCounts(allPermissions);
  setText("domains-count-all", String(counts.all));
  setText("domains-count-allowed", String(counts.allowed));
  setText("domains-count-blocked", String(counts.blocked));
}

function renderRows(): void {
  const listEl = document.getElementById("domains-list")!;
  listEl.innerHTML = "";

  const sorted = [...filterDomains(allPermissions, activeFilter, search)].sort((a, b) =>
    a.domain.localeCompare(b.domain),
  );

  if (allPermissions.length === 0) {
    appendEmpty(listEl, "No domains saved yet.");
    return;
  }
  if (sorted.length === 0) {
    appendEmpty(listEl, "No domains match.");
    return;
  }

  for (const { domain, permission } of sorted) {
    listEl.appendChild(createDomainRow(domain, permission));
  }
}

function appendEmpty(listEl: HTMLElement, text: string): void {
  const empty = document.createElement("div");
  empty.className = "domains-empty";
  empty.textContent = text;
  listEl.appendChild(empty);
}

function createDomainRow(domain: string, permission: DomainPermissionEntry["permission"]): HTMLElement {
  const row = document.createElement("div");
  row.className = "domain-row";

  const left = document.createElement("span");
  left.className = "domain-left";
  left.innerHTML = '<svg class="icon"><use href="#i-globe"/></svg>';
  const name = document.createElement("span");
  name.className = "domain-name";
  name.textContent = domain;
  name.title = domain;
  left.appendChild(name);

  const right = document.createElement("span");
  right.className = "domain-right";
  right.appendChild(createPermToggle(domain, permission));
  right.appendChild(createRemoveButton(domain));
  row.appendChild(left);
  row.appendChild(right);
  return row;
}

function createPermToggle(domain: string, permission: DomainPermissionEntry["permission"]): HTMLElement {
  const toggle = document.createElement("span");
  toggle.className = "perm-toggle";
  for (const [value, label, cls] of [
    [DP.Allowed, "Allow", "allow"],
    [DP.Blocked, "Block", "block"],
  ] as const) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = cls;
    btn.classList.toggle("on", permission === value);
    btn.addEventListener("click", () => {
      if (permission === value) return;
      browser.runtime.sendMessage({ type: RuntimeMessageType.DomainUpdate, domain, permission: value });
    });
    toggle.appendChild(btn);
  }
  return toggle;
}

function createRemoveButton(domain: string): HTMLElement {
  const del = document.createElement("button");
  del.className = "icon-btn";
  del.innerHTML = '<svg class="icon"><use href="#i-x"/></svg>';
  del.title = `Remove ${domain}`;
  del.setAttribute("aria-label", `Remove ${domain}`);
  del.addEventListener("click", () => {
    browser.runtime.sendMessage({ type: RuntimeMessageType.DomainDelete, domain });
  });
  return del;
}

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function setupDomains(): void {
  const input = document.getElementById("domains-input") as HTMLInputElement;
  const addBtn = document.getElementById("domains-add-btn")!;

  function add(): void {
    const raw = input.value.trim().toLowerCase();
    if (!raw) return;
    const domain = raw.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
    if (!domain) return;
    browser.runtime.sendMessage({
      type: RuntimeMessageType.DomainUpdate,
      domain,
      permission: addPermission,
    });
    input.value = "";
    const label = addPermission === DP.Allowed ? "Allowed" : "Blocked";
    showToast(`${label}: ${domain}`);
  }

  addBtn.addEventListener("click", add);
  input.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter") add();
  });

  // The add row's Allow/Block segmented control replaces the old <select>.
  for (const btn of Array.from(document.querySelectorAll("#domains-permission button"))) {
    btn.addEventListener("click", () => {
      addPermission = btn.getAttribute("data-permission") as DomainPermissionEntry["permission"];
      for (const other of Array.from(document.querySelectorAll("#domains-permission button"))) {
        other.classList.toggle("active", other === btn);
      }
    });
  }

  // The filter bar: All / Allowed / Blocked with live counts.
  for (const btn of Array.from(document.querySelectorAll("#domains-filter button"))) {
    btn.addEventListener("click", () => {
      activeFilter = btn.getAttribute("data-filter") as DomainFilter;
      for (const other of Array.from(document.querySelectorAll("#domains-filter button"))) {
        other.classList.toggle("active", other === btn);
      }
      renderRows();
    });
  }

  const searchInput = document.getElementById("domains-search") as HTMLInputElement;
  searchInput.addEventListener("input", () => {
    search = searchInput.value.trim();
    renderRows();
  });
  wireSearchClear("domains-search", "domains-search-clear");
}