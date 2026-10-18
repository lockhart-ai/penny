/**
 * Pure display helpers for the Domains and Config tabs' filter bars and
 * editors (#2569) — data-in/data-out, no DOM, so node:test can pin them.
 */

import type { DomainPermissionEntry, RuntimeConfigParam } from "../protocol.js";

// Type-only imports above — this module must stay import-free at runtime,
// because build:test's esbuild pass (no --bundle) inlines it as-is.
// DomainPermission's runtime values live in protocol.ts; here the literal
// union off the entry type is the same set.
const ALLOWED = "allowed" satisfies DomainPermissionEntry["permission"];

// ============================================================
// Domains — the All / Allowed / Blocked filter plus search
// ============================================================

export type DomainFilter = "all" | "allowed" | "blocked";

export function filterDomains(
  permissions: DomainPermissionEntry[],
  filter: DomainFilter,
  query: string,
): DomainPermissionEntry[] {
  const q = query.toLowerCase();
  return permissions.filter(({ domain, permission }) => {
    const matchesFilter = filter === "all" || (filter === "allowed") === (permission === ALLOWED);
    return matchesFilter && domain.toLowerCase().includes(q);
  });
}

export function domainCounts(permissions: DomainPermissionEntry[]): {
  all: number;
  allowed: number;
  blocked: number;
} {
  const allowed = permissions.filter((p) => p.permission === ALLOWED).length;
  return { all: permissions.length, allowed, blocked: permissions.length - allowed };
}

// ============================================================
// Config — the All params / Modified filter, search, group label,
// and the editor-choice helpers (keyed to the value set, never to
// a param name: a future two-valued param gets the right editor
// with no code change here)
// ============================================================

export type ConfigFilter = "all" | "modified";

export function isModified(param: RuntimeConfigParam): boolean {
  return param.value !== param.default;
}

export function filterConfigParams(
  params: RuntimeConfigParam[],
  filter: ConfigFilter,
  query: string,
): RuntimeConfigParam[] {
  const q = query.toLowerCase();
  return params.filter((p) => {
    const matchesFilter = filter === "all" || isModified(p);
    return matchesFilter && (p.key.toLowerCase().includes(q) || p.description.toLowerCase().includes(q));
  });
}

export function modifiedCount(params: RuntimeConfigParam[]): number {
  return params.filter(isModified).length;
}

/** The dedup group renders as "Collector" to match the Collectors tab
 * rename; parameter keys are unchanged — a display mapping only (#2569). */
const GROUP_DISPLAY_LABELS: Record<string, string> = { Memory: "Collector" };

export function configGroupLabel(group: string): string {
  return GROUP_DISPLAY_LABELS[group] ?? group;
}

/** The 0/1 switch pair renders as a toggle rather than a segmented control. */
export function isSwitchChoices(choices: string[] | undefined): boolean {
  return choices !== undefined && choices.length === 2 && choices[0] === "0" && choices[1] === "1";
}

/** A choice value becomes its control label: `allow_all` → `Allow all`. */
export function choiceLabel(choice: string): string {
  const spaced = choice.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}