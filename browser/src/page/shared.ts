/** Shared page helpers — small, cross-tab, and pure (or self-owned, like the
 * toast).  Split out of page.ts (#2565); everything here moved unchanged. */


// --- Toast ---

let toastTimer: ReturnType<typeof setTimeout> | null = null;

export function showToast(text: string): void {
  const toast = document.getElementById("toast")!;
  toast.textContent = text;
  toast.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2000);
}

// Firefox doesn't render a native clear (✕) for ``type="search"``, so wire a
// custom one: it shows only when there's text, and clearing dispatches an
// ``input`` event so the existing search handler re-runs (unfiltered).
export function wireSearchClear(inputId: string, buttonId: string): void {
  const input = document.getElementById(inputId) as HTMLInputElement | null;
  const button = document.getElementById(buttonId);
  if (!input || !button) return;
  const sync = (): void => {
    button.classList.toggle("visible", input.value.length > 0);
  };
  input.addEventListener("input", sync);
  button.addEventListener("click", () => {
    input.value = "";
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
  sync();
}

// A consistent "Showing matches for X" line above any search-filtered list,
// so it's always obvious the view is filtered rather than complete.
export function createSearchBanner(query: string): HTMLElement {
  const banner = document.createElement("div");
  banner.className = "search-banner";
  banner.textContent = `Showing matches for “${query}”`;
  return banner;
}

export function formatDateTime(iso: string): string {
  try {
    const date = new Date(iso);
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

