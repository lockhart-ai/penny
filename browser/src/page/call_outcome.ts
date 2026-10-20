/**
 * A short, visible label for a prompt call's outcome (#2436) — derived
 * generically from the value Penny's runtime stamped, with no hand-kept
 * list of outcome names: an outcome this code has never heard of still
 * reads as something (`"some_new_outcome"` → `"some new outcome"`), rather
 * than requiring a matching code change before it renders at all.
 *
 * `null` (a row logged before the `call_outcome` column existed) and
 * `"completed"` both mean the call finished normally, so neither gets a
 * label — only a call that did NOT complete is worth calling out on the
 * row.
 */
export function callOutcomeLabel(outcome: string | null): string | null {
  if (outcome === null || outcome === "completed") return null;
  return outcome.replace(/_/g, " ");
}

/**
 * A short, visible label for a completed call whose output was NOT used
 * (#2702) — derived generically from the machine reason the runtime stamped
 * on the row, with no hand-kept list: a reason this code has never heard of
 * still reads as something ("degenerate output" ← "degenerate_output"),
 * rather than requiring a matching code change before it renders at all.
 *
 * `null` (a used call, or a row logged before the column existed) gets no
 * label — only a call whose output was thrown away is worth calling out.
 */
export function discardLabel(reason: string | null): string | null {
  if (reason === null) return null;
  return `discarded · ${reason.replace(/_/g, " ")}`;
}
