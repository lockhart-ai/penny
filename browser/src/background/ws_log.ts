/**
 * Browse-flow step tracing for the background process.
 *
 * The WebSocket lives only in the background script (see background.ts), and a
 * browse appears to tear that process — and its socket — down: the server sees
 * a 1006 close ~3s after each browse begins.  To pin the exact operation that
 * kills it, every step of the browse flow is stamped with the live WebSocket
 * readyState via {@link logStep}; the last line before the socket dies is the
 * culprit.  Output lands in the background page's devtools console
 * (about:debugging → this extension → Inspect).  Flip TRACE to false to silence.
 */

const TRACE = false;

type WsStateProvider = () => number | undefined;

// background.ts owns `ws` and registers a getter so this module can read the
// live readyState without importing background.ts (which would be circular).
let wsStateProvider: WsStateProvider = () => undefined;

export function setWsStateProvider(provider: WsStateProvider): void {
  wsStateProvider = provider;
}

const READY_STATE_NAMES: Record<number, string> = {
  0: "CONNECTING",
  1: "OPEN",
  2: "CLOSING",
  3: "CLOSED",
};

export function wsStateName(): string {
  const state = wsStateProvider();
  if (state === undefined) return "NULL";
  return READY_STATE_NAMES[state] ?? `UNKNOWN(${state})`;
}

export function logStep(area: string, label: string): void {
  if (!TRACE) return;
  console.log(`[trace:${area}] ${label} | ws=${wsStateName()}`);
}
