# Self-narrating tools — the narration seam

Design and evidence for epic #1478 (self-narrating tools). This doc is the contract
the rest of the epic (#1480–#1485) builds on. Issue #1479 lands the foundation: the
narration mechanism plus the single tagged first-person result-framing choke point.

## The problem this solves

Penny searches, reads, and saves silently, then states a bare answer — nothing about a
tool call surfaces in her reply. The epic makes each tool call *speak*: every tool
result leads with a first-person, machine-tagged narration of the action Penny took,
and (in a later ticket) the chat agent opens its reply with a natural recap of those
actions. This is the penny→user half of the natural-language ↔ tool mapping (#1346) and
the substrate the #1471 skills rework reads (demonstrations narrated in the terms the
distiller consumes).

## The seam (this ticket)

Two pieces, both in the tool layer, so every agent that reads tool results inherits the
legibility win — chat *and* collectors (a free adjacency to #1332).

1. **`Tool.to_result_narration(cls, arguments, result) -> str`** — a class method,
   registry-dispatched exactly like the existing `Tool.to_action_str` (the *pre-call*
   status string) is via `Tool.format_status`. It is the symmetric *result* twin, so it
   also takes the `ToolResult` and branches on `result.success`. Generic default here;
   per-tool overrides land in #1480–#1482.
   - success → `You used \`<tool>\` and here's the result:`
   - failure (`result.success is False`) → `You tried to use \`<tool>\` but it didn't work:`

2. **`Tool.format_result(cls, tool_name, arguments, result)`** — the single framing choke
   point, applied once in `Agent._collect_tool_results`. It composes:

   ```
   {narration} ({tool} result)
   {body}
   ```

   i.e. the first-person narration + a retained `(<tool> result)` machine tag, then the
   body (`result.message`). This **replaces** the terse `Result of your \`<tool>\` call:`
   header. The call `arguments` and the whole `ToolResult` are threaded from
   `_execute_single_tool` → `_collect_tool_results` (the site previously received only
   `tool_name` + a body string; `_execute_single_tool` now returns the `ToolResult`).

The tag is composed by `format_result`, not baked into each narration — so the
machine-tag invariant holds uniformly even for a per-tool override that returns an
arbitrary sentence.

## Evidence (a live-model probe against gpt-oss:20b, not inference)

These are the settled findings that constrain the design. They orient the rest of the epic.

- **(a) The recap *prompt* is the load-bearing lever.** Narrated results *alone* never
  produced a recap in Penny's reply; the recap instruction (a separate ticket, #1483)
  always did. The two levers are separable, and the instruction is what does the work —
  so this ticket ships the narration mechanism, not a behaviour change to Penny's replies.

- **(b) Narration must be *tagged*, not pure prose.** Pure first-person prose with **no
  machine tag** RAISED the call-as-text bail rate on the loop-stressed path (5/6 vs. 3/6
  with a tag, pre-fix). The terse header existed precisely to stop gpt-oss reading a prose
  tool body as a fresh instruction (#1332's #1 failure class, "envelope confusion"); the
  OpenAI `role: "tool"` + `tool_call_id` envelope is not honoured reliably by the local
  model when the body reads like prose. A retained `(<tool> result)` tag preserves that
  disambiguation while the header now reads naturally. **This is why the ticket says
  "tagged."** Any per-tool override (#1480–#1482) inherits the tag for free because
  `format_result` — not the override — adds it.

- **(c) Narration is envelope-safe and leak-free on the success path** — 0 leaks, 0 task
  regressions across 15 samples; recaps (with the #1483 instruction) read naturally.

- **(d) Honesty is robust** — 0 hallucinated facts on fruitless searches; the failure
  narration (`didn't work`) is honest, coherent with the #1414 house failure template
  (which keeps the actionable "how to fix" tail in the body).

- **(e) Total-failure replies bypass the model.** When every tool fails, the reply is a
  deterministic `AGENT_TOOLS_UNAVAILABLE` (`_abort_if_all_tools_failed`), so the recap only
  governs success + partial-failure paths where the model composes the reply. Warming that
  canned string is separate/optional.

## Non-regression (this ticket's eval obligation)

`format_result` is a "what the model reads" change, so per house rules it ships with a
**non-regression** eval against the existing cases — the tagged framing must not raise the
call-as-text / envelope-confusion signal vs. the terse header. The #1477 chat-surface
call-as-text guard (`test_chat_call_recovery.py`) must stay at its recovered ceiling, and
`test_chat_response.py` must still hold. Deterministic coverage (`make check`) pins the
framed-string shape for a success and a failure `ToolResult`.

## What this ticket deliberately does NOT do (later in the epic)

- Per-tool narration overrides: browse (#1480), memory/lifecycle (#1481),
  framework-failure (#1482). This ticket ships only the generic default.
- The chat recap instruction (#1483) — the core model-facing lever (finding **a**).
- Honest failure recap (#1484, after #1483).
- The other tool-shaped injection sites — page-context, dedup rejection, rejected calls
  (#1485). The signature change forces those sites to construct a `ToolResult` so the code
  compiles, and the *generic default narration flows through them* in the meantime (a
  page-context browse narrates as a success; a duplicate/rejected call as a failure) —
  that is expected and fine; #1485 gives them bespoke, context-specific narration.
