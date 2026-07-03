# Prompt-Writing Guide for Penny

How to write the model-facing prompts that drive Penny — collector `extraction_prompt`s,
agent system prompts, and tool descriptions — against a local gpt-oss:20b. Distilled from
(1) patterns in heavily-tuned production system prompts (e.g. Claude Code's), and (2) what
our own live-model evals (`make eval`) have repeatedly shown. Every claim here that cites a
number was measured on this codebase.

## The four that matter most

If you remember nothing else:

1. **Numbered lists for instructions** — one action per step.
2. **Explicit tool-call examples** — `N. tool(args) — description`, every call named the same way.
3. **Positive framing for the positive case** — state what TO do, in order.
4. **Emphatic `NEVER` / `ALWAYS` / `IMPORTANT` for specific negative cases** — reserved for the
   concrete pitfalls, not general nagging.

The split in 3 + 4 is the whole reconciliation: **positive spine for the happy path, emphatic
guards for the specific things the model gets wrong.** The rest of this doc is detail and
evidence behind these.

## Structure the task as read → plan → execute

The reliable spine for any agent that reads messy input and acts on it:

- **Read** — gather all the inputs first (the log reads, the current state).
- **Plan** — form the complete picture before touching anything: cohere the inputs into a
  clear intermediate ("what did the user intend; what tool sequence actually ran") *before*
  deciding the action. Don't let the model read, decide, and act in one tangled step — it
  thrashes.
- **Execute** — the action, then the report, then the terminal `done`.

Two corollaries, both measured on this codebase:

- **As few decision branches as possible.** A 4-way classify (NEW / STALE / MISSING / CURRENT)
  made gpt-oss *drown* — it spent 10–12 calls re-analyzing the rules ("Wait, STALE applies
  when… or just a subset?… param differences?") and then either ran out of step budget or
  emitted a `done()` claiming work it never did. Collapsing it to a linear "conform the thing
  to reality" (find match → make it match; else create) removed the thrashing. Prefer one
  imperative path over N buckets the model has to sort into.
- **CRITICAL for tool-call-only agents: planning belongs in the *reasoning*, never as its own
  numbered step.** If you write "2. Summarize the user's intent" as a numbered step, gpt-oss
  tries to *perform* it — by emitting the summary as plain text. In an agent that acts only
  through tool calls, that text fails to parse as a tool call, the loop nudges it, and it
  **spirals** into apologies and confabulated actions (measured: this was *the* dominant
  failure — whole cycles derailed). Fix: the numbered spine is **all tool calls**; the "plan"
  rides inside a tool step's guidance ("1. read_run_calls() — … *in your reasoning*, note the
  intent and the observed sequence"). Add an explicit "act only through tool calls, never
  reply with plain text." *Result: 0 spirals, and the isolated case went to 5/5.*

## The core patterns

1. **Numbered, sequential tool-call steps.** gpt-oss follows a numbered recipe far more
   reliably than prose (measured: prose bails ~60% of the time, numbered ~5%). Structure a
   task as `1. … 2. … 3. …`, one action per step.

2. **Name every tool call explicitly, in ONE consistent format:** `N. tool(args) — description`.
   e.g. `2. collection_read_latest("skills") — every existing skill, with its key, STEPS, and TRIGGER.`
   Do **not** let some steps name their call crisply while an important one is demoted to a
   vague parenthetical ("…, then send a message about it"). *Measured:* a `send_message`
   buried as a parenthetical was skipped **~40%** of cycles; rewritten as a first-class step
   `send_message("I changed the '<skill>' skill: <what>") — tell the user what you changed`
   it was skipped **0%**. If a call matters, show it, formatted identically to the others —
   inconsistent formatting reads as "this one is optional."

3. **Emphatic markers are legitimate — use them deliberately for critical rules.**
   `IMPORTANT:`, `ALWAYS`, `NEVER`, ALL-CAPS on the load-bearing word. Production prompts use
   these liberally and they work. Reserve them for the few rules that actually carry the
   behaviour; if everything is capitalised, nothing is.

4. **Keep load-bearing discriminating rules SPECIFIC and CONCRETE — with a worked example.**
   The rule that makes or breaks a hard judgement must state the exact distinction and show
   it. *Measured:* the skills collector's "two different tool names are different steps even
   when the params are identical — `collection_read_latest(memory='x')` is not
   `log_read(memory='x')`" rule was worth **5/5 vs 0/5** on the tweak-STEPS case. Softening
   it to "match the tools" collapsed it. A vague rule is a dropped rule.

5. **Positive spine, negative guards.** Default to stating what TO do, in order. Add
   `NEVER …` / `DON'T …` guards for the *specific* pitfalls you've seen the model hit. Both
   are fine; the negative guard is for a concrete failure mode, not general nagging.

6. **Declare what's ground truth.** When the model must judge against evidence, tell it what's
   authoritative: "the tool calls you OBSERVED are ground truth — they are what actually ran."
   Judgement anchored to a stated source beats judgement left to vibes.

7. **Examples beat abstraction for a tricky case.** One concrete good/bad pair teaches a
   discrimination that a paragraph of rules won't.

## The anti-pattern: accretion

Do **not** fix a prompt by iteratively bolting caveats onto the existing text — "and you MUST
X", "don't forget Y", "(remember Z)" — until it's a pile of nagging. Each patch fights the
last and the actual instruction drowns. When a rule needs adding, decide where it belongs and
write it there; if the prompt has drifted, **rewrite it whole** — cleanly structured, with the
emphatic load-bearing rules included deliberately. *Clean ≠ vague:* rewriting must **keep**
every specific discriminating rule (see #4). The failure mode to avoid is the mess of
accretion, not emphasis itself.

## gpt-oss:20b specifics (things it gets wrong)

- **Skips a terminal/secondary step** (the send after the write; the `done()`) — make each its
  own explicit numbered step; a step it can fold into another, it will.
- **Copies display formatting into arguments** — the entry listing renders keys as `[key]`, and
  the model passes `key="[key]"` verbatim → "not found". Tell it: the key is the text *inside*
  the brackets, pass it without them. (Or make the tool's error actionable.)
- **Protocol spirals** — on ambiguity it can loop about "can I make multiple calls in one
  reply?" and burn the cycle. A tight numbered sequence reduces this.
- **Punctuation-collapse** on large contexts — a separate degeneracy guard handles this; keep
  prompts lean.

## When a guard doesn't work, aim it at the *exact* pitfall — and re-check it didn't just move

Adding an emphatic guard is easy to get subtly wrong: it fixes the failure you saw and *creates*
a neighbouring one. Measured: to stop the model duplicating a skill, an "always match an existing
skill by its TRIGGER" guard worked — but it made the model conclude "the TRIGGER already covers
this → nothing to do" and **skip the STEPS fix** it was supposed to make. One failure traded for
another. The guard has to point at the *specific* thing the traces show the model doing wrong
(here: skipping the STEPS rewrite), not at a plausible-sounding neighbour. After any guard, re-run
and check the failure *moved*, not just changed shape.

## Prefer simple tool calls; complex ones spiral

The more a call forces the model to construct inline — a full replacement string, deeply nested
args — the more gpt-oss malforms it (placeholder `?`, a missing required field, doubled nesting).
A malformed call fails to parse, the loop nudges it, and it spirals (see the tool-call-only note).
Measured: a bare `collection_write` (create) hit 5/5; the same task via `update_entry` with a
full-content replacement spiralled repeatedly. Two implications: keep the model's calls small, and
when a footgun is mechanical (e.g. the model copies the `[key]` display-brackets into an
`update_entry` key → "not found"), fix it in the **tool** (accept/normalise the input, or make the
error actionable) rather than adding a prompt caveat the model will argue with.

## Isolate one case when tuning

When several behaviours share a prompt, tune them **one at a time** against the single case you're
fixing (`pytest … -k that_case`), not the whole suite. It's faster, and — critically — it stops
you misreading cross-case variance as a prompt effect. Confirm the fixed case, then re-run the
others to catch a regression you introduced.

## Process (how to change a prompt)

- **Dry-run every prompt change against the live model** (`make eval` / a focused case) and read
  the result **before** committing. A prompt you wrote but didn't run tells you nothing.
- **Change ONE lever at a time.** Learned the hard way: rewriting style + structure + the
  specific rule at once made a regression un-attributable and cost ~14 eval rounds chasing
  ghosts. If you change five things and it regresses, you've learned nothing.
- **Read the model's thinking on a failure**, not just the scorer line — that's where the
  reason lives (the harness auto-dumps it for failed samples).
- **Check the scorer before blaming the model.** A surprising 0/N is as often a too-strict
  scorer as a real failure. (Measured twice this codebase: a scorer requiring the literal word
  "skill" rejected perfectly good messages like "added your phrasing for viewing collector
  logs".)
- **Ship a durable eval contract** with every model-facing change — the case that encodes the
  behaviour, so the next change can't silently regress it.

## Structure of a good prompt

1. **Frame** — one or two sentences on what this agent maintains/does and the key mental model
   ("A skill is a tool-call sequence: STEPS + TRIGGER. Work from the tool calls, not the wording.").
2. **Numbered steps** — each a tool call in the consistent `N. tool(args) — description` format.
3. **The load-bearing rules inline** where they apply, emphasised if critical, concrete with an
   example.
4. **Terminal actions explicit** — the user-facing `send_message(...)` and the closing `done(...)`
   as their own final steps.
