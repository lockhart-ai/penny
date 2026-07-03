# Prompt & Recall Efficiency — What Shipped

**Goal.** Keep Penny's prompts as short as possible while the model still gets the
information it needs. Shorter, better-targeted prompts reduce per-message
round-trip latency (prompt-eval/prefill dominates on a local model) without
losing the context that makes replies good. Pursued as a series of *validated*
changes, not a rewrite.

The guiding lens for any change: does it **remove tokens** (a real per-turn latency
win), **add tokens** (only if a measured behaviour needs it), or **reorganize** (free)?

> Two workstreams shipped and are captured here. A third (a skills self-authoring
> loop) was explored and set aside; the durable takeaways from that exploration
> live in `docs/prompt-writing-guide.md` (how to write Penny's gpt-oss prompts).

---

## 1. Static vs. dynamic prompt split — shipped (PR #1296)

**Problem.** The chat system prompt was ~8K tokens, ~90% of it a per-turn recall
block, with a minute-precision timestamp at character 0. On a local
(Ollama/llama.cpp) backend the KV-cache reuses a stable *prefix*; a change at the
front invalidates everything after it.

**What we did.** `_build_messages` now splits by volatility:
- **System prompt (static/slow):** the *date* (stable within a day), a shared
  injected-context note, identity, instructions, memory inventory.
- **Final user turn (volatile):** the *time*, ambient recall, the browser page
  hint, and — for collectors — the run history, all behind a shared
  `### Live context` header (`_final_user_turn`). A one-line note in the system
  prompt declares that the block is background, not user speech or an instruction.
- Collector runtime rules were lifted to the **front** of the collector system
  prompt (identical across every collector → a shareable cached prefix).
- The `INJECTED_CONTEXT_NOTE` / `_HEADER` constants are a single source, so chat
  and collector emit byte-identical framing.

**KV-cache findings (measured via prompt-log replay).**
- Ollama *does* do partial-prefix reuse: appending to a cached prompt is cheap;
  a front-of-prompt edit forces a full re-prefill. So the char-0 timestamp was
  genuine cache poison.
- **Within one agentic loop**, the system prompt is built once and frozen, so
  prefix reuse already works (replaying a real multi-step loop was ~3.3× faster
  than a counterfactual that regenerated the timestamp per step). Takeaway there
  is a *warning*: never regenerate the system prompt per loop step.
- **Across messages** the win is smaller: recall is per-turn dynamic bulk that's
  reprocessed regardless of placement. So the structural split's payoff is
  modest — its real value is **discipline**: a clean static/dynamic boundary and
  a delimited-injection convention that future prompt changes inherit.
- The article's explicit-caching advice (cache_control, cost discounts) is
  Anthropic-API-specific and does **not** map to local Ollama.

**Delineation.** Moving recall into a turn risks the model treating it as the
user's message. A dedicated adversarial eval (`chat-delineation`) confirmed it
doesn't: with a request-shaped entry injected into the Live-context block and a
different actual user message, the model answers the user and ignores the
injected block (5/5).

**Validation.** `make fix check` green; verbatim system-prompt *and* Live-context
turn dumps asserted char-for-char (chat and collector); `make eval`
non-regression (retrieval, recall-answer, browse, collector honesty).

---

## 2. Stage-1 recall routing — shipped (PR #1297)

**Problem.** An audit of real chat turns found recall injecting **~43 entries per
message**, most below Penny's own relevance bar. Two leaks in stage-1:
1. Each `relevant`-inclusion collection was gated *independently* — every
   collection over the floor was admitted, so a message pulled in a whole tail
   of loosely-related collections.
2. The score was the *max over the whole conversation window*, so a collection
   stayed "sticky" for later, unrelated turns.

**What we did.** Stage-1 is now **competitive** and **current-message-only**:
- Score each `relevant` collection by cosine of the **current message** (not max
  over history) to its description anchor.
- Admit only the top **`RECALL_TOP_K`** (new config, default **1**) that clear
  `MEMORY_INCLUSION_THRESHOLD` (0.40). The single on-topic collection wins; the
  runner-ups are dropped.
- `always`/`never` collections unchanged; threshold unchanged.
- `_passes_inclusion` (a per-memory boolean) → `_included_memories` /
  `_top_relevant` (a competitive selection across relevant collections).

**Grounding.** A labelled tuning harness over the real message corpus: top-1
keeps the right collection in ~23/30 on-topic messages, and off-topic messages
route to **zero** collections. Before/after simulation: relevant
collections/message **2.4 → 0.3** (~86% fewer topical entries).

**Key negative result.** A hybrid (embedding + IDF-lexical) score was measured
for stage-1 and did **not** help (worse than embedding-only). Collection
*descriptions* are one terse line — too little lexical surface. Hybrid stays in
stage-2, where entries have real vocabulary.

**Validation.** `make fix check` green; added a deterministic top-1 unit test.
Selection-only change (prompt structure unchanged), so no live-model eval
contract.

---

## Principles that transferred (recall/token work)

- **Sort every change by tokens: removes / adds / reorganizes.** Removers are the
  real latency levers; adders (rationale-in-rules, few-shot examples) are only
  worth it for a *measured* behaviour need; reorganizes are free.
- **Calibrate relevance to the model's own bar.** embeddinggemma cosine on short
  text runs low across the board; judge by the *relative* gap and Penny's own
  0.40 gate, not an absolute intuition.
- **Hybrid (embedding + lexical) helps only where there's lexical surface.**
  Useless for one-line collection descriptions; useful where entries carry real
  vocabulary.

(For prompt-*writing* learnings — numbered tool-call steps, positive spine +
emphatic guards, read→plan→execute, gpt-oss quirks — see
`docs/prompt-writing-guide.md`.)

## Pointers

- Structural split: `agents/base.py` (`_build_messages`, `_final_user_turn`,
  `_build_injected_context`), `prompts.py` (`INJECTED_CONTEXT_*`).
- Stage-1 routing: `agents/chat.py` (`_included_memories`, `_top_relevant`),
  `config_params.py` (`RECALL_TOP_K`).
