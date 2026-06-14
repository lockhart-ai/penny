# Self-Improvement Loop — Goals, Eval, and Dry-Run

> **Status:** Phase 1 is **done** — the eval isolation core lives in
> `penny/penny/tests/eval/`, all suites are migrated onto the real agents, and
> the old `scripts/prompt_validation/` harness is removed. Phases 2–3 (dry-run
> sandbox + model tool) follow. The "drift problem" section below is written in
> the present tense for the record — it describes what the migration eliminated.

## The north star

Penny should be managed *through Penny*: you state a goal, she derives the
mechanics that pursue it, and the gap between the goal and her actual behaviour
is something both you *and she* can observe and close — preferring edits to
data/prompts over code changes (see the "Behavior changes via data/prompt
before code" principle in the root `CLAUDE.md`).

The organizing concept is **intent-as-spec**:

- A collection's `intent` is the **goal** — the user's stated objective,
  captured verbatim at create time ("find me X and tell me when Y").
- Everything else on the collection — `extraction_prompt`, `collector_interval_seconds`,
  `inclusion`/`recall`, the notify condition — is the **implementation** of
  that goal, and all of it is mutable.
- `intent` is **immutable to the agent** (create-only on the tool schema) but
  **editable by the human** (in the addon UI). The human owns the spec; the
  agent owns the implementation. This asymmetry is load-bearing: if the agent
  could edit the goal, it could "close" any gap by moving the goalpost instead
  of fixing behaviour.

With that in place the self-correction loop has a fixed target:

```
state a goal ──▶ it becomes the immutable spec (collection.intent)
            ──▶ Penny derives mechanics (extraction_prompt, cadence, notify)
            ──▶ eval / dry-run measures goal-satisfaction
            ──▶ the quality loop closes the gap (propose → test → apply)
            ──▶ the human refines the goal if the target itself was wrong
```

Goals map cleanly onto collection intents **for the collector's domain**.
Whole-system behavioural goals (chat style, identity, the skills collection)
are not collections and are out of scope here — that's a deliberate boundary,
not an oversight.

## Why this rests on eval

"Penny tunes her own prompts" is only real if there's an **error signal**.
A goal→mechanics derivation on a 20B local model will nail the easy half
(create a collection, store intent) and partially miss the inferred half
(the right cadence, the notify threshold). The eval harness is what *measures
that gap* — it turns a vibe into a closed loop.

But our eval harness has a structural problem.

## The drift problem (what we're fixing in Phase 1)

`scripts/prompt_validation/` is a **second, parallel implementation** of Penny
that deliberately avoids the real code so it can run on the host without the
full dependency set:

- it reads prompt text out of source via **AST** (`class_attr`),
- it **mirrors** the production enums (`_FakeEnum` for `Inclusion`/`RecallMode`),
- it **hand-builds** the tool dicts and the recall block,
- it runs a **duplicate** agentic loop (`converse()`).

Every one of those is a manual copy of something real, and copies drift. The
"million bugs" we hit using the harness are all stub-vs-real mismatches —
structurally guaranteed, not bad luck.

Meanwhile we already have a *faithful* isolation mechanism: the integration
test infra (`conftest.running_penny`) constructs the **real** `Penny` with
mocked boundaries. The only thing it mocks that the harness needs *un*-mocked
is the model. `Penny.__init__` fully wires `chat_agent` and `collector`;
`run()` only adds the channel-listen + scheduler loops. And the tmp test DB
**runs migrations**, so seed skills (0043) and system logs already exist — the
recall block the harness hand-renders comes from the *real* recall path for
free.

So the fix is not to build isolation — we have it. It's to **delete the fake
one and point eval at the real one**, swapping a single boundary:

| boundary       | integration test     | eval (this work)         | dry-run tool (phase 3) |
| -------------- | -------------------- | ------------------------ | ---------------------- |
| **model**      | `MockLlmClient`      | **real** Ollama client   | real                   |
| **DB**         | throwaway + migrate  | throwaway + synth seed   | real, sandboxed        |
| **channel**    | `MockSignalServer`   | `MockSignalServer`       | capture                |
| **side effects** | apply to throwaway | apply to throwaway       | captured / rolled back |
| **browse**     | mocked               | mocked (reproducible)    | mocked or real         |

One isolation core, three consumers, parameterized on `(model, db, side-effect policy)`.

**What dies:** `_harness.py`'s AST loading, the `_FakeEnum` mirrors, the
hand-built tool dicts and recall block, the duplicate `converse()` loop.
**What survives:** the scorers ("what good looks like") and the synthetic
fixtures (board-games / espresso / houseplants) — but the scorers get *better*,
inspecting **persisted DB state after a real run** instead of parsing captured
tool-call JSON.

## Phase 1 — the eval isolation core

A pytest suite under `penny/penny/tests/eval/`:

- **Construction:** reuse `running_penny` with a config whose model points at
  the real Ollama endpoint (`make_config(llm_model=…, llm_api_url=…, db_path=…)`).
  No new fakes, no second construction path.
- **Drive:** chat cases push a real message (`signal_server.push_message`);
  collector cases call `collector.run_for(name)`. Both run the *real* loop end
  to end against the real model.
- **Score:** inspect `penny.db` (created/updated collections, written entries)
  and captured channel messages — the persisted effect *is* the contract.
- **Threshold contracts, not exact-match.** The model is stochastic, so each
  case samples N runs and asserts `pass_rate >= k`. This is the one property
  that makes eval a different animal from the rest of `tests/`.

### Gating

These are **slow and need a live model**, so they are *not* part of CI or
`make check`:

- the `eval` pytest marker is registered in `pyproject.toml`;
- `make check` / `make pytest` run with `-m "not eval"` (deselected);
- `make eval` runs `-m eval` against live Ollama (`gpt-oss:20b` + `embeddinggemma`).

They are committable, reproducible **contract tests** that define the behaviours
we expect — invoked locally on demand, never in CI, never against prod data.

## Phase 2 — the dry-run sandbox (over real prod data)

The runtime needs to answer "if I ran *this* candidate prompt against the
*current* state, what would happen?" without persisting anything. That's the
eval core with a different DB/side-effect policy:

- **reads real but non-consuming** — it sees the logs the real cycle would see,
  but must not advance the read cursor;
- **writes / sends captured, not applied** — record what it *would* write/send.

Three options for the DB substrate, to be decided in Phase 2 (external effects
like `browse`/`send_message` need capture regardless):

1. **transaction rollback** — run the cycle in a txn, inspect, `ROLLBACK`.
   Elegant, but fights the stores' per-call-commit pattern.
2. **snapshot** — copy the relevant DB slice into a throwaway and run there.
   Cleanest isolation, heavier. *(current lean)*
3. **read-through / write-capture proxy** — reads hit prod, writes intercepted.

## Phase 3 — the model tool

Wrap the Phase-2 sandbox as `dry_run_collection(name, candidate_prompt)` →
`{would_write, would_send, summary}`, exposed to the quality collector (and
chat). It's the sandboxed sibling of the existing `TestExtractionPromptTool`
(which runs `collector.run_for` *for real*) — the delta is capture-don't-apply.
The quality collector reads the dry-run output, compares it against the
collection's `intent`, and applies the prompt change only if the gap shrinks
(then notifies the user — apply-then-tell, since there's no reliable approval
gate). Building this *before* eval rides real agents would ship the fake
harness's drift to the model, so it is strictly last.
