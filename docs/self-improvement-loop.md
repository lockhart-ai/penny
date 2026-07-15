# Self-Improvement Loop — Goals, Eval, and Dry-Run

> **Status (superseded in part, #1569):** the model-judgment **`quality`
> self-correction collector this document centers on is RETIRED** — it existed to
> correct drift in `extraction_prompt`s *generated from prose*, and that authoring
> channel is gone (a collector's prompt is now a deterministic render of a taught
> skill, #1590/#1591; a wrong prompt is fixed by the user re-teaching the skill).
> The correction channel moved from model judgment to the structural teach loop,
> so the reviewer retired with the failure mode. Its eval (`test_quality_correction.py`)
> was deleted. The rest of this document — the eval isolation core, intent-as-spec,
> the log→test→fix loop — stands; the `quality`-specific sections are a historical
> record.
>
> Phase 1 (eval isolation) is **done** — the core lives in `penny/penny/tests/eval/`,
> all suites are migrated onto the real agents, and the old
> `scripts/prompt_validation/` harness is removed.

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

### The canonical use-case matrix

Beyond the self-improvement cases, the eval suite is the **canonical coverage of
Penny's core use cases** — and, because it gates on real-model behaviour, the
**yardstick for swapping models** (e.g. off `gpt-oss:20b`). Run the suite against
a candidate model and read the pass-rates and `PERF` lines side by side.

Everything Penny does reduces to **two agent shapes**, each branching on whether
it needs the world or just its own memory — what we're really measuring is
*does the model reason effectively about tool calls* (right tool, right args,
correct next call from what came back). The suite spans that matrix:

| axis | answer from memory/context | reach for the web (browse → reason) |
|---|---|---|
| **chat** (`test_chat_response.py`) | chitchat; recall-grounded answer | browse→answer; multi-hop browse chain |
| **chat authoring** (`test_collection_lifecycle.py`) | create / update / archive / abstain | — |
| **collector** (`test_extractors.py`) | likes / dislikes / knowledge / notify send+move | research-watcher; inner-monologue |
| **meta-collector** | *retired* — the `skills` reconcile collector (#1624) and the `quality` reviewer (#1569) both retired: skills are now structural (taught/instantiated/re-rendered) | — |
| **routing** (`test_retrieval.py`) | two-stage recall | — |
| **peripheral** (`test_peripheral.py`) | startup announcement | schedule NL→cron dispatch (`test_schedule_dispatch.py`) |

The built-in collectors (`likes`, `dislikes`, `knowledge`, `thoughts`) already
exist with their **canonical migration-seeded extraction prompts** in a fresh eval
DB, so an extractor case only seeds the collector's *input* (the `user-messages` /
`browse-results` logs) and runs the real prompt. (The `skills` reconcile collector
and the `quality` reviewer no longer dispatch — retired by #1624 / #1569; the
`notifier` and the `unnotified-/notified-thoughts` pair are archived shells.)

**Query-aware mock browser.** The isolation core stubs browse with one fixed
string — enough to check *whether* the model browsed, not *how it reasoned over
the result*. The `browse=` kwarg on `chat_eval` / `collector_eval` installs
`CannedPage`s keyed by a query/URL substring (`install_browse` in `conftest.py`),
so a case returns a realistic page (facts + a source URL in the visible body) and
a refined follow-up query maps to a different page — letting cases score the
*subsequent* call (the write, the send, the second browse) and even multi-hop
chains.

**Score behaviour, not content.** Because browse content is canned and the model
is stochastic, scorers assert on behaviour (tool called, entry written, message
queued, fact surfaced, nothing spurious created), never on exact wording. Cases
whose chain is long/stochastic (multi-hop browse, inner-monologue) are
`min_pass_rate=None` (report-only) — the X/Y rate prints for inspection without
gating, same convention as the quality cases.

**Performance metrics (model-swap picture).** Each case prints a `PERF` line:
calls, full request wall, in/out tokens, the **reasoning split** (`completion_tokens`
already bundles the thinking trace, so it's split by the stored `thinking`/`content`
char ratio — surfaces token-waste-on-reasoning, the usual challenger-model failure),
and an **end-to-end tok/s** (output ÷ full wall, *including* prompt processing — not
raw decode speed). True decode speed needs Ollama's native timings, which the
OpenAI-compatible `/v1` endpoint our client uses strips; so `test_perf_probe.py` hits
the **native `/api/chat`** for the configured model and prints a `PERF-PROBE` line with
prefill tok/s, decode (generation) tok/s, and reasoning share — the `ollama run
--verbose` numbers, captured per model for head-to-head comparison.

## Phase 2 — the dry-run sandbox (over real prod data)

> **REMOVED (migration 0063).** The `prompt_test` dry-run tool and its
> `_DryRunCollector` sandbox described in this section were taken out. In practice
> gpt-oss couldn't reliably drive the dry-run → read-result → revise → apply loop:
> tracing real failures, it would detect the problem and draft a correct fix, then
> emit the revised prompt as a **text blob instead of a tool call**, and the cycle
> died without applying anything. So the quality collector now rewrites a drifted
> `extraction_prompt` **directly** with `collection_update` (no dry-run) and relies
> on the next cycle to re-check. The rest of this section is kept as design history.

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

## Phase 3 — the model tool (shipped)

Done. The Phase-2 sandbox is wrapped as the **`prompt_test`** tool
(`penny/tools/prompt_test.py` → `Collector.dry_run`): the sandboxed sibling of
`TestExtractionPromptTool` (which runs `collector.run_for` *for real*) — the
delta is capture-don't-apply. We chose the **capturing-tool-surface** sandbox
over snapshot/clone: a throwaway `_DryRunCollector` runs the candidate cycle in
place with side-effecting tools captured, browse stubbed, and the log-read
cursor never committed (non-consuming reads via the `_should_commit_cursor`
hook). No DB copy.

The quality collector reads the dry-run output, compares it against the
collection's `intent`, and applies the change with `collection_update` only once
the dry run is clean — then notifies the user (apply-then-tell; no reliable
approval gate).

**Validated, then graduated.** A stubbed tool first proved (through the real
harness, real collector, real model) that gpt-oss can drive the full loop —
spot drift → draft a fix → `prompt_test` it → read the result → apply. The real
tool passed the same contract. Migration **0055** then seeds the `quality`
collector into every DB so all deployments get it; `Collector.get_tools` gates
`prompt_test` into the surface for that collection's cycles only
(`MEMORY_QUALITY_COLLECTION`). The contracts live in
`tests/eval/test_quality_correction.py`.
