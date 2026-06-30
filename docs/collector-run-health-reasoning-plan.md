# Plan — Give Penny structural signals to reason about fruitless collector runs

> **Status:** All six phases committed on branch `collector-fruitless-run-detection`
> and validated live (see each phase + §3). Phase 5/6 adopted the **suggest-not-apply**
> pivot (quality proposes a fix message; it never calls `collection_update`). Optional
> pre-merge confirmations remain in §3 (full quality-suite run + real-prod-DB
> `render_run_record` dry-run).

This doc is the full plan from the start, written so a fresh context can pick it
up. It is privacy-safe on purpose (the repo is public): it names **no** real
collection, run-id, topic, or user content. Where a concrete example is needed it
uses a generic "a news-style collector".

---

## 1. The premise (what we are actually building)

A background **collector** cycle browses, reasons, optionally writes entries to its
collection and/or sends the user a message, then calls
`done(success=<bool>, summary=<prose>)`. The `summary` is **model-authored prose**
and was, until this work, **unchecked**.

Observed failure: a collector browsed many pages, **every useful read failed**, it
**wrote nothing**, and then closed with `done(success=true, summary="wrote 3 new
entries")`. The summary was a confabulation. The `quality` self-review collector
(which reads `log_read("collector-runs")`) saw only that lying summary and judged
the collection **healthy** — so nothing got corrected, and the collector kept
flailing to max-steps every cycle.

**The point of this effort is NOT to band-aid that one case.** It is to give Penny
**general, neutral, structural signals** about what her collector runs actually did
— so *she* (the quality collector, and the addon) can **reason** about her own
behaviour. We are widening the run record from "a summary the model wrote about
itself" to "the structural facts of what happened, which the model reasons over."

### Framing principle — TOP OF MIND for every phase

> **Describe what happened. Never encode why, or the remedy, for one observed case.**

Concretely, this was corrected twice during phase 3 — do not regress it:

- A run-health flag / counts line states **facts only** ("a browse failed", "0
  writes"). It must NOT say "cannot reach the source", "the URL needs changing",
  "no fresh browse will fix it", etc. We do not know any of that — the model
  reasons about cause and fix.
- The quality prompt (phase 5) must teach **general reasoning** ("find a failing
  run → read that collector's run history → if there's a consistent pattern,
  address it") — NOT "if you see NO WRITES, change the URL." The model decides the
  fix.

If a future edit reads like it only helps the one example we saw, it's wrong.

---

## 2. The smoking gun (why the old signal was blind)

`BrowseTool.execute` (`penny/penny/tools/browse.py`) returns
`ToolResult(message=...)` with the **default `success=True`** even when **every**
sub-page errored. A batched browse with partial failures is normal and the model is
told to "work from whatever succeeded" — so we must NOT mark the whole call failed
on partial errors (doing so would make `⚠ TOOL FAILURES` fire on healthy
multi-query browses → over-correction).

Consequence: `record.failed = not result.success` is **always False for browse**, so
the persisted `promptlog.tool_failures` count — and the existing `⚠ TOOL FAILURES`
run-health flag — are **structurally blind to read failures**. A browse read failure
is visible **only** in the result text (the `## browse error:` section headers).

Therefore: **read-failure counts must be derived from the rendered result text**, not
from a per-call success flag. This is a render-time derivation, no migration.

---

## 3. Where behaviour lives & the validation discipline (READ THIS)

From the root `CLAUDE.md` and `docs/self-improvement-loop.md`:

- **Behaviour changes via data/prompt before code.** Render/structural facts are
  code; the quality *decision* lives in its `extraction_prompt` (a migration-seeded
  memory row).
- **Fix data/rendering in Python FIRST so the signal is visible, THEN the prompt
  loop applies.** That is exactly the ordering here: phases 2–4 (Python: make the
  facts visible + readable) before phase 5 (quality prompt) and phase 6 (eval).
- **Every model-facing change ships a durable eval contract, dry-run against the
  live model BEFORE committing the prompt.** Eval suite: `penny/penny/tests/eval/`,
  run with `make eval` (live Ollama; slow). Quality cases live in
  `tests/eval/test_quality_correction.py`. Browse is stubbed via the `browse=`
  kwarg (query-keyed `CannedPage`/`install_browse` in `tests/eval/conftest.py`).
- **The log→test→fix loop:** drive fixes from real failing promptlog rows
  (`replay.py` reads a row by id and replays vs the live model), genericize for
  privacy, then lift into a committable eval case. Always read the model's
  **thinking** on a failure (auto-dumped). Iterate focused low-N first
  (`EVAL_SAMPLES=3 pytest "…::test_x" -m eval -s`), full suite once at the end.
- **`prompt_test` was removed (migration 0063)** — gpt-oss couldn't drive the
  dry-run→revise→apply loop (emitted the revised prompt as text, not a tool call).
  Quality now rewrites the drifted `extraction_prompt` **directly** with
  `collection_update` and the next cycle re-checks. **Do NOT reintroduce
  `prompt_test`.**

### ⚠ Skipped validation that still must happen before the PR lands

1. **Dry-run the new `render_run_record` against REAL production promptlogs.** The
   render logic is unit-tested (`tests/database/test_run_health.py`) but has NOT
   been eyeballed on real runs. Prod DB: `./data/penny/penny.db` (mounted into the
   `penny` container at `/penny/data/penny/penny.db`, `DB_PATH` default). Run a
   throwaway script in the `penny` Docker image (deps + this branch's code are
   there via the dev override) that loads real runs and prints
   `render_run_record` + `classify_run`. Confirm: the counts line and `⚠ NO WRITES`
   appear correctly on real failing runs, and do NOT appear on healthy ones.
   *(Do not commit any output — it contains real content.)*
2. **`make eval` for the model-facing prompt changes** — phase 1's `done()`
   guidance: ✅ DONE — `tests/eval/test_collector_honesty.py` validates it live,
   baseline (no phase 1) **0/3** → with phase-1 wording **3/3** (working-source guard
   3/3); no wording iteration needed. Phase 5's quality prompt (suggest-not-apply, 0073):
   ✅ DONE — persistent-no-writes 3/3, single-no-writes 3/3, healthy 3/3, rebroadcast 3/3
   (after a tier-1 nudge), silent-drift 2/3, **0 `collection_update` calls across every case**.
   *(P2/P3 non-regression also ✅: the six quality over-correction guards — `healthy`,
   `incomplete`/`max-steps`/`tool-failure`/`run-failure`-not-drift, `quiet-when-healthy`
   — all hold with the new rendering; the lone `quiet` 1/3 slip was flag-INDEPENDENT
   over-correction, the failure the phase-5 suggest-not-apply pivot makes structurally
   impossible.)* Needs Ollama up (gpt-oss:20b); **pause penny first** — it contends for
   Ollama and turns a 47s case into 40+ min. Run cases one at a time.

Tests are run ONLY via:
`make fix check 2>&1 | tee /tmp/check-output.txt; echo "EXIT_CODE=$pipestatus[1]" >> /tmp/check-output.txt`
then read `/tmp/check-output.txt`. Never `make pytest` / `make check` alone /
`docker compose run` for the test suite. (`make eval` is the separate live suite.)

---

## 4. Git / branch state

- Branch: `collector-fruitless-run-detection` (off latest `main`).
- The earlier Memories-tab UI work shipped separately as **PR #1291** (merged).
- Phases 1–3 are in **one commit**: "Make collector runs structurally describe
  browse/write outcomes". Phases 4–6 are now committed too (run-history tool;
  migration 0073 suggest-not-apply quality; flipped eval contracts).
- Always branch from latest main; use `make token` for any `gh`/push
  (assert non-empty token → must be the bot identity); check the PR is still open
  before pushing.

---

## 5. The six phases

### Phase 1 — Honest `done()` guidance  ✅ committed + eval-validated (0/3→3/3, see §3)

Model-facing backstop, attached structurally (runtime invariant, not via the
prompt-writer). File: `penny/penny/agents/collector.py`, the `_RUNTIME_RULES`
class attribute. The `done()` bullet now says the summary must state what
*actually* happened — never claim writes without a successful `collection_write`;
if sources couldn't be read, say so and set `success=false`; the
"no new matches" quiet-cycle path is preserved and distinguished from
"could not read". Verbatim copy updated in `tests/agents/test_collector.py` (the
prompt-assertion test inlines the rules block).

This is the *weak* lever — phases 2–3 make it not matter whether the model
complies, because the facts are read structurally regardless.

### Phase 2 — Structural I/O counts on the run record  ✅ committed

All in `penny/penny/database/memory/objects.py` (the shared
`render_run_record`/`classify_run` that BOTH the quality collector and the addon
prompts/activity tabs read — one representation).

- `_run_io_tally(prompts) -> (browses_ok, browses_failed, reads, writes, sends)`:
  - **browses** = web-read sub-results, counted from `PennyConstants.BROWSE_PAGE_HEADER`
    / `BROWSE_SEARCH_HEADER` (ok) and `BROWSE_ERROR_HEADER` (failed) in the
    **last prompt's `messages`** (the full accumulated conversation → each result
    once; the final step is ~always `done()` so nothing material is missed).
  - **reads** = internal collection/log reads = every call that isn't a browse,
    write, send, or `done` (so `log_read`, `collection_read_*`,
    `read_published_latest`, `read_similar`, `collection_catalog`, …). Plain count
    (they don't meaningfully fail).
  - **writes** = count of `collection_write`/`update_entry`/`collection_delete_entry`/
    `log_append` **calls** (from responses, each once). `writes == 0` is the hard
    fact a lying summary contradicts.
  - **sends** = `send_message` call count.
- `_io_tally_line(prompts)`: renders
  `browses: A ok, B failed · reads: C · writes: D · sends: E` — **writes always
  shown** (so `writes: 0` is always legible); browses/reads/sends shown only when
  nonzero. Returns `None` (no line) for a read-only quiet cycle; shown whenever the
  run browsed, wrote, or sent.
- `render_run_record`: the counts line sits **directly under the `[target] summary`
  header**, before the `⚠` flags and the trace — so the contradiction with the
  summary is adjacent and obvious.

Why "browses" vs "reads" are split: most system collectors read via `log_read`
(which never fails); labelling browse counts as "reads" would print "reads: 0" on a
cycle that read plenty. Keeping them separate makes every label accurate.

### Phase 3 — `NO_WRITES` run-health flag  ✅ committed

A neutral structural flag = **a browse failed AND the run wrote nothing**
(`browses_failed > 0 and writes == 0`). It says nothing about why/fix.

- `penny/penny/validation/conditions.py` (the frozen shared catalog — strings
  mirrored by the quality prompt + the addon TS): new
  `ConditionKey.NO_WRITES`; catalog entry `run_flag=True, collector_only=True`,
  marker `"NO WRITES"`, detail `"one or more browses failed this cycle and the run
  wrote nothing"`. Render order: `NO_WORK_DONE → NO_WRITES → INCOMPLETE →
  TOOL_FAILURES → HALF_FORMED_SEND` (the two "didn't deliver" flags lead; the
  capacity ones follow).
- `objects.py`: `RunHealth.no_writes` field, wired into `flags`, `_flag_is_set`,
  and computed in `classify_run` from `_run_io_tally`.
- It is **distinct from** `TOOL_FAILURES`/`INCOMPLETE`, which migration 0072 tells
  quality to IGNORE as capacity/transience. `NO_WRITES` is **actionable drift**
  (like `NO_WORK_DONE`/`HALF_FORMED_SEND`). What makes it drift and not noise is
  that the quality prompt will require a **persistent pattern across the
  collector's recent runs** (phase 5), not one bad cycle.
- TS mirror updated: `browser/src/protocol.ts` (`RunHealthFlag` union + `RunHealth`
  interface gain `no_writes`); `browser/src/page/page.ts` (default health literal +
  `RUN_HEALTH_LABEL`). It correctly inherits the **actionable red** badge/line
  styling (it is NOT added to the capacity-amber regex at `page.ts` ~`record-line-capacity`).

Tests: `tests/database/test_run_health.py` — added a `_browse_messages(...)` helper
+ a `messages=` param to `_prompt`; 4 existing expected renders updated for the new
counts line; 3 new tests: `no_writes` fires on failed-browses+0-writes (with a
lying summary), is NOT set on a clean quiet browse cycle, and is NOT set when a
partial browse failure still produced a write.

**Verified:** `make fix check` green (757 penny + 176 penny-team, tsc clean,
EXIT_CODE=0) — but see §3 for the missing real-data dry-run + eval.

---

### Phase 4 — A tool to read one collector's run history  ✅ committed (collector_run_history)

So quality can do "find a failing run → **read that collector's recent runs** →
decide if it's a consistent pattern", instead of judging a single cycle.

**Design decisions (carry these):**
- **Do NOT overload `log_read`.** `LogReadTool` (`penny/penny/tools/memory_tools.py`)
  is caller-dispatched: cursor-mode for collectors, window-mode for chat, and "the
  caller never chooses the mode or a size; Python does." A `target`+`n` arg would
  break that. Add a **dedicated tool** instead.
- The model passes the **collector/collection name** (it gets candidates from the
  `log_read("collector-runs")` index it already reads). **Python fixes the count**
  (a `PennyConstants` constant, e.g. reuse a run-history lookback). Keep "model
  doesn't pick size."
- **Return rendered run *records*** (with the new counts line + flags), newest
  first — so the model reasons over the same structural representation, across the
  target's recent cycles.

**Data path (already mostly present):**
- `penny/penny/database/message_store.py`:
  - `recent_run_summaries(run_target, limit)` → `(ts, summary)` only (too thin —
    summaries are the prose we're trying to get past).
  - `get_target_runs(run_target, limit, offset)` → fully serialized run **dicts**
    for the addon (heavier than needed).
  - `_page_of_target_run_ids(session, run_target, limit, offset)` exists and is
    served by the `ix_promptlog_target_runs` partial index (migration 0062) — use
    it.
  - `render_run_record` is imported here already (line ~15).
- **Recommended:** add `db.messages.target_run_records(run_target, limit) ->
  list[str]` that pages target run-ids (reuse `_page_of_target_run_ids`), groups
  their prompts, and returns `render_run_record(...)` per run (newest first). This
  routes through `db.messages` and avoids `RunLog`-facade polymorphism concerns
  (don't add a `RunLog`-only method that a generic tool would have to duck-type).
- New tool e.g. `CollectorRunHistoryTool` (in `memory_tools.py`): args
  `{collector: str}` (Pydantic args model), resolves the `collector-runs` constant
  internally, calls `db.messages.target_run_records(collector, <const>)`, formats
  via the shared `_format_entries(...)` (newest-first ordering). Add it to the
  collector read surface in `Collector.get_tools` / `build_memory_tools`. Tool
  surface stays **uniform** across collections (every read tool is; only the
  prompt decides who calls it) — do NOT gate it to quality-only (that
  non-uniformity is what we removed with `prompt_test`).
- Actionable failure message if the name is unknown / has no runs (per the
  actionable-tool-failure rule).

Tests: an integration test that the tool returns recent records for a target
(fold into existing collector/memory-tool tests where one fits).

### Phase 5 — Quality prompt: suggest, not apply  ✅ committed + eval-validated (migration 0073)

**DESIGN PIVOT (confirmed by the user, June 2026): quality SUGGESTS, never applies.**
See `project_quality_collector_suggest_not_apply`. Quality recently made destructive
edits to a healthy collection (the user paused it), and the `quiet-when-healthy` eval
reproduced the same slip live (the model saw a clean run — no flags — and called
`collection_update` on it anyway, unprompted). So quality is demoted from editor to
**proposer**: it detects → composes a complete, specific edit *suggestion* as a message
→ stops. The user approves ("yes, do it") and the **chat agent** makes the edit through
its normal collection-edit tools. This removes the blast radius entirely (worst case =
a wasted message) and is newly viable because the "messages weren't appearing in Penny's
history" bug is fixed (the user must reliably SEE the suggestion). Phases 1–4 are
unchanged — better detection feeds a better suggestion either way; only the *act* changes.

A new migration (latest quality prompt is **0072**; do not edit it in place —
add the next-numbered migration; run `make migrate-validate`). Rewrite the
`quality` collector's `extraction_prompt` so its sequence is, in general terms:

1. Read recent runs (`log_read("collector-runs")`) and find a run that looks like a
   problem.
2. **Read that collector's own recent run history** (the phase-4 tool) — is the
   problem a one-off or a **consistent pattern across cycles**?
3. If there's a persistent pattern that contradicts the collection's `intent`,
   `send_message` a **complete, specific edit suggestion** (which collection, what's
   wrong, the concrete change to make — enough that a "yes" is executable), then
   `done()`. **Quality never calls `collection_update`.** If it's a one-off /
   capacity / transience, change nothing (quiet cycles are normal).

Constraints:
- **Remove `collection_update` (and any entry-mutation tool) from quality's tool
  surface** so the suggest-only contract is *structural*, not just prompted — the
  model cannot apply even if it tries. This is the real fix for the over-correction
  reproduced in `quiet-when-healthy`. (Quality keeps reads + `send_message` + `done`.)
- **No hardcoded remedy.** Do NOT write "if NO WRITES, change the URL/search
  terms." Describe the *flag's meaning generally* (a tier keyed on the flag, like
  the existing tiers — gpt-oss responds well to numbered, one-shot-style prompts —
  per `project_numbered_prompts_beat_prose`), then let the model diagnose and
  *suggest* the fix.
- Keep 0072's conservatism intact: `⚠ INCOMPLETE` / `⚠ TOOL FAILURES` and a
  `failed`/`cancelled` run that called real tools remain **ignored** as
  capacity/transience. `NO_WRITES` is the new **actionable** signal, but only when
  **persistent** (that's what distinguishes it from a transient failed browse).
- Runtime invariants append structurally; only authoring rules go through the
  prompt; the catalog marker/detail strings are frozen — if you change them it's a
  coordinated migration + TS update.
- The chat agent must be able to act on "yes, apply that" — it has the collection-edit
  tools already; the work is making sure the approved suggestion carries enough to
  execute from (and that the suggestion surfaces in history, now that it does).

### Phase 6 — Eval contract (suggest-not-apply)  ✅ committed + eval-validated

Add to `penny/penny/tests/eval/test_quality_correction.py` a **pair** (the
drift-N/quirk-N discipline used by PR #1276) — scored on the **suggest-not-apply**
contract:

- **Drift case:** a collector whose browses fail across cycles and writes nothing
  → quality detects the pattern and **sends a specific, actionable edit suggestion**
  (names the collection + the concrete change). Assert: a suggestion message was sent
  AND **nothing was mutated** (the `extraction_prompt` is unchanged — quality has no
  mutation tool). Score the message's *presence + specificity*, not its wording. Use
  the `browse=` kwarg with `CannedPage(fails=True)` / `ALL_BROWSES_FAIL` to install
  failing pages across cycles.
- **Quirk / healthy guard:** a collector with a working source that reads fine and
  writes → quality **leaves it alone** (no suggestion, no mutation). This proves we
  didn't reintroduce over-correction (the thing 0072 guarded against, and the slip
  `quiet-when-healthy` showed is now structurally impossible without the edit tool).

Validate with `make eval` (focused low-N first; read the thinking on failures;
full suite once at the end). Also pair with a deterministic `tests/` mock test if
there's a mechanism to pin (e.g. assert quality's tool surface excludes
`collection_update`). The phase-1 `done()` guidance is covered by
`tests/eval/test_collector_honesty.py` (already validated 0/3→3/3) plus confirming
the extractor cases don't regress under the new runtime rules.

---

## 6. Key file map

- `penny/penny/agents/collector.py` — `_RUNTIME_RULES` (phase 1); `get_tools`
  (phase 4 surface); `_run_history_section`/`recent_run_summaries` (own-history
  block, separate from phase 4's other-collector read).
- `penny/penny/database/memory/objects.py` — `_run_io_tally`, `_io_tally_line`,
  `RunHealth`, `classify_run`, `render_run_record`, `RunLog` facade.
- `penny/penny/validation/conditions.py` — the frozen condition catalog
  (`ConditionKey`, `run_flag_conditions`).
- `penny/penny/database/message_store.py` — `recent_run_summaries`,
  `get_target_runs`, `_page_of_target_run_ids`, `_runs_for` (+ add
  `target_run_records` for phase 4).
- `penny/penny/tools/memory_tools.py` — `LogReadTool`; add `CollectorRunHistoryTool`.
- `penny/penny/tools/browse.py` — the always-`success=True` browse (the smoking gun;
  do not "fix" by failing on partial errors).
- `penny/penny/constants.py` — `BROWSE_PAGE_HEADER` / `BROWSE_SEARCH_HEADER` /
  `BROWSE_ERROR_HEADER` / `SECTION_SEPARATOR`; add a run-history-count constant.
- `browser/src/protocol.ts`, `browser/src/page/page.ts`, `browser/page/page.css` —
  the addon RunHealth mirror (phase 3 done; capacity regex must keep excluding
  `NO WRITES`).
- Tests: `tests/database/test_run_health.py` (render/flags), `tests/agents/test_collector.py`
  (verbatim prompt), `tests/eval/test_quality_correction.py` (phase 6).
- Migrations: `penny/penny/database/migrations/` (phase 5 = next number after 0072).

## 7. Hard constraints (do not violate)

- **Privacy:** repo is PUBLIC. Never put real collection names, run-ids, topics, or
  user content in commits/PRs/code/tests/this doc. Genericize.
- **Framing:** structural facts only in the data layer; general reasoning in the
  prompt; no hardcoded single-case remedies anywhere (§1).
- **Tests:** only `make fix check` (piped to `/tmp/check-output.txt`) for the
  suite; `make eval` for live model; ignore IDE/Pyright diagnostics.
- **DB:** ask before touching the production DB / `runtime_config`.
- **Model-facing changes ship an eval contract, dry-run before commit** (§3).
