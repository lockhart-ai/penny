# Eval Run Report Format

The comment-ready markdown one eval run posts to its iteration PR. This is the
**format contract** for that report: what every run comment carries, section by
section, so the renderer (#1693), the manifest writer (#1692), and a human
reading the PR all share one shape.

Read this alongside:

- **`docs/agent-task-workflow.md` §4** — the *protocol* (a PR is live from the
  first run; every run posts its report as a **new** comment). This document is
  the *format* those comments take.
- **`docs/self-improvement-loop.md`** — why the loop rests on eval and the
  through-line `manifest → check-diff → thinking → the comment stream`.
- **`penny/penny/tests/eval/conftest.py`** — where the report is rendered
  (`_write_sample_report`, `_sample_turns`, `_place_checks` today; extended by
  #1693). The base transcript shape below is what that code already emits.

## The one rule: key points in the comment, bulk on disk

A run comment carries the **evaluation's key points** — the verdict, what
regressed, and the reasoning at the turns that regressed — never the bulk. The
comment is read as GitHub markdown, in the PR's comment stream, as the durable
record of the iteration; anything a reader would scroll past belongs on disk.

**Goes in the comment:**

- the run manifest header (commit, model, config, N, the required lever)
- the dual RESULT line (mean-of-scores + all-pass rate) per case, run totals at top
- the per-sample verdict and turn-by-turn transcript table
- check stamps on the rows, REGRESSED marks, fragile flags, failure-cause counts
- check rationales (observed vs expected) on failed / regressed checks
- model thinking at failed / regressed turns, in collapsed `<details>`

**Stays local (never in the comment):**

- raw per-case **JSONL** result records (#1692) — the machine-diff artifact
- the verbatim **dirty-diff** the manifest saves (the header names the commit +
  whether the tree was dirty; the diff body itself is a local artifact)
- **archives** of prior runs' artifacts
- the ephemeral **per-sample DBs** (`<case>-<n>.db`) kept beside the reports
- passing-turn thinking, and any full transcript a reader would never open

The split is load-bearing: the comment stays skimmable so the *review* happens in
the comment, while the JSONL/DB artifacts stay diffable so the *next run* can
compute REGRESSED marks against them.

---

## Anatomy of a run comment

A run comment is: **one manifest header**, then **one block per case**, each
case block holding **one `<details>` per sample**. The sections below specify
each part; the [worked example](#worked-example) at the end renders all of them
with synthetic content.

### 1. Manifest header (required lever)

The first thing in the comment. Rendered from the run manifest (#1692) — the
inputs this run ran under, so a reader (and the diff against a prior run) knows
*what changed going in*.

Fields:

- **lever** — **required**, one line: the run's hypothesis (what this run changed
  vs. the last, e.g. "moved the notify-suffix guidance from the skill into the
  collector prompt"). A report run with no lever is an **error**, not a warning
  (#1692) — the loop's whole point is attributing a score shift to an input
  change, and an unlabelled run can't be attributed.
- **commit** — the branch commit the run ran against, and whether the tree was
  **clean** or **dirty** (a dirty run's exact diff is saved locally, not inlined).
- **model** / **embedding** — the text + embedding models under test (the
  model-swap yardstick reads these).
- **config** — `N` (EVAL_SAMPLES) and any non-default knobs.
- **prior** — the run this one diffs against (the prior comment / artifact), or
  "none (first run)" when there's no baseline yet, in which case REGRESSED marks
  are absent (nothing to flip against).

### 2. Run totals

Immediately under the manifest: the run-level aggregate across all cases —
mean-of-scores, all-pass rate, and the failure-cause tally (below). One glance
says whether the lever helped.

### 3. Per-case verdict — the dual RESULT line

Each case leads with a **dual RESULT line** (#1694's strict+partial):

- **mean** — mean of the per-sample scores (graded partial credit averaged).
- **all-pass** — the fraction of samples that scored a perfect 1.0 (the strict
  gate — the old binary pass-rate).

Both, always — a case can carry a healthy mean while few samples are *fully*
clean, and the gap is signal. When a case gates (`min_pass_rate` set), the line
names the threshold and which metric it gates on; a report-only case says so.

### 4. Failure-cause / pathology counts

Per case (and summed in the run totals), every **failed sample** is tagged with
its cause (#1695), and the tally renders:

- **behavioral** — the model got it wrong (the real signal the loop chases).
- **pathology** — a known failure mode fired: degeneracy reroll exhausted,
  leaked-Harmony-envelope, a detected collapse, a bare call-fragment reply. Noise,
  not comprehension. Detected structurally off the persisted RESPONSE with the same
  `text_validity` detectors the agent-loop reroll guard runs live — an eval-injected
  recovery trigger (a synthetic response that never reaches the persisted `response`)
  is structurally excluded, so a forced bail is never mistaken for a pathology.
- **harness** — a timeout or infrastructure fault, not the model at all.

A cause is only assigned to a **failed** sample, and pathology outranks a timeout
(the poison is the root cause, a downstream timeout its symptom).

The case reports the score **two ways**: raw (all samples) and
**pathology-excluded** — the mean over every sample that is NOT a pathology
failure (passing + behavioral + harness; only pathology drops out of the
denominator). The pathology-excluded score is the honest read of model behaviour;
the raw score and the pathology **count** stay visible so a spike in pathology is
itself legible (it usually means context got too large, not that the prompt is wrong).

### 5. Sample transcript table (the base format)

One collapsed `<details>` per sample. The `<summary>` names the sample, its
verdict, and any flags. Inside: the turn-by-turn transcript — the base shape the
harness already emits (`_sample_turns` / `_place_checks`), read verbatim from the
sample's promptlog.

The table is `| # | Actor | Content |`, one row per turn, actors:

| actor | glyph |
|---|---|
| user turn | 👤 user |
| Penny → tool call | 🔧 Penny → tool |
| tool result | 📥 tool result |
| Penny (reply / thinking-free text) | 🤖 Penny |

**Check stamps** land on the row each check anchors to: a ✅ or ❌ appended to the
actor cell of the first turn whose content matches the check's anchor (a tool
name like `collection_write(`, or the final reply). Checks that anchor to no row
— a *missing* expected action (a tool call that never happened), or a whole-run
check — fall to a footer line under the table.

The sample verdict in the summary is `✅ PASS` / `❌ FAIL` for a binary sample, or
`✅ N/M checks` for a graded one (M = the **scored** checks; advisory/flavour
checks render but don't count, and n/a checks are out of M entirely).

### 6. Mark legend

Stamps beyond the base ✅/❌:

| mark | meaning |
|---|---|
| ✅ | check passed |
| ❌ | check failed |
| ❌ 🔻 REGRESSED | failed **and** passed in the prior run — a flip, distinct from a check that was already red (#1693) |
| ➖ n/a | the check's branch didn't run; excluded from the graded denominator (#1694's ignore state) |
| 🔶 fragile | the sample **passed**, but with rejected calls / retries / recovery events — green, but not clean (#1694) |

A **REGRESSED** mark only exists when the manifest names a prior run; on a first
run there's nothing to flip against, so failures are plain ❌.

A **fragile** flag rides the sample verdict (in the `<summary>`), not a single
check — it says "this green came with a wobble". The sample names what wobbled
(which call was rejected/retried/recovered) in a one-line note under its table.

### 7. Check rationales (observed vs expected)

Every **failed** or **regressed** check renders its **rationale** (#1694) — what
was expected vs. what was observed, so a red mark is self-explaining without
opening the transcript:

```
❌ 🔻 REGRESSED  send_message queued — expected 1 send, saw 0 (run ended at done() with no send)
```

Passing checks render no rationale (they'd only add noise). The rationale is the
check's own `expected`/`observed`, not a model summary — it's computed from the
persisted state, the same doctrine as the scorer.

### 8. Thinking at failed / regressed turns

The highest-signal artifact, and today it dies before review (it only reaches
captured stdout via `_dump_thinking`). The report lifts the model's **thinking**
into the comment — but *only* at the turns anchored to a **failed or regressed**
check, each in its own collapsed `<details>` so passing turns don't bloat the
comment (#1693):

```
<details><summary>💭 thinking · turn 8 (send_message) — ❌ 🔻 REGRESSED</summary>

> I already wrote the winter-watering entry, so the watch is satisfied. The user
> didn't ask to be pinged, so I'll close out with done() rather than send.

</details>
```

Thinking at a *passing* turn is omitted (available in the local per-sample DB if
ever needed). On a first run with no baseline, thinking renders at **failed**
turns only (there are no regressed ones yet).

---

## Field glossary (names shared across #1692 / #1694 / #1695)

The report renders these; the artifact (#1692) and check machinery (#1694/#1695)
produce them. Keep the names identical across all three so a JSONL record and its
rendered comment read as the same run.

| field | source | meaning |
|---|---|---|
| `lever` | manifest (#1692) | required one-line hypothesis for the run |
| `commit` / `dirty` | manifest (#1692) | branch commit + clean/dirty flag |
| `model` / `embedding` / `samples` | manifest (#1692) | the run's model config + N |
| `case_id` | artifact (#1692) | `<file>::<case>` identifier |
| `family` | artifact (#1692) | family tag (explicit param, module-derived default) |
| `mean` | artifact (#1692) | mean of per-sample scores |
| `all_pass` | artifact (#1692) | fraction of samples scoring 1.0 |
| `pathology_excluded_mean` | #1695 | mean over every NON-pathology sample (passing + behavioral + harness) — pathology failures drop out of the denominator |
| `sample_causes[]` | #1695 | per-sample cause aligned with `sample_scores`: `behavioral` \| `pathology` \| `harness`, or `null` for a pass |
| `cause_counts` | #1695 | failed-sample tally `{behavioral, pathology, harness}` (derived, render-ready) |
| `checks[]` | #1694 | per-check `label` · `ok` · `scored` · `ignored` · `expected` · `observed` |
| `regressed` | #1693 | check flipped ok→fail vs the prior run's artifact |
| `fragile` | #1694 | sample passed with rejected calls / retries / recoveries |

---

## Worked example

One complete run comment, rendered. Entirely synthetic content — a houseplant-care
collector, in the flavour of the repo's real eval fixtures (board games,
houseplants, espresso). This is what the renderer (#1693) posts verbatim as a run
comment on the iteration PR.

---

> ## Eval run — `2026-07-20T14:32Z`
>
> **lever:** moved the notify-suffix guidance out of the `water-my-ferns` skill and into the collector prompt, so a `notify=true` watch always sends after a changed write
> **commit:** `a1b2c3d` (clean) · **prior:** run `2026-07-20T13:05Z` (comment ↑)
> **model:** gpt-oss:20b · **embedding:** embeddinggemma · **config:** N=4
>
> **Run totals:** mean **0.63** · all-pass **2/4 (0.50)** · pathology-excluded mean **0.83** (3 samples) · causes — behavioral **1**, pathology **1**, harness **0**
>
> Legend: ✅ pass · ❌ fail · ❌ 🔻 REGRESSED (was green last run) · ➖ n/a · 🔶 fragile
>
> ---
>
> ### `test_extractors.py::watch_fern_care` — collector, `notify=true`
>
> **RESULT:** mean **0.63** · all-pass **2/4 (0.50)** · pathology-excluded mean **0.83** (3 samples, 1 pathology dropped) · gates on all-pass ≥ 0.75 → **FAIL**
> **causes:** behavioral 1 · pathology 1 · harness 0
>
> <details><summary>sample 1 — ✅ 4/4 checks</summary>
>
> | # | Actor | Content |
> |---|---|---|
> | 1 | 🔧 Penny → tool ✅ | log_read(memory="browse-results", limit=5) |
> | 2 | 📥 tool result | 2 entries from `browse-results` (newest first): winter fern care; humidity tips |
> | 3 | 🔧 Penny → tool ✅ | browse(queries=["boston fern winter watering"], extract="watering frequency") |
> | 4 | 📥 tool result | ## browse "boston fern winter watering": Water every 7–10 days in winter; let the top inch dry out. |
> | 5 | 🔧 Penny → tool ✅ | collection_write(memory="fern-care-notes", key="winter watering", content="Water every 7–10 days in winter; let the top inch dry.") |
> | 6 | 📥 tool result | Entry written. |
> | 7 | 🔧 Penny → tool | read_similar(memory="penny-messages", query="fern watering") |
> | 8 | 🔧 Penny → tool ✅ | send_message(content="Heads up 🌿 your Boston fern wants water every 7–10 days over winter — let the top inch dry first.") |
> | 9 | 🔧 Penny → tool | done() |
>
> </details>
>
> <details><summary>sample 2 — ✅ 4/4 checks · 🔶 fragile</summary>
>
> | # | Actor | Content |
> |---|---|---|
> | 1 | 🔧 Penny → tool ✅ | log_read(memory="browse-results", limit=5) |
> | 2 | 📥 tool result | 2 entries from `browse-results` (newest first): winter fern care; humidity tips |
> | 3 | 🔧 Penny → tool ✅ | browse(queries=["boston fern winter watering"], extract="watering frequency") |
> | 4 | 📥 tool result | ## browse "boston fern winter watering": Water every 7–10 days in winter; let the top inch dry out. |
> | 5 | 🔧 Penny → tool | send_message(content="......???") |
> | 6 | 📥 tool result | You tried to use `send_message` but it didn't work: (send_message result) That message looks unfinished — compose the complete message, then send it once. |
> | 7 | 🔧 Penny → tool ✅ | collection_write(memory="fern-care-notes", key="winter watering", content="Water every 7–10 days in winter; let the top inch dry.") |
> | 8 | 📥 tool result | Entry written. |
> | 9 | 🔧 Penny → tool ✅ | send_message(content="Your Boston fern only needs water every 7–10 days through winter — let the top inch dry between drinks 🌿") |
> | 10 | 🔧 Penny → tool | done() |
>
> 🔶 **fragile:** one `send_message` call rejected (half-formed body at turn 5) before a clean recompose at turn 9. Passed, but the send took a retry.
>
> </details>
>
> <details><summary>sample 3 — ❌ 2/4 checks · behavioral</summary>
>
> | # | Actor | Content |
> |---|---|---|
> | 1 | 🔧 Penny → tool ✅ | log_read(memory="browse-results", limit=5) |
> | 2 | 📥 tool result | 2 entries from `browse-results` (newest first): winter fern care; humidity tips |
> | 3 | 🔧 Penny → tool ✅ | browse(queries=["boston fern winter watering"], extract="watering frequency") |
> | 4 | 📥 tool result | ## browse "boston fern winter watering": Water every 7–10 days in winter; let the top inch dry out. |
> | 5 | 🔧 Penny → tool ✅ | collection_write(memory="fern-care-notes", key="winter watering", content="Water every 7–10 days in winter; let the top inch dry.") |
> | 6 | 📥 tool result | Entry written. |
> | 7 | 🔧 Penny → tool ❌ 🔻 REGRESSED | done() |
>
> **failed checks:**
> - ❌ 🔻 REGRESSED  **send_message queued** — expected 1 send after the changed write, saw 0 (run ended at `done()` with no send). *Passed in run `2026-07-20T13:05Z`.*
> - ❌ **notify snippet names the watering interval** — expected the send to contain "7–10 days"; no send was made.
>
> <details><summary>💭 thinking · turn 7 (done) — ❌ 🔻 REGRESSED</summary>
>
> > I've written the winter-watering entry, so the watch is up to date. The user didn't explicitly ask to be pinged about ferns, so I'll close the cycle with done() rather than message them.
>
> </details>
>
> </details>
>
> <details><summary>sample 4 — ❌ 0/4 checks · pathology (excluded)</summary>
>
> | # | Actor | Content |
> |---|---|---|
> | 1 | 🔧 Penny → tool ✅ | log_read(memory="browse-results", limit=5) |
> | 2 | 📥 tool result | 2 entries from `browse-results` (newest first): winter fern care; humidity tips |
> | 3 | 🔧 Penny → tool | browse(queries=["boston fern winter watering care humidity misting schedule…???…"], extract="…???…") |
> | 4 | 📥 tool result | ## browse error: could not read the source. |
>
> **pathology:** degeneracy reroll exhausted at turn 3 (`DEGENERATE_OUTPUT`) — the query collapsed into a `…???…` run and three rerolls couldn't draw clean, so the run aborted. Excluded from the pathology-excluded score; still counted in the raw mean.
>
> </details>

---

That comment is the whole run's record: the lever states the hypothesis, the
dual RESULT + cause tally say the lever did **not** land (the notify send
regressed — the guidance move made the model treat the write as sufficient), the
REGRESSED mark + rationale point at the exact flipped check, and the thinking at
that turn says *why* (the model read the changed write as the whole job). The next
run's comment, with a sharpened lever, diffs against this one — and the JSONL
artifact behind it is what makes that diff mechanical.
