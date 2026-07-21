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
- run totals at top: the dual metrics (mean-of-scores + all-pass), the failure-cause
  tally, and the per-family rollup
- per case: the dual RESULT line **with timings** (calls · wall · tokens)
- per case: the **check summary table** — one row per check, a column per sample
  (✅ / ❌ / ➖ n-a / ❌🔻 REGRESSED / blank when absent), advisory checks marked,
  and a `pass` column of passed/present — rendered **outside** the folded transcript
- per case: the **miss-rationale** legend — one bullet per failed check, keyed by its
  table number, naming the failing samples and the observed-vs-expected rationale
- per case: the **per-sample index** — one line per sample (verdict · check fraction +
  score · fragile · cause), so a reader triages before unfolding
- the per-sample turn-by-turn transcript table, folded into a collapsed `<details>`
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

A run comment is: **one manifest header**, **run totals**, then **one block per
case**. Each case block leads with its RESULT line, then surfaces the graded
mechanics **outside** any fold — a **check summary table**, a **miss-rationale**
legend, and a **per-sample index** — and finally folds the turn-by-turn
transcripts into a collapsed `<details>` (one `#### sample` block inside per
sample). This is the v2 layout (#1725): the whole point is that the graded
mechanics (rationales, n-a, advisory, fragile, cause, REGRESSED) are visible at a
glance, not hidden until you unfold. The sections below specify each part; the
[worked example](#worked-example) at the end renders all of them with synthetic
content.

The assembler (`penny/tests/eval/assemble.py`) composes this from the per-run
artifacts alone — `manifest.json`, `results.jsonl` (the per-case `CaseArtifact`
records, whose `CheckOutcome`s now carry per-sample `cells`, `scored`, and
`rationales`, and whose `sample_fragile` list carries the per-sample fragile
flag), and each `<case_id>.md` transcript — reading the baseline (`EVAL_BASELINE`)
for the REGRESSED cells. The per-sample transcript blocks are still written
incrementally by `conftest.py`'s `_write_sample_report` as each sample runs.

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

Immediately under the manifest: the run-level aggregate across all cases — the
dual metrics line (mean-of-scores + all-pass), the failure-cause tally, and a
compact **family rollup**. One glance says whether the lever helped.

The **family rollup** is a single `families:` line naming each family the run
spans with its mean and case count — `families: extractors 0.69 (1 case) ·
chat_response 1.00 (1 case)` — so a multi-family run's score isn't a single blur.
A family's mean is the mean over that family's samples.

### 3. Per-case verdict — the dual RESULT line + timings

Each case leads with a **dual RESULT line** (#1694's strict+partial), then the
run's **timings**:

- **mean** — mean of the per-sample scores (graded partial credit averaged).
- **all-pass** — the fraction of samples that scored a perfect 1.0 (the strict
  gate — the old binary pass-rate).
- **timings** — `<calls> calls · <s>s · <in>K in / <out>K out`, so the cost of
  the case reads beside its score (omitted when the case logged no model call).

Both metrics, always — a case can carry a healthy mean while few samples are
*fully* clean, and the gap is signal. The per-case failure-cause tally is no
longer repeated on its own line under RESULT (v2): the cause lives in the run
totals (aggregate) and on each sample's line in the per-sample index below.

### 3a. Check summary table (the v2 centerpiece)

Rendered **outside** the folded transcript, so the graded mechanics are visible
without unfolding. One row per check, numbered (the number keys the miss legend),
a column per sample, and a `pass` column of passed/present:

```
| # | check | s1 | s2 | s3 | s4 | pass |
|---|---|---|---|---|---|---|
| 1 | read browse-results | ✅ | ✅ | ✅ | ✅ | 4/4 |
| 3 | write queued | ✅ | ✅ | ❌🔻 | ❌🔻 | 2/4 |
```

Cell glyphs: `✅` pass · `❌` fail · `❌🔻` a REGRESSED failure (failed here, fully
green in the baseline run) · `➖` not-applicable (the check's branch didn't run
this sample) · blank when the scorer emitted no such check for that sample. An
**advisory** check (`scored=False`) is marked in its label (`… _(advisory)_`) and
stays out of every per-sample score. The table renders only for a graded case; a
binary case shows its transcript alone.

### 3b. Miss rationales

Under the table, one bullet per check that failed any sample, **keyed by its
table number**, naming the failing samples and the observed-vs-expected rationale:

```
- (3) write queued — *run ended at done() with no send* (s3, s4) · 🔻 regressed from `run-…`
```

`(all)` replaces the sample list when every sample failed the check; a check that
flipped from a fully-green baseline names the prior run it regressed from. This
is also the index the thinking blocks want (#1725 point 4): a failed check's
bullet names exactly which sample(s) to unfold.

### 3c. Per-sample index

Above the folded transcript, one line per sample — verdict, check fraction +
score, fragile flag, cause — so a reader triages before unfolding:

```
- s2 — ✅ pass · 4/4 (1.00) · fragile
- s4 — ❌ fail · 1/4 (0.25) · pathology (excluded)
```

The check fraction (`k/m`) is reconstructed from the per-sample cells the same way
`SampleResult.graded` scores (advisory + not-applicable excluded); a binary sample
shows only its score. The sample index **always names every sample** — including a
harness-timeout sample that produced no completed turn — so the report's sample
count always matches N (visible degradation, #1725/F2). That timeout sample's
folded transcript block is an honest placeholder rather than a silently-omitted
sample.

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

The whole case's transcripts fold into **one** collapsed `<details>` (v2), each
sample a `#### sample N — <verdict>` heading inside it. The verdict line names the
sample, its check fraction, its cause (a failed sample), and any fragile flag; the
per-sample index (§3c) is the un-folded triage view of the same verdicts. Inside
each sample block: the turn-by-turn transcript — the base shape the harness already
emits (`_sample_turns` / `_place_checks`), read verbatim from the sample's promptlog.
A sample that produced no completed turn (a harness timeout) renders an honest
placeholder line instead of a table, so no sample is silently omitted (§3c, #1725/F2).

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

The sample verdict is `✅ PASS` / `❌ FAIL` for a binary sample, or `✅ N/M checks`
for a graded one (M = the **scored** checks; advisory/flavour checks render but
don't count, and n/a checks are out of M entirely). A **failed** sample carries its
cause tag (`· behavioral` / `· harness` / `· pathology (excluded)`, #1725) and a
passed-but-shaky sample a `· fragile` tail.

### 6. Mark legend

Cell / stamp glyphs (the check table uses the same vocabulary in each per-sample cell):

| mark | meaning |
|---|---|
| ✅ | check passed |
| ❌ | check failed |
| ❌🔻 (table cell) / ❌ 🔻 REGRESSED (transcript stamp) | failed **and** passed in the prior run — a flip, distinct from a check that was already red (#1693) |
| ➖ n/a | the check's branch didn't run; excluded from the graded denominator (#1694's ignore state) |
| blank cell | the scorer emitted no such check for that sample (the check table only) |
| `· fragile` | the sample **passed**, but with rejected calls / retries / recovery events — green, but not clean (#1694) |

A **REGRESSED** mark only exists when a baseline is present (`EVAL_BASELINE`); on a
first run there's nothing to flip against, so failures are plain ❌.

A **fragile** flag rides the sample verdict (a `· fragile` tail) and the per-sample
index line, not a single check — it says "this green came with a wobble". The
transcript names what wobbled (which call was rejected/retried/recovered) in the
recovery-frame row under its table.

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

### 9. How the baseline is supplied, and where the flip is named (#1693)

The prior run the report diffs against is named by **`EVAL_BASELINE`** — a path to a
prior run's **report directory** (its `results.jsonl` is used) or that `results.jsonl`
directly. The Makefile forwards it into the eval container exactly like
`EVAL_REPORT_DIR` / `EVAL_LEVER`. Unset (or a path with no `results.jsonl` — a first
run) → no baseline, no REGRESSED marks, no error.

The diff is **per-check, per-case** against the prior run's `CaseArtifact` records
(#1692, consumed unchanged). A now-failing check is **REGRESSED** only when it was
**fully green** in the prior run — passed in *every* prior sample (`passed == total`);
a check that was already flaky (`2/4`) is not a flip. The comparison is by
`(case_id, label)`.

Implementation deltas from the sketch above:

- **The prior run id is surfaced inline on the regressed check**, not (yet) as a
  manifest-header `prior:` line — #1692's `render_manifest_header` / `RunManifest`
  carry no baseline reference, so a cross-region edit to add one is deferred. Each
  regressed check names its baseline run in the checks legend:
  `❌ 🔻 REGRESSED  <label> — <rationale> (was passing in \`<run-id>\`)`.
- **Thinking renders at failed/regressed *tool-call* turns.** A tool-call turn maps
  cleanly to the promptlog row whose response emitted it (and carries its thinking); a
  reply-anchored or whole-run (footer) failed check has no single producing turn, so it
  renders its rationale in the legend but no thinking block.

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
| `sample_fragile[]` | #1725 | per-sample fragile flag aligned with `sample_scores` (the assembler's per-sample index reads it) |
| `checks[]` | #1694/#1725 | per-check `CheckOutcome`: `label` · `passed` · `total` · `scored` · `cells[]` (per-sample `passed`/`failed`/`na`/`absent`) · `rationales[]` |
| `regressed` | #1693 | check flipped ok→fail vs the prior run's artifact (a ❌🔻 cell + a regressed-from note) |
| `fragile` | #1694 | sample passed with rejected calls / retries / recoveries |

---

## Worked example

One complete run comment, rendered by the assembler. Entirely synthetic content — a
houseplant-care collector, in the flavour of the repo's real eval fixtures (board
games, houseplants, espresso). This is the **verbatim markdown the renderer emits**
(shown in a fenced block so the tables and `<details>` read as source, not rendered);
it is what gets posted as a run comment on the iteration PR. The run diffs against a
prior run supplied via `EVAL_BASELINE`, so the `write queued` flip renders as a
`❌🔻` cell and a `🔻 regressed from …` note.

````markdown
### run-20260720T143200Z-a1b2c3d4

- commit: `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0`
- model: `gpt-oss:20b`
- N: 4
- lever: moved the notify-suffix guidance out of the water-my-ferns skill into the collector prompt

## Run totals

mean 0.69 · all-pass 2/4
pathology-excluded mean 0.83 (3 samples) · causes — behavioral 1 · pathology 1 · harness 0
families: extractors 0.69 (1 case)

### `test_extractors.py::watch_fern_care` — extractors

**RESULT:** mean 0.69 · all-pass 2/4 · 34 calls · 452s · 118.4K in / 9.3K out

**Checks**

| # | check | s1 | s2 | s3 | s4 | pass |
|---|---|---|---|---|---|---|
| 1 | read browse-results | ✅ | ✅ | ✅ | ✅ | 4/4 |
| 2 | browse called | ✅ | ✅ | ✅ | ❌ | 3/4 |
| 3 | write queued | ✅ | ✅ | ❌🔻 | ❌🔻 | 2/4 |
| 4 | notify names the interval | ✅ | ✅ | ❌ | ❌ | 2/4 |
| 5 | read_similar called _(advisory)_ | ✅ | ✅ | ✅ | ❌ | 3/4 |

**Miss rationales**
- (2) browse called — *the query collapsed into a degenerate run* (s4)
- (3) write queued — *run ended at done() with no send* (s3, s4) · 🔻 regressed from `run-20260720T130500Z-cafe0000`
- (4) notify names the interval — *no send was made, so the interval never appeared* (s3, s4)
- (5) read_similar called (s4)

**Samples**
- s1 — ✅ pass · 4/4 (1.00)
- s2 — ✅ pass · 4/4 (1.00) · fragile
- s3 — ❌ fail · 2/4 (0.50) · behavioral
- s4 — ❌ fail · 1/4 (0.25) · pathology (excluded)

<details><summary>transcripts — test_extractors.py::watch_fern_care</summary>

#### sample 1 — ✅ 4/4 checks

| # | Actor | Content |
|---|---|---|
| 1 | 🔧 Penny → tool ✅ | log_read(memory="browse-results", limit=5) |
| 2 | 📥 tool result | 2 entries from `browse-results` (newest first): winter fern care; humidity tips |
| 3 | 🔧 Penny → tool ✅ | browse(queries=["boston fern winter watering"], extract="watering frequency") |
| 4 | 📥 tool result | ## browse "boston fern winter watering": Water every 7-10 days in winter. |
| 5 | 🔧 Penny → tool ✅ | collection_write(memory="fern-care-notes", key="winter watering", content="Water every 7-10 days.") |
| 6 | 📥 tool result | Entry written. |
| 7 | 🔧 Penny → tool ✅ | send_message(content="Heads up 🌿 your Boston fern wants water every 7-10 days over winter.") |
| 8 | 🔧 Penny → tool | done() |

#### sample 3 — ❌ 2/4 checks · behavioral

| # | Actor | Content |
|---|---|---|
| 1 | 🔧 Penny → tool ✅ | log_read(memory="browse-results", limit=5) |
| 2 | 🔧 Penny → tool ✅ | browse(queries=["boston fern winter watering"], extract="watering frequency") |
| 3 | 🔧 Penny → tool ❌ 🔻 REGRESSED | done() |

_checks: ❌ 🔻 REGRESSED write queued — run ended at done() with no send (was passing in `run-20260720T130500Z-cafe0000`)_

<details><summary>💭 thinking · turn 3 (done) — ❌ 🔻 REGRESSED</summary>

> The entry is already written, so I'll close with done() rather than message them.

</details>

</details>
````

That comment is the whole run's record. The **check table** shows the failure is
clustered — every sample read the browse-results log (row 1) while the notify
`write queued` step (row 3) failed on s3 and s4, and the `❌🔻` cells + the miss
legend's `🔻 regressed from …` note say the lever *regressed* it (the guidance move
made the model treat the write as sufficient). The **per-sample index** triages at a
glance — a clean pass, a fragile pass, a behavioral miss, and a pathology sample
(excluded from the honest mean) — and the folded transcript's thinking at the flipped
`done()` turn says *why*. The next run's comment, with a sharpened lever, diffs
against this one — and the `results.jsonl` artifact behind it (whose `CheckOutcome`
cells the table renders) is what makes that diff mechanical.
