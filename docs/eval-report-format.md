# Eval Run Report Format

The comment-ready markdown one eval run posts to its iteration PR — the **transcript-integrated**
format (iteration-6, #1725). The graded-check mechanics live **inside** the transcript, in causal
order (contract → thinking → action → verdict-on-action), so a reader follows the run as it
happened instead of hunting a separate table. This is the format contract: what every run comment
carries, section by section, so the renderer (`report.py`), the assembler (`assemble.py`), the
extractor (`conftest.py`), and a human reading the PR all share one shape.

Read this alongside:

- **`docs/agent-task-workflow.md` §4** — the *protocol* (a PR is live from the first run; every run
  posts its report as a **new** comment). This document is the *format* those comments take.
- **`penny/tests/eval/report.py`** — the pure renderer of the grammar below (the row types, the
  per-step tables, the banner, folding). Hand-built inputs render identically to extracted ones.
- **`penny/tests/eval/conftest.py`** (`_write_sample_report` + the `_build_transcript` extraction) —
  turns a sample's persisted promptlog into the `report.py` model.
- **`penny/tests/eval/assemble.py`** — composes the run header + per-sample blocks + footer.

## The one rule: the PR comment IS the record; the bulk stays local

A run comment carries the **whole evaluation** — the verdict, every step, the thinking at every
model action — because the comment *is* the durable, inspectable record of the iteration (read as
GitHub markdown, in the PR's comment stream). The heavy raw artifacts never enter the comment and
are **never committed** (see [What gets committed](#what-gets-committed)).

---

## Row grammar (one fixed form per row type)

Every per-step table uses these rows, identically everywhere:

| row type | column-1 label | body (column 2) | score cell (column 3) |
|---|---|---|---|
| **step header** | `step N · 👤` | `"user message"` | the step verdict (✅ / ❌ / ✅→❌) |
| **expected** | `expected` | `Cn [class]marker label` | empty — or the verdict for a no-evidence-row contract |
| **💭** | `💭` | an ALWAYS-collapsed `<details><summary>thinking</summary>…</details>`, one per model action, directly ABOVE it (`💭 (empty)` when the model emitted none) | always empty |
| **actual** | `actual` | one transcript event (`🔧 call` · `📥 result` · `🤖 reply` · `👤 nudge` · `🧩 micro`) | the check verdict on the anchor row; `⚠ recovery event` on a nudge; else empty |
| **baseline** | `baseline` | the prior run's anchor event (diff mode only) | the prior verdict, tagged `*(prior run)*` |
| **note** | `note` | free text, always last | always empty |

**Check identity + class.** Each check renders `Cn [class]marker label` on its `expected` row.
`Cn` is `C1, C2, …` in scorer order; a framework guard is `Gn`. `[class]` is an authoring tag —
`[spine]` (call-spine) · `[reply]` (reply-content) · `[state]` (durable state) · `[proc]`
(procedure) · `[guard]` (framework-injected). `marker` is `⚖` (a **scored** check, counts toward
the score) or `ℹ` (an **advisory** check — renders, never scores); an n/a check carries neither.

**Verdict cell.** `mark [Cn] [— rationale] [· cause]`. The marks:

| mark | meaning |
|---|---|
| `✅` | check passed |
| `❌` | check failed (carries its observed-vs-expected `rationale` + the sample's `cause`) |
| `✅→❌ **REGRESSED**` | failed here, fully green in the baseline run — a flip (diff mode) |
| `❌→✅ **FIXED**` | passed here, failing in the baseline run (diff mode) |
| `➖ n/a` | the check's branch didn't run this sample — out of the graded denominator |
| `⚠ recovery event` | on a nudge row: the loop refused/recovered a call (flags a passing sample `fragile`) |

**Verdict placement (the anchor rule).** A check anchored to a transcript event renders its
`expected` row atop that event's step and its verdict on that event's `actual` row. A check with
**no anchor row** — a whole-run property, or a *missing* expected action that never happened —
falls to the **run-close** table (`| run-close | whole-conversation contracts | k/n |`), where its
verdict sits on its own `expected` row (these have no evidence row of their own). This is the
deterministic realization of "run-close = checks with no anchor row."

## Per-sample banner + folding

Each sample opens with a banner naming its stats before you read a row:

```
#### sample N — <verdict> · <k/n> (<score>) · [fragile ·] [cause ·] <duration>s · <calls> calls
```

A clean pass carries no cause; a passed-but-shaky sample carries `fragile`; a **harness-timeout**
sample (no completed turn) omits `k/n` (the scorer never ran) and renders an honest placeholder body
instead of a table — so the report's sample count always matches N (visible degradation).

**Sample-level folding.** A clean-pass sample folds whole (the entire block inside a `<details>`
whose summary is its banner); a **failed / fragile / regressed** sample renders **unfolded**.
Density follows failure at the sample level, so an N=8 mostly-green report stays one screen while
the step grammar inside stays uniform.

## Micro-context (🧩) — an official actor

A browse call carrying an `extract` micro-instruction spawns a single-shot extraction sub-model
(`browse-extract`). Its exchange renders inline, in ledger order, as two `actual` rows — the
instruction + page content INTO the sub-model (`🧩 micro-context ← …`) and its extracted value OUT
(`🧩 micro-context → EXTRACTED: …`) — with the sub-model's own `💭 (micro-context)` above the OUT
row. A multi-page `extract` browse renders one pair per page. The main-loop context never sees the
page body; only the typed value returns — the report is the one place that exchange is visible.

## Run header

The comment opens with the run header (no per-sample transcript above it):

- **identity** — `**<run-id>** · commit \`<sha>\` [(dirty)] · <model> · N=<n> · **lever:** <lever>`.
  The **lever** is required (the run's hypothesis; an unlabelled run can't attribute a score shift).
- **RESULT** — one line: `mean · all-pass · pathology-excluded · causes — behavioral B · pathology P
  · harness H · families: <fam> <mean> [(k cases)] · … · <calls> calls · <s>s · <in>K in / <out>K out`.
  *mean* is the partial-credit mean the case gates on; *all-pass* the strict count of perfect
  samples; *pathology-excluded* the honest mean over every non-pathology sample; the *families*
  rollup names each family's mean (a case count only when a family spans more than one case).
- **gate** — one line per gated case: `**gate:** [<case>: ]⚖ <threshold> on <metric> → **PASS/FAIL**
  (<value>)`, where `<metric>` is `mean` or `pathology-excluded` (the honest-threshold opt-in, #1698).
  A report-only case (`min_pass_rate=None`) has no gate line.
- **flips** (diff mode) — `flips: <label> ✅→❌ (s1, s3) · …`, one entry per check that was fully
  green in the baseline but failed a sample here. This is the one cross-sample join the transcript
  flow can't give you; it joins on `(case_id, label)`.

Then one section per case — a `### \`<case_id>\` — <family>` heading **only when the run spans
multiple cases** — above that case's per-sample blocks. A single-case run needs no divider.

## Diff mode (when `EVAL_BASELINE` is set)

Same grammar plus the `baseline` row (the prior run's anchor event, its cell the prior verdict) and
the flip badges (`✅→❌ REGRESSED` / `❌→✅ FIXED`) in place of the plain glyph; the step header shows
the step-level flip, and the run header gains the `flips:` index. Off-diff (no baseline, or a first
run) there are no baseline rows, no flip badges, and no flips line — no error.

## Deterministic cell hygiene

- **Escaping.** Every cell escapes `|` (→ `\|`) and renders newlines as `<br>`, so a tool call or a
  multi-line result stays inside its cell.
- **Truncation.** An `actual` cell over ~500 chars renders its head + `…` with the full escaped text
  in a nested collapsed `<details>`; a browse page body renders as its fetch handle + first line.
  One rule, applied by the renderer — never ad-hoc.

## Footer

The n≤1 pointer from the comment back to the raw evidence:

```
_artifacts (local, never committed): `<report dir>` · per-sample DBs beside them · re-render: `EVAL_REPORT_DIR=<report dir> make assemble`_
```

## What gets committed

**Nothing.** Run artifacts are **never committed to the repo** — baselines included. The durable
record of every run is its **assembled report, posted as a PR comment** (the iteration protocol,
#1711). ALL raw artifacts — `manifest.json`, `results.jsonl`, the per-case `<case_id>.md`
transcripts, the per-sample `<case>-<n>.db` files, and `dirty.diff` — live **locally on the eval
host** under the `data/` tree, and **`EVAL_BASELINE` diffs against those local paths**. The report
footer points at that local directory for audit. There is no committed-baseline tier.

## Check-label stability (an authoring note)

The diff joins on `(case_id, label)`, so **renaming a check silently breaks its regression
continuity** — the flip/REGRESSED machinery can no longer match it against the baseline. Relabeling
a check is therefore a recorded scorer-semantics change, not a cosmetic edit.

---

## Worked example

One complete run comment, rendered by the assembler — a chat-browse case at N=3 (a clean pass folded,
plus a harness-timeout sample the format makes visible). Entirely synthetic, shown in a fenced block
so the tables and `<details>` read as source. This is the verbatim markdown a run posts as a comment.

````markdown
**run-20260721T051017Z-abba710a** · commit `abba710a` · gpt-oss:20b · N=3 · **lever:** switch the representative case to chat-browse (prior case outmoded)
**RESULT:** mean 0.67 · all-pass 2/3 · pathology-excluded 0.67 · causes — behavioral 0 · pathology 0 · harness 1 · families: browse-answer 0.67 · 19 calls · 148s · 54.2K in / 5.9K out
**gate:** ⚖ 0.75 on mean → **❌ FAIL** (0.67)

<details><summary>sample 1 — ✅ pass · 2/2 (1.00) · 41s · 6 calls</summary>

| step 1 · 👤 | "what's the deepest lake in the world?" | ✅ |
|---|---|---|
| expected | C1 [spine]⚖ browsed for a current-info question |  |
| expected | C2 [reply]⚖ reply surfaces the browsed fact |  |
| 💭 | <details><summary>thinking</summary>User wants the deepest lake. Verify with a source rather than answer from memory.</details> |  |
| actual | 🔧 browse({"queries":["wiki/Lake_Baikal"],"extract":"maximum depth"}) | ✅ C1 |
| actual | 🧩 micro-context ← Instruction: maximum depth · Content: Lake Baikal is the deepest lake at 1,642 metres. |  |
| 💭 | <details><summary>thinking (micro-context)</summary>The content states 1,642 metres. Extract that value.</details> |  |
| actual | 🧩 micro-context → EXTRACTED: 1642 |  |
| actual | 📥 You opened wiki/Lake_Baikal (browse result) · 1642 |  |
| 💭 | <details><summary>thinking</summary>Answer with the fact and the source.</details> |  |
| actual | 🤖 Lake Baikal is the deepest, at 1,642 m. 🌊 | ✅ C2 |

</details>

#### sample 3 — ❌ fail · harness · 120s · 13 calls

_(no completed turns recorded — the sample produced no finished model call, e.g. a harness timeout)_

_artifacts (local, never committed): `/penny/data/eval-reports/run-20260721T051017Z` · per-sample DBs beside them · re-render: `EVAL_REPORT_DIR=/penny/data/eval-reports/run-20260721T051017Z make assemble`_
````

Sample 2 (a second clean pass) folds the same way and is omitted here. The run reads top-down: the
gate line says the lever did **not** clear the bar (the timeout sample dragged the mean under 0.75),
and the F2 placeholder makes the timeout *visible* — its per-sample DB is one local hop away for the
full parse-failure trace, per the footer.

---

## Field glossary (names shared across the artifact + renderer)

| field | source | meaning |
|---|---|---|
| `lever` | manifest | required one-line hypothesis for the run |
| `commit` / `dirty` | manifest | branch commit + clean/dirty flag |
| `model` / `samples` | manifest | the run's model + N |
| `case_id` / `family` | artifact | `<file>::<case>` identifier + family tag |
| `mean` / `all_pass_rate` | artifact | partial-credit mean + strict all-pass fraction |
| `pathology_excluded_mean` | artifact | mean over every NON-pathology sample (the honest read) |
| `sample_scores` / `sample_causes` / `sample_fragile` | artifact | per-sample score · cause · fragile flag |
| `cause_counts` | artifact | failed-sample tally `{behavioral, pathology, harness}` |
| `checks[]` | artifact | per-check `CheckOutcome`: `label · passed · total · scored · cells[] · rationales[]` |
| `min_pass_rate` / `gate_metric` | artifact (#1725) | the gate threshold + which score it compares (`mean` \| `pathology-excluded`) |

## Decided ambiguities (resolved here)

Points the reference grammar left open, decided for this implementation:

1. **Unanchored checks go to run-close** (not into a step) — the deterministic anchor rule. A
   check whose expected action never happened has no evidence row, so it renders in run-close.
2. **The `families:` rollup omits the case count for a single-case family** (`browse-answer 0.67`)
   and shows it only when a family spans more than one case (`recall 1.00 (2 cases)`).
3. **A per-case heading renders only in a multi-case run** — a single-case run's samples follow the
   run header directly.
4. **`FIXED` flips are not computed by the extractor** (the baseline record carries "was fully
   green", not "was fully red"); the renderer supports the badge, but the flips index surfaces
   **regressions** only.
5. **Thinking renders for every model action** (not only failed turns, superseding the earlier
   capture) — an empty thought before a degenerate act is itself signal (`💭 (empty)`).
