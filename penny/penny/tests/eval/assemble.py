"""Run-comment assembler (#1717): compose a completed run's artifacts into THE
postable PR comment.

The per-run artifacts and per-case report blocks all exist after a ``make eval``
run — ``manifest.json`` + ``results.jsonl`` (``artifacts.py``) and one
``<case_id>.md`` transcript per case (``conftest.py``'s ``_write_sample_report``)
— but no step composes them into the ONE markdown document the protocol
(``docs/eval-report-format.md``) specifies. This module is that step.

Given a completed run's report directory it emits one markdown comment, in the
format-spec's section order (v2, #1725):

  1. the manifest header (commit · model · N · lever), via ``render_manifest_header``.
  2. the run totals — the run-level aggregate across every case (mean-of-scores,
     all-pass rate, the failure-cause tally, and the per-family rollup), computed
     by flattening every case's per-sample scores/causes.
  3. one block per case: the RESULT line (mean + all-pass + timings), the **check
     summary table** (one row per check, a column per sample, rendered from the
     ``CaseArtifact``'s per-check cells so the graded mechanics are visible without
     unfolding), the **miss-rationale** legend, the per-**sample** index (verdict ·
     score · cause · fragile), then the case's transcript folded into a ``<details>``.

The check table's REGRESSED cells and the miss legend's regressed-from note read the
baseline (``EVAL_BASELINE`` via ``baseline_from_env``), the same flip source the
per-sample transcript marks (#1693). Pure artifact consumption: no model, no git, no
network. Per-case metrics derive from each ``CaseArtifact``'s ``sample_scores`` /
``sample_causes`` / ``checks`` so a case's line and the run totals compute the same way
(a passed sample carries no cause, so the all-pass count is the count of ``None`` causes).

Run it via ``python -m penny.tests.eval.assemble <report_dir>`` (the assembled
comment is written to stdout).
"""

from __future__ import annotations

import sys
from pathlib import Path

from penny.tests.eval.artifacts import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    CaseArtifact,
    CheckCell,
    CheckOutcome,
    FailureCause,
    RunManifest,
    count_causes,
    pathology_excluded,
    render_cause_summary,
    render_manifest_header,
)
from penny.tests.eval.baseline import Baseline, baseline_from_env

# ── Section literals (no magic strings) ──────────────────────────────────────
RUN_TOTALS_HEADING = "## Run totals"
CHECKS_HEADING = "**Checks**"
MISS_HEADING = "**Miss rationales**"
SAMPLES_HEADING = "**Samples**"
NO_TRANSCRIPT = "_(no transcript recorded)_"
ADVISORY_MARK = " _(advisory)_"
SECTION_SEPARATOR = "\n\n"

# Per-sample cell glyphs in the check summary table (a REGRESSED fail carries 🔻).
_CELL_MARK = {
    CheckCell.PASSED: "✅",
    CheckCell.FAILED: "❌",
    CheckCell.NA: "➖",
    CheckCell.ABSENT: " ",
}
_REGRESSED_CELL = "❌🔻"

USAGE = "usage: python -m penny.tests.eval.assemble <report_dir>"


def assemble_run_comment(report_dir: Path) -> str:
    """Compose the run's whole PR comment from its report directory (the summary method).

    Reads ``manifest.json`` + ``results.jsonl`` + each ``<case_id>.md`` and renders the
    manifest header, the run totals, and one folded block per case, in spec order."""
    manifest = load_manifest(report_dir)
    artifacts = load_case_artifacts(report_dir)
    baseline = baseline_from_env()  # a prior run's results.jsonl → REGRESSED cells (#1693/#1725)
    sections = [
        render_manifest_header(manifest).rstrip("\n"),
        render_run_totals(artifacts),
    ]
    sections += [
        render_case_block(report_dir, manifest, artifact, baseline) for artifact in artifacts
    ]
    return SECTION_SEPARATOR.join(sections) + "\n"


# ── Artifact loading (the manifest is required; results/transcripts tolerate absence) ──
def load_manifest(report_dir: Path) -> RunManifest:
    """Read the run's ``manifest.json``, or fail with an actionable message if it's absent."""
    path = report_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} in {report_dir} — is this a completed eval run's report "
            f"directory? Run `EVAL_REPORT_DIR={report_dir} … make eval` first."
        )
    return RunManifest.model_validate_json(path.read_text())


def load_case_artifacts(report_dir: Path) -> list[CaseArtifact]:
    """Read every case record from ``results.jsonl`` (one per non-blank line), in file order.

    A missing/empty file → no cases: a manifest can exist before any case has recorded."""
    path = report_dir / RESULTS_FILENAME
    if not path.is_file():
        return []
    return [
        CaseArtifact.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ── Rendering ────────────────────────────────────────────────────────────────
def render_run_totals(artifacts: list[CaseArtifact]) -> str:
    """The run-level aggregate across every case — the dual metrics line, the cause tally, and
    (when the run spans cases) the per-family rollup, each on its own skimmable line."""
    scores, causes = _flatten(artifacts)
    result_body, cause_line = _result_metrics(scores, causes)
    lines = [RUN_TOTALS_HEADING, "", result_body, cause_line]
    families = _family_rollup_line(artifacts)
    if families:
        lines.append(families)
    return "\n".join(lines)


def _family_rollup_line(artifacts: list[CaseArtifact]) -> str:
    """The compact per-family rollup — ``families: <fam> <mean> (<k> case[s]) · …`` — over the
    families the run spans (each family's mean is the mean of its samples). Empty for no cases."""
    groups: dict[str, list[CaseArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(artifact.family, []).append(artifact)
    if not groups:
        return ""
    parts = []
    for family, group in groups.items():
        scores, _causes = _flatten(group)
        mean = sum(scores) / len(scores) if scores else 0.0
        cases = len(group)
        parts.append(f"{family} {mean:.2f} ({cases} {'case' if cases == 1 else 'cases'})")
    return f"families: {' · '.join(parts)}"


def render_case_block(
    report_dir: Path, manifest: RunManifest, artifact: CaseArtifact, baseline: Baseline | None
) -> str:
    """One case's block (the summary method): heading, the RESULT line with timings, the check
    summary table, the miss-rationale legend, the per-sample index, then the folded transcript."""
    blocks = [
        render_check_table(artifact, baseline),
        render_miss_rationales(artifact, baseline),
        render_samples(artifact),
    ]
    lines = [f"### `{artifact.case_id}` — {artifact.family}", "", _case_result_line(artifact)]
    for block in blocks:
        if block:
            lines += ["", block]
    lines += [
        "",
        f"<details><summary>transcripts — {artifact.case_id}</summary>",
        "",
        _transcript_block(report_dir, manifest, artifact.case_id),
        "",
        "</details>",
    ]
    return "\n".join(lines)


def _case_result_line(artifact: CaseArtifact) -> str:
    """The per-case RESULT line: ``mean … · all-pass k/n`` then the run's timings (calls · wall ·
    tokens), so the cost of the case reads beside its score. All-pass = the count of ``None``
    causes."""
    total = len(artifact.sample_scores)
    mean = sum(artifact.sample_scores) / total if total else 0.0
    all_pass = sum(1 for cause in artifact.sample_causes if cause is None)
    return f"**RESULT:** mean {mean:.2f} · all-pass {all_pass}/{total}{_timings_suffix(artifact)}"


def _timings_suffix(artifact: CaseArtifact) -> str:
    """The ``· <calls> calls · <s>s · <in>K in / <out>K out`` tail on the RESULT line (empty when
    the case logged no model call)."""
    timings = artifact.timings
    if not timings.calls:
        return ""
    seconds = timings.duration_ms / 1000
    return (
        f" · {timings.calls} calls · {seconds:.0f}s · "
        f"{timings.input_tokens / 1000:.1f}K in / {timings.output_tokens / 1000:.1f}K out"
    )


def render_check_table(artifact: CaseArtifact, baseline: Baseline | None) -> str:
    """The per-case check summary table (#1725): one row per check (numbered, keying the miss
    legend), a column per sample (pass / fail / n-a / regressed / blank when the check was absent
    that sample), advisory checks marked, and a ``pass`` column of passed/present. Empty when the
    case has no graded checks (a binary case renders its transcript alone)."""
    if not artifact.checks:
        return ""
    samples = artifact.samples
    headers = ["#", "check", *[f"s{i + 1}" for i in range(samples)], "pass"]
    rows = [
        _check_row(index, outcome, artifact.case_id, baseline, samples)
        for index, outcome in enumerate(artifact.checks, start=1)
    ]
    return "\n".join(
        [
            CHECKS_HEADING,
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "---|" * len(headers),
            *rows,
        ]
    )


def _check_row(
    index: int, outcome: CheckOutcome, case_id: str, baseline: Baseline | None, samples: int
) -> str:
    """One check's table row: its number, label (advisory-marked), the per-sample cells, and the
    ``passed/present`` pass column. A regressed check renders its failing cells as ❌🔻."""
    regressed = baseline is not None and baseline.was_passing(case_id, outcome.label)
    cells = [_cell_mark(_cell_at(outcome, i), regressed=regressed) for i in range(samples)]
    label = outcome.label + ("" if outcome.scored else ADVISORY_MARK)
    return f"| {index} | {label} | " + " | ".join(cells) + f" | {outcome.passed}/{outcome.total} |"


def _cell_mark(cell: CheckCell, *, regressed: bool) -> str:
    """The glyph for one sample's cell — ❌🔻 for a regressed failure, else the plain mark."""
    if cell == CheckCell.FAILED and regressed:
        return _REGRESSED_CELL
    return _CELL_MARK[cell]


def _cell_at(outcome: CheckOutcome, index: int) -> CheckCell:
    """The check's cell for sample ``index`` — ``absent`` when the scorer emitted no such check
    that sample (a shorter/empty cells list, e.g. an artifact built before per-sample cells)."""
    return outcome.cells[index] if index < len(outcome.cells) else CheckCell.ABSENT


def render_miss_rationales(artifact: CaseArtifact, baseline: Baseline | None) -> str:
    """The miss legend under the table: one bullet per check that failed any sample, keyed by its
    table number, naming the failing samples and its observed-vs-expected rationale (#1725). A
    regressed check names the prior run it flipped from. Empty when nothing missed."""
    lines = [
        _miss_line(index, outcome, artifact, baseline)
        for index, outcome in enumerate(artifact.checks, start=1)
        if any(_cell_at(outcome, i) == CheckCell.FAILED for i in range(artifact.samples))
    ]
    return "\n".join([MISS_HEADING, *lines]) if lines else ""


def _miss_line(
    index: int, outcome: CheckOutcome, artifact: CaseArtifact, baseline: Baseline | None
) -> str:
    """One miss bullet: ``- (n) <label> — *<rationale>* (<samples>)`` plus a regressed-from note
    when the check flipped from a fully-green baseline."""
    samples = artifact.samples
    failed = [i for i in range(samples) if _cell_at(outcome, i) == CheckCell.FAILED]
    where = "all" if len(failed) == samples else ", ".join(f"s{i + 1}" for i in failed)
    rationale = f" — *{'; '.join(outcome.rationales)}*" if outcome.rationales else ""
    note = ""
    if baseline is not None and baseline.was_passing(artifact.case_id, outcome.label):
        prior = baseline.run_id_for(artifact.case_id)
        note = f" · 🔻 regressed from `{prior}`" if prior else " · 🔻 regressed"
    return f"- ({index}) {outcome.label}{rationale} ({where}){note}"


def render_samples(artifact: CaseArtifact) -> str:
    """The per-sample index above the folded transcript: one bullet per sample with its verdict,
    check fraction + score, fragile flag, and cause — so a reader triages before unfolding (#1725).
    Always present (it names every sample, so the report's sample count matches N)."""
    total = len(artifact.sample_scores)
    if not total:
        return ""
    return "\n".join([SAMPLES_HEADING, *[_sample_line(artifact, i) for i in range(total)]])


def _sample_line(artifact: CaseArtifact, index: int) -> str:
    """``- sN — <verdict> · <k/m> (<score>) · [fragile ·] <cause>`` for one sample, from the
    artifact alone (the check fraction reconstructed from the per-sample cells)."""
    score = artifact.sample_scores[index]
    parts = [f"s{index + 1} — {'✅ pass' if score >= 1.0 else '❌ fail'}"]
    fraction = _sample_checks(artifact, index)
    parts.append(f"{fraction[0]}/{fraction[1]} ({score:.2f})" if fraction else f"{score:.2f}")
    if index < len(artifact.sample_fragile) and artifact.sample_fragile[index]:
        parts.append("fragile")
    cause = artifact.sample_causes[index]
    if cause is not None:
        parts.append("pathology (excluded)" if cause == FailureCause.PATHOLOGY else cause.value)
    return "- " + " · ".join(parts)


def _sample_checks(artifact: CaseArtifact, index: int) -> tuple[int, int] | None:
    """This sample's ``(passed, scored)`` check counts, reconstructed from the per-sample cells the
    same way ``SampleResult.graded`` scores: not-applicable/absent cells drop, then the scored ones
    (or all applicable when none are scored). ``None`` when the sample exercised no check."""
    applicable = [
        outcome
        for outcome in artifact.checks
        if _cell_at(outcome, index) in (CheckCell.PASSED, CheckCell.FAILED)
    ]
    if not applicable:
        return None
    scored = [outcome for outcome in applicable if outcome.scored] or applicable
    passed = sum(1 for outcome in scored if _cell_at(outcome, index) == CheckCell.PASSED)
    return passed, len(scored)


def _result_metrics(scores: list[float], causes: list[FailureCause | None]) -> tuple[str, str]:
    """The dual RESULT body (``mean … · all-pass k/n``) and the cause summary line, from one
    set of per-sample scores + causes. All-pass = the count of ``None`` causes (a passed
    sample carries none), so run totals and a per-case line are computed identically."""
    total = len(scores)
    mean = sum(scores) / total if total else 0.0
    all_pass = sum(1 for cause in causes if cause is None)
    excluded_mean, kept = pathology_excluded(scores, causes)
    cause_line = render_cause_summary(count_causes(causes), excluded_mean, kept)
    return f"mean {mean:.2f} · all-pass {all_pass}/{total}", cause_line


def _flatten(artifacts: list[CaseArtifact]) -> tuple[list[float], list[FailureCause | None]]:
    """Every case's per-sample scores and causes concatenated — the run-totals denominator."""
    scores: list[float] = []
    causes: list[FailureCause | None] = []
    for artifact in artifacts:
        scores.extend(artifact.sample_scores)
        causes.extend(artifact.sample_causes)
    return scores, causes


def _transcript_block(report_dir: Path, manifest: RunManifest, case_id: str) -> str:
    """The case's ``<case_id>.md`` transcript with its leading manifest header stripped (the
    assembler renders that header once, atop the whole comment). A missing/empty transcript
    renders an honest placeholder rather than an empty ``<details>``."""
    path = report_dir / f"{case_id}.md"
    if not path.is_file():
        return NO_TRANSCRIPT
    text = path.read_text()
    header = render_manifest_header(manifest) + "\n"  # exactly what write_case_header stamped
    if text.startswith(header):
        text = text[len(header) :]
    return text.strip() or NO_TRANSCRIPT


# ── CLI: python -m penny.tests.eval.assemble <report_dir> ─────────────────────
def main(argv: list[str]) -> int:
    """Write the assembled comment for ``argv[0]`` (a report dir) to stdout; 1 on a bad dir."""
    if len(argv) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        comment = assemble_run_comment(Path(argv[0]))
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    sys.stdout.write(comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
