"""Run-comment assembler (#1717): compose a completed run's artifacts into THE
postable PR comment.

The per-run artifacts and per-case report blocks all exist after a ``make eval``
run — ``manifest.json`` + ``results.jsonl`` (``artifacts.py``) and one
``<case_id>.md`` transcript per case (``conftest.py``'s ``_write_sample_report``)
— but no step composes them into the ONE markdown document the protocol
(``docs/eval-report-format.md``) specifies. This module is that step.

Given a completed run's report directory it emits one markdown comment, in the
format-spec's section order:

  1. the manifest header (commit · model · N · lever), via ``render_manifest_header``.
  2. the run totals — the run-level aggregate across every case (mean-of-scores,
     all-pass rate, and the failure-cause tally), computed by flattening every
     case's per-sample scores/causes.
  3. one block per case: its dual RESULT line (mean + all-pass) and cause summary
     (``render_cause_summary`` — this is where the per-case aggregates finally
     render; the incremental per-sample flow in ``conftest.py`` cannot, since a
     cause tally is a whole-case aggregate), then the case's transcript report
     folded into a collapsed ``<details>``.

Pure artifact consumption: no model, no git, no network. The per-case metrics
are derived from each ``CaseArtifact``'s ``sample_scores`` / ``sample_causes``
so a case's RESULT line and the run totals are computed the same way (a passed
sample carries no cause, so the all-pass count is the count of ``None`` causes —
see ``_sample_cause`` in ``artifacts.py``).

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
    FailureCause,
    RunManifest,
    count_causes,
    pathology_excluded,
    render_cause_summary,
    render_manifest_header,
)

# ── Section literals (no magic strings) ──────────────────────────────────────
RUN_TOTALS_HEADING = "## Run totals"
NO_TRANSCRIPT = "_(no transcript recorded)_"
SECTION_SEPARATOR = "\n\n"

USAGE = "usage: python -m penny.tests.eval.assemble <report_dir>"


def assemble_run_comment(report_dir: Path) -> str:
    """Compose the run's whole PR comment from its report directory (the summary method).

    Reads ``manifest.json`` + ``results.jsonl`` + each ``<case_id>.md`` and renders the
    manifest header, the run totals, and one folded block per case, in spec order."""
    manifest = load_manifest(report_dir)
    artifacts = load_case_artifacts(report_dir)
    sections = [
        render_manifest_header(manifest).rstrip("\n"),
        render_run_totals(artifacts),
    ]
    sections += [render_case_block(report_dir, manifest, artifact) for artifact in artifacts]
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
    """The run-level aggregate across every case — the dual metrics line then the cause tally,
    parallel to a per-case block (and each line stays skimmable rather than one long run-on)."""
    scores, causes = _flatten(artifacts)
    result_body, cause_line = _result_metrics(scores, causes)
    return f"{RUN_TOTALS_HEADING}\n\n{result_body}\n{cause_line}"


def render_case_block(report_dir: Path, manifest: RunManifest, artifact: CaseArtifact) -> str:
    """One case's block: heading, the dual RESULT line, its cause summary, and the folded
    transcript. The RESULT line and cause summary are the per-case aggregates that the
    incremental per-sample flow could not render."""
    result_body, cause_line = _result_metrics(artifact.sample_scores, artifact.sample_causes)
    transcripts = _transcript_block(report_dir, manifest, artifact.case_id)
    return "\n".join(
        [
            f"### `{artifact.case_id}` — {artifact.family}",
            "",
            f"**RESULT:** {result_body}",
            cause_line,
            "",
            f"<details><summary>transcripts — {artifact.case_id}</summary>",
            "",
            transcripts,
            "",
            "</details>",
        ]
    )


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
