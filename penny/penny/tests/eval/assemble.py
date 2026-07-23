"""Run-comment assembler (#1717/#1725): compose a completed run's artifacts into THE
postable PR comment — the durable record of the iteration.

The per-run artifacts and per-case report blocks all exist after a ``make eval`` run —
``manifest.json`` + ``results.jsonl`` (``artifacts.py``) and one ``<case_id>.md``
transcript per case (``conftest.py``'s ``_write_sample_report``, now the iteration-6
transcript-integrated blocks rendered by ``report.py``) — but no step composes them into
the ONE markdown document the format spec (``docs/eval-report-format.md``) specifies. This
module is that step.

Given a completed run's report directory it emits one markdown comment (v3, #1725; compact by
default, #1753):

  1. the **run header** — one identity line (run id · commit · model · N · lever), the
     **RESULT** line (mean · all-pass · pathology-excluded · cause tally · per-family
     rollup · timings), a **gate** line per gated case (``⚖ threshold on metric → PASS/FAIL``),
     and — in diff mode — a **flips** index (each regressed check + the samples it flipped in).
  2. one section per case — its heading (only when the run spans multiple cases) above the
     case's per-sample transcript blocks. Every block folds whole under its banner (uniform
     collapse); in the default **compact** comment a clean-pass sample renders its banner line
     ONLY (no transcript body — that stays in the on-disk ``<case_id>.md`` the footer points at),
     while failed/fragile/regressed samples keep their full (collapsed) step tables. ``full=True``
     (CLI ``--full`` / ``make assemble EVAL_FULL=1``) emits every sample's full folded body.
  3. the **footer** — the local artifact directory + the ``make assemble`` re-render line.

Pure artifact + transcript consumption: no model, no git, no network — so it's exercised by
plain (non-eval) whole-render tests. The gate value is read from each ``CaseArtifact``'s
``min_pass_rate`` / ``gate_metric``; the flips index resolves the baseline from the run's DURABLE
manifest reference (``RunManifest.baseline``, recorded at eval time; ``EVAL_BASELINE`` overrides
for an ad-hoc re-diff), joining on ``(case_id, label)`` — the same diff key the per-sample REGRESSED
marks use. Reading a durable reference (not a volatile env at assemble time) is what keeps the
header flips index consistent with the per-row badges baked into the transcripts (#1752).

Run it via ``python -m penny.tests.eval.assemble <report_dir>`` (writes the comment to stdout).
"""

from __future__ import annotations

import sys
from pathlib import Path

from penny.tests.eval import report
from penny.tests.eval.artifacts import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    CaseArtifact,
    CheckCell,
    FailureCause,
    RunManifest,
    count_causes,
    pathology_excluded,
    render_manifest_header,
)
from penny.tests.eval.baseline import Baseline, resolve_baseline

# ── Section literals (no magic strings) ──────────────────────────────────────
RESULT_LABEL = "**RESULT:**"
GATE_LABEL = "**gate:**"
FLIPS_LABEL = "flips:"
FAMILIES_LABEL = "families:"
CAUSES_LABEL = "causes —"
NO_TRANSCRIPT = "_(no transcript recorded)_"
SECTION_SEPARATOR = "\n\n"
GATING_GLYPH = "⚖"
FLIP_GLYPH = "✅→❌"
UNKNOWN_COMMIT = "unknown"
FULL_FLAG = "--full"  # emit every sample's full body, superseding the compact default (#1753)

USAGE = "usage: python -m penny.tests.eval.assemble [--full] <report_dir>"


def assemble_run_comment(report_dir: Path, *, full: bool = False) -> str:
    """Compose the run's whole PR comment from its report directory (the summary method): the run
    header, one section per case (heading only when multi-case), and the local-artifacts footer.

    ``full=False`` (the default) is compact: a clean-pass sample renders its banner line only
    (#1753); ``full=True`` emits every sample's full (collapsed) transcript body."""
    manifest = load_manifest(report_dir)
    artifacts = load_case_artifacts(report_dir)
    # The flips index reads the run's DURABLE baseline reference (recorded in the manifest at eval
    # time), so it survives to assemble time even when `make assemble` carries no EVAL_BASELINE —
    # the divergence that dropped the flips line while the baked per-row REGRESSED badges stayed
    # (#1752).
    baseline = resolve_baseline(manifest.baseline)
    multi = len(artifacts) > 1
    sections = [render_run_header(manifest, artifacts, baseline)]
    sections += [
        _case_section(report_dir, manifest, artifact, multi, full=full) for artifact in artifacts
    ]
    sections.append(render_footer(report_dir))
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


# ── The run header (identity · RESULT · gate · flips) ────────────────────────
def render_run_header(
    manifest: RunManifest, artifacts: list[CaseArtifact], baseline: Baseline | None
) -> str:
    """The run header: the identity line, the RESULT line, a gate line per gated case, and (in
    diff mode) the flips index."""
    dirty = " (dirty)" if manifest.dirty else ""
    lines = [
        f"**{manifest.run_id}** · commit `{_short(manifest.commit)}`{dirty} · {manifest.model} · "
        f"N={manifest.samples} · **lever:** {manifest.lever}",
        f"{RESULT_LABEL} {render_result_line(artifacts)}",
    ]
    lines += render_gate_lines(artifacts)
    flips = render_flips_line(artifacts, baseline)
    if flips:
        lines.append(flips)
    return "\n".join(lines)


def _short(commit: str) -> str:
    """The 8-char short commit for the header (``unknown`` passes through)."""
    return commit if commit == UNKNOWN_COMMIT else commit[:8]


def render_result_line(artifacts: list[CaseArtifact]) -> str:
    """The run-level RESULT line: the dual metrics, the pathology-excluded mean, the cause tally,
    the per-family rollup, and the summed timings — one skimmable line."""
    scores, causes = _flatten(artifacts)
    total = len(scores)
    mean = sum(scores) / total if total else 0.0
    all_pass = sum(1 for cause in causes if cause is None)
    excluded_mean, _kept = pathology_excluded(scores, causes)
    counts = count_causes(causes)
    parts = [
        f"mean {mean:.2f}",
        f"all-pass {all_pass}/{total}",
        f"pathology-excluded {excluded_mean:.2f}",
        f"{CAUSES_LABEL} behavioral {counts.behavioral} · pathology {counts.pathology} · "
        f"harness {counts.harness}",
        _family_rollup(artifacts),
    ]
    timings = _timings(artifacts)
    if timings:
        parts.append(timings)
    return " · ".join(parts)


def _family_rollup(artifacts: list[CaseArtifact]) -> str:
    """``families: <fam> <mean> [(<n> cases)] · …`` — each family's mean over its samples, with a
    case count only when the family spans more than one case."""
    groups: dict[str, list[CaseArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(artifact.family, []).append(artifact)
    parts = []
    for family, group in groups.items():
        scores, _causes = _flatten(group)
        mean = sum(scores) / len(scores) if scores else 0.0
        suffix = f" ({len(group)} cases)" if len(group) > 1 else ""
        parts.append(f"{family} {mean:.2f}{suffix}")
    return f"{FAMILIES_LABEL} {' · '.join(parts)}"


def _timings(artifacts: list[CaseArtifact]) -> str:
    """The summed run timings — ``<calls> calls · <s>s · <in>K in / <out>K out`` (empty when no
    model call was logged)."""
    calls = sum(artifact.timings.calls for artifact in artifacts)
    if not calls:
        return ""
    duration_ms = sum(artifact.timings.duration_ms for artifact in artifacts)
    input_tokens = sum(artifact.timings.input_tokens for artifact in artifacts)
    output_tokens = sum(artifact.timings.output_tokens for artifact in artifacts)
    return (
        f"{calls} calls · {duration_ms / 1000:.0f}s · "
        f"{input_tokens / 1000:.1f}K in / {output_tokens / 1000:.1f}K out"
    )


def render_gate_lines(artifacts: list[CaseArtifact]) -> list[str]:
    """One gate line per gated case (``min_pass_rate`` set): the threshold, which score it gates,
    the gated value, and PASS/FAIL. In a multi-case run each gate names its case."""
    lines = []
    for artifact in artifacts:
        if artifact.min_pass_rate is None:
            continue
        gated = (
            artifact.mean if artifact.gate_metric == "mean" else artifact.pathology_excluded_mean
        )
        verdict = "✅ PASS" if gated >= artifact.min_pass_rate else "❌ FAIL"
        prefix = f"`{artifact.case_id}`: " if len(artifacts) > 1 else ""
        lines.append(
            f"{GATE_LABEL} {prefix}{GATING_GLYPH} {artifact.min_pass_rate} on "
            f"{artifact.gate_metric} → **{verdict}** ({gated:.2f})"
        )
    return lines


def render_flips_line(artifacts: list[CaseArtifact], baseline: Baseline | None) -> str:
    """The diff-mode flips index — each check that was fully green in the baseline but failed a
    sample here (a regression), with the samples it flipped in. Empty off-diff / on a clean run."""
    if baseline is None:
        return ""
    entries = []
    for artifact in artifacts:
        for outcome in artifact.checks:
            if not baseline.was_passing(artifact.case_id, outcome.label):
                continue
            fails = [i for i, cell in enumerate(outcome.cells) if cell == CheckCell.FAILED]
            if fails:
                where = ", ".join(f"s{index + 1}" for index in fails)
                entries.append(f"{outcome.label} {FLIP_GLYPH} ({where})")
    return f"{FLIPS_LABEL} {' · '.join(entries)}" if entries else ""


def _flatten(artifacts: list[CaseArtifact]) -> tuple[list[float], list[FailureCause | None]]:
    """Every case's per-sample scores and causes concatenated — the run-totals denominator."""
    scores: list[float] = []
    causes: list[FailureCause | None] = []
    for artifact in artifacts:
        scores.extend(artifact.sample_scores)
        causes.extend(artifact.sample_causes)
    return scores, causes


# ── Per-case section + footer ────────────────────────────────────────────────
def _case_section(
    report_dir: Path, manifest: RunManifest, artifact: CaseArtifact, multi: bool, *, full: bool
) -> str:
    """One case's section: its per-sample transcript blocks, under a ``### case — family`` heading
    only when the run spans multiple cases (a single-case run needs no divider)."""
    body = _transcript_block(report_dir, manifest, artifact, full=full)
    if multi:
        return f"### `{artifact.case_id}` — {artifact.family}\n\n{body}"
    return body


def _transcript_block(
    report_dir: Path, manifest: RunManifest, artifact: CaseArtifact, *, full: bool
) -> str:
    """The case's ``<case_id>.md`` transcript with its leading manifest header stripped (the run
    header carries the run identity once), re-normalized for the comment (#1753). A missing/empty
    transcript renders a placeholder."""
    path = report_dir / f"{artifact.case_id}.md"
    if not path.is_file():
        return NO_TRANSCRIPT
    text = path.read_text()
    header = render_manifest_header(manifest) + "\n"  # exactly what write_case_header stamped
    if text.startswith(header):
        text = text[len(header) :]
    transcript = text.strip()
    if not transcript:
        return NO_TRANSCRIPT
    return _compact_transcript(transcript, artifact, full=full)


def _compact_transcript(transcript: str, artifact: CaseArtifact, *, full: bool) -> str:
    """Re-normalize a case's per-sample blocks for the comment (#1753): every block folds whole
    under its banner; in compact mode (``full=False``, the default) a clean-pass sample renders its
    banner line ONLY — no body — while failed/fragile/regressed samples keep their full (collapsed)
    tables. Re-folds an old unfolded ``#### `` block on the way, so a re-assembled prior run is
    uniform too. Clean-pass is read from the artifact, never the ``.md``'s shape."""
    clean = _clean_pass_by_number(artifact)
    blocks = []
    for block in report.split_sample_blocks(transcript):
        number, banner, body = report.parse_sample_block(block)
        if not full and clean.get(number, False):
            blocks.append(report.sample_banner_line(number, banner))
        else:
            blocks.append(report.fold_sample(number, banner, body))
    return SECTION_SEPARATOR.join(blocks) if blocks else NO_TRANSCRIPT


def _clean_pass_by_number(artifact: CaseArtifact) -> dict[int, bool]:
    """Per-sample-number clean-pass flags (passed AND not fragile) — the compact-mode banner-only
    predicate. A passed sample carries no cause; a fragile pass carries one only via its flag, so
    both axes are read. Sample N lives at index N-1 in the aligned per-sample lists."""
    fragile = artifact.sample_fragile
    return {
        index + 1: cause is None and not (index < len(fragile) and fragile[index])
        for index, cause in enumerate(artifact.sample_causes)
    }


def render_footer(report_dir: Path) -> str:
    """The n≤1 pointer from the comment back to the raw evidence — the LOCAL artifact directory
    (nothing is committed, #1725 policy) and the ``make assemble`` re-render line."""
    return (
        f"_artifacts (local, never committed): `{report_dir}` · per-sample DBs beside them · "
        f"re-render: `EVAL_REPORT_DIR={report_dir} make assemble`_"
    )


# ── CLI: python -m penny.tests.eval.assemble [--full] <report_dir> ────────────
def main(argv: list[str]) -> int:
    """Write the assembled comment for the report dir to stdout; 1 on a bad dir. ``--full`` emits
    every sample's full body (the everything-in form); the default is compact (#1753)."""
    full = FULL_FLAG in argv
    positional = [arg for arg in argv if arg != FULL_FLAG]
    if len(positional) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    try:
        comment = assemble_run_comment(Path(positional[0]), full=full)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    sys.stdout.write(comment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
