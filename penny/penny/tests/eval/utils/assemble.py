"""Run-comment assembler (#1717/#1725): compose a completed run's artifacts into THE
postable PR comment — the durable record of the iteration.

The per-run artifacts and per-case report blocks all exist after a ``make eval`` run —
``manifest.json`` + ``results.jsonl`` (``artifacts.py``) and one ``<case_id>.md``
transcript per case (``conftest.py``'s ``_write_sample_report``, now the iteration-6
transcript-integrated blocks rendered by ``report.py``, under the case document its own
three sections and shared prompts render into) — but no step composes them into
the ONE markdown document the format spec (``docs/eval-report-format.md``) specifies. This
module is that step.

Given a completed run's report directory it emits one markdown comment (v3, #1725):

  1. the **run header** — one identity line (run id · commit · model · N · lever), the
     **RESULT** line (mean · all-pass · pathology-excluded · cause tally · per-family
     rollup · timings), a **gate** line per gated case (``⚖ threshold on metric → PASS/FAIL``),
     and — in diff mode — a **flips** index (each regressed check + the samples it flipped in).
  2. one section per case — its heading (only when the run spans multiple cases) above the
     case's per-sample transcript blocks. EVERY sample block folds whole under its banner — the
     one and only rendering (#1753/#1759): collapsed by default, its full body always a click
     away, identical in the on-disk ``<case_id>.md`` and this comment (there is no banner-only /
     compact form — "default collapsed" never means the body is dropped).
  3. the **footer** — the local artifact directory + the ``make assemble`` re-render line.

Pure artifact + transcript consumption: no model, no git, no network — so it's exercised by
plain (non-eval) whole-render tests. The gate value is read from each ``CaseArtifact``'s
``min_pass_rate`` / ``gate_metric``; the flips index resolves the baseline from the run's DURABLE
manifest reference (``RunManifest.baseline``, recorded at eval time; ``EVAL_BASELINE`` overrides
for an ad-hoc re-diff), joining on ``(case_id, label)`` — the same diff key the per-sample REGRESSED
marks use. Reading a durable reference (not a volatile env at assemble time) is what keeps the
header flips index consistent with the per-row badges baked into the transcripts (#1752).

Run it via ``python -m penny.tests.eval.utils.assemble <report_dir>``
(writes the comment to stdout).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from penny.tests.eval.utils import cohort, comment_split, report
from penny.tests.eval.utils.artifacts import (
    MANIFEST_FILENAME,
    CaseArtifact,
    CheckCell,
    RunManifest,
    load_results_lines,
    render_manifest_header,
)
from penny.tests.eval.utils.baseline import Baseline, resolve_baseline

# ── Section literals (no magic strings) ──────────────────────────────────────
GATE_LABEL = "**gate:**"
FLIPS_LABEL = "flips:"
NO_TRANSCRIPT = "_(no transcript recorded)_"
SECTION_SEPARATOR = "\n\n"

# The most a single sample fold may render to before it is given internal seams (#1917).  The
# splitter's own per-part budget, read from it rather than restated: a fold larger than one part
# can never be packed into a postable comment however the document is cut, because a fold is
# the finest seam the splitter has.
SAMPLE_FOLD_BUDGET = comment_split.PART_BUDGET
GATING_GLYPH = "⚖"
FLIP_GLYPH = "✅→❌"
UNKNOWN_COMMIT = "unknown"

USAGE = "usage: python -m penny.tests.eval.utils.assemble <report_dir>"


def assemble_run_comment(report_dir: Path) -> str:
    """Compose the run's whole PR comment from its report directory (the summary method): the run
    header, one section per case (heading only when multi-case), and the local-artifacts footer.

    EVERY sample block folds whole under its banner — collapsed by default, its full body always a
    click away, identical to the on-disk ``.md`` (#1753/#1759, the one and only rendering).

    A case's preamble — its three sections plus everything its samples share — is carried through
    verbatim from the ``.md``, so the document the assembler posts and the one on disk are one
    rendering rather than two that can disagree."""
    manifest = load_manifest(report_dir)
    artifacts = load_case_artifacts(report_dir)
    # The flips index reads the run's DURABLE baseline reference (recorded in the manifest at eval
    # time), so it survives to assemble time even when `make assemble` carries no EVAL_BASELINE —
    # the divergence that dropped the flips line while the baked per-row REGRESSED badges stayed
    # (#1752).
    baseline = resolve_baseline(manifest.baseline)
    multi = len(artifacts) > 1
    sections = [render_run_header(manifest, artifacts, baseline)]
    sections += [_case_section(report_dir, manifest, artifact, multi) for artifact in artifacts]
    sections.append(render_footer(report_dir))
    # Nothing is lifted here any more (#1997): a case states its shared system prompts ONCE on
    # its own document, at the moment they are read off the run, so the assembler composes
    # sections that are already deduplicated rather than diffing finished markdown to discover
    # what they had in common.
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
    """Read every case record in the run dir (one per non-blank line), in file order.

    A run under xdist writes one results file per worker, so the whole run is the union of
    them — reading only ``results.jsonl`` would silently report a fraction of the cases as
    if it were all of them.  No files → no cases: a manifest can exist before any case has
    recorded."""
    return [CaseArtifact.model_validate_json(line) for line in load_results_lines(report_dir)]


# ── The run header (identity · RESULT · gate · flips) ────────────────────────
def render_run_header(
    manifest: RunManifest, artifacts: list[CaseArtifact], baseline: Baseline | None
) -> str:
    """The run header: what was run and what it cost, a gate line per gated case, and (in diff
    mode) the flips index.

    There is no RESULT line any more (#1997).  Every number on it belonged to the design #1994
    replaced — `mean` and `all-pass` are aggregates of PER-SAMPLE scores, and a sample has no
    score now; `pathology-excluded` and the behavioural/pathology/harness tally are the
    three-cause taxonomy that assertions, variance and harness supersede.  It sat ABOVE the new
    per-case summary line and contradicted it, so a reader met the obsolete summary first.

    The fields still live in ``results.jsonl``, where the per-case gate line and the baseline
    diff read them; what is gone is presenting them as the run's headline."""
    dirty = " (dirty)" if manifest.dirty else ""
    provider = f" via `{manifest.provider}`" if manifest.provider else ""
    lines = [
        f"**{manifest.run_id}** · commit `{_short(manifest.commit)}`{dirty}",
        f"**model:** `{manifest.model}` · `{manifest.endpoint}`{provider} · "
        f"**embeddings:** `{manifest.embedding_model}`",
        f"**lever:** {manifest.lever}",
    ]
    cost = render_run_cost(artifacts)
    if cost:
        lines.append(cost)
    lines += render_gate_lines(artifacts)
    flips = render_flips_line(artifacts, baseline)
    if flips:
        lines.append(flips)
    return "\n".join(lines)


def _short(commit: str) -> str:
    """The 8-char short commit for the header (``unknown`` passes through)."""
    return commit if commit == UNKNOWN_COMMIT else commit[:8]


def render_run_cost(artifacts: list[CaseArtifact]) -> str:
    """What the run spent, PER SAMPLE — never as a total (#1994 §4a).

    A total is not comparable across cohort sizes, the same trap the entropy denominator is, and
    the figure that decides a model for local hardware is what one sample costs: measured on the
    same fixtures, gpt-oss spends ~39,600 in / 5,200 out against gemma's ~39,800 / 13,200. Input
    is OURS — prompt and context design, so a rise is what a prompt edit regresses; output is the
    MODEL's, so a rise on a fixed prompt is a model or config change."""
    samples = sum(artifact.samples for artifact in artifacts)
    calls = sum(artifact.timings.calls for artifact in artifacts)
    if not samples or not calls:
        return ""
    per = cohort.per_sample_cost(
        samples=samples,
        calls=calls,
        duration_ms=sum(a.timings.duration_ms for a in artifacts),
        input_tokens=sum(a.timings.input_tokens for a in artifacts),
        output_tokens=sum(a.timings.output_tokens for a in artifacts),
        reasoning_tokens=sum(a.timings.reasoning_tokens for a in artifacts),
    )
    if per is None:
        return ""
    thinking = (
        f" ({per.reasoning_tokens:,.0f} thinking, {per.reasoning_share:.0%})"
        if per.reasoning_tokens
        else ""
    )
    return (
        f"**cost/sample:** {per.input_tokens:,.0f} in / {per.output_tokens:,.0f} out{thinking} · "
        f"{per.calls:,.1f} calls · {per.seconds:,.0f}s  _over {samples} samples_"
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


# ── Per-case section + footer ────────────────────────────────────────────────
def _case_section(
    report_dir: Path, manifest: RunManifest, artifact: CaseArtifact, multi: bool
) -> str:
    """One case's section: its per-sample transcript blocks, under a ``### case — family`` heading
    only when the run spans multiple cases (a single-case run needs no divider)."""
    body = _transcript_block(report_dir, manifest, artifact)
    if multi:
        return f"### `{artifact.case_id}` — {artifact.family}\n\n{body}"
    return body


def _transcript_block(report_dir: Path, manifest: RunManifest, artifact: CaseArtifact) -> str:
    """The case's ``<case_id>.md`` transcript with its leading manifest header stripped (the run
    header carries the run identity once), re-normalized for the comment. A missing/empty
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
    return _folded_transcript(transcript, artifact.expand_samples, artifact.standing_counts)


def _folded_transcript(
    transcript: str,
    expand: Sequence[int] = (),
    artifact_counts: Mapping[str, int] | None = None,
) -> str:
    """Order a case for the COMMENT: its scores, the sample it nominated with the prompts that
    sample was run with, then the case's own inputs and its outliers.

    The two SIZE transforms live here and nowhere else (#1997), because the artifact on disk is
    the complete record and this is the index into it: a sample the case did not nominate is
    counted on one line, and an expanded sample has its thinking traces shortened to their head
    and their length. Measured on the reference port, thinking was 68% of every sample and one
    case ran to 787,681 characters. Neither transform can reach the ``.md``."""
    head, prompts, tail = _split_case(transcript)
    preamble, sample_blocks = report.split_case_transcript(head)
    nominated = set(expand)
    kept = [f"{report.SAMPLE_ROW} {number}" for number in sorted(nominated)]
    blocks = [preamble] if preamble else []
    others: list[int] = []
    for block in sample_blocks:
        number, banner, body = report.parse_sample_block(block)
        if nominated and number not in nominated:
            others.append(number)
            continue
        folded = report.summarise_thinking(body)
        if not nominated:
            # A case that named no representative keeps every sample in its own banner form —
            # the unported path, unchanged.
            blocks.append(report.fold_sample_parts(number, banner, folded, SAMPLE_FOLD_BUDGET))
            continue
        blocks.append(
            report.fold(
                report.representative_summary(banner, body.count("| step ")),
                report.render_representative(
                    banner=banner,
                    number=number,
                    prompts=report.elide_unused_prompts(prompts, kept),
                    transcript=folded,
                ),
            )
        )
    if others:
        blocks.append(_accounting(artifact_counts))
    if tail:
        blocks.append(tail)
    return SECTION_SEPARATOR.join(blocks) if blocks else NO_TRANSCRIPT


def _accounting(counts: Mapping[str, int] | None) -> str:
    """The one line standing in for the samples the comment does not carry."""
    tally = counts or {}
    return report.samples_accounted(
        matched=tally.get(cohort.Standing.TYPICAL.value, 0),
        diverged=tally.get(cohort.Standing.OUTLIER.value, 0),
    )


def _split_case(transcript: str) -> tuple[str, str, str]:
    """Split a case's document on the markers its own renderer wrote: what precedes the samples,
    the prompts they were run with, and what closes the case."""
    head, _, rest = transcript.partition(report.CASE_PROMPTS_MARKER)
    prompts, _, tail = rest.partition(report.CASE_TAIL_MARKER)
    if not rest:
        head, _, tail = transcript.partition(report.CASE_TAIL_MARKER)
        return head.rstrip(), "", tail.strip()
    return head.rstrip(), prompts.strip(), tail.strip()


def render_footer(report_dir: Path) -> str:
    """What the glyphs mean, then the n≤1 pointer from the comment back to the raw evidence — the
    LOCAL artifact directory (nothing is committed, #1725 policy) and the re-render line.

    The legend rides on the RUN rather than on each case: at ~100 cases a per-case copy is a
    hundred restatements of the same four glyphs, and the whole point of the one-line-per-case
    default view is that a hundred of them fit on a page."""
    return (
        f"{report.GLYPH_KEY}\n\n"
        f"_artifacts (local, never committed): `{report_dir}` · per-sample DBs beside them · "
        f"re-render: `EVAL_REPORT_DIR={report_dir} make assemble`_"
    )


# ── CLI: python -m penny.tests.eval.utils.assemble <report_dir> ─────────────────────
def main(argv: list[str]) -> int:
    """Write the assembled comment for the report dir to stdout; 1 on a bad dir. Every sample folds
    whole under its banner (collapsed by default, full body a click away) — the one rendering."""
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
