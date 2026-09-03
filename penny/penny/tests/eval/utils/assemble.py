"""Run-comment assembler (#1717/#1725): compose a completed run's artifacts into THE
postable PR comment — the durable record of the iteration.

The per-run artifacts and per-case report blocks all exist after a ``make eval`` run —
``manifest.json`` + ``results.jsonl`` (``artifacts.py``) and one ``<case_id>.md``
transcript per case (``conftest.py``'s ``_write_sample_report``, now the iteration-6
transcript-integrated blocks rendered by ``report.py``, under the case document its own
three sections and shared prompts render into) — but no step composes them into the ONE
markdown document that gets posted. This module is that step, and it is also where that
document's shape is DEFINED: ``test_assemble.py`` pins it as whole-render literals, so
there is no prose spec to drift from what is actually emitted.

Given a completed run's report directory it emits one markdown comment (v3, #1725):

  1. the **run roll-up** — ONLY when the run spans more than one case, because a single case
     names the report itself and its own table already carries every number one would repeat:
     a verdict over every deterministic check, the run's identity as a table, a **gate** line
     per gated case (``⚖ threshold on metric → PASS/FAIL``), and — in diff mode — a **flips**
     index (each regressed check + the samples it flipped in).
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
    VarianceReading,
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

USAGE = (
    "usage: python -m penny.tests.eval.utils.assemble <report_dir>\n"
    "       python -m penny.tests.eval.utils.assemble --comments <out_dir> <report_dir>..."
)
# The flag that switches the CLI from "print one document" to "write every postable part".
COMMENTS_FLAG = "--comments"


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
    # An EMPTY section is dropped rather than joined: a single-case run emits no roll-up, and
    # separating nothing from the first case would open the comment on a blank band.
    return SECTION_SEPARATOR.join(part for part in sections if part) + "\n"


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


# ── The run header (verdict · identity table · lever · gate · flips) ─────────
#
# A REPORT, not a debug log.  The verdict LEADS: the run id and the model are scaffolding, and
# the counted reading with its colour is the finding — it used to sit last, under five dense
# `key: value` lines.  Run-level facts are stated ONCE, in a table, because this has to scale:
# at ~33 cases the commit and the provider must not repeat per case, and each case becomes a
# scannable `####` line underneath.
# A run-level roll-up OUTRANKS the cases it summarises (`##` against their `###`).
_RUN_HEADING = "## Eval run · `{model}`"
# What a roll-up is FOR: summarising more than one thing.
_ROLLS_UP = 2
_RUN_TABLE_HEAD = "| measure | reading |\n|---|---|"
# What the roll-up AGGREGATES, said rather than implied: one number over several cases must not
# read as though it covered a single one.
_RUN_VERDICT = "**{glyph} {passed} / {total} checks · {rate:.0%}**{variance} — {scope}"
# The spread half of the headline: what varies MOST across the run, over EVERY feature. Not
# filtered to what could carry a ceiling — a ceiling catches a RISE and saturation governs that,
# while this answers which aspect is most variant right now, and filtering hides the answer. The
# shape rides beside the magnitude, because "one aspect spiking" and "everything wobbling" are
# different findings that a maximum alone cannot separate.
_RUN_VARIANCE = "  ·  **{glyph} variance max H {entropy:.3f}** `{feature}` · {varying}/{total} vary"


def render_run_header(
    manifest: RunManifest, artifacts: list[CaseArtifact], baseline: Baseline | None
) -> str:
    """The run's ROLL-UP — and only when there is something to roll up.

    A single-case run has none: the case names the report and its table already carries every
    number a run header would repeat, so printing one states the same figures twice under two
    headings. Several cases genuinely need a total, and it sits at `##` ABOVE them, outranking
    the `###` each case takes.

    Keyed on the case count rather than special-cased at one: the rule is "a roll-up summarises
    more than one thing", which is true at every N.

    There is no RESULT line any more (#1997). Every number on it belonged to the design #1994
    replaced — `mean` and `all-pass` are aggregates of PER-SAMPLE scores, and a sample has no
    score now.  There is no LEVER line either: it described the CODE CHANGE under test, which is
    changelog rather than result, and the commit and the PR carry that already.  What a run is
    testing, when it needs saying, belongs in plain language beside the report."""
    gates = render_gate_lines(artifacts)
    flips = render_flips_line(artifacts, baseline)
    extra = [*gates, *([flips] if flips else [])]
    if len(artifacts) < _ROLLS_UP:
        return "\n".join(extra)
    lines = [_RUN_HEADING.format(model=manifest.model), "", render_run_verdict(artifacts), ""]
    lines += [_RUN_TABLE_HEAD, *_run_rows(manifest, artifacts)]
    if extra:
        lines += ["", *extra]
    return "\n".join(lines)


def _run_rows(manifest: RunManifest, artifacts: list[CaseArtifact]) -> list[str]:
    """The run's identity, one fact per row — commit, where it ran, what it cost, which run."""
    dirty = " (dirty)" if manifest.dirty else ""
    rows = [f"| commit | `{_short(manifest.commit)}`{dirty} |"]
    if manifest.endpoint:
        provider = f" via **{manifest.provider}**" if manifest.provider else ""
        rows.append(f"| provider | `{manifest.endpoint}`{provider} |")
    rows.append(f"| embeddings | `{manifest.embedding_model}` |")
    cost = render_run_cost(artifacts)
    if cost:
        # Cost is PER SAMPLE and identical across every case in a run, so it is run metadata
        # rather than a finding about any case.  Its detailed fold stays in the case document.
        rows.append(f"| cost / sample | {cost} |")
    rows.append(f"| run | `{manifest.run_id}` |")
    return rows


def render_run_verdict(artifacts: list[CaseArtifact]) -> str:
    """The run's one counted reading, coloured, over every deterministic check it made.

    Scored checks only, so an advisory row cannot pad the denominator.  The scope clause names
    what the number covers — a run over several cases says so rather than letting one figure
    imply it described a single one."""
    rows = [check for artifact in artifacts for check in artifact.checks if check.scored]
    passed = sum(check.passed for check in rows)
    total = sum(check.total for check in rows)
    samples = sum(artifact.samples for artifact in artifacts)
    dead = cohort.Standing.DEAD.value
    excluded = sum(artifact.standing_counts.get(dead, 0) for artifact in artifacts)
    scope = (
        f"{len(artifacts)} case{'s' if len(artifacts) != 1 else ''} · "
        f"{samples} sample{'s' if samples != 1 else ''} · {excluded} excluded"
    )
    if not total:
        return f"**no deterministic checks** — {scope}"
    return _RUN_VERDICT.format(
        glyph=report.rate_glyph(passed / total),
        passed=passed,
        total=total,
        rate=passed / total,
        variance=_run_variance(artifacts),
        scope=scope,
    )


def _run_variance(artifacts: list[CaseArtifact]) -> str:
    """The run's spread reading — the SAME statistic the per-case line carries, so a reader can
    see how a case contributed to the total.

    Every feature counts, saturated ones included: saturation decides whether a CEILING is
    proposed, and that is a guard against a rise rather than an answer to "what is most variant
    here". Filtering by it would hide the very aspect worth surfacing.

    Empty when the run recorded no variance at all, which is what a record written before the
    field existed decodes as: a header that silently reported H 0.000 for an unmeasured run would
    be inventing a reading rather than omitting one."""
    readings = [feature for artifact in artifacts for feature in artifact.variance]
    if not readings:
        return ""
    top = max(readings, key=lambda feature: feature.entropy)
    return _RUN_VARIANCE.format(
        glyph=report.UNGATED_GLYPH,
        entropy=top.entropy,
        feature=top.name,
        varying=sum(1 for feature in readings if feature.distinct > 1),
        total=len(readings),
    )


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
    thinking = f" ({per.reasoning_share:.0%} thinking)" if per.reasoning_tokens else ""
    return (
        f"{per.input_tokens:,.0f} in · {per.output_tokens:,.0f} out{thinking} · "
        f"{per.calls:,.1f} calls · {per.seconds:,.0f}s"
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
            report.titled_fold(
                report.REPRESENTATIVE_HEADING,
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
    whole under its banner (collapsed by default, full body a click away) — the one rendering.

    With ``--comments <out_dir>``, write every POSTABLE part instead — one document per case
    across every run dir named, then the run summary last — and print their file names in post
    order, so the caller posts them with one loop and no knowledge of how a run is divided."""
    if argv[:1] == [COMMENTS_FLAG]:
        return _write_comments(argv[1:])
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


def _write_comments(argv: list[str]) -> int:
    """Write every postable part for the runs named, in post order, into ``argv[0]``.

    The SPLITTER still runs over each document, because a single case can outgrow the comment
    cap on its own — what changes is that it is given one case at a time rather than a whole
    run, so its one legal seam has a document it can actually cut."""
    if len(argv) < 2:
        print(USAGE, file=sys.stderr)
        return 2
    out_dir, report_dirs = Path(argv[0]), [Path(one) for one in argv[1:]]
    try:
        documents = [
            document
            for report_dir in report_dirs
            for document in assemble_case_comments(report_dir)
        ]
        documents.append(render_run_summary(report_dirs))
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    parts = [part for document in documents for part in comment_split.split_run_comment(document)]
    refusal = next(
        (
            reason
            for document in documents
            if (reason := comment_split.build_noise_reason(document))
        ),
        None,
    ) or comment_split.unsplittable_reason(parts)
    if refusal:
        print(f"assemble: refusing to post — {refusal}", file=sys.stderr)
        return 1
    for name in comment_split.write_parts(parts, out_dir):
        print(name)
    return 0


# ── One comment per case, and the run's own summary (#2020) ──────────────────
#
# A run's cases are posted as SEPARATE comments — one per case per model — rather than as
# parts of one document.  Two reasons, and they are the same reason twice: a case report is
# what a reader opens and refers back to, and the standing posting convention is one comment
# per case; and a whole-run document is a size problem with no good answer, because the only
# seam a comment may be cut on is a fold boundary and a three-case run assembled to 68,385
# characters against a 65,536 cap.  Split by case, each document is its own size.
#
# The splitter still runs over each one — a SINGLE case can outgrow the cap on its own — so
# nothing here replaces it; what changes is what a document contains.
CASE_REPORT_SUFFIX = ".md"

SUMMARY_HEADING = "## Run summary"
_SUMMARY_MODEL_HEADING = "### `{model}`"
_SUMMARY_CHECKS = "**{glyph} {passed} / {total} checks · {rate:.0%}** — {cases}"
_SUMMARY_NO_CHECKS = "**no deterministic checks recorded**"
_SUMMARY_VARIANCE = (
    "**{glyph} variance max H {entropy:.3f}** `{feature}` in `{case_id}` · "
    "{varying} of {total} features vary{blind}"
)
_SUMMARY_NO_VARIANCE = "**no features measured**"
_SUMMARY_BLIND = " · {count} blind"
# The unit is IN THE HEADING, both places, because a token figure that could be either a
# total or a per-sample average is worse than none: they answer different questions and the
# difference here is a factor of fifteen.  The model header answers "what did this run cost",
# which is a total; the per-case column answers "how do these cases compare", which is per
# sample — the comparable form, since cases need not drive the same number of samples.
_SUMMARY_TABLE_HEAD = (
    "| # | case | deterministic | variance | samples | tokens / sample |\n|---|---|---|---|---|---|"
)
_SUMMARY_ROW = "| {position} | `{case_id}` | {checks} | {variance} | {samples} | {cost} |"
_SUMMARY_COST = "**{input:,} in · {output:,} out**{thinking} — run total, every sample driven"
_SUMMARY_THINKING = " ({share:.0%} thinking)"
_ROW_COST = "{input:,.0f} in · {output:,.0f} out"
_SUMMARY_CASES = "{count} case{plural}"
_ROW_CHECKS = "{glyph} {passed}/{total} · {rate:.0%}"
_ROW_VARIANCE = "{glyph} max H {entropy:.3f} `{feature}` · {varying}/{total} vary"
_ROW_ABSENT = "—"
_ROW_SAMPLES = "{pooled} pooled + {excluded} excluded"
_ROW_SAMPLES_UNPOOLED = "{samples} driven"


def started_cases(report_dir: Path) -> list[str]:
    """Every case the run STARTED, by id, in index order.

    Read off the per-case report files rather than off the results records, because those are
    two different counts and only this one answers the question a reader has.  A case writes
    its report header at the top of its runner, BEFORE any sample; it appends a results record
    only if it closes.  So a case that started and died is counted here and has no report — a
    reader sees ``1/4``, ``2/4``, ``4/4`` and the gap is visible, where counting closed cases
    would silently renumber to ``1/3`` and hide it.

    Sorted by case id: the index has to be stable across a re-run of the same set or the
    numbers mean nothing between rounds, and file order is the xdist worker's assignment,
    which differs run to run."""
    return sorted(path.stem for path in report_dir.glob(f"*{CASE_REPORT_SUFFIX}"))


def case_index(report_dir: Path, case_id: str) -> str:
    """Where ``case_id`` sits among the cases its run covers, as a heading renders it.

    N is per MODEL without having to say so: a run measures ONE model (``EVAL_MODELS``
    resolves one per invocation, so two models are two runs, each with its own directory and
    its own manifest), so the run's case count IS that model's."""
    cases = started_cases(report_dir)
    if case_id not in cases:
        return ""
    return report.render_case_index(cases.index(case_id) + 1, len(cases))


def assemble_case_comments(report_dir: Path) -> list[str]:
    """One postable document per case in the run, in index order.

    Each carries its case's own report and nothing else: the case document already states the
    run's identity in its own table, so a run header above it would repeat every figure under
    a second heading."""
    manifest = load_manifest(report_dir)
    by_id = {artifact.case_id: artifact for artifact in load_case_artifacts(report_dir)}
    documents = []
    for case_id in started_cases(report_dir):
        artifact = by_id.get(case_id)
        if artifact is None:
            continue
        body = report.with_case_index(
            _transcript_block(report_dir, manifest, artifact), case_index(report_dir, case_id)
        )
        documents.append(f"{body}\n\n{render_footer(report_dir)}\n")
    return documents


def render_run_summary(report_dirs: Sequence[Path]) -> str:
    """The whole run's numbers, one section per model — the LAST comment a run posts.

    GENERATED rather than written: it is the arithmetic the per-case headers already do, one
    level up, so it states what a reader would otherwise total by hand and never anything a
    person had to judge."""
    sections = [SUMMARY_HEADING]
    sections += [
        _model_section(load_manifest(report_dir), _ordered_artifacts(report_dir))
        for report_dir in report_dirs
    ]
    return SECTION_SEPARATOR.join(sections) + "\n"


def _ordered_artifacts(report_dir: Path) -> list[CaseArtifact]:
    """The run's case records in INDEX order, so ``3/5`` in a heading and the third row of the
    table are the same case."""
    by_id = {artifact.case_id: artifact for artifact in load_case_artifacts(report_dir)}
    return [by_id[case_id] for case_id in started_cases(report_dir) if case_id in by_id]


def _model_section(manifest: RunManifest, artifacts: Sequence[CaseArtifact]) -> str:
    """One model's whole run: its two headline readings, then a row per case."""
    lines = [_SUMMARY_MODEL_HEADING.format(model=manifest.model), ""]
    lines += [_summary_checks(artifacts), "", _summary_variance(artifacts), ""]
    cost = _summary_cost(artifacts)
    if cost:
        lines += [cost, ""]
    lines += [_SUMMARY_TABLE_HEAD]
    lines += [
        _SUMMARY_ROW.format(
            position=position,
            case_id=artifact.case_id,
            checks=_row_checks(artifact),
            variance=_row_variance(artifact),
            samples=_row_samples(artifact),
            cost=_row_cost(artifact),
        )
        for position, artifact in enumerate(artifacts, start=1)
    ]
    return "\n".join(lines)


def _summary_cost(artifacts: Sequence[CaseArtifact]) -> str:
    """What the model's whole run SPENT — a total, over every sample it drove.

    Every driven sample, INCLUDING the ones run health later excluded: a sample that burned
    tokens and was then dropped from the pool still cost them, so this and the pooled count
    beside it answer different questions and must not be reconciled.  The case record's
    timings are summed as each sample's drive returns, before any exclusion is decided, so
    that is already what they hold.

    Not comparable across runs of different sizes, which is exactly why the per-case column
    is per sample instead — the two units are split by the question each is asked."""
    total_in = sum(artifact.timings.input_tokens for artifact in artifacts)
    total_out = sum(artifact.timings.output_tokens for artifact in artifacts)
    reasoning = sum(artifact.timings.reasoning_tokens for artifact in artifacts)
    if not total_in and not total_out:
        return ""
    thinking = (
        _SUMMARY_THINKING.format(share=reasoning / total_out) if reasoning and total_out else ""
    )
    return _SUMMARY_COST.format(input=total_in, output=total_out, thinking=thinking)


def _row_cost(artifact: CaseArtifact) -> str:
    """What ONE sample of this case cost, which is the comparable form across cases.

    Divided by the samples DRIVEN rather than the ones pooled, because the tokens were spent
    driving all of them — dividing by the smaller pooled count would inflate every case that
    excluded a sample, and inflate it most where the harness misfired most."""
    per = cohort.per_sample_cost(
        samples=artifact.samples,
        calls=artifact.timings.calls,
        duration_ms=artifact.timings.duration_ms,
        input_tokens=artifact.timings.input_tokens,
        output_tokens=artifact.timings.output_tokens,
        reasoning_tokens=artifact.timings.reasoning_tokens,
    )
    if per is None or not (per.input_tokens or per.output_tokens):
        return _ROW_ABSENT
    return _ROW_COST.format(input=per.input_tokens, output=per.output_tokens)


def _scored_checks(artifact: CaseArtifact) -> tuple[int, int]:
    """One case's deterministic total — its scored checks, passed of present.

    Advisory checks are out, exactly as they are in the case's own header: a row that renders
    but does not count must not pad a denominator one level up either."""
    scored = [check for check in artifact.checks if check.scored]
    return sum(check.passed for check in scored), sum(check.total for check in scored)


def _summary_checks(artifacts: Sequence[CaseArtifact]) -> str:
    """The model's deterministic total, summed across its cases."""
    totals = [_scored_checks(artifact) for artifact in artifacts]
    passed, total = sum(one for one, _ in totals), sum(one for _, one in totals)
    if not total:
        return _SUMMARY_NO_CHECKS
    count = len(artifacts)
    return _SUMMARY_CHECKS.format(
        glyph=report.rate_glyph(passed / total),
        passed=passed,
        total=total,
        rate=passed / total,
        cases=_SUMMARY_CASES.format(count=count, plural="" if count == 1 else "s"),
    )


def _measured(artifact: CaseArtifact) -> list[VarianceReading]:
    """The features a case actually measured.

    A BLIND feature read its absent value on every sample: it scores ``0.000``, which is the
    number a cohort in perfect agreement scores and the opposite finding, so it is neither
    agreement nor spread and cannot stand in an aggregate as though it were either."""
    return [feature for feature in artifact.variance if not feature.blind]


def _summary_variance(artifacts: Sequence[CaseArtifact]) -> str:
    """The model's spread, as the MAXIMUM across its cases — never a mean.

    Normalised entropy is a reading of one ``(feature, model, N)``, so averaging across
    features would produce a number that is a reading of nothing — which is why a ceiling is
    recorded per feature and a comparison across either qualifier is refused.  The maximum is
    a real reading that exists in the data, so it is reported WITH where it came from — the
    feature and the case — beside how many features moved at all and how many saw nothing."""
    readings = [
        (feature, artifact.case_id) for artifact in artifacts for feature in _measured(artifact)
    ]
    blind = sum(len(artifact.variance) - len(_measured(artifact)) for artifact in artifacts)
    if not readings:
        return _SUMMARY_NO_VARIANCE
    top, case_id = max(readings, key=lambda pair: pair[0].entropy)
    return _SUMMARY_VARIANCE.format(
        glyph=report.UNGATED_GLYPH,
        entropy=top.entropy,
        feature=top.name,
        case_id=case_id,
        varying=sum(1 for feature, _ in readings if feature.distinct > 1),
        total=len(readings),
        blind=_SUMMARY_BLIND.format(count=blind) if blind else "",
    )


def _row_checks(artifact: CaseArtifact) -> str:
    """One case's deterministic reading, coloured the way its own header colours it."""
    passed, total = _scored_checks(artifact)
    if not total:
        return _ROW_ABSENT
    return _ROW_CHECKS.format(
        glyph=report.rate_glyph(passed / total), passed=passed, total=total, rate=passed / total
    )


def _row_variance(artifact: CaseArtifact) -> str:
    """One case's spread — its own maximum, over the features that saw something."""
    measured = _measured(artifact)
    if not measured:
        return _ROW_ABSENT
    top = max(measured, key=lambda feature: feature.entropy)
    return _ROW_VARIANCE.format(
        glyph=report.UNGATED_GLYPH,
        entropy=top.entropy,
        feature=top.name,
        varying=sum(1 for feature in measured if feature.distinct > 1),
        total=len(measured),
    )


def _row_samples(artifact: CaseArtifact) -> str:
    """What the case's rate is OVER — pooled beside excluded.

    Stated because the two read identically in a rate and mean opposite things: a
    reroll-exhausted sample costs a 15-sample case about five points, and without the excluded
    count beside it that loss reads as the model getting something wrong.

    POOLED is read off the claims rather than carried as its own number: a cohort claim is
    answered for every complete sample and has no third outcome, so a claim's ``total`` IS the
    pooled count — a second copy could only drift from it.  A case that drove no cohort has no
    claims to read and no pooling to report, so it states what it drove."""
    pooled = max((check.total for check in artifact.checks if check.scored), default=0)
    if not artifact.expand_samples or not pooled:
        return _ROW_SAMPLES_UNPOOLED.format(samples=artifact.samples)
    return _ROW_SAMPLES.format(pooled=pooled, excluded=max(artifact.samples - pooled, 0))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
