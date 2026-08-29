"""Whole-render tests for the run-comment assembler (``assemble.py``, v3 / #1725).

NOT eval-marked — they drive the deterministic assembler over a SYNTHETIC report directory
(manifest + results.jsonl + per-case ``.md`` transcripts rendered by ``report.py``), so they run
inside ``make check``: no git, no model, no container. The assembled comment is asserted as a
WHOLE-RENDER literal (pr-review-guide §6): the run header (identity · RESULT · gate), the
per-sample transcript where EVERY sample folds whole under its banner — the one and only rendering
(#1753/#1759 — collapsed by default, full body a click away, byte-identical to the on-disk ``.md``,
no compact/banner-only form), the multi-family rollup + per-case headings, the diff-mode flips
index, the local-artifacts footer, and the CLI contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from penny.tests.eval.utils import report
from penny.tests.eval.utils.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckCell,
    CheckOutcome,
    FailureCause,
    RunManifest,
    build_manifest,
    render_manifest_header,
)
from penny.tests.eval.utils.assemble import (
    USAGE,
    assemble_run_comment,
    load_manifest,
    main,
)

_TIMINGS = CaseTimings(calls=19, duration_ms=148000, input_tokens=54200, output_tokens=5900)
_P = CheckCell.PASSED
_F = CheckCell.FAILED


def _write_run(
    report_dir: Path,
    manifest: RunManifest,
    artifacts: list[CaseArtifact],
    transcripts: dict[str, str],
) -> None:
    """Materialise a completed run's report dir: the manifest, one ``results.jsonl`` line per case,
    and each named case's ``<case_id>.md`` prefixed with the manifest header ``write_case_header``
    stamps (so the assembler's header-strip is exercised). A case absent from ``transcripts`` gets
    no ``.md`` (the honest missing-transcript placeholder)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    with (report_dir / "results.jsonl").open("w") as handle:
        for artifact in artifacts:
            handle.write(artifact.model_dump_json() + "\n")
    header = render_manifest_header(manifest) + "\n"
    for case_id, body in transcripts.items():
        (report_dir / f"{case_id}.md").write_text(header + body)


def _footer(report_dir: Path) -> str:
    """The run-level tail: the glyph legend once, then the pointer back to the raw evidence."""
    return (
        f"{report.GLYPH_KEY}\n\n"
        f"_artifacts (local, never committed): `{report_dir}` · per-sample DBs beside them · "
        f"re-render: `EVAL_REPORT_DIR={report_dir} make assemble`_\n"
    )


def _browse_sample() -> str:
    """A one-sample browse-answer transcript block, as the ``.md`` writer stores it — always folded
    whole (#1753)."""
    events = [
        report.Event(report.EventKind.USER, "deepest lake?"),
        report.Event(report.EventKind.CALL, "browse({...})", thinking="verify"),
        report.Event(report.EventKind.REPLY, "Baikal 1642m", thinking="answer"),
    ]
    checks = [report.CheckView("C1", "browsed", "spine", True, False, True, anchor_index=1)]
    banner = report.render_banner(passed=True, duration_s=45, calls=8)
    sample = report.build_sample(
        number=1, banner=banner, events=events, checks=checks, run_close_score="1/1"
    )
    return report.render_sample(sample) + "\n\n"


_BROWSE_SAMPLE_FOLDED = (
    "<details><summary>sample 1 — ✅ pass · 45s · 8 calls</summary>\n"
    "\n"
    '| step 1 · 👤 | "deepest lake?" | ✅ |\n'
    "|---|---|---|\n"
    "| expected | C1 [spine]⚖ browsed |  |\n"
    "| 💭 | verify |  |\n"
    "| actual | 🔧 browse({...}) | ✅ C1 |\n"
    "| 💭 | answer |  |\n"
    "| actual | 🤖 Baikal 1642m |  |\n"
    "\n"
    "</details>"
)
# The forbidden banner-only heading form — the assembler must NEVER emit it (#1759, compact gone).
_BROWSE_SAMPLE_BANNER_ONLY = "#### sample 1 — ✅ pass · 45s · 8 calls"


def test_single_gated_case_whole_render(tmp_path: Path) -> None:
    """A single gated case: the run header (identity · RESULT with timings · a gate line), the
    clean-pass sample's FULL folded body (#1759 — collapsed by default, body always present, never
    banner-only), and the footer — no per-case heading (single case)."""
    manifest = build_manifest(
        commit="abba710a03ae3555148fea6a86712e9af020499a",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=3,
        lever="framework baseline",
        now=datetime(2026, 7, 21, 5, 10, 17, tzinfo=UTC),
    )
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_chat_response.py::browse_answer",
        family="browse-answer",
        mean=0.67,
        all_pass_rate=2 / 3,
        pathology_excluded_mean=0.67,
        samples=3,
        sample_scores=[1.0, 1.0, 0.0],
        sample_causes=[None, None, FailureCause.HARNESS],
        sample_fragile=[False, False, False],
        cause_counts=CauseCounts(harness=1),
        checks=[CheckOutcome(label="browsed", passed=2, total=3, scored=True, cells=[_P, _P, _F])],
        timings=_TIMINGS,
        min_pass_rate=0.75,
        gate_metric="mean",
    )
    _write_run(tmp_path, manifest, [artifact], {artifact.case_id: _browse_sample()})
    assert assemble_run_comment(tmp_path) == (
        # A single-case run emits NO roll-up: the case names the report and its own table
        # carries every number one would repeat.
        "**gate:** ⚖ 0.75 on mean → **❌ FAIL** (0.67)\n"
        "\n" + _BROWSE_SAMPLE_FOLDED + "\n"
        "\n" + _footer(tmp_path)
    )


def test_two_family_run_with_missing_transcript_whole_render(tmp_path: Path) -> None:
    """A two-family run: the RESULT line's family rollup, per-case ``### case — family`` headings
    (present only when the run spans multiple cases), and a case whose ``.md`` is absent folding an
    honest placeholder rather than crashing. No gate line (no case gates)."""
    manifest = build_manifest(
        commit="beef1234beef1234beef1234beef1234beef1234",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=2,
        lever="two families",
        now=datetime(2026, 7, 20, 9, 0, 0, tzinfo=UTC),
    )
    alpha = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_a.py::one",
        family="alpha",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=2,
        sample_scores=[1.0, 1.0],
        sample_causes=[None, None],
        sample_fragile=[False, False],
        cause_counts=CauseCounts(),
        checks=[],
        timings=_TIMINGS,
    )
    beta = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_b.py::two",
        family="beta",
        mean=0.5,
        all_pass_rate=0.5,
        pathology_excluded_mean=0.5,
        samples=2,
        sample_scores=[1.0, 0.0],
        sample_causes=[None, FailureCause.BEHAVIORAL],
        sample_fragile=[False, False],
        cause_counts=CauseCounts(behavioral=1),
        checks=[],
        timings=_TIMINGS,
    )
    hi_events = [
        report.Event(report.EventKind.USER, "hi"),
        report.Event(report.EventKind.REPLY, "hey", thinking=""),
    ]
    banner = report.render_banner(passed=True, duration_s=10, calls=2)
    hi_block = (
        report.render_sample(
            report.build_sample(
                number=1, banner=banner, events=hi_events, checks=[], run_close_score="1/1"
            )
        )
        + "\n\n"
    )
    _write_run(tmp_path, manifest, [alpha, beta], {alpha.case_id: hi_block})  # beta: no transcript
    assert assemble_run_comment(tmp_path) == (
        # TWO cases, so there is something to roll up — at `##`, outranking their `###`.
        "## Eval run · `gpt-oss:20b`\n"
        "\n"
        "**no deterministic checks** — 2 cases · 4 samples · 0 excluded\n"
        "\n"
        "| measure | reading |\n"
        "|---|---|\n"
        "| commit | `beef1234` |\n"
        "| provider | `http://localhost:11434` |\n"
        "| embeddings | `embeddinggemma` |\n"
        "| cost / sample | 27,100 in · 2,950 out · 9.5 calls · 74s |\n"
        "| run | `run-20260720T090000Z-beef1234` |\n"
        "\n"
        "### `test_a.py::one` — alpha\n"
        "\n"
        "<details><summary>sample 1 — ✅ pass · 10s · 2 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "hi" |  |\n'
        "|---|---|---|\n"
        "| actual | 🤖 hey |  |\n"
        "\n"
        "</details>\n"
        "\n"
        "### `test_b.py::two` — beta\n"
        "\n"
        "_(no transcript recorded)_\n"
        "\n" + _footer(tmp_path)
    )


def test_diff_mode_flips_index_whole_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a baseline present, a check that was fully green there but failed a sample here adds a
    ``flips:`` index line to the run header (joined on ``(case_id, label)``)."""
    manifest = build_manifest(
        commit="abba710a03ae3555148fea6a86712e9af020499a",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=3,
        lever="framework baseline",
        now=datetime(2026, 7, 21, 5, 10, 17, tzinfo=UTC),
    )
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_chat_response.py::browse_answer",
        family="browse-answer",
        mean=0.67,
        all_pass_rate=2 / 3,
        pathology_excluded_mean=0.67,
        samples=3,
        sample_scores=[1.0, 1.0, 0.0],
        sample_causes=[None, None, FailureCause.HARNESS],
        sample_fragile=[False, False, False],
        cause_counts=CauseCounts(harness=1),
        checks=[CheckOutcome(label="browsed", passed=2, total=3, scored=True, cells=[_P, _P, _F])],
        timings=_TIMINGS,
        min_pass_rate=0.75,
        gate_metric="mean",
    )
    prior = tmp_path / "prior"
    prior.mkdir()
    prior_artifact = CaseArtifact(
        run_id="run-prior-cafe",
        case_id="test_chat_response.py::browse_answer",
        family="browse-answer",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=3,
        sample_scores=[1.0, 1.0, 1.0],
        sample_causes=[None, None, None],
        cause_counts=CauseCounts(),
        checks=[CheckOutcome(label="browsed", passed=3, total=3)],
        timings=_TIMINGS,
    )
    (prior / "results.jsonl").write_text(prior_artifact.model_dump_json() + "\n")
    monkeypatch.setenv("EVAL_BASELINE", str(prior))
    run = tmp_path / "run"
    _write_run(run, manifest, [artifact], {artifact.case_id: _browse_sample()})
    comment = assemble_run_comment(run)
    assert "**gate:** ⚖ 0.75 on mean → **❌ FAIL** (0.67)\nflips: browsed ✅→❌ (s3)\n" in comment
    assert comment.startswith("**gate:**"), "a single-case run leads with its gate, not a roll-up"


def _hold_run(
    report_dir: Path,
    prior_dir: Path,
    *,
    recorded_baseline: str | None,
) -> RunManifest:
    """Materialise the real ``idle-elicit-hold`` shape (#1752): a 10-sample classifier run whose
    scored ``decided idle`` check failed samples 7 and 10 (cells ``…P P F P P F``), diffed against a
    prior run where it was fully green (10/10). ``recorded_baseline`` is the manifest's durable
    baseline reference (``None`` reproduces a pre-#1752 manifest). Returns the run's manifest."""
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior = CaseArtifact(
        run_id="run-20260723T013634Z-9a034ca0",
        case_id="test_conversation_machine.py::idle-elicit-hold",
        family="state-classifier",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=10,
        sample_scores=[1.0] * 10,
        sample_causes=[None] * 10,
        cause_counts=CauseCounts(),
        checks=[CheckOutcome(label="decided idle", passed=10, total=10)],
        timings=_TIMINGS,
    )
    (prior_dir / "results.jsonl").write_text(prior.model_dump_json() + "\n")
    manifest = build_manifest(
        commit="d1429159776f24c038c91e4ea5ffb00addbbabb3",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=10,
        lever="beat 2 baseline",
        now=datetime(2026, 7, 23, 2, 3, 47, tzinfo=UTC),
        baseline=recorded_baseline,
    )
    cells = [_P, _P, _P, _P, _P, _P, _F, _P, _P, _F]  # decided idle failed s7 + s10
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_conversation_machine.py::idle-elicit-hold",
        family="state-classifier",
        mean=0.8,
        all_pass_rate=0.8,
        pathology_excluded_mean=0.8,
        samples=10,
        sample_scores=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        sample_causes=[None] * 6 + [FailureCause.BEHAVIORAL, None, None, FailureCause.BEHAVIORAL],
        sample_fragile=[False] * 10,
        cause_counts=CauseCounts(behavioral=2),
        checks=[
            CheckOutcome(label="decided idle", passed=8, total=10, scored=True, cells=cells),
            CheckOutcome(
                label="draw well-formed (tagged, in-union)",
                passed=10,
                total=10,
                scored=False,
                cells=[_P] * 10,
            ),
        ],
        timings=_TIMINGS,
        min_pass_rate=0.8,
        gate_metric="mean",
    )
    _write_run(report_dir, manifest, [artifact], {})
    return manifest


def test_flips_index_from_durable_manifest_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-header flips index resolves the baseline from the run's DURABLE manifest reference
    (``RunManifest.baseline``), so ``make assemble`` renders it with NO ``EVAL_BASELINE`` in the
    environment — the exact divergence that dropped the flips line on the real run while the per-row
    REGRESSED badges (baked into the transcripts at eval time) stayed (#1752). Reconstructed from
    the real ``idle-elicit-hold`` shape: ``decided idle`` was fully green in the baseline and failed
    s7/s10 here, so the header carries its ``flips`` index from durable state."""
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    prior = tmp_path / "prior"
    run = tmp_path / "run"
    manifest = _hold_run(run, prior, recorded_baseline=str(prior))
    comment = assemble_run_comment(run)
    # The FLIPS index is what this test is about; the header's layout is pinned whole by the
    # two render tests above, and re-pinning it here would make them one change apart.
    assert "flips: decided idle ✅→❌ (s7, s10)" in comment
    assert "**gate:** ⚖ 0.8 on mean → **✅ PASS** (0.80)" in comment
    # A single-case run has no roll-up table for the run id to sit in; its identity lives in
    # the case's own table, which the per-case document renders.
    assert manifest.run_id


def test_flips_index_absent_without_a_baseline_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-fix real-run case: a manifest with no recorded baseline AND no ``EVAL_BASELINE`` in
    the environment is off-diff — no flips line, no error (#1752). This is the buggy state the
    durable reference cures, pinned so it can't silently return."""
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    prior = tmp_path / "prior"
    run = tmp_path / "run"
    _hold_run(run, prior, recorded_baseline=None)
    assert "flips:" not in assemble_run_comment(run)


def _fail_sample() -> str:
    """A one-sample failure block, as the ``.md`` writer stores it — folded whole (#1753)."""
    events = [
        report.Event(report.EventKind.USER, "add a reminder"),
        report.Event(report.EventKind.REPLY, "done!", thinking="skip it"),
    ]
    checks = [
        report.CheckView(
            "C1",
            "reminder set",
            "state",
            True,
            False,
            False,
            rationale="no cadence written",
            cause="behavioral",
            anchor_index=1,
        ),
    ]
    banner = report.render_banner(
        passed=False,
        cause="behavioral",
        duration_s=60,
        calls=3,
    )
    sample = report.build_sample(
        number=2, banner=banner, events=events, checks=checks, run_close_score="0/1"
    )
    return report.render_sample(sample)


_FAIL_SAMPLE_FOLDED = (
    "<details><summary>sample 2 — ❌ fail · behavioral · 60s · 3 calls</summary>\n"
    "\n"
    '| step 1 · 👤 | "add a reminder" | ❌ |\n'
    "|---|---|---|\n"
    "| expected | C1 [state]⚖ reminder set |  |\n"
    "| 💭 | skip it |  |\n"
    "| actual | 🤖 done! | ❌ C1 — no cadence written · behavioral |\n"
    "\n"
    "</details>"
)


def _mixed_run(
    tmp_path: Path,
    expand_samples: list[int] | None = None,
    standing_counts: dict[str, int] | None = None,
) -> tuple[RunManifest, CaseArtifact]:
    """A single-case run whose ``.md`` holds a clean-pass sample (1) then a failure (2), both folded
    whole on disk — the fixture the fold-every-sample + ``.md``-parity tests share."""
    manifest = build_manifest(
        commit="abba710a03ae3555148fea6a86712e9af020499a",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=2,
        lever="mixed run",
        now=datetime(2026, 7, 21, 5, 10, 17, tzinfo=UTC),
    )
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_chat_response.py::browse_answer",
        family="browse-answer",
        mean=0.5,
        all_pass_rate=0.5,
        pathology_excluded_mean=0.5,
        samples=2,
        expand_samples=expand_samples or [],
        standing_counts=standing_counts or {},
        sample_scores=[1.0, 0.0],
        sample_causes=[None, FailureCause.BEHAVIORAL],
        sample_fragile=[False, False],
        cause_counts=CauseCounts(behavioral=1),
        checks=[CheckOutcome(label="reminder set", passed=1, total=2, scored=True, cells=[_P, _F])],
        timings=_TIMINGS,
    )
    transcript = _browse_sample() + _fail_sample() + "\n\n"
    _write_run(tmp_path, manifest, [artifact], {artifact.case_id: transcript})
    return manifest, artifact


def test_every_sample_folds_whole_in_the_comment(tmp_path: Path) -> None:
    """The one and only rendering (#1759): EVERY sample — the clean pass AND the failure — renders
    its full folded body inside a collapsed ``<details>``. There is no banner-only / compact
    form."""
    _mixed_run(tmp_path)
    comment = assemble_run_comment(tmp_path)
    assert comment == (
        _BROWSE_SAMPLE_FOLDED + "\n\n" + _FAIL_SAMPLE_FOLDED + "\n\n" + _footer(tmp_path)
    )


def test_a_case_level_preamble_reaches_the_comment_verbatim(tmp_path: Path) -> None:
    """A ported case writes its three-section report above its samples (#1995), and the
    assembler carries it through unchanged — ONE rendering, on disk and in the comment, rather
    than two that can disagree.  It lands above the folds, which is where it is read from."""
    manifest, artifact = _mixed_run(tmp_path)
    sections = "#### `browse-answer` — end-state assertions, variance, harness\n\n**A.** …"
    path = tmp_path / f"{artifact.case_id}.md"
    header = render_manifest_header(manifest) + "\n"
    path.write_text(f"{header}{sections}\n\n{path.read_text()[len(header) :].lstrip()}")
    comment = assemble_run_comment(tmp_path)
    assert sections in comment
    assert comment.index(sections) < comment.index(_BROWSE_SAMPLE_FOLDED)
    assert _BROWSE_SAMPLE_FOLDED in comment


def test_md_and_comment_render_every_sample_identically(tmp_path: Path) -> None:
    """A case that nominated no sample reproduces every one of them: the on-disk ``.md`` holds
    every sample's full folded body, and the comment carries the same bodies. Only a case whose
    cohort named a representative is indexed rather than reproduced (below)."""
    _, artifact = _mixed_run(tmp_path)
    on_disk = (tmp_path / f"{artifact.case_id}.md").read_text()
    assert _BROWSE_SAMPLE_FOLDED in on_disk
    assert _BROWSE_SAMPLE_BANNER_ONLY not in on_disk
    comment = assemble_run_comment(tmp_path)
    assert _BROWSE_SAMPLE_FOLDED in comment
    assert _BROWSE_SAMPLE_BANNER_ONLY not in comment


def test_missing_manifest_raises_actionable(tmp_path: Path) -> None:
    """No ``manifest.json`` → a FileNotFoundError naming the fix (this isn't a completed run)."""
    with pytest.raises(FileNotFoundError) as excinfo:
        load_manifest(tmp_path)
    assert "manifest.json" in str(excinfo.value)
    assert "make eval" in str(excinfo.value)


def test_cli_writes_comment_and_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` writes the assembled comment to stdout on a good dir (exit 0); a missing arg is
    usage on stderr (exit 2), a dir with no manifest is the error on stderr (exit 1)."""
    manifest = build_manifest(
        commit="abba710a03ae",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=1,
        lever="ship it",
        now=datetime(2026, 7, 21, 5, 10, 17, tzinfo=UTC),
    )
    artifact = CaseArtifact(
        run_id=manifest.run_id,
        case_id="test_a.py::one",
        family="alpha",
        mean=1.0,
        all_pass_rate=1.0,
        pathology_excluded_mean=1.0,
        samples=1,
        sample_scores=[1.0],
        sample_causes=[None],
        sample_fragile=[False],
        cause_counts=CauseCounts(),
        checks=[],
        timings=_TIMINGS,
    )
    _write_run(tmp_path, manifest, [artifact], {})
    assert main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == assemble_run_comment(tmp_path)
    assert main([]) == 2
    assert capsys.readouterr().err.strip() == USAGE
    assert main([str(tmp_path / "does-not-exist")]) == 1
    assert "manifest.json" in capsys.readouterr().err


def test_only_the_nominated_sample_is_carried_in_full(tmp_path: Path) -> None:
    """The scaling fix (#1997): the artifact is the complete record and the comment is an INDEX
    into it, so a case that named a representative carries THAT sample and counts the rest on one
    line. Seventeen collapsed stubs each saying "not expanded here" is seventeen folds that say
    nothing, and at a hundred cases it is seventeen hundred."""
    _, artifact = _mixed_run(tmp_path, expand_samples=[1], standing_counts={"typical": 1})
    comment = assemble_run_comment(tmp_path)

    assert report.REPRESENTATIVE_LABEL in comment, "the carried sample is labelled as such"
    assert "deepest lake?" in comment, "and carried whole"
    assert "1 that matched it" in comment, "the rest are ACCOUNTED for on one line"
    assert "sample 2 — " not in comment, "not seventeen stubs"
    on_disk = (tmp_path / f"{artifact.case_id}.md").read_text()
    assert "sample 2 — " in on_disk, "the artifact keeps every sample regardless"


def test_a_long_thinking_trace_is_shortened_only_in_the_comment(tmp_path: Path) -> None:
    """Thinking was 68% of every sample — the single biggest lever — so the comment carries its
    head; the label already states the length, so shortening changes only the body. The trace is
    never touched on disk, and a SHORT trace is left alone: rewriting it would add nothing and
    cost bytes."""
    long_body = report.thinking_row("x" * 900).render()
    shortened = report.summarise_thinking(long_body)

    assert "thinking — 900 chars" in shortened, "the label states the real length either way"
    assert "x" * 900 not in shortened, "the trace itself is not restated in the index"
    assert len(shortened) < len(long_body) // 3
    short_body = report.thinking_row("brief").render()
    assert report.summarise_thinking(short_body) == short_body, "nothing to save, nothing rewritten"


def test_the_comment_carries_the_prompts_its_representative_was_run_with(tmp_path: Path) -> None:
    """`chat` has one distinct text PER sample — the self-state header feeds each its own minted
    names back — so carrying all of them means carrying the cohort many times over to show a text
    that differs in three lines. The comment keeps what every sample shared and what the sample it
    actually carries was given; the rest point at the artifact that holds them."""
    shared = report.PromptVariant(context="framer", text="F" * 400, samples=["sample 1"], total=1)
    mine = report.PromptVariant(context="chat", text="M" * 400, samples=["sample 1"], total=2)
    theirs = report.PromptVariant(context="chat", text="T" * 400, samples=["sample 2"], total=2)
    rendered = report.render_prompt_variants([shared, mine, theirs])

    elided = report.elide_unused_prompts(rendered, ["sample 1"])
    assert "F" * 400 in elided, "a prompt every sample shared is kept"
    assert "M" * 400 in elided, "and so is the one the carried sample was run with"
    assert "T" * 400 not in elided, "another sample's wording is not restated in the index"
    assert "other 1 samples" in elided
    assert "`chat`" in elided, "and the line names the context it dropped"


def test_the_samples_the_comment_does_not_carry_are_accounted_for() -> None:
    """An accounting, never a claim that the samples it left out AGREED — on a variant cohort
    that claim is false, and it renders directly above the blocks showing how they differed.

    The arithmetic closes the way the summary line's does: representative + matched +
    diverged = pooled."""
    consistent = report.samples_accounted(matched=12, diverged=2)
    assert "Of 15 pooled samples" in consistent
    assert "12 that matched it" in consistent and "2 that diverged" in consistent

    # The end that mattered: when nothing agreed, say so rather than report it as a small number.
    variant = report.samples_accounted(matched=0, diverged=14)
    assert "**No pooled sample matched the representative**" in variant
    assert "all 14 of the others diverged" in variant
    assert "agreed" not in variant
