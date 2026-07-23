"""Whole-render tests for the run-comment assembler (``assemble.py``, v3 / #1725).

NOT eval-marked — they drive the deterministic assembler over a SYNTHETIC report directory
(manifest + results.jsonl + per-case ``.md`` transcripts rendered by ``report.py``), so they run
inside ``make check``: no git, no model, no container. The assembled comment is asserted as a
WHOLE-RENDER literal (pr-review-guide §6): the run header (identity · RESULT · gate), the
compact-by-default per-sample transcript (#1753 — clean passes banner-only, failures collapsed),
the ``--full`` everything-in form, the ``.md``-vs-comment divergence, the multi-family rollup +
per-case headings, the diff-mode flips index, the local-artifacts footer, and the CLI contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from penny.tests.eval import report
from penny.tests.eval.artifacts import (
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
from penny.tests.eval.assemble import (
    FULL_FLAG,
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
    return (
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
    banner = report.render_banner(
        passed=True, score=1.0, passed_checks=1, total_checks=1, duration_s=45, calls=8
    )
    sample = report.build_sample(
        number=1, banner=banner, events=events, checks=checks, run_close_score="1/1"
    )
    return report.render_sample(sample) + "\n\n"


_BROWSE_SAMPLE_FOLDED = (
    "<details><summary>sample 1 — ✅ pass · 1/1 (1.00) · 45s · 8 calls</summary>\n"
    "\n"
    '| step 1 · 👤 | "deepest lake?" | ✅ |\n'
    "|---|---|---|\n"
    "| expected | C1 [spine]⚖ browsed |  |\n"
    "| 💭 | <details><summary>thinking</summary>verify</details> |  |\n"
    "| actual | 🔧 browse({...}) | ✅ C1 |\n"
    "| 💭 | <details><summary>thinking</summary>answer</details> |  |\n"
    "| actual | 🤖 Baikal 1642m |  |\n"
    "\n"
    "</details>"
)
_BROWSE_SAMPLE_BANNER = "#### sample 1 — ✅ pass · 1/1 (1.00) · 45s · 8 calls"


def test_single_gated_case_whole_render(tmp_path: Path) -> None:
    """A single gated case: the run header (identity · RESULT with timings · a gate line), the
    COMPACT clean-pass sample (banner line only — its body stays in the ``.md``), and the footer —
    no per-case heading (single case)."""
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
        "**run-20260721T051017Z-abba710a** · commit `abba710a` · gpt-oss:20b · N=3 · "
        "**lever:** framework baseline\n"
        "**RESULT:** mean 0.67 · all-pass 2/3 · pathology-excluded 0.67 · causes — behavioral 0 · "
        "pathology 0 · harness 1 · families: browse-answer 0.67 · 19 calls · 148s · "
        "54.2K in / 5.9K out\n"
        "**gate:** ⚖ 0.75 on mean → **❌ FAIL** (0.67)\n"
        "\n" + _BROWSE_SAMPLE_BANNER + "\n"
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
    banner = report.render_banner(
        passed=True, score=1.0, passed_checks=1, total_checks=1, duration_s=10, calls=2
    )
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
        "**run-20260720T090000Z-beef1234** · commit `beef1234` · gpt-oss:20b · N=2 · "
        "**lever:** two families\n"
        "**RESULT:** mean 0.75 · all-pass 3/4 · pathology-excluded 0.75 · causes — behavioral 1 · "
        "pathology 0 · harness 0 · families: alpha 1.00 · beta 0.50 · 38 calls · 296s · "
        "108.4K in / 11.8K out\n"
        "\n"
        "### `test_a.py::one` — alpha\n"
        "\n"
        "#### sample 1 — ✅ pass · 1/1 (1.00) · 10s · 2 calls\n"
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
    assert comment.startswith("**run-20260721T051017Z-abba710a** · commit `abba710a`")


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
    header = assemble_run_comment(run).split("\n\n", 1)[0]
    assert header == (
        f"**{manifest.run_id}** · commit `d1429159` · gpt-oss:20b · N=10 · "
        "**lever:** beat 2 baseline\n"
        "**RESULT:** mean 0.80 · all-pass 8/10 · pathology-excluded 0.80 · "
        "causes — behavioral 2 · pathology 0 · harness 0 · families: state-classifier 0.80 · "
        "19 calls · 148s · 54.2K in / 5.9K out\n"
        "**gate:** ⚖ 0.8 on mean → **✅ PASS** (0.80)\n"
        "flips: decided idle ✅→❌ (s7, s10)"
    )


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
        score=0.0,
        passed_checks=0,
        total_checks=1,
        cause="behavioral",
        duration_s=60,
        calls=3,
    )
    sample = report.build_sample(
        number=2, banner=banner, events=events, checks=checks, run_close_score="0/1"
    )
    return report.render_sample(sample)


_FAIL_SAMPLE_FOLDED = (
    "<details><summary>sample 2 — ❌ fail · 0/1 (0.00) · behavioral · 60s · 3 calls</summary>\n"
    "\n"
    '| step 1 · 👤 | "add a reminder" | ❌ |\n'
    "|---|---|---|\n"
    "| expected | C1 [state]⚖ reminder set |  |\n"
    "| 💭 | <details><summary>thinking</summary>skip it</details> |  |\n"
    "| actual | 🤖 done! | ❌ C1 — no cadence written · behavioral |\n"
    "\n"
    "</details>"
)


def _mixed_run(tmp_path: Path) -> tuple[RunManifest, CaseArtifact]:
    """A single-case run whose ``.md`` holds a clean-pass sample (1) then a failure (2), both folded
    whole on disk — the fixture the compact/full/divergence tests share."""
    manifest = build_manifest(
        commit="abba710a03ae3555148fea6a86712e9af020499a",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=2,
        lever="compact mode",
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


def test_compact_comment_banner_only_pass_and_collapsed_failure(tmp_path: Path) -> None:
    """The default compact comment (#1753): the clean-pass sample renders its banner line ONLY (no
    body), while the failure keeps its full step table inside a collapsed ``<details>``."""
    _mixed_run(tmp_path)
    comment = assemble_run_comment(tmp_path)  # compact is the default
    assert comment == (
        "**run-20260721T051017Z-abba710a** · commit `abba710a` · gpt-oss:20b · N=2 · "
        "**lever:** compact mode\n"
        "**RESULT:** mean 0.50 · all-pass 1/2 · pathology-excluded 0.50 · causes — behavioral 1 · "
        "pathology 0 · harness 0 · families: browse-answer 0.50 · 19 calls · 148s · "
        "54.2K in / 5.9K out\n"
        "\n" + _BROWSE_SAMPLE_BANNER + "\n"
        "\n" + _FAIL_SAMPLE_FOLDED + "\n"
        "\n" + _footer(tmp_path)
    )


def test_full_mode_emits_every_sample_body(tmp_path: Path) -> None:
    """``full=True`` (CLI ``--full`` / ``make assemble EVAL_FULL=1``) is the everything-in form: the
    same clean-pass sample now renders its full folded body, identical to the on-disk ``.md``."""
    _mixed_run(tmp_path)
    comment = assemble_run_comment(tmp_path, full=True)
    assert comment == (
        "**run-20260721T051017Z-abba710a** · commit `abba710a` · gpt-oss:20b · N=2 · "
        "**lever:** compact mode\n"
        "**RESULT:** mean 0.50 · all-pass 1/2 · pathology-excluded 0.50 · causes — behavioral 1 · "
        "pathology 0 · harness 0 · families: browse-answer 0.50 · 19 calls · 148s · "
        "54.2K in / 5.9K out\n"
        "\n" + _BROWSE_SAMPLE_FOLDED + "\n"
        "\n" + _FAIL_SAMPLE_FOLDED + "\n"
        "\n" + _footer(tmp_path)
    )


def test_md_keeps_full_body_while_compact_comment_diverges(tmp_path: Path) -> None:
    """The ``.md``-vs-comment divergence (#1753): the on-disk ``<case_id>.md`` keeps EVERY sample's
    full folded transcript (the footer's audit target), while the compact comment shows the
    clean-pass sample banner-only — the same source, two renderings, the ``--full`` comment
    matching the ``.md``."""
    _, artifact = _mixed_run(tmp_path)
    on_disk = (tmp_path / f"{artifact.case_id}.md").read_text()
    # The .md keeps the clean pass's full folded body, untouched by the compact comment.
    assert _BROWSE_SAMPLE_FOLDED in on_disk
    assert _BROWSE_SAMPLE_BANNER not in on_disk
    compact = assemble_run_comment(tmp_path)
    full = assemble_run_comment(tmp_path, full=True)
    assert _BROWSE_SAMPLE_BANNER in compact and _BROWSE_SAMPLE_FOLDED not in compact
    assert _BROWSE_SAMPLE_FOLDED in full  # --full round-trips the .md body into the comment


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


def test_cli_full_flag_routes_to_full_form(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--full`` (in any argv position) routes ``main`` to the everything-in form; without it the
    CLI emits the compact default (#1753)."""
    _mixed_run(tmp_path)
    assert main([FULL_FLAG, str(tmp_path)]) == 0
    assert capsys.readouterr().out == assemble_run_comment(tmp_path, full=True)
    assert main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == assemble_run_comment(tmp_path)  # compact default
