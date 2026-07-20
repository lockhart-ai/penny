"""Mechanism tests for the per-run eval artifacts (``artifacts.py``).

These are NOT eval-marked — they drive the manifest / per-case JSONL / report-header
writers directly with synthetic fixture data (no git, no model, no container), so
they run inside ``make check``. Two synthetic runs produce mechanically diffable
JSONL; the report header is asserted whole; a report run with no lever fails fast.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from penny.tests.eval.artifacts import (
    DIRTY_DIFF_FILENAME,
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    CaseTimings,
    EvalRun,
    MissingLeverError,
    RunManifest,
    build_case_artifact,
    build_manifest,
    default_family,
    render_manifest_header,
    run_from_env,
)
from penny.tests.eval.conftest import Check, SampleResult

_NOW = datetime(2026, 7, 20, 0, 45, 12, tzinfo=UTC)
_COMMIT = "abcdef1234567890abcdef1234567890abcdef12"
_TIMINGS = CaseTimings(calls=18, duration_ms=214300, input_tokens=41230, output_tokens=5120)


def _manifest(*, dirty_diff: str = "diff body\n", lever: str = "test hypothesis") -> RunManifest:
    return build_manifest(
        commit=_COMMIT,
        dirty_diff=dirty_diff,
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=5,
        lever=lever,
        now=_NOW,
    )


def test_run_id_is_stamp_and_short_commit() -> None:
    assert _manifest().run_id == "run-20260720T004512Z-abcdef12"


def test_manifest_header_whole_render_dirty() -> None:
    assert render_manifest_header(_manifest()) == (
        """### run-20260720T004512Z-abcdef12

- commit: `abcdef1234567890abcdef1234567890abcdef12` (dirty)
- model: `gpt-oss:20b`
- N: 5
- lever: test hypothesis
"""
    )


def test_manifest_header_whole_render_clean() -> None:
    assert render_manifest_header(_manifest(dirty_diff="")) == (
        """### run-20260720T004512Z-abcdef12

- commit: `abcdef1234567890abcdef1234567890abcdef12`
- model: `gpt-oss:20b`
- N: 5
- lever: test hypothesis
"""
    )


def test_default_family_strips_test_prefix() -> None:
    assert default_family("penny.tests.eval.test_chat_response") == "chat_response"
    assert default_family("test_extractors") == "extractors"
    assert default_family("weird_module") == "weird_module"


def test_build_case_artifact_aggregates_scores_and_checks() -> None:
    results = [
        SampleResult.graded([Check("c1", ok=True), Check("c2", ok=True)]),
        SampleResult.graded([Check("c1", ok=True), Check("c2", ok=False)]),
    ]
    artifact = build_case_artifact(
        run_id="run-x", case_id="case-x", family="fam", results=results, timings=_TIMINGS
    )
    assert artifact.mean == 0.75  # (1.0 + 0.5) / 2
    assert artifact.all_pass_rate == 0.5  # one of two samples fully passed
    assert artifact.samples == 2
    assert artifact.sample_scores == [1.0, 0.5]
    assert [(c.label, c.passed, c.total) for c in artifact.checks] == [
        ("c1", 2, 2),
        ("c2", 1, 2),
    ]
    assert artifact.timings == _TIMINGS


def test_binary_case_records_empty_check_outcomes() -> None:
    results = [SampleResult.binary([]), SampleResult.binary(["a fail reason"])]
    artifact = build_case_artifact(
        run_id="run-x", case_id="bin", family="fam", results=results, timings=_TIMINGS
    )
    assert artifact.mean == 0.5
    assert artifact.all_pass_rate == 0.5
    assert artifact.sample_scores == [1.0, 0.0]
    assert artifact.checks == []


def test_run_from_env_off_report_returns_none() -> None:
    assert run_from_env({}) is None


def test_run_from_env_missing_lever_fails_fast(tmp_path: Path) -> None:
    base = {"EVAL_REPORT_DIR": str(tmp_path)}
    for env in (base, {**base, "EVAL_LEVER": "   "}):  # unset and whitespace-only both fail
        with pytest.raises(MissingLeverError) as excinfo:
            run_from_env(env)
        assert "EVAL_LEVER" in str(excinfo.value)
        assert "make eval" in str(excinfo.value)  # actionable: names the fix


def test_run_from_env_decodes_the_makefile_contract(tmp_path: Path) -> None:
    run = run_from_env(
        {
            "EVAL_REPORT_DIR": str(tmp_path),
            "EVAL_LEVER": "sharpen the done() nudge",
            "EVAL_COMMIT": _COMMIT,
            "EVAL_DIRTY_DIFF": "some diff",
            "EVAL_SAMPLES": "3",
            "LLM_MODEL": "some-model:1b",
            "LLM_EMBEDDING_MODEL": "some-embed",
        },
        now=_NOW,
    )
    assert run is not None
    assert run.manifest.lever == "sharpen the done() nudge"
    assert run.manifest.commit == _COMMIT
    assert run.manifest.dirty is True
    assert run.manifest.diff_file == DIRTY_DIFF_FILENAME
    assert run.manifest.model == "some-model:1b"
    assert run.manifest.embedding_model == "some-embed"
    assert run.manifest.samples == 3


def test_write_inputs_writes_manifest_and_verbatim_diff(tmp_path: Path) -> None:
    run = EvalRun(
        tmp_path,
        _manifest(dirty_diff="--- a\n+++ b\n@@ real diff @@\n"),
        "--- a\n+++ b\n@@ real diff @@\n",
    )
    run.write_inputs()
    manifest = RunManifest.model_validate_json((tmp_path / MANIFEST_FILENAME).read_text())
    assert manifest.run_id == run.manifest.run_id
    assert manifest.lever == "test hypothesis"
    assert (tmp_path / DIRTY_DIFF_FILENAME).read_text() == "--- a\n+++ b\n@@ real diff @@\n"


def test_clean_run_writes_no_diff_file(tmp_path: Path) -> None:
    run = EvalRun(tmp_path, _manifest(dirty_diff=""), "")
    run.write_inputs()
    assert not (tmp_path / DIRTY_DIFF_FILENAME).exists()
    assert (
        RunManifest.model_validate_json((tmp_path / MANIFEST_FILENAME).read_text()).dirty is False
    )


def test_case_header_stamped_once_atop_report(tmp_path: Path) -> None:
    run = EvalRun(tmp_path, _manifest(), "diff")
    run.write_case_header("some-case")
    run.write_case_header("some-case")  # idempotent — a second call must not re-stamp
    report = (tmp_path / "some-case.md").read_text()
    assert report == render_manifest_header(run.manifest) + "\n"


def test_two_runs_produce_mechanically_diffable_jsonl(tmp_path: Path) -> None:
    """The through-line: manifest says what changed going in, the JSONL diffs coming out."""
    # Two runs at distinct commits/times — the realistic before/after of one lever pull.
    manifest_a = build_manifest(
        commit=_COMMIT,
        dirty_diff="lever A diff",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=2,
        lever="lever A",
        now=_NOW,
    )
    manifest_b = build_manifest(
        commit="beef1234beef1234beef1234beef1234beef1234",
        dirty_diff="",
        model="gpt-oss:20b",
        embedding_model="embeddinggemma",
        samples=2,
        lever="lever B",
        now=datetime(2026, 7, 20, 0, 46, 13, tzinfo=UTC),
    )
    dir_a, dir_b = tmp_path / "run-a", tmp_path / "run-b"
    run_a = EvalRun(dir_a, manifest_a, "lever A diff")
    run_b = EvalRun(dir_b, manifest_b, "")
    run_a.write_inputs()
    run_b.write_inputs()

    # Same case, two runs — run B regresses c2 and c1 relative to run A.
    results_a = [
        SampleResult.graded([Check("c1", ok=True), Check("c2", ok=True)]),
        SampleResult.graded([Check("c1", ok=True), Check("c2", ok=False)]),
    ]
    results_b = [
        SampleResult.graded([Check("c1", ok=True), Check("c2", ok=False)]),
        SampleResult.graded([Check("c1", ok=False), Check("c2", ok=False)]),
    ]
    run_a.append_case(
        build_case_artifact(
            run_id=run_a.manifest.run_id,
            case_id="case-x",
            family="fam",
            results=results_a,
            timings=_TIMINGS,
        )
    )
    run_b.append_case(
        build_case_artifact(
            run_id=run_b.manifest.run_id,
            case_id="case-x",
            family="fam",
            results=results_b,
            timings=_TIMINGS,
        )
    )

    record_a = _only_record(dir_a)
    record_b = _only_record(dir_b)

    # Mechanically diffable: identical schema (keys + order), identical case identity.
    assert list(record_a) == list(record_b)
    assert record_a["case_id"] == record_b["case_id"] == "case-x"
    assert record_a["family"] == record_b["family"] == "fam"
    assert [c["label"] for c in record_a["checks"]] == [c["label"] for c in record_b["checks"]]

    # The regression is legible in the diff: the mean dropped and c1/c2 pass-counts fell.
    assert record_a["mean"] == 0.75 and record_b["mean"] == 0.25
    assert record_a["all_pass_rate"] == 0.5 and record_b["all_pass_rate"] == 0.0
    assert _passed(record_a, "c1") == 2 and _passed(record_b, "c1") == 1
    assert _passed(record_a, "c2") == 1 and _passed(record_b, "c2") == 0

    # Each run's own manifest is the input-side of that diff (distinct run ids join back).
    assert record_a["run_id"] != record_b["run_id"]
    assert _read_manifest(dir_a).lever == "lever A"
    assert _read_manifest(dir_b).lever == "lever B"


def _only_record(report_dir: Path) -> dict:
    lines = (report_dir / RESULTS_FILENAME).read_text().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def _passed(record: dict, label: str) -> int:
    return next(c["passed"] for c in record["checks"] if c["label"] == label)


def _read_manifest(report_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((report_dir / MANIFEST_FILENAME).read_text())
