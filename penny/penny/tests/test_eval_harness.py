"""Deterministic tests for the eval-harness scoring + report machinery (issue #1694).

These drive the ``tests/eval/conftest.py`` scoring/report code directly with fixture
``Check`` / ``SampleResult`` data and a seeded promptlog — no live model, no ``eval`` marker —
so they run inside ``make check`` and pin the new ergonomics: check ``rationale``, the
not-applicable (``ignored``) third state, the fragile-pass verdict, the dual strict+partial
RESULT line, and the ``tool_not_called`` negative-constraint primitive.  Whole-render literal
assertions cover every new report shape.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    Check,
    SampleResult,
    _assert_threshold,
    _write_sample_report,
    sample_is_fragile,
    tool_call_rejected,
    tool_not_called,
    tool_was_called,
)


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "harness.db"))
    db.create_tables()
    return db


def _log_prompt(db: Database, *, messages=None, response=None) -> None:
    db.messages.log_prompt(
        model="test-model",
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        response=response if response is not None else {},
        run_id="r1",
    )


def _tool_call_response(name: str) -> dict:
    call = {"function": {"name": name, "arguments": "{}"}}
    return {"choices": [{"message": {"tool_calls": [call]}}]}


def _tool_frame(content: str) -> list[dict]:
    return [{"role": "tool", "content": content}]


# ── Scoring: the not-applicable (ignored) third state + rationale in failed labels ──


def test_graded_excludes_ignored_and_advisory_from_denominator() -> None:
    result = SampleResult.graded(
        [
            Check("state written", ok=True),
            Check("read count", ok=False),
            Check("routing clean", ok=False, scored=False),  # advisory: renders, doesn't count
            Check.na("browse branch", rationale="no browse this sample"),  # n/a: out of denom
        ]
    )
    assert result.total == 2  # only the two scored, applicable checks
    assert result.score == 0.5  # 1 of 2 scored checks passed — advisory doesn't move it
    assert not result.passed
    # An applicable failed check lands in ``failed`` whether or not it's scored (advisory
    # "routing clean" included); a not-applicable check never does.
    assert result.failed == ["read count", "routing clean"]
    assert len(result.checks) == 4  # every check preserved for the report


def test_graded_all_ignored_is_vacuous_pass() -> None:
    result = SampleResult.graded([Check.na("branch a"), Check.na("branch b")])
    assert result.total == 0
    assert result.score == 1.0
    assert result.passed
    assert result.failed == []


def test_graded_failed_label_carries_rationale() -> None:
    result = SampleResult.graded([Check("reads", ok=False, rationale="expected 3 reads, saw 1")])
    assert result.failed == ["reads — expected 3 reads, saw 1"]


def test_check_na_constructor() -> None:
    check = Check.na("browse branch", rationale="not exercised")
    assert check.ignored
    assert check.rationale == "not exercised"
    assert check.ok  # n/a is not a failure


# ── The negative-constraint primitive + the fragility scan ──


def test_tool_not_called_reads_the_promptlog(tmp_path) -> None:
    db = _make_db(tmp_path)
    _log_prompt(db, response=_tool_call_response("collection_write"))
    assert tool_was_called(db, "collection_write")
    assert not tool_not_called(db, "collection_write")
    assert tool_not_called(db, "send_message")


def test_tool_call_rejected_named_and_any(tmp_path) -> None:
    db = _make_db(tmp_path)
    _log_prompt(db, messages=_tool_frame("You tried to use `update_entry` but it didn't work: no"))
    assert tool_call_rejected(db, "update_entry")
    assert tool_call_rejected(db)  # any-tool probe
    assert not tool_call_rejected(db, "collection_write")


def test_sample_is_fragile_detects_recovery_frames(tmp_path) -> None:
    db = _make_db(tmp_path)
    _log_prompt(db, messages=_tool_frame("You used `browse` and here's the result: ok"))
    assert not sample_is_fragile(db)
    _log_prompt(db, messages=_tool_frame("You tried to use `browse` but it didn't work: down"))
    assert sample_is_fragile(db)


# ── Whole-render assertions for the new report shapes ──


def test_report_renders_rationale_and_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save X"},
            {"role": "assistant", "tool_calls": [write_call]},
        ],
    )
    result = SampleResult.graded(
        [
            Check("write happened", ok=True, anchor="collection_write("),
            Check("read count", ok=False, rationale="expected 3 reads, saw 1"),
            Check.na("browse branch", rationale="no browse this sample"),
        ]
    )
    _write_sample_report(db, "rationale-case", 0, result=result, reply="saved")
    expected = (
        "#### sample 1 — ❌ 1/2 checks · 1 n/a\n"
        "\n"
        "| # | Actor | Content |\n"
        "|---|---|---|\n"
        "| 1 | 👤 user | save X |\n"
        "| 2 | 🔧 Penny → tool ✅ | collection_write({}) |\n"
        "| 3 | 🤖 Penny | saved |\n"
        "\n"
        "_checks: ❌ read count — expected 3 reads, saw 1 · "
        "➖ browse branch — no browse this sample_\n"
        "\n"
    )
    assert (tmp_path / "rationale-case.md").read_text() == expected


def test_report_renders_passed_fragile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    db = _make_db(tmp_path)
    browse_call = {"function": {"name": "browse", "arguments": "{}"}}
    reject = "You tried to use `browse` but it didn't work: down"
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "tool_calls": [browse_call]},
            {"role": "tool", "content": reject},
        ],
    )
    _write_sample_report(db, "fragile-case", 0, result=SampleResult.binary([]), reply="found it")
    expected = (
        "#### sample 1 — ✅ PASS · fragile\n"
        "\n"
        "| # | Actor | Content |\n"
        "|---|---|---|\n"
        "| 1 | 👤 user | look it up |\n"
        "| 2 | 🔧 Penny → tool | browse({}) |\n"
        "| 3 | 📥 tool result | You tried to use `browse` but it didn't work: down |\n"
        "| 4 | 🤖 Penny | found it |\n"
        "\n"
    )
    assert (tmp_path / "fragile-case.md").read_text() == expected


# ── The dual strict+partial RESULT line ──


def test_result_line_reports_dual_metric(capsys) -> None:
    results = [
        SampleResult.graded([Check("a", ok=True), Check("b", ok=True)]),  # 1.0, all-pass
        SampleResult.graded([Check("a", ok=True), Check("b", ok=False)]),  # 0.5, not all-pass
    ]
    _assert_threshold("dual-case", results, None)
    out = capsys.readouterr().out
    assert "RESULT [dual-case] mean 0.75 · all-pass 1/2 across 2 samples (report-only)" in out


def test_result_line_detail_carries_rationale(capsys) -> None:
    _assert_threshold(
        "detail-case",
        [SampleResult.graded([Check("reads", ok=False, rationale="expected 3 reads, saw 1")])],
        None,
    )
    out = capsys.readouterr().out
    assert "RESULT [detail-case] mean 0.00 · all-pass 0/1 across 1 samples (report-only)" in out
    assert "  [1] 0.00 — reads — expected 3 reads, saw 1" in out


def test_result_line_gated_pass_names_mean_threshold(capsys) -> None:
    _assert_threshold("gate-case", [SampleResult.binary([]), SampleResult.binary([])], 0.75)
    out = capsys.readouterr().out
    assert "RESULT [gate-case] mean 1.00 · all-pass 2/2 across 2 samples (need mean >=0.75)" in out


def test_result_line_gate_fails_below_threshold() -> None:
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold("red-case", [SampleResult.binary(["boom"])], 0.75)
