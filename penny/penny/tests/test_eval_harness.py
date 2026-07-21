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

# Importing the memory-tools module registers those tools (``Tool.__init_subclass__``) so
# ``Tool.format_result`` dispatches their real ``to_result_narration`` — the rejection-probe
# tests below build frames from the PRODUCTION templates, never hand-invented text.
import penny.tools.memory_tools  # noqa: F401  (imported for registration side effect)
from penny.database import Database
from penny.tests.eval.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckOutcome,
    FailureCause,
)
from penny.tests.eval.baseline import load_baseline
from penny.tests.eval.conftest import (
    Check,
    SampleResult,
    _assert_threshold,
    _bail_fired_check,
    _cycle_recovered_check,
    _frame_attributes_to,
    _guarded_graded,
    _scorer_is_graded,
    _stamp_cause,
    _write_sample_report,
    run_exhibited_pathology,
    sample_is_fragile,
    tool_call_rejected,
    tool_not_called,
    tool_was_called,
)
from penny.tools.base import FRAMEWORK_NARRATION_INVALID_ARGS, Tool
from penny.tools.models import ToolResult


def _make_db(tmp_path, name: str = "harness") -> Database:
    db = Database(str(tmp_path / f"{name}.db"))
    db.create_tables()
    return db


def _log_prompt(db: Database, *, messages=None, response=None, thinking=None) -> None:
    db.messages.log_prompt(
        model="test-model",
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        response=response if response is not None else {},
        thinking=thinking,
        run_id="r1",
    )


def _tool_call_response(name: str, arguments: str = "{}") -> dict:
    call = {"function": {"name": name, "arguments": arguments}}
    return {"choices": [{"message": {"tool_calls": [call]}}]}


def _content_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _tool_frame(content: str) -> list[dict]:
    return [{"role": "tool", "content": content}]


def _framed_result(
    tool_name: str,
    arguments: dict,
    *,
    ok: bool,
    mutated: bool = False,
    narration: str | None = None,
) -> list[dict]:
    """A tool-role turn carrying the REAL production frame ``Tool.format_result`` emits for a
    call to ``tool_name`` — the registry-dispatched narration + the ``(<tool> result)`` tag +
    body — so the rejection probe is tested against the shapes it must actually recognise,
    never hand-invented text (#1726)."""
    result = ToolResult(message="body", success=ok, mutated=mutated, narration=narration)
    return _tool_frame(Tool.format_result(tool_name, arguments, result))


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


def test_tool_call_rejected_matches_backticked_tool_name_form(tmp_path) -> None:
    # The framework arg-validation failure leads with the backticked TOOL name
    # (`FRAMEWORK_NARRATION_INVALID_ARGS`) — the shape the per-tool probe already matched.
    db = _make_db(tmp_path)
    frame = _framed_result(
        "update_entry",
        {"memory": "trip-notes", "key": "hotel"},
        ok=False,
        narration=FRAMEWORK_NARRATION_INVALID_ARGS.format(tool_name="update_entry"),
    )
    assert "`update_entry`" in frame[0]["content"]  # the tool name IS backticked in this form
    _log_prompt(db, messages=frame)
    assert tool_call_rejected(db, "update_entry")
    assert tool_call_rejected(db)  # any-tool probe
    assert not tool_call_rejected(db, "collection_write")


def test_tool_call_rejected_matches_memory_tool_target_backticked_form(tmp_path) -> None:
    # A memory-tool execute-time failure backticks the TARGET, not the tool — the tool is
    # named only in the `(<tool> result)` tag.  Before #1726 a per-tool probe matched solely
    # the backticked tool name and went blind to these, false-greening every memory-surface
    # rejection check.  Frames are the PRODUCTION templates (via `Tool.format_result`).
    db = _make_db(tmp_path)
    write_frame = _framed_result("collection_write", {"memory": "trip-notes"}, ok=False)
    update_frame = _framed_result(
        "update_entry", {"memory": "trip-notes", "key": "hotel"}, ok=False
    )
    _log_prompt(db, messages=write_frame)
    _log_prompt(db, messages=update_frame)

    # The bug's signature: the tool name is NOT backticked — only the tag names it.
    assert "`collection_write`" not in write_frame[0]["content"]
    assert "(collection_write result)" in write_frame[0]["content"]
    assert "`update_entry`" not in update_frame[0]["content"]
    assert "(update_entry result)" in update_frame[0]["content"]

    # The fix: attributed by the tag, each rejection is visible to its per-tool probe again.
    assert tool_call_rejected(db, "collection_write")
    assert tool_call_rejected(db, "update_entry")
    assert tool_call_rejected(db)  # any-tool probe
    assert not tool_call_rejected(db, "log_append")  # a tag names exactly one tool

    # The attribution primitive recognises the tag shape and never cross-attributes.
    assert _frame_attributes_to(write_frame[0]["content"], "collection_write")
    assert not _frame_attributes_to(write_frame[0]["content"], "update_entry")


def test_sample_is_fragile_detects_recovery_frames(tmp_path) -> None:
    db = _make_db(tmp_path)
    _log_prompt(
        db,
        messages=_framed_result(
            "collection_write",
            {"memory": "trip-notes", "entries": [{"key": "hotel"}]},
            ok=True,
            mutated=True,
        ),
    )
    assert not sample_is_fragile(db)
    # A memory-tool target-backticked rejection is a recovery frame too: `sample_is_fragile`
    # filters no tool name, so its `_RECOVERY_FRAMES` set catches "didn't work" regardless of
    # which (target-backticked) tool produced it — no attribution gap here (#1726 audit).
    _log_prompt(db, messages=_framed_result("collection_write", {"memory": "trip-notes"}, ok=False))
    assert sample_is_fragile(db)


# ── The graded runner paths: dispatch + framework guard-as-Check (#1697) ──


def test_scorer_is_graded_dispatches_on_return_type() -> None:
    # A graded scorer returns Checks; a binary one returns failure strings; empty → binary (pass).
    assert _scorer_is_graded([Check("wrote entry", ok=True)])
    assert not _scorer_is_graded(["did not write the entry"])
    assert not _scorer_is_graded([])


def test_bail_fired_and_cycle_recovered_guard_checks() -> None:
    # Each guard is a scored Check: it passes silently (no rationale) when the contract fired, and
    # fails with a rationale naming the vacuous contract when it did not — so a run the injected
    # trigger never reached can't score green off the scorer's own checks alone.
    fired = _bail_fired_check(True)
    assert fired.ok and fired.scored and fired.rationale is None
    missed = _bail_fired_check(False)
    assert not missed.ok and missed.rationale is not None
    recovered = _cycle_recovered_check(True)
    assert recovered.ok and recovered.rationale is None
    stalled = _cycle_recovered_check(False)
    assert not stalled.ok and stalled.rationale is not None


def test_guarded_graded_prepends_guard_and_gates_a_vacuous_contract() -> None:
    # A scorer whose own check PASSES but whose injected bail never fired: the prepended guard
    # (leading the list) drags the sample below a full pass — the vacuous-contract catch.
    vacuous = _guarded_graded([Check("wrote the entry", ok=True)], [_bail_fired_check(False)])
    assert vacuous.total == 2  # guard + scorer check, both scored
    assert vacuous.score == 0.5 and not vacuous.passed
    assert vacuous.checks[0].label == "forced bail fired — contract exercised"  # guard leads
    # With the bail fired, the same scorer sample is a clean full pass.
    clean = _guarded_graded([Check("wrote the entry", ok=True)], [_bail_fired_check(True)])
    assert clean.passed and clean.total == 2


def test_guarded_graded_no_guards_is_the_startup_peripheral_path() -> None:
    # startup_eval (and the peripheral / prompt-format runners) dispatch with NO framework
    # guards — no injection — so _guarded_graded(scored, []) grades purely over the scorer's
    # own Checks.  A 2-of-3 graded text scorer scores 0.67 where the old binary scorer scored
    # 0.0 on the same miss: the monotonicity the conversion buys (graded mean >= binary mean).
    result = _guarded_graded(
        [Check("generated", ok=True), Check("length", ok=True), Check("voice", ok=False)], []
    )
    assert result.total == 3
    assert round(result.score, 2) == 0.67
    assert not result.passed
    assert result.failed == ["voice"]
    # A clean all-pass graded text scorer is a full pass, and a binary text scorer's failure
    # strings still route through the binary path (a text scorer that returns strings).
    assert _guarded_graded([Check("only", ok=True)], []).passed
    assert not _scorer_is_graded(["fell back to the canned message"])


def test_report_renders_injected_guard_check_in_footer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save X"},
            {"role": "assistant", "tool_calls": [write_call]},
        ],
    )
    # The scorer's own check passed (anchored to the write row), but the injected bail-fired guard
    # failed — so the guard-as-Check lands in the footer with its vacuous-contract rationale.
    result = _guarded_graded(
        [Check("wrote the entry", ok=True, anchor="collection_write(")],
        [_bail_fired_check(False)],
    )
    _write_sample_report(db, "guard-case", 0, result=result, reply="saved")
    expected = (
        "#### sample 1 — ❌ 1/2 checks\n"
        "\n"
        "| # | Actor | Content |\n"
        "|---|---|---|\n"
        "| 1 | 👤 user | save X |\n"
        "| 2 | 🔧 Penny → tool ✅ | collection_write({}) |\n"
        "| 3 | 🤖 Penny | saved |\n"
        "\n"
        "_checks: ❌ forced bail fired — contract exercised — "
        "the injected bail never fired — the recovery contract was not exercised_\n"
        "\n"
    )
    assert (tmp_path / "guard-case.md").read_text() == expected


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


# ── Honest-threshold restoration: gate on the pathology-excluded mean (#1698) ──


def test_gate_pathology_excluded_gates_on_the_honest_mean(capsys) -> None:
    # One clean pass + one pathology failure: the raw mean is 0.50, but the pathology sample
    # drops out of the pathology-excluded denominator, so the honest read is 1.00.  Opting in
    # (gate_pathology_excluded=True) gates on that honest 1.00 and clears an 0.8 bar the raw
    # mean would miss — the mechanism behind the speakable sequence cases' 0.6→0.8 restore.
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["collapse"])
    pathological.cause = FailureCause.PATHOLOGY
    _assert_threshold("honest-case", [passed, pathological], 0.8, gate_pathology_excluded=True)
    out = capsys.readouterr().out
    assert (
        "RESULT [honest-case] mean 0.50 · all-pass 1/2 across 2 samples "
        "(need pathology-excluded mean >=0.8)" in out
    )


def test_gate_pathology_excluded_still_fails_on_a_behavioral_miss() -> None:
    # A BEHAVIORAL failure stays in the pathology-excluded denominator, so the honest mean is
    # 0.50 — the opt-in gate is not a free pass; only reroll-guard pathology noise is excluded.
    passed = SampleResult.binary([])
    behavioral = SampleResult.binary(["wrong end state"])
    behavioral.cause = FailureCause.BEHAVIORAL
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold("behav-case", [passed, behavioral], 0.8, gate_pathology_excluded=True)


def test_pathology_noise_sinks_the_raw_gate_without_the_opt_in() -> None:
    # The flag is load-bearing: the SAME clean-pass + pathology-fail pair FAILS the default
    # raw-mean gate (0.50 < 0.8) — exactly the flake the honest-threshold restoration removes
    # by opting the case into the pathology-excluded gate above.
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["collapse"])
    pathological.cause = FailureCause.PATHOLOGY
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold("raw-gate-case", [passed, pathological], 0.8)


# ── Failure-cause partition (#1695): the structural pathology scan + stamping ──


def test_run_exhibited_pathology_detects_reroll_guard_signals(tmp_path) -> None:
    # Each of the four reroll-guard conditions the loop discards + re-rolls on, read
    # post-hoc off the persisted RESPONSE (the same text_validity detectors run live).
    degenerate = _make_db(tmp_path, "degen")
    _log_prompt(degenerate, response=_content_response("winter watering......???"))
    assert run_exhibited_pathology(degenerate)  # DEGENERATE_OUTPUT in content

    harmony = _make_db(tmp_path, "harmony")
    _log_prompt(harmony, response=_content_response("leaked <|call|> to=functions.browse"))
    assert run_exhibited_pathology(harmony)  # TOOL_CALL_LEAK

    fragment = _make_db(tmp_path, "fragment")
    _log_prompt(fragment, response=_content_response('{"memory": "notes"}'))
    assert run_exhibited_pathology(fragment)  # bare call-fragment reply (no tool calls)

    bad_name = _make_db(tmp_path, "name")
    _log_prompt(bad_name, response=_tool_call_response("Functions?????"))
    assert run_exhibited_pathology(bad_name)  # collapse-shaped tool NAME

    poison_arg = _make_db(tmp_path, "arg")
    _log_prompt(
        poison_arg, response=_tool_call_response("collection_write", '{"content": "..???.."}')
    )
    assert run_exhibited_pathology(poison_arg)  # collapse in a serialised tool-call argument


def test_run_exhibited_pathology_ignores_clean_and_input_only_poison(tmp_path) -> None:
    # A healthy run — a real tool call + a clean reply — carries no pathology signal.
    clean = _make_db(tmp_path, "clean")
    _log_prompt(clean, response=_tool_call_response("collection_write"))
    _log_prompt(clean, response=_content_response("Here's your answer."))
    assert not run_exhibited_pathology(clean)
    # Poison in the INPUT messages (e.g. an injected bail echoed into history) is NOT the
    # model's output — the scan reads only the response, so an injected trigger stays invisible.
    injected = _make_db(tmp_path, "injected")
    _log_prompt(
        injected,
        messages=[{"role": "assistant", "content": "Hi there! ......???"}],
        response=_tool_call_response("collection_write"),
    )
    assert not run_exhibited_pathology(injected)


def test_stamp_cause_partitions_pass_pathology_harness_behavioral(tmp_path) -> None:
    # Pass → no cause, regardless of the DB.
    passed_db = _make_db(tmp_path, "pass")
    passed = SampleResult.binary([])
    _stamp_cause(passed_db, passed)
    assert passed.cause is None

    # Failed + poison in the response → pathology.
    poison_db = _make_db(tmp_path, "poison")
    _log_prompt(poison_db, response=_content_response("collapse...???"))
    pathological = SampleResult.binary(["wrong end state"])
    _stamp_cause(poison_db, pathological)
    assert pathological.cause == FailureCause.PATHOLOGY

    # Failed, clean output → behavioral (the model simply got it wrong).
    clean_db = _make_db(tmp_path, "behav")
    _log_prompt(clean_db, response=_content_response("A confident but wrong answer."))
    behavioral = SampleResult.binary(["wrong end state"])
    _stamp_cause(clean_db, behavioral)
    assert behavioral.cause == FailureCause.BEHAVIORAL

    # Timeout on a clean DB → harness; but poison outranks the timeout symptom.
    timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(clean_db, timeout, timed_out=True)
    assert timeout.cause == FailureCause.HARNESS
    poison_timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(poison_db, poison_timeout, timed_out=True)
    assert poison_timeout.cause == FailureCause.PATHOLOGY


def test_result_line_renders_cause_summary(capsys) -> None:
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["poison"])
    pathological.cause = FailureCause.PATHOLOGY
    _assert_threshold("cause-case", [passed, pathological], None)
    out = capsys.readouterr().out
    assert "RESULT [cause-case] mean 0.50 · all-pass 1/2 across 2 samples (report-only)" in out
    # The pathology sample drops out of the excluded denominator, so the honest read is 1.00.
    assert (
        "  pathology-excluded mean 1.00 (1 samples) · "
        "causes — behavioral 0 · pathology 1 · harness 0" in out
    )


# ── Regression diff: a prior run's results.jsonl → REGRESSED marks (#1693) ──

_BASELINE_RUN_ID = "run-20260719T130500-a1b2c3d4"


def _write_baseline(directory, *, case_id: str, checks: list[CheckOutcome]) -> None:
    """Write a one-case ``results.jsonl`` — the prior run the report diffs against."""
    directory.mkdir(parents=True, exist_ok=True)
    artifact = CaseArtifact(
        run_id=_BASELINE_RUN_ID,
        case_id=case_id,
        family="extractors",
        mean=1.0,
        all_pass_rate=1.0,
        samples=4,
        sample_scores=[1.0, 1.0, 1.0, 1.0],
        checks=checks,
        # An all-green prior run (#1695 fields): every sample passed, so no causes and a
        # pathology-excluded mean equal to the raw mean.
        pathology_excluded_mean=1.0,
        sample_causes=[None, None, None, None],
        cause_counts=CauseCounts(),
        timings=CaseTimings(calls=0, duration_ms=0, input_tokens=0, output_tokens=0),
    )
    (directory / "results.jsonl").write_text(artifact.model_dump_json() + "\n")


def test_baseline_flags_only_a_fully_green_flip(tmp_path) -> None:
    _write_baseline(
        tmp_path / "prior",
        case_id="watch-fern",
        checks=[
            CheckOutcome(
                label="send queued", passed=4, total=4
            ),  # fully green → a flip if it fails
            CheckOutcome(label="write happened", passed=2, total=4),  # already flaky → not a flip
        ],
    )
    baseline = load_baseline(str(tmp_path / "prior"))
    assert baseline is not None
    assert baseline.was_passing("watch-fern", "send queued")
    assert not baseline.was_passing("watch-fern", "write happened")  # 2/4 was not fully green
    assert not baseline.was_passing("watch-fern", "unknown check")  # absent → no flip
    assert not baseline.was_passing("other-case", "send queued")  # absent case → no flip
    assert baseline.run_id_for("watch-fern") == _BASELINE_RUN_ID


def test_baseline_absent_or_empty_is_none(tmp_path) -> None:
    assert load_baseline(str(tmp_path / "does-not-exist")) is None  # missing → graceful None
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "results.jsonl").write_text("\n")  # blank lines only
    assert load_baseline(str(tmp_path / "empty")) is None


def _done_bail_sample(db: Database) -> None:
    """A collector-style run that closed with ``done()`` instead of sending — the promptlog row
    carries the model's thinking, so a failed/regressed done turn can surface it."""
    done_call = {"function": {"name": "done", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "run the fern watch"},
            {"role": "assistant", "tool_calls": [done_call]},
        ],
        response={"choices": [{"message": {"tool_calls": [done_call]}}]},
        thinking="The entry is already written, so I'll close with done() rather than notify.",
    )


def test_report_marks_regressed_and_renders_thinking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_BASELINE", str(tmp_path / "prior"))
    _write_baseline(
        tmp_path / "prior",
        case_id="watch-fern",
        checks=[CheckOutcome(label="send queued", passed=4, total=4)],
    )
    db = _make_db(tmp_path)
    _done_bail_sample(db)
    result = SampleResult.graded(
        [Check("send queued", ok=False, anchor="done(", rationale="expected 1 send, saw 0")]
    )
    _write_sample_report(db, "watch-fern", 2, result=result, reply="")
    expected = (
        "#### sample 3 — ❌ 0/1 checks\n"
        "\n"
        "| # | Actor | Content |\n"
        "|---|---|---|\n"
        "| 1 | 👤 user | run the fern watch |\n"
        "| 2 | 🔧 Penny → tool ❌ 🔻 REGRESSED | done({}) |\n"
        "\n"
        "_checks: ❌ 🔻 REGRESSED send queued — expected 1 send, saw 0 "
        "(was passing in `run-20260719T130500-a1b2c3d4`)_\n"
        "\n"
        "<details><summary>💭 thinking · turn 2 (done) — ❌ 🔻 REGRESSED</summary>\n"
        "\n"
        "> The entry is already written, so I'll close with done() rather than notify.\n"
        "\n"
        "</details>\n"
        "\n"
    )
    assert (tmp_path / "watch-fern.md").read_text() == expected


def test_report_no_baseline_plain_fail_still_shows_thinking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)  # first run — nothing to flip against
    db = _make_db(tmp_path)
    _done_bail_sample(db)
    result = SampleResult.graded(
        [Check("send queued", ok=False, anchor="done(", rationale="expected 1 send, saw 0")]
    )
    _write_sample_report(db, "watch-fern", 0, result=result, reply="")
    expected = (
        "#### sample 1 — ❌ 0/1 checks\n"
        "\n"
        "| # | Actor | Content |\n"
        "|---|---|---|\n"
        "| 1 | 👤 user | run the fern watch |\n"
        "| 2 | 🔧 Penny → tool ❌ | done({}) |\n"
        "\n"
        "_checks: ❌ send queued — expected 1 send, saw 0_\n"
        "\n"
        "<details><summary>💭 thinking · turn 2 (done) — ❌</summary>\n"
        "\n"
        "> The entry is already written, so I'll close with done() rather than notify.\n"
        "\n"
        "</details>\n"
        "\n"
    )
    assert (tmp_path / "watch-fern.md").read_text() == expected


def test_report_omits_thinking_at_a_passing_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save it"},
            {"role": "assistant", "tool_calls": [write_call]},
        ],
        response={"choices": [{"message": {"tool_calls": [write_call]}}]},
        thinking="Writing the entry now.",
    )
    result = SampleResult.graded([Check("write happened", ok=True, anchor="collection_write(")])
    _write_sample_report(db, "pass-case", 0, result=result, reply="done")
    # Whole-render: a passing tool-call turn carries its ✅ row-mark and NOTHING else — no
    # thinking <details> (even though the row has thinking) and no REGRESSED, so the comment
    # doesn't bloat on clean passes.
    expected = (
        "#### sample 1 — ✅ 1/1 checks\n"
        "\n"
        "| # | Actor | Content |\n"
        "|---|---|---|\n"
        "| 1 | 👤 user | save it |\n"
        "| 2 | 🔧 Penny → tool ✅ | collection_write({}) |\n"
        "| 3 | 🤖 Penny | done |\n"
        "\n"
    )
    assert (tmp_path / "pass-case.md").read_text() == expected
