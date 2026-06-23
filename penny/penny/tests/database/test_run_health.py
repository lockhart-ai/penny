"""Run-health classification — the shared signal Penny's quality collector and
the addon's prompts tab both read.

``classify_run`` and ``render_run_record`` are pure functions of a run's
``PromptLog`` rows, so these build rows directly (no DB) and assert the four
failure modes are flagged structurally: a bail (no work done), an incomplete run
(hit the step ceiling), a tool-failure spiral, and a half-formed send.  A healthy
worked run and a healthy quiet read carry no flags.
"""

import json

from penny.database.memory import classify_run, half_formed_send_reason, render_run_record
from penny.database.models import PromptLog
from penny.text_validity import is_unfinished_fragment


def _call(name: str, args: dict) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _prompt(
    calls: list[dict],
    *,
    outcome: str | None = None,
    reason: str | None = None,
    target: str = "games",
    tool_failures: int | None = None,
) -> PromptLog:
    """One promptlog row.  Only the run's LAST row carries outcome/tool_failures
    (that's how ``set_run_outcome`` stamps it)."""
    message: dict = {"role": "assistant", "content": "", "tool_calls": calls}
    return PromptLog(
        model="m",
        messages="[]",
        response=json.dumps({"choices": [{"message": message}]}),
        run_id="r",
        run_target=target,
        run_outcome=outcome,
        run_reason=reason,
        tool_failures=tool_failures,
    )


_DONE_OK = _call("done", {"success": True, "summary": "done"})


def test_bailed_run_is_flagged():
    """A no_work/failed run whose only call is done() did no work — bailed."""
    run = [_prompt([_DONE_OK], outcome="no_work", reason="nothing new")]
    health = classify_run(run)
    assert health.bailed is True
    assert health.flags == ["no_work_done"]
    assert health.regressive is True
    record = render_run_record(run)
    assert "⚠ NO WORK DONE" in record


def test_incomplete_run_is_flagged_and_shows_trace():
    """An incomplete run (work landed, never closed done) is surfaced with its trace."""
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([], outcome="incomplete", reason="max steps exceeded"),
    ]
    health = classify_run(run)
    assert health.incomplete is True
    assert health.bailed is False
    assert "incomplete" in health.flags
    record = render_run_record(run)
    assert "⚠ INCOMPLETE" in record
    assert "write(games" in record  # trace shown so the run can be judged


def test_tool_failure_count_is_flagged():
    """A run that hit tool failures and kept going is flagged with the count."""
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([_DONE_OK], outcome="worked", reason="wrote one", tool_failures=2),
    ]
    health = classify_run(run)
    assert health.tool_failures == 2
    assert "tool_failures" in health.flags
    assert "⚠ TOOL FAILURES (2)" in render_run_record(run)


def test_half_formed_send_is_flagged_on_a_worked_run():
    """The real notifier shape: a worked run that ALSO sent a half-formed message
    ("Hi there! ......???") before the real one.  The bad send is flagged."""
    run = [
        _prompt(
            [
                _call("send_message", {"content": "Hi there! ......???"}),
                _call(
                    "send_message", {"content": "Heads up — a new title dropped, details inside."}
                ),
                _DONE_OK,
            ],
            outcome="worked",
            reason="delivered a notification",
        )
    ]
    health = classify_run(run)
    assert health.degenerate_send is True
    assert "half_formed_send" in health.flags
    record = render_run_record(run)
    assert "⚠ HALF-FORMED SEND" in record
    assert "Hi there! ......???" in record  # the offending send shown, untruncated


def test_healthy_worked_run_has_no_flags():
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([_DONE_OK], outcome="worked", reason="wrote one", tool_failures=0),
    ]
    health = classify_run(run)
    assert health.flags == []
    assert health.regressive is False
    record = render_run_record(run)
    assert "⚠" not in record
    assert "write(games" in record


def test_healthy_quiet_read_is_not_a_bail():
    """A no_work run that DID read before done() is a healthy quiet cycle, not a
    bail — no flags, header-only (no trace to tempt an over-correction)."""
    run = [
        _prompt(
            [_call("log_read", {"memory": "user-messages"}), _DONE_OK],
            outcome="no_work",
            reason="nothing new",
        )
    ]
    health = classify_run(run)
    assert health.flags == []
    record = render_run_record(run)
    assert "⚠" not in record
    assert "log_read" not in record  # header-only


def test_unfinished_fragment_predicate_is_narrow():
    """The half-formed fingerprint catches ellipsis+spam but spares real punctuation."""
    assert is_unfinished_fragment("Hi there! ......???") is True
    assert is_unfinished_fragment("Wait... what?!") is False
    assert is_unfinished_fragment("Hmm...?") is False
    assert is_unfinished_fragment("Heads up — a new title dropped, details inside.") is False


def test_half_formed_send_reason_is_the_shared_rule():
    """The one rule the send_message gate refuses on AND classify_run flags on:
    blank/punctuation, bare URL, bail-out phrase, and unfinished fragment are all
    half-formed; a real message is not.  ``_is_degenerate_send`` (the flag side)
    is defined as ``half_formed_send_reason(...) is not None``, so this predicate
    is the single source of truth for both."""
    assert half_formed_send_reason("Hi there! ......???") is not None
    assert half_formed_send_reason("???!!! ...") is not None
    assert half_formed_send_reason("https://example.com/page") is not None
    assert half_formed_send_reason("I don't know") is not None
    assert half_formed_send_reason("still uses the original …") is not None  # truncation tail
    assert half_formed_send_reason("Heads up — a new title dropped, details inside.") is None
