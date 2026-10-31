"""Run-health classification — the shared signal Penny's quality collector and
the addon's prompts tab both read.

``classify_run`` and ``render_run_record`` are pure functions of a run's
``PromptLog`` rows, so these build rows directly (no DB) and assert the four
failure modes are flagged structurally: a bail (no work done), an incomplete run
(hit the step ceiling), a tool-failure spiral, and a half-formed send.  A healthy
worked run and a healthy quiet read carry no flags.
"""

import json
from datetime import UTC, datetime

import pytest

from penny.agents.models import ModelCallError, RunAbort
from penny.constants import CycleTrigger, DispatchReason, PennyConstants
from penny.database.memory import (
    LoggedToolCall,
    RunHealthLog,
    classify_run,
    dispatch_line,
    half_formed_send_reason,
    render_run_calls,
    render_run_health,
    render_run_record,
)
from penny.database.models import PromptLog
from penny.llm.models import LlmToolCallFunction, strip_harmony_control_tokens
from penny.tests.conftest import LEAKED_HARMONY_TOKEN
from penny.text_validity import is_unfinished_fragment


def _call(name: str, args: dict, call_id: str | None = None) -> dict:
    """One wire-form tool call.  ``call_id`` defaults to the tool name; pass an
    explicit id when a fixture repeats a tool so each call keys its own result."""
    return {
        "id": call_id or name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _tool_result(call_id: str, content: str) -> dict:
    """One accumulated tool-result turn, keyed to its call id."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _browse_messages(*, pages: int = 0, searches: int = 0, errors: int = 0) -> str:
    """A tool-result message JSON carrying ``pages``/``searches`` browse successes and
    ``errors`` failures — the section headers ``_run_io_tally`` counts.  Lives on the
    run's last prompt (where the full accumulated conversation sits)."""
    sections = (
        [f"{PennyConstants.BROWSE_PAGE_HEADER}url\ntext"] * pages
        + [f"{PennyConstants.BROWSE_SEARCH_HEADER}query\nresults"] * searches
        + [f"{PennyConstants.BROWSE_ERROR_HEADER}url\nCould not read this page"] * errors
    )
    content = PennyConstants.SECTION_SEPARATOR.join(sections)
    return json.dumps([{"role": "tool", "tool_call_id": "b", "content": content}])


def _prompt(
    calls: list[dict],
    *,
    outcome: str | None = None,
    reason: str | None = None,
    target: str | None = "games",
    tool_failures: int | None = None,
    messages: str = "[]",
    trailing: str | None = None,
    run_id: str = "r",
    thinking: str | None = None,
    discard_reason: str | None = None,
    completion_tokens: int | None = None,
    duration_ms: int | None = None,
    timestamp: datetime | None = None,
    due_at: datetime | None = None,
    dispatch_reason: str | None = None,
    dispatch_attempt: int | None = None,
) -> PromptLog:
    """One promptlog row.  Only the run's LAST row carries outcome/tool_failures
    (that's how ``set_run_outcome`` stamps it); ``messages`` holds the tool-result
    turns (browse sections) the I/O tally reads off the final prompt, and
    ``trailing`` the tail no later call carried (#1778 — NULL for the overwhelming
    majority of runs, which round-trip every result).  ``target`` is ``None`` for
    chat-run fixtures (chat stamps no run_target); ``run_id`` is fixed per fixture
    so whole-render literals are exact.  ``thinking`` (#2841) is this CALL's own
    raw thinking — ``None`` (the default) for the overwhelming majority of
    fixtures, which predate the feature and render identically either way.
    ``discard_reason``/``completion_tokens``/``duration_ms`` (#2843) are per-row
    facts the facts clause tallies/sums across every row of a run, not just the
    last.  ``timestamp`` (defaults to the model's own ``now()`` when omitted) and
    ``due_at``/``dispatch_reason``/``dispatch_attempt`` (#2862) are the run's own
    dispatch stamp, carried on the same last row as ``outcome``."""
    message: dict = {"role": "assistant", "content": "", "tool_calls": calls}
    kwargs: dict = {
        "model": "m",
        "messages": messages,
        "response": json.dumps({"choices": [{"message": message}]}),
        "run_id": run_id,
        "run_target": target,
        "run_outcome": outcome,
        "run_reason": reason,
        "tool_failures": tool_failures,
        "trailing_messages": trailing,
        "thinking": thinking,
        "discard_reason": discard_reason,
        "completion_tokens": completion_tokens,
        "duration_ms": duration_ms,
        "run_due_at": due_at,
        "run_dispatch_reason": dispatch_reason,
        "run_dispatch_attempt": dispatch_attempt,
    }
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return PromptLog(**kwargs)


# The argless ``done()`` sentinel (#1569): no ``success``/``summary`` — the run
# record is generated from the ledger, so the header shows the structural outcome.
_DONE_OK = _call("done", {})


def _facts(
    calls: int,
    *,
    discarded: str = "none",
    tool_errors: str = "none",
    sent: str = "sent 0 messages",
) -> str:
    """The trailing facts clause ``render_run_calls``/``render_run_health`` both
    append (#2843) — assembled from plain parts, never by calling the renderer
    under test, so these literals pin the exact FORMAT rather than echo the
    code.  None of this file's fixtures set ``completion_tokens``/``duration_ms``,
    so those two fields are always ``0``; a dedicated test below pins them with
    non-zero values."""
    return (
        f"{calls} calls · 0 output tok · 0 seconds · discarded: {discarded} · "
        f"tool errors: {tool_errors} · {sent}"
    )


def test_bailed_run_is_flagged():
    """A no_work/failed run whose only call is done() did no work — bailed.

    Full verbatim render (#1569): the ``[target] <outcome>`` line (the structural
    outcome enum, since a clean ``done()`` close stamps no reason), the
    NO-WORK-DONE flag line, then the single argless ``done()`` call.  No
    ``#``/timestamp — each consumer supplies its own (see ``render_run_record``)."""
    run = [_prompt([_DONE_OK], outcome="no_work")]
    health = classify_run(run)
    assert health.bailed is True
    assert health.flags == ["no_work_done"]
    assert health.regressive is True
    assert (
        render_run_record(run)
        == """\
[games] no_work
⚠ NO WORK DONE — reached done (or made no tool call) without any \
read/write/browse step first; the collector is not following its instructions
done"""
    )


def test_incomplete_run_is_flagged_and_shows_trace():
    """An incomplete run (work landed, never closed done) is surfaced with its trace."""
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([], outcome="incomplete", reason="max steps exceeded"),
    ]
    health = classify_run(run)
    assert health.incomplete is True
    assert health.bailed is False
    assert (
        render_run_record(run)
        == """\
[games] max steps exceeded
writes: 1
⚠ INCOMPLETE — hit the step ceiling without a closing done; work landed but \
the cycle never finished cleanly
collection_write{collection: 'games', entries: 'x'}"""
    )


def test_no_tool_call_run_is_incomplete_not_bailed():
    """A run that recorded NO tool call and hit the step ceiling (the model spun on
    rejected premature-done()s until max-steps) is capacity, not a deliberate bail
    — INCOMPLETE, never NO WORK DONE.  Regression: this used to flag NO WORK DONE
    and churn a collector whose other cycles worked fine."""
    run = [_prompt([], outcome="failed", reason="max steps exceeded — no done() call")]
    health = classify_run(run)
    assert health.bailed is False
    assert health.incomplete is True
    assert health.flags == ["incomplete"]
    assert (
        render_run_record(run)
        == """\
[games] max steps exceeded — no done() call
⚠ INCOMPLETE — hit the step ceiling without a closing done; work landed but \
the cycle never finished cleanly"""
    )


def test_aborted_run_record_names_the_failed_model_call():
    """A run the MODEL CALL killed says so in its own record (#1909).

    A failing call raises before the client persists anything, so it writes no
    promptlog row — before this, every such run rendered as the generic
    ``cycle ended without a done() call`` line and the whole class was diagnosable
    only by exclusion (31 of 75 measured collector cycles).  The reason the loop
    stamps carries the abort's structural facts, and it renders verbatim as the
    record's header line: which step died, the tool the last successful step ran,
    and the error's class and message.

    It carries NO ⚠ flag, and that is right: the run neither bailed (it made a real
    read before it died) nor ran out of steps, so no health condition applies — the
    header IS the whole story, which is exactly what the flags could not say."""
    abort = RunAbort(
        step=4,
        after_tool="collection_list",
        error=ModelCallError(error_class="LlmTimeoutError", message="Request timed out."),
    )
    run = [
        _prompt([_call("collection_list", {"collection": "games"})]),
        _prompt([], outcome="failed", reason=abort.render()),
    ]
    health = classify_run(run)
    assert health.bailed is False
    assert health.flags == []
    assert (
        render_run_record(run) == "[games] model call failed at step 4 after collection_list: "
        "LlmTimeoutError: Request timed out."
    )


def test_tool_failure_count_is_flagged():
    """A run that hit tool failures and kept going is flagged with the count."""
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([_DONE_OK], outcome="worked", tool_failures=2),
    ]
    health = classify_run(run)
    assert health.tool_failures == 2
    assert (
        render_run_record(run)
        == """\
[games] worked
writes: 1
⚠ TOOL FAILURES (2) — a tool call returned an error and the run kept going
collection_write{collection: 'games', entries: 'x'}"""
    )


def test_half_formed_send_is_flagged_on_a_worked_run():
    """The real notifier shape: a worked run that ALSO sent a half-formed message
    ("Hi there! ......???") before the real one.  The bad send is flagged and shown
    in the trace, untruncated."""
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
        )
    ]
    health = classify_run(run)
    assert health.degenerate_send is True
    assert (
        render_run_record(run)
        == """\
[games] worked
writes: 0 · sends: 2
⚠ HALF-FORMED SEND — a message went out with no real content (empty, \
punctuation-only, or an unfinished fragment)
send_message{'Hi there! ......???'}
send_message{'Heads up — a new title dropped, details inside.'}"""
    )


def test_healthy_worked_run_has_no_flags():
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([_DONE_OK], outcome="worked", tool_failures=0),
    ]
    health = classify_run(run)
    assert health.flags == []
    assert health.regressive is False
    assert (
        render_run_record(run)
        == """\
[games] worked
writes: 1
collection_write{collection: 'games', entries: 'x'}"""
    )


def test_healthy_quiet_read_is_not_a_bail():
    """A no_work run that DID read before done() is a healthy quiet cycle, not a
    bail — no flags, heading-only (no trace to tempt an over-correction)."""
    run = [
        _prompt(
            [_call("log_read", {"memory": "user-messages"}), _DONE_OK],
            outcome="no_work",
        )
    ]
    health = classify_run(run)
    assert health.flags == []
    assert render_run_record(run) == "[games] no_work"


def test_logged_log_call_renders_its_log_argument_under_either_key():
    """The trace renders a log call's log name under the CURRENT argument name,
    however the row was logged: a call logged before the rename stored ``memory``,
    and the read projection — never the stored row — maps it to ``log``."""
    run = [_prompt([_call("log_read", {"memory": "user-messages"}), _DONE_OK], outcome="no_work")]
    assert "step 1: log_read{log: 'user-messages'}" in render_run_calls(
        run, with_thinking=False, delivered_text=None
    )
    fresh = [_prompt([_call("log_read", {"log": "user-messages"}), _DONE_OK], outcome="no_work")]
    assert "step 1: log_read{log: 'user-messages'}" in render_run_calls(
        fresh, with_thinking=False, delivered_text=None
    )


def test_no_writes_flagged_when_browses_fail_and_nothing_written():
    """The ai-news shape: the run browsed, browses failed, and it wrote nothing.
    Before #1569 the model's ``done(summary=...)`` could CLAIM "wrote 3 new entries"
    and that lie rendered as the header; now the record is GENERATED from the ledger
    — the header is the structural ``no_work`` outcome, and the counts line +
    ``no_writes`` flag are the two bare facts (a browse failed AND zero writes).
    What it means is the model's to reason about — the flag asserts nothing about
    cause or remedy."""
    run = [
        _prompt(
            [_call("browse", {"queries": ["a", "b"]}), _DONE_OK],
            outcome="no_work",
            messages=_browse_messages(pages=1, errors=2),
        )
    ]
    health = classify_run(run)
    assert health.no_writes is True
    assert health.flags == ["no_writes"]
    assert (
        render_run_record(run)
        == """\
[games] no_work
browses: 1 ok, 2 failed · writes: 0
⚠ NO WRITES — one or more browses failed this cycle and the run wrote nothing
browse{['a', 'b']}"""
    )


def test_clean_browse_quiet_cycle_is_not_no_writes():
    """Browsed fine, found nothing to write — a healthy quiet cycle, not NO WRITES.
    The flag needs a browse *failure*; clean reads that simply yielded nothing don't
    trip it.  Counts still render (so the shape is legible) but no flag, no trace."""
    run = [
        _prompt(
            [_call("browse", {"queries": ["a"]}), _DONE_OK],
            outcome="no_work",
            messages=_browse_messages(pages=1),
        )
    ]
    health = classify_run(run)
    assert health.no_writes is False
    assert health.flags == []
    assert (
        render_run_record(run)
        == """\
[games] no_work
browses: 1 ok, 0 failed · writes: 0"""
    )


@pytest.mark.parametrize(
    "emitted_suffix", ["", LEAKED_HARMONY_TOKEN], ids=["clean", "leaked-harmony-token"]
)
def test_browse_failures_but_wrote_is_not_no_writes(emitted_suffix):
    """A partial browse failure that still produced a write is not NO WRITES — the
    run wrote from the sources that succeeded, exactly what browse's partial-failure
    contract intends.

    Run twice over the names the model EMITTED, clean and carrying a leaked Harmony
    control token (#2013).  Dispatch strips the leak at the live boundary, so the write
    and the close both ran either way and only the ledger kept the raw name — so every
    reading of this run is identical, and both renders are asserted whole.  Read raw,
    the same run reports three lies: the write counts as a *read*, so ``writes: 0``
    against a failed browse raises ⚠ NO WRITES on a run that wrote; the leaked
    ``done()`` renders as a step of the trace; and the conclusion says the run never
    closed."""
    run = [
        _prompt(
            [
                _call("browse", {"queries": ["a", "b"]}),
                _call(
                    f"collection_write{emitted_suffix}",
                    {"memory": "games", "entries": [{"content": "x"}]},
                    call_id="w1",
                ),
                _call(f"done{emitted_suffix}", {}, call_id="d1"),
            ],
            outcome="worked",
            run_id="wrote-fixed",
            messages=_browse_messages(pages=1, errors=1),
        )
    ]
    health = classify_run(run)
    assert health.no_writes is False
    assert health.flags == []
    assert (
        render_run_record(run)
        == """\
[games] worked
browses: 1 ok, 1 failed · writes: 1
browse{['a', 'b']}
collection_write{collection: 'games', entries: 'x'}"""
    )
    assert (
        render_run_calls(run, with_thinking=False, delivered_text=None)
        == f"""\
run wrote-fixed
[games]
    step 1: browse{{['a', 'b']}}
    step 2: collection_write{{collection: 'games', entries: 'x'}}
done: worked
{_facts(1)}"""
    )


def test_a_call_that_named_only_control_tokens_named_no_tool():
    """A name that is ENTIRELY control tokens named no tool at all (#2013).

    The repair keeps the leading identifier when there is one; here there is none, so
    the call renders as the same ``?`` sentinel a nameless call has always rendered as,
    rather than as a tool nothing can call.  The verbatim name stays in the ledger."""
    assert LoggedToolCall.from_function({"name": LEAKED_HARMONY_TOKEN}).tool == ""
    run = [_prompt([_call(LEAKED_HARMONY_TOKEN, {})], outcome="worked")]
    assert classify_run(run).flags == []
    assert (
        render_run_record(run)
        == """\
[games] worked
?"""
    )


def test_write_gate_stop_ended_run_shows_trace_and_reason():
    """A write-gate STOP-ended run (#1587): a watch's unchanged re-observation ends
    the collector run at the write chokepoint — a collection_write with NO closing
    done(), a clean ``no_work`` outcome, and the declared stop reason stamped as
    ``run_reason``.  Whole render: the ``[target] <stop reason>`` header, the counts
    line, then the stop-point write trace — NO ⚠ flag (a clean stop is healthy), and
    the reason is verifiable from the record alone."""
    run = [
        _prompt(
            [
                _call(
                    "collection_write",
                    {"memory": "games", "entries": [{"key": "price", "content": "$42"}]},
                )
            ],
            outcome="no_work",
            reason="the value was unchanged since the last observation",
        )
    ]
    health = classify_run(run)
    assert health.flags == []
    assert health.regressive is False
    assert (
        render_run_record(run)
        == """\
[games] the value was unchanged since the last observation
writes: 1
collection_write{collection: 'games', entries: '$42'}"""
    )


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


# ── Ledger provenance closure (#1560) ────────────────────────────────────────


def test_logged_tool_call_round_trips_to_the_wire_form():
    """A logged tool call and the outgoing wire call are the SAME structure —
    ``replay(logged_call) == original_call`` (#1560).  The model emits a call as
    ``{"name", "arguments": "<json>"}``; ``LoggedToolCall.from_function`` reads it,
    ``to_wire`` re-emits the identical envelope, and re-parsing that envelope
    through the executor's own boundary (``LlmToolCallFunction``) yields the same
    name + arguments — so the logged form is canonical, never a paraphrase, and a
    logged call could be re-executed unchanged (or promoted to a skill step by
    copy)."""
    original = {"name": "collection_write", "arguments": '{"memory": "watch", "entries": []}'}
    logged = LoggedToolCall.from_function(original)
    # Round-trip identity at the structured level.
    assert LoggedToolCall.from_function(logged.to_wire()) == logged
    # Replay through the SAME parse the executor uses: the re-emitted wire call
    # produces the identical name + arguments as parsing the original would.
    replayed = LlmToolCallFunction(
        name=logged.to_wire()["name"], arguments=json.loads(logged.to_wire()["arguments"])
    )
    parsed_original = LlmToolCallFunction(
        name=original["name"], arguments=json.loads(original["arguments"])
    )
    assert replayed == parsed_original
    # A leaked Harmony control token is kept VERBATIM on the logged name (the log never
    # paraphrases the call, and the leak is a real diagnostic signal) — while ``tool``
    # answers "which tool is this?" the way production answers it at the live boundary,
    # so a consumer comparing names never has to know the leak exists (#2013).
    leaked = LoggedToolCall.from_function(
        {"name": f"collection_write{LEAKED_HARMONY_TOKEN}", "arguments": "{}"}
    )
    assert leaked.name == f"collection_write{LEAKED_HARMONY_TOKEN}"
    assert leaked.to_wire()["name"] == f"collection_write{LEAKED_HARMONY_TOKEN}"
    assert LoggedToolCall.from_function(leaked.to_wire()) == leaked
    assert leaked.tool == "collection_write"
    assert leaked.tool == strip_harmony_control_tokens(leaked.name)


# The run-calls render contract (#1560): every rendered surface the model reads is
# asserted as its ENTIRE structure in ONE literal — a reviewer sees exactly what the
# model sees.  The kitchen-sink case folds every input shape the surface can render
# into one scenario; the sub-cases below isolate the variations (collector run,
# pure-reply run with no egress, empty history).  Fixtures are fully deterministic:
# fixed run ids, fixed call ids, fictional content.


def _chat_kitchen_sink_run() -> list[PromptLog]:
    """One chat run exercising EVERY shape the run-calls render can show: the
    user's opening message, a browse step, a collection write with its gate
    outcome, a mid-run rejected ``done()`` (the step-number GAP), a
    generate_image with its media id, a failed call, a duplicate-rejected write,
    the final reply, and the egress attachment."""
    user_turn = {
        "role": "user",
        "content": f"Live context{PennyConstants.SECTION_SEPARATOR}"
        "draw me a cartoon fox and log any new fox shows",
    }
    first = _prompt(
        [_call("browse", {"queries": ["new fox cartoons"]})],
        run_id="run-fixed",
        target=None,
        messages=json.dumps([user_turn]),
    )
    second = _prompt(
        [
            _call(
                "collection_write",
                {"memory": "shows", "entries": [{"key": "fox-tales", "content": "Fox Tales S2"}]},
                call_id="w1",
            ),
            _call("done", {}, call_id="d1"),
        ],
        run_id="run-fixed",
        target=None,
    )
    third = _prompt(
        [
            _call("generate_image", {"description": "a cartoon fox"}, call_id="g1"),
            _call("collection_get", {"collection": "old-shelf", "key": "fox tales"}, call_id="r1"),
            _call(
                "collection_write",
                {
                    "memory": "shows",
                    "entries": [{"key": "fox-tales-2", "content": "Fox Tales again"}],
                },
                call_id="w2",
            ),
        ],
        run_id="run-fixed",
        target=None,
    )
    accumulated = [
        user_turn,
        _tool_result(
            "browse", "## browse: https://example.com/fox-cartoons\nThree new fox shows reviewed."
        ),
        _tool_result("w1", "Wrote 1 entry: fox-tales."),
        _tool_result("d1", "done() rejected — make a real tool call first."),
        _tool_result("g1", f"{PennyConstants.GENERATED_IMAGE_RESULT_PREFIX}7 of: a cartoon fox."),
        _tool_result("r1", "Memory 'old-shelf' not found."),
        _tool_result("w2", "Duplicate of fox-tales."),
    ]
    final = PromptLog(
        model="m",
        messages=json.dumps(accumulated),
        response=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done — logged the new season and drew your fox!",
                        }
                    }
                ]
            }
        ),
        run_id="run-fixed",
        run_target=None,
    )
    return [first, second, third, final]


def test_render_run_calls_chat_kitchen_sink_full_literal():
    """The chat-run render contract, whole-output (#1560, criteria 3 + 5 + the
    anchor invariant): the run names itself (``run <id>``), each step is the
    canonical call projection with its compact result (``step <N>: <call> =>
    <outcome>``), the mid-run rejected ``done()`` consumes step 3 and leaves a
    visible GAP (numbers are persisted coordinates — never renumbered), the write
    gate outcomes (written / duplicate-rejected) and the failed call render
    honestly, and the egress trace names the delivered media by typed id — so a
    delivery is inspected by a read, not confabulated."""
    assert render_run_calls(_chat_kitchen_sink_run(), with_thinking=False, delivered_text=None) == (
        "run run-fixed\n"
        "user: draw me a cartoon fox and log any new fox shows\n"
        "    step 1: browse{['new fox cartoons']} => "
        "## browse: https://example.com/fox-cartoons\n"
        "    step 2: collection_write{collection: 'shows', entries: 'Fox Tales S2'} => "
        "Wrote 1 entry: fox-tales.\n"
        "    step 4: generate_image{description: 'a cartoon fox'} => "
        "Generated image #7 of: a cartoon fox.\n"
        "    step 5: collection_get{collection: 'old-shelf', key: 'fox tales'} => "
        "Memory 'old-shelf' not found.\n"
        "    step 6: collection_write{collection: 'shows', entries: 'Fox Tales again'} => "
        "Duplicate of fox-tales.\n"
        "penny: Done — logged the new season and drew your fox!\n"
        "    attached: image #7\n"
        f"{_facts(4)}"
    )


def test_render_run_calls_collector_run_full_literal():
    """A collector run through the same lens, whole-output: origin is the bound
    target (no user message), steps are the canonical projection, and the
    conclusion is the run's STRUCTURAL outcome — ``done: <outcome>`` (#1569), never
    a model-authored summary — no egress line (nothing attached)."""
    run = [
        _prompt(
            [_call("browse", {"queries": ["budget handhelds"]})],
            run_id="coll-fixed",
        ),
        _prompt(
            [
                _call(
                    "collection_write",
                    {
                        "memory": "games",
                        "entries": [{"key": "pocket-go", "content": "Pocket Go pick"}],
                    },
                    call_id="w1",
                ),
                _DONE_OK,
            ],
            run_id="coll-fixed",
            outcome="worked",
            messages=json.dumps(
                [
                    _tool_result("browse", "## browse: https://example.com/budget\nBudget picks."),
                    _tool_result("w1", "Wrote 1 entry: pocket-go."),
                ]
            ),
        ),
    ]
    assert render_run_calls(run, with_thinking=False, delivered_text=None) == (
        "run coll-fixed\n"
        "[games]\n"
        "    step 1: browse{['budget handhelds']} => ## browse: https://example.com/budget\n"
        "    step 2: collection_write{collection: 'games', entries: 'Pocket Go pick'} => "
        "Wrote 1 entry: pocket-go.\n"
        "done: worked\n"
        f"{_facts(2)}"
    )


# A short thinking string the examples below attach to step 2 (the write) — short
# enough to render WHOLE even by default, so the excerpt-vs-full distinction (the
# next test) needs its own longer fixture.
_SHORT_THINKING = (
    "Pocket Go looked like the clear budget pick on this page, so I'm writing it down now."
)

# A thinking string over the 600-character cap, built so the excerpt
# (``_thinking_excerpt``'s own cut: the last 600 characters, advanced to the next
# sentence boundary) is known and asserted whole below — computed once, by hand,
# against the same algorithm, never re-derived from the implementation under test.
_LONG_THINKING = (
    "Looking at the budget handhelds page, nothing here stands out yet. " * 10
    + "Pocket Go is clearly the strongest budget pick on this page, so I will write it down now."
)
_LONG_THINKING_EXCERPT = (
    "Looking at the budget handhelds page, nothing here stands out yet. " * 7
    + "Pocket Go is clearly the strongest budget pick on this page, so I will write it down now."
)


def _collector_run_with_thinking(thinking: str) -> list[PromptLog]:
    """The same two-step collector run ``test_render_run_calls_collector_run_full_
    literal`` uses, with ``thinking`` attached to step 2's own call (the write) —
    shared by the thinking-specific tests below so each one only states what it
    adds over the plain literal."""
    return [
        _prompt([_call("browse", {"queries": ["budget handhelds"]})], run_id="coll-fixed"),
        _prompt(
            [
                _call(
                    "collection_write",
                    {
                        "memory": "games",
                        "entries": [{"key": "pocket-go", "content": "Pocket Go pick"}],
                    },
                    call_id="w1",
                ),
                _DONE_OK,
            ],
            run_id="coll-fixed",
            outcome="worked",
            thinking=thinking,
            messages=json.dumps(
                [
                    _tool_result("browse", "## browse: https://example.com/budget\nBudget picks."),
                    _tool_result("w1", "Wrote 1 entry: pocket-go."),
                ]
            ),
        ),
    ]


def test_render_run_calls_with_thinking_shows_a_short_thinking_whole():
    """``with_thinking=True`` (run_get's own call, #2841) renders a step's thinking
    — short enough to sit under the excerpt cap, so it shows WHOLE either way —
    on its own line right before that step's call line."""
    run = _collector_run_with_thinking(_SHORT_THINKING)
    assert render_run_calls(run, with_thinking=True, delivered_text=None) == (
        "run coll-fixed\n"
        "[games]\n"
        "    step 1: browse{['budget handhelds']} => ## browse: https://example.com/budget\n"
        f"    step 2 thinking: {_SHORT_THINKING}\n"
        "    step 2: collection_write{collection: 'games', entries: 'Pocket Go pick'} => "
        "Wrote 1 entry: pocket-go.\n"
        "done: worked\n"
        f"{_facts(2)}"
    )


def test_render_run_calls_with_thinking_false_shows_nothing_even_when_logged():
    """``with_thinking=False`` (``run_list``'s own call, #2841) renders no thinking
    line at all, even for a step whose call DID log one — the batch reader states
    it wants none, rather than relying on every step happening to carry none."""
    run = _collector_run_with_thinking(_SHORT_THINKING)
    rendered = render_run_calls(run, with_thinking=False, delivered_text=None)
    assert "thinking" not in rendered

    # And a step with NO logged thinking renders identically whichever flag is
    # passed — the excerpt is ABSENT, not merely empty.
    bare = [
        _prompt([_call("browse", {"queries": ["budget handhelds"]})], run_id="coll-fixed"),
        _prompt([_DONE_OK], run_id="coll-fixed", outcome="worked"),
    ]
    assert render_run_calls(bare, with_thinking=True, delivered_text=None) == render_run_calls(
        bare, with_thinking=False, delivered_text=None
    )


def test_render_run_calls_thinking_excerpt_cuts_to_last_600_chars_at_sentence_boundary():
    """Thinking over the 600-character cap renders only its closing part — the
    last ~600 characters, advanced to the NEXT sentence boundary inside that tail
    so the excerpt opens on a whole sentence rather than a severed one (#2841)."""
    run = _collector_run_with_thinking(_LONG_THINKING)
    rendered = render_run_calls(run, with_thinking=True, delivered_text=None)
    assert f"    step 2 thinking: {_LONG_THINKING_EXCERPT}\n" in rendered
    assert _LONG_THINKING not in rendered  # the excerpt, never the whole thing


def test_render_run_calls_step_arg_shows_that_steps_full_thinking():
    """``run_get(run=…, step=2)`` (#2841) renders step 2's thinking WHOLE — the
    text the default excerpted render cuts — while every other step (none here
    with thinking) stays excerpted."""
    run = _collector_run_with_thinking(_LONG_THINKING)
    rendered = render_run_calls(run, with_thinking=True, step=2, delivered_text=None)
    assert f"    step 2 thinking: {_LONG_THINKING}\n" in rendered
    assert f"    step 2 thinking: {_LONG_THINKING_EXCERPT}\n" not in rendered


def test_render_run_calls_unmatched_step_appends_an_honest_line_not_a_no_op():
    """A ``step`` naming no step this run rendered (#2841) gets the run's full
    record PLUS one trailing line saying so and naming the valid range — never a
    silent no-op, which would read as "that step has no thinking" rather than
    "that step doesn't exist"."""
    run = _collector_run_with_thinking(_SHORT_THINKING)
    rendered = render_run_calls(run, with_thinking=True, step=9, delivered_text=None)
    assert rendered.startswith(render_run_calls(run, with_thinking=True, delivered_text=None))
    assert rendered.endswith("\nrun coll-fixed has no step 9; its steps are 1–2")


def test_render_run_calls_unmatched_step_on_a_run_with_no_steps():
    """A run that rendered no steps at all (every call was ``done()``) gets the
    "has no steps" line rather than a fabricated range."""
    run = [_prompt([_DONE_OK], run_id="bail-fixed", outcome="no_work")]
    rendered = render_run_calls(run, with_thinking=True, step=1, delivered_text=None)
    assert rendered.endswith("\nrun bail-fixed has no steps")


def test_render_run_calls_write_gate_stop_conclusion_full_literal():
    """A write-gate STOP-ended run through the sequence lens, whole-output (#1587):
    the run closed at the chokepoint with no ``done()``, so the conclusion renders
    ``stopped: <reason>`` — never a fabricated ``done:`` — and the stop-point write
    is the sole step."""
    run = [
        _prompt(
            [
                _call(
                    "collection_write",
                    {"memory": "games", "entries": [{"key": "price", "content": "$42"}]},
                    call_id="w1",
                )
            ],
            run_id="stop-fixed",
            outcome="no_work",
            reason="the value was unchanged since the last observation",
            messages=json.dumps(
                [
                    _tool_result(
                        "w1",
                        "Unchanged: 'price' already holds the same value — "
                        "no change since the last write (entry).",
                    )
                ]
            ),
        )
    ]
    assert (
        render_run_calls(run, with_thinking=False, delivered_text=None)
        == f"""\
run stop-fixed
[games]
    step 1: collection_write{{collection: 'games', entries: '$42'}} => Unchanged: 'price' \
already holds the same value — no change since the last write (entry).
stopped: the value was unchanged since the last observation
{_facts(1)}"""
    )


def test_render_run_calls_pure_reply_run_full_literal():
    """A tool-less chat turn, whole-output: just the run id, the ask, and the
    reply — no steps, no egress."""
    turn = {
        "role": "user",
        "content": f"Live context{PennyConstants.SECTION_SEPARATOR}how are you?",
    }
    only = PromptLog(
        model="m",
        messages=json.dumps([turn]),
        response=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "doing great — thanks for asking!",
                        }
                    }
                ]
            }
        ),
        run_id="chat-quiet",
        run_target=None,
    )
    assert (
        render_run_calls([only], with_thinking=False, delivered_text=None)
        == f"""\
run chat-quiet
user: how are you?
penny: doing great — thanks for asking!
{_facts(1)}"""
    )


def test_render_run_calls_bare_utterance_origin_full_literal():
    """A REAL chat row carries the user's message as the BARE utterance — no fused
    Live-context ``---`` prefix (#1661).  With no separator the whole turn IS the
    origin, so the render shows ``user: <utterance>`` (the prior split-only code
    returned ``""`` here, dropping the origin entirely).  The fused/separator form
    is pinned by ``test_render_run_calls_pure_reply_run_full_literal`` above."""
    turn = {"role": "user", "content": "how are you?"}
    only = PromptLog(
        model="m",
        messages=json.dumps([turn]),
        response=json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "doing great!"}}]}
        ),
        run_id="chat-bare",
        run_target=None,
    )
    assert (
        render_run_calls([only], with_thinking=False, delivered_text=None)
        == f"""\
run chat-bare
user: how are you?
penny: doing great!
{_facts(1)}"""
    )


def _terminal_write_run(*, trailing: str | None) -> list[PromptLog]:
    """A run that ended the instant its ``collection_write`` came back failed — the
    shape every gap case shares (``max_steps`` on a tool step, a write-gate STOP, a
    reroll abort, an exception): the call is durable in ``response``, but its result
    had no next call to ride into."""
    call = _call(
        "collection_write",
        {"memory": "shelf", "entries": [{"key": "price", "content": "$42"}]},
        call_id="w1",
    )
    return [
        _prompt(
            [call],
            run_id="tail-fixed",
            target=None,
            messages=json.dumps([{"role": "user", "content": "log the new shelf price"}]),
            trailing=trailing,
        )
    ]


def test_render_run_calls_terminal_result_rides_the_run_tail_full_literal():
    """A run that ended immediately after executing a tool renders that call AND its
    result (#1778), whole-output.

    A tool result is otherwise written down only as a side effect of riding into the
    NEXT call's ``messages``, so a run with no next call recorded its terminal outcome
    nowhere — and those are exactly the runs worth reading afterwards (a failed final
    attempt, a write-gate STOP, ``max_steps`` on a tool step).  The trailing tail the
    loop stamps at close closes it: the step renders ``=> <result>`` like every other.

    The paired assertion is the load-bearing half — the SAME run without a tail renders
    the call bare, which is the bug: indistinguishable from a call whose result was
    empty."""
    tail = json.dumps(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call(
                        "collection_write",
                        {"memory": "shelf", "entries": [{"key": "price", "content": "$42"}]},
                        call_id="w1",
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "w1",
                "content": "Memory 'shelf' not found — did you mean 'shelves'?",
                PennyConstants.TOOL_RESULT_SUCCESS_KEY: False,
            },
        ]
    )
    run = _terminal_write_run(trailing=tail)
    assert (
        render_run_calls(run, with_thinking=False, delivered_text=None)
        == f"""\
run tail-fixed
user: log the new shelf price
    step 1: collection_write{{collection: 'shelf', entries: '$42'}} => Memory 'shelf' not found \
— did you mean 'shelves'?
{_facts(1, tool_errors="collection_write×1")}"""
    )
    # Without the tail (every pre-#1778 row) the same run renders the call bare — the
    # gap this closes.
    assert (
        render_run_calls(
            _terminal_write_run(trailing=None), with_thinking=False, delivered_text=None
        )
        == f"""\
run tail-fixed
user: log the new shelf price
    step 1: collection_write{{collection: 'shelf', entries: '$42'}}
{_facts(1)}"""
    )


def test_render_run_calls_empty_history():
    """No prompts → the honest empty marker, not a fabricated frame."""
    assert render_run_calls([], with_thinking=False, delivered_text=None) == "(no data)"


def test_render_run_calls_first_line_whole():
    """Result-compaction logic (not a render contract): a bulk result renders as
    its first line WHOLE — no cap (#2642) — stored whole in the ledger, rendered
    by reference.  The expected render is computed (a 150-char run of x's is the
    fixture), so this stays a logic test; the render contracts above are the
    whole-output literals."""
    long_line = "## browse: " + "x" * 150
    run = [
        _prompt(
            [_call("browse", {"queries": ["a"]})],
            run_id="run-long",
            target=None,
            messages=json.dumps([_tool_result("browse", f"{long_line}\npage body")]),
        )
    ]
    assert (
        render_run_calls(run, with_thinking=False, delivered_text=None)
        == f"run run-long\n    step 1: browse{{['a']}} => {long_line}\n{_facts(1)}"
    )


# ── Per-run operational facts (#2843) ─────────────────────────────────────────


def test_run_facts_line_reports_cost_discards_and_tool_errors_whole():
    """The facts clause sums calls/output-tokens/time across EVERY row of the
    run (not just the last), and tallies discards and tool errors across every
    row too — whole-render literal.  Two rows: 4000 + 123 output tokens,
    1500ms + 2000ms, one discarded draw on the first row, and a failed
    ``browse`` call (the structural ``tool_success`` stamp, #1600) on the
    second's own tool-result turn."""
    run = [
        _prompt(
            [],
            run_id="facts-1",
            completion_tokens=4000,
            duration_ms=1500,
            discard_reason="hallucinated_urls",
        ),
        _prompt(
            [_call("browse", {"queries": ["a"]}, call_id="b1"), _DONE_OK],
            run_id="facts-1",
            outcome="worked",
            completion_tokens=123,
            duration_ms=2000,
            messages=json.dumps(
                [
                    {
                        "role": "tool",
                        "tool_call_id": "b1",
                        "content": "could not reach the page",
                        PennyConstants.TOOL_RESULT_SUCCESS_KEY: False,
                    }
                ]
            ),
        ),
    ]
    assert (
        render_run_calls(run, with_thinking=False, delivered_text=None)
        == """\
run facts-1
[games]
    step 1: browse{['a']} => could not reach the page
done: worked
2 calls · 4123 output tok · 3 seconds · discarded: hallucinated_urls×1 · \
tool errors: browse×1 · sent 0 messages"""
    )


def test_run_facts_line_reports_a_link_only_when_the_delivered_text_has_one():
    """``delivered_text`` (#2843) is tested for a URL verbatim — never
    reconstructed from a tool call's arguments (there is no ``send_message``
    tool any more, #1911).  A run that sent nothing names zero; one that sent
    a message with no link names zero links; one whose delivered text carries
    a URL names one."""
    run = [_prompt([_DONE_OK], run_id="facts-2", outcome="worked")]
    assert "sent 0 messages" in render_run_calls(run, with_thinking=False, delivered_text=None)
    assert "sent 1 message, links: 0" in render_run_calls(
        run, with_thinking=False, delivered_text="all done!"
    )
    assert "sent 1 message, links: 1" in render_run_calls(
        run,
        with_thinking=False,
        delivered_text="done — see https://example.com/report for the details",
    )


def test_render_run_health_names_chat_and_collector_sources():
    """``render_run_health`` (the ``run-health`` facade's record): a collector
    run's header carries its target AND its structural outcome; a chat run
    carries no stamped outcome on an ordinary turn (#2355 part D), so its
    header names only the source — never a guessed outcome."""
    collector_run = [_prompt([_DONE_OK], run_id="coll-x", target="games", outcome="worked")]
    assert render_run_health(collector_run, None) == (
        "run coll-x [games] worked · 1 calls · 0 output tok · 0 seconds · "
        "discarded: none · tool errors: none · sent 0 messages"
    )
    chat_run = [_prompt([], run_id="chat-x", target=None)]
    assert render_run_health(chat_run, "hi!") == (
        "run chat-x [chat] · 1 calls · 0 output tok · 0 seconds · "
        "discarded: none · tool errors: none · sent 1 message, links: 0"
    )


# ── The dispatch-reason line (#2862) ───────────────────────────────────────
#
# A job's run record states, in one line, when it started, which occurrence it
# was for, and WHY it ran then — a closed set, decided at dispatch and read
# back here from the three stamped columns, never re-derived.  ``dispatch_line``
# is pure over its three arguments, so these fixtures need no DB: a
# collector cycle due at 9:30 AM, on America/New_York's clock (EDT in July).

_NY = "America/New_York"
_DUE = datetime(2026, 7, 20, 13, 30, tzinfo=UTC)  # 9:30 AM EDT


def test_dispatch_line_on_time():
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r1",
            timestamp=datetime(2026, 7, 20, 13, 32, tzinfo=UTC),  # 9:32 AM EDT
            due_at=_DUE,
            dispatch_reason=DispatchReason.ON_TIME.value,
        )
    ]
    assert dispatch_line(prompts, _NY, "daily") == (
        "started 09:32 AM EDT for the daily 09:30 AM EDT run · on time"
    )


def test_dispatch_line_catch_up_across_a_restart():
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r2",
            timestamp=datetime(2026, 7, 20, 23, 12, tzinfo=UTC),  # 7:12 PM EDT
            due_at=_DUE,
            dispatch_reason=DispatchReason.CATCH_UP_OFFLINE.value,
        )
    ]
    assert dispatch_line(prompts, _NY, "daily") == (
        "started 07:12 PM EDT for the daily 09:30 AM EDT run · "
        "catching up a missed run: Penny was offline at 09:30 AM EDT"
    )


def test_dispatch_line_catch_up_without_a_restart():
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r3",
            timestamp=datetime(2026, 7, 20, 23, 12, tzinfo=UTC),  # 7:12 PM EDT
            due_at=_DUE,
            dispatch_reason=DispatchReason.CATCH_UP.value,
        )
    ]
    assert dispatch_line(prompts, _NY, "daily") == (
        "started 07:12 PM EDT for the daily 09:30 AM EDT run · catching up a missed run"
    )


def test_dispatch_line_retry_after_a_preemption():
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r4",
            timestamp=datetime(2026, 7, 20, 13, 34, tzinfo=UTC),  # 9:34 AM EDT
            due_at=_DUE,
            dispatch_reason=DispatchReason.RETRY_PREEMPTED.value,
            dispatch_attempt=2,
        )
    ]
    assert dispatch_line(prompts, _NY, "daily") == (
        "started 09:34 AM EDT for the daily 09:30 AM EDT run · "
        "retrying after an interrupted attempt (attempt 2)"
    )


def test_dispatch_line_retry_after_a_failure():
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r5",
            timestamp=datetime(2026, 7, 20, 13, 35, tzinfo=UTC),  # 9:35 AM EDT
            due_at=_DUE,
            dispatch_reason=DispatchReason.RETRY_FAILED.value,
            dispatch_attempt=2,
        )
    ]
    assert dispatch_line(prompts, _NY, "daily") == (
        "started 09:35 AM EDT for the daily 09:30 AM EDT run · "
        "retrying after a failed attempt (attempt 2)"
    )


def test_dispatch_line_on_demand():
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r6",
            timestamp=datetime(2026, 7, 20, 18, 14, tzinfo=UTC),  # 2:14 PM EDT
            dispatch_reason=DispatchReason.ON_DEMAND.value,
        )
    ]
    assert dispatch_line(prompts, _NY, "daily") == "started 02:14 PM EDT on demand"


def test_dispatch_line_empty_for_a_row_predating_the_stamp():
    """A chat run, or any collector row logged before #2862, carries no
    dispatch stamp — ``dispatch_line`` renders nothing, never a guess."""
    prompts = [_prompt([], run_id="old", target=None)]
    assert dispatch_line(prompts, _NY, "daily") == ""


def test_render_run_health_prepends_the_dispatch_line():
    """``render_run_health`` carries the dispatch line (#2862) as its own
    leading line, ahead of the existing header/facts line, unchanged."""
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r7",
            target="games",
            timestamp=datetime(2026, 7, 20, 13, 32, tzinfo=UTC),
            due_at=_DUE,
            dispatch_reason=DispatchReason.ON_TIME.value,
        )
    ]
    assert render_run_health(prompts, None, timezone_name=_NY, schedule_words="daily") == (
        "started 09:32 AM EDT for the daily 09:30 AM EDT run · on time\n"
        "run r7 [games] worked · 1 calls · 0 output tok · 0 seconds · "
        "discarded: none · tool errors: none · sent 0 messages"
    )


def test_render_run_calls_prepends_the_dispatch_line():
    """``render_run_calls`` (``run_list``/``run_get``) carries the same leading
    dispatch line (#2862), ahead of the ``run <id>`` anchor line."""
    prompts = [
        _prompt(
            [_DONE_OK],
            outcome="worked",
            run_id="r8",
            target="games",
            timestamp=datetime(2026, 7, 20, 13, 32, tzinfo=UTC),
            due_at=_DUE,
            dispatch_reason=DispatchReason.ON_TIME.value,
        )
    ]
    rendered = render_run_calls(
        prompts, with_thinking=False, delivered_text=None, timezone_name=_NY, schedule_words="daily"
    )
    assert rendered.startswith(
        "started 09:32 AM EDT for the daily 09:30 AM EDT run · on time\nrun r8\n"
    )


def test_run_delivered_text_reads_the_chat_direct_reply(db):
    """A chat run's own reply is linked by ``messagelog.run_id`` (#2630) — the
    STORED, already-delivered text, read back verbatim."""
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        "here's what I found: https://example.com/page",
        run_id="chat-run-1",
    )
    assert (
        db.messages.delivered_text_for_run("chat-run-1")
        == "here's what I found: https://example.com/page"
    )
    assert db.messages.delivered_text_for_run("no-such-run") is None


def test_run_delivered_text_reads_a_delivered_collector_report(db):
    """A collector's notifying close hands off through the send queue
    (``report_run_id``, #2656) — the DELIVERED row's own stored content, read
    only once ``sent_at`` is stamped; an undelivered (never-sent) compose
    names nothing, never a guess at what it would have said."""
    item_id = db.send_queue.enqueue_report("games", CycleTrigger.CADENCE, "coll-run-1")
    db.send_queue.store_composed(item_id, "a new deal dropped")
    # Not yet delivered — composed but no ``sent_at`` stamp.
    assert db.messages.delivered_text_for_run("coll-run-1") is None
    db.send_queue.mark_sent(item_id)
    assert db.messages.delivered_text_for_run("coll-run-1") == "a new deal dropped"


def test_run_health_facade_covers_chat_and_collector_runs(db):
    """Unlike ``collector-runs`` (which excludes chat — it stamps no
    completion row), the ``run-health`` facade's dispatch is ``RunHealthLog``
    and its read covers BOTH a chat turn and a collector cycle."""
    memory = db.memory(PennyConstants.MEMORY_RUN_HEALTH_LOG)
    assert isinstance(memory, RunHealthLog)
    db.messages.log_prompt(
        model="m",
        messages=[],
        response={"choices": [{"message": {"content": "hi there!"}}]},
        agent_name=PennyConstants.CHAT_AGENT_NAME,
        run_id="health-chat-1",
    )
    db.messages.log_prompt(
        model="m",
        messages=[],
        response={"choices": [{"message": {"tool_calls": [_DONE_OK]}}]},
        agent_name="collector",
        run_id="health-coll-1",
        run_target="games",
    )
    db.messages.set_run_outcome("health-coll-1", "worked", "")
    entries = memory.newest_entries(k=10)
    contents = [entry.content for entry in entries]
    assert any("[chat]" in c and "run health-chat-1" in c for c in contents)
    assert any("[games] worked" in c and "run health-coll-1" in c for c in contents)
