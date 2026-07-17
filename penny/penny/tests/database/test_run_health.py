"""Run-health classification — the shared signal Penny's quality collector and
the addon's prompts tab both read.

``classify_run`` and ``render_run_record`` are pure functions of a run's
``PromptLog`` rows, so these build rows directly (no DB) and assert the four
failure modes are flagged structurally: a bail (no work done), an incomplete run
(hit the step ceiling), a tool-failure spiral, and a half-formed send.  A healthy
worked run and a healthy quiet read carry no flags.
"""

import json

from penny.constants import PennyConstants
from penny.database.memory import (
    LoggedToolCall,
    classify_run,
    half_formed_send_reason,
    project_run,
    render_run_calls,
    render_run_record,
)
from penny.database.models import PromptLog
from penny.llm.models import LlmToolCallFunction
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
    run_id: str = "r",
) -> PromptLog:
    """One promptlog row.  Only the run's LAST row carries outcome/tool_failures
    (that's how ``set_run_outcome`` stamps it); ``messages`` holds the tool-result
    turns (browse sections) the I/O tally reads off the final prompt.  ``target``
    is ``None`` for chat-run fixtures (chat stamps no run_target); ``run_id`` is
    fixed per fixture so whole-render literals are exact."""
    message: dict = {"role": "assistant", "content": "", "tool_calls": calls}
    return PromptLog(
        model="m",
        messages=messages,
        response=json.dumps({"choices": [{"message": message}]}),
        run_id=run_id,
        run_target=target,
        run_outcome=outcome,
        run_reason=reason,
        tool_failures=tool_failures,
    )


# The argless ``done()`` sentinel (#1569): no ``success``/``summary`` — the run
# record is generated from the ledger, so the header shows the structural outcome.
_DONE_OK = _call("done", {})


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
⚠ NO WORK DONE — reached done() (or made no tool call) without any \
read/write/browse step first; the collector is not following its instructions
done()"""
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
⚠ INCOMPLETE — hit the step ceiling without a closing done(); work landed but \
the cycle never finished cleanly
collection_write(memory='games', entries='x')"""
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
⚠ INCOMPLETE — hit the step ceiling without a closing done(); work landed but \
the cycle never finished cleanly"""
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
collection_write(memory='games', entries='x')"""
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
send_message('Hi there! ......???')
send_message('Heads up — a new title dropped, details inside.')"""
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
collection_write(memory='games', entries='x')"""
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
browse(['a', 'b'])"""
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


def test_browse_failures_but_wrote_is_not_no_writes():
    """A partial browse failure that still produced a write is not NO WRITES — the
    run wrote from the sources that succeeded, exactly what browse's partial-failure
    contract intends."""
    run = [
        _prompt(
            [
                _call("browse", {"queries": ["a", "b"]}),
                _call("collection_write", {"memory": "games", "entries": [{"content": "x"}]}),
                _DONE_OK,
            ],
            outcome="worked",
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
browse(['a', 'b'])
collection_write(memory='games', entries='x')"""
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
collection_write(memory='games', entries='$42')"""
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
            _call("collection_get", {"memory": "old-shelf", "key": "fox tales"}, call_id="r1"),
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
    assert (
        render_run_calls(_chat_kitchen_sink_run())
        == """\
run run-fixed
user: draw me a cartoon fox and log any new fox shows
    step 1: browse(['new fox cartoons']) => ## browse: https://example.com/fox-cartoons
    step 2: collection_write(memory='shows', entries='Fox Tales S2') => Wrote 1 entry: fox-tales.
    step 4: generate_image(description='a cartoon fox') => Generated image #7 of: a cartoon fox.
    step 5: collection_get(memory='old-shelf', key='fox tales') => Memory 'old-shelf' not found.
    step 6: collection_write(memory='shows', entries='Fox Tales again') => Duplicate of fox-tales.
penny: Done — logged the new season and drew your fox!
    attached: image #7"""
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
    assert (
        render_run_calls(run)
        == """\
run coll-fixed
[games]
    step 1: browse(['budget handhelds']) => ## browse: https://example.com/budget
    step 2: collection_write(memory='games', entries='Pocket Go pick') => Wrote 1 entry: pocket-go.
done: worked"""
    )


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
        render_run_calls(run)
        == """\
run stop-fixed
[games]
    step 1: collection_write(memory='games', entries='$42') => Unchanged: 'price' \
already holds the same value — no change since the last write (entry).
stopped: the value was unchanged since the last observation"""
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
        render_run_calls([only])
        == """\
run chat-quiet
user: how are you?
penny: doing great — thanks for asking!"""
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
        render_run_calls([only])
        == """\
run chat-bare
user: how are you?
penny: doing great!"""
    )


def test_project_run_origin_message_bare_and_fused_forms():
    """The origin message a run projects for skill authoring (#1661).  A REAL chat
    row is the BARE utterance (no ``---`` prefix), so with no separator the whole
    turn IS the origin — the prior split-only code returned ``""`` and killed the
    skill's description/intent derivation.  A turn that DOES carry a fused
    Live-context block still has it stripped back off (unchanged behavior)."""
    utterance = "save the ridge elevation"
    bare = _prompt(
        [_call("browse", {"queries": ["x"]})],
        run_id="bare",
        target=None,
        messages=json.dumps([{"role": "user", "content": utterance}]),
    )
    assert project_run([bare]).origin_message == utterance

    fused = _prompt(
        [_call("browse", {"queries": ["x"]})],
        run_id="fused",
        target=None,
        messages=json.dumps(
            [{"role": "user", "content": f"live ctx{PennyConstants.SECTION_SEPARATOR}{utterance}"}]
        ),
    )
    assert project_run([fused]).origin_message == utterance


def test_render_run_calls_empty_history():
    """No prompts → the honest empty marker, not a fabricated frame."""
    assert render_run_calls([]) == "(no data)"


def test_render_run_calls_truncates_bulk_results_by_reference():
    """Result-compaction logic (not a render contract): a bulk result renders as
    its first line capped with an ellipsis — stored whole in the ledger, rendered
    by reference.  The expected preview is computed (a 150-char run of x's is the
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
    expected_preview = f"{long_line[:120].rstrip()}…"
    assert render_run_calls(run) == f"run run-long\n    step 1: browse(['a']) => {expected_preview}"
