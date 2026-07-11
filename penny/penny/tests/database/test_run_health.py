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
    render_run_calls,
    render_run_record,
)
from penny.database.models import PromptLog
from penny.llm.models import LlmToolCallFunction
from penny.text_validity import is_unfinished_fragment


def _call(name: str, args: dict) -> dict:
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


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
    target: str = "games",
    tool_failures: int | None = None,
    messages: str = "[]",
) -> PromptLog:
    """One promptlog row.  Only the run's LAST row carries outcome/tool_failures
    (that's how ``set_run_outcome`` stamps it); ``messages`` holds the tool-result
    turns (browse sections) the I/O tally reads off the final prompt."""
    message: dict = {"role": "assistant", "content": "", "tool_calls": calls}
    return PromptLog(
        model="m",
        messages=messages,
        response=json.dumps({"choices": [{"message": message}]}),
        run_id="r",
        run_target=target,
        run_outcome=outcome,
        run_reason=reason,
        tool_failures=tool_failures,
    )


_DONE_OK = _call("done", {"success": True, "summary": "done"})


def test_bailed_run_is_flagged():
    """A no_work/failed run whose only call is done() did no work — bailed.

    Full verbatim render: the ``[target] summary`` line, the NO-WORK-DONE flag
    line, then the single tool call.  No ``#``/timestamp — each
    consumer supplies its own (see ``render_run_record``)."""
    run = [_prompt([_DONE_OK], outcome="no_work", reason="nothing new")]
    health = classify_run(run)
    assert health.bailed is True
    assert health.flags == ["no_work_done"]
    assert health.regressive is True
    assert (
        render_run_record(run)
        == """\
[games] nothing new
⚠ NO WORK DONE — reached done() (or made no tool call) without any \
read/write/browse step first; the collector is not following its instructions
done(success=True, summary='done')"""
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
        _prompt([_DONE_OK], outcome="worked", reason="wrote one", tool_failures=2),
    ]
    health = classify_run(run)
    assert health.tool_failures == 2
    assert (
        render_run_record(run)
        == """\
[games] wrote one
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
            reason="delivered a notification",
        )
    ]
    health = classify_run(run)
    assert health.degenerate_send is True
    assert (
        render_run_record(run)
        == """\
[games] delivered a notification
writes: 0 · sends: 2
⚠ HALF-FORMED SEND — a message went out with no real content (empty, \
punctuation-only, or an unfinished fragment)
send_message('Hi there! ......???')
send_message('Heads up — a new title dropped, details inside.')"""
    )


def test_healthy_worked_run_has_no_flags():
    run = [
        _prompt([_call("collection_write", {"memory": "games", "entries": [{"content": "x"}]})]),
        _prompt([_DONE_OK], outcome="worked", reason="wrote one", tool_failures=0),
    ]
    health = classify_run(run)
    assert health.flags == []
    assert health.regressive is False
    assert (
        render_run_record(run)
        == """\
[games] wrote one
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
            reason="nothing new",
        )
    ]
    health = classify_run(run)
    assert health.flags == []
    assert render_run_record(run) == "[games] nothing new"


def test_no_writes_flagged_when_browses_fail_and_nothing_written():
    """The ai-news shape: the run browsed, browses failed, and it wrote nothing —
    yet its done() summary claims otherwise.  ``no_writes`` is the two bare facts (a
    browse failed AND zero writes); the counts line under the summary makes the
    contradiction with the prose plain.  What it means is the model's to reason
    about — the flag asserts nothing about cause or remedy."""
    run = [
        _prompt(
            [_call("browse", {"queries": ["a", "b"]}), _DONE_OK],
            outcome="no_work",
            reason="wrote 3 new entries",
            messages=_browse_messages(pages=1, errors=2),
        )
    ]
    health = classify_run(run)
    assert health.no_writes is True
    assert health.flags == ["no_writes"]
    assert (
        render_run_record(run)
        == """\
[games] wrote 3 new entries
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
            reason="no new matches this cycle",
            messages=_browse_messages(pages=1),
        )
    ]
    health = classify_run(run)
    assert health.no_writes is False
    assert health.flags == []
    assert (
        render_run_record(run)
        == """\
[games] no new matches this cycle
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
            reason="wrote one despite a dead source",
            messages=_browse_messages(pages=1, errors=1),
        )
    ]
    health = classify_run(run)
    assert health.no_writes is False
    assert health.flags == []
    assert (
        render_run_record(run)
        == """\
[games] wrote one despite a dead source
browses: 1 ok, 1 failed · writes: 1
browse(['a', 'b'])
collection_write(memory='games', entries='x')"""
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


def _chat_run_with_generated_image(media_id: int) -> list[PromptLog]:
    """A chat run that drew an image: opening user message, a generate_image call,
    its result naming the stored media id, and a final plain reply."""
    opening = json.dumps(
        [
            {
                "role": "user",
                "content": f"live context{PennyConstants.SECTION_SEPARATOR}draw me a cat",
            }
        ]
    )
    first = PromptLog(
        model="m",
        messages=opening,
        response=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_call("generate_image", {"description": "a cat"})],
                        }
                    }
                ]
            }
        ),
        run_id="run-abc",
        agent_name=PennyConstants.CHAT_AGENT_NAME,
    )
    last_messages = json.dumps(
        [
            {
                "role": "tool",
                "tool_call_id": "generate_image",
                "content": f"{PennyConstants.GENERATED_IMAGE_RESULT_PREFIX}{media_id} of: a cat. "
                "It will be delivered...",
            }
        ]
    )
    last = PromptLog(
        model="m",
        messages=last_messages,
        response=json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "Here's your cat!"}}]}
        ),
        run_id="run-abc",
        agent_name=PennyConstants.CHAT_AGENT_NAME,
    )
    return [first, last]


def test_render_run_calls_shows_run_id_origin_tools_reply_and_egress_media():
    """read_run_calls' chat-run render is an anchor surface that shows what the
    turn actually did AND what went out (#1560, criteria 3 + 5 + the anchor
    invariant): the run's own id, the user's ask, the tool sequence, the reply,
    and the delivered media by addressable id — so a delivery is inspected by a
    read, not confabulated."""
    rendered = render_run_calls(_chat_run_with_generated_image(42))
    assert "run run-abc" in rendered  # the run names itself (addressable, typed)
    assert "user: draw me a cat" in rendered  # origin
    # Canonical projection: numbered step (typed id) + call syntax + compact result
    # — not first-person narration.
    assert "step 1: generate_image(description='a cat')" in rendered
    assert "=> Generated image #42" in rendered  # the step's outcome, by reference
    assert "penny: Here's your cat!" in rendered  # what went out (the reply)
    assert "attached: image #42" in rendered  # what went out WITH it, by id


def test_render_run_calls_step_numbers_are_stable_coordinates_with_gaps():
    """Step numbers are the run's FULL tool-call ordinals — a persisted coordinate,
    not render-time numbering (#1560/#1471).  A ``done`` mid-sequence consumes its
    index, so the shown steps keep their absolute numbers and the omitted one leaves
    a GAP — never a renumber, so ``(run_id, steps 2–5)`` means the same thing in
    every view."""
    run = [
        _prompt(
            [
                _call("browse", {"queries": ["a"]}),
                _call("done", {"success": False, "summary": "premature"}),
                _call("collection_write", {"memory": "games", "entries": [{"content": "x"}]}),
            ],
            outcome="worked",
            reason="wrote one",
        )
    ]
    rendered = render_run_calls(run)
    # done (index 2) is not shown; browse is step 1 and the write keeps its absolute
    # index 3 — the gap proves numbers aren't recomputed over the shown subset.
    assert "step 1: browse" in rendered
    assert "step 3: collection_write" in rendered
    assert "step 2:" not in rendered
