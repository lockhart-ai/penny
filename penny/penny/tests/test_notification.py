"""Telling the user about a finished collector cycle (#1911).

Three layers, tested where each lives:

- **The program's calls, and when a run has covered them** (``penny.program``) — pure,
  so every program shape a routine can have is a plain case rather than a live run.
- **The notify document** (``render_notification``) — pure, so the exact text the draw
  reads is pinned whole, including the nothing-matched form and a no-write program's.
- **The send gate + enqueue** (``CollectorNotifier.queue``) — the three declines that
  need runtime state, and the enqueue that is the successful handoff.

The micro-context's own prompt is pinned in ``tests/tools/test_micro_context_prompts.py``;
the trigger, the coverage exit and the run records are driven end to end through the real
collector loop in ``tests/agents/test_collector.py``.
"""

from __future__ import annotations

import pytest

from penny.agents.models import ToolCallRecord
from penny.notification import (
    CollectorNotifier,
    CycleCall,
    NotificationInput,
    NotificationOutcome,
    RelatedMessages,
    WrittenEntry,
    render_notification,
)
from penny.program import ProgramCall, covered_calls, is_covered, program_calls

pytestmark = pytest.mark.bare_db


_SURFACE = frozenset({"browse", "collection_write", "collection_read_latest", "log_read", "done"})


def _ran(*calls: tuple[str, bool]) -> list[ToolCallRecord]:
    """Executed records: ``(tool, succeeded)`` pairs, in the order they ran."""
    return [
        ToolCallRecord(tool=tool, arguments={}, failed=not succeeded) for tool, succeeded in calls
    ]


# ── The program's calls (#1911) ───────────────────────────────────────────────


def test_program_calls_read_both_dialects_and_ignore_prose():
    """A program's calls are read off its numbered steps, in both dialects a stored
    prompt is written in: the rendered ``N. tool(args)`` a taught routine produces, and
    the legacy hand-authored ``N. Call tool("x") …`` prose that names its call
    mid-sentence.  A step that calls nothing contributes nothing, and a step naming a
    tool this cycle does not run with is not a call it could cover."""
    rendered = (
        "Watch the timetable.\n"
        "1. browse(queries=['https://ex.example/t'], extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry', entries=[{'key': 'x', 'content': 'y'}])"
    )
    assert program_calls(rendered, _SURFACE) == (
        ProgramCall(ordinal=1, tool="browse"),
        ProgramCall(ordinal=2, tool="collection_write"),
    )

    hand_authored = (
        "You extract the user's negative preferences from their recent messages.\n"
        '1. Call log_read("user-messages") to fetch new messages you have not seen.\n'
        "2. Identify every genuine DISLIKE — a thing the user avoids.\n"
        '3. Call collection_write("dislikes", entries=[...]) once with all of them.'
    )
    assert program_calls(hand_authored, _SURFACE) == (
        ProgramCall(ordinal=1, tool="log_read"),
        ProgramCall(ordinal=3, tool="collection_write"),
    )

    # Prose alone names no call; a verb off this cycle's surface is not one either.
    assert program_calls("Watch the page and write down what changed.", _SURFACE) == ()
    assert program_calls("1. send_message(content=<the message>)", _SURFACE) == ()


def test_a_step_contributes_at_most_its_first_call():
    """One step is one move of the routine: a later mention in the same step is prose
    about it — a hand-authored step offering two ways to correct an entry is one step,
    not two calls to cover."""
    step = "1. Call update_entry(key=<key>, content=<new>) or collection_delete_entry(key=<key>)."
    surface = _SURFACE | {"update_entry", "collection_delete_entry"}
    assert program_calls(step, surface) == (ProgramCall(ordinal=1, tool="update_entry"),)


def test_coverage_advances_in_order_over_the_programs_own_calls():
    """The forward cursor across the program shapes a routine can have.

    A retry does not consume its step (the failed attempt is the same step trying
    again), a read the model interjected of its own accord does not count against
    coverage, a tool the program names twice needs two successful executions, and a run
    that took the steps out of order has not carried out the routine it was given."""
    program = program_calls(
        "1. browse(queries=['a'])\n2. collection_write(memory='m', entries=[])", _SURFACE
    )

    # Straight through.
    assert is_covered(program, _ran(("browse", True), ("collection_write", True)))
    # A failed call, then the retry: the failure passes over, the retry covers.
    assert is_covered(
        program, _ran(("browse", False), ("browse", True), ("collection_write", True))
    )
    # An interjected read between the steps costs nothing.
    assert is_covered(
        program,
        _ran(("browse", True), ("collection_read_latest", True), ("collection_write", True)),
    )
    # Half-run, and out of order, are both uncovered — and the cursor says how far.
    assert not is_covered(program, _ran(("browse", True)))
    assert covered_calls(program, _ran(("browse", True))) == 1
    assert not is_covered(program, _ran(("collection_write", True), ("browse", True)))

    # A tool the program names twice needs two successful executions of it.
    twice = program_calls("1. browse(queries=['a'])\n2. browse(queries=['b'])", _SURFACE)
    assert not is_covered(twice, _ran(("browse", True)))
    assert is_covered(twice, _ran(("browse", True), ("browse", True)))


def test_a_program_with_no_calls_is_never_covered():
    """An EMPTY program is not instantly complete — there is nothing to read, so the
    read cannot be what ends the cycle.  That is what keeps a purely prose prompt on
    its ``done()`` terminator instead of closing before it has done anything."""
    assert is_covered((), _ran(("browse", True))) is False
    assert is_covered((), []) is False


def test_a_program_with_no_write_at_all_is_covered_like_any_other():
    """Coverage is of the program's OWN calls, never of a tool identity: a routine that
    only reads completes exactly the way a routine that writes does.  (The code owner's
    correction: some skills will not have writes at all.)"""
    read_only = program_calls('1. log_read("user-messages")', _SURFACE)
    assert is_covered(read_only, _ran(("log_read", True)))


# ── The notify document (#1911) ───────────────────────────────────────────────


_DOCUMENT = NotificationInput(
    collection="ferry-departures",
    description="the dawn sailing on the north pier timetable",
    routine="check_timetable",
    calls=(
        CycleCall(
            call="browse(queries=['https://northpier.example/departures'])",
            result="EXTRACTED: 06:12",
        ),
        CycleCall(
            call="collection_write(memory='ferry-departures', entries=['06:12'])",
            result="Wrote 1 entry.",
        ),
    ),
    written=(
        WrittenEntry(memory="ferry-departures", key="dawn sailing", content="06:12 from the pier"),
    ),
    related=(
        RelatedMessages(
            source="user-messages",
            lines=("1. [2026-08-19 07:03 UTC] when does the dawn one leave?",),
        ),
        RelatedMessages(source="penny-messages", lines=()),
    ),
)


def test_notify_document_renders_whole():
    """The whole document, char-for-char — what the draw reads, assembled entirely
    framework-side: what ran, every call with its result verbatim, what landed in the
    store, and the past messages Python looked up.  A log that matched nothing renders
    no section of its own; only a document where NEITHER matched says so."""
    assert render_notification(_DOCUMENT) == (
        "The `ferry-departures` routine just ran on its own, and it is time to tell the user.\n"
        "What this collection is for: the dawn sailing on the north pier timetable\n"
        "The routine it runs: check_timetable\n"
        "\n"
        "## What the cycle did\n"
        "1. browse(queries=['https://northpier.example/departures'])\n"
        "   → EXTRACTED: 06:12\n"
        "2. collection_write(memory='ferry-departures', entries=['06:12'])\n"
        "   → Wrote 1 entry.\n"
        "\n"
        "## What it wrote this cycle\n"
        "- into `ferry-departures`, under `dawn sailing`: 06:12 from the pier\n"
        "\n"
        "## What the two of you have said about this before\n"
        "### user-messages\n"
        "1. [2026-08-19 07:03 UTC] when does the dawn one leave?"
    )


def test_notify_document_states_a_no_write_cycle_and_a_nothing_matched_lookup():
    """The two plain absences, rendered as statements rather than as gaps.

    A routine with NO write completes and notifies like any other, so "wrote nothing"
    is an ordinary shape here; and nothing matching in either message log is the
    ORDINARY case, so it says there is no callback to make — which is what stops a
    callback line being invented.  A failed call is marked as one, so the draw never
    reads an error as a find."""
    document = NotificationInput(
        collection="tide-times",
        description="the tide table for the north pier",
        calls=(
            CycleCall(call="log_read(memory='user-messages')", result="No entries."),
            CycleCall(
                call="browse(queries=['https://tides.example'])",
                result="## browse error: unreachable",
                failed=True,
            ),
        ),
        related=(
            RelatedMessages(source="user-messages", lines=()),
            RelatedMessages(source="penny-messages", lines=()),
        ),
    )

    assert render_notification(document) == (
        "The `tide-times` routine just ran on its own, and it is time to tell the user.\n"
        "What this collection is for: the tide table for the north pier\n"
        "\n"
        "## What the cycle did\n"
        "1. log_read(memory='user-messages')\n"
        "   → No entries.\n"
        "2. browse(queries=['https://tides.example'])\n"
        "   → (this call failed) ## browse error: unreachable\n"
        "\n"
        "## What it wrote this cycle\n"
        "It wrote nothing down this cycle.\n"
        "\n"
        "## What the two of you have said about this before\n"
        "Nothing in your past messages matched this — there is no callback to make."
    )


# ── The send gate + the enqueue (#1911) ───────────────────────────────────────


_RECIPIENT = "+15551234567"
_MECHANISM = "ferry-departures"
_MESSAGE = "Hey! The dawn sailing is at 06:12 now."


def _notifier(db) -> CollectorNotifier:
    from unittest.mock import MagicMock

    return CollectorNotifier(db, MagicMock(), MagicMock())


def _register_user(db) -> None:
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )


def test_a_drawn_message_is_queued_under_its_mechanism(db):
    """The handoff: the drawn text goes into the send queue unchanged, attributed to
    the collection whose cycle produced it, and waits there for the drainer to deliver
    it on the cooldown.  Enqueue IS the success — nothing about WHEN it goes out is
    decided here."""
    _register_user(db)

    assert _notifier(db).queue(_MECHANISM, _MESSAGE) is NotificationOutcome.QUEUED

    pending = db.send_queue.next_pending()
    assert pending is not None
    assert pending.content == _MESSAGE
    assert pending.collection == _MECHANISM
    assert pending.sent_at is None


def test_the_three_declines_queue_nothing(db):
    """The declines that need runtime state or are correct no-ops: a muted user, no
    registered recipient, and content that reads as a model refusal.  Each is the same
    enumerated outcome — nothing was sent — and none of them queues anything."""
    # No registered user at all.
    assert _notifier(db).queue(_MECHANISM, _MESSAGE) is NotificationOutcome.NOT_DELIVERABLE
    assert db.send_queue.next_pending() is None

    _register_user(db)
    refusal = "I'm sorry, I can't help with that as an AI language model."
    assert _notifier(db).queue(_MECHANISM, refusal) is NotificationOutcome.NOT_DELIVERABLE
    assert db.send_queue.next_pending() is None

    db.users.set_muted(_RECIPIENT)
    assert _notifier(db).queue(_MECHANISM, _MESSAGE) is NotificationOutcome.NOT_DELIVERABLE
    assert db.send_queue.next_pending() is None


def test_a_half_formed_message_never_reaches_the_queue(db):
    """The content gate the draw's own acceptance rule enforces one layer up, held
    again here so nothing reaches the queue unvalidated whatever put it there.  Judged
    as a WHOLE message: a trailing-off body is refused, while one that merely embeds an
    ellipsis mid-sentence is a complete message and goes out."""
    _register_user(db)
    notifier = _notifier(db)

    assert notifier.queue(_MECHANISM, "Hi there! ......???") is NotificationOutcome.NOT_DELIVERABLE
    assert db.send_queue.next_pending() is None

    assert notifier.queue(_MECHANISM, "anyway… that's the gist 🤓") is NotificationOutcome.QUEUED
    pending = db.send_queue.next_pending()
    assert pending is not None and pending.content == "anyway… that's the gist 🤓"


def test_send_queue_store_round_trip(db):
    """Enqueue → next_pending (FIFO) → mark_sent removes it from the pending tail."""
    first = db.send_queue.enqueue(content="one", collection="likes")
    db.send_queue.enqueue(content="two", collection="tide-times")

    pending = db.send_queue.next_pending()
    assert pending is not None and pending.id == first and pending.content == "one"
    assert [item.content for item in db.send_queue.pending_items()] == ["one", "two"]

    db.send_queue.mark_sent(first)
    nxt = db.send_queue.next_pending()
    assert nxt is not None and nxt.content == "two"
    assert [item.content for item in db.send_queue.pending_items()] == ["two"]
