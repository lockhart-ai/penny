"""Telling the user about a finished collector cycle (#1911, reshaped by #2656).

Three layers, tested where each lives:

- **The job report** (``render_job_report`` / ``build_job_report``) — the delivery
  turn's whole input, pure, so the exact text the chat agent reads is pinned whole:
  the job's name and description, why it ran, and every written entry FULL, with the
  key it is stored under.  Nothing is condensed and nothing is capped — the composer's
  cut was the measured failure (#2656).
- **The send gate** (``withheld_reason``) — the declines that need runtime state or
  are correct no-ops, read as reasons the drainer records.
- **The composing report on the queue** (``db.send_queue``) — the hand-off row the
  cycle enqueues and the drainer composes from: its states, its bounded attempts,
  and the round trip.

The delivery turn itself is driven end to end in ``tests/test_report_delivery.py``;
the trigger and the run records are driven through the real collector loop in
``tests/agents/test_collector.py``.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session

from penny.constants import CycleEnd, CycleTrigger, PennyConstants
from penny.database.models import SendQueueItem
from penny.notification import (
    NOTIFICATION_NOTES,
    NOTIFIED_CYCLE_ENDS,
    JobReport,
    NotificationOutcome,
    WrittenEntry,
    build_job_report,
    render_job_report,
    withheld_reason,
)

pytestmark = pytest.mark.bare_db


# ── The job report (#2656) ────────────────────────────────────────────────────


def test_a_job_report_renders_whole():
    """The whole delivery turn's input, char-for-char — assembled entirely
    framework-side: the job's name, what it is for, why it ran, and the written
    entries FULL, in the order the cycle wrote them, each with the key it is stored
    under.  The key is the anchor a follow-up needs; the content is the payload the
    message is written from.  No entry is condensed and no section is capped."""
    report = JobReport(
        collection="ferry-departures",
        description="the dawn sailing on the north pier timetable",
        trigger=CycleTrigger.CADENCE,
        run_id="cycle-run-1",
        entries=(
            WrittenEntry(
                memory="ferry-departures", key="dawn sailing", content="06:12 from the pier"
            ),
        ),
    )
    assert render_job_report(report) == (
        'Your job "ferry-departures" just finished a run and asked to tell the user.\n'
        "What the job is for: the dawn sailing on the north pier timetable\n"
        "It ran on its own schedule.\n"
        "\n"
        "## What the job wrote this run\n"
        "These entries are what the job kept for the user this run — the message "
        "tells them about every one.\n"
        "- into `ferry-departures`, under `dawn sailing`: 06:12 from the pier\n"
        "\n"
        "Write the one message to send the user about this run, from what the job wrote."
    )


def test_a_report_states_a_no_write_cycle_and_a_user_triggered_run():
    """The two plain absences, rendered as statements rather than as gaps.

    A routine with NO write completes and notifies like any other, so "wrote nothing"
    is an ordinary shape here; and a cycle the user pressed "run this now" on says
    so — the same fact the queued row's lane carries."""
    report = JobReport(
        collection="tide-times",
        description="the tide table for the north pier",
        trigger=CycleTrigger.ON_DEMAND,
        run_id="cycle-run-2",
        entries=(),
    )
    assert render_job_report(report) == (
        'Your job "tide-times" just finished a run and asked to tell the user.\n'
        "What the job is for: the tide table for the north pier\n"
        "It ran just now because the user asked it to run.\n"
        "\n"
        "## What the job wrote this run\n"
        "It wrote nothing down this run.\n"
        "\n"
        "Write the one message to send the user about this run, from what the job wrote."
    )


def test_a_keyless_entry_renders_without_a_key(db):
    """A log append has no key, and its line says where it went and nothing else."""
    report = JobReport(
        collection="tide-log",
        description="the day's tides",
        trigger=CycleTrigger.CADENCE,
        run_id="cycle-run-3",
        entries=(WrittenEntry(memory="tide-log", key=None, content="high at 06:41"),),
    )
    rendered = render_job_report(report)
    assert "- into `tide-log`: high at 06:41" in rendered


def test_a_built_report_carries_the_full_written_entries(db):
    """The assembly reads the STORE: the collection row's own description, and every
    entry whose current value the run wrote — WHOLE, whatever its length.  The
    composer's document condensed an entry to a 500-character preview and the message
    named only what the preview held (#2656's failure); this is the fix's other half —
    the chat agent is shown everything."""
    from penny.database.memory.objects import EntryInput
    from penny.tests.conftest import require_memory

    db.memories.create_collection("harbour-watch", "the pier lanterns on the harbour front")
    require_memory(db, "harbour-watch").write(
        [EntryInput(key="lanterns", content="pier 4 went dark overnight")],
        author="producer",
        run_id="cycle-run-9",
    )
    built = build_job_report(db, "harbour-watch", "cycle-run-9", trigger=CycleTrigger.CADENCE)
    assert built is not None
    assert built.collection == "harbour-watch"
    assert built.description == "the pier lanterns on the harbour front"
    assert built.trigger is CycleTrigger.CADENCE
    assert built.entries == (
        WrittenEntry(memory="harbour-watch", key="lanterns", content="pier 4 went dark overnight"),
    )
    # And the render carries the entry WHOLE — no cut, no omission marker.
    assert "- into `harbour-watch`, under `lanterns`: pier 4 went dark overnight" in (
        render_job_report(built)
    )


def test_a_report_for_a_vanished_collection_is_a_named_failure(db):
    """A collection that vanished between the enqueue and the compose builds no
    report: the caller records the failure on the row rather than composing from a
    half-assembled one."""
    assert build_job_report(db, "no-such-job", "run-x", trigger=CycleTrigger.CADENCE) is None


# ── The send gate (#1911, as a reason) ────────────────────────────────────────


_RECIPIENT = "+15551234567"
_MECHANISM = "ferry-departures"
_MESSAGE = "Hey! The dawn sailing is at 06:12 now."


def _register_user(db) -> None:
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )


def test_a_sendable_message_is_declined_nowhere(db):
    """The pass case: a complete, non-refusal message from a registered, unmuted user
    is declined nowhere, and the drainer stores and delivers it."""
    _register_user(db)
    assert withheld_reason(db, _MESSAGE) is None


def test_the_declines_name_why_nothing_will_be_sent(db):
    """The declines that need runtime state or are correct no-ops — decisions no
    retry changes: content that reads as a model refusal, no registered recipient,
    and a muted user.  Each names its reason — the drainer records it on the row —
    and none of them is a silent skip."""
    # No registered user at all.
    assert withheld_reason(db, _MESSAGE) == "there is no registered recipient"

    _register_user(db)
    refusal = withheld_reason(db, "I'm sorry, I can't help with that as an AI language model.")
    assert refusal == "the message read as a model refusal"

    db.users.set_muted(_RECIPIENT)
    # A user-triggered run is no exception: the lane says WHEN a message goes out, not
    # WHETHER — a muted user is still muted when the compose happens.
    assert withheld_reason(db, _MESSAGE) == "the user is muted"


def test_a_message_that_is_not_whole_is_a_compose_failure_not_a_decline(db):
    """The whole-message rule is a RETRY, not a withhold: a trailing-off body is a
    compose failure (the drainer counts it against the row's bounded budget, the way
    the retired composer re-rolled it), while one that merely embeds an ellipsis
    mid-sentence is a complete message and goes out.  The rule is the shared
    ``half_formed_send_reason`` — the one definition the send path validates with."""
    from penny.text_validity import half_formed_send_reason

    assert half_formed_send_reason("Hi there! ......???") is not None
    assert half_formed_send_reason("anyway… that's the gist 🤓") is None


# ── The composing report on the queue (#2656) ─────────────────────────────────


def test_a_report_row_is_queued_without_content(db):
    """The hand-off: the cycle's row carries the collection, the lane and the run id,
    and NO content yet — the message is written later, by the delivery turn the
    drainer runs.  It is reachable as the pending head either lane-blind or by lane."""
    row_id = db.send_queue.enqueue_report(_MECHANISM, CycleTrigger.CADENCE, "cycle-run-1")

    pending = db.send_queue.next_pending()
    assert pending is not None and pending.id == row_id
    assert pending.content == ""
    assert pending.collection == _MECHANISM
    assert pending.report_run_id == "cycle-run-1"
    assert pending.origin == CycleTrigger.CADENCE
    assert db.send_queue.next_pending(origin=CycleTrigger.CADENCE) is not None
    assert db.send_queue.next_pending(origin=CycleTrigger.ON_DEMAND) is None


def test_a_composed_report_is_stored_then_delivered_and_marked_sent(db):
    """The compose-and-deliver round trip on the row: the turn's message is stored on
    the row (so the audit trail says what went out), delivery marks it sent, and the
    pending tail empties."""
    db.send_queue.enqueue_report(_MECHANISM, CycleTrigger.ON_DEMAND, "cycle-run-1")
    row = db.send_queue.next_pending()
    assert row is not None

    db.send_queue.store_composed(row.id, _MESSAGE)
    stored = db.send_queue.next_pending()
    assert stored is not None and stored.content == _MESSAGE
    assert stored.report_run_id == "cycle-run-1"

    db.send_queue.mark_sent(row.id)
    assert db.send_queue.next_pending() is None


def test_compose_failures_are_counted_then_failed_visibly(db):
    """The bounded retry: each failed compose bumps ``attempts`` and leaves the row
    pending, until the budget is spent and the row is stamped ``failed_at`` — kept as
    the audit trail, excluded from the pending queries structurally, never counted
    as a message the user received."""
    db.send_queue.enqueue_report(_MECHANISM, CycleTrigger.CADENCE, "cycle-run-1")
    row = db.send_queue.next_pending()
    assert row is not None

    for expected_attempt in range(1, PennyConstants.COLLECTOR_RETRY_ATTEMPTS):
        assert db.send_queue.bump_compose_attempts(row.id) == expected_attempt
        assert db.send_queue.next_pending() is not None

    assert db.send_queue.bump_compose_attempts(row.id) == PennyConstants.COLLECTOR_RETRY_ATTEMPTS
    db.send_queue.mark_failed(row.id)

    assert db.send_queue.next_pending() is None
    with Session(db.engine) as session:
        final = session.get(SendQueueItem, row.id)
        assert final is not None
        assert final.failed_at is not None
        assert final.sent_at is None
        assert final.cancelled_at is None


def test_a_declined_report_is_cancelled_not_failed(db):
    """A withhold on purpose is a DECISION, not a failure: the row is stamped
    ``cancelled_at`` — the same audit-trail state an archived collector's queued
    sends get (#1634) — and stays distinct from a failed compose."""
    db.send_queue.enqueue_report(_MECHANISM, CycleTrigger.CADENCE, "cycle-run-1")
    row = db.send_queue.next_pending()
    assert row is not None

    db.send_queue.mark_cancelled(row.id)
    assert db.send_queue.next_pending() is None
    with Session(db.engine) as session:
        final = session.get(SendQueueItem, row.id)
        assert final is not None
        assert final.cancelled_at is not None
        assert final.failed_at is None
        assert final.sent_at is None


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


# ── The outcome vocabulary ────────────────────────────────────────────────────


def test_the_outcome_tables_are_total():
    """Both tables are indexed with NO default — an outcome in neither, or a note
    missing, would be a KeyError inside a cycle's settlement or a silent second
    answer, so the totality is pinned here rather than left to whoever adds the
    next outcome.  ``NOT_DRAWN`` retired with the composing job (#2656): a message
    that is never written is no longer a cycle outcome at all."""
    assert set(NOTIFIED_CYCLE_ENDS) == set(NotificationOutcome)
    assert set(NOTIFIED_CYCLE_ENDS.values()) == {
        CycleEnd.CLOSED_NOTIFIED,
        CycleEnd.CLOSED_DECLINED,
    }
    assert set(NOTIFICATION_NOTES) == set(NotificationOutcome)
    assert "NOT_DRAWN" not in NotificationOutcome.__members__
