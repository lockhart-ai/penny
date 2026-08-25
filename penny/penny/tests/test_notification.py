"""Telling the user about a finished collector cycle (#1911).

Three layers, tested where each lives:

- **The program's calls, and when a run has covered them** (``penny.program``) — pure,
  so every program shape a routine can have is a plain case rather than a live run.
- **The notify document** (``render_notification``) — pure, so the exact text the draw
  reads is pinned whole, including the nothing-matched form, a no-write program's, and
  (#1934) the bounded shape an oversized cycle assembles to.
- **The send gate + enqueue** (``CollectorNotifier.queue``) — the three declines that
  need runtime state, and the enqueue that is the successful handoff.

The micro-context's own prompt is pinned in ``tests/tools/test_micro_context_prompts.py``;
the trigger, the coverage exit and the run records are driven end to end through the real
collector loop in ``tests/agents/test_collector.py``.
"""

from __future__ import annotations

import pytest

from penny.constants import CycleTrigger, PennyConstants
from penny.notification import (
    CollectorNotifier,
    CycleCall,
    NotificationInput,
    NotificationOutcome,
    RelatedMessages,
    WrittenEntry,
    render_notification,
)
from penny.program import ProgramCall, program_calls

pytestmark = pytest.mark.bare_db


_SURFACE = frozenset({"browse", "collection_write", "collection_read_latest", "log_read", "done"})


# ── The program's calls (#1911) ───────────────────────────────────────────────


def test_program_calls_read_the_rendered_dialect_only():
    """A program's calls are read off its numbered steps in the ONE dialect that exists
    (#1911): the rendered ``N. tool(args)`` a taught routine produces.

    The call must OPEN the step.  Prose that merely NAMES a call — the seeded rows'
    ``4. Call collection_write("x", …) once with all of them`` — reads as no call at
    all, which is the point: those rows were dropped in the soft reboot, and a lenient
    parse over prose can only manufacture a program nobody wrote.  A step naming a tool
    this cycle does not run with is not a call it could cover either."""
    rendered = (
        "Watch the timetable.\n"
        "1. browse(queries=['https://ex.example/t'], extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry', entries=[{'key': 'x', 'content': 'y'}])"
    )
    assert program_calls(rendered, _SURFACE) == (
        ProgramCall(ordinal=1, tool="browse"),
        ProgramCall(ordinal=2, tool="collection_write"),
    )

    # THE WATCHED BOUNDARY: prose naming its calls mid-sentence is not a program.
    hand_authored = (
        "You extract the user's negative preferences from their recent messages.\n"
        '1. Call log_read("user-messages") to fetch new messages you have not seen.\n'
        "2. Identify every genuine DISLIKE — a thing the user avoids.\n"
        '3. Call collection_write("dislikes", entries=[...]) once with all of them.'
    )
    assert program_calls(hand_authored, _SURFACE) == ()

    # Prose alone names no call; a verb off this cycle's surface is not one either.
    assert program_calls("Watch the page and write down what changed.", _SURFACE) == ()
    assert program_calls("1. send_message(content=<the message>)", _SURFACE) == ()


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
    framework-side: what ran, what landed in the store, how the cycle got there with
    every call and its result, and the past messages Python looked up.

    The ORDINARY case is unchanged by the #1934 bound: nothing here overflows a section
    budget, so every value is verbatim and no omission marker appears anywhere.  What
    #1934 did change is the ORDER — the entries the cycle wrote lead, the earlier
    conversation is labelled background and comes last.  A log that matched nothing
    renders no section of its own; only a document where NEITHER matched says so."""
    assert render_notification(_DOCUMENT) == (
        "The `ferry-departures` routine just ran on its own, and it is time to tell the user.\n"
        "What this collection is for: the dawn sailing on the north pier timetable\n"
        "The routine it runs: check_timetable\n"
        "\n"
        "## What it wrote this cycle\n"
        "- into `ferry-departures`, under `dawn sailing`: 06:12 from the pier\n"
        "\n"
        "## How it got there\n"
        "1. browse(queries=['https://northpier.example/departures'])\n"
        "   → EXTRACTED: 06:12\n"
        "2. collection_write(memory='ferry-departures', entries=['06:12'])\n"
        "   → Wrote 1 entry.\n"
        "\n"
        "## Background: earlier messages\n"
        "These messages are from before this cycle ran. What this cycle found is at the "
        "top of this document.\n"
        "\n"
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
        "## What it wrote this cycle\n"
        "It wrote nothing down this cycle.\n"
        "\n"
        "## How it got there\n"
        "1. log_read(memory='user-messages')\n"
        "   → No entries.\n"
        "2. browse(queries=['https://tides.example'])\n"
        "   → (this call failed) ## browse error: unreachable\n"
        "\n"
        "## Background: earlier messages\n"
        "Nothing in your past messages matched this — there is no callback to make."
    )


# ── The document's bound and its order (#1934) ────────────────────────────────


# The failure shape, recreated with synthetic content: one browse call whose RESULT is
# a whole fetched page, one write call whose ARGUMENTS restate the payload it carried,
# entries holding that same text a third time, more calls and entries than a section
# shows, and a background section of long prior messages.  The measured production
# document assembled to 50,805 characters this way.
_PAGE = "## browse: https://harbour.example/\n" + "\n".join(
    f"{n}. the keel lantern on pier {n} went dark overnight" for n in range(1, 400)
)
_DIGEST = " ".join(f"pier {n} dark;" for n in range(1, 900))
_WRITE_CALL = (
    "collection_write(memory='harbour-watch', "
    f"entries=[{{'key': 'lanterns', 'content': '{_DIGEST}'}}])"
)
_LONG_MESSAGE = "1. [2026-01-02 07:03 UTC] " + " ".join(
    f"which pier was it, {n}?" for n in range(1, 60)
)
# An entry KEY is model-authored and length-gated nowhere upstream, and it renders once
# per written entry — so it is bounded here like every other free-text field.
_LONG_KEY = "lanterns-" + "-".join(str(n) for n in range(1, 60))

_OVERSIZED = NotificationInput(
    collection="harbour-watch",
    description="the pier lanterns on the harbour front",
    routine="watch_lanterns",
    calls=(
        CycleCall(call="browse(queries=['https://harbour.example/'])", result=_PAGE),
        CycleCall(call=_WRITE_CALL, result="Wrote 10 entries to 'harbour-watch'."),
        *(
            CycleCall(
                call=f"collection_read_latest(memory='harbour-watch', k={n})",
                result="No entries.",
            )
            for n in range(3, 12)
        ),
    ),
    written=(
        WrittenEntry(memory="harbour-watch", key=_LONG_KEY, content=_DIGEST),
        *(
            WrittenEntry(memory="harbour-watch", key=f"pier-{n}", content=f"pier {n} is dark")
            for n in range(2, 11)
        ),
    ),
    related=(
        RelatedMessages(
            source="user-messages",
            lines=(
                _LONG_MESSAGE,
                *(f"{n}. [2026-01-0{n} 07:03 UTC] which lantern?" for n in range(2, 8)),
            ),
        ),
        RelatedMessages(
            source="penny-messages",
            lines=(
                "1. [2026-01-08 09:00 UTC] (sent by harbour-watch) "
                "The pier 1 lantern went dark overnight.",
            ),
        ),
    ),
)

# The document's declared ceiling: every free-text field's own budget, plus an allowance
# for the frame those budgets sit in (the headings, the background lead, the per-item
# prefixes, the memory and routine names, and the omission markers themselves).  Derived
# from the constants rather than picked, so tightening a budget tightens the claim with
# it — and a field left OUT of this sum is a field that can defeat the ceiling, which is
# why the entry key is in it.
_SECTION_BUDGETS = (
    PennyConstants.NOTIFY_WRITTEN_ENTRIES
    * (PennyConstants.NOTIFY_WRITTEN_CONTENT_CHARS + PennyConstants.NOTIFY_ENTRY_KEY_CHARS)
    + PennyConstants.NOTIFY_CYCLE_CALLS
    * (PennyConstants.NOTIFY_CALL_CHARS + PennyConstants.NOTIFY_CALL_RESULT_CHARS)
    + 2 * PennyConstants.NOTIFY_RELATED_MESSAGES * PennyConstants.NOTIFY_RELATED_LINE_CHARS
    + PennyConstants.NOTIFY_DESCRIPTION_CHARS
)
_FRAME_ALLOWANCE = 2_000
_DECLARED_BOUND = _SECTION_BUDGETS + _FRAME_ALLOWANCE


def test_an_oversized_cycle_assembles_under_the_declared_bound():
    """The whole bounded document, char-for-char.

    Every one of the three bulk carriers is condensed against its own budget and every
    cut STATES itself — the page-sized browse result, the write call whose arguments
    restate its payload, and the entry holding that text again — as is the entry KEY
    nothing upstream length-gates, and each section that holds more items than it shows
    ends in an honest count of the rest.  The entries the cycle wrote still LEAD, the
    earlier conversation is still labelled background and still last, and the whole
    thing lands under the ceiling the constants declare."""
    rendered = render_notification(_OVERSIZED)

    assert rendered == (
        "The `harbour-watch` routine just ran on its own, and it is time to tell the user.\n"
        "What this collection is for: the pier lanterns on the harbour front\n"
        "The routine it runs: watch_lanterns\n"
        "\n"
        "## What it wrote this cycle\n"
        f"- into `harbour-watch`, under `{_LONG_KEY[:120]}"
        f"… [{len(_LONG_KEY) - 120} characters omitted]`: {_DIGEST[:500]}"
        f"… [{len(_DIGEST) - 500} characters omitted]\n"
        "- into `harbour-watch`, under `pier-2`: pier 2 is dark\n"
        "- into `harbour-watch`, under `pier-3`: pier 3 is dark\n"
        "- into `harbour-watch`, under `pier-4`: pier 4 is dark\n"
        "- into `harbour-watch`, under `pier-5`: pier 5 is dark\n"
        "- into `harbour-watch`, under `pier-6`: pier 6 is dark\n"
        "- into `harbour-watch`, under `pier-7`: pier 7 is dark\n"
        "- into `harbour-watch`, under `pier-8`: pier 8 is dark\n"
        "2 more entries not shown.\n"
        "\n"
        "## How it got there\n"
        "1. browse(queries=['https://harbour.example/'])\n"
        f"   → {_PAGE[:300]}… [{len(_PAGE) - 300} characters omitted]\n"
        f"2. {_WRITE_CALL[:150]}… [{len(_WRITE_CALL) - 150} characters omitted]\n"
        "   → Wrote 10 entries to 'harbour-watch'.\n"
        "3. collection_read_latest(memory='harbour-watch', k=3)\n"
        "   → No entries.\n"
        "4. collection_read_latest(memory='harbour-watch', k=4)\n"
        "   → No entries.\n"
        "5. collection_read_latest(memory='harbour-watch', k=5)\n"
        "   → No entries.\n"
        "6. collection_read_latest(memory='harbour-watch', k=6)\n"
        "   → No entries.\n"
        "7. collection_read_latest(memory='harbour-watch', k=7)\n"
        "   → No entries.\n"
        "8. collection_read_latest(memory='harbour-watch', k=8)\n"
        "   → No entries.\n"
        "9. collection_read_latest(memory='harbour-watch', k=9)\n"
        "   → No entries.\n"
        "10. collection_read_latest(memory='harbour-watch', k=10)\n"
        "   → No entries.\n"
        "1 more call not shown.\n"
        "\n"
        "## Background: earlier messages\n"
        "These messages are from before this cycle ran. What this cycle found is at the "
        "top of this document.\n"
        "\n"
        "### user-messages\n"
        f"{_LONG_MESSAGE[:200]}… [{len(_LONG_MESSAGE) - 200} characters omitted]\n"
        "2. [2026-01-02 07:03 UTC] which lantern?\n"
        "3. [2026-01-03 07:03 UTC] which lantern?\n"
        "4. [2026-01-04 07:03 UTC] which lantern?\n"
        "5. [2026-01-05 07:03 UTC] which lantern?\n"
        "2 more messages not shown.\n"
        "\n"
        "### penny-messages\n"
        "1. [2026-01-08 09:00 UTC] (sent by harbour-watch) "
        "The pier 1 lantern went dark overnight."
    )

    # The input carried the production failure's bulk; the render carries the budget.
    raw = len(_PAGE) + len(_WRITE_CALL) + len(_DIGEST) * 2 + len(_LONG_MESSAGE)
    assert raw > 50_000
    assert len(rendered) < _DECLARED_BOUND


def test_a_prior_notification_cannot_displace_this_cycles_writes():
    """The failure this order exists to stop: the previous notification's own lead sat
    in the closest-prior-messages section, and the composed message re-reported it as
    today's top story.

    A prior message repeating what an OLDER cycle found now reaches the draw last, in a
    section that says what it is, and this cycle's own write is the first thing after
    the header — so the freshest fact is the one nearest the top, whatever a background
    line happens to say."""
    document = NotificationInput(
        collection="harbour-watch",
        description="the pier lanterns on the harbour front",
        calls=(CycleCall(call="browse(queries=['https://harbour.example/'])", result="read"),),
        written=(WrittenEntry(memory="harbour-watch", key="lanterns", content="pier 4 went dark"),),
        related=(
            RelatedMessages(source="user-messages", lines=()),
            RelatedMessages(
                source="penny-messages",
                lines=("1. [2026-01-08 09:00 UTC] (sent by harbour-watch) pier 1 went dark",),
            ),
        ),
    )

    rendered = render_notification(document)
    assert rendered == (
        "The `harbour-watch` routine just ran on its own, and it is time to tell the user.\n"
        "What this collection is for: the pier lanterns on the harbour front\n"
        "\n"
        "## What it wrote this cycle\n"
        "- into `harbour-watch`, under `lanterns`: pier 4 went dark\n"
        "\n"
        "## How it got there\n"
        "1. browse(queries=['https://harbour.example/'])\n"
        "   → read\n"
        "\n"
        "## Background: earlier messages\n"
        "These messages are from before this cycle ran. What this cycle found is at the "
        "top of this document.\n"
        "\n"
        "### penny-messages\n"
        "1. [2026-01-08 09:00 UTC] (sent by harbour-watch) pier 1 went dark"
    )

    # This cycle's find leads; the stale one is reachable only inside the labelled
    # background section, which is the last thing in the document.
    assert rendered.index("pier 4 went dark") < rendered.index("## Background: earlier messages")
    assert rendered.index("pier 1 went dark") > rendered.index("## Background: earlier messages")
    assert rendered.index("## What it wrote this cycle") < rendered.index("## How it got there")


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
    it.  Enqueue IS the success — nothing about WHEN it goes out is decided here, but
    the row does carry WHAT SET THE CYCLE RUNNING (#1939), which is what the drainer
    reads to pick a delivery lane."""
    _register_user(db)

    assert (
        _notifier(db).queue(_MECHANISM, _MESSAGE, CycleTrigger.CADENCE)
        is NotificationOutcome.QUEUED
    )

    pending = db.send_queue.next_pending()
    assert pending is not None
    assert pending.content == _MESSAGE
    assert pending.collection == _MECHANISM
    assert pending.origin == CycleTrigger.CADENCE
    assert pending.sent_at is None


def test_a_user_triggered_cycles_message_is_queued_in_the_on_demand_lane(db):
    """The same handoff for a cycle the USER pressed "run this now" on: the queued row
    says so, so the drainer can deliver it without waiting out the anti-spam cooldown
    meant for messages nobody asked for (#1939).  The lane is a READ of what set the
    cycle running — nothing here re-decides it from the content."""
    _register_user(db)

    assert (
        _notifier(db).queue(_MECHANISM, _MESSAGE, CycleTrigger.ON_DEMAND)
        is NotificationOutcome.QUEUED
    )

    pending = db.send_queue.next_pending()
    assert pending is not None
    assert pending.origin == CycleTrigger.ON_DEMAND
    # And it is reachable BY lane, which is how the drainer asks for it.
    on_demand = db.send_queue.next_pending(origin=CycleTrigger.ON_DEMAND)
    assert on_demand is not None and on_demand.id == pending.id
    assert db.send_queue.next_pending(origin=CycleTrigger.CADENCE) is None


def test_the_three_declines_queue_nothing(db):
    """The declines that need runtime state or are correct no-ops: a muted user, no
    registered recipient, and content that reads as a model refusal.  Each is the same
    enumerated outcome — nothing was sent — and none of them queues anything."""
    # No registered user at all.
    assert (
        _notifier(db).queue(_MECHANISM, _MESSAGE, CycleTrigger.CADENCE)
        is NotificationOutcome.NOT_DELIVERABLE
    )
    assert db.send_queue.next_pending() is None

    _register_user(db)
    refusal = "I'm sorry, I can't help with that as an AI language model."
    assert (
        _notifier(db).queue(_MECHANISM, refusal, CycleTrigger.CADENCE)
        is NotificationOutcome.NOT_DELIVERABLE
    )
    assert db.send_queue.next_pending() is None

    db.users.set_muted(_RECIPIENT)
    # A user-triggered run is no exception: the lane says WHEN a message goes out, not
    # WHETHER — a muted user is still muted when they press the button.
    assert (
        _notifier(db).queue(_MECHANISM, _MESSAGE, CycleTrigger.ON_DEMAND)
        is NotificationOutcome.NOT_DELIVERABLE
    )
    assert db.send_queue.next_pending() is None


def test_a_half_formed_message_never_reaches_the_queue(db):
    """The content gate the draw's own acceptance rule enforces one layer up, held
    again here so nothing reaches the queue unvalidated whatever put it there.  Judged
    as a WHOLE message: a trailing-off body is refused, while one that merely embeds an
    ellipsis mid-sentence is a complete message and goes out."""
    _register_user(db)
    notifier = _notifier(db)

    assert (
        notifier.queue(_MECHANISM, "Hi there! ......???", CycleTrigger.CADENCE)
        is NotificationOutcome.NOT_DELIVERABLE
    )
    assert db.send_queue.next_pending() is None

    assert (
        notifier.queue(_MECHANISM, "anyway… that's the gist 🤓", CycleTrigger.CADENCE)
        is NotificationOutcome.QUEUED
    )
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
