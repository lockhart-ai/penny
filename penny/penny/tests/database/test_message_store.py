"""Tests for MessageStore conversation queries."""

from datetime import UTC, datetime, timedelta

from sqlmodel import Session

from penny.clock import Clock
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import MessageLog
from penny.tests.schema_template import migrated_db

# A controllable clock (#3125) — read ``_when`` verbatim, advanced only by
# direct reassignment, never real time.  Passed into ``migrated_db``'s own
# ``clock=`` constructor parameter so EVERY store (``db.messages`` included)
# shares this exact instance — unlike reassigning ``db.clock`` after
# construction, which only the Database facade's own direct readers
# (``db.clock.now()``) would see, never a store's own clock captured at
# ``__init__`` (``MessageStore._clock``, #3125's own new ``now()`` reads).


class _FixedClock(Clock):
    def __init__(self, when: datetime) -> None:
        self._when = when

    def now(self) -> datetime:
        return self._when


# One synthetic instant, to the microsecond — what every row in an ordering test is
# stamped with so the tie those tests are about happens on every run.
ONE_INSTANT = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)


def _log_user_message(db: Database, sender: str, content: str) -> int | None:
    return db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING,
        sender,
        content,
    )


def _log_threaded_reply(db: Database, recipient: str, content: str, parent_id: int) -> int | None:
    return db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        content,
        parent_id=parent_id,
        recipient=recipient,
    )


def _log_autonomous_send(db: Database, recipient: str, content: str) -> int | None:
    """Mirror what ``send_message`` produces — outgoing message with no
    parent thread.  This is the codepath the bug fix targets."""
    return db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        content,
        parent_id=None,
        recipient=recipient,
    )


def _pin_to_one_instant(db: Database, *message_ids: int | None) -> None:
    """Stamp every given row with the SAME timestamp, to the microsecond.

    ``log_message`` stamps ``now()``, so whether two consecutive writes land in one
    microsecond is a property of the machine rather than of the code under test — on a
    slow run they separate and the tie never happens.  Pinning them makes it happen every
    time.  Ids are untouched, so they still carry the order the rows were written in,
    which is the only remaining fact about which one came first."""
    with Session(db.engine) as session:
        for message_id in message_ids:
            row = session.get(MessageLog, message_id)
            assert row is not None, f"message {message_id} must exist to be pinned"
            row.timestamp = ONE_INSTANT
            session.add(row)
        session.commit()


class TestGetMessagesSinceIncludesAutonomousOutgoing:
    """Regression: ``send_message`` lands as parent_id=None — chat-turn
    builder needs to surface those so Penny sees her prior turn when the
    user replies to a notification."""

    USER = "+15551234567"

    def test_autonomous_send_appears_in_chat_turns(self, db):
        """The bug: a collector cycle's ``send_message`` followed by a user
        reply should produce a two-turn history, not a one-turn history."""

        # Penny says something autonomously (collector's send_message).
        _log_autonomous_send(db, self.USER, "your appointment is tomorrow at 2pm")
        # User replies — fresh message, no quote-reply, parent_id=None.
        _log_user_message(db, self.USER, "what time?")

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)

        assert [m.content for m in messages] == [
            "your appointment is tomorrow at 2pm",
            "what time?",
        ]
        assert messages[0].direction == PennyConstants.MessageDirection.OUTGOING
        assert messages[1].direction == PennyConstants.MessageDirection.INCOMING

    def test_threaded_replies_still_included(self, db):
        """Quote-replies (parent_id set) keep working alongside autonomous sends."""
        incoming_id = _log_user_message(db, self.USER, "hey penny")
        assert incoming_id is not None
        _log_threaded_reply(db, self.USER, "hey there", parent_id=incoming_id)

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)
        assert [m.content for m in messages] == ["hey penny", "hey there"]

    def test_mixed_autonomous_and_threaded(self, db):
        """A real conversation has both shapes — incoming, threaded reply,
        autonomous notification, incoming reply.  All four flow into chat
        turns in chronological order."""
        msg_id = _log_user_message(db, self.USER, "morning")
        assert msg_id is not None
        _log_threaded_reply(db, self.USER, "morning!", parent_id=msg_id)
        _log_autonomous_send(db, self.USER, "by the way, your appointment is at 2pm")
        _log_user_message(db, self.USER, "thanks")

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)
        assert [m.content for m in messages] == [
            "morning",
            "morning!",
            "by the way, your appointment is at 2pm",
            "thanks",
        ]

    def test_since_filter_drops_old_autonomous_sends(self, db):
        """Stale notifications from before the window don't bleed into
        the current conversation."""
        old_id = _log_autonomous_send(db, self.USER, "old notification")
        # Backdate the stale send so it's clearly before our cutoff —
        # ``log_message`` stamps with ``now()`` and the rest of this test
        # runs in microseconds.
        with Session(db.engine) as session:
            row = session.get(MessageLog, old_id)
            assert row is not None
            row.timestamp = datetime.now(UTC) - timedelta(hours=1)
            session.add(row)
            session.commit()

        cutoff = datetime.now(UTC) - timedelta(minutes=1)
        _log_user_message(db, self.USER, "hi")

        messages = db.messages.get_messages_since(self.USER, since=cutoff, limit=20)
        assert "old notification" not in [m.content for m in messages]
        assert "hi" in [m.content for m in messages]

    def test_autonomous_send_to_other_recipient_not_included(self, db):
        """Autonomous sends to a different user don't leak into this user's
        chat turns."""
        _log_autonomous_send(db, "+15559999999", "for someone else")
        _log_user_message(db, self.USER, "hey")

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)
        assert [m.content for m in messages] == ["hey"]


class TestMessagesInOneMicrosecondKeepTheirWriteOrder:
    """A burst of messages can share a timestamp to the microsecond, and the conversation
    still has to read back in the order it was said.

    ``get_messages_since`` merges three queries — the user's messages, Penny's threaded
    replies to them, and her autonomous sends — and on ``timestamp`` alone that merge is a
    stable sort over a concatenation: every incoming row lands ahead of every outgoing one
    whatever the write order was.  Measured, that swapped a reply and the message answering
    it in a seeded conversation, which is the whole history the model then reads."""

    USER = "+15551234567"

    def test_a_reply_stays_ahead_of_the_message_that_follows_it(self, db):
        """The failing shape: Penny answers, the user says thanks, and both land in the
        same microsecond.  The thanks is incoming and the answer is a threaded reply, so
        with nothing separating them the thanks comes back FIRST — the conversation reads
        as if she were thanked for something she had not said yet."""
        ask_id = _log_user_message(db, self.USER, "is the ferry running?")
        assert ask_id is not None
        reply_id = _log_threaded_reply(db, self.USER, "yep, on the hour", parent_id=ask_id)
        ack_id = _log_user_message(db, self.USER, "thanks!")
        _pin_to_one_instant(db, ask_id, reply_id, ack_id)

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)
        assert [m.content for m in messages] == [
            "is the ferry running?",
            "yep, on the hour",
            "thanks!",
        ]
        # The other oldest-first reader of the same rows agrees with it.
        assert db.messages.recent_conversation(limit=20) == [
            (PennyConstants.MessageDirection.INCOMING, "is the ferry running?"),
            (PennyConstants.MessageDirection.OUTGOING, "yep, on the hour"),
            (PennyConstants.MessageDirection.INCOMING, "thanks!"),
        ]

    def test_an_autonomous_send_stays_ahead_of_the_reply_to_it(self, db):
        """The same tie between an autonomous send and a user message: Penny says
        something unprompted and the user comes straight back at it, in one microsecond.
        Autonomous sends are the LAST leg of the merge, so this is the order that inverts
        — and inverting it is what makes a reply read as an unanswered question."""
        send_id = _log_autonomous_send(db, self.USER, "the ferry is delayed")
        reply_id = _log_user_message(db, self.USER, "by how long?")
        _pin_to_one_instant(db, send_id, reply_id)

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)
        assert [m.content for m in messages] == ["the ferry is delayed", "by how long?"]

    def test_a_send_written_after_a_user_message_stays_after_it(self, db):
        """The SAME pair in the other write order, on the same microsecond — the one the
        concatenation happened to get right by putting incoming first.  Pinned because a
        tiebreaker that fixed one direction by inverting the other would fix nothing."""
        ask_id = _log_user_message(db, self.USER, "anything on the ferry?")
        send_id = _log_autonomous_send(db, self.USER, "the ferry is delayed")
        _pin_to_one_instant(db, ask_id, send_id)

        messages = db.messages.get_messages_since(self.USER, since=datetime.min, limit=20)
        assert [m.content for m in messages] == [
            "anything on the ferry?",
            "the ferry is delayed",
        ]


class TestLiveStepMarkerPublishing:
    """publish_call_started/publish_tool_started/publish_tool_finished (#2439) —
    the same subscriber fan-out pattern subscribe_prompt_logged already uses,
    a no-op with nobody subscribed (no connected browser)."""

    def test_call_started_is_a_no_op_with_no_subscribers(self, db):
        db.messages.publish_call_started(
            run_id="run1",
            parent_run_id=None,
            parent_tool_call_id=None,
            agent_name="chat",
            prompt_type="idle",
            model="test-model",
            run_target=None,
        )  # must not raise

    def test_call_started_notifies_every_subscriber_with_the_given_identity(self, db):
        first: list = []
        second: list = []
        db.messages.subscribe_call_started(first.append)
        db.messages.subscribe_call_started(second.append)

        db.messages.publish_call_started(
            run_id="run1",
            parent_run_id="parent1",
            parent_tool_call_id="tool-call-1",
            agent_name="chat",
            prompt_type="idle",
            model="test-model",
            run_target="some-collection",
        )

        for events in (first, second):
            assert len(events) == 1
            event = events[0]
            assert event.run_id == "run1"
            assert event.parent_run_id == "parent1"
            assert event.parent_tool_call_id == "tool-call-1"
            assert event.agent_name == "chat"
            assert event.prompt_type == "idle"
            assert event.model == "test-model"
            assert event.run_target == "some-collection"
            assert event.started_at is not None

    def test_tool_started_and_tool_finished_carry_the_dispatched_identity(self, db):
        started: list = []
        finished: list = []
        db.messages.subscribe_tool_started(started.append)
        db.messages.subscribe_tool_finished(finished.append)

        db.messages.publish_tool_started(
            run_id="run1", tool_call_id="call1", tool_name="search", status="Searching…"
        )
        db.messages.publish_tool_finished(
            run_id="run1",
            tool_call_id="call1",
            tool_name="search",
            status="Searching…",
            success=False,
        )

        assert len(started) == len(finished) == 1
        assert started[0].run_id == finished[0].run_id == "run1"
        assert started[0].tool_call_id == finished[0].tool_call_id == "call1"
        assert finished[0].success is False

    def test_a_raising_subscriber_never_blocks_the_rest(self, db):
        """One subscriber raising must not stop the others — the same guarantee
        every other publish here gives (#2441)."""
        received: list = []

        def bad_listener(event):
            raise RuntimeError("boom")

        db.messages.subscribe_call_started(bad_listener)
        db.messages.subscribe_call_started(received.append)

        db.messages.publish_call_started(
            run_id="run1",
            parent_run_id=None,
            parent_tool_call_id=None,
            agent_name=None,
            prompt_type=None,
            model="test-model",
            run_target=None,
        )

        assert len(received) == 1


# ── The exchange pairing and the related-exchanges reads (#3010) ─────────────


class TestExchangeSides:
    """The exchange a reply row closes, reconstructed off the run stamps —
    the one pairing the write-time stamp, the backfill and the render share."""

    USER = "+15551234567"

    def _seed_turn(self, db: Database) -> int:
        """One exchange with everything the stamps can carry: the triggering
        ask, an interim send, a message that JOINED the run mid-turn, and the
        closing reply — all under one run id."""
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            self.USER,
            "should each run read the log first?",
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "reading the log now.",
            run_id="run-sides",
            recipient=self.USER,
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            self.USER,
            "also the bloom notes",
            run_id="run-sides",
        )
        reply_id = db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "done — each run reads the log first, then writes the summary.",
            parent_id=ask_id,
            run_id="run-sides",
            recipient=self.USER,
        )
        assert reply_id is not None, "the closing reply failed to log"
        return reply_id

    def test_sides_read_off_the_run_stamps(self, db):
        reply_id = self._seed_turn(db)
        reply = db.messages.get_by_id(reply_id)
        assert reply is not None

        sides = db.messages.exchange_sides_for(reply)

        assert [row.content for row in sides.user_rows] == [
            "should each run read the log first?",
            "also the bloom notes",
        ]
        assert [row.content for row in sides.reply_rows] == [
            "reading the log now.",
            "done — each run reads the log first, then writes the summary.",
        ]
        assert sides.joined_text() == (
            "should each run read the log first?\nalso the bloom notes\n"
            "reading the log now.\ndone — each run reads the log first, then writes the summary."
        )

    def test_sides_without_a_run_degrade_to_the_single_row(self, db):
        """A pre-#2630 reply row carries no run id: its own text is the reply
        side, its ``parent_id`` row the user side — never a guessed neighbour."""
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, self.USER, "the ask"
        )
        reply_id = db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "the reply",
            parent_id=ask_id,
            recipient=self.USER,
        )
        reply = db.messages.get_by_id(reply_id)
        assert reply is not None

        sides = db.messages.exchange_sides_for(reply)

        assert [row.id for row in sides.user_rows] == [ask_id]
        assert [row.id for row in sides.reply_rows] == [reply_id]

    def test_only_the_closing_reply_is_an_exchange_vector_candidate(self, db):
        """The vector belongs on the row that CLOSES the exchange — the reply
        with both the run id and the parent.  An interim send carries the run
        id but no parent (there was no id to link against yet), so the
        backfill's selector passes it over: one exchange, one vector."""
        self._seed_turn(db)

        pending = db.messages.replies_without_exchange_embeddings(limit=100)

        assert [row.content for row in pending] == [
            "done — each run reads the log first, then writes the summary."
        ]


class TestExchangeWindowBoundary:
    """The window boundary the related-exchanges retrieval reads — anchored on
    the same user-turn count the history's grow/cut cycle uses."""

    USER = "+15551234567"

    def _seed_exchange(self, db: Database, content_user: str, content_reply: str, at: datetime):
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, self.USER, content_user, timestamp=at
        )
        reply_id = db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            content_reply,
            parent_id=ask_id,
            timestamp=at + timedelta(minutes=1),
            recipient=self.USER,
        )
        return ask_id, reply_id

    def test_user_turn_at_offset_walks_newest_first(self, db):
        old = datetime(2026, 1, 1, 12, 0)
        new = datetime(2026, 1, 3, 9, 0)
        self._seed_exchange(db, "old ask", "old reply", old)
        newest_ask_id, _ = self._seed_exchange(db, "new ask", "new reply", new)

        newest = db.messages.user_turn_at_offset(self.USER, 0)
        older = db.messages.user_turn_at_offset(self.USER, 1)
        assert newest is not None and newest.id == newest_ask_id
        assert older is not None and older.content == "old ask"
        assert db.messages.user_turn_at_offset(self.USER, 5) is None

    def test_exchange_vectors_before_stops_at_the_boundary(self, db):
        first = datetime(2026, 1, 1, 12, 0)
        second = datetime(2026, 1, 2, 12, 0)
        third = datetime(2026, 1, 3, 12, 0)
        _, first_reply_id = self._seed_exchange(db, "first ask", "first reply", first)
        _, second_reply_id = self._seed_exchange(db, "second ask", "second reply", second)
        self._seed_exchange(db, "third ask", "third reply", third)
        db.messages.set_exchange_embedding(first_reply_id, b"vec-1")
        db.messages.set_exchange_embedding(second_reply_id, b"vec-2")

        # The boundary is the second exchange's own ask (what a window keeping
        # the newest two user turns would anchor on): the first exchange's
        # reply precedes it, the second's own reply follows it — so only the
        # first exchange's vector comes back.
        boundary = db.messages.user_turn_at_offset(self.USER, 1)
        assert boundary is not None

        candidates = db.messages.exchange_vectors_before(boundary)

        assert [row.id for row in candidates] == [first_reply_id]

    def test_exchange_vectors_before_is_empty_inside_the_window(self, db):
        only = datetime(2026, 1, 1, 12, 0)
        _, only_reply_id = self._seed_exchange(db, "only ask", "only reply", only)
        db.messages.set_exchange_embedding(only_reply_id, b"vec")
        boundary = db.messages.user_turn_at_offset(self.USER, 0)
        assert boundary is not None
        assert db.messages.exchange_vectors_before(boundary) == []

    def test_session_started_at_is_the_sessions_oldest_closed_turn_with_no_boundary_yet(self, db):
        """#3125: with no boundary ever recorded, ``session_anchor`` falls
        back to "everything stored is one session" — ``session_started_at``
        is the session's oldest CLOSED turn, over the identical counts
        ``session_turns`` is read from, so it can never disagree with it."""
        first = datetime(2026, 1, 1, 12, 0)
        second = datetime(2026, 1, 1, 12, 5)
        self._seed_exchange(db, "first ask", "first reply", first)
        self._seed_exchange(db, "second ask", "second reply", second)

        anchor = db.messages.session_anchor(self.USER)

        assert anchor.session_turns == 2
        assert anchor.live_turn_open is False
        assert anchor.session_started_at == first

    def test_session_started_at_excludes_the_open_live_turn(self, db):
        """The open live turn (no reply yet) is dropped from BOTH the counts
        and ``session_started_at`` — asking for it must never read it back as
        the session's own start."""
        first = datetime(2026, 1, 1, 12, 0)
        self._seed_exchange(db, "first ask", "first reply", first)
        live_at = datetime(2026, 1, 1, 12, 5)
        db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, self.USER, "live ask", timestamp=live_at
        )

        anchor = db.messages.session_anchor(self.USER)

        assert anchor.live_turn_open is True
        assert anchor.session_started_at == first

    def test_session_started_at_is_none_with_no_turns_at_all(self, db):
        """No incoming row at all → no session to anchor on — ``None``, never
        a guessed instant."""
        anchor = db.messages.session_anchor(self.USER)

        assert anchor.session_started_at is None

    def test_session_started_at_is_the_live_turn_when_it_opens_the_session(self, tmp_path):
        """``session_turns == 0`` means the live turn IS the session's first
        one (the docstring's own named case) — ``session_started_at`` falls
        back to that live turn's own timestamp, the only instant available:
        nothing from DURING this turn can have landed in the store yet.
        Reached here via an explicit idle rotation (#3125) — ``session_anchor``
        itself never rotates, so without that write the stale exchange would
        simply count toward the session, gap or no gap."""
        stale = datetime(2026, 1, 1, 12, 0)
        live_at = stale + timedelta(seconds=3600)  # past the 1800s idle gap
        db = migrated_db(str(tmp_path / "test.db"), clock=_FixedClock(live_at))
        self._seed_exchange(db, "stale ask", "stale reply", stale)
        db.messages.record_idle_rotation(self.USER, 1800)
        db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, self.USER, "live ask", timestamp=live_at
        )

        anchor = db.messages.session_anchor(self.USER)

        assert anchor.session_turns == 0
        assert anchor.live_turn_open is True
        assert anchor.session_started_at == live_at

    def test_record_idle_rotation_is_a_noop_before_the_idle_gap_clears(self, tmp_path):
        """Still well inside the idle gap: no boundary gets written, and
        ``session_anchor`` keeps counting the closed turns as one session."""
        first = datetime(2026, 1, 1, 12, 0)
        second = datetime(2026, 1, 1, 12, 5)
        # The second reply lands one minute after its own ask (``_seed_exchange``);
        # 30s after that is still well inside the 1800s idle gap.
        now = second + timedelta(minutes=1, seconds=30)
        db = migrated_db(str(tmp_path / "test.db"), clock=_FixedClock(now))
        self._seed_exchange(db, "first ask", "first reply", first)
        self._seed_exchange(db, "second ask", "second reply", second)

        rotated = db.messages.record_idle_rotation(self.USER, 1800)

        assert rotated is False
        anchor = db.messages.session_anchor(self.USER)
        assert anchor.session_turns == 2
        assert anchor.total_user_turns == 2
        assert anchor.session_started_at == first

    def test_record_idle_rotation_writes_once_the_gap_has_passed(self, tmp_path):
        """Once the gap since the newest stored message clears
        ``idle_seconds``, ``record_idle_rotation`` writes a boundary at
        ``now`` — the SAME write a real turn's own inline call and the idle
        scheduler task share — and the next ``session_anchor`` read comes
        back rotated (``session_turns == 0``)."""
        stale = datetime(2026, 1, 1, 12, 0)
        now = stale + timedelta(seconds=3600)  # past the 1800s idle gap
        db = migrated_db(str(tmp_path / "test.db"), clock=_FixedClock(now))
        self._seed_exchange(db, "stale ask", "stale reply", stale)

        rotated = db.messages.record_idle_rotation(self.USER, 1800)

        assert rotated is True
        anchor = db.messages.session_anchor(self.USER)
        assert anchor.session_turns == 0
        assert anchor.total_user_turns == 1
        assert anchor.session_started_at == now

    def test_record_idle_rotation_never_fires_twice_for_the_same_gap(self, tmp_path):
        """Calling it again with nothing new stored is a no-op (#3125) — the
        SAME rule both the idle scheduler task and a real turn's own inline
        call rely on never double-rotating."""
        stale = datetime(2026, 1, 1, 12, 0)
        now = stale + timedelta(seconds=3600)
        clock = _FixedClock(now)
        db = migrated_db(str(tmp_path / "test.db"), clock=clock)
        self._seed_exchange(db, "stale ask", "stale reply", stale)
        assert db.messages.record_idle_rotation(self.USER, 1800) is True

        clock._when = now + timedelta(seconds=10)
        rotated_again = db.messages.record_idle_rotation(self.USER, 1800)

        assert rotated_again is False
        assert db.messages.session_anchor(self.USER).session_started_at == now

    def test_record_idle_rotation_with_no_turns_at_all_is_a_noop(self, tmp_path):
        """Nothing stored yet — nothing to measure idleness from."""
        db = migrated_db(str(tmp_path / "test.db"), clock=_FixedClock(datetime(2026, 1, 1, 12, 0)))

        assert db.messages.record_idle_rotation(self.USER, 1800) is False
        assert db.messages.session_anchor(self.USER).session_started_at is None

    def test_record_cap_rotation_fires_at_the_turn_that_passes_the_cap(self, tmp_path):
        """#3125: ``CHAT_SESSION_MAX_TURNS`` stays a stored rotation too — a
        session whose history has grown to fill the cap (``window=2``,
        ``max_turns=4`` ⇒ threshold ``session_turns > 2``) rotates at THAT
        turn, resetting ``session_turns`` to 0 the same way an idle rotation
        does."""
        start = datetime(2026, 1, 1, 12, 0)
        now = start + timedelta(minutes=10)
        db = migrated_db(str(tmp_path / "test.db"), clock=_FixedClock(now))
        for n in range(4):
            self._seed_exchange(db, f"ask {n}", f"reply {n}", start + timedelta(minutes=n))

        # Before the cap: with no boundary at all, session_turns == total == 4.
        assert db.messages.session_anchor(self.USER).session_turns == 4

        rotated = db.messages.record_cap_rotation(self.USER, window=2, max_turns=4)

        assert rotated is True
        anchor = db.messages.session_anchor(self.USER)
        assert anchor.session_turns == 0
        assert anchor.session_started_at == now

    def test_record_cap_rotation_is_a_noop_below_the_cap(self, db):
        start = datetime(2026, 1, 1, 12, 0)
        self._seed_exchange(db, "ask", "reply", start)

        assert db.messages.record_cap_rotation(self.USER, window=2, max_turns=4) is False
