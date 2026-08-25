"""Tests for MessageStore conversation queries."""

from datetime import UTC, datetime, timedelta

from sqlmodel import Session

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import MessageLog

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
