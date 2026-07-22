"""Tests for MessageStore conversation queries."""

from datetime import UTC, datetime, timedelta

from penny.constants import PennyConstants
from penny.database import Database


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
        from sqlmodel import Session

        from penny.database.models import MessageLog

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
