"""Tests for SendQueueDrainer — cooldown-honouring delivery of queued messages.

``send_message`` enqueues; this drainer delivers.  The cooldown logic is the
same gate ``send_message`` used to apply inline, just relocated so a cooldown
*delays* a message instead of dropping it:

- No prior Penny message (or the user has spoken since) → deliver immediately.
- A recent Penny message with no user reply since → hold until the flat
  ``SEND_COOLDOWN_SECONDS`` window elapses.

Delivery pops the oldest pending row (FIFO), one per tick, marks it sent, and
attributes it to the collection that queued it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from penny.constants import ChannelType, PennyConstants
from penny.database import Database
from penny.database.memory import Inclusion, RecallMode
from penny.scheduler.send_queue_drainer import SendQueueDrainer

_PENNY_LOG = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_USER_LOG = PennyConstants.MEMORY_USER_MESSAGES_LOG

_RECIPIENT = "+15551234567"
_COLLECTION = "notified-thoughts"


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    # Marker rows so db.memory(...) dispatches the messagelog facades the
    # cooldown helpers read.
    db.memories.create_log(_PENNY_LOG, "outbound", Inclusion.NEVER, RecallMode.RECENT)
    db.memories.create_log(_USER_LOG, "inbound", Inclusion.NEVER, RecallMode.RECENT)
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )
    db.devices.register(ChannelType.SIGNAL, _RECIPIENT, "Signal", is_default=True)
    return db


def _make_config(cooldown_seconds: float = 600.0):
    runtime = type("Runtime", (), {"SEND_COOLDOWN_SECONDS": cooldown_seconds})()
    return type("Config", (), {"runtime": runtime})()


def _make_channel():
    channel = type("Channel", (), {})()
    channel.send_response = AsyncMock(return_value=42)
    return channel


def _make_drainer(db, channel, cooldown_seconds: float = 600.0) -> SendQueueDrainer:
    drainer = SendQueueDrainer(db=db, config=_make_config(cooldown_seconds))
    drainer.set_channel(channel)
    return drainer


def _penny_sent(db, content: str) -> None:
    db.messages.log_message(PennyConstants.MessageDirection.OUTGOING, "penny", content)


def _user_said(db, content: str) -> None:
    db.messages.log_message(PennyConstants.MessageDirection.INCOMING, _RECIPIENT, content)


@pytest.mark.asyncio
async def test_drain_delivers_when_no_prior_send(tmp_path):
    """Empty conversation → cooldown vacuously elapsed → deliver + mark sent."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    drainer = _make_drainer(db, channel)
    db.send_queue.enqueue(content="hey there!", collection=_COLLECTION)

    did_work = await drainer.execute()

    assert did_work is True
    channel.send_response.assert_awaited_once()
    kwargs = channel.send_response.await_args.kwargs
    assert kwargs["recipient"] == _RECIPIENT
    assert kwargs["content"] == "hey there!"
    assert kwargs["author"] == _COLLECTION
    # Row is stamped delivered — never re-sent.
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_drain_holds_when_cooldown_not_elapsed(tmp_path):
    """Recent Penny send, no user reply since → hold the queued message."""
    db = _make_db(tmp_path)
    _penny_sent(db, "prior")  # count = 1, no user reply since
    channel = _make_channel()
    drainer = _make_drainer(db, channel, cooldown_seconds=3600.0)
    db.send_queue.enqueue(content="hey again!", collection=_COLLECTION)

    did_work = await drainer.execute()

    assert did_work is False
    channel.send_response.assert_not_awaited()
    # Still pending — delayed, not dropped.
    pending = db.send_queue.next_pending()
    assert pending is not None and pending.content == "hey again!"


@pytest.mark.asyncio
async def test_drain_delivers_when_user_replied_since_last_send(tmp_path):
    """User spoke since Penny's last send → conversational → deliver now."""
    db = _make_db(tmp_path)
    _penny_sent(db, "prior")
    _user_said(db, "actually, follow-up")
    channel = _make_channel()
    drainer = _make_drainer(db, channel, cooldown_seconds=3600.0)
    db.send_queue.enqueue(content="responding", collection=_COLLECTION)

    did_work = await drainer.execute()

    assert did_work is True
    channel.send_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_drain_no_work_when_queue_empty(tmp_path):
    """Nothing queued → no work, no send."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    drainer = _make_drainer(db, channel)

    did_work = await drainer.execute()

    assert did_work is False
    channel.send_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_drain_delivers_one_per_tick_in_fifo_order(tmp_path):
    """Two queued → a single execute() delivers only the oldest; the rest waits."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    drainer = _make_drainer(db, channel)
    db.send_queue.enqueue(content="first", collection=_COLLECTION)
    db.send_queue.enqueue(content="second", collection=_COLLECTION)

    did_work = await drainer.execute()

    assert did_work is True
    channel.send_response.assert_awaited_once()
    assert channel.send_response.await_args.kwargs["content"] == "first"
    # Only the oldest is delivered this tick; the next stays pending.
    pending = db.send_queue.next_pending()
    assert pending is not None and pending.content == "second"


@pytest.mark.asyncio
async def test_drain_no_channel_is_noop(tmp_path):
    """No channel wired → drainer reports no work rather than crashing."""
    db = _make_db(tmp_path)
    drainer = SendQueueDrainer(db=db, config=_make_config())
    db.send_queue.enqueue(content="queued", collection=_COLLECTION)

    assert await drainer.execute() is False
    # Message remains pending for when a channel is available.
    assert db.send_queue.next_pending() is not None
