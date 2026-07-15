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
from sqlmodel import Session, select

from penny.constants import ChannelType, MutationAction, PennyConstants
from penny.database import Database
from penny.database.models import SendQueueItem
from penny.database.mutation_store import MutationDetail, render_mutation
from penny.scheduler.send_queue_drainer import SendQueueDrainer

_PENNY_LOG = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_USER_LOG = PennyConstants.MEMORY_USER_MESSAGES_LOG

_RECIPIENT = "+15551234567"  # the user's primary sender identity
# Penny's OWN registered Signal number — what production seeds as the default
# device on a fresh install (config.signal_number).  Deliberately different from
# the user's number so a drainer that routes to the default device identifier
# would misroute to Note-to-Self rather than the user.
_PENNY_NUMBER = "+15559998888"
_IOS_IDENTIFIER = "ios-keychain-id"
_COLLECTION = "notified-thoughts"


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    # Marker rows so db.memory(...) dispatches the messagelog facades the
    # cooldown helpers read.
    db.memories.create_log(_PENNY_LOG, "outbound")
    db.memories.create_log(_USER_LOG, "inbound")
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="Toronto",
        timezone="America/Toronto",
        date_of_birth="1990-01-01",
    )
    # Production seeding: the seeded Signal default device is Penny's own number,
    # NOT the user's — the drainer must still route to the primary sender.
    db.devices.register(ChannelType.SIGNAL, _PENNY_NUMBER, "Signal", is_default=True)
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
    # Routes to the user's primary sender, NOT the seeded Signal default device
    # (Penny's own number) — otherwise the send would land in Note-to-Self.
    assert kwargs["recipient"] == _RECIPIENT
    assert kwargs["recipient"] != _PENNY_NUMBER
    assert kwargs["content"] == "hey there!"
    assert kwargs["author"] == _COLLECTION
    # Emission provenance (#1568): the queued row's collection is carried onto
    # send_response as the mechanism, so the delivered messagelog row names its
    # cause (NULL = direct reply).
    assert kwargs["mechanism"] == _COLLECTION
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
async def test_drain_routes_to_ios_default_device_identifier(tmp_path):
    """iOS default device → the device identifier *is* the recipient.

    On iOS the registered device identifier is the delivery target (not the
    single primary sender), so the drainer must route drained messages there.
    """
    db = _make_db(tmp_path)
    # An iOS device registers as default — registration keeps a single default,
    # so it supersedes the seeded Signal device.
    db.devices.register(ChannelType.IOS, _IOS_IDENTIFIER, "iPhone", is_default=True)
    channel = _make_channel()
    drainer = _make_drainer(db, channel)
    db.send_queue.enqueue(content="ping", collection=_COLLECTION)

    did_work = await drainer.execute()

    assert did_work is True
    kwargs = channel.send_response.await_args.kwargs
    assert kwargs["recipient"] == _IOS_IDENTIFIER


@pytest.mark.asyncio
async def test_drain_no_channel_is_noop(tmp_path):
    """No channel wired → drainer reports no work rather than crashing."""
    db = _make_db(tmp_path)
    drainer = SendQueueDrainer(db=db, config=_make_config())
    db.send_queue.enqueue(content="queued", collection=_COLLECTION)

    assert await drainer.execute() is False
    # Message remains pending for when a channel is available.
    assert db.send_queue.next_pending() is not None


@pytest.mark.asyncio
async def test_archiving_collection_cancels_pending_sends(tmp_path):
    """Teardown means silence through the queue (#1634): a send queued while the
    collector was alive is cancelled when the collection is archived, so the
    drainer delivers NOTHING — even though the cooldown is vacuously elapsed and
    would otherwise let it through.  The row is stamped cancelled (a visible audit
    trail; ``sent_at`` stays NULL — it was never sent), and the archive's mutation
    event names the cancelled count."""
    db = _make_db(tmp_path)
    db.memories.create_collection("hedgehog-watch", "hedgehog sightings", created_by_run_id="run-c")
    db.send_queue.enqueue(content="a hedgehog!", collection="hedgehog-watch")

    db.memories.archive("hedgehog-watch", run_id="run-arch")

    channel = _make_channel()
    drainer = _make_drainer(db, channel)  # cooldown vacuously elapsed (no prior send)
    did_work = await drainer.execute()

    # Nothing delivered — the only queued row was cancelled by the archive.
    assert did_work is False
    channel.send_response.assert_not_awaited()
    assert db.send_queue.next_pending() is None
    assert db.send_queue.pending_items() == []

    # Visible cancellation: the row is kept (audit trail), stamped cancelled_at,
    # and sent_at stays NULL — the single-source-of-truth "was it sent" is clean.
    with Session(db.engine) as session:
        rows = session.exec(select(SendQueueItem)).all()
    assert len(rows) == 1
    assert rows[0].cancelled_at is not None
    assert rows[0].sent_at is None

    # The archive's mutation event names the count — the legible surface the
    # cancellation renders on (self-state activity, memory_metadata).
    events = db.mutations.history("hedgehog-watch", limit=10)
    archive_event = next(e for e in events if e.action == MutationAction.ARCHIVED.value)
    assert archive_event.detail is not None
    assert (
        MutationDetail.model_validate_json(archive_event.detail).note == "cancelled 1 pending send"
    )
    assert render_mutation(archive_event).endswith("— cancelled 1 pending send")


@pytest.mark.asyncio
async def test_archive_cancel_is_scoped_and_pluralizes(tmp_path):
    """Cancellation is scoped to the archived collection's rows: a sibling
    collection's pending send survives and is what the drainer delivers, while
    both of the archived collection's rows are cancelled and the count pluralizes
    (#1634)."""
    db = _make_db(tmp_path)
    db.memories.create_collection("hedgehog-watch", "hedgehog sightings")
    db.memories.create_collection("weather-watch", "weather")
    db.send_queue.enqueue(content="a hedgehog!", collection="hedgehog-watch")
    db.send_queue.enqueue(content="another hedgehog!", collection="hedgehog-watch")
    db.send_queue.enqueue(content="rain today", collection="weather-watch")

    db.memories.archive("hedgehog-watch", run_id="run-arch")

    # The sibling's send is untouched — the drainer delivers it.
    channel = _make_channel()
    drainer = _make_drainer(db, channel)
    assert await drainer.execute() is True
    assert channel.send_response.await_args.kwargs["content"] == "rain today"

    # Both of the archived collection's rows are cancelled (kept, sent_at NULL).
    with Session(db.engine) as session:
        hedgehog = session.exec(
            select(SendQueueItem).where(SendQueueItem.collection == "hedgehog-watch")
        ).all()
    assert len(hedgehog) == 2
    assert all(row.cancelled_at is not None and row.sent_at is None for row in hedgehog)

    events = db.mutations.history("hedgehog-watch", limit=10)
    archive_event = next(e for e in events if e.action == MutationAction.ARCHIVED.value)
    assert archive_event.detail is not None
    assert (
        MutationDetail.model_validate_json(archive_event.detail).note == "cancelled 2 pending sends"
    )


@pytest.mark.asyncio
async def test_unarchive_does_not_resurrect_cancelled_sends(tmp_path):
    """A cancelled send belongs to the dead epoch — unarchiving the collection
    does NOT bring it back (#1634): the row stays cancelled and the drainer never
    delivers it."""
    db = _make_db(tmp_path)
    db.memories.create_collection("hedgehog-watch", "hedgehog sightings")
    db.send_queue.enqueue(content="a hedgehog!", collection="hedgehog-watch")
    db.memories.archive("hedgehog-watch")

    db.memories.unarchive("hedgehog-watch")

    channel = _make_channel()
    drainer = _make_drainer(db, channel)
    assert await drainer.execute() is False
    channel.send_response.assert_not_awaited()
    assert db.send_queue.next_pending() is None
    # The row is still cancelled — unarchive is not a resurrection.
    with Session(db.engine) as session:
        rows = session.exec(select(SendQueueItem)).all()
    assert len(rows) == 1 and rows[0].cancelled_at is not None
