"""Tests for SendQueueDrainer — cooldown-honouring delivery of queued messages.

``send_message`` enqueues; this drainer delivers.  The cooldown logic is the
same gate ``send_message`` used to apply inline, just relocated so a cooldown
*delays* a message instead of dropping it:

- No prior Penny message (or the user has spoken since) → deliver immediately.
- A recent Penny message with no user reply since → hold until the flat
  ``SEND_COOLDOWN_SECONDS`` window elapses.

Delivery pops the oldest pending row (FIFO), one per tick, marks it sent, and
attributes it to the collection that queued it.

There are two delivery lanes (#1939), read off the row's own ``origin``: a cadence
cycle's message waits out the cooldown above, and a message from a cycle the user
pressed "run this now" on goes out on the next tick — they are sitting there waiting
for it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse
from penny.constants import (
    ChannelType,
    CycleTrigger,
    MutationAction,
    PennyConstants,
    QueueItemKind,
)
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import SendQueueItem
from penny.database.mutation_store import MutationDetail, render_mutation
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmResponse
from penny.scheduler.send_queue_drainer import SendQueueDrainer
from penny.tests.conftest import require_memory, wait_until
from penny.tests.schema_template import schema_only_db

_PENNY_LOG = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_USER_LOG = PennyConstants.MEMORY_USER_MESSAGES_LOG

_RECIPIENT = "+15551234567"  # the user's primary sender identity
# Penny's OWN Signal number (config.signal_number) — #2765's startup fix makes
# sure this is NEVER registered as a device.  Kept here only so tests can
# assert the drainer never resolves to it.
_PENNY_NUMBER = "+15559998888"
_IOS_IDENTIFIER = "ios-keychain-id"
_DISCORD_CHANNEL_ID = "9876543210"
_COLLECTION = "notified-thoughts"


def _make_db(tmp_path) -> Database:
    db = schema_only_db(str(tmp_path / "test.db"))
    # Marker rows so db.memory(...) dispatches the messagelog facades the
    # cooldown helpers read.
    db.memories.create_log(_PENNY_LOG, "outbound")
    db.memories.create_log(_USER_LOG, "inbound")
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="New York",
        timezone="America/New_York",
        date_of_birth="1990-01-01",
    )
    # #2765: production now seeds the Signal default device as the USER's own
    # number — never Penny's own (config.signal_number, which is never
    # registered at all) — so the drainer's default-device read lands on the
    # user directly, with no special-casing.
    db.devices.register(ChannelType.SIGNAL, _RECIPIENT, "Signal", is_default=True)
    return db


def _make_config(cooldown_seconds: float = 600.0):
    runtime = type("Runtime", (), {"SEND_COOLDOWN_SECONDS": cooldown_seconds})()
    return type("Config", (), {"runtime": runtime})()


def _make_channel():
    return SimpleNamespace(send_response=AsyncMock(return_value=42))


class _IdleCollector:
    """A collector double exposing the one read the drainer makes off it (#3025):
    no job run in progress, always.  The tests that need a run ACTUALLY in
    progress drive a real ``Collector`` (below) — this only stands in for the
    ordinary-row tests, whose subject is the cooldown lanes, not the run gate."""

    def is_cycle_running(self) -> bool:
        return False


def _make_drainer(db, channel, cooldown_seconds: float = 600.0, *, collector=None):
    # The chat agent only matters to the compose path (#2656), which these
    # ordinary-row tests never reach — a mock stands in for it.
    drainer = SendQueueDrainer(
        db=db,
        config=_make_config(cooldown_seconds),
        chat_agent=MagicMock(),
        collector=collector if collector is not None else _IdleCollector(),
    )
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
    # Routes to the seeded Signal default device's identifier — the user's own
    # number (#2765) — never Penny's own, which is never registered at all.
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
async def test_a_user_triggered_runs_message_skips_the_cooldown_a_cadence_one_waits(tmp_path):
    """The THREE lanes, in one world (#1939, #2964).

    The cooldown is anti-spam for messages nobody asked for.  A message from a cycle
    the user pressed "run this now" on, or a sandbox preview the user asked to run,
    is one they are sitting in front of waiting for, so both go out before the next
    tick — while the cadence message queued alongside them, older and otherwise
    first in line, keeps waiting.  Observed live: an on-demand run finished and its
    result then waited ~10.5 minutes because an unrelated chat reply seconds earlier
    had started the cooldown.
    """
    db = _make_db(tmp_path)
    _penny_sent(db, "prior")  # the cooldown is running: no user reply since
    channel = _make_channel()
    drainer = _make_drainer(db, channel, cooldown_seconds=3600.0)
    # The cadence message is OLDEST, so strict FIFO would hand it out first — asking
    # by lane is what stops it holding the other two's own results behind it.
    db.send_queue.enqueue(content="the daily digest", collection=_COLLECTION)
    db.send_queue.enqueue(
        content="your run finished",
        collection=_COLLECTION,
        origin=CycleTrigger.ON_DEMAND,
    )
    db.send_queue.enqueue(
        content="your preview finished",
        collection=_COLLECTION,
        origin=CycleTrigger.SANDBOX,
    )

    assert await drainer.execute() is True
    assert channel.send_response.await_args.kwargs["content"] == "your run finished"

    # The SANDBOX lane is tried next — also skips the cooldown, also ahead of the
    # still-older cadence message.
    channel.send_response.reset_mock()
    assert await drainer.execute() is True
    assert channel.send_response.await_args.kwargs["content"] == "your preview finished"

    # The cadence message is delayed, not dropped — and the next tick still holds it.
    channel.send_response.reset_mock()
    assert await drainer.execute() is False
    channel.send_response.assert_not_awaited()
    pending = db.send_queue.next_pending()
    assert pending is not None and pending.content == "the daily digest"


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
async def test_drainer_resolves_signal_default_to_the_users_number(tmp_path):
    """Signal's default device is the user's own number (#2765) — never
    Penny's own seeded endpoint, which is never registered at all."""
    db = _make_db(tmp_path)
    drainer = _make_drainer(db, _make_channel())

    recipient = drainer._resolve_recipient()

    assert recipient == _RECIPIENT
    assert recipient != _PENNY_NUMBER


@pytest.mark.asyncio
async def test_drainer_resolves_discord_default_to_the_channel_id(tmp_path):
    """Discord's default device identifier is the channel id — the only
    endpoint Discord's own ``_send_raw`` ever delivers to (it ignores its
    ``recipient`` argument and always posts to the configured channel), so
    reading the default device directly is correct there too, not merely
    tolerated: the primary sender it used to fall back to is a Discord USER
    id, which no device is ever registered under."""
    db = _make_db(tmp_path)
    db.devices.register(ChannelType.DISCORD, _DISCORD_CHANNEL_ID, "Discord", is_default=True)
    drainer = _make_drainer(db, _make_channel())

    recipient = drainer._resolve_recipient()

    assert recipient == _DISCORD_CHANNEL_ID
    assert recipient != _PENNY_NUMBER


@pytest.mark.asyncio
async def test_drainer_falls_back_to_primary_sender_with_no_default_device(tmp_path):
    """No device registered at all yet (a fresh Signal install before its
    first message ever arrives) — the drainer still has a recipient to try:
    the primary sender, the one case the simplified read can't cover."""
    db = schema_only_db(str(tmp_path / "test.db"))
    db.memories.create_log(_PENNY_LOG, "outbound")
    db.memories.create_log(_USER_LOG, "inbound")
    db.users.save_info(
        sender=_RECIPIENT,
        name="user",
        location="New York",
        timezone="America/New_York",
        date_of_birth="1990-01-01",
    )
    drainer = _make_drainer(db, _make_channel())

    assert drainer._resolve_recipient() == _RECIPIENT


@pytest.mark.asyncio
async def test_drain_no_channel_is_noop(tmp_path):
    """No channel wired → drainer reports no work rather than crashing."""
    db = _make_db(tmp_path)
    drainer = SendQueueDrainer(
        db=db, config=_make_config(), chat_agent=MagicMock(), collector=_IdleCollector()
    )
    db.send_queue.enqueue(content="queued", collection=_COLLECTION)

    assert await drainer.execute() is False
    # Message remains pending for when a channel is available.
    assert db.send_queue.next_pending() is not None


@pytest.mark.asyncio
async def test_deliver_composed_sends_without_queuing(tmp_path):
    """``deliver_composed`` (#2913) is the no-queue, no-idle-wait path a caller
    that already ran its own delivery turn uses — the startup changelog
    announcement. It records a send exactly like a drained row does: the same
    recipient resolution and the same messagelog stamp (mechanism, run id)."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    drainer = _make_drainer(db, channel)

    delivered = await drainer.deliver_composed(
        "the announcement", collection=_COLLECTION, run_id="run-123"
    )

    assert delivered is True
    channel.send_response.assert_awaited_once()
    kwargs = channel.send_response.await_args.kwargs
    assert kwargs["content"] == "the announcement"
    assert kwargs["recipient"] == _RECIPIENT
    assert kwargs["mechanism"] == _COLLECTION
    assert kwargs["run_id"] == "run-123"
    # Nothing was ever written to the queue.
    assert db.send_queue.pending_items() == []


@pytest.mark.asyncio
async def test_deliver_composed_no_channel_or_recipient_reports_false(tmp_path):
    """No channel wired → ``deliver_composed`` reports failure rather than
    crashing, the same honesty ``execute()`` gives a queued row."""
    db = _make_db(tmp_path)
    drainer = SendQueueDrainer(
        db=db, config=_make_config(), chat_agent=MagicMock(), collector=_IdleCollector()
    )

    delivered = await drainer.deliver_composed(
        "the announcement", collection=_COLLECTION, run_id="run-123"
    )

    assert delivered is False


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
    # cancellation renders on (self-state activity, memory_get).
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


# ── The run gate (#3025): a job run in progress holds queued reports back ──

_REPORT_COLLECTION = "indie-release-watch"


class _GatedLlm(LlmClient):
    """A model client that holds its first call open until the test releases it —
    the deterministic stand-in for a job run mid-cycle, no timers.  The test
    awaits the collector's own ``is_cycle_running()`` to see the run start, then
    sets ``release``; the canned text response ends the cycle, which settles and
    frees the cycle lock."""

    def __init__(self) -> None:
        super().__init__("http://localhost:9", "test-model", max_retries=1, retry_delay=0.0)
        self.release = asyncio.Event()

    async def chat(
        self,
        messages: list[LlmMessage],
        tools: list[dict] | None = None,
        format: dict | str | None = None,
        *,
        call_link: CallLink,
        agent_name: str | None = None,
        prompt_type: str | None = None,
        run_target: str | None = None,
        token_budget: int | None = None,
        thinking_enabled: bool = True,
    ) -> LlmResponse:
        await self.release.wait()
        return LlmResponse(
            message=LlmMessage(role="assistant", content="All caught up."),
            model=self.model,
        )


def _make_real_collector(db: Database, test_config, model_client) -> Collector:
    """A real Collector wired to the given model client — the genuine run path
    (``run_for`` → ``_execute_cycle`` → ``_run_locked_cycle`` under the cycle
    lock) is what holds the gate closed in the tests below.  ``run_for`` is the
    user-requested path that lives OUTSIDE the scheduler — the case the cooldown
    never kept apart from a queued report's delivery turn; the scheduled and
    sandbox paths hold the same lock, so the gate reads them the same way."""
    return Collector(
        model_client=model_client,
        db=db,
        config=test_config,
        embedding_model_client=LlmClient(
            "http://localhost:9", "test-model", max_retries=1, retry_delay=0.0
        ),
        process_started_at=datetime(2000, 1, 1, tzinfo=UTC),
    )


def _seed_report_row(db: Database, run_id: str) -> None:
    """A buildable composing report row: the collection plus the entry its run
    wrote, so once the gate opens the compose path can actually assemble the
    fact set instead of failing the row."""
    db.memories.create_collection(
        _REPORT_COLLECTION,
        "New indie releases",
        extraction_prompt="Check the store page for new releases.",
        notify=True,
    )
    require_memory(db, _REPORT_COLLECTION).write_rule_decided(
        [
            EntryInput(
                key="Astral Drift",
                content="Astral Drift — a new indie metroidvania, announced for spring.",
            )
        ],
        author="producer",
        run_id=run_id,
    )
    db.send_queue.enqueue_report(
        _REPORT_COLLECTION, CycleTrigger.ON_DEMAND, run_id, kind=QueueItemKind.REPORT
    )


@pytest.mark.asyncio
async def test_a_queued_report_is_not_delivered_while_a_job_run_is_in_progress(
    tmp_path, test_config
):
    """#3025: with a job run mid-cycle, the drainer holds its whole tick back —
    the queued report stays pending, nothing is composed, the channel hears
    nothing.  The run here is a real user-requested one (``run_for``), the path
    outside the scheduler the old idle/cooldown gates never saw."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    model = _GatedLlm()
    collector = _make_real_collector(db, test_config, model)
    drainer = _make_drainer(db, channel, collector=collector)
    _seed_report_row(db, "run-report-1")

    cycle = asyncio.create_task(collector.run_for(_REPORT_COLLECTION))
    await wait_until(collector.is_cycle_running)

    assert await drainer.execute() is False
    channel.send_response.assert_not_awaited()
    # Delayed, never dropped: the row is exactly as it was.
    assert db.send_queue.next_pending() is not None

    model.release.set()
    await cycle


@pytest.mark.asyncio
async def test_a_queued_report_is_delivered_once_the_cycle_ends(tmp_path, test_config):
    """#3025: the same report, held back while the cycle ran, goes out on the
    first tick with no run in progress."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    model = _GatedLlm()
    collector = _make_real_collector(db, test_config, model)
    # The compose's delivery turn — a canned reply is all this test needs from it;
    # the turn itself is test_report_delivery.py's subject.
    chat = MagicMock()
    chat.handle_job_report = AsyncMock(
        return_value=ControllerResponse(answer="Astral Drift was announced for spring!")
    )
    drainer = SendQueueDrainer(db=db, config=_make_config(), chat_agent=chat, collector=collector)
    drainer.set_channel(channel)
    _seed_report_row(db, "run-report-1")

    cycle = asyncio.create_task(collector.run_for(_REPORT_COLLECTION))
    await wait_until(collector.is_cycle_running)
    assert await drainer.execute() is False
    model.release.set()
    await cycle
    await wait_until(lambda: not collector.is_cycle_running())

    assert await drainer.execute() is True
    channel.send_response.assert_awaited_once()
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_delivery_is_unchanged_when_no_job_run_is_in_progress(tmp_path, test_config):
    """#3025: with a real collector sitting idle (no run in progress), delivery
    behaves exactly as before — the queued message goes out this tick."""
    db = _make_db(tmp_path)
    channel = _make_channel()
    collector = _make_real_collector(
        db,
        test_config,
        LlmClient("http://localhost:9", "test-model", max_retries=1, retry_delay=0.0),
    )
    drainer = _make_drainer(db, channel, collector=collector)
    db.send_queue.enqueue(content="all quiet on the watch", collection=_COLLECTION)

    assert await drainer.execute() is True
    channel.send_response.assert_awaited_once()
    assert db.send_queue.next_pending() is None
