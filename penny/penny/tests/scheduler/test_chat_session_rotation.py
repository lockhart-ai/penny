"""Tests for ChatSessionRotationTask — the idle scheduler task that writes
the chat session's stored boundary (#3125).

``record_idle_rotation`` itself (the write rule both this task and a real
turn's own ``_dispatch_turn`` share) is pinned directly against
``MessageStore`` in ``tests/database/test_message_store.py``; these tests
pin the SCHEDULED TASK's own gates — never while a collector cycle runs,
never with no primary user, and that it defers to the write's own
idempotency rather than tracking anything of its own."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from penny.clock import Clock
from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.scheduler.chat_session_rotation import ChatSessionRotationTask
from penny.tests.conftest import TEST_SENDER
from penny.tests.schema_template import schema_only_db


class _FixedClock(Clock):
    """A controllable clock (#3125) — mutable, never advancing on its own,
    passed to ``schema_only_db``'s own ``clock=`` constructor parameter so
    ``db.messages._clock`` is this exact instance from construction."""

    def __init__(self, when: datetime) -> None:
        self._when = when

    def now(self) -> datetime:
        return self._when


class _FakeCycleGate:
    def __init__(self, running: bool = False) -> None:
        self.running = running

    def is_cycle_running(self) -> bool:
        return self.running


def _seed_user(db) -> None:
    db.users.save_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )


def _seed_closed_exchange(db, at: datetime) -> None:
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hi", timestamp=at
    )
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        "hey",
        parent_id=ask_id,
        timestamp=at + timedelta(seconds=5),
        recipient=TEST_SENDER,
    )


@pytest.mark.asyncio
async def test_fires_and_records_the_boundary_once_idle(tmp_path) -> None:
    stale = datetime(2026, 1, 1, 9, 0)
    now = stale + timedelta(seconds=900)
    clock = _FixedClock(now.replace(tzinfo=UTC))
    db = schema_only_db(str(tmp_path / "test.db"), clock=clock)
    _seed_user(db)
    _seed_closed_exchange(db, stale)
    task = ChatSessionRotationTask(
        db=db, cycle_gate=_FakeCycleGate(running=False), runtime=RuntimeParams(db)
    )

    fired = await task.execute()

    assert fired is True
    anchor = db.messages.session_anchor(TEST_SENDER)
    assert anchor.session_turns == 0
    assert anchor.session_started_at == now


@pytest.mark.asyncio
async def test_does_not_fire_before_the_idle_gap_clears(tmp_path) -> None:
    stale = datetime(2026, 1, 1, 9, 0)
    now = stale + timedelta(seconds=30)  # well inside the default 600s gap
    clock = _FixedClock(now.replace(tzinfo=UTC))
    db = schema_only_db(str(tmp_path / "test.db"), clock=clock)
    _seed_user(db)
    _seed_closed_exchange(db, stale)
    task = ChatSessionRotationTask(
        db=db, cycle_gate=_FakeCycleGate(running=False), runtime=RuntimeParams(db)
    )

    fired = await task.execute()

    assert fired is False


@pytest.mark.asyncio
async def test_never_fires_while_a_collector_cycle_is_running(tmp_path) -> None:
    stale = datetime(2026, 1, 1, 9, 0)
    now = stale + timedelta(seconds=900)
    clock = _FixedClock(now.replace(tzinfo=UTC))
    db = schema_only_db(str(tmp_path / "test.db"), clock=clock)
    _seed_user(db)
    _seed_closed_exchange(db, stale)
    task = ChatSessionRotationTask(
        db=db, cycle_gate=_FakeCycleGate(running=True), runtime=RuntimeParams(db)
    )

    fired = await task.execute()

    assert fired is False
    # No boundary was ever written — the no-row fallback still treats the
    # stale exchange as the (only) session's own start, never None.
    assert db.messages.session_anchor(TEST_SENDER).session_started_at == stale


@pytest.mark.asyncio
async def test_never_fires_with_no_primary_user_configured(tmp_path) -> None:
    db = schema_only_db(str(tmp_path / "test.db"))
    task = ChatSessionRotationTask(
        db=db, cycle_gate=_FakeCycleGate(running=False), runtime=RuntimeParams(db)
    )

    fired = await task.execute()

    assert fired is False
