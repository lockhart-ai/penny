"""Whole-render tests for the why_did_i_send_that tool (#1568).

The reverse index over emission provenance — from a delivered ``messagelog`` row's
``mechanism`` back to the registry row that sent it and the user message that
created it.  Model-facing, so every render shape is pinned char-for-char (the
review guide's whole-render assertion discipline): a direct reply, an autonomous
send with a spawning user message, an autonomous send from a seeded mechanism (no
source message), and the not-found failure.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlmodel import Session

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import MemoryRow, MessageLog
from penny.tools.memory_tools import WhyDidISendThatTool

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING


def _db(tmp_path) -> Database:
    db = Database(str(tmp_path / "why.db"))
    db.create_tables()
    return db


def _add_mechanism(
    db: Database,
    *,
    name: str,
    interval: int | None,
    source_message_id: int | None,
    created_at: datetime,
) -> None:
    with Session(db.engine) as session:
        session.add(
            MemoryRow(
                name=name,
                type="collection",
                description=f"{name} watch",
                inclusion="never",
                recall="recent",
                extraction_prompt="1. browse the page",
                collector_interval_seconds=interval,
                source_message_id=source_message_id,
                created_by_run_id="run-xyz",
                created_at=created_at,
                updated_at=created_at,
            )
        )
        session.commit()


def _add_send(db: Database, *, content: str, mechanism: str | None) -> int:
    with Session(db.engine) as session:
        row = MessageLog(
            direction=_OUTGOING,
            sender="penny",
            content=content,
            mechanism=mechanism,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        assert row.id is not None
        return row.id


async def _explain(db: Database, message_id: int):
    return await WhyDidISendThatTool(db).execute(message_id=message_id)


@pytest.mark.asyncio
async def test_direct_reply_render(tmp_path):
    """A NULL-mechanism row is a direct reply — never gated, no mechanism."""
    db = _db(tmp_path)
    message_id = _add_send(db, content="sure, here you go!", mechanism=None)
    result = await _explain(db, message_id)
    assert result.success is True
    assert result.message == "Direct reply in conversation."


@pytest.mark.asyncio
async def test_autonomous_with_source_message_render(tmp_path):
    """An autonomous send names its mechanism, cadence, when it was created, and
    the user request that created it."""
    db = _db(tmp_path)
    source_id = db.messages.log_message(
        _INCOMING, "+15551230000", "watch the switch 2 price for me"
    )
    assert source_id is not None
    _add_mechanism(
        db,
        name="price-watch",
        interval=21600,
        source_message_id=source_id,
        created_at=datetime(2026, 7, 11, 9, 14, tzinfo=UTC),
    )
    message_id = _add_send(db, content="the price just dropped!", mechanism="price-watch")
    result = await _explain(db, message_id)
    assert result.message == (
        "Sent by 'price-watch' (every 6h), created 2026-07-11 09:14 UTC "
        'from your message: "watch the switch 2 price for me"'
    )


@pytest.mark.asyncio
async def test_autonomous_without_source_message_render(tmp_path):
    """A seeded / system mechanism has no spawning user message — the render is
    just mechanism + cadence + created-when."""
    db = _db(tmp_path)
    _add_mechanism(
        db,
        name="thoughts",
        interval=5400,
        source_message_id=None,
        created_at=datetime(2026, 7, 11, 9, 14, tzinfo=UTC),
    )
    message_id = _add_send(db, content="been thinking about jazz today", mechanism="thoughts")
    result = await _explain(db, message_id)
    assert result.message == ("Sent by 'thoughts' (every 90m), created 2026-07-11 09:14 UTC.")


@pytest.mark.asyncio
async def test_autonomous_with_vanished_mechanism_render(tmp_path):
    """A send whose mechanism has left the registry (row deleted, never archived —
    archives keep the row) still renders honestly from the stamp alone."""
    db = _db(tmp_path)
    message_id = _add_send(db, content="an orphaned ping", mechanism="ghost-watch")
    result = await _explain(db, message_id)
    assert result.success is True
    assert result.message == "Sent by 'ghost-watch' (a background mechanism)."


@pytest.mark.asyncio
async def test_unknown_message_id_is_actionable_failure(tmp_path):
    """An id with no message is a failed call naming where a real id comes from."""
    db = _db(tmp_path)
    result = await _explain(db, 4242)
    assert result.success is False
    assert result.message == (
        "No message with id 4242 was found.  Pass the id of a message that "
        "exists (from the self-state activity block or a prior read)."
    )
