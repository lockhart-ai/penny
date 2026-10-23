"""Backfill ``device_id`` on Signal messagelog rows left NULL by #2765.

Startup used to register Penny's own Signal number as the device instead of
the user's, so no incoming Signal message — and no reply to one — ever
linked to a device.  This migration repairs the historical rows with generic
criteria only: a phone-number-shaped (``+``-prefixed) sender or recipient on
``messagelog``, the same shape ``0016_add_device_table`` already used to
recognise a Signal row.  It never touches a Discord/iOS/browser row (none of
their identifiers are ``+``-shaped) and never references this deployment's
own number.

Direction matters here, and matters a lot: on a live deployment, an
``INCOMING`` row's sender and an ``OUTGOING`` row's sender are different
parties (the user, and Penny herself) even though both numbers are
``+``-shaped. An earlier version of this migration matched sender for
EVERY NULL row regardless of direction — so an outgoing reply (sender =
Penny's own number, recipient = the user) got linked to Penny's own
device whenever that stale row still existed (seeded by the pre-#2765
startup bug, not by this migration), because the sender-match clause had
no direction filter to keep it out. Each UPDATE below is now direction-
scoped AND reads only the user's own party for that direction — an
incoming row's sender, an outgoing row's recipient — so neither clause can
ever touch the other direction's Penny-keyed column.

This migration never touches ``device.is_default``. Seeding every distinct
historical sender as a device needs no default to be correct, and forcing
one here risks clobbering a genuinely different primary channel's default
(iOS, browser) on a deployment that has since moved off Signal. Hands-off,
every historical row still resolves identically because the backfill below
never depends on which seeded device is marked default. The user's own
Signal device still ends up default on any deployment where Signal is the
primary channel — that's `_register_signal_channel`'s job, every startup.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_SIGNAL_SENDER_SHAPE = "+%"


def up(conn: sqlite3.Connection) -> None:
    """Seed a Signal device per distinct historical sender, then backfill
    ``device_id`` on NULL rows that match a seeded identifier, by direction."""
    if not _table_exists(conn, "messagelog") or not _table_exists(conn, "device"):
        return
    seeded = _seed_signal_devices(conn)
    backfilled = _backfill_device_id(conn)
    if seeded or backfilled:
        logger.info(
            "0126 seeded %d Signal device(s) and backfilled device_id on %d messagelog row(s)",
            seeded,
            backfilled,
        )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    )


def _seed_signal_devices(conn: sqlite3.Connection) -> int:
    """Register every distinct phone-shaped INCOMING sender as a (non-default)
    Signal device — one row per real number the deployment has ever heard
    from, not just the first one ``0016`` happened to seed. Deliberately
    reads ``direction='incoming'`` only: an outgoing row's sender is Penny's
    own number, never a device to register. Returns how many new device rows
    this inserted (``INSERT OR IGNORE`` skips any that ``0016`` or a prior run
    already seeded)."""
    senders = conn.execute(
        "SELECT DISTINCT sender FROM messagelog WHERE direction='incoming' AND sender LIKE ?",
        (_SIGNAL_SENDER_SHAPE,),
    ).fetchall()
    now = datetime.now(UTC).isoformat()
    inserted = 0
    for (sender,) in senders:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO device"
            " (channel_type, identifier, label, is_default, created_at)"
            " VALUES ('signal', ?, 'Signal', 0, ?)",
            (sender, now),
        )
        inserted += cursor.rowcount
    return inserted


def _backfill_device_id(conn: sqlite3.Connection) -> int:
    """Link every still-NULL row whose OWN-DIRECTION party matches a seeded
    device: an incoming row by its sender, an outgoing row by its recipient.
    Each clause is scoped to its own direction so an outgoing row's sender
    (Penny's own number, never a party this migration seeds) can never be
    read by the incoming clause, and vice versa. Returns the total rows
    updated across both directions."""
    incoming = conn.execute(
        "UPDATE messagelog"
        " SET device_id = (SELECT id FROM device WHERE identifier = messagelog.sender)"
        " WHERE device_id IS NULL AND direction='incoming' AND sender LIKE ?"
        " AND sender IN (SELECT identifier FROM device)",
        (_SIGNAL_SENDER_SHAPE,),
    ).rowcount
    outgoing = conn.execute(
        "UPDATE messagelog"
        " SET device_id = (SELECT id FROM device WHERE identifier = messagelog.recipient)"
        " WHERE device_id IS NULL AND direction='outgoing' AND recipient LIKE ?"
        " AND recipient IN (SELECT identifier FROM device)",
        (_SIGNAL_SENDER_SHAPE,),
    ).rowcount
    return incoming + outgoing
