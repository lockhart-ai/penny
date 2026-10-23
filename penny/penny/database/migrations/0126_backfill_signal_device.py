"""Backfill ``device_id`` on Signal messagelog rows left NULL by #2765.

Startup used to register Penny's own Signal number as the device instead of
the user's, so no incoming Signal message — and no reply to one — ever
linked to a device.  This migration repairs the historical rows with generic
criteria only: a phone-number-shaped (``+``-prefixed) sender or recipient on
``messagelog``, the same shape ``0016_add_device_table`` already used to
recognise a Signal row.  It never touches a Discord/iOS/browser row (none of
their identifiers are ``+``-shaped) and never references this deployment's
own number.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_SIGNAL_SENDER_SHAPE = "+%"


def up(conn: sqlite3.Connection) -> None:
    """Seed a Signal device per distinct historical sender, mark one default,
    then backfill ``device_id`` on NULL rows that match a seeded identifier."""
    if not _table_exists(conn, "messagelog") or not _table_exists(conn, "device"):
        return
    seeded = _seed_signal_devices(conn)
    _mark_signal_default(conn)
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
    """Register every distinct phone-shaped incoming sender as a (non-default)
    Signal device — one row per real number the deployment has ever heard
    from, not just the first one ``0016`` happened to seed.  Returns how many
    new device rows this inserted (``INSERT OR IGNORE`` skips any that
    ``0016`` or a prior run already seeded)."""
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


def _mark_signal_default(conn: sqlite3.Connection) -> None:
    """Exactly one default: the user's own record when there is one, read the
    same way ``UserStore.get_primary_sender`` reads it — the sole ``userinfo``
    row — else the most-seen phone-shaped incoming sender.  No-op if nothing
    phone-shaped got seeded (a non-Signal deployment)."""
    any_signal = conn.execute(
        "SELECT 1 FROM device WHERE channel_type='signal' AND identifier LIKE ?",
        (_SIGNAL_SENDER_SHAPE,),
    ).fetchone()
    if not any_signal:
        return
    primary = _primary_sender(conn)
    if primary is None:
        return
    conn.execute("UPDATE device SET is_default = 0")
    conn.execute(
        "UPDATE device SET is_default = 1 WHERE channel_type='signal' AND identifier = ?",
        (primary,),
    )


def _primary_sender(conn: sqlite3.Connection) -> str | None:
    """The established user's number if a profile already exists, else the
    most frequent phone-shaped incoming sender (a fresh/unprofiled deployment
    whose live messages still name the one real user)."""
    if _table_exists(conn, "userinfo"):
        row = conn.execute("SELECT sender FROM userinfo LIMIT 1").fetchone()
        if row and row[0] and row[0].startswith("+"):
            return row[0]
    row = conn.execute(
        "SELECT sender, COUNT(*) AS c FROM messagelog"
        " WHERE direction='incoming' AND sender LIKE ?"
        " GROUP BY sender ORDER BY c DESC, sender ASC LIMIT 1",
        (_SIGNAL_SENDER_SHAPE,),
    ).fetchone()
    return row[0] if row else None


def _backfill_device_id(conn: sqlite3.Connection) -> int:
    """Link every still-NULL row whose sender or recipient matches a device
    this migration just seeded — incoming by sender, outgoing by recipient.
    Returns the total rows updated across both directions."""
    incoming = conn.execute(
        "UPDATE messagelog"
        " SET device_id = (SELECT id FROM device WHERE identifier = messagelog.sender)"
        " WHERE device_id IS NULL AND sender LIKE ?"
        " AND sender IN (SELECT identifier FROM device)",
        (_SIGNAL_SENDER_SHAPE,),
    ).rowcount
    outgoing = conn.execute(
        "UPDATE messagelog"
        " SET device_id = (SELECT id FROM device WHERE identifier = messagelog.recipient)"
        " WHERE device_id IS NULL AND recipient LIKE ?"
        " AND recipient IN (SELECT identifier FROM device)",
        (_SIGNAL_SENDER_SHAPE,),
    ).rowcount
    return incoming + outgoing
