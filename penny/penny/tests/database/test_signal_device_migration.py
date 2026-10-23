"""Tests for migration 0126 — backfilling ``device_id`` on Signal messagelog
rows left NULL by #2765 (startup registered Penny's own number, not the
user's, so no Signal device ever matched an incoming sender or an outgoing
recipient).

Seeds the pre-migration shape directly (raw ``messagelog`` rows, no device
row at all — the state a real deployment is in today), runs the full
migration chain, and asserts the repaired state: a Signal device per
distinct historical sender, exactly one of them default, and every
previously-NULL Signal row linked to its device.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from penny.database import Database
from penny.database.migrate import migrate

_PRIMARY = "+15551234567"
_RARE = "+15559999999"
_PENNY_OWN_NUMBER = "+15558887777"  # Penny's own sender on outgoing rows


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — migrations off so we control timing."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _seed_message(
    conn: sqlite3.Connection,
    direction: str,
    sender: str,
    recipient: str | None,
    timestamp: datetime,
) -> None:
    conn.execute(
        "INSERT INTO messagelog"
        " (direction, sender, recipient, content, timestamp, is_reaction, processed)"
        " VALUES (?, ?, ?, 'hi', ?, 0, 0)",
        (direction, sender, recipient, timestamp.isoformat()),
    )


def test_backfills_device_id_and_defaults_the_primary_sender(tmp_path):
    """Two real Signal numbers have messaged in — one far more than the
    other. With no ``userinfo`` profile yet, the migration must default the
    MOST FREQUENT sender, not just any of them, and link every NULL row for
    BOTH numbers (not only the default one)."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(3):
        _seed_message(conn, "incoming", _PRIMARY, None, now + timedelta(minutes=i))
        _seed_message(
            conn, "outgoing", _PENNY_OWN_NUMBER, _PRIMARY, now + timedelta(minutes=i, seconds=30)
        )
    _seed_message(conn, "incoming", _RARE, None, now + timedelta(hours=1))
    # A non-Signal (Discord-shaped) row — must never be touched.
    _seed_message(conn, "incoming", "123456789012345678", None, now + timedelta(hours=2))
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    devices = {
        row[0]: (row[1], row[2])  # identifier -> (channel_type, is_default)
        for row in conn.execute(
            "SELECT identifier, channel_type, is_default FROM device"
        ).fetchall()
    }
    assert devices[_PRIMARY] == ("signal", 1)
    assert devices[_RARE] == ("signal", 0)
    assert sum(is_default for _, is_default in devices.values()) == 1

    device_ids = dict(conn.execute("SELECT identifier, id FROM device").fetchall())
    rows = conn.execute(
        "SELECT direction, sender, recipient, device_id FROM messagelog ORDER BY timestamp"
    ).fetchall()
    for direction, sender, recipient, device_id in rows[:-1]:  # all but the Discord row
        if direction == "incoming":
            assert device_id == device_ids[sender]
        else:
            assert device_id == device_ids[recipient]
    discord_row = rows[-1]
    assert discord_row[3] is None
    conn.close()


def test_falls_back_to_most_frequent_sender_without_a_profile(tmp_path):
    """No ``userinfo`` row at all — the default must be the sender seen most
    often, and every Signal row (both senders) gets a non-NULL device_id."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    _seed_message(conn, "incoming", _RARE, None, now)
    for i in range(2):
        _seed_message(conn, "incoming", _PRIMARY, None, now + timedelta(minutes=i + 1))
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    default_identifier = conn.execute(
        "SELECT identifier FROM device WHERE is_default = 1"
    ).fetchone()[0]
    assert default_identifier == _PRIMARY

    null_count = conn.execute(
        "SELECT COUNT(*) FROM messagelog WHERE direction='incoming' AND device_id IS NULL"
    ).fetchone()[0]
    assert null_count == 0
    conn.close()


def test_primary_sender_wins_over_a_more_frequent_rare_number(tmp_path):
    """A ``userinfo`` profile already names the real user — that wins the
    default even though another number happens to have messaged in more."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        _seed_message(conn, "incoming", _RARE, None, now + timedelta(minutes=i))
    _seed_message(conn, "incoming", _PRIMARY, None, now + timedelta(hours=1))
    conn.execute(
        "INSERT INTO userinfo"
        " (sender, name, location, timezone, date_of_birth, created_at, updated_at)"
        " VALUES (?, 'User', 'Toronto', 'America/Toronto', '1990-01-01', ?, ?)",
        (_PRIMARY, now.isoformat(), now.isoformat()),
    )
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    default_identifier = conn.execute(
        "SELECT identifier FROM device WHERE is_default = 1"
    ).fetchone()[0]
    assert default_identifier == _PRIMARY
    assert conn.execute("SELECT COUNT(*) FROM device WHERE is_default = 1").fetchone()[0] == 1
    conn.close()


def test_never_touches_a_non_signal_row(tmp_path):
    """A Discord-shaped sender/recipient (no leading ``+``) is never seeded
    as a Signal device and never gets a ``device_id`` written by 0126 — a
    real Signal row is seeded first here so the assertion isolates 0126's
    own ``+``-only criterion from ``0016``'s separate, pre-existing
    first-incoming-sender seed, rather than relying on this row being the
    only one in the database."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    discord_sender = "987654321098765432"
    _seed_message(conn, "incoming", _PRIMARY, None, now)
    _seed_message(conn, "incoming", discord_sender, None, now + timedelta(hours=1))
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM device WHERE identifier = ?", (discord_sender,)
        ).fetchone()[0]
        == 0
    )
    row = conn.execute(
        "SELECT device_id FROM messagelog WHERE sender = ?", (discord_sender,)
    ).fetchone()
    assert row[0] is None
    conn.close()
