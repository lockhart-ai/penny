"""Tests for migration 0126 — backfilling ``device_id`` on Signal messagelog
rows left NULL by #2765 (startup registered Penny's own number, not the
user's, so no Signal device ever matched an incoming sender or an outgoing
recipient).

Every test here isolates 0126 from the rest of the migration chain: it marks
every PRIOR migration already applied (the real production timeline — by the
time 0126 ever runs, 0016 and everything else already ran, once, long
before), then seeds the exact pre-0126 state directly, then calls
``migrate()`` so only 0126 executes. Running the full chain from an empty
schema instead would make ``0016`` re-seed a device from scratch against
data it would never actually see in production — on a DB this test pre-seeds
with a stale Penny-number device row, that re-seed collides with the very
bug this migration fixes, in a way 0016 never would on the real timeline
(0016 ran against a device-table-free DB; the Penny-number row didn't exist
until years of later runtime bug restarts created it).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from penny.database import Database
from penny.database.migrate import _discover_migrations, migrate

_PRIMARY = "+15551234567"  # the real user's number
_RARE = "+15559999999"  # a second, less-frequent real user number
_PENNY_OWN_NUMBER = "+15558887777"  # Penny's own sender on outgoing rows
_MIGRATION_NAME = "0126_backfill_signal_device"


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — migrations off so we control timing."""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _isolate_0126(conn: sqlite3.Connection) -> None:
    """Mark every migration except 0126 as already applied, so ``migrate()``
    runs ONLY 0126 against whatever this test seeded — the real production
    shape, where every earlier migration (0016 included) ran long ago."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    now = datetime.now(UTC).isoformat()
    for name, _path in _discover_migrations():
        if name == _MIGRATION_NAME:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO _migrations (name, applied_at) VALUES (?, ?)", (name, now)
        )


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


def test_backfills_device_id_by_direction_for_both_senders(tmp_path):
    """Two real Signal numbers have messaged in — one far more than the
    other. Every NULL row for BOTH numbers must link, by direction: an
    incoming row to the device matching its sender, an outgoing row to the
    device matching its recipient (never its sender — see the regression
    test below for why that distinction is load-bearing)."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0126(conn)
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
    identifiers = {row[0] for row in conn.execute("SELECT identifier FROM device").fetchall()}
    assert identifiers == {_PRIMARY, _RARE}

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


def test_outgoing_replies_never_link_to_pennys_stale_default_device(tmp_path):
    """The exact live-DB shape (#2765 follow-up): a Signal device row for
    PENNY'S OWN number already exists and is marked default — the residue of
    the pre-fix startup bug — and NO device for the user's own number exists
    yet (that absence is the whole bug). Outgoing replies (sender = Penny's
    own number, recipient = the user) must link to the USER's device by
    recipient — never to Penny's device by a direction-blind sender match,
    which would silently steal them onto the one device startup is about to
    delete, orphaning them back to NULL."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0126(conn)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    # The stale default, exactly as the pre-fix runtime left it. No device
    # for the user's own number exists — that's the bug #2765 fixes.
    conn.execute(
        "INSERT INTO device (channel_type, identifier, label, is_default, created_at)"
        " VALUES ('signal', ?, 'Signal', 1, ?)",
        (_PENNY_OWN_NUMBER, now.isoformat()),
    )
    for i in range(5):
        _seed_message(conn, "incoming", _PRIMARY, None, now + timedelta(minutes=i))
        _seed_message(
            conn,
            "outgoing",
            _PENNY_OWN_NUMBER,
            _PRIMARY,
            now + timedelta(minutes=i, seconds=30),
        )
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    penny_device_id = conn.execute(
        "SELECT id FROM device WHERE identifier = ?", (_PENNY_OWN_NUMBER,)
    ).fetchone()[0]
    user_device_id = conn.execute(
        "SELECT id FROM device WHERE identifier = ?", (_PRIMARY,)
    ).fetchone()[0]
    assert penny_device_id != user_device_id

    rows = conn.execute("SELECT direction, device_id FROM messagelog").fetchall()
    assert len(rows) == 10
    for _direction, device_id in rows:
        assert device_id == user_device_id
        assert device_id != penny_device_id
    conn.close()


def test_never_touches_a_non_signal_row(tmp_path):
    """A Discord-shaped sender/recipient (no leading ``+``) is never seeded
    as a Signal device and never gets a ``device_id`` written by 0126."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0126(conn)
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


def test_does_not_touch_is_default(tmp_path):
    """The migration never marks or clears ``is_default`` — a pre-existing
    default (whatever it is) is left exactly as found, and every Signal
    device it seeds comes in as non-default, because reconciling the single
    global default is startup's job (``_register_signal_channel``), not a
    migration that runs on every deployment including ones no longer primary
    on Signal."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0126(conn)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    conn.execute(
        "INSERT INTO device (channel_type, identifier, label, is_default, created_at)"
        " VALUES ('ios', 'some-ios-device', 'iPhone', 1, ?)",
        (now.isoformat(),),
    )
    _seed_message(conn, "incoming", _PRIMARY, None, now)
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    default_identifier = conn.execute(
        "SELECT identifier FROM device WHERE is_default = 1"
    ).fetchone()[0]
    assert default_identifier == "some-ios-device"
    assert conn.execute("SELECT COUNT(*) FROM device WHERE is_default = 1").fetchone()[0] == 1
    new_signal_device = conn.execute(
        "SELECT is_default FROM device WHERE identifier = ?", (_PRIMARY,)
    ).fetchone()[0]
    assert new_signal_device == 0
    conn.close()
