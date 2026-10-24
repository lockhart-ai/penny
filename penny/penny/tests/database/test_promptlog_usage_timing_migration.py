"""Tests for migration 0129 — backfilling ``promptlog``'s five usage/timing
columns from each row's own stored ``response`` JSON (#2793).

Every test isolates 0129 from the rest of the migration chain the same way
``test_signal_device_migration.py`` isolates 0126: every prior migration is
marked already applied, this test seeds the exact pre-0129 ``promptlog`` rows
directly, then ``migrate()`` runs only 0129 against them.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from penny.database import Database
from penny.database.migrate import _discover_migrations, migrate

_MIGRATION_NAME = "0129_promptlog_usage_timing"


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _isolate_0129(conn: sqlite3.Connection) -> None:
    """Mark every migration except 0129 as already applied, so ``migrate()``
    runs ONLY 0129 against whatever this test seeded."""
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


def _seed_row(conn: sqlite3.Connection, response: dict | None) -> int:
    cursor = conn.execute(
        "INSERT INTO promptlog (model, messages, response, timestamp) VALUES ('m', '[]', ?, ?)",
        (
            json.dumps(response) if response is not None else "{}",
            datetime.now(UTC).isoformat(),
        ),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _row(conn: sqlite3.Connection, row_id: int) -> tuple:
    return conn.execute(
        "SELECT prompt_tokens, cached_tokens, completion_tokens, prefill_ms, generation_ms"
        " FROM promptlog WHERE id = ?",
        (row_id,),
    ).fetchone()


def test_backfills_each_runner_shape_and_leaves_absent_fields_null(tmp_path):
    """One row per runner shape, plus a usage-less row — every column lands
    exactly like ``usage_timing_from_response`` would compute it live, and an
    absent field backfills to NULL, never a guessed zero."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0129(conn)

    omlx_id = _seed_row(
        conn,
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 20},
                "prompt_eval_duration": 0.5,
                "generation_duration": 1.2,
            }
        },
    )
    ollama_id = _seed_row(
        conn,
        {
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "prompt_eval_duration": 500_000_000,
                "eval_duration": 1_200_000_000,
            }
        },
    )
    openrouter_id = _seed_row(
        conn,
        {
            "usage": {
                "prompt_tokens": 60,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 5},
            }
        },
    )
    no_usage_id = _seed_row(conn, {"choices": []})
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    assert _row(conn, omlx_id) == (100, 20, 50, 500, 1200)
    assert _row(conn, ollama_id) == (80, None, 40, 500, 1200)
    assert _row(conn, openrouter_id) == (60, 5, 30, None, None)
    assert _row(conn, no_usage_id) == (None, None, None, None, None)
    conn.close()


def test_a_row_with_no_promptlog_table_is_a_no_op(tmp_path):
    """A schema predating ``promptlog`` (none in practice, but the migration's
    own table-existence guard) never raises."""
    db_path = str(tmp_path / "bare.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE _migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
    now = datetime.now(UTC).isoformat()
    for name, _path in _discover_migrations():
        if name == _MIGRATION_NAME:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO _migrations (name, applied_at) VALUES (?, ?)", (name, now)
        )
    conn.commit()
    conn.close()

    migrate(db_path)  # does not raise


def test_rerunning_the_backfill_is_idempotent(tmp_path):
    """A second application re-derives the same values from the same stored
    ``response`` — never doubling or drifting."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0129(conn)
    row_id = _seed_row(conn, {"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    conn.commit()
    conn.close()

    migrate(db.db_path)
    conn = sqlite3.connect(db.db_path)
    first = _row(conn, row_id)
    conn.close()

    migrate(db.db_path)
    conn = sqlite3.connect(db.db_path)
    second = _row(conn, row_id)
    conn.close()

    assert first == second == (10, None, 5, None, None)
