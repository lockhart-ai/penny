"""Drop the ``email_rule.user_id`` column — Penny is single-user (#1737).

Type: schema

``EmailRule.user_id`` was multi-user scaffolding: on a single-user deployment
every rule carried the same value (``signal_number or "default"``) and every
query filtered on it redundantly.  The column and its index are removed; the
``provider`` column stays (it distinguishes a future non-Zoho email backend).

Guarded (table present? column present? then drop) so it is a no-op on a fresh
``create_tables``-first DB — the current ``EmailRule`` model no longer declares
``user_id``, so ``create_tables`` never materialises it and this drop finds
nothing to remove.  On an upgraded prod DB the column exists and is dropped; its
single-column index (``ix_email_rule_user_id``) is dropped first so the column
drop is clean on every SQLite version.

Universal — a plain column drop — so it is safe on every deployment.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "email_rule" not in tables:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(email_rule)").fetchall()}
    if "user_id" not in columns:
        return
    conn.execute("DROP INDEX IF EXISTS ix_email_rule_user_id")
    conn.execute("ALTER TABLE email_rule DROP COLUMN user_id")
    conn.commit()
