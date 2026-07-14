"""Emission provenance — the delivered message names its mechanism (#1568).

Type: schema

Part of the tool-surface hardening epic (#1554): the reverse index — from an
autonomous send back to the mechanism that caused it.  One schema addition,
universal (no deployment-specific rows):

**``messagelog.mechanism``** — the bound collection whose autonomous cycle
produced this send; NULL = a direct reply (a chat turn with a live triggering
user message names no mechanism).  Stamped by the ``SendQueueDrainer`` at
delivery time from the queued row's ``collection``, so "which mechanism sent
this, and from which request?" is a read (``why_did_i_send_that``), not an
80-minute diagnosis.

Plus the partial ``ix_messagelog_emission_time`` index: ``recent_emissions``
(the self-state activity block's autonomous-send lines) runs on the chat prompt
build hot path filtering ``mechanism IS NOT NULL`` and ordering by
``timestamp DESC, id DESC`` over the unbounded table, and mechanism-bearing rows
are sparse, so the filter+sort needs its own index.

Schema only.  A fresh ``create_tables()`` DB already carries these from the
models (the guards below skip them); a production copy that predates them gets
them added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_messagelog_mechanism(conn)
    conn.commit()


def _add_messagelog_mechanism(conn: sqlite3.Connection) -> None:
    """Add ``mechanism`` to ``messagelog`` + the partial emission index (idempotent)."""
    if not _table_exists(conn, "messagelog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()}
    if "mechanism" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN mechanism TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_messagelog_emission_time "
        "ON messagelog(timestamp, id) WHERE mechanism IS NOT NULL"
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
