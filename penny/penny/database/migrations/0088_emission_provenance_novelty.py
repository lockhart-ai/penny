"""Emission provenance + novelty-keyed suppression columns (#1568).

Type: schema

Wave of the tool-surface hardening epic (#1554): the reverse index — from an
autonomous send back to the mechanism that caused it, plus the novelty key that
gates a mechanism from re-sending the same news.  Two schema additions, both
universal (no deployment-specific rows):

1. **``messagelog`` provenance** — ``mechanism`` (the bound collection that sent
   this message, NULL = a direct reply) and ``novelty_key`` (the emission's
   novelty identity, NULL = a direct reply).  A direct reply is never gated and
   carries neither; an autonomous send carries both, stamped by the drainer at
   delivery time from the ``send_queue`` row.  With ``mechanism`` on the delivered
   row, "which mechanism sent this, and from which request?" is a read
   (``why_did_i_send_that``), not an 80-minute diagnosis.

2. **``send_queue`` novelty + suppression** — ``novelty_key`` (computed in Python
   from the cycle's write-gate outcomes at enqueue time) and ``suppressed_reason``
   (set on a durably-recorded emission that was held back because its novelty key
   matched the last one for the same mechanism; NULL for a real emission).  A
   suppressed row is never delivered — it stays ``sent_at IS NULL`` but is excluded
   from the pending drain by its non-NULL ``suppressed_reason`` — so the record of
   "we chose not to re-send this, and why" is durable and datetime-ordered.

Schema only.  A fresh ``create_tables()`` DB already carries these from the models
(the guards below skip them); a production copy that predates them gets them added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_messagelog_provenance(conn)
    _add_send_queue_novelty(conn)
    conn.commit()


def _add_messagelog_provenance(conn: sqlite3.Connection) -> None:
    """Add ``mechanism`` + ``novelty_key`` to ``messagelog`` (idempotent), plus the
    partial index over emission rows — ``recent_emissions`` runs on the self-state
    render hot path (every chat prompt build) filtering ``mechanism IS NOT NULL``
    and ordering by ``timestamp DESC, id DESC`` over the unbounded table, and
    mechanism-bearing rows are sparse, so the filter+sort needs its own index."""
    if not _table_exists(conn, "messagelog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()}
    if "mechanism" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN mechanism TEXT")
    if "novelty_key" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN novelty_key TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_messagelog_emission_time "
        "ON messagelog(timestamp, id) WHERE mechanism IS NOT NULL"
    )


def _add_send_queue_novelty(conn: sqlite3.Connection) -> None:
    """Add ``novelty_key`` + ``suppressed_reason`` to ``send_queue`` (idempotent)."""
    if not _table_exists(conn, "send_queue"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(send_queue)").fetchall()}
    if "novelty_key" not in columns:
        conn.execute("ALTER TABLE send_queue ADD COLUMN novelty_key TEXT")
    if "suppressed_reason" not in columns:
        conn.execute("ALTER TABLE send_queue ADD COLUMN suppressed_reason TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
