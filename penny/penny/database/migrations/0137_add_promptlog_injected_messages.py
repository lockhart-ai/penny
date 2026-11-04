"""Add ``promptlog.injected_messages`` — the user turns a chat run JOINED mid-run (#2985).

Type: schema

A message that arrives while a chat turn is still in its tool loop joins that turn:
the agent loop drains it off the turn's inbox after a tool step and appends it to the
conversation as a ``user`` turn, so the next model call carries it. The run's own
``messages`` snapshots hold those injected turns only transiently (each call's input);
the run's LAST prompt carries the full accumulated conversation, but nothing on the
record says which ``user`` turns in it are the user's mid-run messages versus the
framework's own (the narrate-facts turn, a validator nudge) — and history rendering
needs exactly that, positioned: the injected message renders after the tool step that
preceded it, never merged into the triggering exchange.

This column holds that: a JSON list of ``{"content": ..., "after_tool_call_id": ...}``
in drain order — the record form of what the user said (the same shape the message's
``messagelog`` row logs, image-history marker included) and the ORIGINAL id of the tool
call whose results preceded the injection. Stamped on the run's last prompt row by
``set_run_injected_messages``, beside ``trailing_messages``/``link_handles``; NULL for
a run that joined nothing, and for every row that isn't its run's last — so every
pre-existing run renders byte-identically.

One nullable column. Schema only, universal — a fresh ``create_tables()`` DB already
carries it from the model (the guard skips it); a production copy that predates it
gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_promptlog_injected_messages(conn)
    conn.commit()


def _add_promptlog_injected_messages(conn: sqlite3.Connection) -> None:
    """Add the nullable ``injected_messages`` column to ``promptlog`` (idempotent)."""
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "injected_messages" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN injected_messages TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
