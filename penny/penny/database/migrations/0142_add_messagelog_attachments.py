"""Add ``messagelog.attachments_json`` — an incoming message's photo attachments (#3084).

Type: schema

The iOS app can now send photos with a message, and its history surface needs
the user's own photo back after a sync so it renders in the user's bubble.  The
conversation record's text keeps the deterministic ``[sent a photo]`` marker
where the image was (``_logged_incoming_content``) — history for the MODEL never
resends the image — while this column carries the JSON list of data URIs the
pipeline was given, served on the iOS wire only.  Written by the base
pipeline's incoming-log sites; NULL for every text-only row.

One nullable column, no index (nothing orders or filters on it — the iOS
history page reads it off rows it already fetched).  Schema only, universal —
a fresh ``create_tables()`` DB already carries it from the model (the guard
skips it); a production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "messagelog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()}
    if "attachments_json" not in columns:
        conn.execute("ALTER TABLE messagelog ADD COLUMN attachments_json TEXT")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
