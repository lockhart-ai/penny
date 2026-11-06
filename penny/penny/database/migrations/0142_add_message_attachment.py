"""Add ``message_attachment`` — an incoming message's photo attachments (#3084).

Type: schema

The iOS app can now send photos with a message, and its history surface needs
the user's own photo back after a sync so it renders in the user's bubble.  The
conversation record's text keeps the deterministic ``[sent a photo]`` marker
(``_logged_incoming_content``) — history for the MODEL never resends the image
— while this table carries one row per photo (the data URI the pipeline was
given, in ``position`` order), written by the base pipeline's incoming-log
sites and served on the iOS wire only.

A separate table keyed by a foreign key to ``messagelog``, NOT a column on
``messagelog`` itself: the data URIs run to hundreds of kilobytes each, and
``messagelog`` is the hottest table in the app — a blob column there would
load into every history window, conversation map, ``find`` and ``log_read``
read.  Read in one batched query per iOS history page, only when the client
asks for attachments.

Schema only, universal — a fresh ``create_tables()`` DB already carries the
table from the model (the guard skips it); a production copy that predates it
gets it created.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "message_attachment"):
        conn.execute(
            """
            CREATE TABLE message_attachment (
                id INTEGER NOT NULL PRIMARY KEY,
                message_id INTEGER NOT NULL REFERENCES messagelog (id),
                position INTEGER NOT NULL,
                data_uri VARCHAR NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX ix_message_attachment_message_id ON message_attachment (message_id)"
        )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:name", {"name": name}
    ).fetchone()
    return row is not None
