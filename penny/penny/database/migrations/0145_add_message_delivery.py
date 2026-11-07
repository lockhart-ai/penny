"""Add ``message_delivery`` — one attempt to reach one device with one outgoing
message (#3137).

Type: schema

Penny's own messages now fan out to every registered device, and
``messagelog.external_id`` is a single column that can hold only one
platform id — it stops working the moment one logged row reaches several
channels, each minting its own.  This table is the delivery record: one row
per ``(message, device)`` attempt, written by the primary send and by every
fan-out send alike, so "which clients got a message" is a read, never a
reconstruction from scattered logs.

Schema only, universal — a fresh ``create_tables()`` DB already carries the
table from the model (the guard skips it); a production copy that predates
it gets it created.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "message_delivery"):
        conn.execute(
            """
            CREATE TABLE message_delivery (
                id INTEGER NOT NULL PRIMARY KEY,
                message_id INTEGER NOT NULL REFERENCES messagelog (id),
                device_id INTEGER NOT NULL REFERENCES device (id),
                delivered_at DATETIME NOT NULL,
                external_id VARCHAR,
                error VARCHAR
            )
            """
        )
        conn.execute("CREATE INDEX ix_message_delivery_message_id ON message_delivery (message_id)")
        conn.execute("CREATE INDEX ix_message_delivery_device_id ON message_delivery (device_id)")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:name", {"name": name}
    ).fetchone()
    return row is not None
