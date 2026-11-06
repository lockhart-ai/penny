"""Add ``link_thumbnail`` — link-card image addresses recorded at browse time (#3072).

Type: schema

The extension pairs each link card on a read page with its own image ADDRESS
(``BrowserToolResponse.link_thumbnails``, sent since #3071); the channel
upserts each entry into this table, keyed by the URL normalised the way the
link-handle table normalises a link.  A reply citing a link with no captured
``media`` row gets the card's image downloaded at egress and stored as a media
row, so the exact-URL tier fires for pages she cites without ever opening —
the og:image fetch runs only when no card is recorded.  Address only, never
bytes: the download happens at send time for the one URL being cited.

The table is a bounded cache (the store keeps the newest ~5,000 rows on
insert).  Schema only, universal — a fresh ``create_tables()`` DB already
carries the table from the model (the guard skips it); a production copy that
predates it gets it created.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "link_thumbnail"):
        conn.execute(
            """
            CREATE TABLE link_thumbnail (
                url VARCHAR NOT NULL PRIMARY KEY,
                image_url VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                seen_at DATETIME NOT NULL
            )
            """
        )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:name", {"name": name}
    ).fetchone()
    return row is not None
