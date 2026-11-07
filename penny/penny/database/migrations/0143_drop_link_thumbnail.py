"""Drop the ``link_thumbnail`` table — nothing reads it any more (#3113).

Type: schema

Penny's server makes no web requests of its own: the egress fetches that
downloaded a link card's thumbnail (and the page's own ``og:image``) at send
time are retired, and with them the only two readers of this table —
``MessageChannel._capture_link_thumbnail`` (``lookup``) and
``BrowserChannel._record_link_thumbnails`` (``record``), both deleted
alongside this migration. A link card's own image now arrives as data on the
browse tool response and is stored directly as a ``media`` row at read time
(``BrowserChannel._store_link_card_media``), so there is no address-only
intermediate to keep.

Guarded (table present? then drop) so it is a no-op on a fresh
``create_tables``-first DB — the current models no longer declare the table,
so ``create_tables`` never materialises it and this drop finds nothing to
remove. On an upgraded prod DB the table exists (migration 0141) and is
dropped.

Universal — a plain table drop, no data to preserve (the table was always a
bounded cache of addresses, never anything downstream needed to survive) —
so it is safe on every deployment.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "link_thumbnail" not in tables:
        return
    conn.execute("DROP TABLE link_thumbnail")
    conn.commit()
