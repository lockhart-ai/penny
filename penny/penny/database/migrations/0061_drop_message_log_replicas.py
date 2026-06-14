"""Drop the user/penny message-log replicas — they're facades now.

Type: data

``user-messages`` and ``penny-messages`` are read facades over ``messagelog``
(the canonical store): ``MemoryStore`` serves their reads — ``log_read``,
``read_similar``, recall's hybrid ranking + temporal-neighbour expansion —
straight from ``messagelog`` rows, keyed by direction.  So the duplicated
``memory_entry`` rows the channel ingress/egress used to append are dead weight
and a data-duplication smell.

Drop the entry rows; keep the two ``memory`` rows themselves (the log markers
that carry inclusion/recall for routing).  ``messagelog.embedding`` is populated
by the startup embedding backfill (no rows are copied between tables).
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "DELETE FROM memory_entry WHERE memory_name IN ('user-messages', 'penny-messages')"
    )
    conn.commit()
