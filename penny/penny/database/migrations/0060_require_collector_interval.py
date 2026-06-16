"""Backfill NULL collector intervals to one hour — the cadence is now required.

Type: data

``collector_interval_seconds`` is required at the tool surface for every
collection that runs a collector (``CollectionCreateArgs`` requires it), and the
dispatcher no longer falls back to a global default — a collector collection
with no interval is skipped (``Collector._is_ready``), not silently run at 300s.

Collections created before the requirement may still carry NULL.  This sets
them — and their snap-back ``base_interval_seconds`` — to one hour so they keep
running on a sane cadence instead of being skipped.  Only collector collections
(``extraction_prompt IS NOT NULL``) are touched; logs and passive collections
have no cadence by design and keep their NULL.
"""

from __future__ import annotations

import sqlite3

ONE_HOUR_SECONDS = 3600


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET collector_interval_seconds = ? "
        "WHERE collector_interval_seconds IS NULL AND extraction_prompt IS NOT NULL",
        (ONE_HOUR_SECONDS,),
    )
    conn.execute(
        "UPDATE memory SET base_interval_seconds = collector_interval_seconds "
        "WHERE base_interval_seconds IS NULL AND extraction_prompt IS NOT NULL"
    )
    conn.commit()
