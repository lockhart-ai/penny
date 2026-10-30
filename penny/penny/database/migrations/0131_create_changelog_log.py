"""Create the ``changelog`` log memory (#2845).

Entries written FOR Penny about changes to her tools, prompts and behaviour
— one per PR that changed what she does, authored in second person and
recorded at startup from ``penny/penny/changelog.yaml``
(``penny.changelog.record_new_entries``). A plain append-only log, same
shape as ``browse-results``/``user-messages`` — its rows ARE the stored
content, so it needs no special facade class, unlike ``collector-runs``.

The model never gets this ambiently — she reads it explicitly with
``log_read('changelog')`` when asked what changed, or when she's attributing
a behaviour shift during her own self-reflection. (No ``recall`` column:
migration 0093 dropped the whole recall substrate well before this one was
written — a log this new reads only through ``log_read``, same as
``collector-runs``/``penny-messages``/``user-messages``.)

Idempotent — INSERT OR IGNORE. The marker row's existence is all this
migration is responsible for; the entries themselves are written at
startup, never by a migration (deployment timing, not universal data).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory (name, type, description, archived, created_at) "
        "VALUES (?, 'log', ?, 0, ?)",
        (
            "changelog",
            "Changes made to your tools, prompts and behaviour, recorded at each deploy",
            now,
        ),
    )
    conn.commit()
