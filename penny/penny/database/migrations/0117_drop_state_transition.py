"""Drop the ``state_transition`` table — the conversation state machine it logged
is retired (#2538).

Type: schema

The naive-jobs branch (#2455 / #2537) deleted the conversation machine (the
classifier, states, round framing) in favour of a free-text job prompt; no code
has written a ``state_transition`` row since, ``MachineStore``/``db.machine``
stayed in place only as inert plumbing, and both are removed alongside this
migration.  Guarded (table present? then drop) so it is a no-op on a fresh
``create_tables``-first DB — the current models no longer declare the table, so
``create_tables`` never materialises it and this drop finds nothing to remove.
On an upgraded prod DB the table exists (empty, since nothing ever wrote to it on
this branch) and is dropped.

Universal — a plain table drop, no data to preserve — so it is safe on every
deployment.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "state_transition" not in tables:
        return
    conn.execute("DROP TABLE state_transition")
    conn.commit()
