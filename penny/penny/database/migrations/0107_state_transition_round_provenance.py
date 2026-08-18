"""The round's PROVENANCE on the transition log — ``state_transition.round_provenance``
(#1902).

A round's run-end extraction writes its routine under the name the round's framing pinned,
so a round TEACHING a new job creates that registry row while a round RE-TEACHING one the
user already has overwrites it — and from the row alone those two writes are identical.
The difference is a fact about the ROUND, and it decides what calling the round off owes
the registry: a minted routine goes, a re-taught one goes BACK to what it was, and a round
that only bound a routine the user already had wrote nothing and is owed nothing.

Skipping the delete would not be enough for the re-teach case, which is why the column
holds the whole pre-round row: by the time the user bails, the round's own extraction has
already replaced what that canonical routine DOES, so leaving it standing would leave an
abandoned, half-corrected program live under a name existing jobs still run.

One nullable TEXT column holding serialized ``RoundProvenance`` JSON, beside ``skill_frame``
and ``round_shortfall`` because it is the same kind of round state, settled at the same
moment (the move that mints the round's routine — the last point at which "what was here
before" is still readable) and carried the same way: set on entry, carried while parked,
NULL once idle.

Schema only, universal, idempotent, and seeds nothing: every existing row keeps a NULL
provenance, which is exactly the un-tracked round this deployment has been running.  A
fresh ``create_tables()`` DB already carries the column from the model (the guard skips
it); a production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_round_provenance_column(conn)
    conn.commit()


def _add_round_provenance_column(conn: sqlite3.Connection) -> None:
    """Add the provenance column when the table exists without it (idempotent)."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "state_transition" not in tables:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(state_transition)")}
    if "round_provenance" in columns:
        return
    conn.execute("ALTER TABLE state_transition ADD COLUMN round_provenance TEXT")
