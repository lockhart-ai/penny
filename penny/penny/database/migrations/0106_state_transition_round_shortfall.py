"""The round's partial binding on the transition log — ``state_transition.round_shortfall``
(#1894).

Type: schema

The binder now runs at REQUEST entry as well as at apply entry, so a round parked waiting
for a detail knows exactly what it is waiting on: the routine that covers the ask, the
values the user's words already settled, and the parameters that got none.  That answer has
to outlive the turn that drew it — the next message is classified against a NAMED gap, and
the apply draw completes the binding from these settled values plus the arriving message
rather than re-reading the whole conversation.

It lives beside ``skill_frame`` because it is the same kind of round state for the other
half of the entry: a framing is a round whose values are all in hand, a shortfall is one
whose values are not.  One nullable TEXT column holding serialized ``RoundShortfall`` JSON,
carried by the state that can read it — set entering request, carried while the round stays
parked there, dropped by any move that leaves.

Schema only, universal, idempotent, and seeds nothing: every existing row keeps a NULL
binding, which is exactly the un-bound request round this deployment has been running.  A
fresh ``create_tables()`` DB already carries the column from the model (the guard skips it);
a production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_round_shortfall_column(conn)
    conn.commit()


def _add_round_shortfall_column(conn: sqlite3.Connection) -> None:
    """Add the partial-binding column when the table exists without it (idempotent)."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "state_transition" not in tables:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(state_transition)")}
    if "round_shortfall" in columns:
        return
    conn.execute("ALTER TABLE state_transition ADD COLUMN round_shortfall TEXT")
