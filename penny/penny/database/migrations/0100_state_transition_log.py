"""Conversation state machine persistence — the transition log (#1706).

Type: schema

The classifier (beats 0–5) decides one transition per incoming message over the
current state's out-edges, but nothing held the result: a decision that
evaporates with the turn is a function, not a machine.  Every parked state
(``elicit`` / ``learn`` / ``request``) exists to be READ by the NEXT message's
classification, so the state must outlive the turn that set it.

**One table.**  ``state_transition`` is an append-only log and the machine's
whole state is a fold over it — the newest row's ``to_state`` is where the
machine stands, its ``anchor_message_id`` is the ask a parked round is anchored
to, its ``created_at`` is when it last moved.  A materialized current-state row
alongside would carry nothing that isn't derivable, and would mean two writes
per move that can disagree; one write means a failed write moves nothing.  Each
row carries (from, to, cause, outcome, anchor, message, run, bound skill, when),
indexed on ``created_at`` for the latest-row read and on ``cause`` / ``run_id``
for the per-edge scoring joins.

Schema only — touches no deployment-specific rows, and seeds nothing: the cold
start is the ABSENCE of history (no rows = idle by the caller's definition of
idle), never a value frozen into DDL that a later rename could strand.  A fresh
``create_tables()`` DB already carries the table from the model (the guard skips
it); a production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _create_state_transition_table(conn)
    conn.commit()


def _create_state_transition_table(conn: sqlite3.Connection) -> None:
    """Create the transition log + its recency / scoring indexes (idempotent)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state_transition ("
        "  id INTEGER PRIMARY KEY,"
        "  from_state TEXT NOT NULL,"
        "  to_state TEXT NOT NULL,"
        "  cause TEXT NOT NULL,"
        "  outcome TEXT,"
        "  anchor_message_id INTEGER REFERENCES messagelog(id),"
        "  message_id INTEGER REFERENCES messagelog(id),"
        "  run_id TEXT,"
        "  skill_name TEXT,"
        "  created_at TIMESTAMP NOT NULL"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_state_transition_created_at ON state_transition(created_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_state_transition_cause ON state_transition(cause)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_state_transition_run_id ON state_transition(run_id)"
    )
