"""Conversation state machine persistence — the machine row + its ledger (#1706).

Type: schema

The classifier (beats 0–5) decides one transition per incoming message over the
current state's out-edges, but nothing held the result: a decision that
evaporates with the turn is a function, not a machine.  Every parked state
(``elicit`` / ``learn`` / ``request``) exists to be READ by the NEXT message's
classification, so the state must outlive the turn that set it.

Two tables, the same split the mutation ledger draws between materialized truth
and audit trail:

1. ``conversation_machine`` — WHERE the machine stands: state + the anchoring
   message (a real FK into ``messagelog``, never a copy of its text) + when it
   last moved.  One row; v1 is a single active machine, concurrency deferred.
2. ``state_transition`` — HOW it got there: one row per move, carrying
   (from, to, cause, outcome, message, run, bound skill, when).  Indexed on
   ``created_at`` for the recency read and on ``cause``/``run_id`` for the
   per-edge scoring joins.

Schema only — touches no deployment-specific rows, and seeds no machine row: the
cold start is created lazily by ``MachineStore.current`` at the CALLER's default
state, so the machine's initial state is never a value frozen into DDL that a
later rename could strand.  A fresh ``create_tables()`` DB already carries both
tables from the models (the guards below skip them); a production copy that
predates them gets them added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _create_conversation_machine_table(conn)
    _create_state_transition_table(conn)
    conn.commit()


def _create_conversation_machine_table(conn: sqlite3.Connection) -> None:
    """Create the single-row machine-state table (idempotent)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS conversation_machine ("
        "  id INTEGER PRIMARY KEY,"
        "  state TEXT NOT NULL,"
        "  anchor_message_id INTEGER REFERENCES messagelog(id),"
        "  updated_at TIMESTAMP NOT NULL"
        ")"
    )


def _create_state_transition_table(conn: sqlite3.Connection) -> None:
    """Create the transition ledger + its recency / scoring indexes (idempotent)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state_transition ("
        "  id INTEGER PRIMARY KEY,"
        "  from_state TEXT NOT NULL,"
        "  to_state TEXT NOT NULL,"
        "  cause TEXT NOT NULL,"
        "  outcome TEXT,"
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
