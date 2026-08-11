"""The round's framing on the transition log — ``state_transition.skill_frame`` (#1868).

Type: schema

Entering learn now FRAMES the round before the turn runs: a single-shot draw over the
user's own turns writes the routine's interface, and Python derives and builds the
container its results are kept in.  A draw varies and a re-draw is not a re-read, so that
decision has to be RECORDED — the turn's own instruction renders it, run-end extraction
reuses it instead of drawing again, and a correction re-entering learn compares against it
to decide whether the job's identity shifted.

It lives on the move that settled it, beside ``skill_name``, because it answers the same
kind of question about the transition — what routine this is about — for the case where
the routine does not exist yet.  One nullable TEXT column holding serialized
``RoundFraming`` JSON (the signature plus the container's name), carried with the anchor's
lifecycle: set on entry to learn, carried while the round stays parked, NULL once idle.

Schema only, universal, idempotent, and seeds nothing: every existing row keeps a NULL
framing, which is exactly the unframed round this deployment has been running.  A fresh
``create_tables()`` DB already carries the column from the model (the guard skips it); a
production copy that predates it gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_skill_frame_column(conn)
    conn.commit()


def _add_skill_frame_column(conn: sqlite3.Connection) -> None:
    """Add the round-framing column when the table exists without it (idempotent)."""
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "state_transition" not in tables:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(state_transition)")}
    if "skill_frame" in columns:
        return
    conn.execute("ALTER TABLE state_transition ADD COLUMN skill_frame TEXT")
