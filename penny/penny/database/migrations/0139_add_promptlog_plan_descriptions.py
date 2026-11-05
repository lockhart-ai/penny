"""Add ``promptlog.plan_descriptions`` — a SANDBOX run's per-call plan wording (#2995).

Type: schema

In a sandbox run (#2964) every ``mutates`` call is forced onto its plan path, and
(#2995) the result the job sees reads like the ordinary result for that call — the run
must look live to the job.  The PLAN wording ("PLAN — nothing was changed: this call
would …") then exists nowhere on the run record, yet the sandbox report is supposed to
show every call and what it WOULD have done.

This column holds that: a JSON map of ``{"<tool_call_id>": "<plan wording>"}``,
stamped on the run's last prompt row by ``set_run_plan_descriptions`` from the run's
own tool records at settlement; ``build_sandbox_report`` reads it to render each call
with its plan wording.  NULL for every non-sandbox run (a chat ``plan=true`` call
shows the wording to its own caller, so it needs no stamp), and for every row that
isn't its run's last — so every pre-existing run renders byte-identically.

One nullable column. Schema only, universal — a fresh ``create_tables()`` DB already
carries it from the model (the guard skips it); a production copy that predates it
gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    _add_promptlog_plan_descriptions(conn)
    conn.commit()


def _add_promptlog_plan_descriptions(conn: sqlite3.Connection) -> None:
    """Add the nullable ``plan_descriptions`` column to ``promptlog`` (idempotent)."""
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "plan_descriptions" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN plan_descriptions TEXT")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
