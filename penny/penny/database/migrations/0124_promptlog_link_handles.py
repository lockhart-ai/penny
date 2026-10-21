"""Record the numbered link table a run rendered, on the run's last prompt row (#2717).

Type: schema

Links travel as numbered handles: every URL the model reads arrives as ``[n]``,
and Python expands the number when a message is sent or an entry is written, so
the model never types a URL.  The expansion means the model-facing record (the
promptlog's logged turns) carries handles, while the artefacts built from the
run — stored entries, the outgoing reply — carry real URLs.

The eval harness's provenance checks read what a run was GIVEN off the
model-facing turns.  Without the table, a real URL in a stored entry or a reply
traces to nothing: the run was given the page, as its handle.  So the run's
``HandleTable`` — number → URL, assigned on first sight — is stamped here as
JSON on the LAST prompt row for ``run_id`` at run close, by
``MessageStore.set_run_link_handles`` (called from ``Agent._run_agentic_loop``'s
finally, beside ``set_run_trailing_messages``).

The column is a run-record field the model never sees: it is read by the eval
harness (``given_to_the_model`` expands the given text through it) and by
nothing that renders into a prompt.  NULL on every row logged before this
migration and on runs whose surfaces rendered no handles.

A fresh ``create_tables()`` DB already carries the column from the model (the
guard below is a no-op there); a production copy that predates this migration
gets it added.
"""

from __future__ import annotations

import sqlite3


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    if "link_handles" not in columns:
        conn.execute("ALTER TABLE promptlog ADD COLUMN link_handles TEXT")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
