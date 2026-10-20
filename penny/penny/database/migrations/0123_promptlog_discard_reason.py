"""Record why a completed model call's output was not used (#2702).

Type: schema

``LlmClient.chat`` writes one ``promptlog`` row per actual call, stamped with
``call_outcome`` — so a call that never finished is already distinguishable.
But a call that finished and whose output was then thrown away by the CALLER
reads as ``call_outcome="completed"`` exactly like a call whose output was
used: a degenerate draw the loop discarded and re-rolled, a draw a validator
returned ``Retry`` for, a final answer that came back empty, a micro-context
draw the poison screen rejected, a draw cut by its own token budget, a
structured draw whose JSON failed its Pydantic parse.  Several such rows in a
row look like normal responses when most were thrown away — the addon's
Prompts tab could not tell them apart from the call that counted.

Two nullable columns, added once:

- ``discard_reason`` — the machine reason the code knew at the rejection
  point: a ``ConditionKey`` value (``penny.validation.conditions``), persisted
  verbatim.  NULL on every used call and every pre-existing row.
- ``discard_detail`` — optional short detail string.  NULL everywhere today;
  the column exists so a site can say more without a further migration.

Set after the fact by ``MessageStore.mark_prompt_discarded`` from each
rejection site, via the row id ``log_prompt`` returns.  Additive and never
model-facing: only the addon serializers and the live push read these
(``run_get`` / the self-state renders are untouched).

A fresh ``create_tables()`` DB already carries these columns from the model
(the guard below is a no-op there); a production copy that predates this
migration gets them added.
"""

from __future__ import annotations

import sqlite3

_NEW_COLUMNS = ("discard_reason", "discard_detail")


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "promptlog"):
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    for column in _NEW_COLUMNS:
        if column not in columns:
            conn.execute(f"ALTER TABLE promptlog ADD COLUMN {column} TEXT")
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
