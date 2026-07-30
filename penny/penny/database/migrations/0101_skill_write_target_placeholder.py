"""Give every stored skill's scoped-write target its placeholder substitution (#1777).

Type: data

A skill must hardcode nothing from the round that taught it.  Until #1777 the
scoped-write TARGET was exempt: ``distill_steps`` skipped it, so the leaf carried no
substitution and the recipe rendered the demonstrated collection's name verbatim
(``collection_write(memory='<the demo's collection>', …)``).  That literal is what the
0-call ambient firing surface shows — no instantiation runs there, so nothing
retargets it — and it read as a promise about where the routine writes.

Skills distilled from now on record a ``placeholder`` substitution for that leaf, but
a skill taught BEFORE this migration still carries the bare literal, so its recipe
would keep leaking.  This adds the missing substitution to those rows.

UNIVERSAL: a generic, content-shape rewrite over every row of the ``skill`` table (the
0096 precedent — "a scoped-write step whose ``memory`` leaf has no substitution"),
never a deployment-specific skill by name.  Non-destructive: the demonstrated name
stays in the step's ``arguments`` as the verbatim ledger copy, exactly as every other
placeholder leaf keeps its value — only the PRESENTATION changes.  Idempotent: a step
that already has a substitution addressing ``memory`` (a post-#1777 skill, or the stray
parameter/binding ``retarget_writes`` has always tolerated) is left alone.

The tool set and the description are FROZEN COPIES of ``penny.database.skills``
(``SCOPED_WRITE_TOOLS`` / ``WRITE_TARGET_DESCRIPTION``) — a migration is a historical
artifact and must not change behaviour when a live constant is later edited.
"""

from __future__ import annotations

import json
import sqlite3

_SCOPED_WRITE_TOOLS = frozenset({"collection_write", "update_entry", "collection_delete_entry"})
_WRITE_TARGET_DESCRIPTION = "the collection this is set up on"
_MEMORY_PATH = ["memory"]


def up(conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "skill" not in tables:
        return
    for name, steps_json in conn.execute("SELECT name, steps FROM skill").fetchall():
        steps = json.loads(steps_json)
        # A list, not a short-circuiting ``any(...)`` generator: EVERY write step of the
        # skill must be visited, not just up to the first one that changed.
        changed = [_add_placeholder(step) for step in steps]
        if not any(changed):
            continue
        conn.execute("UPDATE skill SET steps = ? WHERE name = ?", (json.dumps(steps), name))
    conn.commit()


def _add_placeholder(step: dict) -> bool:
    """Append the write-target placeholder to one step, reporting whether it changed.
    Only a scoped-write step whose ``memory`` argument is a plain string and carries no
    substitution yet qualifies."""
    if step.get("tool") not in _SCOPED_WRITE_TOOLS:
        return False
    if not isinstance(step.get("arguments", {}).get("memory"), str):
        return False
    substitutions = step.setdefault("substitutions", [])
    if any(sub.get("path") == _MEMORY_PATH for sub in substitutions):
        return False
    substitutions.append(
        {
            "path": list(_MEMORY_PATH),
            "kind": "placeholder",
            "parameter": None,
            "step": None,
            "description": _WRITE_TARGET_DESCRIPTION,
        }
    )
    return True
