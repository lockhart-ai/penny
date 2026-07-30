"""Carry the attachment mark onto every stored skill's scoped-write target (#1783).

Type: data

#1783 removed the tool whitelist from the instantiation path.  Binding is no longer
"every ``collection_write``/``update_entry``/``collection_delete_entry`` step's
``memory`` argument" recomputed at the render seam — it is "every leaf still carrying
the ATTACHMENT MARK", a per-leaf fact distillation records and the labeller either
clears (the user chose the value) or lets stand (the assistant did).  A skill is a
learned sequence of ARBITRARY tool calls, so nothing downstream may key on which tool a
leaf sits in; reading a mark off the skill is what makes that possible.

Skills distilled from now on carry the mark.  A skill taught BEFORE this one does not,
so ``retarget_writes`` would find nothing to bind and its rendered ``extraction_prompt``
would show ``{the collection this is set up on}`` where the collection's own name
belongs — the routine would no longer state where it acts.  This sets the mark on those
rows so an apply lands on the target regardless of when the skill was taught.

UNIVERSAL: a generic, content-shape rewrite over every row of the ``skill`` table (the
0096/0101 precedent — "a scoped-write step's ``memory`` leaf"), never a
deployment-specific skill by name.  Non-destructive: only the leaf's mark changes; the
demonstrated name stays in ``arguments`` as the verbatim ledger copy and the
substitution keeps its kind and description.  Idempotent: a leaf already marked is left
alone, and a leaf with no substitution at all gets the same placeholder 0101 gives it
(so a row 0101 could not reach — one inserted between the two migrations — is covered
here rather than silently losing its binding).

The tool set and the description are FROZEN COPIES of what ``penny.database.skills``
held when this ran — a migration is a historical artifact and must not change behaviour
when a live constant is later edited.  The live path names no tools at all; this
enumeration exists only to describe skills taught while it did.
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
        changed = [_mark_target(step) for step in steps]
        if not any(changed):
            continue
        conn.execute("UPDATE skill SET steps = ? WHERE name = ?", (json.dumps(steps), name))
    conn.commit()


def _mark_target(step: dict) -> bool:
    """Mark one step's write-target leaf as attachment-filled, reporting whether it
    changed.  Only a scoped-write step whose ``memory`` argument is a plain string
    qualifies; the leaf's existing substitution is marked in place, and a leaf with none
    gets the 0101 placeholder already marked."""
    if step.get("tool") not in _SCOPED_WRITE_TOOLS:
        return False
    if not isinstance(step.get("arguments", {}).get("memory"), str):
        return False
    substitutions = step.setdefault("substitutions", [])
    existing = [sub for sub in substitutions if sub.get("path") == _MEMORY_PATH]
    if not existing:
        substitutions.append(_marked_placeholder())
        return True
    return any(_set_mark(sub) for sub in existing)


def _set_mark(sub: dict) -> bool:
    """Set the attachment mark on one substitution, reporting whether it changed."""
    if sub.get("attachment") is True:
        return False
    sub["attachment"] = True
    return True


def _marked_placeholder() -> dict:
    """The write-target placeholder 0101 adds, already carrying the mark — for a skill
    row that reached this migration without one."""
    return {
        "path": list(_MEMORY_PATH),
        "kind": "placeholder",
        "parameter": None,
        "step": None,
        "description": _WRITE_TARGET_DESCRIPTION,
        "attachment": True,
    }
