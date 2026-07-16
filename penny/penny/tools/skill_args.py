"""Pydantic arg models for the skill tool surface (#1590).

``skill_create`` authors a skill by reference to the ledger; ``skill_read``
lists/renders the versionless skill registry.  Each tool validates its kwargs
through one of these as its first line, per the Pydantic-everywhere rule.
"""

from __future__ import annotations

from penny.tools.models import ToolArgs


class SkillCreateArgs(ToolArgs):
    """Args for ``skill_create(name, steps, from_run=<optional>)``.

    ``name`` is the skill's human-readable title (the unique key — re-teaching the
    same name replaces the skill).  ``steps`` is a contiguous ordinal range over the
    demonstrated run's tool calls, written ``"2-5"`` (or a single ``"3"``); the range
    is parsed and bounds-checked by the tool so an out-of-range or malformed range
    gets an actionable error.  ``from_run`` is OPTIONAL (#1651): omitted, it resolves
    to the demonstration just performed — the most recent completed chat run before
    this one — so the overwhelmingly common "save that as a skill" case plumbs no run
    id (an unreachable required argument invited confabulation).  Pass an explicit id
    (still a single run — cross-run splicing is structurally impossible) only to
    promote an OLDER, non-adjacent run.
    """

    name: str
    steps: str
    from_run: str | None = None


class SkillReadArgs(ToolArgs):
    """Args for ``skill_read``.  ``name`` renders one skill's full recipe; omit it
    to list every skill (name + intent).  A blank name lists, too."""

    name: str | None = None
