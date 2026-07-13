"""Pydantic arg models for the skill tool surface (#1590).

``skill_create`` authors a skill by reference to the ledger; ``skill_read``
lists/renders the versionless skill registry.  Each tool validates its kwargs
through one of these as its first line, per the Pydantic-everywhere rule.
"""

from __future__ import annotations

from penny.tools.models import ToolArgs


class SkillCreateArgs(ToolArgs):
    """Args for ``skill_create(name, from_run, steps)``.

    ``name`` is the skill's human-readable title (the unique key — re-teaching the
    same name replaces the skill).  ``from_run`` is the run id of ONE verified
    demonstration (a single run — cross-run splicing is structurally impossible).
    ``steps`` is a contiguous ordinal range over that run's tool calls, written
    ``"2-5"`` (or a single ``"3"``); the range is parsed and bounds-checked by the
    tool so an out-of-range or malformed range gets an actionable error.
    """

    name: str
    from_run: str
    steps: str


class SkillReadArgs(ToolArgs):
    """Args for ``skill_read``.  ``name`` renders one skill's full recipe; omit it
    to list every skill (name + intent).  A blank name lists, too."""

    name: str | None = None
