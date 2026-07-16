"""Pydantic arg models for the skill tool surface (#1590).

``skill_create`` authors a skill by reference to the ledger; ``skill_read``
lists/renders the versionless skill registry.  Each tool validates its kwargs
through one of these as its first line, per the Pydantic-everywhere rule.
"""

from __future__ import annotations

from penny.tools.models import ToolArgs


class SkillCreateArgs(ToolArgs):
    """Args for ``skill_create(name)``.

    ``name`` is the skill's human-readable title (the unique key — re-teaching the
    same name replaces the skill).  It is the ONLY argument: the tool captures the
    whole run immediately preceding this one (the demonstration you just ran here in
    chat), so the model supplies neither a run id nor a step range — both are ledger
    coordinates it can't reliably produce mid-conversation.
    """

    name: str


class SkillReadArgs(ToolArgs):
    """Args for ``skill_read``.  ``name`` renders one skill's full recipe; omit it
    to list every skill (name + intent).  A blank name lists, too."""

    name: str | None = None
