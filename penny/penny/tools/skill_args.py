"""Pydantic arg models for the skill READ surface (#1590).

There is no ``skill_create`` tool — skills are distilled automatically at chat-run
end (#1658).  ``skill_read`` lists/renders the versionless skill registry; it
validates its kwargs through this model as its first line, per Pydantic-everywhere.
"""

from __future__ import annotations

from penny.tools.models import ToolArgs


class SkillReadArgs(ToolArgs):
    """Args for ``skill_read``.  ``name`` renders one skill's full recipe; omit it
    to list every skill (name + intent).  A blank name lists, too."""

    name: str | None = None
