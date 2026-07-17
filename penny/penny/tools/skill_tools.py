"""The skill READ surface — ``skill_read`` (list / render), #1590.

Skills are no longer model-authored: there is no ``skill_create`` tool.  A skill is
distilled deterministically from a qualifying chat run's own ledger at run end
(``penny.skill_extraction``), certified-by-execution with provenance-inferred
parameters.
The model's only skill actions are resolve (``find``), READ (``skill_read``, here),
and instantiate/attach (``collection_create(skill=…)`` / ``collection_update(skill=…)``).
``skill_read`` renders the versionless registry; ``render_skill_full`` is the shared
whole-skill render (the read surface AND the run-end narration frame use it).
"""

from __future__ import annotations

from typing import Any

from penny.database import Database
from penny.database.models import Skill
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import SkillParameter, render_skill, slug_skill_name
from penny.tools.base import Tool
from penny.tools.models import ToolResult
from penny.tools.skill_args import SkillReadArgs

# ── Full render (shared by the read surface and the run-end narration frame) ───

_STEP_INDENT = "  "


def _parameters_block(parameters: list[SkillParameter]) -> str:
    """The ``parameters:`` block (#1668): one ``- <name> (required): <description>``
    line per SKILL-level parameter (the description omitted cleanly when None), the
    semantic ``name`` being the binding key at instantiation.  Collapses to a single
    ``parameters: none`` line for a parameter-less skill."""
    if not parameters:
        return "parameters: none"
    lines = ["parameters:"]
    for parameter in parameters:
        required = "required" if parameter.required else "optional"
        line = f"{_STEP_INDENT}- {parameter.name} ({required})"
        if parameter.description:
            line += f": {parameter.description}"
        lines.append(line)
    return "\n".join(lines)


def render_skill_full(skill: Skill) -> str:
    """The whole skill as text (#1668, the code owner's sketch) — its name, what it's
    for, the ``parameters:`` block (semantic names + descriptions), and the numbered
    recipe (parameters shown as ``{name}``, display form == invocation form).
    ``skill_read`` returns it for one skill, and the run-end narration frame (#1658)
    embeds it so the model narrates what it just learned FROM the render, not from
    memory; the ambient Skills section renders it wholesale."""
    steps = steps_from_json(skill.steps)
    parameters = parameters_from_json(skill.parameters)
    recipe = "\n".join(f"{_STEP_INDENT}{line}" for line in render_skill(steps).splitlines())
    lines = [
        f"skill '{skill.name}'",
        f"what it's for: {skill.intent}",
        _parameters_block(parameters),
        "steps:",
        recipe,
    ]
    return "\n".join(lines)


# ── skill_read ────────────────────────────────────────────────────────────────


class SkillReadTool(Tool):
    """List skills, or render one skill's full recipe."""

    name = "skill_read"
    description = (
        "Read your saved skills — reusable tool-call recipes. Pass `name` to see "
        "one skill's full recipe (its steps and fill-in-the-blank parameters); omit "
        "`name` to list every skill with what it's for."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill to render; omit to list all skills.",
            }
        },
        "required": [],
    }
    args_model = SkillReadArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = arguments.get("name")
        if not result.success:
            return "You tried to read your skills but it didn't work:"
        if name:
            return f'You looked up the "{name}" skill:'
        return "You listed your skills:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = SkillReadArgs(**kwargs)
        if args.name:
            return self._render_one(args.name)
        return self._list_all()

    def _render_one(self, name: str) -> ToolResult:
        skill = self._db.skills.get(name)
        if skill is None:
            return ToolResult(message=self._not_found_message(name), success=False)
        return ToolResult(message=render_skill_full(skill))

    def _list_all(self) -> ToolResult:
        skills = self._db.skills.list_all()
        if not skills:
            return ToolResult(
                message="No skills yet — teach one by demonstrating a flow here in "
                "chat, and I'll learn it automatically."
            )
        lines = [f"- {skill.name}: {skill.intent}" for skill in skills]
        return ToolResult(message="Your skills:\n" + "\n".join(lines))

    def _not_found_message(self, name: str) -> str:
        available = ", ".join(skill.name for skill in self._db.skills.list_all())
        listing = f" Your skills: {available}." if available else ""
        return (
            f"No skill named '{slug_skill_name(name)}'.{listing} "
            "List them with skill_read() (no name)."
        )
