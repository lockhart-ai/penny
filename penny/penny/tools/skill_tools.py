"""The skill READ surface — ``skill_read`` (list / render), #1590.

Skills are no longer model-authored: there is no ``skill_create`` tool.  A skill is
distilled deterministically from a qualifying chat run's own ledger at run end
(``penny.skill_extraction``), certified-by-execution with provenance-inferred
parameters.
The model's only skill actions are resolve (``find``), READ (``skill_read``, here),
and instantiate/attach (``collection_set(skill=…)`` / ``collection_set(skill=…)``).
``skill_read`` renders the versionless registry.  Three renders live here, because the
consumers want different things (#1804/#1943): ``render_skill_full`` — name, intent,
parameters AND the numbered tool-call recipe — is what an explicit read of ONE skill
returns, the one place the steps are the answer; ``render_skill_brief`` — what a skill
IS and what it NEEDS, on one line — is what the surfaces that JUDGE a skill read (the
ambient ``### Skills and rules`` section every turn, and the run-end narration frame);
``render_skill_shape`` — the same steps at SHAPE altitude, one skimmable line — is what
the learn-close narration adds beside the brief so the person who knows what the routine
was meant to do can see what it actually captured.
"""

from __future__ import annotations

from typing import Any

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import Skill
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import SkillParameter, SkillStep, render_skill, slug_skill_name
from penny.tools.base import Tool
from penny.tools.models import ToolResult
from penny.tools.skill_args import SkillReadArgs

# ── The three renders: brief (what it is), shape (what it runs), full (how) ───

_STEP_INDENT = "  "

# The step-shape line's own punctuation (#1943).  The arrow is the sequence, the overflow
# tail is the honest cut — it is deliberately NOT joined by the arrow, so the count can
# never be misread as one more step, and it states its number the way every other bounded
# render in the house does.
_SHAPE_SEPARATOR = " → "
_SHAPE_OVERFLOW = " … {count} more steps not shown."
# A step's tool is named in the house's inline-prose dialect — single markdown backticks,
# no parens.  Both halves are deliberate.  The backticks are what the canonical-call-
# notation rules permit for a tool named in prose rather than called; and NO parens is the
# point of this render, because a call shape here would be a call shape the reply reads
# aloud (#1799, the measured leak this frame exists downstream of).  A parens-less mention
# reads as "a thing that exists" — which is exactly what a step of a RECORD is.
_SHAPE_TOOL = "`{tool}`"


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


def _needs_clause(parameters: list[SkillParameter]) -> str:
    """The ``(needs: <parameter> — <what to supply>; …)`` tail of the brief render:
    one entry per SKILL-level parameter, the semantic ``name`` being the binding key
    at instantiation and the description saying what to supply for it (omitted
    cleanly when None).  A parameter-less skill gets NO tail — the line renders
    byte-identically to one that never had parameters, rather than asserting an
    empty "needs: none" the reader has to parse before ignoring.

    No required/optional marking here: every parameter ``distill_steps`` produces is
    required by construction, so "needs" is the whole truth on this surface, and the
    ``required``/``optional`` precision stays in ``render_skill_full`` — one
    ``skill_read(name=<name>)`` away — where it can be honest about a shape this
    codebase does not yet produce."""
    if not parameters:
        return ""
    needs = "; ".join(
        f"{parameter.name} — {parameter.description}" if parameter.description else parameter.name
        for parameter in parameters
    )
    return f" (needs: {needs})"


def render_skill_brief(skill: Skill) -> str:
    """What a skill IS and what it NEEDS, on ONE line (#1804): ``<name> — <what it's
    for> (needs: <parameter> — <what to supply>; …)``.

    This is what a surface JUDGING a skill wants — enough to decide whether the skill
    covers the ask and to bind its parameters — and it is deliberately all a judging
    surface gets.  The numbered recipe is how the routine is CARRIED OUT, which no
    chat turn does any more (the collector runs it, from the ``extraction_prompt``
    rendered at instantiation), so on a surface read every turn it is context cost
    for a decision it does not inform.

    One line whatever the skill is: eight steps across four plugin tools render the
    same as two, because steps do not render at all here, and a routine that needs
    nothing simply carries no tail.  (Deliberately the same shape as the state
    classifier's ``SkillCandidate.render`` — the two surfaces judge coverage from the
    same facts in the same words, so they cannot disagree from having been shown
    different evidence.  Not single-sourced: ``conversation_machine`` is a leaf that
    must not import the database package this module pulls in.)"""
    return f"{skill.name} — {skill.intent}{_needs_clause(parameters_from_json(skill.parameters))}"


def _counted_arguments(arguments: dict[str, Any]) -> str:
    """The ``<n> <argument name>`` clauses of one step's shape — one per top-level
    argument holding a LIST of more than one thing, in the order the call declared them.

    The noun is the ARGUMENT'S OWN NAME, read off the call as data: nothing here knows
    which tools exist, so a plugin verb nobody has heard of counts its own list under its
    own word for it.  A list of ONE renders nothing at all — one is what a call ordinarily
    carries, so a count of it says nothing the tool name did not, and the line stays short
    enough to skim.  A scalar or a dict argument is ONE thing however many fields it has,
    so neither is counted; what this line is for is the step that did something several
    times over.

    The lengths are read from the STORED arguments, which is what makes the line stable:
    every seam that touches a skill's arguments afterwards (a parameter binding, an
    attachment retarget, the render's own substitutions) replaces a LEAF in place and
    never changes a list's length, so the shape of a routine at rest and the shape of the
    job it becomes are the same shape."""
    return ", ".join(
        f"{len(value)} {name}"
        for name, value in arguments.items()
        if isinstance(value, list) and len(value) > 1
    )


def _step_shape(step: SkillStep) -> str:
    """One step at shape altitude: its tool, and how many of anything it handles."""
    named = _SHAPE_TOOL.format(tool=step.tool)
    counted = _counted_arguments(step.arguments)
    return f"{named} {counted}" if counted else named


def render_skill_shape(skill: Skill) -> str:
    """What the routine RUNS, in order, on ONE skimmable line (#1943) — for the round
    that motivated it, a fetch of three pages, a write of four entries, and then two
    stray fetches, each tool named in the inline-prose backtick dialect.

    The learn-close reply names the routine and what it needs, and until this it said
    nothing about its SHAPE — so a step the demonstration picked up by accident was
    invisible until a collector enacted it (the motivating round carried two stray
    fetches and a placeholder write, all three enacted verbatim afterwards).  The
    demonstrator is the only party who knows what the routine was FOR, and the learn-close
    reply is the one moment they are guaranteed to be present, so this is the review
    moment: the line is read, not judged, and correcting it is re-teaching the routine —
    there is no edit surface and this adds none.

    Read off the RECORD (the distilled steps), never off what the run remembers doing, for
    the reason the brief render beside it is: the two must not be able to disagree.  It is
    the same steps :func:`render_skill_full` numbers out in full — one altitude up, with
    no argument VALUES and no call shape at all, so there is nothing here for the reply to
    read aloud as a recipe (the #1799 leak) and the whole recipe stays one
    ``skill_read(name=<name>)`` away.

    No tool is named in LOGIC: each segment is whatever the step's own call was, so a
    routine spanning tools this module has never heard of renders exactly like one that
    does not.  A demonstration longer than ``SKILL_SHAPE_STEPS`` states the count it left
    off rather than trailing away.  There is no empty form: a skill reaches the registry
    only from a run that certified at least one call, so a step-less skill is not a state
    this render can be handed."""
    steps = steps_from_json(skill.steps)
    shown = steps[: PennyConstants.SKILL_SHAPE_STEPS]
    line = _SHAPE_SEPARATOR.join(_step_shape(step) for step in shown)
    remaining = len(steps) - len(shown)
    if not remaining:
        return line
    return f"{line}{_SHAPE_OVERFLOW.format(count=remaining)}"


def render_skill_full(skill: Skill) -> str:
    """The whole skill as text (#1668, the code owner's sketch) — its name, what it's
    for, the ``parameters:`` block (semantic names + descriptions), and the numbered
    recipe (parameters shown as ``{name}``, display form == invocation form).

    ``skill_read`` returns it for ONE skill, and that is now its only consumer
    (#1804): an explicit read of a single skill is the one place the steps are what
    was asked for.  The surfaces that merely judge a skill read
    :func:`render_skill_brief` instead."""
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
