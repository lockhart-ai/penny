"""The skill substrate's renders (#1590/#1804/#1943/#2305).

A skill is distilled deterministically from a qualifying chat run's own ledger at run
end (``penny.skill_extraction``), certified-by-execution with provenance-inferred
parameters.  The model's skill actions are resolve (``find``), read one by name
(``skill_get``, #2305), and instantiate/attach (``collection_configure(skill=…)``).  Both
``skill_get`` and a configured job's ``memory_get`` read the same ``parameters:``
line format in ``penny.tools.collection_instantiation`` (``render_skill_parameters``) —
``skill_get`` passes no bound values (the bare declaration), ``memory_get`` passes
the job's own.

Three renders live here, because the consumers want different things: ``render_skill_brief``
— what a skill IS and what it NEEDS, on one line — is what the surfaces that JUDGE a
skill read (the ambient ``### Known skills`` section every turn, and the run-end
narration frame); ``render_skill_shape`` — the same steps at SHAPE altitude, one
skimmable line — is what the learn-close narration adds beside the brief so the person
who knows what the skill was meant to do can see what it actually captured;
``render_skill_full`` — name, what it's for, its ``parameters:`` block, and its numbered
``steps:`` — is what an explicit read of one skill by name (``skill_get``) returns.
"""

from __future__ import annotations

from typing import Any

from penny.constants import PennyConstants
from penny.database.models import Skill
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import SkillParameter, SkillStep, render_skill
from penny.tools.collection_instantiation import render_skill_parameters

# ── The three renders: brief (what it is), shape (what it runs), full (how) ──

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


def _needs_clause(parameters: list[SkillParameter]) -> str:
    """The ``(parameters: <parameter> — <what to supply>; …)`` tail of the brief render:
    one entry per SKILL-level parameter, the semantic ``name`` being the binding key
    at instantiation and the description saying what to supply for it (omitted
    cleanly when None).  A parameter-less skill gets NO tail — the line renders
    byte-identically to one that never had parameters, rather than asserting an
    empty "parameters: none" the reader has to parse before ignoring.

    No required/optional marking here: every parameter ``distill_steps`` produces is
    required by construction, so "parameters" is the whole truth on this surface, and
    the ``required``/``optional`` precision stays in ``penny.tools.collection_instantiation``'s
    ``render_skill_parameters`` — read via ``skill_get`` on the bare skill, or a
    configured job's own ``memory_get`` — where it can be honest about a shape
    this codebase does not yet produce."""
    if not parameters:
        return ""
    needs = "; ".join(
        f"{parameter.name} — {parameter.description}" if parameter.description else parameter.name
        for parameter in parameters
    )
    return f" (parameters: {needs})"


def render_skill_brief(skill: Skill) -> str:
    """What a skill IS and what it NEEDS, on ONE line (#1804): ``<name> — <what it's
    for> (parameters: <parameter> — <what to supply>; …)``.

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
    the same steps the stored ``extraction_prompt`` numbers out in full — one altitude up,
    with no argument VALUES and no call shape at all, so there is nothing here for the
    reply to read aloud as a recipe (the #1799 leak) and the whole recipe stays one
    ``skill_get(skill=<name>)`` call away.

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
    """The whole skill as text, by name (#2305): its name, what it's for, the
    ``parameters:`` block (semantic names + descriptions, no bound values — this is
    the bare skill, not a configured job), and the numbered ``steps:`` (parameters
    shown as ``{name}``, display form == invocation form).

    ``skill_get`` returns this for the one skill it names — the read for when the
    name is already known and the parameters and steps are what's wanted."""
    parameters = render_skill_parameters(parameters_from_json(skill.parameters))
    recipe = render_skill(steps_from_json(skill.steps))
    lines = [
        f"skill '{skill.name}'",
        f"what it's for: {skill.intent}",
        parameters,
        "steps:",
        recipe,
    ]
    return "\n".join(lines)
