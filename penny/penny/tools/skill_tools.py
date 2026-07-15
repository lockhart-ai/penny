"""The skill tool surface — ``skill_create`` (author by reference) and
``skill_read`` (list / render), #1590.

``skill_create(name, from_run, steps=<range>)`` is the ONLY write path into a
skill: the model points at a contiguous range of ONE verified run's tool-call
ordinals and the system copies those calls out of the ledger, enforcing
**certified-by-execution** (every selected call succeeded in the source run) and
factoring each argument by provenance into declared holes.  Cross-run splicing is
structurally impossible — the single ``from_run`` argument is the whole selection
scope.  ``skill_read`` renders the versionless registry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import RunProjection, RunProjectionStep, project_run
from penny.database.models import Skill
from penny.database.skill_store import holes_from_json, steps_from_json
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillHole,
    distill_steps,
    render_skill,
    slug_skill_name,
)
from penny.llm.similarity import embed_text
from penny.tools.base import Tool
from penny.tools.models import ToolResult
from penny.tools.skill_args import SkillCreateArgs, SkillReadArgs

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# ``done`` consumes a tool-call ordinal (matching ``render_run_calls``) but is a
# loop-control call, never a skill step — excluded from every selected range.
_DONE_TOOL = PennyConstants.DONE_TOOL_NAME


class SkillCreateError(Exception):
    """An actionable ``skill_create`` refusal — ``str(self)`` is the model-readable
    message, returned verbatim as the failed ``ToolResult``.

    The ``MemoryAccessError`` pattern: the selection/certification helpers raise,
    ``execute`` catches once — no string-typed sentinel returns in the success
    channel."""


# ── Certified-by-execution: read each selected step's structural success stamp ─
#
# The ledger now persists a per-call success bit beside each tool-result frame
# (#1600 — ``RunProjectionStep.success``, hydrated from the ``tool_success`` stamp
# the framework wrote at execution time), so "did this call succeed?" is a boolean
# read, not a narration parse.  The gate itself lives in ``_require_certified``.


# ── Range parsing (``"2-5"`` / ``"2..5"`` / ``"3"``) ──────────────────────────

_RANGE_HELP = 'write it as a range like "2-5" (steps 2 through 5) or a single step like "3".'


def _parse_range(raw: str) -> tuple[int, int]:
    """Parse a ``steps`` argument into ``(start, end)``; raises an actionable
    ``SkillCreateError`` on a malformed range.  Accepts ``"N"``, ``"N-M"``,
    and ``"N..M"``."""
    text = raw.strip().replace("..", "-")
    parts = text.split("-")
    try:
        bounds = [int(part.strip()) for part in parts]
    except ValueError:
        raise SkillCreateError(f"Couldn't read steps={raw!r} — {_RANGE_HELP}") from None
    if len(bounds) == 1:
        start = end = bounds[0]
    elif len(bounds) == 2:
        start, end = bounds
    else:
        raise SkillCreateError(f"Couldn't read steps={raw!r} — {_RANGE_HELP}")
    if start < 1 or end < start:
        raise SkillCreateError(
            f"steps={raw!r} isn't a valid range (start≥1, end≥start) — {_RANGE_HELP}"
        )
    return start, end


# ── Full render (shared by the create result and the read surface) ────────────


def _holes_line(holes: list[SkillHole]) -> str:
    if not holes:
        return "holes: none"
    rendered = ", ".join(
        f"{hole.name} ({'required' if hole.required else 'optional'})" for hole in holes
    )
    return f"holes: {rendered}"


def _render_skill_full(skill: Skill) -> str:
    """The whole skill as text — its name, intent, declared holes, and the numbered
    recipe (holes shown as ``{name}``).  ``skill_create`` returns this so the user
    sees exactly what was learned; ``skill_read`` returns it for one skill."""
    steps = steps_from_json(skill.steps)
    holes = holes_from_json(skill.holes)
    lines = [
        f"skill '{skill.name}'",
        f"intent: {skill.intent}",
        _holes_line(holes),
        "steps:",
        render_skill(steps),
    ]
    return "\n".join(lines)


# ── skill_create ──────────────────────────────────────────────────────────────


class SkillCreateTool(Tool):
    """Author a skill by reference to one verified run's ledger.

    A skill's step count is bounded by the shared step budget of the run that
    demonstrates it (``MAX_STEPS`` == ``BACKGROUND_MAX_STEPS`` by default —
    teaching happens in chat, so teachable == executable)."""

    name = "skill_create"
    description = (
        "Save a verified run's tool-call sequence as a reusable skill — a named "
        "recipe you can later instantiate as a collection.  You point at ONE run "
        "you already ran cleanly (its id, from `read_run_calls`) and a contiguous "
        "range of its steps; the system copies those exact calls (never retyped) "
        "and figures out which arguments are parameters.\n"
        "\n"
        "Fields:\n"
        '- `name` — the skill\'s title (e.g. "Watch a page field"). Re-teaching '
        "the same name replaces it.\n"
        "- `from_run` — the run id of the clean demonstration. A skill comes from "
        "ONE run; you can't splice steps from several.\n"
        '- `steps` — the step range to keep, like "2-5" (steps 2 through 5) or a '
        'single "3". Trim off any incidental lookups at the start or end.\n'
        "\n"
        "Every selected step must have SUCCEEDED in that run (a skill only contains "
        "calls that actually worked). Arguments you took from the user's request "
        "become fill-in-the-blank holes; a value that came from an earlier step "
        "becomes 'the value from step N'; anything else is baked in. Returns the "
        "learned skill so you can confirm it back."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill's title (unique; re-teach replaces)",
            },
            "from_run": {
                "type": "string",
                "description": "The run id of the clean demonstration (from read_run_calls)",
            },
            "steps": {
                "type": "string",
                "description": 'The step range to keep, e.g. "2-5" or a single "3"',
            },
        },
        "required": ["name", "from_run", "steps"],
    }
    args_model = SkillCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = arguments.get("name")
        label = f' "{name}"' if name else ""
        if not result.success:
            return f"You tried to save the skill{label} but it didn't work:"
        return f"You saved the skill{label}:"

    def __init__(self, db: Database, llm_client: LlmClient, author: str) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = SkillCreateArgs(**kwargs)
        try:
            return await self._create_from_ledger(args)
        except SkillCreateError as exc:
            return ToolResult(message=str(exc), success=False)

    async def _create_from_ledger(self, args: SkillCreateArgs) -> ToolResult:
        """The whole authoring flow, reading like a table of contents: parse the
        range, load + project the run, select the slice, certify it, distill and
        persist.  Every refusal is a ``SkillCreateError`` caught once above."""
        start, end = _parse_range(args.steps)
        prompts = self._db.messages.get_run_prompts(args.from_run)
        if not prompts:
            raise SkillCreateError(self._no_run_message(args.from_run))
        projection = project_run(prompts)
        selected = self._select(projection, start, end)
        self._require_certified(selected, args.from_run)
        return await self._create(args.name, args.from_run, projection, selected)

    def _select(self, projection: RunProjection, start: int, end: int) -> list[RunProjectionStep]:
        """The run's non-``done`` steps whose ordinal is in ``[start, end]`` — a
        contiguous slice of surviving ordinals (a ``done`` in the range is a gap,
        never renumbered).  Raises when the range names no runnable step."""
        chosen = [
            step
            for step in projection.steps
            if start <= step.ordinal <= end and step.call.name != _DONE_TOOL
        ]
        if chosen:
            return chosen
        available = ", ".join(
            str(step.ordinal) for step in projection.steps if step.call.name != _DONE_TOOL
        )
        raise SkillCreateError(
            f"No tool-call steps in range {start}-{end} for this run. Its steps are: "
            f"{available or '(none)'}. Pick a range that covers the steps you want."
        )

    def _require_certified(self, selected: list[RunProjectionStep], from_run: str) -> None:
        """The certified-by-execution gate: raises naming the first selected step
        whose call did NOT succeed in the source run (a skill only contains calls
        that worked).

        Reads the STRUCTURAL per-call success stamp (``RunProjectionStep.success``,
        #1600) — a boolean the framework wrote at execution time from the tool's
        ``ToolResult.success``, not the framed result prose.  A step certifies only
        when its stamp is exactly ``True``; a recorded failure (``False``) or a
        missing stamp (``None`` — a run logged before #1600 carries none) refuses, so
        an uncertain call never optimistically passes (refuse-to-certify-uncertain:
        visible degradation over silent success).

        The invariant holds universally: ``skill_create`` is the ONLY write path
        into a skill (there is no seed library — migration 0084 ships the table
        empty), so every stored step passed this gate."""
        for step in selected:
            if step.success is not True:
                raise SkillCreateError(
                    f"Can't save this skill: step {step.ordinal} "
                    f"({step.call.name}) didn't succeed in run {from_run}, and a skill "
                    "may only contain calls that worked. Re-demonstrate the flow so "
                    "every step succeeds, then save that run's range."
                )

    async def _create(
        self,
        name: str,
        from_run: str,
        projection: RunProjection,
        selected: list[RunProjectionStep],
    ) -> ToolResult:
        """Distill the certified slice into a skill, embed its description, upsert
        it, and return the learned skill's full render."""
        inputs = [
            DistillInput(
                source_ordinal=step.ordinal,
                tool=step.call.name,
                arguments=step.call.arguments,
                result=projection.results.get(step.call_id, "") if step.call_id else "",
            )
            for step in selected
        ]
        steps, holes = distill_steps(inputs, projection.origin_message)
        description = projection.origin_message or f"Skill: {name}"
        draft = SkillDraft(
            name=name,
            intent=description,
            description=description,
            steps=steps,
            holes=holes,
            source_run_id=from_run,
        )
        embedding = await embed_text(self._llm, description)
        skill, replaced = self._db.skills.upsert(
            draft, author=self._author, description_embedding=embedding
        )
        lead = (
            f"Replaced the previous version of '{skill.name}'."
            if replaced
            else f"Learned skill '{skill.name}'."
        )
        return ToolResult(message=f"{lead}\n{_render_skill_full(skill)}", mutated=True)

    @staticmethod
    def _no_run_message(from_run: str) -> str:
        return (
            f"No run found with id {from_run!r}. Use read_run_calls(target='chat') "
            "(or a collector's name) to find the id of the run you want to save, then "
            "pass that exact id as from_run."
        )


# ── skill_read ────────────────────────────────────────────────────────────────


class SkillReadTool(Tool):
    """List skills, or render one skill's full recipe."""

    name = "skill_read"
    description = (
        "Read your saved skills — reusable tool-call recipes. Pass `name` to see "
        "one skill's full recipe (its steps and fill-in-the-blank holes); omit "
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
        return ToolResult(message=_render_skill_full(skill))

    def _list_all(self) -> ToolResult:
        skills = self._db.skills.list_all()
        if not skills:
            return ToolResult(
                message="No skills yet — teach one by demonstrating a flow, then "
                "skill_create(name=<title>, from_run=<run id>, steps=<range>)."
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
