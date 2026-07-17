"""The skill tool surface — ``skill_create`` (snapshot the preceding run) and
``skill_read`` (list / render), #1590.

``skill_create(name)`` is the ONLY write path into a skill, and it is **name-only**:
the model supplies just a title, and the system snapshots the run immediately
preceding this one (the routine the user just demonstrated in chat) — copying ALL
its non-``done`` tool calls out of the ledger, enforcing **certified-by-execution**
(every captured call succeeded in the source run) and factoring each argument by
provenance into declared holes.  No run id and no step range come from the model —
both are ledger coordinates it can't reliably produce mid-conversation; the
preceding run is resolved structurally by the executing agent's own run id.
Cross-run splicing is structurally impossible — one run in, one skill out.
``skill_read`` renders the versionless registry.
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
# loop-control call, never a skill step — excluded from the captured steps.
_DONE_TOOL = PennyConstants.DONE_TOOL_NAME

# The refusal when there's no preceding run to snapshot, or the preceding run had
# no runnable (non-``done``) step — actionable: run the routine once, then save it.
_NOTHING_TO_SAVE = (
    "Nothing to save yet — run the routine once here in chat first "
    "(browse, extract, write), then tell me to save it as a skill."
)


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
# read, not a narration parse.  The filter itself lives in ``_certified_steps``:
# failed calls are DROPPED from the recipe, not refused (#1659) — a routine only
# contains the calls that worked, and if none did there is nothing to save.


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
    """Snapshot the preceding run into a skill.

    A skill's step count is bounded by the shared step budget of the run that
    demonstrates it (``MAX_STEPS`` == ``BACKGROUND_MAX_STEPS`` by default —
    teaching happens in chat, so teachable == executable)."""

    name = "skill_create"
    description = (
        "Save what you JUST did as a reusable skill — a named recipe you can later "
        "instantiate as a collection. It snapshots the routine you ran in the run "
        "right before this one (browse, extract, write, …), copying those exact "
        "tool calls (never retyped) and figuring out which arguments are "
        "parameters.\n"
        "\n"
        "You pass ONLY a `name`. You do NOT pass a run id or a step range — the "
        "system captures the whole preceding run for you.\n"
        "\n"
        "Fields:\n"
        '- `name` — the skill\'s title (e.g. "Watch a page field"). Re-teaching '
        "the same name replaces it.\n"
        "\n"
        "Any call in that run that didn't succeed is left OUT of the recipe (a "
        "skill only contains calls that actually worked); if none of them succeeded, "
        "there's nothing to save. Each argument value becomes a fill-in-the-blank "
        "hole you supply when you reuse the skill, unless it came from an earlier "
        "step's result — then it renders as 'the value from step N'. Returns the "
        "learned skill so you can confirm it back."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill's title (unique; re-teach replaces)",
            },
        },
        "required": ["name"],
    }
    args_model = SkillCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = arguments.get("name")
        label = f' "{name}"' if name else ""
        if not result.success:
            return f"You tried to save the skill{label} but it didn't work:"
        return f"You saved the skill{label}:"

    def __init__(
        self, db: Database, llm_client: LlmClient, author: str, run_id: str | None = None
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author
        # The current run's id (this ``skill_create`` cycle, run B) — the anchor
        # for resolving the preceding run (run A) to snapshot.
        self._run_id = run_id

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = SkillCreateArgs(**kwargs)
        try:
            return await self._create_from_ledger(args)
        except SkillCreateError as exc:
            return ToolResult(message=str(exc), success=False)

    async def _create_from_ledger(self, args: SkillCreateArgs) -> ToolResult:
        """The whole authoring flow, reading like a table of contents: resolve the
        preceding run, load + project it, select ALL its non-``done`` steps, keep the
        ones that SUCCEEDED, distill and persist.  Every refusal is a
        ``SkillCreateError`` caught once above."""
        source_run = self._preceding_run_id()
        if source_run is None:
            raise SkillCreateError(_NOTHING_TO_SAVE)
        projection = project_run(self._db.messages.get_run_prompts(source_run))
        selected = self._select(projection)
        certified = self._certified_steps(selected)
        return await self._create(args.name, source_run, projection, certified)

    def _preceding_run_id(self) -> str | None:
        """The run immediately preceding this one for the executing agent — the
        demonstration to snapshot (run A, the routine just run before this
        ``skill_create`` cycle).  ``None`` when there's no current run id to anchor
        against, or no earlier run of this agent exists (an actionable refusal)."""
        if self._run_id is None:
            return None
        return self._db.messages.most_recent_run_id_before(self._run_id, self._author)

    def _select(self, projection: RunProjection) -> list[RunProjectionStep]:
        """EVERY non-``done`` step of the preceding run — the whole demonstration
        captured verbatim (no range selection; a ``done`` is loop control, never a
        skill step).  Raises the nothing-to-save refusal when the run had no
        runnable step."""
        chosen = [step for step in projection.steps if step.call.name != _DONE_TOOL]
        if not chosen:
            raise SkillCreateError(_NOTHING_TO_SAVE)
        return chosen

    def _certified_steps(self, selected: list[RunProjectionStep]) -> list[RunProjectionStep]:
        """Keep only the steps that SUCCEEDED in the source run, dropping any that
        failed — a failed exploratory call isn't part of the routine (#1659).

        Reads the STRUCTURAL per-call success stamp (``RunProjectionStep.success``,
        #1600) — a boolean the framework wrote at execution time from the tool's
        ``ToolResult.success``, not the framed result prose.  A step survives only
        when its stamp is exactly ``True``; a recorded failure (``False``) or a
        missing stamp (``None`` — a run logged before #1600, uncertain) is left out,
        so an uncertain call never sneaks into a skill (visible degradation over
        silent success).  Certified-by-execution then holds by construction — every
        STORED step succeeded.

        If NOTHING survives there's nothing to save (a skill is never empty): the
        demonstrated routine didn't actually work, so it refuses with the same
        actionable nothing-to-save guidance (re-run it cleanly, then save)."""
        certified = [step for step in selected if step.success is True]
        if not certified:
            raise SkillCreateError(_NOTHING_TO_SAVE)
        return certified

    async def _create(
        self,
        name: str,
        from_run: str,
        projection: RunProjection,
        certified: list[RunProjectionStep],
    ) -> ToolResult:
        """Distill the certified slice into a skill, embed its description, upsert
        it, and return the learned skill's full render.  ``source_ordinal`` keeps
        each step's ORIGINAL run ordinal; the skill-local ``ordinal`` (and any
        binding) renumbers against the surviving steps in :func:`distill_steps`."""
        inputs = [
            DistillInput(
                source_ordinal=step.ordinal,
                tool=step.call.name,
                arguments=step.call.arguments,
                result=projection.results.get(step.call_id, "") if step.call_id else "",
            )
            for step in certified
        ]
        steps, holes = distill_steps(inputs)
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
                "skill_create(name=<title>)."
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
