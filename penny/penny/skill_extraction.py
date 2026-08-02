"""Automatic skill extraction at chat-run end (#1658, epic #1554).

Skills are no longer model-authored.  There is no ``skill_create`` tool: at the
end of every qualifying CHAT run the framework distils a skill *deterministically*
from that run's own ledger rows — the same certified-by-execution snapshot the
retired tool produced, now fired by the run finishing instead of a model call.

``SkillExtractor.extract(run_id)`` is the whole pipeline, composed of named steps
(house style: the summary method reads like a table of contents):

* **qualify** — all structural, each a named check: the run is the chat agent's,
  it made ≥1 tool call, no text-bail nudge poisoned it, and its SUCCEEDED,
  COLLECTOR-runnable calls form a read+write taxonomy (a routine that senses AND
  acts).  A purely-read run (answering a question) and a purely-write run ('remember
  this' — the storage atom) do NOT qualify; failed calls are FILTERED, so a run whose
  only write failed is a pure read and is excluded.  Lifecycle calls a demo made
  (e.g. ``collection_set`` to set up the container) are dropped like orientation
  calls — a skill renders into a collector prompt, so only collector-runnable steps
  belong in it, and they count for nothing in the taxonomy (#1668).
* **distill** — ``distill_steps`` over the surviving (certified, non-``done``)
  steps: strips the framework ``reasoning`` leaf and classifies bindings vs. candidate
  parameters (#1659/#1660/#1662) — EVERY unexplained leaf, whatever tool it sits on
  and whatever argument it fills (#1783; no write target, no privileged argument, no
  tool whitelist).  A leaf whose demonstrated value named one of Penny's own
  COLLECTIONS additionally carries the attachment mark, so the collection the routine
  is attached to can fill it (see ``SkillSubstitution.attachment``).
* **name + adjudicate** — a GENERIC verb-noun label + a one-line generic description,
  written by a single-shot naming micro-context (#1665, the SECOND customer of the
  micro-context machinery) over the distilled routine — so a skill is named by its
  CONTRACT ("look up a price on a listing page and record it"), not by the instance
  ("read-the-aurora-deck-2-listing"), and cross-instance ``find`` can match it.  The
  same draw decides, per CANDIDATE parameter, whether the USER supplied that value
  (#1770): one they did is a real parameter (semantic name + description); one the
  assistant derived from a result or invented is a **placeholder** — dropped from the
  parameter list and rendered as what belongs there, never as the frozen demonstrated
  value.  On ANY naming failure the fallback is the deterministic slug of the
  triggering message (URLs removed, ≤6 words) + that message as the description, and
  every candidate keeps its arg-derived required parameter — extraction NEVER blocks
  on the rewrite, and a missing verdict never deletes a parameter.
* **shape** — a SECOND single-shot micro-context (#1803, the FOURTH customer) then
  decides what the routine IS: its name and which of the kept values it is ABOUT
  rather than pointed at.  A value the routine is about becomes a CONSTANT — baked
  into the step, never asked for again — so a skill can no longer name itself for a
  value and then demand it (`record-product-price` requiring a `what_to_extract` its
  own name already gave, which stopped the routine firing from the natural second
  ask).  It is a separate draw because provenance and role are different questions
  answered from different evidence, and folding them into one collapsed the
  provenance binary that already worked.  Name and constants come out TOGETHER, which
  is what makes them unable to contradict.  Its failure costs nothing: every value
  stays a bindable parameter under the labeller's name.
* **dedup (REPLACE semantics)** — exact name match → REPLACE; else a same-shape,
  same-meaning skill (the GENERIC ``description_embedding`` converges cross-instance)
  → REPLACE keeping ITS name; otherwise insert.

Every outcome is TYPED and loggable — the extracted/replaced skill, or a
no-extraction outcome naming which gate failed — never a silent ``None`` (visible
degradation over silent success).  The module reads ``promptlog`` and writes the
``skill`` table, so it imports ``penny.database``; it holds no engine and no tool
imports (the extraction pipeline, not the tool surface).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict
from similarity.embeddings import cosine_similarity

from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import RunProjection, RunProjectionStep, project_run
from penny.database.memory import _similarity as sim
from penny.database.memory.types import DedupThresholds, MemoryType
from penny.database.models import PromptLog, Skill
from penny.database.skill_store import steps_from_json
from penny.database.skills import (
    WRITE_TARGET_DESCRIPTION,
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    distill_steps,
    render_skill,
)
from penny.llm.similarity import embed_text
from penny.prompts import Prompt
from penny.text_validity import is_blank
from penny.tools.micro_context import (
    MicroContext,
    ParameterLabel,
    ParameterVerdict,
    SkillLabel,
    SkillShape,
)

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# The tools that WRITE durable state (mirrors ``objects._WRITE_TOOLS``, the
# run-record write set): a run qualifies as a skill only when its succeeded calls
# include at least one of these AND at least one read-shaped call.  ``done`` is
# loop control (excluded everywhere); every other tool is read-shaped.
WRITE_SHAPED_TOOLS = frozenset(
    {"collection_write", "update_entry", "collection_delete_entry", "log_append"}
)

# Registry-navigation verbs: the model uses these to ORIENT — resolve a skill or
# collection (``find``), read a skill's params (``skill_read``), inspect a
# collection's config (``memory_metadata``), or list the catalog
# (``collection_catalog``) — before it acts.  They are not part of the routine a
# skill captures (a re-run re-orients itself), and a ``find`` result ECHOES its
# query, which manufactured a FALSE binding when captured as a step (#1665).  So
# orientation calls are dropped from the distilled steps AND do not count as the
# qualifying CONTENT read: a find + write run is a pure write (the storage atom),
# not a skill.  The qualifying read must be a content read (browse, log_read,
# collection_read_latest, read_similar, collection_get, entry reads).
# Preceding conversation turns fed to the naming step — the user's instigating
# ask ('can you watch …') usually sits a turn or two before the demonstration,
# and the skill's name/description must carry that INTENT (#1658).
_NAMING_CONVERSATION_TURNS = 6

ORIENTATION_TOOLS = frozenset({"find", "skill_read", "memory_metadata", "collection_catalog"})

# The resolve-by-meaning verb (and its arg): its ``query`` phrases seed the run-end
# naming micro-context (#1665's step-1 doctrine sends the GENERIC task phrase to
# find), a naming signal even though the find call itself is dropped from the recipe.
_FIND_TOOL = "find"
_FIND_QUERY_ARG = "query"

_DONE_TOOL = PennyConstants.DONE_TOOL_NAME

# Deterministic naming: strip URLs, lowercase, keep the first few word tokens.
_URL_PATTERN = re.compile(r"https?://\S+")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_NAME_MAX_WORDS = 6
_FALLBACK_NAME = "learned-skill"


class ExtractionGate(StrEnum):
    """The closed set of reasons a run does NOT yield a skill — each a named,
    loggable qualify-gate failure (never a silent no-op)."""

    NOT_CHAT = "not_chat_run"
    NO_TOOL_CALLS = "no_tool_calls"
    BAILED = "text_bail_nudge_in_run"
    NO_CERTIFIED_STEPS = "no_certified_steps"
    PURE_READ = "pure_read_no_write"
    PURE_WRITE = "pure_write_no_read"


class SkillExtracted(BaseModel):
    """A run qualified and a skill was persisted — ``replaced`` is True when an
    existing skill (by name, or same shape + meaning) was overwritten.
    ``origin_message`` is the run's triggering message (the INSTANCE the skill was
    demonstrated on), carried so the narration frame can name it alongside the
    generic name/intent (#1665) — the skill's own ``description`` is now generic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: Skill
    replaced: bool
    origin_message: str


class NoExtraction(BaseModel):
    """A run did NOT yield a skill — ``gate`` names which qualify check failed."""

    gate: ExtractionGate


SkillExtractionResult = SkillExtracted | NoExtraction


def _runnable_steps(projection: RunProjection) -> list[RunProjectionStep]:
    """Every non-``done`` step of the run (the demonstration's real tool calls)."""
    return [step for step in projection.steps if step.call.name != _DONE_TOOL]


def _leaf_at(arguments: dict, path: list[str | int]):
    """The argument leaf a substitution's JSON path addresses (the step carries the
    call's VERBATIM arguments, so the demonstrated value is still in place)."""
    node = arguments
    for key in path:
        node = node[key]
    return node


def _certified_steps(
    projection: RunProjection, collector_surface: frozenset[str]
) -> list[RunProjectionStep]:
    """The run's non-``done``, non-ORIENTATION, COLLECTOR-runnable steps that SUCCEEDED
    — the routine that actually worked (#1659 filter-not-refuse; #1665 orientation-out;
    #1668 collector-surface-only).  Reads the structural per-call success stamp
    (``RunProjectionStep.success``, #1600): a step survives only when its stamp is
    exactly ``True`` (a recorded failure or a missing stamp is uncertain and left out),
    it is not a registry-navigation verb (``ORIENTATION_TOOLS`` — dropped from the recipe
    and not counted as the qualifying read), AND its tool is one a COLLECTOR can run
    (``collector_surface`` — a skill renders into a collector prompt, so a lifecycle call
    the demo made mid-run, e.g. ``collection_set`` to set up the container, is dropped
    from the recipe; it's not a step a collector could run and counts for nothing in the
    taxonomy).  So certified-by-execution + routine-only + runnable hold by construction."""
    return [
        step
        for step in _runnable_steps(projection)
        if step.success is True
        and step.call.name not in ORIENTATION_TOOLS
        and step.call.name in collector_surface
    ]


def _slug_name(origin_message: str) -> str:
    """A deterministic skill name from the triggering message: URLs removed,
    lowercased, non-alphanumeric runs collapsed to hyphens, capped at the first
    ``_NAME_MAX_WORDS`` words (e.g. 'read the aurora deck 2 listing at <url>, find
    the price, remember it.' → 'read-the-aurora-deck-2-listing').  The full message
    stays the description / find anchor; only the name is truncated."""
    without_urls = _URL_PATTERN.sub(" ", origin_message)
    words = _WORD_PATTERN.findall(without_urls.lower())
    return "-".join(words[:_NAME_MAX_WORDS]) or _FALLBACK_NAME


class SkillExtractor:
    """The run-end skill-extraction pipeline — one instance per chat agent, holding
    its DB + embedding client (threaded, never ambient state)."""

    def __init__(
        self,
        db: Database,
        embedding_client: LlmClient,
        model_client: LlmClient,
        *,
        agent_name: str,
        collector_tool_surface: frozenset[str],
    ) -> None:
        self._db = db
        self._embedding = embedding_client
        # The text model client drives the run-end naming micro-context (#1665) — the
        # SECOND customer of the micro-context machinery, threaded in (never ambient).
        self._micro_context = MicroContext(model_client)
        # The chat agent's name — the 'is this a chat run?' qualify anchor and the
        # author stamped on every extracted skill.
        self._agent_name = agent_name
        # The names of the tools a COLLECTOR can run (#1668) — threaded in (this module
        # holds no tool imports), single-sourced from ``collector_tool_surface`` so a
        # captured step a collector could never run (a lifecycle call the demo made) is
        # dropped from the recipe rather than baked into an uninstantiable skill.
        self._collector_surface = collector_tool_surface

    async def extract(self, run_id: str) -> SkillExtractionResult:
        """Extract a skill from one completed run's ledger rows — the summary method.

        Reads the run's prompts, projects them, runs the structural qualify gates,
        distils the surviving steps, names + dedups, and persists.  Returns the
        extracted skill or a typed no-extraction outcome naming the failed gate."""
        prompts = self._db.messages.get_run_prompts(run_id)
        projection = project_run(prompts)
        certified = _certified_steps(projection, self._collector_surface)
        gate = self._disqualify(prompts, projection, certified)
        if gate is not None:
            return NoExtraction(gate=gate)
        draft = await self._draft(run_id, projection, certified)
        return await self._persist(draft, projection.origin_message)

    def _disqualify(
        self,
        prompts: list[PromptLog],
        projection: RunProjection,
        certified: list[RunProjectionStep],
    ) -> ExtractionGate | None:
        """Run the ordered qualify gates; the FIRST failure's gate is returned
        (``None`` == qualifies).  Order: chat-run · has-calls · health · taxonomy."""
        if not self._is_chat_run(prompts):
            return ExtractionGate.NOT_CHAT
        if not _runnable_steps(projection):
            return ExtractionGate.NO_TOOL_CALLS
        if _has_text_bail_nudge(prompts):
            return ExtractionGate.BAILED
        if not certified:
            return ExtractionGate.NO_CERTIFIED_STEPS
        return _taxonomy_gate(certified)

    def _is_chat_run(self, prompts: list[PromptLog]) -> bool:
        """The run belongs to the chat agent (its prompts carry the chat
        ``agent_name`` — a browse micro-context row shares the run but never IS
        the whole run)."""
        return bool(prompts) and any(p.agent_name == self._agent_name for p in prompts)

    # ── Distill + name → draft ────────────────────────────────────────────────

    async def _draft(
        self,
        run_id: str,
        projection: RunProjection,
        certified: list[RunProjectionStep],
    ) -> SkillDraft:
        """Distil the certified steps into structured steps + candidate parameters,
        name the skill GENERICALLY and adjudicate each candidate via a single-shot
        micro-context (#1665/#1668/#1770 — a verb-noun name + description, then per
        candidate either a semantic name/description or a placeholder verdict), and
        bundle it for the store.

        On ANY naming failure the fallback is the deterministic slug of the triggering
        message + that message as the description, and candidates keep their
        arg-derived names as required parameters — the model writes LABELS and
        VERDICTS only; steps are untouched otherwise, and extraction never blocks on
        the rewrite.

        The SHAPE draw (#1803) then decides what the routine IS — its name and which
        of the kept values it is ABOUT — and its verdict supersedes the labeller's
        name, because a name written apart from that decision is what let a skill
        call itself a price watcher and then ask what to watch.  It runs only when
        there is something to decide, and its own failure costs nothing: every value
        stays a bindable parameter under the labeller's name."""
        steps, parameters = distill_steps(
            self._distill_inputs(projection, certified), self._attachment_names()
        )
        fallback_name = _slug_name(projection.origin_message)
        fallback_description = projection.origin_message or f"Skill: {fallback_name}"
        # ONE bounded read of the recent turns, passed to both draws: they read the
        # same window, and issuing the identical query twice per extraction only
        # invites the two draws to disagree about what the user said.
        conversation = self._db.messages.recent_conversation(_NAMING_CONVERSATION_TURNS)
        label = await self._label_skill(steps, parameters, projection, conversation)
        shape = await self._shape_skill(label, steps, projection.origin_message, conversation)
        name, description = _naming(label, shape, fallback_name, fallback_description)
        steps, parameters = _apply_parameter_labels(
            steps, parameters, label, _constant_keys(label, shape)
        )
        return SkillDraft(
            name=name,
            intent=description,
            description=description,
            steps=steps,
            parameters=parameters,
            source_run_id=run_id,
        )

    async def _label_skill(
        self,
        steps: list[SkillStep],
        parameters: list[SkillParameter],
        projection: RunProjection,
        conversation: list[tuple[str, str]],
    ) -> SkillLabel | None:
        """One single-shot naming micro-context over the rendered routine
        (#1665/#1668/#1770).

        Content = the numbered recipe with parameters as ``{variables}`` + the
        triggering message + the run's ``find`` query phrases + the CANDIDATE parameter
        list (each candidate's current arg-derived name, demonstrated value, and the
        arg site(s) it fills); the micro-context writes a GENERIC name + description
        AND, per candidate, either a semantic name/description (the user supplied it)
        or a placeholder description (they did not) — poison-screened + one reroll,
        its own ledger attribution.  ``None`` on any failure — the caller falls back to
        the slug + arg-derived names."""
        content = build_naming_content(
            steps, parameters, projection.origin_message, conversation, _find_phrases(projection)
        )
        return await self._micro_context.label_skill(content, run_target=self._agent_name)

    async def _shape_skill(
        self,
        label: SkillLabel | None,
        steps: list[SkillStep],
        origin_message: str,
        conversation: list[tuple[str, str]],
    ) -> SkillShape | None:
        """One single-shot SHAPE micro-context over the routine's kept values (#1803).

        Content = what the USER asked for (their turns, nothing the assistant said)
        + the values the labeller kept, each with its semantic name and the value it
        was demonstrated with — the labeller's per-value DESCRIPTION is deliberately
        dropped (see :class:`ShapeableValue`).  Deliberately smaller than the labelling
        content: the question is what the routine is FOR, and the steps, the arg sites,
        and the assistant's own wording are all evidence about how it was carried out.

        Takes the origin message directly rather than a whole projection — it is all it
        ever read from one, the same narrowing ``build_naming_content`` got.

        Skipped — ``None``, so nothing changes — when the labelling draw failed
        outright or kept nothing shapeable, since there is then no closed set of
        values to decide over."""
        values = _shapeable_values(label, steps)
        if label is None or not values:
            return None
        content = build_shape_content(values, origin_message, conversation, label.description)
        return await self._micro_context.shape_skill(
            content, [value.name for value in values], run_target=self._agent_name
        )

    def _attachment_names(self) -> frozenset[str]:
        """The names of Penny's own COLLECTIONS — the things a routine can be ATTACHED
        to (#1783).  A demonstrated leaf holding one of these names is decided by the
        attachment, so distillation marks it and the render seam binds it to whatever
        collection the skill is applied to.  Archived rows are included (a demonstration
        may have used one), logs are not: the attachment is always a collection, and
        re-pointing a log reference at one would produce a call the memory layer
        refuses.  Registry-derived — nothing here knows which tools a skill contains."""
        return frozenset(
            row.name
            for row in self._db.memories.list_all()
            if row.type == MemoryType.COLLECTION.value
        )

    @staticmethod
    def _distill_inputs(
        projection: RunProjection, certified: list[RunProjectionStep]
    ) -> list[DistillInput]:
        """One ``DistillInput`` per certified step — its ordinal, tool, verbatim
        arguments, and framed result (``distill_steps`` reads the result to infer
        bindings; the ``reasoning`` strip lives inside it, #1659)."""
        return [
            DistillInput(
                source_ordinal=step.ordinal,
                tool=step.call.name,
                arguments=step.call.arguments,
                result=projection.results.get(step.call_id, "") if step.call_id else "",
            )
            for step in certified
        ]

    # ── Persist (embed → dedup → upsert) ──────────────────────────────────────

    async def _persist(self, draft: SkillDraft, origin_message: str) -> SkillExtracted:
        """Embed the GENERIC description, resolve the dedup target
        (name-or-shape+meaning), and upsert — REPLACE by name, so a re-demonstration
        of the same routine overwrites the prior skill in place.  ``origin_message``
        (the demonstrated-on instance) rides back for the narration frame (#1665)."""
        embedding = await embed_text(self._embedding, draft.description)
        target_name = self._dedup_target(draft, embedding)
        if target_name != draft.name:
            draft = draft.model_copy(update={"name": target_name})
        skill, replaced = self._db.skills.upsert(
            draft, author=self._agent_name, description_embedding=embedding
        )
        logger.info(
            "Auto-extracted skill %r (%s) from run %s",
            skill.name,
            "replaced" if replaced else "new",
            draft.source_run_id,
        )
        return SkillExtracted(skill=skill, replaced=replaced, origin_message=origin_message)

    def _dedup_target(self, draft: SkillDraft, embedding: list[float] | None) -> str:
        """The name to upsert under (REPLACE semantics): (a) an exact name match →
        replace it; (b) else a same-tool-sequence, same-meaning skill → replace THAT
        one keeping its name; otherwise the fresh slug (insert)."""
        if self._db.skills.get(draft.name) is not None:
            return draft.name
        match = self._shape_and_meaning_match(draft, embedding)
        return match.name if match is not None else draft.name

    def _shape_and_meaning_match(
        self, draft: SkillDraft, embedding: list[float] | None
    ) -> Skill | None:
        """An existing skill with the SAME ordered tool sequence AND a description
        embedding within the house content-dedup threshold of this draft's — the
        clean/flaky re-demonstration collapse (#1658).  The threshold is the shared
        ``MEMORY_DEDUP_CONTENT_SIM_STRICT`` (never a new number)."""
        if embedding is None:
            return None
        threshold = DedupThresholds.from_runtime(RuntimeParams(self._db)).content_sim_strict
        candidate_shape = _tool_sequence(draft.steps)
        for skill in self._db.skills.list_all():
            if skill.description_embedding is None:
                continue
            if _tool_sequence(steps_from_json(skill.steps)) != candidate_shape:
                continue
            existing = sim.maybe_deserialize(skill.description_embedding)
            if existing is not None and cosine_similarity(embedding, existing) >= threshold:
                return skill
        return None


def build_naming_content(
    steps: list[SkillStep],
    parameters: list[SkillParameter],
    origin_message: str,
    conversation: list[tuple[str, str]],
    find_phrases: Sequence[str] = (),
) -> str:
    """The naming micro-context's content (#1665/#1668/#1770): the numbered recipe
    (parameters as ``{variables}``), the message that first demonstrated it, any
    ``find`` query phrases from the run (the generic task phrases the step-1 doctrine
    sends to find — a naming signal), and the CANDIDATE parameter list — each
    candidate's current arg-derived name, demonstrated value, and the arg site(s) it
    fills — so the model can both relabel each semantically and judge whether the USER
    supplied its value at all.  They are named *candidates* because the distiller's
    "everything else is a parameter" is a default the labeller adjudicates.

    PUBLIC because the labelling eval builds it too: that case drives ``label_skill``
    alone, so it must render the same content production renders and not a copy that
    can drift.  Takes the origin message and find phrases directly rather than a whole
    projection — they are all it ever read from one."""
    # The demonstrating message is a USER turn, and it must be rendered as one.
    # Presented under its own unattributed heading, the labeller read the
    # conversation block as the only record of what the user said, did not find
    # the demonstrated values there, and concluded the assistant had produced
    # them — correct reasoning over a presentation that hid who was speaking
    # (#1770; the thinking traces say it in as many words: "the user didn't give
    # the URL directly").  So it joins the conversation, deduped in case the
    # recent-turns window already carries it.
    turns = [*conversation]
    demonstration = (PennyConstants.MessageDirection.INCOMING, origin_message)
    if origin_message and demonstration not in turns:
        turns.append(demonstration)
    parts = []
    if turns:
        rendered = "\n".join(
            f"{'user' if direction == PennyConstants.MessageDirection.INCOMING else 'penny'}: "
            f"{content}"
            for direction, content in turns
        )
        parts.append(
            "Conversation that led to the construction of this routine "
            f"(the LAST user turn is the one that demonstrated it):\n{rendered}"
        )
    parts.append(f"Routine steps:\n{render_skill(steps)}")
    if find_phrases:
        parts.append("Search phrases used to look for a skill:\n" + "\n".join(find_phrases))
    param_lines = _parameter_lines(steps, parameters)
    if param_lines:
        parts.append(
            "Candidate parameters (each currently named after the tool arg it fills):\n"
            + param_lines
        )
    return "\n\n".join(parts)


class ShapeableValue(NamedTuple):
    """One value the SHAPE draw decides over (#1803): the labeller's semantic ``name``
    (the anchor the draw repeats back, and the parameter's eventual binding key), the
    ``current`` arg-derived key the verdict maps home to, and the ``demonstrated``
    value — without which "is the routine ABOUT this?" cannot be answered at all.

    The labeller's one-line description is deliberately NOT carried across.  It is
    written to answer "what should someone supply here", so it describes every value
    as a fill-in slot — *"plain-language phrase describing the information to pull"* —
    and a draw shown that reads it as an argument for PARAMETER, which is what four of
    five draws did.  Correct reasoning over a description that had already decided the
    question; the name and the demonstrated value say what the value IS without
    arguing what role it plays."""

    name: str
    current: str
    demonstrated: str


def _shapeable_values(label: SkillLabel | None, steps: list[SkillStep]) -> list[ShapeableValue]:
    """The values the shape draw may decide over (#1803) — the ones the labeller
    adjudicated as PARAMETER, in step order.

    Three exclusions, each because the value is not the draw's to decide.  A
    PLACEHOLDER never reached the user, so there is nothing they could have meant it
    to be about.  A candidate the labeller never covered is already in its
    per-candidate fallback, and absence stays absence rather than becoming input to a
    second judgment.  And an ATTACHMENT-MARKED leaf is decided by where the routine is
    APPLIED (#1783) — baking it would make a rendered program name a collection it was
    demonstrated on rather than the one it runs against, which is the one thing the
    retarget seam exists to prevent.  Keyed to the MARK, not to any tool name."""
    if label is None:
        return []
    marked = _attachment_marked(steps)
    return [
        ShapeableValue(
            name=parameter.name,
            current=current,
            demonstrated=_parameter_facts(steps, current)[0],
        )
        for current, parameter in label.parameters.items()
        if parameter.verdict == ParameterVerdict.PARAMETER
        and not is_blank(parameter.name)
        and current not in marked
    ]


def _attachment_marked(steps: list[SkillStep]) -> frozenset[str]:
    """The candidate names carrying the attachment mark at any leaf (#1783) — what
    applying the routine somewhere decides, and so never a constant."""
    return frozenset(
        sub.parameter
        for step in steps
        for sub in step.substitutions
        if sub.attachment and sub.parameter is not None
    )


def build_shape_content(
    values: list[ShapeableValue],
    origin_message: str,
    conversation: list[tuple[str, str]],
    round_summary: str,
) -> str:
    """The shape micro-context's content (#1803): what the USER asked for, then the
    values the routine used.  PUBLIC because the shape eval builds it too — that case
    hands the draw synthetic values standing in for what the labeller would have
    emitted, so it must go through THIS function and not a copy that can drift.

    Only the user's turns are rendered.  The question is what the routine is FOR, and
    the user is the only one who can say — the assistant's replies describe how it was
    carried out, which is the evidence that led the model to name a routine after
    whatever it happened to do.  The demonstrating message joins them as the last
    turn (deduped, since the recent window may already carry it): it is a user turn,
    and the labelling draw already proved that hiding who was speaking makes the model
    reason correctly to the wrong answer (#1770).

    Each value renders as its name and the value it was demonstrated with, and nothing
    else — see :class:`ShapeableValue` for why the labeller's PER-VALUE description is
    dropped.  The labeller's ROUTINE description is the opposite case and is passed:
    it is a statement of what the round was FOR ("track a marketplace item's current
    price…"), which is the very thing this draw has to settle, and it reads the user's
    words for it rather than describing a slot to fill."""
    incoming = PennyConstants.MessageDirection.INCOMING
    asks = [content for direction, content in conversation if direction == incoming]
    if origin_message and origin_message not in asks:
        asks.append(origin_message)
    lines = "\n".join(f"- {value.name} = {value.demonstrated!r}" for value in values)
    parts = [f"What the user asked for:\n{'\n'.join(asks)}"]
    if round_summary:
        parts.append(f"What the round did, in one line:\n{round_summary}")
    parts.append(f"The values the routine used to do it:\n{lines}")
    return "\n\n".join(parts)


def _naming(
    label: SkillLabel | None,
    shape: SkillShape | None,
    fallback_name: str,
    fallback_description: str,
) -> tuple[str, str]:
    """The routine's name + description, from the FIRST draw that produced one (#1803):
    the shape draw, then the labeller, then the deterministic slug.

    The shape draw wins because it wrote the name in the same decision as the
    constants, so the two cannot disagree — which is the whole defect this closes.
    Each fallback is a rung down, and naming never blocks extraction."""
    if shape is not None:
        return _slug_name(shape.name), shape.description
    if label is not None:
        return _slug_name(label.name), label.description
    return fallback_name, fallback_description


def _constant_keys(label: SkillLabel | None, shape: SkillShape | None) -> frozenset[str]:
    """The CURRENT (arg-derived) keys of the values the shape draw called CONSTANT
    (#1803), mapped home through the semantic names the labeller gave them.

    The two draws stay in their own types — the shape draw's answer never becomes a
    ``ParameterVerdict``, which is the labeller's output and predates this. They meet
    here, in the extractor that orchestrates both, and nowhere else. An unmatched name
    simply yields no key: the shape draw membership-validates its own answer, so this
    is belt and braces rather than a silent drop."""
    if label is None or shape is None:
        return frozenset()
    return frozenset(
        current
        for current, parameter in label.parameters.items()
        if parameter.verdict == ParameterVerdict.PARAMETER and parameter.name in shape.fixed
    )


def _parameter_lines(steps: list[SkillStep], parameters: list[SkillParameter]) -> str:
    """One line per candidate parameter for the naming content (#1668/#1770): its
    current name, the value it was demonstrated with, and the tool-arg site(s) it fills
    — the facts the model needs both to give it a semantic name and description and to
    judge whether the user supplied that value."""
    lines: list[str] = []
    for parameter in parameters:
        value, sites = _parameter_facts(steps, parameter.name)
        site_text = ", ".join(sites) if sites else "(unknown)"
        lines.append(f"- {parameter.name}: fills {site_text}; demonstrated value: {value!r}")
    return "\n".join(lines)


def _parameter_facts(steps: list[SkillStep], parameter: str) -> tuple[str, list[str]]:
    """The demonstrated value and the arg site(s) a parameter fills, read structurally
    off the steps' substitutions (#1668): every ``HOLE`` substitution naming
    ``parameter`` contributes its ``<tool>.<path>`` site and the literal at that path
    (all such leaves share one value — the distiller dedups by value)."""
    value = ""
    sites: list[str] = []
    for step in steps:
        for sub in step.substitutions:
            if sub.kind != SkillSubKind.HOLE or sub.parameter != parameter:
                continue
            sites.append(f"{step.tool}.{_render_path(sub.path)}")
            value = _value_at_path(step.arguments, sub.path)
    return value, sites


def _render_path(path: list[str | int]) -> str:
    """A leaf's JSON path as a readable arg site — ``["queries", 0]`` → ``queries[0]``,
    ``["entries", 0, "key"]`` → ``entries[0].key``."""
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}" if rendered else part
    return rendered


def _value_at_path(arguments: dict, path: list[str | int]) -> str:
    """The string leaf at ``path`` in a step's ``arguments`` (the demonstrated value);
    ``""`` if the path doesn't resolve to a string."""
    node: object = arguments
    for part in path:
        node = _child_at(node, part)
        if node is None:
            return ""
    return node if isinstance(node, str) else ""


def _child_at(node: object, part: str | int) -> object:
    """The child of ``node`` at ``part`` — a dict key (str part) or a list index (int
    part) — or ``None`` when ``part`` doesn't address a child."""
    if isinstance(node, dict) and isinstance(part, str):
        return node.get(part)
    if isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
        return node[part]
    return None


# ── Semantic parameter labels: apply + deterministic hardening (#1668) ─────────

_PARAM_WHITESPACE = re.compile(r"\s+")
_PARAM_NON_IDENTIFIER = re.compile(r"[^a-z0-9_]")


def _slug_parameter_name(raw: str) -> str:
    """Harden a model-written semantic parameter name into an identifier-safe binding
    key (#1668, load-bearing — the name is the params binding key at instantiation):
    lowercase, whitespace → underscores, strip anything but ``[a-z0-9_]``, trim stray
    underscores.  Empty when nothing survives (the caller then keeps the arg-derived
    name — per-parameter fallback)."""
    lowered = _PARAM_WHITESPACE.sub("_", raw.strip().lower())
    return _PARAM_NON_IDENTIFIER.sub("", lowered).strip("_")


def _apply_parameter_labels(
    steps: list[SkillStep],
    parameters: list[SkillParameter],
    label: SkillLabel | None,
    constant_keys: frozenset[str] = frozenset(),
) -> tuple[list[SkillStep], list[SkillParameter]]:
    """Apply each candidate's ROLE (#1668/#1770/#1783/#1803) — the three-way the two
    draws settled between them.

    A PARAMETER relabels the candidate with its hardened semantic name + description,
    maps the rename through every leaf site, and CLEARS any attachment mark there (a
    value the user chose is theirs to bind, never something the attachment overwrites).
    A key in ``constant_keys`` — the SHAPE draw's answer, arriving as its own argument
    rather than as a verdict, since the verdict union is the labeller's — is what the
    routine is ABOUT, so it stops being a parameter and its leaf sites lose their
    substitutions entirely — a leaf nothing covers renders verbatim,
    which is precisely a baked value, so no new render path exists for it. A
    PLACEHOLDER says the user never supplied that value, so its leaf sites become
    placeholder substitutions carrying the labeller's description, and any attachment
    mark stands.

    A candidate the label doesn't cover — or whose semantic name slugs to empty — keeps
    its arg-derived required parameter (per-candidate fallback, not all-or-nothing, and
    absence is never a drop), EXCEPT at an attachment-marked leaf: nobody can bind what
    the attachment decides, so an unadjudicated marked leaf falls back to a placeholder
    carrying the fixed ``WRITE_TARGET_DESCRIPTION`` (#1777's string, kept as exactly
    that fallback).  A name collision gets a numeric suffix, since the name is the
    binding key.  ``label is None`` (the whole draw failed) runs the same pass with no
    verdicts, so the marked-leaf fallback still applies."""
    verdicts = label.parameters if label is not None else {}
    adjudicated = frozenset(verdicts)
    attachment_filled = _attachment_filled(steps, adjudicated)
    rename: dict[str, str] = {}
    placeholders: dict[str, str] = {}
    constants: set[str] = set()
    used: set[str] = set()
    named: list[SkillParameter] = []
    for parameter in parameters:
        param_label = verdicts.get(parameter.name)
        if parameter.name in attachment_filled:
            continue  # every site is the attachment's to fill — not a parameter at all
        if param_label is not None and param_label.verdict == ParameterVerdict.PLACEHOLDER:
            placeholders[parameter.name] = param_label.description
            continue
        if parameter.name in constant_keys:
            constants.add(parameter.name)
            continue
        named.append(_relabelled(parameter, param_label, rename, used))
    rewritten = [
        _rewrite_step_leaves(step, rename, placeholders, frozenset(constants), adjudicated)
        for step in steps
    ]
    return rewritten, named


def _attachment_filled(steps: list[SkillStep], adjudicated: frozenset[str]) -> frozenset[str]:
    """The candidates the ATTACHMENT fills outright (#1783) — every one of whose leaf
    sites is marked and none of which the labeller adjudicated.  They are dropped
    before naming rather than after: a declared parameter nobody's call reads would be
    a required input instantiation refuses to proceed without and nothing consumes, and
    reserving its name would push a real parameter onto a numeric suffix for nothing."""
    sites: dict[str, list[bool]] = {}
    for step in steps:
        for sub in step.substitutions:
            if sub.kind == SkillSubKind.HOLE and sub.parameter:
                sites.setdefault(sub.parameter, []).append(sub.attachment)
    return frozenset(
        name for name, marks in sites.items() if all(marks) and name not in adjudicated
    )


def _relabelled(
    parameter: SkillParameter,
    param_label: ParameterLabel | None,
    rename: dict[str, str],
    used: set[str],
) -> SkillParameter:
    """One real parameter under its semantic label (#1668): the hardened name (falling
    back to the arg-derived one when the label is absent or slugs to empty), made
    unique, recorded in ``rename`` for the leaf sites, plus the one-line description."""
    candidate = _slug_parameter_name(param_label.name) if param_label is not None else ""
    final = _unique_name(candidate or parameter.name, used)
    used.add(final)
    rename[parameter.name] = final
    description = param_label.description if param_label and param_label.description else None
    return parameter.model_copy(update={"name": final, "description": description})


def _unique_name(candidate: str, used: set[str]) -> str:
    """``candidate`` if unused, else ``candidate_2`` / ``candidate_3`` / … — parameter
    names are binding keys, so two must never collide (#1668)."""
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def _rewrite_step_leaves(
    step: SkillStep,
    rename: dict[str, str],
    placeholders: dict[str, str],
    constants: frozenset[str],
    adjudicated: frozenset[str],
) -> SkillStep:
    """A copy of ``step`` with every ``HOLE`` substitution resolved against the
    verdicts (#1668/#1770/#1783/#1803): a real parameter's ``parameter`` field is
    remapped through ``rename`` (so every leaf site follows it to the semantic name and
    the render substitutes by that name); a candidate called a PLACEHOLDER becomes a
    ``PLACEHOLDER`` substitution carrying its description, so the render shows what
    belongs there instead of freezing the demonstrated value; and a CONSTANT loses its
    substitution entirely — a leaf no substitution covers renders verbatim, which is
    exactly what a value the routine is ABOUT should do."""
    subs = [
        rewritten
        for sub in step.substitutions
        if (rewritten := _rewrite_substitution(sub, rename, placeholders, constants, adjudicated))
        is not None
    ]
    return step.model_copy(update={"substitutions": subs})


def _rewrite_substitution(
    sub: SkillSubstitution,
    rename: dict[str, str],
    placeholders: dict[str, str],
    constants: frozenset[str],
    adjudicated: frozenset[str],
) -> SkillSubstitution | None:
    """One substitution under the verdicts — renamed, converted to a placeholder,
    DROPPED (a constant: the leaf keeps its demonstrated value, which is what makes
    this routine this routine, #1803), or left exactly as it was (a binding, or a
    parameter with no verdict)."""
    if sub.kind != SkillSubKind.HOLE or sub.parameter is None:
        return sub
    if sub.parameter in constants:
        return None
    if sub.parameter in placeholders:
        return _as_placeholder(sub, placeholders[sub.parameter])
    if sub.attachment and sub.parameter not in adjudicated:
        # Nobody can bind what the attachment decides, and no draw described it — so the
        # fixed fallback string stands in, and the mark survives for the render seam.
        return _as_placeholder(sub, WRITE_TARGET_DESCRIPTION)
    if sub.parameter in rename:
        # A verdict said the USER supplied this, so it is a parameter they bind — the
        # attachment must not overwrite it, and the mark is cleared.
        return sub.model_copy(update={"parameter": rename[sub.parameter], "attachment": False})
    return sub


def _as_placeholder(sub: SkillSubstitution, description: str) -> SkillSubstitution:
    """``sub`` as a PLACEHOLDER carrying ``description`` — the leaf renders as what
    belongs there, never the demonstrated value.  Any attachment mark rides along: an
    internally-chosen leaf that named a collection is still the attachment's to fill."""
    return sub.model_copy(
        update={
            "kind": SkillSubKind.PLACEHOLDER,
            "parameter": None,
            "description": description,
        }
    )


def _find_phrases(projection: RunProjection) -> list[str]:
    """The non-blank ``find(query=…)`` phrases across the run's steps — the generic
    task phrases (#1665's step-1 doctrine sends the GENERIC task to find), a naming
    signal even though the find call itself is dropped from the recipe."""
    phrases: list[str] = []
    for step in projection.steps:
        if step.call.name != _FIND_TOOL:
            continue
        query = step.call.arguments.get(_FIND_QUERY_ARG)
        if isinstance(query, str) and query.strip():
            phrases.append(query.strip())
    return phrases


def _tool_sequence(steps: list[SkillStep]) -> list[str]:
    """The ordered list of a skill's step tool names — its shape fingerprint."""
    return [step.tool for step in steps]


def _taxonomy_gate(certified: list[RunProjectionStep]) -> ExtractionGate | None:
    """The read/write taxonomy over the SUCCEEDED calls: a routine SENSES and ACTS,
    so it needs ≥1 write-shaped call AND ≥1 read-shaped call.  A pure read is
    answering; a pure write is the storage atom ('remember this'); neither is a
    skill.  ``None`` == the taxonomy is satisfied."""
    tools = [step.call.name for step in certified]
    has_write = any(tool in WRITE_SHAPED_TOOLS for tool in tools)
    has_read = any(tool not in WRITE_SHAPED_TOOLS for tool in tools)
    if not has_write:
        return ExtractionGate.PURE_READ
    if not has_read:
        return ExtractionGate.PURE_WRITE
    return None


def _has_text_bail_nudge(prompts: list[PromptLog]) -> bool:
    """True when the run's prompt rows carry either text-bail nudge marker — the
    model failed to route a call through the tool channel at some step, so the run
    is unhealthy and must not be captured as a routine.  Reads the nudge CONSTANTS
    (``Prompt.TOOL_FORMAT_NUDGE`` / ``Prompt.CHAT_CALL_AS_TEXT_NUDGE``), decoding
    each prompt's ``messages`` so a multi-line nudge matches its real content, not a
    JSON-escaped blob."""
    markers = (Prompt.TOOL_FORMAT_NUDGE, Prompt.CHAT_CALL_AS_TEXT_NUDGE)
    for prompt in prompts:
        for message in _decoded_messages(prompt):
            content = message.get("content") or ""
            if any(marker in content for marker in markers):
                return True
    return False


def _decoded_messages(prompt: PromptLog) -> list[dict]:
    """One prompt row's ``messages`` JSON decoded to dicts (empty when absent)."""
    if not prompt.messages:
        return []
    decoded = json.loads(prompt.messages)
    return decoded if isinstance(decoded, list) else []
