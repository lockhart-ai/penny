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
* **label** — a semantic name + a one-line "what belongs here each run" for EVERY spot
  in the routine, written by a single-shot leaf-labelling micro-context (#1828, the
  SECOND customer of the micro-context machinery) over the distilled routine.  It
  judges nothing: every leaf is a placeholder unconditionally, so no draw is asked
  where a value came from (#1824 — the per-candidate provenance verdict it replaces
  pinned at ~0.7-0.8 across three wording interventions, and compounded with every
  extra leaf).  A labelled spot renders as what belongs there, never as the frozen
  demonstrated value; one the draw missed keeps its arg-derived name.  Extraction
  NEVER blocks on the rewrite.
* **the interface, INTERIM** — a skill's name, description and parameters are decided
  from the user's ask alone by the FRAMER, and that beat lands next (#1824).  Until it
  does, the name/description are the deterministic slug of the triggering message
  (URLs removed, ≤6 words) + that message, and a fully-labelled routine comes out with
  NO parameters: stated here, on #1828, as a decision rather than discovered as a
  surprise.  ``shape_skill`` (#1803, the FOURTH customer) no longer fires from this
  pipeline — it decided which values a routine was ABOUT, over a closed set of values
  the labeller's verdicts produced, and there are no verdicts any more.
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
from penny.tools.micro_context import MicroContext, SkillLabels

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
# Preceding conversation turns fed to the labelling step — the user's instigating
# ask ('can you watch …') usually sits a turn or two before the demonstration, and
# what a spot IS is only legible against why the routine exists (#1658/#1828).
_NAMING_CONVERSATION_TURNS = 6

ORIENTATION_TOOLS = frozenset({"find", "skill_read", "memory_metadata", "collection_catalog"})

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
        """Distil the certified steps into structured steps + spots, label every spot
        via a single-shot micro-context (#1828 — a semantic name + one line of what
        belongs there, per spot), and bundle it for the store.

        The name and description are the deterministic slug of the triggering message
        for now: naming the routine and deciding its parameters is the FRAMER's
        decision from the user's ask alone (#1824), and that beat lands next.  So a
        freshly taught skill comes out slug-named and parameter-less, which is stated
        on #1828 as a decision rather than discovered as a surprise.

        The labelling draw's failure costs nothing either way: every spot simply keeps
        its arg-derived name, and extraction never blocks on the rewrite."""
        steps, parameters = distill_steps(
            self._distill_inputs(projection, certified), self._attachment_names()
        )
        name = _slug_name(projection.origin_message)
        description = projection.origin_message or f"Skill: {name}"
        conversation = self._db.messages.recent_conversation(_NAMING_CONVERSATION_TURNS)
        labels = await self._label_skill(steps, parameters, projection, conversation)
        steps, parameters = _apply_leaf_labels(steps, parameters, labels)
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
    ) -> SkillLabels | None:
        """One single-shot labelling micro-context over the rendered routine (#1828).

        Content = the conversation that led to the routine + the numbered recipe with
        every spot as a ``{variable}`` + the placeholder list (each spot's current
        arg-derived name, the arg site(s) it fills, and its demonstrated value); the
        micro-context writes one line per spot — a semantic name and what belongs
        there each run — poison-screened + one reroll, its own ledger attribution.

        The spots' current names ride along as the COVERAGE set (#1828): the draw is
        accepted only when every one of them came back with a well-formed line, so a
        decayed tag costs the whole draw rather than one spot its label.  ``None`` then,
        and the caller keeps every spot's arg-derived name."""
        content = build_naming_content(steps, parameters, projection.origin_message, conversation)
        return await self._micro_context.label_skill(
            content, [parameter.name for parameter in parameters], run_target=self._agent_name
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
) -> str:
    """The leaf labeller's content (#1828): the conversation that led to the routine,
    the numbered recipe (every spot as a ``{variable}``), and the PLACEHOLDER list —
    each spot's current arg-derived name, the arg site(s) it fills, and the value it
    was demonstrated with — so the model can name what each spot IS and say what
    belongs there each run.

    They are named *placeholders*, not candidates, because there is no longer a
    question about any of them: every leaf is a placeholder unconditionally, and the
    routine's interface is the framer's separate decision (#1824).  The run's ``find``
    query phrases went with it — that section's consumer was routine naming, which has
    left this draw.

    PUBLIC because the labelling eval builds it too: that case drives ``label_skill``
    alone, so it must render the same content production renders and not a copy that
    can drift.  Takes the origin message directly rather than a whole projection — it
    is all it ever read from one."""
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
    placeholder_lines = _placeholder_lines(steps, parameters)
    if placeholder_lines:
        parts.append(
            f"Placeholders (each currently named after the tool arg it fills):\n{placeholder_lines}"
        )
    return "\n\n".join(parts)


class ShapeableValue(NamedTuple):
    """One value the SHAPE draw decides over (#1803): the semantic ``name`` (the anchor
    the draw repeats back), the ``current`` arg-derived key the answer maps home to,
    and the ``demonstrated`` value — without which "is the routine ABOUT this?" cannot
    be answered at all.

    Since #1828 the run-end pipeline no longer produces these: the shape draw decided
    over the values a per-candidate VERDICT kept, and there are no verdicts any more.
    The draw and its contract are untouched, and its own eval (``test_skill_shape.py``)
    drives it with authored values — which is what this type and
    :func:`build_shape_content` serve until the framer beat settles where the question
    lives.

    A per-value description is deliberately NOT carried across.  It is
    written to answer "what should someone supply here", so it describes every value
    as a fill-in slot — *"plain-language phrase describing the information to pull"* —
    and a draw shown that reads it as an argument for PARAMETER, which is what four of
    five draws did.  Correct reasoning over a description that had already decided the
    question; the name and the demonstrated value say what the value IS without
    arguing what role it plays."""

    name: str
    current: str
    demonstrated: str


def build_shape_content(
    values: list[ShapeableValue],
    origin_message: str,
    conversation: list[tuple[str, str]],
    round_summary: str,
) -> str:
    """The shape micro-context's content (#1803): what the USER asked for, then the
    values the routine used.  PUBLIC because the shape eval builds it — that case hands
    the draw authored values, so it must go through THIS function and not a copy that
    can drift; since #1828 it is the only caller (see :class:`ShapeableValue`).

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


def _placeholder_lines(steps: list[SkillStep], parameters: list[SkillParameter]) -> str:
    """One line per offered spot for the labelling content (#1828): its current name,
    the tool-arg site(s) it fills, and the value it was demonstrated with — the three
    facts the model names the spot from, each rendered verbatim so its line maps back
    with no guess.

    Sites are joined with ``and`` because a spot filling two of them is ONE spot, and
    the line has to read that way: the draw's whole job on such a leaf is a single name
    covering both uses."""
    lines: list[str] = []
    for parameter in parameters:
        value, sites = _parameter_facts(steps, parameter.name)
        site_text = " and ".join(sites) if sites else "(unknown)"
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


# ── Leaf labels: apply, and the binding-key hardening rule (#1668/#1828) ───────

_PARAM_WHITESPACE = re.compile(r"\s+")
_PARAM_NON_IDENTIFIER = re.compile(r"[^a-z0-9_]")


def slug_parameter_name(raw: str) -> str:
    """Harden a model-written semantic name into an identifier-safe binding key
    (#1668, load-bearing — a skill's parameter name is its binding key at
    instantiation, ``params={'url': …}``): lowercase, whitespace → underscores, strip
    anything but ``[a-z0-9_]``, trim stray underscores.  Empty when nothing survives.

    PUBLIC since #1828, because the rule outlived its caller for one beat: the leaf
    labeller draws a semantic name per spot, but where that name lands as a key is the
    runtime-join beat's question, so nothing binds one today.  The labelling eval
    scores "did the draw produce a usable binding key" through THIS function rather
    than a copy of it, so the check can never drift from the rule it is checking."""
    lowered = _PARAM_WHITESPACE.sub("_", raw.strip().lower())
    return _PARAM_NON_IDENTIFIER.sub("", lowered).strip("_")


def _apply_leaf_labels(
    steps: list[SkillStep],
    parameters: list[SkillParameter],
    labels: SkillLabels | None,
) -> tuple[list[SkillStep], list[SkillParameter]]:
    """Apply the labeller's per-spot labels (#1828) — the interim wiring, declared on
    the ticket rather than hidden.

    EVERY labelled spot becomes a ``PLACEHOLDER`` substitution carrying the labeller's
    description, so the render shows what belongs there instead of freezing the
    demonstrated value, and it stops being a parameter — a fully-labelled routine
    therefore comes out with NO parameters at all until the framer beat decides the
    interface.  Any attachment mark stands: nothing can clear it any more, because a
    destination is never a parameter (#1827 principle 4).

    An unlabelled spot keeps its arg-derived required parameter, EXCEPT at an
    attachment-marked leaf: nobody can bind what the attachment decides, so an
    unlabelled marked leaf falls back to a placeholder carrying the fixed
    ``WRITE_TARGET_DESCRIPTION`` (#1777's string, kept as exactly that fallback).
    Since #1828 an accepted draw covers every spot, so "unlabelled" means the WHOLE
    draw failed (``labels is None``) — all-or-nothing at the draw, never a hole inside
    an accepted one — and this pass runs the same way with nothing labelled.

    A label whose DESCRIPTION is blank is treated as no label at all — the same
    fallback, everywhere at once.  The description is what the leaf renders as, so a
    blank one would put an empty ``{}`` where the recipe should say what belongs there:
    a spot that stopped being bindable and says nothing, which is strictly worse than
    the arg-derived name it replaced.  The grammar lets a line stop after the name (so
    a drawn name is still readable, and the labelling eval scores the missing
    description as its own miss); the CONSUMER is where it has to be usable."""
    drawn = labels.labels if labels is not None else {}
    usable = {
        current: label.description
        for current, label in drawn.items()
        if not is_blank(label.description)
    }
    labelled = frozenset(usable)
    attachment_filled = _attachment_filled(steps, labelled)
    placeholders: dict[str, str] = {}
    kept: list[SkillParameter] = []
    for parameter in parameters:
        if parameter.name in attachment_filled:
            continue  # every site is the attachment's to fill — not a parameter at all
        if parameter.name in usable:
            placeholders[parameter.name] = usable[parameter.name]
            continue
        kept.append(parameter)
    rewritten = [_rewrite_step_leaves(step, placeholders, labelled) for step in steps]
    return rewritten, kept


def _attachment_filled(steps: list[SkillStep], labelled: frozenset[str]) -> frozenset[str]:
    """The spots the ATTACHMENT fills outright (#1783) — every one of whose leaf sites
    is marked and none of which the labeller named.  They are dropped before naming
    rather than after: a declared parameter nobody's call reads would be a required
    input instantiation refuses to proceed without and nothing consumes."""
    sites: dict[str, list[bool]] = {}
    for step in steps:
        for sub in step.substitutions:
            if sub.kind == SkillSubKind.HOLE and sub.parameter:
                sites.setdefault(sub.parameter, []).append(sub.attachment)
    return frozenset(name for name, marks in sites.items() if all(marks) and name not in labelled)


def _rewrite_step_leaves(
    step: SkillStep,
    placeholders: dict[str, str],
    labelled: frozenset[str],
) -> SkillStep:
    """A copy of ``step`` with every ``HOLE`` substitution resolved against the labels
    (#1770/#1783/#1828): a labelled spot becomes a ``PLACEHOLDER`` substitution
    carrying its description, so the render shows what belongs there instead of
    freezing the demonstrated value; everything else is left exactly as it was."""
    subs = [_rewrite_substitution(sub, placeholders, labelled) for sub in step.substitutions]
    return step.model_copy(update={"substitutions": subs})


def _rewrite_substitution(
    sub: SkillSubstitution,
    placeholders: dict[str, str],
    labelled: frozenset[str],
) -> SkillSubstitution:
    """One substitution under the labels — converted to a placeholder carrying what
    belongs there, or left exactly as it was (a binding, or a spot no line covered)."""
    if sub.kind != SkillSubKind.HOLE or sub.parameter is None:
        return sub
    if sub.parameter in placeholders:
        return _as_placeholder(sub, placeholders[sub.parameter])
    if sub.attachment and sub.parameter not in labelled:
        # Nobody can bind what the attachment decides, and no line described it — so the
        # fixed fallback string stands in, and the mark survives for the render seam.
        return _as_placeholder(sub, WRITE_TARGET_DESCRIPTION)
    return sub


def _as_placeholder(sub: SkillSubstitution, description: str) -> SkillSubstitution:
    """``sub`` as a PLACEHOLDER carrying ``description`` — the leaf renders as what
    belongs there, never the demonstrated value.  Any attachment mark rides along: what
    a routine is applied to decides where it writes, and no label changes that."""
    return sub.model_copy(
        update={
            "kind": SkillSubKind.PLACEHOLDER,
            "parameter": None,
            "description": description,
        }
    )


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
