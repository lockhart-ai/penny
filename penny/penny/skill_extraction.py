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
* **label the leaves** — every candidate leaf is a PLACEHOLDER, unconditionally, and a
  single-shot LEAF LABELLER (#1824, the SECOND customer of the micro-context
  machinery) NAMES each one: a semantic name for the spot plus one line of what belongs
  there each run.  Its whole input is the routine's calls — the IMPLEMENTATION — and it
  decides nothing about the skill.  A leaf it never covers keeps its arg-derived name,
  per leaf, so extraction never blocks on the naming.
* **frame the skill** — an INDEPENDENT single-shot SKILL FRAMER (#1824, the FOURTH
  customer) writes the skill's public signature from the user's own messages alone —
  the INTERFACE: a GENERIC verb-noun name ("look up a price on a listing page and
  record it", never "read-the-aurora-deck-2-listing", so cross-instance ``find`` can
  match it), a one-line description, and the parameters someone must supply to set it
  up on a new occasion.  Name, description and parameters come out of ONE decision,
  which is what stops a skill from naming itself for something it then asks for.  On
  ANY failure the fallback is the deterministic slug of the triggering message (URLs
  removed, ≤6 words) + that message as the description, with no parameters declared.

  Neither draw reads the other's output, and neither sees the other's input: they can
  run in either order or at once.  That separation is the ticket's whole point — the
  pipeline it replaces asked ONE draw, per leaf, whether the USER supplied that value,
  which is an interface question asked of implementation artifacts.  Measured across
  three independent wordings, that verdict pinned at ~0.7-0.8 and would not move
  (#1821/#1823): a reworded extract instruction and a storage key slugged out of the
  user's own URL are both the user's words re-worded by the assistant, and what
  separates them is whether the THING the value is for was asked for — a question about
  the round, asked once, not about a string, asked four times.
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
from typing import TYPE_CHECKING

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
    LeafLabels,
    LeafPlaceholder,
    MicroContext,
    SkillFraming,
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
# Preceding conversation turns fed to the FRAMING step — the user's instigating
# ask ('can you watch …') usually sits a turn or two before the demonstration,
# and the skill's name/description must carry that INTENT (#1658).
_FRAMING_CONVERSATION_TURNS = 6

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
        """Distil the certified steps, then take the two INDEPENDENT run-end draws and
        bundle the result for the store (#1824).

        The two draws answer two questions from two disjoint inputs, and neither reads
        the other's output — they could run in either order or at once.  The LEAF
        LABELLER names every placeholder in the routine's calls, from the calls alone
        (the implementation).  The SKILL FRAMER writes the skill's name, description and
        parameters, from the user's own messages alone (the interface).

        Each failure degrades on its own: a failed labelling leaves every placeholder
        under its arg-derived name, and a failed framing falls back to the deterministic
        slug of the triggering message with no parameters.  Extraction never blocks on
        either."""
        steps, parameters = distill_steps(
            self._distill_inputs(projection, certified), self._attachment_names()
        )
        conversation = self._db.messages.recent_conversation(_FRAMING_CONVERSATION_TURNS)
        leaves = await self._label_leaves(steps, parameters)
        framing = await self._frame_skill(projection.origin_message, conversation)
        name, description = _naming(framing, projection.origin_message)
        return SkillDraft(
            name=name,
            intent=description,
            description=description,
            steps=_apply_leaf_labels(steps, parameters, leaves),
            parameters=_declared_parameters(framing),
            source_run_id=run_id,
        )

    async def _label_leaves(
        self, steps: list[SkillStep], parameters: list[SkillParameter]
    ) -> LeafLabels | None:
        """One single-shot LEAF LABELLING micro-context over the routine's calls (#1824).

        Content = the numbered recipe + one line per placeholder (its current
        arg-derived name, the value that sat there in the demonstration, and the arg
        site(s) it fills) — and NOTHING the user said.  Naming a spot is answered by the
        calls it sits between; handing this draw the conversation is what made it answer
        an interface question it has no business being asked.

        ``None`` on any failure — every placeholder then keeps its arg-derived name."""
        content = build_leaf_labelling_content(steps, parameters)
        return await self._micro_context.label_leaves(
            content, [parameter.name for parameter in parameters], run_target=self._agent_name
        )

    async def _frame_skill(
        self, origin_message: str, conversation: list[tuple[str, str]]
    ) -> SkillFraming | None:
        """One single-shot SKILL FRAMING micro-context over the user's ask (#1824).

        Content = the user's own turns and nothing else — not the calls, not the values
        they carried, not the labeller's names.  What a skill IS and what it ASKS FOR
        are decided by what was asked for; the calls are how that ask was carried out
        this once, and every measured attempt to read the interface off them hit the
        same ceiling (#1821/#1823).

        ``None`` on any failure — the caller falls back to the slug of the triggering
        message with no parameters."""
        content = build_framing_content(origin_message, conversation)
        return await self._micro_context.frame_skill(content, run_target=self._agent_name)

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


def build_leaf_labelling_content(steps: list[SkillStep], parameters: list[SkillParameter]) -> str:
    """The leaf labeller's content (#1824): the numbered recipe (placeholders as
    ``{variables}``) and one line per placeholder — its current arg-derived name, the
    value that sat there in the demonstration, and the arg site(s) it fills.

    NOTHING the user said is rendered here, and that omission is the design.  Naming a
    spot is answered by the call it sits in and the calls around it; the conversation
    only ever bore on whether the user SUPPLIED that value, which is an interface
    question this draw is no longer asked (#1824).  Handing it the ask again would put
    the same category error back.

    PUBLIC because the leaf-labelling eval builds it too: that case drives
    ``label_leaves`` alone, so it must render the same content production renders and
    not a copy that can drift."""
    parts = [f"Routine steps, in the order they ran:\n{render_skill(steps)}"]
    placeholder_lines = _placeholder_lines(steps, parameters)
    if placeholder_lines:
        parts.append(
            "Placeholders to name (each currently named after the tool argument it "
            f"fills):\n{placeholder_lines}"
        )
    return "\n\n".join(parts)


def build_framing_content(origin_message: str, conversation: list[tuple[str, str]]) -> str:
    """The skill framer's content (#1824): what the USER asked for, and nothing else.

    Only their turns are rendered — not the calls, not the values those calls carried,
    not the labeller's names.  What a skill IS and what it ASKS FOR are decided by what
    was asked for; the calls are how that ask was carried out this once, and every
    measured attempt to read the interface off them hit the same ceiling
    (#1821/#1823).  The demonstrating message joins the turns as the last one (deduped,
    since the recent window may already carry it).

    PUBLIC because the framing eval builds it too — same reason as its sibling."""
    incoming = PennyConstants.MessageDirection.INCOMING
    asks = [content for direction, content in conversation if direction == incoming]
    if origin_message and origin_message not in asks:
        asks.append(origin_message)
    return f"What the user asked for:\n{'\n'.join(asks)}"


def _naming(framing: SkillFraming | None, origin_message: str) -> tuple[str, str]:
    """The skill's name + description: the framer's, else the deterministic fallback —
    the slug of the triggering message plus that message (#1824).

    The framer owns both because it wrote them in the same decision as the parameters,
    so a skill cannot name itself for something it then asks for.  A failed draw is one
    rung down, never a block."""
    if framing is not None:
        return _slug_name(framing.name), framing.description
    fallback_name = _slug_name(origin_message)
    return fallback_name, origin_message or f"Skill: {fallback_name}"


def _placeholder_lines(steps: list[SkillStep], parameters: list[SkillParameter]) -> str:
    """One line per placeholder for the leaf-labelling content (#1824): its current
    arg-derived name, the value it held in the demonstration, and the tool-arg site(s)
    it fills — the facts a name for the spot is read off."""
    lines: list[str] = []
    for parameter in parameters:
        value, sites = _parameter_facts(steps, parameter.name)
        site_text = ", ".join(sites) if sites else "(unknown)"
        lines.append(f"- {parameter.name}: fills {site_text}; value that first time: {value!r}")
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


def slug_parameter_name(raw: str) -> str:
    """Harden a model-written semantic name into an identifier-safe binding key (#1668,
    load-bearing — the name is the params binding key at instantiation): lowercase,
    whitespace → underscores, strip anything but ``[a-z0-9_]``, trim stray underscores.
    Empty when nothing survives (the caller then keeps the arg-derived name — per-name
    fallback).

    PUBLIC because the leaf-labelling eval asserts a drawn name HARDENS to a usable
    key, and it must assert the shipped rule rather than a copy of it that can drift."""
    lowered = _PARAM_WHITESPACE.sub("_", raw.strip().lower())
    return _PARAM_NON_IDENTIFIER.sub("", lowered).strip("_")


def _apply_leaf_labels(
    steps: list[SkillStep],
    parameters: list[SkillParameter],
    leaves: LeafLabels | None,
) -> list[SkillStep]:
    """Turn EVERY candidate leaf into a named PLACEHOLDER (#1824).

    There is no role left to decide here.  A leaf is a spot in a tool call, the leaf
    labeller says what belongs in it, and what the SKILL asks for is the framer's
    answer to a different question — so this pass only ever carries a name and a
    description onto the leaves the distiller already found.

    Three fallbacks, each per leaf rather than all-or-nothing: a leaf the draw never
    covered takes its arg-derived name as its description, so the recipe still renders
    a legible ``{extract}`` instead of an empty slot; a leaf whose drawn name slugs to
    nothing keeps the arg-derived one; and an uncovered ATTACHMENT-MARKED leaf takes
    the fixed ``WRITE_TARGET_DESCRIPTION`` (#1777's string, kept as exactly that
    fallback) because what fills it is decided by where the routine is applied.  Marks
    and step-result bindings are untouched: both are structural facts about the calls,
    and no draw has a say in either."""
    drawn = _drawn_labels(leaves, parameters)
    return [_placeholder_step(step, drawn) for step in steps]


def _drawn_labels(
    leaves: LeafLabels | None, parameters: list[SkillParameter]
) -> dict[str, LeafPlaceholder]:
    """The WELL-FORMED label per candidate leaf, keyed by its arg-derived name — the
    ones the draw actually covered, so a leaf's absence from this map is exactly "no
    label came back for it" and the fallbacks read off one fact."""
    labels = leaves.placeholders if leaves is not None else {}
    offered = {parameter.name for parameter in parameters}
    return {
        current: label
        for current, label in labels.items()
        if current in offered and not is_blank(label.description)
    }


def _placeholder_step(step: SkillStep, drawn: dict[str, LeafPlaceholder]) -> SkillStep:
    """A copy of ``step`` with every ``HOLE`` substitution rewritten as a named
    PLACEHOLDER; bindings and every other substitution pass through untouched."""
    return step.model_copy(
        update={"substitutions": [_placeholder_leaf(sub, drawn) for sub in step.substitutions]}
    )


def _placeholder_leaf(
    sub: SkillSubstitution, drawn: dict[str, LeafPlaceholder]
) -> SkillSubstitution:
    """One substitution as a named placeholder — or unchanged, when it is not a
    candidate leaf (a step-result binding stays deterministic work).

    An unlabelled leaf falls back to its arg-derived name, which renders exactly as the
    recipe rendered before any draw; an unlabelled ATTACHMENT-MARKED one falls back to
    the fixed write-target wording instead, since what fills it is decided by where the
    routine is applied.  The mark rides along either way, for the render seam."""
    if sub.kind != SkillSubKind.HOLE or sub.parameter is None:
        return sub
    label = drawn.get(sub.parameter)
    fallback = WRITE_TARGET_DESCRIPTION if sub.attachment else sub.parameter
    return sub.model_copy(
        update={
            "kind": SkillSubKind.PLACEHOLDER,
            "parameter": None,
            "name": (slug_parameter_name(label.name) or sub.parameter) if label else sub.parameter,
            "description": label.description if label else fallback,
        }
    )


def _declared_parameters(framing: SkillFraming | None) -> list[SkillParameter]:
    """The skill's declared parameters — the FRAMER's answer, hardened (#1824).

    Every name is slugged into an identifier (it is the binding key at instantiation,
    ``params={name: value}``) and made unique; one that slugs to nothing is dropped,
    since a parameter nobody can name is not bindable.  A failed framing declares none:
    the skill still renders and still reads, it just asks for nothing until it is
    re-taught."""
    if framing is None:
        return []
    used: set[str] = set()
    parameters: list[SkillParameter] = []
    for drawn in framing.parameters:
        slug = slug_parameter_name(drawn.name)
        if not slug:
            continue
        name = _unique_name(slug, used)
        used.add(name)
        parameters.append(
            SkillParameter(name=name, required=True, description=drawn.description or None)
        )
    return parameters


def _unique_name(candidate: str, used: set[str]) -> str:
    """``candidate`` if unused, else ``candidate_2`` / ``candidate_3`` / … — parameter
    names are binding keys, so two must never collide (#1668)."""
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


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
