"""Automatic skill extraction at chat-run end (#1658, epic #1554).

Skills are no longer model-authored.  There is no ``skill_create`` tool: at the
end of every LEARN chat run the framework distils a skill *deterministically*
from that run's own ledger rows — the same certified-by-execution snapshot the
retired tool produced, now fired by the run finishing instead of a model call.

``SkillExtractor.extract(run_id, state=…)`` is the whole pipeline, composed of named
steps (house style: the summary method reads like a table of contents):

* **the state gate** — extraction runs if and only if the turn's landed conversation
  state is ``learn`` (#1850).  The state is the machine's own decision about THIS
  turn, made before the turn ran and threaded in as a parameter — this module never
  reads the machine, so what a run yields cannot depend on ambient state read after
  the fact.  Absence of machine history is idle, and idle does not learn.  Teaching
  is the one thing that mints a routine: an apply turn's extra enactment or an idle
  one-off that happens to browse-and-write used to qualify on shape alone and mint a
  skill from a round that taught nothing (the measured escape, PR #1849).
* **qualify** — what remains is MECHANICS, not shape requisites: the run is the chat
  agent's, it made ≥1 tool call, and ≥1 of its calls SUCCEEDED and is
  COLLECTOR-runnable.  There is NO read/write taxonomy any more (#1850): a learn run
  that only read and a learn run that only wrote are both routines the user just
  taught, and a routine's shape is not the framework's to judge.  Failed calls are
  FILTERED, so a run whose only write failed extracts the read that worked.
  Lifecycle calls a demo made (e.g. ``collection_set`` to set up the container) are
  dropped like orientation calls — a skill renders into a collector prompt, so only
  collector-runnable steps belong in it (#1668).  There is no health gate any more
  (#1839): the call-shaped-text bails it keyed on are discarded and re-rolled by the
  agent loop, so they never enter a completed run's rows at all — a recovered run is
  indistinguishable from a clean one BECAUSE recovery no longer writes into the run.
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
* **frame** — the routine's INTERFACE, written from the user's ask ALONE (#1830): a
  generic name, a one-line description, and one parameter per piece the user would have
  to say again to set the routine up on a new occasion.  It never sees the tool calls and
  the labeller never sees the ask, which is what stops the two halves contradicting each
  other (#1824).  Since #1868 it is normally READ rather than drawn — the framer runs when
  the machine ENTERS learn, so the routine was named before the round ran and this pass
  reuses that decision (a re-draw is not a re-read, and the turn was instructed under the
  entry name).  The run-end draw survives for a round nothing framed, and a failed framing
  still falls to the deterministic slug of the triggering message (URLs removed, ≤6 words)
  with nothing to bind.
  **Interim, declared**: nothing joins a framed parameter to a leaf of the rendered
  program yet (the runtime-join beat), so the parameters live at SKILL level — rendered
  in the registry, enforced at ``collection_set``, decisive for job identity — over a
  recipe that still reads in the labeller's placeholder descriptions.
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

import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from similarity.embeddings import cosine_similarity

from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState, RoundFraming
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
from penny.text_validity import is_blank
from penny.tools.micro_context import MicroContext, SkillLabels, SkillSignature

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# Registry-navigation verbs: the model uses these to ORIENT — resolve a skill or
# collection (``find``), read a skill's params (``skill_read``), inspect a
# collection's config (``memory_metadata``), or list the catalog
# (``collection_catalog``) — before it acts.  They are not part of the routine a
# skill captures (a re-run re-orients itself), and a ``find`` result ECHOES its
# query, which manufactured a FALSE binding when captured as a step (#1665).  So
# orientation calls are dropped from the distilled steps: what survives is the
# routine itself (browse, log_read, collection_read_latest, read_similar,
# collection_get, entry reads, and the writes).
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
    loggable gate failure (never a silent no-op)."""

    NOT_LEARN = "not_a_learn_turn"
    NOT_CHAT = "not_chat_run"
    NO_TOOL_CALLS = "no_tool_calls"
    NO_CERTIFIED_STEPS = "no_certified_steps"


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
    it is not a registry-navigation verb (``ORIENTATION_TOOLS`` — a re-run re-orients
    itself, so orientation is not part of the routine), AND its tool is one a COLLECTOR
    can run (``collector_surface`` — a skill renders into a collector prompt, so a
    lifecycle call the demo made mid-run, e.g. ``collection_set`` to set up the
    container, is dropped from the recipe; it's not a step a collector could run).  So
    certified-by-execution + routine-only + runnable hold by construction.

    The list this returns feeds the last quantity gate: a learn run that captured
    nothing is ``NO_CERTIFIED_STEPS`` rather than an empty skill, which is what #1839's
    honest learn-failure reply reads."""
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


def _naming(signature: SkillSignature | None, origin_message: str) -> tuple[str, str]:
    """What the routine is CALLED and what it is FOR — the framer's answer, else the
    deterministic slug of the triggering message (#1830).

    The framer's name is carried VERBATIM: it is prose the user reads back in the
    ambient registry, and resolving a skill by name is tolerant by construction
    (name-or-meaning, #1591).  Only a PARAMETER name is hardened, because that one is a
    key something binds against."""
    if signature is not None:
        return signature.name, signature.description
    name = _slug_name(origin_message)
    return name, origin_message or f"Skill: {name}"


def _interface_parameters(
    signature: SkillSignature | None, distilled: list[SkillParameter]
) -> list[SkillParameter]:
    """What the routine ASKS FOR — the framer's parameters, else the failure rung.

    The framer owns the interface, so its parameters ARE the skill's: minted from the
    ask, they name pieces of the user's own request rather than tool arguments, and
    they carry the one-line what-to-supply the ambient ``needs:`` row renders.

    **The declared interim of this beat**: nothing joins a framed parameter to a leaf
    of the rendered program yet — that is the runtime-join beat.  So the parameters
    live at SKILL level over an all-placeholder recipe: the registry row renders them,
    ``collection_set`` enforces that each one is bound, and job identity (``is_same_job``
    tier 1: same skill + same params) works again — while the program itself still
    reads in the labeller's descriptions.

    With no framing, the skill falls back to whatever the labelling pass left: nothing
    when every spot was named, and the arg-derived spots when that draw failed too."""
    if signature is None:
        return distilled
    return [
        SkillParameter(name=parameter.name, required=True, description=parameter.description)
        for parameter in signature.parameters
    ]


def attachment_names(db: Database) -> frozenset[str]:
    """The names of Penny's own COLLECTIONS — the things a routine can be ATTACHED
    to (#1783).  A demonstrated leaf holding one of these names is decided by the
    attachment, so distillation marks it and the render seam binds it to whatever
    collection the skill is applied to.

    Archived rows are included (a demonstration may have used one), logs are not: the
    attachment is always a collection, and re-pointing a log reference at one would
    produce a call the memory layer refuses.  Registry-derived — nothing here knows
    which tools a skill contains.

    PUBLIC because the elicit>learn eval decides whether a learned routine HAS a
    destination to mark, and that question is this same registry policy: a scorer
    restating it would be a second copy free to drift from the rule extraction marks
    on."""
    return frozenset(
        row.name for row in db.memories.list_all() if row.type == MemoryType.COLLECTION.value
    )


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

    async def extract(
        self,
        run_id: str,
        *,
        state: ConversationState | None,
        framing: RoundFraming | None = None,
    ) -> SkillExtractionResult:
        """Extract a skill from one completed run's ledger rows — the summary method.

        Checks the turn's landed state, then reads the run's prompts, projects them,
        runs the structural qualify gates, distils the surviving steps, names + dedups,
        and persists.  Returns the extracted skill or a typed no-extraction outcome
        naming the failed gate.

        ``state`` is what the conversation machine decided for THIS turn, threaded in
        by the caller (``None`` = no machine decided it, which is idle).  It is a
        parameter rather than a read of ``db.machine`` so extraction cannot disagree
        with the state the turn was actually run under.

        ``framing`` is the round's ENTRY framing (#1868), threaded in the same way: the
        interface was decided before the round ran, so this run-end pass READS it instead
        of drawing it again — a second draw over the same turns would be free to answer
        differently, and the name the turn was instructed under would stop being the name
        the skill is filed under.  Absent (nothing frames, or the entry draw failed), the
        framing is drawn here exactly as it was before the entry hook existed."""
        if state is not ConversationState.LEARN:
            return self._not_a_learn_turn(run_id, state)
        prompts = self._db.messages.get_run_prompts(run_id)
        projection = project_run(prompts)
        certified = _certified_steps(projection, self._collector_surface)
        gate = self._disqualify(prompts, projection, certified)
        if gate is not None:
            return NoExtraction(gate=gate)
        draft = await self._draft(run_id, projection, certified, framing)
        return await self._persist(draft, projection.origin_message)

    @staticmethod
    def _not_a_learn_turn(run_id: str, state: ConversationState | None) -> NoExtraction:
        """The state gate's refusal (#1850) — a NAMED outcome, logged with the state
        that produced it, never a silent skip.

        Teaching is the only thing that mints a routine, so every other turn declines
        here: an apply turn that enacts a skill (and browses and writes doing it), an
        idle one-off, a request turn negotiating what a skill needs.  The state carries
        the reason, which is why it is logged beside the gate — 'this run did not
        qualify' says nothing a reader can act on; 'this was an apply turn' does."""
        logger.debug(
            "No skill extracted from run %s (%s: the turn's state was %s)",
            run_id,
            ExtractionGate.NOT_LEARN,
            state,
        )
        return NoExtraction(gate=ExtractionGate.NOT_LEARN)

    def _disqualify(
        self,
        prompts: list[PromptLog],
        projection: RunProjection,
        certified: list[RunProjectionStep],
    ) -> ExtractionGate | None:
        """Run the ordered qualify gates; the FIRST failure's gate is returned
        (``None`` == qualifies).  Order: chat-run · has-calls · certified.

        Every one is MECHANICS — was this a chat run at all, did it do anything, did
        anything it did survive certification — never a judgment about the routine's
        SHAPE.  The read+write taxonomy that used to sit at the end of this list is
        gone (#1850): what makes a demonstration a routine is that the user was
        teaching it, which the state gate already settled."""
        if not self._is_chat_run(prompts):
            return ExtractionGate.NOT_CHAT
        if not _runnable_steps(projection):
            return ExtractionGate.NO_TOOL_CALLS
        if not certified:
            return ExtractionGate.NO_CERTIFIED_STEPS
        return None

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
        framing: RoundFraming | None,
    ) -> SkillDraft:
        """Distil the certified steps into structured steps + spots, label them, and
        bundle the result with the round's interface (#1824).

        The LABELLER names every spot in the implementation, from the demonstration; the
        INTERFACE — name, description, parameters — comes from the user's ask alone.  They
        share no evidence and no outputs, and either may be missing alone: a failed
        labelling leaves every spot with its arg-derived name, a missing interface leaves
        the routine slug-named with no parameters.  Extraction never blocks on either.

        Since #1868 the interface is normally the round's ENTRY framing, read rather than
        drawn — the routine was named before the round ran, and this is the same routine.
        The run-end draw survives as the path for a round nothing framed."""
        steps, parameters = distill_steps(
            self._distill_inputs(projection, certified), attachment_names(self._db)
        )
        conversation = self._db.messages.recent_conversation(_NAMING_CONVERSATION_TURNS)
        labels = await self._label_skill(steps, parameters, projection, conversation)
        signature = await self._signature(framing, projection, conversation)
        steps, parameters = _apply_leaf_labels(steps, parameters, labels)
        name, description = _naming(signature, projection.origin_message)
        return SkillDraft(
            name=name,
            intent=description,
            description=description,
            steps=steps,
            parameters=_interface_parameters(signature, parameters),
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
        there each run — poison-screened + re-rolled, its own ledger attribution.

        The spots' current names ride along as the COVERAGE set (#1828): the draw is
        accepted only when every one of them came back with a well-formed line, so a
        decayed tag costs the whole draw rather than one spot its label.  ``None`` then,
        and the caller keeps every spot's arg-derived name."""
        content = build_naming_content(steps, parameters, projection.origin_message, conversation)
        return await self._micro_context.label_skill(
            content, [parameter.name for parameter in parameters], run_target=self._agent_name
        )

    async def _signature(
        self,
        framing: RoundFraming | None,
        projection: RunProjection,
        conversation: list[tuple[str, str]],
    ) -> SkillSignature | None:
        """The routine's interface: the round's ENTRY framing when it has one, else a
        run-end framing draw (#1868).

        Reusing the entry framing is what makes the round coherent end to end — the turn
        was instructed under this routine's name and wrote into the container derived from
        it, so filing the skill under a name a second draw preferred would leave the
        container pointing at a routine that no longer exists by that name."""
        if framing is not None:
            return framing.signature
        return await self._frame_skill(projection, conversation)

    async def _frame_skill(
        self, projection: RunProjection, conversation: list[tuple[str, str]]
    ) -> SkillSignature | None:
        """One single-shot framing micro-context over the user's ask (#1830).

        Content = the round's user turns and nothing else, so the interface is decided
        from what they wanted rather than from what the round happened to do.  The draw
        writes the routine's name, its one-line description, and one parameter line per
        piece the user would have to say again — accepted only when it minted at least
        one well-formed, distinctly-named parameter.  ``None`` otherwise, and the caller
        falls back to the deterministic slug with nothing to bind."""
        content = build_framing_content(projection.origin_message, conversation)
        return await self._micro_context.frame_skill(content, run_target=self._agent_name)

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
        (the demonstrated-on instance) rides back for the narration frame (#1665).

        The embedding anchors on the FRAMER's description now (#1830) — a statement of
        the kind of task, which is what makes two demonstrations of the same routine on
        different occasions converge; before the framer landed this anchored on the
        triggering message itself, which is the occasion rather than the kind."""
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


def build_framing_content(origin_message: str, conversation: list[tuple[str, str]]) -> str:
    """The framer's content (#1830): the round's USER turns, one per line, and NOTHING
    else — no headings, no values, no summary of what was done.

    That emptiness is the contract.  What a routine is FOR, and what someone would have
    to say to set it running again, is answered by the ask; every other thing a round
    leaves behind is evidence of HOW it was carried out, and a draw shown that reasons
    from the mechanics instead (this is measured: a values list handed over with the ask
    argued four of five draws into treating every value as something to be supplied).
    The assistant's own turns go the same way — its replies describe what it did, which
    is exactly what a routine must not be named after.

    The demonstrating message joins the user's turns as the last one (deduped, since the
    recent window may already carry it): it is a user turn like the rest, and hiding who
    was speaking is what once made a draw reason correctly to the wrong answer (#1770).

    PUBLIC because the framing eval builds it: that case drives ``frame_skill`` alone,
    so its input must be rendered by THIS function rather than a copy that can drift."""
    incoming = PennyConstants.MessageDirection.INCOMING
    asks = [content for direction, content in conversation if direction == incoming]
    if origin_message and origin_message not in asks:
        asks.append(origin_message)
    return "\n".join(asks)


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


# ── Leaf labels: apply (#1770/#1828) ──────────────────────────────────────────


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
