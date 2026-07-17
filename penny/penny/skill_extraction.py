"""Automatic skill extraction at chat-run end (#1658, epic #1554).

Skills are no longer model-authored.  There is no ``skill_create`` tool: at the
end of every qualifying CHAT run the framework distils a skill *deterministically*
from that run's own ledger rows — the same certified-by-execution snapshot the
retired tool produced, now fired by the run finishing instead of a model call.

``SkillExtractor.extract(run_id)`` is the whole pipeline, composed of named steps
(house style: the summary method reads like a table of contents):

* **qualify** — all structural, each a named check: the run is the chat agent's,
  it made ≥1 tool call, no text-bail nudge poisoned it, and its SUCCEEDED calls
  form a read+write taxonomy (a routine that senses AND acts).  A purely-read run
  (answering a question) and a purely-write run ('remember this' — the storage
  atom) do NOT qualify; failed calls are FILTERED, so a run whose only write
  failed is a pure read and is excluded.
* **distill** — ``distill_steps`` over the surviving (certified, non-``done``)
  steps: strips the framework ``reasoning`` leaf, excludes the retarget-owned
  write target, classifies bindings vs. required holes (#1659/#1660/#1662).
* **name** — a deterministic slug of the run's triggering message (URLs removed,
  ≤6 words); the full message stays the skill's description / find anchor.
* **dedup (REPLACE semantics)** — exact name match → REPLACE; else a same-shape,
  same-meaning skill → REPLACE keeping ITS name; otherwise insert.

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
from penny.database.memory.types import DedupThresholds
from penny.database.models import PromptLog, Skill
from penny.database.skill_store import steps_from_json
from penny.database.skills import DistillInput, SkillDraft, SkillStep, distill_steps
from penny.llm.similarity import embed_text
from penny.prompts import Prompt

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
    existing skill (by name, or same shape + meaning) was overwritten."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: Skill
    replaced: bool


class NoExtraction(BaseModel):
    """A run did NOT yield a skill — ``gate`` names which qualify check failed."""

    gate: ExtractionGate


SkillExtractionResult = SkillExtracted | NoExtraction


def _runnable_steps(projection: RunProjection) -> list[RunProjectionStep]:
    """Every non-``done`` step of the run (the demonstration's real tool calls)."""
    return [step for step in projection.steps if step.call.name != _DONE_TOOL]


def _certified_steps(projection: RunProjection) -> list[RunProjectionStep]:
    """The run's non-``done`` steps that SUCCEEDED — the routine that actually
    worked (#1659 filter-not-refuse).  Reads the structural per-call success stamp
    (``RunProjectionStep.success``, #1600): a step survives only when its stamp is
    exactly ``True`` (a recorded failure or a missing stamp is uncertain and left
    out), so certified-by-execution holds by construction."""
    return [step for step in _runnable_steps(projection) if step.success is True]


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

    def __init__(self, db: Database, embedding_client: LlmClient, *, agent_name: str) -> None:
        self._db = db
        self._embedding = embedding_client
        # The chat agent's name — the 'is this a chat run?' qualify anchor and the
        # author stamped on every extracted skill.
        self._agent_name = agent_name

    async def extract(self, run_id: str) -> SkillExtractionResult:
        """Extract a skill from one completed run's ledger rows — the summary method.

        Reads the run's prompts, projects them, runs the structural qualify gates,
        distils the surviving steps, names + dedups, and persists.  Returns the
        extracted skill or a typed no-extraction outcome naming the failed gate."""
        prompts = self._db.messages.get_run_prompts(run_id)
        projection = project_run(prompts)
        certified = _certified_steps(projection)
        gate = self._disqualify(prompts, projection, certified)
        if gate is not None:
            return NoExtraction(gate=gate)
        draft = self._draft(run_id, projection, certified)
        return await self._persist(draft)

    # ── Qualify (all structural) ──────────────────────────────────────────────

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

    def _draft(
        self,
        run_id: str,
        projection: RunProjection,
        certified: list[RunProjectionStep],
    ) -> SkillDraft:
        """Distil the certified steps into structured steps + holes, name the skill
        off the triggering message, and bundle it for the store."""
        steps, holes = distill_steps(self._distill_inputs(projection, certified))
        name = _slug_name(projection.origin_message)
        description = projection.origin_message or f"Skill: {name}"
        return SkillDraft(
            name=name,
            intent=description,
            description=description,
            steps=steps,
            holes=holes,
            source_run_id=run_id,
        )

    @staticmethod
    def _distill_inputs(
        projection: RunProjection, certified: list[RunProjectionStep]
    ) -> list[DistillInput]:
        """One ``DistillInput`` per certified step — its ordinal, tool, verbatim
        arguments, and framed result (``distill_steps`` reads the result to infer
        bindings; ``reasoning`` / write-target handling live inside it, #1659)."""
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

    async def _persist(self, draft: SkillDraft) -> SkillExtracted:
        """Embed the description, resolve the dedup target (name-or-shape+meaning),
        and upsert — REPLACE by name, so a re-demonstration of the same routine
        overwrites the prior skill in place."""
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
        return SkillExtracted(skill=skill, replaced=replaced)

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
