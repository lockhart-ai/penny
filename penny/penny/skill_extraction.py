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
  (e.g. ``collection_create`` to set up the container) are dropped like orientation
  calls — a skill renders into a collector prompt, so only collector-runnable steps
  belong in it, and they count for nothing in the taxonomy (#1668).
* **distill** — ``distill_steps`` over the surviving (certified, non-``done``)
  steps: strips the framework ``reasoning`` leaf, excludes the retarget-owned
  write target, classifies bindings vs. required parameters (#1659/#1660/#1662).
* **name** — a GENERIC verb-noun label + a one-line generic description, written by
  a single-shot naming micro-context (#1665, the SECOND customer of the micro-context
  machinery) over the distilled routine — so a skill is named by its CONTRACT ("look
  up a price on a listing page and record it"), not by the instance ("read-the-aurora-
  deck-2-listing"), and cross-instance ``find`` can match it.  On ANY naming failure
  the fallback is the deterministic slug of the triggering message (URLs removed, ≤6
  words) + that message as the description — extraction NEVER blocks on the rewrite.
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
from penny.database.memory.types import DedupThresholds
from penny.database.models import PromptLog, Skill
from penny.database.skill_store import steps_from_json
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    distill_steps,
    render_skill,
)
from penny.llm.similarity import embed_text
from penny.prompts import Prompt
from penny.tools.micro_context import MicroContext, SkillLabel

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
    the demo made mid-run, e.g. ``collection_create`` to set up the container, is dropped
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

    async def _draft(
        self,
        run_id: str,
        projection: RunProjection,
        certified: list[RunProjectionStep],
    ) -> SkillDraft:
        """Distil the certified steps into structured steps + parameters, name the
        skill AND its parameters GENERICALLY via a single-shot micro-context (#1665/
        #1668, a verb-noun name + description + a semantic name/description per
        parameter over the rendered routine), and bundle it for the store.

        On ANY naming failure the fallback is the deterministic slug of the triggering
        message + that message as the description, and parameters keep their
        arg-derived names — the model writes LABELS only; steps/parameters are
        untouched otherwise, and extraction never blocks on the rewrite."""
        steps, parameters = distill_steps(self._distill_inputs(projection, certified))
        fallback_name = _slug_name(projection.origin_message)
        fallback_description = projection.origin_message or f"Skill: {fallback_name}"
        label = await self._label_skill(steps, parameters, projection)
        name = _slug_name(label.name) if label is not None else fallback_name
        description = label.description if label is not None else fallback_description
        steps, parameters = _apply_parameter_labels(steps, parameters, label)
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
    ) -> SkillLabel | None:
        """One single-shot naming micro-context over the rendered routine (#1665/#1668).

        Content = the numbered recipe with parameters as ``{variables}`` + the
        triggering message + the run's ``find`` query phrases + the parameter list
        (each parameter's current arg-derived name, demonstrated value, and the arg
        site(s) it fills); the micro-context writes a GENERIC name + description AND a
        semantic name/description per parameter (poison-screened + one reroll, its own
        ledger attribution).  ``None`` on any failure — the caller falls back to the
        slug + arg-derived names."""
        content = _naming_content(steps, parameters, projection)
        return await self._micro_context.label_skill(content, run_target=self._agent_name)

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


def _naming_content(
    steps: list[SkillStep], parameters: list[SkillParameter], projection: RunProjection
) -> str:
    """The naming micro-context's content (#1665/#1668): the numbered recipe
    (parameters as ``{variables}``, so the model treats them as user-supplied), the
    message that first demonstrated it, any ``find`` query phrases from the run (the
    generic task phrases the step-1 doctrine sends to find — a naming signal), and the
    parameter list — each parameter's current arg-derived name, demonstrated value,
    and the arg site(s) it fills — so the model can relabel each semantically."""
    parts = [
        f"Routine steps:\n{render_skill(steps)}",
        f"First demonstrated by this message:\n{projection.origin_message}",
    ]
    find_phrases = _find_phrases(projection)
    if find_phrases:
        parts.append("Search phrases used to look for a skill:\n" + "\n".join(find_phrases))
    param_lines = _parameter_lines(steps, parameters)
    if param_lines:
        parts.append(
            "Parameters (each currently named after the tool arg it fills):\n" + param_lines
        )
    return "\n\n".join(parts)


def _parameter_lines(steps: list[SkillStep], parameters: list[SkillParameter]) -> str:
    """One line per parameter for the naming content (#1668): its current name, the
    value it was demonstrated with, and the tool-arg site(s) it fills — the facts the
    model needs to give it a semantic name and description."""
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
) -> tuple[list[SkillStep], list[SkillParameter]]:
    """Relabel each parameter with its hardened semantic name + description and map the
    rename through every leaf site (#1668).  A parameter the label doesn't cover — or
    whose semantic name slugs to empty — keeps its arg-derived name (per-parameter
    fallback, not all-or-nothing); a collision gets a numeric suffix, since the name
    is the binding key.  ``label is None`` leaves everything untouched."""
    if label is None:
        return steps, parameters
    rename: dict[str, str] = {}
    used: set[str] = set()
    relabelled: list[SkillParameter] = []
    for parameter in parameters:
        param_label = label.parameters.get(parameter.name)
        candidate = _slug_parameter_name(param_label.name) if param_label is not None else ""
        final = _unique_name(candidate or parameter.name, used)
        used.add(final)
        rename[parameter.name] = final
        description = param_label.description if param_label and param_label.description else None
        relabelled.append(parameter.model_copy(update={"name": final, "description": description}))
    renamed_steps = [_rename_step_parameters(step, rename) for step in steps]
    return renamed_steps, relabelled


def _unique_name(candidate: str, used: set[str]) -> str:
    """``candidate`` if unused, else ``candidate_2`` / ``candidate_3`` / … — parameter
    names are binding keys, so two must never collide (#1668)."""
    if candidate not in used:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in used:
        suffix += 1
    return f"{candidate}_{suffix}"


def _rename_step_parameters(step: SkillStep, rename: dict[str, str]) -> SkillStep:
    """A copy of ``step`` with every ``HOLE`` substitution's ``parameter`` field
    remapped through ``rename`` (#1668) — so every leaf site follows its parameter to
    the semantic name, and the render substitutes by that name."""
    subs = [
        sub.model_copy(update={"parameter": rename[sub.parameter]})
        if sub.kind == SkillSubKind.HOLE and sub.parameter in rename
        else sub
        for sub in step.substitutions
    ]
    return step.model_copy(update={"substitutions": subs})


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
