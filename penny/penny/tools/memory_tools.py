"""Tool-layer wrappers over the memory access layer.

Every tool validates its kwargs through a Pydantic args model as its first
line (per CLAUDE.md), calls ``db.memories.*``, and returns a serializable
string the model can reason over.

Author attribution is passed explicitly: write-capable tools take an
``author: str`` at construction time (the agent that owns the tool).
``build_memory_tools(db, embedding_client, author)`` is the factory each
agent calls with its own ``self.name`` so writes are attributed correctly.

Tools that need embeddings (writes, similarity reads, ``exists``) take an
``LlmClient`` in ``__init__``. On a transient embed failure at write time the
write is REFUSED, not stored without a vector (#1412): every stored entry must
carry its similarity vector, so ``collection_write`` / ``log_append`` return an
actionable failure and persist nothing. Similarity reads return empty on the
same transient failure.
"""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from penny.config_params import RuntimeParams
from penny.constants import (
    WRITE_GATE_MUTATING_OUTCOMES,
    WRITE_GATE_STOP_REASONS,
    PennyConstants,
    WriteGateOutcome,
)
from penny.database import Database
from penny.database.memory import (
    DedupThresholds,
    EntryInput,
    LogEntryInput,
    Memory,
    MemoryAccessError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    ResolvedEntry,
    ResolvedHit,
    ResolvedKind,
    ResolvedMatch,
    WriteResult,
    render_key,
    render_run_calls,
    strip_display_brackets,
)
from penny.database.memory.types import slug
from penny.database.models import MemoryEntry, MemoryRow, Skill
from penny.database.mutation_store import render_mutation
from penny.database.skill_store import holes_from_json, steps_from_json
from penny.database.skills import render_skill, retarget_writes, unbound_required_holes
from penny.datetime_utils import format_log_timestamp
from penny.llm.similarity import embed_text
from penny.text_validity import check_extraction_prompt, check_extraction_prompt_tools
from penny.tools.base import Tool
from penny.tools.collection_instantiation import (
    SkillResolution,
    SkillResolutionKind,
    Trigger,
    TriggerError,
    parse_datetime,
    parse_trigger,
    render_active_duplicate,
    render_ambiguous,
    render_creation_echo,
    render_inert_echo,
    render_no_skill_found,
    render_reinstantiation_echo,
    render_tombstone_duplicate,
    render_trigger_clause,
    render_unbound_holes,
)
from penny.tools.memory_args import (
    CatalogArgs,
    CollectionCreateArgs,
    CollectionDeleteEntryArgs,
    CollectionEntrySpec,
    CollectionGetArgs,
    CollectionMergeArgs,
    CollectionUpdateArgs,
    CollectionWriteArgs,
    ExistsArgs,
    FindArgs,
    GetEventArgs,
    LogAppendArgs,
    LogCreateArgs,
    MemoryNameArgs,
    ReadLatestArgs,
    ReadLogArgs,
    ReadRandomArgs,
    ReadRunCallsArgs,
    ReadSimilarArgs,
    UpdateEntryArgs,
)
from penny.tools.models import NoArgs, ToolResult
from penny.tools.skill_tools import SkillReadTool

if TYPE_CHECKING:
    from penny.agents.collector import Collector
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


# ── Shared formatting ───────────────────────────────────────────────────────


def _format_entries(
    entries: list[MemoryEntry],
    *,
    source: str | None = None,
    ordering: str | None = None,
) -> str:
    """Render a list of entries as a numbered string the model can read.

    Leads with a ``N entries from `source` (ordering):`` header so the model
    reads the body as *fetched data* rather than a fresh instruction — the
    failure mode when a returned user message itself reads like a directive.
    ``source`` is the memory name; ``ordering`` is a human hint ("oldest
    first", "most relevant first") since the order differs per read tool and
    matters when the model concatenates entries.  Keyed entries (collection)
    render the key in **invocation form** — ``key='<key>'`` — so the displayed
    form IS the form a key-taking tool accepts: the model copies what it reads
    straight into a ``key=`` argument instead of the old ``[key]`` display,
    whose brackets it pasted verbatim into key args (``key="[key]"`` → "not
    found").  The key-taking tools still reject a bracket-wrapped key with a
    teaching error naming the bare key (``_bracket_key_rejection``) — the
    standing guard for that ingrained habit.  The timestamp keeps its ``[...]``
    brackets: it is never passed as an argument, so bracket framing there
    carries no copy-through hazard and now reads unambiguously as display
    metadata, distinct from the copyable key.  Keyless entries (log) show just
    content.  An empty read names the source and states it's empty (not an error),
    so the model doesn't confuse absence with a failure or re-read the same way.
    """
    if not entries:
        if source is None:
            return "(no entries)"
        return f"No entries in `{source}` — it's empty or nothing matched (not an error)."
    lines = []
    for index, entry in enumerate(entries, start=1):
        prefix = f"{render_key(entry.key)} " if entry.key else ""
        stamp = f"[{format_log_timestamp(entry.created_at)}] "
        lines.append(f"{index}. {stamp}{prefix}{entry.content}")
    body = "\n".join(lines)
    if source is None:
        return body
    noun = "entry" if len(entries) == 1 else "entries"
    suffix = f" ({ordering})" if ordering else ""
    return f"{len(entries)} {noun} from `{source}`{suffix}:\n{body}"


def _resolve(db: Database, name: str) -> Memory:
    """The ``Memory`` object for ``name``; raises ``MemoryNotFoundError`` when it
    doesn't exist.

    The single dispatch every content tool funnels through.  The ``MemoryTool``
    base turns any ``MemoryAccessError`` — missing (here), wrong shape, or
    read-only (raised by the object) — into the tool's readable result, so a
    tool body just resolves and operates, with no type check or try/except.
    """
    memory = db.memory(name)
    if memory is None:
        raise MemoryNotFoundError(name)
    return memory


_BRACKET_KEY_REJECTION = (
    "Key '{key}' not found in '{memory}'. The enclosing [brackets] are not part "
    "of the key — entry listings show keys as key='...' and the key is passed "
    "bare, without brackets. This entry's key is '{bare}'. Retry with key='{bare}'."
)


def _named(arguments: dict, arg_key: str, fallback: str) -> str:
    """Backtick-quoted value of a name-like argument for narration, or a
    caller-chosen fallback noun when the call omitted it (an arg-validation
    failure still narrates).  Tools name their target under different keys —
    ``memory``, ``name``, ``target``, ``collector`` — so the narration reads
    whichever key applies."""
    value = arguments.get(arg_key)
    return f"`{value}`" if value else fallback


def _memory_name(arguments: dict, fallback: str) -> str:
    """Backtick-quoted ``memory`` argument for narration (the common case)."""
    return _named(arguments, "memory", fallback)


def _entry_label(arguments: dict) -> str:
    """Quoted entry key for narration, or a generic noun when the call omitted it."""
    key = arguments.get("key")
    return f'"{key}"' if key else "an entry"


def _written_label(arguments: dict) -> str:
    """Name the write's single entry by key, or a count of several, for narration."""
    entries = arguments.get("entries", [])
    keys = [entry.get("key") for entry in entries if isinstance(entry, dict) and entry.get("key")]
    if len(keys) == 1:
        return f'"{keys[0]}"'
    if keys:
        return f"{len(keys)} entries"
    return "an entry"


def _bracket_key_rejection(memory: Memory, memory_name: str, key: str) -> ToolResult | None:
    """A teaching rejection when a missed key is the bracket-wrapped display
    form of an entry that exists.

    Entry lists used to render entries as ``[key] content`` and the model copied
    that display form — brackets included — into later key arguments; the render
    now shows keys in invocation form (``key='<key>'``), but this guard stays for
    the model's ingrained bracket habit.  Lookups stay
    strictly exact; the mistaken input is never absorbed by a silent retry
    (normalizing a hallucinated shape conforms the system to the model's error,
    and that compounds).  Instead the miss is diagnosed and rejected with the
    bare key named ready to reuse — the duplicate-write precedent: name the
    mistake and the model's next move.  Returns ``None`` when the key isn't
    bracket-wrapped, or when the bare form doesn't exist either — the ordinary
    not-found error stands.  A key that genuinely contains brackets exact-matches
    before any tool reaches this check, so it is never second-guessed.
    """
    bare = strip_display_brackets(key)
    if bare == key or not memory.get(bare):
        return None
    return ToolResult(
        message=_BRACKET_KEY_REJECTION.format(key=key, memory=memory_name, bare=bare),
        success=False,
    )


class MemoryTool(Tool):
    """Base for tools that operate on a memory.

    ``execute`` runs the subclass's ``_run`` and turns any memory exception into
    the tool's readable result — a ``MemoryAccessError`` (missing / wrong shape /
    read-only) or a ``MemoryAlreadyExistsError`` (create conflict).  Every such
    exception renders its own message, so the handling is uniformly
    ``return str(exc)`` and lives here once: a tool body just calls the op and
    lets the error propagate — no per-tool try/except, no format strings.
    """

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            return await self._run(**kwargs)
        except (MemoryAccessError, MemoryAlreadyExistsError) as exc:
            # Wrong-shape / read-only / missing-memory refusals are failed calls.
            return ToolResult(message=str(exc), success=False)

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult:
        """Resolve/create a memory and operate on it; let any memory exception
        propagate to :meth:`execute`."""


def _format_collection_echo(memory: Any, verb: str) -> str:
    """Render a created or updated collection as a structured echo.

    The chat agent uses this to confirm-back what landed without
    making up fields.  Includes the full extraction_prompt verbatim
    so the model's reply can summarize accurately (the model previously
    confabulated this because the create/update returns were one-liners).
    """
    return (
        f"{verb} collection '{memory.name}':\n"
        f"  trigger: {render_trigger_clause(memory)}\n"
        f"  notify: {memory.notify}\n"
        f"  description: {memory.description}\n"
        f"  extraction_prompt: |\n    "
        f"{(memory.extraction_prompt or '').replace(chr(10), chr(10) + '    ')}"
    )


# ── Description-anchor embed degradation (visible, self-healing) ─────────────
#
# A description doubles as the resolve-by-meaning anchor (#1558).  Unlike an entry
# write (which fails hard, #1412, because a vectorless entry is invisible to
# read_similar and corrupts dedup), a collection/log is still fully created or updated when its
# description embed fails transiently — only its meaning anchor is missing, and
# the startup description backfill re-embeds any ``NULL`` anchor.  So the
# create/update succeeds, but the degradation is NAMED in the result rather than
# left silent (visible-degradation): the anchor is unset until it self-heals.  No
# retry is demanded — the row already exists and retrying the create would only
# collide.
_DESCRIPTION_EMBED_DEGRADED = (
    " (Heads up: couldn't embed its description just now — a transient embedding "
    "error — so its meaning anchor is unset and it won't resolve via find "
    "until it self-heals on the next restart.)"
)


def _description_degraded_suffix(description: str | None, embedding: list[float] | None) -> str:
    """A visible-degradation note when a description was supplied but its embed failed.

    Empty (no note) unless a description was given *and* its embedding came back
    ``None`` — i.e. the anchor was left ``NULL`` for the backfill to re-heal."""
    if description is not None and embedding is None:
        return _DESCRIPTION_EMBED_DEGRADED
    return ""


# ── Provenance + lifecycle rendering (operational registry, #1566) ──────────
#
# ``collection_catalog`` (the inventory surface) and ``memory_metadata`` (the
# single exact-name lookup) render the SAME lifecycle block, so one read answers
# the whole "who asked for it, what run created it, is it live, when does it
# end" question.  It's sourced structurally from the registry columns, never
# reconstructed — so an archived mechanism stays enumerable and inspectable,
# marked with its archive time rather than vanishing from the catalog.

_STATUS_ACTIVE = "active"
_STATUS_ARCHIVED = "archived"
_EXPIRES_NEVER = "never"
# First N characters of the spawning message, rendered as "the ask" (#1566).
_ASK_EXCERPT_CHARS = 80


def _ask_excerpt(db: Database, source_message_id: int | None) -> str | None:
    """A one-line excerpt (first ``_ASK_EXCERPT_CHARS`` chars) of the user message
    that spawned a mechanism, or ``None`` when there is no source message (seeded
    / system rows) or the row can't be found.  Whitespace is collapsed so a
    multi-line ask reads as one line."""
    if source_message_id is None:
        return None
    message = db.messages.get_by_id(source_message_id)
    if message is None:
        return None
    collapsed = " ".join(message.content.split())
    if len(collapsed) <= _ASK_EXCERPT_CHARS:
        return collapsed
    return f"{collapsed[:_ASK_EXCERPT_CHARS].rstrip()}…"


def _status_line(row: MemoryRow) -> str:
    """``status: active`` or ``status: archived <UTC datetime>``.  A just-archived
    row renders clearly marked, never absent; the archive timestamp is
    ``updated_at``, which ``MemoryStore.archive`` stamps at archive time."""
    if row.archived:
        return f"status: {_STATUS_ARCHIVED} {format_log_timestamp(row.updated_at)}"
    return f"status: {_STATUS_ACTIVE}"


def _expires_line(row: MemoryRow) -> str:
    """``expires: <UTC datetime>`` when an end condition is set, else
    ``expires: never``."""
    if row.expires_at is not None:
        return f"expires: {format_log_timestamp(row.expires_at)}"
    return f"expires: {_EXPIRES_NEVER}"


def _created_line(row: MemoryRow, ask: str | None) -> str:
    """``created: <UTC datetime>`` plus the creating run and spawning message when
    they exist: ``… by run <run_id> from message <id> ("<ask>")``.  Seeded /
    system rows carry neither, so the line is just the timestamp."""
    parts = [f"created: {format_log_timestamp(row.created_at)}"]
    if row.created_by_run_id is not None:
        parts.append(f"by run {row.created_by_run_id}")
    if row.source_message_id is not None:
        if ask is not None:
            parts.append(f'from message {row.source_message_id} ("{ask}")')
        else:
            parts.append(f"from message {row.source_message_id}")
    return " ".join(parts)


def _lifecycle_block(db: Database, row: MemoryRow) -> list[str]:
    """The shared provenance + lifecycle lines — status, end condition, and the
    creating run/message — read from the registry columns.  One definition so
    the catalog and the metadata lookup answer the lifecycle question the same
    way."""
    return [
        _status_line(row),
        _expires_line(row),
        _created_line(row, _ask_excerpt(db, row.source_message_id)),
    ]


def _skill_provenance_line(row: MemoryRow) -> str | None:
    """``from skill: <name> (<param>=<value>, …)`` — the skill this collection was
    instantiated from (#1591's front door) and the params bound into its render, so
    the render names the recipe's origin.  The skill name is a live anchor: one
    ``skill_read(<name>)`` / ``find`` hop reaches its steps + holes (n≤1), and
    the bound params are the reachable input a future rebind/re-render consumes.
    Params are omitted when the skill had no holes (``from skill: <name>``).
    Returns ``None`` for a hand-authored / seeded collection (``skill_name`` NULL),
    so its render stays byte-identical to the unmarked pre-provenance shape — the
    unmarked case is the quiet default."""
    if row.skill_name is None:
        return None
    params: dict[str, str] = json.loads(row.skill_params) if row.skill_params else {}
    if not params:
        return f"from skill: {row.skill_name}"
    bound = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"from skill: {row.skill_name} ({bound})"


def _recent_changes_block(db: Database, name: str) -> list[str]:
    """The collection's recent config-change history from the mutation ledger
    (#1560) — each line naming its run id, so "when was this archived, and by
    what?" is a read, and the surface stays an anchor (the run is one hop away,
    never guessed).  Bounded by the shared ``RUN_HISTORY_RECORDS`` (the same
    recent-N used for a memory's run history), so no new limit is invented.  Empty
    (no block) for a collection with no recorded mutations (seeded / migration
    rows)."""
    events = db.mutations.history(name, PennyConstants.RUN_HISTORY_RECORDS)
    if not events:
        return []
    return ["", "Recent changes (newest first):", *(f"  {render_mutation(e)}" for e in events)]


# ── Metadata ────────────────────────────────────────────────────────────────


# A skill query whose embedding failed transiently — resolution couldn't run, so
# the create / re-render is refused with a retry (never a silent proceed to
# NO_SKILL_FOUND).  Tool-agnostic wording: shared by create (#1591) + update (#1620).
_SKILL_RESOLVE_EMBED_FAILURE = (
    "Couldn't resolve the skill '{query}' just now — a transient embedding error, so I "
    "can't tell whether a matching skill exists. Retry in a moment."
)


# ── Shared skill resolve + render (#1591 create / #1620 re-render) ───────────
#
# The create front door and the update re-render path resolve a skill the same
# way and render its steps the same way, so the machinery lives here once and both
# tools compose it (never duplicate it): a collection created from a skill and one
# re-rendered from it stamp a byte-identical prompt from identical steps + params.


async def resolve_skill(db: Database, llm_client: LlmClient, query: str) -> SkillResolution:
    """Resolve a ``skill`` arg by name-or-meaning (#1591): an exact name is a clean
    MATCHED; otherwise rank the registry by meaning — any positive candidate is
    AMBIGUOUS (never silently pick a fuzzy match), none is NO_SKILL_FOUND, a
    transient embed miss is EMBED_FAILED."""
    exact = db.skills.get(query)
    if exact is not None:
        return SkillResolution(kind=SkillResolutionKind.MATCHED, skill=exact)
    vec = await embed_text(llm_client, query)
    if vec is None:
        return SkillResolution(kind=SkillResolutionKind.EMBED_FAILED)
    candidates = db.skills.resolve_by_meaning(vec, PennyConstants.FIND_MATCH_LIMIT)
    if not candidates:
        return SkillResolution(kind=SkillResolutionKind.NO_SKILL_FOUND)
    return SkillResolution(kind=SkillResolutionKind.AMBIGUOUS, candidates=candidates)


def unresolved_skill_result(query: str, resolution: SkillResolution) -> ToolResult:
    """The enumerated result for a resolution that produced no skill: the AMBIGUOUS
    candidates, the NO_SKILL_FOUND elicitation, or the transient EMBED_FAILED retry
    — nothing is instantiated in any case."""
    if resolution.kind == SkillResolutionKind.AMBIGUOUS:
        return ToolResult(message=render_ambiguous(query, resolution.candidates), success=False)
    if resolution.kind == SkillResolutionKind.NO_SKILL_FOUND:
        return ToolResult(message=render_no_skill_found(query), success=False)
    return ToolResult(message=_SKILL_RESOLVE_EMBED_FAILURE.format(query=query), success=False)


def render_skill_prompt(
    db: Database, llm_client: LlmClient, skill: Skill, params: dict[str, str], target_name: str
) -> tuple[str, ToolResult | None]:
    """Validate the bound params against ``skill``'s holes, then render its steps +
    params into the numbered TEXT ``extraction_prompt`` for the collection ``target_name``.
    An unbound required hole → the actionable naming error; a rendered prompt that is too
    short or names an unrunnable tool → the same authoring-time rejection.  The re-render
    preserves the steps-1..A / no-stored-``done()`` invariant by construction — it is the
    same render fn ``collection_create`` stamps at birth.

    Every scoped-write step's ``memory`` argument is retargeted to ``target_name`` at
    this seam (#1629): applying a skill to a collection is what DEFINES where its writes
    land, so the demo-run target the skill baked in is replaced by the collection's own
    name — the rendered program never lies about its write target, on either the one-call
    create or the adopt path."""
    missing = unbound_required_holes(holes_from_json(skill.holes), params)
    if missing:
        return "", ToolResult(message=render_unbound_holes(skill.name, missing), success=False)
    steps = retarget_writes(steps_from_json(skill.steps), target_name)
    prompt = render_skill(steps, params)
    if (too_short := check_extraction_prompt(prompt)) is not None:
        return "", ToolResult(message=too_short, success=False)
    rejection = _reject_unknown_extraction_tools(db, llm_client, prompt)
    return prompt, rejection


def _once_form_interval() -> int:
    """The cadence a once-shaped (``run_at``) or on_advance trigger is paced at — the
    dispatcher tick from runtime config (eligible each tick, its real gate deciding when
    it runs), reused rather than a new invented default.  Shared by ``collection_create``
    and ``collection_update``'s trigger parse."""
    return int(RuntimeParams().COLLECTOR_TICK_INTERVAL)


def validate_source_log(db: Database, source_log: str | None) -> ToolResult | None:
    """For an on_advance trigger, ``source_log`` must name an existing LOG (#1604) — a
    collection can't be a frontier source, and a missing name would never advance.
    Returns an actionable ``ToolResult`` naming the problem + the fix, or ``None`` when
    the source is valid (or there's no on_advance).  Shared by create + update."""
    if source_log is None:
        return None
    source = db.memories.get(source_log)
    if source is None:
        return ToolResult(
            message=(
                f"on_advance source '{source_log}' isn't a memory I have — copy an exact log "
                "name from your store map (collection_catalog / find resolves one), then "
                "call the tool again."
            ),
            success=False,
        )
    if source.type != MemoryType.LOG.value:
        return ToolResult(
            message=(
                f"on_advance source '{source_log}' is a {source.type}, not a log — the "
                "trigger fires on a LOG advancing (an event stream). Name a log, or use "
                "interval for a recurring cadence."
            ),
            success=False,
        )
    return None


# A trigger / notify / expiry on a skill-less create: an inert collection has no job
# to describe, so the job-shaped arg is refused with the two-step fix (#1629).
_INERT_JOB_ARGS_REFUSAL = (
    "Can't set a trigger, notify, or expiry on '{name}' without a skill — those describe a "
    "JOB, and a skill-less collection is inert storage with no job to run. Create it as "
    "storage now (name + description only), then once you've taught the skill attach it with "
    "collection_update(name='{name}', skill=<title>, trigger=\"every <seconds>\", "
    "notify=<true/false>)."
)

# A skill create with no trigger: the collection wouldn't know when to run, so it's
# refused naming the three forms (a skill collection must be scheduled, #1631).
_MISSING_TRIGGER = (
    "A skill collection needs a trigger so it knows when to run. Set trigger to one of: "
    '"every <seconds>" (recurring), "once at <ISO> [xN]" (scheduled / one-shot), or '
    '"on advance of <log>" (wake when a source log advances).'
)


class CollectionCreateTool(MemoryTool):
    """Instantiate a collection from a skill, or set up an INERT storage container —
    the front door of collector creation (#1591, stage ⑤ of #1562; inert #1629).

    A collection is never authored with an inline procedure any more: it names a
    ``skill`` (resolved by name or meaning against the skill registry), binds the
    skill's parameter holes from ``params``, and the skill's steps RENDER into the
    collection's ``extraction_prompt`` at creation (a deterministic snapshot).  The
    resolution is an enumerated union — a clean name match instantiates; a fuzzy
    match returns ranked candidates to choose from; no match returns the teach-me
    elicitation.  Idempotency at birth (#1567) refuses a near-duplicate of an
    existing collection unless creation is made deliberate (``create_anyway``).
    """

    name = "collection_create"
    description = (
        "Set up a background collection. A collection is storage plus an OPTIONAL job.\n"
        "\n"
        "TWO ways to call it:\n"
        "- WITH a `skill`: instantiate that skill — its recipe becomes the collection's "
        "routine, run on the `trigger` you set. You don't write steps here; the skill "
        "supplies them.\n"
        "- WITHOUT a `skill`: an INERT storage collection — it holds entries but nothing "
        "runs against it yet. Use this to set up the container first when you're about "
        "to TEACH a skill: create it (name + description only), demonstrate the routine "
        "once here in chat — you learn it automatically as a skill — then attach that "
        "skill with collection_update to make it do the job. Don't pass a trigger / "
        "notify / expiry with a skill-less create — an inert collection has no job to "
        "schedule.\n"
        "\n"
        "Fields:\n"
        "- `name` — unique slug for the collection (lowercase, hyphens).\n"
        "- `description` — REQUIRED. What this collection is for, in the user's own words "
        '("keep an eye on the price of that jacket"). It is the collection\'s meaning '
        "anchor (how find resolves it), so get it right and confirm it back.\n"
        "- `skill` — OPTIONAL. The skill to instantiate, by exact name or a "
        'paraphrase of what it does ("watch a page for a change"). Omit it for an inert '
        "storage collection. If your paraphrase matches several skills I'll list them to "
        "choose from; if it matches none I'll walk you through teaching one.\n"
        "- `params` — a map binding the skill's holes to values "
        '(e.g. {"url": "https://…", "field": "price"}). Every REQUIRED hole '
        "must be bound or the call is refused naming what's missing.\n"
        '- `trigger` — ONE string, in one of three forms: "every <seconds>" (a recurring '
        'cadence, e.g. "every 3600" for hourly), "once at <ISO datetime> [xN]" (run at a '
        'time, N times — "once at 2026-07-20T09:00:00Z" is a one-time reminder that '
        'archives itself, "... x3" runs three times), or "on advance of <log>" (the '
        "collection wakes as soon as that source LOG gets a new entry — chain one "
        "collector off another's output). An unreadable trigger is refused naming the "
        "three forms.\n"
        "- `expires_at` — OPTIONAL. An ISO datetime end condition; the collection "
        "archives itself once it passes, so a bounded watch needs no teardown.\n"
        "- `notify` — Set `true` when the user wants to be told about / kept posted "
        "on / alerted to new or changed entries as they're found. Leave `false` (the "
        "default) for a silent collection they'll ask about later.\n"
        "\n"
        "Returns a structured echo of what landed (description, skill, bound params, "
        "trigger, notify, expiry, and the rendered routine). Confirm it back — don't "
        "invent fields it didn't return."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique collection name (slug-style: lowercase, hyphens)",
            },
            "description": {
                "type": "string",
                "description": (
                    "REQUIRED. What this collection is for, in the user's words — the goal "
                    'it serves ("a running list of good retro JRPGs to play"), not the '
                    "mechanism. The meaning anchor; confirm it back."
                ),
            },
            "skill": {
                "type": "string",
                "description": (
                    "OPTIONAL. The skill to instantiate — its exact name, or a paraphrase "
                    "of what it does (resolved by meaning). Omit for an inert storage "
                    "collection. A fuzzy match returns candidates to choose from; no match "
                    "walks you through teaching one."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Bindings for the skill's fill-in-the-blank holes: {hole: value}. Every "
                    "required hole must be bound."
                ),
            },
            "trigger": {
                "type": "string",
                "description": (
                    "One of three forms: 'every <seconds>' (recurring cadence, e.g. "
                    "'every 3600' hourly), 'once at <ISO datetime> [xN]' (run at a time, N "
                    "times — 'once at 2026-07-20T09:00:00Z' is a one-shot, '... x3' three "
                    "times), or 'on advance of <log>' (wake when a source log advances)."
                ),
            },
            "expires_at": {
                "type": "string",
                "description": (
                    "OPTIONAL ISO-8601 datetime end condition — the collection archives itself "
                    "when it passes."
                ),
            },
            "notify": {
                "type": "boolean",
                "description": (
                    "true = tell the user about new/changed entries (they asked to be told / "
                    "kept posted / alerted); false (default) = silent."
                ),
            },
            "create_anyway": {
                "type": "boolean",
                "description": (
                    "Reactive override — set only when a near-duplicate refusal says to."
                ),
            },
        },
        "required": ["name", "description"],
    }
    args_model = CollectionCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "name", "a new collection")
        if not result.success:
            return f"You tried to set up the {name} collection but it didn't work:"
        return f"You set up the {name} collection:"

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient,
        created_by_run_id: str | None = None,
    ) -> None:
        self._db = db
        self._llm_client = llm_client
        # The creating run's id (``promptlog.run_id``), passed as an explicit
        # parameter from the chat agent's turn context (#1566) — never ambient
        # state.  ``None`` for a collector / non-chat creator, which leaves the
        # column NULL.  ``source_message_id`` is NOT known at creation (the
        # channel logs the spawning message only after the run returns), so the
        # channel links it afterward by this run id — see
        # ``MemoryStore.link_source_message``.
        self._created_by_run_id = created_by_run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        """A skill-less create yields an INERT storage collection (#1629); a skill
        create instantiates that skill.  The two paths read as a table of contents."""
        args = CollectionCreateArgs(**kwargs)
        if args.skill is None:
            return await self._create_inert(args)
        return await self._create_from_skill(args, args.skill)

    async def _create_from_skill(self, args: CollectionCreateArgs, skill_query: str) -> ToolResult:
        """Instantiate a collection from a skill: parse the trigger, resolve the
        skill, bind + render its steps (writes retargeted to this collection, #1629),
        refuse a near-duplicate, then create and echo.  ``skill_query`` is ``args.skill``
        narrowed to non-``None`` by the caller."""
        parsed = self._parse_trigger(args)
        if isinstance(parsed, ToolResult):
            return parsed
        trigger, expires_at = parsed
        if source_error := validate_source_log(self._db, trigger.source_log):
            return source_error
        resolution = await resolve_skill(self._db, self._llm_client, skill_query)
        skill = resolution.skill
        if skill is None:
            # AMBIGUOUS / NO_SKILL_FOUND / EMBED_FAILED — nothing created.
            return unresolved_skill_result(skill_query, resolution)
        prompt, render_error = render_skill_prompt(
            self._db, self._llm_client, skill, args.params, slug(args.name)
        )
        if render_error is not None:
            return render_error
        return await self._instantiate(args, skill, prompt, trigger, expires_at)

    async def _create_inert(self, args: CollectionCreateArgs) -> ToolResult:
        """A skill-less create: an INERT collection (#1629) — storage only, no
        ``extraction_prompt`` / cadence / notify, so it never dispatches (the
        dispatcher selects on ``extraction_prompt IS NOT NULL``).  A job-shaped arg
        alongside is refused (an inert container has no job); idempotency-at-birth
        still applies; the echo is honest about being storage-only."""
        if job_error := self._reject_job_args(args):
            return job_error
        description_embedding = await embed_text(self._llm_client, args.description)
        if not args.create_anyway:
            dup = self._db.memories.find_duplicate_collection(args.name, description_embedding)
            if dup is not None:
                return self._duplicate_result(dup)
        memory = self._db.memories.create_collection(
            args.name,
            args.description,
            description_embedding=description_embedding,
            created_by_run_id=self._created_by_run_id,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        return ToolResult(message=f"{render_inert_echo(memory)}{suffix}", mutated=True)

    @staticmethod
    def _reject_job_args(args: CollectionCreateArgs) -> ToolResult | None:
        """A skill-less (inert) create describes storage, not a job — a trigger /
        notify / expiry passed alongside has nothing to attach to, so it's refused
        naming the fix (attach a skill via collection_update once one's taught) rather
        than silently dropped (visible degradation over silent success)."""
        has_job = any((args.trigger is not None, args.expires_at is not None, args.notify))
        if not has_job:
            return None
        return ToolResult(message=_INERT_JOB_ARGS_REFUSAL.format(name=args.name), success=False)

    def _parse_trigger(
        self, args: CollectionCreateArgs
    ) -> ToolResult | tuple[Trigger, datetime | None]:
        """Parse the single ``trigger`` arg + optional end condition before any skill
        work, so a bad schedule fails fast.  A skill collection needs a trigger (it must
        know when to run), so a missing one is refused naming the three forms; an
        unparseable one surfaces ``parse_trigger``'s teaching rejection verbatim."""
        if args.trigger is None:
            return ToolResult(message=_MISSING_TRIGGER, success=False)
        try:
            trigger = parse_trigger(args.trigger, _once_form_interval())
            expires_at = parse_datetime(args.expires_at, "expires_at") if args.expires_at else None
        except TriggerError as exc:
            return ToolResult(message=str(exc), success=False)
        return trigger, expires_at

    async def _instantiate(
        self,
        args: CollectionCreateArgs,
        skill: Skill,
        extraction_prompt: str,
        trigger: Trigger,
        expires_at: datetime | None,
    ) -> ToolResult:
        """Idempotency at birth (#1567), then create the collection and echo it.

        ``description`` is the routing/dedup meaning anchor; ``notify`` drives the
        run-time notify suffix on the collector (#1557 — the sole emission path since
        the notifier consumer was retired)."""
        description_embedding = await embed_text(self._llm_client, args.description)
        if not args.create_anyway:
            dup = self._db.memories.find_duplicate_collection(args.name, description_embedding)
            if dup is not None:
                return self._duplicate_result(dup)
        memory = self._db.memories.create_collection(
            args.name,
            args.description,
            extraction_prompt=extraction_prompt,
            collector_interval_seconds=trigger.collector_interval_seconds,
            description_embedding=description_embedding,
            notify=args.notify,
            created_by_run_id=self._created_by_run_id,
            expires_at=expires_at,
            run_at=trigger.run_at,
            max_runs=trigger.max_runs,
            # Record which skill rendered this collection and with what bindings, so
            # the catalog / metadata render can name it (#1603) — the substrate a
            # future rebind/re-render reads its current bindings from.
            skill_name=skill.name,
            skill_params=args.params,
            source_log=trigger.source_log,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        echo = render_creation_echo(memory, skill.name, args.params)
        return ToolResult(message=f"{echo}{suffix}", mutated=True)

    @staticmethod
    def _duplicate_result(dup: MemoryRow) -> ToolResult:
        """The idempotency refusal — the tombstone confirm-shape for an archived
        near-duplicate, the active-collection reuse refusal otherwise (#1567)."""
        if dup.archived:
            return ToolResult(message=render_tombstone_duplicate(dup), success=False)
        return ToolResult(message=render_active_duplicate(dup), success=False)


class LogCreateTool(MemoryTool):
    """Create a new append-only log."""

    name = "log_create"
    description = (
        "Create a new append-only log. Logs store keyless entries in time order "
        "and are meant for streams of events (messages, measurements, etc.). "
        "Provide a content-reflective description."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique log name"},
            "description": {
                "type": "string",
                "description": "Content-reflective one-line summary (the meaning anchor)",
            },
        },
        "required": ["name", "description"],
    }
    args_model = LogCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "name", "a new log")
        if not result.success:
            return f"You tried to set up the {name} log but it didn't work:"
        return f"You set up the {name} log:"

    def __init__(self, db: Database, llm_client: LlmClient) -> None:
        self._db = db
        self._llm_client = llm_client

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = LogCreateArgs(**kwargs)
        description_embedding = await embed_text(self._llm_client, args.description)
        self._db.memories.create_log(
            args.name,
            args.description,
            description_embedding=description_embedding,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        message = f"Created log '{args.name}'.{suffix}"
        return ToolResult(message=message, mutated=True)


class CollectionArchiveTool(MemoryTool):
    """Archive a collection — keeps its data but retires the mechanism."""

    name = "collection_archive"
    description = (
        "Archive a collection. The data stays intact but the collection is "
        "retired — its collector stops running and it drops out of the active "
        "memory list (it stays in the archived-inclusive catalog as a tombstone) "
        "until unarchived."
    )
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }
    args_model = MemoryNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        if not result.success:
            return f"You tried to archive {memory} but it didn't work:"
        return f"You archived {memory}:"

    def __init__(self, db: Database, run_id: str | None = None) -> None:
        self._db = db
        # The chat run archiving this collection — recorded as the mutation's
        # cause (#1560).  The scheduler's max_runs/expiry archive takes a
        # different path (actor=system); this tool is always a user-run archive.
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.archive(args.memory, run_id=self._run_id)
        return ToolResult(message=f"Archived '{args.memory}'.", mutated=True)


class CollectionUnarchiveTool(MemoryTool):
    """Restore a previously archived collection — its collector resumes."""

    name = "collection_unarchive"
    description = "Unarchive a previously archived collection."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }
    args_model = MemoryNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        if not result.success:
            return f"You tried to restore {memory} from the archive but it didn't work:"
        return f"You restored {memory} from the archive:"

    def __init__(self, db: Database, run_id: str | None = None) -> None:
        self._db = db
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.unarchive(args.memory, run_id=self._run_id)
        return ToolResult(message=f"Unarchived '{args.memory}'.", mutated=True)


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionGetTool(MemoryTool):
    """Exact-key lookup in a collection."""

    name = "collection_get"
    description = (
        "Look up an entry by its exact key in a collection. Returns the entry's "
        "content if found, or a 'not found' message otherwise."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "key": {"type": "string"},
        },
        "required": ["memory", "key"],
    }
    args_model = CollectionGetArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        entry = _entry_label(arguments)
        if not result.success:
            return f"You tried to look up {entry} in {memory} but it didn't work:"
        return f"You looked up {entry} in {memory}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionGetArgs(**kwargs)
        memory = _resolve(self._db, args.memory)
        rows = memory.get(args.key)
        if not rows:
            rejection = _bracket_key_rejection(memory, args.memory, args.key)
            if rejection is not None:
                return rejection
            return ToolResult(
                message=f"Key '{args.key}' not found in '{args.memory}'. List the available "
                f"keys with collection_keys('{args.memory}'), or search by content with "
                f"read_similar(memory='{args.memory}', anchor=<what you're looking for>). "
                f"If it exists under a different key, refresh that entry with "
                f"update_entry(key=<the key you found>, content=<the new content>) — "
                f"collection_write(memory='{args.memory}', entries=<the new key and content>) "
                f"creates NEW keys only and rejects an existing key as a duplicate."
            )
        return ToolResult(message=_format_entries(rows, source=args.memory))


class CollectionReadLatestTool(MemoryTool):
    """Return the newest entries in a collection, newest first.

    Collection-only: logs are read through the cursored ``log_read``, never
    through a newest-first scan (which would bypass the cursor and silently miss
    entries).  A log target gets a readable refusal.
    """

    name = "collection_read_latest"
    description = (
        "Return the newest entries in a collection, newest first. Omit `k` to "
        "return every entry. Collections only — to read a log use `log_read(<log>)`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory"],
    }
    args_model = ReadLatestArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "collection")
        if not result.success:
            return f"You tried to look up your {memory} but it didn't work:"
        return f"You looked up your {memory}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadLatestArgs(**kwargs)
        entries = _resolve(self._db, args.memory).read_latest(args.k)
        return ToolResult(
            message=_format_entries(entries, source=args.memory, ordering="most recent first")
        )


class CollectionReadRandomTool(MemoryTool):
    """Return entries sampled uniformly at random from a collection."""

    name = "collection_read_random"
    description = "Return `k` entries sampled uniformly at random. Omit `k` to return all."
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["memory"],
    }
    args_model = ReadRandomArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        if not result.success:
            return f"You tried to pull a random sample from {memory} but it didn't work:"
        return f"You pulled a random sample from {memory}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadRandomArgs(**kwargs)
        entries = _resolve(self._db, args.memory).read_random(args.k)
        return ToolResult(
            message=_format_entries(entries, source=args.memory, ordering="random sample")
        )


class ReadSimilarTool(MemoryTool):
    """Return entries most similar to an anchor phrase (collections or logs)."""

    name = "read_similar"
    description = (
        "Return entries from a memory ordered by content similarity to an "
        "`anchor` phrase. Works for both collections and logs — use this "
        "to find past conversations on a topic (search `user-messages` or "
        "`penny-messages`), past browse results, related preferences or "
        "facts, or any other historically-relevant entry."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "anchor": {
                "type": "string",
                "description": "Text whose meaning drives the similarity search",
            },
            "k": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory", "anchor"],
    }
    args_model = ReadSimilarArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "your memory")
        anchor = arguments.get("anchor")
        if not result.success:
            return f"You tried to search {memory} but it didn't work:"
        if anchor:
            return f'You searched {memory} for "{anchor}":'
        return f"You searched {memory}:"

    def __init__(self, db: Database, llm_client: LlmClient) -> None:
        self._db = db
        self._llm = llm_client

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadSimilarArgs(**kwargs)
        vec = await embed_text(self._llm, args.anchor)
        if vec is None:
            logger.warning("%s: anchor embedding failed transiently", self.name)
            return ToolResult(
                message="Couldn't embed the anchor (a transient embedding error). "
                "Retry, or read a collection with collection_read_latest(<collection>) "
                "or a log with log_read(<log>) instead.",
                success=False,
            )
        entries = _resolve(self._db, args.memory).read_similar(vec, args.k)
        return ToolResult(
            message=_format_entries(entries, source=args.memory, ordering="most relevant first")
        )


class CollectionKeysTool(MemoryTool):
    """List the unique keys currently in a collection."""

    name = "collection_keys"
    description = "List the unique keys in a collection (insertion order)."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }
    args_model = MemoryNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        if not result.success:
            return f"You tried to list the keys in {memory} but it didn't work:"
        return f"You listed the keys in {memory}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        keys = _resolve(self._db, args.memory).keys()
        if not keys:
            return ToolResult(
                message=f"No keys in `{args.memory}` — the collection is empty (not an error)."
            )
        return ToolResult(message="\n".join(f"- {key}" for key in keys))


# ── Collection writes ───────────────────────────────────────────────────────


_SCOPE_REFUSAL_MESSAGE = "Refused: this collector can only write to '{scope}', not '{memory}'."

# A duplicate rejection binds the matched existing key straight into an
# ``update_entry`` call PER ENTRY (see ``_format_duplicate``), not as a placeholder —
# so the model refreshes the row that already exists instead of re-using its OWN
# rejected key and ping-ponging on key-not-found.  Merely naming the matched key in
# parentheses left the embedding-match arm recovering at only 47% (#1405).  These
# trailing closes only frame how the cycle may END; the actionable call lives per
# entry.
_DUPLICATE_CLOSE_SOME = "Refresh with richer info, or skip these."
# All proposed entries were duplicates — nothing new landed.  The collector variant
# names ``done()`` (its close tool); the chat variant must NOT (chat has no ``done``).
_DUPLICATE_CLOSE_ALL_COLLECTOR = (
    "Nothing new to add — refresh one of the above with richer info, "
    "otherwise call done() to close the cycle."
)
_DUPLICATE_CLOSE_ALL_CHAT = (
    "Nothing new to add this time — refresh one of the above only if you have richer info."
)


def _format_duplicate(result: WriteResult) -> str:
    """Bind the matched existing key into an ``update_entry`` imperative for this
    rejected candidate — name the collision AND the exact next call to make, so the
    model refreshes the existing row instead of re-using its own rejected key and
    ping-ponging on key-not-found (the 47%-recovery embedding-match arm, #1405).

    When the matched entry is keyless (``matched_key`` missing) there is no key
    to bind an ``update_entry`` call to, so the honest next move is to skip it or
    write distinct content — named as such rather than dangling an imperative the
    model can't act on."""
    if result.matched_key:
        collision = (
            f"'{result.key}' already exists"
            if result.matched_key == result.key
            else f"'{result.key}' duplicates existing '{result.matched_key}'"
        )
        return (
            f"{collision} — call update_entry(key='{result.matched_key}', "
            f"content=<richer info>) to refresh it"
        )
    return (
        f"'{result.key}' duplicates an existing keyless entry — no key to update; "
        f"skip it or write distinct content"
    )


def _format_changed(results: list[WriteResult]) -> str:
    """The change-gate CHANGED part (#1587/#1633): the exact key already existed with
    a DIFFERENT value — the observed value changed, so the write gate auto-refreshed
    the stored baseline IN PLACE (through the shared update path, stamping the writing
    run).  Nothing further is needed — no ``update_entry`` call (the refresh already
    happened); naming it just keeps the next observation of the same value reading as
    UNCHANGED."""
    keys = ", ".join(f"'{r.key}'" for r in results)
    noun = "entry" if len(results) == 1 else "entries"
    return f"Changed: {keys} — the stored baseline was refreshed to the new value ({noun})."


def _format_unchanged(results: list[WriteResult]) -> str:
    """The change-gate UNCHANGED part (#1587): the exact key already holds this exact
    value, so there is nothing to update — the watch's "no change" signal (which, on
    a collector-scoped write, also STOPs the run at the chokepoint)."""
    keys = ", ".join(f"'{r.key}'" for r in results)
    noun = "entry" if len(results) == 1 else "entries"
    return (
        f"Unchanged: {keys} already holds the same value — no change since the last write ({noun})."
    )


def _format_written(memory: str, keys: list[str]) -> str:
    """The NEW_KEY part: the entries that actually landed (new keys)."""
    noun = "entry" if len(keys) == 1 else "entries"
    return f"Wrote {len(keys)} {noun} to '{memory}': {', '.join(keys)}."


def _format_degenerate(results: list[WriteResult]) -> str:
    """The DEGENERATE part: content rejected as degenerate, with the remedy."""
    labelled = ", ".join(f"{r.key} ({r.reason})" for r in results)
    return (
        f"Rejected as degenerate content: {labelled}.  Re-write these with substantive "
        f"descriptive text (not a bare URL, punctuation, or a bail-out phrase)."
    )


def _format_unexpected(results: list[WriteResult]) -> str:
    """The UNEXPECTED escape part (#1587): a write the change-gate could not classify
    — surfaced for review rather than forced into a wrong box or silently dropped
    (the visible-degradation principle).  Unreachable today (the write path is
    total), but wired end-to-end so the escape label is honest if the union grows."""
    keys = ", ".join(f"'{r.key}'" for r in results)
    return f"Unclassified write outcome for {keys} — flagged for review (this shouldn't happen)."


# ── Embed-failure at write time (fail-hard, #1412) ──────────────────────────
#
# Every stored entry MUST carry its similarity vector: an entry without one is
# invisible to ``read_similar`` (which skips it) and silently weakens dedup.  So
# a transient embed failure at write time REFUSES the write outright rather than
# persisting a vectorless row and reporting an optimistic success (the
# visible-degradation principle — a failed capability produces honest state, not
# a hidden one).  ``embed_text`` already retries ``llm_max_retries`` times before
# returning ``None``, so ``None`` means the embedding service was unavailable
# across every attempt; the failure is actionable and binds a retry — the only
# correct move for a transient outage.
#
# No data-loss on the append path: load-bearing capture (user/agent messages,
# browse results, collector runs) is written by Python channel paths that
# tolerate a missing vector via the startup backfill — never by these tools.  The
# ``AppendableLogName`` / ``SYSTEM_LOGS`` gate keeps ``log_append`` off those
# logs, so failing here can only affect a model-driven append to a user-created
# log, where the model simply retries.
_EMBED_WRITE_FAILURE_COLLECTION = (
    "Couldn't embed {keys} (a transient embedding error) — nothing was written, "
    "since an entry is only stored once it has a similarity vector. Retry "
    "collection_write(memory='{memory}', entries=<the same entries>) in a moment."
)
_EMBED_WRITE_FAILURE_LOG = (
    "Couldn't embed this entry (a transient embedding error) — nothing was "
    "appended, since an entry is only stored once it has a similarity vector. "
    "Retry log_append(memory='{memory}', content=<the same content>) in a moment."
)


class CollectionWriteTool(MemoryTool):
    """Write entries to a collection with similarity-based dedup."""

    name = "collection_write"
    description = (
        "Write one or more entries to a collection. Each entry has a short "
        "`key` (topic/identifier) and a longer `content` body. Dedup runs "
        "per entry — duplicates are reported but not treated as errors."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["key", "content"],
                },
            },
        },
        "required": ["memory", "entries"],
    }
    args_model = CollectionWriteArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        if not result.success:
            return f"You tried to save to {memory} but it didn't work:"
        if not result.mutated:
            return f"You didn't add anything new to {memory} — it was already there:"
        return f"You saved {_written_label(arguments)} to {memory}:"

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient,
        author: str,
        scope: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author
        self._scope = scope
        # The writing run — stamped on each new entry (created/last-written) so a
        # stored value cites the run that produced it (#1560).
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionWriteArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return ToolResult(
                message=_SCOPE_REFUSAL_MESSAGE.format(scope=self._scope, memory=args.memory),
                success=False,
            )
        memory = _resolve(self._db, args.memory)
        entries = [await self._build_entry(spec) for spec in args.entries]
        embed_failure = self._embed_failure(args.memory, entries)
        if embed_failure is not None:
            return embed_failure
        results = memory.write(entries, author=self._author, run_id=self._run_id)
        return self._format_results(args.memory, results)

    def _embed_failure(self, memory: str, entries: list[EntryInput]) -> ToolResult | None:
        """Refuse the write, atomically, if any entry lost a vector to a transient
        embed failure — a vectorless entry is invisible to read_similar and
        dedup-weakening, so nothing is persisted and the model retries once embedding
        recovers
        (fail-hard, #1412)."""
        missing = [
            entry.key
            for entry in entries
            if entry.content_embedding is None or entry.key_embedding is None
        ]
        if not missing:
            return None
        keys = ", ".join(f"'{key}'" for key in missing)
        logger.warning("collection_write: embedding unavailable for %s in %s", keys, memory)
        return ToolResult(
            message=_EMBED_WRITE_FAILURE_COLLECTION.format(keys=keys, memory=memory),
            success=False,
        )

    async def _build_entry(self, spec: CollectionEntrySpec) -> EntryInput:
        return EntryInput(
            key=spec.key,
            content=spec.content,
            key_embedding=await embed_text(self._llm, spec.key),
            content_embedding=await embed_text(self._llm, spec.content),
        )

    def _format_results(self, memory: str, results: list[WriteResult]) -> ToolResult:
        """Compose the write's model-facing result from the change-gate outcomes.

        Each entry carries one ``WriteGateOutcome`` (#1587); this buckets them, logs
        the rejections, and renders one part per non-empty bucket.  ``mutated`` is
        true when any entry changed durable state — a NEW_KEY write OR a
        KEY_EXISTS_CHANGED baseline auto-refresh (#1633); an all-duplicate /
        unchanged / degenerate batch changed nothing, so it reads as no-work for the
        throttle.  ``stop`` carries the write-gate STOP (collector context only),
        honored by the collector loop."""
        by = self._bucket(results)
        self._log_rejections(memory, by)
        parts = self._message_parts(memory, by)
        message = " ".join(parts) if parts else "(no entries written)"
        return ToolResult(
            message=message,
            mutated=any(result.outcome in WRITE_GATE_MUTATING_OUTCOMES for result in results),
            stop=self._stop_outcome(results),
        )

    @staticmethod
    def _bucket(results: list[WriteResult]) -> dict[WriteGateOutcome, list[WriteResult]]:
        """Group results by their change-gate outcome (every member present, so a
        lookup is total and no branch guesses an empty bucket)."""
        by: dict[WriteGateOutcome, list[WriteResult]] = {o: [] for o in WriteGateOutcome}
        for result in results:
            by[result.outcome].append(result)
        return by

    @staticmethod
    def _log_rejections(memory: str, by: dict[WriteGateOutcome, list[WriteResult]]) -> None:
        if duplicates := by[WriteGateOutcome.DUPLICATE]:
            logger.info(
                "collection_write: %d duplicate(s) rejected in %s: %s",
                len(duplicates),
                memory,
                ", ".join(r.key for r in duplicates),
            )
        if degenerate := by[WriteGateOutcome.DEGENERATE]:
            logger.info(
                "collection_write: %d degenerate entry(ies) rejected in %s: %s",
                len(degenerate),
                memory,
                ", ".join(f"{r.key!r} ({r.reason})" for r in degenerate),
            )

    def _message_parts(
        self, memory: str, by: dict[WriteGateOutcome, list[WriteResult]]
    ) -> list[str]:
        """One part per non-empty outcome bucket, composed in a fixed order — a
        table of contents over the per-outcome formatters."""
        written = [r.key for r in by[WriteGateOutcome.NEW_KEY]]
        changed = by[WriteGateOutcome.KEY_EXISTS_CHANGED]
        unchanged = by[WriteGateOutcome.KEY_EXISTS_UNCHANGED]
        duplicates = by[WriteGateOutcome.DUPLICATE]
        degenerate = by[WriteGateOutcome.DEGENERATE]
        # "Nothing landed" = the batch was ONLY duplicates: the close then names the
        # cycle-ending move (done() for a collector), not just per-entry refreshes.
        nothing_landed = not (written or changed or unchanged or degenerate)
        unexpected = by[WriteGateOutcome.UNEXPECTED]
        parts = [
            _format_written(memory, written) if written else None,
            _format_changed(changed) if changed else None,
            _format_unchanged(unchanged) if unchanged else None,
            self._duplicate_part(duplicates, all_duplicates=nothing_landed) if duplicates else None,
            _format_degenerate(degenerate) if degenerate else None,
            _format_unexpected(unexpected) if unexpected else None,
        ]
        return [part for part in parts if part]

    def _duplicate_part(self, duplicates: list[WriteResult], *, all_duplicates: bool) -> str:
        """The DUPLICATE part: per-entry ``update_entry`` binds + the trailing close."""
        labelled = "; ".join(_format_duplicate(r) for r in duplicates)
        close = self._duplicate_close(all_duplicates=all_duplicates)
        return f"Rejected as duplicates: {labelled}.  {close}"

    def _stop_outcome(self, results: list[WriteResult]) -> WriteGateOutcome | None:
        """The write-gate STOP for this call — honored only in a collector (must-act)
        context, so chat gets the enumerated text but never a loop-stop (#1587).

        Fires when the whole write resolved to a single STOP-worthy outcome (nothing
        new landed): a watch's unchanged re-observation.  Which outcomes are
        STOP-worthy is the declared ``WRITE_GATE_STOP_REASONS`` table (data), so
        later stages extend the table, not this code."""
        if self._scope is None or not results:
            return None
        outcomes = {r.outcome for r in results}
        if len(outcomes) == 1:
            only = next(iter(outcomes))
            if only in WRITE_GATE_STOP_REASONS:
                return only
        return None

    def _duplicate_close(self, *, all_duplicates: bool) -> str:
        """The trailing framing after the per-entry rejections (each already binds its
        own ``update_entry`` call).  When the whole batch was duplicates, a collector
        (``scope`` set) may close with ``done()``; the chat agent (``scope`` is
        ``None``) has no ``done`` tool, so it gets a variant that never names one."""
        if not all_duplicates:
            return _DUPLICATE_CLOSE_SOME
        if self._scope is not None:
            return _DUPLICATE_CLOSE_ALL_COLLECTOR
        return _DUPLICATE_CLOSE_ALL_CHAT


class UpdateEntryTool(MemoryTool):
    """Replace the content of an existing entry in a collection."""

    name = "update_entry"
    description = (
        "Replace the content of an existing entry in a collection, identified "
        "by key. Returns an error if the key doesn't exist."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "description": "Collection name"},
            "key": {"type": "string", "description": "Entry key within the collection"},
            "content": {
                "type": "string",
                "description": "New content to replace the existing entry",
            },
        },
        "required": ["memory", "key", "content"],
    }
    args_model = UpdateEntryArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        entry = _entry_label(arguments)
        if not result.success:
            return f"You tried to update {entry} in {memory} but it didn't work:"
        if not result.mutated:
            return f"You couldn't find {entry} to update in {memory}:"
        return f"You updated {entry} in {memory}:"

    def __init__(
        self, db: Database, author: str, scope: str | None = None, run_id: str | None = None
    ) -> None:
        self._db = db
        self._author = author
        self._scope = scope
        # The rewriting run — advances the entry's last_written_by_run_id (#1560).
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = UpdateEntryArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return ToolResult(
                message=_SCOPE_REFUSAL_MESSAGE.format(scope=self._scope, memory=args.memory),
                success=False,
            )
        memory = _resolve(self._db, args.memory)
        outcome = memory.update(args.key, args.content, self._author, run_id=self._run_id)
        if outcome == "not_found":
            rejection = _bracket_key_rejection(memory, args.memory, args.key)
            if rejection is not None:
                return rejection
            return ToolResult(
                message=f"Key '{args.key}' not found in '{args.memory}' — update only replaces "
                f"existing entries. Write it as a new entry with "
                f"collection_write(memory='{args.memory}', entries=<the new key and content>), "
                f"or list the current keys with collection_keys('{args.memory}') if you "
                f"expected it to exist."
            )
        return ToolResult(message=f"Updated '{args.key}' in '{args.memory}'.", mutated=True)


# ── Re-render actionable refusals (#1620) ────────────────────────────────────

# A raw extraction_prompt passed alongside skill/params: the render owns the prompt,
# so the two are mutually exclusive rather than one silently winning.
_REINSTANTIATE_CONFLICT = (
    "Can't set extraction_prompt AND skill/params in the same call — re-rendering from "
    "a skill REPLACES the prompt with its rendered steps, so an extraction_prompt you "
    "pass would be discarded. Either re-render from the skill (skill= / params=) OR edit "
    "the prompt text directly (extraction_prompt=), not both."
)

# params-only rebind on a collection that was never instantiated from a skill (a
# legacy hand-authored one): there are no holes to bind — name how to adopt one.
_REBIND_NO_SKILL = (
    "Can't rebind params on '{name}': it wasn't instantiated from a skill (it's "
    "hand-authored), so it has no parameter holes to bind. Pass skill=<name> to adopt a "
    "skill onto it, or edit its recipe directly with extraction_prompt=<the full body>."
)

# The collection's pinned skill name no longer resolves (deleted / renamed) — a
# refresh/rebind can't re-render, so name the recovery (re-teach, or point elsewhere).
_SKILL_GONE = (
    "The skill '{skill}' this collection was built from no longer exists — it may have "
    "been renamed or removed. Re-teach it (then it re-renders), or pass skill=<name> to "
    "re-render this collection from a different skill."
)

# An adopt that attached a skill but no trigger: the collection has a routine yet no
# cadence, so it won't dispatch until one is set (#1629 — visible over silent).
_NO_TRIGGER_NOTE = (
    "\n\nHeads up: this collection now has a routine but no trigger, so it won't run yet. "
    "Set one with collection_update(name='{name}', trigger=\"every <seconds>\") (or "
    '"once at <ISO> [xN]", or "on advance of <log>").'
)


def _current_skill_params(row: MemoryRow) -> dict[str, str]:
    """The params currently bound into a collection's skill render (JSON on the row),
    or ``{}`` when it has none — the refresh/rebind default when no new params are
    passed, so a plain ``skill=<same>`` refresh keeps the existing bindings."""
    return json.loads(row.skill_params) if row.skill_params else {}


class CollectionUpdateTool(MemoryTool):
    """Update collection metadata, or re-render its prompt from a skill (#1620).

    Chat-facing.  Lets the user evolve a collection mid-conversation — refining
    the description/cadence, flipping notify, or re-rendering the ``extraction_prompt``
    from a skill: **refresh** (the same skill re-taught → re-render from its current
    steps), **rebind** (new ``params``, same skill), **swap** (a different ``skill``),
    or **adopt** (give a legacy skill=NULL collection a skill for the first time).  All
    fields except ``name`` are optional; only the ones supplied are applied.  Supplying
    ``skill`` or ``params`` takes the re-render path (which re-stamps skill provenance
    and records a mutation event); omitting both leaves the prompt untouched.
    """

    name = "collection_update"
    description = (
        "Update an existing collection's metadata. Only supplied fields "
        "are changed.\n"
        "\n"
        "Fields:\n"
        "- `name` (required) — the collection to update.\n"
        "- `description` — content-reflective one-line summary AND the "
        "meaning anchor (find / resolve-by-meaning). Changing it "
        "re-embeds it, so keep it an accurate summary of the subject "
        "matter. It does not drive the collector — change the "
        "extraction_prompt for that.\n"
        "- `notify` — flip notify-on-new. `true` starts telling the "
        "user about new entries (they asked to be kept posted / alerted); "
        "`false` silences it (the collector keeps gathering). Omit to leave "
        "unchanged.\n"
        "- `extraction_prompt` — FULL replacement body, not a diff. "
        "Drives what the collector actually does. Read the current body "
        "via `memory_metadata(<collection>)` first if you need to preserve any "
        "of it. Don't combine with `skill`/`params` — a re-render owns the prompt.\n"
        "- `skill` — re-render the collection's routine from a skill's CURRENT "
        "steps (by exact name or a paraphrase, resolved the same way as "
        "`collection_create`). Use the SAME skill name to refresh after re-teaching "
        "it, or a DIFFERENT skill to swap. On a legacy hand-authored collection this "
        "adopts a skill for the first time (its old text is replaced by the render).\n"
        "- `params` — rebind the skill's fill-in-the-blank holes to new values and "
        "re-render, keeping the same skill. Omit to keep the current bindings. Every "
        "required hole must be bound or the call is refused naming what's missing.\n"
        "- `trigger` — the job's schedule as ONE string (set it to change the schedule, "
        'else omit to leave it): "every <seconds>" (recurring, e.g. "every 3600"), '
        '"once at <ISO datetime> [xN]" (run at a time, N times), or "on advance of <log>" '
        "(wake when a source log advances). Setting it REPLACES the whole schedule. This "
        "is how you give an INERT collection its cadence when you adopt a skill onto it.\n"
        "- `expires_at` — OPTIONAL ISO datetime end condition; the collection archives "
        "itself once it passes.\n"
        "\n"
        "Returns a structured echo of the updated state. The echo is "
        "authoritative — if a field you tried to set isn't in it, the "
        'update didn\'t land; fix it and try again rather than saying "done".\n'
        "\n"
        "IMPORTANT: `extraction_prompt` is a FULL replacement — the whole "
        "prompt body, never a diff or a fragment; read the current body via "
        "`memory_metadata(<collection>)` first if you need to keep any of it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Collection name to update"},
            "description": {
                "type": "string",
                "description": (
                    "Content-reflective one-line summary AND the meaning anchor "
                    "(find / resolve-by-meaning) — changing it re-embeds "
                    "the collection. Keep it an accurate summary of the "
                    "subject matter."
                ),
            },
            "notify": {
                "type": "boolean",
                "description": (
                    "Flip notify-on-new: true = start telling the user about "
                    "new entries; false = stop (keep gathering silently). Omit "
                    "to leave unchanged."
                ),
            },
            "extraction_prompt": {
                "type": "string",
                "description": (
                    "FULL rewritten body — replaces the whole prompt, not "
                    "a diff. Drives what the collector actually does. Read "
                    "current body via memory_metadata(<collection>) first for "
                    "scope or silent-flip changes. Mutually exclusive with skill/params."
                ),
            },
            "skill": {
                "type": "string",
                "description": (
                    "Re-render the collection's routine from a skill's current steps "
                    "(exact name or paraphrase). Same name = refresh after re-teaching; "
                    "a different skill = swap; on a hand-authored collection = adopt."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Rebind the skill's holes to new values and re-render, same skill. "
                    "Omit to keep current bindings. Every required hole must be bound."
                ),
            },
            "trigger": {
                "type": "string",
                "description": (
                    "The job's schedule, ONE string in one of three forms — setting it "
                    "REPLACES the whole schedule: 'every <seconds>' (recurring), 'once at "
                    "<ISO datetime> [xN]' (run at a time, N times), or 'on advance of <log>' "
                    "(wake when a source log advances). Omit to leave the schedule unchanged."
                ),
            },
            "expires_at": {
                "type": "string",
                "description": (
                    "OPTIONAL ISO-8601 datetime end condition — the collection archives "
                    "itself when it passes."
                ),
            },
        },
        "required": ["name"],
    }
    args_model = CollectionUpdateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "name", "a collection")
        if not result.success:
            return f"You tried to update {name}'s settings but it didn't work:"
        return f"You updated {name}'s settings:"

    def __init__(self, db: Database, llm_client: LlmClient, run_id: str | None = None) -> None:
        self._db = db
        self._llm_client = llm_client
        # The chat run making this config change — recorded as the update
        # mutation's cause (#1560).
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        """Parse the apply-time trigger union (#1629), then re-render from a skill when
        ``skill``/``params`` is given (the #1620 refresh / rebind / swap / adopt cases),
        else a plain metadata edit.  Reads as a table of contents."""
        args = CollectionUpdateArgs(**kwargs)
        parsed = self._parse_trigger(args)
        if isinstance(parsed, ToolResult):
            return parsed
        trigger, expires_at = parsed
        if trigger is not None and (
            source_error := validate_source_log(self._db, trigger.source_log)
        ):
            return source_error
        if args.skill is not None or args.params is not None:
            return await self._reinstantiate(args, trigger, expires_at)
        return await self._edit_metadata(args, trigger, expires_at)

    def _parse_trigger(
        self, args: CollectionUpdateArgs
    ) -> ToolResult | tuple[Trigger | None, datetime | None]:
        """Parse the optional ``trigger`` arg + end condition at apply time (#1631, the
        same one-arg three-form trigger collection_create uses).  Returns
        ``(None, expires)`` when no trigger is set (cadence untouched) or the parsed
        ``(Trigger, expires)``; an unparseable trigger surfaces ``parse_trigger``'s
        teaching rejection verbatim."""
        try:
            expires_at = parse_datetime(args.expires_at, "expires_at") if args.expires_at else None
            trigger = (
                parse_trigger(args.trigger, _once_form_interval())
                if args.trigger is not None
                else None
            )
        except TriggerError as exc:
            return ToolResult(message=str(exc), success=False)
        return trigger, expires_at

    async def _edit_metadata(
        self, args: CollectionUpdateArgs, trigger: Trigger | None, expires_at: datetime | None
    ) -> ToolResult:
        """The plain metadata edit — description / notify / trigger, and the prompt ONLY
        if a raw ``extraction_prompt`` is supplied (a skill re-render is the other path).
        Omitting skill/params AND extraction_prompt leaves the collection's routine
        untouched; a trigger applies whether or not the prompt changes (#1604 follow-up)."""
        if rejection := _reject_unknown_extraction_tools(
            self._db, self._llm_client, args.extraction_prompt
        ):
            return rejection
        embedding = await self._description_embedding(args)
        memory = self._apply_update(
            args, args.extraction_prompt, None, None, embedding, trigger, expires_at
        )
        suffix = _description_degraded_suffix(args.description, embedding)
        message = f"{_format_collection_echo(memory, 'Updated')}{suffix}"
        return ToolResult(message=message, mutated=True)

    async def _reinstantiate(
        self, args: CollectionUpdateArgs, trigger: Trigger | None, expires_at: datetime | None
    ) -> ToolResult:
        """Re-render the collection's prompt from a skill's CURRENT steps and re-stamp
        its provenance (#1620): refresh (same skill re-taught), rebind (new params),
        swap (a different skill), or adopt (a legacy skill=NULL collection — the second
        half of the #1629 teach bootstrap).  Writes are retargeted to this collection at
        the render seam (#1629); the re-render preserves steps-1..A / no-stored-``done()``
        by construction.  A raw ``extraction_prompt`` alongside conflicts — the render
        owns the prompt."""
        if args.extraction_prompt is not None:
            return ToolResult(message=_REINSTANTIATE_CONFLICT, success=False)
        resolved = await self._resolve_target(args)
        if isinstance(resolved, ToolResult):
            return resolved
        skill, params = resolved
        prompt, render_error = render_skill_prompt(
            self._db, self._llm_client, skill, params, slug(args.name)
        )
        if render_error is not None:
            return render_error
        embedding = await self._description_embedding(args)
        memory = self._apply_update(
            args, prompt, skill.name, params, embedding, trigger, expires_at
        )
        suffix = _description_degraded_suffix(args.description, embedding)
        echo = render_reinstantiation_echo(memory, skill.name, params)
        return ToolResult(
            message=f"{echo}{suffix}{self._no_trigger_note(memory)}",
            mutated=True,
        )

    async def _resolve_target(
        self, args: CollectionUpdateArgs
    ) -> ToolResult | tuple[Skill, dict[str, str]]:
        """The target skill + params for the re-render.  ``params`` default to the
        collection's CURRENT bindings (a refresh keeps them) unless new ones are passed
        (a rebind).  A ``skill`` arg resolves by name-or-meaning (swap / adopt /
        refresh); without one the collection's current skill is reused (rebind),
        refused actionably if it has none or if the pinned skill is gone."""
        current = _resolve(self._db, args.name).row
        params = args.params if args.params is not None else _current_skill_params(current)
        if args.skill is not None:
            resolution = await resolve_skill(self._db, self._llm_client, args.skill)
            if resolution.skill is None:
                return unresolved_skill_result(args.skill, resolution)
            return resolution.skill, params
        if current.skill_name is None:
            return ToolResult(message=_REBIND_NO_SKILL.format(name=current.name), success=False)
        skill = self._db.skills.get(current.skill_name)
        if skill is None:
            return ToolResult(message=_SKILL_GONE.format(skill=current.skill_name), success=False)
        return skill, params

    async def _description_embedding(self, args: CollectionUpdateArgs) -> list[float] | None:
        """Re-embed the routing anchor only when the description changes; else ``None``
        (the anchor stays put)."""
        if args.description is None:
            return None
        return await embed_text(self._llm_client, args.description)

    def _apply_update(
        self,
        args: CollectionUpdateArgs,
        extraction_prompt: str | None,
        skill_name: str | None,
        skill_params: dict[str, str] | None,
        description_embedding: list[float] | None,
        trigger: Trigger | None,
        expires_at: datetime | None,
    ) -> MemoryRow:
        """Thread the update through the store — the metadata fields, the computed
        prompt / skill provenance / anchor, and the apply-time trigger (#1629).  A
        ``trigger`` REPLACES the whole trigger (``replace_trigger``); ``expires_at`` sets
        the end condition.  Records the mutation event with the run id + changed fields
        (trigger / expires_at)."""
        return self._db.memories.update_collection_metadata(
            args.name,
            description=args.description,
            extraction_prompt=extraction_prompt,
            description_embedding=description_embedding,
            notify=args.notify,
            skill_name=skill_name,
            skill_params=skill_params,
            collector_interval_seconds=trigger.collector_interval_seconds if trigger else None,
            run_at=trigger.run_at if trigger else None,
            max_runs=trigger.max_runs if trigger else None,
            source_log=trigger.source_log if trigger else None,
            expires_at=expires_at,
            replace_trigger=trigger is not None,
            run_id=self._run_id,
        )

    @staticmethod
    def _no_trigger_note(memory: MemoryRow) -> str:
        """A visible-degradation note (#1629) when an adopt gave a collection a routine
        but no trigger — it has an ``extraction_prompt`` yet no cadence / run_at /
        source_log, so it won't dispatch until one is set.  Named, not silent."""
        if memory.extraction_prompt is None:
            return ""
        if (
            memory.collector_interval_seconds is not None
            or memory.run_at is not None
            or memory.source_log is not None
        ):
            return ""
        return _NO_TRIGGER_NOTE.format(name=memory.name)


class MemoryMetadataTool(MemoryTool):
    """Return the metadata fields for a single memory (collection or log).

    Genuinely shape-agnostic — metadata describes the memory itself, not its
    contents — so it's named ``memory_metadata`` and applies to either shape.
    """

    name = "memory_metadata"
    description = (
        "Return metadata for a memory: description, notify flag, trigger, last "
        "collected timestamp, archived state, and extraction prompt.  Works for both "
        "collections and logs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "description": "Collection or log name"},
        },
        "required": ["memory"],
    }
    args_model = MemoryNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a memory")
        if not result.success:
            return f"You tried to check the details of {memory} but it didn't work:"
        return f"You checked the details of {memory}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        return ToolResult(message=self._format(_resolve(self._db, args.memory).row))

    def _format(self, memory: Any) -> str:
        last_collected = (
            format_log_timestamp(memory.last_collected_at)
            if memory.last_collected_at is not None
            else "never"
        )
        updated = format_log_timestamp(memory.updated_at)
        # The provenance + lifecycle block (status / expires / created-from-message):
        # the same lines collection_catalog renders, so this exact-name lookup answers
        # the full lifecycle question — who asked for it, what run created it, whether
        # it's live, and when it ends (#1566).
        lifecycle = _lifecycle_block(self._db, memory)
        # The skill-provenance line (#1603) — which skill rendered this collection's
        # recipe, and the params bound into it — sits with the description, ahead of the
        # extraction prompt it produced.  Absent for a hand-authored / seeded
        # collection (skill_name NULL), so that render is unchanged.
        skill = _skill_provenance_line(memory)
        # Lead with what the collection is FOR (description) and what it DOES (the
        # recipe), because that is the substance; the operational settings
        # (cadence/timestamps) are secondary and go last.  Ordered this way — and with
        # the nudge below — so a model asked "what does this do?" walks through the
        # recipe's steps instead of reciting the cadence/notify trivia (the failure the
        # #1530 legibility baseline surfaced).
        lines = [
            f"name: {memory.name}",
            f"type: {memory.type}",
            f"description: {memory.description}",
            *([skill] if skill is not None else []),
            "",
            "What it does each cycle — the recipe below is the collection's actual "
            "behaviour.  When explaining the collection, walk through THESE steps, not the "
            "operational settings.",
            f"extraction prompt: {memory.extraction_prompt or 'none'}",
            "",
            "Operational settings (cadence — secondary):",
            f"notify: {memory.notify}",
            self._trigger_line(memory),
            *lifecycle,
            f"updated: {updated}",
            f"last collected: {last_collected}",
            # The config-change history from the mutation ledger (#1560) — so a
            # mechanism can enumerate everything done to it (create/update/archive)
            # in time order, each change citing the run that made it.
            *_recent_changes_block(self._db, memory.name),
        ]
        return "\n".join(lines)

    @staticmethod
    def _trigger_line(memory: Any) -> str:
        """The copyable ``trigger`` clause (#1631, display form == invocation form):
        ``trigger: every <seconds>`` | ``once at <ISO> [xN]`` | ``on advance of <log>``
        for a collection with a trigger, or ``trigger: none`` for a log / an inert
        collection with no cadence yet — so the render never emits a half-formed clause."""
        has_trigger = (
            memory.collector_interval_seconds is not None
            or memory.run_at is not None
            or memory.source_log is not None
        )
        return f"trigger: {render_trigger_clause(memory) if has_trigger else 'none'}"


class CollectionCatalogTool(MemoryTool):
    """List every user collection — live AND archived — with its full recipe and
    lifecycle.

    The inventory surface: each user collection (logs and framework collectors
    excluded) with its lifecycle block (status / expires / created-from-message),
    description, ``notify`` flag, and full ``extraction_prompt`` — the
    prompts that actually run.  **Archived-inclusive** (#1566): archiving a
    mechanism changes its status, never its visibility, so a just-archived
    collection still renders (clearly marked ``status: archived <when>``) and a
    count over the catalog is correct with respect to the database.  (The
    ``skills`` reconcile collector that once read this to reground on real
    collections was retired by #1624 — skills are structural now; the catalog
    remains the chat agent's read-only window onto what Penny collects.)
    """

    name = "collection_catalog"
    description = (
        "List every collection — live and archived — with its full gather "
        "recipe and lifecycle: name, status (active / archived with when), end "
        "condition, when and from which message it was created, description, "
        "whether it notifies the user "
        "(notify), and its extraction_prompt.  Use it to see what Penny "
        "collects, how each collection is built, and which have been retired.  "
        "Logs and framework collectors are omitted."
    )
    parameters = {"type": "object", "properties": {}}
    args_model = CatalogArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to review your collections but it didn't work:"
        return "You reviewed your collection catalog:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        CatalogArgs(**kwargs)
        rows = sorted(
            (row for row in self._db.memories.list_all() if self._is_user_collection(row)),
            key=lambda row: row.name,
        )
        if not rows:
            return ToolResult(message="(no collections)")
        return ToolResult(message="\n\n".join(self._format(row) for row in rows))

    @staticmethod
    def _is_user_collection(row: MemoryRow) -> bool:
        """A user collection — not a log, not a framework collector
        (skills/quality/notifier).  Includes INERT collections (no
        ``extraction_prompt`` yet, #1629): storage the user set up belongs in the
        inventory, marked as having no routine.  Archived-inclusive (#1566): a
        retired collection still enumerates, marked by its status."""
        return (
            row.type == MemoryType.COLLECTION and row.name not in PennyConstants.SYSTEM_COLLECTIONS
        )

    def _format(self, row: MemoryRow) -> str:
        # The skill-provenance line (#1603) sits right before the recipe it produced,
        # naming which skill rendered this collection's extraction_prompt.  It's
        # absent for a hand-authored / seeded collection (skill_name NULL), so that
        # render stays byte-identical to the unmarked pre-provenance shape.
        lines = [
            f"## {row.name}",
            *_lifecycle_block(self._db, row),
            f"description: {row.description}",
            f"notify: {row.notify}",
        ]
        if (skill := _skill_provenance_line(row)) is not None:
            lines.append(skill)
        lines.append(self._recipe_line(row))
        return "\n".join(lines)

    @staticmethod
    def _recipe_line(row: MemoryRow) -> str:
        """The recipe render — the full ``extraction_prompt`` for a collection with a
        job, or an honest inert marker (#1629) when the collection is storage only (no
        skill attached yet), never a bare ``None``."""
        if row.extraction_prompt is None:
            return "extraction_prompt: (none — inert storage, no skill attached yet)"
        return f"extraction_prompt:\n{row.extraction_prompt}"


class CollectionMergeTool(MemoryTool):
    """Merge all entries from one collection into another, then archive the source."""

    name = "collection_merge"
    description = (
        "Move every entry from `from_memory` into `to_memory`, then archive "
        "`from_memory`.  Entries whose key already exists in `to_memory` are "
        "dropped (destination wins).  Use this to resolve duplicate collections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "from_memory": {
                "type": "string",
                "description": "Collection to merge from (will be archived)",
            },
            "to_memory": {"type": "string", "description": "Collection to merge into (kept)"},
        },
        "required": ["from_memory", "to_memory"],
    }
    args_model = CollectionMergeArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        from_memory = _named(arguments, "from_memory", "one collection")
        to_memory = _named(arguments, "to_memory", "another")
        if not result.success:
            return f"You tried to merge {from_memory} into {to_memory} but it didn't work:"
        return f"You merged {from_memory} into {to_memory}:"

    def __init__(self, db: Database, author: str, run_id: str | None = None) -> None:
        self._db = db
        self._author = author
        # The chat run merging — the source-archive mutation's cause (#1560).
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionMergeArgs(**kwargs)
        return ToolResult(message=self._merge(args.from_memory, args.to_memory), mutated=True)

    def _merge(self, from_name: str, to_name: str) -> str:
        source = _resolve(self._db, from_name)
        source_keys = source.keys()
        if not source_keys:
            self._db.memories.archive(from_name, run_id=self._run_id)
            return f"'{from_name}' was empty — archived with nothing to move."
        moved, dropped = self._move_entries(source, to_name, source_keys)
        self._db.memories.archive(from_name, run_id=self._run_id)
        return self._summary(from_name, to_name, moved, dropped)

    def _move_entries(self, source: Memory, to_name: str, keys: list[str]) -> tuple[int, list[str]]:
        moved = 0
        dropped: list[str] = []
        for key in keys:
            outcome = source.move(key, to_name, author=self._author)
            if outcome == "ok":
                moved += 1
            else:
                dropped.append(key)
        return moved, dropped

    def _summary(self, from_name: str, to_name: str, moved: int, dropped: list[str]) -> str:
        head = f"Merged '{from_name}' → '{to_name}': {moved} moved"
        if dropped:
            named = ", ".join(f"'{key}'" for key in dropped)
            head += f", {len(dropped)} dropped (already in '{to_name}': {named})"
        return f"{head}. '{from_name}' archived."


class CollectionDeleteEntryTool(MemoryTool):
    """Delete an entry from a collection by key."""

    name = "collection_delete_entry"
    description = (
        "Delete the entry with the given key from a collection. Returns the "
        "number of entries removed (zero if the key did not exist)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "key": {"type": "string"},
        },
        "required": ["memory", "key"],
    }
    args_model = CollectionDeleteEntryArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        entry = _entry_label(arguments)
        if not result.success:
            return f"You tried to remove {entry} from {memory} but it didn't work:"
        if not result.mutated:
            return f"You couldn't find {entry} to remove from {memory}:"
        return f"You removed {entry} from {memory}:"

    def __init__(self, db: Database, scope: str | None = None) -> None:
        self._db = db
        self._scope = scope

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionDeleteEntryArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return ToolResult(
                message=_SCOPE_REFUSAL_MESSAGE.format(scope=self._scope, memory=args.memory),
                success=False,
            )
        memory = _resolve(self._db, args.memory)
        removed = memory.delete(args.key)
        if removed == 0:
            rejection = _bracket_key_rejection(memory, args.memory, args.key)
            if rejection is not None:
                return rejection
            return ToolResult(
                message=f"No entry with key '{args.key}' in '{args.memory}' — nothing to delete. "
                f"List the current keys with collection_keys('{args.memory}') to find it."
            )
        return ToolResult(message=f"Deleted '{args.key}' from '{args.memory}'.", mutated=True)


# ── Log reads ───────────────────────────────────────────────────────────────


_LOG_READ_CURSOR_DESCRIPTION = (
    "Return the next batch of entries appended to this log since you last read "
    "it.  A cursor tracks where you left off — review everything it returns; the "
    "next cycle hands you the next batch.  You never specify a count."
)
_LOG_READ_WINDOW_DESCRIPTION = (
    "Return recent entries from this log (a short look-back window), oldest "
    "first.  Use for 'what just happened' questions."
)


class CursorReadTool(MemoryTool):
    """Base for read tools that track a pending per-source cursor advance.

    During a cycle a cursored read records, per source memory, the high-water
    timestamp it consumed (in ``_pending``).  The orchestration layer commits
    those advances after a successful cycle and discards them after a failed one
    — so a cursor only moves forward over input the agent actually processed, and
    a crash re-reads rather than skips.  Subclasses do the read and populate
    ``_pending`` (the advance shape differs — a batch's max vs a single returned
    entry); the commit/discard lifecycle is shared here so the orchestration can
    treat every cursored tool uniformly (``isinstance(tool, CursorReadTool)``).
    """

    def __init__(self, db: Database, agent_name: str) -> None:
        self._db = db
        self._agent_name = agent_name
        self._pending: dict[str, datetime] = {}

    def commit_pending(self) -> None:
        """Persist each source's pending cursor after a successful cycle."""
        for memory_name, last_read_at in self._pending.items():
            self._db.cursors.advance_committed(self._agent_name, memory_name, last_read_at)
        self._pending.clear()

    def discard_pending(self) -> None:
        """Drop pending cursor advances — a failed cycle keeps cursors put."""
        self._pending.clear()

    def _advance_pending(self, memory: str, timestamps: list[datetime]) -> None:
        """Track the highest timestamp seen this run as the pending cursor."""
        if not timestamps:
            return
        max_seen = max(timestamps)
        prev = self._pending.get(memory)
        if prev is None or max_seen > prev:
            self._pending[memory] = max_seen


class LogReadTool(CursorReadTool):
    """Read entries from a log — one tool, caller-dispatched behaviour.

    For a collector (``scope`` set) it's CURSOR-based: it returns the next
    bounded batch since the agent's last committed cursor, so a review job works
    through everything in batches and can't miss entries.  Cursor advance is
    *pending* until ``commit_pending`` after a successful run (a failed run
    discards it).  For chat/schedule (``scope`` is None) it's WINDOW-based: the
    most recent look-back window, stateless — an ad-hoc "what just happened"
    read.  The caller never chooses the mode or a size; Python does, from who's
    asking — so the model can't pick the wrong one.
    """

    name = "log_read"
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }
    args_model = ReadLogArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        log = _memory_name(arguments, "a log")
        if not result.success:
            return f"You tried to read {log} but it didn't work:"
        return f"You read {log}:"

    def __init__(self, db: Database, agent_name: str, scope: str | None) -> None:
        super().__init__(db, agent_name)
        self._cursor_mode = scope is not None
        self.description = (
            _LOG_READ_CURSOR_DESCRIPTION if self._cursor_mode else _LOG_READ_WINDOW_DESCRIPTION
        )

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadLogArgs(**kwargs)
        memory = _resolve(self._db, args.memory)
        entries = self._read_cursor(memory) if self._cursor_mode else self._read_window(memory)
        return ToolResult(
            message=_format_entries(entries, source=args.memory, ordering="oldest first")
        )

    def _read_cursor(self, memory: Memory) -> list[MemoryEntry]:
        """The next bounded batch since this agent's committed cursor.  The log
        owns the read (first-read = most-recent N; later = next N since the
        cursor); the tool only tracks the pending advance.  Uniform across every
        log backing — ``collector-runs`` renders runs, the message logs read
        ``messagelog``, all through the same ``read_batch``."""
        cursor = self._db.cursors.get(self._agent_name, memory.name)
        entries = memory.read_batch(cursor, PennyConstants.LOG_READ_LIMIT)
        self._advance_pending(memory.name, [entry.created_at for entry in entries])
        return entries

    def _read_window(self, memory: Memory) -> list[MemoryEntry]:
        return memory.read_window(PennyConstants.LOG_READ_WINDOW_SECONDS)


class ReadRunCallsTool(CursorReadTool):
    """Read a source's recent runs as their tool-call SEQUENCES.

    A sibling of ``log_read`` — same cursored machinery (the next bounded batch
    since this reader's committed cursor, oldest-first, pending until the cycle
    succeeds) — but a different lens: each run renders as its ``origin → the tool
    calls → conclusion`` (``render_run_calls``), the sequence-lens view of what a run
    *did*.  Orthogonal to the target: ``"chat"`` renders conversational runs
    (``user: <message>`` → tools → ``penny: <reply>``); a collector's name renders
    that collector's runs (``[target]`` → tools → ``done: <outcome>``).  Lets a
    reader see the tool sequence a request drove — for authoring skills, or for
    inspecting what a collector actually did.  The runs come from ``promptlog``; the
    cursor is per-target.
    """

    name = "read_run_calls"
    args_model = ReadRunCallsArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        target = _named(arguments, "target", "a source")
        if not result.success:
            return f"You tried to review {target}'s recent runs but it didn't work:"
        return f"You reviewed {target}'s recent runs:"

    def __init__(self, db: Database, agent_name: str) -> None:
        super().__init__(db, agent_name)
        # Discover the valid targets from the DB so the model sees the current set —
        # 'chat' plus every collector (a collection with an extraction_prompt).
        targets = self._available_targets()
        listed = ", ".join(targets)
        self.description = (
            "Return a source's recent runs as tool-call sequences — each run shows its "
            "origin (the user's message, or the collector's bound target), the tool "
            "calls made, and the outcome. Returns the next batch since you last read; "
            f"call again for older runs. target is one of: {listed} "
            '("chat" for conversations, a collector name for that collector\'s runs).'
        )
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": targets, "description": f"one of {listed}"}
            },
            "required": ["target"],
        }

    def _available_targets(self) -> list[str]:
        collectors = sorted(
            row.name
            for row in self._db.memories.list_all()
            if row.extraction_prompt and not row.archived
        )
        return [PennyConstants.CHAT_AGENT_NAME, *collectors]

    @staticmethod
    def _cursor_key(target: str) -> str:
        # Namespaced so a per-target run-calls cursor never collides with a real
        # memory's cursor (a collector named ``target`` also has log cursors).
        return f"run-calls:{target}"

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadRunCallsArgs(**kwargs)
        # Resolve the target against the valid set FIRST — 'chat' plus every live
        # collector, exactly what the description enumerates — so a typo'd/unknown
        # target gets the actionable memory-not-found refusal instead of a silent
        # empty batch that reads as "this collector has no runs".
        if args.target not in self._available_targets():
            raise MemoryNotFoundError(args.target)
        key = self._cursor_key(args.target)
        cursor = self._db.cursors.get(self._agent_name, key)
        groups = self._db.messages.run_call_groups(
            args.target, cursor, PennyConstants.RUN_CALLS_LIMIT
        )
        entries = [
            MemoryEntry(
                memory_name=key,
                key=None,
                content=render_run_calls(group),
                author=PennyConstants.MessageAuthor.COLLECTOR,
                created_at=group[-1].timestamp,
            )
            for group in groups
            if group
        ]
        self._advance_pending(key, [entry.created_at for entry in entries])
        return ToolResult(
            message=_format_entries(entries, source=args.target, ordering="oldest first")
        )


# An id with no recognised type tag: name what IS addressable + the guess-free
# fallbacks (find for names, the activity block for events), never a silent
# empty (the anchor discipline — a rendered id resolves in one call, a malformed
# one recovers).
_GET_EVENT_UNTYPED = (
    "'{event_id}' isn't a typed event id.  Pass a run token exactly as your "
    "current-state activity block renders it — get_event(event_id='run <id>'), "
    "copying the whole `run <id>` token.  (Runs are the addressable events right "
    "now; to resolve a collection or log by meaning use find(query=<text>), "
    "and your activity block names every recent event.)"
)

# A well-formed run token that matched no recorded run — say so and point at the
# lists that carry valid ids, rather than an empty read that reads as "clean run".
_GET_EVENT_NO_RUN = (
    "No run found with id '{run_id}'.  It must be an id your current-state activity "
    "block rendered (a `run <id>` line, or a change's `(run <id>)` cause); "
    "read_run_calls(target='chat') — or a collector's name — lists recent run ids "
    "if you need to find a valid one."
)


class GetEventTool(Tool):
    """Resolve ONE ledger event by the typed id the activity block renders (#1580).

    The self-state activity block renders each background run as ``run <id>`` (and
    each config change names its ``(run <id>)`` cause) — typed anchors meant to be
    consumed VERBATIM.  ``get_event`` is the verb that consumes them: it parses the
    type tag and returns that event's detail.  Today the one addressable event kind
    is a run, so ``get_event(event_id='run <id>')`` returns that run's tool-call
    SEQUENCE — the same canonical projection ``read_run_calls`` renders, but for the
    single run the id names (a point lookup, where ``read_run_calls`` browses a
    source's run history).  An id with no recognised tag, or a run nothing recorded,
    gets an actionable refusal naming what IS addressable and how to find a valid id
    (a rendered id resolves in one call; a bad one recovers, never silently empties).

    Deliberately run-only (the smaller conforming shape, #1580): the send case is a
    *render* — an autonomous send's provenance appears inline wherever the message
    renders (``penny-messages`` reads, the header's sent lines), so it's a read, not
    an event lookup (#1568/#1608); and a mutation carries its detail inline in
    ``memory_metadata``'s change history and names its causing ``run <id>`` here, so
    no ``mut``-tagged arm is built until a surface actually renders that anchor (a
    verb needs a rendered id to consume — machinery follows a customer, not before).
    """

    name = "get_event"
    description = (
        "Look up ONE event from your activity log by the typed id it rendered — a "
        "`run <id>` line, or a change's `(run <id>)` cause.  Pass the whole token, "
        "get_event(event_id='run <id>'), copying the id exactly as rendered, and "
        "get that run's tool-call sequence: what it did, step by step.  Use it to "
        "inspect a specific run you saw in your current state; to browse a source's "
        "run history instead use read_run_calls(target=<'chat' or a collector name>)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": (
                    "The typed event id, verbatim as your activity block rendered "
                    "it — the whole `run <id>` token."
                ),
            }
        },
        "required": ["event_id"],
    }
    args_model = GetEventArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        event = _named(arguments, "event_id", "an event")
        if not result.success:
            return f"You tried to look up {event} but it didn't work:"
        return f"You looked up {event}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        # Plain ``Tool``, not ``MemoryTool``: this reads the ledger by id, resolves
        # no memory, and raises no MemoryAccessError — both misses are enumerated
        # refusals below, so the base's except template would be inert here.
        args = GetEventArgs(**kwargs)
        run_id = self._parse_run_id(args.event_id)
        if run_id is None:
            return ToolResult(
                message=_GET_EVENT_UNTYPED.format(event_id=args.event_id), success=False
            )
        prompts = self._db.messages.get_run_prompts(run_id)
        if not prompts:
            return ToolResult(message=_GET_EVENT_NO_RUN.format(run_id=run_id), success=False)
        return ToolResult(message=render_run_calls(prompts))

    @staticmethod
    def _parse_run_id(event_id: str) -> str | None:
        """The run id from a rendered ``run <id>`` token, or ``None`` when the token
        carries no recognised type tag.

        Tolerates the paren framing the mutation line renders (``(run <id>)``) so
        both rendered forms resolve verbatim; ``RUN_EVENT_PREFIX`` is the one shared
        constant the render emits and this parse strips, so they can't drift."""
        token = event_id.strip().strip("()").strip()
        prefix = PennyConstants.RUN_EVENT_PREFIX
        if not token.lower().startswith(prefix):
            return None
        run_id = token[len(prefix) :].strip()
        return run_id or None


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendTool(MemoryTool):
    """Append a keyless entry to a log."""

    name = "log_append"
    description = "Append one keyless entry to a log. No dedup runs; every append is stored."
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["memory", "content"],
    }
    args_model = LogAppendArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a log")
        if not result.success:
            return f"You tried to add an entry to {memory} but it didn't work:"
        return f"You added an entry to {memory}:"

    def __init__(
        self, db: Database, llm_client: LlmClient, author: str, run_id: str | None = None
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author
        # The appending run — stamped on the new log entry (#1560).
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = LogAppendArgs(**kwargs)
        memory = _resolve(self._db, args.memory)
        vec = await embed_text(self._llm, args.content)
        if vec is None:
            logger.warning("log_append: embedding unavailable for an entry in %s", memory.name)
            return ToolResult(
                message=_EMBED_WRITE_FAILURE_LOG.format(memory=args.memory), success=False
            )
        memory.append(
            [LogEntryInput(content=args.content, content_embedding=vec)],
            author=self._author,
            run_id=self._run_id,
        )
        return ToolResult(message=f"Appended to '{args.memory}'.", mutated=True)


# ── Resolve by meaning (find) ────────────────────────────────────────────────
#
# Identity and affordances are answered together: every hit renders its exact
# identity AND how to address it — the specific tool + call shape that operates
# on it.  The type→addressing map is DETERMINISTIC (the thing's family fixes its
# finite action set); the model never derives it, it copies the call verbatim.
# This dissolves the entry-vs-collection footgun (a skill entry addressed with
# collection tools) by answering "what can be done with it" in the same breath as
# "what is it" (#1558).  Stored entries are a fourth family (#1640): the hit
# CARRIES the value, so for a short fact the find IS the answer.

_FIND_EMBED_FAILURE = (
    "Couldn't embed your query (a transient embedding error) — the meaning search "
    "needs it. Retry find(query=<what the thing is about>) in a moment, or list "
    "everything with collection_catalog()."
)

# Zero matches is an honest empty, not an error — name the wider nets (the catalog
# spans archived collections too; the self-state header names active mechanisms,
# logs, and recent activity) so the model never dead-ends on a miss.  If the query
# was a how-to/task lookup, the final clause signposts the task→skill→teach dead end
# (#1650): no skill matched → ask the user to teach it once.
_FIND_EMPTY = (
    'Nothing of yours matched "{query}". Widen the net: '
    "collection_catalog() lists every collection (archived included), and your "
    "current-state header names your active mechanisms, logs, and recent activity. "
    "If you were looking for how to do a task and no skill matched, ask the user to "
    "walk you through it once here in chat — you'll learn it as a new skill "
    "automatically."
)

# Ambiguity is RETURNED, never silently resolved: several hits come back ranked
# with how to narrow.
_FIND_AMBIGUOUS_TAIL = (
    "Ranked by closeness — if one is what you meant, use its addressing above; "
    "otherwise narrow by its exact name."
)

# A stored entry's value renders as one compact, whitespace-collapsed line — a
# short fact shows whole (the find IS the answer, #1640), a long one is capped
# like the run-trace result preview so a big value can't blow up the result.
_ENTRY_PREVIEW_CHARS = 120


def _find_state(match: ResolvedMatch) -> str:
    """The live/archived state word: a taught skill is always ``live``; a
    collection or log is ``active`` or ``archived``."""
    if match.kind == ResolvedKind.SKILL:
        return "live"
    return "archived" if match.archived else "active"


def _find_type_label(match: ResolvedMatch) -> str:
    """The family noun a hit renders — a taught skill names the skill registry
    (the sole skills store, #1624) so the skill-vs-collection distinction is
    explicit."""
    if match.kind == ResolvedKind.SKILL:
        return "taught skill"
    return match.kind.value  # "collection" / "log"


def _find_addressing(match: ResolvedMatch) -> str:
    """The deterministic type→addressing map: the object's family (plus archived
    state) fixes the exact tool call that operates on it, so following the stated
    addressing succeeds with no further guessing (the anchor discipline).  The
    model never derives this mapping — it copies the call verbatim."""
    name = match.name
    if match.kind == ResolvedKind.SKILL:
        return (
            f"read it with skill_read('{name}'); to change it, demonstrate it again "
            f"in chat — I relearn it automatically, replacing this one"
        )
    if match.kind == ResolvedKind.LOG:
        return f"read it with log_read('{name}')"
    if match.archived:
        return (
            f"restore it with collection_unarchive('{name}'); its entries stay readable "
            f"with collection_read_latest('{name}')"
        )
    return (
        f"read it with collection_read_latest('{name}'), reconfigure it with "
        f"collection_update(name='{name}', ...), archive it with "
        f"collection_archive('{name}')"
    )


def _entry_preview(content: str) -> str:
    """One compact, whitespace-collapsed line of an entry's stored value (#1640) —
    shown whole when short (the find carries the answer), capped when long."""
    collapsed = " ".join(content.split())
    if len(collapsed) <= _ENTRY_PREVIEW_CHARS:
        return collapsed
    return f"{collapsed[:_ENTRY_PREVIEW_CHARS].rstrip()}…"


def _entry_identity(entry: ResolvedEntry) -> str:
    """The entry's head identity: a keyed collection entry shows its invocation-form
    key in its collection; a keyless log entry shows its ``#<id>`` handle in its
    log (the two forms the entry can be read back by)."""
    if entry.key is None:
        return f"entry #{entry.entry_id} in `{entry.memory_name}` ({entry.container_kind.value})"
    return f"entry {render_key(entry.key)} in `{entry.memory_name}`"


def _entry_addressing(entry: ResolvedEntry) -> str:
    """How to read the stored entry back — deterministic from whether it is keyed: a
    keyed collection entry is one ``collection_get`` away (re-reads the exact value
    the find already carried); a keyless log entry names its ``<log>#<id>`` handle
    and the ``log_read`` that surfaces it (the entry_by_id-style retrieval, #1640)."""
    if entry.key is None:
        handle = f"{entry.memory_name}{PennyConstants.MEMORY_HANDLE_SEPARATOR}{entry.entry_id}"
        return f"log_read('{entry.memory_name}') — the full entry is {handle}"
    return f"collection_get(memory='{entry.memory_name}', key='{entry.key}')"


def _render_entry_hit(index: int, entry: ResolvedEntry) -> str:
    """One stored-entry hit (#1640): its identity + the value it carries, then how to
    read it back — mirroring the object hit's two-line shape."""
    head = f'{index}. {_entry_identity(entry)} — "{_entry_preview(entry.content)}"'
    return f"{head}\n   read it: {_entry_addressing(entry)}"


def _render_object_hit(index: int, match: ResolvedMatch) -> str:
    """One object hit: ``N. <name> — <state> <type>[: <scope>]`` then the
    verbatim-usable addressing on the next line."""
    head = f"{index}. {match.name} — {_find_state(match)} {_find_type_label(match)}"
    if match.label:
        head += f": {match.label}"
    return f"{head}\n   how to use it: {_find_addressing(match)}"


def _render_find_hit(index: int, hit: ResolvedHit) -> str:
    """One fused hit, best-first ranked — dispatched by family: a stored entry
    renders its value + how to read it, an object its state/type + how to use it."""
    if isinstance(hit, ResolvedEntry):
        return _render_entry_hit(index, hit)
    return _render_object_hit(index, hit)


class FindTool(MemoryTool):
    """Find anything of Penny's own by meaning — the guess-free fallback when the
    exact name/key isn't ambient (#1558, #1640).

    Embedding search (plain cosine, the #1565 explicit-search path) over every
    registry row's description anchor (collections + logs, ARCHIVED INCLUDED),
    every taught skill's description anchor (the ``skill`` table, the sole skills
    store — #1624), AND every stored entry's content/key anchors across the
    non-archived collections + real logs (#1640), ranked best-first in ONE fused
    list.  An object hit returns its exact identity, family, live/archived state,
    AND how to address it — the specific tool + call shape that operates on it,
    fixed deterministically by its type (never derived by the model).  An entry hit
    CARRIES the stored value plus how to read it back, so for a short fact the find
    IS the answer (one call, no container guess).  Ambiguity is returned, not
    resolved: several candidates come back ranked with how to narrow; zero matches
    is an honest empty naming the catalog + self-state header as the wider net.
    Dissolves the name-guessing loop and the entry-vs-collection footgun by
    answering "what is it" and "what can be done with it" in one result.
    """

    name = "find"
    description = (
        "Find anything of your own by meaning, when you don't know its exact name — "
        "a collection, a log, a taught skill, or a single stored entry (a fact you "
        "saved).  Pass `query`, a paraphrase of what it's about (\"what I said that "
        'product costs").  Returns the closest matches, best first: a '
        "collection/log/skill with its exact name, whether it's active or archived, "
        "and the exact tool call that operates on it (archived included); a stored "
        "entry with the value it holds and how to read it back.  Use it instead of "
        "guessing a name: a guessed name fails, this resolves the real one — and for "
        "a short fact the returned value IS the answer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A paraphrase of what the thing is about — its meaning, not its exact name."
                ),
            },
        },
        "required": ["query"],
    }
    args_model = FindArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        query = arguments.get("query")
        label = f' for "{query}"' if query else ""
        if not result.success:
            return f"You tried to find something of your own{label} but it didn't work:"
        return f"You looked for something of your own{label}:"

    def __init__(self, db: Database, llm_client: LlmClient) -> None:
        self._db = db
        self._llm = llm_client

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = FindArgs(**kwargs)
        vec = await embed_text(self._llm, args.query)
        if vec is None:
            logger.warning("find: query embedding failed transiently")
            return ToolResult(message=_FIND_EMBED_FAILURE, success=False)
        hits = self._db.memories.resolve_objects(vec, PennyConstants.FIND_MATCH_LIMIT)
        return ToolResult(message=self._format(args.query, hits))

    def _format(self, query: str, hits: list[ResolvedHit]) -> str:
        if not hits:
            return _FIND_EMPTY.format(query=query)
        body = "\n".join(_render_find_hit(i, h) for i, h in enumerate(hits, start=1))
        if len(hits) == 1:
            return f'Found 1 thing matching "{query}":\n{body}'
        return (
            f'Found {len(hits)} things matching "{query}", best first:\n{body}\n'
            f"{_FIND_AMBIGUOUS_TAIL}"
        )


# ── Introspection / lifecycle ───────────────────────────────────────────────


_EXISTS_EMBED_DEGRADED = (
    "inconclusive — the embedding service was unavailable this cycle, so only "
    "exact-key matching ran and the similarity dedup was skipped; no exact-key "
    "match was found, but a near-duplicate can't be ruled out. Re-probe next "
    "cycle with exists(memories={memories}, content=<content>, key=<key>), or "
    "write only if you're sure the entry is new."
)


class ExistsTool(MemoryTool):
    """Probe whether an equivalent entry already exists across a set of memories."""

    name = "exists"
    description = (
        "Check whether an entry equivalent to the given key/content already "
        "exists in any of the listed memories. Uses the same similarity-based "
        "dedup rule as `collection_write(memory=<collection>, entries=<key + "
        "content>)`. Use this before writing to avoid duplicates that span "
        "multiple collections."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of memories to search",
            },
            "content": {"type": "string"},
            "key": {"type": "string", "description": "Optional — enables exact-key shortcut"},
        },
        "required": ["memories", "content"],
    }
    args_model = ExistsArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to check whether that entry already exists but it didn't work:"
        return "You checked whether that entry already exists:"

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient,
        thresholds: DedupThresholds | None = None,
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._thresholds = thresholds

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ExistsArgs(**kwargs)
        # When the model probes with content but no key, treat the content
        # as a name-like probe — using it as both ``key`` and ``content``
        # lets the dedup's key-TCR signal fire against existing entries
        # whose ``key`` matches.  Without this, ``exists(content="Catan")``
        # returned "no" against an existing ``key="Catan"`` because
        # content-cosine alone (candidate "Catan" vs the existing entry's
        # long description) sat below the strict threshold.
        key = args.key if args.key else args.content
        key_vec = await embed_text(self._llm, key)
        content_vec = await embed_text(self._llm, args.content)
        # ``exists`` validates every name and raises ``MemoryNotFoundError`` on
        # an unknown one — the base ``execute`` renders that as the actionable
        # not-found refusal, so a typo'd probe never misreports "no" and
        # green-lights the write it was checking for.
        found = self._db.memories.exists(
            args.memories,
            key,
            key_vec,
            content_vec,
            thresholds=self._thresholds,
        )
        if found:
            return ToolResult(message="yes")
        # A "no" that rode a failed embed only ran the exact-key signal — the
        # similarity dedup was skipped, so the answer is inconclusive.  Surface
        # that instead of a confident "no" (visible degradation over a silent
        # miss that could green-light a near-duplicate write).
        if key_vec is None or content_vec is None:
            return ToolResult(message=_EXISTS_EMBED_DEGRADED.format(memories=args.memories))
        return ToolResult(message="no")


class DoneTool(Tool):
    """Signal the cycle is finished — an argless sentinel (#1569).

    ``done()`` takes no arguments: it just marks the cycle finished.  The run
    record is GENERATED from the run's canonical ledger rows (its tool calls +
    write-gate outcomes + structural counts), so there is no model-authored
    ``success``/``summary`` to confabulate — what is generated cannot lie."""

    name = PennyConstants.DONE_TOOL_NAME
    description = (
        "Call this — with NO arguments — when the cycle is finished.  It just marks "
        "the cycle done; the run record is generated automatically from the tool "
        "calls you actually made, so there is nothing to summarise."
    )
    parameters = {"type": "object", "properties": {}}
    args_model = NoArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        return "You wrapped up the cycle:"

    async def execute(self, **kwargs: Any) -> ToolResult:
        # The done call itself always succeeds and never mutates state; the run's
        # outcome is derived structurally from the ledger, not from this call.
        return ToolResult(message="Cycle complete.")


# ── On-demand collector trigger ─────────────────────────────────────────────


class TestExtractionPromptTool(Tool):
    """Immediately run the collector for a named collection, bypassing the schedule."""

    name = "test_extraction_prompt"
    timeout = 300.0  # collector cycles include browse calls that can take several minutes
    description = (
        "Immediately trigger one collector cycle for the named collection, bypassing "
        "the normal idle-gated schedule.  Use this while authoring or refining an "
        "extraction_prompt to verify the collector reads the right sources and writes "
        "the expected entries.  Returns the cycle's structural outcome and tool trace."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "description": "Collection name to test"},
        },
        "required": ["memory"],
    }
    args_model = MemoryNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _memory_name(arguments, "a collection")
        if not result.success:
            return f"You ran the {memory} collector to test it, but the cycle didn't succeed:"
        return f"You ran the {memory} collector to test it:"

    def __init__(self, collector: Collector) -> None:
        self._collector = collector

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        success, summary = await self._collector.run_for(args.memory)
        marker = "✅" if success else "❌"
        # ``success`` must flow into the structured field, not live only in the ❌
        # marker text — otherwise every failure path (not-found, archived, no/short
        # prompt, cycle failure) reads as a passing call to ``record.failed`` /
        # ``tool_failures`` / run-health accounting.
        return ToolResult(message=f"{marker} {summary}", success=success)


# ── Factory ─────────────────────────────────────────────────────────────────

# The agent name passed to ``build_memory_tools`` purely to enumerate the collector's
# tool NAMES — it only sets a tool's cursor identity, which is irrelevant to the names.
_VOCAB_PROBE_AGENT = "collector"


def _collector_tool_surface(db: Database, llm_client: LlmClient) -> frozenset[str]:
    """The names of every tool a collector runs with — the surface an
    ``extraction_prompt`` may legitimately call.

    Discovered from the *real* assembly (``build_memory_tools`` + browse + done +
    send_message, i.e. ``BackgroundAgent.get_tools``) rather than a hardcoded list, so
    it can never drift from what a collector actually runs — add a collector tool and
    it's covered for free.  ``include_lifecycle=False`` mirrors the collector's masked
    surface (#1556): the registry-shape tools are absent from a cadence run, so an
    ``extraction_prompt`` that names ``collection_create`` / ``collection_update`` /
    archive / merge is rejected at authoring time rather than persisted into a prompt
    the collector could never run.  ``BrowseTool`` / ``SendMessageTool`` are imported
    lazily: ``send_message`` imports ``DoneTool`` from this module, so a top-level
    import here would close that cycle.
    """
    from penny.tools.browse import BrowseTool
    from penny.tools.send_message import SendMessageTool

    memory_names = {
        tool.name
        for tool in build_memory_tools(db, llm_client, _VOCAB_PROBE_AGENT, include_lifecycle=False)
    }
    return frozenset(memory_names | {BrowseTool.name, DoneTool.name, SendMessageTool.name})


def _reject_unknown_extraction_tools(
    db: Database, llm_client: LlmClient, extraction_prompt: str | None
) -> ToolResult | None:
    """A failed ``ToolResult`` if ``extraction_prompt`` names a tool no collector can
    run (a hallucinated ``extract_text``), else ``None``.  A ``None`` prompt — an
    update leaving it unchanged — is nothing to check.  Runs at ``collection_create`` /
    ``collection_update`` time so a fictitious call is never persisted into a prompt the
    collector would then fail to run every cycle."""
    if extraction_prompt is None:
        return None
    surface = _collector_tool_surface(db, llm_client)
    if error := check_extraction_prompt_tools(extraction_prompt, surface):
        return ToolResult(message=error, success=False)
    return None


def build_memory_tools(
    db: Database,
    llm_client: LlmClient,
    agent_name: str,
    scope: str | None = None,
    run_id: str | None = None,
    include_lifecycle: bool = True,
) -> list[Tool]:
    """Construct the memory tool surface for an agent.

    **Reads + entry mutations for every agent; lifecycle (registry-shape) tools
    only when ``include_lifecycle`` is set.**  Capability within a tier is not
    curated by omission; instead every tool funnels its resolve + op through one
    ``try``:
    ``_resolve(db, name)`` (missing → ``MemoryNotFoundError``) and the method on
    the returned ``Memory`` object (wrong shape → ``WrongShapeError``; read-only
    facade → ``ReadOnlyMemoryError``).  All three share the ``MemoryAccessError``
    base, so the tool catches that one and returns ``str(exc)`` verbatim — no
    sentinel return, no per-call type check, no branching on a name or shape:

    * **Wrong shape.** A keyed read/write on a log (or a cursored ``log_read``
      on a collection) hits a base no-op that raises ``WrongShapeError``.  This
      is what stops a newest-first ``collection_read_latest`` from bypassing a
      log's cursor.
    * **Read-only facades.** ``log_append`` to ``user-messages`` /
      ``penny-messages`` / ``collector-runs`` is refused up front
      (``SYSTEM_LOGS``); the facades also raise ``ReadOnlyMemoryError`` if
      reached, since they're views over ``messagelog`` / ``promptlog``.
    * **Collector binding.** ``scope`` pins a collector's entry
      mutations to its bound collection ``X`` (the ``scope`` check in
      each entry-mutation tool).  Chat passes ``scope=None`` — its
      entry mutations are unrestricted, since edits are user-directed.

    ``read_similar`` (embedding search) and ``memory_metadata`` are the
    genuinely shape-agnostic reads — they work on either shape.  ``find``
    (resolve-by-meaning, #1558/#1640) spans the whole registry — collections, logs,
    taught skills, and stored entries — and returns each hit's exact identity fused
    with how to
    address it, the guess-free fallback that every not-found error points at.

    ``DoneTool`` / ``send_message`` are intentionally not here — they're
    loop-control, not capability, added in ``BackgroundAgent.get_tools``.
    Chat replies via final text and must not have ``done`` available, or
    the model may call it instead of producing a reply.

    ``run_id`` is the id of the run that built the surface — the chat turn's run,
    or the collector cycle's — passed as an explicit parameter, never ambient state
    (#1560).  It threads to every write- and mutation-capable tool as the executing
    run: entry writes stamp it as ``created_by_run_id`` / ``last_written_by_run_id``
    on the rows they add or rewrite; ``collection_create`` records it as the new
    mechanism's ``created_by_run_id`` (#1566); and create / update / archive /
    unarchive record it on the durable mutation event they emit (the ledger's
    provenance closure).  ``None`` for a non-run caller, which leaves the columns
    NULL.  (The spawning ``source_message_id`` is linked afterward by the channel,
    since the message id isn't known until the run returns.)

    ``include_lifecycle`` gates the registry-shape tier — ``collection_create`` /
    ``collection_update`` / ``collection_merge`` / ``collection_archive`` /
    ``collection_unarchive`` / ``log_create``.  Chat-style agents get it (the user
    evolves collections through them); a cadence-fired collector run passes
    ``False`` (#1556), so those tools are structurally ABSENT from its surface — a
    background poll cannot create, reconfigure, merge, or archive mechanisms, no
    matter what its extraction_prompt says.  The declaration lives on the agent
    (``Agent._include_lifecycle_tools``, overridden by ``Collector``), not as a
    branch here.
    """
    reads: list[Tool] = [
        CollectionReadLatestTool(db),
        ReadSimilarTool(db, llm_client),
        CollectionGetTool(db),
        CollectionReadRandomTool(db),
        CollectionKeysTool(db),
        MemoryMetadataTool(db),
        CollectionCatalogTool(db),
        LogReadTool(db, agent_name, scope),
        ReadRunCallsTool(db, agent_name),
        GetEventTool(db),
        ExistsTool(db, llm_client),
        FindTool(db, llm_client),
    ]
    lifecycle: list[Tool] = [
        CollectionCreateTool(db, llm_client, created_by_run_id=run_id),
        CollectionUpdateTool(db, llm_client, run_id=run_id),
        CollectionMergeTool(db, agent_name, run_id=run_id),
        CollectionArchiveTool(db, run_id=run_id),
        CollectionUnarchiveTool(db, run_id=run_id),
        LogCreateTool(db, llm_client),
        # Skill INSPECTION rides the chat (lifecycle) surface.  There is no
        # skill_create tool — skills are distilled automatically at chat-run end
        # (#1658); the model only reads them (skill_read) and instantiates them
        # (collection_create / collection_update).  A cadence collector follows the
        # rendered text prompt and never touches the skill registry.
        SkillReadTool(db),
    ]
    mutations: list[Tool] = [
        CollectionWriteTool(db, llm_client, agent_name, scope=scope, run_id=run_id),
        UpdateEntryTool(db, agent_name, scope=scope, run_id=run_id),
        CollectionDeleteEntryTool(db, scope=scope),
        LogAppendTool(db, llm_client, agent_name, run_id=run_id),
    ]
    return reads + (lifecycle if include_lifecycle else []) + mutations
