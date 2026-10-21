"""Tool-layer wrappers over the memory access layer.

Every tool validates its kwargs through a Pydantic args model as its first
line (per CLAUDE.md), calls ``db.memories.*``, and returns a serializable
string the model can reason over.

Author attribution is passed explicitly: write-capable tools take an
``author: str`` at construction time (the agent that owns the tool).
``build_memory_tools(db, embedding_client, author)`` is the factory each
agent calls with its own ``self.name`` so writes are attributed correctly.

Tools that need embeddings (writes, similarity reads) take an
``LlmClient`` in ``__init__``. On a transient embed failure at write time the
write is REFUSED, not stored without a vector (#1412): every stored entry must
carry its similarity vector, so ``collection_write`` / ``log_append`` return an
actionable failure and persist nothing. Similarity reads return empty on the
same transient failure.
"""

from __future__ import annotations

import difflib
import logging
from abc import abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from similarity.embeddings import is_name_variant

from penny.constants import (
    WRITE_GATE_MUTATING_OUTCOMES,
    MutationEntityType,
    PennyConstants,
    WriteGateOutcome,
)
from penny.database import Database
from penny.database.memory import (
    Collection,
    EntryInput,
    Log,
    LogEntryInput,
    Memory,
    MemoryAccessError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    MessageLogMemory,
    ResolvedEntry,
    ResolvedHit,
    ResolvedKind,
    ResolvedMatch,
    ResolvedMessage,
    RunLog,
    WriteResult,
    render_key,
    render_memory_not_found,
    render_run_calls,
    strip_display_brackets,
)
from penny.database.models import MemoryEntry, MemoryRow
from penny.database.mutation_store import render_mutation
from penny.datetime_utils import format_log_timestamp, format_user_timestamp, user_timezone_name
from penny.llm.call_link import CallLink
from penny.llm.similarity import embed_text
from penny.text_validity import check_extraction_prompt
from penny.tools.base import Tool
from penny.tools.collection_instantiation import (
    SCHEDULE_EXAMPLES,
    Schedule,
    ScheduleError,
    describe_schedule_field,
    expiry_was_normalised,
    has_schedule,
    parse_expires_at,
    parse_schedule,
    render_creation_echo,
    render_expiry_update_line,
    render_inert_echo,
    render_prompt_replacement_diff,
    render_reinstantiation_echo,
    render_tombstone_duplicate,
    sentinel_expiry_note,
)
from penny.tools.job_summary import job_summary_line
from penny.tools.memory_args import (
    CollectionCreateArgs,
    CollectionEntrySpec,
    CollectionGetArgs,
    CollectionKeysArgs,
    CollectionListArgs,
    CollectionNameArgs,
    CollectionRemoveArgs,
    CollectionSetArgs,
    CollectionUpdateArgs,
    CollectionWriteArgs,
    DoneArgs,
    FindArgs,
    GetEventArgs,
    LogAppendArgs,
    LogCreateArgs,
    LogGetArgs,
    MemoryNameArgs,
    ReadLogArgs,
    ReadRunCallsArgs,
    ReadSampleArgs,
)
from penny.tools.models import ToolResult

if TYPE_CHECKING:
    from penny.agents.models import ToolCallRecord
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


# ── Shared formatting ───────────────────────────────────────────────────────


def format_entries(
    entries: list[MemoryEntry],
    *,
    source: str | None = None,
    ordering: str | None = None,
    show_ids: bool = False,
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
    content — unless ``show_ids`` is set, when each line carries ``id=<n>``
    beside its timestamp, the invocation form ``log_get`` accepts; only the
    reads of real stored logs pass it, so every other render is unchanged.
    An empty read names the source and states it's empty (not an error),
    so the model doesn't confuse absence with a failure or re-read the same way.

    PUBLIC because the read tools are no longer its only caller (#1914): the
    collector's composed prompt renders the bound collection's current entries
    through this same function, so the keys a cycle reads AMBIENTLY are in the
    same invocation form the keys it reads from a tool result are.  Two renders
    of the same thing that could drift apart is exactly the defect that made a
    cycle invent a fresh key each time.
    """
    if not entries:
        if source is None:
            return "(no entries)"
        return f"No entries in `{source}` — it's empty or nothing matched (not an error)."
    lines = []
    for index, entry in enumerate(entries, start=1):
        prefix = f"{render_key(entry.key)} " if entry.key else ""
        if show_ids and entry.key is None:
            # A keyless entry renders its id in the invocation form the id-taking
            # read accepts — the mirror of the keyed entry's ``key=`` prefix.
            prefix = f"id={entry.id} "
        stamp = f"[{format_log_timestamp(entry.created_at)}] "
        lines.append(f"{index}. {stamp}{prefix}{entry.content}")
    body = "\n".join(lines)
    if source is None:
        return body
    noun = "entry" if len(entries) == 1 else "entries"
    suffix = f" ({ordering})" if ordering else ""
    return f"{len(entries)} {noun} from `{source}`{suffix}:\n{body}"


# One collection's share of what a run WROTE (#1946) — the count, the collection, and
# the keys, which together are the whole of what a reply can truthfully claim it saved.
_WRITES_LANDED_LINE = "{count} {noun} landed in '{collection}': {keys}"


def render_writes_landed(entries: list[MemoryEntry]) -> str | None:
    """What a run actually WROTE, grouped by collection — the record a reply states
    instead of counting up what it remembers attempting (#1946).

    ``None`` when nothing landed, which is most turns: a run that wrote nothing has no
    writes to narrate, and the frames this feeds fire only when there is a record.

    KEYLESS entries are left out.  A key is the structural mark of a registry write; an
    append with no key is a stream append — browse scratch, above all — and it is the same
    mark the self-state header's own writes clause reads, so the ambient render and this
    one cannot disagree about what a run wrote.  Nothing here knows a tool name: the
    entries come from the ledger's own ``last_written_by_run_id`` stamp, so a routine that
    writes through a plugin's verb lands here exactly like one that uses a built-in.
    """
    landed: dict[str, list[str]] = {}
    for entry in entries:
        if entry.key is None:
            continue
        landed.setdefault(entry.memory_name, []).append(entry.key)
    if not landed:
        return None
    return "\n".join(_writes_landed_line(name, keys) for name, keys in landed.items())


def _writes_landed_line(collection: str, keys: list[str]) -> str:
    return _WRITES_LANDED_LINE.format(
        count=len(keys),
        noun="entry" if len(keys) == 1 else "entries",
        collection=collection,
        keys=", ".join(f"'{key}'" for key in keys),
    )


# What a run that TRIED to keep something and kept nothing is told instead (#2185): the
# plain fact first, then each try with the answer the tool itself gave, so "refused as
# already there" and "the arguments were wrong" read as the different outcomes they are.
_NOTHING_LANDED_HEAD = (
    "Nothing this turn tried to keep is in the store. What each try came back with:"
)
_NOTHING_LANDED_TRY = "- `{tool}`: {outcome}"
# A result that spans lines keeps every line, indented under its own bullet.
_NOTHING_LANDED_CONTINUATION = "\n  "
# A record carries its tool's answer once the call has run, which every record this is
# handed has; a record without one says so rather than rendering a blank.
_NOTHING_LANDED_NO_ANSWER = "(no answer was recorded)"


def render_nothing_landed(attempts: list[ToolCallRecord]) -> str | None:
    """What a run that asked the store to keep something, and kept none of it, is told
    — each try and the tool's own answer to it, verbatim (#2185).

    ``None`` when the run tried nothing, which is what keeps this off every turn that only
    read or talked: a run that never asked to keep anything has no attempt to report, and
    telling it "nothing was saved" would be a record about a question nobody asked.

    The caller hands over only the calls whose tool DECLARES it keeps entries
    (``Tool.stores_entries``); nothing here names a tool.  Each answer is the result the
    tool returned, never a paraphrase of it, because the reply is composed from this and a
    reworded refusal is one the reply can misread."""
    if not attempts:
        return None
    tries = [
        _NOTHING_LANDED_TRY.format(
            tool=attempt.tool,
            outcome=(attempt.result or _NOTHING_LANDED_NO_ANSWER).replace(
                "\n", _NOTHING_LANDED_CONTINUATION
            ),
        )
        for attempt in attempts
    ]
    return "\n".join([_NOTHING_LANDED_HEAD, *tries])


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


def _collection_name(arguments: dict, fallback: str) -> str:
    """Backtick-quoted ``collection`` argument for narration — the six entry
    tools (``collection_get`` / ``collection_keys`` / ``collection_list`` /
    ``collection_sample`` / ``collection_write`` / ``collection_remove``) name
    their target this way; every other memory-keyed tool still reads ``memory``
    (``_memory_name``)."""
    return _named(arguments, "collection", fallback)


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


def _key_did_you_mean(memory: Memory, key: str) -> str:
    """The `` — did you mean key '<nearest>'?`` lead for an entry-key miss (#1674),
    or ``""`` when no existing key in the collection is close (the message then
    stays byte-identical to before — no empty artifact).

    The tools that hit a key miss (``collection_get`` / ``collection_write`` /
    ``collection_remove``)
    carry no embedding client, so this is the string (typo) leg: a stdlib
    ``difflib`` close-match over the collection's OWN keys, gated on the same
    conservative ``DID_YOU_MEAN_STRING_CUTOFF`` the name suggestion uses.  Same
    lexical intent as the write-gate dedup's key signal, without an async embed."""
    keys = memory.keys()
    if not keys:
        return ""
    matches = difflib.get_close_matches(
        key, keys, n=1, cutoff=PennyConstants.DID_YOU_MEAN_STRING_CUTOFF
    )
    return f" — did you mean key '{matches[0]}'?" if matches else ""


class MemoryTool(Tool):
    """Base for tools that operate on a memory.

    ``execute`` runs the subclass's ``_run`` and turns any memory exception into
    the tool's readable result — a ``MemoryAccessError`` (missing / wrong shape /
    read-only) or a ``MemoryAlreadyExistsError`` (create conflict).  Every such
    exception renders its own message, so the handling is uniformly
    ``return str(exc)`` and lives here once: a tool body just calls the op and
    lets the error propagate — no per-tool try/except, no format strings.

    A **name miss** (``MemoryNotFoundError``) is the one exception the base
    enriches rather than passing through verbatim (#1674): it recomputes the
    message here, where the db + (for embedding-capable tools) the embedding
    client live, so the render LEADS with the nearest existing memory — a typo
    of it (string leg, universal) or a semantically-close name (meaning leg,
    where a client is available).  ``_suggestion_embedding`` is the hook an
    embedding-capable name-dispatch tool overrides to add the meaning leg; the
    default (string leg only) fits every read tool with no client.
    """

    # Set by every subclass ``__init__`` — the base reads it to compute a
    # did-you-mean suggestion on a name miss (#1674).
    _db: Database

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        try:
            return await self._run(**kwargs)
        except MemoryNotFoundError as exc:
            return await self._suggest_and_fail(exc)
        except (MemoryAccessError, MemoryAlreadyExistsError) as exc:
            # Wrong-shape / read-only refusals are failed calls.
            return ToolResult(message=str(exc), success=False)

    async def _suggest_and_fail(self, exc: MemoryNotFoundError) -> ToolResult:
        """Re-render a name miss leading with the nearest existing memory (#1674).

        The string (typo) leg always runs (no embedding); the meaning leg runs only
        when this tool supplied an embedding of the missed name.  No close candidate
        → ``suggestion`` is ``None`` and the message is byte-identical to before."""
        embedding = await self._suggestion_embedding(exc.name)
        suggestion = self._db.memories.nearest_memory_name(exc.name, embedding)
        return ToolResult(message=render_memory_not_found(exc.name, suggestion), success=False)

    async def _suggestion_embedding(self, name: str) -> list[float] | None:
        """The embedding of a missed name for the MEANING leg of a did-you-mean
        suggestion, or ``None`` (string leg only) for a tool with no embedding
        client.  Overridden by the embedding-capable name-dispatch tools (#1674)."""
        return None

    @abstractmethod
    async def _run(self, **kwargs: Any) -> ToolResult:
        """Resolve/create a memory and operate on it; let any memory exception
        propagate to :meth:`execute`."""


def _format_collection_echo(
    memory: Any,
    verb: str,
    timezone_name: str | None,
    priors: Mapping[str, str | None],
) -> str:
    """Render a created or updated collection as a structured echo.

    The chat agent uses this to confirm-back what landed without
    making up fields.  Includes the full extraction_prompt verbatim
    so the model's reply can summarize accurately (the model previously
    confabulated this because the create/update returns were one-liners).
    The schedule renders via ``describe_schedule_field`` — plain words with the exact
    rule in a code span (#2425) when the collection has one, else ``schedule: none`` for
    an inert/unscheduled one, so a plain update on an unscheduled collection never
    echoes a blank clause (#1666).
    The end condition renders too (#2251 — this echo used to carry no expiry line at
    all, so a call that gave a job its first end had nothing confirming it landed),
    on the user's own clock (#2206) and as ``was … → now …`` when this call moved it.
    """
    return (
        f"{verb} collection '{memory.name}':\n"
        f"  schedule: {describe_schedule_field(memory, timezone_name)}\n"
        f"  notify: {memory.notify}\n"
        f"  expires: {render_expiry_update_line(memory, timezone_name, priors)}\n"
        f"  description: {memory.description}\n"
        f"  steps: |\n    "
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
# ``memory_get`` (the single exact-name lookup) renders this lifecycle
# block, so one read answers the whole "who asked for it, what run created it,
# is it live, when does it end" question.  It's sourced structurally from the
# registry columns, never reconstructed — so an archived mechanism stays
# inspectable by name, marked with its archive time rather than vanishing.

_STATUS_ACTIVE = "active"
_STATUS_ARCHIVED = "archived"
_EXPIRES_NEVER = "never"


def _ask_excerpt(db: Database, source_message_id: int | None) -> str | None:
    """The user message that spawned a mechanism, whitespace collapsed so a
    multi-line ask reads as one line and rendered whole — no cap (#2642) — or
    ``None`` when there is no source message (seeded / system rows) or the row
    can't be found."""
    if source_message_id is None:
        return None
    message = db.messages.get_by_id(source_message_id)
    if message is None:
        return None
    return " ".join(message.content.split())


def _status_line(row: MemoryRow) -> str:
    """``status: active`` or ``status: archived <UTC datetime>``.  A just-archived
    row renders clearly marked, never absent; the archive timestamp is
    ``updated_at``, which ``MemoryStore.archive`` stamps at archive time."""
    if row.archived:
        return f"status: {_STATUS_ARCHIVED} {format_log_timestamp(row.updated_at)}"
    return f"status: {_STATUS_ACTIVE}"


def _expires_line(row: MemoryRow, timezone_name: str | None) -> str:
    """``expires: <datetime>`` on the USER's own clock (#2206/#2251 — the same
    clock a stated end is read on) when an end condition is set, else
    ``expires: never``."""
    if row.expires_at is not None:
        return f"expires: {format_user_timestamp(row.expires_at, timezone_name)}"
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
        _expires_line(row, user_timezone_name(db)),
        _created_line(row, _ask_excerpt(db, row.source_message_id)),
    ]


def _recent_changes_block(db: Database, name: str) -> list[str]:
    """The collection's recent config-change history from the mutation ledger
    (#1560) — each line naming its run id, so "when was this archived, and by
    what?" is a read, and the surface stays an anchor (the run is one hop away,
    never guessed).  Bounded by the shared ``RUN_HISTORY_RECORDS`` (the same
    recent-N used for a memory's run history), so no new limit is invented.  Empty
    (no block) for a collection with no recorded mutations (seeded / migration
    rows)."""
    events = db.mutations.history(
        name,
        PennyConstants.RUN_HISTORY_RECORDS,
        # This block is a COLLECTION's history, so it is scoped to one (#1902): the ledger
        # carries routines too, and a routine sharing this collection's name would
        # otherwise render as changes to the collection.
        entity_type=MutationEntityType.COLLECTION,
    )
    if not events:
        return []
    return ["", "Recent changes (newest first):", *(f"  {render_mutation(e)}" for e in events)]


# ── Metadata ────────────────────────────────────────────────────────────────


# A schedule whose UNTIL= and an ``expires_at`` argument both say when to stop: two
# answers to one question, so neither is taken and the model is asked to say it once.
_CONFLICTING_END_CONDITION = (
    "This says when to stop twice — the schedule ends at UNTIL and expires_at says "
    "'{expires}'. Say it once: keep UNTIL in the schedule and drop expires_at, or drop "
    "UNTIL and pass expires_at on its own."
)


def resolve_expiry(db: Database, schedule: Schedule, expires_at: str | None) -> datetime | None:
    """The collection's ONE end condition, from the two places it can be stated (#1857).

    A schedule's ``UNTIL=`` is lifted onto the same column ``expires_at`` sets, so
    stating both is stating the answer twice — refused rather than silently resolved in
    one direction.  Words are read in the user's own timezone and counted from the
    database's own clock, which is why this takes the database: the zone and the instant
    are fetched here and passed to the pure parser.

    ``None`` means the collection has no end condition — either none was stated, or the
    one stated was a far-future sentinel the parser read as "forever" (#1944), which the
    caller names via ``sentinel_expiry_note``.
    """
    if expires_at is None:
        return schedule.expires_at
    if schedule.expires_at is not None:
        raise ScheduleError(_CONFLICTING_END_CONDITION.format(expires=expires_at))
    return parse_expires_at(expires_at, user_timezone_name(db), db.clock.now())


# A schedule / notify / expiry on a prompt-less create: an inert collection has no job
# to describe, so the job-shaped arg is refused with the two-step fix.
_INERT_JOB_ARGS_REFUSAL = (
    "Can't set a schedule, notify, or expiry on '{name}' without a prompt — those describe "
    "a JOB, and a prompt-less collection is inert storage with no job to run. Create it as "
    "storage now (name + description only), then attach a job later with "
    "collection_configure(collection='{name}', prompt=<what the job's cycles should do>, "
    "schedule=<recurrence rule>, notify=<true/false>)."
)

# A job create with no schedule: the collection wouldn't know when to run, so it's
# refused showing the grammar (a job collection must be scheduled, #1857).
_MISSING_SCHEDULE = (
    "A job needs a schedule so it knows when to run. Set schedule to one "
    "RRULE line:\n"
    f"{SCHEDULE_EXAMPLES}"
)

# The ``expires_at`` argument description — stated ONCE here and read by every surface
# that declares the argument (create, update, and the ``collection_configure`` front door,
# #2250): before this, ``collection_configure``'s own schema carried a bare
# ``{"type": "string"}`` for it with no description at all, so chat never saw what
# create/update behind it already say. States plainly that the value is read on the
# user's own clock, with no zone — the model never converts one (#2206).
EXPIRES_AT_PARAM_DESCRIPTION = (
    "When it ends — a date and time on your own clock, written with no zone, like "
    "'2026-11-02T22:00:00'. Leave it out when there's no end date."
)

# The ``schedule`` argument's description — stated ONCE here and read by every surface
# that declares the argument (create, update, and the ``collection_configure`` front door,
# #2222): before this, ``collection_configure``'s own schema carried a bare
# ``{"type": "string"}`` for it with no description at all, so chat never saw what
# create/update behind it already say — measured on gemma, "check it every morning"
# landed with no hour at all in a third of calls and a different guessed hour in the
# rest (case ``transition-request-to-apply``, 10/15). States the hour a named part of
# the day means when the user gives no hour of their own (code-owner ruling).
SCHEDULE_PARAM_DESCRIPTION = (
    "When it runs, as one recurrence rule: 'FREQ=HOURLY' (every hour), "
    "'FREQ=MINUTELY;INTERVAL=90' (every 90 minutes), 'FREQ=DAILY;BYHOUR=8' "
    "(8am daily), 'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9' (weekdays at "
    "9). BYHOUR is the hour on the user's own clock — write the hour "
    "they said. When they name a part of the day instead of an hour, write BYHOUR "
    "as morning=8, afternoon=14, evening=19, night=22. Add COUNT= to run a set "
    "number of times, and a leading 'DTSTART:<time>' line to say when it starts; "
    "a DTSTART stamp ending in Z is an exact moment in UTC."
)

# The ``prompt`` argument's description — stated ONCE here and read by every surface
# that declares the argument, the way ``schedule`` / ``expires_at`` do (#2250/#2222):
# before this, ``collection_configure``'s schema carried a bare ``{"type": "string"}``
# for it, so the omit-to-keep rule (#2719) lived only in the prose and the arg gate's
# blank-refusal envelope rendered the field bare.
PROMPT_PARAM_DESCRIPTION = (
    "The job's instructions, as free text. Leave it OUT to change only "
    "schedule / expiry / notify — the stored prompt stays exactly as it is. "
    "Supplying it REPLACES the whole instructions text."
)


class CollectionCreateTool(MemoryTool):
    """Set up a job from a free-text prompt, or an INERT storage container —
    the front door of collector creation (#2455 naive baseline; was #1591).

    A collection is never authored from a learned skill any more: it names a
    ``prompt`` — the model's own free-text instructions — and that text becomes the
    collection's ``extraction_prompt`` VERBATIM, with no rendering seam in between.

    Idempotency at birth (#1567/#1775) lives one level up, on ``CollectionSetTool``
    — the registered front door, and the only place that can answer a duplicate by
    UPDATING the collection we already have instead of refusing.  By the time this
    class runs, the target is known to be a new job.
    """

    name = "collection_configure"
    # The creation / re-render echo points at the metadata read for what was configured.
    advises = ("memory_get",)
    description = (
        "Create or reconfigure a background job in ONE idempotent call. A collection "
        "is storage plus an OPTIONAL job.\n"
        "\n"
        "TWO ways to call it:\n"
        "- WITH a `prompt`: the collection becomes a JOB — the text you write is what "
        "the job's cycles follow each time they run, on the `schedule` you set.\n"
        "- WITHOUT a `prompt`: an INERT storage collection — it holds entries but "
        "nothing runs against it yet. Don't pass a schedule / notify / expiry with a "
        "prompt-less create — an inert collection has no job to run.\n"
        "\n"
        "Fields:\n"
        "- `collection` — unique slug for the collection (lowercase, hyphens).\n"
        "- `description` — REQUIRED. What this collection is for, in the user's own words "
        '("keep an eye on the price of that jacket"). It is the collection\'s meaning '
        "anchor (how find resolves it), so get it right and confirm it back.\n"
        "- `prompt` — OPTIONAL. The job's instructions, written as free text — what to "
        "read, what to do with it, and what to remember, as you would explain the task "
        "to someone doing it once. Omit it for an inert storage collection.\n"
        "- `schedule` — when it runs, as ONE recurrence rule (RRULE). Say how often "
        'with FREQ and INTERVAL, and the time of day with BYHOUR: "FREQ=HOURLY" is '
        'every hour, "FREQ=MINUTELY;INTERVAL=90" is every 90 minutes, '
        '"FREQ=DAILY;BYHOUR=8" is 8am daily, '
        '"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9" is weekdays at 9. BYHOUR is the '
        "hour on the user's own clock, so write the hour they said and leave the "
        "converting to me. To run a set number of times add "
        'COUNT — "DTSTART:20260720T090000Z\\nFREQ=DAILY;COUNT=1" runs once at that '
        "moment and then retires; a DTSTART stamp ending in Z is an exact moment in UTC, "
        "so that one you do convert. Put a DTSTART line first whenever the user said when "
        "to start. A rule I can't read comes back with the grammar and, when I can work "
        "out what you meant, the exact string to send instead.\n"
        f"- `expires_at` — OPTIONAL. {EXPIRES_AT_PARAM_DESCRIPTION} The collection "
        "archives itself once it passes, so a bounded watch needs no teardown.\n"
        "- `notify` — Set `true` when the user wants to be told about / kept posted "
        "on / alerted to new or changed entries as they're found. Leave `false` (the "
        "default) for a silent collection they'll ask about later.\n"
        "\n"
        "Returns a structured echo of what landed (description, prompt, schedule, "
        "notify, expiry). Confirm it back — don't invent fields it didn't return. "
        "Plain storage needs no call at all — collection_write creates storage "
        "automatically; use collection_configure for jobs and config."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {
                "type": "string",
                "description": (
                    "Unique collection name (slug-style: lowercase, hyphens). Build it "
                    "from the user's own words for this task — the specific place, page, "
                    "or thing they named — so two similar jobs get names that tell them "
                    "apart."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "REQUIRED. What this collection is for, in the user's words — the goal "
                    'it serves ("a running list of good retro JRPGs to play"), not the '
                    "mechanism. The meaning anchor; confirm it back."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "OPTIONAL. The job's instructions, as free text: what to read, what "
                    "to do with it, and what to remember. Omit for an inert storage "
                    "collection."
                ),
            },
            "schedule": {
                "type": "string",
                "description": SCHEDULE_PARAM_DESCRIPTION,
            },
            "expires_at": {
                "type": "string",
                "description": EXPIRES_AT_PARAM_DESCRIPTION,
            },
            "notify": {
                "type": "boolean",
                "description": (
                    "true = tell the user about new/changed entries (they asked to be told / "
                    "kept posted / alerted); false (default) = silent."
                ),
            },
        },
        "required": ["collection", "description"],
    }
    args_model = CollectionCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "collection", "a new collection")
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
        """A prompt-less create yields an INERT storage collection; a prompt create
        stands up the job.  The two paths read as a table of contents."""
        args = CollectionCreateArgs(**kwargs)
        if args.prompt is None:
            return await self._create_inert(args)
        return await self._create_with_prompt(args, args.prompt)

    async def _create_with_prompt(self, args: CollectionCreateArgs, prompt: str) -> ToolResult:
        """Stand up a job from the model's own free-text prompt: parse the schedule,
        validate the prompt's length, then create and echo."""
        parsed = self._parse_schedule(args)
        if isinstance(parsed, ToolResult):
            return parsed
        schedule, expires_at = parsed
        if (too_short := check_extraction_prompt(prompt)) is not None:
            return ToolResult(message=too_short, success=False)
        return await self._instantiate(args, prompt, schedule, expires_at)

    async def _create_inert(self, args: CollectionCreateArgs) -> ToolResult:
        """A prompt-less create: an INERT collection — storage only, no
        ``extraction_prompt`` / cadence / notify, so it never dispatches (the
        dispatcher selects on ``extraction_prompt IS NOT NULL``).  A job-shaped arg
        alongside is refused (an inert container has no job); the echo is honest
        about being storage-only."""
        if job_error := self._reject_job_args(args):
            return job_error
        description_embedding = await embed_text(self._llm_client, args.description)
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
        """A prompt-less (inert) create describes storage, not a job — a schedule /
        notify / expiry passed alongside has nothing to attach to, so it's refused
        naming the fix (attach a prompt via collection_configure) rather than silently
        dropped (visible degradation over silent success)."""
        has_job = any((args.schedule is not None, args.expires_at is not None, args.notify))
        if not has_job:
            return None
        return ToolResult(message=_INERT_JOB_ARGS_REFUSAL.format(name=args.name), success=False)

    def _parse_schedule(
        self, args: CollectionCreateArgs
    ) -> ToolResult | tuple[Schedule, datetime | None]:
        """Parse the single ``schedule`` arg + optional end condition before any other
        work, so a bad schedule fails fast.  A job collection needs a schedule (it must
        know when to run), so a missing one is refused showing the grammar; an
        unreadable one surfaces ``parse_schedule``'s teaching rejection verbatim."""
        if args.schedule is None:
            return ToolResult(message=_MISSING_SCHEDULE, success=False)
        try:
            schedule = parse_schedule(args.schedule)
            expires_at = resolve_expiry(self._db, schedule, args.expires_at)
        except ScheduleError as exc:
            return ToolResult(message=str(exc), success=False)
        return schedule, expires_at

    async def _instantiate(
        self,
        args: CollectionCreateArgs,
        extraction_prompt: str,
        schedule: Schedule,
        expires_at: datetime | None,
    ) -> ToolResult:
        """Create the collection and echo it.

        ``description`` is the routing meaning anchor; ``notify`` drives the
        run-time notify suffix on the collector (#1557 — the sole emission path since
        the notifier consumer was retired)."""
        description_embedding = await embed_text(self._llm_client, args.description)
        memory = self._db.memories.create_collection(
            args.name,
            args.description,
            extraction_prompt=extraction_prompt,
            schedule=schedule.rule,
            description_embedding=description_embedding,
            notify=args.notify,
            created_by_run_id=self._created_by_run_id,
            expires_at=expires_at,
            max_runs=schedule.max_runs,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        echo = render_creation_echo(memory, user_timezone_name(self._db))
        expiry_note = sentinel_expiry_note(args.expires_at, expires_at)
        return ToolResult(message=f"{echo}{suffix}{expiry_note}", mutated=True)


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
            "log": {"type": "string", "description": "Unique log name"},
            "description": {
                "type": "string",
                "description": "Content-reflective one-line summary (the meaning anchor)",
            },
        },
        "required": ["log", "description"],
    }
    args_model = LogCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "log", "a new log")
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
            args.log,
            args.description,
            description_embedding=description_embedding,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        message = f"Created log '{args.log}'.{suffix}"
        return ToolResult(message=message, mutated=True)


class CollectionArchiveTool(MemoryTool):
    """Archive a collection — keeps its data but retires the mechanism."""

    name = "collection_archive"
    description = (
        "Archive a collection. The data stays intact but the collection is "
        "retired — its collector stops running and it drops out of the active "
        "memory list (it stays reachable by its exact name, as a tombstone) "
        "until unarchived."
    )
    parameters = {
        "type": "object",
        "properties": {"collection": {"type": "string"}},
        "required": ["collection"],
    }
    args_model = CollectionNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _named(arguments, "collection", "a collection")
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
        args = CollectionNameArgs(**kwargs)
        self._db.memories.archive(args.collection, run_id=self._run_id)
        return ToolResult(message=f"Archived '{args.collection}'.", mutated=True)


class CollectionUnarchiveTool(MemoryTool):
    """Restore a previously archived collection — its collector resumes."""

    name = "collection_unarchive"
    description = "Unarchive a previously archived collection."
    parameters = {
        "type": "object",
        "properties": {"collection": {"type": "string"}},
        "required": ["collection"],
    }
    args_model = CollectionNameArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        memory = _named(arguments, "collection", "a collection")
        if not result.success:
            return f"You tried to restore {memory} from the archive but it didn't work:"
        return f"You restored {memory} from the archive:"

    def __init__(self, db: Database, run_id: str | None = None) -> None:
        self._db = db
        self._run_id = run_id

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionNameArgs(**kwargs)
        self._db.memories.unarchive(args.collection, run_id=self._run_id)
        return ToolResult(message=f"Unarchived '{args.collection}'.", mutated=True)


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionListTool(MemoryTool):
    """Return the newest entries in a collection, newest first.

    Collection-only: logs are read through the cursored ``log_read``, never
    through a newest-first scan (which would bypass the cursor and silently miss
    entries).  A log target gets a readable refusal.
    """

    name = "collection_list"
    # Collections only — the description sends a log read to the cursored reader.
    advises = ("log_read",)
    description = (
        "Return the newest entries in a collection, newest first. Omit `k` to "
        "return every entry. Collections only — to read a log use `log_read(<log>)`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
            "k": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["collection"],
    }
    args_model = CollectionListArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = _collection_name(arguments, "collection")
        if not result.success:
            return f"You tried to look up your {collection} but it didn't work:"
        return f"You looked up your {collection}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionListArgs(**kwargs)
        entries = _resolve(self._db, args.collection).read_latest(args.k)
        return ToolResult(
            message=format_entries(entries, source=args.collection, ordering="most recent first")
        )


class CollectionGetTool(MemoryTool):
    """Look up one entry by its exact key."""

    name = "collection_get"
    # A key that does not exist sends the model to the list of keys, a meaning
    # search, or a fresh write under the key it actually has.
    advises = ("collection_keys", "find", "collection_write")
    description = (
        "Look up an entry by its exact key in a collection. Returns the entry's "
        "content if found, or a 'not found' message otherwise."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
            "key": {"type": "string"},
        },
        "required": ["collection", "key"],
    }
    args_model = CollectionGetArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = _collection_name(arguments, "a collection")
        entry = _entry_label(arguments)
        if not result.success:
            return f"You tried to look up {entry} in {collection} but it didn't work:"
        return f"You looked up {entry} in {collection}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionGetArgs(**kwargs)
        collection = _resolve(self._db, args.collection)
        rows = collection.get(args.key)
        if not rows:
            rejection = _bracket_key_rejection(collection, args.collection, args.key)
            if rejection is not None:
                return rejection
            did_you_mean = _key_did_you_mean(collection, args.key)
            return ToolResult(
                message=f"Key '{args.key}' not found in '{args.collection}'{did_you_mean or '.'} "
                f"List the available "
                f"keys with collection_keys('{args.collection}'), or search by meaning with "
                f"find(query=<what you're looking for>, memory='{args.collection}'). "
                f"If it exists under a different key, write the new content there: "
                f"collection_write(collection='{args.collection}', "
                f"entries=[{{'key': <the key you found>, 'content': <the new content>}}]) — "
                f"the store replaces the content of the key you name."
            )
        return ToolResult(message=format_entries(rows, source=args.collection))


class CollectionSampleTool(MemoryTool):
    """Pick entries at random from a collection."""

    name = "collection_sample"
    description = "Return `k` entries sampled uniformly at random. Omit `k` to return all."
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["collection"],
    }
    args_model = ReadSampleArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = _collection_name(arguments, "a collection")
        if not result.success:
            return f"You tried to pull a random sample from {collection} but it didn't work:"
        return f"You pulled a random sample from {collection}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadSampleArgs(**kwargs)
        entries = _resolve(self._db, args.collection).read_random(args.k)
        return ToolResult(
            message=format_entries(entries, source=args.collection, ordering="random sample")
        )


class CollectionKeysTool(MemoryTool):
    """List the keys currently stored in a collection."""

    name = "collection_keys"
    description = "List the unique keys in a collection (insertion order)."
    parameters = {
        "type": "object",
        "properties": {"collection": {"type": "string"}},
        "required": ["collection"],
    }
    args_model = CollectionKeysArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = _collection_name(arguments, "a collection")
        if not result.success:
            return f"You tried to list the keys in {collection} but it didn't work:"
        return f"You listed the keys in {collection}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionKeysArgs(**kwargs)
        keys = _resolve(self._db, args.collection).keys()
        if not keys:
            return ToolResult(
                message=f"No keys in `{args.collection}` — the collection is empty (not an error)."
            )
        return ToolResult(message="\n".join(f"- {key}" for key in keys))


# ── Collection writes ───────────────────────────────────────────────────────


_SCOPE_REFUSAL_MESSAGE = "Refused: this collector can only write to '{scope}', not '{collection}'."

# The standing-rules guard (#2633): the `rules` collection renders in full in every
# prompt as standing instructions, so a page (or any job input) that says "always do
# X" must not be able to write itself into every prompt through a job cycle.  Only
# the user's own chat turn changes it.  Keyed to the STRUCTURAL mark of a collector
# context — the non-None `_scope` binding every entry-mutating tool already carries —
# never to a tool-name list, so any future entry-mutating tool built with a scope
# inherits the guard.  Refuses before the scope refusal, so the honest, actionable
# answer is what the model reads rather than the generic one.
_RULES_GUARD_REFUSAL = (
    "`rules` holds only standing instructions the user gives you in chat, so a job "
    "can't change it. If this job found something the user might want as a rule, say "
    "so in its notification and let them decide."
)


def _rules_guard(scope: str | None, collection: str) -> ToolResult | None:
    """The standing-rules refusal on a collector-scoped call targeting `rules`.

    ``None`` when the call is a chat turn's (``scope=None`` — its writes to `rules`
    are the one path rules are kept by) or targets any other collection."""
    if scope is None or collection != PennyConstants.MEMORY_RULES_COLLECTION:
        return None
    return ToolResult(message=_RULES_GUARD_REFUSAL, success=False)


# Write-target materialization (#1570): a chat write never dies on a name miss.
# A near-certain name variant REDIRECTS (narrated, never silent); nothing close
# AUTO-CREATES the collection (narrated).  The middle band keeps the
# did-you-mean refusal.  Gates calibrated in ``PennyConstants`` (see the
# MEMORY_NAME_REDIRECT_* comment for the archive-derived table).
_WRITE_REDIRECT_NOTE = (
    "Note: no collection named '{asked}' exists — '{match}' is the same name in a "
    "different spelling, so this write went there.\n"
)
_WRITE_AUTOCREATE_NOTE = (
    "Note: no collection named '{name}' existed and nothing close matched, so it was "
    "created (storage only, no job) and your entries went in.\n"
)
_AUTO_CREATED_DESCRIPTION = PennyConstants.AUTO_CREATED_DESCRIPTION_PREFIX + " '{key}'"


def _is_name_variant(asked: str, existing: str) -> bool:
    """Near-certain 'same collection, different spelling' (#2270) — ``is_name_variant``
    lives in the dependency-light ``similarity`` package, over the one pair of
    thresholds, so the near-match rule is declared once rather than copied."""
    return is_name_variant(
        asked,
        existing,
        char_ratio=PennyConstants.MEMORY_NAME_REDIRECT_CHAR_RATIO,
        tcr=PennyConstants.MEMORY_NAME_REDIRECT_TCR,
    )


# The DIVERGENT-VALUE rejection (#1405, restated by #1919).  Every ``DUPLICATE`` is a
# collision whose CONTENT differs now — the same-value case has its own outcome and its
# own STOP — so what this write carries is NEWS about an entry that already exists, and
# there is exactly one call that lands it.
#
# It states three facts and offers nothing: the collection already holds this subject
# under a key it NAMES, any other key collides with that same entry again, and here is
# the call.  What it replaced ended "…to refresh it if you have richer info", which reads
# as advice to weigh rather than as the situation — measured, 4 of 5 samples filed the
# new value under a fresh key instead of landing it on the matched one.  The matched key
# is rendered VERBATIM inside the call, so the recovery is a copy rather than a guess.
_DUPLICATE_REJECTION = (
    "'{key}' was not written — this collection already holds an entry about that, stored "
    "under the key '{matched}', and what you wrote is a different value from what that "
    "entry holds. Any other key you write it under collides with that same entry again. "
    "To land the new value: collection_write(collection='{collection}', entries=[{{'key': "
    "'{matched}', 'content': <the new value>}}]) — the store replaces the content of the "
    "key you name."
)
# The matched entry has no key, so no keyed write can address it — named as
# such rather than dangling an imperative the model cannot act on.
_DUPLICATE_REJECTION_KEYLESS = (
    "'{key}' was not written — it collides with an existing entry that has no key, and a "
    "keyed write cannot replace a keyless entry. Leave it out, or write content that does "
    "not collide"
)

# The trailing framing after the per-entry rejections.  Each one already names its own
# call, so the close says only how much of the batch is in that position — and neither
# line hedges, because since #1919 there is no "only if you have richer info" case left
# for a DUPLICATE to be in.
_DUPLICATE_CLOSE_ALL = "Nothing in this batch landed — each one belongs on the entry named above."
_DUPLICATE_CLOSE_SOME = "The rest of the batch landed; these belong on the entries named above."


def _format_duplicate(collection: str, result: WriteResult) -> str:
    """One self-contained rejection for a rejected candidate: what this collection already
    holds it under, that no other key escapes the collision, and the exact call that lands
    the new value.

    Self-contained because the recovery has to be readable without the lines around it —
    the failure this addresses was the model treating the matched key as one option among
    several and filing its value somewhere new.  ``collection`` is the write's resolved
    target, so the recovery call is copyable verbatim whatever redirect produced it."""
    if result.matched_key:
        return _DUPLICATE_REJECTION.format(
            key=result.key, matched=result.matched_key, collection=collection
        )
    return _DUPLICATE_REJECTION_KEYLESS.format(key=result.key)


def _format_changed(results: list[WriteResult]) -> str:
    """The change-gate CHANGED part (#1587/#1633): the exact key already existed with
    a DIFFERENT value — the observed value changed, so the write gate auto-refreshed
    the stored baseline IN PLACE (through the shared update path, stamping the writing
    run).  Nothing further is needed — the refresh already happened; the next
    observation of the same value reads as UNCHANGED."""
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


def _format_duplicate_unchanged(results: list[WriteResult]) -> str:
    """The DUPLICATE_UNCHANGED part (#1919): the value is already recorded under a
    DIFFERENT key, so there is nothing new — ``KEY_EXISTS_UNCHANGED``'s twin for the
    reworded-key path, and STOP-worthy for the same reason.

    It names the key the value is filed under rather than the one this write proposed,
    because that is the entry that holds it now; and, unlike the divergent-value
    rejection beside it, it prescribes NO call — there is nothing to do about a value
    that is already stored, and landing it again would be an instruction to rewrite an
    entry with what it already says."""
    labelled = ", ".join(f"'{r.key}' is already stored as '{r.matched_key}'" for r in results)
    noun = "entry" if len(results) == 1 else "entries"
    return f"Already recorded: {labelled} — no change since the last write ({noun})."


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
    "collection_write(collection='{collection}', entries=<the same entries>) in a moment."
)
_EMBED_WRITE_FAILURE_LOG = (
    "Couldn't embed this entry (a transient embedding error) — nothing was "
    "appended, since an entry is only stored once it has a similarity vector. "
    "Retry log_append(log='{log}', content=<the same content>) in a moment."
)


class CollectionWriteTool(MemoryTool):
    """Write entries to a collection with similarity-based dedup."""

    name = "collection_write"
    # ``collection_remove`` rides beside the write because the collector's
    # corrections runtime rule names the pair together: a routine that writes may find
    # an entry that should be removed rather than rewritten, and offering one half of
    # that without the other would teach a correction the surface cannot carry out.
    advises = ("collection_remove",)
    stores_entries = True
    destination_param = "collection"
    description = (
        "Write one or more entries to a collection. Each entry has a short "
        "`key` (topic/identifier) and a longer `content` body. Dedup runs "
        "per entry — duplicates are reported but not treated as errors."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
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
        "required": ["collection", "entries"],
    }
    args_model = CollectionWriteArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = _collection_name(arguments, "a collection")
        if not result.success:
            return f"You tried to save to {collection} but it didn't work:"
        if not result.mutated:
            return f"You didn't add anything new to {collection} — it was already there:"
        return f"You saved {_written_label(arguments)} to {collection}:"

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

    async def _suggestion_embedding(self, name: str) -> list[float] | None:
        # This tool embeds, so a write to a mistyped collection gets the MEANING
        # leg too (#1674) — the motivating 'aurora-deone' write path.
        return await embed_text(self._llm, name)

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionWriteArgs(**kwargs)
        if (guard := _rules_guard(self._scope, args.collection)) is not None:
            return guard
        if self._scope is not None and args.collection != self._scope:
            return ToolResult(
                message=_SCOPE_REFUSAL_MESSAGE.format(
                    scope=self._scope, collection=args.collection
                ),
                success=False,
            )
        target, note = await self._materialize_target(args)
        collection = _resolve(self._db, target)
        entries = [await self._build_entry(spec) for spec in args.entries]
        embed_failure = self._embed_failure(target, entries)
        if embed_failure is not None:
            return embed_failure
        results = collection.write(entries, author=self._author, run_id=self._run_id)
        result = self._format_results(target, results)
        if note:
            return result.model_copy(update={"message": note + result.message})
        return result

    async def _materialize_target(self, args: CollectionWriteArgs) -> tuple[str, str]:
        """Resolve the write target, materializing on a miss (#1570): a
        near-certain name variant REDIRECTS to the existing collection; nothing
        close AUTO-CREATES it (chat surface only — a scoped write is pinned to a
        collection that exists).  Both narrate; the middle band re-raises into
        the did-you-mean refusal.  Returns (actual target, note-or-empty)."""
        if self._scope is not None or self._db.memories.get(args.collection) is not None:
            return args.collection, ""
        variants = [
            row.name
            for row in self._db.memories.list_all()
            if row.type == MemoryType.COLLECTION.value
            and not row.archived
            and _is_name_variant(args.collection, row.name)
        ]
        if len(variants) == 1:
            logger.info("collection_write: redirected '%s' → '%s'", args.collection, variants[0])
            return variants[0], _WRITE_REDIRECT_NOTE.format(
                asked=args.collection, match=variants[0]
            )
        if variants:
            # Two-plus near-variants is genuine ambiguity — the did-you-mean
            # refusal (raised by _resolve on the original name) owns it.
            return args.collection, ""
        description = _AUTO_CREATED_DESCRIPTION.format(key=args.entries[0].key)
        self._db.memories.create_collection(
            args.collection,
            description,
            description_embedding=await embed_text(self._llm, description),
            created_by_run_id=self._run_id,
        )
        logger.info("collection_write: auto-created collection '%s'", args.collection)
        return args.collection, _WRITE_AUTOCREATE_NOTE.format(name=args.collection)

    def _embed_failure(self, collection: str, entries: list[EntryInput]) -> ToolResult | None:
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
        logger.warning("collection_write: embedding unavailable for %s in %s", keys, collection)
        return ToolResult(
            message=_EMBED_WRITE_FAILURE_COLLECTION.format(keys=keys, collection=collection),
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
        throttle."""
        by = self._bucket(results)
        self._log_rejections(memory, by)
        parts = self._message_parts(memory, by)
        message = " ".join(parts) if parts else "(no entries written)"
        return ToolResult(
            message=message,
            mutated=any(result.outcome in WRITE_GATE_MUTATING_OUTCOMES for result in results),
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
        already = by[WriteGateOutcome.DUPLICATE_UNCHANGED]
        duplicates = by[WriteGateOutcome.DUPLICATE]
        degenerate = by[WriteGateOutcome.DEGENERATE]
        # "Nothing landed" = the batch was ONLY divergent-value duplicates: the close then
        # names there is nothing new, not just per-entry refreshes.  An already-recorded
        # entry landed nothing either, so it does not count as something having happened.
        nothing_landed = not (written or changed or unchanged or degenerate)
        unexpected = by[WriteGateOutcome.UNEXPECTED]
        parts = [
            _format_written(memory, written) if written else None,
            _format_changed(changed) if changed else None,
            _format_unchanged(unchanged) if unchanged else None,
            _format_duplicate_unchanged(already) if already else None,
            self._duplicate_part(memory, duplicates, all_duplicates=nothing_landed)
            if duplicates
            else None,
            _format_degenerate(degenerate) if degenerate else None,
            _format_unexpected(unexpected) if unexpected else None,
        ]
        return [part for part in parts if part]

    def _duplicate_part(
        self, memory: str, duplicates: list[WriteResult], *, all_duplicates: bool
    ) -> str:
        """The DUPLICATE part: one self-contained rejection per entry + the trailing close.

        One per LINE rather than joined into a sentence: each rejection names a key, a call
        and an argument, and running several of them together is what makes a matched key
        hard to pick out and copy.  There is no "rejected as duplicates" lead-in any more
        (#1919) — these are not duplicates, they are different values for entries that
        exist, and calling them duplicates is the framing that had the model write them
        somewhere else."""
        rejections = "\n".join(_format_duplicate(memory, result) for result in duplicates)
        return f"{rejections}\n{self._duplicate_close(all_duplicates=all_duplicates)}"

    @staticmethod
    def _duplicate_close(*, all_duplicates: bool) -> str:
        """The trailing framing after the per-entry rejections (each already names its
        own ``collection_write`` call).  It says how much of the batch is in that
        position and names no call of its own — the actionable one is per entry, where
        the key it needs is."""
        if all_duplicates:
            return _DUPLICATE_CLOSE_ALL
        return _DUPLICATE_CLOSE_SOME


# An adopt that attached a prompt but no schedule: the collection has a job yet
# nothing to run it on, so it won't dispatch until one is set (visible over silent).
_NO_SCHEDULE_NOTE = (
    "\n\nHeads up: this collection now has a prompt but no schedule, so it won't run "
    "yet. Set one with collection_configure(collection='{name}', schedule=<recurrence rule>) — the "
    "rule for the cadence the user asked for, e.g. FREQ=HOURLY or FREQ=DAILY;BYHOUR=8."
)


# A ``prompt`` that is the job's own one-line summary from the self-state Jobs line
# (#2685) — copied back verbatim, it would replace the job's real instructions with
# a rendering OF them, and the collector would then run that rendering.  Refused,
# not stripped: nothing is normalised on her behalf, she gets the refusal and
# retries with real instructions or leaves ``prompt`` out.
_SUMMARY_AS_PROMPT_REFUSAL = (
    "That's how your current state summarises '{name}', not its instructions — "
    "leave `prompt` out to keep them, or pass the full new instructions."
)


def _summary_as_prompt_refusal(row: MemoryRow, prompt: str) -> str | None:
    """The refusal when ``prompt`` is the job's own current-state summary line
    (#2685), else ``None``.

    Matched against ``job_summary_line`` — the SAME render the Jobs section hangs
    under the row, so what the tool refuses is exactly what the state shows, keyed
    to this job's row and never to a tool name or a sample's wording.  Two ways:
    the whole line verbatim (whitespace-trimmed), and its ``reads <page> — "``
    prefix — the copied line with the instructions tail edited or cut.  Anything
    else passes: real instructions that merely mention the page are the normal
    case, and the render's page-less form (no ``reads `` clause) is matched by the
    verbatim comparison alone."""
    if row.extraction_prompt is None:
        return None
    summary = job_summary_line(row).strip()
    passed = prompt.strip()
    if passed == summary:
        return _SUMMARY_AS_PROMPT_REFUSAL.format(name=row.name)
    if summary.startswith("reads ") and passed.startswith(summary[: summary.index(' — "') + 4]):
        return _SUMMARY_AS_PROMPT_REFUSAL.format(name=row.name)
    return None


class CollectionSetTool(MemoryTool):
    """The ONE idempotent create-or-update entry point for a collection (the
    code-owner fusion): identity is dispatched HERE, in python — the model never
    reasons about whether the collection already exists.

    Three cases, in order.  The name EXISTS → the update path (adopt / edit the
    prompt, atomic schedule replace).  The name is new but names a job we
    ALREADY HAVE (#1567/#1775, ``find_duplicate_collection``) → the same update
    path, aimed at the collection we have, because ``collection_configure`` dispatching on
    the name alone is exactly what lets an invented name become a second copy of one
    job.  Otherwise → the create path.  Both inner paths keep their full validation —
    this class is a dispatcher, not a re-implementation."""

    name = "collection_configure"
    # The creation echo points at the metadata read for what was configured.
    advises = ("memory_get",)
    description = (
        "Create or reconfigure a collection in ONE idempotent call. If `collection` "
        "doesn't exist it comes into being with this config; if it does, only "
        "the fields you set change — you never need to know which. "
        "`description` = what it's for, in the user's words (required the first "
        "time). `prompt` = the instructions this job's cycles follow each time they "
        "run — write it as you would explain the task to someone doing it once: "
        "what to read, what to do with it, and what to remember. On a job that "
        "already exists, a schedule, expiry or notify change needs ONLY that field — "
        "leave `prompt` out and the job's instructions stay exactly as they are "
        "(collection_configure(collection=<the job's name>, "
        "schedule='FREQ=DAILY;BYHOUR=8') changes only when it runs). Pass `prompt` "
        "only when you mean to replace the instructions; the reply then shows the "
        "lines the old prompt had that the new one doesn't. "
        "`schedule` says "
        "when it runs, as one recurrence rule: "
        '"FREQ=HOURLY", "FREQ=DAILY;BYHOUR=8", '
        '"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9". `notify` turns THIS '
        "collection's notifications on or off — true tells the user about "
        "new/changed entries, false stops telling them while the job keeps "
        "running. It is the per-collection switch: turning it off silences this "
        "one collection and nothing else, and leaves the collection itself in "
        "place. Plain storage needs no call at all — collection_write creates "
        "storage automatically; use collection_configure for jobs and config."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
            "description": {"type": "string"},
            "prompt": {
                "type": "string",
                "description": PROMPT_PARAM_DESCRIPTION,
            },
            "schedule": {"type": "string", "description": SCHEDULE_PARAM_DESCRIPTION},
            "expires_at": {"type": "string", "description": EXPIRES_AT_PARAM_DESCRIPTION},
            "notify": {"type": "boolean"},
        },
        "required": ["collection"],
    }
    args_model = CollectionSetArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "collection", "a collection")
        if not result.success:
            return f"You tried to set up {name} but it didn't work:"
        return f"You set up {name}:"

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient,
        run_id: str | None = None,
    ) -> None:
        self._db = db
        self._llm_client = llm_client
        self._create = CollectionCreateTool(db, llm_client, created_by_run_id=run_id)
        self._update = CollectionUpdateTool(db, llm_client, run_id=run_id)

    async def _suggestion_embedding(self, name: str) -> list[float] | None:
        return await embed_text(self._llm_client, name)

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionSetArgs(**kwargs)
        return await self._dispatch(args)

    async def _dispatch(self, args: CollectionSetArgs) -> ToolResult:
        """Dispatch on IDENTITY, not on the name alone: the collection this call
        names, else the one it duplicates, else a new one."""
        if self._db.memories.get(args.collection) is not None:
            return await self._apply_to(args, args.collection)
        duplicate = self._db.memories.find_duplicate_collection(args.collection)
        if duplicate is not None:
            return await self._same_job(args, duplicate)
        if args.description is None:
            return ToolResult(
                message=_SET_BIRTH_NEEDS_DESCRIPTION.format(name=args.collection), success=False
            )
        return await self._create._run(
            name=args.collection,
            description=args.description,
            prompt=args.prompt,
            schedule=args.schedule,
            expires_at=args.expires_at,
            notify=bool(args.notify),
        )

    async def _apply_to(self, args: CollectionSetArgs, name: str) -> ToolResult:
        """Apply this call's fields to the EXISTING collection ``name`` — the update
        path, with the target's own name substituted for the one the call used."""
        return await self._update._run(
            name=name,
            description=args.description,
            notify=args.notify,
            prompt=args.prompt,
            schedule=args.schedule,
            expires_at=args.expires_at,
        )

    async def _same_job(self, args: CollectionSetArgs, duplicate: MemoryRow) -> ToolResult:
        """A new name for a job we already have (#1775).  An ARCHIVED match stays a
        refusal — reviving a retired mechanism is the user's call, and the tombstone
        confirm-shape asks for it (#1567).  An ACTIVE one is UPDATED in place rather
        than refused: the refusal cost an entire observed turn (the model retried, the
        second refusal was identical, and the all-failed abort ended the turn with no
        reply), while the update is what the user asked for on the collection that
        already serves it."""
        if duplicate.archived:
            return ToolResult(message=render_tombstone_duplicate(duplicate), success=False)
        result = await self._apply_to(args, duplicate.name)
        template = _SAME_JOB_UPDATED if result.success else _SAME_JOB_TARGETED
        note = template.format(requested=args.collection, existing=duplicate.name)
        return result.model_copy(update={"message": f"{note}{result.message}"})


_SET_BIRTH_NEEDS_DESCRIPTION = (
    "'{name}' doesn't exist yet, so this call would create it — but a new "
    "collection needs a `description` (what it's for, in the user's words; it's "
    "how the collection is found by meaning later). Call collection_configure again "
    "with description=<what it holds>."
)

# The same-job redirect (#1775): the requested name was a second name for a job that
# already has a collection, so the call landed on THAT collection.  Named in the
# result because everything downstream — reads, further edits, what Penny tells the
# user — has to use the real name, not the one the call invented.
_SAME_JOB_UPDATED = (
    "'{requested}' is the same job as '{existing}', which already exists — so I "
    "updated '{existing}' instead of creating a second collection for it. It's "
    "'{existing}' from here on: read it with collection_list('{existing}') or "
    "adjust it with collection_configure(collection='{existing}', ...). If this really is a "
    "separate task, set it up again with a name that carries what makes it "
    "different.\n"
)

# The same redirect when the update itself failed — the target is still named (the
# error below is about '{existing}', not the requested name), but nothing changed.
_SAME_JOB_TARGETED = (
    "'{requested}' is the same job as the existing collection '{existing}', so this "
    "was aimed at '{existing}' rather than creating a second collection — but it "
    "didn't land:\n"
)


class CollectionUpdateTool(MemoryTool):
    """Update collection metadata, including a full replacement of its prompt
    (#2455 naive baseline; was #1620's skill re-render).

    Chat-facing.  Lets the user evolve a collection mid-conversation — refining
    the description/cadence, flipping notify, or replacing the ``extraction_prompt``
    wholesale with new free text (an edit, a swap to a different job, or adopting a
    prompt onto an inert collection for the first time).  All fields except ``name``
    are optional; only the ones supplied are applied.  Supplying ``prompt`` replaces
    the stored prompt verbatim (and records a mutation event); omitting it leaves
    the prompt untouched.
    """

    name = "collection_configure"
    # The creation / re-render echo points at the metadata read for what was configured.
    advises = ("memory_get",)
    description = (
        "Update an existing collection's metadata. Only supplied fields "
        "are changed.\n"
        "\n"
        "Fields:\n"
        "- `collection` (required) — the collection to update.\n"
        "- `description` — content-reflective one-line summary AND the "
        "meaning anchor (find / resolve-by-meaning). Changing it "
        "re-embeds it, so keep it an accurate summary of the subject "
        "matter. It does not drive the collector — change `prompt` for that.\n"
        "- `notify` — flip notify-on-new. `true` starts telling the "
        "user about new entries (they asked to be kept posted / alerted); "
        "`false` silences it (the collector keeps gathering). Omit to leave "
        "unchanged.\n"
        "- `prompt` — FULL replacement of the job's instructions, not a diff. "
        "Drives what the collector actually does each cycle. Read the current "
        "body via `memory_get(<collection>)` first if you need to preserve any "
        "of it. On an inert (prompt-less) collection, setting this adopts a job "
        "for the first time.\n"
        "- `schedule` — when the job runs, as ONE recurrence rule (set it to change the "
        'schedule, else omit to leave it): "FREQ=HOURLY" (every hour), '
        '"FREQ=MINUTELY;INTERVAL=90" (every 90 minutes), "FREQ=DAILY;BYHOUR=8" (8am '
        'daily), "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9" (weekdays at 9). BYHOUR is '
        "the hour on the user's own clock — write the hour they said. "
        "Add COUNT= to run a set number of times, and a leading DTSTART line to say "
        "when it starts; a DTSTART stamp ending in Z is an exact moment in UTC. "
        "Setting it REPLACES the whole schedule. This is how you give an "
        "INERT collection its schedule when you adopt a prompt onto it.\n"
        f"- `expires_at` — OPTIONAL. {EXPIRES_AT_PARAM_DESCRIPTION} The collection "
        "archives itself once it passes.\n"
        "\n"
        "Returns a structured echo of the updated state. The echo is "
        "authoritative — if a field you tried to set isn't in it, the "
        'update didn\'t land; fix it and try again rather than saying "done".\n'
        "\n"
        "IMPORTANT: `prompt` is a FULL replacement — the whole "
        "prompt body, never a diff or a fragment; read the current body via "
        "`memory_get(<collection>)` first if you need to keep any of it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string", "description": "Collection name to update"},
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
            "prompt": {
                "type": "string",
                "description": (
                    "FULL replacement of the job's instructions, as free text. Omit to "
                    "leave the current prompt untouched."
                ),
            },
            "schedule": {
                "type": "string",
                "description": (
                    f"{SCHEDULE_PARAM_DESCRIPTION} Setting it REPLACES the whole schedule; "
                    "omit to leave it unchanged."
                ),
            },
            "expires_at": {
                "type": "string",
                "description": EXPIRES_AT_PARAM_DESCRIPTION,
            },
        },
        "required": ["collection"],
    }
    args_model = CollectionUpdateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "collection", "a collection")
        if not result.success:
            return f"You tried to update {name}'s settings but it didn't work:"
        return f"You updated {name}'s settings:"

    def __init__(self, db: Database, llm_client: LlmClient, run_id: str | None = None) -> None:
        self._db = db
        self._llm_client = llm_client
        # The chat run making this config change — recorded as the update
        # mutation's cause (#1560).
        self._run_id = run_id

    async def _suggestion_embedding(self, name: str) -> list[float] | None:
        # This tool embeds, so an update to a mistyped collection gets the MEANING
        # leg too (#1674).
        return await embed_text(self._llm_client, name)

    async def _run(self, **kwargs: Any) -> ToolResult:
        """Parse the apply-time schedule (#1857), then either replace the prompt
        (when ``prompt`` is given) or a plain metadata edit.  Reads as a table of
        contents."""
        args = CollectionUpdateArgs(**kwargs)
        parsed = self._parse_schedule(args)
        if isinstance(parsed, ToolResult):
            return parsed
        schedule, expires_at = parsed
        if args.prompt is not None:
            return await self._replace_prompt(args, schedule, expires_at)
        return await self._edit_metadata(args, schedule, expires_at)

    def _parse_schedule(
        self, args: CollectionUpdateArgs
    ) -> ToolResult | tuple[Schedule | None, datetime | None]:
        """Parse the optional ``schedule`` arg + end condition at apply time (#1857, the
        same one-arg RRULE ``collection_configure`` uses).  Returns ``(None, expires)`` when no
        schedule is set (the stored one is untouched) or the parsed
        ``(Schedule, expires)``; an unreadable schedule surfaces ``parse_schedule``'s
        teaching rejection verbatim."""
        try:
            schedule = parse_schedule(args.schedule) if args.schedule is not None else None
            expires_at = self._expiry(schedule, args.expires_at)
        except ScheduleError as exc:
            return ToolResult(message=str(exc), success=False)
        return schedule, expires_at

    def _expiry(self, schedule: Schedule | None, expires_at: str | None) -> datetime | None:
        """The end condition for this edit — the schedule's own ``UNTIL=`` when a new
        schedule was passed, else the ``expires_at`` argument on its own.  With no new
        schedule there is no ``UNTIL=`` to conflict with, so a bare ``expires_at`` edit
        reads as it always did."""
        if schedule is not None:
            return resolve_expiry(self._db, schedule, expires_at)
        if expires_at is None:
            return None
        return parse_expires_at(expires_at, user_timezone_name(self._db), self._db.clock.now())

    async def _edit_metadata(
        self, args: CollectionUpdateArgs, schedule: Schedule | None, expires_at: datetime | None
    ) -> ToolResult:
        """The plain metadata edit — description / notify / schedule, with no
        ``prompt`` supplied, so the collection's ``extraction_prompt`` is untouched."""
        embedding = await self._description_embedding(args)
        memory = self._apply_update(args, None, embedding, schedule, expires_at)
        suffix = _description_degraded_suffix(args.description, embedding)
        expiry_note = sentinel_expiry_note(args.expires_at, expires_at)
        echo = _format_collection_echo(
            memory, "Updated", user_timezone_name(self._db), self._priors(args.name)
        )
        message = f"{echo}{suffix}{expiry_note}"
        return ToolResult(message=message, mutated=True)

    def _priors(self, name: str) -> Mapping[str, str | None]:
        """This call's recorded before-values (#1946) — ``{}`` with no run id to key
        the mutation event by, same as every untracked edit today."""
        if self._run_id is None:
            return {}
        return self._db.mutations.priors_for_run(
            name, self._run_id, entity_type=MutationEntityType.COLLECTION
        )

    async def _replace_prompt(
        self, args: CollectionUpdateArgs, schedule: Schedule | None, expires_at: datetime | None
    ) -> ToolResult:
        """Replace the collection's ``extraction_prompt`` wholesale with ``args.prompt``
        (#2455 naive baseline) — an edit to a running job, a swap to a different job, or
        adopting a prompt onto a legacy inert (prompt-less) collection for the first
        time.  The prompt the model wrote IS the prompt the collector runs; there is no
        rendering seam to validate against, only the same length check authoring takes —
        plus the summary-line refusal (#2685): a ``prompt`` that is this job's own
        current-state summary line would replace the instructions with a rendering of
        them, so it's refused and she retries."""
        assert args.prompt is not None
        row = self._db.memories.get(args.name)
        prompt_before = row.extraction_prompt if row is not None else None
        if row is not None:
            refusal = _summary_as_prompt_refusal(row, args.prompt)
            if refusal is not None:
                return ToolResult(message=refusal, success=False)
        if (too_short := check_extraction_prompt(args.prompt)) is not None:
            return ToolResult(message=too_short, success=False)
        embedding = await self._description_embedding(args)
        memory = self._apply_update(args, args.prompt, embedding, schedule, expires_at)
        suffix = _description_degraded_suffix(args.description, embedding)
        echo = render_reinstantiation_echo(memory, user_timezone_name(self._db))
        # A replacement of an EXISTING prompt carries the line diff (#2719): what the old
        # prompt had that the new one doesn't, verbatim, so a rebuild-from-the-summary sees
        # the lines it dropped.  Reported, never blocked — the model may rewrite a job.
        diff = render_prompt_replacement_diff(prompt_before, args.prompt)
        diff_block = f"\n{diff}" if diff is not None else ""
        expiry_note = sentinel_expiry_note(args.expires_at, expires_at)
        return ToolResult(
            message=f"{echo}{suffix}{diff_block}{self._no_schedule_note(memory)}{expiry_note}",
            mutated=True,
        )

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
        description_embedding: list[float] | None,
        schedule: Schedule | None,
        expires_at: datetime | None,
    ) -> MemoryRow:
        """Thread the update through the store — the metadata fields, the prompt /
        anchor, and the apply-time schedule (#1857).  A
        ``schedule`` REPLACES the stored one along with its run quota
        (``replace_schedule``); ``expires_at`` sets the end condition, and a call that
        stated an end condition meaning "forever" REMOVES it (#1944) — an invented expiry
        must be able to come off a collection that already carries one, which is how the
        sentinel became sticky in the first place.  Records the mutation event with the
        run id + changed fields (schedule / expires_at)."""
        return self._db.memories.update_collection_metadata(
            args.name,
            description=args.description,
            extraction_prompt=extraction_prompt,
            description_embedding=description_embedding,
            notify=args.notify,
            schedule=schedule.rule if schedule else None,
            max_runs=schedule.max_runs if schedule else None,
            expires_at=expires_at,
            clear_expiry=expiry_was_normalised(args.expires_at, expires_at),
            replace_schedule=schedule is not None,
            run_id=self._run_id,
        )

    @staticmethod
    def _no_schedule_note(memory: MemoryRow) -> str:
        """A visible-degradation note (#1629) when an adopt gave a collection a routine
        but no schedule — it has an ``extraction_prompt`` yet nothing to run it on, so it
        won't dispatch until one is set.  Named, not silent."""
        if memory.extraction_prompt is None:
            return ""
        if has_schedule(memory):
            return ""
        return _NO_SCHEDULE_NOTE.format(name=memory.name)


class MemoryMetadataTool(MemoryTool):
    """Return the metadata fields for a single memory (collection or log).

    Genuinely shape-agnostic — metadata describes the memory itself, not its
    contents — so it's named ``memory_get`` and applies to either shape.
    """

    name = "memory_get"
    description = (
        "Return metadata for a memory: description, notify flag, schedule, last "
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
        # this exact-name lookup answers the full lifecycle question — who asked for
        # it, what run created it, whether it's live, and when it ends (#1566).
        lifecycle = _lifecycle_block(self._db, memory)
        # Lead with what the collection is FOR (description) and what it DOES (the
        # prompt), because that is the substance; the operational settings
        # (cadence/timestamps) are secondary and go last.  Ordered this way — and with
        # the nudge below — so a model asked "what does this do?" walks through the
        # prompt instead of reciting the cadence/notify trivia (the failure the
        # #1530 legibility baseline surfaced).
        lines = [
            f"name: {memory.name}",
            f"type: {memory.type}",
            f"description: {memory.description}",
            "",
            "What it does each cycle — the prompt below is the collection's actual "
            "behaviour.  When explaining the collection, read THIS, not the "
            "operational settings.",
            f"prompt: {memory.extraction_prompt or 'none'}",
            "",
            "Operational settings (cadence — secondary):",
            f"notify: {memory.notify}",
            self._schedule_line(memory, user_timezone_name(self._db)),
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
    def _schedule_line(memory: Any, timezone_name: str | None) -> str:
        """The schedule in plain words with the exact rule in a code span (#2425), or
        ``schedule: none`` for a log / an inert collection with no cadence yet — so the
        render never emits a half-formed clause (#1666, via the shared
        ``describe_schedule_field``)."""
        return f"schedule: {describe_schedule_field(memory, timezone_name)}"


class CollectionRemoveTool(MemoryTool):
    """Remove an entry from a collection by key."""

    name = "collection_remove"
    # A key that does not exist sends the model to the list of keys that do.
    advises = ("collection_keys",)
    description = (
        "Remove the entry with the given key from a collection. Returns the "
        "number of entries removed (zero if the key did not exist)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {"type": "string"},
            "key": {"type": "string"},
        },
        "required": ["collection", "key"],
    }
    args_model = CollectionRemoveArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collection = _collection_name(arguments, "a collection")
        entry = _entry_label(arguments)
        if not result.success:
            return f"You tried to remove {entry} from {collection} but it didn't work:"
        if not result.mutated:
            return f"You couldn't find {entry} to remove from {collection}:"
        return f"You removed {entry} from {collection}:"

    def __init__(self, db: Database, scope: str | None = None) -> None:
        self._db = db
        self._scope = scope

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionRemoveArgs(**kwargs)
        if (guard := _rules_guard(self._scope, args.collection)) is not None:
            return guard
        if self._scope is not None and args.collection != self._scope:
            return ToolResult(
                message=_SCOPE_REFUSAL_MESSAGE.format(
                    scope=self._scope, collection=args.collection
                ),
                success=False,
            )
        collection = _resolve(self._db, args.collection)
        removed = collection.delete(args.key)
        if removed == 0:
            rejection = _bracket_key_rejection(collection, args.collection, args.key)
            if rejection is not None:
                return rejection
            # Lead with the nearest existing key (typo leg) when one is close (#1674),
            # else the message is byte-identical to before.
            did_you_mean = _key_did_you_mean(collection, args.key)
            if did_you_mean:
                lead = (
                    f"No entry with key '{args.key}' in '{args.collection}'{did_you_mean} "
                    f"Nothing to delete."
                )
            else:
                lead = f"No entry with key '{args.key}' in '{args.collection}' — nothing to delete."
            return ToolResult(
                message=f"{lead} "
                f"List the current keys with collection_keys('{args.collection}') to find it."
            )
        return ToolResult(message=f"Deleted '{args.key}' from '{args.collection}'.", mutated=True)


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
        "properties": {"log": {"type": "string"}},
        "required": ["log"],
    }
    args_model = ReadLogArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        log = _named(arguments, "log", "a log")
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
        memory = _resolve(self._db, args.log)
        entries = self._read_cursor(memory) if self._cursor_mode else self._read_window(memory)
        return ToolResult(
            message=format_entries(
                entries,
                source=args.log,
                ordering="oldest first",
                show_ids=self._shows_ids(memory),
            )
        )

    @staticmethod
    def _shows_ids(memory: Memory) -> bool:
        """``show_ids`` only for a REAL stored log — the entries' ids are then
        consumable with ``log_get`` (``Memory.entry_by_id``).  The message facades
        and the run log render other backings entirely, so an id beside their
        lines would be an anchor nothing reads back; their renders stay
        byte-identical."""
        return isinstance(memory, Log) and not isinstance(memory, (MessageLogMemory, RunLog))

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


class LogGetTool(MemoryTool):
    """Read one entry of a log in full, by the id shown beside it.

    A log read renders each entry's id beside its line (``format_entries`` with
    ``show_ids``), and that id is the anchor this tool consumes: it reads the
    stored entry whole through ``Memory.entry_by_id``, scoped to the log the
    call names.  The id-taking read has its own refusals for the stores that
    aren't id-addressable logs — a collection is keyed, the two message logs
    and the run log are read by their own verbs.
    """

    name = "log_get"
    advises = ("log_read",)
    description = (
        "Return one entry of a log in full, by the id shown beside that entry. "
        "Pass the log's name and the id from a log read."
    )
    parameters = {
        "type": "object",
        "properties": {
            "log": {"type": "string"},
            "id": {"type": "integer"},
        },
        "required": ["log", "id"],
    }
    args_model = LogGetArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        log = _named(arguments, "log", "a log")
        entry = arguments.get("id")
        entry_label = f"entry {entry}" if entry else "an entry"
        if not result.success:
            return f"You tried to read {entry_label} of {log} but it didn't work:"
        return f"You read {entry_label} of {log}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = LogGetArgs(**kwargs)
        memory = _resolve(self._db, args.log)
        if isinstance(memory, Collection):
            return ToolResult(message=self._collection_refusal(args.log), success=False)
        if isinstance(memory, MessageLogMemory):
            return ToolResult(message=self._message_log_refusal(args.log), success=False)
        if isinstance(memory, RunLog):
            return ToolResult(message=self._run_log_refusal(args.log), success=False)
        entry = memory.entry_by_id(args.id)
        if entry is None:
            return ToolResult(
                message=(
                    f"No entry {args.id} in '{args.log}'. Read it with "
                    f"log_read('{args.log}') to see its entries and their ids."
                ),
                success=False,
            )
        return ToolResult(message=format_entries([entry], source=args.log))

    @staticmethod
    def _collection_refusal(name: str) -> str:
        return f"'{name}' is a collection, not a log: its entries are read by key, not by id."

    @staticmethod
    def _message_log_refusal(name: str) -> str:
        return (
            f"'{name}' holds messages, and a message is not read by id. "
            f"Read them with log_read('{name}')."
        )

    @staticmethod
    def _run_log_refusal(name: str) -> str:
        return (
            f"'{name}' holds run records, and a run is read by the `run <id>` shown beside "
            f"it, not by an entry id. Read them with log_read('{name}')."
        )


class ReadRunCallsTool(CursorReadTool):
    """Read a source's recent runs as their tool-call SEQUENCES.

    A sibling of ``log_read`` — same cursored machinery (the next bounded batch
    since this reader's committed cursor, oldest-first, pending until the cycle
    succeeds) — but a different lens: each run renders as its ``origin → the tool
    calls → conclusion`` (``render_run_calls``), the sequence-lens view of what a run
    *did*.  Orthogonal to the source: ``"chat"`` renders conversational runs
    (``user: <message>`` → tools → ``penny: <reply>``); a collector's name renders
    that collector's runs (``[target]`` → tools → ``done: <outcome>``).  Lets a
    reader see the tool sequence a request drove — for authoring skills, or for
    inspecting what a collector actually did.  The runs come from ``promptlog``; the
    cursor is per-source.
    """

    name = "run_list"
    args_model = ReadRunCallsArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        source = _named(arguments, "source", "a source")
        if not result.success:
            return f"You tried to review {source}'s recent runs but it didn't work:"
        return f"You reviewed {source}'s recent runs:"

    def __init__(self, db: Database, agent_name: str) -> None:
        super().__init__(db, agent_name)
        # Discover the valid sources from the DB so the model sees the current set —
        # 'chat' plus every collector (a collection with an extraction_prompt).
        sources = self._available_targets()
        listed = ", ".join(sources)
        self.description = (
            "Return a source's recent runs as tool-call sequences — each run shows its "
            "origin (the user's message, or the collector's bound target), the tool "
            "calls made, and the outcome. Returns the next batch since you last read; "
            f"call again for older runs. source is one of: {listed} "
            '("chat" for conversations, a collector name for that collector\'s runs).'
        )
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": sources, "description": f"one of {listed}"}
            },
            "required": ["source"],
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
        # Resolve the source against the valid set FIRST — 'chat' plus every live
        # collector, exactly what the description enumerates — so a typo'd/unknown
        # source gets the actionable memory-not-found refusal instead of a silent
        # empty batch that reads as "this collector has no runs".
        if args.source not in self._available_targets():
            raise MemoryNotFoundError(args.source)
        key = self._cursor_key(args.source)
        cursor = self._db.cursors.get(self._agent_name, key)
        groups = self._db.messages.run_call_groups(
            args.source, cursor, PennyConstants.RUN_CALLS_LIMIT
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
            message=format_entries(entries, source=args.source, ordering="oldest first")
        )


# An id with no recognised type tag: name what IS addressable + the guess-free
# fallbacks (find for names, the activity block for events), never a silent
# empty (the anchor discipline — a rendered id resolves in one call, a malformed
# one recovers).
_GET_EVENT_UNTYPED = (
    "'{run}' isn't a typed run id.  Pass a run token exactly as your "
    "current-state activity block renders it — run_get(run='run <id>'), "
    "copying the whole `run <id>` token.  (Runs are the addressable events right "
    "now; to resolve a collection or log by meaning use find(query=<text>), "
    "and your activity block names every recent event.)"
)

# A well-formed run token that matched no recorded run — say so and point at the
# lists that carry valid ids, rather than an empty read that reads as "clean run".
_GET_EVENT_NO_RUN = (
    "No run found with id '{run_id}'.  It must be an id your current-state activity "
    "block rendered (a `run <id>` line, or a change's `(run <id>)` cause); "
    "run_list(source='chat') — or a collector's name — lists recent run ids "
    "if you need to find a valid one."
)


class GetEventTool(Tool):
    """Resolve ONE ledger run by the typed id the activity block renders (#1580).

    The self-state activity block renders each background run as ``run <id>`` (and
    each config change names its ``(run <id>)`` cause) — typed anchors meant to be
    consumed VERBATIM.  ``run_get`` is the verb that consumes them: it parses the
    type tag and returns that run's detail.  Today the one addressable event kind
    is a run, so ``run_get(run='run <id>')`` returns that run's tool-call
    SEQUENCE — the same canonical projection ``run_list`` renders, but for the
    single run the id names (a point lookup, where ``run_list`` browses a
    source's run history).  An id with no recognised tag, or a run nothing recorded,
    gets an actionable refusal naming what IS addressable and how to find a valid id
    (a rendered id resolves in one call; a bad one recovers, never silently empties).

    Deliberately run-only (the smaller conforming shape, #1580): the send case is a
    *render* — an autonomous send's provenance appears inline wherever the message
    renders (``penny-messages`` reads, the header's sent lines), so it's a read, not
    an event lookup (#1568/#1608); and a mutation carries its detail inline in
    ``memory_get``'s change history and names its causing ``run <id>`` here, so
    no ``mut``-tagged arm is built until a surface actually renders that anchor (a
    verb needs a rendered id to consume — machinery follows a customer, not before).
    """

    name = "run_get"
    # A malformed or unmatched id recovers through the lists that carry valid ones.
    advises = ("run_list", "find")
    description = (
        "Look up ONE run from your activity log by the typed id it rendered — a "
        "`run <id>` line, or a change's `(run <id>)` cause.  Pass the whole token, "
        "run_get(run='run <id>'), copying the id exactly as rendered, and "
        "get that run's tool-call sequence: what it did, step by step.  Use it to "
        "inspect a specific run you saw in your current state; to browse a source's "
        "run history instead use run_list(source=<'chat' or a collector name>)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "run": {
                "type": "string",
                "description": (
                    "The typed run id, verbatim as your activity block rendered "
                    "it — the whole `run <id>` token."
                ),
            }
        },
        "required": ["run"],
    }
    args_model = GetEventArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        run = _named(arguments, "run", "a run")
        if not result.success:
            return f"You tried to look up {run} but it didn't work:"
        return f"You looked up {run}:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        # Plain ``Tool``, not ``MemoryTool``: this reads the ledger by id, resolves
        # no memory, and raises no MemoryAccessError — both misses are enumerated
        # refusals below, so the base's except template would be inert here.
        args = GetEventArgs(**kwargs)
        run_id = self._parse_run_id(args.run)
        if run_id is None:
            return ToolResult(message=_GET_EVENT_UNTYPED.format(run=args.run), success=False)
        prompts = self._db.messages.get_run_prompts(run_id)
        if not prompts:
            return ToolResult(message=_GET_EVENT_NO_RUN.format(run_id=run_id), success=False)
        # The model calls made ON BEHALF of this run (#2355 part D) — a browse-extract
        # draw, say — nested under the step that made them, so a run whose tool
        # succeeded but whose own extraction didn't finish shows BOTH facts here.
        child_draws = self._db.messages.get_child_draws(run_id)
        return ToolResult(message=render_run_calls(prompts, child_draws))

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
            "log": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["log", "content"],
    }
    args_model = LogAppendArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        log = _named(arguments, "log", "a log")
        if not result.success:
            return f"You tried to add an entry to {log} but it didn't work:"
        return f"You added an entry to {log}:"

    def __init__(
        self, db: Database, llm_client: LlmClient, author: str, run_id: str | None = None
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author
        # The appending run — stamped on the new log entry (#1560).
        self._run_id = run_id

    async def _suggestion_embedding(self, name: str) -> list[float] | None:
        # This tool embeds, so an append to a mistyped log gets the MEANING leg too (#1674).
        return await embed_text(self._llm, name)

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = LogAppendArgs(**kwargs)
        memory = _resolve(self._db, args.log)
        vec = await embed_text(self._llm, args.content)
        if vec is None:
            logger.warning("log_append: embedding unavailable for an entry in %s", memory.name)
            return ToolResult(message=_EMBED_WRITE_FAILURE_LOG.format(log=args.log), success=False)
        memory.append(
            [LogEntryInput(content=args.content, content_embedding=vec)],
            author=self._author,
            run_id=self._run_id,
        )
        return ToolResult(message=f"Appended to '{args.log}'.", mutated=True)


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
    "needs it. Retry find(query=<what the thing is about>) in a moment, or check "
    "the list in your current-state header."
)

# Zero matches is an honest empty, not an error — nothing of yours matched, and the
# state at the top of every turn already lists every collection and log (archived
# ones by name) so the model never dead-ends on a miss.
_FIND_EMPTY = (
    'Nothing of yours matched "{query}" — the state at the top of this prompt '
    "lists every collection and log you have (archived ones by name)."
)

# The same empty read, worded for the case where the conversation logs DID match
# something (#2289) — "nothing of yours matched" would contradict the messages
# listed right below it, so this leads with "nothing stored matched" instead; the
# rest of the guidance (the state header's list) still applies, since it's still
# true that nothing in the STORE matched.
_FIND_EMPTY_STORE = (
    'Nothing stored matched "{query}" — the state at the top of this prompt '
    "lists every collection and log you have (archived ones by name)."
)

# The conversation logs' own share of find's result (#2289), listed AFTER the store
# hits and under this heading — never interleaved with them.  A message surfaces
# only when it would have ranked in the fused top FIND_MATCH_LIMIT alongside find's
# own candidates (``MemoryStore.resolve_messages``), so a long conversation can
# crowd out other MESSAGES at worst, never the store hits above it.
_FIND_MESSAGES_HEADING = "From the conversation:"

# Ambiguity is RETURNED, never silently resolved: several hits come back ranked
# with how to narrow.
_FIND_AMBIGUOUS_TAIL = (
    "Ranked by closeness — if one is what you meant, use its addressing above; "
    "otherwise narrow by its exact name."
)


def _find_state(match: ResolvedMatch) -> str:
    """The live/archived state word: a collection or log is ``active`` or
    ``archived``."""
    return "archived" if match.archived else "active"


def _find_type_label(match: ResolvedMatch) -> str:
    """The family noun a hit renders."""
    return match.kind.value  # "collection" / "log"


def _find_addressing(match: ResolvedMatch) -> str:
    """The deterministic type→addressing map: the object's family (plus archived
    state) fixes the exact tool call that operates on it, so following the stated
    addressing succeeds with no further guessing (the anchor discipline).  The
    model never derives this mapping — it copies the call verbatim."""
    name = match.name
    if match.kind == ResolvedKind.LOG:
        return f"read it with log_read('{name}')"
    if match.archived:
        return (
            f"restore it with collection_unarchive('{name}'); its entries stay readable "
            f"with collection_list('{name}')"
        )
    return (
        f"read it with collection_list('{name}'), reconfigure it with "
        f"collection_configure(collection='{name}', ...), archive it with "
        f"collection_archive('{name}')"
    )


def _entry_preview(content: str) -> str:
    """One compact, whitespace-collapsed line of an entry's stored value (#1640) —
    the find IS the answer, so the whole value renders; no cap (#2642)."""
    return " ".join(content.split())


def _entry_identity(entry: ResolvedEntry) -> str:
    """The entry's head identity: a keyed collection entry shows its invocation-form
    key in its collection; a keyless log entry shows its ``#<id>`` id in its log —
    the id the addressing line below reads back with ``log_get``."""
    if entry.key is None:
        return f"entry #{entry.entry_id} in `{entry.memory_name}` ({entry.container_kind.value})"
    return f"entry {render_key(entry.key)} in `{entry.memory_name}`"


def _entry_addressing(entry: ResolvedEntry) -> str:
    """How to read the stored entry back — deterministic from whether it is keyed: a
    keyed collection entry is one ``collection_get`` away (re-reads the exact value
    the find already carried); a keyless log entry is one ``log_get`` away, by the
    id the find itself shows in its identity line (#1640)."""
    if entry.key is None:
        return f"log_get(log='{entry.memory_name}', id={entry.entry_id})"
    return f"collection_get(collection='{entry.memory_name}', key='{entry.key}')"


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
    return "\n".join([head, f"   how to use it: {_find_addressing(match)}"])


def _render_find_hit(index: int, hit: ResolvedHit) -> str:
    """One fused hit, best-first ranked — dispatched by family: a stored entry
    renders its value + how to read it, an object its state/type + how to use it."""
    if isinstance(hit, ResolvedEntry):
        return _render_entry_hit(index, hit)
    return _render_object_hit(index, hit)


def _render_message_hit(index: int, message: ResolvedMessage) -> str:
    """One message hit (#2289) — the message itself, read back exactly as
    ``read_similar`` used to show it on that log: who said it (``memory_name`` names
    which log it's from — ``user-messages`` or ``penny-messages``), when
    (``format_log_timestamp``, the same form ``log_read`` renders), and its full
    content, with the ``(sent by <mechanism>)`` clause already in it where the row
    carries one (#1568).  No preview cap and no addressing line — the message IS
    the answer."""
    stamp = format_log_timestamp(message.created_at)
    return f"{index}. {message.memory_name} [{stamp}]: {message.content}"


def _render_scoped_memory(memory: Memory, entries: list[MemoryEntry]) -> str:
    """``find(query, memory=<name>)`` (#2289) — one named memory's own entries,
    ranked by closeness to the query (``Memory.read_similar``), read back in
    full rather than the 120-char preview: naming the list IS the answer.  A
    collection or log renders through the SAME ``format_entries`` every read
    tool uses, so the empty/populated shapes match what she already reads
    elsewhere.  A message log (``user-messages`` / ``penny-messages``) renders
    through the SAME per-entry line find's own conversation section uses
    (``_render_message_hit``) instead — a plain ``format_entries`` line has no
    way to say who said it."""
    if not isinstance(memory, MessageLogMemory):
        return format_entries(entries, source=memory.name, ordering="closest to the query first")
    if not entries:
        return format_entries(entries, source=memory.name)
    body = "\n".join(
        _render_message_hit(
            i,
            ResolvedMessage(
                memory_name=memory.name, content=entry.content, created_at=entry.created_at
            ),
        )
        for i, entry in enumerate(entries, start=1)
    )
    noun = "entry" if len(entries) == 1 else "entries"
    return f"{len(entries)} {noun} from `{memory.name}` (closest to the query first):\n{body}"


class FindTool(MemoryTool):
    """Find anything of Penny's own by meaning — the guess-free fallback when the
    exact name/key isn't ambient (#1558, #1640, #2289).

    Embedding search (plain cosine, the #1565 explicit-search path) over every
    registry row's description anchor (collections + logs, ARCHIVED INCLUDED), AND
    every stored entry's content/key anchors across the
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

    ALSO searches what the user and Penny said in conversation (#2289): the two
    message logs are ranked in the same fused pass, but reported as a SEPARATE
    section after the store hits (``MemoryStore.resolve_messages``) rather than
    merged into the list above — a message is read back whole, with no addressing
    line, since the message itself is the answer.

    An optional ``memory`` (#2289) NARROWS the search to one memory she can
    already name, instead of the whole registry: ``Memory.read_similar`` on the
    object ``db.memory(name)`` resolves, returning just that memory's own nearest
    entries — works uniformly on a collection, a log, the two message logs, or an
    archived collection.  An unknown name gets the house did-you-mean refusal
    (``_suggestion_embedding`` below gives it the meaning leg, since this tool
    already carries an embedding client).
    """

    name = "find"
    # The ADDRESSING it renders (#1919).  A find hit's whole second line is "and here is
    # the exact call that operates on it", chosen deterministically from the hit's kind
    # and archived state (`_object_addressing` / `_entry_addressing`) — so every verb
    # those renders can name is a sibling this tool points the model at, and a
    # program-scoped surface carrying `find` has to carry them too or the line resolves
    # in nothing.  The gap was invisible until the coherence scanner learned to follow a
    # helper's own constants; declaring it is the fix that test's docstring prescribes.
    # Message hits carry no addressing line (#2289), so they add no verb here.
    advises = (
        "collection_list",
        "collection_get",
        "collection_configure",
        "collection_archive",
        "collection_unarchive",
        "log_read",
        "log_get",
    )
    description = (
        "Find anything of your own by meaning, when you don't know its exact name — "
        "a collection, a log, a single stored entry (a fact you "
        "saved), or something the user or you said earlier in the conversation.  Pass "
        "`query`, a paraphrase of what it's about (\"what I said that product "
        'costs").  Returns the closest matches, best first: a collection/log '
        "with its exact name, whether it's active or archived, and the exact tool call "
        "that operates on it (archived included); a stored entry with the value it "
        "holds and how to read it back; a matching message from the conversation, "
        "shown in full under its own heading.  Use it "
        "instead of guessing a name: a guessed name fails, this resolves the real one "
        "— and for a short fact or a past message, the returned value IS the answer. "
        "Pass `memory` too to search inside one store you can already name instead of "
        "everything — its own nearest entries come back, best first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A paraphrase of what the thing is about — its meaning, not its exact "
                    "name. Matches a collection, a log, a stored entry, or "
                    "a past message from either side of the conversation."
                ),
            },
            "memory": {
                "type": "string",
                "description": (
                    "Optional. Leave it out to search everything you have. Given the exact "
                    "name of one of your collections or logs, the search narrows to just "
                    "that one and returns its nearest entries by meaning."
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

    async def _suggestion_embedding(self, name: str) -> list[float] | None:
        # A scoped find (`memory=`, #2289) is a name-dispatch call too, so a miss
        # gets the MEANING leg exactly like the other embedding-capable name-dispatch
        # tools (#1674) — this tool already carries the embedding client.
        return await embed_text(self._llm, name)

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = FindArgs(**kwargs)
        vec = await embed_text(self._llm, args.query)
        if vec is None:
            logger.warning("find: query embedding failed transiently")
            return ToolResult(message=_FIND_EMBED_FAILURE, success=False)
        if args.memory is not None:
            return self._run_scoped(args.memory, vec)
        hits = self._db.memories.resolve_objects(vec, PennyConstants.FIND_MATCH_LIMIT)
        # The conversation logs' own share of the fused ranking (#2289) — computed
        # independently of ``hits`` above, so a long conversation can never shrink the
        # store's own result; see ``MemoryStore.resolve_messages``.
        messages = self._db.memories.resolve_messages(vec, PennyConstants.FIND_MATCH_LIMIT)
        return ToolResult(message=self._format(args.query, hits, messages))

    def _run_scoped(self, name: str, vec: list[float]) -> ToolResult:
        """``find(query, memory=<name>)`` (#2289) — one memory she can already name,
        searched by meaning alone (``Memory.read_similar``, the same ranking
        ``resolve_objects``/``resolve_messages`` use), never the whole registry.
        An unknown ``name`` raises ``MemoryNotFoundError``, which ``execute``
        catches and re-renders with the did-you-mean suggestion."""
        memory = _resolve(self._db, name)
        entries = memory.read_similar(vec, PennyConstants.FIND_MATCH_LIMIT)
        return ToolResult(message=_render_scoped_memory(memory, entries))

    def _format(self, query: str, hits: list[ResolvedHit], messages: list[ResolvedMessage]) -> str:
        if not hits and not messages:
            return _FIND_EMPTY.format(query=query)
        parts = [self._format_store(query, hits)]
        if messages:
            message_body = "\n".join(
                _render_message_hit(i, m) for i, m in enumerate(messages, start=1)
            )
            parts.append(f"{_FIND_MESSAGES_HEADING}\n{message_body}")
        return "\n\n".join(parts)

    def _format_store(self, query: str, hits: list[ResolvedHit]) -> str:
        """The store half of find's result.  Reached with an empty ``hits`` only
        when ``messages`` is non-empty (``_format`` returns the plain
        ``_FIND_EMPTY`` directly for the fully-empty case) — paired with a
        messages section below it, "nothing of yours matched" would contradict
        what follows, so the empty case here always reads "nothing STORED
        matched" instead (``_FIND_EMPTY_STORE``)."""
        if not hits:
            return _FIND_EMPTY_STORE.format(query=query)
        body = "\n".join(_render_find_hit(i, h) for i, h in enumerate(hits, start=1))
        if len(hits) == 1:
            return f'Found 1 thing matching "{query}":\n{body}'
        return (
            f'Found {len(hits)} things matching "{query}", best first:\n{body}\n'
            f"{_FIND_AMBIGUOUS_TAIL}"
        )


# ── Introspection / lifecycle ───────────────────────────────────────────────


class DoneTool(Tool):
    """Signal the cycle is finished, and say whether it has news (#2385).

    The run record is GENERATED from the run's canonical ledger rows (its tool calls +
    write-gate outcomes + structural counts), so there is no model-authored
    ``success``/``summary`` to confabulate — what is generated cannot lie.  The one
    argument the close carries is the notify CHOICE (#2385): a required ``notify``
    boolean the framework reads after the cycle to decide whether the composer runs.
    Saying true does not send anything by itself — Penny tells the user afterwards,
    from what the cycle wrote (#2656), and a job with notifications off or a muted
    user sends nothing either way.

    It came back because the alternative was worse.  Deriving the close from PROGRAM
    COVERAGE (#1911) fixed the path in advance, and a cycle that departed from it for
    a good reason — taking the store's own advice to ``update_entry`` where the program
    said ``collection_write`` — could then never close: no coverage to reach, and no
    tool to say so with.  Measured, those cycles died rerolling on an attempt to state
    in prose what they no longer had a call for.  A model that has finished can always
    make one more call; it cannot always walk a route decided before it started.  It
    was an argless sentinel at that return (#1569); the notify argument was added when
    the quiet cycle became a decision the model records rather than a STOP the gate
    issues (#2385)."""

    name = PennyConstants.DONE_TOOL_NAME
    description = (
        "Call this when the cycle is finished, with notify set to true or false. A "
        "cycle does not message the user itself: what the cycle writes down, plus "
        "this close with notify=true, is how they find out — Penny tells them "
        "afterwards, from what the cycle wrote. Say true when this cycle found or "
        "stored something the user would want to hear now; say false when nothing "
        "changed since what you last told them — a quiet cycle is a normal outcome, "
        "not a failure. A job with notifications off or a muted user sends nothing "
        "either way. The run record is generated from your tool calls, so there is "
        "nothing to summarise."
    )
    parameters = {
        "type": "object",
        "properties": {
            "notify": {
                "type": "boolean",
                "description": (
                    "true if the user should hear about this cycle; false if they "
                    "shouldn't. Compare what the collection holds and what you last "
                    "told them with what you found this cycle."
                ),
            },
        },
        "required": ["notify"],
    }
    args_model = DoneArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        return "You wrapped up the cycle:"

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        # The done call itself always succeeds and never mutates state; the run's
        # outcome is derived structurally from the ledger, not from this call.
        return ToolResult(message="Cycle complete.")


# ── On-demand collector trigger ─────────────────────────────────────────────


# ── Factory ─────────────────────────────────────────────────────────────────


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
      is what stops a newest-first ``collection_list`` from bypassing a
      log's cursor.
    * **Read-only facades.** ``log_append`` to ``user-messages`` /
      ``penny-messages`` / ``collector-runs`` is refused up front
      (``SYSTEM_LOGS``); the facades also raise ``ReadOnlyMemoryError`` if
      reached, since they're views over ``messagelog`` / ``promptlog``.
    * **Collector binding.** ``scope`` pins a collector's entry
      mutations to its bound collection ``X`` (the ``scope`` check in
      each entry-mutation tool).  Chat passes ``scope=None`` — its
      entry mutations are unrestricted, since edits are user-directed.

    ``memory_get`` is the genuinely shape-agnostic read — it works on either
    shape.  ``find`` (resolve-by-meaning, #1558/#1640/#2289) spans the whole
    registry — collections, logs, and stored entries — plus the two
    conversation logs, and returns each hit's exact identity fused with how to
    address it, the guess-free fallback that every not-found error points at.

    There is no ``done`` and no ``send_message`` on any surface any more (#1911).
    A chat turn ends by replying, a collector cycle ends with ``done()`` or a
    write-gate STOP, and an autonomous message is the framework's to write and queue
    after that — so neither loop control nor emission is a capability the model
    reaches for.

    ``run_id`` is the id of the run that built the surface — the chat turn's run,
    or the collector cycle's — passed as an explicit parameter, never ambient state
    (#1560).  It threads to every write- and mutation-capable tool as the executing
    run: entry writes stamp it as ``created_by_run_id`` / ``last_written_by_run_id``
    on the rows they add or rewrite; ``collection_configure`` records it as the new
    mechanism's ``created_by_run_id`` (#1566); and create / update / archive /
    unarchive record it on the durable mutation event they emit (the ledger's
    provenance closure).  ``None`` for a non-run caller, which leaves the columns
    NULL.  (The spawning ``source_message_id`` is linked afterward by the channel,
    since the message id isn't known until the run returns.)

    ``include_lifecycle`` gates the registry-shape tier — ``collection_configure`` /
    ``collection_configure`` / ``collection_archive`` /
    ``collection_unarchive`` / ``log_create``.  Chat-style agents get it (the user
    evolves collections through them); a cadence-fired collector run passes
    ``False`` (#1556), so those tools are structurally ABSENT from its surface — a
    background poll cannot create, reconfigure, or archive mechanisms, no
    matter what its extraction_prompt says.  The declaration lives on the agent
    (``Agent._include_lifecycle_tools``, overridden by ``Collector``), not as a
    branch here.

    ``find`` stays on every surface, lifecycle or not: the did-you-mean fallback on
    every other read/write tool here points at it (#1674), so an agent without it
    would have those tools dead-end into a call it cannot make.  The one coherence
    gap this opens — a `find` hit on a lifecycle object addresses it with a
    lifecycle verb (``collection_configure``/``collection_archive``/
    ``collection_unarchive``) a collector surface does not carry — is a known,
    narrow exception: unlike every other tool here, `find`'s OWN closure cannot be
    made to fit inside a collector cycle without either losing the did-you-mean
    target every other tool depends on, or inventing a surface-aware addressing
    `find` has never needed before.
    """
    reads: list[Tool] = [
        CollectionListTool(db),
        CollectionGetTool(db),
        CollectionSampleTool(db),
        CollectionKeysTool(db),
        MemoryMetadataTool(db),
        LogReadTool(db, agent_name, scope),
        LogGetTool(db),
        ReadRunCallsTool(db, agent_name),
        GetEventTool(db),
        FindTool(db, llm_client),
    ]
    lifecycle: list[Tool] = [
        CollectionSetTool(db, llm_client, run_id=run_id),
        CollectionArchiveTool(db, run_id=run_id),
        CollectionUnarchiveTool(db, run_id=run_id),
        LogCreateTool(db, llm_client),
    ]
    mutations: list[Tool] = [
        CollectionWriteTool(db, llm_client, agent_name, scope=scope, run_id=run_id),
        CollectionRemoveTool(db, scope=scope),
        LogAppendTool(db, llm_client, agent_name, run_id=run_id),
    ]
    return reads + (lifecycle if include_lifecycle else []) + mutations
