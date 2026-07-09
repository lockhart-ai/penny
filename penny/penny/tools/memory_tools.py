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

import logging
import random
from abc import abstractmethod
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import (
    DedupThresholds,
    EntryInput,
    Inclusion,
    LogEntryInput,
    Memory,
    MemoryAccessError,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    MemoryType,
    RecallMode,
    WriteResult,
    render_key,
    render_run_calls,
    strip_display_brackets,
)
from penny.database.models import MemoryEntry, MemoryRow
from penny.datetime_utils import format_log_timestamp
from penny.llm.similarity import embed_text
from penny.text_validity import check_extraction_prompt_tools
from penny.tools.base import Tool
from penny.tools.memory_args import (
    CatalogArgs,
    CollectionCreateArgs,
    CollectionDeleteEntryArgs,
    CollectionEntrySpec,
    CollectionGetArgs,
    CollectionMergeArgs,
    CollectionUpdateArgs,
    CollectionWriteArgs,
    CollectorRunHistoryArgs,
    DoneArgs,
    ExistsArgs,
    LogAppendArgs,
    LogCreateArgs,
    MemoryNameArgs,
    ReadLatestArgs,
    ReadLogArgs,
    ReadPublishedLatestArgs,
    ReadRandomArgs,
    ReadRunCallsArgs,
    ReadSimilarArgs,
    UpdateEntryArgs,
)
from penny.tools.models import ToolResult

if TYPE_CHECKING:
    from penny.agents.collector import Collector
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


_INCLUSION_MODES = ", ".join(m.value for m in Inclusion)
_RECALL_MODES = ", ".join(m.value for m in RecallMode)


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


def _humanize_interval(seconds: int | None) -> str:
    """Render a collector interval as a human-readable cadence (e.g. '1h')."""
    if not seconds:
        return "unset"
    if seconds % 86400 == 0:
        days = seconds // 86400
        return f"{days}d"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return f"{hours}h"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes}m"
    return f"{seconds}s"


def _format_collection_echo(memory: Any, verb: str) -> str:
    """Render a created or updated collection as a structured echo.

    The chat agent uses this to confirm-back what landed without
    making up fields.  Includes the full extraction_prompt verbatim
    so the model's reply can summarize accurately (the model previously
    confabulated this because the create/update returns were one-liners).
    """
    intent_line = f"  intent: {memory.intent}\n" if memory.intent else ""
    return (
        f"{verb} collection '{memory.name}':\n"
        f"  interval: {memory.collector_interval_seconds}s "
        f"({_humanize_interval(memory.collector_interval_seconds)})\n"
        f"  inclusion: {memory.inclusion}\n"
        f"  recall: {memory.recall}\n"
        f"  published: {memory.published}\n"
        f"{intent_line}"
        f"  description: {memory.description}\n"
        f"  extraction_prompt: |\n    "
        f"{(memory.extraction_prompt or '').replace(chr(10), chr(10) + '    ')}"
    )


# ── Description-anchor embed degradation (visible, self-healing) ─────────────
#
# A description doubles as the stage-1 routing anchor.  Unlike an entry write
# (which fails hard, #1412, because a vectorless entry is recall-invisible and
# corrupts dedup), a collection/log is still fully created or updated when its
# description embed fails transiently — only its ``relevant``-routing anchor is
# missing, and the startup description backfill re-embeds any ``NULL`` anchor.
# So the create/update succeeds, but the degradation is NAMED in the result
# rather than left silent (visible-degradation): the anchor is unset until it
# self-heals.  No retry is demanded — the row already exists and retrying the
# create would only collide.
_DESCRIPTION_EMBED_DEGRADED = (
    " (Heads up: couldn't embed its description just now — a transient embedding "
    "error — so its relevance-routing anchor is unset and it won't surface via "
    "'relevant' recall until it self-heals on the next restart.)"
)


def _description_degraded_suffix(description: str | None, embedding: list[float] | None) -> str:
    """A visible-degradation note when a description was supplied but its embed failed.

    Empty (no note) unless a description was given *and* its embedding came back
    ``None`` — i.e. the anchor was left ``NULL`` for the backfill to re-heal."""
    if description is not None and embedding is None:
        return _DESCRIPTION_EMBED_DEGRADED
    return ""


# ── Metadata ────────────────────────────────────────────────────────────────


class CollectionCreateTool(MemoryTool):
    """Create a new keyed collection.

    Description doubles as the chat-agent's guide to writing good
    extraction_prompts for new collections.  Dry-run-tuned against
    gpt-oss:20b to land the structural elements the per-collection
    Collector subagent needs (numbered tool calls, quiet-cycle escape,
    correction step, opt-in send_message for notify-on-new) consistently
    across both extract-and-notify and pure-extract user requests.
    """

    name = "collection_create"
    description = (
        "Create a keyed collection memory with a background collector.\n"
        "\n"
        "A collection is a long-lived task: every "
        "`collector_interval_seconds` the Collector subagent runs the "
        "`extraction_prompt` you supply here against the bound collection, "
        "browsing or reading logs, writing structured entries, and "
        "(optionally) calling `send_message(<message>)` to ping the user.\n"
        "\n"
        "Fields:\n"
        "- `name` — unique slug (lowercase, hyphens).\n"
        "- `description` — a content-reflective one-line summary of what "
        "this collection holds.  This IS the routing anchor: a "
        "relevant-inclusion collection is only surfaced when the conversation "
        'matches this text, so describe the actual subject matter ("heavy '
        'euro-style strategy board games"), not the mechanism ("a collection '
        'that stores games").\n'
        f"- `inclusion` ({_INCLUSION_MODES}) — stage-1 routing.  `always`: "
        "always in recall (identity, conventions).  `relevant`: in only when "
        "the conversation matches the description — the default for research "
        "collections.  `never`: silent — never surfaced in chat, only its "
        "collector runs in the background.\n"
        f"- `recall` ({_RECALL_MODES}) — stage-2 entry rendering once a "
        "collection is included.  `relevant` ranks entries against the "
        "conversation (default), `recent` shows newest, `all` shows every "
        "entry.\n"
        "- `extraction_prompt` — REQUIRED.  The system prompt the "
        "collector subagent runs each cycle.  Write it as a NUMBERED list of "
        "explicit steps and tool calls (1., 2., 3.) — never flowing prose: a "
        "numbered recipe is followed far more reliably, while a prose prompt "
        "makes the collector bail without doing the work.  The runtime "
        "appends invariants (quiet-cycle escape, batched writes, "
        "`send_message(<message>)` gating, structured "
        "`done(success=<true|false>, summary=<summary>)`); you "
        "supply only the workflow.\n"
        "- `collector_interval_seconds` — REQUIRED.  How often the "
        "collector runs.\n"
        "- `intent` — REQUIRED.  What the user asked for, in their own "
        "words — the goal the collection serves, captured once at creation "
        "and immutable thereafter.  Describe their request, not the "
        "mechanism.\n"
        "- `published` — Set `true` when the user wants to be **told "
        "about / kept posted on / alerted to** new entries as they're found — "
        "a notifier delivers each new entry to them once.  Leave it `false` "
        "(the default) for a silent collection that just gathers in the "
        "background for the user to ask about later.  Do NOT add a "
        "`send_message(<message>)` step to the extraction_prompt for this — the "
        "collector only gathers; `published=true` is how new finds reach the "
        "user.\n"
        "\n"
        "Returns a structured echo of the stored fields (name, interval, "
        "recall, published, description, full extraction_prompt).  Use the echo to "
        "confirm back to the user — don't invent fields it didn't return.\n"
        "\n"
        "For workflow guidance — when to call this vs "
        "`collection_update(name=<collection>)` or just "
        '`browse(queries=["<topic>"])`, how to shape the extraction_prompt for '
        "common intents (research+notify, digest, silent research, etc.) — see "
        "the skills surfaced in your recall context.\n"
        "\n"
        "IMPORTANT: the `extraction_prompt` MUST be a numbered list of "
        "tool-call steps (1., 2., 3.), never flowing prose — a prose prompt "
        "makes the collector bail without doing the work.\n"
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
                    "Content-reflective one-line summary of what this "
                    "collection holds — the stage-1 routing anchor. Describe "
                    "the subject matter, not the mechanism."
                ),
            },
            "inclusion": {
                "type": "string",
                "enum": [m.value for m in Inclusion],
                "description": (
                    "Stage-1 routing: 'always' (always in recall), 'relevant' "
                    "(in only when the conversation matches the description — "
                    "the usual choice), 'never' (silent background collector)."
                ),
            },
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": (
                    "Stage-2 entry rendering once included: 'relevant' ranks "
                    "entries to the conversation, 'recent' newest, 'all' every "
                    "entry."
                ),
            },
            "extraction_prompt": {
                "type": "string",
                "description": (
                    "REQUIRED. The system prompt the Collector subagent runs "
                    "each cycle. Numbered list of explicit tool calls — see "
                    "tool description for worked examples."
                ),
            },
            "collector_interval_seconds": {
                "type": "integer",
                "description": (
                    "REQUIRED. How often the collector runs. Common values: "
                    "1800 (30m), 3600 (1h, default for active research), "
                    "21600 (6h), 86400 (daily)."
                ),
            },
            "intent": {
                "type": "string",
                "description": (
                    "REQUIRED. What the user asked for, in their words — the "
                    "goal this collection serves. Capture their actual request "
                    '("a running list of good retro JRPGs to play"), not the '
                    "mechanism. This is the spec the collection is later judged "
                    "against and can't be changed after creation, so get it "
                    "right; confirm it back to the user."
                ),
            },
            "published": {
                "type": "boolean",
                "description": (
                    "true = notify the user about new entries (they asked to be "
                    "told / kept posted / alerted as new ones are found); false "
                    "(default) = silent background collection they'll ask about "
                    "later. Don't add a send_message(<message>) step to the prompt — "
                    "published=true is how new finds reach the user."
                ),
            },
        },
        "required": [
            "name",
            "description",
            "inclusion",
            "recall",
            "extraction_prompt",
            "collector_interval_seconds",
            "intent",
        ],
    }
    args_model = CollectionCreateArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        name = _named(arguments, "name", "a new collection")
        if not result.success:
            return f"You tried to set up the {name} collection but it didn't work:"
        return f"You set up the {name} collection:"

    def __init__(self, db: Database, llm_client: LlmClient) -> None:
        self._db = db
        self._llm_client = llm_client

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionCreateArgs(**kwargs)
        if rejection := _reject_unknown_extraction_tools(
            self._db, self._llm_client, args.extraction_prompt
        ):
            return rejection
        description_embedding = await embed_text(self._llm_client, args.description)
        memory = self._db.memories.create_collection(
            args.name,
            args.description,
            Inclusion(args.inclusion),
            RecallMode(args.recall),
            extraction_prompt=args.extraction_prompt,
            collector_interval_seconds=args.collector_interval_seconds,
            description_embedding=description_embedding,
            intent=args.intent,
            published=args.published,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        message = f"{_format_collection_echo(memory, 'Created')}{suffix}"
        return ToolResult(message=message, mutated=True)


class LogCreateTool(MemoryTool):
    """Create a new append-only log."""

    name = "log_create"
    description = (
        "Create a new append-only log. Logs store keyless entries in time order "
        "and are meant for streams of events (messages, measurements, etc.). "
        f"Provide a content-reflective description, an inclusion mode "
        f"({_INCLUSION_MODES}), and an entry-recall mode ({_RECALL_MODES})."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique log name"},
            "description": {
                "type": "string",
                "description": "Content-reflective one-line summary (stage-1 routing anchor)",
            },
            "inclusion": {
                "type": "string",
                "enum": [m.value for m in Inclusion],
                "description": (
                    "Stage-1 routing: 'always', 'relevant' (matches the "
                    "description), or 'never' (silent)."
                ),
            },
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": "Stage-2 entry rendering: 'relevant', 'recent', or 'all'.",
            },
        },
        "required": ["name", "description", "inclusion", "recall"],
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
            Inclusion(args.inclusion),
            RecallMode(args.recall),
            description_embedding=description_embedding,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        message = f"Created log '{args.name}'.{suffix}"
        return ToolResult(message=message, mutated=True)


class CollectionArchiveTool(MemoryTool):
    """Archive a collection — keeps data, removes it from ambient recall."""

    name = "collection_archive"
    description = (
        "Archive a collection. The data stays intact but the collection is "
        "excluded from the chat agent's ambient recall until unarchived."
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

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.archive(args.memory)
        return ToolResult(message=f"Archived '{args.memory}'.", mutated=True)


class CollectionUnarchiveTool(MemoryTool):
    """Restore a previously archived collection to ambient recall."""

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

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.unarchive(args.memory)
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


# ── Embed-failure at write time (fail-hard, #1412) ──────────────────────────
#
# Every stored entry MUST carry its similarity vector: an entry without one is
# invisible to recall (``read_similar`` skips it) and silently weakens dedup.  So
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
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author
        self._scope = scope

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
        results = memory.write(entries, author=self._author)
        return self._format_results(args.memory, results)

    def _embed_failure(self, memory: str, entries: list[EntryInput]) -> ToolResult | None:
        """Refuse the write, atomically, if any entry lost a vector to a transient
        embed failure — a vectorless entry is recall-invisible and dedup-weakening,
        so nothing is persisted and the model retries once embedding recovers
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
        written = [r.key for r in results if r.outcome == "written"]
        duplicates = [r for r in results if r.outcome == "duplicate"]
        rejected = [r for r in results if r.outcome == "rejected"]
        if duplicates:
            logger.info(
                "collection_write: %d duplicate(s) rejected in %s: %s",
                len(duplicates),
                memory,
                ", ".join(r.key for r in duplicates),
            )
        if rejected:
            logger.info(
                "collection_write: %d degenerate entry(ies) rejected in %s: %s",
                len(rejected),
                memory,
                ", ".join(f"{r.key!r} ({r.reason})" for r in rejected),
            )
        parts: list[str] = []
        if written:
            noun = "entry" if len(written) == 1 else "entries"
            parts.append(f"Wrote {len(written)} {noun} to '{memory}': {', '.join(written)}.")
        if duplicates:
            labelled = [_format_duplicate(r) for r in duplicates]
            close = self._duplicate_close(all_duplicates=not written and not rejected)
            parts.append(f"Rejected as duplicates: {'; '.join(labelled)}.  {close}")
        if rejected:
            labelled = [f"{r.key} ({r.reason})" for r in rejected]
            parts.append(
                f"Rejected as degenerate content: {', '.join(labelled)}.  "
                f"Re-write these with substantive descriptive text (not a bare URL, "
                f"punctuation, or a bail-out phrase)."
            )
        message = " ".join(parts) if parts else "(no entries written)"
        # Work only if a row actually landed — a fully duplicate/rejected batch
        # changed nothing, so it must read as no-work for the throttle.
        return ToolResult(message=message, mutated=bool(written))

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

    def __init__(self, db: Database, author: str, scope: str | None = None) -> None:
        self._db = db
        self._author = author
        self._scope = scope

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = UpdateEntryArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return ToolResult(
                message=_SCOPE_REFUSAL_MESSAGE.format(scope=self._scope, memory=args.memory),
                success=False,
            )
        memory = _resolve(self._db, args.memory)
        outcome = memory.update(args.key, args.content, self._author)
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


class CollectionUpdateTool(MemoryTool):
    """Update collection metadata: description, recall, extraction_prompt, interval.

    Chat-facing.  Lets the user evolve a collection mid-conversation —
    refining its extraction_prompt as the collector's quality becomes
    clearer, swapping recall mode, retiring stale descriptions.  All
    fields except ``name`` are optional; only the ones supplied are
    applied.
    """

    name = "collection_update"
    description = (
        "Update an existing collection's metadata. Only supplied fields "
        "are changed.\n"
        "\n"
        "Fields:\n"
        "- `name` (required) — the collection to update.\n"
        "- `description` — content-reflective one-line summary AND the "
        "stage-1 routing anchor. Changing it re-embeds and re-routes when "
        "the collection surfaces, so keep it an accurate summary of the "
        "subject matter. It does not drive the collector — change the "
        "extraction_prompt for that.\n"
        f"- `inclusion` ({_INCLUSION_MODES}) — stage-1 routing. Flip to "
        "'never' to silence a collection (its collector still runs); 'always' "
        "to always surface it; 'relevant' to gate on the description.\n"
        f"- `recall` ({_RECALL_MODES}) — stage-2 entry rendering once "
        "included.\n"
        "- `published` — flip notify-on-new. `true` starts telling the "
        "user about new entries (they asked to be kept posted / alerted); "
        "`false` silences it (the collector keeps gathering). Omit to leave "
        "unchanged.\n"
        "- `extraction_prompt` — FULL replacement body, not a diff. "
        "Drives what the collector actually does. Read the current body "
        "via `memory_metadata(<collection>)` first if you need to preserve any "
        "of it.\n"
        "- `collector_interval_seconds` — cadence in seconds.\n"
        "\n"
        "Returns a structured echo of the updated state. The echo is "
        "authoritative — if a field you tried to set isn't in it, the "
        'update didn\'t land; fix it and try again rather than saying "done".\n'
        "\n"
        "For workflow guidance — which field maps to which user intent "
        "(scope change vs cadence change vs silent flip), when to call "
        "`memory_metadata(<collection>)` first, when to propose before "
        "applying — see the skills surfaced in your recall context.\n"
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
                    "Content-reflective one-line summary AND the stage-1 "
                    "routing anchor — changing it re-embeds and re-routes "
                    "the collection. Keep it an accurate summary of the "
                    "subject matter."
                ),
            },
            "inclusion": {
                "type": "string",
                "enum": [m.value for m in Inclusion],
                "description": (
                    "Stage-1 routing: 'never' silences a collection (collector "
                    "still runs), 'always' always surfaces it, 'relevant' gates "
                    "on the description."
                ),
            },
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": (
                    "Stage-2 entry rendering once included: 'relevant', 'recent', or 'all'."
                ),
            },
            "published": {
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
                    "scope or silent-flip changes."
                ),
            },
            "collector_interval_seconds": {
                "type": "integer",
                "description": (
                    "How often the collector runs. 1800 (30m), 3600 (1h), "
                    "21600 (6h), 86400 (daily)."
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

    def __init__(self, db: Database, llm_client: LlmClient) -> None:
        self._db = db
        self._llm_client = llm_client

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionUpdateArgs(**kwargs)
        if rejection := _reject_unknown_extraction_tools(
            self._db, self._llm_client, args.extraction_prompt
        ):
            return rejection
        inclusion = Inclusion(args.inclusion) if args.inclusion is not None else None
        recall = RecallMode(args.recall) if args.recall is not None else None
        # Re-embed the routing anchor whenever the description changes.
        description_embedding = (
            await embed_text(self._llm_client, args.description)
            if args.description is not None
            else None
        )
        memory = self._db.memories.update_collection_metadata(
            args.name,
            description=args.description,
            inclusion=inclusion,
            recall=recall,
            extraction_prompt=args.extraction_prompt,
            collector_interval_seconds=args.collector_interval_seconds,
            description_embedding=description_embedding,
            published=args.published,
        )
        suffix = _description_degraded_suffix(args.description, description_embedding)
        message = f"{_format_collection_echo(memory, 'Updated')}{suffix}"
        if args.intent is not None:
            # We serialize `intent` in the metadata the model reads, so it passes it back on
            # an edit.  Accept-and-explain rather than reject the whole call over an immutable
            # field (the model kept getting the update rejected, then giving up).
            message += (
                "\n\n`intent` was not changed — it's fixed at creation and can't be edited via "
                "collection_update (everything else above was applied)."
            )
        return ToolResult(message=message, mutated=True)


class MemoryMetadataTool(MemoryTool):
    """Return the metadata fields for a single memory (collection or log).

    Genuinely shape-agnostic — metadata describes the memory itself, not its
    contents — so it's named ``memory_metadata`` and applies to either shape.
    """

    name = "memory_metadata"
    description = (
        "Return metadata for a memory: description, intent (the user's "
        "original goal), recall mode, collector interval, last collected "
        "timestamp, archived state, and extraction prompt.  Works for both "
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
        interval = (
            f"{memory.collector_interval_seconds}s"
            if memory.collector_interval_seconds is not None
            else "not set"
        )
        last_collected = (
            format_log_timestamp(memory.last_collected_at)
            if memory.last_collected_at is not None
            else "never"
        )
        created = format_log_timestamp(memory.created_at)
        updated = format_log_timestamp(memory.updated_at)
        # Lead with what the collection is FOR (intent) and what it DOES (the recipe),
        # because that is the substance a description should convey; the operational
        # settings (routing/cadence/timestamps) are secondary and go last.  Ordered this
        # way — and with the nudge below — so a model asked "what does this do?" walks
        # through the recipe's steps instead of reciting the cadence/recall trivia (the
        # failure the #1530 legibility baseline surfaced).
        lines = [
            f"name: {memory.name}",
            f"type: {memory.type}",
            f"description: {memory.description}",
            f"intent: {memory.intent or 'none'}",
            "",
            "What it does each cycle — the recipe below is the collection's actual "
            "behaviour.  When explaining the collection, walk through THESE steps, not the "
            "operational settings.",
            f"extraction prompt: {memory.extraction_prompt or 'none'}",
            "",
            "Operational settings (routing + cadence — secondary):",
            f"inclusion: {memory.inclusion}",
            f"recall: {memory.recall}",
            f"published: {memory.published}",
            f"interval: {interval}",
            f"archived: {memory.archived}",
            f"created: {created}",
            f"updated: {updated}",
            f"last collected: {last_collected}",
        ]
        return "\n".join(lines)


class CollectionCatalogTool(MemoryTool):
    """List every active collection with its full gather recipe.

    The skills collector's window onto real use: each non-archived collection
    (logs and framework collectors excluded) with its description, intent,
    ``published`` flag, and full ``extraction_prompt`` — the prompts that
    actually run.  The skills loop distils reusable workflow patterns from these
    and reconciles them against the existing skills, so skills stay grounded in
    the collections that exist rather than in hypothetical teachings.
    """

    name = "collection_catalog"
    description = (
        "List every active collection with its full gather recipe: name, "
        "description, intent (the user's goal in their words), whether it "
        "notifies the user (published), and its extraction_prompt.  Use it to "
        "see what Penny actually collects and how each collection is built.  "
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
            return ToolResult(message="(no active collections)")
        return ToolResult(message="\n\n".join(self._format(row) for row in rows))

    @staticmethod
    def _is_user_collection(row: MemoryRow) -> bool:
        """A non-archived, prompt-bearing collection that gathers user data —
        not a log, not a framework collector (skills/quality/notifier)."""
        return (
            row.type == MemoryType.COLLECTION
            and not row.archived
            and row.extraction_prompt is not None
            and row.name not in PennyConstants.SYSTEM_COLLECTIONS
        )

    @staticmethod
    def _format(row: MemoryRow) -> str:
        return (
            f"## {row.name}\n"
            f"description: {row.description}\n"
            f"intent: {row.intent or '(none)'}\n"
            f"published: {row.published}\n"
            f"extraction_prompt:\n{row.extraction_prompt}"
        )


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

    def __init__(self, db: Database, author: str) -> None:
        self._db = db
        self._author = author

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectionMergeArgs(**kwargs)
        return ToolResult(message=self._merge(args.from_memory, args.to_memory), mutated=True)

    def _merge(self, from_name: str, to_name: str) -> str:
        source = _resolve(self._db, from_name)
        source_keys = source.keys()
        if not source_keys:
            self._db.memories.archive(from_name)
            return f"'{from_name}' was empty — archived with nothing to move."
        moved, dropped = self._move_entries(source, to_name, source_keys)
        self._db.memories.archive(from_name)
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
    that collector's runs (``[target]`` → tools → ``done: <summary>``).  Lets a
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
        # empty batch that reads as "this collector has no runs" (the sibling
        # collector_run_history resolves first for the same reason).
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


class CollectorRunHistoryTool(MemoryTool):
    """Read ONE collector's recent runs as full records, newest first.

    ``log_read("collector-runs")`` gives the cross-collector run index — one
    record per recent run across every collector.  This zooms into a single
    collector: the model passes a collection name (a candidate it spotted in that
    index) and gets that collector's last several runs as the same rendered
    records (counts line + health flags + tool trace).  That's what lets a
    reviewer judge whether a problem is a one-off or a **persistent pattern across
    cycles** before acting on it.  The count is fixed in Python
    (``RUN_HISTORY_RECORDS``) — the model never chooses a size, same as every
    other read.  Stateless (no cursor): re-reading returns the same window.
    """

    name = "collector_run_history"
    parameters = {
        "type": "object",
        "properties": {
            "collector": {
                "type": "string",
                "description": "The collection name whose collector run history to read.",
            }
        },
        "required": ["collector"],
    }
    args_model = CollectorRunHistoryArgs
    description = (
        "Read one collector's recent runs (full records: counts, health flags, "
        "tool trace), newest first — to judge whether a problem is a one-off or a "
        "persistent pattern across cycles.  Pass the collection name; the count is "
        "fixed."
    )

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        collector = _named(arguments, "collector", "a collector")
        if not result.success:
            return f"You tried to review {collector}'s run history but it didn't work:"
        return f"You reviewed {collector}'s run history:"

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = CollectorRunHistoryArgs(**kwargs)
        # Resolve first so an unknown collector returns the actionable
        # "memory not found" refusal (a MemoryAccessError caught by execute),
        # not a silent empty history that reads as "this collector is healthy".
        _resolve(self._db, args.collector)
        records = self._db.messages.target_run_records(
            args.collector, PennyConstants.RUN_HISTORY_RECORDS
        )
        if not records:
            return ToolResult(
                message=(
                    f"No completed runs recorded yet for `{args.collector}` — it may be "
                    "newly created or never have run.  Judge it from its current run in "
                    "the index, not its history."
                )
            )
        return ToolResult(
            message=_format_entries(records, source=args.collector, ordering="most recent first")
        )


class _PublishedItem(NamedTuple):
    """One candidate from a published collection: the source name + its intent
    (for framing) and the entry itself."""

    memory_name: str
    intent: str | None
    entry: MemoryEntry


class ReadPublishedLatestTool(CursorReadTool):
    """Fan-in cursored read across every ``published`` collection — the consumer
    side of the pub/sub layer.

    Each call picks **one** published (non-archived) collection *at random* among
    those with an entry this consumer hasn't seen, and returns its ``n`` oldest
    unseen entries, each tagged with its source collection and that collection's
    intent so a generic consumer prompt can frame a message without naming
    sources.  Random rotation — rather than a global oldest-first pool — is what
    keeps any one collection's burst from monopolizing delivery: a single
    collector run stamps many entries with near-identical timestamps, so a global
    oldest-first drain delivered a whole burst back-to-back (hours of one topic)
    and starved low-volume collections for days.  Picking a random eligible
    collection each cycle gives every collection with something new an equal shot,
    so they drain evenly and no burst blocks the rest.

    A per-``(consumer, source)`` cursor tracks progress (pending until the cycle
    commits) and advances **only for the entries actually returned** — never for
    a source merely scanned — so nothing is skipped.  A source this consumer has
    no cursor for yet starts ``PUBLISHED_COLDSTART_LOOKBACK_SECONDS`` back, so a
    freshly published backlog isn't replayed in full.

    Mirrors ``LogReadTool``'s pending/commit lifecycle: the orchestration layer
    calls ``commit_pending`` after a successful cycle, ``discard_pending`` after
    a failed one.
    """

    name = "read_published_latest"
    description = (
        "Return the oldest entries you haven't seen yet across every collection "
        "that publishes to you, each tagged with its source collection and that "
        "collection's intent.  A cursor tracks where you left off per source, so "
        "the next call returns the next ones and nothing repeats.  You never name "
        "a source — this spans all of them.  Use it to find the next new thing "
        "worth telling the user about."
    )
    parameters = {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "Max entries to return (default 1 — the single oldest unseen).",
            },
        },
    }
    args_model = ReadPublishedLatestArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        if not result.success:
            return "You tried to check for new entries to share but it didn't work:"
        return "You checked for new entries to share:"

    async def _run(self, **kwargs: Any) -> ToolResult:
        args = ReadPublishedLatestArgs(**kwargs)
        selected = self._select(args.n)
        for item in selected:
            self._advance_pending(item.memory_name, [item.entry.created_at])
        return ToolResult(message=self._format(selected))

    def _select(self, n: int) -> list[_PublishedItem]:
        """Pick one published source at random among those with unseen entries and
        return its oldest n — random rotation so no one collection's burst
        monopolizes delivery (see the class docstring)."""
        sources = self._sources_with_unseen()
        if not sources:
            return []
        source = random.choice(sources)
        cursor = self._cursor_for(source.name)
        entries = self._db.memory(source.name).read_since(cursor, n)
        return [_PublishedItem(source.name, source.intent, entry) for entry in entries]

    def _sources_with_unseen(self) -> list[MemoryRow]:
        """Published, non-archived collections (this consumer aside) holding at
        least one entry past this consumer's cursor."""
        sources: list[MemoryRow] = []
        for row in self._db.memories.list_all():
            if not row.published or row.archived or row.name == self._agent_name:
                continue
            if self._db.memory(row.name).read_since(self._cursor_for(row.name), 1):
                sources.append(row)
        return sources

    def _cursor_for(self, memory_name: str) -> datetime:
        """This consumer's committed cursor for ``memory_name``, or the cold-start
        window when it has never read this source."""
        committed = self._db.cursors.get(self._agent_name, memory_name)
        if committed is not None:
            return committed
        return datetime.now(UTC) - timedelta(
            seconds=PennyConstants.PUBLISHED_COLDSTART_LOOKBACK_SECONDS
        )

    def _format(self, items: list[_PublishedItem]) -> str:
        if not items:
            return "(no new published entries)"
        lines = []
        for index, item in enumerate(items, start=1):
            intent = f" (intent: {item.intent})" if item.intent else ""
            key = f"{render_key(item.entry.key)} " if item.entry.key else ""
            stamp = f"[{format_log_timestamp(item.entry.created_at)}] "
            lines.append(
                f"{index}. {stamp}from `{item.memory_name}`{intent}: {key}{item.entry.content}"
            )
        noun = "entry" if len(items) == 1 else "entries"
        return f"{len(items)} new published {noun} (oldest first):\n" + "\n".join(lines)


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

    def __init__(self, db: Database, llm_client: LlmClient, author: str) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author

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
        )
        return ToolResult(message=f"Appended to '{args.memory}'.", mutated=True)


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
    """Signal the cycle is finished, with a structured success + summary report."""

    name = PennyConstants.DONE_TOOL_NAME
    description = (
        "Call this when the cycle is finished.  REQUIRED: `success` (true if "
        "you did what the prompt asked, false on no-op or failure) and "
        "`summary` (one-sentence prose describing what the cycle actually "
        "did — entries written, messages sent, why no-op).  Both are logged "
        "to `collector-runs` for auditing; `reasoning` alone is never a "
        "valid done call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the cycle did what the prompt asked.",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence description of what was done.",
            },
        },
        "required": ["success", "summary"],
    }
    args_model = DoneArgs

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        # The done call itself always succeeds; its ``success`` argument reports
        # the *cycle's* outcome, so the narration reflects that.
        if arguments.get("success") is False:
            return "You wrapped up the cycle, marking it unfinished:"
        return "You wrapped up the cycle:"

    async def execute(self, **kwargs: Any) -> ToolResult:
        args = DoneArgs(**kwargs)
        marker = "success" if args.success else "no-op/fail"
        # The ``success`` arg reports the *cycle's* outcome (read from the call's
        # arguments by the collector); the done call itself always succeeds and
        # never mutates state.
        return ToolResult(message=f"Cycle complete ({marker}): {args.summary}")


# ── On-demand collector trigger ─────────────────────────────────────────────


class TestExtractionPromptTool(Tool):
    """Immediately run the collector for a named collection, bypassing the schedule."""

    name = "test_extraction_prompt"
    timeout = 300.0  # collector cycles include browse calls that can take several minutes
    description = (
        "Immediately trigger one collector cycle for the named collection, bypassing "
        "the normal idle-gated schedule.  Use this while authoring or refining an "
        "extraction_prompt to verify the collector reads the right sources and writes "
        "the expected entries.  Returns the cycle's success flag and done() summary."
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
    it's covered for free.  ``BrowseTool`` / ``SendMessageTool`` are imported lazily:
    ``send_message`` imports ``DoneTool`` from this module, so a top-level import here
    would close that cycle.
    """
    from penny.tools.browse import BrowseTool
    from penny.tools.send_message import SendMessageTool

    memory_names = {tool.name for tool in build_memory_tools(db, llm_client, _VOCAB_PROBE_AGENT)}
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
) -> list[Tool]:
    """Construct the memory tool surface for an agent.

    **One uniform surface for every agent** — reads + lifecycle (shape)
    + entry mutations (contents).  Capability is no longer curated by
    omission; instead every tool funnels its resolve + op through one ``try``:
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
    genuinely shape-agnostic reads — they work on either shape.

    ``DoneTool`` / ``send_message`` are intentionally not here — they're
    loop-control, not capability, added in ``BackgroundAgent.get_tools``.
    Chat replies via final text and must not have ``done`` available, or
    the model may call it instead of producing a reply.
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
        CollectorRunHistoryTool(db),
        ReadPublishedLatestTool(db, agent_name),
        ExistsTool(db, llm_client),
    ]
    lifecycle: list[Tool] = [
        CollectionCreateTool(db, llm_client),
        CollectionUpdateTool(db, llm_client),
        CollectionMergeTool(db, agent_name),
        CollectionArchiveTool(db),
        CollectionUnarchiveTool(db),
        LogCreateTool(db, llm_client),
    ]
    mutations: list[Tool] = [
        CollectionWriteTool(db, llm_client, agent_name, scope=scope),
        UpdateEntryTool(db, agent_name, scope=scope),
        CollectionDeleteEntryTool(db, scope=scope),
        LogAppendTool(db, llm_client, agent_name),
    ]
    return reads + lifecycle + mutations
