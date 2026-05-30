"""Tool-layer wrappers over the memory access layer.

Every tool validates its kwargs through a Pydantic args model as its first
line (per CLAUDE.md), calls ``db.memories.*``, and returns a serializable
string the model can reason over.

Author attribution is passed explicitly: write-capable tools take an
``author: str`` at construction time (the agent that owns the tool).
``build_memory_tools(db, embedding_client, author)`` is the factory each
agent calls with its own ``self.name`` so writes are attributed correctly.

Tools that need embeddings (writes, similarity reads, ``exists``) take an
``LlmClient`` in ``__init__``. If no embedding client is configured they
degrade gracefully: writes proceed without key/content vectors, similarity
reads return empty.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import (
    DedupThresholds,
    EntryInput,
    LogEntryInput,
    MemoryAlreadyExistsError,
    MemoryNotFoundError,
    RecallMode,
    WriteResult,
)
from penny.database.models import MemoryEntry
from penny.llm.similarity import embed_text
from penny.tools.base import Tool
from penny.tools.memory_args import (
    CollectionDeleteEntryArgs,
    CollectionEntrySpec,
    CollectionGetArgs,
    CollectionMergeArgs,
    CollectionMoveArgs,
    CollectionUpdateArgs,
    CollectionWriteArgs,
    CreateMemoryArgs,
    DoneArgs,
    ExistsArgs,
    LogAppendArgs,
    MemoryNameArgs,
    ReadLatestArgs,
    ReadNextArgs,
    ReadRandomArgs,
    ReadRecentArgs,
    ReadSimilarArgs,
    UpdateEntryArgs,
)

if TYPE_CHECKING:
    from penny.agents.collector import Collector
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)


_RECALL_MODES = ", ".join(m.value for m in RecallMode)
EXTRACTION_PROMPT_MIN_CHARS = 25


# ── Shared formatting ───────────────────────────────────────────────────────


def _format_entries(entries: list[MemoryEntry]) -> str:
    """Render a list of entries as a bulleted string the model can read.

    Keyed entries (collection) include the key; keyless entries (log) show
    just content. Empty lists produce a clear "no entries" sentinel so the
    model doesn't confuse absence with error.
    """
    if not entries:
        return "(no entries)"
    lines = []
    for entry in entries:
        prefix = f"[{entry.key}] " if entry.key else ""
        lines.append(f"- {prefix}{entry.content}")
    return "\n".join(lines)


def check_extraction_prompt(prompt: str | None) -> str | None:
    """Return an error string if prompt is set but too short, else None."""
    if prompt is None or len(prompt) >= EXTRACTION_PROMPT_MIN_CHARS:
        return None
    return (
        f"extraction_prompt is too short ({len(prompt)} chars — minimum "
        f"{EXTRACTION_PROMPT_MIN_CHARS}).  Provide a full numbered-step prompt "
        f"(see the collection_create description for the required shape) or omit "
        f"extraction_prompt entirely to create a manually-managed collection."
    )


# ── Metadata ────────────────────────────────────────────────────────────────


class CollectionCreateTool(Tool):
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
        "Create a new keyed collection.  Provide ``description``, "
        f"``recall`` mode ({_RECALL_MODES}), and an ``extraction_prompt``.  "
        "Without the extraction_prompt the collection stays empty — it is "
        "the system prompt the per-collection Collector subagent runs each "
        "cycle.\n"
        "\n"
        "# Writing extraction_prompts\n"
        "\n"
        "Every extraction_prompt is a numbered list of explicit tool calls "
        "plus a short tail.  The collector executes the steps in order — "
        "flowing prose loses the model.\n"
        "\n"
        "## The standard shape\n"
        "\n"
        "```\n"
        "[One-sentence statement of what this collector does.]\n"
        "\n"
        "1. [Read source — explicit tool call.]\n"
        "2. [Identify what counts; list what to skip.]\n"
        "3. [Optionally: browse for current info.]\n"
        "4. [Write — explicit collection_write tool call.]\n"
        "5. [Optionally: send_message, gated on successful write.]\n"
        "6. [Handle corrections.]\n"
        "7. done().\n"
        "\n"
        '[Tail: edge-case rules, "if nothing matches, just done()".]\n'
        "```\n"
        "\n"
        "## Worked examples — clone the closest, customise\n"
        "\n"
        "Match the user's request to a shape: \"find me X and tell me when "
        'there\'s something new" → **Research + notify**.  "Track topics I '
        'mention" → **Pure extraction**.\n'
        "\n"
        '### Research + notify on new finds (for "find me X, track them, tell '
        "me when there's something new\")\n"
        "\n"
        "> Collect [topic] — [scope].\n"
        ">\n"
        '> 1. log_read_next("user-messages") and log_read_next("browse-results") '
        "for recent context and any pages fetched since last cycle.\n"
        "> 2. browse the web for new [topic] items when there's a topical "
        "opening.  Read actual pages — never cite from search snippets alone.\n"
        "> 3. Each entry: key is the item's name (3-10 words); content is name "
        "+ description + a real source URL pulled from a page browsed THIS "
        "cycle.\n"
        '> 4. collection_write("[bound collection]", entries=[...]) batching '
        "all new items.\n"
        "> 5. ONLY IF the write succeeded (not duplicate-rejected): "
        'send_message with a one-or-two-sentence "found something new for '
        '[topic]" note, conversational, finish with an emoji.\n'
        "> 6. If a recent message indicates an existing entry is wrong "
        "(closed, link dead, plans changed), update_entry or "
        "collection_delete_entry.\n"
        "> 7. done().  If nothing new, just done().\n"
        ">\n"
        "> Cite only sources you actually browsed this cycle.  Never invent URLs.\n"
        "\n"
        "### Pure extraction (the ``likes`` shape, for tracking topics the "
        "user mentions in chat)\n"
        "\n"
        "> Extract the user's positive preferences from their recent messages.\n"
        ">\n"
        '> 1. log_read_next("user-messages") — fetch new messages.\n'
        "> 2. Identify every LIKE — a thing the user wants, enjoys, or "
        "expresses positive sentiment about.  Skip questions, troubleshooting, "
        'meta-instructions ("remember this", "track that").\n'
        "> 3. Each entry: key is the topic fully-qualified (3-10 words: "
        '"Talk (album) by Yes" not "the album"); content is the user\'s raw '
        "message.\n"
        '> 4. collection_write("likes", entries=[...]) batching all extracted '
        "likes.\n"
        "> 5. If a recent message indicates an existing like is no longer "
        "accurate, update_entry or collection_delete_entry.\n"
        "> 6. done().  If nothing matches, just done() without writing.\n"
        "\n"
        "## Authoring rules\n"
        "\n"
        "- Name every tool explicitly in the steps: "
        '``log_read_next("X")``, ``collection_write("X", entries=[...])``, '
        "``send_message(content=...)``, ``done()``.  The collector won't "
        "call a tool the prompt doesn't name.\n"
        "- Only include ``send_message`` if the user explicitly asked for "
        'notification-on-new ("tell me when there\'s something new").  '
        "Pure-tracking requests should NOT include send_message — the user "
        "doesn't want a chat ping for every entry.\n"
        "\n"
        "(Runtime invariants — quiet-cycle escape, batched writes, "
        "send_message gating, source-provenance, correction handling, "
        "structured ``done(success, summary)`` reporting — are appended "
        "automatically; you don't need to include them.)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique collection name"},
            "description": {
                "type": "string",
                "description": "One-line summary shown in the memory registry",
            },
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": "How the chat agent surfaces this collection in ambient context",
            },
            "extraction_prompt": {
                "type": "string",
                "description": (
                    "Instructions for the per-collection collector subagent. "
                    "Should describe what to extract, from which logs, and how "
                    "to handle corrections/removals."
                ),
            },
        },
        "required": ["name", "description", "recall"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CreateMemoryArgs(**kwargs)
        if error := check_extraction_prompt(args.extraction_prompt):
            return error
        try:
            self._db.memories.create_collection(
                args.name,
                args.description,
                RecallMode(args.recall),
                extraction_prompt=args.extraction_prompt,
            )
        except MemoryAlreadyExistsError:
            return f"Collection '{args.name}' already exists."
        return f"Created collection '{args.name}'."


class LogCreateTool(Tool):
    """Create a new append-only log."""

    name = "log_create"
    description = (
        "Create a new append-only log. Logs store keyless entries in time order "
        "and are meant for streams of events (messages, measurements, etc.). "
        f"Provide a short description and a recall mode ({_RECALL_MODES})."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique log name"},
            "description": {"type": "string", "description": "One-line summary"},
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
                "description": "How the chat agent surfaces this log in ambient context",
            },
        },
        "required": ["name", "description", "recall"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CreateMemoryArgs(**kwargs)
        try:
            self._db.memories.create_log(args.name, args.description, RecallMode(args.recall))
        except MemoryAlreadyExistsError:
            return f"Log '{args.name}' already exists."
        return f"Created log '{args.name}'."


class CollectionArchiveTool(Tool):
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

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.archive(args.memory)
        return f"Archived '{args.memory}'."


class CollectionUnarchiveTool(Tool):
    """Restore a previously archived collection to ambient recall."""

    name = "collection_unarchive"
    description = "Unarchive a previously archived collection."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        self._db.memories.unarchive(args.memory)
        return f"Unarchived '{args.memory}'."


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionGetTool(Tool):
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

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionGetArgs(**kwargs)
        rows = self._db.memories.get_entry(args.memory, args.key)
        if not rows:
            return f"Key '{args.key}' not found in '{args.memory}'."
        return _format_entries(rows)


class ReadLatestTool(Tool):
    """Return the newest entries in a memory (works for collections and logs)."""

    name = "read_latest"
    description = (
        "Return the newest entries in a memory, newest first. Works for "
        "both collections and logs. Omit ``k`` to return every entry."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadLatestArgs(**kwargs)
        entries = self._db.memories.read_latest(args.memory, args.k)
        return _format_entries(entries)


class CollectionReadRandomTool(Tool):
    """Return entries sampled uniformly at random from a collection."""

    name = "collection_read_random"
    description = "Return ``k`` entries sampled uniformly at random. Omit ``k`` to return all."
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "k": {"type": "integer"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadRandomArgs(**kwargs)
        entries = self._db.memories.read_random(args.memory, args.k)
        return _format_entries(entries)


class ReadSimilarTool(Tool):
    """Return entries most similar to an anchor phrase (collections or logs)."""

    name = "read_similar"
    description = (
        "Return entries from a memory ordered by content similarity to an "
        "``anchor`` phrase. Works for both collections and logs — use this "
        "to find past conversations on a topic (search ``user-messages`` or "
        "``penny-messages``), past browse results, related preferences or "
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

    def __init__(self, db: Database, llm_client: LlmClient | None) -> None:
        self._db = db
        self._llm = llm_client

    async def execute(self, **kwargs: Any) -> str:
        args = ReadSimilarArgs(**kwargs)
        vec = await embed_text(self._llm, args.anchor)
        if vec is None:
            logger.warning(
                "%s: similarity search unavailable — no embedding model configured", self.name
            )
            return "(similarity search unavailable — no embedding model configured)"
        entries = self._db.memories.read_similar(args.memory, vec, args.k)
        return _format_entries(entries)


class CollectionKeysTool(Tool):
    """List the unique keys currently in a collection."""

    name = "collection_keys"
    description = "List the unique keys in a collection (insertion order)."
    parameters = {
        "type": "object",
        "properties": {"memory": {"type": "string"}},
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        keys = self._db.memories.keys(args.memory)
        if not keys:
            return "(no keys)"
        return "\n".join(f"- {key}" for key in keys)


# ── Collection writes ───────────────────────────────────────────────────────


def _format_duplicate(result: WriteResult) -> str:
    """Format one duplicate result for the rejection message.

    Names the matching existing key when present so the model can pivot
    to ``update_entry`` instead of silently dropping fresher info.
    Falls back to just the candidate key when ``matched_key`` is missing
    (e.g. the matched existing entry had no key set)."""
    if result.matched_key and result.matched_key != result.key:
        return f"{result.key} (matches existing '{result.matched_key}')"
    return result.key


class CollectionWriteTool(Tool):
    """Write entries to a collection with similarity-based dedup."""

    name = "collection_write"
    description = (
        "Write one or more entries to a collection. Each entry has a short "
        "``key`` (topic/identifier) and a longer ``content`` body. Dedup runs "
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

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient | None,
        author: str,
        scope: str | None = None,
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author
        self._scope = scope

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionWriteArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return (
                f"Refused: this collector can only write to '{self._scope}', not '{args.memory}'."
            )
        entries = [await self._build_entry(spec) for spec in args.entries]
        results = self._db.memories.write(args.memory, entries, author=self._author)
        return self._format_results(args.memory, results)

    async def _build_entry(self, spec: CollectionEntrySpec) -> EntryInput:
        return EntryInput(
            key=spec.key,
            content=spec.content,
            key_embedding=await embed_text(self._llm, spec.key),
            content_embedding=await embed_text(self._llm, spec.content),
        )

    def _format_results(self, memory: str, results: list[WriteResult]) -> str:
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
            parts.append(
                f"Rejected as duplicates: {', '.join(labelled)}.  "
                f"Use ``update_entry`` to refresh an existing row if you have richer info."
            )
        if rejected:
            labelled = [f"{r.key} ({r.reason})" for r in rejected]
            parts.append(f"Rejected as degenerate content: {', '.join(labelled)}.")
        return " ".join(parts) if parts else "(no entries written)"


class UpdateEntryTool(Tool):
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

    def __init__(self, db: Database, author: str, scope: str | None = None) -> None:
        self._db = db
        self._author = author
        self._scope = scope

    async def execute(self, **kwargs: Any) -> str:
        args = UpdateEntryArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return (
                f"Refused: this collector can only write to '{self._scope}', not '{args.memory}'."
            )
        outcome = self._db.memories.update(args.memory, args.key, args.content, self._author)
        if outcome == "not_found":
            return f"Key '{args.key}' not found in '{args.memory}'."
        return f"Updated '{args.key}' in '{args.memory}'."


class CollectionUpdateTool(Tool):
    """Update collection metadata: description, recall, extraction_prompt, interval.

    Chat-facing.  Lets the user evolve a collection mid-conversation —
    refining its extraction_prompt as the collector's quality becomes
    clearer, swapping recall mode, retiring stale descriptions.  All
    fields except ``name`` are optional; only the ones supplied are
    applied.
    """

    name = "collection_update"
    description = (
        "Update an existing collection's metadata.  Provide the collection "
        "``name`` plus any of: ``description``, ``recall`` "
        f"({_RECALL_MODES}), ``extraction_prompt``, ``collector_interval_seconds``. "
        "Only the fields you supply are changed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Collection name to update"},
            "description": {"type": "string"},
            "recall": {
                "type": "string",
                "enum": [m.value for m in RecallMode],
            },
            "extraction_prompt": {"type": "string"},
            "collector_interval_seconds": {"type": "integer"},
        },
        "required": ["name"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionUpdateArgs(**kwargs)
        if error := check_extraction_prompt(args.extraction_prompt):
            return error
        recall = RecallMode(args.recall) if args.recall is not None else None
        try:
            self._db.memories.update_collection_metadata(
                args.name,
                description=args.description,
                recall=recall,
                extraction_prompt=args.extraction_prompt,
                collector_interval_seconds=args.collector_interval_seconds,
            )
        except MemoryNotFoundError:
            return f"Collection '{args.name}' not found."
        return f"Updated '{args.name}'."


class CollectionMetadataTool(Tool):
    """Return the metadata fields for a single memory (collection or log)."""

    name = "collection_metadata"
    description = (
        "Return metadata for a memory: description, recall mode, "
        "collector interval, last collected timestamp, archived state, "
        "and extraction prompt.  Works for both collections and logs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string", "description": "Collection or log name"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        memory = self._db.memories.get(args.memory)
        if memory is None:
            return f"Memory '{args.memory}' not found."
        return self._format(memory)

    def _format(self, memory: Any) -> str:
        interval = (
            f"{memory.collector_interval_seconds}s"
            if memory.collector_interval_seconds is not None
            else "not set"
        )
        last_collected = (
            memory.last_collected_at.strftime("%Y-%m-%d %H:%M:%S")
            if memory.last_collected_at is not None
            else "never"
        )
        created = memory.created_at.strftime("%Y-%m-%d %H:%M:%S")
        updated = memory.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"name: {memory.name}",
            f"type: {memory.type}",
            f"description: {memory.description}",
            f"recall: {memory.recall}",
            f"archived: {memory.archived}",
            f"created: {created}",
            f"updated: {updated}",
            f"interval: {interval}",
            f"last collected: {last_collected}",
            f"extraction prompt: {memory.extraction_prompt or 'none'}",
        ]
        return "\n".join(lines)


class CollectionMoveTool(Tool):
    """Move an entry between collections by key."""

    name = "collection_move"
    description = (
        "Move the entry with the given key from one collection to another. "
        "Fails with 'collision' if the target already has an entry with that key."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "from_memory": {"type": "string"},
            "to_memory": {"type": "string"},
        },
        "required": ["key", "from_memory", "to_memory"],
    }

    def __init__(self, db: Database, author: str, scope: str | None = None) -> None:
        self._db = db
        self._author = author
        self._scope = scope
        if scope is not None:
            # When scoped, to_memory is always the bound collection — make it optional
            # so the model doesn't fail validation if it omits the predetermined value.
            self.parameters = {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "from_memory": {"type": "string"},
                    "to_memory": {
                        "type": "string",
                        "description": f"Destination collection; defaults to '{scope}'.",
                    },
                },
                "required": ["key", "from_memory"],
            }

    async def execute(self, **kwargs: Any) -> str:
        if self._scope is not None and "to_memory" not in kwargs:
            kwargs["to_memory"] = self._scope
        args = CollectionMoveArgs(**kwargs)
        # Scope constrains the destination side of the move (the write).
        # Source-side ``from_memory`` is unrestricted — moving an entry
        # OUT of another collection into the bound scope is allowed,
        # since the only entry that ends up written is in scope.
        if self._scope is not None and args.to_memory != self._scope:
            return (
                f"Refused: this collector can only write to '{self._scope}', "
                f"not '{args.to_memory}'."
            )
        outcome = self._db.memories.move(
            args.key, args.from_memory, args.to_memory, author=self._author
        )
        if outcome == "not_found":
            return f"Key '{args.key}' not found in '{args.from_memory}'."
        if outcome == "collision":
            return f"Cannot move: '{args.to_memory}' already has a '{args.key}' entry."
        return f"Moved '{args.key}' from '{args.from_memory}' to '{args.to_memory}'."


class CollectionMergeTool(Tool):
    """Merge all entries from one collection into another, then archive the source."""

    name = "collection_merge"
    description = (
        "Move every entry from ``from_memory`` into ``to_memory``, then archive "
        "``from_memory``.  Entries whose key already exists in ``to_memory`` are "
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

    def __init__(self, db: Database, author: str) -> None:
        self._db = db
        self._author = author

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionMergeArgs(**kwargs)
        return self._merge(args.from_memory, args.to_memory)

    def _merge(self, from_name: str, to_name: str) -> str:
        source_keys = self._db.memories.keys(from_name)
        if not source_keys:
            self._db.memories.archive(from_name)
            return f"'{from_name}' was empty — archived with nothing to move."
        moved, dropped = self._move_entries(from_name, to_name, source_keys)
        self._db.memories.archive(from_name)
        return self._summary(from_name, to_name, moved, dropped)

    def _move_entries(self, from_name: str, to_name: str, keys: list[str]) -> tuple[int, int]:
        moved = 0
        dropped = 0
        for key in keys:
            outcome = self._db.memories.move(key, from_name, to_name, author=self._author)
            if outcome == "ok":
                moved += 1
            else:
                dropped += 1
        return moved, dropped

    def _summary(self, from_name: str, to_name: str, moved: int, dropped: int) -> str:
        parts = [f"Merged '{from_name}' → '{to_name}': {moved} moved"]
        if dropped:
            parts.append(f"{dropped} dropped (key already in destination)")
        parts.append(f"'{from_name}' archived.")
        return ", ".join(parts[:2]) + f". {parts[-1]}" if dropped else f"{parts[0]}. {parts[-1]}"


class CollectionDeleteEntryTool(Tool):
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

    def __init__(self, db: Database, scope: str | None = None) -> None:
        self._db = db
        self._scope = scope

    async def execute(self, **kwargs: Any) -> str:
        args = CollectionDeleteEntryArgs(**kwargs)
        if self._scope is not None and args.memory != self._scope:
            return (
                f"Refused: this collector can only write to '{self._scope}', not '{args.memory}'."
            )
        removed = self._db.memories.delete(args.memory, args.key)
        if removed == 0:
            return f"No entry with key '{args.key}' in '{args.memory}'."
        return f"Deleted '{args.key}' from '{args.memory}'."


# ── Log reads ───────────────────────────────────────────────────────────────


class LogReadRecentTool(Tool):
    """Return log entries created within the past ``window_seconds`` seconds."""

    name = "log_read_recent"
    description = (
        "Return entries created within the past ``window_seconds`` seconds, "
        "oldest first. Use for 'what just happened' queries."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "window_seconds": {
                "type": "integer",
                "description": (
                    "Look-back window in seconds; defaults to "
                    f"{PennyConstants.LOG_READ_RECENT_DEFAULT_WINDOW_SECONDS} (1 hour) if omitted"
                ),
            },
            "cap": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database) -> None:
        self._db = db

    async def execute(self, **kwargs: Any) -> str:
        args = ReadRecentArgs(**kwargs)
        entries = self._db.memories.read_recent(args.memory, args.window_seconds, args.cap)
        return _format_entries(entries)


class LogReadNextTool(Tool):
    """Read entries appended since the agent's last committed cursor.

    Cursor advance is *pending* until the orchestration layer calls
    ``commit_pending`` after a successful run.  A failed run discards the
    pending cursor, so the next run sees the same entries again.
    """

    name = "log_read_next"
    description = (
        "Return entries appended to a log since this agent's last committed read. "
        "Use this to process new content incrementally without re-seeing entries "
        "from earlier runs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "memory": {"type": "string"},
            "cap": {"type": "integer", "description": "Max entries; omit for all"},
        },
        "required": ["memory"],
    }

    def __init__(self, db: Database, agent_name: str) -> None:
        self._db = db
        self._agent_name = agent_name
        self._pending: dict[str, datetime] = {}

    async def execute(self, **kwargs: Any) -> str:
        args = ReadNextArgs(**kwargs)
        cursor = self._db.cursors.get(self._agent_name, args.memory)
        if cursor is None:
            # First cycle on this log — bound the fetch to the most-recent N
            # entries instead of every entry since the beginning of time.
            # Otherwise a brand-new collector would dump the full log history
            # (months of user messages) into the first cycle's context.
            # Newest-first from read_latest, reversed so cursor advance
            # tracks the latest entry's timestamp.
            entries = list(
                reversed(
                    self._db.memories.read_latest(
                        args.memory, k=PennyConstants.LOG_READ_NEXT_INITIAL_LIMIT
                    )
                )
            )
        else:
            entries = self._db.memories.read_since(args.memory, cursor, args.cap)
        if entries:
            max_seen = max(e.created_at for e in entries)
            prev = self._pending.get(args.memory)
            if prev is None or max_seen > prev:
                self._pending[args.memory] = max_seen
        return _format_entries(entries)

    def commit_pending(self) -> None:
        """Persist the highest timestamp seen during this run as the new cursor.

        Called by the orchestration layer after a successful run.  Discards
        any pending state on completion so a re-used tool instance starts
        fresh.
        """
        for memory_name, last_read_at in self._pending.items():
            self._db.cursors.advance_committed(self._agent_name, memory_name, last_read_at)
        self._pending.clear()

    def discard_pending(self) -> None:
        """Drop pending cursor advance — used after a failed run."""
        self._pending.clear()


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendTool(Tool):
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

    def __init__(self, db: Database, llm_client: LlmClient | None, author: str) -> None:
        self._db = db
        self._llm = llm_client
        self._author = author

    async def execute(self, **kwargs: Any) -> str:
        args = LogAppendArgs(**kwargs)
        vec = await embed_text(self._llm, args.content)
        self._db.memories.append(
            args.memory,
            [LogEntryInput(content=args.content, content_embedding=vec)],
            author=self._author,
        )
        return f"Appended to '{args.memory}'."


# ── Introspection / lifecycle ───────────────────────────────────────────────


class ExistsTool(Tool):
    """Probe whether an equivalent entry already exists across a set of memories."""

    name = "exists"
    description = (
        "Check whether an entry equivalent to the given key/content already "
        "exists in any of the listed memories. Uses the same similarity-based "
        "dedup rule as ``collection_write``. Use this before writing to avoid "
        "duplicates that span multiple collections."
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

    def __init__(
        self,
        db: Database,
        llm_client: LlmClient | None,
        thresholds: DedupThresholds | None = None,
    ) -> None:
        self._db = db
        self._llm = llm_client
        self._thresholds = thresholds

    async def execute(self, **kwargs: Any) -> str:
        args = ExistsArgs(**kwargs)
        # When the model probes with content but no key, treat the content
        # as a name-like probe — using it as both ``key`` and ``content``
        # lets the dedup's key-TCR signal fire against existing entries
        # whose ``key`` matches.  Without this, ``exists(content="Kepler
        # Museum")`` returned "no" against an existing ``key="Kepler
        # Museum"`` because content-cosine alone (candidate "Kepler
        # Museum" vs the existing entry's long description) sat below
        # the strict threshold.
        key = args.key if args.key else args.content
        key_vec = await embed_text(self._llm, key)
        content_vec = await embed_text(self._llm, args.content)
        found = self._db.memories.exists(
            args.memories,
            key,
            key_vec,
            content_vec,
            thresholds=self._thresholds,
        )
        return "yes" if found else "no"


class DoneTool(Tool):
    """Signal the cycle is finished, with a structured success + summary report."""

    name = "done"
    description = (
        "Call this when the cycle is finished.  Pass ``success`` (true if "
        "you did what the prompt asked, false on no-op or failure) and "
        "``summary`` (one-sentence prose describing what the cycle actually "
        "did — entries written, messages sent, why no-op).  Both are logged "
        "to ``collector-runs`` for auditing."
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

    async def execute(self, **kwargs: Any) -> str:
        args = DoneArgs(**kwargs)
        marker = "success" if args.success else "no-op/fail"
        return f"Cycle complete ({marker}): {args.summary}"


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

    def __init__(self, collector: Collector) -> None:
        self._collector = collector

    async def execute(self, **kwargs: Any) -> str:
        args = MemoryNameArgs(**kwargs)
        success, summary = await self._collector.run_for(args.memory)
        marker = "✅" if success else "❌"
        return f"{marker} {summary}"


# ── Factory ─────────────────────────────────────────────────────────────────


def build_memory_tools(
    db: Database,
    llm_client: LlmClient | None,
    agent_name: str,
    scope: str | None = None,
) -> list[Tool]:
    """Construct the memory tool surface for an agent.

    Two distinct surfaces, mutually exclusive:

    * ``scope=None`` — **chat surface**.  Reads + lifecycle tools
      (``collection_create``, ``collection_update`` for metadata,
      ``collection_archive`` / ``collection_unarchive``, ``log_create``).
      Chat owns the *shape* of memory.  No entry-mutation tools at all
      (writes / updates / deletes / moves of entries, log appends) —
      those belong to collectors.

    * ``scope=X`` — **collector surface** for a collector bound to
      collection ``X``.  Reads (unrestricted — a collector may pull
      context from other memories) + entry mutations pinned to ``X``
      (``collection_write`` / ``update_entry`` /
      ``collection_delete_entry``, plus ``collection_move`` when
      ``to_memory == X``) + ``log_append`` (logs are append-only inputs;
      not the scope constraint) + ``send_message`` (added by
      ``BackgroundAgent.get_tools`` when channel is wired).  Collectors
      own the *contents* of their bound collection.

    Reads are shape-agnostic (``read_latest`` / ``read_similar``); the
    parallel ``collection_*`` / ``log_*`` versions were merged earlier
    since they share the same access-layer call.  ``read_all`` was
    removed — pagination via ``read_latest(memory, k=N)`` is always
    safer than dumping a 1,000-entry collection into the prompt.

    ``DoneTool`` is intentionally not in this surface — it's a
    background-agent terminator added in ``BackgroundAgent.get_tools``
    alongside ``send_message``.  Chat replies via final text and must
    not have ``done`` available, or the model may call it instead of
    producing a reply.
    """
    reads: list[Tool] = [
        ReadLatestTool(db),
        ReadSimilarTool(db, llm_client),
        CollectionGetTool(db),
        CollectionReadRandomTool(db),
        CollectionKeysTool(db),
        CollectionMetadataTool(db),
        LogReadRecentTool(db),
        LogReadNextTool(db, agent_name),
        ExistsTool(db, llm_client),
    ]
    if scope is not None:
        # Collector: reads + entry mutations on `scope` + log_append
        return reads + [
            CollectionWriteTool(db, llm_client, agent_name, scope=scope),
            UpdateEntryTool(db, agent_name, scope=scope),
            CollectionDeleteEntryTool(db, scope=scope),
            CollectionMoveTool(db, agent_name, scope=scope),
            LogAppendTool(db, llm_client, agent_name),
        ]
    # Chat: reads + lifecycle, no entry mutations
    return [
        CollectionCreateTool(db),
        CollectionUpdateTool(db),
        CollectionMergeTool(db, agent_name),
        CollectionArchiveTool(db),
        CollectionUnarchiveTool(db),
        LogCreateTool(db),
        *reads,
    ]
