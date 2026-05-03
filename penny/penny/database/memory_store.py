"""Memory access layer — collections and logs, unified.

A *memory* is Penny's data primitive: a named, typed container of entries.
Two shapes share one schema:

  * collection — keyed set with similarity-based dedup on write
  * log        — append-only, keyless time-stream

Both live in a single `memory_entry` table with `key` nullable for logs.
Entries are immutable once written — `update` replaces whole content for a
given key.

Dedup on collection writes evaluates three signals against each existing entry
(thresholds live in ``PennyConstants``):

  1. ``tcr(candidate.key, existing.key)`` — token-containment ratio, lexical
  2. ``cos(candidate.key_embedding, existing.key_embedding)`` — paraphrase
  3. ``cos(candidate.content_embedding, existing.content_embedding)``

A candidate is a duplicate if ANY signal meets its strict threshold, OR if any
TWO signals meet their relaxed thresholds. Signals are skipped when either
side is missing (no key on a log entry, no embedding when no model configured),
so the rule degrades gracefully to "only what's comparable fires."
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel
from similarity.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    normalize_unicode,
    serialize_embedding,
    token_containment_ratio,
)
from sqlalchemy import func
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.database.models import Memory, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryType(StrEnum):
    COLLECTION = "collection"
    LOG = "log"


class RecallMode(StrEnum):
    OFF = "off"
    RECENT = "recent"
    RELEVANT = "relevant"
    ALL = "all"


class MemoryTypeError(Exception):
    """Raised when an operation is called against the wrong memory type."""


class MemoryNotFoundError(Exception):
    """Raised when an operation targets a memory that doesn't exist."""


class MemoryAlreadyExistsError(Exception):
    """Raised when a collection or log with the given name already exists."""


class DedupThresholds(BaseModel):
    """Per-signal strict + relaxed thresholds for the memory dedup rule."""

    key_tcr_strict: float
    key_tcr_relaxed: float
    key_sim_strict: float
    key_sim_relaxed: float
    content_sim_strict: float
    content_sim_relaxed: float

    @classmethod
    def from_runtime(cls, runtime: RuntimeParams) -> DedupThresholds:
        """Read the six dedup thresholds from runtime config."""
        return cls(
            key_tcr_strict=runtime.MEMORY_DEDUP_KEY_TCR_STRICT,
            key_tcr_relaxed=runtime.MEMORY_DEDUP_KEY_TCR_RELAXED,
            key_sim_strict=runtime.MEMORY_DEDUP_KEY_SIM_STRICT,
            key_sim_relaxed=runtime.MEMORY_DEDUP_KEY_SIM_RELAXED,
            content_sim_strict=runtime.MEMORY_DEDUP_CONTENT_SIM_STRICT,
            content_sim_relaxed=runtime.MEMORY_DEDUP_CONTENT_SIM_RELAXED,
        )


class EntryInput(BaseModel):
    """Input row for collection_write — key, content, and optional embeddings."""

    key: str
    content: str
    key_embedding: list[float] | None = None
    content_embedding: list[float] | None = None


class LogEntryInput(BaseModel):
    """Input row for log append — keyless content plus optional embedding."""

    content: str
    content_embedding: list[float] | None = None


WriteOutcome = Literal["written", "duplicate", "rejected"]


class WriteResult(BaseModel):
    key: str
    outcome: WriteOutcome
    entry_id: int | None = None
    # Existing entry's key when ``outcome == "duplicate"`` — surfaces in
    # the rejection message so the model can pivot to ``update_entry``
    # when it has fresher info for the existing row.
    matched_key: str | None = None
    # Human-readable reason when ``outcome == "rejected"``.
    reason: str | None = None


MoveOutcome = Literal["ok", "not_found", "collision"]
UpdateOutcome = Literal["ok", "not_found"]


class EntrySide(NamedTuple):
    """One side of a dedup pair: the key plus its key/content embeddings."""

    key: str | None
    key_vec: list[float] | None
    content_vec: list[float] | None


def _slug(name: str) -> str:
    """Normalize a memory name: unicode dash variants → ASCII hyphen, lowercase."""
    return normalize_unicode(name).lower()


class MemoryStore:
    """CRUD for memories (collections, logs) and their entries.

    Summary of the public surface:
        * metadata: create_collection, create_log, get, list_all, archive, unarchive
        * collection writes: write, update, move, delete
        * log writes: append
        * reads: get_entry, read_latest, read_recent, read_since, read_random,
          read_similar, read_all, keys
        * introspection: exists

    Similarity operations require the caller to pass pre-computed embeddings;
    this layer stays sync. The tool layer (Stage 2) owns async embedding.
    """

    def __init__(self, engine, runtime: RuntimeParams | None = None):
        self.engine = engine
        # Runtime accessor for /config-tunable dedup thresholds.  Tests
        # pass nothing and get a vanilla ``RuntimeParams()`` that falls
        # back to ``ConfigParam.default`` values; production wires the
        # live runtime so dedup respects /config tweaks.
        self._runtime = runtime if runtime is not None else RuntimeParams()
        # Optional change callback — fired after any write/update/delete or
        # metadata mutation so external observers (e.g. the browser channel)
        # can broadcast a refresh.  Argument is the affected memory name, or
        # ``None`` for fan-outs that aren't scoped to one memory.
        self._on_memory_changed: Callable[[str | None], None] | None = None

    def _default_thresholds(self) -> DedupThresholds:
        """Resolve dedup thresholds from the wired runtime accessor."""
        return DedupThresholds.from_runtime(self._runtime)

    def _session(self) -> Session:
        return Session(self.engine)

    # ── Metadata ────────────────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        description: str,
        recall: RecallMode,
        archived: bool = False,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
    ) -> Memory:
        return self._create_memory(
            name,
            MemoryType.COLLECTION,
            description,
            recall,
            archived,
            extraction_prompt=extraction_prompt,
            collector_interval_seconds=collector_interval_seconds,
        )

    def create_log(
        self, name: str, description: str, recall: RecallMode, archived: bool = False
    ) -> Memory:
        # Logs are inputs, not curated outputs — no extraction_prompt by design.
        return self._create_memory(name, MemoryType.LOG, description, recall, archived)

    def _create_memory(
        self,
        name: str,
        type_: MemoryType,
        description: str,
        recall: RecallMode,
        archived: bool,
        *,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
    ) -> Memory:
        name = _slug(name)
        if self.get(name) is not None:
            raise MemoryAlreadyExistsError(name)
        with self._session() as session:
            memory = Memory(
                name=name,
                type=type_.value,
                description=description,
                recall=recall.value,
                archived=archived,
                extraction_prompt=extraction_prompt,
                collector_interval_seconds=collector_interval_seconds,
                created_at=datetime.now(UTC),
            )
            session.add(memory)
            session.commit()
            session.refresh(memory)
            logger.debug("Created %s memory %s", type_.value, name)
        self._notify_changed(name)
        return memory

    def get(self, name: str) -> Memory | None:
        with self._session() as session:
            return session.get(Memory, _slug(name))

    def list_all(self) -> list[Memory]:
        with self._session() as session:
            return list(session.exec(select(Memory).order_by(Memory.name)).all())

    def entry_counts(self) -> dict[str, int]:
        """Return ``{memory_name: entry_count}`` for every memory in one query.

        Used by the addon's Memories tab to show counts in the list view
        without N+1 round-trips.  Memories with zero entries are absent —
        callers should default to 0 when looking up a missing key.
        """
        with self._session() as session:
            rows = session.exec(
                select(MemoryEntry.memory_name, func.count(MemoryEntry.id)).group_by(  # ty: ignore[invalid-argument-type]
                    MemoryEntry.memory_name
                )  # ty: ignore[invalid-argument-type]
            ).all()
        return dict(rows)

    def _notify_changed(self, name: str | None) -> None:
        """Fire the change callback if registered.  Safe to call without a hook."""
        if self._on_memory_changed is not None:
            self._on_memory_changed(name)

    def archive(self, name: str) -> None:
        self._set_archived(name, True)

    def unarchive(self, name: str) -> None:
        self._set_archived(name, False)

    def _set_archived(self, name: str, archived: bool) -> None:
        name = _slug(name)
        with self._session() as session:
            memory = session.get(Memory, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.archived = archived
            memory.updated_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
        self._notify_changed(name)

    def update_collection_metadata(
        self,
        name: str,
        *,
        description: str | None = None,
        recall: RecallMode | None = None,
        extraction_prompt: str | None = None,
        collector_interval_seconds: int | None = None,
    ) -> Memory:
        """Update fields on an existing collection.  Only set fields are applied."""
        name = _slug(name)
        self._require_type(name, MemoryType.COLLECTION)
        with self._session() as session:
            memory = session.get(Memory, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            if description is not None:
                memory.description = description
            if recall is not None:
                memory.recall = recall.value
            if extraction_prompt is not None:
                memory.extraction_prompt = extraction_prompt
            if collector_interval_seconds is not None:
                memory.collector_interval_seconds = collector_interval_seconds
            memory.updated_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
            session.refresh(memory)
        self._notify_changed(name)
        return memory

    def mark_collected(self, name: str) -> None:
        """Stamp ``last_collected_at = now`` after a dispatcher cycle.

        Called whether the collector did real work or just exited via
        ``done()`` — what matters for cadence is that the *check* happened.
        """
        name = _slug(name)
        with self._session() as session:
            memory = session.get(Memory, name)
            if memory is None:
                raise MemoryNotFoundError(name)
            memory.last_collected_at = datetime.now(UTC)
            session.add(memory)
            session.commit()
        self._notify_changed(name)

    # ── Collection writes ───────────────────────────────────────────────────

    def write(
        self,
        name: str,
        entries: list[EntryInput],
        author: str,
        thresholds: DedupThresholds | None = None,
    ) -> list[WriteResult]:
        """Write entries to a collection with per-entry dedup.

        Returns one WriteResult per input entry with its outcome. Dedup is
        evaluated against existing entries in the same memory using the
        configured thresholds (or the module defaults).
        """
        name = _slug(name)
        self._require_type(name, MemoryType.COLLECTION)
        thresholds = thresholds or self._default_thresholds()
        existing = self._load_entries_with_vectors(name)
        results: list[WriteResult] = []
        with self._session() as session:
            for entry in entries:
                results.append(self._write_one(session, name, entry, author, existing, thresholds))
            session.commit()
        if any(r.outcome == "written" for r in results):
            self._notify_changed(name)
        return results

    def _write_one(
        self,
        session: Session,
        name: str,
        entry: EntryInput,
        author: str,
        existing: list[EntrySide],
        thresholds: DedupThresholds,
    ) -> WriteResult:
        rejection_reason = _degenerate_reason(entry.content)
        if rejection_reason is not None:
            logger.debug("Rejected degenerate collection entry %r: %s", entry.key, rejection_reason)
            return WriteResult(key=entry.key, outcome="rejected", reason=rejection_reason)
        candidate = EntrySide(entry.key, entry.key_embedding, entry.content_embedding)
        matched = self._is_duplicate(candidate, existing, thresholds)
        if matched is not None:
            return WriteResult(
                key=entry.key,
                outcome="duplicate",
                matched_key=matched.key,
            )
        row = MemoryEntry(
            memory_name=name,
            key=entry.key,
            content=entry.content,
            author=author,
            key_embedding=_maybe_serialize(entry.key_embedding),
            content_embedding=_maybe_serialize(entry.content_embedding),
            created_at=datetime.now(UTC),
        )
        session.add(row)
        session.flush()
        existing.append(candidate)
        return WriteResult(key=entry.key, outcome="written", entry_id=row.id)

    def update(self, name: str, key: str, content: str, author: str) -> UpdateOutcome:
        """Replace the content of every entry with `key` in a collection.

        Most collections have a single entry per key (dedup keeps it that way),
        but the method operates on all matching rows for safety.
        """
        name = _slug(name)
        self._require_type(name, MemoryType.COLLECTION)
        with self._session() as session:
            rows = self._entries_by_key(session, name, key)
            if not rows:
                return "not_found"
            for row in rows:
                row.content = content
                row.author = author
                session.add(row)
            session.commit()
        self._notify_changed(name)
        return "ok"

    def move(self, key: str, from_name: str, to_name: str, author: str) -> MoveOutcome:
        """Move every entry with `key` from one collection to another.

        Returns "collision" if a target-collection entry with the same key
        already exists (the caller resolves the collision).
        """
        from_name = _slug(from_name)
        to_name = _slug(to_name)
        self._require_type(from_name, MemoryType.COLLECTION)
        self._require_type(to_name, MemoryType.COLLECTION)
        with self._session() as session:
            src_rows = self._entries_by_key(session, from_name, key)
            if not src_rows:
                return "not_found"
            if self._entries_by_key(session, to_name, key):
                return "collision"
            for row in src_rows:
                row.memory_name = to_name
                row.author = author
                session.add(row)
            session.commit()
        self._notify_changed(from_name)
        self._notify_changed(to_name)
        return "ok"

    def delete(self, name: str, key: str) -> int:
        """Delete every entry with `key` in a collection. Returns rows removed."""
        name = _slug(name)
        self._require_type(name, MemoryType.COLLECTION)
        with self._session() as session:
            rows = self._entries_by_key(session, name, key)
            for row in rows:
                session.delete(row)
            session.commit()
        if rows:
            self._notify_changed(name)
        return len(rows)

    # ── Log writes ──────────────────────────────────────────────────────────

    def append(self, name: str, entries: list[LogEntryInput], author: str) -> list[MemoryEntry]:
        """Append one or more entries to a log memory. No dedup; keyless."""
        name = _slug(name)
        self._require_type(name, MemoryType.LOG)
        created: list[MemoryEntry] = []
        with self._session() as session:
            for entry in entries:
                row = MemoryEntry(
                    memory_name=name,
                    key=None,
                    content=entry.content,
                    author=author,
                    key_embedding=None,
                    content_embedding=_maybe_serialize(entry.content_embedding),
                    created_at=datetime.now(UTC),
                )
                session.add(row)
                created.append(row)
            session.commit()
            for row in created:
                session.refresh(row)
        if created:
            self._notify_changed(name)
        return created

    # ── Reads ───────────────────────────────────────────────────────────────

    def get_entry(self, name: str, key: str) -> list[MemoryEntry]:
        with self._session() as session:
            return self._entries_by_key(session, _slug(name), key)

    def read_latest(self, name: str, k: int | None = None) -> list[MemoryEntry]:
        """Return entries newest-first. With `k=None`, returns every entry."""
        name = _slug(name)
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == name)
                .order_by(MemoryEntry.created_at.desc())  # type: ignore[union-attr]
            )
            if k is not None:
                query = query.limit(k)
            return list(session.exec(query).all())

    def read_latest_matching(
        self, name: str, content_prefix: str, k: int | None = None
    ) -> list[MemoryEntry]:
        """Newest-first entries whose ``content`` starts with ``content_prefix``.

        Used to scope a log read to one tag — e.g. ``collector-runs``
        entries for a specific target collection (the content format is
        ``[<target>] <marker> <summary>``).
        """
        name = _slug(name)
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(
                    MemoryEntry.memory_name == name,
                    MemoryEntry.content.like(f"{content_prefix}%"),  # ty: ignore[unresolved-attribute]
                )
                .order_by(MemoryEntry.created_at.desc())  # type: ignore[union-attr]
            )
            if k is not None:
                query = query.limit(k)
            return list(session.exec(query).all())

    def read_recent(
        self, name: str, window_seconds: int, cap: int | None = None
    ) -> list[MemoryEntry]:
        cutoff = datetime.now(UTC).timestamp() - window_seconds
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=UTC)
        return self.read_since(_slug(name), cutoff_dt, cap)

    def read_since(self, name: str, cursor: datetime, cap: int | None = None) -> list[MemoryEntry]:
        name = _slug(name)
        with self._session() as session:
            query = (
                select(MemoryEntry)
                .where(MemoryEntry.memory_name == name, MemoryEntry.created_at > cursor)
                .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
            )
            if cap is not None:
                query = query.limit(cap)
            return list(session.exec(query).all())

    def read_random(self, name: str, k: int | None = None) -> list[MemoryEntry]:
        """Return `k` entries sampled uniformly at random. `k=None` returns all."""
        name = _slug(name)
        with self._session() as session:
            rows = list(
                session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == name)).all()
            )
        if k is None or len(rows) <= k:
            return rows
        return random.sample(rows, k)

    def read_similar(
        self,
        name: str,
        anchor: list[float],
        k: int | None = None,
        floor: float = 0.0,
    ) -> list[MemoryEntry]:
        """Return entries similar to ``anchor`` ordered by cosine.

        After scoring + sort, an adaptive cluster-strength gate suppresses
        flat noise plateaus entirely; the cutoff combines a relative band
        against the cluster center with an absolute floor.

        Entries without a ``content_embedding`` are skipped.  ``floor`` is
        merged with the configured ``MEMORY_RELEVANT_ABSOLUTE_FLOOR`` —
        cutoff is the larger of the two.  With ``k=None`` every qualifying
        entry is returned.
        """
        name = _slug(name)
        rows = self._embedded_rows(name)
        scored = self._score(rows, [anchor])
        cutoff = _adaptive_cutoff(scored, floor)
        if cutoff is None:
            return []
        ordered = [row for score, row in scored if score >= cutoff]
        return ordered if k is None else ordered[:k]

    def read_similar_hybrid(
        self,
        name: str,
        conversation_anchors: list[list[float]],
        k: int | None = None,
        floor: float = 0.0,
        exclude_contents: set[str] | None = None,
    ) -> list[MemoryEntry]:
        """Hybrid relevance over a conversation window.

        ``conversation_anchors`` is ordered oldest→newest; the last element
        is the *current* message, the rest are prior turns providing topic
        context.  Each candidate is scored as

            max(weighted_decay_over_history, cosine_to_current)

        so a strong direct hit on the current message stands alone, while
        vague follow-ups still benefit from conversation drift.  The
        adaptive cutoff is applied identically to ``read_similar``.

        Single-element ``conversation_anchors`` reduces cleanly to the same
        ranking as ``read_similar`` — the weighted-decay branch is just the
        lone cosine, so ``max()`` picks it.

        ``exclude_contents`` (optional) drops corpus rows whose content
        matches any string in the set.  Use this to keep the conversation
        anchors themselves out of their own retrieval: the chat path
        writes the user's incoming message to ``user-messages`` before
        recall runs, and any conversation-history message similarly lives
        in the corpus, so without exclusion they'd self-match their own
        anchor at cosine ≈ 1.0 and dominate the hit list.

        Low-information rows (fewer than ``MEMORY_RELEVANT_MIN_WORDS``
        word tokens) are filtered out of the corpus before scoring **for
        log-shaped memories only**.  Empty strings, lone punctuation,
        and stock greetings otherwise dominate cosine ranking on short
        keyword queries.  Collections have keyed entries where short
        content is deliberate (``"anime"``, ``"cyberpunk"`` are valid
        like preferences); filtering them would wipe out 75%+ of the
        user's actual stated topics.
        """
        name = _slug(name)
        if not conversation_anchors:
            return []
        rows = self._embedded_rows(name)
        if exclude_contents:
            rows = [r for r in rows if r.content not in exclude_contents]
        if self._memory_type(name) == MemoryType.LOG.value:
            rows = [r for r in rows if not _is_low_info(r.content)]
        scored = self._score(rows, conversation_anchors)
        cutoff = _adaptive_cutoff(scored, floor)
        if cutoff is None:
            return []
        ordered = [row for score, row in scored if score >= cutoff]
        return ordered if k is None else ordered[:k]

    def _embedded_rows(self, name: str) -> list[MemoryEntry]:
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry).where(
                        MemoryEntry.memory_name == name,
                        MemoryEntry.content_embedding.is_not(None),  # type: ignore[union-attr]
                    )
                ).all()
            )

    def _score(
        self,
        rows: list[MemoryEntry],
        anchors: list[list[float]],
    ) -> list[tuple[float, MemoryEntry]]:
        """Score each candidate as ``max(weighted_decay, current_cos) - α·proxy``.

        Vectorized: stacks all candidate embeddings into an (N, D) matrix and
        all anchors into an (M, D) matrix, then a single matmul produces the
        full (N, M) cosine table.  The centrality-magnet penalty is the dot
        of each row with the corpus centroid (one mean + one matvec); in
        normalized space this is rank-equivalent to mean cosine to every
        other entry, which keeps generic boilerplate from leaking into
        unrelated queries.  Per-query work stays O(N·D); no precompute, no
        cache.

        Single-anchor reduces cleanly: with M=1 the weighted-decay branch is
        the lone cosine, so ``max()`` picks it.
        """
        valid: list[tuple[bytes, MemoryEntry]] = [
            (row.content_embedding, row)
            for row in rows
            if row.id is not None and row.content_embedding is not None
        ]
        if not valid or not anchors:
            return []
        valid_rows = [row for _, row in valid]
        matrix = _stack_normalized(blob for blob, _ in valid)
        anchor_matrix = _stack_normalized_anchors(anchors)
        cos_matrix = matrix @ anchor_matrix.T  # (N, M)
        hybrid = _hybrid_scores(cos_matrix)
        adjusted = hybrid - (
            PennyConstants.MEMORY_RELEVANT_CENTRALITY_PENALTY * _centrality_via_centroid(matrix)
        )
        order = np.argsort(-adjusted)
        return [(float(adjusted[i]), valid_rows[i]) for i in order]

    def expand_with_temporal_neighbors(
        self,
        name: str,
        hits: list[MemoryEntry],
        window_minutes: int,
    ) -> list[MemoryEntry]:
        """Expand each hit by ±``window_minutes`` of surrounding entries.

        Used by relevant-mode recall on log-shaped memories: similarity
        finds the conversational anchor; the temporal window pulls in the
        follow-ups that share no entity overlap with the current message
        but live in the same conversation as a real hit.

        Returns the union of all in-window entries deduplicated by id and
        ordered chronologically (oldest→newest).
        """
        name = _slug(name)
        if not hits:
            return []
        delta = timedelta(minutes=window_minutes)
        seen_ids: set[int] = set()
        expanded: list[MemoryEntry] = []
        with self._session() as session:
            for hit in hits:
                start = hit.created_at - delta
                end = hit.created_at + delta
                rows = session.exec(
                    select(MemoryEntry)
                    .where(
                        MemoryEntry.memory_name == name,
                        MemoryEntry.created_at >= start,
                        MemoryEntry.created_at <= end,
                    )
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
                for row in rows:
                    if row.id is not None and row.id not in seen_ids:
                        seen_ids.add(row.id)
                        expanded.append(row)
        expanded.sort(key=lambda r: r.created_at)
        return expanded

    def read_all(self, name: str) -> list[MemoryEntry]:
        name = _slug(name)
        with self._session() as session:
            return list(
                session.exec(
                    select(MemoryEntry)
                    .where(MemoryEntry.memory_name == name)
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )

    def keys(self, name: str) -> list[str]:
        name = _slug(name)
        with self._session() as session:
            rows = list(
                session.exec(
                    select(MemoryEntry.key)
                    .where(
                        MemoryEntry.memory_name == name,
                        MemoryEntry.key.is_not(None),  # type: ignore[union-attr]
                    )
                    .order_by(MemoryEntry.created_at.asc())  # type: ignore[union-attr]
                ).all()
            )
        seen: set[str] = set()
        ordered: list[str] = []
        for key in rows:
            if key is None or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    # ── Introspection ───────────────────────────────────────────────────────

    def exists(
        self,
        names: list[str],
        key: str | None,
        key_embedding: list[float] | None,
        content_embedding: list[float] | None,
        thresholds: DedupThresholds | None = None,
    ) -> bool:
        """Check whether an equivalent entry already exists in any of the named memories.

        Runs the same similarity-based dedup used by `write`, plus an exact
        key-match shortcut when a key is supplied. Returns True on the first hit.
        """
        names = [_slug(n) for n in names]
        thresholds = thresholds or self._default_thresholds()
        candidate = EntrySide(key, key_embedding, content_embedding)
        for name in names:
            if key is not None and self.get_entry(name, key):
                return True
            existing = self._load_entries_with_vectors(name)
            if self._is_duplicate(candidate, existing, thresholds):
                return True
        return False

    # ── Internals ───────────────────────────────────────────────────────────

    def _require_type(self, name: str, expected: MemoryType) -> None:
        memory = self.get(name)
        if memory is None:
            raise MemoryNotFoundError(name)
        if memory.type != expected.value:
            raise MemoryTypeError(f"memory '{name}' is a {memory.type}, not a {expected.value}")

    def _memory_type(self, name: str) -> str:
        """Return ``memory.type`` ("collection" or "log") or empty string."""
        memory = self.get(name)
        return memory.type if memory else ""

    def _entries_by_key(self, session: Session, name: str, key: str) -> list[MemoryEntry]:
        return list(
            session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == name, MemoryEntry.key == key)
            ).all()
        )

    def _load_entries_with_vectors(self, name: str) -> list[EntrySide]:
        """Load every entry for `name` as EntrySide triples (key, key_vec, content_vec).

        Entries without a given embedding or key contribute None on that axis.
        """
        with self._session() as session:
            rows = list(
                session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == name)).all()
            )
        return [
            EntrySide(
                r.key,
                _maybe_deserialize(r.key_embedding),
                _maybe_deserialize(r.content_embedding),
            )
            for r in rows
        ]

    def _is_duplicate(
        self,
        candidate: EntrySide,
        existing: list[EntrySide],
        thresholds: DedupThresholds,
    ) -> EntrySide | None:
        """Return the first existing entry that ``candidate`` collides with
        under the dedup rule, or ``None`` if no match.  Returning the
        matched side (instead of bool) lets callers surface *which*
        existing entry blocked the write — the rejection message can
        then say ``"matches existing 'Kepler Museum'"`` so the model
        can pivot to ``update_entry`` when it has fresher info.

        Truthy/falsy in a bool context is preserved (``None`` is falsy,
        ``EntrySide`` is truthy), so ``if self._is_duplicate(...)``
        callsites continue to work."""
        for side in existing:
            if self._pair_is_duplicate(candidate, side, thresholds):
                return side
        return None

    def _pair_is_duplicate(
        self,
        candidate: EntrySide,
        existing: EntrySide,
        thresholds: DedupThresholds,
    ) -> bool:
        """Apply the three-signal dedup rule to a single candidate/existing pair.

        Signals that can't be computed (missing keys, missing embeddings) are
        skipped. Fire if any one signal hits its strict threshold or any two
        signals hit their relaxed thresholds.
        """
        signals = _score_signals(candidate, existing, thresholds)
        if any(score >= strict for score, strict, _ in signals):
            return True
        relaxed_hits = sum(1 for score, _, relaxed in signals if score >= relaxed)
        return relaxed_hits >= 2


def _score_signals(
    candidate: EntrySide,
    existing: EntrySide,
    thresholds: DedupThresholds,
) -> list[tuple[float, float, float]]:
    """Return (score, strict_threshold, relaxed_threshold) for every applicable signal."""
    out: list[tuple[float, float, float]] = []
    if candidate.key is not None and existing.key is not None:
        out.append(
            (
                token_containment_ratio(candidate.key, existing.key),
                thresholds.key_tcr_strict,
                thresholds.key_tcr_relaxed,
            )
        )
    key_cos = _safe_cosine(candidate.key_vec, existing.key_vec)
    if key_cos is not None:
        out.append((key_cos, thresholds.key_sim_strict, thresholds.key_sim_relaxed))
    content_cos = _safe_cosine(candidate.content_vec, existing.content_vec)
    if content_cos is not None:
        out.append((content_cos, thresholds.content_sim_strict, thresholds.content_sim_relaxed))
    return out


_WORD_TOKEN_RE = re.compile(r"\w+")

# Matches content that is a bare URL with no surrounding description.
_BARE_URL_RE = re.compile(r"^https?://\S+$")

# LLM bail-out phrases that produce useless knowledge entries.
_WRITE_BAILOUT_PHRASES: frozenset[str] = frozenset(
    {
        "not sure",
        "i'm not sure",
        "i am not sure",
        "i cannot help with that",
        "i can't help with that",
        "i don't know",
        "i do not know",
        "n/a",
        "no information",
        "no information available",
        "unable to summarize",
        "unable to provide a summary",
        "no content available",
        "content not available",
        "page not available",
        "content unavailable",
        "access denied",
        "error",
    }
)


def _degenerate_reason(content: str) -> str | None:
    """Return a rejection reason if ``content`` is too degenerate to store.

    Catches empty/pure-punctuation strings, bare URLs, and known LLM
    bail-out phrases.  Returns ``None`` when content is acceptable.
    Applied at collection write time to keep the corpus clean.
    """
    stripped = content.strip()
    if not _WORD_TOKEN_RE.findall(stripped):
        return "content has no word tokens (empty, punctuation, or ellipsis only)"
    if _BARE_URL_RE.match(stripped):
        return "content is a bare URL with no descriptive text"
    if stripped.lower() in _WRITE_BAILOUT_PHRASES:
        return f"content matches a known LLM bail-out phrase: {stripped!r}"
    return None


def _is_low_info(content: str) -> bool:
    """Return True if ``content`` carries less than the configured minimum
    word count and should be filtered from similarity scoring.

    The filter targets entries that geometrically dominate cosine
    rankings on short keyword anchors despite having no topical
    payload — empty strings, lone punctuation, stock greetings, bare
    URL fragments.  Entries that pass the filter still appear in
    other recall paths (recent / all / read_latest tool calls);
    only the relevant-mode similarity corpus is filtered.
    """
    return len(_WORD_TOKEN_RE.findall(content)) < PennyConstants.MEMORY_RELEVANT_MIN_WORDS


def _maybe_serialize(vec: list[float] | None) -> bytes | None:
    return serialize_embedding(vec) if vec is not None else None


def _maybe_deserialize(blob: bytes | None) -> list[float] | None:
    return deserialize_embedding(blob) if blob is not None else None


def _safe_cosine(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return cosine_similarity(a, b)


def _stack_normalized(blobs: Iterable[bytes]) -> np.ndarray:
    """Stack serialized embeddings into an L2-normalized (N, D) float32 matrix.

    Uses ``np.frombuffer`` so each blob materializes via a zero-copy view
    that's then assigned into the matrix — ~1 ms for 1500×768 in practice.
    """
    blob_list = list(blobs)
    if not blob_list:
        return np.zeros((0, 0), dtype=np.float32)
    dim = len(blob_list[0]) // 4
    matrix = np.empty((len(blob_list), dim), dtype=np.float32)
    for index, blob in enumerate(blob_list):
        matrix[index] = np.frombuffer(blob, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _stack_normalized_anchors(anchors: list[list[float]]) -> np.ndarray:
    """Stack anchor vectors into an L2-normalized (M, D) float32 matrix."""
    matrix = np.asarray(anchors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _hybrid_scores(cos_matrix: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Combine a (N, M) cosine matrix into a per-row hybrid score.

    Hybrid = max(weighted_decay_over_history, cosine_to_current).  Anchors
    are oldest→newest, so weights go ``decay**(M-1) … decay**0`` and the
    last column is the current message.  With M=1 the weighted branch
    equals the current branch and ``maximum`` returns that single cosine.
    """
    anchor_count = cos_matrix.shape[1]
    weights = np.array(
        [decay ** (anchor_count - 1 - i) for i in range(anchor_count)],
        dtype=np.float32,
    )
    weighted = (cos_matrix * weights).sum(axis=1) / weights.sum()
    current = cos_matrix[:, -1]
    return np.maximum(weighted, current)


def _centrality_via_centroid(matrix: np.ndarray) -> np.ndarray:
    """Per-row mean cosine to all OTHER rows, computed via the corpus centroid.

    Algebraically identical to the O(N²) loop ``mean_{j≠i}(cos(v_i, v_j))``:

        mean_{j≠i}(cos) = (N · v_i · centroid − 1) / (N − 1)

    where ``centroid = matrix.mean(axis=0)`` and rows are L2-normalized so
    ``v_i · v_i = 1``.  Cost is one ``mean`` and one matrix-vector product —
    O(N · D) per query, no precompute, no cache.

    Returns zeros for corpora of fewer than 2 rows (no neighbors to average).
    """
    n = matrix.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    centroid = matrix.mean(axis=0)
    return (n * (matrix @ centroid) - 1) / (n - 1)


def _adaptive_cutoff(scored: list[tuple[float, MemoryEntry]], floor: float) -> float | None:
    """Adaptive cutoff for similarity-ranked retrieval.

    With at least ``GATE_SAMPLE_SIZE`` candidates, applies a cluster-strength
    gate: if the head-mean / sample-mean ratio falls below
    ``CLUSTER_GATE``, returns ``None`` to suppress the result entirely
    (flat noise plateau, no real cluster).  Otherwise the cutoff combines
    a relative band against the cluster center with the absolute floor.

    Below the cold-start sample-size threshold, the gate is skipped and the
    larger of the configured absolute floor and the caller's ``floor`` is
    used directly.
    """
    if not scored:
        return None
    head_size = PennyConstants.MEMORY_RELEVANT_GATE_HEAD_SIZE
    sample_size = PennyConstants.MEMORY_RELEVANT_GATE_SAMPLE_SIZE
    absolute_floor = max(floor, PennyConstants.MEMORY_RELEVANT_ABSOLUTE_FLOOR)
    if len(scored) >= sample_size:
        head_mean = sum(score for score, _ in scored[:head_size]) / head_size
        sample_mean = sum(score for score, _ in scored[:sample_size]) / sample_size
        if (
            sample_mean <= 0
            or head_mean / sample_mean < PennyConstants.MEMORY_RELEVANT_CLUSTER_GATE
        ):
            return None
        return max(
            head_mean * PennyConstants.MEMORY_RELEVANT_RELATIVE_RATIO,
            absolute_floor,
        )
    return absolute_floor
