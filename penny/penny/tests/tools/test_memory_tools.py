"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, RecallMode, WriteResult
from penny.database.models import MemoryEntry
from penny.llm.client import LlmClient
from penny.llm.models import LlmConnectionError
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionDeleteEntryTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMergeTool,
    CollectionReadLatestTool,
    CollectionReadRandomTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    CollectorRunHistoryTool,
    DoneTool,
    ExistsTool,
    LogAppendTool,
    LogCreateTool,
    LogReadTool,
    ReadPublishedLatestTool,
    ReadRunCallsTool,
    ReadSimilarTool,
    TestExtractionPromptTool,
    UpdateEntryTool,
    _format_duplicate,
    build_memory_tools,
)


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — no migrations.

    Migration 0026 seeds three system log memories; these tool tests
    exercise the tool surface in isolation and declare exactly the
    memories they need.
    """
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _make_llm_client(mock_llm) -> LlmClient:
    """Build an LlmClient whose default embed handler returns distinct vectors
    per input text, so identical inputs collide and distinct inputs don't."""
    mock_llm.set_embed_handler(_hash_embed)
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _hash_embed(model: str, text: str | list[str]) -> list[list[float]]:
    """Deterministic embedding: text → unit vector where one axis is 1.0.

    Identical strings map to identical vectors; distinct strings map to
    different axes (cosine = 0), so dedup and similarity behave sensibly in
    tests without depending on a real embedding model.
    """
    inputs = text if isinstance(text, list) else [text]
    return [_single_hash_vec(t) for t in inputs]


def _single_hash_vec(text: str, dim: int = 4096) -> list[float]:
    """Bag-of-words deterministic embedding.  Each word picks an axis via
    SHA-256 → modulo ``dim``; the vector is L2-normalised so cosine is
    comparable across strings.  Identical strings map to identical
    vectors; strings sharing words have meaningful cosine > 0;
    fully-distinct strings map to cosine = 0."""
    vec = [0.0] * dim
    words = text.lower().split() or [text]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:8], "big") % dim
        vec[axis] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class _FailingEmbedClient:
    """An embedding client whose every embed call fails transiently.

    ``embed_text`` catches ``LlmError`` and returns ``None``, so the write path
    hits the fail-hard branch and refuses to persist a vectorless entry (#1412).
    """

    model = "test-model"

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        raise LlmConnectionError("embedding backend unavailable")


class _KeyOnlyFailingEmbedClient:
    """Fails only the embed of a specific string (the key), returning a real
    vector for everything else (the content).

    ``CollectionWriteTool._build_entry`` embeds key then content in two calls, so
    this reproduces a transient miss that lands on the key alone — the entry would
    be stored missing its key vector, which the write path refuses (an entry must
    carry all its vectors).  The startup backfill is the safety net for a
    key-null row that reaches the corpus by another path (migration-seeded content,
    #1468), but the write path still refuses at write time rather than persist a
    vectorless, recall-invisible row that only a later restart would repair.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        self.model = "test-model"

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        inputs = text if isinstance(text, list) else [text]
        if self._key in inputs:
            raise LlmConnectionError("key embed failed")
        return [_single_hash_vec(t) for t in inputs]


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_collection_persists(self, tmp_path):
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="positive prefs",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt=(
                "Extract user likes from user-messages log and write to likes collection."
            ),
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
            published=True,
        )
        # Structured echo: collection name, interval, recall, prompt body all surfaced
        # so the chat agent can confirm-back without confabulating.
        assert "Created collection 'likes'" in result.message
        assert "interval: 3600s (1h)" in result.message
        assert "recall: relevant" in result.message
        # published surfaces in the echo so the chat agent confirms notify-on-new back.
        assert "published: True" in result.message
        assert "Extract user likes" in result.message  # extraction_prompt is echoed verbatim
        # Intent captured at creation is persisted and echoed back so the user
        # can correct it now — the only time it's settable.
        assert "intent: a running list the user asked me to keep" in result.message
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["likes"].type == "collection"
        assert memories["likes"].recall == "relevant"
        assert memories["likes"].description == "positive prefs"
        assert memories["likes"].collector_interval_seconds == 3600
        assert memories["likes"].intent == "a running list the user asked me to keep"
        # The notify-on-new flag persists; omitting it defaults to silent (False).
        assert memories["likes"].published is True

    @pytest.mark.asyncio
    async def test_create_log_persists(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="user-messages", description="inbound", inclusion="always", recall="recent"
        )
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["user-messages"].type == "log"
        assert memories["user-messages"].recall == "recent"

    @pytest.mark.asyncio
    async def test_create_collection_duplicate_returns_user_friendly_message(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="ai-news",
            description="first",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="ai-news",
            description="second slightly different",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert "already exists" in result.message
        assert "ai-news" in result.message
        # Original collection is unchanged
        memory = db.memories.get("ai-news")
        assert memory is not None
        assert memory.description == "first"

    @pytest.mark.asyncio
    async def test_create_log_duplicate_returns_user_friendly_message(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="first", inclusion="always", recall="recent"
        )
        result = await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="second", inclusion="never", recall="recent"
        )
        assert "already exists" in result.message
        assert "events" in result.message

    @pytest.mark.asyncio
    async def test_create_rejects_short_extraction_prompt(self, tmp_path):
        # The length rule now lives on CollectionCreateArgs (the args_model), so
        # the rejection is produced by the pre-execute Tool.run gate — an
        # actionable validation envelope naming the field and the fix.
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="yes",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert result.success is False
        assert "extraction_prompt" in result.message
        assert "too short" in result.message
        assert "minimum" in result.message
        assert db.memories.get("notes") is None  # collection not created

    @pytest.mark.asyncio
    async def test_create_rejects_blank_description(self, tmp_path):
        # The description doubles as the stage-1 routing anchor — a blank one
        # would embed an empty string and never match, so it's refused.  The
        # blank-check lives on the args_model, so the refusal comes from the
        # pre-execute Tool.run gate.
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            description="   ",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert result.success is False
        assert "description cannot be blank" in result.message
        assert db.memories.get("notes") is None  # collection not created

    @pytest.mark.asyncio
    async def test_create_rejects_invalid_inclusion_with_valid_modes(self, tmp_path):
        # An out-of-enum inclusion is rejected at the args_model with an
        # actionable message that lists the valid modes (not a bare ValueError
        # bubbling from the store-layer Inclusion(...) cast).
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            description="real description",
            inclusion="sometimes",
            recall="recent",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert result.success is False
        assert "inclusion must be one of" in result.message
        assert "always" in result.message and "relevant" in result.message
        assert db.memories.get("notes") is None

    @pytest.mark.asyncio
    async def test_create_rejects_invalid_recall_with_valid_modes(self, tmp_path):
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            description="real description",
            inclusion="never",
            recall="sometimes",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert result.success is False
        assert "recall must be one of" in result.message
        assert "recent" in result.message and "all" in result.message
        assert db.memories.get("notes") is None

    @pytest.mark.asyncio
    async def test_create_rejects_missing_extraction_prompt(self, tmp_path):
        # extraction_prompt is required — a collection without one is passive
        # (nothing fills it), so the tool surface refuses to create one.
        db = _make_db(tmp_path)
        with pytest.raises(ValidationError):
            await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
                name="notes",
                description="x",
                inclusion="never",
                recall="recent",
                collector_interval_seconds=3600,
            )
        assert db.memories.get("notes") is None

    @pytest.mark.asyncio
    async def test_create_rejects_missing_interval(self, tmp_path):
        # collector_interval_seconds is required — a collection without one
        # has no cadence and never runs.
        db = _make_db(tmp_path)
        with pytest.raises(ValidationError):
            await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
                name="notes",
                description="x",
                inclusion="never",
                recall="recent",
                extraction_prompt="Extract things from somewhere.",
            )
        assert db.memories.get("notes") is None

    @pytest.mark.asyncio
    async def test_create_accepts_long_enough_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        prompt = "Extract likes from user-messages log and write to collection."
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt=prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert "Created" in result.message

    @pytest.mark.asyncio
    async def test_create_rejects_fictitious_tool_call(self, tmp_path):
        # A hallucinated tool named in the extraction_prompt (extract_text) is
        # rejected at the args_model, before execute — so a fictitious call can
        # never be persisted into a prompt the collector would later fail to run.
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt=(
                'Collect things.\n1. browse(["x"])\n2. extract_text(page)\n'
                '3. collection_write("notes", entries=[{key: "k", content: "c"}])\n4. done()'
            ),
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert result.success is False
        assert "extract_text" in result.message
        assert db.memories.get("notes") is None  # nothing persisted

    @pytest.mark.asyncio
    async def test_update_rejects_short_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        original_prompt = "test fixture extraction prompt"
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        # The optional extraction_prompt rule on CollectionUpdateArgs validates
        # only when present, via the pre-execute Tool.run gate.
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).run(
            name="notes", extraction_prompt="yes"
        )
        assert result.success is False
        assert "extraction_prompt" in result.message
        assert "too short" in result.message
        # Update rejected — original prompt preserved unchanged
        assert db.memories.get("notes").extraction_prompt == original_prompt

    @pytest.mark.asyncio
    async def test_update_rejects_fictitious_tool_call(self, tmp_path):
        db = _make_db(tmp_path)
        original_prompt = (
            'Collect notes.\n1. browse(["x"])\n'
            '2. collection_write("notes", entries=[{key: "k", content: "c"}])\n3. done()'
        )
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        # A rewrite that introduces a fictitious tool is rejected via the pre-execute
        # Tool.run gate, and the stored prompt is left untouched.
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            extraction_prompt=(
                'Collect notes.\n1. browse(["x"])\n2. extract_text(page)\n'
                '3. collection_write("notes", entries=[{key: "k", content: "c"}])\n4. done()'
            ),
        )
        assert result.success is False
        assert "extract_text" in result.message
        assert db.memories.get("notes").extraction_prompt == original_prompt

    @pytest.mark.asyncio
    async def test_update_treats_blank_fields_as_omitted(self, tmp_path, mock_llm):
        # Models emit "" for an optional field they mean to leave alone (gpt-oss
        # was observed passing extraction_prompt="" alongside a recall change).
        # A blank must be skipped, not written through: the recall change lands
        # while the existing prompt/description survive untouched.
        db = _make_db(tmp_path)
        original_prompt = "test fixture extraction prompt that is long enough"
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            description="real description",
            inclusion="relevant",
            recall="all",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            recall="relevant",
            extraction_prompt="",
            description="   ",
            published=True,
        )
        assert "Updated" in result.message
        updated = db.memories.get("notes")
        assert updated.recall == "relevant"  # the real change landed
        assert updated.extraction_prompt == original_prompt  # blank skipped, not blanked
        assert updated.description == "real description"  # blank skipped, not blanked
        # published flips on the update path (created silent by default → notify-on-new).
        assert updated.published is True

    @pytest.mark.asyncio
    async def test_create_surfaces_description_embed_degradation(self, tmp_path):
        """A transient description-embed failure still creates the collection, but the
        result NAMES the degraded routing anchor and leaves it NULL for the startup
        backfill to re-heal (#1468) — a visible degradation, not a silent success."""
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="notes",
            description="a running list of notes",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt="Extract notes from user-messages and write to collection.",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert "Created" in result.message
        assert result.mutated is True
        assert "transient embedding error" in result.message
        assert "self-heal" in result.message
        # Row exists; anchor left NULL for the backfill to re-heal.
        row = db.memories.get("notes")
        assert row is not None
        assert row.description_embedding is None

    @pytest.mark.asyncio
    async def test_log_create_surfaces_description_embed_degradation(self, tmp_path):
        db = _make_db(tmp_path)
        result = await LogCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="events", description="event stream", inclusion="relevant", recall="recent"
        )
        assert "Created log" in result.message
        assert "transient embedding error" in result.message
        assert db.memories.get("events").description_embedding is None

    @pytest.mark.asyncio
    async def test_update_failed_description_embed_clears_stale_anchor(self, tmp_path):
        """Changing a description whose embed fails clears the anchor to NULL — it does
        NOT leave the old, now-mismatched vector in place (a stale anchor the NULL-only
        description backfill could never detect, #1468).  The new text lands, the anchor
        is left for the backfill to re-heal, and the degradation surfaces."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            description="old subject",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert db.memories.get("notes").description_embedding is not None  # a good anchor first

        result = await CollectionUpdateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="notes", description="a completely different subject"
        )
        assert "Updated" in result.message
        assert "transient embedding error" in result.message
        row = db.memories.get("notes")
        assert row.description == "a completely different subject"  # new text landed
        assert row.description_embedding is None  # stale anchor cleared, not kept


class TestCollectionWritesAndReads:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast", "content": "loves dark roast"},
                {"key": "cold brew", "content": "enjoys cold brew"},
            ],
        )
        assert "Wrote 2 entries to 'likes'" in result.message
        assert result.mutated is True
        latest = await CollectionReadLatestTool(db).execute(memory="likes")
        assert "dark roast" in latest.message
        assert "cold brew" in latest.message
        # Each rendered entry carries an absolute UTC timestamp so the model can
        # place it in time — read-tool output was previously timeless.
        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", latest.message)

    @pytest.mark.asyncio
    async def test_write_empty_entries_is_actionable_not_bare_pydantic(self, tmp_path, mock_llm):
        """An empty ``entries`` batch gets a named, actionable rejection through the
        arg-validation envelope — not Pydantic's bare "List should have at least 1
        item" (the house wording-unification pass)."""
        db = _make_db(tmp_path)
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.run(memory="likes", entries=[])
        assert result.success is False
        assert "at least one entry" in result.message
        assert "List should have at least 1 item" not in result.message

    @pytest.mark.asyncio
    async def test_write_unknown_key_in_entry_names_nested_path(self, tmp_path, mock_llm):
        """A misspelled/extraneous key INSIDE a batch entry surfaces the nested loc
        path and suggests the valid sibling, rather than being silently dropped or
        mis-rendered as the whole ``entries`` field (#1416)."""
        db = _make_db(tmp_path)
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.run(
            memory="likes",
            entries=[{"key": "k", "content": "v", "contnt": "typo"}],
        )
        assert result.success is False
        assert "entries.0.contnt" in result.message
        assert "did you mean 'content'" in result.message

    def test_format_duplicate_binds_key_when_present_and_is_honest_when_keyless(self):
        """The keyed arm binds an ``update_entry(key=...)`` call to the matched key;
        the keyless arm (no key to update) is honest about it and points at the real
        move (skip / write distinct content) — not a dangling refresh imperative."""
        keyed = _format_duplicate(
            WriteResult(key="cold brew", outcome="duplicate", matched_key="cold brew")
        )
        assert "update_entry(key='cold brew'" in keyed
        keyless = _format_duplicate(
            WriteResult(key="cold brew", outcome="duplicate", matched_key=None)
        )
        assert "no key to update" in keyless
        assert "update_entry(" not in keyless

    @pytest.mark.asyncio
    async def test_write_reports_duplicate_via_tcr(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        result = await write.execute(
            memory="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        assert "Rejected as duplicates" in result.message
        # A fully duplicate-rejected batch wrote nothing — it must read as a
        # no-op so the collector's work/no-work split (and auto-throttle) sees
        # the truth rather than counting the rejected write as "work".
        assert result.mutated is False
        # The candidate's own key is named, the existing key it collided with is
        # named, *and* the matched key is BOUND straight into the update_entry call
        # (not a <existing key> placeholder) — so the model refreshes 'dark roast'
        # rather than re-using its own rejected 'dark roast coffee' key and
        # ping-ponging on key-not-found (#1405).
        assert "dark roast coffee" in result.message
        assert "duplicates existing 'dark roast'" in result.message
        assert "update_entry(key='dark roast', content=<richer info>)" in result.message
        # Whole batch was duplicates → the "nothing new" hint fires.  This is a
        # chat-scope write (scope=None), which has no ``done`` tool, so the hint
        # must NOT name it — chat and collector share this tool surface.
        assert "Nothing new to add this time" in result.message
        assert "done()" not in result.message

    @pytest.mark.asyncio
    async def test_write_all_duplicates_collector_scope_hints_done(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        # A collector binds its writes to one collection via ``scope``.
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test", scope="likes")
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        all_duplicates = await write.execute(
            memory="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        # Whole batch was duplicates and this is a collector, so the close names
        # ``done()`` — the model can close the cycle instead of key-hunting — while
        # the per-entry rejection still binds the matched key into update_entry.
        assert "Nothing new to add" in all_duplicates.message
        assert "done()" in all_duplicates.message
        assert "update_entry(key='dark roast', content=<richer info>)" in all_duplicates.message
        # A re-write under the SAME key binds that key straight into update_entry —
        # 'already exists' names the collision, then the exact refresh call.
        same_key = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        assert "'dark roast' already exists" in same_key.message
        assert "update_entry(key='dark roast', content=<richer info>)" in same_key.message
        # A batch with a genuinely new entry alongside a duplicate gets the
        # per-entry bound refresh + the refresh-or-skip close, never "nothing new".
        partial = await write.execute(
            memory="likes",
            entries=[
                {"key": "cold brew", "content": "a brand new distinct entry"},
                {"key": "dark roast blend", "content": "first body"},
            ],
        )
        assert "Wrote 1 entry" in partial.message
        assert "Rejected as duplicates" in partial.message
        assert "update_entry(key='dark roast', content=<richer info>)" in partial.message
        assert "or skip these" in partial.message
        assert "Nothing new to add" not in partial.message
        assert partial.mutated is True
        # A batch whose entries each duplicate a DIFFERENT existing key must bind
        # EVERY matched key into its own update_entry call — not just the first
        # (#1405: resolve a match for every rejected key in the batch).  'cold brew'
        # now exists (written above), so both entries collide on distinct keys.
        multi = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast blend", "content": "first body"},
                {"key": "cold brew coffee", "content": "a brand new distinct entry"},
            ],
        )
        assert "update_entry(key='dark roast', content=<richer info>)" in multi.message
        assert "update_entry(key='cold brew', content=<richer info>)" in multi.message
        assert multi.mutated is False

    @pytest.mark.asyncio
    async def test_get_returns_entry_or_not_found(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "hello"}]
        )
        assert "hello" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message
        missing = await CollectionGetTool(db).execute(memory="likes", key="absent")
        assert "not found" in missing.message
        # The proven-win read guidance (collection_keys / read_similar; 47%→88%
        # recovery) is intact, AND the rejection now closes the residual write-vs-
        # update decision: once the model finds the entry under a different key it
        # must UPDATE that entry, not collection_write it (which the dedup rejects
        # as a duplicate — the ~1-call ping-pong this guidance removes).
        assert "collection_keys('likes')" in missing.message
        assert "read_similar(memory='likes', anchor=<what you're looking for>)" in missing.message
        assert "update_entry(key=<the key you found>, content=<the new content>)" in missing.message
        assert "creates NEW keys only" in missing.message
        # A bracket-wrapped key (the model's ingrained habit from the old `[key]`
        # display form) is never silently resolved — it's rejected with a teaching
        # error that names the mistake, the current key='...' render, and the bare
        # key ready to reuse.
        bracketed = await CollectionGetTool(db).execute(memory="likes", key="[k]")
        assert bracketed.success is False
        assert bracketed.message == (
            "Key '[k]' not found in 'likes'. The enclosing [brackets] are not part "
            "of the key — entry listings show keys as key='...' and the key is passed "
            "bare, without brackets. This entry's key is 'k'. Retry with key='k'."
        )
        # A bracket-wrapped key whose bare form doesn't exist either gets the
        # ordinary not-found error, not the bracket teaching rejection.
        double_miss = await CollectionGetTool(db).execute(memory="likes", key="[absent]")
        assert "Key '[absent]' not found" in double_miss.message
        assert "Retry with" not in double_miss.message
        # A key that genuinely contains brackets exact-matches with no rejection.
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "[lit]", "content": "bracket literal body"}]
        )
        literal = await CollectionGetTool(db).execute(memory="likes", key="[lit]")
        assert literal.success is True
        assert "bracket literal body" in literal.message

    @pytest.mark.asyncio
    async def test_keys_lists_unique_keys_in_order(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "first", "content": "1"}])
        await write.execute(memory="likes", entries=[{"key": "second", "content": "2"}])
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing.message == "- first\n- second"

    @pytest.mark.asyncio
    async def test_keys_empty_collection_names_source_not_bare_sentinel(self, tmp_path):
        """An empty collection's keys read names the source and marks absence (not an
        error), rather than the bare "(no keys)" sentinel (house wording pass)."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing.message == "No keys in `likes` — the collection is empty (not an error)."

    @pytest.mark.asyncio
    async def test_read_random_returns_all_when_few(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "a", "content": "1"}])
        rendered = await CollectionReadRandomTool(db).execute(memory="likes", k=5)
        assert "key='a' 1" in rendered.message

    @pytest.mark.asyncio
    async def test_read_similar_uses_embedding(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "coffee", "content": "loves coffee"}]
        )
        # Anchor shares the "coffee" word with the entry — the bag-of-words
        # mock embedding gives meaningful cosine, so the entry survives the
        # adaptive cutoff in ``read_similar``.
        rendered = await ReadSimilarTool(db, client).execute(memory="likes", anchor="coffee please")
        assert "coffee" in rendered.message


class TestEmbedFailureRefusesWrite:
    """A transient embed failure at write time REFUSES the write — no vectorless
    (recall-invisible, dedup-weakening) entry is ever persisted (#1412).  The
    prior behaviour stored the entry without a vector and returned an optimistic
    success; now the tool fails with an actionable retry and nothing lands."""

    @staticmethod
    async def _make_relevant_collection(db) -> None:
        # A fully recall-eligible collection: proves the refusal is about the
        # missing vector, not the memory being excluded from recall.
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

    @pytest.mark.asyncio
    async def test_collection_write_refuses_when_embedding_fails(self, tmp_path):
        db = _make_db(tmp_path)
        await self._make_relevant_collection(db)
        write = CollectionWriteTool(db, cast(Any, _FailingEmbedClient()), author="test")
        result = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "loves dark roast"}]
        )
        # Fail-hard: actionable failure naming the transient cause and binding the
        # retry, and no work reported (so the collector throttle sees no-op).
        assert result.success is False
        assert result.mutated is False
        assert "transient embedding error" in result.message
        assert "'dark roast'" in result.message
        assert "collection_write(memory='likes'" in result.message
        # Nothing persisted — the invariant "every stored entry has a vector" holds.
        with Session(db.engine) as session:
            rows = session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == "likes")).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_collection_write_refuses_on_key_only_embed_failure(self, tmp_path):
        # Even when only the key embed fails (content vector fine), storing the
        # entry would leave it missing a vector — so the write is still refused
        # atomically and nothing lands.  (The backfill now also repairs a
        # key-null row, #1468, but the write path won't persist one to begin with.)
        db = _make_db(tmp_path)
        await self._make_relevant_collection(db)
        write = CollectionWriteTool(
            db, cast(Any, _KeyOnlyFailingEmbedClient("dark roast")), author="test"
        )
        result = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "loves dark roast"}]
        )
        assert result.success is False
        assert result.mutated is False
        assert "'dark roast'" in result.message
        with Session(db.engine) as session:
            rows = session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == "likes")).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_log_append_refuses_when_embedding_fails(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, cast(Any, _FailingEmbedClient()), author="test")
        result = await append.execute(memory="events", content="something happened")
        assert result.success is False
        assert result.mutated is False
        assert "transient embedding error" in result.message
        assert "log_append(memory='events'" in result.message
        with Session(db.engine) as session:
            rows = session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == "events")
            ).all()
        assert rows == []


class TestCollectionMutations:
    @pytest.mark.asyncio
    async def test_update_replaces_content(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "old"}]
        )
        result = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "Updated 'k' in 'likes'" in result.message
        fetched = await CollectionGetTool(db).execute(memory="likes", key="k")
        assert "new" in fetched.message
        # A bracket-wrapped key (display form copied from an entry list) is
        # rejected with a teaching error naming the bare key — never absorbed;
        # the entry is untouched.
        bracketed = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="[k]", content="newer"
        )
        assert bracketed.success is False
        assert "Retry with key='k'" in bracketed.message
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message
        # Same teaching rejection on delete: nothing removed, bare key named.
        rejected_delete = await CollectionDeleteEntryTool(db).execute(memory="likes", key="[k]")
        assert rejected_delete.success is False
        assert "Retry with key='k'" in rejected_delete.message
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message
        # A blank replacement is refused (same content bar as collection_write),
        # leaving the existing content untouched rather than blanking the entry.
        # The degenerate-content rule now lives on UpdateEntryArgs.content, so the
        # refusal is produced by the pre-execute Tool.run gate.
        blank = await UpdateEntryTool(db, author="test").run(memory="likes", key="k", content="   ")
        assert blank.success is False
        assert "no word tokens" in blank.message  # what went wrong
        assert "collection_delete_entry" in blank.message  # how to correct it
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message

    @pytest.mark.asyncio
    async def test_update_missing_reports_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "not found" in result.message
        # A bracket-wrapped key whose bare form doesn't exist either gets the
        # ordinary not-found error, not the bracket teaching rejection.
        bracketed = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="[k]", content="new"
        )
        assert "Key '[k]' not found" in bracketed.message
        assert "Retry with" not in bracketed.message

    @pytest.mark.asyncio
    async def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert (
            "Archived 'likes'" in (await CollectionArchiveTool(db).execute(memory="likes")).message
        )
        assert (
            "Unarchived 'likes'"
            in (await CollectionUnarchiveTool(db).execute(memory="likes")).message
        )


class TestLogTools:
    @pytest.mark.asyncio
    async def test_collection_read_latest_refuses_a_log(self, tmp_path, mock_llm):
        """Collection reads error on a log instead of silently bypassing the
        cursored log_read/log_get interface (the read_latest-on-a-log footgun)."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="first"
        )
        rendered = await CollectionReadLatestTool(db).execute(memory="events")
        assert "Refused" in rendered.message
        assert "log_read" in rendered.message

    @pytest.mark.asyncio
    async def test_read_latest_rejects_zero_count(self, tmp_path, mock_llm):
        """``k=0`` (a model guessing zero means "unlimited") reads no entries, so
        the tool would look empty — the arg model refuses it before execute with
        an actionable message (omit k for all), via the Tool.run gate."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="notes", entries=[{"key": "a", "content": "first"}]
        )
        rejected = await CollectionReadLatestTool(db).run(memory="notes", k=0)
        assert rejected.success is False
        assert "at least 1" in rejected.message
        assert "Omit k" in rejected.message
        # A valid read still returns the entry (the guard only rejects k < 1).
        ok = await CollectionReadLatestTool(db).run(memory="notes")
        assert "first" in ok.message

    @pytest.mark.asyncio
    async def test_log_read_window_mode(self, tmp_path, mock_llm):
        """A non-collector caller (scope=None) gets window-mode log_read: recent
        entries within the fixed look-back window — no cursor, no count arg."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="hello")
        rendered = await LogReadTool(db, "chat", scope=None).execute(memory="events")
        assert "hello" in rendered.message
        # A blank append is refused (blank-only — a bare URL is still a valid log
        # entry), so nothing degenerate joins the stream.  The non-blank rule lives
        # on LogAppendArgs.content, so the refusal comes from the Tool.run gate.
        blank = await append.run(memory="events", content="   ")
        assert blank.success is False
        assert "blank" in blank.message

    @pytest.mark.asyncio
    async def test_collector_runs_log_renders_runs_from_promptlog(self, tmp_path):
        """collector-runs is a read facade over promptlog: log_read renders each
        worked run as a record (``[target] summary`` + its tool trace) — no
        stored entries, no keys, no get.  This is the quality collector's review."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="collector-runs", description="audit", inclusion="never", recall="recent"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c0",
                                "type": "function",
                                "function": {
                                    "name": "send_message",
                                    "arguments": '{"content": "Found a new grinder, $300."}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=response,
            agent_name="collector",
            run_id="run-42",
            run_target="espresso-gear",
        )
        db.messages.set_run_outcome("run-42", "worked", "sent an update about a grinder")

        rendered = await LogReadTool(db, "quality", scope="quality").execute(
            memory="collector-runs"
        )

        # collector-runs reads through the uniform log formatter now (it's a log
        # facade like any other) — framed as a fetched batch, runs as records.
        assert "from `collector-runs`" in rendered.message
        assert "[espresso-gear] sent an update about a grinder" in rendered.message
        assert "Found a new grinder, $300." in rendered.message  # the exact message, untruncated

    @pytest.mark.asyncio
    async def test_read_run_calls_renders_by_target(self, tmp_path):
        """read_run_calls is the SEQUENCE lens over runs, orthogonal to target: a
        collector's name renders its runs as ``[target] -> tools -> done``; ``chat``
        renders conversations as ``user -> tools -> penny``.  The valid targets are
        discovered from the DB into its description."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="espresso-gear",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="1. browse for new espresso gear. 2. write it. 3. done().",
            collector_interval_seconds=3600,
            intent="track espresso gear",
        )
        # One completed collector run for espresso-gear.
        coll_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "collection_write",
                                    "arguments": '{"memory": "espresso-gear", '
                                    '"entries": [{"content": "Niche grinder"}]}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=coll_resp,
            agent_name="collector",
            run_id="coll-1",
            run_target="espresso-gear",
        )
        db.messages.set_run_outcome("coll-1", "worked", "wrote a new grinder")
        # One chat run: user message + a tool call, then Penny's reply.
        user_turn = f"live{PennyConstants.SECTION_SEPARATOR}find me a grinder"
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": user_turn}],
            response={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "browse",
                                        "arguments": '{"queries": ["espresso grinder"]}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            agent_name=PennyConstants.CHAT_AGENT_NAME,
            run_id="chat-1",
        )
        db.messages.log_prompt(
            model="m",
            messages=[],
            response={"choices": [{"message": {"content": "Here's a good grinder."}}]},
            agent_name=PennyConstants.CHAT_AGENT_NAME,
            run_id="chat-1",
        )

        tool = ReadRunCallsTool(db, "quality")
        # Targets discovered from the DB (chat + the collector) are in the description.
        assert "espresso-gear" in tool.description
        assert "chat" in tool.description

        collector = await tool.run(target="espresso-gear")
        assert "[espresso-gear]" in collector.message
        assert "collection_write(memory='espresso-gear'" in collector.message
        assert "done: wrote a new grinder" in collector.message

        chat = await tool.run(target="chat")
        assert "user: find me a grinder" in chat.message
        assert "browse(['espresso grinder'])" in chat.message
        assert "penny: Here's a good grinder." in chat.message

        # An unknown/typo'd target resolves to a failed, actionable refusal that
        # names the offending value — not a silent empty batch that reads as "this
        # collector has no runs" (mirrors collector_run_history's resolve-first).
        unknown = await tool.run(target="esspreso-gear")
        assert unknown.success is False
        assert "esspreso-gear" in unknown.message

    @staticmethod
    def _log_run(db, *, run_id: str, target: str, summary: str, write_key: str) -> None:
        """Persist one completed collector run for ``target`` (a write + its
        ``done`` summary) — the promptlog rows ``collector_run_history`` renders.
        ``response`` is a dict (``log_prompt`` serializes it); the inner tool-call
        ``arguments`` is itself a JSON string, as the model emits it."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "w0",
                                "type": "function",
                                "function": {
                                    "name": "collection_write",
                                    "arguments": json.dumps(
                                        {
                                            "memory": target,
                                            "entries": [{"key": write_key, "content": "v"}],
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=response,
            agent_name="collector",
            run_id=run_id,
            run_target=target,
        )
        db.messages.set_run_outcome(run_id, "worked", summary)

    @staticmethod
    async def _create_collection(db, name: str) -> None:
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name=name,
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

    @pytest.mark.asyncio
    async def test_collector_run_history_scopes_to_one_collector_newest_first(self, tmp_path):
        """``collector_run_history`` returns ONE named collector's recent runs as
        records, newest-first — so a reviewer judges a pattern across that
        collector's cycles, not the cross-collector index ``log_read`` gives.
        Other collectors' runs are excluded."""
        db = _make_db(tmp_path)
        await self._create_collection(db, "ai-news")
        await self._create_collection(db, "espresso")
        self._log_run(db, run_id="news-1", target="ai-news", summary="wrote 1", write_key="older")
        self._log_run(db, run_id="news-2", target="ai-news", summary="wrote 2", write_key="newer")
        self._log_run(db, run_id="gear-1", target="espresso", summary="wrote g", write_key="grind")

        result = await CollectorRunHistoryTool(db).execute(collector="ai-news")

        assert "from `ai-news`" in result.message and "most recent first" in result.message
        # Both of this collector's runs, newest first; the other collector absent.
        assert result.message.index("wrote 2") < result.message.index("wrote 1")
        assert "[ai-news]" in result.message and "[espresso]" not in result.message
        assert "wrote g" not in result.message

    @pytest.mark.asyncio
    async def test_collector_run_history_unknown_collector_is_actionable_error(self, tmp_path):
        """An unknown collector name returns a failed, actionable refusal (not a
        silent empty history that would read as 'this collector is healthy')."""
        db = _make_db(tmp_path)
        result = await CollectorRunHistoryTool(db).execute(collector="does-not-exist")
        assert result.success is False
        assert "does-not-exist" in result.message

    @pytest.mark.asyncio
    async def test_collector_run_history_no_runs_yet_is_clear(self, tmp_path):
        """A real collector with no completed runs gets a clear 'no runs yet'
        sentinel — distinct from the unknown-name error, so the model judges it
        from its current run rather than reading absence as health."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="fresh-feed",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectorRunHistoryTool(db).execute(collector="fresh-feed")
        assert result.success is True
        assert "No completed runs" in result.message and "fresh-feed" in result.message

    @pytest.mark.asyncio
    async def test_append_to_system_log_is_refused(self, tmp_path, mock_llm):
        """Invariant #1: the four framework-managed system logs are written
        only by Python side-effects.  ``log_append`` from any agent gets a
        readable refusal and writes nothing — guarding the conversation-turn
        reconstruction and the run audit trail from model-authored entries."""
        db = _make_db(tmp_path)
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        # The reserved-target check is a pure constant lookup, so it lives on
        # LogAppendArgs.memory and the refusal comes from the Tool.run gate.
        for system_log in PennyConstants.SYSTEM_LOGS:
            result = await append.run(memory=system_log, content="forged turn")
            assert result.success is False
            assert system_log in result.message
            assert "system log" in result.message  # what went wrong + how to fix
        # Nothing was created/written — the refusal short-circuits before the store.
        assert db.memories.get(PennyConstants.MEMORY_PENNY_MESSAGES_LOG) is None

    @pytest.mark.asyncio
    async def test_log_similar_with_client(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="relevant", recall="relevant"
        )
        client = _make_llm_client(mock_llm)
        await LogAppendTool(db, client, author="test").execute(
            memory="events", content="coffee is great"
        )
        # Anchor shares words with the entry so the bag-of-words mock
        # embedding gives meaningful cosine and the entry survives the
        # adaptive cutoff in ``read_similar``.
        rendered = await ReadSimilarTool(db, client).execute(
            memory="events", anchor="coffee morning"
        )
        assert "coffee is great" in rendered.message

    @pytest.mark.asyncio
    async def test_read_next_returns_all_entries_when_no_cursor(self, tmp_path, mock_llm):
        """Without a stored cursor, read_next returns every entry in the log."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await read_next.execute(memory="events")

        assert "first" in rendered.message
        assert "second" in rendered.message

    @pytest.mark.asyncio
    async def test_commit_pending_advances_cursor_to_max_seen(self, tmp_path, mock_llm):
        """commit_pending writes the highest timestamp seen during the run."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # A new instance after commit should see no entries (cursor caught up).
        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(memory="events")
        # Empty read names the source and marks it as absence, not an error.
        assert (
            rendered.message
            == "No entries in `events` — it's empty or nothing matched (not an error)."
        )

    @pytest.mark.asyncio
    async def test_discard_pending_leaves_cursor_unchanged(self, tmp_path, mock_llm):
        """discard_pending drops the in-memory state without touching the DB cursor."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(memory="events")
        read_next.discard_pending()

        # Cursor still at None; a new read sees the same entries.
        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(memory="events")
        assert "first" in rendered.message

    @pytest.mark.asyncio
    async def test_first_cycle_bounded_to_latest_n_entries(self, tmp_path, mock_llm):
        """A brand-new collector (no cursor yet) reading a busy log gets the
        most-recent N entries, not every entry since the dawn of time.

        Without this bound, a fresh collector reading ``user-messages`` (which
        has months of chat history in production) would dump the entire log
        into the first cycle's context.
        """
        from penny.constants import PennyConstants

        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        # Append more entries than the bound to confirm trimming
        n_entries = PennyConstants.LOG_READ_LIMIT + 5
        for i in range(n_entries):
            await append.execute(memory="events", content=f"entry-{i:02d}")

        read_next = LogReadTool(db, agent_name="brand-new-collector", scope="brand-new-collector")
        rendered = await read_next.execute(memory="events")

        # Exactly the latest N entries — entry-(n-N) through entry-(n-1)
        # should appear; older entries should not.
        for i in range(n_entries - PennyConstants.LOG_READ_LIMIT, n_entries):
            assert f"entry-{i:02d}" in rendered.message
        # The first 5 entries must be excluded
        assert "entry-00" not in rendered.message
        assert "entry-04" not in rendered.message

    @pytest.mark.asyncio
    async def test_first_cycle_advances_cursor_so_next_cycle_sees_only_new(
        self, tmp_path, mock_llm
    ):
        """After a bounded first cycle commits, the next cycle picks up
        incrementally — even entries that the first cycle's bound excluded
        stay excluded (since they're older than the cursor)."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        for i in range(15):
            await append.execute(memory="events", content=f"old-{i:02d}")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # New entries arrive
        await append.execute(memory="events", content="new-after-cursor")

        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(memory="events")
        assert "new-after-cursor" in rendered.message
        # Old entries excluded by the bound stay excluded
        assert "old-00" not in rendered.message

    @pytest.mark.asyncio
    async def test_cursor_read_is_capped_and_advances_by_batch(self, tmp_path, mock_llm):
        """With a cursor established, a backlog larger than the batch bound is
        returned in bounded chunks — read N, cursor advances by N, the next read
        picks up the next N.  The caller never reasons about a count."""
        from penny.constants import PennyConstants

        limit = PennyConstants.LOG_READ_LIMIT
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")

        # Establish a cursor, then pile up a backlog bigger than one batch.
        await append.execute(memory="events", content="seed")
        seed_read = LogReadTool(db, agent_name="extractor", scope="extractor")
        await seed_read.execute(memory="events")
        seed_read.commit_pending()

        backlog = limit + 3
        for i in range(backlog):
            await append.execute(memory="events", content=f"backlog-{i:02d}")

        first = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered_first = await first.execute(memory="events")
        first.commit_pending()
        # Exactly one batch — the oldest N of the backlog, not all of it.
        assert rendered_first.message.count("backlog-") == limit
        assert "backlog-00" in rendered_first.message
        assert f"backlog-{backlog - 1:02d}" not in rendered_first.message

        # The next read picks up the remainder since the advanced cursor.
        second = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered_second = await second.execute(memory="events")
        assert f"backlog-{backlog - 1:02d}" in rendered_second.message

    @pytest.mark.asyncio
    async def test_per_agent_cursors_are_independent(self, tmp_path, mock_llm):
        """Two agents reading the same log have independent cursor state."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="hello"
        )

        agent_a = LogReadTool(db, agent_name="a", scope="a")
        await agent_a.execute(memory="events")
        agent_a.commit_pending()

        # Agent B has its own cursor and still sees the entry.
        agent_b = LogReadTool(db, agent_name="b", scope="b")
        rendered = await agent_b.execute(memory="events")
        assert "hello" in rendered.message


class TestExistsAndDone:
    @pytest.mark.asyncio
    async def test_exists_yes_via_exact_key(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "dark roast", "content": "body"}]
        )
        result = await ExistsTool(db, client).execute(
            memories=["likes"], key="dark roast", content="body"
        )
        assert result.message == "yes"

    @pytest.mark.asyncio
    async def test_exists_no(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await ExistsTool(db, _make_llm_client(mock_llm)).execute(
            memories=["likes"], key="not there", content="nothing"
        )
        assert result.message == "no"

    @pytest.mark.asyncio
    async def test_exists_unknown_memory_name_is_not_found(self, tmp_path, mock_llm):
        """A misspelled memory name must not read as an empty (always-"no")
        memory — that green-lights the write the model was probing for.  The
        probe fails with the actionable not-found refusal naming the bad value."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await ExistsTool(db, _make_llm_client(mock_llm)).execute(
            memories=["lieks"], content="dark roast"
        )
        assert result.success is False
        assert "lieks" in result.message
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_exists_empty_memories_is_actionable_not_bare_pydantic(self, tmp_path, mock_llm):
        """An empty ``memories`` list gets a named, actionable rejection through the
        arg-validation envelope — not Pydantic's bare "List should have at least 1
        item" (the house wording-unification pass)."""
        db = _make_db(tmp_path)
        result = await ExistsTool(db, _make_llm_client(mock_llm)).run(memories=[], content="x")
        assert result.success is False
        assert "at least one collection name" in result.message
        assert "List should have at least 1 item" not in result.message

    @pytest.mark.asyncio
    async def test_exists_embed_failure_is_inconclusive_not_no(self, tmp_path, mock_llm):
        """When the embed service is down the similarity dedup is skipped, so a
        "no" would be a silent degradation that could green-light a near-duplicate
        write.  The probe surfaces the inconclusive state instead (visible
        degradation)."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)

        def _fail_embed(model: str, text: str | list[str]) -> list[list[float]]:
            raise LlmConnectionError("embedding service unavailable")

        mock_llm.set_embed_handler(_fail_embed)
        result = await ExistsTool(db, client).execute(memories=["likes"], content="nothing")

        assert result.message != "no"
        assert "inconclusive" in result.message

    @pytest.mark.asyncio
    async def test_unicode_hyphen_in_memory_name_normalized(self, tmp_path, mock_llm):
        """Regression: gpt-oss occasionally emits Unicode dashes (U+2010,
        U+2011, …) where ASCII hyphen-minus is expected, breaking string
        comparison in tool args.  Memory-name fields normalise on the way
        in so the rest of the stack sees the canonical form."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="board-games",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        # Non-breaking hyphen U+2011 in the memory name — model output
        # observed in the wild.
        result = await write.execute(
            memory="board‑games",
            entries=[{"key": "k", "content": "v"}],
        )
        assert "Wrote 1 entry to 'board-games'" in result.message

    @pytest.mark.asyncio
    async def test_exists_content_only_uses_content_as_key_probe(self, tmp_path, mock_llm):
        """Regression: ``exists(content="Catan")`` must catch an
        existing entry with ``key="Catan"``, even when the
        existing row's *content* is a long description that doesn't
        cosine-match the short candidate.  The tool now copies content
        into the key slot when the model omits it, letting key-TCR fire
        in the dedup rule."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="board-games",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)
        # Existing entry: short key, long descriptive content.
        await CollectionWriteTool(db, client, author="test").execute(
            memory="board-games",
            entries=[
                {
                    "key": "Catan",
                    "content": (
                        "Catan – A gateway strategy board game of trading and "
                        "settlement, designed by Klaus Teuber, first published "
                        "1995, widely credited with popularising modern hobby "
                        "board gaming."
                    ),
                }
            ],
        )
        # Probe with content only — what the collector usually does when
        # checking a candidate name before writing.
        result = await ExistsTool(db, client).execute(memories=["board-games"], content="Catan")
        assert result.message == "yes"

    @pytest.mark.asyncio
    async def test_done_returns_structured_summary(self):
        result = await DoneTool().execute(success=True, summary="wrote 3 entries")
        assert "wrote 3 entries" in result.message
        assert "success" in result.message

    @pytest.mark.asyncio
    async def test_done_no_op_marker(self):
        result = await DoneTool().execute(success=False, summary="no new matches")
        assert "no new matches" in result.message
        assert "no-op" in result.message

    @pytest.mark.asyncio
    async def test_done_requires_success_and_summary(self):
        with pytest.raises(Exception):  # noqa: B017,PT011 — Pydantic ValidationError
            await DoneTool().execute()


class TestAuthorAttribution:
    @pytest.mark.asyncio
    async def test_writes_stamp_constructor_author(self, tmp_path, mock_llm):
        """Author is bound at tool construction (not pulled from ambient state)."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="preference-extractor"
        ).execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        rows = db.memory("likes").get("k")
        assert rows[0].author == "preference-extractor"


class TestCollectionMerge:
    @pytest.mark.asyncio
    async def test_merge_moves_entries_and_archives_source(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="src", entries=[{"key": "a", "content": "alpha"}])
        await write.execute(memory="src", entries=[{"key": "b", "content": "beta"}])

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "2 moved" in result.message
        assert "archived" in result.message
        assert db.memories.get("src").archived is True
        assert len(db.memory("dst").read_all()) == 2
        assert len(db.memory("src").read_all()) == 0

    @pytest.mark.asyncio
    async def test_merge_drops_colliding_keys(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="src", entries=[{"key": "shared", "content": "from src"}])
        await write.execute(memory="src", entries=[{"key": "unique", "content": "only in src"}])
        await write.execute(memory="dst", entries=[{"key": "shared", "content": "already in dst"}])

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "1 moved" in result.message
        assert "1 dropped" in result.message
        # The dropped collision keys are named, not just counted.
        assert "'shared'" in result.message
        dst_entries = db.memory("dst").read_all()
        assert len(dst_entries) == 2
        contents = {e.key: e.content for e in dst_entries}
        assert contents["shared"] == "already in dst"  # destination wins
        assert contents["unique"] == "only in src"

    @pytest.mark.asyncio
    async def test_merge_empty_source_archives_it(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "archived" in result.message
        assert db.memories.get("src").archived is True


class TestTestExtractionPromptTool:
    """TestExtractionPromptTool delegates to Collector.run_for — test the formatting."""

    class _MockCollector:
        """Duck-typed stub: records the call and returns a configured result."""

        def __init__(self, result: tuple[bool, str]) -> None:
            self._result = result
            self.called_with: str | None = None

        async def run_for(self, collection_name: str) -> tuple[bool, str]:
            self.called_with = collection_name
            return self._result

    @pytest.mark.asyncio
    async def test_success_returns_checkmark_and_summary(self):
        collector = self._MockCollector((True, "Collector cycle complete. wrote 3 entries"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="board-games")
        assert collector.called_with == "board-games"
        assert result.message.startswith("✅")
        assert "wrote 3 entries" in result.message
        assert result.success is True

    @pytest.mark.asyncio
    async def test_failure_returns_x_and_summary(self):
        collector = self._MockCollector((False, "Collector cycle complete. max steps exceeded"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="likes")
        assert result.message.startswith("❌")
        assert "max steps exceeded" in result.message
        # the failure must reach structural accounting, not live only in the ❌ text
        assert result.success is False

    @pytest.mark.asyncio
    async def test_validation_error_returns_x_and_error_message(self):
        collector = self._MockCollector((False, "Collection 'missing' not found."))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="missing")
        assert result.message.startswith("❌")
        assert "not found" in result.message
        assert result.success is False

    @pytest.mark.asyncio
    async def test_unicode_dash_in_memory_name_normalized(self):
        """MemoryNameArgs normalises Unicode dashes before passing to run_for."""
        collector = self._MockCollector((True, "Collector cycle complete. wrote 1 entry"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        await tool.execute(memory="board‑games")  # U+2011 non-breaking hyphen
        assert collector.called_with == "board-games"


class TestFactory:
    """One uniform surface for every agent — reads + lifecycle (shape) + entry
    mutations (contents).  Capability is no longer curated by omission; the
    only per-agent difference is ``scope``, which drives the collector-binding
    *invariant* (see TestScopedFactory), not which tools are present.
    """

    _FULL_SURFACE = {
        # Reads
        "collection_get",
        "collection_read_latest",
        "collection_read_random",
        "collection_keys",
        "memory_metadata",
        "collection_catalog",
        "log_read",
        "read_run_calls",
        "collector_run_history",
        "read_published_latest",
        "read_similar",
        "exists",
        # Lifecycle (shape)
        "collection_create",
        "collection_update",
        "collection_merge",
        "collection_archive",
        "collection_unarchive",
        "log_create",
        # Entry mutations (contents)
        "collection_write",
        "update_entry",
        "collection_delete_entry",
        "log_append",
    }

    def test_chat_surface_is_the_full_set(self, tmp_path, mock_llm):
        """Chat (scope=None) gets every memory tool — entry mutations included,
        unrestricted, since edits are user-directed."""
        db = _make_db(tmp_path)
        tools = build_memory_tools(db, _make_llm_client(mock_llm), agent_name="chat")
        assert {tool.name for tool in tools} == self._FULL_SURFACE

    def test_collector_surface_is_the_same_full_set(self, tmp_path, mock_llm):
        """A bound collector (scope=X) gets the identical surface — scope binds
        its entry mutations to X but does not strip lifecycle/other tools."""
        db = _make_db(tmp_path)
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="collector", scope="likes"
        )
        assert {tool.name for tool in tools} == self._FULL_SURFACE


class TestScopedFactory:
    """Scope binds a collector to one collection.  Writes to other collections
    get a clean refusal at the tool layer, so a confused or jailbroken
    collector can't trash unrelated memories.
    """

    @pytest.mark.asyncio
    async def test_scoped_write_rejects_other_collection(self, tmp_path, mock_llm):
        """A scoped collector that tries to write to a different collection
        gets a clean refusal rather than silently corrupting unrelated data."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="dislikes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="dislikes", entries=[{"key": "k", "content": "v"}])

        assert (
            result.message == "Refused: this collector can only write to 'likes', not 'dislikes'."
        )
        # And nothing was actually written
        assert db.memory("dislikes").get("k") == []

    @pytest.mark.asyncio
    async def test_scoped_write_allows_target_collection(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        assert "Wrote 1 entry" in result.message
        assert db.memory("likes").get("k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_update_entry_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        update = UpdateEntryTool(db, author="collector:likes", scope="likes")
        result = await update.execute(memory="dislikes", key="k", content="v")
        assert "Refused" in result.message

    @pytest.mark.asyncio
    async def test_scoped_delete_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        delete = CollectionDeleteEntryTool(db, scope="likes")
        result = await delete.execute(memory="dislikes", key="k")
        assert "Refused" in result.message


class TestReadPublishedLatest:
    """The consumer side of the pub/sub layer: each call picks one published
    collection at random among those with unseen entries and returns its oldest n,
    advancing only the returned entry's cursor so nothing repeats and nothing is
    skipped.  Random rotation (not a global oldest-first pool) keeps one
    collection's burst from monopolizing delivery."""

    def _publish(self, db: Database, name: str, *, published: bool = True) -> None:
        db.memories.create_collection(
            name, f"{name} desc", Inclusion.NEVER, RecallMode.RECENT, published=published
        )

    def _write(self, db: Database, name: str, key: str, content: str) -> None:
        db.memory(name).write([EntryInput(key=key, content=content)], author="producer")

    def _backdate(self, db: Database, name: str, key: str, when: datetime) -> None:
        with Session(db.engine) as session:
            row = session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == name, MemoryEntry.key == key)
            ).first()
            assert row is not None
            row.created_at = when
            session.add(row)
            session.commit()

    def _pin_choice(self, monkeypatch, name: str) -> list[set[str]]:
        """Force the random pick to a named source; return a list that records the
        candidate pool seen on each call (so a test can assert what was eligible)."""
        pools: list[set[str]] = []

        def pick(sources):
            pools.append({source.name for source in sources})
            return next(source for source in sources if source.name == name)

        monkeypatch.setattr("penny.tools.memory_tools.random.choice", pick)
        return pools

    @pytest.mark.asyncio
    async def test_picks_a_random_published_source_never_an_unpublished_one(
        self, tmp_path, monkeypatch
    ):
        db = _make_db(tmp_path)
        self._publish(db, "silent", published=False)
        self._publish(db, "games")
        self._publish(db, "jobs")
        # The unpublished entry is the oldest of all — under the old global
        # oldest-first pool it would have won; it must never even be a candidate.
        self._write(db, "silent", "old", "oldest but unpublished")
        self._write(db, "games", "stardrift", "Stardrift Saga")
        self._write(db, "jobs", "role", "New engineering role")
        pools = self._pin_choice(monkeypatch, "jobs")
        result = await ReadPublishedLatestTool(db, "notifier").execute(n=1)
        # The randomly-picked source is delivered (not the globally-oldest entry),
        # and the candidate pool is exactly the eligible published collections.
        assert "New engineering role" in result.message
        assert "from `jobs`" in result.message
        # The key renders in invocation form — the copyable key='...' shape,
        # matching every other model-facing entry render (never `[key]`).
        assert "key='role'" in result.message
        assert pools == [{"games", "jobs"}]
        assert "unpublished" not in result.message
        assert "Stardrift Saga" not in result.message

    @pytest.mark.asyncio
    async def test_advances_only_returned_source_and_never_repeats(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path)
        self._publish(db, "games")
        self._publish(db, "jobs")
        self._write(db, "games", "g1", "game one")
        self._write(db, "jobs", "j1", "job one")
        tool = ReadPublishedLatestTool(db, "notifier")
        self._pin_choice(monkeypatch, "games")
        first = await tool.execute(n=1)
        assert "game one" in first.message
        tool.commit_pending()
        # Only the returned source's cursor advanced; the scanned-but-unpicked
        # source is left untouched so its entry isn't skipped.
        assert db.cursors.get("notifier", "games") is not None
        assert db.cursors.get("notifier", "jobs") is None
        # games is drained; the only eligible source left is jobs.
        self._pin_choice(monkeypatch, "jobs")
        second = await tool.execute(n=1)
        assert "job one" in second.message
        assert "game one" not in second.message
        tool.commit_pending()
        # Both caught up now.
        third = await tool.execute(n=1)
        assert third.message == "(no new published entries)"
        # Durable library: entries stay put — drained by cursor, never moved.
        assert len(db.memory("games").read_all()) == 1
        assert len(db.memory("jobs").read_all()) == 1

    @pytest.mark.asyncio
    async def test_random_rotation_offers_every_collection_not_just_the_oldest_burst(
        self, tmp_path, monkeypatch
    ):
        """The starvation fix: a single collector run stamps many entries at once.
        Under the old global oldest-first pool that whole burst drained before a
        newer, smaller collection got a turn.  Random rotation offers every
        eligible collection each cycle, so a newer collection delivers even while
        an older burst still has unsent entries."""
        db = _make_db(tmp_path)
        self._publish(db, "warhammer")
        self._publish(db, "emulator")
        base = datetime.now(UTC) - timedelta(hours=1)
        # A burst of 3 warhammer entries, all OLDER than the single emulator one.
        for index in range(3):
            self._write(db, "warhammer", f"w{index}", f"warhammer {index}")
            self._backdate(db, "warhammer", f"w{index}", base + timedelta(seconds=index))
        self._write(db, "emulator", "e0", "emulator zero")
        self._backdate(db, "emulator", "e0", base + timedelta(minutes=30))
        # emulator is eligible despite the older, unsent warhammer burst sitting
        # ahead of it in time — it is not stuck behind the burst.
        pools = self._pin_choice(monkeypatch, "emulator")
        result = await ReadPublishedLatestTool(db, "notifier").execute(n=1)
        assert "emulator zero" in result.message
        assert pools == [{"warhammer", "emulator"}]

    @pytest.mark.asyncio
    async def test_discard_pending_does_not_advance(self, tmp_path):
        db = _make_db(tmp_path)
        self._publish(db, "games")
        self._write(db, "games", "g1", "game one")
        tool = ReadPublishedLatestTool(db, "notifier")
        await tool.execute(n=1)
        tool.discard_pending()  # a failed cycle drops the advance
        assert db.cursors.get("notifier", "games") is None
        # So the entry is offered again on the next cycle.
        again = await tool.execute(n=1)
        assert "game one" in again.message

    @pytest.mark.asyncio
    async def test_skips_archived_published_collection(self, tmp_path):
        db = _make_db(tmp_path)
        self._publish(db, "games")
        self._write(db, "games", "g1", "game one")
        db.memories.archive("games")
        result = await ReadPublishedLatestTool(db, "notifier").execute(n=1)
        assert result.message == "(no new published entries)"

    @pytest.mark.asyncio
    async def test_cold_start_ignores_backlog_older_than_window(self, tmp_path):
        db = _make_db(tmp_path)
        self._publish(db, "games")
        self._write(db, "games", "old", "ancient game")
        # A brand-new consumer must not replay a long-standing backlog the day it
        # subscribes: an entry older than the cold-start window is treated as seen.
        self._backdate(
            db,
            "games",
            "old",
            datetime.now(UTC)
            - timedelta(seconds=PennyConstants.PUBLISHED_COLDSTART_LOOKBACK_SECONDS + 86400),
        )
        tool = ReadPublishedLatestTool(db, "notifier")
        assert (await tool.execute(n=1)).message == "(no new published entries)"
        # But a fresh entry (within the window) is surfaced.
        self._write(db, "games", "new", "fresh game")
        assert "fresh game" in (await tool.execute(n=1)).message
