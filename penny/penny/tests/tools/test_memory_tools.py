"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, RecallMode
from penny.database.models import MemoryEntry
from penny.llm.client import LlmClient
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionDeleteEntryTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMergeTool,
    CollectionMoveTool,
    CollectionReadLatestTool,
    CollectionReadRandomTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    DoneTool,
    ExistsTool,
    LogAppendTool,
    LogCreateTool,
    LogReadTool,
    ReadPublishedLatestTool,
    ReadSimilarTool,
    TestExtractionPromptTool,
    UpdateEntryTool,
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


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_collection_persists(self, tmp_path):
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
            name="user-messages", description="inbound", inclusion="always", recall="recent"
        )
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["user-messages"].type == "log"
        assert memories["user-messages"].recall == "recent"

    @pytest.mark.asyncio
    async def test_create_collection_duplicate_returns_user_friendly_message(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="ai-news",
            description="first",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectionCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
            name="events", description="first", inclusion="always", recall="recent"
        )
        result = await LogCreateTool(db, None).execute(
            name="events", description="second", inclusion="never", recall="recent"
        )
        assert "already exists" in result.message
        assert "events" in result.message

    @pytest.mark.asyncio
    async def test_create_rejects_short_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, None).execute(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="yes",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert "too short" in result.message
        assert "minimum" in result.message
        assert db.memories.get("notes") is None  # collection not created

    @pytest.mark.asyncio
    async def test_create_rejects_blank_description(self, tmp_path):
        # The description doubles as the stage-1 routing anchor — a blank one
        # would embed an empty string and never match, so it's refused.
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, None).execute(
            name="notes",
            description="   ",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert "description cannot be blank" in result.message
        assert db.memories.get("notes") is None  # collection not created

    @pytest.mark.asyncio
    async def test_create_rejects_missing_extraction_prompt(self, tmp_path):
        # extraction_prompt is required — a collection without one is passive
        # (nothing fills it), so the tool surface refuses to create one.
        db = _make_db(tmp_path)
        with pytest.raises(ValidationError):
            await CollectionCreateTool(db, None).execute(
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
            await CollectionCreateTool(db, None).execute(
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
        result = await CollectionCreateTool(db, None).execute(
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
    async def test_update_rejects_short_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        original_prompt = "test fixture extraction prompt"
        await CollectionCreateTool(db, None).execute(
            name="notes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectionUpdateTool(db, None).execute(name="notes", extraction_prompt="yes")
        assert "too short" in result.message
        # Update rejected — original prompt preserved unchanged
        assert db.memories.get("notes").extraction_prompt == original_prompt

    @pytest.mark.asyncio
    async def test_update_treats_blank_fields_as_omitted(self, tmp_path, mock_llm):
        # Models emit "" for an optional field they mean to leave alone (gpt-oss
        # was observed passing extraction_prompt="" alongside a recall change).
        # A blank must be skipped, not written through: the recall change lands
        # while the existing prompt/description survive untouched.
        db = _make_db(tmp_path)
        original_prompt = "test fixture extraction prompt that is long enough"
        await CollectionCreateTool(db, None).execute(
            name="notes",
            description="real description",
            inclusion="relevant",
            recall="all",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectionUpdateTool(db, None).execute(
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


class TestCollectionWritesAndReads:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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

    @pytest.mark.asyncio
    async def test_write_reports_duplicate_via_tcr(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
        # The candidate's own key is named, *and* the existing key it
        # collided with — gives the model enough context to pivot to
        # update_entry instead of silently dropping fresher info.
        assert "dark roast coffee" in result.message
        assert "matches existing 'dark roast'" in result.message
        assert "update_entry" in result.message

    @pytest.mark.asyncio
    async def test_get_returns_entry_or_not_found(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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

    @pytest.mark.asyncio
    async def test_keys_lists_unique_keys_in_order(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
    async def test_read_random_returns_all_when_few(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
        assert "[a] 1" in rendered.message

    @pytest.mark.asyncio
    async def test_read_similar_uses_embedding(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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

    @pytest.mark.asyncio
    async def test_read_similar_without_llm_client_returns_sentinel(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await ReadSimilarTool(db, None).execute(memory="likes", anchor="whatever")
        assert "Similarity search unavailable" in result.message
        assert "collection_read_latest" in result.message  # points at a working alternative


class TestCollectionMutations:
    @pytest.mark.asyncio
    async def test_update_replaces_content(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
        # A blank replacement is refused (same content bar as collection_write),
        # leaving the existing content untouched rather than blanking the entry.
        blank = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="   "
        )
        assert blank.success is False
        assert "no word tokens" in blank.message  # what went wrong
        assert "collection_delete_entry" in blank.message  # how to correct it
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message

    @pytest.mark.asyncio
    async def test_update_missing_reports_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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

    @pytest.mark.asyncio
    async def test_move_between_collections(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="unnotified",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
            name="notified",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="unnotified", entries=[{"key": "t1", "content": "x"}]
        )
        result = await CollectionMoveTool(db, author="test").execute(
            key="t1", from_memory="unnotified", to_memory="notified"
        )
        assert "Moved 't1'" in result.message

    @pytest.mark.asyncio
    async def test_move_collision(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="a",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
            name="b",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="a", entries=[{"key": "k", "content": "src"}])
        await write.execute(memory="b", entries=[{"key": "k", "content": "dst"}])
        result = await CollectionMoveTool(db, author="test").execute(
            key="k", from_memory="a", to_memory="b"
        )
        assert "already has a 'k' entry" in result.message

    @pytest.mark.asyncio
    async def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="first"
        )
        rendered = await CollectionReadLatestTool(db).execute(memory="events")
        assert "Refused" in rendered.message
        assert "log_read" in rendered.message

    @pytest.mark.asyncio
    async def test_log_read_window_mode(self, tmp_path, mock_llm):
        """A non-collector caller (scope=None) gets window-mode log_read: recent
        entries within the fixed look-back window — no cursor, no count arg."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="hello")
        rendered = await LogReadTool(db, "chat", scope=None).execute(memory="events")
        assert "hello" in rendered.message
        # A blank append is refused (blank-only — a bare URL is still a valid log
        # entry), so nothing degenerate joins the stream.
        blank = await append.execute(memory="events", content="   ")
        assert blank.success is False
        assert "blank" in blank.message

    @pytest.mark.asyncio
    async def test_collector_runs_log_renders_runs_from_promptlog(self, tmp_path):
        """collector-runs is a read facade over promptlog: log_read renders each
        worked run as a record (``[target] summary`` + its tool trace) — no
        stored entries, no keys, no get.  This is the quality collector's review."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
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
    async def test_append_to_system_log_is_refused(self, tmp_path, mock_llm):
        """Invariant #1: the four framework-managed system logs are written
        only by Python side-effects.  ``log_append`` from any agent gets a
        readable refusal and writes nothing — guarding the conversation-turn
        reconstruction and the run audit trail from model-authored entries."""
        db = _make_db(tmp_path)
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        for system_log in PennyConstants.SYSTEM_LOGS:
            result = await append.execute(memory=system_log, content="forged turn")
            assert "Refused" in result.message
            assert system_log in result.message
        # Nothing was created/written — the refusal short-circuits before the store.
        assert db.memories.get(PennyConstants.MEMORY_PENNY_MESSAGES_LOG) is None

    @pytest.mark.asyncio
    async def test_log_similar_with_client(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
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
        assert rendered.message == "(no entries)"

    @pytest.mark.asyncio
    async def test_discard_pending_leaves_cursor_unchanged(self, tmp_path, mock_llm):
        """discard_pending drops the in-memory state without touching the DB cursor."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
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
        await LogCreateTool(db, None).execute(
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
        await CollectionCreateTool(db, None).execute(
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
        await CollectionCreateTool(db, None).execute(
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
    async def test_unicode_hyphen_in_memory_name_normalized(self, tmp_path, mock_llm):
        """Regression: gpt-oss occasionally emits Unicode dashes (U+2010,
        U+2011, …) where ASCII hyphen-minus is expected, breaking string
        comparison in tool args.  Memory-name fields normalise on the way
        in so the rest of the stack sees the canonical form."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
        await CollectionCreateTool(db, None).execute(
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
        await CollectionCreateTool(db, None).execute(
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
        await CollectionCreateTool(db, None).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
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
        await CollectionCreateTool(db, None).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
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
        dst_entries = db.memory("dst").read_all()
        assert len(dst_entries) == 2
        contents = {e.key: e.content for e in dst_entries}
        assert contents["shared"] == "already in dst"  # destination wins
        assert contents["unique"] == "only in src"

    @pytest.mark.asyncio
    async def test_merge_empty_source_archives_it(self, tmp_path):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
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

    @pytest.mark.asyncio
    async def test_failure_returns_x_and_summary(self):
        collector = self._MockCollector((False, "Collector cycle complete. max steps exceeded"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="likes")
        assert result.message.startswith("❌")
        assert "max steps exceeded" in result.message

    @pytest.mark.asyncio
    async def test_validation_error_returns_x_and_error_message(self):
        collector = self._MockCollector((False, "Collection 'missing' not found."))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="missing")
        assert result.message.startswith("❌")
        assert "not found" in result.message

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
        "log_read",
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
        "collection_move",
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
        await CollectionCreateTool(db, None).execute(
            name="likes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
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
            "Refused" in result.message
            and "likes" in result.message
            and "dislikes" in result.message
        )
        # And nothing was actually written
        assert db.memory("dislikes").get("k") == []

    @pytest.mark.asyncio
    async def test_scoped_write_allows_target_collection(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
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
    async def test_scoped_move_allows_into_target(self, tmp_path, mock_llm):
        """Move's destination is the entry write — if to_memory == scope,
        the move is in-bounds even though from_memory is a different memory.
        """
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="t").execute(
            memory="src", entries=[{"key": "k", "content": "v"}]
        )

        move = CollectionMoveTool(db, author="collector:dst", scope="dst")
        result = await move.execute(key="k", from_memory="src", to_memory="dst")
        assert "Moved 'k'" in result.message
        assert db.memory("dst").get("k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_move_defaults_to_memory_from_scope(self, tmp_path, mock_llm):
        """Omitting to_memory on a scoped tool defaults to the bound scope."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, None).execute(
            name="src",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="t").execute(
            memory="src", entries=[{"key": "k", "content": "v"}]
        )

        move = CollectionMoveTool(db, author="collector:dst", scope="dst")
        result = await move.execute(key="k", from_memory="src")
        assert "Moved 'k'" in result.message
        assert db.memory("dst").get("k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_move_to_memory_not_required_in_schema(self, tmp_path):
        """Scoped instance exposes to_memory as optional in its parameters schema."""
        move = CollectionMoveTool(_make_db(tmp_path), author="collector:dst", scope="dst")
        assert "to_memory" not in move.parameters["required"]

    @pytest.mark.asyncio
    async def test_scoped_move_rejects_outbound(self, tmp_path):
        db = _make_db(tmp_path)
        move = CollectionMoveTool(db, author="collector:src", scope="src")
        result = await move.execute(key="k", from_memory="src", to_memory="dst")
        assert "Refused" in result.message and "src" in result.message and "dst" in result.message

    @pytest.mark.asyncio
    async def test_scoped_delete_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        delete = CollectionDeleteEntryTool(db, scope="likes")
        result = await delete.execute(memory="dislikes", key="k")
        assert "Refused" in result.message


class TestReadPublishedLatest:
    """The consumer side of the pub/sub layer: a fan-in cursored read across
    every published collection, oldest-first, advancing only the returned
    entry's cursor so nothing repeats and nothing is skipped."""

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

    @pytest.mark.asyncio
    async def test_fan_in_returns_oldest_published_skipping_unpublished(self, tmp_path):
        db = _make_db(tmp_path)
        self._publish(db, "silent", published=False)
        self._publish(db, "games")
        self._publish(db, "jobs")
        # The unpublished entry is the oldest of all, yet must stay invisible.
        self._write(db, "silent", "old", "oldest but unpublished")
        self._write(db, "games", "stardrift", "Stardrift Saga")
        self._write(db, "jobs", "role", "New engineering role")
        result = await ReadPublishedLatestTool(db, "notifier").execute(n=1)
        # Oldest PUBLISHED entry wins; the older unpublished one is never seen.
        assert "Stardrift Saga" in result.message
        assert "from `games`" in result.message
        assert "unpublished" not in result.message
        assert "New engineering role" not in result.message

    @pytest.mark.asyncio
    async def test_advances_only_returned_source_and_never_repeats(self, tmp_path):
        db = _make_db(tmp_path)
        self._publish(db, "games")
        self._publish(db, "jobs")
        self._write(db, "games", "g1", "game one")  # oldest
        self._write(db, "jobs", "j1", "job one")  # newer
        tool = ReadPublishedLatestTool(db, "notifier")
        first = await tool.execute(n=1)
        assert "game one" in first.message
        tool.commit_pending()
        # Only the returned source's cursor advanced; the scanned-but-unreturned
        # source is left untouched so its entry isn't skipped.
        assert db.cursors.get("notifier", "games") is not None
        assert db.cursors.get("notifier", "jobs") is None
        # Next read surfaces the next-oldest unseen, never the already-sent game.
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
