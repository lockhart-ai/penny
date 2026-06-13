"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from penny.database import Database
from penny.llm.client import LlmClient
from penny.tools.memory_tools import (
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionDeleteEntryTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMergeTool,
    CollectionMoveTool,
    CollectionReadRandomTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    DoneTool,
    ExistsTool,
    LogAppendTool,
    LogCreateTool,
    LogReadNextTool,
    LogReadRecentTool,
    ReadLatestTool,
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
        )
        # Structured echo: collection name, interval, recall, prompt body all surfaced
        # so the chat agent can confirm-back without confabulating.
        assert "Created collection 'likes'" in result
        assert "interval: 3600s (1h)" in result
        assert "recall: relevant" in result
        assert "Extract user likes" in result  # extraction_prompt is echoed verbatim
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["likes"].type == "collection"
        assert memories["likes"].recall == "relevant"
        assert memories["likes"].description == "positive prefs"
        assert memories["likes"].collector_interval_seconds == 3600

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
        )
        result = await CollectionCreateTool(db, None).execute(
            name="ai-news",
            description="second slightly different",
            inclusion="relevant",
            recall="relevant",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        assert "already exists" in result
        assert "ai-news" in result
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
        assert "already exists" in result
        assert "events" in result

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
        )
        assert "too short" in result
        assert "minimum" in result
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
        )
        assert "Created" in result

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
        )
        result = await CollectionUpdateTool(db, None).execute(name="notes", extraction_prompt="yes")
        assert "too short" in result
        # Update rejected — original prompt preserved unchanged
        assert db.memories.get("notes").extraction_prompt == original_prompt


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
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast", "content": "loves dark roast"},
                {"key": "cold brew", "content": "enjoys cold brew"},
            ],
        )
        assert "Wrote 2 entries to 'likes'" in result
        latest = await ReadLatestTool(db).execute(memory="likes")
        assert "dark roast" in latest
        assert "cold brew" in latest

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
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        result = await write.execute(
            memory="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        assert "Rejected as duplicates" in result
        # The candidate's own key is named, *and* the existing key it
        # collided with — gives the model enough context to pivot to
        # update_entry instead of silently dropping fresher info.
        assert "dark roast coffee" in result
        assert "matches existing 'dark roast'" in result
        assert "update_entry" in result

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
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "hello"}]
        )
        assert "hello" in await CollectionGetTool(db).execute(memory="likes", key="k")
        missing = await CollectionGetTool(db).execute(memory="likes", key="absent")
        assert "not found" in missing

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
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "first", "content": "1"}])
        await write.execute(memory="likes", entries=[{"key": "second", "content": "2"}])
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing == "- first\n- second"

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
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "a", "content": "1"}])
        rendered = await CollectionReadRandomTool(db).execute(memory="likes", k=5)
        assert "[a] 1" in rendered

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
        )
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "coffee", "content": "loves coffee"}]
        )
        # Anchor shares the "coffee" word with the entry — the bag-of-words
        # mock embedding gives meaningful cosine, so the entry survives the
        # adaptive cutoff in ``read_similar``.
        rendered = await ReadSimilarTool(db, client).execute(memory="likes", anchor="coffee please")
        assert "coffee" in rendered

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
        )
        result = await ReadSimilarTool(db, None).execute(memory="likes", anchor="whatever")
        assert "similarity search unavailable" in result


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
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "old"}]
        )
        result = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "Updated 'k' in 'likes'" in result
        fetched = await CollectionGetTool(db).execute(memory="likes", key="k")
        assert "new" in fetched

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
        )
        result = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "not found" in result

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
        )
        await CollectionCreateTool(db, None).execute(
            name="notified",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="unnotified", entries=[{"key": "t1", "content": "x"}]
        )
        result = await CollectionMoveTool(db, author="test").execute(
            key="t1", from_memory="unnotified", to_memory="notified"
        )
        assert "Moved 't1'" in result

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
        )
        await CollectionCreateTool(db, None).execute(
            name="b",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="a", entries=[{"key": "k", "content": "src"}])
        await write.execute(memory="b", entries=[{"key": "k", "content": "dst"}])
        result = await CollectionMoveTool(db, author="test").execute(
            key="k", from_memory="a", to_memory="b"
        )
        assert "already has a 'k' entry" in result

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
        )
        assert "Archived 'likes'" in await CollectionArchiveTool(db).execute(memory="likes")
        assert "Unarchived 'likes'" in await CollectionUnarchiveTool(db).execute(memory="likes")


class TestLogTools:
    @pytest.mark.asyncio
    async def test_append_and_read_latest(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")
        rendered = await ReadLatestTool(db).execute(memory="events")
        # Leads with a count + source header so the model reads the body as
        # fetched data; entries are numbered newest-first.
        assert rendered.splitlines() == [
            "2 entries from `events` (most recent first):",
            "1. second",
            "2. first",
        ]

    @pytest.mark.asyncio
    async def test_read_recent_window(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="hello"
        )
        rendered = await LogReadRecentTool(db).execute(memory="events", window_seconds=3600)
        assert "hello" in rendered

    @pytest.mark.asyncio
    async def test_read_recent_default_window(self, tmp_path, mock_llm):
        """log_read_recent is callable with only ``memory`` — window_seconds defaults to 3600."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="hello"
        )
        rendered = await LogReadRecentTool(db).execute(memory="events")
        assert "hello" in rendered

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
        assert "coffee is great" in rendered

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

        read_next = LogReadNextTool(db, agent_name="extractor")
        rendered = await read_next.execute(memory="events")

        assert "first" in rendered
        assert "second" in rendered

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

        read_next = LogReadNextTool(db, agent_name="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # A new instance after commit should see no entries (cursor caught up).
        fresh = LogReadNextTool(db, agent_name="extractor")
        rendered = await fresh.execute(memory="events")
        assert rendered == "(no entries)"

    @pytest.mark.asyncio
    async def test_discard_pending_leaves_cursor_unchanged(self, tmp_path, mock_llm):
        """discard_pending drops the in-memory state without touching the DB cursor."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, None).execute(
            name="events", description="x", inclusion="always", recall="recent"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")

        read_next = LogReadNextTool(db, agent_name="extractor")
        await read_next.execute(memory="events")
        read_next.discard_pending()

        # Cursor still at None; a new read sees the same entries.
        fresh = LogReadNextTool(db, agent_name="extractor")
        rendered = await fresh.execute(memory="events")
        assert "first" in rendered

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
        n_entries = PennyConstants.LOG_READ_NEXT_INITIAL_LIMIT + 5
        for i in range(n_entries):
            await append.execute(memory="events", content=f"entry-{i:02d}")

        read_next = LogReadNextTool(db, agent_name="brand-new-collector")
        rendered = await read_next.execute(memory="events")

        # Exactly the latest N entries — entry-(n-N) through entry-(n-1)
        # should appear; older entries should not.
        for i in range(n_entries - PennyConstants.LOG_READ_NEXT_INITIAL_LIMIT, n_entries):
            assert f"entry-{i:02d}" in rendered
        # The first 5 entries must be excluded
        assert "entry-00" not in rendered
        assert "entry-04" not in rendered

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

        read_next = LogReadNextTool(db, agent_name="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # New entries arrive
        await append.execute(memory="events", content="new-after-cursor")

        fresh = LogReadNextTool(db, agent_name="extractor")
        rendered = await fresh.execute(memory="events")
        assert "new-after-cursor" in rendered
        # Old entries excluded by the bound stay excluded
        assert "old-00" not in rendered

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

        agent_a = LogReadNextTool(db, agent_name="a")
        await agent_a.execute(memory="events")
        agent_a.commit_pending()

        # Agent B has its own cursor and still sees the entry.
        agent_b = LogReadNextTool(db, agent_name="b")
        rendered = await agent_b.execute(memory="events")
        assert "hello" in rendered


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
        )
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "dark roast", "content": "body"}]
        )
        result = await ExistsTool(db, client).execute(
            memories=["likes"], key="dark roast", content="body"
        )
        assert result == "yes"

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
        )
        result = await ExistsTool(db, _make_llm_client(mock_llm)).execute(
            memories=["likes"], key="not there", content="nothing"
        )
        assert result == "no"

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
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        # Non-breaking hyphen U+2011 in the memory name — model output
        # observed in the wild.
        result = await write.execute(
            memory="board‑games",
            entries=[{"key": "k", "content": "v"}],
        )
        assert "Wrote 1 entry to 'board-games'" in result

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
        assert result == "yes"

    @pytest.mark.asyncio
    async def test_done_returns_structured_summary(self):
        result = await DoneTool().execute(success=True, summary="wrote 3 entries")
        assert "wrote 3 entries" in result
        assert "success" in result

    @pytest.mark.asyncio
    async def test_done_no_op_marker(self):
        result = await DoneTool().execute(success=False, summary="no new matches")
        assert "no new matches" in result
        assert "no-op" in result

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
        )
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="preference-extractor"
        ).execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        rows = db.memories.get_entry("likes", "k")
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
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="src", entries=[{"key": "a", "content": "alpha"}])
        await write.execute(memory="src", entries=[{"key": "b", "content": "beta"}])

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "2 moved" in result
        assert "archived" in result
        assert db.memories.get("src").archived is True
        assert len(db.memories.read_all("dst")) == 2
        assert len(db.memories.read_all("src")) == 0

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
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="src", entries=[{"key": "shared", "content": "from src"}])
        await write.execute(memory="src", entries=[{"key": "unique", "content": "only in src"}])
        await write.execute(memory="dst", entries=[{"key": "shared", "content": "already in dst"}])

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "1 moved" in result
        assert "1 dropped" in result
        dst_entries = db.memories.read_all("dst")
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
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "archived" in result
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
        assert result.startswith("✅")
        assert "wrote 3 entries" in result

    @pytest.mark.asyncio
    async def test_failure_returns_x_and_summary(self):
        collector = self._MockCollector((False, "Collector cycle complete. max steps exceeded"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="likes")
        assert result.startswith("❌")
        assert "max steps exceeded" in result

    @pytest.mark.asyncio
    async def test_validation_error_returns_x_and_error_message(self):
        collector = self._MockCollector((False, "Collection 'missing' not found."))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="missing")
        assert result.startswith("❌")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_unicode_dash_in_memory_name_normalized(self):
        """MemoryNameArgs normalises Unicode dashes before passing to run_for."""
        collector = self._MockCollector((True, "Collector cycle complete. wrote 1 entry"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        await tool.execute(memory="board‑games")  # U+2011 non-breaking hyphen
        assert collector.called_with == "board-games"


class TestFactory:
    """Two distinct surfaces, mutually exclusive: chat (lifecycle + reads, no
    entry mutations) vs collector (reads + entry mutations pinned to scope).
    """

    def test_chat_surface_has_lifecycle_and_reads_only(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        tools = build_memory_tools(db, _make_llm_client(mock_llm), agent_name="chat")
        names = {tool.name for tool in tools}
        assert names == {
            # Lifecycle (chat owns the shape of memory)
            "collection_create",
            "collection_update",
            "collection_merge",
            "collection_archive",
            "collection_unarchive",
            "log_create",
            # Reads
            "collection_get",
            "collection_read_random",
            "collection_keys",
            "collection_metadata",
            "log_read_recent",
            "log_read_next",
            "read_latest",
            "read_similar",
            "exists",
        }

    def test_collector_surface_has_scoped_writes_and_reads(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="collector", scope="likes"
        )
        names = {tool.name for tool in tools}
        # Reads + entry mutations (pinned to scope) + log_append; no lifecycle
        assert names == {
            "collection_get",
            "collection_read_random",
            "collection_keys",
            "collection_metadata",
            "log_read_recent",
            "log_read_next",
            "read_latest",
            "read_similar",
            "exists",
            "collection_write",
            "update_entry",
            "collection_delete_entry",
            "collection_move",
            "log_append",
        }


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
        )
        await CollectionCreateTool(db, None).execute(
            name="dislikes",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="dislikes", entries=[{"key": "k", "content": "v"}])

        assert "Refused" in result and "likes" in result and "dislikes" in result
        # And nothing was actually written
        assert db.memories.get_entry("dislikes", "k") == []

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
        )

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        assert "Wrote 1 entry" in result
        assert db.memories.get_entry("likes", "k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_update_entry_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        update = UpdateEntryTool(db, author="collector:likes", scope="likes")
        result = await update.execute(memory="dislikes", key="k", content="v")
        assert "Refused" in result

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
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="t").execute(
            memory="src", entries=[{"key": "k", "content": "v"}]
        )

        move = CollectionMoveTool(db, author="collector:dst", scope="dst")
        result = await move.execute(key="k", from_memory="src", to_memory="dst")
        assert "Moved 'k'" in result
        assert db.memories.get_entry("dst", "k")[0].content == "v"

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
        )
        await CollectionCreateTool(db, None).execute(
            name="dst",
            description="x",
            inclusion="never",
            recall="recent",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="t").execute(
            memory="src", entries=[{"key": "k", "content": "v"}]
        )

        move = CollectionMoveTool(db, author="collector:dst", scope="dst")
        result = await move.execute(key="k", from_memory="src")
        assert "Moved 'k'" in result
        assert db.memories.get_entry("dst", "k")[0].content == "v"

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
        assert "Refused" in result and "src" in result and "dst" in result

    @pytest.mark.asyncio
    async def test_scoped_delete_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        delete = CollectionDeleteEntryTool(db, scope="likes")
        result = await delete.execute(memory="dislikes", key="k")
        assert "Refused" in result
