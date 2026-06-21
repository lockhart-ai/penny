"""Tests for MemoryStore, CursorStore, and MediaStore.

Exercises the data layer for the task/memory framework. Dedup, type
enforcement, log append, cursor monotonicity, and the similarity-based
`exists` check all run through these tests.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import (
    DedupThresholds,
    EntryInput,
    Inclusion,
    LogEntryInput,
    MemoryNotFoundError,
    MemoryTypeError,
    RecallMode,
)
from penny.database.memory._similarity import hybrid_rank_ids
from penny.llm.embeddings import deserialize_embedding, serialize_embedding
from penny.tools.memory_tools import MemoryMetadataTool


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — no migrations.

    Migration 0026 seeds three system log memories; these tests exercise
    the memory primitive in isolation and declare exactly the memories
    they need.
    """
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _unit_vec(idx: int, dim: int = 8) -> list[float]:
    """Return a sparse unit vector with a single 1.0 at position idx."""
    vec = [0.0] * dim
    vec[idx % dim] = 1.0
    return vec


class TestMemoryMetadata:
    def test_create_collection_and_fetch(self, tmp_path):
        db = _make_db(tmp_path)
        memory = db.memories.create_collection(
            "likes", "user positive preferences", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        assert memory.name == "likes"
        assert memory.type == "collection"
        assert memory.recall == "relevant"
        assert memory.archived is False

        fetched = db.memories.get("likes")
        assert fetched is not None
        assert fetched.description == "user positive preferences"

    def test_create_log_and_list(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log(
            "user-messages", "inbound user messages", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memories.create_collection(
            "dislikes", "user negative preferences", Inclusion.RELEVANT, RecallMode.RELEVANT
        )

        names = [s.name for s in db.memories.list_all()]
        assert names == ["dislikes", "user-messages"]

    def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("notes", "scratch", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.archive("notes")
        assert db.memories.get("notes").archived is True
        db.memories.unarchive("notes")
        assert db.memories.get("notes").archived is False

    def test_archive_missing_raises(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(MemoryNotFoundError):
            db.memories.archive("nope")

    def test_unicode_name_normalization(self, tmp_path):
        db = _make_db(tmp_path)
        # U+2011 NON-BREAKING HYPHEN — a unicode dash variant in the name
        db.memories.create_collection("board‑games", "tabletop", Inclusion.NEVER, RecallMode.RECENT)

        # Stored name is normalized to ASCII hyphen
        assert db.memories.get("board-games") is not None
        assert db.memories.get("board‑games").name == "board-games"

        # Write via unicode name lands in the same collection — no duplicate created
        db.memories.memory("board‑games").write(
            [EntryInput(key="catan", content="Catan is a gateway classic")],
            author="test",
        )
        assert len(db.memories.memory("board-games").read_all()) == 1
        assert len(db.memories.list_all()) == 1

    def test_collection_metadata_tool_returns_all_fields(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "board-games",
            "strategy board games",
            Inclusion.NEVER,
            RecallMode.RECENT,
            collector_interval_seconds=300,
            extraction_prompt="Browse for new board games and write entries.",
            published=True,
        )
        tool = MemoryMetadataTool(db)
        result = asyncio.run(tool.execute(memory="board-games"))
        assert "board-games" in result.message
        assert "collection" in result.message
        assert "strategy board games" in result.message
        assert "inclusion: never" in result.message
        assert "recall: recent" in result.message
        # published surfaces in metadata so the chat agent + quality can read notify-on-new.
        assert "published: True" in result.message
        assert "300s" in result.message
        assert "last collected: never" in result.message
        assert "Browse for new board games and write entries." in result.message
        assert "created:" in result.message
        assert "updated:" in result.message

    def test_collection_metadata_tool_no_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("plain", "no collector", Inclusion.NEVER, RecallMode.RECENT)
        tool = MemoryMetadataTool(db)
        result = asyncio.run(tool.execute(memory="plain"))
        assert "extraction prompt: none" in result.message
        # published is opt-in: a collection created without it defaults to silent.
        assert "published: False" in result.message

    def test_updated_at_advances_on_metadata_update(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("col", "desc", Inclusion.NEVER, RecallMode.RECENT)
        before = db.memories.get("col").updated_at
        db.memories.update_collection_metadata("col", description="new desc")
        after = db.memories.get("col").updated_at
        assert after >= before

    def test_collection_metadata_tool_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        tool = MemoryMetadataTool(db)
        result = asyncio.run(tool.execute(memory="nonexistent"))
        assert "not found" in result.message


class TestCollectionWrites:
    def test_write_returns_entry_ids(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )

        results = db.memories.memory("likes").write(
            [
                EntryInput(
                    key="dark roast coffee",
                    content="I love dark roast coffee",
                    key_embedding=_unit_vec(0),
                    content_embedding=_unit_vec(1),
                ),
                EntryInput(
                    key="cold brew",
                    content="cold brew is great",
                    key_embedding=_unit_vec(2),
                    content_embedding=_unit_vec(3),
                ),
            ],
            author="preference-extractor",
        )
        assert [r.outcome for r in results] == ["written", "written"]
        assert all(r.entry_id is not None for r in results)
        assert {r.key for r in results} == {"dark roast coffee", "cold brew"}

    def test_write_dedups_on_key_embedding(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        shared_key_vec = _unit_vec(0)

        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="dark roast",
                    content="dark roast",
                    key_embedding=shared_key_vec,
                    content_embedding=_unit_vec(1),
                )
            ],
            author="preference-extractor",
        )
        results = db.memories.memory("likes").write(
            [
                EntryInput(
                    key="dark roast coffee",
                    content="totally different body",
                    key_embedding=shared_key_vec,
                    content_embedding=_unit_vec(5),
                )
            ],
            author="preference-extractor",
        )
        assert results[0].outcome == "duplicate"
        assert results[0].entry_id is None
        # ``matched_key`` is the *existing* entry's key — what the model
        # should pivot to when calling ``update_entry``, not the rejected
        # candidate's own key.
        assert results[0].matched_key == "dark roast"
        assert len(db.memories.memory("likes").read_all()) == 1

    def test_write_dedups_on_content_embedding(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        shared_content = _unit_vec(4)

        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="first key",
                    content="same body",
                    key_embedding=_unit_vec(0),
                    content_embedding=shared_content,
                )
            ],
            author="preference-extractor",
        )
        results = db.memories.memory("likes").write(
            [
                EntryInput(
                    key="different key entirely",
                    content="same body",
                    key_embedding=_unit_vec(7),
                    content_embedding=shared_content,
                )
            ],
            author="preference-extractor",
        )
        assert results[0].outcome == "duplicate"

    def test_write_without_embeddings_always_accepts(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )

        first = db.memories.memory("likes").write(
            [EntryInput(key="a", content="hello")],
            author="chat",
        )
        second = db.memories.memory("likes").write(
            [EntryInput(key="b", content="hello")],
            author="chat",
        )
        assert first[0].outcome == "written"
        assert second[0].outcome == "written"
        assert len(db.memories.memory("likes").read_all()) == 2

    def test_update_replaces_content(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memories.memory("likes").write(
            [EntryInput(key="k", content="old body")],
            author="chat",
        )

        assert db.memories.memory("likes").update("k", "new body", "chat") == "ok"
        entries = db.memories.memory("likes").get("k")
        assert entries[0].content == "new body"

    def test_update_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        assert db.memories.memory("likes").update("missing", "body", "chat") == "not_found"

    def test_delete_removes_all_matching(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memories.memory("likes").write([EntryInput(key="k", content="a")], author="chat")
        assert db.memories.memory("likes").delete("k") == 1
        assert db.memories.memory("likes").get("k") == []

    def test_move_transfers_entry(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("unnotified", "pending", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.create_collection("notified", "done", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("unnotified").write(
            [EntryInput(key="thought-1", content="x")], author="thinking-agent"
        )

        outcome = db.memories.memory("unnotified").move("thought-1", "notified", author="notifier")
        assert outcome == "ok"
        assert db.memories.memory("unnotified").get("thought-1") == []
        assert len(db.memories.memory("notified").get("thought-1")) == 1

    def test_move_collision(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("a", "src", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.create_collection("b", "dst", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.memory("a").write([EntryInput(key="k", content="src")], author="chat")
        db.memories.memory("b").write([EntryInput(key="k", content="dst")], author="chat")

        assert db.memories.memory("a").move("k", "b", author="chat") == "collision"

    def test_move_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("a", "src", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.create_collection("b", "dst", Inclusion.NEVER, RecallMode.RECENT)
        assert db.memories.memory("a").move("missing", "b", author="chat") == "not_found"


class TestDegenerateContentRejection:
    """Write-time degenerate content guard — empty/punctuation, bare URLs, bail-out phrases."""

    def _make_collection(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "knowledge", "web summaries", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        return db

    def test_pure_punctuation_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="https://example.com", content="?")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert results[0].entry_id is None
        assert len(db.memories.memory("knowledge").read_all()) == 0

    def test_ellipsis_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="https://example.com", content="…")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert len(db.memories.memory("knowledge").read_all()) == 0

    def test_bare_url_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="https://example.com", content="https://example.com/path/to/page")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert len(db.memories.memory("knowledge").read_all()) == 0

    def test_bailout_phrase_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="https://example.com", content="Not sure")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert results[0].reason is not None
        assert len(db.memories.memory("knowledge").read_all()) == 0

    def test_bailout_phrase_case_insensitive(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="url", content="NOT SURE")],
            author="collector",
        )
        assert results[0].outcome == "rejected"

    def test_short_but_valid_content_accepted(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="anime", content="anime")],
            author="collector",
        )
        assert results[0].outcome == "written"

    def test_substantive_content_accepted(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [
                EntryInput(
                    key="https://example.com",
                    content="An in-depth article about the history of coffee roasting techniques.",
                )
            ],
            author="collector",
        )
        assert results[0].outcome == "written"

    def test_mixed_batch_partial_rejection(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [
                EntryInput(key="good", content="A detailed summary of the article content."),
                EntryInput(key="bad", content="?"),
                EntryInput(key="also-bad", content="Not sure"),
            ],
            author="collector",
        )
        outcomes = {r.key: r.outcome for r in results}
        assert outcomes["good"] == "written"
        assert outcomes["bad"] == "rejected"
        assert outcomes["also-bad"] == "rejected"
        assert len(db.memories.memory("knowledge").read_all()) == 1

    def test_rejection_does_not_count_as_written_for_notify(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="url", content="…")],
            author="collector",
        )
        assert not any(r.outcome == "written" for r in results)


class TestLogAppend:
    def test_append_multiple_entries_stored_in_order(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("chatter", "inbound", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("chatter").append(
            [
                LogEntryInput(content="hello"),
                LogEntryInput(content="are you there"),
            ],
            author="user",
        )

        entries = db.memories.memory("chatter").read_all()
        assert [e.content for e in entries] == ["hello", "are you there"]
        assert all(e.key is None for e in entries)
        assert all(e.author == "user" for e in entries)

    def test_append_to_collection_raises(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        with pytest.raises(MemoryTypeError):
            db.memories.memory("likes").append([LogEntryInput(content="nope")], author="user")

    def test_write_to_log_raises(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.ALWAYS, RecallMode.RECENT)
        with pytest.raises(MemoryTypeError):
            db.memories.memory("events").write(
                [EntryInput(key="k", content="v")],
                author="chat",
            )


class TestReads:
    def test_message_log_read_similar_finds_write_time_embedding(self, tmp_path):
        """user-/penny-messages are facades over messagelog: a message logged
        with a write-time embedding is immediately found by read_similar (the
        recall-freshness path — no restart/backfill needed)."""
        db = _make_db(tmp_path)
        log = PennyConstants.MEMORY_USER_MESSAGES_LOG
        # The facade dispatches on the marker row (seeded by migration in prod).
        db.memories.create_log(log, "inbound messages", Inclusion.ALWAYS, RecallMode.RELEVANT)
        direction = PennyConstants.MessageDirection.INCOMING
        db.messages.log_message(
            direction, "+1", "I love jazz", embedding=serialize_embedding([1.0, 0.0, 0.0])
        )
        db.messages.log_message(
            direction, "+1", "weather today", embedding=serialize_embedding([0.0, 1.0, 0.0])
        )

        hits = db.memories.memory(log).read_similar([1.0, 0.0, 0.0])

        assert [h.content for h in hits][:1] == ["I love jazz"]
        assert all(h.author == "user" for h in hits)

    def test_read_latest(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.ALWAYS, RecallMode.RECENT)
        for i in range(5):
            db.memories.memory("events").append([LogEntryInput(content=f"msg-{i}")], author="user")

        latest = db.memories.memory("events").newest_entries(3)
        assert [e.content for e in latest] == ["msg-4", "msg-3", "msg-2"]

        # offset paginates past the newest rows (second page of size 3).
        page_two = db.memories.memory("events").newest_entries(3, offset=3)
        assert [e.content for e in page_two] == ["msg-1", "msg-0"]

    def test_read_since(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.ALWAYS, RecallMode.RECENT)
        db.memories.memory("events").append([LogEntryInput(content="early")], author="user")
        mid = datetime.now(UTC)
        db.memories.memory("events").append([LogEntryInput(content="late")], author="user")

        after = db.memories.memory("events").read_since(mid)
        assert [e.content for e in after] == ["late"]

    def test_read_random_returns_all_when_k_exceeds(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write([EntryInput(key="a", content="1")], author="chat")
        db.memories.memory("likes").write([EntryInput(key="b", content="2")], author="chat")
        picked = db.memories.memory("likes").read_random(5)
        assert {e.key for e in picked} == {"a", "b"}

    def test_read_random_no_k_returns_all(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write([EntryInput(key="a", content="1")], author="chat")
        db.memories.memory("likes").write([EntryInput(key="b", content="2")], author="chat")
        assert {e.key for e in db.memories.memory("likes").read_random()} == {"a", "b"}

    def test_read_random_samples_subset_deterministically(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        for letter in ("a", "b", "c", "d"):
            db.memories.memory("likes").write(
                [EntryInput(key=letter, content=letter)], author="chat"
            )

        import penny.database.memory.objects as memory_store_mod

        captured: dict = {}

        def fake_sample(population, count):
            captured["population_size"] = len(population)
            captured["count"] = count
            return list(population[:count])

        monkeypatch.setattr(memory_store_mod.random, "sample", fake_sample)

        picked = db.memories.memory("likes").read_random(2)
        assert [e.key for e in picked] == ["a", "b"]
        assert captured == {"population_size": 4, "count": 2}

    def test_read_similar_orders_by_cosine(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0, 0.0]
        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="orth",
                    content="orthogonal",
                    content_embedding=[0.0, 0.0, 0.0, 1.0],
                ),
                EntryInput(
                    key="close",
                    content="halfway to anchor",
                    content_embedding=[0.5, 0.0, 0.87, 0.0],
                ),
                EntryInput(
                    key="exact",
                    content="anchor itself",
                    content_embedding=[1.0, 0.0, 0.0, 0.0],
                ),
            ],
            author="chat",
        )

        similar = db.memories.memory("likes").read_similar(anchor, k=2)
        assert [e.key for e in similar] == ["exact", "close"]

    def test_read_similar_respects_floor(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0]
        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="off-topic",
                    content="unrelated",
                    content_embedding=[0.0, 1.0],
                )
            ],
            author="chat",
        )

        assert db.memories.memory("likes").read_similar(anchor, k=5, floor=0.5) == []

    def test_read_similar_demotes_centroid_magnet(self, tmp_path):
        """Centroid-proxy penalty: an entry with high cosine to the anchor
        AND high projection on the corpus centroid is demoted below a less
        central entry whose cosine to the anchor is slightly lower."""
        db = _make_db(tmp_path)
        db.memories.create_collection("notes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        db.memories.memory("notes").write(
            [
                EntryInput(
                    key="magnet",
                    content="centroid magnet",
                    # cos to anchor = 0.9, but lives on the same axis as the crowd
                    content_embedding=[0.9, 0.436, 0.0],
                ),
                EntryInput(
                    key="specific",
                    content="orthogonal to crowd",
                    # cos to anchor = 0.85, orthogonal to the crowd axis
                    content_embedding=[0.85, 0.0, 0.527],
                ),
                EntryInput(
                    key="crowd1",
                    content="boilerplate one",
                    content_embedding=[0.5, 0.866, 0.0],
                ),
                EntryInput(
                    key="crowd2",
                    content="boilerplate two",
                    content_embedding=[0.4, 0.917, 0.0],
                ),
            ],
            author="chat",
        )

        similar = db.memories.memory("notes").read_similar(anchor)
        keys = [e.key for e in similar]
        # Without the centroid-proxy penalty 'magnet' (raw cos 0.9) would lead.
        # With the penalty, 'specific' (raw cos 0.85, far from the crowd) wins.
        assert "specific" in keys and "magnet" in keys
        assert keys.index("specific") < keys.index("magnet")

    def test_read_similar_suppresses_flat_noise_plateau(self, tmp_path):
        """Adaptive cluster gate: a corpus with no real cluster around the
        anchor (head_mean ≈ sample_mean) returns empty rather than emitting
        the noise floor."""
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        # Twenty entries with identical content embeddings — every adjusted
        # score is the same, so head_mean / sample_mean = 1.0, well below
        # CLUSTER_GATE (1.15).
        for i in range(20):
            db.memories.memory("events").append(
                [LogEntryInput(content=f"flat-{i}", content_embedding=[0.7, 0.7, 0.07])],
                author="chat",
            )

        assert db.memories.memory("events").read_similar(anchor) == []

    def test_read_similar_returns_real_cluster_above_noise(self, tmp_path):
        """Adaptive cluster gate: when the corpus has a real cluster around
        the anchor, the gate passes and only the cluster — not the noise
        floor — is returned."""
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        # 5 strong matches (cos ≈ 0.95)
        for i in range(5):
            db.memories.memory("events").append(
                [LogEntryInput(content=f"hit-{i}", content_embedding=[0.95, 0.31, 0.0])],
                author="chat",
            )
        # 15 weak matches (cos ≈ 0.3)
        for i in range(15):
            db.memories.memory("events").append(
                [LogEntryInput(content=f"miss-{i}", content_embedding=[0.3, 0.95, 0.0])],
                author="chat",
            )

        contents = [e.content for e in db.memories.memory("events").read_similar(anchor)]
        assert contents and all(c.startswith("hit-") for c in contents)

    def test_read_similar_hybrid_filters_low_info_entries(self, tmp_path):
        """Entries with fewer than ``MEMORY_RELEVANT_MIN_WORDS`` words are
        excluded from the corpus before scoring.

        Regression: short generic content (empty strings, "Hey!", "?")
        otherwise dominates cosine ranking on short keyword anchors,
        crowding out the real topical hits.
        """
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]

        # Junk entries that would otherwise dominate (1–4 words, high cosine)
        for junk in ("", "?", "Hey!", "hi penny"):
            db.memories.memory("events").append(
                [LogEntryInput(content=junk, content_embedding=[0.95, 0.31, 0.0])],
                author="chat",
            )
        # Real topical content (≥ 5 words, slightly lower cosine)
        real = "a real topical sentence about coffee"
        db.memories.memory("events").append(
            [LogEntryInput(content=real, content_embedding=[0.9, 0.43, 0.0])],
            author="chat",
        )

        hits = db.memories.memory("events").read_similar_hybrid([anchor], "coffee")
        contents = [e.content for e in hits]
        assert real in contents
        for junk in ("", "?", "Hey!", "hi penny"):
            assert junk not in contents

    def test_hybrid_rank_demotes_long_coincidental_match(self):
        """The lexical length penalty drops a long entry that merely *contains*
        the query terms below genuinely-relevant short entries.

        Two short entries each cover half the query and embed close to the
        anchor; a long entry covers the whole query (its big token set contains
        both terms) but embeds far from the anchor. Without length
        normalization the long entry's full coverage lifts it above a short
        on-topic entry; the sqrt penalty (``MEMORY_LEXICAL_LENGTH_B``) demotes
        it to last, where its weak cosine also puts it.
        """
        anchor = [1.0, 0.0, 0.0]
        fillers = " ".join(f"filler{i}" for i in range(60))
        docs = [
            (1, "alpha", [0.99, 0.141, 0.0]),  # on-topic, short, strongest cosine
            (2, f"alpha beta {fillers}", [0.6, 0.8, 0.0]),  # coincidental, long, weakest cosine
            (3, "beta", [0.95, 0.312, 0.0]),  # on-topic, short
        ]
        ranked = hybrid_rank_ids(
            [serialize_embedding(vec) for _, _, vec in docs],
            [content for _, content, _ in docs],
            [entry_id for entry_id, _, _ in docs],
            [anchor],
            "alpha beta",
        )
        assert ranked[-1] == 2  # the long coincidental entry ranks last, not lifted by coverage

    def test_keys_returns_unique_in_insertion_order(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write([EntryInput(key="first", content="1")], author="chat")
        db.memories.memory("likes").write([EntryInput(key="second", content="2")], author="chat")
        assert db.memories.memory("likes").keys() == ["first", "second"]


class TestExists:
    def test_exists_by_exact_key(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write(
            [EntryInput(key="dark roast", content="body")], author="chat"
        )

        assert db.memories.exists(["likes"], "dark roast", None, None) is True
        assert db.memories.exists(["likes"], "not seen", None, None) is False

    def test_exists_by_similarity_across_stores(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("unnotified", "pending", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.create_collection("notified", "done", Inclusion.RELEVANT, RecallMode.RELEVANT)
        shared = _unit_vec(2)
        db.memories.memory("notified").write(
            [
                EntryInput(
                    key="t1",
                    content="already notified",
                    content_embedding=shared,
                )
            ],
            author="notifier",
        )

        assert (
            db.memories.exists(
                ["unnotified", "notified"],
                key="t2",
                key_embedding=None,
                content_embedding=shared,
            )
            is True
        )


class TestCursorStore:
    def test_advance_and_get(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("user-messages", "inbound", Inclusion.RELEVANT, RecallMode.RELEVANT)
        now = datetime.now(UTC)
        db.cursors.advance_committed("preference-extractor", "user-messages", now)

        assert db.cursors.get("preference-extractor", "user-messages") == now

    def test_advance_is_monotonic(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("user-messages", "inbound", Inclusion.RELEVANT, RecallMode.RELEVANT)
        later = datetime.now(UTC)
        earlier = later - timedelta(minutes=5)

        db.cursors.advance_committed("preference-extractor", "user-messages", later)
        db.cursors.advance_committed("preference-extractor", "user-messages", earlier)

        assert db.cursors.get("preference-extractor", "user-messages") == later

    def test_missing_cursor_returns_none(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.cursors.get("preference-extractor", "user-messages") is None


class TestMediaStore:
    def test_put_and_get_roundtrip(self, tmp_path):
        db = _make_db(tmp_path)
        media_id = db.media.put(
            b"binary payload",
            "image/png",
            source_url="https://x.test/a.png",
            title="A Page",
            embedding=serialize_embedding([1.0, 0.0, 0.0]),
        )
        entry = db.media.get(media_id)

        assert entry is not None
        assert entry.data == b"binary payload"
        assert entry.mime_type == "image/png"
        assert entry.source_url == "https://x.test/a.png"
        assert entry.title == "A Page"
        assert entry.embedding is not None
        assert deserialize_embedding(entry.embedding) == [1.0, 0.0, 0.0]

    def test_get_missing_returns_none(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.media.get(99999) is None

    def test_find_nearest_returns_closest(self, tmp_path):
        db = _make_db(tmp_path)
        db.media.put(
            b"a", "image/png", source_url="https://a", embedding=serialize_embedding([1.0, 0.0])
        )
        db.media.put(
            b"b", "image/png", source_url="https://b", embedding=serialize_embedding([0.0, 1.0])
        )

        match = db.media.find_nearest([0.9, 0.1])
        assert match is not None
        assert match.source_url == "https://a"

    def test_find_nearest_no_floor_returns_even_weak_match(self, tmp_path):
        """The single nearest image always wins — a reply is never left
        imageless even when the cosine is poor."""
        db = _make_db(tmp_path)
        db.media.put(
            b"a", "image/png", source_url="https://a", embedding=serialize_embedding([1.0, 0.0])
        )
        match = db.media.find_nearest([0.0, 1.0])
        assert match is not None
        assert match.source_url == "https://a"

    def test_find_nearest_returns_none_when_no_embedded_media(self, tmp_path):
        db = _make_db(tmp_path)
        db.media.put(b"a", "image/png", source_url="https://a")  # no embedding
        assert db.media.find_nearest([1.0, 0.0]) is None


class TestWriteTypeEnforcement:
    def test_write_requires_collection(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.ALWAYS, RecallMode.RECENT)
        with pytest.raises(MemoryTypeError):
            db.memories.memory("events").write([EntryInput(key="k", content="v")], author="chat")

    def test_write_on_missing_store_raises(self, tmp_path):
        db = _make_db(tmp_path)
        # Dispatch surfaces a missing memory as ``None`` — there is no object to
        # write through, which is how callers detect 'not found'.
        assert db.memories.memory("nope") is None

    def test_dedup_thresholds_configurable(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="a",
                    content="body",
                    key_embedding=_unit_vec(0),
                    content_embedding=_unit_vec(1),
                )
            ],
            author="chat",
        )

        strict = DedupThresholds(
            key_tcr_strict=0.99,
            key_tcr_relaxed=0.99,
            key_sim_strict=0.99,
            key_sim_relaxed=0.99,
            content_sim_strict=0.99,
            content_sim_relaxed=0.99,
        )
        result = db.memories.memory("likes").write(
            [
                EntryInput(
                    key="b",
                    content="slightly different body",
                    key_embedding=_unit_vec(7),
                    content_embedding=_unit_vec(6),
                )
            ],
            author="chat",
            thresholds=strict,
        )
        assert result[0].outcome == "written"


class TestDedupSignals:
    """The three-signal rule: any strict hit OR any two relaxed hits → duplicate."""

    def test_tcr_strict_alone_rejects(self, tmp_path):
        """Full token-subset on keys fires without any embeddings."""
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write(
            [EntryInput(key="dark roast", content="first body")],
            author="chat",
        )
        result = db.memories.memory("likes").write(
            [EntryInput(key="dark roast coffee", content="second body")],
            author="chat",
        )
        assert result[0].outcome == "duplicate"

    def test_tcr_relaxed_alone_does_not_fire(self, tmp_path):
        """TCR 2/3 with no other signal is not enough on its own."""
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write(
            [EntryInput(key="applied ai conference", content="first")],
            author="chat",
        )
        result = db.memories.memory("likes").write(
            [EntryInput(key="applied ai conf", content="second")],
            author="chat",
        )
        assert result[0].outcome == "written"

    def test_two_relaxed_signals_reject(self, tmp_path):
        """TCR 2/3 plus a relaxed content-cosine hit (~0.80) → duplicate."""
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="applied ai conference",
                    content="first body",
                    content_embedding=[1.0, 0.0],
                )
            ],
            author="chat",
        )
        # cos([1, 0], [0.80, 0.60]) = 0.80 → relaxed content hit, not strict.
        # TCR("applied ai conf", "applied ai conference") = 2/3 → relaxed key hit.
        # Two relaxed hits → duplicate.
        result = db.memories.memory("likes").write(
            [
                EntryInput(
                    key="applied ai conf",
                    content="second body",
                    content_embedding=[0.80, 0.60],
                )
            ],
            author="chat",
        )
        assert result[0].outcome == "duplicate"

    def test_single_relaxed_signal_passes(self, tmp_path):
        """One signal at relaxed level only (no second signal) is not enough."""
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="coffee roast",
                    content="first",
                    content_embedding=[1.0, 0.0],
                )
            ],
            author="chat",
        )
        result = db.memories.memory("likes").write(
            [
                EntryInput(
                    key="tea brewing",
                    content="second",
                    content_embedding=[0.80, 0.60],
                )
            ],
            author="chat",
        )
        assert result[0].outcome == "written"


class TestEmbeddingBackfill:
    """Startup backfill targets recall-relevant, non-archived entries only.

    Migration-seeded content (skills) and other rows inserted via raw SQL
    arrive with NULL embeddings; the backfill embeds the ones that are
    actually reachable by similarity and skips bulk ``recall=off`` logs
    (``collector-runs``) that would otherwise be a huge pointless embed.
    """

    def test_scopes_to_relevant_unarchived_and_persists(self, tmp_path):
        db = _make_db(tmp_path)
        # A relevant collection (skills-like): entries SHOULD be embedded.
        db.memories.create_collection(
            "skills", "workflow patterns", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memories.memory("skills").write(
            [EntryInput(key="Do X when Y", content="TRIGGER ... STEPS ...")],
            author="system",
        )
        # An off log (collector-runs-like): never surfaces → must be skipped.
        db.memories.create_log("audit-log", "cycle log", Inclusion.NEVER, RecallMode.RECENT)
        db.memories.memory("audit-log").append(
            [LogEntryInput(content="cycle summary")],
            author="collector",
        )
        # An archived collection: never surfaces → must be skipped.
        db.memories.create_collection(
            "old-trip", "archived", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        db.memories.memory("old-trip").write(
            [EntryInput(key="spot", content="some place")],
            author="chat",
        )
        db.memories.archive("old-trip")

        pending = db.memories.get_entries_without_embeddings(limit=100)
        # Only the skills entry qualifies — off-log and archived are excluded.
        assert [e.memory_name for e in pending] == ["skills"]
        assert pending[0].content_embedding is None

        # Persist embeddings, then confirm it drops out of the pending set.
        entry_id = pending[0].id
        assert entry_id is not None
        db.memories.set_entry_embeddings(
            entry_id,
            key_embedding=_unit_vec(0),
            content_embedding=_unit_vec(1),
        )
        assert db.memories.get_entries_without_embeddings(limit=100) == []
        rows = db.memories.memory("skills").read_latest()
        assert rows[0].content_embedding is not None
        assert rows[0].key_embedding is not None
