"""Tests for MemoryStore, CursorStore, and MediaStore.

Exercises the data layer for the task/memory framework. Dedup, type
enforcement, log append, cursor monotonicity, and the similarity-based
`exists` check all run through these tests.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from penny.database import Database
from penny.database.memory_store import (
    DedupThresholds,
    EntryInput,
    LogEntryInput,
    MemoryNotFoundError,
    MemoryTypeError,
    RecallMode,
)
from penny.tools.memory_tools import CollectionMetadataTool


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
            "likes", "user positive preferences", RecallMode.RELEVANT
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
        db.memories.create_log("user-messages", "inbound user messages", RecallMode.RELEVANT)
        db.memories.create_collection("dislikes", "user negative preferences", RecallMode.RELEVANT)

        names = [s.name for s in db.memories.list_all()]
        assert names == ["dislikes", "user-messages"]

    def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("notes", "scratch", RecallMode.OFF)
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
        # U+2011 NON-BREAKING HYPHEN — the exact character from the bug report
        db.memories.create_collection("prague‑highlights", "travel", RecallMode.OFF)

        # Stored name is normalized to ASCII hyphen
        assert db.memories.get("prague-highlights") is not None
        assert db.memories.get("prague‑highlights").name == "prague-highlights"

        # Write via unicode name lands in the same collection — no duplicate created
        db.memories.write(
            "prague‑highlights",
            [EntryInput(key="castle", content="Prague castle was stunning")],
            author="test",
        )
        assert len(db.memories.read_all("prague-highlights")) == 1
        assert len(db.memories.list_all()) == 1

    def test_collection_metadata_tool_returns_all_fields(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "prague-highlights",
            "travel highlights",
            RecallMode.OFF,
            collector_interval_seconds=300,
            extraction_prompt="Browse for Prague spots and write entries.",
        )
        tool = CollectionMetadataTool(db)
        result = asyncio.run(tool.execute(memory="prague-highlights"))
        assert "prague-highlights" in result
        assert "collection" in result
        assert "travel highlights" in result
        assert "off" in result
        assert "300s" in result
        assert "never" in result  # last collected
        assert "Browse for Prague spots and write entries." in result
        assert "created:" in result
        assert "updated:" in result

    def test_collection_metadata_tool_no_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("plain", "no collector", RecallMode.OFF)
        tool = CollectionMetadataTool(db)
        result = asyncio.run(tool.execute(memory="plain"))
        assert "extraction prompt: none" in result

    def test_updated_at_advances_on_metadata_update(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("col", "desc", RecallMode.OFF)
        before = db.memories.get("col").updated_at
        db.memories.update_collection_metadata("col", description="new desc")
        after = db.memories.get("col").updated_at
        assert after >= before

    def test_collection_metadata_tool_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        tool = CollectionMetadataTool(db)
        result = asyncio.run(tool.execute(memory="nonexistent"))
        assert "not found" in result


class TestCollectionWrites:
    def test_write_returns_entry_ids(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)

        results = db.memories.write(
            "likes",
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
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        shared_key_vec = _unit_vec(0)

        db.memories.write(
            "likes",
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
        results = db.memories.write(
            "likes",
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
        assert len(db.memories.read_all("likes")) == 1

    def test_write_dedups_on_content_embedding(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        shared_content = _unit_vec(4)

        db.memories.write(
            "likes",
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
        results = db.memories.write(
            "likes",
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
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)

        first = db.memories.write(
            "likes",
            [EntryInput(key="a", content="hello")],
            author="chat",
        )
        second = db.memories.write(
            "likes",
            [EntryInput(key="b", content="hello")],
            author="chat",
        )
        assert first[0].outcome == "written"
        assert second[0].outcome == "written"
        assert len(db.memories.read_all("likes")) == 2

    def test_update_replaces_content(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        db.memories.write(
            "likes",
            [EntryInput(key="k", content="old body")],
            author="chat",
        )

        assert db.memories.update("likes", "k", "new body", "chat") == "ok"
        entries = db.memories.get_entry("likes", "k")
        assert entries[0].content == "new body"

    def test_update_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        assert db.memories.update("likes", "missing", "body", "chat") == "not_found"

    def test_delete_removes_all_matching(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "positive prefs", RecallMode.RELEVANT)
        db.memories.write("likes", [EntryInput(key="k", content="a")], author="chat")
        assert db.memories.delete("likes", "k") == 1
        assert db.memories.get_entry("likes", "k") == []

    def test_move_transfers_entry(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("unnotified", "pending", RecallMode.OFF)
        db.memories.create_collection("notified", "done", RecallMode.RELEVANT)
        db.memories.write(
            "unnotified", [EntryInput(key="thought-1", content="x")], author="thinking-agent"
        )

        outcome = db.memories.move("thought-1", "unnotified", "notified", author="notifier")
        assert outcome == "ok"
        assert db.memories.get_entry("unnotified", "thought-1") == []
        assert len(db.memories.get_entry("notified", "thought-1")) == 1

    def test_move_collision(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("a", "src", RecallMode.OFF)
        db.memories.create_collection("b", "dst", RecallMode.OFF)
        db.memories.write("a", [EntryInput(key="k", content="src")], author="chat")
        db.memories.write("b", [EntryInput(key="k", content="dst")], author="chat")

        assert db.memories.move("k", "a", "b", author="chat") == "collision"

    def test_move_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("a", "src", RecallMode.OFF)
        db.memories.create_collection("b", "dst", RecallMode.OFF)
        assert db.memories.move("missing", "a", "b", author="chat") == "not_found"


class TestDegenerateContentRejection:
    """Write-time degenerate content guard — empty/punctuation, bare URLs, bail-out phrases."""

    def _make_collection(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("knowledge", "web summaries", RecallMode.RELEVANT)
        return db

    def test_pure_punctuation_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="https://example.com", content="?")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert results[0].entry_id is None
        assert len(db.memories.read_all("knowledge")) == 0

    def test_ellipsis_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="https://example.com", content="…")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert len(db.memories.read_all("knowledge")) == 0

    def test_bare_url_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="https://example.com", content="https://example.com/path/to/page")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert len(db.memories.read_all("knowledge")) == 0

    def test_bailout_phrase_rejected(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="https://example.com", content="Not sure")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert results[0].reason is not None
        assert len(db.memories.read_all("knowledge")) == 0

    def test_bailout_phrase_case_insensitive(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="url", content="NOT SURE")],
            author="collector",
        )
        assert results[0].outcome == "rejected"

    def test_short_but_valid_content_accepted(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="anime", content="anime")],
            author="collector",
        )
        assert results[0].outcome == "written"

    def test_substantive_content_accepted(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
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
        results = db.memories.write(
            "knowledge",
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
        assert len(db.memories.read_all("knowledge")) == 1

    def test_rejection_does_not_count_as_written_for_notify(self, tmp_path):
        db = self._make_collection(tmp_path)
        results = db.memories.write(
            "knowledge",
            [EntryInput(key="url", content="…")],
            author="collector",
        )
        assert not any(r.outcome == "written" for r in results)


class TestLogAppend:
    def test_append_multiple_entries_stored_in_order(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("user-messages", "inbound", RecallMode.RELEVANT)
        db.memories.append(
            "user-messages",
            [
                LogEntryInput(content="hello"),
                LogEntryInput(content="are you there"),
            ],
            author="user",
        )

        entries = db.memories.read_all("user-messages")
        assert [e.content for e in entries] == ["hello", "are you there"]
        assert all(e.key is None for e in entries)
        assert all(e.author == "user" for e in entries)

    def test_append_to_collection_raises(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        with pytest.raises(MemoryTypeError):
            db.memories.append("likes", [LogEntryInput(content="nope")], author="user")

    def test_write_to_log_raises(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", RecallMode.RECENT)
        with pytest.raises(MemoryTypeError):
            db.memories.write(
                "events",
                [EntryInput(key="k", content="v")],
                author="chat",
            )


class TestReads:
    def test_read_latest(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", RecallMode.RECENT)
        for i in range(5):
            db.memories.append("events", [LogEntryInput(content=f"msg-{i}")], author="user")

        latest = db.memories.read_latest("events", 3)
        assert [e.content for e in latest] == ["msg-4", "msg-3", "msg-2"]

    def test_read_since(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", RecallMode.RECENT)
        db.memories.append("events", [LogEntryInput(content="early")], author="user")
        mid = datetime.now(UTC)
        db.memories.append("events", [LogEntryInput(content="late")], author="user")

        after = db.memories.read_since("events", mid)
        assert [e.content for e in after] == ["late"]

    def test_read_random_returns_all_when_k_exceeds(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write("likes", [EntryInput(key="a", content="1")], author="chat")
        db.memories.write("likes", [EntryInput(key="b", content="2")], author="chat")
        picked = db.memories.read_random("likes", 5)
        assert {e.key for e in picked} == {"a", "b"}

    def test_read_random_no_k_returns_all(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write("likes", [EntryInput(key="a", content="1")], author="chat")
        db.memories.write("likes", [EntryInput(key="b", content="2")], author="chat")
        assert {e.key for e in db.memories.read_random("likes")} == {"a", "b"}

    def test_read_random_samples_subset_deterministically(self, tmp_path, monkeypatch):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        for letter in ("a", "b", "c", "d"):
            db.memories.write("likes", [EntryInput(key=letter, content=letter)], author="chat")

        import penny.database.memory_store as memory_store_mod

        captured: dict = {}

        def fake_sample(population, count):
            captured["population_size"] = len(population)
            captured["count"] = count
            return list(population[:count])

        monkeypatch.setattr(memory_store_mod.random, "sample", fake_sample)

        picked = db.memories.read_random("likes", 2)
        assert [e.key for e in picked] == ["a", "b"]
        assert captured == {"population_size": 4, "count": 2}

    def test_read_similar_orders_by_cosine(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0, 0.0]
        db.memories.write(
            "likes",
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

        similar = db.memories.read_similar("likes", anchor, k=2)
        assert [e.key for e in similar] == ["exact", "close"]

    def test_read_similar_respects_floor(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0]
        db.memories.write(
            "likes",
            [
                EntryInput(
                    key="off-topic",
                    content="unrelated",
                    content_embedding=[0.0, 1.0],
                )
            ],
            author="chat",
        )

        assert db.memories.read_similar("likes", anchor, k=5, floor=0.5) == []

    def test_read_similar_demotes_centroid_magnet(self, tmp_path):
        """Centroid-proxy penalty: an entry with high cosine to the anchor
        AND high projection on the corpus centroid is demoted below a less
        central entry whose cosine to the anchor is slightly lower."""
        db = _make_db(tmp_path)
        db.memories.create_collection("notes", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        db.memories.write(
            "notes",
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

        similar = db.memories.read_similar("notes", anchor)
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
        db.memories.create_log("events", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        # Twenty entries with identical content embeddings — every adjusted
        # score is the same, so head_mean / sample_mean = 1.0, well below
        # CLUSTER_GATE (1.15).
        for i in range(20):
            db.memories.append(
                "events",
                [LogEntryInput(content=f"flat-{i}", content_embedding=[0.7, 0.7, 0.07])],
                author="chat",
            )

        assert db.memories.read_similar("events", anchor) == []

    def test_read_similar_returns_real_cluster_above_noise(self, tmp_path):
        """Adaptive cluster gate: when the corpus has a real cluster around
        the anchor, the gate passes and only the cluster — not the noise
        floor — is returned."""
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        # 5 strong matches (cos ≈ 0.95)
        for i in range(5):
            db.memories.append(
                "events",
                [LogEntryInput(content=f"hit-{i}", content_embedding=[0.95, 0.31, 0.0])],
                author="chat",
            )
        # 15 weak matches (cos ≈ 0.3)
        for i in range(15):
            db.memories.append(
                "events",
                [LogEntryInput(content=f"miss-{i}", content_embedding=[0.3, 0.95, 0.0])],
                author="chat",
            )

        contents = [e.content for e in db.memories.read_similar("events", anchor)]
        assert contents and all(c.startswith("hit-") for c in contents)

    def test_read_similar_hybrid_filters_low_info_entries(self, tmp_path):
        """Entries with fewer than ``MEMORY_RELEVANT_MIN_WORDS`` words are
        excluded from the corpus before scoring.

        Regression: short generic content (empty strings, "Hey!", "?")
        otherwise dominates cosine ranking on short keyword anchors,
        crowding out the real topical hits.
        """
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]

        # Junk entries that would otherwise dominate (1–4 words, high cosine)
        for junk in ("", "?", "Hey!", "hi penny"):
            db.memories.append(
                "events",
                [LogEntryInput(content=junk, content_embedding=[0.95, 0.31, 0.0])],
                author="chat",
            )
        # Real topical content (≥ 5 words, slightly lower cosine)
        real = "a real topical sentence about coffee"
        db.memories.append(
            "events",
            [LogEntryInput(content=real, content_embedding=[0.9, 0.43, 0.0])],
            author="chat",
        )

        hits = db.memories.read_similar_hybrid("events", [anchor])
        contents = [e.content for e in hits]
        assert real in contents
        for junk in ("", "?", "Hey!", "hi penny"):
            assert junk not in contents

    def test_keys_returns_unique_in_insertion_order(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write("likes", [EntryInput(key="first", content="1")], author="chat")
        db.memories.write("likes", [EntryInput(key="second", content="2")], author="chat")
        assert db.memories.keys("likes") == ["first", "second"]


class TestExists:
    def test_exists_by_exact_key(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write("likes", [EntryInput(key="dark roast", content="body")], author="chat")

        assert db.memories.exists(["likes"], "dark roast", None, None) is True
        assert db.memories.exists(["likes"], "not seen", None, None) is False

    def test_exists_by_similarity_across_stores(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("unnotified", "pending", RecallMode.OFF)
        db.memories.create_collection("notified", "done", RecallMode.RELEVANT)
        shared = _unit_vec(2)
        db.memories.write(
            "notified",
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
        db.memories.create_log("user-messages", "inbound", RecallMode.RELEVANT)
        now = datetime.now(UTC)
        db.cursors.advance_committed("preference-extractor", "user-messages", now)

        assert db.cursors.get("preference-extractor", "user-messages") == now

    def test_advance_is_monotonic(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("user-messages", "inbound", RecallMode.RELEVANT)
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
        media_id = db.media.put(b"binary payload", "image/png", source_url="https://x.test/a.png")
        entry = db.media.get(media_id)

        assert entry is not None
        assert entry.data == b"binary payload"
        assert entry.mime_type == "image/png"
        assert entry.source_url == "https://x.test/a.png"

    def test_get_missing_returns_none(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.media.get(99999) is None


class TestWriteTypeEnforcement:
    def test_write_requires_collection(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", RecallMode.RECENT)
        with pytest.raises(MemoryTypeError):
            db.memories.write("events", [EntryInput(key="k", content="v")], author="chat")

    def test_write_on_missing_store_raises(self, tmp_path):
        db = _make_db(tmp_path)
        with pytest.raises(MemoryNotFoundError):
            db.memories.write("nope", [EntryInput(key="k", content="v")], author="chat")

    def test_dedup_thresholds_configurable(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write(
            "likes",
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
        result = db.memories.write(
            "likes",
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
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write(
            "likes",
            [EntryInput(key="dark roast", content="first body")],
            author="chat",
        )
        result = db.memories.write(
            "likes",
            [EntryInput(key="dark roast coffee", content="second body")],
            author="chat",
        )
        assert result[0].outcome == "duplicate"

    def test_tcr_relaxed_alone_does_not_fire(self, tmp_path):
        """TCR 2/3 with no other signal is not enough on its own."""
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write(
            "likes",
            [EntryInput(key="applied ai conference", content="first")],
            author="chat",
        )
        result = db.memories.write(
            "likes",
            [EntryInput(key="applied ai conf", content="second")],
            author="chat",
        )
        assert result[0].outcome == "written"

    def test_two_relaxed_signals_reject(self, tmp_path):
        """TCR 2/3 plus a relaxed content-cosine hit (~0.80) → duplicate."""
        db = _make_db(tmp_path)
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write(
            "likes",
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
        result = db.memories.write(
            "likes",
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
        db.memories.create_collection("likes", "x", RecallMode.RELEVANT)
        db.memories.write(
            "likes",
            [
                EntryInput(
                    key="coffee roast",
                    content="first",
                    content_embedding=[1.0, 0.0],
                )
            ],
            author="chat",
        )
        result = db.memories.write(
            "likes",
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
