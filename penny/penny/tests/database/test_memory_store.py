"""Tests for MemoryStore, CursorStore, and MediaStore.

Exercises the data layer for the task/memory framework. Dedup, type
enforcement, log append, cursor monotonicity, and the similarity-based
`exists` check all run through these tests.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta

import pytest

from penny.constants import MutationAction, MutationActor, PennyConstants
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
from penny.datetime_utils import format_log_timestamp
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
        db.memories.create_collection(
            "notes", "scratch", Inclusion.NEVER, RecallMode.RECENT, created_by_run_id="run-create"
        )
        db.memories.archive("notes", run_id="run-arch")
        assert db.memories.get("notes").archived is True
        db.memories.unarchive("notes", run_id="run-unarch")
        assert db.memories.get("notes").archived is False

    def test_registry_mutations_are_durable_events(self, tmp_path):
        """Every create / update / archive / unarchive of a collection writes a
        ledger event carrying (entity, run, actor, what changed) — so a
        mechanism's config history is a read, in time order (#1560, criteria
        2 + 4)."""
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "watch", "a watch", Inclusion.RELEVANT, RecallMode.RECENT, created_by_run_id="run-1"
        )
        db.memories.update_collection_metadata("watch", recall=RecallMode.ALL, run_id="run-2")
        db.memories.archive("watch", run_id="run-3")
        db.memories.unarchive("watch", run_id="run-4")

        events = db.mutations.history("watch", limit=10)
        # Newest first (datetime ordering), one row per mutation.
        assert [e.action for e in events] == [
            MutationAction.UNARCHIVED.value,
            MutationAction.ARCHIVED.value,
            MutationAction.UPDATED.value,
            MutationAction.CREATED.value,
        ]
        assert [e.run_id for e in events] == ["run-4", "run-3", "run-2", "run-1"]
        # A chat-driven change is actor=user-run; the run id is the join key.
        assert all(e.actor == MutationActor.USER_RUN.value for e in events)
        # The update names which field changed (values live in the run's promptlog).
        update = next(e for e in events if e.action == MutationAction.UPDATED.value)
        assert update.detail is not None and "recall" in update.detail
        # A no-op update (nothing supplied) is not a mutation.
        db.memories.update_collection_metadata("watch")
        assert len(db.mutations.history("watch", limit=10)) == 4

    def test_metadata_renders_recent_change_history(self, tmp_path):
        """memory_metadata surfaces the mutation ledger so "when was this
        archived, and by what?" is answerable by a read, each change naming its
        run id (the anchor invariant, #1560)."""
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "watch", "a watch", Inclusion.NEVER, RecallMode.RECENT, created_by_run_id="run-1"
        )
        db.memories.archive("watch", run_id="run-9")
        result = asyncio.run(MemoryMetadataTool(db).execute(memory="watch")).message
        assert "Recent changes" in result
        assert "archived by user-run" in result
        assert "run run-9" in result

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
        # Timestamps render through the shared log-timestamp format (compact UTC),
        # unified with entry lists / run history — not a raw seconds strftime.
        row = db.memories.memory("board-games").row
        assert f"created: {format_log_timestamp(row.created_at)}" in result.message
        assert f"updated: {format_log_timestamp(row.updated_at)}" in result.message
        assert re.search(r"created: \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\n", result.message)
        # A collected memory renders its stamp through the same shared format.
        db.memories.mark_collected("board-games")
        collected = asyncio.run(tool.execute(memory="board-games")).message
        stamped = db.memories.memory("board-games").row.last_collected_at
        assert stamped is not None
        assert f"last collected: {format_log_timestamp(stamped)}" in collected

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
            run_id="run-create",
        )
        # The writing run stamps both anchors on the new entry (#1560).
        created = db.memories.memory("likes").get("k")[0]
        assert created.created_by_run_id == "run-create"
        assert created.last_written_by_run_id == "run-create"

        assert (
            db.memories.memory("likes").update("k", "new body", "chat", run_id="run-edit") == "ok"
        )
        entries = db.memories.memory("likes").get("k")
        assert entries[0].content == "new body"
        # A rewrite advances last_written but leaves the creating run intact — the
        # two anchors keep "who wrote the current value?" distinct from "who
        # created it?", the read-path into the ledger's write history.
        assert entries[0].created_by_run_id == "run-create"
        assert entries[0].last_written_by_run_id == "run-edit"

        # Lookups are strictly exact — a bracket-wrapped key (the `[key]` display
        # form copied from an entry list) is NOT silently normalized at the data
        # layer; the tool boundary rejects it with a teaching error instead.
        assert db.memories.memory("likes").get("[k]") == []
        assert db.memories.memory("likes").update("[k]", "newer body", "chat") == "not_found"
        assert db.memories.memory("likes").get("k")[0].content == "new body"

    def test_literal_bracket_key_matches_exactly_not_stripped(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        # A key that genuinely contains enclosing brackets resolves by exact
        # match — lookups are literal, so the brackets are part of the key.
        db.memories.memory("likes").write(
            [EntryInput(key="[lit]", content="bracketed body")], author="chat"
        )
        assert db.memories.memory("likes").get("[lit]")[0].content == "bracketed body"
        # The unwrapped form genuinely has no entry — the exact hit above wasn't
        # a match against a stray 'lit' entry.
        assert db.memories.memory("likes").get("lit") == []

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

        # Deletion is strictly exact too — a bracket-wrapped key deletes nothing
        # (the tool boundary rejects it with a teaching error; the data layer
        # never absorbs the display form).
        db.memories.memory("likes").write([EntryInput(key="k", content="b")], author="chat")
        assert db.memories.memory("likes").delete("[k]") == 0
        assert db.memories.memory("likes").get("k")[0].content == "b"

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

        # Move is strictly exact too — a bracket-wrapped key is a not_found, never
        # silently normalized to the bare key.
        db.memories.memory("unnotified").write(
            [EntryInput(key="thought-2", content="y")], author="thinking-agent"
        )
        assert (
            db.memories.memory("unnotified").move("[thought-2]", "notified", author="notifier")
            == "not_found"
        )
        assert len(db.memories.memory("unnotified").get("thought-2")) == 1

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

    def test_failed_browse_placeholder_rejected(self, tmp_path):
        """A failed/empty browse extraction ("No summary available") must not enter
        the corpus.  Real production case: a published collection wrote this as an
        entry, so the notifier drained it and delivered a confusing non-find to the
        user — the bail-phrase filter stops it at write time."""
        db = self._make_collection(tmp_path)
        results = db.memories.memory("knowledge").write(
            [EntryInput(key="We can't find that page", content="No summary available")],
            author="collector",
        )
        assert results[0].outcome == "rejected"
        assert results[0].reason is not None
        assert len(db.memories.memory("knowledge").read_all()) == 0

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

    def test_read_similar_returns_populated_homogeneous_collection(self, tmp_path):
        """A populated but homogeneous collection (every entry near-identical,
        like ``skills``' TRIGGER+STEPS recipes) must still return its entries —
        the explicit search is plain nearest-neighbour, with no cluster gate to
        collapse a flat corpus to "No entries" (#1565: that broke the model's
        fuzzy-recovery path exactly when guessing at a skill's identity)."""
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.RELEVANT, RecallMode.RELEVANT)
        anchor = [1.0, 0.0, 0.0]
        # Twenty entries with identical content embeddings — a maximally flat
        # corpus, which the old adaptive cluster gate suppressed entirely.
        for i in range(20):
            db.memories.memory("events").append(
                [LogEntryInput(content=f"flat-{i}", content_embedding=[0.7, 0.7, 0.07])],
                author="chat",
            )

        similar = db.memories.memory("events").read_similar(anchor)
        assert len(similar) == 20

    def test_read_similar_ranks_cluster_ahead_of_weak_matches(self, tmp_path):
        """Plain nearest-neighbour ranking: the strong-cluster entries come back
        ahead of the weak ones, so a bounded ``k`` returns the cluster first."""
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

        top = [e.content for e in db.memories.memory("events").read_similar(anchor, k=5)]
        assert top and all(c.startswith("hit-") for c in top)

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

        # An unknown name is a misspelled probe, not an empty (always-False)
        # memory: it raises so the caller can't misread "no" as "safe to write".
        with pytest.raises(MemoryNotFoundError, match="lieks"):
            db.memories.exists(["lieks"], "dark roast", None, None)

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

    def _put(self, db, data, url, vector=None):
        return db.media.put(
            data,
            "image/png",
            source_url=url,
            embedding=serialize_embedding(vector) if vector else None,
        )

    def test_select_image_prefers_exact_cited_url_newest(self, tmp_path):
        """Tier 1: the message links a page we captured — attach that page's own
        image (newest capture), even when another image embeds closer."""
        db = _make_db(tmp_path)
        # An embedding-identical image from a DIFFERENT page (would win on cosine).
        self._put(db, b"other", "https://other.test/x", [1.0, 0.0])
        self._put(db, b"old", "https://cited.test/p", [0.0, 1.0])  # older capture
        newest = self._put(db, b"new", "https://cited.test/p", [0.0, 1.0])
        # Trailing punctuation on the cited URL is normalized away.
        match = db.media.select_image(["https://cited.test/p."], [1.0, 0.0])
        assert match is not None
        assert match.id == newest
        assert match.data == b"new"

    def test_select_image_exact_url_works_without_embedding(self, tmp_path):
        """Tier 1 needs no embedding — the cited page's image attaches even when
        the message text couldn't be embedded."""
        db = _make_db(tmp_path)
        cited = self._put(db, b"a", "https://cited.test/p")  # no embedding
        match = db.media.select_image(["https://cited.test/p"], None)
        assert match is not None and match.id == cited

    def test_select_image_prefers_cited_domain_nearest(self, tmp_path):
        """Tier 2: the message links a page on a domain we have images from (but
        not that exact page) — pick the embedding-nearest image of that domain,
        not the globally nearest off-domain image."""
        db = _make_db(tmp_path)
        self._put(db, b"far", "https://site.test/a", [0.0, 1.0])
        near = self._put(db, b"near", "https://site.test/b", [0.9, 0.1])
        self._put(db, b"offdomain", "https://elsewhere.test/z", [1.0, 0.0])  # global nearest
        match = db.media.select_image(["https://site.test/c"], [1.0, 0.0])
        assert match is not None
        assert match.id == near

    def test_select_image_jitters_among_top_k_when_no_url(self, tmp_path, monkeypatch):
        """Tier 3: no cited URL — pick uniformly at random among the top-K nearest
        (not the strict argmax), so a magnet image can't repeat on every message."""
        db = _make_db(tmp_path)
        # 6 images at decreasing closeness to the query [1,0]; the 6th is farthest.
        vectors = [[1.0, 0.0], [0.95, 0.05], [0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.0, 1.0]]
        ids = [
            self._put(db, f"v{i}".encode(), f"https://s.test/{i}", v) for i, v in enumerate(vectors)
        ]
        seen = {}

        def choose(pool):
            seen["pool"] = list(pool)
            return pool[-1]  # deliberately NOT the nearest

        monkeypatch.setattr("penny.database.media_store.random.choice", choose)
        match = db.media.select_image([], [1.0, 0.0])
        assert match is not None
        # The pool is the top-K nearest (K=5), excluding the farthest image.
        assert len(seen["pool"]) == PennyConstants.MEDIA_MATCH_JITTER_TOPK
        assert ids[-1] not in seen["pool"]
        # Jitter honoured the random pick, not the argmax.
        assert match.id == seen["pool"][-1]

    def test_select_image_no_floor_attaches_even_weak_match(self, tmp_path):
        """Tier 3 has no floor — a reply is never left imageless: with one image
        the pool is that image regardless of how poor the cosine is."""
        db = _make_db(tmp_path)
        only = self._put(db, b"a", "https://a.test", [1.0, 0.0])
        match = db.media.select_image([], [0.0, 1.0])  # orthogonal
        assert match is not None and match.id == only

    def test_select_image_none_when_no_url_match_and_no_embedded_media(self, tmp_path):
        db = _make_db(tmp_path)
        self._put(db, b"a", "https://a.test")  # no embedding
        assert db.media.select_image([], [1.0, 0.0]) is None  # no embedded media to fall back to
        assert db.media.select_image([], None) is None  # nothing to match at all
        assert db.media.select_image(["https://nope.test/x"], [1.0, 0.0]) is None  # url misses


class TestWriteTypeEnforcement:
    def test_write_requires_collection(self, tmp_path):
        db = _make_db(tmp_path)
        db.memories.create_log("events", "x", Inclusion.ALWAYS, RecallMode.RECENT)
        # The wrong-shape refusal speaks the single house wording — names the value,
        # its actual shape, and binds the read tool that shape supports.
        with pytest.raises(MemoryTypeError, match="'events' is a log, not a collection"):
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

    def test_reembeds_keyed_entry_missing_only_key_vector(self, tmp_path):
        """A keyed entry with a content vector but a NULL key vector is picked up.

        This is the migration-seeded / transient-key-miss state (#1468): selecting
        only on ``content_embedding IS NULL`` would skip it forever.  It qualifies
        while its key vector is unset, and drops out once the key vector lands —
        without the content vector ever being disturbed.
        """
        db = _make_db(tmp_path)
        db.memories.create_collection(
            "likes", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        # Content vector present, key vector NULL (key present) — no dedup neighbour.
        db.memories.memory("likes").write(
            [
                EntryInput(
                    key="dark roast", content="loves dark roast", content_embedding=_unit_vec(1)
                )
            ],
            author="system",
        )
        pending = db.memories.get_entries_without_embeddings(limit=100)
        assert [e.key for e in pending] == ["dark roast"]
        entry = pending[0]
        assert entry.content_embedding is not None
        assert entry.key_embedding is None

        # Fill only the key vector; the row drops out and both vectors are set.
        entry_id = entry.id
        assert entry_id is not None
        db.memories.set_entry_embeddings(
            entry_id, key_embedding=_unit_vec(0), content_embedding=None
        )
        assert db.memories.get_entries_without_embeddings(limit=100) == []
        row = db.memories.memory("likes").read_latest()[0]
        assert row.key_embedding is not None
        assert row.content_embedding is not None

    def test_keyless_log_entry_not_selected_for_missing_key_vector(self, tmp_path):
        """A keyless log entry has a legitimately-null key vector — it must NOT
        qualify on that alone (only on a missing content vector), or the backfill
        would loop forever trying to embed a key that will always be NULL."""
        db = _make_db(tmp_path)
        db.memories.create_log("events", "event stream", Inclusion.ALWAYS, RecallMode.RECENT)
        # Content vector present, no key (log entries are keyless).
        db.memories.memory("events").append(
            [LogEntryInput(content="something happened", content_embedding=_unit_vec(2))],
            author="system",
        )
        assert db.memories.get_entries_without_embeddings(limit=100) == []
