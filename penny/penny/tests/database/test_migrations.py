"""Tests for the database migration system."""

import importlib.util
import json
import re
import sqlite3
from pathlib import Path

import pytest

from penny.database.migrate import (
    _discover_migrations,
    _get_number_prefix,
    migrate,
    validate_migrations,
)


class TestDiscovery:
    """Tests for migration file discovery."""

    def test_discover_finds_migrations(self):
        migrations = _discover_migrations()
        assert len(migrations) >= 1
        assert migrations[0][0] == "0001_initial_schema"

    def test_discover_returns_sorted(self):
        migrations = _discover_migrations()
        names = [name for name, _path in migrations]
        assert names == sorted(names)

    def test_get_number_prefix(self):
        assert _get_number_prefix("0001_add_fields") == "0001"
        assert _get_number_prefix("0042_something") == "0042"


class TestValidation:
    """Tests for migration number validation."""

    def test_validate_passes_with_no_duplicates(self):
        validate_migrations()

    def test_validate_detects_duplicates(self, tmp_path):
        """Create temp migration files with duplicate prefixes and verify detection."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "0001_first.py").write_text(
            "import sqlite3\ndef up(conn: sqlite3.Connection) -> None: pass\n"
        )
        (migrations_dir / "0001_second.py").write_text(
            "import sqlite3\ndef up(conn: sqlite3.Connection) -> None: pass\n"
        )

        # Monkeypatch MIGRATIONS_DIR to use our temp dir
        import penny.database.migrate as mod

        original = mod.MIGRATIONS_DIR
        mod.MIGRATIONS_DIR = migrations_dir
        try:
            with pytest.raises(ValueError, match="Migration number conflict"):
                validate_migrations()
        finally:
            mod.MIGRATIONS_DIR = original


class TestMigrate:
    """Tests for the migration runner."""

    def test_skips_if_db_does_not_exist(self, tmp_path):
        db_path = str(tmp_path / "nonexistent.db")
        count = migrate(db_path)
        assert count == 0
        assert not Path(db_path).exists()

    def test_applies_to_existing_db(self, tmp_path):
        """Migration 0001 should create all tables in a bare database."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # Create a minimal table so the DB file exists
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 96

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'"
            ).fetchall()
        }
        expected = {
            "promptlog",
            "messagelog",
            "userinfo",
            "command_logs",
            "runtime_config",
            "mutestate",
            "thought",
            "preference",
            "device",
        }
        assert expected.issubset(tables)
        # entity and fact tables should NOT exist (dropped by 0004)
        assert "entity" not in tables
        assert "fact" not in tables
        # conversationhistory should NOT exist (dropped by 0024)
        assert "conversationhistory" not in tables
        # schedule should NOT exist (mechanism retired, dropped by 0082)
        assert "schedule" not in tables
        conn.close()

    def test_idempotent(self, tmp_path):
        """Running migrate twice should not fail or re-apply."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        count1 = migrate(db_path)
        count2 = migrate(db_path)
        assert count1 == 96
        assert count2 == 0

    def test_tracks_in_migrations_table(self, tmp_path):
        """Applied migrations should be recorded in _migrations."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in cursor.fetchall()}
        assert "0001_initial_schema" in applied
        conn.close()

    def test_skips_already_applied(self, tmp_path):
        """If _migrations already records a migration, it should not be re-run."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO _migrations (name, applied_at) VALUES (?, ?)",
            ("0001_initial_schema", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        count = migrate(db_path)
        # 0001 is skipped; the rest run = 95 migrations
        assert count == 95

    def test_bootstrap_with_tables_already_present(self, tmp_path):
        """If tables already exist (from SQLModel.create_tables), migration should succeed."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # Simulate a table already created by SQLModel.create_tables() with full schema
        conn.execute("""
            CREATE TABLE messagelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                direction TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                parent_id INTEGER REFERENCES messagelog(id),
                signal_timestamp INTEGER,
                recipient TEXT,
                external_id TEXT,
                is_reaction BOOLEAN NOT NULL DEFAULT 0,
                processed BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        count = migrate(db_path)
        assert count == 96  # all migrations applied

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM _migrations")
        applied = {row[0] for row in cursor.fetchall()}
        assert "0001_initial_schema" in applied
        conn.close()

    def test_0039_fixes_read_last_in_extraction_prompts(self, tmp_path):
        """Migration 0039 replaces read_last( with read_latest( in extraction prompts."""

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        broken_prompt = 'Call read_last("user-messages") to fetch new entries.'
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("my-collection", broken_prompt),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0039_fix_read_last_in_extraction_prompts.py"
        )
        spec = importlib.util.spec_from_file_location("m0039", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'my-collection'"
        ).fetchone()
        assert row is not None
        prompt = row[0]
        assert "read_last(" not in prompt
        assert 'read_latest("user-messages")' in prompt
        conn.close()

    def test_0085_adds_notify_and_copies_published(self, tmp_path):
        """Migration 0085 adds ``memory.notify`` and seeds it from ``published`` —
        so a collection that already notified (published) keeps notifying, and a
        silent one stays silent."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE memory (name TEXT PRIMARY KEY, published INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute("INSERT INTO memory (name, published) VALUES ('watched', 1)")
        conn.execute("INSERT INTO memory (name, published) VALUES ('silent', 0)")
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0085_add_memory_notify.py"
        )
        spec = importlib.util.spec_from_file_location("m0085", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
        assert "notify" in columns
        by_name = dict(conn.execute("SELECT name, notify FROM memory").fetchall())
        assert by_name == {"watched": 1, "silent": 0}
        conn.close()

    def test_0069_regrounds_and_cleans_skills(self, tmp_path):
        """Migration 0069 swaps the skills prompt to the catalog-driven loop, drops
        the chat-derived one-offs + Scheduled digest, rewrites the seeded skills
        into clean positive recipes (no ``[key]`` prefix, no send_message negatives),
        and leaves deployment-specific chat-authored entries untouched."""

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        conn.execute(
            "CREATE TABLE memory_entry (id INTEGER PRIMARY KEY, memory_name TEXT, "
            "key TEXT, content TEXT, author TEXT, content_embedding BLOB)"
        )
        conn.execute("INSERT INTO memory (name, extraction_prompt) VALUES ('skills', 'old prompt')")
        seeded = [
            (
                "Research collection — notify on new finds",
                "[Research collection — notify on new finds] TRIGGER\nuser wants research.\n"
                "STEPS\npublished: true; do NOT add a send_message step to the body.",
                "system",
            ),
            ("Scheduled digest", "TRIGGER\nsend_message at the scheduled time.", "system"),
            ("shorten-greeting", "TRIGGER\na one-off correction.", "skills"),
            (
                "my custom watcher",
                "TRIGGER\na user-taught recipe with a send_message step.",
                "chat",
            ),
        ]
        for key, content, author in seeded:
            conn.execute(
                "INSERT INTO memory_entry (memory_name, key, content, author, content_embedding) "
                "VALUES ('skills', ?, ?, ?, X'00')",
                (key, content, author),
            )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0069_reground_skills_on_collections.py"
        )
        spec = importlib.util.spec_from_file_location("m0069", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)

        # Prompt swapped to the catalog-driven reconcile loop.
        prompt = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name='skills'"
        ).fetchone()[0]
        assert "collection_catalog" in prompt

        entries = dict(
            conn.execute(
                "SELECT key, content FROM memory_entry WHERE memory_name='skills'"
            ).fetchall()
        )
        # One-offs and the orphan digest are gone; chat-authored entry is untouched.
        assert "shorten-greeting" not in entries
        assert "Scheduled digest" not in entries
        assert "my custom watcher" in entries and "send_message" in entries["my custom watcher"]
        # The seeded research skill is rewritten clean: no bracket prefix, no
        # send_message negative, still a TRIGGER recipe on the published model.
        research = entries["Research collection — notify on new finds"]
        assert not research.lstrip().startswith("[")
        assert "send_message" not in research
        assert "TRIGGER" in research and "published: true" in research
        conn.close()

    def test_0037_fixes_knowledge_extraction_prompt(self, tmp_path):
        """Migration 0037 replaces collection_update with update_entry in the
        knowledge extraction_prompt for databases seeded by migration 0031."""

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        broken_prompt = (
            'call collection_update("knowledge", key=<title>, content=<merged paragraph>)'
        )
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("knowledge", broken_prompt),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0037_fix_knowledge_extraction_prompt.py"
        )
        spec = importlib.util.spec_from_file_location("m0037", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'knowledge'"
        ).fetchone()
        assert row is not None
        prompt = row[0]
        assert "collection_update" not in prompt
        assert 'update_entry("knowledge", key=<title>,' in prompt
        conn.close()

    def test_0040_fixes_log_read_log_in_extraction_prompts(self, tmp_path):
        """Migration 0040 replaces log_read_log( with log_read_next( in any extraction_prompt."""

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        broken_prompt = 'Call log_read_log("user-messages") to fetch new entries.'
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("my-collection", broken_prompt),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0040_fix_log_read_log_in_extraction_prompts.py"
        )
        spec = importlib.util.spec_from_file_location("m0040", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'my-collection'"
        ).fetchone()
        assert row is not None
        prompt = row[0]
        assert "log_read_log(" not in prompt
        assert 'log_read_next("user-messages")' in prompt
        conn.close()

    def test_0041_fixes_collection_update_in_all_extraction_prompts(self, tmp_path):
        """Migration 0041 replaces collection_update with update_entry in all
        extraction_prompts, including user-created collections."""

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        conn.executemany(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            [
                (
                    "supplement-routine",
                    "On correction, update the entry via collection_update."
                    " On deletion, remove it.",
                ),
                (
                    "my-collection",
                    'call collection_update("my-collection", key=<k>, content=<c>)',
                ),
                (
                    "no-issue",
                    "Call update_entry to store the result.",
                ),
                (
                    "no-prompt",
                    None,
                ),
            ],
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0041_fix_collection_update_in_extraction_prompts.py"
        )
        spec = importlib.util.spec_from_file_location("m0041cu", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        rows = {
            row[0]: row[1]
            for row in conn.execute("SELECT name, extraction_prompt FROM memory").fetchall()
        }
        conn.close()

        assert "collection_update" not in rows["supplement-routine"]
        assert "update_entry" in rows["supplement-routine"]
        assert "collection_update" not in rows["my-collection"]
        assert "update_entry" in rows["my-collection"]
        assert rows["no-issue"] == "Call update_entry to store the result."
        assert rows["no-prompt"] is None

    def test_0042_fixes_thinking_prompt_browse_call_syntax(self, tmp_path):
        """Migration 0042 replaces bare 'browse' label with explicit call syntax in
        the unnotified-thoughts extraction_prompt for databases seeded by migration 0033."""

        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        old_prompt = "3. browse — search the web and read one or two pages to find something"
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES (?, ?)",
            ("unnotified-thoughts", old_prompt),
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0042_fix_thinking_prompt_browse_call_syntax.py"
        )
        spec = importlib.util.spec_from_file_location("m0042br", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        conn.close()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'unnotified-thoughts'"
        ).fetchone()
        assert row is not None
        prompt = row[0]
        assert "3. browse — search the web" not in prompt
        assert 'browse(queries=["<seed topic>"])' in prompt
        conn.close()

    def test_0047_replaces_run_id_index_with_composite(self, tmp_path):
        """Migration 0047 adds the (run_id, timestamp) composite index used by
        the prompt-log run pagination and drops the redundant single-column
        run_id index from 0021."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE promptlog (id INTEGER PRIMARY KEY, run_id TEXT, timestamp TEXT)")
        conn.execute("CREATE INDEX ix_promptlog_run_id ON promptlog (run_id)")
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0047_promptlog_run_id_timestamp_index.py"
        )
        spec = importlib.util.spec_from_file_location("m0047", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='promptlog'"
            ).fetchall()
        }
        conn.close()

        assert "ix_promptlog_run_id_timestamp" in indexes
        assert "ix_promptlog_run_id" not in indexes

    def test_0048_adds_agent_run_index(self, tmp_path):
        """Migration 0048 adds the (agent_name, run_id, timestamp) index used by
        the per-agent prompt-log filter."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE promptlog "
            "(id INTEGER PRIMARY KEY, agent_name TEXT, run_id TEXT, timestamp TEXT)"
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0048_promptlog_agent_run_index.py"
        )
        spec = importlib.util.spec_from_file_location("m0048", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='promptlog'"
            ).fetchall()
        }
        conn.close()
        assert "ix_promptlog_agent_run_timestamp" in indexes

    def test_0049_partitions_cursors_per_collection(self, tmp_path):
        """Migration 0049 seeds a per-collection cursor from the old shared
        (collector, log) value and drops the dead dispatcher/legacy rows."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, extraction_prompt TEXT)")
        conn.execute(
            "CREATE TABLE agent_cursor (agent_name TEXT, memory_name TEXT, "
            "last_read_at TEXT, updated_at TEXT, PRIMARY KEY (agent_name, memory_name))"
        )
        conn.execute(
            "INSERT INTO memory (name, extraction_prompt) VALUES "
            "('journal', 'log_read_next(\"user-messages\")'), "
            "('likes', 'log_read_next(\"user-messages\")'), "
            "('user-messages', NULL)"
        )
        conn.execute(
            "INSERT INTO agent_cursor VALUES "
            "('collector', 'user-messages', '2026-06-13T00:00:00', '2026-06-13T00:00:00'), "
            "('preference-extractor', 'user-messages', '2026-01-01', '2026-01-01')"
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0049_partition_collector_cursors_per_collection.py"
        )
        spec = importlib.util.spec_from_file_location("m0049", migration_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        # Both collections that read user-messages get their own cursor at the
        # old shared value — no re-processing of history on the next run.
        for collection in ("journal", "likes"):
            row = conn.execute(
                "SELECT last_read_at FROM agent_cursor "
                "WHERE agent_name = ? AND memory_name = 'user-messages'",
                (collection,),
            ).fetchone()
            assert row is not None and row[0] == "2026-06-13T00:00:00"
        # Dead rows are gone: the shared dispatcher row + the pre-dispatcher agent.
        assert (
            conn.execute(
                "SELECT 1 FROM agent_cursor "
                "WHERE agent_name IN ('collector', 'preference-extractor')"
            ).fetchone()
            is None
        )
        conn.close()

    def test_0067_seeds_notifier_consumer_then_0086_retires_it(self, tmp_path):
        """Migration 0067 seeds the notifier consumer (a published-stream drainer,
        silent in chat); across the full chain, 0086 archives it — emission is now
        the ``notify`` flag + the run-time notify suffix, so the consumer is
        retired.  The seeded row survives as an archived tombstone."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT type, archived, extraction_prompt FROM memory WHERE name = 'notifier'"
        ).fetchone()
        conn.close()
        assert row is not None
        type_, archived, prompt = row
        assert type_ == "collection"
        assert archived == 1  # retired by 0086 (emission is now the notify suffix)
        assert "read_published_latest" in prompt  # the 0067-seeded body survives (0087
        # strips only its terminal bare-done step line)

    def test_0068_unifies_thoughts_onto_pubsub(self, tmp_path):
        """Migration 0068 collapses unnotified-/notified-thoughts into one
        `thoughts` producer (no send_message in its body), moves their entries in,
        seeds the notifier cursor to the head, and archives the old collections.
        Across the full chain, 0085 copies its ``published`` into ``notify`` and
        0086 drops ``published`` — so ``thoughts`` now notifies via the flag."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT notify, extraction_prompt FROM memory WHERE name = 'thoughts'"
        ).fetchone()
        archived = dict(
            conn.execute(
                "SELECT name, archived FROM memory "
                "WHERE name IN ('unnotified-thoughts', 'notified-thoughts')"
            ).fetchall()
        )
        conn.close()
        assert row is not None
        notify, prompt = row
        assert notify == 1  # 0085 seeded notify from published=1 — it tells the user
        assert 'collection_write("thoughts"' in prompt  # producer writes to itself
        assert "send_message" not in prompt  # gathers only; the notify suffix does the sending
        # The old move-drain pair is retired (archived), not dispatched.
        assert archived == {"unnotified-thoughts": 1, "notified-thoughts": 1}

    def test_0086_retires_notifier_and_drops_published(self, tmp_path):
        """Migration 0086 retires the pub/sub layer over the full chain: archives the
        notifier consumer, drops the ``published`` column, and rewrites the seeded
        skills that taught ``published`` to teach ``notify`` — the sole emission
        flag now (emission is a collection property + the run-time notify suffix).

        0086's ``published`` → ``notify`` entry rewrites can't be asserted at
        end-of-chain — 0092 (the skills-collection retirement, #1624) deletes every
        seeded ``skills`` entry downstream.  The rewrite of the ``skills``
        collector's own prompt is still witnessed: the archived tombstone keeps its
        ``extraction_prompt`` intact."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()]
        notifier = conn.execute("SELECT archived FROM memory WHERE name = 'notifier'").fetchone()
        skills_prompt = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'skills'"
        ).fetchone()[0]
        conn.close()

        # The retired pub/sub column is gone; ``notify`` is the sole emission flag.
        assert "published" not in columns
        assert "notify" in columns
        # The notifier consumer is archived (a visible tombstone), not deleted.
        assert notifier is not None and notifier[0] == 1
        # The skills tombstone's prompt teaches ``notify`` — 0086's rewrite held.
        assert "notify: true" in skills_prompt
        assert "published" not in skills_prompt

    def test_0087_strips_terminal_done_steps_from_stored_prompts(self, tmp_path):
        """Migration 0087 (over the full chain): the terminal ``done()`` is assembly's
        now, so the seeded prompts' bare terminal done-step lines are stripped —
        ``likes``/``dislikes`` (``6. Call done().``), ``knowledge`` (``4. Call
        done().``), ``quality`` (``5. done().``), ``thoughts`` (``8. done().``), and
        the archived ``notifier``/``unnotified-thoughts`` shells — while compound
        terminal steps (``skills`` step 8, ``notified-thoughts`` step 4) and prose
        *descriptions* of done behaviour survive verbatim (zero-false-positive).
        The ``skills`` prompt survives at end-of-chain on its archived tombstone
        (0092 archives the collection but leaves the prompt intact)."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        prompts = dict(
            conn.execute(
                "SELECT name, extraction_prompt FROM memory WHERE extraction_prompt IS NOT NULL"
            ).fetchall()
        )
        conn.close()

        # No stored prompt retains a numbered step line that is just a done call.
        bare_done = re.compile(r"^\d+\.[ \t]*(?:Call[ \t]+)?done\([^()]*\)\.?[ \t]*$", re.MULTILINE)
        for name, prompt in prompts.items():
            assert not bare_done.search(prompt), f"{name} still has a bare done step line"
        # The compound terminal steps survive 0087's strip; 0089 then rewrites the
        # ``skills`` prompt's ``done(success=…, summary=…)`` conditionals to the
        # argless ``done()`` (#1569), so the compound step is present but argless.
        assert "If nothing changed, done()." in prompts["skills"]
        assert "If there's nothing fresh to share, just done()." in prompts["notified-thoughts"]
        assert "call done() without writing anything" in prompts["knowledge"]
        assert "call done() without writing" in prompts["thoughts"]
        # The stripped prompts keep their remaining steps + trailing prose untouched:
        # only the one done-step line was removed.
        assert "5. If a recent message indicates an existing like" in prompts["likes"]
        assert prompts["quality"].rstrip().endswith("never apply a change yourself.")
        assert prompts["notifier"].rstrip().endswith("— deliver it.")

    def test_0088_adds_emission_provenance_column(self, tmp_path):
        """Migration 0088 adds ``mechanism`` to ``messagelog`` (#1568) — schema
        only, idempotent, on a DB that predates it — plus the partial emission
        index the self-state hot-path scan needs."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE messagelog (id INTEGER PRIMARY KEY, direction TEXT, "
            "content TEXT, timestamp TIMESTAMP)"
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0088_emission_provenance.py"
        )
        spec = importlib.util.spec_from_file_location("m0088", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        # Re-running is a no-op (idempotent) — the guards skip existing columns.
        mod.up(conn)
        message_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(messagelog)").fetchall()
        }
        # The recent-emissions scan (self-state hot path) is served by the partial
        # index — mechanism-bearing rows are sparse in messagelog, so the filter+sort
        # must not walk the whole timestamp order.  EXPLAIN QUERY PLAN proves the
        # planner actually picks it for the exact query recent_emissions runs.
        plan = " ".join(
            str(step)
            for step in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM messagelog "
                "WHERE direction = 'outgoing' AND mechanism IS NOT NULL "
                "ORDER BY timestamp DESC, id DESC LIMIT 8"
            ).fetchall()
        )
        conn.close()

        assert "mechanism" in message_columns
        assert "ix_messagelog_emission_time" in plan, plan

    def test_0089_argless_done_and_retire_quality(self, tmp_path):
        """Migration 0089 (over the full chain, #1569): the ``skills`` collector
        prompt's two ``done(success=…, summary=…)`` conditionals are rewritten to
        the argless ``done()``, and the ``quality`` collection is archived (a
        visible tombstone) with its extraction_prompt left intact.  (The ``skills``
        prompt survives at end-of-chain on its own archived tombstone — 0092
        archives the collection but leaves the prompt intact.)"""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        skills_prompt = conn.execute(
            "SELECT extraction_prompt FROM memory WHERE name = 'skills'"
        ).fetchone()[0]
        archived, quality_prompt = conn.execute(
            "SELECT archived, extraction_prompt FROM memory WHERE name = 'quality'"
        ).fetchone()
        conn.close()

        # The skills prompt teaches only the argless done() — no forbidden args.
        assert "done(success" not in skills_prompt
        assert "If nothing changed, done()." in skills_prompt
        # Quality is archived (tombstone); its prompt is untouched (never dispatched).
        assert archived == 1
        assert quality_prompt is not None

    def test_0092_retires_skills_collection_entirely(self, tmp_path):
        """Migration 0092 (over the full chain, #1624 as amended): the ``skills``
        collection retires ENTIRELY — there is exactly one skills store, the
        ``skill`` table.  The collection is archived (visible tombstone, the
        0086/0089 pattern; its prompt left intact) and every migration-seeded rule
        entry is deleted — on a fresh install the collection is empty."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        prompt, archived = conn.execute(
            "SELECT extraction_prompt, archived FROM memory WHERE name = 'skills'"
        ).fetchone()
        keys = [
            row[0]
            for row in conn.execute(
                "SELECT key FROM memory_entry WHERE memory_name = 'skills'"
            ).fetchall()
        ]
        conn.close()

        # The collection is an archived tombstone: never dispatched (``_is_ready``
        # skips archived rows), hidden from the catalog, out of the store map.
        assert archived == 1
        # Its prompt survives as the tombstone's historical record (0089 pattern).
        assert prompt is not None
        # Every seeded rule entry is gone — a fresh install ships the collection
        # empty; the taught-skill table is the sole skills store.
        assert keys == []

    def test_0074_deletes_degenerate_memory_entries(self, tmp_path):
        """Migration 0074 deletes entries whose key or content carries a
        degeneration-collapse run (in content or key), and leaves clean entries —
        including ones with an ordinary trailing ellipsis — untouched."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE memory_entry (id INTEGER PRIMARY KEY, memory_name TEXT, "
            "key TEXT, content TEXT, author TEXT)"
        )
        rows = [
            (1, "knowledge", "Clean title", "A real summary of a page.", "knowledge"),
            (2, "knowledge", "Anyway…", "A find dropped this week…", "knowledge"),  # clean ellipsis
            (3, "knowledge", "Boss delay", "Delivered a find about Boss ..??.. gear", "knowledge"),
            (4, "knowledge", "New … … … … openings", "poison in the key", "knowledge"),
            (5, "news", "Falcon 9", "Starlink launch summary ……? today", "news"),
        ]
        conn.executemany(
            "INSERT INTO memory_entry (id, memory_name, key, content, author) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0074_delete_degenerate_memory_entries.py"
        )
        spec = importlib.util.spec_from_file_location("m0074", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        surviving = {row[0] for row in conn.execute("SELECT id FROM memory_entry").fetchall()}
        conn.close()

        # 1 (clean) and 2 (ordinary trailing ellipsis) survive; 3/4/5 (poison in
        # content or key) are deleted.
        assert surviving == {1, 2}

    def test_0081_adds_registry_provenance_columns(self, tmp_path):
        """Migration 0081 adds the provenance + lifecycle columns to ``memory``
        (source_message_id, created_by_run_id, expires_at), and is idempotent on
        a table that predates them."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # A ``memory`` table as it existed before the registry columns.
        conn.execute("CREATE TABLE memory (name TEXT PRIMARY KEY, description TEXT)")
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0081_add_registry_provenance_columns.py"
        )
        spec = importlib.util.spec_from_file_location("m0081", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        # A second application is a no-op (guarded on PRAGMA table_info) — proves
        # the migration is safe on a fresh DB whose model already carries them.
        mod.up(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
        conn.close()

        assert {"source_message_id", "created_by_run_id", "expires_at"}.issubset(columns)

    def test_0083_adds_entry_stamps_and_mutation_event_table(self, tmp_path):
        """Migration 0083 adds the two entry run-id stamp columns to
        ``memory_entry`` and creates the ``mutation_event`` table, and is
        idempotent on a DB that already carries them (#1560)."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # A ``memory_entry`` table as it existed before the stamp columns.
        conn.execute(
            "CREATE TABLE memory_entry (id INTEGER PRIMARY KEY, memory_name TEXT, content TEXT)"
        )
        conn.commit()
        conn.close()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0083_ledger_provenance_closure.py"
        )
        spec = importlib.util.spec_from_file_location("m0083", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        mod.up(conn)
        mod.up(conn)  # second application is a no-op (guarded)
        entry_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_entry)").fetchall()
        }
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        mutation_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(mutation_event)").fetchall()
        }
        conn.close()

        assert {"created_by_run_id", "last_written_by_run_id"}.issubset(entry_columns)
        assert "mutation_event" in tables
        assert {"entity_type", "entity_name", "action", "actor", "run_id", "detail"}.issubset(
            mutation_columns
        )

    def test_0093_drops_recall_substrate(self, tmp_path):
        """Migration 0093 (over the full chain): the dead recall columns
        (``inclusion`` + ``recall``) are dropped, and the 0069-seeded research skill
        recipes no longer teach the removed flags (their ``- inclusion: "relevant",
        recall: "relevant"`` line is stripped).  ``description_embedding`` (resolve-
        by-meaning) and ``notify`` (emission) stay."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE _bootstrap (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memory)").fetchall()}
        skill_bodies = dict(
            conn.execute(
                "SELECT key, content FROM memory_entry WHERE memory_name = 'skills' "
                "AND key IN (?, ?)",
                (
                    "Research collection — notify on new finds",
                    "Research collection — silent",
                ),
            ).fetchall()
        )
        conn.close()

        # The dead recall columns are gone; the retained anchors/flags stay.
        assert "inclusion" not in columns
        assert "recall" not in columns
        assert "description_embedding" in columns  # resolve-by-meaning (#1558)
        assert "notify" in columns  # emission-as-property (#1557)
        # The seeded recipes no longer teach the dropped flags.
        for body in skill_bodies.values():
            assert 'inclusion: "relevant", recall: "relevant"' not in body

    def test_0096_renames_skill_holes_to_parameters(self, tmp_path):
        """Migration 0096 (#1668): the ``skill.holes`` column renames to
        ``parameters``, AND each stored skill's ``steps`` JSON renames its per-leaf
        substitution ``hole`` key to ``parameter`` — so a skill demonstrated before
        the rename keeps rendering its parameter names, not an empty placeholder."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        # A pre-0096 skill table (the 0084 shape, ``holes`` column) with one skill whose
        # steps carry a substitution keyed on the OLD ``hole`` key.
        conn.execute(
            "CREATE TABLE skill ("
            "  name TEXT PRIMARY KEY, steps TEXT NOT NULL, holes TEXT NOT NULL,"
            "  intent TEXT NOT NULL, description TEXT NOT NULL, author TEXT NOT NULL)"
        )
        steps = [
            {
                "ordinal": 1,
                "source_ordinal": 1,
                "tool": "browse",
                "arguments": {"queries": ["{url}"]},
                "substitutions": [
                    {"path": ["queries", 0], "kind": "hole", "hole": "url", "step": None}
                ],
            }
        ]
        conn.execute(
            "INSERT INTO skill (name, steps, holes, intent, description, author) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "watch-a-page",
                json.dumps(steps),
                '[{"name": "url", "required": true}]',
                "x",
                "x",
                "c",
            ),
        )
        conn.commit()

        migration_path = (
            Path(__file__).parents[3]
            / "penny"
            / "database"
            / "migrations"
            / "0096_rename_skill_holes_to_parameters.py"
        )
        spec = importlib.util.spec_from_file_location("m0096", migration_path)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        mod.up(conn)

        columns = {row[1] for row in conn.execute("PRAGMA table_info(skill)").fetchall()}
        assert "parameters" in columns and "holes" not in columns
        stored = json.loads(
            conn.execute("SELECT steps FROM skill WHERE name = 'watch-a-page'").fetchone()[0]
        )
        sub = stored[0]["substitutions"][0]
        assert sub["parameter"] == "url" and "hole" not in sub  # key renamed, value kept
        conn.close()
