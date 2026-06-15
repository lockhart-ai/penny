"""Tests for the database migration system."""

import importlib.util
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
        assert count == 59

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
            "schedule",
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
        assert count1 == 59
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
        # 0001 is skipped; 0002 through 0059 run = 58 migrations
        assert count == 58

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
        assert count == 59  # all migrations applied

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
