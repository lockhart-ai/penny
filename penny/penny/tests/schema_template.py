"""Process-cached test-database schema templates.

Building a test database's schema is expensive: ``create_tables()`` issues the
DDL for every table, and ``migrate()`` replays the whole numbered migration
stack (~100 files). Together that is ~110ms, and the suite builds hundreds of
fresh databases — so rebuilding the schema per test was the single largest
chunk of ``make check``'s runtime.

The schema is deterministic, so this module builds each template **once per
process** and hands out fast file copies (~1ms each). A copy is byte-identical
to building the schema from scratch, so tests see no behavioural difference.

Two templates, matching the two things tests actually want:

- ``schema_only_db`` — ``create_tables()`` with no migrations: a bare current-
  models schema with zero rows, for tests that declare exactly the memories
  they need (migration 0026 would otherwise seed system log memories).
- ``migrated_db`` — ``create_tables()`` + ``migrate()``: the real startup
  schema, including every migration-seeded row.
"""

import os
import shutil
import tempfile
from pathlib import Path

from penny.config_params import RuntimeParams
from penny.database import Database
from penny.database.migrate import migrate

_schema_only_template: str | None = None
_migrated_template: str | None = None


def _build_template(*, migrated: bool) -> str:
    """Build a schema template once and return its file path."""
    file_descriptor, path = tempfile.mkstemp(suffix=".db")
    os.close(file_descriptor)
    db = Database(path)
    db.create_tables()
    if migrated:
        migrate(path)
    # Release the connection so the file is fully flushed before it's copied.
    db.engine.dispose()
    return path


def schema_only_template_path() -> str:
    """Path to the process-cached ``create_tables()``-only template."""
    global _schema_only_template
    if _schema_only_template is None:
        _schema_only_template = _build_template(migrated=False)
    return _schema_only_template


def migrated_template_path() -> str:
    """Path to the process-cached ``create_tables()`` + ``migrate()`` template."""
    global _migrated_template
    if _migrated_template is None:
        _migrated_template = _build_template(migrated=True)
    return _migrated_template


def _copy_template(template_path: str, destination: str) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(template_path, destination)


def schema_only_db(path: str, *, runtime: RuntimeParams | None = None) -> Database:
    """A ``Database`` at ``path`` with a bare schema — no migration seed data.

    Byte-identical to ``Database(path)`` + ``create_tables()``, but copied from a
    cached template instead of rebuilding the schema.
    """
    _copy_template(schema_only_template_path(), path)
    return Database(path, runtime=runtime)


def migrated_db(path: str, *, runtime: RuntimeParams | None = None) -> Database:
    """A ``Database`` at ``path`` with the full post-migration schema + seed data.

    Byte-identical to ``Database(path)`` + ``create_tables()`` + ``migrate(path)``,
    but copied from a cached template instead of rebuilding the schema.
    """
    _copy_template(migrated_template_path(), path)
    return Database(path, runtime=runtime)
