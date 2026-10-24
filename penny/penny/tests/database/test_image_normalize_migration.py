"""Tests for migration 0128 — normalizing every stored ``media`` row to the
live size/format rules (#2780).

Most tests isolate 0128 from the rest of the migration chain the same way
``test_signal_device_migration.py`` isolates 0126: every prior migration is
marked already applied, then this test seeds the exact pre-0128 ``media``
rows directly, then ``migrate()`` runs only 0128 against them. The
per-row-failure test instead loads 0128 by path and calls its ``up()``
directly (the ``test_migrations.py`` pattern), so it can monkeypatch the
module's own ``normalize_image`` name without touching ``migrate()``'s
separately-loaded instance.
"""

from __future__ import annotations

import importlib.util
import io
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.migrate import _discover_migrations, migrate
from penny.image_utils import normalize_image as _real_normalize_image

_MIGRATION_NAME = "0128_normalize_media_images"
_MIGRATION_PATH = (
    Path(__file__).parents[3] / "penny" / "database" / "migrations" / f"{_MIGRATION_NAME}.py"
)


def _load_migration_module():
    """Load 0128 directly by file path, the same pattern every other
    hand-loaded migration test in ``test_migrations.py`` uses — a fresh
    module object each call, so monkeypatching it never leaks to another
    test or to ``migrate()``'s own separately-loaded instance."""
    spec = importlib.util.spec_from_file_location("m0128", _MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_db(tmp_path) -> Database:
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _isolate_0128(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    now = datetime.now(UTC).isoformat()
    for name, _path in _discover_migrations():
        if name == _MIGRATION_NAME:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO _migrations (name, applied_at) VALUES (?, ?)", (name, now)
        )


def _seed_media(conn: sqlite3.Connection, data: bytes, mime_type: str) -> int:
    cursor = conn.execute(
        "INSERT INTO media (mime_type, data, created_at) VALUES (?, ?, ?)",
        (mime_type, data, datetime.now(UTC).isoformat()),
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _oversized_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), color=(10, 200, 80)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _animated_gif(frame_count: int = 3) -> bytes:
    frames = [
        Image.new("RGBA", (2000, 500), color=(i * 40 % 256, 10, 200)) for i in range(frame_count)
    ]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buf.getvalue()


def _small_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), color=(10, 200, 80)).save(buf, format="PNG")
    return buf.getvalue()


def test_rewrites_an_oversized_row_and_leaves_a_small_one_alone(tmp_path):
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0128(conn)
    oversized_id = _seed_media(conn, _oversized_jpeg(), "image/jpeg")
    small_data = _small_png()
    small_id = _seed_media(conn, small_data, "image/png")
    conn.commit()
    conn.close()

    count = migrate(db.db_path)
    assert count == 1

    conn = sqlite3.connect(db.db_path)
    oversized_data, oversized_mime = conn.execute(
        "SELECT data, mime_type FROM media WHERE id = ?", (oversized_id,)
    ).fetchone()
    out = Image.open(io.BytesIO(oversized_data))
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP
    assert oversized_mime == "image/jpeg"

    small_row = conn.execute(
        "SELECT data, mime_type FROM media WHERE id = ?", (small_id,)
    ).fetchone()
    assert small_row == (small_data, "image/png")  # untouched, byte-identical
    conn.close()


def test_rewrites_an_oversized_animated_gif_and_keeps_every_frame(tmp_path):
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    _isolate_0128(conn)
    gif_data = _animated_gif(frame_count=3)
    gif_id = _seed_media(conn, gif_data, "image/gif")
    conn.commit()
    conn.close()

    migrate(db.db_path)

    conn = sqlite3.connect(db.db_path)
    data, mime_type = conn.execute(
        "SELECT data, mime_type FROM media WHERE id = ?", (gif_id,)
    ).fetchone()
    assert mime_type == "image/gif"
    out = Image.open(io.BytesIO(data))
    assert getattr(out, "n_frames", 1) == 3
    assert max(out.size) <= PennyConstants.IMAGE_LONG_EDGE_CAP
    conn.close()


def test_a_row_that_fails_to_normalize_is_left_unchanged(tmp_path, monkeypatch):
    """A row whose normalization raises must not take down the whole migration
    — it's left exactly as it was, and every other row still gets rewritten.
    Calls the migration's own ``up()`` directly (loaded by path, like every
    other hand-loaded migration test) so monkeypatching its module-level
    ``normalize_image`` name affects only this one call."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db.db_path)
    broken_data = b"not-actually-an-image-but-pretend-it-raises"
    broken_id = _seed_media(conn, broken_data, "image/jpeg")
    oversized_id = _seed_media(conn, _oversized_jpeg(), "image/jpeg")
    conn.commit()

    module = _load_migration_module()

    def _boom(data, mime_type):
        if data == broken_data:
            raise ValueError("simulated decode failure")
        return _real_normalize_image(data, mime_type)

    monkeypatch.setattr(module, "normalize_image", _boom)
    module.up(conn)
    conn.close()

    conn = sqlite3.connect(db.db_path)
    broken_row = conn.execute(
        "SELECT data, mime_type FROM media WHERE id = ?", (broken_id,)
    ).fetchone()
    assert broken_row == (broken_data, "image/jpeg")  # left unchanged
    oversized_data, _ = conn.execute(
        "SELECT data, mime_type FROM media WHERE id = ?", (oversized_id,)
    ).fetchone()
    out = Image.open(io.BytesIO(oversized_data))
    assert max(out.size) == PennyConstants.IMAGE_LONG_EDGE_CAP  # still rewritten
    conn.close()
