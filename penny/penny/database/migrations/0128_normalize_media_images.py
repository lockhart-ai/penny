"""Normalize every stored ``media`` row to the live size/format rules (#2780).

Images were stored and passed around at full size — page preview images
made up most of the database, and generated images rode at their raw
output size. This migration rewrites every existing ``media`` row through
the same ``normalize_image`` helper ``MediaStore.put`` and the chat ingress
boundary now enforce going forward (a 1024px long-edge cap, a ~500KB byte
budget for static images, animated GIFs downscaled frame-by-frame and kept
animated, never a result larger than the row it replaces), then reclaims
the freed space with VACUUM.

Imports ``penny.image_utils.normalize_image`` rather than reimplementing
the Pillow logic inline here. Most migrations avoid reaching into the live
app (e.g. 0003 repacks embeddings with its own ``struct`` calls instead of
importing ``penny.llm.embeddings``) — the exception here is deliberate: the
issue calls for exactly one normalizing implementation, and duplicating the
Pillow resize/re-encode logic in two places would only risk the two
silently drifting apart. The trade-off that comes with the exception: this
migration runs once, against whatever ``normalize_image``'s rules are *at
the moment this migration runs* — a later change to those rules is never
retroactively re-applied to a row this migration already rewrote (nor to
any row written by ``MediaStore.put`` before that later change). Only a
fresh migration or a direct backfill revisits old rows.
"""

from __future__ import annotations

import logging
import sqlite3

from penny.image_utils import normalize_image

logger = logging.getLogger(__name__)

# Keyset page size — bounds how many decoded images this migration ever holds
# in memory at once, and how long between its own intermediate commits.
_BATCH_SIZE = 200


def up(conn: sqlite3.Connection) -> None:
    """Rewrite every ``media`` row's ``data``/``mime_type`` through
    ``normalize_image`` in bounded batches, then VACUUM once, only if
    anything actually shrank."""
    if not _table_exists(conn, "media"):
        return
    rows_seen, rows_changed, bytes_before, bytes_after = _rewrite_all(conn)
    conn.commit()
    if rows_changed:
        conn.execute("VACUUM")
    conn.commit()
    logger.info(
        "0128 normalized media: visited %d row(s), rewrote %d, %d -> %d bytes",
        rows_seen,
        rows_changed,
        bytes_before,
        bytes_after,
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    )


def _rewrite_all(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
    """Keyset-paginate through every ``media`` row (``id > last_id``, ordered,
    bounded), rewriting each and committing per batch. Returns
    ``(rows_seen, rows_changed, bytes_before, bytes_after)`` across the whole
    table, logging progress after every batch."""
    last_id = 0
    rows_seen = rows_changed = bytes_before = bytes_after = 0
    while True:
        batch = conn.execute(
            "SELECT id, data, mime_type FROM media WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, _BATCH_SIZE),
        ).fetchall()
        if not batch:
            break
        batch_changed = 0
        for row_id, data, mime_type in batch:
            last_id = row_id
            rows_seen += 1
            bytes_before += len(data)
            new_data, new_mime = _normalize_one(row_id, data, mime_type)
            bytes_after += len(new_data)
            if new_data != data or new_mime != mime_type:
                conn.execute(
                    "UPDATE media SET data = ?, mime_type = ? WHERE id = ?",
                    (new_data, new_mime, row_id),
                )
                rows_changed += 1
                batch_changed += 1
        conn.commit()
        logger.info(
            "0128 batch through media id %d: %d/%d row(s) in this batch rewritten"
            " (%d row(s) total so far)",
            last_id,
            batch_changed,
            len(batch),
            rows_seen,
        )
    return rows_seen, rows_changed, bytes_before, bytes_after


def _normalize_one(row_id: int, data: bytes, mime_type: str) -> tuple[bytes, str]:
    """``normalize_image``, but a row whose normalization raises is left
    unchanged rather than failing the whole migration over one bad row."""
    try:
        return normalize_image(data, mime_type)
    except Exception:
        logger.warning(
            "0128 media row %d failed to normalize; left unchanged", row_id, exc_info=True
        )
        return data, mime_type
