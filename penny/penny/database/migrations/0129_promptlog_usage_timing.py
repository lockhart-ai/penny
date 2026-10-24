"""Add ``promptlog`` token/timing columns and backfill them from stored responses (#2793).

Type: schema + data

Each ``promptlog`` row only ever recorded one total ``duration_ms``, so the addon
could not show prompt-processing speed, true generation speed, or how much of the
prompt the server's cache served. This adds five nullable columns —
``prompt_tokens``, ``cached_tokens``, ``completion_tokens``, ``prefill_ms``,
``generation_ms`` — and backfills every existing row from the ``usage`` object
already stored inside its own ``response`` JSON, through the exact same mapping
(``penny.llm.models.usage_timing_from_response``) the live write path now runs on
every new call. One mapping, read here and at write time, never two: duplicating
the runner-shape logic would only risk the two drifting apart (the same exception
``0128_normalize_media_images.py`` already makes for ``normalize_image``).

A row whose ``response`` carries no ``usage`` at all (no reply ever logged, or a
runner that reports none) backfills every column to NULL — never a guessed zero.
Keyset-paginated in bounded batches (id > last_id, ordered, committed per batch),
the same shape ``0128`` uses, so this never holds one long transaction over a
large table nor decodes the whole table's JSON into memory at once.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from penny.llm.models import UsageTiming, usage_timing_from_response

logger = logging.getLogger(__name__)

# Keyset page size — bounds how many decoded responses this migration ever holds
# in memory at once, and how long between its own intermediate commits.
_BATCH_SIZE = 500

_NEW_COLUMNS = (
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "prefill_ms",
    "generation_ms",
)


def up(conn: sqlite3.Connection) -> None:
    """Add the five columns if missing, then backfill every row in bounded
    batches, logging total rows visited plus a populated-count per field."""
    if not _table_exists(conn, "promptlog"):
        return
    _add_missing_columns(conn)
    conn.commit()
    rows_seen, populated = _backfill_all(conn)
    conn.commit()
    logger.info(
        "0129 backfilled promptlog usage timing: visited %d row(s); populated"
        " prompt_tokens=%d cached_tokens=%d completion_tokens=%d prefill_ms=%d"
        " generation_ms=%d",
        rows_seen,
        populated["prompt_tokens"],
        populated["cached_tokens"],
        populated["completion_tokens"],
        populated["prefill_ms"],
        populated["generation_ms"],
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    )


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(promptlog)").fetchall()}
    for column in _NEW_COLUMNS:
        if column not in columns:
            conn.execute(f"ALTER TABLE promptlog ADD COLUMN {column} INTEGER")


def _backfill_all(conn: sqlite3.Connection) -> tuple[int, dict[str, int]]:
    """Keyset-paginate through every ``promptlog`` row, mapping each row's stored
    ``response`` through the shared ``usage_timing_from_response`` and writing the
    five columns. Returns ``(rows_seen, populated_counts_by_field)``."""
    last_id = 0
    rows_seen = 0
    populated = dict.fromkeys(_NEW_COLUMNS, 0)
    while True:
        batch = conn.execute(
            "SELECT id, response FROM promptlog WHERE id > ? ORDER BY id LIMIT ?",
            (last_id, _BATCH_SIZE),
        ).fetchall()
        if not batch:
            break
        for row_id, response_text in batch:
            last_id = row_id
            rows_seen += 1
            timing = _timing_for_row(row_id, response_text)
            _apply_timing(conn, row_id, timing, populated)
        conn.commit()
        logger.info("0129 batch through promptlog id %d: %d row(s) so far", last_id, rows_seen)
    return rows_seen, populated


def _timing_for_row(row_id: int, response_text: str | None) -> UsageTiming:
    """The row's ``UsageTiming``, or all-NULL for a response that fails to parse
    (left unchanged rather than failing the whole migration over one bad row)."""
    if not response_text:
        return UsageTiming()
    try:
        return usage_timing_from_response(json.loads(response_text))
    except json.JSONDecodeError, AttributeError:
        logger.warning("0129 promptlog row %d has an unparseable response; left unchanged", row_id)
        return UsageTiming()


def _apply_timing(
    conn: sqlite3.Connection, row_id: int, timing: UsageTiming, populated: dict[str, int]
) -> None:
    conn.execute(
        "UPDATE promptlog SET prompt_tokens = ?, cached_tokens = ?, completion_tokens = ?,"
        " prefill_ms = ?, generation_ms = ? WHERE id = ?",
        (
            timing.prompt_tokens,
            timing.cached_tokens,
            timing.completion_tokens,
            timing.prefill_ms,
            timing.generation_ms,
            row_id,
        ),
    )
    for field in _NEW_COLUMNS:
        if getattr(timing, field) is not None:
            populated[field] += 1
