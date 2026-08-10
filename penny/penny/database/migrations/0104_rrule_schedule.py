"""Collapse the trigger union onto one ``memory.schedule`` RRULE column (#1857).

Type: schema

The four-form trigger union (``every <seconds>`` | ``once at <ISO> [xN]`` | ``on
advance of <log>`` | ``cron <5-field expression>``) and the collector auto-throttle
are replaced by ONE schedule grammar: an RRULE string the collector gates on
directly.  So six columns retire and one arrives.

* **ADD** ``schedule`` (TEXT, nullable) — the whole schedule, one RRULE line with
  an optional leading ``DTSTART:`` line.  ``COUNT=`` / ``UNTIL=`` are lifted into
  the existing ``max_runs`` / ``expires_at`` columns at parse time, so those two
  stay exactly as they were.
* **DROP** ``collector_interval_seconds`` (the cadence the rule replaces),
  ``run_at`` (folded into ``DTSTART``), ``source_log`` (the ``on advance of``
  form, retired pending its own field + eval coverage), ``cron_expression`` (the
  cron form the rule subsumes), and the auto-throttle pair
  ``base_interval_seconds`` / ``consecutive_idle_runs`` (the whole mechanism is
  deleted — schedules run as stated, and rate protection lives in the send
  cooldown).

**Existing trigger data is NOT converted** — code-owner ruling on #1857:
deployments reset their databases after this arc, so a conversion would be
guesswork nobody would keep.  The drop is therefore mechanical, but it is NOT
silent: every collection that had a trigger is logged by name with what it
carried, and the total is logged as a warning, so a collection that stops
dispatching is diagnosable rather than mysterious.  Each such collection needs a
``schedule`` set through ``collection_set`` before it runs again.

Universal + idempotent.  On the ``create_tables``-first path (a fresh install /
the test schema template) the model already carries ``schedule`` and carries none
of the six, and the earlier guarded migrations that ADD them (0032 / 0053 / 0082 /
0091 / 0098) re-provision them for the intervening migrations that read them — so
this migration finds them present and drops them, converging both paths on the
same schema.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# The columns the one schedule column replaces, each with what it used to mean —
# named in the log line so a dropped value is diagnosable after the fact.
_RETIRED_COLUMNS = {
    "collector_interval_seconds": "recurring cadence, seconds",
    "run_at": "delayed / one-shot start time",
    "source_log": "on-advance source log",
    "cron_expression": "cron schedule",
    "base_interval_seconds": "auto-throttle snap-back cadence",
    "consecutive_idle_runs": "auto-throttle idle counter",
}

# Which of those columns being set means a collection actually HAD a trigger — the
# four schedule members.  The throttle pair is excluded: ``consecutive_idle_runs`` is
# NOT NULL DEFAULT 0, so testing it would select every row in the table (logs included)
# and report a lost trigger for every memory that never had one.
_TRIGGER_COLUMNS = ("collector_interval_seconds", "run_at", "source_log", "cron_expression")

# What the model-facing schedule argument now takes — quoted in the log so the
# recovery is a copy rather than a lookup.
_EXAMPLE_SCHEDULE = "FREQ=HOURLY"


def up(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "memory"):
        return
    _add_schedule_column(conn)
    _log_dropped_triggers(conn)
    _drop_retired_columns(conn)
    conn.commit()


def _add_schedule_column(conn: sqlite3.Connection) -> None:
    """Add the nullable ``schedule`` column (idempotent — present already on a
    ``create_tables``-first database, where the model declares it)."""
    if "schedule" not in _columns(conn, "memory"):
        conn.execute("ALTER TABLE memory ADD COLUMN schedule TEXT")


def _log_dropped_triggers(conn: sqlite3.Connection) -> None:
    """Name every collection whose trigger this migration drops, and what it held.

    Data preservation is waived, but disappearance must not be silent: a collector
    that stops running is otherwise indistinguishable from one that has nothing to
    do.  Each affected collection is logged at WARNING with its old trigger values,
    plus one summary line naming the fix."""
    columns = _columns(conn, "memory")
    present = [column for column in _RETIRED_COLUMNS if column in columns]
    triggers = [column for column in _TRIGGER_COLUMNS if column in columns]
    if not present or not triggers:
        return
    # Interpolated, not bound: these are column NAMES (SQLite binds values only), and
    # every one is a key of this module's own frozen constants — never caller input.
    selected = ", ".join(present)
    condition = " OR ".join(f"{column} IS NOT NULL" for column in triggers)
    rows = conn.execute(f"SELECT name, {selected} FROM memory WHERE {condition}").fetchall()
    for row in rows:
        held = ", ".join(
            f"{column}={value} ({_RETIRED_COLUMNS[column]})"
            for column, value in zip(present, row[1:], strict=True)
            if value is not None
        )
        logger.warning("Dropping trigger config for collection '%s': %s", row[0], held)
    if rows:
        logger.warning(
            "%d collection(s) lost their trigger — schedules are now one RRULE and old "
            "triggers are NOT converted (#1857). Each needs a new one before it runs "
            "again: collection_set(name=<collection>, schedule='%s').",
            len(rows),
            _EXAMPLE_SCHEDULE,
        )


def _drop_retired_columns(conn: sqlite3.Connection) -> None:
    """Drop each retired column that is still present (idempotent)."""
    columns = _columns(conn, "memory")
    for column in _RETIRED_COLUMNS:
        if column in columns:
            conn.execute(f"ALTER TABLE memory DROP COLUMN {column}")


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )
