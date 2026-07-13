"""The skill substrate table + the hand-authored seed library (#1590).

Type: schema + data

Stage ④ of the collector core (#1562 ⭐ rev. 3; epic #1554).  Adds the dedicated
``skill`` table — versionless, one row per name (a re-teach REPLACES the row;
collections carry the rendered TEXT snapshotted at creation, so a skill edit never
retroactively changes an instantiation, and the version pin had no remaining job).
A skill's ``steps`` are the ``LoggedToolCall`` shape (#1578) as JSON, its ``holes``
the declared parameters; #1591's ``collection_create`` renders steps + bound params
into a collection's numbered TEXT ``extraction_prompt``.

Also seeds a SMALL hand-authored library covering the recurring production shapes
(watch-a-page-field, feed/news digest, research-and-notify).  **The seeds are the
sanctioned exception to certified-by-execution**: the invariant "a skill contains
only calls that actually succeeded" governs the ``skill_create`` path (enforced in
``penny.tools.skill_tools``), and a hand-authored seed's demonstration is its
authoring.  Seeds are universal data — ``author='system'``, known titles, no
provenance run — so a fresh install and every deployment get the same library.
``description_embedding`` is NULL (a migration can't embed); #1591's resolution
wiring backfills it, exactly like a memory row's description anchor.

Schema is idempotent (``CREATE TABLE IF NOT EXISTS`` — ``create_tables()`` runs
first on a fresh DB and materialises it from the model, so this is a no-op there
and an add on a production copy that predates it).  Seeds use ``INSERT OR IGNORE``
keyed on the title.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime


def _hole(name: str) -> dict:
    return {"name": name, "required": True}


def _sub_hole(path: list, hole: str) -> dict:
    return {"path": path, "kind": "hole", "hole": hole, "step": None}


def _sub_binding(path: list, step: int) -> dict:
    return {"path": path, "kind": "binding", "hole": None, "step": step}


def _browse_step(
    ordinal: int, query_hole: str, extract: str | None, extract_hole: str | None
) -> dict:
    """A ``browse`` step whose single query is a hole; ``extract`` is either a
    constant instruction or a hole (``extract_hole``)."""
    subs = [_sub_hole(["queries", 0], query_hole)]
    arguments: dict = {"queries": ["<page>"]}
    if extract_hole is not None:
        arguments["extract"] = "<field>"
        subs.append(_sub_hole(["extract"], extract_hole))
    else:
        arguments["extract"] = extract
    return {
        "ordinal": ordinal,
        "source_ordinal": ordinal,
        "tool": "browse",
        "arguments": arguments,
        "substitutions": subs,
    }


def _write_step(ordinal: int, key_hole: str, from_step: int) -> dict:
    """A ``collection_write`` step: the target collection + the entry key are holes;
    the content is bound to an earlier step's result."""
    return {
        "ordinal": ordinal,
        "source_ordinal": ordinal,
        "tool": "collection_write",
        "arguments": {
            "memory": "<collection>",
            "entries": [{"key": "<item>", "content": "<value>"}],
        },
        "substitutions": [
            _sub_hole(["memory"], "collection"),
            _sub_hole(["entries", 0, "key"], key_hole),
            _sub_binding(["entries", 0, "content"], from_step),
        ],
    }


_WATCH_PAGE_FIELD = {
    "name": "Watch a page field",
    "intent": "Keep an eye on one value on a page and record it whenever it changes.",
    "description": "Watch a single field on a web page and save its current value.",
    "holes": [_hole("url"), _hole("field"), _hole("collection")],
    "steps": [
        _browse_step(1, query_hole="url", extract=None, extract_hole="field"),
        _write_step(2, key_hole="field", from_step=1),
    ],
}

_NEWS_DIGEST = {
    "name": "Feed or news digest",
    "intent": "Search a topic on a cadence and collect the notable new items.",
    "description": "Gather notable new items about a topic into a collection.",
    "holes": [_hole("topic"), _hole("collection")],
    "steps": [
        _browse_step(
            1,
            query_hole="topic",
            extract="the notable new items and their sources",
            extract_hole=None,
        ),
        _write_step(2, key_hole="topic", from_step=1),
    ],
}

_RESEARCH_AND_NOTIFY = {
    "name": "Research and notify",
    "intent": (
        "Research a topic on a cadence, save notable new findings, and notify the "
        "user (set published=true when creating the collection)."
    ),
    "description": "Find notable new findings about a topic worth telling the user about.",
    "holes": [_hole("topic"), _hole("collection")],
    "steps": [
        _browse_step(
            1,
            query_hole="topic",
            extract="a notable new finding worth sharing, with its source",
            extract_hole=None,
        ),
        _write_step(2, key_hole="topic", from_step=1),
    ],
}

_SEEDS = [_WATCH_PAGE_FIELD, _NEWS_DIGEST, _RESEARCH_AND_NOTIFY]


def up(conn: sqlite3.Connection) -> None:
    _create_skill_table(conn)
    _seed_library(conn)
    conn.commit()


def _create_skill_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS skill ("
        "  name TEXT PRIMARY KEY,"
        "  steps TEXT NOT NULL,"
        "  holes TEXT NOT NULL,"
        "  intent TEXT NOT NULL,"
        "  description TEXT NOT NULL,"
        "  description_embedding BLOB,"
        "  source_run_id TEXT,"
        "  author TEXT NOT NULL,"
        "  created_at TIMESTAMP NOT NULL,"
        "  updated_at TIMESTAMP NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_author ON skill(author)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_source_run_id ON skill(source_run_id)")


def _seed_library(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    for seed in _SEEDS:
        conn.execute(
            "INSERT OR IGNORE INTO skill "
            "(name, steps, holes, intent, description, description_embedding, "
            " source_run_id, author, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, NULL, 'system', ?, ?)",
            (
                seed["name"],
                json.dumps(seed["steps"]),
                json.dumps(seed["holes"]),
                seed["intent"],
                seed["description"],
                now,
                now,
            ),
        )
