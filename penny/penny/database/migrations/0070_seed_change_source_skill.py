"""Seed a "change a collection's source" skill.

Type: data

The seeded skills cover changing WHAT a collection gathers ("Update collection
scope" — add a topic, narrow, broaden) but nothing frames changing WHERE it
gathers from — pointing an existing collection at a specific URL/site — as a
``collection_update`` of the ``extraction_prompt``.  Live-model eval reproduced
the gap: asked the way users actually phrase it ("for X you should browse this
url to find good ones: <url>"), the chat agent read it as a one-shot "browse
that now" and never reconfigured the collector — 0/8 samples landed the URL in
the prompt.  Adding this skill lifts the same case to 8/8.

A skill is a clean positive recipe — TRIGGER (intent + example phrasings) +
numbered tool-call STEPS — in the one shape migration 0069 established.  This is
an operate-the-system skill (no source collection), so the skills reconcile loop
leaves it alone, exactly like scope/cadence/flip/archive.

Seeded with ``author='system'`` (like every 0043 seed), so the 0069
``author='skills'`` scrub never touches it.  Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

_CHANGE_SOURCE = """TRIGGER
User wants an existing collection to gather from a specific source they name — a URL or
site — instead of where it looks now. This sets where the collector browses on every
future run, so it's a change to the collection's extraction_prompt. Example phrasings:
- "for X, you should browse this url to find good ones: <url>"
- "get X from this site instead: <url>"
- "make X pull from <url> from now on"

STEPS
1. memory_metadata("[X]") — read the collection's current extraction_prompt.
2. collection_update("[X]") with extraction_prompt: the full rewritten recipe whose first
   browse step targets the URL the user gave, verbatim; keep the rest of the recipe
   (extract, write, done) intact.
3. Summarize from the echo — name the collection and its new source."""


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory_entry "
        "(memory_name, key, content, author, key_embedding, content_embedding, created_at) "
        "VALUES ('skills', ?, ?, 'system', NULL, NULL, ?)",
        ("Change collection source", _CHANGE_SOURCE, now),
    )
    conn.commit()
