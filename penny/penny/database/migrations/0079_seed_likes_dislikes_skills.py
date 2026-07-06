"""Seed skills that dispatch like/dislike requests onto the memory collections.

Type: data

The ``/like`` / ``/unlike`` / ``/dislike`` / ``/undislike`` commands retired
(epic #1445, issue #1451) in favour of the chat agent driving the memory tools
(``collection_write`` / ``collection_delete_entry`` / ``collection_read_latest``)
against the ``likes`` and ``dislikes`` collections — the same collections the
ambient extractor fills and recall reads.  These skills are the NL triggers that
make the dispatch reliable — a TRIGGER (intent + example phrasings) plus numbered
tool-call STEPS, in the one clean shape migration 0069 established.

They are operate-the-system skills (no source collection), so the skills reconcile
loop leaves them alone, exactly like the mute/schedule/scope/cadence skills.

The ``preference`` table is untouched — its fate is #1301.  Retirement lands the
four commands onto the collections only; nothing here reads or writes ``preference``.

Seeded with ``author='system'`` (like every 0043 seed), so the 0069
``author='skills'`` scrub never touches them.  Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

_ADD = """TRIGGER
User states something they like or dislike and wants it remembered — a standing
preference, not a passing remark. Two valences:
- LIKE (positive): "I'm really into bouldering", "I love spicy ramen", "I'm a big
  fan of synthwave", "add cold brew to my likes".
- DISLIKE (negative): "I can't stand crowded trains", "I really hate cilantro",
  "put loud open offices on my dislikes".
A neutral or passing opinion ("that movie was fine", "the coffee was okay today")
is NOT a standing preference — do nothing; the background extractor may still pick
it up on its own.

STEPS
1. Decide the valence: a positive sentiment goes to the "likes" collection, a
   negative one to "dislikes".
2. collection_write(memory=<"likes" or "dislikes">, entries=[{"key": <a short topic
   name, e.g. "bouldering">, "content": <the user's statement in their own words>}]).
   Never create a new collection — write to the existing "likes"/"dislikes".
3. Confirm back to the user in plain language what you recorded and where (e.g.
   "Got it — added bouldering to your likes")."""

_REMOVE = """TRIGGER
User wants to drop a like or dislike you're holding for them. Example phrasings:
- "actually, forget about bouldering"
- "I don't care about synthwave anymore"
- "stop tracking that I hate cilantro"
- "take crowded trains off my dislikes"
The user names the topic in their OWN words — they will NOT quote the exact stored
key, so match by MEANING, never by exact text.

STEPS
1. collection_read_latest(memory=<"likes" or "dislikes">) to see the current entries
   and their keys. Pick likes or dislikes by the valence the user is retracting; if
   unclear, read both.
2. Find the entry whose key or content matches the topic the user named BY MEANING
   (e.g. the user says "the violin" and the stored key is "learning the violin").
3. collection_delete_entry(memory=<the collection>, key=<the matched entry's exact
   key>). Never guess a key the read didn't show; never delete by number or position.
4. Confirm exactly what you removed (e.g. "Done — removed bouldering from your
   likes"). If nothing matched, say so instead of deleting something unrelated."""

_LIST = """TRIGGER
User asks what preferences you're holding for them. Example phrasings:
- "what do you think I'm into?"
- "what do I like?"
- "what are my dislikes?"
- "list everything I've said I hate"

STEPS
1. collection_read_latest(memory="likes") for likes, collection_read_latest(memory=
   "dislikes") for dislikes — read whichever valence the user asked about, or both if
   they asked generally.
2. Report the entries back in plain language, or tell the user the list is empty."""

_SKILLS = {
    "Record a like or dislike": _ADD,
    "Forget a like or dislike": _REMOVE,
    "List likes or dislikes": _LIST,
}


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    for key, content in _SKILLS.items():
        conn.execute(
            "INSERT OR IGNORE INTO memory_entry "
            "(memory_name, key, content, author, key_embedding, content_embedding, created_at) "
            "VALUES ('skills', ?, ?, 'system', NULL, NULL, ?)",
            (key, content, now),
        )
    conn.commit()
