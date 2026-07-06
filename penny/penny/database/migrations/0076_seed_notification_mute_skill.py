"""Seed a "mute / unmute notifications" skill.

Type: data

The ``/mute`` + ``/unmute`` slash commands retired into two chat-surface tools
(``notifications_mute`` / ``notifications_unmute``, over the ``MuteState`` row).
Live-model eval showed the dispatch is *mostly* reliable from natural language,
but one resume phrasing ("you can start messaging me again") slipped: the model
reasoned correctly ("call notifications_unmute") yet returned prose without
emitting the call — the classic gpt-oss single-turn tool-call omission.  This
skill closes that gap the way migration 0070 did for the change-source case: a
clean positive recipe whose numbered STEP is literally "call the tool", so the
intent dispatches reliably instead of trailing off into a narrated reply.  It
also names the no-fire boundary (a casual mention of a quiet day is not a mute
request) so the skill can't push the model toward over-firing.

A skill is a TRIGGER (intent + example phrasings) + numbered tool-call STEPS in
the one shape migration 0069 established.  This is an operate-the-system skill
(no source collection), so the skills reconcile loop leaves it alone, exactly
like scope/cadence/flip/archive.  Seeded ``author='system'`` (like every 0043
seed and 0070), so the 0069 ``author='skills'`` scrub never touches it.
Idempotent — INSERT OR IGNORE.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

_MUTE_NOTIFICATIONS = """TRIGGER
User wants to pause or resume Penny's proactive notifications — the thought discoveries,
news, and collection updates Penny sends on her own initiative. Replies to the user's own
messages are never affected. Two directions:
- PAUSE: "stop messaging me for a while", "quiet down", "no updates for now", "mute
  notifications", "take a break from pinging me".
- RESUME: "you can message me again", "turn your updates back on", "start pinging me
  again", "unmute".
A casual mention of things being quiet ("it's been a quiet day") is NOT a request — do
nothing.

STEPS
1. To PAUSE, call notifications_mute (no arguments). To RESUME, call notifications_unmute
   (no arguments).
2. Reply from the tool result — confirm notifications are now muted, or back on."""


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory_entry "
        "(memory_name, key, content, author, key_embedding, content_embedding, created_at) "
        "VALUES ('skills', ?, ?, 'system', NULL, NULL, ?)",
        ("Mute or unmute notifications", _MUTE_NOTIFICATIONS, now),
    )
    conn.commit()
