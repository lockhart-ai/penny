"""Seed the notifier consumer collection.

Type: data

The consumer side of the pub/sub layer.  Each cycle it drains the published
stream — ``read_published_latest(n=1)`` returns the oldest entry it hasn't
delivered yet across every ``published`` collection — grounds it in past
conversation, and sends the user one message about it.  It owns a per-source
cursor (advanced on a successful cycle), so each find is delivered exactly once
and the entry stays in its durable home collection; there is no move/archive.

It's an ordinary collection — identified as a consumer purely by its prompt
calling ``read_published_latest`` (the dispatcher's gate wakes it only when a
published source is past its cursor, and exempts it from auto-throttle).  Seeded
``inclusion='never'`` (internal — never surfaces in chat) and ``published=0``
(it's a consumer, not a source).

The extraction_prompt below is byte-identical to ``NOTIFIER_EXTRACTION_PROMPT``
in ``penny/tests/eval/fixtures.py`` was — the eval drives this *seeded* prompt
directly (a fresh eval DB runs migrations), so what's validated is exactly what
ships.  Cadence is 600s ≈ the autonomous send cooldown, so the consumer doesn't
outpace the drainer; tune later if a backlog needs faster catch-up.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

NOTIFIER_DESCRIPTION = "Delivers new finds from published collections to the user."
NOTIFIER_INTENT = "Tell me about new finds from the collections I asked you to keep me posted on."
NOTIFIER_EXTRACTION_PROMPT = (
    "Deliver a single fresh find to the user from the collections that publish to you.\n"
    "1. read_published_latest(n=1) — fetch one new, not-yet-delivered entry; it names its "
    "source collection and that collection's intent (<key> holds the entry key).  If it "
    'returns nothing, call done(success=true, summary="nothing new to deliver") and stop.\n'
    '2. read_similar(anchor=<entry content>, memory="penny-messages", k=5) — pull up to five '
    "of our past replies semantically close to the find.\n"
    '3. read_similar(anchor=<entry content>, memory="user-messages", k=5) — pull up to five '
    "user messages that relate to it.\n"
    "4. Compose a natural-sounding message, framed by the source collection's intent:\n"
    "   • Open with a friendly greeting,\n"
    "   • State the key details of the find (names, specs, dates, source URLs),\n"
    "   • Refer back to the relevant excerpts from steps 2-3,\n"
    "   • Include an explicit URL from the entry so the user can click to read more,\n"
    "   • Close with an emoji.\n"
    "5. send_message(content=<composed message>) — deliver it.\n"
    "6. done(success=true, summary=<one sentence on what you delivered>)."
)

# 600s ≈ the autonomous send cooldown, so the consumer doesn't outpace the drainer.
_INTERVAL_SECONDS = 600


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO memory "
        "(name, type, description, inclusion, recall, archived, created_at, "
        "extraction_prompt, collector_interval_seconds, base_interval_seconds, "
        "intent, published) "
        "VALUES ('notifier', 'collection', ?, 'never', 'recent', 0, ?, ?, ?, ?, ?, 0)",
        (
            NOTIFIER_DESCRIPTION,
            now,
            NOTIFIER_EXTRACTION_PROMPT,
            _INTERVAL_SECONDS,
            _INTERVAL_SECONDS,
            NOTIFIER_INTENT,
        ),
    )
    conn.commit()
