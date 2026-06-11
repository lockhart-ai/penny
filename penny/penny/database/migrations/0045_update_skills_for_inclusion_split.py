"""Update seeded skills for the inclusion/recall split (migration 0044).

Migration 0044 split the single ``recall`` flag into ``inclusion`` (stage-1
collection routing: always / relevant / never) and ``recall`` (stage-2 entry
rendering: all / relevant / recent).  Five seeded skills from 0043 still
teach the old single-flag model — most damagingly ``recall: "off"`` for
silent collections, which is no longer a valid enum value on the create /
update tools, so a model following those skills would emit a rejected call.

Following the 0040–0042 precedent (fix prompt text via a new data
migration), this rewrites the affected skill entries in place:

  * Research collection — notify on new finds  → inclusion+recall fields
  * Research collection — silent               → silent = inclusion "never"
  * Scheduled digest                           → inclusion+recall fields
  * Update collection scope                    → description is now the
    stage-1 routing anchor, not cosmetic
  * Flip silent ↔ notify                       → flip inclusion, not recall

Content embeddings are nulled on the rewritten rows so the startup backfill
re-vectorizes them (the content changed; a stale vector would mis-rank).
Idempotent — keyed UPDATEs overwrite with identical content on re-run.
"""

from __future__ import annotations

import sqlite3

REPLACEMENTS: dict[str, str] = {
    "Research collection — notify on new finds": """TRIGGER
User wants ongoing research with notifications on new finds. Example phrasings:
- "research X for me, ping me when you find stuff"
- "follow X and let me know about new things"
- "build me a list of X, send me updates"
- "keep an eye on X, tell me when there's something new"
- "i'm going to X next week, find me Y" (with notification ask)

STEPS

This is a single-turn flow. Call the tool, summarize the result, ask the
user if it looks right.

1. Call collection_create with:
   - name: short slug from the topic
   - description: one-line summary of the SUBJECT MATTER (this is the
     routing anchor — the collection only surfaces in chats that match
     this text, so describe the topic, not the mechanism)
   - inclusion: "relevant"
   - recall: "relevant"
   - collector_interval_seconds: 3600 (default; match user's cadence
     words if they gave any — "every 30 min" → 1800, "daily" → 86400)
   - extraction_prompt (numbered, name each tool):
     > Collect [topic] — [scope].
     > 1. browse(...) — a few queries targeting [scope]; read actual pages
     > 2. log_read_next("user-messages") — pick up corrections
     > 3. Each entry: key = item name; content = name + description + URL
     > 4. collection_write("[name]", entries=[...])
     > 5. If write succeeded, send_message: one-sentence "found a new X" + URL
     > 6. If a message flags an entry as wrong, update_entry or collection_delete_entry
     > 7. done(). If nothing new, just done().

2. The return value is a structured echo of what got stored (name,
   interval, inclusion, recall, description, full extraction_prompt).
   Reply to the user with a one-sentence summary using ONLY values from
   the echo — name the collection, the cadence in human terms, what it's
   tracking, that it pings on new finds. End by asking if they want tweaks.
   Example: "Made `mechanical-keyboards` — checks every hour for new
   keyboards, pings you per find. Want any tweaks?"

If they say yes / no changes → done. If they want changes, the next
turn will surface the appropriate update skill (scope / silent flip /
cadence) which handles the change.
""",
    "Research collection — silent": """TRIGGER
User wants ongoing research WITHOUT notifications — they'll check in
themselves. Example phrasings:
- "research X for me, silent, i'll check in"
- "research X but don't ping me"
- "track X quietly"
- "build me a list of X, no notifications"
- "keep tabs on X, i'll ask when i want updates"

STEPS

Single-turn act-then-echo.

1. Call collection_create with:
   - name, description, collector_interval_seconds: as in the notify
     variant
   - inclusion: "never" (silent — never surfaced in chat; the collector
     still runs in the background)
   - recall: "relevant" (entry mode is moot while inclusion is "never",
     but the field is required)
   - extraction_prompt with NO send_message step:
     > Collect [topic] — [scope].
     > 1. browse(...) — a few queries targeting [scope]; read actual pages
     > 2. log_read_next("user-messages") — pick up corrections
     > 3. Each entry: key = item name; content = name + description + URL
     > 4. collection_write("[name]", entries=[...])
     > 5. If a message flags an entry as wrong, update_entry or collection_delete_entry
     > 6. done(). If nothing new, just done().

2. Summarize back from the echo. Mention silent explicitly so the user
   knows there'll be no pings.
   Example: "Made `X` — checks every hour for [topic], silent (ask any
   time). Want any tweaks?"
""",
    "Scheduled digest": """TRIGGER
User wants a periodic summary delivered on a schedule (daily digest,
weekly roundup, hourly check + once-a-day summary). Example phrasings:
- "send me a daily digest of X at 6pm"
- "give me a morning summary of X each day"
- "check X hourly, summarize at end of day"
- "weekly roundup of Y"

STEPS

Single-turn act-then-echo.

1. Call collection_create with:
   - name: slug for the digest collection
   - description: one-line summary of the subject matter (the routing
     anchor)
   - inclusion: "relevant"
   - recall: "relevant"
   - collector_interval_seconds: check cadence (NOT delivery cadence —
     "daily digest at 6pm" still checks hourly = 3600)
   - extraction_prompt (date-keyed entries, scheduled send):
     > Collect [topic] — produce a [delivery cadence] digest.
     > 1. browse(...) — a few queries for today's [topic] items
     > 2. Today's date is the entry key (YYYY-MM-DD). If the entry
     >    exists, update_entry to add new items; otherwise
     >    collection_write a new entry for today.
     > 3. Only send_message at the scheduled delivery time (e.g. 18:00
     >    UTC for 6pm); at other times, write and done() without sending.
     > 4. done().

2. Summarize back from the echo, naming both the check cadence and the
   delivery time.
   Example: "Made `X-digest` — checks hourly for [topic], digest at
   [time]. Want any tweaks?"
""",
    "Update collection scope": """TRIGGER
User wants to change WHAT an existing collection collects — add a topic,
drop a topic, swap focus. Example phrasings:
- "add Y to that collection"
- "drop Y from X"
- "from now on focus on Z instead"
- "broaden X to also cover Y"
- "stop tracking Z, just focus on W"

CRITICAL: Scope lives in the extraction_prompt BODY — that drives the
collector. The description must ALSO be updated to match: it is the
stage-1 routing anchor, so a stale description keeps surfacing the
collection for the old topic and misses the new one.

STEPS

Single-turn: read current state, update, summarize.

1. Call collection_metadata to get the existing extraction_prompt body
   (the description visible in recall isn't the full prompt).

2. Call collection_update with the FULL rewritten extraction_prompt
   body (replace, not diff). Preserve the structural shape (numbered
   steps, named tools). Apply the user's scope change — add the new
   topic, drop the old one, etc. Also update description to the new
   subject matter so routing follows the new scope.

3. Summarize back from the update echo — name the collection and the
   new scope.
   Example: "Updated `X` — now collects [new scope]. Want anything
   else changed?"
""",
    "Flip silent ↔ notify": """TRIGGER
User wants to change whether an existing collection pings them on
finds. Example phrasings:
- "stop pinging me about new X"
- "go silent on X, i'll check in"
- "start pinging me again about X"
- "let me know when there's new stuff in X"

CRITICAL: Silent mode requires BOTH inclusion="never" AND removing the
send_message step from the extraction_prompt body. inclusion controls
ambient surfacing; send_message controls active pings. Leaving the
body alone means the collector keeps paging you every cycle even with
inclusion="never".

STEPS

Single-turn: read, update, summarize.

1. Call collection_metadata for the existing body.

2. Call collection_update with BOTH:
   - inclusion: "never" (silent) or "relevant" (notify)
   - extraction_prompt: rewritten body without (or with) the
     send_message step. Keep all other steps intact.

3. Summarize from the echo.
   Example: "Silenced `X` — no more pings on new finds. Ask any time."
""",
}


def up(conn: sqlite3.Connection) -> None:
    for key, content in REPLACEMENTS.items():
        conn.execute(
            "UPDATE memory_entry SET content = ?, content_embedding = NULL"
            " WHERE memory_name = 'skills' AND key = ?",
            (content, key),
        )
    conn.commit()
