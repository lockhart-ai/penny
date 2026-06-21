"""Rewrite the research/notify skills for the ``published`` flag.

Type: data

The seeded skills (0043/0045) taught the old "notify = add a send_message step
to the collector body + flip inclusion" pattern.  Under pub/sub that's wrong:
a collector only gathers, and notification is the ``published`` flag (a separate
notifier consumer delivers each new entry once).  Live-model eval confirmed the
chat agent faithfully followed the old skills — setting ``published`` correctly
on create but ALSO bolting a send_message into the producer prompt, and editing
inclusion/prompt instead of the flag on "start/stop telling me".

This rewrites the seeded collection skills to the flag model:
  - notify → published: true, NO send_message in the body
  - silent → published: false
  - flip   → toggle published only, never touch the extraction_prompt
  - scope  → "add [things] to the collection" broadens the prompt + description (so the
             collector gathers them going forward), NOT hand-written entries; also fixes
             its stale ``collection_metadata`` reference (renamed to ``memory_metadata``)

Content embeddings are nulled so the startup backfill re-vectorizes the new text.
"""

import sqlite3

_NOTIFY = """[Research collection — notify on new finds] TRIGGER
User wants ongoing research and wants to be told about new finds. Example phrasings:
- "research X for me, ping me when you find stuff"
- "follow X and let me know about new things"
- "build me a list of X, send me updates"
- "keep an eye on X, tell me when there's something new"

STEPS

Single-turn: call the tool, summarize the result, ask the user if it looks right.

0. FIRST check the collections already in your context. If one already covers this
   topic, do NOT create a duplicate — just turn its notifications on with
   collection_update(name, published=true). Only create a new collection when none
   covers the topic.

1. Call collection_create with:
   - name: short slug from the topic
   - description: one-line summary of the SUBJECT MATTER (the routing anchor — the
     collection only surfaces in chats that match this text, so describe the topic,
     not the mechanism)
   - inclusion: "relevant"
   - recall: "relevant"
   - published: true — THIS is what makes new finds reach the user. A separate notifier
     delivers each new entry once; do NOT add a send_message step to the body.
   - collector_interval_seconds: 3600 (default; match the user's cadence words —
     "every 30 min" → 1800, "daily" → 86400)
   - intent: what the user asked for, in their own words
   - extraction_prompt (numbered, name each tool) — the collector ONLY gathers:
     > Collect [topic] — [scope].
     > 1. browse(...) — a few queries targeting [scope]; read actual pages
     > 2. log_read("user-messages") — pick up corrections
     > 3. Each entry: key = item name; content = name + description + URL
     > 4. collection_write("[name]", entries=[...])
     > 5. If a message flags an entry wrong, update_entry or collection_delete_entry
     > 6. done(). If nothing new, just done().

2. The return value echoes what got stored (name, interval, inclusion, recall,
   published, description, full extraction_prompt). Reply with a one-sentence summary
   using ONLY echo values — name the collection, the cadence in human terms, what it
   tracks, and that it'll tell you about new finds (published: true). Ask if they
   want tweaks.

Notification is the published flag, never a send_message step in the collector body.
The collector gathers; the notifier delivers."""

_SILENT = """[Research collection — silent] TRIGGER
User wants ongoing research WITHOUT being told — they'll check in themselves. Phrasings:
- "research X for me, silent, i'll check in"
- "research X but don't ping me"
- "track X quietly"
- "build me a list of X, no notifications"
- "keep tabs on X, i'll ask when i want updates"

STEPS

Single-turn act-then-echo.

1. Call collection_create with:
   - name, description, recall, intent: as in the notify variant
   - inclusion: "never" (silent — never surfaced in chat; the collector still runs)
   - published: false — silent: the collector gathers, nothing notifies you; you read
     the list when you want
   - collector_interval_seconds: match the user's cadence
   - extraction_prompt (numbered) — gather only, no send_message:
     > Collect [topic] — [scope].
     > 1. browse(...) — read actual pages
     > 2. Each entry: key = item name; content = name + description + URL
     > 3. collection_write("[name]", entries=[...])
     > 4. done().

2. Summarize from the echo; note it's silent (published: false) — they'll check the
   list themselves."""

_FLIP = """Flip silent ↔ notify TRIGGER
Use this whenever a collection on the topic ALREADY exists and the user wants to change
whether it tells them about new finds — flip the flag, never create a second collection.
Phrasings:
- "stop pinging me about new X"
- "go silent on X, i'll check in"
- "start pinging me again about X"
- "actually, start telling me when you find new X"
- "turn notifications on/off for X"
- "let me know when there's new stuff in X"

STEPS

Single-turn: update one flag, summarize. Notification is the published flag alone — do
NOT touch the extraction_prompt or inclusion, and do NOT create a new collection.

1. Call collection_update with:
   - name: the collection
   - published: false (to silence) or true (to start telling them)

2. Summarize from the echo. Example: "Silenced `X` — it'll keep gathering, but no more
   pings. Say the word to turn it back on." """

_SCOPE = """TRIGGER
User wants to change WHAT an existing collection collects — add a kind of thing, drop a
topic, swap focus. Example phrasings:
- "add Y to that collection" / "add solo and co-op games to X too"
- "also include Y in X" / "have X cover Y as well"
- "drop Y from X"
- "from now on focus on Z instead"
- "broaden X to also cover Y"

CRITICAL: A scope change is a change to the collection's extraction_prompt BODY (what the
collector gathers going forward) AND its description (the stage-1 routing anchor) — NOT
hand-written entries. Even when the user says "add [things] to the collection", do NOT
collection_write individual entries yourself: broaden the prompt + description so the
collector finds them on its next runs. The collector populates entries; you set its scope.

STEPS

Single-turn: read current state, update, summarize.

1. Call memory_metadata to get the existing extraction_prompt body (the description visible
   in recall isn't the full prompt).

2. Call collection_update with the FULL rewritten extraction_prompt body (replace, not
   diff). Preserve the structural shape (numbered steps, named tools). Apply the scope
   change — widen the browse/targets to include the new kind. Also update description to
   the new subject matter so routing follows the new scope.

3. Summarize from the update echo — name the collection and its new scope (e.g. "now also
   covers solo and co-op games")."""

REPLACEMENTS = {
    "Research collection — notify on new finds": _NOTIFY,
    "Research collection — silent": _SILENT,
    "Flip silent ↔ notify": _FLIP,
    "Update collection scope": _SCOPE,
}


def up(conn: sqlite3.Connection) -> None:
    for key, content in REPLACEMENTS.items():
        conn.execute(
            "UPDATE memory_entry SET content = ?, content_embedding = NULL"
            " WHERE memory_name = 'skills' AND key = ?",
            (content, key),
        )
    conn.commit()
