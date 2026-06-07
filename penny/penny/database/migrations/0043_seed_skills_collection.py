"""Create the ``skills`` collection and seed it with workflow patterns.

A skill is a recipe for composing tools to satisfy a user intent.  When
the user's message embedding matches a skill's TRIGGER text, the skill
surfaces in the chat agent's recall block and the model follows its
STEPS.  Skills are user-writeable (via ``collection_write``) and the
collection has its own extraction_prompt so new skills get extracted
from chat over time.

**All skills are single-turn act-then-echo.**  An earlier draft tried
propose-confirm-act (propose on turn 1, act on turn 2 after user
confirms) — that pattern is incompatible with embedding-based skill
recall, because the skill that surfaced on turn 1 won't necessarily
surface again when the user's confirmation embeds-matches differently.
Instead each skill completes in one turn: call the tool, summarize the
result back to the user from the structured echo, ask for tweaks.  If
the user wants changes, the appropriate update skill surfaces on the
next turn naturally — no cross-turn state needed.

This migration seeds the initial 8 skills covering the collection
lifecycle.  Embeddings are NULL — skills surface via recall once an
embedding-backfill task populates them (separate concern).  In the
meantime they're still readable via ``read_latest("skills")`` /
``collection_get``.

Idempotent — INSERT OR IGNORE on both the collection row and each entry.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

SKILLS_EXTRACTION_PROMPT = (
    "Extract, refine, or remove skills based on recent user/Penny\n"
    "conversations.  Skills are workflow patterns; this collector grows\n"
    "and tunes them over time as the user gives Penny behavioral guidance.\n"
    "\n"
    '1. log_read_next("user-messages") and log_read_next("penny-messages")\n'
    "   for recent conversation context.  You need both — corrections often\n"
    "   reference something Penny just did.\n"
    "\n"
    "2. Look for any of these patterns:\n"
    "\n"
    "   A. NEW TEACHING — user teaches an additive rule for a new pattern:\n"
    '      - "from now on when I say X, do Y"\n'
    '      - "next time you should..."\n'
    '      - "remember this approach"\n'
    '      - "always do X when I ask Y"\n'
    "\n"
    "   B. CORRECTION — user refines existing behavior:\n"
    '      - "stop doing X" / "don\'t always X"\n'
    '      - "wait, that\'s wrong, do Y instead"\n'
    '      - "actually when I X, only do Y for Z" (scope narrowing)\n'
    '      - "from now on don\'t include X in Y"\n'
    '      - "you don\'t need to X for Y — overkill"\n'
    "\n"
    "   C. DEPRECATION — user wants a rule removed entirely:\n"
    '      - "stop doing X entirely"\n'
    '      - "never do X again"\n'
    '      - "forget about doing X"\n'
    '      - "that was a bad idea, drop it"\n'
    "\n"
    "3. For each candidate (A, B, or C):\n"
    '   - Call read_similar("skills", anchor=<short description of candidate>,\n'
    "     k=3) to see if a similar skill exists.\n"
    "   - If a similar skill exists and the candidate is a CORRECTION:\n"
    "     update_entry on the existing skill to refine it.  DO NOT write a\n"
    "     new skill that contradicts the existing one — that fragments the\n"
    "     skill set and produces conflicting guidance on future recall.\n"
    "   - If a similar skill exists and the candidate is a DEPRECATION:\n"
    "     collection_delete_entry on the existing skill.\n"
    "   - If no similar skill exists and the candidate is a NEW TEACHING:\n"
    "     collection_write a new entry.\n"
    "\n"
    "4. Each written/refined entry:\n"
    "   - key: short skill name (5-10 words)\n"
    "   - content: TRIGGER section (intent + verbatim example phrasings)\n"
    "     + STEPS section (concrete tool composition)\n"
    "\n"
    "5. ONLY if a write/update/delete happened: send_message with one\n"
    "   sentence describing what changed.  Examples:\n"
    '   - "Learned a new skill: X"\n'
    '   - "Refined the X skill — now Y"\n'
    '   - "Removed the X skill"\n'
    "\n"
    "6. done().  If no teachings/corrections/deprecations found, just done().\n"
    "\n"
    "## Worked examples\n"
    "\n"
    "### Correction (subtractive)\n"
    'User said: "wait, when I ask about recipes, stop telling me the\n'
    "difficulty — that's annoying.\"\n"
    "Existing skill from read_similar: [recipes-include-prep-and-difficulty]\n"
    "with TRIGGER on recipe queries and STEPS that say include prep time\n"
    "AND difficulty.\n"
    "→ This is a CORRECTION.  Call update_entry on\n"
    '  "recipes-include-prep-and-difficulty" with the same TRIGGER but\n'
    "  STEPS rewritten to include prep time WITHOUT difficulty.\n"
    '→ Do NOT call collection_write to add a separate "exclude-difficulty"\n'
    "  skill — that would leave two contradictory skills in the collection.\n"
    "\n"
    "### Correction (scope narrowing)\n"
    "User said: \"you don't need to check wikipedia for product comparisons\n"
    "— that's overkill.  only do the wiki link for historical or factual\n"
    'topics."\n'
    "Existing skill: [research-with-wikipedia-link] that says always\n"
    "include Wikipedia.\n"
    "→ This is a CORRECTION (scope narrowing).  update_entry on the existing\n"
    "  skill to scope the Wikipedia step to historical/factual queries only.\n"
    "\n"
    "### Deprecation\n"
    'User said: "actually never check wikipedia anymore, that was a bad idea."\n'
    "Existing skill: [research-with-wikipedia-link]\n"
    "→ This is a DEPRECATION.  Call collection_delete_entry on the existing\n"
    "  skill.\n"
    "\n"
    "### New teaching (no existing skill)\n"
    'User said: "from now on always include vegetarian alternatives when\n'
    'i ask about restaurants."\n'
    "read_similar returns no matching skill.\n"
    "→ This is a NEW TEACHING.  collection_write a new entry with TRIGGER\n"
    "  on restaurant queries and STEPS that include the vegetarian-options\n"
    "  requirement."
)

SEED_SKILLS: list[tuple[str, str]] = [
    (
        "Research collection — notify on new finds",
        """TRIGGER
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
   - description: one-line summary of the focus
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
   interval, recall, description, full extraction_prompt). Reply to the
   user with a one-sentence summary using ONLY values from the echo —
   name the collection, the cadence in human terms, what it's tracking,
   that it pings on new finds. End by asking if they want tweaks.
   Example: "Made `mechanical-keyboards` — checks every hour for new
   keyboards, pings you per find. Want any tweaks?"

If they say yes / no changes → done. If they want changes, the next
turn will surface the appropriate update skill (scope / silent flip /
cadence) which handles the change.
""",
    ),
    (
        "Research collection — silent",
        """TRIGGER
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
   - recall: "off" (silent — no ambient surfacing)
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
    ),
    (
        "Scheduled digest",
        """TRIGGER
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
   - description: one-line summary
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
    ),
    (
        "Browse for a one-shot question",
        """TRIGGER
User asks a one-shot question with no time horizon and no ongoing
framing. Example phrasings:
- "find me a good X"
- "what's the best Y for tonight"
- "is there a Z near me"
- "look up X"

STEPS

1. If the user gave URLs, read them. Otherwise browse() for the topic.

2. Answer the question from the pages you read — concrete details,
   include source URLs.

If the user later decides they want ongoing research on the topic,
they'll say so explicitly ("research X for me") and the research skill
will surface on that next turn.
""",
    ),
    (
        "Update collection scope",
        """TRIGGER
User wants to change WHAT an existing collection collects — add a topic,
drop a topic, swap focus. Example phrasings:
- "add Y to that collection"
- "drop Y from X"
- "from now on focus on Z instead"
- "broaden X to also cover Y"
- "stop tracking Z, just focus on W"

CRITICAL: Scope lives in the extraction_prompt BODY, not the description.
Updating only description leaves the collector running the old focus.

STEPS

Single-turn: read current state, update, summarize.

1. Call collection_metadata to get the existing extraction_prompt body
   (the description visible in recall isn't the full prompt).

2. Call collection_update with the FULL rewritten extraction_prompt
   body (replace, not diff). Preserve the structural shape (numbered
   steps, named tools). Apply the user's scope change — add the new
   topic, drop the old one, etc. Also update description to match.

3. Summarize back from the update echo — name the collection and the
   new scope.
   Example: "Updated `X` — now collects [new scope]. Want anything
   else changed?"
""",
    ),
    (
        "Flip silent ↔ notify",
        """TRIGGER
User wants to change whether an existing collection pings them on
finds. Example phrasings:
- "stop pinging me about new X"
- "go silent on X, i'll check in"
- "start pinging me again about X"
- "let me know when there's new stuff in X"

CRITICAL: Silent mode requires BOTH recall=off AND removing the
send_message step from the extraction_prompt body. recall controls
ambient surfacing; send_message controls active pings. Leaving the
body alone means the collector keeps paging you every cycle even with
recall=off.

STEPS

Single-turn: read, update, summarize.

1. Call collection_metadata for the existing body.

2. Call collection_update with BOTH:
   - recall: "off" (silent) or "relevant" (notify)
   - extraction_prompt: rewritten body without (or with) the
     send_message step. Keep all other steps intact.

3. Summarize from the echo.
   Example: "Silenced `X` — no more pings on new finds. Ask any time."
""",
    ),
    (
        "Change collection cadence",
        """TRIGGER
User wants to change how often an existing collection runs. Example
phrasings:
- "check X every 30 minutes instead"
- "make X daily"
- "speed up X to twice an hour"
- "slow X down to weekly"

STEPS

Single-turn.

1. Call collection_update:
   - name: the collection name
   - collector_interval_seconds: from the requested cadence
     ("every 30 min" → 1800, "hourly" → 3600, "daily" → 86400,
     "weekly" → 604800)

2. Summarize from the echo.
   Example: "Updated `X` — now runs every 30 minutes."
""",
    ),
    (
        "Archive a collection",
        """TRIGGER
User wants to close out a collection — done with the topic. Example
phrasings:
- "archive the X collection"
- "we're done with X, close it"
- "drop X"
- "trip got cancelled, close X"
- "stop collecting X"

STEPS

Single-turn.

1. Call collection_archive(memory="[name]").

2. Confirm: "Archived `X` — say to unarchive if you change your mind."
""",
    ),
]


def up(conn: sqlite3.Connection) -> None:
    now = datetime.now(UTC).isoformat()

    conn.execute(
        "INSERT OR IGNORE INTO memory "
        "(name, type, description, recall, archived, created_at, "
        "extraction_prompt, collector_interval_seconds) "
        "VALUES (?, 'collection', ?, 'relevant', 0, ?, ?, ?)",
        (
            "skills",
            "Workflow patterns — how to compose tools to satisfy user intents",
            now,
            SKILLS_EXTRACTION_PROMPT,
            21600,  # 6h — new skills don't form rapidly
        ),
    )

    for key, content in SEED_SKILLS:
        conn.execute(
            "INSERT OR IGNORE INTO memory_entry "
            "(memory_name, key, content, author, "
            "key_embedding, content_embedding, created_at) "
            "VALUES ('skills', ?, ?, 'system', NULL, NULL, ?)",
            (key, content, now),
        )

    conn.commit()
