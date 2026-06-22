"""Reground the skills collector on the real collections instead of chat.

Type: data

The skills collector used to read ``user-messages`` / ``penny-messages`` and
mint a skill from every teaching or correction it saw in chat.  Two failure
modes followed: (1) it reified one-off, collection-specific corrections (a
formatting tweak meant for a single collection) as if they were reusable
skills — noise that never recall-matches anything; and (2) when the user iterated a
collection's ``extraction_prompt`` directly (UI / ``collection_update``), that
refinement never flowed back into the skill describing that kind of collection,
so the skill drifted stale.

This inverts the loop: the skills collector now reads the *real collections*
(via ``collection_catalog``) and distils the reusable, topic-agnostic pattern
behind each one, reconciling against the existing skills — creating a skill for
a new kind of collection, folding a *generalizable* recipe improvement into the
matching skill, and leaving collection-specific quirks in the collection's own
prompt.  Grounding falls out for free: a skill exists because a collection of
that shape exists.  Operate-the-system skills (archive / cadence / flip / scope
/ one-shot) have no source collection, so the loop never finds a reason to touch
them — they stay as seeded.  Notification stays the ``published`` flag (a skill
never teaches a ``send_message`` step in a collector body — see migration 0066).
The loop never deletes a skill (``never-prune``): a pattern with no current
collection still documents how to build one.

Also a one-time cleanup: drop the ``author='skills'`` entries — the chat-derived
one-offs the old loop produced.  The seeded build- and operate-pattern skills
(``author='system'`` / ``'collector'`` / ``'chat'``) are kept; the new loop
refines the build-pattern ones in place as real collections drift.  A fresh DB
has no ``author='skills'`` rows, so this is a no-op there and a scrub in prod.

Idempotent — the UPDATE/DELETE are naturally re-runnable.
"""

from __future__ import annotations

import sqlite3

SKILLS_EXTRACTION_PROMPT = (
    "You maintain the skills list: reusable, TOPIC-AGNOSTIC recipes for KINDS of "
    'collection — e.g. "research a topic and notify on new finds", "research a '
    'topic quietly with no pings", "send a scheduled digest", "watch a page for '
    'changes".  The list already exists and is mostly right.  Each cycle you '
    "reconcile it against the collections that actually exist.  You will RARELY "
    "create a skill (the kinds are already covered) and rarely change one — but "
    "when a collection's recipe has IMPROVED on the skill's method, fold that "
    "improvement back in.\n"
    "\n"
    "1. collection_catalog() — the collections that exist, with their recipes.  "
    'If it returns none, call done(success=true, summary="no collections to '
    'learn from") and stop.\n'
    '2. collection_read_latest("skills", k=30) — read the skills that ALREADY '
    "exist.  This is your menu: you will almost always be matching a collection "
    "to one of these, not inventing a new one.  Each line reads `[key] content`; "
    "the key is the bracketed name, and a skill often repeats that name in "
    "brackets at the START of its content — so to update_entry, pass the key text "
    "ALONE, with NO surrounding brackets (key `Research collection — notify on "
    "new finds`, never `[Research collection — notify on new finds]`).  A "
    "build-a-collection skill embeds the step-by-step extraction_prompt TEMPLATE "
    "it tells chat to write — that template is what you compare against and, when "
    "needed, edit.\n"
    "3. For each collection, name its KIND with the topic stripped out (board "
    'games, indie metroidvanias, espresso gear → "a topic the user tracks"), '
    "and find the skill from step 2 that already describes that KIND.  "
    "Collections of the same kind share ONE skill.\n"
    "4. Compare each collection's recipe against the matched skill (including its "
    "embedded extraction_prompt template) and pick EXACTLY ONE:\n"
    "   a. The skill already covers everything the recipe does → LEAVE IT "
    "UNCHANGED.  Common — the recipe is just the skill applied to a topic.\n"
    "   b. The recipe uses a METHOD the skill's template doesn't mention, and "
    "that method would help ANY collection of this kind → update_entry to fold a "
    "topic-agnostic version of it into the skill's template.  Example: the recipe "
    'says "consult curated lists, then cross-check a reference source before '
    'writing" and the skill only says "browse" — fold the cross-check step into '
    "the template.\n"
    "   c. The recipe's extra step is a quirk of THIS collection only — a tag "
    'prefix like "[MV]", skipping a media type, one topic\'s URL — LEAVE the '
    "skill unchanged; that belongs in the collection's prompt, never the shared "
    "skill.\n"
    "   d. No skill describes this kind at all → collection_write ONE new skill.  "
    "Rare.\n"
    "5. NEVER write a second skill for a kind that already has one — update the "
    "existing one instead.  A skill's key names the KIND (\"research a topic and "
    'notify on new finds"), NEVER a topic ("track indie metroidvanias", '
    '"track newly released games").\n'
    "6. A written or updated skill: key = the kind (5-10 words, topic-free); "
    "content = a TRIGGER section (what the user asks for + example phrasings) and "
    "a STEPS section as a NUMBERED list of tool calls.  Notification is the "
    "`published` flag, never a send_message step.\n"
    "7. NEVER collection_delete_entry a skill — a kind with no current collection "
    "still documents how to build one.\n"
    '8. If you changed nothing, done(success=true, summary="skills already match '
    'the collections").  If you created or updated any skill, send_message ONE '
    'sentence on what changed ("Learned the <X> skill", "Refined the <X> skill — '
    'now <Y>"), then done().'
)


def up(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'skills'",
        (SKILLS_EXTRACTION_PROMPT,),
    )
    # One-time scrub of the chat-derived one-offs the old loop produced.  The
    # seeded build/operate patterns (system/collector/chat authors) stay.
    conn.execute("DELETE FROM memory_entry WHERE memory_name = 'skills' AND author = 'skills'")
    conn.commit()
