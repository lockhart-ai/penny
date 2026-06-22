"""Reground the skills collector on real collections + clean the skill set.

Type: data

The skills collector used to read ``user-messages`` / ``penny-messages`` and
mint a skill from every teaching or correction it saw in chat.  Two failure
modes followed: (1) it reified one-off, collection-specific corrections (a
formatting tweak meant for a single collection) as if they were reusable
skills — noise that never recall-matched anything; and (2) when the user
iterated a collection's ``extraction_prompt`` directly, that refinement never
flowed back into the skill describing that kind of collection.

This inverts the loop: the collector now reads the *real collections* (via
``collection_catalog``) and reconciles the skill set against them — distilling
the topic-agnostic pattern behind each collection, folding a generalizable
recipe improvement into the matching skill, and leaving collection-specific
quirks in the collection's own prompt.  A skill exists because a collection of
that shape exists, so skills stay grounded and prompt edits propagate.

It also rewrites every surviving seeded skill into the one clean shape a skill
should have: a TRIGGER (when the user signals an intent) and numbered STEPS (the
shape of tool calls that implements it) — positive guidance only.  The old
seeds carried legacy negatives ("do NOT add a send_message step") rooted in the
pre-pub/sub structure we've removed; they're superfluous (notification is just
``published: true``) and the model has no memory of the old structure to need
warning against.  The redundant ``[key]`` self-title some seeds repeated at the
start of their content is dropped too (it confused ``update_entry``).

Everything here touches only data present in EVERY deployment — the skills
collection row and the *seeded* skills (referenced by their known keys), plus
generic criteria (``author='skills'``).  It never targets a deployment-specific,
chat-created entry: a user's own chat-authored skill (e.g. a "watch this page"
skill predating pub/sub) is runtime data the reconcile loop fixes against its
live collector — not the migration's job (a migration must not assume one
deployment's rows).  A fresh DB has no ``author='skills'`` rows, so that scrub
is a no-op there.

Idempotent — the UPDATE/DELETE statements are naturally re-runnable.
"""

from __future__ import annotations

import sqlite3

SKILLS_EXTRACTION_PROMPT = (
    "You maintain the skills list.  A skill is a clean, positive recipe: WHEN the "
    "user signals an intent, HERE is the shape of tool calls that implements it — "
    "a TRIGGER (the intent + example phrasings) and numbered STEPS (the tool "
    "calls).  Skills are topic-agnostic and describe KINDS of collection — e.g. "
    '"research a topic and notify on new finds", "research a topic quietly", '
    '"watch a page for changes".  The list already exists and is mostly right; '
    "each cycle you reconcile it against the collections that actually exist.  You "
    "will rarely create or change a skill — do the least that's true.\n"
    "\n"
    "1. collection_catalog() — the collections that exist, with their recipes.  If "
    'it returns none, call done(success=true, summary="no collections to learn '
    'from") and stop.\n'
    '2. collection_read_latest("skills", k=30) — the skills that already exist '
    "(your menu).  Pass a skill's key exactly as listed when you update one.\n"
    "3. For each collection, name its KIND with the topic stripped out (board "
    'games, indie metroidvanias, espresso gear → "a topic the user tracks") and '
    "find the skill that already describes that KIND.  Collections of the same "
    "kind share ONE skill.\n"
    "4. For each collection pick EXACTLY ONE:\n"
    "   a. A skill already covers the kind → LEAVE IT.  Common — the recipe is "
    "just the skill applied to a topic.\n"
    "   b. The recipe uses a method the skill doesn't mention that would help ANY "
    "collection of this kind → update_entry to fold a topic-agnostic version of "
    'it into the skill.  Example: the recipe "consults curated lists, then '
    'cross-checks a reference source before writing" while the skill only says '
    '"browse" — add the cross-check step.\n'
    "   c. The recipe's extra step only makes sense for this one collection (a tag "
    "prefix, a skipped media type, one topic's URL) → leave the skill as is; that "
    "belongs in the collection's own prompt.\n"
    "   d. No skill describes this kind → collection_write ONE new skill.  Rare.\n"
    "5. One skill per kind — update the existing one rather than adding a second.  "
    'A skill\'s key names the KIND ("research a topic and notify on new finds"), '
    'not a topic ("track indie metroidvanias").\n'
    "6. Every skill stays a clean positive recipe: a TRIGGER (intent + example "
    "phrasings) and numbered STEPS of tool calls.  If the kind notifies the user, "
    "its create step sets published: true (a separate notifier delivers new "
    "entries).\n"
    "7. Keep every skill — a kind with no current collection still documents how "
    "to build one.\n"
    '8. If nothing changed, done(success=true, summary="skills already match the '
    'collections").  If you created or updated a skill, send_message one sentence '
    "on what changed, then done()."
)


# The canonical seeded skills, rewritten into the one clean shape — TRIGGER +
# numbered tool-call STEPS, positive only.  Keys are unchanged (kept identical so
# nothing referencing them needs to change); only the bodies are replaced.
_RESEARCH_NOTIFY = """TRIGGER
User wants ongoing research on a topic and wants to hear about new finds. Example phrasings:
- "research X for me, ping me when you find stuff"
- "follow X and let me know about new things"
- "keep an eye on X, tell me when there's something new"

STEPS
1. collection_create with:
   - name: a slug from the topic
   - description: the subject matter in one line (this is the routing anchor)
   - inclusion: "relevant", recall: "relevant"
   - published: true (a notifier delivers each new entry to the user)
   - collector_interval_seconds: the user's cadence (default 3600)
   - intent: what the user asked for, in their own words
   - extraction_prompt (numbered, name each tool):
       Collect [topic] — [scope].
       1. browse(...) — queries targeting [scope]; read actual pages.
       2. Each entry: key = item name; content = name + description + URL.
       3. collection_write("[name]", entries=[...]).
       4. done().
2. Summarize from the returned echo — the collection name, the cadence in plain words, what
   it tracks, and that it'll tell you about new finds. Ask if they want tweaks."""

_RESEARCH_SILENT = """TRIGGER
User wants ongoing research on a topic but no pings — they'll check the list
themselves. Example phrasings:
- "track X quietly"
- "research X, no notifications"
- "keep tabs on X, i'll ask when i want updates"

STEPS
1. collection_create with:
   - name, description, intent: as for a research collection
   - inclusion: "relevant", recall: "relevant"
   - published: false (the collector gathers; nothing pings the user)
   - collector_interval_seconds: the user's cadence
   - extraction_prompt (numbered):
       Collect [topic] — [scope].
       1. browse(...) — read actual pages.
       2. Each entry: key = item name; content = name + description + URL.
       3. collection_write("[name]", entries=[...]).
       4. done().
2. Summarize from the echo; mention it's silent and they can ask to see the list anytime."""

_UPDATE_SCOPE = """TRIGGER
User wants to change WHAT an existing collection gathers — add a topic, drop one,
refocus. Example phrasings:
- "also track Y in X"
- "narrow X to just Z"
- "broaden X to cover Y too"

STEPS
1. memory_metadata("[X]") — read the collection's current extraction_prompt and description.
2. collection_update("[X]") with:
   - extraction_prompt: the full rewritten recipe carrying the new scope
   - description: updated to the new subject matter (the routing anchor follows the scope)
3. Summarize from the echo — name the collection and its new scope."""

_FLIP = """TRIGGER
User wants to change whether a collection notifies them. Example phrasings:
- "start telling me about new X"
- "stop pinging me about X, i'll just look myself"

STEPS
1. collection_update("[X]", published=true) to start notifying, or published=false to go silent.
2. Confirm the change in one sentence."""

_CADENCE = """TRIGGER
User wants to change how often a collection runs. Example phrasings:
- "check X daily instead"
- "speed X up to twice an hour"
- "slow X down to weekly"

STEPS
1. collection_update("[X]", collector_interval_seconds=N) — N from the requested cadence
   ("daily" = 86400, "hourly" = 3600, "every 30 min" = 1800, "weekly" = 604800).
2. Confirm the new cadence in plain words."""

_ARCHIVE = """TRIGGER
User is done with a collection. Example phrasings:
- "stop collecting X"
- "archive X"
- "i'm done with X, close it"

STEPS
1. collection_archive("[X]").
2. Confirm it's archived and can be reopened later."""

_ONE_SHOT = """TRIGGER
User asks a one-shot question with no ongoing tracking. Example phrasings:
- "what's a good X"
- "find me Y"
- "look up Z"

STEPS
1. browse(...) — search and read the pages that answer it.
2. Answer with concrete details and include the source URLs."""

SKILL_REWRITES = {
    "Research collection — notify on new finds": _RESEARCH_NOTIFY,
    "Research collection — silent": _RESEARCH_SILENT,
    "Update collection scope": _UPDATE_SCOPE,
    "Flip silent ↔ notify": _FLIP,
    "Change collection cadence": _CADENCE,
    "Archive a collection": _ARCHIVE,
    "Browse for a one-shot question": _ONE_SHOT,
}


def up(conn: sqlite3.Connection) -> None:
    # 1. The new catalog-driven reconcile loop (replaces the chat-reading prompt).
    conn.execute(
        "UPDATE memory SET extraction_prompt = ? WHERE name = 'skills'",
        (SKILLS_EXTRACTION_PROMPT,),
    )
    # 2. Scrub the chat-derived one-offs the old loop produced (generic), plus the
    #    seeded 'Scheduled digest' skill: no live collector embodies it and the
    #    digest-under-pub/sub shape is unresolved.
    conn.execute("DELETE FROM memory_entry WHERE memory_name = 'skills' AND author = 'skills'")
    conn.execute(
        "DELETE FROM memory_entry WHERE memory_name = 'skills' AND key = 'Scheduled digest'"
    )
    # 3. Rewrite the surviving seeded skills into the one clean shape — TRIGGER +
    #    numbered tool-call STEPS, positive only.  Re-embed (content_embedding NULL
    #    → refilled by the startup backfill).
    for key, content in SKILL_REWRITES.items():
        conn.execute(
            "UPDATE memory_entry SET content = ?, content_embedding = NULL "
            "WHERE memory_name = 'skills' AND key = ?",
            (content, key),
        )
    conn.commit()
