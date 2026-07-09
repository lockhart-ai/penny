"""Legible-prompts contract (#1530, epic #1528) — reason about a collection's
tool-call recipe in natural language, both directions.

A collection's ``extraction_prompt`` is a tool-call sequence.  These cases assert the
chat model can make it legible and editable in plain language — the substrate the
rest of #1528 (and the #1471 teach-by-example rework) rides on:

  * **Legibility** (prompt -> NL): asked "what does this collection do?", Penny reads
    the recipe (``memory_metadata``) and describes the ORDERED tool families in plain
    words, without inventing a step the recipe doesn't have.
  * **Editing + echo** (NL -> prompt): an NL edit lands as a valid ``collection_update``
    (the persisted recipe changes, only real tools) AND Penny echoes the change back.
  * **No-overreach**: a casual mention (no imperative) must not silently rewrite a recipe.
  * **Discuss then adjust** (multi-turn, the full loop): the user and Penny discuss the
    recipe in plain words, then the user adjusts it in plain words a turn later — the edit
    rides on the prior discussion (Penny sees it via the DB history) and still lands + echoes.
  * **Edit operations** (deeper multi-turn): the three distinct edit KINDS in one
    conversation — TWEAK a step, ADD a step, REMOVE a step, then stop — each building on the
    last edited state, all three landing with the recipe's spine intact.
  * **Round-trip** (true two-turn): Penny describes the recipe, then re-encodes that
    description back into the recipe unchanged — the tool families survive in order.

Granularity is inherited from ``test_narration_survival.py``: scored STRUCTURALLY on the
persisted recipe + which action families the NL reflects, never wording.  This is
**eval-first** (#1530) — the cases are baselined against the current model; the gap
drives the structural work in #1531, so several ship report-only (``min_pass_rate=None``).

The seeded ``BOARD_GAMES_EXTRACTION_PROMPT`` families, in order:
  browse (search + read pages) -> collection_write (save) -> send_message (notify) -> done.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, new_collections, seed_collection, tool_was_called
from penny.tests.eval.fixtures import (
    BOARD_GAMES,
    BOARD_GAMES_EXTRACTION_PROMPT,
    BOARD_GAMES_INTENT,
)

pytestmark = pytest.mark.eval

_COLLECTION = "board-games"


def _seed(db: Database) -> None:
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        intent=BOARD_GAMES_INTENT,
        interval=3600,
    )


def _norm(text: str) -> str:
    """Lowercase, straighten curly quotes, strip markdown emphasis — so a scorer
    matches CONTENT, not typography (the recurring false-negative in these contracts)."""
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[*_`]", "", text)


def _first_index(reply: str, pattern: str) -> int:
    """Earliest index a family's pattern matches in the normalized reply, or -1."""
    match = re.search(pattern, _norm(reply))
    return match.start() if match else -1


# ════════════════════════ 1. Legibility (prompt -> NL) ════════════════════════

# The spine of the recipe — search for games, then write them to the collection —
# is what legibility GATES on.  The search step is described many ways ("browse",
# "look on the web", "pulls in new games from databases and hobby sites", "scans"), so
# match a broad verb set AND fall back to the sources a faithful description names (the
# web / sites / databases / online).  The noun fallback is what makes it robust — a
# reply that describes searching almost always names where it searches.  (Verified
# against captured samples before widening: the earlier verb-only regex false-negatived
# "look on the web" and "pulls in … databases", scoring faithful replies as fails.)
_SEARCH = (
    r"\b(search\w*|browse\w*|scours?|scans?|hunts?|crawls?|monitors?|gathers?|pulls?\s+in|"
    r"look\w*\s+(for|up|on|at|across|through)|finds?\s+new|"
    r"checks?\s+(the\s+web|sites|online)|reads?\s+(pages|articles|sites))\b"
    r"|\b(the\s+web|online|the\s+internet|sites?|databases?)\b"
)
# The save step is phrased many ways ("writes entries", "an entry gets added", "store
# them", "keeps a curated list").  Direct persist verbs match bare; the ambiguous ones
# (add/store/keep/log/maintain/curate/compile) must be ANCHORED to an entry/list object,
# so "adding to your shelf" / "keep an eye on" (about the games, not the write) don't match.
_SAVE = (
    r"\b(saves?|saving|writes?|writing|records?|recording)\b|collection_write"
    r"|\b(adds?|adding|stores?|storing|keeps?|keeping|logs?|logging|maintains?|curates?|compiles?|compiling)\b"
    r"[\w\s,'-]{0,20}\b(entry|entries|list|record|records|collection|them|it)\b"
    r"|\bentr(y|ies)\b[^.]{0,30}\b(added|stored|written|saved|created)\b"
)
# Reported (not gated) — the notify step and any fabrication.  Both are phrasing-fragile:
# notify wording varies wildly and conversational "tell you how it works" false-positives;
# a fabrication check false-positives on negation ("doesn't delete", "never emails you").
# For the eval-first BASELINE they're printed for the #1531 gap map, not part of pass/fail —
# turning them into gates (a faithful-notify contract, a negation-aware honesty guard) is
# exactly the structural work #1531 owns.  (The INTERVAL "every hour" + recall/publish flags
# ARE real memory_metadata fields, so naming them is faithful, never embellishment.)
_NOTIFY = (
    r"\b(tell|message|ping|alert|notif\w*|send|let)\w*\s+you\b[^.]{0,45}"
    r"\b(new|found|finds?|when|game|note|about)\b|send_message"
)
_EMBELLISH = r"\b(deletes?|emails?\s+you|draws?\s+an?\s+image|generates?\s+an?\s+image)\b"


def _score_legibility(db: Database, before: set[str], reply: str) -> list[str]:
    print(f"\n[LEGIBILITY reply] {reply.strip()!r}")
    search_i = _first_index(reply, _SEARCH)
    save_i = _first_index(reply, _SAVE)
    notify = _first_index(reply, _NOTIFY) >= 0
    embellish = re.search(_EMBELLISH, _norm(reply))
    print(
        f"[LEGIBILITY idx] search={search_i} save={save_i} "
        f"notify={notify} embellish={embellish.group(0) if embellish else None}"
    )
    fails: list[str] = []
    if search_i < 0:
        fails.append("reply did not reflect the search/browse step")
    if save_i < 0:
        fails.append("reply did not reflect the save/write step")
    # Order is REPORTED, not gated: the model often leads with the collection's PURPOSE
    # ("keeps a list of games") before enumerating the steps, so a save-verb legitimately
    # precedes the search-verb in the whole reply — that's natural phrasing, not misordered
    # steps.  Gating on step-enumeration order (vs. whole-reply order) is #1531 refinement.
    if search_i >= 0 and save_i >= 0 and search_i > save_i:
        print("[LEGIBILITY note] save mentioned before search (purpose-first phrasing)")
    return fails


async def test_legibility_describes_the_recipe(chat_eval: ChatEval) -> None:
    """prompt -> NL: 'what does this collection do?' -> the ordered families, faithfully."""
    await chat_eval(
        case_id="legible-legibility",
        message="what does the board-games collection actually do? walk me through it.",
        seed=_seed,
        score=_score_legibility,
        min_pass_rate=None,  # baseline (eval-first) — gap drives #1531
    )


# ══════════════════════ 2. Editing + echo (NL -> prompt) ══════════════════════


def _score_edit_and_echo(db: Database, before: set[str], reply: str) -> list[str]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "") if row is not None else ""
    print(f"\n[EDIT stored] {stored!r}\n[EDIT reply] {reply.strip()[:240]!r}")
    fails: list[str] = []
    if "designer" not in stored.lower():
        fails.append(f"edit did not land — 'designer' absent from the recipe: {stored!r}")
    if stored == BOARD_GAMES_EXTRACTION_PROMPT:
        fails.append("recipe unchanged — no collection_update applied")
    # A holds: no fictitious tool slipped in.
    if "extract_text" in stored.lower():
        fails.append("a fictitious tool persisted in the recipe")
    # The echo prong (the #1530-new bit): the reply reflects the change.
    if not re.search(r"\bdesigner|who\s+(made|designed|created)\b", _norm(reply)):
        fails.append(f"reply did not echo the change (designer): {reply[:200]!r}")
    return fails


async def test_editing_lands_and_echoes(chat_eval: ChatEval) -> None:
    """NL -> prompt: an NL edit rewrites the recipe (real tools only) and is echoed back."""
    await chat_eval(
        case_id="legible-editing-echo",
        message=(
            "can you also have the board-games collection record each game's "
            "designer when it saves one?"
        ),
        seed=_seed,
        score=_score_edit_and_echo,
        min_pass_rate=None,  # baseline (eval-first)
    )


# ════════════════ 2b. Discuss then adjust (multi-turn — the full loop) ═════════
# The core of #1528: the user and Penny DISCUSS a collector's behaviour in plain words,
# then the user ADJUSTS it in plain words — across turns, so the edit rides on the prior
# discussion (Penny sees turn 1 via the DB history).  Turn 1 is legibility (describe the
# recipe); turn 2 is editing (an NL edit that must still land as a real collection_update
# and echo).  Scored on the SAME persisted-edit contract as the single-turn editing case —
# the point here is that the multi-turn context doesn't derail the adjustment.


async def test_discuss_then_adjust(chat_eval: ChatEval) -> None:
    """Multi-turn: discuss the recipe, then adjust it in NL — the edit still lands + echoes."""
    await chat_eval(
        case_id="legible-discuss-then-adjust",
        messages=[
            "before I change anything — walk me through what the board-games collection does.",
            "got it. can you also have it record each game's designer when it saves one?",
        ],
        seed=_seed,
        score=_score_edit_and_echo,
        min_pass_rate=None,  # baseline (eval-first) — the multi-turn gap drives #1531
    )


# ═══════ 2c. Edit operations across turns (tweak, add, remove, then stop) ══════
# The three distinct edit KINDS, in one conversation, each building on the last edited
# state (not the seed): TWEAK an existing step (record the designer), ADD a new step (a
# solo-play filter), REMOVE a step (the notify send), then STOP.  All three must show in
# the final recipe with the spine intact — this is where multi-turn state-carrying either
# holds or unravels (a later edit reverting an earlier one; a remove that clobbers the
# spine; the closer over-reacting).  "add designer" alone only exercises a tweak.


def _score_edit_operations(db: Database, before: set[str], reply: str) -> list[str]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[EDIT-OPS stored] {stored!r}")
    fails: list[str] = []
    if "designer" not in stored:  # TWEAK: modified the entry-content step
        fails.append(f"tweak (designer) missing from the recipe: {stored!r}")
    if "solo" not in stored:  # ADD: a new solo-play filter step
        fails.append(f"added step (solo-play filter) missing from the recipe: {stored!r}")
    # REMOVE: the notify step is gone iff its content ("found a new game") and an active
    # send_message( call are both absent.  Key on those, NOT the bare word "send_message" —
    # a prose "no send_message / write silently" note (a correct removal) contains the word
    # but is not an active step (the scorer FN the full report surfaced).
    if "found a new game" in stored or "send_message(" in stored:
        fails.append(f"removed step (notify) still present in the recipe: {stored!r}")
    # The spine must survive all three edits (no edit clobbered it).
    for family in ("browse", "collection_write", "done"):
        if family not in stored:
            fails.append(f"an edit clobbered the '{family}' step: {stored!r}")
    # The closing "thanks" turn must not spawn a collection.
    if created := new_collections(db, before):
        fails.append(f"a turn spuriously created a collection: {[m.name for m in created]}")
    return fails


async def test_edit_operations_across_turns(chat_eval: ChatEval) -> None:
    """Deeper multi-turn: get the recipe, TWEAK a step, ADD a step, REMOVE a step, stop —
    each edit builds on the last edited state and all three land, spine intact."""
    await chat_eval(
        case_id="legible-edit-operations",
        messages=[
            "before I change anything — walk me through what the board-games collection does.",
            "got it. have it record each game's designer too when it saves one.",
            "now add a step so it only keeps games that support solo play.",
            "actually, drop the part where it messages me about each new game — "
            "I don't want the pings.",
            "perfect, that's everything — thanks!",
        ],
        seed=_seed,
        score=_score_edit_operations,
        min_pass_rate=None,  # baseline (eval-first) — the deeper multi-turn gap drives #1531
    )


# ═══════════════════════════ 3. No-overreach guard ════════════════════════════


def _score_no_overreach(db: Database, before: set[str], reply: str) -> list[str]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "") if row is not None else ""
    fails: list[str] = []
    if stored != BOARD_GAMES_EXTRACTION_PROMPT:
        fails.append(f"rewrote the recipe on a casual mention (no imperative): {stored!r}")
    if created := new_collections(db, before):
        fails.append(f"created a collection on a casual mention: {[m.name for m in created]}")
    return fails


async def test_no_overreach_on_casual_mention(chat_eval: ChatEval) -> None:
    """A conversational remark with no imperative must not silently edit the recipe."""
    await chat_eval(
        case_id="legible-no-overreach",
        message="ugh, board games have gotten so pricey lately.",
        seed=_seed,
        score=_score_no_overreach,
        min_pass_rate=0.75,
    )


# ════════════════════ 4. Round-trip (true two-turn, report-only) ══════════════
# prompt -> NL -> prompt across two turns: Penny describes the recipe (turn 1), then
# re-encodes that description back into the recipe unchanged (turn 2).  The persisted tool
# families must survive in the same order — a behaviour-preserving round-trip.  (Was a
# single-turn proxy; now that chat_eval drives conversations it's the real thing.)


def _score_roundtrip(db: Database, before: set[str], reply: str) -> list[str]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[ROUNDTRIP stored] {stored!r}")
    fails: list[str] = []
    # A round-trip only happened if Penny RE-ENCODED the recipe — wrote it back via
    # collection_update.  Without this the recipe is the untouched seed and the family
    # checks below pass trivially (Penny just describing it in a text reply = false pass).
    if not tool_was_called(db, "collection_update"):
        fails.append(
            "no re-encode — recipe never written back via collection_update (described in text?)"
        )
    browse_i = stored.find("browse")
    write_i = stored.find("collection_write")
    done_i = stored.rfind("done")
    for name, index in (("browse", browse_i), ("collection_write", write_i), ("done", done_i)):
        if index < 0:
            fails.append(f"round-trip dropped the '{name}' step")
    if browse_i >= 0 and write_i >= 0 and not browse_i < write_i:
        fails.append("round-trip reordered browse/collection_write")
    if write_i >= 0 and done_i >= 0 and not write_i < done_i:
        fails.append("round-trip reordered collection_write/done")
    return fails


async def test_roundtrip_preserves_the_sequence(chat_eval: ChatEval) -> None:
    """prompt -> NL -> prompt (true two-turn): describe the recipe, then re-encode it
    unchanged — the persisted tool families survive in order."""
    await chat_eval(
        case_id="legible-roundtrip",
        messages=[
            "walk me through what the board-games collection does, step by step.",
            "perfect — now update the board-games recipe itself with a cleaned-up "
            "version that does exactly the same thing.",
        ],
        seed=_seed,
        score=_score_roundtrip,
        min_pass_rate=None,  # baseline (eval-first)
    )
