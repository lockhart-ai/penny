"""Legible-prompts contract (#1530, epic #1528) — reason about a collection's
tool-call recipe in natural language, both directions.

A collection's ``extraction_prompt`` is a tool-call sequence.  These cases assert the
chat model can make it legible and editable in plain language — the substrate the
rest of #1528 (and the #1471 teach-by-example rework) rides on:

  * **Legibility** (prompt -> NL): asked "what does this collection do?", Penny reads
    the recipe (``memory_metadata``) and describes the ORDERED tool families in plain
    words, without inventing a step the recipe doesn't have.
  * **Editing + echo** (NL -> prompt): an NL edit lands as a valid ``collection_set``
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
**eval-first** (#1530) — the cases were baselined against the current model; Ticket C
(#1531) iterated the levers until they held, and the cases now **gate** at thresholds set
with margin below their N=5 baselines (legibility/edit/round-trip 0.80, discuss 0.75).

The seeded recipe is guideline-compliant: EVERY step is a canonical ``tool(args)`` call, and
notification is the ``notify`` flag (the run-time notify suffix, #1557), NOT a
``send_message`` step.  Calls, in order: browse (search) -> log_read (the removable one) ->
collection_write (save) -> done.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    count_tool_calls,
    gave_up_mid_run,
    new_collections,
    seed_collection,
    tool_call_rejected,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    BOARD_GAMES,
    BOARD_GAMES_EXTRACTION_PROMPT,
)

pytestmark = pytest.mark.eval

_COLLECTION = "board-games"


def _seed(db: Database) -> None:
    seed_collection(
        db,
        BOARD_GAMES,
        extraction_prompt=BOARD_GAMES_EXTRACTION_PROMPT,
        interval=3600,
        notify=True,  # notify is ON — "don't notify me" flips this to False
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


def _score_legibility(db: Database, before: set[str], reply: str) -> list[Check]:
    print(f"\n[LEGIBILITY reply] {reply.strip()!r}")
    search_i = _first_index(reply, _SEARCH)
    save_i = _first_index(reply, _SAVE)
    read_recipe = tool_was_called(db, "memory_metadata")
    return [
        # Expected tool call: she must READ the recipe (memory_metadata), not answer from
        # the ambient recall block — which surfaces entries + description but NOT the recipe,
        # so an answer-from-recall describes settings, never the steps (the gap the full
        # report surfaced).  Making the read a scored check is what catches that.
        Check(
            "read the recipe (memory_metadata called)",
            read_recipe,
            anchor="memory_metadata(",
            rationale=None if read_recipe else "memory_metadata not called — answered from recall",
        ),
        Check(
            "reply reflects the search/browse step",
            search_i >= 0,
            anchor=REPLY_ANCHOR,
            rationale=None if search_i >= 0 else "no search/browse family described in the reply",
        ),
        Check(
            "reply reflects the save/write step",
            save_i >= 0,
            anchor=REPLY_ANCHOR,
            rationale=None if save_i >= 0 else "no save/write family described in the reply",
        ),
    ]


async def test_legibility_describes_the_recipe(chat_eval: ChatEval) -> None:
    """prompt -> NL: 'what does this collection do?' -> the ordered families, faithfully."""
    await chat_eval(
        case_id="legible-legibility",
        message="what does the board-games collection actually do? walk me through it.",
        seed=_seed,
        score=_score_legibility,
        min_pass_rate=0.80,  # N=5 baseline 1.00 — Ticket C threshold (#1531)
        family="legibility",
    )


# ══════════════════════ 2. Editing + echo (NL -> prompt) ══════════════════════


def _echoes_designer(reply: str) -> bool:
    return bool(re.search(r"\bdesigner|who\s+(made|designed|created)\b", _norm(reply)))


def _score_edit_and_echo(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[EDIT stored] {stored!r}\n[EDIT reply] {reply.strip()[:240]!r}")
    applied = tool_was_called(db, "collection_set")
    rejected = tool_call_rejected(db, "collection_set")
    designer = "designer" in stored
    changed = stored != BOARD_GAMES_EXTRACTION_PROMPT.lower()
    echoed = _echoes_designer(reply)
    return [
        Check(
            "applied the edit (collection_set called)",
            applied,
            anchor="collection_set(",
            rationale=None if applied else "collection_set never called — nothing persisted",
        ),
        # Process fidelity (fragile-pass): a rejected-then-relanded call reads as a wobble.
        Check(
            "no collection_set rejected",
            not rejected,
            rationale="a collection_set call was rejected mid-run" if rejected else None,
        ),
        Check(
            "designer landed in the recipe",
            designer,
            anchor="designer",
            rationale=None if designer else "'designer' absent from the stored recipe",
        ),
        Check(
            "recipe changed from the seed",
            changed,
            rationale=None if changed else "recipe still byte-identical to the seed",
        ),
        Check("no fictitious tool persisted", "extract_text" not in stored),
        Check(
            "reply echoes the change",
            echoed,
            rationale=None if echoed else "reply doesn't mention the designer change",
        ),
    ]


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
        min_pass_rate=0.80,  # N=5 baseline 0.97 — Ticket C threshold (#1531)
        family="editing",
    )


# ════════════════ 2b. Discuss then adjust (multi-turn — the full loop) ═════════
# The core of #1528: the user and Penny DISCUSS a collector's behaviour in plain words,
# then the user ADJUSTS it in plain words — across turns, so the edit rides on the prior
# discussion (Penny sees turn 1 via the DB history).  Turn 1 is legibility (describe the
# recipe); turn 2 is editing (an NL edit that must still land as a real collection_set
# and echo).  Graded per expected tool call: BOTH turns should read the recipe (the discuss
# turn to describe it, the adjust turn before editing) — so a correct run calls
# memory_metadata twice; a sample that answers the discuss turn from ambient recall (no read)
# scores that check as failed instead of hiding behind a green edit.


def _score_discuss_then_adjust(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[DISCUSS stored] {stored!r}\n[DISCUSS reply] {reply.strip()[:240]!r}")
    reads = count_tool_calls(db, "memory_metadata")
    applied = tool_was_called(db, "collection_set")
    rejected = tool_call_rejected(db, "collection_set")
    gave_up = gave_up_mid_run(db)
    designer = "designer" in stored
    echoed = _echoes_designer(reply)
    return [
        # Every turn expects a read: the discuss turn AND the adjust turn — count >= 2 catches
        # a discuss turn answered from recall (the gap the full report surfaced).
        Check(
            "read the recipe each turn (memory_metadata >=2)",
            reads >= 2,
            anchor="memory_metadata(",
            rationale=f"expected >=2 reads, saw {reads}",
        ),
        Check(
            "applied the edit (collection_set called)",
            applied,
            anchor="collection_set(",
            rationale=None if applied else "collection_set never called — nothing persisted",
        ),
        # Process fidelity (fragile-pass): a rejected call or a give-up reply reads as a wobble.
        Check(
            "no collection_set rejected",
            not rejected,
            rationale="a collection_set call was rejected mid-run" if rejected else None,
        ),
        Check(
            "no give-up reply mid-conversation",
            not gave_up,
            rationale="Penny gave up mid-conversation" if gave_up else None,
        ),
        Check(
            "designer landed in the recipe",
            designer,
            anchor="designer",
            rationale=None if designer else "'designer' absent from the stored recipe",
        ),
        Check(
            "reply echoes the change",
            echoed,
            rationale=None if echoed else "reply doesn't mention the designer change",
        ),
    ]


async def test_discuss_then_adjust(chat_eval: ChatEval) -> None:
    """Multi-turn: discuss the recipe, then adjust it in NL — the edit still lands + echoes."""
    await chat_eval(
        case_id="legible-discuss-then-adjust",
        messages=[
            "before I change anything — walk me through what the board-games collection does.",
            "got it. can you also have it record each game's designer when it saves one?",
        ],
        seed=_seed,
        score=_score_discuss_then_adjust,
        min_pass_rate=0.75,  # N=5 baseline 0.90 — Ticket C threshold (#1531)
        family="editing",
    )


# ═══════ 2c. Edit operations across turns (modify / add / remove a CALL, notify-off, stop) ══
# Every recipe step is a tool call, so every edit operates on a CALL, in one conversation,
# each building on the last edited state: MODIFY a call (collection_write's entry content +=
# designer), ADD a call (a browse for Amazon prices), REMOVE a call (the log_read of the
# user's messages), turn NOTIFICATIONS OFF (flip the `notify` flag — notify is a
# flag, not a step), then STOP.  Each must land with the spine intact — this is where
# multi-turn state-carrying holds or unravels.


def _score_edit_operations(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[EDIT-OPS stored] {stored!r}  notify={row.notify if row is not None else None}")
    # REMOVE: the log_read call is gone iff neither its name nor its target survive.
    # NOTIFY-OFF: the notify flag flips false.
    log_read_gone = "log_read" not in stored and "user-messages" not in stored
    read_recipe = tool_was_called(db, "memory_metadata")
    applied = tool_was_called(db, "collection_set")
    rejected = tool_call_rejected(db, "collection_set")
    gave_up = gave_up_mid_run(db)
    designer = "designer" in stored
    priced = "amazon" in stored or "price" in stored
    notify_off = row is not None and not row.notify
    spawned = new_collections(db, before)
    checks = [
        Check(
            "read the recipe (memory_metadata called)",
            read_recipe,
            anchor="memory_metadata(",
            rationale=None if read_recipe else "memory_metadata not called",
        ),
        Check(
            "applied edits (collection_set called)",
            applied,
            anchor="collection_set(",
            rationale=None if applied else "collection_set never called — nothing persisted",
        ),
        # Process fidelity (fragile-pass): the final-state checks below can pass when an
        # intermediate collection_set was REJECTED and a *later* turn re-landed the content
        # (the rejected-`intent`-param + give-up sample the graded outcome hid).  These two
        # catch the broken turn — the reason we don't merge a scorer final-state alone fooled.
        Check(
            "no collection_set rejected",
            not rejected,
            rationale="a collection_set call was rejected mid-run" if rejected else None,
        ),
        Check(
            "no give-up reply mid-conversation",
            not gave_up,
            rationale="Penny gave up mid-conversation" if gave_up else None,
        ),
        Check(
            "modify: designer added to collection_write",
            designer,
            anchor="designer",
            rationale=None if designer else "'designer' absent from the recipe",
        ),
        Check(
            "add: Amazon-price browse call",
            priced,
            anchor="amazon",
            rationale=None if priced else "neither 'amazon' nor 'price' in the recipe",
        ),
        Check(
            'remove: log_read("user-messages") gone',
            log_read_gone,
            rationale=None if log_read_gone else "log_read/user-messages still in the recipe",
        ),
        Check(
            "notify-off: notify set false",
            notify_off,
            anchor='"notify": false',
            rationale=None if notify_off else "notify still on",
        ),
        Check(
            "closer spawned no collection",
            not spawned,
            rationale=None if not spawned else f"spawned {[m.name for m in spawned]}",
        ),
    ]
    checks += [
        Check(
            f"spine intact: {family}",
            family in stored,
            rationale=None if family in stored else f"'{family}' missing from the recipe",
        )
        for family in ("browse", "collection_write", "done")
    ]
    return checks


async def test_edit_operations_across_turns(chat_eval: ChatEval) -> None:
    """Deeper multi-turn: MODIFY a call, ADD a call, REMOVE a call, turn notifications OFF
    (the notify flag), stop — every edit is on a tool call, each builds on the last."""
    await chat_eval(
        case_id="legible-edit-operations",
        messages=[
            "before I change anything — walk me through what the board-games collection does.",
            "got it. have it record each game's designer too when it saves one.",
            "also add a step to look up each game's current price on Amazon.",
            "actually, drop the step where it reads my messages — I don't need that.",
            "and stop notifying me about new finds — no more pings.",
            "perfect, that's everything — thanks!",
        ],
        seed=_seed,
        score=_score_edit_operations,
        min_pass_rate=0.80,  # N=5 baseline 0.95 — Ticket C threshold (#1531)
        family="editing",
    )


# ════════════════════ 3. No silent recipe edit on a casual mention ════════════════════
# NB: this guards ONLY against an unprompted STRUCTURAL change — silently rewriting the recipe
# or spawning a collection when the user just made a passing remark.  It deliberately does NOT
# police browsing or tool-call count: Penny browses in normal chat, and a matched-but-dummy
# browse fixture can make a passing remark look like a "tool maze" that isn't a real regression.


def _score_no_silent_edit(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "") if row is not None else ""
    unchanged = stored == BOARD_GAMES_EXTRACTION_PROMPT
    created = new_collections(db, before)
    return [
        Check(
            "recipe untouched on a casual mention (no imperative)",
            unchanged,
            rationale=None if unchanged else f"rewrote the recipe: {stored!r}",
        ),
        Check(
            "no collection spawned on a casual mention",
            not created,
            rationale=None if not created else f"created {[m.name for m in created]}",
        ),
    ]


async def test_no_silent_recipe_edit_on_casual_mention(chat_eval: ChatEval) -> None:
    """A conversational remark with no imperative must not silently edit the recipe or spawn a
    collection.  (Browsing / a chatty reply is fine — this only guards the structural change.)"""
    await chat_eval(
        case_id="legible-no-silent-recipe-edit",
        message="ugh, board games have gotten so pricey lately.",
        seed=_seed,
        score=_score_no_silent_edit,
        min_pass_rate=0.75,
        family="no-overreach",
    )


# ════════════════════ 4. Round-trip (true two-turn, report-only) ══════════════
# prompt -> NL -> prompt across two turns: Penny describes the recipe (turn 1), then
# re-encodes that description back into the recipe unchanged (turn 2).  The persisted tool
# families must survive in the same order — a behaviour-preserving round-trip.  (Was a
# single-turn proxy; now that chat_eval drives conversations it's the real thing.)


def _score_roundtrip(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[ROUNDTRIP stored] {stored!r}")
    browse_i = stored.find("browse")
    write_i = stored.find("collection_write")
    done_i = stored.rfind("done")
    described = tool_was_called(db, "memory_metadata")
    reencoded = tool_was_called(db, "collection_set")
    rejected = tool_call_rejected(db, "collection_set")
    order_ok = 0 <= browse_i < write_i < done_i
    return [
        Check(
            "described the recipe (memory_metadata called)",
            described,
            anchor="memory_metadata(",
            rationale=None if described else "memory_metadata not called",
        ),
        # A round-trip happened only if Penny RE-ENCODED via collection_set — else the recipe
        # is the untouched seed and the family checks pass trivially (describe-in-text false pass).
        Check(
            "re-encoded it (collection_set called)",
            reencoded,
            anchor="collection_set(",
            rationale=None if reencoded else "collection_set never called — no re-encode",
        ),
        # Process fidelity (fragile-pass): a rejected re-encode reads as a wobble.
        Check(
            "no collection_set rejected",
            not rejected,
            rationale="a collection_set call was rejected mid-run" if rejected else None,
        ),
        Check(
            "browse step preserved",
            browse_i >= 0,
            rationale=None if browse_i >= 0 else "browse missing from the re-encoded recipe",
        ),
        Check(
            "collection_write step preserved",
            write_i >= 0,
            rationale=None
            if write_i >= 0
            else "collection_write missing from the re-encoded recipe",
        ),
        Check(
            "done step preserved",
            done_i >= 0,
            rationale=None if done_i >= 0 else "done missing from the re-encoded recipe",
        ),
        Check(
            "family order preserved (browse < write < done)",
            order_ok,
            rationale=None
            if order_ok
            else f"order broken — browse@{browse_i}, write@{write_i}, done@{done_i}",
        ),
    ]


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
        min_pass_rate=0.80,  # N=5 baseline 0.97 — Ticket C threshold (#1531)
        family="round-trip",
    )
