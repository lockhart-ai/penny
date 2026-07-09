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

The seeded recipe is guideline-compliant: EVERY step is a canonical ``tool(args)`` call, and
notification is pub/sub (the ``published`` flag + the ``notifier`` consumer), NOT a
``send_message`` step.  Calls, in order: browse (search) -> log_read (the removable one) ->
collection_write (save) -> done.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
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
        published=True,  # pub/sub notify is ON — "don't notify me" flips this to False
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
    return [
        # Expected tool call: she must READ the recipe (memory_metadata), not answer from
        # the ambient recall block — which surfaces entries + description but NOT the recipe,
        # so an answer-from-recall describes settings, never the steps (the gap the full
        # report surfaced).  Making the read a scored check is what catches that.
        Check(
            "read the recipe (memory_metadata called)",
            tool_was_called(db, "memory_metadata"),
            anchor="memory_metadata(",
        ),
        Check("reply reflects the search/browse step", search_i >= 0),
        Check("reply reflects the save/write step", save_i >= 0),
    ]


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


def _echoes_designer(reply: str) -> bool:
    return bool(re.search(r"\bdesigner|who\s+(made|designed|created)\b", _norm(reply)))


def _score_edit_and_echo(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[EDIT stored] {stored!r}\n[EDIT reply] {reply.strip()[:240]!r}")
    return [
        Check(
            "applied the edit (collection_update called)",
            tool_was_called(db, "collection_update"),
            anchor="collection_update(",
        ),
        Check("no collection_update rejected", not tool_call_rejected(db, "collection_update")),
        Check("designer landed in the recipe", "designer" in stored, anchor="designer"),
        Check("recipe changed from the seed", stored != BOARD_GAMES_EXTRACTION_PROMPT.lower()),
        Check("no fictitious tool persisted", "extract_text" not in stored),
        Check("reply echoes the change", _echoes_designer(reply)),
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
        min_pass_rate=None,  # baseline (eval-first)
    )


# ════════════════ 2b. Discuss then adjust (multi-turn — the full loop) ═════════
# The core of #1528: the user and Penny DISCUSS a collector's behaviour in plain words,
# then the user ADJUSTS it in plain words — across turns, so the edit rides on the prior
# discussion (Penny sees turn 1 via the DB history).  Turn 1 is legibility (describe the
# recipe); turn 2 is editing (an NL edit that must still land as a real collection_update
# and echo).  Graded per expected tool call: BOTH turns should read the recipe (the discuss
# turn to describe it, the adjust turn before editing) — so a correct run calls
# memory_metadata twice; a sample that answers the discuss turn from ambient recall (no read)
# scores that check as failed instead of hiding behind a green edit.


def _score_discuss_then_adjust(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[DISCUSS stored] {stored!r}\n[DISCUSS reply] {reply.strip()[:240]!r}")
    return [
        # Every turn expects a read: the discuss turn AND the adjust turn — count >= 2 catches
        # a discuss turn answered from recall (the gap the full report surfaced).
        Check(
            "read the recipe each turn (memory_metadata >=2)",
            count_tool_calls(db, "memory_metadata") >= 2,
            anchor="memory_metadata(",
        ),
        Check(
            "applied the edit (collection_update called)",
            tool_was_called(db, "collection_update"),
            anchor="collection_update(",
        ),
        Check("no collection_update rejected", not tool_call_rejected(db, "collection_update")),
        Check("no give-up reply mid-conversation", not gave_up_mid_run(db)),
        Check("designer landed in the recipe", "designer" in stored, anchor="designer"),
        Check("reply echoes the change", _echoes_designer(reply)),
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
        min_pass_rate=None,  # baseline (eval-first) — the multi-turn gap drives #1531
    )


# ═══════ 2c. Edit operations across turns (modify / add / remove a CALL, notify-off, stop) ══
# Every recipe step is a tool call, so every edit operates on a CALL, in one conversation,
# each building on the last edited state: MODIFY a call (collection_write's entry content +=
# designer), ADD a call (a browse for Amazon prices), REMOVE a call (the log_read of the
# user's messages), turn NOTIFICATIONS OFF (flip the pub/sub `published` flag — notify is a
# flag, not a step), then STOP.  Each must land with the spine intact — this is where
# multi-turn state-carrying holds or unravels.


def _score_edit_operations(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[EDIT-OPS stored] {stored!r}  published={row.published if row is not None else None}")
    # REMOVE: the log_read call is gone iff neither its name nor its target survive.
    # NOTIFY-OFF: the pub/sub published flag flips false.
    log_read_gone = "log_read" not in stored and "user-messages" not in stored
    checks = [
        Check(
            "read the recipe (memory_metadata called)",
            tool_was_called(db, "memory_metadata"),
            anchor="memory_metadata(",
        ),
        Check(
            "applied edits (collection_update called)",
            tool_was_called(db, "collection_update"),
            anchor="collection_update(",
        ),
        # Process fidelity: the final-state checks below can pass when an intermediate
        # collection_update was REJECTED and a *later* turn re-landed the content (the
        # rejected-`intent`-param + give-up sample the graded outcome hid).  These two catch
        # the broken turn — the reason we don't merge a scorer that final-state alone fooled.
        Check("no collection_update rejected", not tool_call_rejected(db, "collection_update")),
        Check("no give-up reply mid-conversation", not gave_up_mid_run(db)),
        Check(
            "modify: designer added to collection_write", "designer" in stored, anchor="designer"
        ),
        Check(
            "add: Amazon-price browse call",
            "amazon" in stored or "price" in stored,
            anchor="amazon",
        ),
        Check('remove: log_read("user-messages") gone', log_read_gone),
        Check(
            "notify-off: published set false",
            row is not None and not row.published,
            anchor='"published": false',
        ),
        Check("closer spawned no collection", not new_collections(db, before)),
    ]
    checks += [
        Check(f"spine intact: {family}", family in stored)
        for family in ("browse", "collection_write", "done")
    ]
    return checks


async def test_edit_operations_across_turns(chat_eval: ChatEval) -> None:
    """Deeper multi-turn: MODIFY a call, ADD a call, REMOVE a call, turn notifications OFF
    (the published flag), stop — every edit is on a tool call, each builds on the last."""
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


def _score_roundtrip(db: Database, before: set[str], reply: str) -> list[Check]:
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "").lower() if row is not None else ""
    print(f"\n[ROUNDTRIP stored] {stored!r}")
    browse_i = stored.find("browse")
    write_i = stored.find("collection_write")
    done_i = stored.rfind("done")
    return [
        Check(
            "described the recipe (memory_metadata called)",
            tool_was_called(db, "memory_metadata"),
            anchor="memory_metadata(",
        ),
        # A round-trip happened only if Penny RE-ENCODED via collection_update — else the recipe
        # is the untouched seed and the family checks pass trivially (describe-in-text false pass).
        Check(
            "re-encoded it (collection_update called)",
            tool_was_called(db, "collection_update"),
            anchor="collection_update(",
        ),
        Check("no collection_update rejected", not tool_call_rejected(db, "collection_update")),
        Check("browse step preserved", browse_i >= 0),
        Check("collection_write step preserved", write_i >= 0),
        Check("done step preserved", done_i >= 0),
        Check("family order preserved (browse < write < done)", 0 <= browse_i < write_i < done_i),
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
        min_pass_rate=None,  # baseline (eval-first)
    )
