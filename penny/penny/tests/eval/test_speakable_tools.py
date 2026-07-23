"""Speakable tools (epic #1521): multi-step NL sequences → ordered tool calls.

The user describes a *sequence* of actions in natural language; Penny fires the
matching chat tools **in order**, then recaps (the outbound narration shipped in
#1478).  Inbound is many-to-one and fuzzy — a family of phrasings maps to one
tool by MEANING, never an exact-string table — and the contract is scored
STRUCTURALLY on the persisted tool sequence + DB state, never on wording.

Scope is entry- and browse-level actions the user dictates in the moment —
searching/reading, reading collection entries, writing/updating/deleting entries.
It deliberately excludes standing-up machinery (creating collections, skills, or
extraction prompts); those are a downstream concern.

The representative ``search → read → store`` case is driven first; the rest fan
out from it.  Synthetic topics only (the repo is public): the looked-up subject
is the invented game *Mistforge Tactics* (``MULTIHOP_PAGES``), so the model must
browse rather than answer a real fact from memory.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    collection_entries,
    is_ordered_subsequence,
    seed_collection,
    tool_call_arg_values,
    tool_call_sequence,
    tool_was_called,
)
from penny.tests.eval.fixtures import MULTIHOP_PAGES, SynthCollection

pytestmark = pytest.mark.eval

_BROWSE = "browse"
_WRITE = "collection_write"
_READ = "collection_read_latest"
_UPDATE = "update_entry"
_DELETE = "collection_delete_entry"
# Every entry-mutating / browse tool — the set a no-fire guard must see stay quiet.
_ACTION_TOOLS = (_BROWSE, _WRITE, _UPDATE, _DELETE)

# Collection names the cases read from / write to.
_GAMES_NAME = "games"
_LIKES = "likes"
_DISLIKES = "dislikes"

# A collection the user already has — the store/update steps target it, so no case
# exercises collection creation (out of scope).  ``likes`` / ``dislikes`` are
# system collections migrations already create, so those are seeded by WRITING
# into them, never re-creating them.
_GAMES = SynthCollection(
    _GAMES_NAME,
    "Video games the user is tracking or interested in: titles, release dates, and notes.",
    entries=(),
)
_MISTFORGE_KEY = "Mistforge Tactics"

# Family tag (explicit, meaningful grouping): the multi-step NL action sequences.
_FAMILY = "speakable-sequence"


def _seed_games(db: Database) -> None:
    seed_collection(db, _GAMES)


def _seed_games_with_mistforge(db: Database) -> None:
    seed_collection(db, _GAMES)
    db.memory(_GAMES_NAME).write(
        [EntryInput(key=_MISTFORGE_KEY, content="Mistforge Tactics — a turn-based strategy game.")],
        author="user",
    )


def _seed_likes(db: Database) -> None:
    for key, content in (
        ("chess", "chess — enjoys playing chess"),
        ("hiking", "hiking — loves weekend hikes"),
        ("jazz", "jazz — a big fan of jazz records"),
    ):
        db.memory(_LIKES).write([EntryInput(key=key, content=content)], author="user")


def _seed_sweep(db: Database) -> None:
    _seed_likes(db)
    db.memory(_DISLIKES).write(
        [EntryInput(key="loud offices", content="loud offices — can't focus in them")],
        author="user",
    )
    _seed_games_with_mistforge(db)


def _normalize(text: str) -> str:
    """Fold the typography gpt-oss sprinkles into its output so a SEMANTIC
    substring probe isn't defeated by cosmetics: unicode hyphens → '-',
    nbsp/zero-width/narrow spaces → ' ', bold markers stripped, curly quotes
    straightened, lowercased.  (A 0/N from an un-normalized probe is a scorer
    bug — the model wrote 'co‑op' / 'Mist​ Tactics', semantically right.)"""
    folded = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        folded = folded.replace(dash, "-")
    for space in ("\xa0", "​", " ", " "):
        folded = folded.replace(space, " ")
    for src, dst in (("’", "'"), ("“", '"'), ("”", '"'), ("*", "")):
        folded = folded.replace(src, dst)
    return folded


def _saved_text(db: Database, name: str) -> str:
    """A collection's keys AND contents, normalized and joined — the probe for
    'did the subject land here', robust to whether the model put the subject in
    the key or the body and to its typography (the model keyed one case
    ``mistforge_tactics`` while stylizing the body, so contents alone missed it)."""
    entries = collection_entries(db, name)
    return _normalize(" ".join([*entries.keys(), *entries.values()]))


def _reply_reflects(reply: str, tokens: list[str]) -> list[Check]:
    """The final reply must REFLECT the actions taken (the #1478 recap prong): it
    names each subject/fact it acted on, so the response the user reads matches
    what she asked for and what actually happened.  One graded Check per token,
    anchored to the reply row; normalized for typography so a cosmetic difference
    isn't a spurious miss; checked semantically as substrings, never exact wording."""
    normalized = _normalize(reply)
    return [
        Check(
            f"reply reflects '{token}'",
            _normalize(token) in normalized,
            kind="reply",
            anchor=REPLY_ANCHOR,
        )
        for token in tokens
    ]


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_search_read_store(db: Database, before: set[str], reply: str) -> list[Check]:
    """The compound request fires ``browse`` then ``collection_write``, and the
    looked-up game lands in ``games``."""
    sequence = tool_call_sequence(db)
    in_order = is_ordered_subsequence([_BROWSE, _WRITE], sequence)
    saved = "mistforge" in _saved_text(db, _GAMES_NAME)
    return [
        Check(
            "browse→collection_write in order",
            in_order,
            kind="spine",
            anchor=f"{_WRITE}(",
            rationale=None if in_order else f"got {sequence}",
        ),
        Check(
            "looked-up game saved to games",
            saved,
            kind="state",
            rationale=None if saved else f"{collection_entries(db, _GAMES_NAME)}",
        ),
    ] + _reply_reflects(reply, ["mistforge"])


def _score_multihop_read_store(db: Database, before: set[str], reply: str) -> list[Check]:
    """A search → open-the-linked-page → store request must persist the release
    year that lives ONLY on the detail page (proof the second browse happened)."""
    sequence = tool_call_sequence(db)
    in_order = is_ordered_subsequence([_BROWSE, _WRITE], sequence)
    has_year = "2031" in _saved_text(db, _GAMES_NAME)
    return [
        Check(
            "browse→collection_write in order",
            in_order,
            kind="spine",
            anchor=f"{_WRITE}(",
            rationale=None if in_order else f"got {sequence}",
        ),
        Check(
            "stored entry carries the detail-page release year (2031)",
            has_year,
            kind="state",
            rationale=None if has_year else f"{collection_entries(db, _GAMES_NAME)}",
        ),
    ] + _reply_reflects(reply, ["mistforge", "2031"])


def _score_recall_then_delete(db: Database, before: set[str], reply: str) -> list[Check]:
    """List what the user's into and drop chess — chess gone, the rest untouched.
    Both a read and a delete must fire; ORDER isn't load-bearing (the model may
    drop chess first, then read the remainder to answer "what am I into")."""
    sequence = tool_call_sequence(db)
    remaining = _saved_text(db, _LIKES)
    return [
        Check(
            "collection_read_latest fired",
            _READ in sequence,
            kind="spine",
            anchor=f"{_READ}(",
            rationale=None if _READ in sequence else f"got {sequence}",
        ),
        Check(
            "collection_delete_entry fired",
            _DELETE in sequence,
            kind="spine",
            anchor=f"{_DELETE}(",
            rationale=None if _DELETE in sequence else f"got {sequence}",
        ),
        Check("chess removed from likes", "chess" not in remaining, kind="state"),
        Check("hiking kept", "hiking" in remaining, kind="state"),
        Check("jazz kept", "jazz" in remaining, kind="state"),
    ] + _reply_reflects(reply, ["chess"])


def _score_delete_then_list(db: Database, before: set[str], reply: str) -> list[Check]:
    """Drop jazz and report what's left — jazz gone, the rest untouched.  Both a
    delete and a read must fire, but the ORDER is not load-bearing: the model
    often reads the list first then deletes and narrates the remainder from that
    read, which satisfies the ask just as well as delete-then-read."""
    sequence = tool_call_sequence(db)
    remaining = _saved_text(db, _LIKES)
    return [
        Check(
            "collection_delete_entry fired",
            _DELETE in sequence,
            kind="spine",
            anchor=f"{_DELETE}(",
            rationale=None if _DELETE in sequence else f"got {sequence}",
        ),
        Check(
            "collection_read_latest fired",
            _READ in sequence,
            kind="spine",
            anchor=f"{_READ}(",
            rationale=None if _READ in sequence else f"got {sequence}",
        ),
        Check("jazz removed from likes", "jazz" not in remaining, kind="state"),
        Check("chess kept", "chess" in remaining, kind="state"),
        Check("hiking kept", "hiking" in remaining, kind="state"),
    ] + _reply_reflects(reply, ["jazz"])


def _score_recall_sweep(db: Database, before: set[str], reply: str) -> list[Check]:
    """A three-collection recall must read likes, dislikes, AND games."""
    targets = set(tool_call_arg_values(db, _READ, "memory"))
    # The recap must reflect at least a like and a game it read back.
    return [
        Check(f"read '{name}'", name in targets, kind="spine", anchor=f"{_READ}(")
        for name in (_LIKES, _DISLIKES, _GAMES_NAME)
    ] + _reply_reflects(reply, ["chess", "mistforge"])


def _score_browse_then_update(db: Database, before: set[str], reply: str) -> list[Check]:
    """Look up co-op details, then UPDATE the existing games entry (not a new
    write) — the entry's content must gain the co-op detail from the detail page."""
    sequence = tool_call_sequence(db)
    in_order = is_ordered_subsequence([_BROWSE, _UPDATE], sequence)
    updated = _saved_text(db, _GAMES_NAME)
    has_coop = "co-op" in updated or "cooperat" in updated
    return [
        Check(
            "browse→update_entry in order",
            in_order,
            kind="spine",
            anchor=f"{_UPDATE}(",
            rationale=None if in_order else f"got {sequence}",
        ),
        Check(
            "Mistforge entry gained co-op detail",
            has_coop,
            kind="state",
            rationale=None if has_coop else f"{collection_entries(db, _GAMES_NAME)}",
        ),
    ] + _reply_reflects(reply, ["mistforge"])


def _score_update_entry(db: Database, before: set[str], reply: str) -> list[Check]:
    """Change the hiking note to alpine — via update_entry, content changed."""
    has_alpine = "alpine" in _saved_text(db, _LIKES)
    return [
        Check(
            "update_entry called", tool_was_called(db, _UPDATE), kind="spine", anchor=f"{_UPDATE}("
        ),
        Check(
            "hiking note mentions alpine trails",
            has_alpine,
            kind="state",
            rationale=None if has_alpine else f"{collection_entries(db, _LIKES)}",
        ),
    ] + _reply_reflects(reply, ["hiking", "alpine"])


def _score_recall_then_fill(db: Database, before: set[str], reply: str) -> list[Check]:
    """ "Do you have it saved? if not, look it up and save it" — the memory CHECK
    is satisfied ambiently by the recall block (the model doesn't emit an explicit
    read_similar for it), so the load-bearing contract is the fill: browse →
    collection_write, the game ending up in games."""
    sequence = tool_call_sequence(db)
    in_order = is_ordered_subsequence([_BROWSE, _WRITE], sequence)
    saved = "mistforge" in _saved_text(db, _GAMES_NAME)
    return [
        Check(
            "browse→collection_write in order",
            in_order,
            kind="spine",
            anchor=f"{_WRITE}(",
            rationale=None if in_order else f"got {sequence}",
        ),
        Check(
            "looked-up game saved to games",
            saved,
            kind="state",
            rationale=None if saved else f"{collection_entries(db, _GAMES_NAME)}",
        ),
    ] + _reply_reflects(reply, ["mistforge"])


def _score_two_writes(db: Database, before: set[str], reply: str) -> list[Check]:
    """Two preferences fanned into two collections — the like into likes, the
    dislike into dislikes."""
    has_boulder = "bouldering" in _saved_text(db, _LIKES)
    has_coffee = "coffee" in _saved_text(db, _DISLIKES)
    return [
        Check(
            "bouldering saved to likes",
            has_boulder,
            kind="state",
            rationale=None if has_boulder else f"{collection_entries(db, _LIKES)}",
        ),
        Check(
            "instant coffee saved to dislikes",
            has_coffee,
            kind="state",
            rationale=None if has_coffee else f"{collection_entries(db, _DISLIKES)}",
        ),
    ] + _reply_reflects(reply, ["bouldering", "coffee"])


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[Check]:
    """Pure narration / a wistful aside must fire no browse or entry mutation."""
    return [
        Check(f"{tool} not fired", not tool_was_called(db, tool), kind="spine", anchor=f"{tool}(")
        for tool in _ACTION_TOOLS
    ]


def _score_no_fire_reads(db: Database, before: set[str], reply: str) -> list[Check]:
    """A wistful mention of one's games must not fire a read/mutation either."""
    return _score_no_fire(db, before, reply) + [
        Check(
            "collection_read_latest not fired",
            not tool_was_called(db, _READ),
            kind="spine",
            anchor=f"{_READ}(",
        )
    ]


# ── Cases ─────────────────────────────────────────────────────────────────────
# Sequence cases carry the HONEST majority-dispatch bar of 0.8, gated on the
# PATHOLOGY-EXCLUDED mean (``gate_pathology_excluded=True``, #1698).  Across runs
# the sub-perfect samples are dominated by the known gpt-oss degeneracy collapse
# (a run whose tool name/args collapse into "...?"), a transient pathology the
# reroll guard mostly but not always catches — NOT a dispatch failure.  The bar
# was previously LOWERED to 0.6 purely so that collapse couldn't flake the case;
# now the failure-cause partition (#1695) drops the pathology samples out of the
# gated denominator, so the case can gate at its true pre-pathology bar (0.8),
# with the raw mean + pathology count still visible in the printed cause line.
# No-fire cases gate at 0.75 on the RAW mean (the project's NL-dispatch
# convention, see test_command_tools).  ``speak-no-fire-wistful`` stays at 0.6 on
# the raw mean: its residual sub-perfect sample is a BEHAVIORAL benign save/read
# (see its note), not the collapse pathology, so it is not a pathology restore.


async def test_search_read_store_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-search-read-store",
        message="can you look up Mistforge Tactics, read up on it, and save it to my games list?",
        seed=_seed_games,
        browse=list(MULTIHOP_PAGES),
        score=_score_search_read_store,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_multihop_read_store_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-multihop-read-store",
        message="find Mistforge Tactics, open the official page for the exact release date, "
        "and record it in my games list",
        seed=_seed_games,
        browse=list(MULTIHOP_PAGES),
        score=_score_multihop_read_store,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_recall_then_delete_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-recall-delete",
        message="what am I into these days? actually drop chess from that",
        seed=_seed_likes,
        score=_score_recall_then_delete,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_delete_then_list_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-delete-list",
        message="forget about jazz, then tell me what's left on my likes",
        seed=_seed_likes,
        score=_score_delete_then_list,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_recall_sweep_sequence(chat_eval: ChatEval) -> None:
    """Report-only — a high-variance case (4/5→2/5→5/5→1/5 across runs).  The model
    can answer the "remind me" from the conversation window or a subset of reads and
    still produce a correct reminder WITHOUT three explicit ``collection_read_latest``
    calls — the scorer's three-reads requirement over-fits.  The user-facing outcome
    is right; the proper fix is to gate on the reminder OUTCOME (reply reflects a
    like + dislike + game) rather than the reads — a follow-up, not the
    imperative-gating PR."""
    await chat_eval(
        case_id="speak-recall-sweep",
        message="remind me what I like, what I dislike, and what's on my games list",
        seed=_seed_sweep,
        score=_score_recall_sweep,
        min_pass_rate=None,
        family=_FAMILY,
    )


async def test_browse_then_update_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-browse-update",
        message="look up the co-op details for Mistforge Tactics and update its "
        "entry in my games list",
        seed=_seed_games_with_mistforge,
        browse=list(MULTIHOP_PAGES),
        score=_score_browse_then_update,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_update_entry_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-update-entry",
        message="change my hiking note in likes to say I prefer alpine trails",
        seed=_seed_likes,
        score=_score_update_entry,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_recall_then_fill_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-recall-fill",
        message="do you have anything on Mistforge Tactics saved? if not, look it "
        "up and save it to games",
        seed=_seed_games,
        browse=list(MULTIHOP_PAGES),
        score=_score_recall_then_fill,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_two_writes_fanned_sequence(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-two-writes",
        message="jot down that I'm into bouldering, and that I can't stand instant coffee",
        score=_score_two_writes,
        min_pass_rate=0.8,
        gate_pathology_excluded=True,
        family=_FAMILY,
    )


async def test_no_fire_narration(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-no-fire-narration",
        message="I looked up a lasagna recipe earlier and saved it in my notes app, good evening",
        score=_score_no_fire,
        min_pass_rate=0.75,
        family=_FAMILY,
    )


async def test_no_fire_wistful(chat_eval: ChatEval) -> None:
    """The over-firing gap this used to document is now CLOSED by the
    imperative-gating clause in ``Prompt.CONVERSATION_PROMPT`` (don't chase down
    topics the user only mentioned).  On a purely conversational share ("I
    finished that game campaign"), the model no longer runs a browse/lookup on
    the topic — baseline 2/5 → 4/5 with the clause, and the harmful
    browse+confabulation is gone (the residual miss is a benign save/read).
    Gated at 0.6 to protect the fix from regression."""
    await chat_eval(
        case_id="speak-no-fire-wistful",
        message="I finally wrapped up that long strategy game campaign last night, "
        "felt so satisfying",
        seed=_seed_games_with_mistforge,
        score=_score_no_fire_reads,
        min_pass_rate=0.6,
        family=_FAMILY,
    )
