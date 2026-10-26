"""A numbered link already in the store is cited, not re-searched for (#2832).

The contract is `docs/eval-case-design.md`.

**The shape is production's #2832 turn, genericised.** A collector already wrote
an entry carrying the game's store page as its `Source:` line; a notification
about the find went out without a link; the user noticed and asked for it.
Across the real run's many steps (its own run id's promptlog rows), the model
never cited the handle its own `collection_list` read put in front of it
(`Source: [209]`) — it searched the web five more times, at one point
literally searching for the text `[1]`, and twice drafted a raw URL that the
hallucinated-URL validator discarded. Its own thinking says why:

> The browse tool is returning placeholder links (e.g., `[78]`). I cannot use
> these as real URLs.

> Okay, so `209` is an index/handle in the `browse-results` log, not
> necessarily the log ID. I'll try `log_read('browse-results')` to see the
> last entries and their IDs.

A bare `[n]` beside a number with no domain or title reads as an opaque
internal handle worth resolving, not as a citable link — so the model goes
looking for "the real" address instead of writing the number down. This case
reproduces that shape, condensed to its smallest form: one already-stored
entry whose `Source:` carries a link, one short prior exchange establishing
that the last message about it had no link, and a follow-up asking for it.

**Real game, invented collection.** The real production turn named a real
game on a real storefront; this world swaps in a DIFFERENT real game on the
same real storefront (Steam) — Big Pharma, a real but small, little-known
indie title — because the repo is public and the specific game and
collection are the user's own. Picked deliberately obscure, not a cultural
touchstone like the first swap (Stardew Valley) tried: that run's claim
passed on main anyway, because the model's own training data already
supplies Stardew Valley's Steam URL verbatim once it gives up resolving the
handle, which masks the failure on the one measure that matters most. A
small, rarely-discussed title carries no such safety net. The collection's
name (`indie_game_radar`), its description, and the conversation around it
are invented to the same shape as a real media-watch collection.

**What is asserted.** The final reply — the message actually delivered, after
every handle it cites has been expanded back to the real address — carries the
host-and-path prefix of the store page the world (and the store) actually
gave it. The store is read back: a read-only ask, nothing lands, nothing
already held is disturbed.

**What is NOT claimed:** which tool resolved it — `collection_list`, a fresh
`browse`, or both — how many calls it took, or how she words the reply. Those
are route and are measured as variance and cost, which is the whole point:
the code owner's instruction for this kind of failure is that whether the
model times out or finishes is not what matters, the token cost and the
tool-sequence spread ARE the measurement of the thrash itself.

REPORT-ONLY (``min_pass_rate=None``) on its first run: this run is to show
whether the failure reproduces, not to grade it.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory.objects import EntryInput
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _log_chat_step, _seeded_response
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "stored-link-number-cited"

# ── The world's constants ─────────────────────────────────────────────────────
#
# Big Pharma stands in for the real production game: a different REAL game on
# the same real storefront (Steam), never an invented one — and deliberately a
# small, little-known one, so the model has no training-data shortcut to the
# right answer (see the module docstring on why the first pick, Stardew
# Valley, let the claim pass on main for the wrong reason).  The collection's
# own name, description and schedule are invented to the shape of a real
# media-watch job — the entry's shape (`Title:` / `Media Type:` / `Summary:` /
# `Source:` lines in one free-text `content`) is the one a collector cycle
# really writes.

_GAME = "Big Pharma"
_COLLECTION = "indie_game_radar"
_COLLECTION_DESCRIPTION = "Indie games the user might like, one find per cycle, with a source link."
_COLLECTION_PROMPT = "Find one indie game the user hasn't seen yet and save it with a source link."
_COLLECTION_SCHEDULE = "DTSTART:20261001T120000\nFREQ=DAILY"

_STORE_URL = "https://store.steampowered.com/app/376620/Big_Pharma/"
# The host-and-path prefix every served URL for this game shares — the shape a
# cited, expanded link must carry.  Not a slug: a name-only reply would satisfy
# a slug check without any address in it.
_URL_PREFIX = "store.steampowered.com/app/376620"

# The entry a collector cycle already wrote — the real shape (`Title:` /
# `Media Type:` / `Summary:` / `Source:` lines in one free-text `content`), the
# same shape production's entry carried.
_ENTRY_CONTENT = (
    "Title: Big Pharma\n"
    "Media Type: video game\n"
    "Summary: A pharmaceutical factory tycoon — you design production lines, "
    "discover drug combinations, and grow a small lab into a sprawling plant.\n"
    f"Source: {_STORE_URL}\n"
)

# The prior exchange: the last message about this game had no link, the user
# noticed, and Penny promised to grab it — the condition the follow-up (the
# ask under test) answers.  Plain text both ways; no tool call, since nothing
# about THIS case turns on how that promise was made, only on what answers it.
_PRIOR_ASK = (
    f"oh hey, that message about {_GAME} didn't have a link in it — can you grab "
    "me the store page when you get a sec?"
)
_PRIOR_REPLY = (
    "Oof, good catch — no link snuck through on that one! Let me grab it for you now, one sec! 🔍"
)


def _seed_stored_link(db: Database) -> None:
    """Lay the collection and its already-written entry down through the SAME
    chokepoints a live collector cycle writes through, then seed the prior
    exchange in the message log and the conversation history — both durable,
    the way a real turn's history and a real collector's write both are."""
    run_id = seeded_run_id(_COLLECTION)
    db.memories.create_collection(
        _COLLECTION,
        _COLLECTION_DESCRIPTION,
        extraction_prompt=_COLLECTION_PROMPT,
        schedule=_COLLECTION_SCHEDULE,
        notify=True,
        created_by_run_id=run_id,
    )
    require_memory(db, _COLLECTION).write_rule_decided(
        [EntryInput(key=_GAME, content=_ENTRY_CONTENT)],
        author="chat",
        run_id=run_id,
    )
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _PRIOR_ASK
    )
    assert ask_id is not None, "the seeded prior ask failed to log — nothing for the run to link to"
    _log_chat_step(
        db,
        run_id=run_id,
        messages=[{"role": "user", "content": _PRIOR_ASK}],
        response=_seeded_response(_PRIOR_REPLY),
    )
    db.messages.log_message(PennyConstants.MessageDirection.OUTGOING, "penny", _PRIOR_REPLY)


# ── The web behind browse, for the route that searches instead ────────────────
#
# A store page carrying its own `Source:` line in its visible body — the shape
# every canned page here uses — and a search page linking it.  Present so a
# sample that chooses to search rather than read the collection can still
# reach a real, servable address; which route a sample takes is variance, not
# a claim.

_STORE_PAGE = CannedPage(
    match="big pharma",
    text=(
        "Title: Big Pharma on Steam\n"
        f"Source: {_STORE_URL}\n"
        "\n"
        "Big Pharma is a pharmaceutical factory tycoon. You design production "
        "lines, discover drug combinations, and grow a small lab into a "
        "sprawling plant — and decide how ethical a pharma company you want to "
        "run.\n"
        "\n"
        "Price: $19.99\n"
        "Platforms: Windows, macOS, Linux\n"
    ),
)

_SEARCH_PAGE = CannedPage(
    match="big pharma",
    text=(
        "Title: big pharma — search results\n"
        f"[Big Pharma on Steam]({_STORE_URL})\n"
        "A pharmaceutical factory tycoon where you design production lines and "
        "grow a small lab into a sprawling plant.\n"
    ),
    search=True,
)

_WORLD = World(
    name="an already-stored game with a store page behind browse",
    pages=(_STORE_PAGE, _SEARCH_PAGE),
    keeps=(),
    excludes=(),
    answers=(),
)

# ── The five wordings of the follow-up ────────────────────────────────────────
#
# All of them nudge for the link the prior exchange promised, naming neither a
# tool nor an expected route — the facts they carry are the same unresolved
# promise, worded the way a person nudges a second time.

_ASK = "okay so did you find the link?"
_ASKS = (
    "any luck grabbing that link yet?",
    "hey, did you ever get me that link?",
    "so... did that link turn up?",
    "you still haven't sent me the link though?",
)


def _assert_every_wording_nudges_the_promise() -> None:
    """Every wording asks about the SAME unresolved link, and none of them names
    a tool, a game, or a store — the referent comes only from the seeded prior
    exchange, which is the condition this case turns on."""
    for ask in (_ASK, *_ASKS):
        folded = ask.casefold()
        assert "link" in folded, f"this wording names no link — {ask!r}"
        for banned in ("browse", "search", "steam", _GAME.casefold(), "collection"):
            assert banned not in folded, f"this wording names {banned!r} — {ask!r}"
    assert _URL_PREFIX in _STORE_PAGE.text, "the store page carries no served URL prefix"
    assert _STORE_URL in _SEARCH_PAGE.text, "the search page links no store page"
    assert _STORE_URL in _ENTRY_CONTENT, "the seeded entry carries no store URL"


_assert_every_wording_nudges_the_promise()


# ── The claim ──────────────────────────────────────────────────────────────


def _reply_carries_the_stored_link(sample: SampleObservation, _world: World) -> Answer:
    """The DELIVERED reply — handles already expanded back to real addresses by
    the time it reached the mock channel — carries this game's store URL
    prefix. Typography-folded containment, the same read `test_browse_cites_
    source.py` uses: a sample that resolved the handle from the collection, one
    that re-browsed for it, and one that did both all pass the same way, since
    which route got there is not what this claims."""
    hit = _URL_PREFIX in fold_typography(sample.reply)
    return hit, f"served URL prefix {_URL_PREFIX!r} in the delivered reply: {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_stored_link_number_is_cited_not_rebrowsed(chat_eval: ChatEval, model: str) -> None:
    """A collection already holds an entry whose stored link renders as a bare
    number; asked for that link a second time, Penny's reply carries the real
    address rather than another round of searching."""
    cohort = await chat_eval(
        case_id="stored-link-number-cited",
        behaviour=(
            "In the chat agent, when Penny already holds a numbered link for what "
            "the user is asking about, her reply carries the real address it "
            "stands for."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_stored_link,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ASKS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the first run shows whether this reproduces
        family=_FAMILY,
        timeout=420.0,  # room for a death-spiral sample that searches repeatedly before answering
    )
    # STORE — a read-only ask: the seeded collection and its entry are exactly
    # as the turn found them.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE — the claim this issue exists for.
    cohort.claim(
        "reply: it carries the real address of the stored/served link",
        _reply_carries_the_stored_link,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
