"""#2774 reproduction + fix: a detail the user stated days ago, in a message now
outside both the history window and the conversation map, is findable by WORD as
well as by meaning — and the conversation gets its own share of find's result
regardless of what the store holds.

The ask names THREE things at once — the real pet, Clementine, plus two words
("Basil", "Nutmeg") that read as pet names but are never once said anywhere in
this world. That is the exact shape the issue's own prod evidence names: "a
query naming several names at once", where one blended embedding dilutes any
one name's signal (a query naming several pets at once — e.g. "Mochi,
Biscuit, or Pepper" — missed the one that mattered, plain cosine diluted
against the other names). The two decoy collections
are seeded on the LITERAL words "basil" (an herb) and "nutmeg" (a spice) — not
because anyone asked about herbs or spices, but because that literal overlap
is exactly what pulled a plain-cosine query away from the one message that
actually answers the ask, on the unpatched mechanism this case was written
against. Resolving it needs BOTH halves of the fix: the keyword leg catches
"Clementine" by word regardless of how diluted the query's embedding is, and
the conversation section's own independent budget means the store's entries
(however well THEY score) can never fill every slot and push the message out.

``_DAYS_BACK`` (4) is chosen to clear BOTH bounds the ask must be outside of:
``CHAT_HISTORY_TURNS`` (5 exchanges — filler pairs below push the exchange past
it by count) and ``CHAT_MAP_DAYS`` (2 — today and yesterday; 4 days back is
outside that horizon by date, not just by a hair). Every name, detail and
collection here is invented for this case — the repo is public.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlmodel import col, update

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import MessageLog
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import REPLY_SPREAD, TOOL_SEQUENCE, fold_typography
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.lookup_world import (
    _FILLER_PAIRS,
    _FILLER_PENNY,
    _FILLER_USER,
    assert_seeded_turns_embedded,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "find-hybrid-search"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY

# Outside CHAT_MAP_DAYS (2, today/yesterday) by date; the filler pairs below
# also push the exchange outside CHAT_HISTORY_TURNS (5) by count.
_DAYS_BACK = 4

_PET_ASK_FROM_USER = (
    "oh also — Clementine will only eat the Wilder Harvest brand of wet food, "
    "she turns her nose up at every other brand"
)
_PET_REPLY = "noted — Wilder Harvest it is, I'll remember that for her."
_PET_ANSWER = "wilder harvest"

assert _PET_ANSWER in _PET_ASK_FROM_USER.lower()

# Decoys that literally contain two of the ask's three words ("basil", "nutmeg")
# — on the unpatched embedding-only ranking these compete directly with the
# pet message for the shared result slots, since the blended query's own
# embedding leans on whichever of its three words a candidate matches. Plus one
# generic profile collection, closer to the issue's own "four user_profile
# facts" shape. None of them mention Clementine or the brand, so the ask can't
# be answered by a blind read of whatever the store happens to hold.
_DECOY_HERBS: SynthCollection = SynthCollection(
    "herb-garden",
    "Notes on growing culinary herbs on the balcony.",
    (
        "basil — pinch back the flowers weekly for a bushier plant.",
        "rosemary — let the soil dry out between waterings.",
    ),
)
_DECOY_SPICES: SynthCollection = SynthCollection(
    "spice-rack",
    "Notes on keeping the spice rack stocked and fresh.",
    (
        "nutmeg — buy it whole and grate fresh, pre-ground loses flavor fast.",
        "cinnamon — the Ceylon kind, not cassia, for the good stuff.",
    ),
)
_DECOY_PROFILE: SynthCollection = SynthCollection(
    "about-me",
    "Background facts about the user.",
    (
        "career — product manager at a mid-size logistics company.",
        "interests — woodworking, long-distance cycling, and jazz vinyl.",
    ),
)
_DECOYS: tuple[SynthCollection, ...] = (_DECOY_HERBS, _DECOY_SPICES, _DECOY_PROFILE)

assert not any(_PET_ANSWER in entry for synth in _DECOYS for entry in synth.entries)


def _backdate_messages(db: Database, message_ids: list[int], days: int) -> None:
    """Push the pet exchange's message rows back by ``days`` — true of the seeded
    world, not of anything the model did (the same shape
    ``test_conversation_map_recall.py``'s own ``_backdate_messages`` uses)."""
    stamp = db.clock.now() - timedelta(days=days)
    with db.get_session() as session:
        session.exec(
            update(MessageLog).where(col(MessageLog.id).in_(message_ids)).values(timestamp=stamp)
        )
        session.commit()


def _seed_filler(db: Database) -> None:
    """Filler pairs that push the pet exchange out of the history window by
    count — logged at seed time, so they are the NEWEST turns and the window's
    five user turns all land among them, none of them the pet exchange."""
    for index in range(_FILLER_PAIRS):
        db.messages.log_message(_INCOMING, TEST_SENDER, _FILLER_USER[index % len(_FILLER_USER)])
        db.messages.log_message(
            _OUTGOING, _PENNY, _FILLER_PENNY[index % len(_FILLER_PENNY)], recipient=TEST_SENDER
        )


def _seed_pet_exchange(db: Database) -> None:
    """The exchange that carries the one detail the ask depends on, backdated
    past both the history window and the conversation map, then buried under
    filler and decoy ground."""
    ask_id = db.messages.log_message(_INCOMING, TEST_SENDER, _PET_ASK_FROM_USER)
    reply_id = db.messages.log_message(_OUTGOING, _PENNY, _PET_REPLY, recipient=TEST_SENDER)
    assert ask_id is not None and reply_id is not None, (
        "the seeded exchange failed to log — nothing for find to recover"
    )
    _seed_filler(db)
    _backdate_messages(db, [ask_id, reply_id], _DAYS_BACK)


def _assert_the_world(penny: Penny) -> None:
    """The case's premise: the decoys exist, the pet exchange is really in the
    store (embedded, so both find's legs can reach it), and it carries the one
    detail the claim depends on."""
    db = penny.db
    names = {row.name for row in db.memories.list_all() if not row.archived}
    for synth in _DECOYS:
        assert synth.name in names, f"{synth.name}: the decoy collection is missing"
    assert_seeded_turns_embedded(db)


_ASK = "hey remind me what you remember about Clementine, Basil, or Nutmeg"
_ALSO_PHRASED = (
    "what have I told you about Clementine, Basil, or Nutmeg?",
    "remind me what you know about Clementine, Basil, and Nutmeg",
    "what's the deal with Clementine, Basil, or Nutmeg again?",
)


def _every_wording_names_all_three_but_not_the_answer(*asks: str) -> None:
    """Each wording blends all three words into one query — the mechanism this
    case depends on — and none of them names the brand (that is the fact only
    the backdated message carries)."""
    for ask in asks:
        folded = fold_typography(ask)
        assert "clementine" in folded
        assert "basil" in folded
        assert "nutmeg" in folded
        assert _PET_ANSWER not in folded


_WORLD = World(
    name="a pet detail buried behind decoys, days back",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(_PET_ANSWER,),
    stores=_DECOYS,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_find_recalls_a_detail_from_days_back(chat_eval: ChatEval, model: str) -> None:
    """Days later, past both the history window and the conversation map, the
    user asks about a detail they themselves stated once, blended into a query
    naming two other words that never appear anywhere in this world — a
    literal word (a brand name) an embedding alone can blend away, with decoy
    ground in the store sharing two of the query's three words so a blind
    read can't answer it either."""
    _every_wording_names_all_three_but_not_the_answer(_ASK, *_ALSO_PHRASED)
    cohort = await chat_eval(
        case_id="find-recalls-old-message-detail",
        behaviour=(
            "In the chat agent, when the user asks about a detail they themselves "
            "stated days ago — outside the history window and the conversation map, "
            "with generic ground sitting in the store — Penny recovers the detail "
            "and states it."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_pet_exchange,
        prepare=_assert_the_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — the decoys all survive the turn as they were seeded.
    cohort.assert_the_running_mechanisms_survive()
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
