"""Reading one entry from a collection at random: one case (#2812).

The contract is `docs/eval-case-design.md`.

**Why this case exists.** `collection_sample` — a second tool whose only job was a uniform
random draw over a collection — was retired and folded into `collection_list(order='random')`
(#2812: no chat or collector call ever reached it in 444 measured prod tool calls, and the
same read is now one argument on the tool that already reads a collection). A changed tool
no case exercised before, so this is the new one: a chat ask for "one of my X, picked at
random" has to still reach a collection read, and the entry the reply names has to be one the
collection actually holds — the one thing a random draw cannot get right by inventing an
answer instead of reading one.

**The claims.** PRESERVATION — the collection the ask reads holds exactly what it was seeded
with; a pure read names nothing to write. PROVENANCE — every specific value in the reply
traces to something the turn was given, which is what catches a free-chosen cocktail that
was never in the list standing in for a real draw.

**What is NOT claimed, and why.** Which call reached the switch is a ROUTE — `collection_list`
with `order='random'` is one path, `collection_list` read whole and picked over, or `find`,
are others a plugin or a future tool could add — so `TOOL_SEQUENCE` is measured, never
asserted. Nothing claims WHICH of the four cocktails the reply names: a uniform draw is
Penny's own coin flip, not a value this case can state in advance (`docs/principles.md` §4.1).

Four real cocktails stand in for a saved list — the repo is public, so nothing here is drawn
from any real person's own saved collection.

REPORT-ONLY (``min_pass_rate=None``): the ceiling this run proposes is the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "nl-dispatch"

_CASE_ID = "collection-random-pick"

_COCKTAILS = SynthCollection(
    "cocktails-to-try",
    "Cocktails I want to make at home: name, spirits, how it's built.",
    entries=(
        "Negroni — gin, Campari, sweet vermouth, equal parts, stirred.",
        "Old Fashioned — bourbon, sugar, angostura bitters, orange twist.",
        "Daiquiri — white rum, lime juice, simple syrup, shaken.",
        "Paper Plane — bourbon, Aperol, Amaro Nonino, lemon juice, equal parts.",
    ),
)
_HELD_KEYS = tuple(key for key, _ in _COCKTAILS.keyed)

_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=(), stores=(_COCKTAILS,))

_ASK = "pick one of my cocktails to try this weekend, totally at random"
_ALSO_PHRASED = (
    "give me a random one from my cocktail list",
    "surprise me with one of my saved cocktails, picked at random",
    "can you grab a random cocktail from the list I've been keeping?",
    "just roll the dice on one of my cocktails for this weekend",
)

_BEHAVIOUR = (
    "In the chat agent, when the user asks for one of their saved collection's entries at "
    "random, Penny reports one the collection actually holds."
)


def assert_cocktails_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs: the
    collection holds exactly the four seeded cocktails, so a reply naming anything else
    cannot be answered by the seed."""
    held = sorted(entry.key or "" for entry in require_memory(penny.db, _COCKTAILS.name).read_all())
    expected = sorted(_HELD_KEYS)
    assert held == expected, f"{_CASE_ID}: {_COCKTAILS.name} must hold {expected}, it holds {held}"


# ── The claim, as a pure function over one sample ─────────────────────────────


def _the_collection_is_unchanged(sample: SampleObservation, _world: World) -> Answer:
    """The collection still holds exactly its four seeded cocktails — a pure read asks for
    nothing to be written, so any drift (an addition, a removal, a rewrite) is a turn that
    did something the ask never asked for."""
    held = {entry.key for entry in sample.held if entry.collection == _COCKTAILS.name}
    expected = set(_HELD_KEYS)
    return held == expected, f"{_COCKTAILS.name} holds {sorted(held)}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_random_pick_from_a_collection_names_one_it_actually_holds(
    chat_eval: ChatEval, model: str
) -> None:
    """A saved list of four cocktails, and an ask for one of them at random."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.MEMORY,
        model=model,
        prepare=assert_cocktails_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — PRESERVATION: the ask reads, it does not write.
    cohort.claim(
        "state: the collection still holds exactly its four cocktails",
        _the_collection_is_unchanged,
        SpecCategory.STORE,
    )

    # PROVENANCE — the pick the reply names has to trace to the collection it was given,
    # never an invented cocktail standing in for a real draw.
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — the call is a route, and more than one
    # reaches the same read.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
