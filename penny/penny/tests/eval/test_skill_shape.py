"""Live-model contract for the run-end skill SHAPE draw (#1803).

A demonstrated round hands over several values, and the labeller
(``test_skill_labelling.py``) says where each one came from.  That question has an
answer in every round and it answers it well — but it is not the question that
decides whether the learned routine can ever fire.  *"Keep an eye on the aurora
deck 2 price"* gives up two values, BOTH from the user, and nothing in that round
says which of them will vary the next time.  Deciding it anyway is what produced
skills that named themselves for a value and then required it —
`record-product-price` demanding a `what_to_extract` its own name already gave —
so the routine could not fire from the natural second ask.

So a SECOND micro-context decides what the routine IS: given the user's message and
only the values the labeller kept, it writes the routine's name and description AND
marks each value CONSTANT (the routine is about it) or PARAMETER (it is pointed at
it).  One decision, so the two halves cannot contradict.

These cases score THAT draw.  The labeller's own contract lives next door and runs
when the labeller changes — its two cases are not re-run here, since a shape
iteration says nothing about provenance.

All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import LabellerEval

pytestmark = pytest.mark.eval

_FAMILY = "skill-shape"

_TARGET = "aurora-prices"
_PRICE = "$499"
_LISTING = "https://faux-market.example/aurora-deck-2"

# The instigating ask carries the INTENT — it is the only place the round says what
# the user was actually after, and therefore the only evidence for what the routine
# is about rather than merely pointed at.
_ASK = "can you keep an eye on the aurora deck 2 price for me?"
_UTTERANCE = f"read {_LISTING}, find the current price, and remember it"

_BROWSE = (
    "browse",
    {"queries": [_LISTING], "extract": "the current price"},
    f"You opened the Aurora Deck 2 listing (browse result)\n{_PRICE}",
    True,
)
_WRITE = (
    "collection_write",
    {"memory": _TARGET, "entries": [{"key": "aurora deck 2 price", "content": _PRICE}]},
    "You saved an entry to aurora-prices: (collection_write result)\nWrote 1 entry.",
    True,
)


@pytest.mark.asyncio
async def test_the_value_the_routine_is_about_becomes_a_constant(labeller_eval: LabellerEval):
    """A skill must not name itself for a value it then asks the user to supply.

    The page VARIES between uses and stays a parameter; what the routine is FOR is
    baked in, and the collector still runs against it.

    **The scored direction is one of two coherent answers, and it is the one this ask
    asks for.**  This round could honestly become a price watcher pointed at a page
    (the price baked) OR a pull-anything-off-a-page routine (both values asked for),
    and the draw is free to write either — what it may never do is commit to one in
    the name and the other in the parameters.  Scoring the first reads the user's
    stated intent as the tiebreak: they said *the price*.  Report-only until that
    reading is confirmed against a real run — a scorer encoding the wrong intent
    would fail the model for being right, which is the failure this suite exists to
    avoid."""
    await labeller_eval(
        case_id="shape-value-the-routine-is-about-is-constant",
        utterance=_UTTERANCE,
        conversation=[_ASK],
        calls=[_BROWSE, _WRITE],
        target=_TARGET,
        user_values=[_LISTING],
        constant_values=["the current price"],
        assistant_values=[],
        min_pass_rate=None,
        family=_FAMILY,
    )
