"""Live-model contract for the run-end skill SHAPE micro-context (#1803), in isolation.

A demonstrated round hands over several values, and the parameter labeller
(``test_skill_labelling.py``) says where each one came from.  That question has an
answer in every round and it answers it well — but it is not the question that
decides whether the learned routine can ever fire.  *"Keep an eye on the aurora deck
2 price"* gives up two values, BOTH from the user, and nothing in that round says
which of them will vary next time.  Deciding it anyway is what produced skills that
named themselves for a value and then required it — `record-product-price` demanding
a `what_to_extract` its own name already gave — so the routine could not fire from
the natural second ask.

So a SECOND micro-context decides what the routine IS: given the user's message and
only the values the labeller kept, it writes the routine's name and description AND
marks each value CONSTANT (the routine is about it) or PARAMETER (it is pointed at
it).  One decision, so the two halves cannot contradict.

**These cases drive that draw and nothing else.**  The labeller does not run: its
output is supplied as a fixture, so the only live variable is the judgment under
test.  Driving both handed this decision a different input every sample — three
different names for the same value across five samples, one of them mangled by a
labeller parse slip — and a miss could not be attributed to the draw being iterated
on.  Wiring the two together is worth doing as a separate, broader integration case;
it is not what a scoped contract should be.

Synthetic means the VALUES are authored.  The content is rendered by
``build_shape_content`` and the call is made by ``MicroContext.shape_skill`` — the
shipped code and the shipped prompt, never a copy that can drift.

All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import pytest

from penny.skill_extraction import ShapeableValue
from penny.tests.eval.conftest import ShapeEval

pytestmark = pytest.mark.eval

_FAMILY = "skill-shape"

_LISTING = "https://faux-market.example/aurora-deck-2"

# The instigating ask carries the INTENT — the only place the round says what the user
# was actually after, and so the only evidence for what the routine is ABOUT rather
# than merely pointed at.
_ASK = "can you keep an eye on the aurora deck 2 price for me?"
_DEMONSTRATION = f"read {_LISTING}, find the current price, and remember it"

# The labeller's one-line summary of the round, verbatim from a measured run.  Its
# ROUTINE description is passed to the shaper where its PER-VALUE descriptions are not:
# this one states what the round was FOR, which is the question under test.
_SUMMARY = (
    "Keep track of a specific item's current price by fetching its page and storing the value."
)

# The semantic names the labeller emitted for this round in the same measured run —
# copied rather than invented, so the fixture is what the upstream draw really produces.
# Its one-line descriptions are deliberately absent: they describe every value as a
# fill-in slot, which argued the decision under test (see ``ShapeableValue``).
_URL = ShapeableValue(name="url", current="queries", demonstrated=_LISTING)
_WHAT_TO_FIND = ShapeableValue(
    name="what_to_find", current="extract", demonstrated="the current price"
)


@pytest.mark.asyncio
async def test_the_value_the_routine_is_about_becomes_a_constant(shape_eval: ShapeEval):
    """A skill must not name itself for a value it then asks the user to supply.

    The page VARIES between uses and stays a parameter; what the routine is FOR is
    baked in.

    **The scored direction is one of two coherent answers, and it is the one this ask
    asks for.**  This round could honestly become a price watcher pointed at a page
    (the price a constant) OR a pull-anything-off-a-page routine (both values
    parameters), and the draw is free to write either — what it may never do is commit
    to one in the name and the other in the parameters.  Scoring the first reads the
    user's stated intent as the tiebreak: they said *the price*.  Report-only until
    that reading is confirmed against a real run — a scorer encoding the wrong intent
    would fail the model for being right, which is the failure this suite exists to
    avoid."""
    await shape_eval(
        case_id="shape-value-the-routine-is-about-is-constant",
        values=[_URL, _WHAT_TO_FIND],
        constants=["what_to_find"],
        round_summary=_SUMMARY,
        conversation=[_ASK, _DEMONSTRATION],
        min_pass_rate=None,
        family=_FAMILY,
    )
