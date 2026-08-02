"""Live-model contract for the run-end SKILL FRAMER (#1824), in isolation.

The framer writes a skill's public signature — what it is called, what it is for, and
what someone has to say to set it up again — from the user's own messages and nothing
else.  These cases drive that draw and nothing else: no routine, no values, no
labeller output reaches it, because none of that reaches it in production either.

**The question these cases pin.**  A demonstrated round hands over several pieces of
information, and the pipeline this replaces asked, once per leaf, whether the USER
supplied that value.  Measured across three independent wordings, that verdict pinned
at ~0.7-0.8 and would not move (#1821/#1823): a reworded extract instruction and a
storage key slugged from the user's own URL are both their words re-worded by the
assistant.  What separates them is whether the THING the value is for was asked for —
which is a question about the round, asked ONCE, at the interface.  So it is asked
here, positively: given what this skill IS (the name and description this same draw
writes), which of the pieces they handed over must they say again?

Scoring reads the drawn parameter NAMES — the binding keys, the part of a framing a
user actually has to type — as token families, never as expected strings, and the
framing's own name/description ride along advisory.  What the drawn wording says about
a skill's quality is a reading no scorer should fake; what a machine can hold is which
pieces the signature asks for, and whether it generalizes past the occasion.

All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import FramingEval, ParameterFamily

pytestmark = pytest.mark.eval

_FAMILY = "skill-framing"

_LISTING = "https://faux-market.example/aurora-deck-2"

# The particulars of THIS occasion.  A framing that carries any of them has named the
# instance rather than the kind of task.
_INSTANCE = ("aurora", "deck", "faux-market", "499")

# The pieces a framing case reasons about, as the words a parameter for each would use
# in its name.  Order matters where two could match one name: the page family is
# listed first, so a parameter named for the page it fetches counts as the page even
# when its name also mentions what is on that page.
_PAGE = ParameterFamily("the page", ("url", "page", "link", "address", "site", "listing", "uri"))
_WHAT_TO_FIND = ParameterFamily(
    "what to look for", ("price", "extract", "find", "information", "info", "detail", "details")
)
_WHERE_TO_PUT_IT = ParameterFamily(
    "where to put it",
    # Wide on purpose: a destination parameter is as likely to be named for the place
    # the user called it as for the idea of a place, and both are the same answer.
    (
        "collection",
        "memory",
        "store",
        "storage",
        "destination",
        "folder",
        "log",
        "notes",
        "where",
        "place",
        "file",
        "save",
        "target",
    ),
)
_RATING = ParameterFamily("the thing to watch", ("rating", "seller", "score", "review", "stars"))


# ── Case 1: the floor case — the page is asked for, the task is not ────────────

_FLOOR_ASK = [
    "can you keep an eye on the aurora deck 2 price for me?",
    f"yeah go to {_LISTING}, find the price, and remember it",
]


@pytest.mark.asyncio
async def test_the_page_is_a_parameter_and_the_task_is_not(framing_eval: FramingEval):
    """The floor case, in both directions at once.

    They named a page and they named what to look for on it.  The page VARIES between
    uses, so it has to be said again — a parameter.  What to look for is the point of
    the ask, so the skill's own name and description carry it, and asking for it again
    would be asking them to say what they came for.  And nobody said where to keep the
    result, so nothing about a destination belongs in the signature at all.

    Report-only until the reading is confirmed against a real run: a scorer encoding
    the wrong intent would fail the model for being right, which is the failure this
    suite exists to avoid."""
    await framing_eval(
        case_id="framing-floor-case-page-is-the-parameter",
        conversation=_FLOOR_ASK,
        expected=[_PAGE],
        absent=[_WHAT_TO_FIND, _WHERE_TO_PUT_IT],
        instance=_INSTANCE,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 2: a two-piece ask — the page plus a particular thing to watch ────────


@pytest.mark.asyncio
async def test_a_two_piece_ask_keeps_both_pieces(framing_eval: FramingEval):
    """The user names a page AND a specific thing to watch on it.

    The page is a parameter for the same reason as the floor case.  The thing to watch
    has TWO coherent framings — a skill that watches seller ratings (the framing carries
    it) or one pointed at whatever the user names (it is a parameter) — and the draw is
    free to write either, so scoring one would be scoring a preference.  What it may
    never do is drop the piece entirely, leaving a signature that neither says what it
    watches nor asks: that is the one direction this case calls wrong."""
    await framing_eval(
        case_id="framing-two-piece-ask-keeps-both",
        conversation=[
            "there's something on that marketplace listing i want to track",
            f"watch the seller rating on {_LISTING} and tell me if it drops",
        ],
        expected=[_PAGE],
        carried=[_RATING],
        instance=_INSTANCE,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 3: two destinations the user named — two parameters ───────────────────


@pytest.mark.asyncio
async def test_two_named_destinations_are_two_parameters(framing_eval: FramingEval):
    """The user names two different places to put two different results.

    A framing carries what the skill IS — pulling two facts off a listing and filing
    them — but it cannot carry WHICH two places, and there are two of them, so the
    signature has to ask for both.  One destination parameter is as wrong as none: the
    skill would silently file two results into one place, which is not what was asked
    for.  This is the case that would break a signature keyed to 'the destination'
    rather than to what the ask named."""
    await framing_eval(
        case_id="framing-two-destinations-are-two-parameters",
        conversation=[
            f"go to {_LISTING}, put the price in my price-log collection and the "
            "seller rating in my seller-notes collection",
        ],
        expected=[_PAGE, _WHERE_TO_PUT_IT._replace(count=2)],
        instance=_INSTANCE,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
