"""Live-model contract for the run-end SKILL FRAMER (#1824), in isolation.

The framer writes a skill's public signature — what it is called, what it is for, and
what someone has to say to set it up again — from the user's own messages and nothing
else.  These cases drive that draw and nothing else: no routine, no values, no labeller
output reaches it, because none of that reaches it in production either.

**The question these cases pin.**  A demonstrated round hands over several pieces of
information, and the pipeline this replaces asked, once per leaf, whether the USER
supplied that value.  Measured across three independent wordings, that verdict pinned at
~0.7-0.8 (#1821/#1823): a reworded extract instruction and a storage key slugged from
the user's own URL are both their words re-worded by the assistant.  What separates them
is whether the THING the value is for was asked for — a question about the round, asked
ONCE, at the interface.  So it is asked here, positively: given what this skill IS (the
name and description this same draw writes), which of the pieces they handed over must
they say again?

**Each case sweeps a POOL of asks in the journey register** (sample i →
``pool[i % len(pool)]``) — a marketplace listing, a ferry timetable, a bakery's
specials, a trail report — so N samples sweep how the question is PUT rather than
re-rolling one wording.

**The contract is EXACT** (the code owner's ruling): a case names the families the
signature must ask for, and NOTHING ELSE may be asked.  A skill that asks for the page
AND what to look for is wrong the same way whether the extra parameter is the task, a
destination, or something nobody named — the framing was supposed to carry it.  Scoring
reads the drawn parameter NAMES first and their DESCRIPTIONS second, as token families,
never as expected strings; the framing's own name and description ride along advisory.

All content is synthetic (faux-market / harbour-ferry / corner-bakery / ridge-trails).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import FramingEval, ParameterFamily

pytestmark = pytest.mark.eval

_FAMILY = "skill-framing"

# The particulars of the occasions these asks name.  A framing that carries any of them
# has named the instance rather than the kind of task.
_INSTANCE = ("aurora", "deck", "faux-market", "499", "harbour", "ferry", "bakery", "summit")

# The one piece a journey-register ask always leaves open: the page they pointed at.
# Wide, because a parameter for it is as likely to be named for the thing as for its
# address — either way it is the same single piece.
_PAGE = ParameterFamily(
    "the page they named",
    ("url", "page", "link", "address", "site", "listing", "uri", "webpage", "source"),
)

# Where a result goes, when the ask actually named somewhere.  Wide for the same
# reason: a destination parameter is as likely to be named for the place the user
# called it as for the idea of a place.
_WHERE_TO_PUT_IT = ParameterFamily(
    "where to put it",
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
        "bucket",
    ),
)


# ── Case 1: the floor case — the page is the only thing left to say ────────────


@pytest.mark.asyncio
async def test_the_page_is_the_only_parameter(framing_eval: FramingEval):
    """The floor case, in both directions at once: the parameter set is EXACTLY the
    page.

    They named a page and they named what to look for on it.  The page VARIES between
    uses, so it has to be said again.  What to look for is the POINT of the ask, so the
    skill's own name and description carry it, and asking for it again would be asking
    them to say what they came for.  Nobody said where to keep the result, so nothing
    about a destination belongs in the signature either — and the exactness check is
    what makes both of those one contract rather than a list of things to avoid.

    Report-only until the reading is confirmed against a real run: a scorer encoding the
    wrong intent would fail the model for being right, which is the failure this suite
    exists to avoid."""
    await framing_eval(
        case_id="framing-floor-case-page-is-the-only-parameter",
        pool=[
            [
                "can you keep an eye on the aurora deck 2 price for me?",
                "yeah go to https://faux-market.example/aurora-deck-2, find the price, "
                "and remember it",
            ],
            [
                "have a look at https://harbour-ferry.example/timetable, find the time of "
                "the first sailing, and remember it"
            ],
            [
                "i keep missing the good soup",
                "check https://corner-bakery.example/specials for what the soup of the day "
                "is and keep track of it",
            ],
            ["pull up https://ridge-trails.example/summit-loop and save whether the trail's open"],
        ],
        expected=[_PAGE],
        instance=_INSTANCE,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 2: a two-piece ask — naming two things to find changes nothing ────────


@pytest.mark.asyncio
async def test_a_two_piece_ask_still_asks_only_for_the_page(framing_eval: FramingEval):
    """The user names a page and TWO things to find on it — and the parameter set is
    still exactly the page.

    Naming two things rather than one does not make either of them something the user
    would have to re-supply: both are what the skill is FOR, so both belong in the
    framing, and the signature that asks for them has made the user restate the point of
    their own request twice over.  This is the case that catches a framer treating
    "several things" as a reason to become generic."""
    await framing_eval(
        case_id="framing-two-piece-ask-still-only-the-page",
        pool=[
            [
                "watch the price and the seller rating on "
                "https://faux-market.example/aurora-deck-2 for me"
            ],
            [
                "on https://harbour-ferry.example/timetable keep an eye on the first "
                "sailing and the last sailing"
            ],
            [
                "check https://corner-bakery.example/specials for the soup of the day and "
                "the bread of the day, remember both"
            ],
            [
                "look at https://ridge-trails.example/summit-loop and track the trail "
                "status and the snow depth"
            ],
        ],
        expected=[_PAGE],
        instance=_INSTANCE,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 3: two destinations the user named — two more parameters ──────────────


@pytest.mark.asyncio
async def test_two_named_destinations_are_two_parameters(framing_eval: FramingEval):
    """The user names two different places to put two different results.

    A framing carries what the skill IS — pulling two facts off a page and filing them —
    but it cannot carry WHICH two places, and there are two of them, so the signature has
    to ask for both.  One destination parameter is as wrong as none: the skill would
    silently file two results into one place, which is not what was asked for.  This is
    the case that would break a signature keyed to "the destination" rather than to what
    the ask named, and it is the direction that keeps user-named destinations reachable
    at all."""
    await framing_eval(
        case_id="framing-two-destinations-are-two-parameters",
        pool=[
            [
                "go to https://faux-market.example/aurora-deck-2, put the price in my "
                "price-log collection and the seller rating in my seller-notes collection"
            ],
            [
                "read https://harbour-ferry.example/timetable — first sailing goes in my "
                "morning-runs collection, last sailing in my evening-runs collection"
            ],
            [
                "check https://corner-bakery.example/specials, save the soup to my "
                "soup-log collection and the bread to my bread-log collection"
            ],
            [
                "pull https://ridge-trails.example/summit-loop, trail status into my "
                "trail-status collection and snow depth into my snow-depth collection"
            ],
        ],
        expected=[_PAGE, _WHERE_TO_PUT_IT._replace(count=2)],
        instance=_INSTANCE,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
