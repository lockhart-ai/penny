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

**The floor case sweeps the code owner's ruled fixture set** (sample i →
``pool[i % len(pool)]``) — eight teach turns across everyday domains, each handing over
a url and one thing to find, so N samples sweep how the question is PUT rather than
re-rolling one wording.  They are the SAME eight the leaf module's rounds are built
from, so a finding on one draw is readable against the other.

**The contract is EXACT** (the code owner's ruling): a case names the families the
signature must ask for, and NOTHING ELSE may be asked.  A skill that asks for the page
AND what to look for is wrong the same way whether the extra parameter is the task, a
destination, or something nobody named — the framing was supposed to carry it.  Scoring
reads the drawn parameter NAMES first and their DESCRIPTIONS second, as token families,
never as expected strings; the framing's own name and description ride along advisory.

**Both cases are journey-register, and that is a standing rule, not a coincidence.**  An
isolated eval mirrors the journey's distribution — so a pool shape the journey never
produces does not belong here however interesting the judgment it probes.  A case where
the user explicitly names collections to file results in ("save the soup to my soup-log
collection") was removed for exactly that reason; what it covered — a user-named
destination staying a parameter (#1783) — is recorded as an open gap on #1824 and comes
back when the journey grows a destination-naming beat, in that beat's real phrasing.

All content is synthetic (faux-market / harbour-ferry / corner-bakery / ridge-trails).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import FramingEval, ParameterFamily

pytestmark = pytest.mark.eval

_FAMILY = "skill-framing"

# The PARTICULARS of the occasions these asks name — a framing carrying any of them has
# named the instance instead of the kind of task.
#
# KIND-words are deliberately absent, and the distinction is the whole point (the code
# owner's ruling after they cost nine samples in ``run-20260802T144944Z``): a skill that
# watches ferry timetables SHOULD say "ferry", and one that reads a bakery's specials
# SHOULD say "bakery" — that is what it IS.  What it must never say is which ferry,
# which bakery, which listing.  So the operator, the slug and the demonstrated value are
# instance; the noun for the thing is not.
INSTANCE_PARTICULARS = (
    "aurora",
    "deck",
    "faux-market",
    "499",
    "harbour",
    "summit",
    "corner-bakery",
    "ridge-trails",
    "harborseals",
    "bay-tides",
    "town-library",
    "birding-club",
)

# The one piece a journey-register ask always leaves open: the page they pointed at.
# Wide, because a parameter for it is as likely to be named for the thing as for its
# address — either way it is the same single piece.
_PAGE = ParameterFamily(
    "the page they named",
    ("url", "page", "link", "address", "site", "listing", "uri", "webpage", "source"),
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

    The pool is the code owner's ruled fixture set: eight teach turns, each providing a
    url + one thing to find + "remember it", and each a SINGLE turn — the teach turn is
    the whole learn-step input.  No instigating turn NAMES the subject, because the
    journey's own instigating ask is deictic ("watch this auction for me") and the
    subject's name exists only on the page; a subject-naming turn produced 4 of 6 misses
    in `run-20260802T161131Z` and was non-journey phrasing inherited from the #1803
    shaper fixture.  "remember it" contributes nothing to the signature.

    Report-only until the reading is confirmed against a real run: a scorer encoding the
    wrong intent would fail the model for being right, which is the failure this suite
    exists to avoid."""
    await framing_eval(
        case_id="framing-floor-case-page-is-the-only-parameter",
        pool=[
            [
                "read https://harbour-ferry.example/timetable, find the time of the "
                "first sailing, and remember it"
            ],
            [
                "open https://corner-bakery.example/specials, look for the soup of the "
                "day, and remember it"
            ],
            [
                "check https://ridge-trails.example/summit-loop and remember whether "
                "the trail's open"
            ],
            [
                "pull up https://harborseals.example/colony-count, find the colony "
                "count, and remember it"
            ],
            [
                "sure — go to https://bay-tides.example/table, find this morning's low "
                "tide time, and remember it"
            ],
            [
                "it's easy: read https://town-library.example/new-arrivals, find the "
                "newest mystery title, and remember it"
            ],
            [
                "ok: 1. open https://birding-club.example/sightings 2. find the latest "
                "sighting 3. remember it"
            ],
            ["grab the price off https://faux-market.example/aurora-deck-2 and remember it"],
        ],
        expected=[_PAGE],
        instance=INSTANCE_PARTICULARS,
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
        instance=INSTANCE_PARTICULARS,
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
