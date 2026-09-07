"""Live-model contract for the skill BINDER (#1867, beat 1 of #1866).

The framer writes a routine's interface once, from the round that taught it.  The binder
runs every time that routine is asked for again: given the signature exactly as it
already stands, and the user's own words of THIS round, it says what each declared
parameter's value is.

It mints nothing and it judges nothing.  The parameter set is an INPUT, so the only
decision in the draw is which part of the user's words fills each declared parameter —
and that makes the whole answer checkable in Python before it is ever scored: production
refuses a value that is not a literal span of what the user said, and refuses a draw that
answers for a parameter nobody declared or leaves one unanswered.  What these cases
measure is what is left after that: whether it picked the RIGHT span, and whether it knew
when to decline.

Seven cases score each declared parameter on its own, both directions of the contract:

* ``bind-listing-page`` — one url parameter, the ask names the page.
* ``bind-two-parameters`` — the page AND what to look for on it, out of one message.
* ``bind-daily-special`` — one url, the ask states its cadence in the same breath.
* ``bind-count-page`` — one url under a threshold ask.
* ``bind-new-arrivals`` — one url under an act-now ask with an end date in it.
* ``bind-missing-page`` — the SHORTFALL: an ask that describes the job and names no page.
* ``bind-missing-keyword`` — the shortfall beside a successful bind: the page is there,
  what to look for on it is not.

Seven more are the slot's CANONICAL set (#2006/#2057), each one ask in five wordings
pooled into a cohort of fifteen and claimed against ``docs/eval-case-design.md`` rather
than the per-parameter scorer below.  Together they are the binder's whole decision space
— {complete bind, shortfall} × {single slot, multi slot}, plus the completion draw:

* ``binder-binds-the-page-and-lets-neither-term-in``
* ``binder-invents-nothing-for-a-condition-the-signature-cannot-hold``
* ``binder-takes-two-different-spans-for-two-parameters``
* ``binder-tells-two-same-kinded-slots-apart``
* ``binder-reports-the-only-parameter-missing-rather-than-binding-a-near-value``
* ``binder-fills-one-and-names-the-other-missing``
* ``binder-fills-the-still-open-parameter-when-the-value-arrives``

Since #1894 the binder is the ONE door for every entry against a routine the registry
already holds — a cold apply and a request the classifier drew directly both come through
it — and a round coming back for a missing detail hands over what it already SETTLED, so
only the still-open parameters are drawn.  Every case but the last drives the COLD shape:
the whole declared set, nothing settled, which is the ask the idle→apply and idle→request
beats measure — a second ask pointing a routine Penny already knows at a new space.  Each
one carries its job's TERMS as well (every hour until sunday, each day, every two hours
until friday), which is the second thing every case checks: terms are settled where the
job is set running, so a term inside a bound value is the draw reading them as part of the
thing to point at.  ``binder-fills-the-still-open-parameter-when-the-value-arrives`` is
the WARM shape — a signature offering only what the parked round left open, over the two
turns the round has now heard.

Each case's input is rendered by the shipped ``render_spoken_turns`` +
``build_binding_content`` — never hand-written — so the draw reads exactly what production
would render.  ``rendered_input`` is that document, pinned byte-for-byte by a
deterministic drift probe in ``make check`` (see ``tests/test_eval_harness.py``): a
fixture that drifts from the pair it claims is a case measuring nothing, and it must fail
before any GPU time, not after.

Every case is report-only; the thresholds are the code owner's once the first numbers are
read.  All content is synthetic, and the pages it names are the ones the transition suites
already use.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.database.skills import SkillParameter
from penny.tests.eval.conftest import (
    BIND_MISSING,
    BIND_OUTCOME,
    EVAL_MODELS,
    BinderEval,
    BindOutcome,
    BoundExpectation,
    bound_value_field,
)
from penny.tests.eval.utils.assertions import Answer, Cohort, WorldClaim
from penny.tests.eval.utils.cohort import (
    FIELD_UNSET,
    Consequence,
    Feature,
    SampleObservation,
    SpecCategory,
    output_field,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.micro_context import spoken_form

pytestmark = pytest.mark.eval

_FAMILY = "skill-binding"

# How every behaviour sentence in this file opens.  The locus is the SHIPPED agent name,
# written once: seven cases naming the same micro-context by hand is seven chances for one
# of them to name it something the report cannot be grouped by.
_LOCUS = f"In the {PennyConstants.SKILL_BIND_AGENT_NAME} micro-context, "


class BindingFixture(NamedTuple):
    """One agreed case: the routine as it already stands, the user's turns asking for it
    again, the document the shipped renderers must produce from the pair, what each
    declared parameter should come back as, and the job terms the ask carries — none of
    which may appear inside a value."""

    case_id: str
    skill: str
    intent: str
    parameters: tuple[SkillParameter, ...]
    turns: tuple[str, ...]
    rendered_input: str
    expectations: tuple[BoundExpectation, ...]
    forbidden: tuple[str, ...]


async def _run_case(binder_eval: BinderEval, fixture: BindingFixture) -> None:
    """Drive one case's signature + turns through the binder.  Every case is report-only:
    the thresholds are the code owner's to set once the first numbers are read."""
    await binder_eval(
        case_id=fixture.case_id,
        turns=fixture.turns,
        skill=fixture.skill,
        intent=fixture.intent,
        parameters=fixture.parameters,
        expectations=fixture.expectations,
        forbidden=fixture.forbidden,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


# The two signatures the cases are drawn against, each one the shape the framer really
# produces for that kind of routine (the transitions suite seeds the same pair): one
# parameter for a routine whose framing already says what it is looking for, and two when
# the thing to look for is its own piece.
_PRICE_PARAMETERS = (SkillParameter(name="url", description="The URL of the listing to watch"),)
_TIMETABLE_PARAMETERS = (
    SkillParameter(name="url", description="the URL of the timetable page to fetch"),
    SkillParameter(name="keyword", description="text indicating which timetable entry to look for"),
)


# ── Case 1: one url, and the ask names the page ───────────────────────────────
#
# Reference values (read at review, never matched):
#   url = https://faux-market.example/keel-lantern

_LISTING = BindingFixture(
    case_id="bind-listing-page",
    skill="monitor_price",
    intent="Monitors a web listing and reports when its price changes.",
    parameters=_PRICE_PARAMETERS,
    turns=(
        "can you watch this listing for me and let me know when the price changes? "
        "https://faux-market.example/keel-lantern — every hour until sunday night is fine",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: monitor_price\n"
        "what it is for: Monitors a web listing and reports when its price changes.\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: The URL of the listing to watch\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you watch this listing for me and let me know when the price changes? "
        "https://faux-market.example/keel-lantern — every hour until sunday night is fine"
    ),
    expectations=(BoundExpectation("url", "faux-market.example/keel-lantern"),),
    forbidden=("every hour", "sunday"),
)


@pytest.mark.asyncio
async def test_the_page_in_the_ask_fills_the_one_parameter(binder_eval: BinderEval) -> None:
    """The simplest shape there is: one declared parameter, one address in the message.

    The whole ask is one turn and carries its cadence and its end in the same breath — so
    what is measured beside the bind is restraint, because "every hour until sunday night"
    sits directly beside the url the value has to be."""
    await _run_case(binder_eval, _LISTING)


# ── Case 2: two parameters, both out of one message ───────────────────────────
#
# Reference values (read at review, never matched):
#   url     = https://northpier.example/departures
#   keyword = dawn sailing

_TWO_PARAMETERS = BindingFixture(
    case_id="bind-two-parameters",
    skill="check_ferry_timetable",
    intent="Check a ferry timetable page for updates and report the status of a specified line",
    parameters=_TIMETABLE_PARAMETERS,
    turns=(
        "every morning can you check the north pier timetable at "
        "https://northpier.example/departures and let me know when they add the dawn sailing?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: check_ferry_timetable\n"
        "what it is for: Check a ferry timetable page for updates and report the status "
        "of a specified line\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the timetable page to fetch\n"
        "- keyword: text indicating which timetable entry to look for\n"
        "\n"
        "What the user said, in their own words:\n"
        "every morning can you check the north pier timetable at "
        "https://northpier.example/departures and let me know when they add the dawn sailing?"
    ),
    expectations=(
        BoundExpectation("url", "northpier.example/departures"),
        BoundExpectation("keyword", "dawn sailing"),
    ),
    forbidden=("every morning",),
)


@pytest.mark.asyncio
async def test_two_declared_parameters_take_two_different_spans(binder_eval: BinderEval) -> None:
    """The stress case for filling a signature: one message supplies BOTH the page and the
    thing to look for on it, and they are different kinds of value in the same sentence.

    A binder that reads the page for both, or the phrase for both, has bound a routine
    that will read the right page for the wrong thing — which is why each parameter is its
    own check rather than a count."""
    await _run_case(binder_eval, _TWO_PARAMETERS)


# ── Case 3: one url, the cadence stated as part of the sentence ───────────────
#
# Reference values (read at review, never matched):
#   url = https://harborbakery.example/menu

_DAILY_SPECIAL = BindingFixture(
    case_id="bind-daily-special",
    skill="fetch_daily_special",
    intent="retrieve the daily special from a bakery webpage",
    parameters=(
        SkillParameter(name="url", description="the URL where the daily specials are listed"),
    ),
    turns=(
        "can you get the daily special from https://harborbakery.example/menu each day "
        "and tell me what it is?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: fetch_daily_special\n"
        "what it is for: retrieve the daily special from a bakery webpage\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL where the daily specials are listed\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you get the daily special from https://harborbakery.example/menu each day "
        "and tell me what it is?"
    ),
    expectations=(BoundExpectation("url", "harborbakery.example/menu"),),
    forbidden=("each day",),
)


@pytest.mark.asyncio
async def test_the_cadence_in_the_sentence_is_not_part_of_the_value(
    binder_eval: BinderEval,
) -> None:
    """ "each day" sits between the url and the rest of the sentence, so the value and the
    term are neighbours in the text.

    They are settled in different places — the value points the routine, the cadence is
    set when the job is stood up — so a bind that swept the cadence in has made the
    routine's identity depend on how often it runs."""
    await _run_case(binder_eval, _DAILY_SPECIAL)


# ── Case 4: one url under a threshold ask ─────────────────────────────────────
#
# Reference values (read at review, never matched):
#   url = https://riverotters.example/census

_COUNT = BindingFixture(
    case_id="bind-count-page",
    skill="monitor_webpage_number",
    intent="track a numeric value on a webpage over time to detect changes",
    parameters=(SkillParameter(name="url", description="the webpage to monitor"),),
    turns=(
        "keep track of the otter count at https://riverotters.example/census every week "
        "and let me know if it drops",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: monitor_webpage_number\n"
        "what it is for: track a numeric value on a webpage over time to detect changes\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the webpage to monitor\n"
        "\n"
        "What the user said, in their own words:\n"
        "keep track of the otter count at https://riverotters.example/census every week "
        "and let me know if it drops"
    ),
    expectations=(BoundExpectation("url", "riverotters.example/census"),),
    forbidden=("every week",),
)


@pytest.mark.asyncio
async def test_a_threshold_ask_still_binds_only_the_page(binder_eval: BinderEval) -> None:
    """The ask carries a condition — tell me if it drops — and the routine declares one
    parameter, the page.

    So the condition has nowhere to go, and a signature with nowhere to put something is
    exactly where an invented parameter or a padded value would show up."""
    await _run_case(binder_eval, _COUNT)


# ── Case 5: one url under an act-now ask with an end date in it ───────────────
#
# Reference values (read at review, never matched):
#   url = https://eastbranch.example/new-titles

_NEW_ARRIVALS = BindingFixture(
    case_id="bind-new-arrivals",
    skill="retrieve_newest_item",
    intent="Checks a web page and returns its newest arrival",
    parameters=(SkillParameter(name="url", description="the URL of the list to check"),),
    turns=(
        "watch https://eastbranch.example/new-titles every two hours until friday and "
        "tell me when something new shows up",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: retrieve_newest_item\n"
        "what it is for: Checks a web page and returns its newest arrival\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the list to check\n"
        "\n"
        "What the user said, in their own words:\n"
        "watch https://eastbranch.example/new-titles every two hours until friday and "
        "tell me when something new shows up"
    ),
    expectations=(BoundExpectation("url", "eastbranch.example/new-titles"),),
    forbidden=("every two hours", "friday"),
)


@pytest.mark.asyncio
async def test_the_url_opens_the_ask_and_the_terms_follow_it(binder_eval: BinderEval) -> None:
    """The address is the FIRST thing in the message and both terms follow it, which is the
    layout most likely to produce a value that runs on past its end.

    A cadence and an end date immediately after the url is where "watch <url> every two
    hours until friday" becomes one long value if the draw takes the rest of the
    clause."""
    await _run_case(binder_eval, _NEW_ARRIVALS)


# ── Case 6: the shortfall — the job is described and no page is named ─────────
#
# Reference values (read at review, never matched):
#   url = MISSING

_MISSING_PAGE = BindingFixture(
    case_id="bind-missing-page",
    skill="monitor_price",
    intent="Monitors a web listing and reports when its price changes.",
    parameters=_PRICE_PARAMETERS,
    turns=(
        "can you keep an eye on the price of that brass lantern i was looking at and "
        "tell me when it changes? every hour is fine",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: monitor_price\n"
        "what it is for: Monitors a web listing and reports when its price changes.\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: The URL of the listing to watch\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you keep an eye on the price of that brass lantern i was looking at and "
        "tell me when it changes? every hour is fine"
    ),
    expectations=(BoundExpectation("url"),),
    forbidden=("every hour",),
)


@pytest.mark.asyncio
async def test_an_ask_that_names_no_page_reports_the_page_missing(binder_eval: BinderEval) -> None:
    """The ask is a perfectly good description of the job and supplies nothing to point it
    at: the user refers to a listing they were looking at and never says which.

    The temptation is a value that is right there in the sentence and is not a page —
    "that brass lantern" reads like an answer, and a routine bound to it would go and
    watch nothing.  Naming the parameter missing is the answer the contract asks for, and
    since #1885 it is what ROUTES the turn into request — an enumerated outcome the
    machine acts on, never a failed draw."""
    await _run_case(binder_eval, _MISSING_PAGE)


# ── Case 7: the shortfall beside a successful bind ────────────────────────────
#
# Reference values (read at review, never matched):
#   url     = https://northpier.example/departures
#   keyword = MISSING

_MISSING_KEYWORD = BindingFixture(
    case_id="bind-missing-keyword",
    skill="check_ferry_timetable",
    intent="Check a ferry timetable page for updates and report the status of a specified line",
    parameters=_TIMETABLE_PARAMETERS,
    turns=(
        "can you check the timetable at https://northpier.example/departures every "
        "morning and keep me posted?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: check_ferry_timetable\n"
        "what it is for: Check a ferry timetable page for updates and report the status "
        "of a specified line\n"
        "\n"
        "What it needs, one line each:\n"
        "- url: the URL of the timetable page to fetch\n"
        "- keyword: text indicating which timetable entry to look for\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you check the timetable at https://northpier.example/departures every "
        "morning and keep me posted?"
    ),
    expectations=(
        BoundExpectation("url", "northpier.example/departures"),
        BoundExpectation("keyword"),
    ),
    forbidden=("every morning",),
)


@pytest.mark.asyncio
async def test_one_parameter_binds_while_the_other_is_reported_missing(
    binder_eval: BinderEval,
) -> None:
    """The two directions in one draw: the page is in the message and what to look for on
    it is not.

    This is the shape the request state exists for — enough of the ask has landed to be
    worth keeping, and one named thing is outstanding — so the answer has to carry both
    halves: the missing parameter named, and the bound one not thrown away on the way to
    reporting it.  Since #1894 that surviving half becomes the round's own state, handed
    back to the next draw as its settled values, so a page given now is never asked for
    again."""
    await _run_case(binder_eval, _MISSING_KEYWORD)


# ── The ported cases: the binder's whole decision space (#2006/#2057) ─────────
#
# Seven cohort cases, each one ask in five wordings pooled into fifteen samples and claimed
# against ``docs/eval-case-design.md``.  Together they cover {complete bind, shortfall} ×
# {single slot, multi slot}, plus the completion draw.  Five things they share, stated once
# here rather than seven times over:
#
# STORE is EMPTY, and that is the correct report.  A micro-context is one call returning a
# typed result — it moves no machine and writes to no store — so there is nothing for a
# store claim to read.  What a binding becomes is round state on a later transition, and
# none of these cases runs that.
#
# THREE claims are missing because PRODUCTION already validates them, and a thin set should
# read as closed rather than as unrun.  Two are ``_fills_the_declared_signature``'s: the
# drawn names equal the declared names as a multiset (so no parameter is answered twice,
# left unanswered or invented), and every value is a literal span of what the user said.
# The third is the framework's own arithmetic rather than a validator — nothing is bound for
# a parameter reported missing, because ``_skill_binding`` builds ``values`` from the value
# lines and ``names`` from the rest, so a name in one can never be in the other.
#
# A COMPLETE binding's "it names nothing missing" is the outcome claim said a second way, so
# the cases expecting one make a single LANDED claim.  Those cases also do not MEASURE what
# the draw reported missing: the field is written on every sample and reads the EMPTY STRING
# when nothing is missing, which ``cohort._is_blind``'s first arm reports as blind rather
# than as agreement — the same treatment the missing-keyword case gives the value axis of
# the parameter it expects nobody to bind.
#
# The shared CLAIMS stay in this file rather than graduating into ``assertions.py`` at their
# second customer, and this is the one reason: they read the binder's own output fields
# (``BIND_OUTCOME``, ``bound_value_field``), which the driver declares in ``conftest.py`` —
# and ``conftest`` imports ``assertions``, so the claims cannot move without inverting that.
#
# And the FACTS are constant across each case's five wordings.  Which facts is DERIVED, not
# declared: a case states the terms its claims FORBID, the anchors come off its own
# expectations, and ``PortedBinding.states`` is the union — so a fact the arms carry that no
# claim reads cannot be written down, and a term a claim forbids that the arms do not all
# carry cannot either.  ``make check``'s arms probe holds the rest before any GPU time.

# Five wordings of one ask, per ``docs/eval-case-design.md`` §6.  Named because the arms
# probe and the drivers must agree on it: a case pooling four would be measuring a smaller
# cohort than every recorded ceiling was taken at.
PHRASINGS_PER_COHORT = 5

# The host every synthetic page in this suite lives on.  A case whose ask must name NO page
# forbids this rather than any one address: what makes such an arm wrong is that it supplies
# a page at all, not which page it supplies.
ANY_PAGE_HOST = ".example"


class PortedBinding(NamedTuple):
    """One PORTED case: the sentence it checks, the signature and first wording it is built
    on, the other four wordings, and the terms its claims forbid.

    Declared once so the case body and ``make check``'s arms probe read the SAME arms.  A
    cohort pools five wordings into one number, which is only legal if they differ in their
    WORDS and agree on their FACTS — and a probe over a second copy of the arms would be
    checking a fixture nobody runs.

    ``forbids`` is what this case's *nothing invented* claim rules out of every bound value:
    the job's terms, and — where the ask names something that is not an answer — the
    temptation itself.  It lives HERE rather than on the fixture because the fixture's own
    ``forbidden`` feeds the inline per-parameter scorer, and a cohort claim reading it
    inherits whatever that scorer happened to need.  ``binder-invents-nothing-for-a-condition
    -the-signature-cannot-hold`` is why: its fixture forbids the cadence alone, so a claim
    reading the fixture would score green on a value that swallowed the whole condition
    clause — the exact draw the case is named for.

    ``also_states`` is for a fact the arms must carry that no claim reads directly — the role
    word that says which of two same-kinded pages answers which parameter.  Everything else
    in ``states`` is derived, so a fact nobody measures cannot be declared.

    ``never_states`` are the spans the ARRIVING turn may not carry — the last turn, which for
    a single-turn ask is the whole of it.  A case claiming a parameter is unsupplied must
    declare one, or nothing holds the arms to supplying nothing for it."""

    case_id: str
    behaviour: str
    fixture: BindingFixture
    also_phrased: tuple[tuple[str, ...], ...]
    forbids: tuple[str, ...]
    also_states: tuple[str, ...] = ()
    never_states: tuple[str, ...] = ()

    @property
    def arms(self) -> tuple[tuple[str, ...], ...]:
        """Every wording of this case's ask, the fixture's own first."""
        return (self.fixture.turns, *self.also_phrased)

    @property
    def anchors(self) -> tuple[str, ...]:
        """The span each value claim reads, for the parameters the ask supplies one for."""
        return tuple(one.anchor for one in self.fixture.expectations if one.anchor)

    @property
    def states(self) -> tuple[str, ...]:
        """Every span each arm must carry verbatim — the anchors, the forbidden terms, and
        whatever else makes the ask answerable."""
        return self.anchors + self.forbids + self.also_states


async def _drive_ported(binder_eval: BinderEval, ported: PortedBinding, model: str) -> Cohort:
    """Drive one ported case's five wordings over its one signature.

    The signature is the SAME on every arm — a case whose signature moved would be binding a
    different routine, which is a different behaviour rather than a different wording."""
    return await binder_eval(
        case_id=ported.case_id,
        behaviour=ported.behaviour,
        model=model,
        turns=ported.fixture.turns,
        also_phrased=ported.also_phrased,
        skill=ported.fixture.skill,
        intent=ported.fixture.intent,
        parameters=ported.fixture.parameters,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


def _anchor_for(fixture: BindingFixture, parameter: str) -> str:
    """What the ask supplies for one DECLARED parameter, looked up by NAME.

    By name rather than by position, so a signature edit that reordered the parameters breaks
    here rather than silently swapping which parameter a claim is about.  A name the fixture
    declares nothing for RAISES — a claim about a parameter this signature does not have
    could never be answered, and raising is what stops it reading as fifteen misses instead.

    It raises where the claim is BUILT, which is in the case body: a typo fails that case
    loudly at its first line rather than at ``make check``, and the drift probe's own
    expectations-equal-parameters assertion is what holds the two declarations together."""
    for one in fixture.expectations:
        if one.parameter == parameter:
            return one.anchor
    raise KeyError(f"{fixture.case_id} declares no parameter named {parameter!r}")


def _bound_value_axis(ported: PortedBinding, parameter: str) -> Feature:
    """The measured axis carrying what the draw bound one DECLARED parameter to.

    COSMETIC, because a draw that kept a url's scheme and one that dropped it point the
    routine at the same page.  ``absent=FIELD_UNSET`` because this observer really can OMIT
    the field (#2061): ``_binding_output`` writes a value line only for a parameter the draw
    BOUND, so a parameter it reported missing yields no field at all — the omission is the
    reading that says the draw produced nothing here, and a cohort where no sample filled it
    must read as blind rather than as fifteen samples agreeing.  Declared in ONE place for
    every case, because a per-site declaration is a per-site chance to forget it.

    Routed through :func:`_anchor_for` for its refusal alone: an axis over a parameter the
    signature does not declare would read unset on every sample and render as a blind
    feature, which is a quiet way to measure nothing."""
    _anchor_for(ported.fixture, parameter)
    return output_field(
        bound_value_field(parameter), consequence=Consequence.COSMETIC, absent=FIELD_UNSET
    )


def _bound_every_parameter(sample: SampleObservation, _world: World) -> Answer:
    """The draw came back a COMPLETE binding rather than a shortfall.

    An enumerated outcome, asserted by equality: production answers with ``BoundValues`` when
    every parameter it was offered got a value and ``MissingParameters`` when one did not,
    and which of the two it wrote is the decision these cases are about.  Nothing upstream
    decides it — the draw chooses, per parameter, whether to write a value line or a missing
    line.

    The wrong answer names the interesting failure: a shortfall over an ask that really does
    supply everything is a round parked waiting for a detail the user already gave."""
    outcome = sample.field(BIND_OUTCOME)
    return outcome == BindOutcome.COMPLETE.value, f"came back {outcome}"


def _reported_a_shortfall(sample: SampleObservation, _world: World) -> Answer:
    """The draw came back a SHORTFALL rather than a complete binding.

    The other side of the same enumerated field, and not validated anywhere upstream — the
    draw chooses, per parameter, whether to write a value line or a missing line.

    The wrong answer is the interesting one: a complete binding on an ask that supplies
    nothing for a parameter means the draw filled it from something in the sentence that does
    not answer it, which is a routine that will read the right page for the wrong thing."""
    outcome = sample.field(BIND_OUTCOME)
    return outcome == BindOutcome.SHORTFALL.value, f"came back {outcome}"


def _names_exactly_missing(ported: PortedBinding, parameter: str) -> WorldClaim:
    """It names ``parameter``, and only it, as the thing nothing supplies.

    Equality over a closed set — the parameters the signature DECLARES — so this is the
    smallest datum that identifies the answer: naming another parameter as well would be a
    routine that cannot be pointed anywhere despite being handed an address, and naming
    nothing at all is the outcome claim said a second way.

    Refuses a parameter the ask DOES supply a span for: such a parameter is one the draw
    should bind, so a case claiming it missing has stated a contradiction rather than a
    behaviour."""
    if _anchor_for(ported.fixture, parameter):
        raise ValueError(f"{ported.case_id}: the ask supplies {parameter!r}, so it is bound")

    def claim(sample: SampleObservation, _world: World) -> Answer:
        reported = sample.field(BIND_MISSING)
        return reported == parameter, f"reported {reported!r}"

    return claim


def _binds_the_span_the_ask_supplies(ported: PortedBinding, parameter: str) -> WorldClaim:
    """One parameter carries the span its own ask supplies — the *nothing omitted* half.

    Production validates that a value is A span of the user's words and re-rolls until it is,
    so what is left to measure is whether it is THE span: an ask carrying an address, a
    cadence and an object has several spans a draw could take, and only one of them answers
    this parameter.

    The anchor is read off the fixture's own expectation, by NAME.  It is the SMALLEST datum
    unique in this world — for a page, the host and path without the scheme, because a draw
    that kept ``https://`` and one that dropped it read the ask exactly as well.  Compared
    through the shipped ``spoken_form``, so what a case calls a match and what production
    calls a span are one definition."""
    anchor = _anchor_for(ported.fixture, parameter)

    def claim(sample: SampleObservation, _world: World) -> Answer:
        bound = sample.field(bound_value_field(parameter))
        if bound == FIELD_UNSET:
            return False, "no value came back for it"
        carried = spoken_form(anchor) in spoken_form(bound)
        return carried, f"bound {bound!r}, not the span the ask supplies for it"

    return claim


def _no_value_carries_the_terms(ported: PortedBinding) -> WorldClaim:
    """No bound value carries a term the ask settles somewhere else — the *nothing invented*
    half.

    How often a routine runs, when it stops, and the condition it reports under are settled
    where the job is set running, never by the binder — so one of those words INSIDE a value
    means the draw read it as part of the thing to point the routine at, and the routine's
    identity then depends on how often it runs.  Where the ask names something that is not an
    answer at all, that phrase is forbidden too: binding it is the failure the case exists
    for.  It is a fact of this world rather than a vocabulary rule — the case declares the
    words its own ask carries, exactly as a world declares what must not be kept from a page,
    so nothing here is keyed to a list of timing words in general.

    A draw that bound NOTHING answers this true, and correctly: no bound value carries the
    terms because there are none.  What catches that draw is the case's own LANDED claim,
    which says which of the two enumerated answers was owed."""

    def claim(sample: SampleObservation, _world: World) -> Answer:
        bound = {
            parameter.name: sample.field(bound_value_field(parameter.name))
            for parameter in ported.fixture.parameters
        }
        offenders = [
            f"{name} ({term})"
            for name, value in bound.items()
            if value != FIELD_UNSET
            for term in ported.forbids
            if spoken_form(term) in spoken_form(value)
        ]
        return not offenders, f"carried the terms: {'; '.join(offenders)}"

    return claim


def _no_value_takes_another_parameters_span(ported: PortedBinding) -> WorldClaim:
    """No parameter is bound to the span that answers a DIFFERENT one — the *nothing
    invented* half where a signature declares more than one slot.

    This is the failure the anchor claims cannot see on their own: a draw that took the whole
    clause for both parameters carries the right span in each and has still bound a routine
    that will read the right page for the wrong thing, and one that swapped two same-kinded
    slots has pointed the job at the right pages in the wrong roles.

    A draw that bound nothing answers this true for the reason above, and the LANDED claim is
    what says whether binding nothing was the answer owed."""

    def claim(sample: SampleObservation, _world: World) -> Answer:
        bound = {
            one.parameter: sample.field(bound_value_field(one.parameter))
            for one in ported.fixture.expectations
        }
        offenders = [
            f"{name} ({other.anchor})"
            for name, value in bound.items()
            if value != FIELD_UNSET
            for other in ported.fixture.expectations
            if other.parameter != name and other.anchor
            if spoken_form(other.anchor) in spoken_form(value)
        ]
        return not offenders, f"took another parameter's span: {'; '.join(offenders)}"

    return claim


# ── Ported: one parameter filled, one reported missing ────────────────────────
#
# The survivor is ``bind-missing-keyword``, because it is the one ask that states the whole
# behaviour in a single draw: the page is in the message and what to look for on it is not,
# so the answer has to carry both halves — the value it could read, and the parameter
# nothing supplied — while the cadence sitting beside the address stays out of either.
#
# An arm is a SEQUENCE of turns, matching the framer's shape: an ask is however many turns
# the user took to make it, and ``render_spoken_turns`` renders them as the haystack the
# span check reads.

_MISSING_KEYWORD_PHRASINGS = (
    (
        "every morning could you look at https://northpier.example/departures for me "
        "and let me know?",
    ),
    (
        "would you mind checking https://northpier.example/departures every morning and "
        "telling me what you find?",
    ),
    ("please look at https://northpier.example/departures every morning and keep me in the loop",),
    (
        "i'd like https://northpier.example/departures checked every morning — just let "
        "me know how it looks",
    ),
)

_MISSING_KEYWORD_PORT = PortedBinding(
    case_id="binder-fills-one-and-names-the-other-missing",
    behaviour=(
        f"{_LOCUS}when a routine Penny already knows is pointed at something new and the ask "
        "supplies only part of what it declares, Penny fills each parameter from the span "
        "that supplies it and names the one nothing supplies — taking the job's cadence into "
        "neither."
    ),
    fixture=_MISSING_KEYWORD,
    also_phrased=_MISSING_KEYWORD_PHRASINGS,
    forbids=("every morning",),
    # What an ask that DOES supply the keyword says, read off the sibling case drawn against
    # the same signature — so "no arm names an entry" is anchored to a real one rather than
    # to a phrase spelled here.
    never_states=(_anchor_for(_TWO_PARAMETERS, "keyword"),),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_page_binds_and_the_entry_is_reported_missing(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: check this timetable page every morning and keep me posted.

    The page is in the message and what to look for on it is not, so the answer has to carry
    both halves — the missing parameter named, and the bound one not thrown away on the way
    to reporting it — with the cadence in neither.

    A correct draw binds nothing for the keyword, so that parameter's own value axis is
    deliberately not MEASURED: it would be absent from every sample, which the pooler reports
    as blind rather than as agreement.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _MISSING_KEYWORD_PORT, model)
    # LANDED — the CLOSED fields of the typed result, asserted by equality: which of the two
    # enumerated answers the draw wrote, and which declared parameter it named.
    cohort.claim(
        "state: the draw reported a shortfall rather than a complete binding",
        _reported_a_shortfall,
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: it names the keyword as the one thing nothing supplies",
        _names_exactly_missing(_MISSING_KEYWORD_PORT, "keyword"),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE — the OPEN field, which for this shape is each bound value, read in both
    # directions: the value carries the span of the ask that supplies it, and it carries
    # nothing the parameter does not own.
    cohort.claim(
        "state: the url is bound to the address the ask supplies",
        _binds_the_span_the_ask_supplies(_MISSING_KEYWORD_PORT, "url"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no bound value carries the job's cadence",
        _no_value_carries_the_terms(_MISSING_KEYWORD_PORT),
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — the draw's own structured fields.
    #
    # The OUTCOME and what it reported missing are CONSEQUENTIAL: a complete binding and a
    # shortfall are two different end states, and which parameter is named decides what the
    # turn goes on to ask for.  The bound url is COSMETIC — a draw that kept the scheme and
    # one that dropped it point the routine at the same page.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        output_field(BIND_OUTCOME),
        output_field(BIND_MISSING),
        _bound_value_axis(_MISSING_KEYWORD_PORT, "url"),
    )


# ── Ported: the page binds and neither term comes with it ─────────────────────
#
# The survivor is ``bind-listing-page``, because it is the one ask whose value sits between
# BOTH terms: a cadence and an end date, either of which a draw taking the rest of the
# clause would sweep in.  ``bind-daily-special`` and ``bind-new-arrivals`` collapse into it
# — they differ in where the cadence sits relative to the url, which is layout, which is
# wording, which is the arm axis.

_LISTING_PHRASINGS = (
    (
        "every hour until sunday night, could you check "
        "https://faux-market.example/keel-lantern and tell me if the price moves?",
    ),
    (
        "please keep an eye on the price at https://faux-market.example/keel-lantern "
        "every hour until sunday and let me know when it changes",
    ),
    (
        "i'd like https://faux-market.example/keel-lantern watched every hour until "
        "sunday night — just tell me when the price changes",
    ),
    (
        "would you mind checking the price on https://faux-market.example/keel-lantern "
        "every hour until sunday and pinging me when it moves?",
    ),
)

_LISTING_PORT = PortedBinding(
    case_id="binder-binds-the-page-and-lets-neither-term-in",
    behaviour=(
        f"{_LOCUS}when a routine Penny already knows is asked for again and the ask states "
        "the job's cadence and its end beside the page, Penny binds the page and takes "
        "neither term into the value."
    ),
    fixture=_LISTING,
    also_phrased=_LISTING_PHRASINGS,
    forbids=("every hour", "sunday"),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_page_binds_and_neither_term_comes_with_it(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: watch this listing every hour until sunday.

    The simplest shape there is — one declared parameter, one address in the message — with
    the two things most likely to ride along sitting either side of it.  What is measured
    beside the bind is restraint in both directions at once: a cadence and an end date are
    settled where the job is set running, so either one inside the value is the draw reading
    the terms as part of the thing to point at.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _LISTING_PORT, model)
    # LANDED — the CLOSED field of the typed result, asserted by equality.
    cohort.claim(
        "state: the draw bound every declared parameter",
        _bound_every_parameter,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE — the OPEN field, read in both directions.
    cohort.claim(
        "state: the url is bound to the address the ask supplies",
        _binds_the_span_the_ask_supplies(_LISTING_PORT, "url"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no bound value carries the job's cadence or its end",
        _no_value_carries_the_terms(_LISTING_PORT),
        SpecCategory.PROVENANCE,
    )

    cohort.measure(
        output_field(BIND_OUTCOME),
        _bound_value_axis(_LISTING_PORT, "url"),
    )


# ── Ported: a condition the signature cannot hold ─────────────────────────────
#
# The survivor is ``bind-count-page``: the ask states a condition — tell me if it drops —
# and the routine declares one parameter, the page.  The condition has nowhere to go, and a
# signature with nowhere to put something is exactly where an invented parameter or a padded
# value would show up.  So the condition clause is FORBIDDEN alongside the cadence: a value
# running on to the end of "…/census and tell me if it drops" is a literal span of the ask,
# which production accepts, and it is the draw this case exists to catch.

_COUNT_PHRASINGS = (
    ("every week could you look at https://riverotters.example/census and tell me if it drops?",),
    (
        "please check the otter count on https://riverotters.example/census every week "
        "— i want to hear if it drops",
    ),
    ("i'd like https://riverotters.example/census checked every week, and a heads up if it drops",),
    (
        "would you mind following https://riverotters.example/census every week and "
        "letting me know if it drops?",
    ),
)

_COUNT_PORT = PortedBinding(
    case_id="binder-invents-nothing-for-a-condition-the-signature-cannot-hold",
    behaviour=(
        f"{_LOCUS}when the ask carries a condition the routine's signature has nowhere to "
        "put, Penny binds the page it names and pads the value with none of it."
    ),
    fixture=_COUNT,
    also_phrased=_COUNT_PHRASINGS,
    forbids=("every week", "if it drops"),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_condition_with_nowhere_to_go_pads_no_value(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: keep track of this count every week and tell me if it drops.

    The routine needs a page and nothing else, so the condition the ask carries has nowhere
    in the signature to live.  Production already refuses a draw that answers for a parameter
    nobody declared, so what is left to measure is the other way it could go: the condition
    padded into the value that does exist.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _COUNT_PORT, model)
    # LANDED
    cohort.claim(
        "state: the draw bound every declared parameter",
        _bound_every_parameter,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE
    cohort.claim(
        "state: the url is bound to the page the ask names",
        _binds_the_span_the_ask_supplies(_COUNT_PORT, "url"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no bound value carries the job's cadence or its condition",
        _no_value_carries_the_terms(_COUNT_PORT),
        SpecCategory.PROVENANCE,
    )

    cohort.measure(
        output_field(BIND_OUTCOME),
        _bound_value_axis(_COUNT_PORT, "url"),
    )


# ── Ported: two parameters, two different kinds of span ───────────────────────
#
# The survivor is ``bind-two-parameters``: one message supplies BOTH the page and the thing
# to look for on it, and they are different kinds of value in the same sentence.

_TWO_PARAMETER_PHRASINGS = (
    (
        "could you look at https://northpier.example/departures every morning and tell "
        "me when the dawn sailing shows up?",
    ),
    (
        "please watch https://northpier.example/departures every morning for the dawn "
        "sailing and let me know",
    ),
    (
        "i'd like https://northpier.example/departures checked every morning — tell me "
        "when the dawn sailing appears",
    ),
    (
        "would you mind checking every morning whether "
        "https://northpier.example/departures lists the dawn sailing yet?",
    ),
)

_TWO_PARAMETERS_PORT = PortedBinding(
    case_id="binder-takes-two-different-spans-for-two-parameters",
    behaviour=(
        f"{_LOCUS}when one message supplies both the page and the thing to look for on it, "
        "Penny binds each parameter to the span that answers it and neither to the other's."
    ),
    fixture=_TWO_PARAMETERS,
    also_phrased=_TWO_PARAMETER_PHRASINGS,
    forbids=("every morning",),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_two_parameters_take_the_two_spans_that_answer_them(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: check this timetable every morning for the dawn sailing.

    A binder that reads the page for both, or the phrase for both, has bound a routine that
    will read the right page for the wrong thing — which is why each parameter is its own
    claim and the cross-slot check is a third one: taking the whole clause for both satisfies
    each anchor while binding neither parameter to what answers it.

    The span claims are CONTAINMENT, so an over-long bind that stops short of the other
    parameter's span still satisfies them (#1884 item 2) — a full rate here means each value
    came from the right part of the ask, not that it stops where that part stops.  Closing it
    needs a term forbidden on ONE arm rather than on the case, which the arms contract does
    not have and which is a harness change, not a fixture one.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _TWO_PARAMETERS_PORT, model)
    # LANDED
    cohort.claim(
        "state: the draw bound every declared parameter",
        _bound_every_parameter,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE
    cohort.claim(
        "state: the url is bound to the page the ask names",
        _binds_the_span_the_ask_supplies(_TWO_PARAMETERS_PORT, "url"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the keyword is bound to the entry the ask names",
        _binds_the_span_the_ask_supplies(_TWO_PARAMETERS_PORT, "keyword"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: neither parameter is bound to the span that answers the other",
        _no_value_takes_another_parameters_span(_TWO_PARAMETERS_PORT),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no bound value carries the job's cadence",
        _no_value_carries_the_terms(_TWO_PARAMETERS_PORT),
        SpecCategory.PROVENANCE,
    )

    cohort.measure(
        output_field(BIND_OUTCOME),
        _bound_value_axis(_TWO_PARAMETERS_PORT, "url"),
        _bound_value_axis(_TWO_PARAMETERS_PORT, "keyword"),
    )


# ── Ported: two slots of the SAME kind ────────────────────────────────────────
#
# New (#2057).  Every other multi-slot case declares parameters of different kinds, where
# what a value IS mostly says which slot it answers.  Here both are pages, so nothing about
# the kind tells them apart — only what the ask says about each — and mis-assignment across
# slots is likeliest exactly there.  The addresses carry no hint of the roles either, so the
# assignment can only be read out of the words around them, which is why those words are the
# case's own ``also_states``.
#
# TWO OF THE FIVE ARMS NAME THE RETURN FIRST, and that is what makes the claims mean
# anything: with every arm mentioning the outbound first, a binder that never reads the role
# words at all — taking the addresses in the order they appear and filling the signature in
# its declared order — scores full marks, and the case would report a positional habit as
# comprehension.  Which leg is on which page is IDENTICAL on every arm; only where each is
# mentioned moves, which is layout, which is wording, which is the arm axis.
#
# Reference values (read at review, never matched):
#   outbound_url = https://northpier.example/departures
#   return_url   = https://southquay.example/departures

_CROSSING_PARAMETERS = (
    SkillParameter(
        name="outbound_url", description="the URL of the page listing the outbound crossing"
    ),
    SkillParameter(
        name="return_url", description="the URL of the page listing the return crossing"
    ),
)

_CROSSING = BindingFixture(
    case_id="binder-tells-two-same-kinded-slots-apart",
    skill="compare_crossing_fares",
    intent=(
        "Compares the fares for the outbound and return legs of a crossing and reports "
        "which costs more"
    ),
    parameters=_CROSSING_PARAMETERS,
    turns=(
        "every monday can you check https://northpier.example/departures for the outbound "
        "crossing and https://southquay.example/departures for the return, and tell me "
        "which is cheaper?",
    ),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: compare_crossing_fares\n"
        "what it is for: Compares the fares for the outbound and return legs of a "
        "crossing and reports which costs more\n"
        "\n"
        "What it needs, one line each:\n"
        "- outbound_url: the URL of the page listing the outbound crossing\n"
        "- return_url: the URL of the page listing the return crossing\n"
        "\n"
        "What the user said, in their own words:\n"
        "every monday can you check https://northpier.example/departures for the outbound "
        "crossing and https://southquay.example/departures for the return, and tell me "
        "which is cheaper?"
    ),
    expectations=(
        BoundExpectation("outbound_url", "northpier.example/departures"),
        BoundExpectation("return_url", "southquay.example/departures"),
    ),
    forbidden=("every monday",),
)

_CROSSING_PHRASINGS = (
    (
        "the return is on https://southquay.example/departures and the outbound is on "
        "https://northpier.example/departures — compare the fares every monday please",
    ),
    (
        "every monday could you compare the outbound fare at "
        "https://northpier.example/departures against the return fare at "
        "https://southquay.example/departures?",
    ),
    (
        "i'd like the outbound crossing on https://northpier.example/departures and the "
        "return crossing on https://southquay.example/departures checked every monday for "
        "whichever is cheaper",
    ),
    (
        "would you mind pricing the return at https://southquay.example/departures and "
        "the outbound at https://northpier.example/departures every monday?",
    ),
)

_CROSSING_PORT = PortedBinding(
    case_id=_CROSSING.case_id,
    behaviour=(
        f"{_LOCUS}when a routine declares two parameters of the same kind and one message "
        "supplies both, Penny binds each to the page the ask gives for that parameter and "
        "never to the other's."
    ),
    fixture=_CROSSING,
    also_phrased=_CROSSING_PHRASINGS,
    forbids=("every monday",),
    # The role words are what say which address answers which parameter — no claim reads
    # them, and an arm missing one would leave the assignment unanswerable.
    also_states=("outbound", "return"),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_two_same_kinded_slots_take_the_pages_the_ask_gives_each(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: price the outbound on one page and the return on another.

    Two declared parameters of the same kind, both supplied, and the addresses say nothing
    about which is which — so the only evidence for the assignment is the word the ask puts
    beside each one, and two arms name the return first so that reading it is the only way
    through.  A draw that swaps them binds a routine pointed at the right pages in the wrong
    roles, which every downstream check would read as a correct binding: the names match the
    signature, both values are spans of what the user said, and the container's derived name
    carries both.  Nothing but this case looks at which went where.

    The span claims are CONTAINMENT, so an over-long bind that swallowed its neighbouring
    clause satisfies them (#1884 item 2) — reading a full rate here means the roles were told
    apart, not that each value stops where it should.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _CROSSING_PORT, model)
    # LANDED
    cohort.claim(
        "state: the draw bound every declared parameter",
        _bound_every_parameter,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE
    cohort.claim(
        "state: the outbound url is bound to the page the ask gives for it",
        _binds_the_span_the_ask_supplies(_CROSSING_PORT, "outbound_url"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the return url is bound to the page the ask gives for it",
        _binds_the_span_the_ask_supplies(_CROSSING_PORT, "return_url"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: neither url is bound to the page that answers the other",
        _no_value_takes_another_parameters_span(_CROSSING_PORT),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no bound value carries the job's cadence",
        _no_value_carries_the_terms(_CROSSING_PORT),
        SpecCategory.PROVENANCE,
    )

    cohort.measure(
        output_field(BIND_OUTCOME),
        _bound_value_axis(_CROSSING_PORT, "outbound_url"),
        _bound_value_axis(_CROSSING_PORT, "return_url"),
    )


# ── Ported: the only parameter, and the ask names no page ─────────────────────
#
# The survivor is ``bind-missing-page``: the ask is a perfectly good description of the job
# and supplies nothing to point it at.  The temptation is a value that is right there in the
# sentence and is not a page — "that brass lantern" reads like an answer, and a routine bound
# to it would go and watch nothing — so the object itself is FORBIDDEN beside the cadence.
# Without that this case's only provenance claim would be green on the very draw it names.

_MISSING_PAGE_PHRASINGS = (
    ("every hour, could you check whether the price of that brass lantern has moved?",),
    ("i'd like to know when the price of that brass lantern changes — check every hour",),
    (
        "would you mind watching that brass lantern's price every hour and telling me "
        "when it shifts?",
    ),
    ("please let me know every hour if the price on that brass lantern changes",),
)

_MISSING_PAGE_PORT = PortedBinding(
    case_id="binder-reports-the-only-parameter-missing-rather-than-binding-a-near-value",
    behaviour=(
        f"{_LOCUS}when the ask describes the job and names no page, Penny reports the page "
        "missing rather than binding the thing the ask does name."
    ),
    fixture=_MISSING_PAGE,
    also_phrased=_MISSING_PAGE_PHRASINGS,
    forbids=("brass lantern", "every hour"),
    # Every page this suite names lives on this host, so an arm carrying one has supplied
    # the very thing the case exists to find missing.
    never_states=(ANY_PAGE_HOST,),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_ask_with_no_page_in_it_reports_the_page_missing(
    binder_eval: BinderEval, model: str
) -> None:
    """One ask in five wordings: watch the price of a thing the user never says where to find.

    Naming the parameter missing is the answer the contract asks for, and since #1885 it is
    what ROUTES the turn into request — an enumerated outcome the machine acts on, never a
    failed draw.  Two claims: which of the two enumerated answers the draw wrote, and the one
    thing a wrong draw would leave behind — whatever it bound instead.

    A correct draw binds nothing, so the bound-value axis is deliberately not MEASURED — it
    would read unset on every sample, which the pooler reports as blind rather than as
    agreement.  The outcome axis is where a draw that filled it anyway diverges.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _MISSING_PAGE_PORT, model)
    # LANDED — the one CLOSED field with any discretion in it, asserted by equality.
    #
    # WHICH parameter it named is NOT claimed here, and on this signature that is closed the
    # same way a complete binding's "it names nothing missing" is: the routine declares ONE
    # parameter, and `_fills_the_declared_signature` accepts a draw only when the drawn names
    # equal the declared names as a multiset — so a shortfall at all IS a shortfall naming the
    # url, and a claim over it would run 15/15 by construction.  It is a real claim on the
    # two-parameter sibling, where the draw chooses between two names, and it is made there.
    cohort.claim(
        "state: the draw reported a shortfall rather than a complete binding",
        _reported_a_shortfall,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE — a correct draw leaves no open field, so what this reads is what a WRONG
    # one bound: the object the ask does name, or the cadence beside it, neither of which is
    # a page.
    cohort.claim(
        "state: no value carries the object or the cadence the ask names instead of a page",
        _no_value_carries_the_terms(_MISSING_PAGE_PORT),
        SpecCategory.PROVENANCE,
    )

    # What the draw reported missing is NOT measured, for the reason it is not claimed: it is
    # fully determined by the outcome on a one-parameter signature, so its H would be 0.000
    # whatever happened — a number that reads as agreement where there was no discretion.
    cohort.measure(output_field(BIND_OUTCOME))


# ── Ported: the completion draw, when the value finally arrives ───────────────
#
# New (#2057), and the WARM shape — every other case in this file drives the cold one.
# Since #1894 a round parked in request hands the binder what it already SETTLED, and only
# the still-OPEN parameters are rendered and offered: a parameter a parked round already
# settled is not a question to ask again, it is an answer to carry.
#
# So the fixture is that document: the signature offers the keyword ALONE, and the user's
# turns are the two the round has now heard — the under-specified ask that parked it (the
# same one ``binder-fills-one-and-names-the-other-missing`` reports the shortfall on) and the
# reply that just arrived.  The page is still in the words and no longer in the signature,
# which is the whole temptation: the only open slot is the entry, and the most page-shaped
# thing in the document answers a parameter nobody offered — so the page it settled is
# FORBIDDEN in the value, beside the cadence.
#
# Reference values (read at review, never matched):
#   keyword = dawn sailing

_ARRIVING_KEYWORD = BindingFixture(
    case_id="binder-fills-the-still-open-parameter-when-the-value-arrives",
    skill=_MISSING_KEYWORD.skill,
    intent=_MISSING_KEYWORD.intent,
    parameters=(_TIMETABLE_PARAMETERS[1],),
    turns=(*_MISSING_KEYWORD.turns, "the dawn sailing — that's the one i'm after"),
    rendered_input=(
        "The routine that has been asked for:\n"
        "name: check_ferry_timetable\n"
        "what it is for: Check a ferry timetable page for updates and report the status "
        "of a specified line\n"
        "\n"
        "What it needs, one line each:\n"
        "- keyword: text indicating which timetable entry to look for\n"
        "\n"
        "What the user said, in their own words:\n"
        "can you check the timetable at https://northpier.example/departures every "
        "morning and keep me posted?\n"
        "the dawn sailing — that's the one i'm after"
    ),
    expectations=(BoundExpectation("keyword", "dawn sailing"),),
    forbidden=("northpier.example/departures", "every morning"),
)

# The four other wordings vary the ARRIVING turn alone.  The turn that parked the round is
# the round's own history — a fact of the world every arm is answered against, not something
# the user is saying again — so it is byte-identical throughout, and what moves is the reply
# whose value the draw has to read.
_ARRIVING_KEYWORD_PHRASINGS = (
    (*_MISSING_KEYWORD.turns, "it's the dawn sailing i want to know about"),
    (*_MISSING_KEYWORD.turns, "sorry, i meant the dawn sailing"),
    (*_MISSING_KEYWORD.turns, "i'm looking for the dawn sailing"),
    (*_MISSING_KEYWORD.turns, "the dawn sailing please"),
)

_ARRIVING_KEYWORD_PORT = PortedBinding(
    case_id=_ARRIVING_KEYWORD.case_id,
    behaviour=(
        f"{_LOCUS}when a parked round hands back what it already settled and only the "
        "still-open parameter is offered, Penny fills it from the turn that just arrived "
        "and never from what the round had already settled."
    ),
    fixture=_ARRIVING_KEYWORD,
    also_phrased=_ARRIVING_KEYWORD_PHRASINGS,
    forbids=("northpier.example/departures", "every morning"),
    # The page is in the PARKED turn and must not be in the arriving one: a reply that named
    # it again would make "filled from the turn that just arrived" unanswerable.
    never_states=(ANY_PAGE_HOST,),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_still_open_parameter_is_filled_from_the_turn_that_arrived(
    binder_eval: BinderEval, model: str
) -> None:
    """A parked round's missing detail arrives, in five wordings of the reply.

    The signature offers only what is still open, so the page the round settled two turns ago
    is present in the words and absent from the question — and a draw that answers the open
    parameter with it produces a routine that looks for its own address in the timetable.
    That is what the second provenance claim reads, and nothing downstream could catch it:
    the value would be a literal span of what the user said, the drawn names would equal the
    offered ones, and the completed binding would derive a perfectly ordinary container name.

    The other direction is the case's point: the value comes from the turn that just arrived.

    **What these claims cannot see is an OVER-LONG bind** (#1884 item 2): the span claim is
    containment and the no-terms claim reads a per-case list every arm must carry, so a draw
    binding the whole arriving turn — "the dawn sailing — that's the one i'm after" — carries
    the anchor, carries no forbidden term, and scores a full mark.  A full rate here means the
    value came from the right turn, not that it stops where the entry stops.  Closing it needs
    a term forbidden on ONE arm rather than on the case, which the arms contract does not have
    and which is a harness change, not a fixture one.

    See the section note above for the empty STORE category and the three claims closed
    upstream."""
    cohort = await _drive_ported(binder_eval, _ARRIVING_KEYWORD_PORT, model)
    # LANDED
    cohort.claim(
        "state: the draw bound the parameter that was still open",
        _bound_every_parameter,
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section note.

    # PROVENANCE
    cohort.claim(
        "state: the keyword is bound to the entry the arriving turn supplies",
        _binds_the_span_the_ask_supplies(_ARRIVING_KEYWORD_PORT, "keyword"),
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: it carries neither the page nor the cadence the parked round already settled",
        _no_value_carries_the_terms(_ARRIVING_KEYWORD_PORT),
        SpecCategory.PROVENANCE,
    )

    cohort.measure(
        output_field(BIND_OUTCOME),
        _bound_value_axis(_ARRIVING_KEYWORD_PORT, "keyword"),
    )


# Every ported case, for the arms probe in ``make check`` — one place, so the probe and the
# live runs can never be checking two different cohorts.
PORTED_BINDINGS = (
    _MISSING_KEYWORD_PORT,
    _LISTING_PORT,
    _COUNT_PORT,
    _TWO_PARAMETERS_PORT,
    _CROSSING_PORT,
    _MISSING_PAGE_PORT,
    _ARRIVING_KEYWORD_PORT,
)

# Every case's document, for the deterministic drift probes in ``make check`` — one place, so
# the probes and the live runs can never be checking two different fixtures.  The last two
# are driven only through the cohort path above; they are fixtures because a document nobody
# pins is a case measuring whatever it happens to render.
FIXTURES = (
    _LISTING,
    _TWO_PARAMETERS,
    _DAILY_SPECIAL,
    _COUNT,
    _NEW_ARRIVALS,
    _MISSING_PAGE,
    _MISSING_KEYWORD,
    _CROSSING,
    _ARRIVING_KEYWORD,
)
