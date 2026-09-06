"""Live-model contract for the skill FRAMER (#1830, the #1824 inversion).

The framer has ONE job: given the user's own turns of the round and nothing else, write
the routine's public interface — a generic name, a one-line description, and the
parameter(s) the user would have to say again to set the same routine running on a new
occasion.  It never sees the tool calls; the leaf labeller, which names the routine's
implementation, never sees the ask (#1824).  Nothing is offered to it: it MINTS the
parameters by reading what the user said.

The suite is in two halves.  **Seven INLINE cases** score each sample through the per-check
scorer below, all the same look-up → find → remember shape, deliberately spanning the three
multiplicity shapes — one argument, two of the SAME type, two of DIFFERENT types — while
varying topic, how much the ask names, and how many pieces have to be re-supplied:

* ``frame-availability-page-only`` — the ask names the point, so what to check bakes.
* ``frame-two-sources-two-parameters`` — two pages named: TWO distinct parameters.
* ``frame-ticker-only-parameter`` — "tell me when it moves" contributes no parameter.
* ``frame-single-turn-floor`` — one turn: the finding bakes, the page survives.
* ``frame-search-parameter`` — the look-up is a text search, not a page.
* ``frame-two-types-page-and-title`` — a page + a title: two pieces, different types.
* ``frame-two-same-type-symbols`` — two symbols, where a single list parameter is most
  tempting and each must stay its own scalar.

**Seven PORTED cases** (#2006/#2056) are the framer's decisions covered in ISOLATION, each
one ask in five wordings pooled into a cohort of fifteen and claimed against
``docs/eval-case-design.md`` rather than the scorer.  Every one of them states its behaviour
in the fixed sentence form, asserts the parameter SET by equality under LANDED, and reports
STORE and PROVENANCE empty with the reason:

* ``framer-mints-only-the-piece-that-varies`` — the ticker ask: the symbol is the one piece
  that varies, and the share price and the notification are both settled elsewhere.
* ``framer-keeps-two-of-a-kind-as-two-parameters`` — two URLS, where run 1 measured all five
  samples folding both into one ``sites — list of URLs``.
* ``framer-mints-both-pieces-when-they-are-different-kinds`` — a catalog page and a book
  looked up on it.
* ``framer-names-a-search-as-a-search`` and
  ``framer-names-a-page-as-a-page-and-invents-no-search`` — a PAIR with opposite expected
  answers: a search-family parameter is correct for the first and is exactly the invention
  the second exists to catch (round 8's recorded failure).
* ``framer-frames-from-a-single-turn`` — the whole teach in one sentence, so the routine's
  purpose and the piece that varies have to be separated out of one turn rather than read
  off two.
* ``framer-keeps-three-of-a-kind-as-three-parameters`` — the same-kind ask at three, where
  the list is more tempting than it is at two.

What the ported cases deliberately do NOT claim, and why, is stated in each of them: the
genericity contract (that a routine's name says the KIND of task and never THIS occasion) is
the converse of a provenance claim and fits none of the design's three categories, so it is
read by a person on the modal sample; and the three obvious-looking claims about a usable
signature are ``_mints_a_usable_signature``'s, closed upstream.

Every case's input is the round's USER turns, rendered by the shipped
``build_framing_content`` — never hand-written — so the draw reads exactly what production
would render, and both halves are held against what they claim by a deterministic drift
probe in ``make check`` (see ``tests/test_eval_harness.py``): an inline case pins its
document byte-for-byte as ``rendered_input``, and a ported case declares its five wordings
beside the facts every one of them carries (``PortedArms``).  A fixture that drifts from
what it claims is a case measuring nothing, and it must fail before any GPU time, not after.

INLINE scoring is the parameter SET, exactly — each expected family answered by exactly one
drawn parameter, nothing else asked for — plus the structural check that the name and
description say the KIND of task and never the occasion.  The reference outputs below
each inline case are agreed TARGETS read at joint review, never strings a scorer matches: a
parameter the reference calls ``url`` may come back as ``page_to_watch`` and pass.  Every
drawn name, description and parameter rides ADVISORY so a reader sees what the model
committed to.  A PORTED case makes the first two of those as CLAIMS instead
(``_claim_the_parameter_set``) and does not make the third at all, for the reason above.

Since #1868 the draw happens when the machine ENTERS learn rather than at run end: the
round's identity is settled before the round runs, and run-end extraction READS that
framing instead of drawing again — a run-end draw survives only for a round nothing
framed.  Both entries render their document through the same shipped
``build_framing_content``, so the draw this case drives is the one either path makes.  The
draw also gives each parameter the VALUE the round demonstrated it with, and those values
are what a job's container is NAMED from — so each parameter's advisory carries its drawn
value and the run closes with the container name the shipped derivation makes of them.
That a value is a literal span of the user's own words is the production validator's job
(an accepted draw cannot carry a value nobody said); WHICH span was the right one is the
same kind of judgment as a name, so it is rendered for review rather than matched by a
fixture.

All content is synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    FRAME_DESCRIPTION,
    FRAME_NAME,
    FRAME_PARAMETERS,
    FramerEval,
    ParameterFamily,
    classify_by_family,
    frame_parameter_name,
    frame_parameter_says,
    framed_parameters,
)
from penny.tests.eval.utils.assertions import Answer, Cohort, WorldClaim
from penny.tests.eval.utils.cohort import (
    Consequence,
    SampleObservation,
    SpecCategory,
    output_field,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "skill-framing"


class FramingFixture(NamedTuple):
    """One agreed case: the user turns that are its whole input, the document the
    shipped renderer must produce from them, the parameters the ask genuinely requires,
    and the occasion's own words — which may appear in neither the name nor the
    description."""

    case_id: str
    turns: tuple[str, ...]
    rendered_input: str
    parameters: tuple[ParameterFamily, ...]
    instance_tokens: tuple[str, ...]


async def _run_case(framer_eval: FramerEval, fixture: FramingFixture) -> None:
    """Drive one case's turns through the framer.  Every case is report-only: the
    thresholds are the code owner's to set once the first numbers are read."""
    await framer_eval(
        case_id=fixture.case_id,
        turns=fixture.turns,
        parameters=fixture.parameters,
        instance_tokens=fixture.instance_tokens,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


# The breadth agreed for "the page the routine is pointed at" — a piece the user
# re-supplies that is a place to go.  Shared by the page cases because it is the same
# piece under three topics; a case that needs to tell two of them apart adds ordinals.
#
# A page family is NAME-ONLY (the code owner's ruling on the first run): a parameter is
# the page when it is NAMED as the page.  Two identical `city — name of the location on
# the site …` draws scored opposite ways when a description-level mention of "site"
# could promote one, which is the scorer answering for the draw.
_PLACE_TOKENS = ("url", "page", "link", "address", "site", "source", "product", "listing")


def _page_family(label: str, *extra: str) -> ParameterFamily:
    """The page/url family for a case, name-only, with any case-specific words."""
    return ParameterFamily(label, (*_PLACE_TOKENS, *extra), name_only=True)


# The ordinal families a same-kind ask is answered by.  Two things of one kind have no type
# to tell them apart, so what a correct draw distinguishes them by is POSITION, and the
# breadth is every ordinary way of writing that — the ordinal word, the cardinal, the digit,
# the latinate form, and (through `_name_tokens`' letter-suffix reading) `_a`/`_b`/`_c`.
#
# Declared ONCE, like `_page_family`, because the breadth is the code owner's and a
# same-kind case that spelled its own copy would make widening it a several-site edit.
_ORDINAL_BREADTH = (
    ("first", "one", "1", "primary"),
    ("second", "two", "2", "secondary"),
    ("third", "three", "3", "tertiary"),
)


def _ordinal_family(label: str, position: int, *extra: str) -> ParameterFamily:
    """The ``position``-th ordinal family (1-based), with any case-specific words.

    ``extra`` is where a case states a word that only identifies a position IN THAT CASE:
    "the other one" picks out the second of TWO and picks out nothing among three, so the
    two-of-a-kind cases pass it and the three-of-a-kind case does not."""
    return ParameterFamily(label, (*_ORDINAL_BREADTH[position - 1], *extra))


_FIRST_OF_A_KIND = _ordinal_family("first source", 1)
_SECOND_OF_A_KIND = _ordinal_family("second source", 2, "other")


class PortedArms(NamedTuple):
    """One ported case's five wordings, together with the facts every one of them carries.

    The facts sit BESIDE the arms because they are the case's own claim about them: every
    claim the case makes hinges on what the ask REQUIRES, so an arm that stopped naming one
    of them would have its samples answering a different question under one case id.  A
    second copy of the list in the probe would drift from the arms exactly as silently as no
    probe at all, so the probe reads this.

    ``carries`` is a substring every arm must contain and ``never`` one no arm may.
    ``any_of`` is a fact stated as an ALTERNATION — every arm must carry one of its
    substrings, which is how a fact that survives being reworded ("let me know" / "tell me" /
    "ping me") is declared rather than left out.  Without it such a fact is invisible to the
    probe and can go missing from one arm silently, which is the drift the probe exists for.

    ``turns_per_arm`` is how many turns an arm takes, since an ask is however many turns the
    user needed for it and the single-turn case's whole subject is that number."""

    case_id: str
    arms: tuple[tuple[str, ...], ...]
    carries: tuple[str, ...]
    any_of: tuple[str, ...] = ()
    never: tuple[str, ...] = ()
    turns_per_arm: int = 2


def _answers(family: ParameterFamily, among: Sequence[ParameterFamily]) -> WorldClaim:
    """A claim that EXACTLY ONE minted parameter answers one piece the ask requires.

    The parameter set is a CLOSED field, so this is asserted by equality under LANDED rather
    than traced under PROVENANCE — but it is still one half of a pair, and its other half is
    the count below: nothing the ask requires was left out, and nothing it does not was
    added.

    "Answers" is decided by :func:`classify_by_family`, the one classification discipline
    every suite that asks what a drawn parameter is for reads.  That is a closed equivalence
    class agreed with the code owner and pinned in ``make check``, not a judgement made per
    run: the family's tokens are the breadth a piece may be named at, so ``ticker``,
    ``symbol`` and ``stock_symbol`` are one answer and a reference name is a target rather
    than a string to match.  Which is what makes this an assertion rather than a reading —
    a differently-worded correct answer passes, and no correct draw can answer it twice.

    Nothing at all answering it is a piece the routine can no longer be pointed at; two
    answering it is the same piece asked for twice.

    ``among`` is the case's WHOLE family set and the classification runs over it ONCE, not
    once per family.  Classifying against one family at a time would break the discipline it
    is asking for: the name pass is meant to run over every family before any description is
    read, so a name that already claimed the page would otherwise still be offered to the
    title family's description fallback and answer that one too.  A single-family case
    passes ``(family,)`` and reads exactly as it did."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = framed_parameters(sample)
        matched = [
            name
            for (name, _says), family_of in zip(
                drawn, classify_by_family(drawn, among), strict=True
            )
            if family_of is not None and family_of.label == family.label
        ]
        return len(matched) == 1, f"{len(matched)} answer it: {matched or 'none'}"

    return answer


def _asks_for_nothing_else(required: int) -> WorldClaim:
    """The count — the other half of the pair above, and every case's negative direction.

    Anything beyond the pieces the ask requires is a piece the user would be made to
    re-supply that their own ask already settled — the cadence, the notification, a search
    nobody asked for, the thing the routine is FOR.  So the count IS the negative direction,
    and it is stated here rather than as a list of timing words or a family of search words:
    a rule keyed to the vocabulary of the clause in front of us would not fire for a clause
    nobody enumerated, and a parameter minted for anything the ask does not require fails
    this claim whatever it is called."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = sample.field(FRAME_PARAMETERS)
        return drawn == str(required), f"minted {drawn}"

    return answer


def _claim_the_parameter_set(cohort: Cohort, families: Sequence[ParameterFamily]) -> None:
    """The LANDED block every ported framing case makes, over the families its ask requires.

    A framing draw has no enumerated outcome to land on — a signature came back or nothing
    did, and nothing is the completeness gate's — so the parameter SET is what this category
    holds for this shape, asserted by equality: one drawn parameter per piece the ask
    requires, and no parameter for anything it does not.

    Shared rather than repeated at each case because there is more than one customer for it
    now; what stays per-case is the family set, which is where each case's own behaviour
    lives."""
    for one in families:
        cohort.claim(
            f"state: exactly one parameter answers the {one.label}",
            _answers(one, families),
            SpecCategory.LANDED,
        )
    cohort.claim(
        "state: the routine asks for nothing else",
        _asks_for_nothing_else(len(families)),
        SpecCategory.LANDED,
    )


def _measure_the_draw(cohort: Cohort, positions: int) -> None:
    """The axes every ported framing case measures — the draw's own structured fields.

    The COUNT is consequential: a routine asking for two things is a different interface from
    one asking for one.  The name, the description and each drawn parameter's own name and
    line are COSMETIC — ``stock_tracker`` and ``share_price_watcher`` leave the same
    interface behind — and the naming spread is the framer's known system-level finding (0.90
    on both measured models), so it belongs in the variance table and never as a fact about
    one sample.

    Only the positions the ask REQUIRES are measured: a position past them exists on a
    divergent sample alone, so its axis would read ``unset`` for the pack and score a
    disagreement as agreement.

    No tool sequence and no reply spread: a single call makes neither."""
    cohort.measure(
        output_field(FRAME_PARAMETERS),
        output_field(FRAME_NAME, consequence=Consequence.COSMETIC),
        output_field(FRAME_DESCRIPTION, consequence=Consequence.COSMETIC),
        *[
            output_field(field(position), consequence=Consequence.COSMETIC)
            for position in range(1, positions + 1)
            for field in (frame_parameter_name, frame_parameter_says)
        ],
    )


# ── Case 1: the ask names the point, so only the page is left to say ──────────
#
# Reference output (read at review, never matched):
#   NAME: stock-watcher
#   DESCRIPTION: watch a product page for the item coming back in stock
#   PARAMETER url — the product page to check

_AVAILABILITY = FramingFixture(
    case_id="frame-availability-page-only",
    turns=(
        "can you keep an eye on bookbarn.example/atlas-of-clouds and let me know "
        "when it's back in stock",
        "go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and remember that",
    ),
    rendered_input=(
        "can you keep an eye on bookbarn.example/atlas-of-clouds and let me know "
        "when it's back in stock\n"
        "go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and remember that"
    ),
    parameters=(_page_family("url"),),
    instance_tokens=("bookbarn", "atlas", "clouds"),
)


@pytest.mark.asyncio
async def test_what_the_ask_named_bakes_and_the_page_survives(framer_eval: FramerEval):
    """The user said what they were after — the item coming back in stock — so a routine
    that asks what to check would be asking them to restate what they came for.  The page
    is the one piece a new occasion needs, and the framing carries the rest."""
    await _run_case(framer_eval, _AVAILABILITY)


# ── The ported case: a page and no search, in five wordings ───────────────────
#
# The NEGATIVE half of the pair `framer-names-a-search-as-a-search` opens: a search-family
# parameter is the right answer there and the wrong one here, which is what makes them two
# cases rather than one behaviour under two sets of facts.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same address, asks
# whether the book is in stock, and carries a clause about being TOLD when it is.  What
# varies is only how a person says that.  The telling clause is load-bearing — it is the
# second thing the case claims is not a parameter — so it is declared as an alternation
# rather than left to be noticed, because it is the one fact here that survives being
# reworded into different words on every arm.
#
# No arm says "search" — the word is what the case is about, so an arm carrying it would be
# asking for the very thing the case claims is not asked for.
_PAGE_ONLY_PHRASINGS = (
    (
        "could you watch bookbarn.example/atlas-of-clouds and tell me when it's in stock again",
        "open bookbarn.example/atlas-of-clouds, see whether it's in stock, and remember that",
    ),
    (
        "i'd like bookbarn.example/atlas-of-clouds followed — let me know once it's back in stock",
        "read bookbarn.example/atlas-of-clouds, find out whether it's in stock, and save that",
    ),
    (
        "keep tabs on bookbarn.example/atlas-of-clouds for me — ping me when it's in stock",
        "visit bookbarn.example/atlas-of-clouds, check if it's in stock, and keep that",
    ),
    (
        "can you track bookbarn.example/atlas-of-clouds and say when it's back in stock?",
        "look at bookbarn.example/atlas-of-clouds, check whether it's in stock, and store that",
    ),
)

PAGE_ONLY_ARMS = PortedArms(
    case_id="framer-names-a-page-as-a-page-and-invents-no-search",
    arms=(_AVAILABILITY.turns, *_PAGE_ONLY_PHRASINGS),
    carries=("bookbarn.example/atlas-of-clouds", "stock"),
    any_of=("let me know", "tell me", "ping me", "say when"),
    never=("search", "query"),
)

_PAGE_ONLY_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when an ask names a page "
    "and no search, Penny mints the page and nothing beside it."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_page_ask_mints_the_page_and_invents_nothing_beside_it(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: watch this address, tell me when the book is in stock.

    The address is the one piece a new occasion has to supply.  What the user came for — the
    availability — is what the routine IS, and being told when it changes is settled where
    the job is set running, so neither is a parameter.  Round 8's recorded failure is the
    third thing that is not one: a ``search_term`` promoted for an ask that named a page and
    no search, invented whole rather than taken from anything the user provided.

    **That failure is measured by the COUNT and not by a search family**, deliberately.  A
    claim naming the search vocabulary would be keyed to the phrasing of the one failure in
    front of us and would not fire for an invention nobody enumerated; the count fails on a
    parameter minted for anything the ask does not require, whatever it is called.

    **STORE is EMPTY, and that is the correct report** — a micro-context is one call that
    returns a typed result, so it moves no machine and writes to no store.

    **PROVENANCE is EMPTY, and that needs its reason stated**: a framing's open fields are an
    identifier and two lines of deliberately generic prose, and the suite's one instrument
    for the invented direction does not transfer to them.  The measurement behind that is in
    ``test_the_symbol_is_the_parameter_and_everything_else_bakes``; it is not re-derived here.

    **Three claims are missing because production already validates them** — that the draw
    minted at least one parameter, that no two share a name, and that every demonstrated
    value is a literal span of the user's own turns are all ``_mints_a_usable_signature``'s,
    re-rolled until they hold, so each would run 15/15 by construction.
    """
    cohort = await framer_eval(
        case_id=PAGE_ONLY_ARMS.case_id,
        behaviour=_PAGE_ONLY_BEHAVIOUR,
        model=model,
        turns=_AVAILABILITY.turns,
        also_phrased=_PAGE_ONLY_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED
    _claim_the_parameter_set(cohort, _AVAILABILITY.parameters)

    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_AVAILABILITY.parameters))


# ── Case 2: two pages named is two parameters ─────────────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: headline-collector
#   DESCRIPTION: collect the top headline from each of the news front pages it is
#                pointed at
#   PARAMETER first_site — the first front page to read
#   PARAMETER second_site — the second front page to read

_TWO_SOURCES = FramingFixture(
    case_id="frame-two-sources-two-parameters",
    turns=(
        "hey could you keep an eye on the morning headlines for me",
        "read citydesk.example/front and harborpost.example/front, and remember each "
        "site's top headline",
    ),
    rendered_input=(
        "hey could you keep an eye on the morning headlines for me\n"
        "read citydesk.example/front and harborpost.example/front, and remember each "
        "site's top headline"
    ),
    parameters=(_FIRST_OF_A_KIND, _SECOND_OF_A_KIND),
    instance_tokens=("citydesk", "harborpost"),
)


@pytest.mark.asyncio
async def test_two_sources_become_two_distinct_parameters(framer_eval: FramerEval):
    """The user pointed the routine at two pages, so both have to be re-suppliable and
    they have to be told apart.  A framer that collapses them into one parameter, or goes
    generic and asks for none, has lost a piece the user named — which is why the COUNT
    is the load-bearing check here."""
    await _run_case(framer_eval, _TWO_SOURCES)


# ── The ported case: two of a kind, in five wordings ──────────────────────────
#
# The world is URLS rather than symbols because that is where the failure was MEASURED: on
# run 1 all five two-sources samples drew one `sites — list of URLs`, folding two pieces the
# user named into a single list parameter.  The same-kind symbols ask is the facts-only
# sibling of this one and stays inline.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names both front pages and asks
# for each site's top headline.  An arm naming one page would leave a family claim with
# nothing that could answer it, and an arm asking for something other than the headline
# would be a different ask under one case id.
_TWO_SOURCES_PHRASINGS = (
    (
        "could you watch the morning news for me",
        "open citydesk.example/front and harborpost.example/front, find each site's top "
        "headline, and remember them",
    ),
    (
        "i'd like the morning headlines followed",
        "check citydesk.example/front and harborpost.example/front, and keep each one's "
        "top headline",
    ),
    (
        "can you keep track of the morning headlines?",
        "look at citydesk.example/front and harborpost.example/front, and save the top "
        "headline from each",
    ),
    (
        "keep up with the morning headlines for me",
        "go to citydesk.example/front and harborpost.example/front, get each site's top "
        "headline, and store them",
    ),
)

TWO_SOURCES_ARMS = PortedArms(
    case_id="framer-keeps-two-of-a-kind-as-two-parameters",
    arms=(_TWO_SOURCES.turns, *_TWO_SOURCES_PHRASINGS),
    carries=("citydesk.example/front", "harborpost.example/front", "top headline"),
)

_TWO_SOURCES_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when one ask points a "
    "routine at two things of the same kind, Penny mints two distinct scalar parameters "
    "rather than folding them into one list."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_two_of_a_kind_stay_two_distinct_parameters(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: read these two front pages, keep each one's top headline.

    Two things of the SAME kind are where a single list parameter is most tempting, and a
    list is not a parameter — what a user says fills one whole.  Both pages have to be
    re-suppliable next time and they have to be tellable apart, so a correct draw mints two
    scalars distinguished by position.  The headline is what the routine IS and belongs in
    the framing.

    **Test 1 is what separates this from ``framer-mints-only-the-piece-that-varies``**: a
    one-parameter draw is the correct answer there and the wrong one here, so the two are two
    cases rather than one behaviour under two sets of facts.  The failure this case exists to
    catch fails all three claims at once — a single ``sites — list of URLs`` answers neither
    ordinal family and mints one where two are required — and that is several unmet
    contracts, not one counted three times.

    **STORE is EMPTY, and that is the correct report** — one call returns a typed result; it
    moves no machine and writes to no store.

    **PROVENANCE is EMPTY, and that needs its reason stated**: a framing's open fields are an
    identifier and two lines of deliberately generic prose, and the suite's one instrument
    for the invented direction does not transfer to them.  The measurement behind that is in
    ``test_the_symbol_is_the_parameter_and_everything_else_bakes``; it is not re-derived here.

    **Three claims are missing because production already validates them** — at least one
    parameter minted, no two sharing a name, every demonstrated value a literal span of the
    user's own turns — all ``_mints_a_usable_signature``'s, re-rolled until they hold.
    """
    cohort = await framer_eval(
        case_id=TWO_SOURCES_ARMS.case_id,
        behaviour=_TWO_SOURCES_BEHAVIOUR,
        model=model,
        turns=_TWO_SOURCES.turns,
        also_phrased=_TWO_SOURCES_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED
    _claim_the_parameter_set(cohort, _TWO_SOURCES.parameters)

    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_TWO_SOURCES.parameters))


# ── Case 3: cadence and notification are not signature ────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: stock-tracker
#   DESCRIPTION: track a stock's share price
#   PARAMETER ticker — the stock symbol to track

_TICKER = FramingFixture(
    case_id="frame-ticker-only-parameter",
    turns=(
        "can you track a stock for me and tell me when it moves",
        "look up VLT, find the share price, and remember it under VLT",
    ),
    rendered_input=(
        "can you track a stock for me and tell me when it moves\n"
        "look up VLT, find the share price, and remember it under VLT"
    ),
    parameters=(ParameterFamily("ticker", ("ticker", "symbol", "stock", "share", "company")),),
    instance_tokens=("vlt",),
)


@pytest.mark.asyncio
async def test_the_share_price_is_the_routine_and_the_ticker_is_the_parameter(
    framer_eval: FramerEval,
):
    """The share price is what the skill IS, so it belongs in the framing; the ticker is
    the one thing said again next time.  "Tell me when it moves" is settled when the
    routine is set running and contributes nothing to the signature — a framer that turns
    it into a parameter has made a delivery preference into something to be re-supplied."""
    await _run_case(framer_eval, _TICKER)


# ── The ported case: the ticker ask, in five wordings ─────────────────────────
#
# The survivor is ``frame-ticker-only-parameter``, because its ask is the plainest
# statement of the whole behaviour AND of its negative direction at once: what the user
# came for (the share price) bakes into the framing, the one thing that varies (the
# symbol) becomes the parameter, and "tell me when it moves" — which is settled where the
# job is set running — becomes neither.
#
# THE FACTS ARE CONSTANT across the five wordings, because the claims hinge on them: every
# arm names the symbol VLT, asks for its share price, says to remember it under VLT, and
# carries a clause about being told when it changes.  What varies is only how a person
# says that.  An arm that dropped the notification clause would be a different ask and its
# samples would be measuring a different behaviour under one case id.
#
# An arm is a SEQUENCE of turns, not a sentence: the round's ask is two turns (the standing
# want, then the demonstration), and ``build_framing_content`` renders the user's turns one
# per line as the whole document.  So a wording of this ask is two turns said differently.
_TICKER_PHRASINGS = (
    (
        "could you keep tabs on a stock for me and let me know when it changes",
        "look up VLT, get the share price, and remember it under VLT",
    ),
    (
        "i'd like a stock followed, with a heads up whenever it moves",
        "check VLT, find the share price, and save it under VLT",
    ),
    (
        "can you watch a stock for me? tell me if it shifts",
        "look VLT up, find the share price, and keep it under VLT",
    ),
    (
        "keep an eye on a stock and ping me when the number moves",
        "look up VLT, find the share price, and store it under VLT",
    ),
)

TICKER_ARMS = PortedArms(
    case_id="framer-mints-only-the-piece-that-varies",
    arms=(_TICKER.turns, *_TICKER_PHRASINGS),
    carries=("VLT", "share price"),
    any_of=("moves", "changes", "shifts"),
)

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename; this is
# the contract, and it renders above every number in the report.
_TICKER_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when a demonstrated "
    "round is turned into a reusable routine, Penny mints one parameter for the piece a new "
    "occasion has to supply and leaves what the user came for — and when they want telling "
    "— in the framing rather than in the interface."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_symbol_is_the_parameter_and_everything_else_bakes(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: track a stock, tell me when it moves, here is the symbol.

    The share price is what the routine IS, so it belongs in the framing; the symbol is the
    one thing said again next time; and being told when it moves is settled where the job is
    set running.  A framer that turns the notification into a parameter has made a delivery
    preference into something to be re-supplied.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    to no store — so there is nothing for a store claim to read.  The signature this draw
    returns is persisted later, by run-end extraction, and this case never runs it.

    **Three claims are missing because production already validates them**, and a thin set
    should read as closed rather than as unrun: that the draw minted at least one parameter,
    that no two parameters share a name, and that every demonstrated value is a literal span
    of the user's own turns are all ``_mints_a_usable_signature``'s, re-rolled until they
    hold, so each would run 15/15 by construction.

    **The PROVENANCE category is empty too, and it needs its reason stated.**  Fact
    alignment reads a draw's OPEN fields, and this draw's are the routine's ``name``, its
    ``description``, each parameter's line, and each parameter's demonstrated ``value``.  The
    value is closed upstream — production refuses one that is not a literal span of the
    user's turns.  The other three are an identifier and two lines of deliberately GENERIC
    prose, which carry no traceable value at all on a correct draw, and the suite's one
    instrument for the invented direction (``unsourced_specifics``) does not transfer to
    them: measured against this case's own document, it reads a Title-Cased but perfectly
    correct framing as four inventions, while a made-up exchange in ``the Nasdaq exchange``
    passes — capitalisation is a rendering the draw chooses, and a single-word invention is
    the probe's declared blind spot.  A check that fails a correct run for a cosmetic reason
    and misses the thing it is for is not an assertion, so this category is empty rather than
    filled with it.

    **And one real contract is NOT measured here, deliberately.**  That the routine's name
    and description say the KIND of task and never THIS occasion — a framing carrying ``VLT``
    is a routine that can only ever run once — fits none of the three assertion categories:
    it is the CONVERSE of a provenance claim (the offending token is in the ask, so nothing
    is invented and nothing is omitted), and inventing a category to keep one measurement is
    how a closed list stops being closed.  It stays what the design calls wrong-but-stable —
    read by a person opening the modal sample, where the drawn name renders verbatim.
    """
    cohort = await framer_eval(
        case_id=TICKER_ARMS.case_id,
        behaviour=_TICKER_BEHAVIOUR,
        model=model,
        turns=_TICKER.turns,
        also_phrased=_TICKER_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the CLOSED field of the typed result: the parameter SET it minted, by
    # equality.  A framing draw has no enumerated outcome to land on (a signature came back
    # or nothing did, and nothing is the completeness gate's), so the set is what this
    # category holds for this shape.
    _claim_the_parameter_set(cohort, _TICKER.parameters)

    # STORE — empty by construction; see the docstring.

    # PROVENANCE — EMPTY, and see the docstring: this draw has no open field a fact
    # alignment can read.

    # What is MEASURED — the draw's own structured fields, at the one position this ask
    # requires.
    _measure_the_draw(cohort, positions=len(_TICKER.parameters))


# ── Case 4: one turn is enough to frame ───────────────────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: temperature-recorder
#   DESCRIPTION: record the daily high temperature from a weather page
#   PARAMETER url — the weather page to read

_SINGLE_TURN = FramingFixture(
    case_id="frame-single-turn-floor",
    turns=("go to weather.example/lisbon, find today's high temperature, and remember it",),
    rendered_input="go to weather.example/lisbon, find today's high temperature, and remember it",
    parameters=(_page_family("url", "forecast", "weather"),),
    instance_tokens=("lisbon",),
)


@pytest.mark.asyncio
async def test_a_single_turn_teach_still_frames_one_parameter(framer_eval: FramerEval):
    """One page and one thing to find, taught in a single turn: the ask at its minimum
    still says what the routine is for, so the finding bakes into the framing and the
    page survives as the parameter."""
    await _run_case(framer_eval, _SINGLE_TURN)


# ── The ported case: the whole teach is one turn, in five wordings ────────────
#
# What makes this its own case is the DOCUMENT, not the facts.  Every other framing ask
# arrives as a standing want and then a demonstration, so what the routine is FOR and the
# piece that varies are said in two different turns.  Here they are one sentence, and the
# framer has to separate them out of it — a materially different decision from the same
# separation read off two turns.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same page, asks for
# today's high temperature, and says to remember it.  Every arm is ONE turn, which is the
# whole subject.
_SINGLE_TURN_PHRASINGS = (
    ("open weather.example/lisbon, get today's high temperature, and save it",),
    ("read weather.example/lisbon, find today's high temperature, and keep it",),
    ("check weather.example/lisbon for today's high temperature and store it",),
    ("visit weather.example/lisbon, look up today's high temperature, and remember it",),
)

SINGLE_TURN_ARMS = PortedArms(
    case_id="framer-frames-from-a-single-turn",
    arms=(_SINGLE_TURN.turns, *_SINGLE_TURN_PHRASINGS),
    carries=("weather.example/lisbon", "high temperature"),
    turns_per_arm=1,
)

_SINGLE_TURN_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when the whole teach is a "
    "single turn, Penny still separates what the routine is for from the piece a new occasion "
    "has to supply."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_single_turn_teach_still_separates_the_piece_that_varies(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings, each a single turn: read this page, find today's high, keep it.

    The ask at its minimum still says what the routine is for.  The temperature is the
    finding and bakes into the framing; the page is the one piece a new occasion has to
    supply.  A framer that asks what to find has made the user restate what they came for,
    and one that asks for nothing at all has written a routine that can only repeat the
    occasion it was shown.

    **The count is where both of those land**, which is why it is the negative direction
    here rather than a rule about temperatures.

    **STORE is EMPTY, and that is the correct report** — one call returns a typed result; it
    moves no machine and writes to no store.

    **PROVENANCE is EMPTY, and that needs its reason stated**: a framing's open fields are an
    identifier and two lines of deliberately generic prose, and the suite's one instrument
    for the invented direction does not transfer to them.  The measurement behind that is in
    ``test_the_symbol_is_the_parameter_and_everything_else_bakes``; it is not re-derived here.

    **Three claims are missing because production already validates them** — at least one
    parameter minted, no two sharing a name, every demonstrated value a literal span of the
    user's own turns — all ``_mints_a_usable_signature``'s, re-rolled until they hold.
    """
    cohort = await framer_eval(
        case_id=SINGLE_TURN_ARMS.case_id,
        behaviour=_SINGLE_TURN_BEHAVIOUR,
        model=model,
        turns=_SINGLE_TURN.turns,
        also_phrased=_SINGLE_TURN_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED
    _claim_the_parameter_set(cohort, _SINGLE_TURN.parameters)

    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_SINGLE_TURN.parameters))


# ── Case 5: the look-up is a search, not a page ───────────────────────────────
#
# Reference output (read at review, never matched):
#   NAME: ticket-price-watcher
#   DESCRIPTION: watch an event's cheapest ticket price
#   PARAMETER ticket_search — the search that finds the event's ticket listings

_SEARCH = FramingFixture(
    case_id="frame-search-parameter",
    turns=(
        "can you keep an eye on ticket prices for aurora fest?",
        "search for aurora fest tickets, find the cheapest ticket price, and remember it",
    ),
    rendered_input=(
        "can you keep an eye on ticket prices for aurora fest?\n"
        "search for aurora fest tickets, find the cheapest ticket price, and remember it"
    ),
    parameters=(
        ParameterFamily("ticket search", ("search", "query", "event", "listing", "listings")),
    ),
    instance_tokens=("aurora", "fest"),
)


@pytest.mark.asyncio
async def test_a_search_is_the_parameter_and_the_cheapest_price_is_the_framing(
    framer_eval: FramerEval,
):
    """The look-up is a text search rather than a url, and what varies next time is which
    event is being searched for.  The cheapest price is what the routine is for, so it
    belongs in the name and description, not in a parameter."""
    await _run_case(framer_eval, _SEARCH)


# ── The ported case: the look-up is a search, in five wordings ────────────────
#
# The POSITIVE half of the pair `framer-names-a-page-as-a-page-and-invents-no-search`
# closes: a search-family parameter is the right answer here and the wrong one there.
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same event, says the
# look-up is a search, and asks for the cheapest ticket price.  No arm names an address — an
# arm that did would give the draw a page to mint instead, which is the other case's world
# and not this one.  What the probe can hold is the narrow half of that: no arm carries the
# reserved `.example` domain these fixtures write every address with, so an address written
# the way this suite writes one cannot get in unnoticed.
_SEARCH_PHRASINGS = (
    (
        "could you watch what tickets for aurora fest are going for?",
        "search for aurora fest tickets, get the cheapest ticket price, and save it",
    ),
    (
        "i'd like aurora fest ticket prices followed",
        "look up aurora fest tickets with a search, find the cheapest ticket price, and keep it",
    ),
    (
        "keep tabs on ticket prices for aurora fest",
        "run a search for aurora fest tickets, find the cheapest ticket price, and store it",
    ),
    (
        "can you track how much aurora fest tickets cost?",
        "search aurora fest tickets, find the cheapest ticket price, and remember it",
    ),
)

SEARCH_ARMS = PortedArms(
    case_id="framer-names-a-search-as-a-search",
    arms=(_SEARCH.turns, *_SEARCH_PHRASINGS),
    carries=("aurora fest", "search", "cheapest ticket price"),
    never=("example",),
)

_SEARCH_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when the look-up an ask "
    "describes is a text search rather than an address, Penny mints one parameter named for "
    "the search."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_search_look_up_is_named_as_a_search(framer_eval: FramerEval, model: str) -> None:
    """One ask in five wordings: search for this event's tickets, keep the cheapest price.

    Nothing in this ask is an address.  What varies next time is which event is being
    searched for, so the parameter is the search — and a routine that asked for a page would
    be asking for something the user never gave and a later occasion could not supply.  The
    cheapest price is what the routine IS and belongs in the framing.

    **Test 1 is what separates this from ``framer-names-a-page-as-a-page-and-invents-no-search``**:
    the search-family parameter that is correct here is exactly the invention that case
    exists to catch, so a framer cannot satisfy both by habit.  The two are named together
    because either alone is passable by a framer that always answers the same way.

    **STORE is EMPTY, and that is the correct report** — one call returns a typed result; it
    moves no machine and writes to no store.

    **PROVENANCE is EMPTY, and that needs its reason stated**: a framing's open fields are an
    identifier and two lines of deliberately generic prose, and the suite's one instrument
    for the invented direction does not transfer to them.  The measurement behind that is in
    ``test_the_symbol_is_the_parameter_and_everything_else_bakes``; it is not re-derived here.

    **Three claims are missing because production already validates them** — at least one
    parameter minted, no two sharing a name, every demonstrated value a literal span of the
    user's own turns — all ``_mints_a_usable_signature``'s, re-rolled until they hold.
    """
    cohort = await framer_eval(
        case_id=SEARCH_ARMS.case_id,
        behaviour=_SEARCH_BEHAVIOUR,
        model=model,
        turns=_SEARCH.turns,
        also_phrased=_SEARCH_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED
    _claim_the_parameter_set(cohort, _SEARCH.parameters)

    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_SEARCH.parameters))


# ── Case 6: two re-suppliable pieces of DIFFERENT types ───────────────────────
#
# Reference output (read at review, never matched):
#   NAME: catalog-checker
#   DESCRIPTION: check whether a book is available in a library catalog
#   PARAMETER catalog_page — the catalog page to check
#   PARAMETER title — the book to look for

_PAGE_AND_TITLE = FramingFixture(
    case_id="frame-two-types-page-and-title",
    turns=(
        "can you watch the library catalog for a book i'm waiting on?",
        "open town-library.example/catalog, find The Glass Harbour, and remember "
        "whether it's available",
    ),
    rendered_input=(
        "can you watch the library catalog for a book i'm waiting on?\n"
        "open town-library.example/catalog, find The Glass Harbour, and remember "
        "whether it's available"
    ),
    parameters=(
        _page_family("catalog page", "catalog"),
        ParameterFamily("title", ("title", "book", "item", "name")),
    ),
    instance_tokens=("glass", "harbour"),
)


@pytest.mark.asyncio
async def test_two_pieces_of_different_types_are_two_parameters(framer_eval: FramerEval):
    """Availability bakes, exactly as it does when the ask names one page — but here the
    thing being looked for is not IN the address, it is looked up ON the page, so the
    page and the book are two re-suppliable pieces of different types.

    The contrast with the page-only case is the whole point: what the framing carries is
    the finding, not the number of things the routine is pointed at."""
    await _run_case(framer_eval, _PAGE_AND_TITLE)


# ── The ported case: a place to go and a thing to look for, in five wordings ──
#
# THE FACTS ARE CONSTANT across the five wordings: every arm names the same catalog page,
# the same book, and asks whether it is available.  What varies is only how a person says
# that.  No arm calls the look-up a search — the thing being looked for is found ON the
# page, and an arm that framed it as a search would be asking for a different piece.
_PAGE_AND_TITLE_PHRASINGS = (
    (
        "could you keep an eye on the library catalog for a book i want?",
        "go to town-library.example/catalog, look up The Glass Harbour, and remember whether "
        "it's available",
    ),
    (
        "i'd like the library catalog watched for a book i'm after",
        "check town-library.example/catalog for The Glass Harbour and save whether it's available",
    ),
    (
        "can you track a book in the library catalog for me?",
        "visit town-library.example/catalog, find The Glass Harbour there, and keep whether "
        "it's available",
    ),
    (
        "keep tabs on the library catalog for a book i'm waiting on",
        "read town-library.example/catalog, look for The Glass Harbour, and store whether "
        "it's available",
    ),
)

PAGE_AND_TITLE_ARMS = PortedArms(
    case_id="framer-mints-both-pieces-when-they-are-different-kinds",
    arms=(_PAGE_AND_TITLE.turns, *_PAGE_AND_TITLE_PHRASINGS),
    carries=("town-library.example/catalog", "The Glass Harbour", "available"),
    never=("search",),
)

_PAGE_AND_TITLE_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when an ask names both a "
    "place to go and a thing to look for on it, Penny mints a parameter for each rather than "
    "baking the thing to look for into the framing."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_place_and_a_thing_to_look_for_are_two_parameters(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: watch this catalog for this book, keep whether it's in.

    Availability is what the routine IS and bakes into the framing, exactly as it does when
    the ask names one page.  What differs is that the thing being looked for is not IN the
    address — it is looked up ON the page — so the catalog and the book are two re-suppliable
    pieces of different kinds and both have to survive into the interface.

    **Test 1 is what separates this from a page-only ask**: a one-parameter draw is correct
    for an ask whose finding is settled by the address alone and wrong here, where a routine
    minting only the page can be pointed at a new catalog but never at a new book.

    **STORE is EMPTY, and that is the correct report** — one call returns a typed result; it
    moves no machine and writes to no store.

    **PROVENANCE is EMPTY, and that needs its reason stated**: a framing's open fields are an
    identifier and two lines of deliberately generic prose, and the suite's one instrument
    for the invented direction does not transfer to them.  The measurement behind that is in
    ``test_the_symbol_is_the_parameter_and_everything_else_bakes``; it is not re-derived here.

    **Three claims are missing because production already validates them** — at least one
    parameter minted, no two sharing a name, every demonstrated value a literal span of the
    user's own turns — all ``_mints_a_usable_signature``'s, re-rolled until they hold.
    """
    cohort = await framer_eval(
        case_id=PAGE_AND_TITLE_ARMS.case_id,
        behaviour=_PAGE_AND_TITLE_BEHAVIOUR,
        model=model,
        turns=_PAGE_AND_TITLE.turns,
        also_phrased=_PAGE_AND_TITLE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED
    _claim_the_parameter_set(cohort, _PAGE_AND_TITLE.parameters)

    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_PAGE_AND_TITLE.parameters))


# ── Case 7: two of the SAME type, where the list temptation is strongest ──────
#
# Reference output (read at review, never matched):
#   NAME: stock-tracker
#   DESCRIPTION: track each given stock's share price
#   PARAMETER first_ticker — the first stock symbol to track
#   PARAMETER second_ticker — the second stock symbol to track

_TWO_SYMBOLS = FramingFixture(
    case_id="frame-two-same-type-symbols",
    turns=(
        "can you keep an eye on a couple of stocks for me?",
        "look up VLT and MERI, find each share price, and remember them",
    ),
    rendered_input=(
        "can you keep an eye on a couple of stocks for me?\n"
        "look up VLT and MERI, find each share price, and remember them"
    ),
    parameters=(_ordinal_family("first ticker", 1), _ordinal_family("second ticker", 2, "other")),
    instance_tokens=("vlt", "meri"),
)


@pytest.mark.asyncio
async def test_two_symbols_of_the_same_type_stay_two_scalar_parameters(framer_eval: FramerEval):
    """Two things of the SAME type in one ask are where a single list parameter is most
    tempting — and a list is not a parameter: what a user says fills one whole.  The
    share price bakes into the framing exactly as it does for one symbol, and the count
    lives in the parameters rather than in a description that promises "two"."""
    await _run_case(framer_eval, _TWO_SYMBOLS)


# ── The ported case: three of a kind, in five wordings ────────────────────────
#
# This case has no legacy ancestor.  It is the same-kind ask at THREE, where the list is
# more tempting than it is at two — three scalars is where a description that promises "the
# news sites" starts to look like the tidier interface — and nothing has measured whether a
# framer that keeps two apart keeps three apart.
#
# The ordinal families here are the shared breadth at three positions, and the second does
# NOT take the two-of-a-kind cases' "other": among two things "the other one" identifies one
# of them, and among three it identifies nothing, so keeping it would let a draw that told
# only two of the three apart answer the second family off a word that says nothing about
# which page it means.
_THREE_SOURCES_FAMILIES = (
    _ordinal_family("first source", 1),
    _ordinal_family("second source", 2),
    _ordinal_family("third source", 3),
)

# THE FACTS ARE CONSTANT across the five wordings: every arm names all three front pages and
# asks for each site's top headline.
_THREE_SOURCES_TURNS = (
    "hey could you keep an eye on the morning headlines for me",
    "read citydesk.example/front, harborpost.example/front and riverchronicle.example/front, "
    "and remember each site's top headline",
)

_THREE_SOURCES_PHRASINGS = (
    (
        "could you watch the morning news for me",
        "open citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, find each site's top headline, and remember them",
    ),
    (
        "i'd like the morning headlines followed",
        "check citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, and keep each one's top headline",
    ),
    (
        "can you keep track of the morning headlines?",
        "look at citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, and save the top headline from each",
    ),
    (
        "keep up with the morning headlines for me",
        "go to citydesk.example/front, harborpost.example/front and "
        "riverchronicle.example/front, get each site's top headline, and store them",
    ),
)

THREE_SOURCES_ARMS = PortedArms(
    case_id="framer-keeps-three-of-a-kind-as-three-parameters",
    arms=(_THREE_SOURCES_TURNS, *_THREE_SOURCES_PHRASINGS),
    carries=(
        "citydesk.example/front",
        "harborpost.example/front",
        "riverchronicle.example/front",
        "top headline",
    ),
)

_THREE_SOURCES_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_FRAME_AGENT_NAME} micro-context, when one ask points a "
    "routine at three things of the same kind, Penny mints three distinct scalar parameters "
    "rather than folding them into one list."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_three_of_a_kind_stay_three_distinct_parameters(
    framer_eval: FramerEval, model: str
) -> None:
    """One ask in five wordings: read these three front pages, keep each one's top headline.

    The same-kind behaviour at three.  Every page has to be re-suppliable and they have to be
    tellable apart, so a correct draw mints three scalars distinguished by position; a single
    list parameter, or two scalars and a description that promises the rest, has lost a piece
    the user named.

    **Test 1 is what separates this from the two-of-a-kind case**: a two-parameter draw is
    the correct answer there and the wrong one here.  Whether the behaviour survives the
    extra piece is the open question — the two-of-a-kind failure was a fold into one list,
    and nothing has measured whether the fold returns when there is one more thing to fold.

    **STORE is EMPTY, and that is the correct report** — one call returns a typed result; it
    moves no machine and writes to no store.

    **PROVENANCE is EMPTY, and that needs its reason stated**: a framing's open fields are an
    identifier and two lines of deliberately generic prose, and the suite's one instrument
    for the invented direction does not transfer to them.  The measurement behind that is in
    ``test_the_symbol_is_the_parameter_and_everything_else_bakes``; it is not re-derived here.

    **Three claims are missing because production already validates them** — at least one
    parameter minted, no two sharing a name, every demonstrated value a literal span of the
    user's own turns — all ``_mints_a_usable_signature``'s, re-rolled until they hold.
    """
    cohort = await framer_eval(
        case_id=THREE_SOURCES_ARMS.case_id,
        behaviour=_THREE_SOURCES_BEHAVIOUR,
        model=model,
        turns=_THREE_SOURCES_TURNS,
        also_phrased=_THREE_SOURCES_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED
    _claim_the_parameter_set(cohort, _THREE_SOURCES_FAMILIES)

    # STORE — EMPTY; see the docstring.

    # PROVENANCE — EMPTY; see the docstring.

    _measure_the_draw(cohort, positions=len(_THREE_SOURCES_FAMILIES))


# EVERY ported case's arms, for the deterministic probe in ``make check`` — one place, so the
# probe and the live runs can never be checking two different sets of wordings.
PORTED_ARMS = (
    TICKER_ARMS,
    PAGE_ONLY_ARMS,
    TWO_SOURCES_ARMS,
    SINGLE_TURN_ARMS,
    SEARCH_ARMS,
    PAGE_AND_TITLE_ARMS,
    THREE_SOURCES_ARMS,
)


# Every case, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (
    _AVAILABILITY,
    _TWO_SOURCES,
    _TICKER,
    _SINGLE_TURN,
    _SEARCH,
    _PAGE_AND_TITLE,
    _TWO_SYMBOLS,
)
