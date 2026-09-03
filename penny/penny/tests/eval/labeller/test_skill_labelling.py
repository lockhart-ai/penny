"""Live-model contract for the run-end LEAF LABELLER (#1828, the #1824 inversion).

The labeller has ONE job: given the demonstrated routine's spots, write a semantic name
and one line of what belongs there for EVERY spot offered.  It judges nothing — no
provenance verdict, no routine name, no routine description.  The interface half (name ·
description · parameters, decided from the user's ask alone) is the framer's, a separate
draw that never sees this one's evidence.

Six cases, all the same look-up → extract → remember shape, varying topic, leaf
structure and conversation length, so what moves between them is the NAMING and nothing
else:

* ``leaf-topic-availability`` — the canonical shape on a different semantics
  (availability, not price): do the names come from THIS demonstration?
* ``leaf-two-sources-distinct-names`` — two spots on one argument: are they told apart?
* ``leaf-shared-spot-one-name`` — one spot filling two sites: ONE line covering both?
* ``leaf-single-turn-teach`` — no elicit round: the conversation block at its minimum.
* ``leaf-search-not-page`` — the look-up is a text search: named as a search, not a page?
* ``namer-tells-two-sources-apart`` — the slot's CANONICAL case (#2006): the two-sources
  demonstration in five wordings over one ledger, pooled into a cohort of fifteen and
  claimed against ``docs/eval-case-design.md`` rather than the per-spot scorer below.

Each case is a fixture LEDGER, and its input document is rendered from that ledger by
the shipped ``distill_steps`` + ``build_naming_content`` — never hand-written — so the
draw reads exactly what production would render.  ``rendered_input`` is that document,
pinned byte-for-byte by a deterministic drift probe in ``make check`` (see
``tests/test_eval_harness.py``): a fixture that drifts from the pair it claims is a case
measuring nothing, and it must fail before any GPU time, not after.

Scoring is per offered spot — a line came back · its name hardens to a usable binding
key · it is not the arg name handed back · its description says what belongs there —
plus each case's own structural claim, with every drawn label carried ADVISORY so a
reader sees what the model committed to.  Whether a name is WELL judged is read at
review against the reference outputs on #1828; no scorer fakes that.

All content is synthetic.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    DemoCall,
    DemoTurn,
    LabellerEval,
    label_name_field,
    label_says_field,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    Consequence,
    SampleObservation,
    SpecCategory,
    output_field,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.micro_context import slug_parameter_name

pytestmark = pytest.mark.eval

_FAMILY = "skill-labelling"


def _user(text: str) -> DemoTurn:
    """One user turn of the conversation that led to the routine."""
    return (PennyConstants.MessageDirection.INCOMING, text)


def _penny(text: str) -> DemoTurn:
    """One assistant turn — the elicit round, which is what makes the next user turn a
    demonstration rather than an ask."""
    return (PennyConstants.MessageDirection.OUTGOING, text)


class LabellingFixture(NamedTuple):
    """One agreed case: the ledger that produces its input document, the document the
    shipped renderer must produce from it, and what the draw is scored on.

    ``leaves`` names each offered spot by its DEMONSTRATED VALUE, because the semantic
    name is the model's to choose and the arg-derived one is what the case is asking it
    to improve on.  ``distinct_names`` and ``shared_spot`` are the two structural claims
    only some cases make."""

    case_id: str
    conversation: tuple[DemoTurn, ...]
    utterance: str
    calls: tuple[DemoCall, ...]
    target: str
    leaves: tuple[str, ...]
    rendered_input: str
    distinct_names: tuple[tuple[str, str], ...] = ()
    shared_spot: str = ""


async def _run_case(labeller_eval: LabellerEval, fixture: LabellingFixture) -> None:
    """Drive one case's fixture through the labeller.  Every case is report-only: the
    thresholds are the code owner's to set once the first numbers are read."""
    await labeller_eval(
        case_id=fixture.case_id,
        utterance=fixture.utterance,
        conversation=fixture.conversation,
        calls=fixture.calls,
        target=fixture.target,
        leaves=fixture.leaves,
        distinct_names=fixture.distinct_names,
        shared_spot=fixture.shared_spot,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


_ELICIT = (
    "i don't have a routine for that yet — can you walk me through it once? what "
    "should i read, what am i looking for, what should i remember?"
)
_CONVERSATION_HEADING = (
    "Conversation that led to the construction of this routine "
    "(the LAST user turn is the one that demonstrated it):"
)
_PLACEHOLDER_HEADING = "Placeholders (each currently named after the tool arg it fills):"


# ── Case 1: the canonical shape, on availability rather than price ─────────────

_AVAILABILITY = LabellingFixture(
    case_id="leaf-topic-availability",
    conversation=(
        _user(
            "can you keep an eye on bookbarn.example/atlas-of-clouds and let me know "
            "when it's back in stock"
        ),
        _penny(_ELICIT),
    ),
    utterance=(
        "go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and remember that"
    ),
    calls=(
        (
            "browse",
            {
                "queries": ["bookbarn.example/atlas-of-clouds"],
                "extract": "whether it is in stock",
            },
            "You opened the atlas of clouds page (browse result)\nin stock",
            True,
        ),
        (
            "collection_write",
            {
                "memory": "book-availability",
                "entries": [{"key": "atlas of clouds availability", "content": "in stock"}],
            },
            "You saved an entry to book-availability: (collection_write result)\nWrote 1 entry.",
            True,
        ),
    ),
    target="book-availability",
    leaves=(
        "bookbarn.example/atlas-of-clouds",
        "whether it is in stock",
        "atlas of clouds availability",
        "book-availability",
    ),
    rendered_input=(
        f"{_CONVERSATION_HEADING}\n"
        "user: can you keep an eye on bookbarn.example/atlas-of-clouds and let me know "
        "when it's back in stock\n"
        f"penny: {_ELICIT}\n"
        "user: go to bookbarn.example/atlas-of-clouds, check whether it's in stock, and "
        "remember that\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}], extract={extract})\n"
        "2. collection_write(memory={memory}, entries=["
        "{'key': {key}, 'content': the value from step 1}])\n"
        "\n"
        f"{_PLACEHOLDER_HEADING}\n"
        "- queries: fills browse.queries[0]; "
        "demonstrated value: 'bookbarn.example/atlas-of-clouds'\n"
        "- extract: fills browse.extract; demonstrated value: 'whether it is in stock'\n"
        "- memory: fills collection_write.memory; demonstrated value: 'book-availability'\n"
        "- key: fills collection_write.entries[0].key; "
        "demonstrated value: 'atlas of clouds availability'"
    ),
)


@pytest.mark.asyncio
async def test_every_spot_is_named_for_this_demonstration(labeller_eval: LabellerEval):
    """The canonical shape with different semantics: the routine watches AVAILABILITY,
    not price.  Every spot must be named for what it is in THIS routine — a labeller
    working from a memorised price template would name the extract spot for a price
    nobody mentioned."""
    await _run_case(labeller_eval, _AVAILABILITY)


# ── Case 2: two spots on one argument must draw two names ─────────────────────

_TWO_SOURCES = LabellingFixture(
    case_id="leaf-two-sources-distinct-names",
    conversation=(
        _user("hey could you keep an eye on the morning headlines for me"),
        _penny(
            "i don't have a routine for that yet — walk me through it once? what should "
            "i read and what should i save?"
        ),
    ),
    utterance=(
        "read citydesk.example/front and harborpost.example/front, and remember each "
        "site's top headline"
    ),
    calls=(
        (
            "browse",
            {
                "queries": ["citydesk.example/front", "harborpost.example/front"],
                "extract": "the top headline",
            },
            "You read both front pages (browse result)\nharbour vote passes",
            True,
        ),
        (
            "collection_write",
            {
                "memory": "headlines",
                "entries": [{"key": "morning headlines", "content": "harbour vote passes"}],
            },
            "You saved an entry to headlines: (collection_write result)\nWrote 1 entry.",
            True,
        ),
    ),
    target="headlines",
    leaves=(
        "citydesk.example/front",
        "harborpost.example/front",
        "the top headline",
        "morning headlines",
        "headlines",
    ),
    distinct_names=(("citydesk.example/front", "harborpost.example/front"),),
    rendered_input=(
        f"{_CONVERSATION_HEADING}\n"
        "user: hey could you keep an eye on the morning headlines for me\n"
        "penny: i don't have a routine for that yet — walk me through it once? what "
        "should i read and what should i save?\n"
        "user: read citydesk.example/front and harborpost.example/front, and remember "
        "each site's top headline\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}, {queries-2}], extract={extract})\n"
        "2. collection_write(memory={memory}, entries=["
        "{'key': {key}, 'content': the value from step 1}])\n"
        "\n"
        f"{_PLACEHOLDER_HEADING}\n"
        "- queries: fills browse.queries[0]; demonstrated value: 'citydesk.example/front'\n"
        "- queries-2: fills browse.queries[1]; demonstrated value: 'harborpost.example/front'\n"
        "- extract: fills browse.extract; demonstrated value: 'the top headline'\n"
        "- memory: fills collection_write.memory; demonstrated value: 'headlines'\n"
        "- key: fills collection_write.entries[0].key; demonstrated value: 'morning headlines'"
    ),
)


@pytest.mark.asyncio
async def test_two_sources_draw_distinct_names(labeller_eval: LabellerEval):
    """Two spots on the same argument are two spots: each takes its own value every
    run, so one name for both loses which site is which.  A labeller that calls them
    both `news_page` has collapsed a distinction the routine depends on."""
    await _run_case(labeller_eval, _TWO_SOURCES)


# ── The ported case: two sources, one demonstration, five wordings ────────────
#
# The survivor is ``leaf-two-sources-distinct-names``, because it states the behaviour and
# its negative direction in one demonstration: every spot is named for what it does in THIS
# routine, and the two sites — which take their own value every run — must not collapse
# onto one name.  Its sibling ``leaf-shared-spot-one-name`` cannot be the survivor: a split
# shared spot keys a line to a name nobody offered, which the production validator refuses
# and re-rolls, so its whole claim runs 15/15 by construction.
#
# THE ARM IS THE DEMONSTRATING UTTERANCE, and the LEDGER is held constant across the five.
# Distillation is deterministic Python over the calls, so identical calls mean identical
# spots under identical current names on every arm — which is what makes one ``by_value``
# map, one offered set and one set of claims legal over the pool.  What moves is the words
# the user used to demonstrate it: both addresses appear verbatim in every wording, and so
# does the thing being kept, because the claims hinge on them.
_TWO_SOURCES_PHRASINGS = (
    "have a look at citydesk.example/front and harborpost.example/front, and save the "
    "top headline from each",
    "open citydesk.example/front and harborpost.example/front and keep the lead story from both",
    "check citydesk.example/front and harborpost.example/front, then remember whichever "
    "headline is at the top of each",
    "go to citydesk.example/front and harborpost.example/front and note down the top "
    "headline on each one",
)

# The case's id, its five arms and the two spots it tells apart, named at module level so
# the deterministic probe in ``make check`` can hold every arm against the ledger it claims
# — one offered set, one map home — before any GPU time is spent.
TWO_SOURCES_CASE_ID = "namer-tells-two-sources-apart"
TWO_SOURCES_ARMS = (_TWO_SOURCES.utterance, *_TWO_SOURCES_PHRASINGS)

# The two spots on the SAME argument, by the CURRENT (argument-derived) name distillation
# gives each — the anchor the input document renders verbatim and the key every field is
# filed under.  Named here rather than spelled at each claim, so a ledger edit that renamed
# a spot breaks the probe rather than quietly voiding a claim.
_FIRST_SOURCE = "queries"
_SECOND_SOURCE = "queries-2"
OFFERED_SPOTS = (_FIRST_SOURCE, _SECOND_SOURCE, "extract", "memory", "key")

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename; this is
# the contract, and it renders above every number in the report.
_TWO_SOURCES_BEHAVIOUR = (
    f"In the {PennyConstants.SKILL_NAMING_AGENT_NAME} micro-context, when a demonstrated "
    "routine reads two different pages into one argument, Penny gives every spot its own "
    "name for what it supplies in THIS routine — never the argument's own name handed back, "
    "and never one name covering both sites."
)


def _drawn_names(sample: SampleObservation) -> dict[str, str]:
    """What the draw called each offered spot, keyed by the spot's current name."""
    return {spot: sample.field(label_name_field(spot)) for spot in OFFERED_SPOTS}


def _every_name_hardens_to_a_key(sample: SampleObservation, _world: World) -> Answer:
    """Every spot's name survives the SHIPPED hardener as something a binding can use.

    Imported, never re-implemented: a name becomes a key through ``slug_parameter_name`` at
    instantiation, so what a case calls usable and what production calls a key are one
    definition.  A name that hardens to nothing — punctuation, an empty line after the
    separator — leaves the spot named by nothing, which is the answer OMITTED rather than
    given.

    One claim over all five spots rather than five: every spot here is offered under the
    same contract, so which of them failed is the rationale's job.  (An extract instruction
    naming several DIFFERENT things is the case that needs one claim each — the things
    degrade one at a time and are not interchangeable.)"""
    unusable = [
        spot for spot, name in _drawn_names(sample).items() if not slug_parameter_name(name)
    ]
    return not unusable, f"hardens to nothing: {unusable}"


def _no_spot_was_handed_back_its_own_name(sample: SampleObservation, _world: World) -> Answer:
    """No spot came back named after the tool argument it fills.

    The document offers each spot under its argument-derived name, so a draw answering
    ``queries`` with ``queries`` has described the spot rather than named it — the answer is
    OMITTED, and the routine is left exactly as unreadable as it was."""
    echoed = [
        spot
        for spot, name in _drawn_names(sample).items()
        if slug_parameter_name(name) == slug_parameter_name(spot)
    ]
    return not echoed, f"echoed the argument name: {echoed}"


def _every_spot_says_what_belongs_there(sample: SampleObservation, _world: World) -> Answer:
    """Every spot's line carries the one thing to supply there each run.

    The description is the grammar's one optional field, so a line that stops after its name
    is well-formed and reaches the caller — the spot is named and nobody can tell what goes
    in it.  The instruction asks for both halves; a blank omits one."""
    silent = [spot for spot in OFFERED_SPOTS if not sample.field(label_says_field(spot)).strip()]
    return not silent, f"no description: {silent}"


def _the_two_sources_drew_different_names(sample: SampleObservation, _world: World) -> Answer:
    """The two sites are two spots, and they read back as two.

    The demonstration SUPPLIES the distinction — each site takes its own value every run —
    so a draw calling both of them ``news_page`` has omitted it, and at run time one name
    for two spots cannot say which site is which.  Compared HARDENED, because that is what
    the names become: ``News Page`` and ``news_page`` are one key, not two."""
    first, second = _drawn_names(sample)[_FIRST_SOURCE], _drawn_names(sample)[_SECOND_SOURCE]
    hardened = (slug_parameter_name(first), slug_parameter_name(second))
    return hardened[0] != hardened[1], f"both drew {hardened[0]!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_two_sources_are_named_apart_however_the_demonstration_is_worded(
    labeller_eval: LabellerEval, model: str
) -> None:
    """One demonstration in five wordings: read two front pages, keep the top headline of
    each.

    **The LANDED category is empty for this case, and it is CLOSED UPSTREAM rather than
    unrun.**  A labelling draw's closed field is WHICH spots were named, and production
    validates exactly that: ``_labels_every_spot`` accepts a draw only when every offered
    spot has one well-formed line and nothing else does, re-rolling while it does not.  So
    "every spot got a line" runs 15/15 by construction and would measure the validator.
    What is left to claim is what those lines SAY, which is the provenance block below.

    **The STORE category is empty by construction.**  A micro-context is one call that
    returns a typed result — it writes to no store — so there is nothing for a store claim
    to read.

    **One further claim is missing for the same upstream reason**: that a value filling two
    argument sites draws exactly ONE line.  A draw that split it either repeats the spot's
    current name or keys a line to a name nobody offered, and the coverage rule refuses
    both, so no split ever reaches an accepted draw.

    **And the provenance block below carries only ONE of its two directions, which the design
    calls half a check — so here is the other half's reason.**  *Nothing invented* has no
    legal instrument for this shape.  The suite's one probe (``unsourced_specifics``) reads
    URLs, numbers and capitalised name phrases, which is the right instrument for a value
    lifted off a page and the wrong one for an identifier and a line of generic prose:
    measured against this case's own document, it reads a Title-Cased but perfectly correct
    label — ``First Site — The First News Front Page To Read`` — as four inventions, while
    the failure the ticket actually names (a spot named for what the LAST routine did, say
    ``product_price`` on a headline routine) carries no capital, no digit and no url and is
    invisible to it.  A check that fails a correct run for a cosmetic reason and misses the
    thing it is for is not an assertion, so it is left out rather than counted.
    """
    cohort = await labeller_eval(
        case_id=TWO_SOURCES_CASE_ID,
        behaviour=_TWO_SOURCES_BEHAVIOUR,
        model=model,
        utterance=_TWO_SOURCES.utterance,
        also_demonstrated=_TWO_SOURCES_PHRASINGS,
        conversation=_TWO_SOURCES.conversation,
        calls=_TWO_SOURCES.calls,
        target=_TWO_SOURCES.target,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — empty, and closed upstream; see the docstring.
    # STORE — empty by construction; see the docstring.

    # PROVENANCE — the OPEN fields, which for this shape are the whole typed result.  All
    # four are the *nothing omitted* direction: the demonstration offers a spot and asks for
    # a name and a line, and each claim is a way that answer can fail to arrive — nothing
    # usable, the question handed back, no line, or a distinction the demonstration draws
    # collapsed.  The other direction is absent, and the docstring says why.
    cohort.claim(
        "state: every spot's name hardens to a usable binding key",
        _every_name_hardens_to_a_key,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: no spot was handed back its own argument name",
        _no_spot_was_handed_back_its_own_name,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: every spot's line says what belongs there",
        _every_spot_says_what_belongs_there,
        SpecCategory.PROVENANCE,
    )
    cohort.claim(
        "state: the two sources drew different names",
        _the_two_sources_drew_different_names,
        SpecCategory.PROVENANCE,
    )

    # What is MEASURED — what the draw called each spot, one axis per spot.
    #
    # COSMETIC, all five: `first_front_page` and `citydesk_page` leave the same routine
    # behind, so a divergence is a fact about the SYSTEM's naming spread and never about one
    # sample.  This is the framer's own known signature one layer down, and filing it
    # consequential would make almost every sample an outlier.
    #
    # The DESCRIPTION axes are left out: a line of prose is measured by textual spread and
    # this shape has no reply for that machinery to read, so an entropy over free text would
    # report near-total disagreement on every run whatever the draw did.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(
        *(
            output_field(label_name_field(spot), consequence=Consequence.COSMETIC)
            for spot in OFFERED_SPOTS
        )
    )


# ── Case 3: one spot filling two sites draws exactly one name ─────────────────

_SHARED_SPOT = LabellingFixture(
    case_id="leaf-shared-spot-one-name",
    conversation=(
        _user("can you track a stock for me and tell me when it moves"),
        _penny(
            "i can learn that — show me once: what should i look up and what should i remember?"
        ),
    ),
    utterance="look up VLT, find the share price, and remember it under VLT",
    calls=(
        (
            "browse",
            {"queries": ["VLT"], "extract": "the share price"},
            "You looked up VLT (browse result)\n$18.40",
            True,
        ),
        (
            "collection_write",
            {"memory": "stock-prices", "entries": [{"key": "VLT", "content": "$18.40"}]},
            "You saved an entry to stock-prices: (collection_write result)\nWrote 1 entry.",
            True,
        ),
    ),
    target="stock-prices",
    leaves=("VLT", "the share price", "stock-prices"),
    shared_spot="VLT",
    rendered_input=(
        f"{_CONVERSATION_HEADING}\n"
        "user: can you track a stock for me and tell me when it moves\n"
        "penny: i can learn that — show me once: what should i look up and what should "
        "i remember?\n"
        "user: look up VLT, find the share price, and remember it under VLT\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}], extract={extract})\n"
        "2. collection_write(memory={memory}, entries=["
        "{'key': {queries}, 'content': the value from step 1}])\n"
        "\n"
        f"{_PLACEHOLDER_HEADING}\n"
        "- queries: fills browse.queries[0] and collection_write.entries[0].key; "
        "demonstrated value: 'VLT'\n"
        "- extract: fills browse.extract; demonstrated value: 'the share price'\n"
        "- memory: fills collection_write.memory; demonstrated value: 'stock-prices'"
    ),
)


@pytest.mark.asyncio
async def test_a_shared_spot_draws_one_name_covering_both_uses(labeller_eval: LabellerEval):
    """The same value at two sites is structurally ONE spot — the ticker is both what
    the routine looks up and what it files the price under.  The contract is one line
    whose name covers both uses; splitting it invents a spot nobody offered."""
    await _run_case(labeller_eval, _SHARED_SPOT)


# ── Case 4: the conversation block at its minimum — one direct instruction ────

_SINGLE_TURN = LabellingFixture(
    case_id="leaf-single-turn-teach",
    conversation=(),
    utterance="go to weather.example/lisbon, find today's high temperature, and remember it",
    calls=(
        (
            "browse",
            {"queries": ["weather.example/lisbon"], "extract": "today's high temperature"},
            "You opened the lisbon forecast (browse result)\n24 degrees",
            True,
        ),
        (
            "collection_write",
            {
                "memory": "weather",
                "entries": [{"key": "lisbon high temperature", "content": "24 degrees"}],
            },
            "You saved an entry to weather: (collection_write result)\nWrote 1 entry.",
            True,
        ),
    ),
    target="weather",
    leaves=(
        "weather.example/lisbon",
        "today's high temperature",
        "lisbon high temperature",
        "weather",
    ),
    rendered_input=(
        f"{_CONVERSATION_HEADING}\n"
        "user: go to weather.example/lisbon, find today's high temperature, and remember it\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}], extract={extract})\n"
        "2. collection_write(memory={memory}, entries=["
        "{'key': {key}, 'content': the value from step 1}])\n"
        "\n"
        f"{_PLACEHOLDER_HEADING}\n"
        "- queries: fills browse.queries[0]; demonstrated value: 'weather.example/lisbon'\n"
        '- extract: fills browse.extract; demonstrated value: "today\'s high temperature"\n'
        "- memory: fills collection_write.memory; demonstrated value: 'weather'\n"
        "- key: fills collection_write.entries[0].key; "
        "demonstrated value: 'lisbon high temperature'"
    ),
)


@pytest.mark.asyncio
async def test_a_single_turn_teach_still_names_every_spot(labeller_eval: LabellerEval):
    """No elicit round — one direct instruction is the whole conversation.  The
    conversation block at its minimum still has to carry enough for every spot to be
    named for what it is."""
    await _run_case(labeller_eval, _SINGLE_TURN)


# ── Case 5: the look-up is a search, not a page ───────────────────────────────

_SEARCH = LabellingFixture(
    case_id="leaf-search-not-page",
    conversation=(
        _user("can you keep an eye on ticket prices for aurora fest?"),
        _penny(
            "i don't have a routine for that yet — show me once: what should i look up "
            "and what should i remember?"
        ),
    ),
    utterance="search for aurora fest tickets, find the cheapest ticket price, and remember it",
    calls=(
        (
            "browse",
            {"queries": ["aurora fest tickets"], "extract": "the cheapest ticket price"},
            "You searched for aurora fest tickets (browse result)\n$42",
            True,
        ),
        (
            "collection_write",
            {
                "memory": "ticket-prices",
                "entries": [{"key": "aurora fest ticket price", "content": "$42"}],
            },
            "You saved an entry to ticket-prices: (collection_write result)\nWrote 1 entry.",
            True,
        ),
    ),
    target="ticket-prices",
    leaves=(
        "aurora fest tickets",
        "the cheapest ticket price",
        "aurora fest ticket price",
        "ticket-prices",
    ),
    rendered_input=(
        f"{_CONVERSATION_HEADING}\n"
        "user: can you keep an eye on ticket prices for aurora fest?\n"
        "penny: i don't have a routine for that yet — show me once: what should i look "
        "up and what should i remember?\n"
        "user: search for aurora fest tickets, find the cheapest ticket price, and "
        "remember it\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}], extract={extract})\n"
        "2. collection_write(memory={memory}, entries=["
        "{'key': {key}, 'content': the value from step 1}])\n"
        "\n"
        f"{_PLACEHOLDER_HEADING}\n"
        "- queries: fills browse.queries[0]; demonstrated value: 'aurora fest tickets'\n"
        "- extract: fills browse.extract; demonstrated value: 'the cheapest ticket price'\n"
        "- memory: fills collection_write.memory; demonstrated value: 'ticket-prices'\n"
        "- key: fills collection_write.entries[0].key; "
        "demonstrated value: 'aurora fest ticket price'"
    ),
)


@pytest.mark.asyncio
async def test_a_search_spot_is_named_as_a_search(labeller_eval: LabellerEval):
    """The look-up is a text search, not a url.  A spot named `product_page` here
    describes what the last case did, not what this one does — the demonstrated value
    is the evidence for what kind of thing goes in the spot."""
    await _run_case(labeller_eval, _SEARCH)


# Every case, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (_AVAILABILITY, _TWO_SOURCES, _SHARED_SPOT, _SINGLE_TURN, _SEARCH)
