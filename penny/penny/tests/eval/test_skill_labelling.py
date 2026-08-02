"""Live-model contract for the run-end LEAF LABELLER (#1828, the #1824 inversion).

The labeller has ONE job: given the demonstrated routine's spots, write a semantic name
and one line of what belongs there for EVERY spot offered.  It judges nothing — no
provenance verdict, no routine name, no routine description.  The interface half (name ·
description · parameters, decided from the user's ask alone) is the framer's, a separate
draw that never sees this one's evidence.

Five cases, all the same look-up → extract → remember shape, varying topic, leaf
structure and conversation length, so what moves between them is the NAMING and nothing
else:

* ``leaf-topic-availability`` — the canonical shape on a different semantics
  (availability, not price): do the names come from THIS demonstration?
* ``leaf-two-sources-distinct-names`` — two spots on one argument: are they told apart?
* ``leaf-shared-spot-one-name`` — one spot filling two sites: ONE line covering both?
* ``leaf-single-turn-teach`` — no elicit round: the conversation block at its minimum.
* ``leaf-search-not-page`` — the look-up is a text search: named as a search, not a page?

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
from penny.tests.eval.conftest import DemoCall, DemoTurn, LabellerEval

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
