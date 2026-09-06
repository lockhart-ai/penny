"""Live-model contract for the run-end LEAF LABELLER (#1828, the #1824 inversion).

The labeller has ONE job: given the demonstrated routine's spots, write a semantic name
and one line of what belongs there for EVERY spot offered.  It judges nothing — no
provenance verdict, no routine name, no routine description.  The interface half (name ·
description · parameters, decided from the user's ask alone) is the framer's, a separate
draw that never sees this one's evidence.

The file holds two sets, and they are read differently.

**The INLINE cases** came first (#1828) and are scored per offered spot by the runner's own
scorer.  They vary topic, leaf structure and conversation length so that what moves between
them is the NAMING and nothing else: ``leaf-topic-availability`` (the canonical shape on
availability rather than price) · ``leaf-two-sources-distinct-names`` (two spots on one
argument) · ``leaf-shared-spot-one-name`` (one spot filling two sites) ·
``leaf-single-turn-teach`` (no elicit round) · ``leaf-search-not-page`` (the look-up is a
search).

**The PORTED set** (#2004's canonical set, #2058) is the one the epic reads.  Five cases, each
one demonstration in five wordings pooled into a cohort of fifteen, claimed against
``docs/eval-case-design.md`` rather than the per-spot scorer: ``namer-tells-two-sources-apart``
· ``namer-names-a-search-spot-as-a-search`` · ``namer-names-an-availability-spot-for-this-
routine`` · ``namer-names-every-spot-from-a-single-turn-teach`` ·
``namer-names-every-spot-in-a-longer-routine``.  Four of them make the SAME three claims on
different demonstrations — they exercise, they do not newly assert — and each says so in its
own docstring rather than dressing it up.  ``leaf-shared-spot-one-name`` has no ported
counterpart: a split shared spot keys a line to a name nobody offered, which the production
validator refuses and re-rolls, so the claim would run 15/15 by construction.

Each case is a fixture LEDGER, and its input document is rendered from that ledger by
the shipped ``distill_steps`` + ``build_naming_content`` — never hand-written — so the
draw reads exactly what production would render.  ``rendered_input`` is that document,
pinned byte-for-byte by a deterministic drift probe in ``make check`` (see
``tests/test_eval_harness.py``): a fixture that drifts from the pair it claims is a case
measuring nothing, and it must fail before any GPU time, not after.

An INLINE case is scored per offered spot — a line came back · its name hardens to a
usable binding key · it is not the arg name handed back · its description says what
belongs there — plus that case's own structural claim, with every drawn label carried
ADVISORY.  A PORTED case makes the same three statements as COHORT CLAIMS over its
fifteen samples, and the coverage half is not among them: production's
``_labels_every_spot`` validates it and re-rolls, so a claim over it would measure the
validator.  Whether a name is WELL judged is read at review against the reference
outputs on #1828; no scorer fakes that.

All content is synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from penny.tests.eval.utils.assertions import Answer, Cohort, WorldClaim
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


# ── The LONGER routine: four steps, two reads, a plugin tool and a log ────────
#
# Every other ledger in this file is ``browse -> collection_write``, which is the shortest
# routine that can exist and the only shape the labeller has ever been measured on.  #1824's
# own finding is that the failure it replaced "compounded with every extra leaf", so a set
# whose longest routine is two steps cannot see the compounding at all.
#
# This one is four steps and seven spots: two reads (the second a PLUGIN tool, which is the
# case nothing in the codebase can enumerate in advance), one write, and a ``log_append``
# whose free-text line is a spot like any other.  Every shape is one production ships; all
# content is synthetic.
_LONGER_ROUTINE = LabellingFixture(
    case_id="namer-names-every-spot-in-a-longer-routine",
    conversation=(
        _user("can you keep me posted on the depot deliveries?"),
        _penny(
            "i don't have a routine for that yet — walk me through it once? what should i "
            "check, and what should i write down?"
        ),
    ),
    utterance=(
        "check depotline.example/schedule for the next delivery window, look at what's open "
        "on the harbour refit project, save the window in depot-deliveries, and add a note "
        "to depot-checks that you looked"
    ),
    calls=(
        (
            "browse",
            {
                "queries": ["depotline.example/schedule"],
                "extract": "the next delivery window",
            },
            "You opened the depot schedule (browse result)\ntuesday 09:00",
            True,
        ),
        (
            "list_tasks",
            {"project_name": "harbour refit"},
            "You listed the harbour refit tasks (list_tasks result)\n3 still open",
            True,
        ),
        (
            "collection_write",
            {
                "memory": "depot-deliveries",
                "entries": [{"key": "next delivery window", "content": "tuesday 09:00"}],
            },
            "You saved an entry to depot-deliveries: (collection_write result)\nWrote 1 entry.",
            True,
        ),
        (
            "log_append",
            {
                "memory": "depot-checks",
                "content": "checked the depot schedule and the refit tasks",
            },
            "You added an entry to depot-checks: (log_append result)\nAppended 1 entry.",
            True,
        ),
    ),
    target="depot-deliveries",
    leaves=(
        "depotline.example/schedule",
        "the next delivery window",
        "harbour refit",
        "depot-deliveries",
        "next delivery window",
        "depot-checks",
        "checked the depot schedule and the refit tasks",
    ),
    rendered_input=(
        f"{_CONVERSATION_HEADING}\n"
        "user: can you keep me posted on the depot deliveries?\n"
        "penny: i don't have a routine for that yet — walk me through it once? what should "
        "i check, and what should i write down?\n"
        "user: check depotline.example/schedule for the next delivery window, look at "
        "what's open on the harbour refit project, save the window in depot-deliveries, "
        "and add a note to depot-checks that you looked\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}], extract={extract})\n"
        "2. list_tasks(project_name={project_name})\n"
        "3. collection_write(memory={memory}, entries=["
        "{'key': {key}, 'content': the value from step 1}])\n"
        "4. log_append(memory={memory-2}, content={content})\n"
        "\n"
        f"{_PLACEHOLDER_HEADING}\n"
        "- queries: fills browse.queries[0]; demonstrated value: 'depotline.example/schedule'\n"
        "- extract: fills browse.extract; demonstrated value: 'the next delivery window'\n"
        "- project_name: fills list_tasks.project_name; demonstrated value: 'harbour refit'\n"
        "- memory: fills collection_write.memory; demonstrated value: 'depot-deliveries'\n"
        "- key: fills collection_write.entries[0].key; "
        "demonstrated value: 'next delivery window'\n"
        "- memory-2: fills log_append.memory; demonstrated value: 'depot-checks'\n"
        "- content: fills log_append.content; "
        "demonstrated value: 'checked the depot schedule and the refit tasks'"
    ),
)


# ── The PORTED set: five cases, each one demonstration in five wordings ───────
#
# THE ARM IS THE DEMONSTRATING UTTERANCE, and the LEDGER is held constant across the five.
# Distillation is deterministic Python over the calls, so identical calls mean identical spots
# under identical current names on every arm — which is what makes one offered set and one set
# of claims legal over the pool.  What moves is the words the user used to demonstrate it, and
# every demonstrated value a case's claims lean on appears verbatim in all five.

# The two spots on the SAME argument of the two-source ledger, by the CURRENT
# (argument-derived) name distillation gives each — the anchor the input document renders
# verbatim and the key every field is filed under.  Named here rather than spelled at the
# claim, so a ledger edit that renamed a spot breaks the probe rather than quietly voiding it.
_FIRST_SOURCE = "queries"
_SECOND_SOURCE = "queries-2"


class PortedNamingCase(NamedTuple):
    """One ported case: which ledger it drives, in which five wordings, over which spots.

    ``offered`` is what the ledger's distillation hands the draw — the CURRENT
    (argument-derived) names, which the input document renders verbatim and every measured
    field is keyed by.  Stated here rather than derived at run time so a ledger edit that
    renamed a spot breaks the ``make check`` probe rather than quietly voiding a claim.

    ``anchors`` are the demonstrated values every arm must still name.  Five wordings of one
    demonstration means the FACTS are constant and only the words move; an arm that dropped one
    would be a different demonstration answered under the same case id."""

    case_id: str
    behaviour: str
    ledger: LabellingFixture
    arms: tuple[str, ...]
    offered: tuple[str, ...]
    anchors: tuple[str, ...]


# What every case in this file says Penny does, and it is ONE sentence because it is one
# behaviour: the cases differ in the demonstration they run it against, never in what is
# expected of the draw.
_NAMES_EVERY_SPOT = (
    "gives every spot its own name for what it supplies in THIS routine, with one line "
    "saying what belongs there — never the argument's own name handed back"
)


def _behaviour(when: str, does: str = _NAMES_EVERY_SPOT) -> str:
    """The case's one sentence, in the fixed form *In <the locus>, when <X>, Penny <does Y>.*

    The locus is the SHIPPED agent name, read off the constant production draws with, so a
    rename cannot leave five case reports describing an agent that no longer exists."""
    locus = PennyConstants.SKILL_NAMING_AGENT_NAME
    return f"In the {locus} micro-context, when {when}, Penny {does}."


TWO_SOURCES_CASE = PortedNamingCase(
    case_id="namer-tells-two-sources-apart",
    behaviour=_behaviour(
        "a demonstrated routine reads two different pages into one argument",
        f"{_NAMES_EVERY_SPOT}, and never one name covering both sites",
    ),
    ledger=_TWO_SOURCES,
    arms=(
        _TWO_SOURCES.utterance,
        "have a look at citydesk.example/front and harborpost.example/front, and save the "
        "top headline from each",
        "open citydesk.example/front and harborpost.example/front and keep the lead story "
        "from both",
        "check citydesk.example/front and harborpost.example/front, then remember whichever "
        "headline is at the top of each",
        "go to citydesk.example/front and harborpost.example/front and note down the top "
        "headline on each one",
    ),
    offered=(_FIRST_SOURCE, _SECOND_SOURCE, "extract", "memory", "key"),
    anchors=("citydesk.example/front", "harborpost.example/front"),
)

SEARCH_CASE = PortedNamingCase(
    case_id="namer-names-a-search-spot-as-a-search",
    behaviour=_behaviour("the demonstrated look-up is a text search rather than a page"),
    ledger=_SEARCH,
    arms=(
        _SEARCH.utterance,
        "search for aurora fest tickets, get the cheapest ticket price, and keep it",
        "do a search for aurora fest tickets, find the cheapest ticket price, and save that",
        "run a search for aurora fest tickets, work out the cheapest ticket price, and remember it",
        "search for aurora fest tickets, find the cheapest ticket price there, and note it down",
    ),
    offered=("queries", "extract", "memory", "key"),
    anchors=("aurora fest tickets", "cheapest ticket price"),
)

AVAILABILITY_CASE = PortedNamingCase(
    case_id="namer-names-an-availability-spot-for-this-routine",
    behaviour=_behaviour("the demonstrated routine watches availability rather than a price"),
    ledger=_AVAILABILITY,
    arms=(
        _AVAILABILITY.utterance,
        "open bookbarn.example/atlas-of-clouds, see whether it's in stock, and keep that",
        "have a look at bookbarn.example/atlas-of-clouds, find out if it's in stock, and "
        "remember it",
        "check bookbarn.example/atlas-of-clouds for whether it's in stock, and note that down",
        "read bookbarn.example/atlas-of-clouds, work out whether it's in stock, and save that",
    ),
    offered=("queries", "extract", "memory", "key"),
    anchors=("bookbarn.example/atlas-of-clouds", "in stock"),
)

SINGLE_TURN_CASE = PortedNamingCase(
    case_id="namer-names-every-spot-from-a-single-turn-teach",
    behaviour=_behaviour(
        "one direct instruction is the whole conversation, with no elicit round before it"
    ),
    ledger=_SINGLE_TURN,
    arms=(
        _SINGLE_TURN.utterance,
        "open weather.example/lisbon, get today's high temperature, and keep it",
        "check weather.example/lisbon for today's high temperature and save it",
        "read weather.example/lisbon, find today's high temperature, and note it down",
        "have a look at weather.example/lisbon, work out today's high temperature, and "
        "remember that",
    ),
    offered=("queries", "extract", "memory", "key"),
    anchors=("weather.example/lisbon", "high temperature"),
)

LONGER_ROUTINE_CASE = PortedNamingCase(
    case_id=_LONGER_ROUTINE.case_id,
    behaviour=_behaviour(
        "a demonstrated routine runs four steps and offers seven spots across two reads, a "
        "write and a log"
    ),
    ledger=_LONGER_ROUTINE,
    arms=(
        _LONGER_ROUTINE.utterance,
        "look at depotline.example/schedule for the next delivery window, check what's open "
        "on the harbour refit project, keep the window in depot-deliveries, and add a note "
        "to depot-checks that you looked",
        "read depotline.example/schedule and find the next delivery window, see what's open "
        "on the harbour refit project, save the window in depot-deliveries, and write a note "
        "to depot-checks that you looked",
        "open depotline.example/schedule, work out the next delivery window, list what's "
        "open on the harbour refit project, remember the window in depot-deliveries, and "
        "note in depot-checks that you looked",
        "go to depotline.example/schedule for the next delivery window, look up what's open "
        "on the harbour refit project, store the window in depot-deliveries, and put a note "
        "in depot-checks that you looked",
    ),
    offered=("queries", "extract", "project_name", "memory", "key", "memory-2", "content"),
    anchors=(
        "depotline.example/schedule",
        "next delivery window",
        "harbour refit",
        "depot-deliveries",
        "depot-checks",
    ),
)

# Every ported case, for the deterministic arm probe in ``make check`` — one place, so the
# probe and the live runs can never be checking two different sets of arms.
PORTED_CASES = (
    TWO_SOURCES_CASE,
    SEARCH_CASE,
    AVAILABILITY_CASE,
    SINGLE_TURN_CASE,
    LONGER_ROUTINE_CASE,
)


def _drawn_names(sample: SampleObservation, spots: Sequence[str]) -> dict[str, str]:
    """What the draw called each offered spot, keyed by the spot's current name."""
    return {spot: sample.field(label_name_field(spot)) for spot in spots}


def _hardens_to_a_key(spots: Sequence[str]) -> WorldClaim:
    """Every spot's name survives the SHIPPED hardener as something a binding can use.

    Imported, never re-implemented: a name becomes a key through ``slug_parameter_name`` at
    instantiation, so what a case calls usable and what production calls a key are one
    definition.  A name that hardens to nothing — punctuation, an empty line after the
    separator — leaves the spot named by nothing, which is the answer OMITTED rather than
    given.

    One claim over all the spots rather than one each: every spot is offered under the same
    contract, so which of them failed is the rationale's job.  (An extract instruction naming
    several DIFFERENT things is the case that needs one claim each — the things degrade one at
    a time and are not interchangeable.)"""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = _drawn_names(sample, spots)
        unusable = [spot for spot, name in drawn.items() if not slug_parameter_name(name)]
        return not unusable, f"hardens to nothing: {unusable}"

    return answer


def _not_handed_back_its_own_name(spots: Sequence[str]) -> WorldClaim:
    """No spot came back named after the tool argument it fills.

    The document offers each spot under its argument-derived name, so a draw answering
    ``queries`` with ``queries`` has described the spot rather than named it — the answer is
    OMITTED, and the routine is left exactly as unreadable as it was."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = _drawn_names(sample, spots)
        echoed = [
            spot
            for spot, name in drawn.items()
            if slug_parameter_name(name) == slug_parameter_name(spot)
        ]
        return not echoed, f"echoed the argument name: {echoed}"

    return answer


def _says_what_belongs_there(spots: Sequence[str]) -> WorldClaim:
    """Every spot's line carries the one thing to supply there each run.

    The description is the grammar's one optional field, so a line that stops after its name
    is well-formed and reaches the caller — the spot is named and nobody can tell what goes in
    it.  The instruction asks for both halves; a blank omits one."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        silent = [spot for spot in spots if not sample.field(label_says_field(spot)).strip()]
        return not silent, f"no description: {silent}"

    return answer


# The three claims EVERY naming case makes, as data: the labeller is handed a set of spots and
# asked for a name and a line for each, and these are the three ways that answer can fail to
# arrive — nothing usable, the question handed back, or no line at all.  One table rather than
# three spellings per case, because five cases make exactly this set and a claim edited in one
# of five places is a claim that has stopped meaning one thing.
#
# All three are the *nothing omitted* direction of fact alignment.  The other direction is
# absent from every case in this file, and each case's docstring says why.
_NAMING_CLAIMS = (
    ("state: every spot's name hardens to a usable binding key", _hardens_to_a_key),
    ("state: no spot was handed back its own argument name", _not_handed_back_its_own_name),
    ("state: every spot's line says what belongs there", _says_what_belongs_there),
)


def _claim_every_spot_was_named_for_this_routine(cohort: Cohort, spots: Sequence[str]) -> None:
    """Make the three shared naming claims over one case's own offered spots."""
    for label, build in _NAMING_CLAIMS:
        cohort.claim(label, build(spots), SpecCategory.PROVENANCE)


def _measure_what_each_spot_was_called(cohort: Cohort, spots: Sequence[str]) -> None:
    """What the draw called each spot, one axis per spot.

    COSMETIC, every one: ``first_front_page`` and ``citydesk_page`` leave the same routine
    behind, so a divergence is a fact about the SYSTEM's naming spread and never about one
    sample.  This is the framer's own known signature one layer down, and filing it
    consequential would make almost every sample an outlier.

    The DESCRIPTION axes are left out: a line of prose is measured by textual spread and this
    shape has no reply for that machinery to read, so an entropy over free text would report
    near-total disagreement on every run whatever the draw did.

    No tool sequence and no reply spread: a single call makes neither."""
    cohort.measure(
        *(output_field(label_name_field(spot), consequence=Consequence.COSMETIC) for spot in spots)
    )


async def _run_ported_case(
    labeller_eval: LabellerEval, case: PortedNamingCase, model: str
) -> Cohort:
    """Drive one ported case's ledger through the labeller in its five wordings.

    Every ported case is report-only (``min_pass_rate=None``): assertions carry no floor, the
    run counts them, and whether a number is a failure is the code owner's call."""
    return await labeller_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        model=model,
        utterance=case.arms[0],
        also_demonstrated=case.arms[1:],
        conversation=case.ledger.conversation,
        calls=case.ledger.calls,
        target=case.ledger.target,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )


def _the_two_sources_drew_different_names(sample: SampleObservation, _world: World) -> Answer:
    """The two sites are two spots, and they read back as two.

    The demonstration SUPPLIES the distinction — each site takes its own value every run — so
    a draw calling both of them ``news_page`` has omitted it, and at run time one name for two
    spots cannot say which site is which.  Compared HARDENED, because that is what the names
    become: ``News Page`` and ``news_page`` are one key, not two."""
    drawn = _drawn_names(sample, (_FIRST_SOURCE, _SECOND_SOURCE))
    hardened = (
        slug_parameter_name(drawn[_FIRST_SOURCE]),
        slug_parameter_name(drawn[_SECOND_SOURCE]),
    )
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

    This is the ONLY case in the file that makes a claim of its own: the two-sources
    demonstration is the one that supplies a distinction a name can collapse.
    """
    cohort = await _run_ported_case(labeller_eval, TWO_SOURCES_CASE, model)
    # LANDED — empty, and closed upstream; see the docstring.
    # STORE — empty by construction; see the docstring.
    # PROVENANCE — the OPEN fields, which for this shape are the whole typed result.
    _claim_every_spot_was_named_for_this_routine(cohort, TWO_SOURCES_CASE.offered)
    cohort.claim(
        "state: the two sources drew different names",
        _the_two_sources_drew_different_names,
        SpecCategory.PROVENANCE,
    )
    _measure_what_each_spot_was_called(cohort, TWO_SOURCES_CASE.offered)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_search_spot_is_named_as_a_search_however_it_is_worded(
    labeller_eval: LabellerEval, model: str
) -> None:
    """One demonstration in five wordings: search for a festival's tickets, keep the cheapest
    price.

    **This case asserts nothing the two-sources case does not already assert.**  It makes the
    same three claims on a different demonstration, and that is worth saying plainly rather
    than dressing up: its value is the third mechanism in the design's §9 — a person reading
    the modal sample — plus its contribution to the naming-spread variance axis.

    What it supplies that no other ledger does is a look-up that is a SEARCH PHRASE and not a
    url, which is where the failure this slot exists for shows itself: a spot named
    ``product_page`` here is named for what the LAST routine did.  **That failure has no legal
    instrument** — ``unsourced_specifics`` reads urls, digits and capitalised phrases, and
    ``product_page`` carries none of the three — so the samples are what a reader reads.

    **LANDED is empty and closed upstream** (``_labels_every_spot`` validates coverage and
    re-rolls); **STORE is empty by construction** (one call, a typed result, no store).  The
    provenance block carries only the *nothing omitted* direction, for the reason above.
    """
    cohort = await _run_ported_case(labeller_eval, SEARCH_CASE, model)
    # LANDED — empty, and closed upstream; see the docstring.
    # STORE — empty by construction; see the docstring.
    # PROVENANCE — the OPEN fields, which for this shape are the whole typed result.
    _claim_every_spot_was_named_for_this_routine(cohort, SEARCH_CASE.offered)
    _measure_what_each_spot_was_called(cohort, SEARCH_CASE.offered)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_availability_spot_is_named_for_this_routine_however_it_is_worded(
    labeller_eval: LabellerEval, model: str
) -> None:
    """One demonstration in five wordings: read a book's page, keep whether it is in stock.

    **This case asserts nothing the two-sources case does not already assert.**  Same three
    claims, different demonstration; its value is the modal sample a person reads and the
    naming-spread axis, not a new claim.

    What it supplies is a routine about AVAILABILITY rather than a price — the semantics every
    other ledger in the canonical set shares.  A labeller working from a memorised price
    template names the extract spot for a price nobody mentioned, and that is exactly the
    wrong-but-stable row the design says only a human reading one sample catches: a
    ``current_price`` label on this routine carries no url, no digit and no capital, so
    ``unsourced_specifics`` cannot see it either.

    **LANDED is empty and closed upstream** (``_labels_every_spot`` validates coverage and
    re-rolls); **STORE is empty by construction** (one call, a typed result, no store).  The
    provenance block carries only the *nothing omitted* direction, for the reason above.
    """
    cohort = await _run_ported_case(labeller_eval, AVAILABILITY_CASE, model)
    # LANDED — empty, and closed upstream; see the docstring.
    # STORE — empty by construction; see the docstring.
    # PROVENANCE — the OPEN fields, which for this shape are the whole typed result.
    _claim_every_spot_was_named_for_this_routine(cohort, AVAILABILITY_CASE.offered)
    _measure_what_each_spot_was_called(cohort, AVAILABILITY_CASE.offered)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_single_turn_teach_names_every_spot_however_it_is_worded(
    labeller_eval: LabellerEval, model: str
) -> None:
    """One demonstration in five wordings, with NO elicit round: one direct instruction is the
    whole conversation.

    **This case asserts nothing the two-sources case does not already assert.**  Same three
    claims, different demonstration; its value is the modal sample and the naming-spread axis.

    What it supplies is the conversation block at its MINIMUM.  Every other ledger hands the
    draw an ask and an elicit round before the demonstration; this one hands it one sentence,
    so a draw that had been leaning on the surrounding conversation to work out what a spot is
    for has nothing to lean on.  A spot mis-named for want of that context still carries no
    url, no digit and no capital, so it is read rather than asserted.

    **LANDED is empty and closed upstream** (``_labels_every_spot`` validates coverage and
    re-rolls); **STORE is empty by construction** (one call, a typed result, no store).  The
    provenance block carries only the *nothing omitted* direction, for the reason above.
    """
    cohort = await _run_ported_case(labeller_eval, SINGLE_TURN_CASE, model)
    # LANDED — empty, and closed upstream; see the docstring.
    # STORE — empty by construction; see the docstring.
    # PROVENANCE — the OPEN fields, which for this shape are the whole typed result.
    _claim_every_spot_was_named_for_this_routine(cohort, SINGLE_TURN_CASE.offered)
    _measure_what_each_spot_was_called(cohort, SINGLE_TURN_CASE.offered)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_longer_routine_names_every_spot_however_it_is_worded(
    labeller_eval: LabellerEval, model: str
) -> None:
    """One demonstration in five wordings, over a FOUR-step routine offering SEVEN spots.

    **This case asserts nothing the two-sources case does not already assert.**  Same three
    claims, a bigger demonstration; its value is the modal sample and the naming-spread axis.

    What it supplies is LENGTH, which nothing else in the set has.  Every other ledger here is
    ``browse -> collection_write`` — the shortest routine that can exist — while #1824's own
    finding about the failure it replaced is that it "compounded with every extra leaf".  A set
    measured only on two-step routines cannot see compounding at all.  This one adds a second
    read through a PLUGIN tool (the case nothing in the codebase enumerates in advance), a
    ``log_append`` whose free-text line is a spot like any other, and two collections rather
    than one — so two spots share the base name ``memory`` and are offered as ``memory`` and
    ``memory-2``, which is the concrete way a longer routine makes the argument-derived names
    less informative.

    **LANDED is empty and closed upstream** (``_labels_every_spot`` validates coverage and
    re-rolls) — and on seven spots it is doing more work than on four, so a rise in excluded
    samples here is a finding about the draw rather than about this case.  **STORE is empty by
    construction** (one call, a typed result, no store).  The provenance block carries only the
    *nothing omitted* direction, for the reason above.
    """
    cohort = await _run_ported_case(labeller_eval, LONGER_ROUTINE_CASE, model)
    # LANDED — empty, and closed upstream; see the docstring.
    # STORE — empty by construction; see the docstring.
    # PROVENANCE — the OPEN fields, which for this shape are the whole typed result.
    _claim_every_spot_was_named_for_this_routine(cohort, LONGER_ROUTINE_CASE.offered)
    _measure_what_each_spot_was_called(cohort, LONGER_ROUTINE_CASE.offered)


# Every ledger, for the deterministic drift probes in ``make check`` — one place, so the
# probes and the live runs can never be checking two different fixtures.
FIXTURES = (
    _AVAILABILITY,
    _TWO_SOURCES,
    _SHARED_SPOT,
    _SINGLE_TURN,
    _SEARCH,
    _LONGER_ROUTINE,
)
