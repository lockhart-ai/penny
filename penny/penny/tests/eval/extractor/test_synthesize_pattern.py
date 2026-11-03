"""The ``synthesize`` micro-context, in isolation (#2932): what it makes of a short
numbered list of already-retrieved hits.

Retrieval is a separate, deterministic concern, pinned by unit tests
(``tests/tools/test_memory_tools.py``) — these cases never touch it. What is
measured here is the draw alone: given several things found in someone's own
memory, does it name a real pattern across them and point back to the items
that carry it, and does it stay honest — inventing nothing — when the items
share no pattern at all.

Every item here is synthetic and unrelated to any real person's life: a
household's notes about a sticking vent, a book list, a grocery list, a
travel idea, a plant, a car — none of it drawn from any real use.

Two cases, not one arm of the same case, because they do not share an
expected shape (#1994 §A's pairing rule): the planted-pattern world's
items genuinely share a theme and the no-pattern world's genuinely do not,
so a draw that always claimed a pattern would pass the first and fail the
second.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from penny.database.memory.types import ResolvedEntry, ResolvedKind
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.synthesize import (
    SYNTHESIZE_VALUE,
    SynthesizeEval,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import SampleObservation, SpecCategory, unsourced_specifics
from penny.tests.eval.utils.worlds import World
from penny.tools.memory_tools import SynthesizeTool

pytestmark = pytest.mark.eval

_FAMILY = "synthesize"

_QUERY = "What does my memory say about things going wrong?"

# A fixed, arbitrary moment — the draw never reads it, only ``_render_items_for_draw``'s
# content/key; carried only because ``ResolvedEntry`` requires it.
_WHEN = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


def _entry(entry_id: int, memory_name: str, content: str) -> ResolvedEntry:
    return ResolvedEntry(
        entry_id=entry_id,
        memory_name=memory_name,
        container_kind=ResolvedKind.COLLECTION,
        key=None,
        content=content,
        created_at=_WHEN,
    )


# The planted theme: a vent left shut shows up across three different, otherwise
# unrelated collections. Two distractors (unrelated facts, two more collections)
# sit between them so a draw cannot just read "the first N" and get it right.
_PATTERN_ITEMS = [
    _entry(
        1,
        "greenhouse-notes",
        "Noticed the tomatoes wilt by midafternoon whenever the vent stays shut past mid-morning.",
    ),
    _entry(2, "book-list", "Finished reading a mystery novel set in a lighthouse."),
    _entry(
        3,
        "kitchen-log",
        "The vent stayed shut again today and the bread dough didn't rise properly.",
    ),
    _entry(4, "grocery-notes", "Need to buy more olive oil and rice on the next trip."),
    _entry(
        5,
        "workshop-journal",
        "Left the shop vent closed overnight and the glue took twice as long to cure.",
    ),
]
_PLANTED_INDICES = {1, 3, 5}
_DISTRACTOR_INDICES = {2, 4}
_THEME_WORD = "vent"

# No shared theme at all: five distinct, ordinary notes, each in its own collection,
# none of them about anything going wrong.
_NO_PATTERN_ITEMS = [
    _entry(1, "book-list", "Finished reading a mystery novel set in a lighthouse."),
    _entry(2, "grocery-notes", "Need to buy more olive oil and rice on the next trip."),
    _entry(3, "travel-ideas", "Thinking about visiting a coastal town next spring."),
    _entry(4, "plant-care", "The fern by the window needs repotting soon."),
    _entry(5, "car-maintenance", "The car's oil change is due next month."),
]

# A deterministic probe held at IMPORT time (module-level code still runs under
# `make check`'s collection even with the `eval` marker, same as the extractor
# cases' own fixture probes) — the planted theme word really is on every
# planted item and never on a distractor or a no-pattern item, so the case
# measures what it claims to.
for _item in _PATTERN_ITEMS:
    assert (_THEME_WORD in _item.content.lower()) == (_item.entry_id in _PLANTED_INDICES), (
        f"item {_item.entry_id} should carry {_THEME_WORD!r} iff it is planted "
        f"({_item.entry_id in _PLANTED_INDICES})"
    )
for _item in _NO_PATTERN_ITEMS:
    assert _THEME_WORD not in _item.content.lower(), (
        f"the no-pattern world's item {_item.entry_id} must not carry {_THEME_WORD!r}"
    )
del _item

_ITEM_NUMBER = re.compile(r"\b[1-5]\b")


def _render(items: list[ResolvedEntry]) -> str:
    """The numbered list text the draw reads — the SAME render
    ``SynthesizeTool`` itself hands the micro-context, so the case exercises
    production's own rendering, not a hand-copied duplicate."""
    return SynthesizeTool._render_items_for_draw(items)  # noqa: SLF001 (shared render, by design)


def _cited_item_numbers(value: str) -> set[int]:
    """Every standalone digit 1-5 in the synthesis — the item numbers it cites.
    None of the world's own content carries a digit, so any that appears here
    is a reference to one of the five numbered items."""
    return {int(match) for match in _ITEM_NUMBER.findall(value)}


def _names_the_theme(sample: SampleObservation, _world: World) -> Answer:
    value = sample.field(SYNTHESIZE_VALUE)
    named = _THEME_WORD in value.lower()
    return named, f"theme word {_THEME_WORD!r} not named: {value!r}"


def _cites_only_planted_items(sample: SampleObservation, _world: World) -> Answer:
    value = sample.field(SYNTHESIZE_VALUE)
    cited = _cited_item_numbers(value)
    stray = cited - _PLANTED_INDICES
    ok = bool(cited) and not stray
    wanted = sorted(_PLANTED_INDICES)
    return ok, f"cited {sorted(cited) or 'nothing'}, wanted a non-empty subset of {wanted}"


def _nothing_invented(sample: SampleObservation, _world: World) -> Answer:
    """Every specific value in the synthesis traces to what the draw was GIVEN
    (#2932) — the honesty direction the no-pattern world's own claim is: with
    nothing to name, the strongest thing left to ask is that nothing is made
    up. Read over the WHOLE given (#2078), not a hand-built copy of ``items``,
    so the claim covers the query and the draw's own framing too."""
    value = sample.field(SYNTHESIZE_VALUE)
    invented = unsourced_specifics(value, sample.given)
    return not invented, f"not in what it was given: {invented}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_planted_theme_across_several_collections_is_named(
    synthesize_eval: SynthesizeEval, model: str
) -> None:
    """In the synthesis micro-context, when several retrieved items across
    different collections share a theme, Penny's synthesis names it and
    points back to the numbered items that carry it."""
    cohort = await synthesize_eval(
        case_id="synthesize-vent-pattern",
        behaviour=(
            "In the synthesis micro-context, when several retrieved items share a "
            "theme, Penny's synthesis names it and points back to the items that "
            "carry it."
        ),
        area=Area.MEMORY,
        model=model,
        items=_render(_PATTERN_ITEMS),
        query=_QUERY,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    cohort.claim(
        "state: the synthesis names the shared theme",
        _names_the_theme,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the synthesis cites only the items that carry the theme",
        _cites_only_planted_items,
        SpecCategory.PROVENANCE,
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_unrelated_items_show_no_invented_pattern(
    synthesize_eval: SynthesizeEval, model: str
) -> None:
    """In the synthesis micro-context, when the retrieved items share no
    theme, Penny's synthesis states only what the items actually say — the
    PAIR to the planted-theme case above: a draw that always claimed a
    pattern would pass that one and fail this one."""
    cohort = await synthesize_eval(
        case_id="synthesize-no-pattern",
        behaviour=(
            "In the synthesis micro-context, when the retrieved items share no "
            "theme, Penny's synthesis states only what the items actually say."
        ),
        area=Area.MEMORY,
        model=model,
        items=_render(_NO_PATTERN_ITEMS),
        query=_QUERY,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    cohort.claim(
        "state: every specific value in the synthesis is sourced",
        _nothing_invented,
        SpecCategory.PROVENANCE,
    )
