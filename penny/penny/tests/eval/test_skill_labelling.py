"""Live-model contract for the run-end LEAF LABELLER (#1824), in isolation.

Every leaf of a demonstrated round is a placeholder, unconditionally, and this draw's
entire job is to NAME each one: a short semantic name for the spot plus one line of
what belongs there each time the routine runs.  Its whole input is the routine's tool
calls — the implementation — so these cases hand it a FIXTURE ledger and nothing else:
no conversation exists in its content to leak an interface question into it.

**What is no longer measured, and why.**  Until #1824 this same draw also ruled, per
candidate, whether the USER supplied that value — and the case that motivated the
suite scored an assistant-composed entry NOT becoming a required parameter.  That
invariant is now true BY CONSTRUCTION: a leaf cannot become a parameter at all, since
what a skill asks for is decided once, at the interface, from the user's own ask
(``test_skill_framing.py``).  So the old guard case is retired into what is left worth
measuring on this draw — naming QUALITY and COVERAGE — and the assembled-key shape it
used to catch becomes an ordinary naming case: the storage key simply gets a sensible
descriptive name, with no wrong-verdict failure mode left behind it.

Scoring stays deliberately broad: a spot's label is checked for being WELL-FORMED (a
name that hardens to a usable binding key, a non-blank description) and ON TOPIC (any
of several plausible wordings), never against an expected string — the name is the
model's to choose.

Deliberately NOT scored: what a demonstrated round chooses to write.  If a round
writes two entries, two entries are the skill — the code owner's ruling on #1770,
unchanged.  These cases fix the round and vary only the judgment.

All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import LeafEval, LeafSlot

pytestmark = pytest.mark.eval

_FAMILY = "leaf-labelling"

_TARGET = "prices"
_PRICE = "$499"
_LISTING = "https://faux-market.example/aurora-deck-2"
_EXTRACT = "the price shown on the product page"

# The floor-case round: read a page for one fact, then write that fact down.  The
# price BINDS to step 1 (structural dataflow), so it is not offered — bindings survive
# as deterministic work and this draw never sees them.
_BROWSE = (
    "browse",
    {"queries": [_LISTING], "extract": _EXTRACT},
    f"You opened the Aurora Deck 2 listing (browse result)\n{_PRICE}",
    True,
)


def _write(key: str) -> tuple[str, dict, str, bool]:
    """The round's write, under whichever key the case varies."""
    return (
        "collection_write",
        {"memory": _TARGET, "entries": [{"key": key, "content": _PRICE}]},
        f"You saved an entry to {_TARGET}: (collection_write result)\nWrote 1 entry.",
        True,
    )


# What each spot in that round holds, and the words a label for it could plausibly
# use.  Alternatives, matched broadly — several wordings describe the same spot, and
# the contract is that the draw described THAT spot rather than a neighbouring one.
_PAGE_SLOT = LeafSlot(_LISTING, ("url", "page", "address", "link", "site", "listing"))
_WHAT_TO_PULL_SLOT = LeafSlot(
    _EXTRACT, ("extract", "pull", "find", "information", "detail", "value", "look for", "data")
)
_DESTINATION_SLOT = LeafSlot(
    _TARGET, ("collection", "memory", "store", "storage", "destination", "where", "save", "write")
)


# ── Case 1: the floor case — every spot in a two-call routine gets a usable name ─


@pytest.mark.asyncio
async def test_every_placeholder_in_the_floor_case_is_named(leaf_eval: LeafEval):
    """The floor case: a browse + write routine whose every offered spot must come back
    with a well-formed, on-topic label.

    This is the coverage direction — a spot with no line keeps its arg-derived name,
    which is legible but says nothing about what belongs there, and a routine full of
    those is a program nobody can fill in.  Four spots, four names, each hardening to a
    usable binding key."""
    await leaf_eval(
        case_id="leaf-floor-case-every-placeholder-named",
        calls=[_BROWSE, _write("Aurora Deck 2 price")],
        target=_TARGET,
        slots=[
            _PAGE_SLOT,
            _WHAT_TO_PULL_SLOT,
            _DESTINATION_SLOT,
            LeafSlot(
                "Aurora Deck 2 price", ("key", "label", "name", "entry", "identifier", "title")
            ),
        ],
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 2: the assembled key — a spot whose value the round built for itself ───


@pytest.mark.asyncio
async def test_an_assembled_storage_key_gets_a_descriptive_name(leaf_eval: LeafEval):
    """The shape that used to be the hard case: a write key the round assembled out of
    the page's own slug (``'Aurora Deck 2'``), which reads as the user's words for a
    filing decision they never made.

    Under the per-leaf verdict pipeline that ambiguity was the whole difficulty — call
    it theirs and the skill demands a label nobody supplied.  There is no verdict left
    to get wrong, so what remains is whether the draw describes the SPOT (what the entry
    gets filed under each run) rather than the value it held once."""
    await leaf_eval(
        case_id="leaf-assembled-key-gets-a-descriptive-name",
        calls=[_BROWSE, _write("Aurora Deck 2")],
        target=_TARGET,
        slots=[
            _PAGE_SLOT,
            _WHAT_TO_PULL_SLOT,
            _DESTINATION_SLOT,
            LeafSlot("Aurora Deck 2", ("key", "label", "name", "entry", "identifier", "title")),
        ],
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
