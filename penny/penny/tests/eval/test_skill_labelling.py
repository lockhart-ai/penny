"""Live-model contract for the run-end LEAF LABELLER (#1824), in isolation.

Every leaf of a demonstrated round is a placeholder, unconditionally, and this draw's
entire job is to NAME each one: a short semantic name for the spot plus one line of
what belongs there each time the routine runs.  Its whole input is the routine's tool
calls — the implementation — so these cases hand it a FIXTURE ledger and nothing else:
no conversation exists in its content to leak an interface question into it.

**Each case sweeps a POOL of routines, not one frozen fixture.**  Sample i draws
``pool[i % len(pool)]``, the same doctrine the classifier evals' phrasing pools encode:
one demonstration measured N times reports how that demonstration happened to land,
where eight routines of the same SHAPE across different everyday domains report the
judgment.  The eight are the code owner's ruled fixture set — the SAME teach turns the
framer's floor case sweeps, so a finding on one draw is readable against the other.

**What is scored, and what is not.**  A spot passes when its line is WELL-FORMED: a
plausible semantic name that hardens to a usable binding key, plus a non-empty
description.  That is the code owner's ruling and it is the right altitude — an
alternatives set narrow enough to reject one good wording rejects the draw for being
phrased differently rather than for being wrong.  The drawn label itself renders
ADVISORY beside it, together with a selector-syntax watch keyed to actual CSS/XPath/
regex claims (never to ordinary verbs like "scrape", which describe the job correctly).

**What is no longer measured, and why.**  Until #1824 this same draw also ruled, per
candidate, whether the USER supplied that value — and the case that motivated the suite
scored an assistant-composed entry NOT becoming a required parameter.  That invariant is
now true BY CONSTRUCTION: a leaf cannot become a parameter at all, since what a skill
asks for is decided once, at the interface, from the user's own ask
(``test_skill_framing.py``).  So the old guard case is retired into what is left worth
measuring — naming quality and coverage — and the assembled-key shape it used to catch
becomes an ordinary naming case.

Deliberately NOT scored: what a demonstrated round chooses to write.  If a round writes
two entries, two entries are the skill — the code owner's ruling on #1770, unchanged.

All content is synthetic (faux-market / harbour-ferry / corner-bakery / ridge-trails).
"""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import DemoCall, LeafEval, LeafRound

pytestmark = pytest.mark.eval

_FAMILY = "leaf-labelling"


def _calls(
    *, url: str, extract: str, found: str, memory: str, opened: str, key: str
) -> tuple[DemoCall, ...]:
    """One demonstrated routine's ledger: read a page for one fact, write that fact
    down.  The FOUND value binds structurally to step 1, so it is never offered — a
    step-result binding stays deterministic work and this draw never sees it."""
    return (
        (
            "browse",
            {"queries": [url], "extract": extract},
            f"{opened} (browse result)\n{found}",
            True,
        ),
        (
            "collection_write",
            {"memory": memory, "entries": [{"key": key, "content": found}]},
            f"You saved an entry to {memory}: (collection_write result)\nWrote 1 entry.",
            True,
        ),
    )


# The eight rounds the pools are built from — one per teach turn in the code owner's
# ruled fixture set, the SAME set the framer's floor case sweeps.  Same SHAPE every time
# (browse one page for one fact, write it down), different domains and values, so a
# sample's result is about the judgment rather than about one demonstration.
_ROUNDS = (
    {
        "url": "https://harbour-ferry.example/timetable",
        "extract": "the time of the first sailing",
        "found": "06:40",
        "memory": "sailings",
        "opened": "You opened the harbour ferry timetable",
    },
    {
        "url": "https://corner-bakery.example/specials",
        "extract": "the soup of the day",
        "found": "carrot and coriander",
        "memory": "specials",
        "opened": "You opened the corner bakery specials board",
    },
    {
        "url": "https://ridge-trails.example/summit-loop",
        "extract": "whether the trail is open",
        "found": "open, muddy in places",
        "memory": "trail-notes",
        "opened": "You opened the summit loop trail page",
    },
    {
        "url": "https://harborseals.example/colony-count",
        "extract": "the colony count",
        "found": "48 hauled out",
        "memory": "colony-counts",
        "opened": "You opened the colony count page",
    },
    {
        "url": "https://bay-tides.example/table",
        "extract": "this morning's low tide time",
        "found": "05:12",
        "memory": "tides",
        "opened": "You opened the bay tide table",
    },
    {
        "url": "https://town-library.example/new-arrivals",
        "extract": "the newest mystery title",
        "found": "The Quiet Harbour",
        "memory": "new-arrivals",
        "opened": "You opened the town library new arrivals list",
    },
    {
        "url": "https://birding-club.example/sightings",
        "extract": "the latest sighting",
        "found": "kingfisher at the weir",
        "memory": "sightings",
        "opened": "You opened the birding club sightings board",
    },
    {
        "url": "https://faux-market.example/aurora-deck-2",
        "extract": "the price shown on the product page",
        "found": "$499",
        "memory": "prices",
        "opened": "You opened the Aurora Deck 2 listing",
    },
)

# The keys each round's write used — the half the two cases vary.  A DESCRIBED key says
# what the entry is; an ASSEMBLED one is the page's own name, slugged out of what the
# user pasted.
_DESCRIBED_KEYS = (
    "first sailing",
    "soup of the day",
    "summit loop status",
    "colony count",
    "morning low tide",
    "newest mystery title",
    "latest sighting",
    "Aurora Deck 2 price",
)
_ASSEMBLED_KEYS = (
    "Harbour Ferry",
    "Corner Bakery",
    "Summit Loop",
    "Harborseals",
    "Bay Tides",
    "Town Library",
    "Birding Club",
    "Aurora Deck 2",
)


def _pool(keys: tuple[str, ...]) -> list[LeafRound]:
    """One case's pool: the eight rounds under the given write keys, each declaring the
    four spots whose names are scored."""
    return [
        LeafRound(
            calls=_calls(**fixture, key=key),
            target=fixture["memory"],
            slots=(fixture["url"], fixture["extract"], fixture["memory"], key),
        )
        for fixture, key in zip(_ROUNDS, keys, strict=True)
    ]


# ── Case 1: the floor case — every spot in a two-call routine gets a usable name ─


@pytest.mark.asyncio
async def test_every_placeholder_in_the_floor_case_is_named(leaf_eval: LeafEval):
    """The floor case: a browse + write routine whose every offered spot must come back
    with a well-formed label.

    This is the coverage direction — a spot with no line keeps its arg-derived name,
    which is legible but says nothing about what belongs there, and a routine full of
    those is a program nobody can fill in.  Four spots per round, four names."""
    await leaf_eval(
        case_id="leaf-floor-case-every-placeholder-named",
        pool=_pool(_DESCRIBED_KEYS),
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )


# ── Case 2: the assembled key — a spot whose value the round built for itself ───


@pytest.mark.asyncio
async def test_an_assembled_storage_key_gets_a_descriptive_name(leaf_eval: LeafEval):
    """The shape that used to be the hard case: a write key the round assembled out of
    the page's own name, which reads as the user's words for a filing decision they
    never made.

    Under the per-leaf verdict pipeline that ambiguity was the whole difficulty — call
    it theirs and the skill demands a label nobody supplied.  There is no verdict left
    to get wrong, so what remains is whether the draw describes the SPOT (what the entry
    gets filed under each run) rather than the value it held once."""
    await leaf_eval(
        case_id="leaf-assembled-key-gets-a-descriptive-name",
        pool=_pool(_ASSEMBLED_KEYS),
        min_pass_rate=None,  # report-only until sample-verified with the code owner
        family=_FAMILY,
    )
