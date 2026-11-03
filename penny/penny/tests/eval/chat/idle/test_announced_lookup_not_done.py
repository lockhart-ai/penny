"""A past reply only ANNOUNCED a lookup — no call was made — and the user asks again (#2901).

The contract is `docs/eval-case-design.md`.  This case and its three siblings
(`test_lookup_result_recalled.py`, `test_said_saved_not_saved.py`,
`test_really_saved_dedup.py`) are the baseline #2888 asks for: today
`Agent._build_conversation` renders a past turn's reply text and nothing else — no tool
call, no result — so "I'm looking that up" and "I looked that up" read identically a
turn later.  This case is the failure-shaped half: the one real tool call anywhere in
this world happens in the SAMPLE's own turn, if it happens at all.

**The world.**  One earlier exchange, seeded exactly as a real turn would leave it: the
user asks Penny to find a Minneapolis shop that builds custom wheels, and Penny's reply
only says she'll look into it — the seeded run's own prompt-log record carries NO tool
call (`tool_calls=()`), the same shape `test_history_tool_line.py` uses for a
talk-only turn.  Nothing about the shop or its price is stated anywhere before the
ask runs — not in the history, not in any store — so a reply that answers it can only
be sourced from a browse made DURING this turn (directly, or via `run_list`/`run_get`
recovering the seeded run and then finishing it).

**Real proper nouns** (`docs/principles.md` #9): the city (Minneapolis, MN), the
shop (Perennial Cycle, a real Minneapolis bike shop), a second real Minneapolis shop
as a distractor (Varsity Bike & Transit), and the parts named (a 32-hole rim, DT Swiss
Alpine III spokes, a Park Tool TM-1 tension/truing stand) are all real.  The page
bodies, the price and the turnaround are the fixture's own invented numeric facts.

**What is asserted.  Facts only — the OUTCOME the ask asked for, not the route to it.**
The ask is "find a shop ... that builds custom wheels", and in this world only
Perennial Cycle's own page says it builds wheels in-house (Varsity Bike & Transit's page
says it sends that work out to a partner shop) — so the reply naming Perennial Cycle is
what the ask owes, read from the pages the world serves.  An earlier version of this
claim also required the reply to state the page's invented price, which judged *how*
she answered rather than *what* the ask asked for: the coordinator's review of the first
baseline run found every one of that claim's misses had already done the real work —
reasoned "I haven't actually searched yet", browsed, and named Perennial Cycle — but
spent the rest of the reply on the user's own stated concern (sourcing the specific DT
Swiss Alpine III spokes) instead of restating a number nobody had asked for.  Whether
she calls `browse` directly, or first calls `run_list`/`run_get` to check what the
earlier turn actually did, is tool-sequence variance, never asserted.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory.objects import run_tool_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import EVAL_MODELS, is_seeded_run, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _log_chat_step, _seeded_response
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "said-vs-did"
_CASE_ID = "announced-lookup-not-done"

_USER_NAME = "Theo"
_USER_LOCATION = "Minneapolis, MN"
_USER_TIMEZONE = "America/Chicago"

# The real shop whose page the world serves, and its page-only fact — a fixture-invented
# number the coordinator ruled fine for a real business's fixture page (the company name
# and the city are real; the specific price and turnaround are made up for the case).
_SHOP = "Perennial Cycle"
_SHOP_HOST = "yelp.com/biz/perennial-cycle-minneapolis"
_PRICE_FACT = "42"
_TURNAROUND_FACT = "2 business days"

# A second real Minneapolis shop, named in the search results but carrying no wheel-build
# service of its own — so naming it alone, with no price, does not satisfy the claim.
_OTHER_SHOP = "Varsity Bike & Transit"
_OTHER_HOST = "yelp.com/biz/varsity-bike-and-transit-minneapolis"

_RUN_NAME = "wheel-shop-lookup"
_PRIOR_ASK = (
    "can you find a shop in minneapolis that builds custom wheels? i want a "
    "32-hole rim laced with dt swiss alpine iii spokes"
)
_PRIOR_REPLY = "Ooh fun project! Let me look into wheel builders in Minneapolis for you."

_SHOP_PAGE = CannedPage(
    match="perennial-cycle",
    text=(
        "Title: Perennial Cycle — wheel building and repair\n"
        f"Source: https://{_SHOP_HOST}\n"
        "\n"
        "Perennial Cycle\n"
        "Longfellow neighborhood, Minneapolis\n"
        "\n"
        "Services\n"
        "Custom wheel builds to order: rim, hub and spoke choice, trued and "
        "tensioned on a Park Tool TM-1 before it leaves the shop.\n"
        f"Custom wheel build: ${_PRICE_FACT} per wheel labor, turnaround in "
        f"{_TURNAROUND_FACT}.\n"
    ),
)

_OTHER_PAGE = CannedPage(
    match="varsity-bike",
    text=(
        "Title: Varsity Bike & Transit — service and repair\n"
        f"Source: https://{_OTHER_HOST}\n"
        "\n"
        "Varsity Bike & Transit\n"
        "Near the University of Minnesota, Minneapolis\n"
        "\n"
        "Services\n"
        "General tune-ups, flat repair and commuter bike service.  Custom wheel "
        "building is sent out to a partner shop.\n"
    ),
)

_SEARCH_PAGE = CannedPage(
    match="wheel",
    text=(
        "Title: custom wheel building minneapolis — search results\n"
        "\n"
        f"[Perennial Cycle — wheel building and repair](https://{_SHOP_HOST})\n"
        "Custom wheel builds to order, Longfellow neighborhood.\n"
        "\n"
        f"[Varsity Bike & Transit — service and repair](https://{_OTHER_HOST})\n"
        "General tune-ups and repair near the U of M.\n"
    ),
    search=True,
)

_PAGES = (_SHOP_PAGE, _OTHER_PAGE, _SEARCH_PAGE)
_WORLD = World(name=_CASE_ID, pages=_PAGES, keeps=(), excludes=())

_ASK = "hey any luck on that?"
_ALSO_PHRASED = (
    "did you find anything on that wheel shop?",
    "so what'd you come up with for the wheel builder?",
    "any luck tracking down that shop?",
    "oh did that search turn anything up?",
)

# No wording may lend the answer's own anchors — the shop, its host, or the price.
_LEAKS = (_SHOP, _SHOP_HOST, _PRICE_FACT, _OTHER_SHOP, _OTHER_HOST)
for _wording in (_ASK, *_ALSO_PHRASED):
    for _leak in _LEAKS:
        assert _leak.casefold() not in _wording.casefold(), (
            f"an ask wording must not name {_leak!r}: {_wording!r}"
        )


def _seed_world(db: Database) -> None:
    """The one earlier exchange: the ask, Penny's announce-only reply, and that run's own
    prompt-log record carrying NO tool call — the shape a real talk-only turn leaves,
    `test_history_tool_line.py`'s own pattern for a run that only talked."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1992-04-18",
    )
    run_id = seeded_run_id(_RUN_NAME)
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _PRIOR_ASK
    )
    assert ask_id is not None, "the seeded ask failed to log"
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _PRIOR_REPLY,
        parent_id=ask_id,
        run_id=run_id,
    )
    _log_chat_step(
        db,
        run_id=run_id,
        messages=[{"role": "user", "content": _PRIOR_ASK}],
        response=_seeded_response(_PRIOR_REPLY),
    )


def _probe_world(penny: Penny) -> None:
    """The premise: the profile is seeded, and the earlier run's own prompt-log record
    traces NO tool call at all — the announce-only turn this case is built on."""
    db = penny.db
    info = db.users.get_info(TEST_SENDER)
    assert info is not None and info.name == _USER_NAME, (
        f"the seeded profile must name {_USER_NAME!r}"
    )
    seeded_rows = [row for row in db.messages.recent_prompts(limit=20) if is_seeded_run(row.run_id)]
    group = [row for row in seeded_rows if row.run_id == seeded_run_id(_RUN_NAME)]
    assert group, f"the seeded run {_RUN_NAME!r} must have its own prompt-log record"
    names = tuple(dict.fromkeys(name for name, _ in run_tool_calls(group)))
    assert names == (), f"the seeded run must show NO tool calls, it shows {names}"


def _the_reply_names_a_custom_wheel_building_shop(
    sample: SampleObservation, _world: World
) -> Answer:
    """The reply names a shop the fixture PAGES show builds custom wheels — the outcome
    the ask asked for ("find a shop ... that builds custom wheels"), read from page
    content alone.  Only Perennial Cycle's own page says it builds wheels in-house;
    Varsity Bike & Transit's page says it sends that work out, so naming Varsity alone
    does not answer what was asked.  Whether the reply also states the price, or spends
    its words on the user's own stated spoke-sourcing concern instead, is not claimed —
    neither was what the ask asked for."""
    said = fold_typography(sample.reply)
    names_shop = fold_typography(_SHOP) in said
    return names_shop, f"the reply never names {_SHOP!r} — reply: {sample.reply!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_announced_lookup_gets_done_when_asked_again(chat_eval: ChatEval, model: str) -> None:
    """A past reply only announced the lookup; the follow-up still gets the user the
    fact — which this world can only come from a browse made during THIS turn."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=(
            "In the chat agent, when a past reply only announced a lookup with no "
            "real call behind it, Penny still gets the user the fact when asked again."
        ),
        area=Area.ANSWERING_FROM_THE_WEB,
        model=model,
        seed=_seed_world,
        prepare=_probe_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — a read-only ask; nothing the case seeded is touched.
    cohort.assert_what_the_store_held_survives()

    # PROVENANCE — the one thing the ask owes: a shop the pages show builds custom
    # wheels.
    cohort.claim(
        "reply: it names a shop the pages show builds custom wheels",
        _the_reply_names_a_custom_wheel_building_shop,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
