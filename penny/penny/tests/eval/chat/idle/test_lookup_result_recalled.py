"""A past reply really DID the lookup and answered — the control half of #2901.

The contract is `docs/eval-case-design.md`.  Sibling of `test_announced_lookup_not_done.py`:
same world shape, opposite history — here the earlier run's own prompt-log record
carries a REAL `browse` call and a real result, and the reply already named the shop
and its price.  The user's follow-up just asks for that same fact again.  This is the
baseline's CONTROL: it should already hold on `main`, since the fact sits in the plain
reply TEXT `_build_conversation` renders today, with no native tool-call history needed
to recover it — #2888 is not expected to move this case, only
`test_announced_lookup_not_done.py`'s.

**Why no blanket reply-sourcing claim here.**  A sample that simply recalls its OWN
earlier reply (rather than re-browsing or calling `run_list`/`run_get`) is CORRECT
behaviour, but `given.py`'s provenance split excludes assistant turns from what a round
was "given" by design ("a value Penny invents early in a turn rides into the history and
would then source itself") — so `assert_every_value_in_the_reply_is_sourced()` would
read an honest echo of her own true past answer as unsourced.  This case therefore makes
one narrow claim (the price reaches the user again) and leaves the route — echo,
re-browse, or `run_list`/`run_get` — as pure tool-sequence variance, never asserted.

**Real proper nouns** (`docs/principles.md` #9): same world as the sibling case —
Minneapolis, Perennial Cycle, Varsity Bike & Transit, DT Swiss Alpine III spokes, a Park
Tool TM-1.  The price and turnaround are the fixture's own invented numbers.

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
from penny.tests.eval.end_to_end.world import (
    _log_chat_step,
    _seeded_response,
    _tool_result_turn,
    _wire_tool_call,
)
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
_CASE_ID = "lookup-result-recalled"

_USER_NAME = "Theo"
_USER_LOCATION = "Minneapolis, MN"
_USER_TIMEZONE = "America/Chicago"

_SHOP = "Perennial Cycle"
_SHOP_HOST = "yelp.com/biz/perennial-cycle-minneapolis"
_PRICE_FACT = "42"
_TURNAROUND_FACT = "2 business days"

_OTHER_SHOP = "Varsity Bike & Transit"
_OTHER_HOST = "yelp.com/biz/varsity-bike-and-transit-minneapolis"

_RUN_NAME = "wheel-shop-lookup-done"
_PRIOR_ASK = (
    "can you find a shop in minneapolis that builds custom wheels? i want a "
    "32-hole rim laced with dt swiss alpine iii spokes"
)
_PRIOR_REPLY = (
    f"Found one — Perennial Cycle in the Longfellow neighborhood builds wheels to "
    f"order, ${_PRICE_FACT} per wheel labor, about {_TURNAROUND_FACT} turnaround."
)

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

_ASK = "oh nice, how much was that again?"
_ALSO_PHRASED = (
    "wait what was the price on that one?",
    "remind me how much that wheel build runs?",
    "sorry what'd you say the cost was?",
    "how much were they charging again?",
)

_LEAKS = (_SHOP, _SHOP_HOST, _PRICE_FACT, _OTHER_SHOP, _OTHER_HOST)
for _wording in (_ASK, *_ALSO_PHRASED):
    for _leak in _LEAKS:
        assert _leak.casefold() not in _wording.casefold(), (
            f"an ask wording must not name {_leak!r}: {_wording!r}"
        )


def _seed_world(db: Database) -> None:
    """The one earlier exchange: the ask, Penny's DONE reply, and that run's own
    prompt-log record carrying a real `browse` call and its real result."""
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
    conversation: list[dict] = [{"role": "user", "content": _PRIOR_ASK}]
    browse_args = {
        "items": [
            {
                "query": "custom wheel building minneapolis",
                "looking_for": "a shop that builds custom wheels",
            }
        ]
    }
    call_id = "call-seeded-wheel-browse"
    call = _wire_tool_call(call_id, "browse", browse_args)
    _log_chat_step(
        db, run_id=run_id, messages=conversation, response=_seeded_response(tool_calls=[call])
    )
    result_text = (
        f"You searched for {browse_args['items'][0]['query']!r} (browse result)\n{_SHOP_PAGE.text}"
    )
    conversation = [
        *conversation,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        _tool_result_turn(call_id, result_text),
    ]
    _log_chat_step(
        db, run_id=run_id, messages=conversation, response=_seeded_response(_PRIOR_REPLY)
    )


def _probe_world(penny: Penny) -> None:
    """The premise: the profile is seeded, and the earlier run's own prompt-log record
    traces exactly the one real `browse` call behind its reply."""
    db = penny.db
    info = db.users.get_info(TEST_SENDER)
    assert info is not None and info.name == _USER_NAME, (
        f"the seeded profile must name {_USER_NAME!r}"
    )
    seeded_rows = [row for row in db.messages.recent_prompts(limit=20) if is_seeded_run(row.run_id)]
    group = [row for row in seeded_rows if row.run_id == seeded_run_id(_RUN_NAME)]
    assert group, f"the seeded run {_RUN_NAME!r} must have its own prompt-log record"
    names = tuple(dict.fromkeys(name for name, _ in run_tool_calls(group)))
    assert names == ("browse",), f"the seeded run must trace one browse call, it shows {names}"


def _the_reply_restates_the_price(sample: SampleObservation, _world: World) -> Answer:
    """The reply restates the price fact — whichever way it got there (an honest echo
    of her own true earlier answer, a `run_list`/`run_get` check, or a fresh re-browse
    are all fine; that route is tool-sequence variance, never asserted here)."""
    said = fold_typography(sample.reply)
    return _PRICE_FACT in said, f"the reply never restates the price — reply: {sample.reply!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_completed_lookups_fact_is_recalled_on_followup(
    chat_eval: ChatEval, model: str
) -> None:
    """A past reply really did the lookup and answered; a later follow-up asking for
    that same fact again still gets it — the healthy control beside the announced-but-
    not-done sibling."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=(
            "In the chat agent, when a past reply really did a lookup and answered, "
            "a later follow-up asking for that same fact again still gets it."
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

    # PROVENANCE — the one thing the ask owes: the price, restated.  No blanket
    # reply-sourcing claim here — see the module docstring on why an honest echo of
    # her own earlier true answer must not read as unsourced.
    cohort.claim(
        "reply: it restates the price from the earlier, completed lookup",
        _the_reply_restates_the_price,
        SpecCategory.PROVENANCE,
        kind="reply",
    )

    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
