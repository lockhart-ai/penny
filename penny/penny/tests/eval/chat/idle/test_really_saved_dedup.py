"""Facts really WERE saved earlier — adding one more must not duplicate them (#2901).

The contract is `docs/eval-case-design.md`.  The healthy counterpart to
`test_said_saved_not_saved.py`: same ask, same earlier conversation shape, but this
time the seeded run's own prompt-log record carries a REAL `memory_access` write call and
the store already holds the three facts it wrote.  The user asks for the same one more
detail to be added.  The failure this guards against is the opposite of the sibling
case's: a model that cannot tell "I already saved this" from "I only said I did" may
rewrite the three facts it already holds alongside the new one, leaving duplicates —
exactly the confusion #2888 exists to prevent, read from its other side.

**The world.**  Identical shape to `test_said_saved_not_saved.py` — a 32-hole rim, DT
Swiss Alpine III spokes, a 2.0mm gauge — but the earlier run's write really happened:
`db.memories.create_collection` + `Memory.write_rule_decided` lay the entry down through
the SAME store chokepoints a live turn would use, and the seeded prompt-log record
carries the matching `memory_access` write call and its real result.

**Real proper nouns** (`docs/principles.md` #9): same real cycling terms as the sibling
case.

**What is asserted.  Facts only.**  STORE: afterwards, the store holds each of the four
facts EXACTLY ONCE — the three that were already there, untouched, and the one just
added.  PROVENANCE: nothing stored or said is invented.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.memory.objects import run_tool_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
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
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "said-vs-did"
_CASE_ID = "really-saved-dedup"

_USER_NAME = "Theo"
_USER_LOCATION = "Minneapolis, MN"
_USER_TIMEZONE = "America/Chicago"

_RIM_FACT = "32-hole"
_SPOKES_FACT = "alpine iii"
_GAUGE_FACT = "2.0mm"
_NEW_FACT = "700c"

_COLLECTION = "wheel_build_specs"
_COLLECTION_DESCRIPTION = "Wheel build specs for next time"
_ENTRY_KEY = "wheel_build"
_ENTRY_CONTENT = f"{_RIM_FACT} rim, dt swiss {_SPOKES_FACT} spokes, {_GAUGE_FACT} gauge"

_RUN_NAME = "wheel-specs-save-real"
_PRIOR_ASK = (
    "hey can you remember my wheel build specs for next time — "
    f"{_RIM_FACT} rim, dt swiss {_SPOKES_FACT} spokes, {_GAUGE_FACT} gauge"
)
_PRIOR_REPLY = "Got it, I've saved your wheel build specs for next time!"

_ASK = f"oh also jot down that it's a {_NEW_FACT} wheel"
_ALSO_PHRASED = (
    f"oh and add that it's {_NEW_FACT}",
    f"one more thing — it's a {_NEW_FACT} rim size",
    f"also note it's running {_NEW_FACT}",
    f"forgot to say, the wheel size is {_NEW_FACT}",
)

for _wording in (_ASK, *_ALSO_PHRASED):
    for _leak in (_RIM_FACT, _SPOKES_FACT, _GAUGE_FACT):
        assert _leak not in _wording.casefold(), (
            f"an ask wording must not name {_leak!r}: {_wording!r}"
        )


def _seed_world(db: Database) -> None:
    """The one earlier exchange, this time with a REAL write behind it: the collection
    created and its entry written through the store's own chokepoints, and the seeded
    run's own prompt-log record carrying the matching `memory_access` write call and its
    real result — the shape a real save leaves."""
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
    db.memories.create_collection(_COLLECTION, _COLLECTION_DESCRIPTION, created_by_run_id=run_id)
    require_memory(db, _COLLECTION).write_rule_decided(
        [EntryInput(key=_ENTRY_KEY, content=_ENTRY_CONTENT)], author="chat", run_id=run_id
    )

    conversation: list[dict] = [{"role": "user", "content": _PRIOR_ASK}]
    write_args = {
        "target": _COLLECTION,
        "action": "write",
        "payload": {"entries": [{"key": _ENTRY_KEY, "content": _ENTRY_CONTENT}]},
    }
    call_id = "call-seeded-wheel-write"
    call = _wire_tool_call(call_id, "memory_access", write_args)
    _log_chat_step(
        db, run_id=run_id, messages=conversation, response=_seeded_response(tool_calls=[call])
    )
    result_text = f"Saved to {_COLLECTION}: {_ENTRY_KEY} — {_ENTRY_CONTENT}"
    conversation = [
        *conversation,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        _tool_result_turn(call_id, result_text),
    ]
    _log_chat_step(
        db, run_id=run_id, messages=conversation, response=_seeded_response(_PRIOR_REPLY)
    )


def _probe_world(penny: Penny) -> None:
    """The premise: the profile is seeded, the earlier run traces exactly one real
    `memory_access` write call, and the store already holds the three facts it wrote."""
    db = penny.db
    info = db.users.get_info(TEST_SENDER)
    assert info is not None and info.name == _USER_NAME, (
        f"the seeded profile must name {_USER_NAME!r}"
    )
    seeded_rows = [row for row in db.messages.recent_prompts(limit=20) if is_seeded_run(row.run_id)]
    group = [row for row in seeded_rows if row.run_id == seeded_run_id(_RUN_NAME)]
    assert group, f"the seeded run {_RUN_NAME!r} must have its own prompt-log record"
    names = tuple(dict.fromkeys(name for name, _ in run_tool_calls(group)))
    assert names == ("memory_access",), (
        f"the seeded run must trace one memory_access write call, it shows {names}"
    )
    held = require_memory(db, _COLLECTION).read_all()
    assert len(held) == 1 and held[0].content == _ENTRY_CONTENT, (
        f"{_COLLECTION!r} must already hold the one real entry, it holds {held}"
    )


def _count_holding(sample: SampleObservation, token: str) -> int:
    folded = fold_typography(token)
    return sum(1 for entry in sample.held if folded in fold_typography(entry.text))


def _the_store_holds_each_fact_exactly_once(sample: SampleObservation, _world: World) -> Answer:
    """Afterwards, the store holds each of the four facts EXACTLY ONCE — the three
    that were already there, untouched, and the one just added.  A sample that
    rewrites the already-saved facts alongside the new one leaves a second entry
    carrying them, which is the duplicate this case is built to catch."""
    counts = {
        token: _count_holding(sample, token)
        for token in (_RIM_FACT, _SPOKES_FACT, _GAUGE_FACT, _NEW_FACT)
    }
    wrong = {token: count for token, count in counts.items() if count != 1}
    return not wrong, f"expected each fact stored exactly once, counted {counts}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_adding_one_item_never_duplicates_the_already_saved_ones(
    chat_eval: ChatEval, model: str
) -> None:
    """Facts really were saved earlier; adding one more item never duplicates what's
    already there."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=(
            "In the chat agent, when earlier details really were saved, adding one "
            "more item never duplicates what's already there."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_probe_world,
        world=World(name=_CASE_ID, pages=(), keeps=(), excludes=()),
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — the claim this case exists to make: no duplicates.
    cohort.claim(
        "state: the store holds each of the four wheel-build facts exactly once",
        _the_store_holds_each_fact_exactly_once,
        SpecCategory.STORE,
    )

    # PROVENANCE — nothing stored or said is invented beyond the ask and the earlier
    # user turn.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
