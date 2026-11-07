"""A past reply claimed a save that never happened, then one more item is added (#2901).

The contract is `docs/eval-case-design.md`.  Sibling of
`test_announced_lookup_not_done.py`/`test_lookup_result_recalled.py` (the lookup half
of #2901's baseline) and `test_really_saved_dedup.py` (this case's own healthy
counterpart): `_build_conversation` renders only reply text, so "I've saved that" and a
real `memory_write` read identically a turn later.  This is the MEMORY-side failure
shape: the earlier reply claimed a save, the seeded run's own prompt-log record carries
NO write call, and nothing exists anywhere in the store — so the three originally-asked
facts survive only in the EARLIER USER TURN's own text (never in Penny's own past reply,
which states them nowhere — keeping every later restatement sourced from the user's
ask, never from her own prior words, per `given.py`'s provenance split).

**The world.**  One earlier exchange: the user asks Penny to remember a wheel build's
specs for next time (a 32-hole rim, DT Swiss Alpine III spokes, 2.0mm gauge); Penny's
reply says only that she saved it, naming no detail — seeded with NO tool call.  The
user then asks for one more detail to be added (the wheel is 700c).  For the addition to
land without losing the earlier three, Penny must notice nothing is actually there yet
and write all four — a model that trusts its own earlier "saved!" and writes only the
new item leaves the first three gone for good.

**Real proper nouns** (`docs/principles.md` #9): a 32-hole rim, DT Swiss Alpine III
spokes, a 2.0mm spoke gauge and a 700c wheel are all real cycling terms.

**What is asserted.  Facts only.**  STORE: afterwards, the store holds all four facts —
the three the user stated first, and the one just added.  PROVENANCE: nothing stored or
said is invented.  Which collection, if any, or whether she checks before writing, is
tool-sequence variance, never asserted.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code
owner's to accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import MemoryType
from penny.database.memory.objects import run_tool_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import EVAL_MODELS, is_seeded_run, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _log_chat_step, _seeded_response
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
_CASE_ID = "said-saved-not-saved"

_USER_NAME = "Theo"
_USER_LOCATION = "Minneapolis, MN"
_USER_TIMEZONE = "America/Chicago"

# The three facts the user states FIRST — real cycling terms, invented only in that
# they describe no specific real customer's real wheel.  Stated in the USER's own
# turn, never in Penny's reply, so a later restatement sources from the ask, not from
# her own earlier words (``given.py``'s provenance split excludes assistant turns).
_RIM_FACT = "32-hole"
_SPOKES_FACT = "alpine iii"
_GAUGE_FACT = "2.0mm"
_NEW_FACT = "700c"

_RUN_NAME = "wheel-specs-save-claimed"
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
    """The one earlier exchange: the ask stating the three facts, Penny's save-claimed
    reply (naming none of them), and that run's own prompt-log record carrying NO
    write call — the shape a claimed-but-never-made save leaves."""
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
    """The premise: the profile is seeded, the earlier run traces NO write call, and
    nothing in the store yet holds any of the three facts — the claimed save never
    happened."""
    db = penny.db
    info = db.users.get_info(TEST_SENDER)
    assert info is not None and info.name == _USER_NAME, (
        f"the seeded profile must name {_USER_NAME!r}"
    )
    seeded_rows = [row for row in db.messages.recent_prompts(limit=20) if is_seeded_run(row.run_id)]
    group = [row for row in seeded_rows if row.run_id == seeded_run_id(_RUN_NAME)]
    assert group, f"the seeded run {_RUN_NAME!r} must have its own prompt-log record"
    names = tuple(dict.fromkeys(name for name, _ in run_tool_calls(group)))
    assert names == (), f"the seeded run must show NO write call, it shows {names}"
    # COLLECTIONS only — the same scope ``_held_entries`` reads the claim's own
    # ``sample.held`` from. A LOG-type memory (``user-messages``, the migration-seeded
    # verbatim mirror of every message) necessarily already holds this text, since it is
    # exactly what the seeded ask said; that is not a write anybody could duplicate into.
    for row in db.memories.list_all():
        if row.type != MemoryType.COLLECTION:
            continue
        memory = db.memory(row.name)
        if memory is None:
            continue
        for entry in memory.read_all():
            text = fold_typography(f"{entry.key or ''} {entry.content}")
            assert _RIM_FACT not in text, f"{row.name!r} must not already hold {_RIM_FACT!r}"


def _held_contains(sample: SampleObservation, token: str) -> bool:
    folded = fold_typography(token)
    return any(folded in fold_typography(entry.text) for entry in sample.held)


def _the_store_holds_all_four_facts(sample: SampleObservation, _world: World) -> Answer:
    """Afterwards, the store holds all four facts: the three the user stated in the
    EARLIER turn (never actually saved, despite the claim) and the one just added.  A
    sample that trusts its own earlier "saved!" and writes only the new item loses the
    first three for good — the failure this case is built to catch."""
    missing = [
        token
        for token in (_RIM_FACT, _SPOKES_FACT, _GAUGE_FACT, _NEW_FACT)
        if not _held_contains(sample, token)
    ]
    return not missing, f"the store never ends up holding {missing}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_adding_one_item_recovers_the_never_saved_earlier_ones(
    chat_eval: ChatEval, model: str
) -> None:
    """A past reply claimed a save that never happened; adding one more item still
    lands all four facts, the never-saved earlier ones included."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=(
            "In the chat agent, when a past reply claimed a save that never "
            "happened, adding one more item lands every detail, the earlier ones "
            "included."
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
    # STORE — the claim this case exists to make: all four facts land.
    cohort.claim(
        "state: the store holds all four wheel-build facts, the never-saved ones included",
        _the_store_holds_all_four_facts,
        SpecCategory.STORE,
    )

    # PROVENANCE — nothing stored or said is invented beyond the ask and the earlier
    # user turn.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
