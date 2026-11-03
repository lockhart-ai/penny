"""Plan mode (#2933): checking before saving leaves the store untouched.

The contract is `docs/eval-case-design.md`.

**ONE case**, ``plan-before-write-leaves-store-untouched``.  A collection already holds one
entry; the live turn gives an updated value for that SAME thing and, in the same breath, asks
Penny to check what would happen before she actually saves it.  The obvious write here would
MERGE onto the existing entry (an exact-key update) — the shape #2933 built `plan` for: a
call whose outcome the user wants to see before it lands, because the identity ladder's own
merges can land on an entry the user didn't expect.

**What this case asserts.  Two claims.**

- STORE (facts asserted, survival): the collection ends holding EXACTLY what it held when the
  turn started — same key, same content, no new entry.  A plan-only turn that actually wrote
  the new value, or wrote it under a second key, fails this openly.
- REPLY (what reaches the user): the reply names at least one of the two values in play — the
  entry's CURRENT content or the NEW content the user just gave — so "I checked" is backed by
  something the user can verify, not an empty acknowledgement.  Never a claim on which tool she
  called or how she worded it beyond that.

What is NOT claimed: that she calls `collection_write{plan: true}` specifically, or declines to
call `collection_write` at all — the store's own end state is the fact, and the tool sequence is
variance, measured and reported.

REPORT-ONLY (``min_pass_rate=None``): the ceiling here is the code owner's to accept once the
numbers are read.  The collection, its entries, and the user's own words below are all invented
for this case and unrelated to any real conversation, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "plan-before-write"

_COLLECTION = "garden-notes"
_SEEDED_KEY = "succulents"
_SEEDED_CONTENT = "Water every 2 weeks, more often in summer."
_NEW_CONTENT = "every 10 days now, not every 2 weeks"

_USER_NAME = "Odile"
_USER_LOCATION = "Winnipeg, MB"
_USER_TIMEZONE = "America/Winnipeg"

ASK = (
    "the succulents note needs updating — it should say every 10 days now, not every 2 "
    "weeks. before you save that, can you check what it'll actually do to what's already "
    "there? don't save it for real yet, just tell me what would happen."
)
ALSO_PHRASED: tuple[str, ...] = (
    "update the succulents note to every 10 days instead of every 2 weeks, but check first "
    "what that write would do before you actually make it — don't save anything yet.",
    "I want to change the succulents watering note to every 10 days, not every 2 weeks. "
    "before you do it for real, preview what it would change so I can see first.",
)


def _seed_world(db: Database) -> None:
    """The user, and one collection holding the single entry the live turn is about."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1990-01-01",
    )
    db.memories.create_collection(
        _COLLECTION,
        "notes on caring for houseplants",
        extraction_prompt=None,
        schedule=None,
    )
    seed_run = seeded_run_id("seeded-garden-notes")
    require_memory(db, _COLLECTION).write_rule_decided(
        [EntryInput(key=_SEEDED_KEY, content=_SEEDED_CONTENT)],
        author="chat",
        run_id=seed_run,
    )


def _premise() -> Preparer:
    """The collection starts holding exactly the one seeded entry — asserted before the
    live turn runs, so a later miss is the turn's, not the seed's."""

    def probe(penny: Penny) -> None:
        held = require_memory(penny.db, _COLLECTION).read_all()
        assert [(e.key, e.content) for e in held] == [(_SEEDED_KEY, _SEEDED_CONTENT)], (
            f"`{_COLLECTION}` must start with exactly the seeded entry, it holds "
            f"{[(e.key, e.content) for e in held]!r}"
        )

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _collection_entries(entries: list) -> list[tuple[str | None, str]]:
    return [(entry.key, entry.content) for entry in entries if entry.collection == _COLLECTION]


def _store_ends_exactly_as_it_started(sample: SampleObservation, _world: World) -> Answer:
    """STORE: a plan-only turn changes nothing — the collection ends holding exactly the one
    entry it started with, same key and content, no second entry for the "new" value."""
    before = _collection_entries(sample.held_before)
    after = _collection_entries(sample.held)
    return after == before, f"`{_COLLECTION}` started {before!r} and ended {after!r}"


def _reply_names_a_real_value(sample: SampleObservation, _world: World) -> Answer:
    """REPLY: what reaches the user backs up "I checked" with a value they can verify —
    either the entry's current content or the new content they just gave, not an empty
    acknowledgement."""
    folded = sample.reply.casefold()
    names_current = "2 weeks" in folded
    names_new = "10 days" in folded
    return names_current or names_new, f"the reply is: {sample.reply!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_checking_before_saving_leaves_the_store_untouched(
    chat_eval: ChatEval, model: str
) -> None:
    """Asked to check what a write would do before saving it for real, Penny's turn leaves
    the store exactly as it started, and her reply names a real value from the check."""
    cohort = await chat_eval(
        case_id="plan-before-write-leaves-store-untouched",
        behaviour=(
            "In the chat agent, when the user asks Penny to check what a write would do "
            "before saving it for real, the store ends exactly as it started and her reply "
            "names a real value from the check."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(
            name="plan-before-write-leaves-store-untouched", pages=(), keeps=(), excludes=()
        ),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=5,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=120.0,
    )
    cohort.claim(
        "state: the collection ends exactly as it started",
        _store_ends_exactly_as_it_started,
        SpecCategory.STORE,
    )
    cohort.claim(
        "reply: names a real value from the check",
        _reply_names_a_real_value,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(*_MEASURED)
