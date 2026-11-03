"""Turning a running job back into plain storage by clearing its prompt and schedule
(#2894).

The contract is `docs/eval-case-design.md`.

**The failure this reproduces.** Asked to make a job's collection plain storage again,
`collection_configure` refused an empty `prompt` and an empty `schedule` alike ("an
empty string sets nothing"), and a short placeholder prompt was refused as too short.
There was no way to turn a running job back into plain storage without destroying the
collection and losing its entries. The fix lets `prompt=""` or `schedule=""` clear the
WHOLE job — whichever axis the call blanks, the other comes with it, since a schedule
with no prompt has nothing to run and a prompt with no schedule never runs — leaving
the collection as plain storage with its entries intact.

**ONE case**, `job-becomes-plain-storage`. `heirloom-tomatoes` starts as a running
job: a weekly research prompt, a schedule, and three entries it already collected.
The live turn is the user asking to stop it being a job and just keep what's there as
plain storage.

**What this case asserts. Two claims, both facts, both survival.**

- Afterwards `heirloom-tomatoes` has no stored prompt and no schedule — the
  registry's own read of the row, never the reply's wording.
- Its three seeded entries survive, content unchanged — turning off the job must
  never touch what it already collected.

What is NOT claimed: which tool she calls, how many calls it takes, or how she words
the reply — variance, measured in the tool sequence and the reply spread.

REPORT-ONLY (`min_pass_rate=None`). Every name and detail here is invented and
unrelated to any real conversation, because the repo is public.
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
    MechanismRecord,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "job-becomes-plain-storage"

_CONTAINER = "heirloom-tomatoes"

_USER_NAME = "Marguerite"
_USER_LOCATION = "Halifax, NS"
_USER_TIMEZONE = "America/Halifax"

_DESCRIPTION = "heirloom tomato varieties suited to a short growing season"
_PROMPT = (
    "Search for heirloom tomato varieties bred for short growing seasons. For each "
    "one, note the variety name, days to maturity, and a one-line flavor note."
)
_SCHEDULE = "FREQ=WEEKLY;BYDAY=MO;BYHOUR=8"

# The job's three entries, already collected before the live turn — the content this
# case must see survive byte-for-byte, whatever the live turn does to the job itself.
_SEEDED_ENTRIES: tuple[tuple[str, str], ...] = (
    ("brandywine", "Brandywine — 80 days to maturity, rich and smoky flavor."),
    ("cherokee_purple", "Cherokee Purple — 85 days to maturity, sweet and tangy."),
    ("green_zebra", "Green Zebra — 75 days to maturity, tart and bright."),
)

ASK = "can you stop heirloom-tomatoes from being a job? just keep it as plain storage"
ALSO_PHRASED: tuple[str, ...] = (
    "turn heirloom-tomatoes back into plain storage, no prompt and no schedule",
    "heirloom-tomatoes doesn't need to keep running as a job — just hang onto what's already there",
)


def _seed_world(db: Database) -> None:
    """A running job on `heirloom-tomatoes` — a weekly research prompt, a schedule,
    and three entries it already collected."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1988-04-12",
    )
    created_run = seeded_run_id("job-created")
    db.memories.create_collection(
        _CONTAINER,
        _DESCRIPTION,
        extraction_prompt=_PROMPT,
        schedule=_SCHEDULE,
        notify=False,
        created_by_run_id=created_run,
    )
    require_memory(db, _CONTAINER).write_rule_decided(
        [EntryInput(key=key, content=content) for key, content in _SEEDED_ENTRIES],
        author="chat",
        run_id=created_run,
    )


def _premise() -> Preparer:
    """`heirloom-tomatoes` starts as a running job — its own prompt and schedule —
    holding exactly its three seeded entries, asserted before the live turn runs so a
    later miss is the turn's, not the seed's."""

    def probe(penny: Penny) -> None:
        db = penny.db
        row = db.memories.get(_CONTAINER)
        assert row is not None
        assert row.extraction_prompt == _PROMPT
        assert row.schedule == _SCHEDULE
        held = require_memory(db, _CONTAINER).read_all()
        assert {e.key for e in held} == {key for key, _content in _SEEDED_ENTRIES}, (
            f"`{_CONTAINER}` must start with its three seeded entries, "
            f"it holds {[e.key for e in held]!r}"
        )

    return probe


# ── The claims ────────────────────────────────────────────────────────────────


def _job_row(sample: SampleObservation) -> MechanismRecord | None:
    return next((one for one in sample.mechanisms if one.name == _CONTAINER), None)


def _no_prompt_and_no_schedule(sample: SampleObservation, _world: World) -> Answer:
    """STORE: the job is gone — no stored prompt, no schedule — read off the
    registry, never the reply's wording."""
    row = _job_row(sample)
    if row is None:
        return False, f"{_CONTAINER!r} is no longer in the registry"
    return (row.program is None and row.schedule is None), f"{_CONTAINER!r} holds {row!r}"


def _entries_contents(sample: SampleObservation) -> list[str]:
    return [entry.content for entry in sample.held if entry.collection == _CONTAINER]


def _entries_survive(sample: SampleObservation, _world: World) -> Answer:
    """STORE (survival, never restraint): all three seeded entries are still there,
    content unchanged, whatever this turn did to the job's prompt or schedule."""
    held = _entries_contents(sample)
    missing = [content for _key, content in _SEEDED_ENTRIES if content not in held]
    return not missing, f"missing from `{_CONTAINER}`: {missing!r}; it holds {held!r}"


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_running_job_can_become_plain_storage(chat_eval: ChatEval, model: str) -> None:
    """Asked to stop a running job and keep it as plain storage: the job's prompt and
    schedule are both gone, and its entries survive untouched.

    The failure this exists to catch is the measured one: a refusal on the blank
    `prompt` or `schedule` ("an empty string sets nothing"), or a placeholder prompt
    rejected as too short, leaving the job running when the user asked it to stop."""
    cohort = await chat_eval(
        case_id="job-becomes-plain-storage",
        behaviour=(
            "In the chat agent, when the user asks to stop a running job and just "
            "keep its collection as plain storage, Penny clears both the prompt and "
            "the schedule, leaving its entries untouched."
        ),
        area=Area.STANDING_JOBS,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(name="job-becomes-plain-storage", pages=(), keeps=(), excludes=()),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=5,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=120.0,
    )
    cohort.claim(
        "state: the job has no stored prompt and no schedule",
        _no_prompt_and_no_schedule,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the three seeded entries survive, content unchanged",
        _entries_survive,
        SpecCategory.STORE,
    )
    cohort.measure(*_MEASURED)
