"""A new per-city job must never be redirected onto a storage-only trip collection whose
name it merely contains (#2893).

The contract is `docs/eval-case-design.md`.

**The prod shape, anonymised.** A storage-only collection — entries only, no `prompt`, no
`schedule` — already existed, named for a trip by its season and year
(``spring_2027``). The user then asked for a separate research job for each city on that
trip. `collection_configure`'s duplicate-job check (`similarity/dedup.py` `is_same_job`)
read the new, longer name as "the same job" as the existing storage collection purely
because its tokens contained the storage collection's own — and silently updated the
storage collection in place with one city's job, overwriting its settings (and the next
city's job overwrote it again). The fix (#2893) is two deterministic layers: the duplicate
check only ever compares against a collection that already has a job, and a name that adds
a real distinguishing token is read as a different job rather than noise.

**The claims.** PRESERVATION — the trip's storage collection still holds every itinerary
entry it was seeded with. STATE — it still has no `prompt` and no `schedule`: whatever jobs
the turn stood up for Lisbon, Porto and Seville, none of them landed ON this collection.

**What is NOT claimed, and why.** Whether the turn actually creates three new collections,
what it calls them, and whether every city gets its own job are all ROUTES — the model is
free to ask a clarifying question, create fewer jobs than asked, or phrase the new jobs
however it likes — so the tool sequence and the reply are measured, never asserted
(`docs/principles.md` §4.1, §4.3). This case's only fact is about the ONE collection that
already existed when the turn began.

The host is real (a spring itinerary across Lisbon, Porto and Seville); nothing here is
drawn from any real person's own trip — the repo is public.

REPORT-ONLY (``min_pass_rate=None``): the ceiling this run proposes is the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    MechanismRecord,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "standing-collection-identity"

_CASE_ID = "standing-job-skips-storage-match"

# A storage-only collection — entries only, no prompt, no schedule — named for the trip by
# its season and year, the exact shape the prod collection was in.  Seeded through
# ``World.stores`` (``seed_collection`` with no ``extraction_prompt``/``schedule``), so it
# is plain storage from the first turn, never a job.
_TRIP = SynthCollection(
    "spring_2027",
    "Itinerary for the spring 2027 trip: flights, hotels, and the train legs between cities.",
    entries=(
        "Lisbon hotel — Alfama district, 3 nights, check in Mar 14.",
        "Porto hotel — Ribeira district, 2 nights, check in Mar 17.",
        "Seville hotel — Santa Cruz district, 2 nights, check in Mar 19.",
        "Flight home — Mar 21, depart Seville.",
    ),
)
_HELD_KEYS = tuple(key for key, _ in _TRIP.keyed)

_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=(), stores=(_TRIP,))

# Every wording names the trip collection's own name, the way the user who triggered the
# prod bug did — "my spring_2027 trip" — so a turn reaching for a name reuses it, which is
# exactly the shape the duplicate check has to get right.
_ASK = (
    "Can you set up a research job for each city on my spring_2027 trip — Lisbon, Porto, "
    "and Seville — checking each morning for good food and market tips?"
)
_ALSO_PHRASED = (
    "For my spring_2027 trip, set up a separate watch for each city — Lisbon, Porto, "
    "Seville — that checks every morning for good food and market tips.",
    "I need three jobs for spring_2027: one each for Lisbon, Porto, and Seville, checking "
    "daily for food and market recommendations.",
    "Set up morning research jobs for Lisbon, Porto, and Seville — these are the cities on "
    "my spring_2027 trip — looking for food and market tips.",
    "Can you create a daily research job for each spring_2027 city — Lisbon, Porto, "
    "Seville — to look for food and market finds?",
)

_BEHAVIOUR = (
    "In chat, when the user asks for a new per-city research job on a trip that already has "
    "a storage-only collection named for it, Penny creates the new jobs without turning that "
    "storage collection into one of them."
)


def assert_trip_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs: the trip
    collection holds exactly its four seeded itinerary entries, and carries no prompt and no
    schedule — plain storage, not yet a job."""
    held = sorted(entry.key or "" for entry in require_memory(penny.db, _TRIP.name).read_all())
    expected = sorted(_HELD_KEYS)
    assert held == expected, f"{_CASE_ID}: {_TRIP.name} must hold {expected}, it holds {held}"
    row = penny.db.memories.get(_TRIP.name)
    assert row is not None and row.extraction_prompt is None and row.schedule is None, (
        f"{_CASE_ID}: {_TRIP.name} must start as plain storage (no prompt, no schedule)"
    )


# ── The claims, as pure functions over one sample ─────────────────────────────


def _trip_row(sample: SampleObservation) -> MechanismRecord | None:
    """The trip collection's own registry row as the sample left it, or ``None`` if the turn
    removed it from the registry altogether."""
    return next((one for one in sample.mechanisms if one.name == _TRIP.name), None)


def _the_trip_collection_keeps_its_entries(sample: SampleObservation, _world: World) -> Answer:
    """The trip collection still holds exactly its four seeded itinerary entries — nothing
    the turn did to set up per-city jobs may drop or rewrite what it already held."""
    held = {entry.key for entry in sample.held if entry.collection == _TRIP.name}
    expected = set(_HELD_KEYS)
    return held == expected, f"{_TRIP.name} holds {sorted(held)}"


def _the_trip_collection_still_has_no_job(sample: SampleObservation, _world: World) -> Answer:
    """The trip collection still carries no `prompt` and no `schedule` — whatever jobs the
    turn stood up for the named cities, none of them landed ON this collection.  A violating
    sample is nameable: one that read the new, longer name as the same job as this one and
    wrote a city's prompt and schedule onto it."""
    row = _trip_row(sample)
    if row is None:
        return False, f"{_TRIP.name!r} is no longer in the registry"
    has_job = row.program is not None or row.schedule is not None
    return not has_job, f"program={'set' if row.program else 'empty'}, schedule={row.schedule!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_new_citys_job_never_lands_on_the_trips_storage_collection(
    chat_eval: ChatEval, model: str
) -> None:
    """A storage-only trip collection, and an ask for a separate research job per city on
    that same trip — the exact shape that let a new job's name read as the same job as the
    collection it was never meant to touch."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.STANDING_JOBS,
        model=model,
        prepare=assert_trip_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE
    cohort.claim(
        "state: the trip collection still holds every seeded itinerary entry",
        _the_trip_collection_keeps_its_entries,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the trip collection still has no prompt and no schedule",
        _the_trip_collection_still_has_no_job,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — how many jobs the turn creates, and what
    # it calls them, are routes.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
