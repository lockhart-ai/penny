"""End-to-end: a known routine, with both of its values changed by the ask (#2375, follow-up
added #2494).

**The declared second kind** `docs/eval-case-design.md` names: a SCRIPT of real chat turns —
here a cold ask that changes BOTH of the routine's varying values, the page and the thing to
look for on it, in one turn — with a FOLLOW-UP agreement driven only while no job has been
created yet this run — then the job the script stood up, run through its OWN real cycles
against that page.

The follow-up answers a Penny who offers before acting on a two-parameter rebind: she may
land on an offer ("set it up on the new page and line?") rather than creating the job on the
cold ask alone, and a one-turn script would never answer it.

The taught routine is the ferry timetable watcher of ``transition_world`` — seeded, never
edited: a two-parameter routine whose round bound a page and a line to look for on it ("the
late sailing line", on the harbour timetable).  The cold ask points BOTH parameters somewhere
new: a seasonal-sailings page the routine was never taught on, and a different line on it —
"the night sailing" — which that page carries BESIDE the taught line, at the taught line's own
value.  So a collector that read the wrong line on the right page, or the right line on the
wrong page, gets a real, catchable artifact either way: the taught line's value is on the new
page, and the taught page itself stays live in every register.

This is deliberately the two-parameter cold bind — the suite's least stable stand-up, recorded
as the ``-two-params`` quarantine in ``test_known_routine_new_space.py`` — asked end to end,
through the job's own cycles.  Findings are the point; the numbers go to the code owner.

**Claims read only:** what reached the user (the chat phase's replies and the job's own
sends), the created job's own terms (cadence, end, notify), and provenance of everything
sent or replied — never what a cycle left in the store, which is hers to keep as she wishes
and is measured, not claimed (``ENTRIES_STORED``, #2490), nor which collection the job sits
on, the routine's shape or bound values, or which conversation-machine states the script
passed through on the way (#2375's ruling).

**The pages are real-size** — several kilobytes, the watched line among the notices a real
harbour page carries.  The night sailing's two times are mutually exclusive as numerals, and
each appears exactly once on its page, so a sample that invented a plausible time cannot
satisfy a claim that names the one the page actually carries.

Report-only (``min_pass_rate=None``). All content is synthetic, on an ``example`` domain,
because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import MemoryType
from penny.tests.eval.conftest import EVAL_MODELS, EndToEndArm, EndToEndEval
from penny.tests.eval.end_to_end.world import END_TO_END_WALK, sent_at
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TURNS_DRIVEN,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import CannedPage, datum
from penny.tests.eval.utils.job_end import UNTIL_SUNDAY_NIGHT
from penny.tests.eval.utils.transition_world import (
    _FERRY_DEMONSTRATED,
    _FERRY_SKILL,
    _FERRY_TIMETABLE,
    _FERRY_TIMETABLE_URL,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "end-to-end-two-values"
_FAMILY = "end-to-end"

_BEHAVIOUR = (
    "In a job stood up from a routine Penny already knows, when the ask changes both the page "
    "and the thing to look for, the user is told the new thing from the new page."
)

# ── The new page the ask names, and the two readings its cycles see ──────────

_NEW_URL = "https://harborferries.example/seasonal-sailings"

# The OLD item, carried at the value the routine was taught on — read off the ferry family's
# own demonstrated round so the page and the taught value cannot drift apart.
_OLD_ITEM_LINE = _FERRY_DEMONSTRATED.entry_value
# The marker the "not the late sailing's" half of the first-cycle claim reads — a stable
# substring of the old item's line.
_OLD_ITEM_MARKER = "not scheduled"

# The NEW item, and the two readings its value takes.
_NEW_ITEM_TOKEN = "night sailing"
_NEW_BASELINE_TIME = "9:40"
_NEW_MOVED_TIME = "10:15"
_BASELINE_SPAN = f"Night sailing: departs {_NEW_BASELINE_TIME}pm, from the north pontoon."
_MOVED_SPAN = f"Night sailing: departs {_NEW_MOVED_TIME}pm, from the north pontoon."

_SAILINGS_LINK = f"[Seasonal sailings]({_NEW_URL})"

# Real-size: a title line, a description, the old item's line and the new item's side by side,
# a link back to the taught timetable, then the notices a real harbour page carries — fares,
# dock work, bike boarding, passes, the coffee window, weather holds, lost property, group
# bookings.  Several kilobytes, the watched line among all of it.
_BASELINE_PAGE = CannedPage(
    match="seasonal-sailings",
    text=(
        "Title: Seasonal sailings — harbor ferries | harborferries\n"
        f"{_NEW_URL}\n"
        "\n"
        "Seasonal notes for a fictional harbour authority, revised each Monday.\n"
        f"{_SAILINGS_LINK}\n"
        "\n"
        f"{_OLD_ITEM_LINE}\n"
        f"{_BASELINE_SPAN}\n"
        f"[Timetable home]({_FERRY_TIMETABLE_URL})\n"
        "\n"
        "Fares are collected at the gangway; cards only after dusk.\n"
        "Dock two is closed for resurfacing until further notice — the pontoon landing is in "
        "use instead.\n"
        "Bikes board free before six; a supervised rack sits mid-deck.\n"
        "Commuter passes are sold at the kiosk by the north gate and cover both landings.\n"
        "The coffee window on the upper deck opens half an hour after the first crossing.\n"
        "Weather holds are announced on the boards and rarely last past an hour.\n"
        "Lost property is held at the office by the pontoon ramp for thirty days.\n"
        "Group bookings of ten or more should call the harbour office ahead.\n"
        "This fictional authority runs one fleet; times drift with the tide table.\n"
    ),
)
_MOVED_PAGE = datum(_BASELINE_PAGE, _NEW_BASELINE_TIME, _NEW_MOVED_TIME)

# The entry condition is COLD apart from the routine — seeded, one skill — so the job's two
# cycles are: the new page as the ask names it, then the night sailing's moved reading.  The
# TAUGHT timetable rides in every register beside the new page: a cycle that reads it gets the
# taught page back, and the taught line sits on the new page itself, so a wrong-line read is a
# real, catchable artifact either way.
_CYCLES = (
    [_BASELINE_PAGE, _FERRY_TIMETABLE],
    [_MOVED_PAGE, _FERRY_TIMETABLE],
)

# The ground the SCRIPT's own turn is answered against — both pages, installed before the ask
# runs.  ``keeps``/``excludes``/``answers`` are all empty by report: the ask sets a job running,
# so it asks to be TOLD rather than to be given a value back, and it names nothing to exclude.
_WORLD = World(name=_CASE_ID, pages=(_BASELINE_PAGE, _FERRY_TIMETABLE), keeps=(), excludes=())

# The terms the ask gives: fires hourly, stops sunday night, tells the user.  Read against
# ``_the_new_job`` in ``assertions.py`` — existentially, since "a job she created" is the
# claim, never "exactly one".
_CADENCE_SECONDS = 3600
_ASKED_END = UNTIL_SUNDAY_NIGHT

# ── The five wordings of the one cold ask ─────────────────────────────────────
#
# What varies is how a person says it: which verb opens it, where the terms sit relative to
# the address, "changes" or "moves".  What does NOT vary is the page and the item (both named
# in every wording — BOTH of the routine's varying values are supplied by the ask), the
# cadence, the end condition and the telling clause.
_ASKS: tuple[str, ...] = (
    f"can you watch {_NEW_URL} and let me know when the night sailing changes? every hour "
    "until sunday night",
    f"keep an eye on the night sailing at {_NEW_URL} — hourly, until sunday night — and tell "
    "me when it changes",
    f"could you check {_NEW_URL} every hour until sunday night and let me know if the night "
    "sailing moves?",
    f"watch the night sailing on {_NEW_URL} and tell me when it changes please — hourly, up "
    "to sunday night",
    f"i'd like the night sailing at {_NEW_URL} checked every hour until sunday night, with a "
    "message when it changes",
)

_CADENCE_TOKEN = "hour"
_END_TOKEN = "sunday night"
_TELLING_CLAUSES = ("let me know", "tell me", "message")

# The FOLLOW-UP (#2494): driven only while the cold ask above hasn't already created the job —
# the answer to a Penny who offers to rebind the routine onto the new page and line and waits,
# rather than acting on the cold ask's own terms.
_AGREEMENT = "yes, set it up as a new job on that page and line"


def _assert_every_wording_carries_the_facts() -> None:
    """Every one of the five wordings carries the facts the claims below read — the page, the
    item, the cadence, the end and the telling clause — and none of them names the OLD item,
    so a dropped fact or a pointed-at-the-old-item ask fails in this assertion rather than as
    a puzzling score after a paid run."""
    for ask in _ASKS:
        assert _NEW_URL in ask, f"{_CASE_ID}: this wording names no page — {ask!r}"
        assert _NEW_ITEM_TOKEN in ask, f"{_CASE_ID}: this wording names no item — {ask!r}"
        missing = [token for token in (_CADENCE_TOKEN, _END_TOKEN) if token not in ask]
        assert not missing, f"{_CASE_ID}: this wording gives none of {missing} — {ask!r}"
        assert any(clause in ask for clause in _TELLING_CLAUSES), (
            f"{_CASE_ID}: this wording never asks to be told — {ask!r}"
        )
        assert "late sailing" not in ask and _OLD_ITEM_MARKER not in ask, (
            f"{_CASE_ID}: this wording names the old item — {ask!r}"
        )
    # The new page must actually carry the old item's line at the taught value — that is what
    # makes the "not the late sailing's" half of the first-cycle claim below able to fail —
    # and each of the new item's times must appear exactly once on its own page (``datum``
    # enforces the second).
    assert _OLD_ITEM_MARKER in _BASELINE_PAGE.text, (
        f"{_CASE_ID}: the new page carries no old item, so the negative half of the first "
        "cycle's claim has nothing to be false of"
    )


_assert_every_wording_carries_the_facts()


def _seed(db: Database) -> None:
    """The entry condition: no routine taught by THIS seed and no job running — the ferry
    routine is laid into the registry by the runner's own ``seed_skills`` step, after this
    seed returns.  Asserted rather than assumed, so a world that drifted fails here rather
    than after a live run.

    Checked against COLLECTIONS only, never the whole store — a fresh eval database is built
    exactly like production (``create_tables()`` then ``migrate()``), and migrations seed
    system log memories (``penny-messages``, the browse-results log, …) into every database,
    cold teach or not."""
    assert not db.skills.list_all(), "the registry must start empty before seed_skills runs"
    collections = [row for row in db.memories.list_all() if row.type == MemoryType.COLLECTION]
    assert not collections, f"no collection may exist yet for a cold entry, found {collections}"


_ARMS = tuple(
    EndToEndArm(
        label=ask,
        turns=(ask, _AGREEMENT),
        seed=_seed,
        seed_skills=[_FERRY_SKILL],
        world=_WORLD,
        cycles=_CYCLES,
    )
    for ask in _ASKS
)


# ── The claims ───────────────────────────────────────────────────────────────


def _told_the_night_sailing_time_by_the_first_cycle(
    sample: SampleObservation, _world: World
) -> Answer:
    """By the end of the first cycle the user has been told the night sailing's time — and
    not the late sailing's, whose line sits on the same page at the value the routine was
    taught on.

    The positive half reads the night sailing's time in any chat reply OR any notification so
    far: the chat phase's replies with every cycle's own send stripped back out (``reply``
    joins both), then the messages the first cycle put on the queue — nothing from a later
    cycle counts, because "by the end of the first cycle" is the claim's own bound.  The
    negative half reads the same surfaces for the late sailing's marker: a cycle that read
    the wrong line on the right page told the user the wrong thing."""
    chat_replies = sample.reply
    for message in sample.notifications:
        if message in chat_replies:
            chat_replies = chat_replies.replace(message, "")
    notifications_so_far = sent_at(sample.cycles, 0)
    told_surfaces = [chat_replies, *notifications_so_far]
    night = any(_NEW_BASELINE_TIME in surface for surface in told_surfaces)
    late = any(_OLD_ITEM_MARKER in surface for surface in told_surfaces)
    return night and not late, (
        f"the night sailing's {_NEW_BASELINE_TIME} told: {night}; "
        f"the late sailing's line told: {late}"
    )


def _told_the_moved_time_when_it_moved(sample: SampleObservation, _world: World) -> Answer:
    """When the time moved, the user was told the moved time — the messages sent on the
    second cycle carry the moved departure the page now posts.  Read off the cycle's own
    sends, so a cycle that ran and sent nothing, or sent the old time, fails here."""
    sent = sent_at(sample.cycles, 1)
    told = [message for message in sent if _NEW_MOVED_TIME in message]
    return bool(told), f"sent on the moved cycle: {sent}; carrying {_NEW_MOVED_TIME}: {told}"


_MEASURED = (
    END_TO_END_WALK,
    TOOL_SEQUENCE,
    ENTRIES_STORED,
    JOB_TERMS,
    REPLY_SPREAD,
    TURNS_DRIVEN,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_known_routines_job_records_the_new_thing_from_the_new_page(
    end_to_end_eval: EndToEndEval, model: str
) -> None:
    """A cold ask changing BOTH of the routine's varying values — the page and the line to
    look for on it — with terms, and an agreement driven only while no job exists yet; then
    the job the script stood up runs for real across the new page, whose watched value
    holds, then moves, while the taught line beside it stays put."""
    cohort = await end_to_end_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        arms=list(_ARMS),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — the job's own terms, read existentially over whatever it created.
    cohort.assert_a_new_job_fires_every(_CADENCE_SECONDS)
    cohort.assert_a_new_job_notifies()
    cohort.assert_a_new_job_ends_when_asked(_ASKED_END)

    # SENT — the two cycles, each read as what reached the user at that position, union
    # over however many jobs the script created.  What a cycle left in the store is hers,
    # measured as variance (``ENTRIES_STORED``), never claimed (#2490).
    cohort.claim(
        "state: by the end of the first cycle the user has been told the night sailing's "
        "time, and not the late sailing's",
        _told_the_night_sailing_time_by_the_first_cycle,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: when the time moved, the user was told the moved time",
        _told_the_moved_time_when_it_moved,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
