"""Operating a job that is already running, and reading jobs back: eight cases (#2008, tranche 3;
#2215; #2365).

The contract is `docs/eval-case-design.md`.

**One case per ACTION on a running job.**  A correct sample for *retire* archives the job,
which is exactly what a correct sample for *flip the switch* must not do — so a sample that is
right for one is wrong for another, and each action is a case with its own sentence.

| action | case | what the row must show |
|---|---|---|
| switch it off | ``standing-notify-off`` | ``notify`` off, the rest of the row as it was |
| switch it on | ``standing-notify-on`` | ``notify`` on, the rest of the row as it was |
| retire it | ``standing-archive`` | ``archived``, and what it gathered still readable |
| re-time it | ``standing-schedule-fix-prior`` | the rule fires at the hour they asked for |
| give it an end | ``standing-end-changed`` | it stops when the ask said, firing when it was due |
| read one back | ``standing-describe-routine`` | the row as it was; the reply names its page |
| list them | ``standing-list-running`` | every row as it was; the reply names each job |
| two of the same kind match | ``standing-ambiguous-job-match`` | neither moved; reply names both |

**The two directions of the switch are two cases, because they start in two worlds.**
``standing-notify-off`` is answered on a job seeded telling the user and
``standing-notify-on`` on the same job seeded quiet; each claims the switch the ask named,
which its own seed holds the other way.  The OFF direction is where the wrong levers are — a
measured sample asked to stop one watch's pings read the state as offering no granularity and
retired the whole job, or reached for the global mute.

**No claim reads where the machine landed, or a run's outcome (#2496, naive baseline).**  There
is no conversation machine left to land on this branch, so every landing claim is dropped
rather than ported — consistent with #2496's ruling on the naive branch generally.

**What is NOT claimed, and why.**  That she called ``collection_configure`` is a ROUTE — many routes
reach one end state, and a rule keyed to that name would simply not fire for a verb nobody
enumerated — so it is measured in the tool sequence and never asserted, where a divergence reads
as a sample worth opening rather than as a claim about which verb was right.

**What each case claims about the rest of the row is its END STATE, read through the store's
own ledger**, never a list of fields written here: every field whose value the row holds now
differs from what the turn found there (``MechanismRecord.moved_this_run``).  The row's settable
surface is wider than any list a case would write — its description, its expiry, its run quota,
the values a rebind binds — and an enumeration exempts whatever it forgot; and it is the value
rather than the call, so a turn that restated a field it was not asked to touch has moved
nothing.  That applies to the retire case too, since archiving is reversible: the same job asked
for again revives this row, so a retire that also rebound the routine or re-timed the cadence
hands back a different job under the same name.

**A field the ledger keeps no prior for is read off the row.**  ``moved_this_run`` compares the
priors the store records, and it records none for a run quota.  So the end case, where a rule
rewritten to carry its own end is a legitimate way to state one, reads the moments the rule
fires at from now on.

**No case claims what the turn did NOT do** (``docs/principles.md`` §4.3).  The world holds its
jobs — one, or three where the ask is to list them — what each gathered and the global switch,
so those are what the cases claim still holds; whether a turn also stood something new up is
its own call, measured in the tool sequence.

**The world is built the way production builds one (#1911/migration 0108: nothing is
pre-seeded; #2455 naive baseline: no skill or routine behind a job).**  Every collection here is
one the user built: the container is created storage-only, then configured through the store's
own metadata write with its plain-text prompt (``StandingJob.program``) naming the page it
checks and what it looks for — the same free-text shape ``seed_known_routine`` writes into
``extraction_prompt``.  A job seeded this way reads the production shape exactly, through
``penny.tests.eval.end_to_end.world.seed_standing_jobs``.

``standing-describe-routine`` reads the same job back rather than operating it.  What it
asserts is what the reply CITES from the job's record — the page its program fetches — that
every value the reply states traces to what the round was given, and the job still as it was;
which read she reaches for, and how she words the walk-through, are measured.

``standing-list-running`` reads THREE jobs back at once: the same kind of job pointed at three
pages, each looking for its own kind of thing.  The reply carries one token per job — the
thing that job looks for, which its page, its bound value and its description all spell and
no other job's record does — and every job, everything each gathered and the global switch
are as the turn found them.

``standing-ambiguous-job-match`` (#2365) is the one case here where the ask does NOT single a
job out.  Two jobs of the SAME kind, each watching its own page for a price drop, stand
side by side; the ask names the kind of job ("the price watch") and nothing that tells the two
apart.  Nothing in the registry resolves this — both rows match the description equally — so
the ruled contract is that neither changes without the user saying which: the claims read that
both rows hold exactly what they were seeded with, and that the reply names both jobs, never
one of them silently.  Whether she actually asks which one, as opposed to naming both and
stopping, or picking one on her own, is a route rather than a fact the world settles — measured
in the tool sequence and the reply's own spread, never asserted.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every page, shop and job is synthetic, on an
``example`` domain, because the repo is public.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import NamedTuple

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.database import Database
from penny.datetime_utils import zone_or_utc
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Preparer,
)
from penny.tests.eval.end_to_end.world import StandingJob, seed_standing_jobs
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
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.job_end import UNTIL_SUNDAY_NIGHT
from penny.tests.eval.utils.worlds import World
from penny.tools.collection_instantiation import next_occurrence

pytestmark = pytest.mark.eval

_OPERATIONS_FAMILY = "standing-collection-operations"
_LEGIBILITY_FAMILY = "standing-collection-legibility"


# The standing job every case operates on: pointed at the page it was already watching,
# running each morning and telling the user when it finds something.  Synthetic throughout
# (the repo is public): an invented shop on an example domain, an invented thing to watch for.
DEMONSTRATED_PAGE = "https://quillmarket.example.com/typewriters"
DEMONSTRATED_WATCH = "portable typewriters"
DEMONSTRATED_ENTRY = "Hermes Baby — 1958 portable, case included, $180."

_FINDS = StandingJob(
    container="typewriter-finds",
    url=DEMONSTRATED_PAGE,
    watched_for=DEMONSTRATED_WATCH,
    description="Portable typewriters newly listed at the Quill Market shop.",
    schedule="FREQ=DAILY;BYHOUR=7",
    notify=True,
    holdings=(
        ("Hermes Baby", DEMONSTRATED_ENTRY),
        ("Olivetti Lettera 32", "Olivetti Lettera 32 — 1963 portable, case included, $220."),
    ),
)

# What the job already holds before any turn runs — read ONCE off the fixture and asserted by
# the premise probe, so a fixture edit cannot leave the kept-its-entries claim naming keys the
# world stopped seeding.
_HELD_KEYS = tuple(key for key, _ in _FINDS.holdings)

# The same job with its own switch OFF — the entry condition of the case that asks for its
# notifications back.  A job's container is named explicitly (#2455 naive baseline: there is
# no routine to derive one from), so this is the same row under the same name, differing in
# that one field.
_QUIET_FINDS = _FINDS._replace(notify=False)

# Two more jobs of the same kind, for the case that asks what is running: each pointed at
# its own page and looking for its own kind of thing, on its own cadence and switch.
_KETTLES = StandingJob(
    container="kettle-finds",
    url="https://brasswick.example.com/kettles",
    watched_for="enamel kettles",
    description="Enamel kettles newly listed at the Brasswick shop.",
    schedule="FREQ=DAILY;BYHOUR=9",
    notify=True,
    holdings=(("Tidewater No. 4", "Tidewater No. 4 — two-quart enamel kettle, $48."),),
)
_ATLASES = StandingJob(
    container="atlas-finds",
    url="https://fernhollow.example.com/atlases",
    watched_for="pocket atlases",
    description="Pocket atlases newly listed at the Fernhollow shop.",
    schedule="FREQ=HOURLY",
    notify=False,
    holdings=(("Marlowe 1931", "Marlowe 1931 — pocket atlas, cloth boards, $35."),),
)

# How the user refers to it: their own words for the job, never its derived name.  The
# derived name renders on the ambient mechanisms line, so resolving those words to that
# collection is a read the turn is expected to make — asking with the container's own
# name would hand the model the answer to half the case.  It lends the read-back case's
# answer token nothing either, which that case enforces on every wording.
_THEIR_WORDS = "the typewriter watch"


# ── Reading a job's schedule as the hour it fires at ──────────────────────────
#
# An RRULE can say the same hour several ways — a ``BYHOUR=`` clause, or a ``DTSTART`` whose
# own time carries it — and which form a re-timing draw reaches for is the model choosing how
# to write a value.  So the claim reads the hour the rule COMES ROUND ON, through production's
# own ``next_occurrence``, rather than matching the clause a draw happened to use.  The anchor
# and zone are fixed here because the question is only which hour of the day the rule names;
# they decide which DAY an occurrence lands on and nothing this case reads.

_RULE_ANCHOR = datetime(2026, 1, 1, tzinfo=UTC)


def _hour_the_rule_fires_at(schedule: str | None) -> int | None:
    """The hour of day a stored rule comes round on, or ``None`` when it names none.

    ``None`` is a real reading rather than an error: a job with no schedule does not run,
    and a rule that will not parse is one the collector could not run either — both are
    honest answers to "what hour does this fire at", and the claim that reads it fails
    naming the rule instead of taking the run down."""
    if not schedule:
        return None
    try:
        occurrence = next_occurrence(schedule, _RULE_ANCHOR, None)
    except ValueError, TypeError:
        return None
    return None if occurrence is None else occurrence.hour


def _the_hour_it_runs_at(job: StandingJob) -> int:
    """The hour a SEEDED job's rule fires at, refusing an unreadable one.

    The claim's reader tolerates a rule that names no hour, because a turn is free to leave
    one; a fixture is not — a world whose job runs at no readable hour would make the whole
    re-timing case a comparison against nothing."""
    hour = _hour_the_rule_fires_at(job.schedule)
    assert hour is not None, f"the seeded rule {job.schedule!r} must name an hour it fires at"
    return hour


# The two hours this job has: the one it was seeded running at, read off the seeded rule
# rather than restated, and the one the ask moves it to.  The pair must be DISTINCT, or the
# re-timing claim is answered by the seed.
_SEEDED_HOUR = _the_hour_it_runs_at(_FINDS)
_FIXED_HOUR = 11

assert _SEEDED_HOUR != _FIXED_HOUR, (
    f"the job must not already run at the hour the ask asks for — both are {_FIXED_HOUR}"
)


# ── Reading the clock times a reply names ─────────────────────────────────────
#
# Clock times in the three forms a reply writes them.  Each yields a 24-hour hour, so a
# reply saying "11am", "11:00" and "11 in the morning" is read the same way whichever it
# reached for — what is being measured is WHICH hour it named, never how it spelled it.

_MERIDIEM = re.compile(r"\b(\d{1,2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b")
_TWENTY_FOUR = re.compile(r"\b(\d{1,2}):\d{2}\b")
_DAYPART = re.compile(r"\b(\d{1,2})\s*(?:o'?clock\s*)?in the (morning|afternoon|evening|night)\b")
_AFTERNOON_PARTS = ("afternoon", "evening", "night")


def _hours_named(reply: str) -> set[int]:
    """Every hour the reply states, as a 24-hour number."""
    folded = reply.lower()
    hours = {
        _to_24(int(hour), meridiem.startswith("p")) for hour, meridiem in _MERIDIEM.findall(folded)
    }
    hours |= {int(hour) for hour in _TWENTY_FOUR.findall(folded)}
    hours |= {
        _to_24(int(hour), part in _AFTERNOON_PARTS) for hour, part in _DAYPART.findall(folded)
    }
    return {hour for hour in hours if 0 <= hour <= 23}


def _to_24(hour: int, afternoon: bool) -> int:
    return (hour % 12) + 12 if afternoon else hour % 12


# ── The case shape ────────────────────────────────────────────────────────────


class _OperationCase(NamedTuple):
    """One thing a user does to a job that is already running, in five wordings.

    ``ask`` and ``also_phrased`` are five wordings of ONE message against one world — the
    cohort's arms.  What varies is only how a person says it; the job, its routine, its
    cadence, its switch and the end state expected of the turn are constant, which is what
    makes the fifteen samples one number and what lets a claim name a value at all.

    ``job`` is the standing job the ask is about, seeded through the production instantiation
    seam.  ``renders`` is the clause its self-state row must carry before the turn runs,
    asserted rather than hoped for, since the model can only reach for a lever the state
    presents: the switch, in the direction it is seeded, for the cases that flip it or retire
    the job — where its presence is also what makes the global mute a WRONG lever rather than
    the only one — the stored rule for the cases about when it runs and when it stops, and the
    switch again for the two read-backs, which move nothing.

    ``answers`` is what the reply owes, and only the two read-back cases owe anything: each
    operation's ask is an INSTRUCTION, and "done, that one's quiet now" is a complete answer.

    ``others`` are the jobs running BESIDE ``job`` — empty everywhere but the case that asks
    what is running, whose answer is every one of them.

    ``pages`` is what a browse returns — empty everywhere but the case whose ask names an
    address, where a turn that opens that address has to find a page there.
    """

    case_id: str
    behaviour: str
    job: StandingJob
    renders: str
    ask: str
    also_phrased: tuple[str, ...]
    answers: tuple[str, ...] = ()
    others: tuple[StandingJob, ...] = ()
    pages: tuple[CannedPage, ...] = ()

    @property
    def jobs(self) -> tuple[StandingJob, ...]:
        """Every job the world holds when the turn begins: the one the ask is about, then the
        ones running beside it."""
        return (self.job, *self.others)

    @property
    def ground(self) -> World:
        """The world every arm of this case is answered against.

        Every field is EMPTY, and each is a report rather than an omission.  ``stores`` is
        empty although the world DOES seed a collection, because a declared store is laid down
        as plain entries and this one is a JOB: its container is stood up through the production
        instantiation seam, with a program, a cadence, a switch and its provenance, which is the
        whole thing these cases operate on.  Declaring it here would seed its holdings a second
        time.  ``pages`` is the case's own, and empty for every ask about a job's
        configuration, which no page answers — a page set there would hand a sample that went
        browsing something to talk about instead of letting the wrong turn read as one.
        ``keeps`` states what a round must have written down and none of these turns is asked
        to write anything; ``excludes`` states what it was told to leave out and none of them
        excludes anything in as many words.  ``answers`` is the case's own: empty for an
        instruction, whose complete answer names nothing."""
        return World(
            name=self.case_id, pages=self.pages, keeps=(), excludes=(), answers=self.answers
        )


def _job_row(sample: SampleObservation, container: str) -> MechanismRecord | None:
    """The standing job's own registry row as the sample left it, or ``None`` if the turn
    took it out of the registry altogether."""
    return next((one for one in sample.mechanisms if one.name == container), None)


def assert_the_operation_world(db: Database, case: _OperationCase) -> None:
    """The case's premise, asserted out loud before the turn runs.

    Every part is the precondition of a claim below: the lever the ask is about RENDERS on
    the job's own self-state row (with nothing there, a measured sample asked to stop one
    watch's pings read the state as offering no granularity and retired the whole job), and
    every job running beside it renders a row of its own; each job already holds what its
    fixture says it gathered, so "it kept them" is a claim about the turn; the job the ask is
    about stores no end, so an end the turn leaves on it is the turn's; and proactive notifications
    are ON, so "they are still on for everything else" is a claim about this turn rather than
    about the world it started in.

    Takes the DATABASE rather than the running Penny so the same assertions run without a
    model: a premise that quietly stopped holding turns a scored claim into a claim about the
    fixture, and the cheapest place to catch that is ``make check``."""
    _assert_the_row_renders(db, case)
    for job in case.jobs:
        _assert_the_job_is_as_seeded(db, case.case_id, job)
    row = db.memories.get(case.job.container)
    assert row is not None and row.expires_at is None and row.max_runs is None, (
        f"{case.case_id}: the job must start with no end, or an end the turn leaves on it "
        "is answered by the seed"
    )
    assert not db.users.is_muted(TEST_SENDER), (
        f"{case.case_id}: notifications must start ON, or the claim that they are still on "
        "is answered by the seed"
    )


def _assert_the_job_is_as_seeded(db: Database, case_id: str, job: StandingJob) -> None:
    """One job holds what its fixture says it gathered — read off the store, which is what
    the claims read afterwards.  There is no routine binding left to check (#2455 naive
    baseline): the job's whole identity is its own row, so what it holds is the whole
    premise."""
    held = sorted(entry.key or "" for entry in require_memory(db, job.container).read_all())
    gathered = sorted(key for key, _ in job.holdings)
    assert held == gathered, f"{case_id}: {job.container} must hold {gathered}, it holds {held}"


def _premise(case: _OperationCase) -> Preparer:
    """The prepare hook: the case's premise, asserted inside the sample it belongs to."""

    def probe(penny: Penny) -> None:
        assert_the_operation_world(penny.db, case)

    return probe


def _assert_the_row_renders(db: Database, case: _OperationCase) -> None:
    """The job's self-state row carries the clause the ask is about.

    Read through the header's own render, so the probe and the surface cannot drift into
    asserting different words for one state; a world that stopped carrying the lever would
    make every sample measure the model's guesswork rather than its reading, and would do
    so silently."""
    rendered = SelfStateHeader(db, TEST_SENDER).render()
    rows = {job.container: _mechanism_row(rendered, job.container) for job in case.jobs}
    missing = sorted(container for container, row in rows.items() if row is None)
    assert not missing, f"{missing} must each render as a mechanism:\n{rendered}"
    row = rows[case.job.container] or ""
    assert case.renders in row, (
        f"{case.job.container} must render {case.renders!r} — it renders {row!r}"
    )


def _mechanism_row(rendered: str, container: str) -> str | None:
    """One job's own row in the rendered self-state, or ``None`` when it has none."""
    return next(
        (line for line in rendered.splitlines() if line.startswith(f"- {container} ")), None
    )


async def _drive(
    chat_eval: ChatEval, model: str, case: _OperationCase, family: str = _OPERATIONS_FAMILY
) -> Cohort:
    """Drive one case on its standing jobs: each as its own apply turn left it, and the
    premise re-asserted before the turn."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        area=Area.STANDING_JOBS,
        model=model,
        seed=seed_standing_jobs(*case.jobs),
        prepare=_premise(case),
        world=case.ground,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=family,
        timeout=240.0,
    )


_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ── The claims, as pure functions over one sample ─────────────────────────────
#
# Every one is a read of what the turn LEFT BEHIND — the job's own row, what it holds, the
# registry around it, the global switch.  None reads a tool NAME: a job cannot change its own
# configuration by accident, so the row already says whether the edit happened, and a rule
# keyed to ``collection_configure`` would simply not fire for a plugin verb nobody enumerated.
#
# They stay LOCAL rather than graduating into ``assertions.py``.  A claim graduates at the
# second CUSTOMER, and the cases below are one behaviour family in one file: a second
# FILE asking one of these questions is what would make it shared, and none has yet.

_ClaimFn = Callable[[SampleObservation, World], Answer]


def _the_switch_is(container: str, *, on: bool) -> _ClaimFn:
    """The job's own notification switch reads the way the ask asked for.

    A violating sample is nameable: one that left it alone and said it had changed it, and —
    the measured one — one that stopped the pings by retiring the whole job, which moves
    ``archived`` and leaves this exactly where it was."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        return row.notifies == on, f"notify is {'on' if row.notifies else 'off'}"

    return answer


def _the_job_is_retired(container: str) -> _ClaimFn:
    """The job is ARCHIVED — a visible tombstone rather than a deletion.

    Archived, never deleted: a retired mechanism stays enumerable and the same job asked for
    again unarchives it, so a turn that removed the row took the user's entries with it.  A
    violating sample is nameable: one that left the job live and merely said it had stopped,
    and one that silenced it instead of retiring it."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} was removed from the registry rather than archived"
        return row.archived, f"{container!r} is still active"

    return answer


def _the_job_is_still_live(container: str) -> _ClaimFn:
    """The job is still RUNNING — not retired, still scheduled, still carrying a program.

    The other half of every edit that changes ONE thing: "she changed what was asked" and
    "she left the job running" are two claims, and a turn that quietly retired the job or
    emptied its program while flipping a flag has done something else entirely, which the
    flag claim alone would call a pass.  A violating sample is nameable: the measured one
    that archived the whole job to stop its pings."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        live = not row.archived and row.schedule is not None and row.program is not None
        return live, (
            f"archived={row.archived}, schedule={row.schedule!r}, "
            f"program={'set' if row.program else 'empty'}"
        )

    return answer


def _nothing_else_moved(container: str, *, asked: frozenset[str]) -> _ClaimFn:
    """Every field on the job's row holds what it held before the turn, except the ones the ask
    named — PRESERVATION of the rest of the job.

    Read as the row's END STATE through the store's own ledger (``moved_this_run``: each field
    whose value now differs from the prior the ledger recorded for it) rather than as a diff
    over fields this file lists: the row's settable surface is wider than any list a case would
    think to write — its description, its expiry, its run quota, the values a rebind binds — so
    an enumeration silently exempts whatever it forgot.  A field the store learns to change
    tomorrow is compared for free, and a field a call merely restated has not moved.

    ``asked`` is the store's own labels for the fields the ask moves, empty for an ask that
    moves none.  A wrong label can only make this claim STRICTER — the named field would count
    as drift — so a typo fails loudly instead of quietly passing.

    A violating sample is nameable, and the rationale names the field: one that flips the switch
    and re-times the job on the way through, one that rebinds it to another routine, one that
    rewrites its program while reporting only the flag, and one that stamps an end date on a job
    nobody asked to bound."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        others = sorted(set(row.moved_this_run) - asked)
        return not others, f"also moved {others}"

    return answer


# The store's own labels for the fields these asks move: an update's field name, and the archive
# flag's (which records its prior under this name, #1946).
_NOTIFIES = frozenset({"notify"})
_SCHEDULE = frozenset({"schedule"})
_ARCHIVED = frozenset({"archived"})
_NONE: frozenset[str] = frozenset()


def _the_rule_fires_at(container: str, hour: int) -> _ClaimFn:
    """The job now comes round at the hour they asked for.

    Read as the hour the RULE fires at rather than as the clause a draw wrote, so a re-timing
    that states the hour through a ``DTSTART`` and one that states it through ``BYHOUR``
    answer the same way — the hour is the fact, the clause is how it was written.  A violating
    sample is nameable: one that reports the move and leaves the rule where it was, and one
    that moves it to some third hour nobody asked for."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        fires = _hour_the_rule_fires_at(row.schedule)
        return fires == hour, f"the rule {row.schedule!r} fires at {fires}"

    return answer


def _it_kept_what_it_gathered(container: str) -> _ClaimFn:
    """Everything the job had already collected is still readable.

    None of these operations is a licence to clear a job out — a watch silenced by quietly
    emptying its collection, or retired by deleting it, has taken something the user can still
    read.  A violating sample is nameable: one that drops the entries along with the job."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = {entry.key or "" for entry in sample.held if entry.collection == container}
        missing = sorted(set(_HELD_KEYS) - held)
        return not missing, f"{container} no longer holds {missing}"

    return answer


def _notifications_are_still_on_everywhere(sample: SampleObservation, _world: World) -> Answer:
    """Proactive notifications are still ON for the user — PRESERVATION of the global switch.

    The neighbouring lever, and the measured wrong one: asked to quiet ONE job, a sample that
    reads the per-collection switch as unavailable reaches for the global mute, which silences
    everything and looks like success from inside the turn.  A violating sample is nameable in
    all three directions — silencing everything to quiet one job, to retire one, or because
    "it checks too early" read as "stop bothering me"."""
    return not sample.muted, "the turn muted notifications for everything"


# The labels the cases share.  Named once because a label is a diff-join key: several copies of
# one sentence are that many chances for a typo to split one claim's history in two.
_STILL_LIVE = "state: the job is still live (active, scheduled, with a program to run)"
_KEPT_ITS_ENTRIES = "state: what it gathered is still there"
_NOTHING_ELSE_MOVED = "state: nothing else on the job's row moved"
_STILL_ON_EVERYWHERE = "state: notifications are still on for everything else"


# ═══ flip the switch ═════════════════════════════════════════════════════════
#
# Asked in as many words to turn one job's notifications off.  The ask is explicit on purpose
# — the vocabulary the mute contracts settled on, where what is wanted is said rather than
# implied — because what this measures is whether the per-collection switch is REACHABLE, not
# whether an oblique phrasing can be decoded.

_NOTIFY_OFF = _OperationCase(
    case_id="standing-notify-off",
    behaviour=(
        "In the chat agent, when the user asks for one running job's notifications to be "
        "turned off, Penny turns that job's own switch off, with the job still running as it "
        "was and proactive notifications still on everywhere else."
    ),
    job=_FINDS,
    renders=SelfStateHeader.MECHANISM_NOTIFIES,
    ask=f"turn off notifications for {_THEIR_WORDS}",
    also_phrased=(
        f"stop the notifications on {_THEIR_WORDS}",
        f"can you turn notifications off for {_THEIR_WORDS}?",
        f"switch notifications off for {_THEIR_WORDS} please",
        f"i don't want notifications from {_THEIR_WORDS} any more",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_turning_notifications_off_silences_only_that_job(
    chat_eval: ChatEval, model: str
) -> None:
    """One job's switch, flipped off, with the job still running behind it.

    The failure this exists to catch is a turn that stops the pings by reaching for a bigger
    lever than the ask named — retiring the whole job, or muting notifications globally — both
    of which look like success from inside the turn and neither of which is what was asked."""
    cohort = await _drive(chat_eval, model, _NOTIFY_OFF)
    # STORE — the field the ask named moved, and nothing else did.
    cohort.claim(
        "state: the job's notifications are off",
        _the_switch_is(_FINDS.container, on=False),
        SpecCategory.STORE,
    )
    cohort.claim(_STILL_LIVE, _the_job_is_still_live(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        _NOTHING_ELSE_MOVED,
        _nothing_else_moved(_FINDS.container, asked=_NOTIFIES),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ flip the switch back ════════════════════════════════════════════════════
#
# The same switch from the other world: the job is running QUIET, and the user asks for its
# notifications back.  The wordings are the off case's turned round, so the pair differs in the
# entry condition and the direction asked for and in nothing else.

_NOTIFY_ON = _OperationCase(
    case_id="standing-notify-on",
    behaviour=(
        "In the chat agent, when the user asks for one running job's notifications to be "
        "turned back on, Penny turns that job's own switch on, with the job still running as "
        "it was and proactive notifications still on everywhere else."
    ),
    job=_QUIET_FINDS,
    renders=SelfStateHeader.MECHANISM_QUIET,
    ask=f"turn notifications back on for {_THEIR_WORDS}",
    also_phrased=(
        f"start the notifications on {_THEIR_WORDS} again",
        f"can you turn notifications back on for {_THEIR_WORDS}?",
        f"switch notifications back on for {_THEIR_WORDS} please",
        f"i want notifications from {_THEIR_WORDS} again",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_turning_notifications_back_on_wakes_only_that_job(
    chat_eval: ChatEval, model: str
) -> None:
    """One quiet job's switch, flipped on, with the job still running behind it.

    The job is seeded with its switch OFF and its row says so, so a sample that left the row
    alone fails the headline whatever its reply said.  The global switch is a separate lever
    one row up, already on in this world: it is claimed as what the turn found and left."""
    cohort = await _drive(chat_eval, model, _NOTIFY_ON)
    # STORE — the field the ask named moved, and nothing else did.
    cohort.claim(
        "state: the job's notifications are on",
        _the_switch_is(_QUIET_FINDS.container, on=True),
        SpecCategory.STORE,
    )
    cohort.claim(_STILL_LIVE, _the_job_is_still_live(_QUIET_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        _NOTHING_ELSE_MOVED,
        _nothing_else_moved(_QUIET_FINDS.container, asked=_NOTIFIES),
        SpecCategory.STORE,
    )
    cohort.claim(
        _KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_QUIET_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ retire it ═══════════════════════════════════════════════════════════════
#
# The user is done with the job.  Archived rather than deleted is the contract everywhere in
# the registry: a retired mechanism stays enumerable and the same job asked for again revives
# it, so a turn that DELETED the collection would have taken the user's entries with it.

_ARCHIVE = _OperationCase(
    case_id="standing-archive",
    behaviour=(
        "In the chat agent, when the user says they are done with a running job, Penny "
        "retires it as a tombstone that still holds everything it gathered, with proactive "
        "notifications still on everywhere else."
    ),
    job=_FINDS,
    renders=SelfStateHeader.MECHANISM_NOTIFIES,
    ask=f"i'm done with {_THEIR_WORDS} — you can retire that one",
    also_phrased=(
        f"i don't need {_THEIR_WORDS} any more, you can retire it",
        f"you can retire {_THEIR_WORDS} — i'm finished with it",
        f"shut {_THEIR_WORDS} down, i'm done with it",
        f"{_THEIR_WORDS} has run its course, please retire it",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_retiring_a_job_archives_it_and_keeps_what_it_gathered(
    chat_eval: ChatEval, model: str
) -> None:
    """The job is retired as a tombstone, with everything it collected still readable and
    nothing else about it disturbed.

    Retiring is REVERSIBLE — that is what archiving rather than deleting is for, and the same
    job asked for again revives this row — so what the turn leaves on it is what the revived
    job would run.  A retire that also rebound the routine, rewrote the program, re-timed the
    cadence or flipped the switch hands back a different job under the same name, which is why
    the drift claim below reads the whole row rather than the archive flag alone."""
    cohort = await _drive(chat_eval, model, _ARCHIVE)
    # STORE
    cohort.claim(
        "state: the job is retired (archived)",
        _the_job_is_retired(_FINDS.container),
        SpecCategory.STORE,
    )
    cohort.claim(
        _NOTHING_ELSE_MOVED,
        _nothing_else_moved(_FINDS.container, asked=_ARCHIVED),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ re-time it ══════════════════════════════════════════════════════════════
#
# The user changes WHEN a job runs.  Two hours are in play from the first word of the ask
# ("too early" is a claim about the hour it has now), and after the edit lands the row holds
# only the hour that won — so the prior is the one fact about this turn that nothing in the
# world still answers.  The observed regression is exactly that gap: a schedule-fix turn whose
# thinking read the real stored rule and whose fix was correct, while the reply stated a prior
# value contradicting both (#1946).

_RE_TIME = _OperationCase(
    case_id="standing-schedule-fix-prior",
    behaviour=(
        "In the chat agent, when the user says a running job checks at the wrong time and "
        "names a new one, Penny re-times that job with the rest of it as it was, and every "
        "clock time she names is one the job has actually had."
    ),
    job=_FINDS,
    renders=_FINDS.schedule,
    ask=f"{_THEIR_WORDS} is checking too early — move it to {_FIXED_HOUR} in the morning",
    also_phrased=(
        f"{_THEIR_WORDS} runs too early for me, shift it to {_FIXED_HOUR} in the morning",
        f"can you move {_THEIR_WORDS} to {_FIXED_HOUR} in the morning? it's too early as it is",
        f"{_THEIR_WORDS} goes off too early — make it {_FIXED_HOUR} in the morning instead",
        f"push {_THEIR_WORDS} back to {_FIXED_HOUR} in the morning, it's checking too early",
    ),
)

# The ask must lend the claims NOTHING about the hour the job has NOW: a reply naming it can
# only have read it off the state.  Enforced rather than trusted, on every wording, the way
# every other leak guard in this file is.
for _wording in (_RE_TIME.ask, *_RE_TIME.also_phrased):
    assert _SEEDED_HOUR not in _hours_named(_wording), (
        f"a re-timing wording must not name the hour the job already runs at: {_wording!r}"
    )


def _every_hour_it_names_is_one_the_job_has_had(sample: SampleObservation, _world: World) -> Answer:
    """Every clock time in the reply is one of the job's own two — the hour it had, or the
    hour it now has.

    Any third hour is INVENTED: the ask supplies only the new one and the seeded world only
    the old one, so containment in that pair is the structural form of "the stated prior
    matches what was actually there".  A violating sample is nameable, and is the measured
    one: a turn whose fix was correct and whose reply said the job used to run at some hour it
    never ran at.

    This is the case's provenance claim rather than the general reply-sourcing one, because a
    clock time has several correct renderings and the general claim compares digits: a reply
    stating the new hour as ``11:00`` writes digits the ask never wrote, and would be read as
    an invention.  The hours are read here in the unit the fact actually has, so how a reply
    spelled one is not part of the question."""
    named = _hours_named(sample.reply)
    theirs = {_SEEDED_HOUR, _FIXED_HOUR}
    return named <= theirs, f"named {sorted(named)}, the job has had {sorted(theirs)}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_re_timing_a_job_states_no_hour_it_never_ran_at(
    chat_eval: ChatEval, model: str
) -> None:
    """The job checks too early, so it is moved — and the reply must not invent the hour it
    was moved FROM.

    A reply that names no hour at all satisfies the containment claim, and that is correct
    rather than a free pass: it has claimed nothing about when the job used to run, so there
    is nothing in it to be wrong.  Whether she states the prior at all is a separate question
    the record frame asks for and this case deliberately does not demand — the harm is a
    stated prior that is false, not a confirmation that stays quiet about it."""
    cohort = await _drive(chat_eval, model, _RE_TIME)
    # STORE
    cohort.claim(
        "state: the job now runs at the hour they asked for",
        _the_rule_fires_at(_FINDS.container, _FIXED_HOUR),
        SpecCategory.STORE,
    )
    cohort.claim(_STILL_LIVE, _the_job_is_still_live(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        _NOTHING_ELSE_MOVED,
        _nothing_else_moved(_FINDS.container, asked=_SCHEDULE),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.claim(
        "reply: every clock time it names is one the job has had",
        _every_hour_it_names_is_one_the_job_has_had,
        SpecCategory.PROVENANCE,
        kind="reply",
    )

    cohort.measure(*_MEASURED)


# ═══ give it an end ══════════════════════════════════════════════════════════
#
# The job runs with no end, and the user says how long it should keep going.  The end is read
# the way every stored end is (``utils/job_end.py``): as when the job STOPS on the user's own
# clock, counted from the sample's pinned turn, whichever of the three ways a job stores an end
# the turn reached for — an expiry, an ``UNTIL=`` in the rule, or a ``COUNT=``.

_ASKED_END = UNTIL_SUNDAY_NIGHT

_END_CHANGED = _OperationCase(
    case_id="standing-end-changed",
    behaviour=(
        "In the chat agent, when the user says how long a running job should keep going, "
        "Penny gives that job the end they named, with the job running as it ran until then "
        "and proactive notifications still on everywhere else."
    ),
    job=_FINDS,
    renders=_FINDS.schedule,
    ask=f"keep {_THEIR_WORDS} going until sunday night",
    also_phrased=(
        f"can you keep {_THEIR_WORDS} running until sunday night?",
        f"i only need {_THEIR_WORDS} until sunday night — keep it going till then",
        f"let {_THEIR_WORDS} run until sunday night, then it can stop",
        f"{_THEIR_WORDS} should keep going until sunday night and stop after that",
    ),
)

# Every wording states the end in the words the claim's window is declared for — ENFORCED,
# since an arm that said "the weekend" would be read against a window it never named.
for _wording in (_END_CHANGED.ask, *_END_CHANGED.also_phrased):
    assert _ASKED_END.says in _wording, f"an end wording must say {_ASKED_END.says!r}: {_wording!r}"


# How many of the job's coming firings the rhythm claim reads.  Three is enough to tell a moved
# hour and a widened cadence apart from the rhythm the job had, and few enough that an end on
# the right night leaves all of them standing.
_FIRINGS_READ = 3

_FIRING_SPOKEN = "%a %H:%M"


def _coming_firings(
    schedule: str, row: MechanismRecord, timezone: str | None, after: datetime
) -> list[datetime] | None:
    """The next few moments a rule fires after ``after``, through production's own
    ``next_occurrence`` — anchored where the collector anchors it, on the user's clock.
    Fewer than asked for when the rule ends first; ``None`` for a rule that will not parse or
    a row with no creation moment to anchor it at, an honest reading the claim fails on."""
    if row.created_at is None:
        return None
    fired: list[datetime] = []
    moment = after
    try:
        for _ in range(_FIRINGS_READ):
            upcoming = next_occurrence(schedule, row.created_at, timezone, moment)
            if upcoming is None:
                break
            fired.append(upcoming)
            moment = upcoming
    except ValueError, TypeError:
        return None
    return fired


def _spoken_firings(fired: list[datetime] | None, timezone: str | None) -> str:
    """Firings as a rationale says them: the weekday and the hour, where the user is."""
    if fired is None:
        return "nothing readable"
    zone = zone_or_utc(timezone)
    return str([moment.astimezone(zone).strftime(_FIRING_SPOKEN) for moment in fired])


def _it_runs_when_it_ran(job: StandingJob) -> _ClaimFn:
    """Every time the job fires from now on is a time it would have fired anyway —
    PRESERVATION of when it runs, read off the rule's own coming firings.

    Its own claim because the ledger cannot make it here: a rule rewritten to carry its own
    end is one of the ways a job stores an end, so the schedule field is one the ask may move,
    and "the rule's text changed" no longer says whether its rhythm did.  The job's coming
    firings are compared with the ones the rule it was seeded with makes from the same row,
    so both sides share one anchor and one clock; a job that now stops sooner makes fewer of
    them and is still on its rhythm, since where it stops is the end claim's to answer.  A
    violating sample is nameable: one that gives the job its end and, restating the rule to do
    it, moves the hour or turns a daily check into a weekly one."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, job.container)
        if row is None or row.schedule is None or sample.turn_at is None:
            return False, f"{job.container!r} has no schedule, or the sample no turn, to read"
        ran = _coming_firings(job.schedule, row, sample.timezone, sample.turn_at)
        runs = _coming_firings(row.schedule, row, sample.timezone, sample.turn_at)
        holds = ran is not None and runs is not None and runs == ran[: len(runs)]
        return holds, (
            f"the rule {row.schedule!r} next fires {_spoken_firings(runs, sample.timezone)}, "
            f"it was due {_spoken_firings(ran, sample.timezone)}"
        )

    return answer


# The store's labels for the fields an end may be written to: its expiry, and the rule itself
# when the end is stated inside it.
_END_FIELDS = frozenset({"expires_at", "schedule"})


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_giving_a_job_an_end_stops_it_when_the_ask_said(
    chat_eval: ChatEval, model: str
) -> None:
    """The job keeps running, and now stops when the user said it should.

    The job is seeded with no end at all, so a sample that left the row alone fails the
    headline.  What the end claim reads is when the job stops, never which field holds it; the
    cadence claim beside it is what keeps "restated the rule" from being a way to re-time it."""
    cohort = await _drive(chat_eval, model, _END_CHANGED)
    # STORE — the end the ask named, and the rest of the job as it was.
    cohort.assert_the_job_ends_when_asked(_FINDS.container, _ASKED_END)
    cohort.claim(_STILL_LIVE, _the_job_is_still_live(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        "state: every time it fires from now on is a time it was already due",
        _it_runs_when_it_ran(_FINDS),
        SpecCategory.STORE,
    )
    cohort.claim(
        _NOTHING_ELSE_MOVED,
        _nothing_else_moved(_FINDS.container, asked=_END_FIELDS),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ read it back ═══════════════════════════════════════════════════════════
#
# "What does that thing actually do?" — the same job, asked about rather than operated on.  The
# ambient header carries each routine as one row (its name, what it is for, what it needs) and
# not its steps (#1804), so a walk-through is answered from the job's RECORD — its rendered
# program, or the routine's own steps.  Which read she reaches for is a route, measured in the
# tool sequence; how she words the walk-through is the reply's spread.

# The page the routine fetches, as the SMALLEST datum a faithful walk-through carries.  Every
# render of this job's record spells the shop this way — the address its program fetches, the
# container name derived from that address, the description it was stood up with — and no
# wording of the ask does, so a reply naming it cites the record.  Case-folded on both sides,
# so "Quill Market" and ``quillmarket.example.com`` are the same fact.
_ROUTINE_PAGE_TOKEN = "quill"

_DESCRIBE = _OperationCase(
    case_id="standing-describe-routine",
    behaviour=(
        "In the chat agent, when the user asks what a standing job does, Penny describes the "
        "job from its record, and every value in her reply traces to that record."
    ),
    job=_FINDS,
    renders=SelfStateHeader.MECHANISM_NOTIFIES,
    ask=f"what does {_THEIR_WORDS} actually do? walk me through it.",
    also_phrased=(
        f"walk me through what {_THEIR_WORDS} does, step by step",
        f"how does {_THEIR_WORDS} work? what does it actually do each time it runs?",
        f"can you explain what {_THEIR_WORDS} does when it runs?",
        f"what exactly happens when {_THEIR_WORDS} runs? talk me through it",
    ),
    answers=(_ROUTINE_PAGE_TOKEN,),
)

# The answer token is on the job's record and in none of the wordings — ENFORCED rather than
# trusted, because the leak is invisible once it exists: a question carrying the token would
# credit a reply for repeating the question back.
assert _ROUTINE_PAGE_TOKEN in _FINDS.program, "the answer token must be on the job's own program"
for _wording in (_DESCRIBE.ask, *_DESCRIBE.also_phrased):
    assert _ROUTINE_PAGE_TOKEN not in _wording.casefold(), (
        f"a read-back wording must not carry the answer token: {_wording!r}"
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_she_describes_the_routine_the_job_runs(chat_eval: ChatEval, model: str) -> None:
    """The job asked about rather than operated on: the reply names the page its routine
    fetches, every specific it states is one the round was given, and the job is exactly as
    the turn found it.

    What is NOT claimed is how the walk-through is worded — a reply describing the two moves in
    any words is as faithful as another, so the phrasing is the reply's spread and never a
    pattern the reply must match (``docs/principles.md`` §4.7).  An invented step that names a
    specific — another site, an hour the job does not run at — is what the sourcing claim reads."""
    cohort = await _drive(chat_eval, model, _DESCRIBE, family=_LEGIBILITY_FAMILY)
    # STORE — the page the record carries, stated; and the job, as the turn found it.
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        _NOTHING_ELSE_MOVED,
        _nothing_else_moved(_FINDS.container, asked=_NONE),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ list what is running ════════════════════════════════════════════════════
#
# "What are you watching for me?" — every running job, asked about at once.  Each one renders
# ambiently as a row (its name, its cadence, its switch) and again in the store map with what
# it is for, so the answer is in front of the turn; which of those she reads it off, and
# whether she reaches for a read as well, are routes.

# What each job looks for, as the SMALLEST datum a faithful listing carries.  Every render of
# a job's record spells its own — the address its program fetches, the value it is bound to
# look for, the name derived from both, the description it was stood up with — so a reply that
# names the job at all, by any of them, carries it; and no other job's record does, so a reply
# carrying all three has named all three.  Case-folded on both sides.
_LISTED = ((_FINDS, "typewriter"), (_KETTLES, "kettle"), (_ATLASES, "atlas"))
_LISTED_JOBS = tuple(job for job, _ in _LISTED)

_LIST_RUNNING = _OperationCase(
    case_id="standing-list-running",
    behaviour=(
        "In the chat agent, when the user asks what Penny is watching for them, Penny names "
        "every running job from its record, with every job, what each gathered and "
        "proactive notifications as they were."
    ),
    job=_FINDS,
    others=(_KETTLES, _ATLASES),
    renders=SelfStateHeader.MECHANISM_NOTIFIES,
    ask="what are you watching for me right now?",
    also_phrased=(
        "which jobs do you have running for me at the moment?",
        "can you list everything you're currently keeping an eye on for me?",
        "remind me what you're watching for me these days",
        "what do you have running in the background for me right now?",
    ),
    answers=tuple(token for _, token in _LISTED),
)

# Each token is on its own job's record — its page, what it looks for, its description — and
# on no other job's, and in none of the wordings.  ENFORCED, because either leak is invisible
# once it exists: a token two jobs share credits a reply for naming one of them twice, and a
# question carrying one credits a reply for repeating the question back.
for _job, _token in _LISTED:
    _record = f"{_job.program} {_job.description} {_job.container} {_job.holdings}".casefold()
    assert all(
        _token in text.casefold()
        for text in (_job.url, _job.watched_for, _job.description, _job.container)
    ), f"{_token!r} must be on every part of its own job's record"
    for _other, _other_token in _LISTED:
        assert _other is _job or _other_token not in _record, (
            f"{_other_token!r} must not be on {_job.container}'s record"
        )
    for _wording in (_LIST_RUNNING.ask, *_LIST_RUNNING.also_phrased):
        assert _token not in _wording.casefold(), (
            f"a listing wording must not carry an answer token: {_wording!r}"
        )


def _every_job_is_as_it_was(sample: SampleObservation, _world: World) -> Answer:
    """Every running job's row holds what it held before the turn — PRESERVATION of the jobs
    the ask was only about.

    Read as each row's END STATE (``moved_this_run``), so a turn that restated a value has
    moved nothing.  A violating sample is nameable: one that tidies while it lists — retiring
    a job it judged stale, or switching one's notifications to match the others."""
    drifted: list[str] = []
    for job in _LISTED_JOBS:
        row = _job_row(sample, job.container)
        if row is None or row.moved_this_run:
            drifted.append(f"{job.container}: {row.moved_this_run if row else 'gone'}")
    return not drifted, f"moved {drifted}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_she_lists_every_job_she_is_running(chat_eval: ChatEval, model: str) -> None:
    """Three jobs running, and a question about all of them: the reply names each one, every
    specific it states is one the round was given, and the world is exactly as the turn found
    it.

    What is NOT claimed is how the listing is worded or how much of each job it gives — a
    cadence, a switch, a page — since the ask names none of them; a specific the reply does
    state and the record does not is what the sourcing claim reads."""
    cohort = await _drive(chat_eval, model, _LIST_RUNNING, family=_LEGIBILITY_FAMILY)
    # STORE — each job named in the reply; and every job, what each gathered and the global
    # switch, as the turn found them.
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: every running job is as it was", _every_job_is_as_it_was, SpecCategory.STORE
    )
    cohort.assert_what_the_store_held_survives()
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ an ask that matches two of the same kind ════════════════════════════════
#
# Two jobs of the SAME kind, each watching its own page for a price drop — the #2365
# shape: "turn off notifications for the price watch" when two price watches are running,
# and the ask names only the KIND, never which one.  Nothing else in the conversation picks
# between them either, so the entry condition is genuinely ambiguous rather than ambiguous
# in appearance with a tiebreaker sitting in the message.
#
# The ruled contract (code owner, filed on #2365): neither job changes without the user
# saying which.  That is a STORE claim about both rows, and a COMPLETENESS claim that the
# reply names both jobs — the same ``World.answers`` containment ``standing-list-running``
# reads, pointed at two rows instead of three.  Whether she also ASKS which one, in as many
# words, is NOT claimed: that is the route a correct turn could take, not the only one a
# correct turn must take, so it stays measured in the tool sequence and the reply's own
# spread (#2373's "facts asserted, behaviour measured" rule).

_PRICE_WATCH_A = StandingJob(
    container="camera-kit-price-watch",
    url="https://foxden.example.com/camera-kit",
    watched_for="price drops on the camera kit",
    description="Price drops on the camera kit at Foxden.",
    schedule="FREQ=DAILY;BYHOUR=8",
    notify=True,
    holdings=(("Foxden camera kit", "Foxden camera kit — now $340, down from $380."),),
)
_PRICE_WATCH_B = StandingJob(
    container="turntable-price-watch",
    url="https://cedarlane.example.com/turntable",
    watched_for="price drops on the turntable",
    description="Price drops on the turntable at Cedarlane.",
    schedule="FREQ=DAILY;BYHOUR=18",
    notify=True,
    holdings=(("Cedarlane turntable", "Cedarlane turntable — now $210, down from $250."),),
)

# The two tokens the reply has to carry to have named BOTH jobs — each on its own job's
# values and description, on neither other job's, and in NEITHER wording: the ask names only
# the kind ("the price watch"), never which shop or item, or the model would be reading the
# answer off the question instead of off the state.
_AMBIGUOUS_MATCH_TOKENS = (
    (_PRICE_WATCH_A, "camera kit"),
    (_PRICE_WATCH_B, "turntable"),
)

for _job, _token in _AMBIGUOUS_MATCH_TOKENS:
    assert _token in f"{_job.url} {_job.watched_for} {_job.description}".casefold(), (
        f"{_token!r} must be on {_job.container}'s own record"
    )
for _other, _other_token in _AMBIGUOUS_MATCH_TOKENS:
    for _job, _token in _AMBIGUOUS_MATCH_TOKENS:
        if _other is not _job:
            assert (
                _other_token not in f"{_job.url} {_job.watched_for} {_job.description}".casefold()
            ), f"{_other_token!r} must not be on {_job.container}'s record"

_AMBIGUOUS_MATCH = _OperationCase(
    case_id="standing-ambiguous-job-match",
    behaviour=(
        "In the chat agent, when an ask names a running job by a description two running "
        "jobs of the same kind both match, and nothing else in the conversation picks "
        "between them, Penny changes neither job and names both in her reply."
    ),
    job=_PRICE_WATCH_A,
    others=(_PRICE_WATCH_B,),
    renders=SelfStateHeader.MECHANISM_NOTIFIES,
    ask=(
        "hey, can you turn off notifications for the price watch? it's been pinging me "
        "nonstop today and I just want it to go quiet for now, I'll check back on it myself"
    ),
    also_phrased=(
        "can you mute the price watch for me? it keeps buzzing my phone and I'd rather "
        "look at it on my own time instead of getting pinged every time something moves",
        "do me a favor and switch off notifications on the price tracker — it's been "
        "going off a lot today and I don't need the alerts right now",
        "could you turn the price alert notifications off? i'm getting pinged way too "
        "much and want to just check it myself when i feel like it",
        "quick one — can you stop the price watch from notifying me? it's fine running "
        "in the background, i just don't want the pings",
    ),
    answers=tuple(token for _, token in _AMBIGUOUS_MATCH_TOKENS),
)

# No wording leaks either job's answer token — ENFORCED, the way every other leak guard in
# this file is, because the leak is invisible once it exists: a wording naming "camera kit"
# or "turntable" would hand the model half the disambiguation the case exists to deny it.
for _wording in (_AMBIGUOUS_MATCH.ask, *_AMBIGUOUS_MATCH.also_phrased):
    for _, _token in _AMBIGUOUS_MATCH_TOKENS:
        assert _token not in _wording.casefold(), (
            f"an ambiguous-match wording must not carry an answer token: {_wording!r}"
        )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_ask_matching_two_jobs_of_the_same_kind_changes_neither(
    chat_eval: ChatEval, model: str
) -> None:
    """Two price watches running, and an ask that names only the kind — neither job's row may
    move, and the reply has to name both, whether she does that by asking which one or by
    describing the two and leaving the choice open.

    The failure this exists to catch is a SILENT pick: a turn that reads "the price watch" as
    close enough to one of the two rows and flips its switch alone, which looks like success
    from inside the turn and leaves the user's actual job untouched and unmentioned."""
    cohort = await _drive(chat_eval, model, _AMBIGUOUS_MATCH)
    # STORE — neither row moved, at all; the reply names both.
    cohort.claim(
        "state: the camera-kit price watch is unchanged",
        _nothing_else_moved(_PRICE_WATCH_A.container, asked=_NONE),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the turntable price watch is unchanged",
        _nothing_else_moved(_PRICE_WATCH_B.container, asked=_NONE),
        SpecCategory.STORE,
    )
    cohort.claim(_STILL_ON_EVERYWHERE, _notifications_are_still_on_everywhere, SpecCategory.STORE)
    cohort.assert_the_reply_answers_the_ask()

    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# Every case, in one place — so the deterministic probe in ``test_eval_harness.py`` can drive
# each one's seeder and premise without a GPU.
OPERATION_CASES = (
    _NOTIFY_OFF,
    _NOTIFY_ON,
    _ARCHIVE,
    _RE_TIME,
    _END_CHANGED,
    _DESCRIBE,
    _LIST_RUNNING,
    _AMBIGUOUS_MATCH,
)
