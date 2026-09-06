"""Operating a job that is already running: three cases (#2008, tranche 3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**One sentence, THREE cases.**  #2008 states the behaviour as *Penny flips a running job's
notification switch, retires it, or re-times it without disturbing the job itself — and
without inventing the hour it used to run at*, and that sentence names three ACTIONS rather
than three wordings of one.  The design's test decides it: a correct sample for *retire*
archives the job, which is exactly what a correct sample for *flip the switch* must not do —
so a sample that is right for one is wrong for another, and they are three cases with three
sentences.  The count came out at three rather than the ticket's one, and the reason is
recorded here.

| action | case | what the row must show |
|---|---|---|
| flip the switch | ``standing-notify-off`` | ``notify`` off, everything else on the row as it was |
| retire it | ``standing-archive`` | ``archived``, and what it gathered still readable |
| re-time it | ``standing-schedule-fix-prior`` | the rule fires at the hour they asked for |

**Which survivor the flip kept, and why.**  ``standing-notify-off`` and ``standing-notify-on``
are the same sentence in two entry conditions, so one of them survives — and the OFF direction
strictly dominates, because it is the only one whose world can produce the failure the
behaviour is about.  Asked to stop one watch's pings, a measured sample enumerated what it
could see — archive the collection, or mute notifications globally — read that as no
granularity, and retired the whole job.  Neither wrong move is available in the other
direction: a job cannot be woken by archiving it or by lifting a global mute.  So
``standing-notify-on`` is QUARANTINED here, not deleted, and comes back the day the ON
direction's own residual is the thing being measured (#1927 records it at 3 of 5 against the
OFF direction's 5 of 5, and the leak is the classifier's rather than this turn's).

**Every case claims the landing, and that is deliberate.**  Changing how a running job behaves
is idle by the machine's own boundary (#1927) — the state definitions say so in as many words —
so ``idle`` is the contract rather than a hope, and a sample that lands in ``request`` or
``apply`` is the residual that ticket records, showing up as a number instead of as a note.

**What is NOT claimed, and why.**  That she called ``collection_set`` is a ROUTE — many routes
reach one end state, and a rule keyed to that name would simply not fire for a verb nobody
enumerated — so it is measured in the tool sequence and never asserted.  How much of that the
tool sequence can SEE differs by case, and the report says which: the chat observation narrows a
sample's calls to ``ENACTING_TOOLS``, which carries ``collection_set`` and not
``collection_archive``, so the two cases that reconfigure a job read a real spread while the
retire case reads BLIND, in red.  Reported as a harness gap rather than worked around here.  On
the retire case the
switch and the cadence are not claimed unchanged either: an archived job is out of the
dispatcher's reach whatever its switch says, so a turn that also silenced it did nothing the
user can observe, and claiming it would fail a correct run over a difference with no
consequence.

**The world is built the way production builds one (#1911/migration 0108: nothing is
pre-seeded).**  Every collection here is one the user built: the container is created
storage-only, then configured through the store's own metadata write with the program rendered
through the production instantiation seam's three steps — the attachment bound to the
container, the runtime join (#1907) writing each bound value into the leaf the demonstration
put it in, then the render.  The routine itself goes into the registry through
``chat_eval(seed_skills=…)``, embedded like a real one.  A hand-authored prose prompt would be
a config defect the collector cannot read (#1916's strict dialect), so a world seeded that way
would be claiming a job that could never run.

``standing-describe-routine`` at the bottom is LEFT EXACTLY AS IT WAS.  Reading a job's routine
back is none of #2008's twelve behaviours and sits on no slot of #2004's map, so it is neither
ported nor quarantined here — it keeps its own scorer until it has a slot of its own.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every page, shop and job is synthetic, on an
``example`` domain, because the repo is public.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, NamedTuple

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillStep,
    bind_parameters,
    derive_collection_name,
    distill_steps,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.penny import Penny
from penny.program import program_calls
from penny.skill_extraction import _apply_leaf_labels, _interface_parameters
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    DESCRIBES_FETCH,
    DESCRIBES_SAVE,
    EVAL_MODELS,
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Preparer,
    Seeder,
    describes,
    seeded_run_id,
    tool_was_called,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    MechanismRecord,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.collection_instantiation import next_occurrence
from penny.tools.micro_context import FramedParameter, LeafLabel, SkillLabels, SkillSignature

pytestmark = pytest.mark.eval

_OPERATIONS_FAMILY = "standing-collection-operations"
_LEGIBILITY_FAMILY = "standing-collection-legibility"

# The run that stood the jobs up — a seeded prior turn, so every reader of "what did the
# model do this sample" excludes it (a job's own creation is history, not this turn's work).
_STOOD_UP_RUN = seeded_run_id("stood-up-the-job")

# The two calls the taught routine makes, in order.  Named once: the seeder holds the
# rendered program against them in the STRICT dialect (#1916), so a fixture that stops
# rendering a runnable program fails at the seed rather than as a puzzling miss.
_BROWSE = "browse"
_WRITE = "collection_write"
_PROGRAM_TOOLS = (_BROWSE, _WRITE)

# The reads that answer "what does this thing do".  ``memory_metadata`` renders the
# collection's own rendered program; ``skill_read`` returns the routine's steps.  Both are
# a READ of the recipe — which is the contract — so the spine check accepts either rather
# than keying on the verb that happened to be used when the case was written.
_METADATA = "memory_metadata"
_SKILL_READ = "skill_read"


# ── The taught routine ────────────────────────────────────────────────────────


class _Demonstration(NamedTuple):
    """The round that taught the routine, as its ledger recorded it: the page it read,
    what it was told to pull off it, the container it wrote into, and the entry that
    landed.  Two steps — read the page, write what it said — the canonical taught round."""

    url: str
    extract: str
    collection: str
    entry_key: str
    entry_value: str


def _demonstrated_ledger(demonstration: _Demonstration) -> list[DistillInput]:
    """The two certified steps distillation reads, with their results carrying the real
    ``(<tool> result)`` frame — so the frame is stripped the way production strips it and
    the write's content binds to the browse's PAYLOAD rather than to its narration."""
    return [
        DistillInput(
            source_ordinal=1,
            tool=_BROWSE,
            arguments={"queries": [demonstration.url], "extract": demonstration.extract},
            result=(f"You opened {demonstration.url} (browse result)\n{demonstration.entry_value}"),
        ),
        DistillInput(
            source_ordinal=2,
            tool=_WRITE,
            arguments={
                "memory": demonstration.collection,
                "entries": [{"key": demonstration.entry_key, "content": demonstration.entry_value}],
            },
            result=(
                f"You saved an entry to {demonstration.collection}: "
                f"({_WRITE} result)\nWrote 1 entry."
            ),
        ),
    ]


def _leaf_at(arguments: dict[str, Any], path: list[str | int]) -> Any:
    """The demonstrated value a substitution addresses — indexed directly, since a path a
    step's own substitution carries either resolves or the step is corrupt."""
    node: Any = arguments
    for part in path:
        node = node[part]
    return node


def _labels_by_value(steps: list[SkillStep], authored: dict[str, LeafLabel]) -> SkillLabels:
    """The labeller's draw, authored against the demonstrated VALUES and mapped onto the
    arg-derived spot names the distiller happened to mint.

    Keyed by value because that is what the fixture MEANS — a naming change inside
    distillation would otherwise leave this silently unmapped.  Every authored label must
    map home: an accepted draw covers every spot (#1828), so a partly-labelled routine is
    a shape run-end extraction cannot produce, and it fails here rather than seeding a
    world nothing makes."""
    labels: dict[str, LeafLabel] = {}
    for step in steps:
        for substitution in step.substitutions:
            if substitution.parameter is None:
                continue
            value = str(_leaf_at(step.arguments, substitution.path))
            if value in authored:
                labels[substitution.parameter] = authored[value]
    assert len(labels) == len(authored), (
        f"every authored label must map home — matched {sorted(labels)} of {sorted(authored)}"
    )
    return SkillLabels(labels=labels)


def _taught_routine(
    demonstration: _Demonstration, signature: SkillSignature, authored: dict[str, LeafLabel]
) -> SkillDraft:
    """The skill a demonstrated round leaves in the registry, built by the production
    pipeline over that round's ledger: ``distill_steps`` for the structure, the labeller's
    draw applied by ``_apply_leaf_labels``, the framer's signature applied by
    ``_interface_parameters``.

    Only the two DRAWS are written by hand, which is what a fixture is for — so the
    starting world is the shape extraction really produces (an all-placeholder recipe over
    the SKILL-level parameters the framer minted, each carrying the value its round
    demonstrated it with) rather than a convenient copy of it."""
    steps, parameters = distill_steps(
        _demonstrated_ledger(demonstration), frozenset({demonstration.collection})
    )
    steps, distilled = _apply_leaf_labels(steps, parameters, _labels_by_value(steps, authored))
    return SkillDraft(
        name=signature.name,
        intent=signature.description,
        description=signature.description,
        steps=steps,
        parameters=_interface_parameters(signature, distilled),
        source_run_id=seeded_run_id("taught-the-routine"),
    )


# The one routine every case stands on, and the round that taught it.  Synthetic
# throughout (the repo is public): an invented shop on an example domain, an invented
# thing to watch for.
_DEMONSTRATED_PAGE = "https://quillmarket.example.com/typewriters"
_DEMONSTRATED_WATCH = "portable typewriters"
_DEMONSTRATED_ENTRY = "Hermes Baby — 1958 portable, case included, $180."

WATCH_ROUTINE = _taught_routine(
    _Demonstration(
        url=_DEMONSTRATED_PAGE,
        extract=_DEMONSTRATED_WATCH,
        collection="typewriter-finds",
        entry_key="Hermes Baby",
        entry_value=_DEMONSTRATED_ENTRY,
    ),
    SkillSignature(
        name="track_new_listings",
        description="Check a page for newly listed items of a given kind and keep what it finds.",
        parameters=(
            FramedParameter(
                name="page",
                description="the address of the page to check each run",
                value=_DEMONSTRATED_PAGE,
            ),
            FramedParameter(
                name="watched_for",
                description="the kind of thing to look for on that page",
                value=_DEMONSTRATED_WATCH,
            ),
        ),
    ),
    {
        _DEMONSTRATED_PAGE: LeafLabel(
            name="page", description="the address of the page to check each run"
        ),
        _DEMONSTRATED_WATCH: LeafLabel(
            name="watched_for", description="the kind of thing to look for on that page"
        ),
        # The destination is a spot like any other, and additionally carries the attachment
        # mark — applying the routine somewhere is what fills it (#1783/#1828).
        "typewriter-finds": LeafLabel(
            name="destination", description="the collection this is set up on"
        ),
        # The entry's key and content are deliberately NOT here: both came out of the
        # browse, so distillation BINDS them to that step (#1659's structural provenance)
        # and no spot is left for a label to name.  Listing one would be a fixture claiming
        # a draw the labeller was never offered — which is what the map-home assertion in
        # `_labels_by_value` catches.
    },
)


# ── A standing job: the routine, applied ──────────────────────────────────────


class StandingJob(NamedTuple):
    """One collection as an apply turn leaves it: the routine it runs, the values it is
    pointed at, and the TERMS the turn set (#1869).

    ``holdings`` is what the job has already gathered — a standing job that has never run
    is not the world these stories are about, and it is what the archive case reads to say
    a retired job KEPT what it collected."""

    routine: SkillDraft
    values: dict[str, str]
    description: str
    schedule: str
    notify: bool
    holdings: tuple[tuple[str, str], ...] = ()

    @property
    def ordered_values(self) -> list[str]:
        """The bound values in the routine's DECLARED order — what the container's name is
        derived from, and not an order a fixture is free to choose."""
        return [self.values[parameter.name] for parameter in self.routine.parameters]

    @property
    def container(self) -> str:
        """The collection the job runs in, through the SHIPPED derivation — spelling the
        name out would be a second copy of the scheme jobs are identified by."""
        return derive_collection_name(slug_skill_name(self.routine.name), self.ordered_values)

    @property
    def program(self) -> str:
        """The program the apply turn stores, through the production instantiation seam's
        own three steps in its own order: the attachment bound to the container, the
        runtime join (#1907) writing each bound value into the leaf the demonstration put
        it in, then the render.

        Composed here rather than called, because the shipped seam takes the registry ROW
        and the runner lays the registry down after the seed runs — so what a fixture must
        not do is invent a fourth step or reorder these three."""
        attached = retarget_writes(self.routine.steps, self.container)
        joined = bind_parameters(attached, self.routine.parameters, self.values)
        return render_skill(joined, self.values)


def seed_standing_jobs(*jobs: StandingJob) -> Seeder:
    """Stand each job up the way the production path does: the container CREATED storage
    only (the round's find-or-create), then configured — the rendered program, the RRULE
    schedule, notify, and the routine plus its bound values stamped as provenance — then
    whatever it has already gathered.

    Both writes go through the real store methods, so the mutation ledger records them
    citing the seeded run: a later "who changed this, and when" read finds a job with a
    history rather than a row that appeared from nowhere."""

    def seed(db: Database) -> None:
        for job in jobs:
            _stand_up(db, job)

    return seed


def _stand_up(db: Database, job: StandingJob) -> None:
    # The creating run is STAMPED, and it is load-bearing rather than tidy: the store records a
    # collection's birth as a mutation event citing whatever run created it, and an unstamped
    # one cites nothing — which every reader of "did THIS turn touch this row" then counts as
    # this turn's work, because a seeded run is recognised by its id and a missing id is not
    # one.  Measured: the two jobs this world lends the log-read cases made their
    # "no mechanism was created or changed" claim read 0 of 15 for a reason that had nothing to
    # do with the turn.
    db.memories.create_collection(job.container, job.description, created_by_run_id=_STOOD_UP_RUN)
    db.memories.update_collection_metadata(
        job.container,
        extraction_prompt=job.program,
        schedule=job.schedule,
        replace_schedule=True,
        notify=job.notify,
        skill_name=slug_skill_name(job.routine.name),
        skill_params=job.values,
        run_id=_STOOD_UP_RUN,
    )
    _assert_the_job_can_run(job)
    if job.holdings:
        require_memory(db, job.container).write(
            [EntryInput(key=key, content=content) for key, content in job.holdings],
            author="collector",
            run_id=_STOOD_UP_RUN,
        )


def _assert_the_job_can_run(job: StandingJob) -> None:
    """The stored program really is a program, and it is pointed at something.

    Two ways a seeded job can be a job in name only, and both are silent: its steps stop
    parsing in the STRICT rendered dialect (#1916), which leaves a cycle with a surface of
    the terminator alone and a run record naming a config defect; or the RUNTIME JOIN
    (#1907) stops filling its leaves, which leaves a program still describing what belongs
    in each spot instead of naming the page it fetches.  Either way every case standing on
    this world would be measuring nothing, so both fail here, out loud."""
    program = job.program
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_PROGRAM_TOOLS)))
    assert parsed == _PROGRAM_TOOLS, (
        f"{job.container}: the stored program must parse as {_PROGRAM_TOOLS}, got {parsed}\n"
        f"{program}"
    )
    unjoined = [value for value in job.values.values() if value not in program]
    assert not unjoined, (
        f"{job.container}: the runtime join must write every bound value into the program — "
        f"{unjoined} missing from\n{program}"
    )


# The standing job every case operates on: the routine above, pointed at the page it was
# taught on, running each morning and telling the user when it finds something.
_FINDS = StandingJob(
    routine=WATCH_ROUTINE,
    values={"page": _DEMONSTRATED_PAGE, "watched_for": _DEMONSTRATED_WATCH},
    description="Portable typewriters newly listed at the Quill Market shop.",
    schedule="FREQ=DAILY;BYHOUR=7",
    notify=True,
    holdings=(
        ("Hermes Baby", _DEMONSTRATED_ENTRY),
        ("Olivetti Lettera 32", "Olivetti Lettera 32 — 1963 portable, case included, $220."),
    ),
)

# Held at IMPORT, not only at seed time: `make check` collects this module without running
# a case, so a fixture that stopped rendering a runnable program is caught by the plain
# suite instead of surviving until someone spends GPU on it.
_assert_the_job_can_run(_FINDS)

# What the job's own row says before any turn runs — read ONCE and reused by the premise
# probe and by every "nothing else moved" claim, so a fixture edit cannot leave a claim
# quietly asserting the shape the job used to have.
_SEEDED_ROUTINE = slug_skill_name(_FINDS.routine.name)
_SEEDED_PROGRAM = _FINDS.program
_HELD_KEYS = tuple(key for key, _ in _FINDS.holdings)

# How the user refers to it: their own words for the job, never its derived name.  The
# derived name renders on the ambient mechanisms line, so resolving those words to that
# collection is a read the turn is expected to make — asking with the container's own
# name would hand the model the answer to half the case.
#
# It also lends NO word to the legibility scorer below, which credits a reply for
# describing the routine's two moves: an ask carrying one of those words would let a reply
# that merely echoes the question score as a faithful description.
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

    ``job`` is the world's one standing job, seeded through the production instantiation
    seam.  ``renders`` is the clause its self-state row must carry before the turn runs,
    asserted rather than hoped for, since the model can only reach for a lever the state
    presents: the switch for the two cases about stopping something — where its presence is
    also what makes the global mute a WRONG lever rather than the only one — and the stored
    rule for the case about moving it.
    """

    case_id: str
    behaviour: str
    job: StandingJob
    renders: str
    ask: str
    also_phrased: tuple[str, ...]

    @property
    def ground(self) -> World:
        """The world every arm of this case is answered against.

        Every field is EMPTY, and each is a report rather than an omission.  ``pages`` is
        empty because these asks are about a job's configuration and no page answers one —
        a page set here would hand a sample that went browsing something to talk about
        instead of letting the wrong turn read as one.  ``keeps`` states what a round must
        have written down and none of these turns is asked to write anything;
        ``excludes`` states what it was told to leave out and none of them excludes
        anything in as many words.  ``answers`` states what the reply owes, and each of
        these asks is an INSTRUCTION — "done, that one's quiet now" is a complete answer —
        so requiring a token would fail a correct run for something nobody requested."""
        return World(name=self.case_id, pages=(), keeps=(), excludes=())


def _job_row(sample: SampleObservation, container: str) -> MechanismRecord | None:
    """The standing job's own registry row as the sample left it, or ``None`` if the turn
    took it out of the registry altogether."""
    return next((one for one in sample.mechanisms if one.name == container), None)


def assert_the_operation_world(db: Database, case: _OperationCase) -> None:
    """The case's premise, asserted out loud before the turn runs.

    Three things, and every one of them is the precondition of a claim below: the lever the
    ask is about RENDERS on the job's own self-state row (with nothing there, a measured
    sample asked to stop one watch's pings read the state as offering no granularity and
    retired the whole job); the job already holds what the retire case says it keeps; and
    proactive notifications are ON, so "she did not silence everything instead" is a claim
    about this turn rather than about the world it started in.

    Takes the DATABASE rather than the running Penny so the same assertions run without a
    model: a premise that quietly stopped holding turns a scored claim into a claim about the
    fixture, and the cheapest place to catch that is ``make check``."""
    _assert_the_row_renders(db, case)
    keys = sorted(entry.key or "" for entry in require_memory(db, case.job.container).read_all())
    assert keys == sorted(_HELD_KEYS), (
        f"{case.case_id}: the job must already hold {sorted(_HELD_KEYS)}, it holds {keys}"
    )
    assert not db.users.is_muted(TEST_SENDER), (
        f"{case.case_id}: notifications must start ON, or the claim that this turn did not "
        "silence everything is answered by the seed"
    )


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
    row = next(
        (line for line in rendered.splitlines() if line.startswith(f"- {case.job.container} ")),
        None,
    )
    assert row is not None, f"{case.job.container} must render as a mechanism:\n{rendered}"
    assert case.renders in row, (
        f"{case.job.container} must render {case.renders!r} — it renders {row!r}"
    )


async def _drive(chat_eval: ChatEval, model: str, case: _OperationCase) -> Cohort:
    """Drive one operation case: the standing job its own apply turn left behind, the
    routine it runs in the registry, and the premise re-asserted before the turn."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        model=model,
        seed=seed_standing_jobs(case.job),
        seed_skills=[WATCH_ROUTINE],
        prepare=_premise(case),
        world=case.ground,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_OPERATIONS_FAMILY,
        timeout=240.0,
    )


# What every case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are absent: they read the
# routines in the registry, and none of these turns teaches one — so on a correct cohort they
# read the world's own seeded routine on every sample and pool to a serene 0.000 that is
# neither agreement nor blindness but a reading of the FIXTURE.  A sample that DID mint one
# left idle, which the landing claim reads.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ── The claims, as pure functions over one sample ─────────────────────────────
#
# Every one is a read of what the turn LEFT BEHIND — the job's own row, what it holds, the
# registry around it, the global switch.  None reads a tool NAME: a job cannot change its own
# configuration by accident, so the row already says whether the edit happened, and a rule
# keyed to ``collection_set`` would simply not fire for a plugin verb nobody enumerated.
#
# They stay LOCAL rather than graduating into ``assertions.py``.  A claim graduates at the
# second CUSTOMER, and the three cases below are one behaviour family in one file: a second
# FILE asking one of these questions is what would make it shared, and none has yet.

_ClaimFn = Callable[[SampleObservation, World], Answer]


def _the_job_is_still_registered(container: str) -> _ClaimFn:
    """The job's row is still there.

    A violating sample is nameable: one that DELETES the collection to stop it — which takes
    the entries the user can still read with it — rather than archiving it as a tombstone."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = sorted(one.name for one in sample.mechanisms)
        return _job_row(sample, container) is not None, f"the registry holds {held}"

    return answer


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


def _only_the_named_field_moved(container: str, *, moved: str) -> _ClaimFn:
    """Nothing else on the job's own row moved.

    The claim the mutation ledger cannot make: ``changed_this_run`` is true of this row by
    construction — the turn was TOLD to change it — so what it did NOT change has to be read
    off the fields.  Every field the ask did not name is compared against the value the job
    was seeded with, and the one it did name is skipped by name.

    A violating sample is nameable: one that flips the switch and re-times the job on the way
    through, one that rebinds it to another routine, and one that rewrites the program while
    reporting only the flag."""
    seeded = {
        _NOTIFIES: _FINDS.notify,
        _SCHEDULE: _FINDS.schedule,
        _ROUTINE: _SEEDED_ROUTINE,
        _PROGRAM: _SEEDED_PROGRAM,
    }

    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = _job_row(sample, container)
        if row is None:
            return False, f"{container!r} is no longer in the registry"
        now = {
            _NOTIFIES: row.notifies,
            _SCHEDULE: row.schedule,
            _ROUTINE: row.routine,
            _PROGRAM: row.program,
        }
        drifted = sorted(
            field for field, was in seeded.items() if field != moved and now[field] != was
        )
        return not drifted, f"also moved {drifted}"

    return answer


# The row fields a change is read off, named once because two readers agree on each of them:
# the drift comparison above, which skips the one the ask named, and the label that names it.
_NOTIFIES = "notify"
_SCHEDULE = "schedule"
_ROUTINE = "routine"
_PROGRAM = "program"


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


def _nothing_else_was_touched(container: str) -> _ClaimFn:
    """No mechanism but the job itself was created, retired or edited.

    Read off the mutation LEDGER rather than off a field-by-field diff, so a change nobody
    enumerated is caught too.  A violating sample is nameable: one that stands a second
    container up beside the job instead of editing it, and one that reaches into a neighbour
    while it is in there."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        touched = sorted(
            one.name
            for one in sample.mechanisms
            if one.name != container and (one.born_this_run or one.changed_this_run)
        )
        return not touched, f"also created or changed {touched}"

    return answer


def _nothing_was_silenced_everywhere(sample: SampleObservation, _world: World) -> Answer:
    """Proactive notifications are still ON for the user.

    The neighbouring lever, and the measured wrong one: asked to quiet ONE job, a sample that
    reads the per-collection switch as unavailable reaches for the global mute, which silences
    everything and looks like success from inside the turn.  A violating sample is nameable in
    all three directions — silencing everything to quiet one job, to retire one, or because
    "it checks too early" read as "stop bothering me"."""
    return not sample.muted, "the turn muted notifications for everything"


# The labels the three cases share.  Named once because a label is a diff-join key: three
# copies of one sentence are three chances for a typo to split one claim's history in two.
_STILL_REGISTERED = "state: the job is still in the registry"
_STILL_LIVE = "state: the job is still live (active, scheduled, with a program to run)"
_KEPT_ITS_ENTRIES = "state: what it gathered is still there"
_NOTHING_ELSE_TOUCHED = "state: no other mechanism was created or changed"
_NOT_MUTED_EVERYWHERE = "state: notifications were not silenced everywhere instead"


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
        "turned off, Penny flips that job's own switch and leaves the job running — its "
        "cadence, its routine and its program untouched, every other mechanism untouched, "
        "and proactive notifications still on everywhere else."
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
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the field the ask named moved, and nothing else did.
    cohort.claim(
        _STILL_REGISTERED, _the_job_is_still_registered(_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(
        "state: the job's notifications are off",
        _the_switch_is(_FINDS.container, on=False),
        SpecCategory.STORE,
    )
    cohort.claim(_STILL_LIVE, _the_job_is_still_live(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        "state: nothing else on the job's row moved",
        _only_the_named_field_moved(_FINDS.container, moved=_NOTIFIES),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        _NOTHING_ELSE_TOUCHED, _nothing_else_was_touched(_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(_NOT_MUTED_EVERYWHERE, _nothing_was_silenced_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
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
        "retires it as a tombstone that still holds everything it gathered — no other "
        "mechanism touched, and nothing silenced anywhere else."
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
    """The job is retired as a tombstone, with everything it collected still readable.

    Its switch and its cadence are deliberately NOT claimed unchanged: an archived job is out
    of the dispatcher's reach whatever either says, so a turn that also silenced it on the way
    out did nothing the user can observe, and claiming it would fail a correct run over a
    difference with no consequence."""
    cohort = await _drive(chat_eval, model, _ARCHIVE)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(
        _STILL_REGISTERED, _the_job_is_still_registered(_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(
        "state: the job is retired (archived)",
        _the_job_is_retired(_FINDS.container),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        _NOTHING_ELSE_TOUCHED, _nothing_else_was_touched(_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(_NOT_MUTED_EVERYWHERE, _nothing_was_silenced_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
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
        "names a new one, Penny re-times that job and leaves everything else about it alone — "
        "and every clock time she names is one the job has actually had."
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
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(
        _STILL_REGISTERED, _the_job_is_still_registered(_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(
        "state: the job now runs at the hour they asked for",
        _the_rule_fires_at(_FINDS.container, _FIXED_HOUR),
        SpecCategory.STORE,
    )
    cohort.claim(_STILL_LIVE, _the_job_is_still_live(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        "state: nothing else on the job's row moved",
        _only_the_named_field_moved(_FINDS.container, moved=_SCHEDULE),
        SpecCategory.STORE,
    )
    cohort.claim(_KEPT_ITS_ENTRIES, _it_kept_what_it_gathered(_FINDS.container), SpecCategory.STORE)
    cohort.claim(
        _NOTHING_ELSE_TOUCHED, _nothing_else_was_touched(_FINDS.container), SpecCategory.STORE
    )
    cohort.claim(_NOT_MUTED_EVERYWHERE, _nothing_was_silenced_everywhere, SpecCategory.STORE)

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.claim(
        "reply: every clock time it names is one the job has had",
        _every_hour_it_names_is_one_the_job_has_had,
        SpecCategory.PROVENANCE,
        kind="reply",
    )

    cohort.measure(*_MEASURED)


# Every ported case, in one place — so the deterministic probe in ``test_eval_harness.py`` can
# drive each one's seeder and premise without a GPU.
OPERATION_CASES = (_NOTIFY_OFF, _ARCHIVE, _RE_TIME)


# ═══ NOT tranche 3's — left exactly as it was ════════════════════════════════
#
# "What does that thing actually do?" — reading a job's routine back, answered from the
# routine rather than from what the ambient header happens to say about it (#1804 took the
# recipe off that header, so the recipe is a read now, not a recall).  Reading a job is none
# of #2008's twelve behaviours and sits on no slot of #2004's map, so it is neither ported nor
# quarantined: it keeps its own scorer until it has a slot of its own.
#
# The reply has to describe the routine's two moves — it reads the page, it saves what it
# finds — and the families are described many ways, so each is matched against a broad verb
# set with a noun fallback.  Both patterns were widened against captured samples (the earlier
# verb-only search pattern false-negatived "look on the web" and "pulls in … databases",
# scoring faithful descriptions as misses): a scorer that reads only one phrasing measures
# wording rather than fidelity.  The patterns themselves are SHARED
# (``penny.tests.eval.conftest``): the learn-close narration story asks the same question of a
# reply about a routine it just learned (#1943), and two copies of one policy are two
# contracts.

# The ask, and the rule that it lends the scorer nothing — ENFORCED rather than trusted,
# because the leak is invisible once it exists: a question carrying one of the words the
# patterns match would credit a reply for repeating the question back.
_LEGIBILITY_ASK = f"what does {_THEIR_WORDS} actually do? walk me through it."
assert not describes(_LEGIBILITY_ASK, DESCRIBES_FETCH) and not describes(
    _LEGIBILITY_ASK, DESCRIBES_SAVE
), f"the ask must lend no word to the patterns that score the reply: {_LEGIBILITY_ASK!r}"


def landed_state_check(db: Database) -> Check:
    """Advisory: where the conversation machine landed the turn.

    The legibility case runs the production path, so the machine classifies the message
    before the chat agent sees it, and which state it landed in decides what the turn was
    even instructed to do — a reader of a surprising sample wants that first.  Rendered
    rather than demanded: this story is about what the turn SAYS, and pinning the edge would
    be measuring the machine's own beat here."""
    latest = db.machine.latest_transition()
    landed = latest.to_state if latest is not None else ConversationState.IDLE.value
    return Check(f"the machine landed the turn in {landed}", True, scored=False, kind="state")


def _describes_checks(reply: str) -> list[Check]:
    """The routine's two moves, as the reply describes them — one check each, so a reply
    that got half of it right reads as half right rather than as a failure."""
    return [
        Check(
            f"reply: it describes {claim}",
            describes(reply, pattern),
            kind="reply",
            anchor=REPLY_ANCHOR,
            rationale=None if describes(reply, pattern) else f"no {family} family in the reply",
        )
        for claim, family, pattern in (
            ("the page being read", "fetch/read", DESCRIBES_FETCH),
            ("what it finds being saved", "save/write", DESCRIBES_SAVE),
        )
    ]


def _score_legibility(db: Database, _before: set[str], reply: str) -> list[Check]:
    """She READ the routine and described what it does.

    The read is the spine: the ambient header carries each routine as one row — its name,
    what it is for, what it needs — and NOT its steps (#1804), so a reply describing the
    steps without a read is describing something it never saw.  Either read answers the
    question (the collection's rendered program, or the routine's own steps), so the check
    is about having read rather than about which verb did it."""
    read = tool_was_called(db, _METADATA) or tool_was_called(db, _SKILL_READ)
    return [
        Check(
            "spine: she read the routine rather than recalling it",
            read,
            kind="spine",
            anchor=f"{_METADATA}(",
            rationale=None if read else f"neither {_METADATA} nor {_SKILL_READ} was called",
        ),
        *_describes_checks(reply),
        landed_state_check(db),
    ]


async def test_she_describes_the_routine_the_job_runs(chat_eval: ChatEval) -> None:
    """Report-only.  "What does that actually do?" — answered from the routine, in plain
    words, without inventing a step it does not have."""
    await chat_eval(
        case_id="standing-describe-routine",
        message=_LEGIBILITY_ASK,
        seed=seed_standing_jobs(_FINDS),
        seed_skills=[WATCH_ROUTINE],
        score=_score_legibility,
        min_pass_rate=None,
        family=_LEGIBILITY_FAMILY,
    )
