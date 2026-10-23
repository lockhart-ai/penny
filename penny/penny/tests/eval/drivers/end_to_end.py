"""The ``end_to_end_eval`` driver: a scripted chat phase, then the job's own real cycles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import NamedTuple

import pytest

from penny.config import Config
from penny.database import Database
from penny.datetime_utils import user_timezone_name
from penny.penny import Penny
from penny.tests.eval.conftest import (
    EVAL_LLM_IS_LOCAL,
    SAMPLES,
    TURN_REPLY_WAIT_SECONDS_LOCAL,
    UNSTATED,
    PassRate,
    SampleResult,
    Seeder,
    _aborted_on_model_error,
    _arms,
    _chat_tool_sequence,
    _declare_case,
    _drive_turns,
    _driven_cohort,
    _dump_thinking,
    _exclusion,
    _held_entries,
    _last_scripted_turn_at,
    _machine_walk,
    _mechanism_records,
    _PendingCase,
    _require_ask,
    _require_report_only,
    _run_samples,
    _scheduled_by_this_round,
    _seed_sample,
    _stamp_cause,
    _stored_entries,
    _write_sample_report,
    collection_names,
    given_to_the_model,
    outgoing_replies,
    sample_number,
)
from penny.tests.eval.drivers.collector_cycles import _drive_cycles, _DrivenCycles, cycle_script
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer

# ── End-to-end eval (#2375): scripted chat turns, then the job's own real cycles ─
#
# A chat case answers ONE turn; a collector case answers ONE cycle against a job seeded
# directly.  Neither can say whether what one layer builds is what the next layer actually
# runs on.  An end-to-end case drives a whole SCRIPT of real chat turns — a cold ask, and
# whatever the routine still has to be taught or re-pointed through — and then runs the job
# those turns created through its OWN real cycles, against pages that move between them.
#
# The entry condition (what the chat phase starts from) is still SEEDED, the way every case's
# is; what is new is that the round's own turns are DRIVEN rather than seeded as having
# already happened, because the question is whether the round's output is the cycle's input.
EndToEndEval = Callable[..., Awaitable["Cohort"]]

# How long each scripted turn's reply + settle wait may take.  REMOTE is this driver's own
# tuned value, unchanged; LOCAL swaps in the shared local turn deadline (#2713) — see
# ``TURN_REPLY_WAIT_SECONDS_LOCAL``'s own comment in conftest.py for the measurement.
_END_TO_END_TURN_REPLY_WAIT_SECONDS_REMOTE = 240.0
_END_TO_END_TURN_REPLY_WAIT_SECONDS = (
    TURN_REPLY_WAIT_SECONDS_LOCAL
    if EVAL_LLM_IS_LOCAL
    else _END_TO_END_TURN_REPLY_WAIT_SECONDS_REMOTE
)


class EndToEndArm(NamedTuple):
    """ONE arm of an end-to-end cohort: a scripted wording of the whole conversation, and the
    job's own cycles that follow it.

    ``turns`` is this wording's WHOLE script — the first turn is the whole natural ask (the
    page, what to watch, the cadence, the end, and "tell me when it changes"), and every turn
    after it is a FOLLOW-UP: driven only while no job has been created yet THIS RUN, read
    structurally off the registry before it is sent (#2494).  A Penny who stands the job up
    on the first turn is driven once; a Penny who asks first is driven through whatever
    follow-ups the script still needs — the walkthrough, then an acceptance restating the
    terms, or a plain agreement.  Each turn is written to read sensibly against whatever
    Penny actually said back to the one before it, since the arm drives her real reply rather
    than a scripted one.  ``seed``/``world`` lay the chat phase's ground down — an empty
    machine, or a job already configured and running elsewhere.  ``cycles`` is the
    created job's OWN real cycles that follow the script, each against its own browse
    register: the page as taught, then however the case asks it to move."""

    label: str
    turns: Sequence[str]
    seed: Seeder | None
    world: World
    cycles: Sequence[list[CannedPage]]


def _discover_the_new_jobs(db: Database, before: set[str]) -> list[str]:
    """Every collection the script's own turns created and scheduled — found STRUCTURALLY, off
    the registry's before/after read, never by a name the case expects.  A routine minted
    fresh names its own container, and a later variant may put the same ask's job somewhere
    else entirely (#2375), so the driver finds whatever the round built rather than look up a
    name it was told in advance."""
    return _scheduled_by_this_round(db, before)


def _no_job_created_yet(db: Database, before: set[str]) -> bool:
    """Whether the script has stood up NO job yet this run — the structural gate a follow-up
    turn is driven under (#2494).  The same registry read ``_discover_the_new_jobs`` makes at
    the script's end, called again before every turn after the first: a Penny who already
    created a job on an earlier turn gets no further turns driven at her, a Penny who hasn't
    gets the next one."""
    return not _discover_the_new_jobs(db, before)


async def _drive_end_to_end_script(
    penny: Penny,
    server: MockSignalServer,
    *,
    turns: Sequence[str],
    before: set[str],
    timeout: float,
    retryable: bool,
) -> tuple[str, int]:
    """Drive one arm's WHOLE script for real: the first turn unconditionally, then every
    further turn only while no job has been created yet this run (#2494).

    Returns the LAST reply actually driven and how many turns were sent — the count becomes
    the ``TURNS_DRIVEN`` variance feature, since a Penny who stands the job up on turn one and
    a Penny who asks first and is walked through it are both measured on the same claims, and
    only differ in how many turns it took."""
    reply = ""
    sent = 0
    for turn in turns:
        if sent and not _no_job_created_yet(penny.db, before):
            break
        reply = await _drive_turns(penny, server, [turn], timeout=timeout, retryable=retryable)
        sent += 1
    return reply, sent


async def _drive_end_to_end_cycles(
    penny: Penny, jobs: Sequence[str], cycles: Sequence[list[CannedPage]]
) -> tuple[list[eval_cohort.CycleSnapshot], list[_DrivenCycles]]:
    """Run EVERY created job's own real cycles against the arm's page registers.

    Returns the flattened snapshots a claim reads as a union (see ``CycleSnapshot``) beside
    each job's own ``_DrivenCycles`` — kept apart because the ran-guard and the walk string
    need the richer per-job record, while a claim only ever wants the flat list."""
    per_job = [await _drive_cycles(penny, job, cycles) for job in jobs]
    snapshots = [
        eval_cohort.CycleSnapshot(
            job=job,
            index=cycle.index,
            held=[
                eval_cohort.StoredEntry(collection=job, key=key, content=content)
                for key, content in cycle.after.items()
            ],
            sent=list(cycle.sent),
            outcome=cycle.outcome,
            reason=cycle.reason,
        )
        for job, driven in zip(jobs, per_job, strict=True)
        for cycle in driven.observed
    ]
    return snapshots, per_job


def _end_to_end_walk(db: Database, jobs: Sequence[str], per_job: Sequence[_DrivenCycles]) -> str:
    """The sample's whole walk: the conversation machine's moves, then every created job's
    own ordered cycle script — ``idle→elicit, elicit→learn, learn→apply, wrote, quiet,
    wrote+told`` for the ordinary one-job case that ran, each extra job's script joined
    beside it.

    Names the two ways a round can fail to leave anything for a cycle to read, structurally
    rather than by excluding the sample (#2375's review): ``no job`` when the script created
    nothing to run cycles against, ``refused`` when it created one and the dispatcher turned
    every cycle away.  Both are real behavioural failures a variant can reproduce, so the
    claims below have to be able to see them — never a harness-side void."""
    if not jobs:
        return f"{_machine_walk(db)}, no job"
    scripts = [cycle_script(driven.observed) for driven in per_job if driven.observed]
    cycles_walk = " | ".join(scripts) if scripts else "refused"
    return f"{_machine_walk(db)}, {cycles_walk}"


def _observe_end_to_end(
    db: Database,
    *,
    name: str,
    phrasing: str,
    arm: int,
    reply: str,
    before: set[str],
    jobs: list[str],
    per_job: list[_DrivenCycles],
    cycles: list[eval_cohort.CycleSnapshot],
    seeded_replies: int,
    turns_driven: int,
) -> eval_cohort.SampleObservation:
    """Everything an end-to-end sample left behind, read while its database is still live —
    the chat phase's own reads (the machine walk, the mechanisms, what it wrote, what it said)
    joined with the cycle phase's (what the created jobs' cycles held and sent), so a claim
    never has to know which phase produced which fact.

    A sample where the script built no job, or where every job it built had its cycles
    refused, is COMPLETE, not excluded (#2375's review): those are Penny's own failures —
    exactly what an end-to-end case exists to catch — and a variant that regresses either one
    has to be able to fail a claim here, not disappear from the pool.  Only a chat-phase
    model/infra error still voids the sample; it answered no turn at all to read anything
    off."""
    chat_exclusion = _exclusion(db, reply)
    if chat_exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=chat_exclusion
        )
    sent = [message for snapshot in cycles for message in snapshot.sent]
    # ``seeded_replies`` is how many messages the seeded world wrote into the
    # ``penny-messages`` log BEFORE the scripted turns ran; the list is oldest first, so the
    # slice drops them and the reply holds only what THIS run said (#2375).
    combined_reply = "\n".join([*outgoing_replies(db)[seeded_replies:], *sent])
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        turn_at=_last_scripted_turn_at(db),
        timezone=user_timezone_name(db),
        walk=_end_to_end_walk(db, jobs, per_job),
        entries=_stored_entries(db),
        held=_held_entries(db),
        mechanisms=_mechanism_records(db, before),
        tool_sequence=[
            *_chat_tool_sequence(db),
            *[call.tool for driven in per_job for cycle in driven.observed for call in cycle.calls],
        ],
        reply=combined_reply,
        aborted_on_model_error=_aborted_on_model_error(reply),
        given=given_to_the_model(db),
        notifications=sent,
        cycles=cycles,
        turns_driven=turns_driven,
    )


async def _drive_end_to_end_sample(
    penny: Penny,
    server: MockSignalServer,
    *,
    case_id: str,
    sample_index: int,
    arm: EndToEndArm,
    timeout: float,
    retryable: bool,
    observe: Callable[..., eval_cohort.SampleObservation],
) -> SampleResult:
    """ONE sample: seed the chat phase's ground, run the whole script for REAL, find every
    job it created, then run each one's own cycles for REAL against the arm's pages."""
    await _seed_sample(
        penny,
        world=arm.world,
        seed=arm.seed,
        browse=list(arm.world.pages),
        prepare=None,
    )
    # The seeded world's own sends are already in the message log before the turns run — a
    # COUNT (not a timestamp; the eval clock is pinned) so the observation reads only this
    # run's replies (#2375).
    seeded_replies = len(outgoing_replies(penny.db))
    before = collection_names(penny.db)
    reply, turns_driven = await _drive_end_to_end_script(
        penny, server, turns=arm.turns, before=before, timeout=timeout, retryable=retryable
    )
    jobs = _discover_the_new_jobs(penny.db, before)
    if jobs:
        cycles, per_job = await _drive_end_to_end_cycles(penny, jobs, arm.cycles)
    else:
        cycles, per_job = [], []
    # No drive-time guard: the cohort's claims are the only way a sample is graded, and a
    # sample that built no job or ran no cycle is COMPLETE — see ``_observe_end_to_end``.
    result = SampleResult.graded([])
    _stamp_cause(penny.db, result)
    # ``driven`` is only the turns ACTUALLY sent — a script whose first turn stood the job up
    # drove one, never the follow-ups the report would otherwise show as having been said.
    _write_sample_report(
        penny.db,
        case_id,
        sample_index,
        result=result,
        reply=reply,
        driven=list(arm.turns[:turns_driven]),
    )
    result.observation = observe(
        penny.db,
        reply=reply,
        before=before,
        jobs=jobs,
        per_job=per_job,
        snapshots=cycles,
        seeded_replies=seeded_replies,
        turns_driven=turns_driven,
    )
    _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
    return result


@pytest.fixture
def end_to_end_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[EndToEndEval]:
    """Drive one end-to-end sample per arm — a scripted whole conversation, then the job it
    created's own real cycles — and return the case's COHORT.

    A case passes ``arms`` — five scripted wordings of the same whole conversation, each
    bringing its own seed, its own chat-phase world and its own per-cycle pages.  Which jobs
    exist to run cycles against is discovered structurally after the script runs; a case
    never names the container its claims hinge on (#2375)."""
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        arms: Sequence[EndToEndArm] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        family: str | None = None,
    ) -> Cohort:
        """Drive one end-to-end script per sample and return the COHORT.

        A case with no ``arms`` is refused by name, the way every other driver refuses a case
        that names nothing to drive."""
        _require_ask(case_id, "end_to_end_eval", "arms=[EndToEndArm(...), …]", arms)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.WHOLE_TURN,
            behaviour=behaviour,
            area=area,
            wording=arms[0].label,
        )
        eval_artifacts.begin_case(case_id)
        driving = _arms(
            [arm.label for arm in arms],
            [arm.world for arm in arms],
            samples_per_phrasing,
            samples,
        )
        pending = _cohorts.setdefault(
            case_id,
            _PendingCase(
                case_id=case_id,
                family=family,
                module=request.module.__name__,
                behaviour=catalogued.sentence,
            ),
        )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            phrasing = driving.label(sample_index)
            arm_index = driving.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"

            # ``snapshots``, never ``cycles`` — the latter is a reserved per-sample-scorer
            # argument name the AST pin in ``test_eval_harness.py`` refuses on any driver.
            def observe(
                db: Database,
                *,
                reply: str,
                before: set[str],
                jobs: list[str],
                per_job: list[_DrivenCycles],
                snapshots: list[eval_cohort.CycleSnapshot],
                seeded_replies: int,
                turns_driven: int,
            ) -> eval_cohort.SampleObservation:
                return _observe_end_to_end(
                    db,
                    name=name,
                    phrasing=phrasing,
                    arm=arm_index,
                    reply=reply,
                    before=before,
                    jobs=jobs,
                    per_job=per_job,
                    cycles=snapshots,
                    seeded_replies=seeded_replies,
                    turns_driven=turns_driven,
                )

            return await _drive_end_to_end_sample(
                penny,
                server,
                case_id=case_id,
                sample_index=sample_index,
                arm=arms[arm_index],
                timeout=_END_TO_END_TURN_REPLY_WAIT_SECONDS,
                retryable=retryable,
                observe=observe,
            )

        results, perf, voided = await _run_samples(
            make_config,
            tmp_path,
            case_id=case_id,
            samples=driving.driven,
            drive=_drive,
            model=model,
        )
        cohort = _driven_cohort(case_id, model, results, voided, driving.arms)
        pending.add(cohort, results, perf, intended=driving.driven)
        return cohort

    yield _run
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
