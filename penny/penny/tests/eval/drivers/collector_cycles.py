"""The ``collector_cycles_eval`` driver: one real collector cycle per sample."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import pytest
from sqlmodel import Session, col, select

from penny.config import Config
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import PromptLog, SendQueueItem
from penny.penny import Penny
from penny.tests.eval.conftest import (
    _PERF_WINDOW,
    NO_CYCLE,
    NO_MEASURED_TURN,
    PROVIDER_REFUSED,
    SAMPLES,
    UNSTATED,
    Check,
    PassRate,
    Preparer,
    SampleResult,
    Seeder,
    _arms,
    _declare_case,
    _driven_cohort,
    _dump_thinking,
    _embed_seeds,
    _PendingCase,
    _provider_refused,
    _require_ask,
    _require_report_only,
    _response_tool_calls,
    _run_samples,
    _stamp_cause,
    _stored_entries,
    _write_sample_report,
    collection_entries,
    given_to_the_model,
    install_browse,
    live_prompts,
    measured_turn_ran,
    sample_number,
    seed_user,
    seed_world_stores,
    tool_call_name,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer


class CycleCall(NamedTuple):
    """One tool call a collector cycle made — its name and the arguments it carried.

    The arguments travel as the mapping the ledger stores, because a reader asks things of
    them a typed shape could not answer: WHICH page a browse went to is a question about a
    tool nobody enumerated, and a skill is an arbitrary sequence of them."""

    tool: str
    arguments: dict


@dataclass
class CycleObservation:
    """What ONE collector cycle did, read off persisted state after it closed (#1905).

    A watch is only a watch across cycles — "it changed" and "it already told me" are
    claims about the SECOND run, not the first — so a case that drives more than one cycle
    needs each cycle's footprint kept apart rather than a single end-state snapshot in
    which the two have already merged.

    ``before``/``after`` are the collection's entries as the cycle found and left them,
    ``sent`` what it put on the SEND QUEUE (read explicitly — see ``queued_sends``),
    ``calls`` what it called in order, ``served`` the page text the browse tool actually
    returned, and ``outcome``/``reason`` the run record it closed with (a write-gate STOP
    names itself in the reason).

    ``served`` exists because a multi-cycle case installs a DIFFERENT browse register per
    cycle, and every claim the case makes rests on each cycle having been shown its own.
    A register that failed to swap is invisible in every other reading — the model would
    behave correctly for the page it saw, and the whole case would report as the model
    getting it wrong."""

    index: int
    before: dict[str, str]
    after: dict[str, str]
    sent: list[str]
    calls: list[CycleCall]
    served: list[str]
    outcome: str | None
    reason: str | None

    @property
    def changed(self) -> bool:
        """Whether the cycle moved the collection's durable state."""
        return self.after != self.before

    @property
    def tools(self) -> list[str]:
        """The names of the calls it made, in order."""
        return [call.tool for call in self.calls]


def queued_sends(db: Database, collection: str) -> list[str]:
    """Every message a collection has put on the send queue, oldest first — pending and
    already-delivered alike.

    Read EXPLICITLY here rather than through ``pending_items`` because the queue is the
    harness's known blind spot: a collector cycle ENQUEUES, and the drainer that delivers
    is a separate schedule that may or may not have run by the time a claim is answered.  A
    pending-only read therefore reports a delivered notification as silence, which is the
    one thing a notify contract must never get wrong."""
    return [row.content for row in _send_queue_rows(db) if row.collection == collection]


def _send_queue_rows(db: Database) -> list[SendQueueItem]:
    """Every send-queue row that IS or WILL BE a message to the user, oldest first.

    Cancelled rows are excluded structurally, as the store's own readers exclude them
    (#1634): a cancelled row was never sent, so counting one as a notification would
    report a message the user never got."""
    with Session(db.engine) as session:
        return list(
            session.exec(
                select(SendQueueItem)
                .where(col(SendQueueItem.cancelled_at).is_(None))
                .order_by(col(SendQueueItem.created_at).asc())
            ).all()
        )


def pages_served(db: Database) -> list[str]:
    """Every page the browse tool has returned this sample, oldest first — read off the
    browse-results log, which is where the tool journals what it actually fetched.

    The harness's own integrity read: a case that installs a different register per cycle
    is claiming each cycle saw a different world, and this is the only place that claim
    can be checked against what the tool really returned."""
    memory = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    entries = memory.read_all() if memory is not None else []
    return [entry.content for entry in entries]


def _live_run_ids(db: Database) -> set[str]:
    """Every run id this sample has written — read through ``live_prompts``, so the seeded
    ledger is excluded by the one chokepoint rather than by remembering to."""
    return {row.run_id for row in live_prompts(db, _PERF_WINDOW) if row.run_id}


def _observe_cycle(
    db: Database,
    collection: str,
    *,
    index: int,
    before: dict[str, str],
    sent_before: int,
    served_before: int,
    runs_before: set[str],
) -> CycleObservation:
    """One cycle's footprint, composed from the state it left: what it wrote, what it
    queued, what it called, what it was SERVED, and the run record it closed with."""
    runs = _live_run_ids(db) - runs_before
    rows = [row for row in live_prompts(db, _PERF_WINDOW) if row.run_id in runs]
    closed = next((row for row in rows if row.run_outcome is not None), None)
    return CycleObservation(
        index=index,
        before=before,
        after=collection_entries(db, collection),
        sent=queued_sends(db, collection)[sent_before:],
        calls=_ordered_calls(rows),
        served=pages_served(db)[served_before:],
        outcome=closed.run_outcome if closed is not None else None,
        reason=closed.run_reason if closed is not None else None,
    )


def _ordered_calls(rows: list[PromptLog]) -> list[CycleCall]:
    """The calls one run's rows carry, oldest first — each with the arguments it was made
    with, decoded from the JSON string the wire stores them as."""
    return [
        CycleCall(
            tool=tool_call_name(call),
            arguments=_decoded_arguments(call.get("function", {}).get("arguments")),
        )
        for row in sorted(rows, key=lambda row: row.timestamp)
        for call in _response_tool_calls(row)
    ]


def _decoded_arguments(raw: object) -> dict:
    """One logged call's arguments as a mapping — an unparseable payload reads as no
    arguments rather than raising, since a reader asking WHICH page was fetched wants the
    calls that do carry one."""
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


CollectorCyclesEval = Callable[..., Awaitable["Cohort"]]


class _DrivenCycles(NamedTuple):
    """What driving a case's cycles produced: each cycle's footprint, and whether the
    dispatcher actually RAN every one of them.

    ``ran`` is the harness half.  ``run_for`` refuses WITHOUT running a cycle at all when
    the collection is missing, archived, or carries no usable program — and a refusal
    leaves every check failing on a world the model never saw, which the failure-cause
    partition would then tag behavioral.  So the refusal travels as its own guard rather
    than as a puzzling zero.

    A refusal is read STRUCTURALLY — the cycle produced no run of its own — rather than
    from the returned flag, which is also false for a cycle that really ran and failed."""

    observed: list[CycleObservation]
    ran: bool
    refusal: str | None


async def _drive_cycles(
    penny: Penny,
    collection: str,
    cycles: Sequence[list[CannedPage]],
) -> _DrivenCycles:
    """Run the real collector cycle once per entry in ``cycles``, each against that
    entry's own browse register, observing what each one left behind.

    The register is re-installed between cycles because that is the whole point: the
    world MOVED between two runs of the same watch, and what the second cycle does about
    it is the contract."""
    observed: list[CycleObservation] = []
    refusal: str | None = None
    for index, pages in enumerate(cycles):
        install_browse(penny, pages)
        before = collection_entries(penny.db, collection)
        sent_before = len(queued_sends(penny.db, collection))
        served_before = len(pages_served(penny.db))
        runs_before = _live_run_ids(penny.db)
        _, message = await penny.collector.run_for(collection)
        if _live_run_ids(penny.db) == runs_before and refusal is None:
            refusal = message
        observed.append(
            _observe_cycle(
                penny.db,
                collection,
                index=index,
                before=before,
                sent_before=sent_before,
                served_before=served_before,
                runs_before=runs_before,
            )
        )
    return _DrivenCycles(observed=observed, ran=refusal is None, refusal=refusal)


def _cycles_ran_check(driven: _DrivenCycles) -> Check:
    """The 'the dispatcher actually ran every cycle' guard as a ``Check`` on the sample's
    own block, so a sample whose collection the dispatcher refused can never read as the
    model doing nothing.  The cohort excludes that sample by name (``NO_CYCLE``)."""
    return Check(
        "every cycle ran",
        driven.ran,
        kind="guard",
        rationale=None
        if driven.ran
        else f"the dispatcher refused the collection: {driven.refusal}",
    )


class CycleArm(NamedTuple):
    """ONE arm of a collector cohort: a wording of the job's instruction, and the page that
    answers it.

    A collector has no user turn, so its natural language is the ``extract`` instruction its
    rendered program carries — written by the ``SkillSubstitution`` on that path and printed by
    ``render_skill``, so a wording reaches the model through the shipped instantiation seam
    rather than through a hand-authored program — plus the prose of the page that answers it.
    Those two are what vary across the arms.

    The FACTS do not: one url, one set of bound values, one reading, and a byte-identical datum
    line on every page.  Constant facts are what let a claim name a value — the store holds the
    price the page carries — instead of retreating to a shape that cannot tell a right reading
    from a plausible one.

    ONE register, because a case drives ONE cycle.  Which behaviour a case measures is set by
    the ENTRY CONDITION its ``seed`` lays down — an empty collection or one already holding a
    reading — not by scripting several cycles inside a single test.

    ``text`` is what makes this arm this arm, rendered verbatim in the report so a reader who
    sees "phrasing 3 diverged" can see which wording that was.  ``seed`` lays THIS arm's job
    and entry condition down, ``pages`` is the register its cycle reads, and ``world`` is those
    same pages — carried so a claim is answered against the ground its own sample actually
    read, since the prose differs per arm."""

    text: str
    seed: Seeder
    pages: list[CannedPage]
    world: World


# One cycle's terminal SHAPE, as the enactment contract reads it.  A vocabulary of three words
# rather than a record per cycle: the whole watch contract is "what did each cycle do", so the
# cycles compose into one ordered script that a claim reads and a variance feature measures —
# which is why a multi-cycle sample needs no per-cycle field on the observation.
CYCLE_WROTE = "wrote"
CYCLE_QUIET = "quiet"
CYCLE_TOLD = "told"
CYCLE_DEAD = "no run"


def _cycle_shape(cycle: CycleObservation) -> str:
    """What ONE cycle did: whether it moved the collection, and whether it spoke.

    Read off persisted state — the collection's entries as the cycle found and left them, and
    the SEND QUEUE — never off a harness flag.  A cycle that produced no run of its own is
    named as such rather than reading as a silent success."""
    if cycle.outcome is None and not cycle.calls:
        return CYCLE_DEAD
    moved = CYCLE_WROTE if cycle.changed else CYCLE_QUIET
    return f"{moved}+{CYCLE_TOLD}" if cycle.sent else moved


def cycle_script(cycles: Sequence[CycleObservation]) -> str:
    """The ordered shapes of a sample's cycles — ``wrote, quiet, wrote+told``.

    The whole watch contract in one value: a claim reads a position of it and the variance
    table measures the script entire, so a cohort where one arm stopped notifying shows up as
    a divergence rather than as a number nobody looks at."""
    return ", ".join(_cycle_shape(cycle) for cycle in cycles)


def _cycles_exclusion(db: Database, script: str) -> str | None:
    """Why a cycle-driven sample cannot be counted, or ``None`` when it can.

    A cycle the dispatcher REFUSED never ran against the world the case built, so every claim
    would be answered on a world the model never saw — the failure-cause partition would then
    tag it behavioural, which is the one thing it is not.  Checked ahead of that: a call the
    PROVIDER refused outright (``PROVIDER_REFUSED``, #2526) inside any of the sample's cycles
    is the same class of infrastructure fault as a refused cycle, whichever cycle it hit."""
    if not measured_turn_ran(db):
        return NO_MEASURED_TURN
    if _provider_refused(db):
        return PROVIDER_REFUSED
    if CYCLE_DEAD in script:
        return NO_CYCLE
    return None


def _observe_cycles(
    db: Database,
    driven: _DrivenCycles,
    *,
    name: str,
    phrasing: str,
    arm: int,
    collection: str,
) -> eval_cohort.SampleObservation:
    """Read what ONE cycle-driven sample left behind — its own observer, not the chat one.

    The tool sequence is read from the CYCLES' own calls rather than through
    ``chat_run_tool_sequences``, whose agent filter would return an empty list here on every
    sample (#2017).  The notifications come off the SEND QUEUE, because a cycle enqueues and
    the drainer is a separate schedule — a pending-only read of outgoing messages reports a
    delivered notification as silence — and they are carried as a LIST as well as joined text,
    since "told once" and "told twice" are different findings a blob cannot tell apart.

    ``held``, ``run_outcome`` and ``run_reason`` come off the LAST cycle, which for an
    arm-driven case is its only one.  They are what a collector has instead of a machine walk:
    the store's own end state, and the record fields the run closed with.  ``held`` names the
    BOUND collection on every entry, because it is the one container this job writes to and a
    claim reads the whole entry — key and content — exactly as a chat sample's does."""
    script = cycle_script(driven.observed)
    exclusion = _cycles_exclusion(db, script)
    if exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion
        )
    sent = [message for cycle in driven.observed for message in cycle.sent]
    last = driven.observed[-1] if driven.observed else None
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        walk=script,
        entries=_stored_entries(db),
        tool_sequence=[call.tool for cycle in driven.observed for call in cycle.calls],
        reply="\n".join(sent),
        given=given_to_the_model(db),
        held=[
            eval_cohort.StoredEntry(collection=collection, key=key, content=content)
            for key, content in (last.after if last is not None else {}).items()
        ],
        run_outcome=last.outcome if last is not None else None,
        run_reason=last.reason if last is not None else None,
        notifications=list(sent),
    )


@pytest.fixture
def collector_cycles_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[CollectorCyclesEval]:
    """Drive ONE real collector cycle (``run_for``) per sample and return the case's COHORT.

    The one collector driver there is.  A case passes ``arms`` — five wordings of ONE job's
    instruction, each bringing its own seed, its own page and its own world — and which
    behaviour it measures is set by the ENTRY CONDITION each arm's seed lays down (#1905).
    The cycle's footprint is read off persisted state as a ``CycleObservation``.

    Each sample is hermetic (its own mock Signal server, DB and real-model Penny).  Seeds
    run first, then embeddings backfill, then ``prepare`` gets the constructed Penny — a
    loud world probe, so a drifted seed fails in the seed rather than after GPU time."""

    _cohorts: dict[str, _PendingCase] = {}

    async def _sample(
        penny: Penny,
        *,
        case_id: str,
        sample_index: int,
        collection: str,
        arm: CycleArm,
        prepare: Preparer | None,
        observe: Callable[[Database, _DrivenCycles], eval_cohort.SampleObservation],
    ) -> SampleResult:
        """ONE sample against a constructed Penny: lay its world down, probe it, drive its
        cycle, and write its report block."""
        seed_user(penny.db)
        seed_world_stores(penny.db, arm.world)
        arm.seed(penny.db)
        await _embed_seeds(penny)
        if prepare is not None:
            prepare(penny)
        driven = await _drive_cycles(penny, collection, [arm.pages])
        # The sample is graded from its cohort's CLAIMS, made once every sample has run, so it
        # scores nothing at drive time and carries only the ran-guard.
        result = SampleResult.graded([_cycles_ran_check(driven)])
        _stamp_cause(penny.db, result)
        _write_sample_report(penny.db, case_id, sample_index, result=result)
        # Read while THIS sample's database is still open — the only moment what the cycle
        # left behind is available at all.
        result.observation = observe(penny.db, driven)
        _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
        return result

    async def _run(
        *,
        case_id: str,
        collection: str,
        behaviour: str = "",
        area: Area | None = None,
        arms: Sequence[CycleArm] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        prepare: Preparer | None = None,
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        family: str | None = None,
    ) -> Cohort:
        """Drive one collector cycle per sample and return the COHORT.

        The arms do NOT share a world, which is why the world lives on the arm rather than
        on the cohort.  A case with no ``arms`` is refused by name."""
        _require_ask(case_id, "collector_cycles_eval", "arms=[CycleArm(...), …]", arms)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.COLLECTOR_CYCLE,
            behaviour=behaviour,
            area=area,
            wording=arms[0].text,
        )
        eval_artifacts.begin_case(case_id)
        driving = _arms(
            [arm.text for arm in arms],
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

        def _observer(
            sample_index: int,
        ) -> Callable[[Database, _DrivenCycles], eval_cohort.SampleObservation]:
            phrasing = driving.label(sample_index)
            arm = driving.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"
            return lambda db, ran: _observe_cycles(
                db, ran, name=name, phrasing=phrasing, arm=arm, collection=collection
            )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            return await _sample(
                penny,
                case_id=case_id,
                sample_index=sample_index,
                collection=collection,
                arm=arms[driving.index_of(sample_index)],
                prepare=prepare,
                observe=_observer(sample_index),
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
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
