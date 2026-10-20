"""The ``job_report_eval`` driver: the delivery turn that writes a notifying cycle's
message (#2656).

The notify composer micro-context this driver used to drive is retired: a finished
cycle enqueues a report row, and the SEND-QUEUE DRAINER composes the message on ONE
chat-agent delivery turn — the framework renders the report (the job's name, what it
is for, why it ran, and the FULL content of every entry the cycle wrote), and the
chat agent writes the message with its whole context.  This driver drives exactly
that path: it seeds the job and the entries one of its cycles wrote (citing the
seeded run id, which is the run the report row names), enqueues the report row the
way ``_notify_if_due`` does, and runs the drainer's real ``execute()`` — no
re-implementation, so the case exercises the shipped report render, the shipped
delivery turn, the shipped gate and the shipped queue.

What is measured is what the delivery turn makes of the report: the message the user
would receive, read off the send queue it was stored to.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence

import pytest

from penny.config import Config
from penny.constants import CycleTrigger
from penny.database import Database
from penny.database.memory import EntryInput
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
    SAMPLES,
    SINGLE_CALL_WAIT_SECONDS,
    UNSTATED,
    PassRate,
    SampleResult,
    _arms,
    _declare_case,
    _draw_exclusion,
    _driven_cohort,
    _dump_thinking,
    _PendingCase,
    _require_ask,
    _require_report_only,
    _run_samples,
    _stamp_cause,
    given_to_the_model,
    sample_number,
    seeded_run_id,
)
from penny.tests.eval.drivers.collector_cycles import deliver_pending_reports, queued_sends
from penny.tests.eval.drivers.extractor import _write_classifier_report
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer

# The report row a notifying cycle queues is enqueued by the DRIVER here, on the
# collection the case seeded — the same three facts ``_notify_if_due`` puts on one
# (the mechanism, the lane, the run whose entries the report is built from).


def _seed_job(db: Database, name: str, description: str, entries: Sequence[EntryInput]) -> str:
    """One job holding the entries ONE of its cycles wrote, cited to a seeded run —
    the run id the report row will name, so ``entries_written_by_run`` reads exactly
    what the case wrote and nothing a neighbour run left."""
    run_id = seeded_run_id(name)
    db.memories.create_collection(name, description, notify=True)
    require_memory(db, name).write(list(entries), author="user", run_id=run_id)
    return run_id


def _observe_delivery(
    db: Database, message: str | None, *, name: str, phrasing: str, arm: int
) -> eval_cohort.SampleObservation:
    """Read what ONE delivery sample left the user — the message off the send queue
    it was stored to, which is what the drainer delivers.

    ``given`` is the delivery turn's own prompt, read through ``given_to_the_model``
    (#2078) — the same contract every other sample's provenance claim reads."""
    exclusion = _draw_exclusion(db, message or "")
    if message is None or exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name,
            phrasing=phrasing,
            arm=arm,
            complete=False,
            exclusion=exclusion or "no message was delivered",
        )
    return eval_cohort.SampleObservation(
        name=name, phrasing=phrasing, arm=arm, given=given_to_the_model(db), reply=message
    )


JobReportEval = Callable[..., Awaitable["Cohort"]]


@pytest.fixture
def job_report_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[JobReportEval]:
    """Drive the DELIVERY TURN (#2656) N times, and NOTHING else — one job, one
    finished cycle's entries, one report row, one message.

    ``base`` carries the job's fixed facts: its name, and the entries its cycle wrote
    (held byte-identical across the case's arms).  ``description``/``also_described``
    carry the one piece of upstream free text the report holds — the collection's own
    description — in five phrasings, the same role the extractor's
    ``instruction``/``also_instructed`` play for the browse extractor.

    The message is the real delivery turn's reply: the shipped report render, the
    shipped chat prompt, the shipped gate and the shipped queue.  Synthetic here
    means the JOB's facts and the DESCRIPTION wordings are authored, never the
    prompt."""

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        collection: str,
        entries: Sequence[EntryInput],
        description: str,
        world: World,
        behaviour: str = "",
        area: Area | None = None,
        also_described: Sequence[str] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = SINGLE_CALL_WAIT_SECONDS,
        family: str | None = None,
    ) -> Cohort:
        """Drive one job's report through the delivery turn, K phrasings of the job's
        description, and return its COHORT.  A case with no ``description`` is refused
        by name."""
        _require_ask(
            case_id,
            "job_report_eval",
            "description=<the collection's own words>",
            description,
        )
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.WHOLE_TURN,
            behaviour=behaviour,
            area=area,
            wording=description,
        )
        eval_artifacts.begin_case(case_id)
        # One job's fixed facts (its name and its cycle's entries), K wordings of its
        # own description — the same shape the extractor has with a fixed page and K
        # wordings of the instruction.
        descriptions = (description, *also_described)
        arms = _arms(descriptions, [world], samples_per_phrasing, samples)
        spoken = arms.spoken
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
            # The sample's own seed: the job with THIS arm's description wording, the
            # entries its cycle wrote, and the report row naming that cycle's run.
            run_id = _seed_job(penny.db, collection, spoken[sample_index], tuple(entries))
            penny.db.send_queue.enqueue_report(collection, CycleTrigger.CADENCE, run_id)
            message: str | None = None
            try:
                await asyncio.wait_for(deliver_pending_reports(penny), timeout=timeout)
                sent = queued_sends(penny.db, collection)
                message = sent[0] if sent else None
                # The case is graded from its cohort's CLAIMS, made after every sample
                # has run, so the sample itself scores nothing at drive time.
                result = SampleResult.graded([])
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no delivery within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            phrasing = arms.label(sample_index)
            result.observation = _observe_delivery(
                penny.db,
                message,
                name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                phrasing=phrasing,
                arm=arms.index_of(sample_index),
            )
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=spoken[sample_index],
                agent_names=("chat",),
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        results, perf, voided = await _run_samples(
            make_config, tmp_path, case_id=case_id, samples=arms.driven, drive=_drive, model=model
        )
        cohort = _driven_cohort(case_id, model, results, voided, arms.arms)
        pending.add(cohort, results, perf, intended=arms.driven)
        return cohort

    yield _run
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
