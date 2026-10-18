"""The ``notify_composer_eval`` driver: the notify composer micro-context alone."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence

import pytest

from penny.config import Config
from penny.constants import PennyConstants
from penny.database import Database
from penny.llm.call_link import CallLink
from penny.notification import NotificationInput, render_notification
from penny.penny import Penny
from penny.tests.eval.conftest import (
    NO_DRAW,
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
    _micro_context_rows,
    _PendingCase,
    _require_ask,
    _require_report_only,
    _run_samples,
    _stamp_cause,
    given_to_the_model,
    sample_number,
)
from penny.tests.eval.drivers.extractor import _write_classifier_report
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.tools.micro_context import MicroContext


# ── The notify COMPOSER (#1911/#2363): a driver of its own ──────────────────────────────
def _notify_rerolled(db: Database) -> bool:
    """Did the notify draw more than once — the fragile signal, a sample that only got there
    by recovering.  One customer here, so one row is the clean case."""
    return len(_micro_context_rows(db, PennyConstants.NOTIFY_COMPOSE_AGENT_NAME)) > 1


def _observe_notify(
    db: Database, message: str | None, *, name: str, phrasing: str, arm: int
) -> eval_cohort.SampleObservation:
    """Read what ONE notify-compose sample answered — its own observer, not the chat one.

    A single-call context leaves no machine walk, no registry row, no collection entry and no
    outgoing message of its own — the send/queue gate is Python and runs nowhere near this
    draw.  What there IS to read is the drawn message itself, carried as ``reply`` rather than
    a structured ``output`` field: unlike extraction's typed ``MicroContextResult``, this
    customer's whole contract IS the message a user would receive, so the two SHARED reply
    claims (``assert_the_reply_answers_the_ask``, ``assert_every_value_in_the_reply_is_sourced``)
    read it directly rather than needing a composer-specific claim.

    ``given`` is the draw's own prompt, read through ``given_to_the_model`` (#2078) — the
    same contract every other single-call micro-context's provenance claim reads."""
    exclusion = _draw_exclusion(db, message or "")
    if message is None or exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion or NO_DRAW
        )
    return eval_cohort.SampleObservation(
        name=name, phrasing=phrasing, arm=arm, given=given_to_the_model(db), reply=message
    )


NotifyComposerEval = Callable[..., Awaitable["Cohort"]]


@pytest.fixture
def notify_composer_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[NotifyComposerEval]:
    """Drive the notify COMPOSER micro-context (#1911/#2363) N times, and NOTHING else.

    The composer's whole input is ONE framework-assembled document (``penny.notification``):
    what a cycle wrote, how it got there, and the past messages Python already looked up — no
    cycle runs, because what is measured is what the composer makes of a document rather than
    whether a cycle can produce one.  ``base`` carries every fact (the findings, the calls, the
    related messages) FIXED across the case's arms; ``description``/``also_described`` carry
    the one piece of upstream free text the document holds — the collection's own description
    — in five phrasings, the same role ``extractor_eval``'s ``instruction``/``also_instructed``
    play for the browse extractor.

    ``MicroContext.compose_notification`` makes the call, so the case exercises the shipped
    prompt, the shipped parse and the shipped reroll budget.  Synthetic here means the
    DOCUMENT's facts and the DESCRIPTION wordings are authored, never the prompt."""

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        base: NotificationInput,
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
        """Drive one document through the notify-compose draw, K phrasings of its own
        description, and return its COHORT.  A case with no ``description`` is refused by
        name."""
        _require_ask(
            case_id,
            "notify_composer_eval",
            "description=<the collection's own words>",
            description,
        )
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.MICRO_CONTEXT,
            behaviour=behaviour,
            area=area,
            wording=description,
        )
        eval_artifacts.begin_case(case_id)
        # One document's fixed facts, K wordings of its own description — the same shape
        # the extractor has with a fixed page and K wordings of the instruction.
        documents = [
            base.model_copy(update={"description": text}) for text in (description, *also_described)
        ]
        rendered = [render_notification(document) for document in documents]
        arms = _arms(rendered, [world], samples_per_phrasing, samples)
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
            content = spoken[sample_index]
            micro = MicroContext(penny.model_client)
            message: str | None = None
            try:
                message = await asyncio.wait_for(
                    micro.compose_notification(
                        content, call_link=CallLink.new_top(), run_target=penny.chat_agent.name
                    ),
                    timeout=timeout,
                )
                # The case is graded from its cohort's CLAIMS, made after every sample has
                # run, so the sample itself scores nothing at drive time.
                result = SampleResult.graded([])
                result.observe_rerolled(rerolled=_notify_rerolled(penny.db))
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no notify draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            phrasing = arms.label(sample_index)
            result.observation = _observe_notify(
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
                phrasing=content,
                agent_names=(PennyConstants.NOTIFY_COMPOSE_AGENT_NAME,),
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
