"""The ``chat_eval`` driver: one real chat turn per sample, K wordings of one ask."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Sequence

import pytest

from penny.config import Config
from penny.penny import Penny
from penny.tests.eval.conftest import (
    _MODEL_CALL_ATTEMPTS,
    EVAL_LLM_IS_LOCAL,
    SAMPLES,
    TURN_REPLY_WAIT_SECONDS_LOCAL,
    UNSTATED,
    Observer,
    PassRate,
    Preparer,
    SampleResult,
    Seeder,
    _arms,
    _declare_case,
    _drive_sample,
    _driven_cohort,
    _observe_sample,
    _PendingCase,
    _require_ask,
    _require_report_only,
    _run_samples,
    _seed_sample,
    sample_number,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer

# A chat-eval runner: a case passes ``ask`` + ``world`` and gets a :class:`Cohort` back to
# assert against.
ChatEval = Callable[..., Awaitable["Cohort"]]

# How long one chat turn's reply + settle wait may take.  REMOTE is this driver's own
# tuned value, unchanged; LOCAL swaps in the shared local turn deadline (#2713) — see
# ``TURN_REPLY_WAIT_SECONDS_LOCAL``'s own comment in conftest.py for the measurement.
_CHAT_TURN_REPLY_WAIT_SECONDS_REMOTE = 120.0
_CHAT_TURN_REPLY_WAIT_SECONDS = (
    TURN_REPLY_WAIT_SECONDS_LOCAL if EVAL_LLM_IS_LOCAL else _CHAT_TURN_REPLY_WAIT_SECONDS_REMOTE
)


@pytest.fixture
def chat_eval(make_config: Callable[..., Config], tmp_path, request) -> Iterator[ChatEval]:
    """Drive the real chat flow for one ask and return its COHORT.

    A case passes ``ask`` — with ``world`` for what it is answered against, and
    ``also_phrased`` for the other wordings of that SAME request — and gets a
    :class:`Cohort` back to make its claims against.  The three-section report is assembled
    at teardown, once the test body has made them.

    Each sample is fully hermetic — its own mock Signal server, DB, and real-model Penny:
    seed the user and the world (+ any case seed), embed the seeds, push the ask, wait for
    the reply, then read what the turn left behind.  A per-sample server is essential: a
    shared one leaks a prior sample's shut-down channel, which then errors on the next
    sample's broadcast.  A timeout is a sample the pool excludes by name, not a crash.
    """
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        ask: str | None = None,
        also_phrased: Sequence[str] = (),
        world: World | None = None,
        model: str = "",
        samples_per_phrasing: int = 0,
        seed: Seeder | None = None,
        prepare: Preparer | None = None,
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = _CHAT_TURN_REPLY_WAIT_SECONDS,
        family: str | None = None,
    ) -> Cohort:
        """Drive one request in its wordings and return the COHORT.

        Analysis runs over the complete set, which is what the variance statistics want and
        what no single sample can see.  A case with no ``ask`` is refused by name."""
        _require_ask(case_id, "chat_eval", "ask=<the request>", ask)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.WHOLE_TURN,
            behaviour=behaviour,
            area=area,
            wording=ask or "",
        )
        eval_artifacts.begin_case(case_id)
        # Chat's convenience over the general form: one world, K wordings of one ask.
        arms = _arms(
            [ask or "", *also_phrased],
            [world] if world is not None else [],
            samples_per_phrasing,
            samples,
        )
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

        def _observe(sample_index: int) -> Observer:
            phrasing = arms.label(sample_index)
            arm = arms.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"
            return lambda db, reply, before, held_before: _observe_sample(
                db,
                name=name,
                phrasing=phrasing,
                arm=arm,
                reply=reply,
                before=before,
                held_before=held_before,
            )

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            await _seed_sample(
                penny,
                world=world,
                seed=seed,
                browse=list(world.pages) if world is not None else None,
                prepare=prepare,
            )
            return await _drive_sample(
                penny,
                server,
                case_id=case_id,
                sample_index=sample_index,
                ask=spoken[sample_index],
                timeout=timeout,
                retryable=retryable,
                observe=_observe(sample_index),
            )

        results, perf, voided = await _run_samples(
            make_config,
            tmp_path,
            case_id=case_id,
            samples=arms.driven,
            drive=_drive,
            attempts=_MODEL_CALL_ATTEMPTS,
            model=model,
        )
        cohort = _driven_cohort(case_id, model, results, voided, arms.arms)
        pending.add(cohort, results, perf, intended=arms.driven)
        return cohort

    yield _run
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
