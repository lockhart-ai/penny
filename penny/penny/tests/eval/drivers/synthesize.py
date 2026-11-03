"""The ``synthesize_eval`` driver: the synthesis micro-context alone (#2932).

Mirrors ``extractor_eval`` (``drivers/extractor.py``) for the SAME reason: the
whole input is a known, authored text — here a short numbered list of hits
``synthesize`` already retrieved, rather than a fetched page — so the case
drives ``MicroContext.synthesize`` directly, no tool dispatch, no retrieval.
What is measured is what the draw makes of the hits it was handed, never
whether ``find``'s own lookups turned up the right ones (that is a
deterministic unit test, ``tests/tools/test_memory_tools.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Sequence

import pytest

from penny.config import Config
from penny.constants import PageReading
from penny.database import Database
from penny.llm.call_link import CallLink
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
    _PendingCase,
    _require_ask,
    _require_report_only,
    _run_samples,
    _stamp_cause,
    given_to_the_model,
    sample_number,
)
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.tools.micro_context import MicroContext, MicroContextResult, MicroExtractOutcome

SynthesizeEval = Callable[..., Awaitable["Cohort"]]

# The draw's own STRUCTURED answer, named once — the deterministic half
# (``SYNTHESIZE_OUTCOME``) and the variance half (``SYNTHESIZE_VALUE``, the
# synthesis prose itself) read by the same names a case's claims and
# measures use.
SYNTHESIZE_OUTCOME = "outcome"
SYNTHESIZE_VALUE = "value"

# The one outcome meaning the draw never produced a usable synthesis at all
# (its whole reroll budget spent on poison) — excluded rather than scored,
# the synthesis twin of an extraction that never answered.
_SYNTHESIZE_FAILURES = frozenset({MicroExtractOutcome.EXTRACTION_FAILED})


def _synthesis_output(result: MicroContextResult) -> list[eval_cohort.OutputField]:
    """One synthesis draw's structured answer, field by field — both fields
    always, including the empty one on a failure, so a cohort's field
    comparison reads the same axes per sample whatever the outcome."""
    return [
        eval_cohort.OutputField(name=SYNTHESIZE_OUTCOME, value=result.outcome.value),
        eval_cohort.OutputField(name=SYNTHESIZE_VALUE, value=result.value),
    ]


def _observe_synthesis(
    db: Database,
    result: MicroContextResult | None,
    *,
    name: str,
    phrasing: str,
    arm: int,
) -> eval_cohort.SampleObservation:
    """What ONE synthesis sample answered — read directly off the draw's typed
    result, exactly as the extractor's own observer reads its draw (a
    single-call context leaves no machine walk, no store write, no outgoing
    message to read back instead)."""
    answer = (
        "" if result is None or result.outcome in _SYNTHESIZE_FAILURES else result.outcome.value
    )
    exclusion = _draw_exclusion(db, answer)
    if result is None or exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion or NO_DRAW
        )
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        given=given_to_the_model(db),
        output=_synthesis_output(result),
    )


def _synthesis_world(label: str, items: str) -> World:
    """The one ground a synthesis case is answered against, as a
    :class:`World` — the numbered ``items`` text, carried as a ``CannedPage``
    purely so the world has something to render (this driver installs no
    browser; the draw reads ``items`` directly, never through a tool).
    ``keeps``/``excludes`` are empty: those token sets exist for claims about
    what a ROUND chose to store, and this shape stores nothing."""
    return World(
        name=label,
        pages=(CannedPage(match=label, text=items, reading=PageReading.ARTICLE),),
        keeps=(),
        excludes=(),
    )


@pytest.fixture
def synthesize_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[SynthesizeEval]:
    """Drive the SYNTHESIS micro-context (#2932) N times, and nothing else.

    ``items`` is the numbered list the draw reads — authored directly by the
    case, the same shape ``SynthesizeTool._render_items_for_draw`` produces —
    and ``query`` is the question asked of it. ``MicroContext.synthesize``
    makes the call, so the case exercises the shipped prompt and the shipped
    reroll budget.
    """

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        items: str,
        query: str,
        behaviour: str = "",
        area: Area | None = None,
        also_instructed: Sequence[str] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = SINGLE_CALL_WAIT_SECONDS,
        family: str | None = None,
    ) -> Cohort:
        """Drive one set of items + a query through the synthesis draw and
        return its COHORT. A case with no ``query`` is refused by name."""
        _require_ask(case_id, "synthesize_eval", "query=<the question>", query)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.MICRO_CONTEXT,
            behaviour=behaviour,
            area=area,
            wording=query,
        )
        # One set of items, K wordings of the query — the world is a property
        # of the CASE, not of the arm.
        arms = _arms(
            [query, *also_instructed],
            [_synthesis_world(case_id, items)],
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

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            asked = spoken[sample_index]
            micro = MicroContext(penny.model_client, db=penny.db)
            drawn: MicroContextResult | None = None
            try:
                drawn = await asyncio.wait_for(
                    micro.synthesize(
                        items,
                        asked,
                        call_link=CallLink.new_top(),
                        run_target=penny.chat_agent.name,
                    ),
                    timeout=timeout,
                )
                # The case is graded from its cohort's CLAIMS, made after every
                # sample has run, so the sample itself scores nothing at drive time.
                result = SampleResult.graded([])
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no synthesis draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            phrasing = arms.label(sample_index)
            result.observation = _observe_synthesis(
                penny.db,
                drawn,
                name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                phrasing=phrasing,
                arm=arms.index_of(sample_index),
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
    # The case's claims are made in the TEST BODY, after the drive returns — so
    # the report is assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
