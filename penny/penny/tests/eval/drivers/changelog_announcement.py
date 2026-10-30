"""The ``changelog_eval`` driver: the real startup delivery turn (#2846) for a
deploy's new changelog entries, one unrelated ordinary turn, then the ask under
test — one real chat turn per model call, scored on the LAST turn's reply.

Drives the SHIPPED code a real deploy runs: ``ChatAgent.handle_changelog_report``
over the framework-rendered entries, the same content gates
``Penny._send_changelog_announcement`` applies before queuing, and the real
send-queue drainer — no re-implementation, so this case exercises exactly what
production would have left in the conversation before the later turn runs.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence

import pytest

from penny.changelog import ChangelogEntry
from penny.config import Config
from penny.constants import PennyConstants
from penny.llm.call_link import CallLink
from penny.notification import withheld_reason
from penny.penny import Penny
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    _MODEL_CALL_ATTEMPTS,
    EVAL_LLM_IS_LOCAL,
    SAMPLES,
    TURN_REPLY_WAIT_SECONDS_LOCAL,
    UNSTATED,
    PassRate,
    Preparer,
    SampleResult,
    Seeder,
    _arms,
    _declare_case,
    _drive_turns,
    _driven_cohort,
    _dump_thinking,
    _held_entries,
    _ModelCallError,
    _observe_sample,
    _PendingCase,
    _require_ask,
    _require_report_only,
    _run_samples,
    _seed_sample,
    _stamp_cause,
    _write_sample_report,
    collection_names,
    sample_number,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.text_validity import half_formed_send_reason

ChangelogEval = Callable[..., Awaitable["Cohort"]]

# Same shared local turn deadline as ``chat_eval`` (#2713); REMOTE is this driver's
# own tuned value.
_CHANGELOG_TURN_REPLY_WAIT_SECONDS_REMOTE = 120.0
_CHANGELOG_TURN_REPLY_WAIT_SECONDS = (
    TURN_REPLY_WAIT_SECONDS_LOCAL
    if EVAL_LLM_IS_LOCAL
    else _CHANGELOG_TURN_REPLY_WAIT_SECONDS_REMOTE
)


async def _deliver_changelog(
    penny: Penny, entries: Sequence[ChangelogEntry], *, retryable: bool
) -> None:
    """Run the real startup delivery turn: a live chat-agent draw over the
    framework-rendered entries, the same content gates production applies
    (``half_formed_send_reason`` / ``withheld_reason``), then drained through
    the real queue/channel path — so the history a later turn in this sample
    reads is exactly what a real deploy would have left behind.

    ``Penny._send_changelog_announcement``'s own two gates — a profile exists,
    the user has chatted before — are skipped here on purpose: they guard a
    fresh install from being bothered, which has nothing to do with the model
    behaviour this case measures, and ``seed_user`` seeds a profile with no
    message history by construction.

    A model-error reply raises :class:`_ModelCallError` while an attempt
    remains, the same retry ``_drive_turns`` gives an ordinary ask — this is
    the turn's transport failing, not Penny deciding anything."""
    run_id = uuid.uuid4().hex
    response = await penny.chat_agent.handle_changelog_report(
        list(entries), TEST_SENDER, call_link=CallLink.top(run_id)
    )
    answer = (response.answer or "").strip() if response is not None else ""
    if answer == PennyResponse.AGENT_MODEL_ERROR and retryable:
        raise _ModelCallError
    assert answer, "the changelog delivery turn produced no reply"
    assert half_formed_send_reason(answer) is None, "the changelog delivery reply was not whole"
    assert withheld_reason(penny.db, answer) is None, "the changelog delivery reply was withheld"
    penny.db.send_queue.enqueue(answer, collection=PennyConstants.CHANGELOG_LOG_AUTHOR)
    delivered = await penny.send_queue_drainer.execute()
    assert delivered, "the changelog announcement was queued but never delivered"


@pytest.fixture
def changelog_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[ChangelogEval]:
    """Drive the real changelog delivery turn, one unrelated turn, then the ask
    under test, and return the COHORT scored on the LAST turn's reply.

    ``changelog_entries`` is the deploy's own new entries; ``unrelated_turn`` sits
    between the delivery and the ask, so the question is never the very next
    message after the announcement. Each sample is fully hermetic, exactly like
    ``chat_eval``: its own mock Signal server, DB, and real-model Penny."""
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        changelog_entries: Sequence[ChangelogEntry],
        unrelated_turn: str,
        ask: str | None = None,
        also_phrased: Sequence[str] = (),
        world: World | None = None,
        model: str = "",
        samples_per_phrasing: int = 0,
        seed: Seeder | None = None,
        prepare: Preparer | None = None,
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = _CHANGELOG_TURN_REPLY_WAIT_SECONDS,
        family: str | None = None,
    ) -> Cohort:
        """Drive one later question in its wordings, each sample preceded by the
        same real delivery turn and the same unrelated turn, and return the
        cohort. A case with no ``ask`` is refused by name."""
        _require_ask(case_id, "changelog_eval", "ask=<the later question>", ask)
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
            await _deliver_changelog(penny, changelog_entries, retryable=retryable)

            before = collection_names(penny.db)
            held_before = _held_entries(penny.db)
            turns = [unrelated_turn, spoken[sample_index]]
            reply = ""
            try:
                reply = await _drive_turns(
                    penny, server, turns, timeout=timeout, retryable=retryable
                )
                result = SampleResult.graded([])
                _stamp_cause(penny.db, result)
                _write_sample_report(
                    penny.db, case_id, sample_index, result=result, reply=reply, driven=turns
                )
            except TimeoutError:
                result = SampleResult.binary(["no reply within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
                _write_sample_report(penny.db, case_id, sample_index, result=result)

            phrasing = arms.label(sample_index)
            arm = arms.index_of(sample_index)
            name = f"{case_id}-{sample_number(sample_index)} ({phrasing})"
            result.observation = _observe_sample(
                penny.db,
                name=name,
                phrasing=phrasing,
                arm=arm,
                reply=reply,
                before=before,
                held_before=held_before,
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

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
