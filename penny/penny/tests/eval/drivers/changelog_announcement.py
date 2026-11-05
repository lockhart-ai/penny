"""The ``changelog_eval`` driver: the real startup delivery turn (#2846) for a
deploy's new changelog entries, one unrelated ordinary turn, then the ask under
test — one real chat turn per model call, scored on the LAST turn's reply.

Drives the SHIPPED code a real deploy runs: ``ChatAgent.handle_changelog_report``
over the framework-rendered entries, the same content gates
``Penny._send_changelog_announcement`` applies before delivery, and the real
send-queue drainer's own ``deliver_composed`` (#2913 — no queue row, delivered
straight through the channel) — no re-implementation, so this case exercises
exactly what production would have left in the conversation before the later
turn runs.
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
    (``half_formed_send_reason`` / ``withheld_reason``), then delivered straight
    through the real channel path (#2913 — no queue row, no idle wait) — so the
    history a later turn in this sample reads is exactly what a real deploy
    would have left behind.

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
    delivered = await penny.send_queue_drainer.deliver_composed(
        answer, collection=PennyConstants.MEMORY_CHANGELOG_LOG, run_id=run_id
    )
    assert delivered, "the changelog announcement was composed but never delivered"


async def _deliver_changelog_unforced(
    penny: Penny, entries: Sequence[ChangelogEntry], *, retryable: bool
) -> str:
    """The sibling of ``_deliver_changelog`` for a case whose claim is ABOUT whether
    a reply happens at all (#2974): the trigger is now facts-only, with no
    instruction forcing a reply, so "she said nothing this turn" is part of what
    is measured, never a thing this helper asserts against. Returns the text that
    actually reached the user, or ``""`` when nothing did (no draw, a half-formed
    draw, or one the send gate withheld) — each a real, distinct outcome a report
    can read apart, never collapsed into one "it failed" case."""
    run_id = uuid.uuid4().hex
    response = await penny.chat_agent.handle_changelog_report(
        list(entries), TEST_SENDER, call_link=CallLink.top(run_id)
    )
    answer = (response.answer or "").strip() if response is not None else ""
    if answer == PennyResponse.AGENT_MODEL_ERROR and retryable:
        raise _ModelCallError
    if not answer:
        return ""
    if half_formed_send_reason(answer) is not None:
        return ""
    if withheld_reason(penny.db, answer) is not None:
        return ""
    delivered = await penny.send_queue_drainer.deliver_composed(
        answer, collection=PennyConstants.MEMORY_CHANGELOG_LOG, run_id=run_id
    )
    return answer if delivered else ""


@pytest.fixture
def changelog_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[ChangelogEval]:
    """Drive the real changelog delivery turn, then either a later ask (in its
    wordings) or nothing at all, and return the COHORT scored on the LAST turn's
    reply.

    ``changelog_entries`` is the deploy's own new entries. Two shapes, by
    whether ``ask`` is given:

    - ``ask`` set — the ordinary shape: ``unrelated_turn`` sits between the
      delivery and the ask, so the question is never the very next message
      after the announcement (pass ``""`` for a case that needs the ask to
      follow directly instead — no turn is pushed; an empty string is never
      sent as a real message), and the cohort is scored on the ASK's reply.
    - ``ask`` left ``None`` (#2974) — no follow-up at all: the cohort is scored
      on the DELIVERY TURN'S OWN reply instead, for a case whose claims are
      about its discretion (she decides for herself whether, how, and how many
      times to reply over a fact, never an instruction) rather than a later
      turn's recall of it. ``world`` then names what the case is driving, so a
      case still refuses to run with nothing to drive at all. No reply at all
      is a real, measured outcome in this shape, never an assertion failure —
      the trigger carries no instruction forcing one.

    Each sample is fully hermetic, exactly like ``chat_eval``: its own mock
    Signal server, DB, and real-model Penny."""
    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        changelog_entries: Sequence[ChangelogEntry],
        unrelated_turn: str = "",
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
        # This sample's own Penny construction (#2891): the real startup
        # path (`Penny._record_changelog_entries`) runs for every sample
        # regardless of `changelog_entries` above (which only drives the
        # DIRECT `_deliver_changelog` call below) — `Config.changelog_commits`
        # defaults to empty, so a case that never sets this never has its
        # sample's `changelog` log touched by anything but `_deliver_changelog`
        # itself. A case may pass its OWN list here (never the build's real
        # file) to keep that log non-empty and consistent with what it
        # delivers.
        changelog_commits: Sequence[ChangelogEntry] = (),
    ) -> Cohort:
        """Drive the delivery turn, then (with ``ask`` given) one later question
        in its wordings — or nothing — and return the cohort.  A case naming
        neither an ``ask`` nor a ``world`` is refused by name: a delivery-turn-
        alone case still has to say what it is driving."""
        given_to_drive: Sequence[object] | None = (
            ask if ask is not None else ([world] if world is not None else None)
        )
        _require_ask(
            case_id, "changelog_eval", "ask=<the later question> or world=<...>", given_to_drive
        )
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
            [ask or "", *also_phrased] if ask is not None else [""],
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
            before = collection_names(penny.db)
            held_before = _held_entries(penny.db)

            if ask is None:
                answer = await _deliver_changelog_unforced(
                    penny, changelog_entries, retryable=retryable
                )
                reply = final_reply = answer
                result = SampleResult.graded([])
                _stamp_cause(penny.db, result)
                _write_sample_report(penny.db, case_id, sample_index, result=result, reply=reply)
            else:
                await _deliver_changelog(penny, changelog_entries, retryable=retryable)
                turns = [turn for turn in (unrelated_turn, spoken[sample_index]) if turn]
                reply = ""
                final_reply = ""
                try:
                    messages = await _drive_turns(
                        penny, server, turns, timeout=timeout, retryable=retryable
                    )
                    reply = "\n".join(messages)
                    final_reply = messages[-1] if messages else ""
                    result = SampleResult.graded([])
                    _stamp_cause(penny.db, result)
                    _write_sample_report(
                        penny.db,
                        case_id,
                        sample_index,
                        result=result,
                        reply=final_reply,
                        driven=turns,
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
                final_reply=final_reply,
                before=before,
                held_before=held_before,
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

        committed_commits = list(changelog_commits)

        def _make_config_with_commits(**overrides: object) -> Config:
            return make_config(changelog_commits=list(committed_commits), **overrides)

        results, perf, voided = await _run_samples(
            _make_config_with_commits,
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
