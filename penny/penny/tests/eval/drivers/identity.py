"""The ``identity_eval`` driver: the collection write's entry-identity ladder alone."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import NamedTuple

import pytest

from penny.config import Config
from penny.database import Database
from penny.database.memory.identity import EntryDecider
from penny.llm.call_link import CallLink
from penny.llm.decision_client import (
    DecisionAnswer,
    DecisionClient,
    DecisionError,
    decision_api_url,
)
from penny.penny import Penny
from penny.tests.eval.conftest import (
    _UNSETTLED_BANNER,
    NO_DRAW,
    SAMPLES,
    SINGLE_CALL_WAIT_SECONDS,
    UNSTATED,
    Check,
    PassRate,
    SampleResult,
    _arms,
    _cause_word,
    _declare_case,
    _driven_cohort,
    _dump_thinking,
    _embed_seeds,
    _PendingCase,
    _record_sample_block,
    _require_ask,
    _require_report_only,
    _run_samples,
    _scored_counts,
    _stamp_cause,
    collection_entries,
    live_prompt_perf,
    sample_number,
    seed_user,
    seed_world_stores,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.baseline import baseline_from_env
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.fixtures import SYNTH_KEY_SEPARATOR, SynthCollection
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.tools.base import Tool
from penny.tools.memory_tools import build_memory_tools

IdentityEval = Callable[..., Awaitable["Cohort"]]


def _synth_entry(pair: tuple[str, str]) -> str:
    """One entry as a ``SynthCollection`` declares it: ``key — content``, the store's own
    shape (``SYNTH_KEY_SEPARATOR``)."""
    return f"{pair[0]}{SYNTH_KEY_SEPARATOR}{pair[1]}"


class IdentityArm(NamedTuple):
    """ONE arm of an identity cohort: the PAIR the ladder is asked about.

    The collection write's identity ladder takes no user words — its whole input is the
    collection's stored state and the new entry written over it, both authored upstream
    (the stored entries by the rounds that made them, the new entry by the writing draw).
    So the arm is the pair: a stored entry already in the collection, one unrelated filler
    entry beside it, and the new entry the driven write carries. The five wordings are five
    look-alike SHAPES — which is the coverage axis this ladder has, the collector's
    five-prose-variants shape with the pair standing where the page stands.

    The FACTS are fixed within an arm: the stored entry, the filler and the new entry are
    byte-identical across that arm's samples, which is what lets a claim name a value the
    store must carry. Every pair is one the deterministic rules do NOT settle — the
    rules-settled territory is unit-tested in ``tests/``; what reaches the decision model
    is what this ladder's cases measure.

    ``stored_anchor``/``filler_anchor``/``new_anchor`` are the smallest unique datum of
    each entry — what a claim reads the end state through, and what the coherence probe in
    the case file holds against the arm's own declared entries."""

    text: str
    collection: str
    description: str
    stored: tuple[str, str]
    filler: tuple[str, str]
    new: tuple[str, str]
    stored_anchor: str
    filler_anchor: str
    new_anchor: str
    # Which side of the pair the merge case expects to survive — "stored" (unchanged)
    # or "new" (the merge rewrote it) — and that side's anchor. Unused by the
    # keep-apart case, whose end state carries all three anchors; declared here so one
    # arm shape serves both and a claim reads the arm's own expectation.
    survivor: str
    survivor_anchor: str

    @property
    def world(self) -> World:
        """The seeded collection this arm is answered against, as a ``World`` — the
        stored entry and the filler, declared the way the store holds them."""
        return World(
            name=self.collection,
            pages=(),
            stores=(
                SynthCollection(
                    self.collection,
                    self.description,
                    (_synth_entry(self.stored), _synth_entry(self.filler)),
                ),
            ),
            keeps=(),
            excludes=(),
        )


_DECISION_CONTEXT = "entry-identity"

# The actor label every decision-model event carries, so the 🧩 rows read as the one
# sub-model that answered them.


class _RecordedAsk(NamedTuple):
    """One question the ladder put to the decision model, and what came back.

    Recorded by the wrapper client the driver hands the ladder — the runtime is
    unchanged, the recording sits here only — so the sample's render can show the state
    shown, the question asked, the choice made and every option's probability. A failed
    call records its error as the answer (the ladder degrades it to ``unsure``), which is
    the honest render of a question that got no answer."""

    state: str
    instructions: str
    criteria: dict[str, str]
    choice: str | None
    probabilities: dict[str, float]
    error: str | None


class _RecordingDecisionClient(DecisionClient):
    """The decision client the driver hands the ladder: the real client underneath, plus
    a recording of every question and answer for the sample's render."""

    def __init__(self, inner: DecisionClient) -> None:
        super().__init__(model=inner.model, api_url=inner.api_url, timeout=inner.timeout)
        self._inner = inner
        self.asked: list[_RecordedAsk] = []

    async def decide(
        self, *, state: str, instructions: str, criteria: dict[str, str]
    ) -> DecisionAnswer:
        try:
            answer = await self._inner.decide(
                state=state, instructions=instructions, criteria=criteria
            )
        except DecisionError as error:
            self.asked.append(
                _RecordedAsk(
                    state=state,
                    instructions=instructions,
                    criteria=criteria,
                    choice=None,
                    probabilities={},
                    error=str(error),
                )
            )
            raise
        self.asked.append(
            _RecordedAsk(
                state=state,
                instructions=instructions,
                criteria=criteria,
                choice=answer.choice,
                probabilities=answer.probabilities,
                error=None,
            )
        )
        return answer

    async def list_models(self) -> list[str]:
        return await self._inner.list_models()


def _collection_tool(penny: Penny) -> tuple[Tool, _RecordingDecisionClient]:
    """The real ``collection_write`` tool, built the way production builds the memory
    surface: through ``build_memory_tools``, with the agent's own embedding client and the
    decision step constructed from config, exactly as ``Agent.__init__`` does — its
    decision client wrapped by the driver's recording one."""
    if not penny.config.llm_decision_model:
        raise ValueError(
            "identity_eval measures the entry-identity ladder's decision step: set "
            "LLM_DECISION_MODEL. The no-decider degraded mode is a different behaviour "
            "(#2715's named fallback), not this case."
        )
    recording = _RecordingDecisionClient(
        DecisionClient(
            model=penny.config.llm_decision_model, api_url=decision_api_url(penny.config)
        )
    )
    tools = build_memory_tools(
        penny.db,
        penny.embedding_model_client,
        penny.chat_agent.name,
        decider=EntryDecider(recording),
        include_lifecycle=False,
    )
    return next(tool for tool in tools if tool.name == "collection_write"), recording


def _seeded_world_check(arm: IdentityArm, seeded: dict[str, str]) -> Check:
    """The loud probe of the world the seed just laid down: exactly the two declared
    entries. A seed that drifted fails HERE, before the driven write gives the ladder
    anything to decide about."""
    expected = {arm.stored[0]: _synth_entry(arm.stored), arm.filler[0]: _synth_entry(arm.filler)}
    ok = seeded == expected
    return Check(
        "seeded world held",
        ok,
        kind="guard",
        rationale=None
        if ok
        else f"the seed laid down {sorted(seeded)}, expected {sorted(expected)}",
    )


def _carrying_key(entries: dict[str, str], anchor: str) -> str | None:
    """The first key whose whole entry (key and content) carries ``anchor``, case-
    insensitively — the one an anchor-presence claim reads."""
    for key, content in entries.items():
        if anchor.lower() in f"{key} {content}".lower():
            return key
    return None


def _rows(collection: str, entries: dict[str, str]) -> list[eval_cohort.StoredEntry]:
    """A collection's entries as the observation carries them — the same three facts the
    store holds, in the one ``StoredEntry`` shape every claim reads."""
    return [
        eval_cohort.StoredEntry(collection=collection, key=key, content=content)
        for key, content in entries.items()
    ]


def _entry_lines(entries: dict[str, str]) -> str:
    """Entries as the render shows them: one ``key: content`` line each."""
    return "\n".join(f"{key}: {content}" for key, content in entries.items())


def _identity_events(
    arm: IdentityArm,
    seeded: dict[str, str],
    asked: list[_RecordedAsk],
    after: dict[str, str],
    tool_message: str | None,
) -> list[report.Event]:
    """One sample's transcript, hand-built — the inputs the ladder was given, each
    question it asked and what came back, the store it left, and what the tool said.

    The decision model draws nothing the promptlog carries (its calls ride Ollama's
    ``/v1/systemone`` directly), so this is the only place the questions survive — the
    recording wrapper captured them at ask time."""
    events = [
        report.Event(
            report.EventKind.USER,
            f"Collection '{arm.collection}': {arm.description}\n"
            f"Stored before the write:\n{_entry_lines(seeded)}",
        ),
        report.Event(
            report.EventKind.MICRO_IN,
            f"{arm.new[0]}: {arm.new[1]}",
            context=_DECISION_CONTEXT,
        ),
    ]
    for one in asked:
        options = "; ".join(f"{name} — {text}" for name, text in one.criteria.items())
        events.append(
            report.Event(
                report.EventKind.MICRO_IN,
                f"{one.state}\nquestion: {one.instructions}\noptions: {options}",
                context=_DECISION_CONTEXT,
            )
        )
        if one.error is not None:
            answer = f"(no answer — the call failed: {one.error})"
        else:
            answer = one.choice or "(empty choice)"
            if one.probabilities:
                spread = "; ".join(
                    f"{name} {value:.3f}" for name, value in one.probabilities.items()
                )
                answer = f"{answer} · probabilities: {spread}"
        events.append(report.Event(report.EventKind.MICRO_OUT, answer, context=_DECISION_CONTEXT))
    events.append(
        report.Event(report.EventKind.RESULT, f"Stored after the write:\n{_entry_lines(after)}")
    )
    events.append(
        report.Event(report.EventKind.REPLY, tool_message or "(no write result within timeout)")
    )
    return events


def _check_views(result: SampleResult, anchor_index: int, case_id: str) -> list[report.CheckView]:
    """The sample's checks resolved against the transcript (the extractor's shape): the
    seeded-world guard binds to the opener step, and a baseline flip resolves per
    ``(case_id, label)``."""
    baseline = baseline_from_env()
    views: list[report.CheckView] = []
    for index, check in enumerate(result.checks, start=1):
        regressed = (
            not check.ok
            and not check.ignored
            and baseline is not None
            and baseline.was_passing(case_id, check.label)
        )
        views.append(
            report.CheckView(
                check_id=f"C{index}",
                label=check.label,
                kind=check.kind,
                scored=check.scored,
                ignored=check.ignored,
                ok=check.ok,
                rationale=check.rationale,
                cause=_cause_word(result.cause) if not check.ok else None,
                anchor_index=anchor_index,
                regressed=regressed,
            )
        )
    return views


def _write_identity_report(
    db: Database,
    case_id: str,
    sample_index: int,
    *,
    result: SampleResult,
    arm: IdentityArm,
    seeded: dict[str, str],
    asked: list[_RecordedAsk],
    after: dict[str, str],
    tool_message: str | None,
) -> None:
    """One sample's transcript-integrated block, hand-built (the generic writer reads a
    chat-shaped transcript from the promptlog, and this sample's draw leaves none there —
    the store and the recording are the transcript), rendered by the SAME pure report
    grammar and appended to the same ``<case_id>.md``. No-op off-report."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir:
        return
    events = _identity_events(arm, seeded, asked, after, tool_message)
    checks = _check_views(result, anchor_index=0, case_id=case_id)
    passed_checks, total = _scored_counts(result)
    transcript = report.build_sample(
        number=sample_number(sample_index),
        banner=_UNSETTLED_BANNER,
        events=events,
        checks=checks,
        run_close_score=f"{passed_checks}/{total}",
    )
    _record_sample_block(case_id, sample_index, transcript, result, live_prompt_perf(db))


def _observe_write(
    db: Database,
    arm: IdentityArm,
    *,
    name: str,
    phrasing: str,
    arm_index: int,
    seeded: dict[str, str],
    asked: list[_RecordedAsk],
    after: dict[str, str],
    tool_message: str | None,
) -> eval_cohort.SampleObservation:
    """What ONE driven write left behind, read while its database is still live — the
    collection's entries as the write found and left them. This path's own observer: no
    machine walk, no run record and no chat reply exists to read, so a chat-shaped
    observation would be structurally fine and substantively hollow. The sample's
    structured OUTPUT is the write's own account — single-line fields, what the
    representative table and the case's coverage cell read."""
    answered = sum(1 for one in asked if one.error is None)
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm_index,
        held=_rows(arm.collection, after),
        held_before=_rows(arm.collection, seeded),
        output=[
            eval_cohort.OutputField(
                name="collection", value=f"{arm.collection} — {arm.description}"
            ),
            eval_cohort.OutputField(name="the write", value=f"{arm.new[0]}: {arm.new[1]}"),
            eval_cohort.OutputField(
                name="decision questions",
                value=f"{len(asked)} asked, {answered} answered",
            ),
            eval_cohort.OutputField(name="stored after", value=", ".join(after.keys())),
            eval_cohort.OutputField(
                name="tool reply", value=tool_message or "(no write result within timeout)"
            ),
        ],
    )


# The one measured axis, and the only one this shape needs: how many entries the
# collection holds when the write is done. It is arm-INDEPENDENT — the merge case expects
# the pair's survivor and the filler, the keep-apart case all three — so the pooled
# entropy measures the ladder's stability rather than the arms' coverage, and a divergence
# is a different end state (consequential).
ENTRY_COUNT = eval_cohort.Feature(
    "entries in the collection",
    lambda o: str(len(o.held)),
    consequence=eval_cohort.Consequence.CONSEQUENTIAL,
)


@pytest.fixture
def identity_eval(make_config: Callable[..., Config], tmp_path, request) -> Iterator[IdentityEval]:
    """Drive the collection write's entry-identity ladder N times, and NOTHING else.

    Each sample is hermetic: its own database and its own real-model Penny. The driver
    seeds the arm's collection through the store's OWN write path (``seed_collection`` →
    ``write_rule_decided``, the manual rules-only path — the seed must not put a decision
    model in the loop), backfills its embeddings so the shortlist ranks what production
    would rank, probes the seeded world loudly, then drives ONE real ``collection_write``
    call — built through ``build_memory_tools`` with the real decision client from config —
    and reads the store it left behind.

    The chat model draws nothing on this path: what is measured is the ladder — the rules,
    the embedding shortlist and the decision model's answers — which is why the driver
    refuses to run without ``LLM_DECISION_MODEL`` set.
    """

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        behaviour: str = "",
        area: Area | None = None,
        arms: Sequence[IdentityArm] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        family: str | None = None,
        timeout: float = SINGLE_CALL_WAIT_SECONDS,
    ) -> Cohort:
        """Drive one identity pair per arm and return the case's COHORT.

        A case with no ``arms`` is refused by name, as every driver refuses its ask."""
        _require_ask(case_id, "identity_eval", "arms=[IdentityArm(...), …]", arms)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.MICRO_CONTEXT,
            behaviour=behaviour,
            area=area,
            wording=arms[0].text,
        )
        if not os.environ.get("LLM_DECISION_MODEL"):
            raise ValueError(
                f"{case_id}: identity_eval measures the entry-identity ladder's decision "
                "step — set LLM_DECISION_MODEL for this run (the degraded no-decider mode "
                "is a different behaviour, not this case)."
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

        async def _drive(
            penny: Penny, server: MockSignalServer, sample_index: int, retryable: bool
        ) -> SampleResult:
            arm = arms[driving.index_of(sample_index)]
            phrasing = driving.label(sample_index)
            arm_index = driving.index_of(sample_index)
            seed_user(penny.db)
            seed_world_stores(penny.db, arm.world)
            await _embed_seeds(penny)
            seeded = collection_entries(penny.db, arm.collection)
            seeded_check = _seeded_world_check(arm, seeded)
            if not seeded_check.ok:
                # A drifted seed is a harness defect, not a sample's behaviour — void it
                # rather than grade a ladder decision made against a world nobody declared.
                raise ValueError(
                    f"{case_id} sample {sample_number(sample_index)}: {seeded_check.rationale}"
                )
            tool, recording = _collection_tool(penny)
            try:
                write_result = await asyncio.wait_for(
                    tool.run(
                        call_link=CallLink.new_top(),
                        collection=arm.collection,
                        entries=[{"key": arm.new[0], "content": arm.new[1]}],
                    ),
                    timeout=timeout,
                )
                tool_message: str | None = write_result.message
            except TimeoutError:
                tool_message = None
                result = SampleResult.binary(["no write result within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
                _write_identity_report(
                    penny.db,
                    case_id,
                    sample_index,
                    result=result,
                    arm=arm,
                    seeded=seeded,
                    asked=recording.asked,
                    after=collection_entries(penny.db, arm.collection),
                    tool_message=tool_message,
                )
                result.observation = eval_cohort.SampleObservation(
                    name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                    phrasing=phrasing,
                    arm=arm_index,
                    complete=False,
                    exclusion=NO_DRAW,
                )
                _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
                return result
            after = collection_entries(penny.db, arm.collection)
            # The sample is graded from its cohort's CLAIMS, made once every sample has
            # run, so it scores nothing at drive time and carries only its guard.
            result = SampleResult.graded([seeded_check])
            _stamp_cause(penny.db, result)
            _write_identity_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                arm=arm,
                seeded=seeded,
                asked=recording.asked,
                after=after,
                tool_message=tool_message,
            )
            result.observation = _observe_write(
                penny.db,
                arm,
                name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                phrasing=phrasing,
                arm_index=arm_index,
                seeded=seeded,
                asked=recording.asked,
                after=after,
                tool_message=tool_message,
            )
            _dump_thinking(penny.db, case_id, sample_index, failed=not result.passed)
            return result

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
    # The case's claims are made in the TEST BODY, after the drive returns — so the report
    # is assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
