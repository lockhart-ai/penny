"""The ``extractor_eval`` driver: the browse extraction micro-context alone."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import NamedTuple

import pytest

from penny.config import Config
from penny.constants import PageReading, PennyConstants
from penny.database import Database
from penny.database.models import PromptLog
from penny.llm.call_link import CallLink
from penny.penny import Penny
from penny.tests.eval.conftest import (
    _UNSETTLED_BANNER,
    NO_DRAW,
    SAMPLES,
    SINGLE_CALL_WAIT_SECONDS,
    UNSTATED,
    PassRate,
    SampleResult,
    _arms,
    _cause_word,
    _declare_case,
    _draw_exclusion,
    _driven_cohort,
    _dump_thinking,
    _micro_context_rows,
    _PendingCase,
    _record_case_prompts,
    _record_sample_block,
    _require_ask,
    _require_report_only,
    _response_text,
    _run_samples,
    _scored_counts,
    _stamp_cause,
    _system_prompts,
    given_to_the_model,
    live_prompt_perf,
    sample_number,
)
from penny.tests.eval.utils import artifacts as eval_artifacts
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.baseline import Baseline, baseline_from_env
from penny.tests.eval.utils.catalogue import Area, Layer
from penny.tests.eval.utils.fixtures import CannedPage
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.tools.link_handles import HandleTable
from penny.tools.micro_context import MicroContext, MicroContextResult, MicroExtractOutcome

# ── Browse EXTRACTION: what a page half-carries (#1942) ───────────────────────

ExtractorEval = Callable[..., Awaitable["Cohort"]]


class FieldExpectation(NamedTuple):
    """One thing an extract instruction asks for, and how the page answers it (#1942).

    ``anchor`` is the span of the page that supplies it, compared through the SHIPPED
    ``spoken_form`` — so a value passes whether or not the draw kept the punctuation or
    the article in front of it.  Deliberately not an equality: which words carry a fact
    has a little play in it, and a claim demanding one exact string would be answering
    for the draw.

    An EMPTY anchor is the ABSENT direction: the page supplies nothing for this, so it is
    named here to say what the instruction asked for and the page did not have."""

    field: str
    anchor: str = ""


def _classifier_events(phrasing: str, rows: list[PromptLog]) -> list[report.Event]:
    """The hand-built event stream for one single-draw micro-context sample: the
    phrasing opens the step, then one 🧩 in/out pair PER DRAW — a reroll renders as
    a second pair, so recovery is visible in the transcript, never summarized away.
    Each pair is labelled with the drawing context's ledger identity (#1773), the
    same actor label the chat-run extractor emits."""
    events = [report.Event(report.EventKind.USER, phrasing)]
    for row in rows:
        messages = json.loads(row.messages) if row.messages else []
        user = next(
            (m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), ""
        )
        context = row.agent_name or ""
        events.append(report.Event(report.EventKind.MICRO_IN, user, context=context))
        events.append(
            report.Event(
                report.EventKind.MICRO_OUT,
                _response_text(row) or "(empty)",
                thinking=row.thinking or "",
                context=context,
            )
        )
    return events


def _classifier_check_views(
    result: SampleResult, anchor_index: int, baseline: Baseline | None, case_id: str
) -> list[report.CheckView]:
    """Every check anchored to the FINAL draw's 🧩→ row (the decision), with the
    baseline flip resolved per ``(case_id, label)`` exactly like the extractor's."""
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


def _write_classifier_report(
    db: Database,
    case_id: str,
    sample_index: int,
    *,
    result: SampleResult,
    phrasing: str,
    agent_names: Sequence[str],
) -> None:
    """One micro-context sample's transcript block — hand-built (the
    generic extractor is chat-run-shaped; this sample is one step whose actor is the
    🧩 micro-context, the spec's official sub-model actor), rendered by the SAME pure
    report grammar and appended to the same ``<case_id>.md``. No-op off-report.
    ``agent_names`` selects the customers' rows, rendered in one ledger-ordered
    stream — each 🧩 pair carries its own context label, so two customers read as the
    two actors they are.

    The sample's system prompts are deposited for the case document the same way the
    chat path deposits them (#2062), off the SAME rows the transcript renders — so every
    prompt the document states belongs to an actor the reader can see acting, and a
    microcontext failure carries the state the draw was given rather than sending the
    reader to the sample database for it."""
    report_dir = os.environ.get("EVAL_REPORT_DIR")
    if not report_dir:
        return
    rows = _micro_context_rows(db, *agent_names)
    if not rows:
        transcript = report.SampleTranscript(
            sample_number(sample_index),
            _UNSETTLED_BANNER,
            [],
            placeholder=report.NO_TURNS_PLACEHOLDER,
        )
    else:
        _record_case_prompts(case_id, sample_index, _system_prompts(rows))
        events = _classifier_events(phrasing, rows)
        checks = _classifier_check_views(result, len(events) - 1, baseline_from_env(), case_id)
        passed_checks, total = _scored_counts(result)
        transcript = report.build_sample(
            number=sample_number(sample_index),
            banner=_UNSETTLED_BANNER,
            events=events,
            checks=checks,
            run_close_score=f"{passed_checks}/{total}",
        )
    _record_sample_block(case_id, sample_index, transcript, result, live_prompt_perf(db))


def _extract_rerolled(db: Database) -> bool:
    """Did the extraction draw more than once — the fragile signal, a sample that only
    got there by recovering.  One customer here, so one row is the clean case."""
    return len(_micro_context_rows(db, PennyConstants.BROWSE_EXTRACT_AGENT_NAME)) > 1


# The fields of a browse extraction's structured answer, named once.  A case asserts them
# and measures them by these names, so the two halves of #2017's framing — deterministic and
# variance — read the same value rather than two spellings of it.
#
# Two fields (#2615): the tagged shape retired its own ``reason``/``missing`` split —
# what the page has for each thing asked, any shortfall, any "none of it" are all said
# inside the one plain-text ``value`` now, so there is nothing left to carry separately.
EXTRACT_OUTCOME = "outcome"
EXTRACT_VALUE = "value"

# The one outcome that means the draw never produced a usable answer at all (#2615: the
# tagged shape's separate "clean but untagged" failure is gone with it, so there is only
# one way left to land here).  A sample landing on it is EXCLUDED rather than scored: it
# is the micro-context twin of a chat turn that produced no reply, and grading it would
# report a failed contract for a contract nobody exercised.
_EXTRACT_FAILURES = frozenset({MicroExtractOutcome.EXTRACTION_FAILED})


def _extraction_output(
    result: MicroContextResult, handles: HandleTable
) -> list[eval_cohort.OutputField]:
    """One extraction draw's STRUCTURED ANSWER, field by field.

    Both fields, always — including the empty one on a failure — so the cohort's field
    comparison reads the same set of axes per sample whatever the outcome.  The value is
    expanded through the case's own link table (#2717) before it is observed: the draw
    answers in handles (the page arrives as them), the user-visible fact is the URL behind
    each one, and the claims read that fact — so the field holds the expanded form."""
    return [
        eval_cohort.OutputField(name=EXTRACT_OUTCOME, value=result.outcome.value),
        eval_cohort.OutputField(name=EXTRACT_VALUE, value=handles.expand(result.value)),
    ]


def _observe_extraction(
    db: Database,
    result: MicroContextResult | None,
    *,
    name: str,
    phrasing: str,
    arm: int,
    handles: HandleTable,
) -> eval_cohort.SampleObservation:
    """Read what ONE extraction sample answered — its own observer, not the chat one.

    A single-call context leaves no machine walk, no registry row, no collection entry and no
    outgoing message, so every read ``_observe_sample`` makes would come back empty here and
    the row would be structurally fine and substantively hollow.  What there IS to read is the
    draw's typed result, and it is read directly rather than fished back out of the ledger.

    ``given`` is the draw's own prompt, read through ``given_to_the_model`` (#2078) — the page
    and the instruction under the extraction contract, so a provenance claim compares the
    answer against the whole of what the draw was handed, a plain "none of it" answer
    included.  It is expanded through the case's own link table (#2717), the same one the
    document was rendered with, so the answer's expanded URLs trace to the page as given."""
    answer = "" if result is None or result.outcome in _EXTRACT_FAILURES else result.outcome.value
    exclusion = _draw_exclusion(db, answer)
    if result is None or exclusion is not None:
        return eval_cohort.SampleObservation(
            name=name, phrasing=phrasing, arm=arm, complete=False, exclusion=exclusion or NO_DRAW
        )
    return eval_cohort.SampleObservation(
        name=name,
        phrasing=phrasing,
        arm=arm,
        given=given_to_the_model(db, handles),
        output=_extraction_output(result, handles),
    )


def _extraction_world(url: str, page: str, reading: PageReading) -> World:
    """The one page an extraction case is answered against, as a :class:`World`.

    Built from the case's own ``url``/``page``/``reading`` rather than declared separately, so
    there is one source of truth for what the draw read and the report's world fold renders
    exactly the document the draw was handed.  ``keeps``/``excludes`` are empty: those token
    sets exist for claims about what a ROUND chose to store, and this shape stores nothing."""
    return World(
        name=url, pages=(CannedPage(match=url, text=page, reading=reading),), keeps=(), excludes=()
    )


@pytest.fixture
def extractor_eval(
    make_config: Callable[..., Config], tmp_path, request
) -> Iterator[ExtractorEval]:
    """Drive the browse EXTRACTION micro-context (#1588/#1942) N times, and NOTHING else.

    The extraction's whole input is one fetched page's section and one instruction, so
    the case is those two things — no browse runs, because what is measured is what the
    extractor makes of a page rather than whether the page can be fetched.  The section is
    built the way ``BrowseTool._page_section`` builds it, so the draw reads the document
    production hands it.

    ``MicroContext.extract`` makes the call, so the case exercises the shipped prompt, the
    shipped parse and the shipped reroll budget.  Synthetic here means the PAGE and the
    INSTRUCTION are authored, never the prompt.
    """

    _cohorts: dict[str, _PendingCase] = {}

    async def _run(
        *,
        case_id: str,
        url: str,
        page: str,
        instruction: str,
        behaviour: str = "",
        area: Area | None = None,
        also_instructed: Sequence[str] = (),
        samples_per_phrasing: int = 0,
        model: str = "",
        samples: int = SAMPLES,
        min_pass_rate: PassRate = UNSTATED,
        timeout: float = SINGLE_CALL_WAIT_SECONDS,
        family: str | None = None,
        reading: PageReading = PageReading.ARTICLE,
    ) -> Cohort:
        """Drive one page + instruction through the extraction draw and return its COHORT.

        A case passes ``instruction`` and ``also_instructed`` — the other wordings of the SAME
        request — and gets a :class:`Cohort` back to assert against.  The arm here is the
        ``extract`` instruction, which is PENNY's own text: production writes it at the browse
        call site, so what the cohort measures is whether the answer survives how the calling
        draw happened to word the request.  The page is FIXED across the arms; it is the world.
        A case with no ``instruction`` is refused by name.

        ``reading`` (#2388) declares how the extension would have read ``page`` — ARTICLE (the
        default) or LINK_LIST — so a case presents the extractor's structural note the same
        way production does: the note rides ``MicroContext.extract``'s own ``reading`` argument,
        never guessed here from the page's shape.
        """
        _require_ask(case_id, "extractor_eval", "instruction=<the extract request>", instruction)
        _require_report_only(case_id, min_pass_rate)
        catalogued = _declare_case(
            request,
            case_id=case_id,
            layer=Layer.MICRO_CONTEXT,
            behaviour=behaviour,
            area=area,
            wording=instruction,
        )
        eval_artifacts.begin_case(case_id)
        # One page, K wordings of the instruction — the same shape chat has, and the world is
        # a property of the CASE rather than of the arm.
        arms = _arms(
            [instruction, *also_instructed],
            [_extraction_world(url, page, reading)],
            samples_per_phrasing,
            samples,
        )
        spoken = arms.spoken
        # The document production hands a draw: built through the run's link table
        # (#2717) — the section header and every link in the page arrive as
        # handles — and expanded back through the SAME table when the answer is
        # observed, so a claim's URL anchor traces to the page as given.
        handles = HandleTable()
        content = handles.convert_browse_section(
            f"{PennyConstants.BROWSE_PAGE_HEADER}{url}\n{page}"
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
            asked = spoken[sample_index]
            micro = MicroContext(penny.model_client, db=penny.db)
            extracted: MicroContextResult | None = None
            try:
                extracted = await asyncio.wait_for(
                    micro.extract(
                        content,
                        asked,
                        call_link=CallLink.new_top(),
                        run_target=penny.chat_agent.name,
                        reading=reading,
                    ),
                    timeout=timeout,
                )
                # The case is graded from its cohort's CLAIMS, made after every sample has
                # run, so the sample itself scores nothing at drive time.
                result = SampleResult.graded([])
                result.observe_rerolled(rerolled=_extract_rerolled(penny.db))
                _stamp_cause(penny.db, result)
            except TimeoutError:
                result = SampleResult.binary(["no extraction draw within timeout"])
                _stamp_cause(penny.db, result, timed_out=True)
            phrasing = arms.label(sample_index)
            result.observation = _observe_extraction(
                penny.db,
                extracted,
                name=f"{case_id}-{sample_number(sample_index)} ({phrasing})",
                phrasing=phrasing,
                arm=arms.index_of(sample_index),
                handles=handles,
            )
            _write_classifier_report(
                penny.db,
                case_id,
                sample_index,
                result=result,
                phrasing=asked,
                agent_names=(PennyConstants.BROWSE_EXTRACT_AGENT_NAME,),
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
    # The case's claims are made in the TEST BODY, after the drive returns — so the report is
    # assembled here, once the body has had its say.
    for pending in _cohorts.values():
        pending.finish()
    _cohorts.clear()
