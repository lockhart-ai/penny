"""Deterministic tests for the eval-harness scoring + report machinery (issue #1694).

These drive the ``tests/eval/conftest.py`` scoring/report code directly with fixture
``Check`` / ``SampleResult`` data and a seeded promptlog — no live model, no ``eval`` marker —
so they run inside ``make check`` and pin the new ergonomics: check ``rationale``, the
not-applicable (``ignored``) third state, the fragile-pass verdict, the dual strict+partial
RESULT line, and the ``tool_not_called`` negative-constraint primitive.  Whole-render literal
assertions cover every new report shape.

The labeller runner's learn → render step (#1770/#1782/#1828) is pinned here too — the
fixture ledger through the SHIPPED distiller and renderer, and the five agreed cases' input
documents byte-for-byte — so a runner helper that calls into machinery a later change
removed, or a fixture that drifts from the pair its case claims, fails inside ``make
check`` rather than only on the ``eval``-marked run the marker deselects.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pytest

# Importing the memory-tools module registers those tools (``Tool.__init_subclass__``) so
# ``Tool.format_result`` dispatches their real ``to_result_narration`` — the rejection-probe
# tests below build frames from the PRODUCTION templates, never hand-invented text.
import penny.tools.memory_tools  # noqa: F401  (imported for registration side effect)
from penny.agents.collector import Collector
from penny.constants import PennyConstants, RunOutcome
from penny.conversation_machine import RoundShortfall
from penny.database import Database
from penny.database.memory import MemoryType
from penny.database.models import MemoryRow, Skill
from penny.database.skills import (
    SkillParameter,
    build_binding_content,
    render_spoken_turns,
    slug_skill_name,
)
from penny.llm.models import LlmMessage, LlmToolCall, LlmToolCallFunction
from penny.notification import NOTIFICATION_NOTES, NotificationOutcome
from penny.skill_extraction import build_framing_content
from penny.tests import eval as eval_package
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.binder.test_skill_binding import FIXTURES as BINDING_FIXTURES
from penny.tests.eval.chat.apply.test_known_routine_new_space import (
    IDLE_APPLY_CASES,
    assert_new_space_is_unknown,
)
from penny.tests.eval.chat.apply.test_missing_value_arrives import _names_the_cadence_check
from penny.tests.eval.chat.idle.test_bracket_key_recovery import (
    BRACKET_KEY_CASES,
    _seed_board_games,
    assert_board_games_world,
)
from penny.tests.eval.chat.idle.test_chat_reply import (
    _STORED_TITLES,
    _carries,
    _honest_about_the_duplicate,
)
from penny.tests.eval.chat.idle.test_choose_dispatch import (
    _OPTIONS as _CHOOSE_OPTIONS,
)
from penny.tests.eval.chat.idle.test_choose_dispatch import (
    CHOOSE_CASES,
    _gave_an_opinion_check,
    _reply_reports,
    assert_choose_world,
)
from penny.tests.eval.chat.idle.test_command_tools import (
    IMAGE_CASES,
    _claims_no_picture_check,
    assert_image_world,
    install_image_client,
)
from penny.tests.eval.chat.idle.test_email_dispatch import (
    EMAIL_CASES,
    _claims_no_search_check,
    assert_mailbox_world,
    install_mailbox,
)
from penny.tests.eval.chat.idle.test_round_ends_in_idle import (
    BAIL_CASES,
    _claims_no_job_check,
    assert_the_round_built_what_it_claims,
)
from penny.tests.eval.chat.learn.test_correction_re_runs_the_round import (
    CORRECTION_CASES,
    SHAPE_DELTA_WITHOUT_RE_RUNNING,
    SHAPE_RE_RAN_AND_APPLIED,
    _correction_shape,
    assert_the_correction_is_unsaid,
    assert_the_teach_round_is_parked,
    seed_corrected_round,
)
from penny.tests.eval.chat.learn.test_teach_arrives_whole import (
    assert_the_teach_is_new_to_the_world,
)
from penny.tests.eval.chat.request.test_ask_is_one_value_short import (
    _asks_for_what_is_missing_check,
    _does_not_re_ask_check,
)
from penny.tests.eval.collector.test_collector_enactment import (
    _STOP_REASON as _STOP,
)
from penny.tests.eval.collector.test_collector_enactment import (
    DIRECTION_CHECK_LABEL,
    ENACTMENT_CASES,
    GATE_CASES,
    _assert_the_baseline_is_stored,
    _EnactmentCase,
    _score_enactment,
    assert_applied_world,
    configured_terms,
    rendered_program,
    seed_applied_job,
    seed_gate_world,
)
from penny.tests.eval.conftest import (
    PENNY_LOGGER,
    BoundExpectation,
    Check,
    CycleCall,
    CycleObservation,
    FieldExpectation,
    ParameterFamily,
    SampleResult,
    _assert_threshold,
    _bail_fired_check,
    _case_prompts,
    _cycle_recovered_check,
    _flush_sample_blocks,
    _frame_attributes_to,
    _guarded_graded,
    _labelling_input,
    _score_binding,
    _score_extraction,
    _score_framing,
    _score_labelling,
    _scorer_is_graded,
    _stamp_cause,
    _without_examples,
    _write_sample_report,
    continue_nudge_fired,
    count_tool_calls,
    draw_rerolled,
    env_seconds,
    is_seeded_run,
    live_prompt_perf,
    routing_clean,
    run_exhibited_pathology,
    sample_is_fragile,
    sample_log_path,
    sample_logging,
    seeded_run_id,
    tool_call_rejected,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.extractor.test_browse_extract_fields import FIXTURES as EXTRACT_FIELD_FIXTURES
from penny.tests.eval.framer.test_skill_framing import FIXTURES as FRAMING_FIXTURES
from penny.tests.eval.labeller.test_skill_labelling import FIXTURES as LABELLING_FIXTURES
from penny.tests.eval.utils import report
from penny.tests.eval.utils.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckOutcome,
    FailureCause,
)
from penny.tests.eval.utils.baseline import load_baseline
from penny.tests.eval.utils.dispatch_world import assert_no_collections, collection_names
from penny.tests.eval.utils.fixtures import BOARD_GAMES
from penny.tests.eval.utils.transition_world import (
    APPLY_CASES,
    IDLE_LEARN_CASES,
    IDLE_REQUEST_CASES,
    JOURNEY_CONFIRMATIONS,
    LAST_SPOKEN_TURNS,
    REQUEST_APPLY_CASES,
    _interface_check,
    _overlaps,
    _round_reported_checks,
    _said_back,
    assert_composed_world,
    assert_parked_in_request_world,
    assert_round_cites_its_run,
    assert_round_is_framed,
    assert_seeded_ledger,
    assert_values_are_new,
    cadence_seconds,
    parked_binding,
    rule_parts,
    seed_composed_world,
    seed_learned_round,
    seed_parked_in_request,
)
from penny.tests.schema_template import migrated_db, schema_only_db
from penny.tools.base import FRAMEWORK_NARRATION_INVALID_ARGS, Tool
from penny.tools.collection_instantiation import _LINE_ESCAPE
from penny.tools.micro_context import (
    BoundValues,
    FramedParameter,
    LeafLabel,
    MicroContextResult,
    MicroExtractOutcome,
    MissingParameters,
    SkillLabels,
    SkillSignature,
    spoken_form,
)
from penny.tools.models import ToolResult

# The empty-response continue nudge, verbatim.  It is a LITERAL rather than a constant
# because #1937 deleted ``Prompt.CONTINUE_NUDGE`` with the validator that appended it —
# an empty draw is discarded and re-rolled now, so no new run can carry this frame.  The
# harness still has to READ one: these rows are historical promptlog rows, and the
# fragility / nudge-frame probes must keep recognising them (the same legacy-leg
# discipline the eval conftest's retired bail markers keep).
_RETIRED_CONTINUE_NUDGE = "Please provide your response."


def test_a_forwarded_but_unset_duration_reads_as_its_default(monkeypatch) -> None:
    """`make eval` forwards every variable, so an unset one arrives EMPTY, not absent.

    Read with a plain default that empty string reaches `float("")`, which raises while
    the module is still being imported — the eval suite then dies before any sample runs,
    with a traceback about a conftest rather than about the run.
    """
    monkeypatch.setenv("SOME_DURATION", "")
    assert env_seconds("SOME_DURATION", 20.0) == 20.0
    monkeypatch.delenv("SOME_DURATION")
    assert env_seconds("SOME_DURATION", 20.0) == 20.0
    monkeypatch.setenv("SOME_DURATION", "45")
    assert env_seconds("SOME_DURATION", 20.0) == 45.0


async def test_concurrent_samples_keep_their_logs_apart(tmp_path) -> None:
    """Two samples in flight must not write into each other's log.

    The ``penny`` logger is process-global, so concurrent samples attach concurrent
    handlers — and before the records carried a sample identity, every open log took
    every sample's lines. Measured on a 2-worker run, one sample's log held ten agent
    shutdowns instead of one, which is what made a stalled sample undiagnosable from it.
    """
    logger = logging.getLogger(PENNY_LOGGER)

    async def run_sample(name: str) -> Path:
        with sample_logging(str(tmp_path / f"{name}.db")) as path:
            logger.info("start %s", name)
            await asyncio.sleep(0)  # yield, so the two samples interleave for real
            logger.info("end %s", name)
            return path

    first, second = await asyncio.gather(run_sample("alpha"), run_sample("beta"))
    assert first.read_text().count("alpha") == 2
    assert "beta" not in first.read_text()
    assert second.read_text().count("beta") == 2
    assert "alpha" not in second.read_text()


def _sample_report_text(directory, case_id: str) -> str:
    """A case's report file, read the way a runner produces it.

    A sample's block is HELD when it is written and the file is laid down when the case
    closes, so the two steps stay together here rather than in each test.
    """
    _flush_sample_blocks(case_id)
    return (directory / f"{case_id}.md").read_text()


def _make_db(tmp_path, name: str = "harness") -> Database:
    db = schema_only_db(str(tmp_path / f"{name}.db"))
    return db


def _log_prompt(
    db: Database,
    *,
    messages=None,
    response=None,
    thinking=None,
    agent_name=None,
    run_id: str = "r1",
) -> None:
    db.messages.log_prompt(
        model="test-model",
        messages=messages if messages is not None else [{"role": "user", "content": "hi"}],
        response=response if response is not None else {},
        thinking=thinking,
        agent_name=agent_name,
        run_id=run_id,
    )


def _tool_call_response(name: str, arguments: str = "{}") -> dict:
    call = {"function": {"name": name, "arguments": arguments}}
    return {"choices": [{"message": {"tool_calls": [call]}}]}


def _content_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _tool_frame(content: str) -> list[dict]:
    return [{"role": "tool", "content": content}]


def _framed_result(
    tool_name: str,
    arguments: dict,
    *,
    ok: bool,
    mutated: bool = False,
    narration: str | None = None,
) -> list[dict]:
    """A tool-role turn carrying the REAL production frame ``Tool.format_result`` emits for a
    call to ``tool_name`` — the registry-dispatched narration + the ``(<tool> result)`` tag +
    body — so the rejection probe is tested against the shapes it must actually recognise,
    never hand-invented text (#1726)."""
    result = ToolResult(message="body", success=ok, mutated=mutated, narration=narration)
    return _tool_frame(Tool.format_result(tool_name, arguments, result))


# ── Scoring: the not-applicable (ignored) third state + rationale in failed labels ──


def test_graded_excludes_ignored_and_advisory_from_denominator() -> None:
    result = SampleResult.graded(
        [
            Check("state written", ok=True),
            Check("read count", ok=False),
            Check("routing clean", ok=False, scored=False),  # advisory: renders, doesn't count
            Check.na("browse branch", rationale="no browse this sample"),  # n/a: out of denom
        ]
    )
    assert result.total == 2  # only the two scored, applicable checks
    assert result.score == 0.5  # 1 of 2 scored checks passed — advisory doesn't move it
    assert not result.passed
    # An applicable failed check lands in ``failed`` whether or not it's scored (advisory
    # "routing clean" included); a not-applicable check never does.
    assert result.failed == ["read count", "routing clean"]
    assert len(result.checks) == 4  # every check preserved for the report


def test_graded_all_ignored_is_vacuous_pass() -> None:
    result = SampleResult.graded([Check.na("branch a"), Check.na("branch b")])
    assert result.total == 0
    assert result.score == 1.0
    assert result.passed
    assert result.failed == []


def test_graded_failed_label_carries_rationale() -> None:
    result = SampleResult.graded([Check("reads", ok=False, rationale="expected 3 reads, saw 1")])
    assert result.failed == ["reads — expected 3 reads, saw 1"]


def test_check_na_constructor() -> None:
    check = Check.na("browse branch", rationale="not exercised")
    assert check.ignored
    assert check.rationale == "not exercised"
    assert check.ok  # n/a is not a failure


# ── The negative-constraint primitive + the fragility scan ──


def test_tool_not_called_reads_the_promptlog(tmp_path) -> None:
    db = _make_db(tmp_path)
    _log_prompt(db, response=_tool_call_response("collection_write"))
    assert tool_was_called(db, "collection_write")
    assert not tool_not_called(db, "collection_write")
    assert tool_not_called(db, "send_message")


def test_a_sample_s_penny_log_lands_beside_its_db(tmp_path) -> None:
    """The per-sample log file (#1909): every line the penny loggers emit while a sample
    runs is written beside that sample's DB, under the same stem.

    A failing model call writes no promptlog row — it raises before the client's persist
    step — so what the run said about the failure exists only as logger output, which
    pytest captures and then discards for every sample that passes.  Unfiltered by
    design: a DEBUG line lands as readily as the WARNING the client logs when a chat
    call fails.  Scoped to one sample, so the handler and the raised level are both gone
    afterwards and nothing leaks into the next sample or the rest of the suite."""
    db_path = str(tmp_path / "watch-a-page-0.db")
    logger = logging.getLogger("penny.agents.base")
    level_before = logging.getLogger("penny").level
    handlers_before = list(logging.getLogger("penny").handlers)

    with sample_logging(db_path) as path:
        logger.warning("LLM chat failed: Connection refused")
        logger.debug("Tool parse error: raw='...'")

    assert path == sample_log_path(db_path) == tmp_path / "watch-a-page-0.log"
    captured = path.read_text()
    assert "LLM chat failed: Connection refused" in captured
    assert "Tool parse error: raw='...'" in captured
    # Scoped: the handler is gone and the level restored, so a line logged after the
    # sample cannot land in its file.
    logger.warning("after the sample")
    assert "after the sample" not in path.read_text()
    assert logging.getLogger("penny").level == level_before
    assert logging.getLogger("penny").handlers == handlers_before


def test_every_runner_stands_its_sample_up_through_the_logging_seam() -> None:
    """``eval_penny`` is the ONLY caller of ``run_penny_with_server`` in the eval harness
    (#1909), so every runner's samples get their penny log written beside their DB.

    The seam's whole claim is that it "covers the runners that exist and the ones added
    later", and nothing held it: a runner that stands its sample up directly is
    byte-identical in every other respect, its samples pass exactly as before, and the
    only symptom is a log file that is silently never written — for the runner whose
    samples the log was needed on.  Measured: the multi-cycle collector runner was added
    before the seam existed and the rebase left it bypassing it, which is the case this
    reads.  Structural, off the module's own AST, so a runner added tomorrow is covered
    without anybody remembering this rule.

    Read as a FILE rather than through an import: a ``conftest.py`` imported as an
    ordinary module is a second copy of every fixture it defines, which is a pytest
    collection error in ~700 tests, not a readable failure."""
    tree = ast.parse(_EVAL_CONFTEST.read_text())
    callers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Name) and inner.id == _SAMPLE_SEAM
    }
    assert callers == {"eval_penny"}, (
        f"only eval_penny may stand a sample up — {sorted(callers - {'eval_penny'})} "
        f"call {_SAMPLE_SEAM} directly, so their samples write no per-sample log"
    )


# The construction helper every eval sample must reach THROUGH ``eval_penny`` rather than
# directly, named once since the pin above reads it by name, and the harness file the pin
# parses — located off the package the runners live in, never off a string path.
_SAMPLE_SEAM = "run_penny_with_server"
_EVAL_CONFTEST = Path(str(eval_package.__file__)).parent / "conftest.py"


def test_a_cadence_is_read_from_the_rule_not_from_its_spelling() -> None:
    """The learn → apply cases score the CADENCE the acceptance asked for, never the rule
    spelling that says so (#1857) — so every rule that fires at the asked-for interval
    passes, whatever FREQ/INTERVAL pair the model chose to write it with.

    Pinned without a GPU because it is pure: the reader walks the rule's own occurrences,
    so a daily cadence written three different ways reads as one answer, and the
    time-of-day anchor is read as a stated PART (dateutil defaults an unstated hour to the
    start's, so the parsed rule cannot tell a chosen hour from an inherited one)."""
    hourly = ("FREQ=HOURLY", "FREQ=HOURLY;INTERVAL=1", "FREQ=MINUTELY;INTERVAL=60")
    daily = ("FREQ=DAILY", "FREQ=DAILY;BYHOUR=8", "FREQ=HOURLY;INTERVAL=24")
    for rule in hourly:
        assert cadence_seconds(rule) == 3600, rule
    for rule in daily:
        assert cadence_seconds(rule) == 86400, rule
    assert cadence_seconds("FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR,SA,SU;BYHOUR=8") == 86400
    assert cadence_seconds("FREQ=WEEKLY") == 604800
    assert cadence_seconds("FREQ=HOURLY;INTERVAL=2") == 7200
    assert cadence_seconds("FREQ=MINUTELY;INTERVAL=120") == 7200
    assert cadence_seconds("FREQ=DAILY;COUNT=1") is None, "a rule that fires once has no cadence"

    assert "BYHOUR" in rule_parts("FREQ=DAILY;BYHOUR=8"), "a stated hour reads as stated"
    assert "BYHOUR" not in rule_parts("FREQ=DAILY"), "an unstated hour is not invented"
    two_line = f"DTSTART:20260101T080000Z{_LINE_ESCAPE}RRULE:FREQ=DAILY;BYHOUR=8"
    assert rule_parts(two_line) == {"FREQ", "BYHOUR"}, "the rule line is read past its DTSTART"
    assert cadence_seconds(two_line) == 86400, "a one-line render round-trips into the reader"


def test_every_apply_case_seeds_a_round_that_cites_its_own_run(tmp_path) -> None:
    """The learn → apply seeds write a whole prior turn — messages, promptlog rows, the
    round's container and its entry, both transition rows — and their loud probes only run
    under ``make eval``, where a raise costs an hour of GPU before it is seen.  Drive each
    case's seeder here instead, against a real migrated database, and run the three probes
    that read the LEDGER: the round was FRAMED on the way in (#1868/#1869 — the move carries
    the framing and the container it names exists, inert), the round's calls are in the
    ledger, and everything it produced cites the run that produced it.

    The registry probe is not run here — it reads fixture skills the harness seeds after the
    case's own seed — so this pins exactly the half that is code."""
    for index, case in enumerate(APPLY_CASES):
        db = migrated_db(str(tmp_path / f"apply-{index}.db"))
        seed_learned_round(case)(db)
        assert_round_is_framed(db, case)
        assert_seeded_ledger(db, case)
        assert_round_cites_its_run(db, case)


def test_the_idle_world_seeds_five_finished_journeys_and_lands_idle(tmp_path) -> None:
    """The idle → apply cases share ONE composed world — five journeys walked to their
    end, the five jobs they left running, and a stretch of small talk after them.  A
    seeder that raises fails five cases at once, and a seeder that quietly drifts makes
    all five of them turns answered against a world nothing produces.

    Driven here against a real migrated database, with the seeder's own loud probe: the
    machine idle and unanchored, five live jobs, every round readable under its own run,
    and the small talk logged both ways.  Each case's novelty claim rides along — the
    values its ask supplies appear NOWHERE in that history, which is the whole basis of
    the "bound from the message" check.  The registry probe stays out, exactly as it does
    for the learn → apply pin: the harness seeds fixture skills after the case's seed."""
    db = migrated_db(str(tmp_path / "composed-idle.db"))
    seed_composed_world()(db)
    assert_composed_world(db)
    for case in IDLE_APPLY_CASES:
        assert_new_space_is_unknown(db, case)


def test_every_short_ask_s_own_world_seeds_and_reads_back(tmp_path) -> None:
    """Each idle → request case's world — which may hold FEWER journeys than the module's
    full five — seeds cleanly and reads back as the history it claims, with the case's own
    values still new to it.

    Its own database per distinct world, because that is what the sample gets.  A case
    seeding a reduced world is the one place the composed seeder is driven with something
    other than its default, so a `_JOURNEYS` reference left behind in a probe or a window
    would fail HERE — naming the world — rather than an hour into a GPU run as a case
    answered against a history nothing produced."""
    for index, case in enumerate(IDLE_REQUEST_CASES):
        db = migrated_db(str(tmp_path / f"composed-request-{index}.db"))
        seed_composed_world(case.journeys)(db)
        assert_composed_world(db, case.journeys)
        assert_values_are_new(db, case.case_id, case.settled.values())


def test_every_short_ask_falls_one_value_short_of_the_routine_it_names() -> None:
    """Each idle → request case accounts for its routine's declared parameters EXACTLY —
    every one either settled by the ask or named missing — and names at least one missing.

    Pinned without a GPU because it is the case's own premise, and both ways of getting it
    wrong are silent on a run.  A case claiming a parameter the routine dropped describes a
    shortfall the binder can never produce; a case whose ``missing`` went empty describes an
    ask the binder can COMPLETE, so the turn would land in apply and every check in that
    beat would read as a routing failure the model never made.  Read off the fixture DRAFT,
    which is what the runner upserts into the registry the binder is handed."""
    for case in IDLE_REQUEST_CASES:
        declared = sorted(parameter.name for parameter in case.skill.parameters)
        accounted = sorted([*case.settled, *case.missing])
        assert declared == accounted, (
            f"{case.case_id}: the routine declares {declared}, the case accounts for {accounted}"
        )
        assert case.missing, f"{case.case_id}: a request case is an ask the words fall SHORT of"


def test_every_supply_is_answered_against_a_world_parked_on_its_own_ask(tmp_path) -> None:
    """Each request → apply case's world — the composed history its short ask was measured
    against, then the turn that ask opened — seeds cleanly and reads back as the history it
    claims: the jobs still running, every round readable under its own run, the conversation
    ending on the ask and the reply that asked for what it left out, and the machine parked
    in request on that ask, carrying the binding it is waiting on (#1894) and no framing.

    Driven here against a real migrated database because the seeder is CODE, and its loud
    probe otherwise runs only under ``make eval`` — where a raise costs an hour of GPU
    before it is seen.  Its own database per case, because that is what the sample gets, and
    because a case seeding a reduced world is where a leftover ``_JOURNEYS`` reference would
    show.  The case's own premise rides along in the same loop: the supply answers exactly
    what the ask fell short of, and the two turns together answer the routine's declared
    parameters — read off the fixture DRAFT, since the registry the probe reads is seeded by
    the runner and not here.  Both ways of getting that wrong are silent on a run, and each
    turns the whole beat into a measurement of something else.  The reply check rides along
    for the reason its sibling beat's does: a cadence vocabulary that cannot match the
    answer the case itself calls correct would score every sample a miss, and that is a
    scorer bug this suite has shipped once already.

    The recorded BINDING is read back through the production model here too, against the
    values the case declares: the seeder writes what a request turn's binder left (#1894),
    and everything the turn under test is answered from — the waiting-on section the
    classifier is shown, and the values the next binder completes from — is that one row."""
    for index, case in enumerate(REQUEST_APPLY_CASES):
        db = migrated_db(str(tmp_path / f"parked-request-{index}.db"))
        seed_parked_in_request(case)(db)
        assert_parked_in_request_world(db, case)
        assert_values_are_new(db, case.case_id, case.supplies.values())
        parked = db.machine.latest_transition()
        assert parked is not None and parked.round_shortfall is not None
        waiting = RoundShortfall.model_validate_json(parked.round_shortfall)
        assert waiting == parked_binding(case)
        assert waiting.skill == slug_skill_name(case.parked.skill.name)
        assert waiting.bound == case.parked.settled
        assert tuple(one.name for one in waiting.missing) == case.parked.missing
        named = _names_the_cadence_check(case.reference, case)
        assert named.ok, f"{case.case_id}: {named.rationale} — reference: {case.reference!r}"
        declared = sorted(parameter.name for parameter in case.parked.skill.parameters)
        assert declared == sorted(case.bound), (
            f"{case.case_id}: the routine declares {declared}, the two turns settle "
            f"{sorted(case.bound)}"
        )
        assert sorted(case.supplies) == sorted(case.parked.missing), (
            f"{case.case_id}: the ask fell short of {sorted(case.parked.missing)}, "
            f"the supply answers {sorted(case.supplies)}"
        )


def test_every_applied_job_seeds_and_reads_back_as_the_apply_turn_left_it(tmp_path) -> None:
    """Each collector-enactment case's world — the parked round its supply answered, then
    the turn that stood the job up — seeds cleanly and reads back as the exit state it
    claims: the container derived from the routine and the values, the routine's steps
    rendered into it, the turn's schedule, notify on, nothing stored yet, and the machine
    landed in apply carrying the completed framing.

    Driven here against a real migrated database for the reason its sibling beats' pins
    are: the seeder is CODE, and its loud probe otherwise runs only under ``make eval``,
    where each sample costs TWO live cycles.  Its own database per case, because that is
    what the sample gets, and because the held-binding world seeds fewer journeys than the
    module's five — where a leftover ``_JOURNEYS`` reference would show.

    The case's own premise rides along in the same loop, and both ways of getting it wrong
    are silent on a run.  Values that did not cover the routine's declared parameters would
    leave the job pointed at nothing while every cycle check read as the collector's miss;
    values that disagreed with what the two turns settled would name a container for a job
    nobody asked for.  The measured binding is a SPAN of what the case declares (the
    held-binding turn bound the whole spoken phrase), so containment is the reading.

    The RUNTIME JOIN (#1907) is asserted by the probe against each case's declared
    ``joins`` — so a job whose program stopped carrying the page it fetches, or one whose
    recorded reason for not carrying it went stale, fails here rather than after three live
    cycles per sample.  The declared set must name only parameters the job actually binds:
    a name outside that is a claim about a leaf nothing could fill."""
    for index, case in enumerate(ENACTMENT_CASES):
        db = migrated_db(str(tmp_path / f"applied-{index}.db"))
        seed_applied_job(case)(db)
        assert_applied_world(db, case)
        declared = sorted(parameter.name for parameter in case.skill.parameters)
        assert declared == sorted(case.values), (
            f"{case.case_id}: the routine declares {declared}, the job binds {sorted(case.values)}"
        )
        for name, settled in case.parked.bound.items():
            assert settled in case.values[name], (
                f"{case.case_id}: the job's {name!r} is {case.values[name]!r}, the two turns "
                f"settled {settled!r}"
            )
        assert set(case.joins) <= set(case.values), (
            f"{case.case_id}: the join can only fill parameters the job binds, got "
            f"{sorted(set(case.joins) - set(case.values))}"
        )


def test_every_gate_case_seeds_the_baseline_its_direction_rests_on(tmp_path) -> None:
    """Each notification-gate case's world seeds cleanly and leaves the collection holding
    exactly the observation a prior cycle would have: one entry, under the measured modal
    key, carrying the quiet value.

    The KEY is what the negative direction rests on, and getting it wrong is silent. A
    cycle that writes under a different key makes a NEW_KEY entry — a change, which
    notifies — so the case would report the gate broken when what broke was the fixture.
    The probe's second claim is the one that keeps it honest: the key is rendered in the
    HOLDINGS block the cycle reads (#1914), which is the surface that makes reusing it the
    presented path rather than a guess. Driven here against a real migrated database
    because the probe otherwise runs only under ``make eval``.

    Its own database per case, because that is what the sample gets — and because the two
    directions of a pair share a seeding, so a defect in one is a defect in both."""
    for index, case in enumerate(GATE_CASES):
        db = migrated_db(str(tmp_path / f"gate-{index}.db"))
        seed_gate_world(case)(db)
        assert_applied_world(db, case.applied, holding={case.key_now: case.stored})
        _assert_the_baseline_is_stored(db, case)


def test_every_gate_pair_keeps_its_two_values_distinguishable() -> None:
    """The value a gate case stores and the value its positive twin's page carries are
    different values — in the bare form and in the instruction-labelled pair (#1918).

    The pair is the whole design: one direction needs the served datum to EQUAL what is
    stored (so the write gate stops), the other needs it to DIFFER (so the write lands and
    notifies). A stored value that contained the changed one, or the other way round,
    would make one of the two unreachable however well the cycle behaved."""
    for case in GATE_CASES:
        stored, changed = case.stored.casefold(), case.applied.fact.changed.casefold()
        assert stored != changed, f"{case.case_id}: the two directions need two values"
        label = case.container.casefold()
        for form in (changed, f"{label}: {changed}"):
            assert stored not in form, (
                f"{case.case_id}: the stored value {case.stored!r} must not live inside the "
                f"changed one's stored form {form!r}"
            )
        assert changed not in stored, (
            f"{case.case_id}: the changed value must not live inside the stored one"
        )
        expected = case.applied.fact.changed if case.notifies else case.stored
        assert case.datum == expected, (
            f"{case.case_id}: the served datum must be the one its direction measures"
        )


def test_every_change_cycle_page_moves_exactly_the_fact_its_case_watches() -> None:
    """Each enactment case's two pages differ, and differ on the ONE fact the case says
    they do: the quiet page carries what the job was set up around and not what comes
    next, the altered page the other way round.

    Pinned without a GPU because both ways of getting it wrong are silent on a run and each
    makes the change cycle measure nothing.  A variant that matched no replacement is
    byte-identical to the page before it, so the "the collection holds what the page says"
    check would read the SAME value twice and score the change cycle a miss on every
    sample; a variant carrying both facts would score it green without the value ever
    moving.

    The ONE-SPAN claim is the code owner's ruling made checkable: "change the datum the
    skill is looking for, not the structure".  A pair that also moved a date, a wording or
    an ordering gives a cycle a second thing to notice, and then what the change cycle
    measured is no longer the datum."""
    for case in ENACTMENT_CASES:
        assert case.quiet.text != case.altered.text, f"{case.case_id}: the pages must differ"
        _assert_one_span_moved(case)
        _assert_the_halves_exclude_each_other(case)
        assert case.fact.quiet in case.quiet.text, f"{case.case_id}: the quiet fact must be there"
        assert case.fact.changed not in case.quiet.text, (
            f"{case.case_id}: the quiet page must not already carry {case.fact.changed!r}"
        )
        assert case.fact.changed in case.altered.text, (
            f"{case.case_id}: the altered page must carry {case.fact.changed!r}"
        )
        assert case.fact.quiet not in case.altered.text, (
            f"{case.case_id}: the altered page must no longer carry {case.fact.quiet!r}"
        )


def _assert_the_halves_exclude_each_other(case: _EnactmentCase) -> None:
    """A case's two expected values are MUTUALLY EXCLUSIVE — neither contained in the
    other, in the bare form or in the instruction-labelled pair a cycle may store (#1918).

    The change cycle asserts one value is present and the other gone, so a quiet half that
    lived inside the changed half's stored form would make that check unsatisfiable: the
    cycle would do exactly the right thing and score a miss.  Cheap to state, silent to
    get wrong — "not scheduled" inside "scheduled 05:20" was the shape that motivated it."""
    quiet, changed = case.fact.quiet.casefold(), case.fact.changed.casefold()
    # The labelled form a cycle may store: the extract INSTRUCTION, then the value.  The
    # instruction is whatever leaf the routine points at, so the label is stood in for by
    # the job's own terms — what matters is that a prefix cannot make one half contain the
    # other, and any prefix would do.
    label = case.container.casefold()
    for form in (changed, f"{label}: {changed}"):
        assert quiet not in form, (
            f"{case.case_id}: the quiet value {case.fact.quiet!r} must not live inside the "
            f"changed one's stored form {form!r}"
        )
    for form in (quiet, f"{label}: {quiet}"):
        assert changed not in form, (
            f"{case.case_id}: the changed value {case.fact.changed!r} must not live inside "
            f"the quiet one's stored form {form!r}"
        )


def _assert_one_span_moved(case: _EnactmentCase) -> None:
    """The two pages differ in EXACTLY one contiguous span — the datum and nothing else.

    Read line by line through ``difflib`` rather than trusted to the derivation: the
    helper that builds a variant enforces a single replacement, but the QUIET page is
    itself derived, and two independently-derived texts can diverge anywhere.  This is
    the claim that makes the change cycle a measurement of the datum."""
    moves = [
        block
        for block in SequenceMatcher(
            None, case.quiet.text.splitlines(), case.altered.text.splitlines()
        ).get_opcodes()
        if block[0] != "equal"
    ]
    assert len(moves) == 1, (
        f"{case.case_id}: the change page must move the datum and nothing else — "
        f"{len(moves)} spans differ"
    )


def test_the_enactment_scorer_passes_three_cycles_that_did_the_job(tmp_path) -> None:
    """A baseline cycle, then a quiet one that re-read the same value and stopped at the
    write chokepoint in silence, then a change cycle that said so once — every check the
    beat gates on scores green.

    The same tripwire the reply beats carry (the "check the scorer before you blame the
    model" rule, applied before the run rather than after it): a scorer that cannot pass
    the behaviour the case itself calls correct scores every sample a miss, and this suite
    has shipped that bug once already.  THREE cycles per sample makes it the most expensive
    place in the suite to find it, so it is found here.

    The direction check is the one row that legitimately varies — it reads the CONFIGURED
    terms rather than the cycles — so it is read as a report and excluded from the claim."""
    db = migrated_db(str(tmp_path / "enactment-scorer.db"))
    for case in ENACTMENT_CASES:
        for check in _score_enactment(db, _three_good_cycles(case), case=case):
            if check.label == DIRECTION_CHECK_LABEL:
                continue
            assert check.ok, f"{case.case_id}: {check.label} — {check.rationale}"


# The key an ideal cycle stores its reading under — a fixture stand-in for whatever the
# routine's own program names, since the scorer reads keys and contents alike.
_ENACTMENT_KEY = "current"

# The surface a configured collection's cycle runs with — its program's own two calls
# (#1911) plus the terminator assembly injects a step for (#1916), which is what the
# runtime-rules block renders against.
_ENACTMENT_SURFACE = ("browse", "collection_write", PennyConstants.DONE_TOOL_NAME)


def _three_good_cycles(case: _EnactmentCase) -> list[CycleObservation]:
    """The three cycles the beat's contract describes, each doing exactly what is asked of
    it: the baseline write, the silent re-read that stops at the chokepoint, and the change
    that queues one message naming what moved."""
    quiet = {_ENACTMENT_KEY: case.fact.quiet}
    return [
        _did_the_job(case, index=0, before={}, after=quiet),
        _did_the_job(
            case, index=1, before=quiet, after=quiet, outcome=RunOutcome.NO_WORK, reason=_STOP
        ),
        _did_the_job(
            case,
            index=2,
            before=quiet,
            after={_ENACTMENT_KEY: case.fact.changed},
            sent=[f"heads up — it now says {case.fact.changed}"],
            reason=NOTIFICATION_NOTES[NotificationOutcome.QUEUED],
        ),
    ]


def _did_the_job(
    case: _EnactmentCase,
    *,
    index: int,
    before: dict[str, str],
    after: dict[str, str],
    sent: list[str] | None = None,
    outcome: RunOutcome = RunOutcome.WORKED,
    reason: str | None = None,
) -> CycleObservation:
    """One cycle that did exactly what the watch contract asks of it: fetched the page the
    job is pointed at, wrote what it said, queued whatever the change warranted, and closed
    with a run record stating it — and closed with ``done()``, which #1916 restored as
    how a cycle says it has finished.  A cycle that STOPPED at the chokepoint never
    reaches the terminator, so it carries no ``done`` record."""
    calls = [
        CycleCall(tool="browse", arguments={"queries": [case.values["url"]]}),
        CycleCall(tool="collection_write", arguments={"memory": case.container}),
    ]
    if reason != _STOP:
        calls.append(CycleCall(tool=PennyConstants.DONE_TOOL_NAME, arguments={}))
    return CycleObservation(
        index=index,
        before=before,
        after=after,
        sent=sent or [],
        calls=calls,
        served=[f"## browse: {case.values['url']}\n{next(iter(after.values()))}"],
        outcome=outcome.value,
        reason=reason,
    )


def test_every_configured_term_reads_off_the_prompt_the_collector_composes() -> None:
    """Every surface the directionality check reads is text the collector really composes
    for that collection (#1907) — the instructions, the routine and what it is for, the
    values by name, and the collection's own name and description.

    The check answers WHERE a term survived configuration, so its surfaces have to be the
    collector's own rather than a plausible restatement of them; a surface that drifted out
    of the composed prompt would report a condition as reachable when nothing the cycle
    reads carries it.  Driven against the SHIPPED composer with the job's row and its
    routine, so the claim is a read rather than a second copy of the composition."""
    for case in ENACTMENT_CASES:
        routine = slug_skill_name(case.skill.name)
        row = MemoryRow(
            name=case.container,
            type=MemoryType.COLLECTION.value,
            description=case.job.description,
            extraction_prompt=rendered_program(case),
            skill_name=routine,
            skill_params=json.dumps(case.values),
            notify=True,
        )
        skill = Skill(
            name=routine,
            intent=case.skill.description,
            description=case.skill.description,
            steps="[]",
            parameters=json.dumps([{"name": name} for name in case.values]),
            author=PennyConstants.CHAT_AGENT_NAME,
        )
        composed = Collector._compose_prompt(row, skill, frozenset(_ENACTMENT_SURFACE))
        for surface, text in configured_terms(case).items():
            assert text in composed, (
                f"{case.case_id}: the {surface!r} surface must be text the collector reads — "
                f"{text!r} is not in the composed prompt"
            )


def test_every_bail_is_answered_against_the_parked_world_it_claims(tmp_path) -> None:
    """Each bail case's world — the parked state its own edge was measured against — seeds
    cleanly and reads back as that state, with its container premise intact.

    Driven here against a real migrated database for the reason its sibling beats' pins are:
    the seeders are CODE, and their loud probes otherwise run only under ``make eval``, where
    a raise costs an hour of GPU before it is seen.  Its own database per case, because that
    is what the sample gets, and because two of these worlds seed FEWER journeys than the
    module's five — where a leftover ``_JOURNEYS`` reference would show.

    The container premise rides along in the same loop, and both ways of getting it wrong
    are silent on a run: a container that arrived already archived would score the beat's
    one cleanup claim green for free, and a world carrying a framing nobody accounted for
    would make that claim's n/a a real miss.  The registry claim stays out, exactly as it
    does for the learn → apply pin: the harness seeds fixture skills after the case's seed."""
    for index, case in enumerate(BAIL_CASES):
        db = migrated_db(str(tmp_path / f"bail-{index}.db"))
        case.world.seed(db)
        case.world.seeded(db)
        assert_the_round_built_what_it_claims(db, case)


def test_every_teach_is_answered_against_a_world_that_knows_neither_page_nor_fact(
    tmp_path,
) -> None:
    """Each idle → learn case's teach names a page the composed world has never seen, and
    that page carries a fact the world has never said.

    Both claims are silent on a run if they break, and each one hollows the beat in its own
    way.  A page the history already read makes the demonstration a re-run rather than a
    round.  A fact the history already SAYS makes the SAID == DID check green with the page
    unread, because that check reads every turn of Penny's this sample and the seeded
    confirmations are turns of Penny's.  The case's own premise rides along in the same
    reading — the teach names its page, and the fixture is the page that url reaches.

    One database for the five, because they share one world: the composed seeder's own loud
    probe runs here too, so a drift fails naming the world rather than an hour into a GPU
    run.  The registry claim stays out, exactly as it does for the sibling beats' pins: the
    harness seeds fixture skills after the case's own seed."""
    db = migrated_db(str(tmp_path / "composed-teach.db"))
    seed_composed_world()(db)
    assert_composed_world(db)
    for case in IDLE_LEARN_CASES:
        assert_the_teach_is_new_to_the_world(db, case)


def test_the_teach_scorer_passes_each_case_s_own_reference_reply() -> None:
    """Every idle → learn case's reference reply — the answer the case itself calls correct
    — passes that beat's two REPLY checks.

    The same tripwire the request and bail beats carry (the "check the scorer before you
    blame the model" rule, applied before the run rather than after it): a reply check that
    cannot pass the agreed answer scores every sample a miss, and this suite has shipped
    that bug once already.  Both checks read the reply alone, so they run here rather than
    costing an hour of GPU to find."""
    for case in IDLE_LEARN_CASES:
        for check in _round_reported_checks(case.stored, case.reference, [case.reference]):
            assert check.ok, f"{case.case_id}: {check.label} — reference: {case.reference!r}"


def test_every_correction_is_answered_against_the_round_it_corrects(tmp_path) -> None:
    """Each learn → learn case's world — the composed history plus the completed teach round
    its correction answers — seeds cleanly and reads back as that state: the five jobs still
    running, the round readable under its own run with its container inert and holding what
    the demonstration wrote, the conversation ending on the teach and the report, and the
    machine parked in learn on that teach carrying both pieces of the round's entry state —
    the framing a correction carries rather than re-draws, and the minted-fresh provenance
    (#1902) saying the routine standing under that pinned name is the round's own.

    Driven here against a real migrated database for the reason its sibling beats' pins are:
    the seeder is CODE, and its loud probe otherwise runs only under ``make eval``, where a
    raise costs an hour of GPU before it is seen.  Its own database per case, because that
    is what the sample gets.

    The case's own premise rides along, and every way of getting it wrong is silent on a
    run: a corrected target the page does not carry contracts the round to find something
    that is not there; one the correction spells out itself makes a stored value prove
    nothing about a page being read; one equal to the value already stored is a redirect to
    where the round already was; and one the world has already said or stored would score
    green with the page unread.  The registry claim stays out, exactly as it does for the
    sibling beats' pins: the harness seeds fixture skills after the case's own seed."""
    for index, case in enumerate(CORRECTION_CASES):
        db = migrated_db(str(tmp_path / f"corrected-{index}.db"))
        seed_corrected_round(case)(db)
        assert_the_teach_round_is_parked(db, case)
        assert_the_correction_is_unsaid(db, case)


def test_the_correction_scorer_passes_each_case_s_own_reference_reply() -> None:
    """Every learn → learn case's reference reply — the answer the case itself calls correct
    — passes that beat's two REPLY checks.

    The same tripwire every beat over this world carries (the "check the scorer before you
    blame the model" rule, applied before the run rather than after it): a reply check that
    cannot pass the agreed answer scores every sample a miss.  Here it reads the CORRECTED
    value, which is the whole point — a report naming the value the round replaced would be
    the wrong answer stated well."""
    for case in CORRECTION_CASES:
        for check in _round_reported_checks(case.corrected, case.reference, [case.reference]):
            assert check.ok, f"{case.case_id}: {check.label} — reference: {case.reference!r}"


def test_every_way_a_correction_can_be_answered_has_its_own_name() -> None:
    """The shape naming PARTITIONS the observations it is composed from — every combination
    of "did it re-run", "did it store the corrected value" and "did it store the one it
    replaced" lands on a phrase of its own, and the claim breaks a tie in exactly one of
    them.

    Pinned because that phrase is what the report hands the code owner to answer the
    question this beat exists for, and a naming that collapsed two observations onto one
    wording would read as an answer while hiding which failure occurred — which the first
    draft did, reporting a run that stored the corrected value AND re-stored the one it
    replaced as a clean delta-apply.

    Stated as PROPERTIES rather than as a second copy of the table, which would pass by
    agreeing with whatever the function does: the eight observations must produce eight
    distinct phrases, and the claim may only decide the one where nothing was fetched and
    nothing was written — the only place a reply is the sole evidence there is.  Two anchors
    say which combination holds the pass and which holds the shape this whole beat watches
    for, read from the module's own constants so a rewording moves both sites at once."""
    named = {
        (refetched, stored, kept, said): _correction_shape(
            refetched=refetched, stored=stored, kept=kept, said=said
        )
        for refetched in (True, False)
        for stored in (True, False)
        for kept in (True, False)
        for said in (True, False)
    }
    observations = [key[:3] for key in named if key[3]]
    decided_by_the_claim = {
        triple for triple in observations if named[(*triple, True)] != named[(*triple, False)]
    }
    assert decided_by_the_claim == {(False, False, False)}, (
        f"the claim must decide one observation, it decided {sorted(decided_by_the_claim)}"
    )
    phrases = {named[(*triple, False)] for triple in observations}
    assert len(phrases) == len(observations), (
        f"every observation must have its own name, got {named}"
    )
    assert named[True, True, False, False] == SHAPE_RE_RAN_AND_APPLIED
    assert named[False, True, False, False] == SHAPE_DELTA_WITHOUT_RE_RUNNING


def test_the_bracket_key_world_probe_passes_the_world_its_seed_lays_down(db) -> None:
    """The bracket-key guards' own seed satisfies its own loud probe.

    The probe asserts three premises — the collection is inert, it holds the target under
    its bare multi-word key, and the read surface renders that key in invocation form — and
    all three are properties of the FIXTURE, so a fixture edit that broke any of them would
    otherwise surface as two guards failing an hour into a GPU run."""
    _seed_board_games(db)
    for case in BRACKET_KEY_CASES:
        assert_board_games_world(db, case)


def test_the_registry_claim_reads_collections_not_the_system_log_markers(db) -> None:
    """The dispatch stories' registry claim is about COLLECTIONS, and a fresh migrated
    database satisfies it.

    The regression pin for the defect that failed all eight dispatch cases at the probe on
    their first live run, before a single sample was driven: the claim was read through a
    helper that returns every ``memory`` row, and migration 0026 seeds four system LOG
    markers into every database that has ever migrated — so the probe was asserting
    something no Penny has ever been true of.  Both directions, since a read that filtered
    everything out would pass this as vacuously as the correct one: the four markers are
    invisible to it, and a real collection is not."""
    assert sorted(row.name for row in db.memories.list_all()) == [
        "browse-results",
        "collector-runs",
        "penny-messages",
        "user-messages",
    ]
    assert collection_names(db) == []
    assert_no_collections(db, "a fresh migrated database")

    _seed_board_games(db)
    assert collection_names(db) == [BOARD_GAMES.name]
    with pytest.raises(AssertionError, match=BOARD_GAMES.name):
        assert_no_collections(db, "a database holding a collection")


async def test_each_dispatch_probe_accepts_the_world_its_own_hook_stands_up(
    mock_llm, running_penny, test_config
) -> None:
    """Every dispatch story's loud probe passes against a REAL migrated database and a REAL
    chat surface — both halves of all three, inside ``make check``.

    The probes run at eval time only, so the ``eval`` marker is exactly what let a probe
    that could never pass reach a live run through green CI.  Driving them here against the
    same construction path a sample uses closes that: the config-gated hooks really do
    register their tools on ``get_tools``, and the registry claim really is satisfiable in
    the world a sample is seeded into.  No model call is made — the probes only read the
    surface and the store."""
    async with running_penny(test_config) as penny:
        install_mailbox(penny)
        install_image_client(penny)
        for email_case in EMAIL_CASES:
            assert_mailbox_world(penny, email_case)
        for image_case in IMAGE_CASES:
            assert_image_world(penny, image_case)
        for choose_case in CHOOSE_CASES:
            assert_choose_world(penny, choose_case)


def test_the_dispatch_no_fire_scorers_pass_each_case_s_own_reference_reply() -> None:
    """Every dispatch case's reference reply — the answer the case itself calls correct —
    passes its module's one reply floor, and a reply that really does make the claim fails
    it.

    The same tripwire the chat beats carry (the "check the scorer before you blame the
    model" rule, applied before the run rather than after it).  Both floors read a
    vocabulary, and a vocabulary that cannot match the agreed answer would score every
    sample a miss — while one that matches nothing at all would pass a reply claiming an
    inbox was searched or a picture drawn.  Both halves are checked, because only the pair
    keeps the floor meaning anything."""
    for case in EMAIL_CASES:
        claimed = _claims_no_search_check(case.reference)
        assert claimed.ok, f"{case.case_id}: {claimed.rationale} — reference: {case.reference!r}"
    assert not _claims_no_search_check("i checked your inbox — nothing from priya.").ok

    for case in IMAGE_CASES:
        drew = _claims_no_picture_check(case.reference)
        assert drew.ok, f"{case.case_id}: {drew.rationale} — reference: {case.reference!r}"
    assert not _claims_no_picture_check("here's the picture you asked for!").ok


def test_the_choose_scorer_reads_each_case_s_own_reference_reply() -> None:
    """The choose story's two reply checks both pass the answers the cases call correct,
    and both refuse the answers they exist to catch.

    Said-equals-did is a comparison against the pick the RUN produced, so the pin replays
    the fire case's reference as though the tool had returned the option it names: it must
    agree with that option and DISAGREE with the two it does not, since a comparison that
    passed every option would let a free-chosen reply score green behind a real call.  The
    opinion floor is the same pair on the other direction."""
    fires = next(case for case in CHOOSE_CASES if case.fires)
    named = [option for option in _CHOOSE_OPTIONS if _reply_reports(option, fires.reference)]
    assert len(named) == 1, f"the reference must report exactly one pick, it reports {named}"

    judgment = next(case for case in CHOOSE_CASES if not case.fires)
    opinion = _gave_an_opinion_check(judgment.reference)
    assert opinion.ok, f"{judgment.case_id}: {opinion.rationale}"
    assert not _gave_an_opinion_check("honestly, whichever you like the sound of.").ok


def test_the_bail_scorer_passes_each_case_s_own_reference_reply() -> None:
    """Every bail case's reference reply — the answer the case itself calls correct — passes
    that beat's one reply check.

    The same tripwire the request beats carry, on the check most likely to trip it: a
    job-was-set-up vocabulary wide enough to catch an ordinary acknowledgement would score
    every sample a miss, and the correct answers here are exactly the sentences that say
    "dropped it" without saying anything is running."""
    for case in BAIL_CASES:
        claimed = _claims_no_job_check(case.reference)
        assert claimed.ok, f"{case.case_id}: {claimed.rationale} — reference: {case.reference!r}"


def test_the_request_scorer_passes_each_case_s_own_reference_reply() -> None:
    """Every idle → request case's reference reply — the answer the case itself calls
    correct — passes that beat's two REPLY checks.

    The tripwire for a scorer bug (the "check the scorer before you blame the model" rule,
    applied before the run rather than after it).  Both checks read a vocabulary, and a
    vocabulary that cannot match the agreed answer would score the beat's own reference a
    miss on every sample: measured once already, where a found-thing set built from
    parameter names had no entry for "looking out for", which is how a person writes it.
    Cheap and deterministic, so it runs here rather than costing an hour of GPU to find."""
    for case in IDLE_REQUEST_CASES:
        asked = _asks_for_what_is_missing_check(case.reference, case)
        assert asked.ok, f"{case.case_id}: {asked.rationale} — reference: {case.reference!r}"
        held = _does_not_re_ask_check(case.reference, case)
        assert held.ok or held.ignored, (
            f"{case.case_id}: {held.rationale} — reference: {case.reference!r}"
        )


def test_a_settled_value_is_said_back_however_an_address_is_written() -> None:
    """A value the ask already settled counts as said back when the reply repeats it,
    case-folded and with any address scheme stripped first.

    The reply is written for a person, so an address routinely comes back without its
    scheme — and matching the stored form literally would score that a re-ask, which is the
    scorer inventing a failure.  What it must still REFUSE is the half that would stop the
    check meaning anything: a reply that names something else, or names nothing."""
    address = "https://northpier.example/departures"
    for reply in (
        "i'll watch https://northpier.example/departures every morning — what for?",
        "got it, northpier.example/departures. what should i look out for?",
        "watching NORTHPIER.EXAMPLE/DEPARTURES — what am i looking for?",
    ):
        assert _said_back(address, reply), reply
    for reply in ("sure — which page should i watch?", "what's the link?", ""):
        assert not _said_back(address, reply), reply
    assert _said_back("dawn sailing", "which page has the dawn sailing on it?")
    assert not _said_back("dawn sailing", "which sailing did you mean?")


def test_a_parameter_binds_on_any_value_that_locates_the_expected_one() -> None:
    """A bound parameter matches its expected phrase by OVERLAP in either direction
    (code-owner ruling) — so every spelling that locates the same thing passes, and an
    unrelated value still fails.

    The measured case: told to watch for the dawn sailing, a routine bound to `dawn` finds
    exactly the line `dawn sailing` would, and one-way containment scored that a binding
    failure when it is a wording preference.  Pinned here rather than on a GPU because the
    matcher is pure — and because what it must still REFUSE is the half that would quietly
    stop meaning anything."""
    expected = "dawn sailing"
    for value in ("dawn", "Dawn", "dawn sailing", "Dawn Sailing", "the dawn sailing line"):
        assert _overlaps(expected, [value]), f"{value!r} locates the dawn sailing"
    for value in ("late sailing", "timetable", "north pier"):
        assert not _overlaps(expected, [value]), f"{value!r} locates something else"
    assert not _overlaps(expected, []), "a routine that bound nothing matches nothing"
    for empty in ("", "   "):
        assert not _overlaps(expected, [empty]), "an empty value is evidence of nothing"
    assert _overlaps(expected, ["north pier", "dawn"]), "one bound value is enough"


def test_the_idle_worlds_window_carries_pennys_turns_in_order(tmp_path) -> None:
    """Penny's side of the seeded history reaches the CONVERSATION, not just the record.

    ``get_messages_since`` — what ``_build_conversation`` reads — takes the incoming
    messages plus Penny's replies TO THOSE MESSAGES, matched by ``parent_id``, plus
    autonomous sends (which carry no parent). An unthreaded outgoing row satisfies neither
    leg, so it is logged and invisible: the window comes back all-user and the same-role
    merge folds the whole history into ONE giant user turn. That is what the first live run
    of these cases answered — nineteen turns stacked into one message reading as a pile of
    unanswered requests — so the threading is pinned here rather than rediscovered on a GPU.

    The exhaustive turn-for-turn equality is ``assert_composed_world``'s; what this adds is
    the two claims a reader of the case cares about — every journey's confirmation is an
    ASSISTANT turn, in journey order, and the small talk is what the window ends on."""
    db = migrated_db(str(tmp_path / "composed-window.db"))
    seed_composed_world()(db)
    window = db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=200)
    assistant = [
        row.content for row in window if row.direction == PennyConstants.MessageDirection.OUTGOING
    ]
    confirmations = [line for line in assistant if line in JOURNEY_CONFIRMATIONS]
    assert confirmations == list(JOURNEY_CONFIRMATIONS), (
        f"every apply confirmation is an assistant turn, in order — got {confirmations}"
    )
    tail = [(row.direction, row.content) for row in window[-len(LAST_SPOKEN_TURNS) :]]
    assert tail == list(LAST_SPOKEN_TURNS), f"the window must end on the small talk, got {tail}"


def test_thinking_tokens_are_read_from_the_provider_not_guessed(tmp_path) -> None:
    """The reasoning count is a READ of what the provider reported, with the character
    ratio only as the fallback where a backend reports none.

    Read with care: it counts the SEPARATE reasoning channel, so it does NOT compare
    across models that put their thinking in different places — one reasoning inline in
    its visible content reports zero while thinking just as hard. A reported zero means
    "nothing in the reasoning channel", never "this model did not think", and reading it
    as the latter is how a whole suite run was drawn against a model whose reasoning the
    provider had quietly switched off. What compares cleanly is total OUTPUT tokens: a
    local GPU has to generate those wherever the thinking lives.
    """
    db = _make_db(tmp_path)
    response = {
        "choices": [{"message": {"role": "assistant", "content": "the visible answer"}}],
        "usage": {
            "prompt_tokens": 400,
            "completion_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 90},
        },
    }
    _log_prompt(db, response=response, run_id="r1")
    perf = live_prompt_perf(db)
    assert perf.input_tokens == 400
    assert perf.output_tokens == 120, "completion_tokens already bundles the reasoning"
    assert perf.reasoning_tokens == 90, "the provider's own count, not a ratio of characters"

    # A backend that reports no detail leaves it 0 — read by the caller as "not reported"
    # and never as a confident zero, which is why the fallback ratio inputs are still kept.
    bare = _make_db(tmp_path / "bare")
    _log_prompt(
        bare,
        response={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        run_id="r1",
    )
    assert live_prompt_perf(bare).reasoning_tokens == 0


def test_a_seeded_prior_turn_is_not_read_as_this_samples_work(tmp_path) -> None:
    """A case may seed the promptlog of turns that happened BEFORE the one under test
    (#1846), so the sample is answered against the state those turns really left.  Those
    rows are history: every "what did the model do" reader excludes them by their run id,
    which the seeder mints under the shared prefix.

    Pinned here because the exclusion is what keeps a negative check honest — a seeded
    round's browse must not read as this turn's, which is exactly the check the learn →
    apply cases score ("she set it running instead of running it again").  A live run's
    own rows are untouched, so every other case reads identically to before."""
    db = _make_db(tmp_path)
    _log_prompt(db, response=_tool_call_response("browse"), run_id=seeded_run_id("learn-turn"))
    assert tool_not_called(db, "browse"), "a seeded prior turn's call is not this sample's"
    assert count_tool_calls(db, "browse") == 0
    assert live_prompt_perf(db).calls == 0, "a seeded row is not one of this sample's calls"

    _log_prompt(db, response=_tool_call_response("browse"), run_id="r1")
    assert tool_was_called(db, "browse"), "the sample's own call still reads"
    assert count_tool_calls(db, "browse") == 1, "only the live call is counted"
    assert live_prompt_perf(db).calls == 1

    assert is_seeded_run(seeded_run_id("learn-turn"))
    assert not is_seeded_run("r1")
    assert not is_seeded_run(None), "an unstamped row is a live row, not a seeded one"


def test_tool_call_rejected_matches_backticked_tool_name_form(tmp_path) -> None:
    # The framework arg-validation failure leads with the backticked TOOL name
    # (`FRAMEWORK_NARRATION_INVALID_ARGS`) — the shape the per-tool probe already matched.
    db = _make_db(tmp_path)
    frame = _framed_result(
        "update_entry",
        {"memory": "trip-notes", "key": "hotel"},
        ok=False,
        narration=FRAMEWORK_NARRATION_INVALID_ARGS.format(tool_name="update_entry"),
    )
    assert "`update_entry`" in frame[0]["content"]  # the tool name IS backticked in this form
    _log_prompt(db, messages=frame)
    assert tool_call_rejected(db, "update_entry")
    assert tool_call_rejected(db)  # any-tool probe
    assert not tool_call_rejected(db, "collection_write")


def test_tool_call_rejected_matches_memory_tool_target_backticked_form(tmp_path) -> None:
    # A memory-tool execute-time failure backticks the TARGET, not the tool — the tool is
    # named only in the `(<tool> result)` tag.  Before #1726 a per-tool probe matched solely
    # the backticked tool name and went blind to these, false-greening every memory-surface
    # rejection check.  Frames are the PRODUCTION templates (via `Tool.format_result`).
    db = _make_db(tmp_path)
    write_frame = _framed_result("collection_write", {"memory": "trip-notes"}, ok=False)
    update_frame = _framed_result(
        "update_entry", {"memory": "trip-notes", "key": "hotel"}, ok=False
    )
    _log_prompt(db, messages=write_frame)
    _log_prompt(db, messages=update_frame)

    # The bug's signature: the tool name is NOT backticked — only the tag names it.
    assert "`collection_write`" not in write_frame[0]["content"]
    assert "(collection_write result)" in write_frame[0]["content"]
    assert "`update_entry`" not in update_frame[0]["content"]
    assert "(update_entry result)" in update_frame[0]["content"]

    # The fix: attributed by the tag, each rejection is visible to its per-tool probe again.
    assert tool_call_rejected(db, "collection_write")
    assert tool_call_rejected(db, "update_entry")
    assert tool_call_rejected(db)  # any-tool probe
    assert not tool_call_rejected(db, "log_append")  # a tag names exactly one tool

    # The attribution primitive recognises the tag shape and never cross-attributes.
    assert _frame_attributes_to(write_frame[0]["content"], "collection_write")
    assert not _frame_attributes_to(write_frame[0]["content"], "update_entry")


def test_sample_is_fragile_detects_recovery_frames(tmp_path) -> None:
    db = _make_db(tmp_path)
    _log_prompt(
        db,
        messages=_framed_result(
            "collection_write",
            {"memory": "trip-notes", "entries": [{"key": "hotel"}]},
            ok=True,
            mutated=True,
        ),
    )
    assert not sample_is_fragile(db)
    # A memory-tool target-backticked rejection is a recovery frame too: `sample_is_fragile`
    # filters no tool name, so its `_RECOVERY_FRAMES` set catches "didn't work" regardless of
    # which (target-backticked) tool produced it — no attribution gap here (#1726 audit).
    _log_prompt(db, messages=_framed_result("collection_write", {"memory": "trip-notes"}, ok=False))
    assert sample_is_fragile(db)


def test_sample_is_fragile_counts_a_user_turn_recovery_nudge(tmp_path) -> None:
    # #1735 finding 2: the render marks a continue / parse-failure USER-turn nudge `⚠ recovery
    # event`, but `sample_is_fragile` used to count only TOOL-role recovery frames — so a
    # nudge-recovered pass banner'd clean, decoupled from the render.  Widened (single-sourced
    # through the render's own `_is_nudge`), a recovery nudge now flags the sample fragile too.
    db = _make_db(tmp_path)
    _log_prompt(db, messages=[{"role": "user", "content": "when does the shop open?"}])
    assert not sample_is_fragile(db)  # a real user ask is not a recovery event
    # The empty-response CONTINUE_NUDGE, injected as a user turn — a recovery nudge.
    _log_prompt(db, messages=[{"role": "user", "content": "Please provide your response."}])
    assert sample_is_fragile(db)


# ── The loop-health advisory: a re-rolled draw, not a deleted nudge (#1839/#1841) ──


def test_draw_rerolled_reads_the_repeated_context_a_discarded_draw_leaves(tmp_path) -> None:
    # #1840 deleted the text-bail nudges the old probe matched on, so no new run can carry those
    # markers.  What a discarded draw DOES still leave is the second draw: the loop re-calls on
    # the unchanged message list and the client persists every completed draw, so a re-rolled
    # step is two rows with byte-identical `messages` — while an ordinary step's context has
    # grown by the turns the previous step appended and can never repeat.
    db = _make_db(tmp_path)
    step_one = [{"role": "user", "content": "what does the deck cost?"}]
    step_two = [
        *step_one,
        {"role": "assistant", "content": "checking the listing"},
        {"role": "tool", "content": "$499"},
    ]
    _log_prompt(db, messages=step_one)
    _log_prompt(db, messages=step_two)
    assert not draw_rerolled(db)
    assert routing_clean(db)

    # The discarded draw's row: the SAME context, drawn again.
    _log_prompt(db, messages=step_two)
    assert draw_rerolled(db)
    assert not routing_clean(db)

    # A micro-context re-draws the same way, and its shape-violation re-draw (`_draw`, the outer
    # loop) mints a FRESH run id each time — so the read keys on the repeated context alone and
    # never on the run it belongs to.
    micro = _make_db(tmp_path, "micro")
    document = [{"role": "user", "content": "the rendered routine"}]
    frame_agent = PennyConstants.SKILL_FRAME_AGENT_NAME
    _log_prompt(micro, messages=document, run_id="draw-1", agent_name=frame_agent)
    assert not draw_rerolled(micro)
    _log_prompt(micro, messages=document, run_id="draw-2", agent_name=frame_agent)
    assert draw_rerolled(micro)


def test_routing_clean_keeps_the_legacy_bail_marker_and_continue_nudge_halves(tmp_path) -> None:
    # A promptlog written BEFORE #1840 carries the retired bail nudge as a user turn.  Nothing
    # can write one now, but the marker stays as the legacy leg so a historical row still reads.
    legacy = _make_db(tmp_path, "legacy")
    _log_prompt(
        legacy,
        messages=[{"role": "user", "content": "That could not be parsed as a tool call."}],
    )
    assert draw_rerolled(legacy)
    assert not routing_clean(legacy)

    # The empty-response retry nudge is still live, and is the verdict's other half — a sample
    # that only continued because it was nudged is not cleanly routed either.
    nudged = _make_db(tmp_path, "nudged")
    _log_prompt(nudged, messages=[{"role": "user", "content": _RETIRED_CONTINUE_NUDGE}])
    assert not draw_rerolled(nudged)
    assert continue_nudge_fired(nudged)
    assert not routing_clean(nudged)


# ── The graded runner paths: dispatch + framework guard-as-Check (#1697) ──


def test_scorer_is_graded_dispatches_on_return_type() -> None:
    # A graded scorer returns Checks; a binary one returns failure strings; empty → binary (pass).
    assert _scorer_is_graded([Check("wrote entry", ok=True)])
    assert not _scorer_is_graded(["did not write the entry"])
    assert not _scorer_is_graded([])


def test_bail_fired_and_cycle_recovered_guard_checks() -> None:
    # Each guard is a scored Check: it passes silently (no rationale) when the contract fired, and
    # fails with a rationale naming the vacuous contract when it did not — so a run the injected
    # trigger never reached can't score green off the scorer's own checks alone.
    fired = _bail_fired_check(True)
    assert fired.ok and fired.scored and fired.rationale is None
    missed = _bail_fired_check(False)
    assert not missed.ok and missed.rationale is not None
    recovered = _cycle_recovered_check(True)
    assert recovered.ok and recovered.rationale is None
    stalled = _cycle_recovered_check(False)
    assert not stalled.ok and stalled.rationale is not None


def test_guarded_graded_prepends_guard_and_gates_a_vacuous_contract() -> None:
    # A scorer whose own check PASSES but whose injected bail never fired: the prepended guard
    # (leading the list) drags the sample below a full pass — the vacuous-contract catch.
    vacuous = _guarded_graded([Check("wrote the entry", ok=True)], [_bail_fired_check(False)])
    assert vacuous.total == 2  # guard + scorer check, both scored
    assert vacuous.score == 0.5 and not vacuous.passed
    assert vacuous.checks[0].label == "forced bail fired — contract exercised"  # guard leads
    # With the bail fired, the same scorer sample is a clean full pass.
    clean = _guarded_graded([Check("wrote the entry", ok=True)], [_bail_fired_check(True)])
    assert clean.passed and clean.total == 2


def test_guarded_graded_no_guards_is_the_startup_peripheral_path() -> None:
    # startup_eval (and the peripheral / prompt-format runners) dispatch with NO framework
    # guards — no injection — so _guarded_graded(scored, []) grades purely over the scorer's
    # own Checks.  A 2-of-3 graded text scorer scores 0.67 where the old binary scorer scored
    # 0.0 on the same miss: the monotonicity the conversion buys (graded mean >= binary mean).
    result = _guarded_graded(
        [Check("generated", ok=True), Check("length", ok=True), Check("voice", ok=False)], []
    )
    assert result.total == 3
    assert round(result.score, 2) == 0.67
    assert not result.passed
    assert result.failed == ["voice"]
    # A clean all-pass graded text scorer is a full pass, and a binary text scorer's failure
    # strings still route through the binary path (a text scorer that returns strings).
    assert _guarded_graded([Check("only", ok=True)], []).passed
    assert not _scorer_is_graded(["fell back to the canned message"])


def test_report_renders_injected_guard_check_in_footer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save X"},
            {"role": "assistant", "tool_calls": [write_call]},
        ],
    )
    # The scorer's own check passed (anchored to the write row), but the injected bail-fired guard
    # failed — so the guard-as-Check lands in the footer with its vacuous-contract rationale.
    result = _guarded_graded(
        [Check("wrote the entry", ok=True, anchor="collection_write(")],
        [_bail_fired_check(False)],
    )
    _write_sample_report(db, "guard-case", 0, result=result, reply="saved")
    text = _sample_report_text(tmp_path, "guard-case")
    # the guard failed → 1/2
    assert text.startswith("<details><summary>sample 1 — ❌ fail ·")
    assert "| actual | 🔧 collection_write({}) | ✅ C1 |" in text
    assert (
        "| expected | G1 [guard]⚖ forced bail fired — contract exercised | "
        "❌ G1 — the injected bail never fired — the recovery contract was not exercised |"
    ) in text


# ── Whole-render assertions for the new report shapes ──


def test_report_renders_rationale_and_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save X"},
            {"role": "assistant", "tool_calls": [write_call]},
        ],
    )
    result = SampleResult.graded(
        [
            Check("write happened", ok=True, anchor="collection_write("),
            Check("read count", ok=False, rationale="expected 3 reads, saw 1"),
            Check.na("browse branch", rationale="no browse this sample"),
        ]
    )
    _write_sample_report(db, "rationale-case", 0, result=result, reply="saved")
    text = _sample_report_text(tmp_path, "rationale-case")
    assert text.startswith("<details><summary>sample 1 — ❌ fail ·")
    assert "| actual | 🔧 collection_write({}) | ✅ C1 |" in text
    assert "| expected | C2 ⚖ read count | ❌ C2 — expected 3 reads, saw 1 |" in text
    assert "| expected | C3 browse branch | ➖ n/a — no browse this sample |" in text


def test_report_renders_passed_fragile(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    db = _make_db(tmp_path)
    browse_call = {"function": {"name": "browse", "arguments": "{}"}}
    reject = "You tried to use `browse` but it didn't work: down"
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "tool_calls": [browse_call]},
            {"role": "tool", "content": reject},
        ],
    )
    _write_sample_report(db, "fragile-case", 0, result=SampleResult.binary([]), reply="found it")
    text = _sample_report_text(tmp_path, "fragile-case")
    # fragile still folds whole now (#1753); banner carries the fragile flag
    assert text.startswith("<details><summary>sample 1 — ✅ pass · fragile ·")
    assert "| actual | 🔧 browse({}) |" in text
    assert "| actual | 📥 You tried to use `browse` but it didn't work: down |" in text


def test_report_blocks_land_in_sample_order_whatever_finishes_first(tmp_path, monkeypatch) -> None:
    """A case's samples may finish in any order; its report may not be written in one.

    ``EVAL_CONCURRENCY`` above 1 lets sample 3 come back before sample 1, so blocks are
    HELD as they are written and laid down by sample index when the case closes.  Written
    back-to-front here, since that is the ordering an append-as-you-go writer got wrong.
    """
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    db = _make_db(tmp_path)
    _log_prompt(db, messages=[{"role": "user", "content": "look it up"}])
    for sample_index in (2, 0, 1):
        _write_sample_report(
            db, "out-of-order", sample_index, result=SampleResult.binary([]), reply="found it"
        )
    text = _sample_report_text(tmp_path, "out-of-order")
    banners = [line for line in text.splitlines() if line.startswith("<details><summary>sample ")]
    assert [banner.split(" — ")[0] for banner in banners] == [
        "<details><summary>sample 1",
        "<details><summary>sample 2",
        "<details><summary>sample 3",
    ]
    # The flush empties the case, so a second read is not a second copy of every block.
    assert _sample_report_text(tmp_path, "out-of-order") == text


def test_report_renders_the_terminal_call_and_result_from_the_run_tail(
    tmp_path, monkeypatch
) -> None:
    """A sample whose run ended the instant a tool returned renders that call AND its
    result (#1778).

    The transcript is built from what each promptlog row CARRIED, and a terminal result
    rides into no later call — so the harness rendered the call and simply stopped, which
    is indistinguishable from a run where nothing happened.  The trailing tail the loop
    stamps at close is walked alongside the carried turns, so the transcript cannot be
    missing an entry — and the terminal failure is visible to the fragile probe, which
    reads the same turns."""
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    arguments = {"memory": "shelf"}
    call = {"id": "s1", "function": {"name": "collection_set", "arguments": json.dumps(arguments)}}
    _log_prompt(
        db,
        messages=[{"role": "user", "content": "attach the skill"}],
        response={"choices": [{"message": {"tool_calls": [call]}}]},
    )
    frame = Tool.format_result(
        "collection_set",
        arguments,
        ToolResult(message="Memory 'shelf' not found.", success=False),
    )
    db.messages.set_run_trailing_messages(
        "r1",
        [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "s1", "content": frame},
        ],
    )
    result = SampleResult.graded([Check("attached the skill", ok=False, anchor="collection_set(")])
    _write_sample_report(db, "tail-case", 0, result=result, reply="")
    text = _sample_report_text(tmp_path, "tail-case")
    assert '| actual | 🔧 collection_set({"memory": "shelf"}) | ❌ C1 |' in text
    assert f"| actual | 📥 {report.escape_cell(frame)} |" in text
    assert sample_is_fragile(db)


def test_report_banner_and_verdict_carry_the_failure_cause(tmp_path, monkeypatch) -> None:
    # The banner + the failed check's verdict carry the structural cause (#1725): a behavioral
    # miss reads ``❌ fail · behavioral`` on the banner and ``· behavioral`` on the
    # done-turn verdict, so a reader triages before unfolding.
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    _done_bail_sample(db)
    result = SampleResult.graded(
        [Check("send queued", ok=False, anchor="done(", rationale="expected 1 send, saw 0")]
    )
    result.cause = FailureCause.BEHAVIORAL  # the runner stamps this before _write_sample_report
    _write_sample_report(db, "watch-fern", 0, result=result, reply="")
    text = _sample_report_text(tmp_path, "watch-fern")
    assert text.startswith("<details><summary>sample 1 — ❌ fail · behavioral ·")
    assert "| actual | 🔧 done({}) | ❌ C1 — expected 1 send, saw 0 · behavioral |" in text
    assert "<details><summary>thinking — 75 chars</summary>The entry is already written" in text


def test_report_timeout_sample_renders_placeholder_block(tmp_path, monkeypatch) -> None:
    # A harness timeout produces no completed turn, so the transcript would otherwise silently omit
    # the sample.  It gets an explicit placeholder block (#1725/F2) — its verdict names the harness
    # cause and the body says why there is no table — so the report's sample count always matches N.
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)  # no promptlog rows — the sample timed out before any completed call
    timed = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(db, timed, timed_out=True)
    _write_sample_report(db, "timeout-case", 2, result=timed)
    text = _sample_report_text(tmp_path, "timeout-case")
    # no k/n: the scorer never ran
    assert text.startswith("<details><summary>sample 3 — ❌ fail · harness ·")
    assert report.NO_TURNS_PLACEHOLDER in text


# ── The dual strict+partial RESULT line ──


def test_result_line_reports_dual_metric(capsys) -> None:
    results = [
        SampleResult.graded([Check("a", ok=True), Check("b", ok=True)]),  # 1.0, all-pass
        SampleResult.graded([Check("a", ok=True), Check("b", ok=False)]),  # 0.5, not all-pass
    ]
    _assert_threshold("dual-case", results, None, intended=2)
    out = capsys.readouterr().out
    assert "RESULT [dual-case] mean 0.75 · all-pass 1/2 across 2 samples (report-only)" in out


def test_result_line_detail_carries_rationale(capsys) -> None:
    _assert_threshold(
        "detail-case",
        [SampleResult.graded([Check("reads", ok=False, rationale="expected 3 reads, saw 1")])],
        None,
        intended=1,
    )
    out = capsys.readouterr().out
    assert "RESULT [detail-case] mean 0.00 · all-pass 0/1 across 1 samples (report-only)" in out
    assert "  [1] 0.00 — reads — expected 3 reads, saw 1" in out


def test_result_line_gated_pass_names_mean_threshold(capsys) -> None:
    _assert_threshold(
        "gate-case", [SampleResult.binary([]), SampleResult.binary([])], 0.75, intended=2
    )
    out = capsys.readouterr().out
    assert "RESULT [gate-case] mean 1.00 · all-pass 2/2 across 2 samples (need mean >=0.75)" in out


def test_result_line_gate_fails_below_threshold() -> None:
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold("red-case", [SampleResult.binary(["boom"])], 0.75, intended=1)


# ── A mostly-dead cohort is not a result, whatever the survivors scored (#1996) ──


def test_a_mostly_dead_cohort_fails_before_any_score_is_compared() -> None:
    """Two of five samples ever ran, and BOTH passed — the case still refuses.

    This is the run that reported `6 passed, EXIT=0` over 34 dead samples: every mean it
    printed was computed from whatever survived. The refusal has to come before the
    threshold, because a perfect score over a fraction is the exact shape that looked green.
    """
    survivors = [SampleResult.binary([]), SampleResult.binary([])]
    with pytest.raises(pytest.fail.Exception) as failure:
        _assert_threshold("dead-cohort", survivors, 0.75, intended=5)

    message = str(failure.value)
    assert "2 of 5 samples produced their measured turn" in message
    assert "more than half the samples it intended" in message
    assert "at least 3 here" in message


def test_a_report_only_case_refuses_a_dead_cohort_too() -> None:
    """Report-only means "don't gate the SCORE", never "tolerate having no result"."""
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold("quiet-case", [], None, intended=5)


def test_a_cohort_that_squeaks_past_the_bar_still_scores(capsys) -> None:
    """Three of five is a real read — reduced, and the RESULT line says the N it used."""
    _assert_threshold("thin-case", [SampleResult.binary([])] * 3, 0.75, intended=5)
    assert "across 3 samples" in capsys.readouterr().out


# ── Honest-threshold restoration: gate on the pathology-excluded mean (#1698) ──


def test_gate_pathology_excluded_gates_on_the_honest_mean(capsys) -> None:
    # One clean pass + one pathology failure: the raw mean is 0.50, but the pathology sample
    # drops out of the pathology-excluded denominator, so the honest read is 1.00.  Opting in
    # (gate_pathology_excluded=True) gates on that honest 1.00 and clears an 0.8 bar the raw
    # mean would miss — the mechanism behind the speakable sequence cases' 0.6→0.8 restore.
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["collapse"])
    pathological.cause = FailureCause.PATHOLOGY
    _assert_threshold(
        "honest-case", [passed, pathological], 0.8, intended=2, gate_pathology_excluded=True
    )
    out = capsys.readouterr().out
    assert (
        "RESULT [honest-case] mean 0.50 · all-pass 1/2 across 2 samples "
        "(need pathology-excluded mean >=0.8)" in out
    )


def test_gate_pathology_excluded_still_fails_on_a_behavioral_miss() -> None:
    # A BEHAVIORAL failure stays in the pathology-excluded denominator, so the honest mean is
    # 0.50 — the opt-in gate is not a free pass; only reroll-guard pathology noise is excluded.
    passed = SampleResult.binary([])
    behavioral = SampleResult.binary(["wrong end state"])
    behavioral.cause = FailureCause.BEHAVIORAL
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold(
            "behav-case", [passed, behavioral], 0.8, intended=2, gate_pathology_excluded=True
        )


def test_pathology_noise_sinks_the_raw_gate_without_the_opt_in() -> None:
    # The flag is load-bearing: the SAME clean-pass + pathology-fail pair FAILS the default
    # raw-mean gate (0.50 < 0.8) — exactly the flake the honest-threshold restoration removes
    # by opting the case into the pathology-excluded gate above.
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["collapse"])
    pathological.cause = FailureCause.PATHOLOGY
    with pytest.raises(pytest.fail.Exception):
        _assert_threshold("raw-gate-case", [passed, pathological], 0.8, intended=2)


# ── Failure-cause partition (#1695): the structural pathology scan + stamping ──


def test_run_exhibited_pathology_detects_reroll_guard_signals(tmp_path) -> None:
    # Each of the four reroll-guard conditions the loop discards + re-rolls on, read
    # post-hoc off the persisted RESPONSE (the same text_validity detectors run live).
    degenerate = _make_db(tmp_path, "degen")
    _log_prompt(degenerate, response=_content_response("winter watering......???"))
    assert run_exhibited_pathology(degenerate)  # DEGENERATE_OUTPUT in content

    harmony = _make_db(tmp_path, "harmony")
    _log_prompt(harmony, response=_content_response("leaked <|call|> to=functions.browse"))
    assert run_exhibited_pathology(harmony)  # TOOL_CALL_LEAK

    fragment = _make_db(tmp_path, "fragment")
    _log_prompt(fragment, response=_content_response('{"memory": "notes"}'))
    assert run_exhibited_pathology(fragment)  # bare call-fragment reply (no tool calls)

    empty_object = _make_db(tmp_path, "empty")
    _log_prompt(empty_object, response=_content_response("{}"))
    assert run_exhibited_pathology(empty_object)  # bare `{}` empty-object reply (#1732 spiral tail)

    bad_name = _make_db(tmp_path, "name")
    _log_prompt(bad_name, response=_tool_call_response("Functions?????"))
    assert run_exhibited_pathology(bad_name)  # collapse-shaped tool NAME

    poison_arg = _make_db(tmp_path, "arg")
    _log_prompt(
        poison_arg, response=_tool_call_response("collection_write", '{"content": "..???.."}')
    )
    assert run_exhibited_pathology(poison_arg)  # collapse in a serialised tool-call argument


def test_run_exhibited_pathology_ignores_clean_and_input_only_poison(tmp_path) -> None:
    # A healthy run — a real tool call + a clean reply — carries no pathology signal.
    clean = _make_db(tmp_path, "clean")
    _log_prompt(clean, response=_tool_call_response("collection_write"))
    _log_prompt(clean, response=_content_response("Here's your answer."))
    assert not run_exhibited_pathology(clean)
    # Poison in the INPUT messages (e.g. an injected bail echoed into history) is NOT the
    # model's output — the scan reads only the response, so an injected trigger stays invisible.
    injected = _make_db(tmp_path, "injected")
    _log_prompt(
        injected,
        messages=[{"role": "assistant", "content": "Hi there! ......???"}],
        response=_tool_call_response("collection_write"),
    )
    assert not run_exhibited_pathology(injected)


def test_stamp_cause_partitions_pass_pathology_harness_behavioral(tmp_path) -> None:
    # Pass → no cause, regardless of the DB.
    passed_db = _make_db(tmp_path, "pass")
    passed = SampleResult.binary([])
    _stamp_cause(passed_db, passed)
    assert passed.cause is None

    # Failed + poison in the response → pathology.
    poison_db = _make_db(tmp_path, "poison")
    _log_prompt(poison_db, response=_content_response("collapse...???"))
    pathological = SampleResult.binary(["wrong end state"])
    _stamp_cause(poison_db, pathological)
    assert pathological.cause == FailureCause.PATHOLOGY

    # Failed, clean output → behavioral (the model simply got it wrong).
    clean_db = _make_db(tmp_path, "behav")
    _log_prompt(clean_db, response=_content_response("A confident but wrong answer."))
    behavioral = SampleResult.binary(["wrong end state"])
    _stamp_cause(clean_db, behavioral)
    assert behavioral.cause == FailureCause.BEHAVIORAL

    # Timeout on a clean DB → harness; but poison outranks the timeout symptom.
    timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(clean_db, timeout, timed_out=True)
    assert timeout.cause == FailureCause.HARNESS
    poison_timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(poison_db, poison_timeout, timed_out=True)
    assert poison_timeout.cause == FailureCause.PATHOLOGY


def test_nudge_loop_spiral_classifies_pathology_not_harness(tmp_path) -> None:
    # #1732: the #1731 spiral — a chat run that looped on recovery nudges, re-browsing the same
    # page each cycle, and died at the turn timeout with a bare `{}` reply.  Reconstructed via
    # the REAL production serialization: valid browse tool calls (not poison), a production
    # nudge injected as a user turn (INPUT — never the classification signal), and the terminal
    # `{}` content response (the model's OUTPUT poison).  The parse-failure nudge that opened
    # the original spiral is retired (#1839 rerolls that draw instead), so the surviving
    # empty-content nudge stands in — the boundary being pinned is about INPUT frames as a
    # class, not about any one nudge.
    spiral = _make_db(tmp_path, "spiral")
    for _ in range(4):
        _log_prompt(
            spiral,
            messages=[{"role": "user", "content": _RETIRED_CONTINUE_NUDGE}],
            response=_tool_call_response(
                "browse", '{"queries": ["https://example.test/lake"], "extract": "the depth"}'
            ),
        )
    _log_prompt(spiral, response=_content_response("{}"))  # the terminal bare-`{}` reply
    # The bare `{}` OUTPUT trips the widened call-fragment detector, so the spiral reads as
    # pathology — and pathology OUTRANKS the timeout that actually ended the run (#1695 order),
    # so the loop no longer hides behind a bare `harness` tag.
    assert run_exhibited_pathology(spiral)
    spiral_timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(spiral, spiral_timeout, timed_out=True)
    assert spiral_timeout.cause == FailureCause.PATHOLOGY


def test_single_nudge_injected_recovery_stays_non_pathology(tmp_path) -> None:
    # The immunity boundary (#1732): a DELIBERATELY-injected recovery trigger produces exactly
    # ONE live nudge (the production recovery responding to the forced bail).  Counting nudge
    # frames would false-tag its fail path pathology; the output-only scan does not.  Built the
    # same way — a real production nudge in the INPUT — but the persisted OUTPUTS are all clean
    # (the injected bail's synthetic response never persists) and there is no bare `{}` reply.
    recovery = _make_db(tmp_path, "recovery")
    _log_prompt(
        recovery,
        messages=[{"role": "user", "content": _RETIRED_CONTINUE_NUDGE}],
        response=_tool_call_response("browse", '{"queries": ["https://example.test/lake"]}'),
    )
    _log_prompt(recovery, response=_content_response("Lake Baikal is the deepest, at 1,642 m."))
    assert not run_exhibited_pathology(recovery)  # a lone nudge frame is not a pathology signal
    # A failed injected-recovery sample that TIMED OUT stays harness, never pathology — the
    # forced trigger is invisible to the output-only scan, exactly as #1695 requires.
    recovery_timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(recovery, recovery_timeout, timed_out=True)
    assert recovery_timeout.cause == FailureCause.HARNESS


def test_result_line_renders_cause_summary(capsys) -> None:
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["poison"])
    pathological.cause = FailureCause.PATHOLOGY
    _assert_threshold("cause-case", [passed, pathological], None, intended=2)
    out = capsys.readouterr().out
    assert "RESULT [cause-case] mean 0.50 · all-pass 1/2 across 2 samples (report-only)" in out
    # The pathology sample drops out of the excluded denominator, so the honest read is 1.00.
    assert (
        "  pathology-excluded mean 1.00 (1 samples) · "
        "causes — behavioral 0 · pathology 1 · harness 0" in out
    )


# ── Regression diff: a prior run's results.jsonl → REGRESSED marks (#1693) ──

_BASELINE_RUN_ID = "run-20260719T130500-a1b2c3d4"


def _write_baseline(directory, *, case_id: str, checks: list[CheckOutcome]) -> None:
    """Write a one-case ``results.jsonl`` — the prior run the report diffs against."""
    directory.mkdir(parents=True, exist_ok=True)
    artifact = CaseArtifact(
        run_id=_BASELINE_RUN_ID,
        case_id=case_id,
        family="extractors",
        mean=1.0,
        all_pass_rate=1.0,
        samples=4,
        sample_scores=[1.0, 1.0, 1.0, 1.0],
        checks=checks,
        # An all-green prior run (#1695 fields): every sample passed, so no causes and a
        # pathology-excluded mean equal to the raw mean.
        pathology_excluded_mean=1.0,
        sample_causes=[None, None, None, None],
        cause_counts=CauseCounts(),
        timings=CaseTimings(calls=0, duration_ms=0, input_tokens=0, output_tokens=0),
    )
    (directory / "results.jsonl").write_text(artifact.model_dump_json() + "\n")


def test_baseline_flags_only_a_fully_green_flip(tmp_path) -> None:
    _write_baseline(
        tmp_path / "prior",
        case_id="watch-fern",
        checks=[
            CheckOutcome(
                label="send queued", passed=4, total=4
            ),  # fully green → a flip if it fails
            CheckOutcome(label="write happened", passed=2, total=4),  # already flaky → not a flip
        ],
    )
    baseline = load_baseline(str(tmp_path / "prior"))
    assert baseline is not None
    assert baseline.was_passing("watch-fern", "send queued")
    assert not baseline.was_passing("watch-fern", "write happened")  # 2/4 was not fully green
    assert not baseline.was_passing("watch-fern", "unknown check")  # absent → no flip
    assert not baseline.was_passing("other-case", "send queued")  # absent case → no flip
    assert baseline.run_id_for("watch-fern") == _BASELINE_RUN_ID


def test_baseline_absent_or_empty_is_none(tmp_path) -> None:
    assert load_baseline(str(tmp_path / "does-not-exist")) is None  # missing → graceful None
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "results.jsonl").write_text("\n")  # blank lines only
    assert load_baseline(str(tmp_path / "empty")) is None


def _done_bail_sample(db: Database) -> None:
    """A collector-style run that closed with ``done()`` instead of sending — the promptlog row
    carries the model's thinking, so a failed/regressed done turn can surface it."""
    done_call = {"function": {"name": "done", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "run the fern watch"},
            {"role": "assistant", "tool_calls": [done_call]},
        ],
        response={"choices": [{"message": {"tool_calls": [done_call]}}]},
        thinking="The entry is already written, so I'll close with done() rather than notify.",
    )


def test_report_marks_regressed_and_renders_thinking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_BASELINE", str(tmp_path / "prior"))
    _write_baseline(
        tmp_path / "prior",
        case_id="watch-fern",
        checks=[CheckOutcome(label="send queued", passed=4, total=4)],
    )
    db = _make_db(tmp_path)
    _done_bail_sample(db)
    result = SampleResult.graded(
        [Check("send queued", ok=False, anchor="done(", rationale="expected 1 send, saw 0")]
    )
    _write_sample_report(db, "watch-fern", 2, result=result, reply="")
    text = _sample_report_text(tmp_path, "watch-fern")
    assert text.startswith("<details><summary>sample 3 — ❌ fail ·")
    assert '| step 1 · 👤 | "run the fern watch" | ✅→❌ |' in text  # the flip on the step header
    assert "| actual | 🔧 done({}) | ✅→❌ **REGRESSED** C1 — expected 1 send, saw 0 |" in text
    assert "<details><summary>thinking — 75 chars</summary>The entry is already written" in text


def test_report_no_baseline_plain_fail_still_shows_thinking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)  # first run — nothing to flip against
    db = _make_db(tmp_path)
    _done_bail_sample(db)
    result = SampleResult.graded(
        [Check("send queued", ok=False, anchor="done(", rationale="expected 1 send, saw 0")]
    )
    _write_sample_report(db, "watch-fern", 0, result=result, reply="")
    text = _sample_report_text(tmp_path, "watch-fern")
    assert text.startswith("<details><summary>sample 1 — ❌ fail ·")
    assert "| actual | 🔧 done({}) | ❌ C1 — expected 1 send, saw 0 |" in text
    assert "<details><summary>thinking — 75 chars</summary>The entry is already written" in text
    assert "REGRESSED" not in text  # first run — nothing to flip against


def test_thinking_attaches_across_compact_and_pretty_serializations(tmp_path, monkeypatch) -> None:
    # #1735 finding 1 (HIGH): a call's thinking is captured off `promptlog.response` (the model's
    # COMPACT emission) but the transcript row is built off the NEXT prompt's `messages` (a
    # `json.dumps` of the parsed args — spaced + ASCII-escaped).  Keyed on the raw strings the two
    # NEVER matched, so real thinking silently dropped on EVERY tool call.  Built REAL-SHAPED here:
    # the two sides come from the two DIFFERENT production serializations, not one shared string
    # (the blind spot the old fixtures had — both were the same string).  Canonical keying attaches
    # the thinking; the unicode arg also renders unescaped (finding 3, LOW).
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    args = {"queries": ["café hours"]}  # a curly-apostrophe-free café: the ASCII-escape case
    # Response side (production): the model's raw COMPACT emission, persisted verbatim.
    compact = json.dumps(args, separators=(",", ":"))
    # Messages side (production reconstruction): LlmMessage.to_input_message re-dumps the PARSED
    # args via default json.dumps — spaced + ASCII-escaped — a genuinely DIFFERENT string.
    reconstructed = LlmMessage(
        role="assistant",
        tool_calls=[
            LlmToolCall(id="c1", function=LlmToolCallFunction(name="browse", arguments=args))
        ],
    ).to_input_message()
    assert compact != reconstructed["tool_calls"][0]["function"]["arguments"]  # two serializations
    _log_prompt(
        db,
        messages=[{"role": "user", "content": "when does the café open?"}, reconstructed],
        response={
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"function": {"name": "browse", "arguments": compact}}]
                    }
                }
            ]
        },
        thinking="Search the web for the café's opening hours.",
    )
    # kind="spine" also exercises finding 4 end-to-end: a case Check's class renders as `[spine]`
    # on its expected row through the real extraction path (not just report.py's pure renderer).
    result = SampleResult.graded([Check("browsed", ok=True, anchor="browse(", kind="spine")])
    # reply="" → no trailing reply action, so the ONLY 💭 row is the browse call's — a clean probe.
    _write_sample_report(db, "thinking-key", 0, result=result, reply="")
    text = _sample_report_text(tmp_path, "thinking-key")
    assert "| expected | C1 [spine]⚖ browsed |  |" in text  # finding 4: the [class] tag renders
    # The thinking sits directly ABOVE the browse call (attached, not the silent 💭 (empty) the key
    # mismatch produced), and the arg renders as a real ``é`` — not a ``\uXXXX`` escape.
    assert (
        "| 💭 | <details><summary>thinking — 44 chars</summary>"
        "Search the web for the café's opening hours.</details> |  |\n"
        '| actual | 🔧 browse({"queries": ["café hours"]}) | ✅ C1 |'
    ) in text
    assert "💭 (empty)" not in text
    assert "\\u00e9" not in text  # finding 3: the escape is gone, the character is rendered


def test_report_renders_fragile_via_user_turn_nudge(tmp_path, monkeypatch) -> None:
    # #1735 finding 2: a passing sample whose ONLY recovery was a user-turn nudge (no tool-role
    # rejection) now banners `✅ pass · fragile` and renders the nudge row as `⚠ recovery event` —
    # render and fragile-probe agree (they were decoupled at the debut).
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save X"},
            {"role": "assistant", "tool_calls": [write_call]},
            {"role": "user", "content": "Please provide your response."},  # CONTINUE_NUDGE
        ],
    )
    _write_sample_report(db, "nudge-fragile", 0, result=SampleResult.binary([]), reply="saved")
    text = _sample_report_text(tmp_path, "nudge-fragile")
    # fragile still folds whole now (#1753)
    assert text.startswith("<details><summary>sample 1 — ✅ pass · fragile ·")
    assert "| actual | 👤 *(nudge)* Please provide your response. | ⚠ recovery event |" in text


def test_report_renders_thinking_for_every_action(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    write_call = {"function": {"name": "collection_write", "arguments": "{}"}}
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "save it"},
            {"role": "assistant", "tool_calls": [write_call]},
        ],
        response={"choices": [{"message": {"tool_calls": [write_call]}}]},
        thinking="Writing the entry now.",
    )
    result = SampleResult.graded([Check("write happened", ok=True, anchor="collection_write(")])
    _write_sample_report(db, "pass-case", 0, result=result, reply="done")
    # #1725 supersedes the failed-turns-only capture: thinking renders for EVERY model action,
    # including a passing one (in its own collapsed <details> above the action). A clean pass
    # folds the whole block into a <details>.
    text = _sample_report_text(tmp_path, "pass-case")
    assert text.startswith("<details><summary>sample 1 — ✅ pass ·")
    assert (
        "| 💭 | <details><summary>thinking — 22 chars</summary>"
        "Writing the entry now.</details> |  |" in text
    )
    assert "| actual | 🔧 collection_write({}) | ✅ C1 |" in text
    assert "REGRESSED" not in text


# ── Every micro-context renders as its own actor, in ledger order (#1773) ────────────────────

_BROWSE_ARGS = '{"queries": ["lake"], "extract": "depth"}'
_BROWSE_CALL = {"function": {"name": "browse", "arguments": _BROWSE_ARGS}}
_REPLY_TEXT = "Lake Baikal — 1,642 m."


def _three_micro_context_ledger(db: Database) -> None:
    """One chat turn's promptlog exactly as production writes it (#1773): the state classifier
    draws BEFORE the chat agent, a browse-extract sub-model runs INSIDE the browse call, and the
    run-end skill labeller names the routine AFTER the reply — each with its own ``agent_name``,
    interleaved with the two main-agent rows in ledger order."""
    _log_prompt(
        db,
        agent_name=PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
        messages=[
            {"role": "system", "content": "Pick one state."},
            {"role": "user", "content": "current: idle · newest message: deepest lake?"},
        ],
        response=_content_response("STATE: idle"),
        thinking="a question, not a task",
    )
    _log_prompt(
        db,
        messages=[
            {"role": "system", "content": "You are Penny."},
            {"role": "user", "content": "deepest lake?"},
        ],
        response={"choices": [{"message": {"tool_calls": [_BROWSE_CALL]}}]},
        thinking="check a source",
    )
    _log_prompt(
        db,
        agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        messages=[
            {"role": "system", "content": "Extract one value."},
            {"role": "user", "content": "Instruction: depth · Content: 1,642 m"},
        ],
        response=_content_response("EXTRACTED: 1642"),
        thinking="the value is right there",
    )
    _log_prompt(
        db,
        messages=[
            {"role": "system", "content": "You are Penny."},
            {"role": "user", "content": "deepest lake?"},
            {"role": "assistant", "tool_calls": [_BROWSE_CALL]},
            {"role": "tool", "content": "You opened the page (browse result) · 1642"},
        ],
        response=_content_response(_REPLY_TEXT),
        thinking="the source says 1,642 m",
    )
    _log_prompt(
        db,
        agent_name=PennyConstants.SKILL_NAMING_AGENT_NAME,
        messages=[
            {"role": "system", "content": "Name the routine."},
            {"role": "user", "content": "steps: browse"},
        ],
        response=_content_response("NAME: look-up-lake-depth"),
        thinking="a generic name",
    )


def test_every_micro_context_renders_as_an_actor_in_ledger_order(tmp_path, monkeypatch) -> None:
    # #1773: `_micro_batches` admitted only browse-extract rows, so the classifier's decision and
    # the labeller's adjudication were invisible — and, being main-agent rows to the turn walk,
    # their scoped slices rendered as PHANTOM `👤 user` steps ahead of the real one.  All three
    # micro-contexts now render as named actors at the anchor their placement declares: the
    # classifier at the head of the turn it decided, the browse extraction after the call that
    # spawned it (unchanged FIFO pairing), the labeller closing the turn.  Whole-render literal.
    #
    # #1997: the sample carries only its OWN sequence.  Each actor's SYSTEM PROMPT is deposited
    # for the case document, which states each distinct one once instead of restating four of
    # them under every sample — so the second half of this test is that the extraction still
    # captures all four, at the one moment the sample's database is live to be read.
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    _three_micro_context_ledger(db)
    result = SampleResult.graded([Check("browsed", ok=True, anchor="browse(", kind="spine")])
    _case_prompts.pop("micro-actors", None)
    _write_sample_report(db, "micro-actors", 0, result=result, reply=_REPLY_TEXT)

    assert [
        (sample, prompt.context, prompt.text) for sample, prompt in _case_prompts["micro-actors"]
    ] == [
        ("sample 1", "state-classifier", "Pick one state."),
        ("sample 1", "", "You are Penny."),
        ("sample 1", "browse-extract", "Extract one value."),
        ("sample 1", "skill-namer", "Name the routine."),
    ], "every actor's prompt reaches the case document, none render under the sample"

    assert _sample_report_text(tmp_path, "micro-actors") == (
        "<details><summary>sample 1 — ✅ pass · 0s · 5 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "deepest lake?" | ✅ |\n'
        "|---|---|---|\n"
        "| expected | C1 [spine]⚖ browsed |  |\n"
        "| actual | 🧩 state-classifier ← user turn: current: idle · newest message: "
        "deepest lake? |  |\n"
        "| 💭 | thinking (state-classifier) — 22 chars: a question, not a task |  |\n"
        "| actual | 🧩 state-classifier → STATE: idle |  |\n"
        "| 💭 | check a source |  |\n"
        '| actual | 🔧 browse({"queries": ["lake"], "extract": "depth"}) | ✅ C1 |\n'
        "| actual | 🧩 browse-extract ← user turn: Instruction: depth · Content: 1,642 m |  |\n"
        "| 💭 | thinking (browse-extract) — 24 chars: the value is right there |  |\n"
        "| actual | 🧩 browse-extract → EXTRACTED: 1642 |  |\n"
        "| actual | 📥 You opened the page (browse result) · 1642 |  |\n"
        "| 💭 | <details><summary>thinking — 23 chars</summary>"
        "the source says 1,642 m</details> |  |\n"
        '| actual | 🤖 "Lake Baikal — 1,642 m." |  |\n'
        "| actual | 🧩 skill-namer ← user turn: steps: browse |  |\n"
        "| 💭 | thinking (skill-namer) — 14 chars: "
        "a generic name |  |\n"
        "| actual | 🧩 skill-namer → NAME: look-up-lake-depth |  |\n"
        "\n"
        "</details>\n"
        "\n"
    )


# ── The labeller runner's learn → render step (#1828) ─────────────────────────

# A demonstration in miniature: the round recorded the price AND a second entry the
# assistant composed itself about the page it had just read — five spots over two
# steps, including one filling two sites.  Synthetic throughout (aurora / faux-market).
_LABELLER_TARGET = "aurora-prices"
_LABELLER_UTTERANCE = "read the aurora deck 2 listing, find the current price, and remember it"
_LABELLER_ASK = "can you keep an eye on the aurora deck 2 price for me?"
_LABELLER_ELICIT = "sure — walk me through it once? what should i read and what should i save?"
_LABELLER_BROWSE = (
    "browse",
    {"queries": ["aurora deck 2 price"], "extract": "the current price"},
    "You used `browse` and here's the result: (browse result)\nEXTRACTED: $499",
    True,
)
_LABELLER_INVENTED_KEY = "aurora deck 2 page source"
_LABELLER_INVENTED_CONTENT = "Page source for the Aurora Deck 2 listing"
_LABELLER_WRITE = (
    "collection_write",
    {
        "memory": _LABELLER_TARGET,
        "entries": [
            {"key": "aurora deck 2 price", "content": "$499"},
            {"key": _LABELLER_INVENTED_KEY, "content": _LABELLER_INVENTED_CONTENT},
        ],
    },
    "You saved entries to aurora-prices: (collection_write result)\nWrote 2 entries.",
    True,
)
_LABELLER_CONVERSATION = [
    (PennyConstants.MessageDirection.INCOMING, _LABELLER_ASK),
    (PennyConstants.MessageDirection.OUTGOING, _LABELLER_ELICIT),
]


def test_labelling_input_renders_the_routine_and_maps_spots_home() -> None:
    """The labelling runner's input, WHOLE, plus the value→spot map its scoring keys on
    (#1828).

    The case drives ``label_skill`` alone, so its input has to be built from the fixture
    ledger by the SHIPPED renderer — this pins that it is, and that a helper calling
    machinery a later change removed fails HERE, inside ``make check``, instead of only
    on the deselected GPU run.

    Everything the render can vary is folded in: the elicit turn renders as ``penny:``
    (the conversation is a conversation, not a list of asks), the demonstrating message
    joins as the last ``user:`` turn, a spot filling two sites states both joined by
    ``and``, and the find-phrases section is gone with the routine naming that consumed
    it."""
    content, by_value = _labelling_input(
        [_LABELLER_BROWSE, _LABELLER_WRITE],
        _LABELLER_TARGET,
        _LABELLER_UTTERANCE,
        _LABELLER_CONVERSATION,
    )

    assert content == (
        "Conversation that led to the construction of this routine "
        "(the LAST user turn is the one that demonstrated it):\n"
        "user: can you keep an eye on the aurora deck 2 price for me?\n"
        "penny: sure — walk me through it once? what should i read and what should i save?\n"
        "user: read the aurora deck 2 listing, find the current price, and remember it\n"
        "\n"
        "Routine steps:\n"
        "1. browse(queries=[{queries}], extract={extract})\n"
        "2. collection_write(memory={memory}, entries=["
        "{'key': {queries}, 'content': the value from step 1}, "
        "{'key': {key}, 'content': {content}}])\n"
        "\n"
        "Placeholders (each currently named after the tool arg it fills):\n"
        # queries[0] and entries[0].key hold the SAME demonstrated value, so the
        # distiller collapses them into ONE spot filling both sites.
        "- queries: fills browse.queries[0] and collection_write.entries[0].key; "
        "demonstrated value: 'aurora deck 2 price'\n"
        "- extract: fills browse.extract; demonstrated value: 'the current price'\n"
        "- memory: fills collection_write.memory; demonstrated value: 'aurora-prices'\n"
        "- key: fills collection_write.entries[1].key; "
        "demonstrated value: 'aurora deck 2 page source'\n"
        "- content: fills collection_write.entries[1].content; "
        "demonstrated value: 'Page source for the Aurora Deck 2 listing'"
    )
    # Keyed by VALUE, because the case's expectations are stated in values.
    assert by_value[_LABELLER_INVENTED_KEY] == "key"
    assert by_value[_LABELLER_INVENTED_CONTENT] == "content"
    assert by_value["the current price"] == "extract"


@pytest.mark.parametrize("fixture", LABELLING_FIXTURES, ids=lambda f: f.case_id)
def test_each_labelling_case_renders_exactly_the_document_it_claims(fixture) -> None:
    """Per-case drift probe (#1828): each agreed case's fixture ledger, through the
    SHIPPED distiller and renderer, produces EXACTLY the input document the case pins —
    every spot, its arg sites, its demonstrated value, and the conversation's speakers.

    The pairs on the ticket are input/output pairs, so a fixture that has drifted from
    its input is a case measuring something nobody agreed to.  It has to fail here, in
    ``make check``, rather than after an hour of GPU time — which is also why the
    document lives beside the fixture rather than in this file: one place, so the probe
    and the live run can never check two different things."""
    content, by_value = _labelling_input(
        fixture.calls, fixture.target, fixture.utterance, fixture.conversation
    )

    assert content == fixture.rendered_input
    # And the case scores the spots that document actually offers — a leaf named in the
    # case but absent from the distilled set would score as a broken fixture at run time.
    assert sorted(by_value) == sorted(fixture.leaves)


_FRAMING_ASK = "can you keep an eye on the aurora deck 2 price for me?"


def test_framing_input_renders_the_users_turns_and_nothing_else() -> None:
    """The framer's content, WHOLE (#1830) — the surface the framing draw actually
    reads, built by the SHIPPED renderer the eval case also calls.

    It is the user's turns, one per line, and NOTHING else: no headings, no values, no
    summary of what the round did.  The assistant's turns are dropped (its replies
    describe how the round was carried out, which is exactly what a routine must not be
    named after), and the demonstrating message joins the asks as the last user turn."""
    content = build_framing_content(
        _LABELLER_UTTERANCE,
        [
            (PennyConstants.MessageDirection.INCOMING, _FRAMING_ASK),
            (PennyConstants.MessageDirection.OUTGOING, "sure — which listing did you mean?"),
        ],
    )

    assert content == (
        "can you keep an eye on the aurora deck 2 price for me?\n"
        "read the aurora deck 2 listing, find the current price, and remember it"
    )

    # The demonstrating message already inside the recent window renders ONCE, never
    # doubled — and a round with nothing but that one turn is one line.
    assert (
        build_framing_content(
            _FRAMING_ASK, [(PennyConstants.MessageDirection.INCOMING, _FRAMING_ASK)]
        )
        == _FRAMING_ASK
    )


@pytest.mark.parametrize("fixture", FRAMING_FIXTURES, ids=lambda f: f.case_id)
def test_each_framing_case_renders_exactly_the_document_it_claims(fixture) -> None:
    """Per-case drift probe (#1830): each agreed case's user turns, through the SHIPPED
    renderer, produce EXACTLY the input document the case pins.

    The pairs on the ticket are input/output pairs, so a fixture that has drifted from
    its input is a case measuring something nobody agreed to.  It has to fail here, in
    ``make check``, rather than after an hour of GPU time."""
    content = build_framing_content(
        "", [(PennyConstants.MessageDirection.INCOMING, turn) for turn in fixture.turns]
    )

    assert content == fixture.rendered_input


def _drawn(name: str, description: str, value: str = "") -> FramedParameter:
    """One drawn parameter for a scorer fixture.

    ``value`` is what the round demonstrated the parameter with (#1868) and defaults to
    empty here, because these cases score the parameter SET and the two generic checks and
    none of them reads a value.  A production draw can never carry an empty one — an
    accepted value is a literal span of the user's own words — so the default is a fixture
    convenience, never a shape the model can produce."""
    return FramedParameter(name=name, description=description, value=value)


def test_score_framing_grades_the_parameter_set_exactly() -> None:
    """The framing case's scoring over a fixture draw (#1830): each expected family
    answered by exactly one drawn parameter, nothing else asked for, and the framing
    generic — with every drawn value riding ADVISORY so a report shows verbatim what the
    model committed to.

    Semantic breadth is the families' job: ``page_to_watch`` answers the family the
    reference calls ``url``.  Name-first classification is what keeps the second
    parameter's description — which mentions a page in passing — from answering it a
    second time."""
    families = (
        ParameterFamily("url", ("url", "page", "link")),
        ParameterFamily("ticket search", ("search", "query", "event")),
    )
    signature = SkillSignature(
        name="ticket-price-watcher",
        description="watch an event's cheapest ticket price",
        parameters=(
            _drawn(
                name="page_to_watch",
                description="the listing page to check",
                value="tickets.example/spring-gala",
            ),
            _drawn(
                name="event_search",
                description="the search that finds the page",
                value="cheapest seat",
            ),
        ),
    )

    scored = _score_framing(signature, families, ("aurora", "fest"))
    assert [(check.label, check.ok, check.scored) for check in scored] == [
        ("asks for the url", True, True),
        ("asks for the ticket search", True, True),
        ("asks for nothing else", True, True),
        ("the framing is generic", True, True),
        ("the parameters are generic", True, True),
        ("named it 'ticket-price-watcher'", True, False),
        ('described it "watch an event\'s cheapest ticket price"', True, False),
        (
            "asks 'page_to_watch' — 'the listing page to check' "
            "(drawn value 'tickets.example/spring-gala')",
            True,
            False,
        ),
        (
            "asks 'event_search' — 'the search that finds the page' (drawn value 'cheapest seat')",
            True,
            False,
        ),
        # The container the framing would build (#1868) — the shipped derivation over the
        # drawn values, so the report shows the name production would use.
        (
            "derives the container "
            "'ticket-price-watcher-tickets-example-spring-gala-cheapest-seat'",
            True,
            False,
        ),
    ]

    # An EXTRA parameter is caught by the count, and a family nothing answers by its own
    # check — the two halves of "the set is exact".
    extra = signature.model_copy(
        update={
            "parameters": (
                *signature.parameters,
                _drawn(name="where_to_save", description="the collection to write to"),
            )
        }
    )
    graded = _by_label(_score_framing(extra, families, ()))
    assert graded["asks for nothing else"] == (False, "drew 3, expected 2")

    # A family nothing answers is its own miss, and one two parameters answer says so.
    missing = signature.model_copy(update={"parameters": signature.parameters[:1]})
    assert _by_label(_score_framing(missing, families, ()))["asks for the ticket search"] == (
        False,
        "no parameter answers it",
    )

    # The occasion in the framing is a structural miss, naming the words it used.
    occasional = signature.model_copy(update={"description": "watch aurora fest ticket prices"})
    framing = _by_label(_score_framing(occasional, families, ("aurora", "fest")))
    assert framing["the framing is generic"] == (False, "named the occasion: aurora, fest")

    # A refused draw fails every scored check with its reason named, never silently.
    refused = _score_framing(None, families, ("aurora",))
    assert [(check.label, check.ok) for check in refused] == [
        ("asks for the url", False),
        ("asks for the ticket search", False),
        ("asks for nothing else", False),
        ("the framing is generic", False),
        ("the parameters are generic", False),
    ]
    assert {check.rationale for check in refused} == {
        "the draw was refused — no signature came back"
    }


@pytest.mark.parametrize("fixture", BINDING_FIXTURES, ids=lambda f: f.case_id)
def test_each_binding_case_renders_exactly_the_document_it_claims(fixture) -> None:
    """Per-case drift probe (#1867): each agreed case's signature + user turns, through
    the SHIPPED renderers, produce EXACTLY the input document the case pins.

    The pairs on the ticket are input/output pairs, so a fixture that has drifted from its
    input is a case measuring something nobody agreed to.  It has to fail here, in ``make
    check``, rather than after an hour of GPU time."""
    spoken = render_spoken_turns(fixture.turns)
    content = build_binding_content(spoken, fixture.skill, fixture.intent, fixture.parameters)

    assert content == fixture.rendered_input
    # And the case scores the parameters the signature actually declares — an expectation
    # naming something the routine does not need could never be answered.
    assert [one.parameter for one in fixture.expectations] == [
        parameter.name for parameter in fixture.parameters
    ]


@pytest.mark.parametrize("fixture", EXTRACT_FIELD_FIXTURES, ids=lambda f: f.case_id)
def test_each_extract_field_case_coheres_with_the_page_it_claims(fixture) -> None:
    """Per-case coherence probe (#1942): every thing a case says its page supplies really
    is on that page, and every case declares at least one thing to look for.

    An anchor that is not on the page could never be extracted, so the case would grade
    the extractor on a fact nobody gave it — and a case declaring nothing at all grades
    nothing.  Both have to fail here, in ``make check``, rather than after GPU time."""
    assert fixture.expectations
    missing = [
        one.field
        for one in fixture.expectations
        if one.anchor and spoken_form(one.anchor) not in spoken_form(fixture.page)
    ]
    assert missing == []


def test_score_extraction_grades_the_outcome_and_each_named_field() -> None:
    """The extraction case's scoring over fixture answers (#1942): the outcome check
    reads the direction off the expectations themselves, each thing the page SUPPLIES is
    scored on its own, and what the page lacks — along with the value the draw returned —
    rides ADVISORY, so one fact is never graded twice.

    Both directions here, because the fix and its guard are one contract: a page carrying
    some of what was asked for is a read that must carry those things, and a page carrying
    none of it is honestly empty."""
    partly = _score_extraction(
        MicroContextResult(
            outcome=MicroExtractOutcome.EXTRACTED,
            value="Lantern festival draws a record crowd — https://news-alpha.example/lantern",
        ),
        [
            FieldExpectation("headline", "Lantern festival draws a record crowd"),
            FieldExpectation("link", "https://news-alpha.example/lantern"),
            FieldExpectation("summary"),
        ],
    )
    assert [(check.label, check.ok, check.scored) for check in partly] == [
        ("reads the page rather than reporting it empty", True, True),
        ("carries the headline", True, True),
        ("carries the link", True, True),
        ("the page carries no summary", True, False),
        (
            "extracted 'Lantern festival draws a record crowd — "
            "https://news-alpha.example/lantern'",
            True,
            False,
        ),
    ]

    # The same page answered as though it were empty — the regression itself: the outcome
    # check fails, and so does every thing the page did in fact carry.
    refused = _score_extraction(
        MicroContextResult(
            outcome=MicroExtractOutcome.NOT_PRESENT, reason="no summaries are given."
        ),
        [
            FieldExpectation("headline", "Lantern festival draws a record crowd"),
            FieldExpectation("summary"),
        ],
    )
    assert [(check.label, check.ok) for check in refused if check.scored] == [
        ("reads the page rather than reporting it empty", False),
        ("carries the headline", False),
    ]
    assert refused[0].rationale == "came back not_present"

    # And a page carrying none of it: the honest absence is the whole contract there, so
    # the outcome check is the only graded one and both gaps render beside it.
    absent = _score_extraction(
        MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason="no prices are listed."),
        [FieldExpectation("closing price"), FieldExpectation("ticker symbol")],
    )
    assert [(check.label, check.ok, check.scored) for check in absent] == [
        ("reports the page carries none of it", True, True),
        ("the page carries no closing price", True, False),
        ("the page carries no ticker symbol", True, False),
        ("not_present: 'no prices are listed.'", True, False),
    ]


def test_score_binding_grades_each_declared_parameter_and_the_terms() -> None:
    """The binding case's scoring over a fixture answer (#1867): one check per declared
    parameter — bound to a span carrying the value the ask supplies, or named missing when
    it supplies none — plus the structural check that no job TERM rode into a value, with
    every drawn value riding ADVISORY.

    The anchor is compared through the production ``spoken_form``, so a value that kept
    the scheme and a value that dropped it both answer the same expectation: which span of
    the ask supplies a value has a little play in it, and a scorer demanding one exact
    string would be answering for the draw."""
    expectations = (
        BoundExpectation("url", "northpier.example/departures"),
        BoundExpectation("keyword", "dawn sailing"),
    )
    bound = BoundValues(
        values={"url": "https://northpier.example/departures", "keyword": "the dawn sailing"}
    )

    scored = _score_binding(bound, expectations, ("every morning",))
    assert [(check.label, check.ok, check.scored) for check in scored] == [
        ("binds the url", True, True),
        ("binds the keyword", True, True),
        ("no job term landed in a value", True, True),
        ("bound 'url' = 'https://northpier.example/departures'", True, False),
        ("bound 'keyword' = 'the dawn sailing'", True, False),
    ]

    # A value carrying the wrong span is its own miss, quoting what came back.
    wrong = BoundValues(values={"url": "the north pier timetable", "keyword": "dawn sailing"})
    assert _by_label(_score_binding(wrong, expectations, ()))["binds the url"] == (
        False,
        "bound 'the north pier timetable', not the value the ask supplies",
    )

    # A term swept into a value is the structural miss, naming the value and the term.
    swept = BoundValues(
        values={
            "url": "https://northpier.example/departures",
            "keyword": "dawn sailing every morning",
        }
    )
    graded = _by_label(_score_binding(swept, expectations, ("every morning",)))
    assert graded["no job term landed in a value"] == (
        False,
        "carried the terms: keyword (every morning)",
    )

    # The SHORTFALL direction: an expectation with no anchor wants the missing outcome,
    # and a value there is a guess the rationale quotes.
    shortfall = (expectations[0], BoundExpectation("keyword"))
    reported = MissingParameters(
        names=("keyword",), values={"url": "https://northpier.example/departures"}
    )
    # An ask stating no terms has nothing to check, so that one is NOT-APPLICABLE rather
    # than a free pass — rendered, out of the denominator.
    assert [
        (check.label, check.ok, check.ignored) for check in _score_binding(reported, shortfall, ())
    ] == [
        ("binds the url", True, False),
        ("reports the keyword missing", True, False),
        ("no job term landed in a value", True, True),
        ("bound 'url' = 'https://northpier.example/departures'", True, False),
        ("reported missing: keyword", True, False),
    ]
    guessed = _by_label(_score_binding(bound, shortfall, ()))
    assert guessed["reports the keyword missing"] == (False, "bound it to 'the dawn sailing'")

    # A refused draw fails every scored check with its reason named, never silently.
    refused = _score_binding(None, expectations, ("every morning",))
    assert [(check.label, check.ok) for check in refused] == [
        ("binds the url", False),
        ("binds the keyword", False),
        ("no job term landed in a value", False),
    ]
    assert {check.rationale for check in refused} == {"the draw was refused — no binding came back"}


def test_the_page_family_classifies_by_name_only() -> None:
    """The description fallback is dropped for the page/url family (#1830, the code
    owner's ruling on the first run).

    The motivating draw: a `city` parameter whose description said *name of the location
    on the site to read*.  Two IDENTICAL draws scored opposite ways, because that
    passing mention of the site could promote one of them to the page — the scorer
    answering for a draw that never named a page at all.  A page is NAMED as one.

    The fallback stays for every other family, which is what lets a well-judged name the
    tokens don't anticipate still land via its description."""
    page = ParameterFamily("url", ("url", "page", "site"), name_only=True)
    search = ParameterFamily("ticket search", ("search", "query"), name_only=False)
    city = _drawn(name="city", description="name of the location on the site to read")

    named_city = SkillSignature(
        name="temperature-recorder", description="record a daily high", parameters=(city,)
    )
    graded = _by_label(_score_framing(named_city, (page,), ()))
    assert graded["asks for the url"] == (False, "no parameter answers it")
    assert graded["asks for nothing else"] == (True, None)

    # A non-page family still reads its description when no name matched anywhere.
    by_description = SkillSignature(
        name="ticket-price-watcher",
        description="watch an event's cheapest ticket price",
        parameters=(_drawn(name="whats_on", description="the search to run"),),
    )
    assert _by_label(_score_framing(by_description, (search,), ()))[
        "asks for the ticket search"
    ] == (
        True,
        None,
    )


def test_a_digit_suffixed_ordinal_pair_classifies_as_the_two_families() -> None:
    """A trailing digit is its own token (#1830, the code owner's ruling on the second
    run): ``site1``/``site2`` is one of the natural ways to write an ordinal pair, and
    the run scored two CORRECT draws as family misses because the scorer read each name
    as a single opaque word.  The families are unchanged; what changed is that the
    tokenizer can see the ordinal that was always there."""
    families = (
        ParameterFamily("first source", ("first", "one", "1", "primary")),
        ParameterFamily("second source", ("second", "two", "2", "secondary")),
    )
    signature = SkillSignature(
        name="headline-collector",
        description="collect the top headline from each front page it is pointed at",
        parameters=(
            _drawn(name="site1", description="the first front page to read"),
            _drawn(name="site2", description="the second front page to read"),
        ),
    )

    graded = _by_label(_score_framing(signature, families, ("citydesk", "harborpost")))
    assert graded["asks for the first source"] == (True, None)
    assert graded["asks for the second source"] == (True, None)
    assert graded["asks for nothing else"] == (True, None)


def test_a_letter_suffixed_ordinal_pair_classifies_as_the_two_families() -> None:
    """Lettering is the other natural way to write an ordinal pair, and it cost two
    correct draws exactly what digits once did.

    The motivating sample drew ``url_a`` / ``url_b`` — two distinct, generic, scalar
    names satisfying the prompt's tell-them-apart rule — and BOTH ordinal family checks
    read "no parameter answers it" while the count and generic checks passed.  Correct
    behaviour, scored wrong.  A trailing single letter is now the position it holds in
    the alphabet, so the pair lands on the families that already carry ``1``/``2``: no
    family gained a token, so nothing that missed before can start matching now."""
    families = (
        ParameterFamily("first source", ("first", "one", "1", "primary")),
        ParameterFamily("second source", ("second", "two", "2", "secondary")),
    )
    signature = SkillSignature(
        name="headline-collector",
        description="collect the top headline from each front page it is pointed at",
        parameters=(
            _drawn(name="url_a", description="the first front page to read"),
            _drawn(name="url_b", description="the second front page to read"),
        ),
    )

    graded = _by_label(_score_framing(signature, families, ("citydesk", "harborpost")))
    assert graded["asks for the first source"] == (True, None)
    assert graded["asks for the second source"] == (True, None)
    assert graded["asks for nothing else"] == (True, None)
    assert graded["the parameters are generic"] == (True, None)

    # Everything that classified before still does — the digit pair, the spelled ordinal,
    # and the underscored digit, each landing on the same family as always.
    for first, second in (("site1", "site2"), ("first_site", "second_site"), ("url_1", "url_2")):
        unchanged = signature.model_copy(
            update={
                "parameters": (
                    _drawn(name=first, description="the first front page to read"),
                    _drawn(name=second, description="the second front page to read"),
                )
            }
        )
        still = _by_label(_score_framing(unchanged, families, ()))
        assert still["asks for the first source"] == (True, None), first
        assert still["asks for the second source"] == (True, None), second


def test_a_letter_reads_as_an_ordinal_only_as_a_suffix_on_a_name() -> None:
    """The two guard directions of the letter rule (#1830).

    A letter is an ordinal only where somebody carved it off an identifier — so a
    DESCRIPTION is untouched (it is prose, where ``a`` is an article; reading it as an
    ordinal would file most descriptions ever written under the first family), and a
    name that is ONLY a letter is left alone (a suffix needs something to be suffixed
    to).  And a case expecting ONE family counts a ``site_a`` once, not twice: the
    ordinal rides alongside the name's own tokens, it does not replace them."""
    ordinal = ParameterFamily("first source", ("first", "one", "1", "primary"))
    page = ParameterFamily("url", ("url", "page", "site"), name_only=True)
    framing = {"name": "headline-collector", "description": "collect a page's top headline"}

    # A description full of articles answers the ordinal family through neither pass.
    prose = SkillSignature(
        **framing,
        parameters=(_drawn(name="whats_on", description="a page to read a headline off"),),
    )
    assert _by_label(_score_framing(prose, (ordinal,), ()))["asks for the first source"] == (
        False,
        "no parameter answers it",
    )

    # A name that is only a letter is a name nobody enumerated, not the first of anything.
    bare = SkillSignature(**framing, parameters=(_drawn(name="a", description="a page"),))
    assert _by_label(_score_framing(bare, (ordinal,), ()))["asks for the first source"] == (
        False,
        "no parameter answers it",
    )

    # One expected family, one letter-suffixed name: answered once.
    single = SkillSignature(
        **framing,
        parameters=(_drawn(name="site_a", description="the front page to read"),),
    )
    graded = _by_label(_score_framing(single, (page,), ()))
    assert graded["asks for the url"] == (True, None)
    assert graded["asks for nothing else"] == (True, None)


def _required(*pairs: tuple[str, str | None]) -> list[SkillParameter]:
    """The learned skill's required parameters, as the interface check reads them."""
    return [
        SkillParameter(name=name, required=True, description=description)
        for name, description in pairs
    ]


def test_the_learn_interface_accepts_the_page_plus_at_most_the_found_thing() -> None:
    """The elicit → learn interface check under the code owner's leeway ruling (2026-08-05).

    The audited draw that prompted it asked for a `search_phrase` beside the url, and the
    thinking read "the late sailing" out of both of the user's own turns — the
    enumerate-then-filter rule applied CORRECTLY, so scoring it a miss would be the scorer
    marking a sound draw wrong.  The page stays mandatory (a routine nobody can point
    anywhere can only repeat its demonstration) and the leeway is exactly one: a second
    parameter of another kind is the invention the rule exists to stop, and a third is one
    however it is named.  The rationale names WHICH reading was drawn, on the pass as well
    as the miss."""
    alone = _interface_check(_required(("url", "the listing page to check")))
    assert (alone.ok, alone.rationale) == (True, "url alone")

    leeway = _interface_check(
        _required(
            ("url", "the timetable page to read"),
            ("search_phrase", "the line to look for on it"),
        )
    )
    assert (leeway.ok, leeway.rationale) == (True, "url + search_phrase (user-named)")

    # A second parameter of any OTHER kind is the invention, whatever it is called.
    invented = _interface_check(
        _required(("url", "the page to read"), ("frequency", "how often to check it"))
    )
    assert (invented.ok, invented.rationale) == (
        False,
        "rejected: frequency answers no accepted family",
    )

    # A third fails even when the first two are the accepted pair.
    third = _interface_check(
        _required(
            ("url", "the page to read"),
            ("search_phrase", "the line to look for"),
            ("collection", "where to keep it"),
        )
    )
    assert third.ok is False

    # The page half is MANDATORY: a found-thing on its own is not an interface.
    orphan = _interface_check(_required(("search_phrase", "the line to look for")))
    assert (orphan.ok, orphan.rationale) == (False, "0 answer the page: []")

    # An accepted parameter still has to say what to supply.
    undescribed = _interface_check(_required(("url", None)))
    assert (undescribed.ok, undescribed.rationale) == (False, "carries no description: url")


def test_a_parameter_named_after_the_occasion_is_not_generic() -> None:
    """The generic check reaches the PARAMETER lines too (#1830) — the enforcement half
    of the parameter-line contract.

    The motivating draw: `citydesk_url — citydesk.example/front`, which names the spot
    after the site it was taught on and then writes that occasion's value where the
    what-to-supply belongs.  It is a routine that can only ever be pointed back at the
    page it learned from.  The same spot written generically — `first_site — the first
    front page to read` — passes, and so does the framing check either way, which is
    why this is its own check rather than a widening of that one."""
    families = (ParameterFamily("first source", ("first", "one", "1", "primary")),)
    instance = ("citydesk", "harborpost")
    framing = {
        "name": "headline-collector",
        "description": "collect the top headline from a news front page",
    }

    occasional = SkillSignature(
        **framing,
        parameters=(_drawn(name="citydesk_url", description="citydesk.example/front"),),
    )
    graded = _by_label(_score_framing(occasional, families, instance))
    assert graded["the parameters are generic"] == (
        False,
        "named the occasion: citydesk_url (citydesk)",
    )
    assert graded["the framing is generic"] == (True, None), "the framing itself is clean"

    generic = SkillSignature(
        **framing,
        parameters=(_drawn(name="first_site", description="the first front page to read"),),
    )
    assert _by_label(_score_framing(generic, families, instance))["the parameters are generic"] == (
        True,
        None,
    )


def test_an_example_clause_is_garnish_not_substance() -> None:
    """An appended example of this occasion's value is STRIPPED before the generic scan
    (#1830, the code owner's ruling on the fourth run).

    The run failed two lines whose substance was exactly right — the thinking drafted
    them exampleless and the `(e.g., …)` appeared only at transcription — so scoring the
    clause marked correct work wrong.  What must still fail is the line's substance: an
    instance token in the NAME, or the value standing as the whole description.

    The third shape is the one that separates the two rulings: a `location` parameter
    with its example stripped is generically WORDED, and still misses the page family,
    because its defect is the type it asks for and not the garnish it wore."""
    page = ParameterFamily("url", ("url", "page", "site", "weather"), name_only=True)

    # Generic substance wearing an example of the occasion — the clause goes.
    garnished = SkillSignature(
        name="temperature-recorder",
        description="record the daily high temperature from a weather page",
        parameters=(
            _drawn(
                name="site_url",
                description=(
                    "the URL to query for the high temperature (e.g., weather.example/lisbon)"
                ),
            ),
        ),
    )
    graded = _by_label(_score_framing(garnished, (page,), ("lisbon",)))
    assert graded["the parameters are generic"] == (True, None)
    assert graded["asks for the url"] == (True, None)

    # The occasion IN the name, and the value standing AS the description: substance.
    echoed = SkillSignature(
        name="headline-collector",
        description="collect the top headline from a news front page",
        parameters=(_drawn(name="citydesk_url", description="citydesk.example/front"),),
    )
    assert _by_label(_score_framing(echoed, (page,), ("citydesk", "harborpost")))[
        "the parameters are generic"
    ] == (False, "named the occasion: citydesk_url (citydesk)")

    # Stripped and generic, but the WRONG KIND of thing — a piece decomposed out of the
    # value the user actually gave, which is the type drift, not the garnish.
    decomposed = SkillSignature(
        name="temperature-recorder",
        description="record the daily high temperature from a weather page",
        parameters=(
            _drawn(
                name="location",
                description='the geographic location to look up (e.g., "lisbon")',
            ),
        ),
    )
    drifted = _by_label(_score_framing(decomposed, (page,), ("lisbon",)))
    assert drifted["the parameters are generic"] == (True, None), "the garnish is not the miss"
    assert drifted["asks for the url"] == (False, "no parameter answers it")


def test_example_clauses_are_stripped_in_their_observed_forms() -> None:
    """The clause shapes a draw actually writes, all reduced to the instruction alone —
    parenthesized or trailing, with or without the comma and the dots."""
    assert _without_examples("the plot to log (e.g., 17)") == "the plot to log"
    assert _without_examples("the plot to log (eg 17)") == "the plot to log"
    assert _without_examples("the plot to log (for example 17)") == "the plot to log"
    assert _without_examples("the plot to log, e.g. 17") == "the plot to log"
    assert _without_examples("the plot to log — such as 17") == "the plot to log"
    # A line with no garnish is untouched, and a word merely containing the letters is
    # not a lead-in ("eggs" is not "e.g.").
    assert _without_examples("which plot in the allotment to log") == (
        "which plot in the allotment to log"
    )
    assert _without_examples("the plot whose eggs are counted") == "the plot whose eggs are counted"


def _by_label(checks) -> dict[str, tuple[bool, str | None]]:
    """A scored list indexed by check label — the diff-join key each check is named
    for."""
    return {check.label: (check.ok, check.rationale) for check in checks}


def test_score_labelling_grades_each_spot_and_carries_the_labels_advisory() -> None:
    """The labelling case's scoring over a fixture draw (#1828): per offered spot, a
    line came back · its name hardens to a usable binding key · it is not the arg name
    handed back · its description says what belongs there.  Every drawn label then rides
    ADVISORY, so a report shows verbatim what the model committed to."""
    by_value = {"the current price": "extract", _LABELLER_INVENTED_KEY: "key"}
    labels = SkillLabels(
        labels={
            "extract": LeafLabel(name="value_to_find", description="what to pull off the page"),
            "key": LeafLabel(name="entry key", description="what to call the entry it saves"),
        }
    )

    scored = _score_labelling(labels, by_value, list(by_value), (), "")
    assert [(check.label, check.ok, check.scored) for check in scored] == [
        ("a line came back: 'the current price'", True, True),
        ("name is a usable binding key: 'the current price'", True, True),
        ("name is not the arg name: 'the current price'", True, True),
        ("description says what belongs there: 'the current price'", True, True),
        ("a line came back: 'aurora deck 2 page source'", True, True),
        ("name is a usable binding key: 'aurora deck 2 page source'", True, True),
        ("name is not the arg name: 'aurora deck 2 page source'", True, True),
        ("description says what belongs there: 'aurora deck 2 page source'", True, True),
        ("drew extract: 'value_to_find' — 'what to pull off the page'", True, False),
        ("drew key: 'entry key' — 'what to call the entry it saves'", True, False),
    ]

    # A FAILED draw is one miss per spot, not four: the three checks that depend on a
    # line are NOT APPLICABLE without one, so a draw that never landed reads as one
    # miss each rather than as four separate failures.  Since #1828 that is the only
    # shape a missing line arrives in — an accepted draw covers every offered spot, so
    # there is no partial map for the scorer to see.
    failed = _score_labelling(None, by_value, list(by_value), (), "")
    assert [(check.label, check.ok, check.ignored) for check in failed[:5]] == [
        ("a line came back: 'the current price'", False, False),
        ("name is a usable binding key: 'the current price'", True, True),
        ("name is not the arg name: 'the current price'", True, True),
        ("description says what belongs there: 'the current price'", True, True),
        ("a line came back: 'aurora deck 2 page source'", False, False),
    ]

    # A line that stopped after its name covers its spot (the grammar's one optional
    # field), so it is accepted — and the missing description is its own miss.
    nameless = SkillLabels(
        labels={
            "extract": LeafLabel(name="value_to_find", description=""),
            "key": LeafLabel(name="entry_key", description="what to call it"),
        }
    )
    quiet = _score_labelling(nameless, by_value, list(by_value), (), "")
    assert [(check.label, check.ok) for check in quiet if not check.ok] == [
        ("description says what belongs there: 'the current price'", False)
    ]

    # The name handed back is the arg name it was shown — the spot was described, not
    # named — and a name that hardens to nothing could never be a binding key.
    echoed = SkillLabels(
        labels={
            "extract": LeafLabel(name="Extract", description="what to pull"),
            "key": LeafLabel(name="!!", description="a label"),
        }
    )
    lazy = _score_labelling(echoed, by_value, list(by_value), (), "")
    assert [(check.label, check.ok) for check in lazy if not check.ok] == [
        ("name is not the arg name: 'the current price'", False),
        ("name is a usable binding key: 'aurora deck 2 page source'", False),
    ]

    # A value the case asserts but the ledger never distilled is a BROKEN FIXTURE, not
    # a naming miss: it fails loudly naming the value, because a drifted fixture
    # scoring green is a case measuring nothing.
    drifted = _score_labelling(labels, by_value, ["a value nothing distilled"], (), "")
    assert [(check.label, check.ok) for check in drifted][:1] == [
        ("a line came back: 'a value nothing distilled'", False)
    ]
    assert "not among the distilled placeholders" in (drifted[0].rationale or "")


def test_score_labelling_reads_the_two_structural_claims() -> None:
    """The two claims only some cases make (#1828).

    Two spots on one argument must draw DIFFERENT names — one name for both loses which
    site is which.  And a spot filling TWO sites must resolve to exactly one label:
    splitting it either repeats the spot's current name or keys a line to a spot nobody
    offered, and both are coverage violations the validator refuses (#1828), so a split
    reaches the scorer as the shared spot having no label at all."""
    two_sources = {"citydesk.example/front": "queries", "harborpost.example/front": "queries-2"}
    pair = ("citydesk.example/front", "harborpost.example/front")
    collapsed = SkillLabels(
        labels={
            "queries": LeafLabel(name="news_page", description="a front page"),
            "queries-2": LeafLabel(name="news page", description="the other front page"),
        }
    )

    [check] = [c for c in _score_labelling(collapsed, two_sources, (), [pair], "") if c.scored]
    assert (check.label, check.ok, check.rationale) == (
        "distinct names: 'citydesk.example/front' vs 'harborpost.example/front'",
        False,
        "both drew 'news_page'",
    )

    told_apart = SkillLabels(
        labels={
            "queries": LeafLabel(name="first_news_site", description="the first front page"),
            "queries-2": LeafLabel(name="second_news_site", description="the second front page"),
        }
    )
    [ok_check] = [c for c in _score_labelling(told_apart, two_sources, (), [pair], "") if c.scored]
    assert ok_check.ok

    shared = {"VLT": "queries", "the share price": "extract"}
    covered = SkillLabels(
        labels={
            "queries": LeafLabel(name="ticker_symbol", description="the symbol to look up"),
            "extract": LeafLabel(name="value_to_find", description="what to pull"),
        }
    )
    [held] = [c for c in _score_labelling(covered, shared, (), (), "VLT") if c.scored]
    assert (held.label, held.ok) == ("one label for the shared spot: 'VLT'", True)

    # A split never reaches an accepted draw, so it arrives as a failed one.
    [claim] = [c for c in _score_labelling(None, shared, (), (), "VLT") if c.scored]
    assert (claim.label, claim.ok, claim.rationale) == (
        "one label for the shared spot: 'VLT'",
        False,
        "the shared spot drew no single line",
    )


# ── The chat-reply scorer's two reply reads (#1919) ──────────────────────────
#
# Both are pure functions over a reply string, and both were tuned against MEASURED
# replies — so they belong inside `make check`, where the `eval` marker cannot deselect
# them, rather than only under the GPU run they score.  Each is pinned in BOTH directions:
# the reply that must now pass, and the one that must still fail.


def test_a_title_typed_with_any_space_still_reads_as_that_title() -> None:
    """A stored title the model typed with a NARROW NO-BREAK space is that title.

    Two measured from-store samples named a seeded title and were scored as naming
    nothing, because the model put U+202F between the words where the token has a plain
    space.  The fold is by whitespace CATEGORY, so the widths it has not met yet are
    covered too — while a reply that genuinely names nothing stored still fails, which is
    the half that keeps the check worth having."""
    title = _STORED_TITLES[1]
    assert title == "quarry hollow"

    for space in (" ", " ", " ", " ", "　", "  ", "\n"):
        typed = f"you had **Quarry{space}Hollow** on your shortlist 🎲"
        assert _carries(typed, title), f"U+{ord(space[0]):04X} broke the title read"

    assert not _carries("you're into board games and trail running 🎲", title)


def test_the_duplicate_reply_read_fails_a_fresh_claim_not_a_neutral_one() -> None:
    """A no-op save is honest as long as the reply does not claim it just happened.

    Measured, a sample that had SEEN the write gate's already-there result confirmed the
    present state neutrally ("it's logged in your interests") and was scored a miss for
    reaching for the wrong words — a false negative, since it claimed nothing untrue.  So
    the read now fails only a claim that THIS turn recorded it, and the genuine
    fresh-claim replies must keep failing, which is real product signal."""
    for honest in (
        "already had that one — sea kayaking is on your list 🛶",
        "got it — sea kayaking is logged in your interests 🛶",
        "yep, sea kayaking 🛶",
        "i just looked and it's already logged 🛶",
    ):
        assert _honest_about_the_duplicate(honest) == (True, None), honest

    fresh, claimed = _honest_about_the_duplicate("nice — sea kayaking is now safely logged 🛶")
    assert (fresh, claimed) == (False, "claimed 'now safely logged'")

    fresh, claimed = _honest_about_the_duplicate("sea kayaking is officially on the radar 🛶")
    assert (fresh, claimed) == (False, "claimed 'officially on the radar'")
