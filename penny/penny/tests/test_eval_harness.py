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
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pytest

# Importing the memory-tools module registers those tools (``Tool.__init_subclass__``) so
# ``Tool.format_result`` dispatches their real ``to_result_narration`` — the rejection-probe
# tests below build frames from the PRODUCTION templates, never hand-invented text.
import penny.tools.memory_tools  # noqa: F401  (imported for registration side effect)
from penny.agents.collector import Collector
from penny.constants import (
    MutationActor,
    MutationEntityType,
    PennyConstants,
    RunOutcome,
    TransitionCause,
)
from penny.conversation_machine import (
    ConversationState,
    RoundShortfall,
    StateDecision,
    build_snapshot,
    presented_edges,
    render_classifier_content,
)
from penny.database import Database
from penny.database.memory import MemoryType
from penny.database.models import MemoryRow, PromptLog, Skill
from penny.database.skills import (
    SkillParameter,
    build_binding_content,
    render_spoken_turns,
    slug_skill_name,
)
from penny.llm.models import (
    LlmMessage,
    LlmResponse,
    LlmToolCall,
    LlmToolCallFunction,
    strip_harmony_control_tokens,
)
from penny.notification import NOTIFICATION_NOTES, NotificationOutcome
from penny.program import program_calls
from penny.skill_extraction import build_framing_content
from penny.tests import eval as eval_package
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.binder.test_skill_binding import (
    _MISSING_KEYWORD,
    _TWO_PARAMETERS,
    MISSING_KEYWORD_ARMS,
)
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
    ANSWERING_CASES,
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
from penny.tests.eval.classifier.test_state_classifier import (
    PASSING_MENTION_ARMS,
    SEEDED_SKILLS,
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
from penny.tests.eval.collector.test_watch_cycles import (
    _BASELINE_AMOUNT as WATCH_BASELINE_AMOUNT,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _BASELINE_PRICE as WATCH_BASELINE_PRICE,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _CONTAINER as WATCH_CONTAINER,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _DATUM as WATCH_DATUM,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _MOVED_AMOUNT as WATCH_MOVED_AMOUNT,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _MOVED_DATUM as WATCH_MOVED_DATUM,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _MOVED_PRICE as WATCH_MOVED_PRICE,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _PROGRAM_CALLS as WATCH_PROGRAM_CALLS,
)
from penny.tests.eval.collector.test_watch_cycles import (
    CASES as WATCH_CASES,
)
from penny.tests.eval.collector.test_watch_cycles import (
    READINGS as WATCH_READINGS,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _arm as watch_arm,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _extract_slot as watch_extract_slot,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _program as watch_program,
)
from penny.tests.eval.conftest import (
    _ACTOR,
    BIND_MISSING,
    BIND_OUTCOME,
    CLASSIFY_OUTCOME,
    CLASSIFY_SKILL,
    CLASSIFY_STATE,
    CYCLE_DEAD,
    EVAL_SEED_AUTHOR,
    EXTRACT_OUTCOME,
    EXTRACT_REASON,
    EXTRACT_VALUE,
    FRAME_DESCRIPTION,
    FRAME_NAME,
    FRAME_PARAMETERS,
    INJECTION_NEVER_FIRED,
    MUTATION_HISTORY_WINDOW,
    NO_CYCLE,
    NO_DRAW,
    NO_MEASURED_TURN,
    NO_REPLY,
    PENNY_LOGGER,
    UNSTATED,
    BoundExpectation,
    Check,
    CycleCall,
    CycleObservation,
    FieldExpectation,
    ParameterFamily,
    ParkedRound,
    SampleResult,
    _assert_threshold,
    _bail_fired_check,
    _binding_output,
    _case_prompts,
    _chat_tool_sequence,
    _classification_output,
    _classifier_snapshot,
    _cycle_recovered_check,
    _cycle_shape,
    _cycles_exclusion,
    _draw_exclusion,
    _exclusion,
    _extraction_output,
    _flush_sample_blocks,
    _frame_attributes_to,
    _framing_output,
    _guarded_graded,
    _guarded_injector,
    _held_entries,
    _InjectTextBail,
    _labelling_input,
    _labelling_output,
    _machine_walk,
    _mechanism_records,
    _observe_binding,
    _observe_classification,
    _observe_extraction,
    _observe_framing,
    _observe_labelling,
    _PendingCase,
    _Perf,
    _refuse_binding_off_request,
    _refuse_unscorable,
    _registry_shortfall,
    _sample_db_path,
    _sample_turns,
    _score_binding,
    _score_extraction,
    _score_framing,
    _score_labelling,
    _scorer_is_graded,
    _stamp_cause,
    _stated_pass_rate,
    _stored_entries,
    _turn_kind,
    _without_examples,
    _write_classifier_report,
    _write_sample_report,
    bound_value_field,
    collection_entries,
    continue_nudge_fired,
    count_tool_calls,
    cycle_script,
    draw_rerolled,
    env_seconds,
    frame_parameter_name,
    frame_parameter_says,
    is_seeded_run,
    label_name_field,
    label_says_field,
    live_prompt_perf,
    measured_turn_ran,
    routing_clean,
    run_exhibited_pathology,
    sample_is_fragile,
    sample_log_path,
    sample_logging,
    sample_number,
    seed_world_stores,
    seeded_run_id,
    tool_call_name,
    tool_call_rejected,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.conftest import (
    _stated_behaviour as stated_behaviour,
)

# ``collection_names`` under an alias: the dispatch world exports one of its own (a sorted
# LIST of what a case seeded), and this is the observation path's snapshot of every memory
# name in the store — two different questions that would silently swap under one name.
from penny.tests.eval.conftest import (
    collection_names as memory_names_now,
)
from penny.tests.eval.extractor.test_browse_extract_fields import FIXTURES as EXTRACT_FIELD_FIXTURES
from penny.tests.eval.framer.test_skill_framing import FIXTURES as FRAMING_FIXTURES
from penny.tests.eval.framer.test_skill_framing import TICKER_ARMS
from penny.tests.eval.labeller.test_skill_labelling import (
    _TWO_SOURCES,
    OFFERED_SPOTS,
    TWO_SOURCES_ARMS,
)
from penny.tests.eval.labeller.test_skill_labelling import FIXTURES as LABELLING_FIXTURES
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report
from penny.tests.eval.utils.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckOutcome,
    FailureCause,
    build_case_artifact,
)
from penny.tests.eval.utils.assertions import Cohort, assertion_rows
from penny.tests.eval.utils.baseline import load_baseline
from penny.tests.eval.utils.cohort import SampleObservation, unsourced_specifics
from penny.tests.eval.utils.dispatch_world import assert_no_collections, collection_names
from penny.tests.eval.utils.fixtures import (
    BOARD_GAMES,
    ENACTING_TOOLS,
    EXTRACT_TAGGED_ANSWER,
    EXTRACT_TAGGED_ASK,
    EXTRACT_TAGGED_BYLINE,
    EXTRACT_TAGGED_PAGE,
    EXTRACT_UNSOURCED_NAME,
    LISTING_URL,
    CannedPage,
    SynthCollection,
)
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
from penny.tests.eval.utils.worlds import World
from penny.tests.schema_template import migrated_db, schema_only_db

# Production's own rule for a message worth delivering, read from where the send path and the
# run-health classifier read it — so the tripwire below pins the agreed reference replies
# against the same definition the ported claim answers with, never a second copy of it.
from penny.text_validity import half_formed_send_reason
from penny.tools.base import FRAMEWORK_NARRATION_INVALID_ARGS, Tool
from penny.tools.collection_instantiation import _LINE_ESCAPE
from penny.tools.micro_context import (
    _BARE_CONTENT_TEMPLATE,
    _USER_TEMPLATE,
    BIND_SKILL_SYSTEM_PROMPT,
    MICRO_CONTEXT_SYSTEM_PROMPT,
    NOT_PRESENT_TAG,
    SKILL_FRAME_SYSTEM_PROMPT,
    SKILL_NAMING_SYSTEM_PROMPT,
    STATE_CLASSIFIER_SYSTEM_PROMPT,
    BoundValues,
    FramedParameter,
    LeafLabel,
    MicroContext,
    MicroContextResult,
    MicroExtractOutcome,
    MissingParameters,
    SkillLabels,
    SkillSignature,
    StateDrawOutcome,
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


def test_a_sample_s_penny_log_lands_beside_its_db(tmp_path, monkeypatch) -> None:
    """The per-sample log file (#1909): every line the penny loggers emit while a sample
    runs is written beside that sample's DB, under the same stem.

    A failing model call writes no promptlog row — it raises before the client's persist
    step — so what the run said about the failure exists only as logger output, which
    pytest captures and then discards for every sample that passes.  Unfiltered by
    design: a DEBUG line lands as readily as the WARNING the client logs when a chat
    call fails.  Scoped to one sample, so the handler and the raised level are both gone
    afterwards and nothing leaks into the next sample or the rest of the suite.

    The stem is the sample's own NUMBER — the one the report calls it by (#2076).  The two
    were once different numbers for the same sample: a report naming ``-11`` sent a reader
    to ``-10.db``, the NEXT sample's database, which had run cleanly — so a correct
    exclusion read as a harness defect and was filed as one.  Pinned from the index a runner
    actually holds, both ends, so the report's name and the file it points at cannot drift
    apart again."""
    monkeypatch.delenv("EVAL_REPORT_DIR", raising=False)
    assert _sample_db_path(tmp_path, "watch-a-page", 0) == str(tmp_path / "watch-a-page-1.db")
    assert _sample_db_path(tmp_path, "watch-a-page", 10) == str(tmp_path / "watch-a-page-11.db")
    assert sample_log_path(_sample_db_path(tmp_path, "watch-a-page", 10)).name == (
        f"watch-a-page-{sample_number(10)}.log"
    ), "the log follows the DB's stem, so it carries the report's number too"
    assert _sample_db_path(tmp_path, "watch-a-page", 10, attempt=1) == str(
        tmp_path / "watch-a-page-11-attempt2.db"
    ), "a re-driven sample keeps its own file and its own number"

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


def test_a_sample_is_numbered_in_exactly_one_place() -> None:
    """``sample_number`` is the only place the harness turns a sample INDEX into the number a
    reader sees (#2076) — every surface reads it rather than spelling the arithmetic again.

    Structural, off the module's own AST, because the rule is one a rebase quietly breaks: the
    numbering was inconsistent for as long as it was open-coded, and a runner added while this
    branch was in flight wrote a fresh ``sample_index + 1`` of its own. A second spelling is
    invisible while it agrees and catastrophic when it stops — a report naming a sample whose
    files belong to its neighbour hands a reader facts about the wrong sample, which is how a
    correct exclusion came to be filed as a harness defect."""
    tree = ast.parse(_EVAL_CONFTEST.read_text())
    spellers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        for inner in ast.walk(node)
        if isinstance(inner, ast.BinOp)
        and isinstance(inner.op, ast.Add)
        and isinstance(inner.left, ast.Name)
        and inner.left.id == _SAMPLE_INDEX
        and isinstance(inner.right, ast.Constant)
        and inner.right.value == 1
    }
    assert spellers == {sample_number.__name__}, (
        f"only {sample_number.__name__} may number a sample — "
        f"{sorted(spellers - {sample_number.__name__})} derive it again, so the report and the "
        f"artifacts it points at can disagree"
    )


# The runner-local index the numbering pin reads by name: 0-based, because that is what a range
# produces, and the one thing no other surface may turn into a number on its own.
_SAMPLE_INDEX = "sample_index"


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
    is what the sample gets, and because one of these worlds seeds FEWER journeys than the
    module's five — where a leftover ``_JOURNEYS`` reference would show.

    The container premise rides along in the same loop, and both ways of getting it wrong
    are silent on a run: a container that arrived already archived would score the learn
    case's one cleanup claim green for free, and a world carrying a framing nobody accounted
    for would leave the other three claiming a container their round never built.  The
    registry claim stays out, exactly as it does for the learn → apply pin: the harness seeds
    fixture skills after the case's seed.

    The cohort's own arithmetic rides along too — five wordings of one ask, which is what
    makes a case's fifteen samples one number rather than a pool of several behaviours."""
    for index, case in enumerate(BAIL_CASES):
        assert len(case.also_phrased) == 4, (
            f"{case.case_id}: a cohort is FIVE wordings of one ask, got "
            f"{1 + len(case.also_phrased)}"
        )
        assert len({case.bail, *case.also_phrased}) == 5, (
            f"{case.case_id}: two of its wordings are the same string"
        )
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


def test_every_bail_reference_reply_is_a_message_penny_would_send() -> None:
    """Every bail case's reference reply — the answer the case itself calls correct — is a
    complete message by production's own rule.

    What the "check the scorer before you blame the model" tripwire becomes for this family
    once its guessed job-is-running vocabulary is gone.  The ported cases make no claim over
    reply text beyond provenance, so there is no scorer left to run the reference through;
    what is left worth pinning is the FIXTURE — these answers are the shortest in the suite
    ("sure thing — skipping it."), and an agreed answer that read as a fragment would be an
    agreed answer Penny would refuse to send, which is a bail nobody could pass."""
    for case in BAIL_CASES:
        reason = half_formed_send_reason(case.reference)
        assert reason is None, f"{case.case_id}: {reason} — reference: {case.reference!r}"


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
    assert not measured_turn_ran(db), "a world and nothing added to it is the dead sample"

    _log_prompt(db, response=_tool_call_response("browse"), run_id="r1")
    assert tool_was_called(db, "browse"), "the sample's own call still reads"
    assert count_tool_calls(db, "browse") == 1, "only the live call is counted"
    assert live_prompt_perf(db).calls == 1
    assert measured_turn_ran(db), "the sample's own row is the turn having run"

    assert is_seeded_run(seeded_run_id("learn-turn"))
    assert not is_seeded_run("r1")
    assert not is_seeded_run(None), "an unstamped row is a live row, not a seeded one"


# Every micro-context that draws on the cohort path, by the ledger identity its rows carry.
# Listed because the completeness gate has to hold for each of them and the framer's is the
# only one a run has ever exercised through it (#2076).
_MICRO_CONTEXT_AGENTS = (
    PennyConstants.SKILL_FRAME_AGENT_NAME,
    PennyConstants.SKILL_BIND_AGENT_NAME,
    PennyConstants.SKILL_NAMING_AGENT_NAME,
    PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
    PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
)


@pytest.mark.parametrize("agent_name", _MICRO_CONTEXT_AGENTS)
def test_a_micro_context_draw_over_a_seeded_world_is_a_measured_turn(tmp_path, agent_name) -> None:
    """A single-call sample's completeness gate reads the SAME window a chat sample's does
    (#2076): the promptlog minus the seeded prior turns, so a micro-context's own row makes
    the turn a measured one exactly like a chat run's does.

    Both directions over every micro-context on the cohort path, because the gate voids a
    sample rather than failing it — a false exclusion is a correct sample silently dropped
    from the pool, and the pooled rate is then computed over one fewer.  The window is keyed
    on the sample's own run identity and on nothing else, so an agent name nobody has drawn
    with yet is covered by construction; parametrized anyway, since the whole cost of
    trusting that in the framer's case was a filed harness defect that did not exist."""
    db = _make_db(tmp_path, name=agent_name)
    _log_prompt(db, response=_content_response("a prior round"), run_id=seeded_run_id("world"))
    assert not measured_turn_ran(db), "the seeded world alone is not this sample's work"
    assert _draw_exclusion(db, "a signature") == NO_MEASURED_TURN

    _log_prompt(db, response=_content_response("NAME: x"), agent_name=agent_name, run_id="live")
    assert measured_turn_ran(db), f"a live {agent_name} row IS the measured turn"
    assert _draw_exclusion(db, "a signature") is None, "a draw that came back is countable"
    assert _draw_exclusion(db, "") == NO_DRAW, "the turn ran; the draw is what failed"


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


class _RecordingClient:
    """A real client's stand-in: records who called, answers with a tool call once."""

    def __init__(self) -> None:
        self.callers: list[str | None] = []

    async def chat(self, messages=None, tools=None, *args, **kwargs):
        self.callers.append(kwargs.get("agent_name"))
        function = LlmToolCallFunction(name="browse", arguments={})
        call = LlmToolCall(id="call-1", function=function)
        return LlmResponse(message=LlmMessage(role="assistant", content="", tool_calls=[call]))


async def test_the_forced_fault_is_confined_to_the_turn_the_case_is_about() -> None:
    """An agent's model client is SHARED with every microcontext built from it, so an
    injector installed on ``chat_agent._model_client`` sees the extractor's calls too.

    Both halves of the trigger are gated on the caller, and both matter.  If another
    agent's tool call can ARM it, the bail is aimed one call early; if another agent's
    call can RECEIVE it, the fault is spent on a turn the case is not about and the turn
    under test runs clean — while ``bail_injected`` reports the contract exercised.  That
    is not hypothetical: on the first ported run the bail landed in ``browse-extract`` on
    12 of 15 gpt-oss samples and 15 of 15 gemma ones, and both cohorts were read as
    recoveries.  A collector runner passes no target and keeps the old any-caller
    behaviour, because there the cycle itself is what is being broken."""
    real = _RecordingClient()
    injector = _InjectTextBail(real, "{}", target_agent=PennyConstants.CHAT_AGENT_NAME)

    # A microcontext's tool-calling response must not arm the trigger...
    await injector.chat([], agent_name="browse-extract")
    assert not injector.bail_injected
    # ...and once the TARGET has made its tool call, the next microcontext call is still
    # the real model's, not the bail's.
    await injector.chat([], agent_name=PennyConstants.CHAT_AGENT_NAME)
    micro = await injector.chat([], agent_name="browse-extract")
    assert micro.has_tool_calls, "the extractor got the real model, not the sabotage"
    assert not injector.bail_injected, "the fault is still unspent"

    bailed = await injector.chat([], agent_name=PennyConstants.CHAT_AGENT_NAME)
    assert injector.bail_injected and bailed.message.content == "{}"
    assert real.callers == ["browse-extract", "chat", "browse-extract"], (
        "every call but the sabotaged one reached the real model"
    )


async def test_an_untargeted_injector_still_fires_on_any_caller() -> None:
    """The collector runners install their injector on the CYCLE's client and name no
    target, so the trigger keeps its any-caller behaviour — this pins that the confinement
    is opt-in and changes nothing for a case that did not ask for it."""
    real = _RecordingClient()
    injector = _InjectTextBail(real, "Done.")
    await injector.chat([], agent_name="collector")
    bailed = await injector.chat([], agent_name="browse-extract")
    assert injector.bail_injected and bailed.message.content == "Done."


def test_a_misfired_injection_is_a_named_exclusion_and_never_also_a_failed_check() -> None:
    """A recovery case's forced fault reports its misfire ONCE, to whichever half of the
    report can carry it (#2009).

    A sample the sabotage never fired on answered an unbroken turn: it exercised no
    recovery, so its end state says nothing about the behaviour the case is named for.  A
    PORTED case says so as a named exclusion and the sample leaves before any claim is
    answered; a case still on the scorer path has no exclusions section, so the guard Check
    stays its carrier.  Pinned in both directions because telling BOTH would count one
    misfire twice — as harness debris and as the model getting it wrong — which is exactly
    the reading a recorded run produced, marking a sample behavioural while the entry it
    was asked to correct had landed."""
    wrapper = _InjectTextBail(object(), "{}")

    def observe(db, reply, before, injected):  # a ported case's observer
        raise AssertionError("not called here")

    assert _guarded_injector(wrapper, None) is wrapper, "a scorer case keeps its guard"
    assert _guarded_injector(wrapper, observe) is None, "an observed case reports it itself"
    assert _guarded_injector(None, observe) is None, "a case that forced nothing has nothing"


def test_the_exclusion_asks_about_the_injector_only_where_one_was_installed(tmp_path) -> None:
    """``injected`` is the injector's own account of whether it fired, and ``None`` means the
    case installed none — so the condition is asked only of a case that forced a fault.

    The order matters and is pinned with it: a sample whose measured turn never ran, or that
    produced no reply, is excluded for THAT rather than for the injector, because those are
    the more fundamental facts and naming the injector for them would send a reader after the
    wrong thing."""
    db = _make_db(tmp_path)
    assert _exclusion(db, "an answer", None) == NO_MEASURED_TURN, "no live turn outranks all"

    _log_prompt(db, response=_content_response("an answer"))
    assert _exclusion(db, "an answer", None) is None, "no injector, nothing to ask"
    assert _exclusion(db, "an answer", True) is None, "the fault fired — a real sample"
    assert _exclusion(db, "an answer", False) == INJECTION_NEVER_FIRED
    assert _exclusion(db, "   ", False) == NO_REPLY, "an empty reply outranks the injector"


def test_a_ported_case_must_state_its_threshold_and_the_inline_path_keeps_its_default() -> None:
    """``min_pass_rate`` is REQUIRED on the cohort path, the way the behaviour sentence is.

    ``None`` cannot double as "not stated" the way an empty behaviour string can, because on
    this path ``None`` IS the value every ported case states deliberately — a report-only
    case.  So an unstated threshold has its own marker, and a ported case reaching the driver
    without one is refused before a sample runs rather than being gated at the inline path's
    default with nobody having decided that.

    Both directions, and the second is what keeps the ~40 unported cases untouched: on the
    inline path an unstated threshold still means 0.75, exactly as it always has."""
    assert _stated_pass_rate("a-case", None, ported=True) is None, "report-only is a VALUE"
    assert _stated_pass_rate("a-case", 0.9, ported=True) == 0.9, "and so is a stated floor"

    with pytest.raises(ValueError, match="must state its threshold"):
        _stated_pass_rate("a-case", UNSTATED, ported=True)

    assert _stated_pass_rate("a-case", UNSTATED, ported=False) == 0.75, "the inline default"

    # The SCORER's inputs take the same shape, and for a sharper reason (#2006): they carry
    # defaults so a ported case can omit them, and an inline case that omits one produces no
    # checks at all — `SampleResult.graded([])` scores 1.0 over nothing, so the case reports a
    # green for every sample it drove.  Refused before a sample runs; a ported case passes.
    _refuse_unscorable("a-case", ported=True, pool=(), expectations=())
    _refuse_unscorable("a-case", ported=False, pool=("something to sweep",))
    with pytest.raises(ValueError, match="must state its expectations"):
        _refuse_unscorable("a-case", ported=False, expectations=())


def test_what_the_store_holds_is_read_apart_from_what_this_round_wrote(tmp_path) -> None:
    """``held`` and ``entries`` are two reads of the same rows, and a claim about what a
    round left ALONE can only be answered by the first.

    ``entries`` is stamped by run id, so it carries only what this round wrote — which makes
    an entry the round never touched and an entry it DELETED look identical there, both
    simply absent.  The bracket-key recovery case's "nothing else in the collection moved"
    claim is answered against the seeded rows this round did not write, so it needs the read
    that can see them."""
    db = _make_db(tmp_path)
    _seed_board_games(db)
    seeded = collection_entries(db, BOARD_GAMES.name)

    assert _stored_entries(db) == [], "a seeded row cites a seeded run, so no round wrote it"
    assert {entry.key for entry in _held_entries(db)} == set(seeded), "the store holds them all"

    require_memory(db, BOARD_GAMES.name).update(
        key="Ark Nova", content="Ark Nova — now with a playtime.", author="chat", run_id="r1"
    )
    assert [(entry.key, entry.content) for entry in _stored_entries(db)] == [
        ("Ark Nova", "Ark Nova — now with a playtime.")
    ], "the round wrote exactly one entry"
    held = {entry.key: entry.content for entry in _held_entries(db)}
    assert set(held) == set(seeded), "and the collection still holds every key it was seeded with"
    assert held["Ark Nova"] == "Ark Nova — now with a playtime."
    assert held["Spirit Island"] == seeded["Spirit Island"], "an untouched row reads as seeded"


def test_a_mechanism_reads_as_born_changed_and_archived_by_the_run_that_did_it(tmp_path) -> None:
    """``_mechanism_records`` reads a registry ROW rather than what it holds, and its three
    booleans are what the round-ends-in-idle claims are answered from.

    Every one of them is silent when it is wrong, which is why it is pinned here rather than
    discovered on a paid run.  ``born_this_run`` is a name-set diff against the snapshot taken
    AFTER the seed, so a seeded collection reading as newly created would fail "nothing was
    created" on every sample.  ``changed_this_run`` is a mutation-ledger read filtered by run
    id, so a seeded event mistaken for a live one would fail "nothing was changed" on every
    sample — and the same read the other way round is what makes an archive visible at all.
    ``archived`` has to survive the row being retired, which is the one thing a reader that
    skipped archived rows would lose.

    The seeded half is laid down by the SEEDER a world's declared store really goes through,
    never by a create spelling the stamp out here: that second copy is what let this stay green
    while ``seed_collection`` cited no run at all, and an unstamped birth reads as a live run's
    work (#2129).  The ledger and the entry stamps are asserted beside the booleans since the
    booleans are a read of them — a stamp that stopped reaching either would leave
    ``changed_this_run`` false for a different reason, and the run that catches that is paid
    for."""
    db = _make_db(tmp_path)
    seeded_by = seeded_run_id(_SEEDED_ROUTES.name)
    seed_world_stores(db, _STORE_BACKED_WORLD)
    before = memory_names_now(db)

    records = {record.name: record for record in _mechanism_records(db, before)}
    seeded = records[_SEEDED_ROUTES.name]
    assert not seeded.born_this_run, "the seed laid it down before the sample's turn began"
    assert not seeded.changed_this_run, "and its only event cites the run that seeded it"
    assert not seeded.archived

    history = db.mutations.history(
        _SEEDED_ROUTES.name, MUTATION_HISTORY_WINDOW, entity_type=MutationEntityType.COLLECTION
    )
    assert [event.run_id for event in history] == [seeded_by], "the birth cites the seeded run"
    held = require_memory(db, _SEEDED_ROUTES.name).read_all()
    assert held, "the world declared entries, so the seed must have written some"
    assert {(entry.created_by_run_id, entry.last_written_by_run_id) for entry in held} == {
        (seeded_by, seeded_by)
    }, "the entry write cites the same seeded run the creation does"

    db.memories.create_collection("a-minted-job", "One the turn made.", created_by_run_id="live-1")
    db.memories.archive(_SEEDED_ROUTES.name, actor=MutationActor.SYSTEM, run_id="live-1")
    records = {record.name: record for record in _mechanism_records(db, before)}

    assert records["a-minted-job"].born_this_run and records["a-minted-job"].changed_this_run
    retired = records[_SEEDED_ROUTES.name]
    assert retired.archived, "an archived row is still READ — that is what the claim reads"
    assert retired.changed_this_run and not retired.born_this_run


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


# ── A cohort case's record says what its case document says (#2125) ──────────
#
# The two renderings of one scored case: the document `make eval-report` posts, and the
# `results.jsonl` record `baseline.py` and the flips index read.  They disagreed, because only
# the document was rendered from the cohort's claims — the record kept the drive-time scores,
# where a ported sample scores nothing and a vacuous 1.0 stands in.  So a case whose document
# said `14 pooled + 1 excluded` recorded fifteen passes and no causes, and a later run diffed
# against it saw no regression.

_COHORT_RECORD_CASE = "cohort-record-case"
_COHORT_RECORD_BEHAVIOUR = "In the chat agent, when the user teaches a round, Penny learns it."


def _drive_time_result(db: Database, observation: SampleObservation) -> SampleResult:
    """One sample as the COHORT PATH leaves it: scored nothing (its claims are answered after
    every sample has run), its fault facts read off its own database, its observation attached."""
    result = _guarded_graded([], [])
    _stamp_cause(db, result)
    result.observation = observation
    return result


def test_a_cohort_cases_record_carries_the_scores_causes_and_exclusions_its_document_states(
    tmp_path, capsys
) -> None:
    """One held claim, one missed, one sample the pool refused — through the real case close."""
    db = _make_db(tmp_path, "cohort-record")
    _log_prompt(db, response=_content_response("A perfectly ordinary draw."))
    observations = [
        SampleObservation(name="s-1", phrasing="the ask", landed=ConversationState.LEARN.value),
        SampleObservation(name="s-2", phrasing="the ask", landed=ConversationState.IDLE.value),
        SampleObservation(
            name="s-3", phrasing="the ask", complete=False, exclusion=NO_MEASURED_TURN
        ),
    ]
    cohort = Cohort(_COHORT_RECORD_CASE, "a-model", list(observations))
    cohort.assert_machine_landed(ConversationState.LEARN)
    results = [_drive_time_result(db, observation) for observation in observations]

    pending = _PendingCase(
        case_id=_COHORT_RECORD_CASE,
        family="chat",
        module="penny.tests.eval.chat.learn.test_case",
        min_pass_rate=None,
        gate_pathology_excluded=False,
        behaviour=_COHORT_RECORD_BEHAVIOUR,
    )
    pending.add(cohort, results, _Perf(), intended=len(results))
    pending.finish()

    # What the DOCUMENT states: two pooled samples holding one of two claim answers, and the
    # third named as excluded rather than subtracted.
    document = eval_cohort.pool(cohort.samples, cohort.features)
    assertions = eval_cohort.assertion_summary(assertion_rows(cohort.claims))
    assert (document.pooled, document.driven) == (2, 3)
    assert [row.reason for row in document.excluded] == [NO_MEASURED_TURN]
    assert (assertions.passed, assertions.total) == (1, 2)

    # What the RECORD states — built exactly as `record_case` builds it, off the same results.
    artifact = build_case_artifact(
        run_id="run-x",
        case_id=_COHORT_RECORD_CASE,
        family="chat",
        results=results,
        timings=CaseTimings(calls=0, duration_ms=0, input_tokens=0, output_tokens=0),
        standing_counts=Counter(
            standing.standing.value
            for standing in eval_cohort.standings(cohort.samples, cohort.features)
        ),
    )
    assert artifact.samples == document.driven
    assert artifact.sample_scores == [1.0, 0.0, 0.0]
    assert artifact.sample_causes == [None, FailureCause.BEHAVIORAL, FailureCause.HARNESS]
    assert artifact.cause_counts == CauseCounts(behavioral=1, harness=1)
    # The excluded sample is present AS excluded: it scores nothing, carries the document's own
    # reason, and reads as the infrastructure loss it is — never as a passing sample.
    assert results[2].failed == [NO_MEASURED_TURN]
    # And the record no longer contradicts itself — the sample its standings call dead is the
    # sample its scores call lost.
    assert artifact.standing_counts["dead"] == len(document.excluded)
    pooled_scores = artifact.sample_scores[: document.pooled]
    assert sum(pooled_scores) / len(pooled_scores) == assertions.rate

    # And the console RESULT line prints that same tally.
    out = capsys.readouterr().out
    assert (
        f"RESULT [{_COHORT_RECORD_CASE}] mean 0.33 · all-pass 1/3 across 3 samples (report-only)"
        in out
    )
    assert (
        "  pathology-excluded mean 0.33 (3 samples) · "
        "causes — behavioral 1 · pathology 0 · harness 1" in out
    )
    assert f"  [3] 0.00 — {NO_MEASURED_TURN}" in out


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


# ── A microcontext sample states its system prompt too (#2062) ───────────────────────────────

_MICRO_PROMPT_CASE = "micro-prompts"
_QUIET_CUSTOMER_CASE = "micro-prompts-quiet"
_CLASSIFIER_PROMPT = "Pick one state."
_NAMER_PROMPT = "Name the routine."


def _two_customer_ledger(db: Database) -> None:
    """One microcontext sample's promptlog as a two-customer run leaves it (#1803): the state
    classifier's draw, the run-end labeller's, and — between them — a main-agent row that is
    neither, so what the deposit is SCOPED to is assertable rather than assumed."""
    _log_prompt(
        db,
        agent_name=PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
        messages=[
            {"role": "system", "content": _CLASSIFIER_PROMPT},
            {"role": "user", "content": "current: idle · newest message: watch the listing"},
        ],
        response=_content_response("STATE: learn"),
        thinking="a task, not a question",
    )
    _log_prompt(
        db,
        messages=[
            {"role": "system", "content": "You are Penny."},
            {"role": "user", "content": "watch the listing"},
        ],
        response=_content_response("on it"),
    )
    _log_prompt(
        db,
        agent_name=PennyConstants.SKILL_NAMING_AGENT_NAME,
        messages=[
            {"role": "system", "content": _NAMER_PROMPT},
            {"role": "user", "content": "steps: browse"},
        ],
        response=_content_response("NAME: watch-the-listing"),
    )


def test_a_microcontext_case_states_the_prompt_its_draw_was_given(tmp_path, monkeypatch) -> None:
    # #2062: `_write_classifier_report` builds its transcript by hand, and never deposited the
    # sample's prompts the way `_build_transcript` does for a chat run — so `_case_prompts` was
    # empty for EVERY microcontext case (extractor, classifier, framer, binder, labeller) and
    # the case document's "state each distinct prompt once" section rendered nothing at all.
    # The prompt is what a reader needs to ask what the state failed to present, so its absence
    # sent every microcontext diagnosis to the sample database for the one thing the report
    # exists to carry.  It now deposits from the SAME rows its transcript renders, so every
    # prompt the document states belongs to an actor the reader can watch acting.
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    _case_prompts.pop(_MICRO_PROMPT_CASE, None)
    customers = (
        PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
        PennyConstants.SKILL_NAMING_AGENT_NAME,
    )
    result = SampleResult.graded([Check("named the routine", ok=True, kind="state")])
    for index in range(2):
        db = _make_db(tmp_path, f"micro-prompts-{index}")
        _two_customer_ledger(db)
        _write_classifier_report(
            db,
            _MICRO_PROMPT_CASE,
            index,
            result=result,
            phrasing="watch the listing",
            agent_names=customers,
        )

    assert [
        (sample, prompt.context, prompt.text)
        for sample, prompt in _case_prompts[_MICRO_PROMPT_CASE]
    ] == [
        ("sample 1", "state-classifier", _CLASSIFIER_PROMPT),
        ("sample 1", "skill-namer", _NAMER_PROMPT),
        ("sample 2", "state-classifier", _CLASSIFIER_PROMPT),
        ("sample 2", "skill-namer", _NAMER_PROMPT),
    ], "both customers under test deposit, in ledger order; the main agent neither was does not"

    # The case document's own section, composed exactly as the two case-close paths compose it
    # (`_record_case_report` for a ported case, `_record_unported_prompts` for one that never
    # built a cohort) — both pop this accumulator and render it through the same two calls.
    assert report.render_prompt_variants(
        report.prompt_variants(_case_prompts.pop(_MICRO_PROMPT_CASE), total=2)
    ) == (
        "<details><summary>System prompts — 2 contexts · 2 shared by every sample</summary>\n"
        "\n"
        "<details><summary>skill-namer — 17 chars · every sample</summary>\n"
        "\n"
        "```\n"
        "Name the routine.\n"
        "```\n"
        "\n"
        "</details>\n"
        "\n"
        "<details><summary>state-classifier — 15 chars · every sample</summary>\n"
        "\n"
        "```\n"
        "Pick one state.\n"
        "```\n"
        "\n"
        "</details>\n"
        "\n"
        "</details>"
    )

    block = _sample_report_text(tmp_path, _MICRO_PROMPT_CASE)
    assert _CLASSIFIER_PROMPT not in block and _NAMER_PROMPT not in block, (
        "a sample carries only its own sequence — the prompt is stated once, on the case (#1997)"
    )

    # A customer that never drew has no prompt to state, so the placeholder sample deposits none
    # rather than an empty entry the case-close path would then have to tell apart from a real one
    # — and it still renders its honest placeholder, pinned whole beside the deposit it withholds.
    quiet = _make_db(tmp_path, _QUIET_CUSTOMER_CASE)
    _two_customer_ledger(quiet)
    _write_classifier_report(
        quiet,
        _QUIET_CUSTOMER_CASE,
        0,
        result=result,
        phrasing="watch the listing",
        agent_names=(PennyConstants.BROWSE_EXTRACT_AGENT_NAME,),
    )
    assert _QUIET_CUSTOMER_CASE not in _case_prompts
    assert _sample_report_text(tmp_path, _QUIET_CUSTOMER_CASE) == (
        "<details><summary>sample 1 — ✅ pass · 0s · 3 calls</summary>\n"
        "\n"
        f"{report.NO_TURNS_PLACEHOLDER}\n"
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


# ── Every declared absent reading is one its observer can produce (#2061) ────
#
# ``Feature.absent`` names the reading that means a feature saw NOTHING, and the pooler marks
# a feature reading it on every sample BLIND — red, proposing no ceiling — so a ``0.000`` that
# means "read nothing" is never taken for the ``0.000`` that means perfect agreement.  What
# nothing checked was that the declared value is one the OBSERVER can ever return, and two were
# not: the collector path's walk has no ``no move`` anywhere in its vocabulary, and every
# structured field declared ``unset`` whether or not its own observer omits it.  A declaration
# no observation yields leaves the guard inert exactly where the design reads as armed, so each
# one is held against the observer that fills it, in ``make check``.


def test_the_chat_walks_absent_reading_is_what_its_own_observer_returns(tmp_path) -> None:
    """``transitions`` declares ``no move``, and ``_machine_walk`` really returns it — for a
    machine that recorded no move at all, which is what a chat sample that never moved leaves
    behind.  So the blindness guard on this path has a reading it can fire on."""
    db = _make_db(tmp_path, "chat-walk")
    assert _machine_walk(db) == eval_cohort.TRANSITIONS.absent

    db.machine.record_transition(
        from_state=ConversationState.IDLE,
        to_state=ConversationState.LEARN,
        cause=TransitionCause.CLASSIFIER,
    )
    assert _machine_walk(db) == "idle→learn", "a recorded move is a real reading"


def test_a_collectors_cycle_script_has_no_absent_reading_and_declares_none(tmp_path) -> None:
    """The collector fills the same ``walk`` field from its own observer, whose vocabulary
    contains no ``no move`` at all — so ``cycle script`` declares none rather than borrowing a
    value it could never read.

    Declaring none is honest here rather than a hole: the one shape that means nothing ran
    EXCLUDES the sample before it is pooled, so among the samples this feature is pooled over
    there is no reading that means nothing was seen."""
    assert eval_cohort.CYCLE_SCRIPT.absent is None

    ran = [
        _observed_cycle(index=0, before={}, after={"price": "84 zorkmids"}),
        _observed_cycle(index=1, before={"price": "84"}, after={"price": "84"}),
        _observed_cycle(index=2, before={"price": "84"}, after={"price": "91"}, sent=["it moved"]),
        _observed_cycle(index=3, before={"p": "84"}, after={"p": "84"}, sent=["still 84"]),
    ]
    shapes = {_cycle_shape(cycle) for cycle in ran}
    assert len(shapes) == 4, f"four cycles, four shapes: {sorted(shapes)}"
    assert eval_cohort.TRANSITIONS.absent not in shapes, (
        f"this observer cannot produce the chat walk's absent reading: {sorted(shapes)}"
    )

    # The one shape that DOES mean nothing ran, and the gate that keeps it out of the pool.
    dead = CycleObservation(
        index=0, before={}, after={}, sent=[], calls=[], served=[], outcome=None, reason=None
    )
    assert _cycle_shape(dead) == CYCLE_DEAD
    db = _make_db(tmp_path, "cycle-script")
    _log_prompt(db)
    assert _cycles_exclusion(db, cycle_script(ran)) is None, "a cohort of real cycles pools"
    assert _cycles_exclusion(db, cycle_script([*ran, dead])) == NO_CYCLE


def _observed_cycle(
    *, index: int, before: dict[str, str], after: dict[str, str], sent: list[str] | None = None
) -> CycleObservation:
    """One cycle that RAN — it made a call and closed with a run record, which is what makes
    its shape one of the four a pooled collector sample can carry."""
    return CycleObservation(
        index=index,
        before=before,
        after=after,
        sent=sent or [],
        calls=[CycleCall(tool="browse", arguments={"queries": ["https://probe.example/page"]})],
        served=["## browse: https://probe.example/page"],
        outcome=RunOutcome.WORKED.value,
        reason=None,
    )


def test_a_structured_fields_unset_is_declared_only_where_its_observer_omits_it() -> None:
    """``unset`` means the field was not in the draw's output AT ALL, so it is a reading only
    an observer that OMITS the field on some outcomes can produce.

    The binder is that observer: a parameter it reports MISSING gets no value field, which is
    what the binder case's bound-value axis declares ``absent=FIELD_UNSET`` on.  Every other
    measured field is emitted on every outcome its own builder has, so ``unset`` is
    unreachable for it and declaring none is the truth — the pooler still catches such a field
    coming back BLANK, which is the reading every feature shares."""
    supplied, unsupplied = _MISSING_KEYWORD.expectations
    shortfall = eval_cohort.SampleObservation(
        name="s1",
        phrasing="the ask",
        output=_binding_output(
            MissingParameters(
                names=(unsupplied.parameter,), values={supplied.parameter: supplied.anchor}
            )
        ),
    )
    assert shortfall.field(bound_value_field(unsupplied.parameter)) == eval_cohort.FIELD_UNSET
    assert shortfall.field(bound_value_field(supplied.parameter)) == supplied.anchor

    by_observer = _observer_outcomes()
    for name, fields in _ALWAYS_EMITTED.items():
        for outputs in by_observer[name]:
            emitted = {field.name for field in outputs}
            assert set(fields) <= emitted, f"{name} omits {sorted(set(fields) - emitted)}"


# The measured fields whose observer emits them on EVERY outcome it has — so ``unset`` cannot
# occur for them and ``output_field`` declares no absent reading.  The framer's per-position
# fields are here because an ACCEPTED signature always carries at least one parameter (the
# ``_mints_a_usable_signature`` floor), and a draw that mints none fails whole and excludes
# the sample rather than pooling a position-less one.
_ALWAYS_EMITTED = {
    "binder": (BIND_OUTCOME, BIND_MISSING),
    "classifier": (CLASSIFY_OUTCOME, CLASSIFY_STATE, CLASSIFY_SKILL),
    "extractor": (EXTRACT_OUTCOME, EXTRACT_VALUE, EXTRACT_REASON),
    "framer": (
        FRAME_NAME,
        FRAME_DESCRIPTION,
        FRAME_PARAMETERS,
        frame_parameter_name(1),
        frame_parameter_says(1),
    ),
    "labeller": (label_name_field(OFFERED_SPOTS[0]), label_says_field(OFFERED_SPOTS[0])),
}


def _observer_outcomes() -> dict[str, list[list[eval_cohort.OutputField]]]:
    """Each observer's output on every outcome its own draw can come back with — a bound
    binding and a shortfall, a decided classification, both extraction readings, a one- and a
    two-parameter signature, and a labelling that answered every spot beside one that answered
    none."""
    framing = {"name": "watch_price", "description": "watch a page's price"}
    one = _drawn(name="url", description="the page to watch", value="https://probe.example/p")
    two = _drawn(name="keyword", description="what to look for", value="price")
    labelled = SkillLabels(
        labels={
            spot: LeafLabel(name=f"{spot}_value", description="what belongs there")
            for spot in OFFERED_SPOTS
        }
    )
    return {
        "binder": [
            _binding_output(BoundValues(values={"url": "https://probe.example/p"})),
            _binding_output(MissingParameters(names=("keyword",), values={})),
        ],
        "classifier": [
            _classification_output(
                StateDecision(outcome=StateDrawOutcome.DECIDED, state=ConversationState.IDLE)
            ),
            _classification_output(
                StateDecision(
                    outcome=StateDrawOutcome.DECIDED,
                    state=ConversationState.APPLY,
                    skill="watch_price",
                )
            ),
        ],
        "extractor": [
            _extraction_output(
                MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value="84 zorkmids")
            ),
            _extraction_output(
                MicroContextResult(
                    outcome=MicroExtractOutcome.NOT_PRESENT, reason="no price is listed."
                )
            ),
        ],
        "framer": [
            _framing_output(SkillSignature(**framing, parameters=(one,))),
            _framing_output(SkillSignature(**framing, parameters=(one, two))),
        ],
        "labeller": [
            _labelling_output(labelled, OFFERED_SPOTS),
            _labelling_output(SkillLabels(labels={}), OFFERED_SPOTS),
        ],
    }


# ── The microcontext arms: five wordings of ONE ask, over constant facts (#2006) ──
#
# Every ported microcontext case pools five arms into one number, which is only legal if
# the arms differ in their WORDS and agree on their FACTS.  Each probe below states that
# for its own case, in ``make check``, before any GPU time — a cohort whose arms disagree
# about the world reports the spread of two behaviours as the instability of one.


def test_every_passing_mention_arm_holds_a_door_open_it_must_decline(tmp_path) -> None:
    """The classifier's arms (#2006): five ways of mentioning the same thing in passing,
    against a registry that really does put the routine doors on offer.

    The second half is what makes the case's one claim mean anything.  ``presented_edges``
    withholds every skill-gated state when the snapshot has no candidates, so a hold
    measured against an empty registry is a hold with no door to decline — it would score
    green while proving nothing.  Asserted from the PRODUCTION snapshot builder, so what
    the probe calls offered is what the draw will be offered.

    The arms themselves: five distinct wordings, each naming the subject a seeded routine
    plainly covers, and none of them asking for anything."""
    assert len(PASSING_MENTION_ARMS) == 5
    assert len(set(PASSING_MENTION_ARMS)) == 5, "five wordings, or the arms are not arms"
    for arm in PASSING_MENTION_ARMS:
        assert "auction listings" in arm, f"every arm names the same subject: {arm!r}"

    db = migrated_db(str(tmp_path / "classifier-arms.db"))
    for draft in SEEDED_SKILLS:
        db.skills.upsert(draft, author="probe")
    offered = presented_edges(
        build_snapshot(db, state=ConversationState.IDLE, message=PASSING_MENTION_ARMS[0])
    )
    assert ConversationState.APPLY in offered, "the apply door must be on offer to be declined"
    assert ConversationState.REQUEST in offered, "so must request"
    assert ConversationState.IDLE in offered, "and idle must be a door the state opens"


# The section a REQUEST-parked round always renders (#1894) — as a literal, because what
# #2084 fixed is that an isolated draw was shown a document this heading was missing from,
# and a probe reading the same constant the render reads could not have said so.
_WAITING_ON_HEADER = "## The details this task is waiting on"


def _waiting_on_block(content: str) -> str:
    """The waiting-on section of a rendered classifier document, alone.

    Sections are joined by a blank line and the section itself carries none, so a split on
    the blank line is the section boundary.  Exactly one, asserted rather than assumed: a
    document carrying none is the #2084 defect itself, and one carrying two is a render
    that has stopped meaning what the case reads it for."""
    blocks = [one for one in content.split("\n\n") if one.startswith(_WAITING_ON_HEADER)]
    assert len(blocks) == 1, f"expected exactly one waiting-on section, got {len(blocks)}"
    return blocks[0]


@pytest.mark.parametrize("case", REQUEST_APPLY_CASES, ids=lambda c: c.case_id)
def test_an_isolated_request_draw_is_shown_the_binding_production_carries(case, tmp_path) -> None:
    """The classifier fixture's REQUEST snapshot renders what the round is waiting on, and
    renders it exactly as the parked round in situ does (#2084).

    ``classifier_eval`` never passed ``round_binding``, so every isolated case drawn from
    request was asked to decide request → apply / request → idle / request → elicit
    without the one section naming the routine, what the user already gave and what is
    still missing — an input the wiring cannot produce, since production reaches request
    only through the binder and records what it settled on the move.  A pass on that
    document proves less than the case claims, and a miss could be the harness's.

    Driven against ONE database twice, so everything else in the two documents is a single
    read and the binding is the only thing that could differ.  The reference is the in-situ
    world's own recorded row — the very JSON ``ConversationMachine.shortfall()`` reads —
    and what is checked against it is what the DRIVER builds, through ``_classifier_snapshot``
    itself rather than a rebuild beside it, so reverting the wiring fails here.  Byte
    identity is the claim: the two documents are one document, heading included."""
    db = migrated_db(str(tmp_path / f"parked-classifier-{case.case_id}.db"))
    seed_parked_in_request(case)(db)
    # The routine the runner lays down after a case's seed — the row the fixture reads the
    # description and declared order off.  No description embedding: the runner computes one
    # so `resolve_by_meaning` can rank the skill, and nothing on the rendered path reads it.
    db.skills.upsert(case.parked.skill, author=EVAL_SEED_AUTHOR)

    parked = db.machine.latest_transition()
    assert parked is not None and parked.round_shortfall is not None
    recorded = RoundShortfall.model_validate_json(parked.round_shortfall)
    declared = ParkedRound(skill=case.parked.skill.name, settled=case.parked.settled)
    fixture = _registry_shortfall(db, case.case_id, declared)
    assert fixture == recorded, (
        f"{case.case_id}: the fixture must build the binding the parked round carries"
    )

    # The driver's own snapshot step, not a rebuild beside it: what #2084 fixed is an
    # argument this call did not pass, and an assertion over a snapshot assembled here
    # would stay green with the fix reverted.
    isolated = render_classifier_content(
        _classifier_snapshot(
            db,
            case_id=case.case_id,
            state=ConversationState.REQUEST,
            message=case.supply,
            penny_last_turn=case.reply,
            task_anchor=case.parked.ask,
            parked_round=declared,
        ),
        case.supply,
    )
    in_situ = render_classifier_content(
        build_snapshot(
            db,
            state=ConversationState.REQUEST,
            message=case.supply,
            penny_last_turn=case.reply,
            task_anchor=case.parked.ask,
            round_binding=recorded,
        ),
        case.supply,
    )
    assert _WAITING_ON_HEADER in isolated, (
        f"{case.case_id}: a request-parked draw must be shown what the round waits on"
    )
    assert _waiting_on_block(isolated) == _waiting_on_block(in_situ)
    assert isolated == in_situ


def test_a_parked_round_the_routine_cannot_be_waiting_on_is_refused(tmp_path) -> None:
    """Every incoherent way to declare a ``parked_round`` is refused before a sample runs.

    Each one would otherwise render a waiting-on section describing a round production
    cannot produce, and every one of them is silent on a run — the case would report a
    number as if the input were production's.  A binding on any state but request, since
    that is the only state carrying one; a routine the case's ``seed_skills`` never
    registered, which has no description or declared order to read; a value under a name
    the routine does not declare, which is narrowed away, so the section would claim the
    round gave nothing it in fact gave; and a round with nothing left missing, which the
    binder would have FRAMED rather than parked.

    The coherent declaration passes in the same breath, because a guard nothing gets past
    is indistinguishable from one that refuses everything."""
    case = REQUEST_APPLY_CASES[0]
    db = migrated_db(str(tmp_path / "parked-round-refusals.db"))
    db.skills.upsert(case.parked.skill, author=EVAL_SEED_AUTHOR)
    declared = ParkedRound(skill=case.parked.skill.name, settled=case.parked.settled)

    _refuse_binding_off_request("a-case", ConversationState.REQUEST, declared)
    _refuse_binding_off_request("a-case", ConversationState.IDLE, None)
    with pytest.raises(ValueError, match="only a round parked in request"):
        _refuse_binding_off_request("a-case", ConversationState.IDLE, declared)

    assert _registry_shortfall(db, "a-case", declared).skill == slug_skill_name(
        case.parked.skill.name
    )
    with pytest.raises(ValueError, match="seed_skills do not register"):
        _registry_shortfall(db, "a-case", ParkedRound(skill="a-routine-nobody-taught"))
    with pytest.raises(ValueError, match="does not declare"):
        _registry_shortfall(
            db, "a-case", declared.model_copy(update={"settled": {"not_a_parameter": "x"}})
        )
    every_value = {one.name: "something the user said" for one in case.parked.skill.parameters}
    with pytest.raises(ValueError, match="waiting on nothing"):
        _registry_shortfall(db, "a-case", declared.model_copy(update={"settled": every_value}))


def test_every_framing_arm_says_one_ask_in_different_words() -> None:
    """The framer's arms (#2006): five wordings of ONE ask, over one set of facts.

    The claims name a parameter family and a count, and both hinge on what the ask
    REQUIRES — so every arm has to require the same thing.  Each carries the symbol, the
    share price it is wanted for, and a clause about being told when it moves; an arm
    missing the notification clause would drop the negative direction the case exists for,
    and one missing the symbol would leave the family claim nothing to be answered by.

    The documents are built through the SHIPPED renderer, because that is what the draw
    reads — a probe over the raw turns would pass on a render that dropped one."""
    assert len(TICKER_ARMS) == 5
    documents = [
        build_framing_content(
            "", [(PennyConstants.MessageDirection.INCOMING, turn) for turn in arm]
        )
        for arm in TICKER_ARMS
    ]
    assert len(set(documents)) == 5, "five wordings, or the arms are not arms"
    for arm, document in zip(TICKER_ARMS, documents, strict=True):
        assert "VLT" in document, f"every arm names the symbol: {arm}"
        assert "share price" in document, f"every arm asks for the share price: {arm}"
        assert any(word in document for word in ("moves", "changes", "shifts")), (
            f"every arm asks to be told when it moves: {arm}"
        )


def test_every_naming_arm_offers_the_same_spots_over_a_different_demonstration() -> None:
    """The labeller's arms (#2006): five wordings of one demonstration, over ONE ledger.

    The ledger is what decides which spots exist and what each is currently called, and
    every claim in the case is keyed by those current names — so the arms must agree on
    them exactly.  They do because distillation is deterministic over the CALLS, which the
    arms do not vary at all; this says so rather than leaving it to be inferred.

    What must differ is the conversation block: five distinct documents, each carrying both
    addresses verbatim, since the case claims the two sources are told apart and an arm
    naming one of them could not be answered."""
    assert len(TWO_SOURCES_ARMS) == 5
    rendered = [
        _labelling_input(_TWO_SOURCES.calls, _TWO_SOURCES.target, arm, _TWO_SOURCES.conversation)
        for arm in TWO_SOURCES_ARMS
    ]
    documents = [content for content, _map in rendered]
    assert len(set(documents)) == 5, "five wordings, or the arms are not arms"
    maps = [tuple(sorted(by_value.items())) for _content, by_value in rendered]
    assert len(set(maps)) == 1, f"the arms offer different spots: {maps}"
    # A SET, not a sequence: the case's claims are keyed by name and its measured axes are
    # one per spot, so what has to hold is that it names exactly the spots the ledger
    # offers.  The order the ledger's distillation happens to walk them in is a render
    # detail, and pinning it here would make an unrelated change to that walk read as the
    # case claiming a spot nobody offers.
    offered = set(dict(maps[0]).values())
    assert offered == set(OFFERED_SPOTS), (
        f"the case claims spots the ledger does not offer: {sorted(offered ^ set(OFFERED_SPOTS))}"
    )
    for arm, document in zip(TWO_SOURCES_ARMS, documents, strict=True):
        assert "citydesk.example/front" in document, f"every arm names the first source: {arm!r}"
        assert "harborpost.example/front" in document, f"every arm names the second: {arm!r}"


def test_every_binding_arm_supplies_the_page_and_never_the_entry() -> None:
    """The binder's arms (#2006): five wordings of one ask, over one signature.

    Three facts have to hold on every arm or a claim is answered against ground its own
    sample never saw: the ADDRESS the url claim anchors on, the CADENCE the no-terms claim
    forbids (byte-identical, since the claim matches it literally), and the ABSENCE of
    anything naming a timetable entry — which is what makes the shortfall a property of the
    ask rather than of one phrasing.

    The signature is asserted constant too: an arm binding a different routine would be a
    different behaviour under one case id."""
    assert len(MISSING_KEYWORD_ARMS) == 5
    spoken_by_arm = [render_spoken_turns(arm) for arm in MISSING_KEYWORD_ARMS]
    documents = [
        build_binding_content(
            spoken,
            _MISSING_KEYWORD.skill,
            _MISSING_KEYWORD.intent,
            _MISSING_KEYWORD.parameters,
        )
        for spoken in spoken_by_arm
    ]
    assert len(set(documents)) == 5, "five wordings, or the arms are not arms"
    supplied, unsupplied = _MISSING_KEYWORD.expectations
    # What an ask that DOES supply the keyword says, read off the sibling case drawn against
    # the same signature — so "no arm names an entry" is anchored to a real one rather than to
    # a phrase spelled here.
    timetable_entry = _TWO_PARAMETERS.expectations[1].anchor
    for arm, spoken in zip(MISSING_KEYWORD_ARMS, spoken_by_arm, strict=True):
        assert supplied.anchor in spoken, f"every arm names the page: {arm}"
        for term in _MISSING_KEYWORD.forbidden:
            assert term in spoken, f"every arm states the cadence verbatim: {arm}"
        assert timetable_entry not in spoken, f"no arm names a timetable entry: {arm}"
    assert not unsupplied.anchor, "the keyword is the parameter the ask supplies nothing for"
    # BLANK the one span that moves and the five documents collapse to one: the signature
    # block renders identically on every arm, so nothing outside the user's own turns varies.
    # Read by REMOVING the turns rather than by splitting on the header they sit under — a
    # header spelled here would be a second copy of a production literal, and if it ever moved
    # the split would return the whole document on both sides and pass on anything.
    blanked = {
        document.replace(spoken, "")
        for document, spoken in zip(documents, spoken_by_arm, strict=True)
    }
    assert len(blanked) == 1, f"the arms differ outside the user's turns: {sorted(blanked)}"


def test_every_watch_arm_lays_down_one_program_differing_only_in_its_extract() -> None:
    """The collector arms are five wordings of ONE instruction, and this is what says so
    (#2017).

    Every arm's stored program must read back as the SAME calls under the strict rendered
    dialect, carry the one listing url, name the shared container, and render its OWN
    ``extract`` slot — and blanking that slot must leave five byte-identical programs.  An arm
    that differed anywhere else would be a different routine, therefore a different behaviour,
    and pooling it with the others would average two behaviours into one score.

    In ``make check`` rather than in the seeder alone: a program the strict parser cannot read
    leaves the cycle holding only its terminator, and a cycle with no browse writes nothing for
    the most boring reason there is — the exact shape of a passing sample."""
    programs = [watch_program(reading) for reading in WATCH_READINGS]
    for reading, program in zip(WATCH_READINGS, programs, strict=True):
        parsed = tuple(call.tool for call in program_calls(program, frozenset(WATCH_PROGRAM_CALLS)))
        assert parsed == WATCH_PROGRAM_CALLS, f"program: {program!r}"
        assert LISTING_URL in program, "the runtime join must fill the browse leaf with the url"
        assert f"'{WATCH_CONTAINER}'" in program, "the attachment must bind to the container"
        assert watch_extract_slot(reading) in program, f"program: {program!r}"

    assert len({reading.extract for reading in WATCH_READINGS}) == len(WATCH_READINGS), (
        "five wordings, or the arms are not arms"
    )
    blanked = {
        program.replace(watch_extract_slot(reading), "extract={}")
        for reading, program in zip(WATCH_READINGS, programs, strict=True)
    }
    assert len(blanked) == 1, f"the arms differ outside the extract slot: {sorted(blanked)}"


@pytest.mark.parametrize("case", WATCH_CASES, ids=lambda one: one.case_id)
def test_the_watch_arms_each_bring_their_own_world_over_the_same_facts(case) -> None:
    """The generalisation these cases exist to exercise — a collector's arms do NOT share a
    world, so a claim is answered against the ground its own sample read — held together with
    the rule that makes the value claims legal.

    Five arms, five distinct worlds, five distinct page bodies, and the SAME facts on every one
    of them: the one url, and the watched datum line byte-identical and appearing exactly once,
    with the price this case's cycle is NOT meant to see absent from every page.  The facts are
    what the claims hinge on (the collection holds the price the page carries), so an arm whose
    page carried a different price would force every claim back to a shape claim — which is what
    this refuses in ``make check``, before any GPU time.

    ONE register per arm, because a case drives one cycle: a second one here would be a second
    behaviour hidden inside a test named for one."""
    arms = [watch_arm(case, reading) for reading in WATCH_READINGS]
    shown = WATCH_DATUM if case.shows == WATCH_BASELINE_PRICE else WATCH_MOVED_DATUM
    unshown = WATCH_MOVED_DATUM if case.shows == WATCH_BASELINE_PRICE else WATCH_DATUM

    shown_amount = shown.removeprefix("Current price: $")
    unshown_amount = unshown.removeprefix("Current price: $")

    assert len({arm.world.name for arm in arms}) == len(WATCH_READINGS)
    assert len({arm.world.says for arm in arms}) == len(WATCH_READINGS), (
        "the prose must differ between arms, or they are one arm sampled five times"
    )
    for arm in arms:
        assert LISTING_URL in arm.world.says
        assert arm.world.says.count(shown) == 1, "one watched line, on every arm"
        assert unshown not in arm.world.says, "the page carries one reading, not both"
        # And at the NUMERAL level, because that is what a claim matches on: the identity of a
        # reading is its number, so a page carrying the other amount anywhere — a neighbouring
        # item's price, a spec figure, a housekeeping note — could satisfy a bare-numeral match
        # with a number the cycle never read.  The page must carry its own amount exactly once
        # and the other one not at all.
        assert arm.world.says.count(shown_amount) == 1, (
            f"{shown_amount!r} must appear only in the watched line: {arm.world.name}"
        )
        assert unshown_amount not in arm.world.says, (
            f"{unshown_amount!r} must appear nowhere on this page: {arm.world.name}"
        )
        assert len(arm.pages) == 1, "one cycle, one register"
        assert list(arm.pages) == list(arm.world.pages), (
            "the world a claim reads must BE the page the cycle was served"
        )


def test_the_three_watch_cases_differ_only_in_entry_condition_and_page() -> None:
    """Three behaviours, three cases, and what separates them is exactly two settings.

    A watch does three things — record a first reading, stay quiet on an unchanged one, rewrite
    and tell on a moved one — and which of them a case measures is decided by what the
    collection holds when its single cycle starts and by what the page says.  Everything else is
    shared, so a difference anywhere else would mean two cases were measuring the same
    behaviour under different names, or one was measuring a behaviour nobody declared."""
    assert len({case.case_id for case in WATCH_CASES}) == len(WATCH_CASES)
    settings = {(case.stored, case.shows) for case in WATCH_CASES}
    assert settings == {
        (None, WATCH_BASELINE_PRICE),
        (WATCH_BASELINE_PRICE, WATCH_BASELINE_PRICE),
        (WATCH_BASELINE_PRICE, WATCH_MOVED_PRICE),
    }
    # The middle case is the only state the write gate's unchanged STOP can fire against, and
    # it can only fire on a value stored under the key the program writes to.
    quiet = next(case for case in WATCH_CASES if case.stored == case.shows)
    assert quiet.stored == WATCH_BASELINE_PRICE
    # Mutually exclusive readings: the moved case asserts one is present and the other gone.
    # Held in BOTH forms — the price the page displays, and the bare amount a claim matches on.
    assert WATCH_BASELINE_PRICE not in WATCH_MOVED_PRICE
    assert WATCH_MOVED_PRICE not in WATCH_BASELINE_PRICE
    assert WATCH_BASELINE_AMOUNT not in WATCH_MOVED_AMOUNT
    assert WATCH_MOVED_AMOUNT not in WATCH_BASELINE_AMOUNT


def test_a_cohorts_claim_is_answered_against_its_own_arms_world() -> None:
    """The per-arm world, end to end: two arms with different ground, and a claim that reads
    the world it is handed answers each sample against its OWN arm.

    This is the requirement that could not be met by widening the observation record — a
    per-arm world cannot be bolted onto a per-cohort field."""
    ground = [watch_arm(WATCH_CASES[0], reading).world for reading in WATCH_READINGS[:2]]
    arms = [
        eval_cohort.Arm(label=f"phrasing {index + 1}", text=world.name, world=world)
        for index, world in enumerate(ground)
    ]
    samples = [
        SampleObservation(name=f"s{index}", phrasing=arm.label, arm=index)
        for index, arm in enumerate(arms)
    ]
    cohort = Cohort("case", "m", samples, arms)
    seen: list[str] = []
    cohort.claim(
        "state: the claim saw its own arm's world",
        lambda _sample, world: (bool(seen.append(world.name)) or True, None),
        eval_cohort.SpecCategory.STORE,
    )
    assert [outcome.ok for outcome in cohort.claims[0].outcomes] == [True, True]
    assert seen == [WATCH_READINGS[0].name, WATCH_READINGS[1].name]


# ── A world's ground, whichever substrate it stands on (#2108) ────────────────
#
# Two worlds of the same shape over different substrates: one page a browse returns, one
# collection already sitting in the store.  Deliberately tiny, so the whole render is a
# literal a reader can check against the fixtures right above it.

_TIDE_ANCHOR = "06:40"
_CLIMB_ANCHOR = "620"

# The keep and the exclude sit on SEPARATE lines, which is what the fields mean: a `keeps`
# token identifies the source, and an `excludes` token appears only on a line the ask rules
# out.  On one line the exclusion could not be read — the anchor could not be stored without
# it — and this fixture is where a case author will look for the worked example.
_TIDE_PAGE = CannedPage(
    match="harbourtides",
    text=(
        "Title: Harbour Tides — today | harbourtides\n"
        "https://harbourtides.example/today\n"
        "\n"
        f"High water at {_TIDE_ANCHOR}.\n"
        "The next low water is six hours later.\n"
    ),
)

_SEEDED_ROUTES = SynthCollection(
    "trail-notes",
    "Trail routes worth running again: distance, climb, and what the footing is like.",
    entries=(
        f"Marrow Ridge loop — 14km with {_CLIMB_ANCHOR}m of climb, dry underfoot.",
        "Fenwick Steps — 8km out and back, relentless stairs.",
    ),
)

_PAGE_BACKED_WORLD = World(
    name="the tide table the turn reads",
    pages=(_TIDE_PAGE,),
    keeps=((_TIDE_ANCHOR,),),
    excludes=("low water",),
    answers=(_TIDE_ANCHOR,),
)

_STORE_BACKED_WORLD = World(
    name="the user's own collection",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(_CLIMB_ANCHOR,),
    stores=(_SEEDED_ROUTES,),
)

# The third shape, and the only one where the two substrates meet: `keeps` is indexed by
# source, so its second token set belongs to the collection and reaching it means indexing
# straight across the page→collection boundary.
_MIXED_WORLD = World(
    name="a page read beside the collection it is checked against",
    pages=(_TIDE_PAGE,),
    keeps=((_TIDE_ANCHOR,), (_CLIMB_ANCHOR,)),
    excludes=("low water",),
    answers=(_TIDE_ANCHOR, _CLIMB_ANCHOR),
    stores=(_SEEDED_ROUTES,),
)

# A world declaring no ground at all — what a case that is not ported hands the cohort.  It
# renders no ground fold, and the closed summary still has to state a substrate.
_GROUNDLESS_WORLD = World(name="nothing declared", pages=(), keeps=(), excludes=())

_PAGE_BACKED_TAIL = """\
#### Test inputs

<details><summary>1 phrasing · 1 page · 1 must-keep, 1 must-not</summary>

<details><summary>Phrasings — 1 wording of one ask</summary>

| # | ask |
|---|---|
| phrasing 1 | when is high water? |

</details>

<details><summary>Seeded ground — 1 page · 1 must-keep, 1 must-not</summary>

| # | source | must be kept |
|---|---|---|
| 1 | page `harbourtides` | `06:40` |

**Must not be kept, from any source** — `low water`

**Must be stated by the reply** — `06:40`

<details><summary>Page 1 — `harbourtides` · 140 chars · keeps `06:40`</summary>

```
Title: Harbour Tides — today | harbourtides
https://harbourtides.example/today

High water at 06:40.
The next low water is six hours later.

```

</details>

</details>

</details>"""

_STORE_BACKED_TAIL = """\
#### Test inputs

<details><summary>1 phrasing · 1 collection · 0 must-keep, 0 must-not</summary>

<details><summary>Phrasings — 1 wording of one ask</summary>

| # | ask |
|---|---|
| phrasing 1 | how much climb did i note down? |

</details>

<details><summary>Seeded ground — 1 collection · 0 must-keep, 0 must-not</summary>

| # | source | must be kept |
|---|---|---|
| 1 | collection `trail-notes` |  |

**Must be stated by the reply** — `620`

<details><summary>Collection 1 — `trail-notes` · 241 chars · keeps —</summary>

```
trail-notes — Trail routes worth running again: distance, climb, and what the footing is like.
Marrow Ridge loop: Marrow Ridge loop — 14km with 620m of climb, dry underfoot.
Fenwick Steps: Fenwick Steps — 8km out and back, relentless stairs.
```

</details>

</details>

</details>"""

_MIXED_TAIL = """\
#### Test inputs

<details><summary>1 phrasing · 1 page, 1 collection · 2 must-keep, 1 must-not</summary>

<details><summary>Phrasings — 1 wording of one ask</summary>

| # | ask |
|---|---|
| phrasing 1 | is the tide right for the marrow ridge loop? |

</details>

<details><summary>Seeded ground — 1 page, 1 collection · 2 must-keep, 1 must-not</summary>

| # | source | must be kept |
|---|---|---|
| 1 | page `harbourtides` | `06:40` |
| 2 | collection `trail-notes` | `620` |

**Must not be kept, from any source** — `low water`

**Must be stated by the reply** — `06:40`, `620`

<details><summary>Page 1 — `harbourtides` · 140 chars · keeps `06:40`</summary>

```
Title: Harbour Tides — today | harbourtides
https://harbourtides.example/today

High water at 06:40.
The next low water is six hours later.

```

</details>

<details><summary>Collection 2 — `trail-notes` · 241 chars · keeps `620`</summary>

```
trail-notes — Trail routes worth running again: distance, climb, and what the footing is like.
Marrow Ridge loop: Marrow Ridge loop — 14km with 620m of climb, dry underfoot.
Fenwick Steps: Fenwick Steps — 8km out and back, relentless stairs.
```

</details>

</details>

</details>"""

_GROUNDLESS_TAIL = """\
#### Test inputs

<details><summary>1 phrasing · 0 pages · 0 must-keep, 0 must-not</summary>

<details><summary>Phrasings — 1 wording of one ask</summary>

| # | ask |
|---|---|
| phrasing 1 | how's your day going? |

</details>

</details>"""


@pytest.mark.parametrize(
    ("world", "ask", "expected"),
    [
        pytest.param(
            _PAGE_BACKED_WORLD, "when is high water?", _PAGE_BACKED_TAIL, id="page-backed"
        ),
        pytest.param(
            _STORE_BACKED_WORLD,
            "how much climb did i note down?",
            _STORE_BACKED_TAIL,
            id="store-backed",
        ),
        pytest.param(
            _MIXED_WORLD,
            "is the tide right for the marrow ridge loop?",
            _MIXED_TAIL,
            id="both-substrates",
        ),
        pytest.param(_GROUNDLESS_WORLD, "how's your day going?", _GROUNDLESS_TAIL, id="no-ground"),
    ],
)
def test_a_world_renders_its_ground_whichever_substrate_it_stands_on(world, ask, expected) -> None:
    """A store-backed world states its ground in the report where a page-backed one does.

    ``World.render`` was built around pages and returned nothing at all for a world whose
    ground is entries seeded into the store, so a case measured against the store showed the
    ask, the claims and the numbers with no way to see what the sample was answering against
    (#2108).  All four shapes go through the ONE path a report takes — same section, same
    position, same table — so the renders differ only where the worlds do: the noun on the
    row, the noun on the fold, and the counts the closed summary states.

    Every shape the surface can render is here, because the boundary between them is where
    this breaks: pages only, store only, BOTH (where ``keeps`` is indexed straight across the
    page→collection boundary and the summary composes ``1 page, 1 collection``), and a world
    declaring no ground at all, which renders no fold and still has to state a substrate.

    Asserted WHOLE, because the defect was an absence: a substring check for what a
    store-backed world does render says nothing about the section that was missing."""
    arm = eval_cohort.Arm(label="phrasing 1", text=ask, world=world)
    assert report.render_case_tail(arms=[arm]) == expected


def test_the_rendered_ground_is_the_store_the_sample_was_actually_given(db) -> None:
    """What the report shows and what the sample was handed are ONE declaration.

    A world's ``stores`` are seeded by the driver through the production create-then-write
    path, and that same declaration is what renders — so the report cannot describe a
    collection nobody laid down, or key an entry differently from the row a provenance claim
    traces an anchor back to.  Read off the store itself rather than off the seeder, since
    agreeing with what actually wrote the rows is the thing the render has to prove."""
    seed_world_stores(db, _STORE_BACKED_WORLD)
    held = collection_entries(db, _SEEDED_ROUTES.name)
    assert held == dict(_SEEDED_ROUTES.keyed)

    rendered = _STORE_BACKED_WORLD.render()
    for key, content in held.items():
        assert f"{key}: {content}" in rendered, f"the report never shows the entry {key!r}"
    assert _CLIMB_ANCHOR in rendered, "the anchor the reply owes must be visible in the ground"


def test_an_agent_turn_nothing_delivered_renders_as_an_aside_not_as_a_reply():
    """🤖 meant two opposite things, and the fold could not say which.

    In a chat fold it is the message the user RECEIVED. In the collector fold it was the cycle
    narrating after its own close — text nothing delivers, since what a user receives is a
    send-queue row the framework enters afterwards.

    The split follows the DELIVERED SET, which already draws exactly this line, rather than
    the fixture: a turn the sample sent is a reply, and a sample that delivered nothing has no
    turn that could be one. Both directions are pinned here, on the same text, because what
    broke was not either rendering on its own — it was that they were identical."""
    narration = "We need to finish cycle with done()."
    row = PromptLog(
        agent_name="collector",
        model="a-model",
        prompt_type="chat",
        messages=json.dumps([{"role": "assistant", "content": narration}]),
        response="{}",
    )

    nobody_told = _sample_turns([row], reply="", driven=(), delivered=())
    assert (_ACTOR["aside"], narration) in nobody_told
    assert (_ACTOR["penny"], narration) not in nobody_told
    assert _turn_kind(_ACTOR["aside"], narration) is report.EventKind.ASIDE

    delivered = _sample_turns([row], reply="", driven=(), delivered=(narration,))
    assert (_ACTOR["penny"], narration) in delivered
    assert _turn_kind(_ACTOR["penny"], narration) is report.EventKind.REPLY


def test_a_sample_whose_arm_cannot_be_resolved_raises_rather_than_passing_vacuously():
    """The direction of the failure is the point.

    The claims that read a world — "something from each page was written", "nothing excluded
    was stored" — are SATISFIED by an empty one, so a sample answered against a fallback would
    report a green check for a question nobody asked, on exactly the claims most likely to be
    wrong.  An unresolvable arm is a harness defect, and a harness defect that reports passes
    is worse than one that stops."""
    world = watch_arm(WATCH_CASES[0], WATCH_READINGS[0]).world
    arms = [eval_cohort.Arm(label="phrasing 1", text="a", world=world)]
    unstamped = SampleObservation(name="s0", phrasing="phrasing 1")  # arm defaults to -1
    cohort = Cohort("case", "m", [unstamped], arms)
    cohort.assert_nothing_excluded_was_stored()

    with pytest.raises(ValueError, match="carries arm -1"):
        _ = cohort.claims


@pytest.mark.parametrize("fixture", EXTRACT_FIELD_FIXTURES, ids=lambda f: f.case_id)
def test_each_extract_field_case_coheres_with_the_page_it_claims(fixture) -> None:
    """Per-case coherence probe (#1942): every thing a case says its page supplies really
    is on that page, exactly once, and every case declares at least one thing to look for.

    An anchor that is not on the page could never be extracted, so the case would grade
    the extractor on a fact nobody gave it — and a case declaring nothing at all grades
    nothing.  Both have to fail here, in ``make check``, rather than after GPU time.

    ONCE, because an anchor is shrunk to the smallest span with no alternative rendering,
    and shrinking has a floor: a span short enough to appear in a second story is satisfied
    by the wrong story, which is a check that cannot fail rather than a permissive one.
    Uniqueness is the property that says the shrink stopped in time, and it is a fact about
    the page, so this is where it is held.  Compared through the shipped ``spoken_form``,
    the same fold the claims use, so the count is over the text a draw is really matched
    against."""
    assert fixture.expectations
    assert fixture.behaviour.startswith("In "), (
        f"{fixture.case_id}: the behaviour reads 'In <the locus>, when <X>, Penny <does Y>.'"
    )
    assert ", when " in fixture.behaviour and " Penny " in fixture.behaviour, fixture.behaviour
    page = spoken_form(fixture.page)
    missing = [
        one.field
        for one in fixture.expectations
        if one.anchor and spoken_form(one.anchor) not in page
    ]
    assert missing == []
    repeated = {
        one.field: page.count(spoken_form(one.anchor))
        for one in fixture.expectations
        if one.anchor and page.count(spoken_form(one.anchor)) != 1
    }
    assert repeated == {}, f"an anchor must identify ONE span of its page: {repeated}"


def test_a_ported_case_states_the_behaviour_it_checks() -> None:
    """A case reaching the COHORT path without its sentence is refused, loudly.

    The case id is a filename — it says which fixture ran, never what was being asked — so a
    report rendering a rate above it states a number with no question attached.  Required on
    the cohort path alone: that is what every ported case runs on, while the inline path
    predates the convention and carries the cases whose porting is still ahead.

    The refusal names the case and the form, because the sentence is the one thing whoever hit
    this has to write."""
    assert stated_behaviour("c", "In the chat agent, when X, Penny does Y.") == (
        "In the chat agent, when X, Penny does Y."
    )
    # Whitespace is not a sentence.
    for empty in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="must state the behaviour it checks"):
            stated_behaviour("watch-writes-the-first-reading", empty)


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


# ── What a micro-context sample was GIVEN is its whole prompt (#2078) ────────
#
# Every observer under one gate, so a customer that goes back to building its haystack from
# the runner's own arguments fails here.  The document is the same for all five: what is
# asserted is that the CONTRACT reached the haystack beside it, and each customer's contract
# and user template are the SHIPPED ones, so the test can never pass against a prompt shape
# production does not use.


def _extraction_observation(db: Database) -> SampleObservation:
    return _observe_extraction(
        db,
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=EXTRACT_TAGGED_ANSWER),
        name="draw-1 (the ask)",
        phrasing="the ask",
        arm=0,
    )


def _classification_observation(db: Database) -> SampleObservation:
    return _observe_classification(
        db,
        StateDecision(
            outcome=StateDrawOutcome.DECIDED,
            state=ConversationState.APPLY,
            skill="watch-a-listing",
        ),
        name="draw-1 (the ask)",
        phrasing="the ask",
        arm=0,
    )


def _labelling_observation(db: Database) -> SampleObservation:
    return _observe_labelling(
        db,
        SkillLabels(labels={"memory": LeafLabel(name="destination", description="where it goes")}),
        ["memory"],
        name="draw-1 (the ask)",
        phrasing="the ask",
        arm=0,
    )


def _framing_observation(db: Database) -> SampleObservation:
    return _observe_framing(
        db,
        SkillSignature(
            name="watch_a_listing",
            description="Watch one listing and record what it says.",
            parameters=(FramedParameter(name="url", description="the page", value="the page"),),
        ),
        name="draw-1 (the ask)",
        phrasing="the ask",
        arm=0,
    )


def _binding_observation(db: Database) -> SampleObservation:
    return _observe_binding(
        db,
        BoundValues(values={"url": "https://sports-beta.example/hockey"}),
        name="draw-1 (the ask)",
        phrasing="the ask",
        arm=0,
    )


# Every micro-context customer: who drew, the contract it drew under, the user template it
# was handed, and the observer that reads the sample back.
_MICRO_CONTEXT_GIVENS = (
    (
        PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        MICRO_CONTEXT_SYSTEM_PROMPT,
        _USER_TEMPLATE,
        _extraction_observation,
    ),
    (
        PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
        STATE_CLASSIFIER_SYSTEM_PROMPT,
        _BARE_CONTENT_TEMPLATE,
        _classification_observation,
    ),
    (
        PennyConstants.SKILL_NAMING_AGENT_NAME,
        SKILL_NAMING_SYSTEM_PROMPT,
        _BARE_CONTENT_TEMPLATE,
        _labelling_observation,
    ),
    (
        PennyConstants.SKILL_FRAME_AGENT_NAME,
        SKILL_FRAME_SYSTEM_PROMPT,
        _BARE_CONTENT_TEMPLATE,
        _framing_observation,
    ),
    (
        PennyConstants.SKILL_BIND_AGENT_NAME,
        BIND_SKILL_SYSTEM_PROMPT,
        _BARE_CONTENT_TEMPLATE,
        _binding_observation,
    ),
)


@pytest.mark.parametrize(
    ("agent_name", "contract", "user_template", "observe"),
    _MICRO_CONTEXT_GIVENS,
    ids=[customer[0] for customer in _MICRO_CONTEXT_GIVENS],
)
def test_every_micro_context_observer_is_given_the_contract_it_drew_under(
    tmp_path, agent_name: str, contract: str, user_template: str, observe
) -> None:
    """A sample's `given` is the prompt its draw received — the contract as well as the
    document (#2078).

    Each observer used to be handed a haystack the runner built from its own arguments, so
    the words the draw was TOLD to answer in were missing from it and read as inventions.
    The extraction case is the one that measured it; the other four carry the same defect on
    their own contracts' vocabulary, so all five are held here rather than one standing in
    for the rest."""
    db = _make_db(tmp_path, "micro-context-given")
    _log_prompt(
        db,
        messages=MicroContext._messages(
            EXTRACT_TAGGED_PAGE, EXTRACT_TAGGED_ASK, contract, user_template
        ),
        agent_name=agent_name,
    )

    observation = observe(db)

    assert observation.complete
    assert contract in observation.given
    assert EXTRACT_TAGGED_PAGE in observation.given


def test_an_extraction_answer_carrying_the_contract_tags_invents_nothing(tmp_path) -> None:
    """The claim that failed, end to end (#2078): the probe over an extraction sample's own
    observation.

    The drawn value marks each story's missing field with `NOT_PRESENT`, whose tail glues
    across the line break into a name phrase the page does not carry — 2 of 15 correct draws
    failed `nothing invented` on it.  With the contract in the haystack the claim holds, and
    the paired guard asks for a real invention back: one byline swapped for a name nowhere in
    the prompt is still reported, so this is not "a capitalised word is never an invention"."""
    db = _make_db(tmp_path, "extraction-given")
    _log_prompt(
        db,
        messages=MicroContext._messages(EXTRACT_TAGGED_PAGE, EXTRACT_TAGGED_ASK),
        agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
    )

    sourced = _extraction_observation(db)
    assert NOT_PRESENT_TAG in sourced.given
    assert unsourced_specifics(sourced.output_text, sourced.given) == []

    invented = _observe_extraction(
        db,
        MicroContextResult(
            outcome=MicroExtractOutcome.EXTRACTED,
            value=EXTRACT_TAGGED_ANSWER.replace(EXTRACT_TAGGED_BYLINE, EXTRACT_UNSOURCED_NAME, 1),
        ),
        name="draw-2 (the ask)",
        phrasing="the ask",
        arm=0,
    )
    assert unsourced_specifics(invented.output_text, invented.given) == ["Casimir", "Oyelaran"]


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


# ── The idle answering cases' own fixtures (#2008) ───────────────────────────
#
# The two reply READS this replaces — a stored-title containment and the duplicate-save
# vocabulary — went with the port: the first is `fold_typography` now, and the second was a
# phrasing match, which the design abolishes.  What is worth pinning in their place is the
# FIXTURE, because every claim these cases make is answered against it and a drift is silent
# on a run: a token no world carries fails a correct reply, and a token the store map already
# renders passes a reply that read nothing.  Cheap and deterministic, so it runs here rather
# than costing a GPU run to find.


def test_every_answering_case_asks_one_thing_in_five_wordings() -> None:
    """A cohort is FIVE wordings of ONE ask, which is what makes its fifteen samples one
    number rather than a pool of several behaviours.

    Two ways to get it wrong and both are silent on a run: four wordings pool twelve samples
    under a number recorded at fifteen, and a repeated wording pools one wording twice while
    claiming the coverage of two."""
    for case in ANSWERING_CASES:
        assert len(case.also_phrased) == 4, (
            f"{case.case_id}: a cohort is FIVE wordings of one ask, got "
            f"{1 + len(case.also_phrased)}"
        )
        assert len({case.ask, *case.also_phrased}) == 5, (
            f"{case.case_id}: two of its wordings are the same string"
        )


def test_every_claimed_answer_is_one_token_its_own_world_carries() -> None:
    """Each case's ``answers`` tokens are single whitespace-free tokens, and each one appears
    in the world the case is answered against.

    ONE TOKEN because ``fold_typography`` folds a declared SET of space characters rather than
    the whole Unicode category — measured, two samples typed a seeded title with U+202F
    between the words and were scored as naming nothing — so a multi-word token can be failed
    by a space nobody has met yet, while a single token cannot.

    ON EXACTLY ONE SOURCE, which is the stronger half and the half a containment test misses.
    A token no source carries fails a correct reply for a fact the fixture never stated — and
    a token carried by TWO is reachable without the hop the case exists to measure, which is
    silent on a run because the reply states the answer either way.  That absence IS the
    one-link-deep case's whole premise: the maker is credited on the gallery page and nowhere
    on the index that points at it, so a reply naming him opened the second page.

    SOURCES rather than pages, since #2114: a world's ground is its pages AND the collections
    it declares already in the store, so a store-backed case is held to the same rule as a
    page-backed one instead of being skipped for having no pages."""
    for case in ANSWERING_CASES:
        for token in case.world.answers:
            assert token.split() == [token], (
                f"{case.case_id}: the answer token {token!r} carries whitespace, so a space "
                "the fold has not met can fail a correct reply"
            )
            carriers = [source.label for source in case.world.sources if token in source.text]
            assert len(carriers) == 1, (
                f"{case.case_id}: the answer token {token!r} is carried by {carriers} of "
                f"{len(case.world.sources)} source(s) — it must sit on exactly one, or the "
                "hop the case measures is reachable without taking it"
            )


def test_every_answering_world_seeds_the_store_its_claims_assume(tmp_path) -> None:
    """Each case's world lays down cleanly and reads back as the world its claims assume — the
    loud probe run here, against a real migrated database, rather than only under ``make eval``
    where a raise costs a GPU run before it is seen.

    Its own database per case, because that is what a sample gets, and seeded through the
    DRIVER's own seam (``seed_world_stores``) rather than a second copy of it — the world's
    declaration is what the sample gets and what the report renders, so it is what this must
    exercise.

    The answer tokens ride along for a store-backed world: the value has to be in the store
    the world declares, since the whole behaviour is that she went and read it."""
    for index, case in enumerate(ANSWERING_CASES):
        if not case.world.stores and case.probe is None:
            continue
        db = migrated_db(str(tmp_path / f"answering-{index}.db"))
        seed_world_stores(db, case.world)
        if case.probe is not None:
            case.probe(db)
        if not case.world.stores:
            continue
        stored = " ".join(
            f"{key} {content}"
            for row in db.memories.list_all()
            if row.type == MemoryType.COLLECTION
            for key, content in collection_entries(db, row.name).items()
        )
        for token in case.world.answers:
            assert token in stored, (
                f"{case.case_id}: the seeded store does not carry the answer token {token!r}"
            )


def test_every_tool_name_read_is_sanitised_the_way_production_sanitises_it() -> None:
    """One chokepoint, and it is production's own function rather than a second spelling of it.

    `strip_harmony_control_tokens` runs where the name is read off the model response
    (`LlmToolCallFunction.name`), so registry lookup, done-detection, dedup and result framing
    all see the clean identifier — and the eval reading the raw name was the single exception.
    Two spellings of one contract drift, and the eval would then measure a normalisation
    production does not do."""
    leaked = {"function": {"name": "collection_write<|channel|>commentary", "arguments": "{}"}}
    assert tool_call_name(leaked) == "collection_write"
    assert tool_call_name(leaked) == strip_harmony_control_tokens(leaked["function"]["name"]), (
        "the eval must not re-implement the strip"
    )
    assert tool_call_name({"function": {"name": "browse"}}) == "browse"
    assert tool_call_name({}) == "", "a malformed call reads as no tool, never as a crash"


def test_the_tool_sequence_reads_calls_no_name_list_names(tmp_path) -> None:
    """The cohort's `tool sequence` feature reads what a chat turn CALLED, off the prompt log,
    keyed to no name list — so a case whose behaviour is an archive, a dispatch or a log read is
    measured rather than reading `no call` on every sample whatever the turn did.

    Two ways a call can vanish are pinned together.  A name nobody enumerated is read — none of
    the three below is an ``ENACTING_TOOLS`` member, and a plugin can contribute a fourth
    tomorrow — and a leaked Harmony control token is stripped the way production strips it, so a
    call whose entries are in the store cannot render as a tool nobody has heard of.  A non-chat
    row stays out: the agent filter is the one filter this reader keeps.

    One chat row, so the order asserted is the emission order inside a response rather than a
    race between two rows' timestamps."""
    db = _make_db(tmp_path)
    _log_prompt(
        db,
        response={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": name, "arguments": "{}"}}
                            for name in (
                                "collection_archive",
                                "log_read<|channel|>commentary",
                                "choose",
                            )
                        ]
                    }
                }
            ]
        },
        agent_name=PennyConstants.CHAT_AGENT_NAME,
        run_id="turn-1",
    )
    _log_prompt(
        db,
        response=_tool_call_response("collection_write"),
        agent_name=Collector.name,
        run_id="cycle-1",
    )

    read = _chat_tool_sequence(db)
    assert read == ["collection_archive", "log_read", "choose"]
    assert not [name for name in read if name in ENACTING_TOOLS], (
        "the premise holds: not one of these names is on the legacy enacting-tool list"
    )
