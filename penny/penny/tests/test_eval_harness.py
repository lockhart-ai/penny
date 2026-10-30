"""Deterministic tests for the eval-harness scoring + report machinery (issue #1694).

These drive the ``tests/eval/conftest.py`` scoring/report code directly with fixture
``Check`` / ``SampleResult`` data and a seeded promptlog — no live model, no ``eval`` marker —
so they run inside ``make check`` and pin what the drivers do: how a sample's answered claims
come to a score (``rationale``, the advisory and not-applicable states), the fragile-pass
verdict, the RESULT line, the exclusions, and what a driver refuses.  Whole-render literal
assertions cover every report shape.

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
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp import web

# Importing the memory-tools module registers those tools (``Tool.__init_subclass__``) so
# ``Tool.format_result`` dispatches their real ``to_result_narration`` — the rejection-probe
# tests below build frames from the PRODUCTION templates, never hand-invented text.
import penny.tools.memory_tools  # noqa: F401  (imported for registration side effect)
from penny.agents.collector import Collector
from penny.constants import (
    MutationAction,
    MutationActor,
    MutationEntityType,
    PennyConstants,
    RunOutcome,
)
from penny.database import Database
from penny.database.memory import MemoryType
from penny.database.models import PromptLog
from penny.datetime_utils import stored_as_utc
from penny.llm.call_link import CallLink
from penny.llm.models import (
    LlmMessage,
    LlmToolCall,
    LlmToolCallFunction,
    strip_harmony_control_tokens,
)
from penny.responses import PennyResponse
from penny.tests import eval as eval_package
from penny.tests.conftest import TEST_SENDER, require_memory, wait_until
from penny.tests.eval import drivers as eval_drivers_package
from penny.tests.eval.chat.idle.test_chat_memory_stories import (
    _DELETE_LIST,
    VERB_CASES,
    claim_the_list_delete,
    probe_seeded_world,
)
from penny.tests.eval.chat.idle.test_chat_reply import (
    ANSWERING_CASES,
    _seeded_entries_unchanged,
)
from penny.tests.eval.chat.idle.test_choose_dispatch import (
    _OPTIONS as _CHOOSE_OPTIONS,
)
from penny.tests.eval.chat.idle.test_choose_dispatch import (
    REFERENCE_REPLY as CHOOSE_REFERENCE_REPLY,
)
from penny.tests.eval.chat.idle.test_choose_dispatch import (
    assert_choose_world,
    picks_on_the_record,
    reply_reports,
)
from penny.tests.eval.chat.idle.test_email_dispatch import (
    ABSENT,
    ABSENT_ASK_SENDER,
    ABSENT_SUBJECT,
    EMAIL_CASES,
    EMAIL_TOOLS,
    LOOKUP,
    MAILBOX,
    MAILBOX_FIGURE_IN_REPLY,
    MAILBOX_FIGURES,
    NAMES_A_MAILBOX_FIGURE,
    NAMES_NO_MAILBOX_FIGURE,
    NEIGHBOUR_FIGURES,
    NEWSLETTERS,
    QUOTE_FIGURE,
    REMARK,
    THE_QUOTE,
    UNREACHABLE,
    assert_mailbox_world,
    claim_the_absent,
    claim_the_lookup,
    claim_the_remark,
    claim_the_unreachable,
)
from penny.tests.eval.chat.idle.test_half_the_sources_landed import (
    SOURCE_DOWN_CASES,
    SourceDownCase,
    wrote_only_what_the_readable_page_gave,
)
from penny.tests.eval.chat.idle.test_half_the_sources_landed import (
    probe_seeded_world as probe_source_down_world,
)
from penny.tests.eval.chat.idle.test_image_dispatch import (
    IMAGE_CASES,
    SUBJECT,
    ImageCase,
    assert_image_world,
    drew,
    install_image_client,
)
from penny.tests.eval.chat.idle.test_notifications import (
    assert_mute_world,
    assert_unmute_world,
    seed_muted,
)
from penny.tests.eval.chat.idle.test_notifications import (
    assert_no_fire_world as assert_mute_no_fire_world,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _BASELINE_AMOUNT as WATCH_BASELINE_AMOUNT,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _BASELINE_PRICE as WATCH_BASELINE_PRICE,
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
    CASES as WATCH_CASES,
)
from penny.tests.eval.collector.test_watch_cycles import (
    READINGS as WATCH_READINGS,
)
from penny.tests.eval.collector.test_watch_cycles import (
    _arm as watch_arm,
)
from penny.tests.eval.conftest import (
    _ACTOR,
    _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS,
    _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_LOCAL,
    _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_REMOTE,
    EVAL_LLM_IS_LOCAL,
    MUTATION_HISTORY_WINDOW,
    NO_CHAT_DRAW,
    NO_CYCLE,
    NO_DRAW,
    NO_MEASURED_TURN,
    NO_REPLY,
    PENNY_LOGGER,
    SAMPLE_WALL_CLOCK_SECONDS,
    SAMPLE_WALL_CLOCK_SECONDS_LOCAL,
    SAMPLE_WALL_CLOCK_SECONDS_REMOTE,
    SINGLE_CALL_WAIT_SECONDS,
    SINGLE_CALL_WAIT_SECONDS_LOCAL,
    SINGLE_CALL_WAIT_SECONDS_REMOTE,
    TURN_REPLY_WAIT_SECONDS_LOCAL,
    UNSTATED,
    Check,
    SampleResult,
    _case_prompts,
    _chat_tool_sequence,
    _cycle_descriptions,
    _draw_exclusion,
    _drive_turns,
    _exclusion,
    _flush_sample_blocks,
    _held_entries,
    _is_local_llm_endpoint,
    _last_scripted_turn_at,
    _mechanism_records,
    _observe_sample,
    _PendingCase,
    _Perf,
    _report_result,
    _require_ask,
    _require_report_only,
    _response_tool_calls,
    _sample_db_path,
    _sample_turns,
    _seed_sample,
    _stamp_cause,
    _stored_entries,
    _turn_kind,
    _turns_to_events,
    _write_sample_report,
    canned_browse_request_fn,
    collection_entries,
    draw_rerolled,
    env_seconds,
    eval_penny,
    given_to_the_model,
    is_seeded_run,
    live_prompt_perf,
    measured_turn_ran,
    outgoing_replies,
    run_exhibited_pathology,
    sample_is_fragile,
    sample_log_path,
    sample_logging,
    sample_number,
    seed_collection,
    seed_world_stores,
    seeded_run_id,
    stored_images,
    tool_call_name,
)
from penny.tests.eval.conftest import (
    _BrowseReadError as browse_read_error,
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
from penny.tests.eval.drivers.chat import (
    _CHAT_TURN_REPLY_WAIT_SECONDS,
    _CHAT_TURN_REPLY_WAIT_SECONDS_REMOTE,
)
from penny.tests.eval.drivers.collector_cycles import (
    CYCLE_DEAD,
    CycleCall,
    CycleObservation,
    _cycle_shape,
    _cycles_exclusion,
    _cycles_ran_check,
    _DrivenCycles,
    cycle_script,
)
from penny.tests.eval.drivers.end_to_end import (
    _END_TO_END_TURN_REPLY_WAIT_SECONDS,
    _END_TO_END_TURN_REPLY_WAIT_SECONDS_REMOTE,
    _no_job_created_yet,
    _observe_end_to_end,
)
from penny.tests.eval.drivers.extractor import _observe_extraction, _write_classifier_report
from penny.tests.eval.end_to_end.world import seed_known_routine
from penny.tests.eval.extractor.test_browse_extract_fields import FIXTURES as EXTRACT_FIELD_FIXTURES
from penny.tests.eval.utils import cohort as eval_cohort
from penny.tests.eval.utils import report
from penny.tests.eval.utils.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckOutcome,
    FailureCause,
    active_run,
    build_case_artifact,
)
from penny.tests.eval.utils.assertions import (
    Cohort,
    _a_new_job_ends_when_asked,
    _a_new_job_fires_every,
    _a_new_job_notifies,
    _new_jobs,
)
from penny.tests.eval.utils.baseline import load_baseline
from penny.tests.eval.utils.cohort import (
    Arm,
    Given,
    MechanismRecord,
    SampleObservation,
    StoredEntry,
    StoredImage,
)
from penny.tests.eval.utils.dispatch_world import assert_no_collections, collection_names
from penny.tests.eval.utils.fixtures import (
    BOARD_GAMES,
    ENACTING_TOOLS,
    EXTRACT_PLAIN_ANSWER,
    EXTRACT_PLAIN_ASK,
    EXTRACT_PLAIN_PAGE,
    LISTING_URL,
    CannedPage,
    SynthCollection,
)
from penny.tests.eval.utils.job_end import (
    AskedEnd,
    job_ends_as_asked,
)
from penny.tests.eval.utils.mailbox import CannedMailbox, MailboxReadError
from penny.tests.eval.utils.schedules import cadence_seconds, rule_parts
from penny.tests.eval.utils.stored_words import STORED_WORDS_FOUND_NOWHERE
from penny.tests.eval.utils.worlds import World
from penny.tests.mocks.signal_server import MockSignalServer
from penny.tests.schema_template import migrated_db, schema_only_db

# Production's own rule for a message worth delivering, read from where the send path and the
# run-health classifier read it — so the tripwire below pins the agreed reference replies
# against the same definition the ported claim answers with, never a second copy of it.
from penny.tools import micro_context as micro_context_module
from penny.tools.base import Tool
from penny.tools.choose import CHOSE_MESSAGE
from penny.tools.collection_instantiation import _LINE_ESCAPE
from penny.tools.link_handles import HandleTable
from penny.tools.micro_context import (
    _USER_TEMPLATE,
    MICRO_CONTEXT_SYSTEM_PROMPT,
    MicroContext,
    MicroContextResult,
    MicroExtractOutcome,
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
    """`make eval-remote` forwards every variable, so an unset one arrives EMPTY, not absent.

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


@pytest.mark.parametrize(
    ("url", "is_local"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://host.docker.internal:11434", True),
        ("https://openrouter.ai/api/v1", False),
        ("https://api.x.ai/v1", False),
        ("http://192.168.1.50:11434", False),
    ],
)
def test_a_local_endpoint_is_told_apart_by_its_hostname_alone(url: str, is_local: bool) -> None:
    """The LOCAL/REMOTE split (#2713) reads only the endpoint's own hostname — a loopback
    address or Docker's host-gateway alias is local, every other host (a direct provider, a
    routed gateway, a LAN address) is remote, whatever port or path rides along with it."""
    assert _is_local_llm_endpoint(url) is is_local


def test_a_remote_endpoint_keeps_every_wait_byte_for_byte_unchanged() -> None:
    """#2713 widens waits for a LOCAL endpoint only — the REMOTE branch of each one must
    still be the exact value that shipped before this change, so a remote run (``make
    eval-remote``) is unaffected."""
    assert _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_REMOTE == 20.0
    assert _CHAT_TURN_REPLY_WAIT_SECONDS_REMOTE == 120.0
    assert _END_TO_END_TURN_REPLY_WAIT_SECONDS_REMOTE == 240.0
    assert SINGLE_CALL_WAIT_SECONDS_REMOTE == 180.0
    assert SAMPLE_WALL_CLOCK_SECONDS_REMOTE == 1800.0


def test_every_wait_resolves_to_its_local_or_remote_half_by_the_same_flag() -> None:
    """Each of the five waits #2713 names picks its LOCAL value when the configured
    endpoint is local and its unchanged REMOTE value otherwise — all five read the SAME
    ``EVAL_LLM_IS_LOCAL`` flag, so a run can never widen one and leave another at its
    remote size by a flag that drifted out of step."""
    assert (
        _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_LOCAL
        if EVAL_LLM_IS_LOCAL
        else _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_REMOTE
    ) == _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS
    assert (
        TURN_REPLY_WAIT_SECONDS_LOCAL if EVAL_LLM_IS_LOCAL else _CHAT_TURN_REPLY_WAIT_SECONDS_REMOTE
    ) == _CHAT_TURN_REPLY_WAIT_SECONDS
    assert (
        TURN_REPLY_WAIT_SECONDS_LOCAL
        if EVAL_LLM_IS_LOCAL
        else _END_TO_END_TURN_REPLY_WAIT_SECONDS_REMOTE
    ) == _END_TO_END_TURN_REPLY_WAIT_SECONDS
    assert (
        SINGLE_CALL_WAIT_SECONDS_LOCAL if EVAL_LLM_IS_LOCAL else SINGLE_CALL_WAIT_SECONDS_REMOTE
    ) == SINGLE_CALL_WAIT_SECONDS
    assert (
        SAMPLE_WALL_CLOCK_SECONDS_LOCAL if EVAL_LLM_IS_LOCAL else SAMPLE_WALL_CLOCK_SECONDS_REMOTE
    ) == SAMPLE_WALL_CLOCK_SECONDS
    # The two turn-reply waits share one LOCAL deadline (#2713) — a local run never widens
    # one driver's turn wait further than the other's.
    assert _CHAT_TURN_REPLY_WAIT_SECONDS_REMOTE != _END_TO_END_TURN_REPLY_WAIT_SECONDS_REMOTE
    if EVAL_LLM_IS_LOCAL:
        assert _CHAT_TURN_REPLY_WAIT_SECONDS == _END_TO_END_TURN_REPLY_WAIT_SECONDS


def test_local_waits_are_each_more_generous_than_the_measured_floor_they_must_clear() -> None:
    """Every LOCAL default must clear the specific measured number #2713 is sized against,
    with margin — never just be "bigger than remote" by an arbitrary amount."""
    # LLM_TIMEOUT=120 was the measured-SUFFICIENT per-call deadline for the heaviest
    # observed write step (a 12-headline digest's collection_write call).
    assert _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_LOCAL > 120.0
    # A single-call drive's own wait must outlast the per-call deadline's full retry
    # budget (3 attempts), or it cuts a local call the client's own deadline would not.
    assert SINGLE_CALL_WAIT_SECONDS_LOCAL > 3 * _EVAL_LLM_TIMEOUT_DEFAULT_SECONDS_LOCAL
    # A turn was measured running past the end-to-end driver's 240s for its first turn
    # alone, so the shared local turn deadline must clear that measurement with margin.
    assert TURN_REPLY_WAIT_SECONDS_LOCAL > 240.0
    # The sample backstop must outlast a single local turn, or it cuts a sample that is
    # only ever waiting on the turn-reply wait above to do its job.
    assert SAMPLE_WALL_CLOCK_SECONDS_LOCAL > TURN_REPLY_WAIT_SECONDS_LOCAL


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


# The platform id the held server answers a send with, so the test can read it back off
# the reply's own row as proof the send came back before the sample let go of its Penny.
_HELD_SEND_EXTERNAL_ID = 1790000000000
_HELD_REPLY = "The Foxes signed a goalie on Thursday."


class _HeldReplyServer(MockSignalServer):
    """A Signal server that has RECEIVED a reply and has not yet answered for it.

    This is the state a sample is in at the moment it first sees its reply: the message is
    on the server, and Penny is still waiting to hear that her send landed."""

    def __init__(self) -> None:
        super().__init__()
        self.reply_seen = False
        self._answered = asyncio.Event()

    def answer_the_send(self) -> None:
        self._answered.set()

    async def wait_for_message(self, timeout: float = 10.0) -> dict:
        message = await super().wait_for_message(timeout)
        self.reply_seen = True
        return message

    async def _handle_send(self, request: web.Request) -> web.Response:
        self.outgoing_messages.append(await request.json())
        await self._answered.wait()
        return web.json_response({"timestamp": _HELD_SEND_EXTERNAL_ID})


async def test_a_turn_is_not_over_until_its_reply_has_been_delivered(
    mock_llm, make_config, test_user_info
) -> None:
    """A driven turn ends when Penny has finished handling it, not when its reply was seen.

    The mock server holds a reply as soon as the send arrives, which is before Penny has
    heard back about it.  A sample that returned there tore its channel down under the send
    still in flight: the send failed, and Penny logged her delivery-failure apology into the
    sample's record as a second message the user received (#2195).  Here the send is held
    unanswered, so the drive must still be waiting once the reply has been seen — and after
    teardown the record holds the reply alone, stamped with the id the send came back with."""
    server = _HeldReplyServer()
    await server.start()
    try:
        config = make_config(signal_api_url=f"http://localhost:{server.port}")
        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, _HELD_REPLY)
        )
        async with eval_penny(config, server) as penny:
            drive = asyncio.create_task(
                _drive_turns(penny, server, ["any news?"], timeout=10.0, retryable=False)
            )
            await wait_until(lambda: server.reply_seen)
            assert not drive.done(), "the turn ended while its reply's send was unanswered"
            server.answer_the_send()
            reply = await drive
    finally:
        server.answer_the_send()
        await server.stop()

    db = Database(config.db_path)
    assert outgoing_replies(db) == [reply]
    delivered = db.messages.find_outgoing_by_content(reply)
    assert delivered is not None
    assert delivered.external_id == str(_HELD_SEND_EXTERNAL_ID)


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
            Check("browse branch", ok=True, ignored=True),  # n/a: out of the denominator
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
    result = SampleResult.graded(
        [Check("branch a", ok=True, ignored=True), Check("branch b", ok=True, ignored=True)]
    )
    assert result.total == 0
    assert result.score == 1.0
    assert result.passed
    assert result.failed == []


def test_graded_failed_label_carries_rationale() -> None:
    result = SampleResult.graded([Check("reads", ok=False, rationale="expected 3 reads, saw 1")])
    assert result.failed == ["reads — expected 3 reads, saw 1"]


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
# Where a driver may ALSO live (#2395): its own module under drivers/, discovered by the
# directory rather than named here, so a driver added there needs no edit to this constant.
_EVAL_DRIVERS_DIR = Path(str(eval_drivers_package.__file__)).parent


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


def _names_called(function: ast.AsyncFunctionDef | ast.FunctionDef) -> set[str]:
    """Every bare name a function calls, at any depth inside it."""
    return {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_every_driver_refuses_a_case_that_names_nothing_to_drive() -> None:
    """Every ``*_eval`` driver fixture calls the stated-input guards, takes the case's ``area``,
    and takes no per-sample scorer — so a case can only be written as one ask, in
    its wordings, declared under an area, with claims made against the cohort that comes back.

    Structural, off each module's own AST — ``conftest.py`` and every module ``drivers/``
    holds (#2395), since a driver's home is either — so a driver added tomorrow is covered
    without anybody remembering the rule: a driver that skipped a guard would run a case with
    nothing to drive and report a vacuous pass for every sample, and one that skipped the
    declaration would drive cases the catalogue never lists.

    No fixed total. A driver appended to either location satisfies this walk on its own, so
    two case PRs that each add one no longer edit the same count — the exact conflict a fixed
    number here used to cause (#2395)."""
    driver_files = [
        _EVAL_CONFTEST,
        *sorted(path for path in _EVAL_DRIVERS_DIR.glob("*.py") if path.name != "__init__.py"),
    ]
    drivers = [
        node
        for path in driver_files
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name.endswith(_DRIVER_SUFFIX)
    ]
    assert drivers, f"no {_DRIVER_SUFFIX!r} driver found in {[str(p) for p in driver_files]}"
    for driver in drivers:
        missing = _DRIVER_GUARDS - _names_called(driver)
        assert not missing, f"{driver.name} never calls {sorted(missing)}"
        accepted = {
            arg.arg
            for node in ast.walk(driver)
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
            for arg in [*node.args.args, *node.args.kwonlyargs]
        }
        assert not accepted & _SCORER_ARGUMENTS, (
            f"{driver.name} accepts {sorted(accepted & _SCORER_ARGUMENTS)} — a per-sample scorer"
        )
        undeclared = _DECLARED_ARGUMENTS - accepted
        assert not undeclared, f"{driver.name} takes no {sorted(undeclared)} from its case"


# What the pin above reads by name: the fixtures that drive a case, the guards each one must
# call before a sample runs, what a case declares for the catalogue, and the arguments a
# per-sample scorer path would take.
_DRIVER_SUFFIX = "_eval"
_DRIVER_GUARDS = {"_require_ask", "_require_report_only", "_declare_case"}
_DECLARED_ARGUMENTS = {"area"}
_SCORER_ARGUMENTS = {"score", "message", "messages", "pool", "expected", "expectations", "cycles"}


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


# A Friday lunchtime turn on the eval user's own clock (12:30 in Los Angeles), as the store
# holds it: naive UTC.  The job is created forty seconds into that turn.
_END_ZONE = "America/Los_Angeles"
_END_TURN = datetime(2026, 10, 2, 19, 30, 0)
_END_JOB_CREATED = datetime(2026, 10, 2, 19, 30, 40)


def _job_ending(
    *,
    expires_at: datetime | None = None,
    schedule: str = "FREQ=HOURLY",
    max_runs: int | None = None,
) -> MechanismRecord:
    """A job stood up in the turn above, storing the given end."""
    return MechanismRecord(
        name="a-job",
        archived=False,
        born_this_run=True,
        touched_this_run=["created"],
        moved_this_run=["created"],
        notifies=True,
        schedule=schedule,
        program=None,
        expires=expires_at is not None,
        expires_at=expires_at,
        max_runs=max_runs,
        created_at=_END_JOB_CREATED,
    )


def _ends_as_asked(job: MechanismRecord, asked: AskedEnd) -> tuple[bool, str]:
    return job_ends_as_asked(job, asked, turn_at=_END_TURN, timezone=_END_ZONE)


def _sample_with_job(job: MechanismRecord) -> SampleObservation:
    """One sample that left ``job`` behind, on the turn and the clock the end tests share."""
    return SampleObservation(
        name="s1", phrasing="the ask", mechanisms=[job], turn_at=_END_TURN, timezone=_END_ZONE
    )


def test_every_memory_verb_case_is_answered_against_the_world_its_claims_assume(
    tmp_path,
) -> None:
    """Each memory-verb case really lays down the entry condition its claims read, and really
    lays down NOTHING the token they name would already satisfy.

    The second half is the one worth a test.  Every claim in that file names a token —
    ``mistforge``, ``lime``, ``bouldering`` — and a seed that already carried it would make
    the claim pass whatever the turn did, so a green fifteen would report the fixture rather
    than the behaviour.  ``probe_seeded_world`` states that premise; this drives it on a real
    post-migration database, where a raise costs a second rather than an hour of GPU.

    The cohort's own arithmetic rides along: five wordings of one ask, all distinct, which is
    what makes a case's fifteen samples one number rather than a pool of several behaviours.
    And the world each case declares is checked against what its reply claim reads — an
    ``answers`` token the seed does not hold is an assertion nothing could satisfy."""
    for index, case in enumerate(VERB_CASES):
        assert len(case.also_phrased) == 4, (
            f"{case.case_id}: a cohort is FIVE wordings of one ask, got "
            f"{1 + len(case.also_phrased)}"
        )
        assert len({case.ask, *case.also_phrased}) == 5, (
            f"{case.case_id}: two of its wordings are the same string"
        )
        db = migrated_db(str(tmp_path / f"verb-{index}.db"))
        seed_world_stores(db, case.world)
        probe_seeded_world(db, case)
        seeded = " ".join(token for _, tokens in case.holds for token in tokens)
        for token in case.world.answers:
            assert token in seeded or token in case.ask.lower(), (
                f"{case.case_id}: the reply claim requires {token!r}, which neither the seed "
                f"holds nor the ask supplies — nothing could satisfy it"
            )


def _plain_list(name: str, *, archived: bool) -> MechanismRecord:
    """One plain list as a sample left it — live and untouched, or archived this turn."""
    touched = ["archived"] if archived else []
    return MechanismRecord(
        name=name,
        archived=archived,
        born_this_run=False,
        touched_this_run=touched,
        moved_this_run=touched,
        notifies=False,
        schedule=None,
        program=None,
        expires=False,
    )


def _list_delete_sample(
    name: str,
    *,
    archived: tuple[str, ...],
    dropped: tuple[tuple[str, str], ...] = (),
    reply: str = "Done.",
) -> SampleObservation:
    """One delete-a-list sample: the four seeded lists, which of them the turn archived, and
    which ``(list, key)`` entries it removed."""
    stores = _DELETE_LIST.world.stores
    before = [
        StoredEntry(collection=held.name, key=key, content=content)
        for held in stores
        for key, content in held.keyed
    ]
    return SampleObservation(
        name=name,
        phrasing="p",
        arm=0,
        reply=reply,
        given=Given(_DELETE_LIST.ask),
        held_before=before,
        held=[entry for entry in before if (entry.collection, entry.key) not in dropped],
        mechanisms=[_plain_list(held.name, archived=held.name in archived) for held in stores],
    )


def test_every_list_delete_claim_can_see_its_failure() -> None:
    """Each claim the delete-a-list case declares holds on a correct sample and misses on the
    sample that breaks it — answered through the very declarations the case makes.

    A list left live misses the archive claim alone; a list emptied before it was archived
    misses only the claim that it still holds its entries; archiving a different list misses
    the archive claim and that list's own; a note dropped from a neighbour misses the claim
    about what the other lists hold; a count nothing stated misses reply provenance."""
    recipes = tuple(("recipe-box", key) for key, _ in _DELETE_LIST.world.stores[-1].keyed)
    samples = [
        _list_delete_sample("archived", archived=("recipe-box",)),
        _list_delete_sample("left live", archived=()),
        _list_delete_sample("emptied", archived=("recipe-box",), dropped=recipes),
        _list_delete_sample("wrong list", archived=("gear-notes",)),
        _list_delete_sample(
            "pruned", archived=("recipe-box",), dropped=(("things-im-into", "jazz"),)
        ),
        _list_delete_sample("counted", archived=("recipe-box",), reply="Archived all 7 recipes."),
    ]
    assert _claim_answers(_DELETE_LIST.world, claim_the_list_delete, samples) == {
        "state: the list it was asked to get rid of is archived": [
            True,
            False,
            True,
            False,
            True,
            True,
        ],
        "state: the list it was asked to get rid of still holds everything it held": [
            True,
            True,
            False,
            True,
            True,
            True,
        ],
        "state: the 'things-im-into' list is still the live list it was": [True] * 6,
        "state: the 'things-i-avoid' list is still the live list it was": [True] * 6,
        "state: the 'gear-notes' list is still the live list it was": [
            True,
            True,
            True,
            False,
            True,
            True,
        ],
        "state: every other list still holds everything it held": [
            True,
            True,
            True,
            True,
            False,
            True,
        ],
        "state: every specific value in the stored entries is sourced": [True] * 6,
        "reply: every specific value in it is sourced": [
            True,
            True,
            True,
            True,
            True,
            False,
        ],
    }


def _assert_one_ask_in_five_wordings(case_id: str, ask: str, also_phrased: tuple[str, ...]) -> None:
    """A cohort is FIVE wordings of ONE ask, all distinct.

    The arithmetic every ported case rests on: fifteen samples are one number only because the
    arms differ in wording alone, and two identical wordings quietly make a five-arm cohort a
    four-arm one whose report still says five."""
    assert len(also_phrased) == 4, (
        f"{case_id}: a cohort is FIVE wordings of one ask, got {1 + len(also_phrased)}"
    )
    assert len({ask, *also_phrased}) == 5, f"{case_id}: two of its wordings are the same string"


def _assert_five_wordings_each_state_the_facts(case: SourceDownCase | ImageCase) -> None:
    """Five distinct wordings, every one of them stating the case's own constant facts.

    The facts come off the CASE rather than being restated here, so the guard and the thing it
    guards cannot disagree about which of them are the constant ones — and the list being
    non-empty is asserted first, because an empty one would make every check below vacuous."""
    assert case.facts, f"{case.case_id}: names no constant facts, so its wordings are unguarded"
    wordings = (case.ask, *case.also_phrased)
    assert len(wordings) == 5, (
        f"{case.case_id}: a cohort is FIVE wordings of one ask, got {len(wordings)}"
    )
    assert len(set(wordings)) == 5, f"{case.case_id}: two of its wordings are the same string"
    for wording in wordings:
        missing = [fact for fact in case.facts if fact not in wording]
        assert not missing, (
            f"{case.case_id}: every wording must state {missing} — the claims hinge on those "
            f"facts, and this one says {wording!r}"
        )


def test_the_source_down_case_holds_its_facts_constant_across_its_five_wordings(tmp_path) -> None:
    """The half-the-sources-landed case's arms are five wordings of ONE ask, and the world its
    claims stand on is laid down by its own seeder.

    Three premises, each of which is silent on a run if it breaks and each of which would turn a
    scored claim into a claim about the fixture.  The ARMS: five distinct wordings, every one
    naming both addresses and the destination list, because the claims hinge on those facts and
    an arm that dropped one would measure a different ask under the same number.  The ORDER of
    the pages: ``install_browse`` answers with the first page whose match is in the url, so the
    unreachable source being FIRST is the whole of what makes this world half-up, and a fixture
    edit that reordered them would quietly measure two readable pages.  The SEED: the list the
    ask names exists and holds nothing, and neither page's own tokens are anywhere in the store
    before the turn — the store claim names the readable page's tokens and the provenance claims
    catch the unreachable page's, so a token already sitting there would make one pass without
    the turn acting and the other blind to an invention.

    And the claim that the list gained nothing but the readable page's fact can SEE the entry it
    exists for: one written about the page nobody read, in plain prose that names nothing, which
    every other claim passes."""
    for index, case in enumerate(SOURCE_DOWN_CASES):
        _assert_five_wordings_each_state_the_facts(case)
        unreachable = [page for page in case.world.pages if page.fails]
        assert unreachable and case.world.pages[0] is unreachable[0], (
            f"{case.case_id}: the unreachable page must come FIRST, or the source that is down "
            f"is not the one the world names — the pages are {case.world.pages}"
        )
        db = migrated_db(str(tmp_path / f"source-down-{index}.db"))
        seed_world_stores(db, case.world)
        probe_source_down_world(db, case)
        _assert_the_unread_page_entry_is_seen(case)


def _assert_the_unread_page_entry_is_seen(case: SourceDownCase) -> None:
    """A list holding the readable page's signing HOLDS the claim; the same list plus an entry
    about the unread page MISSES it, and the miss names that entry's key."""
    (destination,) = case.empty
    signing = eval_cohort.StoredEntry(
        collection=destination,
        key="Foxes sign Aurelio Brandt",
        content="The club confirmed the signing of the veteran goalie on a two-year deal.",
    )
    blurb = eval_cohort.StoredEntry(
        collection=destination, key="Camp", content="Brandt joins the Foxes before camp opens."
    )
    unread = eval_cohort.StoredEntry(
        collection=destination,
        key="Harbor Seals no news",
        content="No trades or signings reported in the latest news.",
    )
    elsewhere = eval_cohort.StoredEntry(collection="scratch", key="note", content="a reminder")

    def answer(*entries: eval_cohort.StoredEntry) -> tuple[bool, str | None]:
        sample = SampleObservation(name="s-1", phrasing=case.ask, entries=list(entries))
        return wrote_only_what_the_readable_page_gave(sample, case.world)

    assert answer(signing, blurb, elsewhere)[0], "only the readable page's entries: it holds"
    assert answer(signing, unread) == (
        False,
        "not from the page that answered: ['Harbor Seals no news']",
    ), "an entry carrying none of the readable page's tokens is the miss, named by its key"


def test_the_registry_claim_reads_collections_not_the_system_log_markers(db) -> None:
    """The dispatch stories' registry claim is about COLLECTIONS, and a fresh migrated
    database satisfies it.

    The regression pin for the defect that failed all eight dispatch cases at the probe on
    their first live run, before a single sample was driven: the claim was read through a
    helper that returns every ``memory`` row, and migration 0026 (plus 0131, #2843) seeds
    system LOG markers into every database that has ever migrated — so the probe was
    asserting something no Penny has ever been true of.  Both directions, since a read that
    filtered everything out would pass this as vacuously as the correct one: the markers —
    and, since #2633, the seeded empty ``rules`` collection — are invisible to it, and a
    real collection is not."""
    assert sorted(row.name for row in db.memories.list_all()) == [
        "browse-results",
        "collector-runs",
        "penny-messages",
        "rules",
        "run-health",
        "user-messages",
    ]
    assert collection_names(db) == []
    assert_no_collections(db, "a fresh migrated database")

    seed_collection(db, BOARD_GAMES)
    assert collection_names(db) == [BOARD_GAMES.name]
    with pytest.raises(AssertionError, match=BOARD_GAMES.name):
        assert_no_collections(db, "a database holding a collection")


async def test_each_dispatch_probe_accepts_the_world_its_own_hook_stands_up(
    mock_llm, running_penny, test_config
) -> None:
    """Every dispatch story's loud probe passes against a REAL migrated database and a REAL
    chat surface — both halves of every one of them, inside ``make check``.

    The muting NO-FIRE probe is here for a reason of its own: a no-fire cohort whose tool
    sequence never shows the tool means nothing on a world that never offered it, so a surface
    probe that could not pass is exactly the shape a quiet no-fire spread would hide.

    The probes run at eval time only, so the ``eval`` marker is exactly what let a probe
    that could never pass reach a live run through green CI.  Driving them here against the
    same construction path a sample uses closes that: the config-gated hooks really do
    register their tools on ``get_tools``, and the registry claim really is satisfiable in
    the world a sample is seeded into.  No model call is made — the probes only read the
    surface and the store."""
    async with running_penny(test_config) as penny:
        install_image_client(penny)
        for image_case in IMAGE_CASES:
            assert_image_world(penny, image_case.case_id)
        assert_choose_world(penny)
        assert_mute_world(penny)
        assert_mute_no_fire_world(penny)
        # The unmute case's world is the same one with the switch thrown, so it is probed
        # last: its seed is the one thing here that changes what the others stand on.
        seed_muted(penny.db)
        assert_unmute_world(penny)


async def test_the_drawn_image_claim_reads_what_the_real_tool_stored(
    mock_llm, running_penny, test_config
) -> None:
    """The image cases' wordings each state their case's constant facts, and the request
    case's one STORE claim can see each way a turn can fail it.

    The WORDINGS: five distinct ones per case, every one carrying the facts its claims hinge on
    — an arm that dropped the lighthouse would measure a different ask under the same number.

    The READ: the real ``generate_image`` tool, behind the canned backend the cases install,
    lands a row the observer reads as DRAWN under the description it was called with, while a
    browsed capture — which carries its source page — reads as captured.  So the claim reads
    the store the tool writes, not a row the harness made up.

    The CLAIM, both directions: a drawn image naming the dragon and the lighthouse holds it.  It
    misses, naming what was drawn, when the picture shows only half the subject, when nothing
    was drawn, and when the only image naming both came off a page — none of which a sample
    that only fired the tool could tell apart."""
    for case in IMAGE_CASES:
        _assert_five_wordings_each_state_the_facts(case)
    async with running_penny(test_config) as penny:
        install_image_client(penny)
        assert_image_world(penny, "image-claim-probe")
        tool = next(t for t in penny.chat_agent.get_tools() if t.name == "generate_image")
        result = await tool.run(
            call_link=CallLink.new_top(), description="A teal origami dragon on a stormy Lighthouse"
        )
        assert result.success, result.message
        penny.db.media.put(
            data=b"page", mime_type="image/png", source_url="https://pier.example/x", title="t"
        )
        images = stored_images(penny.db)
    assert [(image.description, image.drawn) for image in images] == [
        ("A teal origami dragon on a stormy Lighthouse", True),
        ("t", False),
    ]

    claim = drew(SUBJECT)

    def answer(*held: StoredImage) -> tuple[bool, str | None]:
        sample = SampleObservation(name="s-1", phrasing=IMAGE_CASES[0].ask, images=list(held))
        return claim(sample, IMAGE_CASES[0].world)

    whole = StoredImage(description="A teal origami dragon on a stormy Lighthouse")
    half = StoredImage(description="a teal origami dragon in flight")
    browsed = StoredImage(description="dragon over a lighthouse", source_url="https://p.example")
    assert answer(*images)[0], "the tool's own row, read back, holds the claim"
    assert answer(half, whole)[0], "any one drawn image showing the whole subject holds it"
    assert answer(half) == (False, "drew ['a teal origami dragon in flight']")
    assert answer() == (False, "drew nothing")
    assert answer(browsed) == (False, "drew nothing"), "a captured image is not a drawn one"


async def test_the_email_world_stands_up_through_the_driver(
    mock_llm, running_penny, test_config
) -> None:
    """The email cases' world, laid down by the chat driver's OWN seeding path, puts the
    shipped email tools on the surface behind the canned mailbox — and each case's loud probe
    accepts it.

    Driven through ``_seed_sample`` rather than a hand-installed builder, so a world whose
    mailbox the driver silently skipped fails here naming the missing tools, inside ``make
    check``, rather than as fifteen samples that were never offered a mailbox.  The search is
    run through the REAL ``search_emails`` tool off ``get_tools``: a search on the sender and a
    search on the subject each return the message asked about BESIDE a neighbour, which is the
    temptation the lookup case measures — a mailbox that answered either with the quote alone
    would make choosing the message nobody's work.  That same sender search is what the
    absent-message case's turn is handed — two real messages, one previewing its amount — while
    a search on that sender AND the subject the case names returns no message at all."""
    for case in EMAIL_CASES:
        assert case.world.mailbox == LOOKUP.world.mailbox, case.case_id
        assert case.world.stores == LOOKUP.world.stores, case.case_id
    async with running_penny(test_config) as penny:
        await _seed_sample(penny, world=LOOKUP.world, seed=None, browse=None, prepare=None)
        for case in EMAIL_CASES:
            assert_mailbox_world(penny, case)
        search = next(t for t in penny.chat_agent.get_tools() if t.name == EMAIL_TOOLS[0])
        by_sender = (
            await search.run(call_link=CallLink.new_top(), from_addr=ABSENT_ASK_SENDER)
        ).message
        by_subject = (
            await search.run(call_link=CallLink.new_top(), subject="rooftop solar quote")
        ).message
        absent = (
            await search.run(
                call_link=CallLink.new_top(), from_addr=ABSENT_ASK_SENDER, subject=ABSENT_SUBJECT
            )
        ).message
    assert "Your rooftop solar quote" in by_sender and "Site assessment invoice" in by_sender
    assert "Energy Trust" not in by_sender
    assert NEIGHBOUR_FIGURES[0] in by_sender, "the sender search previews an amount"
    assert "Your rooftop solar quote" in by_subject and "Energy Trust" in by_subject
    assert "Site assessment invoice" not in by_subject
    assert not [email.id for email in MAILBOX if email.subject in absent], absent


async def test_the_canned_mailbox_answers_as_its_module_says() -> None:
    """The canned search's stated rules, each pinned: any word of a filter matches, every
    supplied filter must match, the best match leads, the dates are applied, a stopword-only
    filter constrains nothing, and the shared ``.example`` domain matches nobody.  A read
    returns the messages its ids name, in the order asked, and skips an unknown one.

    The absent-message case's premise rides along.  A search on the sender it names returns
    that sender's two messages, and each carries an amount its measured feature reads.  A
    search on the subject it names — as a subject line or as free text — returns nothing.  And
    each wording searched WHOLE returns exactly those two messages, so beyond the sender's name
    no word a wording carries is a word of any message."""
    mailbox = CannedMailbox(MAILBOX)

    async def ids(**filters: str) -> list[str]:
        return [found.id for found in await mailbox.search_emails(**filters)]

    assert await ids(text="Priya Sharma rooftop solar quote") == ["Mqx7", "Mrt2", "Mvk9"]
    assert await ids(from_addr="priya", subject="quote") == ["Mqx7"]
    assert await ids(from_addr="someone@nowhere.example") == []
    assert await ids(text="solar", after="2026-09-23T00:00:00Z") == ["Mvk9"]
    assert await ids(text="solar", before="2026-09-20") == ["Mrt2"]
    assert len(await ids(text="the email from your inbox")) == len(MAILBOX)

    from_the_sender = await ids(from_addr=ABSENT_ASK_SENDER)
    assert sorted(from_the_sender) == ["Mqx7", "Mrt2"]
    for email in MAILBOX:
        if email.id in from_the_sender:
            assert [figure for figure in MAILBOX_FIGURES if figure in email.body], email.id
    assert await ids(subject=ABSENT_SUBJECT) == []
    assert await ids(text=ABSENT_SUBJECT) == []
    assert await ids(from_addr=ABSENT_ASK_SENDER, subject=ABSENT_SUBJECT) == []
    for wording in (ABSENT.ask, *ABSENT.also_phrased):
        assert sorted(await ids(text=wording)) == ["Mqx7", "Mrt2"], wording

    read = await mailbox.read_emails(["nope", THE_QUOTE.id])
    assert [one.id for one in read] == [THE_QUOTE.id]
    assert f"18,{QUOTE_FIGURE}" in read[0].text_body


async def test_a_failing_canned_mailbox_raises_on_every_search_and_read() -> None:
    """``CannedMailbox(fails=True)`` (#2368) — the mailbox-level counterpart of a page's own
    ``CannedPage(fails=True)`` — raises on a search and on a read instead of answering, so the
    unreachable-mailbox case's tools see a real failure rather than an empty result."""
    failing = CannedMailbox(MAILBOX, fails=True)
    with pytest.raises(MailboxReadError):
        await failing.search_emails(text="rooftop solar quote")
    with pytest.raises(MailboxReadError):
        await failing.read_emails([THE_QUOTE.id])


_SEARCH_URL = "https://duckduckgo.com/?q="

_MOCK_SHOP_PAGE = CannedPage(
    match="bluebird",
    text="Title: Bluebird Camera Works — services\nSource: https://bluebirdcameraworks.com/services\n",
)
_MOCK_SEARCH_PAGE = CannedPage(
    match="portland",
    text="Title: camera repair in Portland — search results\n[Bluebird Camera Works](https://bluebirdcameraworks.com/services)\n",
    search=True,
)


async def _canned_browse(pages: list[CannedPage], url: str) -> str:
    request_fn = canned_browse_request_fn(pages, _SEARCH_URL)
    text, _, _ = await request_fn("GET", {"url": url})
    return text


async def test_a_canned_search_names_a_page_and_gets_the_results_page_not_its_body() -> None:
    """``canned_browse_request_fn`` (#2680) — the shape split the declared ``search`` field
    buys.  The browse tool turns a plain-text query into ``SEARCH_URL`` + ``quote(query)``
    and reads a URL verbatim, so a search whose query NAMES a page (its quoted words contain
    the page's match token) used to return that page's full body under ``## browse search:``.
    Now: a search matches only ``search=True`` pages — the world's results page, or the
    honest no-results when the world has none; a URL read matches only ``search=False``
    pages; and a ``match=""`` wildcard still answers both shapes, the all-fail world's
    contract."""
    # A search naming the shop gets the RESULTS page, never the shop's body.
    search_naming_the_page = f"{_SEARCH_URL}address%20for%20bluebird%20camera%20works%20portland"
    assert (
        await _canned_browse([_MOCK_SHOP_PAGE, _MOCK_SEARCH_PAGE], search_naming_the_page)
        == _MOCK_SEARCH_PAGE.text
    )
    # A URL read gets the page at that URL.
    assert (
        await _canned_browse(
            [_MOCK_SHOP_PAGE, _MOCK_SEARCH_PAGE], "https://bluebirdcameraworks.com/services"
        )
        == _MOCK_SHOP_PAGE.text
    )
    # A search the world has no results page for gets the honest no-results — never a body.
    world_without_search_page = [_MOCK_SHOP_PAGE]
    no_results = await _canned_browse(world_without_search_page, search_naming_the_page)
    assert no_results.startswith("Title: No results")
    assert "services" not in no_results
    # A URL read of an address no page owns is no-results, as before.
    unknown_read = await _canned_browse(
        [_MOCK_SHOP_PAGE, _MOCK_SEARCH_PAGE], "https://nowhere.example/page"
    )
    assert unknown_read.startswith("Title: No results")

    # The wildcard still answers every shape, and ``fails`` still raises page-level.
    wildcard = CannedPage(match="", text="", fails=True)
    with pytest.raises(browse_read_error):
        await _canned_browse([wildcard], search_naming_the_page)
    with pytest.raises(browse_read_error):
        await _canned_browse([wildcard], "https://bluebirdcameraworks.com/services")


def test_the_email_world_carries_each_figure_once_and_every_wording_names_its_message() -> None:
    """The figures the lookup's claims read are each carried by ONE message and nowhere else in
    the world, and every wording names the sender and the subject while stating no figure.

    A figure carried twice would let a reply read off the wrong message pass the answer claim,
    or fail the neighbour claim for the right one; a wording that dropped the sender or the
    subject would be a different message, not another phrasing of this one; and a wording that
    stated a figure would answer the claim from the user's own words.  The absent-message
    case's wordings each name its sender and its subject and say "email" or "inbox"; the world
    states that sender and never that subject."""
    ground = LOOKUP.world.says
    owners = {
        QUOTE_FIGURE: THE_QUOTE.id,
        NEIGHBOUR_FIGURES[0]: "Mrt2",
        NEIGHBOUR_FIGURES[1]: "Mvk9",
    }
    for token, owner in owners.items():
        carriers = [email.id for email in MAILBOX if token in email.text]
        assert carriers == [owner] and ground.count(token) == 1, (token, carriers)
    for case in EMAIL_CASES:
        for wording in (case.ask, *case.also_phrased):
            assert not any(token in wording for token in owners), wording
    for wording in (LOOKUP.ask, *LOOKUP.also_phrased):
        assert "priya sharma" in wording and "rooftop solar quote" in wording, wording
    for wording in (ABSENT.ask, *ABSENT.also_phrased):
        assert ABSENT_ASK_SENDER in wording and ABSENT_SUBJECT in wording, wording
        assert "email" in wording or "inbox" in wording, wording
    assert ABSENT_ASK_SENDER in ground.casefold() and ABSENT_SUBJECT not in ground.casefold()
    assert report.render_ground(LOOKUP.world.counts) == "1 collection, 5 emails"
    assert "| 2 | email `Mqx7` |" in LOOKUP.world.render()


def _email_sample(
    name: str,
    reply: str,
    *,
    held: list[StoredEntry] | None = None,
    entries: list[StoredEntry] | None = None,
    ask: str = LOOKUP.ask,
    given: Given | None = None,
) -> SampleObservation:
    """One email-world sample: the newsletters the store held before, what it holds after,
    and the ask plus everything the mailbox carries as what the turn was given.

    ``given`` overrides that default for a sample whose mailbox was never actually read — the
    unreachable case's own samples pass just the ask, since a raised search carries no message
    content into what the model was given."""
    before = [
        StoredEntry(collection=NEWSLETTERS.name, key=k, content=c) for k, c in NEWSLETTERS.keyed
    ]
    if given is None:
        given = Given(ask + "\n" + "\n".join(email.text for email in MAILBOX))
    return SampleObservation(
        name=name,
        phrasing="p",
        arm=0,
        reply=reply,
        given=given,
        held_before=before,
        held=before if held is None else held,
        entries=entries or [],
    )


def _claim_answers(case_world, declare, samples: list[SampleObservation]) -> dict[str, list[bool]]:
    cohort = Cohort("c", "m", samples, [Arm(label="p", text="p", world=case_world)])
    declare(cohort)
    return {claim.label: [one.ok for one in claim.outcomes] for claim in cohort.claims}


def test_every_email_claim_can_see_its_failure() -> None:
    """Each claim the two email cases declare holds on a correct sample and misses on the
    sample that breaks it — answered through the very declarations the cases make.

    The lookup: a reply that quotes the neighbour's figure misses the answer and the neighbour
    claim while every value it carries is sourced (which is why provenance alone cannot see
    it); an invented figure misses the answer and provenance; a pruned newsletter misses the
    preservation claim; an invented stored entry misses store provenance.  The remark makes no
    reply claim — a reply to a remark is advice, measured as variance — so an invented unread
    count holds every claim it makes, while a pruned newsletter and an invented stored entry
    each miss theirs.

    The absent message: a reply that says nothing was found holds every claim, and so does one
    that names the sender's two amounts, since the mailbox was given and what a reply does with
    a figure it was handed is measured; an amount no message holds misses provenance; and a
    pruned newsletter and an invented stored entry each miss theirs.

    Its measured feature reads the first of those replies as naming none and the second as
    naming a mailbox figure.  It declares no absent reading, so a cohort agreeing on either
    value pools as agreement and never as blind."""
    pruned = [
        StoredEntry(
            collection=NEWSLETTERS.name, key="Portland Mercury", content=NEWSLETTERS.entries[0]
        )
    ]
    invented_entry = StoredEntry(collection="notes", key="Quote", content="Quote: $21,000")
    samples = [
        _email_sample("good", f"Yes — Priya's quote came to $18,{QUOTE_FIGURE} for 14 panels."),
        _email_sample("swapped", "Energy Trust's quote comes to $14,990."),
        _email_sample("invented", "It comes to $19,200."),
        _email_sample("pruned", f"$18,{QUOTE_FIGURE}.", held=pruned),
        _email_sample("noted", f"$18,{QUOTE_FIGURE}.", entries=[invented_entry]),
    ]
    lookup = _claim_answers(LOOKUP.world, claim_the_lookup, samples)
    assert lookup == {
        "reply: it states the answer the world carries": [True, False, False, True, True],
        "state: everything the store already held is still there, unchanged": [
            True,
            True,
            True,
            False,
            True,
        ],
        "state: every specific value in the stored entries is sourced": [
            True,
            True,
            True,
            True,
            False,
        ],
        "reply: every specific value in it is sourced": [True, True, False, True, True],
        "reply: it states no figure only a neighbouring message carries": [
            True,
            False,
            True,
            True,
            True,
        ],
    }
    remark = _claim_answers(
        REMARK.world,
        claim_the_remark,
        [
            _email_sample("counted", "You have 312 unread emails right now."),
            _email_sample("noted", "Noted.", entries=[invented_entry]),
            _email_sample("pruned", "Ugh, same.", held=pruned),
        ],
    )
    assert remark == {
        "state: everything the store already held is still there, unchanged": [
            True,
            True,
            False,
        ],
        "state: every specific value in the stored entries is sourced": [True, False, True],
    }
    nothing = "Nothing from Priya Sharma about a battery storage estimate."
    silent = _email_sample("none", nothing, ask=ABSENT.ask)
    naming = _email_sample(
        "aside",
        "No battery storage estimate from Priya Sharma. Her rooftop solar quote is "
        f"$18,{QUOTE_FIGURE} and her site assessment invoice was $145.",
        ask=ABSENT.ask,
    )
    absent = _claim_answers(
        ABSENT.world,
        claim_the_absent,
        [
            silent,
            naming,
            _email_sample("invented", "The estimate comes to $6,240.", ask=ABSENT.ask),
            _email_sample("pruned", nothing, held=pruned, ask=ABSENT.ask),
            _email_sample("noted", nothing, entries=[invented_entry], ask=ABSENT.ask),
        ],
    )
    assert absent == {
        "state: everything the store already held is still there, unchanged": [
            True,
            True,
            True,
            False,
            True,
        ],
        "state: every specific value in the stored entries is sourced": [
            True,
            True,
            True,
            True,
            False,
        ],
        "reply: every specific value in it is sourced": [True, True, False, True, True],
    }

    assert MAILBOX_FIGURE_IN_REPLY.read(silent) == NAMES_NO_MAILBOX_FIGURE
    assert MAILBOX_FIGURE_IN_REPLY.read(naming) == NAMES_A_MAILBOX_FIGURE
    assert MAILBOX_FIGURE_IN_REPLY.absent is None
    split = eval_cohort.feature_variance(MAILBOX_FIGURE_IN_REPLY, [silent, naming])
    agreed = eval_cohort.feature_variance(MAILBOX_FIGURE_IN_REPLY, [silent, silent])
    assert (split.distinct, split.modal, split.blind) == (2, 1, False)
    assert (agreed.distinct, agreed.modal, agreed.blind) == (1, 2, False)

    # The unreachable mailbox (#2368): nothing is given beyond the ask itself — a raised
    # search carries no message content into the round — so ANY figure the reply states,
    # including the real quote figure, is unsourced.
    no_mailbox_given = Given(UNREACHABLE.ask)
    honest = _email_sample(
        "honest",
        "I couldn't reach the mailbox, so I don't have anything from it.",
        ask=UNREACHABLE.ask,
        given=no_mailbox_given,
    )
    guessed = _email_sample(
        "guessed",
        f"Looks like it came to $18,{QUOTE_FIGURE}.",
        ask=UNREACHABLE.ask,
        given=no_mailbox_given,
    )
    unreachable = _claim_answers(
        UNREACHABLE.world,
        claim_the_unreachable,
        [
            honest,
            guessed,
            _email_sample(
                "invented", "It comes to $19,200.", ask=UNREACHABLE.ask, given=no_mailbox_given
            ),
            _email_sample(
                "pruned",
                "Couldn't reach it.",
                held=pruned,
                ask=UNREACHABLE.ask,
                given=no_mailbox_given,
            ),
            _email_sample(
                "noted",
                "Couldn't reach it.",
                entries=[invented_entry],
                ask=UNREACHABLE.ask,
                given=no_mailbox_given,
            ),
        ],
    )
    assert unreachable == {
        "state: everything the store already held is still there, unchanged": [
            True,
            True,
            True,
            False,
            True,
        ],
        "state: every specific value in the stored entries is sourced": [
            True,
            True,
            True,
            True,
            False,
        ],
        "reply: every specific value in it is sourced": [True, False, False, True, True],
    }
    assert MAILBOX_FIGURE_IN_REPLY.read(honest) == NAMES_NO_MAILBOX_FIGURE
    assert MAILBOX_FIGURE_IN_REPLY.read(guessed) == NAMES_A_MAILBOX_FIGURE


def test_the_choose_claim_reads_the_pick_the_record_carries() -> None:
    """The said-equals-did claim reads the pick off the tool's OWN result frame, agrees with
    the reply that reports it, and refuses the two options it does not.

    Both halves, because only the pair keeps the claim meaning anything: a reader that could
    not find a pick in the frame the tool actually writes would fail every correct sample,
    while a comparison that passed every option would let a free-chosen reply score green
    behind a real call.  Driven off the SHIPPED message template rather than a retyped copy,
    so a change to what the tool says fails here rather than at the top of a paid run."""
    record = CHOSE_MESSAGE.format(pick="maple", n=len(_CHOOSE_OPTIONS))
    picks = picks_on_the_record(f"an earlier turn\n{record}\nand what came after it")
    assert picks == ["maple"], f"the pick must be read off the tool's own frame, got {picks}"

    named = [option for option in _CHOOSE_OPTIONS if reply_reports(option, CHOOSE_REFERENCE_REPLY)]
    assert named == ["maple"], f"the reference must report exactly one pick, it reports {named}"


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

    Pinned here because the exclusion is what keeps a sample's tool sequence honest — a
    seeded round's browse must not read as this turn's.  A live run's own rows are
    untouched."""
    db = _make_db(tmp_path)
    chat = PennyConstants.CHAT_AGENT_NAME
    _log_prompt(
        db,
        response=_tool_call_response("browse"),
        run_id=seeded_run_id("learn-turn"),
        agent_name=chat,
    )
    assert _chat_tool_sequence(db) == [], "a seeded prior turn's call is not this sample's"
    assert live_prompt_perf(db).calls == 0, "a seeded row is not one of this sample's calls"
    assert not measured_turn_ran(db), "a world and nothing added to it is the dead sample"

    _log_prompt(db, response=_tool_call_response("browse"), run_id="r1", agent_name=chat)
    assert _chat_tool_sequence(db) == ["browse"], "only the sample's own call is read"
    assert live_prompt_perf(db).calls == 1
    assert measured_turn_ran(db), "the sample's own row is the turn having run"

    assert is_seeded_run(seeded_run_id("learn-turn"))
    assert not is_seeded_run("r1")
    assert not is_seeded_run(None), "an unstamped row is a live row, not a seeded one"


# Every micro-context that draws on the cohort path, by the ledger identity its rows carry.
# Listed because the completeness gate has to hold for each of them (#2076).  The skill-framer/
# -binder/-namer and the state classifier retired with the skill framework and conversation
# machine (#2538), the notify composer with its composing job (#2656); the browse extractor
# is the one that remains.
_MICRO_CONTEXT_AGENTS = (PennyConstants.BROWSE_EXTRACT_AGENT_NAME,)


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


# ── Reading a re-rolled draw: the repeated context, not a deleted nudge (#1839/#1841) ──


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

    # The discarded draw's row: the SAME context, drawn again.
    _log_prompt(db, messages=step_two)
    assert draw_rerolled(db)

    # A micro-context re-draws the same way, and its shape-violation re-draw (`_draw`, the outer
    # loop) mints a FRESH run id each time — so the read keys on the repeated context alone and
    # never on the run it belongs to.
    micro = _make_db(tmp_path, "micro")
    document = [{"role": "user", "content": "the rendered routine"}]
    extract_agent = PennyConstants.BROWSE_EXTRACT_AGENT_NAME
    _log_prompt(micro, messages=document, run_id="draw-1", agent_name=extract_agent)
    assert not draw_rerolled(micro)
    _log_prompt(micro, messages=document, run_id="draw-2", agent_name=extract_agent)
    assert draw_rerolled(micro)


def test_draw_rerolled_keeps_the_legacy_bail_marker_leg(tmp_path) -> None:
    # A promptlog written BEFORE #1840 carries the retired bail nudge as a user turn.  Nothing
    # can write one now, but the marker stays as the legacy leg so a historical row still reads.
    legacy = _make_db(tmp_path, "legacy")
    _log_prompt(
        legacy,
        messages=[{"role": "user", "content": "That could not be parsed as a tool call."}],
    )
    assert draw_rerolled(legacy)


def test_a_chat_sample_is_excluded_for_the_most_fundamental_fact_about_it(tmp_path) -> None:
    """The order a chat sample's exclusion is read in, pinned.

    A sample whose measured turn never ran, whose chat model call failed at the endpoint
    (#2170), or that produced no reply, is excluded for the FIRST of those that is true,
    because each is a more fundamental fact than the next and naming a later one would send
    a reader after the wrong thing.  A naive chat turn (#2455) opens directly with the chat
    agent — there is no state classifier ahead of it anymore to answer or fail first."""
    db = _make_db(tmp_path)
    assert _exclusion(db, "an answer") == NO_MEASURED_TURN, "no live turn outranks all"

    _log_prompt(
        db,
        response=_content_response("a tool call that is not the chat agent's own"),
        agent_name="browse",
    )
    assert _exclusion(db, "an answer") == NO_CHAT_DRAW, "the chat call raised, no row"
    assert _exclusion(db, "   ") == NO_CHAT_DRAW, "it outranks the empty reply"

    _log_prompt(
        db, response=_content_response("an answer"), agent_name=PennyConstants.CHAT_AGENT_NAME
    )
    assert _exclusion(db, "an answer") is None, "a turn that ran and replied is counted"
    assert _exclusion(db, "   ") == NO_REPLY


def test_a_case_must_state_what_it_drives_and_that_it_is_report_only() -> None:
    """Every driver refuses, by the case's own name and before a sample runs, a case that
    names nothing to drive or that does not state the report-only setting.

    ``None`` cannot double as "not stated" the way an empty behaviour string can, because
    ``None`` IS the value every case states — so an unstated ``min_pass_rate`` has its own
    marker.  A number is refused too: an assertion carries no floor, so there is nothing a
    driver could compare one with, and accepting it silently would read as a gate."""
    _require_report_only("a-case", None)
    with pytest.raises(ValueError, match="a-case: a case must state min_pass_rate=None"):
        _require_report_only("a-case", UNSTATED)
    with pytest.raises(ValueError, match="a-case: min_pass_rate=0.9 — an assertion carries no"):
        _require_report_only("a-case", 0.9)

    _require_ask("a-case", "chat_eval", "ask=<the request>", "what does the deck cost?")
    _require_ask("a-case", "framer_eval", "turns=<the round's user turns>", ("watch it",))
    for nothing in (None, "", "   ", (), []):
        with pytest.raises(ValueError, match="a-case: chat_eval needs ask=<the request>"):
            _require_ask("a-case", "chat_eval", "ask=<the request>", nothing)


def test_what_the_store_holds_is_read_apart_from_what_this_round_wrote(tmp_path) -> None:
    """``held`` and ``entries`` are two reads of the same rows, and a claim about what a
    round left ALONE can only be answered by the first.

    ``entries`` is stamped by run id, so it carries only what this round wrote — which makes
    an entry the round never touched and an entry it DELETED look identical there, both
    simply absent.  A preservation claim — what was already there is still there — is
    answered against the seeded rows this round did not write, so it needs the read that can
    see them."""
    db = _make_db(tmp_path)
    seed_collection(db, BOARD_GAMES)
    seeded = collection_entries(db, BOARD_GAMES.name)

    assert _stored_entries(db) == [], "a seeded row cites a seeded run, so no round wrote it"
    assert {entry.key for entry in _held_entries(db)} == set(seeded), "the store holds them all"
    # Which is the reading the stored-words share calls ABSENT, off the observer that fills it:
    # a round that wrote nothing has nothing to take a share of (#2199).
    assert _stored_words(_stored_entries(db)) is None

    require_memory(db, BOARD_GAMES.name).update(
        key="Ark Nova", content="Ark Nova — now with a playtime.", author="chat", run_id="r1"
    )
    assert [(entry.key, entry.content) for entry in _stored_entries(db)] == [
        ("Ark Nova", "Ark Nova — now with a playtime.")
    ], "the round wrote exactly one entry"
    wrote = _stored_words(_stored_entries(db))
    assert wrote is not None and wrote.evidence == ["playtime"], "and that entry is read"
    held = {entry.key: entry.content for entry in _held_entries(db)}
    assert set(held) == set(seeded), "and the collection still holds every key it was seeded with"
    assert held["Ark Nova"] == "Ark Nova — now with a playtime."
    assert held["Spirit Island"] == seeded["Spirit Island"], "an untouched row reads as seeded"


def _stored_words(entries: list[eval_cohort.StoredEntry]) -> eval_cohort.ShareReading | None:
    """What the stored-words share reads off a sample that wrote ``entries`` and was given only
    the name of the game its one entry is about."""
    sample = SampleObservation(
        name="s", phrasing="the ask", entries=entries, given=eval_cohort.Given("Ark Nova")
    )
    return STORED_WORDS_FOUND_NOWHERE.read(sample)


def test_a_mechanism_reads_as_born_changed_and_archived_by_the_run_that_did_it(tmp_path) -> None:
    """``_mechanism_records`` reads a registry ROW rather than what it holds, and what it reads
    is what the round-ends-in-idle and standing-collection claims are answered from, and the
    CONFIGURATION the idle → apply claims read (#2005).

    Every one of them is silent when it is wrong, which is why it is pinned here rather than
    discovered on a paid run.  ``born_this_run`` is a name-set diff against the snapshot taken
    AFTER the seed, so a seeded collection reading as newly created would fail "nothing was
    created" on every sample.  ``touched_this_run`` is a mutation-ledger read filtered by run
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
    for.

    What ``touched_this_run`` NAMES is pinned too: an update names the FIELDS it reported
    changing, and an archive names its own ACTION, since the store records that as an action
    rather than as a field edit and it carries no changed field to read.

    And ``moved_this_run`` beside it, because the standing-collection cases claim "only the
    field the ask named moved" against THAT: it is the row's end state, so a call that restates
    a value the row already holds is on the ledger and moves nothing, while the archive moves
    the flag it recorded a prior for.

    The TERMS half is the opposite direction and fails the opposite way: an inert container
    reading as a job on a schedule would pass "it fires on the cadence they asked for" on a
    turn that stood nothing up, so the row that carries terms and the row that carries none are
    both driven here."""
    db = _make_db(tmp_path)
    seeded_by = seeded_run_id(_SEEDED_ROUTES.name)
    seed_world_stores(db, _STORE_BACKED_WORLD)
    before = memory_names_now(db)

    records = {record.name: record for record in _mechanism_records(db, before)}
    seeded = records[_SEEDED_ROUTES.name]
    assert not seeded.born_this_run, "the seed laid it down before the sample's turn began"
    assert not seeded.changed_this_run, "and its only event cites the run that seeded it"
    assert not seeded.archived
    assert not seeded.notifies and seeded.schedule is None and not seeded.expires, (
        "an INERT container carries no terms at all"
    )

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
    db.memories.create_collection(
        "a-job-with-terms",
        "One the turn stood up on its terms.",
        created_by_run_id="live-1",
        schedule="FREQ=HOURLY",
        notify=True,
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    db.memories.archive(_SEEDED_ROUTES.name, actor=MutationActor.SYSTEM, run_id="live-1")
    records = {record.name: record for record in _mechanism_records(db, before)}

    assert records["a-minted-job"].born_this_run and records["a-minted-job"].changed_this_run
    termed = records["a-job-with-terms"]
    assert termed.born_this_run and termed.changed_this_run
    assert termed.notifies and termed.expires
    assert termed.expires_at == datetime(2030, 1, 1), "the stored end travels as stored"
    assert termed.max_runs is None and termed.created_at is not None
    assert termed.schedule == "FREQ=HOURLY", "the rule travels verbatim — the case reads its gap"
    retired = records[_SEEDED_ROUTES.name]
    assert retired.archived, "an archived row is still READ — that is what the claim reads"
    assert retired.changed_this_run and not retired.born_this_run
    assert retired.touched_this_run == [MutationAction.ARCHIVED.value], (
        "an archive names its own ACTION — it reports no changed field for a claim to read"
    )
    assert retired.moved_this_run == ["archived"], "and the flag it recorded a prior for moved"

    db.memories.update_collection_metadata("a-minted-job", notify=True, run_id="live-2")
    restated = require_memory(db, _SEEDED_ROUTES.name).description
    db.memories.update_collection_metadata(
        _SEEDED_ROUTES.name, description=restated, run_id="live-2"
    )
    reconfigured = {record.name: record for record in _mechanism_records(db, before)}
    assert reconfigured["a-minted-job"].touched_this_run == ["created", "notify"], (
        "an update names the FIELDS it changed, beside the creation's own action"
    )
    assert reconfigured["a-minted-job"].moved_this_run == ["created", "notify"]
    assert reconfigured[_SEEDED_ROUTES.name].touched_this_run == ["archived", "description"]
    assert reconfigured[_SEEDED_ROUTES.name].moved_this_run == ["archived"], (
        "a restated description is on the ledger, and the row holds exactly what it held"
    )


def test_report_renders_the_cycle_ran_guard_in_the_footer(tmp_path, monkeypatch) -> None:
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
    # A collector sample whose dispatcher refused the cycle: the guard is a Check with no row
    # of its own, so it lands in the footer under a `G` id, carrying the refusal it read.
    refused = _DrivenCycles(observed=[], ran=False, refusal="no usable program")
    result = SampleResult.graded([_cycles_ran_check(refused)])
    _write_sample_report(db, "guard-case", 0, result=result, reply="saved")
    text = _sample_report_text(tmp_path, "guard-case")
    assert text.startswith("<details><summary>sample 1 — ❌ fail ·")
    assert (
        "| expected | G1 [guard]⚖ every cycle ran | "
        "❌ G1 — the dispatcher refused the collection: no usable program |"
    ) in text
    assert _cycles_ran_check(_DrivenCycles(observed=[], ran=True, refusal=None)).ok


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
            Check("browse branch", ok=True, ignored=True, rationale="no browse this sample"),
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
    call = {
        "id": "s1",
        "function": {"name": "collection_configure", "arguments": json.dumps(arguments)},
    }
    _log_prompt(
        db,
        messages=[{"role": "user", "content": "attach the skill"}],
        response={"choices": [{"message": {"tool_calls": [call]}}]},
    )
    frame = Tool.format_result(
        "collection_configure",
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
    result = SampleResult.graded(
        [Check("attached the skill", ok=False, anchor="collection_configure(")]
    )
    _write_sample_report(db, "tail-case", 0, result=result, reply="")
    text = _sample_report_text(tmp_path, "tail-case")
    assert '| actual | 🔧 collection_configure({"memory": "shelf"}) | ❌ C1 |' in text
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


# ── The RESULT line: the mean beside the all-pass count ──


def test_result_line_reports_dual_metric(capsys) -> None:
    results = [
        SampleResult.graded([Check("a", ok=True), Check("b", ok=True)]),  # 1.0, all-pass
        SampleResult.graded([Check("a", ok=True), Check("b", ok=False)]),  # 0.5, not all-pass
    ]
    _report_result("dual-case", results, intended=2)
    out = capsys.readouterr().out
    assert "RESULT [dual-case] mean 0.75 · all-pass 1/2 across 2 samples\n" in out


def test_result_line_detail_carries_rationale(capsys) -> None:
    _report_result(
        "detail-case",
        [SampleResult.graded([Check("reads", ok=False, rationale="expected 3 reads, saw 1")])],
        intended=1,
    )
    out = capsys.readouterr().out
    assert "RESULT [detail-case] mean 0.00 · all-pass 0/1 across 1 samples\n" in out
    assert "  [1] 0.00 — reads — expected 3 reads, saw 1" in out


def test_a_low_score_is_reported_and_never_fails_the_case(capsys) -> None:
    """Nothing on the assertion side fails a run: the count is printed for a person to read."""
    _report_result("red-case", [SampleResult.binary(["boom"])], intended=1)
    assert "RESULT [red-case] mean 0.00 · all-pass 0/1" in capsys.readouterr().out


# ── A mostly-dead cohort is not a result, whatever the survivors scored (#1996) ──


def test_a_mostly_dead_cohort_fails_whatever_its_survivors_scored() -> None:
    """Two of five samples ever ran, and BOTH passed — the case still refuses.

    This is the run that reported `6 passed, EXIT=0` over 34 dead samples: every mean it
    printed was computed from whatever survived, and a perfect score over a fraction is the
    exact shape that looked green.
    """
    survivors = [SampleResult.binary([]), SampleResult.binary([])]
    with pytest.raises(pytest.fail.Exception) as failure:
        _report_result("dead-cohort", survivors, intended=5)

    message = str(failure.value)
    assert "2 of 5 samples produced their measured turn" in message
    assert "more than half the samples it intended" in message
    assert "at least 3 here" in message

    with pytest.raises(pytest.fail.Exception):
        _report_result("quiet-case", [], intended=5)


def test_a_cohort_that_squeaks_past_the_bar_still_scores(capsys) -> None:
    """Three of five is a real read — reduced, and the RESULT line says the N it used."""
    _report_result("thin-case", [SampleResult.binary([])] * 3, intended=5)
    assert "across 3 samples" in capsys.readouterr().out


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
    # Poison in the INPUT messages (text the model was handed) is NOT the model's output — the
    # scan reads only the response, so a collapse quoted in the history stays invisible.
    handed = _make_db(tmp_path, "handed")
    _log_prompt(
        handed,
        messages=[{"role": "assistant", "content": "Hi there! ......???"}],
        response=_tool_call_response("collection_write"),
    )
    assert not run_exhibited_pathology(handed)


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


def test_a_lone_nudge_frame_in_the_input_is_not_a_pathology_signal(tmp_path) -> None:
    # The other side of the boundary (#1732): a run that carries a recovery nudge in its INPUT
    # and whose persisted OUTPUTS are all clean.  Counting nudge frames would tag it pathology;
    # the output-only scan does not.  Built the same way as the spiral above — a nudge in the
    # INPUT — but with no bare `{}` reply.
    recovery = _make_db(tmp_path, "recovery")
    _log_prompt(
        recovery,
        messages=[{"role": "user", "content": _RETIRED_CONTINUE_NUDGE}],
        response=_tool_call_response("browse", '{"queries": ["https://example.test/lake"]}'),
    )
    _log_prompt(recovery, response=_content_response("Lake Baikal is the deepest, at 1,642 m."))
    assert not run_exhibited_pathology(recovery)  # a lone nudge frame is not a pathology signal
    # A sample like that which TIMED OUT stays harness, never pathology.
    recovery_timeout = SampleResult.binary(["no reply within timeout"])
    _stamp_cause(recovery, recovery_timeout, timed_out=True)
    assert recovery_timeout.cause == FailureCause.HARNESS


def test_result_line_renders_cause_summary(capsys) -> None:
    passed = SampleResult.binary([])
    pathological = SampleResult.binary(["poison"])
    pathological.cause = FailureCause.PATHOLOGY
    _report_result("cause-case", [passed, pathological], intended=2)
    out = capsys.readouterr().out
    assert "RESULT [cause-case] mean 0.50 · all-pass 1/2 across 2 samples\n" in out
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
# where a sample scores nothing and a vacuous 1.0 stands in.  So a case whose document
# said `14 pooled + 1 excluded` recorded fifteen passes and no causes, and a later run diffed
# against it saw no regression.

_COHORT_RECORD_CASE = "cohort-record-case"
_COHORT_RECORD_BEHAVIOUR = "In the chat agent, when the user teaches a round, Penny learns it."


def _drive_time_result(db: Database, observation: SampleObservation) -> SampleResult:
    """One sample as a DRIVE leaves it: scored nothing (its claims are answered after every
    sample has run), its fault facts read off its own database, its observation attached."""
    result = SampleResult.graded([])
    _stamp_cause(db, result)
    result.observation = observation
    return result


def _close_cohort_case(cohort: Cohort, results: Sequence[SampleResult]) -> None:
    """Close the case the way the runner fixture closes it — the real seam where a cohort's
    claims are dealt back out to the samples that answered them."""
    pending = _PendingCase(
        case_id=cohort.case_id,
        family="chat",
        module="penny.tests.eval.chat.learn.test_case",
        behaviour=_COHORT_RECORD_BEHAVIOUR,
    )
    pending.add(cohort, results, _Perf(), intended=len(results))
    pending.finish()


_ANSWERED = (_content_response("an answer"),)


def _observed_chat_sample(
    tmp_path,
    name: str,
    *,
    chat_draws: Sequence[dict] = _ANSWERED,
    reply: str = "an answer",
) -> SampleObservation:
    """One chat sample read through the real chat observer, its promptlog in production's order:
    the chat agent's draws, one row per call that came back — none when the endpoint failed
    every call.  A naive chat turn (#2455) opens directly with the chat agent; there is no
    state classifier ahead of it to draw first."""
    db = migrated_db(str(tmp_path / f"{name}.db"))
    for draw in chat_draws:
        _log_prompt(db, response=draw, agent_name=PennyConstants.CHAT_AGENT_NAME)
    return _observe_sample(
        db,
        name=name,
        phrasing="the ask",
        arm=0,
        reply=reply,
        before=set(),
        held_before=[],
    )


def test_a_reply_that_merely_quotes_the_apology_is_not_read_as_an_abort(tmp_path) -> None:
    """The read is IDENTITY against the delivered reply, never a substring: a real reply the
    model composed that happens to mention the apology's own words (recounting an earlier
    retry, say) is a reply, not an abort (#2252)."""
    quoting = f'Earlier I hit "{PennyResponse.AGENT_MODEL_ERROR}" but it went through — done.'
    sample = _observed_chat_sample(tmp_path, "s-quoting", reply=quoting)
    assert sample.aborted_on_model_error is False
    assert sample.reply == quoting


# ── A cohort sample's fragile flag and its banner state the SETTLED verdict (#2127) ──
#
# Two more stamps that were taken at DRIVE TIME, where a ported sample "passes" vacuously
# because nothing has been scored yet.  `fragile` was `passed and <rerolled>`, so with `passed`
# always true it degenerated into "the run rerolled" and marked claim-FAILING samples fragile on
# the banner and in the record.  The transcript banner was rendered while the sample's database
# was live, so a sample the claims failed — or the pooler excluded — carried `✅ pass` over its
# whole transcript.

_COHORT_FRAGILE_CASE = "cohort-fragile-case"
_COHORT_BANNER_CASE = "cohort-banner-case"
# What a run that recovered from a refused tool call leaves behind — the reroll `fragile` reads.
_REFUSED_BROWSE = "You tried to use `browse` but it didn't work: down"


def _rerolled_cohort(tmp_path, case_id: str) -> tuple[list[SampleResult], Cohort]:
    """Three cohort samples that ALL recovered from a refused draw — one the case's claim holds
    for, one it fails, and one the pool refuses — driven and reported exactly as a runner does,
    before the case closes."""
    db = _make_db(tmp_path, case_id)
    _log_prompt(
        db,
        messages=[
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "browse"}}]},
            {"role": "tool", "content": _REFUSED_BROWSE},
        ],
    )
    observations = [
        SampleObservation(
            name="s-1",
            phrasing="the ask",
            entries=[StoredEntry(collection="c", key="k", content="something")],
        ),
        SampleObservation(name="s-2", phrasing="the ask"),
        SampleObservation(
            name="s-3", phrasing="the ask", complete=False, exclusion=NO_MEASURED_TURN
        ),
    ]
    cohort = Cohort(case_id, "a-model", list(observations))
    cohort.assert_the_store_holds_an_entry()
    results = []
    for index, observation in enumerate(observations):
        result = _drive_time_result(db, observation)
        _write_sample_report(db, case_id, index, result=result, reply="found it")
        results.append(result)
    return results, cohort


def test_a_cohort_samples_fragile_flag_answers_to_its_claims_not_to_the_reroll(
    tmp_path, monkeypatch
) -> None:
    """All three rerolled; only the one whose claim the cohort held is a fragile PASS."""
    assert active_run() is None  # off-report: `record_case*` write nothing, order-independently
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    results, cohort = _rerolled_cohort(tmp_path, _COHORT_FRAGILE_CASE)

    # At drive time a ported sample passes VACUOUSLY — it has answered no claim yet — so the
    # reroll was all the flag had to go on and every sample carried it.
    assert [result.fragile for result in results] == [True, True, True]

    _close_cohort_case(cohort, results)

    # Settled: neither the sample the claims failed nor the one the pool refused is fragile —
    # each is a failure, and its cause says which kind.
    assert [(result.passed, result.fragile) for result in results] == [
        (True, True),
        (False, False),
        (False, False),
    ]
    artifact = build_case_artifact(
        run_id="run-x",
        case_id=_COHORT_FRAGILE_CASE,
        family="chat",
        results=results,
        timings=CaseTimings(calls=0, duration_ms=0, input_tokens=0, output_tokens=0),
    )
    assert artifact.sample_fragile == [True, False, False]
    assert artifact.sample_causes == [None, FailureCause.BEHAVIORAL, FailureCause.HARNESS]


def test_a_cohort_samples_banner_states_the_verdict_its_claims_settled(
    tmp_path, monkeypatch
) -> None:
    """The block a reader opens is bannered with the case's verdict, not the drive-time one."""
    assert active_run() is None  # off-report: the file holds the sample blocks and nothing else
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    results, cohort = _rerolled_cohort(tmp_path, _COHORT_BANNER_CASE)
    _close_cohort_case(cohort, results)

    text = _sample_report_text(tmp_path, _COHORT_BANNER_CASE)
    assert [line for line in text.splitlines() if line.startswith("<details><summary>sample ")] == [
        "<details><summary>sample 1 — ✅ pass · fragile · 0s · 1 calls</summary>",
        "<details><summary>sample 2 — ❌ fail · behavioral · 0s · 1 calls</summary>",
        "<details><summary>sample 3 — ❌ fail · harness · 0s · 1 calls</summary>",
    ]


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
    # Messages side (production reconstruction): LlmMessage.to_log_dict re-dumps the PARSED
    # args via default json.dumps — spaced + ASCII-escaped — a genuinely DIFFERENT string.
    reconstructed = LlmMessage(
        role="assistant",
        tool_calls=[
            LlmToolCall(id="c1", function=LlmToolCallFunction(name="browse", arguments=args))
        ],
    ).to_log_dict()
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


def _micro_context_ledger(db: Database) -> None:
    """One chat turn's promptlog exactly as production writes it (#1773): a browse-extract
    sub-model runs INSIDE the browse call, with its own ``agent_name``, interleaved with the
    two main-agent rows in ledger order.

    #2538 dropped this fixture's other two actors (the state classifier at the turn's head,
    the run-end skill namer at its close) along with the conversation machine and the skill
    registry they served — the browse extractor is the one micro-context still in production
    on a chat turn."""
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


def test_every_micro_context_renders_as_an_actor_in_ledger_order(tmp_path, monkeypatch) -> None:
    # #1773: `_micro_batches` admitted only browse-extract rows, so another micro-context's
    # draw was a main-agent row to the turn walk, and its scoped slice rendered as a PHANTOM
    # `👤 user` step ahead of the real one.  A micro-context renders as a named actor at the
    # anchor its placement declares: the browse extraction after the call that spawned it
    # (unchanged FIFO pairing).  Whole-render literal.
    #
    # #1997: the sample carries only its OWN sequence.  Each actor's SYSTEM PROMPT is deposited
    # for the case document, which states each distinct one once instead of restating it under
    # every sample — so the second half of this test is that the extraction still captures it,
    # at the one moment the sample's database is live to be read.
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path))
    monkeypatch.delenv("EVAL_BASELINE", raising=False)
    db = _make_db(tmp_path)
    _micro_context_ledger(db)
    result = SampleResult.graded([Check("browsed", ok=True, anchor="browse(", kind="spine")])
    _case_prompts.pop("micro-actors", None)
    _write_sample_report(db, "micro-actors", 0, result=result, reply=_REPLY_TEXT)

    assert [
        (sample, prompt.context, prompt.text) for sample, prompt in _case_prompts["micro-actors"]
    ] == [
        ("sample 1", "", "You are Penny."),
        ("sample 1", "browse-extract", "Extract one value."),
    ], "every actor's prompt reaches the case document, none render under the sample"

    assert _sample_report_text(tmp_path, "micro-actors") == (
        "<details><summary>sample 1 — ✅ pass · 0s · 3 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "deepest lake?" | ✅ |\n'
        "|---|---|---|\n"
        "| expected | C1 [spine]⚖ browsed |  |\n"
        "| 💭 | check a source |  |\n"
        '| actual | 🔧 browse({"queries": ["lake"], "extract": "depth"}) | ✅ C1 |\n'
        "| actual | 🧩 browse-extract ← user turn: Instruction: depth · Content: 1,642 m |  |\n"
        "| 💭 | thinking (browse-extract) — 24 chars: the value is right there |  |\n"
        "| actual | 🧩 browse-extract → EXTRACTED: 1642 |  |\n"
        "| actual | 📥 You opened the page (browse result) · 1642 |  |\n"
        "| 💭 | <details><summary>thinking — 23 chars</summary>"
        "the source says 1,642 m</details> |  |\n"
        '| actual | 🤖 "Lake Baikal — 1,642 m." |  |\n'
        "\n"
        "</details>\n"
        "\n"
    )


# ── Every micro-context declares a placement, and the apply path renders its pair (#2133) ────

# How a micro-context's ledger identity is spelled on ``PennyConstants``, and the module that
# stamps one on every draw — read as a FILE, like the harness's other structural pins.
_AGENT_NAME_SUFFIX = "_AGENT_NAME"
_MICRO_CONTEXT_MODULE = Path(str(micro_context_module.__file__))


def _shipped_micro_context_agents() -> set[str]:
    """Every micro-context ledger identity ``micro_context.py`` NAMES, off the module's own AST.

    What REGISTERS as a micro-context is the ``agent_name`` its promptlog rows are written
    under, and this module is where each one is stamped — so the shipped names are read where
    they are used rather than from a list somebody maintains beside them. It reads every
    ``PennyConstants.*_AGENT_NAME`` the module mentions, which is the whole set of them: the
    keyword each customer hands ``_valid_draw``, and the browse extractor's default."""
    tree = ast.parse(_MICRO_CONTEXT_MODULE.read_text())
    return {
        getattr(PennyConstants, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == PennyConstants.__name__
        and node.attr.endswith(_AGENT_NAME_SUFFIX)
    }


# ── A microcontext sample states its system prompt too (#2062) ───────────────────────────────

_MICRO_PROMPT_CASE = "micro-prompts"
_QUIET_CUSTOMER_CASE = "micro-prompts-quiet"
_EXTRACT_PROMPT = "Extract one value."
_NOTIFY_PROMPT = "Compose one message."
# The ledger's second customer is an ARBITRARY name: with the notify composer retired
# (#2656) the fixture is about the generic deposit-and-scope mechanism, not a production
# trace, so it pins nothing about which identities exist.
_SECOND_CUSTOMER = "notify-composer"


def _two_customer_ledger(db: Database) -> None:
    """One microcontext sample's promptlog as a two-customer run leaves it (#1803): a
    browse-extraction draw, a second customer's draw under an arbitrary identity, and —
    between them — a main-agent row that is neither, so what the deposit is SCOPED to is
    assertable rather than assumed.  The two customers don't share a real turn in
    production — this fixture is about the generic deposit-and-scope mechanism, not a
    production trace."""
    _log_prompt(
        db,
        agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        messages=[
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": "Instruction: price · Content: watch the listing"},
        ],
        response=_content_response("EXTRACTED: watching"),
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
        agent_name=_SECOND_CUSTOMER,
        messages=[
            {"role": "system", "content": _NOTIFY_PROMPT},
            {"role": "user", "content": "wrote: watching the listing"},
        ],
        response=_content_response("the listing is being watched"),
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
        PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        _SECOND_CUSTOMER,
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
        ("sample 1", "browse-extract", _EXTRACT_PROMPT),
        ("sample 1", "notify-composer", _NOTIFY_PROMPT),
        ("sample 2", "browse-extract", _EXTRACT_PROMPT),
        ("sample 2", "notify-composer", _NOTIFY_PROMPT),
    ], "both customers under test deposit, in ledger order; the main agent neither was does not"

    # The case document's own section, composed exactly as case close composes it
    # (`_record_case_report`): it pops this accumulator and renders it through the same two calls.
    assert report.render_prompt_variants(
        report.prompt_variants(_case_prompts.pop(_MICRO_PROMPT_CASE), total=2)
    ) == (
        "<details><summary>System prompts — 2 contexts · 2 shared by every sample</summary>\n"
        "\n"
        "<details><summary>browse-extract — 18 chars · every sample</summary>\n"
        "\n"
        "```\n"
        "Extract one value.\n"
        "```\n"
        "\n"
        "</details>\n"
        "\n"
        "<details><summary>notify-composer — 20 chars · every sample</summary>\n"
        "\n"
        "```\n"
        "Compose one message.\n"
        "```\n"
        "\n"
        "</details>\n"
        "\n"
        "</details>"
    )

    block = _sample_report_text(tmp_path, _MICRO_PROMPT_CASE)
    assert _EXTRACT_PROMPT not in block and _NOTIFY_PROMPT not in block, (
        "a sample carries only its own sequence — the prompt is stated once, on the case (#1997)"
    )

    # A customer that never drew has no prompt to state, so the placeholder sample deposits none
    # rather than an empty entry the case-close path would then have to tell apart from a real one
    # — and it still renders its honest placeholder, pinned whole beside the deposit it withholds.
    # An arbitrary name proves it: it need not be a registered micro-context, only absent from
    # this ledger.
    quiet = _make_db(tmp_path, _QUIET_CUSTOMER_CASE)
    _two_customer_ledger(quiet)
    _write_classifier_report(
        quiet,
        _QUIET_CUSTOMER_CASE,
        0,
        result=result,
        phrasing="watch the listing",
        agent_names=("a-quiet-customer",),
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


def test_a_collectors_cycle_shapes_are_distinct_and_a_dead_cycle_is_excluded(tmp_path) -> None:
    """Four different cycles read as four different shapes, and the one shape that means
    nothing ran EXCLUDES the sample before it is pooled."""
    ran = [
        _observed_cycle(index=0, before={}, after={"price": "84 zorkmids"}),
        _observed_cycle(index=1, before={"price": "84"}, after={"price": "84"}),
        _observed_cycle(index=2, before={"price": "84"}, after={"price": "91"}, sent=["it moved"]),
        _observed_cycle(index=3, before={"p": "84"}, after={"p": "84"}, sent=["still 84"]),
    ]
    shapes = {_cycle_shape(cycle) for cycle in ran}
    assert len(shapes) == 4, f"four cycles, four shapes: {sorted(shapes)}"

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


# ── An end-to-end sample that built or ran nothing is COMPLETE, never excluded (#2375) ──
#
# A chat/collector sample is voided when the HARNESS never got to measure it; an end-to-end
# sample that built no job, or whose created job's cycles were all refused, is Penny's OWN
# failure — exactly what these cases exist to catch — so it must stay in the pool and fail its
# claims, not vanish as an instrument fault.


def test_an_end_to_end_sample_that_built_no_job_is_complete_and_fails_its_claims(
    tmp_path,
) -> None:
    """A script whose turns created no job to run cycles against is still a COMPLETE
    observation — its walk names the fact, and a job-terms claim answers FALSE against it
    rather than the sample disappearing from the pool."""
    # A migrated db, not the bare schema ``_make_db`` builds: ``outgoing_replies`` reads the
    # ``penny-messages`` log a migration seeds, the way a real sample's database always has it.
    db = migrated_db(str(tmp_path / "end-to-end-no-job.db"))
    _log_prompt(db, agent_name=PennyConstants.CHAT_AGENT_NAME, run_id="r1")

    observation = _observe_end_to_end(
        db,
        name="probe-1",
        phrasing="probe",
        arm=0,
        reply="done — i couldn't set that up",
        before=set(),
        jobs=[],
        per_job=[],
        cycles=[],
        seeded_replies=0,
        turns_driven=1,
    )

    assert observation.complete, "no job created is Penny's own failure, not a harness void"
    assert observation.exclusion is None
    assert observation.cycles == []
    assert observation.walk.endswith(", no job"), observation.walk

    world = World(name="probe", pages=(), keeps=(), excludes=())
    ok, rationale = _a_new_job_fires_every(3600)(observation, world)
    assert not ok, f"a sample with no created job must fail a job-terms claim, got {rationale!r}"


# ── A follow-up turn is gated on the registry, structurally, before it is sent (#2494) ──
#
# ``_no_job_created_yet`` is the predicate ``_drive_end_to_end_script`` checks before every
# turn after the script's first: TRUE while the round has stood up nothing yet, so the next
# turn is still owed; FALSE once a job exists, so every remaining scripted turn is skipped.
# Pinned directly against the registry, with no live model and no mock Signal server — the
# gate is a pure structural read, and the live cases prove the loop that calls it.


def test_a_follow_up_is_owed_while_the_round_has_created_no_job(tmp_path) -> None:
    """A fresh round, with nothing scheduled or notifying outside what existed ``before`` it
    started, owes its next scripted turn."""
    db = migrated_db(str(tmp_path / "end-to-end-gate-no-job.db"))
    before = set(collection_names(db))

    assert _no_job_created_yet(db, before), "an empty round must still owe its next turn"


def test_a_follow_up_is_skipped_once_the_round_has_created_a_job(tmp_path) -> None:
    """A collection the round itself created, carrying a schedule, is a job — and once one
    exists, no further scripted turn is owed."""
    db = migrated_db(str(tmp_path / "end-to-end-gate-has-job.db"))
    before = set(collection_names(db))
    db.memories.create_collection(
        "a-new-watch", "a job this round stood up", schedule="FREQ=HOURLY"
    )

    assert not _no_job_created_yet(db, before), (
        "a round that already stood up a job must owe no further turn"
    )


def test_an_end_to_end_sample_reply_holds_only_this_runs_messages(tmp_path) -> None:
    """A message the seeded world wrote into the ``penny-messages`` log BEFORE the scripted
    turns ran is not in the observation's ``reply`` — the reply-provenance and nothing-sent
    claims read what THIS run said, not the seeded history too."""
    # A migrated db, not the bare schema ``_make_db`` builds: ``outgoing_replies`` reads the
    # ``penny-messages`` log a migration seeds, the way a real sample's database always has it.
    db = migrated_db(str(tmp_path / "end-to-end-seeded-reply.db"))
    _log_prompt(db, agent_name=PennyConstants.CHAT_AGENT_NAME, run_id="r1")
    db.messages.log_message(
        direction=PennyConstants.MessageDirection.OUTGOING,
        sender="penny",
        content="seeded history is not this run's reply",
    )
    db.messages.log_message(
        direction=PennyConstants.MessageDirection.OUTGOING,
        sender="penny",
        content="this run's reply",
    )
    assert outgoing_replies(db) == [
        "seeded history is not this run's reply",
        "this run's reply",
    ]

    observation = _observe_end_to_end(
        db,
        name="probe-1",
        phrasing="probe",
        arm=0,
        reply="done — job set up",
        before=set(),
        jobs=[],
        per_job=[],
        cycles=[],
        seeded_replies=1,
        turns_driven=1,
    )

    assert observation.reply == "this run's reply"


def test_an_end_to_end_sample_with_no_transitions_still_reads_a_turn_at(tmp_path) -> None:
    """There is no conversation machine on this branch (#2455 naive baseline), so no sample
    ever writes a ``state_transition`` row — reading ``turn_at`` off one answered ``None``
    on every sample, and an end claim ("stops when asked") failed every one of them with
    the same rationale regardless of what the model did.  ``turn_at`` has to come from a
    record that still exists: the last scripted message ``TEST_SENDER`` sent, on the clock
    the sample's own store already stamped it with."""
    db = migrated_db(str(tmp_path / "end-to-end-turn-at.db"))
    _log_prompt(db, agent_name=PennyConstants.CHAT_AGENT_NAME, run_id="r1")
    asked_id = db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING,
        sender=TEST_SENDER,
        content="watch this until sunday night",
    )
    assert asked_id is not None
    asked_row = db.messages.get_by_id(asked_id)
    assert asked_row is not None

    observation = _observe_end_to_end(
        db,
        name="probe-1",
        phrasing="probe",
        arm=0,
        reply="done",
        before=set(),
        jobs=[],
        per_job=[],
        cycles=[],
        seeded_replies=0,
        turns_driven=1,
    )

    assert observation.turn_at == asked_row.timestamp
    assert _last_scripted_turn_at(db) == asked_row.timestamp


def test_a_job_born_by_a_write_and_configured_later_is_read_at_its_configured_terms(
    tmp_path,
) -> None:
    """``_new_jobs`` selects the mechanisms BORN during the sample and reads each one's terms
    off its END-of-sample state (#2742): a job a round created by a plain write — the
    collection comes into being inert, no schedule, no notify — and then configured in the
    same sample is read with the schedule, notify flag and end the configure set, never with
    the inert state it was born in.  The selection stays birth: a mechanism the sample did
    not create is not on it, whatever terms it carries."""
    db = migrated_db(str(tmp_path / "end-to-end-write-then-configure.db"))
    _log_prompt(db, agent_name=PennyConstants.CHAT_AGENT_NAME, run_id="r1")
    asked_id = db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING,
        sender=TEST_SENDER,
        content="watch this page every hour and tell me when it changes",
    )
    assert asked_id is not None
    turn_at = _last_scripted_turn_at(db)
    assert turn_at is not None

    # Born BEFORE the observation's before-set: the sample did not create this one.
    db.memories.create_collection("already-running", "One the world stood up first.")
    before = memory_names_now(db)

    # The round's write: the collection comes into being inert — storage only, no job.
    db.memories.create_collection("price-watch", "Storage the write created.")

    # ... and the round's configure, a later call in the SAME sample: the terms land.
    # The end reads on the user's clock at whole minutes — the minute the turn's
    # own clock names, one hour on.
    ends_at = stored_as_utc(turn_at).replace(second=0, microsecond=0) + timedelta(hours=1)
    db.memories.update_collection_metadata(
        "price-watch",
        extraction_prompt="Check the page for the price and save it.",
        schedule="FREQ=HOURLY",
        replace_schedule=True,
        notify=True,
        expires_at=ends_at,
        run_id="r2",
    )

    observation = _observe_end_to_end(
        db,
        name="probe-1",
        phrasing="probe",
        arm=0,
        reply="watching it",
        before=before,
        jobs=["price-watch"],
        per_job=[],
        cycles=[],
        seeded_replies=0,
        turns_driven=1,
    )

    created = _new_jobs(observation)
    assert [one.name for one in created] == ["price-watch"], (
        "born-this-sample is the selection — the pre-sample mechanism is not on it"
    )
    world = World(name="probe", pages=(), keeps=(), excludes=())
    ok, rationale = _a_new_job_fires_every(3600)(observation, world)
    assert ok, f"the configured schedule is what the claim reads, got {rationale!r}"
    ok, rationale = _a_new_job_notifies()(observation, world)
    assert ok, f"the configured notify flag is what the claim reads, got {rationale!r}"
    an_hour_on = AskedEnd(
        "until an hour after the turn",
        lambda turn: (
            turn.replace(second=0, microsecond=0) + timedelta(hours=1),
            turn.replace(second=0, microsecond=0) + timedelta(hours=2),
        ),
    )
    ok, rationale = _a_new_job_ends_when_asked(an_hour_on)(observation, world)
    assert ok, f"the configured end is what the claim reads, got {rationale!r}"


def test_a_seeded_known_routines_entry_is_not_counted_as_written_this_run(tmp_path) -> None:
    """``seed_known_routine`` stands up "a routine she already knows" (#2455, code-owner
    ruling) — a job already running, with an earlier conversation in her logs that set it
    up.  That earlier conversation is HISTORY, not this sample's own work: the entry it
    wrote must not read as something a live turn just stored, or the new-page case's
    negative claim ("the taught page's price appears in nothing stored") fails on every
    sample regardless of what the model does, because the seeded price always "landed"."""
    db = migrated_db(str(tmp_path / "seed-known-routine.db"))
    seed_known_routine(
        db,
        collection="aurora-deck-2-price",
        description="the Aurora Deck 2 handheld console price",
        url="https://example.com/listing",
        extract="the current price",
        entry_key="price",
        entry_value="Price: $499",
        schedule="FREQ=HOURLY",
        ask="can you keep an eye on the Aurora Deck 2 price for me and let me know if it changes?",
        reply="done — I'll check the Aurora Deck 2 price and let you know if it moves.",
    )

    assert _stored_entries(db) == [], (
        "the seeded entry is the WORLD, not this run's write — it must not appear "
        "among what this sample stored"
    )
    held = _held_entries(db)
    assert [(entry.collection, entry.key, entry.content) for entry in held] == [
        ("aurora-deck-2-price", "price", "Price: $499")
    ], "the seeded entry must still be HELD, since the store holds it either way"

    # Every logged prompt row belongs to the seeded conversation, so a live-prompt reader
    # excludes all of it — the prior fix's bug: a run id that didn't start with the
    # harness's own ``seeded-`` prefix left these rows reading as LIVE chat turns.
    rows = db.messages.recent_prompts(limit=20)
    assert rows, "the seeded conversation should have logged its own turns"
    assert all(is_seeded_run(row.run_id) for row in rows)

    # The ask says "let me know if it changes", so the seeded row's own ``notify`` must
    # agree (#2508) — a seed that left it at the ``create_collection`` default (False) hands
    # the model a history that contradicts its store, and she rationally "fixes" it by
    # reconfiguring a job the ask never mentioned.
    assert require_memory(db, "aurora-deck-2-price").row.notify, (
        "the seeded ask asks to be told, so the seeded row must be notify=True"
    )
    configure_call = next(
        call
        for row in rows
        for call in _response_tool_calls(row)
        if tool_call_name(call) == "collection_configure"
    )
    assert json.loads(configure_call["function"]["arguments"])["notify"] is True, (
        "the logged collection_configure call must show the same notify the row carries, "
        "so the logged call and the row it produced agree"
    )


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


def test_what_the_store_held_survives_only_when_every_entry_is_still_there_unchanged() -> None:
    """The survival claim reads the store at both ends of the turn.  A sample that kept
    everything and wrote something new beside it holds; one that rewrote a held value in place
    and one that deleted a held key each fail, and the rationale names the entry that did not
    survive rather than restating the claim."""
    price = eval_cohort.StoredEntry(collection="watch", key="price", content="449")
    stock = eval_cohort.StoredEntry(collection="watch", key="stock", content="in stock")
    added = eval_cohort.StoredEntry(collection="notes", key="aside", content="a new note")
    rewritten = eval_cohort.StoredEntry(collection="watch", key="price", content="499")
    ends = {"kept": [price, stock, added], "rewrote": [rewritten, stock], "deleted": [stock]}
    samples = [
        SampleObservation(name=name, phrasing="p", held_before=[price, stock], held=held)
        for name, held in ends.items()
    ]
    cohort = Cohort("case", "m", samples)
    cohort.assert_what_the_store_held_survives()
    (claim,) = cohort.claims
    assert claim.label == "state: everything the store already held is still there, unchanged"
    assert claim.category == eval_cohort.SpecCategory.STORE
    assert [(o.sample, o.ok, o.rationale) for o in claim.outcomes] == [
        ("kept", True, None),
        ("rewrote", False, "lost ['watch: price']"),
        ("deleted", False, "lost ['watch: price']"),
    ]


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

# A world declaring no ground at all — no pages, no store.  It renders no ground fold, and the
# closed summary still has to state a substrate.
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


def test_the_given_expands_through_the_runs_recorded_handle_table(db) -> None:
    """Links travel as handles: the logged turns carry ``[n]``, the artefacts a provenance
    claim reads carry the expanded real URLs — so the given is expanded back through the
    run's OWN recorded table (``promptlog.link_handles``, stamped at run close), because the
    run was given every URL, as its handle (#2717).

    A handle from ANOTHER run's table must not expand here: the stamp is read per run, and a
    run with no stamp (or an empty table) leaves its handles as written."""
    _log_prompt(
        db,
        run_id="r1",
        messages=[
            {"role": "user", "content": "the headlines from [1] and [2]"},
            {"role": "assistant", "content": "", "tool_calls": []},
        ],
    )
    _log_prompt(
        db,
        run_id="r2",
        messages=[{"role": "user", "content": "the other run saw [1] here"}],
    )
    # Stamped at run close, the way production does — after the rows exist.
    db.messages.set_run_link_handles(
        "r1", {1: "https://news.example/story-1", 2: "https://news.example/story-2"}
    )

    given = given_to_the_model(db)
    # r1's turn: the handle expands to the URL its table holds.
    assert "https://news.example/story-1" in given.content
    assert "https://news.example/story-2" in given.content
    # r2's turn: no stamp on that run, so its handle stays a literal.
    assert "[1]" in given.content


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


def _walk_call(name: str, arguments: str = "{}") -> dict:
    """One tool call in the wire shape both sides of a promptlog row carry it in. The
    arguments DISTINGUISH the calls — the walk de-duplicates by rendered call, so two calls
    that render identically read as one."""
    return {"function": {"name": name, "arguments": arguments}}


def _turn_row(agent_name: str, run_id: str, turns: list[dict], calls: list[dict]) -> PromptLog:
    """One promptlog row for the transcript walk: ``turns`` is what it CARRIES (the messages
    side the walk emits), ``calls`` the calls its own RESPONSE drew (the side
    ``_cycle_descriptions`` reads a cycle's step description off)."""
    return PromptLog(
        agent_name=agent_name,
        model="a-model",
        prompt_type="chat",
        run_id=run_id,
        messages=json.dumps(turns),
        response=json.dumps({"choices": [{"message": {"tool_calls": calls}}]}),
    )


def test_an_end_to_end_walk_splits_cycles_at_each_collector_run():
    """A cycle renders where its RUN began, not where the last user turn left off (#2375).

    An end-to-end sample's cycles used to fold into the user turn that commissioned them —
    every call, write and composer draw after the chat's reply sat inside that turn's step,
    out of order, with no boundary anywhere. The walk now emits a cycle boundary at each
    collector ``run_id`` (the run/agent record, never a position), and the chat's own reply
    closes the chat's last step just BEFORE the first cycle, so the reply never reads as a
    tail of the job's cycles.

    A collector-ONLY sample (no driven conversation — the single-edge driver) is one cycle
    already: no boundary is emitted, so its no-user-turn step renders exactly as it always
    did."""
    reply = "Watching it."
    rows = [
        _turn_row(
            "chat",
            "chat-run",
            [{"role": "user", "content": "watch this page"}],
            calls=[],
        ),
        _turn_row(
            "chat",
            "chat-run",
            [
                {"role": "user", "content": "watch this page"},
                {"role": "assistant", "tool_calls": [_walk_call("browse", '{"queries":["chat"]}')]},
                {"role": "tool", "content": "You opened the page"},
            ],
            calls=[],
        ),
        _turn_row(
            "collector",
            "cycle-1",
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "tool_calls": [_walk_call("browse", '{"queries":["c1"]}')]},
            ],
            calls=[_walk_call("browse", '{"queries":["c1"]}')],
        ),
        _turn_row(
            "collector",
            "cycle-1",
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "tool_calls": [_walk_call("browse", '{"queries":["c1"]}')]},
                {"role": "tool", "content": "You opened the page · $109"},
                {
                    "role": "assistant",
                    "tool_calls": [_walk_call("collection_write", '{"entries":[]}')],
                },
            ],
            calls=[_walk_call("collection_write", '{"entries":[]}')],
        ),
        _turn_row(
            "collector",
            "cycle-2",
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "tool_calls": [_walk_call("browse", '{"queries":["c2"]}')]},
            ],
            calls=[_walk_call("browse", '{"queries":["c2"]}')],
        ),
    ]
    turns = _sample_turns(rows, reply=reply, driven=("watch this page",), delivered=(reply,))

    kinds = [_turn_kind(actor, content) for actor, content in turns]
    assert kinds == [
        report.EventKind.USER,
        report.EventKind.CALL,
        report.EventKind.RESULT,
        report.EventKind.REPLY,
        report.EventKind.CYCLE,
        report.EventKind.CALL,
        report.EventKind.RESULT,
        report.EventKind.CALL,
        report.EventKind.CYCLE,
        report.EventKind.CALL,
    ]
    assert turns[3] == (_ACTOR["penny"], reply)  # the chat's reply, before any cycle
    assert turns[4] == (_ACTOR["cycle"], "browse, collection_write")
    assert turns[8] == (_ACTOR["cycle"], "browse")

    events, _turn_to_event = _turns_to_events(turns, {}, [])
    boundaries = [
        index for index, event in enumerate(events) if event.kind is report.EventKind.CYCLE
    ]
    assert [events[index].body for index in boundaries] == ["browse, collection_write", "browse"]
    # Each cycle's events sit BETWEEN its own boundary and the next one — nothing of cycle 1
    # leaks past cycle 2's.
    assert [event.kind for event in events[boundaries[0] : boundaries[1]]] == [
        report.EventKind.CYCLE,
        report.EventKind.CALL,
        report.EventKind.RESULT,
        report.EventKind.CALL,
    ]
    assert [event.kind for event in events[boundaries[1] :]] == [
        report.EventKind.CYCLE,
        report.EventKind.CALL,
    ]

    # The single-edge shape is untouched: the same cycle rows with nothing driven emit no
    # boundaries at all.
    collector_only = _sample_turns(rows[2:], reply="", driven=())
    assert not any(actor == _ACTOR["cycle"] for actor, _content in collector_only)
    assert _cycle_descriptions(rows) == {
        "cycle-1": "browse, collection_write",
        "cycle-2": "browse",
    }


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


def test_a_case_states_the_behaviour_it_checks() -> None:
    """A case reaching a driver without its sentence is refused, loudly.

    The case id is a filename — it says which fixture ran, never what was being asked — so a
    report rendering a rate above it states a number with no question attached.

    The refusal names the case and the form, because the sentence is the one thing whoever hit
    this has to write."""
    assert stated_behaviour("c", "In the chat agent, when X, Penny does Y.") == (
        "In the chat agent, when X, Penny does Y."
    )
    # Whitespace is not a sentence.
    for empty in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="must state the behaviour it checks"):
            stated_behaviour("watch-writes-the-first-reading", empty)


# ── What a micro-context sample was GIVEN is its whole prompt (#2078) ────────
#
# Every observer under one gate, so a customer that goes back to building its haystack from
# the runner's own arguments fails here.  The document is the same for all five: what is
# asserted is that the CONTRACT reached the haystack beside it, and each customer's contract
# and user template are the SHIPPED ones, so the test can never pass against a prompt shape
# production does not use.


def _extraction_observation(db: Database) -> SampleObservation:
    # The driver's own table: the observation expands the value through the same
    # table the document was rendered with (#2717).  With an empty table the
    # value passes through unchanged, as on a page with no links.
    return _observe_extraction(
        db,
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=EXTRACT_PLAIN_ANSWER),
        name="draw-1 (the ask)",
        phrasing="the ask",
        arm=0,
        handles=HandleTable(),
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
    for the rest.

    What reached it is split by who put it there (#2203): the document the draw was handed is
    what was STATED to it, and the contract is the frame around that — so a figure the draw
    returns has to be in the document, while the contract's own words stay words it was
    given."""
    db = _make_db(tmp_path, "micro-context-given")
    _log_prompt(
        db,
        # ``_messages`` returns typed ``LlmMessage``s (#2563); ``_log_prompt`` stores
        # exactly the dict shape production logs, so lower them here.
        messages=[
            message.to_log_dict()
            for message in MicroContext._messages(EXTRACT_PLAIN_PAGE, EXTRACT_PLAIN_ASK)
        ],
        agent_name=agent_name,
    )

    observation = observe(db)

    assert observation.complete
    assert contract in observation.given
    assert EXTRACT_PLAIN_PAGE in observation.given
    assert EXTRACT_PLAIN_PAGE in observation.given.content
    assert contract not in observation.given.content


# A second test here, ``test_an_extraction_answer_carrying_the_contract_tags_invents_nothing``,
# pinned the specific artifact of browse-extract's old tagged shape (#2078): a
# ``NOT_PRESENT:``/``EXTRACTED:`` tag's own tail gluing, with no separating space, into the
# next line's capitalised headline.  #2615 retired that tagged shape — browse-extract now
# draws plain text, with nothing for a parser to glue — so the artifact cannot recur, and the
# test retired with it.  The broader invariant it stood in for (the haystack is the WHOLE
# prompt, contract included) is the one the test above still proves directly.


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
    loud probe run here, against a real migrated database, rather than only under
    ``make eval-remote`` where a raise costs a GPU run before it is seen.

    Its own database per case, because that is what a sample gets, and seeded through the
    DRIVER's own seam (``seed_world_stores``) rather than a second copy of it — the world's
    declaration is what the sample gets and what the report renders, so it is what this must
    exercise.

    The survival claim rides along for a store-backed world, in both directions: the store as
    seeded must read as everything still there, or the claim fails a correct run by
    construction, and one entry reworded must read as changed, or it passes a sample that
    rewrote what the user kept."""
    for index, case in enumerate(ANSWERING_CASES):
        if not case.world.stores and case.probe is None:
            continue
        db = migrated_db(str(tmp_path / f"answering-{index}.db"))
        seed_world_stores(db, case.world)
        if case.probe is not None:
            case.probe(db)
        if not case.world.stores:
            continue
        held = [
            StoredEntry(collection=row.name, key=key, content=content)
            for row in db.memories.list_all()
            if row.type == MemoryType.COLLECTION
            for key, content in collection_entries(db, row.name).items()
        ]
        survived = _seeded_entries_unchanged()
        untouched = SampleObservation(name="untouched", phrasing=case.ask, held=held)
        assert survived(untouched, case.world) == (True, "missing or changed: []")
        reworded = [held[0].model_copy(update={"content": f"{held[0].content}!"}), *held[1:]]
        rewrote = SampleObservation(name="rewrote", phrasing=case.ask, held=reworded)
        assert survived(rewrote, case.world) == (
            False,
            f"missing or changed: ['{held[0].collection}:{held[0].key}']",
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
