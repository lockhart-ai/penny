"""Unit tests for the dispatcher Collector — picks ready collections per cycle.

Construction-level + dispatch-selection tests only.  Full lifecycle
integration (scheduling, log → write → cursor advance) is exercised
through the existing test_chat_agent / test_message integration tests
plus the migrated likes/dislikes/knowledge prompts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from penny.agents.base import CycleResult
from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory_store import Inclusion, LogEntryInput, RecallMode
from penny.database.models import Memory
from penny.llm.client import LlmClient
from penny.tools.memory_tools import LogReadTool


def _llm_client() -> LlmClient:
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _make_collector(test_config, tmp_path) -> tuple[Collector, Database]:
    db = Database(str(tmp_path / "t.db"))
    db.create_tables()
    collector = Collector(
        model_client=_llm_client(),
        db=db,
        config=test_config,
    )
    return collector, db


def _get(db: Database, name: str) -> Memory:
    """Fetch a memory that the test just created — asserts it exists (typed)."""
    memory = db.memories.get(name)
    assert memory is not None
    return memory


def test_collector_name_is_singular(test_config, tmp_path):
    """One agent identity ("collector") for promptlog/run tagging across all
    collections.  Read cursors do NOT key on this name — they key on the bound
    collection (see test_collector_cursors_partition_per_collection)."""
    collector, _ = _make_collector(test_config, tmp_path)
    assert collector.name == "collector"


async def test_collector_cursors_partition_per_collection(test_config, tmp_path):
    """Two collections reading the same log get independent cursors.

    The dispatcher drives every collection under one ``name`` ("collector"),
    so keying the cursor on the agent name collapsed all collections reading a
    log onto one shared cursor — whichever ran first consumed the new entries
    and starved the rest.  ``get_tools`` keys on the bound collection instead.
    """
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_log("user-messages", "log", Inclusion.ALWAYS, RecallMode.RECENT)
    db.memories.append(
        "user-messages",
        [LogEntryInput(content="hello there", content_embedding=None)],
        author="user",
    )

    def _log_read_for(collection: str) -> LogReadTool:
        db.memories.create_collection(collection, "d", Inclusion.NEVER, RecallMode.RECENT)
        collector._current_target = db.memories.get(collection)
        tool = next(t for t in collector.get_tools() if isinstance(t, LogReadTool))
        collector._current_target = None
        return tool

    alpha = _log_read_for("alpha")
    alpha_result = await alpha.execute(memory="user-messages")
    assert "hello there" in alpha_result
    # Framing: the read leads with a count + source header so the model reads
    # the body as fetched data, not a fresh instruction.
    assert "1 entry from `user-messages`" in alpha_result
    alpha.commit_pending()  # advance alpha's cursor past the entry

    beta = _log_read_for("beta")
    assert "hello there" in await beta.execute(memory="user-messages"), (
        "beta starved by alpha's cursor — collections share one cursor"
    )

    # Cursors key on the collection, never on the dispatcher identity.
    assert db.cursors.get("alpha", "user-messages") is not None
    assert db.cursors.get("collector", "user-messages") is None


def test_dispatcher_returns_none_when_no_collections_have_prompts(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("plain", "no collector wired", Inclusion.NEVER, RecallMode.RECENT)
    assert collector._next_ready_collection() is None


_VALID_EXTRACTION_PROMPT = "Extract relevant items from user-messages log."


def test_dispatcher_picks_collection_with_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
    )
    target = collector._next_ready_collection()
    assert target is not None
    assert target.name == "wired"


def test_dispatcher_skips_archived(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
    )
    db.memories.archive("wired")
    assert collector._next_ready_collection() is None


def test_dispatcher_skips_collection_with_too_short_extraction_prompt(test_config, tmp_path):
    """A collection whose extraction_prompt is below the 25-char minimum is skipped.

    Prevents the LLM from receiving a nonsensical (often function-call-shaped)
    instruction body that causes tool-name hallucinations.
    """
    collector, db = _make_collector(test_config, tmp_path)
    # "test_extraction_prompt" is 22 chars — below the 25-char minimum.
    db.memories.create_collection(
        "test-col",
        "x",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="test_extraction_prompt",
    )
    assert collector._next_ready_collection() is None


def test_dispatcher_skips_collections_within_interval(test_config, tmp_path):
    """A collection just collected stays out of the running until its
    interval has elapsed."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        collector_interval_seconds=300,
    )
    db.memories.mark_collected("wired")  # last_collected_at = now
    assert collector._next_ready_collection() is None


def test_dispatcher_picks_most_overdue(test_config, tmp_path):
    """When multiple collections are ready the oldest last_collected_at wins."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "fresh",
        "x",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        collector_interval_seconds=60,
    )
    db.memories.create_collection(
        "stale",
        "x",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        collector_interval_seconds=60,
    )
    # Both collected, but `stale` was much earlier
    db.memories.mark_collected("fresh")
    # Backdate `stale`'s last_collected_at by an hour
    with db.engine.connect() as conn:
        from sqlalchemy import text

        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = 'stale'"),
            {"ts": (datetime.now(UTC) - timedelta(hours=1)).isoformat()},
        )
        conn.commit()

    target = collector._next_ready_collection()
    assert target is not None
    assert target.name == "stale"


def test_dispatcher_uses_default_interval_when_unset(test_config, tmp_path):
    """A collection with NULL collector_interval_seconds falls back to the
    PennyConstants default."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired", "x", Inclusion.NEVER, RecallMode.RECENT, extraction_prompt=_VALID_EXTRACTION_PROMPT
    )
    # Just collected → not ready until DEFAULT_INTERVAL elapses
    db.memories.mark_collected("wired")
    assert collector._next_ready_collection() is None

    # Backdate by exactly the default interval
    backdate = datetime.now(UTC) - timedelta(seconds=PennyConstants.COLLECTOR_DEFAULT_INTERVAL + 1)
    with db.engine.connect() as conn:
        from sqlalchemy import text

        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = 'wired'"),
            {"ts": backdate.isoformat()},
        )
        conn.commit()

    assert collector._next_ready_collection() is not None


@pytest.mark.asyncio
async def test_get_tools_raises_outside_cycle(test_config, tmp_path):
    """The tool surface is per-target — accessing it without an active
    cycle is a programmer error, not a silent empty list."""
    collector, _ = _make_collector(test_config, tmp_path)
    with pytest.raises(RuntimeError, match="outside an execute"):
        collector.get_tools()


# ── Composed system prompt (target identity + extraction_prompt + runtime tail) ──


def test_compose_prompt_wraps_extraction_with_target_and_runtime_rules():
    """Snapshot the full composed system prompt — exact-string assertion catches
    structural drift in the framing OR the runtime-rules tail.  The runtime
    rules are load-bearing (provenance, batched writes, gated send_message,
    structured done) — chat doesn't relay them, the collector base attaches
    them on every cycle."""
    target = Memory(
        name="board-games",
        type="collection",
        description="Strategy board games worth buying",
        recall=RecallMode.RELEVANT.value,
        archived=False,
        extraction_prompt=(
            "Collect board games from chat and browse logs.\n"
            '1. log_read("user-messages")\n'
            "2. browse for new games\n"
            '3. collection_write("board-games", entries=[...])\n'
            "4. done()."
        ),
    )

    composed = Collector._compose_prompt(target)

    expected = (
        "You are the collector for the `board-games` collection.\n"
        "Description: Strategy board games worth buying\n"
        "\n"
        "Collect board games from chat and browse logs.\n"
        '1. log_read("user-messages")\n'
        "2. browse for new games\n"
        '3. collection_write("board-games", entries=[...])\n'
        "4. done().\n"
        "\n"
        "## Runtime rules (always apply)\n"
        "\n"
        "- Single batched ``collection_write`` per cycle — not one call per entry.\n"
        "- ``send_message`` (when the prompt above asks for notify-on-new) is gated on a "
        "successful write: only call it after ``collection_write`` returns without "
        "duplicate-rejection.\n"
        "- Always end the cycle with ``done(success=<bool>, summary=<one-sentence prose>)``. "
        "``success`` is true if the cycle did what the prompt asked, false on no-op or failure. "
        "``summary`` describes what actually happened (entries written, messages sent, why no-op). "
        'If nothing matches the prompt, call ``done(success=true, summary="no new matches this '
        'cycle")`` — quiet cycles are normal.\n'
        "- For corrections: if a recent message indicates an existing entry is wrong, stale, "
        "closed, or otherwise no longer accurate, ``update_entry`` or ``collection_delete_entry`` "
        "rather than appending alongside.\n"
        "- Cite only what you actually browsed this cycle.  Never invent a URL to populate a "
        '"Source:" field — if no real source was fetched, omit the field.\n'
        "- Don't dedup manually — the store rejects duplicates on write automatically."
    )

    assert composed == expected, (
        f"Composed prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )


# ── Collector-runs audit log ─────────────────────────────────────────────


def _seed_collector_runs_log(db: Database) -> None:
    """Migration 0034 creates the log in production; tests using create_tables
    directly need to declare it themselves."""
    db.memories.create_log("collector-runs", "audit log", Inclusion.NEVER, RecallMode.RECENT)


def _target() -> Memory:
    return Memory(
        name="board-games",
        type="collection",
        description="x",
        recall=RecallMode.RECENT.value,
        archived=False,
        extraction_prompt="x",
    )


def test_cycle_result_classifies_worked_no_work_failed():
    """One determination: ``done(success=False)`` / no done() → ``failed``; a
    clean completion → ``worked`` or ``no_work`` by whether a state-changing
    tool actually fired."""
    worked = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "wrote 2"}),
        ],
    )
    assert Collector._cycle_result(worked) == (RunOutcome.WORKED, "wrote 2")

    no_work = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="read_latest", arguments={}),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "no new matches"}),
        ],
    )
    assert Collector._cycle_result(no_work) == (RunOutcome.NO_WORK, "no new matches")

    failed = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="done", arguments={"success": False, "summary": "no URL"})],
    )
    assert Collector._cycle_result(failed)[0] == RunOutcome.FAILED

    no_done = ControllerResponse(
        answer="", tool_calls=[ToolCallRecord(tool="browse", arguments={"queries": ["x"]})]
    )
    assert Collector._cycle_result(no_done)[0] == RunOutcome.FAILED


def test_log_run_writes_outcome_marker_word_and_summary(test_config, tmp_path):
    """Each ``collector-runs`` entry leads with the outcome (marker + word) so
    the state is legible to both the user and Penny reading the log."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)

    collector._log_run(_target(), RunOutcome.WORKED, "wrote 2 new games", "run-worked")
    collector._log_run(_target(), RunOutcome.NO_WORK, "nothing new", "run-noop")
    collector._log_run(_target(), RunOutcome.FAILED, "no source URL found", "run-failed")
    entries = db.memories.read_latest("collector-runs")
    joined = "\n".join(e.content for e in entries)

    assert "[board-games] ✅ worked — wrote 2 new games" in joined
    assert "💤 no_work — nothing new" in joined
    assert "❌ failed — no source URL found" in joined
    # Each run entry is keyed by its run_id so quality can log_get the trace.
    assert {"run-worked", "run-noop", "run-failed"} <= {e.key for e in entries}


# ── Promptlog run-outcome tagging ────────────────────────────────────────


def test_tag_promptlog_run_stamps_outcome_reason_target(test_config, tmp_path):
    """The cycle's outcome + summary + bound target land on the matching
    promptlog row so the addon's prompts tab can render the outcome badge."""
    collector, db = _make_collector(test_config, tmp_path)
    db.messages.log_prompt(
        model="test",
        messages=[],
        response={},
        agent_name="collector",
        run_id="run-xyz",
        run_target="board-games",
    )

    collector._tag_promptlog_run("run-xyz", RunOutcome.WORKED, "wrote 2 new games")

    runs = db.messages.get_prompt_log_runs()
    assert runs[0]["run_outcome"] == "worked"
    assert runs[0]["run_reason"] == "wrote 2 new games"
    assert runs[0]["run_target"] == "board-games"


def test_tag_promptlog_run_with_unknown_run_id_is_noop(test_config, tmp_path):
    """If no promptlog rows exist for the run_id (cycle raised before the
    loop logged anything), tagging silently does nothing rather than
    crashing or smearing onto an unrelated row."""
    collector, db = _make_collector(test_config, tmp_path)

    collector._tag_promptlog_run("never-logged", RunOutcome.FAILED, "x")

    assert db.messages.get_prompt_log_runs() == []


def test_should_stop_loop_ignores_failed_done(test_config, tmp_path):
    """Regression: a malformed ``done(reasoning="x")`` (missing required
    ``success``/``summary``) used to terminate the cycle anyway because
    ``should_stop_loop`` only checked the tool name.  Now it also requires
    the record to not be marked failed, so the loop continues until the
    model retries with valid args."""
    collector, _ = _make_collector(test_config, tmp_path)
    failed_done = ToolCallRecord(tool="done", arguments={"reasoning": "x"})
    failed_done.failed = True
    assert collector.should_stop_loop([failed_done]) is False

    valid_done = ToolCallRecord(tool="done", arguments={"success": True, "summary": "wrote 2"})
    assert collector.should_stop_loop([valid_done]) is True


def test_text_form_done_is_recovered(test_config, tmp_path):
    """Regression: gpt-oss occasionally emits ``done(...)`` as plain text
    content instead of as a structured tool call.  The agent loop now
    parses the text-form back into a synthesised ``ToolCallRecord`` so
    ``_extract_done_args`` sees the model's intent rather than reporting
    a spurious ``"max steps exceeded"``."""
    from penny.agents.base import _parse_text_form_done

    raw_args = _parse_text_form_done('{"reasoning":"x","success":true,"summary":"wrote 2 entries"}')
    assert raw_args == {"reasoning": "x", "success": True, "summary": "wrote 2 entries"}

    wrapped_args = _parse_text_form_done('done({"success": false, "summary": "no-op"})')
    assert wrapped_args == {"success": False, "summary": "no-op"}

    # Genuinely text content (not a done call) returns None.
    assert _parse_text_form_done("Hi there!") is None
    assert _parse_text_form_done("") is None
    # JSON without success/summary isn't a done call.
    assert _parse_text_form_done('{"some": "other"}') is None


# ── run_for: on-demand cycle trigger ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_for_collection_not_found(test_config, tmp_path):
    collector, _ = _make_collector(test_config, tmp_path)
    success, message = await collector.run_for("does-not-exist")
    assert success is False
    assert "does-not-exist" in message
    assert "not found" in message


@pytest.mark.asyncio
async def test_run_for_archived_collection(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "archived-col", "x", Inclusion.NEVER, RecallMode.RECENT, extraction_prompt="x" * 30
    )
    db.memories.archive("archived-col")
    success, message = await collector.run_for("archived-col")
    assert success is False
    assert "archived" in message


@pytest.mark.asyncio
async def test_run_for_no_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("bare-col", "x", Inclusion.NEVER, RecallMode.RECENT)
    success, message = await collector.run_for("bare-col")
    assert success is False
    assert "extraction_prompt" in message
    assert "collection_update" in message


@pytest.mark.asyncio
async def test_run_for_rejects_too_short_extraction_prompt(test_config, tmp_path):
    """run_for returns an error for a sub-minimum extraction_prompt instead of
    running the cycle, preventing the same hallucination path as the dispatcher."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "short-col",
        "x",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="test_extraction_prompt",
    )
    success, message = await collector.run_for("short-col")
    assert success is False
    assert "too short" in message


@pytest.mark.asyncio
async def test_run_for_runs_cycle_and_returns_done_summary(test_config, tmp_path):
    from penny.agents.base import CycleResult

    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    db.memories.create_collection(
        "test-col",
        "test",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="Extract things from user-messages.",
    )

    async def mock_run_cycle(run_id: str) -> CycleResult:
        return CycleResult(
            success=True,
            response=ControllerResponse(
                answer="",
                tool_calls=[
                    ToolCallRecord(
                        tool="done",
                        arguments={"success": True, "summary": "wrote 2 entries"},
                    )
                ],
            ),
        )

    collector._run_cycle = mock_run_cycle  # ty: ignore[invalid-assignment]

    success, message = await collector.run_for("test-col")
    assert success is True
    assert "Collector cycle complete" in message
    assert "wrote 2 entries" in message
    assert "1. done(" in message


def test_format_tool_trace_numbers_calls_and_truncates_args():
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="log_read", arguments={"memory": "user-messages"}),
            ToolCallRecord(tool="browse", arguments={"queries": ["board game " * 10]}),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "wrote 2 entries"}),
        ],
    )
    trace = Collector._format_tool_trace(response)
    lines = trace.splitlines()
    assert lines[0] == "1. log_read(memory=user-messages)"
    assert lines[1].startswith("2. browse(queries=")
    assert "..." in lines[1]  # long query was truncated
    assert (
        len(lines[1]) <= len("2. browse(queries=") + 50 + 2
    )  # name + max rendered arg + closing paren
    assert lines[2].startswith("3. done(")


def test_format_tool_trace_empty_when_no_calls():
    assert Collector._format_tool_trace(None) == ""
    assert Collector._format_tool_trace(ControllerResponse(answer="", tool_calls=[])) == ""


def test_tag_promptlog_run_isolates_neighbouring_cycles(test_config, tmp_path):
    """Regression: ``run_id`` is now owned per-cycle by ``execute`` instead
    of being smuggled through ``self._last_run_id``.  Cycle B can't smear
    onto cycle A's promptlog row even if A's loop crashed and B's
    cleanup runs later."""
    collector, db = _make_collector(test_config, tmp_path)
    target_a = Memory(
        name="notified-thoughts",
        type="collection",
        description="x",
        recall=RecallMode.RECENT.value,
        archived=False,
        extraction_prompt="x",
    )
    target_b = Memory(
        name="card-games",
        type="collection",
        description="x",
        recall=RecallMode.RECENT.value,
        archived=False,
        extraction_prompt="x",
    )

    db.messages.log_prompt(
        model="test",
        messages=[],
        response={},
        agent_name="collector",
        run_id="run-A",
        run_target=target_a.name,
    )
    db.messages.log_prompt(
        model="test",
        messages=[],
        response={},
        agent_name="collector",
        run_id="run-B",
        run_target=target_b.name,
    )

    collector._tag_promptlog_run("run-A", RunOutcome.NO_WORK, "ok-A")
    collector._tag_promptlog_run("run-B", RunOutcome.NO_WORK, "ok-B")

    runs = {r["run_id"]: r for r in db.messages.get_prompt_log_runs()}
    assert runs["run-A"]["run_target"] == "notified-thoughts"
    assert runs["run-A"]["run_reason"] == "ok-A"
    assert runs["run-B"]["run_target"] == "card-games"
    assert runs["run-B"]["run_reason"] == "ok-B"


@pytest.mark.asyncio
async def test_cycle_runs_under_lock(test_config, tmp_path):
    """Every extraction cycle holds the cycle lock, so an on-demand trigger
    and the background cadence can never run two cycles at once and clobber
    the shared ``_current_target``."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    db.memories.create_collection(
        "games",
        "x",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
    )

    observed: dict = {}

    async def fake_run_cycle(run_id: str) -> CycleResult:
        observed["locked"] = collector._cycle_lock.locked()
        observed["target"] = collector._current_target.name
        return CycleResult(success=True, response=ControllerResponse(answer="done"))

    collector._run_cycle = fake_run_cycle  # ty: ignore[invalid-assignment]
    success, _ = await collector.run_for("games")

    assert success is True
    assert observed["locked"] is True
    assert observed["target"] == "games"
    # Lock is released once the cycle finishes.
    assert collector._cycle_lock.locked() is False


# ── Auto-throttle ─────────────────────────────────────────────────────────────


def _idle_response() -> ControllerResponse:
    """A cycle that only read and exited — no work."""
    return ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="read_latest", arguments={}, failed=False),
            ToolCallRecord(tool="done", arguments={}, failed=False),
        ],
    )


def _work_response() -> ControllerResponse:
    """A cycle that wrote an entry — produced work."""
    return ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, failed=False)],
    )


def test_produced_work_distinguishes_state_changes():
    assert Collector._produced_work(_work_response()) is True
    assert Collector._produced_work(_idle_response()) is False
    assert Collector._produced_work(None) is False
    # A failed mutation isn't work.
    failed = ControllerResponse(
        answer="", tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, failed=True)]
    )
    assert Collector._produced_work(failed) is False


def test_create_stamps_base_interval(test_config, tmp_path):
    """The create cadence becomes the snap-back base."""
    _, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "quiet",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="x" * 30,
        collector_interval_seconds=3600,
    )
    assert _get(db, "quiet").base_interval_seconds == 3600


def test_throttle_backs_off_after_n_idle_runs_then_snaps_back(test_config, tmp_path):
    """N consecutive idle cycles double the interval; a productive cycle snaps it
    back to the user's cadence.  Uses the default COLLECTOR_THROTTLE_AFTER (3)."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "quiet",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="x" * 30,
        collector_interval_seconds=3600,
    )

    # Idle cycles accumulate; the 3rd doubles the interval and resets the counter.
    for interval, idle in [(3600, 1), (3600, 2), (7200, 0)]:
        collector._apply_throttle(_get(db, "quiet"), RunOutcome.NO_WORK)
        m = _get(db, "quiet")
        assert (m.collector_interval_seconds, m.consecutive_idle_runs) == (interval, idle)

    # Three more idle cycles double again: 2h → 4h.
    for _ in range(3):
        collector._apply_throttle(_get(db, "quiet"), RunOutcome.NO_WORK)
    assert _get(db, "quiet").collector_interval_seconds == 14400

    # A productive cycle snaps back to the base cadence and clears the counter.
    collector._apply_throttle(_get(db, "quiet"), RunOutcome.WORKED)
    m = _get(db, "quiet")
    assert (m.collector_interval_seconds, m.consecutive_idle_runs) == (3600, 0)


def test_throttle_caps_at_max_interval(test_config, tmp_path):
    """Backoff never doubles past COLLECTOR_MAX_INTERVAL (default 604800 = weekly)."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "quiet",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="x" * 30,
        collector_interval_seconds=3600,
    )
    # Far more idle cycles than needed to blow past the ceiling unclamped.
    for _ in range(40):
        collector._apply_throttle(_get(db, "quiet"), RunOutcome.NO_WORK)
    assert _get(db, "quiet").collector_interval_seconds == 604800


def test_editing_interval_resets_base_and_idle(test_config, tmp_path):
    """Editing the interval re-declares the intended cadence and clears throttle."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "quiet",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt="x" * 30,
        collector_interval_seconds=3600,
    )
    # Throttle it up first.
    for _ in range(3):
        collector._apply_throttle(_get(db, "quiet"), RunOutcome.NO_WORK)
    assert _get(db, "quiet").collector_interval_seconds == 7200

    db.memories.update_collection_metadata("quiet", collector_interval_seconds=1800)
    m = _get(db, "quiet")
    assert m.collector_interval_seconds == 1800
    assert m.base_interval_seconds == 1800
    assert m.consecutive_idle_runs == 0
