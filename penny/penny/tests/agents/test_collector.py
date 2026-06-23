"""Unit tests for the dispatcher Collector — picks ready collections per cycle.

Construction-level + dispatch-selection tests only.  Full lifecycle
integration (scheduling, log → write → cursor advance) is exercised
through the existing test_chat_agent / test_message integration tests
plus the migrated likes/dislikes/knowledge prompts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from penny.agents.base import CycleResult
from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.constants import RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput, Inclusion, LogEntryInput, RecallMode
from penny.database.models import MemoryRow
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


def _get(db: Database, name: str) -> MemoryRow:
    """Fetch a memory that the test just created — asserts it exists (typed)."""
    memory = db.memories.get(name)
    assert memory is not None
    return memory


def _memory(db: Database, name: str):
    """Resolve a memory object that the test just created — asserts it exists."""
    memory = db.memory(name)
    assert memory is not None
    return memory


def _backdate_collected(db: Database, name: str, *, minutes: int) -> None:
    """Push a collection's last_collected_at into the past so its interval floor
    is clear and only the cursor gate decides readiness."""
    with db.engine.connect() as conn:
        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = :name"),
            {"ts": (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat(), "name": name},
        )
        conn.commit()


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
    db.memories.create_log("chatter", "log", Inclusion.ALWAYS, RecallMode.RECENT)
    chatter = db.memory("chatter")
    assert chatter is not None
    chatter.append(
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
    alpha_result = await alpha.execute(memory="chatter")
    assert "hello there" in alpha_result.message
    # Framing: the read leads with a count + source header so the model reads
    # the body as fetched data, not a fresh instruction.
    assert "1 entry from `chatter`" in alpha_result.message
    alpha.commit_pending()  # advance alpha's cursor past the entry

    beta = _log_read_for("beta")
    assert "hello there" in (await beta.execute(memory="chatter")).message, (
        "beta starved by alpha's cursor — collections share one cursor"
    )

    # Cursors key on the collection, never on the dispatcher identity.
    assert db.cursors.get("alpha", "chatter") is not None
    assert db.cursors.get("collector", "chatter") is None


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
        collector_interval_seconds=3600,
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
        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = 'stale'"),
            {"ts": (datetime.now(UTC) - timedelta(hours=1)).isoformat()},
        )
        conn.commit()

    target = collector._next_ready_collection()
    assert target is not None
    assert target.name == "stale"


def test_dispatcher_skips_collection_without_interval(test_config, tmp_path):
    """The interval is required: a collector collection with NULL
    collector_interval_seconds is skipped entirely — never run at a default
    cadence — until a cadence is set."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired", "x", Inclusion.NEVER, RecallMode.RECENT, extraction_prompt=_VALID_EXTRACTION_PROMPT
    )
    # Never collected would be "always ready" with an interval, but a NULL
    # interval makes the dispatcher skip it.
    assert collector._next_ready_collection() is None

    # Even backdated 30 days, a NULL-interval collection never becomes ready.
    db.memories.mark_collected("wired")
    backdate = datetime.now(UTC) - timedelta(days=30)
    with db.engine.connect() as conn:
        conn.execute(
            text("UPDATE memory SET last_collected_at = :ts WHERE name = 'wired'"),
            {"ts": backdate.isoformat()},
        )
        conn.commit()
    assert collector._next_ready_collection() is None

    # Setting a cadence makes it eligible.
    db.memories.update_collection_metadata("wired", collector_interval_seconds=3600)
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
    target = MemoryRow(
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


def _target() -> MemoryRow:
    return MemoryRow(
        name="board-games",
        type="collection",
        description="x",
        recall=RecallMode.RECENT.value,
        archived=False,
        extraction_prompt="x",
    )


def test_cycle_result_classifies_worked_no_work_failed():
    """One determination, split by clean-close AND by whether real work landed:
    successful ``done()`` → ``worked``/``no_work``; no successful ``done()`` but
    durable work changed → ``incomplete`` (the work is real, it just never closed
    cleanly); no successful ``done()`` and nothing changed → ``failed`` bail."""
    worked = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "wrote 2"}),
        ],
    )
    assert Collector._cycle_result(worked) == (RunOutcome.WORKED, "wrote 2")

    no_work = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_read_latest", arguments={}),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "no new matches"}),
        ],
    )
    assert Collector._cycle_result(no_work) == (RunOutcome.NO_WORK, "no new matches")

    # Wrote durable state but never closed with a successful done() (hit max
    # steps / trailed off) → incomplete, NOT failed: the work landed.
    incomplete = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, mutated=True)],
    )
    assert Collector._cycle_result(incomplete)[0] == RunOutcome.INCOMPLETE

    # done(success=False) but real work still landed → incomplete (work is real).
    failed_done_but_wrote = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            ToolCallRecord(tool="done", arguments={"success": False, "summary": "partial"}),
        ],
    )
    assert Collector._cycle_result(failed_done_but_wrote)[0] == RunOutcome.INCOMPLETE

    # done(success=False) with nothing changed → a real failure.
    failed = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="done", arguments={"success": False, "summary": "no URL"})],
    )
    assert Collector._cycle_result(failed)[0] == RunOutcome.FAILED

    # No done() and nothing changed (only a read/browse) → a real bail.
    no_done = ControllerResponse(
        answer="", tool_calls=[ToolCallRecord(tool="browse", arguments={"queries": ["x"]})]
    )
    assert Collector._cycle_result(no_done)[0] == RunOutcome.FAILED


def test_tool_failures_counts_failed_calls():
    """The persisted failed-tool count is the number of ToolCallRecords that
    failed — the structural signal the run-health classifier reads."""
    assert Collector._tool_failures(None) == 0
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="browse", arguments={}, failed=True),
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            ToolCallRecord(tool="log_read", arguments={}, failed=True),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "ok"}),
        ],
    )
    assert Collector._tool_failures(response) == 2


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

    collector._tag_promptlog_run("run-xyz", RunOutcome.WORKED, "wrote 2 new games", 0)

    runs = db.messages.get_prompt_log_runs()
    assert runs[0]["run_outcome"] == "worked"
    assert runs[0]["run_reason"] == "wrote 2 new games"
    assert runs[0]["run_target"] == "board-games"


def test_tag_promptlog_run_with_unknown_run_id_is_noop(test_config, tmp_path):
    """If no promptlog rows exist for the run_id (cycle raised before the
    loop logged anything), tagging silently does nothing rather than
    crashing or smearing onto an unrelated row."""
    collector, db = _make_collector(test_config, tmp_path)

    collector._tag_promptlog_run("never-logged", RunOutcome.FAILED, "x", 0)

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
    target_a = MemoryRow(
        name="notified-thoughts",
        type="collection",
        description="x",
        recall=RecallMode.RECENT.value,
        archived=False,
        extraction_prompt="x",
    )
    target_b = MemoryRow(
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

    collector._tag_promptlog_run("run-A", RunOutcome.NO_WORK, "ok-A", 0)
    collector._tag_promptlog_run("run-B", RunOutcome.NO_WORK, "ok-B", 0)

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
            ToolCallRecord(tool="collection_read_latest", arguments={}, failed=False),
            ToolCallRecord(tool="done", arguments={}, failed=False),
        ],
    )


def _work_response() -> ControllerResponse:
    """A cycle that actually wrote an entry — produced work (``mutated=True``)."""
    return ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, failed=False, mutated=True)
        ],
    )


def test_produced_work_distinguishes_state_changes():
    assert Collector._produced_work(_work_response()) is True
    assert Collector._produced_work(_idle_response()) is False
    assert Collector._produced_work(None) is False
    # A failed mutation isn't work.
    failed = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, failed=True)],
    )
    assert Collector._produced_work(failed) is False
    # The bug this fix targets: a duplicate-rejected write doesn't error
    # (``failed=False``) but changed nothing (``mutated=False``), so it must read
    # as no-work — otherwise the throttle re-arms every cycle and never backs off.
    duplicate = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, failed=False)],
    )
    assert Collector._produced_work(duplicate) is False


def test_consumed_input_advances_cursor_on_work_even_without_done():
    """The read cursor advances when the cycle closed via the terminator OR did
    real work.  A write that then hit max steps (no done()) still consumed its
    input — so the cursor must move, else the next tick re-reads the same batch,
    re-attempts the already-landed write, and dedup-rejects it (a wasted cycle)."""
    read_only = ControllerResponse(
        answer="", tool_calls=[ToolCallRecord(tool="log_read", arguments={}, mutated=False)]
    )
    # Closed via the terminator → input consumed regardless of work.
    assert Collector._consumed_input(True, read_only) is True
    # No terminator, but a real write landed → consumed (advance the cursor).
    assert Collector._consumed_input(False, _work_response()) is True
    # No terminator and nothing changed → not consumed; re-read next tick.
    assert Collector._consumed_input(False, read_only) is False


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

    # An ``incomplete`` cycle is productive too (real work landed) — it snaps the
    # interval back rather than counting toward backoff.
    for _ in range(3):
        collector._apply_throttle(_get(db, "quiet"), RunOutcome.NO_WORK)
    assert _get(db, "quiet").collector_interval_seconds == 7200
    collector._apply_throttle(_get(db, "quiet"), RunOutcome.INCOMPLETE)
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


# ── Cursor gate (skip-when-no-new-input) ──────────────────────────────────────


def _make_log_driven_collection(db: Database, *, log: str, prompt_names_log: bool) -> None:
    """A log + a collection whose prompt may or may not name that log."""
    db.memories.create_log(log, "log", Inclusion.ALWAYS, RecallMode.RECENT)
    _memory(db, log).append([LogEntryInput(content="first", content_embedding=None)], author="user")
    prompt = (
        f'Extract relevant items: call log_read("{log}") then collection_write.'
        if prompt_names_log
        else "Extract relevant items from somewhere not named as a log here."
    )
    db.memories.create_collection(
        "watcher",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=prompt,
        collector_interval_seconds=60,
    )


async def test_log_driven_collection_skipped_until_its_log_advances(test_config, tmp_path):
    """A collection caught up on its only input log is skipped without entering
    the model; a new log entry makes it ready again.  This is the gate that
    replaces idle-throttling for log-driven collections."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_log_driven_collection(db, log="chatter", prompt_names_log=True)

    # No cursor yet → not gate-eligible → runs (the first cycle establishes it).
    assert collector._next_ready_collection() is not None

    # Simulate a completed read: cursor sits at the head of the log.
    head = _memory(db, "chatter").read_batch(None, 10)[-1].created_at
    db.cursors.advance_committed("watcher", "chatter", head)
    db.memories.mark_collected("watcher")
    _backdate_collected(db, "watcher", minutes=10)  # clear the interval floor

    # Caught up on its only input → the gate skips it.
    assert collector._next_ready_collection() is None

    # A new log entry past the cursor → the gate lets it run.
    _memory(db, "chatter").append(
        [LogEntryInput(content="second", content_embedding=None)], author="user"
    )
    ready = collector._next_ready_collection()
    assert ready is not None and ready.name == "watcher"


async def test_stale_cursor_is_pruned_and_never_gates(test_config, tmp_path):
    """A cursor for a log the prompt no longer names is pruned, not honoured —
    so a since-dropped read can't falsely keep a collection running (its log
    still advancing) nor falsely starve it.  With no live cursor the collection
    is interval-driven and runs."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_log_driven_collection(db, log="chatter", prompt_names_log=False)

    # Leftover cursor for "chatter", which the prompt does NOT name; the log has
    # advanced far past it.
    db.cursors.advance_committed("watcher", "chatter", datetime.now(UTC) - timedelta(days=1))
    db.memories.mark_collected("watcher")
    _backdate_collected(db, "watcher", minutes=10)

    # Not gated on the stale cursor → runs (interval-driven), and it's pruned.
    ready = collector._next_ready_collection()
    assert ready is not None and ready.name == "watcher"
    assert db.cursors.get("watcher", "chatter") is None


def test_log_driven_collection_is_exempt_from_throttle(test_config, tmp_path):
    """A collection with a live log cursor never throttles — the gate skips its
    idle ticks, so widening its interval would only stall catch-up.  It stays
    pinned at base no matter how many no-work cycles are applied."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_log("chatter", "log", Inclusion.ALWAYS, RecallMode.RECENT)
    db.memories.create_collection(
        "watcher",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt='Extract via log_read("chatter").',
        collector_interval_seconds=300,
    )
    db.cursors.advance_committed("watcher", "chatter", datetime.now(UTC))

    for _ in range(10):
        collector._apply_throttle(_get(db, "watcher"), RunOutcome.NO_WORK)

    m = _get(db, "watcher")
    assert m.collector_interval_seconds == 300
    assert m.consecutive_idle_runs == 0


def test_input_pending_tristate(test_config, tmp_path):
    """The gate signal: None (no live cursor → interval-driven), False (live
    cursor, caught up → skip), True (live cursor behind its log → run)."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_log("chatter", "log", Inclusion.ALWAYS, RecallMode.RECENT)
    db.memories.create_collection(
        "watcher",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt='Extract via log_read("chatter").',
        collector_interval_seconds=300,
    )
    # No cursor → not gate-eligible.
    assert collector._input_pending(_get(db, "watcher")) is None

    # Cursor present but the log is empty → caught up.
    db.cursors.advance_committed("watcher", "chatter", datetime.now(UTC))
    assert collector._input_pending(_get(db, "watcher")) is False

    # An entry appended past the cursor → input pending.
    _memory(db, "chatter").append(
        [LogEntryInput(content="new", content_embedding=None)], author="user"
    )
    assert collector._input_pending(_get(db, "watcher")) is True


# ── Consumer gate (pub/sub fan-in) ────────────────────────────────────────────


_NOTIFIER_PROMPT = (
    "Notifier: call read_published_latest(n=1), ground it in past messages, "
    "send_message, then done."
)


def _make_consumer(db: Database) -> None:
    """A consumer collection whose prompt drains the published stream."""
    db.memories.create_collection(
        "notifier",
        "d",
        Inclusion.NEVER,
        RecallMode.RECENT,
        extraction_prompt=_NOTIFIER_PROMPT,
        collector_interval_seconds=60,
    )


def _publish_source(db: Database, name: str, key: str, content: str) -> None:
    db.memories.create_collection(name, "d", Inclusion.NEVER, RecallMode.RECENT, published=True)
    _memory(db, name).write([EntryInput(key=key, content=content)], author="producer")


def _backdate_entry(db: Database, name: str, key: str, *, days: int) -> None:
    with db.engine.connect() as conn:
        conn.execute(
            text("UPDATE memory_entry SET created_at = :ts WHERE memory_name = :n AND key = :k"),
            {"ts": (datetime.now(UTC) - timedelta(days=days)).isoformat(), "n": name, "k": key},
        )
        conn.commit()


def test_consumer_skipped_until_a_published_source_advances(test_config, tmp_path):
    """A consumer (its prompt calls read_published_latest) is gated on the
    published stream: it runs when a source has an unseen entry, is skipped when
    caught up on its own cursor, and runs again when a new entry lands."""
    collector, db = _make_collector(test_config, tmp_path)
    _publish_source(db, "games", "g1", "game one")
    _make_consumer(db)
    # A published source has an unseen entry → the consumer is ready.
    ready = collector._next_ready_collection()
    assert ready is not None and ready.name == "notifier"

    # Simulate the consumer having drained it: its cursor sits at the source head.
    head = _memory(db, "games").read_all()[-1].created_at
    db.cursors.advance_committed("notifier", "games", head)
    db.memories.mark_collected("notifier")
    _backdate_collected(db, "notifier", minutes=10)  # clear the interval floor
    # Caught up across every published source → the gate skips it.
    assert collector._next_ready_collection() is None

    # A new published entry past the cursor → ready again.
    _memory(db, "games").write([EntryInput(key="g2", content="game two")], author="producer")
    ready2 = collector._next_ready_collection()
    assert ready2 is not None and ready2.name == "notifier"


def test_consumer_not_woken_by_unpublished_source(test_config, tmp_path):
    """An unpublished collection with fresh entries does not wake a consumer —
    only published collections feed the stream."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("silent", "d", Inclusion.NEVER, RecallMode.RECENT)
    _memory(db, "silent").write([EntryInput(key="s1", content="secret")], author="user")
    _make_consumer(db)
    # No published source has anything → the consumer is skipped, never run blind.
    assert collector._next_ready_collection() is None


def test_consumer_cold_start_ignores_old_published_backlog(test_config, tmp_path):
    """A consumer subscribing for the first time ignores a long-standing backlog
    (entries older than the cold-start window) so it doesn't flood on day one."""
    collector, db = _make_collector(test_config, tmp_path)
    _publish_source(db, "games", "old", "ancient game")
    _backdate_entry(db, "games", "old", days=8)  # beyond the 7-day cold-start window
    _make_consumer(db)
    assert collector._next_ready_collection() is None
    # A fresh entry inside the window wakes it.
    _memory(db, "games").write([EntryInput(key="new", content="fresh game")], author="producer")
    ready = collector._next_ready_collection()
    assert ready is not None and ready.name == "notifier"


def test_consumer_is_exempt_from_throttle(test_config, tmp_path):
    """A consumer is gate-controlled like a log-driven collection: no-work cycles
    never widen its interval (it only wakes when the stream has something)."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_consumer(db)
    for _ in range(10):
        collector._apply_throttle(_get(db, "notifier"), RunOutcome.NO_WORK)
    m = _get(db, "notifier")
    assert m.collector_interval_seconds == 60
    assert m.consecutive_idle_runs == 0
