"""Unit tests for the dispatcher Collector — picks ready collections per cycle.

Construction-level + dispatch-selection tests only.  Full lifecycle
integration (scheduling, log → write → cursor advance) is exercised
through the existing test_chat_agent / test_message integration tests
plus the migrated likes/dislikes/knowledge prompts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory_store import RecallMode
from penny.database.models import Memory
from penny.llm.client import LlmClient


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


def test_collector_name_is_singular(test_config, tmp_path):
    """One agent identity for all collections — cursors stay partitioned via
    (agent_name, memory_name) on agent_cursor."""
    collector, _ = _make_collector(test_config, tmp_path)
    assert collector.name == "collector"


def test_dispatcher_returns_none_when_no_collections_have_prompts(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("plain", "no collector wired", RecallMode.OFF)
    assert collector._next_ready_collection() is None


_VALID_EXTRACTION_PROMPT = "Extract relevant items from user-messages log."


def test_dispatcher_picks_collection_with_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        RecallMode.OFF,
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
        RecallMode.OFF,
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
        "test-col", "x", RecallMode.OFF, extraction_prompt="test_extraction_prompt"
    )
    assert collector._next_ready_collection() is None


def test_dispatcher_skips_collections_within_interval(test_config, tmp_path):
    """A collection just collected stays out of the running until its
    interval has elapsed."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        RecallMode.OFF,
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
        RecallMode.OFF,
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        collector_interval_seconds=60,
    )
    db.memories.create_collection(
        "stale",
        "x",
        RecallMode.OFF,
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
        "wired", "x", RecallMode.OFF, extraction_prompt=_VALID_EXTRACTION_PROMPT
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
        name="prague-trip",
        type="collection",
        description="Prague attractions, restaurants, and bars worth visiting",
        recall=RecallMode.RELEVANT.value,
        archived=False,
        extraction_prompt=(
            "Collect Prague spots from chat and browse logs.\n"
            '1. log_read_next("user-messages")\n'
            "2. browse for new spots\n"
            '3. collection_write("prague-trip", entries=[...])\n'
            "4. done()."
        ),
    )

    composed = Collector._compose_prompt(target)

    expected = (
        "You are the collector for the `prague-trip` collection.\n"
        "Description: Prague attractions, restaurants, and bars worth visiting\n"
        "\n"
        "Collect Prague spots from chat and browse logs.\n"
        '1. log_read_next("user-messages")\n'
        "2. browse for new spots\n"
        '3. collection_write("prague-trip", entries=[...])\n'
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
    db.memories.create_log("collector-runs", "audit log", RecallMode.OFF)


def test_log_run_writes_done_summary_on_success(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}),
            ToolCallRecord(
                tool="done",
                arguments={"success": True, "summary": "wrote 2 new spots"},
            ),
        ],
    )
    collector._log_run(target, response)
    entries = db.memories.read_latest("collector-runs")
    assert len(entries) == 1
    assert "[prague-trip]" in entries[0].content
    assert "✅" in entries[0].content
    assert "wrote 2 new spots" in entries[0].content


def test_log_run_marks_failure_when_done_says_so(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(
                tool="done",
                arguments={"success": False, "summary": "no source URL found"},
            ),
        ],
    )
    collector._log_run(target, response)
    content = db.memories.read_latest("collector-runs")[0].content
    assert "❌" in content
    assert "no source URL found" in content


def test_log_run_handles_no_done_call(test_config, tmp_path):
    """If the cycle hits max_steps without ever calling done(), the audit
    log still gets a row — with success=false and a sentinel summary."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="browse", arguments={"queries": ["x"]}),
        ],
    )
    collector._log_run(target, response)
    content = db.memories.read_latest("collector-runs")[0].content
    assert "❌" in content
    assert "max steps" in content.lower() or "no done" in content.lower()


# ── Promptlog run-outcome tagging ────────────────────────────────────────


def test_tag_promptlog_run_stamps_success_reason_target(test_config, tmp_path):
    """The cycle's done(success, summary) and the bound target name land on
    the matching promptlog row so the addon's prompts tab can render the
    green/red collector-result tag.  ``run_id`` and ``response`` are
    passed in directly — no instance state."""
    collector, db = _make_collector(test_config, tmp_path)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    db.messages.log_prompt(
        model="test",
        messages=[],
        response={},
        agent_name="collector",
        run_id="run-xyz",
    )
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(
                tool="done",
                arguments={"success": True, "summary": "wrote 2 new spots"},
            ),
        ],
    )

    collector._tag_promptlog_run(target, "run-xyz", response)

    runs = db.messages.get_prompt_log_runs()
    assert runs[0]["run_success"] is True
    assert runs[0]["run_reason"] == "wrote 2 new spots"
    assert runs[0]["run_target"] == "prague-trip"


def test_tag_promptlog_run_with_unknown_run_id_is_noop(test_config, tmp_path):
    """If no promptlog rows exist for the run_id (cycle raised before the
    loop logged anything), tagging silently does nothing rather than
    crashing or smearing onto an unrelated row."""
    collector, db = _make_collector(test_config, tmp_path)
    target = Memory(
        name="prague-trip",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )

    collector._tag_promptlog_run(target, "never-logged", response=None)

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
    db.memories.create_collection("archived-col", "x", RecallMode.OFF, extraction_prompt="x" * 30)
    db.memories.archive("archived-col")
    success, message = await collector.run_for("archived-col")
    assert success is False
    assert "archived" in message


@pytest.mark.asyncio
async def test_run_for_no_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("bare-col", "x", RecallMode.OFF)
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
        "short-col", "x", RecallMode.OFF, extraction_prompt="test_extraction_prompt"
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
        RecallMode.OFF,
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
            ToolCallRecord(tool="log_read_next", arguments={"memory": "user-messages"}),
            ToolCallRecord(tool="browse", arguments={"queries": ["prague spa " * 10]}),
            ToolCallRecord(tool="done", arguments={"success": True, "summary": "wrote 2 entries"}),
        ],
    )
    trace = Collector._format_tool_trace(response)
    lines = trace.splitlines()
    assert lines[0] == "1. log_read_next(memory=user-messages)"
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
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )
    target_b = Memory(
        name="prague-highlights",
        type="collection",
        description="x",
        recall=RecallMode.OFF.value,
        archived=False,
        extraction_prompt="x",
    )

    db.messages.log_prompt(
        model="test", messages=[], response={}, agent_name="collector", run_id="run-A"
    )
    db.messages.log_prompt(
        model="test", messages=[], response={}, agent_name="collector", run_id="run-B"
    )

    response_a = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="done", arguments={"success": True, "summary": "ok-A"})],
    )
    response_b = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="done", arguments={"success": True, "summary": "ok-B"})],
    )

    collector._tag_promptlog_run(target_a, "run-A", response_a)
    collector._tag_promptlog_run(target_b, "run-B", response_b)

    runs = {r["run_id"]: r for r in db.messages.get_prompt_log_runs()}
    assert runs["run-A"]["run_target"] == "notified-thoughts"
    assert runs["run-A"]["run_reason"] == "ok-A"
    assert runs["run-B"]["run_target"] == "prague-highlights"
    assert runs["run-B"]["run_reason"] == "ok-B"
