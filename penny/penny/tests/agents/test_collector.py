"""Unit tests for the dispatcher Collector — picks ready collections per cycle.

Construction-level + dispatch-selection tests only.  Full lifecycle
integration (scheduling, log → write → cursor advance) is exercised
through the existing test_chat_agent / test_message integration tests
plus the migrated likes/dislikes/knowledge prompts.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import text

from penny.agents.base import CycleResult
from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse, ToolCallRecord
from penny.constants import (
    WRITE_GATE_STOP_REASONS,
    MutationAction,
    MutationActor,
    RunOutcome,
    WriteGateOutcome,
)
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import MemoryRow
from penny.llm.client import LlmClient
from penny.llm.models import LlmResponse
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tools.memory_tools import LogReadTool, build_memory_tools


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
        embedding_model_client=_llm_client(),
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
    db.memories.create_log("chatter", "log")
    chatter = db.memory("chatter")
    assert chatter is not None
    chatter.append(
        [LogEntryInput(content="hello there", content_embedding=None)],
        author="user",
    )

    def _log_read_for(collection: str) -> LogReadTool:
        db.memories.create_collection(collection, "d")
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
    db.memories.create_collection("plain", "no collector wired")
    assert collector._next_ready_collection() is None


_VALID_EXTRACTION_PROMPT = "Extract relevant items from user-messages log."


def test_inert_collection_never_dispatches_then_adopt_makes_it_run(test_config, tmp_path):
    """An INERT collection (#1629: no extraction_prompt) is never picked by the
    dispatcher even though it's a live, non-archived row — inertness, not archival, is
    what excludes it. Giving it a routine + cadence (an adopt) makes the very next tick
    pick it up."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("deals-watch", "inert storage the user set up")
    row = db.memories.get("deals-watch")
    assert row is not None and not row.archived  # a live container, just no job
    assert collector._next_ready_collection() is None  # inert never dispatches
    # Adopt a skill onto it (a routine + cadence) — now the dispatcher picks it up.
    db.memories.update_collection_metadata(
        "deals-watch",
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        collector_interval_seconds=3600,
    )
    picked = collector._next_ready_collection()
    assert picked is not None and picked.name == "deals-watch"


def test_dispatcher_picks_collection_with_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
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
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        collector_interval_seconds=60,
    )
    db.memories.create_collection(
        "stale",
        "x",
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
    db.memories.create_collection("wired", "x", extraction_prompt=_VALID_EXTRACTION_PROMPT)
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


# ── Scoped tool surface: a cadence run cannot reshape the registry (#1556) ──

_LIFECYCLE_TOOL_NAMES = frozenset(
    {
        "collection_create",
        "collection_update",
        "collection_merge",
        "collection_archive",
        "collection_unarchive",
        "log_create",
        "skill_read",
    }
)


def test_collector_surface_excludes_lifecycle_tools(test_config, tmp_path):
    """A cadence-fired collector run's surface is read / write-entry / browse /
    notify — the registry-shape tools are ABSENT, not merely discouraged, so a
    background poll cannot create, reconfigure, merge, or archive a mechanism."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("watch", "d")
    collector._current_target = db.memories.get("watch")
    try:
        names = {tool.name for tool in collector.get_tools()}
    finally:
        collector._current_target = None
    assert names.isdisjoint(_LIFECYCLE_TOOL_NAMES), (
        f"collector surface leaked lifecycle tools: {names & _LIFECYCLE_TOOL_NAMES}"
    )
    # It keeps its actual job: reads, scoped entry writes, browse, notify, done.
    assert {"collection_write", "collection_read_latest", "browse", "done"} <= names


def test_build_memory_tools_lifecycle_toggle(test_config, tmp_path):
    """The chat-style surface keeps the lifecycle tier; the collector surface
    drops it.  The distinction is a single declared flag, not a per-tool branch."""
    _, db = _make_collector(test_config, tmp_path)
    chat_names = {t.name for t in build_memory_tools(db, _llm_client(), "chat")}
    collector_names = {
        t.name for t in build_memory_tools(db, _llm_client(), "collector", include_lifecycle=False)
    }
    assert chat_names >= _LIFECYCLE_TOOL_NAMES
    assert collector_names.isdisjoint(_LIFECYCLE_TOOL_NAMES)
    # Reads + entry mutations are present in BOTH — only the lifecycle tier differs.
    assert {"collection_write", "collection_read_latest"} <= collector_names <= chat_names


# ── Once-shaped trigger: run_at delays the fire, max_runs retires it (#1556) ──

_ONE_SHOT_PROMPT = "Browse the web for a daily fact and write one entry each cycle."


def test_dispatcher_skips_collection_before_run_at(test_config, tmp_path):
    """A collection with a future ``run_at`` doesn't fire until that time — a
    delayed / one-shot start, gated before the interval floor."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "delayed",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
        run_at=datetime.now(UTC) + timedelta(hours=1),
    )
    assert collector._next_ready_collection() is None


def test_dispatcher_runs_collection_at_run_at(test_config, tmp_path):
    """Once ``run_at`` has passed the collection becomes eligible like any other."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "due",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
        run_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    target = collector._next_ready_collection()
    assert target is not None and target.name == "due"


def test_max_runs_archives_after_quota(test_config, tmp_path):
    """After ``max_runs`` completed (non-cancelled) cycles the scheduler archives
    the collection — a one-shot reminder retires itself.  The count is read from
    the ledger (completed promptlog runs), and a cancelled run never counts."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "one-shot",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
        run_at=datetime.now(UTC) - timedelta(minutes=1),
        max_runs=2,
    )

    def _record_run(run_id: str, outcome: RunOutcome) -> None:
        db.messages.log_prompt(
            model="test",
            messages=[],
            response={},
            agent_name="collector",
            run_id=run_id,
            run_target="one-shot",
        )
        collector._tag_promptlog_run(run_id, outcome, "s", 0)

    # A cancelled run does not burn the allotment.
    _record_run("r-cancelled", RunOutcome.CANCELLED)
    assert db.messages.count_completed_runs("one-shot") == 0

    # First completed run: below the quota, still active.
    _record_run("r1", RunOutcome.WORKED)
    collector._archive_if_run_limit_reached(_get(db, "one-shot"), "r1")
    assert _get(db, "one-shot").archived is False

    # Second completed run reaches the quota → archived (system-actor mutation),
    # and the row remains as a visible tombstone.
    _record_run("r2", RunOutcome.NO_WORK)
    collector._archive_if_run_limit_reached(_get(db, "one-shot"), "r2")
    archived = _get(db, "one-shot")
    assert archived.archived is True
    assert collector._next_ready_collection() is None
    # The system archive is a durable, attributable ledger event (#1560): actor is
    # the scheduler (no model in the loop), the run that triggered it is the join
    # key, and the cause (the run limit) is carried in the note — so "when was this
    # archived, and by what?" is a read.
    events = db.mutations.history("one-shot", limit=10)
    archive_events = [e for e in events if e.action == MutationAction.ARCHIVED.value]
    assert len(archive_events) == 1
    assert archive_events[0].actor == MutationActor.SYSTEM.value
    assert archive_events[0].run_id == "r2"
    assert "run limit" in (archive_events[0].detail or "")


def test_unlimited_collection_never_auto_archives(test_config, tmp_path):
    """An ordinary recurring collection (``max_runs`` NULL) is never retired by
    the run-limit path no matter how many times it has run."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "recurring",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
    )
    for run_id in ("a", "b", "c"):
        db.messages.log_prompt(
            model="test",
            messages=[],
            response={},
            agent_name="collector",
            run_id=run_id,
            run_target="recurring",
        )
        collector._tag_promptlog_run(run_id, RunOutcome.WORKED, "s", 0)
    collector._archive_if_run_limit_reached(_get(db, "recurring"), "c")
    assert _get(db, "recurring").archived is False


# ── End condition: expires_at ends the watch (#1562) ──────────────────────────


@pytest.mark.asyncio
async def test_dispatcher_skips_and_retires_expired_collection(test_config, tmp_path):
    """A past ``expires_at`` ends the watch: the collection never starts another
    cycle (``_is_ready`` skips it — a pure gate) and the next dispatcher pass
    system-archives it (the ``_retire_expired`` sweep), so an expiry that passed
    while Penny was down retires the collection rather than running it.  Proven
    through the real dispatcher."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "fortnight-watch",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    # Pure readiness gate: an expired collection is never dispatched — even before
    # the sweep archives it (the skip is the predicate's, not a side effect).
    assert collector._next_ready_collection() is None
    assert _get(db, "fortnight-watch").archived is False

    # The dispatcher pass retires it, and no cycle runs (the model is never entered).
    ran = await collector.execute()
    assert ran is False

    archived = _get(db, "fortnight-watch")
    assert archived.archived is True
    assert collector._next_ready_collection() is None
    # A durable, attributable system archive (#1560): the scheduler is the actor,
    # there is no run to attribute (Penny was down past the expiry), and the cause
    # (the expiry) is carried in the note.
    events = db.mutations.history("fortnight-watch", limit=10)
    archive_events = [e for e in events if e.action == MutationAction.ARCHIVED.value]
    assert len(archive_events) == 1
    assert archive_events[0].actor == MutationActor.SYSTEM.value
    assert archive_events[0].run_id is None
    assert "reached expiry" in (archive_events[0].detail or "")


@pytest.mark.asyncio
async def test_expiry_passing_mid_cycle_archives_post_cycle(mock_llm, test_config, tmp_path):
    """A watch whose ``expires_at`` has passed by the time a cycle finishes is
    system-archived post-cycle (beside the ``max_runs`` retire) — the mid-life end
    condition.  Driven through a real cycle (``run_for`` → ``_execute_cycle``): the
    model writes one entry, then the post-cycle check retires the collection, and
    the archive is attributed to that cycle's own run (unlike the while-down sweep,
    which has none)."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "expiring-watch",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    def handler(request: dict, count: int) -> LlmResponse:
        if count == 1:  # one real write so the cycle isn't a premature-done bail
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {"memory": "expiring-watch", "entries": [{"key": "today", "content": "a fact"}]},
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    await collector.run_for("expiring-watch")

    archived = _get(db, "expiring-watch")
    assert archived.archived is True
    events = db.mutations.history("expiring-watch", limit=10)
    archive_events = [e for e in events if e.action == MutationAction.ARCHIVED.value]
    assert len(archive_events) == 1
    assert archive_events[0].actor == MutationActor.SYSTEM.value
    assert archive_events[0].run_id is not None
    assert "reached expiry" in (archive_events[0].detail or "")


@pytest.mark.asyncio
async def test_unexpired_collection_is_not_retired(test_config, tmp_path):
    """The expiry retire fires only once the end condition has passed: a future
    ``expires_at`` dispatches normally and the sweep leaves it alone, and a NULL
    ``expires_at`` (no end condition) is never retired."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "future-watch",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db.memories.create_collection(
        "eternal-watch",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        collector_interval_seconds=3600,
    )

    now = datetime.now(UTC)
    ready = {m.name for m in db.memories.list_all() if collector._is_ready(m, now)}
    assert "future-watch" in ready and "eternal-watch" in ready

    collector._retire_expired()
    assert _get(db, "future-watch").archived is False
    assert _get(db, "eternal-watch").archived is False
    # Direct guard: the post-cycle check also declines both.
    assert collector._archive_if_expired(_get(db, "future-watch"), "r") is False
    assert collector._archive_if_expired(_get(db, "eternal-watch"), "r") is False


# ── Composed system prompt (target identity + extraction_prompt + runtime tail) ──


def test_compose_prompt_wraps_extraction_with_target_and_runtime_rules():
    """Snapshot the full composed system prompt — exact-string assertion catches
    structural drift in the framing OR the runtime-rules tail.  The runtime
    rules are load-bearing (provenance, batched writes, gated send_message,
    structured done) — chat doesn't relay them, the collector base attaches
    them on every cycle.  notify=false: the injected tail is the terminal
    ``done()`` alone, numbered continuing from the stored prompt (#1557)."""
    target = MemoryRow(
        name="board-games",
        type="collection",
        description="Strategy board games worth buying",
        archived=False,
        extraction_prompt=(
            "Collect board games from chat and browse logs.\n"
            '1. log_read("user-messages")\n'
            "2. browse for new games\n"
            '3. collection_write("board-games", entries=[...])'
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
        "4. done()\n"
        "\n"
        "## Runtime rules (always apply)\n"
        "\n"
        "- Single batched `collection_write(entries=[...])` per cycle — not one call per entry.\n"
        "- End every cycle with `done()` — it takes NO arguments.  It just marks the cycle "
        "finished; the run record is generated automatically from the tool calls you actually "
        "made, so there is nothing to summarise or report.\n"
        "- If nothing new matched, this is a QUIET cycle: do NOT force a `collection_write` "
        "just to have one — read your sources, then call `done()`.  Quiet cycles are normal "
        "and expected.\n"
        "- For corrections: if a recent message indicates an existing entry is wrong, stale, "
        "closed, or otherwise no longer accurate, `update_entry(key=<key>, content=<corrected "
        "content>)` or `collection_delete_entry(key=<key>)` rather than appending alongside.\n"
        "- Cite only what you actually browsed this cycle.  Never invent a URL to populate a "
        '"Source:" field — if no real source was fetched, omit the field.\n'
        "- Don't dedup manually — the store rejects duplicates on write automatically."
    )

    assert composed == expected, (
        f"Composed prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )


# ── Notify steps + injected terminal (emission-as-property, #1557) ────────────

# A skill-rendered extraction_prompt (every step one canonical tool call, NO
# done() — the chat ledger has no done tool, so a render cannot produce one) —
# the kitchen-sink case the continuous-script render is asserted against.
_NOTIFY_RENDERED_PROMPT = (
    "Collect indie metroidvania releases and keep me posted on the good ones.\n"
    '1. browse(queries=["new indie metroidvania releases"], extract="pull out the '
    'release name, a one-line hook, and the URL")\n'
    '2. collection_write("indie-metroidvanias", entries=[{key: <release name>, '
    "content: <name + hook + URL>}])"
)


def test_compose_prompt_appends_notify_steps_and_terminal_done_when_notify_true():
    """A notify=true collection composes ONE continuous numbered program (#1557):
    the stored steps 1..A, the notify steps A+1..A+4, then the injected terminal
    ``done()`` — no headers, no lead-in, asserted char-for-char (kitchen-sink: a
    skill-rendered extraction_prompt).  Nothing here is ever written into the
    stored prompt; it is appended at assembly time only."""
    target = MemoryRow(
        name="indie-metroidvanias",
        type="collection",
        description="Indie metroidvania releases the user tracks",
        archived=False,
        notify=True,
        extraction_prompt=_NOTIFY_RENDERED_PROMPT,
    )

    composed = Collector._compose_prompt(target)

    expected = (
        "You are the collector for the `indie-metroidvanias` collection.\n"
        "Description: Indie metroidvania releases the user tracks\n"
        "\n"
        f"{_NOTIFY_RENDERED_PROMPT}\n"
        '3. read_similar(memory="user-messages", anchor=<what you just found>, k=5) — '
        "the user's past messages closest to this find.\n"
        '4. read_similar(memory="penny-messages", anchor=<what you just found>, k=5) — '
        "your own past replies about it.\n"
        "5. Compose one short, friendly message: a quick greeting, what you just found "
        "(the key detail in plain words), the source URL if there is one, and — only if "
        "one of those past messages is genuinely related — a one-line callback to it.\n"
        "6. send_message(content=<the message>)\n"
        "7. done()\n"
        "\n"
        f"{Collector._RUNTIME_RULES}"
    )
    assert composed == expected, (
        f"Composed notify prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )


def test_compose_prompt_numbers_injected_steps_from_one_for_prose_prompt():
    """Uniform for legacy hand-authored collections (#1557), including the
    unnumbered-prose shape: with no leading step numbers in the stored prompt,
    A = 0, so the injected notify steps number 1..4 and the terminal done() is 5
    — byte-for-byte."""
    legacy_prompt = (
        "Watch the summit webcam page, read the status banner, and record the "
        "current trail status in the collection under the key `trail`."
    )
    target = MemoryRow(
        name="summit-status",
        type="collection",
        description="Summit trail status",
        archived=False,
        notify=True,
        extraction_prompt=legacy_prompt,
    )

    composed = Collector._compose_prompt(target)

    expected = (
        "You are the collector for the `summit-status` collection.\n"
        "Description: Summit trail status\n"
        "\n"
        f"{legacy_prompt}\n"
        '1. read_similar(memory="user-messages", anchor=<what you just found>, k=5) — '
        "the user's past messages closest to this find.\n"
        '2. read_similar(memory="penny-messages", anchor=<what you just found>, k=5) — '
        "your own past replies about it.\n"
        "3. Compose one short, friendly message: a quick greeting, what you just found "
        "(the key detail in plain words), the source URL if there is one, and — only if "
        "one of those past messages is genuinely related — a one-line callback to it.\n"
        "4. send_message(content=<the message>)\n"
        "5. done()\n"
        "\n"
        f"{Collector._RUNTIME_RULES}"
    )
    assert composed == expected, (
        f"Legacy notify prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )


def test_collector_notify_steps_constants_pinned():
    """The notify-step template + the injected terminal, pinned verbatim (#1557).
    The template carries no numbers (assembly numbers it, continuing from the
    stored prompt) and no done() (the terminal is assembly's).  ``read_similar``'s
    signature is (memory, anchor, k) — a drift here changes what every notify=true
    collector is told to do."""
    assert Prompt.COLLECTOR_NOTIFY_STEPS == (
        'read_similar(memory="user-messages", anchor=<what you just found>, k=5) — '
        "the user's past messages closest to this find.",
        'read_similar(memory="penny-messages", anchor=<what you just found>, k=5) — '
        "your own past replies about it.",
        "Compose one short, friendly message: a quick greeting, what you just found "
        "(the key detail in plain words), the source URL if there is one, and — only if "
        "one of those past messages is genuinely related — a one-line callback to it.",
        "send_message(content=<the message>)",
    )
    assert Prompt.COLLECTOR_DONE_STEP == "done()"


_NOTIFY_SEED_KEY = "Hollow Verge"
_NOTIFY_SEED_CONTENT = "Hollow Verge — a hand-drawn metroidvania. https://ex.example/hv"


def _seed_notify_collection(db: Database) -> None:
    """A notify=true collection holding one existing entry, plus a primary user so a
    notify-step send_message can enqueue."""
    db.users.save_info(
        sender="+15551230000",
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )
    db.memories.create_collection(
        "indie-metroidvanias",
        "Indie metroidvania releases",
        extraction_prompt=_NOTIFY_RENDERED_PROMPT,
        collector_interval_seconds=3600,
        notify=True,
    )
    db.memory("indie-metroidvanias").write(
        [EntryInput(key=_NOTIFY_SEED_KEY, content=_NOTIFY_SEED_CONTENT)], author="producer"
    )


@pytest.mark.asyncio
async def test_notify_cycle_sends_nothing_on_a_no_change_write(mock_llm, test_config, tmp_path):
    """STOP interplay (#1557): a notify=true run that re-observes the watched key with
    an UNCHANGED value STOPs at the write gate and never reaches the notify suffix — so
    a no-change cycle emits NOTHING.  Drives the real loop with a mocked model."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    collector.set_channel(cast(Any, object()))  # presence flag: enables send_message

    def handler(request: dict, count: int) -> LlmResponse:
        # The only step the model gets to make: the write STOPs the loop before send.
        return mock_llm._make_tool_call_response(
            request,
            "collection_write",
            {
                "memory": "indie-metroidvanias",
                "entries": [{"key": _NOTIFY_SEED_KEY, "content": _NOTIFY_SEED_CONTENT}],
            },
        )

    mock_llm.set_response_handler(handler)

    await collector.run_for("indie-metroidvanias")

    # The write-gate STOP closed the cycle at the chokepoint — the model was asked
    # exactly once (the write), and nothing was queued to the user.
    assert len(mock_llm.requests) == 1
    assert db.send_queue.next_pending() is None


@pytest.mark.asyncio
async def test_notify_cycle_composes_and_sends_on_a_productive_write(
    mock_llm, test_config, tmp_path
):
    """The productive path (#1557): a notify=true run that writes a NEW entry does not
    STOP, so it runs the notify suffix in the same cycle — reaching send_message and
    queuing one message to the user."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    collector.set_channel(cast(Any, object()))

    def handler(request: dict, count: int) -> LlmResponse:
        if count == 1:  # a productive write — a genuinely new find, no STOP
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "indie-metroidvanias",
                    "entries": [
                        {
                            "key": "Cinder Drift",
                            "content": "Cinder Drift — a new metroidvania. https://ex.example/cd",
                        }
                    ],
                },
            )
        if count == 2:  # the notify suffix's send step
            return mock_llm._make_tool_call_response(
                request, "send_message", {"content": "Found a new one: Cinder Drift! 🎮"}
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(handler)

    await collector.run_for("indie-metroidvanias")

    pending = db.send_queue.next_pending()
    assert pending is not None
    assert "Cinder Drift" in pending.content


@pytest.mark.asyncio
async def test_changed_cycle_auto_refreshes_baseline_then_next_cycle_is_quiet(
    mock_llm, test_config, tmp_path
):
    """The anti-spam proof (#1633): the last prose gate in the watch chain is gone.

    A notify=true watch collector observes its key.  The source value CHANGES, so the
    model writes the SAME key with a new value → the write gate auto-refreshes the
    stored baseline IN PLACE (stamping the writing run) and, because CHANGED is not a
    STOP, the notify suffix runs and emits ONCE.  The refresh is structural, at the
    write chokepoint — the model needs no ``update_entry`` step (its absence from the
    write-gate mechanism is pinned in test_memory_store / test_memory_tools; here the
    model, like the live journey, only writes/notifies).  The NEXT cycle re-observes
    the now-current value: the gate reads UNCHANGED and STOPs before the notify suffix,
    so it emits NOTHING.  Changed once → notified once → quiet.  Driven through the
    real collector loop with a mocked model."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)  # baseline: _NOTIFY_SEED_KEY = _NOTIFY_SEED_CONTENT
    collector.set_channel(cast(Any, object()))  # presence flag: enables send_message

    new_value = f"{_NOTIFY_SEED_CONTENT} — now with a playable demo!"

    # Cycle 1: the source changed.  The model writes the SAME key with a new value,
    # then (CHANGED is not a STOP) runs the notify suffix and sends once — no
    # update_entry step (the gate refreshed the baseline itself).
    def changed_handler(request: dict, count: int) -> LlmResponse:
        if count == 1:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "indie-metroidvanias",
                    "entries": [{"key": _NOTIFY_SEED_KEY, "content": new_value}],
                },
            )
        if count == 2:
            return mock_llm._make_tool_call_response(
                request, "send_message", {"content": f"Update on {_NOTIFY_SEED_KEY}!"}
            )
        return mock_llm._make_tool_call_response(request, "done", {})

    mock_llm.set_response_handler(changed_handler)
    await collector.run_for("indie-metroidvanias")

    # The gate auto-refreshed the baseline in place: one row, now the new value,
    # stamped by the writing run — via the write alone, no update_entry.
    stored = db.memory("indie-metroidvanias").get(_NOTIFY_SEED_KEY)
    assert len(stored) == 1
    assert stored[0].content == new_value
    assert stored[0].last_written_by_run_id is not None
    # CHANGED reached the notify suffix — exactly one message queued.
    assert [item.content for item in db.send_queue.pending_items()] == [
        f"Update on {_NOTIFY_SEED_KEY}!"
    ]

    # Cycle 2: the source is unchanged since the refresh.  The model re-observes the
    # same value → the gate reads UNCHANGED and STOPs at the write, before the notify
    # suffix.  The model is asked exactly once and nothing new is queued.
    requests_before_cycle_2 = len(mock_llm.requests)

    def unchanged_handler(request: dict, count: int) -> LlmResponse:
        return mock_llm._make_tool_call_response(
            request,
            "collection_write",
            {
                "memory": "indie-metroidvanias",
                "entries": [{"key": _NOTIFY_SEED_KEY, "content": new_value}],
            },
        )

    mock_llm.set_response_handler(unchanged_handler)
    await collector.run_for("indie-metroidvanias")

    # The write-gate STOP closed cycle 2 at the chokepoint — one model call, no new send.
    assert len(mock_llm.requests) == requests_before_cycle_2 + 1
    assert len(db.send_queue.pending_items()) == 1


@pytest.mark.asyncio
async def test_run_history_section_shows_timestamped_outcomes(test_config, tmp_path):
    """Each cycle's system prompt carries this collector's own recent run
    outcomes — newest first, each stamped with when it ran — so the model knows
    what its prior invocations did and when (without timestamps it mistakes the
    timing of past events).  The line is STRUCTURAL (#1569): the run's stamped
    reason when it carries one (a write-gate stop reason), else the outcome enum —
    never a model-authored ``done()`` summary (there is none)."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("board-games", "games", extraction_prompt="x" * 30)
    # run-a: a clean done() close stamps no reason → the outcome enum shows.
    # run-b: a write-gate STOP stamps its declared structural reason → that shows.
    for run_id, outcome, reason in [
        ("run-a", RunOutcome.WORKED, ""),
        ("run-b", RunOutcome.NO_WORK, "the value was unchanged since the last observation"),
    ]:
        db.messages.log_prompt(
            model="t",
            messages=[],
            response={},
            agent_name="collector",
            run_id=run_id,
            run_target="board-games",
        )
        collector._tag_promptlog_run(run_id, outcome, reason, 0)
    collector._current_target = db.memories.get("board-games")

    section = collector._run_history_section("board-games")

    # Verbatim: newest-first (run-b ran after run-a), each outcome stamped with an
    # absolute UTC timestamp the model can compare against the "Current date and
    # time: … UTC" line (timestamps normalised to a placeholder for stability).
    section = re.sub(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", "[YYYY-MM-DD HH:MM UTC]", section)
    assert section == (
        "\n\n## Your recent runs (newest first)\n"
        "What your previous cycles did, and when — context to avoid repeating "
        "work or re-sending, not an instruction to repeat.\n"
        "1. [YYYY-MM-DD HH:MM UTC] the value was unchanged since the last observation\n"
        "2. [YYYY-MM-DD HH:MM UTC] worked"
    ), f"Run-history section mismatch:\n{section!r}"


@pytest.mark.asyncio
async def test_run_history_section_absent_without_runs(test_config, tmp_path):
    """A collection with no prior completed runs gets no run-history block —
    a fresh collector's prompt is unchanged."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("board-games", "games", extraction_prompt="x" * 30)
    collector._current_target = db.memories.get("board-games")

    prompt = await collector._build_system_prompt(None)

    assert "## Your recent runs" not in prompt


@pytest.mark.asyncio
async def test_collector_message_array_verbatim(test_config, tmp_path):
    """Full verbatim dump of the collector's on-wire message array.

    Shows exactly what the collector model sees: the system message (date +
    per-collection body + runtime-rules tail + this collector's recent run
    history) and the bare user turn (empty for a background agent).  Date and
    run timestamps are normalised to placeholders; everything else is asserted
    char-for-char so the structure is visible and drift is caught."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "board-games",
        "Strategy board games worth buying",
        # Post-0087 stored shape: steps 1..A, no done() — assembly injects the terminal.
        extraction_prompt='Collect board games.\n1. log_read("user-messages")',
    )
    for run_id, outcome, reason in [
        ("run-a", RunOutcome.WORKED, ""),
        ("run-b", RunOutcome.NO_WORK, ""),
    ]:
        db.messages.log_prompt(
            model="t",
            messages=[],
            response={},
            agent_name="collector",
            run_id=run_id,
            run_target="board-games",
        )
        collector._tag_promptlog_run(run_id, outcome, reason, 0)
    collector._current_target = db.memories.get("board-games")

    system_prompt = await collector._build_system_prompt(None)
    messages = collector._build_messages("", None, system_prompt)

    # ── System message: date + body + runtime-rules tail + run history ─────
    system_text = re.sub(
        r"Current date and time: [^\n]*", "Current date and time: DATE", messages[0]["content"]
    )
    system_text = re.sub(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", "[YYYY-MM-DD HH:MM UTC]", system_text
    )
    expected_system = (
        "Current date and time: DATE\n"
        "\n"
        "You are the collector for the `board-games` collection.\n"
        "Description: Strategy board games worth buying\n"
        "\n"
        "Collect board games.\n"
        '1. log_read("user-messages")\n'
        "2. done()\n"
        "\n"
        "## Runtime rules (always apply)\n"
        "\n"
        "- Single batched `collection_write(entries=[...])` per cycle — not one call per entry.\n"
        "- End every cycle with `done()` — it takes NO arguments.  It just marks the cycle "
        "finished; the run record is generated automatically from the tool calls you actually "
        "made, so there is nothing to summarise or report.\n"
        "- If nothing new matched, this is a QUIET cycle: do NOT force a `collection_write` "
        "just to have one — read your sources, then call `done()`.  Quiet cycles are normal "
        "and expected.\n"
        "- For corrections: if a recent message indicates an existing entry is wrong, stale, "
        "closed, or otherwise no longer accurate, `update_entry(key=<key>, content=<corrected "
        "content>)` or `collection_delete_entry(key=<key>)` rather than appending alongside.\n"
        "- Cite only what you actually browsed this cycle.  Never invent a URL to populate a "
        '"Source:" field — if no real source was fetched, omit the field.\n'
        "- Don't dedup manually — the store rejects duplicates on write automatically.\n"
        "\n"
        "## Your recent runs (newest first)\n"
        "What your previous cycles did, and when — context to avoid repeating "
        "work or re-sending, not an instruction to repeat.\n"
        "1. [YYYY-MM-DD HH:MM UTC] no_work\n"
        "2. [YYYY-MM-DD HH:MM UTC] worked"
    )
    assert system_text == expected_system, (
        f"System mismatch:\n{system_text!r}\n\nvs expected:\n{expected_system!r}"
    )

    # ── User turn: bare (empty) — a collector runs with no user message ────
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == ""


def _datetime_line(messages: list[dict]) -> str:
    """The rendered 'Current date and time:' anchor line from a message array."""
    return messages[0]["content"].split("\n", 1)[0]


def test_datetime_anchor_renders_in_profile_timezone(test_config, tmp_path):
    """The 'Current date and time' anchor renders in the user's profile timezone,
    not UTC.  A Kolkata profile (IST, UTC+5:30, no DST) gets an IST-labelled clock
    — otherwise the model is handed a UTC time under a non-UTC profile and, near
    local midnight, the wrong calendar day."""
    collector, db = _make_collector(test_config, tmp_path)
    db.users.save_info(
        sender="+15550001111",
        name="Ada",
        location="Bengaluru, India",
        timezone="Asia/Kolkata",
        date_of_birth="1990-01-01",
    )

    # Bracket the render with before/after snapshots so a minute rollover between
    # them can't flake the exact-stamp assertion.
    fmt = "%A, %B %d, %Y at %I:%M %p IST"
    before = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(fmt)
    line = _datetime_line(collector._build_messages("", None, "body"))
    after = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(fmt)

    assert line.startswith("Current date and time: ")
    assert line.endswith(" IST"), line
    assert "UTC" not in line
    # The local wall-clock stamp matches now-in-Kolkata, not now-in-UTC.
    assert any(stamp in line for stamp in (before, after)), line


def test_datetime_anchor_falls_back_to_utc_without_profile(test_config, tmp_path):
    """No profile / timezone (fresh install) → the anchor stays UTC."""
    collector, _ = _make_collector(test_config, tmp_path)

    line = _datetime_line(collector._build_messages("", None, "body"))

    assert line.startswith("Current date and time: ")
    assert line.endswith(" UTC"), line


# ── Collector-runs audit log ─────────────────────────────────────────────


def _seed_collector_runs_log(db: Database) -> None:
    """Migration 0034 creates the log in production; tests using create_tables
    directly need to declare it themselves."""
    db.memories.create_log("collector-runs", "audit log")


def _target() -> MemoryRow:
    return MemoryRow(
        name="board-games",
        type="collection",
        description="x",
        archived=False,
        extraction_prompt="x",
    )


def test_cycle_result_classifies_worked_no_work_incomplete_failed():
    """Structural outcome from the tool trace ALONE (#1569) — ``done()`` is an
    argless sentinel, so there is no ``success``/``summary`` to read.  A ``done()``
    close is ``worked``/``no_work`` by whether real work landed; without a ``done()``
    the run never closed cleanly → ``incomplete`` (work landed) / ``failed`` (a
    bail).  The reason is structural: EMPTY on a clean ``done()`` close (the run
    record's header falls back to the outcome enum), the no-``done()`` reason
    otherwise."""
    # done() + work → worked, empty reason.
    worked = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            ToolCallRecord(tool="done", arguments={}),
        ],
    )
    assert Collector._cycle_result(worked) == (RunOutcome.WORKED, "")

    # done() + read only (nothing produced) → no_work, empty reason (a quiet cycle).
    no_work = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_read_latest", arguments={}),
            ToolCallRecord(tool="done", arguments={}),
        ],
    )
    assert Collector._cycle_result(no_work) == (RunOutcome.NO_WORK, "")

    # Wrote durable state but never closed with done() (trailed off) → incomplete
    # (the work is real), stamped with the structural no-done() reason.
    incomplete = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, mutated=True)],
    )
    assert Collector._cycle_result(incomplete) == (
        RunOutcome.INCOMPLETE,
        "cycle ended without a done() call",
    )

    # No done() and nothing changed (only a read/browse) → a real bail, and hitting
    # the step cap is distinguished from trailing off (the AGENT_MAX_STEPS sentinel).
    maxed = ControllerResponse(
        answer=PennyResponse.AGENT_MAX_STEPS,
        tool_calls=[ToolCallRecord(tool="browse", arguments={"queries": ["x"]})],
    )
    assert Collector._cycle_result(maxed) == (
        RunOutcome.FAILED,
        "max steps exceeded — no done() call",
    )


def test_cycle_result_write_gate_stop_closes_cleanly():
    """A write-gate STOP (#1587) closes the cycle at the chokepoint with NO done():
    a watch's unchanged re-observation carries ``stop_reason`` on the write record —
    the outcome is a clean ``no_work`` (nothing changed) stamped with the declared
    stop reason, NOT a ``failed`` bail (the mislabel that would fire if the missing
    done() fell through to the no-``done()`` path).  A STOP that also changed durable
    state stays ``worked``."""
    reason = WRITE_GATE_STOP_REASONS[WriteGateOutcome.KEY_EXISTS_UNCHANGED]

    unchanged_stop = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(
                tool="collection_write",
                arguments={},
                mutated=False,
                stop_reason=WriteGateOutcome.KEY_EXISTS_UNCHANGED,
            )
        ],
    )
    assert Collector._cycle_result(unchanged_stop) == (RunOutcome.NO_WORK, reason)

    # A STOP preceded by a real write this cycle stays worked (work landed).
    stop_after_work = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            ToolCallRecord(
                tool="collection_write",
                arguments={},
                mutated=False,
                stop_reason=WriteGateOutcome.KEY_EXISTS_UNCHANGED,
            ),
        ],
    )
    assert Collector._cycle_result(stop_after_work) == (RunOutcome.WORKED, reason)


def test_should_stop_loop_honors_write_gate_stop(test_config, tmp_path):
    """The collector loop exits on a successful done() OR a write-gate STOP record
    (#1587); a plain write record (no done, no stop) does not stop it."""
    collector, _ = _make_collector(test_config, tmp_path)
    done = ToolCallRecord(tool="done", arguments={})
    stop = ToolCallRecord(
        tool="collection_write", arguments={}, stop_reason=WriteGateOutcome.KEY_EXISTS_UNCHANGED
    )
    plain = ToolCallRecord(tool="collection_write", arguments={}, mutated=True)
    assert collector.should_stop_loop([done]) is True
    assert collector.should_stop_loop([stop]) is True
    assert collector.should_stop_loop([plain]) is False


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
            ToolCallRecord(tool="done", arguments={}),
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

    valid_done = ToolCallRecord(tool="done", arguments={})
    assert collector.should_stop_loop([valid_done]) is True


# ── run_for: on-demand cycle trigger ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_for_collection_not_found(test_config, tmp_path):
    collector, _ = _make_collector(test_config, tmp_path)
    success, message = await collector.run_for("does-not-exist")
    assert success is False
    assert "does-not-exist" in message
    assert "not found" in message
    # matches the house memory-not-found wording (str(MemoryNotFoundError))
    assert "collection_create" in message


@pytest.mark.asyncio
async def test_run_for_archived_collection(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("archived-col", "x", extraction_prompt="x" * 30)
    db.memories.archive("archived-col")
    success, message = await collector.run_for("archived-col")
    assert success is False
    assert "archived" in message
    # names the exact recovery move — unarchive this collection
    assert "collection_unarchive('archived-col')" in message


@pytest.mark.asyncio
async def test_run_for_no_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("bare-col", "x")
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
        extraction_prompt="test_extraction_prompt",
    )
    success, message = await collector.run_for("short-col")
    assert success is False
    assert "too short" in message


@pytest.mark.asyncio
async def test_run_for_runs_cycle_and_returns_structural_outcome(test_config, tmp_path):
    """``run_for``'s on-demand message is STRUCTURAL (#1569): the run's outcome (or
    its write-gate stop reason) plus the tool trace — never a model ``done()``
    summary.  A done()-only cycle (nothing produced) reads as ``no_work``."""
    from penny.agents.base import CycleResult

    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    db.memories.create_collection(
        "test-col",
        "test",
        extraction_prompt="Extract things from user-messages.",
    )

    async def mock_run_cycle(run_id: str) -> CycleResult:
        return CycleResult(
            success=True,
            response=ControllerResponse(
                answer="",
                tool_calls=[ToolCallRecord(tool="done", arguments={})],
            ),
        )

    collector._run_cycle = mock_run_cycle  # ty: ignore[invalid-assignment]

    success, message = await collector.run_for("test-col")
    assert success is True
    assert message.startswith("Collector cycle complete: no_work")
    assert "1. done()" in message


def test_format_tool_trace_numbers_calls_and_truncates_args():
    response = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="log_read", arguments={"memory": "user-messages"}),
            ToolCallRecord(tool="browse", arguments={"queries": ["board game " * 10]}),
            ToolCallRecord(tool="done", arguments={}),
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
        archived=False,
        extraction_prompt="x",
    )
    target_b = MemoryRow(
        name="card-games",
        type="collection",
        description="x",
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
    db.memories.create_log(log, "log")
    _memory(db, log).append([LogEntryInput(content="first", content_embedding=None)], author="user")
    prompt = (
        f'Extract relevant items: call log_read("{log}") then collection_write.'
        if prompt_names_log
        else "Extract relevant items from somewhere not named as a log here."
    )
    db.memories.create_collection(
        "watcher",
        "d",
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
    db.memories.create_log("chatter", "log")
    db.memories.create_collection(
        "watcher",
        "d",
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
    db.memories.create_log("chatter", "log")
    db.memories.create_collection(
        "watcher",
        "d",
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


def _make_on_advance_collection(db: Database, *, source: str) -> None:
    """A collection whose trigger is a declared on_advance ``source_log`` — its
    prompt does NOT name the source, so only the declaration keeps it a live input
    (proving the trigger, not an inferred cursor)."""
    db.memories.create_log(source, "an event stream")
    db.memories.create_collection(
        "chained",
        "digest the upstream events",
        extraction_prompt="1. digest the new events into entries.",
        collector_interval_seconds=60,
        source_log=source,
    )


async def test_on_advance_collection_fires_on_source_advance(test_config, tmp_path):
    """The on_advance trigger (#1604): a declared source LOG gates the collection —
    it runs first to establish the cursor, then skips while caught up and wakes the
    moment the source advances, all via the frontier read, no model judgment."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_on_advance_collection(db, source="events-log")

    # Cold-start: no cursor for the declared source yet → pending → runs (the first
    # cycle establishes the cursor the frontier check then reads).
    assert collector._input_pending(_get(db, "chained")) is True
    assert collector._next_ready_collection() is not None

    # Seed the source, simulate a completed read: cursor sits at the head.
    _memory(db, "events-log").append(
        [LogEntryInput(content="first", content_embedding=None)], author="user"
    )
    head = _memory(db, "events-log").read_batch(None, 10)[-1].created_at
    db.cursors.advance_committed("chained", "events-log", head)
    db.memories.mark_collected("chained")
    _backdate_collected(db, "chained", minutes=10)  # clear the interval floor

    # Caught up on its declared source → the gate skips it.
    assert collector._input_pending(_get(db, "chained")) is False
    assert collector._next_ready_collection() is None

    # The source advances → the gate wakes the collection.
    _memory(db, "events-log").append(
        [LogEntryInput(content="second", content_embedding=None)], author="user"
    )
    assert collector._input_pending(_get(db, "chained")) is True
    ready = collector._next_ready_collection()
    assert ready is not None and ready.name == "chained"


def test_on_advance_source_cursor_is_never_pruned(test_config, tmp_path):
    """The declared on_advance ``source_log`` is a live input even though the prompt
    doesn't name it — unlike a stale inferred cursor, it is NOT pruned, so the
    trigger can't be silently swept away (#1604)."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_on_advance_collection(db, source="events-log")
    db.cursors.advance_committed("chained", "events-log", datetime.now(UTC))

    live = collector._live_cursors(_get(db, "chained"))
    assert [name for name, _ in live] == ["events-log"]
    # The cursor survives the pruning pass because it is the declared source.
    assert db.cursors.get("chained", "events-log") is not None
