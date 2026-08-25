"""Unit tests for the dispatcher Collector — picks ready collections per cycle.

Construction-level + dispatch-selection tests only.  Full lifecycle
integration (scheduling, log → write → cursor advance) is exercised
through the existing test_chat_agent / test_message integration tests
plus the migrated likes/dislikes/knowledge prompts.

THE CLOSE IS A CALL AGAIN (#1916, reverting #1911's coverage exit).  A cycle ends on a
successful ``done()`` or a write-gate STOP, assembly injects the terminal step back into
every composed prompt, and every assertion that used to read completion off program
coverage is re-derived onto the ``done`` record.  The coverage-subject tests themselves —
a half-run program never closing, an interjected read or a retry not consuming a step,
an empty program never being covered — lived in ``test_notification.py`` beside the
parser they exercised and were deleted there with ``is_covered``; nothing in this module
was retired, because nothing here took coverage as its subject.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from penny.agents.base import CycleResult
from penny.agents.collector import Collector
from penny.agents.models import ControllerResponse, ModelCallError, RunAbort, ToolCallRecord
from penny.constants import (
    COLLECTOR_UNREADABLE_PROGRAM_REASON,
    CYCLE_END_RETRY_REASONS,
    HEALTHY_CYCLE_ENDS,
    WRITE_GATE_STOP_REASONS,
    CycleEnd,
    CycleTrigger,
    MutationAction,
    MutationActor,
    PennyConstants,
    RunOutcome,
    WriteGateOutcome,
)
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import MemoryRow
from penny.database.skills import SkillParameter
from penny.llm.client import LlmClient
from penny.llm.models import LlmConnectionError, LlmResponse
from penny.notification import NOTIFICATION_NOTES, NOTIFIED_CYCLE_ENDS, NotificationOutcome
from penny.program import ProgramCall
from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tests.conftest import require_memory
from penny.tests.mocks.llm_patches import MockLlmClient, deterministic_embed
from penny.tests.schema_template import schema_only_db
from penny.tests.tools.test_memory_tools import (
    _TAUGHT_LINE,
    _TAUGHT_URL,
    seed_timetable_skill,
)
from penny.tools.base import Tool
from penny.tools.memory_tools import (
    CollectionSetTool,
    DoneTool,
    LogReadTool,
    build_memory_tools,
    collector_tool_surface,
)
from penny.tools.micro_context import NOTIFY_SYSTEM_PROMPT


def _llm_client(db: Database | None = None) -> LlmClient:
    """The model client a test collector calls through.  ``db`` is what makes it LOG
    its prompts — production wires one into every client, so a test that reads a
    cycle's own ledger rows (its run outcome, its stamped reason) passes one too."""
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.0,
    )


def _make_collector(test_config, tmp_path) -> tuple[Collector, Database]:
    db = schema_only_db(str(tmp_path / "t.db"))
    collector = Collector(
        model_client=_llm_client(db),
        db=db,
        config=test_config,
        embedding_model_client=_llm_client(),
    )
    return collector, db


# The UNSCOPED collector surface — every tool the runtime rules name, so a
# ``_compose_prompt`` render under it carries all four rules.  A program-scoped cycle
# passes a narrower set and drops the rules it cannot carry out (#1911).  ``done`` rides
# every surface, scoped or not (#1916): assembly injects the terminal step into every
# composed prompt, so the close has to be callable whatever the program contains.
_FULL_SURFACE = frozenset(
    {"collection_write", "update_entry", "collection_delete_entry", "browse", "done"}
)


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


_STAMP = "[YYYY-MM-DD HH:MM UTC]"


def _normalise_stamps(text: str) -> str:
    """Collapse every rendered log timestamp to one placeholder so a whole-render
    literal is stable — the stamps are real wall-clock values written by the test
    itself.  One helper, because three renders in this file carry stamps (a run-history
    line, a holdings entry, the whole message array) and three copies of the pattern
    would drift."""
    return re.sub(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", _STAMP, text)


def _backdate_collected(db: Database, name: str, *, minutes: int) -> None:
    """Push a collection's last_collected_at into the past so its schedule has come
    round again and only the cursor gate decides readiness."""
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
        # A program that READS the log — the surface is scoped to its calls (#1911), so
        # the tool under test is on it because the routine names it.
        db.memories.create_collection(
            collection, "d", extraction_prompt='1. log_read(memory="chatter")'
        )
        collector._bind(db.memories.get(collection))
        tool = next(t for t in collector.get_tools() if isinstance(t, LogReadTool))
        collector._bind(None)
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
    what excludes it. Giving it a routine + schedule (an adopt) makes the very next tick
    pick it up."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("deals-watch", "inert storage the user set up")
    row = db.memories.get("deals-watch")
    assert row is not None and not row.archived  # a live container, just no job
    assert collector._next_ready_collection() is None  # inert never dispatches
    # Adopt a skill onto it (a routine + schedule) — now the dispatcher picks it up.
    db.memories.update_collection_metadata(
        "deals-watch",
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        schedule="FREQ=HOURLY",
    )
    picked = collector._next_ready_collection()
    assert picked is not None and picked.name == "deals-watch"


def test_dispatcher_picks_collection_with_extraction_prompt(test_config, tmp_path):
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        schedule="FREQ=HOURLY",
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


def test_dispatcher_skips_collections_before_their_next_occurrence(test_config, tmp_path):
    """A collection just collected stays out of the running until its schedule comes
    round again."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "wired",
        "has a collector",
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        schedule="FREQ=MINUTELY;INTERVAL=5",
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
        schedule="FREQ=MINUTELY",
    )
    db.memories.create_collection(
        "stale",
        "x",
        extraction_prompt=_VALID_EXTRACTION_PROMPT,
        schedule="FREQ=MINUTELY",
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


def test_dispatcher_skips_collection_without_schedule(test_config, tmp_path):
    """The schedule is required: a collector collection with a NULL ``schedule`` is
    skipped entirely — never run at some default cadence — until one is set."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("wired", "x", extraction_prompt=_VALID_EXTRACTION_PROMPT)
    # Never collected would be "always ready" with a schedule, but a NULL one
    # makes the dispatcher skip it.
    assert collector._next_ready_collection() is None

    # Even backdated 30 days, a schedule-less collection never becomes ready.
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
    db.memories.update_collection_metadata("wired", schedule="FREQ=HOURLY")
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
        "collection_set",
        "collection_merge",
        "collection_archive",
        "collection_unarchive",
        "log_create",
        "skill_read",
    }
)


def test_collector_surface_excludes_lifecycle_tools(test_config, tmp_path):
    """A cadence-fired collector run cannot reshape the registry: the lifecycle tools
    are ABSENT, not merely discouraged, so a background poll cannot create,
    reconfigure, merge, or archive a mechanism.

    Asserted against the collector's UNSCOPED surface — everything a cycle could
    possibly be given — because that is the set the scoping then narrows, and the mask
    has to hold before the narrowing as well as after."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("watch", "d")
    collector._bind(db.memories.get("watch"))
    try:
        names = set(collector_tool_surface(db, collector._model_client))
    finally:
        collector._bind(None)
    assert names.isdisjoint(_LIFECYCLE_TOOL_NAMES), (
        f"collector surface leaked lifecycle tools: {names & _LIFECYCLE_TOOL_NAMES}"
    )
    # It keeps its actual job: reads, scoped entry writes, browse, choose.  ``choose``
    # rides the collector surface so a demonstrated random-pick step renders into a
    # runnable collector program (skill capture).
    assert {"collection_write", "collection_read_latest", "browse", "choose"} <= names


def test_choose_is_capture_eligible_on_the_collector_surface(test_config, tmp_path):
    """``collector_tool_surface`` — the set the skill extractor filters captured steps
    to (#1668) — includes ``choose``, so a demonstrated ``choose`` step survives into a
    skill instead of being dropped as an un-runnable call.  It is neither a lifecycle
    tool (which the collector can't run) nor a write, just a read-shaped picker."""
    _, db = _make_collector(test_config, tmp_path)
    surface = collector_tool_surface(db, _llm_client())
    assert "choose" in surface
    assert surface.isdisjoint(_LIFECYCLE_TOOL_NAMES)


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


# ── Schedule gating: the RRULE decides when a collection is due (#1857) ──────

_ONE_SHOT_PROMPT = "Browse the web for a daily fact and write one entry each cycle."

# BYSECOND pins the occurrence to the top of the minute, so a fixed-clock assertion
# lands on the stated time rather than inheriting created_at's seconds.
_TWICE_DAILY = "FREQ=DAILY;BYHOUR=8,20;BYMINUTE=0;BYSECOND=0"

# How RFC 5545 writes a DTSTART value.
_RRULE_STAMP = "%Y%m%dT%H%M%SZ"

# The production shape of #1932: a briefing asked for at a morning hour, stated the way
# the user said it — an hour, with no zone anywhere in the rule.
_LOCAL_MORNING = "FREQ=DAILY;BYHOUR=9;BYMINUTE=30"
# A zone that is NOT UTC in both directions and observes DST, so a local hour and a UTC
# hour can never coincide by accident and the fall-back is testable.
_USER_ZONE = "America/Toronto"


def _set_profile_timezone(db: Database, timezone: str) -> None:
    """Give the deployment a user profile carrying an IANA zone — what turns a schedule's
    stated hour into an hour on the user's clock (#1932)."""
    db.users.save_info(
        sender="test-user",
        name="Test User",
        location="Somewhere",
        timezone=timezone,
        date_of_birth="1990-01-01",
    )


def _scheduled(
    name: str, schedule: str, created_at: datetime, last: datetime | None = None
) -> MemoryRow:
    """A configured collection row for a readiness assertion — the schedule, when it was
    created (the rule's anchor) and when it last ran, and nothing else."""
    return MemoryRow(
        name=name,
        type="collection",
        description="d",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule=schedule,
        created_at=created_at,
        last_collected_at=last,
    )


def test_local_morning_schedule_fires_on_the_users_clock(test_config, tmp_path):
    """The failure this fixes (#1932): a schedule stated as a local morning hour ran in
    UTC, so a UTC-4 profile's briefing fired FOUR HOURS EARLY while the reply that set it
    up promised the local time.  An RRULE says an hour and never a zone, so the hour is
    read on the user's own clock: 09:30 for them, 13:30 UTC."""
    collector, db = _make_collector(test_config, tmp_path)
    _set_profile_timezone(db, _USER_ZONE)
    memory = _scheduled(
        "morning-briefing",
        _LOCAL_MORNING,
        # 10:00 on the user's clock, so the rule's first occurrence is the NEXT morning.
        created_at=datetime(2026, 7, 20, 14, 0, tzinfo=UTC),
        last=datetime(2026, 7, 21, 13, 30, tzinfo=UTC),  # 09:30 their time, the 21st
    )
    # 09:30 UTC is 05:30 where the user is — the four-hours-early fire, now not due.
    assert collector._is_ready(memory, datetime(2026, 7, 22, 9, 30, tzinfo=UTC)) is False
    # 13:30 UTC is 09:30 where the user is — the hour they actually asked for.
    assert collector._is_ready(memory, datetime(2026, 7, 22, 13, 30, tzinfo=UTC)) is True


def test_local_schedule_keeps_its_hour_across_a_dst_boundary(test_config, tmp_path):
    """A stated hour is a WALL-CLOCK hour, so it survives the clocks changing: the same
    09:30 local is 13:30 UTC through the summer and 14:30 UTC once the zone falls back
    (2026-11-01 in this one).  A fixed offset captured at creation would fire an hour
    early for the rest of the year."""
    collector, db = _make_collector(test_config, tmp_path)
    _set_profile_timezone(db, _USER_ZONE)
    memory = _scheduled(
        "morning-briefing",
        _LOCAL_MORNING,
        created_at=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),  # 08:00 their time
        last=datetime(2026, 10, 31, 13, 30, tzinfo=UTC),  # 09:30 their time, still UTC-4
    )
    # The summer offset, the day AFTER the clocks went back: not their morning any more.
    assert collector._is_ready(memory, datetime(2026, 11, 1, 13, 30, tzinfo=UTC)) is False
    # 14:30 UTC is 09:30 on the new offset — the same hour on their clock.
    assert collector._is_ready(memory, datetime(2026, 11, 1, 14, 30, tzinfo=UTC)) is True


def test_schedule_without_a_profile_timezone_stays_on_utc(test_config, tmp_path):
    """A fresh install has no profile, so there is no clock to read the hour on: UTC —
    the same fallback the current-time anchor and the ``expires_at`` parse take, so
    behaviour on a profile-less deployment is exactly what it always was."""
    collector, _ = _make_collector(test_config, tmp_path)  # no profile saved
    memory = _scheduled(
        "morning-briefing",
        _LOCAL_MORNING,
        created_at=datetime(2026, 7, 20, 14, 0, tzinfo=UTC),
        last=datetime(2026, 7, 21, 9, 30, tzinfo=UTC),
    )
    assert collector._is_ready(memory, datetime(2026, 7, 22, 9, 29, tzinfo=UTC)) is False
    assert collector._is_ready(memory, datetime(2026, 7, 22, 9, 30, tzinfo=UTC)) is True


def test_unresolvable_profile_timezone_falls_back_to_utc_and_says_so(test_config, tmp_path, caplog):
    """A profile zone the system can't resolve puts the schedule back on UTC — which is
    the #1932 failure restored — so it degrades VISIBLY: the fallback names the value it
    couldn't read, rather than leaving a job firing hours off with nothing to read."""
    collector, db = _make_collector(test_config, tmp_path)
    _set_profile_timezone(db, "Nowhere/Fictional")
    memory = _scheduled(
        "morning-briefing",
        _LOCAL_MORNING,
        created_at=datetime(2026, 7, 20, 14, 0, tzinfo=UTC),
        last=datetime(2026, 7, 21, 9, 30, tzinfo=UTC),
    )
    with caplog.at_level(logging.WARNING, logger="penny.datetime_utils"):
        assert collector._is_ready(memory, datetime(2026, 7, 22, 9, 30, tzinfo=UTC)) is True
    assert "Nowhere/Fictional" in caplog.text


def test_schedule_occurrences_do_not_inherit_the_anchors_seconds(test_config, tmp_path):
    """An occurrence takes its seconds from the rule's anchor, so a collection created at
    :51 past was a job that fired at :51 past for ever (#1932).  The anchor's seconds are
    zeroed, so an hourly job created at 10:15:51 comes round at 11:15:00 flat."""
    collector, _ = _make_collector(test_config, tmp_path)
    created = datetime(2026, 7, 20, 10, 15, 51, 123456, tzinfo=UTC)
    memory = _scheduled("hourly", "FREQ=HOURLY", created_at=created, last=created)
    assert collector._is_ready(memory, datetime(2026, 7, 20, 11, 15, tzinfo=UTC)) is True


def test_floating_dtstart_is_read_on_the_users_clock(test_config, tmp_path):
    """A ``DTSTART`` with no ``Z`` states no zone, which RFC 5545 calls local time — so
    it is read on the user's clock like every other zone-less anchor.  It also has to
    come back timezone-AWARE: a naive occurrence compared against ``now`` raised, and the
    readiness scan is shared by every collection, so one such row stopped them all."""
    collector, db = _make_collector(test_config, tmp_path)
    _set_profile_timezone(db, _USER_ZONE)
    memory = _scheduled(
        "floating",
        "DTSTART:20260720T090000\nFREQ=DAILY;COUNT=1",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    # 09:00 on their clock is 13:00 UTC, so noon UTC is still an hour early.
    assert collector._is_ready(memory, datetime(2026, 7, 20, 12, 0, tzinfo=UTC)) is False
    assert collector._is_ready(memory, datetime(2026, 7, 20, 13, 0, tzinfo=UTC)) is True


def test_explicit_utc_dtstart_is_an_exact_instant(test_config, tmp_path):
    """A ``DTSTART:…Z`` states its zone, so the user's profile doesn't move it — an exact
    instant stays an exact instant, which is what a one-shot at a named moment relies on."""
    collector, db = _make_collector(test_config, tmp_path)
    _set_profile_timezone(db, _USER_ZONE)
    memory = _scheduled(
        "one-shot",
        "DTSTART:20260720T090000Z\nFREQ=DAILY;COUNT=1",
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    assert collector._is_ready(memory, datetime(2026, 7, 20, 8, 59, tzinfo=UTC)) is False
    assert collector._is_ready(memory, datetime(2026, 7, 20, 9, 0, tzinfo=UTC)) is True


def test_dispatcher_skips_collection_before_its_dtstart(test_config, tmp_path):
    """A schedule whose DTSTART is in the future doesn't fire until that time — the
    delayed / one-shot start, now a line of the rule rather than its own column."""
    collector, db = _make_collector(test_config, tmp_path)
    starts = datetime.now(UTC) + timedelta(hours=1)
    db.memories.create_collection(
        "delayed",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule=f"DTSTART:{starts.strftime(_RRULE_STAMP)}\nFREQ=DAILY;COUNT=1",
    )
    assert collector._next_ready_collection() is None


def test_dispatcher_runs_collection_once_its_dtstart_has_passed(test_config, tmp_path):
    """Once the rule's first occurrence has passed the collection is eligible."""
    collector, db = _make_collector(test_config, tmp_path)
    started = datetime.now(UTC) - timedelta(minutes=1)
    db.memories.create_collection(
        "due",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule=f"DTSTART:{started.strftime(_RRULE_STAMP)}\nFREQ=DAILY;COUNT=1",
    )
    target = collector._next_ready_collection()
    assert target is not None and target.name == "due"


def test_schedule_ready_only_at_next_occurrence(test_config, tmp_path):
    """A collection is ready iff ``now`` has reached the rule's next occurrence after
    its last run (#1857).  Deterministic around a fixed clock: BYHOUR=8,20 fires at
    08:00 and 20:00 UTC, so a run at 08:00 makes 20:00 the next fire."""
    collector, _ = _make_collector(test_config, tmp_path)
    memory = MemoryRow(
        name="twice-daily",
        type="collection",
        description="d",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule=_TWICE_DAILY,
        created_at=datetime(2026, 7, 20, 0, 0, tzinfo=UTC),
        last_collected_at=datetime(2026, 7, 20, 8, 0, tzinfo=UTC),
    )
    # Between occurrences (noon) → not due.
    assert collector._is_ready(memory, datetime(2026, 7, 20, 12, 0, tzinfo=UTC)) is False
    # Exactly at the next occurrence (20:00) → due.
    assert collector._is_ready(memory, datetime(2026, 7, 20, 20, 0, tzinfo=UTC)) is True
    # Past it → still due.
    assert collector._is_ready(memory, datetime(2026, 7, 20, 20, 30, tzinfo=UTC)) is True


def test_schedule_never_run_bases_next_occurrence_on_created_at(test_config, tmp_path):
    """A collection that has never run counts from ``created_at`` — which is also the
    rule's default start, so one created between occurrences waits for the next."""
    collector, _ = _make_collector(test_config, tmp_path)
    memory = MemoryRow(
        name="fresh",
        type="collection",
        description="d",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule=_TWICE_DAILY,
        last_collected_at=None,
        created_at=datetime(2026, 7, 20, 9, 0, tzinfo=UTC),  # after the 08:00 occurrence
    )
    assert collector._is_ready(memory, datetime(2026, 7, 20, 12, 0, tzinfo=UTC)) is False
    assert collector._is_ready(memory, datetime(2026, 7, 20, 20, 0, tzinfo=UTC)) is True


def test_schedule_without_dtstart_is_due_immediately_then_paces(test_config, tmp_path):
    """A rule with no DTSTART anchors at ``created_at``, so a freshly created
    collection is due on the very next tick — and once it has run, the rule paces it."""
    collector, _ = _make_collector(test_config, tmp_path)
    created = datetime(2026, 7, 20, 9, 0, tzinfo=UTC)
    memory = MemoryRow(
        name="hourly",
        type="collection",
        description="d",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule="FREQ=HOURLY",
        created_at=created,
        last_collected_at=None,
    )
    assert collector._is_ready(memory, created) is True
    memory.last_collected_at = created
    assert collector._is_ready(memory, created + timedelta(minutes=30)) is False
    assert collector._is_ready(memory, created + timedelta(hours=1)) is True


def test_exhausted_schedule_is_never_ready_again(test_config, tmp_path):
    """A rule whose occurrences are spent (a used-up ``COUNT=``) yields no next
    occurrence, so the collection never becomes ready again — the run-quota archive
    then retires it."""
    collector, _ = _make_collector(test_config, tmp_path)
    created = datetime(2026, 7, 20, 9, 0, tzinfo=UTC)
    memory = MemoryRow(
        name="one-shot",
        type="collection",
        description="d",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule="FREQ=DAILY;COUNT=1",
        created_at=created,
        last_collected_at=created,
    )
    assert collector._is_ready(memory, created + timedelta(days=30)) is False


def test_unreadable_stored_schedule_skips_rather_than_crashing(test_config, tmp_path):
    """A stored rule the parse gate would have refused (only reachable by a hand-edited
    row) skips that ONE collection — the dispatcher keeps serving every other."""
    collector, _ = _make_collector(test_config, tmp_path)
    memory = MemoryRow(
        name="hand-edited",
        type="collection",
        description="d",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule="every 3600",
        created_at=datetime(2026, 7, 20, 9, 0, tzinfo=UTC),
    )
    assert collector._is_ready(memory, datetime(2026, 7, 21, 9, 0, tzinfo=UTC)) is False


def test_max_runs_archives_after_quota(test_config, tmp_path):
    """After ``max_runs`` completed (non-cancelled) cycles the scheduler archives
    the collection — a one-shot reminder retires itself.  The count is read from
    the ledger (completed promptlog runs), and a cancelled run never counts."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "one-shot",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule="FREQ=HOURLY;COUNT=2",
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
        schedule="FREQ=HOURLY",
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
        schedule="FREQ=HOURLY",
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
        schedule="FREQ=HOURLY",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    def handler(request: dict, count: int) -> LlmResponse:
        if count == 1:  # a working step, then the close — an ordinary cycle shape
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {"memory": "expiring-watch", "entries": [{"key": "today", "content": "a fact"}]},
            )
        return mock_llm._make_tool_call_response(request, DoneTool.name, {})

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
        schedule="FREQ=HOURLY",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db.memories.create_collection(
        "eternal-watch",
        "x",
        extraction_prompt=_ONE_SHOT_PROMPT,
        schedule="FREQ=HOURLY",
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
    rules are load-bearing (batched writes, in-place corrections, browse provenance,
    no manual dedup) — chat doesn't relay them, the collector base attaches
    them on every cycle.  The injected tail is the terminal ``done()`` alone,
    numbered continuing from the stored prompt (#1557, restored #1916): the stored
    program's highest step is 3, so the close is step 4.

    This is the no-routine collection — hand-authored or seeded, ``skill_name``
    NULL — so the routine block #1907 adds is absent entirely and the prompt is
    byte-identical to what it was before the join existed."""
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

    composed = Collector._compose_prompt(target, None, _FULL_SURFACE)

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


# ── The composed prompt carries the skill and the terminal (#1911/#1916) ─────

# A skill-rendered extraction_prompt (every step one canonical tool call, NO
# done() — the chat ledger has no done tool, so a render cannot produce one) —
# the kitchen-sink case the composed prompt is asserted against.
_NOTIFY_RENDERED_PROMPT = (
    "Collect indie metroidvania releases and keep me posted on the good ones.\n"
    '1. browse(queries=["new indie metroidvania releases"], extract="pull out the '
    'release name, a one-line hook, and the URL")\n'
    '2. collection_write("indie-metroidvanias", entries=[{key: <release name>, '
    "content: <name + hook + URL>}])"
)


def test_compose_prompt_carries_the_skill_and_the_terminal_when_notify_true():
    """THE WATCHED DELETION (#1911): a notify=true collection's composed prompt is the
    stored program plus the terminal close, and nothing else — asserted char-for-char,
    so the retired notify tail cannot come back by accident.

    What is gone is the four notify steps: telling the user is a framework-entered
    micro-context after the cycle, not four more steps inside it — 42 of 49 measured
    cycle deaths landed in that tail.  A notify=true prompt is byte-identical to the
    same collection's notify=false prompt, because nothing about telling the user is
    part of a cycle any more.

    What is BACK is the injected terminal ``done()`` (#1916), numbered after the
    program's own two steps.  Deriving the close from coverage instead fixed the route
    in advance, so a cycle that departed from it for a good reason had nothing left to
    close on: it could neither reach coverage nor say it had finished."""
    quiet = MemoryRow(
        name="indie-metroidvanias",
        type="collection",
        description="Indie metroidvania releases the user tracks",
        archived=False,
        extraction_prompt=_NOTIFY_RENDERED_PROMPT,
    )
    target = quiet.model_copy(update={"notify": True})

    composed = Collector._compose_prompt(target, None, _FULL_SURFACE)

    expected = (
        "You are the collector for the `indie-metroidvanias` collection.\n"
        "Description: Indie metroidvania releases the user tracks\n"
        "\n"
        f"{_NOTIFY_RENDERED_PROMPT}\n"
        "3. done()\n"
        "\n"
        f"{Collector._runtime_rules(_FULL_SURFACE)}"
    )
    assert composed == expected, (
        f"Composed notify prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )
    # The flag no longer changes a single character of what the model reads.
    assert Collector._compose_prompt(quiet, None, _FULL_SURFACE) == composed
    # Named gate cases: the retired notify steps are absent, the terminal is present.
    for retired in ("read_similar", "send_message"):
        assert retired not in composed
    assert f"3. {Prompt.COLLECTOR_DONE_STEP}" in composed


def test_compose_prompt_numbers_the_terminal_from_one_on_a_prose_prompt():
    """The boundary case of the step numbering (#1557, restored #1916): an unnumbered
    prose prompt has no highest step, so ``A`` is 0 and the injected terminal is step 1
    — the whole prompt still reads as one continuous program rather than a paragraph
    with a step number out of nowhere under it.

    Uniform for legacy hand-authored collections: the tail is the close alone, so a
    prose prompt gains exactly one line and nothing about telling the user."""
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

    composed = Collector._compose_prompt(target, None, _FULL_SURFACE)

    expected = (
        "You are the collector for the `summit-status` collection.\n"
        "Description: Summit trail status\n"
        "\n"
        f"{legacy_prompt}\n"
        "1. done()\n"
        "\n"
        f"{Collector._runtime_rules(_FULL_SURFACE)}"
    )
    assert composed == expected, (
        f"Legacy notify prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )


def test_the_retired_notify_prompt_constant_is_gone():
    """THE WATCHED DELETION, named (#1911): the prompt constant that carried the
    in-cycle notify tail no longer exists.

    Pinned as a gate rather than left to the absence of a reference, because a
    re-introduction would be silent otherwise — and the whole design turns on the cycle
    having no notify steps at all: telling the user is a framework-entered
    micro-context that runs after the cycle closes."""
    assert not hasattr(Prompt, "COLLECTOR_NOTIFY_STEPS")


def test_the_terminal_step_constant_is_back():
    """THE REVERSION, named (#1916): assembly supplies a terminal step again, and it is
    the bare argless call — no ``success``, no ``summary``, nothing for the model to
    confabulate, since the run record is generated from the ledger.

    Pinned beside its retired sibling so the pair reads as one decision: the notify
    steps stay gone, the close comes back."""
    assert Prompt.COLLECTOR_DONE_STEP == "done()"


def test_send_message_is_not_on_the_collector_surface(test_config, tmp_path):
    """THE WATCHED DELETION, at the surface (#1911): a collector cycle has no
    ``send_message`` tool, so a cycle cannot message the user at all — telling them is
    the framework's, after the cycle closes.

    ``done`` is the one loop-control call left, and it is the ORDINARY end (#1916), not
    an early out.  The authoring-time guard reads the same surface, so a stored program
    naming ``send_message`` is refused rather than persisted, while one naming ``done``
    is accepted — assembly writes that step itself."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("board-games", "games", extraction_prompt="x" * 30)
    collector._current_target = db.memories.get("board-games")

    names = {tool.name for tool in collector.get_tools()}

    assert "send_message" not in names
    assert "send_message" not in collector_tool_surface(db, collector._model_client)


# ── The cycle's surface is scoped to its program (#1911) ──────────────────────

# A two-call program. Its scoped surface is those two calls plus the correction
# siblings ``collection_write``'s own rejection text points at — and nothing else.
_SCOPED_PROGRAM = (
    "1. browse(queries=['https://northpier.example/departures'], extract='the dawn sailing')\n"
    "2. collection_write(memory='ferry-departures', entries=[{'key': 'x', 'content': 'y'}])"
)

# NOT a program: prose whose steps do not open with a call, which since #1911's strict
# parser is unreadable rather than leniently scanned.
_PROSE_PROGRAM = (
    "Watch the summit webcam page, read the status banner, and record the current "
    "trail status in the collection under the key `trail`."
)


def _surface_for(collector: Collector, db: Database, name: str) -> set[str]:
    """The tool names the cycle for ``name`` actually runs with."""
    collector._bind(_get(db, name))
    return {tool.name for tool in collector.get_tools()}


def test_a_readable_program_scopes_the_surface_to_its_own_calls(test_config, tmp_path):
    """THE SCOPED SURFACE (#1911, the code owner's ruling): "we know the tools
    beforehand so we can dynamically restrict the tool calling surface to only the tools
    in the actual skill".

    The surface is EXACTLY the program's two calls CLOSED over the advice relation —
    asserted as an equality, because the point is what is ABSENT: ``read_similar`` is
    gone, so the interjected chat-flavoured read the measured tail decayed on is
    structurally unavailable rather than discouraged.

    The closure is TRANSITIVE, and ``collection_keys`` is the proof: the write's
    rejection names ``update_entry``, and ``update_entry``'s own not-found message names
    ``collection_keys`` — two hops from the program, and still reachable, because a
    rendered instruction has to resolve however deep the chain of advice runs.

    ``done`` is the one member the closure did NOT put there (#1916).  It joins
    unconditionally, because assembly injects the terminal step into every composed
    prompt and a step the prompt names must be a call the surface carries."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "ferry-departures", "the dawn sailing", extraction_prompt=_SCOPED_PROGRAM
    )

    surface = _surface_for(collector, db, "ferry-departures")

    assert surface == {
        "browse",
        "collection_write",
        "update_entry",
        "collection_delete_entry",
        "collection_keys",
        "done",
    }
    assert "read_similar" not in surface


def test_every_scoped_surface_carries_the_terminator(test_config, tmp_path):
    """THE REVERSION (#1916): ``done`` is back, and it rides EVERY collector surface —
    a scoped cycle can always say it has finished.

    The coverage exit it replaced fixed the route in advance: a cycle that took the
    store's own advice (an ``update_entry`` where the program said
    ``collection_write``) could never cover its program, and with no terminator to
    reach for it died rerolling on an attempt to state in prose that it was done — 11
    aborts in one measured run.  A model that has finished can always make one more
    call; it cannot always walk a path decided before it started.

    Asserted at four places because a terminator that exists but is not HANDED to the
    model is the same defect as no terminator: the bound cycle's surface, the SCHEMA the
    model actually reads, the authoring guard (so a stored program naming ``done`` is
    accepted rather than refused as a hallucination), and the tool registry itself."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "ferry-departures", "the dawn sailing", extraction_prompt=_SCOPED_PROGRAM
    )

    surface = _surface_for(collector, db, "ferry-departures")
    collector._install_tools(collector.get_tools())
    schema = {tool["function"]["name"] for tool in collector._tool_registry.get_ollama_tools()}

    assert DoneTool.name in surface
    assert DoneTool.name in schema
    assert DoneTool.name in collector_tool_surface(db, collector._model_client)
    assert Tool._registry.get(DoneTool.name) is DoneTool


def test_an_unreadable_program_has_only_the_terminator(test_config, tmp_path):
    """A collection whose stored prompt is NOT a rendered program is a CONFIG DEFECT
    (#1911), not a second way of running.

    The seeded prose rows that made it a mode were dropped in the soft reboot, so a
    prompt whose steps do not open with a call has no job this framework can run.  Its
    surface is the terminator ALONE (#1916) — there is nothing to hand it but the
    ability to close honestly — and its run record names the state rather than letting
    the collection fail quietly forever."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection("summit-status", "trail status", extraction_prompt=_PROSE_PROGRAM)

    surface = _surface_for(collector, db, "summit-status")

    assert surface == {DoneTool.name}
    assert collector._program == ()
    assert (
        collector._unfinished_reason(ControllerResponse(answer="", tool_calls=[]))
        == COLLECTOR_UNREADABLE_PROGRAM_REASON
    )


def test_a_scoped_cycle_reads_only_the_runtime_rules_it_can_carry_out(test_config, tmp_path):
    """The rules are filtered against the surface, so the prompt never instructs a call
    the model cannot make — a read-only routine reads a prompt about reading.

    Whole-render, both directions: a write program keeps every rule its tools support,
    and a read-only one has no rules block at all rather than a heading over nothing."""
    write_surface = frozenset(
        {"browse", "collection_write", "update_entry", "collection_delete_entry"}
    )
    assert Collector._runtime_rules(write_surface) == (
        "## Runtime rules (always apply)\n"
        "\n"
        "- Single batched `collection_write(entries=[...])` per cycle — not one call per entry.\n"
        "- For corrections: if a recent message indicates an existing entry is wrong, stale, "
        "closed, or otherwise no longer accurate, `update_entry(key=<key>, content=<corrected "
        "content>)` or `collection_delete_entry(key=<key>)` rather than appending alongside.\n"
        "- Cite only what you actually browsed this cycle.  Never invent a URL to populate a "
        '"Source:" field — if no real source was fetched, omit the field.\n'
        "- Don't dedup manually — the store rejects duplicates on write automatically."
    )
    # A write program that never browses drops the browse rule and keeps the rest.
    assert "browsed this cycle" not in Collector._runtime_rules(
        frozenset({"collection_write", "update_entry", "collection_delete_entry"})
    )
    # A read-only program supports none of them, so the head goes too.
    assert Collector._runtime_rules(frozenset({"log_read"})) == ""


@pytest.mark.asyncio
async def test_a_stuck_scoped_cycle_records_its_own_cause(mock_llm, test_config, tmp_path):
    """The honest end (#1909/#1911/#1916): a cycle that never finishes its program and
    never reaches for its terminator runs out its step budget, and its run record names
    what ended it rather than reporting a close it never made.

    The terminator being available again does not soften this: ``_cycle_result`` reads
    the ledger for a ``done`` record, so a cycle that made none is unfinished whatever
    the model intended, and the reason line says which of the unfinished shapes it
    was."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_collection(
        "ferry-departures",
        "the dawn sailing",
        extraction_prompt=_SCOPED_PROGRAM,
        schedule="FREQ=HOURLY",
    )

    def handler(request: dict, count: int) -> LlmResponse:
        # Never the write and never the close, so nothing ends the cycle but the cap.
        return mock_llm._make_tool_call_response(
            request, "browse", {"queries": ["https://northpier.example/departures"]}
        )

    mock_llm.set_response_handler(handler)

    success, _ = await collector.run_for("ferry-departures")

    assert success is False
    runs = [run for run in db.messages.get_prompt_log_runs() if run["agent_name"] == "collector"]
    assert len(runs) == 1
    assert runs[0]["run_outcome"] == RunOutcome.FAILED.value
    assert "max steps exceeded" in runs[0]["run_reason"]
    assert db.send_queue.next_pending() is None


# ── What a CONFIGURED collection's cycle reads (#1907) ────────────────────────


async def _configure_timetable_collection(db: Database) -> None:
    """A collection stood up the way chat stands one up: a taught routine, instantiated
    through the real front door with the values the job is pointed at."""
    seed_timetable_skill(db)
    result = await CollectionSetTool(db, cast(Any, MockLlmClient())).execute(
        name="ferry-departures",
        description="the dawn sailing on the north pier timetable",
        skill="check_timetable",
        params={"url": "https://northpier.example/departures", "line": "the dawn sailing"},
        schedule="FREQ=HOURLY",
    )
    assert result.success, result.message


async def _composed_for(collector: Collector, db: Database, name: str) -> str:
    """The per-cycle system prompt the collector really builds for ``name`` — bound and
    with its tools installed, exactly as ``_run_cycle`` does, so the prompt is composed
    against the surface this cycle actually runs with (#1911)."""
    collector._bind(_get(db, name))
    collector._install_tools(collector.get_tools())
    return await collector._build_system_prompt(None)


@pytest.mark.asyncio
async def test_configured_collection_cycle_reads_routine_values_and_joined_program(
    test_config, tmp_path
):
    """The whole composed prompt for a collection chat configured, byte-for-byte — the
    three things a cycle needs, in one prompt (#1907).

    The INSTRUCTIONS carry the page this job fetches and the thing it looks for, because
    the values were joined into the program's leaves when the collection was configured.
    Beside them the ROUTINE says what is running and what it is for, and the VALUES are
    listed by name.  Before this, a cycle read a program of descriptions — told to browse
    "the url of the timetable page to browse each run" — and nothing it could read named
    the page at all.

    The EMPTY holdings shape rides along (#1914): a collection that has never been
    written to says so in the plainest words there are, rather than carrying a heading
    over nothing or leaving the cycle to wonder whether the block failed to render."""
    collector, db = _make_collector(test_config, tmp_path)
    await _configure_timetable_collection(db)

    composed = await _composed_for(collector, db, "ferry-departures")

    expected = (
        "You are the collector for the `ferry-departures` collection.\n"
        "Description: the dawn sailing on the north pier timetable\n"
        "\n"
        "The routine you run: check_timetable — read a timetable page and record a "
        "sailing time\n"
        "The values it is pointed at:\n"
        "- url: https://northpier.example/departures\n"
        "- line: the dawn sailing\n"
        "\n"
        "1. browse(queries=['https://northpier.example/departures'], "
        "extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry-departures', "
        "entries=[{'key': {the key the extracted value is stored under}, "
        "'content': the value from step 1}])\n"
        "3. done()\n"
        "\n"
        f"{Collector._runtime_rules(_FULL_SURFACE)}"
        "\n"
        "\n"
        "## What this collection holds (newest first)\n"
        "It holds nothing yet."
    )
    assert composed == expected, (
        f"Configured collection prompt mismatch:\n{composed!r}\n\nvs expected:\n{expected!r}"
    )


@pytest.mark.asyncio
async def test_a_term_the_collection_was_never_given_reads_as_a_named_gap(test_config, tmp_path):
    """Visible degradation over silent success: re-teaching a routine so it needs
    something more leaves the running collection short of a term, and the cycle SAYS so,
    naming the term and what it wants.

    This is the reachable shape of an unbound parameter — the collection was configured
    against a routine that declared two things and now runs one that declares three — and
    the honest reading is that its stored program is still the one it was configured
    with, which the values block is what makes visible."""
    collector, db = _make_collector(test_config, tmp_path)
    await _configure_timetable_collection(db)
    seed_timetable_skill(
        db,
        parameters=[
            SkillParameter(name="url", description="the timetable page to read", value=_TAUGHT_URL),
            SkillParameter(
                name="line", description="which sailing to look for", value=_TAUGHT_LINE
            ),
            SkillParameter(name="cutoff", description="how late a sailing still counts"),
        ],
    )

    composed = await _composed_for(collector, db, "ferry-departures")

    assert composed == (
        "You are the collector for the `ferry-departures` collection.\n"
        "Description: the dawn sailing on the north pier timetable\n"
        "\n"
        "The routine you run: check_timetable — read a timetable page and record a "
        "sailing time\n"
        "The values it is pointed at:\n"
        "- url: https://northpier.example/departures\n"
        "- line: the dawn sailing\n"
        "- cutoff: nothing was supplied for this — it needs how late a sailing still "
        "counts\n"
        "\n"
        "1. browse(queries=['https://northpier.example/departures'], "
        "extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry-departures', "
        "entries=[{'key': {the key the extracted value is stored under}, "
        "'content': the value from step 1}])\n"
        "3. done()\n"
        "\n"
        f"{Collector._runtime_rules(_FULL_SURFACE)}"
        "\n"
        "\n"
        "## What this collection holds (newest first)\n"
        "It holds nothing yet."
    )


# ── What the collection holds (#1914) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_configured_cycle_reads_the_entries_the_collection_already_holds(
    test_config, tmp_path
):
    """THE PRESENTATION (#1914): the entries a collection holds are rendered in the
    cycle's own prompt, beside the program that acts on them.

    Measured on PR #1906's round-3 run, 15 of 25 quiet cycles completed-and-notified
    because the model invented a fresh key every time — nothing it could read said what
    key the value was already stored under, so the write gate saw a NEW_KEY where the
    value had not changed at all.  Each key renders in INVOCATION form (``key='…'``,
    the shared entry render the read tools use), so the key the next write should land
    on is copied rather than derived: the n≤1 anchor discipline, applied to the one
    fact a re-observing routine cannot do without.

    Presentation only — nothing here says a routine must reuse a key.  A dated digest
    reads its own past keys and files a new one, which is the same read."""
    collector, db = _make_collector(test_config, tmp_path)
    await _configure_timetable_collection(db)
    require_memory(db, "ferry-departures").write(
        [
            EntryInput(key="dawn sailing", content="06:40 from the north pier"),
            EntryInput(key="last return", content="21:15 from the island"),
        ],
        author="collector",
    )

    composed = _normalise_stamps(await _composed_for(collector, db, "ferry-departures"))

    assert composed == (
        "You are the collector for the `ferry-departures` collection.\n"
        "Description: the dawn sailing on the north pier timetable\n"
        "\n"
        "The routine you run: check_timetable — read a timetable page and record a "
        "sailing time\n"
        "The values it is pointed at:\n"
        "- url: https://northpier.example/departures\n"
        "- line: the dawn sailing\n"
        "\n"
        "1. browse(queries=['https://northpier.example/departures'], "
        "extract='the dawn sailing')\n"
        "2. collection_write(memory='ferry-departures', "
        "entries=[{'key': {the key the extracted value is stored under}, "
        "'content': the value from step 1}])\n"
        "3. done()\n"
        "\n"
        f"{Collector._runtime_rules(_FULL_SURFACE)}"
        "\n"
        "\n"
        "## What this collection holds (newest first)\n"
        "The entries it holds right now, and the key each one is stored under.\n"
        f"1. {_STAMP} key='last return' 21:15 from the island\n"
        f"2. {_STAMP} key='dawn sailing' 06:40 from the north pier"
    ), f"Holdings render mismatch:\n{composed!r}"


def _expected_holdings(newest_first: range, tail: str) -> str:
    """The whole holdings block for the numbered day-entries the truncation test seeds,
    written out independently of the production render so the assertion is a contract
    rather than an echo."""
    body = "\n".join(
        f"{position}. {_STAMP} key='day-{index:02d}' reading {index}"
        for position, index in enumerate(newest_first, start=1)
    )
    return (
        "\n"
        "\n"
        "## What this collection holds (newest first)\n"
        "The entries it holds right now, and the key each one is stored under.\n"
        f"{body}\n"
        f"{tail}"
    )


@pytest.mark.asyncio
async def test_holdings_beyond_the_budget_state_how_many_are_not_shown(test_config, tmp_path):
    """The TRUNCATED shape: a collection deeper than the block's reading budget renders
    its newest entries and then says, in a number, how many it left out.

    A prompt-budget bound of the self-state header's kind — the cut is stated, never
    silent, so a cycle reasoning about a collection it cannot see all of knows that is
    what it is doing.  The overflow line names no fetch tool on purpose: a cycle's
    surface is scoped to its program's calls, so a tool named here might not be on it.

    Both counts are pinned — a remainder of two and, after one of the hidden entries is
    deleted, of one — because the singular arm is the one a plural-only literal would
    let rot."""
    collector, db = _make_collector(test_config, tmp_path)
    await _configure_timetable_collection(db)
    collection = require_memory(db, "ferry-departures")
    limit = PennyConstants.COLLECTOR_HOLDINGS_LIMIT
    for index in range(1, limit + 3):
        collection.write(
            [EntryInput(key=f"day-{index:02d}", content=f"reading {index}")], author="collector"
        )

    section = _normalise_stamps(collector._holdings_section("ferry-departures"))

    # Newest-first: the last-written `limit` entries show, the two oldest are counted.
    shown = range(limit + 2, 2, -1)
    assert section == _expected_holdings(shown, "2 older entries not shown."), (
        f"Truncated holdings mismatch:\n{section!r}"
    )

    # One of the hidden entries goes: the remainder reads as the one entry it now is.
    collection.delete("day-01")
    section = _normalise_stamps(collector._holdings_section("ferry-departures"))
    assert section == _expected_holdings(shown, "1 older entry not shown."), (
        f"Singular remainder mismatch:\n{section!r}"
    )


@pytest.mark.asyncio
async def test_a_collection_that_vanished_mid_cycle_renders_no_holdings_block(
    test_config, tmp_path
):
    """Visible degradation, not a blank: a target whose row is gone by the time the
    prompt is composed leaves the block out entirely — the prompt is byte-identical to
    what it was — and says so in the log rather than rendering a heading over nothing."""
    collector, db = _make_collector(test_config, tmp_path)
    await _configure_timetable_collection(db)

    assert collector._holdings_section("a-collection-that-is-not-there") == ""


_NOTIFY_SEED_KEY = "Hollow Verge"
_NOTIFY_SEED_CONTENT = "Hollow Verge — a hand-drawn metroidvania. https://ex.example/hv"

# The two-call program a notify collection runs in these tests: read what is already
# stored, then write what this cycle found.  Both calls are real collector tools that
# need nothing but the database, so a cycle can be driven end to end without stubbing
# the outside world — and two calls is what puts a step between the read and the close,
# which is what makes a STOP landing on the WRITE distinguishable from one landing
# anywhere else.
_NOTIFY_PROGRAM = (
    "Keep the user posted on new indie metroidvania releases.\n"
    '1. collection_read_latest(memory="indie-metroidvanias")\n'
    '2. collection_write("indie-metroidvanias", entries=[{key: <release name>, '
    "content: <name + hook + URL>}])"
)

_DRAWN_MESSAGE = "Hey! Cinder Drift just dropped — a new metroidvania. https://ex.example/cd"


def _seed_notify_collection(
    db: Database, *, notify: bool = True, extraction_prompt: str = _NOTIFY_PROGRAM
) -> None:
    """A collection running the two-call program above, holding one existing entry,
    plus a primary user so a queued notification has a recipient.

    The seeded entry carries its VECTORS, as every production entry does (the write path
    embeds, and the startup backfill fills any gap).  Without them the dedup
    disjunction's content signal cannot be scored at all, so a re-observation collides on
    the key alone and the write gate reads it as a DIFFERENT value under a similar key —
    the conservative answer to no evidence (#1919), and not the state this family means
    to be in.

    ``extraction_prompt`` is a parameter so a test can run the same collection on a
    READ-ONLY routine — the shape that has nothing to mutate, so the only thing its
    cycle does is tell the user (#1914)."""
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
        extraction_prompt=extraction_prompt,
        schedule="FREQ=HOURLY",
        notify=notify,
    )
    require_memory(db, "indie-metroidvanias").write(
        [
            EntryInput(
                key=_NOTIFY_SEED_KEY,
                content=_NOTIFY_SEED_CONTENT,
                key_embedding=deterministic_embed(_NOTIFY_SEED_KEY),
                content_embedding=deterministic_embed(_NOTIFY_SEED_CONTENT),
            )
        ],
        author="producer",
    )


def _results_so_far(request: dict) -> int:
    """How many tool results this cycle's conversation already carries.

    The step a mock handler should take next, read off the CONVERSATION rather than off
    a call ordinal — so a test that runs two cycles gets the same behaviour in each, and
    a cycle cut short by a write-gate STOP simply never asks for the step after it."""
    return sum(1 for message in request["messages"] if message.get("role") == "tool")


def _program_handler(mock_llm, *, key: str = "Cinder Drift", content: str, drawn: str | None):
    """A handler that runs the two-call program, CLOSES with ``done()``, and then
    answers the notify draw.

    It branches on the CONVERSATION — the notify draw by its own system prompt, the
    program step by how many tool results have come back — so the three cycle steps
    (read, write, close) fall out in order and a STOPped write never reaches the close.

    ``drawn`` is what the notify micro-context replies; ``None`` makes it reply with
    something that is not a message at all, which is how the no-usable-draw path is
    exercised without reaching for a mock of the micro-context itself."""

    def handler(request: dict, count: int) -> LlmResponse:
        if request["messages"][0].get("content", "") == NOTIFY_SYSTEM_PROMPT:
            return mock_llm._make_text_response(request, drawn or "I could not think of anything.")
        results = _results_so_far(request)
        if results == 0:
            return mock_llm._make_tool_call_response(
                request, "collection_read_latest", {"memory": "indie-metroidvanias"}
            )
        if results == 1:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {"memory": "indie-metroidvanias", "entries": [{"key": key, "content": content}]},
            )
        return mock_llm._make_tool_call_response(request, DoneTool.name, {})

    return handler


def _collector_run(db: Database) -> dict:
    """The CYCLE's own run row.

    Selected by agent name because the notify draw is its own ledger-visible run beside
    it (that separation is the point: two contexts, two runs), so "the run" has to say
    which one."""
    runs = [run for run in db.messages.get_prompt_log_runs() if run["agent_name"] == "collector"]
    assert len(runs) == 1
    return runs[0]


def _run_reason(db: Database) -> str:
    """The reason stamped on the CYCLE's run — the run record's own account of which
    terminal shape it had."""
    return _collector_run(db)["run_reason"]


def _notify_draws(db: Database) -> list[dict]:
    """The notify micro-context's own runs, attributed under its own ledger identity."""
    return [
        run
        for run in db.messages.get_prompt_log_runs()
        if run["agent_name"] == PennyConstants.NOTIFY_COMPOSE_AGENT_NAME
    ]


@pytest.mark.asyncio
async def test_a_done_closed_cycle_queues_exactly_one_notification(mock_llm, test_config, tmp_path):
    """THE TRIGGER (#1911/#1916): a notify=true cycle that CLOSES with ``done()`` and no
    STOP enters the notify micro-context and hands what it writes to the existing send
    queue.

    Three things are asserted together because they are one mechanism: the loop closed
    on the terminator (the read, the write, then the close — three cycle calls), the
    micro-context was asked exactly once on its own scoped prompt, and the queue holds
    the drawn message under the collection that produced it.  The run record says which
    terminal shape this was, and a clean close carries only what telling the user came
    to — there is no completion phrase in front of it any more."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            content="Cinder Drift — a new metroidvania. https://ex.example/cd",
            drawn=f"MESSAGE: {_DRAWN_MESSAGE}",
        )
    )

    success, _ = await collector.run_for("indie-metroidvanias")

    assert success is True
    # The two program calls and the close, then the ONE notify draw — nothing else.
    assert len(mock_llm.requests) == 4
    assert len(_notify_draws(db)) == 1  # its own ledger identity, beside the cycle's run
    pending = db.send_queue.pending_items()
    assert [item.content for item in pending] == [_DRAWN_MESSAGE]
    assert pending[0].collection == "indie-metroidvanias"
    # This cycle was the USER's own "run this now", and the queued row says so (#1939)
    # — which is what lets the drainer hand it over without waiting out the
    # autonomous-send cooldown meant for messages nobody asked for.
    assert pending[0].origin == CycleTrigger.ON_DEMAND
    assert _run_reason(db) == NOTIFICATION_NOTES[NotificationOutcome.QUEUED]


@pytest.mark.asyncio
async def test_a_done_closed_cycle_without_notify_stays_quiet(mock_llm, test_config, tmp_path):
    """The completed-QUIET shape: the same done-closed cycle on a collection that does
    not notify draws nothing and queues nothing.

    Its record carries NO reason at all (#1916) — a bare clean close has nothing to
    report beyond the outcome enum the header falls back to, and the completion phrase
    that used to open every finished cycle's reason retired with the coverage read that
    was the only thing distinguishing it from a stop."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, notify=False)
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            content="Cinder Drift — a new metroidvania. https://ex.example/cd",
            drawn=f"MESSAGE: {_DRAWN_MESSAGE}",
        )
    )

    await collector.run_for("indie-metroidvanias")

    assert len(mock_llm.requests) == 3  # the program's two calls + the close, no draw
    assert db.send_queue.next_pending() is None
    assert _run_reason(db) == ""


# A routine with no write in it at all — the shape the produced-work fold exists for:
# nothing it does carries ``mutated=True``, so the ONLY thing its cycle produced is the
# notification the framework queued afterwards.
_READ_ONLY_PROGRAM = (
    "Look in on what the collection already holds and tell the user about it.\n"
    '1. collection_read_latest(memory="indie-metroidvanias")'
)


def _read_only_handler(mock_llm, *, drawn: str):
    """Runs the one-call read-only program, closes with ``done()``, then answers the
    notify draw — branching on the CONVERSATION (the draw by its own system prompt, the
    cycle step by how many results have come back) like ``_program_handler``."""

    def handler(request: dict, count: int) -> LlmResponse:
        if request["messages"][0].get("content", "") == NOTIFY_SYSTEM_PROMPT:
            return mock_llm._make_text_response(request, drawn)
        if _results_so_far(request) == 0:
            return mock_llm._make_tool_call_response(
                request, "collection_read_latest", {"memory": "indie-metroidvanias"}
            )
        return mock_llm._make_tool_call_response(request, DoneTool.name, {})

    return handler


@pytest.mark.asyncio
async def test_a_read_only_cycle_that_told_the_user_records_worked(mock_llm, test_config, tmp_path):
    """THE FOLD, end to end (#1914): a cycle whose routine only READ, and which then
    told the user what it found, records ``worked``.

    Its tool trace carries no mutation at all, and since #1911 no tool call carries the
    notification either — so read against the trace alone the cycle looks idle, and the
    run record said ``no_work`` on the cycle that did the single thing the collection
    exists for.  The reason line already named the notification; the outcome now agrees
    with it."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, extraction_prompt=_READ_ONLY_PROGRAM)
    mock_llm.set_response_handler(_read_only_handler(mock_llm, drawn=f"MESSAGE: {_DRAWN_MESSAGE}"))

    await collector.run_for("indie-metroidvanias")

    assert [item.content for item in db.send_queue.pending_items()] == [_DRAWN_MESSAGE]
    run = _collector_run(db)
    assert run["run_outcome"] == RunOutcome.WORKED.value
    assert run["run_reason"] == NOTIFICATION_NOTES[NotificationOutcome.QUEUED]


@pytest.mark.asyncio
async def test_a_read_only_cycle_that_told_nobody_still_records_no_work(
    mock_llm, test_config, tmp_path
):
    """The other direction, unchanged: the same read-only cycle on a collection that
    does NOT notify changed nothing and reached nobody, so it stays ``no_work``.

    The fold counts a queued notification and nothing else — it does not turn every
    completed cycle into work."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, notify=False, extraction_prompt=_READ_ONLY_PROGRAM)
    mock_llm.set_response_handler(_read_only_handler(mock_llm, drawn=f"MESSAGE: {_DRAWN_MESSAGE}"))

    await collector.run_for("indie-metroidvanias")

    assert db.send_queue.next_pending() is None
    run = _collector_run(db)
    assert run["run_outcome"] == RunOutcome.NO_WORK.value
    assert run["run_reason"] == ""


@pytest.mark.asyncio
async def test_an_undrawable_notification_records_honestly_and_reverts_the_cycle(
    mock_llm, test_config, tmp_path
):
    """Visible degradation (#1911) plus the ratified revert (#1936): when no usable
    message can be drawn within the reroll budget, the cycle sends NOTHING, its run
    record says so, and the WHOLE run is thrown out.

    A failed notification draw is not a healthy end — the collector never reached the
    point of reporting anything — so keeping the write would leave the next cycle
    re-observing a value the user was never told about, which is the same swallowed
    change an abort mid-browse causes.  The write is undone and the occurrence stays
    due, so the retry re-observes the change and reports it.

    The draw here comes back as prose with no MESSAGE line at all, which is a contract
    violation the micro-context re-rolls and then fails honestly."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            content="Cinder Drift — a new metroidvania. https://ex.example/cd",
            drawn=None,
        )
    )

    await collector.run_for("indie-metroidvanias")

    assert db.send_queue.next_pending() is None
    # The find is GONE: the cycle that made it never got to report it, so the store is
    # back to the one seeded entry and the retry starts from there.
    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift") == []
    assert [entry.key for entry in require_memory(db, "indie-metroidvanias").read_all()] == [
        _NOTIFY_SEED_KEY
    ]
    assert _get(db, "indie-metroidvanias").last_collected_at is None  # kept for the retry
    # The record carries the note alone (#1916) — a clean close has no phrase of its own
    # to open with — and the note is quoted here because "says so" is this test's whole
    # subject: a reason that named the outcome enum instead would be a silent skip.
    assert _run_reason(db) == NOTIFICATION_NOTES[NotificationOutcome.NOT_DRAWN]
    assert _run_reason(db) == "nothing was sent — no usable message could be written"


@pytest.mark.asyncio
async def test_notify_cycle_sends_nothing_on_a_no_change_write(mock_llm, test_config, tmp_path):
    """STOP interplay (#1557, carried to #1911/#1916): a notify=true run that
    re-observes the watched key with an UNCHANGED value STOPs at the write gate, so a
    no-change cycle emits NOTHING.

    The STOP ends the run at the chokepoint, before the cycle ever reaches the step that
    would close it — which is why the trigger reads the STOP as well as the close.  The
    write's own result is what carries it, so no-news is silent structurally rather than
    by the model choosing not to mention it."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            key=_NOTIFY_SEED_KEY,
            content=_NOTIFY_SEED_CONTENT,
            drawn=f"MESSAGE: {_DRAWN_MESSAGE}",
        )
    )

    await collector.run_for("indie-metroidvanias")

    # The read, then the write that STOPped — and no notify draw after it.
    assert len(mock_llm.requests) == 2
    assert db.send_queue.next_pending() is None
    assert _run_reason(db) == WRITE_GATE_STOP_REASONS[WriteGateOutcome.KEY_EXISTS_UNCHANGED]


@pytest.mark.asyncio
async def test_notify_cycle_sends_nothing_when_the_value_is_stored_under_another_key(
    mock_llm, test_config, tmp_path
):
    """The SAME no-news through the other door (#1919): the cycle re-observes what it
    already holds but words the key differently, so the exact-key comparison never runs
    and the dedup disjunction answers instead — a strict CONTENT match, which is
    DUPLICATE_UNCHANGED and STOPs exactly as the exact key does.  Nothing is queued.

    The run record is the point of the pin.  The STOP lands on the cycle's SECOND call,
    after a real read has already been persisted, so there IS a promptlog row for
    ``set_run_outcome`` to stamp and the declared stop reason reaches it.  A STOP on a
    cycle's very first call leaves no row at all — the run record reads empty and every
    cycle-shaped assertion silently scores against an absent ledger, which is the blind
    spot this pin exists to keep closed."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            key="hollow verge",  # the same release, worded differently — not the exact key
            content=_NOTIFY_SEED_CONTENT,
            drawn=f"MESSAGE: {_DRAWN_MESSAGE}",
        )
    )

    await collector.run_for("indie-metroidvanias")

    assert len(mock_llm.requests) == 2
    assert db.send_queue.next_pending() is None
    assert _run_reason(db) == WRITE_GATE_STOP_REASONS[WriteGateOutcome.DUPLICATE_UNCHANGED]
    # The value stays filed under the key it already had — a re-observation adds nothing.
    entries = require_memory(db, "indie-metroidvanias").read_all()
    assert [entry.key for entry in entries] == [_NOTIFY_SEED_KEY]


@pytest.mark.asyncio
async def test_changed_cycle_auto_refreshes_baseline_then_next_cycle_is_quiet(
    mock_llm, test_config, tmp_path
):
    """The anti-spam proof (#1633, carried to #1911/#1916): the last prose gate in the
    watch chain is gone.

    A notify=true watch collector observes its key.  The source value CHANGES, so the
    model writes the SAME key with a new value → the write gate auto-refreshes the
    stored baseline IN PLACE (stamping the writing run) and, because CHANGED is not a
    STOP, the cycle runs on to its close and the framework tells the user ONCE.  The
    NEXT cycle re-observes the now-current value: the gate reads UNCHANGED and STOPs at
    the write, so it never reaches the close and emits NOTHING.  Changed once →
    notified once → quiet."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)  # baseline: _NOTIFY_SEED_KEY = _NOTIFY_SEED_CONTENT
    new_value = f"{_NOTIFY_SEED_CONTENT} — now with a playable demo!"

    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            key=_NOTIFY_SEED_KEY,
            content=new_value,
            drawn=f"MESSAGE: Update on {_NOTIFY_SEED_KEY}!",
        )
    )
    await collector.run_for("indie-metroidvanias")

    # The gate auto-refreshed the baseline in place: one row, now the new value,
    # stamped by the writing run — via the write alone, no update_entry.
    stored = require_memory(db, "indie-metroidvanias").get(_NOTIFY_SEED_KEY)
    assert len(stored) == 1
    assert stored[0].content == new_value
    assert stored[0].last_written_by_run_id is not None
    assert [item.content for item in db.send_queue.pending_items()] == [
        f"Update on {_NOTIFY_SEED_KEY}!"
    ]

    # Cycle 2: the source is unchanged since the refresh, so the write STOPs and the
    # notify draw is never reached.
    requests_before_cycle_2 = len(mock_llm.requests)
    await collector.run_for("indie-metroidvanias")

    # The read, then the write that STOPped — and no notify draw after it.
    assert len(mock_llm.requests) == requests_before_cycle_2 + 2
    assert len(db.send_queue.pending_items()) == 1


# ── A cycle's writes survive only its healthy conclusion (#1936) ─────────────


def test_the_cycle_end_tables_are_total_and_disjoint():
    """The healthy-end enumeration is DATA, so its totality is the contract (#1936).

    Two tables partition ``CycleEnd``: an end is healthy (keep the writes + spend the
    occurrence) or it carries the line the retry logs.  Both are indexed with NO default
    — a member in neither, or in both, would be a ``KeyError`` raised inside a cycle's
    ``finally`` or a silent second answer, so the partition is pinned here rather than
    left to whoever adds the next end.  Same for the notification arm: every outcome a
    finished cycle can have maps to exactly one end.
    """
    assert set(CycleEnd) >= HEALTHY_CYCLE_ENDS
    assert HEALTHY_CYCLE_ENDS | set(CYCLE_END_RETRY_REASONS) == set(CycleEnd)
    assert not HEALTHY_CYCLE_ENDS & set(CYCLE_END_RETRY_REASONS)
    assert set(NOTIFIED_CYCLE_ENDS) == set(NotificationOutcome)
    assert set(NOTIFIED_CYCLE_ENDS.values()) <= set(CycleEnd)


def _writes_then_dies(mock_llm, *, key: str, content: str):
    """A cycle that WRITES the changed value and is then killed on its next model call.

    The shape the undo journal exists for: everything up to the write went fine, and the
    run died before it could close or report anything.  It counts its own calls rather
    than the mock's ordinal so a second cycle in the same test starts over.
    """
    calls = 0

    def handler(request: dict, count: int) -> LlmResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {"memory": "indie-metroidvanias", "entries": [{"key": key, "content": content}]},
            )
        raise LlmConnectionError("Connection refused")

    return handler


@pytest.mark.asyncio
async def test_an_aborted_cycles_write_is_reverted_so_the_retry_still_reports_the_change(
    mock_llm, test_config, tmp_path
):
    """THE PRODUCTION FAILURE, end to end (#1936).

    A watch cycle observes a CHANGED value, writes it, and then dies before it can close
    or tell anyone.  With the write left behind, the retry re-observed the same page,
    wrote the identical value, and the change-gate read it as UNCHANGED → STOP → the
    user was never told about a change that really happened.  The run consumed the
    change and delivered nothing.

    The journal makes the aborted cycle leave NOTHING: the entry holds the pre-run value
    again, under the pre-run stamps, with no partial write and no refreshed baseline —
    so the retry meets the same world the first attempt did, classifies CHANGED, closes,
    and the notification fires.

    The LEDGER is not part of the deal: the aborted run's promptlog rows are there, with
    its terminal cause on them, because the forensic record of a dead cycle is exactly
    what an operator needs.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)  # baseline: _NOTIFY_SEED_KEY = _NOTIFY_SEED_CONTENT
    new_value = f"{_NOTIFY_SEED_CONTENT} — now with a playable demo!"

    mock_llm.set_response_handler(
        _writes_then_dies(mock_llm, key=_NOTIFY_SEED_KEY, content=new_value)
    )
    success, _ = await collector.run_for("indie-metroidvanias")
    assert success is False

    # PRE-RUN STATE: the value, the creator and the last-writer stamps are all as the
    # seeding write left them — no partial write, no refreshed baseline.
    stored = require_memory(db, "indie-metroidvanias").get(_NOTIFY_SEED_KEY)
    assert len(stored) == 1
    assert stored[0].content == _NOTIFY_SEED_CONTENT
    assert stored[0].author == "producer"
    assert stored[0].created_by_run_id is None
    assert stored[0].last_written_by_run_id is None
    assert db.send_queue.next_pending() is None
    # The occurrence was not spent, so the very next dispatch is the retry (#1935).
    assert _get(db, "indie-metroidvanias").last_collected_at is None
    assert collector._retry_attempts == {("indie-metroidvanias", CycleTrigger.ON_DEMAND): 1}

    # THE LEDGER STAYS: the dead cycle's own rows are on record, naming what killed it.
    aborted_run = _collector_run(db)
    assert aborted_run["run_outcome"] == RunOutcome.FAILED.value
    assert "model call failed" in aborted_run["run_reason"]

    # THE RETRY: the same observation, against the untouched baseline — CHANGED, closed,
    # and the user is told.
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            key=_NOTIFY_SEED_KEY,
            content=new_value,
            drawn=f"MESSAGE: Update on {_NOTIFY_SEED_KEY}!",
        )
    )
    assert (await collector.run_for("indie-metroidvanias"))[0] is True

    refreshed = require_memory(db, "indie-metroidvanias").get(_NOTIFY_SEED_KEY)
    assert len(refreshed) == 1
    assert refreshed[0].content == new_value
    assert refreshed[0].last_written_by_run_id is not None
    assert [item.content for item in db.send_queue.pending_items()] == [
        f"Update on {_NOTIFY_SEED_KEY}!"
    ]
    assert _get(db, "indie-metroidvanias").last_collected_at is not None
    assert collector._retry_attempts == {}


@pytest.mark.asyncio
async def test_a_cycle_cut_short_leaves_nothing_behind_either(mock_llm, test_config, tmp_path):
    """The other two unhealthy ends, on the same claim as the abort (#1936).

    A cycle PREEMPTED by a foreground message and one that RAN OUT OF STEPS both stop
    short of a conclusion, so both are thrown away whole — and neither is one of the
    three causes the ruling names, which is the point: what decides the revert is the
    absence of a healthy end, not a list of ways to fail.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, notify=False)
    calls = 0

    def writes_once(request: dict, count: int) -> LlmResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "indie-metroidvanias",
                    "entries": [{"key": "Cinder Drift", "content": "Cinder Drift — out now."}],
                },
            )
        raise asyncio.CancelledError

    mock_llm.set_response_handler(writes_once)
    with pytest.raises(asyncio.CancelledError):
        await collector.run_for("indie-metroidvanias")

    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift") == []
    assert _get(db, "indie-metroidvanias").last_collected_at is None

    # The step cap: one step, spent on the write, so the cycle never reaches a close.
    collector.get_max_steps = lambda: 1  # ty: ignore[invalid-assignment]
    calls = 0
    success, _ = await collector.run_for("indie-metroidvanias")

    assert success is False
    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift") == []
    assert _get(db, "indie-metroidvanias").last_collected_at is None
    # Both cycles are on the ledger; this one is the FAILED bail — the cap left it
    # unfinished, and its write is gone, so it changed nothing durable to report.
    capped = [
        run
        for run in db.messages.get_prompt_log_runs()
        if run["run_outcome"] == RunOutcome.FAILED.value
    ]
    assert len(capped) == 1
    assert capped[0]["run_reason"] == "max steps exceeded — the program was left unfinished"


_TWO_WRITE_PROGRAM = (
    "Watch for new indie metroidvania releases.\n"
    '1. collection_write("indie-metroidvanias", entries=[{key: <the game>, '
    "content: <what is new>}])\n"
    '2. collection_write("indie-metroidvanias", entries=[{key: <the game>, '
    "content: <what is new>}])"
)


@pytest.mark.asyncio
async def test_a_write_gate_stop_keeps_what_the_cycle_wrote_before_it(
    mock_llm, test_config, tmp_path
):
    """A write-gate STOP is a HEALTHY end (#1936): an early clean no-news exit IS the
    cycle's end point, so whatever landed before it stays.

    The cycle files a genuinely new find, then re-observes the seeded entry unchanged
    and STOPs at the chokepoint without ever reaching a close.  Treating "no close" as
    "not finished" would throw the new find away and the collection would never
    accumulate anything on a quiet cycle.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, notify=False, extraction_prompt=_TWO_WRITE_PROGRAM)
    calls = 0

    def writes_then_stops(request: dict, count: int) -> LlmResponse:
        nonlocal calls
        calls += 1
        entry = (
            {"key": "Cinder Drift", "content": "Cinder Drift — out next spring."}
            if calls == 1
            else {"key": _NOTIFY_SEED_KEY, "content": _NOTIFY_SEED_CONTENT}
        )
        return mock_llm._make_tool_call_response(
            request, "collection_write", {"memory": "indie-metroidvanias", "entries": [entry]}
        )

    mock_llm.set_response_handler(writes_then_stops)
    await collector.run_for("indie-metroidvanias")

    assert _run_reason(db) == WRITE_GATE_STOP_REASONS[WriteGateOutcome.KEY_EXISTS_UNCHANGED]
    keys = {entry.key for entry in require_memory(db, "indie-metroidvanias").read_all()}
    assert keys == {_NOTIFY_SEED_KEY, "Cinder Drift"}
    assert _get(db, "indie-metroidvanias").last_collected_at is not None


@pytest.mark.asyncio
async def test_a_muted_delivery_still_keeps_the_cycles_writes(mock_llm, test_config, tmp_path):
    """A DELIBERATE decline is a healthy end (#1936): the user is muted, so the cycle
    did everything it could and a retry would reach the same answer.

    Its write therefore stands and its occurrence is spent — the opposite of the
    undrawable-message case, where the cycle never got as far as having something to
    decline to send.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)
    db.users.set_muted("+15551230000")
    mock_llm.set_response_handler(
        _program_handler(
            mock_llm,
            content="Cinder Drift — a new metroidvania. https://ex.example/cd",
            drawn=f"MESSAGE: {_DRAWN_MESSAGE}",
        )
    )

    await collector.run_for("indie-metroidvanias")

    assert db.send_queue.next_pending() is None
    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift")
    assert _run_reason(db) == NOTIFICATION_NOTES[NotificationOutcome.NOT_DELIVERABLE]
    assert _get(db, "indie-metroidvanias").last_collected_at is not None


@pytest.mark.asyncio
async def test_a_revert_that_cannot_land_still_settles_the_rest_of_the_cycle(
    mock_llm, test_config, tmp_path
):
    """The revert is a real SQLite write and the writer is single, so it can raise — and
    it runs in the cycle's ``finally``, where an escaping exception would skip the
    occurrence settlement and leave the collection due with NO attempt counted against
    it, re-picked on every tick, unbounded.

    So a failed revert is reported loudly and the cycle's writes are left standing —
    exactly the behaviour that stood before #1936 — while the occurrence still settles
    and the run record is still stamped.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, notify=False)

    def refuse_to_revert(*, revert: bool) -> bool:
        raise OperationalError("UPDATE memory_entry", {}, Exception("database is locked"))

    db.memories.end_cycle = refuse_to_revert  # ty: ignore[invalid-assignment]
    mock_llm.set_response_handler(
        _writes_then_dies(mock_llm, key="Cinder Drift", content="Cinder Drift — out now.")
    )

    success, _ = await collector.run_for("indie-metroidvanias")

    assert success is False
    # The write is still there (nothing was undone), and the cycle settled anyway.
    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift")
    assert _get(db, "indie-metroidvanias").last_collected_at is None  # occurrence kept
    assert collector._retry_attempts == {("indie-metroidvanias", CycleTrigger.ON_DEMAND): 1}
    assert _collector_run(db)["run_outcome"] == RunOutcome.FAILED.value


_LOG_DRIVEN_PROGRAM = (
    "Watch the release feed for new indie metroidvanias.\n"
    '1. log_read(memory="release-feed")\n'
    '2. collection_write("indie-metroidvanias", entries=[{key: <the game>, '
    "content: <what is new>}])"
)


def _reads_then_writes(mock_llm, *, close: bool):
    """A log-driven cycle: read the feed, write what it found, then either CLOSE or die.

    One handler for both halves of the cursor claim, so the two cycles differ in exactly
    the thing under test — how they ended."""
    calls = 0

    def handler(request: dict, count: int) -> LlmResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return mock_llm._make_tool_call_response(
                request, "log_read", {"memory": "release-feed"}
            )
        if calls == 2:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "indie-metroidvanias",
                    "entries": [{"key": "Cinder Drift", "content": "Cinder Drift — next spring."}],
                },
            )
        if close:
            return mock_llm._make_tool_call_response(request, DoneTool.name, {})
        raise LlmConnectionError("Connection refused")

    return handler


@pytest.mark.asyncio
async def test_a_reverted_cycles_read_cursor_stays_where_it_was(mock_llm, test_config, tmp_path):
    """A cursor and a reverted write are ONE claim about how much of the run was real
    (#1936), so they settle on the same verdict.

    A log-driven collection is dispatched when its input log is ahead of its cursor, so
    a cursor that advanced over a reverted cycle's input would leave the collection
    holding neither the write nor the input that produced it — the readiness gate would
    read the log as caught up and skip the retry outright.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db, notify=False, extraction_prompt=_LOG_DRIVEN_PROGRAM)
    db.memories.create_log("release-feed", "release announcements")
    require_memory(db, "release-feed").append(
        [LogEntryInput(content="Cinder Drift ships next spring", content_embedding=None)],
        author="feed",
    )

    mock_llm.set_response_handler(_reads_then_writes(mock_llm, close=False))
    await collector.run_for("indie-metroidvanias")

    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift") == []
    assert db.cursors.get("indie-metroidvanias", "release-feed") is None

    # The retry reads the SAME batch — its input is still there — and closes, which is
    # what moves both the write and the cursor.
    mock_llm.set_response_handler(_reads_then_writes(mock_llm, close=True))
    assert (await collector.run_for("indie-metroidvanias"))[0] is True

    assert require_memory(db, "indie-metroidvanias").get("Cinder Drift")
    assert db.cursors.get("indie-metroidvanias", "release-feed") is not None


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
    section = _normalise_stamps(section)
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
    per-collection body + whatever runtime rules this SURFACE can carry out + this
    collector's recent run history) and the bare user turn (empty for a background
    agent).  Date and run timestamps are normalised to placeholders; everything else is
    asserted char-for-char so the structure is visible and drift is caught.

    THE SCOPED SHAPE (#1911/#1916): this collection's program is one ``log_read``, so
    the cycle's surface is that ``log_read`` and the terminator — and every runtime rule
    names a tool it does not have, so the whole rules block is ABSENT rather than
    instructing calls the model cannot make.  A read-only routine reads a prompt about
    reading, ending on the close it does have.

    The two trailing state blocks sit in their fixed order (#1914): what the collection
    holds — nothing, here, said plainly — then what its recent runs did."""
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
    collector._bind(db.memories.get("board-games"))
    collector._install_tools(collector.get_tools())

    system_prompt = await collector._build_system_prompt(None)
    messages = collector._build_messages("", None, system_prompt)

    # ── System message: date + body + runtime-rules tail + run history ─────
    system_text = _normalise_stamps(
        re.sub(
            r"Current date and time: [^\n]*", "Current date and time: DATE", messages[0]["content"]
        )
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
        "## What this collection holds (newest first)\n"
        "It holds nothing yet.\n"
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


def test_cycle_result_classifies_worked_no_work_and_failed(test_config, tmp_path):
    """Structural outcome from the tool trace ALONE (#1569/#1911/#1916): whether the
    cycle CLOSED is a read of the ledger for a successful ``done`` record, and what it
    then records is decided by whether durable state changed.

    A closed cycle is ``worked``/``no_work`` with an EMPTY reason — a bare clean close
    has nothing to say the outcome enum doesn't — and an unclosed one is a ``failed``
    bail stamped with the reason it ended for.

    A done record that FAILED validation is not a close: the loop keeps going so the
    model can retry, so the run it leaves behind is unfinished like any other.

    ``landed`` is whether the cycle's entry writes survived (#1936).  A cycle that never
    reached a healthy end has its writes reverted, so whatever its trace holds it
    changed nothing durable — which is why every unclosed shape below is scored with
    ``landed=False``, the only combination they occur in.

    The collector here is bound to a one-call program, so the unreadable-program arm
    stays out of the way until the case that wants it."""
    collector, _ = _make_collector(test_config, tmp_path)
    collector._program = (ProgramCall(ordinal=1, tool="log_read"),)
    done = ToolCallRecord(tool=DoneTool.name, arguments={})

    # Closed after writing → worked, and the clean close carries no reason of its own.
    wrote_then_closed = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            done,
        ],
    )
    assert collector._cycle_result(wrote_then_closed, None, True) == (RunOutcome.WORKED, "")

    # Closed having changed nothing → no_work, same empty reason.
    read_then_closed = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="log_read", arguments={}), done],
    )
    assert collector._cycle_result(read_then_closed, None, True) == (RunOutcome.NO_WORK, "")

    # The one thing a clean close DOES carry: what telling the user came to.  A queued
    # notification is also work (#1914) — it is the cycle's second way of doing
    # something, and no tool call records it.
    assert collector._cycle_result(read_then_closed, NotificationOutcome.QUEUED, True) == (
        RunOutcome.WORKED,
        NOTIFICATION_NOTES[NotificationOutcome.QUEUED],
    )

    # Wrote and never closed → the write was REVERTED (#1936), so the record says
    # ``failed`` and not ``incomplete``: nothing durable is left of that write, and a
    # run record claiming a change that exists nowhere is the dishonest reading the
    # undo journal removes.
    unclosed_write = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, mutated=True)],
    )
    assert collector._cycle_result(unclosed_write, None, False) == (
        RunOutcome.FAILED,
        "cycle ended with the program unfinished",
    )

    # A done whose args failed validation is a recorded call, not a close — the run is
    # unfinished, exactly as if the model had never reached for the terminator.
    bad_done = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(tool="collection_write", arguments={}, mutated=True),
            ToolCallRecord(tool=DoneTool.name, arguments={}, failed=True),
        ],
    )
    assert collector._cycle_result(bad_done, None, False) == (
        RunOutcome.FAILED,
        "cycle ended with the program unfinished",
    )

    # Nothing changed (only a read/browse) → a real bail, and hitting the step cap is
    # distinguished from trailing off (the AGENT_MAX_STEPS sentinel).
    maxed = ControllerResponse(
        answer=PennyResponse.AGENT_MAX_STEPS,
        tool_calls=[ToolCallRecord(tool="browse", arguments={"queries": ["x"]})],
    )
    assert collector._cycle_result(maxed, None, False) == (
        RunOutcome.FAILED,
        "max steps exceeded — the program was left unfinished",
    )

    # The loop aborted on a failed model call → the abort's own structural facts are
    # the reason (#1909), instead of the generic line that made the whole class
    # diagnosable only by exclusion.  The aborted response carries no tool calls, so
    # the outcome stays the FAILED bail it always was.
    aborted = ControllerResponse(
        answer=PennyResponse.AGENT_MODEL_ERROR,
        abort=RunAbort(
            step=5,
            after_tool="read_similar",
            error=ModelCallError(error_class="LlmTimeoutError", message="Request timed out."),
        ),
    )
    assert collector._cycle_result(aborted, None, False) == (
        RunOutcome.FAILED,
        "model call failed at step 5 after read_similar: LlmTimeoutError: Request timed out.",
    )

    # A program the framework cannot READ gets its own line (#1911): its surface is the
    # terminator alone, so there was never a job it could carry out, and saying so is
    # the difference between a diagnosable state and one found by exclusion.
    collector._program = ()
    assert collector._cycle_result(unclosed_write, None, False) == (
        RunOutcome.FAILED,
        COLLECTOR_UNREADABLE_PROGRAM_REASON,
    )


def test_cycle_result_write_gate_stop_closes_cleanly(test_config, tmp_path):
    """A write-gate STOP (#1587) closes the cycle at the chokepoint with NO done():
    a watch's unchanged re-observation carries ``stop_reason`` on the write record —
    the outcome is a clean ``no_work`` (nothing changed) stamped with the declared
    stop reason, NOT a ``failed`` bail (the mislabel that would fire if the missing
    done() fell through to the no-``done()`` path).  A STOP that also changed durable
    state stays ``worked``.

    A STOP is a HEALTHY end (#1936) — the cycle's writes survive it — so every case here
    is scored with ``landed=True``, the only state a stopped cycle is ever in."""
    collector, _ = _make_collector(test_config, tmp_path)
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
    assert collector._cycle_result(unchanged_stop, None, True) == (RunOutcome.NO_WORK, reason)

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
    assert collector._cycle_result(stop_after_work, None, True) == (RunOutcome.WORKED, reason)

    # The reworded-key door onto the SAME no-news (#1919) closes the cycle identically
    # and stamps its OWN declared reason, so a run record says which door it came
    # through rather than collapsing both onto one phrase.
    already_reason = WRITE_GATE_STOP_REASONS[WriteGateOutcome.DUPLICATE_UNCHANGED]
    assert already_reason != reason
    already_stop = ControllerResponse(
        answer="",
        tool_calls=[
            ToolCallRecord(
                tool="collection_write",
                arguments={},
                mutated=False,
                stop_reason=WriteGateOutcome.DUPLICATE_UNCHANGED,
            )
        ],
    )
    assert collector._cycle_result(already_stop, None, True) == (
        RunOutcome.NO_WORK,
        already_reason,
    )


def test_should_stop_loop_honors_the_terminator_and_the_write_gate_stop(test_config, tmp_path):
    """The collector loop exits on a successful ``done`` record (#1569, restored #1916)
    or on a write-gate STOP record (#1587) — and on NOTHING ELSE.

    A plain write does NOT stop it, even one that carries out the whole of a one-call
    program: there is no coverage read any more, so what the model has executed is never
    by itself a close.  Nor does a ``done`` whose args failed validation — the loop keeps
    going so the model sees the error and can retry, rather than exiting on a
    recorded-but-empty close."""
    collector, _ = _make_collector(test_config, tmp_path)
    collector._program = (ProgramCall(ordinal=1, tool="collection_write"),)
    stop = ToolCallRecord(
        tool="collection_write", arguments={}, stop_reason=WriteGateOutcome.KEY_EXISTS_UNCHANGED
    )
    # Every member of the declared STOP table closes the loop — the hook reads the
    # record's stop_reason, so #1919's second member needed no loop change.
    already = ToolCallRecord(
        tool="collection_write", arguments={}, stop_reason=WriteGateOutcome.DUPLICATE_UNCHANGED
    )
    whole_program = ToolCallRecord(tool="collection_write", arguments={}, mutated=True)
    assert collector.should_stop_loop([stop]) is True
    assert collector.should_stop_loop([already]) is True
    assert collector.should_stop_loop([ToolCallRecord(tool=DoneTool.name, arguments={})]) is True
    assert collector.should_stop_loop([whole_program]) is False
    assert (
        collector.should_stop_loop([ToolCallRecord(tool=DoneTool.name, arguments={}, failed=True)])
        is False
    )
    assert collector.should_stop_loop([ToolCallRecord(tool="browse", arguments={})]) is False


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


@pytest.mark.asyncio
async def test_aborted_cycle_stamps_the_failed_call_on_the_run(mock_llm, test_config, tmp_path):
    """END TO END (#1909): a cycle whose model call dies MID-PROGRAM stamps the cause
    onto its run, so the sample DB alone says what happened.

    The measured failure class — 31 of 75 collector cycles — left no evidence at all:
    the failing call raises before the client persists, so it writes no promptlog row,
    and the run's reason said only ``cycle ended without a done() call``.  Here step 1's
    read lands (and logs its row), step 2 dies on the transport, and the reason stamped
    on that surviving row names the step, the tool the run had reached, and the error."""
    collector, db = _make_collector(test_config, tmp_path)
    _seed_notify_collection(db)

    def handler(request: dict, count: int) -> LlmResponse:
        if count == 1:
            return mock_llm._make_tool_call_response(
                request, "collection_read_latest", {"memory": "indie-metroidvanias"}
            )
        raise LlmConnectionError("Connection refused")

    mock_llm.set_response_handler(handler)

    await collector.run_for("indie-metroidvanias")

    runs = db.messages.get_prompt_log_runs()
    assert len(runs) == 1
    assert runs[0]["run_outcome"] == RunOutcome.FAILED.value
    assert runs[0]["run_reason"] == (
        "model call failed at step 2 after collection_read_latest: "
        "LlmConnectionError: Connection refused"
    )


# ── The occurrence is spent by a CYCLE, not by a dispatch (#1935) ────────────

# How far back a stamp has to be pushed for the daily rule below to come round again.
_A_DAY_AND_AN_HOUR = 25 * 60

_BRIEFING_PROGRAM = (
    "Write the morning briefing each day.\n"
    '1. collection_read_latest(memory="morning-briefing")\n'
    '2. collection_write("morning-briefing", entries=[{key: <the day>, '
    "content: <the briefing>}])"
)


def _daily_schedule(count: str = "") -> str:
    """A daily job whose fire came round an hour ago.  The rule's own DTSTART fixes the
    phase, so the first occurrence is an hour past (due now) and the next is a day out —
    which is exactly what consuming this one costs."""
    started = datetime.now(UTC) - timedelta(hours=1)
    return f"DTSTART:{started.strftime(_RRULE_STAMP)}\nFREQ=DAILY{count}"


def _seed_daily_briefing(
    db: Database,
    *,
    extraction_prompt: str = _BRIEFING_PROGRAM,
    schedule: str | None = None,
    max_runs: int | None = None,
) -> None:
    db.memories.create_collection(
        "morning-briefing",
        "The day's briefing",
        extraction_prompt=extraction_prompt,
        schedule=schedule or _daily_schedule(),
        max_runs=max_runs,
    )


def _cancel_the_cycle(request: dict, count: int) -> LlmResponse:
    """What a foreground message does to a cycle waiting on a model call: the
    scheduler's ``task.cancel()`` surfaces at the innermost await, which is this one."""
    raise asyncio.CancelledError


def _writes_then_closes(mock_llm) -> Callable[[dict, int], LlmResponse]:
    """A minimal working cycle — one write, then the close.

    It counts its OWN calls rather than reading the mock's ordinal, because a test that
    swaps handlers mid-way inherits a request count the earlier phase already advanced.
    """
    calls = 0

    def handler(request: dict, count: int) -> LlmResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return mock_llm._make_tool_call_response(
                request,
                "collection_write",
                {
                    "memory": "morning-briefing",
                    "entries": [{"key": "today", "content": "the ferries are running"}],
                },
            )
        return mock_llm._make_tool_call_response(request, DoneTool.name, {})

    return handler


@pytest.mark.asyncio
async def test_cancelled_cycle_leaves_the_days_occurrence_still_due(
    mock_llm, test_config, tmp_path
):
    """OBSERVED LIVE (#1935): a chat message cancelled a daily job's cycle seconds in —
    before any model call had come back — and the day's occurrence was BURNED.  The
    stamp advanced the schedule past a fire that never happened, so the briefing
    silently skipped a day.

    A cycle that did nothing spent nothing, so its occurrence stays due: the very next
    dispatcher pass picks the same collection up, and the attempt that actually runs is
    the one that consumes the fire.  The retry budget resets with it, so tomorrow's
    occurrence starts with its own.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db)
    mock_llm.set_response_handler(_cancel_the_cycle)

    with pytest.raises(asyncio.CancelledError):
        await collector.execute()

    assert _get(db, "morning-briefing").last_collected_at is None
    still_due = collector._next_ready_collection()
    assert still_due is not None and still_due.name == "morning-briefing"

    # The retry runs the job the day's fire was for — and THAT consumes the occurrence.
    mock_llm.set_response_handler(_writes_then_closes(mock_llm))
    assert await collector.execute() is True

    assert _get(db, "morning-briefing").last_collected_at is not None
    assert collector._next_ready_collection() is None  # waits for tomorrow
    assert collector._retry_attempts == {}
    assert require_memory(db, "morning-briefing").read_latest(5)[0].key == "today"


@pytest.mark.asyncio
async def test_retry_bound_consumes_the_occurrence_so_a_cancelled_collection_cant_hot_loop(
    mock_llm, test_config, tmp_path
):
    """The stamp's ORIGINAL job survives the retry (#1935): it exists so a collection
    that fails every time isn't re-attempted on every tick.

    A collection cancelled on every single attempt is re-dispatched a bounded number of
    times and then has its occurrence consumed anyway, so it drops out of the running
    and waits for its next fire.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db)
    mock_llm.set_response_handler(_cancel_the_cycle)

    for attempt in range(PennyConstants.COLLECTOR_RETRY_ATTEMPTS):
        with pytest.raises(asyncio.CancelledError):
            await collector.execute()
        assert _get(db, "morning-briefing").last_collected_at is None, attempt
        assert collector._next_ready_collection() is not None, attempt

    # The budget is spent — this attempt consumes the occurrence whatever happened.
    with pytest.raises(asyncio.CancelledError):
        await collector.execute()

    assert _get(db, "morning-briefing").last_collected_at is not None
    assert collector._next_ready_collection() is None
    assert collector._retry_attempts == {}


@pytest.mark.asyncio
async def test_a_deterministic_defect_consumes_the_occurrence_without_retrying(
    mock_llm, test_config, tmp_path
):
    """A retry is only worth an occurrence when the next attempt could go differently.

    A collection whose stored prompt names no runnable call fails identically every time
    it is dispatched — a CONFIG DEFECT, not a bad draw — so its occurrence is consumed
    on the first attempt.  Read off the stored program rather than off what the run did,
    the defect holds whichever way the cycle ended: on a failed model call (phase A) or
    preempted by a foreground message (phase B), which is what makes it outrank both
    stochastic causes.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db, extraction_prompt=_ONE_SHOT_PROMPT)  # prose, no numbered call

    def die_on_the_model_call(request: dict, count: int) -> LlmResponse:
        raise LlmConnectionError("Connection refused")

    mock_llm.set_response_handler(die_on_the_model_call)
    await collector.run_for("morning-briefing")

    assert _get(db, "morning-briefing").last_collected_at is not None
    assert collector._next_ready_collection() is None  # waits for tomorrow's fire

    _backdate_collected(db, "morning-briefing", minutes=_A_DAY_AND_AN_HOUR)
    assert collector._next_ready_collection() is not None  # the fire has come round
    mock_llm.set_response_handler(_cancel_the_cycle)
    with pytest.raises(asyncio.CancelledError):
        await collector.execute()

    # Consumed again — a cancellation buys an unrunnable collection nothing.
    assert collector._next_ready_collection() is None
    assert collector._retry_attempts == {}


@pytest.mark.asyncio
async def test_a_quiet_completed_cycle_still_consumes_its_occurrence(
    mock_llm, test_config, tmp_path
):
    """The over-correction guard: only a cycle that never got to RUN keeps its
    occurrence.

    A cycle that ran and found nothing worth writing — it read, then closed with
    ``done()``, the ordinary quiet cadence — did spend the fire it was dispatched for,
    so it stamps exactly as it always has and the collection waits for its next
    occurrence rather than being re-attempted on the next tick.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db)

    def reads_then_closes(request: dict, count: int) -> LlmResponse:
        if count == 1:
            return mock_llm._make_tool_call_response(
                request, "collection_read_latest", {"memory": "morning-briefing"}
            )
        return mock_llm._make_tool_call_response(request, DoneTool.name, {})

    mock_llm.set_response_handler(reads_then_closes)
    assert await collector.execute() is True

    assert _get(db, "morning-briefing").last_collected_at is not None
    assert collector._next_ready_collection() is None
    assert collector._retry_attempts == {}
    runs = db.messages.get_prompt_log_runs()
    assert runs[0]["run_outcome"] == RunOutcome.NO_WORK.value


@pytest.mark.asyncio
async def test_aborted_cycle_that_changed_nothing_keeps_its_occurrence(
    mock_llm, test_config, tmp_path
):
    """The other stochastic cause (#1909's ``RunAbort``): a model call that died on the
    transport is a bad draw, not a verdict on the collection — attempted again it may
    well come back — so a cycle it killed before anything landed keeps its occurrence
    and the run record still names the terminal cause.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db)

    def die_on_the_model_call(request: dict, count: int) -> LlmResponse:
        raise LlmConnectionError("Connection refused")

    mock_llm.set_response_handler(die_on_the_model_call)
    assert await collector.execute() is False

    assert _get(db, "morning-briefing").last_collected_at is None
    still_due = collector._next_ready_collection()
    assert still_due is not None and still_due.name == "morning-briefing"
    # The budget spent is the SCHEDULE's own (#1939) — this was a cadence dispatch.
    assert collector._retry_attempts == {("morning-briefing", CycleTrigger.CADENCE): 1}


@pytest.mark.asyncio
async def test_on_demand_attempts_do_not_spend_the_schedules_own_retry_budget(
    mock_llm, test_config, tmp_path
):
    """The user pressing "run this now" and the scheduler dispatching the day's fire are
    two different askers, so they carry two budgets (#1939).

    With one shared count, three failed clicks left the schedule's own attempt with
    nothing: the next foreground preemption consumed the occurrence on its first try
    and the job silently skipped its day — which is the exact failure the deferral
    exists to prevent, reached through the door the user was pressing.  Each budget
    still bounds one burst of the same size; once ANY cycle settles the fire, both
    reset, because what a count bounds is attempts at one occurrence.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db)

    def die_on_the_model_call(request: dict, count: int) -> LlmResponse:
        raise LlmConnectionError("Connection refused")

    mock_llm.set_response_handler(die_on_the_model_call)
    for _ in range(PennyConstants.COLLECTOR_RETRY_ATTEMPTS):
        success, _ = await collector.run_for("morning-briefing")
        assert success is False

    # The clicks spent the user's budget, and the day's fire is still due.
    assert collector._retry_attempts == {
        ("morning-briefing", CycleTrigger.ON_DEMAND): PennyConstants.COLLECTOR_RETRY_ATTEMPTS
    }
    assert _get(db, "morning-briefing").last_collected_at is None

    # The schedule's own attempt arrives with its budget untouched, so a preemption
    # still keeps the occurrence rather than burning the day.
    mock_llm.set_response_handler(_cancel_the_cycle)
    with pytest.raises(asyncio.CancelledError):
        await collector.execute()

    assert _get(db, "morning-briefing").last_collected_at is None
    assert collector._retry_attempts[("morning-briefing", CycleTrigger.CADENCE)] == 1

    # A cycle that spends the fire clears every asker's count with it.
    mock_llm.set_response_handler(_writes_then_closes(mock_llm))
    assert await collector.execute() is True
    assert _get(db, "morning-briefing").last_collected_at is not None
    assert collector._retry_attempts == {}


@pytest.mark.asyncio
async def test_a_kept_occurrence_does_not_burn_a_bounded_schedules_run(
    mock_llm, test_config, tmp_path
):
    """A cycle that keeps its occurrence must not spend a run of a bounded schedule
    either — the two are the same fire counted twice.

    An abort is not a cancellation, so it takes the outcome-tagging branch and its
    ``failed`` run DOES count toward ``max_runs``.  Without the skip a ``COUNT=1``
    one-shot that died on a transport wobble would archive itself while its occurrence
    sat due, and an archived collection is never dispatched again — so the retry the
    occurrence was kept for could never fire.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db, schedule=_daily_schedule(";COUNT=1"), max_runs=1)

    def die_on_the_model_call(request: dict, count: int) -> LlmResponse:
        raise LlmConnectionError("Connection refused")

    mock_llm.set_response_handler(die_on_the_model_call)
    assert await collector.execute() is False

    assert _get(db, "morning-briefing").archived is False
    still_due = collector._next_ready_collection()
    assert still_due is not None and still_due.name == "morning-briefing"

    # And the retry runs the one-shot it was kept for, which then retires it.
    mock_llm.set_response_handler(_writes_then_closes(mock_llm))
    assert await collector.execute() is True
    assert _get(db, "morning-briefing").archived is True


@pytest.mark.asyncio
async def test_a_kept_occurrence_still_raises_the_addon_refresh(mock_llm, test_config, tmp_path):
    """The stamp is the collector path's ONLY refresh signal — it is what wakes the
    addon surfaces' detail view — so a cycle that withholds it raises the change itself.

    Otherwise the on-demand trigger the addons expose ("run this now", which reaches
    ``run_for``) would leave the panel showing nothing when a cycle dies on its model
    call, with any entries it managed to write before dying invisible until something
    else touched the row.
    """
    collector, db = _make_collector(test_config, tmp_path)
    _seed_daily_briefing(db)
    refreshed: list[str | None] = []
    db.memories._on_memory_changed = lambda name: refreshed.append(name)

    def die_on_the_model_call(request: dict, count: int) -> LlmResponse:
        raise LlmConnectionError("Connection refused")

    mock_llm.set_response_handler(die_on_the_model_call)
    success, _ = await collector.run_for("morning-briefing")

    assert success is False
    assert _get(db, "morning-briefing").last_collected_at is None  # occurrence kept
    assert refreshed == ["morning-briefing"]


def test_tag_promptlog_run_with_unknown_run_id_is_noop(test_config, tmp_path):
    """If no promptlog rows exist for the run_id (cycle raised before the
    loop logged anything), tagging silently does nothing rather than
    crashing or smearing onto an unrelated row."""
    collector, db = _make_collector(test_config, tmp_path)

    collector._tag_promptlog_run("never-logged", RunOutcome.FAILED, "x", 0)

    assert db.messages.get_prompt_log_runs() == []


@pytest.mark.asyncio
async def test_run_for_collection_not_found(test_config, tmp_path):
    collector, _ = _make_collector(test_config, tmp_path)
    success, message = await collector.run_for("does-not-exist")
    assert success is False
    assert "does-not-exist" in message
    assert "not found" in message
    # matches the house memory-not-found wording (str(MemoryNotFoundError))
    assert "collection_set" in message


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
    assert "collection_set" in message


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
    """``run_for``'s on-demand message is STRUCTURAL (#1569): the run's stamped reason
    when it carries one (a write-gate stop reason, a notification note), else the
    outcome enum — plus the tool trace.  The terminator is argless (#1916), so there is
    still no model summary for it to relay.

    This cycle read, closed cleanly, and changed nothing, so its reason is EMPTY and the
    message falls back to the outcome enum — the same fallback the run record's own
    header takes."""
    from penny.agents.base import CycleResult

    collector, db = _make_collector(test_config, tmp_path)
    _seed_collector_runs_log(db)
    db.memories.create_collection(
        "test-col",
        "test",
        extraction_prompt='Extract things.\n1. log_read(memory="user-messages")',
    )

    async def mock_run_cycle(run_id: str) -> CycleResult:
        return CycleResult(
            success=True,
            response=ControllerResponse(
                answer="",
                tool_calls=[
                    ToolCallRecord(tool="log_read", arguments={"memory": "user-messages"}),
                    ToolCallRecord(tool=DoneTool.name, arguments={}),
                ],
            ),
        )

    collector._run_cycle = mock_run_cycle  # ty: ignore[invalid-assignment]

    success, message = await collector.run_for("test-col")
    assert success is True
    assert message.startswith(f"Collector cycle complete: {RunOutcome.NO_WORK.value}")
    assert "1. log_read(memory=user-messages)" in message


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
        target = collector._current_target
        assert target is not None
        observed["locked"] = collector._cycle_lock.locked()
        observed["target"] = target.name
        return CycleResult(success=True, response=ControllerResponse(answer="done"))

    collector._run_cycle = fake_run_cycle  # ty: ignore[invalid-assignment]
    success, _ = await collector.run_for("games")

    assert success is True
    assert observed["locked"] is True
    assert observed["target"] == "games"
    # Lock is released once the cycle finishes.
    assert collector._cycle_lock.locked() is False


# ── Cycle outcome: what counts as work ───────────────────────────────────────


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
    assert Collector._produced_work(_work_response(), None) is True
    assert Collector._produced_work(_idle_response(), None) is False
    assert Collector._produced_work(None, None) is False
    # A failed mutation isn't work.
    failed = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, failed=True)],
    )
    assert Collector._produced_work(failed, None) is False
    # The bug this fix targets: a duplicate-rejected write doesn't error
    # (``failed=False``) but changed nothing (``mutated=False``), so it must read
    # as no-work — a successful no-op changed nothing.
    duplicate = ControllerResponse(
        answer="",
        tool_calls=[ToolCallRecord(tool="collection_write", arguments={}, failed=False)],
    )
    assert Collector._produced_work(duplicate, None) is False
    # THE FOLD (#1914): the framework-queued notification is the cycle's SECOND way of
    # doing something, and since #1911 no tool call records it — so a cycle whose
    # routine only read, and which then told the user what it found, is work.  Only
    # QUEUED counts: the two failure outcomes put nothing in front of the user, so they
    # leave the cycle exactly as idle as its tool trace says it was.
    assert Collector._produced_work(_idle_response(), NotificationOutcome.QUEUED) is True
    assert Collector._produced_work(None, NotificationOutcome.QUEUED) is True
    assert Collector._produced_work(_idle_response(), NotificationOutcome.NOT_DRAWN) is False
    assert Collector._produced_work(_idle_response(), NotificationOutcome.NOT_DELIVERABLE) is False
    # A write still carries the cycle on its own when nothing was sent.
    assert Collector._produced_work(_work_response(), NotificationOutcome.NOT_DRAWN) is True


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
        schedule="FREQ=MINUTELY",
    )


async def test_log_driven_collection_skipped_until_its_log_advances(test_config, tmp_path):
    """A collection caught up on its only input log is skipped without entering
    the model; a new log entry makes it ready again — the second gate, after the
    schedule says it is due."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_log_driven_collection(db, log="chatter", prompt_names_log=True)

    # No cursor yet → not gate-eligible → runs (the first cycle establishes it).
    assert collector._next_ready_collection() is not None

    # Simulate a completed read: cursor sits at the head of the log.
    head = _memory(db, "chatter").read_batch(None, 10)[-1].created_at
    db.cursors.advance_committed("watcher", "chatter", head)
    db.memories.mark_collected("watcher")
    _backdate_collected(db, "watcher", minutes=10)  # its schedule has come round

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
    runs on its schedule alone."""
    collector, db = _make_collector(test_config, tmp_path)
    _make_log_driven_collection(db, log="chatter", prompt_names_log=False)

    # Leftover cursor for "chatter", which the prompt does NOT name; the log has
    # advanced far past it.
    db.cursors.advance_committed("watcher", "chatter", datetime.now(UTC) - timedelta(days=1))
    db.memories.mark_collected("watcher")
    _backdate_collected(db, "watcher", minutes=10)

    # Not gated on the stale cursor → runs on its schedule alone, and it's pruned.
    ready = collector._next_ready_collection()
    assert ready is not None and ready.name == "watcher"
    assert db.cursors.get("watcher", "chatter") is None


def test_input_pending_tristate(test_config, tmp_path):
    """The gate signal: None (no live cursor → the schedule alone decides), False
    (live cursor, caught up → skip), True (live cursor behind its log → run)."""
    collector, db = _make_collector(test_config, tmp_path)
    db.memories.create_log("chatter", "log")
    db.memories.create_collection(
        "watcher",
        "d",
        extraction_prompt='Extract via log_read("chatter").',
        schedule="FREQ=MINUTELY;INTERVAL=5",
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
