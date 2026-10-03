"""The clock seam (``penny/clock.py``) and the clock an eval sample runs on (#2207).

Deterministic — no live model.  Four things are pinned here:

- a running Penny reads ONE clock: the time it tells the model, the instant it counts a
  user's words about time from, and every row it writes come off the clock on its config;
- nothing else in the runtime reads the machine's clock, beyond the places listed here
  with what each one measures;
- the harness's ``PinnedClock`` starts where it is told and keeps time, so nothing it
  stamps collapses onto one instant;
- an end a case's ask names relative to the turn is resolved by production and judged by
  the case's claim on that same clock, so the answer is the same whenever the run is made.

The real clock's own path is pinned where it always was: the current-date line against the
machine's clock in ``tests/agents/test_collector.py`` and ``tests/agents/test_chat_agent.py``.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import DateTime, func, select
from sqlmodel import SQLModel

import penny as penny_package
from penny.agents.collector import Collector
from penny.clock import Clock
from penny.constants import ChannelType, PennyConstants, TransitionCause
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.skills import SkillDraft
from penny.datetime_utils import current_datetime_line
from penny.llm.client import LlmClient
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import _observe_sample, _real_model_config, seed_user
from penny.tests.eval.utils.assertions import _ends_when_asked
from penny.tests.eval.utils.clock import SAMPLE_STARTS_AT, SAMPLE_TIMEZONE, PinnedClock
from penny.tests.eval.utils.cohort import SampleObservation
from penny.tests.eval.utils.job_end import until_tonight_at
from penny.tests.eval.utils.worlds import World
from penny.tests.schema_template import migrated_db
from penny.tools.collection_instantiation import parse_schedule
from penny.tools.memory_tools import resolve_expiry


class _Ticks:
    """A seconds-counter that moves one millisecond every time it is read.

    Handed to a ``PinnedClock`` in place of the machine's monotonic counter, so what the
    clock reads is a function of how often it was asked and never of how long the test
    took — and two consecutive readings are never the same instant."""

    def __init__(self) -> None:
        self._reads = 0

    def __call__(self) -> float:
        self._reads += 1
        return self._reads / 1000


def _pinned_at(instant: datetime) -> PinnedClock:
    return PinnedClock(instant, elapsed=_Ticks())


# ── The pinned clock itself ───────────────────────────────────────────────────


def test_a_pinned_clock_starts_where_it_is_told_and_keeps_time() -> None:
    """It reads its starting instant when built and advances with the counter it is given,
    so it is a clock moved by a constant offset rather than a stopped one: later readings
    are later, by exactly what elapsed."""
    elapsed = [100.0]
    clock = PinnedClock(SAMPLE_STARTS_AT, elapsed=lambda: elapsed[0])
    assert clock.now() == SAMPLE_STARTS_AT
    assert clock.now().tzinfo is UTC, "the runtime's stamps are UTC, whatever zone the pin names"

    elapsed[0] += 90
    assert clock.now() == SAMPLE_STARTS_AT + timedelta(seconds=90)


def test_every_sample_starts_on_a_wednesday_afternoon_where_the_user_is(make_config) -> None:
    """The instant every eval sample starts at, read the way the model is shown it: on the
    wall clock of the profile ``seed_user`` writes.  And the one seam every runner builds its
    config through hands the sample that clock."""
    local = SAMPLE_STARTS_AT.astimezone(ZoneInfo(SAMPLE_TIMEZONE))
    assert local.strftime("%A %H:%M") == "Wednesday 14:00"

    config = _real_model_config(make_config, signal_api_url="http://localhost:1", db_path="unused")
    assert isinstance(config.clock, PinnedClock)
    started = config.clock.now()
    assert SAMPLE_STARTS_AT <= started < SAMPLE_STARTS_AT + timedelta(minutes=1)


def test_a_config_nobody_gave_a_clock_reads_the_machines(make_config) -> None:
    """Production builds its config with no clock, and gets the real one."""
    clock = make_config().clock
    assert type(clock) is Clock
    before = datetime.now(UTC)
    assert before <= clock.now() <= datetime.now(UTC)


# ── A running Penny reads one clock ───────────────────────────────────────────

# A late evening, years from any day this test will run on, so an instant read off the
# machine's clock and one read off the pinned clock can never be mistaken for each other.
_LATE_EVENING = datetime(2031, 3, 5, 23, 30, tzinfo=ZoneInfo(SAMPLE_TIMEZONE))


def _read_off_the_machines_clock(db: Database, began: datetime) -> list[str]:
    """Every datetime column holding a value from the test's own real-time window.

    The database was handed a clock years away, so a value inside the window the test ran
    in was read off the machine's clock by a write that went round the seam."""
    ended = datetime.now(UTC)
    found: list[str] = []
    with db.engine.connect() as connection:
        for table in SQLModel.metadata.sorted_tables:
            for column in table.columns:
                if not isinstance(column.type, DateTime):
                    continue
                count = connection.execute(
                    select(func.count()).select_from(table).where(column.between(began, ended))
                ).scalar_one()
                if count:
                    found.append(f"{table.name}.{column.name} ({count} rows)")
    return found


def _write_through_every_store(db: Database) -> None:
    """One write down each store path that stamps a time, the inserts and the updates both —
    the paths a single chat turn does not reach on its own."""
    db.memories.create_collection("playlists", "favorite playlists", schedule="FREQ=HOURLY")
    require_memory(db, "playlists").write(
        [EntryInput(key="morning", content="prog rock")], author="user"
    )
    db.memories.create_log("tips", "useful tips")
    require_memory(db, "tips").append([LogEntryInput(content="tune before playing")], author="user")
    db.memories.update_collection_metadata("playlists", description="playlists worth keeping")
    db.memories.mark_collected("playlists")
    queued = db.send_queue.enqueue("a queued message", "playlists")
    db.send_queue.mark_sent(queued)
    db.send_queue.enqueue("one that never goes out", "playlists")
    db.memories.archive("playlists")
    db.cursors.advance_committed("reader", "tips", db.clock.now())
    db.cursors.set_position("reader", "tips", db.clock.now())
    for permission in ("allowed", "blocked"):
        db.domain_permissions.set_permission("example.com", permission)
    rule = db.email_rules.create(PennyConstants.PROVIDER_ZOHO, "a rule", "{}", "{}")
    db.email_rules.mark_applied(rule.id or 0)
    db.media.put(b"bytes", "image/png")
    db.skills.upsert(
        SkillDraft(
            name="a routine", intent="i", description="d", steps=[], parameters=[], source_run_id=""
        ),
        author="chat",
    )
    for name in ("Someone", "Someone Else"):
        db.users.save_info("+15550009999", name, "Seattle, WA", SAMPLE_TIMEZONE, "1990-01-01")
    db.users.set_muted("+15550009999")
    device = db.devices.register(ChannelType.IOS, "an-ios-device", "A phone")
    db.ios.upsert_registration(
        device=device, apns_token="token", apns_environment="sandbox", app_version="1"
    )
    item = db.ios.enqueue_outbox(
        device_id=device.id or 0,
        content="a message",
        attachments=None,
        source_type=None,
        source_name=None,
        source_hint=None,
        push_title="t",
        push_summary="s",
    )
    db.ios.mark_push_sent(item.id or 0)
    db.ios.mark_acked(device.id or 0, [item.id or 0])


@pytest.mark.asyncio
async def test_a_running_penny_tells_and_records_the_time_on_its_configs_clock(
    signal_server, mock_llm, make_config, test_user_info, running_penny
) -> None:
    """One turn through the real channel, agent, tools and stores, on a clock years away.

    The model is told that clock's time, every row the turn leaves behind carries it, and
    the rows are still in the order they were written."""
    began = datetime.now(UTC)
    config = make_config(clock=_pinned_at(_LATE_EVENING))
    mock_llm.set_default_flow(
        final_response="here's what i found!",
        search_query="https://weather.example.com/today",
    )

    async with running_penny(config) as penny:
        db = penny.db
        await signal_server.push_message(sender=TEST_SENDER, content="/config idle_seconds 600")
        await signal_server.wait_for_message(timeout=10.0)
        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather today?")
        await signal_server.wait_for_message(timeout=10.0)

        system_text = next(
            message["content"]
            for message in mock_llm.requests[0]["messages"]
            if message["role"] == "system"
        )
        assert system_text.split("\n", 1)[0] == (
            "Current date and time: Wednesday, March 05, 2031 at 11:30 PM PST"
        )

        asked = db.messages.get_user_messages(TEST_SENDER)[-1]
        answered = require_memory(db, PennyConstants.MEMORY_PENNY_MESSAGES_LOG).newest_entries(k=1)
        assert answered[0].created_at > asked.timestamp, "the reply is stamped after its ask"
        stamps = [row.timestamp for row in reversed(db.messages.recent_prompts(50))]
        assert len(stamps) > 1 and stamps == sorted(set(stamps)), (
            "each model call is stamped after the one before it"
        )

        _write_through_every_store(db)
        assert _read_off_the_machines_clock(db, began) == []

        # And that read can see a miss: one row written through a database nobody handed a
        # clock is stamped off the machine's, and is found.
        Database(config.db_path).users.set_muted("someone else")
        assert _read_off_the_machines_clock(db, began) == ["mutestate.muted_at (1 rows)"]


# Every place the runtime reads the MACHINE's clock, with how many reads it makes and what
# they measure.  None of them is Penny's sense of the time: each times the process, or
# talks to a system that keeps a clock of its own.
_ON_THE_MACHINES_CLOCK = {
    # The seam itself: the real clock's one reading.
    "clock.py": 1,
    # How long the process has been up.
    "penny.py": 1,
    # The models' own defaults — the fallback for a row built outside a store.
    "database/models.py": 25,
    # When a migration was applied, in the runner's own bookkeeping table.
    "database/migrate.py": 1,
    # How long a model call took.
    "llm/client.py": 2,
    # When an OAuth token the service issued runs out.
    "plugins/zoho/base_client.py": 1,
    "plugins/zoho/mail_client.py": 1,
    # The window asked of a remote calendar, which lives on the real clock.
    "plugins/zoho/calendar_tools.py": 3,
    # When a push token was minted, as the push service counts it.
    "channels/ios/apns.py": 1,
    # How long ago a browser socket last made itself heard.
    "channels/browser/channel.py": 4,
}

# The calls that read a wall clock: ``datetime.now`` / ``datetime.utcnow`` / ``date.today``
# and ``time.time``.  A monotonic counter measures an interval and names no time.
_WALL_CLOCK_READS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("date", "today"),
    ("time", "time"),
}


def _wall_clock_reads(source: str) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr) in _WALL_CLOCK_READS
        for node in ast.walk(ast.parse(source))
    )


def test_nothing_else_in_the_runtime_reads_the_machines_clock() -> None:
    """The seam holds only while nothing goes round it, and a read that does is invisible in
    production — both clocks agree there.  So the runtime's own source is read: every module
    that asks the machine what time it is, and how often, is the list above and nothing else.

    A new read belongs on ``db.clock``; one that genuinely measures the process is added to
    the list with what it measures."""
    package = Path(penny_package.__file__).parent
    found = {}
    for path in sorted(package.rglob("*.py")):
        module = path.relative_to(package).as_posix()
        if module.startswith(("tests/", "database/migrations/")):
            continue
        reads = _wall_clock_reads(path.read_text())
        if reads:
            found[module] = reads
    assert found == _ON_THE_MACHINES_CLOCK


def test_the_dispatcher_reads_readiness_on_the_databases_clock(test_config, tmp_path) -> None:
    """A job that ends an hour from now on the database's clock is still live there, however
    long ago that hour passed on the machine's — so the dispatcher picks it up."""
    long_ago = datetime(2020, 1, 15, 14, 0, tzinfo=UTC)
    db = migrated_db(str(tmp_path / "t.db"), clock=_pinned_at(long_ago))
    client = LlmClient(
        api_url="http://localhost:11434", model="test-model", max_retries=1, retry_delay=0.0
    )
    collector = Collector(
        model_client=client, db=db, config=test_config, embedding_model_client=client
    )
    db.memories.create_collection(
        "hourly",
        "an hourly job",
        extraction_prompt="Browse the web for a daily fact and write one entry each cycle.",
        schedule="FREQ=HOURLY",
        expires_at=db.clock.now() + timedelta(hours=1),
    )

    target = collector._next_ready_collection()
    assert target is not None and target.name == "hourly"
    assert collector._archive_if_expired(target, run_id=None) is False


# ── An end named relative to the turn ─────────────────────────────────────────

_JOB = "a-job"
_NO_PAGES = World(name="no pages", pages=(), keeps=(), excludes=())
_UNTIL_TEN_TONIGHT = _ends_when_asked(_JOB, until_tonight_at(22))


def _a_turn_that_stood_a_job_up(
    tmp_path, name: str, starts_at: datetime, *, rule: str, ends: str | None
) -> tuple[Database, SampleObservation]:
    """One apply turn on a sample's clock, read back through the chat observer.

    The classifier draws, the machine lands in apply, and the job is stood up on ``rule``
    with the end ``ends`` says — resolved by the function ``collection_set`` resolves it
    with, against the same database, so the instant the words are counted from is the one
    production counts them from."""
    db = migrated_db(str(tmp_path / f"{name}.db"), clock=_pinned_at(starts_at))
    seed_user(db)
    _log_draw(db, PennyConstants.STATE_CLASSIFIER_AGENT_NAME, "STATE: apply")
    db.machine.record_transition(
        from_state=ConversationState.LEARN.value,
        to_state=ConversationState.APPLY.value,
        cause=TransitionCause.CLASSIFIER,
    )
    schedule = parse_schedule(rule)
    db.memories.create_collection(
        _JOB,
        "a job",
        extraction_prompt="Browse the web for a daily fact and write one entry each cycle.",
        schedule=schedule.rule,
        max_runs=schedule.max_runs,
        expires_at=resolve_expiry(db, schedule, ends),
        notify=True,
    )
    _log_draw(db, PennyConstants.CHAT_AGENT_NAME, "done")
    sample = _observe_sample(
        db,
        name=name,
        phrasing="the ask",
        arm=0,
        reply="done",
        before=set(),
        held_before=[],
    )
    return db, sample


def _log_draw(db: Database, agent_name: str, text: str) -> None:
    db.messages.log_prompt(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        response={"choices": [{"message": {"content": text}}]},
        agent_name=agent_name,
        run_id=agent_name,
    )


def test_an_end_the_ask_names_from_the_turn_is_judged_on_the_samples_clock(tmp_path) -> None:
    """An end like "until 10pm tonight" is resolved by production and judged by the case's
    claim on ONE clock — the sample's — so the answer does not depend on when the run is made
    (#2207).

    **What "tonight" means.**  The evening of the calendar day the sample's clock reads, at
    whatever hour that clock reads it: production counts "10pm today" from the sample's own
    wall clock and the claim's window is 22:00 on the day of the turn, which it reads off the
    move the turn recorded — the same clock twice.

    On the suite's own clock the sample starts at 14:00, so that evening is ahead and a job
    ending then holds — by its expiry, or by a count whose firings are walked from the job's
    creation moment, which is on that clock too.

    At 23:30 the same words name an hour already gone.  Neither side rolls it to tomorrow:
    production stores 22:00 on that date, the claim expects 22:00 on that date, and the claim
    reports the end as one that fell before the turn.  That is the verdict a run made after
    22:00 used to reach off the machine's clock; it is now reachable only by a sample whose
    clock is set that late."""
    db, afternoon = _a_turn_that_stood_a_job_up(
        tmp_path, "afternoon", SAMPLE_STARTS_AT, rule="FREQ=HOURLY", ends="10pm today"
    )
    assert current_datetime_line(db) == (
        "Current date and time: Wednesday, October 14, 2026 at 02:00 PM PDT"
    )
    assert _UNTIL_TEN_TONIGHT(afternoon, _NO_PAGES) == (
        True,
        "the job ends Wed 2026-10-14 22:00 PDT (by its expiry); the ask says until 22:00 "
        "tonight, which is Wed 2026-10-14 22:00 PDT",
    )

    _, counted = _a_turn_that_stood_a_job_up(
        tmp_path, "counted", SAMPLE_STARTS_AT, rule="FREQ=HOURLY;COUNT=8", ends=None
    )
    assert _UNTIL_TEN_TONIGHT(counted, _NO_PAGES) == (
        True,
        "the job ends Wed 2026-10-14 21:00 PDT (after 8 runs of 'FREQ=HOURLY;COUNT=8'); the "
        "ask says until 22:00 tonight, which is Wed 2026-10-14 22:00 PDT",
    )

    late_evening = SAMPLE_STARTS_AT.replace(hour=23, minute=30)
    db, late = _a_turn_that_stood_a_job_up(
        tmp_path, "late", late_evening, rule="FREQ=HOURLY", ends="10pm today"
    )
    assert current_datetime_line(db) == (
        "Current date and time: Wednesday, October 14, 2026 at 11:30 PM PDT"
    )
    assert _UNTIL_TEN_TONIGHT(late, _NO_PAGES) == (
        False,
        "the job ends Wed 2026-10-14 22:00 PDT (by its expiry); the ask says until 22:00 "
        "tonight, which is Wed 2026-10-14 22:00 PDT; that end is before the turn ran at "
        "Wed 2026-10-14 23:30 PDT",
    )
