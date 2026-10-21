"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import BeforeValidator
from sqlmodel import Session, select

from penny.agents.models import ToolCallRecord
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import (
    WriteGateOutcome,
    WriteResult,
    render_run_calls,
)
from penny.database.memory.identity import EntryDecider
from penny.database.models import MemoryEntry, MemoryRow, MessageLog
from penny.database.mutation_store import mutation_change_summary
from penny.datetime_utils import format_log_timestamp
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.llm.decision_client import DecisionAnswer, DecisionClient
from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmConnectionError
from penny.llm.similarity import embed_text
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tests.schema_template import schema_only_db
from penny.tools.collection_instantiation import (
    SCHEDULE_EXAMPLES,
    ScheduleError,
    describe_schedule,
    describe_schedule_field,
    next_occurrence,
    parse_expires_at,
    parse_schedule,
    render_schedule_clause,
)
from penny.tools.job_summary import job_summary_line
from penny.tools.memory_args import (
    OPTIONAL_ARG_FORMS,
    CollectionCreateArgs,
    CollectionSetArgs,
    CollectionUpdateArgs,
    _reject_blank_optional,
)
from penny.tools.memory_tools import (
    _INERT_JOB_ARGS_REFUSAL,
    _NO_SCHEDULE_NOTE,
    _RULES_GUARD_REFUSAL,
    _SAME_JOB_UPDATED,
    EXPIRES_AT_PARAM_DESCRIPTION,
    SCHEDULE_PARAM_DESCRIPTION,
    CollectionArchiveTool,
    CollectionCreateTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionListTool,
    CollectionRemoveTool,
    CollectionSampleTool,
    CollectionSetTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    FindTool,
    GetEventTool,
    LogAppendTool,
    LogCreateTool,
    LogGetTool,
    LogReadTool,
    MemoryMetadataTool,
    ReadRunCallsTool,
    _format_unsure,
    _format_updated,
    build_memory_tools,
    format_entries,
    render_nothing_landed,
    render_writes_landed,
)

# The named degraded mode for tool tests: rules settle what they can, the rest
# is written as new with the honest question.
_NO_MODEL = EntryDecider(None)


class _StubDecisionClient(DecisionClient):
    """A decision model that answers each question with the next canned choice."""

    def __init__(self, *choices: str) -> None:
        self._choices = list(choices)

    async def decide(self, **_: object) -> DecisionAnswer:
        return DecisionAnswer(choice=self._choices.pop(0))


def _decider_saying(*choices: str) -> EntryDecider:
    return EntryDecider(_StubDecisionClient(*choices))


pytestmark = pytest.mark.bare_db


def _make_llm_client(mock_llm) -> LlmClient:
    """Build an LlmClient whose default embed handler returns distinct vectors
    per input text, so identical inputs collide and distinct inputs don't."""
    mock_llm.set_embed_handler(_hash_embed)
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _hash_embed(model: str, text: str | list[str]) -> list[list[float]]:
    """Deterministic embedding: text → unit vector where one axis is 1.0.

    Identical strings map to identical vectors; distinct strings map to
    different axes (cosine = 0), so dedup and similarity behave sensibly in
    tests without depending on a real embedding model.
    """
    inputs = text if isinstance(text, list) else [text]
    return [_single_hash_vec(t) for t in inputs]


def _single_hash_vec(text: str, dim: int = 4096) -> list[float]:
    """Bag-of-words deterministic embedding.  Each word picks an axis via
    SHA-256 → modulo ``dim``; the vector is L2-normalised so cosine is
    comparable across strings.  Identical strings map to identical
    vectors; strings sharing words have meaningful cosine > 0;
    fully-distinct strings map to cosine = 0."""
    vec = [0.0] * dim
    words = text.lower().split() or [text]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:8], "big") % dim
        vec[axis] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class _FailingEmbedClient:
    """An embedding client whose every embed call fails transiently.

    ``embed_text`` catches ``LlmError`` and returns ``None``, so the write path
    hits the fail-hard branch and refuses to persist a vectorless entry (#1412).
    """

    model = "test-model"

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        raise LlmConnectionError("embedding backend unavailable")


class _KeyOnlyFailingEmbedClient:
    """Fails only the embed of a specific string (the key), returning a real
    vector for everything else (the content).

    ``CollectionWriteTool._build_entry`` embeds key then content in two calls, so
    this reproduces a transient miss that lands on the key alone — the entry would
    be stored missing its key vector, which the write path refuses (an entry must
    carry all its vectors).  The startup backfill is the safety net for a
    key-null row that reaches the corpus by another path (migration-seeded content,
    #1468), but the write path still refuses at write time rather than persist a
    vectorless, recall-invisible row that only a later restart would repair.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        self.model = "test-model"

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        inputs = text if isinstance(text, list) else [text]
        if self._key in inputs:
            raise LlmConnectionError("key embed failed")
        return [_single_hash_vec(t) for t in inputs]


# ── Seeding helpers ───────────────────────────────────────────────────────────
#
# ``collection_configure`` takes a free-text ``prompt`` and refuses a near-duplicate
# (#2455 naive baseline). Tests that only need a collection to EXIST (to exercise
# writes/reads/mutations/etc.) seed it directly through the store
# (``_seed_collection``) — the honest, idempotency-free way to stand one up — with a
# deterministic description anchor so dedup/similarity behave.


def _seed_collection(
    db,
    *,
    name: str,
    description: str = "x",
    extraction_prompt: str | None = "test fixture extraction prompt",
    schedule: str | None = "FREQ=HOURLY",
    notify: bool = False,
    archived: bool = False,
) -> MemoryRow:
    """Stand up a collection through the store (no tool, no idempotency check) so a
    test that just needs one to exist doesn't drive the whole create front door."""
    return db.memories.create_collection(
        name,
        description,
        archived=archived,
        extraction_prompt=extraction_prompt,
        schedule=schedule,
        description_embedding=_single_hash_vec(description),
        notify=notify,
    )


class TestMinimalSurfaceCensus:
    """The minimal collection surface (#1631, tightened by #1570, collapsed to a
    free-text ``prompt`` by #2455): the model reasons about six concepts — name ·
    description · prompt · schedule · notify · expires_at.  ``create_anyway`` and a
    diffed ``extraction_prompt`` edit are GONE (dedup always runs; a job's
    instructions are replaced wholesale, never patched).  Pinned so a future
    field can't silently creep back onto the surface."""

    def test_create_accepts_exactly_the_minimal_fields(self):
        assert set(CollectionCreateArgs.model_fields) == {
            "name",
            "description",
            "prompt",
            "schedule",
            "notify",
            "expires_at",
        }

    def test_update_accepts_exactly_the_minimal_fields(self):
        assert set(CollectionUpdateArgs.model_fields) == {
            "name",
            "description",
            "prompt",
            "schedule",
            "notify",
            "expires_at",
        }

    def test_dropped_shrapnel_is_gone_from_both_surfaces(self):
        # ``intent`` and every member of the retired trigger union (#1631/#1857).
        shrapnel = {
            "intent",
            "interval",
            "run_at",
            "max_runs",
            "on_advance",
            "min_interval",
            "trigger",
            "cron_expression",
            "source_log",
        }
        assert shrapnel.isdisjoint(CollectionCreateArgs.model_fields)
        assert shrapnel.isdisjoint(CollectionUpdateArgs.model_fields)

    @pytest.mark.asyncio
    async def test_an_extraction_prompt_argument_is_refused_actionably(self, db):
        """The registered front door REFUSES a ``extraction_prompt`` argument by its
        OLD name, and says what to pass instead (#1919, replacing the retired live
        contract) — the model surface's own instructions field is ``prompt``
        (#2455 naive baseline), written straight into ``extraction_prompt`` with no
        rendering seam, so the stored column's own name is never itself a valid
        argument.  ``ToolArgs``' ``extra="forbid"`` refuses the call at ARG
        VALIDATION before any routing.

        Driven through ``run`` rather than ``execute`` because the actionable envelope is
        what the model reads: the refusal has to NAME the unknown parameter and list the
        ones that exist, or it is a dead end rather than a correction."""
        result = await CollectionSetTool(db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(),
            collection="board-games",
            extraction_prompt="1. extract_text(url)\n2. done()",
        )
        assert result.success is False
        assert "unknown parameter 'extraction_prompt'" in result.message
        # The valid parameters are listed, so the retry is a correction and not a guess.
        assert "prompt" in result.message and "schedule" in result.message
        # And nothing was created off a refused call.
        assert db.memories.get("board-games") is None


class TestScheduleGrammar:
    """The ``schedule`` grammar as a table (#1857): what ``parse_schedule`` accepts, what
    it lifts, and what it refuses.  Pure functions with many edge cases — the one place
    the project's testing convention allows a unit table rather than a flow test."""

    @pytest.mark.parametrize(
        "schedule, rule, max_runs, expires_at",
        [
            # The four shapes the tool description teaches, each stored verbatim.
            ("FREQ=HOURLY", "FREQ=HOURLY", None, None),
            ("FREQ=MINUTELY;INTERVAL=90", "FREQ=MINUTELY;INTERVAL=90", None, None),
            ("FREQ=DAILY;BYHOUR=8", "FREQ=DAILY;BYHOUR=8", None, None),
            (
                "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9",
                "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9",
                None,
                None,
            ),
            # An RRULE: prefix and lower case are the same rule — dateutil reads both,
            # so neither is a rejection the model has to recover from.
            ("RRULE:FREQ=HOURLY", "RRULE:FREQ=HOURLY", None, None),
            ("freq=hourly", "freq=hourly", None, None),
            # Surrounding whitespace is trimmed, never treated as part of the rule.
            ("  FREQ=HOURLY  ", "FREQ=HOURLY", None, None),
            # COUNT lifts into max_runs and STAYS in the rule, so a rendered schedule
            # re-passed lifts the same quota.  UNTIL has no accepted row here at all —
            # it is refused outright (#2511, see test_refused_schedules_teach_actionably).
            ("FREQ=DAILY;COUNT=1", "FREQ=DAILY;COUNT=1", 1, None),
            ("FREQ=DAILY;COUNT=3", "FREQ=DAILY;COUNT=3", 3, None),
            # A DTSTART line, written across two lines OR on one with the \n escape:
            # the same schedule, stored the same way, so the render round-trips.
            (
                "DTSTART:20260720T090000Z\nFREQ=DAILY;COUNT=1",
                "DTSTART:20260720T090000Z\nFREQ=DAILY;COUNT=1",
                1,
                None,
            ),
            (
                "DTSTART:20260720T090000Z\\nFREQ=DAILY;COUNT=1",
                "DTSTART:20260720T090000Z\nFREQ=DAILY;COUNT=1",
                1,
                None,
            ),
        ],
    )
    def test_accepted_schedules_store_verbatim_and_lift_their_end_conditions(
        self, schedule, rule, max_runs, expires_at
    ):
        parsed = parse_schedule(schedule)
        assert (parsed.rule, parsed.max_runs, parsed.expires_at) == (rule, max_runs, expires_at)

    @pytest.mark.parametrize(
        "schedule, expected_lead",
        [
            # The retired trigger forms are not a second schedule language.
            ("every 3600", "I couldn't read the schedule 'every 3600'."),
            ("cron 0 8,20 * * *", "I couldn't read the schedule 'cron 0 8,20 * * *'."),
            ("once at 2026-12-25T09:00:00Z", "I couldn't read the schedule 'once at"),
            ("on advance of events-log", "I couldn't read the schedule 'on advance of"),
            # Plain prose, and a rule part dateutil rejects.
            ("whenever I feel like it", "I couldn't read the schedule 'whenever I feel"),
            ("FREQ=BANANA", "I couldn't read the schedule 'FREQ=BANANA'."),
            # A bare-date UNTIL: dateutil refuses it against a timezone-aware
            # start, which every stored rule has, so the whole rule is refused.
            (
                "FREQ=DAILY;UNTIL=20260901",
                "I couldn't read the schedule 'FREQ=DAILY;UNTIL=20260901'.",
            ),
            # A repairable shape leads with the corrected string, so the retry is a copy.
            ("'FREQ=HOURLY'", "I couldn't read the schedule ''FREQ=HOURLY'' — did you mean"),
            ("FREQ=HOURLY;", "I couldn't read the schedule 'FREQ=HOURLY;' — did you mean"),
            (
                "DTSTART:20260720T090000Z;FREQ=DAILY",
                "I couldn't read the schedule 'DTSTART:20260720T090000Z;FREQ=DAILY' — did "
                "you mean 'DTSTART:20260720T090000Z\\nFREQ=DAILY'?",
            ),
            # Two rules would give two answers to the lift.
            ("FREQ=DAILY\\nFREQ=WEEKLY", "The schedule 'FREQ=DAILY\\nFREQ=WEEKLY' has more"),
            # A quota that isn't a quota.
            ("FREQ=DAILY;COUNT=0", "The schedule 'FREQ=DAILY;COUNT=0' has COUNT=0"),
            # A job's end is said ONE way (#2511) — a schedule's own UNTIL= is refused
            # outright, naming expires_at as where the end goes instead.
            (
                "FREQ=DAILY;UNTIL=20260901T000000Z",
                "The schedule 'FREQ=DAILY;UNTIL=20260901T000000Z' sets UNTIL=20260901T000000Z —",
            ),
        ],
    )
    def test_refused_schedules_teach_actionably(self, schedule, expected_lead):
        with pytest.raises(ScheduleError) as raised:
            parse_schedule(schedule)
        assert str(raised.value).startswith(expected_lead)

    def test_a_stored_untils_read_path_is_unaffected(self):
        """The refusal guards the WRITE boundary only (#2511) — a row whose schedule
        already carries UNTIL= from before the refusal existed still parses and fires
        on it unchanged everywhere a stored rule is read, never re-validated through
        ``parse_schedule``."""
        schedule = "FREQ=DAILY;UNTIL=20261019T000000Z"
        anchor = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
        assert next_occurrence(schedule, anchor, None) == datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
        assert describe_schedule(_row_with_schedule(schedule)) == f"every day (rule: `{schedule}`)"

    def test_the_taught_examples_say_whose_clock_each_time_is_on(self):
        """The grammar the model is taught, asserted WHOLE (#1932).

        Every other assertion on a schedule rejection interpolates ``SCHEDULE_EXAMPLES``
        from the very source it is checking, so the examples are the one model-facing
        string nothing pinned — and they used to say a stated hour was UTC, which the
        readiness gate no longer makes true.  The two clocks in the grammar are opposite
        and both are named: an HOUR is the user's, a ``Z`` stamp is an exact UTC moment,
        so a rule that mixes them cannot be read as stating one clock throughout."""
        assert SCHEDULE_EXAMPLES == (
            "- FREQ=HOURLY — every hour\n"
            "- FREQ=MINUTELY;INTERVAL=90 — every 90 minutes\n"
            "- FREQ=DAILY;BYHOUR=8 — once a day at 8 in the morning, the user's own clock\n"
            "- FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9 — weekdays at 9 in the morning, "
            "the user's own clock\n"
            "- DTSTART:20260720T090000Z\\nFREQ=DAILY;COUNT=1 — once, at that exact moment "
            "in UTC, then it retires"
        )

    def test_did_you_mean_carries_a_string_that_actually_parses(self):
        """The corrected string is a COPY, not a suggestion to re-derive: whatever the
        did-you-mean echoes must itself be accepted, or the retry loops."""
        with pytest.raises(ScheduleError) as raised:
            parse_schedule("'FREQ=HOURLY;'")
        corrected = str(raised.value).split("did you mean '")[1].split("'?")[0]
        assert parse_schedule(corrected).rule == "FREQ=HOURLY"


class TestExpiresAtGrammar:
    """``expires_at`` takes an exact time OR the user's own words, read in the USER's
    timezone (#1857).  The table is the measured register: what lands, what it lands on,
    and what is honestly refused."""

    # A fixed 'now' so relative words resolve deterministically: 20:00 UTC = 16:00 in
    # the user's zone, the wall clock their words are counted from.
    _NOW = datetime(2026, 8, 9, 20, 0, tzinfo=UTC)
    _ZONE = "America/Toronto"

    @pytest.mark.parametrize(
        "value, expected",
        [
            # ISO carrying its OWN zone (Z or an explicit offset) is read exactly as
            # written, whatever zone the user is in (#2206) — an explicit offset is
            # never re-read on the user's clock.
            ("2026-09-01T09:00:00Z", datetime(2026, 9, 1, 9, 0, tzinfo=UTC)),
            ("2026-09-01T09:00:00+00:00", datetime(2026, 9, 1, 9, 0, tzinfo=UTC)),
            ("2026-10-18T23:59:59-07:00", datetime(2026, 10, 19, 6, 59, 59, tzinfo=UTC)),
            # A NAIVE ISO value is a wall-clock reading with no zone of its own, so it
            # is read on the USER's clock (#2206) — never defaulted to UTC, which put a
            # bounded watch's end hours away from the evening it was asked for.
            # America/Toronto is EDT (UTC-4) on this date.
            ("2026-09-01 09:00", datetime(2026, 9, 1, 13, 0, tzinfo=UTC)),
            ("2026-09-01", datetime(2026, 9, 1, 4, 0, tzinfo=UTC)),
            # The exact naive shape a live model wrote for ``expires_at`` on an
            # end-to-end eval run (#2455 naive baseline) — pinned literally so a
            # regression back to a UTC default shows here, not only three hours off
            # in a live-model report. America/Toronto is still EDT (UTC-4) in October.
            ("2026-10-18T22:00:00", datetime(2026, 10, 19, 2, 0, tzinfo=UTC)),
            # Words, read in the user's zone: 10pm where they are is 02:00 UTC next day.
            ("10pm today", datetime(2026, 8, 10, 2, 0, tzinfo=UTC)),
            ("tomorrow at 9am", datetime(2026, 8, 10, 13, 0, tzinfo=UTC)),
            # A relative span counts from the user's own wall clock, not from UTC's.
            ("in two weeks", datetime(2026, 8, 23, 20, 0, tzinfo=UTC)),
        ],
    )
    def test_accepted_end_conditions(self, value, expected):
        assert parse_expires_at(value, self._ZONE, self._NOW) == expected

    @pytest.mark.parametrize(
        "value",
        [
            # Measured: dateparser finds no time in these, so they are refused rather
            # than guessed at — words that point at a time without saying one.
            "10pm tonight",
            "the end of the month",
            "next friday",
            # And plain nonsense.
            "not-a-real-date",
            "banana",
        ],
    )
    def test_refused_end_conditions_teach_both_shapes(self, value):
        with pytest.raises(ScheduleError) as raised:
            parse_expires_at(value, self._ZONE, self._NOW)
        message = str(raised.value)
        assert message.startswith(f"I couldn't read expires_at='{value}' as a time.")
        assert "2026-09-01T09:00:00Z" in message
        assert "'tomorrow at 9am'" in message

    def test_no_profile_timezone_reads_the_words_in_utc(self):
        """A fresh install has no profile, so there is no zone to read words in — UTC,
        the same fallback the current-time anchor takes, rather than a refusal."""
        assert parse_expires_at("10pm today", None, self._NOW) == datetime(
            2026, 8, 9, 22, 0, tzinfo=UTC
        )

    def test_no_profile_timezone_reads_a_naive_iso_value_in_utc_too(self):
        """The same fallback applies to the ISO branch (#2206): with no zone to read a
        naive value on, it falls back to UTC rather than refusing."""
        assert parse_expires_at("2026-09-01 09:00", None, self._NOW) == datetime(
            2026, 9, 1, 9, 0, tzinfo=UTC
        )

    @pytest.mark.parametrize(
        "value",
        [
            # The measured case: the apply turn's date-shaped way of writing "forever"
            # for a job meant to run indefinitely.
            "2099-12-31",
            "2099-12-31T23:59:59Z",
            # The other sentinel in circulation — the maximum representable date.
            "9999-12-31",
        ],
    )
    def test_far_future_sentinels_read_as_no_end_date(self, value):
        """A time past the horizon is not an end date, it is a stand-in for "forever" —
        so it reads as NO end condition rather than being stored and then rendered back
        as a real one on every surface that shows an expiry (#1944)."""
        assert parse_expires_at(value, self._ZONE, self._NOW) is None

    def test_a_distant_but_meanable_end_date_still_lands(self):
        """The over-correction guard: the rule is a horizon, not a suspicion of
        distance.  A decade out is still a date someone can point at, so it is stored as
        one.  Naive, so it's read on the user's clock (#2206) — America/Toronto is EST
        (UTC-5) in January."""
        assert parse_expires_at("2036-01-01", self._ZONE, self._NOW) == datetime(
            2036, 1, 1, 5, 0, tzinfo=UTC
        )


class TestScheduleRoundTrip:
    """The rendered schedule IS the copyable input (display form == invocation form,
    #1857): the rule a surface shows, passed verbatim back as the ``schedule`` arg,
    parses to the same stored config.  With one grammar the render is the stored string
    itself, so what this pins is that a two-line rule survives the round trip through a
    one-line surface — the only place the two forms can diverge.

    It is also what makes anchoring a rule in the user's zone (#1932) safe to do at the
    READINESS gate rather than at parse: the stored rule is never rewritten, so a local
    morning hour renders back as the hour the user said, and re-passing a rendered
    schedule cannot convert it a second time."""

    @pytest.mark.parametrize(
        "schedule, clause",
        [
            ("FREQ=HOURLY", "FREQ=HOURLY"),
            ("FREQ=MINUTELY;INTERVAL=90", "FREQ=MINUTELY;INTERVAL=90"),
            ("FREQ=DAILY;BYHOUR=8", "FREQ=DAILY;BYHOUR=8"),
            (
                "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9",
                "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9",
            ),
            # The #1932 shape: a morning hour on the user's clock, stored as stated.
            ("FREQ=DAILY;BYHOUR=9;BYMINUTE=30", "FREQ=DAILY;BYHOUR=9;BYMINUTE=30"),
            (
                "DTSTART:20261225T090000Z\\nFREQ=DAILY;COUNT=1",
                "DTSTART:20261225T090000Z\\nFREQ=DAILY;COUNT=1",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_schedule_clause_round_trips(self, db, schedule, clause):
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="round-trip",
            description="round trip",
            prompt="Check Cinder Peak's elevation above sea level and save it.",
            schedule=schedule,
        )
        assert result.success
        # The stored rule renders back AS the copyable clause (stored → display).
        stored = db.memories.get("round-trip")
        assert stored is not None
        assert render_schedule_clause(stored) == clause
        # That clause, re-parsed, renders to the IDENTICAL clause (display → parse →
        # display) and lifts the same end conditions.
        reparsed = parse_schedule(clause)
        rerendered = MemoryRow(
            name="x",
            type="collection",
            description="d",
            schedule=reparsed.rule,
            max_runs=reparsed.max_runs,
            expires_at=reparsed.expires_at,
        )
        assert render_schedule_clause(rerendered) == clause
        assert (rerendered.max_runs, rerendered.expires_at) == (
            stored.max_runs,
            stored.expires_at,
        )

    @pytest.mark.asyncio
    async def test_a_users_zone_never_rewrites_the_stored_rule(self, db):
        """A profile with a timezone changes WHEN a rule fires (#1932), never WHAT it
        says.  Normalising the hour into UTC at store time would have made the render a
        translation of the ask rather than the ask, and re-passing that render would have
        converted it a second time — so the zone is applied at the readiness gate and the
        rule stays the user's own words."""
        db.users.save_info(
            sender="test-user",
            name="Test User",
            location="Somewhere",
            timezone="America/Toronto",
            date_of_birth="1990-01-01",
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="morning-briefing",
            description="a morning briefing",
            prompt="Check Cinder Peak's elevation above sea level and save it.",
            schedule="FREQ=DAILY;BYHOUR=9;BYMINUTE=30",
        )
        assert result.success
        stored = db.memories.get("morning-briefing")
        assert stored is not None
        assert stored.schedule == "FREQ=DAILY;BYHOUR=9;BYMINUTE=30"
        assert render_schedule_clause(stored) == "FREQ=DAILY;BYHOUR=9;BYMINUTE=30"

    @pytest.mark.asyncio
    async def test_creation_refuses_the_schedules_own_until_and_names_expires_at(self, db):
        """A schedule stating its own ``UNTIL=`` is refused through the real tool call,
        naming ``expires_at`` as where a job's end goes instead — a job's end is said
        ONE way."""
        db.users.save_info(
            sender="test-user",
            name="Test User",
            location="Somewhere",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="weekend-watch",
            description="a weekend watch",
            prompt="Check Cinder Peak's elevation above sea level and save it.",
            schedule="FREQ=DAILY;UNTIL=20261019T000000Z",
        )
        assert result.success is False
        assert "sets UNTIL=20261019T000000Z" in result.message
        assert "expires_at" in result.message
        assert db.memories.get("weekend-watch") is None


def _row_with_schedule(schedule: str | None) -> MemoryRow:
    """A bare row carrying only a schedule — ``describe_schedule``/``has_schedule`` read
    nothing else off it."""
    return MemoryRow(name="x", type="collection", description="d", schedule=schedule)


class TestDescribeSchedule:
    """Plain words for a schedule, with its exact rule beside them in a code span
    (#2425): the fix for a raw ``DTSTART``/RRULE reaching the model and a reply copying
    it verbatim.  ``describe_schedule`` is built from the rule's own parts rather than a
    lookup of the taught shapes, so this also covers combinations the grammar allows but
    nothing taught (e.g. a comma list of BYHOUR values, COUNT > 1 with no DTSTART)."""

    @pytest.mark.parametrize(
        "schedule, words",
        [
            ("FREQ=HOURLY", "every hour (rule: `FREQ=HOURLY`)"),
            (
                "FREQ=MINUTELY;INTERVAL=90",
                "every 90 minutes (rule: `FREQ=MINUTELY;INTERVAL=90`)",
            ),
            ("FREQ=DAILY;BYHOUR=8", "every day at 08:00 AM (rule: `FREQ=DAILY;BYHOUR=8`)"),
            (
                "FREQ=DAILY;BYHOUR=8,20",
                "every day at 08:00 AM and 08:00 PM (rule: `FREQ=DAILY;BYHOUR=8,20`)",
            ),
            (
                "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9",
                "weekdays at 09:00 AM (rule: `FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9`)",
            ),
            (
                "FREQ=WEEKLY;BYDAY=SA,SU",
                "weekends (rule: `FREQ=WEEKLY;BYDAY=SA,SU`)",
            ),
            (
                "FREQ=WEEKLY;BYDAY=MO,WE,FR",
                "Monday, Wednesday, and Friday (rule: `FREQ=WEEKLY;BYDAY=MO,WE,FR`)",
            ),
            ("FREQ=DAILY;COUNT=3", "every day, 3 times (rule: `FREQ=DAILY;COUNT=3`)"),
        ],
    )
    def test_taught_and_untaught_shapes_read_as_words(self, schedule, words):
        assert describe_schedule(_row_with_schedule(schedule)) == words

    def test_an_explicit_utc_dtstart_converts_to_the_users_clock(self):
        """A model-invented ``DTSTART:…Z`` reads on the USER's clock, not UTC (#1932) —
        the same reading an occurrence gets — so the words say what the model actually
        set up rather than a UTC instant nobody but it would compute."""
        schedule = "DTSTART:20261014T210000Z\nFREQ=HOURLY"
        words = describe_schedule(_row_with_schedule(schedule), "America/Los_Angeles")
        assert words == (
            "starting Wednesday, October 14, 2026 at 02:00 PM PDT, then every hour "
            "(rule: `DTSTART:20261014T210000Z FREQ=HOURLY`)"
        )

    def test_a_floating_dtstart_is_read_as_the_users_own_wall_clock(self):
        """A ``DTSTART`` with no zone states no zone — RFC 5545 reads that as the user's
        own wall clock, never UTC — so it renders the stated hour, not a converted one."""
        schedule = "DTSTART:20261014T140000\nFREQ=HOURLY"
        words = describe_schedule(_row_with_schedule(schedule), "America/Los_Angeles")
        assert words == (
            "starting Wednesday, October 14, 2026 at 02:00 PM PDT, then every hour "
            "(rule: `DTSTART:20261014T140000 FREQ=HOURLY`)"
        )

    def test_a_one_shot_dtstart_retires_instead_of_restating_its_cadence(self):
        """``COUNT=1`` reads as 'once … then it retires' — the cadence words
        (``every day``) would be misleading for a rule that never repeats."""
        schedule = "DTSTART:20261225T090000Z\nFREQ=DAILY;COUNT=1"
        words = describe_schedule(_row_with_schedule(schedule))
        assert words == (
            "once, at Friday, December 25, 2026 at 09:00 AM UTC, then it retires "
            "(rule: `DTSTART:20261225T090000Z FREQ=DAILY;COUNT=1`)"
        )

    def test_no_schedule_is_empty_for_the_clause_and_none_for_the_field(self):
        row = _row_with_schedule(None)
        assert describe_schedule(row) == ""
        assert describe_schedule_field(row) == "none"

    def test_the_end_condition_is_not_restated_inside_the_words(self):
        """``UNTIL=`` is already lifted onto ``expires_at`` and rendered on its own
        line — stating it again here would say the same fact twice.  Reads a row
        directly rather than through ``parse_schedule``, since #2511 refuses a fresh
        write stating its own ``UNTIL=``; this covers a row that already carries one
        from before that refusal existed."""
        schedule = "FREQ=DAILY;UNTIL=20260901T000000Z"
        words = describe_schedule(_row_with_schedule(schedule))
        assert words == f"every day (rule: `{schedule}`)"


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_log_persists(self, db):
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="user-messages", description="inbound"
        )
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["user-messages"].type == "log"

    @pytest.mark.asyncio
    async def test_create_log_duplicate_returns_user_friendly_message(self, db):
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="first"
        )
        result = await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="second"
        )
        assert "already exists" in result.message
        assert "events" in result.message

    @pytest.mark.asyncio
    async def test_update_refuses_a_blank_field_and_applies_nothing(self, db, mock_llm):
        # Models emit "" for an optional field they mean to leave alone (gpt-oss was
        # observed passing extraction_prompt="" alongside another change).  That is
        # REFUSED at the arg gate now (#1776) rather than silently read as "omitted" —
        # and the refusal is total: the notify flip riding alongside it doesn't land
        # either, so the model resends the call it meant instead of half of it.
        original_prompt = "test fixture extraction prompt that is long enough"
        _seed_collection(
            db,
            name="notes",
            description="real description",
            extraction_prompt=original_prompt,
            schedule="FREQ=HOURLY",
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(),
            name="notes",
            description="   ",
            notify=True,
        )
        assert result.success is False
        # The whole render — the arg-validation envelope pairs the field's own schema
        # hint with the teaching reason and the retry instruction.
        assert result.message == (
            "description (string: Content-reflective one-line summary AND the meaning "
            "anchor (find / resolve-by-meaning) — changing it re-embeds the collection. "
            "Keep it an accurate summary of the subject matter.): an empty string sets "
            "nothing — omit description entirely to leave it unset, or pass what the "
            "collection is for, in the user's own words. "
            "Call collection_configure(<valid arguments>) again."
        )
        untouched = db.memories.get("notes")
        assert untouched.extraction_prompt == original_prompt
        assert untouched.description == "real description"
        assert untouched.notify is False  # the whole call was refused, flip included

    @pytest.mark.asyncio
    async def test_update_rejects_the_dropped_intent_arg(self, db, mock_llm):
        # ``intent`` was dropped from the surface (#1631 — ``description`` absorbs it),
        # so passing it is an unknown parameter refused by the ``extra="forbid"`` gate,
        # not silently accepted-and-ignored (the old accept-but-ignore ceremony died).
        _seed_collection(
            db,
            name="notes",
            description="real description",
            extraction_prompt="test fixture extraction prompt that is long enough",
            schedule="FREQ=HOURLY",
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(),
            name="notes",
            intent="a rewritten goal the model tried to set",
        )
        assert result.success is False
        assert "intent" in result.message  # names the unknown field
        # The collection's description is untouched.
        assert db.memories.get("notes").description == "real description"

    @pytest.mark.asyncio
    async def test_create_surfaces_description_embed_degradation(self, db):
        """A transient description-embed failure still creates the collection, but the
        result NAMES the degraded routing anchor and leaves it NULL for the startup
        backfill to re-heal (#1468) — a visible degradation, not a silent success."""
        result = await CollectionCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            CallLink.new_top(),
            name="notes",
            description="a running list of notes",
            prompt="Check Cinder Peak's elevation above sea level and save it.",
            schedule="FREQ=HOURLY",
        )
        assert "Created" in result.message
        assert result.mutated is True
        assert "transient embedding error" in result.message
        assert "self-heal" in result.message
        # Row exists; anchor left NULL for the backfill to re-heal.
        row = db.memories.get("notes")
        assert row is not None
        assert row.description_embedding is None

    @pytest.mark.asyncio
    async def test_log_create_surfaces_description_embed_degradation(self, db):
        result = await LogCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            CallLink.new_top(), log="events", description="event stream"
        )
        assert "Created log" in result.message
        assert "transient embedding error" in result.message
        assert db.memories.get("events").description_embedding is None

    @pytest.mark.asyncio
    async def test_update_failed_description_embed_clears_stale_anchor(self, db):
        """Changing a description whose embed fails clears the anchor to NULL — it does
        NOT leave the old, now-mismatched vector in place (a stale anchor the NULL-only
        description backfill could never detect, #1468).  The new text lands, the anchor
        is left for the backfill to re-heal, and the degradation surfaces."""
        _seed_collection(
            db,
            name="notes",
            description="old subject",
            extraction_prompt="test fixture extraction prompt that is long enough",
            schedule="FREQ=HOURLY",
        )
        assert db.memories.get("notes").description_embedding is not None  # a good anchor first

        result = await CollectionUpdateTool(db, cast(Any, _FailingEmbedClient())).execute(
            CallLink.new_top(), name="notes", description="a completely different subject"
        )
        assert "Updated" in result.message
        assert "transient embedding error" in result.message
        row = db.memories.get("notes")
        assert row.description == "a completely different subject"  # new text landed
        assert row.description_embedding is None  # stale anchor cleared, not kept


# ── Inert collections (#1629, #2455 naive baseline) ──────────────────────────

# The prompt-less creation echo — storage only, no job, and the honest next step.


_INERT_ECHO_LITERAL = (
    "Set up collection 'deals-watch' — storage only, no job yet:\n"
    "  description: track the trail-runner shoe deals\n"
    "  status: inert (no prompt attached)\n"
    "It'll hold whatever gets written to it, but nothing runs against it until it has a "
    "prompt. Attach one with collection_configure(collection='deals-watch', "
    "prompt=<what the job's cycles should do>, schedule=<recurrence rule>)."
)


class TestInertCollections:
    """Prompt-less create yields an INERT storage collection (#1629/#2455): no
    extraction_prompt / cadence / notify, never dispatches, metadata-visible,
    idempotency still applies — and a job-shaped arg alongside is refused."""

    @pytest.mark.asyncio
    async def test_prompt_less_create_is_inert_storage_only(self, db):
        """A create with no prompt lands exactly one storage-only row — no
        extraction_prompt, cadence, or notify — and the echo is honest about being
        inert (whole render)."""
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="deals-watch", description="track the trail-runner shoe deals"
        )
        assert result.success and result.mutated
        assert result.message == _INERT_ECHO_LITERAL
        row = db.memories.get("deals-watch")
        assert row is not None
        assert row.extraction_prompt is None  # no job
        assert row.schedule is None  # no cadence
        assert row.notify is False  # silent
        assert row.archived is False  # a live, usable container

    @pytest.mark.asyncio
    async def test_inert_collection_is_metadata_visible(self, db):
        """An inert collection reads back through ``memory_get``, marked as
        storage with no routine — not hidden for lacking a prompt (#1629)."""
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="deals-watch", description="track the trail-runner shoe deals"
        )
        result = await MemoryMetadataTool(db).execute(CallLink.new_top(), memory="deals-watch")
        assert "prompt: none" in result.message
        assert "description: track the trail-runner shoe deals" in result.message

    @pytest.mark.asyncio
    async def test_job_arg_on_skill_less_create_is_refused(self, db):
        """A schedule / notify / expiry on a skill-less create has no job to attach to —
        refused naming the two-step fix (whole render), nothing created."""
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="deals-watch",
            description="track the trail-runner shoe deals",
            schedule="FREQ=HOURLY",
        )
        assert result.success is False
        assert result.message == _INERT_JOB_ARGS_REFUSAL.format(name="deals-watch")
        assert db.memories.get("deals-watch") is None

    @pytest.mark.asyncio
    async def test_inert_create_of_an_existing_job_updates_it(self, db):
        """Idempotency-at-birth (#1567) still applies to a skill-less create, and since
        #1775 it RESOLVES rather than refuses: 'deals-watch' normalises to the same token
        as the existing 'deals' (the role word is dropped), so the call lands on that
        collection — nothing new created, and the message names where it went."""
        _seed_collection(
            db,
            name="deals",
            description="track the trail-runner shoe deals",
        )
        result = await CollectionSetTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            collection="deals-watch",
            description="track the trail-runner shoe deals",
        )
        assert result.success and result.mutated
        assert db.memories.get("deals-watch") is None
        assert result.message.startswith(
            _SAME_JOB_UPDATED.format(requested="deals-watch", existing="deals")
        )
        assert db.memories.get("deals").description == "track the trail-runner shoe deals"


class TestBlankOptionalArgsAreRefused:
    """A blank optional string on the collection surface is a TEACHING REJECTION at the
    arg gate (#1776), never a silent drop.

    Models fill an optional argument they mean to omit with ``""`` (a chronic gpt-oss
    habit).  #1646 coerced that to ``None`` — read as "omitted" — which made the two
    indistinguishable: the observed run set up an hourly watch with ``expires_at=""``
    on a request that said "until 10pm tonight" and the end condition simply vanished,
    every other field correct, nothing in the reply or the record marking the
    difference.  Omitting an argument and passing ``""`` for it are different calls,
    and only the first means "leave this alone".

    The rule lives on the Pydantic args model (``OptionalText`` / ``OptionalSkill``), so
    ``execute`` never sees a blank and nothing is created or reconfigured on a call the
    model has to redo — and it covers every optional string on the surface, not just the
    one field the miss was observed on.
    """

    @pytest.mark.asyncio
    async def test_blank_expires_at_is_refused_naming_the_field_and_the_form(self, db):
        """The observed case, whole render: the drop is named, the remedy is both halves
        (omit it, or pass the accepted form), and no collection is created."""
        result = await CollectionSetTool(db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(),
            collection="deals-watch",
            description="track the trail-runner shoe deals",
            prompt="Read the trail-runner shoe deals page and note what's on sale.",
            schedule="FREQ=HOURLY",
            expires_at="",
        )
        assert result.success is False
        assert result.message == (
            "expires_at (string: When it ends — a date and time on your own clock, "
            "written with no zone, like '2026-11-02T22:00:00'. Leave it out when "
            "there's no end date.): an empty string sets nothing — omit expires_at "
            "entirely to leave it unset, or pass when it ends, e.g. "
            '2026-03-01T09:00:00Z or "in two weeks". Call collection_configure(<valid '
            "arguments>) again."
        )
        assert db.memories.get("deals-watch") is None

    def test_every_blank_refusing_field_has_a_form(self):
        """``OPTIONAL_ARG_FORMS`` is TOTAL over the fields that refuse a blank —
        derived from the three arg models themselves, not from a copy of the table.

        The validator indexes the table directly (no default), so a new
        ``OptionalText`` / ``OptionalSkill`` field added without an entry would raise a
        ``KeyError`` *inside* validation — escaping ``Tool.run``, which catches only
        ``ValidationError``, as a framework exception rather than a teaching rejection.
        This is the pin that makes that a red test instead of a production traceback."""
        declared = {
            name
            for model in (CollectionCreateArgs, CollectionSetArgs, CollectionUpdateArgs)
            for name, field in model.model_fields.items()
            if any(
                isinstance(meta, BeforeValidator) and meta.func is _reject_blank_optional
                for meta in field.metadata
            )
        }
        assert declared == set(OPTIONAL_ARG_FORMS)
        assert declared  # the introspection really found the fields

    @pytest.mark.asyncio
    async def test_every_optional_string_names_its_own_form(self, db):
        """Uniform across the surface — each optional string refuses a blank with ITS
        accepted form, so the guard isn't special-cased to ``expires_at``.  A
        whitespace-only value is the same miss as an empty one.  ``expires_at`` and
        ``schedule`` each carry their own schema descriptor (#2250/#2222, each shared
        across create/update/set), so their lines also name them; the remaining fields
        are still bare in this tool's schema."""
        tool = CollectionSetTool(db, cast(Any, MockLlmClient()))
        descriptors = {
            "expires_at": EXPIRES_AT_PARAM_DESCRIPTION,
            "schedule": SCHEDULE_PARAM_DESCRIPTION,
        }
        for field, form in OPTIONAL_ARG_FORMS.items():
            result = await tool.run(CallLink.new_top(), collection="deals-watch", **{field: "   "})
            assert result.success is False
            descriptor = (
                f"{field} (string: {descriptors[field]})"
                if field in descriptors
                else f"{field} (string)"
            )
            assert result.message == (
                f"{descriptor}: an empty string sets nothing — omit {field} entirely "
                f"to leave it unset, or pass {form}. "
                f"Call collection_configure(<valid arguments>) again."
            )
        assert db.memories.get("deals-watch") is None

    @pytest.mark.asyncio
    async def test_blank_is_refused_identically_on_create_and_update(self, db):
        """Both halves of the idempotent front door hold the same rule — the update path
        (an existing collection) refuses a blank exactly as the create path does, so a
        blank can never quietly leave an existing job's field alone either."""
        _seed_collection(db, name="deals-watch", description="track the shoe deals")
        result = await CollectionSetTool(db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(), collection="deals-watch", schedule=""
        )
        assert result.success is False
        assert result.message == (
            f"schedule (string: {SCHEDULE_PARAM_DESCRIPTION}): an empty string sets "
            "nothing — omit schedule entirely to leave it unset, or pass one RRULE "
            'line, e.g. "FREQ=HOURLY" for hourly. Call collection_configure(<valid '
            "arguments>) again."
        )
        # The existing row is untouched — the refusal happens before execute.
        row = db.memories.get("deals-watch")
        assert row.schedule == "FREQ=HOURLY"
        assert row.extraction_prompt == "test fixture extraction prompt"

    @pytest.mark.asyncio
    async def test_omitting_an_optional_arg_is_still_the_way_to_leave_it_alone(self, tmp_path):
        """The paired over-correction guard: refusing a blank must not make OMISSION
        harder.  Leaving the optional args off still creates the inert container and
        still edits metadata without touching the cadence or the routine — proven side by
        side on twin DBs so neither run's state leaks into the other."""
        create_dir, update_dir = tmp_path / "create", tmp_path / "update"
        create_dir.mkdir()
        update_dir.mkdir()
        create_db, update_db = (
            schema_only_db(str(create_dir / "test.db")),
            schema_only_db(str(update_dir / "test.db")),
        )

        created = await CollectionSetTool(create_db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(),
            collection="deals-watch",
            description="track the trail-runner shoe deals",
        )
        assert created.success and created.mutated
        assert created.message == _INERT_ECHO_LITERAL

        _seed_collection(update_db, name="deals-watch", description="track the shoe deals")
        updated = await CollectionSetTool(update_db, cast(Any, MockLlmClient())).run(
            CallLink.new_top(),
            collection="deals-watch",
            description="track the trail-runner shoe deals",
        )
        assert updated.success
        row = update_db.memories.get("deals-watch")
        assert row is not None
        assert row.description == "track the trail-runner shoe deals"
        assert row.schedule == "FREQ=HOURLY"  # cadence untouched
        assert row.extraction_prompt == "test fixture extraction prompt"  # routine untouched
        assert row.expires_at is None


class TestCollectionUpdateScheduleAtApply:
    """Schedule + notify are apply-time properties on collection_configure (#1629/#1857: one
    RRULE ``schedule``, plus ``expires_at`` and ``notify``) — recorded in the mutation
    event's changed fields."""

    @pytest.mark.asyncio
    async def test_adopt_applies_interval_notify_recorded_in_mutation(self, db):
        """Adopting a prompt with a schedule + notify sets both and records them in the
        mutation event's changed fields alongside the re-render."""
        _seed_collection(
            db,
            name="legacy",
            extraction_prompt="hand-authored prose long enough to pass",
            notify=False,
        )
        result = await CollectionUpdateTool(
            db, cast(Any, MockLlmClient()), run_id="run-adopt"
        ).execute(
            CallLink.new_top(),
            name="legacy",
            prompt="Check Cinder Peak's elevation above sea level and save it.",
            schedule="FREQ=HOURLY;INTERVAL=2",
            notify=True,
        )
        assert result.success
        stored = db.memories.get("legacy")
        assert stored.schedule == "FREQ=HOURLY;INTERVAL=2"
        assert stored.notify is True
        event = next(e for e in db.mutations.history("legacy", 5) if e.run_id == "run-adopt")
        summary = mutation_change_summary(event)
        assert "schedule" in summary and "notify" in summary and "extraction_prompt" in summary

    @pytest.mark.asyncio
    async def test_schedule_replaces_the_whole_schedule(self, db):
        """Setting a schedule replaces the stored one ALONG WITH the run quota lifted out
        of it (#1857), so a new rule never inherits the previous rule's ``COUNT=``."""
        _seed_collection(db, name="watch", schedule="FREQ=HOURLY")
        # Switch to a bounded one-shot: the quota comes with the rule.
        once = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="watch",
            schedule="DTSTART:20261225T090000Z\\nFREQ=DAILY;COUNT=1",
        )
        assert once.success
        row = db.memories.get("watch")
        assert row.schedule == "DTSTART:20261225T090000Z\nFREQ=DAILY;COUNT=1"
        assert row.max_runs == 1
        # Switch back to an unbounded recurring rule — the quota must clear with it.
        back = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="watch", schedule="FREQ=MINUTELY;INTERVAL=30"
        )
        assert back.success
        row = db.memories.get("watch")
        assert row.schedule == "FREQ=MINUTELY;INTERVAL=30"
        assert row.max_runs is None

    @pytest.mark.asyncio
    async def test_expires_at_at_apply_reads_the_users_words(self, db):
        """``expires_at`` takes plain words as well as a datetime, read in the user's
        own timezone (#1857) — a bare edit carries no rule, so there is no UNTIL to
        conflict with."""
        _seed_collection(db, name="watch", schedule="FREQ=HOURLY")
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="watch", expires_at="2026-09-01T09:00:00Z"
        )
        assert result.success
        stored = db.memories.get("watch").expires_at
        assert stored.replace(tzinfo=UTC) == datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
        # A real end date is stored as itself, so nothing was normalised to report.
        assert "no end date" not in result.message

    @pytest.mark.asyncio
    async def test_far_future_expiry_at_apply_takes_a_stored_sentinel_off(self, db):
        """The apply-time half of the sentinel rule (#1944) — the door the live failure
        came through, since the invented ``2099-12-31`` rendered as the mechanism's end
        condition and was copied back on every later ``collection_configure``.

        The collection is seeded ALREADY CARRYING the sentinel, so the call has something
        to remove: normalising the argument alone would leave the stored date exactly
        where it was while the result claimed the job runs until it is stopped, and the
        stickiness the rule exists to end would survive it.  The removal is recorded as a
        change like any other, so the row does not quietly differ from the ledger."""
        _seed_collection(db, name="watch", schedule="FREQ=HOURLY")
        db.memories.update_collection_metadata(
            "watch", expires_at=datetime(2099, 12, 31, tzinfo=UTC)
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient()), run_id="run-x").execute(
            CallLink.new_top(), name="watch", expires_at="2099-12-31T00:00:00Z"
        )
        assert result.success
        assert db.memories.get("watch").expires_at is None
        event = next(e for e in db.mutations.history("watch", 5) if e.run_id == "run-x")
        assert "expires_at" in mutation_change_summary(event)
        # Whole render: the echo now states the expiry too (#2251) — what it was
        # before this call, and what it came to — so the note isn't the only thing
        # that tells the turn what became of the value it passed.
        assert result.message == (
            "Updated collection 'watch':\n"
            "  schedule: every hour (rule: `FREQ=HOURLY`)\n"
            "  notify: False\n"
            "  expires: was 2099-12-31 00:00 UTC → now never\n"
            "  description: x\n"
            "  steps: |\n"
            "    test fixture extraction prompt"
            "\n(Note: expires_at='2099-12-31T00:00:00Z' is further ahead than an end date "
            "anyone means, so I read it as no end date — the expiry is unset and this runs "
            "until it is stopped. Leave expires_at off when there is no end date.)"
        )

    @pytest.mark.asyncio
    async def test_adopt_without_schedule_warns_it_wont_run(self, db):
        """Adopting a prompt onto an inert collection with NO schedule leaves it with
        nothing to run on — the echo renders ``schedule: none`` (never a half-formed
        clause, #1666) and carries a visible no-schedule note (#1629), not a silent
        won't-run.  Whole render, so the honest schedule line and the tail note are both
        pinned."""
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="deals-watch", description="track the trail-runner shoe deals"
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="deals-watch",
            prompt="Check Cinder Peak's elevation above sea level and save it.",
        )
        assert result.success
        expected_echo = (
            "I'll run against 'deals-watch' and quietly store what it finds.\n"
            "Re-configured collection 'deals-watch':\n"
            "  description: track the trail-runner shoe deals\n"
            "  schedule: none\n"
            "  notify: False\n"
            "  expires: never\n"
            "  prompt: |\n"
            "    Check Cinder Peak's elevation above sea level and save it."
        )
        assert result.message == expected_echo + _NO_SCHEDULE_NOTE.format(name="deals-watch")
        assert db.memories.get("deals-watch").schedule is None

    @pytest.mark.asyncio
    async def test_plain_update_on_schedule_less_collection_echoes_none(self, db):
        """The live-observed #1666 bug: a plain metadata update (no skill/params) on a
        collection with no schedule echoes ``schedule: none`` — never a half-formed
        clause.  Whole render off an inert collection (the natural schedule-less case)."""
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="deals-watch", description="track the trail-runner shoe deals"
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), name="deals-watch", description="track the road-runner shoe deals"
        )
        assert result.success
        assert result.message == (
            "Updated collection 'deals-watch':\n"
            "  schedule: none\n"
            "  notify: False\n"
            "  expires: never\n"
            "  description: track the road-runner shoe deals\n"
            "  steps: |\n    "
        )

    @pytest.mark.asyncio
    async def test_a_prompt_that_is_the_jobs_summary_line_is_refused(self, db):
        """A re-time that also passes the job's own current-state summary line as its
        ``prompt`` (#2685) is refused: the summary is a rendering OF the instructions,
        and storing it would clobber them with it.  Nothing is stripped or normalised
        on her behalf — the refusal names the fix, and the row is untouched."""
        row = _seed_collection(
            db,
            name="watch",
            extraction_prompt=(
                "Check https://quillmarket.example.com/typewriters for portable "
                "typewriters and save it."
            ),
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="watch",
            prompt=job_summary_line(row).strip(),
            schedule="FREQ=DAILY;BYHOUR=11",
        )
        assert not result.success
        assert result.message == (
            "That's how your current state summarises 'watch', not its instructions — "
            "leave `prompt` out to keep them, or pass the full new instructions."
        )
        stored = db.memories.get("watch")
        assert stored.extraction_prompt == row.extraction_prompt
        assert stored.schedule == "FREQ=HOURLY"

    @pytest.mark.asyncio
    async def test_a_summary_line_with_its_instructions_tail_changed_is_refused_too(self, db):
        """The copied line with the instructions tail edited or cut (#2685) — it still
        starts with the ``reads <page> — "`` prefix the render builds from this job's
        own page — is refused the same way, and the row is untouched."""
        row = _seed_collection(
            db,
            name="watch",
            extraction_prompt=(
                "Check https://quillmarket.example.com/typewriters for portable "
                "typewriters and save it."
            ),
        )
        summary = job_summary_line(row).strip()
        prefix = summary[: summary.index(' — "') + 4]
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="watch",
            prompt=f"{prefix}keep an eye out for portable typewriters under $400.",
            schedule="FREQ=DAILY;BYHOUR=11",
        )
        assert not result.success
        assert result.message == (
            "That's how your current state summarises 'watch', not its instructions — "
            "leave `prompt` out to keep them, or pass the full new instructions."
        )
        assert db.memories.get("watch").extraction_prompt == row.extraction_prompt

    @pytest.mark.asyncio
    async def test_real_instructions_that_merely_mention_the_page_are_accepted(self, db):
        """New instructions that happen to name the same page (#2685) are the normal
        case — a prompt whose words merely overlap the render's page URL passes and is
        stored verbatim."""
        _seed_collection(
            db,
            name="watch",
            extraction_prompt=(
                "Check https://quillmarket.example.com/typewriters for portable "
                "typewriters and save it."
            ),
        )
        new_prompt = (
            "Read https://quillmarket.example.com/typewriters, list every portable "
            "typewriter added since the last cycle, and save each with its price."
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(),
            name="watch",
            prompt=new_prompt,
            schedule="FREQ=DAILY;BYHOUR=11",
        )
        assert result.success
        stored = db.memories.get("watch")
        assert stored.extraction_prompt == new_prompt
        assert stored.schedule == "FREQ=DAILY;BYHOUR=11"


class TestCollectionWritesAndReads:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, db, mock_llm):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        result = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[
                {"key": "dark roast", "content": "loves dark roast"},
                {"key": "cold brew", "content": "enjoys cold brew"},
            ],
        )
        assert "Wrote 2 entries to 'likes'" in result.message
        assert result.mutated is True
        latest = await CollectionListTool(db).execute(CallLink.new_top(), collection="likes")
        assert "dark roast" in latest.message
        assert "cold brew" in latest.message
        # Each rendered entry carries an absolute UTC timestamp so the model can
        # place it in time — read-tool output was previously timeless.
        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", latest.message)

    @pytest.mark.asyncio
    async def test_collection_configure_dispatches_on_existence(self, db, mock_llm):
        """The ONE idempotent entry point (the code-owner fusion): a missing name
        CREATES with full birth validation; an existing name UPDATES only the set
        fields — the model never reasons about which case applies.  Birth without
        a description, and birth with a hand-written extraction_prompt, are both
        actionable refusals."""
        tool = CollectionSetTool(db, _make_llm_client(mock_llm), run_id="run-set")
        # Birth needs a description.
        result = await tool.execute(CallLink.new_top(), collection="trip-plans")
        assert result.success is False and "needs a `description`" in result.message
        # extraction_prompt is not a model argument AT ALL (#1570): unknown-parameter
        # rejection through the arg envelope, on the set tool like everywhere else.
        result = await tool.run(
            CallLink.new_top(),
            collection="trip-plans",
            description="trip planning notes",
            extraction_prompt="1. x" * 10,
        )
        assert result.success is False and "extraction_prompt" in result.message
        # Create path: inert storage, provenance stamped.
        result = await tool.execute(
            CallLink.new_top(), collection="trip-plans", description="trip planning notes"
        )
        assert result.success is True
        row = db.memories.get("trip-plans")
        assert row is not None and row.created_by_run_id == "run-set"
        assert row.extraction_prompt is None and row.notify is False
        # Update path: same verb, only the set field changes.
        result = await tool.execute(CallLink.new_top(), collection="trip-plans", notify=True)
        assert result.success is True
        row = db.memories.get("trip-plans")
        assert row.notify is True and row.description == "trip planning notes"

    @pytest.mark.asyncio
    async def test_write_target_materialization(self, db, mock_llm):
        """A chat write never dies on a name miss (#1570): a near-certain name
        variant REDIRECTS — narrated, never silent — (typo/char leg AND the
        same-intent full-token-containment leg), the ambiguous middle band keeps
        the did-you-mean refusal, and nothing-close AUTO-CREATES the collection
        (placeholder description + created_by_run_id stamp).  A collector-scoped
        write is never materialized."""
        _seed_collection(db, name="gear-notes", description="gear notes")
        write = CollectionWriteTool(
            db,
            _make_llm_client(mock_llm),
            author="test",
            decider=_NO_MODEL,
            run_id="run-materialize",
        )
        # Typo (char leg): 'gear-ntoes' → 'gear-notes'.
        result = await write.execute(
            CallLink.new_top(),
            collection="gear-ntoes",
            entries=[{"key": "tent", "content": "3kg tent"}],
        )
        assert result.mutated is True
        assert result.message.startswith(
            "Note: no collection named 'gear-ntoes' exists — 'gear-notes' is the same "
            "name in a different spelling, so this write went there.\n"
        )
        assert db.memories.get("gear-ntoes") is None
        assert (
            "tent"
            in (
                await CollectionListTool(db).execute(CallLink.new_top(), collection="gear-notes")
            ).message
        )
        # Same-intent extension (TCR leg): 'gear' ⊂ 'gear-notes'.
        result = await write.execute(
            CallLink.new_top(),
            collection="gear",
            entries=[{"key": "stove", "content": "canister stove"}],
        )
        assert result.mutated is True and "'gear-notes'" in result.message
        assert db.memories.get("gear") is None
        # Nothing close → auto-create, narrated, provenance stamped.
        result = await write.execute(
            CallLink.new_top(),
            collection="trail-recipes",
            entries=[{"key": "oats", "content": "overnight oats"}],
        )
        assert result.mutated is True
        assert result.message.startswith(
            "Note: no collection named 'trail-recipes' existed and nothing close "
            "matched, so it was created (storage only, no job) and your entries "
            "went in.\n"
        )
        row = db.memories.get("trail-recipes")
        assert row is not None and row.created_by_run_id == "run-materialize"
        assert row.description == "auto-created to hold 'oats'"
        assert row.extraction_prompt is None
        # A collector-scoped write is pinned — no materialization, plain refusal.
        scoped = CollectionWriteTool(
            db,
            _make_llm_client(mock_llm),
            author="collector",
            decider=_NO_MODEL,
            scope="gear-notes",
        )
        result = await scoped.execute(
            CallLink.new_top(), collection="something-else", entries=[{"key": "x", "content": "y"}]
        )
        assert result.success is False
        assert db.memories.get("something-else") is None

    @pytest.mark.asyncio
    async def test_write_empty_entries_is_actionable_not_bare_pydantic(self, db, mock_llm):
        """An empty ``entries`` batch gets a named, actionable rejection through the
        arg-validation envelope — not Pydantic's bare "List should have at least 1
        item" (the house wording-unification pass)."""
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        result = await write.run(CallLink.new_top(), collection="likes", entries=[])
        assert result.success is False
        assert "at least one entry" in result.message
        assert "List should have at least 1 item" not in result.message

    @pytest.mark.asyncio
    async def test_write_unknown_key_in_entry_names_nested_path(self, db, mock_llm):
        """A misspelled/extraneous key INSIDE a batch entry surfaces the nested loc
        path and suggests the valid sibling, rather than being silently dropped or
        mis-rendered as the whole ``entries`` field (#1416)."""
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        result = await write.run(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "k", "content": "v", "contnt": "typo"}],
        )
        assert result.success is False
        assert "entries.0.contnt" in result.message
        assert "did you mean 'content'" in result.message

    def test_format_updated_shows_the_prior_and_the_restore_call(self):
        """The merge reply (#2715) shows the PRIOR value verbatim plus the one call
        that restores it and writes the new value under its own key — a wrong
        identity match is visible and reversible in a single call.  A cross-key
        merge names the match; an exact-key change does not."""
        merged = _format_updated(
            "likes",
            WriteResult(
                key="Genesis (band)",
                outcome=WriteGateOutcome.KEY_EXISTS_CHANGED,
                matched_key="genesis",
                prior_value="Genesis, the prog band",
            ),
        )
        assert "Updated 'genesis'" in merged
        assert "matched as the same thing as 'Genesis (band)'" in merged
        assert 'It held: "Genesis, the prog band"' in merged
        assert (
            "collection_write(collection='likes', "
            "entries=[{'key': 'genesis', 'content': \"Genesis, the prog band\"}])" in merged
        )
        assert "write yours under its own key ('Genesis (band)')" in merged
        exact = _format_updated(
            "watch",
            WriteResult(
                key="price",
                outcome=WriteGateOutcome.KEY_EXISTS_CHANGED,
                matched_key="price",
                prior_value="$219",
            ),
        )
        assert "Updated 'price'" in exact
        assert "matched as the same thing" not in exact
        assert 'It held: "$219"' in exact

    def test_format_unsure_names_the_candidate_and_both_calls(self):
        """The honest question (#2715) names the candidate key and what each call
        would do — update it if the write is the same thing, or let the new entry
        stand if it is a different thing.  It never tells the model to overwrite
        an entry the write is not about: the update call is conditioned."""
        asked = _format_unsure(
            "daily_headlines",
            WriteResult(
                key="2026-10-21",
                outcome=WriteGateOutcome.NEW_KEY,
                unsure_candidate="2026-10-20",
            ),
        )
        assert "'2026-10-21' was written as a new entry" in asked
        assert "also holds '2026-10-20'" in asked
        assert (
            "collection_write(collection='daily_headlines', "
            "entries=[{'key': '2026-10-20', 'content': <the value>}])" in asked
        )
        assert "if it is a different thing, the new entry stands as written" in asked

    @pytest.mark.asyncio
    async def test_write_unsettled_pair_without_a_model_writes_through_and_asks(self, db, mock_llm):
        """The named degraded mode (#2715): a look-alike pair the rules cannot
        settle is WRITTEN as new — unsure is treated as different — and the reply
        asks the writing model honestly whether it meant the shortlisted
        candidate, naming that key and what each call would do.  The motivating
        production failure (a rejection telling the model to overwrite yesterday's
        digest) cannot recur: nothing is rejected, nothing is overwritten."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        result = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        # The write landed (the entry exists), so it counts as work.
        assert result.mutated is True
        assert "'dark roast coffee' was written as a new entry" in result.message
        assert "also holds 'dark roast'" in result.message
        assert (
            "collection_write(collection='likes', "
            "entries=[{'key': 'dark roast', 'content': <the value>}])" in result.message
        )
        assert "if it is a different thing, the new entry stands as written" in result.message
        assert len(db.memories.memory("likes").read_all()) == 2

    @pytest.mark.asyncio
    async def test_write_merges_when_the_decider_says_new_value(self, db, mock_llm):
        """The decision step (#2715): a rules-unsettled look-alike goes to the
        decision model; ``new_value`` MERGES — the stored value replaced in place,
        and the reply shows the prior verbatim plus the one call that restores it."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_decider_saying("new_value")
        )
        await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        result = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast coffee", "content": "a fresher value"}],
        )
        assert result.mutated is True
        assert "Updated 'dark roast'" in result.message
        assert "matched as the same thing as 'dark roast coffee'" in result.message
        assert 'It held: "first body"' in result.message
        assert (
            "collection_write(collection='likes', "
            "entries=[{'key': 'dark roast', 'content': \"first body\"}])" in result.message
        )
        entries = db.memories.memory("likes").read_all()
        assert len(entries) == 1
        assert entries[0].content == "a fresher value"

    @pytest.mark.asyncio
    async def test_write_unchanged_and_already_recorded_prescribe_no_call(self, db, mock_llm):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        # A collector binds its writes to one collection via ``scope``.
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL, scope="likes"
        )
        await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        # A re-write under the SAME key with the SAME value is the change-gate's
        # UNCHANGED outcome (#1587) — the watch's "no change" signal, reported as
        # such (not a generic "duplicate").  It carried a STOP the collector loop
        # honored until #2385 removed the unchanged-value STOP: the result now
        # returns normally (stop is None) and the cycle goes on to its close, where
        # the quiet cycle is the model's recorded decision.
        same_key = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        assert "Unchanged: 'dark roast' already holds the same value" in same_key.message
        assert same_key.mutated is False
        assert same_key.stop is None
        # A batch with a genuinely new entry alongside a look-alike lands the new
        # one and asks honestly about the other — the ladder's unsure outcome,
        # never a rejection that leaves a write stranded (#2715).
        partial = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[
                {"key": "cold brew", "content": "a brand new distinct entry"},
                {"key": "dark roast blend", "content": "roasted longer this season"},
            ],
        )
        assert "Wrote 2 entries" in partial.message
        assert "'dark roast blend' was written as a new entry" in partial.message
        assert "also holds 'dark roast'" in partial.message
        assert partial.mutated is True
        # A collision on the VALUE under a DIFFERENT key is the other branch (#1919):
        # the same no-news the exact key reads, so it prescribes NO call — there is
        # nothing to do about a value that is already stored.  The identical-content
        # rule settles it without a model.
        same_value = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "morning roast selection", "content": "first body"}],
        )
        assert (
            same_value.stop is None
        )  # no write outcome stops since #2385 — a result, not a verdict
        assert same_value.mutated is False
        assert (
            "Already recorded: 'morning roast selection' is already stored as 'dark roast'"
            in same_value.message
        )
        assert "collection_write(" not in same_value.message
        assert "was not written" not in same_value.message

    @pytest.mark.asyncio
    async def test_chat_scope_already_recorded_write_has_no_stop(self, db, mock_llm):
        """A same-value collision under a different key reports the same text on the chat
        surface and NEVER a loop-stop — STOP applies to must-act cadence contexts only
        (#1587/#1919), exactly as for the exact-key UNCHANGED case below."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        already = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "morning roast selection", "content": "first body"}],
        )
        assert "Already recorded: 'morning roast selection'" in already.message
        assert already.stop is None

    @pytest.mark.asyncio
    async def test_chat_scope_unchanged_write_has_no_stop(self, db, mock_llm):
        """The chat surface (scope=None) gets the SAME enumerated UNCHANGED text but
        NEVER a loop-stop — STOP applies to must-act cadence contexts only (#1587).
        Contrast the collector-scope write above, which sets ``stop``."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        # No scope → chat surface.
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        unchanged = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "first body"}],
        )
        assert "Unchanged: 'dark roast' already holds the same value" in unchanged.message
        assert unchanged.stop is None

    @pytest.mark.asyncio
    async def test_change_gate_changed_auto_refreshes_baseline(self, db, mock_llm):
        """The CHANGED auto-refresh result text (#1633): re-writing an EXACT key with a
        DIFFERENT value refreshes the stored baseline IN PLACE through the shared write
        gate — the result reports the refresh with no further call for the model
        instruction (the wasted call that would teach the model to redo what already
        happened), ``mutated=True`` (durable state changed), and CHANGED is never
        STOP-worthy so no loop-stop even for a collector-scoped write.  The gate is
        shared, so a chat-surface write (``scope=None``) gets the same auto-refresh and
        stays non-STOP with honest text."""
        _seed_collection(
            db,
            name="watch",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL, scope="watch"
        )
        await write.execute(
            CallLink.new_top(), collection="watch", entries=[{"key": "price", "content": "$42"}]
        )
        changed = await write.execute(
            CallLink.new_top(), collection="watch", entries=[{"key": "price", "content": "$40"}]
        )
        assert changed.message == (
            "Updated 'price'. It held: \"$42\". If they are different things, "
            "restore it with collection_write(collection='watch', "
            "entries=[{'key': 'price', 'content': \"$42\"}]) and write yours "
            "under its own key ('price')."
        )
        assert changed.mutated is True
        assert changed.stop is None
        # The baseline was refreshed in place — one row, now the new value.
        refreshed = db.memory("watch").get("price")
        assert len(refreshed) == 1
        assert refreshed[0].content == "$40"
        # The gate is shared: a chat-surface write auto-refreshes identically, never STOPs.
        chat_write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="chat", decider=_NO_MODEL
        )
        await chat_write.execute(
            CallLink.new_top(), collection="watch", entries=[{"key": "note", "content": "v1"}]
        )
        chat_changed = await chat_write.execute(
            CallLink.new_top(), collection="watch", entries=[{"key": "note", "content": "v2"}]
        )
        assert chat_changed.message == (
            "Updated 'note'. It held: \"v1\". If they are different things, "
            "restore it with collection_write(collection='watch', "
            "entries=[{'key': 'note', 'content': \"v1\"}]) and write yours "
            "under its own key ('note')."
        )
        assert chat_changed.mutated is True
        assert chat_changed.stop is None
        assert db.memory("watch").get("note")[0].content == "v2"


class TestWritesLandedRender:
    """What a run actually WROTE, rendered from the ledger rows themselves (#1946).

    The frame this feeds exists because a turn's own account of its writes counts what it
    ATTEMPTED: a draw the reroll guard discarded and a write the change-gate refused both
    feel like writes from inside the run, and the store holds neither.  So the render is
    driven straight from entry rows here — the same rows ``entries_written_by_run``
    returns — and every shape the frame has to survive is one of these fixtures."""

    @staticmethod
    def _entry(memory_name: str, key: str | None, content: str = "a value") -> MemoryEntry:
        return MemoryEntry(memory_name=memory_name, key=key, content=content, author="chat")

    def test_multiple_entries_render_grouped_by_collection(self):
        """Count, collection, keys — the whole of what a reply can truthfully claim it
        saved, one line per collection."""
        assert render_writes_landed(
            [
                self._entry("lantern-prices", "keel lantern"),
                self._entry("lantern-prices", "aurora lantern"),
                self._entry("lantern-notes", "restock window"),
            ]
        ) == (
            "2 entries landed in 'lantern-prices': 'keel lantern', 'aurora lantern'\n"
            "1 entry landed in 'lantern-notes': 'restock window'"
        )

    def test_a_run_that_wrote_nothing_has_no_frame(self):
        """No writes, no record to narrate — which is most turns, and is why the frame
        cannot simply fire on every chat run."""
        assert render_writes_landed([]) is None

    def test_a_discarded_write_is_absent_because_it_never_landed(self):
        """The observed regression's shape: a demonstration whose first write LANDED and
        whose second draw was discarded by the reroll guard before it ever became a call.

        The run remembers two; the ledger holds one, because a discarded draw writes no
        entry.  The frame carries only what landed, and that asymmetry is the whole
        point of reading the store instead of counting the turn's intentions."""
        assert (
            render_writes_landed([self._entry("lantern-prices", "keel lantern")])
            == "1 entry landed in 'lantern-prices': 'keel lantern'"
        )

    def test_keyless_appends_are_not_writes(self):
        """A key is the structural mark of a registry write; an append with no key is a
        stream append — browse scratch above all — and it is the same mark the self-state
        header's own writes clause reads, so the two renders cannot disagree.

        A run that only appended to a log therefore has no frame at all, which is what
        keeps an ordinary browsing turn from being nudged into narrating its scratch."""
        assert render_writes_landed([self._entry("browse-results", None, "a fetched page")]) is None
        assert (
            render_writes_landed(
                [
                    self._entry("browse-results", None, "a fetched page"),
                    self._entry("lantern-prices", "keel lantern"),
                ]
            )
            == "1 entry landed in 'lantern-prices': 'keel lantern'"
        )

    def test_a_run_that_tried_and_kept_nothing_is_told_each_try(self):
        """The other half of the record (#2185): a run that asked the store to keep
        something and kept none of it.  It leaves no entry stamp, so without this it read
        exactly like a run that never tried, and the reply narrated the save it meant.

        Each try renders with the answer its tool gave, verbatim — a refusal that spans
        lines keeps every line, indented under its own bullet — so "already there" and "the
        arguments were wrong" stay the different outcomes they are."""
        attempts = [
            ToolCallRecord(
                tool="collection_write",
                arguments={},
                failed=False,
                result=(
                    "Already recorded: 'sea kayaking' is already stored as 'sea kayaking' "
                    "— no change since the last write (entry)."
                ),
            ),
            ToolCallRecord(
                tool="update_entry",
                arguments={},
                failed=True,
                result="Key 'kayak' not found in 'hobbies'.\nWrite it as a new entry.",
            ),
        ]
        assert render_nothing_landed(attempts) == (
            "Nothing this turn tried to keep is in the store. What each try came back with:\n"
            "- `collection_write`: Already recorded: 'sea kayaking' is already stored as "
            "'sea kayaking' — no change since the last write (entry).\n"
            "- `update_entry`: Key 'kayak' not found in 'hobbies'.\n"
            "  Write it as a new entry."
        )

    def test_a_run_that_never_tried_has_no_record(self):
        """No try, no record: a run that only read or talked asked nothing of the store,
        and "nothing was saved" would be a record about a question nobody asked."""
        assert render_nothing_landed([]) is None


class TestEmbedFailureRefusesWrite:
    """A transient embed failure at write time REFUSES the write — no vectorless
    (recall-invisible, dedup-weakening) entry is ever persisted (#1412).  The
    prior behaviour stored the entry without a vector and returned an optimistic
    success; now the tool fails with an actionable retry and nothing lands."""

    @staticmethod
    async def _make_relevant_collection(db) -> None:
        # A fully recall-eligible collection: proves the refusal is about the
        # missing vector, not the memory being excluded from recall.
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )

    @pytest.mark.asyncio
    async def test_collection_write_refuses_when_embedding_fails(self, db):
        await self._make_relevant_collection(db)
        write = CollectionWriteTool(
            db, cast(Any, _FailingEmbedClient()), author="test", decider=_NO_MODEL
        )
        result = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "loves dark roast"}],
        )
        # Fail-hard: actionable failure naming the transient cause and binding the
        # retry, and no work reported (so the collector throttle sees no-op).
        assert result.success is False
        assert result.mutated is False
        assert "transient embedding error" in result.message
        assert "'dark roast'" in result.message
        assert "collection_write(collection='likes'" in result.message
        # Nothing persisted — the invariant "every stored entry has a vector" holds.
        with Session(db.engine) as session:
            rows = session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == "likes")).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_collection_write_refuses_on_key_only_embed_failure(self, db):
        # Even when only the key embed fails (content vector fine), storing the
        # entry would leave it missing a vector — so the write is still refused
        # atomically and nothing lands.  (The backfill now also repairs a
        # key-null row, #1468, but the write path won't persist one to begin with.)
        await self._make_relevant_collection(db)
        write = CollectionWriteTool(
            db,
            cast(Any, _KeyOnlyFailingEmbedClient("dark roast")),
            author="test",
            decider=_NO_MODEL,
        )
        result = await write.execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "dark roast", "content": "loves dark roast"}],
        )
        assert result.success is False
        assert result.mutated is False
        assert "'dark roast'" in result.message
        with Session(db.engine) as session:
            rows = session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == "likes")).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_log_append_refuses_when_embedding_fails(self, db):
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, cast(Any, _FailingEmbedClient()), author="test")
        result = await append.execute(
            CallLink.new_top(), log="events", content="something happened"
        )
        assert result.success is False
        assert result.mutated is False
        assert "transient embedding error" in result.message
        assert "log_append(log='events'" in result.message
        with Session(db.engine) as session:
            rows = session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == "events")
            ).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_get_returns_entry_or_not_found(self, db, mock_llm):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        ).execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "k", "content": "hello"}]
        )
        assert (
            "hello"
            in (
                await CollectionGetTool(db).execute(CallLink.new_top(), collection="likes", key="k")
            ).message
        )
        missing = await CollectionGetTool(db).execute(
            CallLink.new_top(), collection="likes", key="absent"
        )
        assert "not found" in missing.message
        # A key one typo away from a stored one is named in the miss, so the retry is a copy.
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        ).execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "sea kayaking", "content": "weekends"}],
        )
        near = await CollectionGetTool(db).execute(
            CallLink.new_top(), collection="likes", key="sea kayakng"
        )
        assert "did you mean key 'sea kayaking'?" in near.message
        # The proven-win read guidance (collection_keys / find by meaning), AND the
        # rejection now closes the residual write-vs-update decision: once the model
        # finds the entry under a different key it must UPDATE that entry, not
        # collection_write it (which the dedup rejects as a duplicate).
        assert "collection_keys('likes')" in missing.message
        assert "find(query=<what you're looking for>, memory='likes')" in missing.message
        assert (
            "collection_write(collection='likes', entries=[{'key': <the key you found>, "
            "'content': <the new content>}])" in missing.message
        )
        assert "the store replaces the content of the key you name" in missing.message
        # A bracket-wrapped key (the model's ingrained habit from the old `[key]`
        # display form) is never silently resolved — it's rejected with a teaching
        # error that names the mistake, the current key='...' render, and the bare
        # key ready to reuse.
        bracketed = await CollectionGetTool(db).execute(
            CallLink.new_top(), collection="likes", key="[k]"
        )
        assert bracketed.success is False
        assert bracketed.message == (
            "Key '[k]' not found in 'likes'. The enclosing [brackets] are not part "
            "of the key — entry listings show keys as key='...' and the key is passed "
            "bare, without brackets. This entry's key is 'k'. Retry with key='k'."
        )
        # A bracket-wrapped key whose bare form doesn't exist either gets the
        # ordinary not-found error, not the bracket teaching rejection.
        double_miss = await CollectionGetTool(db).execute(
            CallLink.new_top(), collection="likes", key="[absent]"
        )
        assert "Key '[absent]' not found" in double_miss.message
        assert "Retry with" not in double_miss.message
        # A key that genuinely contains brackets exact-matches with no rejection.
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        ).execute(
            CallLink.new_top(),
            collection="likes",
            entries=[{"key": "[lit]", "content": "bracket literal body"}],
        )
        literal = await CollectionGetTool(db).execute(
            CallLink.new_top(), collection="likes", key="[lit]"
        )
        assert literal.success is True
        assert "bracket literal body" in literal.message

    @pytest.mark.asyncio
    async def test_keys_lists_unique_keys_in_order(self, db, mock_llm):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "first", "content": "1"}]
        )
        await write.execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "second", "content": "2"}]
        )
        listing = await CollectionKeysTool(db).execute(CallLink.new_top(), collection="likes")
        assert listing.message == "- first\n- second"

    @pytest.mark.asyncio
    async def test_keys_empty_collection_names_source_not_bare_sentinel(self, db):
        """An empty collection's keys read names the source and marks absence (not an
        error), rather than the bare "(no keys)" sentinel (house wording pass)."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        listing = await CollectionKeysTool(db).execute(CallLink.new_top(), collection="likes")
        assert listing.message == "No keys in `likes` — the collection is empty (not an error)."

    @pytest.mark.asyncio
    async def test_sample_returns_all_when_few(self, db, mock_llm):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "a", "content": "1"}]
        )
        rendered = await CollectionSampleTool(db).execute(
            CallLink.new_top(), collection="likes", k=5
        )
        assert "key='a' 1" in rendered.message


class TestCollectionMutations:
    @pytest.mark.asyncio
    async def test_write_replaces_content_under_the_existing_key(self, db, mock_llm):
        """A correction is the same write call under the key the entry is stored
        under: the change gate refreshes the stored baseline in place and reports
        the refresh — the one write, with the one check."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "k", "content": "old"}]
        )
        result = await write.execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "k", "content": "new"}]
        )
        assert "Updated 'k'. It held: \"old\"." in result.message
        assert "restore it with collection_write" in result.message
        assert db.memory("likes").get("k")[0].content == "new"
        # A bracket-wrapped key (display form copied from an entry list) on the
        # delete path is rejected with a teaching error naming the bare key —
        # never absorbed; the entry is untouched.
        rejected_delete = await CollectionRemoveTool(db).execute(
            CallLink.new_top(), collection="likes", key="[k]"
        )
        assert rejected_delete.success is False
        assert "Retry with key='k'" in rejected_delete.message
        assert db.memory("likes").get("k")[0].content == "new"

    @pytest.mark.asyncio
    async def test_archive_and_unarchive(self, db):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        assert (
            "Archived 'likes'"
            in (
                await CollectionArchiveTool(db).execute(CallLink.new_top(), collection="likes")
            ).message
        )
        assert (
            "Unarchived 'likes'"
            in (
                await CollectionUnarchiveTool(db).execute(CallLink.new_top(), collection="likes")
            ).message
        )


class TestDidYouMeanSuggestions:
    """Every miss leads with its nearest match — did-you-mean on memory-name and
    entry-key lookups (#1674).  The suggestion carries the anchor into the render
    so the fix is a copy, not another lookup; a miss with nothing close renders
    byte-identically to before, so no empty "did you mean" artifact appears."""

    @pytest.mark.asyncio
    async def test_memory_name_miss_leads_with_the_nearest_by_typo(self, db):
        """A mistyped memory name (the motivating 'aurora-deone' → 'aurora-deck-2')
        leads the not-found render with the nearest existing name — the string
        (typo) leg, universal because it needs no embedding client."""
        _seed_collection(db, name="aurora-deck-2", description="a listening deck")
        result = await CollectionListTool(db).execute(CallLink.new_top(), collection="aurora-deone")
        assert result.success is False
        assert result.message == (
            "Memory 'aurora-deone' not found — did you mean 'aurora-deck-2'? Check the "
            "name (it may be misspelled), or find it by meaning with "
            "find(query=<what it's about>) — it resolves your collections, logs, and "
            "skills (archived included) and names the exact tool for each. Or create it "
            "with collection_configure(collection='aurora-deone') / "
            "log_create(log='aurora-deone') if it should exist."
        )

    @pytest.mark.asyncio
    async def test_memory_name_miss_with_no_close_candidate_is_byte_identical(self, db):
        """A miss with nothing close renders EXACTLY today's message — no empty
        'did you mean' artifact, no changed punctuation."""
        _seed_collection(db, name="aurora-deck-2", description="a listening deck")
        result = await CollectionListTool(db).execute(CallLink.new_top(), collection="xyzzy")
        assert result.success is False
        assert result.message == (
            "Memory 'xyzzy' not found. Check the name (it may be misspelled), or find "
            "it by meaning with find(query=<what it's about>) — it resolves your "
            "collections, logs, and skills (archived included) and names the exact tool "
            "for each. Or create it with collection_configure(collection='xyzzy') / "
            "log_create(log='xyzzy') if it should exist."
        )

    @pytest.mark.asyncio
    async def test_memory_name_miss_leads_with_the_nearest_by_meaning(self, db, mock_llm):
        """When the typo leg finds nothing, an embedding-capable tool (log_append)
        falls back to the MEANING leg — the collection whose description anchor is
        closest to the missed name, clearing the reused dedup threshold."""
        _seed_collection(db, name="garage", description="tracking oil changes and tire rotations")
        tool = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        result = await tool.execute(
            CallLink.new_top(), log="oil changes and tire rotations", content="my car"
        )
        assert result.success is False
        assert result.message == (
            "Memory 'oil changes and tire rotations' not found — did you mean 'garage'? "
            "Check the name (it may be misspelled), or find it by meaning with "
            "find(query=<what it's about>) — it resolves your collections, logs, and "
            "skills (archived included) and names the exact tool for each. Or create it "
            "with collection_configure(collection='oil changes and tire rotations') / "
            "log_create(log='oil changes and tire rotations') if it should exist."
        )

    @pytest.mark.asyncio
    async def test_delete_entry_key_miss_leads_with_the_nearest_key(self, db, mock_llm):
        """Same key suggestion on collection_remove — the nearest key leads,
        the existing 'nothing to delete' guidance follows."""
        _seed_collection(db, name="notes")
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        await write.execute(
            CallLink.new_top(),
            collection="notes",
            entries=[{"key": "quarterly-report", "content": "x"}],
        )
        result = await CollectionRemoveTool(db).execute(
            CallLink.new_top(), collection="notes", key="quarterly-reprot"
        )
        assert result.mutated is False
        assert result.message == (
            "No entry with key 'quarterly-reprot' in 'notes' — did you mean key "
            "'quarterly-report'? Nothing to delete. List the current keys with "
            "collection_keys('notes') to find it."
        )
        # No close key → byte-identical to today's message.
        far = await CollectionRemoveTool(db).execute(
            CallLink.new_top(), collection="notes", key="zzzzz"
        )
        assert far.message == (
            "No entry with key 'zzzzz' in 'notes' — nothing to delete. "
            "List the current keys with collection_keys('notes') to find it."
        )


class TestLogTools:
    @pytest.mark.asyncio
    async def test_log_get_reads_one_entry_in_full(self, db, mock_llm):
        """``log_get`` returns one stored entry whole, by the id a log read shows
        beside it — the browse-results fetch-handle path, and any log the user
        created."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(CallLink.new_top(), log="events", content="first entry")
        await append.execute(CallLink.new_top(), log="events", content="second entry")
        entry = db.memories.memory("events").read_all()[1]
        result = await LogGetTool(db).execute(CallLink.new_top(), log="events", id=entry.id)
        assert result.success is True
        assert result.message == (
            f"1 entry from `events`:\n1. [{format_log_timestamp(entry.created_at)}] second entry"
        )

    @pytest.mark.asyncio
    async def test_log_get_read_shows_the_ids_its_log_renders(self, db, mock_llm):
        """The id ``log_get`` consumes is the one a log read of a REAL stored log
        renders beside each entry (``format_entries(show_ids=True)``) — display
        form == invocation form, one copy away."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(CallLink.new_top(), log="events", content="hello")
        entry = db.memories.memory("events").read_all()[0]
        rendered = await LogReadTool(db, "chat", scope=None).execute(
            CallLink.new_top(), log="events"
        )
        assert rendered.message == (
            f"1 entry from `events` (oldest first):\n"
            f"1. [{format_log_timestamp(entry.created_at)}] id={entry.id} hello"
        )
        result = await LogGetTool(db).execute(CallLink.new_top(), log="events", id=entry.id)
        assert "hello" in result.message

    @pytest.mark.asyncio
    async def test_log_get_refuses_the_non_id_addressable_stores(self, db, mock_llm):
        """Each store that isn't an id-addressable log is refused by name, saying how
        it IS read: a collection by key, messages and run records through
        ``log_read``.  The refusals name no tool but ``log_read``, so a program whose
        only route to ``log_get`` is a stored page keeps a closed surface."""
        _seed_collection(db, name="notes", description="x")
        db.memories.create_log(PennyConstants.MEMORY_USER_MESSAGES_LOG, "inbound")
        db.memories.create_log(PennyConstants.MEMORY_COLLECTOR_RUNS_LOG, "audit")
        tool = LogGetTool(db)
        collection = await tool.execute(CallLink.new_top(), log="notes", id=1)
        assert collection.success is False
        assert collection.message == (
            "'notes' is a collection, not a log: its entries are read by key, not by id."
        )
        user_log = PennyConstants.MEMORY_USER_MESSAGES_LOG
        messages = await tool.execute(CallLink.new_top(), log=user_log, id=1)
        assert messages.success is False
        assert messages.message == (
            f"'{user_log}' holds messages, and a message is not read by id. "
            f"Read them with log_read('{user_log}')."
        )
        run_log = PennyConstants.MEMORY_COLLECTOR_RUNS_LOG
        runs = await tool.execute(CallLink.new_top(), log=run_log, id=1)
        assert runs.success is False
        assert runs.message == (
            f"'{run_log}' holds run records, and a run is read by the `run <id>` shown "
            f"beside it, not by an entry id. Read them with log_read('{run_log}')."
        )

    @pytest.mark.asyncio
    async def test_log_get_unknown_id_names_the_log_read(self, db, mock_llm):
        """An id the log holds no entry for is refused with the read that shows
        the ids that DO exist."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        result = await LogGetTool(db).execute(CallLink.new_top(), log="events", id=99)
        assert result.success is False
        assert result.message == (
            "No entry 99 in 'events'. Read it with log_read('events') to see its "
            "entries and their ids."
        )

    @pytest.mark.asyncio
    async def test_log_get_unknown_log_gets_the_did_you_mean_refusal(self, db, mock_llm):
        """An unknown log name raises ``MemoryNotFoundError``, which the base turns
        into the standard not-found message — no separate refusal here."""
        result = await LogGetTool(db).execute(CallLink.new_top(), log="evnts", id=1)
        assert result.success is False
        assert "Memory 'evnts' not found" in result.message

    @pytest.mark.asyncio
    async def test_format_entries_show_ids_stays_off_other_renders(self, db, mock_llm):
        """``show_ids`` renders the id only where it is passed: a message facade's
        read (and every other entry render) stays byte-identical to before."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            CallLink.new_top(), log="events", content="hello"
        )
        entry = db.memories.memory("events").read_all()[0]
        plain = format_entries([entry], source="events", ordering="oldest first")
        assert f"id={entry.id}" not in plain
        with_ids = format_entries([entry], source="events", show_ids=True)
        assert f"id={entry.id}" in with_ids

    @pytest.mark.asyncio
    async def test_collection_list_refuses_a_log(self, db, mock_llm):
        """Collection reads error on a log instead of silently bypassing the
        cursored log_read/log_get interface (the read_latest-on-a-log footgun)."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            CallLink.new_top(), log="events", content="first"
        )
        rendered = await CollectionListTool(db).execute(CallLink.new_top(), collection="events")
        assert "Refused" in rendered.message
        assert "log_read" in rendered.message

    @pytest.mark.asyncio
    async def test_read_latest_rejects_zero_count(self, db, mock_llm):
        """``k=0`` (a model guessing zero means "unlimited") reads no entries, so
        the tool would look empty — the arg model refuses it before execute with
        an actionable message (omit k for all), via the Tool.run gate."""
        _seed_collection(
            db,
            name="notes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        ).execute(
            CallLink.new_top(), collection="notes", entries=[{"key": "a", "content": "first"}]
        )
        rejected = await CollectionListTool(db).run(CallLink.new_top(), collection="notes", k=0)
        assert rejected.success is False
        assert "at least 1" in rejected.message
        assert "Omit k" in rejected.message
        # A valid read still returns the entry (the guard only rejects k < 1).
        ok = await CollectionListTool(db).run(CallLink.new_top(), collection="notes")
        assert "first" in ok.message

    @pytest.mark.asyncio
    async def test_log_read_window_mode(self, db, mock_llm):
        """A non-collector caller (scope=None) gets window-mode log_read: recent
        entries within the fixed look-back window — no cursor, no count arg."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(CallLink.new_top(), log="events", content="hello")
        rendered = await LogReadTool(db, "chat", scope=None).execute(
            CallLink.new_top(), log="events"
        )
        assert "hello" in rendered.message
        # A blank append is refused (blank-only — a bare URL is still a valid log
        # entry), so nothing degenerate joins the stream.  The non-blank rule lives
        # on LogAppendArgs.content, so the refusal comes from the Tool.run gate.
        blank = await append.run(CallLink.new_top(), log="events", content="   ")
        assert blank.success is False
        assert "blank" in blank.message

    @pytest.mark.asyncio
    async def test_collector_runs_log_renders_runs_from_promptlog(self, db):
        """collector-runs is a read facade over promptlog: log_read renders each
        worked run as a record (``[target] summary`` + its tool trace) — no
        stored entries, no keys, no get.  This is the quality collector's review."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="collector-runs", description="audit"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c0",
                                "type": "function",
                                "function": {
                                    "name": "send_message",
                                    "arguments": '{"content": "Found a new grinder, $300."}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=response,
            agent_name="collector",
            run_id="run-42",
            run_target="espresso-gear",
        )
        db.messages.set_run_outcome("run-42", "worked", "sent an update about a grinder")

        rendered = await LogReadTool(db, "quality", scope="quality").execute(
            CallLink.new_top(), log="collector-runs"
        )

        # collector-runs reads through the uniform log formatter now (it's a log
        # facade like any other) — framed as a fetched batch, runs as records.
        assert "from `collector-runs`" in rendered.message
        assert "[espresso-gear] sent an update about a grinder" in rendered.message
        assert "Found a new grinder, $300." in rendered.message  # the exact message, untruncated

    @pytest.mark.asyncio
    async def test_run_list_renders_by_source(self, db):
        """run_list is the SEQUENCE lens over runs, orthogonal to source: a
        collector's name renders its runs as ``[target] -> tools -> done``; ``chat``
        renders conversations as ``user -> tools -> penny``.  The valid sources are
        discovered from the DB into its description."""
        _seed_collection(
            db,
            name="espresso-gear",
            description="x",
            extraction_prompt="1. browse for new espresso gear. 2. write it. 3. done().",
            schedule="FREQ=HOURLY",
        )
        # One completed collector run for espresso-gear.
        coll_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "collection_write",
                                    "arguments": '{"memory": "espresso-gear", '
                                    '"entries": [{"content": "Niche grinder"}]}',
                                }
                            },
                            # A worked run closes with the argless done() (#1569) —
                            # without it the run would (correctly) read as a
                            # write-gate STOP (#1587).
                            {
                                "function": {
                                    "name": "done",
                                    "arguments": "{}",
                                }
                            },
                        ]
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=coll_resp,
            agent_name="collector",
            run_id="coll-1",
            run_target="espresso-gear",
        )
        db.messages.set_run_outcome("coll-1", "worked", "wrote a new grinder")
        # One chat run: user message + a tool call, then Penny's reply.
        user_turn = f"live{PennyConstants.SECTION_SEPARATOR}find me a grinder"
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": user_turn}],
            response={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "browse",
                                        "arguments": '{"queries": ["espresso grinder"]}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            agent_name=PennyConstants.CHAT_AGENT_NAME,
            run_id="chat-1",
        )
        db.messages.log_prompt(
            model="m",
            messages=[],
            response={"choices": [{"message": {"content": "Here's a good grinder."}}]},
            agent_name=PennyConstants.CHAT_AGENT_NAME,
            run_id="chat-1",
        )

        tool = ReadRunCallsTool(db, "quality")
        # Sources discovered from the DB (chat + the collector) are in the description.
        assert "espresso-gear" in tool.description
        assert "chat" in tool.description

        collector = await tool.run(CallLink.new_top(), source="espresso-gear")
        assert "[espresso-gear]" in collector.message
        assert "collection_write(collection='espresso-gear'" in collector.message
        # Argless done() → the conclusion is the run's STRUCTURAL outcome (#1569),
        # never a model summary.
        assert "done: worked" in collector.message

        chat = await tool.run(CallLink.new_top(), source="chat")
        assert "user: find me a grinder" in chat.message
        assert "browse(['espresso grinder'])" in chat.message
        assert "penny: Here's a good grinder." in chat.message
        # Each rendered run names its own id, so the surface is an anchor: a reader
        # can reference the run it's inspecting rather than guess it (#1560).
        assert "run chat-1" in chat.message

        # An unknown/typo'd source resolves to a failed, actionable refusal that
        # names the offending value — not a silent empty batch that reads as "this
        # collector has no runs".
        unknown = await tool.run(CallLink.new_top(), source="esspreso-gear")
        assert unknown.success is False
        assert "esspreso-gear" in unknown.message

    @staticmethod
    def _log_run(db, *, run_id: str, target: str, summary: str, write_key: str) -> None:
        """Persist one completed collector run for ``target`` (a write + its
        ``done`` summary) — the promptlog rows ``run_get`` / ``run_list``
        render.  ``response`` is a dict (``log_prompt`` serializes it); the inner
        tool-call ``arguments`` is itself a JSON string, as the model emits it."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "w0",
                                "type": "function",
                                "function": {
                                    "name": "collection_write",
                                    "arguments": json.dumps(
                                        {
                                            "memory": target,
                                            "entries": [{"key": write_key, "content": "v"}],
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=response,
            agent_name="collector",
            run_id=run_id,
            run_target=target,
        )
        db.messages.set_run_outcome(run_id, "worked", summary)

    @staticmethod
    async def _create_collection(db, name: str) -> None:
        _seed_collection(
            db,
            name=name,
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )

    @pytest.mark.asyncio
    async def test_run_get_resolves_a_run_id_to_its_canonical_projection(self, db):
        """``run_get(run='run <id>')`` consumes the header's typed run anchor
        VERBATIM and returns exactly the run's canonical tool-call projection — the
        same ``render_run_calls`` view ``run_list`` renders, but for the single
        run the id names (#1580, the run-id ↔ target anchor unification).  The result
        leads with the run's own id, so the rendered anchor and the argument are one."""
        await self._create_collection(db, "ai-news")
        self._log_run(db, run_id="news-1", target="ai-news", summary="wrote 1", write_key="mixtral")

        result = await GetEventTool(db).run(CallLink.new_top(), run="run news-1")

        # Whole-render: the tool returns the canonical projection byte-for-byte.
        expected = render_run_calls(db.messages.get_run_prompts("news-1"))
        assert result.message == expected
        assert result.message.startswith("run news-1")
        assert "collection_write(collection='ai-news'" in result.message

    @pytest.mark.asyncio
    async def test_run_get_lists_a_draw_made_on_the_runs_behalf(self, db):
        """``run_get`` nests the model call a micro-context made FOR this run's own
        ``browse`` step under that step (#2355 part D) — the unfinished extraction
        draw the motivating incident lost entirely, now one call away alongside the
        tool call that dispatched it."""
        await self._create_collection(db, "ai-news")
        self._log_run(db, run_id="news-1", target="ai-news", summary="wrote 1", write_key="mixtral")
        # A browse step the run itself made, keyed "b0" (distinct from the write's
        # own "w0"), plus the nested draw browse dispatched on its behalf — mints its
        # own run_id for its own ledger attribution, linked back via parent_run_id /
        # parent_tool_call_id rather than sharing "news-1".
        db.messages.log_prompt(
            model="m",
            messages=[],
            response={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "b0",
                                    "type": "function",
                                    "function": {
                                        "name": "browse",
                                        "arguments": json.dumps(
                                            {"queries": ["https://example.test"], "extract": "x"}
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            agent_name="collector",
            run_id="news-1",
            run_target="ai-news",
        )
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=None,
            agent_name="browse-extract",
            run_id="draw-1",
            parent_run_id="news-1",
            parent_tool_call_id="b0",
            call_outcome="cancelled",
            duration_ms=300_000,
        )

        result = await GetEventTool(db).run(CallLink.new_top(), run="run news-1")

        lines = result.message.splitlines()
        browse_index = next(i for i, line in enumerate(lines) if "browse(" in line)
        assert (
            "model call: browse-extract — cancelled after 300000ms (5 minutes)"
            in lines[browse_index + 1]
        )

    @pytest.mark.asyncio
    async def test_run_get_tolerates_the_paren_framed_mutation_anchor(self, db):
        """The mutation activity line renders its causing run as ``(run <id>)``; the
        parse strips the framing so BOTH rendered forms resolve to the same run — the
        rendered token is consumable verbatim whichever line it came from (#1580)."""
        await self._create_collection(db, "ai-news")
        self._log_run(db, run_id="news-1", target="ai-news", summary="wrote 1", write_key="mixtral")

        bare = await GetEventTool(db).run(CallLink.new_top(), run="run news-1")
        framed = await GetEventTool(db).run(CallLink.new_top(), run="(run news-1)")
        assert framed.message == bare.message
        assert framed.success is True

    @pytest.mark.asyncio
    async def test_run_get_untyped_id_is_actionable(self, db):
        """An id with no recognised type tag is refused, not silently emptied — the
        message names what IS addressable and the guess-free fallbacks (find +
        the activity block), never a bare 'not found' (#1580)."""
        result = await GetEventTool(db).run(CallLink.new_top(), run="news-1")
        assert result.success is False
        assert "news-1" in result.message
        assert "find(query=" in result.message

    @pytest.mark.asyncio
    async def test_run_get_unknown_run_is_actionable(self, db):
        """A well-formed run token that matched no recorded run gets a failed,
        actionable refusal naming the id + where valid ids are listed — never an
        empty read that reads as a clean, call-less run (#1580)."""
        result = await GetEventTool(db).run(CallLink.new_top(), run="run does-not-exist")
        assert result.success is False
        assert "does-not-exist" in result.message
        assert "run_list" in result.message

    @pytest.mark.asyncio
    async def test_append_to_system_log_is_refused(self, db, mock_llm):
        """Invariant #1: the four framework-managed system logs are written
        only by Python side-effects.  ``log_append`` from any agent gets a
        readable refusal and writes nothing — guarding the conversation-turn
        reconstruction and the run audit trail from model-authored entries."""
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        # The reserved-target check is a pure constant lookup, so it lives on
        # LogAppendArgs.memory and the refusal comes from the Tool.run gate.
        for system_log in PennyConstants.SYSTEM_LOGS:
            result = await append.run(CallLink.new_top(), log=system_log, content="forged turn")
            assert result.success is False
            assert system_log in result.message
            assert "system log" in result.message  # what went wrong + how to fix
        # Nothing was created/written — the refusal short-circuits before the store.
        assert db.memories.get(PennyConstants.MEMORY_PENNY_MESSAGES_LOG) is None

    @pytest.mark.asyncio
    async def test_log_similar_with_client(self, db, mock_llm):
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        client = _make_llm_client(mock_llm)
        await LogAppendTool(db, client, author="test").execute(
            CallLink.new_top(), log="events", content="coffee is great"
        )
        # Anchor shares words with the entry so the bag-of-words mock embedding gives
        # meaningful cosine and the entry ranks in `Memory.read_similar` — the store
        # method `read_similar` (the tool) used to wrap; the tool is gone, the method
        # stays, read by `notification.py` and by `find`'s message candidates alike.
        vec = await embed_text(client, "coffee morning")
        assert vec is not None
        similar = db.memory("events").read_similar(vec)
        assert any("coffee is great" in entry.content for entry in similar)

    @pytest.mark.asyncio
    async def test_read_next_returns_all_entries_when_no_cursor(self, db, mock_llm):
        """Without a stored cursor, read_next returns every entry in the log."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(CallLink.new_top(), log="events", content="first")
        await append.execute(CallLink.new_top(), log="events", content="second")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await read_next.execute(CallLink.new_top(), log="events")

        assert "first" in rendered.message
        assert "second" in rendered.message

    @pytest.mark.asyncio
    async def test_commit_pending_advances_cursor_to_max_seen(self, db, mock_llm):
        """commit_pending writes the highest timestamp seen during the run."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(CallLink.new_top(), log="events", content="first")
        await append.execute(CallLink.new_top(), log="events", content="second")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(CallLink.new_top(), log="events")
        read_next.commit_pending()

        # A new instance after commit should see no entries (cursor caught up).
        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(CallLink.new_top(), log="events")
        # Empty read names the source and marks it as absence, not an error.
        assert (
            rendered.message
            == "No entries in `events` — it's empty or nothing matched (not an error)."
        )

    @pytest.mark.asyncio
    async def test_discard_pending_leaves_cursor_unchanged(self, db, mock_llm):
        """discard_pending drops the in-memory state without touching the DB cursor."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(CallLink.new_top(), log="events", content="first")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(CallLink.new_top(), log="events")
        read_next.discard_pending()

        # Cursor still at None; a new read sees the same entries.
        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(CallLink.new_top(), log="events")
        assert "first" in rendered.message

    @pytest.mark.asyncio
    async def test_first_cycle_bounded_to_latest_n_entries(self, db, mock_llm):
        """A brand-new collector (no cursor yet) reading a busy log gets the
        most-recent N entries, not every entry since the dawn of time.

        Without this bound, a fresh collector reading ``user-messages`` (which
        has months of chat history in production) would dump the entire log
        into the first cycle's context.
        """
        from penny.constants import PennyConstants

        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        # Append more entries than the bound to confirm trimming
        n_entries = PennyConstants.LOG_READ_LIMIT + 5
        for i in range(n_entries):
            await append.execute(CallLink.new_top(), log="events", content=f"entry-{i:02d}")

        read_next = LogReadTool(db, agent_name="brand-new-collector", scope="brand-new-collector")
        rendered = await read_next.execute(CallLink.new_top(), log="events")

        # Exactly the latest N entries — entry-(n-N) through entry-(n-1)
        # should appear; older entries should not.
        for i in range(n_entries - PennyConstants.LOG_READ_LIMIT, n_entries):
            assert f"entry-{i:02d}" in rendered.message
        # The first 5 entries must be excluded
        assert "entry-00" not in rendered.message
        assert "entry-04" not in rendered.message

    @pytest.mark.asyncio
    async def test_first_cycle_advances_cursor_so_next_cycle_sees_only_new(self, db, mock_llm):
        """After a bounded first cycle commits, the next cycle picks up
        incrementally — even entries that the first cycle's bound excluded
        stay excluded (since they're older than the cursor)."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        for i in range(15):
            await append.execute(CallLink.new_top(), log="events", content=f"old-{i:02d}")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(CallLink.new_top(), log="events")
        read_next.commit_pending()

        # New entries arrive
        await append.execute(CallLink.new_top(), log="events", content="new-after-cursor")

        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(CallLink.new_top(), log="events")
        assert "new-after-cursor" in rendered.message
        # Old entries excluded by the bound stay excluded
        assert "old-00" not in rendered.message

    @pytest.mark.asyncio
    async def test_cursor_read_is_capped_and_advances_by_batch(self, db, mock_llm):
        """With a cursor established, a backlog larger than the batch bound is
        returned in bounded chunks — read N, cursor advances by N, the next read
        picks up the next N.  The caller never reasons about a count."""
        from penny.constants import PennyConstants

        limit = PennyConstants.LOG_READ_LIMIT
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")

        # Establish a cursor, then pile up a backlog bigger than one batch.
        await append.execute(CallLink.new_top(), log="events", content="seed")
        seed_read = LogReadTool(db, agent_name="extractor", scope="extractor")
        await seed_read.execute(CallLink.new_top(), log="events")
        seed_read.commit_pending()

        backlog = limit + 3
        for i in range(backlog):
            await append.execute(CallLink.new_top(), log="events", content=f"backlog-{i:02d}")

        first = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered_first = await first.execute(CallLink.new_top(), log="events")
        first.commit_pending()
        # Exactly one batch — the oldest N of the backlog, not all of it.
        assert rendered_first.message.count("backlog-") == limit
        assert "backlog-00" in rendered_first.message
        assert f"backlog-{backlog - 1:02d}" not in rendered_first.message

        # The next read picks up the remainder since the advanced cursor.
        second = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered_second = await second.execute(CallLink.new_top(), log="events")
        assert f"backlog-{backlog - 1:02d}" in rendered_second.message

    @pytest.mark.asyncio
    async def test_per_agent_cursors_are_independent(self, db, mock_llm):
        """Two agents reading the same log have independent cursor state."""
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            CallLink.new_top(), log="events", description="x"
        )
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            CallLink.new_top(), log="events", content="hello"
        )

        agent_a = LogReadTool(db, agent_name="a", scope="a")
        await agent_a.execute(CallLink.new_top(), log="events")
        agent_a.commit_pending()

        # Agent B has its own cursor and still sees the entry.
        agent_b = LogReadTool(db, agent_name="b", scope="b")
        rendered = await agent_b.execute(CallLink.new_top(), log="events")
        assert "hello" in rendered.message


class TestEmissionProvenanceRender:
    """The messagelog facade renders emission provenance inline (#1568).

    An outgoing row stamped with the mechanism that sent it reads
    ``(sent by <mechanism>) <content>`` between the timestamp and content on
    every read path (all facade reads synthesize entries through ``_to_entry``)
    — so pulling a message by recency or relevance IS resolving its source.  A
    NULL-mechanism row (a direct reply) renders byte-identical to the
    pre-provenance shape — the unmarked case stays the quiet default."""

    def _seed(self, db: Database) -> None:
        db.memories.create_log(
            PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
            "outbound messages",
        )
        with Session(db.engine) as session:
            session.add(
                MessageLog(
                    direction=PennyConstants.MessageDirection.OUTGOING,
                    sender="penny",
                    content="sure, happy to help!",
                    embedding=serialize_embedding([0.0, 1.0, 0.0]),
                    timestamp=datetime(2026, 7, 2, 9, 10, tzinfo=UTC),
                )
            )
            session.add(
                MessageLog(
                    direction=PennyConstants.MessageDirection.OUTGOING,
                    sender="penny",
                    content="Heads up: the price dropped to $42!",
                    mechanism="price-watch",
                    embedding=serialize_embedding([1.0, 0.0, 0.0]),
                    timestamp=datetime(2026, 7, 2, 9, 14, tzinfo=UTC),
                )
            )
            session.commit()

    @pytest.mark.asyncio
    async def test_log_read_renders_provenance_whole(self, db):
        """The whole log_read render, both shapes pinned: the direct reply line is
        byte-identical to the pre-provenance shape (``N. [stamp] content``); the
        mechanism-stamped line carries the inline marker."""
        self._seed(db)
        result = await LogReadTool(db, agent_name="reader", scope="reader").execute(
            CallLink.new_top(), log=PennyConstants.MEMORY_PENNY_MESSAGES_LOG
        )
        assert result.message == (
            "2 entries from `penny-messages` (oldest first):\n"
            "1. [2026-07-02 09:10 UTC] sure, happy to help!\n"
            "2. [2026-07-02 09:14 UTC] (sent by price-watch) Heads up: the price dropped to $42!"
        )

    def test_read_similar_carries_provenance_too(self, db):
        """The similarity path returns the same synthesized entries, so a
        relevance hit on an autonomous send carries its mechanism inline — one
        call resolves both the message and its source."""
        self._seed(db)
        facade = db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG)
        assert facade is not None
        hits = facade.read_similar([1.0, 0.0, 0.0], k=1)
        assert [hit.content for hit in hits] == [
            "(sent by price-watch) Heads up: the price dropped to $42!"
        ]


class TestMemoryNameNormalization:
    @pytest.mark.asyncio
    async def test_unicode_hyphen_in_memory_name_normalized(self, db, mock_llm):
        """Regression: gpt-oss occasionally emits Unicode dashes (U+2010,
        U+2011, …) where ASCII hyphen-minus is expected, breaking string
        comparison in tool args.  Memory-name fields normalise on the way
        in so the rest of the stack sees the canonical form."""
        _seed_collection(
            db,
            name="board-games",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="test", decider=_NO_MODEL
        )
        # Non-breaking hyphen U+2011 in the collection name — model output
        # observed in the wild.
        result = await write.execute(
            CallLink.new_top(),
            collection="board‑games",
            entries=[{"key": "k", "content": "v"}],
        )
        assert "Wrote 1 entry to 'board-games'" in result.message


class TestAuthorAttribution:
    @pytest.mark.asyncio
    async def test_writes_stamp_constructor_author(self, db, mock_llm):
        """Author is bound at tool construction (not pulled from ambient state)."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="preference-extractor", decider=_NO_MODEL
        ).execute(CallLink.new_top(), collection="likes", entries=[{"key": "k", "content": "v"}])

        rows = db.memory("likes").get("k")
        assert rows[0].author == "preference-extractor"


class TestFactory:
    """One uniform surface for every agent — reads + lifecycle (shape) + entry
    mutations (contents).  Capability is no longer curated by omission; the
    only per-agent difference is ``scope``, which drives the collector-binding
    *invariant* (see TestScopedFactory), not which tools are present.
    """

    _FULL_SURFACE = {
        # Reads
        "collection_list",
        "collection_get",
        "collection_sample",
        "collection_keys",
        "memory_get",
        "log_read",
        "log_get",
        "run_list",
        "run_get",
        "find",
        # Lifecycle (shape)
        "collection_configure",
        "collection_archive",
        "collection_unarchive",
        "log_create",
        # Entry mutations (contents)
        "collection_write",
        "collection_remove",
        "log_append",
    }

    def test_every_tool_advertises_the_arguments_it_validates(self, db, mock_llm):
        """A tool's schema is what the model is shown; its args model is what the call
        is validated against.  They must name the same arguments, or the model is told
        to pass one name and refused for not passing another."""
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="chat", decider=_NO_MODEL
        )
        for tool in tools:
            advertised = set(tool.parameters.get("properties", {}))
            validated = set(tool.args_model.model_fields)
            assert advertised == validated, (
                f"{tool.name}: schema names {sorted(advertised)}, "
                f"args model validates {sorted(validated)}"
            )

    def test_chat_surface_is_the_full_set(self, db, mock_llm):
        """Chat (scope=None) gets every memory tool — entry mutations included,
        unrestricted, since edits are user-directed."""
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="chat", decider=_NO_MODEL
        )
        assert {tool.name for tool in tools} == self._FULL_SURFACE

    def test_collector_surface_is_the_same_full_set(self, db, mock_llm):
        """A bound collector (scope=X) gets the identical surface — scope binds
        its entry mutations to X but does not strip lifecycle/other tools."""
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="collector", decider=_NO_MODEL, scope="likes"
        )
        assert {tool.name for tool in tools} == self._FULL_SURFACE


class TestScopedFactory:
    """Scope binds a collector to one collection.  Writes to other collections
    get a clean refusal at the tool layer, so a confused or jailbroken
    collector can't trash unrelated memories.
    """

    @pytest.mark.asyncio
    async def test_scoped_write_rejects_other_collection(self, db, mock_llm):
        """A scoped collector that tries to write to a different collection
        gets a clean refusal rather than silently corrupting unrelated data."""
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )
        _seed_collection(
            db,
            name="dislikes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )

        write = CollectionWriteTool(
            db,
            _make_llm_client(mock_llm),
            author="collector:likes",
            decider=_NO_MODEL,
            scope="likes",
        )
        result = await write.execute(
            CallLink.new_top(), collection="dislikes", entries=[{"key": "k", "content": "v"}]
        )

        assert (
            result.message == "Refused: this collector can only write to 'likes', not 'dislikes'."
        )
        # And nothing was actually written
        assert db.memory("dislikes").get("k") == []

    @pytest.mark.asyncio
    async def test_scoped_write_allows_target_collection(self, db, mock_llm):
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            schedule="FREQ=HOURLY",
        )

        write = CollectionWriteTool(
            db,
            _make_llm_client(mock_llm),
            author="collector:likes",
            decider=_NO_MODEL,
            scope="likes",
        )
        result = await write.execute(
            CallLink.new_top(), collection="likes", entries=[{"key": "k", "content": "v"}]
        )

        assert "Wrote 1 entry" in result.message
        assert db.memory("likes").get("k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_delete_rejects_other_collection(self, db):
        delete = CollectionRemoveTool(db, scope="likes")
        result = await delete.execute(CallLink.new_top(), collection="dislikes", key="k")
        assert "Refused" in result.message

    @pytest.mark.asyncio
    async def test_scoped_write_to_rules_is_refused(self, db, mock_llm):
        """`rules` renders in full in every prompt as standing instructions, so a
        job cycle must not be able to write itself into every prompt (#2633): a
        collector-scoped write targeting it is refused with the honest answer and
        nothing lands."""
        _seed_collection(
            db,
            name=PennyConstants.MEMORY_RULES_COLLECTION,
            description="Standing instructions the user has given",
            extraction_prompt=None,
            schedule=None,
        )
        write = CollectionWriteTool(
            db,
            _make_llm_client(mock_llm),
            author="collector:likes",
            decider=_NO_MODEL,
            scope="likes",
        )
        result = await write.execute(
            CallLink.new_top(), collection="rules", entries=[{"key": "k", "content": "v"}]
        )
        assert result.message == _RULES_GUARD_REFUSAL
        assert not result.success
        assert db.memory("rules").get("k") == []

    @pytest.mark.asyncio
    async def test_scoped_remove_from_rules_is_refused(self, db, mock_llm):
        """The same guard on the remove path — a job cycle cannot take a standing
        rule out either."""
        _seed_collection(
            db,
            name=PennyConstants.MEMORY_RULES_COLLECTION,
            description="Standing instructions the user has given",
            extraction_prompt=None,
            schedule=None,
        )
        seed = CollectionWriteTool(db, _make_llm_client(mock_llm), author="chat", decider=_NO_MODEL)
        await seed.execute(
            CallLink.new_top(),
            collection=PennyConstants.MEMORY_RULES_COLLECTION,
            entries=[{"key": "always-x", "content": "always do x"}],
        )
        delete = CollectionRemoveTool(db, scope="likes")
        result = await delete.execute(
            CallLink.new_top(), collection=PennyConstants.MEMORY_RULES_COLLECTION, key="always-x"
        )
        assert result.message == _RULES_GUARD_REFUSAL
        assert not result.success
        assert db.memory(PennyConstants.MEMORY_RULES_COLLECTION).get("always-x") != []

    @pytest.mark.asyncio
    async def test_chat_turn_write_to_rules_still_lands(self, db, mock_llm):
        """The chat agent's own writes (scope=None) are the one path rules are
        kept by — unchanged by the guard."""
        _seed_collection(
            db,
            name=PennyConstants.MEMORY_RULES_COLLECTION,
            description="Standing instructions the user has given",
            extraction_prompt=None,
            schedule=None,
        )
        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="chat", decider=_NO_MODEL
        )
        result = await write.execute(
            CallLink.new_top(),
            collection=PennyConstants.MEMORY_RULES_COLLECTION,
            entries=[{"key": "always-x", "content": "always do x"}],
        )
        assert result.success
        assert "Wrote 1 entry" in result.message
        assert db.memory(PennyConstants.MEMORY_RULES_COLLECTION).get("always-x") != []


class TestRegistryProvenanceAndLifecycle:
    """Operational registry (#1566): ``collection_configure`` stamps the spawning
    message + creating run; ``memory_get`` renders a status / expires /
    created-from-message block — so any mechanism can state who asked for it,
    what created it, whether it's live, and when it ends, and an archived one
    stays inspectable by its exact name.
    """

    async def _create(self, db, name, *, created_by_run_id=None, notify=False):
        # Instantiate through the real front door: a plain prompt lands straight
        # into the collection, and the provenance (creating run) is stamped.
        return await CollectionCreateTool(
            db, cast(Any, MockLlmClient()), created_by_run_id=created_by_run_id
        ).execute(
            CallLink.new_top(),
            name=name,
            description=f"the user's goal for {name}",
            prompt="Browse for fresh items on the topic and save the newest ones.",
            schedule="FREQ=HOURLY",
            notify=notify,
        )

    def _spawn(self, db, run_id, content):
        """Log an incoming user message and link it to ``run_id`` — mirroring the
        channel's post-run provenance link (the message id isn't known until the
        run returns)."""
        message_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, "user", content
        )
        db.memories.link_source_message(run_id, message_id)
        return message_id

    @pytest.mark.asyncio
    async def test_create_stamps_run_then_channel_links_message(self, db):
        # collection_configure stamps only the creating run — the spawning message
        # isn't known until the run returns, and there's no end condition.
        await self._create(db, "espresso-reviews", created_by_run_id="run-espresso-01")
        row = db.memories.get("espresso-reviews")
        assert row.created_by_run_id == "run-espresso-01"
        assert row.source_message_id is None
        assert row.expires_at is None
        # The channel then links the spawning message by run id.
        message_id = self._spawn(
            db, "run-espresso-01", "can you keep an eye on new espresso machine reviews?"
        )
        assert db.memories.get("espresso-reviews").source_message_id == message_id

    @pytest.mark.asyncio
    async def test_metadata_renders_full_lifecycle(self, db):
        await self._create(db, "audiobooks", created_by_run_id="run-audiobooks-01")
        message_id = self._spawn(
            db, "run-audiobooks-01", "keep a running list of good sci-fi audiobooks"
        )
        result = await MemoryMetadataTool(db).execute(CallLink.new_top(), memory="audiobooks")
        # One exact-name call answers the whole lifecycle question.
        assert "status: active" in result.message
        assert "expires: never" in result.message
        assert "by run run-audiobooks-01" in result.message
        assert f'from message {message_id} ("keep a running list of good sci-fi audiobooks")' in (
            result.message
        )

    @pytest.mark.asyncio
    async def test_archived_collection_stays_metadata_inspectable(self, db):
        """Archiving changes status, never reachability (#1566/#2288): a collection
        archived by exact name still answers ``memory_get`` fully, clearly
        marked, and the self-state header's compact archived line (#2288) is the
        wider index that names it alongside every other archived store."""
        await self._create(db, "kickstarters", created_by_run_id="run-ks-01")
        message_id = self._spawn(db, "run-ks-01", "watch for new board game kickstarters")
        # Archiving must change status, never reachability by name.
        await CollectionArchiveTool(db).execute(CallLink.new_top(), collection="kickstarters")
        result = await MemoryMetadataTool(db).execute(CallLink.new_top(), memory="kickstarters")

        assert "status: archived" in result.message
        # Provenance (message + ask excerpt) renders on the created line.
        assert (
            f'from message {message_id} ("watch for new board game kickstarters")' in result.message
        )

    @pytest.mark.asyncio
    async def test_long_ask_renders_whole(self, db):
        long_ask = (
            "please keep a really thorough running list of every single new mechanical "
            "keyboard group buy you can find anywhere on the internet, forever"
        )
        await self._create(db, "keyboards", created_by_run_id="run-kb-01")
        self._spawn(db, "run-kb-01", long_ask)
        result = await MemoryMetadataTool(db).execute(CallLink.new_top(), memory="keyboards")
        # The ask renders WHOLE — no invented truncation (#2642): the exact user
        # words that spawned the job are what the render shows.
        assert long_ask in result.message

    @pytest.mark.asyncio
    async def test_expires_at_renders_end_condition(self, db):
        expiry = datetime(2026, 12, 25, 9, 0, tzinfo=UTC)
        db.memories.create_collection(
            "holiday-watch",
            "seasonal watch subject matter",
            extraction_prompt=("1. gather holiday deals.\n2. done()."),
            schedule="FREQ=HOURLY",
            expires_at=expiry,
        )
        result = await MemoryMetadataTool(db).execute(CallLink.new_top(), memory="holiday-watch")
        # A set end condition renders on the user's own clock (#2206/#2251 — UTC here,
        # since no profile is seeded), not "never".
        assert "expires: Friday, December 25, 2026 at 09:00 AM UTC" in result.message

    @pytest.mark.asyncio
    async def test_schedule_renders_in_plain_words_with_its_exact_rule_in_metadata(self, db):
        """Every collection's metadata carries its schedule in plain words, with the
        EXACT rule alongside in a code span (#2425) — #1857's display == invocation
        contract stays true (the rule is still there, copyable, on one line even for a
        two-line DTSTART+RRULE shape), it just no longer stands alone as the text a
        reply would echo verbatim."""
        _seed_collection(
            db,
            name="one-off",
            extraction_prompt="1. check it.",
            schedule="DTSTART:20261225T090000Z\nFREQ=DAILY;COUNT=1",
        )
        _seed_collection(
            db,
            name="plain",
            extraction_prompt="1. check it.",
            schedule="FREQ=HOURLY",
        )
        tool = MemoryMetadataTool(db)
        one_off = await tool.execute(CallLink.new_top(), memory="one-off")
        plain = await tool.execute(CallLink.new_top(), memory="plain")
        assert (
            "schedule: once, at Friday, December 25, 2026 at 09:00 AM UTC, then it "
            "retires (rule: `DTSTART:20261225T090000Z FREQ=DAILY;COUNT=1`)"
        ) in one_off.message
        assert "schedule: every hour (rule: `FREQ=HOURLY`)" in plain.message


# ── find: resolve-by-meaning, identity fused with affordances (#1558, #1640) ──

_FIND_VOCAB = {
    "aurora": 0,
    "beacon": 1,
    "cascade": 2,
    "gamma": 3,
    "delta": 4,
    "echo": 5,
    "foxtrot": 6,
    "reusable": 7,
    "recipes": 8,
    "orbit": 9,
    "nebula": 10,
}
_FIND_DIM = 16


def _axis_vec(text: str) -> list[float]:
    """A collision-free, L2-normalised vector: each known word owns a fixed axis
    (unknown words ignored), so cosine between two texts is exactly
    (shared words) / (sqrt(len_a) * sqrt(len_b)) — deterministic scores for an
    exact whole-render literal."""
    vec = [0.0] * _FIND_DIM
    for word in text.lower().split():
        axis = _FIND_VOCAB.get(word)
        if axis is not None:
            vec[axis] += 1.0
    norm = sum(value * value for value in vec) ** 0.5
    return [value / norm for value in vec] if norm else vec


def _axis_embed(model: str, text: str | list[str]) -> list[list[float]]:
    inputs = text if isinstance(text, list) else [text]
    return [_axis_vec(one) for one in inputs]


def _axis_client(mock_llm) -> LlmClient:
    """A client whose embeddings are the fixed-axis vectors above — so every
    stored description/content anchor and the query share one exact geometry."""
    mock_llm.set_embed_handler(_axis_embed)
    return LlmClient(
        api_url="http://localhost:11434", model="test-model", max_retries=1, retry_delay=0.0
    )


async def _create_collection(db, client: LlmClient, name: str, description: str) -> None:
    """Instantiate a collection whose description anchor is ``description`` (what
    ``find`` resolves over), via a plain prompt (#2455 naive baseline) — going
    through ``CollectionCreateTool`` rather than the ``collection_configure`` front
    door skips the same-job check (#1775), so these tests can stand up several
    deliberately-similar collections."""
    await CollectionCreateTool(db, client).execute(
        CallLink.new_top(),
        name=name,
        description=description,
        prompt="Browse for items and note what's found.",
        schedule="FREQ=HOURLY",
    )


class TestFind:
    """Resolve-by-meaning over the whole registry (collections and logs),
    fusing exact identity with how to address it (#1558).  The result is
    model-facing text, so each mode is
    asserted as a whole render."""

    _KITCHEN_SINK = (
        'Found 3 things matching "aurora beacon cascade", best first:\n'
        "1. aurora-watch — active collection: aurora beacon cascade\n"
        "   how to use it: read it with collection_list('aurora-watch'), "
        "reconfigure it with collection_configure(collection='aurora-watch', ...), archive it "
        "with collection_archive('aurora-watch')\n"
        "2. aurora-archive — archived collection: aurora beacon delta\n"
        "   how to use it: restore it with collection_unarchive('aurora-archive'); its "
        "entries stay readable with collection_list('aurora-archive')\n"
        "3. aurora-log — active log: aurora echo foxtrot\n"
        "   how to use it: read it with log_read('aurora-log')\n"
        "Ranked by closeness — if one is what you meant, use its addressing above; "
        "otherwise narrow by its exact name."
    )

    @staticmethod
    async def _seed_world(db, client: LlmClient) -> None:
        """One object of every renderable family, all sharing the query's meaning:
        an active collection, an archived collection, and a log."""
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        await _create_collection(db, client, "aurora-archive", "aurora beacon delta")
        await CollectionArchiveTool(db).execute(CallLink.new_top(), collection="aurora-archive")
        await LogCreateTool(db, client).execute(
            CallLink.new_top(),
            log="aurora-log",
            description="aurora echo foxtrot",
        )

    @pytest.mark.asyncio
    async def test_kitchen_sink_fuses_identity_and_affordances(self, db, mock_llm):
        """A query matching a mixed set returns each hit's exact identity, family,
        live/archived state, AND the deterministic addressing — best-first."""
        client = _axis_client(mock_llm)
        await self._seed_world(db, client)
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.success
        assert result.message == self._KITCHEN_SINK

    @pytest.mark.asyncio
    async def test_type_argument_is_rejected(self, db, mock_llm):
        """``find`` accepts exactly ``query`` (#1643): a passed ``type`` gets the
        standard unknown-argument rejection (``ToolArgs`` ``extra="forbid"``) — the
        guess-free tool never offers a family filter to guess with.  Rejected before
        ``execute``, so the arg-validation envelope names the offending field."""
        client = _axis_client(mock_llm)
        await self._seed_world(db, client)
        result = await FindTool(db, client).run(
            CallLink.new_top(), query="aurora beacon cascade", type="skill"
        )
        assert result.success is False
        assert "unknown parameter 'type'" in result.message
        assert "valid parameters: query" in result.message

    @pytest.mark.asyncio
    async def test_single_confident_match(self, db, mock_llm):
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "solo-watch", "aurora beacon cascade")
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.message == (
            'Found 1 thing matching "aurora beacon cascade":\n'
            "1. solo-watch — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_list('solo-watch'), "
            "reconfigure it with collection_configure(collection='solo-watch', ...), archive it "
            "with collection_archive('solo-watch')"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_returns_all_candidates_ranked(self, db, mock_llm):
        """Several matches come back ranked with how to narrow — never one silently
        chosen."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "watch-primary", "aurora beacon cascade")
        await _create_collection(db, client, "watch-secondary", "aurora beacon delta")
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.message == (
            'Found 2 things matching "aurora beacon cascade", best first:\n'
            "1. watch-primary — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_list('watch-primary'), reconfigure "
            "it with collection_configure(collection='watch-primary', ...), archive it "
            "with collection_archive('watch-primary')\n"
            "2. watch-secondary — active collection: aurora beacon delta\n"
            "   how to use it: read it with collection_list('watch-secondary'), reconfigure "
            "it with collection_configure(collection='watch-secondary', ...), archive it "
            "with collection_archive('watch-secondary')\n"
            "Ranked by closeness — if one is what you meant, use its addressing above; "
            "otherwise narrow by its exact name."
        )

    @pytest.mark.asyncio
    async def test_zero_matches_is_honest_empty(self, db, mock_llm):
        """A query unrelated to everything returns an honest empty pointing at the
        state at the top of the prompt (#2288) — not an error, no dead end."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        result = await FindTool(db, client).execute(CallLink.new_top(), query="orbit nebula")
        assert result.success
        assert result.message == (
            'Nothing of yours matched "orbit nebula" — the state at the top of this '
            "prompt lists every collection and log you have (archived ones by name)."
        )

    @pytest.mark.asyncio
    async def test_transient_embed_failure_is_actionable(self, db):
        """A transient query-embed failure returns an actionable retry, not a silent
        empty — the miss is named, the fix bound."""
        result = await FindTool(db, cast(Any, _FailingEmbedClient())).execute(
            CallLink.new_top(), query="anything"
        )
        assert result.success is False
        assert "Couldn't embed your query" in result.message
        assert "find(query=" in result.message

    @pytest.mark.asyncio
    async def test_stored_entry_carries_content_and_addressing(self, db, mock_llm):
        """A fact stored in a collection is found by the meaning of its content/key —
        the hit CARRIES the value plus the key the entry lives under, and it is one
        collection_get away, so for a short fact the find IS the answer (#1640).  The
        collection's OWN description is orthogonal, so only the entry surfaces (its
        container was not guessed)."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "knowledge-notes", "orbit nebula")
        await CollectionWriteTool(db, client, author="test", decider=_NO_MODEL).execute(
            CallLink.new_top(),
            collection="knowledge-notes",
            entries=[{"key": "aurora beacon", "content": "cascade gamma delta"}],
        )
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.success
        assert result.message == (
            'Found 1 thing matching "aurora beacon cascade":\n'
            "1. entry key='aurora beacon' in `knowledge-notes` — \"cascade gamma delta\"\n"
            "   read it: collection_get(collection='knowledge-notes', key='aurora beacon')"
        )

    @pytest.mark.asyncio
    async def test_entry_ranks_beside_its_collection(self, db, mock_llm):
        """The entry and its (also-matching) collection come back in ONE best-first
        list — the object outranks the entry here, both rendered by their own house
        pattern (#1640)."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        await CollectionWriteTool(db, client, author="test", decider=_NO_MODEL).execute(
            CallLink.new_top(),
            collection="aurora-watch",
            entries=[{"key": "aurora beacon", "content": "cascade gamma"}],
        )
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.message == (
            'Found 2 things matching "aurora beacon cascade", best first:\n'
            "1. aurora-watch — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_list('aurora-watch'), "
            "reconfigure it with collection_configure(collection='aurora-watch', ...), archive it "
            "with collection_archive('aurora-watch')\n"
            "2. entry key='aurora beacon' in `aurora-watch` — \"cascade gamma\"\n"
            "   read it: collection_get(collection='aurora-watch', key='aurora beacon')\n"
            "Ranked by closeness — if one is what you meant, use its addressing above; "
            "otherwise narrow by its exact name."
        )

    @pytest.mark.asyncio
    async def test_keyless_log_entry_addresses_a_read_with_log_get(self, db, mock_llm):
        """A keyless log entry renders its ``#<id>`` identity + the ``log_get`` call
        that reads it back by that id — a log has no key to re-read by (#1640).
        The log's OWN description is orthogonal, so only the entry surfaces.  An
        entry longer than the old preview cap renders WHOLE — the find IS the
        answer, no invented truncation (#2642)."""
        client = _axis_client(mock_llm)
        await LogCreateTool(db, client).execute(
            CallLink.new_top(), log="aurora-log", description="orbit nebula"
        )
        long_entry = (
            "aurora echo foxtrot: the full survey came back with every detail "
            "worth keeping about the launch window and the launch window's "
            "backstop schedule"
        )
        assert len(long_entry) > 120
        await LogAppendTool(db, client, author="test").execute(
            CallLink.new_top(), log="aurora-log", content=long_entry
        )
        entry_id = db.memories.memory("aurora-log").read_all()[0].id
        result = await FindTool(db, client).execute(CallLink.new_top(), query="aurora echo foxtrot")
        assert result.success
        assert result.message == (
            'Found 1 thing matching "aurora echo foxtrot":\n'
            f'1. entry #{entry_id} in `aurora-log` (log) — "{long_entry}"\n'
            f"   read it: log_get(log='aurora-log', id={entry_id})"
        )

    @pytest.mark.asyncio
    async def test_message_from_each_log_ranks_in_its_own_section(self, db, mock_llm):
        """A message from EITHER conversation log is returned by find when it ranks
        in the fused top FIND_MATCH_LIMIT (#2289) — read back in full, who said it
        (which log it's from), when, and its content — listed AFTER the store hits,
        under their own heading, never interleaved with them.  An unrelated message
        (cosine <= 0, same rule every other find candidate obeys) stays out."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        db.memories.create_log(PennyConstants.MEMORY_USER_MESSAGES_LOG, "inbound")
        db.memories.create_log(PennyConstants.MEMORY_PENNY_MESSAGES_LOG, "outbound")
        db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            "user",
            "aurora beacon cascade delta",
            embedding=serialize_embedding(_axis_vec("aurora beacon cascade delta")),
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "aurora beacon gamma",
            embedding=serialize_embedding(_axis_vec("aurora beacon gamma")),
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "orbit nebula",
            embedding=serialize_embedding(_axis_vec("orbit nebula")),
        )
        user_stamp = format_log_timestamp(
            db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG).read_all()[0].created_at
        )
        penny_stamp = format_log_timestamp(
            db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG).read_all()[0].created_at
        )

        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.success
        assert result.message == (
            'Found 1 thing matching "aurora beacon cascade":\n'
            "1. aurora-watch — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_list('aurora-watch'), "
            "reconfigure it with collection_configure(collection='aurora-watch', ...), archive it "
            "with collection_archive('aurora-watch')\n"
            "\n"
            "From the conversation:\n"
            f"1. user-messages [{user_stamp}]: aurora beacon cascade delta\n"
            f"2. penny-messages [{penny_stamp}]: aurora beacon gamma"
        )
        assert "orbit nebula" not in result.message

    @pytest.mark.asyncio
    async def test_many_matching_messages_never_displace_store_hits(self, db, mock_llm):
        """However many messages match — even several that score as strongly as
        the collection itself — the store section is unaffected (#2289): it is
        computed independently (`MemoryStore.resolve_objects`), and the
        conversation section is reported separately, after it, capped at
        FIND_MATCH_LIMIT of its own."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        db.memories.create_log(PennyConstants.MEMORY_USER_MESSAGES_LOG, "inbound")
        # More messages than FIND_MATCH_LIMIT, every one scoring as well as the
        # collection itself (an exact match on the query's own words).
        for _ in range(PennyConstants.FIND_MATCH_LIMIT + 2):
            db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                "user",
                "aurora beacon cascade",
                embedding=serialize_embedding(_axis_vec("aurora beacon cascade")),
            )

        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.success
        assert result.message.startswith(
            'Found 1 thing matching "aurora beacon cascade":\n'
            "1. aurora-watch — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_list('aurora-watch'), "
            "reconfigure it with collection_configure(collection='aurora-watch', ...), archive it "
            "with collection_archive('aurora-watch')\n"
            "\n"
            "From the conversation:\n"
        )
        assert result.message.count("user-messages [") <= PennyConstants.FIND_MATCH_LIMIT

    @pytest.mark.asyncio
    async def test_empty_store_with_matching_messages_says_nothing_stored(self, db, mock_llm):
        """Zero store matches beside a found message reads "nothing STORED
        matched" rather than "nothing of yours matched" (#2289) — the wholesale
        empty wording would contradict the conversation section listed right
        after it."""
        client = _axis_client(mock_llm)
        db.memories.create_log(PennyConstants.MEMORY_USER_MESSAGES_LOG, "inbound")
        db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            "user",
            "aurora beacon cascade",
            embedding=serialize_embedding(_axis_vec("aurora beacon cascade")),
        )
        stamp = format_log_timestamp(
            db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG).read_all()[0].created_at
        )

        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade"
        )
        assert result.success
        assert result.message == (
            'Nothing stored matched "aurora beacon cascade" — the state at the top of '
            "this prompt lists every collection and log you have (archived ones by "
            "name).\n\n"
            "From the conversation:\n"
            f"1. user-messages [{stamp}]: aurora beacon cascade"
        )

    @pytest.mark.asyncio
    async def test_scoped_search_returns_only_that_memory_full_content(self, db, mock_llm):
        """``find(query, memory=<name>)`` (#2289) narrows to one memory's own
        nearest entries, full content — never the whole-registry list, and
        never another store's hit even when it would also match the query."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        await CollectionWriteTool(db, client, author="test", decider=_NO_MODEL).execute(
            CallLink.new_top(),
            collection="aurora-watch",
            entries=[{"key": "aurora beacon", "content": "cascade gamma delta echo foxtrot"}],
        )
        # Would also match the unscoped search — proves the scope actually narrows.
        await _create_collection(db, client, "aurora-notes", "aurora beacon cascade")
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade", memory="aurora-watch"
        )
        assert result.success
        assert "from `aurora-watch` (closest to the query first)" in result.message
        assert "cascade gamma delta echo foxtrot" in result.message
        assert "aurora-notes" not in result.message
        assert "how to use it" not in result.message  # no object-hit addressing here

    @pytest.mark.asyncio
    async def test_scoped_search_on_a_message_log(self, db, mock_llm):
        """Scoped on ``user-messages``/``penny-messages`` (#2289), the SAME
        per-entry render find's own conversation section uses — who said it,
        when, full content — reusing ``MessageLogMemory``'s own rendering."""
        client = _axis_client(mock_llm)
        db.memories.create_log(PennyConstants.MEMORY_USER_MESSAGES_LOG, "inbound")
        db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            "user",
            "aurora beacon cascade delta",
            embedding=serialize_embedding(_axis_vec("aurora beacon cascade delta")),
        )
        result = await FindTool(db, client).execute(
            CallLink.new_top(),
            query="aurora beacon cascade",
            memory=PennyConstants.MEMORY_USER_MESSAGES_LOG,
        )
        assert result.success
        assert "user-messages" in result.message
        assert "aurora beacon cascade delta" in result.message
        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", result.message)

    @pytest.mark.asyncio
    async def test_scoped_search_on_an_archived_collection(self, db, mock_llm):
        """A scoped find works on an archived collection too (#2289) — the same
        name the self-state header's archived line already names."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-archive", "aurora beacon delta")
        await CollectionWriteTool(db, client, author="test", decider=_NO_MODEL).execute(
            CallLink.new_top(),
            collection="aurora-archive",
            entries=[{"key": "aurora beacon", "content": "cascade gamma"}],
        )
        await CollectionArchiveTool(db).execute(CallLink.new_top(), collection="aurora-archive")
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="aurora beacon cascade", memory="aurora-archive"
        )
        assert result.success
        assert "cascade gamma" in result.message

    @pytest.mark.asyncio
    async def test_scoped_search_unknown_name_leads_with_nearest_by_meaning(self, db, mock_llm):
        """An unknown ``memory`` name gets the house did-you-mean refusal
        (#2289) with the MEANING leg too, since ``find`` already carries an
        embedding client (``_suggestion_embedding``)."""
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "garage", "aurora beacon cascade")
        result = await FindTool(db, client).execute(
            CallLink.new_top(), query="anything", memory="aurora beacon cascade"
        )
        assert result.success is False
        assert "did you mean 'garage'?" in result.message
