"""Integration tests for ChatAgent message handling.

Test organization:
1. Full integration (happy path) — comprehensive end-to-end message flow
2. Special success cases — no tool call, anti-refusal
3. Error / edge cases — XML leak regression, short response warning, delivery failure
4. Memory inventory (rendered for every agent's prompt)
5. Ambient recall (chat-only — each recall mode + self-match exclusion)
6. Tool surface (chat-only — entry-mutation tools removed)
"""

import base64
import json
import re
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import Session, select

from penny.agents.chat import ChatAgent
from penny.agents.conversation_map import ConversationMap
from penny.agents.models import HistoryTurn, MessageRole
from penny.changelog import ChangelogEntry
from penny.constants import ChatPromptType, MutationAction, MutationActor, PennyConstants
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import Media, MemoryEntry, MemoryRow, MessageLog, PromptLog
from penny.datetime_utils import stored_as_utc, user_timezone_name, zone_or_utc
from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction
from penny.prompts import Prompt
from penny.tests.conftest import ONE_PX_PNG_B64, TEST_SENDER, require_memory, wait_until
from penny.tests.mocks.llm_patches import MockLlmClient, deterministic_embed
from penny.tools.link_handles import HandleTable
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool
from penny.validation import ConditionKey


def _session_anchor(agent, user: str = TEST_SENDER):
    """The anchor ``_session_state_section``/``get_history`` now take
    explicitly (#3078, so the real turn and the chat pre-warm share one
    read) — computed fresh, the same way ``_dispatch_turn`` does, for a
    direct test call outside a live turn."""
    idle_seconds = int(agent.config.runtime.CHAT_SESSION_IDLE_SECONDS)
    return agent.db.messages.session_anchor(user, idle_seconds)


# ── 1. Full integration (happy path) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_message_flow(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """
    Test the complete message flow:
    1. User sends a message via Signal
    2. Penny receives and processes it
    3. Ollama returns a tool call (fetch)
    4. Fetch tool executes (mocked)
    5. Ollama returns final response
    6. Penny sends reply via Signal
    """
    config = make_config()

    # Configure Ollama to return fetch tool call, then final response.
    # Use a URL query so the browse side-effect produces a page entry in
    # the browse-results log (search-only queries don't, by design).
    mock_llm.set_default_flow(
        final_response="here's what i found about your question! 🌟",
        search_query="https://weather.example.com/today",
    )

    async with running_penny(config) as penny:
        # Verify we have a WebSocket connection
        assert len(signal_server._websockets) == 1, "Penny should have connected to WebSocket"

        # Memory seed: exercise every rendering path in one verbatim assertion.
        # Test-only names avoid colliding with the five system LOG markers created
        # by migration 0026/0131 (user-messages, penny-messages, browse-results,
        # collector-runs, changelog).  No COLLECTION is seeded any more — 0097 nuked the generic
        # catch-alls and 0108 dropped the last one (#1911's soft reboot), which is why
        # the mechanisms section below is empty and the memory list is logs + the
        # collections this test made.
        # Active memories rendered in alphabetical order: "playlists" < "tips".
        penny.db.memories.create_collection("playlists", "favorite playlists")
        penny.db.memory("playlists").write_rule_decided(
            [EntryInput(key="morning", content="prog rock")],
            author="user",
        )
        penny.db.memories.create_log("tips", "useful tips")
        penny.db.memory("tips").append(
            [LogEntryInput(content="tune before playing")], author="user"
        )
        # Off and archived memories are seeded with entries so the verbatim
        # prompt assertion below proves they are filtered out of ambient recall.
        penny.db.memories.create_collection("secrets", "hidden")
        penny.db.memory("secrets").write_rule_decided(
            [EntryInput(key="do-not-share", content="classified")],
            author="user",
        )
        penny.db.memories.create_collection("old-facts", "archived")
        penny.db.memory("old-facts").write_rule_decided(
            [EntryInput(key="stale", content="no longer relevant")],
            author="user",
        )
        penny.db.memories.archive("old-facts")

        # Send incoming message
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what's the weather like today?",
        )

        # Wait for response
        response = await signal_server.wait_for_message(timeout=10.0)

        # Verify the response
        assert response["recipients"] == [TEST_SENDER]
        assert "here's what i found" in response["message"].lower()

        # Verify Ollama was called twice (tool call + final response)
        assert len(mock_llm.requests) == 2, "Expected 2 Ollama calls (tool + final)"

        # First request should have user message
        first_request = mock_llm.requests[0]
        messages = first_request.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert any("weather" in m.get("content", "").lower() for m in user_messages)

        # Second request should include tool result
        second_request = mock_llm.requests[1]
        messages = second_request.get("messages", [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) >= 1, "Second request should include tool result"

        # Full system-prompt structure assertion, over ALL THREE system messages
        # (#3089): the LEADING one (identity + instructions — the reused prefix
        # across every session), the SESSION-STATE one (Jobs + Your memory +
        # Recent activity + About the user, bounded to the session's own
        # start — everything seeded here landed before this turn's session
        # began, so it all renders), and the per-turn DYNAMIC one (just the
        # clock — there's no page and no related exchange on a first-ever
        # turn).  Per-entry timestamps are normalised to a placeholder so the
        # verbatim assertion stays stable across runs without freezing the
        # clock.
        system_messages = [m for m in first_request["messages"] if m.get("role") == "system"]
        assert len(system_messages) == 3, (
            "Expected the leading, session-state, and dynamic system messages "
            f"(#3089), got {len(system_messages)}"
        )
        leading_text = system_messages[0]["content"]
        assert leading_text == _BASIC_FLOW_LEADING_EXPECTED, (
            f"Leading system message mismatch:\n{leading_text!r}"
        )
        session_state_text = system_messages[1]["content"]
        session_state_text = re.sub(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "YYYY-MM-DD HH:MM", session_state_text
        )
        assert session_state_text == _BASIC_FLOW_SESSION_STATE_EXPECTED, (
            f"Session-state system message mismatch:\n{session_state_text!r}"
        )
        dynamic_text = system_messages[2]["content"]
        # The date/time anchor is the WHOLE dynamic message on a first-ever
        # turn (no page, no related exchange), rendered in the user's PROFILE
        # timezone, never a bare UTC — otherwise the model reasons from a UTC
        # clock under a non-UTC profile and, for the hours around local
        # midnight, from the wrong calendar day.  The zone is read off the
        # profile row rather than restated here, so the fixture's timezone and
        # the expectation cannot drift apart.  The collector side of the same
        # shared renderer (``datetime_utils.current_datetime_line``) is pinned
        # in test_collector.py.
        profile = penny.db.users.get_info(TEST_SENDER)
        assert profile is not None, "The test user profile should exist"
        local_abbrev = datetime.now(ZoneInfo(profile.timezone)).strftime("%Z")
        assert dynamic_text.startswith("Current date and time: ")
        assert dynamic_text.endswith(f" {local_abbrev}"), dynamic_text
        assert "UTC" not in dynamic_text, dynamic_text
        assert "\n" not in dynamic_text

        # Verify typing indicators were sent
        assert len(signal_server.typing_events) >= 1, "Should have sent typing indicator"

        # Verify messages were logged to database
        incoming_messages = penny.db.messages.get_user_messages(TEST_SENDER)
        assert len(incoming_messages) >= 1, "Incoming message should be logged"

        with penny.db.get_session() as session:
            outgoing = list(
                session.exec(select(MessageLog).where(MessageLog.direction == "outgoing")).all()
            )
        assert len(outgoing) >= 1, "Outgoing message should be logged"

        # Verify device_id FK is populated on both incoming and outgoing
        test_device = penny.db.devices.get_by_identifier(TEST_SENDER)
        assert test_device is not None, "Test device should be registered"
        assert incoming_messages[0].device_id == test_device.id
        assert outgoing[0].device_id == test_device.id

        # The message logs are read facades over messagelog: the flow's
        # incoming/outgoing messages surface through read_all, with two
        # conversational authors — the user (incoming) or Penny (outgoing).
        user_msg_entries = penny.db.memory("user-messages").read_all()
        assert any(e.content == "what's the weather like today?" for e in user_msg_entries)
        assert all(e.author == "user" for e in user_msg_entries)

        penny_msg_entries = penny.db.memory("penny-messages").read_all()
        assert any("here's what i found" in e.content.lower() for e in penny_msg_entries)
        assert all(e.author == "penny" for e in penny_msg_entries)

        browse_entries = penny.db.memory("browse-results").read_all()
        # Mock browse provider is wired in conftest; the tool was invoked once.
        assert len(browse_entries) >= 1
        assert all(e.author == "chat" for e in browse_entries)


# ── 1a. Leading / session-state / dynamic (#3089; previously static/dynamic
#        only, #2745/#2814) ──────────────


def _seed_closed_exchange(db, *, when: datetime) -> None:
    """One CLOSED user/reply pair, so ``MessageStore.session_anchor`` has a
    real session to anchor ``as_of`` on — without one, ``_session_as_of``
    falls back to the live clock on every call, and the byte-identical-
    within-a-session claim below would hold only by accident."""
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hi", timestamp=when
    )
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        "hey!",
        parent_id=ask_id,
        recipient=TEST_SENDER,
        timestamp=when,
    )


@pytest.mark.asyncio
async def test_leading_system_prompt_never_changes_with_clock_jobs_or_activity(
    db, test_config, monkeypatch
):
    """The FIRST system message — identity, standing rules, instructions — is
    the prefix the model server reuses across EVERY session (#3089; Jobs, the
    store map, and the conversation map all moved OUT into their own
    once-per-session message, previously #2814), so nothing that varies —
    not the clock, not a new job, not a new collection, not an activity event
    — may leak into it.  The companion claim, that the SESSION-STATE message
    changes only at a new session, is
    ``test_session_state_message_byte_identical_within_a_session_across_a_clock_run_and_write``
    below."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent.config = test_config
    agent._pending_page_context = None
    agent._current_user = TEST_SENDER

    class _FixedClock:
        def __init__(self, when: datetime) -> None:
            self._when = when

        def now(self) -> datetime:
            return self._when

    monkeypatch.setattr(db, "clock", _FixedClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC)))
    first = await agent._build_system_prompt(TEST_SENDER)

    monkeypatch.setattr(db, "clock", _FixedClock(datetime(2026, 1, 1, 12, 5, tzinfo=UTC)))
    db.memories.create_collection("late-addition", "created between the two renders")
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        "heads up, done!",
        recipient=TEST_SENDER,
        mechanism="a-mechanism",
    )
    second = await agent._build_system_prompt(TEST_SENDER)

    assert second == first
    assert "late-addition" not in first
    assert "Current date and time" not in first

    messages = agent._build_messages("hi", None, first)
    # Leading, dynamic state next, user LAST — no history, and no
    # session-state message (``agent._session_state`` is never set outside
    # ``_dispatch_turn``, exactly as before).
    assert [m.role for m in messages] == ["system", "system", "user"]
    assert messages[1].content.startswith("Current date and time: ")
    assert messages[-1].content == "hi"


@pytest.mark.asyncio
async def test_session_state_message_byte_identical_within_a_session_across_a_clock_run_and_write(
    db, test_config, monkeypatch
):
    """The cache invariant (#3089; previously #3055's weaker claim about the
    dynamic message): Jobs, Your memory, Recent activity and About the user
    now ride ONE session-state message, bounded by the session's own start
    (``ChatAgent._session_as_of``) — so a clock shift, a job run, and a job
    write, all happening AFTER the session began, leave it BYTE-IDENTICAL.
    The companion case, a NEW session (an idle rotation), is
    ``test_session_state_message_rebuilds_only_at_a_new_session`` below."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent.config = test_config
    agent._pending_page_context = None
    agent._current_user = TEST_SENDER

    session_start = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    _seed_closed_exchange(db, when=session_start)

    with Session(db.engine) as session:
        session.add(
            MemoryRow(
                name="hourly-watch",
                type="collection",
                description="a running job",
                extraction_prompt="1. go",
                schedule="FREQ=HOURLY",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        session.add(
            MemoryRow(
                name="watched",
                type="collection",
                description="the store it writes",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        session.commit()

    class _FixedClock:
        def __init__(self, when: datetime) -> None:
            self._when = when

        def now(self) -> datetime:
            return self._when

    monkeypatch.setattr(db, "clock", _FixedClock(session_start))
    before = agent._session_state_section(TEST_SENDER, _session_anchor(agent))

    # All within the SAME session (no idle gap past CHAT_SESSION_IDLE_SECONDS):
    # the clock moves, the job's cycle runs (a completed collector run in the
    # promptlog — the ledger row the last-run stamp used to read live), and it
    # writes one entry (the write that used to tick the store map's count).
    monkeypatch.setattr(db, "clock", _FixedClock(session_start + timedelta(minutes=30)))
    run_id = "run-cache-invariant"
    with Session(db.engine) as session:
        session.add(
            PromptLog(
                model="test-model",
                messages="[]",
                response=json.dumps({"choices": [{"message": {"tool_calls": []}}]}),
                agent_name="collector",
                run_id=run_id,
                run_outcome="worked",
                run_target="hourly-watch",
                timestamp=session_start + timedelta(minutes=30),
            )
        )
        session.add(
            MemoryEntry(
                memory_name="watched",
                key="first finding",
                content="value for first finding",
                author="collector",
                created_at=session_start + timedelta(minutes=30),
                created_by_run_id=run_id,
                last_written_by_run_id=run_id,
            )
        )
        session.commit()

    after = agent._session_state_section(TEST_SENDER, _session_anchor(agent))

    # Byte-identical across the run and the write — the cache invariant.
    assert after == before
    # Both pre-existing rows show (created before the session began); the
    # run that completed and the entry that was written DURING the session
    # are exactly what stay invisible until the next rotation (#3089) — the
    # tool result that made them is what carries the news meanwhile.
    assert "hourly-watch" in after
    assert "- watched" in after
    assert "last run WORKED" not in after
    assert "- watched 1 entries" not in after


@pytest.mark.asyncio
async def test_session_state_message_rebuilds_only_at_a_new_session(db, test_config):
    """A job or collection created or archived mid-session neither appears nor
    disappears until the next idle rotation (the code owner's ruling, #3089):
    the create/archive call is already in history as a tool result, so
    nothing is hidden.  Once a NEW session starts — an idle gap past
    ``CHAT_SESSION_IDLE_SECONDS`` — ``_session_as_of`` reads a new instant and
    the message correctly rebuilds.

    Anchored on the REAL clock throughout, never a monkeypatched one:
    ``MemoryRow.created_at``'s own default factory is a bare
    ``datetime.now(UTC)`` call (never ``db.clock``), so "late-addition"'s
    creation time is always real wall-clock time regardless of what ``db.clock``
    is set to — the session boundaries below are chosen relative to it instead
    of fighting it."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent.config = test_config
    agent._pending_page_context = None
    agent._current_user = TEST_SENDER

    # A session that began well before "now" — comfortably before
    # "late-addition" will be created at the real wall-clock instant below.
    session_start = datetime.now(UTC) - timedelta(hours=2)
    _seed_closed_exchange(db, when=session_start)
    first = agent._session_state_section(TEST_SENDER, _session_anchor(agent))

    # Still mid-session (no new incoming row, so the anchor is unchanged): a
    # new collection, created at real "now", appears nowhere yet.
    db.memories.create_collection("late-addition", "created mid-session")
    still_mid_session = agent._session_state_section(TEST_SENDER, _session_anchor(agent))
    assert still_mid_session == first
    assert "late-addition" not in still_mid_session

    # A new session — an idle gap past CHAT_SESSION_IDLE_SECONDS, anchored
    # far enough past real "now" that it is also past "late-addition"'s own
    # creation instant — makes a NEW closed exchange its own session's first
    # turn, and the collection is now visible.
    idle_seconds = int(agent.config.runtime.CHAT_SESSION_IDLE_SECONDS)
    new_session_start = datetime.now(UTC) + timedelta(seconds=idle_seconds + 60)
    _seed_closed_exchange(db, when=new_session_start)
    rebuilt = agent._session_state_section(TEST_SENDER, _session_anchor(agent))

    assert rebuilt != first
    assert "late-addition" in rebuilt


@pytest.mark.asyncio
async def test_history_stays_between_static_and_user_state_right_before_user(
    db, test_config, monkeypatch
):
    """With history present, the shape is ``[static][history…][state][user]``
    (#2756) — history rides right after the one leading system message (so
    the static text, the tool schemas, and history are the reused prefix),
    and the dynamic state message sits right before the live user turn, the
    one mid-conversation system placement prod's patched oMLX and OpenRouter
    both render in place."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent.config = test_config
    agent._pending_page_context = None
    agent._current_user = TEST_SENDER

    history = [HistoryTurn("user", "earlier question"), HistoryTurn("assistant", "earlier answer")]
    messages = agent._build_messages("hi", history, await agent._build_system_prompt(TEST_SENDER))

    assert [m.role for m in messages] == ["system", "user", "assistant", "system", "user"]
    assert messages[1].content == "earlier question"
    assert messages[2].content == "earlier answer"
    assert messages[3].content.startswith("Current date and time: ")
    assert messages[-1].content == "hi"


# ── Related earlier exchanges (#3010) ────────────────────────────────────────
#
# The exchange — user message + her reply as one text — embedded at write time
# and scored per turn against every exchange older than the history window; the
# top few at or above the floor render IN FULL in the per-turn dynamic system
# message.  The texts below share the plan's own words (garden · weather-log ·
# messages) so the deterministic bag-of-words mock ranks them the way the real
# embedding model ranked the measured prod turn; the fillers sit on unrelated
# topics and score far below any floor.

_PLAN_ASK = (
    "for the garden job, should each run first read the weather-log and then "
    "my recent messages about the garden?"
)
_PLAN_REPLY = (
    "yes — each garden-log run will first read the weather-log and your recent "
    "messages about the garden, then write the usual summary."
)
_PLAN_LIVE = (
    "ok, go ahead and update the garden job like we planned — weather-log first, then my messages?"
)
_PLAN_AT = datetime(2026, 10, 12, 14, 5)

# Enough newer filler exchanges that the plan sits past the history window.
# The first five are an hour old and the rest are fresh, so the CURRENT
# session holds six fillers: the session window (5 + 6 session turns) keeps
# everything but the plan — an unbroken filler run would hold the plan inside
# the window forever, since a session's window always covers the session's
# own turns plus the window's worth of pre-session ones (#3077).
_FILLER_COUNT = 11
_FILLER_STALE = 5
_FILLER_USERS = (
    "what time is the bake sale on saturday?",
    "we should settle the house rules debate at game night sometime.",
    "have you finished that novel you were reading last week?",
)
_FILLER_REPLIES = (
    "the bake sale starts at ten — i'll note it.",
    "whenever you like — bring the rulebook and we'll go through it.",
    "almost — a chapter to go, and the ending is worth the wait.",
)


def _seed_plan_exchange(db) -> tuple[int, int]:
    """The plan exchange, two days back and outside the window, with its
    exchange vector stamped the way the write path stamps one — the user side
    joined with the reply side as ONE text."""
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _PLAN_ASK, timestamp=_PLAN_AT
    )
    reply_id = db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _PLAN_REPLY,
        parent_id=ask_id,
        run_id="run-plan",
        timestamp=_PLAN_AT + timedelta(minutes=1),
        recipient=TEST_SENDER,
    )
    assert ask_id is not None and reply_id is not None
    exchange_text = f"{_PLAN_ASK}\n{_PLAN_REPLY}"
    db.messages.set_exchange_embedding(
        reply_id, serialize_embedding(deterministic_embed(exchange_text))
    )
    return ask_id, reply_id


def _seed_filler_exchanges(db) -> None:
    """The newer exchanges that push the plan out of the window — each with its
    own exchange vector, so they are real scored candidates, not absent ones."""
    for index in range(_FILLER_COUNT):
        user_text = _FILLER_USERS[index % len(_FILLER_USERS)]
        reply_text = _FILLER_REPLIES[index % len(_FILLER_REPLIES)]
        stale = db.clock.now() - timedelta(hours=1) if index < _FILLER_STALE else None
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text, timestamp=stale
        )
        reply_id = db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            reply_text,
            parent_id=ask_id,
            recipient=TEST_SENDER,
            timestamp=stale,
        )
        assert ask_id is not None and reply_id is not None
        db.messages.set_exchange_embedding(
            reply_id,
            serialize_embedding(deterministic_embed(f"{user_text}\n{reply_text}")),
        )


def _render_agent(db, config) -> ChatAgent:
    """A ChatAgent built for direct section renders — the same shape the
    prompt-structure tests above use, with the mock embedding client wired so
    the live message embeds deterministically."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent.config = config
    agent._pending_page_context = None
    agent._current_user = TEST_SENDER
    agent._embedding_model_client = MockLlmClient()
    return agent


@pytest.mark.asyncio
async def test_related_exchanges_render_in_the_dynamic_section(db, make_config):
    """An exchange older than the history window that scores at or above the
    floor renders IN FULL in the per-turn dynamic system message — both sides
    whole, the message ids, and the conversation map's own day anchor — while
    the static prefix stays untouched and the unrelated fillers stay out."""
    config = make_config(related_exchanges_min_similarity=0.35)
    ask_id, reply_id = _seed_plan_exchange(db)
    _seed_filler_exchanges(db)
    agent = _render_agent(db, config)

    section = await agent._related_exchanges_section(
        _PLAN_LIVE, TEST_SENDER, ChatPromptType.USER_MESSAGE
    )
    assert section is not None
    assert section == (
        "## Related earlier exchanges\n\n"
        f"Related earlier exchange · Oct 12 14:05 (messages {ask_id}, reply {reply_id}) — "
        "log_read{'user-messages', day: '2026-10-12'}\n"
        f'User: "{_PLAN_ASK}"\n'
        f'Penny: "{_PLAN_REPLY}"'
    )
    # The section rides the per-turn DYNAMIC message, right before the live
    # turn — never the static prefix the server reuses.  The per-turn attr is
    # set by ``_dispatch_turn``; the render test mimics that one assignment.
    agent._related_exchanges = section
    assert agent._dynamic_system_section().endswith(section)
    assert "Related earlier exchanges" not in await agent._build_system_prompt(TEST_SENDER)


@pytest.mark.asyncio
async def test_related_exchanges_stay_quiet_below_the_floor(db, test_config):
    """The default floor (0.43) is above this world's natural mock overlap, so
    no section renders — the quiet default, not an empty block."""
    _seed_plan_exchange(db)
    _seed_filler_exchanges(db)
    agent = _render_agent(db, test_config)

    section = await agent._related_exchanges_section(
        _PLAN_LIVE, TEST_SENDER, ChatPromptType.USER_MESSAGE
    )

    assert section is None
    agent._related_exchanges = section
    assert "## Related earlier exchanges" not in agent._dynamic_system_section()


@pytest.mark.asyncio
async def test_related_exchanges_exclude_the_history_window(db, make_config):
    """An exchange the history window still carries never renders — the
    boundary is the same grow/cut function the history itself is built from,
    so the section can never duplicate what history already shows."""
    config = make_config(related_exchanges_min_similarity=0.1)
    _seed_plan_exchange(db)
    agent = _render_agent(db, config)

    section = await agent._related_exchanges_section(
        _PLAN_LIVE, TEST_SENDER, ChatPromptType.USER_MESSAGE
    )

    assert section is None
    agent._related_exchanges = section
    assert "## Related earlier exchanges" not in agent._dynamic_system_section()


@pytest.mark.asyncio
async def test_related_exchanges_show_the_top_scoring_when_more_score_than_the_max(db, make_config):
    """MORE exchanges score above the floor than ``RELATED_EXCHANGES_MAX``
    allows — the block shows exactly the TOP-scoring ones, never the weakest
    above the floor.  The four seeded exchanges rank strictly under the
    deterministic bag-of-words mock (0.82 · 0.67 · 0.58 · 0.47 against the
    live message's words: 0.89 · 0.57 · 0.47 · 0.35), the max is 2, and the
    top exchange's reply runs past the side clip — so the render also pins
    the clip's form."""
    config = make_config(related_exchanges_max=2, related_exchanges_min_similarity=0.4)
    live = "alpha bravo charlie delta echo foxtrot"
    long_reply = "alpha bravo charlie delta echo foxtrot. " * 20
    assert len(long_reply) > 600
    # Oldest first, weakest first: the block renders the top TWO by score, in
    # the block's own oldest-first order.
    seeded = []
    for minute, user_text, reply_text in (
        (5, "alpha bravo — oldest notes", "just the two words."),
        (10, "alpha bravo charlie — earlier notes", "three words in."),
        (15, "alpha bravo charlie delta — partial notes", "delta notes, and the rest."),
        (20, "alpha bravo charlie delta echo foxtrot — the full plan", long_reply),
    ):
        at = _PLAN_AT + timedelta(minutes=minute)
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text, timestamp=at
        )
        reply_id = db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            reply_text,
            parent_id=ask_id,
            run_id=f"run-ranked-{minute}",
            timestamp=at + timedelta(seconds=30),
            recipient=TEST_SENDER,
        )
        assert ask_id is not None and reply_id is not None
        db.messages.set_exchange_embedding(
            reply_id,
            serialize_embedding(deterministic_embed(f"{user_text}\n{reply_text}")),
        )
        seeded.append((user_text, reply_text, ask_id, reply_id))
    # Newer exchanges push all four past the history window — logged WITHOUT
    # exchange vectors, so they are not candidates of their own.  The first
    # five are an hour old (same split as ``_seed_filler_exchanges``): the
    # current session holds six fillers, so the session window keeps the four
    # ranked exchanges out (#3077).
    for index in range(11):
        stale = db.clock.now() - timedelta(hours=1) if index < _FILLER_STALE else None
        ask_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            f"window filler {index}",
            timestamp=stale,
        )
        db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            f"the filler reply number {index}.",
            parent_id=ask_id,
            recipient=TEST_SENDER,
            timestamp=stale,
        )
    agent = _render_agent(db, config)

    section = await agent._related_exchanges_section(live, TEST_SENDER, ChatPromptType.USER_MESSAGE)

    assert section is not None
    (_, _, second_ask, second_reply), (_, _, top_ask, top_reply) = seeded[2], seeded[3]
    assert section == (
        "## Related earlier exchanges\n\n"
        f"Related earlier exchange · Oct 12 14:20 (messages {second_ask}, reply {second_reply}) — "
        "log_read{'user-messages', day: '2026-10-12'}\n"
        'User: "alpha bravo charlie delta — partial notes"\n'
        'Penny: "delta notes, and the rest."\n\n'
        f"Related earlier exchange · Oct 12 14:25 (messages {top_ask}, reply {top_reply}) — "
        "log_read{'user-messages', day: '2026-10-12'}\n"
        'User: "alpha bravo charlie delta echo foxtrot — the full plan"\n'
        f'Penny: "{long_reply[:600]}…"'
    )


# ── Related exchanges vs the conversation map's own lines (#3085) ──────────

_MAP_ASK = "garden job plan — read the weather log first, then recent messages"
_MAP_REPLY = "yes, the garden job will read the weather log first, then your messages."
_MAP_LIVE = "go ahead with the garden job plan — weather log first?"
# Two lines, each well under ``USER_CLIP_CHARS`` — the map's own clip keeps
# only the first (``_clip`` splits on "\n" before measuring length), so this
# text is a lone map line that is NOT shown whole: the boundary ``clips_whole``
# exists to catch (#3085).
_MAP_ASK_MULTILINE = (
    "garden job plan — first line names the weather-log order\n"
    "second line says to check the recent garden messages too"
)


def _map_exchange(
    db,
    index: int,
    *,
    at: datetime,
    embed: bool = False,
    ask: str | None = None,
    reply: str | None = None,
) -> tuple[int, int]:
    """One outcome-bearing exchange — never collapse fodder on the
    conversation map, so it renders its own one-to-one line (or folds with
    its day, same as any other).  Only the target exchange (``embed=True``)
    carries an exchange vector, so it is the sole related-exchanges
    candidate and nothing else in the seeded run can score.  ``ask``/
    ``reply`` default to the shared plan text when embedding, or override it
    (e.g. the multi-line text above) — every other exchange gets its own
    filler text so nothing collides."""
    run_id = f"map-exchange-{index}"
    if ask is None:
        ask = _MAP_ASK if embed else f"a recorded-outcome ask, number {index}"
    if reply is None:
        reply = _MAP_REPLY if embed else f"a recorded-outcome reply, number {index}."
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, ask, timestamp=at
    )
    reply_id = db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        reply,
        parent_id=ask_id,
        run_id=run_id,
        recipient=TEST_SENDER,
        timestamp=at,
    )
    assert ask_id is not None and reply_id is not None
    db.memories.create_collection(f"garden-notes-{index}", "Notes.", created_by_run_id=run_id)
    require_memory(db, f"garden-notes-{index}").write_rule_decided(
        [EntryInput(key=f"note-{index}", content=f"note-{index} — recorded.")],
        author="penny",
        run_id=run_id,
    )
    if embed:
        db.messages.set_exchange_embedding(
            reply_id, serialize_embedding(deterministic_embed(f"{ask}\n{reply}"))
        )
    return ask_id, reply_id


def _map_stamp(db, at: datetime) -> str:
    """The SAME absolute-time format every map/related-exchanges line renders
    through, so a whole-literal assertion never hardcodes a date that quietly
    rots as real time passes it (#3085)."""
    return stored_as_utc(at).astimezone(zone_or_utc(user_timezone_name(db))).strftime("%b %d %H:%M")


def _map_day(db, at: datetime) -> str:
    """The SAME local-date anchor every map/related-exchanges line renders
    through."""
    return stored_as_utc(at).astimezone(zone_or_utc(user_timezone_name(db))).date().isoformat()


@pytest.mark.asyncio
async def test_related_exchange_on_a_map_line_renders_her_reply_alone(db, make_config):
    """A match whose user words the conversation map ALREADY shows, whole, on
    its own one-to-one line (#3085) drops its own ``User:`` side here —
    Penny's reply is the half the map never carries, so only it renders."""
    config = make_config(
        chat_history_turns=2, chat_session_max_turns=3, related_exchanges_min_similarity=0.1
    )
    now = db.clock.now()
    ask_id, reply_id = _map_exchange(db, 0, at=now, embed=True)
    for index in range(1, 5):
        _map_exchange(db, index, at=now)
    # Sanity: the exchange is on the map's own one-to-one line, AND shown whole.
    map_ = ConversationMap(db, TEST_SENDER, config.runtime)
    assert ask_id in map_.one_to_one_user_ids()
    assert ConversationMap.clips_whole(_MAP_ASK)
    agent = _render_agent(db, config)

    section = await agent._related_exchanges_section(
        _MAP_LIVE, TEST_SENDER, ChatPromptType.USER_MESSAGE
    )

    assert section == (
        "## Related earlier exchanges\n\n"
        f"Related earlier exchange · {_map_stamp(db, now)} "
        f"(messages {ask_id}, reply {reply_id}) — "
        f"log_read{{'user-messages', day: '{_map_day(db, now)}'}}\n"
        f'Penny: "{_MAP_REPLY}"'
    )


@pytest.mark.asyncio
async def test_related_exchange_on_a_folded_day_renders_both_sides(db, make_config):
    """A match whose user words sit on a FOLDED day (#3085) — the map
    collapsed that day's lines away, so it never carries this exchange's
    words whole — renders both sides, exactly as before the map existed."""
    config = make_config(
        chat_history_turns=2,
        chat_session_max_turns=3,
        chat_session_idle_seconds=7 * 24 * 3600,
        chat_map_max_lines=5,
        related_exchanges_min_similarity=0.1,
    )
    now = db.clock.now()
    yesterday = now - timedelta(days=1)
    ask_id = reply_id = None
    for index in range(8):  # window=2: older than it = indices 0-5 (#3056's own math)
        embed = index == 0
        a_id, r_id = _map_exchange(db, index, at=yesterday if index < 5 else now, embed=embed)
        if embed:
            ask_id, reply_id = a_id, r_id
    assert ask_id is not None and reply_id is not None
    # Sanity: the exchange's day folded, so it carries no one-to-one line.
    map_ = ConversationMap(db, TEST_SENDER, config.runtime)
    assert ask_id not in map_.one_to_one_user_ids()
    agent = _render_agent(db, config)

    section = await agent._related_exchanges_section(
        _MAP_LIVE, TEST_SENDER, ChatPromptType.USER_MESSAGE
    )

    assert section == (
        "## Related earlier exchanges\n\n"
        f"Related earlier exchange · {_map_stamp(db, yesterday)} "
        f"(messages {ask_id}, reply {reply_id}) — "
        f"log_read{{'user-messages', day: '{_map_day(db, yesterday)}'}}\n"
        f'User: "{_MAP_ASK}"\n'
        f'Penny: "{_MAP_REPLY}"'
    )


@pytest.mark.asyncio
async def test_related_exchange_longer_than_the_clip_keeps_both_sides(db, make_config):
    """A match that IS a lone map line (#3085, structurally) still keeps both
    sides when the map's own clip of its text is not the text itself — a
    multi-line ask is cut to its first line there (the map never carries the
    rest), the same as a text over ``USER_CLIP_CHARS`` is cut to a head+tail
    excerpt.  Either way the lone line's content differs from the full text,
    so this block never treats it as already shown whole."""
    config = make_config(
        chat_history_turns=2, chat_session_max_turns=3, related_exchanges_min_similarity=0.1
    )
    now = db.clock.now()
    ask_id, reply_id = _map_exchange(db, 0, at=now, embed=True, ask=_MAP_ASK_MULTILINE)
    for index in range(1, 5):
        _map_exchange(db, index, at=now)
    # Sanity: structurally its own line, but the map's clip cuts it to the
    # first line alone — not the text itself.
    map_ = ConversationMap(db, TEST_SENDER, config.runtime)
    assert ask_id in map_.one_to_one_user_ids()
    assert not ConversationMap.clips_whole(_MAP_ASK_MULTILINE)
    agent = _render_agent(db, config)

    section = await agent._related_exchanges_section(
        _MAP_LIVE, TEST_SENDER, ChatPromptType.USER_MESSAGE
    )

    assert section == (
        "## Related earlier exchanges\n\n"
        f"Related earlier exchange · {_map_stamp(db, now)} "
        f"(messages {ask_id}, reply {reply_id}) — "
        f"log_read{{'user-messages', day: '{_map_day(db, now)}'}}\n"
        f'User: "{_MAP_ASK_MULTILINE}"\n'
        f'Penny: "{_MAP_REPLY}"'
    )


def _wire_call(call_id: str, name: str, arguments: dict) -> dict:
    """One tool call as the wire/log stores it — ``arguments`` as a JSON string,
    the shape ``LoggedToolCall.from_function`` decodes."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_a_run_that_called_tools_renders_them_natively_into_a_later_turns_history(db):
    """A reply a run backed with real tool calls (#2915/#2888) expands into that
    run's own native call + tool-result turns ahead of the reply text — the fix
    for the false-save failure: a LATER turn sees the act beside the words,
    never only the words a copy-prone history used to carry alone.

    Three steps of one run: a ``browse`` call, a ``collection_write`` call, then
    the plain-text close — pinned whole, so the exact shape (assistant
    ``tool_calls`` with empty content, one ``tool`` message per result, headline
    only) is the literal contract, not an inference from a partial assert.  The
    browse result's stored content is EMPTY, pinning the other half of the
    headline rule: no first line falls back to the tool's own name plus
    ``(result)``, never a blank message."""
    run_id = "hist-run-1"
    first_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_wire_call("call_1", "browse", {"url": "https://example.com"})],
                }
            }
        ]
    }
    second_messages = [
        {"role": "user", "content": "check the weather and save it"},
        first_response["choices"][0]["message"],
        {"role": "tool", "tool_call_id": "call_1", "content": ""},
    ]
    second_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _wire_call(
                            "call_2",
                            "collection_write",
                            {"collection": "weather_log", "entries": [{"content": "Sunny, 18C"}]},
                        )
                    ],
                }
            }
        ]
    }
    third_messages = [
        *second_messages,
        second_response["choices"][0]["message"],
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": "Wrote 1 entry: weather_log.\nFull record follows for reference.",
        },
    ]
    third_response = {
        "choices": [
            {"message": {"role": "assistant", "content": "Checked and saved — sunny, 18C!"}}
        ]
    }
    db.messages.log_prompt(
        model="m", messages=[second_messages[0]], response=first_response, run_id=run_id
    )
    db.messages.log_prompt(
        model="m", messages=second_messages, response=second_response, run_id=run_id
    )
    db.messages.log_prompt(
        model="m", messages=third_messages, response=third_response, run_id=run_id
    )

    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    turn = HistoryTurn("assistant", "Checked and saved — sunny, 18C!", run_id=run_id)
    expanded = agent._expand_history_turn(turn)

    assert [m.role for m in expanded] == ["assistant", "tool", "assistant", "tool", "assistant"]
    assert expanded[0].content == "" and expanded[2].content == ""
    browse_call = expanded[0].tool_calls[0]
    write_call = expanded[2].tool_calls[0]
    assert browse_call.function.name == "browse"
    assert browse_call.function.arguments == {"url": "https://example.com"}
    assert write_call.function.name == "collection_write"
    assert write_call.function.arguments == {
        "collection": "weather_log",
        "entries": [{"content": "Sunny, 18C"}],
    }
    # Each call's own id is freshly minted, never the id it logged under, and
    # pairs with its OWN result turn by that fresh id — never the other call's.
    assert browse_call.id != "call_1" and write_call.id != "call_2"
    assert browse_call.id != write_call.id
    assert expanded[1].tool_call_id == browse_call.id
    assert expanded[3].tool_call_id == write_call.id
    # An empty result falls back to the tool's own name plus "(result)" —
    # never a blank message; a non-empty one carries ONLY its first line.
    assert expanded[1].content == "browse (result)"
    assert expanded[3].content == "Wrote 1 entry: weather_log."
    assert expanded[1].tool_success is True and expanded[3].tool_success is True
    # The reply text itself, last, unchanged.
    assert expanded[-1].content == "Checked and saved — sunny, 18C!"
    assert expanded[-1].tool_calls is None


def test_an_autonomous_turns_trigger_renders_ahead_of_the_reply(db):
    """An autonomous delivery turn's trigger (#2974) — the live turn's own USER
    message, prefixed with ``PennyConstants.SYSTEM_NOTIFICATION_PREFIX`` —
    renders again, byte-identical, immediately before her reply in later
    history: read back off the run's own promptlog row, never stored a second
    time.  The turn is looked at at all only because its messagelog row was
    autonomous (``HistoryTurn.mechanism`` non-null here) — an ordinary chat
    turn's run, whose ``mechanism`` is ``None``, is never consulted even
    though its run also ends in a user turn (pinned by the tool-call test
    above, whose run ends in ``user`` and renders no trigger)."""
    run_id = "job-report-run-1"
    trigger_text = (
        PennyConstants.SYSTEM_NOTIFICATION_PREFIX
        + 'Your job "trail-conditions" finished a run and asked to tell the user.\n'
        "What the job is for: Tracks trail-closure notices.\n"
        "It ran on its own schedule."
    )
    messages = [
        {"role": "system", "content": "## Identity\nYou are Penny."},
        {"role": "user", "content": trigger_text},
    ]
    response = {
        "choices": [{"message": {"role": "assistant", "content": "Trail's closed this week."}}]
    }
    db.messages.log_prompt(model="m", messages=messages, response=response, run_id=run_id)

    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    turn = HistoryTurn(
        "assistant", "Trail's closed this week.", run_id=run_id, mechanism="trail-conditions"
    )
    expanded = agent._expand_history_turn(turn)

    assert [m.role for m in expanded] == ["user", "assistant"]
    assert expanded[0].content == trigger_text
    assert expanded[1].content == "Trail's closed this week."


def test_the_trigger_is_found_past_a_discarded_or_rerolled_first_draw(db):
    """The run's chronologically FIRST row is never assumed clean (#2974): a
    draw rejected and rerolled still gets its own row, discard-stamped, but
    its ``messages`` column — written at send time — still carries the real
    trigger, and that row answers.  A second row from the reroll exists — the
    redraw is blind, so its messages are byte-identical to the first row's
    (#3052) — and is passed over: the walk takes the FIRST qualifying row,
    and the trigger row precedes it, so the reroll is never mistaken for the
    trigger."""
    run_id = "job-report-run-reroll"
    trigger_text = (
        PennyConstants.SYSTEM_NOTIFICATION_PREFIX
        + 'Your job "trail-conditions" finished a run and asked to tell the user.'
    )
    first_messages = [{"role": "user", "content": trigger_text}]
    first_response = {"choices": [{"message": {"role": "assistant", "content": "GARBLE???"}}]}
    first_id = db.messages.log_prompt(
        model="m", messages=first_messages, response=first_response, run_id=run_id
    )
    assert first_id is not None
    db.messages.mark_prompt_discarded(first_id, "degenerate_output")

    reroll_response = {
        "choices": [{"message": {"role": "assistant", "content": "Trail's closed this week."}}]
    }
    db.messages.log_prompt(
        model="m", messages=first_messages, response=reroll_response, run_id=run_id
    )

    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    turn = HistoryTurn(
        "assistant", "Trail's closed this week.", run_id=run_id, mechanism="trail-conditions"
    )
    expanded = agent._expand_history_turn(turn)

    assert [m.role for m in expanded] == ["user", "assistant"]
    assert expanded[0].content == trigger_text


def test_no_trigger_renders_when_the_runs_rows_are_pruned(db):
    """A run whose promptlog rows are gone entirely (retention pruning, or any
    other cause) renders her reply plainly (#2974) rather than failing or
    guessing at lost text — the same shape a talk-only reply with no run at
    all already gets."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    turn = HistoryTurn("assistant", "Trail's closed this week.", run_id="pruned-run")
    expanded = agent._expand_history_turn(turn)

    assert expanded == [LlmMessage(role=MessageRole.ASSISTANT, content="Trail's closed this week.")]


def test_no_trigger_renders_for_a_direct_reply_whose_run_carries_a_prefixed_ask(db):
    """The gate is the messagelog row's own mechanism, never the text (#2974): a
    DIRECT reply's run also ends in a user turn — and a user who literally types
    the notification prefix into chat makes that ask LOOK like a trigger — but
    with no mechanism on her reply's row, the run is never consulted at all and
    the message renders as the plain reply it is."""
    run_id = "direct-reply-run-1"
    messages = [
        {"role": "system", "content": "## Identity\nYou are Penny."},
        {
            "role": "user",
            "content": PennyConstants.SYSTEM_NOTIFICATION_PREFIX + "you there?",
        },
    ]
    response = {
        "choices": [{"message": {"role": "assistant", "content": "Trail's closed this week."}}]
    }
    db.messages.log_prompt(model="m", messages=messages, response=response, run_id=run_id)

    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    turn = HistoryTurn("assistant", "Trail's closed this week.", run_id=run_id)
    expanded = agent._expand_history_turn(turn)

    assert expanded == [LlmMessage(role=MessageRole.ASSISTANT, content="Trail's closed this week.")]


def test_a_trigger_precedes_a_run_that_also_joined_mid_run_messages(db):
    """A run can carry BOTH #2974's trigger and #2985's joined messages: the
    trigger renders first, immediately before the run's calls; each joined
    message renders after the tool step it followed (its own anchor); a joined
    message whose anchor matches no logged call renders after the last pair;
    the reply is last.  Trigger and joins keep their own seams — the join is
    never merged into the trigger's user turn."""
    run_id = "delivery-run-with-joins"
    trigger_text = (
        PennyConstants.SYSTEM_NOTIFICATION_PREFIX
        + 'Your job "trail-conditions" finished a run and asked to tell the user.'
    )
    first_messages = [{"role": "user", "content": trigger_text}]
    first_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_wire_call("call_1", "browse", {"url": "https://example.com"})],
                }
            }
        ]
    }
    second_messages = [
        *first_messages,
        first_response["choices"][0]["message"],
        {"role": "tool", "tool_call_id": "call_1", "content": "Trail closed — 30cm fresh snow."},
    ]
    second_response = {
        "choices": [{"message": {"role": "assistant", "content": "Trail's closed this week."}}]
    }
    db.messages.log_prompt(
        model="m", messages=first_messages, response=first_response, run_id=run_id
    )
    db.messages.log_prompt(
        model="m", messages=second_messages, response=second_response, run_id=run_id
    )
    # One join anchored to the tool step it followed, one whose anchor no
    # longer matches any logged call.
    db.messages.set_run_injected_messages(
        run_id,
        [
            {"content": "did it close?", "after_tool_call_id": "call_1"},
            {"content": "any webcams?", "after_tool_call_id": "call_gone"},
        ],
    )

    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    turn = HistoryTurn(
        "assistant", "Trail's closed this week.", run_id=run_id, mechanism="trail-conditions"
    )
    expanded = agent._expand_history_turn(turn)

    assert [m.role for m in expanded] == [
        "user",  # the trigger, first
        "assistant",  # the run's call
        "tool",  # the call's result
        "user",  # the join that followed that step
        "user",  # the unanchored join, after the last pair
        "assistant",  # the reply, last
    ]
    assert expanded[0].content == trigger_text
    assert expanded[1].tool_calls is not None and expanded[1].tool_calls[0].id != "call_1"
    assert expanded[2].tool_call_id == expanded[1].tool_calls[0].id
    assert expanded[2].content == "Trail closed — 30cm fresh snow."
    assert expanded[3].content == "did it close?"
    assert expanded[4].content == "any webcams?"
    assert expanded[5].content == "Trail's closed this week."
    assert expanded[5].tool_calls is None


@pytest.mark.asyncio
async def test_history_window_is_a_byte_identical_prefix_across_growing_turns(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Between cuts, each turn's history is the previous turn's history plus
    exactly one new exchange — never a changed message, link handle numbers
    included (#2756).  A URL planted in the first reply keeps the same handle
    number on every later turn, right up to the cut that drops it."""
    config = make_config(chat_history_turns=2, chat_session_max_turns=3)  # cuts at 4

    async with running_penny(config) as penny:
        renders: list[list[str]] = []
        for n in range(1, 5):  # four exchanges: turns at M=1,2,3 grow; M=4 cuts
            user_id = penny.db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                TEST_SENDER,
                f"user message {n}",
            )
            reply = "reply 1 https://example.com/a" if n == 1 else f"reply {n}"
            penny.db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                penny.config.signal_number,
                reply,
                parent_id=user_id,
                recipient=TEST_SENDER,
            )
            history = penny.chat_agent.get_history(
                TEST_SENDER, anchor=_session_anchor(penny.chat_agent)
            )
            table = HandleTable()
            converted = [
                table.convert(turn.content) if turn.role == "assistant" else turn.content
                for turn in (history or [])
            ]
            renders.append(converted)

        # M=1,2: fewer turns than the window (2) — everything so far, growing.
        # The handle names its domain (#2832) beside the number.
        assert renders[0] == ["user message 1", "reply 1 {link 101} example.com"]
        assert renders[1] == [
            "user message 1",
            "reply 1 {link 101} example.com",
            "user message 2",
            "reply 2",
        ]
        # M=3: within the window's growth (up to 2*2-1=3) — a byte-identical
        # extension of renders[1], same handle number for the planted URL.
        assert renders[2][: len(renders[1])] == renders[1]
        assert renders[2] == [
            "user message 1",
            "reply 1 {link 101} example.com",
            "user message 2",
            "reply 2",
            "user message 3",
            "reply 3",
        ]
        # M=4: the cut turn — back down to 2 exchanges, the oldest (and the
        # planted URL) dropped.  Allowed to differ (#2756).
        assert renders[3] == ["user message 3", "reply 3", "user message 4", "reply 4"]


@pytest.mark.asyncio
async def test_the_conversation_map_within_the_session_state_message_is_stable_except_on_a_cut_turn(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The conversation map rides the END of the session-state message
    (#3089; previously the static leading message, #2752/#2757), appended by
    ``_session_state_section``.  Across two turns neither of which is a cut,
    the exchanges older than the history window don't change, so the map is
    byte-identical; on a cut turn it gains exactly the exchanges the history
    dropped, whole, at the map's end, with every earlier line unchanged.

    This test scopes every comparison to the map's own SUFFIX of the
    session-state message (``map_suffix``), not the whole string: each loop
    iteration also creates a fresh collection and writes to it — Jobs and
    the store map ride the SAME message now, and since #3089 bounds them to
    the session's own start, every turn here (one continuous session, no
    idle gap) sees the SAME Jobs/store-map content regardless, which this
    test does not need to reason about since it reads only the map."""
    config = make_config(chat_history_turns=2, chat_session_max_turns=3)  # cuts at 4 and 6

    async with running_penny(config) as penny:
        statics: list[str] = []
        for n in range(1, 7):  # six exchanges: a cut at turn 4, another at turn 6
            run_id = f"{n:032x}"
            penny.db.memories.create_collection(f"list-{n}", f"notes {n}", created_by_run_id=run_id)
            require_memory(penny.db, f"list-{n}").write_rule_decided(
                [EntryInput(key=f"key-{n}", content=f"key-{n} — a note.")],
                author=PennyConstants.MessageAuthor.PENNY,
                run_id=run_id,
            )
            user_id = penny.db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                TEST_SENDER,
                f"user message {n}",
            )
            penny.db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                penny.config.signal_number,
                f"reply {n}",
                parent_id=user_id,
                recipient=TEST_SENDER,
                run_id=run_id,
            )
            statics.append(
                penny.chat_agent._session_state_section(
                    TEST_SENDER, _session_anchor(penny.chat_agent)
                )
                or ""
            )

        def map_suffix(static: str) -> str:
            """The map's own header-through-end slice, or "" when absent —
            everything Jobs/the store map ahead of it is cut away, since only
            THIS slice is under test here."""
            return static[start:] if (start := static.find(ConversationMap.HEADER)) != -1 else ""

        def map_lines(static: str) -> list[str]:
            return map_suffix(static).split("\n")[1:-1]  # content lines only

        # Turns 1-3: before any cut, nothing is older than the window yet —
        # the map is empty on all three, so the static message carries no map
        # section at all on any of them.
        assert map_suffix(statics[0]) == map_suffix(statics[1]) == map_suffix(statics[2]) == ""

        # Turn 4 is the first cut: exchanges 1 and 2 newly age out — the
        # static message grows to carry the map for the first time.
        assert map_suffix(statics[3]) != map_suffix(statics[2])
        turn4_lines = map_lines(statics[3])
        assert len(turn4_lines) == 2
        assert "user message 1" in turn4_lines[0]
        assert "user message 2" in turn4_lines[1]

        # Turn 5 is NOT a cut (the window is growing again after turn 4's
        # cut) — the map is byte-identical.
        assert map_suffix(statics[4]) == map_suffix(statics[3])

        # Turn 6 is the next cut: exchanges 3 and 4 age out, appended at the
        # map's end — turn 4/5's two lines stay an exact, byte-identical
        # prefix of turn 6's four.
        assert map_suffix(statics[5]) != map_suffix(statics[4])
        turn6_lines = map_lines(statics[5])
        assert turn6_lines[:2] == turn4_lines
        assert len(turn6_lines) == 4
        assert "user message 3" in turn6_lines[2]
        assert "user message 4" in turn6_lines[3]


# ── 1b. Provenance stamping ─────────────────────────────────


@pytest.mark.asyncio
async def test_collection_configure_stamps_chat_provenance(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """A collection created from a chat message records that message as its
    source and the turn's run as its creator (#1566).

    Proves the provenance is threaded end-to-end — the channel logs the incoming
    message first, ``handle`` mints the run_id and passes both down to
    ``collection_configure`` — rather than reconstructed after the fact.
    """
    ask = "can you keep a running list of new indie platformers for me?"

    def handler(request, _count):
        messages = request.get("messages") or []
        blob = " ".join(str(message.get("content", "")) for message in messages)
        # Only steer OUR chat turn; other agents' calls (background collectors)
        # get a plain text no-op so they can't create anything.
        if ask not in blob:
            return LlmResponse(
                message=LlmMessage(role="assistant", content="nothing to do"),
                model="test-model",
            )
        if any(message.get("role") == "tool" for message in messages):
            return LlmResponse(
                message=LlmMessage(role="assistant", content="done! i'll keep a list. 🌟"),
                model="test-model",
            )
        return LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LlmToolCall(
                        id="call_0",
                        function=LlmToolCallFunction(
                            name="collection_configure",
                            arguments={
                                "collection": "indie-platformers",
                                "description": "a running list of new indie platformers",
                                "prompt": "Browse for new indie platformers and note each one.",
                                "schedule": "FREQ=HOURLY",
                            },
                        ),
                    )
                ],
            ),
            model="test-model",
        )

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:

        def _provenance_stamped() -> bool:
            row = penny.db.memories.get("indie-platformers")
            return row is not None and row.source_message_id is not None

        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        # The row is created mid-turn (``collection_configure``), but its
        # source message is stamped only once the turn finishes — the
        # channel logs the incoming message AFTER ``handle`` returns, then
        # links it by ``run_id`` (``link_source_message``, #1566). Waiting on
        # row existence alone — not this, the full provenance contract — is
        # exactly the race #2919 saw: the row can be read between those two
        # steps, with ``source_message_id`` still unset.
        await wait_until(_provenance_stamped)

        row = penny.db.memories.get("indie-platformers")
        # The creating run is recorded, and the source points back at the exact
        # incoming message that spawned the collection.
        assert row.created_by_run_id is not None
        assert row.source_message_id is not None
        source = penny.db.messages.get_by_id(row.source_message_id)
        assert source is not None
        assert source.content == ask
        assert source.direction == PennyConstants.MessageDirection.INCOMING

        # The create is also a durable ledger event whose run is the SAME turn run
        # that stamped the row — proving the run id threads end-to-end from the
        # channel through the tool surface into the mutation ledger (#1560, C1/C4).
        events = penny.db.mutations.history("indie-platformers", limit=5)
        assert [e.action for e in events] == [MutationAction.CREATED.value]
        assert events[0].actor == MutationActor.USER_RUN.value
        assert events[0].run_id == row.created_by_run_id
        # And that run served this live conversation, not a mechanism: a chat run is
        # enumerable by ``run_list(source="chat")`` and stamps no run_target —
        # the served-entity closure that identifies it by its cause.
        chat_runs = penny.db.messages.run_call_groups(PennyConstants.CHAT_AGENT_NAME, None, 20)
        this_run = next(
            (g for g in chat_runs if any(p.run_id == row.created_by_run_id for p in g)), None
        )
        assert this_run is not None
        assert all(p.run_target is None for p in this_run)


# The writes record's own heading (#1946/#2185) — the narrate-from-the-RECORD frame,
# delivered straight onto the write's own tool result the moment it lands (#2473).
_WRITES_RECORD_HEADING = Prompt.WRITES_LANDED_RECORD.split("\n")[0]


def _writes_record(messages: list[dict]) -> str | None:
    """The LATEST writes record in view — appended to the end of a tool-result message
    the moment its write landed (#2473), never a message of its own and never edited into
    the system prompt (#2030).  Read from the END of the conversation backward, so a run
    with two writes reads the most recent (full-picture) addendum rather than an earlier
    one it has already superseded.  ``None`` when no draw has had one appended yet."""
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = str(message.get("content", ""))
            start = content.find(_WRITES_RECORD_HEADING)
            if start >= 0:
                return content[start:]
    return None


def _tool_call(call_id: str, name: str, arguments: dict) -> LlmResponse:
    return LlmResponse(
        message=LlmMessage(
            role="assistant",
            content="",
            tool_calls=[
                LlmToolCall(
                    id=call_id, function=LlmToolCallFunction(name=name, arguments=arguments)
                )
            ],
        ),
        model="test-model",
    )


def _text(content: str) -> LlmResponse:
    return LlmResponse(message=LlmMessage(role="assistant", content=content), model="test-model")


# The collapse fingerprint the reroll guard discards a draw on — the canonical shape
# from the corpus, carried in a tool call's ARGUMENTS, where it most often lands.
_COLLAPSE = "...??…?..?????"


@pytest.mark.asyncio
async def test_a_turn_narrates_the_writes_that_landed_not_the_ones_it_drew(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """The observed regression, recreated (#1946): one write LANDS, a second write draw
    is DISCARDED by the reroll guard, and the turn is left remembering two.

    A discarded draw never becomes a call, so it writes nothing and leaves nothing behind
    — but from inside the run it is indistinguishable from a write that happened, which
    is how a demonstration came to report every item pushed while the store held fewer.
    The frame is read off the ledger's own entry stamps, appended to the landed write's
    own tool result the moment it lands (#2473) — so the model's first reply already
    reads what actually landed, with no draft to discard and no second draw."""
    ask = "save the keel lantern and the aurora lantern prices for me"
    captured: dict[str, str | None] = {"writes": None}
    drew_the_second = {"hit": False}

    def handler(request, _count):
        messages = request.get("messages") or []
        blob = " ".join(str(m.get("content", "")) for m in messages)
        if ask not in blob:
            return _text("nothing to do")
        tool_turns = [m for m in messages if m.get("role") == "tool"]
        if not tool_turns:  # the write that LANDS
            return _tool_call(
                "c0",
                "collection_write",
                {
                    "collection": "lantern-prices",
                    "entries": [{"key": "keel lantern", "content": "$18"}],
                },
            )
        if not drew_the_second["hit"]:
            # The write that never happens: the draw collapses inside its own arguments,
            # so the guard throws it away and re-rolls the UNCHANGED context — this
            # handler is called again with the same messages and answers below.
            drew_the_second["hit"] = True
            return _tool_call(
                "c1",
                "collection_write",
                {
                    "collection": "lantern-prices",
                    "entries": [{"key": "aurora lantern", "content": _COLLAPSE}],
                },
            )
        # The collapse was discarded and re-rolled on the UNCHANGED context: tool_turns
        # still counts the one landed write, whose own tool result already carries the
        # record — in view before this, the model's FIRST and only, draft.
        captured["writes"] = _writes_record(messages)
        return _text("saved the keel lantern — the aurora one didn't land 🌊")

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("lantern-prices", "lantern prices")

        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        reply = await signal_server.wait_for_message(timeout=10.0)

        # The store holds ONE entry, and the frame states exactly that — not the two the
        # run drew.  Nothing here counts tool calls: the read is the entry stamps, so a
        # routine writing through a tool this test never heard of lands the same way.
        assert drew_the_second["hit"] is True
        stored = require_memory(penny.db, "lantern-prices").read_latest(10)
        assert [entry.key for entry in stored] == ["keel lantern"]
        assert captured["writes"] == Prompt.WRITES_LANDED_RECORD.format(
            record="1 entry landed in 'lantern-prices': 'keel lantern'"
        )
        # And the reply the user receives is the one drafted with the record already in
        # view — not a second one corrected after the fact.
        assert "didn't land" in reply["message"]


@pytest.mark.asyncio
async def test_a_turn_whose_every_save_was_refused_is_told_nothing_landed(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """The shape #2185 names: the turn asked the store to keep something and none of it
    landed, so there is no entry stamp to render — and before the record had an
    attempts half, nothing the reply was written from said "nothing was saved".

    The save here is refused as unchanged (the key already holds that value), which is
    the store answering, not failing.  The record lists the try with that answer,
    verbatim, read off the run's own call records by the tool's declared mark —
    appended to the refused call's own tool result the moment it comes back (#2473), so
    the model's first reply already reads it."""
    ask = "note the keel lantern price for me, it's $18"
    captured: dict[str, str | None] = {"writes": None}

    def handler(request, _count):
        messages = request.get("messages") or []
        blob = " ".join(str(m.get("content", "")) for m in messages)
        if ask not in blob:
            return _text("nothing to do")
        if not [m for m in messages if m.get("role") == "tool"]:
            return _tool_call(
                "c0",
                "collection_write",
                {
                    "collection": "lantern-prices",
                    "entries": [{"key": "keel lantern", "content": "$18"}],
                },
            )
        captured["writes"] = _writes_record(messages)
        return _text("that one was already in there at $18 🏮")

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("lantern-prices", "lantern prices")
        require_memory(penny.db, "lantern-prices").write_rule_decided(
            [EntryInput(key="keel lantern", content="$18")], author="user"
        )

        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        reply = await signal_server.wait_for_message(timeout=10.0)

        assert captured["writes"] == Prompt.WRITES_LANDED_RECORD.format(
            record=(
                "Nothing this turn tried to keep is in the store. What each try came back "
                "with:\n"
                "- `collection_write`: Unchanged: 'keel lantern' already holds the same "
                "value — no change since the last write (entry)."
            )
        )
        assert "already" in reply["message"]


@pytest.mark.asyncio
async def test_a_turn_that_writes_twice_states_the_full_record_each_time(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """A later step's addendum states the record AS OF NOW, not a delta (#2473): a run
    with two landed writes never leaves an earlier addendum understating what has
    landed, so the model is never shown two "current" records that disagree."""
    ask = "save the keel lantern and the aurora lantern prices for me"
    seen: list[str | None] = []

    def handler(request, _count):
        messages = request.get("messages") or []
        blob = " ".join(str(m.get("content", "")) for m in messages)
        if ask not in blob:
            return _text("nothing to do")
        tool_turns = [m for m in messages if m.get("role") == "tool"]
        if not tool_turns:
            return _tool_call(
                "c0",
                "collection_write",
                {
                    "collection": "lantern-prices",
                    "entries": [{"key": "keel lantern", "content": "$18"}],
                },
            )
        if len(tool_turns) == 1:
            seen.append(_writes_record(messages))
            return _tool_call(
                "c1",
                "collection_write",
                {
                    "collection": "lantern-prices",
                    "entries": [{"key": "aurora lantern", "content": "$22"}],
                },
            )
        seen.append(_writes_record(messages))
        return _text("saved both lantern prices for you")

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("lantern-prices", "lantern prices")

        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        await signal_server.wait_for_message(timeout=10.0)

        assert len(seen) == 2
        # Before the second write, the record states only the one that had landed.
        assert seen[0] == Prompt.WRITES_LANDED_RECORD.format(
            record="1 entry landed in 'lantern-prices': 'keel lantern'"
        )
        # After it, the record is recomputed fresh — the WHOLE picture, not a delta
        # naming only the new entry — so nothing here reads as a correction of the
        # first.
        assert seen[1] == Prompt.WRITES_LANDED_RECORD.format(
            record=("2 entries landed in 'lantern-prices': 'keel lantern', 'aurora lantern'")
        )


@pytest.mark.asyncio
async def test_a_turn_that_wrote_nothing_narrates_nothing(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """No writes, no frame — the zero-entry shape (#1946).

    It is what keeps the frame from firing on every chat turn: an ordinary conversation
    has no record to narrate, and a turn that browsed has only its own scratch, which is
    keyless and therefore not a write at all.

    A turn that used tools but never asked the store to keep anything gets no record
    either (#2185): none of its calls carries a tool's keep mark, so there is no attempt
    to report, and deciding it SHOULD have tried would mean reading the ask's words."""
    ask = "what do you make of the weather today?"
    saw_writes = {"hit": False}

    def handler(request, _count):
        messages = request.get("messages") or []
        blob = " ".join(str(m.get("content", "")) for m in messages)
        if ask not in blob:
            return _text("nothing to do")
        if _writes_record(messages) is not None:
            saw_writes["hit"] = True
            return _text("(this should never be reached)")
        if not [m for m in messages if m.get("role") == "tool"]:
            return _tool_call("c0", "collection_list", {"collection": "weather-notes"})
        return _text("grey and stubborn about it, same as yesterday 🌫️")

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("weather-notes", "notes about the weather")
        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        reply = await signal_server.wait_for_message(timeout=10.0)

        assert "stubborn" in reply["message"]
        assert saw_writes["hit"] is False


# ── 2. Special success cases ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_message_without_tool_call(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """Test handling a message where Ollama doesn't call a tool."""

    # Configure Ollama to return direct response (no tool call)
    def direct_response(request, count):
        return mock_llm._make_text_response(request, "just a simple response! 🌟")

    mock_llm.set_response_handler(direct_response)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="hello penny",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert response["recipients"] == [TEST_SENDER]
        assert "simple response" in response["message"].lower()

        # Only one Ollama call (no tool)
        assert len(mock_llm.requests) == 1


# ── 3. Error / edge cases ────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_response",
    [
        "<function=search><parameter=query>Canadian wildfires</parameter></function>",
        '<tools><search>{"query": "unusual instruments"}</search></tools>',
    ],
    ids=["function-param-xml", "tools-xml"],
)
async def test_xml_tool_call_not_leaked_to_user(
    malformed_response,
    signal_server,
    mock_llm,
    test_config,
    test_user_info,
    running_penny,
):
    """
    Regression test for #262: malformed tool call leaked to user.

    When a model emits XML-like markup in the content field instead of using
    structured tool_calls, the agent retries without consuming an agentic loop
    step, and the clean response reaches the user.
    """
    clean_response = "here are some great movies for you!"

    def handler(request, count):
        if count == 1:
            return mock_llm._make_text_response(request, malformed_response)
        return mock_llm._make_text_response(request, clean_response)

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="recommend a movie",
        )

        response = await signal_server.wait_for_message(timeout=10.0)

        assert mock_llm._request_count >= 2, (
            "Agent should have retried when XML markup was in content"
        )
        assert response["message"] == clean_response


@pytest.mark.asyncio
async def test_signal_progress_reactions_track_tool_calls(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """Penny's progress is shown as a morphing emoji reaction on the user's message.

    The dispatch loop reacts to the user's incoming message with 💭 (thinking)
    immediately, swaps to a tool-specific emoji as each tool batch fires
    (🔍 for search, 📖 for read), and removes the reaction once the agent
    finishes. The final response is sent via the normal send path so it
    carries text + image attachments + quote-replies just like before — no
    in-place message editing, no orphan thinking bubble.
    """
    final_answer = "here's the weather forecast for your area! 🌤️"
    # set_default_flow returns a single browse tool call (search query),
    # then the final text response.
    mock_llm.set_default_flow(final_response=final_answer, search_query="weather today")

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather?")

        final_msg = await signal_server.wait_for_message_containing(final_answer)

        # The final response is a single fresh message (no edit, no
        # follow-up split). It carries the full agent answer.
        assert final_msg["recipients"] == [TEST_SENDER]
        assert final_answer in final_msg["message"]

        # And only one outgoing message — no "thinking..." bubble, no
        # follow-up image bubble, no edits.
        response_bubbles = [
            m for m in signal_server.outgoing_messages if m.get("message") == final_msg["message"]
        ]
        assert len(response_bubbles) == 1

        # Reactions: 💭 sent at start, 🔍 swapped in once the search tool
        # batch fires, then a remove at delivery time. All three target the
        # same incoming message (the user's question).
        ops = [(e["op"], e.get("reaction")) for e in signal_server.reaction_events]
        assert ("send", "\U0001f4ad") in ops, f"expected initial 💭, got {ops}"
        assert ("send", "\U0001f50d") in ops, f"expected 🔍 search reaction, got {ops}"
        assert ops[-1][0] == "remove", f"final op should be a clear, got {ops}"

        # Every reaction (send and remove) is targeted at the user's
        # incoming message — same target_author + target timestamp throughout.
        targets = {
            (e.get("target_author"), e.get("timestamp")) for e in signal_server.reaction_events
        }
        assert len(targets) == 1, f"reactions should target a single message, got {targets}"


@pytest.mark.asyncio
async def test_signal_progress_reaction_uses_read_emoji_for_url_query(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """When the agent's tool call is a URL fetch (not a text search), the
    progress reaction morphs to 📖 instead of 🔍.

    Covers the URL branch of BrowseTool.to_progress_emoji — the search-only
    test above only exercises the text-query branch.
    """
    final_answer = "great article! 📖"
    # Drive the default flow with a URL query so BrowseTool.to_progress_emoji
    # picks the 📖 (reading) branch instead of 🔍 (searching).
    mock_llm.set_default_flow(
        final_response=final_answer,
        search_query="https://example.com/article",
    )

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="read this for me")
        await signal_server.wait_for_message_containing(final_answer)

        ops = [(e["op"], e.get("reaction")) for e in signal_server.reaction_events]
        assert ("send", "\U0001f4d6") in ops, f"expected 📖 read reaction, got {ops}"
        assert ("send", "\U0001f50d") not in ops, (
            f"URL queries should not trigger the search emoji, got {ops}"
        )


@pytest.mark.asyncio
async def test_signal_progress_clears_reaction_on_failure(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """If the agent crashes mid-run the dispatch loop must still clear the
    reaction so the user isn't left with a stale 💭 on their message forever.
    """

    def boom(request, count):
        raise RuntimeError("simulated agent failure")

    mock_llm.set_response_handler(boom)

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="hello there")

        # Wait for the dispatch loop to set the initial reaction and then
        # clear it from the finally block.
        await wait_until(lambda: any(e["op"] == "remove" for e in signal_server.reaction_events))

        ops = [(e["op"], e.get("reaction")) for e in signal_server.reaction_events]
        assert ops[0] == ("send", "\U0001f4ad"), f"expected initial 💭 send, got {ops}"
        assert any(o[0] == "remove" for o in ops), f"clear must happen on failure, got {ops}"


@pytest.mark.asyncio
async def test_delivery_failure_sends_notice(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """Test that a delivery failure notice is sent to the user when all send retries fail.

    When signal-cli returns a 400 SocketException on every attempt, the channel
    exhausts its retries and returns None from send_message.  _dispatch_to_agent
    should detect this and send a brief failure notice so the user knows to retry.
    """
    mock_llm.set_default_flow(
        final_response="my answer to your question",
    )

    # test_config uses llm_max_retries=1, so SignalChannel makes 2 total send
    # attempts (attempt 0 + 1 retry) for the main response.  Queue 2 transient
    # SocketException errors to exhaust those attempts; the 3rd request (the
    # failure notice) gets the default 200 success.
    socket_error = {
        "error": (
            "Failed to send message: Failed to get response for request"
            " (SocketException) (UnexpectedErrorException)"
        )
    }
    signal_server.queue_send_error(400, socket_error)
    signal_server.queue_send_error(400, socket_error)

    async with running_penny(test_config):
        await signal_server.push_message(sender=TEST_SENDER, content="hello there")

        notice = await signal_server.wait_for_message_containing("trouble")
        assert notice["recipients"] == [TEST_SENDER]
        assert len(mock_llm.requests) == 2


# ── 4. Memory inventory ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inventory_lists_non_archived_memories(
    signal_server, mock_llm, test_config, running_penny
):
    """Inventory names every non-archived memory regardless of recall mode."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("likes-test", "positive prefs")
        penny.db.memories.create_collection("dislikes-test", "negative prefs")
        penny.db.memories.create_log("messages-test", "convo log")

        result = penny.chat_agent._memory_inventory_section()

        assert result is not None
        assert "### Memory Inventory" in result
        assert "likes-test (collection, 0 entries) — positive prefs" in result
        # off still listed
        assert "dislikes-test (collection, 0 entries) — negative prefs" in result
        assert "messages-test (log, 0 entries) — convo log" in result


@pytest.mark.asyncio
async def test_inventory_excludes_archived(signal_server, mock_llm, test_config, running_penny):
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("retired-test", "archived")
        penny.db.memories.archive("retired-test")

        result = penny.chat_agent._memory_inventory_section()

        assert result is not None
        assert "retired-test" not in result


# ── 6. Tool surface ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_tool_surface_excludes_entry_mutations(
    signal_server, mock_llm, test_config, running_penny
):
    """Chat gets the full memory surface — entry mutations included.

    Capability is no longer curated by omission: the user can direct chat
    to fix a skill, correct a thought, seed a collection, or re-roll an
    entry mid-conversation.  Invalid calls are rejected by deterministic
    invariants (e.g. ``log_append`` to a system log) with a readable
    refusal, not by withholding the tool.  Loop-control tools (``done`` /
    ``send_message``) stay background-only and must NOT be here, or the
    model may call ``done`` instead of replying.
    """
    async with running_penny(test_config) as penny:
        names = {tool.name for tool in penny.chat_agent.get_tools()}

        # Entry mutations — now available to chat (user-directed edits).
        assert "collection_write" in names
        assert "collection_remove" in names
        assert "log_append" in names

        # Lifecycle / shape.
        assert "collection_configure" in names
        # The fused surface: the two prior lifecycle verbs are GONE.
        assert "collection_create" not in names
        assert "collection_update" not in names
        assert "log_create" in names
        # The archive/unarchive pair is folded onto one tool (#2812): archived=true/false.
        assert "collection_archive" in names
        assert "collection_unarchive" not in names

        # Reads.
        assert "collection_list" in names
        assert "collection_get" in names
        # collection_sample is retired (#2812): collection_list(order='random') replaces it.
        assert "collection_sample" not in names
        assert "collection_keys" in names
        assert "memory_get" in names
        # Ledger reads: run_list browses a source's runs; run_get resolves
        # one run by the typed id the activity block renders (#1580).
        assert "run_list" in names
        assert "run_get" in names
        # Resolve-by-meaning — the guess-free fallback every not-found points at
        # (#1558) — and the sole search over the conversation logs (#2289).
        assert "find" in names
        assert "read_similar" not in names

        # Fair random selection — the model is a biased chooser, so "pick one at
        # random" is a Python coin flip on the chat surface (like browse, on both).
        assert "choose" in names

        # Notifications — one chat-surface tool over the MuteState row (the retired
        # /mute + /unmute commands, then the retired mute/unmute tool pair, #2812),
        # dispatched from natural language with the direction as an `on` bool.
        assert "notifications_set" in names
        assert "notifications_mute" not in names
        assert "notifications_unmute" not in names

        # Loop-control stays background-only — never on the chat surface.
        assert "done" not in names
        assert "send_message" not in names

        # Previewing a job's own cycle (#2964) — chat-only, wired to the real
        # Collector running_penny built.
        assert "collection_sandbox_run" in names


@pytest.mark.asyncio
async def test_generate_image_registered_only_when_image_client_configured(
    signal_server, mock_llm, test_config, running_penny
):
    """generate_image mirrors the retired /draw command's conditionality.

    The tool is on the chat surface only when an image model is configured —
    absent by default (no image client), present once one is wired.
    """
    async with running_penny(test_config) as penny:
        assert penny.chat_agent._image_client is None
        assert "generate_image" not in {tool.name for tool in penny.chat_agent.get_tools()}

        penny.chat_agent._image_client = AsyncMock()
        assert "generate_image" in {tool.name for tool in penny.chat_agent.get_tools()}


@pytest.mark.asyncio
async def test_email_tools_registered_only_when_mailbox_configured(
    signal_server, mock_llm, test_config, running_penny
):
    """The email tools mirror the retired /email + /zoho commands' conditionality.

    They are on the chat surface only when a mailbox is configured — absent by
    default (no email_tools_builder), present once one is wired. The builder
    takes (user_query, today) and is invoked fresh per turn, so read_emails can
    summarise against the current question.
    """
    async with running_penny(test_config) as penny:
        assert penny.chat_agent._email_tools_builder is None
        assert "search_emails" not in {tool.name for tool in penny.chat_agent.get_tools()}

        email_client = AsyncMock()
        penny.chat_agent._email_tools_builder = lambda user_query, today: [
            SearchEmailsTool(email_client),
            ReadEmailsTool(email_client, penny.chat_agent._model_client, user_query, today),
        ]
        names = {tool.name for tool in penny.chat_agent.get_tools()}
        assert "search_emails" in names
        assert "read_emails" in names


@pytest.mark.asyncio
async def test_collection_sandbox_run_dispatches_to_the_wired_collector(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """``collection_sandbox_run`` (#2964) reaches whatever previews a job's cycle —
    the one chat-tool test that swaps ``ChatAgent._collector`` for a recording
    fake, exactly as ``_image_client``/``_email_tools_builder`` are swapped above;
    the production wiring (a real ``Collector``, ``penny.py``) is proven by
    ``test_run_sandbox_*`` in ``tests/agents/test_collector.py``."""

    class _RecordingRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def run_sandbox(self, collection_name: str) -> tuple[bool, str]:
            self.calls.append(collection_name)
            return True, f"Started a sandbox preview of '{collection_name}'."

    ask = "can you test my indie-metroidvanias job before I rely on it?"

    def handler(request, _count):
        messages = request.get("messages") or []
        blob = " ".join(str(message.get("content", "")) for message in messages)
        if ask not in blob:
            return LlmResponse(
                message=LlmMessage(role="assistant", content="nothing to do"), model="test-model"
            )
        if any(message.get("role") == "tool" for message in messages):
            return LlmResponse(
                message=LlmMessage(role="assistant", content="On it — I'll let you know."),
                model="test-model",
            )
        return LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="",
                tool_calls=[
                    LlmToolCall(
                        id="call_0",
                        function=LlmToolCallFunction(
                            name="collection_sandbox_run",
                            arguments={"collection": "indie-metroidvanias"},
                        ),
                    )
                ],
            ),
            model="test-model",
        )

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        runner = _RecordingRunner()
        penny.chat_agent._collector = runner

        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        await wait_until(lambda: runner.calls == ["indie-metroidvanias"])


@pytest.mark.asyncio
async def test_generate_image_delivers_the_drawn_image_deterministically(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A drawn image is delivered deterministically to its own reply, by id.

    The model calls generate_image with a faithful description, then replies in
    text; the tool stores the image in the media table and stamps its id onto the
    tool result, and egress attaches *exactly that row* to the reply — the
    media_ids path takes precedence over the fuzzy nearest-image ladder (which
    still serves replies that didn't generate anything).  A decoy browsed image
    is seeded first: the jittered fallback could ship it on a fuzzy match, so
    asserting the drawn bytes (not the decoy) proves the generate→deliver link
    is structural.
    """
    config = make_config()

    def handler(request: dict, count: int):
        if count == 1:
            return mock_llm._make_tool_call_response(
                request,
                "generate_image",
                {"description": "a teal origami dragon perched on a coffee mug"},
            )
        return mock_llm._make_text_response(request, "here's your teal origami dragon! 🐉")

    mock_llm.set_response_handler(handler)

    image_client = AsyncMock()
    image_client.generate_image.return_value = ONE_PX_PNG_B64

    async with running_penny(config) as penny:
        # A decoy browsed image already in the table — a candidate the removed
        # jitter could have attached to a no-URL reply.
        penny.db.media.put(
            data=b"decoy-browsed-image",
            mime_type="image/png",
            source_url="https://decoy.test/p",
            title="an unrelated page",
            embedding=serialize_embedding([1.0, 0.0]),
        )
        penny.chat_agent._image_client = image_client
        await signal_server.push_message(
            sender=TEST_SENDER, content="draw me a teal origami dragon on a coffee mug"
        )
        reply = await signal_server.wait_for_message_containing("dragon", timeout=10.0)

        # The tool ran with the model's faithful description.
        image_client.generate_image.assert_awaited_once()
        assert "dragon" in image_client.generate_image.await_args.kwargs["prompt"]

        # The drawn image was stored (side-channel) WITH an embedding of its
        # description: delivery to this reply is by id, but the row stays
        # matchable by the nearest-image ladder for future replies.
        with penny.db.get_session() as session:
            media_rows = session.exec(select(Media)).all()
        drawn = [row for row in media_rows if row.source_url is None]
        assert len(drawn) == 1
        assert drawn[0].mime_type == "image/png"
        assert drawn[0].embedding is not None
        assert drawn[0].data == base64.b64decode(ONE_PX_PNG_B64)

        # ...and exactly that drawn image (not the decoy) is attached to the reply.
        assert reply.get("base64_attachments") == [f"data:image/png;base64,{ONE_PX_PNG_B64}"]


@pytest.mark.asyncio
async def test_generate_image_reply_token_is_stripped_from_the_wire_kept_in_the_record(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The ``[image <id>]`` token the result hands her (#3033) is egress-only.

    The wire text never carries the token — an id this turn did NOT generate is
    stripped too, so a fabricated token can't reach the user as text — while the
    stored reply keeps it as she wrote it (#2722), and the drawn image still
    attaches through ``generated_media_ids`` (the no-token reply shape is the
    test above)."""
    config = make_config()

    def handler(request: dict, count: int):
        if count == 1:
            return mock_llm._make_tool_call_response(
                request,
                "generate_image",
                {"description": "a teal origami dragon perched on a coffee mug"},
            )
        return mock_llm._make_text_response(
            request, "here's your teal origami dragon!\n\n{image 999}\nenjoy"
        )

    mock_llm.set_response_handler(handler)

    image_client = AsyncMock()
    image_client.generate_image.return_value = ONE_PX_PNG_B64

    async with running_penny(config) as penny:
        penny.chat_agent._image_client = image_client
        await signal_server.push_message(
            sender=TEST_SENDER, content="draw me a teal origami dragon on a coffee mug"
        )
        reply = await signal_server.wait_for_message_containing("dragon", timeout=10.0)

        # The wire text carries no token — not even the fabricated one — and the
        # blank-line run the strip opened where the token stood alone is
        # collapsed back to the paragraph break.
        assert reply["message"] == "here's your teal origami dragon!\n\nenjoy"

        # The image still attached, by id, regardless of the token in the text.
        assert reply.get("base64_attachments") == [f"data:image/png;base64,{ONE_PX_PNG_B64}"]

        # The conversation record keeps the token as she wrote it (#2722).
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING
                )
            ).all()
        assert any("{image 999}" in row.content for row in outgoing)


@pytest.mark.asyncio
async def test_call_written_as_text_after_prose_is_rerolled_not_sent(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A reply that leads with a line of prose and then a tool call serialized as
    JSON text (#2787 — seen in prod: "On it!" followed by a ``generate_image`` call
    object) is discarded and re-rolled blind on the unchanged context (#3052) —
    shown neither the draft nor a note naming the tool it never ran — exactly like
    the whole-reply envelope :func:`is_call_as_text_bail` already catches.  The
    draft must never reach the user, and the call it describes must actually run on
    the next draw.
    """
    config = make_config()
    draft = (
        'On it!\n\n{"action": "generate_image", "action_input": '
        '"{\\"description\\": \\"a kite shaped like a sea turtle\\"}", '
        '"reasoning": "the user asked for a drawing"}'
    )

    def handler(request: dict, count: int):
        if count == 1:
            return mock_llm._make_text_response(request, draft)
        if count == 2:
            return mock_llm._make_tool_call_response(
                request,
                "generate_image",
                {"description": "a kite shaped like a sea turtle"},
            )
        return mock_llm._make_text_response(request, "here's your sea turtle kite! 🪁")

    mock_llm.set_response_handler(handler)

    image_client = AsyncMock()
    image_client.generate_image.return_value = ONE_PX_PNG_B64

    async with running_penny(config) as penny:
        penny.chat_agent._image_client = image_client
        await signal_server.push_message(
            sender=TEST_SENDER, content="draw me a kite shaped like a sea turtle"
        )
        reply = await signal_server.wait_for_message_containing("turtle", timeout=10.0)

        # The call ran, on the REDRAWN step — never the one holding the JSON text.
        image_client.generate_image.assert_awaited_once()
        assert "sea turtle" in image_client.generate_image.await_args.kwargs["prompt"]
        assert len(mock_llm.requests) == 3
        # The redraw is drawn blind on the unchanged context (#3052) — byte-identical
        # to the discarded draw's own request, no draft and no note appended.
        redraw = mock_llm.requests[1]["messages"]
        assert redraw == mock_llm.requests[0]["messages"]
        assert draft not in str(redraw)

        # The raw call-as-text never reached the user, in this reply or any other
        # the channel sent.
        assert "action_input" not in (reply.get("message") or "")
        for sent in signal_server.outgoing_messages:
            assert "action_input" not in (sent.get("message") or "")

        # The discarded draw's own promptlog row says why (#2702).
        with penny.db.get_session() as session:
            rows = list(session.exec(select(PromptLog)).all())
        discard_reasons = [row.discard_reason for row in rows if row.discard_reason]
        assert discard_reasons == [ConditionKey.CALL_AS_TEXT.value]


@pytest.mark.asyncio
async def test_call_written_as_text_in_brace_form_is_rerolled_not_sent(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A reply that leads with prose and ends with the call written as TEXT in
    gemma's brace form (#2867 — a regression from #2863, which moved every
    model-facing call example to the brace form without teaching the
    call-as-text check to recognise it) is discarded and re-rolled exactly
    like the parenthesized textual syntax already is (see the test above) —
    never reaching the user, the redraw drawn blind on the unchanged context
    (#3052).

    The draft is the prod leak verbatim, with the run id swapped: her
    thinking ended "Let's use run_get to investigate," her reply apologised
    and said she'd investigate the run right now, then ended with
    ``run_get{run: '…'}`` as plain text — no ``<|tool_call>`` wrapper, so
    nothing ran and the line went to the user.
    """
    config = make_config()
    draft = (
        "I'm going to investigate that run right now. 🕵️‍♀️\n\n"
        "run_get{run: '0123456789abcdef0123456789abcdef'}"
    )

    def handler(request: dict, count: int):
        if count == 1:
            return mock_llm._make_text_response(request, draft)
        return mock_llm._make_text_response(request, "all done looking into that run! 🕵️")

    mock_llm.set_response_handler(handler)

    async with running_penny(config) as penny:
        await signal_server.push_message(
            sender=TEST_SENDER, content="can you look into that bad job report?"
        )
        reply = await signal_server.wait_for_message_containing("all done", timeout=10.0)

        # The redraw is drawn blind on the unchanged context (#3052) — byte-identical
        # to the discarded draw's own request, no draft and no note appended.
        assert len(mock_llm.requests) == 2
        redraw = mock_llm.requests[1]["messages"]
        assert redraw == mock_llm.requests[0]["messages"]
        assert draft not in str(redraw)

        # The raw call-as-text never reached the user, in this reply or any
        # other the channel sent.
        assert "run_get{" not in (reply.get("message") or "")
        for sent in signal_server.outgoing_messages:
            assert "run_get{" not in (sent.get("message") or "")

        # The discarded draw's own promptlog row says why (#2702).
        with penny.db.get_session() as session:
            rows = list(session.exec(select(PromptLog)).all())
        discard_reasons = [row.discard_reason for row in rows if row.discard_reason]
        assert discard_reasons == [ConditionKey.CALL_AS_TEXT.value]


# ── 7. Quote-reply handling ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quote_reply_sets_parent_id_and_uses_thread_context(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Quote-reply: incoming message gets parent_id; agent sees the quoted thread.

    When a user quote-replies to a prior Penny message the incoming log row
    must have parent_id pointing to the quoted outgoing message, and the LLM
    call for the reply must include the original exchange in its history.
    """
    from sqlmodel import select

    from penny.database.models import MessageLog

    config = make_config()

    # Phase 1 — first exchange
    mock_llm.set_default_flow(final_response="It's sunny in New York!")
    async with running_penny(config) as penny:
        await signal_server.push_message(sender=TEST_SENDER, content="what's the weather?")
        await signal_server.wait_for_message_containing("New York", timeout=10.0)

        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
        assert outgoing is not None
        outgoing_id = outgoing.id
        outgoing_content = outgoing.content  # may have markdown stripped

        # Phase 2 — quote-reply to Penny's response
        mock_llm.set_default_flow(final_response="Boston is cloudy today!")
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what about Boston?",
            quote={"id": 12345, "author": "+15551234567", "text": outgoing_content},
        )
        await signal_server.wait_for_message_containing("Boston", timeout=10.0)

        # The incoming quote-reply should have parent_id pointing to Penny's prior reply
        with penny.db.get_session() as session:
            quote_reply_msg = session.exec(
                select(MessageLog).where(
                    MessageLog.direction == "incoming",
                    MessageLog.content == "what about Boston?",
                )
            ).first()
        assert quote_reply_msg is not None
        assert quote_reply_msg.parent_id == outgoing_id, (
            f"Quote-reply incoming message must have parent_id={outgoing_id},"
            f" got {quote_reply_msg.parent_id}"
        )

        # The LLM call for the quote-reply must include the original exchange in history.
        # Requests 0-1 are the first message (tool call + final).
        # Requests 2+ are for the quote-reply.
        assert len(mock_llm.requests) >= 3, "Expected requests for both exchanges"
        quote_reply_request = mock_llm.requests[2]
        history_msgs = [
            m for m in quote_reply_request["messages"] if m.get("role") in ("user", "assistant")
        ]
        roles_contents = [(m["role"], m["content"]) for m in history_msgs]
        assert any("weather" in c.lower() for _, c in roles_contents), (
            "LLM history for quote-reply must include the original question"
        )
        assert any("sunny" in c.lower() or "New York" in c for _, c in roles_contents), (
            "LLM history for quote-reply must include Penny's original reply"
        )


@pytest.mark.asyncio
async def test_startup_backfill_fills_missing_key_vector(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The real startup backfill (penny._backfill_memory_embeddings) fills a keyed
    entry that carries its content vector but is missing its key vector (#1468).

    Startup ran the backfill on the seeded DB; we then insert the gap row and
    re-run the backfill to confirm the re-embed path writes the KEY vector (not
    just content) — the migration-seeded / transient-key-miss state a
    content_embedding-only predicate would skip forever.
    """
    config = make_config()
    match_vec = deterministic_embed("dark roast")
    async with running_penny(config) as penny:
        # A non-seeded collection so the gap row is the only pending entry.
        penny.db.memories.create_collection("backfill-gap", "gap fixture")
        # Content vector present, key vector NULL (the gap state).
        penny.db.memory("backfill-gap").write_rule_decided(
            [EntryInput(key="dark roast", content="loves dark roast", content_embedding=match_vec)],
            author="system",
        )
        entry = penny.db.memory("backfill-gap").read_latest()[0]
        assert entry.content_embedding is not None
        assert entry.key_embedding is None

        embedded = await penny._backfill_memory_embeddings(batch_limit=100)
        assert embedded >= 1

        healed = penny.db.memory("backfill-gap").read_latest()[0]
        assert healed.key_embedding is not None  # the re-embed path wrote the key vector
        assert healed.content_embedding is not None
        # No entry is left with an unfilled vector after the backfill.
        assert penny.db.memories.get_entries_without_embeddings(limit=100) == []


@pytest.mark.asyncio
async def test_startup_backfill_fills_missing_exchange_vector(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The same startup backfill (``Penny._backfill_message_embeddings``) fills a
    reply row missing its exchange embedding (#3010) — the vector computed from
    BOTH sides of the exchange (``ExchangeSides.joined_text``), the exact text
    the write-time stamp embeds — and passes the interim send of the same run
    over: the vector belongs on the row that closes the exchange, one exchange,
    one vector."""
    config = make_config()
    ask_text = "should each run read the weather-log first?"
    reply_text = "yes — weather-log, then your recent messages."
    async with running_penny(config) as penny:
        ask_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, TEST_SENDER, ask_text
        )
        assert ask_id is not None
        # An interim send of the same run: run id, no parent — not a closing reply.
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            "reading the log now.",
            run_id="run-gap",
            recipient=TEST_SENDER,
        )
        reply_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            "penny",
            reply_text,
            parent_id=ask_id,
            run_id="run-gap",
            recipient=TEST_SENDER,
        )
        assert reply_id is not None

        embedded = await penny._backfill_message_embeddings(batch_limit=100)
        assert embedded >= 2

        reply = penny.db.messages.get_by_id(reply_id)
        assert reply is not None
        # The exchange text covers the whole turn: the ask, the interim send,
        # and the closing reply — exactly what the block will render.
        expected = serialize_embedding(
            deterministic_embed(f"{ask_text}\nreading the log now.\n{reply_text}")
        )
        assert reply.exchange_embedding == expected
        assert penny.db.messages.replies_without_exchange_embeddings(limit=100) == [], (
            "the closing reply is vectorised — nothing with both stamps is left pending"
        )
        rows = penny.db.messages.recent_direct_messages(since=datetime.min, limit=10)
        interim_rows = [row for row in rows if row.run_id == "run-gap" and row.parent_id is None]
        assert len(interim_rows) == 1
        assert interim_rows[0].exchange_embedding is None, (
            "the interim send must stay vectorless — it closes no exchange"
        )


@pytest.mark.asyncio
async def test_startup_records_changelog_entries_once(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The real startup step (``Penny._record_changelog_entries``, #2845/#2891)
    records every commit its commit list carries (``Config.changelog_commits``
    — a PARAMETER, #2891) into the `changelog` log on the first startup, each
    stamped with that startup's own time — then adds nothing on a second
    call, the idempotency a redeploy relies on."""
    entries = [
        ChangelogEntry(pr=9401, text="feat(penny): add a new way to save a note"),
        ChangelogEntry(pr=9402, text="fix(penny): rename the lookup tool"),
    ]
    config = make_config(changelog_commits=list(entries))
    # Naive-or-aware tolerant: compare both sides with tzinfo stripped, within a
    # generous window, rather than assume which shape SQLite round-trips.
    now_naive = datetime.now(UTC).replace(tzinfo=None)

    async with running_penny(config) as penny:
        changelog = penny.db.memory(PennyConstants.MEMORY_CHANGELOG_LOG)
        assert changelog is not None
        recorded = changelog.read_all()
        assert [row.content for row in recorded] == [entry.render() for entry in entries]
        for row in recorded:
            stamp = row.created_at.replace(tzinfo=None)
            assert abs((stamp - now_naive).total_seconds()) < 300, (
                "each entry should be stamped with this startup's own time"
            )

        # A second startup call is idempotent — nothing new is recorded.
        penny._record_changelog_entries()
        assert len(changelog.read_all()) == len(entries)


_BASIC_FLOW_LEADING_EXPECTED = (
    "## Identity\n"
    "You are Penny. You and the user are friends who text regularly. This is "
    "mid-conversation — not a fresh chat.\n"
    "\n"
    "Voice:\n"
    "- Reply like you're continuing a text thread.\n"
    "- React to what the user actually said before giving information. If they corrected "
    "you, own it. If they expressed excitement, match it. If they asked a follow-up, "
    "connect it to what came before.\n"
    "- Present information naturally but you can still use short formatted blocks (bold "
    "names, links) when listing products or facts. Just wrap them in conversational text, "
    "not a clinical dump.\n"
    "- Finish every message with an emoji.\n"
    "\n"
    "## Instructions\n"
    "It's fine to greet them back when they greet you; otherwise just pick up the thread.\n"
    "\n"
    "Search memory before browsing. Your memory tools "
    "(`collection_list{<collection>}`, `find{query: <text>}`, "
    "`log_read{<log>}`, etc.) read everything stored — the 'Your memory' "
    "list in the 'Penny's current state' section below names every store you can pull "
    "from, and the mechanisms + recent activity there are your own operational state "
    "(what you're running, what you just did). Only browse if memory doesn't have what "
    "the user needs, or for current/external info (news, products, prices, fresh facts).\n"
    "\n"
    "You are having a conversation, and doing whatever the user asks of you along the "
    "way.\n"
    "\n"
    "Reply to what they actually said. Don't act on something they only mentioned in "
    "passing — no lookup or browse they didn't ask for. Do what they do ask: look "
    "something up, check it, recall it, change it, or remember it.\n"
    "\n"
    "If a change you're asked to make could fit more than one running job — the ask "
    "names what kind of job it is but not which one, and nothing earlier in the "
    "conversation already picked one — say which jobs it could mean and ask which one, "
    "in that one message. Change nothing until they answer; naming them is a complete "
    "reply on its own.\n"
    "\n"
    "If they ask for a new job like one you already run, it's fine to offer to set it up "
    "the "
    "same way — name that job and what it does, step by step, and make it a new job with "
    "`collection_configure{collection: <name>, prompt: <the steps>}` once they agree. When "
    "they ask to change a job you already run — more sources, a different format, a new "
    "time — just change that job, only what they asked for: reconfigure it with "
    "`collection_configure{collection: <that job>}` and tell them what changed. When "
    "nothing you run is like what they ask for, work out how to do it "
    "yourself, and ask for the steps only if you can't.\n"
    "\n"
    "When something is worth remembering, `collection_write` it — into a collection that "
    "fits, or under a new name; the write creates the collection if it doesn't exist. "
    "That is a plain write, not a job: no schedule, no notify.\n"
    "\n"
    "When a 'Current Browser Page' section appears below, the user is browsing that page "
    "right now. If they say 'this page', 'this thread', 'this article', or anything "
    "ambiguous, they mean the Current Browser Page — not something from earlier in the "
    "conversation.\n"
    "\n"
    "How to use the browse tool:\n"
    "One browse call runs a whole lookup: pass items, each a search, a link token like "
    "{link 101}, or a URL, and each with what it is looking for in plain language — that is the "
    'whole specification ("the opening hours" or "anything the page says about refunds" '
    "are complete asks). A search is picked and read inside the same call, so you get "
    "back what the pages say, with a handle for every page you were given.\n"
    "\n"
    "Let the answer come from the pages you read — every fact, name, and detail in your "
    "response must come from the pages you read or your memory.\n"
    "\n"
    "Links carry tokens, like {link 101}. When you reference something from a page, cite its "
    "token — never an address. The links that exist are the ones you were "
    "shown, in this turn's pages or in your own stored entries.\n"
    "\n"
    "When the user changes topics, just go with it.\n"
    "\n"
    "Open your reply with the story of what you just did:\n"
    "1. Each tool result you got this turn opens with a line addressed to you, naming what "
    'that call actually did — e.g. "You searched for X and found…", "You saved X to '
    '`likes`", "You didn\'t add anything new — it was already there", "You couldn\'t '
    'find X to remove". In your reply you are the one speaking and the user is the one '
    'being spoken to, so each of those lines becomes something I did: "I searched for X '
    'and found…", "I saved X to your likes", "I didn\'t add anything new — it was already '
    'there", "I couldn\'t find X to remove". Lead your reply with a brief, natural recap '
    "that reflects EACH "
    "of those lines, in order — every call this turn, whether it succeeded, changed "
    "nothing, or failed — woven into a sentence, NOT a bulleted log.\n"
    "2. Mirror the OUTCOME each tool reported, never what you set out to do: if a save "
    "was already there, say it was already there; if a lookup came back empty, say so; if "
    "a call failed, say so. NEVER imply something changed when the tool said it didn't.\n"
    "3. Then give the answer.\n"
    "On a plain reply with no tool calls, skip the recap and just respond.\n"
    "\n"
    "Always include specific details (specs, dates, prices) and at least one source "
    "link (its token, like {link 101}) so the user can follow up."
)

# The ONE session-state system message (#3089; previously split across the
# static message's Jobs/Your memory, #2814, and the dynamic message's Recent
# activity/Memory sizes/About the user, #3055) — Jobs and Your memory each
# carry their live clauses on the SAME line as their registry ones now.
_BASIC_FLOW_SESSION_STATE_EXPECTED = (
    "## Penny's current state\n"
    "\n"
    "### Jobs\n"
    "notifications: on — proactive task notifications are delivered\n"
    "(no jobs yet)\n"
    "\n"
    "### Recent activity\n"
    "change · YYYY-MM-DD HH:MM UTC · old-facts archived by user-run\n"
    "change · YYYY-MM-DD HH:MM UTC · old-facts created by user-run\n"
    "change · YYYY-MM-DD HH:MM UTC · secrets created by user-run\n"
    "change · YYYY-MM-DD HH:MM UTC · playlists created by user-run\n"
    "\n"
    "### Your memory\n"
    "- browse-results (log) — Every browse-tool fetch result — searched only when named\n"
    "- changelog (log) — Changes made to your tools, prompts and behaviour, "
    "recorded at each deploy\n"
    "- collector-runs (log) — One entry per Collector cycle: the target + its "
    "generated structural record (outcome, counts, tool trace)\n"
    "- penny-messages (log) — Every outgoing Penny reply\n"
    "- playlists (collection) 1 entries — favorite playlists\n"
    "- rules (collection) 0 entries — Standing instructions the user has given: things "
    "to always or never do\n"
    "- run-health (log) — One line per run: cost, discards, tool errors, and what was "
    "sent — chat and every job\n"
    "- secrets (collection) 1 entries — hidden\n"
    "- tips (log) 1 entries — useful tips\n"
    "- user-messages (log) — Every incoming user message\n"
    "archived: `old-facts`\n"
    "\n"
    "### About the user\n"
    "- name: Test User\n"
    "- timezone: America/Los_Angeles\n"
    "- location: Seattle, WA\n"
    "\n"
    "To look deeper: memory_get{<name>} for a collection's full config and "
    "change history, run_get{run <id>} for one run's tool calls, "
    "collection_list{<name>} reads a store's entries, and find{query: <text>} "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)
