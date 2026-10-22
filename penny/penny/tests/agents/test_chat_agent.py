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
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import select

from penny.agents.chat import ChatAgent
from penny.constants import MutationAction, MutationActor, PennyConstants
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import Media, MessageLog
from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction
from penny.prompts import Prompt
from penny.tests.conftest import ONE_PX_PNG_B64, TEST_SENDER, require_memory, wait_until
from penny.tests.mocks.llm_patches import deterministic_embed
from penny.tools.read_emails import ReadEmailsTool
from penny.tools.search_emails import SearchEmailsTool

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
        # Test-only names avoid colliding with the four system LOG markers created
        # by migration 0026 (user-messages, penny-messages, browse-results,
        # collector-runs).  No COLLECTION is seeded any more — 0097 nuked the generic
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

        # Full system-prompt structure assertion, over BOTH system messages
        # (#2745): the static one (identity + instructions — the reused prefix,
        # byte-identical across turns) and the dynamic one (the clock + Penny's
        # current state).  Per-entry timestamps are normalised to a placeholder
        # so the verbatim assertion stays stable across runs without freezing
        # the clock.
        system_messages = [m for m in first_request["messages"] if m.get("role") == "system"]
        assert len(system_messages) == 2, (
            "Expected the static and the dynamic system messages (#2745), "
            f"got {len(system_messages)}"
        )
        static_text = system_messages[0]["content"]
        assert static_text == _BASIC_FLOW_STATIC_EXPECTED, (
            f"Static system message mismatch:\n{static_text!r}"
        )
        dynamic_text = system_messages[1]["content"]
        lines = dynamic_text.split("\n")
        # The date/time anchor opens the DYNAMIC message, and it is rendered in
        # the user's PROFILE timezone, never a bare UTC — otherwise the model
        # reasons from a UTC clock under a non-UTC profile and, for the hours
        # around local midnight, from the wrong calendar day.  The zone is read
        # off the profile row rather than restated here, so the fixture's
        # timezone and the expectation cannot drift apart.  The collector side
        # of the same shared renderer (``datetime_utils.current_datetime_line``)
        # is pinned in test_collector.py.
        profile = penny.db.users.get_info(TEST_SENDER)
        assert profile is not None, "The test user profile should exist"
        local_abbrev = datetime.now(ZoneInfo(profile.timezone)).strftime("%Z")
        assert lines[0].startswith("Current date and time: ")
        assert lines[0].endswith(f" {local_abbrev}"), lines[0]
        assert "UTC" not in lines[0], lines[0]
        state_text = "\n".join(lines[1:])
        state_text = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "YYYY-MM-DD HH:MM", state_text)
        assert state_text == _BASIC_FLOW_STATE_EXPECTED, (
            f"Dynamic system message mismatch:\n{state_text!r}"
        )

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


# ── 1a. The static/dynamic system split (#2745) ──────────────


@pytest.mark.asyncio
async def test_static_system_prefix_byte_identical_across_turns(db, monkeypatch):
    """The FIRST system message is the prefix the model server reuses across
    turns (#2745), so nothing per-turn — the clock, entry counts, job rows — may
    leak into it.  Two renders with a shifted clock and a changed registry must
    produce a byte-identical first message, while the dynamic second message
    DID move (so the guard cannot pass vacuously)."""
    agent = ChatAgent.__new__(ChatAgent)
    agent.db = db
    agent._pending_page_context = None
    agent._current_user = TEST_SENDER

    class _FixedClock:
        def __init__(self, when: datetime) -> None:
            self._when = when

        def now(self) -> datetime:
            return self._when

    monkeypatch.setattr(db, "clock", _FixedClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC)))
    first = agent._build_messages("hi", None, await agent._build_system_prompt(TEST_SENDER))

    monkeypatch.setattr(db, "clock", _FixedClock(datetime(2026, 1, 1, 12, 5, tzinfo=UTC)))
    db.memories.create_collection("late-addition", "created between the two renders")
    second = agent._build_messages("hi", None, await agent._build_system_prompt(TEST_SENDER))

    # Both system messages present, static first, dynamic second.
    assert [m.role for m in first[:2]] == ["system", "system"]
    assert [m.role for m in second[:2]] == ["system", "system"]
    # The reused prefix: byte-identical across a shifted clock and new state.
    assert second[0].content == first[0].content
    # The dynamic message carries the clock and moved between the renders.
    assert first[1].content.startswith("Current date and time: ")
    assert second[1].content != first[1].content
    # And the history + the bare user turn close the array.
    assert second[-1].role == "user"
    assert second[-1].content == "hi"


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
        await signal_server.push_message(sender=TEST_SENDER, content=ask)
        await wait_until(lambda: penny.db.memories.get("indie-platformers") is not None)

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
        assert "collection_archive" in names
        assert "collection_unarchive" in names

        # Reads.
        assert "collection_list" in names
        assert "collection_get" in names
        assert "collection_sample" in names
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

        # Notification mute/unmute — chat-surface tools over the MuteState row
        # (the retired /mute + /unmute commands), dispatched from natural language.
        assert "notifications_mute" in names
        assert "notifications_unmute" in names

        # Loop-control stays background-only — never on the chat surface.
        assert "done" not in names
        assert "send_message" not in names


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


_BASIC_FLOW_STATIC_EXPECTED = (
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
    "Every tool call has a `reasoning` field — use it to think out loud. Explain what "
    "you're looking for, what you already know, and what you'll do with the result.\n"
    "\n"
    "Search memory before browsing. Your memory tools "
    "(`collection_list(<collection>)`, `find(query=<text>)`, "
    "`log_read(<log>)`, etc.) read everything stored — the 'Your memory' "
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
    "`collection_configure(collection=<name>, prompt=<the steps>)` once they agree. When "
    "they ask to change a job you already run — more sources, a different format, a new "
    "time — just change that job, only what they asked for: reconfigure it with "
    "`collection_configure(collection=<that job>)` and tell them what changed. When "
    "nothing you run is like what they ask for, work out how to do it "
    "yourself, and ask for the steps only if you can't.\n"
    "\n"
    "When something is worth remembering, `collection_write` it — into a collection that "
    "fits, or under a new name; the write creates the collection if it doesn't exist. "
    "That is a plain write, not a job: no schedule, no notify.\n"
    "\n"
    "When a 'Current Browser Page' section appears above, the user is browsing that page "
    "right now. If they say 'this page', 'this thread', 'this article', or anything "
    "ambiguous, they mean the Current Browser Page — not something from earlier in the "
    "conversation.\n"
    "\n"
    "How to use the browse tool:\n"
    "One browse call runs a whole lookup: pass items, each a search, a link number like "
    "[4], or a URL, and each with what it is looking for in plain language — that is the "
    'whole specification ("the opening hours" or "anything the page says about refunds" '
    "are complete asks). A search is picked and read inside the same call, so you get "
    "back what the pages say, with a handle for every page you were given. There are no "
    "CSS selectors, XPaths, or HTML parsing anywhere in your tools; never ask the user "
    "for page structure, snippets, or selectors — reading pages is your job.\n"
    "\n"
    "After reading pages, you MUST respond with what you found. Do not make additional "
    "tool calls to re-fetch or supplement pages you already read. If a page had limited "
    "content, report what was there.\n"
    "\n"
    "Let the answer come from the pages you read — every fact, name, and detail in your "
    "response must come from the pages you read or your memory.\n"
    "\n"
    "Links are numbered, like [3]. When you reference something from a page, cite its "
    "number in brackets — never an address. Links you were shown this turn are the only "
    "links there are.\n"
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
    "link (its number in brackets) so the user can follow up."
)

# The dynamic system message's state header, everything after the clock line —
# the leading "\n" is the blank line that separated it from the clock.
_BASIC_FLOW_STATE_EXPECTED = (
    "\n"
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
    "- browse-results (log, 0 entries) — Every browse-tool fetch result\n"
    "- collector-runs (log, 0 entries) — One entry per Collector cycle: the target + its "
    "generated structural record (outcome, counts, tool trace)\n"
    "- penny-messages (log, 0 entries) — Every outgoing Penny reply\n"
    "- playlists (collection, 1 entries) — favorite playlists\n"
    "- rules (collection, 0 entries) — Standing instructions the user has given: things "
    "to always or never do\n"
    "- secrets (collection, 1 entries) — hidden\n"
    "- tips (log, 1 entries) — useful tips\n"
    "- user-messages (log, 0 entries) — Every incoming user message\n"
    "archived: `old-facts`\n"
    "\n"
    "### About the user\n"
    "- name: Test User\n"
    "- timezone: America/Los_Angeles\n"
    "- location: Seattle, WA\n"
    "\n"
    "To look deeper: memory_get(<name>) for a collection's full config and "
    "change history, run_get(run <id>) for one run's tool calls, "
    "collection_list(<name>) reads a store's entries, and find(query=<text>) "
    "searches them all by meaning — a collection, a stored entry, or "
    "something said in the conversation."
)
