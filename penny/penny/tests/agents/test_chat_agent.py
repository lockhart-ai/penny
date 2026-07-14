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
from unittest.mock import AsyncMock

import pytest
from sqlmodel import select

from penny.constants import MutationAction, MutationActor, PennyConstants
from penny.database.memory import EntryInput, Inclusion, LogEntryInput, RecallMode
from penny.database.models import Media, MessageLog
from penny.database.skills import SkillDraft, SkillStep
from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction
from penny.tests.conftest import ONE_PX_PNG_B64, TEST_SENDER, wait_until
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

        # Seed full context: notified thought, dislike, active memory
        thought = penny.db.thoughts.add(TEST_SENDER, "Recent thought about amps")
        if thought:
            penny.db.thoughts.mark_notified(thought.id)
        penny.db.preferences.add(
            user=TEST_SENDER,
            content="Country music",
            valence="negative",
        )
        # Memory seed: exercise every rendering path in one verbatim assertion.
        # Test-only names avoid colliding with system memories created by
        # migrations 0026/0027/0068 (user-messages, penny-messages,
        # browse-results, likes, dislikes, knowledge, thoughts).
        # Active memories rendered in alphabetical order: "playlists" < "tips".
        penny.db.memories.create_collection(
            "playlists", "favorite playlists", Inclusion.ALWAYS, RecallMode.ALL
        )
        penny.db.memory("playlists").write(
            [EntryInput(key="morning", content="prog rock")],
            author="user",
        )
        penny.db.memories.create_log("tips", "useful tips", Inclusion.ALWAYS, RecallMode.RECENT)
        penny.db.memory("tips").append(
            [LogEntryInput(content="tune before playing")], author="user"
        )
        # Off and archived memories are seeded with entries so the verbatim
        # prompt assertion below proves they are filtered out of ambient recall.
        penny.db.memories.create_collection("secrets", "hidden", Inclusion.NEVER, RecallMode.RECENT)
        penny.db.memory("secrets").write(
            [EntryInput(key="do-not-share", content="classified")],
            author="user",
        )
        penny.db.memories.create_collection(
            "old-facts", "archived", Inclusion.ALWAYS, RecallMode.ALL
        )
        penny.db.memory("old-facts").write(
            [EntryInput(key="stale", content="no longer relevant")],
            author="user",
        )
        penny.db.memories.archive("old-facts")
        # The seeded skills collection is inclusion=always; embeddings are now a
        # required prerequisite, so its entries would surface here in a ranking
        # that's an artifact of the deterministic test embedder, not production.
        # Neutralize its recall so this test asserts only its own seeded memories
        # (relevant-mode skill rendering is covered by the recall tests below);
        # it stays listed in the Memory Inventory.
        penny.db.memories.update_collection_metadata("skills", inclusion=Inclusion.NEVER)

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

        # Full system prompt structure assertion.  Per-entry timestamps are
        # normalised to a placeholder so the verbatim assertion stays stable
        # across runs without freezing the clock.
        system_text = [
            m.get("content", "") for m in first_request["messages"] if m.get("role") == "system"
        ][0]
        lines = system_text.split("\n")
        assert lines[0].startswith("Current date and time: ")
        rest = "\n".join(lines[1:])
        rest = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "YYYY-MM-DD HH:MM", rest)
        expected = _BASIC_FLOW_EXPECTED
        assert rest == expected, f"System prompt mismatch:\n{rest!r}\n\nvs expected:\n{expected!r}"

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

        # No conversation echo thoughts should be logged
        # (old _log_conversation_thought is removed; thoughts come from tool reasoning only)
        thoughts = penny.db.thoughts.get_recent(TEST_SENDER, limit=10)
        conversation_echoes = [
            t for t in thoughts if t.content.startswith("Conversation: user said")
        ]
        assert len(conversation_echoes) == 0, "Conversation echo thoughts should not be logged"


# ── 1b. Provenance stamping ─────────────────────────────────


@pytest.mark.asyncio
async def test_collection_create_stamps_chat_provenance(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """A collection created from a chat message records that message as its
    source and the turn's run as its creator (#1566).

    Proves the provenance is threaded end-to-end — the channel logs the incoming
    message first, ``handle`` mints the run_id and passes both down to
    ``collection_create`` — rather than reconstructed after the fact.
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
                            name="collection_create",
                            arguments={
                                "name": "indie-platformers",
                                "intent": "a running list of new indie platformers",
                                "skill": "gather-platformers",
                                "interval": 3600,
                            },
                        ),
                    )
                ],
            ),
            model="test-model",
        )

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        # The front door instantiates a skill (#1591) — seed the hole-less skill the
        # chat turn instantiates, so the create resolves and stamps its provenance.
        penny.db.skills.upsert(
            SkillDraft(
                name="gather-platformers",
                intent="gather new indie platformers",
                description="gather new indie platformers",
                steps=[
                    SkillStep(
                        ordinal=1,
                        source_ordinal=1,
                        tool="browse",
                        arguments={"queries": ["new indie platformers"], "extract": "the newest"},
                        substitutions=[],
                    )
                ],
                holes=[],
                source_run_id="run-teach",
            ),
            author="chat",
        )
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
        # enumerable by ``read_run_calls(target="chat")`` and stamps no run_target —
        # the served-entity closure that identifies it by its cause.
        chat_runs = penny.db.messages.run_call_groups(PennyConstants.CHAT_AGENT_NAME, None, 20)
        this_run = next(
            (g for g in chat_runs if any(p.run_id == row.created_by_run_id for p in g)), None
        )
        assert this_run is not None
        assert all(p.run_target is None for p in this_run)


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
        penny.db.memories.create_collection(
            "likes-test", "positive prefs", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        penny.db.memories.create_collection(
            "dislikes-test", "negative prefs", Inclusion.NEVER, RecallMode.RECENT
        )
        penny.db.memories.create_log(
            "messages-test", "convo log", Inclusion.ALWAYS, RecallMode.RECENT
        )

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
        penny.db.memories.create_collection(
            "retired-test", "archived", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
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
        assert "update_entry" in names
        assert "collection_delete_entry" in names
        assert "log_append" in names

        # Lifecycle / shape.
        assert "collection_create" in names
        assert "collection_update" in names
        assert "log_create" in names
        assert "collection_archive" in names
        assert "collection_unarchive" in names

        # Reads.
        assert "collection_read_latest" in names
        assert "read_similar" in names
        assert "collection_get" in names
        assert "collection_keys" in names
        assert "memory_metadata" in names
        assert "collection_catalog" in names
        assert "collector_run_history" in names
        # Resolve-by-meaning — the guess-free fallback every not-found points at (#1558).
        assert "find_mine" in names

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
        penny.db.memories.create_collection(
            "backfill-gap", "gap fixture", Inclusion.RELEVANT, RecallMode.RELEVANT
        )
        # Content vector present, key vector NULL (the gap state).
        penny.db.memory("backfill-gap").write(
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


_BASIC_FLOW_EXPECTED = (
    "\n"
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
    "The user is talking to you — no greetings, no sign-offs, just pick up the thread.\n"
    "\n"
    "Don't chase down topics the user only mentioned in passing. When they're just "
    "sharing news, reacting to their day, or thinking out loud, reply like a friend and "
    "don't run a browse or lookup they didn't ask for. Two things are still yours to act "
    "on: when they tell you about themselves — what they like, dislike, or are into — "
    "remember it; and when they directly ask you to look something up, save, recall, "
    "change, or check something, do it.\n"
    "\n"
    "Every tool call has a `reasoning` field — use it to think out loud. Explain what "
    "you're looking for, what you already know, and what you'll do with the result.\n"
    "\n"
    "Search memory before browsing. Your memory tools "
    "(`collection_read_latest(<collection>)`, `read_similar(query=<query>)`, "
    "`log_read(<log>)`, etc.) read everything stored — the 'Your memory' list in the "
    "'Penny's current state' section below names every store you can pull from, and the "
    "mechanisms + recent activity there are your own operational state (what you're "
    "running, what you just did). Only browse if memory doesn't have what the user needs, "
    "or for current/external info (news, products, prices, fresh facts).\n"
    "\n"
    "Compose your tools directly to satisfy what the user asks. If the user teaches you a "
    'new pattern ("from now on when I say X, do Y"), write it as a new entry in your '
    "`skills` collection so it's saved for next time.\n"
    "\n"
    "When a 'Current Browser Page' section appears above, the user is browsing that page "
    "right now. If they say 'this page', 'this thread', 'this article', or anything "
    "ambiguous, they mean the Current Browser Page — not something from earlier in the "
    "conversation.\n"
    "\n"
    "How to use the browse tool:\n"
    "1. If the user gave you URLs, read them directly — pass the URLs in the queries "
    "array. Do NOT search for a site the user already linked.\n"
    "2. If the user gave you a topic (no URLs), call browse to discover relevant pages.\n"
    "3. Read the most promising pages by passing their URLs in the queries array (e.g., "
    'queries: ["https://example.com/page"]). Real pages have full details that search '
    "snippets leave out.\n"
    "\n"
    "After reading pages, you MUST respond with what you found. Do not make additional "
    "tool calls to re-fetch or supplement pages you already read. If a page had limited "
    "content, report what was there.\n"
    "\n"
    "Do NOT answer from search snippets alone — read actual pages first.\n"
    "\n"
    "Every fact, name, and detail in your response must come from pages you read or your "
    "memory — not from search snippet summaries.\n"
    "\n"
    "Search results contain a 'Sources:' section at the bottom with real URLs. When you "
    "reference something from a search, use ONLY these source URLs. Copy them exactly — "
    "character for character. If a topic has no matching source URL, mention it without a "
    "URL.\n"
    "\n"
    "When the user changes topics, just go with it.\n"
    "\n"
    "Open your reply with the story of what you just did:\n"
    "1. Each tool result you got this turn opens with a first-person line naming what "
    'that call actually did — e.g. "You searched for X and found…", "You saved X to '
    '`likes`", "You didn\'t add anything new — it was already there", "You couldn\'t find X '
    'to remove". Lead your reply with a brief, natural recap that reflects EACH of those '
    "lines, in order — every call this turn, whether it succeeded, changed nothing, or "
    "failed — woven into a sentence, NOT a bulleted log.\n"
    "2. Mirror the OUTCOME each tool reported, never what you set out to do: if a save "
    "was already there, say it was already there; if a lookup came back empty, say so; if "
    "a call failed, say so. NEVER imply something changed when the tool said it didn't.\n"
    "3. Then give the answer.\n"
    "On a plain reply with no tool calls, skip the recap and just respond.\n"
    "\n"
    "Always include specific details (specs, dates, prices) and at least one source URL "
    "so the user can follow up.\n"
    "\n"
    "## Penny's current state\n"
    "\n"
    "### Active mechanisms\n"
    "- dislikes — active · every 5m · no runs yet\n"
    "- knowledge — active · every 5m · no runs yet\n"
    "- likes — active · every 5m · no runs yet\n"
    "- notified-thoughts — archived YYYY-MM-DD HH:MM UTC · no runs yet\n"
    "- notifier — archived YYYY-MM-DD HH:MM UTC · no runs yet\n"
    "- quality — active · every 1h · no runs yet\n"
    "- skills — active · every 6h · no runs yet\n"
    "- thoughts — active · every 90m · no runs yet\n"
    "- unnotified-thoughts — archived YYYY-MM-DD HH:MM UTC · no runs yet\n"
    "\n"
    "### Recent activity\n"
    "change · YYYY-MM-DD HH:MM UTC · skills updated by user-run — changed inclusion\n"
    "change · YYYY-MM-DD HH:MM UTC · old-facts archived by user-run\n"
    "change · YYYY-MM-DD HH:MM UTC · old-facts created by user-run\n"
    "change · YYYY-MM-DD HH:MM UTC · secrets created by user-run\n"
    "change · YYYY-MM-DD HH:MM UTC · playlists created by user-run\n"
    "\n"
    "### Your memory\n"
    "- browse-results (log, 0 entries) — Every browse-tool fetch result\n"
    "- collector-runs (log, 0 entries) — One entry per Collector cycle: target + success "
    "marker + done() summary\n"
    "- dislikes (collection, 0 entries) — Topics the user has expressed negative "
    "sentiment about\n"
    "- knowledge (collection, 0 entries) — Summarized facts from web pages Penny has read\n"
    "- likes (collection, 0 entries) — Topics the user has expressed positive sentiment "
    "about\n"
    "- penny-messages (log, 0 entries) — Every outgoing Penny reply\n"
    "- playlists (collection, 1 entries) — favorite playlists\n"
    "- quality (collection, 0 entries) — Reviews Penny's own runs and messages and "
    "corrects collection prompts that have drifted from their stated intent\n"
    "- secrets (collection, 1 entries) — hidden\n"
    "- skills (collection, 13 entries) — Workflow patterns — how to compose tools to "
    "satisfy user intents\n"
    "- thoughts (collection, 0 entries) — Penny's inner-monologue thoughts about the "
    "user's interests.\n"
    "- tips (log, 1 entries) — useful tips\n"
    "- user-messages (log, 0 entries) — Every incoming user message\n"
    "\n"
    "### About the user\n"
    "- name: Test User\n"
    "- timezone: America/Los_Angeles\n"
    "- location: Seattle, WA\n"
    "\n"
    "To look deeper: memory_metadata(<name>) for a collection's full config and change "
    "history, read_run_calls(<target>) for a run's tool calls, "
    "collection_read_latest(<name>) or read_similar(memory=<name>, anchor=<text>) for "
    "stored entries, find_mine(query=<text>) to resolve a name by meaning, and "
    "collection_catalog() for every collection."
)
