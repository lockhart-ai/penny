"""Integration tests for ChatAgent message handling.

Test organization:
1. Full integration (happy path) — comprehensive end-to-end message flow
2. Special success cases — no tool call, anti-refusal
3. Error / edge cases — XML leak regression, short response warning, delivery failure
4. Memory inventory (rendered for every agent's prompt)
5. Ambient recall (chat-only — each recall mode + self-match exclusion)
6. Tool surface (chat-only — entry-mutation tools removed)
"""

import hashlib
import re
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlmodel import Session, select

from penny.database.memory_store import EntryInput, LogEntryInput, RecallMode
from penny.database.models import MemoryEntry, MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until

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
        # migrations 0026 and 0027 (user-messages, penny-messages,
        # browse-results, likes, dislikes, knowledge, notified-thoughts,
        # unnotified-thoughts).
        # Active memories rendered in alphabetical order: "playlists" < "tips".
        penny.db.memories.create_collection("playlists", "favorite playlists", RecallMode.ALL)
        penny.db.memories.write(
            "playlists",
            [EntryInput(key="morning", content="prog rock")],
            author="user",
        )
        penny.db.memories.create_log("tips", "useful tips", RecallMode.RECENT)
        penny.db.memories.append(
            "tips", [LogEntryInput(content="tune before playing")], author="user"
        )
        # Off and archived memories are seeded with entries so the verbatim
        # prompt assertion below proves they are filtered out of ambient recall.
        penny.db.memories.create_collection("secrets", "hidden", RecallMode.OFF)
        penny.db.memories.write(
            "secrets",
            [EntryInput(key="do-not-share", content="classified")],
            author="user",
        )
        penny.db.memories.create_collection("old-facts", "archived", RecallMode.ALL)
        penny.db.memories.write(
            "old-facts",
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
        expected = """\

## Identity
You are Penny. You and the user are friends who text regularly. \
This is mid-conversation — not a fresh chat.

Voice:
- Reply like you're continuing a text thread.
- React to what the user actually said before giving information. \
If they corrected you, own it. If they expressed excitement, match it. \
If they asked a follow-up, connect it to what came before.
- Present information naturally but you can still use short formatted blocks \
(bold names, links) when listing products or facts. \
Just wrap them in conversational text, not a clinical dump.
- Finish every message with an emoji.

## Context
### User Profile
The user's name is Test User.

### Memory Inventory
- browse-results (log, 0 entries) — Every browse-tool fetch result
- collector-runs (log, 0 entries) — One entry per Collector cycle: \
target + success marker + done() summary
- dislikes (collection, 0 entries) — Topics the user has expressed negative sentiment about
- knowledge (collection, 0 entries) — Summarized facts from web pages Penny has read
- likes (collection, 0 entries) — Topics the user has expressed positive sentiment about
- notified-thoughts (collection, 0 entries) — Thoughts already shared with the user
- penny-messages (log, 0 entries) — Every outgoing Penny reply
- playlists (collection, 1 entries) — favorite playlists
- secrets (collection, 1 entries) — hidden
- tips (log, 1 entries) — useful tips
- unnotified-thoughts (collection, 0 entries) — Pending thoughts to share with the user
- user-messages (log, 1 entries) — Every incoming user message

### playlists
favorite playlists

#### [morning] · YYYY-MM-DD HH:MM
prog rock

### tips
useful tips

#### YYYY-MM-DD HH:MM
tune before playing

## Instructions
The user is talking to you — no greetings, no sign-offs, just pick up \
the thread.

Every tool call has a `reasoning` field — use it to think out loud. \
Explain what you're looking for, what you already know, \
and what you'll do with the result.

Search memory first. The recall block above shows the most relevant \
entries verbatim, and your memory tools (`read_latest`, \
`read_similar`, etc.) cover everything else stored. \
Only browse if memory \
doesn't have what the user needs, or for current/external info \
(news, products, prices, fresh facts).

When the user wants to start tracking a new topic — a trip, project, \
list of recipes, anything — call ``collection_create`` with a clear \
``extraction_prompt``. The extraction_prompt is the brain of a \
background agent that fills the collection from chat and browse \
activity automatically; without it the collection stays empty. \
You do NOT curate entries yourself — there's no write tool on your \
surface. Just create the collection, mention it in your reply \
("I'll keep a list of Prague spots for you"), and continue the \
conversation; the collector does the rest in the background. A \
good extraction_prompt names what to extract, which logs to read \
(usually penny-messages and browse-results for research topics), \
and how to handle corrections (update or delete stale entries when \
the user flags them).

When a 'Current Browser Page' section appears above, the user is browsing \
that page right now. If they say 'this page', 'this thread', 'this article', \
or anything ambiguous, they mean the Current Browser Page — not something \
from earlier in the conversation.

How to use the browse tool:
1. If the user gave you URLs, read them directly — pass the URLs in the \
queries array. Do NOT search for a site the user already linked.
2. If the user gave you a topic (no URLs), call browse to discover \
relevant pages.
3. Read the most promising pages by passing their URLs in the queries \
array (e.g., queries: ["https://example.com/page"]). \
Real pages have full details that search snippets leave out.

After reading pages, you MUST respond with what you found. Do not make \
additional tool calls to re-fetch or supplement pages you already read. \
If a page had limited content, report what was there.

Do NOT answer from search snippets alone — read actual pages first.

Every fact, name, and detail in your response must come from pages you \
read or your recall context — not from search snippet summaries.

Search results contain a 'Sources:' section at the bottom with real URLs. \
When you reference something from a search, use ONLY these source URLs. \
Copy them exactly — character for character. If a topic has no matching \
source URL, mention it without a URL.

When the user changes topics, just go with it.

Always include specific details (specs, dates, prices) and at least one \
source URL so the user can follow up."""
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

        # Stage 9 side-effect writes: every message flow populates the three
        # system log memories so the recall assembler can read them next turn.
        user_msg_entries = penny.db.memories.read_all("user-messages")
        assert any(e.content == "what's the weather like today?" for e in user_msg_entries)
        assert all(e.author == "user" for e in user_msg_entries)

        penny_msg_entries = penny.db.memories.read_all("penny-messages")
        assert any("here's what i found" in e.content.lower() for e in penny_msg_entries)
        assert all(e.author == "chat" for e in penny_msg_entries)

        browse_entries = penny.db.memories.read_all("browse-results")
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


# ── 1b. Ambient-recall integration cases ─────────────────────────────────


@pytest.mark.asyncio
async def test_chat_prompt_renders_relevant_mode_via_embedding(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A collection with recall=relevant surfaces its matching entry in the chat prompt.

    The entry's content_embedding and the current-message embedding are both
    the same unit vector (cosine=1), so the entry ranks first against the
    0.0 floor.  A second orthogonal entry stays below nothing — with floor=0.0
    it's included too, but only the matching one is asserted.
    """
    config = make_config()
    match_vec = [1.0, 0.0, 0.0]

    async with running_penny(config) as penny:
        penny.db.memories.create_collection("trivia", "facts", RecallMode.RELEVANT)
        penny.db.memories.write(
            "trivia",
            [
                EntryInput(
                    key="espresso",
                    content="espresso uses 9 bars of pressure",
                    content_embedding=match_vec,
                )
            ],
            author="user",
        )

        mock_client = AsyncMock()
        mock_client.embed = AsyncMock(side_effect=lambda texts: [match_vec] * len(texts))
        penny.chat_agent._embedding_model_client = mock_client
        penny.chat_agent._pending_page_context = None

        prompt = await penny.chat_agent._build_system_prompt(
            TEST_SENDER, content="tell me about espresso"
        )

    assert "### trivia" in prompt
    assert "facts" in prompt
    assert "[espresso]" in prompt
    assert "espresso uses 9 bars of pressure" in prompt


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


@pytest.mark.asyncio
async def test_conversation_prompt_includes_antirefusal_instruction(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """
    Regression test for #775: CONVERSATION_PROMPT must include an explicit instruction
    to never refuse a request, so the model always provides something useful.
    """
    mock_llm.set_default_flow(
        final_response="here are some vegan options! 🌱",
    )

    async with running_penny(test_config):
        await signal_server.push_message(
            sender=TEST_SENDER,
            content="what are the best vegan restaurants?",
        )
        await signal_server.wait_for_message(timeout=10.0)

    # Verify the system prompt instructs the model to always provide something useful
    first_request = mock_llm.requests[0]
    messages = first_request.get("messages", [])
    system_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
    assert "must respond with what you found" in system_text.lower(), (
        "CONVERSATION_PROMPT should instruct the model to respond with available results"
    )


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
async def test_short_response_logged_as_warning(
    signal_server, mock_llm, test_config, test_user_info, running_penny, caplog
):
    """
    Regression test for #775: short/apologetic responses should be logged as warnings.

    When the model returns a very short response (< 10 words), a warning is logged
    to make these cases visible for debugging. The response is still delivered to
    the user — the warning is diagnostic, not suppressive.
    """
    import logging

    apologetic_response = "I'm sorry, but I can't help with that."

    def handler(request, count):
        return mock_llm._make_text_response(request, apologetic_response)

    mock_llm.set_response_handler(handler)

    with caplog.at_level(logging.WARNING, logger="penny.agents.base"):
        async with running_penny(test_config):
            await signal_server.push_message(
                sender=TEST_SENDER,
                content="what are the best vegan restaurants in downtown metropolis?",
            )
            response = await signal_server.wait_for_message(timeout=10.0)

    # Response is still delivered to the user
    assert response["message"] == apologetic_response

    # Warning was logged for the short response
    short_response_warnings = [r for r in caplog.records if "Short response detected" in r.message]
    assert len(short_response_warnings) >= 1, "Should log a warning for short responses"


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
        penny.db.memories.create_collection("likes-test", "positive prefs", RecallMode.RELEVANT)
        penny.db.memories.create_collection("dislikes-test", "negative prefs", RecallMode.OFF)
        penny.db.memories.create_log("messages-test", "convo log", RecallMode.RECENT)

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
        penny.db.memories.create_collection("retired-test", "archived", RecallMode.RELEVANT)
        penny.db.memories.archive("retired-test")

        result = penny.chat_agent._memory_inventory_section()

        assert result is not None
        assert "retired-test" not in result


# ── 5. Ambient recall ─────────────────────────────────────────────────────


def _hash_embed_vec(text: str, dim: int = 4096) -> list[float]:
    """Bag-of-words deterministic embedding for similarity tests.

    Each word picks an axis via SHA-256; the vector is L2-normalised so
    cosine is comparable across strings.  Identical strings map to
    identical vectors; strings sharing words have meaningful cosine > 0;
    fully-distinct strings map to cosine = 0.
    """
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


def _install_hash_embedding(agent) -> None:
    """Replace the agent's embedding client with a deterministic hash embedder."""
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(side_effect=lambda texts: [_hash_embed_vec(t) for t in texts])
    agent._embedding_model_client = mock_client


def _write_embedded(db, name: str, key: str | None, content: str) -> None:
    """Write an entry with a deterministic content embedding."""
    vec = _hash_embed_vec(content)
    if key is None:
        db.memories.append(
            name, [LogEntryInput(content=content, content_embedding=vec)], author="test"
        )
    else:
        db.memories.write(
            name, [EntryInput(key=key, content=content, content_embedding=vec)], author="test"
        )


def _backfill_created_at(db, name: str, content: str, when: datetime) -> None:
    """Override an entry's created_at timestamp for temporal-window tests."""
    with Session(db.engine) as session:
        rows = session.exec(
            select(MemoryEntry).where(
                MemoryEntry.memory_name == name, MemoryEntry.content == content
            )
        ).all()
        for row in rows:
            row.created_at = when
            session.add(row)
        session.commit()


@pytest.mark.asyncio
async def test_recall_recent_mode_renders_latest_entries(
    signal_server, mock_llm, test_config, running_penny
):
    async with running_penny(test_config) as penny:
        penny.db.memories.create_log("conversation-test", "shared chat log", RecallMode.RECENT)
        penny.db.memories.append(
            "conversation-test", [LogEntryInput(content="first message")], author="test"
        )
        penny.db.memories.append(
            "conversation-test", [LogEntryInput(content="second message")], author="test"
        )

        result = await penny.chat_agent._recall_section(current_message="anything")

        assert result is not None
        assert "first message" in result
        assert "second message" in result


@pytest.mark.asyncio
async def test_recall_all_mode_renders_all_entries(
    signal_server, mock_llm, test_config, running_penny
):
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("playlists-test", "saved playlists", RecallMode.ALL)
        penny.db.memories.write(
            "playlists-test",
            [EntryInput(key="morning", content="prog rock")],
            author="test",
        )
        penny.db.memories.write(
            "playlists-test",
            [EntryInput(key="evening", content="lo-fi")],
            author="test",
        )

        result = await penny.chat_agent._recall_section(current_message=None)

        assert result is not None
        assert "[morning]" in result and "prog rock" in result
        assert "[evening]" in result and "lo-fi" in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_uses_embedding(
    signal_server, mock_llm, test_config, running_penny
):
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("prefs-test", "user prefs", RecallMode.RELEVANT)
        _write_embedded(
            penny.db, "prefs-test", "coffee", "really loves dark roast coffee in the morning"
        )
        _write_embedded(
            penny.db, "prefs-test", "noise", "really hates loud construction at sunrise"
        )
        _install_hash_embedding(penny.chat_agent)

        result = await penny.chat_agent._recall_section(current_message="dark roast coffee")

        assert result is not None
        assert "really loves dark roast coffee in the morning" in result
        assert "really hates loud construction at sunrise" not in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_hybrid_lifts_via_history(
    signal_server, mock_llm, test_config, running_penny
):
    """A vague current message ('yeah') alone wouldn't surface the entry —
    the prior turn shares the entry's keywords, so weighted-decay scoring
    pulls the entry above the absolute floor."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("prefs-test", "user prefs", RecallMode.RELEVANT)
        _write_embedded(
            penny.db, "prefs-test", "coffee", "really loves dark roast coffee in the morning"
        )
        _install_hash_embedding(penny.chat_agent)

        result = await penny.chat_agent._recall_section(
            current_message="yeah",
            conversation_history=["dark roast coffee in the morning"],
        )

        assert result is not None
        assert "really loves dark roast coffee in the morning" in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_log_expands_with_temporal_neighbors(
    signal_server, mock_llm, test_config, running_penny
):
    """A single keyword match in a log pulls in neighboring entries via the
    temporal expansion window (±MEMORY_RELEVANT_NEIGHBOR_WINDOW_MINUTES)."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_log("conversation-test", "shared chat log", RecallMode.RELEVANT)
        _write_embedded(
            penny.db, "conversation-test", None, "dark roast coffee notes from this week"
        )
        _write_embedded(
            penny.db, "conversation-test", None, "follow up question number one in conversation"
        )
        _write_embedded(
            penny.db, "conversation-test", None, "follow up question number two in conversation"
        )
        _write_embedded(
            penny.db, "conversation-test", None, "stale earlier comment from last month"
        )

        base = datetime.now(UTC)
        _backfill_created_at(
            penny.db,
            "conversation-test",
            "stale earlier comment from last month",
            base - timedelta(hours=1),
        )
        _backfill_created_at(
            penny.db,
            "conversation-test",
            "dark roast coffee notes from this week",
            base - timedelta(minutes=2),
        )
        _backfill_created_at(
            penny.db,
            "conversation-test",
            "follow up question number one in conversation",
            base - timedelta(minutes=1),
        )
        _backfill_created_at(
            penny.db, "conversation-test", "follow up question number two in conversation", base
        )
        _install_hash_embedding(penny.chat_agent)

        result = await penny.chat_agent._recall_section(current_message="dark roast coffee")

        assert result is not None
        assert "dark roast coffee notes from this week" in result
        assert "follow up question number one in conversation" in result
        assert "follow up question number two in conversation" in result
        assert "stale earlier comment from last month" not in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_log_excludes_self_match(
    signal_server, mock_llm, test_config, running_penny
):
    """The current turn (and conversation-history turns) live in log corpora
    before recall runs.  Without filtering, those entries self-match their
    own anchor at cosine ≈ 1.0 and dominate the hit list.  ``anchor_contents``
    excludes them so historical content surfaces."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_log("user-messages-test", "incoming", RecallMode.RELEVANT)

        # Historical entry (30 days old) — contains the topic.
        _write_embedded(
            penny.db,
            "user-messages-test",
            None,
            "what video games should i try",
        )
        historical = datetime.now(UTC) - timedelta(days=30)
        _backfill_created_at(
            penny.db,
            "user-messages-test",
            "what video games should i try",
            historical,
        )

        # Current turn — same content as the anchor.
        current_text = "do you remember any conversations about video games"
        _write_embedded(penny.db, "user-messages-test", None, current_text)
        _install_hash_embedding(penny.chat_agent)

        result = await penny.chat_agent._recall_section(current_message=current_text)

        assert result is not None
        # Historical hit surfaces (not drowned out by the self-match)
        assert "what video games should i try" in result
        # Current-turn entry must NOT appear via the relevant path — it's the anchor.
        assert current_text not in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_collection_skips_temporal_expansion(
    signal_server, mock_llm, test_config, running_penny
):
    """Collections don't have a temporal-stream meaning, so similarity hits
    are returned without neighbor expansion even if entries are nearby in time."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("prefs-test", "user prefs", RecallMode.RELEVANT)
        _write_embedded(
            penny.db, "prefs-test", "coffee", "really loves dark roast coffee in the morning"
        )
        _write_embedded(
            penny.db, "prefs-test", "noise", "really hates loud construction at sunrise"
        )
        _install_hash_embedding(penny.chat_agent)

        result = await penny.chat_agent._recall_section(current_message="dark roast coffee")

        assert result is not None
        assert "really loves dark roast coffee in the morning" in result
        assert "really hates loud construction at sunrise" not in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_without_history_skips_vague_message(
    signal_server, mock_llm, test_config, running_penny
):
    """Vague current message with no history → absolute floor suppresses everything."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("prefs-test", "user prefs", RecallMode.RELEVANT)
        _write_embedded(
            penny.db, "prefs-test", "coffee", "really loves dark roast coffee in the morning"
        )
        _install_hash_embedding(penny.chat_agent)

        result = await penny.chat_agent._recall_section(current_message="yeah")

        # The test-only memory contributes nothing; existing system memories
        # are also unlikely to score above the floor for "yeah".
        assert result is None or "really loves dark roast coffee in the morning" not in result


@pytest.mark.asyncio
async def test_recall_relevant_mode_without_embedding_client_returns_none(
    signal_server, mock_llm, test_config, running_penny
):
    """No embedding client → relevant memories contribute nothing."""
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("prefs-test", "user prefs", RecallMode.RELEVANT)
        penny.db.memories.write(
            "prefs-test",
            [EntryInput(key="coffee", content="loves coffee")],
            author="test",
        )
        penny.chat_agent._embedding_model_client = None

        result = await penny.chat_agent._recall_section(current_message="coffee")

        # The test-only relevant memory can't render without embeddings;
        # any pre-existing system memory in non-relevant mode might still
        # render, so just assert the test-only entry is absent.
        assert result is None or "loves coffee" not in result


@pytest.mark.asyncio
async def test_recall_off_mode_skipped(signal_server, mock_llm, test_config, running_penny):
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("hidden-test", "not shown", RecallMode.OFF)
        penny.db.memories.write(
            "hidden-test",
            [EntryInput(key="k", content="classified content")],
            author="test",
        )

        result = await penny.chat_agent._recall_section(current_message=None)

        assert result is None or "classified content" not in result


@pytest.mark.asyncio
async def test_recall_archived_memory_skipped(signal_server, mock_llm, test_config, running_penny):
    async with running_penny(test_config) as penny:
        penny.db.memories.create_collection("old-test", "archived", RecallMode.RECENT)
        penny.db.memories.write(
            "old-test",
            [EntryInput(key="k", content="stale content")],
            author="test",
        )
        penny.db.memories.archive("old-test")

        result = await penny.chat_agent._recall_section(current_message=None)

        assert result is None or "stale content" not in result


# ── 6. Tool surface ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_tool_surface_excludes_entry_mutations(
    signal_server, mock_llm, test_config, running_penny
):
    """Chat must not have any tool that mutates collection entries.

    Entry curation (insert / update / delete / move) is owned by the
    per-collection collector subagents.  Stripping these from chat
    structurally prevents the "I added that to your memory" failure
    mode where chat narrates a write that never happened — chat
    literally has no write tool to fabricate a write with.
    """
    async with running_penny(test_config) as penny:
        names = {tool.name for tool in penny.chat_agent.get_tools()}

        # Forbidden entry mutations — those are collector-only
        assert "collection_write" not in names
        assert "update_entry" not in names
        assert "collection_move" not in names
        assert "collection_delete_entry" not in names
        assert "log_append" not in names

        # Lifecycle stays so the user can create / modify / archive collections
        # mid-conversation.  ``collection_update`` here is the metadata tool,
        # not the entry-content tool (that was renamed to ``update_entry``).
        assert "collection_create" in names
        assert "collection_update" in names
        assert "log_create" in names
        assert "collection_archive" in names
        assert "collection_unarchive" in names

        # Reads stay — chat needs them for "tell me what's saved" style queries
        assert "read_latest" in names
        assert "read_similar" in names
        assert "collection_get" in names
        assert "collection_keys" in names


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
