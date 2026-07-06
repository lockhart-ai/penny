"""Integration tests for Agent context building (conversation, profile, page hint, sentiment)."""

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER

# ── Conversation building ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversation_builds_user_assistant_turns(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Conversation history alternates user/assistant turns."""
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "hello penny",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "hey there!",
            parent_id=1,
            recipient=TEST_SENDER,
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        assert len(conversation) == 2
        assert conversation[0][0] == "user"
        assert "hello penny" in conversation[0][1]
        assert conversation[1][0] == "assistant"
        assert "hey there" in conversation[1][1]


@pytest.mark.asyncio
async def test_conversation_merges_consecutive_same_role(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Consecutive messages from the same role are merged with newlines.

    Both threaded replies (parent_id set) and autonomous outgoing sends
    (parent_id=None — what ``send_message`` from collector cycles
    produces) flow into the chat-turns array so Penny still sees the
    prior turn when the user replies to a notification.
    """
    config = make_config()

    async with running_penny(config) as penny:
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "first message",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "second message",
        )
        # Direct reply (parent_id set).
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "response",
            parent_id=2,
            recipient=TEST_SENDER,
        )
        # Autonomous notification (no parent_id) — also part of the
        # conversation now.
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "proactive thought",
            recipient=TEST_SENDER,
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        contents = " ".join(c for _, c in conversation)
        # Two user messages merged into one turn, then the two outgoing
        # messages merged into the next turn.
        assert len(conversation) == 2
        assert "first message" in conversation[0][1]
        assert "second message" in conversation[0][1]
        assert "response" in contents
        assert "proactive thought" in contents


# ── Profile context ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_profile_context_includes_name(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Profile context includes user name."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._profile_section(TEST_SENDER)
        assert context is not None
        assert "Test User" in context


@pytest.mark.asyncio
async def test_profile_context_none_for_unknown_user(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Profile context returns None for users without profile info."""
    config = make_config()

    async with running_penny(config) as penny:
        context = penny.chat_agent._profile_section("+1999999999")
        assert context is None


# ── Page context ─────────────────────────────────────────────────────────


def test_page_context_injected_as_synthetic_tool_call():
    """Page context is injected as a search tool call + result in messages."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext
    from penny.tools import Tool, ToolResult

    page_context = PageContext(
        title="Example Product Page",
        url="https://example.com/product",
        text="This is a great product that costs $49.99",
    )
    messages: list[dict] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "what is this page?"},
    ]
    ChatAgent._inject_page_context(messages, page_context)

    assert len(messages) == 4
    # Assistant tool call uses BrowseTool format (name="browse", URL in queries)
    # and carries the standard id/type so the result can reference it.
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID
    assert messages[2]["tool_calls"][0]["type"] == "function"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "browse"
    assert messages[2]["tool_calls"][0]["function"]["arguments"]["queries"] == [
        "https://example.com/product"
    ]
    # Tool result: standard tool_call_id envelope (no ad-hoc tool_name) and the
    # same tagged first-person framing every real tool result gets (a successful
    # synthetic browse), so the web content can't read as a fresh instruction.
    # The synthetic call reads the page URL directly, so BrowseTool's per-tool
    # narration (#1480) frames it as "You opened <url>" — the same args the
    # injection passes reproduce the leading header.
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID
    assert "tool_name" not in messages[3]
    assert messages[3]["content"].startswith(
        Tool.format_result(
            "browse", {"queries": ["https://example.com/product"]}, ToolResult(message="")
        )
    )
    # The retained machine tag keeps the model parsing this as a tool result.
    assert "(browse result)" in messages[3]["content"]
    assert "$49.99" in messages[3]["content"]
    assert "Example Product Page" in messages[3]["content"]


def test_page_context_not_injected_when_empty():
    """No injection when page context has no text."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    messages: list[dict] = [{"role": "user", "content": "hi"}]
    ChatAgent._inject_page_context(messages, PageContext(title="T", url="U", text=""))
    assert len(messages) == 1  # unchanged


def test_page_hint_in_system_prompt():
    """System prompt includes a minimal page hint with title and URL."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    context = PageContext(title="Cool Article", url="https://example.com/article", text="content")
    agent = ChatAgent.__new__(ChatAgent)
    agent._pending_page_context = context
    hint = agent._page_hint_section()
    assert hint is not None
    assert "Cool Article" in hint
    assert "https://example.com/article" in hint
    # Should NOT contain the full text — that's in the tool result
    assert "content" not in hint
