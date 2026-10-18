"""Integration tests for Agent context building (conversation, profile, page hint, sentiment)."""

import json

import pytest
from sqlmodel import Session, col, select

from penny.constants import PennyConstants
from penny.database.models import MessageLog, PromptLog
from penny.llm.models import LlmMessage
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


@pytest.mark.asyncio
async def test_conversation_pairs_tool_calls_by_run_id(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Each past assistant turn carries the tool line its own run's ledger rows
    back (#2630), paired by the run id stamped on the reply — not guessed from
    timestamps.  A turn that only talked reads ``[Tool calls: none]``; an
    autonomous send with no run behind it carries no line at all."""
    config = make_config()

    async with running_penny(config) as penny:
        mock_llm.set_default_flow(
            final_response="here's what i found!",
            search_query="https://weather.example.com/today",
        )
        await signal_server.push_message(
            sender=TEST_SENDER, content="what's the weather like today?"
        )
        await signal_server.wait_for_message(timeout=10.0)

        # The reply's row carries the same run id the chat run's own promptlog
        # rows carry — both stamped by the framework, which is the whole join.
        with Session(penny.db.engine) as session:
            reply = session.exec(
                select(MessageLog)
                .where(
                    MessageLog.direction == PennyConstants.MessageDirection.OUTGOING,
                    col(MessageLog.run_id).is_not(None),
                )
                .order_by(col(MessageLog.id).desc())
            ).first()
            assert reply is not None, "the turn's reply must be logged"
            chat_run_ids = set(
                session.exec(
                    select(col(PromptLog.run_id)).where(
                        PromptLog.agent_name == PennyConstants.CHAT_AGENT_NAME
                    )
                ).all()
            )
        assert reply.run_id in chat_run_ids, (
            "the reply's run id must be the chat run's own id, not a guess"
        )

        # A second turn that only talked: its run traces with no tools.
        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "glad that helped!"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="thanks!")
        await signal_server.wait_for_message(timeout=10.0)

        # An autonomous send between turns has no run behind it — logged the way
        # the drainer logs one — and must carry no tool line.
        autonomous_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "a proactive note",
            recipient=TEST_SENDER,
        )
        assert autonomous_id is not None

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        assistant_turns = [content for role, content in conversation if role == "assistant"]
        assert assistant_turns == [
            "here's what i found!\n[Tool calls: browse]",
            "glad that helped!\n[Tool calls: none]\na proactive note",
        ]


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
    from penny.prompts import Prompt

    page_context = PageContext(
        title="Example Product Page",
        url="https://example.com/product",
        text="This is a great product that costs $49.99",
    )
    messages: list[LlmMessage] = [
        LlmMessage(role="system", content="system prompt"),
        LlmMessage(role="user", content="what is this page?"),
    ]
    ChatAgent._inject_page_context(messages, page_context)

    assert len(messages) == 4
    # The stored/logged shape (what ``to_log_dict`` renders): the injected call's
    # ``arguments`` is a JSON string like every other tool call (#2604).
    rendered = [message.to_log_dict() for message in messages]
    # Assistant tool call uses BrowseTool format (name="browse", URL in queries)
    # and carries the standard id/type so the result can reference it.
    assert rendered[2]["role"] == "assistant"
    assert rendered[2]["tool_calls"][0]["id"] == ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID
    assert rendered[2]["tool_calls"][0]["type"] == "function"
    assert rendered[2]["tool_calls"][0]["function"]["name"] == "browse"
    assert json.loads(rendered[2]["tool_calls"][0]["function"]["arguments"])["queries"] == [
        "https://example.com/product"
    ]
    # Tool result: standard tool_call_id envelope (no ad-hoc tool_name) and the same
    # tagged first-person envelope every real tool result gets, but with BESPOKE
    # narration (#1485) naming this as the page the user is viewing — NOT BrowseTool's
    # per-tool "You opened <url>" narration (#1480): a page-context injection is the
    # page the user is looking at, not a browse Penny ran — so the web content can't
    # read as a fresh instruction.
    assert rendered[3]["role"] == "tool"
    assert rendered[3]["tool_call_id"] == ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID
    assert "tool_name" not in rendered[3]
    assert rendered[3]["content"].startswith(f"{Prompt.PAGE_CONTEXT_NARRATION} (browse result)\n")
    # The narration names the page-context provenance (not the generic "You used ...").
    assert "the page the user is currently viewing" in rendered[3]["content"]
    # The retained machine tag keeps the model parsing this as a tool result.
    assert "(browse result)" in rendered[3]["content"]
    # The page body (title + content) is preserved after the framing.
    assert "$49.99" in rendered[3]["content"]
    assert "Example Product Page" in rendered[3]["content"]


def test_page_context_not_injected_when_empty():
    """No injection when page context has no text."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    messages: list[LlmMessage] = [LlmMessage(role="user", content="hi")]
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
