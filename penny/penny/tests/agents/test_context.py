"""Integration tests for Agent context building (conversation, profile, page hint, sentiment)."""

import json

import pytest
from sqlmodel import Session, col, select

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import MessageLog, PromptLog
from penny.llm.models import LlmMessage
from penny.tests.conftest import TEST_SENDER
from penny.tools.link_handles import HandleTable

# ── History window size — the pure grow/cut function (#2756) ────────────


@pytest.mark.parametrize(
    "total_user_turns, window, expected",
    [
        (1, 5, 1),  # fewer turns than the window — show everything so far
        (5, 5, 5),  # exactly the window — nothing to cut yet
        (6, 5, 6),  # growing
        (9, 5, 9),  # the growth ceiling (2 * 5 - 1)
        (10, 5, 5),  # the cut — back to the window
        (11, 5, 6),  # growth resumes after the cut
        (4, 2, 2),  # a cut at a small window (2 * 2 = 4)
        (3, 2, 3),  # growth at a small window (2 * 2 - 1 = 3)
    ],
)
def test_history_window_size_is_a_pure_grow_cut_function(total_user_turns, window, expected):
    """``Agent._history_window_size`` is a pure function of the TRUE turn
    count and the configured window (#2756) — no stored cursor, so the same
    inputs always give the same answer, and consecutive turns (M, M+1) never
    disagree about a turn already in the window."""
    assert Agent._history_window_size(total_user_turns, window) == expected


@pytest.mark.asyncio
async def test_count_user_turns_is_uncapped_by_the_message_context_limit(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """``MessageStore.count_user_turns`` is the anchor the grow/cut window is
    computed against (#2756) — it must keep counting past
    MESSAGE_CONTEXT_LIMIT, unlike the capped raw pool ``get_messages_since``
    fetches, or the window would freeze on an old cut point in a long-lived
    conversation."""
    config = make_config(message_context_limit=3)

    async with running_penny(config) as penny:
        for n in range(1, 6):  # five user turns — two more than the raw cap
            penny.db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                TEST_SENDER,
                f"user message {n}",
            )

        assert penny.db.messages.count_user_turns(TEST_SENDER) == 5


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
async def test_conversation_carries_the_replies_and_their_run_ids(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The history carries each past reply as the user received it and nothing
    else (#2646): no tool line appended, whatever the run called — the line read
    as part of how her own replies look, and one real turn copied it in place
    of a call.  The framework's run-id stamping stays: the reply's row carries
    the same run id the chat run's promptlog rows carry, so a reader of the
    record can still walk from a reply to the calls behind it."""
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
        # rows carry — both stamped by the framework.
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

        # A second turn that only talked, and an autonomous send between turns
        # with no run behind it — logged the way the drainer logs one.
        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "glad that helped!"),
            answers_micro_contexts=False,
        )
        await signal_server.push_message(sender=TEST_SENDER, content="thanks!")
        await signal_server.wait_for_message(timeout=10.0)
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
            # The browse read a page, so the reply-side URL fallback had a source
            # to append (#2648): the reply carries the read page's URL.  No tool
            # line after any reply (#2646): the history carries the replies alone.
            "here's what i found!\n\nhttps://weather.example.com/today",
            "glad that helped!\na proactive note",
        ]


@pytest.mark.asyncio
async def test_conversation_window_grows_then_cuts_back(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The history window grows by one exchange per turn from
    CHAT_HISTORY_TURNS up to twice that minus one, then cuts back (#2756;
    previously a flat last-N, #2746) — most turns extend the prior turn's
    history (an append-only prefix) rather than replacing it outright."""
    config = make_config()  # default window: 5

    async with running_penny(config) as penny:
        sizes: list[int] = []
        for n in range(1, 12):  # 11 exchanges — through one full grow/cut cycle
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
            )
            conversation = penny.chat_agent._build_conversation(TEST_SENDER)
            sizes.append(len(conversation) // 2)  # user turns kept, each with its reply

        # Grows 1..5 (nothing to trim yet), then 6,7,8,9 (up to 2*5-1), then
        # cuts to 5 at the 10th turn, then grows again to 6 at the 11th.
        assert sizes == [1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6]

        # At the 9th turn (window size 9, its maximum before the cut), the
        # oldest kept turn is turn 1 — everything so far.
        conversation = penny.chat_agent._build_conversation(TEST_SENDER)  # after turn 11
        # The 11th turn's window (size 6) keeps turns 6-11.
        assert [content for _, content in conversation] == [
            "user message 6",
            "reply 6",
            "user message 7",
            "reply 7",
            "user message 8",
            "reply 8",
            "user message 9",
            "reply 9",
            "user message 10",
            "reply 10",
            "user message 11",
            "reply 11",
        ]


@pytest.mark.asyncio
async def test_conversation_window_follows_config_value(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """CHAT_HISTORY_TURNS is a runtime config param: a changed value changes the
    window on the next build.  Four exchanges at a window of 2 land exactly on
    a cut point (2 * 2 = 4), so this also pins the grow/cut formula's cut case,
    not just its growth case (#2756)."""
    config = make_config(chat_history_turns=2)

    async with running_penny(config) as penny:
        for n in range(1, 5):
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
            )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        assert [content for _, content in conversation] == [
            "user message 3",
            "reply 3",
            "user message 4",
            "reply 4",
        ]


@pytest.mark.asyncio
async def test_thread_history_grows_then_cuts_back(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A quote-reply's thread history follows the same grow/cut window
    (#2756; previously a flat last-N cap, #2746).  The thread path has no
    cheap global turn count of its own, so it anchors on the fetched thread's
    own user-turn count (a short-chain approximation, flagged in the PR) —
    exact here since the chain (8 exchanges) never approaches its own
    20-message fetch cap."""
    config = make_config()  # default window: 5

    async with running_penny(config) as penny:
        previous_id = None
        for n in range(1, 9):  # an 8-exchange chain — within the window's growth (up to 9)
            user_id = penny.db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                TEST_SENDER,
                f"user message {n}",
                parent_id=previous_id,
            )
            previous_id = penny.db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                penny.config.signal_number,
                f"reply {n}",
                parent_id=user_id,
                recipient=TEST_SENDER,
            )

        thread_history = penny.chat_agent.get_history(TEST_SENDER, thread_parent_id=previous_id)
        # 8 exchanges is within the window's growth (5..9) — everything so far.
        assert [content for _, content in thread_history] == [
            "user message 1",
            "reply 1",
            "user message 2",
            "reply 2",
            "user message 3",
            "reply 3",
            "user message 4",
            "reply 4",
            "user message 5",
            "reply 5",
            "user message 6",
            "reply 6",
            "user message 7",
            "reply 7",
            "user message 8",
            "reply 8",
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
    ChatAgent._inject_page_context(messages, page_context, HandleTable())

    assert len(messages) == 4
    # The stored/logged shape (what ``to_log_dict`` renders): the injected call's
    # ``arguments`` is a JSON string like every other tool call (#2604).
    rendered = [message.to_log_dict() for message in messages]
    # Assistant tool call uses BrowseTool format (name="browse", the page's own
    # handle in queries — a URL the model reads arrives as a handle, #2717) and
    # carries the standard id/type so the result can reference it.
    assert rendered[2]["role"] == "assistant"
    assert rendered[2]["tool_calls"][0]["id"] == ChatAgent.PAGE_CONTEXT_TOOL_CALL_ID
    assert rendered[2]["tool_calls"][0]["type"] == "function"
    assert rendered[2]["tool_calls"][0]["function"]["name"] == "browse"
    assert json.loads(rendered[2]["tool_calls"][0]["function"]["arguments"])["queries"] == ["[1]"]
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
    # The page body (title + content) is preserved after the framing; the page's
    # own address renders as its handle, never a raw URL (#2717).
    assert "$49.99" in rendered[3]["content"]
    assert "Example Product Page" in rendered[3]["content"]
    assert "https://example.com/product" not in rendered[3]["content"]
    assert "URL: [1]" in rendered[3]["content"]


def test_page_context_not_injected_when_empty():
    """No injection when page context has no text."""
    from penny.agents.chat import ChatAgent
    from penny.channels.base import PageContext

    messages: list[LlmMessage] = [LlmMessage(role="user", content="hi")]
    ChatAgent._inject_page_context(
        messages, PageContext(title="T", url="U", text=""), HandleTable()
    )
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
