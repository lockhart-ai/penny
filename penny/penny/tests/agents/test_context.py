"""Integration tests for Agent context building (conversation, profile, page hint, sentiment)."""

import json
from datetime import timedelta

import pytest
from sqlmodel import Session, col, select

from penny.agents.base import Agent
from penny.constants import PennyConstants
from penny.database.models import MessageLog, PromptLog
from penny.llm.models import LlmMessage
from penny.tests.conftest import TEST_SENDER
from penny.tools.link_handles import HandleTable

# ── History window size — the pure session-window function (#3077) ──────


@pytest.mark.parametrize(
    "session_turns, total_user_turns, window, max_turns, expected",
    [
        (1, 1, 5, 20, 1),  # fewer turns than the window — show everything so far
        (5, 5, 5, 20, 5),  # exactly the window — nothing to cut yet
        (0, 10, 5, 20, 5),  # the session's first turn — the cut back to the window
        (1, 11, 5, 20, 6),  # growth resumes, one exchange per turn
        (15, 30, 5, 20, 20),  # the growth ceiling — the cap itself (5 + 15 % 16)
        (16, 31, 5, 20, 5),  # past the cap — the cut back to the window
        (31, 46, 5, 20, 20),  # the cap again, one full cycle later (5 + 31 % 16)
        (2, 6, 5, 20, 6),  # a small history — everything fits under the cap
        (2, 5, 2, 4, 4),  # a small window reaching its small cap (2 + 2 % 3)
        (3, 6, 2, 4, 2),  # the cap cut at a small window (2 + 3 % 3)
    ],
)
def test_history_window_size_is_a_pure_function_of_the_session(
    session_turns, total_user_turns, window, max_turns, expected
):
    """``Agent._history_window_size`` is a pure function of the session
    anchor (#3077) — no stored cursor, so the same state always gives the
    same answer, and consecutive turns within a session (M, M+1) never
    disagree about a turn already in the window."""
    assert (
        Agent._history_window_size(session_turns, total_user_turns, window, max_turns) == expected
    )


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

        # The session anchor (#3077) reads the same uncapped count, and — no
        # idle gap among the five — puts them all in one session.
        anchor = penny.db.messages.session_anchor(TEST_SENDER, 1800)
        assert anchor.total_user_turns == 5
        assert anchor.session_turns == 5


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
async def test_conversation_merges_consecutive_user_turns_only(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Consecutive INCOMING messages are merged with newlines, as always.

    Consecutive OUTGOING rows are NOT merged (#2974 — a threaded reply then an
    autonomous send with no incoming message between them used to melt into
    one voice with no seam, which is what let an autonomous send's own
    trigger-bearing turn disappear into whatever unrelated reply preceded it;
    each now keeps its own turn and its own run id)."""
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
        # Autonomous notification (no parent_id) — its own turn now, not
        # merged onto "response".
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "proactive thought",
            recipient=TEST_SENDER,
        )

        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        # One merged user turn, then "response" and "proactive thought" as
        # two SEPARATE assistant turns.
        assert len(conversation) == 3
        assert "first message" in conversation[0][1]
        assert "second message" in conversation[0][1]
        assert conversation[1].content == "response"
        assert conversation[2].content == "proactive thought"


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
        assistant_turns = [turn.content for turn in conversation if turn.role == "assistant"]
        assert assistant_turns == [
            # The browse read a page, so the reply-side URL fallback had a source
            # to append (#2648): the reply carries the read page's URL.  No tool
            # line after any reply (#2646): the history carries the replies alone.
            "here's what i found!\n\nhttps://weather.example.com/today",
            # The direct reply and the autonomous send right after it are TWO
            # separate turns (#2974), not merged into one voice — each keeps
            # its own run id.
            "glad that helped!",
            "a proactive note",
        ]


@pytest.mark.asyncio
async def test_conversation_window_grows_within_a_session_as_a_prefix_extension(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """Within a session the history window grows by one exchange per turn and
    never cuts (#3077; previously grew to 2N-1 and cut back about one turn in
    five, #2756) — every turn's history is a strict prefix-extension of the
    previous turn's, an append-only prefix that reuses the cached prompt
    prefix instead of rebuilding it."""
    config = make_config()  # default window: 5, session cap: 20

    async with running_penny(config) as penny:
        previous: list[str] = []
        for n in range(1, 12):  # 11 exchanges — one session, no cut anywhere
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
            contents = [turn.content for turn in conversation]
            # The new turn's history is the previous turn's, plus this
            # exchange — never a replacement.
            assert contents == [*previous, f"user message {n}", f"reply {n}"]
            previous = contents

        assert len(previous) == 22  # all 11 exchanges, each with its reply


@pytest.mark.asyncio
async def test_conversation_cuts_back_to_the_window_after_an_idle_gap(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """The first user message after an idle gap of at least
    CHAT_SESSION_IDLE_SECONDS starts a new session (#3077): the window cuts
    to the last CHAT_HISTORY_TURNS exchanges — the only cut an active
    conversation ever sees.  Before that message, the eight exchanges are
    still one (growing) session and the window keeps them all."""
    config = make_config()  # default window: 5, idle: 1800s

    async with running_penny(config) as penny:
        stale = penny.db.clock.now() - timedelta(hours=1)
        for n in range(1, 9):
            user_id = penny.db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                TEST_SENDER,
                f"user message {n}",
                timestamp=stale,
            )
            penny.db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                penny.config.signal_number,
                f"reply {n}",
                parent_id=user_id,
                recipient=TEST_SENDER,
                timestamp=stale,
            )
        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        assert len(conversation) == 16  # one session — everything kept

        # The ninth exchange lands an hour later: a new session, the window
        # cuts to the last 5 exchanges plus the session's own first one.
        user_id = penny.db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING,
            TEST_SENDER,
            "user message 9",
        )
        penny.db.messages.log_message(
            PennyConstants.MessageDirection.OUTGOING,
            penny.config.signal_number,
            "reply 9",
            parent_id=user_id,
            recipient=TEST_SENDER,
        )
        conversation = penny.chat_agent._build_conversation(TEST_SENDER)
        # The cut dropped exchanges 1-3; the window is the last 5 pre-session
        # exchanges plus the new session's own first one (its turn is now
        # logged, so the build sees session_turns=1 — the live turn itself
        # saw the flat 5).
        assert [turn.content for turn in conversation] == [
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
            "user message 9",
            "reply 9",
        ]


@pytest.mark.asyncio
async def test_conversation_cuts_once_at_the_session_cap(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A session whose history passes CHAT_SESSION_MAX_TURNS cuts back to
    CHAT_HISTORY_TURNS once and grows again (#3077) — a marathon stays
    bounded while cuts stay rare (every cap − window + 1 turns, not every
    window turns)."""
    config = make_config(chat_session_max_turns=8)  # window 5, cap 8 — a cycle of 4

    async with running_penny(config) as penny:
        sizes: list[int] = []
        for n in range(1, 14):  # one session of 13 exchanges
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
            sizes.append(len(conversation) // 2)

        # Grows one per turn to the cap (8 at the 11th), cuts once at the 12th
        # (8 stored turns — one past the cap), grows again — a cut every 4
        # turns at this small cap, never every 5 like the old grow/cut.
        assert sizes == [1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 5, 6]


@pytest.mark.asyncio
async def test_conversation_window_follows_config_value(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """CHAT_HISTORY_TURNS is a runtime config param: a changed value changes
    the window on the next build.  Six exchanges an hour old, then two fresh
    ones: the fresh pair starts a session, whose window is 2 — the last two
    pre-session exchanges plus the session's own two, not the default 5
    (#3077, the cut at a session's first turn)."""
    config = make_config(chat_history_turns=2)

    async with running_penny(config) as penny:
        stale = penny.db.clock.now() - timedelta(hours=1)
        for n in range(1, 7):
            user_id = penny.db.messages.log_message(
                PennyConstants.MessageDirection.INCOMING,
                TEST_SENDER,
                f"user message {n}",
                timestamp=stale,
            )
            penny.db.messages.log_message(
                PennyConstants.MessageDirection.OUTGOING,
                penny.config.signal_number,
                f"reply {n}",
                parent_id=user_id,
                recipient=TEST_SENDER,
                timestamp=stale,
            )
        for n in (7, 8):
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
        assert [turn.content for turn in conversation] == [
            "user message 5",
            "reply 5",
            "user message 6",
            "reply 6",
            "user message 7",
            "reply 7",
            "user message 8",
            "reply 8",
        ]


@pytest.mark.asyncio
async def test_thread_history_grows_then_cuts_back(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """A quote-reply's thread history follows the same session-window rule
    (#3077, its own-count fallback: a thread chain has no session of its
    own — ruled so on the issue).  The chain (8 exchanges) stays under both
    the window's growth and its own 20-message fetch cap, so the window
    keeps it all."""
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
        # 8 exchanges is within the window's growth (5 upward, cap 20) — everything so far.
        assert [turn.content for turn in thread_history] == [
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
