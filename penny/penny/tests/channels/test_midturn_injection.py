"""Integration tests for mid-turn message injection (#2985).

Every test drives the channel's public entry — ``handle_message`` with a parsed
Signal envelope, exactly what the receive loop calls — with the LLM mocked and
the model call the test needs to stay in flight gated on an ``asyncio.Event``
(a wrapper around the mock client's ``chat``, never a fixed sleep).
"""

from __future__ import annotations

import asyncio
import time

import pytest
from sqlmodel import col, select

from penny.channels.signal import SignalChannel
from penny.channels.turns import TurnRegistry
from penny.constants import ProgressEmoji
from penny.database import Database
from penny.database.models import MessageLog, PromptLog
from penny.llm import LlmClient
from penny.tests.conftest import NULL_SANDBOX_RUNNER, TEST_SENDER, wait_until


def _world(test_config, db):
    """A ``ChatAgent`` + ``SignalChannel`` pair plus the channel's own registry,
    built the way ``test_signal_channel`` builds its world."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    client = LlmClient(
        api_url=test_config.llm_api_url,
        model=test_config.llm_model,
        db=db,
        max_retries=test_config.llm_max_retries,
        retry_delay=test_config.llm_retry_delay,
    )
    message_agent = ChatAgent(
        system_prompt=Prompt.CONVERSATION_PROMPT,
        model_client=client,
        embedding_model_client=client,
        tools=[],
        db=db,
        config=test_config,
        collector=NULL_SANDBOX_RUNNER,
    )
    registry = TurnRegistry()
    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        turn_registry=registry,
    )
    return channel, registry


def _envelope(config, sender: str, content: str) -> dict:
    """A plain Signal text-message envelope — the parsed shape the receive loop
    hands ``handle_message``."""
    ts = int(time.time() * 1000)
    return {
        "account": config.signal_number,
        "envelope": {
            "source": sender,
            "sourceNumber": sender,
            "sourceUuid": "test-uuid",
            "sourceName": "Test User",
            "sourceDevice": 1,
            "timestamp": ts,
            "serverReceivedTimestamp": ts,
            "serverDeliveredTimestamp": ts,
            "dataMessage": {
                "timestamp": ts,
                "message": content,
                "expiresInSeconds": 0,
                "isExpirationUpdate": False,
                "viewOnce": False,
            },
        },
    }


def _gate_chat(mock_llm, monkeypatch, gate_on: int, *, fail_last_user: str | None = None):
    """Wrap the mock client's ``chat`` so call ``gate_on`` blocks until released,
    and every call whose LAST user message is ``fail_last_user`` raises — the
    errored-turn tests' kill switch.  Matching the message, not a call ordinal,
    is what makes it deterministic: the real client RETRIES a failed call, and
    every retry must fail too for the run to die; a retry carries the same
    conversation, so the same last user turn.  Returns ``(entered, release,
    attempted)``: the test awaits ``entered`` to know the gated call is in
    flight, pushes its mid-turn messages, then sets ``release``; ``attempted``
    records every call's messages, including the ones that failed and so never
    landed in ``mock_llm.requests``."""
    entered = asyncio.Event()
    release = asyncio.Event()
    attempted: list[list[dict]] = []
    original = mock_llm.chat
    calls = {"n": 0}

    def _field(message, field):
        # The lower patch hands wire dicts; a direct caller hands LlmMessages.
        return message.get(field) if isinstance(message, dict) else getattr(message, field, None)

    async def gated_chat(messages=None, tools=None, format=None, *, call_link, **kwargs):
        calls["n"] += 1
        attempted.append(
            [
                message.to_log_dict() if hasattr(message, "to_log_dict") else message
                for message in (messages or [])
            ]
        )
        if calls["n"] == gate_on:
            entered.set()
            await release.wait()
        if fail_last_user is not None:
            last_user = next(
                (
                    _field(message, "content")
                    for message in reversed(messages or [])
                    if _field(message, "role") == "user"
                ),
                None,
            )
            if last_user == fail_last_user:
                raise RuntimeError("model exploded")
        return await original(messages, tools, format, call_link=call_link, **kwargs)

    monkeypatch.setattr(mock_llm, "chat", gated_chat)
    return entered, release, attempted


def _tool_then_reply(mock_llm):
    """Call 1 returns a tool call for a tool that does not exist — the executor
    rejects it with an honest error result at once (no browser-retry ladder, no
    draw to wait out), which is still a full tool step; the next call replies."""

    def handler(request, count):
        if count == 1:
            return mock_llm._make_tool_call_response(request, "no_such_tool", {})
        return mock_llm._make_text_response(request, "here's what i found!")

    mock_llm.set_response_handler(handler, answers_micro_contexts=False)


def _user_messages_after_last_tool(messages: list[dict]) -> list[dict]:
    """The user turns that follow the last tool result in a call's messages —
    where a mid-run injection lands."""
    last_tool = max(i for i, m in enumerate(messages) if m["role"] == "tool")
    return [m for m in messages[last_tool + 1 :] if m["role"] == "user"]


@pytest.mark.asyncio
async def test_midturn_message_joins_after_the_tool_step(
    signal_server, test_config, mock_llm, test_user_info, monkeypatch
):
    """B arriving while A's turn is in its tool loop joins that turn: one run,
    B rendered as a user message right after the tool result in the next model
    call, and the ⏳ waiting mark set on arrival and removed at the drain."""
    db = Database(test_config.db_path)
    channel, _registry = _world(test_config, db)
    _tool_then_reply(mock_llm)
    entered, release, attempted = _gate_chat(mock_llm, monkeypatch, gate_on=1)

    turn_a = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "A")))
    await entered.wait()

    turn_b = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "B")))
    await wait_until(
        lambda: any(
            event["op"] == "send" and event["reaction"] == ProgressEmoji.QUEUED
            for event in signal_server.reaction_events
        ),
        timeout=5.0,
    )
    await turn_b  # a join returns at once

    release.set()
    await turn_a

    # One turn, one run id across both calls (read off the promptlog — the mock's
    # low-level patch records a fresh call link per call, not the run's).
    assert len(mock_llm.requests) == 2
    with db.get_session() as session:
        logged_runs = {
            row.run_id
            for row in session.exec(
                select(PromptLog).where(col(PromptLog.run_id).isnot(None))
            ).all()
        }
    assert len(logged_runs) == 1
    # B sits right after the tool result in the next call's messages.
    injected = _user_messages_after_last_tool(mock_llm.requests[1]["messages"])
    assert [m["content"] for m in injected] == ["B"]
    # One reply answers the whole extended turn.
    assert len(signal_server.outgoing_messages) == 1
    # The waiting mark came off at the drain.
    assert any(
        event["op"] == "remove" and event["reaction"] == ProgressEmoji.QUEUED
        for event in signal_server.reaction_events
    )

    await channel.close()


@pytest.mark.asyncio
async def test_several_queued_messages_inject_in_order(
    signal_server, test_config, mock_llm, test_user_info, monkeypatch
):
    """Messages queued while a turn runs inject in arrival order."""
    db = Database(test_config.db_path)
    channel, _registry = _world(test_config, db)
    _tool_then_reply(mock_llm)
    entered, release, attempted = _gate_chat(mock_llm, monkeypatch, gate_on=1)

    turn_a = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "A")))
    await entered.wait()
    turn_b = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "B")))
    turn_c = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "C")))
    await wait_until(
        lambda: (
            sum(
                1
                for event in signal_server.reaction_events
                if event["op"] == "send" and event["reaction"] == ProgressEmoji.QUEUED
            )
            == 2
        ),
        timeout=5.0,
    )
    await turn_b
    await turn_c

    release.set()
    await turn_a

    injected = _user_messages_after_last_tool(mock_llm.requests[1]["messages"])
    assert [m["content"] for m in injected] == ["B", "C"]
    assert len(mock_llm.requests) == 2

    await channel.close()


@pytest.mark.asyncio
async def test_message_queued_during_the_final_call_starts_a_followup_turn(
    signal_server, test_config, mock_llm, test_user_info, monkeypatch
):
    """B arriving during the turn's final model call was never drained — the
    finished turn's series starts a second turn for it at once, with A's turn
    in its history, and B gets its own reply."""
    db = Database(test_config.db_path)
    channel, _registry = _world(test_config, db)
    _tool_then_reply(mock_llm)
    entered, release, attempted = _gate_chat(mock_llm, monkeypatch, gate_on=2)

    turn_a = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "A")))
    await entered.wait()  # the final call is in flight

    turn_b = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "B")))
    await wait_until(
        lambda: any(
            event["op"] == "send" and event["reaction"] == ProgressEmoji.QUEUED
            for event in signal_server.reaction_events
        ),
        timeout=5.0,
    )
    await turn_b

    release.set()
    await wait_until(lambda: len(signal_server.outgoing_messages) == 2, timeout=10.0)
    await turn_a

    replies = [message["message"] for message in signal_server.outgoing_messages]
    assert "here's what i found!" in replies[0]
    assert "here's what i found!" in replies[1]
    # Three model calls: A's tool step, A's reply, then B's own turn.
    assert len(mock_llm.requests) == 3
    # B's turn saw A's finished exchange in its history.
    history = mock_llm.requests[2]["messages"]
    assert any(m["role"] == "user" and m["content"] == "A" for m in history)
    assert any(m["role"] == "assistant" and "here's what i found!" in m["content"] for m in history)
    assert any(m["role"] == "user" and m["content"] == "B" for m in history)
    # B was never drained into A's run, so its row carries no run stamp.
    with db.get_session() as session:
        rows = list(session.exec(select(MessageLog).where(MessageLog.content == "B")).all())
    assert len(rows) == 1 and rows[0].run_id is None

    await channel.close()


@pytest.mark.asyncio
async def test_errored_turn_still_answers_every_queued_message(
    signal_server, test_config, mock_llm, test_user_info, monkeypatch
):
    """A turn whose model call dies mid-run reports the honest failure for its
    own message, and every message still queued (never drained) is answered by
    its own follow-up turn — nothing is dropped, nothing left unanswered."""
    from penny.responses import PennyResponse

    db = Database(test_config.db_path)
    channel, _registry = _world(test_config, db)
    _tool_then_reply(mock_llm)
    # A's own call dies on every attempt (its last user turn is A's message);
    # B's follow-up turn's calls (last user turn B) go through.
    entered, release, attempted = _gate_chat(mock_llm, monkeypatch, gate_on=1, fail_last_user="A")

    turn_a = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "A")))
    await entered.wait()
    turn_b = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "B")))
    await wait_until(
        lambda: any(
            event["op"] == "send" and event["reaction"] == ProgressEmoji.QUEUED
            for event in signal_server.reaction_events
        ),
        timeout=5.0,
    )
    await turn_b

    release.set()
    await wait_until(lambda: len(signal_server.outgoing_messages) == 2, timeout=10.0)
    await turn_a

    # The failure notice answers A; B got a real reply from its own turn.
    first, second = signal_server.outgoing_messages[0], signal_server.outgoing_messages[1]
    assert first["message"] == PennyResponse.FACTS_TURN_FAILED_BEFORE_REPLY
    assert "here's what i found!" in second["message"]
    # Two chat calls: A's failed one, then B's turn.
    assert len(mock_llm.requests) == 2

    await channel.close()


@pytest.mark.asyncio
async def test_errored_turn_covers_its_injected_messages(
    signal_server, test_config, mock_llm, test_user_info, monkeypatch
):
    """A message already INJECTED into a turn that then dies belongs to that
    turn: its row is stamped with the run it joined, the turn's own failure
    notice answers it, and no follow-up turn re-asks it."""
    from penny.responses import PennyResponse

    db = Database(test_config.db_path)
    channel, _registry = _world(test_config, db)
    _tool_then_reply(mock_llm)
    # Gate call 1 so B is queued before the drain; the reply call then dies on
    # every attempt — B is its last user turn by then (the drain injected it).
    entered, release, attempted = _gate_chat(mock_llm, monkeypatch, gate_on=1, fail_last_user="B")

    turn_a = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "A")))
    await entered.wait()
    turn_b = asyncio.create_task(channel.handle_message(_envelope(test_config, TEST_SENDER, "B")))
    await wait_until(
        lambda: any(
            event["op"] == "send" and event["reaction"] == ProgressEmoji.QUEUED
            for event in signal_server.reaction_events
        ),
        timeout=5.0,
    )
    await turn_b

    release.set()
    await turn_a

    # The drain injected B into call 2 before it died (read off the attempted
    # messages — a failed call never lands in ``mock_llm.requests``).
    injected = _user_messages_after_last_tool(attempted[1])
    assert [m["content"] for m in injected] == ["B"]
    # One notice covers the turn; no second turn re-asks B.
    assert len(signal_server.outgoing_messages) == 1
    assert (
        signal_server.outgoing_messages[0]["message"]
        == PennyResponse.FACTS_TURN_FAILED_BEFORE_REPLY
    )
    assert len(mock_llm.requests) == 1
    # B's row is stamped with the run it joined.
    with db.get_session() as session:
        rows = list(session.exec(select(MessageLog).where(MessageLog.content == "B")).all())
        joined_runs = {
            row.run_id
            for row in session.exec(
                select(PromptLog).where(col(PromptLog.run_id).isnot(None))
            ).all()
        }
    assert len(rows) == 1
    assert rows[0].run_id is not None
    assert rows[0].run_id in joined_runs

    await channel.close()
