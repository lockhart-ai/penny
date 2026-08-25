"""Integration tests for Signal reaction handling."""

import asyncio
import contextlib
import json
import time
from unittest.mock import patch

import pytest
from sqlmodel import col, select

from penny.agents import ChatAgent
from penny.channels.manager import ChannelManager
from penny.channels.permission_manager import PermissionManager
from penny.channels.signal import SignalChannel
from penny.constants import ChannelType, PennyConstants, PermissionResolution
from penny.database import Database
from penny.database.models import MessageLog
from penny.llm import LlmClient
from penny.prompts import Prompt
from penny.tests.conftest import TEST_SENDER, wait_until


async def _wait_for_outgoing_ids(penny, contains: str) -> tuple[int, str]:
    """Return ``(id, external_id)`` of the outgoing message whose body contains
    ``contains``, once it is stamped.

    Two races, both real. ``MockSignalServer.wait_for_message`` unblocks the
    moment the send reaches the mock, but the channel writes ``external_id``
    onto the DB row only *after* that send call returns (``_log_and_send`` →
    ``set_external_id``), so reading the row the instant the message lands sees
    ``external_id is None`` — poll until the stamp arrives. And this config runs
    a background collector, so "the first outgoing row" is not necessarily the
    chat reply; match on the body instead of taking whatever landed first.
    """

    def _row(session):
        return session.exec(
            select(MessageLog)
            .where(MessageLog.direction == PennyConstants.MessageDirection.OUTGOING)
            .where(col(MessageLog.content).contains(contains))
        ).first()

    def stamped() -> bool:
        with penny.db.get_session() as session:
            outgoing = _row(session)
            return outgoing is not None and outgoing.external_id is not None

    await wait_until(stamped)
    with penny.db.get_session() as session:
        outgoing = _row(session)
        assert outgoing is not None
        assert outgoing.external_id is not None
        return outgoing.id, outgoing.external_id


@pytest.mark.asyncio
async def test_signal_reaction_message(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
    setup_llm_flow,
):
    """
    Test Signal reaction handling:
    1. Send a message and get a response
    2. React to the response with an emoji
    3. Verify reaction is logged as a regular incoming message
    """
    config = make_config(idle_seconds=0.5)
    setup_llm_flow(
        message_response="here's a cool fact! 🌟",
        background_response="glad you liked that, here's more! 🎉",
    )

    async with running_penny(config) as penny:
        # Send initial message.  Wait for THE REPLY, not merely the next outgoing
        # message: this config runs a background collector (idle_seconds=0.5) whose
        # own send can land first, so "the next message" was never a safe stand-in
        # for "the chat reply" — an ordering assumption that only held while the
        # foreground path was fast enough to win the race every time.
        await signal_server.push_message(sender=TEST_SENDER, content="tell me something cool")
        response = await signal_server.wait_for_message_containing("cool fact", timeout=10.0)
        assert "cool fact" in response["message"].lower()

        # Get the outgoing message's signal timestamp (waiting for the stamp)
        message_id, external_id = await _wait_for_outgoing_ids(penny, "cool fact")

        # Send a reaction to Penny's response
        await signal_server.push_reaction(
            sender=TEST_SENDER,
            emoji="👍",
            target_timestamp=int(external_id),
        )

        # Wait for reaction to be logged in the DB
        def reaction_logged():
            with penny.db.get_session() as session:
                reactions = list(
                    session.exec(
                        select(MessageLog).where(
                            MessageLog.content == "👍",
                            MessageLog.sender == TEST_SENDER,
                            MessageLog.parent_id == message_id,
                        )
                    ).all()
                )
                return len(reactions) == 1

        await wait_until(reaction_logged)

        # Verify reaction details — logged as regular incoming message
        with penny.db.get_session() as session:
            reactions = list(
                session.exec(
                    select(MessageLog).where(
                        MessageLog.content == "👍",
                        MessageLog.sender == TEST_SENDER,
                        MessageLog.parent_id == message_id,
                    )
                ).all()
            )
        assert len(reactions) == 1, "Reaction should be logged"
        reaction = reactions[0]
        assert reaction.content == "👍"
        assert reaction.parent_id == message_id
        assert reaction.is_reaction is True

        # Verify no response was sent to the reaction
        # (only the initial response should exist)
        assert len(signal_server.outgoing_messages) == 1

        # And nothing Penny sent was withdrawn — the mock serves
        # /v1/remote-delete, so an empty log means no request was ever issued.
        assert signal_server.delete_events == []


@pytest.mark.asyncio
async def test_signal_reaction_raw_format(
    signal_server, mock_llm, make_config, test_user_info, running_penny
):
    """
    Test Signal reaction handling with the raw format that Signal actually sends.

    This tests the bug fix for issue #34 where Signal sends:
    - message: None (not an empty string)
    - emoji: "👍" (plain string, not {"value": "👍"} object)
    """
    config = make_config()
    mock_llm.set_default_flow(
        final_response="test response 🌟",
    )

    async with running_penny(config) as penny:
        # Send initial message
        await signal_server.push_message(sender=TEST_SENDER, content="test message")
        await signal_server.wait_for_message_containing("test response", timeout=10.0)

        # Get the outgoing message's signal timestamp (waiting for the stamp)
        message_id, external_id = await _wait_for_outgoing_ids(penny, "test response")

        # Send a reaction using the raw format that Signal actually sends
        # (not the mock format with {"value": emoji})
        ts = int(time.time() * 1000)
        raw_envelope = {
            "envelope": {
                "source": TEST_SENDER,
                "sourceNumber": TEST_SENDER,
                "sourceUuid": "test-uuid-123",
                "sourceName": "Test User",
                "sourceDevice": 1,
                "timestamp": ts,
                "serverReceivedTimestamp": ts,
                "serverDeliveredTimestamp": ts,
                "dataMessage": {
                    "timestamp": ts,
                    "message": None,  # KEY: None, not empty string
                    "reaction": {
                        "emoji": "👍",  # KEY: Plain string, not {"value": "👍"}
                        "targetAuthor": config.signal_number,
                        "targetAuthorNumber": config.signal_number,
                        "targetSentTimestamp": int(external_id),
                        "isRemove": False,
                    },
                },
            },
            "account": config.signal_number,
        }

        # Push the raw envelope to all connected websockets
        for ws in signal_server._websockets:
            if not ws.closed:
                await ws.send_str(json.dumps(raw_envelope))

        # Wait for reaction to be logged in the DB
        def reaction_logged():
            with penny.db.get_session() as session:
                reactions = list(
                    session.exec(
                        select(MessageLog).where(
                            MessageLog.content == "👍",
                            MessageLog.sender == TEST_SENDER,
                            MessageLog.parent_id == message_id,
                        )
                    ).all()
                )
                return len(reactions) == 1

        await wait_until(reaction_logged)

        # Verify reaction details — logged as regular incoming message
        with penny.db.get_session() as session:
            reactions = list(
                session.exec(
                    select(MessageLog).where(
                        MessageLog.content == "👍",
                        MessageLog.sender == TEST_SENDER,
                        MessageLog.parent_id == message_id,
                    )
                ).all()
            )
        assert len(reactions) == 1, "Reaction should be logged"
        reaction = reactions[0]
        assert reaction.content == "👍"
        assert reaction.parent_id == message_id
        assert reaction.is_reaction is True


# --- Permission prompts: acknowledged with a reaction, never remote-deleted ---

PROMPT_DOMAIN = "permission-probe.test"
PROMPT_URL = f"https://{PROMPT_DOMAIN}/article"


def _signal_permission_world(config, db: Database) -> tuple[SignalChannel, PermissionManager]:
    """A real Signal channel behind a real permission manager, on the mock server.

    The production shape end to end: the manager broadcasts through the channel
    manager, so the resolution the Signal channel acknowledges is the one the
    prompt actually ended with — not one the test handed it.
    """
    client = LlmClient(
        api_url=config.llm_api_url,
        model=config.llm_model,
        db=db,
        max_retries=config.llm_max_retries,
        retry_delay=config.llm_retry_delay,
    )
    agent = ChatAgent(
        system_prompt=Prompt.CONVERSATION_PROMPT,
        model_client=client,
        embedding_model_client=client,
        tools=[],
        db=db,
        config=config,
    )
    assert config.signal_number, "the Signal number is what a reaction targets — it must be set"
    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number,
        message_agent=agent,
        db=db,
    )
    manager = ChannelManager(message_agent=agent, db=db)
    manager.register_channel(ChannelType.SIGNAL, channel)
    permissions = PermissionManager(db=db, channel_manager=manager, config=config)
    channel.set_permission_manager(permissions)
    return channel, permissions


async def _answer_elsewhere(
    channel: SignalChannel, permissions: PermissionManager, allowed: bool
) -> None:
    """Answer the pending prompt the way another device would (the addon).

    Signal is then a channel that never heard the answer and learns the prompt
    is over only from the dismiss broadcast — the production shape that was
    producing tombstones.
    """
    await wait_until(lambda: bool(channel._pending_permission_messages), timeout=5.0)
    request_id = next(iter(channel._pending_permission_messages))
    permissions.handle_decision(request_id, allowed)


def _prompt_external_id(db: Database) -> int:
    """The Signal timestamp of the prompt message, off its own messagelog row."""
    with db.get_session() as session:
        row = session.exec(
            select(MessageLog)
            .where(MessageLog.direction == PennyConstants.MessageDirection.OUTGOING)
            .where(col(MessageLog.content).contains(PROMPT_DOMAIN))
        ).first()
        assert row is not None, "the permission prompt was never sent"
        assert row.external_id is not None, "the prompt message was never stamped"
        return int(row.external_id)


@pytest.mark.parametrize(
    ("allowed", "resolution", "denial"),
    [
        (True, PermissionResolution.ALLOWED, None),
        (False, PermissionResolution.BLOCKED, "denied"),
    ],
)
@pytest.mark.asyncio
async def test_answered_permission_prompt_is_marked_with_a_reaction(
    signal_server,
    mock_llm,
    test_config,
    test_user_info,
    allowed,
    resolution,
    denial,
):
    """An answered prompt gets its resolution as a reaction on Penny's own message.

    And no remote delete: the mock serves ``/v1/remote-delete``, so an empty
    ``delete_events`` means nothing asked for one — not that a request 404'd.
    """
    db = Database(test_config.db_path)
    channel, permissions = _signal_permission_world(test_config, db)

    asyncio.create_task(_answer_elsewhere(channel, permissions, allowed))
    expected = pytest.raises(RuntimeError, match=denial) if denial else contextlib.nullcontext()
    with expected:
        await permissions.check_domain(PROMPT_URL)

    assert signal_server.reaction_events == [
        {
            "op": "send",
            "recipient": TEST_SENDER,
            "reaction": resolution.emoji,
            "target_author": test_config.signal_number,
            "timestamp": _prompt_external_id(db),
        }
    ], "the resolution belongs on Penny's own prompt message"
    assert signal_server.delete_events == [], "a resolved prompt is never remote-deleted"
    assert any(
        PROMPT_DOMAIN in sent.get("message", "") for sent in signal_server.outgoing_messages
    ), "the prompt itself stays in the conversation"

    await channel.close()


@pytest.mark.asyncio
async def test_timed_out_permission_prompt_is_marked_distinctly(
    signal_server, mock_llm, test_config, test_user_info
):
    """A prompt nobody answered gets its own mark — not an answer's, and not a delete."""
    db = Database(test_config.db_path)
    channel, permissions = _signal_permission_world(test_config, db)

    with (
        patch.object(PennyConstants, "PERMISSION_PROMPT_TIMEOUT", 0.1),
        pytest.raises(RuntimeError, match="timed out"),
    ):
        await permissions.check_domain(PROMPT_URL)

    assert signal_server.reaction_events == [
        {
            "op": "send",
            "recipient": TEST_SENDER,
            "reaction": PermissionResolution.TIMED_OUT.emoji,
            "target_author": test_config.signal_number,
            "timestamp": _prompt_external_id(db),
        }
    ]
    assert PermissionResolution.TIMED_OUT.emoji not in {
        PermissionResolution.ALLOWED.emoji,
        PermissionResolution.BLOCKED.emoji,
    }, "an expired prompt must not read as an answer"
    assert signal_server.delete_events == [], "an expired prompt is never remote-deleted"

    await channel.close()
