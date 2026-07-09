"""Integration tests for Signal reaction handling."""

import json
import time

import pytest
from sqlmodel import select

from penny.database.models import MessageLog
from penny.tests.conftest import TEST_SENDER, wait_until


async def _wait_for_outgoing_ids(penny) -> tuple[int, str]:
    """Return ``(id, external_id)`` of the outgoing message once it is stamped.

    ``MockSignalServer.wait_for_message`` unblocks the moment the send reaches
    the mock, but the channel writes ``external_id`` onto the DB row only
    *after* that send call returns (``_log_and_send`` → ``set_external_id``).
    Reading the row the instant the message lands therefore races that write and
    intermittently sees ``external_id is None`` — the source of this test's
    flakiness. Poll until the stamp lands before reading.
    """

    def stamped() -> bool:
        with penny.db.get_session() as session:
            outgoing = session.exec(
                select(MessageLog).where(MessageLog.direction == "outgoing")
            ).first()
            return outgoing is not None and outgoing.external_id is not None

    await wait_until(stamped)
    with penny.db.get_session() as session:
        outgoing = session.exec(
            select(MessageLog).where(MessageLog.direction == "outgoing")
        ).first()
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
        # Send initial message
        await signal_server.push_message(sender=TEST_SENDER, content="tell me something cool")
        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cool fact" in response["message"].lower()

        # Get the outgoing message's signal timestamp (waiting for the stamp)
        message_id, external_id = await _wait_for_outgoing_ids(penny)

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
        await signal_server.wait_for_message(timeout=10.0)

        # Get the outgoing message's signal timestamp (waiting for the stamp)
        message_id, external_id = await _wait_for_outgoing_ids(penny)

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
