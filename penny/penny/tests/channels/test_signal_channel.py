"""Integration tests for Signal channel."""

import pytest

from penny.channels.signal import SignalChannel
from penny.constants import ChannelType
from penny.database import Database
from penny.llm import LlmClient
from penny.tests.conftest import TEST_SENDER


@pytest.mark.asyncio
async def test_validate_connectivity_success(signal_server, test_config, mock_llm):
    """Test that validate_connectivity succeeds with a reachable Signal API."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

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
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    # Should not raise
    await channel.validate_connectivity()

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_dns_failure(test_db, mock_llm):
    """Test that validate_connectivity raises ConnectionError on DNS failure."""
    from penny.agents import ChatAgent
    from penny.config import Config
    from penny.prompts import Prompt

    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://nonexistent-hostname-that-will-never-resolve.invalid:8080",
        discord_bot_token=None,
        discord_channel_id=None,
        llm_api_url="http://localhost:11434",
        llm_model="test-model",
        llm_embedding_model="test-embedding-model",
        log_level="DEBUG",
        db_path=test_db,
    )

    db = Database(config.db_path)
    db.create_tables()

    client = LlmClient(
        api_url=config.llm_api_url,
        model=config.llm_model,
        db=db,
        max_retries=config.llm_max_retries,
        retry_delay=config.llm_retry_delay,
    )
    message_agent = ChatAgent(
        system_prompt=Prompt.CONVERSATION_PROMPT,
        model_client=client,
        embedding_model_client=client,
        tools=[],
        db=db,
        config=config,
    )

    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await channel.validate_connectivity(max_attempts=1, retry_delay=0)

    error_message = str(exc_info.value)
    assert "Cannot resolve Signal API hostname" in error_message
    assert "nonexistent-hostname-that-will-never-resolve.invalid" in error_message
    assert "SIGNAL_API_URL" in error_message

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_connection_refused(test_db, mock_llm):
    """Test that validate_connectivity raises ConnectionError when server is unreachable."""
    from penny.agents import ChatAgent
    from penny.config import Config
    from penny.prompts import Prompt

    # Use localhost on a port that's not listening
    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://localhost:19999",  # Unlikely to be in use
        discord_bot_token=None,
        discord_channel_id=None,
        llm_api_url="http://localhost:11434",
        llm_model="test-model",
        llm_embedding_model="test-embedding-model",
        log_level="DEBUG",
        db_path=test_db,
    )

    db = Database(config.db_path)
    db.create_tables()

    client = LlmClient(
        api_url=config.llm_api_url,
        model=config.llm_model,
        db=db,
        max_retries=config.llm_max_retries,
        retry_delay=config.llm_retry_delay,
    )
    message_agent = ChatAgent(
        system_prompt=Prompt.CONVERSATION_PROMPT,
        model_client=client,
        embedding_model_client=client,
        tools=[],
        db=db,
        config=config,
    )

    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await channel.validate_connectivity(max_attempts=1, retry_delay=0)

    error_message = str(exc_info.value)
    assert "Cannot connect to Signal API" in error_message
    assert "http://localhost:19999" in error_message

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_retries_then_succeeds(
    signal_server, test_config, mock_llm, monkeypatch, caplog
):
    """validate_connectivity retries transient probe failures and recovers.

    Regression: cold-boot startup races signal-cli-rest-api (~30–60s warmup);
    a single failed attempt would crash the process and docker would respawn
    it in a tight loop. The retry path should log each failed attempt and
    recover when the API comes up.
    """
    import logging

    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()
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
    )
    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    real_probe = channel._probe_signal_api
    call_count = {"n": 0}

    async def flaky_probe() -> None:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError(f"simulated probe failure {call_count['n']}")
        await real_probe()

    monkeypatch.setattr(channel, "_probe_signal_api", flaky_probe)

    with caplog.at_level(logging.WARNING, logger="penny.channels.signal.channel"):
        await channel.validate_connectivity(max_attempts=5, retry_delay=0)

    assert call_count["n"] == 3
    failure_warnings = [
        record
        for record in caplog.records
        if "Signal API validation attempt" in record.message and record.levelno == logging.WARNING
    ]
    assert len(failure_warnings) == 2

    await channel.close()


@pytest.mark.asyncio
async def test_validate_connectivity_exhausts_attempts_then_logs_and_raises(
    test_db, mock_llm, caplog
):
    """After exhausting all attempts, the final error is logged and raised."""
    import logging

    from penny.agents import ChatAgent
    from penny.config import Config
    from penny.prompts import Prompt

    config = Config(
        channel_type="signal",
        signal_number="+15551234567",
        signal_api_url="http://localhost:19999",
        discord_bot_token=None,
        discord_channel_id=None,
        llm_api_url="http://localhost:11434",
        llm_model="test-model",
        llm_embedding_model="test-embedding-model",
        log_level="DEBUG",
        db_path=test_db,
    )
    db = Database(config.db_path)
    db.create_tables()
    client = LlmClient(
        api_url=config.llm_api_url,
        model=config.llm_model,
        db=db,
        max_retries=config.llm_max_retries,
        retry_delay=config.llm_retry_delay,
    )
    message_agent = ChatAgent(
        system_prompt=Prompt.CONVERSATION_PROMPT,
        model_client=client,
        embedding_model_client=client,
        tools=[],
        db=db,
        config=config,
    )
    channel = SignalChannel(
        api_url=config.signal_api_url,
        phone_number=config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with (
        caplog.at_level(logging.WARNING, logger="penny.channels.signal.channel"),
        pytest.raises(ConnectionError),
    ):
        await channel.validate_connectivity(max_attempts=3, retry_delay=0)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(warnings) == 3
    assert len(errors) == 1
    assert "after 3 attempts" in errors[0].message

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_rejects_empty_without_attachments(signal_server, test_config, mock_llm):
    """Test that send_message raises ValueError for empty text with no attachments."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

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
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    with pytest.raises(ValueError, match="Cannot send empty"):
        await channel.send_message(TEST_SENDER, "", attachments=None, quote_message=None)

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_allows_empty_text_with_attachments(
    signal_server, test_config, mock_llm
):
    """Test that send_message succeeds with empty text when attachments are provided."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

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
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )

    # Should not raise — empty text is fine when attachments are present
    fake_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    result = await channel.send_message(
        TEST_SENDER, "", attachments=[fake_image], quote_message=None
    )
    assert result is not None

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_retries_on_socket_exception_400(signal_server, test_config, mock_llm):
    """Test that send_message retries when signal-cli returns a 400 SocketException."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

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
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        max_retries=2,
        retry_delay=0.01,
    )

    # Queue one transient SocketException 400 then let the retry succeed
    socket_error_body = {
        "error": (
            "Failed to send message: Failed to get response for request"
            " (SocketException) (UnexpectedErrorException)"
        )
    }
    signal_server.queue_send_error(400, socket_error_body)

    result = await channel.send_message(TEST_SENDER, "hello", attachments=None, quote_message=None)

    # Should have succeeded on the retry
    assert result is not None
    # The message was eventually delivered (the successful send is captured)
    assert len(signal_server.outgoing_messages) == 1

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_no_retry_on_non_transient_400(signal_server, test_config, mock_llm):
    """Test that send_message does NOT retry on non-transient 400 errors."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

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
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        max_retries=2,
        retry_delay=0.01,
    )

    # Queue a non-transient 400 (bad recipient format — should not retry)
    signal_server.queue_send_error(400, {"error": "Invalid recipient number"})
    # Queue another 200 success that should NOT be reached if retry is skipped
    signal_server.queue_send_error(400, {"error": "Invalid recipient number"})

    result = await channel.send_message(TEST_SENDER, "hello", attachments=None, quote_message=None)

    # Should have returned None without retrying
    assert result is None
    # No messages should have been captured by the success handler
    assert len(signal_server.outgoing_messages) == 0
    # The second queued error should still be in the queue (retry was not attempted)
    assert len(signal_server._send_response_queue) == 1

    await channel.close()


@pytest.mark.asyncio
async def test_send_message_gives_up_after_max_retries(signal_server, test_config, mock_llm):
    """Test that send_message returns None after exhausting retries on persistent errors."""
    from penny.agents import ChatAgent
    from penny.prompts import Prompt

    db = Database(test_config.db_path)
    db.create_tables()

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
    )

    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
        max_retries=2,
        retry_delay=0.01,
    )

    # Queue 3 transient errors (initial attempt + 2 retries = 3 total)
    socket_error = {"error": "Failed to send message: (SocketException)"}
    for _ in range(3):
        signal_server.queue_send_error(400, socket_error)

    result = await channel.send_message(TEST_SENDER, "hello", attachments=None, quote_message=None)

    # All retries exhausted — should return None
    assert result is None
    # No successful sends
    assert len(signal_server.outgoing_messages) == 0
    # All 3 queued errors were consumed
    assert len(signal_server._send_response_queue) == 0

    await channel.close()


def test_extract_message_sets_channel_type_and_device_identifier():
    """Extracted messages carry channel_type=signal and device_identifier=sender."""
    from unittest.mock import MagicMock

    channel = SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+15551234567",
        message_agent=MagicMock(),
        db=MagicMock(),
    )

    raw = {
        "account": "+15551234567",
        "envelope": {
            "source": TEST_SENDER,
            "sourceNumber": TEST_SENDER,
            "sourceUuid": "test-uuid",
            "sourceName": "Test",
            "sourceDevice": 1,
            "timestamp": 1234567890,
            "serverReceivedTimestamp": 1234567891,
            "serverDeliveredTimestamp": 1234567892,
            "dataMessage": {
                "timestamp": 1234567890,
                "message": "hello",
                "expiresInSeconds": 0,
                "viewOnce": False,
            },
        },
    }

    msg = channel.extract_message(raw)
    assert msg is not None
    assert msg.channel_type == ChannelType.SIGNAL
    assert msg.device_identifier == TEST_SENDER
    assert msg.sender == TEST_SENDER


def test_extract_reaction_sets_channel_type():
    """Extracted reactions carry channel_type=signal."""
    from unittest.mock import MagicMock

    channel = SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+15551234567",
        message_agent=MagicMock(),
        db=MagicMock(),
    )

    raw = {
        "account": "+15551234567",
        "envelope": {
            "source": TEST_SENDER,
            "sourceNumber": TEST_SENDER,
            "sourceUuid": "test-uuid",
            "sourceName": "Test",
            "sourceDevice": 1,
            "timestamp": 1234567890,
            "serverReceivedTimestamp": 1234567891,
            "serverDeliveredTimestamp": 1234567892,
            "dataMessage": {
                "timestamp": 1234567890,
                "message": None,
                "expiresInSeconds": 0,
                "viewOnce": False,
                "reaction": {
                    "emoji": "\U0001f44d",
                    "targetAuthor": "+15551234567",
                    "targetAuthorNumber": "+15551234567",
                    "targetSentTimestamp": 1234567889,
                    "isRemove": False,
                },
            },
        },
    }

    msg = channel.extract_message(raw)
    assert msg is not None
    assert msg.channel_type == ChannelType.SIGNAL
    assert msg.device_identifier == TEST_SENDER
    assert msg.is_reaction is True


def test_reaction_callback_fires_and_consumes_reaction():
    """A registered reaction callback fires on matching timestamp and returns None."""
    from unittest.mock import MagicMock

    channel = SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+15551234567",
        message_agent=MagicMock(),
        db=MagicMock(),
    )

    captured = []
    channel.register_reaction_callback("1234567889", lambda emoji: captured.append(emoji))

    raw = {
        "account": "+15551234567",
        "envelope": {
            "source": TEST_SENDER,
            "sourceNumber": TEST_SENDER,
            "sourceUuid": "test-uuid",
            "sourceName": "Test",
            "sourceDevice": 1,
            "timestamp": 1234567890,
            "serverReceivedTimestamp": 1234567891,
            "serverDeliveredTimestamp": 1234567892,
            "dataMessage": {
                "timestamp": 1234567890,
                "message": None,
                "expiresInSeconds": 0,
                "viewOnce": False,
                "reaction": {
                    "emoji": "\U0001f44d",
                    "targetAuthor": "+15551234567",
                    "targetAuthorNumber": "+15551234567",
                    "targetSentTimestamp": 1234567889,
                    "isRemove": False,
                },
            },
        },
    }

    msg = channel.extract_message(raw)
    assert msg is None, "Reaction should be consumed by callback"
    assert captured == ["\U0001f44d"]
    assert "1234567889" not in channel._reaction_callbacks, "Callback should be one-shot"


def test_reaction_without_callback_returns_normal_message():
    """Reactions without a registered callback return normal IncomingMessage."""
    from unittest.mock import MagicMock

    channel = SignalChannel(
        api_url="http://localhost:8080",
        phone_number="+15551234567",
        message_agent=MagicMock(),
        db=MagicMock(),
    )

    # Register callback for a DIFFERENT timestamp
    channel.register_reaction_callback("9999999999", lambda emoji: None)

    raw = {
        "account": "+15551234567",
        "envelope": {
            "source": TEST_SENDER,
            "sourceNumber": TEST_SENDER,
            "sourceUuid": "test-uuid",
            "sourceName": "Test",
            "sourceDevice": 1,
            "timestamp": 1234567890,
            "serverReceivedTimestamp": 1234567891,
            "serverDeliveredTimestamp": 1234567892,
            "dataMessage": {
                "timestamp": 1234567890,
                "message": None,
                "expiresInSeconds": 0,
                "viewOnce": False,
                "reaction": {
                    "emoji": "\U0001f44e",
                    "targetAuthor": "+15551234567",
                    "targetAuthorNumber": "+15551234567",
                    "targetSentTimestamp": 1234567889,
                    "isRemove": False,
                },
            },
        },
    }

    msg = channel.extract_message(raw)
    assert msg is not None
    assert msg.is_reaction is True
    assert msg.content == "\U0001f44e"


@pytest.mark.asyncio
async def test_send_response_attaches_cited_page_media(signal_server, test_config, mock_llm):
    """The image captured from a page the reply *cites* is attached at egress
    (tier-1 exact-URL match — the deterministic cited-source path); and a
    generated ``media_ids`` that no longer resolves degrades visibly (logged,
    skipped) instead of crashing egress."""
    import base64
    from typing import Any, cast

    from penny.agents import ChatAgent
    from penny.database.migrate import migrate
    from penny.llm.embeddings import serialize_embedding
    from penny.prompts import Prompt
    from penny.tests.mocks.llm_patches import MockLlmClient, deterministic_embed

    db = Database(test_config.db_path)
    db.create_tables()
    migrate(test_config.db_path)

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
    )
    channel = SignalChannel(
        api_url=test_config.signal_api_url,
        phone_number=test_config.signal_number or "+15551234567",
        message_agent=message_agent,
        db=db,
    )
    channel._embedding_model_client = cast(Any, MockLlmClient())

    raw = b"\xff\xd8 jpeg bytes"
    db.media.put(
        raw,
        "image/jpeg",
        source_url="https://ex.com/page",
        title="Ex",
        embedding=serialize_embedding(deterministic_embed("ex")),
    )

    # The reply cites the source page, so its captured image attaches (tier 1).
    await channel.send_response(
        TEST_SENDER, "here's what I found: https://ex.com/page", parent_id=None, author="penny"
    )

    sent = signal_server.outgoing_messages[-1]
    expected = f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"
    assert sent.get("base64_attachments") == [expected]

    # A generated media id that no longer resolves degrades visibly, not fatally:
    # egress logs and skips it, delivering the reply text with no attachment
    # (rather than crashing or silently falling back to a fuzzy match).
    await channel.send_response(
        TEST_SENDER, "here you go", parent_id=None, author="penny", media_ids=[999999]
    )
    assert signal_server.outgoing_messages[-1].get("base64_attachments") is None

    await channel.close()
