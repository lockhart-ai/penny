"""Integration tests for startup announcement feature."""

import asyncio
import contextlib
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from penny.channels.base import IncomingMessage, MessageChannel
from penny.constants import PennyConstants
from penny.llm.models import LlmConnectionError
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, wait_until


def _find_request(mock_llm, needle: str) -> str:
    """The content of the first captured LLM request containing needle."""
    for request in mock_llm.requests:
        for message in request["messages"]:
            if needle in message["content"]:
                return message["content"]
    raise AssertionError(f"No LLM request containing {needle!r}")


class StartupReadyChannel(MessageChannel):
    """Fake channel whose sends are blocked on listener readiness."""

    def __init__(self, message_agent, db):
        super().__init__(message_agent=message_agent, db=db)
        self.listen_started = asyncio.Event()
        self.ready = asyncio.Event()
        self.closed = asyncio.Event()
        self.sent: list[tuple[str, str]] = []

    @property
    def sender_id(self) -> str:
        return "startup-ready"

    async def listen(self) -> None:
        self.listen_started.set()
        await self.closed.wait()

    async def wait_until_ready(self) -> None:
        await self.ready.wait()

    async def _send_raw(
        self,
        recipient: str,
        message: str,
        attachments: list[str] | None = None,
        quote_message=None,
    ) -> int | None:
        self.sent.append((recipient, message))
        return len(self.sent)

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        return True

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        return None

    async def close(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_startup_announcement_with_commit(
    signal_server, test_config, mock_llm, running_penny, monkeypatch
):
    """Test that Penny sends startup announcement with restart message from git commit."""
    # First run: populate database with a sender
    mock_llm.set_default_flow(final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        # Create user profile before messaging (required for chat dispatch)
        penny.db.users.save_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

        # Verify sender is in database
        senders = penny.db.users.get_all_senders()
        assert TEST_SENDER in senders

    # Clear messages from first run
    signal_server.outgoing_messages.clear()

    # Set commit message in environment variable
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "feat: add cool new feature")

    # Second run: configure restart message and verify announcement
    mock_llm.set_default_flow(final_response="i added a cool new feature! check it out")

    before = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
        PennyConstants.CURRENT_DATETIME_FORMAT
    )
    async with running_penny(test_config):
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Should have received startup announcement
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )

        # Find the announcement message (should be the first or only message)
        announcement = signal_server.outgoing_messages[0]

        # Should be sent to TEST_SENDER
        assert TEST_SENDER in announcement["recipients"], (
            f"Expected {TEST_SENDER} in recipients, got {announcement['recipients']}"
        )

        # Should start with wave emoji and include restart message
        message = announcement["message"]
        assert message.startswith("👋"), f"Expected message to start with 👋, got: {message}"
        assert "i added a cool new feature! check it out" in message, (
            f"Expected restart message in announcement, got: {message}"
        )

        # The commit-transform prompt is grounded in the current date, rendered in
        # the user's profile timezone (LA) — the ad-hoc startup flow gets the same
        # dated anchor as the agent-loop envelope, never a bare UTC now().
        after = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            PennyConstants.CURRENT_DATETIME_FORMAT
        )
        transform_prompt = _find_request(mock_llm, "Transform this git commit message")
        assert "Current date and time: " in transform_prompt
        assert any(
            f"Current date and time: {stamp}" in transform_prompt for stamp in (before, after)
        )


@pytest.mark.asyncio
async def test_startup_announcement_fallback_no_git(
    signal_server, test_config, mock_llm, running_penny, monkeypatch
):
    """Test that Penny falls back to 'I just restarted!' when git commit message unavailable."""
    # First run: populate database
    mock_llm.set_default_flow(final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        # Create user profile before messaging (required for chat dispatch)
        penny.db.users.save_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

    # Clear messages
    signal_server.outgoing_messages.clear()

    # Set commit message to "unknown" to simulate missing git info
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "unknown")

    # Second run: verify fallback message
    async with running_penny(test_config):
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Should use fallback message
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )
        announcement = signal_server.outgoing_messages[0]
        message = announcement["message"]
        assert message == "👋 I just restarted!", f"Expected fallback message, got: {message}"


@pytest.mark.asyncio
async def test_startup_announcement_fallback_llm_error(
    signal_server, test_config, mock_llm, running_penny, monkeypatch
):
    """Test that Penny falls back when LLM transformation fails."""
    # First run: populate database
    mock_llm.set_default_flow(final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        # Create user profile before messaging (required for chat dispatch)
        penny.db.users.save_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )

        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

    # Clear messages
    signal_server.outgoing_messages.clear()

    # Set commit message in environment variable
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "feat: some feature")

    # Second run: configure LLM to fail for restart message generation
    def error_handler(request, count):
        raise LlmConnectionError("Ollama is down")

    mock_llm.set_response_handler(error_handler)

    async with running_penny(test_config):
        await wait_until(lambda: len(signal_server.outgoing_messages) > 0)

        # Should use fallback message when LLM fails
        assert len(signal_server.outgoing_messages) >= 1, (
            f"Expected at least 1 message, got {len(signal_server.outgoing_messages)}"
        )
        announcement = signal_server.outgoing_messages[0]
        message = announcement["message"]
        assert message == "👋 I just restarted!", f"Expected fallback message, got: {message}"


@pytest.mark.asyncio
async def test_startup_announcement_no_recipients(
    signal_server, test_config, mock_llm, running_penny
):
    """Test that Penny doesn't crash when there are no recipients."""
    # Start Penny without any prior message history.
    async with running_penny(test_config):
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_startup_announcement_waits_for_channel_ready(make_config, mock_llm, monkeypatch):
    """Startup sends must not block listener startup before the channel is ready."""
    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "unknown")
    config = make_config()
    penny = Penny(config, channel=None)
    channel = StartupReadyChannel(message_agent=penny.chat_agent, db=penny.db)
    penny._init_channel(config, channel)
    penny._connect_scheduler(config)
    penny.db.users.save_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )
    penny.db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING,
        sender=TEST_SENDER,
        content="hello",
    )

    task = asyncio.create_task(penny.run())
    try:
        await wait_until(channel.listen_started.is_set)
        assert channel.sent == []

        channel.ready.set()
        await wait_until(lambda: len(channel.sent) == 1)
        assert channel.sent[0] == (TEST_SENDER, "👋 I just restarted!")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await penny.shutdown()


@pytest.mark.asyncio
async def test_startup_announcement_skipped_with_profile_but_no_messages(
    signal_server, test_config, mock_llm, running_penny, test_user_info
):
    """Profile exists but no message history — announcement should be skipped."""
    # test_user_info creates a profile for TEST_SENDER but sends no messages.
    # The announcement should not fire because there's no message history.
    async with running_penny(test_config):
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_startup_announcement_multiple_devices(
    signal_server, test_config, mock_llm, running_penny, monkeypatch
):
    """Test that Penny sends startup announcement to the primary user.

    Penny is single-user: messages from any device resolve to the same user.
    Startup announcements go to all known senders in the message log.
    """
    mock_llm.set_default_flow(final_response="test response 🌟")

    async with running_penny(test_config) as penny:
        penny.db.users.save_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )
        from penny.constants import ChannelType

        penny.db.devices.register(ChannelType.SIGNAL, TEST_SENDER, "Signal", is_default=True)

        # Send from primary device
        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

        # All messages resolve to the primary sender
        senders = penny.db.users.get_all_senders()
        assert TEST_SENDER in senders

    # Clear messages from first run
    signal_server.outgoing_messages.clear()

    monkeypatch.setenv("GIT_COMMIT_MESSAGE", "feat: update something cool")
    mock_llm.set_default_flow(final_response="i updated something cool")

    async with running_penny(test_config):

        def recipient_notified():
            for msg in signal_server.outgoing_messages:
                if TEST_SENDER in msg.get("recipients", []):
                    return True
            return False

        await wait_until(recipient_notified)

        assert len(signal_server.outgoing_messages) >= 1

        for msg in signal_server.outgoing_messages:
            message = msg["message"]
            assert message.startswith("👋")
            assert "i updated something cool" in message


@pytest.mark.asyncio
async def test_startup_warns_when_embedding_model_not_available(
    signal_server, make_config, mock_llm, running_penny, caplog, monkeypatch
):
    """Startup validation logs a warning when LLM_EMBEDDING_MODEL is not pulled."""
    # Configure an embedding model that is NOT in the available models list
    config = make_config(
        llm_embedding_model="qwen3-embedding:4b", llm_image_model="test-image-model"
    )

    # Patch list_models to return only the base chat model (embedding model absent)
    async def mock_list_models(self):
        return ["test-model", "test-image-model"]

    monkeypatch.setattr("penny.llm.image_client.OllamaImageClient.list_models", mock_list_models)

    with caplog.at_level(logging.WARNING, logger="penny.penny"):
        async with running_penny(config):
            pass

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("qwen3-embedding:4b" in m for m in warning_messages), (
        f"Expected warning about missing embedding model, got: {warning_messages}"
    )
    assert any("LLM_EMBEDDING_MODEL" in m for m in warning_messages), (
        f"Expected env var name in warning, got: {warning_messages}"
    )


@pytest.mark.asyncio
async def test_startup_no_warning_when_embedding_model_available(
    signal_server, make_config, mock_llm, running_penny, caplog, monkeypatch
):
    """Startup validation does not warn when LLM_EMBEDDING_MODEL is present."""
    config = make_config(llm_embedding_model="nomic-embed-text")

    async def mock_list_models(self):
        return ["test-model", "nomic-embed-text"]

    monkeypatch.setattr("penny.llm.image_client.OllamaImageClient.list_models", mock_list_models)

    with caplog.at_level(logging.WARNING, logger="penny.penny"):
        async with running_penny(config):
            pass

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("nomic-embed-text" in m for m in warning_messages), (
        f"Expected no warning for available model, got: {warning_messages}"
    )


@pytest.mark.asyncio
async def test_startup_no_warning_when_no_optional_models_configured(
    signal_server, test_config, mock_llm, running_penny, caplog, monkeypatch
):
    """Startup validation does not warn when no optional models are configured."""

    # test_config has no embedding/vision/image models set
    async def mock_list_models(self):
        return ["test-model"]

    monkeypatch.setattr("penny.llm.image_client.OllamaImageClient.list_models", mock_list_models)

    with caplog.at_level(logging.WARNING, logger="penny.penny"):
        async with running_penny(test_config):
            pass

    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("not available on the Ollama host" in m for m in warning_messages), (
        f"Expected no model-availability warnings, got: {warning_messages}"
    )
