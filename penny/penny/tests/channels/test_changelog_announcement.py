"""Integration tests for the changelog announcement feature (#2846).

Replaces the retired startup commit-message announcement (``startup.py``, #2370):
a deploy's new changelog entries now ride ONE real chat-agent delivery turn
(``ChatAgent.handle_changelog_report``), gated and queued like any other
autonomous send, instead of a one-line model transform of the latest commit.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from penny.agents.models import ControllerResponse
from penny.changelog import ChangelogEntry, render_changelog_announcement
from penny.channels.base import IncomingMessage, MessageChannel
from penny.constants import ChannelType, PennyConstants
from penny.penny import Penny
from penny.preflight import CheckStatus, PreflightCheck, PreflightError
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
        source_name: str | None = None,
        message_log_id: int | None = None,
    ) -> int | None:
        self.sent.append((recipient, message))
        return len(self.sent)

    async def send_typing(self, recipient: str, typing: bool) -> bool:
        return True

    def extract_message(self, raw_data: dict) -> IncomingMessage | None:
        return None

    async def close(self) -> None:
        self.closed.set()


# Two contrived entries — standing in for ``changelog.yaml`` rows in every test
# below; what they say never matters, only that they are the exact ones rendered.
_ENTRIES = [
    ChangelogEntry(pr=9301, text="We added a new way to save a note."),
    ChangelogEntry(pr=9302, text="We renamed the lookup tool."),
]
_REPLY = "got it — I'll let you know if either of those comes up! 🌟"


def _seed_profile_and_history(penny: Penny) -> None:
    """A profile plus one prior exchange — both gates this feature keeps
    (profile exists, the user has chatted before) satisfied at once."""
    penny.db.users.save_info(
        sender=TEST_SENDER,
        name="Test User",
        location="Seattle, WA",
        timezone="America/Los_Angeles",
        date_of_birth="1990-01-01",
    )
    penny.db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, "hey penny"
    )


@pytest.mark.asyncio
async def test_changelog_announcement_dispatches_the_whole_literal_render(test_config):
    """The delivery turn's input is the framework render, verbatim — and the
    reply it writes is queued exactly like any other autonomous send (gated,
    then drained through the ordinary channel path)."""
    # No ``shutdown()`` here — it awaits ``self.channel.close()``, and a bare
    # ``MagicMock`` isn't awaitable; this test drives ``_send_changelog_announcement``
    # directly and never starts ``run()``, so there is nothing to tear down
    # (the same reason the pre-existing iOS-gate test below never shut down either).
    channel = MagicMock()
    channel.send_response = AsyncMock(return_value=1)
    penny = Penny(test_config, channel=channel)
    _seed_profile_and_history(penny)
    penny.chat_agent.handle_changelog_report = AsyncMock(
        return_value=ControllerResponse(answer=_REPLY)
    )

    await penny._send_changelog_announcement(_ENTRIES)

    penny.chat_agent.handle_changelog_report.assert_awaited_once()
    await_args = penny.chat_agent.handle_changelog_report.await_args
    assert await_args is not None
    assert await_args.args[0] == _ENTRIES
    assert await_args.args[1] == TEST_SENDER

    pending = penny.db.send_queue.pending_items()
    assert len(pending) == 1
    assert pending[0].content == _REPLY
    assert pending[0].collection == PennyConstants.CHANGELOG_LOG_AUTHOR

    drained = await penny.send_queue_drainer.execute()
    assert drained is True
    channel.send_response.assert_awaited_once()
    _, call_kwargs = channel.send_response.call_args
    assert call_kwargs["content"] == _REPLY
    assert call_kwargs["recipient"] == TEST_SENDER


@pytest.mark.asyncio
async def test_changelog_announcement_sends_nothing_with_no_new_entries(test_config):
    """A deploy that recorded nothing new (harness-only, CI-only, or a restart
    on an already-recorded commit) never spends a model call."""
    penny = Penny(test_config, channel=MagicMock())
    _seed_profile_and_history(penny)
    penny.chat_agent.handle_changelog_report = AsyncMock()

    await penny._send_changelog_announcement([])

    penny.chat_agent.handle_changelog_report.assert_not_awaited()
    assert penny.db.send_queue.pending_items() == []


@pytest.mark.asyncio
async def test_changelog_announcement_no_recipients(
    signal_server, test_config, mock_llm, running_penny
):
    """No profile at all — the turn never runs."""
    async with running_penny(test_config):
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_changelog_announcement_skipped_with_profile_but_no_messages(
    signal_server, test_config, mock_llm, running_penny, test_user_info
):
    """Profile exists but no message history — the user has never chatted, so the
    turn is skipped rather than spamming a fresh install."""
    async with running_penny(test_config):
        assert len(signal_server.outgoing_messages) == 0


@pytest.mark.asyncio
async def test_changelog_announcement_waits_for_channel_ready(make_config, mock_llm, monkeypatch):
    """Startup sends must not block listener startup before the channel is ready,
    and the delivery turn's prompt is the framework render, byte for byte."""
    monkeypatch.setattr("penny.penny.load_entries", lambda: list(_ENTRIES))
    # A plain always-answer handler, not ``set_default_flow``'s two-step
    # browse-then-answer counter: this is the delivery turn's own FIRST call on
    # a fresh mock, and the model should just write the announcement directly.
    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_text_response(request, _REPLY)
    )
    # A tiny idle threshold (the default is 99999s, chosen to keep background
    # schedules off): the announcement is now QUEUED and delivered by the real
    # send-queue drainer, same as any other autonomous send, so the drainer
    # schedule must actually get to run within the test's wait window.
    config = make_config(idle_seconds=0.1, send_cooldown_seconds=0.0)
    penny = Penny(config, channel=None)
    channel = StartupReadyChannel(message_agent=penny.chat_agent, db=penny.db)
    penny._init_channel(config, channel)
    penny._connect_scheduler(config)
    _seed_profile_and_history(penny)

    task = asyncio.create_task(penny.run())
    try:
        await wait_until(channel.listen_started.is_set)
        assert channel.sent == []

        channel.ready.set()
        await wait_until(lambda: len(channel.sent) == 1)
        assert channel.sent[0] == (TEST_SENDER, _REPLY)

        prompt = _find_request(mock_llm, "reached you with this deploy")
        assert prompt == render_changelog_announcement(_ENTRIES)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await penny.shutdown()


@pytest.mark.asyncio
async def test_ios_operational_changelog_announcement_respects_history_gate(
    test_config, test_user_info
):
    """iOS keeps its own simple operational push, unchanged by this feature
    (prod's default device is Signal): the real changelog turn never runs for
    it, and the push itself still honours the same message-history gate."""
    channel = MagicMock()
    channel.send_message = AsyncMock()
    penny = Penny(test_config, channel=channel)
    device = penny.db.devices.register(ChannelType.IOS, "ios-device-id", "iPad", is_default=True)
    assert device.id is not None
    penny.db.devices.set_default(device.id)
    penny.chat_agent.handle_changelog_report = AsyncMock()

    # No message history yet — the push is suppressed, not spammed per deploy.
    await penny._send_changelog_announcement(_ENTRIES)
    channel.send_message.assert_not_awaited()

    # Once the user has chatted, the operational push fires — and the real
    # delivery turn never runs for an iOS-default deploy.
    penny.db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING, sender=TEST_SENDER, content="hey penny"
    )
    await penny._send_changelog_announcement(_ENTRIES)
    channel.send_message.assert_awaited_once_with("ios-device-id", "Penny is operational.")
    penny.chat_agent.handle_changelog_report.assert_not_awaited()


@pytest.mark.asyncio
async def test_changelog_announcement_multiple_devices(
    signal_server, make_config, mock_llm, running_penny, monkeypatch
):
    """Penny is single-user: a LATER deploy's new entry reaches the registered
    default device, on top of whatever an earlier deploy already recorded.

    Idempotency (#2845) means replaying the SAME entries on a restart announces
    nothing a second time — so this models two real deploys: the first ships
    ``_ENTRIES`` (recorded, but too early in that boot for any profile to exist
    yet to announce to), the second ships one further entry beyond those.

    A tiny idle threshold, same reason as the channel-ready test above: the
    announcement is QUEUED, and the real send-queue drainer must actually run
    within the test's wait window to deliver it."""
    config = make_config(idle_seconds=0.1, send_cooldown_seconds=0.0)
    monkeypatch.setattr("penny.penny.load_entries", lambda: list(_ENTRIES))
    async with running_penny(config) as penny:
        penny.db.users.save_info(
            sender=TEST_SENDER,
            name="Test User",
            location="Seattle, WA",
            timezone="America/Los_Angeles",
            date_of_birth="1990-01-01",
        )
        penny.db.devices.register(ChannelType.SIGNAL, TEST_SENDER, "Signal", is_default=True)
        await signal_server.push_message(sender=TEST_SENDER, content="hey penny")
        await signal_server.wait_for_message(timeout=10.0)

    signal_server.outgoing_messages.clear()
    new_entry = ChangelogEntry(pr=9303, text="We added a weekly digest option.")
    monkeypatch.setattr("penny.penny.load_entries", lambda: [*_ENTRIES, new_entry])
    mock_llm.set_default_flow(final_response=_REPLY)

    async with running_penny(config):

        def recipient_notified():
            return any(
                TEST_SENDER in msg.get("recipients", []) for msg in signal_server.outgoing_messages
            )

        await wait_until(recipient_notified)
        for msg in signal_server.outgoing_messages:
            assert msg["message"] == _REPLY


@pytest.mark.asyncio
async def test_startup_preflight_hard_fails_when_embedding_model_missing(
    make_config, mock_llm, monkeypatch
):
    """A missing embedding model is a hard preflight failure that aborts startup."""
    config = make_config(llm_embedding_model="qwen3-embedding:4b")

    # Endpoint reachable and the chat model present, but the embedding model absent.
    async def only_chat_model(self):
        return ["test-model"]

    monkeypatch.setattr("penny.llm.client.LlmClient.list_models", only_chat_model)

    penny = Penny(config)
    try:
        with pytest.raises(PreflightError) as excinfo:
            await penny._run_preflight()
        assert "qwen3-embedding:4b" in str(excinfo.value)
        assert "LLM_EMBEDDING_MODEL" in str(excinfo.value)
        # The report rides the error, so a caller deciding what to do reads the failed checks.
        assert [f.name for f in excinfo.value.report.failures] == [PreflightCheck.EMBEDDING_MODEL]
    finally:
        await penny.shutdown()


@pytest.mark.asyncio
async def test_startup_preflight_passes_when_models_available(make_config, mock_llm, monkeypatch):
    """Preflight reports no failures and routes to the configured channel when models resolve."""
    config = make_config(llm_embedding_model="nomic-embed-text")

    async def both_models(self):
        return ["test-model", "nomic-embed-text"]

    monkeypatch.setattr("penny.llm.client.LlmClient.list_models", both_models)

    penny = Penny(config)
    try:
        report = await penny._build_preflight().run()
        assert not report.has_failures
        primary = next(r for r in report.results if r.name is PreflightCheck.PRIMARY_CHANNEL)
        assert primary.status is CheckStatus.OK
    finally:
        await penny.shutdown()


@pytest.mark.asyncio
async def test_startup_preflight_skips_unconfigured_optional_models(test_config, mock_llm):
    """An unconfigured image-generation model is skipped by preflight — not warned, not failed."""
    penny = Penny(test_config)
    try:
        report = await penny._build_preflight().run()
        assert not report.has_failures
        names = {r.name for r in report.results}
        assert PreflightCheck.IMAGE_MODEL not in names
    finally:
        await penny.shutdown()
