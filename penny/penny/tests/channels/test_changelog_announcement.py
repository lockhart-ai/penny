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
from sqlmodel import col, select

from penny.agents.models import ControllerResponse
from penny.changelog import ChangelogEntry, record_new_entries, render_changelog_announcement
from penny.channels.base import IncomingMessage, MessageChannel
from penny.constants import ChannelType, PennyConstants
from penny.database.models import PromptLog
from penny.penny import Penny
from penny.preflight import CheckStatus, PreflightCheck, PreflightError
from penny.tests.conftest import TEST_SENDER, require_memory, wait_until


def _find_request(mock_llm, needle: str) -> str:
    """The content of the first captured LLM request containing needle."""
    for request in mock_llm.requests:
        for message in request["messages"]:
            if needle in message["content"]:
                return message["content"]
    raise AssertionError(f"No LLM request containing {needle!r}")


def _prompt_log_for(session, needle: str) -> PromptLog:
    """The logged ``PromptLog`` row whose messages contain ``needle`` — the REAL
    run record, stamped by ``LlmClient.chat()`` itself. Read this for the actual
    run id a call carried: ``mock_llm.requests`` is built from the SDK-level
    intercept (``_MockOpenAIClient.create``), several layers below
    ``LlmClient.chat()``'s own Python-level ``call_link``/``agent_name``/
    ``prompt_type`` arguments — those never reach the wire, so the mock's own
    request log carries none of them, and asserting against it there would
    compare the wrong thing."""
    rows = session.exec(select(PromptLog).where(col(PromptLog.messages).contains(needle))).all()
    if not rows:
        raise AssertionError(f"No logged prompt containing {needle!r}")
    return rows[0]


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


# Two contrived entries — standing in for the generated commit-title list in every test
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
    reply it writes is delivered straight through the channel's normal send
    path (#2913): no queue row, no idle wait."""
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
    # The real run id the delivery turn was dispatched with — read off the
    # SAME ``CallLink`` production built, not re-derived here.
    delivery_run_id = await_args.kwargs["call_link"].run_id

    channel.send_response.assert_awaited_once()
    _, call_kwargs = channel.send_response.call_args
    assert call_kwargs["content"] == _REPLY
    assert call_kwargs["recipient"] == TEST_SENDER
    assert call_kwargs["mechanism"] == PennyConstants.MEMORY_CHANGELOG_LOG
    # #2849's run-health facts read ``messagelog.run_id`` off the delivered row —
    # this is what keeps that read from landing on ``None`` (the channel here is
    # a bare mock, so the real messagelog write is asserted in the channel-ready
    # test below, which drives a real channel through ``send_response``).
    assert call_kwargs["run_id"] == delivery_run_id
    # Nothing was ever queued — delivery happened directly.
    assert penny.db.send_queue.pending_items() == []


@pytest.mark.asyncio
async def test_changelog_announcement_still_sends_with_no_new_entries(test_config):
    """A deploy that recorded nothing new (harness-only, CI-only, or a restart
    on an already-recorded commit) still runs the delivery turn — Penny
    announces on every startup (#2890) — and what it writes is delivered
    directly (#2913), with nothing left in the queue."""
    channel = MagicMock()
    channel.send_response = AsyncMock(return_value=1)
    penny = Penny(test_config, channel=channel)
    _seed_profile_and_history(penny)
    penny.chat_agent.handle_changelog_report = AsyncMock(
        return_value=ControllerResponse(answer=_REPLY)
    )

    await penny._send_changelog_announcement([])

    penny.chat_agent.handle_changelog_report.assert_awaited_once()
    await_args = penny.chat_agent.handle_changelog_report.await_args
    assert await_args is not None
    assert await_args.args[0] == []

    channel.send_response.assert_awaited_once()
    assert channel.send_response.call_args.kwargs["content"] == _REPLY
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
async def test_changelog_announcement_waits_for_channel_ready(make_config, mock_llm):
    """Startup sends must not block listener startup before the channel is ready,
    and the delivery turn's prompt is the framework render, byte for byte.

    The commit list rides ``Config.changelog_commits`` — a PARAMETER (#2891),
    never a monkeypatched module function or a file read. The ``changelog``
    log is pre-seeded with ONE already-recorded PR before this boot, so the
    commit list carries both an old, already-announced PR and a genuinely
    new one: ``record_new_entries``'s own idempotency (never a separate
    selection step — #2891) skips the old one and records/announces only
    the new one."""
    old_entry = ChangelogEntry(pr=9300, text="an already-recorded PR from an earlier deploy")
    new_entry = ChangelogEntry(pr=9301, text="We added a new way to save a note.")
    # A plain always-answer handler, not ``set_default_flow``'s two-step
    # browse-then-answer counter: this is the delivery turn's own FIRST call on
    # a fresh mock, and the model should just write the announcement directly.
    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_text_response(request, _REPLY)
    )
    # No idle/cooldown tuning needed (#2913): the announcement is delivered
    # directly through the channel the moment the delivery turn composes it,
    # never through the send-queue drainer's own idle-gated schedule.
    config = make_config(changelog_commits=[old_entry, new_entry])
    penny = Penny(config, channel=None)
    channel = StartupReadyChannel(message_agent=penny.chat_agent, db=penny.db)
    penny._init_channel(config, channel)
    penny._connect_scheduler(config)
    _seed_profile_and_history(penny)
    record_new_entries(require_memory(penny.db, PennyConstants.MEMORY_CHANGELOG_LOG), [old_entry])

    task = asyncio.create_task(penny.run())
    try:
        await wait_until(channel.listen_started.is_set)
        assert channel.sent == []

        channel.ready.set()
        await wait_until(lambda: len(channel.sent) == 1)
        assert channel.sent[0] == (TEST_SENDER, _REPLY)

        prompt = _find_request(mock_llm, "reached you with this deploy")
        assert prompt == render_changelog_announcement([new_entry])

        changelog = require_memory(penny.db, PennyConstants.MEMORY_CHANGELOG_LOG)
        assert [row.content for row in changelog.read_all()] == [
            old_entry.render(),
            new_entry.render(),
        ]

        # #2849's run-health facts read ``messagelog.run_id`` off the delivered
        # row — a real channel's own ``send_response`` wrote it here, so this is
        # the end-to-end proof it is the delivery turn's run id, not None. Read
        # against the REAL promptlog row (``LlmClient.chat()``'s own stamp), not
        # ``mock_llm.requests`` — that list is built from the SDK-level intercept,
        # below ``call_link``'s own Python-level plumbing, and carries none of it.
        with penny.db.get_session() as session:
            delivery_log = _prompt_log_for(session, "reached you with this deploy")
            delivery_run_id = delivery_log.run_id
        delivered_row = penny.db.messages.find_outgoing_by_content(_REPLY)
        assert delivered_row is not None
        assert delivered_row.run_id == delivery_run_id
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await penny.shutdown()


@pytest.mark.asyncio
async def test_changelog_announcement_startup_sends_with_no_new_entries(make_config, mock_llm):
    """Penny announces on every startup (#2890): a deploy that brings no new
    changelog entries still runs the real delivery turn end to end — recorded,
    composed, and delivered straight through the channel (#2913) — with the
    "nothing changed" render as its input. ``changelog_commits`` defaults to
    empty (#2891), so no override is needed here.

    Delivered directly, so this exercises only ``_run_startup_notifications``
    and the channel — no scheduler, no drainer tick, no idle tuning (the
    flake in #2910 was the drain schedule's first idle-tick on an empty queue
    stamping its own eligibility before the row existed; a direct send never
    goes near that schedule)."""
    mock_llm.set_response_handler(
        lambda request, _count: mock_llm._make_text_response(request, _REPLY)
    )
    config = make_config()
    penny = Penny(config, channel=None)
    channel = StartupReadyChannel(message_agent=penny.chat_agent, db=penny.db)
    penny._init_channel(config, channel)
    _seed_profile_and_history(penny)
    channel.ready.set()

    await penny._run_startup_notifications([])

    assert channel.sent == [(TEST_SENDER, _REPLY)]
    prompt = _find_request(mock_llm, "just restarted")
    assert prompt == render_changelog_announcement([])

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
    signal_server, make_config, mock_llm, running_penny
):
    """Penny is single-user: a LATER deploy's new entry reaches the registered
    default device, on top of whatever an earlier deploy already recorded.

    Idempotency (#2845) means replaying the SAME entries on a restart announces
    nothing a second time — so this models two real deploys: the first ships
    ``_ENTRIES`` (recorded, but too early in that boot for any profile to exist
    yet to announce to), the second ships one further entry beyond those. Each
    boot gets its OWN ``Config`` (``changelog_commits`` is a parameter, #2891)
    — both ``make_config`` calls share the same underlying db/signal-server
    defaults, so the second boot resumes the first boot's own database."""
    config = make_config(
        idle_seconds=0.1, send_cooldown_seconds=0.0, changelog_commits=list(_ENTRIES)
    )
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
    second_boot_config = make_config(
        idle_seconds=0.1,
        send_cooldown_seconds=0.0,
        changelog_commits=[*_ENTRIES, new_entry],
    )
    mock_llm.set_default_flow(final_response=_REPLY)

    async with running_penny(second_boot_config):

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
