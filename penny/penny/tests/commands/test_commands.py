"""Integration tests for /commands command."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlmodel import col, select

from penny.constants import PennyConstants
from penny.database.models import PromptLog, UserInfo
from penny.tests.conftest import TEST_SENDER, wait_until


def _has_message(server, text: str) -> bool:
    return any(text in msg.get("message", "") for msg in server.outgoing_messages)


def _find_request(mock_llm, needle: str) -> str:
    """The content of the first captured LLM request containing needle."""
    for request in mock_llm.requests:
        for message in request["messages"]:
            if needle in message["content"]:
                return message["content"]
    raise AssertionError(f"No LLM request containing {needle!r}")


def _prompt_log_for(session, needle: str) -> PromptLog:
    """The logged ``PromptLog`` row whose messages contain ``needle`` (#2440 —
    a model call with no run id never gets pushed live and is filtered out of
    every run-indexed fetch)."""
    rows = session.exec(select(PromptLog).where(col(PromptLog.messages).contains(needle))).all()
    if not rows:
        raise AssertionError(f"No logged prompt containing {needle!r}")
    return rows[0]


@pytest.mark.asyncio
async def test_profile_update_parse_grounded_in_date(
    signal_server, test_config, mock_llm, running_penny
):
    """The /profile parse prompt carries the current date, rendered in the user's
    profile timezone (LA) — an ad-hoc one-shot flow gets the same dated anchor the
    agent-loop envelope injects, never a bare UTC now()."""

    def handler(request, count):
        return mock_llm._make_text_response(request, '{"name": "Sammy", "location": null}')

    mock_llm.set_response_handler(handler)

    async with running_penny(test_config) as penny:
        with penny.db.get_session() as session:
            session.add(
                UserInfo(
                    sender=TEST_SENDER,
                    name="Sam",
                    location="Seattle",
                    timezone="America/Los_Angeles",
                    date_of_birth="1990-01-01",
                )
            )
            session.commit()

        before = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            PennyConstants.CURRENT_DATETIME_FORMAT
        )
        await signal_server.push_message(sender=TEST_SENDER, content="/profile Sammy")
        await wait_until(lambda: _has_message(signal_server, "I updated your"))
        after = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
            PennyConstants.CURRENT_DATETIME_FORMAT
        )

        parse_prompt = _find_request(mock_llm, "Extract the user's name and/or location")
        assert "Current date and time: " in parse_prompt
        assert any(f"Current date and time: {stamp}" in parse_prompt for stamp in (before, after))

        # The parse call now carries a run id (#2440), so it is pushed live and
        # shows up in a run-indexed fetch instead of being silently filtered out.
        with penny.db.get_session() as session:
            logged = _prompt_log_for(session, "Extract the user's name and/or location")
            assert logged.run_id is not None
            assert logged.discard_reason is None

        # A draw whose JSON fails its parse is rejected at the parse boundary —
        # its own row carries the reason (#2702); the earlier clean parse above
        # reads normally, unmarked.
        mock_llm.set_response_handler(
            lambda request, count: mock_llm._make_text_response(request, "not json at all")
        )
        await signal_server.push_message(sender=TEST_SENDER, content="/profile Denver")
        await wait_until(lambda: _has_message(signal_server, "I couldn't understand that"))
        with penny.db.get_session() as session:
            parse_rows = session.exec(
                select(PromptLog).where(
                    col(PromptLog.messages).contains("Extract the user's name and/or location")
                )
            ).all()
        assert len(parse_rows) == 2
        assert [row.discard_reason for row in parse_rows if row.discard_reason] == ["parse_error"]


@pytest.mark.asyncio
async def test_commands_list(signal_server, test_config, mock_llm, running_penny):
    """Test /commands lists all available commands."""
    async with running_penny(test_config) as _penny:
        # Send /commands
        await signal_server.push_message(sender=TEST_SENDER, content="/commands")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should list registered commands with their descriptions
        assert "**Available Commands**" in response["message"]
        assert "**/commands**" in response["message"]
        assert "**/profile**" in response["message"]
        assert "List all commands" in response["message"]
        assert "View or update your profile" in response["message"]


@pytest.mark.asyncio
async def test_commands_help_specific(signal_server, test_config, mock_llm, running_penny):
    """Test /commands <name> shows help for specific command."""
    async with running_penny(test_config) as _penny:
        # Send /commands profile
        await signal_server.push_message(sender=TEST_SENDER, content="/commands profile")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show profile command help
        assert "**Command: /profile**" in response["message"]
        assert "View your current profile" in response["message"]
        assert "**Usage**:" in response["message"]
        assert "`/profile`" in response["message"]


@pytest.mark.asyncio
async def test_commands_unknown(signal_server, test_config, mock_llm, running_penny):
    """Test /commands <unknown> shows error."""
    async with running_penny(test_config) as _penny:
        # Send /commands unknown
        await signal_server.push_message(sender=TEST_SENDER, content="/commands unknown")

        # Wait for response
        response = await signal_server.wait_for_message(timeout=5.0)

        # Should show error
        assert "Unknown command: /unknown" in response["message"]
        assert "Use /commands to see available commands" in response["message"]
