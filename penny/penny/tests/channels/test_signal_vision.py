"""Integration tests for Signal image attachment handling.

Since #2635 an image rides the current turn's own user message straight to the
chat model — no separate caption call, no vision model, full tool surface.
"""

import base64

import pytest

from penny.constants import PennyConstants
from penny.tests.conftest import TEST_SENDER, wait_until

# Minimal valid JPEG header bytes for testing
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def _image_urls(request) -> list[str]:
    """Every image_url data URI in a request's user messages, in order."""
    urls: list[str] = []
    for message in request["messages"]:
        if message["role"] == "user" and isinstance(message.get("content"), list):
            urls.extend(
                part["image_url"]["url"]
                for part in message["content"]
                if part.get("type") == "image_url"
            )
    return urls


@pytest.mark.asyncio
async def test_image_with_text_goes_to_the_chat_model(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Image + text: one call to the chat model, the image on the turn's user message."""
    config = make_config()
    mock_llm.set_default_flow(final_response="that's a cute cat! 🐱")

    async with running_penny(config) as penny:
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text="what's in this photo?",
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cat" in response["message"].lower()

        await wait_until(lambda: len(mock_llm.requests) >= 1)
        # Every call so far went to the chat model — no separate caption call (#2635)
        assert all(r["model"] == "test-model" for r in mock_llm.requests)
        request = mock_llm.requests[0]

        # The image rides the turn's own user message as a data URI carrying the
        # attachment's real mime type
        expected_uri = f"data:image/jpeg;base64,{base64.b64encode(FAKE_JPEG).decode()}"
        assert _image_urls(request) == [expected_uri]

        # The full tool surface is installed — an image turn is an ordinary turn
        assert request.get("tools")

        # The normal conversation prompt, not a vision-specific instruction
        system_msgs = [m for m in request["messages"] if m["role"] == "system"]
        assert "sent an image" not in system_msgs[0]["content"]

        # The conversation record keeps a marker where the image was, so a later
        # turn reads it without the image ever being resent
        history = penny.chat_agent.get_history(TEST_SENDER)
        assert any(
            role == "user"
            and f"what's in this photo? {PennyConstants.IMAGE_HISTORY_MARKER}" in content
            for role, content in history
        )


@pytest.mark.asyncio
async def test_image_without_text_goes_to_the_chat_model(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Image with no text: one chat-model call, the image on an otherwise-empty user turn."""
    config = make_config()
    mock_llm.set_default_flow(final_response="beautiful sunset! 🌅")

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text=None,  # No text
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert "sunset" in response["message"].lower()

        await wait_until(lambda: len(mock_llm.requests) >= 1)
        assert all(r["model"] == "test-model" for r in mock_llm.requests)
        request = mock_llm.requests[0]
        expected_uri = f"data:image/jpeg;base64,{base64.b64encode(FAKE_JPEG).decode()}"
        assert _image_urls(request) == [expected_uri]
        assert request.get("tools")


@pytest.mark.asyncio
async def test_non_image_attachment_ignored(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Non-image attachments (e.g., PDF) are skipped, not sent to the model."""
    config = make_config()
    mock_llm.set_default_flow(
        final_response="here's what I found about documents 📄",
    )

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=b"%PDF-1.4...",
            content_type="application/pdf",
            text="check this document",
        )

        await signal_server.wait_for_message(timeout=10.0)

        await wait_until(lambda: len(mock_llm.requests) >= 1)
        first_request = mock_llm.requests[0]
        assert _image_urls(first_request) == []
        user_messages = [m for m in first_request["messages"] if m["role"] == "user"]
        assert not any("images" in m for m in user_messages)
