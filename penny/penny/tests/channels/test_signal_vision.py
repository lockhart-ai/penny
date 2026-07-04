"""Integration tests for Signal vision (image attachment) handling."""

import pytest

from penny.prompts import Prompt
from penny.responses import PennyResponse
from penny.tests.conftest import TEST_SENDER, wait_until

# Minimal valid JPEG header bytes for testing
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.mark.asyncio
async def test_image_with_text_captions_then_forwards(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Image + text: vision model captions, then main model responds without search."""
    config = make_config(llm_vision_model="test-vision-model")

    def handler(request, count):
        if count == 1:
            # Vision model captioning call
            return mock_llm._make_text_response(request, "a cute orange cat")
        # Main model: direct response without tool call
        return mock_llm._make_text_response(request, "that's a cute cat! 🐱")

    mock_llm.set_response_handler(handler)

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text="what's in this photo?",
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert "cat" in response["message"].lower()

        # Verify two-step flow: vision model called first, then main model
        await wait_until(lambda: len(mock_llm.requests) >= 2)

        # First call: vision model with images
        caption_request = mock_llm.requests[0]
        assert caption_request["model"] == "test-vision-model"
        user_msgs = [m for m in caption_request["messages"] if m["role"] == "user"]
        # Vision messages use OpenAI content-parts format with image_url entries
        assert any(
            isinstance(m.get("content"), list)
            and any(part.get("type") == "image_url" for part in m["content"])
            for m in user_msgs
        )
        assert any(
            isinstance(m.get("content"), list)
            and any(
                Prompt.VISION_AUTO_DESCRIBE_PROMPT in part.get("text", "")
                for part in m["content"]
                if part.get("type") == "text"
            )
            for m in user_msgs
        )

        # Second call: main model with combined text prompt (no images, no tools)
        foreground_request = mock_llm.requests[1]
        assert foreground_request["model"] == "test-model"
        user_msgs = [m for m in foreground_request["messages"] if m["role"] == "user"]
        assert not any(
            isinstance(m.get("content"), list)
            and any(part.get("type") == "image_url" for part in m["content"])
            for m in user_msgs
        )
        # Combined prompt should contain user text and caption
        expected = Prompt.VISION_IMAGE_CONTEXT.format(
            user_text="what's in this photo?", caption="a cute orange cat"
        )
        assert any(expected in m.get("content", "") for m in user_msgs)
        # Verify no tools were provided (None = tools disabled)
        assert foreground_request.get("tools") is None
        # Verify vision response prompt (not search prompt) was used
        system_msgs = [m for m in foreground_request["messages"] if m["role"] == "system"]
        system_text = system_msgs[0]["content"]
        assert "sent an image" in system_text
        assert Prompt.VISION_RESPONSE_PROMPT in system_text


@pytest.mark.asyncio
async def test_image_without_text_captions_then_forwards(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Image with no text: vision model captions, forwarded without search."""
    config = make_config(llm_vision_model="test-vision-model")

    def handler(request, count):
        if count == 1:
            # Vision model captioning
            return mock_llm._make_text_response(request, "a sunset over the ocean")
        # Main model: direct response without tool call
        return mock_llm._make_text_response(request, "beautiful sunset! 🌅")

    mock_llm.set_response_handler(handler)

    async with running_penny(config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text=None,  # No text
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert "sunset" in response["message"].lower()

        await wait_until(lambda: len(mock_llm.requests) >= 2)

        # First call: vision model captioning with describe prompt
        caption_request = mock_llm.requests[0]
        assert caption_request["model"] == "test-vision-model"

        # Second call: main model with image-only context (no images, no tools)
        foreground_request = mock_llm.requests[1]
        assert foreground_request["model"] == "test-model"
        user_msgs = [m for m in foreground_request["messages"] if m["role"] == "user"]
        assert not any("images" in m for m in user_msgs)
        expected = PennyResponse.VISION_IMAGE_ONLY_CONTEXT.format(caption="a sunset over the ocean")
        assert any(expected in m.get("content", "") for m in user_msgs)
        # Verify no tools were provided (None = tools disabled)
        assert foreground_request.get("tools") is None
        # Verify vision response prompt (not search prompt) was used
        system_msgs = [m for m in foreground_request["messages"] if m["role"] == "system"]
        system_text = system_msgs[0]["content"]
        assert "sent an image" in system_text
        assert Prompt.VISION_RESPONSE_PROMPT in system_text


@pytest.mark.asyncio
async def test_image_without_vision_model_sends_acknowledgment(
    signal_server,
    mock_llm,
    test_config,
    test_user_info,
    running_penny,
):
    """When vision model is not configured, send acknowledgment message."""
    # test_config has no llm_vision_model (None by default)
    async with running_penny(test_config):
        await signal_server.push_image_message(
            sender=TEST_SENDER,
            image_data=FAKE_JPEG,
            text="what's this?",
        )

        response = await signal_server.wait_for_message(timeout=10.0)
        assert PennyResponse.VISION_NOT_CONFIGURED_MESSAGE in response["message"]

        # Verify Ollama was NOT called
        assert len(mock_llm.requests) == 0


@pytest.mark.asyncio
async def test_caption_image_raises_when_vision_client_missing(
    signal_server, mock_llm, test_config, test_user_info, running_penny
):
    """``caption_image`` defends against being called when no vision client exists.

    The channel layer normally rejects image messages before they reach
    the chat agent, but the contract on ``caption_image`` itself is that
    it raises explicitly (instead of relying on an ``assert`` that gets
    stripped under ``python -O``).
    """
    async with running_penny(test_config) as penny:
        penny.chat_agent._vision_model_client = None
        with pytest.raises(RuntimeError, match="vision model client"):
            await penny.chat_agent.caption_image("ZmFrZS1pbWFnZS1iYXNlNjQ=")


@pytest.mark.asyncio
async def test_non_image_attachment_ignored(
    signal_server,
    mock_llm,
    make_config,
    test_user_info,
    running_penny,
):
    """Non-image attachments (e.g., PDF) don't trigger vision pipeline."""
    config = make_config(llm_vision_model="test-vision-model")
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

        # Should process as normal text message (no images in request)
        await wait_until(lambda: len(mock_llm.requests) >= 1)
        first_request = mock_llm.requests[0]
        user_messages = [m for m in first_request["messages"] if m["role"] == "user"]
        assert not any("images" in m for m in user_messages)

        # Should use the main model, not vision model
        assert first_request["model"] == "test-model"
