"""Tests for LlmClient error summarization.

A non-Ollama backend can return a non-JSON error (e.g. a 404 served as an HTML
page). Logging the raw body dumped thousands of characters per occurrence and
buried real signal, so ``_summarize_llm_error`` reports the HTTP status plus a
short, body-free detail instead — and that summary is what propagates through
the raised ``LlmError`` as well.
"""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from penny.llm.client import LlmClient, _summarize_llm_error
from penny.llm.models import LlmNotFoundError

_HTML_ERROR_BODY = (
    f"<!DOCTYPE html><html><head><title>404</title></head><body>{'x' * 5000}</body></html>"
)


def _make_status_error(
    status: int, content_type: str, content: bytes, body: object | None
) -> openai.APIStatusError:
    """Build a real OpenAI status error carrying the given HTTP response."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        status, headers={"content-type": content_type}, content=content, request=request
    )
    return openai.NotFoundError("Error code: 404", response=response, body=body)


class TestSummarizeLlmError:
    def test_html_error_body_is_not_dumped(self) -> None:
        """A 404 served as an HTML page is summarized by type + length, never dumped."""
        error = _make_status_error(404, "text/html", _HTML_ERROR_BODY.encode(), body=None)

        summary = _summarize_llm_error(error)

        assert "HTTP 404" in summary
        assert "non-JSON error body" in summary
        assert "text/html" in summary
        assert "<!DOCTYPE" not in summary  # the raw body never leaks into the log
        assert len(summary) < len(_HTML_ERROR_BODY)  # summarized, not dumped

    def test_json_error_surfaces_message_field(self) -> None:
        """A structured JSON error surfaces its short ``message`` field with the status."""
        body = {"error": {"message": "model `foo` not found", "type": "invalid_request_error"}}
        error = _make_status_error(404, "application/json", json.dumps(body).encode(), body=body)

        assert _summarize_llm_error(error) == "HTTP 404: model `foo` not found"

    def test_connection_error_without_response_uses_own_message(self) -> None:
        """An error with no HTTP response (connection/timeout) falls back to its short str."""
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        error = openai.APIConnectionError(message="Connection refused", request=request)

        summary = _summarize_llm_error(error)

        assert "Connection refused" in summary


class TestChatPropagatesSummarizedError:
    """The summarized message — not the raw HTML body — is what reaches the
    raised ``LlmError``, so an onboarding/profile call that 404s surfaces a
    readable reason instead of a wall of HTML."""

    @pytest.mark.asyncio
    async def test_html_404_raises_summarized_not_found(self, monkeypatch) -> None:
        client = LlmClient(
            api_url="http://localhost:11434",
            model="missing-model",
            max_retries=1,
            retry_delay=0.0,
        )
        error = _make_status_error(404, "text/html", _HTML_ERROR_BODY.encode(), body=None)

        async def raise_not_found(**kwargs):
            raise error

        monkeypatch.setattr(client.client.chat.completions, "create", raise_not_found)

        with pytest.raises(LlmNotFoundError) as exc_info:
            await client.chat([{"role": "user", "content": "hi"}])

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "<!DOCTYPE" not in message  # summarized, not the raw HTML body

        await client.close()
