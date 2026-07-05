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

from penny.llm.client import (
    LlmClient,
    _extract_model_ids,
    _summarize_httpx_error,
    _summarize_llm_error,
)
from penny.llm.models import LlmNotFoundError, LlmResponseError

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


class TestSummarizeHttpxError:
    """``list_embedding_models`` hits a raw ``/v1/embeddings/models`` endpoint
    (not the SDK), so its error summarization takes an ``httpx.Response`` rather
    than an ``openai`` error — same body-free contract as ``_summarize_llm_error``."""

    def test_json_message_field_is_surfaced(self) -> None:
        response = httpx.Response(404, json={"error": {"message": "model `bar` not found"}})

        assert _summarize_httpx_error(response) == "HTTP 404: model `bar` not found"

    def test_html_body_is_summarized_not_dumped(self) -> None:
        response = httpx.Response(
            404, headers={"content-type": "text/html"}, content=_HTML_ERROR_BODY.encode()
        )

        summary = _summarize_httpx_error(response)

        assert "HTTP 404" in summary
        assert "non-JSON error body" in summary
        assert "text/html" in summary
        assert "<!DOCTYPE" not in summary  # the raw body never leaks


class TestExtractModelIds:
    """The fallback endpoint's payload shape varies by provider — a ``data`` or
    ``models`` envelope, dict items keyed by ``id`` or ``name``, or bare strings."""

    def test_openai_data_id_shape(self) -> None:
        payload = {"data": [{"id": "embeddinggemma"}, {"id": "other"}]}

        assert _extract_model_ids(payload) == ["embeddinggemma", "other"]

    def test_models_envelope_with_name_shape(self) -> None:
        payload = {"models": [{"name": "embeddinggemma"}]}

        assert _extract_model_ids(payload) == ["embeddinggemma"]

    def test_bare_list_of_strings(self) -> None:
        assert _extract_model_ids(["a", "b"]) == ["a", "b"]

    def test_unrecognized_items_are_skipped(self) -> None:
        payload = {"data": ["a", {"id": "b"}, {"name": "c"}, {"foo": "bar"}, 123]}

        assert _extract_model_ids(payload) == ["a", "b", "c"]

    def test_non_list_payload_raises(self) -> None:
        with pytest.raises(LlmResponseError):
            _extract_model_ids({"data": {"not": "a list"}})


def _mock_embeddings_endpoint(monkeypatch, handler) -> None:
    """Route the raw httpx client used by ``list_embedding_models`` through an
    httpx ``MockTransport`` — mock at the HTTP boundary, no live network."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("penny.llm.client.httpx.AsyncClient", factory)


class TestListEmbeddingModels:
    @pytest.mark.asyncio
    async def test_parses_ids_from_embeddings_endpoint(self, monkeypatch) -> None:
        """A 200 from /v1/embeddings/models is parsed into model ids."""

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/embeddings/models"
            return httpx.Response(200, json={"data": [{"id": "embeddinggemma"}]})

        _mock_embeddings_endpoint(monkeypatch, handler)
        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )

        assert await client.list_embedding_models() == ["embeddinggemma"]

        await client.close()

    @pytest.mark.asyncio
    async def test_http_error_raises_summarized_response_error(self, monkeypatch) -> None:
        """An HTML 404 from the endpoint raises a summarized LlmResponseError, not the body."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, headers={"content-type": "text/html"}, content=_HTML_ERROR_BODY.encode()
            )

        _mock_embeddings_endpoint(monkeypatch, handler)
        client = LlmClient(
            api_url="http://localhost:11434", model="m", max_retries=1, retry_delay=0.0
        )

        with pytest.raises(LlmResponseError) as exc_info:
            await client.list_embedding_models()

        message = str(exc_info.value)
        assert "HTTP 404" in message
        assert "<!DOCTYPE" not in message  # summarized, not the raw HTML body

        await client.close()
