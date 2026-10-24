"""Unit tests for ``MfluxImageClient`` — the client for the local mflux image server
(``image_server/``, #2792).

Mocks at the HTTP boundary via ``httpx.MockTransport`` (same pattern as
``test_llm_client.py``'s ``TestListEmbeddingModels``), never a live network.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from penny.llm.image_client import ImageServerError, MfluxImageClient

_SERVED_MODEL = "flux2-klein-4b"
_ONE_PX_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")


def _mock_image_server(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Route the raw httpx client ``MfluxImageClient`` uses through a ``MockTransport``."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr("penny.llm.image_client.httpx.AsyncClient", factory)


def _client(**overrides) -> MfluxImageClient:
    kwargs = {
        "api_url": "http://localhost:8765",
        "model": _SERVED_MODEL,
        "width": 768,
        "height": 768,
        "steps": 4,
        "max_retries": 3,
        "retry_delay": 0.0,
    }
    kwargs.update(overrides)
    return MfluxImageClient(**kwargs)


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_success_returns_the_base64_image(self, monkeypatch) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"image": _ONE_PX_PNG_B64})

        _mock_image_server(monkeypatch, handler)
        client = _client()

        result = await client.generate_image("a teal dragon")

        assert result == _ONE_PX_PNG_B64
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_400_is_not_retried_and_names_the_servers_reason(self, monkeypatch) -> None:
        """An unsupported-model 400 is a firm answer — one call, the server's own reason."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(400, json={"error": "unsupported model 'x'"})

        _mock_image_server(monkeypatch, handler)
        client = _client()

        with pytest.raises(ImageServerError, match="unsupported model"):
            await client.generate_image("a teal dragon")

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_500_is_not_retried_and_names_the_servers_reason(self, monkeypatch) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(500, json={"error": "generation failed: out of memory"})

        _mock_image_server(monkeypatch, handler)
        client = _client()

        with pytest.raises(ImageServerError, match="out of memory"):
            await client.generate_image("a teal dragon")

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_503_busy_is_retried_up_to_the_bound(self, monkeypatch) -> None:
        """503 (a generation is already in flight) is the one status worth a bounded retry."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(503, json={"error": "busy, a generation is already in progress"})

        _mock_image_server(monkeypatch, handler)
        client = _client(max_retries=3)

        with pytest.raises(ImageServerError, match="busy"):
            await client.generate_image("a teal dragon")

        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_a_connection_error_is_retried_up_to_the_bound(self, monkeypatch) -> None:
        """The server being unreachable is retried, bounded, same as a busy response."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ConnectError("connection refused")

        _mock_image_server(monkeypatch, handler)
        client = _client(max_retries=3)

        with pytest.raises(httpx.ConnectError):
            await client.generate_image("a teal dragon")

        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_a_503_that_then_succeeds_returns_the_image(self, monkeypatch) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json={"image": _ONE_PX_PNG_B64})

        _mock_image_server(monkeypatch, handler)
        client = _client(max_retries=3)

        result = await client.generate_image("a teal dragon")

        assert result == _ONE_PX_PNG_B64
        assert len(calls) == 2


class TestHealth:
    @pytest.mark.asyncio
    async def test_matching_served_model_is_ok(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"status": "ok", "served_model": _SERVED_MODEL, "model_loaded": True}
            )

        _mock_image_server(monkeypatch, handler)
        client = _client()

        ok, detail = await client.health()

        assert ok is True
        assert _SERVED_MODEL in detail

    @pytest.mark.asyncio
    async def test_mismatched_served_model_is_not_ok(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"status": "ok", "served_model": "a-different-model", "model_loaded": False},
            )

        _mock_image_server(monkeypatch, handler)
        client = _client()

        ok, detail = await client.health()

        assert ok is False
        assert "a-different-model" in detail

    @pytest.mark.asyncio
    async def test_unreachable_server_is_not_ok(self, monkeypatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        _mock_image_server(monkeypatch, handler)
        client = _client()

        ok, detail = await client.health()

        assert ok is False
        assert detail
