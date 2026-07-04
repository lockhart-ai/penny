"""Integration tests for the startup setup-health / preflight surface.

Drives ``Preflight.run()`` — the public entry point of the preflight surface —
with real ``LlmClient``/``OllamaImageClient`` instances whose ``list_models`` is
stubbed at the boundary, exercising the OK / WARN / hard-FAIL matrix for every
check.
"""

from __future__ import annotations

import logging

import pytest

from penny.config import Config
from penny.llm.client import LlmClient
from penny.llm.image_client import OllamaImageClient
from penny.llm.models import LlmConnectionError, LlmResponseError
from penny.preflight import CheckStatus, Preflight, PreflightCheck, PreflightReport

_DUMMY_URL = "http://localhost:11434"


def _llm_client(model: str) -> LlmClient:
    """A real LlmClient (no network until used) for stubbing list_models on."""
    return LlmClient(api_url=_DUMMY_URL, model=model, max_retries=1, retry_delay=0.0)


def _image_client(model: str) -> OllamaImageClient:
    return OllamaImageClient(api_url=_DUMMY_URL, model=model, max_retries=1, retry_delay=0.0)


def _stub_list_models(
    monkeypatch: pytest.MonkeyPatch,
    client: object,
    *,
    available: list[str] | None = None,
    error: Exception | None = None,
) -> None:
    """Replace ``client.list_models`` with a canned result or raised error."""

    async def _list_models() -> list[str]:
        if error is not None:
            raise error
        return available or []

    monkeypatch.setattr(client, "list_models", _list_models)


def _build_preflight(
    monkeypatch: pytest.MonkeyPatch,
    config: Config,
    *,
    chat_available: list[str] | None = None,
    chat_error: Exception | None = None,
    embed_available: list[str] | None = None,
    embed_error: Exception | None = None,
    vision_available: list[str] | None = None,
    image_available: list[str] | None = None,
    browser_enabled: bool = False,
    browser_connected: bool = False,
    resolved_channel_type: str | None = None,
) -> Preflight:
    """Assemble a Preflight over stubbed clients, defaulting every model present."""
    model_client = _llm_client(config.llm_model)
    _stub_list_models(
        monkeypatch,
        model_client,
        available=chat_available if chat_available is not None else [config.llm_model],
        error=chat_error,
    )
    embedding_client = _llm_client(config.llm_embedding_model)
    _stub_list_models(
        monkeypatch,
        embedding_client,
        available=embed_available if embed_available is not None else [config.llm_embedding_model],
        error=embed_error,
    )

    vision_client: LlmClient | None = None
    if config.llm_vision_model:
        vision_client = _llm_client(config.llm_vision_model)
        _stub_list_models(monkeypatch, vision_client, available=vision_available or [])

    image_client: OllamaImageClient | None = None
    if config.llm_image_model:
        image_client = _image_client(config.llm_image_model)
        _stub_list_models(monkeypatch, image_client, available=image_available or [])

    return Preflight(
        config=config,
        model_client=model_client,
        embedding_client=embedding_client,
        vision_client=vision_client,
        image_client=image_client,
        browser_enabled=browser_enabled,
        browser_connected=browser_connected,
        configured_channel_type=config.channel_type,
        resolved_channel_type=resolved_channel_type
        if resolved_channel_type is not None
        else config.channel_type,
    )


def _status(report: PreflightReport, name: PreflightCheck) -> CheckStatus | None:
    for result in report.results:
        if result.name is name:
            return result.status
    return None


@pytest.mark.asyncio
async def test_all_green(monkeypatch, test_config):
    """Happy path: reachable endpoints, resolvable models, aligned routing → no failures."""
    preflight = _build_preflight(monkeypatch, test_config)
    report = await preflight.run()

    assert not report.has_failures
    assert _status(report, PreflightCheck.LLM_ENDPOINT) is CheckStatus.OK
    assert _status(report, PreflightCheck.EMBEDDING_MODEL) is CheckStatus.OK
    assert _status(report, PreflightCheck.PRIMARY_CHANNEL) is CheckStatus.OK
    # Unconfigured optional models and disabled browser are skipped, not warned.
    assert _status(report, PreflightCheck.VISION_MODEL) is None
    assert _status(report, PreflightCheck.IMAGE_MODEL) is None
    assert _status(report, PreflightCheck.BROWSER_ADDON) is None
    # Logging the report never raises.
    report.log(logging.getLogger("test-preflight"))


@pytest.mark.asyncio
async def test_llm_endpoint_unreachable_hard_fails(monkeypatch, test_config):
    """An unreachable LLM endpoint is a hard failure with an actionable message."""
    preflight = _build_preflight(
        monkeypatch, test_config, chat_error=LlmConnectionError("connection refused")
    )
    report = await preflight.run()

    assert report.has_failures
    assert _status(report, PreflightCheck.LLM_ENDPOINT) is CheckStatus.FAIL
    assert "unreachable" in report.failure_summary()
    assert "LLM_API_URL" in report.failure_summary()


@pytest.mark.asyncio
async def test_chat_model_missing_hard_fails(monkeypatch, test_config):
    """A reachable endpoint that doesn't have the configured chat model hard-fails."""
    preflight = _build_preflight(monkeypatch, test_config, chat_available=["some-other-model"])
    report = await preflight.run()

    assert report.has_failures
    assert _status(report, PreflightCheck.LLM_ENDPOINT) is CheckStatus.FAIL
    assert test_config.llm_model in report.failure_summary()


@pytest.mark.asyncio
async def test_chat_model_listing_unsupported_warns(monkeypatch, test_config):
    """A reachable endpoint whose model listing errors is unverifiable → warn, not fail."""
    preflight = _build_preflight(
        monkeypatch, test_config, chat_error=LlmResponseError("HTTP 404: not found")
    )
    report = await preflight.run()

    assert not report.has_failures
    assert _status(report, PreflightCheck.LLM_ENDPOINT) is CheckStatus.WARN


@pytest.mark.asyncio
async def test_embedding_model_missing_hard_fails(monkeypatch, test_config):
    """The embedding model is a required prerequisite — missing is a hard failure."""
    preflight = _build_preflight(monkeypatch, test_config, embed_available=[])
    report = await preflight.run()

    assert report.has_failures
    assert _status(report, PreflightCheck.EMBEDDING_MODEL) is CheckStatus.FAIL
    assert "LLM_EMBEDDING_MODEL" in report.failure_summary()


@pytest.mark.asyncio
async def test_embedding_endpoint_unreachable_hard_fails(monkeypatch, test_config):
    """An unreachable embedding endpoint is a hard failure."""
    preflight = _build_preflight(
        monkeypatch, test_config, embed_error=LlmConnectionError("connection refused")
    )
    report = await preflight.run()

    assert report.has_failures
    assert _status(report, PreflightCheck.EMBEDDING_MODEL) is CheckStatus.FAIL


@pytest.mark.asyncio
async def test_vision_and_image_missing_soft_warn(monkeypatch, make_config):
    """Configured-but-missing vision/image models warn but never fail startup."""
    config = make_config(llm_vision_model="vision-model", llm_image_model="image-model")
    preflight = _build_preflight(monkeypatch, config, vision_available=[], image_available=[])
    report = await preflight.run()

    assert not report.has_failures
    assert _status(report, PreflightCheck.VISION_MODEL) is CheckStatus.WARN
    assert _status(report, PreflightCheck.IMAGE_MODEL) is CheckStatus.WARN


@pytest.mark.asyncio
async def test_browser_addon_disconnected_soft_warn(monkeypatch, test_config):
    """An enabled-but-disconnected browser addon is a soft warning."""
    preflight = _build_preflight(
        monkeypatch, test_config, browser_enabled=True, browser_connected=False
    )
    report = await preflight.run()

    assert not report.has_failures
    assert _status(report, PreflightCheck.BROWSER_ADDON) is CheckStatus.WARN


@pytest.mark.asyncio
async def test_browser_addon_connected_ok(monkeypatch, test_config):
    """A connected browser addon reports OK."""
    preflight = _build_preflight(
        monkeypatch, test_config, browser_enabled=True, browser_connected=True
    )
    report = await preflight.run()

    assert _status(report, PreflightCheck.BROWSER_ADDON) is CheckStatus.OK


@pytest.mark.asyncio
async def test_primary_channel_mismatch_soft_warn(monkeypatch, test_config):
    """A resolved routing target different from the configured primary warns (routing-bug guard)."""
    preflight = _build_preflight(monkeypatch, test_config, resolved_channel_type="browser")
    report = await preflight.run()

    assert not report.has_failures
    assert _status(report, PreflightCheck.PRIMARY_CHANNEL) is CheckStatus.WARN
    warning = next(r for r in report.results if r.name is PreflightCheck.PRIMARY_CHANNEL)
    assert "browser" in warning.detail and test_config.channel_type in warning.detail
