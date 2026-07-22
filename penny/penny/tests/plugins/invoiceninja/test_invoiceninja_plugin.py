"""Tests for the InvoiceNinja plugin."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from penny.plugins.invoiceninja import InvoiceNinjaPlugin
from penny.plugins.invoiceninja.client import InvoiceNinjaClient


def test_invoice_ninja_plugin_not_configured():
    config = MagicMock()
    config.invoiceninja_api_token = None
    config.invoiceninja_url = "https://example.com"
    assert InvoiceNinjaPlugin.is_configured(config) is False


def test_invoice_ninja_plugin_configured():
    config = MagicMock()
    config.invoiceninja_api_token = "token"
    config.invoiceninja_url = "https://example.com"
    assert InvoiceNinjaPlugin.is_configured(config) is True


def test_invoice_ninja_plugin_provides_tools():
    config = MagicMock()
    config.invoiceninja_api_token = "token"
    config.invoiceninja_url = "https://example.com"
    plugin = InvoiceNinjaPlugin(config)
    tools = plugin.get_tools()
    assert len(tools) == 2
    assert {tool.name for tool in tools} == {"verify_invoiceninja_auth", "list_invoices"}


def test_client_sets_required_auth_headers():
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        InvoiceNinjaClient(api_token="test-token", base_url="https://invoicing.example.com")
    mock_client_cls.assert_called_once_with(
        timeout=30.0,
        headers={
            "X-API-TOKEN": "test-token",
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/json",
        },
    )


@pytest.mark.asyncio
async def test_verify_auth_hits_health_check_endpoint():
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/health_check")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(200, json={"status": "ok"}, request=request)
        )
        result = await client.verify_auth()
    assert result is True
    mock_http.get.assert_awaited_once_with("https://invoicing.example.com/api/v1/health_check")


@pytest.mark.asyncio
async def test_verify_auth_raises_on_invalid_token():
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/health_check")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(api_token="bad-token", base_url="https://invoicing.example.com")
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(403, json={"message": "Invalid token"}, request=request)
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.verify_auth()
