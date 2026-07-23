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
    assert len(tools) == 9
    assert {tool.name for tool in tools} == {
        "create_expense",
        "create_expense_category",
        "get_expense",
        "get_expense_category",
        "list_expense_categories",
        "list_expenses",
        "list_invoices",
        "update_expense",
        "verify_invoiceninja_auth",
    }


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


@pytest.mark.asyncio
async def test_create_expense_posts_required_fields():
    request = httpx.Request("POST", "https://invoicing.example.com/api/v1/expenses")
    expense_response = {
        "id": "exp1",
        "amount": 45.0,
        "date": "2026-07-22",
        "status": "logged",
        "vendor_id": "ven1",
        "category_id": "cat1",
    }
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.post = AsyncMock(
            return_value=httpx.Response(200, json={"data": expense_response}, request=request)
        )
        result = await client.create_expense(
            amount=45.0,
            date="2026-07-22",
            vendor_id="ven1",
            category_id="cat1",
            public_notes="Coffee",
        )
    assert result.id == "exp1"
    assert result.amount == 45.0
    mock_http.post.assert_awaited_once()
    call_args = mock_http.post.call_args
    assert call_args[0][0] == "https://invoicing.example.com/api/v1/expenses"
    assert call_args[1]["json"]["date"] == "2026-07-22"
    assert call_args[1]["json"]["amount"] == 45.0
    assert call_args[1]["json"]["vendor_id"] == "ven1"
    assert call_args[1]["json"]["category_id"] == "cat1"
    assert call_args[1]["json"]["public_notes"] == "Coffee"


@pytest.mark.asyncio
async def test_list_expenses_returns_parsed_expenses():
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/expenses")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "exp1",
                            "amount": 45.0,
                            "date": "2026-07-22",
                            "status": "logged",
                            "public_notes": "Coffee",
                        }
                    ]
                },
                request=request,
            )
        )
        expenses = await client.list_expenses(status="logged", limit=10)
    assert len(expenses) == 1
    assert expenses[0].id == "exp1"
    assert expenses[0].status == "logged"
    mock_http.get.assert_awaited_once_with(
        "https://invoicing.example.com/api/v1/expenses",
        params={"per_page": "10", "status": "logged"},
    )


@pytest.mark.asyncio
async def test_list_expenses_omits_per_page_when_no_limit():
    """No invented default cap — ``per_page`` is sent only when a limit is given."""
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/expenses")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(200, json={"data": []}, request=request)
        )
        await client.list_expenses()
    mock_http.get.assert_awaited_once_with(
        "https://invoicing.example.com/api/v1/expenses",
        params={},
    )


@pytest.mark.asyncio
async def test_list_invoices_handles_null_client():
    """An invoice whose ``client`` relationship is JSON null parses without crashing."""
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/invoices")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "inv1", "number": "0001", "client": None, "amount": 10.0}]},
                request=request,
            )
        )
        invoices = await client.list_invoices()
    assert len(invoices) == 1
    assert invoices[0].client_name == "Unknown"


@pytest.mark.asyncio
async def test_get_expense_fetches_by_id():
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/expenses/exp1")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "id": "exp1",
                        "amount": 45.0,
                        "date": "2026-07-22",
                        "status": "logged",
                    }
                },
                request=request,
            )
        )
        expense = await client.get_expense("exp1")
    assert expense.id == "exp1"
    mock_http.get.assert_awaited_once_with("https://invoicing.example.com/api/v1/expenses/exp1")


@pytest.mark.asyncio
async def test_update_expense_puts_only_changed_fields():
    request = httpx.Request("PUT", "https://invoicing.example.com/api/v1/expenses/exp1")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.put = AsyncMock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "id": "exp1",
                        "amount": 50.0,
                        "date": "2026-07-22",
                        "status": "logged",
                    }
                },
                request=request,
            )
        )
        result = await client.update_expense("exp1", amount=50.0)
    assert result.amount == 50.0
    mock_http.put.assert_awaited_once()
    call_args = mock_http.put.call_args
    assert call_args[0][0] == "https://invoicing.example.com/api/v1/expenses/exp1"
    assert call_args[1]["json"] == {"amount": 50.0}


@pytest.mark.asyncio
async def test_create_expense_category_posts_name():
    request = httpx.Request("POST", "https://invoicing.example.com/api/v1/expense_categories")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.post = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"data": {"id": "cat1", "name": "Travel"}},
                request=request,
            )
        )
        category = await client.create_expense_category("Travel")
    assert category.id == "cat1"
    assert category.name == "Travel"
    mock_http.post.assert_awaited_once_with(
        "https://invoicing.example.com/api/v1/expense_categories",
        json={"name": "Travel"},
    )


@pytest.mark.asyncio
async def test_list_expense_categories_returns_parsed_categories():
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/expense_categories")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "cat1", "name": "Travel"}, {"id": "cat2", "name": "Meals"}]},
                request=request,
            )
        )
        categories = await client.list_expense_categories(limit=25)
    assert len(categories) == 2
    assert categories[0].name == "Travel"
    mock_http.get.assert_awaited_once_with(
        "https://invoicing.example.com/api/v1/expense_categories",
        params={"per_page": "25"},
    )


@pytest.mark.asyncio
async def test_get_expense_category_fetches_by_id():
    request = httpx.Request("GET", "https://invoicing.example.com/api/v1/expense_categories/cat1")
    with patch("penny.plugins.invoiceninja.client.httpx.AsyncClient") as mock_client_cls:
        client = InvoiceNinjaClient(
            api_token="test-token", base_url="https://invoicing.example.com"
        )
        mock_http = mock_client_cls.return_value
        mock_http.get = AsyncMock(
            return_value=httpx.Response(
                200,
                json={"data": {"id": "cat1", "name": "Travel"}},
                request=request,
            )
        )
        category = await client.get_expense_category("cat1")
    assert category.id == "cat1"
    assert category.name == "Travel"
    mock_http.get.assert_awaited_once_with(
        "https://invoicing.example.com/api/v1/expense_categories/cat1"
    )
