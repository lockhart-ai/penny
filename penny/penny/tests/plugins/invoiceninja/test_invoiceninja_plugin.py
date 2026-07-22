"""Tests for the InvoiceNinja plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

from penny.plugins.invoiceninja import InvoiceNinjaPlugin


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


def test_invoice_ninja_plugin_provides_list_invoices_tool():
    config = MagicMock()
    config.invoiceninja_api_token = "token"
    config.invoiceninja_url = "https://example.com"
    plugin = InvoiceNinjaPlugin(config)
    tools = plugin.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "list_invoices"
