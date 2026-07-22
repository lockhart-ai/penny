"""Tests for the Zoho plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

from penny.plugins.zoho import ZohoPlugin


def test_zoho_plugin_not_configured():
    config = MagicMock()
    config.zoho_api_id = None
    config.zoho_api_secret = "secret"
    config.zoho_refresh_token = "token"
    assert ZohoPlugin.is_configured(config) is False


def test_zoho_plugin_configured():
    config = MagicMock()
    config.zoho_api_id = "id"
    config.zoho_api_secret = "secret"
    config.zoho_refresh_token = "token"
    assert ZohoPlugin.is_configured(config) is True


def test_zoho_plugin_provides_tools():
    config = MagicMock()
    config.zoho_api_id = "id"
    config.zoho_api_secret = "secret"
    config.zoho_refresh_token = "token"
    plugin = ZohoPlugin(config)
    tools = plugin.get_tools()
    assert len(tools) > 0
    tool_names = {tool.name for tool in tools}
    assert "list_calendars" in tool_names
    assert "list_projects" in tool_names
