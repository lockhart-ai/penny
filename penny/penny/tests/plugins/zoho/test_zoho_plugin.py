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
    config.signal_number = "+15551234567"
    config.runtime.JMAP_REQUEST_TIMEOUT = 30.0
    config.runtime.EMAIL_BODY_MAX_LENGTH = 50000
    config.runtime.EMAIL_SEARCH_LIMIT = 50
    config.runtime.EMAIL_LIST_LIMIT = 20
    db = MagicMock()
    plugin = ZohoPlugin(config, db)
    # db is injected through construction, not reached from config.runtime.
    assert plugin._db is db
    tools = plugin.get_tools()
    assert len(tools) > 0
    tool_names = {tool.name for tool in tools}
    assert "list_calendars" in tool_names
    assert "list_projects" in tool_names
    assert "move_emails" in tool_names
    assert "create_folder" in tool_names
    assert "apply_label" in tool_names
    assert "list_labels" in tool_names
    assert "create_email_rule" in tool_names
    assert "list_email_rules" in tool_names
