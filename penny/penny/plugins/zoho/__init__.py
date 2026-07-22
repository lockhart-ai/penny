"""Zoho plugin for Penny.

Provides calendar and project management via Zoho APIs on the chat tool
surface. Email is handled by the existing penny.zoho client; this plugin adds
calendar and project tools when Zoho credentials are configured.

Required environment variables:
    ZOHO_API_ID       — Zoho OAuth client ID
    ZOHO_API_SECRET   — Zoho OAuth client secret
    ZOHO_REFRESH_TOKEN — Zoho OAuth refresh token
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.config import Config
from penny.plugins import CAPABILITY_CALENDAR, CAPABILITY_PROJECT, Plugin
from penny.plugins.zoho.calendar_client import ZohoCalendarClient
from penny.plugins.zoho.calendar_tools import calendar_tools as calendar_tools
from penny.plugins.zoho.project_tools import project_tools as project_tools
from penny.plugins.zoho.projects_client import ZohoProjectsClient

if TYPE_CHECKING:
    from penny.tools.base import Tool


class ZohoPlugin(Plugin):
    """Zoho integration plugin for calendar and projects."""

    name = "zoho"
    capabilities = [CAPABILITY_CALENDAR, CAPABILITY_PROJECT]

    def __init__(self, config: Config) -> None:
        self._client_id = config.zoho_api_id
        self._client_secret = config.zoho_api_secret
        self._refresh_token = config.zoho_refresh_token
        if not self._client_id or not self._client_secret or not self._refresh_token:
            raise ValueError(
                "ZohoPlugin requires ZOHO_API_ID, ZOHO_API_SECRET, and ZOHO_REFRESH_TOKEN"
            )
        self._calendar_client = ZohoCalendarClient(
            client_id=self._client_id,
            client_secret=self._client_secret,
            refresh_token=self._refresh_token,
        )
        self._projects_client = ZohoProjectsClient(
            client_id=self._client_id,
            client_secret=self._client_secret,
            refresh_token=self._refresh_token,
        )

    @classmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if all Zoho credentials are present."""
        return bool(config.zoho_api_id and config.zoho_api_secret and config.zoho_refresh_token)

    def get_tools(self) -> list[Tool]:
        """Return Zoho calendar and project tools."""
        return [
            *calendar_tools(self._calendar_client),
            *project_tools(self._projects_client),
        ]

    async def close(self) -> None:
        await self._calendar_client.close()
        await self._projects_client.close()


PLUGIN_CLASS = ZohoPlugin
