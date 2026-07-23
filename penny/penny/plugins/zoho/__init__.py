"""Zoho plugin for Penny.

Provides email, calendar, and project management via Zoho APIs on the chat
tool surface. Email organisation tools (move, label, folders, rules) are
included alongside calendar and project tools when Zoho credentials are
configured.

Required environment variables:
    ZOHO_API_ID       — Zoho OAuth client ID
    ZOHO_API_SECRET   — Zoho OAuth client secret
    ZOHO_REFRESH_TOKEN — Zoho OAuth refresh token
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.config import Config
from penny.constants import PennyConstants
from penny.plugins import CAPABILITY_CALENDAR, CAPABILITY_EMAIL, CAPABILITY_PROJECT, Plugin
from penny.plugins.zoho.calendar_client import ZohoCalendarClient
from penny.plugins.zoho.calendar_tools import calendar_tools as calendar_tools
from penny.plugins.zoho.email_tools import (
    ApplyLabelTool,
    CreateEmailRuleTool,
    CreateFolderTool,
    ListEmailRulesTool,
    ListLabelsTool,
    MoveEmailsTool,
)
from penny.plugins.zoho.mail_client import ZohoClient
from penny.plugins.zoho.project_tools import project_tools as project_tools
from penny.plugins.zoho.projects_client import ZohoProjectsClient

if TYPE_CHECKING:
    from penny.database import Database
    from penny.tools.base import Tool


class ZohoPlugin(Plugin):
    """Zoho integration plugin for email, calendar, and projects."""

    name = "zoho"
    capabilities = [CAPABILITY_EMAIL, CAPABILITY_CALENDAR, CAPABILITY_PROJECT]

    def __init__(self, config: Config, db: Database) -> None:
        super().__init__(config, db)
        self._client_id = config.zoho_api_id
        self._client_secret = config.zoho_api_secret
        self._refresh_token = config.zoho_refresh_token
        if not self._client_id or not self._client_secret or not self._refresh_token:
            raise ValueError(
                "ZohoPlugin requires ZOHO_API_ID, ZOHO_API_SECRET, and ZOHO_REFRESH_TOKEN"
            )
        self._email_client = ZohoClient(
            client_id=self._client_id,
            client_secret=self._client_secret,
            refresh_token=self._refresh_token,
            timeout=config.runtime.JMAP_REQUEST_TIMEOUT,
            max_body_length=int(config.runtime.EMAIL_BODY_MAX_LENGTH),
            search_limit=int(config.runtime.EMAIL_SEARCH_LIMIT),
            list_limit=int(config.runtime.EMAIL_LIST_LIMIT),
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
        """Return Zoho email, calendar, and project tools."""
        return [
            MoveEmailsTool(self._email_client),
            CreateFolderTool(self._email_client),
            ApplyLabelTool(self._email_client),
            ListLabelsTool(self._email_client),
            CreateEmailRuleTool(self._db, PennyConstants.PROVIDER_ZOHO),
            ListEmailRulesTool(self._db, PennyConstants.PROVIDER_ZOHO),
            *calendar_tools(self._calendar_client),
            *project_tools(self._projects_client),
        ]

    async def close(self) -> None:
        await self._email_client.close()
        await self._calendar_client.close()
        await self._projects_client.close()


PLUGIN_CLASS = ZohoPlugin
