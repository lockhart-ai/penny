"""InvoiceNinja plugin for Penny.

Provides invoice querying via the InvoiceNinja v5 API.
Required environment variables:
    INVOICENINJA_API_TOKEN — InvoiceNinja API token
    INVOICENINJA_URL       — InvoiceNinja instance URL (e.g. https://invoicing.co)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from penny.config import Config
from penny.plugins import CAPABILITY_INVOICING, Plugin
from penny.plugins.invoiceninja.client import InvoiceNinjaClient
from penny.plugins.invoiceninja.tools import (
    CreateExpenseTool,
    GetExpenseTool,
    ListExpensesTool,
    ListInvoicesTool,
    UpdateExpenseTool,
    VerifyAuthTool,
)

if TYPE_CHECKING:
    from penny.tools.base import Tool


class InvoiceNinjaPlugin(Plugin):
    """InvoiceNinja integration plugin."""

    name = "invoiceninja"
    capabilities = [CAPABILITY_INVOICING]

    def __init__(self, config: Config) -> None:
        self._client = InvoiceNinjaClient(
            api_token=config.invoiceninja_api_token,
            base_url=config.invoiceninja_url,
        )

    @classmethod
    def is_configured(cls, config: Config) -> bool:
        """Return True if InvoiceNinja credentials are present."""
        return bool(config.invoiceninja_api_token and config.invoiceninja_url)

    def get_tools(self) -> list[Tool]:
        """Return InvoiceNinja tools."""
        return [
            VerifyAuthTool(self._client),
            ListInvoicesTool(self._client),
            CreateExpenseTool(self._client),
            ListExpensesTool(self._client),
            GetExpenseTool(self._client),
            UpdateExpenseTool(self._client),
        ]

    async def close(self) -> None:
        await self._client.close()


PLUGIN_CLASS = InvoiceNinjaPlugin
