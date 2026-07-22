"""InvoiceNinja tools — LLM-callable tools for InvoiceNinja."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

logger = logging.getLogger(__name__)


class ListInvoicesArgs(ToolArgs):
    """Arguments for listing invoices."""

    status: str | None = Field(
        default=None,
        description="Filter by status: 'draft', 'sent', 'partial', 'paid', 'overdue'. "
        "Omit to return all invoices.",
    )


class VerifyAuthTool(Tool):
    """Verify connectivity and authentication with InvoiceNinja."""

    name = "verify_invoiceninja_auth"
    description = (
        "Verify that the InvoiceNinja API token and base URL are configured "
        "correctly by making an authenticated health-check request."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    args_model = ToolArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Verifying InvoiceNinja authentication"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Verify auth."""
        await self._client.verify_auth()
        return ToolResult(message="InvoiceNinja authentication is working.")


class ListInvoicesTool(Tool):
    """List invoices from InvoiceNinja."""

    name = "list_invoices"
    description = (
        "List invoices from InvoiceNinja. Returns invoice numbers, client names, "
        "amounts, statuses, and due dates."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": (
                    "Filter by status: 'draft', 'sent', 'partial', 'paid', 'overdue'. "
                    "Omit to return all invoices."
                ),
            },
        },
        "required": [],
    }
    args_model = ListInvoicesArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing invoices"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List invoices."""
        args = ListInvoicesArgs(**kwargs)
        invoices = await self._client.list_invoices(status=args.status)
        if not invoices:
            return ToolResult(message="No invoices found.")

        lines = [f"Found {len(invoices)} invoice(s):\n"]
        for invoice in invoices:
            lines.append(str(invoice))
        return ToolResult(message="\n".join(lines))
