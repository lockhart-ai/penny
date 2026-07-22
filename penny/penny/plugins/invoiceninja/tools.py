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


class CreateExpenseArgs(ToolArgs):
    """Arguments for creating an expense."""

    amount: float = Field(description="Expense amount.")
    date: str = Field(description="Expense date in YYYY-MM-DD format.")
    vendor_id: str = Field(description="InvoiceNinja vendor ID.")
    category_id: str = Field(description="InvoiceNinja expense category ID.")
    public_notes: str | None = Field(
        default=None,
        description="Optional description shown to clients if the expense is invoiced.",
    )
    private_notes: str | None = Field(default=None, description="Optional internal-only notes.")


class CreateExpenseTool(Tool):
    """Create an expense in InvoiceNinja."""

    name = "create_expense"
    description = "Create a new expense in InvoiceNinja."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Expense amount."},
            "date": {"type": "string", "description": "Expense date in YYYY-MM-DD format."},
            "vendor_id": {"type": "string", "description": "InvoiceNinja vendor ID."},
            "category_id": {"type": "string", "description": "InvoiceNinja expense category ID."},
            "public_notes": {
                "type": "string",
                "description": "Optional description shown to clients if the expense is invoiced.",
            },
            "private_notes": {
                "type": "string",
                "description": "Optional internal-only notes.",
            },
        },
        "required": ["amount", "date", "vendor_id", "category_id"],
    }
    args_model = CreateExpenseArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating InvoiceNinja expense"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create an expense."""
        args = CreateExpenseArgs(**kwargs)
        expense = await self._client.create_expense(
            amount=args.amount,
            date=args.date,
            vendor_id=args.vendor_id,
            category_id=args.category_id,
            public_notes=args.public_notes,
            private_notes=args.private_notes,
        )
        return ToolResult(message=f"Created expense: {expense}")


class ListExpensesArgs(ToolArgs):
    """Arguments for listing expenses."""

    status: str | None = Field(
        default=None,
        description="Optional status filter (e.g., 'logged', 'pending', 'invoiced', 'paid').",
    )
    limit: int | None = Field(
        default=None,
        description="Optional cap on the number of expenses; the API's default page size applies "
        "when unset.",
    )


class ListExpensesTool(Tool):
    """List expenses from InvoiceNinja."""

    name = "list_expenses"
    description = "List expenses from InvoiceNinja with optional status and limit filters."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": (
                    "Optional status filter (e.g., 'logged', 'pending', 'invoiced', 'paid')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Optional cap on the number of expenses; the API's default page "
                "size applies when unset.",
            },
        },
        "required": [],
    }
    args_model = ListExpensesArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing expenses"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List expenses."""
        args = ListExpensesArgs(**kwargs)
        expenses = await self._client.list_expenses(status=args.status, limit=args.limit)
        if not expenses:
            return ToolResult(message="No expenses found.")

        lines = [f"Found {len(expenses)} expense(s):\n"]
        for expense in expenses:
            lines.append(str(expense))
        return ToolResult(message="\n".join(lines))


class GetExpenseArgs(ToolArgs):
    """Arguments for fetching a single expense."""

    expense_id: str = Field(description="InvoiceNinja expense ID.")


class GetExpenseTool(Tool):
    """Fetch a single expense from InvoiceNinja."""

    name = "get_expense"
    description = "Fetch a single expense from InvoiceNinja by ID."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expense_id": {"type": "string", "description": "InvoiceNinja expense ID."},
        },
        "required": ["expense_id"],
    }
    args_model = GetExpenseArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Fetching expense"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get an expense."""
        args = GetExpenseArgs(**kwargs)
        expense = await self._client.get_expense(args.expense_id)
        return ToolResult(message=str(expense))


class UpdateExpenseArgs(ToolArgs):
    """Arguments for updating an expense."""

    expense_id: str = Field(description="InvoiceNinja expense ID.")
    amount: float | None = Field(default=None, description="New expense amount.")
    date: str | None = Field(default=None, description="New date in YYYY-MM-DD format.")
    vendor_id: str | None = Field(default=None, description="New vendor ID.")
    category_id: str | None = Field(default=None, description="New expense category ID.")
    public_notes: str | None = Field(
        default=None,
        description="New description shown to clients if the expense is invoiced.",
    )
    private_notes: str | None = Field(default=None, description="New internal-only notes.")


class UpdateExpenseTool(Tool):
    """Update an existing expense in InvoiceNinja."""

    name = "update_expense"
    description = "Update an existing expense in InvoiceNinja. Only provided fields are changed."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expense_id": {"type": "string", "description": "InvoiceNinja expense ID."},
            "amount": {"type": "number", "description": "New expense amount."},
            "date": {"type": "string", "description": "New date in YYYY-MM-DD format."},
            "vendor_id": {"type": "string", "description": "New vendor ID."},
            "category_id": {"type": "string", "description": "New expense category ID."},
            "public_notes": {
                "type": "string",
                "description": "New description shown to clients if the expense is invoiced.",
            },
            "private_notes": {"type": "string", "description": "New internal-only notes."},
        },
        "required": ["expense_id"],
    }
    args_model = UpdateExpenseArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Updating InvoiceNinja expense"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Update an expense."""
        args = UpdateExpenseArgs(**kwargs)
        expense = await self._client.update_expense(
            expense_id=args.expense_id,
            amount=args.amount,
            date=args.date,
            vendor_id=args.vendor_id,
            category_id=args.category_id,
            public_notes=args.public_notes,
            private_notes=args.private_notes,
        )
        return ToolResult(message=f"Updated expense: {expense}")
