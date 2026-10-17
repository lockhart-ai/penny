"""InvoiceNinja tools — LLM-callable tools for InvoiceNinja."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import Field

from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

logger = logging.getLogger(__name__)

# The account a client failure gets (#2369): what happened, with no order attached — the
# model reads the state and reasons from it, same as every other tool result. Catches the
# InvoiceNinja client's own failures — a non-2xx response or a transport error, both
# ``httpx.HTTPError`` (``InvoiceNinjaClient`` raises nothing else: every call ends in
# ``resp.raise_for_status()``, no bare ``RuntimeError`` anywhere in the client) — never a bug
# in this file's own logic, which still propagates to the executor's generic handler as today.
VERIFY_AUTH_FAILED = (
    "InvoiceNinja authentication couldn't be verified ({e}), so nothing was confirmed."
)
LIST_INVOICES_FAILED = "Invoices couldn't be listed ({e}), so nothing was read from InvoiceNinja."
CREATE_EXPENSE_FAILED = (
    "The expense couldn't be created ({e}), so nothing was added to InvoiceNinja."
)
LIST_EXPENSES_FAILED = "Expenses couldn't be listed ({e}), so nothing was read from InvoiceNinja."
GET_EXPENSE_FAILED = (
    "That expense couldn't be fetched ({e}), so nothing was read from InvoiceNinja."
)
UPDATE_EXPENSE_FAILED = (
    "The expense couldn't be updated ({e}), so nothing was changed in InvoiceNinja."
)
CREATE_EXPENSE_CATEGORY_FAILED = (
    "The expense category couldn't be created ({e}), so nothing was added to InvoiceNinja."
)
LIST_EXPENSE_CATEGORIES_FAILED = (
    "Expense categories couldn't be listed ({e}), so nothing was read from InvoiceNinja."
)
GET_EXPENSE_CATEGORY_FAILED = (
    "That expense category couldn't be fetched ({e}), so nothing was read from InvoiceNinja."
)


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
        try:
            await self._client.verify_auth()
        except httpx.HTTPError as e:
            logger.exception("InvoiceNinja auth verification failed")
            return ToolResult(message=VERIFY_AUTH_FAILED.format(e=e), success=False)
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
        try:
            invoices = await self._client.list_invoices(status=args.status)
        except httpx.HTTPError as e:
            logger.exception("Listing invoices failed")
            return ToolResult(message=LIST_INVOICES_FAILED.format(e=e), success=False)
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
        try:
            expense = await self._client.create_expense(
                amount=args.amount,
                date=args.date,
                vendor_id=args.vendor_id,
                category_id=args.category_id,
                public_notes=args.public_notes,
                private_notes=args.private_notes,
            )
        except httpx.HTTPError as e:
            logger.exception("Creating InvoiceNinja expense failed")
            return ToolResult(message=CREATE_EXPENSE_FAILED.format(e=e), success=False)
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
        try:
            expenses = await self._client.list_expenses(status=args.status, limit=args.limit)
        except httpx.HTTPError as e:
            logger.exception("Listing expenses failed")
            return ToolResult(message=LIST_EXPENSES_FAILED.format(e=e), success=False)
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
        try:
            expense = await self._client.get_expense(args.expense_id)
        except httpx.HTTPError as e:
            logger.exception("Fetching expense failed")
            return ToolResult(message=GET_EXPENSE_FAILED.format(e=e), success=False)
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
        try:
            expense = await self._client.update_expense(
                expense_id=args.expense_id,
                amount=args.amount,
                date=args.date,
                vendor_id=args.vendor_id,
                category_id=args.category_id,
                public_notes=args.public_notes,
                private_notes=args.private_notes,
            )
        except httpx.HTTPError as e:
            logger.exception("Updating InvoiceNinja expense failed")
            return ToolResult(message=UPDATE_EXPENSE_FAILED.format(e=e), success=False)
        return ToolResult(message=f"Updated expense: {expense}")


class CreateExpenseCategoryArgs(ToolArgs):
    """Arguments for creating an expense category."""

    name: str = Field(description="Name of the new expense category.")


class CreateExpenseCategoryTool(Tool):
    """Create an expense category in InvoiceNinja."""

    name = "create_expense_category"
    description = "Create a new expense category in InvoiceNinja."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the new expense category."},
        },
        "required": ["name"],
    }
    args_model = CreateExpenseCategoryArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating InvoiceNinja expense category"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create an expense category."""
        args = CreateExpenseCategoryArgs(**kwargs)
        try:
            category = await self._client.create_expense_category(args.name)
        except httpx.HTTPError as e:
            logger.exception("Creating InvoiceNinja expense category failed")
            return ToolResult(message=CREATE_EXPENSE_CATEGORY_FAILED.format(e=e), success=False)
        return ToolResult(message=f"Created expense category: {category}")


class ListExpenseCategoriesArgs(ToolArgs):
    """Arguments for listing expense categories."""

    limit: int = Field(default=50, description="Maximum number of categories to return.")


class ListExpenseCategoriesTool(Tool):
    """List expense categories from InvoiceNinja."""

    name = "list_expense_categories"
    description = "List expense categories from InvoiceNinja."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of categories to return.",
                "default": 50,
            },
        },
        "required": [],
    }
    args_model = ListExpenseCategoriesArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing expense categories"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List expense categories."""
        args = ListExpenseCategoriesArgs(**kwargs)
        try:
            categories = await self._client.list_expense_categories(limit=args.limit)
        except httpx.HTTPError as e:
            logger.exception("Listing expense categories failed")
            return ToolResult(message=LIST_EXPENSE_CATEGORIES_FAILED.format(e=e), success=False)
        if not categories:
            return ToolResult(message="No expense categories found.")

        lines = [f"Found {len(categories)} expense category(ies):\n"]
        for category in categories:
            lines.append(str(category))
        return ToolResult(message="\n".join(lines))


class GetExpenseCategoryArgs(ToolArgs):
    """Arguments for fetching a single expense category."""

    category_id: str = Field(description="InvoiceNinja expense category ID.")


class GetExpenseCategoryTool(Tool):
    """Fetch a single expense category from InvoiceNinja."""

    name = "get_expense_category"
    description = "Fetch a single expense category from InvoiceNinja by ID."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "category_id": {"type": "string", "description": "InvoiceNinja expense category ID."},
        },
        "required": ["category_id"],
    }
    args_model = GetExpenseCategoryArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Fetching expense category"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get an expense category."""
        args = GetExpenseCategoryArgs(**kwargs)
        try:
            category = await self._client.get_expense_category(args.category_id)
        except httpx.HTTPError as e:
            logger.exception("Fetching expense category failed")
            return ToolResult(message=GET_EXPENSE_CATEGORY_FAILED.format(e=e), success=False)
        return ToolResult(message=str(category))
