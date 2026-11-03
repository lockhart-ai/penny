"""InvoiceNinja tools — LLM-callable tools for InvoiceNinja."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import Field

from penny.llm.call_link import CallLink
from penny.tools.base import PLAN_ARG_DESCRIPTION, Tool
from penny.tools.models import ToolArgs, ToolResult
from penny.tools.plan_tool import ResolvingMutationTool

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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
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
    # The structural plan mark (#2947, following #2933) — every ``mutates=True``
    # tool declares this literal field itself (``Tool.__init_subclass__`` asserts
    # it's here, rather than synthesising it, so a type checker can see it on
    # ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CreateExpenseTool(ResolvingMutationTool[CreateExpenseArgs]):
    """Create an expense in InvoiceNinja."""

    name = "create_expense"
    mutates = True
    description = (
        "Create a new expense in InvoiceNinja.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
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
    failure_message = CREATE_EXPENSE_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating InvoiceNinja expense"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def resolve(self, args: CreateExpenseArgs) -> CreateExpenseArgs:
        """Nothing to look up — vendor/category are opaque ids, not names."""
        return args

    def describe(self, resolved: CreateExpenseArgs) -> ToolResult:
        lines = [
            f"create a ${resolved.amount:.2f} expense dated {resolved.date} "
            f"(vendor {resolved.vendor_id}, category {resolved.category_id})."
        ]
        if resolved.public_notes:
            lines.append(f"Public notes: {resolved.public_notes}")
        if resolved.private_notes:
            lines.append(f"Private notes: {resolved.private_notes}")
        return ToolResult(message="\n".join(lines))

    async def apply(self, resolved: CreateExpenseArgs) -> ToolResult:
        """The one mutating call — create the expense."""
        expense = await self._client.create_expense(
            amount=resolved.amount,
            date=resolved.date,
            vendor_id=resolved.vendor_id,
            category_id=resolved.category_id,
            public_notes=resolved.public_notes,
            private_notes=resolved.private_notes,
        )
        return ToolResult(message=f"Created expense: {expense}", mutated=True)


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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
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
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class UpdateExpenseTool(ResolvingMutationTool[UpdateExpenseArgs]):
    """Update an existing expense in InvoiceNinja."""

    name = "update_expense"
    mutates = True
    description = (
        "Update an existing expense in InvoiceNinja. Only provided fields are changed.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
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
    failure_message = UPDATE_EXPENSE_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Updating InvoiceNinja expense"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def resolve(self, args: UpdateExpenseArgs) -> UpdateExpenseArgs:
        """Nothing to look up — ``expense_id`` is given directly."""
        return args

    def describe(self, resolved: UpdateExpenseArgs) -> ToolResult:
        changes = []
        if resolved.amount is not None:
            changes.append(f"amount -> ${resolved.amount:.2f}")
        if resolved.date is not None:
            changes.append(f"date -> {resolved.date}")
        if resolved.vendor_id is not None:
            changes.append(f"vendor -> {resolved.vendor_id}")
        if resolved.category_id is not None:
            changes.append(f"category -> {resolved.category_id}")
        if resolved.public_notes is not None:
            changes.append(f"public notes -> {resolved.public_notes}")
        if resolved.private_notes is not None:
            changes.append(f"private notes -> {resolved.private_notes}")
        change_str = ", ".join(changes) if changes else "no changes"
        return ToolResult(message=f"update expense {resolved.expense_id}: {change_str}.")

    async def apply(self, resolved: UpdateExpenseArgs) -> ToolResult:
        """The one mutating call — update the expense."""
        expense = await self._client.update_expense(
            expense_id=resolved.expense_id,
            amount=resolved.amount,
            date=resolved.date,
            vendor_id=resolved.vendor_id,
            category_id=resolved.category_id,
            public_notes=resolved.public_notes,
            private_notes=resolved.private_notes,
        )
        return ToolResult(message=f"Updated expense: {expense}", mutated=True)


class CreateExpenseCategoryArgs(ToolArgs):
    """Arguments for creating an expense category."""

    name: str = Field(description="Name of the new expense category.")
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CreateExpenseCategoryTool(ResolvingMutationTool[CreateExpenseCategoryArgs]):
    """Create an expense category in InvoiceNinja."""

    name = "create_expense_category"
    mutates = True
    description = (
        "Create a new expense category in InvoiceNinja.\n"
        "\n"
        "Pass `plan=true` to see what this call would do without doing it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the new expense category."},
        },
        "required": ["name"],
    }
    args_model = CreateExpenseCategoryArgs
    failure_message = CREATE_EXPENSE_CATEGORY_FAILED

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating InvoiceNinja expense category"

    def __init__(self, client: Any) -> None:
        self._client = client

    async def resolve(self, args: CreateExpenseCategoryArgs) -> CreateExpenseCategoryArgs:
        """Nothing to look up — the whole call is the args themselves."""
        return args

    def describe(self, resolved: CreateExpenseCategoryArgs) -> ToolResult:
        return ToolResult(message=f"create expense category '{resolved.name}'.")

    async def apply(self, resolved: CreateExpenseCategoryArgs) -> ToolResult:
        """The one mutating call — create the expense category."""
        category = await self._client.create_expense_category(resolved.name)
        return ToolResult(message=f"Created expense category: {category}", mutated=True)


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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
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

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Get an expense category."""
        args = GetExpenseCategoryArgs(**kwargs)
        try:
            category = await self._client.get_expense_category(args.category_id)
        except httpx.HTTPError as e:
            logger.exception("Fetching expense category failed")
            return ToolResult(message=GET_EXPENSE_CATEGORY_FAILED.format(e=e), success=False)
        return ToolResult(message=str(category))
