"""InvoiceNinja API client."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from penny.plugins.invoiceninja.models import Expense, Invoice

logger = logging.getLogger(__name__)


def _expense_from_data(data: dict[str, Any]) -> Expense:
    """Build an Expense from an InvoiceNinja API data object."""
    return Expense(
        id=str(data.get("id", "")),
        amount=float(data.get("amount", 0) or 0),
        date=data.get("date", ""),
        status=data.get("status", "unknown"),
        public_notes=data.get("public_notes"),
        private_notes=data.get("private_notes"),
        vendor_id=data.get("vendor_id"),
        category_id=data.get("category_id"),
    )


class InvoiceNinjaClient:
    """InvoiceNinja v5 API client.

    Authenticates with an API token (Settings → Account Management →
    Integrations → API Tokens) using the ``X-API-TOKEN`` header.
    InvoiceNinja v5 also requires the ``X-Requested-With: XMLHttpRequest``
    header on all API requests.

    Set INVOICENINJA_API_TOKEN and INVOICENINJA_URL in your .env.
    """

    def __init__(self, api_token: str, base_url: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "X-API-TOKEN": api_token,
                "X-Requested-With": "XMLHttpRequest",
                "Content-Type": "application/json",
            },
        )

    async def verify_auth(self) -> bool:
        """Verify that the API token is valid.

        Hits the authenticated ``/api/v1/health_check`` endpoint. Returns
        ``True`` on success and raises ``httpx.HTTPStatusError`` on auth or
        connectivity failures.
        """
        url = f"{self._base_url}/api/v1/health_check"
        resp = await self._http.get(url)
        resp.raise_for_status()
        logger.info("InvoiceNinja authentication verified for %s", self._base_url)
        return True

    async def list_invoices(self, status: str | None = None) -> list[Invoice]:
        """List invoices from InvoiceNinja.

        Args:
            status: Optional status filter (e.g., 'draft', 'sent', 'paid', 'overdue').

        Returns:
            List of Invoice objects.
        """
        url = f"{self._base_url}/api/v1/invoices"
        params: dict[str, str] = {}
        if status:
            params["status"] = status

        resp = await self._http.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        invoices: list[Invoice] = []
        for invoice_data in data.get("data", []):
            invoices.append(
                Invoice(
                    id=str(invoice_data.get("id", "")),
                    number=invoice_data.get("number", ""),
                    client_name=invoice_data.get("client", {}).get("name", "Unknown"),
                    amount=float(invoice_data.get("amount", 0) or 0),
                    status=invoice_data.get("status", "unknown"),
                    due_date=invoice_data.get("due_date"),
                    created_at=invoice_data.get("created_at"),
                )
            )

        logger.info("Listed %d invoices from InvoiceNinja", len(invoices))
        return invoices

    async def create_expense(
        self,
        amount: float,
        date: str,
        vendor_id: str,
        category_id: str,
        *,
        public_notes: str | None = None,
        private_notes: str | None = None,
    ) -> Expense:
        """Create a new expense in InvoiceNinja.

        Args:
            amount: Expense amount.
            date: Expense date in YYYY-MM-DD format.
            vendor_id: InvoiceNinja vendor ID.
            category_id: InvoiceNinja expense category ID.
            public_notes: Optional description shown to clients if invoiced.
            private_notes: Optional internal notes.

        Returns:
            The created Expense object.
        """
        url = f"{self._base_url}/api/v1/expenses"
        payload: dict[str, Any] = {"date": date, "amount": amount}
        if vendor_id:
            payload["vendor_id"] = vendor_id
        if category_id:
            payload["category_id"] = category_id
        if public_notes is not None:
            payload["public_notes"] = public_notes
        if private_notes is not None:
            payload["private_notes"] = private_notes

        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        expense_data = resp.json().get("data", {})
        logger.info("Created InvoiceNinja expense %s", expense_data.get("id"))
        return _expense_from_data(expense_data)

    async def list_expenses(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Expense]:
        """List expenses from InvoiceNinja.

        Args:
            status: Optional status filter.
            limit: Maximum number of expenses to return.

        Returns:
            List of Expense objects.
        """
        url = f"{self._base_url}/api/v1/expenses"
        params: dict[str, str] = {"per_page": str(limit)}
        if status:
            params["status"] = status

        resp = await self._http.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        expenses: list[Expense] = []
        for expense_data in data.get("data", []):
            expenses.append(_expense_from_data(expense_data))

        logger.info("Listed %d expenses from InvoiceNinja", len(expenses))
        return expenses

    async def get_expense(self, expense_id: str) -> Expense:
        """Fetch a single expense by ID.

        Args:
            expense_id: InvoiceNinja expense ID.

        Returns:
            The Expense object.
        """
        url = f"{self._base_url}/api/v1/expenses/{expense_id}"
        resp = await self._http.get(url)
        resp.raise_for_status()
        expense_data = resp.json().get("data", {})
        logger.info("Retrieved InvoiceNinja expense %s", expense_id)
        return _expense_from_data(expense_data)

    async def update_expense(
        self,
        expense_id: str,
        *,
        amount: float | None = None,
        date: str | None = None,
        vendor_id: str | None = None,
        category_id: str | None = None,
        public_notes: str | None = None,
        private_notes: str | None = None,
    ) -> Expense:
        """Update an existing expense.

        Only fields that are explicitly provided are sent to the API.

        Args:
            expense_id: InvoiceNinja expense ID.
            amount: Optional new amount.
            date: Optional new date (YYYY-MM-DD).
            vendor_id: Optional new vendor ID.
            category_id: Optional new category ID.
            public_notes: Optional new public notes.
            private_notes: Optional new private notes.

        Returns:
            The updated Expense object.
        """
        url = f"{self._base_url}/api/v1/expenses/{expense_id}"
        payload: dict[str, Any] = {}
        if amount is not None:
            payload["amount"] = amount
        if date is not None:
            payload["date"] = date
        if vendor_id is not None:
            payload["vendor_id"] = vendor_id
        if category_id is not None:
            payload["category_id"] = category_id
        if public_notes is not None:
            payload["public_notes"] = public_notes
        if private_notes is not None:
            payload["private_notes"] = private_notes

        resp = await self._http.put(url, json=payload)
        resp.raise_for_status()
        expense_data = resp.json().get("data", {})
        logger.info("Updated InvoiceNinja expense %s", expense_id)
        return _expense_from_data(expense_data)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
