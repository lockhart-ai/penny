"""InvoiceNinja API client."""

from __future__ import annotations

import logging

import httpx

from penny.plugins.invoiceninja.models import Invoice

logger = logging.getLogger(__name__)


class InvoiceNinjaClient:
    """InvoiceNinja v5 API client.

    Requires an API token from Settings → API Tokens in InvoiceNinja.
    Set INVOICENINJA_API_TOKEN and INVOICENINJA_URL in your .env.
    """

    def __init__(self, api_token: str, base_url: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "X-Api-Token": api_token,
                "Content-Type": "application/json",
            },
        )

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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
