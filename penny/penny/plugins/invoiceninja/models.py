"""Pydantic models for InvoiceNinja API data."""

from __future__ import annotations

from pydantic import BaseModel


class InvoiceNinjaCredentials(BaseModel):
    """InvoiceNinja API credentials."""

    api_token: str
    base_url: str


class Invoice(BaseModel):
    """An InvoiceNinja invoice."""

    id: str
    number: str
    client_name: str
    amount: float
    status: str
    due_date: str | None = None
    created_at: str | None = None

    def __str__(self) -> str:
        due = f" (due {self.due_date})" if self.due_date else ""
        return f"[{self.number}] {self.client_name} — ${self.amount:.2f} [{self.status}]{due}"
