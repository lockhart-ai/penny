"""Pydantic models for InvoiceNinja API data."""

from __future__ import annotations

from pydantic import BaseModel


class Expense(BaseModel):
    """An InvoiceNinja expense."""

    id: str
    amount: float
    date: str
    status: str
    public_notes: str | None = None
    private_notes: str | None = None
    vendor_id: str | None = None
    category_id: str | None = None

    def __str__(self) -> str:
        notes = f" — {self.public_notes}" if self.public_notes else ""
        return f"[{self.id}] {self.date} ${self.amount:.2f} [{self.status}]{notes}"


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
