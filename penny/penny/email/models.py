"""Provider-agnostic Pydantic models for email data."""

from __future__ import annotations

from pydantic import BaseModel


class EmailAddress(BaseModel):
    """An email address with optional display name."""

    name: str | None = None
    email: str

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} <{self.email}>"
        return self.email


class EmailSummary(BaseModel):
    """Summary of an email returned by search."""

    id: str
    subject: str
    from_addresses: list[EmailAddress]
    received_at: str
    preview: str

    def __str__(self) -> str:
        from_str = ", ".join(str(a) for a in self.from_addresses)
        return f"[{self.id}] {self.received_at} from {from_str}: {self.subject}\n{self.preview}"


class EmailDetail(BaseModel):
    """Full email body returned by read."""

    id: str
    subject: str
    from_addresses: list[EmailAddress]
    to_addresses: list[EmailAddress]
    received_at: str
    text_body: str

    def __str__(self) -> str:
        from_str = ", ".join(str(a) for a in self.from_addresses)
        return (
            f"Subject: {self.subject}\n"
            f"From: {from_str}\n"
            f"Date: {self.received_at}\n\n"
            f"{self.text_body}"
        )
