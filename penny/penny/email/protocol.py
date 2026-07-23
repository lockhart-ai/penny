"""Protocol definition for email clients."""

from __future__ import annotations

from typing import Protocol

from penny.email.models import EmailDetail, EmailSummary


class EmailClient(Protocol):
    """Protocol for email client implementations.

    Both JmapClient (Fastmail) and ZohoClient implement this interface,
    allowing tools to work with either provider.
    """

    async def search_emails(
        self,
        text: str | None = None,
        from_addr: str | None = None,
        subject: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> list[EmailSummary]:
        """Search emails and return summaries."""
        ...

    async def read_emails(self, email_ids: list[str]) -> list[EmailDetail]:
        """Fetch full email bodies by IDs."""
        ...

    async def close(self) -> None:
        """Close the client."""
        ...
