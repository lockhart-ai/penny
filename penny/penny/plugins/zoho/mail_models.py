"""Pydantic models for Zoho Mail API data."""

from __future__ import annotations

from pydantic import BaseModel


class ZohoCredentials(BaseModel):
    """Zoho OAuth credentials for API access."""

    client_id: str
    client_secret: str
    refresh_token: str


class ZohoSession(BaseModel):
    """Cached Zoho OAuth session data."""

    access_token: str
    expires_at: float  # Unix timestamp when token expires


class ZohoAccount(BaseModel):
    """Zoho Mail account information."""

    account_id: str
    email_address: str
    display_name: str | None = None


class ZohoFolder(BaseModel):
    """Zoho Mail folder information."""

    folder_id: str
    folder_name: str
    folder_type: str  # Inbox, Sent, Drafts, Trash, Spam, etc.
    path: str  # e.g., "/Inbox", "/Sent"
    is_archived: bool = False
