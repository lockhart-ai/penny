"""Pydantic models for JMAP data."""

from __future__ import annotations

from pydantic import BaseModel


class JmapSession(BaseModel):
    """Cached JMAP session data."""

    api_url: str
    account_id: str
