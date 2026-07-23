"""Pydantic models for Zoho Calendar and Projects API data."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ZohoSession(BaseModel):
    """Cached Zoho OAuth session data."""

    access_token: str
    expires_at: float  # Unix timestamp when token expires


class ZohoCalendarInfo(BaseModel):
    """Zoho Calendar metadata."""

    caluid: str
    name: str
    color: str | None = None
    timezone: str | None = None
    is_default: bool = False


class BusySlot(BaseModel):
    """A busy time range from a Zoho Calendar freebusy query."""

    start: datetime
    end: datetime


class FreeSlot(BaseModel):
    """An available time range from a Zoho Calendar freeslots query."""

    start: datetime
    end: datetime


class ZohoEvent(BaseModel):
    """Zoho Calendar event."""

    uid: str
    title: str
    start: datetime | None = None
    end: datetime | None = None
    timezone: str | None = None
    description: str | None = None
    location: str | None = None
    is_allday: bool = False
    attendees: list[str] = []
    etag: int | None = None  # Required for updates
    is_recurring: bool = False
    recurrenceid: str | None = None  # For identifying specific occurrence
    rrule: str | None = None  # Recurrence rule for recurring events


class ZohoPortal(BaseModel):
    """Zoho Projects portal information."""

    id: str
    name: str
    is_default: bool = False


class ZohoProject(BaseModel):
    """Zoho Projects project information."""

    id: str
    name: str
    status: str | None = None
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    owner_name: str | None = None


class ZohoTaskList(BaseModel):
    """Zoho Projects task list (milestone)."""

    id: str
    name: str
    status: str | None = None
    flag: str | None = None  # "internal" or "external"


class ZohoTask(BaseModel):
    """Zoho Projects task."""

    id: str
    name: str
    status: str | None = None
    priority: str | None = None  # "none", "low", "medium", "high"
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    completion_percentage: int = 0
    tasklist_id: str | None = None
    tasklist_name: str | None = None
    owners: list[str] = Field(default_factory=list)  # Owner names/ZPUIDs
