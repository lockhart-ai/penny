"""Zoho Calendar tools — LLM-callable tools for calendar management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import Field

from penny.tools.base import Tool
from penny.tools.models import ToolArgs, ToolResult

logger = logging.getLogger(__name__)


class ListCalendarsArgs(ToolArgs):
    """Arguments for listing calendars."""


class GetEventsArgs(ToolArgs):
    """Arguments for getting calendar events."""

    calendar_name: str | None = Field(
        default=None, description="Calendar name (uses 'Default' if not specified)"
    )
    days_ahead: int = Field(default=14, description="Number of days ahead to search")


class CheckAvailabilityArgs(ToolArgs):
    """Arguments for checking availability."""

    start_date: str = Field(description="Start date/time (ISO format or natural language)")
    end_date: str = Field(description="End date/time (ISO format or natural language)")
    attendees: list[str] | None = Field(
        default=None, description="Optional list of attendee emails to check"
    )


class CreateEventArgs(ToolArgs):
    """Arguments for creating a calendar event."""

    title: str = Field(description="Event title")
    start: str = Field(description="Start date/time (ISO format)")
    end: str = Field(description="End date/time (ISO format)")
    calendar_name: str | None = Field(
        default=None, description="Calendar name (uses 'Default' if not specified)"
    )
    description: str | None = Field(default=None, description="Event description")
    location: str | None = Field(default=None, description="Event location")
    attendees: list[str] | None = Field(
        default=None, description="List of attendee email addresses"
    )
    is_allday: bool = Field(default=False, description="Whether this is an all-day event")


class FindFreeSlotsArgs(ToolArgs):
    """Arguments for finding free time slots."""

    duration_minutes: int = Field(description="Required slot duration in minutes")
    days_ahead: int = Field(default=14, description="Number of days ahead to search")
    attendees: list[str] | None = Field(
        default=None, description="Optional attendee emails to consider"
    )


class UpdateEventArgs(ToolArgs):
    """Arguments for updating a calendar event."""

    event_title: str = Field(description="Title of the event to update (for searching)")
    calendar_name: str | None = Field(
        default=None, description="Calendar name where the event exists"
    )
    new_title: str | None = Field(default=None, description="New event title")
    new_start: str | None = Field(default=None, description="New start date/time (ISO format)")
    new_end: str | None = Field(default=None, description="New end date/time (ISO format)")
    new_description: str | None = Field(default=None, description="New event description")
    new_location: str | None = Field(default=None, description="New event location")
    recurrence_edittype: str = Field(
        default="all",
        description="""For recurring events: 'all' (all occurrences),
        'following' (this and future), 'only' (just this one)""",
    )


def _parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO datetime string, tolerating a trailing 'Z'."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ListCalendarsTool(Tool):
    """List available calendars."""

    name = "list_calendars"
    description = (
        "List all available calendars in the user's Zoho Calendar account. "
        "Returns calendar names, colors, and timezones. Use this to discover "
        "what calendars exist before creating events or checking availability."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    args_model = ListCalendarsArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Listing calendars"

    def __init__(self, calendar_client: Any) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List all calendars."""
        calendars = await self._client.get_calendars()
        if not calendars:
            return ToolResult(message="No calendars found.")

        lines = [f"Found {len(calendars)} calendar(s):\n"]
        for cal in calendars:
            default_marker = " (default)" if cal.is_default else ""
            lines.append(f"- **{cal.name}**{default_marker}")
            if cal.timezone:
                lines.append(f"  Timezone: {cal.timezone}")
        return ToolResult(message="\n".join(lines))


class GetEventsTool(Tool):
    """Get upcoming calendar events."""

    name = "get_events"
    description = (
        "Get upcoming events from a calendar. Returns event titles, times, "
        "locations, and attendees. Use this to check what's scheduled or "
        "to find conflicts before scheduling new events."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "calendar_name": {
                "type": "string",
                "description": (
                    "Calendar name to get events from. Uses 'Default' if not specified. "
                    "Examples: 'Default', 'Studio A', 'Personal'"
                ),
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days ahead to search (default: 14)",
            },
        },
        "required": [],
    }
    args_model = GetEventsArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Getting calendar events"

    def __init__(self, calendar_client: Any) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Get events from a calendar."""
        args = GetEventsArgs(**kwargs)

        if args.calendar_name:
            calendar = await self._client.get_calendar_by_name(args.calendar_name)
            if not calendar:
                return ToolResult(message=f"Calendar not found: {args.calendar_name}")
        else:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return ToolResult(message="No default calendar found.")

        start = datetime.now(UTC)
        end = start + timedelta(days=args.days_ahead)

        events = await self._client.get_events(calendar.caluid, start, end)
        if not events:
            return ToolResult(
                message=f"No events found in '{calendar.name}' for the next {args.days_ahead} days."
            )

        lines = [f"Found {len(events)} event(s) in '{calendar.name}':\n"]
        for evt in events:
            start_str = evt.start.strftime("%Y-%m-%d %H:%M") if evt.start else "Unknown"
            end_str = evt.end.strftime("%H:%M") if evt.end else ""
            time_str = f"{start_str} - {end_str}" if end_str else start_str

            lines.append(f"- **{evt.title}**")
            lines.append(f"  Time: {time_str}")
            if evt.location:
                lines.append(f"  Location: {evt.location}")
            if evt.attendees:
                lines.append(f"  Attendees: {', '.join(evt.attendees)}")
            lines.append("")

        return ToolResult(message="\n".join(lines))


class CheckAvailabilityTool(Tool):
    """Check calendar availability for a time range."""

    name = "check_availability"
    description = (
        "Check if a time slot is available on the calendar. Returns busy times "
        "within the specified range. Use this before scheduling meetings to "
        "ensure there are no conflicts."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date/time in ISO format (e.g., '2024-12-15T10:00:00')",
            },
            "end_date": {
                "type": "string",
                "description": "End date/time in ISO format (e.g., '2024-12-15T11:00:00')",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of attendee emails to check availability for",
            },
        },
        "required": ["start_date", "end_date"],
    }
    args_model = CheckAvailabilityArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Checking availability"

    def __init__(self, calendar_client: Any) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Check availability for a time range."""
        args = CheckAvailabilityArgs(**kwargs)

        try:
            start = _parse_iso_datetime(args.start_date)
            end = _parse_iso_datetime(args.end_date)
        except ValueError as e:
            return ToolResult(message=f"Invalid date format: {e}", success=False)

        if not args.attendees:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return ToolResult(message="No default calendar found to check availability.")

            events = await self._client.get_events(calendar.caluid, start, end)
            if not events:
                time_range = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
                return ToolResult(
                    message=f"The time slot {time_range} appears to be **available** "
                    "(no events found)."
                )

            conflicts = []
            for evt in events:
                if evt.start and evt.end and evt.start < end and evt.end > start:
                    conflicts.append(evt)

            if not conflicts:
                time_range = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
                return ToolResult(message=f"The time slot {time_range} is **available**.")

            lines = ["The requested time has conflicts:\n"]
            for evt in conflicts:
                evt_time = (
                    f"{evt.start.strftime('%H:%M')}-{evt.end.strftime('%H:%M')}"
                    if evt.start and evt.end
                    else "unknown time"
                )
                lines.append(f"- **{evt.title}** ({evt_time})")
            return ToolResult(message="\n".join(lines))

        busy_slots = await self._client.check_availability(start, end, args.attendees)

        if not busy_slots:
            time_range = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
            return ToolResult(message=f"The time slot {time_range} is **available**.")

        lines = ["The requested time has conflicts:\n"]
        for slot in busy_slots:
            slot_start = slot.get("start", "")
            slot_end = slot.get("end", "")
            lines.append(f"- Busy: {slot_start} to {slot_end}")

        return ToolResult(message="\n".join(lines))


class CreateEventTool(Tool):
    """Create a new calendar event."""

    name = "create_event"
    description = (
        "Create a new event on a calendar. Specify the title, start/end times, "
        "and optionally a description, location, and attendees. "
        "Use check_availability first to ensure there are no conflicts."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title",
            },
            "start": {
                "type": "string",
                "description": "Start date/time in ISO format (e.g., '2024-12-15T10:00:00')",
            },
            "end": {
                "type": "string",
                "description": "End date/time in ISO format (e.g., '2024-12-15T11:00:00')",
            },
            "calendar_name": {
                "type": "string",
                "description": (
                    "Calendar name to create event on. Uses 'Default' if not specified. "
                    "Examples: 'Default', 'Studio A'"
                ),
            },
            "description": {
                "type": "string",
                "description": "Optional event description",
            },
            "location": {
                "type": "string",
                "description": "Optional event location",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of attendee email addresses",
            },
            "is_allday": {
                "type": "boolean",
                "description": "Whether this is an all-day event (default: false)",
            },
        },
        "required": ["title", "start", "end"],
    }
    args_model = CreateEventArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Creating calendar event"

    def __init__(self, calendar_client: Any) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create a calendar event."""
        args = CreateEventArgs(**kwargs)

        if args.calendar_name:
            calendar = await self._client.get_calendar_by_name(args.calendar_name)
            if not calendar:
                return ToolResult(
                    message=f"Calendar not found: {args.calendar_name}. "
                    "Check the calendar name and try again.",
                    success=False,
                )
        else:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return ToolResult(message="No default calendar found.", success=False)

        try:
            start = _parse_iso_datetime(args.start)
            end = _parse_iso_datetime(args.end)
        except ValueError as e:
            return ToolResult(message=f"Invalid date format: {e}", success=False)

        event = await self._client.create_event(
            caluid=calendar.caluid,
            title=args.title,
            start=start,
            end=end,
            description=args.description,
            location=args.location,
            attendees=args.attendees,
            is_allday=args.is_allday,
        )

        if event:
            time_str = f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%H:%M')}"
            result = [
                f"Event created successfully on '{calendar.name}':\n",
                f"**{event.title}**",
                f"Time: {time_str}",
            ]
            if event.location:
                result.append(f"Location: {event.location}")
            if event.attendees:
                result.append(f"Attendees: {', '.join(event.attendees)}")
            return ToolResult(message="\n".join(result), mutated=True)

        return ToolResult(message="Failed to create event.", success=False)


class FindFreeSlotsTool(Tool):
    """Find available time slots for meetings."""

    name = "find_free_slots"
    description = (
        "Find available time slots of a specified duration within the next N days. "
        "Use this to suggest meeting times when scheduling appointments. "
        "Returns a list of free time slots that can accommodate the requested duration."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "duration_minutes": {
                "type": "integer",
                "description": "Required slot duration in minutes (e.g., 30, 60)",
            },
            "days_ahead": {
                "type": "integer",
                "description": "Number of days ahead to search (default: 14)",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional attendee emails to consider for availability",
            },
        },
        "required": ["duration_minutes"],
    }
    args_model = FindFreeSlotsArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Finding free time slots"

    def __init__(self, calendar_client: Any) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Find free time slots."""
        args = FindFreeSlotsArgs(**kwargs)

        start = datetime.now(UTC)
        end = start + timedelta(days=args.days_ahead)

        free_slots = await self._client.find_free_slots(
            duration_minutes=args.duration_minutes,
            start=start,
            end=end,
            attendees=args.attendees,
        )

        if not free_slots:
            return ToolResult(
                message=f"No free slots of {args.duration_minutes} minutes found "
                f"in the next {args.days_ahead} days."
            )

        lines = [f"Found {len(free_slots)} available slot(s) of {args.duration_minutes} minutes:\n"]
        for slot in free_slots:
            slot_start = slot["start"]
            slot_end = slot["end"]
            lines.append(
                f"- {slot_start.strftime('%Y-%m-%d %H:%M')} to {slot_end.strftime('%H:%M')}"
            )

        return ToolResult(message="\n".join(lines))


class UpdateEventTool(Tool):
    """Update an existing calendar event."""

    name = "update_event"
    description = (
        "Update an existing calendar event. Can change the title, time, description, "
        "or location. For recurring events, you can update all occurrences, just this "
        "one, or this and all future occurrences. First searches for the event by title, "
        "then updates it with the new values."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_title": {
                "type": "string",
                "description": "Title of the event to update (used to find the event)",
            },
            "calendar_name": {
                "type": "string",
                "description": "Calendar name where the event exists (optional)",
            },
            "new_title": {
                "type": "string",
                "description": "New title for the event (optional)",
            },
            "new_start": {
                "type": "string",
                "description": "New start date/time in ISO format, e.g., '2026-04-17T15:00:00'",
            },
            "new_end": {
                "type": "string",
                "description": "New end date/time in ISO format, e.g., '2026-04-17T16:00:00'",
            },
            "new_description": {
                "type": "string",
                "description": "New description for the event (optional)",
            },
            "new_location": {
                "type": "string",
                "description": "New location for the event (optional)",
            },
            "recurrence_edittype": {
                "type": "string",
                "enum": ["all", "following", "only"],
                "description": (
                    "For recurring events: 'all' updates all occurrences, "
                    "'following' updates this and future occurrences, "
                    "'only' updates just this occurrence. Default: 'all'"
                ),
            },
        },
        "required": ["event_title"],
    }
    args_model = UpdateEventArgs

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return f"Updating event: {arguments.get('event_title', 'unknown')}"

    def __init__(self, calendar_client: Any) -> None:
        self._client = calendar_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Update a calendar event."""
        args = UpdateEventArgs(**kwargs)

        if args.calendar_name:
            calendar = await self._client.get_calendar_by_name(args.calendar_name)
            if not calendar:
                return ToolResult(
                    message=f"Calendar not found: {args.calendar_name}. "
                    "Check the calendar name and try again.",
                    success=False,
                )
        else:
            calendar = await self._client.get_default_calendar()
            if not calendar:
                return ToolResult(message="No default calendar found.", success=False)

        start = datetime.now(UTC)
        end = start + timedelta(days=30)

        events = await self._client.get_events(calendar.caluid, start, end)
        matching_events = [e for e in events if args.event_title.lower() in e.title.lower()]

        if not matching_events:
            return ToolResult(
                message=f"No event found matching '{args.event_title}' in calendar "
                f"'{calendar.name}'. Please check the event title and try again.",
                success=False,
            )

        event = matching_events[0]
        logger.info(
            "Found event from search: uid=%s, title=%s, start=%s",
            event.uid,
            event.title,
            event.start,
        )

        full_event = await self._client.get_event(calendar.caluid, event.uid)
        if not full_event or not full_event.etag:
            return ToolResult(
                message=f"Could not retrieve event details for '{event.title}'.", success=False
            )

        logger.info(
            "Full event details: uid=%s, start=%s, recurrenceid=%s, is_recurring=%s",
            full_event.uid,
            full_event.start,
            full_event.recurrenceid,
            full_event.is_recurring,
        )

        if args.new_start:
            try:
                new_start = _parse_iso_datetime(args.new_start)
            except ValueError as e:
                return ToolResult(message=f"Invalid start date format: {e}", success=False)
        else:
            new_start = full_event.start
            if not new_start:
                return ToolResult(
                    message=f"Could not determine start time for event '{event.title}'.",
                    success=False,
                )

        if args.new_end:
            try:
                new_end = _parse_iso_datetime(args.new_end)
            except ValueError as e:
                return ToolResult(message=f"Invalid end date format: {e}", success=False)
        else:
            new_end = full_event.end
            if not new_end:
                return ToolResult(
                    message=f"Could not determine end time for event '{event.title}'.",
                    success=False,
                )

        effective_edittype = args.recurrence_edittype

        if (
            full_event.is_recurring
            and effective_edittype == "all"
            and (args.new_start or args.new_end)
        ):
            logger.info("Switching from 'all' to 'following' for recurring event time change")
            effective_edittype = "following"

        recurrenceid = None
        if full_event.is_recurring and effective_edittype in ("following", "only"):
            occurrence_start = event.start
            if occurrence_start:
                start_utc = occurrence_start.astimezone(UTC)
                recurrenceid = start_utc.strftime("%Y%m%dT%H%M%SZ")
                logger.info(
                    "Generated recurrenceid from search occurrence: %s (from %s)",
                    recurrenceid,
                    occurrence_start,
                )

        if full_event.is_recurring:
            logger.info(
                "Recurring event detected: recurrenceid=%s, edittype=%s",
                recurrenceid,
                effective_edittype,
            )

        updated_event = await self._client.update_event(
            caluid=calendar.caluid,
            event_uid=event.uid,
            etag=full_event.etag,
            title=args.new_title,
            start=new_start,
            end=new_end,
            tz=full_event.timezone,
            description=args.new_description,
            location=args.new_location,
            is_allday=full_event.is_allday,
            recurrence_edittype=effective_edittype,
            recurrenceid=recurrenceid,
            rrule=full_event.rrule,
        )

        if updated_event:
            changes = []
            if args.new_title:
                changes.append(f"Title: {args.new_title}")
            if new_start and new_end:
                changes.append(
                    f"Time: {new_start.strftime('%Y-%m-%d %H:%M')} to {new_end.strftime('%H:%M')}"
                )
            if args.new_location:
                changes.append(f"Location: {args.new_location}")
            if args.new_description:
                changes.append("Description updated")

            edit_scope = {
                "all": "all occurrences",
                "following": "this and future occurrences",
                "only": "only this occurrence",
            }.get(args.recurrence_edittype, "all occurrences")

            result = [
                f"Event updated successfully ({edit_scope}):\n",
                f"**{updated_event.title}**",
            ]
            if changes:
                result.append("Changes:")
                result.extend([f"  - {c}" for c in changes])
            return ToolResult(message="\n".join(result), mutated=True)

        return ToolResult(message="Failed to update event.", success=False)


def calendar_tools(calendar_client: Any) -> list[Tool]:
    """Return all Zoho Calendar tools bound to the given client."""
    return [
        ListCalendarsTool(calendar_client),
        GetEventsTool(calendar_client),
        CheckAvailabilityTool(calendar_client),
        CreateEventTool(calendar_client),
        FindFreeSlotsTool(calendar_client),
        UpdateEventTool(calendar_client),
    ]
