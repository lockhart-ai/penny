"""Zoho Calendar API client."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from penny.constants import PennyConstants
from penny.plugins.zoho.base_client import ZohoOAuthClient
from penny.plugins.zoho.models import BusySlot, FreeSlot, ZohoCalendarInfo, ZohoEvent

logger = logging.getLogger(__name__)


class ZohoCalendarClient(ZohoOAuthClient):
    """Zoho Calendar API client.

    Uses OAuth 2.0 with client credentials to access Zoho Calendar API.
    The OAuth refresh, headers, and HTTP client lifecycle live on
    ``ZohoOAuthClient``; this class adds the calendar-specific endpoints.
    """

    service_label = "Zoho Calendar"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        *,
        timeout: float = PennyConstants.ZOHO_CLIENT_TIMEOUT,
    ) -> None:
        super().__init__(client_id, client_secret, refresh_token, timeout=timeout)
        self._calendars_cache: list[ZohoCalendarInfo] | None = None

    async def get_calendars(self, force_refresh: bool = False) -> list[ZohoCalendarInfo]:
        """Fetch all calendars for the user."""
        if self._calendars_cache and not force_refresh:
            return self._calendars_cache

        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/calendars"

        resp = await self._http.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        calendars_data = data.get("calendars", [])
        self._calendars_cache = [
            ZohoCalendarInfo(
                caluid=cal.get("uid", ""),  # API returns "uid", not "caluid"
                name=cal.get("name", ""),
                color=cal.get("color"),
                timezone=cal.get("timezone"),
                is_default=cal.get("isdefault", False),
            )
            for cal in calendars_data
        ]
        logger.info("Loaded %d calendars", len(self._calendars_cache))
        return self._calendars_cache

    async def get_calendar_by_name(self, name: str) -> ZohoCalendarInfo | None:
        """Get a calendar by name (case-insensitive, fuzzy match)."""
        calendars = await self.get_calendars()
        name_lower = name.lower().strip()

        for cal in calendars:
            if cal.name.lower() == name_lower:
                return cal

        for cal in calendars:
            if name_lower in cal.name.lower() or cal.name.lower() in name_lower:
                return cal

        return None

    async def get_default_calendar(self) -> ZohoCalendarInfo | None:
        """Get the default calendar (named 'Default'), or the first available."""
        calendars = await self.get_calendars()

        for cal in calendars:
            if cal.is_default or cal.name.lower() == "default":
                return cal

        return calendars[0] if calendars else None

    async def get_events(
        self,
        caluid: str,
        start: datetime,
        end: datetime,
    ) -> list[ZohoEvent]:
        """Get events from a calendar within a date range."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/calendars/{caluid}/events"

        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        params = {"range": json.dumps({"start": start_str, "end": end_str})}

        resp = await self._http.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            logger.error("Calendar events API error: %s - %s", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()

        events_data = data.get("events", [])
        events = []
        for evt in events_data:
            dateandtime = evt.get("dateandtime", {})
            events.append(
                ZohoEvent(
                    uid=evt.get("uid", ""),
                    title=evt.get("title", ""),
                    start=self._parse_zoho_datetime(dateandtime.get("start", "")),
                    end=self._parse_zoho_datetime(dateandtime.get("end", "")),
                    timezone=dateandtime.get("timezone"),
                    description=evt.get("description"),
                    location=evt.get("location"),
                    is_allday=evt.get("isallday", False),
                    attendees=[a.get("email", "") for a in evt.get("attendees", [])],
                )
            )

        logger.info("Loaded %d events from calendar %s", len(events), caluid)
        return events

    async def check_availability(
        self,
        start: datetime,
        end: datetime,
        attendees: list[str] | None = None,
    ) -> list[BusySlot]:
        """Check free/busy status for a time range."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/calendars/freebusy"

        start_str = start.strftime("%Y%m%dT%H%M%S")
        end_str = end.strftime("%Y%m%dT%H%M%S")

        if not attendees:
            logger.warning("check_availability requires at least one email address")
            return []

        params = {
            "uemail": attendees[0],
            "sdate": start_str,
            "edate": end_str,
            "ftype": "eventbased",
        }

        resp = await self._http.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            logger.error("Freebusy API error: %s - %s", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()

        freebusy_data = data.get("freebusy", {})
        busy_slots = []
        for slot in freebusy_data.get("busyslots", []):
            slot_start = self._parse_zoho_datetime(slot.get("start", ""))
            slot_end = self._parse_zoho_datetime(slot.get("end", ""))
            if slot_start and slot_end:
                busy_slots.append(BusySlot(start=slot_start, end=slot_end))
        logger.info("Found %d busy slots in range", len(busy_slots))
        return busy_slots

    async def find_free_slots(
        self,
        duration_minutes: int,
        start: datetime,
        end: datetime,
        attendees: list[str] | None = None,
    ) -> list[FreeSlot]:
        """Find available time slots of a given duration."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/freebusy/freeslots"

        start_str = start.strftime("%Y%m%dT%H%M%S")
        end_str = end.strftime("%Y%m%dT%H%M%S")

        params = {
            "sdate": start_str,
            "edate": end_str,
            "duration": str(duration_minutes),
            "mode": "freeslots",
        }

        if attendees:
            params["attendees"] = ",".join(attendees)

        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        free_slots = []
        for slot in data.get("freeslots", []):
            slot_start = self._parse_zoho_datetime(slot.get("start", ""))
            slot_end = self._parse_zoho_datetime(slot.get("end", ""))
            if slot_start and slot_end:
                free_slots.append(FreeSlot(start=slot_start, end=slot_end))

        logger.info("Found %d free slots of %d minutes", len(free_slots), duration_minutes)
        return free_slots

    async def create_event(
        self,
        caluid: str,
        title: str,
        start: datetime,
        end: datetime,
        *,
        description: str | None = None,
        location: str | None = None,
        attendees: list[str] | None = None,
        timezone: str = "UTC",
        is_allday: bool = False,
    ) -> ZohoEvent | None:
        """Create a new calendar event."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/calendars/{caluid}/events"

        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        if is_allday:
            start_str = start_utc.strftime("%Y%m%d")
            end_str = end_utc.strftime("%Y%m%d")
        else:
            start_str = start_utc.strftime("%Y%m%dT%H%M%SZ")
            end_str = end_utc.strftime("%Y%m%dT%H%M%SZ")

        eventdata: dict[str, Any] = {
            "title": title,
            "dateandtime": {
                "start": start_str,
                "end": end_str,
                "timezone": timezone,
            },
            "isallday": is_allday,
        }

        if description:
            eventdata["description"] = description
        if location:
            eventdata["location"] = location
        if attendees:
            eventdata["attendees"] = [{"email": email} for email in attendees]

        eventdata_json = json.dumps(eventdata)
        logger.info("Creating event '%s' on calendar %s", title, caluid)
        logger.debug("Create eventdata: %s", eventdata_json)
        resp = await self._http.post(url, headers=headers, params={"eventdata": eventdata_json})
        resp.raise_for_status()
        data = resp.json()

        event_data = data.get("events", [{}])[0] if data.get("events") else {}
        if event_data.get("uid"):
            self._calendars_cache = None
            return ZohoEvent(
                uid=event_data["uid"],
                title=title,
                start=start,
                end=end,
                timezone=timezone,
                description=description,
                location=location,
                is_allday=is_allday,
                attendees=attendees or [],
            )

        logger.warning("Event creation returned no uid: %s", data)
        return None

    async def update_event(
        self,
        caluid: str,
        event_uid: str,
        etag: int,
        *,
        title: str | None = None,
        start: datetime,
        end: datetime,
        tz: str | None = None,
        description: str | None = None,
        location: str | None = None,
        is_allday: bool = False,
        recurrence_edittype: str = "all",
        recurrenceid: str | None = None,
        rrule: str | None = None,
    ) -> ZohoEvent | None:
        """Update an existing event."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/calendars/{caluid}/events/{event_uid}"

        eventdata: dict[str, Any] = {"etag": etag}

        if title:
            eventdata["title"] = title

        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        if is_allday:
            start_str = start_utc.strftime("%Y%m%d")
            end_str = end_utc.strftime("%Y%m%d")
        else:
            start_str = start_utc.strftime("%Y%m%dT%H%M%SZ")
            end_str = end_utc.strftime("%Y%m%dT%H%M%SZ")

        eventdata["dateandtime"] = {"start": start_str, "end": end_str}
        if tz:
            eventdata["dateandtime"]["timezone"] = tz
        eventdata["isallday"] = is_allday

        if description is not None:
            eventdata["description"] = description

        if location is not None:
            eventdata["location"] = location

        eventdata["recurrence_edittype"] = recurrence_edittype

        if rrule:
            eventdata["isrep"] = True
            eventdata["rrule"] = rrule
            logger.info("Including isrep=True and rrule: %s", rrule)

        if recurrenceid and recurrence_edittype in ("following", "only"):
            eventdata["recurrenceid"] = recurrenceid
            logger.info(
                "Including recurrenceid: %s for edittype: %s",
                recurrenceid,
                recurrence_edittype,
            )

        eventdata_json = json.dumps(eventdata)
        logger.info(
            "Updating event %s on calendar %s (edittype=%s)",
            event_uid,
            caluid,
            recurrence_edittype,
        )
        logger.debug("Update eventdata: %s", eventdata_json)
        resp = await self._http.put(url, headers=headers, params={"eventdata": eventdata_json})
        if resp.status_code != 200:
            logger.error(
                "Event update failed: %s - %s (eventdata: %s)",
                resp.status_code,
                resp.text,
                eventdata_json,
            )
        resp.raise_for_status()
        data = resp.json()

        event_data = data.get("events", [{}])[0] if data.get("events") else {}
        if event_data.get("uid"):
            dateandtime = event_data.get("dateandtime", {})
            return ZohoEvent(
                uid=event_data["uid"],
                title=event_data.get("title", title or ""),
                start=self._parse_zoho_datetime(dateandtime.get("start", "")),
                end=self._parse_zoho_datetime(dateandtime.get("end", "")),
                timezone=dateandtime.get("timezone"),
                description=event_data.get("description"),
                location=event_data.get("location"),
                is_allday=event_data.get("isallday", False),
                attendees=[a.get("email", "") for a in event_data.get("attendees", [])],
            )

        logger.warning("Event update returned no uid: %s", data)
        return None

    async def get_event(self, caluid: str, event_uid: str) -> ZohoEvent | None:
        """Get a specific event by UID."""
        headers = await self._get_headers()
        url = f"{PennyConstants.ZOHO_CALENDAR_API_BASE}/calendars/{caluid}/events/{event_uid}"

        resp = await self._http.get(url, headers=headers)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        event_data = data.get("events", [{}])[0] if data.get("events") else {}
        if not event_data.get("uid"):
            return None

        dateandtime = event_data.get("dateandtime", {})
        rrule = event_data.get("rrule")
        repeat = event_data.get("repeat")
        recurrenceid = event_data.get("recurrenceid")
        is_recurring = (
            event_data.get("isrep", False) or bool(rrule) or bool(repeat) or bool(recurrenceid)
        )

        logger.info(
            "Event details: isrep=%s, rrule=%s, recurrenceid=%s, is_recurring=%s",
            event_data.get("isrep"),
            rrule,
            recurrenceid,
            is_recurring,
        )

        return ZohoEvent(
            uid=event_data["uid"],
            title=event_data.get("title", ""),
            start=self._parse_zoho_datetime(dateandtime.get("start", "")),
            end=self._parse_zoho_datetime(dateandtime.get("end", "")),
            timezone=dateandtime.get("timezone"),
            description=event_data.get("description"),
            location=event_data.get("location"),
            is_allday=event_data.get("isallday", False),
            attendees=[a.get("email", "") for a in event_data.get("attendees", [])],
            etag=event_data.get("etag"),
            is_recurring=is_recurring,
            recurrenceid=event_data.get("recurrenceid"),
            rrule=rrule,
        )

    def _parse_zoho_datetime(self, dt_str: str) -> datetime | None:
        """Parse Zoho datetime string to a datetime object.

        Handles:
        - yyyyMMddTHHmmssZ (UTC)
        - yyyyMMddTHHmmss+HHMM or yyyyMMddTHHmmss-HHMM (offset)
        - yyyyMMdd (all-day)
        """
        if not dt_str:
            return None

        try:
            if "T" in dt_str:
                if "+" in dt_str[8:] or "-" in dt_str[8:]:
                    for i, char in enumerate(dt_str[8:], 8):
                        if char in "+-":
                            base_dt = dt_str[:i]
                            offset_str = dt_str[i:]
                            offset_sign = 1 if offset_str[0] == "+" else -1
                            offset_hours = int(offset_str[1:3])
                            offset_mins = int(offset_str[3:5]) if len(offset_str) >= 5 else 0
                            offset_delta = timedelta(
                                hours=offset_sign * offset_hours,
                                minutes=offset_sign * offset_mins,
                            )
                            tz = timezone(offset_delta)
                            return datetime.strptime(base_dt, "%Y%m%dT%H%M%S").replace(tzinfo=tz)
                dt_str = dt_str.rstrip("Z")
                return datetime.strptime(dt_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
            return datetime.strptime(dt_str, "%Y%m%d").replace(tzinfo=UTC)
        except ValueError:
            logger.warning("Failed to parse datetime: %s", dt_str)
            return None
