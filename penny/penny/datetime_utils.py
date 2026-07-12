"""Utility functions for date/time operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from penny.constants import PennyConstants

if TYPE_CHECKING:
    from penny.database.database import Database

try:
    from geopy.geocoders import Nominatim
    from timezonefinder import TimezoneFinder

    HAS_GEO = True
except ImportError:
    Nominatim: Any = None
    TimezoneFinder: Any = None
    HAS_GEO = False

logger = logging.getLogger(__name__)


def format_log_timestamp(when: datetime) -> str:
    """Render a log/entry timestamp for the model — compact, absolute, UTC.

    Every timed, log-shaped response shown to the model (read-tool entries, the
    recall conversation block, collector run history) should render its
    timestamps through this one helper, so the model can compare them against the
    ``Current date and time: … UTC`` line in the system prompt and reason about
    *when* things happened.  Without a stamp the model mistakes the timing of
    past events.  Naive datetimes are treated as UTC (how they're stored)."""
    if when.tzinfo is not None:
        when = when.astimezone(UTC)
    return when.strftime("%Y-%m-%d %H:%M UTC")


_INTERVAL_UNITS: tuple[tuple[str, int], ...] = (
    ("w", 604800),
    ("d", 86400),
    ("h", 3600),
    ("m", 60),
)


def format_interval(seconds: int) -> str:
    """Render a cadence in seconds as a compact human unit — ``300`` → ``"5m"``,
    ``21600`` → ``"6h"``, ``604800`` → ``"1w"``.

    Used by the self-state header (#1555) to show a collector's cadence at
    rollup altitude.  Picks the largest whole unit the value divides into; a
    value that isn't a whole number of minutes falls back to bare seconds
    (``90`` → ``"90s"``).  These divisors are unit conversions, not invented
    caps — no data is dropped."""
    for unit, unit_seconds in _INTERVAL_UNITS:
        if seconds >= unit_seconds and seconds % unit_seconds == 0:
            return f"{seconds // unit_seconds}{unit}"
    return f"{seconds}s"


async def get_timezone(location: str) -> str | None:
    """
    Derive IANA timezone from natural language location.

    Args:
        location: Natural language location (e.g., "Toronto, Canada")

    Returns:
        IANA timezone string (e.g., "America/Toronto") or None if lookup failed
    """
    if not HAS_GEO:
        logger.error("Geopy/timezonefinder not available")
        return None

    try:
        # Geocode location to lat/lon
        geolocator = Nominatim(user_agent="penny_profile")  # type: ignore[misc]
        geo_result = geolocator.geocode(location)
        if not geo_result:
            logger.warning("Geocoding failed for location: %s", location)
            return None

        # Get timezone from lat/lon
        tf = TimezoneFinder()  # type: ignore[misc]
        timezone = tf.timezone_at(lat=geo_result.latitude, lng=geo_result.longitude)
        if not timezone:
            logger.warning(
                "Timezone lookup failed for location: %s (%f, %f)",
                location,
                geo_result.latitude,
                geo_result.longitude,
            )
            return None

        logger.debug("Resolved timezone for %s: %s", location, timezone)
        return timezone

    except Exception as e:
        logger.warning("Timezone derivation failed for %s: %s", location, e)
        return None


def current_datetime_line(db: Database) -> str:
    """The 'Current date and time: <stamp>' anchor line handed to the model.

    The single source of the dated clock.  The agent-loop envelope
    (``Agent._build_messages``) and every ad-hoc one-shot LLM flow — the
    ``/profile`` parse, the startup announcement, the email summarize — render
    through here so they all reason
    from the same wall clock, in the user's profile timezone (never a bare UTC
    ``now()``).  Falls back to UTC on a fresh install / unknown zone, exactly like
    the envelope.
    """
    stamp = datetime.now(user_timezone(db)).strftime(PennyConstants.CURRENT_DATETIME_FORMAT)
    return f"{PennyConstants.CURRENT_DATETIME_PREFIX}{stamp}"


def user_timezone(db: Database) -> tzinfo:
    """The primary user's IANA timezone for the current-date/time anchor.

    The profile advertises the user's timezone, so the clock the model reasons
    from must match it — otherwise Penny is told the wrong time-of-day always,
    and (for the hours around local midnight) the wrong calendar day.  Falls back
    to UTC when there's no profile / timezone (fresh install) or the stored zone
    is unknown.  Entry/log timestamps stay UTC via ``format_log_timestamp`` —
    those are absolute historical markers, not the current-now anchor.
    """
    iana = _user_timezone_name(db)
    if iana is None:
        return UTC
    try:
        return ZoneInfo(iana)
    except ZoneInfoNotFoundError:
        logger.warning("Unknown profile timezone %r — anchoring in UTC", iana)
        return UTC


def _user_timezone_name(db: Database) -> str | None:
    """The primary user's stored IANA timezone, or None with no profile."""
    sender = db.users.get_primary_sender()
    if sender is None:
        return None
    user_info = db.users.get_info(sender)
    return user_info.timezone if user_info is not None else None
