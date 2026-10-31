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


def stored_as_utc(when: datetime) -> datetime:
    """A datetime read back from the database, made timezone-aware.

    SQLite hands back naive values and every column here is written in UTC, so a naive
    one IS a UTC one.  Public and single-sourced because more than one caller has to
    agree about it: the collector compares stored stamps against ``now``, and the
    schedule anchor converts one onto the user's wall clock (#1932) — two copies of the
    same one-line assumption is one copy too many.
    """
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def utc_isoformat(when: datetime) -> str:
    """A database timestamp as an ISO string that ALWAYS carries an offset.

    The one chokepoint every wire payload's datetimes serialize through (browser
    channel payloads, the prompt-log records).  SQLite hands naive values back,
    and JavaScript parses an offset-less ISO string as LOCAL time — so an
    offset-less wire stamp displays hours off and sorts wrongly against
    timezone-aware ones (a live ``call_started`` time from ``Clock.now()`` is
    aware, so a running entry once sorted above finished runs, #2638).  Naive IS
    UTC (``stored_as_utc``); an already-aware value passes through untouched,
    never shifted."""
    return stored_as_utc(when).isoformat()


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


def format_user_timestamp(when: datetime, timezone_name: str | None) -> str:
    """Render an absolute instant on the USER's own wall clock — the same shape
    ``current_datetime_line`` renders "now" in (weekday, date, time, zone
    abbreviation) — rather than the UTC ``format_log_timestamp`` uses.

    For a fact the user stated or reasons about on their own clock (today: a job's
    ``expires_at``, #2206/#2251) the model should never have to convert a zone to
    confirm what it set — showing it back in UTC is what made "until 10pm tonight"
    unreadable against the "11pm" the model itself wrote. Naive datetimes are read
    as UTC (how every timestamp column is stored); ``timezone_name=None`` falls
    back to UTC, the same fallback ``zone_or_utc`` takes everywhere else. Entry/log
    timestamps are deliberately NOT ported to this — they are absolute historical
    markers (``format_log_timestamp``'s own job), not a fact on the user's clock."""
    local = stored_as_utc(when).astimezone(zone_or_utc(timezone_name))
    return local.strftime(PennyConstants.CURRENT_DATETIME_FORMAT)


_USER_CLOCK_TIME_FORMAT = "%I:%M %p %Z"


def format_user_clock_time(when: datetime, timezone_name: str | None) -> str:
    """The TIME-OF-DAY half of ``format_user_timestamp`` — ``"9:30 AM EDT"``, no
    weekday or date — for a line that already states which day it means, or
    where two instants (a run's own start and the occurrence it was for, #2862)
    sit side by side and restating a shared date on both would only repeat it.
    Same zone/naive-is-UTC handling as ``format_user_timestamp``."""
    local = stored_as_utc(when).astimezone(zone_or_utc(timezone_name))
    return local.strftime(_USER_CLOCK_TIME_FORMAT)


async def get_timezone(location: str) -> str | None:
    """
    Derive IANA timezone from natural language location.

    Args:
        location: Natural language location (e.g., "New York, USA")

    Returns:
        IANA timezone string (e.g., "America/New_York") or None if lookup failed
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

    The instant is the database's own clock (``db.clock``), the one its rows are
    stamped from — so the time the model is told and the time Penny records and
    counts from are one reading.
    """
    local = db.clock.now().astimezone(user_timezone(db))
    stamp = local.strftime(PennyConstants.CURRENT_DATETIME_FORMAT)
    return f"{PennyConstants.CURRENT_DATETIME_PREFIX}{stamp}"


def zone_or_utc(timezone_name: str | None) -> tzinfo:
    """An IANA zone name resolved to a timezone — UTC when there is none, or when the
    system can't resolve the one on file.

    The ONE resolution behind every clock Penny reasons on: the current-date/time anchor
    the model reads, the zone a user's own words about time are read in
    (``parse_expires_at``), and the clock a schedule's stated hour runs on
    (``next_occurrence``, #1932).  Single-sourced because a second copy would be a second
    answer to "whose clock is this?", and the whole point is that they can't disagree.

    An unresolvable zone degrades VISIBLY — logged, naming the value — rather than
    quietly: on the schedule gate a silent UTC fallback restores exactly the wrong-hour
    firing the zone was introduced to fix.  A name that isn't a zone at all raises
    ``ValueError`` rather than ``ZoneInfoNotFoundError``, so both are answered here and
    a bad profile can never surface as a complaint about something else.
    """
    if timezone_name is None:
        return UTC
    try:
        return ZoneInfo(timezone_name)
    except ValueError, ZoneInfoNotFoundError:
        logger.warning("Unknown profile timezone %r — falling back to UTC", timezone_name)
        return UTC


def user_timezone(db: Database) -> tzinfo:
    """The primary user's IANA timezone for the current-date/time anchor.

    The profile advertises the user's timezone, so the clock the model reasons
    from must match it — otherwise Penny is told the wrong time-of-day always,
    and (for the hours around local midnight) the wrong calendar day.  Falls back
    to UTC when there's no profile / timezone (fresh install) or the stored zone
    is unknown.  Entry/log timestamps stay UTC via ``format_log_timestamp`` —
    those are absolute historical markers, not the current-now anchor.
    """
    return zone_or_utc(user_timezone_name(db))


def user_timezone_name(db: Database) -> str | None:
    """The primary user's stored IANA timezone, or None with no profile.

    Public because the words a user says about time are read in their own zone as
    well as shown in it (``parse_expires_at``, #1857): the zone is passed to the
    parser as a parameter, so nothing reaches into the database from the pure
    parsing layer."""
    sender = db.users.get_primary_sender()
    if sender is None:
        return None
    user_info = db.users.get_info(sender)
    return user_info.timezone if user_info is not None else None


_SECOND = "second"
_MINUTE = "minute"
_HOUR = "hour"


def _count_of(n: int, unit: str) -> str:
    """Format a count with its unit, pluralizing if needed."""
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def _two_units(big: int, big_unit: str, small: int, small_unit: str) -> str:
    """Combine two time units, omitting the small one if zero."""
    big_part = _count_of(big, big_unit)
    if small == 0:
        return big_part
    return f"{big_part} {_count_of(small, small_unit)}"


def format_duration_plain_words(milliseconds: int) -> str:
    """Render a duration in milliseconds as human-readable plain words.

    Returns a string like "45 seconds", "1 minute 30 seconds", "1 hour 5 minutes".
    Rules:
    - Under 60 seconds: whole seconds ("45 seconds", "1 second")
    - Otherwise minutes, plus seconds if not whole ("5 minutes", "1 minute 30 seconds")
    - For an hour or more: hours and minutes ("1 hour", "2 hours 15 minutes")
    """
    total_seconds = milliseconds // 1000

    if total_seconds < 60:
        return _count_of(total_seconds, _SECOND)

    total_minutes = total_seconds // 60
    remainder_seconds = total_seconds % 60

    if total_minutes < 60:
        return _two_units(total_minutes, _MINUTE, remainder_seconds, _SECOND)

    total_hours = total_minutes // 60
    remainder_minutes = total_minutes % 60
    return _two_units(total_hours, _HOUR, remainder_minutes, _MINUTE)
