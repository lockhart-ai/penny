"""Utility functions for date/time operations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

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
