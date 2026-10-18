"""Tests for datetime_utils.format_duration_plain_words."""

from penny.datetime_utils import format_duration_plain_words


def test_format_duration_plain_words_under_60_seconds():
    """Durations under 60 seconds render in whole seconds."""
    assert format_duration_plain_words(45000) == "45 seconds"
    assert format_duration_plain_words(1000) == "1 second"


def test_format_duration_plain_words_exactly_60_seconds():
    """Exactly 60 seconds renders as 1 minute."""
    assert format_duration_plain_words(60000) == "1 minute"


def test_format_duration_plain_words_minutes_with_seconds():
    """Durations with minutes and seconds render both."""
    assert format_duration_plain_words(90000) == "1 minute 30 seconds"


def test_format_duration_plain_words_multiple_minutes():
    """Durations over 60 seconds render as minutes plus seconds if needed."""
    assert format_duration_plain_words(300000) == "5 minutes"


def test_format_duration_plain_words_hour_and_minutes():
    """Durations over an hour render as hours and minutes."""
    assert format_duration_plain_words(3700000) == "1 hour 1 minute"


def test_format_duration_plain_words_multiple_hours():
    """Multiple hours render correctly."""
    assert format_duration_plain_words(7200000) == "2 hours"


def test_format_duration_plain_words_hours_minutes_seconds():
    """Hours, minutes, and seconds only show non-zero minutes."""
    # 1 hour 5 minutes (300 seconds = 5 minutes)
    assert format_duration_plain_words(3900000) == "1 hour 5 minutes"
