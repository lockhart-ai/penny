"""When a stored job ENDS, and whether that is the end the ask gave (#2193).

A stand-up ask states its end in the user's own words — "until 10pm tonight", "until sunday
night", or nothing at all — and a job can store an end three ways: an ``expires_at``, an
``UNTIL=`` in its rule (which the schedule parser lifts into ``expires_at``), or a ``COUNT=``
(lifted into ``max_runs``).  The claim this module answers compares the end the job STORES
against the end the ask GAVE, on the user's own clock.  That an end merely exists says
nothing: a job that ended before the turn ran, one that ends a week late and one that ends on
the wrong day all carry one.

**The ask's end is a WINDOW on the user's clock, declared by the case.**  "10pm tonight" is the
single instant 22:00 on the day of the turn; "sunday night" is every instant from 18:00 on the
coming Sunday to 06:00 on the Monday after, both included; an ask that gave no end has no
window and the job must store none.

**A stored end is read as what the job will DO.**  An expiry is an instant, and it holds when
that instant is inside the window.  A counted rule has no instant: it ends at its LAST FIRING,
and any end from that firing up to the firing an uncounted rule would make next stops the
job at the same place.  So a counted end holds when that span and the window share an instant
— the count resolves to the same last firing an end inside the window would.  A firing that
falls exactly on the end is one the job may or may not make, so the span includes both ends.
Where a job stores both, the one that stops it first is its end.

**The user's clock is read the way production reads it.**  Occurrences are walked from the
anchor ``next_occurrence`` uses — the collection's creation moment on the user's wall clock,
seconds zeroed — through the tool's own helpers, so an hour a rule states is the hour the
collector would fire it at.  The timezone is the profile's, read off the sample's database.

Its own module beside ``cohort.py`` rather than inside it, because the anchor helpers come from
the tool that writes a rule and that module imports the database package; ``cohort.py`` is
imported cold by the report assembler and has to stay a dependency-light leaf.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, time, timedelta, tzinfo
from itertools import islice
from typing import NamedTuple

from dateutil.rrule import rrule, rruleset, rrulestr

from penny.datetime_utils import stored_as_utc, zone_or_utc
from penny.tests.eval.utils.cohort import MechanismRecord
from penny.tools.collection_instantiation import _LINE_ESCAPE, _schedule_anchor, _zoned

# One claim's answer: whether it held, and the rationale shown when it did not.
EndAnswer = tuple[bool, str]

# The window an ask's end falls in, given WHEN the turn ran on the user's clock: its first and
# last instant, both included.  One instant is a window whose two ends are equal.
Window = Callable[[datetime], tuple[datetime, datetime]]

# "sunday night", on the user's clock: from the evening of the coming Sunday to early the next
# morning.  ``date.weekday()`` counts Monday as 0, so Sunday is 6.
_SUNDAY = 6
_NIGHT_BEGINS = time(18, 0)
_NIGHT_ENDS = time(6, 0)

# How a moment is written in a rationale: the weekday beside the date, because the measured
# misses are ends on the wrong DAY, and the zone, because the stored value is UTC.
_SPOKEN = "%a %Y-%m-%d %H:%M %Z"


class AskedEnd(NamedTuple):
    """The end an ask states.  ``window`` is ``None`` when the ask gave no end at all."""

    says: str
    window: Window | None


def _tonight_at(hour: int) -> Window:
    def window(turn: datetime) -> tuple[datetime, datetime]:
        instant = turn.replace(hour=hour, minute=0, second=0, microsecond=0)
        return instant, instant

    return window


def _sunday_night(turn: datetime) -> tuple[datetime, datetime]:
    sunday = turn.date() + timedelta(days=(_SUNDAY - turn.weekday()) % 7)
    begins = datetime.combine(sunday, _NIGHT_BEGINS, tzinfo=turn.tzinfo)
    ends = datetime.combine(sunday + timedelta(days=1), _NIGHT_ENDS, tzinfo=turn.tzinfo)
    return begins, ends


def until_tonight_at(hour: int) -> AskedEnd:
    """An ask that ends the job at ``hour`` o'clock on the day of the turn."""
    return AskedEnd(f"until {hour:02d}:00 tonight", _tonight_at(hour))


UNTIL_SUNDAY_NIGHT = AskedEnd("until sunday night", _sunday_night)
NO_END = AskedEnd("no end", None)


class StoredEnd(NamedTuple):
    """When a stored job stops.  ``ends`` is its expiry, or its counted rule's last firing.

    ``next_firing`` is set for a COUNTED end only: when the same rule would fire next had it
    not been counted.  Any end from ``ends`` up to that moment stops the job at the same
    firing, which is what lets a count be compared with an instant."""

    ends: datetime
    next_firing: datetime | None
    stored_as: str


def firings(job: MechanismRecord, timezone: str | None, limit: int) -> list[datetime]:
    """The first ``limit`` moments the job's stored rule fires, on the user's clock.

    Walked from production's own anchor, so a stated hour is the hour the collector fires at
    and a rule that states none inherits the creation moment's, exactly as it will run."""
    return [moment for moment in islice(_walk(job, timezone), limit) if moment is not None]


def _rule(job: MechanismRecord, timezone: str | None) -> rrule | rruleset | None:
    if job.schedule is None or job.created_at is None:
        return None
    text = job.schedule.replace(_LINE_ESCAPE, "\n")
    return rrulestr(text, dtstart=_schedule_anchor(job.created_at, zone_or_utc(timezone)))


def _walk(job: MechanismRecord, timezone: str | None) -> Iterator[datetime | None]:
    rule = _rule(job, timezone)
    zone = zone_or_utc(timezone)
    return (_on_the_users_clock(moment, zone) for moment in rule) if rule is not None else iter(())


def _on_the_users_clock(moment: datetime | None, zone: tzinfo) -> datetime | None:
    """A firing as the user's own wall clock reads it.  A rule anchored with ``DTSTART:…Z``
    yields its firings in UTC, and the hour a claim reads is the hour where the user is."""
    zoned = _zoned(moment, zone)
    return zoned.astimezone(zone) if zoned is not None else None


def _counted_end(job: MechanismRecord, timezone: str | None) -> StoredEnd | None:
    """The end a ``COUNT=`` gives: its last firing, and when an uncounted rule would fire
    next.  ``None`` when the job stores no count, or its rule never fires."""
    if job.max_runs is None:
        return None
    counted = firings(job, timezone, job.max_runs)
    if not counted:
        return None
    return StoredEnd(counted[-1], _firing_after(job, timezone, counted[-1]), _count_text(job))


def _firing_after(job: MechanismRecord, timezone: str | None, last: datetime) -> datetime | None:
    """When the job's rule would fire after ``last`` were it not counted."""
    rule = _rule(job, timezone)
    if not isinstance(rule, rrule):
        return None
    uncounted = rule.replace(count=None)
    first = next(iter(uncounted), None)
    if first is None:
        return None
    clock = last if first.tzinfo is not None else last.replace(tzinfo=None)
    return _on_the_users_clock(uncounted.after(clock), zone_or_utc(timezone))


def _count_text(job: MechanismRecord) -> str:
    return f"after {job.max_runs} runs of {job.schedule!r}"


def _expiry_end(job: MechanismRecord, timezone: str | None) -> StoredEnd | None:
    if job.expires_at is None:
        return None
    expiry = stored_as_utc(job.expires_at).astimezone(zone_or_utc(timezone))
    return StoredEnd(expiry, None, "by its expiry")


def stored_end(job: MechanismRecord, timezone: str | None) -> StoredEnd | None:
    """When the job stops, or ``None`` when it stores no end.  Where it stores both an expiry
    and a count, the one that stops it first."""
    ends = [end for end in (_expiry_end(job, timezone), _counted_end(job, timezone)) if end]
    return min(ends, key=lambda end: end.ends) if ends else None


def _spoken(moment: datetime) -> str:
    return moment.strftime(_SPOKEN)


def _stored_text(end: StoredEnd | None) -> str:
    if end is None:
        return "the job stores no end"
    return f"the job ends {_spoken(end.ends)} ({end.stored_as})"


def _asked_text(asked: AskedEnd, window: tuple[datetime, datetime] | None) -> str:
    if window is None:
        return f"the ask gave {asked.says}"
    begins, ends = window
    span = _spoken(begins) if begins == ends else f"{_spoken(begins)} to {_spoken(ends)}"
    return f"the ask says {asked.says}, which is {span}"


def _inside(end: StoredEnd, window: tuple[datetime, datetime]) -> bool:
    """Whether the stored end stops the job where an end inside the window would."""
    begins, ends = window
    if end.next_firing is None:
        return begins <= end.ends.replace(second=0, microsecond=0) <= ends
    return end.ends <= ends and end.next_firing >= begins


def job_ends_as_asked(
    job: MechanismRecord, asked: AskedEnd, *, turn_at: datetime | None, timezone: str | None
) -> EndAnswer:
    """Whether the job's stored end is the end the ask gave, on the user's clock.

    The rationale names the stored end and the expected one, whichever way it missed."""
    if turn_at is None:
        return False, "the sample recorded no turn to read the ask's end against"
    turn = stored_as_utc(turn_at).astimezone(zone_or_utc(timezone))
    end = stored_end(job, timezone)
    window = asked.window(turn) if asked.window is not None else None
    said = f"{_stored_text(end)}; {_asked_text(asked, window)}"
    if window is None or end is None:
        return window is None and end is None, said
    if end.ends < turn:
        return False, f"{said}; that end is before the turn ran at {_spoken(turn)}"
    return _inside(end, window), said
