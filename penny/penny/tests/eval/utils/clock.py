"""The clock a sample runs on (#2207).

A sample's Penny reads the time from the clock on its ``Config`` (``penny/clock.py``), and
on the real clock an ask that names a time relative to "now" is a different ask at every
run: "until 10pm tonight" has already passed at 23:30, "until sunday night" is today on a
Sunday, and a rule that states no hour fires at whatever hour the run happened to be made.
So every sample starts at ONE declared instant, whenever the run is made.

**It starts there and keeps time.**  ``PinnedClock`` reads ``starts_at`` when it is built
and advances at the real rate after that, so it is the machine's clock moved by a constant
offset.  Rows written a moment apart are still a moment apart, in the same order — a frozen
clock would stamp a whole sample with one instant and collapse every recency ordering the
runtime reads.

**Everything in the sample reads it.**  The current-date line the model is shown, the
instant its words about time are counted from, a job's creation moment and so the anchor
its rule fires from, the collector's readiness, every row the seeders and the turn write,
and the move a claim reads the turn's own time off — all through the sample's database,
so a claim about when a job ends is answered on the clock the job was set up on.

**Which instant: a Wednesday, at two in the afternoon, where the user is.**

- *Mid-week*, so a named day lies clearly ahead: "sunday night" is four days off, never
  tonight and never six days gone, and "tomorrow" is an ordinary weekday.
- *Early afternoon*, so the same day still holds an evening and already holds a morning:
  "tonight" and "10pm" are ahead on the turn's own date, and "tomorrow morning" is
  unambiguous.  It is also neither a morning nor an evening hour itself: a rule that states
  no hour inherits the turn's, and at 14:00 that reads as what it is rather than passing
  for a job somebody asked to run in the morning.
- *Mid-October*, clear of both daylight-saving changes and of a month's end, so no window a
  case reasons about straddles either.
- *In the profile's own zone*, the one ``seed_user`` writes, so the wall-clock reading
  above is the reading the model is shown.

A fixed date rather than a fixed hour of the run's own date: an hour alone still leaves the
day of the week to the calendar, which is the same defect one field over.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from penny.clock import Clock

# The eval user's zone — what ``seed_user`` writes on the profile, and the zone the
# starting instant below is a wall-clock reading in.
SAMPLE_TIMEZONE = "America/Los_Angeles"

# When every sample starts: Wednesday 14 October 2026, 14:00 where the user is.
SAMPLE_STARTS_AT = datetime(2026, 10, 14, 14, 0, tzinfo=ZoneInfo(SAMPLE_TIMEZONE))


class PinnedClock(Clock):
    """A clock that read ``starts_at`` when it was built and has kept time since.

    ``elapsed`` is the seconds-counter the passage of time is read off — the machine's
    monotonic one by default, which a wall-clock adjustment cannot move.  A test passes
    its own, so what the clock reads is stated rather than waited for."""

    def __init__(
        self, starts_at: datetime = SAMPLE_STARTS_AT, elapsed: Callable[[], float] = time.monotonic
    ) -> None:
        self._starts_at = starts_at.astimezone(UTC)
        self._elapsed = elapsed
        self._began = elapsed()

    def now(self) -> datetime:
        return self._starts_at + timedelta(seconds=self._elapsed() - self._began)
