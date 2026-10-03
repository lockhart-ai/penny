"""The clock Penny reads.

One seam for "what time is it".  The current-date line the model reads, the instant a
user's words about time are counted from, the moment a schedule's occurrences are
anchored at, the collector's readiness and expiry checks, the send cooldown, and every
timestamp the database writes all read the SAME clock, so nothing in one process can
disagree about when it is.

Handed down, never ambient: it sits on ``Config``, ``Penny`` passes it to the
``Database``, and every store and every reader takes it from there (``db.clock``).
Nothing swaps it while Penny runs.

What stays on the machine's own clock is whatever measures the PROCESS or talks to a
system with a clock of its own, rather than Penny's sense of the time:

- how long a model call took, and how long the process has been up;
- an OAuth or push token's expiry, which the issuing service counts in real seconds;
- a browser socket's heartbeat age, and the scheduler's monotonic intervals;
- the window a calendar plugin asks a remote calendar for;
- the migration runner's own bookkeeping.

A dependency-free leaf (stdlib only), so the database layer and the config can both
import it without a cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime


class Clock:
    """The real wall clock.  Production's, and the default wherever none is passed."""

    def now(self) -> datetime:
        """The current instant, timezone-aware, in UTC."""
        return datetime.now(UTC)
