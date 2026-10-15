"""Shared machinery for the end-to-end job cases: the one case-local variance feature over the
whole sample's walk, and the small readers over `SampleObservation.cycles` every case's STORE
claims use to read the created job's cycles as a union (#2375) — never by a container name, per
the ruling that drops every claim on which collection a job lands on.
"""

from __future__ import annotations

from collections.abc import Sequence

from penny.tests.eval.utils.cohort import CycleSnapshot, Feature

# The sample's WHOLE walk: the conversation machine's moves through the scripted turns, then
# every created job's own ordered cycle script ("wrote", "quiet", "…+told") — one feature
# because ``TRANSITIONS`` and ``CYCLE_SCRIPT`` both read the identical field under two labels
# that fit neither half of an end-to-end sample on its own.
END_TO_END_WALK = Feature("end-to-end walk", lambda o: o.walk, absent="no move")

# The routine's own name and shape are MEASURED nowhere by these cases, and this is the one
# place that is worth saying rather than leaving as a silent omission: a round that teaches a
# routine from scratch mints a fresh name and shape every sample, the way ``test_teach_arrives_
# whole.py`` reports FRAMER naming spread with ``ROUTINE_NAME`` — but an end-to-end case's
# claims and features never read which collection the job lands on or what the routine is
# called, so there is nothing here that reading the registry would answer differently from
# reading the store the cases already read.


def held_text_at(snapshots: Sequence[CycleSnapshot], index: int) -> str:
    """Every entry held by the created job(s) after cycle ``index`` — the UNION across however
    many jobs the script created, key and content together.

    ``index`` is the arm's own per-cycle position (0 for the first cycle the script's job ran,
    1 for the second, …), not a count: a case names what the store holds AFTER one particular
    cycle, so a reader can tell "the first reading landed" from "the moved reading replaced
    it" without having to recompute an offset."""
    return " ".join(
        entry.text for snapshot in snapshots if snapshot.index == index for entry in snapshot.held
    )


def ran_at(snapshots: Sequence[CycleSnapshot], index: int) -> bool:
    """Whether ANY created job ran a cycle at this position — the guard a negative claim
    needs so it is only answered against a sample that actually reached that cycle.

    Positive claims (``held_text_at`` names a value, ``sent_at`` names a message) read FALSE
    on their own when no snapshot exists at ``index``, because an empty string contains no
    amount and an empty list has no length to equal.  A claim shaped "nothing happened" has
    no such floor — ``not sent_at(..., index)`` is vacuously TRUE of a sample that never got
    there — so it reads this first (#2375's review)."""
    return any(snapshot.index == index for snapshot in snapshots)


def sent_at(snapshots: Sequence[CycleSnapshot], index: int) -> list[str]:
    """Every message the created job(s) put on the send queue DURING cycle ``index`` — the
    union across however many jobs the script created.

    Per-cycle rather than cumulative, because "told once" and "told once more two cycles
    later" are two different findings a running total erases: a watch that notifies on cycle
    one and stays quiet on cycle two must show nothing sent on cycle two, not two messages
    carried forward from the first."""
    return [
        message for snapshot in snapshots if snapshot.index == index for message in snapshot.sent
    ]


__all__ = ["END_TO_END_WALK", "held_text_at", "ran_at", "sent_at"]
