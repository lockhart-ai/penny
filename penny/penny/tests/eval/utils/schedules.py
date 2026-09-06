"""What a stored schedule SAYS — the reading of ``MechanismRecord.schedule`` (#2005).

Its own module rather than a function beside the field it reads, for one physical reason: the
grammar's literals come from the tool that WRITES a rule, and that module imports the database
package — which runs ``penny.validation``'s ``__init__``, reaches ``penny.agents``, and comes
back into the half-built database package.  Anything importing it therefore stops being a
dependency-light leaf, and ``cohort.py`` has to stay one: ``assemble`` imports it cold, with no
database in the process, so a run's report could not be assembled at all.

``assertions.py`` already imports ``ChatAgent`` and is free to depend on the database, so it
takes the cadence claim and this module supplies the reading — which keeps that claim beside
the four siblings it is made with, where a reader looking for the terms a job runs on will
look.

The literals themselves are read from the tool rather than restated: a stored rule renders back
as the copyable ``schedule`` input (#1857), and a second copy of the line and tag spellings
would be a second contract free to drift from it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import islice

from dateutil.rrule import rrulestr

from penny.tools.collection_instantiation import (
    _DTSTART_TAG,
    _LINE_ESCAPE,
    _RRULE_TAG,
)

# The rule part that ANCHORS a recurrence to a time of day.  Read as a PART of the stored
# rule rather than off the parsed object, because dateutil defaults an unstated hour to the
# start's — so the parsed rule cannot tell a stated hour from an inherited one, and only the
# text says whether the model chose one.
HOUR_PART = "BYHOUR"

# Where a rule with no ``DTSTART`` of its own is anchored for measurement.  Any fixed instant
# does: the cadence is the GAP between occurrences, and a gap does not move with the anchor.
_MEASURING_ANCHOR = datetime(2000, 1, 1, tzinfo=UTC)

# How many occurrences a gap needs.  Two — a rule that can only fire once (COUNT=1) has no
# cadence to read, which is a real shape and reads as no cadence rather than as an error.
_OCCURRENCES_FOR_A_GAP = 2


def _rule_body(schedule: str) -> str:
    """The stored schedule's RULE line — the ``DTSTART`` line dropped and any ``RRULE:``
    tag stripped.  A schedule renders on one line with its newline written ``\\n`` (the form
    the parser accepts back), so both spellings are unfolded first."""
    lines = [line for line in schedule.replace(_LINE_ESCAPE, "\n").splitlines() if line.strip()]
    body = next((line for line in reversed(lines) if not line.upper().startswith(_DTSTART_TAG)), "")
    return body[len(_RRULE_TAG) :] if body.upper().startswith(_RRULE_TAG) else body


def rule_parts(schedule: str) -> set[str]:
    """Which PARTS the stored rule states, by name — the declared shape, read structurally
    off the rule rather than by comparing its spelling to one we had in mind."""
    return {part.partition("=")[0].strip().upper() for part in _rule_body(schedule).split(";")}


def cadence_seconds(schedule: str) -> int | None:
    """How often the stored rule FIRES, in seconds — the gap between its first two
    occurrences, measured by walking the rule itself.

    Reading the gap rather than the FREQ/INTERVAL pair is what makes the check answer the
    question the acceptance asked ("every day") instead of a question about spelling: a
    daily cadence written ``FREQ=DAILY``, ``FREQ=HOURLY;INTERVAL=24``, or
    ``FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR,SA,SU`` all fire a day apart, and all three are the
    same answer.  ``None`` when the rule fires at most once."""
    text = schedule.replace(_LINE_ESCAPE, "\n")
    anchored = _DTSTART_TAG in text.upper()
    rule = rrulestr(text) if anchored else rrulestr(text, dtstart=_MEASURING_ANCHOR)
    occurrences = list(islice(iter(rule), _OCCURRENCES_FOR_A_GAP))
    if len(occurrences) < _OCCURRENCES_FOR_A_GAP:
        return None
    return int((occurrences[1] - occurrences[0]).total_seconds())
