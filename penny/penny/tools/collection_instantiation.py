"""The front door of collector creation (#2455 naive baseline; was #1591).

A collection is storage plus an optional job: ``collection_configure`` takes a
``prompt`` — free text the model writes — and stores it verbatim as the
collection's ``extraction_prompt``, the instructions the job's cycles follow.
There is no skill to resolve, no parameters to bind, and no rendering seam: the
prompt the model wrote IS the prompt the collector runs. This module holds the
pure, DB-free pieces of that flow so they are whole-render tested in isolation:

* the **idempotency-at-birth** tombstone result (#1567) — the archived-duplicate
  confirm-shape, naming the retired row and the unarchive that revives it.  (Its
  active-duplicate sibling is gone: an active duplicate is no longer refused, it
  is UPDATED in place — see ``CollectionSetTool._same_job``, #1775);
* the **schedule** parse (one ``schedule`` arg, one grammar: an RRULE string,
  #1857) and the ``expires_at`` end condition (ISO or natural language), with
  ``render_schedule_clause`` rendering the stored rule back verbatim — it IS the
  input form — and ``next_occurrence`` reading that stored rule on the USER'S
  clock (#1932), the seam the collector's readiness gate goes through.
  ``describe_schedule``/``describe_schedule_field`` (#2425) are what every
  model-facing surface reads instead: plain words leading, with that same exact
  rule kept alongside in a code span, so a reply narrating a job's cadence has
  words to copy rather than the raw RRULE;
* the **creation echo** (the prompt · schedule · notify · expiry), so the chat
  agent confirms back exactly what landed.

The orchestration (embed, parse, dedup, create) lives on
``CollectionCreateTool`` in :mod:`penny.tools.memory_tools`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, tzinfo

import dateparser
from dateutil.rrule import rrulestr
from pydantic import BaseModel

from penny.constants import PennyConstants
from penny.database.models import MemoryRow
from penny.datetime_utils import (
    format_log_timestamp,
    format_user_timestamp,
    stored_as_utc,
    zone_or_utc,
)

# ── Idempotency at birth (#1567) ──────────────────────────────────────────────

_TOMBSTONE_DUPLICATE = (
    "There's an archived collection for this: '{name}' (archived {archived_at}) — I didn't "
    "create a duplicate. Bring it back with collection_unarchive('{name}') to resume it, "
    "or set up a fresh one with a clearly different name and description."
)


def render_tombstone_duplicate(row: MemoryRow) -> str:
    """The tombstone-duplicate confirm-shaped result (#1567): surface the archived
    row and its archive time; unarchive or a deliberate override, never a silent
    proceed.  The archive timestamp is ``updated_at`` (stamped at archive)."""
    return _TOMBSTONE_DUPLICATE.format(
        name=row.name, archived_at=format_log_timestamp(row.updated_at)
    )


# ── Schedule: one arg, one grammar — RRULE (#1857) ───────────────────────────


class ScheduleError(Exception):
    """An actionable schedule/end-condition parse or validation failure — the tool
    surfaces ``str(self)`` as the failed result."""


class Schedule(BaseModel):
    """The parsed, store-ready schedule.

    ``rule`` is the RRULE text as given, trimmed and with its lines held as real
    newlines — it is what gets stored and what every surface renders back, so display
    form == invocation form with the rule itself as the copyable anchor.  ``max_runs``
    and ``expires_at`` are the ``COUNT=`` / ``UNTIL=`` parts lifted out of that same
    text into the columns that already own those end conditions; the parts stay in the
    stored rule too, so re-passing a rendered rule lifts the same values and
    round-trips.
    """

    rule: str
    max_runs: int | None = None
    expires_at: datetime | None = None


# One worked example per shape the grammar covers, as copyable lines.  Shared by every
# reject-and-teach text so a rejection always shows the whole grammar rather than only
# the part the failed input got wrong.
SCHEDULE_EXAMPLES = (
    "- FREQ=HOURLY — every hour\n"
    "- FREQ=MINUTELY;INTERVAL=90 — every 90 minutes\n"
    "- FREQ=DAILY;BYHOUR=8 — once a day at 8 in the morning, the user's own clock\n"
    "- FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9 — weekdays at 9 in the morning, the "
    "user's own clock\n"
    "- DTSTART:20260720T090000Z\\nFREQ=DAILY;COUNT=1 — once, at that exact moment in UTC, "
    "then it retires"
)

# The reject-and-teach failure for a schedule ``rrulestr`` won't read.  Names the
# grammar and shows it, so the model rewrites into RRULE rather than inventing a second
# schedule language.  A BLANK schedule never reaches here — it is refused at the arg gate
# with its own teaching rejection (#1776) — so this only ever answers garbled input.
_SCHEDULE_TEACHING = (
    "I couldn't read the schedule '{schedule}'. A schedule is one RRULE line — the "
    "calendar recurrence format — optionally preceded by a DTSTART line saying when it "
    "starts. Copy one of these shapes:\n"
    f"{SCHEDULE_EXAMPLES}\n"
    "Use COUNT= to stop after that many runs, and UNTIL=20260901T000000Z to stop at "
    "a time. "
    "Or leave the schedule out entirely for a storage-only collection."
)

# The same rejection when a mechanical repair of the given text DOES parse: lead with
# the corrected string so the retry is a COPY, never a re-derivation (a model asked to
# rewrite a rejected value re-derives a different one — the measured 7200 → 14400 class).
_SCHEDULE_DID_YOU_MEAN = (
    "I couldn't read the schedule '{schedule}' — did you mean '{corrected}'? Pass that "
    "exact string. A schedule is one RRULE line, optionally preceded by a DTSTART line:\n"
    f"{SCHEDULE_EXAMPLES}"
)

# A schedule carrying more than one rule line: the grammar is ONE rule (plus an optional
# DTSTART), because COUNT/UNTIL are lifted from that one rule into the collection's end
# conditions and two rules would give two answers.
_SCHEDULE_MULTIPLE_RULES = (
    "The schedule '{schedule}' has more than one rule line. A schedule is exactly one "
    "RRULE line, optionally preceded by one DTSTART line — combine what you want into a "
    "single rule:\n"
    f"{SCHEDULE_EXAMPLES}"
)

_DTSTART_TAG = "DTSTART"
_RRULE_TAG = "RRULE:"
_COUNT_PART = "COUNT"
_UNTIL_PART = "UNTIL"
# A schedule's two lines are separated by a newline, and every surface that shows a
# schedule is a one-line surface, so the separator is written as the two characters
# ``\n`` there.  The parser accepts both, so a rendered schedule copies straight back.
_LINE_ESCAPE = "\\n"
# Any dtstart validates the grammar, so validation uses a fixed one rather than a clock
# read — parsing the same text twice must give the same answer.
_VALIDATION_DTSTART = datetime(2000, 1, 1, tzinfo=UTC)
# How an UNTIL value is written.  A bare date is deliberately absent: the rule's
# start is always timezone-aware here, and dateutil refuses a naive UNTIL against
# one, so such a rule never reaches the lift at all.
_UNTIL_FORMAT = "%Y%m%dT%H%M%SZ"
# Wrappers a model puts around a value it is quoting rather than passing.
_WRAPPING_CHARACTERS = "'\"`"


def parse_schedule(schedule: str) -> Schedule:
    """Parse the single ``schedule`` arg into a store-ready :class:`Schedule` (#1857).

    One grammar: an RRULE line, optionally preceded by a ``DTSTART:`` line saying when
    the recurrence starts (default: the collection's ``created_at``, applied by the
    collector).  ``python-dateutil``'s ``rrulestr`` is the authority on whether the text
    is a schedule — a well-known formalism, no bespoke parsing — and ``COUNT=`` /
    ``UNTIL=`` are lifted out into ``max_runs`` / ``expires_at`` while staying in the
    stored rule, so a rendered schedule copies straight back and round-trips.

    Text ``rrulestr`` refuses raises the teaching :class:`ScheduleError`, carrying the
    mechanically-repaired string verbatim when one exists so the retry is a copy.
    """
    text = _canonical_lines(schedule)
    rule_line = _rule_line(text)
    if rule_line is None or not _reads_as_schedule(text):
        raise ScheduleError(_teaching_for(_one_line(schedule.strip())))
    parts = _rule_parts(rule_line)
    return Schedule(
        rule=text,
        max_runs=_lifted_count(parts, text),
        expires_at=_lifted_until(parts, text),
    )


def _one_line(text: str) -> str:
    """Schedule text as it is QUOTED back to the model — one line, its newline written
    as ``\\n``, matching how every render shows a schedule and what the parser accepts.
    A rejection that quotes the value across two lines is a value that can't be copied
    out of the message it appears in."""
    return text.replace("\n", _LINE_ESCAPE)


def _canonical_lines(schedule: str) -> str:
    """The schedule with its lines held as real newlines — a two-line rule written on
    one line with the ``\\n`` escape (how every surface renders it) is the same
    schedule as one written across two, and only one of them can be stored."""
    return schedule.strip().replace(_LINE_ESCAPE, "\n")


def _rule_line(text: str) -> str | None:
    """The schedule's one rule line — the ``DTSTART`` line, when present, says when the
    recurrence starts and carries no parts to lift.  ``None`` when there is no rule line
    at all; a SECOND rule line is its own refusal (the lift has to read one rule, not
    choose between two)."""
    lines = [
        line
        for line in (candidate.strip() for candidate in text.splitlines())
        if line and not line.upper().startswith(_DTSTART_TAG)
    ]
    if len(lines) > 1:
        raise ScheduleError(_SCHEDULE_MULTIPLE_RULES.format(schedule=_one_line(text)))
    return lines[0] if lines else None


def _rule_parts(rule_line: str) -> dict[str, str]:
    """The rule's ``NAME=VALUE`` parts, keyed by upper-cased name.  A malformed part is
    skipped here — ``rrulestr`` is what decides the text is unreadable, so this never
    raises on its own and the one rejection stays in one place."""
    body = rule_line[len(_RRULE_TAG) :] if rule_line.upper().startswith(_RRULE_TAG) else rule_line
    parts: dict[str, str] = {}
    for part in body.split(";"):
        name, separator, value = part.partition("=")
        if separator:
            parts[name.strip().upper()] = value.strip()
    return parts


def _reads_as_schedule(text: str) -> bool:
    """Does ``rrulestr`` accept this text as a recurrence rule?  The single authority on
    the grammar — anything it refuses is a reject-and-teach, whatever the reason."""
    try:
        rrulestr(text, dtstart=_VALIDATION_DTSTART)
    except ValueError, TypeError:
        return False
    return True


def _teaching_for(text: str) -> str:
    """The rejection for unreadable schedule text: the did-you-mean form when a
    mechanical repair of the text parses (so the retry is a copy of a rendered string),
    else the plain teaching form."""
    repaired = _mechanical_repair(text)
    if repaired != text and _reads_as_schedule(_canonical_lines(repaired)):
        return _SCHEDULE_DID_YOU_MEAN.format(schedule=text, corrected=repaired)
    return _SCHEDULE_TEACHING.format(schedule=text)


def _mechanical_repair(text: str) -> str:
    """The purely syntactic tidy-ups of a schedule string — no guess about what the
    text MEANS, so the corrected string it produces is safe to hand back as the exact
    retry.  Strips wrapping quotes/backticks, drops trailing part separators, and puts a
    ``DTSTART`` joined to the rule with a semicolon onto its own line."""
    repaired = text.strip().strip(_WRAPPING_CHARACTERS).strip().rstrip(";, ")
    if repaired.upper().startswith(_DTSTART_TAG) and _LINE_ESCAPE not in repaired:
        head, separator, tail = repaired.partition(";")
        if separator:
            repaired = f"{head}{_LINE_ESCAPE}{tail}"
    return repaired


def _lifted_count(parts: dict[str, str], text: str) -> int | None:
    """``COUNT=`` lifted into ``max_runs`` — the run quota the archive lifecycle already
    owns, so a bounded rule retires its collection the same way a quota always did."""
    raw = parts.get(_COUNT_PART)
    if raw is None:
        return None
    if not raw.isdigit() or int(raw) < 1:
        raise ScheduleError(
            f"The schedule '{_one_line(text)}' has COUNT={raw} — COUNT is how many times it "
            "runs, "
            "so it must be a whole number of at least 1 (COUNT=1 for a one-shot)."
        )
    return int(raw)


def _lifted_until(parts: dict[str, str], text: str) -> datetime | None:
    """``UNTIL=`` lifted into ``expires_at`` — the end condition the archive lifecycle
    already owns.  RFC 5545 writes it as ``YYYYMMDDTHHMMSSZ`` (or a bare date)."""
    raw = parts.get(_UNTIL_PART)
    if raw is None:
        return None
    try:
        return datetime.strptime(raw, _UNTIL_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        pass
    raise ScheduleError(
        f"The schedule '{_one_line(text)}' has UNTIL={raw}, which isn't a time I can read "
        "— write "
        "it as a UTC stamp like UNTIL=20260901T000000Z."
    )


def next_occurrence(
    schedule: str,
    created_at: datetime,
    timezone_name: str | None,
    after: datetime | None = None,
) -> datetime | None:
    """When a stored rule next comes round, read on the USER'S clock (#1932).

    ``after`` is the collection's last run; ``None`` asks for the rule's very FIRST
    occurrence.  Returns a timezone-aware datetime, or ``None`` when the rule has no
    occurrence left (a spent ``COUNT=``, a passed ``UNTIL=``).

    An RRULE states a time of day (``BYHOUR=9``) and never a zone, so the zone comes
    from whoever the rule belongs to: the recurrence is anchored at the collection's
    creation moment ON THE USER'S WALL CLOCK, which is what makes ``BYHOUR=9`` mean nine
    in the morning where they are.  Anchoring it in UTC is what fired a briefing asked
    for at a local morning hour four hours early for a UTC-4 profile, while the reply
    that set it up promised the local time.  The zone is passed in — ``None`` on a fresh
    install, which reads as UTC — so this stays DB-free.

    The rule text is NOT rewritten: a stored schedule is still the user's own words, so
    the render remains the copyable input form and re-passing it cannot convert it a
    second time.

    The anchor's SECONDS are zeroed.  An occurrence inherits whatever seconds its anchor
    carries, so a collection created at :51 past used to be a job that fired at :51 past
    for ever.

    A rule that states its own start keeps it: an explicit ``DTSTART:…Z`` is an exact
    instant and is honoured as one, while a FLOATING ``DTSTART`` (no ``Z``) states no
    zone at all — RFC 5545 calls that local time, so it is read on the same user clock
    as the default anchor and no occurrence can come back naive.

    The first occurrence is read by ITERATING rather than by asking what follows the
    anchor: ``rrulestr`` truncates its start to whole seconds, so "after the anchor"
    would exclude the collection's own first occurrence and a daily rule would wait a
    day to run its first cycle.
    """
    zone = zone_or_utc(timezone_name)
    rule = rrulestr(schedule, dtstart=_schedule_anchor(created_at, zone))
    first = next(iter(rule), None)
    if first is None or after is None:
        return _zoned(first, zone)
    return _zoned(rule.after(_rule_clock(stored_as_utc(after), first, zone)), zone)


def _schedule_anchor(created_at: datetime, zone: tzinfo) -> datetime:
    """Where a rule with no ``DTSTART`` of its own starts: the collection's creation
    moment on the user's wall clock, with its seconds zeroed."""
    return stored_as_utc(created_at).astimezone(zone).replace(second=0, microsecond=0)


def _zoned(occurrence: datetime | None, zone: tzinfo) -> datetime | None:
    """An occurrence as a timezone-aware datetime.  A floating one — from a rule whose
    own ``DTSTART`` states no zone — is the user's wall clock, per RFC 5545, rather than
    a naive value the readiness comparison would raise on."""
    if occurrence is None or occurrence.tzinfo is not None:
        return occurrence
    return occurrence.replace(tzinfo=zone)


def _rule_clock(when: datetime, first: datetime, zone: tzinfo) -> datetime:
    """``when`` on the same clock the rule's own occurrences carry, so ``rrule.after``
    can compare the two: aware against an anchored rule, the user's own naive wall clock
    against a floating one."""
    if first.tzinfo is not None:
        return when
    return when.astimezone(zone).replace(tzinfo=None)


def render_schedule_clause(row: MemoryRow) -> str:
    """The mechanism's schedule rendered back VERBATIM — display form == invocation
    form (#1857): what a surface shows (the self-state mechanisms line,
    ``memory_get``, the creation echo) is the stored rule itself, so it copies
    straight back as the ``schedule`` arg and round-trips to the same stored config.
    With one grammar there is nothing to re-derive at the render, which is the whole
    point of collapsing the union.  A two-line rule renders on ONE line with its
    newline written as ``\\n`` — the form ``parse_schedule`` accepts back, since every
    surface showing a schedule shows it inside a line.  Empty for a collection with no
    schedule — the labelled surfaces go through ``render_schedule_field``, which says
    ``none``."""
    if row.schedule is None:
        return ""
    return _one_line(row.schedule)


# The honest labelled-schedule fallback for a row with no schedule (#1666) — matching
# ``memory_get``'s ``schedule: none`` convention, so no surface emits a blank.
_NO_SCHEDULE_CLAUSE = "none"


def has_schedule(row: MemoryRow) -> bool:
    """True when ``row`` carries a schedule.  False for a log or an inert collection
    with no schedule yet — the single predicate every labelled-schedule render guards
    on."""
    return row.schedule is not None


def render_schedule_field(row: MemoryRow) -> str:
    """The schedule rendered for a labelled echo/metadata field: the stored rule
    verbatim when ``row`` has one, else the honest ``none`` — so an inert/no-schedule
    collection never renders a blank clause (#1666)."""
    return render_schedule_clause(row) if has_schedule(row) else _NO_SCHEDULE_CLAUSE


# ── Plain-words schedule description (#2425) ──────────────────────────────────
#
# The raw RRULE is exact but unreadable prose: a model asked to tell a job's cadence in
# its own words instead copied the grammar verbatim, and once a reply does that an
# invented DTSTART reads as a fact the user stated.  ``describe_schedule`` /
# ``describe_schedule_field`` lead with plain words and keep the exact rule alongside in
# a code span, so #1857's display == invocation contract still holds — the model still
# reads the exact copyable form it would write back to change the schedule — it just no
# longer stands alone as the thing a reply naturally echoes.

_FREQ_UNIT = {
    "SECONDLY": "second",
    "MINUTELY": "minute",
    "HOURLY": "hour",
    "DAILY": "day",
    "WEEKLY": "week",
    "MONTHLY": "month",
    "YEARLY": "year",
}

_WEEKDAY_NAMES = {
    "MO": "Monday",
    "TU": "Tuesday",
    "WE": "Wednesday",
    "TH": "Thursday",
    "FR": "Friday",
    "SA": "Saturday",
    "SU": "Sunday",
}
_WEEKDAY_SET = frozenset({"MO", "TU", "WE", "TH", "FR"})
_WEEKEND_SET = frozenset({"SA", "SU"})


def _clock_phrase(hour: str, minute: str | None) -> str:
    """One BYHOUR[,BYMINUTE] value as a 12-hour clock phrase — BYHOUR is the user's own
    wall-clock hour, never a zone to convert (the grammar's own convention — see
    SCHEDULE_EXAMPLES's "the user's own clock")."""
    stamp = datetime(2000, 1, 1, int(hour), int(minute) if minute is not None else 0)
    return stamp.strftime("%I:%M %p")


def _hours_phrase(byhour: str | None, byminute: str | None) -> str | None:
    """Every BYHOUR value (RRULE allows a comma list) as a clock phrase, joined 'and' —
    ``None`` with no BYHOUR at all, so a caller knows there is no time of day to state."""
    if byhour is None:
        return None
    minutes = byminute.split(",") if byminute else []
    hours = byhour.split(",")
    phrases = [
        _clock_phrase(hour, minutes[i] if i < len(minutes) else None)
        for i, hour in enumerate(hours)
    ]
    return " and ".join(phrases)


def _days_phrase(byday: str) -> str:
    """BYDAY's comma list as plain words — the taught Mon-Fri set reads as 'weekdays'
    (SCHEDULE_EXAMPLES' own word for it), Sat/Sun as 'weekends', anything else spells
    out each day by name."""
    codes = [code.strip().upper() for code in byday.split(",")]
    if frozenset(codes) == _WEEKDAY_SET:
        return "weekdays"
    if frozenset(codes) == _WEEKEND_SET:
        return "weekends"
    names = [_WEEKDAY_NAMES.get(code, code) for code in codes]
    if len(names) <= 2:
        return " and ".join(names)
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _cadence_phrase(parts: Mapping[str, str]) -> str:
    """How often, in words — FREQ x INTERVAL x BYHOUR/BYDAY — built from the rule's own
    parts rather than looked up by shape, so a combination nobody taught still reads as
    words instead of falling back to the raw grammar."""
    freq = parts.get("FREQ", "").upper()
    unit = _FREQ_UNIT.get(freq, freq.lower() or "cycle")
    interval = parts.get("INTERVAL")
    plain = interval in (None, "1")
    every = "every" if plain else f"every {interval}"
    noun = unit if plain else f"{unit}s"

    hours = _hours_phrase(parts.get("BYHOUR"), parts.get("BYMINUTE"))
    if freq == "WEEKLY" and "BYDAY" in parts:
        days = _days_phrase(parts["BYDAY"])
        return f"{days} at {hours}" if hours else days
    return f"{every} {noun} at {hours}" if hours else f"{every} {noun}"


def _has_dtstart(canonical: str) -> bool:
    """Whether the schedule states its own start — the only other line a stored rule
    can carry alongside its one rule line (#1857's grammar)."""
    return any(line.strip().upper().startswith(_DTSTART_TAG) for line in canonical.splitlines())


def _dtstart_phrase(canonical: str, timezone_name: str | None) -> str:
    """The schedule's own DTSTART read on the clock it actually states (#1932/#2425): an
    exact ``…Z`` instant converted to the user's own clock, or a floating (no-zone) stamp
    read AS the user's own wall clock already — never UTC for the floating case, the
    same reading ``_zoned`` gives one of the rule's own occurrences.  Read via
    ``rrulestr`` — the single authority on the grammar — rather than a bespoke format
    guess, since a stored schedule already parsed once at write time and this must read
    it identically."""
    moment = rrulestr(canonical, dtstart=_VALIDATION_DTSTART)._dtstart
    if moment.tzinfo is not None:
        return format_user_timestamp(moment, timezone_name)
    local = moment.replace(tzinfo=zone_or_utc(timezone_name))
    return local.strftime(PennyConstants.CURRENT_DATETIME_FORMAT)


def _describe_rule(rule: str, timezone_name: str | None) -> str:
    """One rule's own words: its cadence, its own start when it states one, and how many
    times when it is bounded.  The end condition (``UNTIL=``) is NOT restated here — it
    is already lifted onto the collection's own ``expires_at`` and rendered on that line,
    so repeating it here would state the same fact twice."""
    canonical = _canonical_lines(rule)
    rule_line = _rule_line(canonical)
    assert rule_line is not None  # a stored rule always has one — parse_schedule refused otherwise
    parts = _rule_parts(rule_line)
    cadence = _cadence_phrase(parts)
    count = parts.get("COUNT")
    start = _dtstart_phrase(canonical, timezone_name) if _has_dtstart(canonical) else None

    if count == "1":
        at = f", at {start}" if start else ""
        return f"once{at}, then it retires"
    times = f", {count} times" if count not in (None, "1") else ""
    if start:
        return f"starting {start}, then {cadence}{times}"
    return f"{cadence}{times}"


def _describe_rule_with_span(rule: str, clause: str, timezone_name: str | None) -> str:
    """Plain words plus the rule's own exact text in a code span, one line: the shared
    tail of ``describe_schedule`` (reading the span off ``render_schedule_clause``) and
    the was→now line's "was" side (reading it off a bare prior string, #1946) — both
    need the identical "<words> (rule: `<rule>`)" shape, so it lives once here.
    ``clause`` is joined onto one line with a plain SPACE rather than the ``\\n`` escape
    ``render_schedule_clause`` uses for its own copy-back form — that escape is what a
    reply echoed verbatim (#2425)."""
    return f"{_describe_rule(rule, timezone_name)} (rule: `{clause.replace(_LINE_ESCAPE, ' ')}`)"


def describe_schedule(row: MemoryRow, timezone_name: str | None = None) -> str:
    """Plain words for the mechanism's schedule, with its exact rule alongside in a code
    span (#1857's display == invocation contract, #2425's plain-words-first fix): the
    model still reads the exact copyable form (``render_schedule_clause``) it would
    write back to change the schedule, it just no longer stands alone as the thing a
    reply would naturally echo.  Empty for no schedule, like ``render_schedule_clause``."""
    if row.schedule is None:
        return ""
    return _describe_rule_with_span(row.schedule, render_schedule_clause(row), timezone_name)


def describe_schedule_field(row: MemoryRow, timezone_name: str | None = None) -> str:
    """The labelled-field counterpart (#1666's ``none`` fallback) over
    ``describe_schedule`` — reads the fallback off ``render_schedule_field`` so that
    function keeps a production caller too."""
    if not has_schedule(row):
        return render_schedule_field(row)
    return describe_schedule(row, timezone_name)


# ── The end condition: ISO first, then the user's own words (#1857) ──────────

# What dateparser is told about the words it is reading: they are the USER's, so they
# are in the USER's timezone and they point forward (an end condition is in the future).
# Returned in UTC, which is what the column holds.
_DATEPARSER_SETTINGS: dict[str, object] = {
    "PREFER_DATES_FROM": "future",
    "RETURN_AS_TIMEZONE_AWARE": True,
    "TO_TIMEZONE": "UTC",
}

# The reject-and-teach failure for an end condition neither reading answered.  Names
# both accepted shapes with a worked example of each, so the retry has somewhere to go.
_EXPIRES_TEACHING = (
    "I couldn't read expires_at='{value}' as a time. Give it either a date and time — "
    "2026-09-01T09:00:00Z — or when it ends in plain words, said the way a time is "
    "said: 'tomorrow at 9am', 'in two weeks', '10pm today'. Words that only point at a "
    "time without saying one ('tonight', 'the end of the month') don't land anywhere I "
    "can store, so say the hour."
)


def parse_expires_at(value: str, timezone_name: str | None, now: datetime) -> datetime | None:
    """Parse the ``expires_at`` end condition into a UTC-aware datetime (#1857), or
    ``None`` when the words state no end date at all (#1944).

    ISO first — an exact time is an exact time — then ``dateparser`` over the user's
    own words, read IN THE USER'S TIMEZONE.  The timezone is what makes the words mean
    what the user meant: '10pm today' is 10pm where they are, and reading it as UTC put
    a watch's end hours away from the evening it was asked for.  ``timezone_name`` is
    the user's IANA zone (``None`` on a fresh install → UTC) and ``now`` the moment
    relative words count from, both passed in rather than read from ambient state.
    Text neither reading answers raises the teaching :class:`ScheduleError`.

    **An end is a time on the USER'S clock (#2206).** A ``Z`` suffix or an explicit
    offset states its own zone and is honoured exactly as written — that reading never
    changes. A value with NO zone is a wall-clock reading with nowhere else to come
    from, so it is resolved on the user's clock here, at the tool boundary, the same
    place the spoken-words branch already reads from — never read as UTC by default.
    The model never converts a zone itself; it writes the hour the user said and this
    function is where that hour is resolved.

    A time past :func:`_is_sentinel_expiry`'s horizon reads as ``None`` — an unbounded
    job's end condition is the ABSENCE of one, so the sentinel is normalised here, at
    the boundary, rather than stored and then rendered back as a real end date on every
    surface that shows one.  The caller states what happened via
    :func:`sentinel_expiry_note`; it is never silent.
    """
    parsed = _parse_iso_datetime(value, timezone_name)
    if parsed is None:
        parsed = _parse_spoken_datetime(value, timezone_name, now)
    if parsed is None:
        raise ScheduleError(_EXPIRES_TEACHING.format(value=value))
    return None if _is_sentinel_expiry(parsed, now) else parsed


def _is_sentinel_expiry(expiry: datetime, now: datetime) -> bool:
    """True when ``expiry`` is so far ahead of ``now`` that it is a stand-in for
    "forever" rather than an end date anyone means (#1944).

    The horizon (``PennyConstants.SENTINEL_EXPIRY_HORIZON_DAYS``, with its rationale)
    is measured FROM ``now``, so nothing here rots as the clock advances past a fixed
    calendar date."""
    return expiry - now > timedelta(days=PennyConstants.SENTINEL_EXPIRY_HORIZON_DAYS)


def expiry_was_normalised(requested: str | None, resolved: datetime | None) -> bool:
    """True when this call STATED an end condition that read as no end date (#1944).

    The one structural read of the sentinel outcome, so the two things that follow from
    it cannot disagree: the row's expiry is CLEARED (an unbounded job must be able to
    take an invented expiry off a collection that already carries one — the stickiness
    the normalisation exists to end) and the result NAMES what happened
    (:func:`sentinel_expiry_note`).  Nothing else produces this pair: unreadable text and
    a doubly-stated end both raise instead."""
    return requested is not None and resolved is None


# What the result says when a stated end date read as no end date (#1944).  Actionable
# on both halves the rule owes: what became of the value that was passed, and what to
# pass instead next time.  It states the RESULTING STATE ("runs until it is stopped")
# rather than only the normalisation, because the turn goes on to tell the user what is
# running and a note that only says a value was dropped leaves that sentence unwritten.
_SENTINEL_EXPIRY_NOTE = (
    "\n(Note: expires_at='{value}' is further ahead than an end date anyone means, so I "
    "read it as no end date — the expiry is unset and this runs until it is stopped. "
    "Leave expires_at off when there is no end date.)"
)


def sentinel_expiry_note(requested: str | None, resolved: datetime | None) -> str:
    """The visible-degradation note for a normalised sentinel expiry (#1944), or empty
    when this call stated no end condition or stated a real one."""
    if not expiry_was_normalised(requested, resolved):
        return ""
    return _SENTINEL_EXPIRY_NOTE.format(value=requested)


def _parse_iso_datetime(value: str, timezone_name: str | None) -> datetime | None:
    """The ISO-8601 reading, or ``None`` when the text isn't one.

    A value carrying its own zone (``Z`` or an explicit offset) is read exactly as
    written.  A NAIVE value (#2206) is a wall-clock reading with no zone of its own,
    so it is read on the USER's clock (``zone_or_utc``, the same resolution every
    other wall-clock reading here takes) and converted to the absolute instant the
    column stores — never defaulted to UTC, which put a bounded watch's end hours
    away from the evening it was asked for."""
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed
    return parsed.replace(tzinfo=zone_or_utc(timezone_name)).astimezone(UTC)


def _parse_spoken_datetime(value: str, timezone_name: str | None, now: datetime) -> datetime | None:
    """The natural-language reading in the user's timezone, or ``None`` when
    ``dateparser`` finds no time in the words."""
    zone = zone_or_utc(timezone_name)
    settings = dict(_DATEPARSER_SETTINGS)
    settings["TIMEZONE"] = str(zone)
    # dateparser counts relative words ("in two weeks") from a NAIVE base it reads in
    # TIMEZONE, so hand it the user's own wall clock — a UTC base puts an evening on
    # the wrong day for anyone west of Greenwich.
    settings["RELATIVE_BASE"] = now.astimezone(zone).replace(tzinfo=None)
    parsed = dateparser.parse(value, settings=settings)  # ty: ignore[invalid-argument-type]
    if parsed is None:
        return None
    return parsed.astimezone(UTC)


# ── Creation echo ─────────────────────────────────────────────────────────────


def _schedule_line(row: MemoryRow, timezone_name: str | None) -> str:
    """The echo's one-line schedule summary — plain words with the exact rule in a code
    span (#2425), or ``schedule: none`` for an inert/unscheduled one (#1666)."""
    return f"  schedule: {describe_schedule_field(row, timezone_name)}"


_NO_EXPIRY_CLAUSE = "never"


def _expires_value(row: MemoryRow, timezone_name: str | None) -> str:
    """The end condition as it renders — on the USER's own clock (#2206/#2251: the
    same clock a stated end is read on, never the UTC a historical timestamp
    renders in), or the honest ``never`` an unbounded job carries.  Split out so
    the labelled echo line and the configured-record line (which may wrap it in a
    before→after clause, #1946) render the same value."""
    if row.expires_at is None:
        return _NO_EXPIRY_CLAUSE
    return format_user_timestamp(row.expires_at, timezone_name)


def _expires_line(row: MemoryRow, timezone_name: str | None) -> str:
    return f"  expires: {_expires_value(row, timezone_name)}"


# ── The plain-language lead line (what the collector WILL do) ───────────────
#
# The instantiation echoes LEAD with one deterministic English sentence composed
# from the STRUCTURED fields — cadence · target · notify — so the model (and
# the user it mirrors it to) reads what the collector will actually do without
# confabulating it.  Template-method over the schedule × the notify flag; the
# detailed field-by-field echo stays below it.


def _lead_cadence_phrase(row: MemoryRow, timezone_name: str | None) -> str:
    """The 'when it runs' clause, in plain words with the exact rule in a code span
    (#2425, ``describe_schedule``) — a bare 'I'll run' when the collection carries no
    schedule at all.  Any run quota is already stated in those words
    (``_describe_rule``'s own COUNT handling), so it is not stated a second time here."""
    if not has_schedule(row):
        return "I'll run"
    return f"On the schedule {describe_schedule(row, timezone_name)} I'll run"


def _lead_notify_tail(row: MemoryRow) -> str:
    """The 'what it does with a change' clause — the notify flag in plain words."""
    if row.notify:
        return "message you when something changes."
    return "quietly store what it finds."


def _lead_line(row: MemoryRow, timezone_name: str | None) -> str:
    """One deterministic English sentence: cadence + target + notify —
    e.g. "On the schedule every 15 minutes (rule: `FREQ=MINUTELY;INTERVAL=15`) I'll run
    against 'aurora-prices' and message you when something changes." /
    "…and quietly store what it finds."."""
    cadence = _lead_cadence_phrase(row, timezone_name)
    return f"{cadence} against '{row.name}' and {_lead_notify_tail(row)}"


def _instantiation_echo(row: MemoryRow, headline: str, timezone_name: str | None) -> str:
    """The shared instantiation confirm-shape — a plain-language LEAD line (what the
    collector will do) over a ``headline`` and description · schedule · notify ·
    expiry · the job's instructions verbatim.  Both the creation echo and the
    re-render echo compose it, so a freshly created collection and a re-configured
    one confirm back the same fields."""
    prompt = (row.extraction_prompt or "").replace("\n", "\n    ")
    lines = [
        _lead_line(row, timezone_name),
        headline,
        f"  description: {row.description}",
        _schedule_line(row, timezone_name),
        f"  notify: {row.notify}",
        _expires_line(row, timezone_name),
        "  prompt: |",
        f"    {prompt}",
    ]
    return "\n".join(lines)


def render_creation_echo(row: MemoryRow, timezone_name: str | None = None) -> str:
    """The structured creation echo — description, schedule, notify, expiry, and
    the job's prompt verbatim — so the chat agent confirms back exactly what
    landed without confabulating a field.  ``timezone_name`` defaults to ``None``
    (UTC) so a caller that predates #2206/#2251 renders byte-identically."""
    return _instantiation_echo(row, f"Created collection '{row.name}':", timezone_name)


# The before→after form a configured field takes when THIS turn moved it (#1946), and
# the word for a field that held nothing.  A was-state is the one thing a reply cannot
# read off the row — the edit overwrote it — so the ledger's recorded prior renders here
# beside the value that replaced it, and stating what it used to be becomes a COPY.
_CHANGED_VALUE = "was {before} → now {after}"
_NO_PRIOR_VALUE = "not set"

# Which configured line each recorded changed-field label renders on.  ``description``
# and ``extraction_prompt`` are deliberately absent: this record carries neither (the
# program is left out by #1799), so their priors have nowhere honest to go and are simply
# not stated.
_SCHEDULE_FIELD = "schedule"
_NOTIFY_FIELD = "notify"
_EXPIRES_FIELD = "expires_at"


def _configured_value(value: str, field: str, priors: Mapping[str, str | None]) -> str:
    """One configured value, rendered as ``was <old> → now <new>`` when this turn moved
    it and plainly when it did not.

    A prior that reads the SAME as the value renders plainly too: "was X → now X" is
    noise a reader has to rule out.  A field with no recorded prior renders exactly as
    it always did, which is what keeps every pre-#1946 event's render byte-identical."""
    if field not in priors:
        return value
    before = priors[field] or _NO_PRIOR_VALUE
    if before == value:
        return value
    return _CHANGED_VALUE.format(before=before, after=value)


def render_expiry_update_line(
    row: MemoryRow, timezone_name: str | None, priors: Mapping[str, str | None]
) -> str:
    """The end condition as a PLAIN ``collection_configure`` update states it (#2251): the
    row's own value, on the user's own clock (#2206, the ``render_applied_configuration``
    reading above) — never absent, which is what the update's echo used to be when it
    moved an expiry and never said so. ``was <old> → now <new>`` when this call changed it
    (#1946's convention, ``_configured_value``), plain otherwise; ``priors`` is `{}` when
    the call has no run id to key a recorded prior by, which renders plainly like every
    other untracked edit."""
    return _configured_value(_expires_value(row, timezone_name), _EXPIRES_FIELD, priors)


def _schedule_configured_value(
    row: MemoryRow, timezone_name: str | None, priors: Mapping[str, str | None]
) -> str:
    """The schedule's own was→now line (#1946's convention), with BOTH sides in plain
    words plus their exact rule (#2425) — not the generic ``_configured_value``, because
    that compares the "now" value (already rendered) against the recorded prior (still
    the bare rule text) byte-for-byte, which a plain-words "now" can never equal.  The
    recorded prior is the raw text the field held before this call (#1946), so the "was"
    side is built straight from it rather than off a row — there is no earlier row left
    to read it from."""
    after = describe_schedule_field(row, timezone_name)
    if _SCHEDULE_FIELD not in priors:
        return after
    before_raw = priors[_SCHEDULE_FIELD]
    if before_raw is None:
        before = _NO_PRIOR_VALUE
    else:
        before = _describe_rule_with_span(before_raw, _one_line(before_raw), timezone_name)
    return after if before == after else _CHANGED_VALUE.format(before=before, after=after)


def render_reinstantiation_echo(row: MemoryRow, timezone_name: str | None = None) -> str:
    """The re-render confirm-shape — render-at-update mirrors render-at-creation, so
    a collection whose prompt/schedule/notify just changed confirms back its NEW
    configuration, in the same shape the creation echo uses.  ``timezone_name``
    defaults to ``None`` (UTC) so a caller that predates #2206/#2251 renders
    byte-identically."""
    return _instantiation_echo(row, f"Re-configured collection '{row.name}':", timezone_name)


# ── Inert creation echo ───────────────────────────────────────────────────────

_INERT_ECHO = (
    "Set up collection '{name}' — storage only, no job yet:\n"
    "  description: {description}\n"
    "  status: inert (no prompt attached)\n"
    "It'll hold whatever gets written to it, but nothing runs against it until it has a "
    "prompt. Attach one with collection_configure(collection='{name}', prompt=<what the "
    "job's cycles should do>, schedule=<recurrence rule>)."
)


def render_inert_echo(row: MemoryRow) -> str:
    """The prompt-less creation echo: a collection with no ``extraction_prompt`` is
    INERT — a container that holds entries but has no job, so it never dispatches.
    The echo is honest about that (storage only, no prompt) and names how to give it
    a job — never claiming a job that doesn't exist (visible degradation over silent
    success)."""
    return _INERT_ECHO.format(name=row.name, description=row.description)
