"""The front door of collector creation (#1591, stage ⑤ of #1562 / epic #1554).

A collection is never authored with an inline procedure any more — it is an
*instantiation of a skill*: ``collection_set`` takes a ``skill`` (resolved by
name or meaning), binds its parameters from ``params``, renders the skill's
steps into the numbered TEXT ``extraction_prompt`` the collector runs, and stamps
it at creation.  This module holds the pure, DB-free pieces of that flow so they
are whole-render tested in isolation:

* the **skill-resolution union** (``SkillResolutionKind`` — MATCHED / AMBIGUOUS /
  NO_SKILL_FOUND / EMBED_FAILED) and its enumerated tool-result renders, including
  the #1471 "walk me through it once" elicitation;
* the **idempotency-at-birth** tombstone result (#1567) — the archived-duplicate
  confirm-shape, naming the retired row and the unarchive that revives it.  (Its
  active-duplicate sibling is gone: an active duplicate is no longer refused, it
  is UPDATED in place — see ``CollectionSetTool._same_job``, #1775);
* the **schedule** parse (one ``schedule`` arg, one grammar: an RRULE string,
  #1857) and the ``expires_at`` end condition (ISO or natural language), with
  ``render_schedule_clause`` rendering the stored rule back verbatim — it IS the
  input form;
* the **creation echo** (skill · params · schedule · notify · expiry · the rendered
  prompt), so the chat agent confirms back exactly what landed.

The orchestration (embed, resolve, validate parameters, dedup, create) lives on
``CollectionCreateTool`` in :mod:`penny.tools.memory_tools`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser
from dateutil.rrule import rrulestr
from pydantic import BaseModel

from penny.database.models import MemoryRow, Skill
from penny.database.skills import SkillParameter
from penny.datetime_utils import format_log_timestamp

# ── Skill resolution union ────────────────────────────────────────────────────


class SkillResolutionKind(StrEnum):
    """The closed set of outcomes when resolving the ``skill`` arg to a stored
    skill (classify-then-act, the enumerated-cases doctrine).  MATCHED proceeds to
    instantiation; AMBIGUOUS and NO_SKILL_FOUND are returned as tool results, never
    silently resolved; EMBED_FAILED is the transient-embedding escape."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NO_SKILL_FOUND = "no_skill_found"
    EMBED_FAILED = "embed_failed"


class SkillResolution(BaseModel):
    """One resolution outcome: the kind plus whichever payload it carries — the
    matched ``skill`` (MATCHED) or the ranked ``candidates`` (AMBIGUOUS)."""

    model_config = {"arbitrary_types_allowed": True}

    kind: SkillResolutionKind
    skill: Skill | None = None
    candidates: list[Skill] = []


_AMBIGUOUS_HEADER = 'I know a few skills close to "{query}" — I won\'t guess which you mean:'
_AMBIGUOUS_TAIL = (
    "To use one, call collection_set again with skill='<its exact name>'. If none of "
    "these is the process you mean, walk me through it once and I'll learn it as a new skill."
)

_NO_SKILL_FOUND = (
    "I don't know how to \"{query}\" yet — there's no skill for it, so there's nothing to "
    "instantiate. No schedule can be set up before the routine is learned. Two cases:\n"
    "a. The user already told you what one round needs — where to look, what to pull "
    "out, what to keep — then DO IT NOW, in this turn: browse, extract just the "
    "ONE value they want "
    "watched (only the price, not a whole name+hook+price blob — a multi-field blob "
    "changes whenever any part does and would false-alarm every cycle), and "
    "collection_write what you actually found — storage is created for you if it "
    "doesn't exist.\n"
    "b. They haven't — reply to the user now, no more tool calls this turn: tell them "
    "you'll learn it from one complete pass, and ask for the whole routine in ONE "
    "message (where to look, what to pull out, what to keep), modelling the example "
    "from what they already said. When it arrives, run case a.\n"
    "Either way the routine is learned automatically as a skill from that round — a "
    "learned notice will tell you the moment it exists; then attach it and set any "
    "schedule or notify they asked for in ONE call: collection_set(name=<the "
    "collection>, skill=<its name>, schedule=..., notify=...)."
)


def render_ambiguous(query: str, candidates: list[Skill]) -> str:
    """SKILL_AMBIGUOUS: the ranked candidates plus how to narrow (pass the exact
    name) or teach a new one — never a silent pick."""
    lines = [_AMBIGUOUS_HEADER.format(query=query)]
    lines.extend(f"{i}. {skill.name} — {skill.intent}" for i, skill in enumerate(candidates, 1))
    lines.append(_AMBIGUOUS_TAIL)
    return "\n".join(lines)


def render_no_skill_found(query: str) -> str:
    """NO_SKILL_FOUND: the #1471 elicitation — ignorance becomes the trigger to
    demonstrate-and-promote, with the exact next call named."""
    return _NO_SKILL_FOUND.format(query=query)


# ── Parameter validation ──────────────────────────────────────────────────────

_UNBOUND_PARAMETERS = (
    "Can't instantiate '{skill}': these required parameters aren't bound:\n{listed}\n"
    "Pass them in params (e.g. params={{{example}}}), then call collection_set again."
)


def _parameter_line(parameter: SkillParameter) -> str:
    """One unbound-parameter line — its semantic name and, when set, its description
    (so a stale/unknown name is answered with the CURRENT names + what they mean)."""
    if parameter.description:
        return f"  - {parameter.name}: {parameter.description}"
    return f"  - {parameter.name}"


def render_unbound_parameters(skill_name: str, missing: list[SkillParameter]) -> str:
    """The parameter-validation error (#1668): name every unbound required parameter
    with its description (the current, semantic names — so a rebind with a stale name
    is answered actionably) and show the exact ``params`` shape to supply
    (actionable-error contract)."""
    listed = "\n".join(_parameter_line(parameter) for parameter in missing)
    example = ", ".join(f"'{parameter.name}': <value>" for parameter in missing)
    return _UNBOUND_PARAMETERS.format(skill=skill_name, listed=listed, example=example)


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
    "- FREQ=DAILY;BYHOUR=8 — once a day at 08:00 UTC\n"
    "- FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9 — weekdays at 09:00 UTC\n"
    "- DTSTART:20260720T090000Z\\nFREQ=DAILY;COUNT=1 — once, at that moment, then it retires"
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


def render_schedule_clause(row: MemoryRow) -> str:
    """The mechanism's schedule rendered back VERBATIM — display form == invocation
    form (#1857): what a surface shows (the self-state mechanisms line,
    ``memory_metadata``, the creation echo) is the stored rule itself, so it copies
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
# ``memory_metadata``'s ``schedule: none`` convention, so no surface emits a blank.
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


def parse_expires_at(
    value: str, timezone_name: str | None, now: datetime | None = None
) -> datetime:
    """Parse the ``expires_at`` end condition into a UTC-aware datetime (#1857).

    ISO first — an exact time is an exact time — then ``dateparser`` over the user's
    own words, read IN THE USER'S TIMEZONE.  The timezone is what makes the words mean
    what the user meant: '10pm today' is 10pm where they are, and reading it as UTC put
    a watch's end hours away from the evening it was asked for.  ``timezone_name`` is
    the user's IANA zone (``None`` on a fresh install → UTC) and ``now`` the moment
    relative words count from, both passed in rather than read from ambient state.
    Text neither reading answers raises the teaching :class:`ScheduleError`.
    """
    iso = _parse_iso_datetime(value)
    if iso is not None:
        return iso
    if now is None:
        now = datetime.now(UTC)
    spoken = _parse_spoken_datetime(value, timezone_name, now)
    if spoken is not None:
        return spoken
    raise ScheduleError(_EXPIRES_TEACHING.format(value=value))


def _parse_iso_datetime(value: str) -> datetime | None:
    """The ISO-8601 reading, or ``None`` when the text isn't one.  A naive value is
    read as UTC, which is what the column holds."""
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _parse_spoken_datetime(value: str, timezone_name: str | None, now: datetime) -> datetime | None:
    """The natural-language reading in the user's timezone, or ``None`` when
    ``dateparser`` finds no time in the words."""
    zone = _user_zone(timezone_name)
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


def _user_zone(timezone_name: str | None) -> ZoneInfo:
    """The user's zone, or UTC when there is no profile / the stored zone is unknown —
    the same fallback the current-time anchor takes, so the clock the model reads and
    the clock its words are read against can't disagree."""
    if timezone_name is None:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


# ── Creation echo ─────────────────────────────────────────────────────────────


def _schedule_line(row: MemoryRow) -> str:
    """The echo's one-line schedule summary — the stored rule verbatim when the
    collection has one, or ``schedule: none`` for an inert/unscheduled one (#1666;
    #1857, display form == invocation form)."""
    return f"  schedule: {render_schedule_field(row)}"


def _params_line(params: dict[str, str]) -> str:
    if not params:
        return "  params: none"
    rendered = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"  params: {rendered}"


def _expires_line(row: MemoryRow) -> str:
    if row.expires_at is None:
        return "  expires: never"
    return f"  expires: {format_log_timestamp(row.expires_at)}"


# ── The plain-language lead line (what the collector WILL do, #1658) ───────────
#
# The instantiation echoes LEAD with one deterministic English sentence composed
# from the STRUCTURED fields — skill · cadence · target · notify — so the model (and
# the user it mirrors it to) reads what the collector will actually do without
# confabulating it.  Template-method over the schedule × the notify flag; the
# detailed field-by-field echo stays below it.


def _lead_cadence_phrase(row: MemoryRow) -> str:
    """The 'when it runs' clause, built off the SAME stored rule
    ``render_schedule_clause`` reads — a bare 'I'll run' when the collection carries no
    schedule at all (defensive — a skill-backed collection always has one)."""
    if not has_schedule(row):
        return "I'll run"
    times = f" ({row.max_runs} times)" if row.max_runs not in (None, 1) else ""
    return f"On the schedule {render_schedule_clause(row)}{times} I'll run"


def _lead_notify_tail(row: MemoryRow) -> str:
    """The 'what it does with a change' clause — the notify flag in plain words."""
    if row.notify:
        return "message you when something changes."
    return "quietly store what it finds."


def _lead_line(row: MemoryRow, skill_name: str) -> str:
    """One deterministic English sentence: cadence + skill + target + notify —
    e.g. "Every 900 seconds I'll run 'watch-a-page' against 'aurora-prices' and
    message you when something changes." / "…and quietly store what it finds."."""
    return (
        f"{_lead_cadence_phrase(row)} '{skill_name}' against '{row.name}' and "
        f"{_lead_notify_tail(row)}"
    )


def _instantiation_echo(
    row: MemoryRow, skill_name: str, params: dict[str, str], headline: str
) -> str:
    """The shared instantiation confirm-shape — a plain-language LEAD line (what the
    collector will do) over a ``headline`` and skill · bound params · schedule ·
    notify · expiry · the full rendered ``extraction_prompt``.  Both the creation
    echo (#1591) and the re-render echo (#1620) compose it, so a freshly created
    collection and a re-rendered one confirm back the same fields."""
    prompt = (row.extraction_prompt or "").replace("\n", "\n    ")
    lines = [
        _lead_line(row, skill_name),
        headline,
        f"  description: {row.description}",
        f"  skill: {skill_name}",
        _params_line(params),
        _schedule_line(row),
        f"  notify: {row.notify}",
        _expires_line(row),
        "  extraction_prompt: |",
        f"    {prompt}",
    ]
    return "\n".join(lines)


def render_creation_echo(row: MemoryRow, skill_name: str, params: dict[str, str]) -> str:
    """The structured creation echo — skill, bound params, schedule, notify, expiry,
    and the full rendered ``extraction_prompt`` — so the chat agent confirms back
    exactly what landed without confabulating a field."""
    return _instantiation_echo(
        row, skill_name, params, f"Created collection '{row.name}' from skill '{skill_name}':"
    )


def render_applied_configuration(row: MemoryRow) -> str | None:
    """What a collection is now CONFIGURED to do, read off the row itself (#1869) — the
    record the run-end narration frame carries.

    The row IS the record: it holds the routine it runs, the values that routine is
    pointed at, its cadence, its end condition and whether it tells the user, all written
    by the call that configured it.  So a turn that no longer supplies the routine or its
    values has somewhere to READ what it just set up, instead of narrating from a memory
    of arguments the framework filled in for it.

    ``None`` when the collection carries no routine — an inert container has no
    configuration to state, and saying it was set up would be the claim the honest-failure
    rule exists to stop.

    Deliberately the echo's fields WITHOUT the rendered program (#1799): what this is read
    for is a description a person can act on, and a block of tool calls in front of that
    request is a block that gets read aloud."""
    if row.skill_name is None:
        return None
    params = skill_params(row)
    return "\n".join(
        [
            _lead_line(row, row.skill_name),
            f"  collection: {row.name}",
            f"  skill: {row.skill_name}",
            _params_line(params),
            _schedule_line(row),
            f"  notify: {row.notify}",
            _expires_line(row),
        ]
    )


def skill_params(row: MemoryRow) -> dict[str, str]:
    """The values a collection's routine is bound to, off its own provenance column —
    ``{}`` when it has none.  One reading of that column, shared by the tool that rebinds
    it and the render that states it back."""
    return json.loads(row.skill_params) if row.skill_params else {}


def render_reinstantiation_echo(row: MemoryRow, skill_name: str, params: dict[str, str]) -> str:
    """The re-render confirm-shape (#1620) — render-at-update mirrors
    render-at-creation, so a refreshed / rebound / swapped / adopted collection
    confirms back its NEW program: the skill it now runs, the bound params, and the
    freshly rendered routine, in the same shape the creation echo uses."""
    return _instantiation_echo(
        row, skill_name, params, f"Re-rendered collection '{row.name}' from skill '{skill_name}':"
    )


# ── Inert creation echo (#1629) ───────────────────────────────────────────────

_INERT_ECHO = (
    "Set up collection '{name}' — storage only, no job yet:\n"
    "  description: {description}\n"
    "  status: inert (no skill attached)\n"
    "It'll hold whatever gets written to it, but nothing runs against it until it has a "
    "skill. Next: run one round into it — read the source, then collection_write what "
    "you actually find into '{name}'; the routine is learned automatically as a skill "
    "from that round. If you don't have the routine yet, reply and ask the user for it "
    "in one message — don't guess."
)


def render_inert_echo(row: MemoryRow) -> str:
    """The skill-less creation echo (#1629): a collection with no ``extraction_prompt``
    is INERT — a container that holds entries but has no job, so it never dispatches.
    The echo is honest about that (storage only, no skill) and names the two-step
    bootstrap that gives it a job (teach a skill, then adopt it via ``collection_set``)
    — never claiming a routine that doesn't exist (visible degradation over silent
    success)."""
    return _INERT_ECHO.format(name=row.name, description=row.description)
