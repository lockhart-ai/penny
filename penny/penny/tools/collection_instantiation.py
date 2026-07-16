"""The front door of collector creation (#1591, stage ⑤ of #1562 / epic #1554).

A collection is never authored with an inline procedure any more — it is an
*instantiation of a skill*: ``collection_create`` takes a ``skill`` (resolved by
name or meaning), binds its parameter holes from ``params``, renders the skill's
steps into the numbered TEXT ``extraction_prompt`` the collector runs, and stamps
it at creation.  This module holds the pure, DB-free pieces of that flow so they
are whole-render tested in isolation:

* the **skill-resolution union** (``SkillResolutionKind`` — MATCHED / AMBIGUOUS /
  NO_SKILL_FOUND / EMBED_FAILED) and its enumerated tool-result renders, including
  the #1471 "walk me through it once" elicitation;
* the **idempotency-at-birth** results (#1567) — the active-duplicate refusal and
  the tombstone-duplicate confirm-shaped result, each naming the existing row and
  the deliberate override;
* the **trigger** parse (one ``trigger`` arg, three enumerated forms: ``every
  <seconds>`` | ``once at <ISO> [xN]`` | ``on advance of <log>``, #1631) and the
  ``expires_at`` end condition, with ``render_trigger_clause`` rendering the stored
  trigger back AS its copyable input form;
* the **creation echo** (skill · params · trigger · notify · expiry · the rendered
  prompt), so the chat agent confirms back exactly what landed.

The orchestration (embed, resolve, validate holes, dedup, create) lives on
``CollectionCreateTool`` in :mod:`penny.tools.memory_tools`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel

from penny.database.memory.types import slug
from penny.database.models import MemoryRow, Skill
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
    "To use one, call collection_create again with skill='<its exact name>'. If none of "
    "these is the process you mean, walk me through it once and I'll learn it as a new skill."
)

_NO_SKILL_FOUND = (
    "I don't know how to \"{query}\" yet — there's no skill for it, so there's nothing to "
    "instantiate. Here's how we teach one:\n"
    '1. Set up the container first: collection_create(name=<slug>, description="{query}") '
    "with NO skill — a storage-only collection nothing runs against yet.\n"
    "2. Walk me through getting the data ONCE, here in chat, so I actually do it: browse, "
    "extract just the ONE value you want watched (pull out only the price, not a whole "
    "name+hook+price blob — a multi-field blob changes whenever any part does and would "
    "false-alarm every cycle), and collection_write that value into the collection.\n"
    "3. Save that run as a skill: skill_create(name=<title>, from_run=<that run's id>, "
    "steps=<range>).\n"
    "4. Attach it to make the collection do the job: collection_update(name=<slug>, "
    'skill=<title>, params={{…}}, trigger="every <seconds>", notify=<true/false>).'
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


# ── Hole validation ───────────────────────────────────────────────────────────

_UNBOUND_HOLES = (
    "Can't instantiate '{skill}': the required parameter(s) {missing} aren't bound. Pass "
    "them in params (e.g. params={{{example}}}), then call collection_create again."
)


def render_unbound_holes(skill_name: str, missing: list[str]) -> str:
    """The hole-validation error: name every unbound required parameter and show
    the exact ``params`` shape to supply (actionable-error contract)."""
    named = ", ".join(missing)
    example = ", ".join(f"'{name}': <value>" for name in missing)
    return _UNBOUND_HOLES.format(skill=skill_name, missing=named, example=example)


# ── Idempotency at birth (#1567) ──────────────────────────────────────────────

_ACTIVE_DUPLICATE = (
    "Already have a collection for this: '{name}' (active) — it covers the same thing, so I "
    "didn't create a second one. Reuse it: read it with collection_read_latest('{name}'), or "
    "adjust it with collection_update(name='{name}', ...). If this really is a distinct task, "
    "create it deliberately with collection_create(..., create_anyway=true)."
)

_TOMBSTONE_DUPLICATE = (
    "There's an archived collection for this: '{name}' (archived {archived_at}) — I didn't "
    "create a duplicate. Bring it back with collection_unarchive('{name}') to resume it, or "
    "start a fresh one deliberately with collection_create(..., create_anyway=true)."
)


def render_active_duplicate(row: MemoryRow) -> str:
    """The active-duplicate refusal (#1567): name the live collection and make
    reuse the easy path, deliberate re-creation the explicit one."""
    return _ACTIVE_DUPLICATE.format(name=row.name)


def render_tombstone_duplicate(row: MemoryRow) -> str:
    """The tombstone-duplicate confirm-shaped result (#1567): surface the archived
    row and its archive time; unarchive or a deliberate override, never a silent
    proceed.  The archive timestamp is ``updated_at`` (stamped at archive)."""
    return _TOMBSTONE_DUPLICATE.format(
        name=row.name, archived_at=format_log_timestamp(row.updated_at)
    )


# ── Trigger: one arg, three enumerated forms (#1631) ─────────────────────────


class TriggerError(Exception):
    """An actionable trigger/end-condition parse or validation failure — the tool
    surfaces ``str(self)`` as the failed result."""


class Trigger(BaseModel):
    """The parsed, store-ready trigger: the cadence the collector paces on plus the
    optional once-shaped overlay (``run_at`` + ``max_runs``) or on_advance overlay
    (``source_log``)."""

    collector_interval_seconds: int
    run_at: datetime | None = None
    max_runs: int | None = None
    source_log: str | None = None


def parse_datetime(value: str, field: str) -> datetime:
    """Parse an ISO-8601 datetime arg (a trigger time / ``expires_at``) into a
    UTC-aware datetime; a naive value is assumed UTC.  Raises an actionable
    ``TriggerError`` naming the field and the accepted shape."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise TriggerError(
            f"Couldn't read {field}={value!r} — write it as an ISO-8601 datetime like "
            "'2026-07-20T14:00:00Z' (or 'YYYY-MM-DD HH:MM')."
        ) from None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


_EVERY_PREFIX = "every "
_ONCE_PREFIX = "once at "
_ON_ADVANCE_PREFIX = "on advance of "

# The reject-and-teach failure for an unrecognised trigger shape (#1631): name the
# three enumerated forms so the model rewrites to one of them instead of inventing a
# fourth.  Each example is a copyable input (display form == invocation form).  The
# closing omission line (#1646) names the other valid move — leave the trigger out
# entirely for a storage-only collection — since a blank trigger now coerces to omitted
# and this rejection is only reached for genuinely garbled (non-blank) input.
_TRIGGER_TEACHING = (
    "I couldn't read the trigger '{trigger}'. Set it to one of these three forms "
    "(copy the shape exactly):\n"
    "- every <seconds> — a recurring cadence (e.g. every 3600 for hourly)\n"
    "- once at <ISO datetime> [xN] — run at a time, optionally N times "
    "(e.g. once at 2026-07-20T09:00:00Z, or once at 2026-07-20T09:00:00Z x3)\n"
    "- on advance of <log> — wake when a source log gets a new entry "
    "(e.g. on advance of browse-results)\n"
    "Or leave the trigger out entirely for a storage-only collection."
)


def parse_trigger(trigger: str, once_form_interval_seconds: int) -> Trigger:
    """Parse the single ``trigger`` arg into a store-ready ``Trigger`` — classify by
    prefix, then validate the form (the enumerated-cases doctrine).  Three forms:
    ``every <seconds>`` (a recurring cadence), ``once at <ISO> [xN]`` (a delayed /
    one-shot schedule, N defaulting to 1), ``on advance of <log>`` (wake when the named
    source LOG advances).  The rendered clause IS this input (``render_trigger_clause``),
    so a displayed trigger copies straight back as the arg and round-trips.  An
    unrecognised shape raises the teaching ``TriggerError`` naming all three."""
    text = trigger.strip()
    lowered = text.lower()
    if lowered.startswith(_EVERY_PREFIX):
        return _parse_every(text[len(_EVERY_PREFIX) :].strip())
    if lowered.startswith(_ONCE_PREFIX):
        return _parse_once(text[len(_ONCE_PREFIX) :].strip(), once_form_interval_seconds)
    if lowered.startswith(_ON_ADVANCE_PREFIX):
        return _parse_on_advance(
            text[len(_ON_ADVANCE_PREFIX) :].strip(), once_form_interval_seconds
        )
    raise TriggerError(_TRIGGER_TEACHING.format(trigger=trigger))


def _parse_every(rest: str) -> Trigger:
    """``every <seconds>`` — a recurring cadence.  A non-integer or non-positive value
    is an actionable refusal naming the shape."""
    if not rest.isdigit():
        raise TriggerError(
            f"'every {rest}' needs a whole number of seconds — e.g. every 3600 for hourly."
        )
    seconds = int(rest)
    if seconds < 1:
        raise TriggerError("A recurring cadence must be at least 1 second (every <seconds>).")
    return Trigger(collector_interval_seconds=seconds)


def _parse_once(rest: str, once_form_interval_seconds: int) -> Trigger:
    """``once at <ISO> [xN]`` — a delayed / one-shot schedule that starts at the ISO
    time and retires after N runs (N defaults to 1).  Paced at the dispatcher tick so
    it is eligible each tick; its real gate is ``run_at``, not the clock."""
    iso, max_runs = _split_run_count(rest)
    return Trigger(
        collector_interval_seconds=once_form_interval_seconds,
        run_at=parse_datetime(iso, "the time in 'once at <time>'"),
        max_runs=max_runs,
    )


def _split_run_count(rest: str) -> tuple[str, int]:
    """Split an optional ``xN`` repeat suffix off a ``once at`` body (``N`` defaults to
    1 — a one-shot).  A non-positive N is refused."""
    head, sep, tail = rest.rpartition(" x")
    if sep and tail.isdigit():
        count = int(tail)
        if count < 1:
            raise TriggerError("The run count in 'once at <time> xN' must be at least 1.")
        return head.strip(), count
    return rest, 1


def _parse_on_advance(rest: str, once_form_interval_seconds: int) -> Trigger:
    """``on advance of <log>`` — wake when the named source LOG advances past this
    collection's cursor (#1604).  Paced at the dispatcher tick (the source frontier is
    the real gate); the name is slugged to the canonical store/cursor key so the
    gate's frontier check can't mismatch on raw casing.  The tool validates the name
    is an existing log (``validate_source_log``)."""
    if not rest:
        raise TriggerError(
            "'on advance of <log>' needs a source log name — e.g. on advance of browse-results."
        )
    return Trigger(collector_interval_seconds=once_form_interval_seconds, source_log=slug(rest))


def render_trigger_clause(row: MemoryRow) -> str:
    """The mechanism's trigger rendered AS the copyable ``trigger`` input — display
    form == invocation form (#1631), the render-teaches-the-call property: the clause a
    surface shows (the self-state mechanisms line, ``memory_metadata``, the creation
    echo) is exactly what ``parse_trigger`` accepts, so it copies straight back and
    round-trips to the same stored config.  ``on advance of <log>`` · ``once at <ISO>
    [xN]`` (the ``xN`` suffix only when it repeats more than once) · ``every
    <seconds>``.  Built off the SAME prefix constants ``parse_trigger`` classifies on, so
    display and invocation can't structurally diverge."""
    if row.source_log is not None:
        return f"{_ON_ADVANCE_PREFIX}{row.source_log}"
    if row.run_at is not None:
        when = row.run_at if row.run_at.tzinfo is not None else row.run_at.replace(tzinfo=UTC)
        suffix = f" x{row.max_runs}" if row.max_runs not in (None, 1) else ""
        return f"{_ONCE_PREFIX}{when.isoformat()}{suffix}"
    return f"{_EVERY_PREFIX}{row.collector_interval_seconds}"


# ── Creation echo ─────────────────────────────────────────────────────────────


def _trigger_line(row: MemoryRow) -> str:
    """The echo's one-line trigger summary — the copyable ``trigger`` clause (#1631,
    display form == invocation form)."""
    return f"  trigger: {render_trigger_clause(row)}"


def _params_line(params: dict[str, str]) -> str:
    if not params:
        return "  params: none"
    rendered = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"  params: {rendered}"


def _expires_line(row: MemoryRow) -> str:
    if row.expires_at is None:
        return "  expires: never"
    return f"  expires: {format_log_timestamp(row.expires_at)}"


def _instantiation_echo(
    row: MemoryRow, skill_name: str, params: dict[str, str], headline: str
) -> str:
    """The shared instantiation confirm-shape — a ``headline`` over skill · bound
    params · trigger · notify · expiry · the full rendered ``extraction_prompt``.
    Both the creation echo (#1591) and the re-render echo (#1620) compose it, so a
    freshly created collection and a re-rendered one confirm back the same fields."""
    prompt = (row.extraction_prompt or "").replace("\n", "\n    ")
    lines = [
        headline,
        f"  description: {row.description}",
        f"  skill: {skill_name}",
        _params_line(params),
        _trigger_line(row),
        f"  notify: {row.notify}",
        _expires_line(row),
        "  extraction_prompt: |",
        f"    {prompt}",
    ]
    return "\n".join(lines)


def render_creation_echo(row: MemoryRow, skill_name: str, params: dict[str, str]) -> str:
    """The structured creation echo — skill, bound params, trigger, notify, expiry,
    and the full rendered ``extraction_prompt`` — so the chat agent confirms back
    exactly what landed without confabulating a field."""
    return _instantiation_echo(
        row, skill_name, params, f"Created collection '{row.name}' from skill '{skill_name}':"
    )


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
    "It'll hold whatever gets written to it, but nothing runs against it until you give it "
    "a skill. Teach me the routine once, save it with skill_create, then attach it with "
    "collection_update(name='{name}', skill=<title>, trigger=\"every <seconds>\") to make "
    "it do something."
)


def render_inert_echo(row: MemoryRow) -> str:
    """The skill-less creation echo (#1629): a collection with no ``extraction_prompt``
    is INERT — a container that holds entries but has no job, so it never dispatches.
    The echo is honest about that (storage only, no skill) and names the two-step
    bootstrap that gives it a job (teach a skill, then adopt it via ``collection_update``)
    — never claiming a routine that doesn't exist (visible degradation over silent
    success)."""
    return _INERT_ECHO.format(name=row.name, description=row.description)
