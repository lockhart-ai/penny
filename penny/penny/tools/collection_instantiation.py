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
* the **trigger union** parse (``interval`` | ``run_at`` + ``max_runs``) and the
  ``expires_at`` end condition;
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
    '1. Set up the container first: collection_create(name=<slug>, intent="{query}") with '
    "NO skill — a storage-only collection nothing runs against yet.\n"
    "2. Walk me through getting the data ONCE, here in chat, so I actually do it (browse, "
    "extract, and collection_write the result into that collection).\n"
    "3. Save that run as a skill: skill_create(name=<title>, from_run=<that run's id>, "
    "steps=<range>).\n"
    "4. Attach it to make the collection do the job: collection_update(name=<slug>, "
    "skill=<title>, params={{…}}, interval=<seconds>, notify=<true/false>)."
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


# ── Trigger union (interval | run_at + max_runs) + end condition ──────────────


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
    """Parse an ISO-8601 datetime arg (``run_at`` / ``expires_at``) into a
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


def build_trigger(
    interval: int | None,
    run_at: str | None,
    max_runs: int | None,
    on_advance: str | None,
    min_interval: int | None,
    once_form_interval_seconds: int,
) -> Trigger:
    """Resolve the exclusive trigger union into a store-ready ``Trigger``.

    Exactly one form: ``interval`` (recurring — paces every ``interval`` seconds),
    OR ``run_at`` + ``max_runs`` (a delayed/one-shot schedule that starts at
    ``run_at`` and retires after ``max_runs`` runs), OR ``on_advance`` (wake when
    the named source LOG advances past this collection's cursor — the source-driven
    trigger, #1604).  Forms without a cadence arg (once, on_advance) are paced at
    ``once_form_interval_seconds`` (the dispatcher tick) so they are eligible each
    tick, their real gate (``run_at`` / the source frontier) deciding when they
    actually run; ``on_advance`` accepts an optional ``min_interval`` floor to cap a
    chatty source.  The collector requires a non-null cadence to ever run, so a
    never-firing trigger is refused here, not created silently (visible
    degradation)."""
    forms = [interval is not None, run_at is not None, on_advance is not None]
    if sum(forms) > 1:
        raise TriggerError(
            "Pick one trigger: interval (a recurring cadence), OR run_at + max_runs "
            "(a scheduled/one-shot run), OR on_advance (wake when a source log advances) "
            "— not more than one."
        )
    if min_interval is not None and on_advance is None:
        raise TriggerError(
            "min_interval only applies to an on_advance trigger — set on_advance=<source log> "
            "to use it, or use interval for a plain recurring cadence."
        )
    if interval is not None:
        if interval < 1:
            raise TriggerError("interval must be at least 1 second.")
        return Trigger(collector_interval_seconds=interval)
    if run_at is not None:
        if max_runs is None or max_runs < 1:
            raise TriggerError(
                "A run_at schedule needs max_runs (how many times to run, at least 1) — "
                "e.g. run_at='2026-07-20T09:00:00Z', max_runs=1 for a one-time reminder."
            )
        return Trigger(
            collector_interval_seconds=once_form_interval_seconds,
            run_at=parse_datetime(run_at, "run_at"),
            max_runs=max_runs,
        )
    if on_advance is not None:
        if min_interval is not None and min_interval < 1:
            raise TriggerError("min_interval must be at least 1 second.")
        # Slug the source to the canonical log name the store + cursor key use, so
        # the gate's ``log_name == source_log`` frontier check can never mismatch on
        # a raw-cased arg (``get`` slugs; the cursor is keyed on the slugged name).
        return Trigger(
            collector_interval_seconds=min_interval or once_form_interval_seconds,
            source_log=slug(on_advance),
        )
    raise TriggerError(
        "This collection has no trigger — set interval (seconds) for a recurring collector, "
        "run_at + max_runs for a scheduled/one-shot run, or on_advance=<source log> to wake "
        "when a source log advances."
    )


# ── Creation echo ─────────────────────────────────────────────────────────────


def humanize_interval(seconds: int | None) -> str:
    """Render a collector interval as a human cadence (e.g. '1h', '30m', '1d',
    'unset').  The single implementation, shared with the collection_update echo in
    ``memory_tools``."""
    if not seconds:
        return "unset"
    for unit_seconds, suffix in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds % unit_seconds == 0:
            return f"{seconds // unit_seconds}{suffix}"
    return f"{seconds}s"


def _trigger_line(row: MemoryRow) -> str:
    """The echo's one-line trigger summary — recurring cadence, the once-shaped
    ``runs at <run_at>, <n> time(s)`` schedule, or the ``on advance of <log>``
    source-driven trigger (#1604)."""
    if row.source_log is not None:
        return f"  trigger: on advance of {row.source_log}"
    if row.run_at is not None:
        times = "once" if row.max_runs == 1 else f"{row.max_runs} times"
        return f"  trigger: runs at {format_log_timestamp(row.run_at)}, {times}"
    return f"  trigger: every {humanize_interval(row.collector_interval_seconds)}"


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
        f"  intent: {row.intent}",
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
    "  intent: {intent}\n"
    "  status: inert (no skill attached)\n"
    "It'll hold whatever gets written to it, but nothing runs against it until you give it "
    "a skill. Teach me the routine once, save it with skill_create, then attach it with "
    "collection_update(name='{name}', skill=<title>, interval=<seconds>) to make it do "
    "something."
)


def render_inert_echo(row: MemoryRow) -> str:
    """The skill-less creation echo (#1629): a collection with no ``extraction_prompt``
    is INERT — a container that holds entries but has no job, so it never dispatches.
    The echo is honest about that (storage only, no skill) and names the two-step
    bootstrap that gives it a job (teach a skill, then adopt it via ``collection_update``)
    — never claiming a routine that doesn't exist (visible degradation over silent
    success)."""
    return _INERT_ECHO.format(name=row.name, intent=row.intent)
