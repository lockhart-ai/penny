"""Pydantic arg models for the memory tool surface.

Each tool validates its kwargs through one of these models as its first line,
per the Pydantic-everywhere rule. Most read tools accept ``k: int | None``
meaning "no cap — return every entry" when omitted; this matches the access
layer's signature.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    ValidationInfo,
    model_validator,
)

from penny.constants import PennyConstants
from penny.text_validity import (
    require_non_blank_description,
    require_non_blank_log_content,
    require_non_degenerate_content,
)
from penny.tools.models import ToolArgs

# Models occasionally substitute Unicode dashes (U+2010–U+2015) for ASCII
# hyphen-minus (U+002D) when emitting memory names — gpt-oss has been
# observed writing ``"board‑games"`` for ``"board-games"``.
# The visual is identical but the string compares unequal, so memory-keyed
# tools (``collection_write``, ``log_read``, etc.) silently failed
# with refusals or "memory not found" errors.  Normalise on the way in so
# the rest of the stack sees a single canonical form.
_UNICODE_DASHES = "‐‑‒–—―−"


def _normalize_dashes(value: object) -> object:
    if not isinstance(value, str):
        return value
    if not any(ch in value for ch in _UNICODE_DASHES):
        return value
    out = value
    for ch in _UNICODE_DASHES:
        out = out.replace(ch, "-")
    return out


MemoryName = Annotated[str, BeforeValidator(_normalize_dashes)]


def _reject_nonpositive_count(value: int | None) -> int | None:
    """Reject a read count of zero (or negative) with an actionable message.

    ``k``/``n`` cap a read; ``None`` means "no cap — every entry".  A model that
    wants *all* entries sometimes guesses ``k=0`` (reading it as "unlimited"),
    but ``.limit(0)`` returns **zero** rows — so the model sees an empty memory
    and wrongly concludes it's empty (observed: the skills collector read
    ``collection_list(k=0)``, saw no skills, and wrote a duplicate instead
    of updating the existing one).  Fail loudly with the fix rather than silently
    return nothing."""
    if value is not None and value < 1:
        raise ValueError(
            f"k={value} would read zero entries — a read count must be at least 1. "
            "Omit k entirely to read every entry."
        )
    return value


ReadCount = Annotated[int | None, AfterValidator(_reject_nonpositive_count)]


# What each optional string argument on the collection surface actually accepts, so a
# blank rejection can NAME the form instead of only refusing.  Total over the fields
# typed ``OptionalText`` / ``OptionalSkill`` on the three collection arg models (pinned
# by a test), so the lookup indexes directly: a field declaring one of those types with
# no entry here is a programming error, not a case to absorb behind a vague default.
OPTIONAL_ARG_FORMS: dict[str, str] = {
    "description": "what the collection is for, in the user's own words",
    "prompt": "the instructions this job's cycles follow, as free text",
    # NOT a second copy of the grammar — ``_SCHEDULE_TEACHING`` (in
    # collection_instantiation) is where that is authored, and a garbled schedule gets
    # it verbatim.  A BLANK one needs the one shape that makes the fix obvious, so this
    # stays a worked example.
    "schedule": 'one RRULE line, e.g. "FREQ=HOURLY" for hourly',
    "expires_at": 'when it ends, e.g. 2026-03-01T09:00:00Z or "in two weeks"',
}

# No trailing period: the arg-validation envelope appends ". Call <tool>(<valid
# arguments>) again." after each field's reason.
_BLANK_OPTIONAL_ARG = (
    "an empty string sets nothing — omit {field} entirely to leave it unset, or pass {form}"
)


def _reject_blank_optional(value: object, info: ValidationInfo) -> object:
    """Refuse a blank/whitespace-only optional string, naming the field and its form.

    Models routinely fill an optional argument they mean to omit with ``""`` (a chronic
    gpt-oss habit).  That used to be COERCED to ``None`` — read as "omitted" — which made
    the two indistinguishable and turned a partially-understood request into silent
    partial compliance: a watch asked for "until 10pm tonight" went live with
    ``expires_at=""`` swallowed, unbounded, every other field correct, and nothing in the
    reply or the record marking the difference (#1776; supersedes the #1646 coercion).

    So a blank is a **teaching rejection** at the arg gate instead — before ``execute``,
    so nothing is created or reconfigured on a call the model has to redo — carrying both
    halves of an actionable failure: what went wrong (a blank sets nothing) and how to fix
    it (omit the argument, or pass the named form).  Uniform across every optional string
    on the surface rather than special-cased to the one field it was observed on.
    """
    if isinstance(value, str) and not value.strip():
        field = str(info.field_name)
        raise ValueError(_BLANK_OPTIONAL_ARG.format(field=field, form=OPTIONAL_ARG_FORMS[field]))
    return value


# Optional text on the collection surface: a blank is refused with the teaching
# rejection above, never silently read as "omitted".
OptionalText = Annotated[str | None, BeforeValidator(_reject_blank_optional)]

# `find`'s own blank-`memory` refusal (#2289) — kept SEPARATE from
# `_reject_blank_optional`/`OPTIONAL_ARG_FORMS` (the `collection_configure` surface's
# shared table, whose own test asserts it is total over that ONE tool's fields):
# `memory` belongs to a different tool, so sharing the table would make that
# invariant false. A blank still reads as "search everything", the one thing it
# would be impossible to tell apart from simply omitting the argument, so it's
# refused rather than silently taken either way.
_BLANK_MEMORY_ARG = (
    "an empty string sets nothing — omit memory entirely to search everything, or "
    "pass the exact name of one collection or log"
)


def _reject_blank_memory(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        raise ValueError(_BLANK_MEMORY_ARG)
    return value


# An optional memory name to narrow `find` to: dashes normalised first (a memory
# name may be slug-ish), then its own blank refusal above.
OptionalMemoryName = Annotated[
    str | None, BeforeValidator(_reject_blank_memory), BeforeValidator(_normalize_dashes)
]


# ── Annotated validator types ─────────────────────────────────────────────────
# One Annotated type per validation concern, wrapping a shared predicate, so a
# field declares its rule by *type* — no per-field @field_validator methods.  The
# required variants raise on a bad value; the optional variants (``OptionalText`` /
# ``OptionalSkill``, above) refuse a blank with the teaching rejection and apply no
# further rule when the argument is genuinely omitted.


def _reject_system_log(value: str) -> str:
    """Raise if ``value`` names a framework-managed system log.

    The four ``SYSTEM_LOGS`` (conversation + run history) are written only by
    Python side-effects; an agent appending to one would forge a turn or audit
    row.  A pure constant lookup, so it's an arg-validation refusal — not a
    runtime decision.
    """
    if value in PennyConstants.SYSTEM_LOGS:
        raise ValueError(
            f"'{value}' is a system log written automatically every turn "
            "(conversation and run history) — you can't append to it. Use a "
            "collection or a log you created for your own notes."
        )
    return value


# A log name an agent may append to: dashes normalised, system logs refused.
AppendableLogName = Annotated[
    str, BeforeValidator(_normalize_dashes), AfterValidator(_reject_system_log)
]

NonBlankDescription = Annotated[str, AfterValidator(require_non_blank_description)]
CollectionContent = Annotated[str, AfterValidator(require_non_degenerate_content)]
NonBlankLogContent = Annotated[str, AfterValidator(require_non_blank_log_content)]


# ── Metadata ────────────────────────────────────────────────────────────────


class CollectionCreateArgs(ToolArgs):
    """Args for ``collection_configure`` — the job front door, or a PROMPT-LESS
    INERT container (#2455 naive baseline).

    A collection is storage plus an OPTIONAL job.  With a ``prompt`` the collection
    is a JOB: the free text becomes the collector's ``extraction_prompt`` verbatim —
    what the job's cycles follow each time they run — and a ``schedule`` schedules
    it.  WITHOUT a ``prompt`` the collection is INERT: storage only — no
    ``extraction_prompt``, no schedule, no ``notify`` — so nothing runs against it
    until a prompt is attached later via ``collection_configure``.  A job-shaped arg
    (a ``schedule`` / ``notify`` / ``expires_at``) alongside a prompt-less create is
    refused, since an inert container has no job to describe.

    ``description`` (required, non-blank) is what the collection is for, in the user's
    own words — the goal it serves and the collection's routing/dedup meaning anchor.
    ``name`` is the unique slug.

    **A blank string on an optional arg is REFUSED, not read as "not set"** (#1776,
    superseding #1646's coercion).  ``prompt`` / ``schedule`` / ``expires_at`` reject
    ``""`` at the arg gate (``OptionalText``) with a teaching error
    naming the field and its accepted form, BEFORE any routing or parsing — so a model
    that fills an optional field with ``""`` (a chronic gpt-oss habit) is corrected
    rather than silently given the omitted behaviour, which turned a half-understood
    request into a half-built mechanism nothing marked as such.

    The **schedule** (job path only) is ONE argument with ONE grammar (#1857,
    ``parse_schedule``): an RRULE line such as ``"FREQ=HOURLY"`` or
    ``"FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9"``, optionally preceded by a
    ``DTSTART:`` line saying when it starts.  Its ``COUNT=`` / ``UNTIL=`` parts are
    lifted into ``max_runs`` / ``expires_at`` at parse.  A schedule ``dateutil``
    can't read is refused with a teaching error showing the grammar.  ``expires_at``
    (optional) is the end condition — an ISO datetime or when it ends in the user's
    own words — and the watch archives itself when it passes.  ``notify`` (default
    false) makes the collection tell the user about new/changed entries; an omission
    stays silent, so it can never accidentally notify.
    """

    name: MemoryName
    description: NonBlankDescription
    # The job's instructions, written as free text by the model; OMITTED yields an
    # INERT storage-only collection (#2455).  A blank ``""`` is refused, never read as
    # omitted (#1776).  OptionalText == update's prompt.
    prompt: OptionalText = None
    # Schedule — one arg, one grammar (parse_schedule, #1857): an RRULE line, with an
    # optional leading DTSTART line.  Its render (render_schedule_clause) IS this input
    # form, verbatim.  OptionalText: a blank ``""`` is refused at the gate (#1776)
    # rather than reaching the parser as a garbled schedule.
    schedule: OptionalText = None
    # End condition (optional) — an ISO datetime or when it ends in plain words; the
    # collection archives itself when it passes.  Parsed in the tool (actionable error
    # on a bad value).  OptionalText: a blank ``""`` is refused at the gate (#1776) —
    # the observed silent drop, where a bounded watch asked for went live unbounded.
    expires_at: OptionalText = None
    # Notify-on-new (emission-as-property, #1557): true when the user asked to be
    # told / kept posted / alerted about new entries.  Defaults false (silent).
    notify: bool = False


class LogCreateArgs(ToolArgs):
    """Args for ``log_create``.

    Logs are append-only streams of events (messages, browse results,
    measurements).  No extraction_prompt — logs are inputs, not curated
    outputs.  No interval — logs don't have a collector.
    """

    log: MemoryName
    description: NonBlankDescription


class MemoryNameArgs(ToolArgs):
    """One-field args for ``memory_get`` — the shape-agnostic metadata lookup."""

    memory: MemoryName


class CollectionNameArgs(ToolArgs):
    """One-field args for ``collection_archive`` / ``collection_unarchive`` — the
    archive pair names its target with the collection-surface argument."""

    collection: MemoryName


class CollectionSetArgs(ToolArgs):
    """Args for ``collection_configure`` — the ONE idempotent create-or-update entry
    point for a collection's existence and job config (the fusion the code owner
    specified: "a single entry point the model can call idempotently for all
    create/update cases").

    ``collection`` missing → the collection comes into being with this config (the
    create path, all its validation intact: birth idempotency-dedup, the
    inert/job-arg refusal).  ``collection``
    exists → only the fields explicitly set change (the update path: adopt a
    prompt, edit it, atomic schedule replace).  The model never reasons about
    which case it is.

    ``description`` is required the FIRST time (it's the meaning anchor);
    optional after.  ``notify`` is tri-state: ``None`` = leave unchanged (birth
    default: silent).  ``prompt`` IS the model surface for a job's instructions
    (#2455 naive baseline) — it is written straight into the collection's
    ``extraction_prompt``, verbatim; a wrong prompt is fixed by passing a new one.

    Every optional string here refuses a blank (#1776): omitting an argument and
    passing ``""`` for it are different calls, and only the first means "leave this
    alone" — the second is a half-expressed intention, corrected at the gate."""

    collection: MemoryName
    description: OptionalText = None
    prompt: OptionalText = None
    schedule: OptionalText = None
    expires_at: OptionalText = None
    notify: bool | None = None


class CollectionUpdateArgs(ToolArgs):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  A blank string is NOT "not set" (#1776): the ``OptionalText``
    fields refuse ``""`` at the arg gate, naming the field and its
    accepted form, so a field the model passes empty is corrected rather than read as
    an omission it never expressed.

    ``prompt`` is the job's instructions axis (#2455 naive baseline): supplying it
    REPLACES the collection's ``extraction_prompt`` verbatim with the new free text.
    Omitting it leaves the prompt untouched (a plain metadata edit).

    The **schedule** is the apply-time job axis — the SAME one-arg RRULE
    ``collection_configure`` accepts (``parse_schedule``, #1857).  Present → the schedule is
    replaced atomically, run quota and all, so a new rule never inherits the old rule's
    ``COUNT=``; absent → the schedule is left untouched.  So a collection's schedule is
    updatable post-create and an inert collection's job is set when a prompt is adopted.
    ``expires_at`` is the end condition.
    """

    name: MemoryName
    description: OptionalText = None
    notify: bool | None = None  # flip notify-on-new on/off; None = leave unchanged
    # Replace the stored extraction_prompt verbatim; None = leave prompt as-is (#2455).
    prompt: OptionalText = None
    # Schedule — one arg, one grammar (parse_schedule, #1857), mirroring collection_configure.
    # Present → replaces the whole schedule atomically; OMITTED → schedule untouched (a
    # blank is refused, #1776).
    schedule: OptionalText = None
    # OptionalText: omitted leaves the end condition alone; a blank ``""`` is refused
    # (#1776), mirroring create.
    expires_at: OptionalText = None


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionListArgs(ToolArgs):
    """Newest-first; ``k=None`` returns all."""

    collection: MemoryName
    k: ReadCount = None


class ReadSampleArgs(ToolArgs):
    """Random sample; ``k=None`` returns all."""

    collection: MemoryName
    k: ReadCount = None


class CollectionGetArgs(ToolArgs):
    """Exact key lookup in a collection."""

    collection: MemoryName
    key: str


class CollectionKeysArgs(ToolArgs):
    """One-field args for ``collection_keys``."""

    collection: MemoryName


# ── Log-specific reads ──────────────────────────────────────────────────────


class ReadLogArgs(ToolArgs):
    """A single ``log_read`` over a log.  The caller names only the log — the
    semantics (cursor batch for collectors, recent window for chat/schedule) and
    all sizes are decided in Python from the caller, never by the model."""

    log: MemoryName


class LogGetArgs(ToolArgs):
    """One entry of a log, by the id shown beside it — the ``log_get`` read.

    ``log`` names the log; ``id`` is the entry id a log read renders beside the
    entry (``format_entries(show_ids=True)``)."""

    log: MemoryName
    id: int


class ReadRunCallsArgs(ToolArgs):
    """One ``run_list`` over a run source — ``"chat"`` for conversational runs,
    or a collector's name for that collector's runs.  Batch size is fixed in Python."""

    source: MemoryName


class GetEventArgs(ToolArgs):
    """Resolve ONE ledger run by the typed id the activity block renders (#1580).

    ``run`` is the whole typed token as it appears on a self-state activity
    line — today the only addressable event is a run (``run <id>``), rendered on
    both the run lines and each mutation line's ``(run <id>)`` cause.  The tool
    parses the type tag and routes; the model copies the rendered token verbatim
    (the n≤1 anchor discipline), so no field here validates the tag — that's the
    tool's enumerated-cases dispatch, which names what IS addressable on a miss."""

    run: str


# ── Collection writes ───────────────────────────────────────────────────────


class CollectionEntrySpec(BaseModel):
    """One entry in a ``collection_write`` batch.

    ``extra="forbid"`` — a misspelled or extraneous key inside a batch entry
    (``{"key": …, "content": …, "id": …}``) surfaces as an actionable rejection
    naming the bad key rather than being silently dropped, exactly like a
    top-level ``ToolArgs`` field.  The envelope resolves the *nested* ``loc``
    (``("entries", 0, "badkey")``) down the parameters schema, so the message
    names the full path and suggests the valid sibling keys (#1416)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    content: str

    @model_validator(mode="before")
    @classmethod
    def _coerce_stringified_object(cls, value: Any) -> Any:
        """Parse a JSON-stringified dict back into a plain dict.

        Some models wrap array elements in outer quotes, producing a JSON string
        that contains an object literal (e.g. '{"key": "foo", "content": "bar"}')
        instead of a bare object. Detect and unwrap it so field validation proceeds
        normally.
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                pass
        return value


def _require_write_entries(value: list[CollectionEntrySpec]) -> list[CollectionEntrySpec]:
    """Reject an empty ``collection_write`` batch with an actionable message.

    A bare ``Field(min_length=1)`` surfaces Pydantic's generic "List should have
    at least 1 item"; name what an entry is so the model fixes the call."""
    if not value:
        raise ValueError("provide at least one entry (each a key plus its content) to write")
    return value


class CollectionWriteArgs(ToolArgs):
    """Batched write to a collection with dedup applied per entry."""

    collection: MemoryName
    entries: Annotated[list[CollectionEntrySpec], AfterValidator(_require_write_entries)]


class CollectionRemoveArgs(ToolArgs):
    """Remove an entry from a collection by key."""

    collection: MemoryName
    key: str


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendArgs(ToolArgs):
    """Append one keyless entry to a log."""

    log: AppendableLogName
    content: NonBlankLogContent


# ── Introspection ───────────────────────────────────────────────────────────


class FindArgs(ToolArgs):
    """Find anything of Penny's own by meaning (#1558, #1640, #1643, #2289).

    ``query`` — a paraphrase of what the thing is about (its meaning, not its
    exact name/key) — is the tool's whole surface when ``memory`` is omitted. The
    search then spans every family (collection | log | skill | entry) in one fused
    best-first list, with matching messages from the two conversation logs listed
    after it; there is no up-front family
    filter (a knob the model reasons about first can only encode a guess, #1643),
    so a passed ``type`` is rejected as an unknown argument (``ToolArgs``
    ``extra="forbid"``).

    ``memory`` (#2289) NARROWS the search to one named memory she can already
    name — a collection, a log, or one of the two conversation logs — and returns
    just that memory's own nearest entries by meaning; a blank is refused rather
    than read as "search everything" (``_reject_blank_memory``).
    """

    query: str
    memory: OptionalMemoryName = None


class DoneArgs(ToolArgs):
    """The one argument the cycle's close carries (#2385): whether the user should
    hear about this cycle.

    ``notify`` is REQUIRED — the choice is the cycle's to make and record, not a
    default to fall into — and the run record is still GENERATED from the ledger
    (the stored tool calls + write-gate outcomes + structural counts), never from a
    model-authored ``success``/``summary``, so there is nothing else here to
    confabulate.  A done call that fails validation is a failed record, not a close,
    so the loop rerolls and the model tries again.  ``DoneTool`` binds this model.
    """

    notify: bool


# ``DoneTool`` binds :class:`DoneArgs` (#2385) — the terminator used to be an
# argless sentinel (#1569); the notify decision moved onto it so the composer runs
# only when the cycle itself said there was news.
