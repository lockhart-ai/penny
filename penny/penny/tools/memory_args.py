"""Pydantic arg models for the memory tool surface.

Each tool validates its kwargs through one of these models as its first line,
per the Pydantic-everywhere rule. Most read tools accept ``k: int | None``
meaning "no cap — return every entry" when omitted; this matches the access
layer's signature. ``MemoryReadArgs``'s collection branch is the exception
(#2771): with ``k``
omitted, the tool caps the read itself at
``PennyConstants.COLLECTION_LIST_DEFAULT_LIMIT`` rather than reading every entry —
an explicit ``k`` is still honoured in full, however large.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

from penny.constants import PennyConstants
from penny.text_validity import (
    require_non_blank_description,
    require_non_degenerate_content,
)
from penny.tools.base import PLAN_ARG_DESCRIPTION
from penny.tools.models import ToolArgs

# Models occasionally substitute Unicode dashes (U+2010–U+2015) for ASCII
# hyphen-minus (U+002D) when emitting memory names — gpt-oss has been
# observed writing ``"board‑games"`` for ``"board-games"``.
# The visual is identical but the string compares unequal, so memory-keyed
# tools (``memory_write``, ``memory_read``, etc.) silently failed
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
# typed ``OptionalText`` / ``OptionalSkill`` anywhere across the three collection arg
# models (pinned by a test), so the lookup indexes directly: a field declaring one of
# those types with no entry here is a programming error, not a case to absorb behind a
# vague default.  ``prompt`` keeps its entry for ``CollectionCreateArgs`` (a brand-new
# collection has nothing to clear, so a blank one there is still refused at the gate);
# on the update surface ``prompt`` carries no such validator at all (see
# ``PromptSetterText`` below) — a blank there can mean "clear an unused one" (#2857),
# a decision only the tool can make with the row in hand.
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

# No trailing period: the arg-validation envelope appends ". Call <tool>{<valid
# arguments>} again." after each field's reason.
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

# `prompt` on the UPDATE surface — `CollectionSetArgs` / `CollectionUpdateArgs`, the
# `collection_configure` front door and its update path (#2857, cascaded by #2894) —
# carries no blank-refusing validator at all: a blank means "clear it", which the arg
# gate can't decide alone (the tool reads the row to know what else needs clearing
# alongside it). See `CollectionUpdateTool._clears_the_job` / `_clear_job`.
PromptSetterText = str | None

# `schedule` on the UPDATE surface (#2894) — the same shape as `PromptSetterText`
# above, for the same reason: a blank `schedule` clears the whole job (schedule AND
# prompt, since one without the other never runs), a decision only
# `CollectionUpdateTool` can make with the row in hand.  `CollectionCreateArgs.schedule`
# keeps the ordinary `OptionalText` (a brand-new collection has nothing to clear).
ScheduleSetterText = str | None

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
# ``require_non_blank_log_content`` is applied directly in ``MemoryWriteTool._run``
# (memory_tools.py), not as an Annotated type here: whether a ``memory_write`` entry's
# ``content`` must be non-blank depends on the resolved target's SHAPE (a log refuses
# blank; a collection's degenerate check, including blank, runs inside ``collection.write``
# itself) — a fact Pydantic can't know before the tool resolves the name.


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
    # The structural plan mark (#2946, following #2933) — every ``mutates=True``
    # tool declares this literal field itself (``Tool.__init_subclass__`` asserts
    # it's here, rather than synthesising it, so a type checker can see it on
    # ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class LogCreateArgs(ToolArgs):
    """Args for ``log_create``.

    Logs are append-only streams of events (messages, browse results,
    measurements).  No extraction_prompt — logs are inputs, not curated
    outputs.  No interval — logs don't have a collector.
    """

    log: MemoryName
    description: NonBlankDescription


class MemoryNameArgs(ToolArgs):
    """One-field args for ``memory_info`` — the shape-agnostic metadata lookup."""

    memory: MemoryName


class CollectionArchiveArgs(ToolArgs):
    """Args for ``collection_archive`` — the archive/unarchive pair folded onto one
    tool (#2812): ``archived=True`` retires the collection (data kept, its collector
    stops running), ``archived=False`` restores it. Neither direction had a single
    use in 444 measured prod tool calls, so the fold is a pure token saving — the
    model still reaches both ends, now by naming the direction as a bool rather than
    picking between two tool names."""

    collection: MemoryName
    archived: bool
    # The structural plan mark (#2933) — every ``mutates=True`` tool declares
    # this literal field itself (``Tool.__init_subclass__`` asserts it's here,
    # rather than synthesising it, so a type checker can see it on ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


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

    Every optional string here refuses a blank (#1776) EXCEPT ``prompt`` and
    ``schedule`` (``PromptSetterText`` / ``ScheduleSetterText``, #2857/#2894):
    omitting an argument and passing ``""`` for it are different calls, and for
    ``description`` / ``expires_at`` only the first means "leave this alone" — the
    second is a half-expressed intention, corrected at the gate.  A blank ``prompt``
    or a blank ``schedule`` is its OWN intention — clear the whole job, since one
    doesn't run without the other — so the tool itself decides, cascading onto
    whichever of the two the call left standing.  See
    ``CollectionUpdateTool._clears_the_job`` / ``_clear_job``."""

    collection: MemoryName
    description: OptionalText = None
    prompt: PromptSetterText = None
    schedule: ScheduleSetterText = None
    expires_at: OptionalText = None
    notify: bool | None = None
    # The structural plan mark (#2946, following #2933) — every ``mutates=True``
    # tool declares this literal field itself (``Tool.__init_subclass__`` asserts
    # it's here, rather than synthesising it, so a type checker can see it on
    # ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CollectionUpdateArgs(ToolArgs):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  A blank string on ``description`` / ``expires_at`` is NOT "not
    set" (#1776): the ``OptionalText`` fields refuse ``""`` at the arg gate, naming
    the field and its accepted form, so a field the model passes empty is corrected
    rather than read as an omission it never expressed.  ``prompt`` and ``schedule``
    are the exception (below).

    ``prompt`` is the job's instructions axis (#2455 naive baseline): supplying it
    REPLACES the collection's ``extraction_prompt`` verbatim with the new free text.
    Omitting it leaves the prompt untouched (a plain metadata edit).  A BLANK
    ``prompt`` or a BLANK ``schedule`` (``PromptSetterText`` / ``ScheduleSetterText``,
    #2857/#2894) clears the WHOLE job — both axes, whichever one the call blanked —
    since a prompt with no schedule never runs and a schedule with no prompt has
    nothing to run; the collection lands as plain storage, entries untouched.  See
    ``CollectionUpdateTool._clears_the_job`` / ``_clear_job``.

    The **schedule** is otherwise the apply-time job axis — the SAME one-arg RRULE
    ``collection_configure`` accepts (``parse_schedule``, #1857).  Present (non-blank)
    → the schedule is replaced atomically, run quota and all, so a new rule never
    inherits the old rule's ``COUNT=``; absent → the schedule is left untouched.  So a
    collection's schedule is updatable post-create and an inert collection's job is
    set when a prompt is adopted.  ``expires_at`` is the end condition.
    """

    name: MemoryName
    description: OptionalText = None
    notify: bool | None = None  # flip notify-on-new on/off; None = leave unchanged
    # Replace the stored extraction_prompt verbatim; None = leave prompt as-is (#2455).
    # PromptSetterText, not OptionalText (#2857): a blank is the tool's own clear
    # decision, not the gate's uniform blank rejection.
    prompt: PromptSetterText = None
    # Schedule — one arg, one grammar (parse_schedule, #1857), mirroring collection_configure.
    # Present (non-blank) → replaces the whole schedule atomically; OMITTED → schedule
    # untouched.  ScheduleSetterText, not OptionalText (#2894): a blank is the tool's own
    # clear decision (cascading onto ``prompt`` too), not the gate's uniform rejection.
    schedule: ScheduleSetterText = None
    # OptionalText: omitted leaves the end condition alone; a blank ``""`` is refused
    # (#1776), mirroring create.
    expires_at: OptionalText = None
    # The structural plan mark (#2946, following #2933) — every ``mutates=True``
    # tool declares this literal field itself (``Tool.__init_subclass__`` asserts
    # it's here, rather than synthesising it, so a type checker can see it on
    # ``args.plan``).
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


# ── Memory reads (collection or log, store-type resolved in Python) ─────────


class MemoryReadArgs(ToolArgs):
    """A single ``memory_read`` over a collection or a log — the caller names
    only the memory; the tool resolves whether it's a collection or a log and
    applies that shape's own option set.  The model never picks the shape.

    Collection options (#2771/#2812): ``k=None`` returns the newest (or, with
    ``order="random"``, samples up to) ``PennyConstants.COLLECTION_LIST_DEFAULT_LIMIT``
    entries rather than every entry; name ``k`` to read more (or fewer) — honoured
    in full, however large. ``order`` picks ``"newest"`` (default, walks back from
    the most recent entry) or ``"random"`` (samples ``k`` entries uniformly).

    Log options (#2752/#2961): with no option, the read is cursor-based for a
    collector (the next batch since its last committed cursor) or window-based for
    chat/schedule (the recent look-back window) — the caller never chooses the
    mode, Python does. ``day`` reads one local day, oldest first, capped — the
    conversation map's anchor reads name the day this way. ``before`` pages
    backward from an id a prior read rendered beside an entry — the newest bounded
    batch older than that entry, oldest first.

    ``k``/``order`` on a log, or ``day``/``before`` on a collection, are refused
    with a teaching error naming the options that actually apply to that shape —
    the two option sets describe genuinely different reading models (a keyed,
    deduped store vs. an append-only stream), so a cross-shape argument is never
    silently reinterpreted."""

    memory: MemoryName
    k: ReadCount = None
    order: Literal["newest", "random"] = "newest"
    day: str | None = None
    before: int | None = None


class MemoryGetArgs(ToolArgs):
    """``memory_get`` — exact key lookup in a collection, or the numeric entry id
    (passed as a string) a ``memory_read`` showed beside an entry, for a log."""

    memory: MemoryName
    key: str


class MemoryKeysArgs(ToolArgs):
    """One-field args for ``memory_keys`` — a collection's keys, or a log's
    entry ids."""

    memory: MemoryName


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
    tool's enumerated-cases dispatch, which names what IS addressable on a miss.

    ``step`` (#2841) names the one step — by the number its own ``step <N>``
    line rendered — whose thinking should come back WHOLE rather than excerpted.
    Optional, and ``ge=1`` since step numbers are 1-based ordinals: a caller
    passing 0 or a negative number gets Pydantic's own actionable rejection
    rather than a tool that silently treats it as "no step named"."""

    run: str
    step: int | None = Field(default=None, ge=1)


# ── Memory writes (collection or log, store-type resolved in Python) ────────


class MemoryEntrySpec(BaseModel):
    """One entry in a ``memory_write`` batch.

    ``key`` is REQUIRED for a collection (the tool refuses a keyless entry
    against a collection with a teaching error) and IGNORED for a log — a log
    entry always lands keyless, exactly as the retired ``log_append`` always
    stored it, whatever a supplied ``key`` said; the write headline for a log
    target always states entries landed keyless, so a model is never taught
    that a log honours a key.

    ``extra="forbid"`` — a misspelled or extraneous key inside a batch entry
    (``{"key": …, "content": …, "id": …}``) surfaces as an actionable rejection
    naming the bad key rather than being silently dropped, exactly like a
    top-level ``ToolArgs`` field.  The envelope resolves the *nested* ``loc``
    (``("entries", 0, "badkey")``) down the parameters schema, so the message
    names the full path and suggests the valid sibling keys (#1416)."""

    model_config = ConfigDict(extra="forbid")

    key: str | None = None
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


def _require_write_entries(value: list[MemoryEntrySpec]) -> list[MemoryEntrySpec]:
    """Reject an empty ``memory_write`` batch with an actionable message.

    A bare ``Field(min_length=1)`` surfaces Pydantic's generic "List should have
    at least 1 item"; name what an entry is so the model fixes the call."""
    if not value:
        raise ValueError(
            "provide at least one entry (each its content, plus a key for a collection) to write"
        )
    return value


class MemoryWriteArgs(ToolArgs):
    """Batched write to a collection (dedup applied per entry, each entry's
    ``key`` required) or an append to a log (keyless, no dedup, any supplied
    ``key`` ignored) — the ``memory`` named decides which, in Python.

    ``memory`` reuses the system-log guard ``log_append`` carried
    (``AppendableLogName``): a model-authored entry can't land on a framework-
    managed log (``user-messages`` / ``penny-messages`` / ``collector-runs`` /
    ``run-health`` / ``changelog``) — safe to apply unconditionally here, since
    no collection is ever one of those five names (the registry's names are
    unique across both shapes)."""

    memory: AppendableLogName
    entries: Annotated[list[MemoryEntrySpec], AfterValidator(_require_write_entries)]
    # The structural plan mark (#2933) — see ``CollectionArchiveArgs.plan``.
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


class CollectionRemoveArgs(ToolArgs):
    """Remove an entry from a collection by key."""

    collection: MemoryName
    key: str
    # The structural plan mark (#2933) — see ``CollectionArchiveArgs.plan``.
    plan: bool = Field(default=False, description=PLAN_ARG_DESCRIPTION)


# ── Introspection ───────────────────────────────────────────────────────────


class FindArgs(ToolArgs):
    """Find anything of Penny's own by meaning (#1558, #1640, #1643, #2289).

    ``query`` — a paraphrase of what the thing is about (its meaning, not its
    exact name/key) — is the tool's whole surface when ``memory`` is omitted. The
    search then spans every family (collection | log | entry) in one fused
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


# ``synthesize``'s own narrowing (#2932) differs from ``find``'s singular ``memory``:
# synthesis is explicitly about a pattern spanning SEVERAL containers, so this takes a
# list.  An empty list carries the same "search everything or search nothing?"
# ambiguity ``_reject_blank_memory`` refuses on a blank string — refused the same way,
# naming the fix, rather than silently read as either.
_BLANK_SYNTHESIZE_MEMORIES_ARG = (
    "an empty list sets nothing — omit memories entirely to search everything, or "
    "pass the exact name of one or more collections or logs"
)


def _reject_empty_memories(value: object) -> object:
    if isinstance(value, list) and len(value) == 0:
        raise ValueError(_BLANK_SYNTHESIZE_MEMORIES_ARG)
    return value


def _normalize_memories_dashes(value: object) -> object:
    if isinstance(value, list):
        return [_normalize_dashes(item) for item in value]
    return value


OptionalMemoryNames = Annotated[
    list[str] | None,
    BeforeValidator(_reject_empty_memories),
    BeforeValidator(_normalize_memories_dashes),
]


class SynthesizeArgs(ToolArgs):
    """What ``synthesize`` (#2932) needs to answer "what does my memory say about X?"

    ``query`` is the whole surface when ``memories``/``since`` are omitted: the same
    paraphrase ``find`` takes, reused for the same two deterministic lookups
    (``MemoryStore.resolve_objects``/``resolve_messages``) — no new search logic.

    ``memories`` (optional) narrows to these named collections/logs instead of
    everything — plural, since a synthesis is explicitly about a pattern that can span
    more than one. An empty list is refused rather than read either way
    (``_reject_empty_memories``).

    ``since`` (optional) drops anything from before this point in time — an ISO 8601
    timestamp, e.g. ``2026-10-01T00:00:00Z``.
    """

    query: str
    memories: OptionalMemoryNames = None
    since: datetime | None = None


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
