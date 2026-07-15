"""Pydantic arg models for the memory tool surface.

Each tool validates its kwargs through one of these models as its first line,
per the Pydantic-everywhere rule. Most read tools accept ``k: int | None``
meaning "no cap — return every entry" when omitted; this matches the access
layer's signature.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, model_validator

from penny.constants import PennyConstants
from penny.database.memory import ResolvedKind
from penny.text_validity import (
    require_extraction_prompt,
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


def _normalize_dash_list(value: object) -> object:
    if not isinstance(value, list):
        return value
    return [_normalize_dashes(item) for item in value]


def _require_memory_names(value: list[str]) -> list[str]:
    """Reject an empty ``memories`` list with an actionable message.

    A bare ``Field(min_length=1)`` surfaces Pydantic's generic "List should have
    at least 1 item" — true but not actionable, and inconsistent with the browse/
    email empty-list gates.  Name what to supply."""
    if not value:
        raise ValueError("provide at least one collection name to check for a duplicate")
    return value


MemoryName = Annotated[str, BeforeValidator(_normalize_dashes)]
# A ``memories`` list that must name at least one collection (the ``exists`` probe).
NonEmptyMemoryNameList = Annotated[
    list[str], BeforeValidator(_normalize_dash_list), AfterValidator(_require_memory_names)
]


def _reject_nonpositive_count(value: int | None) -> int | None:
    """Reject a read count of zero (or negative) with an actionable message.

    ``k``/``n`` cap a read; ``None`` means "no cap — every entry".  A model that
    wants *all* entries sometimes guesses ``k=0`` (reading it as "unlimited"),
    but ``.limit(0)`` returns **zero** rows — so the model sees an empty memory
    and wrongly concludes it's empty (observed: the skills collector read
    ``collection_read_latest(k=0)``, saw no skills, and wrote a duplicate instead
    of updating the existing one).  Fail loudly with the fix rather than silently
    return nothing."""
    if value is not None and value < 1:
        raise ValueError(
            f"k={value} would read zero entries — a read count must be at least 1. "
            "Omit k entirely to read every entry."
        )
    return value


ReadCount = Annotated[int | None, AfterValidator(_reject_nonpositive_count)]


def _blank_to_none(value: object) -> object:
    """Collapse a blank string to ``None`` so an update treats it as "omitted".

    Models routinely emit ``""`` for an optional field they mean to leave
    unchanged — gpt-oss was observed passing ``extraction_prompt=""`` alongside
    another field's change, reasoning "they will not be updated".  But the update
    layer applies any value that ``is not None``, so a blank string would
    overwrite: silently blanking a ``description`` (and re-embedding empty as its
    meaning anchor).  None of these fields has a meaningful empty value, so a
    blank means "skip", never "set to empty".
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


# Optional text on an update: a blank string is coerced to ``None`` (omitted)
# rather than written through, so it can never clobber the existing value.
OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]

# An optional skill name/paraphrase on an update (#1620): dashes normalised (a
# skill name may be slug-ish) then a blank coerced to ``None`` (omitted), so a
# model leaving it empty means "don't re-render", never "resolve the empty skill".
OptionalSkill = Annotated[
    str | None, BeforeValidator(_blank_to_none), BeforeValidator(_normalize_dashes)
]


# ── Annotated validator types ─────────────────────────────────────────────────
# One Annotated type per validation concern, wrapping a shared predicate, so a
# field declares its rule by *type* — no per-field @field_validator methods.  The
# required variants raise on a bad value; the optional variants coerce a blank to
# ``None`` first (so "" means "leave unchanged") and skip the rule when omitted.


def _require_resolved_kind(value: str) -> str:
    """Raise unless ``value`` is a valid ``ResolvedKind``, naming the choices."""
    if value not in {kind.value for kind in ResolvedKind}:
        valid = ", ".join(kind.value for kind in ResolvedKind)
        raise ValueError(f"type must be one of: {valid}.")
    return value


def _skip_none(validator: Any) -> Any:
    """Wrap an AfterValidator predicate so it runs only when the value is set —
    an optional field coerced to ``None`` (omitted) skips the rule."""

    def _validate(value: Any) -> Any:
        return value if value is None else validator(value)

    return _validate


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

# Optional-on-update variants: blank → None (omitted), rule applied only when set.
OptionalExtractionPrompt = Annotated[
    str | None,
    BeforeValidator(_blank_to_none),
    AfterValidator(_skip_none(require_extraction_prompt)),
]
# Optional resolve-by-meaning family filter: blank → None (span all families).
OptionalResolvedKind = Annotated[
    str | None, BeforeValidator(_blank_to_none), AfterValidator(_skip_none(_require_resolved_kind))
]


# ── Metadata ────────────────────────────────────────────────────────────────


class CollectionCreateArgs(ToolArgs):
    """Args for ``collection_create`` — the skill-instantiation front door (#1591),
    or a skill-less INERT container (#1629).

    A collection is storage plus an OPTIONAL job.  With a ``skill`` it INSTANTIATES
    that skill (resolved by name or meaning) — its steps render into the collection's
    ``extraction_prompt``, ``params`` binds the skill's parameter holes, and a
    ``trigger`` schedules it.  WITHOUT a ``skill`` the collection is INERT: storage only
    — no ``extraction_prompt``, no cadence, no ``notify`` — so nothing runs against it
    until a skill is attached later via ``collection_update`` (the two-step teach
    bootstrap).  A job-shaped arg (a ``trigger`` / ``notify`` / ``expires_at``) alongside
    a skill-less create is refused, since an inert container has no job to describe.

    ``description`` (required, non-blank) is what the collection is for, in the user's
    own words — the goal it serves and the collection's routing/dedup meaning anchor.
    ``name`` is the unique slug.

    The **trigger** (skill path only) is ONE argument with three enumerated forms,
    parsed by prefix in the tool (``parse_trigger``): ``"every <seconds>"`` (a recurring
    cadence), ``"once at <ISO time> [xN]"`` (a delayed / one-shot schedule, N runs
    defaulting to 1), or ``"on advance of <log>"`` (the collection wakes when that source
    LOG advances past its cursor).  An unparseable trigger is refused with a teaching
    error naming the three forms.  ``expires_at`` (optional) is the end condition — the
    watch archives itself when it passes.  ``notify`` (default false) makes the collection
    tell the user about new/changed entries; an omission stays silent, so it can never
    accidentally notify.  ``create_anyway`` (default false) is the reactive idempotency
    override — set only when a near-duplicate refusal tells you to.
    """

    name: MemoryName
    description: NonBlankDescription
    # The skill to instantiate; omitted (``None``) yields an INERT storage-only
    # collection — the first half of the two-step teach bootstrap (#1629).
    skill: MemoryName | None = None
    # Bindings for the skill's parameter holes ({url}, {field}, …) → values.
    params: dict[str, str] = {}
    # Trigger — one arg, three enumerated forms, parsed by prefix in the tool
    # (parse_trigger, #1631): "every <seconds>" | "once at <ISO> [xN]" |
    # "on advance of <log>".  Its render (render_trigger_clause) IS this input form.
    trigger: str | None = None
    # End condition (optional) — an ISO-8601 datetime; the collection archives
    # itself when it passes.  Parsed in the tool (actionable error on a bad value).
    expires_at: str | None = None
    # Notify-on-new (emission-as-property, #1557): true when the user asked to be
    # told / kept posted / alerted about new entries.  Defaults false (silent).
    notify: bool = False
    # The reactive idempotency override (#1567) — default false so an omission never
    # silently creates a near-duplicate; set true ONLY in response to a near-duplicate
    # refusal (that refusal is the sole place it's explained).
    create_anyway: bool = False


class LogCreateArgs(ToolArgs):
    """Args for ``log_create``.

    Logs are append-only streams of events (messages, browse results,
    measurements).  No extraction_prompt — logs are inputs, not curated
    outputs.  No interval — logs don't have a collector.
    """

    name: MemoryName
    description: NonBlankDescription


class MemoryNameArgs(ToolArgs):
    """One-field args for ``archive`` / ``unarchive`` / read-all / keys."""

    memory: MemoryName


class CatalogArgs(ToolArgs):
    """No-field args for ``collection_catalog`` — it spans every collection."""


class CollectionUpdateArgs(ToolArgs):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  A blank string counts as "not set": the ``OptionalText``
    fields coerce ``""`` to ``None`` so a field the model passes empty (to
    mean "leave it alone") is skipped rather than overwriting the existing
    value.

    ``skill`` / ``params`` are the re-render axis (#1620): supplying either RE-RENDERS
    the ``extraction_prompt`` from a skill's current steps and re-stamps the
    collection's skill provenance — ``skill`` names a skill (by name or meaning, the
    #1591 resolution union) to refresh / swap / adopt; ``params`` rebinds its holes.
    Omitting both leaves the prompt untouched (a plain metadata edit).  ``params`` is
    ``None`` (reuse the collection's current bindings) vs. a dict (rebind to these).
    ``extraction_prompt`` is the raw-edit escape hatch — a FULL replacement body when
    editing the prompt directly rather than re-rendering from a skill (mutually
    exclusive with ``skill`` / ``params``).

    The **trigger** is the apply-time job axis — the SAME one-arg, three-form trigger
    ``collection_create`` accepts (``parse_trigger``, #1631): ``"every <seconds>"`` |
    ``"once at <ISO> [xN]"`` | ``"on advance of <log>"``.  Present → the whole trigger
    is replaced atomically (the members the new form doesn't use clear); absent → the
    cadence is left untouched.  So a collection's schedule is updatable post-create and
    an inert collection's job is set when a skill is adopted.  ``expires_at`` is the end
    condition.
    """

    name: MemoryName
    description: OptionalText = None
    extraction_prompt: OptionalExtractionPrompt = None
    notify: bool | None = None  # flip notify-on-new on/off; None = leave unchanged
    # Re-render axis (#1620): re-render the prompt from a skill's CURRENT steps.
    skill: OptionalSkill = None  # skill to (re-)instantiate from; None = leave prompt as-is
    params: dict[str, str] | None = None  # rebind the skill's holes; None = reuse current
    # Trigger — one arg, three enumerated forms (parse_trigger, #1631), mirroring
    # collection_create.  Present → replaces the whole trigger atomically; a blank/omit
    # → cadence untouched.  "every <seconds>" | "once at <ISO> [xN]" | "on advance of <log>".
    trigger: OptionalText = None
    expires_at: str | None = None


# ── Collection reads ────────────────────────────────────────────────────────


class CollectionGetArgs(ToolArgs):
    """Exact key lookup in a collection."""

    memory: MemoryName
    key: str


class ReadLatestArgs(ToolArgs):
    """Newest-first; ``k=None`` returns all."""

    memory: MemoryName
    k: ReadCount = None


class ReadRandomArgs(ToolArgs):
    """Random sample; ``k=None`` returns all."""

    memory: MemoryName
    k: ReadCount = None


class ReadSimilarArgs(ToolArgs):
    """Top-k by content cosine similarity to ``anchor`` (embedded by the tool).

    A plain nearest-neighbour search: entries come back ranked best-first so the
    model can judge them.  There is no relevance floor or cluster gate — those
    ambient-recall policies suppressed a populated but homogeneous collection
    (e.g. ``skills``) to "No entries" and broke fuzzy recovery (#1565).  An empty
    result therefore reflects the corpus, not an ambient "nothing matched well
    enough" judgment.  ``k`` caps the count; omit for all.
    """

    memory: MemoryName
    anchor: str
    k: ReadCount = None


# ── Log-specific reads ──────────────────────────────────────────────────────


class ReadLogArgs(ToolArgs):
    """A single ``log_read`` over a log.  The caller names only the log — the
    semantics (cursor batch for collectors, recent window for chat/schedule) and
    all sizes are decided in Python from the caller, never by the model."""

    memory: MemoryName


class ReadRunCallsArgs(ToolArgs):
    """One ``read_run_calls`` over a run source — ``"chat"`` for conversational runs,
    or a collector's name for that collector's runs.  Batch size is fixed in Python."""

    target: MemoryName


class GetEventArgs(ToolArgs):
    """Resolve ONE ledger event by the typed id the activity block renders (#1580).

    ``event_id`` is the whole typed token as it appears on a self-state activity
    line — today the only addressable event is a run (``run <id>``), rendered on
    both the run lines and each mutation line's ``(run <id>)`` cause.  The tool
    parses the type tag and routes; the model copies the rendered token verbatim
    (the n≤1 anchor discipline), so no field here validates the tag — that's the
    tool's enumerated-cases dispatch, which names what IS addressable on a miss."""

    event_id: str


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

    memory: MemoryName
    entries: Annotated[list[CollectionEntrySpec], AfterValidator(_require_write_entries)]


class UpdateEntryArgs(ToolArgs):
    """Replace content for an existing key in a collection."""

    memory: MemoryName
    key: str
    content: CollectionContent


class CollectionMergeArgs(ToolArgs):
    """Merge all entries from one collection into another, then archive the source."""

    from_memory: MemoryName
    to_memory: MemoryName


class CollectionDeleteEntryArgs(ToolArgs):
    """Delete an entry from a collection by key."""

    memory: MemoryName
    key: str


# ── Log writes ──────────────────────────────────────────────────────────────


class LogAppendArgs(ToolArgs):
    """Append one keyless entry to a log."""

    memory: AppendableLogName
    content: NonBlankLogContent


# ── Introspection ───────────────────────────────────────────────────────────


class ExistsArgs(ToolArgs):
    """Cross-memory dedup probe used by thinking-class agents before writes."""

    memories: NonEmptyMemoryNameList
    content: str
    key: str | None = None


class FindMineArgs(ToolArgs):
    """Resolve one of Penny's own objects by meaning (#1558).

    ``query`` is a paraphrase of what the thing is about (its meaning, not its
    exact name); ``type`` optionally narrows to a single family (collection | log
    | skill).  A blank ``type`` means "span all families".
    """

    query: str
    type: OptionalResolvedKind = None


# ``done`` is an argless sentinel (#1569): it just marks the cycle finished.  The
# run record is GENERATED from the run's canonical ledger rows (the stored tool
# calls + write-gate outcomes + structural counts), never from a model-authored
# ``success``/``summary``, so the terminator carries no arguments to confabulate.
# ``DoneTool`` binds :class:`~penny.tools.models.NoArgs`.
