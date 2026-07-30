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


# What each optional string argument on the collection surface actually accepts, so a
# blank rejection can NAME the form instead of only refusing.  Total over the fields
# typed ``OptionalText`` / ``OptionalSkill`` on the three collection arg models (pinned
# by a test), so the lookup indexes directly: a field declaring one of those types with
# no entry here is a programming error, not a case to absorb behind a vague default.
OPTIONAL_ARG_FORMS: dict[str, str] = {
    "description": "what the collection is for, in the user's own words",
    "skill": "the name of a learned skill (or a paraphrase of what it does)",
    # NOT a sixth copy of the four enumerated forms — ``_TRIGGER_TEACHING`` (in
    # collection_instantiation) is where that enumeration is authored, and a garbled
    # trigger gets it verbatim.  A BLANK one needs the one shape that makes the fix
    # obvious, so this stays an example and a pointer to the count.
    "trigger": 'a schedule in one of the four trigger forms, e.g. "every 3600" for hourly',
    "expires_at": "an ISO-8601 datetime, e.g. 2026-03-01T09:00:00Z",
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

# An optional skill name/paraphrase (#1620): dashes normalised first (a skill name may be
# slug-ish), then the same blank refusal — so ``skill=""`` is corrected rather than
# quietly taking the skill-less path.
OptionalSkill = Annotated[
    str | None, BeforeValidator(_reject_blank_optional), BeforeValidator(_normalize_dashes)
]


def _coerce_single_element_list(value: object) -> object:
    """Unwrap a single-element list bound to a skill parameter down to its element (#1666).

    A skill parameter always fills exactly ONE string leaf of a step's arguments, and that
    leaf frequently sits INSIDE a list: a browse step's ``queries[0]`` (distilled from
    ``browse(queries=[<url>])``) becomes a parameter.  The model, following the browse
    tool's real argument shape, then routinely binds that parameter with a one-element
    LIST — ``params={"url": ["https://…"]}`` — rather than the bare string the leaf holds.
    The ``dict[str, str]`` param type refuses it (``params.url: Input should be a valid
    string``), punishing the model for matching the tool's type.

    Since the bound value is a single leaf, unwrap the one element deterministically: the
    render substitutes it straight back into the list position, so a list-bound param
    renders byte-identically to the string-bound form (only the params-value TYPE
    validation was wrong).  A multi-element list is a genuine over-binding — a parameter
    takes exactly one value — so it stays a refusal, made actionable: the error ``loc``
    names the offending parameter (``params.<parameter>``) and the message names the count
    and the expected one-value shape.
    """
    if not isinstance(value, list):
        return value
    if len(value) == 1:
        return value[0]
    raise ValueError(
        f"a skill parameter binds one value, but got a list of {len(value)} — pass a "
        "single value per parameter (a one-item list is unwrapped for you; more than one "
        "isn't). Expected shape: params={'<parameter>': '<value>'}"
    )


# One skill-param binding (#1666): a single-element list is unwrapped to its element
# (the model mirrors the browse tool's list-shaped ``queries`` arg), a multi-element list
# is an actionable refusal, a scalar passes straight through to the ``str`` core rule.
SkillParamValue = Annotated[str, BeforeValidator(_coerce_single_element_list)]


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
    """Args for ``collection_set`` — the skill-instantiation front door (#1591),
    or a skill-less INERT container (#1629).

    A collection is storage plus an OPTIONAL job.  With a ``skill`` it INSTANTIATES
    that skill (resolved by name or meaning) — its steps render into the collection's
    ``extraction_prompt``, ``params`` binds the skill's parameters, and a
    ``trigger`` schedules it.  WITHOUT a ``skill`` the collection is INERT: storage only
    — no ``extraction_prompt``, no cadence, no ``notify`` — so nothing runs against it
    until a skill is attached later via ``collection_set`` (the two-step teach
    bootstrap).  A job-shaped arg (a ``trigger`` / ``notify`` / ``expires_at``) alongside
    a skill-less create is refused, since an inert container has no job to describe.

    ``description`` (required, non-blank) is what the collection is for, in the user's
    own words — the goal it serves and the collection's routing/dedup meaning anchor.
    ``name`` is the unique slug.

    **A blank string on an optional arg is REFUSED, not read as "not set"** (#1776,
    superseding #1646's coercion).  ``skill`` / ``trigger`` / ``expires_at`` reject
    ``""`` at the arg gate (``OptionalSkill`` / ``OptionalText``) with a teaching error
    naming the field and its accepted form, BEFORE any routing or parsing — so a model
    that fills an optional field with ``""`` (a chronic gpt-oss habit) is corrected
    rather than silently given the omitted behaviour, which turned a half-understood
    request into a half-built mechanism nothing marked as such.

    The **trigger** (skill path only) is ONE argument with four enumerated forms,
    parsed by prefix in the tool (``parse_trigger``): ``"every <seconds>"`` (a recurring
    cadence), ``"once at <ISO time> [xN]"`` (a delayed / one-shot schedule, N runs
    defaulting to 1), ``"on advance of <log>"`` (the collection wakes when that source
    LOG advances past its cursor), or ``"cron <5-field expression>"`` (a time-of-day
    recurrence, #1684).  An unparseable trigger is refused with a teaching error naming
    the four forms.  ``expires_at`` (optional) is the end condition — the
    watch archives itself when it passes.  ``notify`` (default false) makes the collection
    tell the user about new/changed entries; an omission stays silent, so it can never
    accidentally notify.
    """

    name: MemoryName
    description: NonBlankDescription
    # The skill to instantiate; OMITTED yields an INERT storage-only collection — the
    # first half of the two-step teach bootstrap (#1629).  A blank ``""`` is refused,
    # never read as omitted (#1776).  OptionalSkill == update's skill.
    skill: OptionalSkill = None
    # Bindings for the skill's parameters ({url}, {field}, …) → values.  A value
    # passed as a single-element list is unwrapped to its element (#1666,
    # SkillParamValue) — the model mirrors the browse tool's list-shaped queries arg.
    params: dict[str, SkillParamValue] = {}
    # Trigger — one arg, four enumerated forms, parsed by prefix in the tool
    # (parse_trigger, #1631/#1684): "every <seconds>" | "once at <ISO> [xN]" |
    # "on advance of <log>" | "cron <5-field expression>".  Its render
    # (render_trigger_clause) IS this input form.  OptionalText: a blank ``""`` is
    # refused at the gate (#1776) rather than reaching the parser as a garbled trigger.
    trigger: OptionalText = None
    # End condition (optional) — an ISO-8601 datetime; the collection archives
    # itself when it passes.  Parsed in the tool (actionable error on a bad value).
    # OptionalText: a blank ``""`` is refused at the gate (#1776) — the observed silent
    # drop, where a bounded watch asked for went live unbounded.
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

    name: MemoryName
    description: NonBlankDescription


class MemoryNameArgs(ToolArgs):
    """One-field args for ``archive`` / ``unarchive`` / read-all / keys."""

    memory: MemoryName


class CatalogArgs(ToolArgs):
    """No-field args for ``collection_catalog`` — it spans every collection."""


class CollectionSetArgs(ToolArgs):
    """Args for ``collection_set`` — the ONE idempotent create-or-update entry
    point for a collection's existence and job config (the fusion the code owner
    specified: "a single entry point the model can call idempotently for all
    create/update cases").

    ``name`` missing → the collection comes into being with this config (the
    create path, all its validation intact: birth idempotency-dedup unless
    skill resolution, the inert/job-arg refusal).  ``name``
    exists → only the fields explicitly set change (the update path: adopt /
    rebind / swap / refresh re-render, atomic trigger replace, raw
    ``extraction_prompt`` edit).  The model never reasons about which case it is.

    ``description`` is required the FIRST time (it's the meaning anchor);
    optional after.  ``notify`` is tri-state: ``None`` = leave unchanged (birth
    default: silent).  There is NO ``extraction_prompt`` argument anywhere on the
    model surface: a collection's routine is only ever a RENDER of a demonstrated
    skill (#1658) — a wrong routine is fixed by re-teaching, never by editing
    prompt text.

    Every optional string here refuses a blank (#1776): omitting an argument and
    passing ``""`` for it are different calls, and only the first means "leave this
    alone" — the second is a half-expressed intention, corrected at the gate."""

    name: MemoryName
    description: OptionalText = None
    skill: OptionalSkill = None
    params: dict[str, SkillParamValue] | None = None
    trigger: OptionalText = None
    expires_at: OptionalText = None
    notify: bool | None = None


class CollectionUpdateArgs(ToolArgs):
    """Update a collection's metadata.

    All fields after ``name`` are optional — only the ones explicitly set
    are applied.  A blank string is NOT "not set" (#1776): the ``OptionalText`` /
    ``OptionalSkill`` fields refuse ``""`` at the arg gate, naming the field and its
    accepted form, so a field the model passes empty is corrected rather than read as
    an omission it never expressed.

    ``skill`` / ``params`` are the re-render axis (#1620): supplying either RE-RENDERS
    the ``extraction_prompt`` from a skill's current steps and re-stamps the
    collection's skill provenance — ``skill`` names a skill (by name or meaning, the
    #1591 resolution union) to refresh / swap / adopt; ``params`` rebinds its parameters.
    Omitting both leaves the prompt untouched (a plain metadata edit).  ``params`` is
    ``None`` (reuse the collection's current bindings) vs. a dict (rebind to these).
    ``extraction_prompt`` is the raw-edit escape hatch — a FULL replacement body when
    editing the prompt directly rather than re-rendering from a skill (mutually
    exclusive with ``skill`` / ``params``).

    The **trigger** is the apply-time job axis — the SAME one-arg, four-form trigger
    ``collection_set`` accepts (``parse_trigger``, #1631/#1684): ``"every <seconds>"`` |
    ``"once at <ISO> [xN]"`` | ``"on advance of <log>"`` | ``"cron <5-field expression>"``.
    Present → the whole trigger is replaced atomically (the members the new form doesn't
    use clear); absent → the cadence is left untouched.  So a collection's schedule is
    updatable post-create and an inert collection's job is set when a skill is adopted.
    ``expires_at`` is the end condition.
    """

    name: MemoryName
    description: OptionalText = None
    notify: bool | None = None  # flip notify-on-new on/off; None = leave unchanged
    # Re-render axis (#1620): re-render the prompt from a skill's CURRENT steps.
    skill: OptionalSkill = None  # skill to (re-)instantiate from; None = leave prompt as-is
    # Rebind the skill's parameters; None = reuse current.  A single-element list value is
    # unwrapped to its element (#1666, SkillParamValue), mirroring create.
    params: dict[str, SkillParamValue] | None = None
    # Trigger — one arg, four enumerated forms (parse_trigger, #1631/#1684), mirroring
    # collection_set.  Present → replaces the whole trigger atomically; OMITTED →
    # cadence untouched (a blank is refused, #1776).  "every <seconds>" | "once at <ISO>
    # [xN]" | "on advance of <log>" | "cron <5-field expression>".
    trigger: OptionalText = None
    # OptionalText: omitted leaves the end condition alone; a blank ``""`` is refused
    # (#1776), mirroring create.
    expires_at: OptionalText = None


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


class FindArgs(ToolArgs):
    """Find anything of Penny's own by meaning (#1558, #1640, #1643).

    ``query`` — a paraphrase of what the thing is about (its meaning, not its
    exact name/key) — is the tool's whole surface.  The search spans every family
    (collection | log | skill | entry) in one fused best-first list; there is no
    up-front family filter (a knob the model reasons about first can only encode a
    guess, #1643), so a passed ``type`` is rejected as an unknown argument
    (``ToolArgs`` ``extra="forbid"``).
    """

    query: str


# ``done`` is an argless sentinel (#1569): it just marks the cycle finished.  The
# run record is GENERATED from the run's canonical ledger rows (the stored tool
# calls + write-gate outcomes + structural counts), never from a model-authored
# ``success``/``summary``, so the terminator carries no arguments to confabulate.
# ``DoneTool`` binds :class:`~penny.tools.models.NoArgs`.
