"""The behaviour taxonomy — one shared vocabulary for how we classify what the
model did.

Every condition we reason about Penny's behaviour through is named here exactly
once, with where it is enforced made explicit:

- ``live``  — caught while the run is happening (a ``ResponseValidator``
  disposition in the agentic loop, or the ``send_message`` arg-validation gate),
  so the model gets a chance to recover.
- ``run_flag`` — surfaces after the fact as a ``⚠`` line on the run record that
  Penny's ``quality`` collector reads and the addon badges, derived structurally
  from the persisted ``promptlog`` rows.

A condition can be both (``HALF_FORMED_SEND`` is gated live *and* flagged
post-hoc; ``NO_WORK_DONE`` is refused live as a premature ``done()`` *and*
flagged post-hoc as a bail) — that overlap is the point: the user, Penny, and a
maintainer see *one* coherent set of behaviours, not a guard that means one thing
in the loop and a different thing in the run log.

This module is a dependency-light leaf — only ``constants`` + pydantic, like
``text_validity`` — so the agentic loop, the database run-health classifier, and
the addon-serving code all import it without an import cycle.

The ``marker`` / ``detail`` strings are the canonical run-record text.  They are
frozen: the seeded ``quality`` prompt (migration 0072) and the addon's TS type
mirror them, so changing a value is a coordinated migration, not an edit here.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ConditionKey(StrEnum):
    """The stable machine key for every behaviour condition.

    Values are the persisted / TS-mirrored identifiers (they supersede the old
    split between ``ValidationReason`` and ``RunHealthFlag``), so they are frozen
    once shipped."""

    # ── Response-level: detectable from a single model response ──────────────
    XML = "xml"
    EMPTY = "empty"
    REFUSAL = "refusal"
    HALLUCINATED_URLS = "hallucinated_urls"
    TOOL_PARSE_ERROR = "tool_parse_error"
    TEXT_INSTEAD_OF_TOOL = "text_instead_of_tool"
    # ── Caught live AND flagged post-hoc ─────────────────────────────────────
    HALF_FORMED_SEND = "half_formed_send"
    NO_WORK_DONE = "no_work_done"
    # ── Run-level: only emergent across a whole run ──────────────────────────
    INCOMPLETE = "incomplete"
    TOOL_FAILURES = "tool_failures"


class BehaviorCondition(BaseModel):
    """One condition in the taxonomy — defined once, consumed everywhere.

    ``label`` is the maintainer/user word for the behaviour.  ``live`` /
    ``run_flag`` / ``collector_only`` declare *where* it applies, so "this check
    runs in some contexts and not others" is data you can read here, not implicit
    in scattered branches.  ``marker`` / ``detail`` are the canonical run-record
    render (present iff ``run_flag``)."""

    model_config = ConfigDict(frozen=True)

    key: ConditionKey
    label: str
    live: bool = False
    run_flag: bool = False
    collector_only: bool = False
    # The "⚠ …" header and the verbose explanation after it, on a run record.
    marker: str | None = None
    detail: str | None = None


# The shared ``⚠`` marker prefix (the addon colours by it; the quality prompt
# names it).  One definition so a render never spells it inline.
HEALTH_MARKER = "⚠"


def _condition(
    key: ConditionKey,
    label: str,
    *,
    live: bool = False,
    run_flag: bool = False,
    collector_only: bool = False,
    marker: str | None = None,
    detail: str | None = None,
) -> BehaviorCondition:
    return BehaviorCondition(
        key=key,
        label=label,
        live=live,
        run_flag=run_flag,
        collector_only=collector_only,
        marker=f"{HEALTH_MARKER} {marker}" if marker else None,
        detail=detail,
    )


# The catalog.  Insertion order is the canonical render order for run-record
# flags (NO_WORK_DONE → INCOMPLETE → TOOL_FAILURES → HALF_FORMED_SEND, the order
# the old ``RunHealth.flags`` emitted).
_CATALOG_ENTRIES: tuple[BehaviorCondition, ...] = (
    # ── Response-shape conditions (live only; chat + collector) ──────────────
    _condition(
        ConditionKey.XML,
        "Response wrapped in XML/markup instead of plain prose",
        live=True,
    ),
    _condition(
        ConditionKey.EMPTY,
        "Response carries no substantive content",
        live=True,
    ),
    _condition(
        ConditionKey.REFUSAL,
        "Response is a model refusal rather than a real answer",
        live=True,
    ),
    _condition(
        ConditionKey.HALLUCINATED_URLS,
        "Response cites a URL that was never in the source material",
        live=True,
    ),
    _condition(
        ConditionKey.TOOL_PARSE_ERROR,
        "Tool call emitted as malformed/plain text the runtime can't parse",
        live=True,
    ),
    _condition(
        ConditionKey.TEXT_INSTEAD_OF_TOOL,
        "Collector narrated prose where a tool call was required",
        live=True,
        collector_only=True,
    ),
    # ── Caught live AND flagged post-hoc ─────────────────────────────────────
    _condition(
        ConditionKey.NO_WORK_DONE,
        "Run did no real work before terminating",
        live=True,
        run_flag=True,
        collector_only=True,
        marker="NO WORK DONE",
        detail=(
            "reached done() (or made no tool call) without any read/write/browse "
            "step first; the collector is not following its instructions"
        ),
    ),
    _condition(
        ConditionKey.INCOMPLETE,
        "Run did work but never closed cleanly",
        run_flag=True,
        collector_only=True,
        marker="INCOMPLETE",
        detail=(
            "hit the step ceiling without a closing done(); work landed but the "
            "cycle never finished cleanly"
        ),
    ),
    _condition(
        ConditionKey.TOOL_FAILURES,
        "One or more tool calls errored during the run",
        run_flag=True,
        collector_only=True,
        marker="TOOL FAILURES",
        detail="a tool call returned an error and the run kept going",
    ),
    _condition(
        ConditionKey.HALF_FORMED_SEND,
        "A message was sent with no real content",
        live=True,
        run_flag=True,
        marker="HALF-FORMED SEND",
        detail=(
            "a message went out with no real content (empty, punctuation-only, or "
            "an unfinished fragment)"
        ),
    ),
)

CATALOG: dict[ConditionKey, BehaviorCondition] = {entry.key: entry for entry in _CATALOG_ENTRIES}


def condition(key: ConditionKey) -> BehaviorCondition:
    """The catalog entry for ``key`` (KeyError if absent — the catalog is total)."""
    return CATALOG[key]


def run_flag_conditions() -> list[BehaviorCondition]:
    """The conditions that surface on a run record, in canonical render order."""
    return [entry for entry in _CATALOG_ENTRIES if entry.run_flag]
