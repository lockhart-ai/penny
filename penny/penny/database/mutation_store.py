"""``MutationStore`` — the registry-mutation event ledger (#1560).

Every create / update / archive / unarchive of a registry entity (a collection)
writes one ``mutation_event`` row here — (entity, run, actor, what changed, when)
— so a mechanism's configuration history is a *read*, not a memory the model
re-asserts from its own past narration.  This is the one ledger table with no
other home: an entry write is a ``promptlog`` tool call and a run is a
``promptlog`` group, but a *system* archive (the scheduler's ``max_runs`` /
``expires_at`` retire) runs no model and logs no prompt, so without this row it
would be invisible.

Audit + provenance, not event sourcing: the ``memory`` row stays the truth of an
entity's current state; this only records the transitions that produced it.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from penny.constants import MutationAction, MutationActor, MutationEntityType
from penny.database.models import MutationEvent
from penny.datetime_utils import format_log_timestamp

logger = logging.getLogger(__name__)


class EnumeratedDecision(BaseModel):
    """One enumerated model decision, recorded as (state slice, options, choice,
    result) — the *options presented*, not just the choice made (#1560).

    The choice alone says what happened; the options are what make history
    replayable (re-run a past decision against a new prompt / model / taxonomy and
    diff) and a misclassification diagnosable (you can't score a choice without
    the menu it was picked from).  Cheap at write time, impossible to reconstruct
    later.  This shape is *accommodated* now — it rides on every event's detail
    and on the canonical logged call — but call sites populate it only with the
    enumerated-decision unions of #1562/#1563; nothing forces it here.
    """

    state_slice: str | None = None
    options: list[str] = Field(default_factory=list)
    choice: str
    result: str | None = None


class FieldPrior(BaseModel):
    """What ONE field held immediately BEFORE the mutation that changed it (#1946).

    ``changed_fields`` says a field moved; this says what it moved FROM — the half
    nothing else can answer, because the ``memory`` row is overwritten in place and
    the run's ``promptlog`` tool call carries only the value the caller asked for.
    Read at the store chokepoint inside the same transaction that applies the change,
    so it is a COPY of the row rather than a recollection of it.

    ``value`` is the prior rendered as the display string the field's own surfaces
    use, and ``None`` means the field held NOTHING — a positive statement, which is
    why this is a model per field rather than a bare mapping: "the row had no
    schedule" and "nobody captured the schedule" are different facts, and a mapping
    can only tell them apart by absence.
    """

    field: str
    value: str | None = None


class MutationDetail(BaseModel):
    """The ``detail`` payload of a ``mutation_event`` — *what* changed, serialized
    to the row's JSON column.

    ``changed_fields`` names the edited fields on an update (the values live
    verbatim in the run's ``promptlog`` tool call, so they're not duplicated
    here).  ``priors`` is the other half of that (#1946) — what each of those fields
    held BEFORE, which the ledger is the only place that can hold, since the row
    itself keeps only the value that won.  ``note`` is a human cause the row can't
    otherwise carry — most importantly a system archive's policy reason ("max_runs
    reached (1 of 1)").  ``decision`` is the options-presented accommodation (above).

    ``priors`` is ADDITIVE: an event written before it existed decodes with an empty
    list and every render of it is byte-identical to what it always was.
    """

    changed_fields: list[str] = Field(default_factory=list)
    priors: list[FieldPrior] = Field(default_factory=list)
    note: str | None = None
    decision: EnumeratedDecision | None = None

    def is_empty(self) -> bool:
        return (
            not self.changed_fields
            and not self.priors
            and self.note is None
            and self.decision is None
        )


def cancelled_sends_note(count: int) -> str:
    """The mutation-detail clause naming how many pending queued sends an archive
    cancelled (#1634).

    Folded into the archive event's ``MutationDetail.note`` so the teardown's
    silence is VISIBLE wherever a mutation renders (the self-state activity block,
    ``memory_metadata``'s "Recent changes") — "cancelled N pending send(s)" reads
    as the human cause the row carries, beside any policy reason a system archive
    already notes."""
    return f"cancelled {count} pending send{'' if count == 1 else 's'}"


def render_mutation(event: MutationEvent) -> str:
    """One mutation as a model-readable line, naming its addressable ids (#1560).

    ``<when> <action> by <actor> (run <id>) — <note / changed fields>``.  The run
    id is rendered so the surface is an *anchor* surface: from a change line the
    model is one ``read_run_calls`` hop from the run that made it, never a guess.
    """
    parts = [f"{format_log_timestamp(event.created_at)} {event.action} by {event.actor}"]
    if event.run_id is not None:
        parts.append(f"(run {event.run_id})")
    detail = _parse_detail(event.detail)
    tail = _detail_tail(detail)
    line = " ".join(parts)
    return f"{line} — {tail}" if tail else line


def _parse_detail(raw: str | None) -> MutationDetail | None:
    if not raw:
        return None
    try:
        return MutationDetail.model_validate_json(raw)
    except ValueError:
        logger.warning("Unparseable mutation_event detail: %.200s", raw)
        return None


def _detail_tail(detail: MutationDetail | None) -> str:
    if detail is None:
        return ""
    if detail.note:
        return detail.note
    if detail.changed_fields:
        return f"changed {', '.join(detail.changed_fields)}"
    return ""


def mutation_change_summary(event: MutationEvent) -> str:
    """The human tail of a mutation — its cause note or its changed-field list —
    or ``""`` when the event carries neither (#1555).

    Public sibling of ``render_mutation`` for callers (the self-state header's
    interleaved activity block) that render their own left column — entity + a
    typed event word — and only need the *what changed* tail, not the whole
    ``<when> <action> by <actor>`` line ``render_mutation`` builds for a
    per-entity change history."""
    return _detail_tail(_parse_detail(event.detail))


class MutationStore:
    """Read/write access to the ``mutation_event`` ledger."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def record(
        self,
        *,
        entity_type: MutationEntityType,
        entity_name: str,
        action: MutationAction,
        actor: MutationActor,
        run_id: str | None = None,
        detail: MutationDetail | None = None,
    ) -> None:
        """Append one mutation event.  Best-effort logging (a failed audit write
        must never fail the mutation it records) — mirrors ``log_prompt``."""
        detail_json = (
            detail.model_dump_json() if detail is not None and not detail.is_empty() else None
        )
        try:
            with self._session() as session:
                session.add(
                    MutationEvent(
                        entity_type=entity_type.value,
                        entity_name=entity_name,
                        action=action.value,
                        actor=actor.value,
                        run_id=run_id,
                        detail=detail_json,
                    )
                )
                session.commit()
        except Exception as exc:
            logger.error("Failed to record mutation event for %s: %s", entity_name, exc)

    def history(
        self, entity_name: str, limit: int, *, entity_type: MutationEntityType | None = None
    ) -> list[MutationEvent]:
        """One entity's mutations, newest first — its configuration history in
        time order (criterion 2/4).  Ordered by ``created_at`` (never id).

        ``entity_type`` narrows the name to one KIND of entity, which a caller rendering a
        particular entity's history has to do now that the ledger carries more than one
        (#1902 added skills): a name identifies a row only within its own table, so a
        collection and a routine that happen to share one would otherwise interleave into
        each other's history.  Left off, the read spans every kind — which is what a
        name-only lookup means."""
        if limit <= 0:
            return []
        query = select(MutationEvent).where(MutationEvent.entity_name == entity_name)
        if entity_type is not None:
            query = query.where(MutationEvent.entity_type == entity_type.value)
        with self._session() as session:
            return list(
                session.exec(
                    query.order_by(col(MutationEvent.created_at).desc()).limit(limit)
                ).all()
            )

    def priors_for_run(
        self,
        entity_name: str,
        run_id: str,
        *,
        entity_type: MutationEntityType | None = None,
    ) -> dict[str, str | None]:
        """What the fields ONE run changed on ``entity_name`` held before it ran (#1946)
        — ``{field: prior value}``, empty when the run changed nothing there.

        The events are folded OLDEST FIRST and the first prior for a field wins, so a
        run that touched the same field twice reports the value it found when it
        started rather than the intermediate one it wrote on the way — which is the
        only "was" a user asking about their own job means.

        Events written before priors existed contribute nothing, so this is empty for
        them and the surfaces that read it render exactly as they did."""
        query = select(MutationEvent).where(
            MutationEvent.entity_name == entity_name,
            MutationEvent.run_id == run_id,
        )
        if entity_type is not None:
            query = query.where(MutationEvent.entity_type == entity_type.value)
        with self._session() as session:
            events = list(
                session.exec(
                    query.order_by(col(MutationEvent.created_at).asc(), col(MutationEvent.id).asc())
                ).all()
            )
        priors: dict[str, str | None] = {}
        for event in events:
            detail = _parse_detail(event.detail)
            for prior in detail.priors if detail is not None else []:
                priors.setdefault(prior.field, prior.value)
        return priors

    def recent(self, limit: int) -> list[MutationEvent]:
        """The most recent mutations across ALL entities, newest first (#1555).

        The cross-entity stream the self-state header interleaves with recent
        runs into one time-ordered activity block — "what did you recently do?"
        over configuration changes.  ``history`` scopes to one entity; this one
        spans every entity.  Ordered by ``created_at`` (never id)."""
        if limit <= 0:
            return []
        with self._session() as session:
            return list(
                session.exec(
                    select(MutationEvent)
                    # ``id`` breaks same-timestamp ties deterministically (newest
                    # id first = creation order) so the activity render is stable.
                    .order_by(col(MutationEvent.created_at).desc(), col(MutationEvent.id).desc())
                    .limit(limit)
                ).all()
            )
