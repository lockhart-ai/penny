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
from sqlmodel import Session, select

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


class MutationDetail(BaseModel):
    """The ``detail`` payload of a ``mutation_event`` — *what* changed, serialized
    to the row's JSON column.

    ``changed_fields`` names the edited fields on an update (the values live
    verbatim in the run's ``promptlog`` tool call, so they're not duplicated
    here).  ``note`` is a human cause the row can't otherwise carry — most
    importantly a system archive's policy reason ("max_runs reached (1 of 1)").
    ``decision`` is the options-presented accommodation (above).
    """

    changed_fields: list[str] = Field(default_factory=list)
    note: str | None = None
    decision: EnumeratedDecision | None = None

    def is_empty(self) -> bool:
        return not self.changed_fields and self.note is None and self.decision is None


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

    def history(self, entity_name: str, limit: int) -> list[MutationEvent]:
        """One entity's mutations, newest first — its configuration history in
        time order (criterion 2/4).  Ordered by ``created_at`` (never id)."""
        if limit <= 0:
            return []
        with self._session() as session:
            return list(
                session.exec(
                    select(MutationEvent)
                    .where(MutationEvent.entity_name == entity_name)
                    .order_by(MutationEvent.created_at.desc())  # type: ignore[union-attr]
                    .limit(limit)
                ).all()
            )

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
                    .order_by(MutationEvent.created_at.desc(), MutationEvent.id.desc())  # type: ignore[union-attr]
                    .limit(limit)
                ).all()
            )
