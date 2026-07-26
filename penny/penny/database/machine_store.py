"""``MachineStore`` — the conversation state machine's durable half (#1706).

The classifier decides one transition per incoming message, but a decision that
evaporates at the end of the turn is not a machine: every parked state
(``elicit`` / ``learn`` / ``request``) exists precisely to be READ by the NEXT
message's classification, so the state has to outlive the turn that set it.
This store is where it lives.

**One table, no materialized twin.** ``state_transition`` is an append-only log
and the machine's whole state is a fold over it — the newest row's ``to_state``
IS where the machine stands, its ``anchor_message_id`` IS the ask a parked round
is anchored to, its ``created_at`` IS when the machine last moved.  So there is
no second row holding a current-state copy: a materialized twin here would carry
*nothing that isn't derivable*, and keeping it would mean two writes per move
that can disagree (the mutation ledger's ``memory`` twin earns itself only
because a ``memory`` row carries name/description/prompt/cadence — facts no
event derives; the shape does not transfer).  One write per move means a failed
write moves nothing, so the state and its audit trail cannot drift.

``from_state`` beside ``to_state`` is deliberate denormalization — the previous
row's ``to_state`` would give it by join, but on an append-only log a row that
states its own whole move is self-describing and cheaper to read.

Deals in plain state STRINGS on purpose: ``conversation_machine.py`` imports the
database package only function-locally to stay a leaf, so the typing seam lives
there and this layer never imports the state enum — the same discipline
``micro_context`` keeps.
"""

from __future__ import annotations

from sqlmodel import Session, select

from penny.constants import TransitionCause
from penny.database.models import StateTransition


class MachineStore:
    """Read/write access to the machine's transition log — its only state."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def latest_transition(self) -> StateTransition | None:
        """The newest move — the machine's current state, anchor, and last-moved
        time in one row.  ``None`` means the machine has never moved, which IS
        the cold start: no row is seeded and none is lazily created, so "nothing
        has happened yet" is represented by the absence of history rather than
        by a row asserting it."""
        with self._session() as session:
            return session.exec(
                select(StateTransition).order_by(
                    StateTransition.created_at.desc(),  # type: ignore[union-attr]
                    StateTransition.id.desc(),  # type: ignore[union-attr]
                )
            ).first()

    def record_transition(
        self,
        *,
        from_state: str,
        to_state: str,
        cause: TransitionCause,
        anchor_message_id: int | None = None,
        outcome: str | None = None,
        message_id: int | None = None,
        run_id: str | None = None,
        skill_name: str | None = None,
    ) -> None:
        """Append one move — the single write that BOTH advances the machine and
        records how it moved.

        Deliberately NOT best-effort (unlike ``MutationStore.record``, whose
        swallowed failure protects the mutation it merely audits): this row *is*
        the state, so a swallowed failure here would silently lose the move
        itself.  It raises, and the caller's move fails with it."""
        with self._session() as session:
            session.add(
                StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    cause=cause.value,
                    anchor_message_id=anchor_message_id,
                    outcome=outcome,
                    message_id=message_id,
                    run_id=run_id,
                    skill_name=skill_name,
                )
            )
            session.commit()

    def link_message(self, run_id: str, message_id: int, *, anchor: bool) -> None:
        """Back-fill the message a run's moves were provoked by, once it has an id.

        The channel logs the incoming message AFTER the turn runs — deliberately,
        so it never doubles into the turn's own recall — but the machine has to
        classify BEFORE the turn.  So the id doesn't exist at write time and is
        linked here instead, matched on the run id, exactly as
        ``link_source_message`` links a spawning message to the mechanism a run
        created (#1566).

        ``anchor`` additionally stamps it as the round's anchor, for the rows
        that opened a round; the caller owns that rule (see
        ``ConversationMachine._link_message``) so anchoring is decided in one
        place, not re-derived here."""
        with self._session() as session:
            rows = session.exec(
                select(StateTransition).where(StateTransition.run_id == run_id)
            ).all()
            for row in rows:
                row.message_id = message_id
                if anchor:
                    row.anchor_message_id = message_id
                session.add(row)
            session.commit()

    def recent_transitions(self, limit: int) -> list[StateTransition]:
        """The machine's most recent moves, newest first — the replay surface.

        Ordered by ``created_at`` (never id), with id breaking same-timestamp
        ties so the sequence is stable: two moves in one turn (a structural
        reset then the classified move that followed it) share a timestamp at
        SQLite's resolution and must not reorder between reads."""
        if limit <= 0:
            return []
        with self._session() as session:
            return list(
                session.exec(
                    select(StateTransition)
                    .order_by(
                        StateTransition.created_at.desc(),  # type: ignore[union-attr]
                        StateTransition.id.desc(),  # type: ignore[union-attr]
                    )
                    .limit(limit)
                ).all()
            )
