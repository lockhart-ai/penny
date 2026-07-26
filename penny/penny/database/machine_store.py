"""``MachineStore`` — the conversation state machine's durable half (#1706).

The classifier decides one transition per incoming message, but a decision that
evaporates at the end of the turn is not a machine: every parked state
(``elicit`` / ``learn`` / ``request``) exists precisely to be READ by the NEXT
message's classification, so the state has to outlive the turn that set it.
This store is where it lives.

Two shapes, the same split the mutation ledger draws:

1. ``conversation_machine`` — the materialized truth (state + the anchoring
   message), one row, read at the top of every turn.
2. ``state_transition`` — the append-only audit of how it got there, one row per
   move INCLUDING the moves that moved nothing (a held draw is the signal you
   need to score the classifier; see ``StateTransition``).

Deals in plain state STRINGS on purpose: ``conversation_machine.py`` imports the
database package only function-locally to stay a leaf, so the typing seam lives
there and this layer never imports the state enum — the same discipline
``micro_context`` keeps.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.constants import TransitionCause
from penny.database.models import ConversationMachineRow, StateTransition

logger = logging.getLogger(__name__)


class MachineStore:
    """Read/write access to the machine's current state + its transition ledger."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def current(self, default_state: str) -> ConversationMachineRow:
        """The machine's row, creating it at ``default_state`` on first read.

        Lazy creation rather than a seeded row: a migration may not run against
        a database whose default state this module has since renamed, and the
        caller owns the enum — so the machine's cold start is idle by the
        CALLER's definition of idle, not by a value frozen into DDL."""
        with self._session() as session:
            row = self._row(session)
            if row is not None:
                return row
            row = ConversationMachineRow(state=default_state)
            session.add(row)
            session.commit()
            session.refresh(row)
            return row

    def advance_to(
        self,
        *,
        state: str,
        anchor_message_id: int | None,
        default_state: str,
    ) -> None:
        """Move the machine's materialized state, stamping ``updated_at``.

        The anchor is written on EVERY advance (never merged with the prior
        value) so that clearing it — what returning to idle does — is the same
        operation as setting it, and a stale anchor cannot survive a state it no
        longer belongs to."""
        with self._session() as session:
            row = self._row(session) or ConversationMachineRow(state=default_state)
            row.state = state
            row.anchor_message_id = anchor_message_id
            row.updated_at = datetime.now(UTC)
            session.add(row)
            session.commit()

    def record_transition(
        self,
        *,
        from_state: str,
        to_state: str,
        cause: TransitionCause,
        outcome: str | None = None,
        message_id: int | None = None,
        run_id: str | None = None,
        skill_name: str | None = None,
    ) -> None:
        """Append one transition event.  Best-effort logging — a failed audit
        write must never fail the move it records (mirrors ``MutationStore``)."""
        try:
            with self._session() as session:
                session.add(
                    StateTransition(
                        from_state=from_state,
                        to_state=to_state,
                        cause=cause.value,
                        outcome=outcome,
                        message_id=message_id,
                        run_id=run_id,
                        skill_name=skill_name,
                    )
                )
                session.commit()
        except Exception as exc:
            logger.error("Failed to record state transition %s → %s: %s", from_state, to_state, exc)

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

    @staticmethod
    def _row(session: Session) -> ConversationMachineRow | None:
        """The single machine row (lowest id wins — v1 is one active machine)."""
        return session.exec(
            select(ConversationMachineRow).order_by(ConversationMachineRow.id.asc())  # type: ignore[union-attr]
        ).first()
