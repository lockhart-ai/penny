"""Email-rule store — persistence for provider email-organisation rules (#1737).

Domain store for the ``email_rule`` table: create a rule, list a provider's
active (enabled) rules, and stamp one applied.  The typed condition/action
Pydantic models live in the owning plugin; this store deals only in the row's
serialized JSON strings, so the database layer never imports a plugin model.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.models import EmailRule

logger = logging.getLogger(__name__)


class EmailRuleStore:
    """Create, list-active, and mark-applied for persisted email rules."""

    def __init__(self, engine):
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def create(self, provider: str, name: str, condition: str, action: str) -> EmailRule:
        """Persist a new enabled rule; ``condition``/``action`` are serialized JSON."""
        with self._session() as session:
            rule = EmailRule(
                provider=provider,
                name=name,
                condition=condition,
                action=action,
                enabled=True,
                created_at=datetime.now(UTC),
            )
            session.add(rule)
            session.commit()
            session.refresh(rule)
            logger.info("Created email rule '%s' for provider %s", name, provider)
            return rule

    def list_active(self, provider: str) -> list[EmailRule]:
        """Every enabled rule for a provider, oldest-first."""
        with self._session() as session:
            return list(
                session.exec(
                    select(EmailRule)
                    .where(EmailRule.provider == provider)
                    .where(EmailRule.enabled == True)  # noqa: E712
                    .order_by(EmailRule.created_at.asc())  # ty: ignore[unresolved-attribute]
                )
            )

    def mark_applied(self, rule_id: int) -> None:
        """Stamp ``last_applied_at`` after a rule ran against an email batch."""
        with self._session() as session:
            rule = session.get(EmailRule, rule_id)
            if rule is None:
                return
            rule.last_applied_at = datetime.now(UTC)
            session.add(rule)
            session.commit()
            logger.debug("Marked email rule %d applied", rule_id)
