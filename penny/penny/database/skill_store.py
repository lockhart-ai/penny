"""``SkillStore`` — persistence for the versionless skill table (#1590).

One row per skill name (no versioning): ``upsert`` creates a skill or REPLACES an
existing one by name, reporting which happened so the tool can say "replaced the
previous version of <name>".  The store owns (de)serialization of the structured
``steps`` / ``holes`` JSON — callers hand it a :class:`SkillDraft` and read back
hydrated :class:`SkillStep` / :class:`SkillHole` objects, never raw JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from sqlmodel import Session, select

from penny.database.memory import _similarity as sim
from penny.database.models import Skill
from penny.database.skills import SkillDraft, SkillHole, SkillStep, slug_skill_name

logger = logging.getLogger(__name__)


def steps_from_json(raw: str) -> list[SkillStep]:
    """Hydrate a skill row's ``steps`` JSON into structured steps."""
    return [SkillStep(**item) for item in json.loads(raw)]


def holes_from_json(raw: str) -> list[SkillHole]:
    """Hydrate a skill row's ``holes`` JSON into declared holes."""
    return [SkillHole(**item) for item in json.loads(raw)]


def steps_to_json(steps: list[SkillStep]) -> str:
    """Serialize structured steps for storage (the ``LoggedToolCall`` shape)."""
    return json.dumps([step.model_dump() for step in steps])


def holes_to_json(holes: list[SkillHole]) -> str:
    """Serialize declared holes for storage."""
    return json.dumps([hole.model_dump() for hole in holes])


class SkillStore:
    """Registry for skills — upsert-by-name, get, list.  ``db.skills``."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine)

    def upsert(
        self,
        draft: SkillDraft,
        *,
        author: str,
        description_embedding: list[float] | None = None,
    ) -> tuple[Skill, bool]:
        """Create the skill, or REPLACE an existing one of the same name.

        Returns ``(skill, replaced)`` — ``replaced`` is ``True`` when a prior skill
        of that name was overwritten (its steps/holes/provenance swapped for the
        new demonstration), so the caller can report the replacement.  ``name`` is
        the unique key; there is no version history (collections carry the rendered
        text, so a re-teach never changes a past instantiation).
        """
        name = slug_skill_name(draft.name)
        now = datetime.now(UTC)
        with self._session() as session:
            existing = session.get(Skill, name)
            replaced = existing is not None
            skill = existing or Skill(
                name=name, steps="", holes="", intent="", description="", author=author
            )
            skill.steps = steps_to_json(draft.steps)
            skill.holes = holes_to_json(draft.holes)
            skill.intent = draft.intent
            skill.description = draft.description
            skill.description_embedding = sim.maybe_serialize(description_embedding)
            skill.source_run_id = draft.source_run_id
            skill.author = author
            skill.updated_at = now
            if not replaced:
                skill.created_at = now
            session.add(skill)
            session.commit()
            session.refresh(skill)
        logger.debug("%s skill %s", "Replaced" if replaced else "Created", name)
        return skill, replaced

    def get(self, name: str) -> Skill | None:
        with self._session() as session:
            return session.get(Skill, slug_skill_name(name))

    def list_all(self) -> list[Skill]:
        """Every skill, name order — the read surface's catalog listing."""
        with self._session() as session:
            return list(session.exec(select(Skill).order_by(Skill.name)).all())
