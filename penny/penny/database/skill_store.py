"""``SkillStore`` — persistence for the versionless skill table (#1590).

One row per skill name (no versioning): ``upsert`` creates a skill or REPLACES an
existing one by name, reporting which happened so the tool can say "replaced the
previous version of <name>".  The store owns (de)serialization of the structured
``steps`` / ``parameters`` JSON — callers hand it a :class:`SkillDraft` and read
back hydrated :class:`SkillStep` / :class:`SkillParameter` objects, never raw JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
from sqlmodel import Session, select

from penny.database.memory import _similarity as sim
from penny.database.models import Skill
from penny.database.skills import SkillDraft, SkillParameter, SkillStep, slug_skill_name

logger = logging.getLogger(__name__)


def steps_from_json(raw: str) -> list[SkillStep]:
    """Hydrate a skill row's ``steps`` JSON into structured steps."""
    return [SkillStep(**item) for item in json.loads(raw)]


def parameters_from_json(raw: str) -> list[SkillParameter]:
    """Hydrate a skill row's ``parameters`` JSON into declared parameters."""
    return [SkillParameter(**item) for item in json.loads(raw)]


def steps_to_json(steps: list[SkillStep]) -> str:
    """Serialize structured steps for storage (the ``LoggedToolCall`` shape)."""
    return json.dumps([step.model_dump() for step in steps])


def parameters_to_json(parameters: list[SkillParameter]) -> str:
    """Serialize declared parameters for storage."""
    return json.dumps([parameter.model_dump() for parameter in parameters])


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
        of that name was overwritten (its steps/parameters/provenance swapped for
        the new demonstration), so the caller can report the replacement.  ``name``
        is the unique key; there is no version history (collections carry the
        rendered text, so a re-teach never changes a past instantiation).
        """
        name = slug_skill_name(draft.name)
        now = datetime.now(UTC)
        with self._session() as session:
            existing = session.get(Skill, name)
            replaced = existing is not None
            skill = existing or Skill(
                name=name, steps="", parameters="", intent="", description="", author=author
            )
            skill.steps = steps_to_json(draft.steps)
            skill.parameters = parameters_to_json(draft.parameters)
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

    def resolve_by_meaning(self, anchor: list[float], limit: int) -> list[Skill]:
        """Skills ranked by description-anchor cosine to ``anchor``, best-first
        (#1591's resolve-by-meaning leg — the 'or meaning' half of name-or-meaning).

        Plain-cosine nearest-neighbour over each skill's ``description_embedding``
        (populated at write), positively-correlated only (cosine > 0 — an
        unrelated skill isn't a candidate, so an off-topic query returns empty →
        NO_SKILL_FOUND), capped at ``limit``.  A skill missing its vector is absent
        (never surfaced unscored).  The caller decides MATCHED vs. AMBIGUOUS vs.
        NO_SKILL from the exact-name lookup + this ranking."""
        scored: list[Skill] = []
        blobs: list[bytes] = []
        for skill in self.list_all():
            if skill.description_embedding is None:
                continue
            scored.append(skill)
            blobs.append(skill.description_embedding)
        if not scored:
            return []
        scores = sim.cosine_scores(blobs, anchor)
        order = list(np.argsort(-scores))
        return [scored[i] for i in order if float(scores[i]) > 0.0][:limit]
