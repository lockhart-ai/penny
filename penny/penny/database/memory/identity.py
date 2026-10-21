"""The entry-identity ladder — rules, shortlist, decision model, honest question.

Whether a new entry is the same as a stored one is decided here, in one place,
so every caller of the write path gets the same answer.  The ladder has three
moving parts, split so no blocking HTTP ever runs on the event loop:

  * :func:`plan_entries` — the sync, pure part: the deterministic rules
    (:mod:`similarity.identity`) settle every pair they are sure of, and the
    embedding shortlist ranks the few stored entries a candidate still needs to
    be asked about.
  * :class:`EntryDecider.decide` — the async part: per shortlisted candidate,
    an identity question (the same specific thing, or two items side by side)
    and — only on yes — a value question, to the decision model
    (``penny.llm.decision_client``).
    No client configured is the *named* degraded mode — the decision step is
    skipped and every shortlisted candidate becomes ``unsure`` — and a failed
    call degrades the same way, logged, never silently settled.
  * The store's write — the sync part — receives the :class:`EntryDecision`s as
    a required parameter and applies them.  This module owns the decision; the
    store only applies it.

Outcome rules: unsure is treated as different (a wrong "different" leaves a
duplicate, visible; a wrong "same" loses a write); every merge names the
matched key and the prior so a wrong match can be seen and reversed.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Literal, NamedTuple

from pydantic import BaseModel
from similarity.identity import (
    KeyRule,
    ShortlistSide,
    pair_rule_verdict,
    shortlist,
)

from penny.database.memory.types import EntryInput
from penny.llm.decision_client import DecisionClient, DecisionError

logger = logging.getLogger(__name__)

# How many shortlisted candidates a decision-model question may be asked about.
# Ranking only above this; the rest of the store is simply not a candidate.
SHORTLIST_K = 3

# Both sides of the question are clipped so a long digest entry cannot blow the
# question up; identity lives in the key and the opening of the value.
_STATE_CLIP = 600


class EntryGiven(NamedTuple):
    """One stored entry as the ladder sees it — key, value, and the two
    embeddings the shortlist ranks on."""

    key: str | None
    content: str
    key_vec: list[float] | None
    content_vec: list[float] | None


class IdentityAnswer(StrEnum):
    """The decision model's identity answer for one candidate/new-entry pair:
    is the new entry the same specific thing as the stored one?"""

    YES = "yes"
    NO = "no"
    UNSURE = "unsure"


class ValueAnswer(StrEnum):
    """The decision model's value answer, asked only when identity said yes:
    does the stored entry already say everything the new entry says?"""

    SAME_VALUE = "same_value"
    NEW_VALUE = "new_value"
    UNSURE = "unsure"


class EntryPlan(NamedTuple):
    """One entry's position after the deterministic part of the ladder.

    ``matched_key`` set means the rules settled the pair as the same entry (and
    ``value_identical`` says whether the stored value is already this one);
    ``candidates`` are the shortlisted store indices the decision model should
    be asked about, best-first, with the rules-different pairs already removed.
    """

    matched_key: str | None
    value_identical: bool
    candidates: list[int]


class EntryDecision(BaseModel):
    """The final decision for one entry, applied by the store's write.

    ``action``: ``new`` — write under its own key; ``merge`` — the same entry
    under ``matched_key`` with a different value, so the stored value is
    replaced; ``unchanged`` — the stored entry already says this.  An ``unsure``
    candidate rides a ``new`` decision: the write proceeds, and the reply asks
    the writing model honestly whether it meant the candidate instead."""

    key: str
    action: Literal["new", "merge", "unchanged"]
    matched_key: str | None = None
    unsure_candidate: str | None = None


def plan_entries(entries: list[EntryInput], existing: list[EntryGiven]) -> list[EntryPlan]:
    """The deterministic part of the ladder, for a whole write batch: rules
    first, then the shortlist for whatever is left."""
    return [_plan_one(entry, existing) for entry in entries]


def _plan_one(entry: EntryInput, existing: list[EntryGiven]) -> EntryPlan:
    matched_key: str | None = None
    value_identical = False
    excluded: set[int] = set()
    for index, side in enumerate(existing):
        rule = _pair_rule(entry, side)
        if rule is KeyRule.DIFFERENT:
            excluded.add(index)
            continue
        if rule is KeyRule.SAME:
            matched_key = side.key
            value_identical = entry.content.strip() == side.content.strip()
            break
    candidates: list[int] = []
    if matched_key is None:
        ranked = shortlist(
            entry.key_embedding,
            entry.content_embedding,
            [ShortlistSide(side.key_vec, side.content_vec) for side in existing],
            SHORTLIST_K,
        )
        candidates = [index for index in ranked if index not in excluded]
    return EntryPlan(
        matched_key=matched_key, value_identical=value_identical, candidates=candidates
    )


def _pair_rule(entry: EntryInput, side: EntryGiven) -> KeyRule | None:
    """The rules for one candidate/stored pair.  A stored entry without a key
    cannot take the key rules and settles nothing deterministically — content
    settles nothing by rule either, so a keyless entry only ever reaches the
    shortlist and the decision model."""
    if side.key is None:
        return None
    return pair_rule_verdict(entry.key, side.key, entry.content, side.content)


# Two questions, asked in order (framing measured in the #2715 review probe:
# a single 4-way "how does the new write relate to the stored entry" question
# reads as "is this new information for the collection" and merges unrelated
# entries; separating IDENTITY from VALUE keeps distinct items apart).
_IDENTITY_INSTRUCTIONS = (
    "Two entries from the same collection are shown.  Each entry in a collection "
    "is its own item: a collection holds many separate things, side by side.  "
    "Is the new entry the same specific thing as the stored entry, merely named "
    "or worded differently?"
)
_IDENTITY_CRITERIA = {
    "yes": "the same specific thing, under another name or wording",
    "no": "two different specific things that both belong in the collection",
}
_VALUE_INSTRUCTIONS = (
    "The stored entry and the new entry are the same specific thing.  Compare "
    "what each one says.  If the new entry says anything the stored entry does "
    "not already say — an added detail, a changed number, a newer observation — "
    "the value is new.  The value is the same only when the stored entry already "
    "says everything the new entry says, even in different words."
)
_VALUE_CRITERIA = {
    "same_value": "the stored entry already says everything the new entry says",
    "new_value": "the new entry adds, changes, or updates something the stored entry does not say",
}
_IDENTITY_BY_CHOICE = {answer.value: answer for answer in IdentityAnswer}
_VALUE_BY_CHOICE = {answer.value: answer for answer in ValueAnswer}


def _question_state(
    entry: EntryInput,
    existing: EntryGiven,
    collection_name: str,
    collection_description: str | None,
) -> str:
    return (
        f"Collection '{collection_name}': {collection_description or 'no description'}\n"
        f"Stored entry (key '{existing.key}'): {_clip(existing.content)}\n"
        f"New entry (key '{entry.key}'): {_clip(entry.content)}"
    )


def _clip(text: str) -> str:
    if len(text) <= _STATE_CLIP:
        return text
    return text[:_STATE_CLIP] + "…"


class EntryDecider:
    """The decision-model step of the ladder.

    Holds the :class:`DecisionClient`, or ``None`` for the named degraded mode
    ("entry matching falls back to asking").  Constructed once, from config, at
    startup — a required value threaded to the write tools, never read from
    ambient state."""

    def __init__(self, client: DecisionClient | None) -> None:
        self._client = client

    async def decide(
        self,
        *,
        plans: list[EntryPlan],
        entries: list[EntryInput],
        existing: list[EntryGiven],
        collection_name: str,
        collection_description: str | None,
    ) -> list[EntryDecision]:
        """Settle each planned entry: a rules-same pair settles outright; the
        shortlisted candidates go to the decision model in rank order — identity
        first, value only on yes — and the first decisive answer wins; every
        candidate left unsure is recorded for the honest question."""
        if self._client is None:
            return rule_only_decisions(plans, entries, existing)
        decisions: list[EntryDecision] = []
        for entry, plan in zip(entries, plans, strict=True):
            decisions.append(
                await self._decide_one(
                    entry, plan, existing, collection_name, collection_description
                )
            )
        return decisions

    async def _decide_one(
        self,
        entry: EntryInput,
        plan: EntryPlan,
        existing: list[EntryGiven],
        collection_name: str,
        collection_description: str | None,
    ) -> EntryDecision:
        if plan.matched_key is not None:
            return EntryDecision(
                key=entry.key,
                action="unchanged" if plan.value_identical else "merge",
                matched_key=plan.matched_key,
            )
        unsure_candidate: str | None = None
        for index in plan.candidates:
            side = existing[index]
            identity = await self._ask_identity(
                entry, side, collection_name, collection_description
            )
            if identity is IdentityAnswer.YES:
                value = await self._ask_value(entry, side, collection_name, collection_description)
                if value is ValueAnswer.SAME_VALUE:
                    return EntryDecision(key=entry.key, action="unchanged", matched_key=side.key)
                if value is ValueAnswer.NEW_VALUE:
                    return EntryDecision(key=entry.key, action="merge", matched_key=side.key)
                if unsure_candidate is None:
                    unsure_candidate = side.key
                continue
            if identity is IdentityAnswer.UNSURE and unsure_candidate is None:
                unsure_candidate = side.key
            # a decided NO moves on to the next candidate quietly
        return EntryDecision(key=entry.key, action="new", unsure_candidate=unsure_candidate)

    async def _ask_identity(
        self,
        entry: EntryInput,
        existing: EntryGiven,
        collection_name: str,
        collection_description: str | None,
    ) -> IdentityAnswer:
        """The identity question about one candidate — the same specific thing,
        or two items that both belong in the collection.  A failed or malformed
        call degrades to ``unsure`` — logged, never silently settled."""
        answer = await self._ask_raw(
            entry,
            existing,
            collection_name,
            collection_description,
            _IDENTITY_INSTRUCTIONS,
            _IDENTITY_CRITERIA,
            "identity",
        )
        verdict = _IDENTITY_BY_CHOICE.get(answer)
        if verdict is None:
            logger.warning(
                "entry identity: decision model chose unknown option %r for '%s' vs '%s'",
                answer,
                entry.key,
                existing.key,
            )
            return IdentityAnswer.UNSURE
        return verdict

    async def _ask_value(
        self,
        entry: EntryInput,
        existing: EntryGiven,
        collection_name: str,
        collection_description: str | None,
    ) -> ValueAnswer:
        """The value question, asked only after identity said yes: does the
        stored entry already say everything the new entry says?"""
        answer = await self._ask_raw(
            entry,
            existing,
            collection_name,
            collection_description,
            _VALUE_INSTRUCTIONS,
            _VALUE_CRITERIA,
            "value",
        )
        verdict = _VALUE_BY_CHOICE.get(answer)
        if verdict is None:
            logger.warning(
                "entry identity: decision model chose unknown option %r for the value "
                "question on '%s' vs '%s'",
                answer,
                entry.key,
                existing.key,
            )
            return ValueAnswer.UNSURE
        return verdict

    async def _ask_raw(
        self,
        entry: EntryInput,
        existing: EntryGiven,
        collection_name: str,
        collection_description: str | None,
        instructions: str,
        criteria: dict[str, str],
        question: str,
    ) -> str:
        """One choice question to the decision model, answered with the raw
        choice string.  A failed call returns ``""`` — unreadable, and every
        caller treats that as unsure."""
        assert self._client is not None  # decide() routes the degraded mode elsewhere
        try:
            answer = await self._client.decide(
                state=_question_state(entry, existing, collection_name, collection_description),
                instructions=instructions,
                criteria=criteria,
            )
        except DecisionError as error:
            logger.warning(
                "entry identity: decision model unavailable for the %s question on "
                "'%s' vs '%s': %s",
                question,
                entry.key,
                existing.key,
                error,
            )
            return ""
        return answer.choice.strip().lower()


def rule_only_decisions(
    plans: list[EntryPlan], entries: list[EntryInput], existing: list[EntryGiven]
) -> list[EntryDecision]:
    """Decisions without a decision model: the rules stand, and every shortlisted
    candidate is ``unsure`` — the honest question, in the named degraded mode.

    This is also the manual write path's identity (the addon and iOS entry
    writes, where no model is in the loop): the rules run identically there, and
    what the rules cannot settle is written as new."""
    decisions: list[EntryDecision] = []
    for entry, plan in zip(entries, plans, strict=True):
        if plan.matched_key is not None:
            decisions.append(
                EntryDecision(
                    key=entry.key,
                    action="unchanged" if plan.value_identical else "merge",
                    matched_key=plan.matched_key,
                )
            )
            continue
        unsure = next((existing[index].key for index in plan.candidates), None)
        decisions.append(EntryDecision(key=entry.key, action="new", unsure_candidate=unsure))
    return decisions
