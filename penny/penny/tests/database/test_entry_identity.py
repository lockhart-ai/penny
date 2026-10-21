"""The entry-identity ladder's decision step (#2715), decider stubbed.

The deterministic rules are pinned in ``tests/test_similarity.py``; what this
module pins is the DECISION step around them — the shortlist feeding one choice
question per candidate, the outcome mapping (``same_value`` → unchanged,
``new_value`` → merge, ``different`` → new), the honest question for ``unsure``
and for the named degraded mode, and the rule-only path the manual write
surface takes.  All fixtures are invented scenarios with real proper nouns.
"""

from __future__ import annotations

from typing import Any

import pytest

from penny.database.memory.identity import (
    SHORTLIST_K,
    EntryDecider,
    EntryGiven,
    plan_entries,
    rule_only_decisions,
)
from penny.database.memory.types import EntryInput
from penny.llm.decision_client import DecisionAnswer, DecisionClient, DecisionError


class _StubDecisionClient(DecisionClient):
    """A decision model answering each question with the next canned choice,
    recording the states it was shown."""

    def __init__(self, *choices: str) -> None:
        self._choices = list(choices)
        self.states: list[str] = []

    async def decide(self, **kwargs: Any) -> DecisionAnswer:
        self.states.append(str(kwargs.get("state")))
        return DecisionAnswer(choice=self._choices.pop(0))


class _FailingDecisionClient(DecisionClient):
    def __init__(self) -> None:
        super().__init__(model="stub", api_url="http://localhost:1")

    async def decide(self, **_: Any) -> DecisionAnswer:
        raise DecisionError("endpoint down")


def _given(key: str, content: str) -> EntryGiven:
    return EntryGiven(key=key, content=content, key_vec=[1.0, 0.0], content_vec=[1.0, 0.0])


def _entry(key: str, content: str) -> EntryInput:
    return EntryInput(
        key=key, content=content, key_embedding=[1.0, 0.0], content_embedding=[1.0, 0.0]
    )


def _decide(stub: DecisionClient, entries: list[EntryInput], existing: list[EntryGiven]):
    plans = plan_entries(entries, existing)
    return EntryDecider(stub).decide(
        plans=plans,
        entries=entries,
        existing=existing,
        collection_name="anime_to_watch",
        collection_description="Anime the user wants to watch",
    )


@pytest.mark.asyncio
async def test_new_value_merges_into_the_matched_entry() -> None:
    """``new_value`` → merge: the same thing carrying a different value replaces
    the stored one; the decision names the matched key."""
    entries = [_entry("genesis p-orridge", "Genesis P-Orridge of Throbbing Gristle")]
    existing = [_given("genesis", "Genesis, the prog band")]
    decisions = await _decide(_StubDecisionClient("new_value"), entries, existing)
    assert decisions[0].action == "merge"
    assert decisions[0].matched_key == "genesis"
    assert decisions[0].unsure_candidate is None


@pytest.mark.asyncio
async def test_same_value_reports_no_news() -> None:
    entries = [_entry("Pink Floyd (band)", "Pink Floyd, the prog band")]
    existing = [_given("pink floyd", "Pink Floyd, the prog band")]
    decisions = await _decide(_StubDecisionClient("same_value"), entries, existing)
    assert decisions[0].action == "unchanged"
    assert decisions[0].matched_key == "pink floyd"


@pytest.mark.asyncio
async def test_different_writes_through_quietly() -> None:
    entries = [_entry("Ghost in the Shell", "the 1995 film")]
    existing = [_given("ghost hound", "the 2007 series")]
    decisions = await _decide(_StubDecisionClient("different"), entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate is None


@pytest.mark.asyncio
async def test_unsure_writes_through_and_names_the_candidate() -> None:
    """``unsure`` is treated as different — the write proceeds — and the first
    unsure candidate is recorded for the honest question in the reply."""
    entries = [_entry("genesis p-orridge", "Genesis P-Orridge of Throbbing Gristle")]
    existing = [_given("genesis", "Genesis, the prog band")]
    decisions = await _decide(_StubDecisionClient("unsure"), entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate == "genesis"


@pytest.mark.asyncio
async def test_an_unsure_candidate_does_not_stop_a_later_decisive_answer() -> None:
    """A shortlist ranked best-first: an ``unsure`` on one candidate leaves the
    question open, and the next candidate's decisive answer settles it."""
    entries = [_entry("genesis", "the band that became Genesis")]
    existing = [
        _given("genesis p-orridge", "Genesis P-Orridge of Throbbing Gristle"),
        _given("genesis the band", "Genesis, the prog band"),
    ]
    decisions = await _decide(_StubDecisionClient("unsure", "new_value"), entries, existing)
    assert decisions[0].action == "merge"
    assert decisions[0].matched_key == "genesis the band"


@pytest.mark.asyncio
async def test_the_shortlist_bounds_the_questions_asked() -> None:
    """One question per shortlisted candidate, at most ``SHORTLIST_K`` — a
    populated collection does not become one decision-model call per entry."""
    words = [
        "apple",
        "bread",
        "cider",
        "dough",
        "elm",
        "fern",
        "grape",
        "hazel",
        "iris",
        "juniper",
        "kelp",
        "lime",
        "moss",
        "nettle",
        "olive",
        "pine",
        "quince",
        "reed",
        "sage",
        "thyme",
    ]
    entries = [_entry("a new topic", "something unrelated")]
    existing = [_given(f"topic {word}", f"stored thing about {word}") for word in words]
    stub = _StubDecisionClient(*(["different"] * SHORTLIST_K))
    decisions = await _decide(stub, entries, existing)
    assert decisions[0].action == "new"
    assert len(stub.states) == SHORTLIST_K


@pytest.mark.asyncio
async def test_a_failed_decision_call_degrades_to_unsure() -> None:
    """An unreachable endpoint is logged and answered as ``unsure`` — never a
    silently-settled decision, and the write still lands (visible degradation)."""
    entries = [_entry("genesis p-orridge", "Genesis P-Orridge of Throbbing Gristle")]
    existing = [_given("genesis", "the prog band")]
    decisions = await _decide(_FailingDecisionClient(), entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate == "genesis"


@pytest.mark.asyncio
async def test_the_degraded_mode_asks_about_the_shortlist_head() -> None:
    """No decision model configured — the named degraded mode: the rules stand,
    every shortlisted candidate is ``unsure``, and the head of the shortlist is
    the candidate the reply will name."""
    entries = [_entry("genesis p-orridge", "Genesis P-Orridge of Throbbing Gristle")]
    existing = [_given("genesis", "the prog band")]
    plans = plan_entries(entries, existing)
    decisions = rule_only_decisions(plans, entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate == "genesis"


@pytest.mark.asyncio
async def test_the_degraded_mode_still_settles_by_the_rules() -> None:
    """The rules run identically without a model: normalised keys equal → merge
    (the store replaces the value), identical content → unchanged."""
    entries = [_entry("Genesis (band)", "a different description"), _entry("gybe", "same text")]
    existing = [
        _given("genesis", "the prog band"),
        _given("godspeed you black emperor", "same text"),
    ]
    plans = plan_entries(entries, existing)
    decisions = rule_only_decisions(plans, entries, existing)
    assert [d.action for d in decisions] == ["merge", "unchanged"]
    assert decisions[0].matched_key == "genesis"
    assert decisions[1].matched_key == "godspeed you black emperor"


@pytest.mark.asyncio
async def test_the_date_rule_never_reaches_the_decision_model() -> None:
    """The production regression: a daily-digest job writing tomorrow's entry is
    settled DIFFERENT by the rules — no candidate is shortlisted, so the
    decision model is never asked and nothing is ever merged across dates."""
    entries = [_entry("2026-10-21", "the October 21 digest")]
    existing = [_given("2026-10-20", "the October 20 digest")]
    stub = _StubDecisionClient("same_value")  # would wrongly merge if consulted
    decisions = await _decide(stub, entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate is None
    assert stub.states == []
