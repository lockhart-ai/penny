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
from similarity.identity import KeyRule

from penny.database.memory.identity import (
    SHORTLIST_K,
    CollectionCandidate,
    CollectionNameDecider,
    EntryDecider,
    EntryGiven,
    collection_name_rule,
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
    decisions = await _decide(_StubDecisionClient("yes", "new_value"), entries, existing)
    assert decisions[0].action == "merge"
    assert decisions[0].matched_key == "genesis"
    assert decisions[0].unsure_candidate is None


@pytest.mark.asyncio
async def test_same_value_reports_no_news() -> None:
    entries = [_entry("Pink Floyd (band)", "Pink Floyd, the prog band")]
    existing = [_given("pink floyd", "Pink Floyd, the prog band")]
    decisions = await _decide(_StubDecisionClient("yes", "same_value"), entries, existing)
    assert decisions[0].action == "unchanged"
    assert decisions[0].matched_key == "pink floyd"


@pytest.mark.asyncio
async def test_different_writes_through_quietly() -> None:
    entries = [_entry("the shining girls", "the 2022 series")]
    existing = [_given("the shining", "the 1980 film")]
    decisions = await _decide(_StubDecisionClient("no"), entries, existing)
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
    decisions = await _decide(_StubDecisionClient("unsure", "yes", "new_value"), entries, existing)
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
    stub = _StubDecisionClient(*(["no"] * SHORTLIST_K))
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
    (the store replaces the value).  Identical content settles NOTHING by rule
    any more, so the look-alike goes to the honest question."""
    entries = [_entry("Genesis (band)", "a different description"), _entry("gybe", "same text")]
    existing = [
        _given("genesis", "the prog band"),
        _given("godspeed you black emperor", "same text"),
    ]
    plans = plan_entries(entries, existing)
    decisions = rule_only_decisions(plans, entries, existing)
    assert [d.action for d in decisions] == ["merge", "new"]
    assert decisions[0].matched_key == "genesis"
    # Ties keep store order: both stored entries share the content, so the
    # older one (genesis) heads the shortlist.
    assert decisions[1].unsure_candidate == "genesis"


@pytest.mark.asyncio
async def test_the_parenthetical_year_rule_never_reaches_the_decision_model() -> None:
    """Two films keyed by year — the pair the parenthetical split exists for:
    the number rule reads BOTH parentheticals when both carry numbers, so
    ``Solaris (1972)`` and ``(2002)`` are settled DIFFERENT by the rules
    and never merged, whatever a decision model would say."""
    entries = [_entry("Solaris (2002)", "the 2002 remake")]
    existing = [_given("Solaris (1972)", "the 1972 film")]
    stub = _StubDecisionClient("yes", "same_value")  # would wrongly merge if consulted
    decisions = await _decide(stub, entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate is None
    assert stub.states == []


@pytest.mark.asyncio
async def test_identical_content_is_a_question_the_model_sees_both_keys_for() -> None:
    """Content settles nothing by rule: two products at the same price are two
    entries.  The pair goes to the decision model, which sees both keys; its
    ``different`` verdict writes the second product through quietly."""
    entries = [_entry("switch lite", "$219")]
    existing = [_given("analogue pocket", "$219")]
    stub = _StubDecisionClient("no")
    decisions = await _decide(stub, entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate is None
    assert "analogue pocket" in stub.states[0]
    assert "switch lite" in stub.states[0]


@pytest.mark.asyncio
async def test_the_date_rule_never_reaches_the_decision_model() -> None:
    """The production regression: a daily-digest job writing tomorrow's entry is
    settled DIFFERENT by the rules — no candidate is shortlisted, so the
    decision model is never asked and nothing is ever merged across dates."""
    entries = [_entry("2026-10-21", "the October 21 digest")]
    existing = [_given("2026-10-20", "the October 20 digest")]
    stub = _StubDecisionClient("yes", "same_value")  # would wrongly merge if consulted
    decisions = await _decide(stub, entries, existing)
    assert decisions[0].action == "new"
    assert decisions[0].unsure_candidate is None
    assert stub.states == []


# ── The collection-name ladder (#2724) ──────────────────────────────────────


def test_collection_name_rules_settle_the_names_they_are_sure_of() -> None:
    """The rules settle what they are sure of and leave the rest: equal after
    normalising (case, separators, punctuation, a leading ``the``, singular/
    plural) is the SAME collection; a differing number token (main part or two
    numeric parentheticals) is DIFFERENT; a typo, a rewording, or a role-word
    extension is undecided."""
    assert collection_name_rule("gear-notes", "gear-note") is KeyRule.SAME
    assert collection_name_rule("Gear_Notes", "gear notes") is KeyRule.SAME
    assert collection_name_rule("the gear notes", "gear-notes") is KeyRule.SAME
    assert collection_name_rule("aurora deck 2", "aurora-deck-2") is KeyRule.SAME
    assert collection_name_rule("aurora-deck-2", "aurora-deck-3") is KeyRule.DIFFERENT
    assert collection_name_rule("Dune (1984)", "Dune (2021)") is KeyRule.DIFFERENT
    assert collection_name_rule("gear-ntoes", "gear-notes") is None
    assert collection_name_rule("trip-plans", "trip-planning") is None
    assert collection_name_rule("team-news", "team-news-alerts") is None


def _candidate(name: str, description: str = "a collection") -> CollectionCandidate:
    return CollectionCandidate(
        name=name, description=description, name_vec=[1.0, 0.0], description_vec=[1.0, 0.0]
    )


async def _decide_name(stub: DecisionClient, asked: str, *candidates: CollectionCandidate):
    return await CollectionNameDecider(stub).decide(asked=asked, candidates=list(candidates))


@pytest.mark.asyncio
async def test_collection_name_yes_redirects_to_the_candidate() -> None:
    """A ``yes`` on the first shortlisted candidate redirects to it."""
    verdict = await _decide_name(_StubDecisionClient("yes"), "gear-ntoes", _candidate("gear-notes"))
    assert verdict.name == "gear-notes"
    assert verdict.unsure is False


@pytest.mark.asyncio
async def test_collection_name_no_exhausts_to_nothing_close() -> None:
    """A decisive NO on every candidate is ``none`` — nothing close; the caller
    auto-creates."""
    verdict = await _decide_name(
        _StubDecisionClient("no", "no"),
        "trail-recipes",
        _candidate("gear-notes"),
        _candidate("likes"),
    )
    assert verdict.name is None
    assert verdict.unsure is False


@pytest.mark.asyncio
async def test_collection_name_unsure_falls_through_to_the_refusal() -> None:
    """An unsure answer (or a failed call) marks the ladder unsure — the caller
    keeps the did-you-mean refusal — but a later decisive YES still wins."""
    verdict = await _decide_name(
        _StubDecisionClient("unsure", "yes"),
        "aurora-dek",
        _candidate("aurora-deck-2"),
        _candidate("likes"),
    )
    assert verdict.name == "likes"
    assert verdict.unsure is False
    unsure_only = await _decide_name(
        _StubDecisionClient("unsure"), "aurora-dek", _candidate("aurora-deck-2")
    )
    assert unsure_only.name is None
    assert unsure_only.unsure is True
    failed = await _decide_name(_FailingDecisionClient(), "aurora-dek", _candidate("aurora-deck-2"))
    assert failed.name is None and failed.unsure is True


@pytest.mark.asyncio
async def test_collection_name_degraded_mode_is_unsure_with_candidates() -> None:
    """No decision client is the named degraded mode: every candidate is
    unsure — the refusal — while no candidates at all is ``none`` (nothing
    close, the caller auto-creates)."""
    with_candidates = await CollectionNameDecider(None).decide(
        asked="gear-ntoes", candidates=[_candidate("gear-notes")]
    )
    assert with_candidates.name is None and with_candidates.unsure is True
    no_candidates = await CollectionNameDecider(None).decide(asked="trail-recipes", candidates=[])
    assert no_candidates.name is None and no_candidates.unsure is False
