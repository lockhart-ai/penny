"""The identity cases' coherence probe — run by ``make check``, before any GPU time.

A case whose pair does not cohere with what it claims measures nothing: an anchor that is
not on the entry it names is a claim that fails or passes for the wrong reason, and a pair
the deterministic rules already settle never reaches the decision model at all — the
driven write would be decided by the rules and the case would measure nothing live. This
probe holds every declared arm against its own entries, the way the extractor cases' probe
holds each anchor against its page.
"""

from __future__ import annotations

import pytest
from similarity.identity import KeyRule, pair_rule_verdict

from penny.tests.eval.drivers.identity import IdentityArm, _synth_entry
from penny.tests.eval.identity.test_entry_identity import _APART_ARMS, _MERGE_ARMS


def _carries(arm: IdentityArm, pair: tuple[str, str], anchor: str) -> bool:
    """Whether the entry ``pair`` declares (as the store will hold it) carries ``anchor``."""
    return anchor.lower() in _synth_entry(pair).lower()


def _all_arms() -> tuple[IdentityArm, ...]:
    return _MERGE_ARMS + _APART_ARMS


def test_each_identity_pair_reaches_the_decision_model() -> None:
    """Every pair is UNDECIDED by the deterministic rules — the premise the cases stand on.

    A pair the rules settle as the SAME entry would merge without the decision model ever
    being asked; one they settle as DIFFERENT would be excluded from the shortlist the
    same way. Either way the driven write would measure the rules, which the unit suite
    already pins, and a green cohort would say nothing about the decision step. The filler
    is held to the weaker bar: the rules may not settle it as the SAME entry as the new
    one, or the driven write would merge into the filler deterministically."""
    for arm in _all_arms():
        assert pair_rule_verdict(arm.new[0], arm.stored[0], arm.new[1], arm.stored[1]) is None, (
            f"{arm.text}: the rules settle the pair; it never reaches the decision model"
        )
        assert pair_rule_verdict(arm.new[0], arm.filler[0], arm.new[1], arm.filler[1]) is not (
            KeyRule.SAME
        ), f"{arm.text}: the rules settle the new entry as the filler's"


@pytest.mark.parametrize("arm", _all_arms(), ids=lambda arm: arm.text)
def test_each_anchor_is_on_its_own_entry_and_nowhere_else(arm: IdentityArm) -> None:
    """Every anchor is the smallest unique datum of the entry it names: carried by that
    entry, and by no other entry of the arm — so finding it in the store is evidence it
    came from there and nowhere else."""
    entries = (arm.stored, arm.filler, arm.new)
    for name, pair, anchor in (
        ("stored", arm.stored, arm.stored_anchor),
        ("filler", arm.filler, arm.filler_anchor),
        ("new", arm.new, arm.new_anchor),
        ("survivor", arm.new if arm.survivor == "new" else arm.stored, arm.survivor_anchor),
    ):
        assert _carries(arm, pair, anchor), f"{arm.text}: the {name} anchor is not on its entry"
        for other in entries:
            if other == pair:
                continue
            assert not _carries(arm, other, anchor), (
                f"{arm.text}: the {name} anchor also appears on another entry of the arm, "
                "so finding it would not identify the entry it came from"
            )


@pytest.mark.parametrize("arm", _all_arms(), ids=lambda arm: arm.text)
def test_each_arm_declares_two_distinct_entries_and_a_filler(arm: IdentityArm) -> None:
    """The stored entry and the filler are different things under different keys, and the
    new entry carries a key of its own — a seed that laid down two names for one entry
    would put the ladder's decision about the PAIR in the seed instead of in the run."""
    assert arm.stored[0].lower() != arm.filler[0].lower()
    assert arm.stored[1] != arm.filler[1]
    assert arm.new[0].lower() != arm.stored[0].lower()
    assert arm.collection.strip() and arm.description.strip()
