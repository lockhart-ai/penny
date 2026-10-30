"""Changelog reference: a later turn answers from what the delivery turn told her (#2846).

**The world.** Three REAL rows from the shipped ``changelog.yaml`` — this project's own
true changelog, never invented tool changes — delivered through the real startup
mechanism: ``ChatAgent.handle_changelog_report`` over the framework-rendered entries,
the same content gates ``Penny._send_changelog_announcement`` applies, then the real
send-queue drainer. One of the three, PR #2820, carries the one fact this case
measures: the removed ``reasoning`` tool-call argument. That detail is not visible
anywhere in her ordinary ambient state — it is a retired implementation detail she can
only know from having been told.

One unrelated ordinary turn follows the announcement, so the question is never the
very next message after it — then the question itself, which never says "reasoning"
or names the argument. It also never asks the open "did anything change" version: her
changelog holds several real tool changes (#2822, #2844 among them), so a question
that broad admits more than one true answer. The question instead points at the one
change this case measures without naming it: something was TAKEN OFF her tool calls —
what was it.

**What is claimed.** The reply to that LATER question mentions the reasoning argument
(a regex on ``reasoning``) — a planted, checkable fact, not a paraphrase word list. On
`main`, there is no delivery turn at all, so this case has no meaningful "before": the
claim would fail outright with nothing to answer from. Report-only on its first run.
"""

from __future__ import annotations

import pytest

from penny.changelog import load_entries
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.changelog_announcement import ChangelogEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "changelog-reference"

# Three real entries this deploy's changelog carries — PR #2820 is the one this case
# measures; the other two are real too, just along for the ride as "a deploy usually
# ships more than one change". Filtered off the SHIPPED file by PR number, never
# hand-copied, so this case always measures the text that actually ships.
_WANTED_PRS = (2820, 2831, 2835)
_REASON_PR = 2820
_ENTRIES = [entry for entry in load_entries() if entry.pr in _WANTED_PRS]
assert {entry.pr for entry in _ENTRIES} == set(_WANTED_PRS), (
    "changelog.yaml no longer carries one of this case's three real PRs — "
    "pick three others that are still shipped"
)
assert any(entry.pr == _REASON_PR for entry in _ENTRIES)

# The planted fact: PR #2820's own real text names the retired `reasoning` argument.
_REASON_WORD = "reasoning"
assert any(_REASON_WORD in entry.text for entry in _ENTRIES if entry.pr == _REASON_PR), (
    f"PR #{_REASON_PR}'s real text no longer mentions {_REASON_WORD!r}"
)

# Between the announcement and the question — ordinary, unrelated small talk, so the
# question is never the very first message after the delivery turn.
_UNRELATED_TURN = "quick one — what's a fun fact about octopuses?"

# The question never says "reasoning" and never names the argument, and never asks the
# open "did anything change" form — her changelog holds several real tool changes
# (#2822, #2844 among them), so that broad a question admits more than one true
# answer. It points at the one change this case measures without naming it: something
# was taken OFF her tool calls — what was it.
_ASK = "did that update take something off your tool calls? what was it?"
_ALSO_PHRASED = (
    "you got a deploy earlier — was something removed from your tool calls? what?",
    "that deploy — did it drop something from how your tool calls work? what was it?",
)
for _wording in (_ASK, *_ALSO_PHRASED, _UNRELATED_TURN):
    assert _REASON_WORD not in _wording.casefold(), (
        f"a wording must not name {_REASON_WORD!r}: {_wording!r}"
    )

_WORLD = World(name="changelog-reference", pages=(), keeps=(), excludes=())


def _reply_mentions_the_reasoning_argument(sample: SampleObservation, _world: World) -> Answer:
    """The reply to the later question mentions the retired ``reasoning`` argument —
    the one fact only the delivery turn's own real text supplied, never the question
    itself."""
    hit = _REASON_WORD in sample.reply.casefold()
    return hit, f"the reply mentions {_REASON_WORD!r}: {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_later_question_is_answered_from_the_changelog_she_was_told(
    changelog_eval: ChangelogEval, model: str
) -> None:
    """A later turn asking what was taken off her tool calls — without naming it — is
    answered from what the real changelog delivery turn actually told her — not
    guessed, since the fact it measures (the retired `reasoning` argument) appears
    nowhere else in this world."""
    cohort = await changelog_eval(
        case_id="changelog-reference",
        behaviour=(
            "In the chat agent, when a deploy's changelog entries have just been "
            "delivered, a later turn asking about one of those changes is answered "
            "from what she was told, not a guess."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        changelog_entries=_ENTRIES,
        unrelated_turn=_UNRELATED_TURN,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=5,
        min_pass_rate=None,  # report-only — main has no delivery turn to compare against
        family=_FAMILY,
        timeout=240.0,
    )
    cohort.assert_what_the_store_held_survives()
    cohort.claim(
        "reply: it mentions the retired reasoning argument",
        _reply_mentions_the_reasoning_argument,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
