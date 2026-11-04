"""Restart notice explained: a later turn is answered from the restart notice
itself, not a guess about an earlier, unrelated exchange (#2974).

**The world.** A deploy that changed nothing in Penny's tools or behaviour —
the real startup delivery turn (``ChatAgent.handle_changelog_report`` with no
new entries), through the same content gates production applies, then the
real send-queue drainer — lands immediately after two ordinary exchanges on a
topic that has nothing to do with deploys, restarts, tools, or behaviour (a
weekend trail-conditions chat). There is no turn between the restart notice
and the question: the complaint follows it directly, mirroring the real
failure this case is drawn from, where an autonomous send with nothing
incoming before it melted into the tail of whatever she'd just said.

**What is claimed.** The user's later question never names what she thinks
was missed; it just questions the "nothing changed" line. A reply that
actually read the restart notice can name what it was ABOUT — her tools or
behaviour — words that appear nowhere else in this world (not in the trail
chat, not in the question). A reply that is still guessing, as every one of
15 verbatim replays of the real failing turn did, recasts the "restart" as
being about something else entirely (her own settings, memory, or the trail
chat) and never uses either word. Report-only: this case's whole point is
measuring the rate, before and after the fix, not gating on it.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.changelog_announcement import ChangelogEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import SampleObservation, SpecCategory
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "restart-notice-explained"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY_AUTHOR = PennyConstants.MessageAuthor.PENNY

# Two ordinary exchanges, oldest first — a topic with no overlap with a
# deploy, a restart, a tool, or a behaviour change, so neither planted word
# below can be satisfied by anything other than a reply that actually read
# the restart notice.
_PRIOR_TURNS: tuple[tuple[str, str], ...] = (
    (
        "the ridgeline loop trail — is it worth doing this weekend?",
        "Should be a great pick! The forecast's clear and it's a steady climb with "
        "a payoff view at the top — bring more water than you think you'll need "
        "though, the exposed stretch near the summit gets hot fast.",
    ),
    (
        "good call, we'll pack extra. any river crossings on that one?",
        "Just the one, about two-thirds in — usually ankle-deep this time of year, "
        "but it's worth checking a recent trip report the morning you go since a "
        "storm upstream can raise it overnight.",
    ),
)

# The planted facts: the real restart notice's own words, naming what the
# deploy did NOT change — appears nowhere else in this world.
_PLANTED_WORDS = ("tools", "behaviour", "behavior")

_ASK = "Everything looks exactly the same? You didn't get notified about all the latest changes?"
_ALSO_PHRASED = (
    "wait, you're saying nothing happened? you didn't hear about any of the recent changes?",
    "hold on — nothing changed at all? what about everything that's been going on?",
    "really, not one update? you'd think something would've come through by now",
    "so you're telling me literally nothing is different? seems off",
)
for _wording in (_ASK, *_ALSO_PHRASED):
    assert not any(word in _wording.casefold() for word in _PLANTED_WORDS), (
        f"a wording must not pre-empt the planted words: {_wording!r}"
    )
for _ask_text, _reply_text in _PRIOR_TURNS:
    assert not any(word in (_ask_text + _reply_text).casefold() for word in _PLANTED_WORDS), (
        "the seeded trail chat must not pre-empt the planted words"
    )

_WORLD = World(name="restart-notice-explained", pages=(), keeps=(), excludes=())


def _seed_prior_exchange(db: Database) -> None:
    """Two ordinary exchanges, logged exactly as the channel would, so the
    real startup delivery turn that follows has a real prior reply to land
    after — the structural precondition the failure needs (#2974): an
    autonomous send with no incoming message between it and the turn before
    it is the one case history used to melt two voices into one with no
    seam."""
    for ask, reply in _PRIOR_TURNS:
        db.messages.log_message(_INCOMING, TEST_SENDER, ask)
        db.messages.log_message(_OUTGOING, _PENNY_AUTHOR, reply, recipient=TEST_SENDER)


def _reply_names_what_the_deploy_touched(sample: SampleObservation, _world: World) -> Answer:
    """The reply names what the restart notice actually said nothing changed
    in — her tools or behaviour — the one fact that is only ever visible by
    having read that notice's own text, never by guessing from the trail
    chat or from the question itself."""
    hit = any(word in sample.reply.casefold() for word in _PLANTED_WORDS)
    return hit, f"the reply names tools/behaviour (the restart notice's own words): {hit}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_later_question_is_answered_from_the_restart_notice_she_was_told(
    changelog_eval: ChangelogEval, model: str
) -> None:
    """A later question doubting a "nothing changed" restart line is answered
    from the restart notice's own text — not a guess built out of whatever
    she was just talking about, which is what the real failing turn did on
    every one of 15 verbatim replays of it."""
    cohort = await changelog_eval(
        case_id="restart-notice-explained",
        behaviour=(
            "In the chat agent, when a restart notice saying nothing changed "
            "immediately follows her own prior reply with no turn between them, "
            "a later question doubting it is answered from the notice's own "
            "text, not a guess about the conversation it followed."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        changelog_entries=(),
        unrelated_turn="",
        seed=_seed_prior_exchange,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — the before/after rate is this case's whole point
        family=_FAMILY,
        timeout=240.0,
    )
    cohort.assert_what_the_store_held_survives()
    cohort.claim(
        "reply: it names what the restart notice said didn't change",
        _reply_names_what_the_deploy_touched,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
