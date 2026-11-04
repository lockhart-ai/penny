"""Restart notice explained: a later turn is answered from the restart notice
itself, not a guess about an earlier, unrelated exchange (#2974).

**The world.** A deploy that changed nothing in Penny's tools or behaviour —
the real startup delivery turn (``ChatAgent.handle_changelog_report`` with no
new entries), through the same content gates production applies, then the
real send-queue drainer — lands immediately after two ordinary exchanges on a
topic that has nothing to do with deploys, restarts, tools, or behaviour (a
garage sort and a board-game night: a CLOSED, opinion-only exchange with no
factual question in it, so the later ask can never be read as that
exchange's own open thread). There is no turn between the restart notice and
the question: the complaint follows it directly, mirroring the real failure
this case is drawn from, where an autonomous send with nothing incoming
before it melted into the tail of whatever she'd just said.

**What is claimed.** Provenance and store survival only: every specific value
the reply carries traces to something the turn was given, and what the store
held before the turn survives it. Nothing about the reply's WORDING is
claimed — whether a reply actually connects the user's question to the
restart notice is a reading of her words, and §2/§9 keep that out of every
check (a phrase list would fail a differently-worded correct reply and pass a
lucky echo).

**What is measured, never claimed.** The rate at which the reply's own words
touch the notice's fact — the restart itself, or that nothing changed —
reported as a distribution with every reply quoted, which is the number the
before/after comparison reads and the material a reader of ONE sample needs;
the connection itself stays the reader's call.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.changelog_announcement import ChangelogEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import SampleObservation, Share, ShareReading
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "restart-notice-explained"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY_AUTHOR = PennyConstants.MessageAuthor.PENNY

# Two ordinary exchanges, oldest first — a topic with no overlap with a
# deploy, a restart, a tool, or a behaviour change.  The exchange is CLOSED
# and opinion-only: a settled plan and two takes on it, no factual question
# anywhere, so the later ask below can never be read as that exchange's own
# open thread.
_PRIOR_TURNS: tuple[tuple[str, str], ...] = (
    (
        "spent the morning clearing out the garage — it's actually sortable now",
        "That's the hardest part done! A sortable garage wants every pile to get a "
        "shelf of its own before the weekend undoes it — label them while you're "
        "still feeling heroic.",
    ),
    (
        "true. and once it's liveable in there we should do a board game night",
        "A garage game night is a great call — the table's already in there, and "
        "there's no better test of a cleanup than a night of people actually "
        "living in it.",
    ),
)
for _ask_text, _reply_text in _PRIOR_TURNS:
    assert "?" not in _ask_text + _reply_text, (
        "the seeded exchange must be closed and opinion-only — no factual question"
    )

_ASK = "Everything looks exactly the same? You didn't get notified about all the latest changes?"
_ALSO_PHRASED = (
    "wait, you're saying nothing happened? you didn't hear about any of the recent changes?",
    "hold on — nothing changed at all? what about everything that's been going on?",
    "really, not one update? you'd think something would've come through by now",
    "so you're telling me literally nothing is different? seems off",
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


# The notice's own two facts, as her reply's words could touch them — the
# restart itself, or that nothing changed.  A SURFACE reading for a measured
# share, never an assertion: a differently-worded reply that connects just as
# well but misses every one of these strings reads as a miss here, and the
# quoted evidence is what the reader actually judges.
_NOTICE_WORDS = (
    "restart",
    "restarted",
    "reboot",
    "rebooted",
    "nothing changed",
    "no changes",
    "nothing new",
    "no updates",
    "unchanged",
)


def _reply_touches_the_notice(sample: SampleObservation) -> ShareReading:
    """Whether the reply's own words touch the restart notice's fact — the
    restart itself, or that nothing changed.  Measured, never claimed (§9):
    whether the reply actually CONNECTS the user's question to the notice is
    the reader's call on the quoted evidence, never a check."""
    hit = any(word in sample.reply.casefold() for word in _NOTICE_WORDS)
    evidence = [sample.reply] if sample.reply.strip() else []
    return ShareReading(value=1.0 if hit else 0.0, evidence=evidence)


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
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.measure(
        Share(
            "reply touches the restart notice",
            _reply_touches_the_notice,
            open_above=0.0,
            says=(
                "of all samples, the share where the reply's own words touch the "
                "notice's fact (the restart, or that nothing changed) — a surface "
                "reading with every reply quoted; the connection itself is the "
                "reader's call, never a check"
            ),
        )
    )
