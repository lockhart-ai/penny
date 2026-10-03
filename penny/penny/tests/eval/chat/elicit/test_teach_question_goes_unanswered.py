"""elicit → elicit: the teach question is answered with a question, and the round stays parked
(#2215).

The contract is `docs/eval-case-design.md`.

She has asked to be walked through a job once, and the user does not walk her through it: they
ask what she would need from them.  Nothing has been taught, nothing has been called off, and
the round is still the round it was — so the turn ends where it began, parked in elicit on the
ask that opened it, and the next message is still read as an answer to her question.

**The world is the reference journey, one message further on.**  `LISTING_SETUP_ASK` and
`LISTING_TEACH_QUESTION` are what `transition-idle-to-elicit` produces and what
`transition-elicit-to-learn` seeds, so this case sits between them: the same ask, the same
teach question, and the message that arrives is neither the steps (elicit → learn) nor a
call-off (elicit → idle).  The seeder is that reference case's own
(`round_parked_in_elicit`), so the three read one round at three moments.

**The wordings are the classifier case's own five.**
`classifier-stays-parked-when-the-teach-question-is-unanswered` draws this decision in
isolation from these same messages, and this case drives the whole turn they open.  Read from
where that case declares them rather than restated, so a miss here is attributable: where the
draw holds there, a turn that lands elsewhere here is the turn's.  Every wording asks the SAME
question back — what she would need from them — and none carries a step, an address or a
call-off; the arms are held to that in `make check`.

**This is a NO-FIRE edge, and it asserts where the machine landed, never what the turn
refrained from.**  Opening the listing, saving its price or standing a watch up is the model's
call on what it was shown, so none of them is a claim: each is measured, as the tool sequence
and the entries stored.

**The claim set is thin, and every absence is a closed question rather than an unrun one:**

* *what the store already held survives* — VACUOUS on this world.  A round parks in elicit
  because nothing covers the ask, and this one opened on a cold machine: no collection, no
  entry, no job.  The claim would print a green row for a question its world cannot ask.
* *the round is still anchored to the ask* — PRODUCTION ALREADY VALIDATES IT.  `_next_anchor`
  keeps the anchor of a machine that is already parked, and the from-state is what this world
  seeds, so the claim is entailed by the landing.
* *nothing was registered* — ENTAILED by the landing.  Run-end extraction fires in `learn` and
  nowhere else, and the only other thing that touches the registry (`abandon_round_skill`) runs
  on an IDLE landing, so no sample can fail it without also failing `assert_machine_landed`.
* *the page went unread* / *nothing was written* — a ROUTE and a restraint.  Both are measured
  in section B.

**PROVENANCE is claimed in both halves.**  The store half reads only what this turn wrote, so it
asks nothing of whether the turn wrote and everything of whether what it wrote was invented.
The reply half is live throughout: a reply that says what she needs by quoting a price nobody's
page was read for made it up.

**`keeps`, `excludes` and `answers` are all EMPTY, and each is a report.**  The turn is not
asked to write anything down; the message rules nothing out; and what it asks for is not a
value the world carries — what she needs in order to learn a job is hers to say, in her own
words, so a required token would be a phrasing match.

REPORT-ONLY (``min_pass_rate=None``).  Every page and url is synthetic, on an ``example``
domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.tests.eval.classifier.test_state_classifier import STILL_CLARIFYING_ARMS
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
)

# The listing the parked round is about, read from the suite's shared fixtures: two copies of
# the page a journey is measured against are two contracts free to drift.
from penny.tests.eval.utils.fixtures import AURORA_LISTING_499
from penny.tests.eval.utils.seeds import Seeder, round_parked_in_elicit
from penny.tests.eval.utils.transition_ledger import _FAMILY

# The round's two turns, read from where the reference case declares them — this case is that
# journey one message on, and a second spelling would let the two drift into two rounds.
from penny.tests.eval.utils.worlds import LISTING_SETUP_ASK, LISTING_TEACH_QUESTION, World

pytestmark = pytest.mark.eval

QUESTION_BACK_CASE_ID = "transition-elicit-to-elicit"

_BEHAVIOUR = (
    "In the chat agent, when she has asked to be taught a job and the user answers with a "
    "question back instead of the instructions, Penny leaves the round parked in elicit."
)

# One message in five wordings: the question back, with the first as the ask and the rest as
# its other phrasings.  Named at module level so the deterministic pin in ``make check`` holds
# the cohort's arithmetic against the same tuple the case drives.
QUESTION_BACK_ARMS = STILL_CLARIFYING_ARMS

# The parked round the question arrives into, laid down rather than hoped for: the ask, her
# teach question threaded to it, that turn's ledger, and the machine parked in elicit on the
# ask.  The seeder asserts every one of those as it writes them.
PARKED_ON_THE_TEACH_QUESTION: Seeder = round_parked_in_elicit(
    LISTING_SETUP_ASK, LISTING_TEACH_QUESTION
)

# The ground the question is answered against: the page the round's ask names, installed and
# reachable, so a turn that DOES go and look finds something rather than failing on a thin
# fixture.  ``keeps``, ``excludes`` and ``answers`` are empty; the module docstring says why.
_PARKED_LISTING = World(
    name=QUESTION_BACK_CASE_ID,
    pages=(AURORA_LISTING_499,),
    keeps=(),
    excludes=(),
)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: the
# registry is empty by construction on this world and no elicit landing can add to it, so both
# would read their absent value on every sample and render blind.
#
# ``TOOL_SEQUENCE`` reads "no call" on a sample that only answered, so a cohort that all did so
# makes it blind and the report says so in red.  Measured anyway, because the divergence it
# exists to catch is this edge's own — a sample that went and opened the listing instead of
# answering reads differently from every other one, and the blindness lifts the moment it does.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_elicit_to_elicit_stays_parked_on_a_question_back(
    chat_eval: ChatEval, model: str
) -> None:
    """elicit → elicit: parked having asked to be walked through the listing watch, the user
    asks what she would need from them.  The teach question is still unanswered, so the round
    is still parked on it when the turn ends."""
    cohort = await chat_eval(
        case_id=QUESTION_BACK_CASE_ID,
        behaviour=_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.ELICIT, ConversationState.ELICIT),
        model=model,
        seed=PARKED_ON_THE_TEACH_QUESTION,
        world=_PARKED_LISTING,
        ask=QUESTION_BACK_ARMS[0],
        also_phrased=QUESTION_BACK_ARMS[1:],
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.ELICIT)

    # STORE — no claim.  The round opened on a cold machine, so there is nothing already held
    # for a survival claim to be about; doing the job instead of answering is measured below.

    # PROVENANCE — both halves.  Whether the turn writes anything is the model's call and is
    # measured; what is claimed is that nothing it did write, and nothing it said, was invented.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
