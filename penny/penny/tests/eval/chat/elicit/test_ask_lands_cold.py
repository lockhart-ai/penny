"""idle → elicit: the ask lands cold, and nothing is enacted (#2005, tranche 2).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

A cold machine, an empty registry, and an ask for something that has to KEEP RUNNING.  No
routine covers it, so the turn's whole job is the question — and the failure the case exists
to catch is doing the job instead of asking to be shown it.

**The survivor, and on what basis.**  The edge had five scenario variants and every one of
them scores at the top: measured over the last three suite runs that carried them, the base
ask and `-no-url`, `-digest`, `-threshold` and `-urgency` all report a mean of 1.00 with every
sample passing (a single 0.97 on two of them).  A measured rate cannot choose between them, so
the choice is the PLAINEST STATEMENT of the behaviour, and then DOMINANCE.  Plainest: "watch
this listing daily and tell me when the price changes" is the ask with nothing else in it — no
digest to assemble, no threshold to compare against, no urgency to resist.  Dominance: it is
the variant whose world can produce the failure the behaviour is NAMED for.  The page is named
in the ask AND installed and reachable, so opening it is one call away; `-no-url` names a
source with no page, and a turn there cannot fail by "opening the page" at all.

The four dropped variants are quarantined rather than deleted, each recorded with the
temptation it probes: `-no-url` (a findable source with no address — the search temptation),
`-digest` (an ask naming what to collect and where to keep it, but never the steps),
`-threshold` (a url in hand plus a number to compare against last time — the baseline-write
temptation), `-urgency` (act-now pressure as a reason to guess).  Their asks and pages live in
`transition_world.py`, so any of them can come back deliberately.

**The ask is the reference case's own.**  `LISTING_SETUP_ASK` is what
`transition-elicit-to-learn` SEEDS as the round it continues, so this case is the turn that
produces the state that case starts from, and the two are one journey read at two moments.
One spelling of the ask, read from where the world declares it.

**This is a NO-FIRE edge, and its two negative claims are written so a violating sample is
nameable:** one that opens the listing and saves the price fails *nothing was written*, and one
that stands the watch up fails *no mechanism was created*.

**Three obvious-looking claims are deliberately absent**, so a thin set reads as closed rather
than as a checklist nobody ran:

* *the page went unread* — a ROUTE (`assert_each_page_was_read`'s converse), which the design
  puts in section B.  A cohort that went and looked shows there, as a tool sequence that moved.
* *nothing was registered* — ENTAILED by the landing.  Run-end extraction fires in `learn` and
  nowhere else, and the only other thing that touches the registry (`abandon_round_skill`) runs
  on an IDLE landing, so no sample can fail it without also failing `assert_machine_landed`.
* *no running mechanism was changed* — VACUOUS on this world, which is a cold machine with no
  mechanism in it at all.  Its three siblings in this tranche make it; this one would print a
  green row for a question its own world cannot ask.

**Four source checks did not port** (the outward column):

* ``Check("state: no page was fetched (browse-results stayed empty)")`` — a ROUTE.  Its
  end-state form is *nothing was written*, which is claimed; the browse itself is measured.
* ``Check("calls: no enacting calls")`` — the same route, read off a tool-name set.
* ``_anchor_check`` — *the ask is stamped as the round's anchor.*  PRODUCTION ALREADY
  VALIDATES IT: ``_next_anchor`` sets the anchor to the instigating message on every move into
  a parked state FROM idle, so on a sample that landed in elicit the claim is entailed by the
  landing and would run at exactly the rate ``assert_machine_landed`` does.
* ``Check("reply: asked for no page structure")`` — a PHRASING match on a vocabulary somebody
  guessed in advance.  What it reached for has no end-state form and is read at review.

**And three the inward column added.**  PROVENANCE, of both kinds: the source case made no
claim of either, so a sample that answered out of its own head — or filed an invented fact —
passed every check it carried.  And `assert_every_delivered_message_is_whole`, which the round-
ends family had to refuse because every one of its worlds SEEDS Penny's own turns and the
claim would then be answered against the fixture's agreed prose.  This world seeds NOTHING —
the machine is cold — so the only message it can read is the question this turn asked, and
"the teach question is a message Penny would send" is a claim about the turn.

**`answers` is EMPTY, and that is a report.**  The ask asks for a job to be set up, not for a
value to be stated, so a correct reply owes no token; requiring one would fail a correct run
for something nobody requested.

REPORT-ONLY (``min_pass_rate=None``).  Every page and url is synthetic, on an ``example``
domain, because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
)

# The listing this ask points at, read from the suite's shared fixtures: two copies of the
# page a journey is measured against are two contracts free to drift.
from penny.tests.eval.utils.fixtures import AURORA_LISTING_499, LISTING_URL
from penny.tests.eval.utils.transition_ledger import _FAMILY

# The ask itself, read from where the reference case declares it — the two cases are one
# journey at two moments, and a second spelling would let them drift into two asks.
from penny.tests.eval.utils.worlds import LISTING_SETUP_ASK, World

pytestmark = pytest.mark.eval

_CASE_ID = "transition-idle-to-elicit"

_BEHAVIOUR = (
    "In the chat agent, when the user asks for something that has to keep running and no "
    "routine she has covers it, Penny asks to be taught the steps once — without opening the "
    "page, writing anything down, or standing a job up on steps nobody has given her."
)

# Four more wordings of that same ask.  What varies is only how a person says it — which verb
# opens it, "daily" or "every day" or "once a day", "changes" or "moves" — while the page, the
# cadence, the thing to watch for and the state the turn must end in are all constant, which is
# what makes the fifteen samples one number.
_SETUP_ASK_PHRASINGS = (
    f"could you keep an eye on this listing every day and tell me when the price changes? "
    f"{LISTING_URL}",
    f"check this listing once a day and let me know if the price moves — {LISTING_URL}",
    f"i'd like this listing watched daily, and a message when the price changes: {LISTING_URL}",
    f"every day, look at {LISTING_URL} and tell me when its price changes",
)

# The ground the ask is answered against: the page it names, installed and reachable, so a turn
# that DOES go and look finds something rather than failing on a thin fixture — which is what
# makes "she asked instead" a real reading rather than a browse that could not have worked.
#
# ``keeps`` is EMPTY and that is a report, not an omission: an elicitation turn is not asked to
# write anything down, so a keeps set here would state a contract the ask never made — and the
# case claims the opposite, that nothing was written at all.  ``excludes`` is empty because the
# ask rules nothing out, and ``answers`` because it requests no value (see the module docstring).
_COLD_LISTING = World(
    name=_CASE_ID,
    pages=(AURORA_LISTING_499,),
    keeps=(),
    excludes=(),
)

# What this case measures.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are deliberately ABSENT: the
# registry is empty by construction on a correct cohort, so both would read their absent value
# on every sample and render blind — and a sample that DID mint a routine is caught by the
# registry claim, where it is a miss rather than a variance rise.
#
# ``TOOL_SEQUENCE`` reads "no call" on a correct sample here too, so a perfectly behaved cohort
# makes it blind and the report says so in red.  Measured anyway, because the divergence it
# exists to catch is exactly this case's named failure — a sample that went and opened the
# listing reads differently from every other one, and the blindness lifts the moment it does.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_idle_to_elicit_asks_to_be_taught(chat_eval: ChatEval, model: str) -> None:
    """idle → elicit: the canonical watch ask, its page named and reachable, arriving on a
    cold machine.  No routine covers it, so the turn IS the question — the listing is never
    opened, nothing is stored, nothing is registered, and the machine parks on the ask."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        world=_COLD_LISTING,
        ask=LISTING_SETUP_ASK,
        also_phrased=_SETUP_ASK_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.ELICIT)

    # STORE — two negatives, one per way of doing the job instead of asking about it, and one
    # positive about the question itself, which this world is the only one in the tranche that
    # can make honestly.
    cohort.assert_nothing_was_written()
    cohort.assert_no_mechanism_was_created()
    cohort.assert_every_delivered_message_is_whole()

    # PROVENANCE — the half the source case had none of.  Nothing is meant to be stored, so
    # the store claim answers over an empty set on a correct sample and names the invention on
    # a sample that wrote one; the reply claim is live throughout, since a teach question that
    # quotes the listing's price read a page it was not asked to read.
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
