"""elicit → learn: the demonstrated round, enacted and reported (#1994/#1995).

THE REFERENCE IMPLEMENTATION — every other case is ported against this shape.  Why a case
asserts end state and measures model output at all is `cohort.py`; what a claim means is
`assertions.py`.  What is here is what is true of THIS case.

Parked on the teach question, the user supplies the steps.  The turn follows them once —
browse, find, remember — mints the routine, and the reply that closes the round tells the user
what that routine will RUN, so a step it captured by accident is visible to the only person who
can tell that it does not belong.  It instantiates NOTHING: the collection the demonstrated
write created carries no skill, no program and no schedule.

ONE source and PROSE, both deliberate.  `test_two_source_teach.py` holds the two-source case;
this one demonstrates three actions against a single page, said as a sentence rather than as a
numbered procedure.

  * PHRASINGS — the same request in five wordings, POOLED into one variance score.  Wording is
    an INPUT axis: what varies is how a person says three things in a sentence, and what is
    scored is that the end state does not move with it.
  * the tool calls MEASURED rather than asserted, because many routes reach one end state.

REPORT-ONLY (``min_pass_rate=None``): the floors and ceilings this run proposes are the code
owner's to accept once the numbers have been read.  All content is synthetic — an invented
marketplace listing — because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    ROUTINE_NAME,
    ROUTINE_SHAPE,
    TOOL_SEQUENCE,
    TRANSITIONS,
)
from penny.tests.eval.utils.seeds import Seeder, round_parked_in_elicit

# The family every state-transition case reports under, read from where that suite declares it:
# this is the canonical case for the elicit → learn edge, and a second spelling of its family
# would split one edge's results across two headings.
from penny.tests.eval.utils.transition_ledger import _FAMILY
from penny.tests.eval.utils.worlds import (
    AURORA_LISTING,
    LISTING_DEMO,
    LISTING_DEMO_PHRASINGS,
    LISTING_SETUP_ASK,
    LISTING_TEACH_QUESTION,
)

pytestmark = pytest.mark.eval

_CASE_ID = "transition-elicit-to-learn"


@pytest.fixture
def standing_elicit_round() -> Seeder:
    """The round the measured turn closes: the user asked for the job, Penny asked to be taught
    one pass, and the machine is parked in ``elicit`` on that ask.

    Seeded rather than hoped for (#1989) — the ask is an imperative about now, which idle's own
    definition claims, so on a cold machine both measured models drew idle on 10 of 10 samples
    and every reply check failed for a reply nobody had been asked to write."""
    return round_parked_in_elicit(LISTING_SETUP_ASK, LISTING_TEACH_QUESTION)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_elicit_to_learn_runs_the_round_and_reports_what_it_captured(
    chat_eval: ChatEval, model: str, standing_elicit_round: Seeder
) -> None:
    """elicit → learn: one demonstrated round, run once across the page it names, and the reply
    that closes it states the steps the routine captured."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        model=model,
        seed=standing_elicit_round,
        world=AURORA_LISTING,
        ask=LISTING_DEMO,
        also_phrased=LISTING_DEMO_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.LEARN)

    # STORE
    cohort.assert_something_from_each_page_was_written()
    cohort.assert_the_write_landed_in_the_round_container()
    cohort.assert_a_routine_reached_the_registry()
    cohort.assert_nothing_was_scheduled()
    cohort.assert_every_spot_is_a_placeholder()
    cohort.assert_the_routine_names_a_destination()

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(
        TOOL_SEQUENCE, ROUTINE_SHAPE, ROUTINE_NAME, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD
    )
