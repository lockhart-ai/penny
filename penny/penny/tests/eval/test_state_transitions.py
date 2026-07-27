"""Per-transition ENACTMENT contracts (#1706) — one edge of the conversation
state machine per case, all on the same auction fixture.

``test_state_classifier.py`` proves each edge is CHOSEN correctly from a scoped
micro-context.  This proves the other half: handed the state that edge lands in,
chat does that state's job and nothing else.  Both halves run for real here —
the machine is wired into the turn, so a case seeds the state the edge starts
from, sends one message, and the production path classifies it and swaps the
instruction before the turn runs.

The auction script, one turn per case (the simplest complete journey; richer
shapes are the later beats' business):

    idle → elicit   "watch this auction for me"                     → asks to be taught
    elicit → learn  "go to the site, find the price, remember it"   → runs it once, remembers
    learn → apply   "now do that hourly until 10pm and tell me"     → a live watch

**Learning attaches nothing** (#1706, replacing #1687's run-end auto-attach): the
machine makes teaching and instantiating two clear turns, so the demonstrated
round leaves a naive collection_write behind — a collection with a value in it
and no skill, no rendered program, nothing scheduled — and the NEXT turn adopts
the skill onto it.  Scoring that separation is most of the point of these cases.
"""

from __future__ import annotations

import pytest

from penny.constants import TransitionCause
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    asked_for_page_structure,
    collection_entries,
    new_collections,
    outgoing_replies,
    routing_clean,
    tool_was_called,
)
from penny.tests.eval.test_watch_journey import AURORA_LISTING_499, LISTING_URL

pytestmark = pytest.mark.eval

_FAMILY = "state-transitions"

# The instigating ask (turn 1 of the script) — the message the machine is parked
# on when a later edge is under test.
_AUCTION_ASK = f"hey can you watch the auction at {LISTING_URL} for me?"

# elicit → learn: the user answers the teach question with the steps.
_TEACH_TURN = f"yeah go to {LISTING_URL}, find the price, and remember it"


def _park(penny, state: ConversationState) -> None:
    """Leave the machine where the edge under test starts from, through the real
    store — a seeded transition row IS the machine's state (#1706), so nothing
    here fakes a state the production path couldn't be in.  The incoming message
    is still classified against it, so a case exercises the edge end to end."""
    penny.db.machine.record_transition(
        from_state=ConversationState.IDLE.value,
        to_state=state.value,
        cause=TransitionCause.CLASSIFIER,
    )


def _landed_state(db: Database) -> str | None:
    latest = db.machine.latest_transition()
    return latest.to_state if latest else None


def _score_elicit_to_learn(db: Database, before: set[str], reply: str) -> list[Check]:
    """The demonstrated round ran, and NOTHING was instantiated.

    "Remember it" is a naive ``collection_write``: it auto-creates a collection
    and puts the value in it.  What must NOT happen is the fold — no skill bound
    to that collection, no rendered program, no schedule.  The skill is learned
    (it exists in the registry) and stays unattached until the user asks for it."""
    rows = new_collections(db, before)
    stored = [content for row in rows for content in collection_entries(db, row.name).values()]
    skills = db.skills.list_all()
    return [
        Check(
            "state: she browsed the listing (the demonstrated fetch happened)",
            tool_was_called(db, "browse"),
            kind="state",
        ),
        Check(
            "state: the browsed price landed durably (remember = a plain write)",
            any("499" in content for content in stored),
            rationale=None if stored else "nothing was written",
            kind="state",
        ),
        Check(
            "state: a skill was learned from the round",
            bool(skills),
            kind="state",
        ),
        Check(
            "state: the skill is NOT attached (learning does not instantiate)",
            bool(rows) and all(row.skill_name is None for row in rows),
            kind="state",
        ),
        Check(
            "state: no program was rendered into the collection",
            bool(rows) and all(row.extraction_prompt is None for row in rows),
            kind="state",
        ),
        Check(
            "state: nothing was scheduled (no trigger, no notify)",
            bool(rows)
            and all(row.collector_interval_seconds is None and not row.notify for row in rows),
            kind="state",
        ),
        Check(
            "reply: she reports the value she stored (SAID == DID)",
            any("499" in text for text in outgoing_replies(db)),
            kind="reply",
        ),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        Check(
            "calls: the machine landed in learn",
            _landed_state(db) == ConversationState.LEARN.value,
            rationale=f"landed in {_landed_state(db)}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no bail or continue nudge fired)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_elicit_to_learn_runs_the_round_and_instantiates_nothing(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn: parked on the teach question, the user supplies the steps.
    She follows them once — browse, find, remember — reports the value she
    actually stored, and learns the skill.  She instantiates NOTHING: the
    collection her write created carries no skill, no program, no schedule."""
    await chat_eval(
        case_id="transition-elicit-to-learn",
        message=_TEACH_TURN,
        browse=[AURORA_LISTING_499],
        prepare=lambda penny: _park(penny, ConversationState.ELICIT),
        score=_score_elicit_to_learn,
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )
