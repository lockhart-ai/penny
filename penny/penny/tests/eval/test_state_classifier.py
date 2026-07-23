"""Per-edge conversation-state classifier contracts (#1706, beat 1): the
idle → elicit edge, both directions, under the cold-start shape (an empty skill
registry — the apply edge is structurally withheld, so the live union the
classifier sees is elicit vs idle).

Each case sweeps a ten-phrasing pool deterministically (sample i →
``pool[i % 10]``), so at N=10 one run covers every phrasing exactly once and the
per-check cells map 1:1 to phrasings — the input-variation doctrine's first
native customer.  The FIRE pool is request-shaped asks for routines no skill
covers (direct, polite, terse, and recurrence-worded — schedule words are
realistic difficulty, not a separate case).  The HOLD pool is ordinary
conversation: greetings, one-shot questions, and the named boundary case — a
PASSING MENTION of a watchable thing, including recurrence words describing the
USER's own habit and topic twins of fire phrasings — which must NOT be chased
into a teach loop.

Gated at 0.8: two clean 1.00 baseline runs at N=10 (turn-audited — every
draw a first-try tagged in-union answer, boundary thinking read) earned the
stable-green bar, so a later change that degrades either direction fails
loudly instead of printing a report-only line.  Fictional-but-believable
fixtures throughout (the repo is public).
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState, MachineSnapshot
from penny.tests.eval.conftest import ClassifierEval

pytestmark = pytest.mark.eval

_FAMILY = "state-classifier"

# The cold-start idle machine: no prior assistant turn, no parked task, no
# skills — the fresh-install shape (#1699's cold-start surface).
_IDLE = MachineSnapshot(state=ConversationState.IDLE)

# Fire direction — a routine is being asked for and nothing covers it.
_FIRE_POOL = [
    "hey can you keep an eye on the harbor ferry timetable for me?",
    "can you watch the price on ridgelinefoxes.example/den-camera-kit?",
    "i want you to check the tide tables every morning and tell me if low tide is before 9",
    "could you track when the farmers market vendor list changes?",
    "keep tabs on the library's new-arrivals page for me",
    "watch harborseals.example/colony-count and let me know when the number moves",
    "start collecting the daily specials from the corner bakery's site, ok?",
    "monitor the trailhead conditions page — i want to know when the pass opens",
    "hey, track auction listings for vintage synths for me",
    "would you keep an eye out for when the ferry adds the late sailing?",
]

# Hold direction — ordinary conversation, incl. the passing-mention boundary
# (phrasings 3/4/6/9 mention watchable things or the user's OWN checking habit;
# 9 is a topic twin of fire phrasing 9).
_HOLD_POOL = [
    "morning! how's it going?",
    "what's the tallest mountain in the andes?",
    "the ferry was packed again this morning, could barely get a seat",
    "i've been checking the auction listings every day lately",
    "thanks, that was really helpful",
    "lol the bakery ran out of croissants before 8 again",
    "what time is it in lisbon right now?",
    "my sister might visit next weekend, thinking we'll hit the tidepools",
    "prices on vintage synths are getting ridiculous these days",
    "remind me what we talked about yesterday?",
]


async def test_idle_to_elicit_fires_on_uncovered_requests(
    classifier_eval: ClassifierEval,
) -> None:
    """Fire: a request-shaped ask for a routine no skill covers classifies
    elicit — the entry edge of the whole teach loop."""
    await classifier_eval(
        case_id="idle-elicit-fire",
        snapshot=_IDLE,
        pool=_FIRE_POOL,
        expected=ConversationState.ELICIT,
        min_pass_rate=0.8,
        family=_FAMILY,
    )


async def test_idle_holds_on_chat_and_passing_mentions(
    classifier_eval: ClassifierEval,
) -> None:
    """Hold: chat, questions, and passing mentions of watchable things classify
    idle — don't chase a mention into a teach loop."""
    await classifier_eval(
        case_id="idle-elicit-hold",
        snapshot=_IDLE,
        pool=_HOLD_POOL,
        expected=ConversationState.IDLE,
        min_pass_rate=0.8,
        family=_FAMILY,
    )
