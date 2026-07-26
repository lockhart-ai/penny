"""Per-state ENACTMENT contracts (#1706) — what chat does once the machine has
decided where it is.

The classifier suite (``test_state_classifier.py``) proves the machine picks the
right state.  This proves the other half: handed THAT state's single instruction,
chat does the state's job and nothing else.  Both halves run here for real — the
machine is wired into the turn, so a case pushes a message and the production
path classifies it, swaps the instruction, and runs the turn.

**idle → elicit (the first beat)**: a routine is asked for that no skill covers.
Elicit's whole job is to ask to be taught it — one message, naming what it needs
— and, just as load-bearing, to NOT start: no browse, no collection, no claim
that anything is running.  The failure this guards is the #1687 one it was
written from: the model half-doing the task, or announcing setup that never
happened.

Scored structurally wherever possible (what ran, what got created, how many
messages went out) rather than on wording, which is the model's to choose.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import ChatEval, Check, tool_not_called

pytestmark = pytest.mark.eval

_FAMILY = "state-enactment"

# An uncovered routine ask: nothing is seeded, so no skill can cover it and the
# machine's only honest landing is elicit.
_UNCOVERED_ASK = "can you keep an eye on the harbor ferry timetable for me?"


def _score_elicit(db: Database, _sends: set[str], reply: str) -> list[Check]:
    """Elicit did its job: it asked, in one message, and started nothing."""
    transitions = db.machine.recent_transitions(10)
    landed = transitions[0].to_state if transitions else None
    collections = [row.name for row in db.memories.list_all() if row.created_by_run_id]
    outgoing = db.memory("penny-messages").read_all()
    return [
        Check(ok=landed == "elicit", label="machine landed in elicit"),
        # The three ways a turn "starts the task" instead of asking for it.
        Check(ok=tool_not_called(db, "browse"), label="did not browse"),
        Check(ok=tool_not_called(db, "collection_create"), label="created no collection"),
        Check(ok=not collections, label="set nothing up"),
        # ONE message — elicit asks once, it does not think out loud across sends.
        Check(ok=len(outgoing) == 1, label="asked in exactly one message"),
        Check(ok="?" in reply, label="asked the user something"),
    ]


async def test_uncovered_routine_ask_elicits_without_starting(chat_eval: ChatEval) -> None:
    """The first enactment beat: an uncovered routine ask lands in elicit and the
    turn ASKS to be taught rather than improvising a start."""
    await chat_eval(
        case_id="elicit-asks-without-starting",
        message=_UNCOVERED_ASK,
        score=_score_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )
