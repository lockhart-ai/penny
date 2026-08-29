"""The PRIORS a case stands its measured turn on (#1995).

A case's first line is the world its turn arrives into, and for most interesting turns that
is not a cold machine: a learn close only exists inside a learn round, an apply close only
inside an apply one.  Left to a live classifier draw over a cold machine, the round the case
is about may simply never happen — measured, an imperative about now drew ``idle`` on 10 of
10 samples on both models, so every reply check failed for a reply nobody had been asked to
write (#1989).

So the priors are laid down rather than hoped for, through the transition suite's own seeding
idiom — imported, not restated, because a seeded machine state, a seeded conversation turn and
a seeded ledger row are one shape, and a second copy of it would be a second contract free to
drift from the one every edge case is measured against.

Each builder returns a ``Seeder``: a plain callable a case hands to the driver as
``seed=``.  Every one of them asserts the state it claims at seed time, so a seed that has
drifted from its own description fails HERE rather than as a puzzling 0.00 after a paid run.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState, MachineSnapshot
from penny.database import Database
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import seeded_run_id
from penny.tests.eval.test_state_transitions import (
    _drawn_state,
    _log_ask,
    _log_chat_step,
    _log_classifier_draw,
    _log_reply,
    _park,
    _seeded_response,
)

# A case's priors: whatever must already be true when the measured turn arrives.
Seeder = Callable[[Database], None]


def round_parked_in_elicit(ask: str, teach_question: str) -> Seeder:
    """The round an ELICITATION turn leaves behind, item for item:

    * the user's ask, logged INCOMING — the message the round is ANCHORED to
    * Penny's teach question, logged OUTGOING and THREADED to it — her last turn, the one a
      demonstration answers, and the only way a reply of hers reaches the window
    * that turn's LEDGER — the draw that chose elicit over a cold machine, and the one chat
      call that answered.  No tool calls: an elicitation turn enacts nothing
    * the machine parked in ``elicit``, carrying that ask as its anchor

    Nothing else: an empty registry, no collection, no page read.  What the round is FOR has
    been said and what it DOES has not, which is exactly the world a demonstration arrives
    into — and exactly what the measured turn is about to do for the first time.
    """
    draw_run = seeded_run_id("elicit-draw")
    turn_run = seeded_run_id("elicit-turn")

    def seed(db: Database) -> None:
        ask_id = _log_ask(db, ask, "standing-elicit-round")
        _log_reply(db, teach_question, answering=ask_id)
        _log_classifier_draw(
            db,
            run_id=draw_run,
            snapshot=MachineSnapshot(state=ConversationState.IDLE),
            message=ask,
            drawn=_drawn_state(ConversationState.ELICIT),
        )
        _log_chat_step(
            db,
            run_id=turn_run,
            messages=[{"role": "user", "content": ask}],
            response=_seeded_response(teach_question),
        )
        _park(
            db,
            ConversationState.ELICIT,
            anchor_message_id=ask_id,
            run_id=turn_run,
            message_id=ask_id,
        )
        _assert_parked_in_elicit(db, ask_id, ask, teach_question)
        _assert_the_round_has_not_started(db)

    return seed


def _assert_parked_in_elicit(db: Database, ask_id: int, ask: str, teach_question: str) -> None:
    """Loud probe: the machine is parked in elicit on THIS ask, and the round reads back as a
    two-turn CONVERSATION rather than as one user turn (Penny's turn reaches the window only
    because it is threaded to the ask)."""
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.ELICIT.value, (
        f"the machine must be parked in elicit, not {latest}"
    )
    assert latest.anchor_message_id == ask_id, (
        f"the park must be anchored to the ask, not {latest.anchor_message_id}"
    )
    expected = [
        (PennyConstants.MessageDirection.INCOMING, ask),
        (PennyConstants.MessageDirection.OUTGOING, teach_question),
    ]
    window = db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=len(expected))
    seen = [(row.direction, row.content) for row in window]
    assert seen == expected, f"the round must read as a two-turn conversation, got {seen}"


def _assert_the_round_has_not_started(db: Database) -> None:
    """The other half of that probe — the seeded turn enacted NOTHING, which is the whole of
    an elicitation turn's contract: no routine in the registry, and no page fetched."""
    assert not db.skills.list_all(), "the round starts with no routine in the registry"
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    fetched = browse_log.read_recent(window_seconds=3600, cap=None) if browse_log else []
    assert not fetched, "the round starts with no page read"
