"""Framing a round at learn ENTRY, and the container it runs into (#1868).

Driven through ``ConversationMachine.advance`` — the public entry point the channel
calls — with a deterministic mock model answering both draws of the turn (the state
classifier's and the framer's, dispatched on their system prompts), so what is exercised
is the real seam: a move that lands in learn draws the interface, derives the container's
name, builds it, and records the framing on the move that settled it.

What is pinned here is everything the framing decides that no later step can undo:

* the container is built INERT — storage with no program, no schedule, no notify — and
  the framing lands on the transition row, so the turn is entered with its destination
  already decided;
* a correction that keeps the job's identity finds the SAME container, and one that
  shifts it archives the empty container it replaces and builds the one it now needs;
* a failed draw builds nothing and records nothing, and the round runs exactly as it did
  before this hook existed;
* the framing rides the anchor's lifecycle — carried while the round stays parked,
  dropped at idle;
* a round that taught nothing takes its empty container with it (#1839), and a container
  holding entries is never taken.

The live-model half — whether the draw picks the right values — is the framing eval's.
All content is synthetic.
"""

from __future__ import annotations

from typing import Any, cast

from penny.constants import TransitionCause
from penny.conversation_machine import (
    ConversationMachine,
    ConversationState,
    RoundFraming,
    StateClassifier,
)
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.skills import SkillDraft
from penny.llm.models import LlmMessage, LlmResponse
from penny.round_framing import RoundFramer, discard_round_container
from penny.tests.conftest import require_memory
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.micro_context import SKILL_FRAME_SYSTEM_PROMPT

# ── The round: an ask, the teach question it got, and the demonstration ────────

_ASK = "keep an eye on the price of the harbor kayak rental page"
_DEMO = "go to harborkayak.example/rentals, find the day rate, and remember it"
_CORRECTION = "actually use harborkayak.example/tours — find the day rate there and remember it"

_FRAMED = (
    "NAME: watch-rental-price\n"
    "DESCRIPTION: keep a rental page's current day rate up to date\n"
    "PARAMETER url — the rental page to read\n"
    "VALUE url: harborkayak.example/rentals"
)
_FRAMED_TOURS = (
    "NAME: watch-rental-price\n"
    "DESCRIPTION: keep a rental page's current day rate up to date\n"
    "PARAMETER url — the rental page to read\n"
    "VALUE url: harborkayak.example/tours"
)

# What ``derive_collection_name`` makes of each — written out rather than computed, so a
# change to the scheme fails HERE, naming the name that moved, instead of agreeing with
# itself.
_CONTAINER = "watch-rental-price-harborkayak-example-rentals"
_TOURS_CONTAINER = "watch-rental-price-harborkayak-example-tours"


def _model(*, state: str, framing: str) -> MockLlmClient:
    """A mock model answering the turn's TWO draws, dispatched on the system prompt.

    Both are micro-contexts on the same client, so the system prompt is what tells them
    apart — the same discipline the run-end pair's mock keeps, and what stops one draw's
    fixture answering the other."""
    model = MockLlmClient()

    def respond(request: dict, _count: int) -> LlmResponse:
        system = next(
            (m.get("content", "") for m in request["messages"] if m.get("role") == "system"), ""
        )
        drawn = framing if system == SKILL_FRAME_SYSTEM_PROMPT else state
        return LlmResponse(message=LlmMessage(role="assistant", content=drawn))

    model.set_response_handler(respond, answers_state_classifier=True)
    return model


def _machine(db: Database, model: MockLlmClient) -> ConversationMachine:
    """A machine over the real store, with the real framer, driven by ``model``."""
    client = cast(Any, model)
    return ConversationMachine(
        db, StateClassifier(client), RoundFramer(db, client, cast(Any, MockLlmClient()))
    )


def _log(db: Database, content: str) -> int:
    message_id = db.messages.log_message(direction="incoming", sender="tester", content=content)
    assert message_id is not None
    return message_id


def _park_in_elicit(db: Database, anchor_id: int) -> None:
    """The state the preceding beat leaves behind: parked in elicit, on the ask."""
    db.machine.record_transition(
        from_state=ConversationState.IDLE.value,
        to_state=ConversationState.ELICIT.value,
        cause=TransitionCause.CLASSIFIER,
        anchor_message_id=anchor_id,
    )


async def _enter_learn(db: Database, framing: str = _FRAMED, message: str = _DEMO) -> None:
    """One move from a parked elicit into learn, framed by ``framing``."""
    machine = _machine(db, _model(state="STATE: learn", framing=framing))
    await machine.advance(message, message_id=_log(db, message), run_id="run-learn")


# ── Entry: the round is framed and its container is built, before the turn ────


async def test_entering_learn_frames_the_round_and_builds_an_inert_container(db):
    """The whole of the entry hook in one move: the draw reads the round's user turns,
    Python derives the container's name from the skill plus the value the user said, and
    builds it — INERT, so the dispatcher (which selects on a rendered program) never runs
    anything against a container whose routine does not exist yet.

    The framing lands on the move that settled it, which is what makes it a READ for
    everything that follows — the turn's own instruction, run-end extraction, and a
    correction comparing identities."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)

    await _enter_learn(db)

    machine = _machine(db, _model(state="STATE: learn", framing=_FRAMED))
    framing = machine.framing()
    assert framing is not None
    assert framing.signature.name == "watch-rental-price"
    assert [(p.name, p.value) for p in framing.signature.parameters] == [
        ("url", "harborkayak.example/rentals")
    ]
    assert framing.container == _CONTAINER

    row = db.memories.get(_CONTAINER)
    assert row is not None
    assert row.description == "keep a rental page's current day rate up to date"
    assert row.extraction_prompt is None
    assert row.schedule is None
    assert row.notify is False
    assert row.skill_name is None
    assert row.created_by_run_id == "run-learn"


async def test_the_framer_reads_the_round_s_own_turns_and_nothing_else(db):
    """The framer's input contract, unchanged by the move (#1830): the ask the round is
    anchored to and the message that just arrived, in that order, and nothing else — no
    assistant turn, no tool calls, no values list.

    Both of those turns existing at this moment is the whole reason the draw moved here."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    db.messages.log_message(
        direction="outgoing", sender="penny", content="walk me through it once?"
    )
    model = _model(state="STATE: learn", framing=_FRAMED)

    machine = _machine(db, model)
    await machine.advance(_DEMO, message_id=_log(db, _DEMO), run_id="run-learn")

    framing_requests = [
        request
        for request in model.requests
        if any(m.get("content") == SKILL_FRAME_SYSTEM_PROMPT for m in request["messages"])
    ]
    assert len(framing_requests) == 1
    assert framing_requests[0]["messages"][1]["content"] == f"{_ASK}\n{_DEMO}"


async def test_a_failed_entry_draw_builds_nothing_and_records_no_framing(db):
    """Honest degradation (#1868): a draw that never met its contract leaves NO container
    and NO framing, and the move is recorded exactly as it was before the hook existed.

    The round then runs unframed and is framed at run end, which is the path this beat
    replaced rather than deleted — a flaky draw costs the round its container, never its
    turn."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)

    await _enter_learn(db, framing="i think this is a price watcher of some sort")

    machine = _machine(db, _model(state="STATE: learn", framing=_FRAMED))
    assert machine.state() is ConversationState.LEARN
    assert machine.framing() is None
    latest = db.machine.latest_transition()
    assert latest is not None and latest.skill_frame is None
    assert db.memories.get(_CONTAINER) is None


# ── Corrections: the entry hook runs again, and identity decides the container ─


async def test_a_correction_that_keeps_the_job_reuses_the_same_container(db):
    """A correction re-enters learn, so the hook runs again — and the same skill bound to
    the same value derives the same name, which FINDS the container the round is already
    writing into.  Find-or-create is tier-1 dedup by construction (#1775), so nothing is
    created and nothing is retired."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    before = db.memories.get(_CONTAINER)
    assert before is not None

    await _enter_learn(db, message="same page, but take the weekend rate instead")

    assert [row.name for row in db.memories.list_all() if row.name == _CONTAINER] == [_CONTAINER]
    after = db.memories.get(_CONTAINER)
    assert after is not None and after.archived is False
    assert after.created_at == before.created_at, "the container was found, not rebuilt"


async def test_a_correction_that_shifts_the_job_archives_the_empty_container(db):
    """A correction that points the round at a different place is a DIFFERENT job, so it
    derives a different name — and the container the round no longer needs is archived
    rather than left in the store map describing a job nobody is doing.

    Archived, not deleted: a retired mechanism stays a visible tombstone, so a container
    that came and went is answerable."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)

    await _enter_learn(db, framing=_FRAMED_TOURS, message=_CORRECTION)

    retired = db.memories.get(_CONTAINER)
    assert retired is not None and retired.archived is True
    built = db.memories.get(_TOURS_CONTAINER)
    assert built is not None and built.archived is False
    machine = _machine(db, _model(state="STATE: learn", framing=_FRAMED))
    framing = machine.framing()
    assert framing is not None and framing.container == _TOURS_CONTAINER


async def test_a_shifted_job_keeps_a_container_that_holds_entries(db):
    """The no-litter rule's one guard, read rather than judged: litter is a container
    minutes old with nothing in it.  A container holding entries is something the user can
    still read, whatever the round did next, so the shift leaves it alone and simply builds
    the one it now needs."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    require_memory(db, _CONTAINER).write([EntryInput(key="day rate", content="$45")], author="chat")

    await _enter_learn(db, framing=_FRAMED_TOURS, message=_CORRECTION)

    kept = db.memories.get(_CONTAINER)
    assert kept is not None and kept.archived is False
    assert db.memories.get(_TOURS_CONTAINER) is not None


async def test_re_teaching_a_job_brings_its_archived_container_back(db):
    """The same job taught again after an earlier round was discarded: its container is
    the one that already carries this job's name, so it comes back rather than being
    shadowed by a second row nobody can reach."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    discard_round_container(
        db, RoundFraming.model_validate_json(_latest_frame(db)), run_id="run-learn"
    )
    discarded = db.memories.get(_CONTAINER)
    assert discarded is not None and discarded.archived is True

    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)

    revived = db.memories.get(_CONTAINER)
    assert revived is not None and revived.archived is False


# ── The framing's lifecycle: carried while parked, cleared at idle ─────────────


async def test_the_framing_is_carried_while_parked_and_cleared_on_break_out(db):
    """The framing belongs to the ROUND, so it rides the anchor's own lifecycle: a later
    move that keeps the machine parked carries it unchanged — which is what lets the turn
    that accepts what was demonstrated read the same container the turn that demonstrated
    it wrote into — and the break-out to idle drops it with everything else."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    _seed_skill(db)

    accepting = _machine(db, _model(state="STATE: apply\nSKILL: watch-rental-price", framing=""))
    await accepting.advance("yes please", message_id=_log(db, "yes please"), run_id="run-apply")
    carried = accepting.framing()
    assert carried is not None and carried.container == _CONTAINER

    bailing = _machine(db, _model(state="STATE: idle", framing=""))
    await bailing.advance("never mind", message_id=_log(db, "never mind"), run_id="run-idle")
    assert bailing.state() is ConversationState.IDLE
    assert bailing.framing() is None


async def test_a_held_draw_frames_nothing_and_touches_no_container(db):
    """Fail → stay holds the REGISTRY too (#1868): a classifier draw that violated its
    contract leaves the machine in learn, but it is not a move INTO learn, so nothing is
    framed and no container is built, archived or revived.

    The machine's own rule is that a contract failure moves nothing; building a container
    off a draw it refused to act on would be that rule holding for the state and not for
    the store."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    before = {row.name for row in db.memories.list_all()}

    held = _machine(db, _model(state="i think we should keep learning here", framing=_FRAMED_TOURS))
    await held.advance(_CORRECTION, message_id=_log(db, _CORRECTION), run_id="run-held")

    assert held.state() is ConversationState.LEARN
    assert {row.name for row in db.memories.list_all()} == before
    kept = db.memories.get(_CONTAINER)
    assert kept is not None and kept.archived is False
    framing = held.framing()
    assert framing is not None and framing.container == _CONTAINER


async def test_a_failed_re_draw_keeps_the_round_s_existing_framing(db):
    """A correction whose framing could not be drawn keeps the container the round already
    has, instead of losing it to a flaky draw — the carry rule read the other way."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)

    await _enter_learn(db, framing="not a framing at all", message="try that again please")

    machine = _machine(db, _model(state="STATE: learn", framing=_FRAMED))
    framing = machine.framing()
    assert framing is not None and framing.container == _CONTAINER


# ── The failed round takes its container with it (#1839) ──────────────────────


async def test_a_failed_round_s_empty_container_is_archived(db):
    """A learn turn that taught nothing leaves a container describing a routine that does
    not exist — minutes old, empty, named for a job nobody can run — so the honest failure
    takes it with it rather than leaving it in the store map."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    framing = RoundFraming.model_validate_json(_latest_frame(db))

    discard_round_container(db, framing, run_id="run-learn")

    row = db.memories.get(_CONTAINER)
    assert row is not None and row.archived is True


async def test_a_failed_round_keeps_a_container_that_holds_entries(db):
    """Same guard, same reason: what the round wrote stays reachable, whatever the round
    failed to learn."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    require_memory(db, _CONTAINER).write([EntryInput(key="day rate", content="$45")], author="chat")

    discard_round_container(db, RoundFraming.model_validate_json(_latest_frame(db)), run_id="r")

    row = db.memories.get(_CONTAINER)
    assert row is not None and row.archived is False


async def test_an_unframed_round_has_nothing_to_discard(db):
    """The degrade path all the way through: a round nothing framed has no container, so
    the failure path is a no-op rather than a guess at which collection to retire."""
    discard_round_container(db, None, run_id="run-learn")

    assert not [row for row in db.memories.list_all() if row.archived]


def _latest_frame(db: Database) -> str:
    """The serialized framing on the newest move — the round state everything reads."""
    latest = db.machine.latest_transition()
    assert latest is not None and latest.skill_frame is not None
    return latest.skill_frame


def _seed_skill(db: Database) -> None:
    """The registry the acceptance turn's apply draw needs a member of — the skill the
    round is about to learn, present so the gated edge is offered at all."""
    db.skills.upsert(
        SkillDraft(
            name="watch-rental-price",
            intent="keep a rental page's current day rate up to date",
            description="keep a rental page's current day rate up to date",
            steps=[],
            parameters=[],
            source_run_id="run-learn",
        ),
        author="chat",
    )
