"""Framing a round at its ENTRY, and the container it runs into (#1868/#1870/#1894).

Driven through ``ConversationMachine.advance`` — the public entry point the channel
calls — with a deterministic mock model answering every draw of the turn (the state
classifier's, the framer's and the binder's, dispatched on their system prompts), so what
is exercised is the real seam: a move that lands in learn or apply settles the round's
interface, derives the container's name, builds it, and records the framing on the move
that settled it.

What is pinned here is everything the framing decides that no later step can undo:

* the container is built INERT — storage with no program, no schedule, no notify — and
  the framing lands on the transition row, so the turn is entered with its destination
  already decided;
* a round's identity is settled ONCE, at its first entry (#1902): a correction re-enters
  learn and makes NO framing draw, so the same routine and the same container carry and
  the corrected write lands in place — while a round still unframed (its entry draw
  failed) has no identity to keep and is drawn again;
* a failed draw builds nothing and records nothing, and the round runs exactly as it did
  before this hook existed;
* the framing rides the anchor's lifecycle — carried while the round stays parked,
  dropped at idle;
* a move into APPLY settles one when the round arrived without one and settles nothing
  when the round already has one (#1875) — and WHICH draw settles it is the beat-4 split
  (#1870): learn MINTS a routine through the framer, a cold apply FILLS the routine the
  classifier bound through the binder, so the container is named after a routine the
  registry actually holds;
* find-or-create on that derived name is tier-1 dedup by construction — the same job
  re-asked runs into the container it already had, a different value mints its own;
* a binder that cannot settle the round at all (a routine the registry does not hold, or a
  draw that came back unusable) builds nothing and records nothing, which is the state the
  apply turn then fails honestly on — while a SHORTFALL (#1885) is the enumerated outcome
  beside it: the routine covers the ask and the words named no value for something it
  needs, so the move lands in REQUEST carrying what the turn has to ask for, still builds
  nothing, and a next message the binder still cannot complete asks again;
* a request the classifier drew DIRECTLY runs that same binder (#1894), so both doors into
  request carry the same partial binding — and that binding is round state: recorded on
  the move, read back by later turns, rendered to the classifier as what the round is
  waiting on, completed from the message that finally supplies the rest — which stands the
  job up in apply, on the container the completed values derive, still anchored to the ask
  that opened the round (#1892) — and dropped the moment the round leaves request;
* a round that taught nothing takes its empty container with it (#1839), and a container
  holding entries is never taken;
* a round that BAILS to idle takes its container with it too (#1896), the emptiness guard
  notwithstanding — the bail discards the round's intermediate state, and what a
  demonstration wrote is part of that state — AND the draft routine the round registered
  goes with it (#1902), while the post-apply structural reset, which clears the framing
  before anything is classified, leaves a job that was just set running alone.

The live-model half — whether a draw picks the right values — is the framing and binding
evals'.  All content is synthetic.
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
from penny.database.skills import SkillDraft, SkillParameter
from penny.llm.models import LlmMessage, LlmResponse
from penny.round_framing import (
    RoundFramer,
    abandon_round_container,
    abandon_round_skill,
    discard_round_container,
)
from penny.tests.conftest import require_memory
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.micro_context import (
    BIND_SKILL_SYSTEM_PROMPT,
    SKILL_FRAME_SYSTEM_PROMPT,
    STATE_CLASSIFIER_SYSTEM_PROMPT,
)

# ── The round: an ask, the teach question it got, and the demonstration ────────

_ASK = "keep an eye on the price of the harbor kayak rental page"
# The same ask with the page IN it — what an ask looks like when it settles the job on its
# own, which is the only shape a framing drawn at apply entry can bind values from.
_ASK_NAMING_THE_PAGE = "keep an eye on the price on harborkayak.example/rentals"
_DEMO = "go to harborkayak.example/rentals, find the day rate, and remember it"
_CORRECTION = "actually use harborkayak.example/tours — find the day rate there and remember it"

# What the routine is called once it is in the registry — the name the framer mints and
# the name a later cold ask binds, which are one string on purpose: the container derived
# at learn entry and the one derived at a cold apply entry then agree, which is what makes
# re-asking for the same job find the container it already had.
_SKILL = "watch-rental-price"
# A second routine in the registry, so a round can come back naming one that is not the one
# it was parked on — the case a carried binding must not survive.
_OTHER_SKILL = "watch-tour-price"

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

# The BINDER's answers for the same routine (#1870): it mints nothing, so it says only
# which part of the user's words fills the parameter the registry already declares.
_BOUND = "VALUE url: harborkayak.example/rentals"
_BOUND_TOURS = "VALUE url: harborkayak.example/tours"

# The two-parameter shape (#1885/#1894): an ask that names WHICH rate and not WHICH page,
# so the same draw settles one parameter and reports the other missing — the only shape
# that can show a partial binding at all, and the one a request round is negotiated from.
_RATE_ASK = "keep an eye on the weekend rate for me"
_SHORT_OF_THE_PAGE = "MISSING url\nVALUE keyword: the weekend rate"

# What ``derive_collection_name`` makes of each — written out rather than computed, so a
# change to the scheme fails HERE, naming the name that moved, instead of agreeing with
# itself.  The framer's minted name and the registry's name are the same string here, which
# is what lets one pair of constants stand for both entries.
_CONTAINER = "watch-rental-price-harborkayak-example-rentals"
_TOURS_CONTAINER = "watch-rental-price-harborkayak-example-tours"
# The same scheme over BOTH values of the two-parameter routine, in declared order — the
# name a round that negotiated its missing page across two turns derives once it is whole.
_TWO_VALUE_CONTAINER = "watch-rental-price-harborkayak-example-rentals-the-weekend-rate"


def _model(*, state: str, framing: str = _FRAMED, binding: str = _BOUND) -> MockLlmClient:
    """A mock model answering every draw the turn can make, dispatched on the system
    prompt.

    All three are micro-contexts on the same client, so the system prompt is what tells
    them apart — the same discipline the run-end pair's mock keeps, and what stops one
    draw's fixture answering another's question.  Which of the framer and the binder
    actually runs is the seam under test, so both are always answerable and a test asserts
    which one was asked."""
    model = MockLlmClient()

    def respond(request: dict, _count: int) -> LlmResponse:
        system = next(
            (m.get("content", "") for m in request["messages"] if m.get("role") == "system"), ""
        )
        drawn = {SKILL_FRAME_SYSTEM_PROMPT: framing, BIND_SKILL_SYSTEM_PROMPT: binding}.get(
            system, state
        )
        return LlmResponse(message=LlmMessage(role="assistant", content=drawn))

    model.set_response_handler(respond, answers_state_classifier=True)
    return model


def _drew_against(model: MockLlmClient, system_prompt: str) -> bool:
    """Whether a draw was made against ``system_prompt`` — which micro-context the turn
    actually asked, read off the requests the client saw."""
    return any(
        any(message.get("content") == system_prompt for message in request["messages"])
        for request in model.requests
    )


def _drawn_content(model: MockLlmClient, system_prompt: str) -> str:
    """The DOCUMENT a micro-context was handed — what that draw was actually asked about,
    read off the same requests, so a test can pin what reached the model rather than only
    that something did."""
    for request in model.requests:
        messages = request["messages"]
        if any(message.get("content") == system_prompt for message in messages):
            return str(messages[-1].get("content", ""))
    raise AssertionError("no draw was made against that system prompt")


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


# ── Corrections: the round's identity is settled ONCE, at its first entry (#1902) ─


async def test_a_correction_keeps_the_round_s_framing_and_draws_nothing(db):
    """A correction re-enters learn to refine the PROGRAM of the round's one job, never to
    decide what that job is (#1902) — so the framer is not asked at all, and the framing
    the round settled at its first entry is what the corrected move records.

    Re-drawing is what forked a round in two: a corrected ask read as a different subject,
    the fresh draw derived a fresh name, find-or-create minted a sibling container under
    it, and run-end extraction registered a second routine beside the first."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    before = db.memories.get(_CONTAINER)
    assert before is not None

    # The correction points at a different page and its draw would frame a different job —
    # neither reaches the round, because no draw is made.
    model = _model(state="STATE: learn", framing=_FRAMED_TOURS)
    correcting = _machine(db, model)
    await correcting.advance(_CORRECTION, message_id=_log(db, _CORRECTION), run_id="run-fix")

    assert not _drew_against(model, SKILL_FRAME_SYSTEM_PROMPT)
    framing = correcting.framing()
    assert framing is not None and framing.container == _CONTAINER
    assert framing.skill == _SKILL
    assert db.memories.get(_TOURS_CONTAINER) is None, "no sibling container was minted"
    after = db.memories.get(_CONTAINER)
    assert after is not None and after.archived is False
    assert after.created_at == before.created_at, "the container was found, not rebuilt"


async def test_a_correction_brings_a_failed_round_s_container_back(db):
    """The one thing a carried re-entry still SETTLES: the container (#1902).

    A learn turn that taught nothing takes its empty container with it (#1839), and the
    correction that follows is exactly the round coming back for it — so find-or-create
    runs on the carried framing and the container is revived, rather than the corrected
    demonstration writing into an archived collection nothing can configure."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    discard_round_container(
        db, RoundFraming.model_validate_json(_latest_frame(db)), run_id="run-learn"
    )
    discarded = db.memories.get(_CONTAINER)
    assert discarded is not None and discarded.archived is True

    await _enter_learn(db, message="try that again, take the weekend rate this time")

    revived = db.memories.get(_CONTAINER)
    assert revived is not None and revived.archived is False


async def test_a_correction_leaves_what_the_round_already_wrote_in_place(db):
    """The correction keeps writing into the container the round has been writing into, so
    what the first demonstration put there is still there and reachable — one identity, one
    container.  The re-draw this replaces would have left those entries behind in a
    container the round no longer pointed at."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    require_memory(db, _CONTAINER).write([EntryInput(key="day rate", content="$45")], author="chat")

    await _enter_learn(db, framing=_FRAMED_TOURS, message=_CORRECTION)

    kept = db.memories.get(_CONTAINER)
    assert kept is not None and kept.archived is False
    assert [entry.key for entry in require_memory(db, _CONTAINER).read_all()] == ["day rate"]
    assert db.memories.get(_TOURS_CONTAINER) is None


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

    accepting = _machine(db, _model(state=f"STATE: apply\nSKILL: {_SKILL}", framing=""))
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


async def test_a_round_that_is_still_unframed_is_drawn_again_on_re_entry(db):
    """The boundary case of the carry rule (#1902): what a re-entry keeps is the round's
    IDENTITY, and a round whose entry draw failed has none to keep.

    So it is drawn again — the round runs unframed until some entry settles it, rather
    than being locked out of ever having a container because its first draw was flaky."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db, framing="not a framing at all")
    unframed = _machine(db, _model(state="STATE: learn", framing=""))
    assert unframed.state() is ConversationState.LEARN and unframed.framing() is None

    model = _model(state="STATE: learn", framing=_FRAMED)
    retrying = _machine(db, model)
    await retrying.advance(_DEMO, message_id=_log(db, _DEMO), run_id="run-retry")

    assert _drew_against(model, SKILL_FRAME_SYSTEM_PROMPT)
    framing = retrying.framing()
    assert framing is not None and framing.container == _CONTAINER
    assert db.memories.get(_CONTAINER) is not None


# ── Apply entry: the BINDER settles it, because apply cannot run unframed (#1870) ─


async def _cold_apply(
    db: Database, ask: str = _ASK_NAMING_THE_PAGE, binding: str = _BOUND, skill: str = _SKILL
) -> tuple[ConversationMachine, MockLlmClient]:
    """A cold ask straight from idle that the classifier answers with apply, binding
    ``skill`` — the shape beat 4 exists for: no round behind it, so the routine comes from
    the registry and the values come from this one message."""
    model = _model(state=f"STATE: apply\nSKILL: {skill}", binding=binding)
    machine = _machine(db, model)
    await machine.advance(ask, message_id=_log(db, ask), run_id="run-apply")
    return machine, model


async def test_a_cold_apply_binds_the_known_routine_and_builds_its_container(db):
    """The edge the whole beat exists for: a routine Penny already knows, pointed at a
    space this one message names.

    Nothing is minted — the classifier bound the routine when it decided apply, so the
    BINDER runs instead of the framer and the only open question is which part of the words
    fills what that routine declares.  The container is then named from a routine the
    registry actually holds, which is what makes it configurable: a name minted here would
    name a job no ``collection_set`` could resolve."""
    _seed_skill(db)

    machine, model = await _cold_apply(db)

    assert machine.state() is ConversationState.APPLY
    assert _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    assert not _drew_against(model, SKILL_FRAME_SYSTEM_PROMPT)
    framing = machine.framing()
    assert framing is not None
    assert framing.container == _CONTAINER
    assert framing.skill == _SKILL
    assert framing.bound_values() == {"url": "harborkayak.example/rentals"}
    row = db.memories.get(_CONTAINER)
    assert row is not None
    assert row.extraction_prompt is None and row.schedule is None
    assert row.created_by_run_id == "run-apply"


async def test_the_same_job_asked_again_runs_into_the_container_it_already_had(db):
    """Tier-1 dedup by construction (#1775): the same routine and the same values derive
    the same name, so re-asking finds the container that exists rather than minting a
    second one beside it.

    That is what makes re-asking a RECONFIGURE — the framing points at the container
    already carrying this job, so the turn's ``collection_set`` lands on it."""
    _seed_skill(db)
    await _cold_apply(db)
    require_memory(db, _CONTAINER).write([EntryInput(key="day rate", content="$45")], author="chat")

    machine, _ = await _cold_apply(db)

    framing = machine.framing()
    assert framing is not None and framing.container == _CONTAINER
    assert [row.name for row in db.memories.list_all() if row.name.startswith("watch-rental")] == [
        _CONTAINER
    ]
    assert require_memory(db, _CONTAINER).read_all(), "the container kept what it already held"


async def test_a_different_value_mints_a_container_of_its_own(db):
    """The other half of the same rule: a different place is a different job, so it derives
    a distinct name and gets its own container — and the job already running is untouched
    by the one being set up beside it."""
    _seed_skill(db)
    await _cold_apply(db)

    machine, _ = await _cold_apply(
        db, ask="also keep an eye on the price on harborkayak.example/tours", binding=_BOUND_TOURS
    )

    framing = machine.framing()
    assert framing is not None and framing.container == _TOURS_CONTAINER
    assert db.memories.get(_CONTAINER) is not None
    assert db.memories.get(_TOURS_CONTAINER) is not None


async def test_an_apply_after_an_unframed_learn_round_binds_from_the_ask_and_the_acceptance(db):
    """The #1875 seam this beat re-implements: a round whose learn-entry draw came back
    malformed reaches apply with nothing settled, and the BINDER settles it there.

    Its input is BOTH user turns — the ask the round is anchored to and the acceptance that
    just arrived — because the value the routine needs was said in the ask and the
    acceptance only agrees to it.  That is the whole reason the turns are handed over in
    order rather than the arriving message alone, and it is what makes the recovery
    possible: the container is built now, named after a routine the registry holds, instead
    of being invented by the turn."""
    anchor_id = _log(db, _ASK_NAMING_THE_PAGE)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db, framing="i think this is a price watcher of some sort")
    assert db.memories.get(_CONTAINER) is None
    _seed_skill(db)

    model = _model(state=f"STATE: apply\nSKILL: {_SKILL}")
    machine = _machine(db, model)
    await machine.advance("yes please", message_id=_log(db, "yes please"), run_id="run-apply")

    assert machine.state() is ConversationState.APPLY
    assert _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    framing = machine.framing()
    assert framing is not None and framing.container == _CONTAINER
    row = db.memories.get(_CONTAINER)
    assert row is not None and row.created_by_run_id == "run-apply"


async def test_a_cold_apply_whose_draw_comes_back_unusable_carries_no_framing(db):
    """The binder's other empty-handed outcome — nothing usable came back at all, as
    against a draw that read the words and found them short.

    Same result either way, because both leave the round unsettled: no container, no
    framing on the move, and an apply turn that fails honestly rather than standing
    something up on a guess."""
    _seed_skill(db)

    machine, model = await _cold_apply(db, binding="not a binding at all")

    assert machine.framing() is None
    assert _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    latest = db.machine.latest_transition()
    assert latest is not None and latest.skill_frame is None
    assert db.memories.get(_CONTAINER) is None


async def test_binding_a_routine_the_registry_lacks_builds_nothing(db):
    """A routine that is not there cannot be filled: no draw is made and no container is
    built, so the round stays unframed and the turn fails honestly on that state (#1875).

    Nothing is minted to keep the turn alive — inventing a routine here would stand up a
    job under a name the registry has never heard of, which is the failure #1870 removes
    rather than a recovery.  Driven at the framer directly because the machine cannot
    reach it: a skill-gated draw is membership-validated against the registry it was
    offered, so this is the guard behind that contract rather than a path through it."""
    model = _model(state="STATE: idle")
    framer = RoundFramer(db, cast(Any, model), cast(Any, MockLlmClient()))

    framing = await framer.bind_entry(
        skill=_SKILL, ask=None, message=_ASK_NAMING_THE_PAGE, run_id="run-apply"
    )

    assert framing is None
    assert not _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    assert db.memories.get(_CONTAINER) is None


async def test_a_cold_apply_the_words_fall_short_of_lands_in_request(db):
    """The binder's SHORTFALL outcome, wired (#1885): the draw read the words correctly and
    they named no value for something the routine needs, so the turn does not fail — it
    lands in REQUEST and asks for the rest.

    The classifier drew apply and bound the routine, which is unchanged and right: the
    routine really does cover the ask.  Only the binder can tell a covered-and-bound ask
    from a covered-but-short one, so the binding is what routes the move.  Nothing is built
    for it — the container's name is derived from every value, so a job missing one has no
    name yet — and the move carries no framing while still carrying the routine it bound and
    the ask it opened on."""
    _seed_skill(db)
    anchor_id = _log(db, _ASK)

    model = _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding="MISSING url")
    machine = _machine(db, model)
    entered = await machine.advance(_ASK, message_id=anchor_id, run_id="run-request")

    assert machine.state() is ConversationState.REQUEST
    assert entered.state is ConversationState.REQUEST
    assert entered.decision.state is ConversationState.APPLY, "the draw itself is unchanged"
    assert _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    latest = db.machine.latest_transition()
    assert latest is not None
    assert latest.from_state == ConversationState.IDLE.value
    assert latest.to_state == ConversationState.REQUEST.value
    assert latest.cause == TransitionCause.CLASSIFIER.value
    assert latest.skill_name == _SKILL
    assert latest.anchor_message_id == anchor_id
    assert latest.skill_frame is None
    assert machine.framing() is None
    assert not [row for row in db.memories.list_all() if row.name.startswith("watch-rental")]


async def test_the_request_turn_is_handed_the_routine_and_what_is_still_missing(db):
    """What the request turn is entered WITH: the routine the ask is covered by, what it is
    for, the values the words already settled, and the ones they did not — each carrying the
    registry's own line of what to supply.

    Every one of those is a string the reply copies rather than works out, which is what
    makes the ask n=0: nothing has to be looked up to write it.  The already-settled values
    travel because throwing them away would have the turn ask a second time for something
    the user has already said."""
    _seed_two_parameter_skill(db)

    machine = _machine(
        db, _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding=_SHORT_OF_THE_PAGE)
    )
    entered = await machine.advance(_RATE_ASK, message_id=_log(db, _RATE_ASK), run_id="run-request")

    shortfall = entered.shortfall
    assert shortfall is not None
    assert shortfall.skill == _SKILL
    assert shortfall.description == "keep a rental page's current day rate up to date"
    assert shortfall.bound == {"keyword": "the weekend rate"}
    assert [(one.name, one.description) for one in shortfall.missing] == [
        ("url", "the rental page to read")
    ]


async def test_a_request_the_next_message_still_falls_short_of_asks_again(db):
    """A message arriving on a parked request that the binder STILL cannot complete lands
    back in request, with the round's binding carried and re-drawn against the new words.

    The same rule read a second time, and the same shape as the learn → learn
    re-demonstration edge: the round stays parked on the ask that opened it, so the next
    message is the retry — and it is still waiting on exactly what it was waiting on."""
    _seed_skill(db)
    anchor_id = _log(db, _ASK)
    short = _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding="MISSING url")
    await _machine(db, short).advance(_ASK, message_id=anchor_id, run_id="run-request")

    again = _machine(db, _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding="MISSING url"))
    entered = await again.advance(
        "the usual one", message_id=_log(db, "the usual one"), run_id="run-request-2"
    )

    assert again.state() is ConversationState.REQUEST
    assert entered.shortfall is not None
    assert [one.name for one in entered.shortfall.missing] == ["url"]
    latest = db.machine.latest_transition()
    assert latest is not None
    assert latest.from_state == ConversationState.REQUEST.value
    assert latest.anchor_message_id == anchor_id, "the round is still parked on its own ask"
    assert latest.round_shortfall is not None, "the round still knows what it is waiting on"
    assert not [row for row in db.memories.list_all() if row.name.startswith("watch-rental")]


async def test_a_request_landing_leaves_the_jobs_already_running_alone(db):
    """A round that could not be settled touches NOTHING that already exists: a live job
    beside it keeps its program, its schedule and its entries.

    The claim is worth its own test because the failure it guards is silent — a turn that
    reached for a container it had no name for would land on whichever one it could find,
    and the job it disturbed is one the user is still relying on."""
    _seed_skill(db)
    db.memories.create_collection("a-live-job", "a job already running")
    db.memories.update_collection_metadata(
        "a-live-job", extraction_prompt="1. browse(...)", schedule="FREQ=DAILY", notify=True
    )
    require_memory(db, "a-live-job").write(
        [EntryInput(key="day rate", content="$45")], author="chat"
    )

    await _cold_apply(db, ask="can you keep an eye on the price", binding="MISSING url")

    row = db.memories.get("a-live-job")
    assert row is not None
    assert row.extraction_prompt == "1. browse(...)"
    assert row.schedule == "FREQ=DAILY"
    assert row.notify is True and row.archived is False
    assert [entry.content for entry in require_memory(db, "a-live-job").read_all()] == ["$45"]


async def test_an_apply_move_that_already_has_a_framing_draws_nothing(db):
    """A round that arrives WITH its framing is answered by what it already has: neither
    draw runs, so nothing can come back naming the job differently.

    A re-draw is not a re-read — the round was taught under the entry name and wrote into
    the container derived from it, so a fresh draw here could point the acceptance at a
    container the demonstration never used."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    _seed_skill(db)
    model = _model(
        state=f"STATE: apply\nSKILL: {_SKILL}", framing=_FRAMED_TOURS, binding=_BOUND_TOURS
    )

    machine = _machine(db, model)
    await machine.advance("yes please", message_id=_log(db, "yes please"), run_id="run-apply")

    assert not _drew_against(model, SKILL_FRAME_SYSTEM_PROMPT)
    assert not _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    framing = machine.framing()
    assert framing is not None and framing.container == _CONTAINER
    assert db.memories.get(_TOURS_CONTAINER) is None


# ── Request entry: the same binder, and the binding the round keeps (#1894) ───


async def _park_in_request(
    db: Database, ask: str = _RATE_ASK
) -> tuple[ConversationMachine, MockLlmClient]:
    """A cold ask the classifier answers with REQUEST, naming the routine — the direct
    door, where the binder never used to run at all."""
    model = _model(state=f"STATE: request\nSKILL: {_SKILL}", binding=_SHORT_OF_THE_PAGE)
    machine = _machine(db, model)
    await machine.advance(ask, message_id=_log(db, ask), run_id="run-request")
    return machine, model


async def test_a_request_draw_binds_the_routine_it_named(db):
    """The door this beat opens: a request the classifier drew DIRECTLY runs the same
    binder a short apply runs, so the turn knows which routine covers the ask, what the
    words already settled, and what they did not.

    Before this, every measured request turn came through here and got nothing — the
    generic ask for "the part that is missing" — so the reply had to re-derive the gap from
    raw conversation.  Nothing is built for it, exactly as on the other door: the
    container's name needs every value."""
    _seed_two_parameter_skill(db)

    machine, model = await _park_in_request(db)

    assert machine.state() is ConversationState.REQUEST
    assert _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    assert not _drew_against(model, SKILL_FRAME_SYSTEM_PROMPT)
    waiting = machine.shortfall()
    assert waiting is not None
    assert waiting.skill == _SKILL
    assert waiting.bound == {"keyword": "the weekend rate"}
    assert [(one.name, one.description) for one in waiting.missing] == [
        ("url", "the rental page to read")
    ]
    assert machine.framing() is None
    assert not [row for row in db.memories.list_all() if row.name.startswith("watch-rental")]


async def test_the_round_s_binding_is_recorded_and_read_back_by_a_later_turn(db):
    """The binding is ROUND STATE, not a turn's hand-off: it lands on the move that
    settled it and a machine built fresh over the same store reads it back.

    That is the whole point of persisting it — the next turn is judged against a NAMED gap
    rather than re-auditing the conversation for completeness, and the turn's own ask says
    the same thing this turn's said, because both are reading one recorded answer."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)

    latest = db.machine.latest_transition()
    assert latest is not None
    assert latest.round_shortfall is not None
    assert latest.skill_frame is None, "a round short of a value has no framing to record"

    reread = _machine(db, _model(state="STATE: idle")).shortfall()
    assert reread is not None
    assert reread.skill == _SKILL
    assert reread.bound == {"keyword": "the weekend rate"}
    assert [one.name for one in reread.missing] == ["url"]


async def test_the_parked_binding_is_what_the_next_turn_is_classified_against(db):
    """The parked binding reaches the classifier as its own section: the routine that was
    named, what the user has already given, and what is still needed.

    Which is what the request → apply condition is asked about — whether this message
    supplies the details that were asked for — so the answer is a read of the rendered
    state rather than a judgment about the whole conversation."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)

    model = _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding=_BOUND)
    supply = "harborkayak.example/rentals"
    await _machine(db, model).advance(supply, message_id=_log(db, supply), run_id="run-apply")

    assert (
        "## The details this task is waiting on\n"
        'skill: "watch-rental-price"\n'
        "already given:\n"
        "- keyword: the weekend rate\n"
        "still needed:\n"
        "- url — the rental page to read"
    ) in _drawn_content(model, STATE_CLASSIFIER_SYSTEM_PROMPT)


async def test_the_message_that_arrives_completes_the_parked_binding(db):
    """The other half of holding the binding: the apply draw COMPLETES it from what the
    round already settled plus the message that just arrived.

    Only the still-open parameter is drawn — the value the user gave two turns ago is read,
    not re-derived, so it cannot come back different — and the container is derived from
    the whole set, which is the job they actually asked for.

    The move it records is the request → apply edge (#1892): it comes FROM request and
    keeps the ask that opened the round as its anchor, which is what says the job that
    stood up is the one the user asked for two turns ago rather than a fresh round this
    message started."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)
    parked = db.machine.latest_transition()
    assert parked is not None and parked.anchor_message_id is not None

    model = _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding=_BOUND)
    machine = _machine(db, model)
    supply = "harborkayak.example/rentals"
    await machine.advance(supply, message_id=_log(db, supply), run_id="run-apply")

    assert machine.state() is ConversationState.APPLY
    framing = machine.framing()
    assert framing is not None
    assert framing.bound_values() == {
        "url": "harborkayak.example/rentals",
        "keyword": "the weekend rate",
    }
    assert framing.container == _TWO_VALUE_CONTAINER
    assert db.memories.get(_TWO_VALUE_CONTAINER) is not None
    assert machine.shortfall() is None, "a round that is bound is waiting on nothing"
    landed = db.machine.latest_transition()
    assert landed is not None
    assert landed.from_state == ConversationState.REQUEST.value
    assert landed.anchor_message_id == parked.anchor_message_id, (
        "the round is still the one the ask opened"
    )
    asked = _drawn_content(model, BIND_SKILL_SYSTEM_PROMPT)
    assert "- url: the rental page to read" in asked
    assert "keyword" not in asked, "a settled parameter is not a question to ask again"


async def test_a_binding_settled_for_one_routine_is_not_carried_into_another(db):
    """What one routine's parameters were bound to says nothing about another's, whatever
    the names happen to be — so a round that comes back naming a DIFFERENT routine is bound
    from scratch, every parameter open.

    The names are the trap: two routines can both declare a ``keyword`` and mean unrelated
    things by it, and a value carried across on a name match would be a value nobody
    supplied for the job actually being set up."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)
    _upsert_named_skill(db, _OTHER_SKILL)

    model = _model(
        state=f"STATE: apply\nSKILL: {_OTHER_SKILL}",
        binding="VALUE url: harborkayak.example/tours\nVALUE keyword: the day rate",
    )
    supply = "harborkayak.example/tours, the day rate"
    await _machine(db, model).advance(supply, message_id=_log(db, supply), run_id="run-apply")

    asked = _drawn_content(model, BIND_SKILL_SYSTEM_PROMPT)
    assert "- url: the rental page to read" in asked
    assert "- keyword: which rate to look for" in asked, "nothing was treated as settled"


async def test_a_re_taught_routine_drops_a_value_it_no_longer_declares(db):
    """A skill is REPLACE-able by name, so what a round settled two turns ago is narrowed
    to what the routine still declares before any of it is carried.

    A value bound to a parameter that no longer exists is a value with nowhere to go, and
    carrying it would put it in a signature the registry does not describe — which the
    container's name is derived from."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)
    _seed_skill(db)  # re-taught: the same routine, declaring only the page now

    model = _model(state=f"STATE: apply\nSKILL: {_SKILL}", binding=_BOUND)
    machine = _machine(db, model)
    supply = "harborkayak.example/rentals"
    await machine.advance(supply, message_id=_log(db, supply), run_id="run-apply")

    framing = machine.framing()
    assert framing is not None
    assert framing.bound_values() == {"url": "harborkayak.example/rentals"}
    assert framing.container == _CONTAINER


async def test_a_routine_that_declares_nothing_is_settled_without_a_draw(db):
    """A round with no open parameter asks the model nothing: a routine that declares
    nothing needs nothing said, so its container is derived from its name alone.

    A failed framing files a skill with no parameters at all, so this shape is reachable —
    and putting a question to the binder that has no possible answer would end an
    apply turn that has everything it needs."""
    _upsert_skill(db, [])

    machine, model = await _cold_apply(db)

    assert machine.state() is ConversationState.APPLY
    assert not _drew_against(model, BIND_SKILL_SYSTEM_PROMPT)
    framing = machine.framing()
    assert framing is not None
    assert framing.bound_values() == {}
    assert framing.container == _SKILL
    assert db.memories.get(_SKILL) is not None


async def test_a_request_draw_the_words_fully_cover_settles_the_round(db):
    """The two answers DISAGREEING: the classifier says something is missing, the binder
    reads the same words and finds everything.

    Recorded behaviour, and the judgement call this beat leaves open: the binding is
    settled like any other complete one — the framing is recorded and the container built,
    so the acceptance that follows enters already framed — while the state stays where the
    draw put it, because redirecting request → apply would be the classifier's decision
    overruled by the binder, which is a design question rather than an implementation one.
    The turn asks in the words it always had, since a round with nothing missing has no
    shortfall to render."""
    _seed_skill(db)
    ask = _ASK_NAMING_THE_PAGE

    model = _model(state=f"STATE: request\nSKILL: {_SKILL}", binding=_BOUND)
    machine = _machine(db, model)
    await machine.advance(ask, message_id=_log(db, ask), run_id="run-request")

    assert machine.state() is ConversationState.REQUEST
    assert machine.shortfall() is None, "nothing is missing, so nothing is waited on"
    framing = machine.framing()
    assert framing is not None and framing.container == _CONTAINER
    assert db.memories.get(_CONTAINER) is not None


async def test_a_round_that_leaves_request_stops_waiting_on_anything(db):
    """The binding's lifecycle: it belongs to a round parked in request, so a move that
    leaves drops it.

    A user who says that routine was the wrong one is waiting on nothing that was ever
    true of the task they still want, and a binding kept past that point is state
    describing a negotiation that is over — which the next turn would reason from."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)

    machine = _machine(db, _model(state="STATE: elicit"))
    wrong = "no, that's not what I meant — I want the whole timetable"
    await machine.advance(wrong, message_id=_log(db, wrong), run_id="run-elicit")

    assert machine.state() is ConversationState.ELICIT
    assert machine.shortfall() is None
    latest = db.machine.latest_transition()
    assert latest is not None and latest.round_shortfall is None


# ── The bail takes the round's container AND its draft routine (#1896/#1902) ──


async def _bail(db: Database, message: str = "actually forget it, i don't need this") -> None:
    """One classified move to idle — the break-out edge every parked state carries."""
    machine = _machine(db, _model(state="STATE: idle", framing=""))
    await machine.advance(message, message_id=_log(db, message), run_id="run-bail")


async def test_a_round_that_bails_to_idle_takes_its_container_with_it(db):
    """The bail preserves nothing: the move drops the anchor, the framing and the binding,
    and the container that framing pointed at is ARCHIVED.

    Clearing the row alone would leave an inert collection named for a job nobody is doing,
    reachable by nothing — the orphan a learn round abandoned mid-teach left behind before
    this.  Archived rather than deleted, so it is a visible tombstone and the same job
    taught again revives it."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    assert db.memories.get(_CONTAINER) is not None

    await _bail(db)

    retired = db.memories.get(_CONTAINER)
    assert retired is not None and retired.archived is True
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.IDLE.value
    assert latest.anchor_message_id is None
    assert latest.skill_frame is None
    assert latest.round_shortfall is None


async def test_a_bail_archives_a_container_the_demonstration_already_wrote_into(db):
    """The one retirement here that is NOT guarded on emptiness, and the reason: what the
    demonstration wrote is the round's own intermediate state, which is exactly what a bail
    discards — not an exception to it.

    The emptiness guard belongs to the retirements that clear LITTER a mechanism left
    behind; this one clears a round the USER called off, and archiving keeps every entry
    readable either way."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    require_memory(db, _CONTAINER).write([EntryInput(key="day rate", content="$45")], author="chat")

    await _bail(db)

    retired = db.memories.get(_CONTAINER)
    assert retired is not None and retired.archived is True
    assert [entry.content for entry in require_memory(db, _CONTAINER).read_all()] == ["$45"]


async def test_retiring_a_container_twice_records_no_second_archive(db):
    """The retirement is a no-op on a container that is already retired, and on a round that
    was never framed at all — read off the row rather than attempted and swallowed.

    A second archive line would say the bail happened twice, which is a mutation ledger
    describing something that did not occur; the ledger is what the store map's "recent
    changes" block renders, so an invented event is one the model then reasons from."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    framing = RoundFraming.model_validate_json(_latest_frame(db))
    await _bail(db)
    recorded = [event.action for event in db.mutations.history(_CONTAINER, 10)]

    abandon_round_container(db, framing, run_id="run-bail-again")
    abandon_round_container(db, None, run_id="run-bail-again")

    assert [event.action for event in db.mutations.history(_CONTAINER, 10)] == recorded


async def test_a_bail_discards_the_draft_routine_the_round_registered(db):
    """The bail's other durable half (#1902): the routine the round registered at run end
    goes with the container it wrote into.

    A bail preserves nothing, and the draft is the round's intermediate state exactly as
    the container is — the difference being that the registry is AMBIENT, so a routine
    left standing for a job the user called off is read by every later turn.  Deleted
    rather than archived: the skill table is versionless and holds no archived flag."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    # What run-end extraction leaves behind: the routine filed under the name the round's
    # framing pinned (#1902's keyed write).
    _seed_skill(db)
    assert db.skills.get(_SKILL) is not None

    await _bail(db)

    assert db.skills.get(_SKILL) is None
    assert db.skills.list_all() == []


async def test_a_bail_after_a_correction_discards_the_one_routine_the_round_has(db):
    """The corrected-then-bailed shape, end to end: the round is taught, corrected, and
    then called off.

    Because the correction kept the round's identity (#1902), the correction's own
    extraction REPLACED the routine rather than adding one — so there is exactly one
    routine to discard, and the bail leaves the registry as it found it.  The fork this
    replaces left two: one per learn turn, and a bail could only ever name the last."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    _seed_skill(db)
    await _enter_learn(db, framing=_FRAMED_TOURS, message=_CORRECTION)
    # The corrected round re-registers under its ONE pinned name — a replacement, so the
    # registry still holds a single routine when the bail arrives.
    _seed_skill(db)
    assert [skill.name for skill in db.skills.list_all()] == [_SKILL]

    await _bail(db)

    assert db.skills.list_all() == []
    retired = db.memories.get(_CONTAINER)
    assert retired is not None and retired.archived is True


async def test_a_bail_takes_a_routine_the_round_re_taught_over(db):
    """The declared REACH of the bail ruling, pinned so the consequence is visible rather
    than latent: the discard names the round's pinned routine and nothing narrows it to the
    round's own registration.

    So a round re-teaching a routine the registry already held — which derives that
    routine's name, and whose run-end extraction therefore REPLACES its row — takes that
    routine with it when the user calls the round off, permanently (the table has no
    archived flag, unlike the container half).  Narrowing this to rows the round itself
    wrote would be a draft/canonical distinction, which the same ruling refuses; this test
    is here so the trade is a decision on the record rather than a surprise."""
    _seed_skill(db)  # a routine standing in the registry BEFORE the round opens
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    assert _latest_frame(db), "the round pinned the name that routine is already filed under"

    await _bail(db)

    assert db.skills.get(_SKILL) is None


async def test_discarding_a_draft_twice_is_a_no_op(db):
    """The discard is a read of an end state, not of a row: a round whose routine never
    reached the registry — one bailed before any turn of it completed — and a second call
    after the first both leave the registry untouched, with no error to absorb."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    framing = RoundFraming.model_validate_json(_latest_frame(db))

    await _bail(db)
    abandon_round_skill(db, framing)
    abandon_round_skill(db, None)

    assert db.skills.list_all() == []


async def test_the_post_apply_reset_leaves_the_round_s_routine_alone(db):
    """The landing a bail must never be confused with, on the registry side: a job the user
    accepted is a live mechanism, so the structural reset after apply drops the framing on
    its own row and the classified idle move that follows finds no round to end.

    Promotion is implicit SURVIVAL — there is no flag to set here.  The routine simply
    outlives the round, because only the round holding that framing could have replaced or
    discarded it."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    _seed_skill(db)
    accepting = _machine(db, _model(state=f"STATE: apply\nSKILL: {_SKILL}", framing=""))
    await accepting.advance("yes please", message_id=_log(db, "yes please"), run_id="run-apply")

    await _bail(db, "great, thanks")

    assert db.skills.get(_SKILL) is not None


async def test_a_bail_from_request_stops_waiting_and_leaves_nothing_behind(db):
    """The same landing from the other parked state: a round short of a value never built
    anything, so the bail has nothing to retire — and what it does drop is the binding it
    was waiting on, along with the ask it was anchored to."""
    _seed_two_parameter_skill(db)
    await _park_in_request(db)

    await _bail(db, "eh never mind, it's not important")

    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.IDLE.value
    assert latest.round_shortfall is None
    assert latest.anchor_message_id is None
    assert not [row for row in db.memories.list_all() if row.name.startswith("watch-rental")]


async def test_the_post_apply_reset_leaves_the_job_s_container_alone(db):
    """The landing a bail must never be confused with: apply has no out-edges, so the next
    message resets the machine structurally BEFORE anything is classified — and that row
    carries no framing, so the classified idle move after it finds no round to end.

    Which is what keeps a job the user has just set running out of this path: the container
    an apply turn configured is a live mechanism, not a round anybody walked away from."""
    anchor_id = _log(db, _ASK)
    _park_in_elicit(db, anchor_id)
    await _enter_learn(db)
    _seed_skill(db)
    accepting = _machine(db, _model(state=f"STATE: apply\nSKILL: {_SKILL}", framing=""))
    await accepting.advance("yes please", message_id=_log(db, "yes please"), run_id="run-apply")
    assert accepting.state() is ConversationState.APPLY

    await _bail(db, "great, thanks")

    kept = db.memories.get(_CONTAINER)
    assert kept is not None and kept.archived is False


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
    """The registry the apply draw needs a member of — the routine the round is about to
    learn, present so the gated edge is offered at all.

    It DECLARES its parameter, because that declaration is the binder's whole input
    (#1870): a cold apply fills what the routine already asks for, so a routine asking for
    nothing would make the binding vacuous and the derived name carry no values."""
    _upsert_skill(db, [SkillParameter(name="url", description="the rental page to read")])


def _seed_two_parameter_skill(db: Database) -> None:
    """The same routine declaring TWO parameters — the shape a partial binding needs: one
    the ask settles and one it does not, so a round can be short of something without being
    short of everything."""
    _upsert_skill(
        db,
        [
            SkillParameter(name="url", description="the rental page to read"),
            SkillParameter(name="keyword", description="which rate to look for"),
        ],
    )


def _upsert_skill(db: Database, parameters: list[SkillParameter]) -> None:
    """The registry row the seeds share — one routine, differing only in what it
    declares."""
    _upsert_named_skill(db, _SKILL, parameters)


def _upsert_named_skill(
    db: Database, name: str, parameters: list[SkillParameter] | None = None
) -> None:
    """One routine in the registry, under ``name`` — a second one exists so a round can
    come back naming a DIFFERENT routine, which is the case a carried binding must not
    survive.  Both declare the same parameter names on purpose: a carry that matched on
    names alone would look correct here."""
    db.skills.upsert(
        SkillDraft(
            name=name,
            intent="keep a rental page's current day rate up to date",
            description="keep a rental page's current day rate up to date",
            steps=[],
            parameters=parameters
            if parameters is not None
            else [
                SkillParameter(name="url", description="the rental page to read"),
                SkillParameter(name="keyword", description="which rate to look for"),
            ],
            source_run_id="run-learn",
        ),
        author="chat",
    )
