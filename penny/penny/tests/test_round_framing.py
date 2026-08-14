"""Framing a round at its ENTRY, and the container it runs into (#1868/#1870).

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
* a correction that keeps the job's identity finds the SAME container, and one that
  shifts it archives the empty container it replaces and builds the one it now needs;
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
* a round that taught nothing takes its empty container with it (#1839), and a container
  holding entries is never taken.

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
from penny.round_framing import RoundFramer, discard_round_container
from penny.tests.conftest import require_memory
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.micro_context import BIND_SKILL_SYSTEM_PROMPT, SKILL_FRAME_SYSTEM_PROMPT

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

# What ``derive_collection_name`` makes of each — written out rather than computed, so a
# change to the scheme fails HERE, naming the name that moved, instead of agreeing with
# itself.  The framer's minted name and the registry's name are the same string here, which
# is what lets one pair of constants stand for both entries.
_CONTAINER = "watch-rental-price-harborkayak-example-rentals"
_TOURS_CONTAINER = "watch-rental-price-harborkayak-example-tours"


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
    db.skills.upsert(
        SkillDraft(
            name=_SKILL,
            intent="keep a rental page's current day rate up to date",
            description="keep a rental page's current day rate up to date",
            steps=[],
            parameters=[
                SkillParameter(name="url", description="the rental page to read"),
                SkillParameter(name="keyword", description="which rate to look for"),
            ],
            source_run_id="run-learn",
        ),
        author="chat",
    )

    ask = "keep an eye on the weekend rate for me"
    machine = _machine(
        db,
        _model(
            state=f"STATE: apply\nSKILL: {_SKILL}",
            binding="MISSING url\nVALUE keyword: the weekend rate",
        ),
    )
    entered = await machine.advance(ask, message_id=_log(db, ask), run_id="run-request")

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
    back in request, with a freshly drawn shortfall.

    The same rule read a second time, and the same shape as the learn → learn
    re-demonstration edge: the round stays parked on the ask that opened it, so the next
    message is the retry.  Nothing is held between the two turns — the binder is handed the
    anchor ask plus the arriving message and derives the whole binding again, exactly as a
    cold apply does."""
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
    latest = db.machine.latest_transition()
    assert latest is not None
    assert latest.from_state == ConversationState.REQUEST.value
    assert latest.anchor_message_id == anchor_id, "the round is still parked on its own ask"
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
    db.skills.upsert(
        SkillDraft(
            name=_SKILL,
            intent="keep a rental page's current day rate up to date",
            description="keep a rental page's current day rate up to date",
            steps=[],
            parameters=[SkillParameter(name="url", description="the rental page to read")],
            source_run_id="run-learn",
        ),
        author="chat",
    )
