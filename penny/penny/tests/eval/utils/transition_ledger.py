"""The seeded ledger: a round's preceding turns, written the way production wrote them.

A transition case starts mid-conversation, so the turns before its beat have to already be in
the store -- the classifier draws that parked it, the chat steps, the asks and replies, the tool
calls and their results. These are the writers that put them there, in production's own shapes,
so a seeded history is indistinguishable from a lived one.

Shared machinery: the transition cases build every world with it, and so do the collector,
speakable-log-read and memory-story worlds.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

from penny.constants import ChatPromptType, PennyConstants, TransitionCause
from penny.conversation_machine import (
    ConversationState,
    RoundFraming,
    RoundProvenance,
    RoundShortfall,
    render_classifier_content,
)
from penny.database import Database
from penny.database.models import MemoryEntry
from penny.database.skill_store import parameters_from_json
from penny.database.skills import (
    DistillInput,
    SkillParameter,
)

# The SHIPPED container derivation, used as itself: a seeded round has to run into the
# container production would have built for it, and a fixture spelling that name out would
# be a second copy of the naming scheme, free to drift from the one jobs are identified by.
# The production draw-application, used as itself: a fixture skill has to be the SHAPE
# run-end extraction really produces, and re-implementing that mapping here would be a
# fixture that drifts from the pipeline it stands in for.  Both halves of the #1824
# split are applied by their own production function — ``_apply_leaf_labels`` for the
# labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's signature.
# ``attachment_names`` is the registry policy for what a routine can be attached to, read
# for the same reason: the scorer asks whether a learned routine HAS a destination, and
# that is the question extraction already answers when it decides which leaves to mark.
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    is_seeded_run,
    seeded_run_id,
)

# The agreed breadth for "the page the routine is pointed at", READ from where the framer
# suite declares it rather than restated here: what a page parameter may reasonably be
# called is one code-owner-agreed vocabulary, and two copies would drift into two
# contracts (the same rule ``ENACTING_TOOLS`` is read under).
# The listing this script is built on, and the enacting-tool set the elicitation
# contract IS — the calls that would mean she acted before being taught.  Both are read
# from the suite's shared fixtures rather than restated here: the passing-mention guard
# in ``test_chat_memory_stories.py`` asks the same question of a turn, and two copies of
# one policy are two contracts free to drift.
# The production tool-result framer, used as itself: a seeded ledger's tool turns have to
# read the way the loop really writes them, and a hand-written frame is a second copy of a
# format the model is shown every turn.
# The schedule's own render + grammar tokens, read from where the tool declares them: a
# stored rule renders back AS the copyable ``schedule`` input (#1857), so the advisory shows
# what she committed to in the form it was set, and the line/tag literals a rule is written
# with are that module's to define — a restated copy here would be a second contract.
# ``parse_schedule`` + ``render_reinstantiation_echo`` are read for the same reason on the
# seeding side: a seeded apply turn stores the rule the tool would have stored and echoes
# back what the tool would have echoed.
from penny.tools.micro_context import (
    SKILL_TAG,
    STATE_CLASSIFIER_SYSTEM_PROMPT,
    STATE_TAG,
    StateDrawOutcome,
)

_FAMILY = "state-transitions"

# The one call that stands a job up — named once, since three sections read it (the
# learn → apply scorer, the seeded apply turn's ledger, and the idle → apply scorer).
_SET_TOOL = "collection_set"

# The call every edge here must NOT make: going and looking instead of asking, teaching or
# setting up.  Named once for the same reason — five sections check for its absence.
_BROWSE_TOOL = "browse"


# ── Shared across the edges ───────────────────────────────────────────────────


def _park(
    db: Database,
    state: ConversationState,
    *,
    anchor_message_id: int | None = None,
    from_state: ConversationState = ConversationState.IDLE,
    run_id: str | None = None,
    message_id: int | None = None,
    skill_name: str | None = None,
    framing: RoundFraming | None = None,
    shortfall: RoundShortfall | None = None,
    provenance: RoundProvenance | None = None,
) -> None:
    """Leave the machine where the edge under test starts from, through the real
    store — a seeded transition row IS the machine's state (#1706), so nothing
    here fakes a state the production path couldn't be in.  The incoming message
    is still classified against it, so a case exercises the edge end to end.

    ``anchor_message_id`` is the instigating ask the parked round is anchored to
    — what the production anchor lifecycle stamps on the way in, and what the
    classifier renders as the task being worked on.

    ``from_state`` is where the move came FROM, defaulting to idle because that is
    where a round opens.  A case seeding a round several moves deep records each
    move it replays, so the ledger reads as the machine really walked it — the
    learn → apply cases park twice, once entering the round and once staying in
    it.

    ``run_id`` and ``message_id`` are the move's links back into the ledger and the
    conversation (#1846): production stamps both, and they are what make a seeded
    transition row indistinguishable from one the machine really recorded.

    ``skill_name`` is what a SKILL-GATED decision bound (apply / request) — the
    column the idle → apply cases read as half of the binding check, so a seeded
    apply move has to carry it exactly as a drawn one does.

    ``framing`` is the round's own framing (#1868/#1869) — settled by the move that
    ENTERED learn and carried by every move after it while the round is parked, so a
    seeded round records it on exactly the moves production records it on.  The apply turn
    reads it as the collection, the routine and the values it configures, which is why a
    seeded round without it is a round the turn under test cannot be answered against.

    ``shortfall`` is the round's PARTIAL binding (#1894) — the framing's other half, settled
    by a move that landed in REQUEST and carried while the round is parked there.  Recorded
    the same way and for the same reason: a later turn READS what the round is waiting on
    (the classifier renders it, and the next binder completes from it), so a seeded parked
    round without one is a round production can no longer produce.

    ``provenance`` is the round's third piece of entry state (#1902) — what the round's own
    registry write REPLACED, taken on the move that OPENS its teaching and carried the
    framing's way.  A seeded round that minted a routine has to carry it for the same reason
    it carries the framing: a bail READS it to decide what the registry is owed, so a round
    parked without one is a round production can no longer produce.

    Every move recorded here is a DECIDED draw: these are the moves the machine
    really made, and a classifier row with no outcome is the fail → stay shape,
    which is not what a seeded history of completed rounds is."""
    db.machine.record_transition(
        from_state=from_state.value,
        to_state=state.value,
        cause=TransitionCause.CLASSIFIER,
        anchor_message_id=anchor_message_id,
        outcome=StateDrawOutcome.DECIDED.value,
        run_id=run_id,
        message_id=message_id,
        skill_name=skill_name,
        skill_frame=framing.model_dump_json() if framing is not None else None,
        round_shortfall=shortfall.model_dump_json() if shortfall is not None else None,
        round_provenance=provenance.model_dump_json() if provenance is not None else None,
    )


def _structural_reset(db: Database) -> None:
    """The post-apply reset, exactly as production writes it: ``apply`` has no
    out-edges, so the NEXT incoming message settles the machine back to idle
    before anything is classified (``ConversationMachine._settle_structural``).

    No run, no anchor, and no message id — the reset is appended before the turn's
    message has one, and ``link_message`` back-fills by run id, which a structural
    row does not carry.  So the row production leaves is exactly this one."""
    db.machine.record_transition(
        from_state=ConversationState.APPLY.value,
        to_state=ConversationState.IDLE.value,
        cause=TransitionCause.STRUCTURAL,
    )


def _landed_state(db: Database) -> str | None:
    latest = db.machine.latest_transition()
    return latest.to_state if latest else None


def _log_ask(db: Database, content: str, case_id: str) -> int:
    """One incoming turn, with its id asserted where it is written.

    Every reply threads to the message it answers, so a turn with no id would leave the
    reply after it unreachable from the conversation window — a failure worth naming at
    the write rather than three probes later."""
    message_id = db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING,
        sender=TEST_SENDER,
        content=content,
    )
    assert message_id is not None, f"{case_id}: a seeded incoming turn must carry a message id"
    return message_id


def _log_reply(db: Database, content: str, *, answering: int) -> None:
    """One turn of Penny's, logged the way the REPLY PATH logs it: THREADED to the
    message it answers, and addressed to the user.

    The thread link is what puts a reply in the conversation window at all.
    ``get_messages_since`` — what ``_build_conversation`` reads — collects the incoming
    messages, then Penny's replies to THOSE messages by ``parent_id``, plus autonomous
    sends (``parent_id IS NULL`` AND addressed to the user).  An unthreaded, unaddressed
    row satisfies NEITHER leg, so it is in the record and out of the conversation: the
    window comes back all-user, and ``_build_conversation``'s same-role merge folds every
    seeded turn into ONE enormous user message.  Measured, that is what the first idle →
    apply run answered — nineteen turns stacked into a single message reading as a pile of
    unanswered requests, which is exactly what the model then tried to satisfy.

    Both stamps are production's (``_log_and_send``): the thread link, and the recipient
    the reply was addressed to.  They are NOT interchangeable — the recipient alone brings
    an unthreaded reply back through the AUTONOMOUS door, which reads the conversation
    right while claiming every one of Penny's turns was a mechanism speaking unprompted
    (verified: dropping the parent link leaves the window's contents identical, which is
    why the probe asserts the link and not only the contents).  With both, a reply matches
    the threaded leg and cannot match the autonomous one, so nothing is counted twice."""
    db.messages.log_message(
        direction=PennyConstants.MessageDirection.OUTGOING,
        sender=PennyConstants.MessageAuthor.PENNY,
        content=content,
        parent_id=answering,
        recipient=TEST_SENDER,
    )


def _entries_written_by_this_run(db: Database) -> list[MemoryEntry]:
    """Every ENTRY this run wrote, wherever it landed — created by it, or REWRITTEN by it.

    Scoring only collections the run CREATED assumed she always makes one — but
    "remember it" may reuse a name that already exists, and then the run's real
    writes are invisible while the reused collection's own seeded prompt and
    trigger read as things she did.  The run-id stamp says exactly what this run
    wrote (#1560), so ask that instead of inferring from newness.

    The whole entry, not its content alone: where in the entry a fact landed is a
    question about key/value semantics that is deliberately open (#1854), so the
    callers read both halves through ``_written_texts``.

    "This run" is now a stamp that is present AND not a seeded one (#1846): a seeded
    round's own entry carries the run that wrote it, exactly as production does, so
    "stamped at all" no longer distinguishes what this sample did from what it was
    handed."""
    written: list[MemoryEntry] = []
    for row in db.memories.list_all():
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        written += [entry for entry in entries if _written_by_this_run(entry)]
    return written


def _written_by_this_run(entry: MemoryEntry) -> bool:
    """Whether THIS sample put an entry's current value there — created by a live run, or
    last rewritten by one.

    BOTH stamps, not the creation one alone (#1900): a round that corrects an earlier one
    writes into a container that already holds an entry, and "remember that instead" under
    the key it used before is an UPDATE — the write gate refreshes the stored value in place
    and only ``last_written_by_run_id`` moves.  Reading creation alone would report that
    write as nothing having been written, which is a scorer bug reported as a model failure.
    Where nothing pre-exists — every other beat that reads this — the two stamps agree, so
    the reading there is unchanged."""
    stamps = (entry.created_by_run_id, entry.last_written_by_run_id)
    return any(stamp is not None and not is_seeded_run(stamp) for stamp in stamps)


def _written_texts(entries: list[MemoryEntry]) -> list[str]:
    """Both halves of every written entry — its KEY and its CONTENT.

    One shape, two customers: what the durable-write check matches the case's fact
    against, and what a rationale names when it missed.  A log entry has no key, so
    what it contributes is its content alone."""
    return [text for entry in entries for text in (entry.key, entry.content) if text]


def _pages_fetched(db: Database) -> list[MemoryEntry]:
    """Every page this run read — the browse-results log's recent window.

    One definition, two customers: the edge that must prove NOTHING was fetched
    (idle → elicit) and the elicit → learn seed's probe, which asserts its world
    STARTS from that same emptiness."""
    return require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).read_recent(
        window_seconds=3600, cap=None
    )


def _declared_parameters(db: Database, skill: str) -> list[SkillParameter]:
    """One registered routine's declared parameters, in declared order — the list the
    binder is handed and the list the request instruction renders from.

    Shared rather than per-beat: three sections ask what a routine declares — the short ask
    to say what it falls short of, the supply to say it completes it, and the derived
    container's name to put the values in the routine's own order — and a second reading of
    the registry is a second answer waiting to disagree."""
    routine = db.skills.get(skill)
    assert routine is not None, f"the routine {skill!r} must be registered"
    return parameters_from_json(routine.parameters)


def _leaf_at(arguments: dict, path: list):
    """The argument leaf a substitution's JSON path addresses — the step carries the
    call's verbatim arguments, so the DEMONSTRATED value is still in place."""
    node = arguments
    for part in path:
        node = node[part]
    return node


# ── The seeded ledger: the preceding turns, written the way production wrote them ─
#
# A case's entry state is everything the turns before it left behind — and most of what
# those turns left is the LEDGER, not the durable rows.  Seeding only the rows produced a
# world nothing can have produced: a collection whose creating run does not exist, a
# mutation line naming no run, an empty browse-results log after a round that read a page,
# and a hand-written description where the auto-create path writes one naming what it
# holds.  Measured against a real post-learn database, that is the whole difference — and
# a live model reasoning over the impoverished version correctly concluded it did not know
# how the page was built and should go and look.
#
# So the seeds write the promptlog too, through the same store the production path writes
# it through, in the same shapes: a classifier draw per turn (its own run id, as
# ``MicroContext`` mints one), the chat run's steps carrying the round's tool calls with
# their arguments and framed results, and the browse-extract call the browse spawned.
# Everything the round produced then cites those runs — the entry's write stamps, the
# collection's ``created_by_run_id`` (which is what puts the run id on the mutation line),
# the skill's ``source_run_id``, and both transition rows.
#
# The run ids are FIXED and carry the harness's seeded prefix, which is what every
# "what did the model do this sample" reader excludes them by (``live_prompts`` /
# ``_sample_prompt_rows``).  Without that exclusion the apply case's "no browse this turn"
# check would read the demonstrated round's browse as this turn's.


class _JourneyRuns(NamedTuple):
    """The run ids ONE journey's seeded turns are written under — a bundle rather
    than five module constants, because a world can hold more than one journey.

    A run id is the join key everything a turn produced cites (the collection, its
    entry, the skill, the transition rows), and it is how the ledger is READ back
    (``get_run_prompts``).  Five journeys sharing one set of ids would read as a
    single impossible run that browsed five pages and wrote five entries — so each
    journey mints its own, under the harness's seeded prefix either way."""

    elicit_draw: str
    elicit_turn: str
    learn_draw: str
    learn_turn: str
    browse_extract: str
    apply_draw: str
    apply_turn: str
    ack_draw: str
    ack_turn: str


def _journey_runs(journey: str) -> _JourneyRuns:
    """One journey's run ids, named after it — deterministic, so a probe can assert
    against them by name, and seeded-prefixed, so every "what did this sample do"
    reader excludes them."""
    return _JourneyRuns(
        elicit_draw=seeded_run_id(f"{journey}-elicit-draw"),
        elicit_turn=seeded_run_id(f"{journey}-elicit-turn"),
        learn_draw=seeded_run_id(f"{journey}-learn-draw"),
        learn_turn=seeded_run_id(f"{journey}-learn-turn"),
        browse_extract=seeded_run_id(f"{journey}-learn-browse-extract"),
        apply_draw=seeded_run_id(f"{journey}-apply-draw"),
        apply_turn=seeded_run_id(f"{journey}-apply-turn"),
        ack_draw=seeded_run_id(f"{journey}-ack-draw"),
        ack_turn=seeded_run_id(f"{journey}-ack-turn"),
    )


# The bundle a case with no journey of its own writes under — the elicit → learn
# cases, which seed exactly one round each and have nothing to distinguish it from.
_DEFAULT_RUNS = _journey_runs("round")

# The model the seeded rows name — the same one the samples themselves run against, read
# the way the harness's own config reads it, so a seeded row and a live row cite one model.
_SEEDED_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")

# The call ids the seeded round's two steps are keyed by.  Production ids come from the
# backend; what matters structurally is that each call's result turn carries ITS id, which
# is how a run's calls pair with their outcomes.
_BROWSE_CALL_ID = "call-seeded-browse"
_WRITE_CALL_ID = "call-seeded-write"


def _seeded_response(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    """One model response in the shape ``raw.model_dump()`` persists — the envelope every
    ledger reader walks (``choices[].message``), with ``tool_calls`` present only when the
    draw made calls, exactly as a text reply's dump has it."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"model": _SEEDED_MODEL, "choices": [{"index": 0, "message": message}]}


def _wire_tool_call(call_id: str, step: DistillInput) -> dict:
    """One tool call as the wire stores it.  ``arguments`` is a JSON STRING — the shape
    ``LoggedToolCall.from_function`` decodes, and the one a dict would silently lose (it
    returns an empty mapping for a non-string, so the call would render argument-less)."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": step.tool, "arguments": json.dumps(step.arguments)},
    }


def _tool_result_turn(call_id: str, step: DistillInput) -> dict:
    """The tool turn the loop appended for one call — the framed result keyed by the call's
    own id, carrying the framework's structural success stamp, which is how a run's calls
    pair with their outcomes rather than by position."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": step.result,
        PennyConstants.TOOL_RESULT_SUCCESS_KEY: True,
    }


def _drawn_state(state: ConversationState, *, skill: str | None = None) -> str:
    """The classifier's own output for a DECIDED draw — the tagged state line, plus the
    second ``SKILL:`` line a skill-gated decision carries (#1706).  Written with the
    module's own tags so a seeded draw reads back through the same parse a live one does."""
    line = f"{STATE_TAG} {state.value}"
    return line if skill is None else f"{line}\n{SKILL_TAG} {skill}"


def _log_classifier_draw(db: Database, *, run_id: str, snapshot, message: str, drawn: str) -> None:
    """The classifier's own row for a seeded turn — its real system prompt over its real
    rendered slice, under its OWN run id (production mints one per draw; the turn's run id
    reaches the classifier only through the transition row)."""
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=[
            {"role": "system", "content": STATE_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": render_classifier_content(snapshot, message)},
        ],
        response=_seeded_response(drawn),
        agent_name=PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
        prompt_type=PennyConstants.STATE_CLASSIFIER_PROMPT_TYPE,
        run_id=run_id,
        run_target=PennyConstants.CHAT_AGENT_NAME,
    )


def _log_chat_step(db: Database, *, run_id: str, messages: list[dict], response: dict) -> None:
    """One step of a seeded chat run — the accumulated conversation as it stood when the
    call was made, and what the model returned."""
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=messages,
        response=response,
        agent_name=PennyConstants.CHAT_AGENT_NAME,
        prompt_type=ChatPromptType.USER_MESSAGE,
        run_id=run_id,
    )
