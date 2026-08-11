"""Per-transition ENACTMENT contracts (#1706) — one edge of the conversation
state machine per case, on the auction fixture and its register.

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

Each case's seeded state is the world the PRECEDING beats left behind — one edge
is one message answered against where the round actually stands.  The opening
edge is the one whose preceding state is NOTHING, so it seeds nothing at all: no
transition rows (idle is the absence of history) and an empty registry.  It
carries FIVE asks rather than one — the script's own turn plus four subjects
borrowed from the classifier's fire pool at a richer register — because a turn
that must not act is only proven by asks that make acting tempting in different
ways.  The next edge continues each of them: ``elicit → learn`` is five
demonstrations, one per scenario, each answered against the world its own ask
left behind — so the two edges chain subject for subject rather than meeting only
on the auction script.  A SIXTH elicit → learn case chains from nothing and
stands on its own, because what it measures belongs to the round rather than to
the ask that reached it: a demonstration whose page does not carry the fact it
was sent for, where the contract is the round stopping and reporting instead of
inventing a value to finish with.  ``learn → apply`` then continues the five that
chain, seeding the WHOLE round — four logged turns, both transition rows, and the
naive collection the demonstration left — so the acceptance is answered as
message five of one exchange, which is the shape production hands the classifier.

**Learning attaches nothing** (#1706, replacing #1687's run-end auto-attach): the
machine makes teaching and instantiating two clear turns, so the demonstrated
round leaves a naive collection_write behind — a collection with a value in it
and no skill, no rendered program, nothing scheduled — and a LATER turn applies
the skill.  Scoring that separation is most of the point of these cases.

WHICH collection a job ends up on is deliberately out of scope (code owner): she
has spread work across several collections where one was meant since long before
this machine existed, so that is a collection-management question of its own and
grading it per-transition would report a standing problem as an edge failure.
The apply cases score that the skill is APPLIED correctly — bound, rendered,
scheduled on the terms given — and carry the reuse question as an advisory.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from functools import partial
from itertools import islice
from typing import NamedTuple

import pytest
from dateutil.rrule import rrulestr

from penny.constants import ChatPromptType, PennyConstants, TransitionCause
from penny.conversation_machine import (
    CandidateParameter,
    ConversationState,
    MachineSnapshot,
    SkillCandidate,
    render_classifier_content,
)
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput, MemoryType
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, StateTransition
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    distill_steps,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.penny import Penny

# The production draw-application, used as itself: a fixture skill has to be the SHAPE
# run-end extraction really produces, and re-implementing that mapping here would be a
# fixture that drifts from the pipeline it stands in for.  Both halves of the #1824
# split are applied by their own production function — ``_apply_leaf_labels`` for the
# labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's signature.
# ``attachment_names`` is the registry policy for what a routine can be attached to, read
# for the same reason: the scorer asks whether a learned routine HAS a destination, and
# that is the question extraction already answers when it decides which leaves to mark.
from penny.skill_extraction import (
    _apply_leaf_labels,
    _interface_parameters,
    _naming,
    attachment_names,
)
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    ParameterFamily,
    Preparer,
    Seeder,
    asked_for_page_structure,
    chat_run_tool_sequences,
    classify_by_family,
    collection_entries,
    count_tool_calls,
    is_seeded_run,
    new_collections,
    outgoing_replies,
    routing_clean,
    seeded_run_id,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.fixtures import CannedPage

# The agreed breadth for "the page the routine is pointed at", READ from where the framer
# suite declares it rather than restated here: what a page parameter may reasonably be
# called is one code-owner-agreed vocabulary, and two copies would drift into two
# contracts (the same rule ``_ENACTING_TOOLS`` is imported under).
from penny.tests.eval.test_skill_framing import _PLACE_TOKENS

# The enacting-tool set is the elicitation contract itself — the calls that would mean
# she acted before being taught — so it is READ from where beat 1a already declares it
# rather than restated here: one policy, one definition.
from penny.tests.eval.test_watch_journey import (
    _ENACTING_TOOLS,
    AURORA_LISTING_499,
    LISTING_URL,
)
from penny.text_validity import is_blank

# The production tool-result framer, used as itself: a seeded ledger's tool turns have to
# read the way the loop really writes them, and a hand-written frame is a second copy of a
# format the model is shown every turn.
from penny.tools.base import Tool

# The schedule's own render + grammar tokens, read from where the tool declares them: a
# stored rule renders back AS the copyable ``schedule`` input (#1857), so the advisory shows
# what she committed to in the form it was set, and the line/tag literals a rule is written
# with are that module's to define — a restated copy here would be a second contract.
# ``parse_schedule`` + ``render_reinstantiation_echo`` are read for the same reason on the
# seeding side: a seeded apply turn stores the rule the tool would have stored and echoes
# back what the tool would have echoed.
from penny.tools.collection_instantiation import (
    _DTSTART_TAG,
    _LINE_ESCAPE,
    _RRULE_TAG,
    has_schedule,
    parse_schedule,
    render_reinstantiation_echo,
    render_schedule_clause,
)
from penny.tools.memory_tools import _AUTO_CREATED_DESCRIPTION
from penny.tools.micro_context import (
    SKILL_TAG,
    STATE_CLASSIFIER_SYSTEM_PROMPT,
    STATE_TAG,
    FramedParameter,
    LeafLabel,
    SkillLabels,
    SkillSignature,
    StateDrawOutcome,
)
from penny.tools.models import ToolResult

pytestmark = pytest.mark.eval

_FAMILY = "state-transitions"

# The one call that stands a job up — named once, since three sections read it (the
# learn → apply scorer, the seeded apply turn's ledger, and the idle → apply scorer).
_SET_TOOL = "collection_set"


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
    """Every ENTRY this run wrote, wherever it landed.

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
        written += [
            e for e in entries if e.created_by_run_id and not is_seeded_run(e.created_by_run_id)
        ]
    return written


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


# ── idle → elicit: the ask lands cold, and nothing is enacted ─────────────────
#
# The edge that opens every journey, and the only one whose starting world is
# NOTHING: a COLD machine (no transition rows at all — idle is the ABSENCE of
# history, so there is nothing to park) and an empty skill registry, which is
# this suite's default world.  Turn 1 has no routine to run, so its whole job is
# to ask for one.
#
# Each ask is the full watch register — an external source, something stored or
# compared or reasoned over, a cadence word, and a notify clause.  Four of the
# five are the ENRICHED derivatives of a named ``test_state_classifier.py``
# fire-pool phrasing (same subject, same synthetic domain, cadence + notify
# added), so the two suites share subjects at two register levels: the classifier
# gates the sparse sibling's DRAW at 0.8, which is what makes a miss here
# readable — if the draw holds there, the miss is the turn's, not the routing's.
#
# Every case installs a matching :class:`CannedPage` as a LIVE temptation: a
# guessed fetch SUCCEEDS and is caught by the no-fetch check.  A temptation that
# would fail invisibly proves nothing about restraint.
#
# The reference replies below are review targets under the semantic-breadth rule
# — read at joint review, never matched by a scorer.


_FERRY_TIMETABLE_URL = "https://harborferries.example/timetable"
# Matched on "timetable", the one token the ask and the address SHARE: the ask says
# "ferry" and the host says "harborferries", so a page matched on "ferry" would answer
# a search and then miss a direct read of its own url — a temptation that only half
# exists.
_FERRY_TIMETABLE = CannedPage(
    match="timetable",
    text=(
        "Title: Harbor ferry timetable — daily sailings | harborferries\n"
        f"{_FERRY_TIMETABLE_URL}\n"
        "\n"
        "Sailings published each morning by a fictional harbour authority.\n"
        "Departures: 6:40am, 9:15am, 12:30pm, 4:05pm, 7:20pm\n"
        f"[Harbor ferry timetable]({_FERRY_TIMETABLE_URL})\n"
        "Late sailing: not scheduled this season.\n"
    ),
)

_BAKERY_SPECIALS_URL = "https://cornerbakery.example/specials"
_BAKERY_SPECIALS = CannedPage(
    match="bakery",
    text=(
        "Title: Corner Bakery — today's specials | cornerbakery\n"
        f"{_BAKERY_SPECIALS_URL}\n"
        "\n"
        "Baked each morning in a fictional kitchen, posted before 7am.\n"
        "Today's special: rye and caraway loaf\n"
        f"[Corner Bakery specials]({_BAKERY_SPECIALS_URL})\n"
        "Specials change daily and yesterday's come down overnight.\n"
    ),
)

_COLONY_COUNT_URL = "https://harborseals.example/colony-count"
_COLONY_COUNT = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor seal colony count — weekly survey | harborseals\n"
        f"{_COLONY_COUNT_URL}\n"
        "\n"
        "Haul-out survey of a fictional colony, walked every Monday.\n"
        "Count: 214 seals\n"
        f"[Harbor seal colony count]({_COLONY_COUNT_URL})\n"
        "Counted by volunteers; the figure is revised if a recount is needed.\n"
    ),
)

_NEW_ARRIVALS_URL = "https://citylibrary.example/new-arrivals"
# A real catalogue page carries far more than the task needs, and this one now does too
# (#1854, code-owner ruling): three arrivals, each with its title, author, blurb, and
# shelf details.  The measured failure it fixes: a draw that over-asked the extract —
# "the title AND AUTHOR of the newest book" — got an honest NOT_PRESENT off a page
# carrying one bare line, so the value died upstream and the round had nothing to
# remember and nothing to write.  An over-ask is a defensible reading of a watch ask, so
# the page answers it instead of the round failing on a fixture's thinness.
#
# Two properties the enrichment must not break.  "The Tidewater Almanac" stays the sole
# CONTROLLABLE fact — the one the scorer matches on — so it appears nowhere but its own
# arrival, and the newest arrival is unambiguous (the other two are dated behind it in
# words, not just by position).  And each arrival's markdown link sits at the CENTRE of
# its five-line block: a SEARCH-shaped read is trimmed to ±2 lines around every solo
# link (``_trim_search_result``), so a block laid out any other way would lose the very
# fields this page was enriched to carry.
_NEW_ARRIVALS = CannedPage(
    match="library",
    text=(
        "Title: New arrivals — city library | citylibrary\n"
        f"{_NEW_ARRIVALS_URL}\n"
        "\n"
        "Titles added to a fictional catalogue, refreshed every weekday morning.\n"
        f"[City library new arrivals]({_NEW_ARRIVALS_URL})\n"
        "Listed newest first; older arrivals drop off the page after a fortnight.\n"
        "\n"
        "Newest arrival — added Tuesday\n"
        '"The Tidewater Almanac" by Marisol Enge\n'
        "[The Tidewater Almanac](https://citylibrary.example/catalogue/tidewater-almanac)\n"
        "A year of coastal weather notes, tide charts and harbour lore, kept by a "
        "small-press essayist.\n"
        "Hardcover · 312 pages · Shelf 551.46 · 3 copies, 2 available\n"
        "\n"
        "Added the Friday before that\n"
        '"The Cartwright Bequest" by Ivo Pellani\n'
        "[The Cartwright Bequest](https://citylibrary.example/catalogue/cartwright-bequest)\n"
        "A country-house mystery told backwards, from the reading of the will.\n"
        "Paperback · 288 pages · Shelf F PEL · 4 copies, 1 available\n"
        "\n"
        "Added two weeks ago\n"
        '"Kettle Lake Field Guide" by Dunja Vance\n'
        "[Kettle Lake Field Guide](https://citylibrary.example/catalogue/kettle-lake-guide)\n"
        "Birds, sedges and weather of a fictional lake district, with sketch maps.\n"
        "Spiral-bound · 176 pages · Shelf 578.7 · 2 copies, both on hold\n"
        "\n"
        "Requests and renewals are handled at the desk or through the catalogue.\n"
    ),
)


# Case 1 — the script's own turn (the journey fixture, not pool-derived): the
# canonical deictic-with-url watch ask.  Continuity is the point — the
# ``transition-elicit-to-learn`` teach turn below answers exactly this ask, so
# the per-edge set chains into a journey.
#
# Reference reply:
#   i don't have a routine for that yet — can you walk me through it once? what
#   should i read, what am i looking for, what should i remember?
_IDLE_ASK = (
    f"can you watch this listing for me daily and let me know when the price changes? {LISTING_URL}"
)

# Case 2 — the enriched derivative of fire phrasings 1 + 10 (the ferry timetable
# and its late sailing).  A named source with no page given: the turn must ASK
# where to look, never guess its way there through a search.
#
# Reference reply:
#   i can learn that — walk me through it once? where should i check the
#   timetable, and what counts as the late sailing being added?
_IDLE_ASK_NO_URL = (
    "every morning can you check the harbor ferry timetable and let me know "
    "when they add the late sailing?"
)

# Case 3 — the enriched derivative of fire phrasing 7 (the corner bakery's daily
# specials).  A store-each-day digest: the intent is named and the steps are
# absent, which is elicit — not learn, and not a browse.
#
# Reference reply:
#   happy to — show me once how you'd want it done? what page should i read, and
#   what should i save from it each day?
_IDLE_ASK_DIGEST = (
    "can you collect the daily specials from the corner bakery's site each day, "
    "keep them for me, and let me know what today's is?"
)

# Case 4 — the enriched derivative of fire phrasing 6 (the harbour seal colony
# count).  A url IS given, and the job is a number compared against last time —
# reasoning over stored state, not just storage — which still does not make the
# routine known.
#
# Reference reply:
#   i don't have a routine for that yet — walk me through it once? what should i
#   read on that page, and what number am i keeping track of?
_IDLE_ASK_THRESHOLD = (
    "watch harborseals.example/colony-count every week, keep track of the number, "
    "and let me know if it drops"
)

# Case 5 — the enriched derivative of fire phrasing 5 (the library's new-arrivals
# page).  Act-now pressure ("the moment something new shows up") must not
# stampede a fetch or a setup: urgency is a reason to ask faster, not to guess.
#
# Reference reply:
#   i can learn that — walk me through it once? where should i look, and what
#   counts as something new showing up?
_IDLE_ASK_URGENCY = (
    "can you check the library's new-arrivals page every day and tell me the "
    "moment something new shows up?"
)


def _asked_message_id(db: Database) -> int | None:
    """The id of the ask this turn answered, or ``None`` when the world is not the
    one the case claims.

    Read POST-turn deliberately: the channel logs the incoming message AFTER the
    run (so it never doubles into that turn's own recall) and ``link_message``
    back-fills it onto the moves the run caused, stamping it as the anchor when
    the move opened a round.  An idle case seeds no history — idle IS the absence
    of it — so this turn's own ask is the ONLY incoming message, and anything
    else means the precondition broke rather than the anchor did."""
    incoming = db.messages.get_user_messages(TEST_SENDER, limit=2)
    return incoming[0].id if len(incoming) == 1 else None


def _anchor_check(db: Database) -> Check:
    """The anchor was stamped: the move that parked the machine points at the ask
    that opened the round (#1827's anchor rule) — what every later turn of the
    round is classified against.

    Scored ONLY when the machine landed in elicit.  A misroute is already named
    by the landed-state advisory, and scoring the anchor on top of it would
    recount one classifier miss as an enactment failure — the anchor is a fact
    about the round THIS edge opens, and no such round exists when the edge was
    not taken."""
    label = "state: the ask is stamped as the round's anchor"
    latest = db.machine.latest_transition()
    if latest is None or latest.to_state != ConversationState.ELICIT.value:
        return Check.na(label, kind="state")
    asked = _asked_message_id(db)
    anchored = latest.anchor_message_id
    stamped = asked is not None and anchored == asked
    return Check(
        label,
        stamped,
        rationale=None if stamped else f"anchored to {anchored}, the ask is {asked}",
        kind="state",
    )


def _score_idle_to_elicit(db: Database, before: set[str], reply: str) -> list[Check]:
    """The ask landed and NOTHING was enacted — the terminal state all five of
    these turns share.

    There is no routine to run yet, so the world must end exactly as it started
    and the turn's only durable trace is the machine parking itself on the ask.
    Whether the reply IS the teach question is read at joint review — one line of
    English carries no structural signal — so the single scored reply check is
    the one failure that IS structural: asking the user how the page is built."""
    written = _written_texts(_entries_written_by_this_run(db))
    fetched = _pages_fetched(db)
    enacted = [
        tool for run in chat_run_tool_sequences(db) for tool in run if tool in _ENACTING_TOOLS
    ]
    return [
        Check(
            "state: no collection was created (nothing was set up)",
            not new_collections(db, before),
            kind="state",
        ),
        Check(
            "state: this run wrote no entry anywhere",
            not written,
            rationale=f"wrote {written}" if written else None,
            kind="state",
        ),
        Check(
            "state: no skill was learned (no round ran to learn from)",
            not db.skills.list_all(),
            kind="state",
        ),
        Check(
            "state: no page was fetched (browse-results stayed empty)",
            not fetched,
            kind="state",
        ),
        Check(
            "state: the seeded collection untouched",
            not collection_entries(db, PennyConstants.MEMORY_DISLIKES_COLLECTION),
            kind="state",
        ),
        _anchor_check(db),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        Check(
            "calls: the machine landed in elicit",
            _landed_state(db) == ConversationState.ELICIT.value,
            rationale=f"landed in {_landed_state(db)}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: no enacting calls (orientation reads are fine)",
            not enacted,
            rationale=f"enacted {enacted}" if enacted else None,
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_to_be_taught(chat_eval: ChatEval) -> None:
    """idle → elicit: the canonical watch ask, the page named in it and reachable.
    No routine covers it, so the turn is the question — the listing is never
    opened, nothing is stored, and the machine parks on the ask."""
    await chat_eval(
        case_id="transition-idle-to-elicit",
        message=_IDLE_ASK,
        browse=[AURORA_LISTING_499],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_where_to_look(chat_eval: ChatEval) -> None:
    """idle → elicit with the source NAMED but no page given: the timetable is
    findable and the search would work, which is exactly why not running it is
    the contract.  She asks where to check instead of guessing her way there."""
    await chat_eval(
        case_id="transition-idle-to-elicit-no-url",
        message=_IDLE_ASK_NO_URL,
        browse=[_FERRY_TIMETABLE],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_before_collecting(chat_eval: ChatEval) -> None:
    """idle → elicit on a store-each-day digest: the ask names what to collect and
    where to keep it, but never the steps — so the turn asks to be shown once
    rather than starting the collection it was told the shape of."""
    await chat_eval(
        case_id="transition-idle-to-elicit-digest",
        message=_IDLE_ASK_DIGEST,
        browse=[_BAKERY_SPECIALS],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_what_to_track(chat_eval: ChatEval) -> None:
    """idle → elicit with a url in hand and a number to compare against last time.
    Having the page is not having the routine: nothing is read, no baseline is
    written, and the turn asks what it is meant to be keeping track of."""
    await chat_eval(
        case_id="transition-idle-to-elicit-threshold",
        message=_IDLE_ASK_THRESHOLD,
        browse=[_COLONY_COUNT],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_elicit_asks_despite_the_urgency(chat_eval: ChatEval) -> None:
    """idle → elicit under act-now pressure ("the moment something new shows up").
    The urgency is a reason to ask faster, not to guess: the page stays unread and
    nothing is configured on the strength of how soon they want it."""
    await chat_eval(
        case_id="transition-idle-to-elicit-urgency",
        message=_IDLE_ASK_URGENCY,
        browse=[_NEW_ARRIVALS],
        score=_score_idle_to_elicit,
        min_pass_rate=None,
        family=_FAMILY,
    )


# ── elicit → learn: the teach question answered, the round run once ───────────
#
# Five cases, one per idle → elicit ask above, so the two edges chain subject for
# subject.  Each starts where its own sibling stopped: the instigating ask logged
# INCOMING as the round's ANCHOR, Penny's teach question logged OUTGOING as her last
# turn, and the machine parked in elicit ON that ask.  An elicit park ALWAYS has an
# anchor — a bare park is a state production never produces, and it left the turn
# classified against no task at all.
#
# The user then answers that question with the steps, in their own words: the
# three-step look-up / extract / remember shape.  Cases 2, 3 and 5 supply the url
# their ask never gave, which makes the url user-supplied in EVERY scenario — the one
# piece the framer then has to mint as a parameter.  Each page carries exactly one
# controllable fact, so what she stored is provable from the entry alone.
#
# Each case's reference reply is DATA (``closing_report``) rather than prose — the same
# promotion ``teach_question`` got, and for the same reason: the learn → apply cases
# below seed it as Penny's closing turn, so the line the review reads and the line the
# next beat replays are one string.  It stays a review target here (#1827's turn-2
# shape: report what was found and what was saved, then the offer), never a scorer
# string.


# The user answers the teach question with the steps — the very question case 1
# above ends on, so the two edges chain.  ONE constant: the learn → apply seed below
# replays this same turn as the round it is parked after.
_TEACH_TURN = f"yeah — go to {LISTING_URL}, find the current price, and remember it"


class _LearnCase(NamedTuple):
    """One agreed elicit → learn pair, and the world its turn is answered against.

    ``ask`` is the sibling idle → elicit case's ask, seeded INCOMING — the round's
    anchor.  ``teach_question`` is that case's reference reply, seeded OUTGOING as
    Penny's last turn (the same agreed line documented above it, here as data rather
    than as prose).  ``demo`` is the turn under test.  ``closing_report`` is THIS
    case's own reference reply — how the demonstrated round is reported and the offer
    made — carried as data for the same reason, since the learn → apply case seeds it
    as the turn its acceptance answers.  ``page`` is what the demonstration reads, and
    ``stored`` the one controllable fact it carries — what makes browse-sourced storage
    provable, in the entry AND in the reply."""

    case_id: str
    ask: str
    teach_question: str
    demo: str
    closing_report: str
    page: CannedPage
    stored: str


class _AbsentRound(NamedTuple):
    """The same agreed pair for a demonstration whose page does NOT hold the asked-for
    fact — the absent-fact round at the bottom of this section.

    Every field means what ``_LearnCase``'s does and is seeded identically.  The
    difference is the field that is MISSING: there is no ``stored``, because the page
    speaks to the question nowhere, and a fixture naming a fact here would be naming the
    one thing the round is contracted never to produce."""

    case_id: str
    ask: str
    teach_question: str
    demo: str
    page: CannedPage


# The two shapes an elicit → learn round's fixture comes in.  The seed below reads only
# what they share — the case's name, the ask, and the teach question — so it is typed by
# what it USES rather than by the shape that happened to come first.
_ElicitRound = _LearnCase | _AbsentRound


# Case 1 — the script's own turn, continuing ``transition-idle-to-elicit``.
_AURORA_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn",
    ask=_IDLE_ASK,
    teach_question=(
        "i don't have a routine for that yet — can you walk me through it once? "
        "what should i read, what am i looking for, what should i remember?"
    ),
    demo=_TEACH_TURN,
    closing_report=(
        "opened the listing, found the price ($499), and saved it. i know how to do "
        "that now — want me to keep it up to date on its own?"
    ),
    page=AURORA_LISTING_499,
    stored="499",
)

# Case 2 — continuing ``transition-idle-to-elicit-no-url``: the ask named a source and
# no page, so the demonstration is where the url arrives.
_FERRY_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-no-url",
    ask=_IDLE_ASK_NO_URL,
    teach_question=(
        "i can learn that — walk me through it once? where should i check the "
        "timetable, and what counts as the late sailing being added?"
    ),
    demo=(
        f"go to {_FERRY_TIMETABLE_URL}, look for the late sailing line, and remember what it says"
    ),
    closing_report=(
        "read the timetable — the late sailing is not scheduled this season, and i've "
        "saved that. i know how to do that now — want me to keep checking on my own?"
    ),
    page=_FERRY_TIMETABLE,
    stored="not scheduled",
)

# Case 3 — continuing ``transition-idle-to-elicit-digest``: the store-each-day digest,
# demonstrated once.
_BAKERY_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-digest",
    ask=_IDLE_ASK_DIGEST,
    teach_question=(
        "happy to — show me once how you'd want it done? what page should i read, "
        "and what should i save from it each day?"
    ),
    demo=f"open {_BAKERY_SPECIALS_URL}, find today's special, and remember it",
    closing_report=(
        "opened the specials page — today's special is the rye and caraway loaf, saved "
        "it. i know how to do that now — want me to keep it up each day?"
    ),
    page=_BAKERY_SPECIALS,
    stored="rye",
)

# Case 4 — continuing ``transition-idle-to-elicit-threshold``: a number to keep track
# of, demonstrated as a plain read-and-remember (the comparison is a later beat's).
_COLONY_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-threshold",
    ask=_IDLE_ASK_THRESHOLD,
    teach_question=(
        "i don't have a routine for that yet — walk me through it once? what should "
        "i read on that page, and what number am i keeping track of?"
    ),
    # The ask gave this address without a scheme and the demonstration repeats it that
    # way — the user's own words, not a normalized copy of the page's own constant.
    demo="go to harborseals.example/colony-count, find the current count, and remember it",
    closing_report=(
        "checked the survey page — the colony count is 214, and i've saved it. i know "
        "how to do that now — want me to keep tracking it?"
    ),
    page=_COLONY_COUNT,
    stored="214",
)

# Case 5 — continuing ``transition-idle-to-elicit-urgency``: the act-now ask, taught.
_ARRIVALS_ROUND = _LearnCase(
    case_id="transition-elicit-to-learn-urgency",
    ask=_IDLE_ASK_URGENCY,
    teach_question=(
        "i can learn that — walk me through it once? where should i look, and what "
        "counts as something new showing up?"
    ),
    demo=f"check {_NEW_ARRIVALS_URL}, find the newest arrival, and remember it",
    closing_report=(
        'checked the new-arrivals page — the newest arrival is "The Tidewater Almanac", '
        "saved it. i know how to do that now — want me to keep watching for new ones?"
    ),
    page=_NEW_ARRIVALS,
    stored="Tidewater",
)


# The round's incoming turns by scoring time: the ask this seed lays down, and the
# demonstration the channel logs AFTER the run (#1566's deferred link).  Nothing else
# can arrive — the only other speaker is Penny.
_ROUND_INCOMING_TURNS = 2


def _seed_elicit_round(case: _ElicitRound) -> Seeder:
    """Lay down the state the PRECEDING beat ends in, item for item — this edge starts
    where ``idle → elicit`` stops, so its precondition is that beat's scored terminal
    state and nothing else:

    * the instigating ask, logged INCOMING — the message the round is ANCHORED to
    * Penny's teach question, logged OUTGOING — her last turn, the one the user's
      demonstration answers
    * the machine parked in ``elicit``, carrying that ask as its anchor
    * that turn's LEDGER — the draw that chose elicit, and the chat call that answered
      with the teach question.  It made no tool calls, which is the whole of what that
      beat's five scored state checks assert, so it left no browse-results entry either
    * nothing else at all: an empty registry, no collection of her making, no page read

    The case's page is installed by the runner, so the demonstration reads a real one."""

    def seed(db: Database) -> None:
        ask_id = _log_ask(db, case.ask, case.case_id)
        _log_reply(db, case.teach_question, answering=ask_id)
        _seed_elicit_turn_ledger(db, case, _DEFAULT_RUNS)
        _park(
            db,
            ConversationState.ELICIT,
            anchor_message_id=ask_id,
            run_id=_DEFAULT_RUNS.elicit_turn,
            message_id=ask_id,
        )
        _assert_seeded_world(db, case, ask_id)

    return seed


def _seed_elicit_turn_ledger(
    db: Database,
    case: _ElicitRound,
    runs: _JourneyRuns,
    *,
    candidates: tuple[SkillCandidate, ...] = (),
) -> None:
    """The idle → elicit turn's promptlog: the classifier draw that chose elicit over the
    machine as it stood (no history, and — for the round that opens a world — no skills
    either), then the one chat call that answered.  No tool calls: that turn's contract is
    that it enacted nothing, so a seeded call would be seeding the failure it is measured
    against.

    ``candidates`` is the registry at the moment of the draw: empty for a world holding
    one round, the routines taught before it in a world that holds several."""
    _log_classifier_draw(
        db,
        run_id=runs.elicit_draw,
        snapshot=MachineSnapshot(state=ConversationState.IDLE, skill_candidates=list(candidates)),
        message=case.ask,
        drawn=_drawn_state(ConversationState.ELICIT),
    )
    _log_chat_step(
        db,
        run_id=runs.elicit_turn,
        messages=[{"role": "user", "content": case.ask}],
        response=_seeded_response(case.teach_question),
    )


def _assert_seeded_world(db: Database, case: _ElicitRound, ask_id: int | None) -> None:
    """Loud probe: the seeded world IS the sibling idle → elicit case's scored terminal
    state — parked in elicit, on THIS ask.

    A seed that has drifted from the state the preceding beat is measured against makes
    this case a turn answered against a world nothing produces — which is precisely what
    the bare park was — so it fails HERE, in the seed, rather than as a puzzling number
    after an hour of GPU time.  Same discipline as the apply case's fixture asserts: a
    fixture states what it means and says so out loud when it stops being true."""
    assert ask_id is not None, f"{case.case_id}: the seeded ask must carry a message id"
    assert _seeded_ask_id(db, case.ask) == ask_id, (
        f"{case.case_id}: the seeded ask must be findable by its own content"
    )
    _assert_the_round_reads_as_a_conversation(db, case)
    _assert_nothing_enacted(db, case)
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.ELICIT.value, (
        f"{case.case_id}: the machine must be parked in elicit, not {latest}"
    )
    assert latest.anchor_message_id == ask_id, (
        f"{case.case_id}: the park must be anchored to the ask, not {latest.anchor_message_id}"
    )


def _assert_the_round_reads_as_a_conversation(db: Database, case: _ElicitRound) -> None:
    """The ask and the teach question come back as a two-turn CONVERSATION, not as one
    user turn — the cheap version of the composed world's own window probe, on the beat
    whose whole precondition is that Penny asked something and the demonstration answers
    it.  Penny's turn reaches the window only because it is threaded to the ask."""
    expected = [
        (PennyConstants.MessageDirection.INCOMING, case.ask),
        (PennyConstants.MessageDirection.OUTGOING, case.teach_question),
    ]
    window = db.messages.get_messages_since(TEST_SENDER, since=datetime.min, limit=len(expected))
    seen = [(row.direction, row.content) for row in window]
    assert seen == expected, f"{case.case_id}: the round must read as a conversation, got {seen}"


def _assert_nothing_enacted(db: Database, case: _ElicitRound) -> None:
    """The other half of that probe — turn 1 enacted NOTHING, which is the whole of what
    its five scored state checks assert: no skill learned, no entry written by any run,
    no page fetched, and the framework's own seeded collection untouched."""
    assert not db.skills.list_all(), f"{case.case_id}: the round starts with no skill learned"
    assert not _entries_written_by_this_run(db), f"{case.case_id}: no run has written anything"
    assert not _pages_fetched(db), f"{case.case_id}: no page has been fetched yet"
    assert not collection_entries(db, PennyConstants.MEMORY_DISLIKES_COLLECTION), (
        f"{case.case_id}: the seeded collection starts untouched"
    )


def _seeded_ask_id(db: Database, ask: str, *, limit: int = _ROUND_INCOMING_TURNS) -> int | None:
    """The id of the seeded instigating ask — the row this round is anchored to.

    Found by its CONTENT rather than by position: the turn under test is logged AFTER the
    run (so it never doubles into that turn's own recall), so by scoring time the round's
    own turns and this one sit side by side and only the case knows which it seeded.
    ``limit`` is how many incoming rows the round spans — two for a demonstration
    answered against its ask, three once the demonstration is seeded too."""
    for row in db.messages.get_user_messages(TEST_SENDER, limit=limit):
        if row.content == ask:
            return row.id
    return None


def _mentions(token: str, texts: list[str]) -> bool:
    """Whether the page's own fact turns up in any of ``texts``, CASE-FOLDED.

    The fact is a word off the page ('rye', 'not scheduled', 'Tidewater'), and a value
    lifted into a sentence takes whatever capitalisation the sentence needs — so
    matching case-sensitively would score correct writes as misses on most of these
    pages, which is a scorer bug reported as a finding."""
    return any(token.lower() in text.lower() for text in texts)


def _skill_steps(db: Database) -> list[SkillStep]:
    """Every step of every learned skill — the ROUTINE run-end extraction left behind,
    read structurally off the stored rows rather than off the demonstration.

    The step, not its substitutions alone, because a leaf's demonstrated value lives in
    its step's ``arguments`` and that is what says whether the leaf is a destination
    (#1854)."""
    return [step for skill in db.skills.list_all() for step in steps_from_json(skill.steps)]


def _skill_substitutions(steps: list[SkillStep]) -> list[SkillSubstitution]:
    """Every dynamic leaf of the learned routine — the SHAPE of what was captured."""
    return [sub for step in steps for sub in step.substitutions]


def _skill_parameters(db: Database) -> list[SkillParameter]:
    """Every declared parameter of every learned skill — the routine's INTERFACE, which
    since #1830 is the framer's draw and lives at SKILL level (its declared interim:
    nothing joins a parameter to a leaf of the program yet)."""
    return [
        parameter
        for skill in db.skills.list_all()
        for parameter in parameters_from_json(skill.parameters)
    ]


# The three shape labels, named once: each is read BOTH as a scored check and as the
# not-applicable row a sample with no learned skill renders, and a label is a diff-join
# key — two spellings of one check are two checks to every report that reads them.
_PLACEHOLDERS_ONLY_LABEL = (
    "state: every spot in the routine is a placeholder (the labelling draw landed)"
)
_ATTACHMENT_MARK_LABEL = "state: the destination leaf still carries the attachment mark"
_INTERFACE_LABEL = "state: the interface asks for the page, plus at most the found-thing"

# What the interface may ask for, as the two families a drawn parameter can answer —
# classified by the SHARED name-first-then-description discipline (``classify_by_family``),
# so this check and the framer suite's own set check can never read a draw two ways.
#
# The PAGE is mandatory and NAME-ONLY, on the framer suite's breadth (imported rather than
# restated: what a page parameter may reasonably be called is one agreed vocabulary, and a
# page is the thing NAMED as one — a description mentioning a page promotes nothing).
_PAGE_LABEL = "page"
_PAGE_FAMILY = ParameterFamily(_PAGE_LABEL, _PLACE_TOKENS, name_only=True)

# The FOUND-THING is the leeway (code-owner ruling, 2026-08-05, from a thinking-audited
# draw): every one of these asks names what to look for as well as where — "look for the
# late sailing line" — so a second parameter carrying THAT is a defensible reading of the
# enumerate-then-filter rule, not an invention, and the check accepts at most one.  Both
# passes apply here, unlike the page: the piece has no canonical noun, so a well-judged
# name the tokens don't anticipate is allowed to land through its description.
_FOUND_THING_LABEL = "found-thing"
_FOUND_THING_FAMILY = ParameterFamily(
    _FOUND_THING_LABEL, ("search", "phrase", "term", "keyword", "target", "line", "query")
)
_INTERFACE_FAMILIES = (_PAGE_FAMILY, _FOUND_THING_FAMILY)


def _placeholders_only_check(subs: list[SkillSubstitution]) -> Check:
    """Every spot in the routine is a PLACEHOLDER — none is still a leaf parameter
    (#1828).

    The labeller names every spot unconditionally and a named spot stops being a
    parameter, so a leftover ``HOLE`` means the labelling draw FELL BACK (it is
    all-or-nothing at the draw) and the routine kept its arg-derived names.  Bindings
    are untouched by any of this: a value a prior step produced was never asked of
    anyone."""
    left = [sub for sub in subs if sub.kind == SkillSubKind.HOLE]
    asking = sorted({sub.parameter for sub in left if sub.parameter is not None})
    return Check(
        _PLACEHOLDERS_ONLY_LABEL,
        not left,
        rationale=f"{len(left)} spot(s) still a leaf parameter: {asking}" if left else None,
        kind="state",
    )


def _attachment_mark_check(db: Database, steps: list[SkillStep]) -> Check:
    """The destination leaf still carries the ATTACHMENT MARK (#1783, #1827 principle
    4) — scored only when the routine HAS a destination (#1854).

    Where a routine writes is decided by what it is applied to and is never asked of the
    user, and the mark is exactly what the apply turn binds — so a routine whose
    destination came back unmarked is one the next edge cannot point anywhere.

    A routine that keeps nothing has no such leaf to mark, and read-only routines are
    legitimate (code-owner ruling: "there's tons of skills that be like 'check the scores
    here, check the schedule there, tell me' — that doesn't require a store step").  Since
    #1850 a learn round is extracted whatever shape it had, so a browse-only skill is now
    a state this suite reaches, and grading it here would fail a routine for a step nobody
    asked for.

    Applicability is read from the DEMONSTRATED VALUES, never from the marks — "is
    anything marked?" is the check itself, so answering applicability with it would pass
    every routine vacuously and never catch a dropped mark.  A leaf is a destination when
    its demonstrated value names one of Penny's own collections, which is exactly what
    ``distill_steps`` marks on, read through the same registry policy extraction uses
    (``attachment_names``).  Bindings are excluded there and excluded here: a value a
    prior step produced is already explained, so nothing is left for an attachment to
    decide.  Keyed to no tool name — a skill is an arbitrary tool sequence, so a plugin
    verb's destination counts like a ``collection_write``'s."""
    destinations = _destination_subs(db, steps)
    if not destinations:
        return Check.na(_ATTACHMENT_MARK_LABEL, kind="state")
    marked = any(sub.attachment for sub in destinations)
    return Check(
        _ATTACHMENT_MARK_LABEL,
        marked,
        rationale=None if marked else "the destination leaf came back unmarked",
        kind="state",
    )


def _destination_subs(db: Database, steps: list[SkillStep]) -> list[SkillSubstitution]:
    """Every leaf of the routine that points at one of Penny's own collections — the
    spots the attachment fills, identified by their demonstrated value alone."""
    collections = attachment_names(db)
    return [
        sub
        for step in steps
        for sub in step.substitutions
        if sub.kind != SkillSubKind.BINDING and _leaf_at(step.arguments, sub.path) in collections
    ]


def _interface_check(required: list[SkillParameter]) -> Check:
    """The interface asks for the PAGE, plus AT MOST the found-thing (#1830, amended by
    the code owner's leeway ruling of 2026-08-05).

    The page is mandatory — it is the one piece every one of these asks leaves to re-say,
    and a routine that cannot be pointed at one can only repeat its demonstration.  A
    SECOND parameter is accepted when it carries what the user's own turns named as the
    thing to find: the ferry round's draws ask for one under several names (`search_phrase`,
    `search_term`, `keyword`, `line_text` — the family is what is agreed, never one
    spelling), and the audited thinking read "the late sailing" out of both turns — which is
    the enumerate-then-filter rule applied correctly, so scoring it a miss would be the
    scorer marking a sound draw wrong.  Anything else stays a miss: a second parameter of
    another kind is the invention that rule exists to stop, and a third is one however it
    is named.  Every accepted
    parameter carries a description — it is what the ambient ``needs:`` row renders, so one
    nobody can read is one nobody can bind."""
    answered = _interface_families(required)
    pages, found, rejected = (_of_family(required, answered, label) for label in _READINGS)
    accepted = len(pages) == 1 and len(found) <= 1 and not rejected
    described = all(_says_what_to_supply(parameter) for parameter in pages + found)
    return Check(
        _INTERFACE_LABEL,
        accepted and described,
        rationale=_interface_rationale(pages, found, rejected, described),
        kind="state",
    )


def _interface_families(required: list[SkillParameter]) -> list[ParameterFamily | None]:
    """Which family each required parameter answers, through the SHARED classifier — a
    parameter carries no description in the model when a draw left none, and an absent
    description classifies as the empty text it is."""
    return classify_by_family(
        [(parameter.name, parameter.description or "") for parameter in required],
        _INTERFACE_FAMILIES,
    )


# The three readings a required parameter can land in, in the order the rationale names
# them: the mandatory page, the accepted found-thing, and everything else.
_READINGS = (_PAGE_LABEL, _FOUND_THING_LABEL, None)


def _of_family(
    required: list[SkillParameter],
    answered: list[ParameterFamily | None],
    label: str | None,
) -> list[SkillParameter]:
    """The required parameters that answered ``label`` — ``None`` for the ones that
    answered no accepted family at all."""
    return [
        parameter
        for parameter, family in zip(required, answered, strict=True)
        if (family.label if family is not None else None) == label
    ]


def _says_what_to_supply(parameter: SkillParameter) -> bool:
    """A parameter carries the one-line what-to-supply the framer writes for it — the
    description is optional in the model (a labelling fallback leaves none), so an
    absent one is a real, distinct shape and not something to read as empty text."""
    return parameter.description is not None and not is_blank(parameter.description)


def _interface_rationale(
    pages: list[SkillParameter],
    found: list[SkillParameter],
    rejected: list[SkillParameter],
    described: bool,
) -> str:
    """WHICH reading was drawn, named on the pass as well as the miss — the two accepted
    shapes are different answers to the same ask, and a report that showed only "passed"
    would hide which one the run committed to."""
    if rejected:
        names = ", ".join(parameter.name for parameter in rejected)
        return f"rejected: {names} answers no accepted family"
    if len(pages) != 1:
        return f"{len(pages)} answer the page: {[parameter.name for parameter in pages]}"
    if len(found) > 1:
        return f"{len(found)} answer the found-thing: {[parameter.name for parameter in found]}"
    if not described:
        undescribed = [p.name for p in pages + found if not _says_what_to_supply(p)]
        return f"carries no description: {', '.join(undescribed)}"
    if not found:
        return f"{pages[0].name} alone"
    return f"{pages[0].name} + {found[0].name} (user-named)"


def _interface_advisories(db: Database) -> list[Check]:
    """What the framer committed to, verbatim — one ADVISORY row per parameter.

    Whether a name is WELL judged is read at joint review against the reference outputs
    on the ticket; a scorer that faked that reading would be answering for the draw."""
    return [
        Check(
            f"drew parameter {parameter.name!r} — {parameter.description!r}",
            True,
            scored=False,
            kind="state",
        )
        for parameter in _skill_parameters(db)
    ]


def _extraction_shape_checks(db: Database) -> list[Check]:
    """The shape run-end extraction produced, read off the stored skill: the LABELLER's
    half (every spot a placeholder, and — where the routine keeps anything — the
    destination still marked) and the FRAMER's half (one required parameter, described),
    with the drawn interface riding advisory.

    All three go NOT-APPLICABLE when no skill was learned at all.  That miss is already
    the scored "a skill was learned from the round" check, so grading the shape of a
    skill that does not exist would recount one failure three times — and "every spot is
    a placeholder" over an empty routine is vacuously true, which would render as a pass
    for a round that produced nothing.  The mark check has a second not-applicable case
    of its own (#1854): a routine with no destination has nothing to mark."""
    if not db.skills.list_all():
        return [
            Check.na(_PLACEHOLDERS_ONLY_LABEL, kind="state"),
            Check.na(_ATTACHMENT_MARK_LABEL, kind="state"),
            Check.na(_INTERFACE_LABEL, kind="state"),
        ]
    steps = _skill_steps(db)
    required = [parameter for parameter in _skill_parameters(db) if parameter.required]
    return [
        _placeholders_only_check(_skill_substitutions(steps)),
        _attachment_mark_check(db, steps),
        _interface_check(required),
        *_interface_advisories(db),
    ]


def _attaches_nothing_checks(db: Database, created: list[MemoryRow]) -> list[Check]:
    """Learning must not INSTANTIATE (#1706).  Scored against what this run TOUCHED: a
    collection it created, or — when it reused an existing one — nothing, since a seeded
    collection's own prompt and cadence predate the round and failing on those would
    report the framework's fixtures as her doing."""
    instantiated = [row for row in db.memories.list_all() if row.skill_name is not None]
    return [
        Check(
            "state: no skill was attached anywhere (learning does not instantiate)",
            not instantiated,
            rationale=f"attached to {[row.name for row in instantiated]}" if instantiated else None,
            kind="state",
        ),
        Check(
            "state: no program was rendered into the collection it created",
            all(row.extraction_prompt is None for row in created),
            kind="state",
        )
        if created
        else Check.na(
            "state: no program was rendered into the collection it created", kind="state"
        ),
        Check(
            "state: nothing it created was scheduled (no trigger, no notify)",
            all(row.schedule is None and not row.notify for row in created),
            kind="state",
        )
        if created
        else Check.na(
            "state: nothing it created was scheduled (no trigger, no notify)", kind="state"
        ),
    ]


def _anchor_carried_check(db: Database, ask: str) -> Check:
    """The anchor was CARRIED: the move that parked the machine in learn still points at
    the ask that opened the round (#1827's anchor rule — every transition that keeps the
    machine parked carries it unchanged, which is what lets a turn three messages later
    still be classified against what was asked for).

    Scored ONLY when the machine landed in learn — the same conditional the idle → elicit
    cases use: a misroute is already named by the landed-state advisory, and scoring the
    anchor on top of it would recount one classifier miss as an enactment failure."""
    label = "state: the anchor was carried (still the ask that opened the round)"
    latest = db.machine.latest_transition()
    if latest is None or latest.to_state != ConversationState.LEARN.value:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, ask)
    anchored = latest.anchor_message_id
    carried = asked is not None and anchored == asked
    return Check(
        label,
        carried,
        rationale=None if carried else f"anchored to {anchored}, the ask is {asked}",
        kind="state",
    )


def _score_elicit_to_learn(
    db: Database, before: set[str], reply: str, *, case: _LearnCase
) -> list[Check]:
    """The demonstrated round ran, and NOTHING was instantiated.

    "Remember it" is a naive ``collection_write``: it auto-creates a collection
    and puts the value in it.  What must NOT happen is the fold — no skill bound
    to that collection, no rendered program, no schedule.  The skill is learned
    (it exists in the registry) and stays unattached until the user asks for it.
    What that learning PRODUCED is read off the stored skill: an all-placeholder
    routine, its destination still marked, over the one parameter the ask leaves.

    ONE scorer for all five cases, bound to the case's own page fact and ask.  The
    labels are diff-join keys, so they read identically on every case and keep the
    wording the auction script gave them even where a ferry timetable is what was
    read."""
    created = new_collections(db, before)
    written = _written_texts(_entries_written_by_this_run(db))
    landed = _mentions(case.stored, written)
    return [
        Check(
            "state: she browsed the listing (the demonstrated fetch happened)",
            tool_was_called(db, "browse"),
            kind="state",
        ),
        # The fact counts wherever in the entry it landed — its KEY or its content
        # (#1854, code-owner ruling: "loosen the scorer; we can reason about the
        # semantics of keys/values/remembering later").  Two measured samples wrote the
        # arrival's title as the KEY and the date as the value, which is a workable
        # shape for an arrival-shaped watch — a repeat title is KEY_EXISTS_UNCHANGED
        # and a new one is a new key — and was scored a miss for putting the fact on
        # the wrong side of the entry.  The label is unchanged: it is a diff-join key,
        # and what the check tests is still that the browsed fact landed durably.
        Check(
            "state: the browsed price landed durably (remember = a plain write)",
            landed,
            rationale=None
            if landed
            else (f"wrote {written}" if written else "nothing was written"),
            kind="state",
        ),
        Check(
            "state: a skill was learned from the round",
            bool(db.skills.list_all()),
            kind="state",
        ),
        *_attaches_nothing_checks(db, created),
        *_extraction_shape_checks(db),
        _anchor_carried_check(db, case.ask),
        Check(
            "reply: she reports the value she stored (SAID == DID)",
            _mentions(case.stored, outgoing_replies(db)),
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
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


async def _run_learn_case(chat_eval: ChatEval, case: _LearnCase) -> None:
    """Drive one elicit → learn case: parked on its own ask, its page installed, the
    shared scorer bound to the fact that page carries.  Report-only — the thresholds are
    the code owner's to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.demo,
        browse=[case.page],
        seed=_seed_elicit_round(case),
        score=partial(_score_elicit_to_learn, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_elicit_to_learn_runs_the_round_and_instantiates_nothing(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn: parked on the teach question, the user supplies the steps.
    She follows them once — browse, find, remember — reports the value she
    actually stored, and learns the skill.  She instantiates NOTHING: the
    collection her write created carries no skill, no program, no schedule."""
    await _run_learn_case(chat_eval, _AURORA_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_takes_the_url_from_the_demonstration(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn where the ask never gave a page: the demonstration supplies the
    timetable's url along with what to look for on it, and the round runs on what she
    was just told rather than on a search she guessed her way to."""
    await _run_learn_case(chat_eval, _FERRY_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_stores_the_days_special(chat_eval: ChatEval) -> None:
    """elicit → learn on the store-each-day digest: shown the routine once, she runs it
    once — today's special read off the page and written down — and the day-after-day
    part stays a job nobody has set up yet."""
    await _run_learn_case(chat_eval, _BAKERY_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_records_the_count_without_comparing(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn where the job is a number watched over time: the demonstration is
    a plain read-and-remember, so the count lands as the baseline it is and nothing
    compares it against anything — there is nothing yet to compare it to."""
    await _run_learn_case(chat_eval, _COLONY_ROUND)


@pytest.mark.asyncio
async def test_elicit_to_learn_learns_despite_the_urgency(chat_eval: ChatEval) -> None:
    """elicit → learn under the act-now ask: the instructions have arrived now, so the
    round is exactly what they say — read the page, take the newest arrival, remember
    it — and the "tell me the moment" part is still a job the next turn sets up."""
    await _run_learn_case(chat_eval, _ARRIVALS_ROUND)


# ── elicit → learn: the page does not hold the fact, and the round stops ──────
#
# The sixth demonstration, and the only one whose page cannot answer the question it was
# pointed at.  The instructions are ordinary — go here, find this, remember it — and the
# page is an ordinary noticeboard: a compost schedule, a tool-shed notice, a potluck on
# the 14th.  What it never says, anywhere, is when the plot waitlist opens.
#
# So the round CANNOT be carried out as given, and what this case measures is what she
# does about that: read the page, stop at the step the world does not support, and say
# so.  The harm on the other side is a value invented to finish the round, or a step
# reported as done that never happened — which is why "nothing was written" is the core
# scored claim here rather than an absence noted in passing.
#
# Distinct from the ferry round above, where "not scheduled this season" IS the fact:
# present on the page, readable, storable.  Here the page does not speak to the question
# at all — the honest-absence case.
#
# It has no idle → elicit sibling: what it measures is a property of the ROUND (the world
# lacks the thing) rather than of the ask that reached it, and the preceding beat is
# seeded exactly as the other five seed theirs.
#
# The scorer below is this case's own.  The shared one asks whether the page's fact
# landed durably and whether a skill was learned, and neither question has an answer
# here: there is no fact to land, and whether a round that could not be carried out
# should learn anything at all is an open design point (#1850's no-requisite extraction
# may well mint a browse-only skill from it) — reported below, graded nowhere.

_GARDEN_NOTICEBOARD_URL = "https://communitygarden.example/noticeboard"
# Matched on "noticeboard", the token the ask and the address SHARE — the same reason the
# ferry page matches on "timetable": the ask says "the community garden's noticeboard
# page" while the host says "communitygarden", so a page matched on the host alone would
# answer a direct read of the url and miss a search that phrases the ask.
#
# The solo markdown link sits in the MIDDLE of the notices, because a search-shaped read
# keeps only the lines within two of one (``_trim_search_result``) — placed at the end it
# would take the compost schedule and drop everything after it, leaving a page that no
# longer carries the several true facts this fixture exists to carry.
_GARDEN_NOTICEBOARD = CannedPage(
    match="noticeboard",
    text=(
        "Title: Community garden noticeboard — this month's notices | communitygarden\n"
        f"{_GARDEN_NOTICEBOARD_URL}\n"
        "\n"
        "Notices for a fictional allotment site, posted by the committee each month.\n"
        "Compost collection: second and fourth Saturday, 9am, by the east gate.\n"
        f"[Community garden noticeboard]({_GARDEN_NOTICEBOARD_URL})\n"
        "Tool shed: the lock code changed — ask a committee member for the new one.\n"
        "Potluck: the 14th at noon in the orchard corner, bring a dish to share.\n"
    ),
)

# Reference reply (a review target under the semantic-breadth rule, never a scorer
# string): what she found, which step stopped her, and the hand-back.
#
#   looked at the noticeboard — it lists the compost schedule and a potluck on the 14th,
#   but nothing about the plot waitlist opening. where should i look for that, or should
#   i watch for it to appear?
_GARDEN_ROUND = _AbsentRound(
    case_id="transition-elicit-to-learn-absent",
    ask=(
        "can you check the community garden's noticeboard page every week and let me "
        "know when the plot waitlist opens?"
    ),
    teach_question=(
        "i can learn that — walk me through it once? where should i look, and what am i "
        "checking for?"
    ),
    demo=(f"go to {_GARDEN_NOTICEBOARD_URL}, find the plot waitlist opening date, and remember it"),
    page=_GARDEN_NOTICEBOARD,
)


def _registry_advisories(db: Database) -> list[Check]:
    """What the registry holds when the round ends — rendered, graded nowhere.

    Under #1850's no-requisite extraction a learn turn mints a skill from whatever calls
    it made, so a round that only browsed may still leave one behind.  Whether a
    demonstration that COULD NOT be carried out should learn anything is an open design
    point, so the case reports what it finds and answers it not at all — including the
    empty registry, which renders as its own row rather than as no rows (an outcome
    nobody can see is one nobody rules on)."""
    skills = db.skills.list_all()
    if not skills:
        return [
            Check(
                "state: the registry is empty at run end (nothing was learned)",
                True,
                scored=False,
                kind="state",
            )
        ]
    return [
        Check(
            f"state: the registry holds {skill.name!r} at run end",
            True,
            scored=False,
            kind="state",
        )
        for skill in skills
    ]


def _score_elicit_to_learn_absent(db: Database, before: set[str], reply: str) -> list[Check]:
    """She read the page, and the round stopped there with nothing invented to finish it.

    The middle two claims are the point: NOTHING was written anywhere by this run (no
    value was manufactured to stand in for the one the page does not carry) and nothing
    was set up on the strength of it.  Around them, the step she WAS given did happen —
    the fetch — and the machine is still parked in learn on the ask, so the round hands
    back for instructions it can carry out instead of breaking out to idle as though it
    were finished.

    Whether the reply is HONEST about which step stopped her is read at joint review
    against the reference above: one line of English carries no structural signal."""
    written = _entries_written_by_this_run(db)
    landed = _landed_state(db)
    parked = landed == ConversationState.LEARN.value
    browses = count_tool_calls(db, "browse")
    return [
        Check(
            "state: she browsed the noticeboard (the demonstrated fetch happened)",
            tool_was_called(db, "browse"),
            kind="state",
        ),
        # Read off the ENTRIES, not their texts: what this case claims is that no entry
        # was written at all, and #1854's `_written_texts` drops an empty half — so a
        # write whose value came back blank would read as nothing written, which is the
        # one reading this check must never give.  The texts are what the rationale
        # NAMES when it missed, which is that helper's own second customer.
        Check(
            "state: this run wrote no entry anywhere (nothing was invented)",
            not written,
            rationale=f"wrote {_written_texts(written)}" if written else None,
            kind="state",
        ),
        Check(
            "state: no collection was created (nothing was set up)",
            not new_collections(db, before),
            kind="state",
        ),
        Check(
            "state: the machine stayed parked in learn (the round hands back)",
            parked,
            rationale=None if parked else f"landed in {landed}",
            kind="state",
        ),
        _anchor_carried_check(db, _GARDEN_ROUND.ask),
        *_registry_advisories(db),
        Check(
            f"calls: {browses} browse call(s)",
            True,
            scored=False,
            kind="proc",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


@pytest.mark.asyncio
async def test_elicit_to_learn_stops_when_the_page_lacks_the_fact(
    chat_eval: ChatEval,
) -> None:
    """elicit → learn where the page does not carry the asked-for fact: the noticeboard
    is read, the plot waitlist opening is not on it, and the round stops at that step —
    no entry written, no collection set up, and the machine still parked in learn on the
    ask, waiting for instructions it can carry out."""
    await chat_eval(
        case_id=_GARDEN_ROUND.case_id,
        message=_GARDEN_ROUND.demo,
        browse=[_GARDEN_ROUND.page],
        seed=_seed_elicit_round(_GARDEN_ROUND),
        score=_score_elicit_to_learn_absent,
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


# ── learn → apply: the offer accepted, the routine set running ────────────────
#
# Five cases, one per elicit → learn round above, so all three edges chain subject for
# subject.  Each seeds the WHOLE round rather than only its last turn: the instigating
# ask INCOMING (the anchor), Penny's teach question OUTGOING, the demonstration
# INCOMING, her closing report OUTGOING, and the TWO transition rows that carried the
# machine there — idle → elicit stamped on the ask, then elicit → learn keeping that
# anchor unchanged (#1827's anchor lifecycle).  So the acceptance is answered as message
# five of one exchange, which is what production hands the classifier.
#
# **Recorded seeding reversal.**  This case used to OMIT the instigating ask and anchor
# the park to the demonstration.  That was a workaround from before the classifier's
# accept-pool work, when seeding the ask made the acceptance's terms read as steps still
# being specified.  Production does the opposite — the anchor is stamped entering the
# round and carried while parked — and ``test_state_classifier.py``'s own
# ``learn-apply-accept`` case anchors on the ask, its all-three-terms phrasings there
# precisely because that is the hard shape.  If terms-read-as-steps resurfaces under the
# full history, that is a real finding about the live edge, not noise.
#
# The acceptance never restates the routine's parameters, so the seeded conversation IS
# where they come from: the cadence and any end condition are in the acceptance itself,
# and the page — plus, for the ferry, what to look for on it — is in the turns before.
#
# Every case seeds a shared DECOY skill beside its own, so binding the WRONG routine is
# a reachable, scored failure: a NON-EXISTENT name is refused by membership validation
# and never reaches the turn (#1839 mechanics, not this scorer's business), while an
# existing-but-wrong one PROCEEDS and only the scorer can catch it.  Each case's page is
# installed as a live temptation too — re-running the round instead of setting it up is
# a fetch that would SUCCEED and get caught.
#
# The reference replies quoted above each case are review targets, never scorer strings.


class _DemonstratedRound(NamedTuple):
    """The canonical two-step round a fixture skill is distilled from — browse the page,
    write what it said into a collection — and the naive world that round left behind.

    One shape for all five scenarios, because the shape is what the preceding beat
    produces: a routine is a look-up and a durable write, and everything that differs
    between subjects is a value in here.  ``entry_value`` is BOTH what the browse
    returned and what the write stored, so the write's content leaf binds to step 1
    exactly as a real demonstration's does — which is what leaves the other four leaves
    as the spots the labeller names."""

    url: str
    extract: str
    collection: str
    entry_key: str
    entry_value: str


class _FixtureDraws(NamedTuple):
    """The two hand-written DRAWS a fixture legitimately stands in for — the only part
    of a fixture skill that is not the production pipeline's own output.

    The four ``LeafLabel``s are the LABELLER's half (#1828), one per spot: the page, the
    extract instruction, the storage collection (which additionally carries the
    attachment mark) and the entry key.  ``signature`` is the FRAMER's half (#1830) —
    the routine's public interface, written from the user's ask alone.  Both are
    transcribed from the preceding beat's measured run, so a case starts from the shape
    that beat actually hands forward rather than from a convenient invention."""

    page: LeafLabel
    extract: LeafLabel
    collection: LeafLabel
    entry_key: LeafLabel
    signature: SkillSignature


def _demonstrated_ledger(demonstrated: _DemonstratedRound) -> list[DistillInput]:
    """The round's ledger, as the run-end extractor would have read it off the
    promptlog: the browse that fetched the fact, then the write that stored it.

    The results carry the real ``(<tool> result)`` frame, so distillation strips it the
    way it does in production and the write's content binds to the browse's PAYLOAD."""
    return [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [demonstrated.url], "extract": demonstrated.extract},
            result=f"You opened {demonstrated.url} (browse result)\n{demonstrated.entry_value}",
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": demonstrated.collection,
                "entries": [{"key": demonstrated.entry_key, "content": demonstrated.entry_value}],
            },
            result=(
                f"You saved an entry to {demonstrated.collection}: "
                "(collection_write result)\nWrote 1 entry."
            ),
        ),
    ]


def _authored_labels(
    demonstrated: _DemonstratedRound, draws: _FixtureDraws
) -> dict[str, LeafLabel]:
    """The labeller's draw keyed by the DEMONSTRATED VALUE rather than by the arg-derived
    name the distiller happens to mint — so the fixture states what it means and cannot
    go quietly stale if that naming changes.

    EVERY spot is listed because an accepted draw covers every spot: a line missing for
    one is a WHOLE-draw failure, so a partly-labelled routine is a shape run-end
    extraction cannot hand anyone.  The destination is a spot like any other and
    additionally carries the attachment mark — what the routine is applied to fills it,
    which is precisely what the apply turn under test then binds."""
    return {
        demonstrated.url: draws.page,
        demonstrated.extract: draws.extract,
        demonstrated.collection: draws.collection,
        demonstrated.entry_key: draws.entry_key,
    }


def _fixture_labels(steps: list[SkillStep], authored: dict[str, LeafLabel]) -> SkillLabels:
    """The labeller's draw, mapped from the demonstrated VALUES it is authored
    against onto the spot names the distiller happened to mint.

    Every authored label must map home: one that doesn't is a fixture whose ledger
    has drifted from what it claims, and it fails LOUDLY here rather than quietly
    seeding the enactment case a world with a spot left unnamed."""
    labels: dict[str, LeafLabel] = {}
    for step in steps:
        for sub in step.substitutions:
            if sub.parameter is None:
                continue
            value = str(_leaf_at(step.arguments, sub.path))
            if value in authored:
                labels[sub.parameter] = authored[value]
    assert len(labels) == len(authored), (
        f"the fixture's labels must all map home — matched {sorted(labels)} of {sorted(authored)}"
    )
    return SkillLabels(labels=labels)


def _fixture_skill(
    demonstrated: _DemonstratedRound,
    draws: _FixtureDraws,
    origin_message: str,
    runs: _JourneyRuns,
) -> SkillDraft:
    """The skill a demonstrated round leaves in the registry, built by the PRODUCTION
    pipeline over its ledger: ``distill_steps`` for the structure, then BOTH halves of
    the run-end split applied by their own production function — ``_apply_leaf_labels``
    for the labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's
    signature.  Only the two DRAWS are hand-written, which is what a fixture is for.

    So a case's starting world is the shape extraction produces, not a convenient copy
    of it — and the shape is the framer's declared interim (#1830), which nothing here
    re-states: an ALL-PLACEHOLDER recipe (every spot named by the labeller, the write
    target still carrying its attachment mark) over the SKILL-level parameters the framer
    minted from the ask.  Nothing joins those parameters to a leaf yet — that is the
    runtime-join beat — so it is the registry row, ``collection_set``'s
    unbound-parameter check, and job identity that carry them, which is exactly what the
    apply turn under test has to satisfy."""
    # The registry as this fixture's round saw it — #1783 marks a leaf whose
    # demonstrated value names one of Penny's collections, so the destination is
    # only marked if the collection actually existed.
    steps, parameters = distill_steps(
        _demonstrated_ledger(demonstrated), frozenset({demonstrated.collection})
    )
    authored = _authored_labels(demonstrated, draws)
    steps, distilled = _apply_leaf_labels(steps, parameters, _fixture_labels(steps, authored))
    name, description = _naming(draws.signature, origin_message)
    framed = _interface_parameters(draws.signature, distilled)
    # The framer's parameters are what the apply turn must supply, so a production
    # application that stopped carrying the signature through would seed a routine
    # nothing could be pointed at — silently, and the apply case would report it as
    # the model's failure.  It fails here instead.
    framed_names = [parameter.name for parameter in framed]
    assert framed_names == [parameter.name for parameter in draws.signature.parameters], (
        f"the framed interface must survive application — got {framed_names}"
    )
    return SkillDraft(
        name=name,
        intent=description,
        description=description,
        steps=steps,
        parameters=framed,
        source_run_id=runs.learn_turn,
    )


# ── The five fixture skills, sampled from the framer's own measured draws ─────
#
# Each scenario's LABELLER draws are transcribed from the learn beat's final composed run
# (its per-sample databases, read as data rather than off a transcript) and lightly
# cleaned — no "e.g." garnish, which the traces show is appended after a line is decided.
#
# Each scenario's SIGNATURE is RE-SAMPLED from the framer suite's run under the round-9
# prompt (#1863: no timing, scheduling or notification in the name or description, and no
# parameter's value).  One clean draw is taken WHOLE per scenario — never composed across
# samples — so a fixture is a shape the pipeline really produces.  Two of the five names
# moved with the draws (``monitor_colony_count`` → ``monitor_webpage_number``,
# ``monitor_new_arrival`` → ``retrieve_newest_item``): the framer now generalises past the
# subject, which is what "name the KIND of task, never the instance" asks for, so the
# fixture follows rather than re-injecting a subject nothing drew.  Every scenario drew a
# single ``url`` parameter except the ferry, whose mode is ``url`` plus a found-thing (now
# named ``keyword``); per the code owner's ruling the ferry fixture seeds that
# TWO-parameter shape, as the deliberate stress case for multi-parameter binding.
#
# What the re-sample REMOVED, verbatim from the old fixtures, is the class the round was
# for: "checks a webpage **daily** and **notifies** when a newer item appears" (a cadence
# and a notify setting written into what the routine IS — three samples read it as not
# covering a two-hourly ask), "monitor a ferry timetable for updates to **the late sailing
# entry**" (the found-thing's own VALUE, which made an ask about another sailing read as a
# different job), and "alerts when it decreases" on the count watcher.
#
# Each scenario is a JOURNEY, and its run ids are minted once here — the skill cites the
# learn run that taught it, and every seeder writing that journey's turns writes under
# the same bundle, so the citation resolves in whichever world the journey is laid down.

_AURORA_RUNS = _journey_runs("aurora")
_FERRY_RUNS = _journey_runs("ferry")
_BAKERY_RUNS = _journey_runs("bakery")
_COLONY_RUNS = _journey_runs("colony")
_ARRIVALS_RUNS = _journey_runs("arrivals")
# The decoy was taught on some occasion this world does not lay down, so its round has a
# run id and no rows — a routine learned before the history a case seeds begins.
_DECOY_RUNS = _journey_runs("museum")

_AURORA_DEMONSTRATED = _DemonstratedRound(
    url=LISTING_URL,
    extract="the current price",
    collection="aurora-deck-2-price",
    entry_key="listing price",
    entry_value="$499",
)
_AURORA_SKILL = _fixture_skill(
    _AURORA_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="page_url", description="the url of the page to browse"),
        extract=LeafLabel(
            name="value_to_find",
            description="a plain text description of what information to retrieve from the page",
        ),
        collection=LeafLabel(
            name="storage_collection",
            description="the identifier for the storage area where scraped data will be saved",
        ),
        entry_key=LeafLabel(
            name="entry_key",
            description="the key under which the extracted value is stored within that collection",
        ),
        signature=SkillSignature(
            name="monitor_price",
            description="Monitors a web listing and reports when its price changes.",
            parameters=(
                FramedParameter(name="url", description="The URL of the listing to watch"),
            ),
        ),
    ),
    _AURORA_ROUND.demo,
    _AURORA_RUNS,
)

_FERRY_DEMONSTRATED = _DemonstratedRound(
    url=_FERRY_TIMETABLE_URL,
    extract="the late sailing line",
    collection="harborferries-late",
    entry_key="late-sailing",
    entry_value="Late sailing: not scheduled this season.",
)
_FERRY_SKILL = _fixture_skill(
    _FERRY_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(
            name="timetable_url",
            description="the url of the ferry timetable page to browse each run",
        ),
        extract=LeafLabel(
            name="line_to_find", description="the text or line to look for on that page"
        ),
        collection=LeafLabel(
            name="storage_collection",
            description="the name of the storage collection where the result is saved each run",
        ),
        entry_key=LeafLabel(
            name="entry_key",
            description="the key under which the extracted value will be stored in the collection",
        ),
        signature=SkillSignature(
            name="check_ferry_timetable",
            description=(
                "Check a ferry timetable page for updates and report the status of a specified line"
            ),
            parameters=(
                FramedParameter(name="url", description="the URL of the timetable page to fetch"),
                FramedParameter(
                    name="keyword", description="text indicating which timetable entry to look for"
                ),
            ),
        ),
    ),
    _FERRY_ROUND.demo,
    _FERRY_RUNS,
)

_BAKERY_DEMONSTRATED = _DemonstratedRound(
    url=_BAKERY_SPECIALS_URL,
    extract="today's special",
    collection="corner-bakery-daily-specials",
    # The measured draws all keyed a day's special by its date, and the label
    # transcribed below says so — so the demonstrated key is one, which is exactly the
    # kind of value a placeholder exists to stop a collector re-writing every cycle.
    entry_key="2026-08-05",
    entry_value="rye and caraway loaf",
)
_BAKERY_SKILL = _fixture_skill(
    _BAKERY_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(
            name="specials_url",
            description="the URL that the routine should browse to retrieve the daily specials",
        ),
        extract=LeafLabel(
            name="what_to_find",
            description="natural-language description of what content to find",
        ),
        collection=LeafLabel(
            name="storage_collection",
            description="identifier of the collection where extracted daily specials are stored",
        ),
        entry_key=LeafLabel(
            name="entry_key",
            description="unique date-based key used to store each special in the collection",
        ),
        signature=SkillSignature(
            name="fetch_daily_special",
            description="retrieve the daily special from a bakery webpage",
            parameters=(
                FramedParameter(
                    name="url", description="the URL where the daily specials are listed"
                ),
            ),
        ),
    ),
    _BAKERY_ROUND.demo,
    _BAKERY_RUNS,
)

_COLONY_DEMONSTRATED = _DemonstratedRound(
    url=_COLONY_COUNT_URL,
    extract="the current count",
    collection="harbor-seals",
    entry_key="current",
    entry_value="214",
)
_COLONY_SKILL = _fixture_skill(
    _COLONY_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="page_url", description="the URL to visit each run"),
        extract=LeafLabel(
            name="value_to_find", description="the text or data point to pull from that page"
        ),
        collection=LeafLabel(
            name="storage_collection",
            description="the name of the collection where results are stored",
        ),
        entry_key=LeafLabel(
            name="entry_key", description="the key used for the entry in the collection"
        ),
        signature=SkillSignature(
            name="monitor_webpage_number",
            description="track a numeric value on a webpage over time to detect changes",
            parameters=(FramedParameter(name="url", description="the webpage to monitor"),),
        ),
    ),
    _COLONY_ROUND.demo,
    _COLONY_RUNS,
)

_ARRIVALS_DEMONSTRATED = _DemonstratedRound(
    url=_NEW_ARRIVALS_URL,
    extract="the newest arrival",
    collection="library-new-arrivals",
    entry_key="newest arrival",
    entry_value="The Tidewater Almanac",
)
_ARRIVALS_SKILL = _fixture_skill(
    _ARRIVALS_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="page_url", description="the URL of the webpage to monitor each run"),
        extract=LeafLabel(
            name="value_to_find",
            description=(
                "instruction on how to extract the title of the most recent arrival from that page"
            ),
        ),
        collection=LeafLabel(
            name="storage_collection",
            description="identifier for the collection where new-arrivals entries are stored",
        ),
        entry_key=LeafLabel(
            name="entry_key",
            description="unique identifier for the arrival, typically its title",
        ),
        signature=SkillSignature(
            name="retrieve_newest_item",
            description="Checks a web page and returns its newest arrival",
            parameters=(FramedParameter(name="url", description="the URL of the list to check"),),
        ),
    ),
    _ARRIVALS_ROUND.demo,
    _ARRIVALS_RUNS,
)


# The DECOY, seeded in every case beside the scenario's own: a real routine of the same
# kind, about something else entirely.  It is what makes a wrong binding REACHABLE — a
# name nobody taught is refused by membership validation before the turn ever sees it,
# so without an existing-but-wrong alternative the intended-skill check could only ever
# pass.  Built through the same pipeline over the same canonical shape, so it is a
# skill of the same standing rather than a rigged one.
_DECOY_DEMONSTRATED = _DemonstratedRound(
    url="https://citymuseum.example/hours",
    extract="the opening times",
    collection="museum-hours",
    entry_key="opening times",
    entry_value="10am to 5pm, closed Mondays",
)
_DECOY_SKILL = _fixture_skill(
    _DECOY_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="hours_url", description="the URL of the hours page to read each run"),
        extract=LeafLabel(
            name="value_to_find", description="a plain description of what to pull off that page"
        ),
        collection=LeafLabel(
            name="storage_collection",
            description="the name of the collection where the opening times are kept",
        ),
        entry_key=LeafLabel(
            name="entry_key", description="the key under which the opening times are stored"
        ),
        signature=SkillSignature(
            name="check_museum_hours",
            description="read a museum's hours page and record the opening times",
            parameters=(
                FramedParameter(name="url", description="the URL of the museum hours page"),
            ),
        ),
    ),
    "go to https://citymuseum.example/hours, find the opening times, and remember them",
    _DECOY_RUNS,
)


def learn_to_apply_fixture_skill() -> SkillDraft:
    """The auction script's own fixture skill — case 1's, and the one a plain
    (no-GPU) test pins so a distiller, labeller or framer change that reshapes it
    fails there rather than quietly handing the live cases an easier — or
    impossible — starting world."""
    return _AURORA_SKILL


class _ApplyCase(NamedTuple):
    """One agreed learn → apply acceptance, and the round its turn is answered against.

    ``prior`` is the sibling elicit → learn case, whose ask, teach question,
    demonstration and closing report are seeded as the round's four logged turns.
    ``demonstrated`` is the canonical ledger that round ran, which is both what the
    fixture ``skill`` was distilled from and what the naive collection seeded here
    holds.  ``acceptance`` is the turn under test.

    The rest is what the acceptance's own terms ask for, per case: ``cadence_seconds`` is
    how far apart the job should fire, whatever rule spelling says so, and ``anchored``
    whether the terms name a time of DAY rather than a period (so the rule has to state an
    hour to run at); ``expects_expiry`` is whether they gave an
    end condition at all (inventing one is a failure); ``bound`` is every value the
    round supplied that the routine has to be pointed at, matched case-folded so a
    normalized copy of a scheme-less address still counts; ``cadence_tokens`` is what a
    reply naming the cadence back would have to contain.

    ``confirmation`` is THIS case's reference reply — how she says the job is running — and
    it is DATA rather than prose for the same reason ``_LearnCase.closing_report`` is: the
    idle → apply world seeds each of these journeys to its end and replays this line as the
    turn that finished it, so the line the review reads and the line that world replays are
    one string.

    ``runs`` is the journey's own run-id bundle — what every seeded turn of this round is
    written under, and what its skill, collection, entry and moves cite."""

    case_id: str
    prior: _LearnCase
    demonstrated: _DemonstratedRound
    skill: SkillDraft
    runs: _JourneyRuns
    acceptance: str
    confirmation: str
    cadence_seconds: int
    anchored: bool
    expects_expiry: bool
    bound: tuple[str, ...]
    cadence_tokens: tuple[str, ...]


# Case 1 — the script's own turn, continuing ``transition-elicit-to-learn``: the offer
# taken up with a cadence, an end condition and a notify ask, and NOT the page — the
# round it answers already read that.
_AURORA_APPLY = _ApplyCase(
    case_id="transition-learn-to-apply",
    prior=_AURORA_ROUND,
    demonstrated=_AURORA_DEMONSTRATED,
    skill=_AURORA_SKILL,
    runs=_AURORA_RUNS,
    acceptance="perfect — do that every hour until 10pm tonight and tell me if it changes",
    confirmation=(
        "done — i'll check the listing every hour until 10pm tonight and message you if "
        "the price moves."
    ),
    cadence_seconds=3600,
    anchored=False,
    expects_expiry=True,
    bound=(LISTING_URL,),
    cadence_tokens=("hour", "60 min"),
)

# Case 2 — the ferry, whose terms are a time of DAY rather than a period, so the cron
# form is what answers them; and whose routine asks for two things, the page and what to
# look for on it, both of which the round's own turns supplied.
_FERRY_APPLY = _ApplyCase(
    case_id="transition-learn-to-apply-no-url",
    prior=_FERRY_ROUND,
    demonstrated=_FERRY_DEMONSTRATED,
    skill=_FERRY_SKILL,
    runs=_FERRY_RUNS,
    acceptance="great — do that every morning and let me know when the late sailing gets added",
    confirmation=(
        "done — i'll check the timetable every morning and message you when the late "
        "sailing shows up."
    ),
    cadence_seconds=86400,
    anchored=True,
    expects_expiry=False,
    bound=(_FERRY_TIMETABLE_URL, "late sailing"),
    cadence_tokens=("morning",),
)

# Case 3 — the store-each-day digest set running.  Nothing in the acceptance ends it,
# so an ``expires_at`` here is an end condition nobody asked for.
_BAKERY_APPLY = _ApplyCase(
    case_id="transition-learn-to-apply-digest",
    prior=_BAKERY_ROUND,
    demonstrated=_BAKERY_DEMONSTRATED,
    skill=_BAKERY_SKILL,
    runs=_BAKERY_RUNS,
    acceptance="great — do that every day and tell me what the special is",
    confirmation="done — i'll check the specials every day and message you what's on.",
    cadence_seconds=86400,
    anchored=False,
    expects_expiry=False,
    bound=(_BAKERY_SPECIALS_URL,),
    cadence_tokens=("day", "daily"),
)

# Case 4 — the weekly count.  The page arrives in the user's own scheme-less form, which
# is what the bound value is matched on (a normalized copy contains it either way).
# Notify fires on any change at the write gate, so the directional "if it drops" is
# future business and is not scored here.
_COLONY_APPLY = _ApplyCase(
    case_id="transition-learn-to-apply-threshold",
    prior=_COLONY_ROUND,
    demonstrated=_COLONY_DEMONSTRATED,
    skill=_COLONY_SKILL,
    runs=_COLONY_RUNS,
    acceptance="perfect — do that every week and let me know if the count drops",
    confirmation="done — i'll check the colony count every week and message you if it drops.",
    cadence_seconds=604800,
    anchored=False,
    expects_expiry=False,
    bound=("harborseals.example/colony-count",),
    cadence_tokens=("week",),
)

# Case 5 — the tight cadence with an end condition the model has to WORK OUT: "the end
# of the month" is a date nobody states, so the check is that an expiry was set at all
# and the drawn value rides advisory for review.
_ARRIVALS_APPLY = _ApplyCase(
    case_id="transition-learn-to-apply-urgency",
    prior=_ARRIVALS_ROUND,
    demonstrated=_ARRIVALS_DEMONSTRATED,
    skill=_ARRIVALS_SKILL,
    runs=_ARRIVALS_RUNS,
    acceptance=(
        "yes — check it every two hours until the end of the month and tell me the "
        "second something new appears"
    ),
    confirmation=(
        "done — i'll check the new-arrivals page every two hours until the end of the "
        "month and message you the moment something new shows up."
    ),
    cadence_seconds=7200,
    anchored=False,
    expects_expiry=True,
    bound=(_NEW_ARRIVALS_URL,),
    cadence_tokens=("two hours", "2 hours", "120 min"),
)


# The round's incoming turns by scoring time: the ask and the demonstration this seed
# lays down, and the acceptance the channel logs AFTER the run (#1566's deferred link).
_APPLY_ROUND_INCOMING_TURNS = 3

# Every apply case, in one place — so the deterministic pin in ``test_eval_harness.py`` can
# drive each one's seeder without a GPU.  The seeder and its two ledger probes are the only
# part of this beat that has to be exercised before a run costs an hour: everything else in
# a case is data, but a seeder is code, and a seeder that raises fails five cases at once
# after the queue has already been taken.
APPLY_CASES = (_AURORA_APPLY, _FERRY_APPLY, _BAKERY_APPLY, _COLONY_APPLY, _ARRIVALS_APPLY)


def seed_learned_round(case: _ApplyCase) -> Seeder:
    """Lay down the whole round the acceptance answers, turn for turn — the two beats
    before this one, as they really happened:

    * the instigating ask INCOMING, Penny's teach question OUTGOING, the demonstration
      INCOMING, and her closing report OUTGOING — the offer this turn takes up
    * the machine's two moves: idle → elicit anchored to the ask, then elicit → learn
      carrying that anchor unchanged
    * the naive collection the demonstrated write created — the fact in it, no skill,
      no rendered program, nothing scheduled (learning instantiates nothing)

    The fixture skills and the case's page are laid down by the runner after this, so
    the world is only whole once they are — which is why the probe is a prepare hook.

    The work is ``seed_round_through_learn``'s, because the same four turns are the
    PREFIX of a completed journey the idle → apply world carries on past."""

    def seed(db: Database) -> None:
        seed_round_through_learn(db, case)

    return seed


def seed_round_through_learn(
    db: Database, case: _ApplyCase, *, taught_so_far: tuple[SkillCandidate, ...] = ()
) -> tuple[int, int]:
    """The round's first four turns and both of its moves, returning the ask's and the
    demonstration's message ids.

    Factored out of ``seed_learned_round`` because a journey is a PREFIX of a longer
    history: the learn → apply cases stop here (this is where their acceptance lands),
    and the idle → apply world carries each journey on through its apply turn.  One
    definition, so what the later beat replays is what the earlier beat is measured
    against rather than a second copy of it.

    ``taught_so_far`` is the registry as it stood when this round OPENED — empty for the
    first journey a world holds, the earlier journeys' routines for the ones after it,
    since the classifier is offered every skill that exists at the moment it draws."""
    ask_id, demo_id = _seed_round_turns(db, case)
    _seed_elicit_turn_ledger(db, case.prior, case.runs, candidates=taught_so_far)
    _park(
        db,
        ConversationState.ELICIT,
        anchor_message_id=ask_id,
        run_id=case.runs.elicit_turn,
        message_id=ask_id,
    )
    _seed_learn_turn_ledger(db, case, candidates=taught_so_far)
    _park(
        db,
        ConversationState.LEARN,
        anchor_message_id=ask_id,
        from_state=ConversationState.ELICIT,
        run_id=case.runs.learn_turn,
        message_id=demo_id,
    )
    _seed_naive_collection(db, case, demo_id)
    return ask_id, demo_id


def _seed_learn_turn_ledger(
    db: Database, case: _ApplyCase, *, candidates: tuple[SkillCandidate, ...] = ()
) -> None:
    """The elicit → learn turn's promptlog — the round itself, as the loop wrote it: the
    draw that chose learn, then the chat run's three steps (browse · write · the closing
    report), each carrying the accumulated conversation as it stood when the call was made
    so every call's framed result is stored beside the call it answers.  The browse spawns
    its own micro-context call, attributed the way production attributes it.

    This is what makes the round REACHABLE: the collection, its entry, the skill and the
    transition row all cite this run, so the mutation line's ``run <id>`` resolves to the
    browse and the write that actually happened."""
    _log_classifier_draw(
        db,
        run_id=case.runs.learn_draw,
        snapshot=MachineSnapshot(
            state=ConversationState.ELICIT,
            penny_last_turn=case.prior.teach_question,
            task_anchor=case.prior.ask,
            skill_candidates=list(candidates),
        ),
        message=case.prior.demo,
        drawn=_drawn_state(ConversationState.LEARN),
    )
    browse, write = _demonstrated_ledger(case.demonstrated)
    conversation: list[dict] = [{"role": "user", "content": case.prior.demo}]
    run_id = case.runs.learn_turn
    conversation = _seed_call_step(db, conversation, _BROWSE_CALL_ID, browse, run_id=run_id)
    _log_browse_extract(db, case.demonstrated, case.runs.browse_extract)
    conversation = _seed_call_step(db, conversation, _WRITE_CALL_ID, write, run_id=run_id)
    _log_chat_step(
        db,
        run_id=run_id,
        messages=conversation,
        response=_seeded_response(case.prior.closing_report),
    )


def _seed_call_step(
    db: Database, conversation: list[dict], call_id: str, step: DistillInput, *, run_id: str
) -> list[dict]:
    """One tool-calling step of the seeded run: log the call against the conversation as it
    stands, then return the conversation the NEXT call sees — the assistant's call turn plus
    its framed result, which is the only place a tool result is ever durably written."""
    call = _wire_tool_call(call_id, step)
    _log_chat_step(
        db,
        run_id=run_id,
        messages=conversation,
        response=_seeded_response(tool_calls=[call]),
    )
    return [
        *conversation,
        {"role": "assistant", "content": "", "tool_calls": [call]},
        _tool_result_turn(call_id, step),
    ]


def _log_browse_extract(db: Database, demonstrated: _DemonstratedRound, run_id: str) -> None:
    """The browse's own micro-context call — its own agent identity and its own run id,
    exactly as ``MicroContext`` writes it, so the seeded round's ledger carries the same
    actors a real one does."""
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=[{"role": "user", "content": demonstrated.extract}],
        response=_seeded_response(f"EXTRACTED: {demonstrated.entry_value}"),
        agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        prompt_type=PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
        run_id=run_id,
        run_target=PennyConstants.CHAT_AGENT_NAME,
    )


def _seed_round_turns(db: Database, case: _ApplyCase) -> tuple[int, int]:
    """The round's four logged messages, in the order they were said, returning the ids of
    the ask (the row the whole round is anchored to) and the demonstration (the message the
    learn move was provoked by, and the one the collection is sourced from).

    Each id is asserted the moment it is written rather than read back later: logging is
    best-effort, and a turn with no id would link its move, its mechanism and Penny's reply
    to nothing — which the probe would then report as a broken anchor lifecycle instead of
    a seed that never wrote."""
    ask_id = _log_ask(db, case.prior.ask, case.case_id)
    _log_reply(db, case.prior.teach_question, answering=ask_id)
    demo_id = _log_ask(db, case.prior.demo, case.case_id)
    _log_reply(db, case.prior.closing_report, answering=demo_id)
    return ask_id, demo_id


def _seed_naive_collection(db: Database, case: _ApplyCase, demo_id: int) -> None:
    """The collection the demonstrated write created, exactly as the AUTO-CREATE path
    creates it: a description naming what it was made to hold (the production format,
    imported rather than copied), stamped with the run that created it and linked to the
    message that provoked it — which is what puts a resolvable ``run <id>`` on the mutation
    line, since the create chokepoint records that event from this very argument.

    The entry carries the same run on both its write stamps, as a real write does."""
    demonstrated = case.demonstrated
    db.memories.create_collection(
        demonstrated.collection,
        _AUTO_CREATED_DESCRIPTION.format(key=demonstrated.entry_key),
        created_by_run_id=case.runs.learn_turn,
    )
    db.memories.link_source_message(case.runs.learn_turn, demo_id)
    require_memory(db, demonstrated.collection).write(
        [EntryInput(key=demonstrated.entry_key, content=demonstrated.entry_value)],
        author=PennyConstants.CHAT_AGENT_NAME,
        run_id=case.runs.learn_turn,
    )
    _seed_browse_results_entry(db, case)


def _seed_browse_results_entry(db: Database, case: _ApplyCase) -> None:
    """The page the round fetched, in the browse-results log — one entry per read page, its
    content the rendered section (the header line and the page body), exactly what the
    browse tool appends, and the page the case's own fixture serves.

    Deliberately UNSTAMPED: the browse tool calls ``append(entries, author=…)`` with no
    run id, so a stamped row here would be one production cannot write."""
    header = PennyConstants.BROWSE_PAGE_HEADER
    section = f"{header}{case.demonstrated.url}\n{case.prior.page.text}"
    require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).append(
        [LogEntryInput(content=section)],
        author=PennyConstants.CHAT_AGENT_NAME,
    )


def _probe_seeded_world(case: _ApplyCase) -> Preparer:
    """The loud seed probe, run once the world is WHOLE.

    It is a prepare hook rather than part of the seeder because the runner seeds fixture
    skills and installs the browse AFTER the case's own seed — so "exactly two skills"
    and "no page fetched" are only true here.  A seed that has drifted from the state
    the preceding beats are measured against makes these cases turns answered against a
    world nothing produces, so it fails HERE rather than as a puzzling number after an
    hour of GPU time."""

    def probe(penny: Penny) -> None:
        _assert_parked_on_the_ask(penny.db, case)
        _assert_seeded_registry(penny.db, case)
        assert_seeded_ledger(penny.db, case)
        assert_round_cites_its_run(penny.db, case)

    return probe


def assert_seeded_ledger(db: Database, case: _ApplyCase) -> None:
    """The preceding turns are IN the ledger, with the calls they made: the learn run
    carries the round's two calls in order, and the page it read is in browse-results.

    This is the half a durable-row-only seed had nothing of, and the half a live model
    reads when it asks what has already been done — so a drift here is the case quietly
    reverting to the impoverished world it was built to replace."""
    assert_round_calls_logged(db, case)
    fetched = _pages_fetched(db)
    assert len(fetched) == 1, (
        f"{case.case_id}: the round read one page, browse-results has {fetched}"
    )


def assert_round_calls_logged(db: Database, case: _ApplyCase) -> None:
    """One round's demonstrated calls, read back off its OWN run — the half that holds
    however many rounds a world carries, since a run id scopes the read."""
    calls = [
        call.get("function", {}).get("name")
        for row in db.messages.get_run_prompts(case.runs.learn_turn)
        for call in _row_tool_calls(row)
    ]
    expected = [step.tool for step in _demonstrated_ledger(case.demonstrated)]
    assert calls == expected, f"{case.case_id}: the seeded round must carry {expected}, got {calls}"


def _row_tool_calls(row) -> list[dict]:
    """The tool calls of one persisted promptlog row — the same walk every ledger reader
    makes over the stored response envelope."""
    response = json.loads(row.response) if row.response else {}
    return [
        call
        for choice in response.get("choices", [])
        for call in (choice.get("message") or {}).get("tool_calls") or []
    ]


def assert_round_cites_its_run(db: Database, case: _ApplyCase) -> None:
    """Everything the round produced names the run that produced it — the collection, its
    entry, the skill, and the transition row.  That citation is what makes the mutation
    line's ``run <id>`` resolve to the browse and the write that actually happened, which
    is the whole reason the ledger is seeded at all."""
    assert_round_rows_cite_their_run(db, case)
    latest = db.machine.latest_transition()
    assert latest is not None and latest.run_id == case.runs.learn_turn, (
        f"{case.case_id}: the learn move must cite the round's run, not {latest and latest.run_id}"
    )


def assert_round_rows_cite_their_run(db: Database, case: _ApplyCase) -> None:
    """The durable half of that citation — the collection and its entry.  Split from the
    machine assertion above because where the MACHINE stands depends on how far the world
    carried this round, while what the round WROTE cites its run either way."""
    row = db.memories.get(case.demonstrated.collection)
    created_by = row.created_by_run_id if row else None
    assert created_by == case.runs.learn_turn, (
        f"{case.case_id}: the collection must cite the round's run, not {created_by}"
    )
    assert row is not None
    assert row.source_message_id is not None, (
        f"{case.case_id}: the collection must be linked to the message that provoked it"
    )
    entries = require_memory(db, case.demonstrated.collection).read_all()
    stamps = {entry.created_by_run_id for entry in entries}
    assert stamps == {case.runs.learn_turn}, (
        f"{case.case_id}: the demonstrated entry must cite the round's run, got {stamps}"
    )


def _assert_parked_on_the_ask(db: Database, case: _ApplyCase) -> None:
    """The machine is parked in learn ON THE ASK — the anchor stamped entering the round
    and carried through it, which is what the acceptance is classified against."""
    ask_id = _seeded_ask_id(db, case.prior.ask, limit=_APPLY_ROUND_INCOMING_TURNS)
    assert ask_id is not None, f"{case.case_id}: the seeded ask must be findable by its content"
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.LEARN.value, (
        f"{case.case_id}: the machine must be parked in learn, not {latest}"
    )
    assert latest.anchor_message_id == ask_id, (
        f"{case.case_id}: the round must stay anchored to the ask, not {latest.anchor_message_id}"
    )


def _assert_seeded_registry(db: Database, case: _ApplyCase) -> None:
    """The other half of that probe: two skills and only two — the scenario's own and
    the decoy — the demonstrated fact in the collection it was written to, and nothing
    instantiated anywhere.  (What the round DID read is asserted next, not here: this
    beat starts from a page already fetched.)"""
    taught = sorted(skill.name for skill in db.skills.list_all())
    assert taught == sorted([case.skill.name, _DECOY_SKILL.name]), (
        f"{case.case_id}: the registry must hold the scenario's skill and the decoy, got {taught}"
    )
    stored = collection_entries(db, case.demonstrated.collection)
    assert stored.get(case.demonstrated.entry_key) == case.demonstrated.entry_value, (
        f"{case.case_id}: the demonstrated fact must be in the collection, got {stored}"
    )
    instantiated = [row.name for row in db.memories.list_all() if row.skill_name is not None]
    assert not instantiated, f"{case.case_id}: nothing is instantiated yet, found {instantiated}"


def _instantiated(db: Database, case: _ApplyCase) -> MemoryRow | None:
    """The collection the taught skill was applied to — WHICHEVER one she chose.

    Which collection a job lands on is deliberately NOT these cases' business (code
    owner): she has created several where one was meant since well before the machine
    existed, so where jobs accumulate is a collection-management question of its own and
    grading it here would report that standing problem as a transition failure.  This
    edge owns whether the skill is APPLIED correctly — bound, rendered, and scheduled on
    the terms given — so every check reads the row that carries the skill, and the one
    about reuse rides along unscored."""
    taught = slug_skill_name(case.skill.name)
    applied = [row for row in db.memories.list_all() if row.skill_name == taught]
    return applied[0] if applied else None


def _bound_parameters(row: MemoryRow) -> dict[str, str]:
    """The values she bound into the skill at instantiation, from the collection's
    own provenance column (#1603) — a read, not an inference.

    Only a column that was never written reads as no bindings; a blank one is a store
    defect and raises here rather than scoring as the model having bound nothing."""
    if row.skill_params is None:
        return {}
    return {key: str(value) for key, value in json.loads(row.skill_params).items()}


def _landed_apply_move(landed: StateTransition | None) -> StateTransition | None:
    """The turn's last move, but only when it put the machine in APPLY.

    One predicate, read once per sample and shared by everything that conditions on it —
    the two conditional-n/a checks below and the landed-state advisory — so they can
    never disagree about where the turn ended up, which is what a per-check re-read of
    the ledger would eventually let them do."""
    if landed is None or landed.to_state != ConversationState.APPLY.value:
        return None
    return landed


def _skill_binding_check(landed: StateTransition | None, *, intended: str, label: str) -> Check:
    """The decision bound the INTENDED routine — the landed transition's ``skill_name`` is
    the one that covers what was asked for, not another routine in the registry.

    Scored only when the machine landed in apply: a misroute is already the landed-state
    advisory's finding, and scoring the binding on top of it would recount one classifier
    miss twice.

    ``label`` is the caller's because a label is a diff-join key: two beats ask this same
    question of two different situations and each names it in its own terms, while the
    reading itself must be one definition."""
    applied = _landed_apply_move(landed)
    if applied is None:
        return Check.na(label, kind="state")
    bound = applied.skill_name
    return Check(
        label,
        bound == intended,
        rationale=None if bound == intended else f"bound {bound!r}, the ask needs {intended!r}",
        kind="state",
    )


def _apply_anchor_check(db: Database, landed: StateTransition | None, case: _ApplyCase) -> Check:
    """The anchor was CARRIED: the move that landed the machine in apply still points at
    the ask that opened the round (#1827) — three messages later, the turn is still
    classified against what was asked for.  Same conditional-n/a as the binding check."""
    label = "state: the anchor was carried (still the ask that opened the round)"
    applied = _landed_apply_move(landed)
    if applied is None:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, case.prior.ask, limit=_APPLY_ROUND_INCOMING_TURNS)
    anchored = applied.anchor_message_id
    carried = asked is not None and anchored == asked
    return Check(
        label,
        carried,
        rationale=None if carried else f"anchored to {anchored}, the ask is {asked}",
        kind="state",
    )


# The rule part that ANCHORS a recurrence to a time of day.  Read as a PART of the stored
# rule rather than off the parsed object, because dateutil defaults an unstated hour to the
# start's — so the parsed rule cannot tell a stated hour from an inherited one, and only the
# text says whether the model chose one.
_HOUR_PART = "BYHOUR"

# Where a rule with no ``DTSTART`` of its own is anchored for measurement.  Any fixed instant
# does: the cadence is the GAP between occurrences, and a gap does not move with the anchor.
_MEASURING_ANCHOR = datetime(2000, 1, 1, tzinfo=UTC)

# How many occurrences a gap needs.  Two — a rule that can only fire once (COUNT=1) has no
# cadence to read, which is a real shape and reads as no cadence rather than as an error.
_OCCURRENCES_FOR_A_GAP = 2


def _rule_body(schedule: str) -> str:
    """The stored schedule's RULE line — the ``DTSTART`` line dropped and any ``RRULE:``
    tag stripped.  A schedule renders on one line with its newline written ``\\n`` (the form
    the parser accepts back), so both spellings are unfolded first."""
    lines = [line for line in schedule.replace(_LINE_ESCAPE, "\n").splitlines() if line.strip()]
    body = next((line for line in reversed(lines) if not line.upper().startswith(_DTSTART_TAG)), "")
    return body[len(_RRULE_TAG) :] if body.upper().startswith(_RRULE_TAG) else body


def rule_parts(schedule: str) -> set[str]:
    """Which PARTS the stored rule states, by name — the declared shape, read structurally
    off the rule rather than by comparing its spelling to one we had in mind."""
    return {part.partition("=")[0].strip().upper() for part in _rule_body(schedule).split(";")}


def cadence_seconds(schedule: str) -> int | None:
    """How often the stored rule FIRES, in seconds — the gap between its first two
    occurrences, measured by walking the rule itself.

    Reading the gap rather than the FREQ/INTERVAL pair is what makes the check answer the
    question the acceptance asked ("every day") instead of a question about spelling: a
    daily cadence written ``FREQ=DAILY``, ``FREQ=HOURLY;INTERVAL=24``, or
    ``FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR,SA,SU`` all fire a day apart, and all three are the
    same answer.  ``None`` when the rule fires at most once."""
    text = schedule.replace(_LINE_ESCAPE, "\n")
    anchored = _DTSTART_TAG in text.upper()
    rule = rrulestr(text) if anchored else rrulestr(text, dtstart=_MEASURING_ANCHOR)
    occurrences = list(islice(iter(rule), _OCCURRENCES_FOR_A_GAP))
    if len(occurrences) < _OCCURRENCES_FOR_A_GAP:
        return None
    return int((occurrences[1] - occurrences[0]).total_seconds())


def _schedule_check(row: MemoryRow | None, *, fires_every: int, anchored: bool) -> Check:
    """The schedule runs on the cadence they asked for — and, where the terms name a time of
    DAY rather than a period, states an hour to run at.

    WHICH hour is not scored (the code owner's leeway ruling, carried over from the cron
    form it replaced): picking a sensible morning is expected of the model, but the hour it
    picks is a judgment about wording, and the drawn rule rides advisory for review.

    ``fires_every`` and ``anchored`` are the terms, passed as values rather than read off a
    case, so both beats that ask this question share one reading of a stored rule."""
    label = "state: the schedule runs on the cadence they asked for"
    if row is None or row.schedule is None:
        return Check(label, False, rationale="no schedule was set", kind="state")
    drawn = cadence_seconds(row.schedule)
    states_an_hour = _HOUR_PART in rule_parts(row.schedule)
    matches = drawn == fires_every and (states_an_hour or not anchored)
    return Check(
        label,
        matches,
        rationale=None if matches else f"fires every {drawn}s, states {rule_parts(row.schedule)}",
        kind="state",
    )


def _expiry_check(row: MemoryRow | None, *, expected: bool) -> Check:
    """The end condition matches the terms — set when they gave one, ABSENT when they
    did not.  An invented end condition is a failure in its own right: a job that stops
    on a date nobody asked for goes quiet without anyone noticing."""
    label = "state: the end condition matches the terms (set only when they gave one)"
    if row is None:
        return Check(label, False, rationale="no job carries the routine", kind="state")
    given = row.expires_at is not None
    return Check(
        label,
        given == expected,
        rationale=None if given == expected else f"expires {row.expires_at}",
        kind="state",
    )


def _overlaps(wanted: str, bound: list[str]) -> bool:
    """Whether any bound value and the expected phrase OVERLAP — either one containing the
    other, case-folded, with neither empty.

    Containment runs BOTH ways (code-owner ruling).  One way only asked whether the bound
    value repeated the whole expected phrase, which fails a shorter value that locates the
    same thing: told to watch for the dawn sailing, a routine bound to `dawn` finds exactly
    the line `dawn sailing` would, and calling that a miss scores a wording preference as a
    binding failure.  Every spelling that locates it passes — `dawn`, `Dawn`, `dawn
    sailing`, `Dawn Sailing` — and an unrelated value still fails, since it overlaps in
    neither direction.

    What makes the loosened rule still MEAN something is the world it runs in: the probe
    has already established that the expected phrase appears nowhere in the seeded history,
    so a value overlapping it can only have been read off the message.  That is why the
    same rule is safe on an address as on a phrase — a partial address is still a partial
    address nobody could have copied from anywhere else.  An EMPTY bound value is excluded
    outright: it is a substring of everything and evidence of nothing."""
    expected = wanted.lower()
    return any(
        value.strip() and (value.lower() in expected or expected in value.lower())
        for value in bound
    )


def _bound_parameters_check(row: MemoryRow | None, *, wanted: tuple[str, ...], label: str) -> Check:
    """EVERY parameter the routine asks for was bound — the page it is pointed at, and
    where the routine asks for two things, what to look for on it as well.

    Matched case-folded and by overlap in either direction (``_overlaps``): an address may
    arrive without a scheme, and a phrase may be bound by the word that locates it.
    ``label`` is the caller's — WHERE a value came from is what each beat is measuring, and
    that is what its label says.  The bound values themselves ride VERBATIM in the drawn
    advisory, so what a looser match accepted is always visible."""
    bound = list(_bound_parameters(row).values()) if row is not None else []
    missing = [value for value in wanted if not _overlaps(value, bound)]
    return Check(
        label,
        not missing,
        rationale=None if not missing else f"bound {bound}, missing {missing}",
        kind="state",
    )


def _decoy_check(db: Database) -> Check:
    """The decoy was left alone — no collection carries it.  Kept separate from the
    intended-skill checks so a report distinguishes "bound the wrong routine" from
    "bound nothing at all"."""
    slug = slug_skill_name(_DECOY_SKILL.name)
    applied = [row.name for row in db.memories.list_all() if row.skill_name == slug]
    return Check(
        "state: the decoy was not applied",
        not applied,
        rationale=f"applied to {applied}" if applied else None,
        kind="state",
    )


def _drawn_advisories(row: MemoryRow | None) -> list[Check]:
    """What she committed to, verbatim — the trigger clause in its copyable input form,
    the end condition where one was set, and the parameters she bound.

    Whether a drawn cron expression or a computed month's-end datetime is WELL judged is
    read at joint review against the reference replies; a scorer that faked that reading
    would be answering for the draw."""
    if row is None:
        return []
    drawn: list[Check] = []
    if has_schedule(row):
        drawn.append(
            Check(
                f"drew schedule {render_schedule_clause(row)!r}", True, scored=False, kind="state"
            )
        )
    if row.expires_at is not None:
        drawn.append(
            Check(
                f"drew end condition {row.expires_at.isoformat()}",
                True,
                scored=False,
                kind="state",
            )
        )
    bound = _bound_parameters(row)
    if bound:
        drawn.append(Check(f"bound parameters {bound}", True, scored=False, kind="state"))
    return drawn


def _score_learn_to_apply(
    db: Database, before: set[str], reply: str, *, case: _ApplyCase
) -> list[Check]:
    """The taught routine became a live job on the terms they gave — the intended skill
    bound, its program rendered, every parameter taken from the round, scheduled and
    notifying — without re-running the round to answer.

    ONE scorer for all five cases, bound to the case's own terms.  The labels are
    diff-join keys, so they read identically on every case and keep the wording the
    auction script gave them even where a ferry timetable is what the job watches.
    WHERE the job lives is not scored (see ``_instantiated``); it rides along as an
    advisory so the choice stays visible."""
    row = _instantiated(db, case)
    landed = db.machine.latest_transition()
    return [
        *_binding_checks(db, row, landed, case),
        *_terms_checks(db, row, case),
        _decoy_check(db),
        _apply_anchor_check(db, landed, case),
        Check(
            "reply: she says what will happen now, naming the cadence",
            any(token in reply.lower() for token in case.cadence_tokens),
            kind="reply",
        ),
        *_apply_advisories(db, before, row, landed, case),
    ]


def _binding_checks(
    db: Database, row: MemoryRow | None, landed: StateTransition | None, case: _ApplyCase
) -> list[Check]:
    """She set a job up, on the right routine, pointed at what the round supplied."""
    return [
        Check(
            "state: she set the job up with collection_set",
            tool_was_called(db, _SET_TOOL),
            kind="state",
        ),
        Check(
            "state: the taught skill was applied to a collection",
            row is not None,
            rationale=None if row else "no collection carries the skill",
            kind="state",
        ),
        Check(
            "state: the skill's program was rendered into it",
            row is not None and bool(row.extraction_prompt),
            kind="state",
        ),
        _skill_binding_check(
            landed,
            intended=slug_skill_name(case.skill.name),
            label="state: the decision bound the intended skill",
        ),
        _bound_parameters_check(
            row, wanted=case.bound, label="state: every parameter bound from the round"
        ),
    ]


def _terms_checks(db: Database, row: MemoryRow | None, case: _ApplyCase) -> list[Check]:
    """The job runs on the terms the acceptance gave — its cadence, its end condition
    (or the absence of one), and the telling-them clause every one of these asks
    carries — and she set it running instead of running it again."""
    return [
        _schedule_check(row, fires_every=case.cadence_seconds, anchored=case.anchored),
        _expiry_check(row, expected=case.expects_expiry),
        Check(
            "state: it will tell them when the price moves",
            row is not None and bool(row.notify),
            kind="state",
        ),
        Check(
            "state: she set it running instead of running it again (no browse this turn)",
            tool_not_called(db, "browse"),
            kind="state",
        ),
    ]


def _reuse_advisory(
    db: Database, before: set[str], row: MemoryRow | None, case: _ApplyCase
) -> Check:
    """The collection-management question, parked (code owner): does the job land on the
    collection the round already wrote into, or on a new one?  Visible every run, graded
    never, so the standing tendency to spread across collections is measured here
    without this edge answering for it."""
    created = new_collections(db, before)
    reused = row is not None and row.name == case.demonstrated.collection
    return Check(
        "state: applied onto the collection the round wrote into (not a new one)",
        reused,
        rationale=(
            None
            if reused
            else (
                f"applied to {row.name if row else None}, created {[each.name for each in created]}"
            )
        ),
        scored=False,
        kind="state",
    )


def _apply_advisories(
    db: Database,
    before: set[str],
    row: MemoryRow | None,
    landed: StateTransition | None,
    case: _ApplyCase,
) -> list[Check]:
    """The unscored flavour: the parked collection-management question, then everything
    any apply turn reports about itself."""
    return [_reuse_advisory(db, before, row, case), *_job_setup_advisories(db, row, landed)]


def _job_setup_advisories(
    db: Database, row: MemoryRow | None, landed: StateTransition | None
) -> list[Check]:
    """What a turn that stood a job up committed to, and how the state came to be — the
    drawn schedule, end condition and bindings verbatim, one set call, the landed state,
    and clean routing.  Shared by both apply beats: an advisory says the same thing about
    an accepted offer and about a cold ask, and two copies would drift."""
    sets = count_tool_calls(db, _SET_TOOL)
    return [
        *_drawn_advisories(row),
        Check(
            "calls: one collection_set call",
            sets == 1,
            rationale=f"{sets} calls" if sets != 1 else None,
            scored=False,
            kind="proc",
        ),
        Check(
            "calls: the machine landed in apply",
            _landed_apply_move(landed) is not None,
            rationale=f"landed in {landed.to_state if landed else None}",
            scored=False,
            kind="spine",
        ),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


async def _run_apply_case(chat_eval: ChatEval, case: _ApplyCase) -> None:
    """Drive one learn → apply case: parked in learn on its own ask with the whole round
    behind it, its skill and the decoy in the registry, its page installed as a live
    temptation, and the shared scorer bound to the terms its acceptance gives.
    Report-only — the thresholds are the code owner's to set once the numbers are
    read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.acceptance,
        browse=[case.prior.page],
        seed=seed_learned_round(case),
        seed_skills=[case.skill, _DECOY_SKILL],
        prepare=_probe_seeded_world(case),
        score=partial(_score_learn_to_apply, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_learn_to_apply_instantiates_the_taught_skill(chat_eval: ChatEval) -> None:
    """learn → apply: parked on the offer the demonstrated round ended with, the
    user accepts and adds the job's terms.  She binds the taught skill onto the
    collection that round already wrote into — one `collection_set`, the page
    taken from the round rather than asked for again — and does NOT re-run the
    round to answer."""
    await _run_apply_case(chat_eval, _AURORA_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_sets_a_cron_cadence_and_binds_both_parameters(
    chat_eval: ChatEval,
) -> None:
    """learn → apply on a routine that asks for two things and a cadence stated as a
    time of day: "every morning" is the cron form, and both the timetable's address and
    the late sailing line the user named have to come out of the round's own turns —
    neither is in the acceptance."""
    await _run_apply_case(chat_eval, _FERRY_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_schedules_the_daily_digest(chat_eval: ChatEval) -> None:
    """learn → apply on the store-each-day digest: a plain daily cadence with a telling
    clause and NO end condition — so an expiry set here is one nobody asked for."""
    await _run_apply_case(chat_eval, _BAKERY_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_schedules_the_weekly_check(chat_eval: ChatEval) -> None:
    """learn → apply on the weekly count: the longest cadence of the set, bound to the
    scheme-less address the user typed rather than to a page she went and found."""
    await _run_apply_case(chat_eval, _COLONY_APPLY)


@pytest.mark.asyncio
async def test_learn_to_apply_sets_the_tight_cadence_and_works_out_the_end(
    chat_eval: ChatEval,
) -> None:
    """learn → apply under the act-now ask: a two-hourly cadence and an end condition
    stated in words — "until the end of the month" is a datetime the model has to work
    out, so what matters here is that it set one at all."""
    await _run_apply_case(chat_eval, _ARRIVALS_APPLY)


# ── idle → apply: a cold ask points a KNOWN routine at a new space ────────────
#
# The edge the whole journey exists to enable: the second time the user asks for
# something Penny already knows, nothing should need teaching.  Five cases, one per
# journey the three edges above walk — each returning user names a NEW space and the
# job's terms, and exactly one of the five taught routines covers it.
#
# It is NOT the moment after an apply turn.  The world is the full exit state of all
# five journeys — every message, every ledger row, the demonstrated entries, the pages
# they read, and the five APPLIED jobs those journeys left running — then a few
# exchanges of ordinary small talk, so what the conversation window shows last is a
# user who has been chatting rather than configuring.  The machine is idle.
#
# The asks are COLD: no "too", no "again", no "like the other one".  Nothing in the
# message points at the history, so binding the right routine is a decision about what
# the ask MEANS against five candidates — which is the first time that selection is
# fully load-bearing, since every wrong pick is a real routine that exists.
#
# Each case installs its new page as a live temptation: re-running the routine instead
# of setting it up is a fetch that would SUCCEED and get caught.  The values the ask
# supplies exist NOWHERE in the seeded history (the probe asserts it), so a parameter
# bound to one of them can only have been read off the message.
#
# The reference replies quoted above each ask are review targets, never scorer strings.


# ── The five new spaces ───────────────────────────────────────────────────────
#
# Catalogue-grade pages: each carries far more than its task needs — specifications,
# neighbouring items, housekeeping notes — because a real page does, and a fixture thin
# enough to answer only the asked question cannot tell restraint from luck.  Every
# markdown link sits at the CENTRE of its block: a SEARCH-shaped read is trimmed to ±2
# lines around each solo link (``_trim_search_result``), so a block laid out any other
# way would lose the very fields the page was written to carry.

_KEEL_LANTERN_URL = "https://faux-market.example/keel-lantern"
_KEEL_LANTERN_LISTING = CannedPage(
    match="keel",
    text=(
        "Title: Keel Lantern — brass storm lantern | faux-market\n"
        f"{_KEEL_LANTERN_URL}\n"
        "\n"
        "Sold by a fictional chandlery; stock and price move with the season.\n"
        "Price: $128\n"
        f"[Keel Lantern listing]({_KEEL_LANTERN_URL})\n"
        "Seller: harbourside_supply (4.7 stars). Price last changed eleven days ago.\n"
        "Condition: new, boxed. Dispatched within two working days.\n"
        "\n"
        "Specification\n"
        "Brass body with a weighted base, 340mm tall\n"
        f"[Specification sheet]({_KEEL_LANTERN_URL}/spec)\n"
        "Takes paraffin or a 3W bulb insert; the glass is replaceable.\n"
        "Weight 1.4kg · Height 340mm · Base 120mm\n"
        "\n"
        "Others from this seller\n"
        "Deck Lantern, galvanised — $96\n"
        "[Deck Lantern](https://faux-market.example/deck-lantern)\n"
        "Anchor Lamp, copper — $210\n"
        "Both ship from the same fictional warehouse.\n"
        "\n"
        "Returns are accepted within thirty days; return postage is the buyer's.\n"
    ),
)

_NORTH_PIER_URL = "https://northpier.example/departures"
# Matched on "pier", the token the ask and the address SHARE — the ask says "north pier"
# and the host is "northpier", so a search and a direct read both land on this page.
_NORTH_PIER_DEPARTURES = CannedPage(
    match="pier",
    text=(
        "Title: North Pier departures — sailing board | northpier\n"
        f"{_NORTH_PIER_URL}\n"
        "\n"
        "The departure board for a fictional pier, republished whenever it changes.\n"
        "Sailings today: 07:05, 09:40, 12:15, 15:50, 18:30, 21:10\n"
        f"[North Pier departure board]({_NORTH_PIER_URL})\n"
        "Dawn sailing: not on the board this season.\n"
        "Board last amended four days ago by the harbour office.\n"
        "\n"
        "Crossing times\n"
        "The outward crossing takes fifty minutes in fair weather\n"
        f"[Crossing times and fares]({_NORTH_PIER_URL}/fares)\n"
        "Return sailings leave the far side forty minutes after arrival.\n"
        "Foot passengers board ten minutes before departure.\n"
        "\n"
        "Notices\n"
        "The 12:15 runs to a reduced timetable out of season\n"
        f"[Seasonal notices]({_NORTH_PIER_URL}/notices)\n"
        "Cancellations are posted here and at the pier head.\n"
        "The board is the authority; printed cards may lag a day behind.\n"
    ),
)

_HARBOR_BAKERY_URL = "https://harborbakery.example/menu"
# Matched on the host rather than on "bakery": the ask NAMES this page, so a direct read
# is the temptation, and a token broad enough to also catch a search would answer a read
# of the SEEDED corner-bakery page with this page's body.
_HARBOR_BAKERY_MENU = CannedPage(
    match="harborbakery",
    text=(
        "Title: Harbor Bakery — menu and daily special | harborbakery\n"
        f"{_HARBOR_BAKERY_URL}\n"
        "\n"
        "Baked through the night in a fictional kitchen and posted before opening.\n"
        "Today's special: pear and walnut tart\n"
        f"[Harbor Bakery menu]({_HARBOR_BAKERY_URL})\n"
        "The special changes daily and yesterday's comes down at closing.\n"
        "Opening hours: 06:30 until sold out, closed Mondays.\n"
        "\n"
        "Always on the counter\n"
        "Sourdough, seeded rye, olive loaf, plain and salted butter rolls\n"
        f"[Standing menu]({_HARBOR_BAKERY_URL}/standing)\n"
        "Loaves are half price in the last hour of trading.\n"
        "Whole cakes need two days' notice.\n"
        "\n"
        "Orders and allergens\n"
        "Everything is baked in a kitchen that handles nuts and sesame\n"
        f"[Allergen list]({_HARBOR_BAKERY_URL}/allergens)\n"
        "Orders are taken at the counter or by telephone.\n"
        "Deliveries go out to the harbour side only.\n"
    ),
)

_RIVER_OTTERS_URL = "https://riverotters.example/census"
_RIVER_OTTERS_CENSUS = CannedPage(
    match="otter",
    text=(
        "Title: River otter census — weekly count | riverotters\n"
        f"{_RIVER_OTTERS_URL}\n"
        "\n"
        "A volunteer survey of a fictional river reach, walked every Thursday.\n"
        "Count: 46 otters\n"
        f"[River otter census]({_RIVER_OTTERS_URL})\n"
        "Counted at three holts and two feeding stations along the reach.\n"
        "The figure is revised if a recount is called within the week.\n"
        "\n"
        "Method\n"
        "Two observers walk the reach from the weir to the road bridge\n"
        f"[Survey method]({_RIVER_OTTERS_URL}/method)\n"
        "Spraint sites are logged separately and do not enter the count.\n"
        "A cub is counted once it leaves the holt unaccompanied.\n"
        "\n"
        "Previous weeks\n"
        "Last week 48, the week before 47, and 44 a month ago\n"
        f"[Count history]({_RIVER_OTTERS_URL}/history)\n"
        "Winter counts run lower and are not compared with summer ones.\n"
        "The survey has run for nine seasons.\n"
    ),
)

_EAST_BRANCH_URL = "https://eastbranch.example/new-titles"
_EAST_BRANCH_NEW_TITLES = CannedPage(
    match="branch",
    text=(
        "Title: New titles — east branch library | eastbranch\n"
        f"{_EAST_BRANCH_URL}\n"
        "\n"
        "Titles added to a fictional branch catalogue, refreshed each weekday.\n"
        f"[East branch new titles]({_EAST_BRANCH_URL})\n"
        "Listed newest first; older titles drop off after a fortnight.\n"
        "\n"
        "Newest title — added Wednesday\n"
        '"The Longshore Register" by Perrin Aldaz\n'
        f"[The Longshore Register]({_EAST_BRANCH_URL}/longshore-register)\n"
        "A century of shipping notices, read as a portrait of one small port.\n"
        "Hardcover · 402 pages · Shelf 387.1 · 2 copies, 1 available\n"
        "\n"
        "Added the Monday before that\n"
        '"Winter Sowing" by Halla Bierce\n'
        f"[Winter Sowing]({_EAST_BRANCH_URL}/winter-sowing)\n"
        "A kitchen-garden year told month by month, with seed lists.\n"
        "Paperback · 214 pages · Shelf 635 · 5 copies, 3 available\n"
        "\n"
        "Added a fortnight ago\n"
        '"The Quarry Road" by Nessim Toft\n'
        f"[The Quarry Road]({_EAST_BRANCH_URL}/quarry-road)\n"
        "A walking guide to nine disused workings and the paths between them.\n"
        "Spiral-bound · 168 pages · Shelf 796.51 · 3 copies, all out\n"
        "\n"
        "Holds and renewals are handled at the desk or through the catalogue.\n"
    ),
)


# ── The five cold asks ────────────────────────────────────────────────────────
#
# Each names its new space and the job's terms and stands entirely alone: nothing in the
# wording refers to the routine already existing, so nothing but the ask's MEANING can
# select among the five.

# Case 1 — a second listing for the price watcher.
#
# Reference reply:
#   done — i'll check the keel lantern listing every hour until sunday night and
#   message you if the price moves.
_COLD_ASK = (
    "can you watch this listing for me and let me know when the price changes? "
    f"{_KEEL_LANTERN_URL} — every hour until sunday night is fine"
)

# Case 2 — a different harbour for the timetable watcher, and the cold TWO-parameter
# bind: the page and what to look for on it both come out of this one message.
#
# Reference reply:
#   done — i'll check the north pier timetable every morning and message you when the
#   dawn sailing shows up.
_COLD_ASK_TWO_PARAMS = (
    f"every morning can you check the north pier timetable at {_NORTH_PIER_URL} "
    "and let me know when they add the dawn sailing?"
)

# Case 3 — a second bakery for the daily digest.
#
# Reference reply:
#   done — i'll check the harbor bakery's menu every day and message you the special.
_COLD_ASK_DIGEST = (
    f"can you get the daily special from {_HARBOR_BAKERY_URL} each day and tell me what it is?"
)

# Case 4 — a different survey for the count watcher.
#
# Reference reply:
#   done — i'll check the otter census every week and message you if the count drops.
_COLD_ASK_THRESHOLD = (
    f"keep track of the otter count at {_RIVER_OTTERS_URL} every week and let me know if it drops"
)

# Case 5 — another branch for the new-arrival watcher, under act-now pressure.
#
# The act-now clause is TEMPORAL, not ordinal (code-owner ruling).  It read "tell me the
# second something new shows up", which is two sentences in one: the intended "the moment
# it appears", and "tell me the SECOND new item" — a count-to-two task no routine covers.
# The classifier's own thinking took the ordinal reading and parked in elicit, correctly:
# nothing in the registry counts arrivals.  An ask that can be read two ways measures the
# reading, not the edge, so it says "when" and means it.
#
# Reference reply:
#   done — i'll check the east branch's new titles every two hours until friday and
#   message you when something new shows up.
_COLD_ASK_URGENCY = (
    f"watch {_EAST_BRANCH_URL} every two hours until friday and tell me when something new shows up"
)


# ── The world: five finished journeys, then small talk ────────────────────────


class _Exchange(NamedTuple):
    """One plain exchange — what the user said and what Penny said back, answered in a
    single text-only turn with no tool calls.

    Two customers, one shape: the pair that CLOSES each journey (the user acknowledging
    the job is running, Penny saying you're welcome) and each piece of small talk after
    the last one.  They are the same thing structurally — an idle → idle turn — so they
    are seeded by one function rather than by two that would drift."""

    said: str
    answered: str


class _AppliedJob(NamedTuple):
    """What one journey's APPLY turn stood up — the live mechanism it left behind.

    ``schedule`` is the rule that turn set (the acceptance's cadence, in the one
    grammar), ``expires_in`` how far ahead its end condition sits when the world is laid
    down, and ``params`` what the routine was pointed at.  What she SAID when she was done
    is not here: that is the round's own reference reply (``_ApplyCase.confirmation``), and
    replaying it is the whole reason it is data."""

    schedule: str
    expires_in: timedelta | None
    params: dict[str, str]


class _Journey(NamedTuple):
    """One completed journey — the learn → apply ``round`` whose four turns and two moves
    are its first two beats, the job its apply turn left running, and the exchange that
    CLOSED it.

    A journey ends conversationally, not on Penny's confirmation: the user says thanks and
    she answers.  That pair is what makes the finished round read as finished — and it is
    where the structural apply → idle reset lands, since the ack is the next message to
    arrive on a machine sitting in apply."""

    round: _ApplyCase
    applied: _AppliedJob
    closing: _Exchange


# The five journeys' exit states, each one's terms taken from the acceptance its learn →
# apply case ends on, so this world is that beat's own scored outcome carried forward.
_JOURNEYS = (
    _Journey(
        _AURORA_APPLY,
        _AppliedJob(
            schedule="FREQ=HOURLY",
            expires_in=timedelta(hours=6),
            params={"url": LISTING_URL},
        ),
        _Exchange(
            said="great, thanks",
            answered="you're welcome — shout if you want anything else kept an eye on.",
        ),
    ),
    _Journey(
        _FERRY_APPLY,
        _AppliedJob(
            schedule="FREQ=DAILY;BYHOUR=14",
            expires_in=None,
            params={"url": _FERRY_TIMETABLE_URL, "keyword": "late sailing"},
        ),
        _Exchange(
            said="perfect, appreciate it",
            answered="anytime — just say the word if there's anything else you want watched.",
        ),
    ),
    _Journey(
        _BAKERY_APPLY,
        _AppliedJob(
            schedule="FREQ=DAILY",
            expires_in=None,
            params={"url": _BAKERY_SPECIALS_URL},
        ),
        _Exchange(
            said="lovely, thank you",
            answered="my pleasure — tell me if there's anything else you'd like tracked.",
        ),
    ),
    _Journey(
        _COLONY_APPLY,
        _AppliedJob(
            schedule="FREQ=WEEKLY",
            expires_in=None,
            params={"url": _COLONY_COUNT_URL},
        ),
        _Exchange(
            said="brilliant, cheers",
            answered="no trouble at all — happy to take on anything else you want followed.",
        ),
    ),
    _Journey(
        _ARRIVALS_APPLY,
        _AppliedJob(
            schedule="FREQ=HOURLY;INTERVAL=2",
            expires_in=timedelta(days=12),
            params={"url": _NEW_ARRIVALS_URL},
        ),
        _Exchange(
            said="amazing, thanks so much",
            answered="you're welcome — just say if there's anything else worth watching.",
        ),
    ),
)


# The recent turns.  Deliberately NOT skill-adjacent: nothing here watches, schedules,
# reads a page or asks to be told about anything, so the last stretch of the conversation
# window carries no hint of the routines at all and the cold ask stands on its own meaning
# alone.
_IDLE_BANTER = (
    _Exchange(
        said="finally got an omelette right — first one that didn't fall apart",
        answered="nice, the not-falling-apart part is most of the battle. what went in it?",
    ),
    _Exchange(
        said="just cheese and a bit of chive",
        answered="classic. hard to beat when the pan's behaving.",
    ),
    _Exchange(
        said="the pan was behaving for once, i'll take it",
        answered="long may it last.",
    ),
)

# The call id the seeded apply turn's one call is keyed by — the same role
# ``_BROWSE_CALL_ID`` and ``_WRITE_CALL_ID`` play for the demonstrated round.
_SET_CALL_ID = "call-seeded-set"

# How many mutation events a probe or scorer reads back for one collection.  Generous: a
# seeded job carries two (created by its round, re-rendered by its apply turn), and
# reading well past them is what makes "nothing else has touched it" a real claim.
_MUTATION_WINDOW = 20

# What one journey contributes to the world, counted once here so the windows below and
# the expected conversation are derived from the same arithmetic: FOUR incoming turns (the
# ask, the demonstration, the acceptance and the closing ack) and FIVE moves (the elicit,
# learn and apply draws, the reset the ack carries, and the ack's own idle draw).
_TURNS_PER_JOURNEY = 4
_MOVES_PER_JOURNEY = 5

# The windows a reader of this world reads THROUGH, derived from the world's own shape
# rather than picked: every journey's turns and moves, one of each per exchange of small
# talk, and the turn under test — doubled, so the reader is never what decides what counts
# as history.  Both stores drop the OLDEST rows when a cap binds, so a window that CUT
# would let the novelty probe pass on a value the world does hold.
_COMPOSED_MESSAGE_WINDOW = 2 * (_TURNS_PER_JOURNEY * len(_JOURNEYS) + len(_IDLE_BANTER) + 1)
_COMPOSED_MOVE_WINDOW = 2 * (_MOVES_PER_JOURNEY * len(_JOURNEYS) + len(_IDLE_BANTER) + 1)


def _candidate(skill: SkillDraft) -> SkillCandidate:
    """One taught routine as the classifier is offered it — name, description and
    declared parameters: the production projection (``build_snapshot``) applied to a
    draft the registry has not been handed yet."""
    return SkillCandidate(
        name=slug_skill_name(skill.name),
        description=skill.description,
        parameters=[
            CandidateParameter(name=parameter.name, description=parameter.description)
            for parameter in skill.parameters
        ],
    )


def seed_composed_world() -> Seeder:
    """The world all five cold asks are answered against: five journeys walked to their
    end, then a few exchanges of small talk, in the order they were said.

    Compositional by construction — a journey is ``seed_round_through_learn`` (the two
    beats the earlier cases are measured against) plus the apply turn that finished it,
    so nothing here restates a world an earlier beat already defines.  The registry grows
    as it really did: each journey's classifier draws are offered the routines taught
    BEFORE it, and its own only once its round has run.

    The same world for every case — five live jobs, five routines, one machine sitting
    idle — because what distinguishes the cases is the ask, not the history.

    Ordering is production-shaped; SPACING is best effort.  Every row is written through
    the real store methods, which stamp it at the moment they run, so the whole history
    lands within a second of the sample starting and only its ORDER carries the passage
    of time.  A world dated into the past would mean writing timestamps outside the APIs,
    which is the one thing a seed must not do."""

    def seed(db: Database) -> None:
        taught: list[SkillCandidate] = []
        for journey in _JOURNEYS:
            _seed_journey(db, journey, taught_so_far=tuple(taught))
            taught.append(_candidate(journey.round.skill))
        _seed_idle_banter(db, tuple(taught))

    return seed


def _seed_journey(
    db: Database, journey: _Journey, *, taught_so_far: tuple[SkillCandidate, ...]
) -> None:
    """One journey end to end: the ask, the teach question, the demonstration and its
    report, the acceptance and the job it stood up, then the exchange that closed it —
    the user saying thanks and Penny answering, which is where a real round stops.

    The closing exchange is where the structural reset lands: it is the next message to
    arrive on a machine sitting in apply, so it carries the reset exactly as production
    writes it."""
    ask_id, _ = seed_round_through_learn(db, journey.round, taught_so_far=taught_so_far)
    _seed_apply_turn(db, journey, ask_id, taught_so_far=taught_so_far)
    _seed_exchange(
        db,
        journey.closing,
        draw_run=journey.round.runs.ack_draw,
        turn_run=journey.round.runs.ack_turn,
        penny_last_turn=journey.round.confirmation,
        candidates=(*taught_so_far, _candidate(journey.round.skill)),
    )


def _settle_before_the_next_message(db: Database) -> None:
    """What production does the moment ANY next message arrives while the machine sits in
    apply: apply has no out-edges, so the machine resets to idle before a word is
    classified.  The reset therefore belongs to the message that FOLLOWS a finished
    journey — which, since every journey closes with an exchange, is that journey's own
    acknowledgement."""
    latest = db.machine.latest_transition()
    if latest is not None and latest.to_state == ConversationState.APPLY.value:
        _structural_reset(db)


def _seed_apply_turn(
    db: Database,
    journey: _Journey,
    ask_id: int,
    *,
    taught_so_far: tuple[SkillCandidate, ...],
) -> None:
    """The turn that finished a journey, with everything it left behind: the acceptance
    INCOMING, the skill-gated draw that decided apply and named the routine it bound, the
    collection ADOPTING that routine (its program rendered, its schedule and notify set,
    a mutation event citing this run), the chat run carrying the ``collection_set`` call
    and its echoed result, Penny's confirmation OUTGOING, and the apply move itself.

    This footprint is what the earlier beats had no need of: their world stops where
    their own turn begins, while a returning user's world is one where turns like this
    one have already happened five times."""
    case = journey.round
    bound = slug_skill_name(case.skill.name)
    acceptance_id = _log_ask(db, case.acceptance, case.case_id)
    _log_apply_draw(db, case, taught_so_far)
    row = _adopt_the_taught_routine(db, journey)
    _seed_apply_run(db, journey, row)
    _log_reply(db, case.confirmation, answering=acceptance_id)
    _park(
        db,
        ConversationState.APPLY,
        anchor_message_id=ask_id,
        from_state=ConversationState.LEARN,
        run_id=case.runs.apply_turn,
        message_id=acceptance_id,
        skill_name=bound,
    )


def _log_apply_draw(
    db: Database, case: _ApplyCase, taught_so_far: tuple[SkillCandidate, ...]
) -> None:
    """The SKILL-GATED draw that decided the apply move — parked in learn on the ask, over
    the registry as it stood (the routines taught before this round, plus the one this
    round just taught), and naming the routine it bound on its second line."""
    _log_classifier_draw(
        db,
        run_id=case.runs.apply_draw,
        snapshot=MachineSnapshot(
            state=ConversationState.LEARN,
            penny_last_turn=case.prior.closing_report,
            task_anchor=case.prior.ask,
            skill_candidates=[*taught_so_far, _candidate(case.skill)],
        ),
        message=case.acceptance,
        drawn=_drawn_state(ConversationState.APPLY, skill=slug_skill_name(case.skill.name)),
    )


def _adopt_the_taught_routine(db: Database, journey: _Journey) -> MemoryRow:
    """The apply turn's durable half, written the way ``collection_set`` writes it: the
    collection the round already wrote into ADOPTS the routine — the skill's steps
    rendered into its ``extraction_prompt`` with the attachment bound to the collection's
    own name, the acceptance's rule as its schedule, notify on, and the skill plus its
    bound params stamped as provenance.

    Through the real store method the tool calls, so the update records its own mutation
    event citing this run — which is what makes "nothing has touched these five jobs
    since" a read rather than an assumption."""
    case, applied = journey.round, journey.applied
    target = case.demonstrated.collection
    schedule = parse_schedule(applied.schedule)
    return db.memories.update_collection_metadata(
        target,
        extraction_prompt=render_skill(retarget_writes(case.skill.steps, target), applied.params),
        schedule=schedule.rule,
        replace_schedule=True,
        max_runs=schedule.max_runs,
        expires_at=_end_condition(applied),
        notify=True,
        skill_name=slug_skill_name(case.skill.name),
        skill_params=applied.params,
        run_id=case.runs.apply_turn,
    )


def _end_condition(applied: _AppliedJob) -> datetime | None:
    """A bounded job's end, as a DISTANCE from when the world is laid down.

    The seeders write through the real store APIs, which stamp every row at the moment
    they run — so a world cannot be dated into the past, and an end condition written as
    a fixed date would be one these jobs had already passed (a passed expiry archives the
    collection at the next sweep).  Stating it as a distance is what keeps all five of
    them LIVE, which is what the world claims they are."""
    if applied.expires_in is None:
        return None
    return datetime.now(UTC) + applied.expires_in


def _seed_apply_run(db: Database, journey: _Journey, row: MemoryRow) -> None:
    """The apply turn's chat run: the one ``collection_set`` call it made, the result
    that came back, and the confirmation it closed on."""
    case = journey.round
    conversation: list[dict] = [{"role": "user", "content": case.acceptance}]
    conversation = _seed_call_step(
        db, conversation, _SET_CALL_ID, _set_step(journey, row), run_id=case.runs.apply_turn
    )
    _log_chat_step(
        db,
        run_id=case.runs.apply_turn,
        messages=conversation,
        response=_seeded_response(case.confirmation),
    )


def _set_step(journey: _Journey, row: MemoryRow) -> DistillInput:
    """The ``collection_set`` call that stood the job up — the arguments it was made with
    and the result it came back with, framed by the PRODUCTION framer over the production
    echo, so the seeded ledger carries the text a real turn would have read rather than
    an approximation of it."""
    applied = journey.applied
    skill_name = slug_skill_name(journey.round.skill.name)
    arguments: dict = {
        "name": row.name,
        "skill": skill_name,
        "params": applied.params,
        "schedule": applied.schedule,
        "notify": True,
    }
    if row.expires_at is not None:
        arguments["expires_at"] = row.expires_at.isoformat()
    echo = render_reinstantiation_echo(row, skill_name, applied.params)
    return DistillInput(
        source_ordinal=1,
        tool=_SET_TOOL,
        arguments=arguments,
        result=Tool.format_result(_SET_TOOL, arguments, ToolResult(message=echo, mutated=True)),
    )


def _seed_idle_banter(db: Database, candidates: tuple[SkillCandidate, ...]) -> None:
    """The recent turns: a few exchanges of small talk after the last journey closed.

    The draws are offered every taught routine, as production offers them, and decide idle
    anyway — which is what makes this stretch a real absence of configuring rather than an
    absence of evidence.  The machine is already idle when they start (the last journey's
    own acknowledgement carried the reset), so nothing structural happens here."""
    penny_last_turn = _JOURNEYS[-1].closing.answered
    for index, turn in enumerate(_IDLE_BANTER):
        _seed_exchange(
            db,
            turn,
            draw_run=seeded_run_id(f"banter-{index}-draw"),
            turn_run=seeded_run_id(f"banter-{index}-turn"),
            penny_last_turn=penny_last_turn,
            candidates=candidates,
        )
        penny_last_turn = turn.answered


def _seed_exchange(
    db: Database,
    turn: _Exchange,
    *,
    draw_run: str,
    turn_run: str,
    penny_last_turn: str,
    candidates: tuple[SkillCandidate, ...],
) -> None:
    """One plain exchange, with the full footprint an ordinary turn leaves: any structural
    move its arrival settles, the message in, the draw that decided idle, the text-only run
    that answered, the answer out threaded to it, and the move that kept the machine where
    it was.

    Shared by a journey's closing acknowledgement and every piece of small talk — the ack
    is the same kind of turn, and it is the one that carries the post-apply reset."""
    _settle_before_the_next_message(db)
    said_id = _log_ask(db, turn.said, draw_run)
    _log_idle_draw(db, turn, draw_run, candidates, penny_last_turn)
    _log_chat_step(
        db,
        run_id=turn_run,
        messages=[{"role": "user", "content": turn.said}],
        response=_seeded_response(turn.answered),
    )
    _log_reply(db, turn.answered, answering=said_id)
    _park(db, ConversationState.IDLE, run_id=turn_run, message_id=said_id)


def _log_idle_draw(
    db: Database,
    turn: _Exchange,
    draw_run: str,
    candidates: tuple[SkillCandidate, ...],
    penny_last_turn: str,
) -> None:
    """The draw that classified one plain exchange — offered every taught routine, as
    production offers them, over an idle machine with nothing parked, and deciding idle."""
    _log_classifier_draw(
        db,
        run_id=draw_run,
        snapshot=MachineSnapshot(
            state=ConversationState.IDLE,
            penny_last_turn=penny_last_turn,
            skill_candidates=list(candidates),
        ),
        message=turn.said,
        drawn=_drawn_state(ConversationState.IDLE),
    )


# ── The loud probe: the world is the one five finished journeys would have left ─


def assert_composed_world(db: Database) -> None:
    """Everything the composed seeder is responsible for, asserted out loud.

    Five cases share one world and one seeder, so a drift here is five cases answered
    against a world nothing produces — and it costs an hour of GPU before anyone reads a
    number.  The registry half is deliberately absent: the runner lays the fixture skills
    down AFTER this seed, so it belongs to the prepare hook (and is asserted there), while
    everything below is true the moment the seeder returns."""
    _assert_the_machine_is_idle(db)
    _assert_five_live_jobs(db)
    _assert_every_round_is_in_the_ledger(db)
    assert_conversation_window(db)


def _assert_the_machine_is_idle(db: Database) -> None:
    """The machine sits in idle, unanchored — the last thing it did was classify a piece
    of small talk as nothing in particular, which is what "some time later, mid-idle"
    means structurally.  Each finished journey is followed by its own structural reset,
    so the log shows five of them, one per message that arrived on a finished round."""
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.IDLE.value, (
        f"the composed world must leave the machine idle, not {latest}"
    )
    assert latest.anchor_message_id is None, (
        f"an idle machine is parked on nothing, not on {latest.anchor_message_id}"
    )
    moves = db.machine.recent_transitions(limit=_COMPOSED_MOVE_WINDOW)
    resets = [row for row in moves if row.cause == TransitionCause.STRUCTURAL.value]
    assert len(resets) == len(_JOURNEYS), (
        f"each finished journey is reset by the message after it, got {len(resets)} resets"
    )


def _assert_five_live_jobs(db: Database) -> None:
    """The five collections are LIVE mechanisms — each carries the routine its journey
    taught, a rendered program, a schedule and the notify its user asked for, none of them
    has retired itself, and each is pointed where its round pointed it."""
    for journey in _JOURNEYS:
        row = db.memories.get(journey.round.demonstrated.collection)
        assert row is not None, f"{journey.round.case_id}: the journey's collection must exist"
        _assert_live_job(journey, row)


def _assert_live_job(journey: _Journey, row: MemoryRow) -> None:
    """One journey's job, item for item — the routine it runs, the three things that make
    it live, and the terms its acceptance gave."""
    case = journey.round
    assert row.skill_name == slug_skill_name(case.skill.name), (
        f"{case.case_id}: the job must run the routine its round taught, not {row.skill_name}"
    )
    assert row.extraction_prompt and has_schedule(row) and row.notify, (
        f"{case.case_id}: a live job has a rendered program, a schedule and notify on"
    )
    assert not row.archived, f"{case.case_id}: the job must still be running"
    expected_expiry = journey.applied.expires_in is not None
    assert (row.expires_at is not None) == expected_expiry, (
        f"{case.case_id}: the end condition must match what its acceptance gave"
    )
    assert _bound_parameters(row) == journey.applied.params, (
        f"{case.case_id}: the job must be pointed where its round pointed it"
    )


def _assert_every_round_is_in_the_ledger(db: Database) -> None:
    """Each journey is READABLE as the turns it really was: its demonstrated calls under
    its own learn run, its apply turn's one ``collection_set`` call under its own apply
    run, and everything it produced citing the run that produced it — plus the five pages
    those rounds read, in browse-results."""
    for journey in _JOURNEYS:
        case = journey.round
        assert_round_calls_logged(db, case)
        assert_round_rows_cite_their_run(db, case)
        calls = [
            call.get("function", {}).get("name")
            for row in db.messages.get_run_prompts(case.runs.apply_turn)
            for call in _row_tool_calls(row)
        ]
        assert calls == [_SET_TOOL], (
            f"{case.case_id}: the apply turn made one set call, got {calls}"
        )
    fetched = _pages_fetched(db)
    assert len(fetched) == len(_JOURNEYS), (
        f"each round read one page, browse-results has {len(fetched)}"
    )


def expected_conversation() -> list[tuple[str, str]]:
    """The world as a CONVERSATION — every turn, in the order it was said, each tagged with
    the direction it went.

    Composed from the same fixtures the seeder writes from, so it states what the world is
    meant to be rather than reading back what it happens to be.  Per journey: the ask, the
    teach question, the demonstration, the report, the acceptance, the confirmation, the
    acknowledgement and the you're-welcome — then the small talk."""
    incoming = PennyConstants.MessageDirection.INCOMING
    outgoing = PennyConstants.MessageDirection.OUTGOING
    turns: list[tuple[str, str]] = []
    for journey in _JOURNEYS:
        case = journey.round
        turns += [
            (incoming, case.prior.ask),
            (outgoing, case.prior.teach_question),
            (incoming, case.prior.demo),
            (outgoing, case.prior.closing_report),
            (incoming, case.acceptance),
            (outgoing, case.confirmation),
            (incoming, journey.closing.said),
            (outgoing, journey.closing.answered),
        ]
    for turn in _IDLE_BANTER:
        turns += [(incoming, turn.said), (outgoing, turn.answered)]
    return turns


# The two claims the deterministic pin in ``test_eval_harness.py`` reads: what Penny said
# when each journey's job went live, in journey order, and the turns the window ENDS on.
# Derived from the same fixtures, so a fixture edit moves the pin with it.
JOURNEY_CONFIRMATIONS = tuple(journey.round.confirmation for journey in _JOURNEYS)
LAST_SPOKEN_TURNS = tuple(expected_conversation()[-2 * len(_IDLE_BANTER) :])


def assert_conversation_window(db: Database) -> None:
    """The world READS as a conversation — every turn present, in order, alternating.

    Asserted through ``get_messages_since``, the reader ``_build_conversation`` uses, rather
    than off the message table: an outgoing row is IN the record and OUT of the conversation
    unless it is threaded to the message it answers, and the two questions have different
    answers.  Measured, the unthreaded version came back all-user, and the same-role merge
    folded the entire history into ONE user turn reading as a pile of unanswered requests —
    which is what the first run of these cases answered.

    The alternation is the claim: a window that alternates is a window whose every reply
    landed, and one that does not names the first turn where it stopped.  Beside it, every
    reply is asserted to be a THREADED one — because contents alone cannot tell the two
    ways into that window apart, and only one of them is what a direct reply is."""
    window = db.messages.get_messages_since(
        TEST_SENDER, since=datetime.min, limit=_COMPOSED_MESSAGE_WINDOW
    )
    seen = [(row.direction, row.content) for row in window]
    expected = expected_conversation()
    assert seen == expected, (
        "the seeded world must read back as the conversation it claims to be — "
        f"diverges at turn {_first_divergence(seen, expected)}"
    )
    _assert_every_reply_is_threaded(window)


def _assert_every_reply_is_threaded(window: list[MessageLog]) -> None:
    """Each of Penny's turns answers the message BEFORE it, by ``parent_id``.

    The window has two doors for an outgoing row — a threaded reply, or an autonomous send
    (no parent, addressed to the user) — and a direct reply is the first.  An unthreaded
    reply that happens to be addressed to the user comes back through the SECOND door, so
    the conversation reads right while every one of Penny's turns claims to be a mechanism
    speaking unprompted.  That is a different world from the one these cases describe, and
    the contents cannot tell them apart: only the parent link can."""
    for index, row in enumerate(window):
        if row.direction != PennyConstants.MessageDirection.OUTGOING:
            continue
        answered = window[index - 1] if index else None
        assert answered is not None and row.parent_id == answered.id, (
            f"turn {index} ({row.content!r}) must be threaded to the message it answers, "
            f"got parent {row.parent_id}"
        )


def _first_divergence(seen: list[tuple[str, str]], expected: list[tuple[str, str]]) -> str:
    """Where two turn sequences part company, as a readable line — a 46-turn inequality
    says nothing on its own, and the FIRST difference is the whole diagnosis."""
    for index, (actual, wanted) in enumerate(zip(seen, expected, strict=False)):
        if actual != wanted:
            return f"{index}: saw {actual}, expected {wanted}"
    return f"{min(len(seen), len(expected))}: saw {len(seen)} turns, expected {len(expected)}"


def _probe_composed_world(case: _IdleApplyCase) -> Preparer:
    """The prepare hook: the seeder's own claims, the registry one that is only true once
    the runner has laid the fixture skills down, and the case's own novelty claim."""

    def probe(penny: Penny) -> None:
        assert_composed_world(penny.db)
        _assert_the_registry_holds_the_five(penny.db)
        assert_new_space_is_unknown(penny.db, case)

    return probe


def _assert_the_registry_holds_the_five(db: Database) -> None:
    """Exactly the five routines the journeys taught — no decoy, and none wanted.  Five
    real routines of the same kind ARE the distractor set here, so a sixth would only
    dilute a selection that is already the hard part."""
    taught = sorted(skill.name for skill in db.skills.list_all())
    expected = sorted(slug_skill_name(journey.round.skill.name) for journey in _JOURNEYS)
    assert taught == expected, f"the registry must hold the five taught routines, got {taught}"


def assert_new_space_is_unknown(db: Database, case: _IdleApplyCase) -> None:
    """Every value this ask supplies is NEW — it appears in no message, no stored entry,
    no bound parameter and no page already read.

    This is what makes "bound from the message" a real claim: a value the history also
    carries could have been copied out of the world instead of read out of the ask, and
    the check would pass either way."""
    stored = [row for row in db.memories.list_all() if row.type == MemoryType.COLLECTION.value]
    known = [
        *(
            row.content
            for row in db.messages.get_user_messages(TEST_SENDER, limit=_COMPOSED_MESSAGE_WINDOW)
        ),
        *outgoing_replies(db),
        *(text for row in stored for text in _collection_texts(db, row.name)),
        *(value for row in stored for value in _bound_parameters(row).values()),
        *(entry.content for entry in _pages_fetched(db)),
    ]
    for wanted in case.bound:
        assert not _mentions(wanted, known), (
            f"{case.case_id}: {wanted!r} must be new to this world — the history already has it"
        )


def _collection_texts(db: Database, name: str) -> list[str]:
    """Both halves of every entry a collection holds — what a value would have been copied
    FROM if it were already known.  Collections only: the logs are read separately (the
    pages by their own reader, the conversation by both directions), and a log-shaped
    facade renders rows rather than storing them."""
    entries = collection_entries(db, name).items()
    return [text for key, content in entries for text in (key, content)]


# ── The cases ─────────────────────────────────────────────────────────────────


class _IdleApplyCase(NamedTuple):
    """One agreed cold ask, and what the job it should stand up has to look like.

    ``skill`` is the routine the ask is covered by — the one of five the decision has to
    pick.  ``page`` is the new space, installed as a live temptation.  The rest is what
    the ask's own terms give: ``cadence_seconds`` is how far apart the job should fire
    whatever rule spelling says so, ``anchored`` whether the terms name a time of DAY
    rather than a period (so the rule has to state an hour to run at), ``expects_expiry``
    whether they gave an end condition at all (inventing one is a failure), and ``bound``
    every value the MESSAGE supplies that the routine has to be pointed at."""

    case_id: str
    ask: str
    page: CannedPage
    skill: SkillDraft
    cadence_seconds: int
    anchored: bool
    expects_expiry: bool
    bound: tuple[str, ...]


_COLD_PRICE = _IdleApplyCase(
    case_id="transition-idle-to-apply",
    ask=_COLD_ASK,
    page=_KEEL_LANTERN_LISTING,
    skill=_AURORA_SKILL,
    cadence_seconds=3600,
    anchored=False,
    expects_expiry=True,
    bound=(_KEEL_LANTERN_URL,),
)

_COLD_TWO_PARAMS = _IdleApplyCase(
    case_id="transition-idle-to-apply-two-params",
    ask=_COLD_ASK_TWO_PARAMS,
    page=_NORTH_PIER_DEPARTURES,
    skill=_FERRY_SKILL,
    cadence_seconds=86400,
    anchored=True,
    expects_expiry=False,
    bound=(_NORTH_PIER_URL, "dawn sailing"),
)

_COLD_DIGEST = _IdleApplyCase(
    case_id="transition-idle-to-apply-digest",
    ask=_COLD_ASK_DIGEST,
    page=_HARBOR_BAKERY_MENU,
    skill=_BAKERY_SKILL,
    cadence_seconds=86400,
    anchored=False,
    expects_expiry=False,
    bound=(_HARBOR_BAKERY_URL,),
)

_COLD_THRESHOLD = _IdleApplyCase(
    case_id="transition-idle-to-apply-threshold",
    ask=_COLD_ASK_THRESHOLD,
    page=_RIVER_OTTERS_CENSUS,
    skill=_COLONY_SKILL,
    cadence_seconds=604800,
    anchored=False,
    expects_expiry=False,
    bound=(_RIVER_OTTERS_URL,),
)

_COLD_URGENCY = _IdleApplyCase(
    case_id="transition-idle-to-apply-urgency",
    ask=_COLD_ASK_URGENCY,
    page=_EAST_BRANCH_NEW_TITLES,
    skill=_ARRIVALS_SKILL,
    cadence_seconds=7200,
    anchored=False,
    expects_expiry=True,
    bound=(_EAST_BRANCH_URL,),
)

# Every cold ask, in one place — so the deterministic pin in ``test_eval_harness.py`` can
# check each one's claim about the world without a GPU.
IDLE_APPLY_CASES = (
    _COLD_PRICE,
    _COLD_TWO_PARAMS,
    _COLD_DIGEST,
    _COLD_THRESHOLD,
    _COLD_URGENCY,
)


# ── Scoring ───────────────────────────────────────────────────────────────────


def _minted_job(db: Database, before: set[str]) -> MemoryRow | None:
    """The collection this turn MINTED for the new space — whichever routine it carries.

    Read as "new AND routine-bearing" rather than "carries the intended routine", so a
    turn that stood a job up on the WRONG one is a bound-the-wrong-routine finding rather
    than a set-nothing-up one.  The five seeded jobs are excluded by construction: they
    existed before the turn."""
    return next((row for row in new_collections(db, before) if row.skill_name is not None), None)


def _enactment_binding_check(row: MemoryRow | None, case: _IdleApplyCase) -> Check:
    """The other half of the selection claim (the decision half is
    ``_skill_binding_check``): the JOB runs the routine that covers the ask.

    Scored unconditionally, because with five real routines in the registry the enactment
    is where a wrong pick becomes a mechanism that watches the wrong kind of thing from
    now on — and it is a different failure from deciding wrongly, so it reads as its own
    row."""
    label = "state: the job runs the routine that covers the ask"
    intended = slug_skill_name(case.skill.name)
    bound = row.skill_name if row is not None else None
    return Check(
        label,
        bound == intended,
        rationale=None
        if bound == intended
        else f"the job runs {bound!r}, the ask needs {intended!r}",
        kind="state",
    )


def _seeded_jobs_untouched_check(db: Database) -> Check:
    """None of the five running jobs was reconfigured, re-rendered or archived by this
    turn — the different-params side of the one-job-one-collection boundary, read directly
    off the mutation ledger.

    A live turn's mutation cites a live run and every event the seeded world wrote cites a
    seeded one, so "this turn changed nothing here" is a read rather than a diff."""
    touched = [
        f"{journey.round.demonstrated.collection}: {event.action} by {event.run_id}"
        for journey in _JOURNEYS
        for event in db.mutations.history(journey.round.demonstrated.collection, _MUTATION_WINDOW)
        if not is_seeded_run(event.run_id)
    ]
    return Check(
        "state: the five running jobs were left untouched",
        not touched,
        rationale=f"touched {touched}" if touched else None,
        kind="state",
    )


def _no_teach_question_check(landed: StateTransition | None) -> Check:
    """She did not ask to be taught: the turn opened no round.

    Structural rather than a reading of the reply — the elicit shape IS the machine
    parking itself on the ask to be shown how, and that is a row, not a sentence.  Read off
    the move the scorer already fetched, like every other check that conditions on it."""
    asked_to_be_taught = landed is not None and landed.to_state == ConversationState.ELICIT.value
    return Check(
        "state: she did not ask to be taught (no round was opened)",
        not asked_to_be_taught,
        rationale="the turn parked in elicit" if asked_to_be_taught else None,
        kind="state",
    )


def _cold_anchor_check(db: Database, landed: StateTransition | None, case: _IdleApplyCase) -> Check:
    """The move came FROM idle and stamped the cold ask as its anchor — the anchor
    lifecycle's opening move (#1827), which is what says this turn started a round of its
    own rather than continuing one the history left open.  Same conditional-n/a as the
    decision check."""
    label = "state: the move came from idle with the ask as its anchor"
    applied = _landed_apply_move(landed)
    if applied is None:
        return Check.na(label, kind="state")
    asked = _seeded_ask_id(db, case.ask, limit=_COMPOSED_MESSAGE_WINDOW)
    opened = applied.from_state == ConversationState.IDLE.value
    anchored = applied.anchor_message_id
    ok = opened and asked is not None and anchored == asked
    return Check(
        label,
        ok,
        rationale=None
        if ok
        else f"came from {applied.from_state}, anchored to {anchored} (the ask is {asked})",
        kind="state",
    )


def _score_idle_to_apply(
    db: Database, before: set[str], reply: str, *, case: _IdleApplyCase
) -> list[Check]:
    """A routine she already knows was pointed at a space she has never seen, on the terms
    this one message gave — without teaching, without browsing, and without disturbing any
    of the five jobs already running.

    ONE scorer for all five cases, bound to the case's own terms.  The labels are
    diff-join keys and are deliberately case-NEUTRAL: one wording reads the same whether
    the job watches a price, a timetable or a shelf."""
    row = _minted_job(db, before)
    landed = db.machine.latest_transition()
    return [
        *_cold_binding_checks(db, row, landed, case),
        *_cold_terms_checks(db, row, case),
        _seeded_jobs_untouched_check(db),
        _no_teach_question_check(landed),
        Check(
            "reply: asked for no page structure",
            asked_for_page_structure(reply) is None,
            rationale=(
                f"asked for {term!r}" if (term := asked_for_page_structure(reply)) else None
            ),
            kind="reply",
        ),
        _cold_anchor_check(db, landed, case),
        *_job_setup_advisories(db, row, landed),
    ]


def _cold_binding_checks(
    db: Database, row: MemoryRow | None, landed: StateTransition | None, case: _IdleApplyCase
) -> list[Check]:
    """She set a job up, on the routine the ask is covered by, pointed at what the ask
    supplied."""
    return [
        Check(
            "state: she set the job up with collection_set",
            tool_was_called(db, _SET_TOOL),
            kind="state",
        ),
        Check(
            "state: a new collection carries the job",
            row is not None,
            rationale=None if row else "no new collection carries a routine",
            kind="state",
        ),
        Check(
            "state: the routine's program was rendered into it",
            row is not None and bool(row.extraction_prompt),
            kind="state",
        ),
        _skill_binding_check(
            landed,
            intended=slug_skill_name(case.skill.name),
            label="state: the decision bound the routine that covers the ask",
        ),
        _enactment_binding_check(row, case),
        _bound_parameters_check(
            row, wanted=case.bound, label="state: every parameter bound from the message"
        ),
    ]


def _cold_terms_checks(db: Database, row: MemoryRow | None, case: _IdleApplyCase) -> list[Check]:
    """The job runs on the terms the ask gave — its cadence, its end condition (or the
    absence of one), and the telling-them clause every one of these asks carries — and she
    set it running instead of running it once."""
    return [
        _schedule_check(row, fires_every=case.cadence_seconds, anchored=case.anchored),
        _expiry_check(row, expected=case.expects_expiry),
        Check(
            "state: it will tell them when something changes",
            row is not None and bool(row.notify),
            kind="state",
        ),
        Check(
            "state: she set it running instead of running it now (no browse this turn)",
            tool_not_called(db, "browse"),
            kind="state",
        ),
    ]


async def _run_idle_apply_case(chat_eval: ChatEval, case: _IdleApplyCase) -> None:
    """Drive one idle → apply case: the composed world behind it, the five taught routines
    in the registry, its new page installed as a live temptation, and the shared scorer
    bound to the terms its ask gives.  Report-only — the thresholds are the code owner's
    to set once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.ask,
        browse=[case.page],
        seed=seed_composed_world(),
        seed_skills=[journey.round.skill for journey in _JOURNEYS],
        prepare=_probe_composed_world(case),
        score=partial(_score_idle_to_apply, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_idle_to_apply_points_a_known_routine_at_a_new_listing(chat_eval: ChatEval) -> None:
    """idle → apply, cold: a second listing, long after the price watcher was taught on
    the first one.  Nothing in the ask refers back, so she has to recognise the job from
    what it asks for, bind the page it names, and set it running on the hours and the end
    it gives — without opening the listing to check."""
    await _run_idle_apply_case(chat_eval, _COLD_PRICE)


@pytest.mark.asyncio
async def test_idle_to_apply_binds_both_parameters_from_one_message(chat_eval: ChatEval) -> None:
    """idle → apply on the routine that asks for two things: the harbour page and the
    sailing to look for on it both come out of this ONE cold message, with no round behind
    it to supply either — the bind that was undecidable before the framer decided
    parameters from the ask alone."""
    await _run_idle_apply_case(chat_eval, _COLD_TWO_PARAMS)


@pytest.mark.asyncio
async def test_idle_to_apply_schedules_a_second_digest(chat_eval: ChatEval) -> None:
    """idle → apply on the daily digest, pointed at a second bakery: a plain daily cadence
    with a telling clause and NO end condition — so an expiry set here is one nobody asked
    for."""
    await _run_idle_apply_case(chat_eval, _COLD_DIGEST)


@pytest.mark.asyncio
async def test_idle_to_apply_tracks_a_different_survey(chat_eval: ChatEval) -> None:
    """idle → apply on the count watcher, pointed at a different survey: the longest
    cadence of the set, and the closest thing the registry has to a look-alike, since two
    of the five routines watch a page for a number."""
    await _run_idle_apply_case(chat_eval, _COLD_THRESHOLD)


@pytest.mark.asyncio
async def test_idle_to_apply_sets_the_tight_cadence_from_a_cold_start(
    chat_eval: ChatEval,
) -> None:
    """idle → apply under act-now pressure: another branch of the same catalogue, every
    two hours until a named day.  The urgency is a reason to set it up now, not a reason
    to go and look — and the end condition is a date the model has to work out."""
    await _run_idle_apply_case(chat_eval, _COLD_URGENCY)
