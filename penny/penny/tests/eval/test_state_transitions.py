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
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    asked_for_page_structure,
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


def _entries_written_by_this_run(db: Database) -> list[str]:
    """Every entry content this run wrote, wherever it landed.

    Scoring only collections the run CREATED assumed she always makes one — but
    "remember it" may reuse a name that already exists, and then the run's real
    writes are invisible while the reused collection's own seeded prompt and
    trigger read as things she did.  The run-id stamp says exactly what this run
    wrote (#1560), so ask that instead of inferring from newness."""
    written = []
    for row in db.memories.list_all():
        memory = db.memory(row.name)
        entries = memory.read_all() if memory is not None else []
        written += [e.content for e in entries if e.created_by_run_id]
    return written


def _leaf_at(arguments: dict, path: list):
    """The argument leaf a substitution's JSON path addresses — the step carries the
    call's verbatim arguments, so the DEMONSTRATED value is still in place."""
    node = arguments
    for part in path:
        node = node[part]
    return node


def _untraceable_parameters(db: Database) -> list[str]:
    """Required parameters whose DEMONSTRATED VALUE the user never supplied.

    A skill's parameters are what the NEXT user must provide to reuse it, so a
    required one nobody could supply makes the skill uninstantiable (#1770 — a
    round that also wrote a note it composed itself turned that note into a
    required `page_source`).  What she chose to write is her latitude and is not
    scored; the SHAPE of the skill it produced is.

    Checks the value, never the label: a correctly-named parameter (`url`,
    described as "the listing page to check") contains neither the address nor
    the word the user used, so testing the NAME reports a real parameter as
    unsupplied — which is exactly what this check did on its first run.

    This teach turn supplies two things — the page and what to find on it — so a
    legitimate parameter was demonstrated with one of them.  Fixture-anchored
    deliberately: no generic rule can decide this (the extract instruction is the
    user's intent in the assistant's words), which is why the labeller judges it
    in production and why the CASE, which knows what its user said, checks here."""
    supplied = (LISTING_URL.lower(), "price")
    untraceable = []
    for skill in db.skills.list_all():
        required = {p.name for p in parameters_from_json(skill.parameters) if p.required}
        demonstrated: dict[str, str] = {}
        for step in steps_from_json(skill.steps):
            for sub in step.substitutions:
                if sub.parameter is not None:
                    demonstrated[sub.parameter] = str(_leaf_at(step.arguments, sub.path)).lower()
        for name in sorted(required):
            value = demonstrated.get(name, "")
            if not any(token in value for token in supplied):
                untraceable.append(name)
    return untraceable


def _score_elicit_to_learn(db: Database, before: set[str], reply: str) -> list[Check]:
    """The demonstrated round ran, and NOTHING was instantiated.

    "Remember it" is a naive ``collection_write``: it auto-creates a collection
    and puts the value in it.  What must NOT happen is the fold — no skill bound
    to that collection, no rendered program, no schedule.  The skill is learned
    (it exists in the registry) and stays unattached until the user asks for it."""
    created = new_collections(db, before)
    written = _entries_written_by_this_run(db)
    instantiated = [row for row in db.memories.list_all() if row.skill_name is not None]
    return [
        Check(
            "state: she browsed the listing (the demonstrated fetch happened)",
            tool_was_called(db, "browse"),
            kind="state",
        ),
        Check(
            "state: the browsed price landed durably (remember = a plain write)",
            any("499" in content for content in written),
            rationale=None if written else "nothing was written",
            kind="state",
        ),
        Check(
            "state: a skill was learned from the round",
            bool(db.skills.list_all()),
            kind="state",
        ),
        # Learning must not instantiate.  Scored against what this run TOUCHED:
        # a collection it created, or — when it reused an existing one — nothing,
        # since a seeded collection's own prompt and cadence predate the round and
        # failing on those would report the framework's fixtures as her doing.
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
        else Check.na("state: no program was rendered into the collection it created"),
        Check(
            "state: nothing it created was scheduled (no trigger, no notify)",
            all(row.collector_interval_seconds is None and not row.notify for row in created),
            kind="state",
        )
        if created
        else Check.na("state: nothing it created was scheduled (no trigger, no notify)"),
        Check(
            "state: every required parameter is one the user supplied",
            not _untraceable_parameters(db),
            rationale=(f"unsupplied: {names}" if (names := _untraceable_parameters(db)) else None),
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
