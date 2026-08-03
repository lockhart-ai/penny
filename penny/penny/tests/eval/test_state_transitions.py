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

Each case's seeded state is the PRECEDING beat's terminal state and nothing
more — one edge is one message answered against where the last edge stopped.
Replaying earlier turns is not neutral: the apply case seeded the instigating
ask as well, and the classifier duly read "the task being worked on" as a setup
still being specified, which it no longer was.  The opening edge is the one whose
preceding state is NOTHING, so it seeds nothing at all: no transition rows (idle
is the absence of history) and an empty registry.  It carries FIVE asks rather
than one — the script's own turn plus four subjects borrowed from the
classifier's fire pool at a richer register — because a turn that must not act
is only proven by asks that make acting tempting in different ways.

**Learning attaches nothing** (#1706, replacing #1687's run-end auto-attach): the
machine makes teaching and instantiating two clear turns, so the demonstrated
round leaves a naive collection_write behind — a collection with a value in it
and no skill, no rendered program, nothing scheduled — and a LATER turn applies
the skill.  Scoring that separation is most of the point of these cases.

WHICH collection a job ends up on is deliberately out of scope (code owner): she
has spread work across several collections where one was meant since long before
this machine existed, so that is a collection-management question of its own and
grading it per-transition would report a standing problem as an edge failure.
The apply case scores that the skill is APPLIED correctly — bound, rendered,
scheduled on the terms given — and carries the reuse question as an advisory.
"""

from __future__ import annotations

import json

import pytest

from penny.constants import PennyConstants, TransitionCause
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.skill_store import parameters_from_json, steps_from_json
from penny.database.skills import (
    WRITE_TARGET_DESCRIPTION,
    DistillInput,
    SkillDraft,
    SkillStep,
    distill_steps,
    slug_skill_name,
)

# The production draw-application, used as itself: this case's fixture skill has to
# be the SHAPE run-end extraction really produces, and re-implementing that mapping
# here would be a fixture that drifts from the pipeline it stands in for.  Both
# halves of the #1824 split are applied by their own production function —
# ``_apply_leaf_labels`` for the labeller's spots, ``_naming`` +
# ``_interface_parameters`` for the framer's signature.
from penny.skill_extraction import _apply_leaf_labels, _interface_parameters, _naming
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    asked_for_page_structure,
    chat_run_tool_sequences,
    collection_entries,
    count_tool_calls,
    new_collections,
    outgoing_replies,
    routing_clean,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.fixtures import CannedPage

# The enacting-tool set is the elicitation contract itself — the calls that would mean
# she acted before being taught — so it is READ from where beat 1a already declares it
# rather than restated here: one policy, one definition.
from penny.tests.eval.test_watch_journey import (
    _ENACTING_TOOLS,
    AURORA_LISTING_499,
    LISTING_URL,
)
from penny.tools.micro_context import FramedParameter, LeafLabel, SkillLabels, SkillSignature

pytestmark = pytest.mark.eval

_FAMILY = "state-transitions"


# ── Shared across the edges ───────────────────────────────────────────────────


def _park(db: Database, state: ConversationState, *, anchor_message_id: int | None = None) -> None:
    """Leave the machine where the edge under test starts from, through the real
    store — a seeded transition row IS the machine's state (#1706), so nothing
    here fakes a state the production path couldn't be in.  The incoming message
    is still classified against it, so a case exercises the edge end to end.

    ``anchor_message_id`` is the instigating ask the parked round is anchored to
    — what the production anchor lifecycle stamps on the way in, and what the
    classifier renders as the task being worked on."""
    db.machine.record_transition(
        from_state=ConversationState.IDLE.value,
        to_state=state.value,
        cause=TransitionCause.CLASSIFIER,
        anchor_message_id=anchor_message_id,
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
_NEW_ARRIVALS = CannedPage(
    match="library",
    text=(
        "Title: New arrivals — city library | citylibrary\n"
        f"{_NEW_ARRIVALS_URL}\n"
        "\n"
        "Titles added to a fictional catalogue, refreshed every weekday morning.\n"
        'Newest arrival: "The Tidewater Almanac" (added Tuesday)\n'
        f"[City library new arrivals]({_NEW_ARRIVALS_URL})\n"
        "Older arrivals drop off the page after a fortnight.\n"
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
    written = _entries_written_by_this_run(db)
    fetched = require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).read_recent(
        window_seconds=3600, cap=None
    )
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
            "calls: clean routing (no bail or continue nudge fired)",
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

# The user answers the teach question with the steps — the very question case 1
# above ends on, so the two edges chain.
_TEACH_TURN = f"yeah go to {LISTING_URL}, find the price, and remember it"


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


# The role of a leaf NO substitution covers — it renders verbatim, which is exactly a
# baked value.  Deliberately NOT a ``SkillSubKind`` member: that enum names kinds of
# SUBSTITUTION, and a constant is the absence of one.
_ROLE_CONSTANT = "constant"


def _find_instruction_role(db: Database) -> str | None:
    """What role the learned skill gives the browse ``extract`` leaf — the value that
    says WHAT to pull off the page (#1803).

    :data:`_ROLE_CONSTANT` when NO substitution covers it: a leaf nothing covers renders
    verbatim, which is exactly a baked value. Otherwise the substitution's own kind
    (``hole`` = still asked for, ``placeholder`` = nobody can supply it). ``None`` when
    no skill or no browse step exists, which the caller reads as nothing to score.

    Read STRUCTURALLY off the leaf's path rather than by matching the demonstrated
    text, because what the assistant passes to ``extract`` is its own wording of the
    user's intent ("the price" / "the current price") and is not predictable. This case
    is a browse round, so naming that step is the same fixture-anchoring
    ``_untraceable_parameters`` already relies on — nothing in production keys off a
    tool name."""
    for skill in db.skills.list_all():
        for step in steps_from_json(skill.steps):
            if step.tool != "browse":
                continue
            covering = [sub for sub in step.substitutions if sub.path == ["extract"]]
            return covering[0].kind.value if covering else _ROLE_CONSTANT
    return None


def _page_is_bindable(db: Database) -> bool:
    """A required parameter was demonstrated with the PAGE — so the routine can be
    pointed at a different listing next time.

    Checks the value, never the label, for the reason ``_untraceable_parameters``
    gives: a correctly-named ``url`` parameter contains neither the address nor any
    word the user used."""
    for skill in db.skills.list_all():
        required = {p.name for p in parameters_from_json(skill.parameters) if p.required}
        for step in steps_from_json(skill.steps):
            for sub in step.substitutions:
                if (
                    sub.parameter in required
                    and LISTING_URL.lower() in str(_leaf_at(step.arguments, sub.path)).lower()
                ):
                    return True
    return False


def _score_elicit_to_learn(db: Database, before: set[str], reply: str) -> list[Check]:
    """The demonstrated round ran, and NOTHING was instantiated.

    "Remember it" is a naive ``collection_write``: it auto-creates a collection
    and puts the value in it.  What must NOT happen is the fold — no skill bound
    to that collection, no rendered program, no schedule.  The skill is learned
    (it exists in the registry) and stays unattached until the user asks for it."""
    created = new_collections(db, before)
    written = _entries_written_by_this_run(db)
    instantiated = [row for row in db.memories.list_all() if row.skill_name is not None]
    find_role = _find_instruction_role(db)
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
        # #1803: the round supplies the page AND what to find on it, both from the
        # user — but only one of them varies between uses.  The skill is a price
        # watcher, so the price is what it IS (baked, never asked for again) and the
        # page is what it is POINTED AT (a parameter).  Before the shape draw both
        # were required parameters, which is why the routine could not fire from the
        # natural second ask.
        Check(
            "state: the page stays a parameter (a new listing can be bound)",
            _page_is_bindable(db),
            kind="state",
        ),
        Check(
            "state: what to find is baked, not asked for again",
            find_role == _ROLE_CONSTANT,
            rationale=(
                None if find_role == _ROLE_CONSTANT else f"the extract leaf is a {find_role}"
            ),
            kind="state",
        )
        if find_role is not None
        else Check.na("state: what to find is baked, not asked for again"),
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
        seed=lambda db: _park(db, ConversationState.ELICIT),
        score=_score_elicit_to_learn,
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


# ── learn → apply: the offer accepted, the routine set running ────────────────

# The world the demonstrated round leaves behind, seeded so this case stays ONE
# turn: the collection the naive write created (the price in it, no skill, no
# program, no schedule — exactly what the elicit → learn case above scores), and
# the skill the run-end extractor distilled from that same round.
_WATCH_COLLECTION = "aurora-deck-2-price"
_DEMO_KEY = "aurora deck 2 price"
_PRICE = "$499"
_EXTRACT = "the current price"

_SKILL_NAME = "watch a listing page for its current price"
_SKILL_DESCRIPTION = "read a listing page and record the price it shows"

# The round's ledger, as the extractor would have read it off the promptlog.
_DEMONSTRATED_ROUND = [
    DistillInput(
        source_ordinal=1,
        tool="browse",
        arguments={"queries": [LISTING_URL], "extract": _EXTRACT},
        result=f"You opened the Aurora Deck 2 listing (browse result)\n{_PRICE}",
    ),
    DistillInput(
        source_ordinal=2,
        tool="collection_write",
        arguments={
            "memory": _WATCH_COLLECTION,
            "entries": [{"key": _DEMO_KEY, "content": _PRICE}],
        },
        result=(
            f"You saved an entry to {_WATCH_COLLECTION}: (collection_write result)\nWrote 1 entry."
        ),
    ),
]

# The LABELLER's draw (#1828), keyed by the DEMONSTRATED VALUE rather than by the
# arg-derived name the distiller happens to mint — so the fixture states what it means
# and cannot go quietly stale if that naming changes.  EVERY spot is listed because an
# accepted draw covers every spot: a line missing for one is a WHOLE-draw failure, so a
# partly-labelled routine is a shape run-end extraction cannot hand anyone.  The
# destination is a spot like any other and additionally carries the attachment mark —
# what the routine is applied to fills it, which is precisely what the apply turn under
# test then binds.
_LABELS = {
    LISTING_URL: LeafLabel(
        name="listing_page", description="the page whose price this routine reads"
    ),
    _EXTRACT: LeafLabel(name="value_to_find", description="what to pull off the page each run"),
    _DEMO_KEY: LeafLabel(name="entry_key", description="what to call the entry it saves"),
    _WATCH_COLLECTION: LeafLabel(name="storage_collection", description=WRITE_TARGET_DESCRIPTION),
}

# The FRAMER's draw (#1830) — the one thing here still hand-written, because a draw is
# what a fixture legitimately stands in for.  The framer writes a routine's INTERFACE
# from the user's ask alone, and this round's ask — watch this listing and tell me when
# the price changes — makes the page the one thing to re-say and the price what the
# routine IS.  Everything that APPLIES this signature is production code below.
_FRAMED_SIGNATURE = SkillSignature(
    name=_SKILL_NAME,
    description=_SKILL_DESCRIPTION,
    parameters=(FramedParameter(name="url", description="the listing page to check"),),
)


def _fixture_labels(steps: list[SkillStep]) -> SkillLabels:
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
            if value in _LABELS:
                labels[sub.parameter] = _LABELS[value]
    assert len(labels) == len(_LABELS), (
        f"the fixture's labels must all map home — matched {sorted(labels)} of {sorted(_LABELS)}"
    )
    return SkillLabels(labels=labels)


def learn_to_apply_fixture_skill() -> SkillDraft:
    """The skill that round leaves in the registry, built by the PRODUCTION pipeline
    over its ledger: ``distill_steps`` for the structure, then BOTH halves of the
    run-end split applied by their own production function — ``_apply_leaf_labels``
    for the labeller's spots, ``_naming`` + ``_interface_parameters`` for the framer's
    signature.  Only the two DRAWS are hand-written, which is what a fixture is for.

    So the case's starting world is the shape extraction produces, not a convenient
    copy of it — and the shape is the framer's declared interim (#1830), which nothing
    here re-states: an ALL-PLACEHOLDER recipe (every spot named by the labeller, the
    write target still carrying its attachment mark) over ONE skill-level parameter,
    the page.  Nothing joins that parameter to a leaf yet — that is the runtime-join
    beat — so it is the registry row, ``collection_set``'s unbound-parameter check,
    and job identity that carry it, which is exactly what the apply turn under test
    has to satisfy."""
    # The registry as this fixture's round saw it — #1783 marks a leaf whose
    # demonstrated value names one of Penny's collections, so the destination is
    # only marked if the collection actually existed.
    steps, parameters = distill_steps(_DEMONSTRATED_ROUND, frozenset({_WATCH_COLLECTION}))
    steps, distilled = _apply_leaf_labels(steps, parameters, _fixture_labels(steps))
    name, description = _naming(_FRAMED_SIGNATURE, _TEACH_TURN)
    framed = _interface_parameters(_FRAMED_SIGNATURE, distilled)
    # The framer's parameter is the one thing the apply turn must supply, so a
    # production application that stopped carrying the signature through would seed a
    # routine nothing could be pointed at — silently, and the apply case would report
    # it as the model's failure.  It fails here instead.
    framed_names = [parameter.name for parameter in framed]
    assert framed_names == [parameter.name for parameter in _FRAMED_SIGNATURE.parameters], (
        f"the framed interface must survive application — got {framed_names}"
    )
    return SkillDraft(
        name=name,
        intent=description,
        description=description,
        steps=steps,
        parameters=framed,
        source_run_id="demonstrated-round",
    )


# The learn round's closing reply, in the shape LEARN_INSTRUCTION asks for: what
# each step produced, what she now knows how to do, and the offer to set it
# running.  That last clause is the message this edge's user turn answers — an
# acceptance is only an acceptance of something.
_PENNY_REPORT = (
    f"Opened the listing, found the price ({_PRICE}), and saved it to "
    f"{_WATCH_COLLECTION}. I know how to do that now — read a listing page and "
    "record the price it shows. Want me to keep it up to date on its own?"
)

# learn → apply: the offer taken up.  It names a cadence, an end condition, and
# a notify ask — but NOT the page, which the round it is answering already read.
_APPLY_TURN = "perfect — do that every hour until 10pm tonight and tell me if it changes"


def _seed_demonstrated_round(db: Database) -> None:
    """Lay down the state the PRECEDING beat ends in, item for item — this edge
    starts where ``elicit → learn`` stops, so its precondition is that beat's
    scored terminal state and nothing else:

    * the teach turn that opened the learn round, and Penny's closing report —
      she ran it, and she says what she now knows how to do
    * the collection her naive write created, holding the price, carrying no
      skill and no program and no schedule (learning instantiates nothing)
    * a learned skill in the registry (seeded by the case's ``seed_skills``)
    * the machine parked in ``learn``, anchored to the teach turn

    The instigating ask ("can you watch this for me?") is deliberately ABSENT.
    It belongs to the beat before — ``idle → elicit`` — and seeding it made the
    classifier read "the task being worked on" as a setup still being specified,
    which is a fair reading of a request that has not been carried out yet.  It
    has been: that is what the learn round did."""
    teach_id = db.messages.log_message(
        direction=PennyConstants.MessageDirection.INCOMING,
        sender=TEST_SENDER,
        content=_TEACH_TURN,
    )
    db.messages.log_message(
        direction=PennyConstants.MessageDirection.OUTGOING,
        sender=PennyConstants.MessageAuthor.PENNY,
        content=_PENNY_REPORT,
    )
    db.memories.create_collection(_WATCH_COLLECTION, "the aurora deck 2 listing price")
    require_memory(db, _WATCH_COLLECTION).write(
        [EntryInput(key=_DEMO_KEY, content=_PRICE)],
        author=PennyConstants.CHAT_AGENT_NAME,
    )
    _park(db, ConversationState.LEARN, anchor_message_id=teach_id)


def _instantiated(db: Database):
    """The collection the taught skill was applied to — WHICHEVER one she chose.

    Which collection a job lands on is deliberately NOT this case's business (code
    owner): she has created several where one was meant since well before the
    machine existed, so where jobs accumulate is a collection-management question
    of its own and grading it here would report that standing problem as a
    transition failure.  This edge owns whether the skill is APPLIED correctly —
    bound, rendered, and scheduled on the terms given — so every check reads the
    row that carries the skill, and the one about reuse rides along unscored."""
    taught = slug_skill_name(_SKILL_NAME)
    applied = [row for row in db.memories.list_all() if row.skill_name == taught]
    return applied[0] if applied else None


def _bound_values(row) -> list[str]:
    """The values she bound into the skill at instantiation, from the collection's
    own provenance column (#1603) — a read, not an inference."""
    return [str(value) for value in json.loads(row.skill_params or "{}").values()]


def _score_learn_to_apply(db: Database, before: set[str], reply: str) -> list[Check]:
    """The taught routine became a live job on the terms they gave — bound to the
    page the round read, rendered, scheduled, and notifying — without re-running
    the round to answer.  WHERE that job lives is not scored (see
    ``_instantiated``); it rides along as an advisory so the choice stays visible."""
    row = _instantiated(db)
    created = new_collections(db, before)
    bound = _bound_values(row) if row else []
    reused = row is not None and row.name == _WATCH_COLLECTION
    sets = count_tool_calls(db, "collection_set")
    return [
        Check(
            "state: she set the job up with collection_set",
            tool_was_called(db, "collection_set"),
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
        Check(
            "state: the page she was taught on is what she bound",
            any(LISTING_URL in value for value in bound),
            rationale=f"bound {bound}",
            kind="state",
        ),
        Check(
            "state: it runs hourly (the cadence they asked for)",
            row is not None and row.collector_interval_seconds == 3600,
            rationale=f"interval {row and row.collector_interval_seconds}",
            kind="state",
        ),
        Check(
            "state: it stops tonight (the end condition they gave)",
            row is not None and row.expires_at is not None,
            kind="state",
        ),
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
        Check(
            "reply: she says what will happen now, naming the cadence",
            any(token in reply.lower() for token in ("hour", "60 min")),
            kind="reply",
        ),
        # Advisory — the collection-management question, parked (code owner): does
        # the job land on the collection the round already wrote into, or on a new
        # one?  Visible every run, graded never, so the standing tendency to spread
        # across collections is measured here without this edge answering for it.
        Check(
            "state: applied onto the collection the round wrote into (not a new one)",
            reused,
            rationale=(
                None
                if reused
                else (
                    f"applied to {row.name if row else None}, "
                    f"created {[each.name for each in created]}"
                )
            ),
            scored=False,
            kind="state",
        ),
        Check(
            "calls: one collection_set call",
            sets == 1,
            rationale=f"{sets} calls" if sets != 1 else None,
            scored=False,
            kind="proc",
        ),
        Check(
            "calls: the machine landed in apply",
            _landed_state(db) == ConversationState.APPLY.value,
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
async def test_learn_to_apply_instantiates_the_taught_skill(chat_eval: ChatEval) -> None:
    """learn → apply: parked on the offer the demonstrated round ended with, the
    user accepts and adds the job's terms.  She binds the taught skill onto the
    collection that round already wrote into — one `collection_set`, the page
    taken from the round rather than asked for again — and does NOT re-run the
    round to answer."""
    await chat_eval(
        case_id="transition-learn-to-apply",
        message=_APPLY_TURN,
        seed=_seed_demonstrated_round,
        seed_skills=[learn_to_apply_fixture_skill()],
        score=_score_learn_to_apply,
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )
