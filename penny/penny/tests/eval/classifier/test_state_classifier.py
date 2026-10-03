"""The conversation-state classifier, measured IN ISOLATION: one case per DECISION
the draw makes (#1706/#2055).

Isolation and aggregate are both wanted (the #2004 ruling).  The chat transition
cases enact these same edges end to end, deliberately rather than redundantly: a
case here hands the draw a lean snapshot and asserts WHICH state it picked, so a
miss is attributable to the draw itself, while an enacted case asserts where the
machine landed after a whole turn.  They catch different things, and the canonical
set holds both.

The classifier's axis is the DECISION rather than the edge, which is why fourteen
out-edges carry seventeen cases.  An edge carries a second case where the door the
draw has to decline is a different door: ``presented_edges`` structurally withholds
every skill-gated state when the registry holds nothing, so idle → elicit drawn
from a cold registry declines LEARN where the same edge drawn from a populated one
declines APPLY.  An empty registry under ordinary conversation is not such a case —
it removes doors from the union the passing-mention hold already declines and adds
none — and switching a running job's notifications is one decision whichever way
the switch goes.

Each case takes ONE decision and says it five ways — five wordings of one ask, one
world, one set of facts, pooled into a cohort of fifteen and claimed against
``docs/eval-case-design.md``.  Snapshots are built PER SAMPLE by the production
``build_snapshot``, so what varies between an empty-registry case and a
populated-registry one is exactly what varies in production: whether the registry
holds skills.

Every case reports an EMPTY ``STORE`` and an EMPTY ``PROVENANCE``, and each says
why in its own docstring — a single call moves no machine and writes to no store, and the
two fields the draw returns are closed sets with nothing in them that could have been
invented.  What a gated draw adds is a SECOND ``LANDED`` claim: the routine it bound.

Fictional-but-believable fixtures throughout (the repo is public).
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.skills import (
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    derive_collection_name,
    render_skill,
)
from penny.tests.eval.conftest import (
    CLASSIFY_SKILL,
    CLASSIFY_STATE,
    EVAL_MODELS,
    ClassifierEval,
    ParkedRound,
    Seeder,
    eval_skill,
)
from penny.tests.eval.utils.assertions import Answer, WorldClaim
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import SampleObservation, SpecCategory, output_field
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "state-classifier"


# ── What a ported case here claims, said once ────────────────────────────────
#
# Every ported case in this file makes the same two kinds of claim over the same two
# structured fields, so they are stated once as factories rather than re-spelled per case:
# a second copy of a claim's body is a second place for it to drift.


def _drew(expected: ConversationState) -> WorldClaim:
    """WHICH member of the offered union the draw picked.

    States BOTH directions at once, which is why no "and not the others" claim stands
    beside it: the expected state is what this message should draw, and every other door
    the current state opens is a door it must not go through.  Splitting that into "held
    idle" and "did not apply" would be the same sentence counted twice.

    The drawn state's MEMBERSHIP in the offered union is not claimed anywhere: production
    validates it (``_state_is_bound``) and re-rolls until it holds, so a claim over it
    would run 15/15 by construction and measure the validator.  WHICH member she picked is
    the open question, and it is this one."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = sample.field(CLASSIFY_STATE)
        return drawn == expected.value, f"drew {drawn}"

    return answer


def _bound(expected: str) -> WorldClaim:
    """WHICH routine a skill-gated draw named on its ``SKILL:`` line.

    The draw contract validates that the name is one of the offered candidates and re-rolls
    while it is not, so *that* it named a real routine is closed upstream exactly as
    membership is.  Which ONE it named is the open question, and it is a different question
    from the state: a draw can land in apply and bind the wrong routine, which would set the
    wrong job running.  So a gated case makes two claims where an ungated one makes one.

    The routine's name is a KEY rather than prose — it is quoted verbatim in Known skills
    and copied back — so equality is the right comparison and there is no smaller unique
    datum to shrink to."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        named = sample.field(CLASSIFY_SKILL)
        return named == expected, f"named {named!r}"

    return answer


# ── A populated registry (price-watch + a distractor skill) ──────────────────

_PRICE_SKILL = "watch a listing price for changes"
_CAFE_SKILL = "collect daily cafe specials"
SEEDED_SKILLS = [
    eval_skill(
        _PRICE_SKILL,
        "keep an eye on a product or listing page and record its current price",
        {"url": "the product or listing page whose price to watch"},
    ),
    eval_skill(
        _CAFE_SKILL,
        "read a cafe or bakery menu page and save the day's specials each morning",
        {"url": "the cafe's menu page"},
    ),
]

# The program a seeded JOB stores — one browse step whose page is the routine's own `url`
# parameter, so `render_skill` writes the bound page into the recipe the way the
# instantiation seam does.  Both seeded routines declare the same one parameter, so one
# shape serves either.
_SEED_STEPS = [
    SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="browse",
        arguments={"queries": ["https://example.test"], "extract": "the value"},
        substitutions=[
            SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, parameter="url")
        ],
    )
]

# The cross-domain registry: a discovery routine whose VERB shape — find new X and keep
# them — an ask can share while being about a different domain entirely, beside the price
# watcher.
_CROSS_DOMAIN_SKILLS = [
    eval_skill(
        "find new job listings",
        "search the job boards for newly posted listings matching a role and save them",
        {"role": "the kind of job to look for", "boards": "the job boards to search"},
    ),
    SEEDED_SKILLS[0],
]


# ── idle → idle: the passing mention, with the routine doors on offer ─────
#
# A passing mention of something a known routine could be pointed at, said five ways:
# five wordings of ONE ask, one world, one set of facts.
#
# THE FACTS ARE CONSTANT across the arms because the claim hinges on them.  Every wording
# is the user describing THEIR OWN recurring habit of looking at the auction listings —
# same subject, same habit framing, same recurrence — so what moves between arms is the
# words and nothing else.  A wording that asked Penny to watch the listings would be a
# different behaviour wearing the same case id.

# The case's id and its five arms, named at module level so the deterministic probe in
# ``make check`` can hold them against the world they claim — that apply and request really
# are on offer — before any GPU time is spent on a hold with no door to decline.
HOLD_CASE_ID = "classifier-holds-idle-on-a-passing-mention"

_PASSING_MENTION = "i've been checking the auction listings every day lately"
_PASSING_MENTION_PHRASINGS = (
    "been going through the auction listings most days lately",
    "these days i end up scrolling the auction listings every evening",
    "i've fallen into a habit of reading the auction listings each morning",
    "lately i look at the auction listings pretty much daily",
)
PASSING_MENTION_ARMS = (_PASSING_MENTION, *_PASSING_MENTION_PHRASINGS)

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED agent name.  The case id is a filename; this is
# the contract, and it renders above every number in the report.
_HOLD_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user "
    "mentions in passing something a routine she already knows could be pointed at, Penny "
    "holds the conversation in idle."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_passing_mention_holds_idle_with_the_routine_doors_open(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The passing-mention hold, with candidates dangling: ordinary conversation that names
    a thing a seeded routine plainly covers, said five ways, against a registry that puts
    apply and request on offer.

    **The STORE category is empty for this case, and that is the correct report.**  A
    micro-context is one call that returns a typed result — it moves no machine and writes
    to no store — so there is nothing for a store claim to read.  (What the classifier
    decides is later WRITTEN by the machine, as a ``state_transition`` row; that write is
    ``ConversationMachine``'s and this case never runs it.)

    **The PROVENANCE category is empty too, and for a different reason: it is CLOSED
    UPSTREAM.**  Fact alignment reads a draw's OPEN fields — the ones the model wrote in its
    own words — and this draw has none.  Both fields it returns are closed sets the harness
    itself supplied: ``state`` is membership-validated against the offered union and
    re-rolled until it is a member, and ``skill`` is GENERATED rather than drawn on an
    ungated state — ``StateDraw`` leaves it empty by construction, so a claim that this
    decision bound no routine would be asserting what ``_state_draw`` writes.  There is no
    value here that could have been invented, so there is nothing to trace.
    """
    cohort = await classifier_eval(
        case_id=HOLD_CASE_ID,
        behaviour=_HOLD_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.IDLE),
        model=model,
        state=ConversationState.IDLE,
        ask=_PASSING_MENTION,
        also_asked=_PASSING_MENTION_PHRASINGS,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw held the conversation in idle",
        _drew(ConversationState.IDLE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the docstring.
    # PROVENANCE — empty because it is closed upstream; see the docstring.

    # What is MEASURED — the draw's own structured fields.
    #
    # ``state`` is the only axis this case can read.  ``outcome`` is constant by the
    # completeness gate (a draw that came back anything but DECIDED left the cohort before
    # any of this), and ``skill`` is empty on every sample because idle binds none — which
    # scores 0.000, the same number a cohort in perfect agreement scores and the opposite
    # finding, so measuring it would print a number where there is no measurement.
    #
    # No tool sequence and no reply spread: a single call makes neither.
    cohort.measure(output_field(CLASSIFY_STATE))


# ── Standing jobs: what a case about a job that is already running seeds ─────

_JOB_SCHEDULE = "FREQ=DAILY;BYHOUR=8"


def _standing_jobs(jobs: list[tuple[str, str]], notify: bool) -> Seeder:
    """The jobs a case speaks about, standing in the world as configured collections —
    all notifying the same way.

    ``notify`` is the state the case's ask is about changing: an ask to silence a job seeds
    it TALKING, so the rendered row carries what the ask would change.  A job whose
    notifications already matched the ask would be a different case.

    A DIRECT seed rather than the production instantiation seam: the classifier reads a
    job's name, routine, notify flag and schedule and never its program, so standing the
    full seam up here would build machinery no scored check can see.  What is kept faithful
    is everything the row is selected and rendered BY — the derived name, the routine's
    registry name, the bound value, an RRULE schedule, and a program rendered by the
    production `render_skill` (a prose prompt would be a row the collector reads as a
    config defect, #1916)."""

    def seed(db: Database) -> None:
        for skill, page in jobs:
            db.memories.create_collection(
                derive_collection_name(skill, [page]),
                f"what {page} says",
                extraction_prompt=render_skill(_SEED_STEPS, {"url": page}),
                schedule=_JOB_SCHEDULE,
                notify=notify,
                skill_name=skill,
                skill_params={"url": page},
            )

    return seed


# ── The parked machine — a round waiting in request ──────────────────────────

# The instigating ask of the round parked in request: a subject, and no address.
_KAYAK_ASK = "keep an eye on the price of the harbor kayak rental page for me"


# ── The parked machine — a round waiting in elicit ───────────────────────────

# The parked-elicit context: the instigating ask (the cold-start ask that opened
# the teach loop) and the teach question the reply answers.  Replies are only
# classifiable against what they answer, so both parked-snapshot fields render.
_FERRY_ASK = "hey can you keep an eye on the harbor ferry timetable for me?"
_TEACH_QUESTION = (
    "I don't know how to do that yet — can you teach me? "
    "What should I read, look for, and remember?"
)


# ── Parked learn — re-entry after a failed demo round ────────────────────────

# The parked-learn context: the demo round failed (the honest failure report is
# the assistant's last turn), the machine holds in learn, and the reply decides
# whether it stays there — a correction that CARRIES instructions, or a question
# that carries none.
_FAILED_ROUND_REPORT = (
    "I tried, but the timetable page wouldn't load, so I couldn't save anything. "
    "Should I try again, or is there a different page I should read?"
)


# ── The six edges with no isolated coverage (#2055 tranche A) ─────────────────
#
# Six ported cases, one per edge, each five wordings of ONE decision over one world.
#
# THREE CLAIM CATEGORIES, and two of them are EMPTY on every case in this file — stated
# once here and referenced from each case rather than restated six times:
#
#   STORE is empty because the shape has nothing there.  A micro-context is ONE call
#   returning a typed result: it moves no machine and writes to no store, so there is no
#   trail for a store claim to read.  (What the classifier decides is later written by
#   ``ConversationMachine`` as a ``state_transition`` row; that write is the machine's and
#   no case here runs it.)
#
#   PROVENANCE is empty because it is CLOSED UPSTREAM.  Fact alignment reads a draw's OPEN
#   fields — the ones the model wrote in its own words — and this draw returns none.  Both
#   fields it returns are closed sets the harness itself supplied: ``state`` is validated
#   against the offered union and re-rolled until it is a member, and ``skill`` is validated
#   against the offered candidates on a gated draw and GENERATED empty on an ungated one.
#   Nothing here could have been invented, so there is nothing to trace.
#
# WHAT IS CLAIMED is the one open question: WHICH member the draw picked — ``_drew`` at the
# top of the file.  For an ungated state that is the whole claim set; for a gated one there
# is a second, ``_bound``, because the ``SKILL:`` line is a second choice out of a second
# offered set and validation guarantees only that it names SOME candidate.


# ── idle → request: a covered ask that is one value short ────────────────────
#
# THE FACTS ARE CONSTANT across the arms: every wording asks for the SAME subject (the
# camera kit listing), for the SAME ongoing job (watch its price), and none of them says
# WHICH page — which is the whole entry condition.  A wording that carried an address
# would be the idle → apply behaviour wearing this case's id.
#
# The registry is seeded, and that is what makes the case mean anything: ``presented_edges``
# withholds every skill-gated state when there are no candidates, so request cannot be drawn
# at all against an empty registry.  The distractor is seeded for the SKILL claim's sake —
# with one candidate, naming the right one is not a choice.

REQUEST_SHORT_CASE_ID = "classifier-draws-request-when-a-known-routine-is-a-value-short"

_REQUEST_SHORT_ASK = "can you keep an eye on the price of the camera kit listing for me?"
_REQUEST_SHORT_PHRASINGS = (
    "i'd like the camera kit listing's price watched",
    "could you track what the camera kit listing is going for?",
    "keep tabs on the price of that camera kit listing",
    "watch the camera kit listing and keep a record of its price",
)
REQUEST_SHORT_ARMS = (_REQUEST_SHORT_ASK, *_REQUEST_SHORT_PHRASINGS)

_REQUEST_SHORT_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when a routine "
    "she already knows covers the ask but the message never says which page it is about, "
    "Penny draws request and binds that routine."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_covered_ask_missing_its_page_draws_request(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The idle → request draw: a known routine covers the ask, and the one value it
    declares is not in the message.

    STORE and PROVENANCE are empty; the section comment above says why for every case in
    this file.  Two LANDED claims, because this state is skill-gated: which state, and
    which routine.
    """
    cohort = await classifier_eval(
        case_id=REQUEST_SHORT_CASE_ID,
        behaviour=_REQUEST_SHORT_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.REQUEST),
        model=model,
        state=ConversationState.IDLE,
        ask=_REQUEST_SHORT_ASK,
        also_asked=_REQUEST_SHORT_PHRASINGS,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed fields of the typed result, asserted by equality
    cohort.claim(
        "state: the draw parked the round in request",
        _drew(ConversationState.REQUEST),
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: the draw bound the routine that covers the ask",
        _bound(_PRICE_SKILL),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    # What is MEASURED — both structured fields the draw fills.  ``skill`` is measured here
    # and not on the ungated cases because this state BINDS one: on an ungated draw it is
    # empty on every sample, which scores 0.000 — the same number perfect agreement scores.
    # ``outcome`` is constant by the completeness gate, and a single call makes no tool
    # sequence and no reply.
    cohort.measure(output_field(CLASSIFY_STATE), output_field(CLASSIFY_SKILL))


# ── idle → learn: teaching that nobody asked for ─────────────────────────────
#
# THE FACTS ARE CONSTANT: every wording teaches the SAME routine — the same page, the same
# cadence, the same thing to remember, the same store verb — and every one says outright
# that it is teaching.  Both halves are the edge's own condition, so an arm missing either
# would be a different behaviour.
#
# AT MOST ONE ARM USES THE CONDITION'S OWN EXEMPLARS.  The shipped idle → learn condition
# illustrates the declaration with three phrases ('let me teach you', 'here's how', 'new job
# for you'), and a pool built out of them measures whether the draw can match a string it was
# handed rather than whether it can read a teach.  Only the first arm opens that way; the
# rest declare the same thing in words the prompt never says, so four of the five are a real
# read (#1783 — a fixture must never lend its own phrasing to the prompt under test).
#
# Seeded registry rather than a cold one, deliberately.  The doors this draw has to decline
# are the skill-gated ones, and they are structurally absent when the registry is empty —
# so a learn measured cold is a learn with nothing to lose to.  The seeded routines are
# watch-shaped and read a page, which is what the edge's own "choose learn even if a known
# skill looks close" clause exists for; neither covers a ferry timetable.

UNPROMPTED_TEACH_CASE_ID = "classifier-draws-learn-when-teaching-arrives-unprompted"

_TEACH_PAGE = "harborferries.example/timetable"
_UNPROMPTED_TEACH_ASK = (
    f"new job for you: each morning read {_TEACH_PAGE} and remember the first sailing"
)
_UNPROMPTED_TEACH_PHRASINGS = (
    f"i've got a new routine for you — each morning read {_TEACH_PAGE} and remember the "
    "first sailing",
    f"this is how i want it done: each morning read {_TEACH_PAGE} and remember the first sailing",
    f"here's a routine i want you doing each morning: read {_TEACH_PAGE} and remember the "
    "first sailing",
    f"adding one to your repertoire — each morning read {_TEACH_PAGE} and remember the "
    "first sailing",
)
UNPROMPTED_TEACH_ARMS = (_UNPROMPTED_TEACH_ASK, *_UNPROMPTED_TEACH_PHRASINGS)

_UNPROMPTED_TEACH_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user "
    "starts teaching a new routine unprompted and their message carries the steps, Penny "
    "draws learn — she does not read the teaching as an ask to be taught, nor as one of "
    "the routines she already has."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_unprompted_teaching_draws_learn(classifier_eval: ClassifierEval, model: str) -> None:
    """The idle → learn draw: the user says they are teaching, and the steps are in the
    message, from a standing start rather than from inside a teach loop.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    learn binds no routine, so there is no second choice to claim.
    """
    cohort = await classifier_eval(
        case_id=UNPROMPTED_TEACH_CASE_ID,
        behaviour=_UNPROMPTED_TEACH_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.LEARN),
        model=model,
        state=ConversationState.IDLE,
        ask=_UNPROMPTED_TEACH_ASK,
        also_asked=_UNPROMPTED_TEACH_PHRASINGS,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw took the round into learn",
        _drew(ConversationState.LEARN),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    # ``skill`` is generated empty on this ungated state, so measuring it would print
    # 0.000 where there is no measurement.
    cohort.measure(output_field(CLASSIFY_STATE))


# ── elicit → idle: the teach round is called off ─────────────────────────────
#
# THE FACTS ARE CONSTANT: every wording calls off the SAME round — the ferry timetable one
# the teach question opened — and none of them carries instructions (which would be learn)
# or asks anything back (which would leave the machine parked in elicit).
#
# No skills are seeded, and nothing is lost by that: elicit's out-edges carry no skill-gated
# state at all, so the union this draw chooses from is the same either way.  A round is in
# elicit BECAUSE nothing covered the ask, and an empty registry is the world that says so.

ELICIT_CALLED_OFF_CASE_ID = "classifier-falls-to-idle-when-the-elicit-round-is-called-off"

_ELICIT_CALLED_OFF_ASK = "actually never mind, forget the ferry timetable thing"
_ELICIT_CALLED_OFF_PHRASINGS = (
    "eh, drop it — i don't need the ferry timetable thing after all",
    "actually let's skip the ferry timetable one, i've changed my mind",
    "forget it, i don't want the ferry timetable thing set up anymore",
    "never mind the ferry timetable thing, it's not worth the trouble",
)
ELICIT_CALLED_OFF_ARMS = (_ELICIT_CALLED_OFF_ASK, *_ELICIT_CALLED_OFF_PHRASINGS)

_ELICIT_CALLED_OFF_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user "
    "calls off the round the teach question opened, Penny falls to idle."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_called_off_elicit_round_falls_to_idle(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The elicit → idle draw: the break-out edge, taken because the user dropped the task.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    idle binds no routine.
    """
    cohort = await classifier_eval(
        case_id=ELICIT_CALLED_OFF_CASE_ID,
        behaviour=_ELICIT_CALLED_OFF_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.ELICIT, ConversationState.IDLE),
        model=model,
        state=ConversationState.ELICIT,
        ask=_ELICIT_CALLED_OFF_ASK,
        also_asked=_ELICIT_CALLED_OFF_PHRASINGS,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw broke the round out to idle",
        _drew(ConversationState.IDLE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── learn → apply: the demonstrated round is accepted ────────────────────────
#
# THE FACTS ARE CONSTANT in the world rather than in the words: every arm answers the SAME
# demonstration report and offer, and every one is a plain acceptance carrying no new
# instructions (which would hold the round in learn) and no retraction (which would break
# it out to idle).
#
# AT MOST ONE ARM USES THE CONDITION'S OWN EXEMPLARS.  The shipped learn → apply condition
# illustrates the acceptance with three ("a yes, a great, a go-ahead"), so only the first
# arm is built on them; the rest accept in words the prompt never says (#1783).
#
# The registry holds the routine the round taught, because that is where a real round
# stands at this moment: run-end extraction files the routine under the name the round
# pinned, so by the time the offer is answered it is a candidate like any other.  The two
# standing distractors read a page and record something from it, exactly as it does — which
# is what makes WHICH routine a question worth claiming.

_FERRY_SKILL = "record the first ferry sailing each day"
_FERRY_ROUTINE = eval_skill(
    _FERRY_SKILL,
    "read a ferry timetable page and remember the first sailing of the day",
    {"url": "the ferry timetable page to read"},
)
_ACCEPTED_ROUND_SKILLS = [_FERRY_ROUTINE, *SEEDED_SKILLS]

# The learn turn's own close, in the shape ``Prompt.LEARN_INSTRUCTION`` asks for: what each
# step produced, what it now knows how to do, and an offer to set it running.
_DEMONSTRATION_REPORT = (
    "I read harborferries.example/timetable and saved the first sailing — 05:20. "
    "So I know how to do this now: read the ferry timetable and remember the first "
    "sailing of the day. Want me to set it up to run on its own each morning?"
)

OFFER_ACCEPTED_CASE_ID = "classifier-draws-apply-when-the-offer-is-accepted"

_OFFER_ACCEPTED_ASK = "yes, that's exactly it — go ahead"
_OFFER_ACCEPTED_PHRASINGS = (
    "perfect, that's what i wanted",
    "nice, that's the shape i was after — let's have it",
    "yep, you nailed it, keep it",
    "that's the one, go with it",
)
OFFER_ACCEPTED_ARMS = (_OFFER_ACCEPTED_ASK, *_OFFER_ACCEPTED_PHRASINGS)

_OFFER_ACCEPTED_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user "
    "accepts the round that was just demonstrated, Penny draws apply and binds the routine "
    "that round taught."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_accepted_offer_draws_apply(classifier_eval: ClassifierEval, model: str) -> None:
    """The learn → apply draw: a plain acceptance of what was just demonstrated, with the
    round's own routine standing in the registry beside two near neighbours.

    STORE and PROVENANCE are empty; the section comment above says why.  Two LANDED claims,
    because this state is skill-gated: which state, and which routine.
    """
    cohort = await classifier_eval(
        case_id=OFFER_ACCEPTED_CASE_ID,
        behaviour=_OFFER_ACCEPTED_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.LEARN, ConversationState.APPLY),
        model=model,
        state=ConversationState.LEARN,
        ask=_OFFER_ACCEPTED_ASK,
        also_asked=_OFFER_ACCEPTED_PHRASINGS,
        penny_last_turn=_DEMONSTRATION_REPORT,
        task_anchor=_FERRY_ASK,
        seed_skills=_ACCEPTED_ROUND_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed fields of the typed result, asserted by equality
    cohort.claim(
        "state: the draw took the round into apply",
        _drew(ConversationState.APPLY),
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: the draw bound the routine the round taught",
        _bound(_FERRY_SKILL),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE), output_field(CLASSIFY_SKILL))


# ── request → apply: the missing value arrives ───────────────────────────────
#
# THE FACTS ARE CONSTANT: every arm supplies the SAME page, byte for byte, and adds nothing
# else — the words around it are the only thing that moves.  That constancy is what lets the
# SKILL claim name a routine: the page is the value the parked round is short of, and the
# routine that declares it is the one the draw has to bind.
#
# THE ROUND'S BINDING IS DECLARED (#2084), because production reaches request only through
# the binder and records what it settled on the move — so a draw parked here is always shown
# the `## The details this task is waiting on` section, naming the routine, what the user has
# already given and what is still needed.  ``_PRICE_SKILL`` declares one parameter and this
# round settled none of it: the ask named a subject and no address, which is what parked it.
#
# Penny's last turn says in PLAIN WORDS what she would do and asks for the missing part by
# what it is, which is what ``Prompt.REQUEST_INSTRUCTION`` instructs a request turn to do —
# it never quotes the registry's own name at the user.  The routine's own name reaches the
# draw the way production puts it there, on the waiting-on section, so binding it is a COPY
# rather than a resolution — and the claim is still open, because the other seeded routine is
# equally copyable and only one of them is the one this round is parked on.

_PARKED_ON_THE_PRICE_WATCH = ParkedRound(skill=_PRICE_SKILL)

_REQUEST_TURN = (
    "I can keep an eye on a listing page and note its price whenever it changes — "
    "which page should I be watching?"
)
_KAYAK_PAGE = "harborkayak.example/rentals/sea-touring"

VALUE_ARRIVED_CASE_ID = "classifier-draws-apply-when-the-missing-value-arrives"

_VALUE_ARRIVED_ASK = _KAYAK_PAGE
_VALUE_ARRIVED_PHRASINGS = (
    f"it's {_KAYAK_PAGE}",
    f"the page is {_KAYAK_PAGE}",
    f"use {_KAYAK_PAGE} for it",
    f"here you go — {_KAYAK_PAGE}",
)
VALUE_ARRIVED_ARMS = (_VALUE_ARRIVED_ASK, *_VALUE_ARRIVED_PHRASINGS)

_VALUE_ARRIVED_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the page the "
    "parked round was waiting on arrives, Penny draws apply and binds the routine she had "
    "named."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_missing_value_arriving_draws_apply(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The request → apply draw: the round parked asking for one detail, and this message
    is that detail.

    STORE and PROVENANCE are empty; the section comment above says why.  Two LANDED claims,
    because this state is skill-gated: which state, and which routine.
    """
    cohort = await classifier_eval(
        case_id=VALUE_ARRIVED_CASE_ID,
        behaviour=_VALUE_ARRIVED_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.REQUEST, ConversationState.APPLY),
        model=model,
        state=ConversationState.REQUEST,
        ask=_VALUE_ARRIVED_ASK,
        also_asked=_VALUE_ARRIVED_PHRASINGS,
        penny_last_turn=_REQUEST_TURN,
        task_anchor=_KAYAK_ASK,
        parked_round=_PARKED_ON_THE_PRICE_WATCH,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed fields of the typed result, asserted by equality
    cohort.claim(
        "state: the draw took the round into apply",
        _drew(ConversationState.APPLY),
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: the draw bound the routine the round was parked on",
        _bound(_PRICE_SKILL),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE), output_field(CLASSIFY_SKILL))


# ── request → idle: the parked round is called off ───────────────────────────
#
# THE FACTS ARE CONSTANT in what every arm withholds, which is what this edge turns on: none
# of them supplies the page (that is apply, the case above), and none of them says the named
# routine was the wrong one while still wanting the task done (that is elicit, request's
# other door).  What is left is the break-out, and idle is the state that owns it.
#
# The registry is seeded so both of those doors are really on offer — with no candidates,
# ``presented_edges`` withholds apply outright and the call-off would be declining a door
# that was never open.  The round declares the same binding the case above does (#2084): a
# call-off measured against a document that never said what the round was waiting for is a
# call-off of nothing in particular.

REQUEST_CALLED_OFF_CASE_ID = "classifier-falls-to-idle-when-a-parked-request-is-called-off"

_REQUEST_CALLED_OFF_ASK = "actually never mind, forget the whole thing"
_REQUEST_CALLED_OFF_PHRASINGS = (
    "eh, drop it — i don't need that watched after all",
    "actually let's skip it, i've changed my mind",
    "forget it, don't set that up",
    "never mind, i'll just look at it myself",
)
REQUEST_CALLED_OFF_ARMS = (_REQUEST_CALLED_OFF_ASK, *_REQUEST_CALLED_OFF_PHRASINGS)

_REQUEST_CALLED_OFF_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user "
    "calls off the round that is parked waiting on a detail, Penny falls to idle."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_called_off_parked_request_falls_to_idle(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The request → idle draw: the round asked for one detail and the user dropped the
    task instead of supplying it.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    idle binds no routine.
    """
    cohort = await classifier_eval(
        case_id=REQUEST_CALLED_OFF_CASE_ID,
        behaviour=_REQUEST_CALLED_OFF_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.REQUEST, ConversationState.IDLE),
        model=model,
        state=ConversationState.REQUEST,
        ask=_REQUEST_CALLED_OFF_ASK,
        also_asked=_REQUEST_CALLED_OFF_PHRASINGS,
        penny_last_turn=_REQUEST_TURN,
        task_anchor=_KAYAK_ASK,
        parked_round=_PARKED_ON_THE_PRICE_WATCH,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw broke the round out to idle",
        _drew(ConversationState.IDLE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── The six decisions inside covered edges (#2055 tranche B) ──────────────────
#
# Tranche A took the six edges nothing had ever drawn in isolation.  These six sit INSIDE
# edges the file already covers, and each is its own decision rather than another wording of
# one: what makes two cases two is that a correct sample for one would be wrong for the
# other, that the union the draw chooses from opens a different door to decline, or that
# they are two asks reaching one end state.
#
# THE THREE CLAIM CATEGORIES ARE THE SECTION ABOVE'S, unchanged and not restated here:
# STORE is empty because a micro-context is one call that writes to no store, and
# PROVENANCE is empty because both fields the draw returns are closed sets the harness
# supplied.  What is claimed is WHICH member the draw picked (``_drew``), plus — on a
# skill-gated draw — WHICH routine it bound (``_bound``).
#
# TWO of the six seed no routine at all, and it means two different things.  For the one
# parked in IDLE it is the case itself: ``presented_edges`` withholds every skill-gated
# state when there are no candidates, so the union is ``{learn, elicit, idle}`` and the door
# the draw has to decline is LEARN, where its populated siblings decline apply — a
# materially different decision, not an easier one.  For the one parked in ELICIT nothing is
# lost either way: elicit's out-edges carry no skill-gated state, and an empty registry is
# simply the world a round parked there is in, since it is parked BECAUSE nothing covered
# the ask.
#
# No case here holds IDLE against an empty registry.  That union is the passing-mention
# hold's with two doors removed and none added, so a hold measured there declines nothing
# the passing-mention case does not already decline.
#
# The notifications boundary is ONE case.  Changing how a running job behaves is the
# decision, and the direction of the switch is a wording of it: the world, the union and the
# landing are the same either way.


# ── idle → idle: the notifications of a job that is already running ───────────
#
# The #1927 boundary.  The skill-gated doors are about STARTING a task, so an ask that
# changes how something already set up behaves holds idle — and the resemblance is the whole
# difficulty: a watch-shaped routine sits in Known skills looking like it covers the subject.
#
# THE FACTS ARE CONSTANT across the arms: every wording names the SAME job — the camera kit
# price watch — and asks for the SAME switch, off.  What moves between arms is the verb
# around the word (turn off · switch off · disable · shut off), which is the code-owner
# ruling on this boundary: the register is settled product vocabulary, so every
# phrasing says NOTIFICATIONS.
#
# THE SUBJECT EXISTS AS A JOB, standing as a configured collection so the snapshot's
# ``## Jobs already running`` section renders it — because a subject the section does NOT
# list is a subject holding idle would be WRONG about: with nothing running by that name,
# "turn its notifications off" has no existing thing to adjust.  A second job stands beside
# it, so resolving "the camera kit price watch" to a derived name is a read among rows
# rather than the only row there is.
#
# THE JOBS ARE SEEDED TALKING, so the rendered row carries the state the ask is about
# changing.  A job whose notifications were already off would be a different case.

_PORTED_JOBS = [
    (_PRICE_SKILL, "foxden.example/camera-kit"),
    (_CAFE_SKILL, "pinehollow.example/bakery"),
]

NOTIFY_OFF_CASE_ID = "classifier-holds-idle-when-a-running-jobs-notifications-are-switched-off"

_NOTIFY_OFF_ASK = "turn notifications off for the camera kit price watch"
_NOTIFY_OFF_PHRASINGS = (
    "switch the camera kit price watch's notifications off",
    "can you disable notifications for the camera kit price watch",
    "notifications off for the camera kit price watch, please",
    "shut the notifications off for the camera kit price watch",
)
NOTIFY_OFF_ARMS = (_NOTIFY_OFF_ASK, *_NOTIFY_OFF_PHRASINGS)

_NOTIFY_OFF_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the ask turns "
    "notifications off for a job that is already running, Penny holds the conversation in "
    "idle."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_switching_a_running_jobs_notifications_off_draws_idle(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The adjustment boundary: one running job's notifications switched off, asked for five
    ways, with the routine that plainly covers the subject standing in Known skills.

    The subject's coverage is load-bearing rather than decorative.  A subject no seeded
    routine covers can hold idle by the pre-existing no-coverage route, so the sample would
    score green while proving nothing about this boundary; with coverage granted, the only
    thing left that can hold these samples idle is that the ask CHANGES a job rather than
    starting one.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    idle binds no routine.
    """
    cohort = await classifier_eval(
        case_id=NOTIFY_OFF_CASE_ID,
        behaviour=_NOTIFY_OFF_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.IDLE),
        model=model,
        state=ConversationState.IDLE,
        ask=_NOTIFY_OFF_ASK,
        also_asked=_NOTIFY_OFF_PHRASINGS,
        seed=_standing_jobs(_PORTED_JOBS, notify=True),
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw held the conversation in idle",
        _drew(ConversationState.IDLE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── idle → elicit: a setup ask against a registry holding nothing ─────────────
#
# THE FACTS ARE CONSTANT: every wording asks for the SAME ongoing job — keep an eye on the
# harbour ferry timetable — and not one of them says HOW.  Both halves are the edge's own
# condition, so an arm carrying steps would be the idle → learn behaviour wearing this
# case's id, and an arm asking for something done once would be idle's.
#
# The registry is EMPTY, which is what makes this a different decision from the populated
# elicit cases rather than a weaker copy of them.  With no candidates, apply and request are
# structurally withheld, so the door this draw has to decline is LEARN — a setup ask read as
# the instructions themselves — and nothing about coverage is being measured at all.  An
# elicit measured against a populated registry is the false-apply guard; this one is the
# cold ask, and a round is in elicit precisely BECAUSE nothing covered it.

COLD_ELICIT_CASE_ID = "classifier-elicits-on-a-cold-registry"

_COLD_ELICIT_ASK = "hey can you keep an eye on the harbor ferry timetable for me?"
_COLD_ELICIT_PHRASINGS = (
    "could you watch the harbor ferry timetable for me from now on?",
    "i'd like you to keep track of the harbor ferry timetable going forward",
    "can you stay on top of the harbor ferry timetable for me?",
    "keep tabs on the harbor ferry timetable for me, would you?",
)
COLD_ELICIT_ARMS = (_COLD_ELICIT_ASK, *_COLD_ELICIT_PHRASINGS)

_COLD_ELICIT_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user asks "
    "for something that keeps running on its own and the registry holds no routine at all, "
    "Penny opens a teach round in elicit."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_setup_ask_on_a_cold_registry_elicits(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The cold idle → elicit draw: an ongoing job asked for five ways, with nothing in the
    registry and no steps in the message.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    elicit binds no routine.
    """
    cohort = await classifier_eval(
        case_id=COLD_ELICIT_CASE_ID,
        behaviour=_COLD_ELICIT_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.ELICIT),
        model=model,
        state=ConversationState.IDLE,
        ask=_COLD_ELICIT_ASK,
        also_asked=_COLD_ELICIT_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw opened a teach round in elicit",
        _drew(ConversationState.ELICIT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── idle → apply: a cold ask among several near neighbours ────────────────────
#
# THE FACTS ARE CONSTANT: every wording names the SAME page, byte for byte, and asks for the
# SAME thing to be watched on it — its price.  Carrying the page is the entry condition (an
# arm withholding it would be the idle → request behaviour), and carrying the price is what
# decides WHICH routine covers the ask.
#
# FOUR near-neighbour routines stand in the registry, all of them page-reading
# record-a-value jobs: the price watch this ask is covered by, a cafe menu, a ferry
# timetable and a courier's tracking page.  That is what makes the SKILL claim a claim.
# Membership is validated and re-rolled upstream, so *that* the draw named some candidate is
# closed; with four plausible ones offered, *which* it named is a real choice, and a draw
# that lands in apply while binding the wrong routine would set the wrong job running.
#
# No arm quotes the routine's registry name.  Resolving "watch the price on this page" to
# the routine that declares a url is the READ the draw is expected to make, and naming it
# would hand the model half the case.

_PARCEL_SKILL = "check a parcel's delivery status"
_PARCEL_ROUTINE = eval_skill(
    _PARCEL_SKILL,
    "read a courier's tracking page and record where a parcel has got to",
    {"url": "the courier's tracking page for the parcel"},
)
_NEAR_NEIGHBOUR_SKILLS = [*SEEDED_SKILLS, _FERRY_ROUTINE, _PARCEL_ROUTINE]

_BOARD_PAGE = "driftline.example/boards/7-2"

COVERED_ASK_CASE_ID = "classifier-applies-and-names-the-routine-that-covers-the-ask"

_COVERED_ASK = f"can you watch the price on {_BOARD_PAGE}?"
_COVERED_ASK_PHRASINGS = (
    f"keep an eye on the price at {_BOARD_PAGE}",
    f"track the price of {_BOARD_PAGE} for me",
    f"watch {_BOARD_PAGE} and record its price",
    f"keep tabs on the price of {_BOARD_PAGE}",
)
COVERED_ASK_ARMS = (_COVERED_ASK, *_COVERED_ASK_PHRASINGS)

_COVERED_ASK_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when a cold ask "
    "supplies everything one of several near-neighbour routines needs, Penny draws apply and "
    "binds the routine that covers it."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_fully_supplied_cold_ask_names_its_routine(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The idle → apply draw with the routine in question: one covered ask said five ways,
    against four routines that all read a page and record something from it.

    STORE and PROVENANCE are empty; the section comment above says why.  Two LANDED claims,
    because this state is skill-gated: which state, and which routine.
    """
    cohort = await classifier_eval(
        case_id=COVERED_ASK_CASE_ID,
        behaviour=_COVERED_ASK_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.APPLY),
        model=model,
        state=ConversationState.IDLE,
        ask=_COVERED_ASK,
        also_asked=_COVERED_ASK_PHRASINGS,
        seed_skills=_NEAR_NEIGHBOUR_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed fields of the typed result, asserted by equality
    cohort.claim(
        "state: the draw took the round into apply",
        _drew(ConversationState.APPLY),
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: the draw bound the routine that covers the ask",
        _bound(_PRICE_SKILL),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE), output_field(CLASSIFY_SKILL))


# ── idle → apply: chat and a covered ask in one message ───────────────────────
#
# THE FACTS ARE CONSTANT on both halves of the message, because the message has two halves
# and the case is about how they combine: the same greeting opens every arm, and the same
# page and the same price close it.  What varies is only the words, so the ONE thing under
# test is that the chat half never suppresses the routine half.
#
# The ask itself is complete — the page is there and the price is named — because boundness
# is the apply-versus-request split's business and not this case's.  An arm that dropped the
# page would be measuring that split while wearing this case's id.

_GRINDER_PAGE = "beanhouse.example/grinders/ek43"

MIXED_MESSAGE_CASE_ID = "classifier-applies-when-chat-carries-a-covered-ask"

_MIXED_MESSAGE_ASK = f"morning! oh and can you watch the price on {_GRINDER_PAGE}?"
_MIXED_MESSAGE_PHRASINGS = (
    f"good morning — also, keep an eye on the price at {_GRINDER_PAGE}?",
    f"morning! quick one: track the price of {_GRINDER_PAGE}",
    f"hey, good morning. one more thing — watch {_GRINDER_PAGE} and its price?",
    f"morning! and while i think of it, keep tabs on the price of {_GRINDER_PAGE}",
)
MIXED_MESSAGE_ARMS = (_MIXED_MESSAGE_ASK, *_MIXED_MESSAGE_PHRASINGS)

_MIXED_MESSAGE_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when one message "
    "carries a chat preamble and a covered ask together, Penny draws apply and binds the "
    "covering routine."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_chat_carrying_a_covered_ask_draws_apply(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The mixed-message boundary: a greeting and a covered ask in one message, said five
    ways, against the two seeded routines.

    A different case from the cold covered ask above under §6's SCENARIO rule, not under the
    correct-sample one: a sample drawing apply and binding the price watcher is correct for
    both, so nothing here would be wrong there.  What makes them two is that they are two
    asks reaching one end state, each carrying its own fifteen — and folding them would
    average two behaviours into one score and report the spread as instability.  The asks
    differ in what tempts the draw away: there, four near neighbours, so the open question is
    WHICH routine; here, two, with a chat half that could suppress the ask entirely.

    STORE and PROVENANCE are empty; the section comment above says why.  Two LANDED claims,
    because this state is skill-gated: which state, and which routine.
    """
    cohort = await classifier_eval(
        case_id=MIXED_MESSAGE_CASE_ID,
        behaviour=_MIXED_MESSAGE_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.APPLY),
        model=model,
        state=ConversationState.IDLE,
        ask=_MIXED_MESSAGE_ASK,
        also_asked=_MIXED_MESSAGE_PHRASINGS,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed fields of the typed result, asserted by equality
    cohort.claim(
        "state: the draw took the round into apply",
        _drew(ConversationState.APPLY),
        SpecCategory.LANDED,
    )
    cohort.claim(
        "state: the draw bound the routine that covers the ask",
        _bound(_PRICE_SKILL),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE), output_field(CLASSIFY_SKILL))


# ── elicit → elicit: the teach question has not been answered ─────────────────
#
# THE FACTS ARE CONSTANT: every wording asks the SAME question back — what would Penny need
# from them — and not one of them answers the teach question.  That is the self-edge's whole
# condition, so an arm carrying steps would be the elicit → learn behaviour and an arm
# calling the round off would be elicit → idle.
#
# No skills are seeded, and nothing is lost by it: elicit's out-edges carry no skill-gated
# state, so the union is the same either way, and a round is parked in elicit BECAUSE
# nothing covered the ask.

STILL_CLARIFYING_CASE_ID = "classifier-stays-parked-when-the-teach-question-is-unanswered"

_STILL_CLARIFYING_ASK = "what would you need from me to do that?"
_STILL_CLARIFYING_PHRASINGS = (
    "what do you need from me for it?",
    "what would you want me to give you for that?",
    "what is it you'd need from me to make that work?",
    "what would i have to give you for it?",
)
STILL_CLARIFYING_ARMS = (_STILL_CLARIFYING_ASK, *_STILL_CLARIFYING_PHRASINGS)

_STILL_CLARIFYING_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the reply to "
    "the teach question asks a question back instead of answering it, Penny leaves the round "
    "parked in elicit."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_unanswered_teach_question_stays_parked(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The elicit → elicit self-edge: a question back, said five ways, against the teach
    question it is answering.

    One of the two edges no chat transition case reaches, so this case is its only isolated
    coverage.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    elicit binds no routine.
    """
    cohort = await classifier_eval(
        case_id=STILL_CLARIFYING_CASE_ID,
        behaviour=_STILL_CLARIFYING_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.ELICIT, ConversationState.ELICIT),
        model=model,
        state=ConversationState.ELICIT,
        ask=_STILL_CLARIFYING_ASK,
        also_asked=_STILL_CLARIFYING_PHRASINGS,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw left the round parked in elicit",
        _drew(ConversationState.ELICIT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── request → elicit: the routine the assistant named was the wrong one ───────
#
# THE FACTS ARE CONSTANT in both halves the edge's condition names — the routine named was
# the wrong one, and the task is still wanted — and the WORDS for both halves move.  That
# separation is the point here rather than a nicety: the condition reads "they say that skill
# is not what they meant, and still want the task done", so five arms all ending in "i still
# want it done" would measure one wording of the discriminating clause against a lexical
# match, and would score green on a draw that had learned the phrase rather than read the
# situation.  So no arm carries that clause: the task is still wanted by "still need this
# doing", "still like it set up", "keep going", or "the job itself is fine", and the probe
# holds the fact as an ALTERNATION rather than pinning any one of them.
#
# An arm carrying only the rejection would be the break-out to idle, and an arm supplying the
# page would be the move to apply.
#
# The registry is seeded so both of the other doors are really on offer — with no candidates
# ``presented_edges`` withholds apply outright, and the rejection would be declining a door
# that was never open.
#
# THE ANCHOR ASK IS ONE THE PRICE WATCHER PLAINLY COVERS, and the arms nonetheless say it is
# a different kind of thing.  That is coherent only because the condition makes the user's
# word decisive: this edge fires on what they SAY about the routine, not on a fresh coverage
# judgement, so the draw is being asked to take the rejection at face value against a
# candidate that still looks like a fit.  A world where the routine visibly did not cover the
# ask would measure the coverage read instead, which the idle → elicit cases already own.
#
# THE ROUND'S BINDING IS DECLARED (#2084), the same one its two sibling request cases
# declare: production reaches request only through the binder, so a draw parked here is
# always shown the `## The details this task is waiting on` section naming the routine, what
# is already given and what is still needed.  That section is what this case's rejection is
# ABOUT — the user is saying the routine named on it was the wrong one — so a snapshot
# without it would put the rejection in front of a draw that was never shown the thing being
# rejected, and any number it produced would describe a conversation production cannot have.

WRONG_ROUTINE_CASE_ID = "classifier-elicits-when-the-named-routine-was-wrong"

_WRONG_ROUTINE_ASK = "no, that's not what i meant — but i do still need this doing"
_WRONG_ROUTINE_PHRASINGS = (
    "that's the wrong routine for this, i'd still like it set up",
    "the job itself is fine, it's the routine that's wrong",
    "not that one — keep going, just not with that skill",
    "that isn't the one i had in mind, and i'd still like the job done",
)
WRONG_ROUTINE_ARMS = (_WRONG_ROUTINE_ASK, *_WRONG_ROUTINE_PHRASINGS)

_WRONG_ROUTINE_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the user says "
    "the routine the assistant named was the wrong one and still wants the task done, Penny "
    "returns the round to elicit."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_rejected_routine_returns_the_round_to_elicit(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The request → elicit draw: the proposal was rejected and the task is still wanted, so
    the routine has to be taught.

    The other edge no chat transition case reaches, so this case is its only isolated
    coverage.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    elicit binds no routine.
    """
    cohort = await classifier_eval(
        case_id=WRONG_ROUTINE_CASE_ID,
        behaviour=_WRONG_ROUTINE_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.REQUEST, ConversationState.ELICIT),
        model=model,
        state=ConversationState.REQUEST,
        ask=_WRONG_ROUTINE_ASK,
        also_asked=_WRONG_ROUTINE_PHRASINGS,
        penny_last_turn=_REQUEST_TURN,
        task_anchor=_KAYAK_ASK,
        parked_round=_PARKED_ON_THE_PRICE_WATCH,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw returned the round to elicit",
        _drew(ConversationState.ELICIT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── The four decisions that complete the map (#2055 tranche C) ────────────────
#
# Tranche A took the six edges nothing had ever drawn in isolation and tranche B the six
# decisions inside covered edges.  These four are the rest of the map, and each names the
# TEMPTATION its world puts in front of the draw — the door that is open and wrong.
#
# THE THREE CLAIM CATEGORIES ARE THE TRANCHE-A SECTION'S, unchanged and not restated: STORE
# is empty because a micro-context is one call that writes to no store, and PROVENANCE is
# empty because both fields the draw returns are closed sets the harness supplied.  What is
# claimed is WHICH member the draw picked (``_drew``).  None of these four draws a
# skill-gated state, so none of them makes a ``_bound`` claim.


# ── idle → elicit: the same verb shape, a different world ─────────────────────
#
# THE FACTS ARE CONSTANT: every wording asks for the SAME ongoing job — new restaurants
# opening downtown, kept as a list — and not one of them says HOW.  Both halves are the
# edge's own condition, so an arm carrying steps would be the idle → learn behaviour wearing
# this case's id, and an arm asking for something done once would be idle's.
#
# THE TEMPTATION IS THE FALSE APPLY, and the registry is what makes it sharp: the seeded
# discovery routine and the ask share their whole VERB SHAPE — find new X and keep them — in
# starkly different domains, so apply and request are really on offer and neither is right.
# A job-listings routine does not cover restaurants, and the code-owner ruling this world
# records is why the domains are stark rather than adjacent: a watch-shaped ask against a
# watch-shaped routine is legitimately COVERED, so a near-miss world measures a
# non-distinction.  Measured against an empty registry this case would prove nothing, since
# ``presented_edges`` withholds both gated doors there; that is the cold-registry case
# (``classifier-elicits-on-a-cold-registry``), and this one is its populated sibling.

UNCOVERED_DOMAIN_CASE_ID = "classifier-elicits-when-a-routine-shares-the-verb-but-not-the-domain"

_UNCOVERED_DOMAIN_ASK = "keep a list of new restaurants opening downtown"
_UNCOVERED_DOMAIN_PHRASINGS = (
    "keep a list going of new restaurants opening downtown",
    "can you keep track of new restaurants opening downtown?",
    "keep collecting the new restaurants that open downtown for me",
    "keep a running list of new downtown restaurant openings",
)
UNCOVERED_DOMAIN_ARMS = (_UNCOVERED_DOMAIN_ASK, *_UNCOVERED_DOMAIN_PHRASINGS)

_UNCOVERED_DOMAIN_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when a routine she "
    "knows matches the ask's verb shape in a different world, Penny opens a teach round in "
    "elicit."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_same_verb_different_domain_ask_still_elicits(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The stark non-coverage contract: one discovery ask said five ways, against a
    discovery routine in another domain.  The routine shares everything with the ask except
    the world it is about, which is the reading a draw can plausibly get wrong.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    elicit binds no routine.
    """
    cohort = await classifier_eval(
        case_id=UNCOVERED_DOMAIN_CASE_ID,
        behaviour=_UNCOVERED_DOMAIN_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.IDLE, ConversationState.ELICIT),
        model=model,
        state=ConversationState.IDLE,
        ask=_UNCOVERED_DOMAIN_ASK,
        also_asked=_UNCOVERED_DOMAIN_PHRASINGS,
        seed_skills=_CROSS_DOMAIN_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw opened a teach round in elicit",
        _drew(ConversationState.ELICIT),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── elicit → learn: the teach question is answered ────────────────────────────
#
# THE FACTS ARE CONSTANT: every wording gives the SAME instructions — the same page, the
# same thing to remember off it — in answer to the same teach question.  An arm asking
# something back would be the parked self-edge, and an arm dropping the round would be the
# break-out to idle.
#
# THE TEMPTATION IS THAT EXISTING ROUTINES DEMOTE THE TEACHING, which is why the registry is
# seeded: the Known skills section renders two watch-adjacent candidates while the user is
# mid-teaching something NEW, and neither covers a ferry timetable.  Teaching must not read
# as still-clarifying, nor as a bail, because something that looks close is on the page.

STEPS_ANSWERED_CASE_ID = "classifier-draws-learn-when-the-teach-question-is-answered"

_STEPS_ANSWERED_ASK = f"sure — read {_TEACH_PAGE} and remember the first sailing"
_STEPS_ANSWERED_PHRASINGS = (
    f"here's what to do: open {_TEACH_PAGE} and save the first sailing",
    f"it's easy — read {_TEACH_PAGE} and keep the first sailing",
    f"go to {_TEACH_PAGE} and write down the first sailing",
    f"you'd read {_TEACH_PAGE} and note the first sailing off it",
)
STEPS_ANSWERED_ARMS = (_STEPS_ANSWERED_ASK, *_STEPS_ANSWERED_PHRASINGS)

_STEPS_ANSWERED_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the reply to "
    "the teach question carries the instructions it asked for, Penny takes the round into "
    "learn."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_answered_teach_question_draws_learn(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The elicit → learn draw under the populated-registry stress: one set of instructions
    said five ways, with watch-adjacent candidates rendered beside them.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    learn binds no routine.
    """
    cohort = await classifier_eval(
        case_id=STEPS_ANSWERED_CASE_ID,
        behaviour=_STEPS_ANSWERED_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.ELICIT, ConversationState.LEARN),
        model=model,
        state=ConversationState.ELICIT,
        ask=_STEPS_ANSWERED_ASK,
        also_asked=_STEPS_ANSWERED_PHRASINGS,
        penny_last_turn=_TEACH_QUESTION,
        task_anchor=_FERRY_ASK,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw took the round into learn",
        _drew(ConversationState.LEARN),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── Parked learn after a failed round: the two directions, one world ──────────
#
# The demo round failed — the honest failure report is the assistant's last turn, and it says
# the page would not load and nothing was saved — and the machine holds in learn.  What the
# reply carries decides whether it stays there: a correction that is actionable NOW, or a
# question that carries no instructions at all.
#
# WHAT EVERY ARM ON BOTH SIDES MAY SAY IS BOUNDED BY THAT REPORT.  A round that saved nothing
# cannot be told it saved the wrong value, so the correction is one the failure leaves open —
# the page has moved, here is the live one — and the questions are about the failure itself.
# A world that could not have happened measures a reading of a contradiction rather than the
# boundary the case names.
#
# THE TWO CASES SHARE ONE WORLD, deliberately.  They are the two directions of one boundary,
# so the only thing allowed to differ between them is the reply; a world that moved as well
# would make the delta between their rates uninterpretable.
#
# THAT WORLD SEEDS ROUTINES.  ``learn``'s out-edges carry a skill-gated state — apply, the
# acceptance door — and ``presented_edges`` withholds it outright on an empty registry, so a
# correction measured cold declines only idle and never faces the door it most resembles: a
# "close! …" reading as acceptance.  Neither seeded routine covers a ferry timetable, so
# apply is a door to decline rather than a defensible answer, and a round parked in learn
# inside a real deployment is parked beside whatever else the registry holds.

CORRECTION_CASE_ID = "classifier-stays-in-learn-on-a-correction"

_LIVE_TIMETABLE_PAGE = "harborferries.example/timetable-v2"

_CORRECTION_ASK = f"that link's dead — use {_LIVE_TIMETABLE_PAGE} instead"
_CORRECTION_PHRASINGS = (
    f"the old link is gone; read {_LIVE_TIMETABLE_PAGE}",
    f"use {_LIVE_TIMETABLE_PAGE}, the page you tried has moved",
    f"the page moved — {_LIVE_TIMETABLE_PAGE} is where it lives now",
    f"{_LIVE_TIMETABLE_PAGE} is the live one, go there instead",
)
CORRECTION_ARMS = (_CORRECTION_ASK, *_CORRECTION_PHRASINGS)

_CORRECTION_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the reply to a "
    "failed demonstration corrects it and the correction is actionable now, Penny holds the "
    "round in learn."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_correction_holds_the_round_in_learn(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The learn → learn self-edge: one correction said five ways, against the failure report
    it answers — the correction-loop invariant, that a failed round holds its state.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    learn binds no routine.
    """
    cohort = await classifier_eval(
        case_id=CORRECTION_CASE_ID,
        behaviour=_CORRECTION_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.LEARN, ConversationState.LEARN),
        model=model,
        state=ConversationState.LEARN,
        ask=_CORRECTION_ASK,
        also_asked=_CORRECTION_PHRASINGS,
        penny_last_turn=_FAILED_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw held the round in learn",
        _drew(ConversationState.LEARN),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))


# ── learn → idle: the same failure, answered with a question ──────────────────
#
# THE FACTS ARE CONSTANT: every wording asks the SAME two things about the same failure —
# what went wrong, and which page was opened — and not one carries an instruction.  An arm
# saying to try again, or naming a different page, would be the correction above.
#
# THE TEMPTATION IS ELICIT, and it is unreachable: the reply is engaged and on-task, so it
# reads like a round still being negotiated — but ``learn`` has no edge back to elicit
# (code-owner ruling: elicit exists to GET instructions, and they have been given), so the
# only home for a reply carrying none is idle.  What the draw can wrongly do is hold in
# learn, reading engagement as instruction.

POST_FAILURE_QUESTION_CASE_ID = (
    "classifier-falls-to-idle-when-a-post-failure-reply-carries-no-instructions"
)

_POST_FAILURE_QUESTION_ASK = "what went wrong exactly? which page did you open?"
_POST_FAILURE_QUESTION_PHRASINGS = (
    "what went wrong there — which page did you try?",
    "which page did you open, and what went wrong?",
    "so what went wrong? which page was it you opened?",
    "tell me what went wrong — which page did you actually open?",
)
POST_FAILURE_QUESTION_ARMS = (
    _POST_FAILURE_QUESTION_ASK,
    *_POST_FAILURE_QUESTION_PHRASINGS,
)

_POST_FAILURE_QUESTION_BEHAVIOUR = (
    f"In the {PennyConstants.STATE_CLASSIFIER_AGENT_NAME} micro-context, when the reply to a "
    "failed demonstration is an engaged question carrying no instructions, Penny falls to "
    "idle."
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_post_failure_question_falls_to_idle(
    classifier_eval: ClassifierEval, model: str
) -> None:
    """The learn → idle draw: one post-failure question said five ways, over the same world
    the correction case runs on.

    STORE and PROVENANCE are empty; the section comment above says why.  ONE LANDED claim:
    idle binds no routine.
    """
    cohort = await classifier_eval(
        case_id=POST_FAILURE_QUESTION_CASE_ID,
        behaviour=_POST_FAILURE_QUESTION_BEHAVIOUR,
        area=Area.CONVERSATION_MACHINE,
        edge=(ConversationState.LEARN, ConversationState.IDLE),
        model=model,
        state=ConversationState.LEARN,
        ask=_POST_FAILURE_QUESTION_ASK,
        also_asked=_POST_FAILURE_QUESTION_PHRASINGS,
        penny_last_turn=_FAILED_ROUND_REPORT,
        task_anchor=_FERRY_ASK,
        seed_skills=SEEDED_SKILLS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — the closed field of the typed result, asserted by equality
    cohort.claim(
        "state: the draw broke the round out to idle",
        _drew(ConversationState.IDLE),
        SpecCategory.LANDED,
    )

    # STORE — empty by construction; see the section comment.
    # PROVENANCE — empty because it is closed upstream; see the section comment.

    cohort.measure(output_field(CLASSIFY_STATE))
