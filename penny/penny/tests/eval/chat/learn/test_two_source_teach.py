"""Story 15, the two-source teach — the fused ask, and the learn round's close.

The estate's only two-page routine: one message asks for something that keeps running
over TWO sources, and the turn has to decompose it, demonstrate it once, and set it
running.  Two cases, sharing one setup ask:

  * the fused ask becomes a running routine — decomposed into its two sources,
    demonstrated, and stood up
  * the learn close states the steps it CAPTURED — the reply that ends the round says
    what the routine it just learned will RUN, for the one person who can tell a
    captured step from a stray one (#1943)

They are here rather than with the memory stories because of where the machine goes:
their ``state_transition`` rows walk idle → elicit → learn (and on to apply for the
first), while every genuine memory story is idle → idle.  The turn being scored is a
LEARN turn, composed from ``Prompt.LEARN_INSTRUCTION``, so learn is the microcontext
under test.

The worlds are the CURRENT runtime's — nothing is pre-seeded (migration 0108), so the
round is parked in elicit the way the user's own earlier turn would have parked it.

Check labels carry one of three prefixes — ``state:`` (end DB facts), ``reply:``
(what Penny said, against what she did), ``calls:`` (call provenance).  State and
reply checks are SCORED; the call spine and the loop-health verdict are ADVISORY
(``scored=False``) — except the learn close's LANDING, which is scored, since a story
about the reply that ends a learn round has no subject at all outside that state
(#1989).  Where a scored claim only exists in a state a sample never reached, it reads
``Check.na(...)`` rather than ❌: a precondition nobody met is not a contract anybody
failed.

Both cases are REPORT-ONLY (``min_pass_rate=None``): the thresholds are the code
owner's to set once the numbers are read.  All content is synthetic — invented teams,
invented markets — because the repo is public.
"""

from __future__ import annotations

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.models import MemoryRow
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Check,
    asked_for_page_structure,
    new_collections,
    outgoing_replies,
)

# The enacting-tool set is read from the suite's shared fixtures, not restated here: the
# state machine's elicitation edge asks the same question of a turn (nothing acted on
# before it was taught), and one policy in two copies is two contracts.
from penny.tests.eval.utils.cohort import (
    CONTAINER_NAME,
    ENTRIES_STORED,
    REPLY_SPREAD,
    ROUTINE_SHAPE,
    TOOL_SEQUENCE,
    TRANSITIONS,
)
from penny.tests.eval.utils.memory_world import (
    _FAMILY,
    _FOXES_TOKENS,
    _SEALS_TOKENS,
    LEARN_CLOSE_ASK,
    _carries,
    _landing_advisory,
    _pages_fetched,
    _routing_advisory,
)
from penny.tests.eval.utils.seeds import Seeder, round_parked_in_elicit

# Standing a ROUND up before the measured turn is the transition suite's idiom, read from
# where that suite declares it rather than restated here: a seeded machine state, a seeded
# conversation turn and a seeded ledger row are one shape, and a second copy of it would be
# a second contract free to drift from the one every edge case is measured against.
#
# Its PROBES are deliberately restated instead (``_assert_parked_in_elicit`` below): that
# suite's are keyed to its own ``_ElicitRound`` case type, which this file has no shape
# for.  What a probe asserts is the seed it stands beside, so the honest cost of not
# widening a neighbour's fixture type is one restated probe, named here rather than left
# for a reader to notice.
from penny.tests.eval.utils.worlds import (
    FOXES_NEWS,
    FOXES_URL,
    SEALS_NEWS,
    SEALS_URL,
    TWO_TEAM_NEWS,
    TWO_TEAM_NEWS_CONTROL,
)

pytestmark = pytest.mark.eval


def _fetched(db: Database, token: str) -> bool:
    """Whether a page carrying ``token`` was actually read this sample."""
    return any(token in entry.content for entry in _pages_fetched(db))


# ═══ Story 15 — the two-source routine ═══════════════════════════════════════
#
# The FUSED ask the first external deployment produced: one message carrying sources,
# cadence and filter, and never an imperative.  Observed live, she elicited page
# mechanics and planned forever; no round ever ran.  Four prompt iterations demanding a
# self-started round moved nothing — the conversational prior at a descriptive message
# is propose-and-confirm.  The design HARNESSES that prior: teaching and setting a job
# running are two different things, so a fused ask is SPLIT OUT LOUD — "teach me the
# find first, in one message; then I'll run it on your schedule."  The user's routine
# reply is the imperative the round fires on, and the closing schedule intent stands
# the job up.
#
# It is the only story in this file whose routine reads TWO pages, which is what it is
# kept for: everything else in the estate demonstrates against one source, so nothing
# else measures a round that has to visit both and keep both.

# The FUSED ask both story-15 cases open on — sources + filter + schedule, no imperative,
# the field shape verbatim.  Named rather than left as a position in the turns list below,
# because the learn close reads it as the ask its seeded round is anchored to: a turn
# inserted at the head of that list would silently re-seed a case two hundred lines away.
_TWO_SOURCE_SETUP_ASK = (
    "hey can you set up news alerts for my favourite teams? the ridgeline "
    f"foxes and the harbor seals — their news pages are {FOXES_URL} and "
    f"{SEALS_URL}. check them twice a day, and alert me about "
    "notable stuff like trades, signings, and injuries — not game scores."
)

# Turn 1 = that fused ask.  Turn 2 = the user's routine, the answer the decompose ask
# requests (the URLs referenced, not retyped — a real user doesn't repeat themselves).
# Turn 3 = pure schedule intent.
_TWO_SOURCE_TURNS = [
    _TWO_SOURCE_SETUP_ASK,
    (
        "sure: 1. go to those two news pages 2. pull out any trades, signings, "
        "or injuries — skip game scores 3. remember the title plus a short "
        "blurb for each"
    ),
    "perfect — now do that twice a day and let me know when something new shows up.",
]

# How she asks to be walked through the round.  Broad by design — measured replies said
# "teach me a single round" and "a quick walkthrough of one round", never the scripted
# phrase — because what is scored is one-complete-pass elicitation, not wording.
_TEACH_ASK_TOKENS = (
    "teach me",
    "walk me through",
    "walkthrough",
    "one message",
    "single message",
    "one round",
    "single round",
    "one pass",
    "one complete pass",
)
_SOURCE_TOKENS = ("ridgelinefoxes", "harborseals", "foxes", "seals")


def _teach_ask(text: str) -> bool:
    return any(token in text.lower() for token in _TEACH_ASK_TOKENS)


def _configured_jobs(db: Database, before: set[str]) -> list[MemoryRow]:
    """Collections created this sample that are CONFIGURED to run: a routine attached
    and its program rendered in (a container the round framed carries neither until the
    job is stood up)."""
    return [
        row
        for row in new_collections(db, before)
        if row.skill_name is not None and row.extraction_prompt is not None
    ]


def _decompose_checks(replies: list[str], *, ran_to_completion: bool) -> list[Check]:
    """Turn 1's verdicts, read off the FIRST reply: she recognized she cannot act yet
    and asked to be taught the round, with the example modelled from THEIR sources so
    "yes, do that" is a complete answer.

    Running the whole chain without asking is spontaneous one-shot success — the end
    goal, not a failure — so the ask is the FALLBACK, and the modelled facet is
    not-applicable when no ask happened."""
    first = replies[0].lower() if replies else ""
    asked = _teach_ask(first)
    modelled = any(token in first for token in _SOURCE_TOKENS)
    modelled_check = (
        Check("reply: the ask was modelled from their own sources", modelled, kind="reply")
        if asked
        else Check.na("reply: the ask was modelled from their own sources", kind="reply")
    )
    return [
        Check(
            "reply: she asked to be taught the round, or ran it herself to completion",
            asked or ran_to_completion,
            kind="reply",
        ),
        modelled_check,
    ]


def _score_two_source_teach(db: Database, before: set[str], reply: str) -> list[Check]:
    replies = outgoing_replies(db)
    jobs = _configured_jobs(db, before)
    job = jobs[0] if jobs else None
    structure = next((asked_for_page_structure(sent) for sent in replies if sent), None)
    return [
        *_decompose_checks(replies, ran_to_completion=bool(jobs) and _carries(db, _FOXES_TOKENS)),
        Check("state: the foxes page was read", _fetched(db, "ridgelinefoxes"), kind="state"),
        Check("state: the seals page was read", _fetched(db, "harborseals"), kind="state"),
        Check(
            "state: something from the foxes page was written down",
            _carries(db, _FOXES_TOKENS),
            kind="state",
        ),
        Check(
            "state: something from the seals page was written down",
            _carries(db, _SEALS_TOKENS),
            kind="state",
        ),
        Check(
            "state: the round taught a routine",
            bool(db.skills.list_all()),
            kind="state",
        ),
        Check(
            "state: the routine was set running on its own container",
            len(jobs) == 1,
            kind="state",
            rationale=f"configured {[row.name for row in jobs]}",
        ),
        Check(
            "state: it runs on a schedule and tells the user",
            job is not None and job.schedule is not None and bool(job.notify),
            kind="state",
            rationale=(
                f"schedule {job.schedule!r}, notify {job.notify}" if job is not None else "no job"
            ),
        ),
        Check(
            "reply: no re-teach ask once the routine exists",
            not (_teach_ask(replies[-1]) if replies else False),
            kind="reply",
        ),
        Check(
            "reply: she never asked how the pages are built",
            structure is None,
            rationale=f"asked for {structure!r}" if structure else None,
            kind="reply",
        ),
        _landing_advisory(db, ConversationState.APPLY),
        _routing_advisory(db),
    ]


async def test_a_fused_two_source_ask_becomes_a_running_routine(chat_eval: ChatEval) -> None:
    """Story 15: the fused ask is split out loud, the routine the user then gives is
    run once across BOTH pages, and the closing "do that twice a day" stands the job
    up on the container the round built."""
    await chat_eval(
        case_id="memory-two-source-teach",
        messages=_TWO_SOURCE_TURNS,
        browse=[FOXES_NEWS, SEALS_NEWS],
        score=_score_two_source_teach,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=300.0,  # three turns: decompose, then the round, then standing it up
    )


# ── Story 15, the learn close (#1994/#1995) ──────────────────────────────────
#
# THE REFERENCE IMPLEMENTATION — every other case is ported against this shape.  Why a case
# asserts end state and measures model output at all is `cohort.py`; what a claim means is
# `assertions.py`.  What is here is what is true of THIS case:
#
# PHRASINGS and the CONTROL are different mechanisms, and reading one as the other is the
# mistake this case exists to prevent:
#   * phrasings — same world, different words → VARIANCE, pooled into one score
#   * control   — same words, different world → an ASSERTION, never pooled
# Wording variation cannot do the control's job: if Penny were pattern-completing from the
# shape of the request, every phrasing would name the same player and every one would be right.
#
# NOT HERE: the single-source variant of this ask.  One page instead of two is a different
# SCENARIO — what the store must hold is a different claim — so it is a different case.

_LEARN_CLOSE_CASE_ID = "memory-learn-close-shape"

# Four more wordings of the SAME demonstration, and what varies is only how a person writes
# three steps: numbered or dashed or spelled out, "skip" or "not" or "ignore", "remember" or
# "keep" or "save", the filter before the destination or after it.  What does NOT vary is that
# they wrote steps at all — the wordings this replaced were paraphrases of a conversational
# request, so they varied the one thing that has to be held fixed and the case measured the
# model's guess at the step boundaries rather than its enactment of them.
#
# None of them retypes the URLs.  The pages are "those two"/"both" because the referent is in
# the seeded turn above, and a user who has just been asked to walk through one pass does not
# paste the addresses back.
LEARN_CLOSE_PHRASINGS = (
    (
        "ok here's one pass: 1. open both of those news pages 2. grab the "
        "trades, signings and injuries — not the game scores 3. keep the "
        "headline and a one-line blurb for each"
    ),
    (
        "sure — 1) read those two pages 2) find any trade, signing or injury "
        "news, ignoring the game scores 3) save the title plus a short summary "
        "for each one"
    ),
    (
        "yep: - visit the two news pages - pick out trades, signings and "
        "injuries, skipping scores - store each headline with a brief note"
    ),
    (
        "of course. step 1: check both news pages. step 2: pull anything about "
        "trades, signings or injuries — game scores don't count. step 3: keep "
        "the title and a short blurb for each."
    ),
)

_LEARN_CLOSE_TEACH_QUESTION = (
    "happy to set that up — but i don't have a routine for it yet. can you walk me "
    "through one pass in a single message? which pages should i read, what counts as "
    "notable, and what should i keep for each one?"
)


@pytest.fixture
def standing_elicit_round() -> Seeder:
    """The round the measured turn closes: the user asked for the job, Penny asked to be taught
    one pass, and the machine is parked in ``elicit`` on that ask.

    Seeded rather than hoped for (#1989) — the ask is an imperative about now, which idle's own
    definition claims, so on a cold machine both measured models drew idle on 10 of 10 samples
    and every reply check failed for a reply nobody had been asked to write."""
    return round_parked_in_elicit(_TWO_SOURCE_SETUP_ASK, _LEARN_CLOSE_TEACH_QUESTION)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_learn_close_states_the_steps_it_captured(
    chat_eval: ChatEval, model: str, standing_elicit_round: Seeder
) -> None:
    """Story 15, the learn close: one demonstrated round, and the reply that closes it tells the
    user what the routine will RUN — so a step it captured by accident is visible to the only
    person who can tell that it does not belong.

    REPORT-ONLY: the floors and ceilings this run proposes are the code owner's to accept once
    the numbers have been read."""
    cohort = await chat_eval(
        case_id=_LEARN_CLOSE_CASE_ID,
        model=model,
        seed=standing_elicit_round,
        world=TWO_TEAM_NEWS,
        ask=LEARN_CLOSE_ASK,
        also_phrased=LEARN_CLOSE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    cohort.assert_machine_landed(ConversationState.LEARN)
    cohort.assert_a_routine_reached_the_registry()
    cohort.assert_the_routine_names_a_destination()
    cohort.assert_the_store_holds_an_entry()
    cohort.assert_each_source_was_kept()
    cohort.assert_nothing_excluded_was_stored()
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    # A SECOND VISIBLE DRIVE, beside the claim it serves: an assertion that quietly made three
    # more model calls would be a nasty surprise.
    control = await chat_eval(
        case_id=_LEARN_CLOSE_CASE_ID,
        model=model,
        seed=standing_elicit_round,
        world=TWO_TEAM_NEWS_CONTROL,
        ask=LEARN_CLOSE_ASK,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    cohort.assert_facts_moved_with_the_world(control)

    cohort.measure(
        TOOL_SEQUENCE, ROUTINE_SHAPE, CONTAINER_NAME, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD
    )
