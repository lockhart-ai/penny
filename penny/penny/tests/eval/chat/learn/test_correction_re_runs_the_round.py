"""Chat in LEARN, entered from learn: the correction re-runs the round.

The round was taught and run; the user now corrects it -- a different target, a different
filter, a value the routine is identified by. The turn re-runs the round on the corrected
instructions and updates the routine, rather than keeping the first version, answering about the
change without re-running, or shaking loose a term that was deferred rather than corrected.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import partial
from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.conversation_machine import (
    ConversationState,
    MachineSnapshot,
    RoundFraming,
    RoundProvenance,
)
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.models import Skill, StateTransition
from penny.database.skills import (
    SkillDraft,
    SkillStep,
    slug_skill_name,
)
from penny.penny import Penny

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
    ChatEval,
    Check,
    Preparer,
    Seeder,
    collection_entries,
    live_prompts,
    new_collections,
    outgoing_replies,
    routing_clean,
    tool_not_called,
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
from penny.tests.eval.utils.transition_ledger import (
    _BROWSE_CALL_ID,
    _BROWSE_TOOL,
    _FAMILY,
    _SET_TOOL,
    _WRITE_CALL_ID,
    _drawn_state,
    _entries_written_by_this_run,
    _journey_runs,
    _JourneyRuns,
    _landed_state,
    _log_ask,
    _log_chat_step,
    _log_classifier_draw,
    _log_reply,
    _pages_fetched,
    _park,
    _seeded_response,
    _written_texts,
)
from penny.tests.eval.utils.transition_world import (
    _CHOIR_REHEARSALS_URL,
    _CLIFF_WALK_URL,
    _COMPOSED_MESSAGE_WINDOW,
    _HARBOUR_SIGNALS_URL,
    _IDLE_BANTER,
    _JOURNEYS,
    _LIVE_JOB_CONTAINERS,
    _PLOT_RULES_URL,
    _TEACH_CLIFF_WALK,
    _TEACH_FREE_EVENT,
    _TEACH_HARBOUR_FLAG,
    _TEACH_REHEARSAL_PIECE,
    _TEACH_WATERING_RULE,
    _TOWN_HALL_EVENTS_URL,
    _assert_every_job_is_live,
    _assert_every_reply_is_threaded,
    _attaches_nothing_checks,
    _candidate,
    _demonstrated_ledger,
    _DemonstratedRound,
    _destination_subs,
    _extraction_shape_checks,
    _first_divergence,
    _fixture_skill,
    _FixtureDraws,
    _framed,
    _framed_checks,
    _landed_in,
    _learned_this_turn,
    _log_browse_extract,
    _mentions,
    _mentions_any,
    _round_framing,
    _round_ran_checks,
    _round_reported_checks,
    _row_tool_calls,
    _said_back,
    _seed_call_step,
    _seeded_ask_id,
    _seeded_jobs_untouched_check,
    _skill_steps,
    _spoken_and_stored,
    _TeachCase,
    _wrote_into_the_container_check,
    expected_conversation,
    seed_composed_world,
)

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
    FramedParameter,
    LeafLabel,
    SkillSignature,
)

pytestmark = pytest.mark.eval


# ── learn → learn: the correction re-runs the round on the corrected instructions ─
#
# Beat 9 (#1900) — the mid-round correction, designed in #1706/#1827 ("a correction re-runs
# the round on the corrected instructions and re-extraction replaces the skill"), worded in
# the learn menu since #1849, and never enacted.  Each world is one of beat 8's five teaches
# PLUS the round that teach ran, seeded exactly as production leaves it: the teaching
# message and Penny's report threaded, that turn's promptlog under seeded-prefix run ids,
# the machine parked in learn on the teaching message with the round's framing recorded on
# the move, the container that framing built holding the demonstrated write, and the routine
# run-end extraction filed it under.  The measured turn is the user changing their mind
# about WHAT to read off the page they already named.
#
# The open question this beat exists to answer is the code owner's, verbatim: "whether a
# small correction on past explicit instructions is sufficient or whether the user needs to
# restate the full corrected set of instructions, we'll try this and see if the model can
# cohere the corrected set of instructions, if not we'll need to modify its response to say
# 'is that correct or can you give me a full corrected set of instructions?'".  So every
# correction here is a DELTA — one sentence naming the new target and nothing else — and the
# full-restatement reply is the recorded fallback design, deliberately NOT built here.
#
# Which is why the scoring DECOMPOSES rather than passing or failing the turn.  Two things
# can go wrong independently — applying the correction to the instructions already given,
# and then carrying out the whole corrected flow — and one verdict cannot tell them apart.
# The five named shapes fall into two families, scored separately.
#
# What the TURN did, read off this run's fetches and writes and named in one phrase per
# combination (``_correction_shape``):
#
#   * re-ran the flow with the correction applied — THE PASS: the same taught page fetched
#     again, the corrected target's token stored, the superseded one not stored;
#   * applied the delta WITHOUT re-running — the corrected value stored or claimed with no
#     fetch this turn, which the seeded browse-results log makes genuinely reachable;
#   * re-ran WITHOUT applying — the page fetched again and the old target stored again.
#
# What the RE-EXTRACTION left, read off the registry structurally rather than off the reply:
#
#   * lost the unchanged parts — the routine dropped the page it was taught on, or the step
#     that keeps what it finds, neither of which the correction touched;
#   * forked instead of replaced — a second routine for the task, or a sibling container
#     beside the one the round already had.
#
# What a correction leaves behind is ONE routine whose program carries the corrected step
# and every step the correction said nothing about.
#
# The FORK was, until #1902, a certainty rather than a risk: entering learn re-drew the
# framing on every decided move in, so a corrected ask read as a different subject derived a
# fresh name, find-or-create minted a sibling container under it, and run-end extraction
# registered a second routine beside the first — ~21 of 25 measured samples.  That is now
# closed on both sides: a learn → learn move CARRIES the round's framing and re-settles only
# the container (``RoundFramer.carry_entry``), and run-end extraction is KEYED by the
# framing's pinned name, so the round's second run replaces its own routine in place.  The
# two checks below are what that deletion is watched by — the gate cases that would catch its
# absence — so they are kept and scored rather than retired as satisfied.
#
# What must NOT happen is still the fold (#1706): nothing is configured, on any of the five.
# Case 4 is where that is a live temptation rather than an absence — its teach stated a
# notify condition that is still waiting for the turn that accepts the offer — and the
# contract is that the correction does not shake it loose.
#
# DELIBERATELY EXCLUDED, recorded rather than built: the URL-change correction ("wrong page
# — it's /signals-today").  A fresh framing there may mint a fresh name, and it is
# ``is_same_job`` dedupe rather than the name that carries job identity at apply time — its
# own beat.  Every correction below is a same-page single-value redirect.
#
# The reference replies are review targets, never scorer strings — carried as DATA so the
# deterministic pin can run this beat's two reply checks through them without a GPU.


# ── The five taught rounds, transcribed from the teach beat's measured draws ──
#
# Each round is what beat 8's final composed run actually produced for that case, read out
# of its per-sample databases as data rather than off a transcript, and taken WHOLE per case
# — one clean modal draw, never composed across samples — so a correction is answered
# against a round the pipeline really hands forward.  Two of the five are worth naming: the
# trail round drew TWO parameters (the page and which loop to read on it, which every sample
# of that case drew), and it is the one whose corrected target moves a value the container's
# name is derived FROM; the rest drew the page alone.
#
# Each journey mints its own run-id bundle, for the reason the composed world's five do: a
# run id is the join key everything a turn produced cites, and two rounds sharing one set
# would read back as a single impossible run.

_SIGNALS_RUNS = _journey_runs("signals")
_TRAIL_RUNS = _journey_runs("trail")
_EVENTS_RUNS = _journey_runs("events")
_WATERING_RUNS = _journey_runs("watering")
_REHEARSAL_RUNS = _journey_runs("rehearsal")

_SIGNALS_FRAMING = _framed(
    SkillSignature(
        name="retrieve_harbour_flag",
        description="Retrieves the current harbour flag from a specified URL",
        parameters=(
            FramedParameter(
                name="url",
                description="the URL to fetch harbor flag information from",
                value=_HARBOUR_SIGNALS_URL,
            ),
        ),
    )
)
_SIGNALS_DEMONSTRATED = _DemonstratedRound(
    url=_HARBOUR_SIGNALS_URL,
    extract="the flag currently flying",
    collection=_SIGNALS_FRAMING.container,
    entry_key="current flag",
    entry_value="Bravo",
)
_SIGNALS_SKILL = _fixture_skill(
    _SIGNALS_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="url", description="the URL to fetch"),
        extract=LeafLabel(
            name="extract_phrase", description="the text to extract from the webpage"
        ),
        collection=LeafLabel(
            name="memory_id", description="unique ID to retrieve or identify the stored data"
        ),
        entry_key=LeafLabel(name="entry_key", description="the key to identify the stored entry"),
        signature=_SIGNALS_FRAMING.signature,
    ),
    _TEACH_HARBOUR_FLAG.teach,
    _SIGNALS_RUNS,
)

_TRAIL_FRAMING = _framed(
    SkillSignature(
        name="fetch_trail_status",
        description="Retrieve the status line for a specified trail from a webpage",
        parameters=(
            FramedParameter(
                name="url",
                description="the webpage URL to fetch trail status from",
                value=_CLIFF_WALK_URL,
            ),
            FramedParameter(
                name="trail",
                description="the name or keyword of the trail to search for",
                value="north loop",
            ),
        ),
    )
)
_TRAIL_DEMONSTRATED = _DemonstratedRound(
    url=_CLIFF_WALK_URL,
    extract="line about the north loop",
    collection=_TRAIL_FRAMING.container,
    entry_key="north-loop",
    entry_value="North loop: diverted inland at the quarry fence while the path is shored up",
)
_TRAIL_SKILL = _fixture_skill(
    _TRAIL_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="url", description="the URL of the page to browse each run"),
        extract=LeafLabel(
            name="search_phrase", description="the text string to locate in the page"
        ),
        collection=LeafLabel(
            name="memory_key", description="unique identifier for the collection entry each run"
        ),
        entry_key=LeafLabel(
            name="entry_key", description="key under which to store the extracted content"
        ),
        signature=_TRAIL_FRAMING.signature,
    ),
    _TEACH_CLIFF_WALK.teach,
    _TRAIL_RUNS,
)

_EVENTS_FRAMING = _framed(
    SkillSignature(
        name="extract_free_event",
        description="Retrieve the free event from the provided events page",
        parameters=(
            FramedParameter(
                name="url",
                description="the address of the events page",
                value=_TOWN_HALL_EVENTS_URL,
            ),
        ),
    )
)
_EVENTS_DEMONSTRATED = _DemonstratedRound(
    url=_TOWN_HALL_EVENTS_URL,
    extract="the event marked free",
    collection=_EVENTS_FRAMING.container,
    entry_key="free",
    entry_value="Lantern parade from the quay to the square",
)
_EVENTS_SKILL = _fixture_skill(
    _EVENTS_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="urls", description="the list of URLs to browse for the task"),
        extract=LeafLabel(
            name="free_event_description",
            description="the description of the free event extracted from the page",
        ),
        collection=LeafLabel(
            name="memory_name",
            description="the unique name for the collection memory to which the entry will be "
            "written",
        ),
        entry_key=LeafLabel(
            name="entry_key",
            description="the key identifier to use for the stored entry in the collection",
        ),
        signature=_EVENTS_FRAMING.signature,
    ),
    _TEACH_FREE_EVENT.teach,
    _EVENTS_RUNS,
)

_WATERING_FRAMING = _framed(
    SkillSignature(
        name="fetch_watering_restriction",
        description="retrieve the current watering restriction clause from a specified web page",
        parameters=(
            FramedParameter(
                name="url",
                description="the web address of the page to fetch",
                value=_PLOT_RULES_URL,
            ),
        ),
    )
)
_WATERING_DEMONSTRATED = _DemonstratedRound(
    url=_PLOT_RULES_URL,
    extract="watering restriction",
    collection=_WATERING_FRAMING.container,
    entry_key="current restriction",
    entry_value="hosepipes before eight in the morning and after seven in the evening",
)
_WATERING_SKILL = _fixture_skill(
    _WATERING_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(name="url", description="the URL of the page to retrieve and scrape"),
        extract=LeafLabel(
            name="keyword", description="the phrase or term to search for on the page"
        ),
        collection=LeafLabel(
            name="memory_id",
            description="identifier to reference this new memory entry in subsequent tasks",
        ),
        entry_key=LeafLabel(
            name="entry_key", description="unique key to store the extracted restriction"
        ),
        signature=_WATERING_FRAMING.signature,
    ),
    _TEACH_WATERING_RULE.teach,
    _WATERING_RUNS,
)

_REHEARSAL_FRAMING = _framed(
    SkillSignature(
        name="fetch_rehearsal_piece",
        description=(
            "Retrieves the current week's rehearsal piece from a specified URL and stores it."
        ),
        parameters=(
            FramedParameter(
                name="url",
                description="the web address to retrieve rehearsal information from",
                value=_CHOIR_REHEARSALS_URL,
            ),
        ),
    )
)
_REHEARSAL_DEMONSTRATED = _DemonstratedRound(
    url=_CHOIR_REHEARSALS_URL,
    extract="this week's rehearsal piece",
    collection=_REHEARSAL_FRAMING.container,
    entry_key="current",
    entry_value="Kittiwake",
)
_REHEARSAL_SKILL = _fixture_skill(
    _REHEARSAL_DEMONSTRATED,
    _FixtureDraws(
        page=LeafLabel(
            name="rehearsal_url", description="the URL where the rehearsal list page can be fetched"
        ),
        extract=LeafLabel(
            name="piece_title",
            description=(
                "the title or identifier of this week's rehearsal piece as extracted from the page"
            ),
        ),
        collection=LeafLabel(
            name="memory_id",
            description=(
                "a unique identifier used to locate and reference the collection entry for this "
                "fetch operation"
            ),
        ),
        entry_key=LeafLabel(
            name="current_key",
            description="the key under which the fetched piece should be stored in the collection",
        ),
        signature=_REHEARSAL_FRAMING.signature,
    ),
    _TEACH_REHEARSAL_PIECE.teach,
    _REHEARSAL_RUNS,
)


# ── The five corrections ──────────────────────────────────────────────────────


class _CorrectionCase(NamedTuple):
    """One agreed learn → learn correction, and the completed round its turn answers.

    ``prior`` is the beat-8 teach this world is built on — its message is the round's
    ANCHOR, its page is what the corrected round re-reads, its ``reference`` is the report
    Penny closed on (seeded as her last turn, which is why that reply is data), and its
    ``stored`` is the value the round put away and this turn SUPERSEDES.  ``demonstrated``,
    ``skill``, ``framing`` and ``runs`` are that round as the teach beat's measured draws
    left it: the ledger it ran, the routine it taught, the routine's identity and container,
    and the run ids everything it produced cites.

    ``correction`` is the turn under test — one sentence naming the new target, which is the
    delta shape the whole beat exists to measure.  ``corrected`` is the distinctive token
    the page carries for that target: what the corrected round must store, and what makes a
    corrected write provable against a page rather than against the correction's own words,
    since no correction names its own answer.  ``target`` is how the correction NAMES that
    target — the words a re-extracted routine's own program should carry if the correction
    reached the instructions rather than only the reply.

    ``reference`` is how the corrected round would be reported and re-offered WELL: DATA
    rather than prose for the reason every other beat's reference reply is, so the
    deterministic pin can run this beat's reply checks through it without a GPU."""

    case_id: str
    prior: _TeachCase
    demonstrated: _DemonstratedRound
    skill: SkillDraft
    framing: RoundFraming
    runs: _JourneyRuns
    correction: str
    corrected: str
    target: tuple[str, ...]
    reference: str


# Case 1 — the wrong THING on the page: the daily flag was read, the storm signal beside it
# is what was wanted.  The plainest delta of the five — same page, same kind of value, one
# line over — and the one whose correction opens with a self-correction ("oh wait").
_CORRECT_TO_STORM_SIGNAL = _CorrectionCase(
    case_id="transition-learn-to-learn",
    prior=_TEACH_HARBOUR_FLAG,
    demonstrated=_SIGNALS_DEMONSTRATED,
    skill=_SIGNALS_SKILL,
    framing=_SIGNALS_FRAMING,
    runs=_SIGNALS_RUNS,
    correction=(
        "oh wait — not the daily flag, i meant the storm signal next to it, remember that instead"
    ),
    corrected="cone",
    target=("storm",),
    reference=(
        "read the signals page again — the storm signal beside the flag is the north cone, "
        "up for a gale off the headland, and i've saved that instead. want me to keep an "
        "eye on it?"
    ),
)

# Case 2 — the wrong LINE: the routine was taught on the north loop and the south loop is
# what was meant.  The stress case for identity, because which loop to read is a VALUE the
# round's framing carries and the container's name is derived from — so this is the one
# where keeping the job in one place is a real question rather than an arithmetic certainty.
_CORRECT_TO_SOUTH_LOOP = _CorrectionCase(
    case_id="transition-learn-to-learn-parameter",
    prior=_TEACH_CLIFF_WALK,
    demonstrated=_TRAIL_DEMONSTRATED,
    skill=_TRAIL_SKILL,
    framing=_TRAIL_FRAMING,
    runs=_TRAIL_RUNS,
    correction="sorry — i meant the south loop line, use that one",
    corrected="gravel",
    target=("south",),
    reference=(
        "read the trail status again — the south loop is open end to end, with fresh gravel "
        "over the mud at the field gate, and that's what i've saved now. want me to keep "
        "checking it?"
    ),
)

# Case 3 — the wrong FILTER: the round kept the free event, and the criterion is now
# family-friendly instead.  The correction retires the old criterion explicitly ("free
# doesn't matter"), and the page pulls the two apart — the family-friendly one is priced —
# so a round that kept the old filter lands on a different event.
_CORRECT_TO_FAMILY_FRIENDLY = _CorrectionCase(
    case_id="transition-learn-to-learn-filter",
    prior=_TEACH_FREE_EVENT,
    demonstrated=_EVENTS_DEMONSTRATED,
    skill=_EVENTS_SKILL,
    framing=_EVENTS_FRAMING,
    runs=_EVENTS_RUNS,
    correction="actually keep the one marked family-friendly instead — free doesn't matter",
    corrected="puppet",
    target=("family",),
    reference=(
        "read the events list again — the family-friendly one is the puppet matinee in the "
        "long room, £3 a seat, and i've saved that instead. want me to keep it up each "
        "month?"
    ),
)

# Case 4 — the wrong SECTION, with a term still deferred: the teach stated a notify
# condition that is waiting for the turn that accepts the offer, and the correction moves
# the round to a different rule of the same site.  What must not happen is the condition
# being shaken loose into a job — configuring here is the fold this beat scores against
# exactly as the teach beat did.
_CORRECT_TO_COMPOST_RULES = _CorrectionCase(
    case_id="transition-learn-to-learn-deferred-terms",
    prior=_TEACH_WATERING_RULE,
    demonstrated=_WATERING_DEMONSTRATED,
    skill=_WATERING_SKILL,
    framing=_WATERING_FRAMING,
    runs=_WATERING_RUNS,
    correction="no not that — the compost rules are what i want, save what those say",
    corrected="turf",
    target=("compost",),
    reference=(
        "opened the plot rules again — the compost rules say green waste only in the bays, "
        "no turf, no roots and nothing cooked, and i've saved that instead. want me to set "
        "it running?"
    ),
)

# Case 5 — the wrong ITEM: this week's piece was stored and next week's is what was wanted.
# The tightest redirect of the five — two rows of one board, both a piece's title — so a
# corrected round that re-reads carelessly lands back on the value it already had.
_CORRECT_TO_NEXT_WEEK = _CorrectionCase(
    case_id="transition-learn-to-learn-neighbour",
    prior=_TEACH_REHEARSAL_PIECE,
    demonstrated=_REHEARSAL_DEMONSTRATED,
    skill=_REHEARSAL_SKILL,
    framing=_REHEARSAL_FRAMING,
    runs=_REHEARSAL_RUNS,
    correction="oops — i wanted next week's piece, not this week's",
    corrected="guillemot",
    target=("next week",),
    reference=(
        'fetched the rehearsal board again — next week\'s piece is "Guillemot", the closing '
        "chorus, and i've saved that instead of this week's. want me to keep it up to date "
        "on its own?"
    ),
)

# Every correction, in one place — so the deterministic pin in ``test_eval_harness.py`` can
# drive each one's seeder, its premise and its reply checks without a GPU.
CORRECTION_CASES = (
    _CORRECT_TO_STORM_SIGNAL,
    _CORRECT_TO_SOUTH_LOOP,
    _CORRECT_TO_FAMILY_FRIENDLY,
    _CORRECT_TO_COMPOST_RULES,
    _CORRECT_TO_NEXT_WEEK,
)

# The round adds two turns to the composed world — the teach and the report it closed on —
# so every reader of this world reads two rows further than the world underneath it.  Same
# derivation as the parked-request window: a ceiling over the world the case actually seeds,
# never a number picked to fit.
_CORRECTION_MESSAGE_WINDOW = _COMPOSED_MESSAGE_WINDOW + 2

# The provenance every teach round here carries (#1902): PRESENT, replacing NOTHING.
#
# Present because the round MINTED its routine — the framer ran at its entry, which is the
# only draw that mints and therefore the only one that records provenance at all.  Replacing
# nothing because the name that framing pinned was new to the registry at that moment:
# production reads ``snapshot_replaced_skill`` on the move that opens the teaching, and every
# one of these five rounds teaches a job none of the world's five journeys taught (asserted,
# rather than assumed, in the probe below — a fixture renamed onto a journey's routine would
# make the round a RE-TEACH and this the wrong state entirely).
#
# The production TYPE, serialized by the production call, rather than a hand-written JSON
# string — a seeded round has to be indistinguishable from one the machine really recorded,
# and a second spelling of this shape is one free to drift from it.
MINTED_FRESH = RoundProvenance()


# ── Seeding: the composed world, then the round the teach ran ─────────────────


def seed_corrected_round(case: _CorrectionCase) -> Seeder:
    """Lay down the world the correction is answered against: the composed five-journey
    history, then the teach round that beat 8 measured, walked to where it stops.

    Compositional by construction — the world underneath is ``seed_composed_world``'s, and
    what is added is one round's own footprint — so nothing here restates a history an
    earlier beat already defines.  The round is seeded through the state machine's real
    store, so what is laid down is a machine parked in learn rather than a shell of one.

    The fixture skills and the case's page are laid down by the runner after this, which is
    why the probe is a prepare hook rather than part of this seeder."""

    def seed(db: Database) -> None:
        seed_composed_world()(db)
        _seed_teach_round(db, case)

    return seed


def _seed_teach_round(db: Database, case: _CorrectionCase) -> None:
    """The teach and everything the round it opened left behind.

    One turn, not two: this is the single-turn teach beat 8 measures, so the round is opened
    and run by the same message and there is no elicit turn in front of it.  What it leaves
    is a machine parked in learn ON that message, carrying the framing AND the provenance
    the entry settled, and a container holding what the demonstration wrote."""
    teach_id = _log_ask(db, case.prior.teach, case.case_id)
    _log_reply(db, case.prior.reference, answering=teach_id)
    _seed_teach_turn_ledger(db, case)
    _park(
        db,
        ConversationState.LEARN,
        anchor_message_id=teach_id,
        run_id=case.runs.learn_turn,
        message_id=teach_id,
        framing=case.framing,
        provenance=MINTED_FRESH,
    )
    _seed_teach_container(db, case, teach_id)


def _seed_teach_turn_ledger(db: Database, case: _CorrectionCase) -> None:
    """The teach turn's promptlog — the draw that chose learn from a COLD machine offered
    every routine the world already holds, then the chat run's three steps (browse · write ·
    the closing report), each carrying the conversation as it stood when the call was made.

    The idle snapshot has no task anchor, which is what a teach arriving unprompted really
    is: nothing was parked, and the message both opened the round and ran it."""
    _log_classifier_draw(
        db,
        run_id=case.runs.learn_draw,
        snapshot=MachineSnapshot(
            state=ConversationState.IDLE,
            penny_last_turn=_IDLE_BANTER[-1].answered,
            skill_candidates=[_candidate(journey.round.skill) for journey in _JOURNEYS],
        ),
        message=case.prior.teach,
        drawn=_drawn_state(ConversationState.LEARN),
    )
    browse, write = _demonstrated_ledger(case.demonstrated)
    conversation: list[dict] = [{"role": "user", "content": case.prior.teach}]
    run_id = case.runs.learn_turn
    conversation = _seed_call_step(db, conversation, _BROWSE_CALL_ID, browse, run_id=run_id)
    _log_browse_extract(db, case.demonstrated, case.runs.browse_extract)
    conversation = _seed_call_step(db, conversation, _WRITE_CALL_ID, write, run_id=run_id)
    _log_chat_step(
        db, run_id=run_id, messages=conversation, response=_seeded_response(case.prior.reference)
    )


def _seed_teach_container(db: Database, case: _CorrectionCase, teach_id: int) -> None:
    """The round's CONTAINER as the entry framer builds it (#1868) — inert, described by the
    framer's own line, stamped with the run that created it and linked to the message that
    provoked it — holding the demonstrated write, plus the page the round read in
    browse-results.

    That last row is what makes one of this beat's failure shapes REACHABLE rather than
    hypothetical: the corrected value is already on a page the log holds, so a turn can
    reach it without fetching anything, and a check that only asked "was the new value
    stored?" would score that green."""
    demonstrated = case.demonstrated
    db.memories.create_collection(
        case.framing.container,
        case.framing.signature.description,
        created_by_run_id=case.runs.learn_turn,
    )
    db.memories.link_source_message(case.runs.learn_turn, teach_id)
    require_memory(db, demonstrated.collection).write(
        [EntryInput(key=demonstrated.entry_key, content=demonstrated.entry_value)],
        author=PennyConstants.CHAT_AGENT_NAME,
        run_id=case.runs.learn_turn,
    )
    section = f"{PennyConstants.BROWSE_PAGE_HEADER}{demonstrated.url}\n{case.prior.page.text}"
    require_memory(db, PennyConstants.MEMORY_BROWSE_RESULTS_LOG).append(
        [LogEntryInput(content=section)], author=PennyConstants.CHAT_AGENT_NAME
    )


# ── The loud probe: parked mid-round, on a page whose corrected value nobody has said ─


def _probe_correction_world(case: _CorrectionCase) -> Preparer:
    """The prepare hook: the world's own claims, the registry one that is only true once the
    runner has laid the fixture skills down, and the case's own premise."""

    def probe(penny: Penny) -> None:
        assert_the_teach_round_is_parked(penny.db, case)
        assert_the_correction_registry_holds(penny.db, case)
        assert_the_correction_is_unsaid(penny.db, case)

    return probe


def assert_the_teach_round_is_parked(db: Database, case: _CorrectionCase) -> None:
    """The world IS a completed teach round sitting inside the composed history — five jobs
    still running, this round readable under its own run, the conversation ending on the
    teach and the report, and the machine parked in learn on that teach with its framing.

    A seed that has drifted from the state beat 8 is measured against makes these cases
    turns answered against a world nothing produces, so it fails HERE rather than as a
    puzzling number after an hour of GPU time."""
    _assert_every_job_is_live(db, _JOURNEYS)
    _assert_the_corrected_conversation(db, case)
    _assert_the_teach_round_is_in_the_ledger(db, case)
    _assert_parked_on_the_teach(db, case)


def _assert_the_corrected_conversation(db: Database, case: _CorrectionCase) -> None:
    """The world reads back as the conversation it claims to be — the composed history's own
    turns, then the teach and the report that closed the round.

    Read through ``get_messages_since``, the reader ``_build_conversation`` uses, for the
    reason the composed probe reads it there: an unthreaded reply is in the record and out
    of the conversation, and only the parent link tells the two apart."""
    expected = [
        *expected_conversation(_JOURNEYS),
        (PennyConstants.MessageDirection.INCOMING, case.prior.teach),
        (PennyConstants.MessageDirection.OUTGOING, case.prior.reference),
    ]
    window = db.messages.get_messages_since(
        TEST_SENDER, since=datetime.min, limit=_CORRECTION_MESSAGE_WINDOW
    )
    seen = [(row.direction, row.content) for row in window]
    assert seen == expected, (
        f"{case.case_id}: the seeded world must read back as the conversation it claims — "
        f"diverges at turn {_first_divergence(seen, expected)}"
    )
    _assert_every_reply_is_threaded(window)


def _assert_the_teach_round_is_in_the_ledger(db: Database, case: _CorrectionCase) -> None:
    """The teach round is READABLE as the turn it really was: its two calls under its own
    run, the container and the entry citing that run, and its page in browse-results beside
    the five the composed world's rounds read.

    The page count is the claim the shared composed probe cannot make here — this world has
    read one page more than the journeys did, and that extra page is the very one the
    correction sends the round back to."""
    calls = [
        call.get("function", {}).get("name")
        for row in db.messages.get_run_prompts(case.runs.learn_turn)
        for call in _row_tool_calls(row)
    ]
    expected = [step.tool for step in _demonstrated_ledger(case.demonstrated)]
    assert calls == expected, f"{case.case_id}: the seeded round must carry {expected}, got {calls}"
    row = db.memories.get(case.framing.container)
    assert row is not None and row.created_by_run_id == case.runs.learn_turn, (
        f"{case.case_id}: the round's container must exist and cite the round's run"
    )
    assert row.skill_name is None and row.extraction_prompt is None and row.schedule is None, (
        f"{case.case_id}: the container is INERT — learning instantiates nothing"
    )
    stored = collection_entries(db, case.framing.container)
    assert stored.get(case.demonstrated.entry_key) == case.demonstrated.entry_value, (
        f"{case.case_id}: the demonstrated fact must be in the round's container, got {stored}"
    )
    assert len(_pages_fetched(db)) == len(_JOURNEYS) + 1, (
        f"{case.case_id}: each journey read one page and so did this round"
    )


def _assert_parked_on_the_teach(db: Database, case: _CorrectionCase) -> None:
    """The machine is parked in learn ON THE TEACH, carrying both pieces of entry state the
    round settled — the framing a re-entry carries, and the provenance a bail would read.

    Both are what the correction turn is answered against: the framing is the identity the
    re-entry keeps rather than re-draws (#1902), and the provenance is what says the routine
    standing under that name is the round's own."""
    teach_id = _seeded_ask_id(db, case.prior.teach, limit=_CORRECTION_MESSAGE_WINDOW)
    assert teach_id is not None, f"{case.case_id}: the seeded teach must be findable by content"
    latest = db.machine.latest_transition()
    assert latest is not None and latest.to_state == ConversationState.LEARN.value, (
        f"{case.case_id}: the machine must be parked in learn, not {latest}"
    )
    assert latest.anchor_message_id == teach_id, (
        f"{case.case_id}: the round must be anchored to the teach, not {latest.anchor_message_id}"
    )
    assert latest.skill_frame == case.framing.model_dump_json(), (
        f"{case.case_id}: the learn move must carry the round's framing, not {latest.skill_frame}"
    )
    _assert_the_round_minted_its_routine(db, case, latest)


def _assert_the_round_minted_its_routine(
    db: Database, case: _CorrectionCase, latest: StateTransition
) -> None:
    """The round's PROVENANCE says it minted its routine over nothing (#1902) — present on
    the move that opened the teaching, carrying no replaced row, and pinned to the name the
    round's framing settled.

    Three claims, each silent on a run if it breaks.  An ABSENT provenance is the state a
    skill-gated round records, so a world seeded without one describes a round that taught
    nothing — and a bail from it would leave the routine standing.  A provenance carrying a
    replaced ROW would say the user already had this routine, which would make the corrected
    round a re-teach and a bail a restore.  And the state is only truthful while the round's
    pinned name really is new to the world: production reads the registry at the moment the
    round opens, so a fixture renamed onto one of the journeys' routines would be a re-teach
    seeded as a mint."""
    assert latest.round_provenance == MINTED_FRESH.model_dump_json(), (
        f"{case.case_id}: the opening move must carry a minted-fresh provenance, "
        f"not {latest.round_provenance}"
    )
    assert case.framing.skill == case.skill.name, (
        f"{case.case_id}: the round's framing must pin the routine it taught, "
        f"not {case.framing.skill!r}"
    )
    taught_before = {slug_skill_name(journey.round.skill.name) for journey in _JOURNEYS}
    assert slug_skill_name(case.framing.skill) not in taught_before, (
        f"{case.case_id}: {case.framing.skill!r} is one of the world's routines — the round "
        "would be RE-teaching it, so minted-fresh is the wrong provenance"
    )


def assert_the_correction_registry_holds(db: Database, case: _CorrectionCase) -> None:
    """The registry holds the world's five routines and the one this round taught — and
    nothing else.

    The count is what the fork check reads against, so a world arriving with an extra
    routine would make "the re-extraction replaced rather than forked" pass or fail for a
    reason nobody could see."""
    taught = sorted(skill.name for skill in db.skills.list_all())
    expected = sorted(
        [
            *(slug_skill_name(journey.round.skill.name) for journey in _JOURNEYS),
            slug_skill_name(case.skill.name),
        ]
    )
    assert taught == expected, f"{case.case_id}: the registry must hold {expected}, got {taught}"


def assert_the_correction_is_unsaid(db: Database, case: _CorrectionCase) -> None:
    """The case's premise: the correction points at something REAL on the page it was taught
    on, and nobody has said it yet.

    Four claims, each one silent on a run if it breaks.  The page must actually carry the
    corrected target, or the corrected round is contracted to find something that is not
    there.  The correction must NOT carry it, or a stored value proves nothing about a page
    being read.  It must differ from the value the round already stored, or the redirect is
    a redirect to where the round already was.  And it must be new to everything this world
    has SAID or STORED — never to the pages it has read, which is where it lives by
    construction and is exactly what makes the applied-the-delta-without-re-running shape
    reachable."""
    assert case.corrected.lower() in case.prior.page.text.lower(), (
        f"{case.case_id}: the page must carry {case.corrected!r} for the corrected round to find"
    )
    assert case.corrected.lower() not in case.correction.lower(), (
        f"{case.case_id}: the correction must not carry its own answer {case.corrected!r}"
    )
    assert case.corrected.lower() != case.prior.stored.lower(), (
        f"{case.case_id}: the corrected value must differ from the one the round stored"
    )
    assert not _mentions(case.corrected, _spoken_and_stored(db)), (
        f"{case.case_id}: {case.corrected!r} must be unsaid and unstored in this world"
    )


# ── Scoring: the shapes, told apart ──────────────────────────────────────────


def _addresses_asked_for(db: Database) -> list[str]:
    """Every address this turn asked a browse FOR — the queries it sent, deliberately not
    the pages that came back (``_pages_fetched``, which reads the browse-results log).

    The two are different questions and this beat needs this one: what is being asked is
    whether the round went and looked again, and the log it would otherwise be read from
    already holds the page from the round being corrected."""
    return [
        query
        for row in live_prompts(db)
        for call in _row_tool_calls(row)
        for query in _call_queries(call)
    ]


def _call_queries(call: dict) -> list[str]:
    """The addresses one logged call asked for, or nothing at all — the browse tool's own
    argument, decoded the way the ledger stores it (a JSON STRING, never a mapping).

    An undecodable argument blob reads as no addresses rather than raising, the same reading
    ``tool_call_arg_values`` makes: a malformed draw is a thing a live model produces, and a
    scorer that died on one would lose the whole sample rather than score it."""
    if call.get("function", {}).get("name") != _BROWSE_TOOL:
        return []
    try:
        arguments = json.loads(call.get("function", {}).get("arguments") or "{}")
    except json.JSONDecodeError, TypeError:
        return []
    return [query for query in arguments.get("queries") or [] if isinstance(query, str)]


class _CorrectionReadings(NamedTuple):
    """What this turn DID with the correction, read ONCE and shared by everything that
    conditions on it — the same discipline ``_landed_in`` keeps for the turn's last move.

    Two checks and the shape line all ask these same questions, and three independent
    re-readings of the ledger are three answers waiting to disagree about one sample: a
    report whose rows say the page was fetched again and whose shape line says it was not is
    a report nobody can act on."""

    fetched: list[str]
    written: list[str]
    refetched: bool
    stored: bool
    kept: bool
    said: bool


def _correction_readings(db: Database, case: _CorrectionCase) -> _CorrectionReadings:
    """The observations, taken off the ledger together: what this turn asked a browse for,
    what it wrote, and whether each of the two values in play turned up — in what it stored
    and, for the corrected one, in what it said."""
    fetched = _addresses_asked_for(db)
    written = _written_texts(_entries_written_by_this_run(db))
    return _CorrectionReadings(
        fetched=fetched,
        written=written,
        refetched=any(_said_back(case.prior.url, query) for query in fetched),
        stored=_mentions(case.corrected, written),
        kept=_mentions(case.prior.stored, written),
        said=_mentions(case.corrected, outgoing_replies(db)),
    )


def _refetched_check(readings: _CorrectionReadings) -> Check:
    """The round RE-RAN: this turn asked for the page it was taught on.

    The distinction the beat turns on, and the reason it is the page rather than any fetch:
    the corrected value is already in browse-results, so a turn can produce it without going
    anywhere, and "did a browse happen" would score that identically to a real re-run.  The
    address is matched with its scheme stripped, because a page named back in the user's own
    scheme-less form is plainly the same page."""
    fetched = readings.fetched
    return Check(
        "state: the round re-ran on the page it was taught on",
        readings.refetched,
        rationale=None
        if readings.refetched
        else (f"fetched {fetched}" if fetched else "nothing was fetched"),
        kind="state",
    )


def _superseded_check(readings: _CorrectionReadings) -> Check:
    """The value the correction REPLACED is not this round's result.

    Scored apart from "the corrected value landed" because the two miss for different
    reasons: a round that stored the old value again re-ran without applying the correction,
    while a round that stored both applied it without letting go of what it replaced.  Read
    over this run's own writes, so the entry the seeded round left behind — which still
    holds the old value, and legitimately — is none of this turn's business."""
    return Check(
        "state: the superseded value is not this round's result",
        not readings.kept,
        rationale=f"wrote {readings.written}" if readings.kept else None,
        kind="state",
    )


def _one_routine_check(db: Database, learned: list[Skill]) -> Check:
    """The correction left ONE routine for the task — re-extraction REPLACED rather than
    forked (#1706/#1827).

    Read against the world's own five, so what is counted is the routines this round is
    responsible for however the round ended up named.  Two is the fork: the re-extraction
    filed beside the routine the teach taught rather than over it, leaving the user two
    routines for one job.  Since #1902 the write is KEYED by the name the round's framing
    pinned, so this is the gate case watching that — a regression there would put the second
    routine back and nothing else would notice.

    Not applicable when nothing was re-extracted, the same guard its three siblings carry:
    the world arrives holding exactly one routine for this round, so a turn that learned
    nothing would pass a check whose label claims a replacement happened — and that absence
    is already the scored "a skill was learned from the round"."""
    label = "state: one routine for the round (the re-extraction replaced rather than forked)"
    if not learned:
        return Check.na(label, kind="state")
    world = {slug_skill_name(journey.round.skill.name) for journey in _JOURNEYS}
    for_the_round = sorted({skill.name for skill in db.skills.list_all()} - world)
    return Check(
        label,
        len(for_the_round) == 1,
        rationale=None
        if len(for_the_round) == 1
        else f"{len(for_the_round)} routines for the round: {for_the_round}",
        kind="state",
    )


def _kept_its_container_check(db: Database, before: set[str], case: _CorrectionCase) -> Check:
    """The corrected round ran into the container it already had — no SIBLING beside it.

    The other half of the fork, on the store rather than the registry: find-or-create means
    a round that keeps its identity derives the same name and continues into what it was
    already writing, while one that shifts it builds a second container for what the user
    experiences as one job.  Since #1902 a correction CARRIES the round's framing and
    re-settles only the container, so this is the gate case watching that carry.

    Conditioned on the machine landing in LEARN rather than on anything being learned, since
    the container is settled by the ENTRY draw and a re-framing that minted a sibling and
    then failed extraction has forked the store all the same.  A turn that went somewhere
    else re-framed nothing, so there is no landing here to grade — that miss is the
    landed-state advisory's."""
    label = "state: the corrected round kept the container it already had"
    if _landed_in(db.machine.latest_transition(), ConversationState.LEARN) is None:
        return Check.na(label, kind="state")
    minted = [row.name for row in new_collections(db, before)]
    return Check(
        label,
        not minted,
        rationale=f"minted {minted} beside {case.framing.container!r}" if minted else None,
        kind="state",
    )


def _demonstrated_values(steps: list[SkillStep]) -> list[str]:
    """Every string leaf the re-extracted routine's steps were demonstrated with — the
    verbatim call arguments the ledger copied.

    Read as a flat list of VALUES rather than per tool or per argument, because a skill is an
    arbitrary sequence of tool calls: which call carries the page and which carries what to
    look for is not something a scorer can know, and a reading keyed to either would stop
    firing the moment a round is demonstrated with a tool nobody enumerated."""
    return [text for step in steps for text in _leaf_strings(step.arguments)]


def _leaf_strings(node: object) -> list[str]:
    """Every string at the leaves of one call's arguments, however deeply they nest."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [text for value in node.values() for text in _leaf_strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _leaf_strings(value)]
    return []


def _kept_the_page_check(learned: list[Skill], case: _CorrectionCase) -> Check:
    """The re-extracted routine still points at the page it was taught on — a step the
    correction said nothing about, and the first way a delta loses the parts it did not
    touch."""
    label = "state: the re-extracted routine still points at the page it was taught on"
    if not learned:
        return Check.na(label, kind="state")
    values = _demonstrated_values(_skill_steps(learned))
    kept = any(_said_back(case.prior.url, value) for value in values)
    return Check(
        label,
        kept,
        rationale=None if kept else f"the routine names no page: {values}",
        kind="state",
    )


def _kept_the_write_check(db: Database, learned: list[Skill]) -> Check:
    """The re-extracted routine still KEEPS what it finds — the other uncorrected step.

    Read the way every destination is read here (#1783/#1854): a leaf whose demonstrated
    value names one of Penny's own collections, through the same registry policy extraction
    marks on, so a routine that stores through a verb nobody enumerated still counts."""
    label = "state: the re-extracted routine still keeps what it finds"
    if not learned:
        return Check.na(label, kind="state")
    destinations = _destination_subs(db, _skill_steps(learned))
    return Check(
        label,
        bool(destinations),
        rationale=None if destinations else "the routine keeps nothing",
        kind="state",
    )


def _corrected_step_check(learned: list[Skill], case: _CorrectionCase) -> Check:
    """The re-extracted routine looks for what the CORRECTION named — the step that did
    change, present in the program rather than only in the reply.

    A routine that re-ran correctly and then distilled the instruction it was originally
    given is a routine that will fetch the wrong thing every cycle, which is the harm this
    beat's re-extraction half is about."""
    label = "state: the re-extracted routine looks for what the correction named"
    if not learned:
        return Check.na(label, kind="state")
    values = _demonstrated_values(_skill_steps(learned))
    named = _mentions_any(case.target, " ".join(values))
    return Check(
        label,
        named,
        rationale=None if named else f"names none of {list(case.target)}: {values}",
        kind="state",
    )


def _correction_anchor_check(db: Database, case: _CorrectionCase) -> Check:
    """The round stayed on its OWN anchor: the move came from learn and still points at the
    teach that opened it (#1827's anchor rule — a correction continues a round rather than
    starting one).

    Scored only when the machine landed in learn, the same conditional every other beat's
    anchor check uses: a misroute is already named by the landed-state advisory, and scoring
    the anchor on top of it would recount one classifier miss as an enactment failure."""
    label = "state: the correction continued the round it corrects (from learn, same anchor)"
    latest = db.machine.latest_transition()
    if latest is None or latest.to_state != ConversationState.LEARN.value:
        return Check.na(label, kind="state")
    taught = _seeded_ask_id(db, case.prior.teach, limit=_CORRECTION_MESSAGE_WINDOW)
    continued = latest.from_state == ConversationState.LEARN.value
    ok = continued and taught is not None and latest.anchor_message_id == taught
    return Check(
        label,
        ok,
        rationale=None
        if ok
        else (
            f"came from {latest.from_state}, anchored to {latest.anchor_message_id} "
            f"(the teach is {taught})"
        ),
        kind="state",
    )


# Every way a correction can be answered, named once — the phrases the report hands the code
# owner, and diff-join keys like every other label here: two of them are asserted from the
# deterministic pin as well, so a wording spelled out at both sites would drift a word at a
# time.  There is one per combination of the three things a turn either did or did not do,
# plus the claim, which is only evidence where nothing was fetched and nothing was written.
SHAPE_RE_RAN_AND_APPLIED = "re-ran with the correction applied"
SHAPE_RE_RAN_AND_KEPT_BOTH = "re-ran and stored both values"
SHAPE_RE_RAN_UNAPPLIED = "re-ran without applying it"
SHAPE_RE_RAN_AND_STORED_NOTHING = "re-ran and stored neither value"
SHAPE_DELTA_WITHOUT_RE_RUNNING = "applied the delta without re-running"
SHAPE_DELTA_AND_OLD_WITHOUT_RE_RUNNING = "stored both values without re-running"
SHAPE_OLD_WITHOUT_RE_RUNNING = "re-stored the old value without re-running"
SHAPE_CLAIMED_WITHOUT_RE_RUNNING = "claimed the corrected value without re-running"
SHAPE_NEITHER = "neither ran nor applied"


def _correction_shape(*, refetched: bool, stored: bool, kept: bool, said: bool) -> str:
    """WHICH shape this sample landed in, in one phrase.

    The checks above each answer one question, and the code owner's question is about the
    COMBINATION — whether a delta reached the instructions at all, and whether the flow that
    followed was the corrected one.  So the readings are named here once, together, rather
    than left to be reassembled from three rows in a report.

    Every combination of the three observations gets its own phrase.  Collapsing any pair is
    the failure this exists to prevent, and the first draft did exactly that: a run that
    stored the corrected value AND re-stored the one it replaced read as a clean delta-apply,
    which is the difference between a small coherence miss and a routine that keeps both.
    The CLAIM breaks a tie in one place only — where nothing was fetched and nothing was
    written, a reply is the only evidence there is."""
    if refetched:
        if stored:
            return SHAPE_RE_RAN_AND_KEPT_BOTH if kept else SHAPE_RE_RAN_AND_APPLIED
        return SHAPE_RE_RAN_UNAPPLIED if kept else SHAPE_RE_RAN_AND_STORED_NOTHING
    if stored:
        return SHAPE_DELTA_AND_OLD_WITHOUT_RE_RUNNING if kept else SHAPE_DELTA_WITHOUT_RE_RUNNING
    if kept:
        return SHAPE_OLD_WITHOUT_RE_RUNNING
    return SHAPE_CLAIMED_WITHOUT_RE_RUNNING if said else SHAPE_NEITHER


def _correction_shape_advisory(readings: _CorrectionReadings) -> Check:
    """The shape, as an ADVISORY row — the answer read at joint review, beside the scored
    checks it is composed from and off the same reading they are, so the two can never
    disagree about one sample."""
    return Check(
        "shape: how the correction was answered",
        True,
        rationale=_correction_shape(
            refetched=readings.refetched,
            stored=readings.stored,
            kept=readings.kept,
            said=readings.said,
        ),
        scored=False,
        kind="state",
    )


def _re_extraction_checks(db: Database, learned: list[Skill], case: _CorrectionCase) -> list[Check]:
    """What the RE-EXTRACTION left in the registry: one routine for the round, carrying the
    step the correction changed over the two it never mentioned.

    Grouped because they are one reading of one row — a correction's durable result is the
    routine it leaves behind, and asking whether that routine is singular, still pointed at
    its page, still keeping what it finds, and now looking for the right thing is four
    questions about the same object."""
    return [
        _one_routine_check(db, learned),
        _kept_the_page_check(learned, case),
        _kept_the_write_check(db, learned),
        _corrected_step_check(learned, case),
    ]


def _score_learn_to_learn(
    db: Database, before: set[str], reply: str, *, case: _CorrectionCase
) -> list[Check]:
    """The correction was applied TO the instructions already given, and the whole corrected
    flow was carried out.

    The demonstrated round's own contract, re-run: the page read again, the CORRECTED value
    landed in the round's container, and nothing set up.  Beside it, the claims that are this
    beat's own — the page was really re-fetched, the superseded value is not the result, one
    routine and one container came out of it, and the re-extracted program carries the
    corrected step over the steps the correction never mentioned.

    ONE scorer for all five cases, bound to the case's own page tokens.  The labels are
    diff-join keys and the ones shared with the learn beats keep their wording, so all three
    learn entries report under the same rows."""
    created = new_collections(db, before)
    framing = _round_framing(db)
    learned = _learned_this_turn(db)
    readings = _correction_readings(db, case)
    return [
        _refetched_check(readings),
        *_round_ran_checks(db, case.corrected),
        _superseded_check(readings),
        *_framed_checks(db, framing),
        _wrote_into_the_container_check(db, framing),
        _kept_its_container_check(db, before, case),
        Check("state: a skill was learned from the round", bool(learned), kind="state"),
        *_re_extraction_checks(db, learned, case),
        *_attaches_nothing_checks(db, created, already_running=_LIVE_JOB_CONTAINERS),
        Check("state: she configured nothing", tool_not_called(db, _SET_TOOL), kind="state"),
        *_extraction_shape_checks(db, learned),
        _correction_anchor_check(db, case),
        _seeded_jobs_untouched_check(db),
        *_round_reported_checks(case.corrected, reply, outgoing_replies(db)),
        _correction_shape_advisory(readings),
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


async def _run_correction_case(chat_eval: ChatEval, case: _CorrectionCase) -> None:
    """Drive one learn → learn case: the composed world with its taught round parked in it,
    the five journeys' routines and this round's in the registry, the page its instructions
    named installed so the corrected round reads a real one, and the shared scorer bound to
    the tokens that page carries.  Report-only — the thresholds are the code owner's to set
    once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        message=case.correction,
        browse=[case.prior.page],
        seed=seed_corrected_round(case),
        seed_skills=[*(journey.round.skill for journey in _JOURNEYS), case.skill],
        prepare=_probe_correction_world(case),
        score=partial(_score_learn_to_learn, case=case),
        min_pass_rate=None,
        timeout=240.0,
        family=_FAMILY,
    )


@pytest.mark.asyncio
async def test_learn_to_learn_re_runs_the_round_on_the_corrected_target(
    chat_eval: ChatEval,
) -> None:
    """learn → learn, the plainest delta: the round read today's flag and the storm signal
    beside it is what was wanted.  The page is re-read, the storm signal's own value stored
    in place of the flag's, the routine re-extracted over the corrected instruction, and the
    offer made again with nothing set running."""
    await _run_correction_case(chat_eval, _CORRECT_TO_STORM_SIGNAL)


@pytest.mark.asyncio
async def test_learn_to_learn_corrects_a_value_the_round_is_identified_by(
    chat_eval: ChatEval,
) -> None:
    """learn → learn where the corrected piece is a VALUE the round's framing carries: the
    routine was taught on the north loop and the south loop is what was meant.  The
    container's name is derived from that value, so this is the case where keeping one job
    in one place is a real question rather than an arithmetic certainty."""
    await _run_correction_case(chat_eval, _CORRECT_TO_SOUTH_LOOP)


@pytest.mark.asyncio
async def test_learn_to_learn_swaps_the_filter_the_round_was_taught(chat_eval: ChatEval) -> None:
    """learn → learn where the correction retires a FILTER: the round kept the free event
    and the criterion is family-friendly now, with free explicitly no longer mattering.  The
    page pulls the two apart, so a round that kept the old filter lands on a different
    event and says so."""
    await _run_correction_case(chat_eval, _CORRECT_TO_FAMILY_FRIENDLY)


@pytest.mark.asyncio
async def test_learn_to_learn_corrects_without_shaking_the_deferred_term_loose(
    chat_eval: ChatEval,
) -> None:
    """learn → learn on the round whose teach also stated a NOTIFY condition: the correction
    moves it from the watering restriction to the compost rules, and the condition stays
    where it was — waiting for the turn that accepts the offer.  Configuring it here is the
    teach-and-instantiate fold, reached this time through a correction."""
    await _run_correction_case(chat_eval, _CORRECT_TO_COMPOST_RULES)


@pytest.mark.asyncio
async def test_learn_to_learn_moves_one_row_over_on_the_same_board(chat_eval: ChatEval) -> None:
    """learn → learn on the tightest redirect: this week's piece was stored and next week's
    is what was wanted — two rows of one board, both a piece's title — so a re-read that is
    not careful lands back on the value the round already had."""
    await _run_correction_case(chat_eval, _CORRECT_TO_NEXT_WEEK)
