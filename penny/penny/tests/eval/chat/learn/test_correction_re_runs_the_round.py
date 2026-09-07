"""learn → learn: the correction re-runs the round against the corrected target (#2005, t3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

The round was taught and run, and Penny closed it by reporting what she stored.  The user now
corrects what the demonstration was AIMED at — same page, one line over — and the turn re-runs
the round on the corrected instructions: the corrected value lands in the round's own
container, the value it replaced does not, and the routine the round left behind is updated in
place rather than forked.

**The survivor, and on what basis: MEASURED RATE.**  The edge's five variants have real
per-variant numbers over the seven suite runs that carried all five.  Discounting the one run
where every case in the suite collapsed together (0.54 on all five, an infrastructure failure
rather than a behaviour), the corrected-parameter redirect leads on both readings that separate
them: 0.500 of samples fully passing (the best of the five, against 0.433 · 0.400 · 0.367 ·
0.267) and the highest floor, never below 0.89 mean where `-filter` and `-neighbour` both fall
to 0.62.  Its mean, 0.945, is a third of a claim behind `-deferred-terms`' 0.948 on a
fifteen-check case — noise, and the all-pass reading is what "already passes consistently"
means at the sample level.

Dominance agrees rather than competing.  What the correction moves here is a value the round's
IDENTITY is derived from — the container's name is `derive_collection_name(routine, values)` —
so keeping the round in one place is a real question on this world and an arithmetic certainty
on the others, which is what makes it the world able to produce the fork this case's claims are
about.

The four dropped variants are quarantined rather than deleted, each with the temptation it
probes: the storm-signal redirect (the plainest delta, one line over on the same page, and the
one whose correction opens with a self-correction), `-filter` (the criterion swapped, with the
old one retired in as many words), `-deferred-terms` (a notify condition the teach DEFERRED,
which must not be shaken loose into a job by a correction), `-neighbour` (two rows of one board,
so a careless re-read lands back on the value it already had).  Every `_CorrectionCase` fixture
stays in this file, so any of them can come back deliberately.

**What is claimed is the corrected round's END STATE, in two places.**  The STORE: the corrected
value landed, in the round's own container, and the value it supersedes is not among this run's
writes.  The REGISTRY: one routine stands for the round — replaced, not forked — and it now
looks for what the correction named while still naming the page and still keeping what it finds.
The registry half reads the routine's demonstrated values, flat and tool-agnostic, because a
correction that reaches the reply and not the program leaves a routine that fetches the wrong
thing every cycle.

**Six source checks did not port** (the outward column):

* ``_refetched_check`` — *the round re-ran on the page it was taught on*, read off the addresses
  this turn asked a browse FOR.  A ROUTE: `LANDED`? no, a fetch is not a landing.  `STORE`? no,
  nothing was stored by asking.  It is model output, and it is measured in section B, where a
  cohort that stopped re-reading shows as a variance rise rather than as one sample's failed
  check.  Its end-state form is the corrected-value claim below — the correction does not carry
  its own answer and the world has never said or stored it, so a sample holding it read a page.
  What that claim deliberately CANNOT distinguish is a re-fetch from a read of the round's own
  page sitting in browse-results; the design's answer to that is the tool sequence, not a claim.
* ``_superseded_check`` — the right question, read over this run's writes against a value spelled
  out in the case.  It ports as ``assert_nothing_excluded_was_stored``, which asks it of the
  world's own excluded token, so the comparison is a read of the page rather than a fixture
  field the claim and the world could disagree about.
* ``_kept_its_container_check`` — *the corrected round kept the container it already had*.  Ports
  as ``assert_no_mechanism_was_created``: a round that shifted its identity mints a SIBLING
  container, which is a created mechanism however the shift happened.
* ``_correction_anchor_check`` — PRODUCTION ALREADY VALIDATES IT.  ``_next_anchor`` keeps the
  anchor of a machine that is already parked, and the from-state is what this world seeds, so
  both halves are entailed by the landing.
* ``Check("state: she configured nothing", tool_not_called(db, _SET_TOOL))`` — a ROUTE keyed to
  a tool NAME.  Its end-state form is *the round's container is still inert*, which catches a
  configuration however it was reached and catches a plugin verb nobody enumerated.
* ``*_round_reported_checks`` — PHRASING matches on the reply, which is the thing this design
  exists to abolish.  ``test_eval_harness.py`` still drives them over all five references, which
  is where a scorer that cannot pass the answer the case itself calls correct gets caught.

**And the shape naming retires with the scorer.**  ``_correction_shape`` composed four readings
into one phrase for the code owner to read; three of them now live where the design puts them —
*stored* and *kept* are claims, *refetched* is the tool sequence — and the fourth (*said*) was a
phrasing match.  What the phrase gave that no single row does is the COMBINATION, and the
report's modal-sample reading is what replaces it.  Recorded here rather than silently dropped.

**Two the inward column added**: PROVENANCE, of both kinds.  The source case made no claim of
either, so a sample that stored a value nobody's page carries — or reported one — passed
everything it had.

**`answers` is EMPTY, and that is a report.**  A correction asks for no value: it says which
line was meant, and the corrected reading is what the round goes and finds.  Requiring a token
of the reply would fail a correct run for something nobody requested.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.  Every page, url and job is synthetic, on an
``example`` domain, because the repo is public.
"""

from __future__ import annotations

from datetime import datetime
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
from penny.database.models import StateTransition
from penny.database.skills import (
    SkillDraft,
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
    EVAL_MODELS,
    ChatEval,
    Preparer,
    Seeder,
    collection_entries,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    ROUTINE_SHAPE,
    TOOL_SEQUENCE,
    TRANSITIONS,
    RoutineRecord,
    SampleObservation,
    SpecCategory,
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
    _FAMILY,
    _WRITE_CALL_ID,
    _drawn_state,
    _journey_runs,
    _JourneyRuns,
    _log_ask,
    _log_chat_step,
    _log_classifier_draw,
    _log_reply,
    _pages_fetched,
    _park,
    _seeded_response,
)
from penny.tests.eval.utils.transition_world import (
    _CHOIR_REHEARSALS_URL,
    _CLIFF_WALK_URL,
    _COMPOSED_MESSAGE_WINDOW,
    _HARBOUR_SIGNALS_URL,
    _IDLE_BANTER,
    _JOURNEYS,
    _PLOT_RULES_URL,
    _TEACH_CLIFF_WALK,
    _TEACH_FREE_EVENT,
    _TEACH_HARBOUR_FLAG,
    _TEACH_REHEARSAL_PIECE,
    _TEACH_WATERING_RULE,
    _TOWN_HALL_EVENTS_URL,
    _assert_every_job_is_live,
    _assert_every_reply_is_threaded,
    _candidate,
    _demonstrated_ledger,
    _DemonstratedRound,
    _first_divergence,
    _fixture_skill,
    _FixtureDraws,
    _framed,
    _log_browse_extract,
    _mentions,
    _row_tool_calls,
    _said_back,
    _seed_call_step,
    _seeded_ask_id,
    _spoken_and_stored,
    _TeachCase,
    expected_conversation,
    seed_composed_world,
    tool_call_name,
)
from penny.tests.eval.utils.worlds import World

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
    case_id="transition-learn-to-learn-storm-signal",
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
    case_id="transition-learn-to-learn",
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
        assert_every_wording_names_the_corrected_line()

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
        tool_call_name(call)
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


# ── The survivor, its wordings, and the world they are answered against ───────

_SURVIVOR = _CORRECT_TO_SOUTH_LOOP

# The ids these two carry swapped when #2005 collapsed the edge onto one canonical case: the
# SURVIVOR takes the edge's id, because the epic is one case per EDGE and the edge is what an
# id names, and the quarantined variant takes an id naming what it now is.  ONE id must never
# name two cases — ``case_id`` is threaded through the seeder and every probe assertion, so a
# probe failure on the live case would print the quarantined fixture's name.  Read off the
# survivor rather than restated, so the two can never disagree.
_CASE_ID = _SURVIVOR.case_id

# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The case id is a filename; this is the contract.
_BEHAVIOUR = (
    "In the chat agent, when the user corrects what a demonstrated round was aimed at, Penny "
    "re-runs the round against the corrected target — keeping what it says in the round's own "
    "container, letting go of the value it replaces, and updating the routine she taught rather "
    "than filing a second one beside it."
)

# Four more wordings of that same correction.  What varies is only how a person says they meant
# something else — which apology opens it, "use that one" or "go with that", whether the wrong
# line is named at all.  What does NOT vary is WHICH line is meant: the case claims the value
# that line carries, and it can only do that because the target is constant across the arms.
_CORRECTION_PHRASINGS = (
    "oops, i meant the south loop — use that one instead",
    "actually it's the south loop line i want, not that one",
    "my mistake — the south loop is the one, use that",
    "sorry, wrong line — i meant the south loop, go with that",
)

# The round's own container and the token the correction names, read off the survivor rather
# than restated: the container's name is derived from the routine and its values, and a second
# copy of either here would be free to drift from the round the world actually seeds.
_ROUND_CONTAINER = _SURVIVOR.framing.container
_CORRECTED_TARGET = _SURVIVOR.target[0].lower()

# The ground every arm is answered against: the one page the round was taught on, which is also
# where the corrected reading lives.
#
# ``keeps`` is the corrected value's own distinctive token and ``excludes`` is the superseded
# one's — a pair that is mutually exclusive and absent from everything else this page carries,
# which is what lets the two claims read the page rather than a taste.  ``answers`` is EMPTY and
# that is a report; the module docstring says why.
_CORRECTED_TRAIL = World(
    name=_CASE_ID,
    pages=(_SURVIVOR.prior.page,),
    keeps=((_SURVIVOR.corrected,),),
    excludes=(_SURVIVOR.prior.stored,),
)


def assert_every_wording_names_the_corrected_line() -> None:
    """Every arm names the line the correction redirects to, and none of them carries the answer.

    The facts are held constant across a cohort's arms because the assertions hinge on them, and
    both halves are load-bearing here: a wording that named no line would be a correction with no
    target, and one that spelled out the corrected VALUE would let a sample store it without
    reading anything.  The target is matched case-folded, since which case a person types is
    exactly what a paraphrase is free to vary.

    It takes no case, because it is not general: the wordings and the target it reads are this
    module's own survivor.  A ``case`` parameter would read as generality it does not have —
    hand it another correction and it would assert that this survivor's target is in it."""
    for wording in (_SURVIVOR.correction, *_CORRECTION_PHRASINGS):
        assert _CORRECTED_TARGET in wording.lower(), (
            f"{_CASE_ID}: this wording names no target — {wording!r}"
        )
        assert _SURVIVOR.corrected.lower() not in wording.lower(), (
            f"{_CASE_ID}: this wording carries its own answer {_SURVIVOR.corrected!r} — {wording!r}"
        )


# ── The claims: what the corrected round left in the store and in the registry ─
#
# Four of them are local to this case, and each is parametrised by the correction's own target,
# so none could graduate even at a second customer.  What they read is the routine the ROUND
# owns — the row standing under the name its framing pinned — never "the routines", because the
# world holds five more that its history taught and a claim over all of them would be answered
# mostly by the fixture.
#
# What no claim here reads is a TOOL NAME or a step ORDER: a skill is an arbitrary tool
# sequence, so the question is what the registry holds afterwards and never which verb put it
# there.  ``RoutineRecord.demonstrated_values`` is flat for the same reason — which call carries
# the page and which carries what to look for is not something a claim can know.


_ROUND_ROUTINE = slug_skill_name(_SURVIVOR.skill.name)

# The routines this world's history taught, which the round's own is NOT one of (the probe
# asserts exactly that).  Named once because two claims subtract it.
_WORLD_ROUTINES = frozenset(slug_skill_name(journey.round.skill.name) for journey in _JOURNEYS)


def _the_rounds_routine(sample: SampleObservation) -> RoutineRecord | None:
    """The routine the round owns, by the name its framing pinned.

    ``None`` where nothing stands under that name, which is a real reading — a correction whose
    re-extraction failed leaves the round with no routine at all — and every claim below answers
    FALSE on it rather than returning early, because a routine that is not there does not point
    at a page, does not look for anything, and does not keep what it finds."""
    return next((one for one in sample.routines if one.name == _ROUND_ROUTINE), None)


def _one_routine_stands_for_the_round(sample: SampleObservation, _world: World) -> Answer:
    """Exactly one routine stands for the round — the re-extraction REPLACED rather than forked.

    Two is the fork: the corrected round filed its routine beside the one the teach taught
    rather than over it, leaving the user two routines for one job.  Counted against the world's
    own five, so what is counted is what this round is responsible for however it ended up
    named, and the count is a list length rather than an inference."""
    for_the_round = sorted({one.name for one in sample.routines} - _WORLD_ROUTINES)
    return for_the_round == [_ROUND_ROUTINE], f"the round's routines are {for_the_round}"


def _the_routine_looks_for_the_corrected_target(sample: SampleObservation, _world: World) -> Answer:
    """The routine now looks for what the correction NAMED — the step that changed, present in
    the program rather than only in the reply.

    A routine that re-ran correctly and then distilled the instruction it was ORIGINALLY given
    fetches the wrong thing every cycle, for ever, and nothing else here would see it."""
    routine = _the_rounds_routine(sample)
    if routine is None:
        return False, f"no routine stands under {_ROUND_ROUTINE!r}"
    values = " ".join(routine.demonstrated_values).lower()
    return _CORRECTED_TARGET in values, f"the routine names {routine.demonstrated_values}"


def _the_routine_still_names_its_page(sample: SampleObservation, _world: World) -> Answer:
    """The routine still points at the page it was taught on — a step the correction said
    nothing about, and the first way a delta loses the parts it did not touch.

    Matched with the scheme stripped, because a page named back in the user's own scheme-less
    form is plainly the same page."""
    routine = _the_rounds_routine(sample)
    if routine is None:
        return False, f"no routine stands under {_ROUND_ROUTINE!r}"
    named = any(_said_back(_SURVIVOR.prior.url, value) for value in routine.demonstrated_values)
    return named, f"the routine names no page: {routine.demonstrated_values}"


def _the_routine_still_keeps_what_it_finds(sample: SampleObservation, _world: World) -> Answer:
    """The routine still names somewhere to ACT — the other step the correction never mentioned.

    Read off the ATTACHMENT MARK, which distillation sets on any leaf whose demonstrated value
    named one of Penny's own collections, so it is true of a write, of a log append and of a
    plugin verb nobody has heard of, and false of a routine that only browses.  Its own claim
    rather than the shared one, which reads every routine in the registry and would be answered
    here mostly by the world's seeded five."""
    routine = _the_rounds_routine(sample)
    if routine is None:
        return False, f"no routine stands under {_ROUND_ROUTINE!r}"
    return routine.names_a_destination, "the routine keeps nothing"


def _the_round_container_is_still_inert(sample: SampleObservation, _world: World) -> Answer:
    """The round's container carries no job — a correction TEACHES, it does not instantiate.

    The end-state form of "she configured nothing", and the negative direction of the two
    stand-up edges' terms claims: a correction that set the round running has answered an offer
    the user never accepted.  A container that is gone entirely fails it too, since a retired
    container is not an inert one."""
    row = next((one for one in sample.mechanisms if one.name == _ROUND_CONTAINER), None)
    if row is None:
        return False, f"{_ROUND_CONTAINER!r} is no longer in the registry at all"
    terms = row.schedule is not None or row.notifies or row.expires
    return not terms, (
        f"{_ROUND_CONTAINER!r} carries a job: schedule {row.schedule!r}, "
        f"notifies {row.notifies}, expires {row.expires}"
    )


# What this case measures.  ``ROUTINE_SHAPE`` is IN, unlike every other ported transition case:
# this is the one edge whose turn RE-EXTRACTS, so the shape it reads is the round's own new
# program against the world's constant five rather than the fixture alone.
#
# ``ROUTINE_NAME`` is OUT, and it is the near miss.  Since #1902 the re-extraction is keyed by
# the name the round's framing PINNED — which is what this case's own one-routine claim asserts
# — so on a correct cohort the feature reads the five seeded names plus one pinned one, the same
# value on every sample: a reading of the FIXTURE, which is why the four sibling cases exclude
# it.  And the one divergence it could report is a FORK, a different end state, while the
# feature is declared cosmetic — so the finding would land in the wrong column.  The fork is
# caught by ``_one_routine_stands_for_the_round``, where it is a miss rather than a spread.
#
# ``JOB_TERMS`` is ABSENT for the other reason: a correct correction stands nothing up, so on a
# correct cohort it reads its absent value on every sample and the report would mark it blind in
# red for behaving exactly as this case requires.  A sample that DID configure the round is
# caught by the inert claim.
_MEASURED = (TOOL_SEQUENCE, ROUTINE_SHAPE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_learn_to_learn_re_runs_the_round_against_the_corrected_target(
    chat_eval: ChatEval, model: str
) -> None:
    """learn → learn: the round was taught on one line of the trail status and the user meant
    the one below it.  The corrected round reads the page again, keeps what the corrected line
    says, and the routine it leaves behind looks for that line from now on — in the container it
    already had, with nothing set running."""
    cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        seed=seed_corrected_round(_SURVIVOR),
        seed_skills=[*(journey.round.skill for journey in _JOURNEYS), _SURVIVOR.skill],
        prepare=_probe_correction_world(_SURVIVOR),
        world=_CORRECTED_TRAIL,
        ask=_SURVIVOR.correction,
        also_phrased=_CORRECTION_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.LEARN)

    # STORE — what the corrected round kept, and where.
    cohort.assert_something_from_each_page_was_written()
    cohort.assert_the_write_landed_in_the_round_container()
    cohort.assert_nothing_excluded_was_stored()
    cohort.assert_no_mechanism_was_created()
    cohort.assert_only_the_rounds_own_mechanism_changed(_ROUND_CONTAINER)
    cohort.claim(
        "state: the round's container carries no job",
        _the_round_container_is_still_inert,
        SpecCategory.STORE,
    )
    # STORE — and what it left in the registry.
    cohort.claim(
        "state: one routine stands for the round",
        _one_routine_stands_for_the_round,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the routine looks for what the correction named",
        _the_routine_looks_for_the_corrected_target,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the routine still points at the page it was taught on",
        _the_routine_still_names_its_page,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the routine still keeps what it finds",
        _the_routine_still_keeps_what_it_finds,
        SpecCategory.STORE,
    )

    # PROVENANCE — the half the source case had none of.  A corrected round writes, so the store
    # claim is live on every sample rather than answering over an empty set.
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)
