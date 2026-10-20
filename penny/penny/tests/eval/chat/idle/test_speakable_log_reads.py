"""Looking back at her own logs: two cases (#2008, tranche 3, folding #2001).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**#2008 states the behaviour as one sentence and lists two candidates; the derivation came out
at TWO cases, and the reason is what the wrong answer would have been.**  Applying the design's
test — *would a correct sample for one be wrong for the other?* — to the four legacy candidates:

* ``speak-logread-penny-messages-recall`` — **survivor.**  Asked what she SAID, with the
  answer out of the conversation window and a topic sitting right there to browse instead.
* ``speak-logread-collector-runs`` — **survivor, and a different sentence.**  Asked WHY a
  job is in trouble, where the wrong answer is not a browse but the ambient header.
* ``speak-logread-browse-results`` — not a case here: the same sentence as the first, on a
  weaker world.
* ``speak-logread-user-messages-act`` — not a case here: a different behaviour.  Read-then-ACT
  is *save* with a log as the source, ported in tranche 1 as ``memory-look-up-and-save``.

The two survivors split because the ALTERNATIVE differs, and the alternative is what each case
exists to rule out.  For the message recall it is BROWSING — the ask names a topic the open web
answers, so the failure is going to look it up instead of looking back.  For the run record it
is the SELF-STATE HEADER, which renders every job, its cadence and each recent run's outcome, so
"how have they been doing" is fully answered at zero calls and a model that reasons rightly
skips the read (measured, #1990: four of five samples answered from the header, correctly, and a
case that scored the read marked all four wrong).  A sample that browsed is wrong for the first
and irrelevant to the second; a sample that answered from ambient state is wrong for the second
and impossible in the first.  Two sentences, two cases — and splitting a behaviour in two is a
smaller mistake than collapsing two into one.

``speak-logread-browse-results`` is not kept because its world cannot produce the failure its
own sentence is about: "what have you been looking up lately?" names no topic, so there is
nothing for a browsing sample to browse, and the negative direction is unreachable.

**#2001's fixture re-aiming is folded into the run-record survivor**, and it is the technique
the whole case rests on: **make the right answer unreachable by any route except the one under
test.**  A cycle's OUTCOME is ambient; the REASON behind it never is, and a failed cycle writes
nothing, so no collection holds it either.  The failing cycle's stamped reason carries an
invented word that exists in this world in exactly ONE place — the run record — so a reply
carrying it can only have read it.  #2001's world-integrity probe rides along
(``assert_the_run_record_is_the_only_route``), asserted per sample AND pinned deterministically
in ``test_eval_harness.py``, because both halves of that world were silently wrong once and it
went unnoticed through three full suites: a cycle whose trace said it wrote while its entry
never landed left the collection empty and dropped the writes clause production renders, so the
case measured a world THINNER than the real one, against a "read the collections instead"
alternative that was a dead end rather than a wrong answer.

**The conversation confound (the message case).**  The chat agent injects the last
``MESSAGE_CONTEXT_LIMIT`` turns as in-context history, so a salient message INSIDE that window
is answered from context and never needs a read — a false gap.  So the case seeds the salient
turn FIRST, then ``_FILLER_PAIRS`` neutral turns after it, pushing it out of BOTH the context
window and the per-direction top-N fetch.  Nothing else can leak it in: the chat prompt injects
no speculative recalled-content block (#1555/#1583).

**What is NOT claimed, in either case: which route she took, or what she did not do.**  Whether
a browse happened, and whether a read did, are ROUTES — the design's own worked example of a
check that fits none of the three categories — so they are measured in the tool sequence, where
a sample that went looking stands apart from every other one.  Their END-STATE form is the
world's own foreclosure: the answer exists in exactly one place, so a sample that browsed
instead cannot state it, and the reply claim reads that directly.  Nor is it claimed that the
turn wrote or created nothing (``docs/principles.md`` §4.3): what is claimed instead is that
what the world already held is still there, where it held anything.

**The world is one the user built (#1911/migration 0108: nothing is pre-seeded; #2455 naive
baseline: no skill or routine behind a job).**  The two jobs the run-record case reads about are
STANDING JOBS — the same naive-shape job ``test_standing_collection.py`` owns (a second copy
here would be a second world, free to drift) — with real completed cycles behind them, seeded
through ``penny.tests.eval.end_to_end.world.seed_job_cycle``.

**No claim reads where the machine landed, or a run's outcome (#2496, naive baseline).**  There
is no conversation machine left to land on this branch, so every landing claim is dropped
rather than ported — consistent with #2496's ruling on the naive branch generally.

``run_list`` is collector-internal rather than user-dispatchable — its argument is a
collection source and "what did your last run do" is not a phrasing a user reaches for — so
neither case dispatches it.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  The jobs' hosts are real (MobyGames, AllTrails, The
Astorian); the topics and pages behind them are made up, because the repo is public.  The
three one-place answer tokens (``silverleaf``, ``thistledown``, ``driftglass``) stay
deliberately invented: each must appear in exactly one place in its world, which is the
mechanism that forces a read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory.objects import render_run_calls
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory

# The standing-job world the run-record case reads about, and the cycle-seeding machinery every
# job-history case shares — read from where they are defined rather than restated, so a job
# seeded here is the same production shape a real apply turn leaves behind and a collector run
# seeded here has the same envelope every other seeded run does.
from penny.tests.eval.chat.idle.test_standing_collection import StandingJob, seed_standing_jobs
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Preparer,
    Seeder,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _seeded_response, _wire_tool_call, seed_job_cycle
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.dispatch_world import assert_surface_carries, collection_rows
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for both cases in this module.
_FAMILY = "speakable-logread"

# The tools a cycle's own program makes — named once, since the seeder builds its ledger from
# them and a fixture that stopped rendering a runnable program would claim a job that could
# never have run.
_BROWSE = "browse"
_WRITE = "collection_write"

# The cross-collector index the run-record case's answer lives in.
_COLLECTOR_RUNS = PennyConstants.MEMORY_COLLECTOR_RUNS_LOG

# The general store-addressing READ verbs a look-back can be answered through.  Asserted on the
# surface before the turn, for the reason every dispatch story asserts its own: a case about
# reading that is answered by a model never offered a read verb scores a model miss for a
# HARNESS failure.  It is a premise about the SURFACE and not a route the case expects — which
# verb she reaches for is hers, and the claims below read no tool name at all.
_READ_VERBS = ("log_read", "find")

# Wide enough that a world seeding a conversation window is read WHOLE: the probe's answer has
# to be "none of them", not "none of the first few".
_VECTOR_PROBE_LIMIT = 500

# Directions/authors for seeding the conversation logs.
_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY


# ── The case shape ───────────────────────────────────────────────────────────


class _LogReadCase(NamedTuple):
    """One ask about what she did or said, in five wordings, against one seeded world.

    ``ask`` and ``also_phrased`` are five wordings of ONE message — the cohort's arms.  What
    varies is only how a person says it; the world, the answer and the end state expected of
    the turn are constant, which is what makes the fifteen samples one number and what lets a
    claim name a value at all.

    ``answer`` is the SMALLEST datum that is unique in the world and reachable by exactly one
    route.  It is what ``World.answers`` states and what the provenance claim traces, so the
    two cannot drift into naming different tokens.
    """

    case_id: str
    area: Area
    behaviour: str
    seed: Seeder
    premise: Callable[[Database], None]
    answer: str
    ask: str
    also_phrased: tuple[str, ...]

    @property
    def ground(self) -> World:
        """The world every arm of this case is answered against.

        ``pages`` is EMPTY and that is the point rather than an omission: the answer lives in a
        log and nowhere else, so a browse comes back with nothing and going looking is a dead
        end instead of a second source.  ``stores`` is empty for a different reason: the
        run-record world seeds two collections, but they are JOBS stood up through the store's
        own chokepoints rather than plain holdings, and a declared store is laid down as
        entries alone — declaring them would seed what their cycles wrote a second time.

        ``keeps`` and ``excludes`` are empty because the turn
        answers a question and is asked to write nothing, so either would state a contract the
        ask never made.  ``answers`` is the one token the ask requests."""
        return World(name=self.case_id, pages=(), keeps=(), excludes=(), answers=(self.answer,))


def _probe(case: _LogReadCase) -> Preparer:
    """The prepare hook: the chat surface these cases need, the case's own premise, and the one
    half that only exists once the runner has embedded the seeds."""

    def probe(penny: Penny) -> None:
        assert_surface_carries(penny, case.case_id, _READ_VERBS)
        case.premise(penny.db)
        _assert_the_seeded_turns_are_recallable(penny.db)

    return probe


def _assert_the_seeded_turns_are_recallable(db: Database) -> None:
    """Every seeded message carries a vector — asserted BEFORE the turn is driven.

    Separate from the case's own premise because it is the one claim the plain pin cannot
    drive: the seeder writes the messages and the RUNNER embeds them, so on a database nobody
    ran it would fail for the fixture rather than for the world.  A seeded message with no
    embedding cannot rank in any similarity read, so a case whose answer lives in that message
    is UNREACHABLE — the model aims its read correctly, finds nothing, and the case scores 0.00
    with nothing in the transcript to say why.  That is exactly what happened here once.

    Read through the production backfill's OWN query, so "did the seeds get vectors" is the
    same question the runner answered rather than a second opinion about it."""
    unvectored = db.messages.messages_without_embeddings(limit=_VECTOR_PROBE_LIMIT)
    assert not unvectored, (
        "every seeded message must carry an embedding before the turn runs — "
        f"{len(unvectored)} without one, first: {unvectored[0].content!r}"
    )


# ── Conversation seeding (out-of-window) ─────────────────────────────────────
# ≥ MESSAGE_CONTEXT_LIMIT (20), so the salient turn is pushed out of BOTH the context window
# AND the per-direction top-20 fetch, with margin.
_FILLER_PAIRS = 24

# Neutral, topic-free chit-chat — it names no recommendation, so it can answer the case from
# context under no wording.
_FILLER_USER = (
    "morning!",
    "how's your day going?",
    "thanks for that",
    "cool, makes sense",
    "what's the weather looking like?",
    "ok noted",
    "sounds good",
    "haha nice",
    "appreciate it",
    "got it, thanks",
    "all good here",
    "talk to you later",
)
_FILLER_PENNY = (
    "morning! good to hear from you",
    "going well, thanks for asking",
    "anytime, happy to help",
    "glad that makes sense",
    "clear skies today, mild and calm",
    "sounds good to me",
    "you got it",
    "hah, right?",
    "of course",
    "no problem at all",
    "nice, catch you later",
    "take care!",
)


def _seed_out_of_window(direction: str, salient: str) -> Seeder:
    """Seed the salient turn FIRST (oldest), then ``_FILLER_PAIRS`` neutral user/Penny turns
    after it — so the salient turn is genuinely out of context and retrieving it requires a
    read.  Penny filler carries the real recipient so it counts as an autonomous outgoing turn
    in the context builder, exercising the same push-out production would."""

    def seed(db: Database) -> None:
        if direction == _INCOMING:
            db.messages.log_message(_INCOMING, TEST_SENDER, salient)
        else:
            db.messages.log_message(_OUTGOING, _PENNY, salient, recipient=TEST_SENDER)
        for index in range(_FILLER_PAIRS):
            db.messages.log_message(_INCOMING, TEST_SENDER, _FILLER_USER[index % len(_FILLER_USER)])
            db.messages.log_message(
                _OUTGOING, _PENNY, _FILLER_PENNY[index % len(_FILLER_PENNY)], recipient=TEST_SENDER
            )

    return seed


# The recommendation she made a long time ago, and the invented word inside it the reply is read
# for.  ``silverleaf`` exists in this world in exactly ONE place — the seeded outgoing turn — so
# a reply carrying it can only have gone and read it.
_SUGGESTION_MESSAGE = (
    "for your moss terrarium, I'd really go with silverleaf moss — it handles low light and "
    "stays compact"
)
_SUGGESTION_TOKEN = "silverleaf"


def _seed_the_suggestion(db: Database) -> None:
    _seed_out_of_window(_OUTGOING, _SUGGESTION_MESSAGE)(db)


def assert_the_conversation_is_the_only_route(db: Database) -> None:
    """The message case's premise: the answer is reachable ONLY by reading back.

    The recommendation's own word is in NEITHER the store NOR the ambient header, so nothing
    but a read of her own messages can put it in a reply — and the salient turn itself sits
    behind ``_FILLER_PAIRS`` later turns, out of both the context window and the per-direction
    fetch, so the conversation the turn is handed cannot answer it either."""
    _assert_unreachable_except_by_reading(db, _SUGGESTION_TOKEN)


def _assert_unreachable_except_by_reading(db: Database, token: str) -> None:
    """The answer token is in no COLLECTION and not in the ambient header.

    Both halves are what make the reply claim a claim about READING rather than about the
    world: a token the store holds is one a lookup can reach, and a token the header renders is
    one that needs no call at all — and if either were true, a correct sample would rightly
    skip the read the case is about.

    Collections only, and deliberately: the four migration-0026 system LOG markers are where
    each case's answer is meant to live, so counting them would make the assertion unsatisfiable
    by construction.  The rows are read through the shape read the dispatch worlds share, so
    what counts as a collection is the runtime's definition rather than a list of names."""
    for row in collection_rows(db):
        held = " ".join(
            f"{key} {content}" for key, content in collection_entries(db, row.name).items()
        )
        assert token not in fold_typography(held), (
            f"{row.name} carries {token!r} — the answer must not be reachable by reading a "
            "collection, or a sample that never looked back can state it"
        )
    header = SelfStateHeader(db, TEST_SENDER).render()
    assert token not in fold_typography(header), (
        f"the self-state header carries {token!r} — the read the case is about would be "
        f"redundant and skipping it would be correct:\n{header}"
    )


# ── The run-record world: two standing jobs and the cycles behind them ───────
#
# The jobs are the standing-collection story's own naive-shape world (#2455 naive baseline) —
# two rows stood up through the store's own chokepoints, so the pair is two real jobs rather
# than two rows shaped like jobs.  What this case adds is their HISTORY: completed cycles,
# which is what the collector-runs index renders — including one that FAILED, the cycle the
# case is actually about.

_PATCH_NOTES_JOB = StandingJob(
    container="into-the-breach-patch-notes",
    url="https://subsetgames.com/patch-notes",
    watched_for="balance changes",
    description="Notable new Into the Breach patch notes worth knowing about.",
    schedule="FREQ=HOURLY",
    notify=True,
)
_TRAIL_JOB = StandingJob(
    container="tillamook-head-trail",
    url="https://www.alltrails.com/explore/trail/oregon/tillamook-head-trail",
    watched_for="trail conditions",
    description="Current conditions on the Tillamook Head hiking trail.",
    schedule="FREQ=HOURLY",
    notify=False,
)
_SEEDED_JOBS = (_PATCH_NOTES_JOB, _TRAIL_JOB)

# The failed cycle's stamped reason, and the invented word inside it the reply is read for.
# ``thistledown`` exists in this world in exactly ONE place: this string, stamped on the run and
# rendered by the run record.  The self-state header renders that cycle's OUTCOME and never the
# reason behind it, and a failed cycle writes nothing, so no collection holds it either.
_TRAIL_FAILURE_REASON = (
    "Couldn't read the page — it bounced to a notice about the thistledown survey closure."
)
_FAILURE_TOKEN = "thistledown"


class _SeededCycle(NamedTuple):
    """One completed cycle in a job's history: which job ran, what it came back with (a key and
    content, or nothing on a quiet or failed cycle), and how the run recorded itself."""

    job: StandingJob
    name: str
    outcome: RunOutcome
    summary: str
    wrote: tuple[str, str] | None = None

    @property
    def run_id(self) -> str:
        """The seeded id every row of this cycle carries — minted once, so the ledger row, the
        entry it wrote and any probe naming the run all say the same thing."""
        return seeded_run_id(self.name)

    @property
    def closed(self) -> bool:
        """Whether the cycle reached ``done()``.  A FAILED one did not — never reaching a
        healthy end is what failing means (#1936) — and that is also what makes its stamped
        reason READ: a cycle that closed cleanly renders its outcome instead, so a failure
        seeded as closed would hide the very sentence this case is about."""
        return self.outcome is not RunOutcome.FAILED


_SEEDED_CYCLES = (
    _SeededCycle(
        _PATCH_NOTES_JOB,
        "patch-notes-cycle-1",
        RunOutcome.WORKED,
        "Recorded the 2.3 balance patch.",
        wrote=("Patch 2.3", "Patch 2.3 — ember mage rebalance."),
    ),
    _SeededCycle(
        _TRAIL_JOB,
        "trail-cycle-1",
        RunOutcome.WORKED,
        "Logged the Tillamook Head trail status.",
        wrote=("Tillamook Head trail status", "Tillamook Head — muddy after rain."),
    ),
    _SeededCycle(
        _PATCH_NOTES_JOB,
        "patch-notes-cycle-2",
        RunOutcome.NO_WORK,
        "No new patch notes this cycle.",
    ),
    _SeededCycle(_TRAIL_JOB, "trail-cycle-2", RunOutcome.FAILED, _TRAIL_FAILURE_REASON),
)


def _cycle_calls(cycle: _SeededCycle) -> list[dict]:
    """One cycle's calls, as the job's own program makes them: read the page, write what it
    found (when it found something), close.  A quiet cycle just reads and closes; a failed cycle
    reads and stops there, because that is what its outcome means.

    Built with ``_wire_tool_call``, the same wire shape ``seed_known_routine``'s own turns use
    — the run's RESULT text plays no part here (``seed_job_cycle`` logs the calls alone, with
    no accumulated ``messages`` for a result to sit in), only the calls a reader's ``run_get``
    needs to see."""
    run_id = cycle.run_id
    calls = [
        _wire_tool_call(
            f"{run_id}-0",
            _BROWSE,
            {"queries": [cycle.job.url], "extract": cycle.job.watched_for},
        )
    ]
    if cycle.wrote is not None:
        key, content = cycle.wrote
        calls.append(
            _wire_tool_call(
                f"{run_id}-{len(calls)}",
                _WRITE,
                {"collection": cycle.job.container, "entries": [{"key": key, "content": content}]},
            )
        )
    if cycle.closed:
        calls.append(_wire_tool_call(f"{run_id}-{len(calls)}", PennyConstants.DONE_TOOL_NAME, {}))
    return calls


def _seed_run(db: Database, cycle: _SeededCycle) -> None:
    """Seed one completed collector cycle through the shared cycle-history helper
    (``seed_job_cycle``): the ``promptlog`` row that IS its ``collector-runs`` content, the
    outcome + reason stamped on it, and — when the cycle wrote — the entry it left behind,
    attributed to this run.

    That last part is what makes the world COHERE, and it was missing once (#2001).  The
    cycle's trace says it wrote and its reason says what it recorded; if the entry never
    lands, the collection the model is pointed at is empty, no ``last_written_by_run_id`` is
    stamped, and the self-state header silently drops the writes clause production renders
    (#1641).  The case would then measure a world THINNER than the real one, against a "read
    the collections instead" alternative that is a dead end rather than a wrong answer."""
    seed_job_cycle(
        db,
        container=cycle.job.container,
        run_name=cycle.name,
        calls=_cycle_calls(cycle),
        outcome=cycle.outcome,
        summary=cycle.summary,
        wrote=cycle.wrote,
    )


def _seed_collector_activity(db: Database) -> None:
    """Two standing jobs, then the completed cycles behind them — the cross-collector history
    the ``collector-runs`` index renders."""
    seed_standing_jobs(*_SEEDED_JOBS)(db)
    for cycle in _SEEDED_CYCLES:
        _seed_run(db, cycle)


def assert_the_run_record_is_the_only_route(db: Database) -> None:
    """The run-record case's premise, in both halves (#2001).

    Every cycle that says it wrote really did, attributed to the run that made it and ambient in
    the header — so the "read the collections instead" alternative is a real one the case is
    ruling out rather than a dead end it never had to.  And the failing cycle's reason renders in
    the run record while appearing in NO collection and NOT in the header — so a reply carrying
    it can only have read it.

    Both halves were silently wrong once and went unnoticed through three full suites, which is
    why they are asserted per sample as well as pinned deterministically."""
    header = SelfStateHeader(db, TEST_SENDER).render()
    for cycle in _SEEDED_CYCLES:
        if cycle.wrote is None:
            continue
        key, content = cycle.wrote
        assert collection_entries(db, cycle.job.container).get(key) == content, (
            f"{cycle.name}: its trace says it wrote {key!r}, the collection does not hold it"
        )
        assert db.memories.writes_by_run([cycle.run_id]).get(cycle.run_id), (
            f"{cycle.name}: the write is not attributed to the cycle that made it, so the "
            "header drops the writes clause production renders"
        )
        assert key in header, f"{cycle.name}: {key!r} is not ambient, but production makes it so"
    record = " ".join(entry.content for entry in require_memory(db, _COLLECTOR_RUNS).read_all())
    assert _FAILURE_TOKEN in fold_typography(record), (
        "the failing cycle's reason must render in the run record — it is the whole answer the "
        "case asks for"
    )
    _assert_unreachable_except_by_reading(db, _FAILURE_TOKEN)


# ── The claims, as pure functions over one sample ────────────────────────────
#
# Neither case reads a tool NAME, and neither claims that a call did or did not happen: whether
# a call was made is a ROUTE — the design's own worked example of a check that fits none of the
# three categories — so it is measured in the tool sequence, and its end-state form is the
# world's own foreclosure, which the answer claim reads directly: the answer token exists in
# one place only, so a reply carrying it cites that place.
#
# The one local claim stays LOCAL rather than graduating into ``assertions.py``: a claim
# graduates at the second CUSTOMER, and only the run-record world holds anything to preserve.


def _both_jobs_are_as_they_were(sample: SampleObservation, _world: World) -> Answer:
    """Both running jobs are still configured as they were seeded, and each still holds what
    its cycles wrote — PRESERVATION of everything the world held before the turn.

    Read as the row's END STATE (``moved_this_run``), never as which fields a call named, so a
    turn that restated a value is not counted as having changed it.  A violating sample is
    nameable: one that "fixes" the failing job while reporting on it — re-timing it, retiring
    it, re-pointing it — when the ask was only why it is in trouble."""
    drifted: list[str] = []
    for job in _SEEDED_JOBS:
        row = next((one for one in sample.mechanisms if one.name == job.container), None)
        if row is None or row.moved_this_run:
            drifted.append(f"{job.container}: {row.moved_this_run if row else 'gone'}")
    held = {(entry.collection, entry.key) for entry in sample.held}
    lost = [
        f"{cycle.job.container}:{cycle.wrote[0]}"
        for cycle in _SEEDED_CYCLES
        if cycle.wrote is not None and (cycle.job.container, cycle.wrote[0]) not in held
    ]
    return not drifted and not lost, f"moved {drifted}, no longer holds {lost}"


async def _drive(chat_eval: ChatEval, model: str, case: _LogReadCase) -> Cohort:
    """Drive one log-read case: the history its own world seeds, the routines that world needs
    in the registry, and the premise re-asserted before the turn."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        area=case.area,
        model=model,
        seed=case.seed,
        prepare=_probe(case),
        world=case.ground,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


# What both cases measure.  ``TOOL_SEQUENCE`` is where a sample that went and looked the
# topic up stands apart from one that looked back — the divergence these cases are named
# against — and it is measured rather than asserted, because WHICH read verb answers a
# look-back is hers to choose and the claims below name none.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ═══ what she said ═══════════════════════════════════════════════════════════

_PENNY_MESSAGES = _LogReadCase(
    case_id="speak-logread-penny-messages-recall",
    area=Area.MEMORY,
    behaviour=(
        "In the chat agent, when the user asks what she told them and the answer is out of the "
        "conversation window, Penny states what her own earlier message actually said."
    ),
    seed=_seed_the_suggestion,
    premise=assert_the_conversation_is_the_only_route,
    answer=_SUGGESTION_TOKEN,
    ask=(
        "dig back through our old messages — what exactly did you tell me to use for my "
        "moss terrarium?"
    ),
    also_phrased=(
        "look back through our old messages — what did you say i should use for my moss terrarium?",
        "you recommended something for my moss terrarium a while back — dig it up, what was it?",
        "go back over what you've told me: which one did you say to use for my moss terrarium?",
        "search back through our conversation — what did you tell me to put in my moss terrarium?",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_what_she_said_comes_back_out_of_her_own_messages(
    chat_eval: ChatEval, model: str
) -> None:
    """A recommendation made long enough ago to be out of the window, and a topic the open web
    would happily answer instead.

    The word the reply is read for is invented, so it exists only in the turn she sent — which
    is what turns "did she look back" from an assumption into something the answer itself
    settles."""
    cohort = await _drive(chat_eval, model, _PENNY_MESSAGES)
    # STORE — the reply carries the word her own message holds, and only that message holds.
    # No preservation claim: the world holds no collection, only the conversation, and the
    # conversation is the log the answer is read out of.
    cohort.assert_the_reply_answers_the_ask()

    # PROVENANCE — the reply half.  The STORE half is absent, and this is a report rather than an
    # omission: the ask keeps nothing, and whether a sample wrote something anyway is measured in
    # ENTRIES_STORED.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ what her jobs did ═══════════════════════════════════════════════════════

_COLLECTOR_RUNS_CASE = _LogReadCase(
    case_id="speak-logread-collector-runs",
    area=Area.STANDING_JOBS,
    behaviour=(
        "In the chat agent, when the user asks how her background jobs are doing and why any of "
        "them is in trouble, Penny states the reason the failing cycle's run record gives, with "
        "both jobs still running as they were."
    ),
    seed=_seed_collector_activity,
    premise=assert_the_run_record_is_the_only_route,
    answer=_FAILURE_TOKEN,
    ask=(
        "how have your background jobs been doing lately? if any of them is having trouble i "
        "want to know why"
    ),
    also_phrased=(
        "how are your background jobs doing lately? and if one of them is in trouble, tell me why",
        "give me a rundown on your background jobs — and if any of them is struggling, what's "
        "behind it?",
        "what have your background jobs been up to lately? if something is going wrong i want "
        "the reason",
        "check in on your background jobs for me — and if one of them isn't working, why not?",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_why_a_job_is_in_trouble_comes_out_of_the_run_record(
    chat_eval: ChatEval, model: str
) -> None:
    """Two running jobs with real cycles behind them, one of which failed.

    The ask reaches PAST the self-state header deliberately.  "How have they been doing" is
    answered by the header alone — every mechanism, every cadence, every run's outcome — so a
    case that stopped there scored the read as effort rather than as routing and marked a
    correct header answer wrong (#1990).  What the header cannot carry is WHY a cycle failed,
    and the second clause is what asks for it: the opening stays verbatim so the case is still
    the same introspection ask, and the clause makes the answer live somewhere only a read can
    reach."""
    cohort = await _drive(chat_eval, model, _COLLECTOR_RUNS_CASE)
    # STORE — the reply carries the word only the run record holds, and the two jobs the world
    # started with are still there as they were.
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: both jobs are still configured as they were, holding what they gathered",
        _both_jobs_are_as_they_were,
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply half.  The STORE half is absent for the message case's reason: the
    # ask keeps nothing.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ what went wrong with an earlier turn ════════════════════════════════════
#
# #2355's own case 1 — sibling of ``speak-logread-collector-runs`` above, and a DIFFERENT
# sentence for the same reason that pair split: the alternative each rules out differs.  A
# collector cycle's reason is the one fact the self-state header CANNOT carry (it renders
# only the outcome), which is what forces a read; a CHAT turn that fell back to the
# honest-account machinery (#2355 part D) stamps a run exactly the same way, but its reason
# renders straight into the header's own activity line (``SelfStateHeader._chat_failure_line``)
# — so the header alone already answers "what went wrong", at zero calls, and `run_get` on
# the typed `run <id>` it carries is the one-call drill-down to the tool call and the model
# call that did not finish (never forced; measured in TOOL_SEQUENCE like every other route
# here).  The world is a single SEEDED past turn — #2355 part A's own link
# (`parent_run_id`/`parent_tool_call_id`) is exercised too, so a sample that does look
# further finds the SAME record `run_get` renders in production, not a thinner stand-in.

_FAILED_PAGE = "https://www.dailyastorian.com/"
_FAILED_EXTRACT_ASK = "the day's headlines"
# The invented word names SPECIFIC CONTENT the extraction got as far as before timing out
# — never a detail of the SITE or the ASK (the user's own message already names "the
# astorian" and "headlines", so a reply that only restates THOSE answers correctly without
# ever having read the record at all — measured: a first draft of this reason put the
# invented word in the site's own name, and 10 of 15 samples gave a correct, honest account
# of the cause while never saying it, scoring as a miss for a claim that was testing the
# wrong thing). A model giving the full honest answer ("it got partway through — as far as
# X — before stopping") has to name X, and X exists in exactly one place: this reason.
_FAILED_TURN_REASON = (
    "the page was read and stored; the extraction got as far as a headline about a "
    "driftglass festival before it timed out, so the rest of the headlines never came through"
)
_FAILED_TURN_TOKEN = "driftglass"


def _seed_the_failed_turn(db: Database) -> None:
    """One past chat turn (#2355 parts A+D): a browse call that read and stored the page,
    the nested extraction draw that never finished (linked by `parent_run_id`/
    `parent_tool_call_id`, exactly as production links one), and the run stamped FAILED
    with the reason the header's activity line — and `run_get` — both carry."""
    run_id = seeded_run_id("failed-browse-turn")
    browse_args = {"queries": [_FAILED_PAGE], "extract": _FAILED_EXTRACT_ASK}
    call_id = f"{run_id}-0"
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=_seeded_response(tool_calls=[_wire_tool_call(call_id, _BROWSE, browse_args)]),
        agent_name=PennyConstants.CHAT_AGENT_NAME,
        run_id=run_id,
    )
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=None,
        agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        run_id=f"{run_id}-draw",
        parent_run_id=run_id,
        parent_tool_call_id=call_id,
        call_outcome="cancelled",
        duration_ms=300_000,
    )
    db.messages.set_run_outcome(run_id, RunOutcome.FAILED.value, _FAILED_TURN_REASON)


def assert_the_failed_turn_record_is_the_only_route(db: Database) -> None:
    """The failed-turn case's premise (#2355's case 1): the cause is reachable through the
    self-state header's own activity line (zero calls) or `run_get` on its typed anchor (one
    call), and the collections hold it nowhere — the "it got written down somewhere" route
    this case has none of.

    Also checks the fixture itself produced what it claims (#2001's lesson, applied here): the
    seeded run's `run_get` render carries BOTH the browse call and the nested unfinished draw,
    so a sample that DOES look further finds the same record production would show it rather
    than a thinner stand-in."""
    for row in collection_rows(db):
        held = " ".join(
            f"{key} {content}" for key, content in collection_entries(db, row.name).items()
        )
        assert _FAILED_TURN_TOKEN not in fold_typography(held), (
            f"{row.name} carries {_FAILED_TURN_TOKEN!r} — the cause must not be reachable by "
            "reading a collection"
        )
    header = SelfStateHeader(db, TEST_SENDER).render()
    assert _FAILED_TURN_TOKEN in fold_typography(header), (
        "the self-state header's activity line must carry the failed turn's own reason — "
        f"it did not:\n{header}"
    )
    run_id = seeded_run_id("failed-browse-turn")
    rendered = render_run_calls(
        db.messages.get_run_prompts(run_id), db.messages.get_child_draws(run_id)
    )
    assert _BROWSE in rendered, "the seeded run's own browse call must render in run_get"
    assert "cancelled" in rendered, (
        "the nested extraction draw must render under its step — the fixture claims an "
        f"unfinished model call and run_get does not show one:\n{rendered}"
    )


_FAILED_TURN_CASE = _LogReadCase(
    case_id="speak-logread-failed-chat-turn",
    area=Area.MEMORY,
    behaviour=(
        "In the chat agent, when the user asks what went wrong with an earlier turn whose "
        "page was read but whose extraction did not finish, Penny states the cause the run's "
        "record carries."
    ),
    seed=_seed_the_failed_turn,
    premise=assert_the_failed_turn_record_is_the_only_route,
    answer=_FAILED_TURN_TOKEN,
    ask=(
        "hey, earlier i asked you to check the astorian's headlines and it never "
        "worked out — what happened there?"
    ),
    also_phrased=(
        "what went wrong before when i asked you to pull the headlines off the astorian?",
        "that astorian headlines thing didn't work earlier — what happened?",
        "can you tell me why checking the astorian's headlines failed last time?",
        "i tried to get you to grab the astorian's headlines a bit ago and it "
        "didn't pan out — why not?",
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_what_went_wrong_comes_out_of_the_turns_own_record(
    chat_eval: ChatEval, model: str
) -> None:
    """A single seeded past turn: the page was read and stored, the extraction never came
    back, and the run is stamped the way #2355 part D stamps one.

    No forced read: the header's own activity line already states the cause, so a correct
    sample can answer at zero calls — exactly like the "how are your jobs doing" half of the
    collector-runs sibling, and unlike that case's SECOND half, which the header cannot
    answer and which this case does not need (there is no "job" here to ask a cause about,
    only a turn)."""
    cohort = await _drive(chat_eval, model, _FAILED_TURN_CASE)
    # STORE — the reply carries the word only the turn's own record holds.  No preservation
    # claim: the world holds no collection and no mechanism for the turn to have disturbed.
    cohort.assert_the_reply_answers_the_ask()

    # PROVENANCE — the reply half.  The STORE half is absent for the message case's reason:
    # the ask keeps nothing.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# Every ported case, in one place — so the deterministic probe in ``test_eval_harness.py`` can
# drive each one's seeder and premise without a GPU.
LOG_READ_CASES = (_PENNY_MESSAGES, _COLLECTOR_RUNS_CASE, _FAILED_TURN_CASE)
