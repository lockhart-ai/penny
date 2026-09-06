"""Looking back at her own logs: two cases (#2008, tranche 3, folding #2001).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**#2008 states the behaviour as one sentence and lists two candidates; the derivation came out
at TWO cases, and the reason is what the wrong answer would have been.**  Applying the design's
test — *would a correct sample for one be wrong for the other?* — to the four legacy candidates:

* ``speak-logread-penny-messages-recall`` — **survivor.**  Asked what she SAID, with the
  answer out of the conversation window and a topic sitting right there to browse instead.
* ``speak-logread-collector-runs`` — **survivor, and a different sentence.**  Asked WHY a
  job is in trouble, where the wrong answer is not a browse but the ambient header.
* ``speak-logread-browse-results`` — quarantined: the same sentence as the first, on a
  weaker world.
* ``speak-logread-user-messages-act`` — quarantined: a different behaviour.  Read-then-ACT
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

``speak-logread-browse-results`` is quarantined rather than kept because its world cannot
produce the failure its own sentence is about: "what have you been looking up lately?" names no
topic, so there is nothing for a browsing sample to browse, and the negative direction is
unreachable.  It comes back the day the browse log's own retrieval is what is being measured.

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

**What is NOT claimed, in either case: that she did not browse.**  Whether a browse happened is
a ROUTE — the design's own worked example of a check that fits none of the three categories —
so it is measured in the tool sequence, where a sample that went looking stands apart from every
other one.  Its END-STATE form is the world's own foreclosure: the answer exists in exactly one
place, so a sample that browsed instead cannot state it, and the reply claim reads that
directly.

**The world is one the user built (#1911/migration 0108: nothing is pre-seeded).**  The two jobs
the run-record case reads about are STANDING JOBS — a taught routine applied to two pages through
the production instantiation seam, with real completed cycles behind them (see
``test_standing_collection.py``, which owns that world; a second copy here would be a second
world, free to drift).  A hand-authored prose prompt would be a config defect the collector
cannot read (#1916's strict dialect), so a fixture seeded that way would claim jobs that could
never have run.

``read_run_calls`` is collector-internal rather than user-dispatchable — its argument is a
collection target and "what did your last run do" is not a phrasing a user reaches for — so
neither case dispatches it.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every topic, page and job is invented, because the repo
is public.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.constants import PennyConstants, RunOutcome
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.skills import DistillInput, SkillDraft
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory

# The seeded-ledger wire helpers the transition suite writes its own history with, and the
# standing-job world the run-record case reads about — read from where they are defined rather
# than restated, so a collector run seeded here has the same envelope every other seeded run
# does and the jobs are the ones a real apply turn leaves behind.
from penny.tests.eval.chat.idle.test_standing_collection import (
    WATCH_ROUTINE,
    StandingJob,
    seed_standing_jobs,
)
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Preparer,
    Seeder,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.dispatch_world import collection_rows
from penny.tests.eval.utils.transition_ledger import _seeded_response, _wire_tool_call
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

# Wide enough that a world seeding a conversation window is read WHOLE: the probe's answer has
# to be "none of them", not "none of the first few".
_VECTOR_PROBE_LIMIT = 500

# Directions/authors for seeding the conversation logs.
_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY
_COLLECTOR = PennyConstants.MessageAuthor.COLLECTOR


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
    behaviour: str
    seed: Seeder
    premise: Callable[[Database], None]
    answer: str
    ask: str
    also_phrased: tuple[str, ...]
    skills: tuple[SkillDraft, ...] = ()

    @property
    def ground(self) -> World:
        """The world every arm of this case is answered against.

        ``pages`` is EMPTY and that is the point rather than an omission: the answer lives in a
        log and nowhere else, so a browse comes back with nothing and going looking is a dead
        end instead of a second source.  ``keeps`` and ``excludes`` are empty because the turn
        answers a question and is asked to write nothing, so either would state a contract the
        ask never made.  ``answers`` is the one token the ask requests."""
        return World(name=self.case_id, pages=(), keeps=(), excludes=(), answers=(self.answer,))


def _probe(case: _LogReadCase) -> Preparer:
    """The prepare hook: the case's own premise, plus the one half that only exists once the
    runner has embedded the seeds."""

    def probe(penny: Penny) -> None:
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
# The jobs are the standing-collection story's own world — ONE taught routine, applied to two
# pages, so the pair is two real jobs rather than two rows shaped like jobs.  What this case
# adds is their HISTORY: completed cycles, which is what the collector-runs index renders —
# including one that FAILED, the cycle the case is actually about.

_PATCH_NOTES_JOB = StandingJob(
    routine=WATCH_ROUTINE,
    values={
        "page": "https://mistforge.example.com/patch-notes",
        "watched_for": "balance changes",
    },
    description="Notable new Mistforge Tactics patch notes worth knowing about.",
    schedule="FREQ=HOURLY",
    notify=True,
)
_TRAIL_JOB = StandingJob(
    routine=WATCH_ROUTINE,
    values={
        "page": "https://trails.example.com/verdant-hollow",
        "watched_for": "trail conditions",
    },
    description="Current conditions on the Verdant Hollow hiking trail.",
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
        "Logged today's trail status.",
        wrote=("today", "Verdant Hollow — muddy after rain."),
    ),
    _SeededCycle(
        _PATCH_NOTES_JOB,
        "patch-notes-cycle-2",
        RunOutcome.NO_WORK,
        "No new patch notes this cycle.",
    ),
    _SeededCycle(_TRAIL_JOB, "trail-cycle-2", RunOutcome.FAILED, _TRAIL_FAILURE_REASON),
)


def _browse_result(cycle: _SeededCycle) -> str:
    """What the cycle's page read came back with — the finding it went on to write, the failure
    that ended it, or nothing new on a quiet cycle.  A cycle whose trace says it read nothing
    while its reason says it recorded a patch is a world at odds with itself, and the model
    reads both."""
    page, watched_for = cycle.job.values["page"], cycle.job.values["watched_for"]
    if not cycle.closed:
        return f"You tried to open {page} ({_BROWSE} result)\n{cycle.summary}"
    found = cycle.wrote[1] if cycle.wrote is not None else "nothing new."
    return f"You opened {page} ({_BROWSE} result)\n{watched_for}: {found}"


def _read_step(cycle: _SeededCycle) -> DistillInput:
    """The cycle's page read.  Every cycle makes one — what it comes back with is what the
    cycle went on to do, or the reason it stopped."""
    return DistillInput(
        source_ordinal=1,
        tool=_BROWSE,
        arguments={
            "queries": [cycle.job.values["page"]],
            "extract": cycle.job.values["watched_for"],
        },
        result=_browse_result(cycle),
    )


def _write_step(cycle: _SeededCycle, key: str, content: str) -> DistillInput:
    """The entry the cycle kept, as the call that kept it."""
    return DistillInput(
        source_ordinal=2,
        tool=_WRITE,
        arguments={"memory": cycle.job.container, "entries": [{"key": key, "content": content}]},
        result=f"You saved an entry to {cycle.job.container}: ({_WRITE} result)\nWrote 1 entry.",
    )


def _close_step(ordinal: int) -> DistillInput:
    """The ``done()`` a healthy cycle ends on."""
    return DistillInput(
        source_ordinal=ordinal,
        tool=PennyConstants.DONE_TOOL_NAME,
        arguments={},
        result=f"You finished the cycle. ({PennyConstants.DONE_TOOL_NAME} result)\nDone.",
    )


def _cycle_steps(cycle: _SeededCycle) -> list[DistillInput]:
    """One cycle's calls, as the job's own program makes them: read the page, write what it
    found (when it found something), close.  A quiet cycle just reads and closes; a failed cycle
    reads and stops there, because that is what its outcome means."""
    steps = [_read_step(cycle)]
    if cycle.wrote is not None:
        steps.append(_write_step(cycle, *cycle.wrote))
    if not cycle.closed:
        return steps
    return [*steps, _close_step(len(steps) + 1)]


def _seed_run(db: Database, cycle: _SeededCycle) -> None:
    """Seed one completed collector cycle: the ``promptlog`` row that IS its ``collector-runs``
    content, the outcome + reason stamped on it, and — when the cycle wrote — the entry it left
    behind, attributed to this run.

    That last part is what makes the world COHERE, and it was missing (#2001).  The cycle's
    trace says it wrote and its reason says what it recorded; if the entry never lands, the
    collection the model is pointed at is empty, no ``last_written_by_run_id`` is stamped, and
    the self-state header silently drops the writes clause production renders (#1641).  The case
    would then measure a world THINNER than the real one, against a "read the collections
    instead" alternative that is a dead end rather than a wrong answer.

    The envelope is built by the seeded-ledger helpers the transition suite already writes
    history with, so a run seeded here and a run seeded there are one wire shape rather than two
    hand-built ones free to differ.  The id is a SEEDED one, so every reader of "what did the
    model do this sample" excludes it: a job's past cycles are history, and counting them as
    this turn's calls would report a quiet turn as a busy one."""
    steps = _cycle_steps(cycle)
    calls = [_wire_tool_call(f"{cycle.run_id}-{index}", step) for index, step in enumerate(steps)]
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=_seeded_response(tool_calls=calls),
        agent_name=_COLLECTOR,
        run_id=cycle.run_id,
        run_target=cycle.job.container,
    )
    db.messages.set_run_outcome(cycle.run_id, cycle.outcome.value, cycle.summary)
    _seed_what_the_cycle_kept(db, cycle)


def _seed_what_the_cycle_kept(db: Database, cycle: _SeededCycle) -> None:
    """The entry a cycle wrote, landed in the job's collection under that cycle's OWN run id —
    so the write is attributed the way production attributes it, and the header renders the
    writes clause it renders in a real deployment."""
    if cycle.wrote is None:
        return
    key, content = cycle.wrote
    require_memory(db, cycle.job.container).write(
        [EntryInput(key=key, content=content)], author=_COLLECTOR, run_id=cycle.run_id
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
# Neither case reads a tool NAME, and neither claims that a browse did not happen: whether a
# call was made is a ROUTE — the design's own worked example of a check that fits none of the
# three categories — so it is measured in the tool sequence, and its end-state form is the
# world's own foreclosure, which the answer claim reads directly.
#
# Both stay LOCAL rather than graduating into ``assertions.py``.  A claim graduates at the
# second CUSTOMER, and the two cases below are one behaviour family in one FILE.


def _looking_something_up_wrote_nothing(sample: SampleObservation, _world: World) -> Answer:
    """The turn answered a question and wrote no entry anywhere.

    A violating sample is nameable: one that answers by first filing the answer into a list, and
    one that "helpfully" copies what it read into a collection of its own."""
    wrote = sorted(f"{entry.collection}:{entry.key}" for entry in sample.entries)
    return not wrote, f"wrote {wrote}"


def _no_mechanism_was_created_or_changed(sample: SampleObservation, _world: World) -> Answer:
    """No mechanism was stood up, retired or edited.

    Read off the mutation LEDGER rather than off a field-by-field diff, so a change nobody
    enumerated is caught too.  A violating sample is nameable: one that reads a question about
    what she said as a request to start watching the topic, and one that reaches into a running
    job while reporting on it."""
    touched = sorted(
        one.name for one in sample.mechanisms if one.born_this_run or one.changed_this_run
    )
    return not touched, f"created or changed {touched}"


def _the_answer_came_from_what_she_was_given(sample: SampleObservation, world: World) -> Answer:
    """The token the answer turns on is in what the model was GIVEN this turn.

    The whole behaviour, in end-state form.  ``given`` is the user's turns, the tool results and
    the system prompt — never Penny's own turns, because a value she states out of her own
    account of it launders itself — and each case's world is built so the token appears in no
    collection and in no ambient render.  So it reaches ``given`` if and only if a read put it
    there, and every other route fails: a sample that browsed the topic gets an empty page, one
    that recalled the salient turn from conversation reads it off an assistant turn this
    excludes, and one that invented a plausible answer never had it at all."""
    given = fold_typography(sample.given)
    missing = [token for token in world.answers if fold_typography(token) not in given]
    return not missing, f"nothing the round was given carries {missing}"


async def _drive(chat_eval: ChatEval, model: str, case: _LogReadCase) -> Cohort:
    """Drive one log-read case: the history its own world seeds, the routines that world needs
    in the registry, and the premise re-asserted before the turn."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        model=model,
        seed=case.seed,
        seed_skills=list(case.skills),
        prepare=_probe(case),
        world=case.ground,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


# What both cases measure.  ``ROUTINE_SHAPE`` and ``ROUTINE_NAME`` are absent: they read the
# routines in the registry, and neither turn teaches one — so on a correct cohort they read the
# world's own seeded routines on every sample and pool to a serene 0.000 that is neither
# agreement nor blindness but a reading of the FIXTURE.  ``TOOL_SEQUENCE`` is where the
# behaviour's own NEGATIVE direction lives, and only that half: the chat observation narrows a
# sample's calls to ``ENACTING_TOOLS``, which carries ``browse`` and no read verb — so a sample
# that went and looked the topic up reads differently from every other one, while a sample that
# read her logs and a sample that did nothing at all both read "no call".  That is the right way
# round for these two cases, since browsing is the failure they are named against; the missing
# half is reported as a harness gap rather than worked around here.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ═══ what she said ═══════════════════════════════════════════════════════════

_PENNY_MESSAGES = _LogReadCase(
    case_id="speak-logread-penny-messages-recall",
    behaviour=(
        "In the chat agent, when the user asks what she told them and the answer is out of the "
        "conversation window, Penny goes back through her own messages and states what she "
        "actually said — rather than looking the topic up afresh — and changes nothing."
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
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the reply carries what the log holds, and the turn left everything as it was.
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: answering wrote nothing", _looking_something_up_wrote_nothing, SpecCategory.STORE
    )
    cohort.claim(
        "state: no mechanism was created or changed",
        _no_mechanism_was_created_or_changed,
        SpecCategory.STORE,
    )

    # PROVENANCE — the half the source case had none of, and the half this behaviour IS.
    cohort.claim(
        "state: the answer came from what the round was given, not from her own account of it",
        _the_answer_came_from_what_she_was_given,
        SpecCategory.PROVENANCE,
    )
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ what her jobs did ═══════════════════════════════════════════════════════

_COLLECTOR_RUNS_CASE = _LogReadCase(
    case_id="speak-logread-collector-runs",
    behaviour=(
        "In the chat agent, when the user asks how her background jobs are doing and why any of "
        "them is in trouble, Penny answers out of the run record — the only place a failed "
        "cycle's reason exists — rather than from what the ambient header already carries, and "
        "changes nothing."
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
    skills=(WATCH_ROUTINE,),
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
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: answering wrote nothing", _looking_something_up_wrote_nothing, SpecCategory.STORE
    )
    cohort.claim(
        "state: no mechanism was created or changed",
        _no_mechanism_was_created_or_changed,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.claim(
        "state: the answer came from what the round was given, not from what was already ambient",
        _the_answer_came_from_what_she_was_given,
        SpecCategory.PROVENANCE,
    )
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# Every ported case, in one place — so the deterministic probe in ``test_eval_harness.py`` can
# drive each one's seeder and premise without a GPU.
LOG_READ_CASES = (_PENNY_MESSAGES, _COLLECTOR_RUNS_CASE)
