"""Digging through what was already said and done: the user asks Penny to look back
over old messages, over what she read, or over what her jobs have been doing.

Four logs answer that ask, and the case names which one the request means:

  log_read("user-messages" | "penny-messages" | "browse-results" | "collector-runs")

Every case is scored on the persisted tool CALL (which log she named) plus DB state, never
on wording.  Synthetic topics throughout (the repo is public): an invented hobby (``lantern
kiting``), an invented recommendation (``silverleaf moss``), invented pages on example
domains.

**The conversation confound (the two message cases).**  The chat agent already injects the
last ``MESSAGE_CONTEXT_LIMIT`` (=20) turns as in-context history, so a salient message
INSIDE that window is answered from context and never needs a ``log_read`` — a false gap.
So both conversation cases seed the salient turn FIRST, then ``_FILLER_PAIRS`` (24) neutral
turns after it, pushing it out of BOTH the context window and the per-direction top-N fetch
(``get_messages_since`` caps each direction at 20).  Retrieval then genuinely requires a
read.  Nothing else can leak it in: the chat prompt injects no speculative recalled-content
block (the ambient inversion, #1555, and the recall substrate's removal, #1583).

**The self-state confound (the collector-runs case).**  The same confound with a different
ambient carrier, and for a long time it went untreated here.  Penny's self-state header
renders her Active mechanisms and Recent activity EVERY turn — each job's cadence, its last
run's outcome, one line per recent run — so "how have the jobs been doing" is fully answered
at zero calls, and a model that reasons notices and skips the read.  (Measured, #1990: with
reasoning enabled four of five samples answered from the header, correctly, and a case that
scored the read marked all four wrong.  The more intelligent behaviour scored lower.)  The
treatment mirrors ``_FILLER_PAIRS``: aim the ask at what the header STRUCTURALLY cannot
carry.  A cycle's OUTCOME is ambient; the REASON behind it never is, and a failed cycle
writes nothing, so no collection holds it either.  Retrieval genuinely requires a read
again, and — as with ``silverleaf`` — the reply is what proves the read happened.

**The world is one the user built (#1911/migration 0108: nothing is pre-seeded).**  The
collection the act case saves into is a plain user-built container, and the two jobs the
collector-runs case reads about are STANDING JOBS — a taught routine applied to two pages
through the production instantiation seam, with real completed runs behind them (see
``test_standing_collection.py``, which owns that world; a second copy of it here would be a
second world, free to drift from the one chat really leaves).  A hand-authored prose prompt
would be a config defect the collector cannot read (#1916's strict dialect), so a fixture
seeded that way would claim jobs that could never have run.

**Report-only** (``min_pass_rate=None``) throughout: the gates these cases carried (0.6 on
three of them) predate the conversation machine fronting every turn, so they are numbers
from a different runtime.  Re-baselining under this one is a read for the code owner, not a
threshold for this rewrite to pick.  Each sample also renders, ADVISORY, the state the
machine landed the turn in — a compound imperative can land in ``learn``, whose run-end
terminal replaces the reply (#1839), and that shows in the row rather than as a puzzling
reply miss.

``read_run_calls`` is collector-internal rather than user-dispatchable — its argument is a
collection target and "what did your last run do" is not a phrasing a user reaches for
(baseline 0/3, the model browsing or writing instead) — so no case dispatches it.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput, LogEntryInput
from penny.database.skills import DistillInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Seeder,
    collection_entries,
    seeded_run_id,
    tool_call_arg_values,
    tool_call_sequence,
)
from penny.tests.eval.fixtures import SynthCollection

# The seeded-ledger wire helpers the transition suite writes its own history with — read
# from there rather than restated, so a collector run seeded here has the same envelope
# every other seeded run does (the precedent is test_collector_enactment.py, which builds
# its world from that module's vocabulary for the same reason).
from penny.tests.eval.test_standing_collection import (
    WATCH_ROUTINE,
    StandingJob,
    landed_state_check,
    seed_standing_jobs,
)
from penny.tests.eval.test_state_transitions import _seeded_response, _wire_tool_call

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module.
_FAMILY = "speakable-logread"

# ── Tool + log names (constants, never magic strings) ────────────────────────
_LOG_READ = "log_read"
_BROWSE = "browse"
_WRITE = "collection_write"

# The four logs the cases read.
_USER_MESSAGES = PennyConstants.MEMORY_USER_MESSAGES_LOG
_PENNY_MESSAGES = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_BROWSE_RESULTS = PennyConstants.MEMORY_BROWSE_RESULTS_LOG
_COLLECTOR_RUNS = PennyConstants.MEMORY_COLLECTOR_RUNS_LOG

# The argument every store-addressing tool names its target with — what a read is keyed to
# here, since the store is the claim and the verb is the model's choice.
_MEMORY_ARG = "memory"

# The other two arguments a RUN is addressed by.  A collector run is identified in exactly
# three ways — the cross-collector log's name (``memory``), the run's own id (``event_id``)
# and the run's target (``target``) — and different tools take different ones.  Naming the
# anchors rather than the verbs is what lets a tool nobody listed count as a run-record read.
_TARGET_ARG = "target"
_EVENT_ARG = "event_id"

# Wide enough that a world seeding a conversation window is read WHOLE: the probe's answer
# has to be "none of them", not "none of the first few".
_VECTOR_PROBE_LIMIT = 500

# Directions/authors for seeding the conversation logs.
_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY
_COLLECTOR = PennyConstants.MessageAuthor.COLLECTOR

# The collection the act case saves into — user-built storage, no job attached, which is
# what a collection someone made to keep a list in looks like after the soft reboot.
_HOBBIES = SynthCollection(
    "hobbies",
    "Things the user is into — the hobbies they've said they want kept track of.",
    entries=(
        "Sea glass hunting — beachcombing the north shore on cold mornings.",
        "Sourdough — a stiff starter kept on the counter.",
    ),
)


# ── Typography-fold + recap helpers (kept local, minimal) ─────────────────────


def _normalize(text: str) -> str:
    """Fold the typography gpt-oss sprinkles so a SEMANTIC substring probe isn't defeated
    by cosmetics: unicode hyphens → '-', nbsp/zero-width/narrow spaces → ' ', bold markers
    stripped, curly quotes straightened, lowercased.  (A 0/N from an un-normalized probe is
    a scorer bug, not a model failure.)"""
    folded = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        folded = folded.replace(dash, "-")
    for space in ("\xa0", "​", " ", " "):
        folded = folded.replace(space, " ")
    for src, dst in (("’", "'"), ("“", '"'), ("”", '"'), ("*", "")):
        folded = folded.replace(src, dst)
    return folded


def _saved_text(db: Database, name: str) -> str:
    """A collection's keys AND contents, normalized and joined — the probe for "did the
    subject land here", robust to whether the model put the subject in the key or the body
    and to its typography."""
    entries = collection_entries(db, name)
    return _normalize(" ".join([*entries.keys(), *entries.values()]))


def _reply_reflects(reply: str, tokens: list[str]) -> list[Check]:
    """The final reply must REFLECT what was read/done (the #1478 recap prong): it names
    each subject it acted on.  One graded Check per token, anchored to the reply row;
    normalized for typography, checked as substrings, never exact wording."""
    normalized = _normalize(reply)
    return [
        Check(
            f"reply reflects '{token}'",
            _normalize(token) in normalized,
            anchor=REPLY_ANCHOR,
            kind="reply",
        )
        for token in tokens
    ]


def _dispatched(db: Database, tool: str, field: str, expected: str) -> bool:
    """Did the model call ``tool`` with ``field == expected`` (typography-folded) at least
    once this run?  The structural dispatch probe — reads the persisted promptlog, not a
    harness spy."""
    values = [_normalize(value) for value in tool_call_arg_values(db, tool, field)]
    return _normalize(expected) in values


def _read_the_store(db: Database, store: str) -> bool:
    """Did ANY call this run read that store?

    Keyed to the STORE, never to a verb: several tools take a ``memory`` argument and more
    than one of them is a legitimate way to look something up in a log — a cursored
    ``log_read`` walks it, ``read_similar`` ranks it against an anchor, and a plugin could
    add a third tomorrow.  Which one a recall-shaped ask reaches for is the model's call to
    make; what the case is about is whether it went to the right place.

    The verb set is READ from what the run actually called rather than enumerated here, so
    a tool nobody listed participates for free — the alternative is a name-set that
    silently fails to fire for the one shape nobody thought of."""
    return any(_dispatched(db, tool, _MEMORY_ARG, store) for tool in set(tool_call_sequence(db)))


def _assert_the_seeded_turns_are_recallable(penny: Penny) -> None:
    """Every seeded message carries a vector — asserted BEFORE the turn is driven.

    A seeded message with no embedding cannot rank in any similarity read, so a case whose
    answer lives in that message is UNREACHABLE: the model aims its read correctly, finds
    nothing, and the case scores 0.00 with nothing in the transcript to say why.  That is
    exactly what happened here, and it is the third vector-less seed of this fleet — so the
    class fails at prepare, naming the row, rather than after a GPU run.

    Read through the production backfill's OWN query, so "did the seeds get vectors" is the
    same question the runner answered rather than a second opinion about it."""
    unvectored = penny.db.messages.messages_without_embeddings(limit=_VECTOR_PROBE_LIMIT)
    assert not unvectored, (
        "every seeded message must carry an embedding before the turn runs — "
        f"{len(unvectored)} without one, first: {unvectored[0].content!r}"
    )


# ── Conversation seeding (out-of-window) ─────────────────────────────────────
# ≥ MESSAGE_CONTEXT_LIMIT (20), so the salient turn is pushed out of BOTH the context
# window AND the per-direction top-20 fetch, with margin.
_FILLER_PAIRS = 24

# Neutral, topic-free chit-chat — names no hobby / recommendation, so it can answer neither
# conversation case from context.
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
    """Seed the salient turn FIRST (oldest), then ``_FILLER_PAIRS`` neutral user/Penny
    turns after it — so the salient turn is genuinely out of context and its retrieval
    requires a ``log_read``.  Penny filler carries the real recipient so it counts as an
    autonomous outgoing turn in the context builder, exercising the same push-out prod
    would."""

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


# ── The two conversation cases' salient turns + their probes ─────────────────
_HOBBY_MESSAGE = (
    "honestly I've completely fallen for lantern kiting lately — can't stop doing it on weekends"
)
_HOBBY_TOKEN = "lantern"

_SUGGESTION_MESSAGE = (
    "for your moss terrarium, I'd really go with silverleaf moss — it handles low light and "
    "stays compact"
)
_SUGGESTION_TOKEN = "silverleaf"


def _seed_hobby(db: Database) -> None:
    """The hobby said long ago, plus the user-built list it is asked to be added to."""
    _seed_out_of_window(_INCOMING, _HOBBY_MESSAGE)(db)
    db.memories.create_collection(_HOBBIES.name, _HOBBIES.description)
    require_memory(db, _HOBBIES.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in _HOBBIES.entries],
        author="user",
    )


def _seed_suggestion(db: Database) -> None:
    _seed_out_of_window(_OUTGOING, _SUGGESTION_MESSAGE)(db)


# ── Browse-history seeding ───────────────────────────────────────────────────
# Distinctive, invented browsed topics on example domains — the reply must name at least
# one, proving it summarized what was actually read.
_BROWSE_ENTRIES = (
    "## browse: https://coast.example.com/tidewatch-cove\n"
    "Title: Tidewatch Cove tide pools guide\n"
    "Tidewatch Cove has some of the richest tide pools on the coast, best explored at low "
    "tide in the early morning.\n",
    "## browse: https://jazz.example.com/selmer-restoration\n"
    "Title: Restoring a vintage Selmer saxophone\n"
    "A step-by-step on re-padding and re-lacquering a vintage Selmer alto saxophone.\n",
    "## browse: https://trails.example.com/verdant-hollow\n"
    "Title: Verdant Hollow trail conditions\n"
    "The Verdant Hollow trail is a 7-mile loop with a steep final ascent; check the "
    "conditions after rain.\n",
)
_BROWSE_TOPIC_TOKENS = ("tidewatch", "selmer", "verdant")


def _seed_browse_history(db: Database) -> None:
    require_memory(db, _BROWSE_RESULTS).append(
        [LogEntryInput(content=content) for content in _BROWSE_ENTRIES], author="chat"
    )


# ── Collector-activity seeding: two standing jobs and the runs behind them ────
#
# The jobs are the standing-collection story's own world — ONE taught routine, applied to
# two pages, so the pair is two real jobs rather than two rows shaped like jobs.  What this
# case adds is their HISTORY: completed cycles, which is what the collector-runs index
# reads — including one that FAILED, the cycle the case is actually about.

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

# The failed cycle's stamped reason, and the invented word inside it the reply is scored on.
# ``thistledown`` exists in this world in exactly ONE place: this string, stamped on the run
# and rendered by the run record.  The self-state header renders that cycle's OUTCOME
# (``FAILED``) and never the reason behind it, and a failed cycle writes nothing, so no
# collection holds it either.  A reply carrying the word therefore came from the run record
# and from nowhere else — the ``silverleaf`` construction, applied to the one case in this
# module that never had it.
_TRAIL_FAILURE_REASON = (
    "Couldn't read the page — it bounced to a notice about the thistledown survey closure."
)
_FAILURE_TOKEN = "thistledown"


class _SeededCycle(NamedTuple):
    """One completed cycle in a job's history: which job ran, what it came back with (a key
    and content, or nothing on a quiet or failed cycle), and how the run recorded itself."""

    job: StandingJob
    name: str
    outcome: RunOutcome
    summary: str
    wrote: tuple[str, str] | None = None

    @property
    def run_id(self) -> str:
        """The seeded id every row of this cycle carries — minted once, so the ledger row,
        the entry it wrote and any probe naming the run all say the same thing."""
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
_SEEDED_RUN_IDS = tuple(cycle.run_id for cycle in _SEEDED_CYCLES)


def _browse_result(cycle: _SeededCycle) -> str:
    """What the cycle's page read came back with — the finding it went on to write, the
    failure that ended it, or nothing new on a quiet cycle.  A cycle whose trace says it
    read nothing while its reason says it recorded a patch is a world at odds with itself,
    and the model reads both."""
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
    found (when it found something), close.  A quiet cycle just reads and closes; a failed
    cycle reads and stops there, because that is what its outcome means."""
    steps = [_read_step(cycle)]
    if cycle.wrote is not None:
        steps.append(_write_step(cycle, *cycle.wrote))
    if not cycle.closed:
        return steps
    return [*steps, _close_step(len(steps) + 1)]


def _seed_run(db: Database, cycle: _SeededCycle) -> None:
    """Seed one completed collector cycle: the ``promptlog`` row that IS its
    ``collector-runs`` content, the outcome + reason stamped on it, and — when the cycle
    wrote — the entry it left behind, attributed to this run.

    That last part is what makes the world COHERE, and it was missing.  The cycle's trace
    says it wrote and its reason says what it recorded; if the entry never lands, the
    collection the model is pointed at is empty, no ``last_written_by_run_id`` is stamped,
    and the self-state header silently drops the ``· wrote '<key>' → `<collection>``` clause
    production renders (#1641).  The case would then measure a world THINNER than the real
    one, against a 'read the collections instead' alternative that is a dead end rather than
    a wrong answer.

    The envelope is built by the seeded-ledger helpers the transition suite already writes
    history with, so a run seeded here and a run seeded there are one wire shape rather than
    two hand-built ones free to differ.  The id is a SEEDED one, so every reader of "what did
    the model do this sample" excludes it: a job's past cycles are history, and counting them
    as this turn's calls would report a quiet turn as a busy one."""
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
    """The entry a cycle wrote, landed in the job's collection under that cycle's OWN run
    id — so the write is attributed the way production attributes it, and the header renders
    the writes clause it renders in a real deployment."""
    if cycle.wrote is None:
        return
    key, content = cycle.wrote
    require_memory(db, cycle.job.container).write(
        [EntryInput(key=key, content=content)], author=_COLLECTOR, run_id=cycle.run_id
    )


def _seed_collector_activity(db: Database) -> None:
    """Two standing jobs, then the completed cycles behind them — the cross-collector
    history the ``collector-runs`` index renders."""
    seed_standing_jobs(*_SEEDED_JOBS)(db)
    for cycle in _SEEDED_CYCLES:
        _seed_run(db, cycle)


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_user_messages_act(db: Database, _before: set[str], reply: str) -> list[Check]:
    """Asked to look back over what was said and add it to a list, she reads the
    user-messages log and the out-of-window hobby lands in the list the user named."""
    read = _dispatched(db, _LOG_READ, "memory", _USER_MESSAGES)
    saved = _HOBBY_TOKEN in _saved_text(db, _HOBBIES.name)
    return [
        Check(
            f"spine: she read the {_USER_MESSAGES} log",
            read,
            anchor=f"{_LOG_READ}(",
            kind="spine",
        ),
        Check(
            f"state: the hobby landed in {_HOBBIES.name!r}",
            saved,
            rationale=None
            if saved
            else f"{_HOBBIES.name} holds {collection_entries(db, _HOBBIES.name)}",
            kind="state",
        ),
        *_reply_reflects(reply, [_HOBBY_TOKEN]),
        landed_state_check(db),
    ]


def _score_penny_messages_recall(db: Database, _before: set[str], reply: str) -> list[Check]:
    """Asked what SHE said, she goes to the penny-messages log and relays the out-of-window
    recommendation.

    The spine is the STORE, not the verb: walking the log and ranking it against the topic
    are both legitimate ways to look back at what she said, and the measured samples chose
    the second.  What stays exact is the reply check — ``silverleaf`` is an invented word
    that exists in this world only in the seeded turn, so a reply carrying it can only have
    read it."""
    read = _read_the_store(db, _PENNY_MESSAGES)
    return [
        Check(
            f"spine: she went to the {_PENNY_MESSAGES} log",
            read,
            anchor=f"{_MEMORY_ARG}=",
            rationale=None if read else f"no call this run named {_PENNY_MESSAGES}",
            kind="spine",
        ),
        *_reply_reflects(reply, [_SUGGESTION_TOKEN]),
        landed_state_check(db),
    ]


def _score_browse_results(db: Database, _before: set[str], reply: str) -> list[Check]:
    """Asked what she has been looking up, she reads the browse-results log and names
    something she actually read."""
    read = _dispatched(db, _LOG_READ, "memory", _BROWSE_RESULTS)
    named = any(token in _normalize(reply) for token in _BROWSE_TOPIC_TOKENS)
    return [
        Check(
            f"spine: she read the {_BROWSE_RESULTS} log",
            read,
            anchor=f"{_LOG_READ}(",
            kind="spine",
        ),
        Check(
            "reply: it names something she had browsed",
            named,
            anchor=REPLY_ANCHOR,
            rationale=None if named else f"named none of {list(_BROWSE_TOPIC_TOKENS)}",
            kind="reply",
        ),
        landed_state_check(db),
    ]


def _named_the_run_record(db: Database, tool: str) -> bool:
    """Did this tool's calls this run bind any of the three run anchors to THIS world's run
    history — the cross-collector log, one of the jobs as a run target, or a seeded run id?"""
    if _dispatched(db, tool, _MEMORY_ARG, _COLLECTOR_RUNS):
        return True
    if any(_dispatched(db, tool, _TARGET_ARG, job.container) for job in _SEEDED_JOBS):
        return True
    return any(
        _normalize(run_id) in _normalize(value)
        for value in tool_call_arg_values(db, tool, _EVENT_ARG)
        for run_id in _SEEDED_RUN_IDS
    )


def _consulted_the_run_record(db: Database) -> bool:
    """Did ANY call this run go to the run record?

    Keyed to the ANCHORS a run is addressed by, never to a verb — the rule
    ``_read_the_store`` states, widened past a single argument name because a run has three
    anchors and different tools take different ones: the log's NAME
    (``log_read(memory=collector-runs)``), a run TARGET (``read_run_calls(target=<job>)``)
    and a run ID (``get_event(event_id="run <id>")``).  All three are rendered verbatim in
    the self-state header, all three lead to the same records, and which she reaches for is
    her call to make.  Reading a job's COLLECTION binds that collection to ``memory`` and
    matches none of them — which is what keeps "the index rather than the collections" a
    real distinction rather than a preference for one verb.

    The verb set is READ from what the run actually called rather than enumerated here, so a
    tool nobody listed participates for free."""
    return any(_named_the_run_record(db, tool) for tool in set(tool_call_sequence(db)))


def _score_collector_runs(db: Database, _before: set[str], reply: str) -> list[Check]:
    """Asked how her background jobs have been doing and why any of them is in trouble, she
    goes to the run record — by whichever verb — and tells the user what it says.

    The scored claim is the OUTCOME: the reply carries the failed cycle's reason, a phrase
    that exists in this world only in the run record.  The header renders that cycle's
    outcome and never its reason, and a failed cycle wrote nothing, so no collection holds
    it — a reply carrying it can only have come from the run record.  That makes "the index
    rather than the collections" something the case OBSERVES in the answer instead of
    something it demands of the call sequence.

    The route is scored too, one rung below the answer, because the two failures underneath
    a silent 0.00 are different and a reader needs to tell them apart: "answered from the
    header without looking" fails both checks, "looked and then said nothing useful" fails
    only the reply.  The old single check scored those two identically — and scored a
    correct header answer as a miss."""
    consulted = _consulted_the_run_record(db)
    told = _FAILURE_TOKEN in _normalize(reply)
    return [
        Check(
            "spine: she went to the run record",
            consulted,
            rationale=None if consulted else "no call this run named the run record",
            kind="spine",
        ),
        Check(
            "reply: it says WHY the failing cycle failed",
            told,
            anchor=REPLY_ANCHOR,
            rationale=None if told else f"the reply never names {_FAILURE_TOKEN!r}",
            kind="reply",
        ),
        landed_state_check(db),
    ]


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_user_messages_act(chat_eval: ChatEval) -> None:
    """Report-only.  The out-of-window hobby is retrieved from ``user-messages`` and lands
    in the list the user pointed at.  A follow-up could gate on the OUTCOME alone (the
    hobby in the list) rather than on the read — requiring the read is a claim about the
    mechanism, and the user's ask is about the outcome."""
    await chat_eval(
        case_id="speak-logread-user-messages-act",
        family=_FAMILY,
        message=(
            "look back over everything i've told you and add what i said i'm into to my "
            "hobbies list"
        ),
        seed=_seed_hobby,
        prepare=_assert_the_seeded_turns_are_recallable,
        score=_score_user_messages_act,
        min_pass_rate=None,
    )


async def test_penny_messages_recall(chat_eval: ChatEval) -> None:
    """Report-only — the standing measurement of the #1524 recall-vocabulary gap.

    Its earlier baseline (0/3, every sample browsing the topic instead of looking back at
    what she said) is NOT what the last run showed: the samples went to
    ``penny-messages`` — by ``read_similar``, anchored on the topic — and found nothing,
    because the seeded turn carried no embedding and could not rank.  So the two things
    that scored 0.00 were a scorer keyed to one verb and a world the answer was
    unreachable in; both are fixed, and what the case measures now is whether she looks
    back at all and whether what she says came from what she read."""
    await chat_eval(
        case_id="speak-logread-penny-messages-recall",
        family=_FAMILY,
        message="dig back through our old messages — what exactly did you tell me "
        "to use for my moss terrarium?",
        seed=_seed_suggestion,
        prepare=_assert_the_seeded_turns_are_recallable,
        score=_score_penny_messages_recall,
        min_pass_rate=None,
    )


async def test_browse_results(chat_eval: ChatEval) -> None:
    """Report-only.  Reading a system log and summarizing it — the direction that
    dispatches most reliably."""
    await chat_eval(
        case_id="speak-logread-browse-results",
        family=_FAMILY,
        message="what have you been looking up lately? give me the gist",
        seed=_seed_browse_history,
        score=_score_browse_results,
        min_pass_rate=None,
    )


async def test_collector_runs(chat_eval: ChatEval) -> None:
    """Report-only.  The chat-side introspection ask about background work — the enactment
    suite drives cycles, but nothing else asks her ABOUT them.

    The ask reaches PAST the self-state header deliberately.  "How have they been doing" is
    answered by the header alone — every mechanism, every cadence, every run's outcome — so
    a case that stopped there scored the read as effort rather than as routing, and marked a
    correct header answer wrong (#1990).  What the header cannot carry is WHY a cycle
    failed, and the second clause is what asks for it: the opening stays verbatim so the
    case is still the same introspection ask, and the clause makes the answer live somewhere
    only a read can reach."""
    await chat_eval(
        case_id="speak-logread-collector-runs",
        family=_FAMILY,
        message=(
            "how have your background jobs been doing lately? if any of them is having "
            "trouble i want to know why"
        ),
        seed=_seed_collector_activity,
        seed_skills=[WATCH_ROUTINE],
        score=_score_collector_runs,
        min_pass_rate=None,
    )
