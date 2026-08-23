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

# Wide enough that a world seeding a conversation window is read WHOLE: the probe's answer
# has to be "none of them", not "none of the first few".
_VECTOR_PROBE_LIMIT = 500

# Directions/authors for seeding the conversation logs.
_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY

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
# case adds is their HISTORY: completed runs, which is what the collector-runs index reads.

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


def _seed_run(
    db: Database,
    *,
    target: str,
    run_id: str,
    outcome: RunOutcome,
    summary: str,
    steps: list[DistillInput],
) -> None:
    """Seed one completed collector run as a ``promptlog`` row (+ its outcome).

    That row IS the ``collector-runs`` content — a run renders once ``set_run_outcome``
    stamps ``run_outcome`` on it, and the response carries the calls the run made.  The
    envelope is built by the seeded-ledger helpers the transition suite already writes
    history with, so a run seeded here and a run seeded there are one wire shape rather
    than two hand-built ones free to differ.

    The id is a SEEDED one, so every reader of "what did the model do this sample"
    excludes it: a job's past cycles are history, and counting them as this turn's calls
    would report a quiet turn as a busy one."""
    calls = [_wire_tool_call(f"{run_id}-{index}", step) for index, step in enumerate(steps)]
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=_seeded_response(tool_calls=calls),
        agent_name="collector",
        run_id=run_id,
        run_target=target,
    )
    db.messages.set_run_outcome(run_id, outcome.value, summary)


def _cycle_steps(job: StandingJob, *, wrote: tuple[str, str] | None) -> list[DistillInput]:
    """One cycle's calls, as the job's own program makes them: read the page, write what it
    found (when it found something), close.  A quiet cycle just reads and closes."""
    page, watched_for = job.values["page"], job.values["watched_for"]
    steps = [
        DistillInput(
            source_ordinal=1,
            tool=_BROWSE,
            arguments={"queries": [page], "extract": watched_for},
            result=f"You opened {page} ({_BROWSE} result)\n{watched_for}: nothing new.",
        )
    ]
    if wrote is not None:
        key, content = wrote
        steps.append(
            DistillInput(
                source_ordinal=2,
                tool=_WRITE,
                arguments={"memory": job.container, "entries": [{"key": key, "content": content}]},
                result=f"You saved an entry to {job.container}: ({_WRITE} result)\nWrote 1 entry.",
            )
        )
    return [
        *steps,
        DistillInput(
            source_ordinal=len(steps) + 1,
            tool=PennyConstants.DONE_TOOL_NAME,
            arguments={},
            result=f"You finished the cycle. ({PennyConstants.DONE_TOOL_NAME} result)\nDone.",
        ),
    ]


class _SeededCycle(NamedTuple):
    """One completed cycle in a job's history: which job ran, what it came back with (a
    key and content, or nothing on a quiet cycle), and how the run recorded itself."""

    job: StandingJob
    name: str
    outcome: RunOutcome
    summary: str
    wrote: tuple[str, str] | None = None


_SEEDED_CYCLES = (
    _SeededCycle(
        _PATCH_NOTES_JOB,
        "patch-notes-cycle-1",
        RunOutcome.WORKED,
        "Recorded the 2.3 balance patch.",
        wrote=("Patch 2.3", "Patch 2.3 — ember mage rebalance."),
    ),
    _SeededCycle(
        _PATCH_NOTES_JOB,
        "patch-notes-cycle-2",
        RunOutcome.NO_WORK,
        "No new patch notes this cycle.",
    ),
    _SeededCycle(
        _TRAIL_JOB,
        "trail-cycle-1",
        RunOutcome.WORKED,
        "Logged today's trail status.",
        wrote=("today", "Verdant Hollow — muddy after rain."),
    ),
)


def _seed_collector_activity(db: Database) -> None:
    """Two standing jobs, then the completed cycles behind them — the cross-collector
    history the ``collector-runs`` index renders."""
    seed_standing_jobs(_PATCH_NOTES_JOB, _TRAIL_JOB)(db)
    for cycle in _SEEDED_CYCLES:
        _seed_run(
            db,
            target=cycle.job.container,
            run_id=seeded_run_id(cycle.name),
            outcome=cycle.outcome,
            summary=cycle.summary,
            steps=_cycle_steps(cycle.job, wrote=cycle.wrote),
        )


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


def _score_collector_runs(db: Database, _before: set[str], _reply: str) -> list[Check]:
    """Asked how her background jobs have been doing, she reads the cross-collector
    collector-runs index rather than the collections themselves."""
    read = _dispatched(db, _LOG_READ, "memory", _COLLECTOR_RUNS)
    return [
        Check(
            f"spine: she read the {_COLLECTOR_RUNS} index",
            read,
            anchor=f"{_LOG_READ}(",
            kind="spine",
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
    suite drives cycles, but nothing else asks her ABOUT them."""
    await chat_eval(
        case_id="speak-logread-collector-runs",
        family=_FAMILY,
        message="how have your background jobs been doing lately?",
        seed=_seed_collector_activity,
        seed_skills=[WATCH_ROUTINE],
        score=_score_collector_runs,
        min_pass_rate=None,
    )
