"""Speakable log reads (epic #1521, issue #1523, Wave 2): NL requests that ask
Penny to READ from a log/message/run-log surface, scored STRUCTURALLY.

The sibling of ``test_speakable_tools.py`` (entry/browse actions); this file covers
the *read-a-log* half of the speakable surface — a family of phrasings that map,
by MEANING, onto one of the log-read tools:

  log_read("user-messages" | "penny-messages" | "browse-results" | "collector-runs")

Every case is scored on the persisted tool CALL (the log the model named) + DB
state, never on wording.  Synthetic topics only (the repo is public): invented
hobbies (``lantern kiting``), invented recommendations (``silverleaf moss``), and
invented collector collections (``patch-notes``).

**The conversation confound (cases 1–2).**  The chat agent already injects the
last ``MESSAGE_CONTEXT_LIMIT`` (=20) turns as in-context history, so a salient
message *inside* that window is answered from context and never needs a
``log_read`` — a false gap.  So the two conversation cases seed the salient turn
FIRST, then ``_FILLER_PAIRS`` (24) neutral user/Penny turns after it, pushing the
salient turn out of BOTH the 20-message context window AND the per-direction
top-N fetch (``get_messages_since`` caps each direction at 20).  Retrieval then
genuinely requires a ``log_read``.

**No ambient-recall confound.**  The chat prompt no longer injects a
speculative recalled-content block (the ambient inversion, #1555, and the recall
substrate's removal, #1583), so a topically-matching out-of-window user message
can't leak into the prompt by similarity — retrieval genuinely requires a
``log_read``.  Case 1's status (gated vs. report-only) is decided by its baseline
— see its docstring.

**read_run_calls is collector-internal, not user-dispatchable — dropped.**  Its
arg is a collection ``target`` and "what did your last run do" is not a phrasing
a user reaches for; baseline was 0/3, the model browsing/writing instead of ever
calling it — confirming the epic's scoping of it as collector-only.  It survives
only in the no-fire guard's forbidden set.
"""

from __future__ import annotations

import json

import pytest

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import LogEntryInput
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    ChatEval,
    collection_entries,
    tool_call_arg_values,
    tool_was_called,
)

pytestmark = pytest.mark.eval

# ── Tool + log names (constants, never magic strings) ────────────────────────
_LOG_READ = "log_read"
_READ_RUN_CALLS = "read_run_calls"
_BROWSE = "browse"
_WRITE = "collection_write"
_UPDATE = "update_entry"
_DELETE = "collection_delete_entry"

# The four logs the cases read.
_USER_MESSAGES = PennyConstants.MEMORY_USER_MESSAGES_LOG
_PENNY_MESSAGES = PennyConstants.MEMORY_PENNY_MESSAGES_LOG
_BROWSE_RESULTS = PennyConstants.MEMORY_BROWSE_RESULTS_LOG
_COLLECTOR_RUNS = PennyConstants.MEMORY_COLLECTOR_RUNS_LOG

# Every read/mutation tool a no-fire guard must see stay quiet.
_READ_TOOLS = (_LOG_READ, _READ_RUN_CALLS)
_ACTION_TOOLS = (_BROWSE, _WRITE, _UPDATE, _DELETE)

# Directions/authors for seeding the conversation logs.
_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY

_LIKES = "likes"

# Collector collections (the cross-collector run index case).  Both carry an
# extraction_prompt (so they're valid read_run_calls targets); the scheduler never
# ticks in the eval (COLLECTOR_TICK_INTERVAL is bumped past any timeout), so they
# stay inert during the chat turn.
_PATCH_NOTES = "patch-notes"
_TRAIL_CONDITIONS = "trail-conditions"


# ── Typography-fold + recap helpers (kept local, minimal — mirror Wave 1) ─────


def _normalize(text: str) -> str:
    """Fold the typography gpt-oss sprinkles so a SEMANTIC substring probe isn't
    defeated by cosmetics: unicode hyphens → '-', nbsp/zero-width/narrow spaces →
    ' ', bold markers stripped, curly quotes straightened, lowercased.  (A 0/N
    from an un-normalized probe is a scorer bug, not a model failure.)"""
    folded = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        folded = folded.replace(dash, "-")
    for space in ("\xa0", "​", " ", " "):
        folded = folded.replace(space, " ")
    for src, dst in (("’", "'"), ("“", '"'), ("”", '"'), ("*", "")):
        folded = folded.replace(src, dst)
    return folded


def _saved_text(db: Database, name: str) -> str:
    """A collection's keys AND contents, normalized and joined — the probe for
    'did the subject land here', robust to whether the model put the subject in
    the key or the body and to its typography."""
    entries = collection_entries(db, name)
    return _normalize(" ".join([*entries.keys(), *entries.values()]))


def _reply_reflects(reply: str, tokens: list[str]) -> list[str]:
    """The final reply must REFLECT what was read/done (the #1478 recap prong):
    it names each subject it acted on.  Normalized for typography, checked as
    substrings, never on exact wording."""
    normalized = _normalize(reply)
    return [
        f"reply doesn't reflect '{token}' from what was read"
        for token in tokens
        if _normalize(token) not in normalized
    ]


def _dispatched(db: Database, tool: str, field: str, expected: str) -> bool:
    """Did the model call ``tool`` with ``field == expected`` (typography-folded)
    at least once this run?  The structural dispatch probe — reads the persisted
    promptlog, not a harness spy."""
    values = [_normalize(v) for v in tool_call_arg_values(db, tool, field)]
    return _normalize(expected) in values


# ── Conversation seeding (out-of-window) ─────────────────────────────────────
# ≥ MESSAGE_CONTEXT_LIMIT (20), so the salient turn is pushed out of BOTH the
# context window AND the per-direction top-20 fetch, with margin.
_FILLER_PAIRS = 24

# Neutral, topic-free chit-chat — names no hobby / recommendation, so it can
# answer neither conversation case from context nor from ambient recall.
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


def _seed_out_of_window(direction: str, salient: str):
    """Seed the salient turn FIRST (oldest), then ``_FILLER_PAIRS`` neutral
    user/Penny turns after it — so the salient turn is genuinely out of context
    and its retrieval requires a ``log_read``.  Penny filler carries the real
    recipient so it counts as an autonomous outgoing turn in the context builder,
    exercising the same push-out that prod would."""

    def _apply(db: Database) -> None:
        if direction == _INCOMING:
            db.messages.log_message(_INCOMING, TEST_SENDER, salient)
        else:
            db.messages.log_message(_OUTGOING, _PENNY, salient, recipient=TEST_SENDER)
        for index in range(_FILLER_PAIRS):
            db.messages.log_message(_INCOMING, TEST_SENDER, _FILLER_USER[index % len(_FILLER_USER)])
            db.messages.log_message(
                _OUTGOING, _PENNY, _FILLER_PENNY[index % len(_FILLER_PENNY)], recipient=TEST_SENDER
            )

    return _apply


# ── Case-1/2 salient turns + their probes ────────────────────────────────────
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
    _seed_out_of_window(_INCOMING, _HOBBY_MESSAGE)(db)


def _seed_suggestion(db: Database) -> None:
    _seed_out_of_window(_OUTGOING, _SUGGESTION_MESSAGE)(db)


# ── Case-3 browse-history seeding ────────────────────────────────────────────
# Distinctive, invented browsed topics on example domains — the reply must name
# at least one, proving it summarized what was actually read.
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
    db.memory(_BROWSE_RESULTS).append(
        [LogEntryInput(content=content) for content in _BROWSE_ENTRIES], author="chat"
    )


# ── Case-4/5/6 collector-activity seeding ────────────────────────────────────
_PATCH_NOTES_PROMPT = (
    "Collect notable new Mistforge Tactics patch notes.\n"
    "1. browse the web for the latest Mistforge Tactics patch notes.\n"
    '2. collection_write("patch-notes", entries=[{key: patch, content: patch + summary}]).\n'
    "3. done()."
)
_TRAIL_CONDITIONS_PROMPT = (
    "Track current conditions for the Verdant Hollow hiking trail.\n"
    "1. browse the web for the latest Verdant Hollow trail conditions.\n"
    '2. collection_write("trail-conditions", entries=[{key: date, content: conditions}]).\n'
    "3. done()."
)


def _seed_run(
    db: Database,
    *,
    target: str,
    run_id: str,
    outcome: RunOutcome,
    summary: str,
    calls: list[tuple[str, dict]],
) -> None:
    """Seed one completed collector run as a ``promptlog`` row (+ its outcome).

    That row IS the ``collector-runs`` / ``read_run_calls`` content — a run renders
    once ``set_run_outcome`` stamps ``run_outcome`` on it, and the response carries
    the tool calls the run made (so ``read_run_calls`` has a sequence to render)."""
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"{run_id}-{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for index, (name, args) in enumerate(calls)
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }
    db.messages.log_prompt(
        model="seed",
        messages=[],
        response=response,
        agent_name="collector",
        run_id=run_id,
        run_target=target,
    )
    db.messages.set_run_outcome(run_id, outcome.value, summary)


def _seed_collector_activity(db: Database) -> None:
    """Two synthetic collector collections + a few completed runs each — the
    cross-collector run index (``test_collector_runs``)."""
    db.memories.create_collection(
        _PATCH_NOTES,
        "New Mistforge Tactics patch notes worth knowing about.",
        extraction_prompt=_PATCH_NOTES_PROMPT,
        collector_interval_seconds=3600,
    )
    db.memories.create_collection(
        _TRAIL_CONDITIONS,
        "Current conditions for the Verdant Hollow hiking trail.",
        extraction_prompt=_TRAIL_CONDITIONS_PROMPT,
        collector_interval_seconds=3600,
    )
    _seed_run(
        db,
        target=_PATCH_NOTES,
        run_id="patch-notes-run-1",
        outcome=RunOutcome.WORKED,
        summary="Recorded the 2.3 balance patch.",
        calls=[
            ("browse", {"queries": ["mistforge tactics patch notes"]}),
            (
                "collection_write",
                {
                    "memory": _PATCH_NOTES,
                    "entries": [
                        {"key": "Patch 2.3", "content": "Patch 2.3 — ember mage rebalance."}
                    ],
                },
            ),
            ("done", {}),
        ],
    )
    _seed_run(
        db,
        target=_PATCH_NOTES,
        run_id="patch-notes-run-2",
        outcome=RunOutcome.NO_WORK,
        summary="No new patch notes this cycle.",
        calls=[
            ("browse", {"queries": ["mistforge tactics patch notes"]}),
            ("done", {}),
        ],
    )
    _seed_run(
        db,
        target=_TRAIL_CONDITIONS,
        run_id="trail-conditions-run-1",
        outcome=RunOutcome.WORKED,
        summary="Logged today's trail status.",
        calls=[
            ("browse", {"queries": ["verdant hollow trail conditions"]}),
            (
                "collection_write",
                {
                    "memory": _TRAIL_CONDITIONS,
                    "entries": [{"key": "today", "content": "Verdant Hollow — muddy after rain."}],
                },
            ),
            ("done", {}),
        ],
    )


# ── Scorers ──────────────────────────────────────────────────────────────────


def _score_user_messages_act(db: Database, _before: set[str], reply: str) -> list[str]:
    """ "Look back over everything I've told you and save what I'm into" must read
    the user-messages log and land the out-of-window hobby in ``likes``."""
    fails: list[str] = []
    if not _dispatched(db, _LOG_READ, "memory", _USER_MESSAGES):
        fails.append("did not log_read the user-messages log to look back over past messages")
    if _HOBBY_TOKEN not in _saved_text(db, _LIKES):
        fails.append(f"the hobby wasn't saved to likes: {collection_entries(db, _LIKES)}")
    return fails + _reply_reflects(reply, [_HOBBY_TOKEN])


def _score_penny_messages_recall(db: Database, _before: set[str], reply: str) -> list[str]:
    """ "Scroll way back and remind me what you suggested" must read the
    penny-messages log and relay the out-of-window recommendation."""
    fails: list[str] = []
    if not _dispatched(db, _LOG_READ, "memory", _PENNY_MESSAGES):
        fails.append("did not log_read the penny-messages log to recall the past suggestion")
    return fails + _reply_reflects(reply, [_SUGGESTION_TOKEN])


def _score_browse_results(db: Database, _before: set[str], reply: str) -> list[str]:
    """ "What have you been looking up lately" must read the browse-results log and
    name at least one thing that was browsed."""
    fails: list[str] = []
    if not _dispatched(db, _LOG_READ, "memory", _BROWSE_RESULTS):
        fails.append("did not log_read the browse-results log for what was looked up")
    normalized = _normalize(reply)
    if not any(token in normalized for token in _BROWSE_TOPIC_TOKENS):
        fails.append(f"reply named none of the browsed topics {list(_BROWSE_TOPIC_TOKENS)}")
    return fails


def _score_collector_runs(db: Database, _before: set[str], reply: str) -> list[str]:
    """ "How have your background collectors been doing" must read the cross-collector
    collector-runs index."""
    if not _dispatched(db, _LOG_READ, "memory", _COLLECTOR_RUNS):
        return ["did not log_read the collector-runs index for how the collectors are doing"]
    return []


def _score_no_fire(db: Database, _before: set[str], reply: str) -> list[str]:
    """A wistful aside about rereading old chats must fire NO log read, browse, or
    mutation — the false-positive guard for speakable log reads."""
    return [
        f"{tool} fired on a message that only mused, without asking for anything"
        for tool in (*_READ_TOOLS, *_ACTION_TOOLS)
        if tool_was_called(db, tool)
    ]


# ── Cases ─────────────────────────────────────────────────────────────────────
# Gated at 0.6 (the NL-dispatch convention) are the two capabilities that dispatch
# reliably at N=5 — reading a SYSTEM log and summarizing it: browse-results (5/5)
# and collector-runs (5/5).  The other two are report-only with the gap/confound
# documented in each case docstring and handed to #1522/#1524: user-messages-act
# (ambient recall substitutes for the log_read) and penny-messages-recall (browses
# instead of recalling what Penny said).  The no-fire log guard now gates at 0.6 —
# its over-firing is closed by the imperative-gating clause in CONVERSATION_PROMPT.
# Per-collector run introspection is no longer a dispatchable case: the ambient
# self-state header already carries each mechanism's last-run outcome, and the
# per-collection scoped verb (collector_run_history) retired with the read-surface
# reconciliation (#1580).


async def test_user_messages_act(chat_eval: ChatEval) -> None:
    """Report-only.  The out-of-window hobby is retrieved from ``user-messages``
    via a ``log_read``, and the model saves it to ``likes``.  The user-facing
    outcome (interest saved) is what matters; requiring the ``log_read``
    specifically over-fits to one mechanism.  A follow-up could gate on the
    OUTCOME (hobby in likes) instead of the tool."""
    await chat_eval(
        case_id="speak-logread-user-messages-act",
        message="look back over everything I've told you and save what I said I'm into to my likes",
        seed=_seed_hobby,
        score=_score_user_messages_act,
        min_pass_rate=None,
    )


async def test_penny_messages_recall(chat_eval: ChatEval) -> None:
    """Report-only — a #1524 vocabulary gap.  Asked to recall a past *Penny*
    suggestion, gpt-oss reaches for a fresh lookup (browse / read_similar over
    knowledge) instead of ``log_read("penny-messages")``: its instinct is to
    answer the topic, not to look back at what it said (baseline 0/3, all three
    browsing a browseable topic).  Even hammering the conversation framing ("dig
    back through our old messages") doesn't overcome it.  Kept report-only until
    the speakable vocabulary teaches "what did you say/suggest/recommend before"
    → penny-messages."""
    await chat_eval(
        case_id="speak-logread-penny-messages-recall",
        message="dig back through our old messages — what exactly did you tell me "
        "to use for my moss terrarium?",
        seed=_seed_suggestion,
        score=_score_penny_messages_recall,
        min_pass_rate=None,
    )


async def test_browse_results(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-logread-browse-results",
        message="what have you been looking up lately? give me the gist",
        seed=_seed_browse_history,
        score=_score_browse_results,
        min_pass_rate=0.6,
    )


async def test_collector_runs(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="speak-logread-collector-runs",
        message="how have your background collectors been doing lately?",
        seed=_seed_collector_activity,
        score=_score_collector_runs,
        min_pass_rate=0.6,
    )


async def test_no_fire(chat_eval: ChatEval) -> None:
    """The over-firing on a log-adjacent musing is now closed by the
    imperative-gating clause in ``Prompt.CONVERSATION_PROMPT``; gated at 0.6."""
    await chat_eval(
        case_id="speak-logread-no-fire",
        message="man, I really should reread our old chats sometime",
        score=_score_no_fire,
        min_pass_rate=0.6,
    )
