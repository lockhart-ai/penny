"""One refer-back case: two days later, past the history window and with the
plan's day folded into the conversation map's one ``memory_access`` line, the user
asks Penny to go ahead with a change they AGREED for one of her jobs (#3010).

The agreed plan is CONDITIONAL — once the ``weather-log`` feed is running
daily, add an opening step that reads ``weather-log`` AND the user's recent
garden messages first, keeping the job's existing prompt — and her reply
restated both legs.  What masks the plan is the VISIBLE recent window, not
naming: prod's ask named the job too.  The window shows two OTHER recent
edits to the same job (a bloom-calendar section, then a ``synthesize``
drafting step, each made on request and carried in the job's seeded
prompt), and — the look-alike — an exchange where she just read the
``weather-log`` by hand and answered what the week's frost risk means for
the beds.  So "the way we talked about" reads as what she can see, and the
plan's distinctive leg — the conversation messages — appears nowhere
visible: the plan exchange sits outside the 5-exchange history window, and
its whole day has folded into the map's collapsed-day line, which carries
only a count, never the plan's words.  The plan's condition is visibly met
(``weather-log`` holds dated daily entries, and she read it this week), so
the case measures recall, not the blocker.  The SAME case measures which
way she answers before and after the related-exchanges section,
unmodified: nothing here names a route or a tool.

REPORT-ONLY (``min_pass_rate=None``): the claims record what the turn did,
and the ceilings these runs propose are the code owner's to accept once the
numbers have been read.  Every list, entry and person is invented, because
the repo is public.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from penny.agents.conversation_map import ConversationMap
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import LogEntryInput
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import (
    _log_chat_step,
    _seeded_response,
    _tool_result_turn,
    _wire_tool_call,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    MechanismRecord,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.lookup_world import assert_seeded_turns_embedded
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "related-exchanges"

_INCOMING = PennyConstants.MessageDirection.INCOMING
_OUTGOING = PennyConstants.MessageDirection.OUTGOING
_PENNY = PennyConstants.MessageAuthor.PENNY

# The plan exchange sits two days back — behind the map's default two-day
# horizon, so the world WIDENS the horizon to three days and buries the day
# under the map's line ceiling instead: the exchange's whole day folds into
# the collapsed-day line, and the plan is on no 1:1 line.
_DAYS_BACK = 2
# The horizon widening, seeded as a runtime-config override — the same three-tier
# lookup /config writes, so the map's bounds are what they'd be in prod with the
# horizon one day wider.
_CHAT_MAP_DAYS_WIDENED = "3"

# The job the plan is about, with the prompt it already runs on — the two
# sections the VISIBLE recent exchanges added on request are already in it, so
# the world's state matches that history.  The claim's survival side reads this
# back off the registry.
_GARDEN_JOB = "garden-log"
_GARDEN_JOB_DESCRIPTION = "A weekly summary of the garden's watering and bloom notes."
_PROMPT_SUMMARY = (
    "Write a short summary of the garden's week from the notes in this "
    "collection: what was watered, what bloomed, and anything that needs "
    "attention before next week."
)
_BLOOM_SECTION = " Include a bloom calendar section listing what is flowering now."
_SYNTH_SECTION = (
    " Draft the summary with "
    "synthesize{query: <what the garden did this week>, sources: ['garden-log']}."
)
# The prompt after the bloom-calendar edit (the first visible configure), and
# after the synthesize edit (the second) — the state the job is seeded in.
_PROMPT_WITH_BLOOM = _PROMPT_SUMMARY + _BLOOM_SECTION
_GARDEN_JOB_PROMPT = _PROMPT_WITH_BLOOM + _SYNTH_SECTION
_GARDEN_JOB_SCHEDULE = "DTSTART:20261015T090000\nFREQ=WEEKLY"

# The plan's weather leg, seeded as a real log the way a live feed shows: dated
# daily entries — the plan's condition (running daily) visibly met — with the
# frost facts the look-alike exchange answered from.
_WEATHER_LOG = "weather-log"
_WEATHER_LOG_DESCRIPTION = "What the weather did, noted as it happens, every day."
_WEATHER_ENTRIES = (
    "monday — hard night frost early, the beds by the fence white until nine.",
    "tuesday — dry and breezy, no frost overnight, the soil worked easily.",
    "wednesday — light shower around four, a milder night, no frost expected.",
)

# The agreed plan — CONDITIONAL, each side short enough to render whole (well
# inside the block's 600-char side clip), both legs named in the words the user
# will later refer back to.
_PLAN_ASK = (
    "once the weather-log feed is running daily, add an opening step to the "
    "garden-log job that first reads the weather-log log and my recent "
    "messages about the garden, and keep the existing prompt as it is."
)
_PLAN_REPLY = (
    "yes — as soon as weather-log is running daily, each garden-log run gets an "
    "opening step that reads the weather-log log and your recent messages about "
    "the garden first, then writes its usual summary with the existing prompt "
    "unchanged. i'll add it then."
)
_PLAN_RUN = "garden-log-plan"

# The tokens the claims read.  The plan's distinctive leg is the conversation
# messages — prod's missed leg; the weather leg alone is the LOOK-ALIKE (she
# read that log by hand this week), so it does not count.  Both survive the
# model's own rewording.
_LEG_MESSAGES = "message"
_LEG_WEATHER = "weather"
# The job's own summary instructions the survival claim reads off.
_PROMPT_WATERING = "watered"
_PROMPT_BLOOM = "bloomed"

# ── The visible recent window ────────────────────────────────────────────────
#
# Two OTHER recent edits to the same job, made on request and carried in the
# job's seeded prompt, then the LOOK-ALIKE: she just read the weather-log by
# hand and answered from it.  "The way we talked about" reads as these — the
# plan's own leg appears in none of them.

_BLOOM_ASK = "the garden-log job should list what's flowering — add a bloom calendar section to it"
_BLOOM_REPLY = (
    "done — the job's prompt now includes a bloom calendar section listing what "
    "is flowering now, alongside the weekly summary it already writes."
)
_BLOOM_CONFIGURE = {"collection": _GARDEN_JOB, "prompt": _PROMPT_WITH_BLOOM}
_BLOOM_RESULT = (
    "Re-configured collection 'garden-log': the prompt now carries the bloom "
    "calendar section beside the weekly summary. This is what the store map "
    "shows for it."
)

_SYNTH_ASK = (
    "and have it draft the summary from its own notes — point it at synthesize over the collection"
)
_SYNTH_REPLY = (
    "done — the job now drafts its summary with synthesize over its own notes "
    "before saving, on top of the bloom calendar section."
)
_SYNTH_CONFIGURE = {"collection": _GARDEN_JOB, "prompt": _GARDEN_JOB_PROMPT}
_SYNTH_RESULT = (
    "Re-configured collection 'garden-log': the prompt now drafts the summary "
    "with synthesize over the collection's notes. This is what the store map "
    "shows for it."
)

_FROST_ASK = "look at the weather-log and tell me what this week's frost risk means for the beds"
_FROST_REPLY = (
    "read the weather-log — monday took a hard night frost that whitened the "
    "fence beds until nine, but tuesday and wednesday were frost-free and "
    "milder, so nothing needs covering before the weekend."
)
_FROST_RESULT = "3 entries from 'weather-log' (oldest first):\n" + "\n".join(_WEATHER_ENTRIES)

# Then 1-2 unrelated exchanges close the window.
_TAIL_USERS = (
    "remind me what time the pharmacy closes today",
    "do we still have the big blue water bottle anywhere?",
)
_TAIL_REPLIES = (
    "it closes at six on weekdays — you've got plenty of time",
    "it's in the hallway cupboard, behind the umbrella stand",
)

# The fold: unrelated exchanges on the plan's day, enough of them to push the
# map past its line ceiling so the OLDEST WHOLE DAY — the plan's — collapses
# into the one memory_access line.  Each side clears the short-exchange threshold,
# so every filler renders its own line and the ceiling is what folds the day.
_FOLD_FILLERS = 45
# A second day of fillers (yesterday), so the map holds more than one day and
# the ceiling collapse has somewhere to fold the oldest TO — the collapse needs
# a remaining day, and these are also what keeps the plan's day the oldest.
_DAY_MINUS_ONE_FILLERS = 10
_FILLER_USERS = (
    "the lemon drizzle cake caught at the edges again, so next time it goes in "
    "at a lower heat for ten minutes longer — write that on the recipe card",
    "we never settled whether the tiebreaker in that board game goes to the "
    "most remaining coins or the fewest — check the rulebook before friday",
    "that novel's last stretch finally explains the missing chapter, and the "
    "reveal lands better than i expected — i'll lend you my copy when i'm done",
    "the soup stock needs straining twice for a clear broth, and the second "
    "strain should cool first — worth remembering for the weekend batch",
    "the puzzle we started on sunday is missing one corner piece, so before "
    "we glue anything i want to turn the spare-room upside down for it",
)
_FILLER_REPLIES = (
    "noted — a lower heat and ten minutes longer, and the edges should stay "
    "pale; i'll keep it with the recipe card so the next bake starts right",
    "i'll look it up tonight and settle it before your friday session, so the "
    "tiebreaker is written down somewhere neither of us has to argue from memory",
    "glad the reveal paid off — a lending copy is a good excuse for a reread, "
    "and i'll remind you where you left the bookmark when the time comes",
    "a double strain with a cool-down between makes a clear broth every time, "
    "and i'll remember the weekend batch so the timing lines up with yours",
    "before any glue comes out, a proper hunt for that corner piece — spare "
    "room included — and i'll keep the puzzle boxed until it turns up",
)

# A decoy store the world carries so a blind read of the registry cannot answer
# the ask — the plan's legs exist only in the folded exchange.  Seeded through
# the WORLD declaration (``stores=``), the way the sibling cases lay decoys down:
# the driver's own seeding cites the collection's seeded run for both the create
# and the entry write, so the row's birth never reads as this turn's work
# (#2129 — an unstamped creation reads as a live run's on the mutation ledger).
DECOY_LISTS: tuple[SynthCollection, ...] = (
    SynthCollection(
        "pantry-restock",
        "Staples the pantry is short of.",
        ("Rice — the long-grain bag ran out.", "Lentils — red, half a bag left."),
    ),
)


def _seed_exchange(
    db: Database,
    user_text: str,
    reply_text: str,
    run_name: str,
    at: datetime,
) -> tuple[int, int]:
    """One exchange the way production stamps one: the reply carries the run id
    (#2630) and the parent link to the user message it answers — the stamps the
    exchange pairing reads."""
    run_id = seeded_run_id(run_name)
    ask_id = db.messages.log_message(_INCOMING, TEST_SENDER, user_text, timestamp=at)
    reply_id = db.messages.log_message(
        _OUTGOING,
        _PENNY,
        reply_text,
        parent_id=ask_id,
        run_id=run_id,
        timestamp=at + timedelta(minutes=1),
        recipient=TEST_SENDER,
    )
    assert ask_id is not None and reply_id is not None, (
        "the seeded exchange failed to log — nothing for the block to render"
    )
    return ask_id, reply_id


def _seed_tool_exchange(
    db: Database,
    user_text: str,
    reply_text: str,
    run_name: str,
    at: datetime,
    tool: str,
    arguments: dict,
    tool_result: str,
) -> tuple[int, int]:
    """One exchange whose reply a REAL tool call backed, recorded the way
    production records a turn: the run's promptlog carries the ask, the call,
    and the result — so the visible history renders the act beside the words
    (#2915), and the reply carries the same run id."""
    run_id = seeded_run_id(run_name)
    ask_id = db.messages.log_message(_INCOMING, TEST_SENDER, user_text, timestamp=at)
    assert ask_id is not None
    call_id = f"{run_id}-0"
    _log_chat_step(
        db,
        run_id=run_id,
        messages=[{"role": "user", "content": user_text}],
        response=_seeded_response(tool_calls=[_wire_tool_call(call_id, tool, arguments)]),
    )
    _log_chat_step(
        db,
        run_id=run_id,
        messages=[
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_wire_tool_call(call_id, tool, arguments)],
            },
            _tool_result_turn(call_id, tool_result),
        ],
        response=_seeded_response(reply_text),
    )
    reply_id = db.messages.log_message(
        _OUTGOING,
        _PENNY,
        reply_text,
        parent_id=ask_id,
        run_id=run_id,
        timestamp=at + timedelta(minutes=1),
        recipient=TEST_SENDER,
    )
    assert reply_id is not None
    return ask_id, reply_id


def _seed_world(db: Database) -> None:
    """The whole world: the job on its two-section prompt, the daily weather
    log, the VISIBLE recent window (two job edits, the weather look-alike,
    two unrelated closes), the conditional plan exchange two days back, and
    the fold fillers — with the horizon widened so the plan's day folds
    rather than leaving the map outright."""
    db.config.set(
        "CHAT_MAP_DAYS",
        _CHAT_MAP_DAYS_WIDENED,
        "the plan's day sits two days back and must render as the map's folded-day line",
    )
    db.memories.create_collection(
        _GARDEN_JOB,
        _GARDEN_JOB_DESCRIPTION,
        extraction_prompt=_GARDEN_JOB_PROMPT,
        schedule=_GARDEN_JOB_SCHEDULE,
        notify=True,
        created_by_run_id=seeded_run_id(_GARDEN_JOB),
    )
    db.memories.create_log(_WEATHER_LOG, _WEATHER_LOG_DESCRIPTION)
    require_memory(db, _WEATHER_LOG).append(
        [LogEntryInput(content=entry) for entry in _WEATHER_ENTRIES], author="user"
    )
    clock_now = db.clock.now()
    # The visible recent window, newest last: the two job edits, the weather
    # look-alike, then the unrelated closes.
    for index, (user_text, reply_text) in enumerate(zip(_TAIL_USERS, _TAIL_REPLIES, strict=True)):
        _seed_exchange(
            db,
            user_text,
            reply_text,
            f"recent-tail-{index}",
            clock_now - timedelta(minutes=4 - index),
        )
    _seed_tool_exchange(
        db,
        _FROST_ASK,
        _FROST_REPLY,
        "frost-lookalike",
        clock_now - timedelta(minutes=10),
        "log_read",
        {"log": _WEATHER_LOG},
        _FROST_RESULT,
    )
    _seed_tool_exchange(
        db,
        _SYNTH_ASK,
        _SYNTH_REPLY,
        "synth-edit",
        clock_now - timedelta(minutes=20),
        "collection_configure",
        _SYNTH_CONFIGURE,
        _SYNTH_RESULT,
    )
    _seed_tool_exchange(
        db,
        _BLOOM_ASK,
        _BLOOM_REPLY,
        "bloom-edit",
        clock_now - timedelta(minutes=30),
        "collection_configure",
        _BLOOM_CONFIGURE,
        _BLOOM_RESULT,
    )

    plan_at = clock_now - timedelta(days=_DAYS_BACK)
    _seed_exchange(db, _PLAN_ASK, _PLAN_REPLY, _PLAN_RUN, plan_at)
    for index in range(_FOLD_FILLERS):
        _seed_exchange(
            db,
            _FILLER_USERS[index % len(_FILLER_USERS)],
            _FILLER_REPLIES[index % len(_FILLER_REPLIES)],
            f"fold-filler-{index}",
            plan_at + timedelta(minutes=5 + index),
        )
    yesterday_at = clock_now - timedelta(days=1)
    for index in range(_DAY_MINUS_ONE_FILLERS):
        _seed_exchange(
            db,
            _FILLER_USERS[(index + 2) % len(_FILLER_USERS)],
            _FILLER_REPLIES[(index + 2) % len(_FILLER_REPLIES)],
            f"day-minus-one-filler-{index}",
            yesterday_at + timedelta(minutes=10 + index),
        )


def _assert_the_plan_world(penny: Penny) -> None:
    """The case's premise, asserted before the turn runs: the job starts on its
    two-section prompt, the weather log holds its daily entries, every seeded
    message carries a vector, and — the fold — the map renders the plan's day
    as its collapsed memory_access line with NONE of the plan's words on any 1:1
    line."""
    db = penny.db
    row = db.memories.get(_GARDEN_JOB)
    assert row is not None, f"{_GARDEN_JOB} must be in the registry"
    assert row.extraction_prompt == _GARDEN_JOB_PROMPT, (
        f"{_GARDEN_JOB} must start on the seeded prompt, it holds {row.extraction_prompt!r}"
    )
    assert row.schedule == _GARDEN_JOB_SCHEDULE and row.notify and not row.archived
    weather = require_memory(db, _WEATHER_LOG).read_all()
    assert len(weather) == len(_WEATHER_ENTRIES), (
        "the weather log must hold its daily entries — the plan's condition, visibly met"
    )
    assert_seeded_turns_embedded(db)
    map_anchor = db.messages.session_anchor(TEST_SENDER)
    map_render = ConversationMap(db, TEST_SENDER, penny.config.runtime, map_anchor).render()
    fold_marker = "exchanges — memory_access{"
    fold_lines = [line for line in map_render.splitlines() if fold_marker in line]
    assert fold_lines, (
        f"the plan's day must render as the map's collapsed memory_access line, the map "
        f"rendered: {map_render!r}"
    )
    assert _LEG_WEATHER not in fold_typography(map_render), (
        "the plan's words must sit on no map line — the fold carries a count, never text"
    )


def _mechanism(sample: SampleObservation, name: str) -> MechanismRecord | None:
    return next((row for row in sample.mechanisms if row.name == name), None)


def _turn_text(sample: SampleObservation) -> str:
    """The two loci the plan's legs can reach — the job's configured prompt and
    her reply, whichever the turn wrote."""
    garden = _mechanism(sample, _GARDEN_JOB)
    return fold_typography(f"{garden.program if garden else ''}\n{sample.reply}")


def _the_messages_leg_is_named(sample: SampleObservation, _world: World) -> Answer:
    """The plan's distinctive leg — the user's recent garden messages — reaches
    her reply or the configured prompt.  The weather leg alone is the
    look-alike (she read that log by hand this week), so it does not count."""
    if _LEG_MESSAGES in _turn_text(sample):
        return True, None
    return False, "the plan's messages leg never named in the reply or the job's prompt"


def _the_weather_leg_is_named(sample: SampleObservation, _world: World) -> Answer:
    """The plan's weather leg, named in either locus — recorded separately
    because it is the leg the visible look-alike could hand her without the
    plan."""
    if _LEG_WEATHER in _turn_text(sample):
        return True, None
    return False, "the plan's weather leg never named in the reply or the job's prompt"


def _both_legs_are_named(sample: SampleObservation, _world: World) -> Answer:
    """The whole plan — both legs — made it to a locus she will act from."""
    text = _turn_text(sample)
    missing = [
        leg
        for leg, token in (
            ("the weather-log log", _LEG_WEATHER),
            ("the user's recent messages", _LEG_MESSAGES),
        )
        if token not in text
    ]
    if not missing:
        return True, None
    return False, f"the plan's legs never named: {', '.join(missing)}"


def _the_existing_prompt_survives(sample: SampleObservation, _world: World) -> Answer:
    """The job's own summary instructions survive whatever the turn wrote: an
    unconfigured job keeps its prompt as seeded, and a configured one keeps
    them beside whatever the plan added."""
    garden = _mechanism(sample, _GARDEN_JOB)
    if garden is None:
        return False, "the garden-log job is gone"
    program = fold_typography(garden.program or "")
    lost = [
        what
        for what, token in (
            ("what was watered", _PROMPT_WATERING),
            ("what bloomed", _PROMPT_BLOOM),
        )
        if token not in program
    ]
    if not lost:
        return True, None
    return False, f"the job's own summary instructions dropped: {', '.join(lost)}"


def _every_wording_says_go_ahead(*asks: str) -> None:
    """Each wording carries the fact the claims depend on: an agreed change, to
    be made now — and none of them names either leg (that is the fact the
    folded exchange carries)."""
    for ask in asks:
        folded = fold_typography(ask)
        assert "garden" in folded
        assert _LEG_WEATHER not in folded
        assert _LEG_MESSAGES not in folded


_PLAN_ASK_LIVE = (
    "ok so now you've got what you need to update the garden job the way we talked about?"
)
_PLAN_PHRASINGS = (
    "so you're all set — go ahead and update the garden job the way we discussed",
    "now that the feed is daily, can you make that garden job change we agreed on?",
    "go ahead with the garden job update like we planned",
    "you've got everything now — make the garden job change we talked about the other night",
)

_PLAN_WORLD = World(
    name=(
        "a garden job two visible edits and a hand-read weather log behind it, "
        "and a two-day-old agreed plan folded out of the map"
    ),
    pages=(),
    keeps=(),
    excludes=(),
    stores=DECOY_LISTS,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_agreed_job_plan_referred_back_two_days_later_is_made_from_the_plan(
    chat_eval: ChatEval, model: str
) -> None:
    """Two days later, with the plan's day folded into the map's one memory_access
    line and the visible window full of look-alikes, the user asks Penny to go
    ahead with the change they agreed — the plan's messages leg is the fact
    the turn has to recover, and the job's own prompt must survive whatever
    she writes."""
    _every_wording_says_go_ahead(_PLAN_ASK_LIVE, *_PLAN_PHRASINGS)
    cohort = await chat_eval(
        case_id="job-plan-referred-back-two-days-later",
        behaviour=(
            "In the chat agent, when the user refers back — two days later, past the "
            "conversation window, with the plan's day folded into the conversation "
            "map's collapsed memory_access line and the visible window full of look-alike "
            "job edits — to a change they agreed for one of Penny's jobs, Penny makes "
            "the change from the agreed plan: the plan's distinctive leg — the user's "
            "recent messages — is named in her reply or in the job's configured "
            "prompt, and the job's existing prompt survives."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        prepare=_assert_the_plan_world,
        world=_PLAN_WORLD,
        ask=_PLAN_ASK_LIVE,
        also_phrased=_PLAN_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,
    )
    # STORE — everything but the job the ask invites her to change survives.
    cohort.assert_every_mechanism_but_the_rounds_own_survives(_GARDEN_JOB)
    cohort.assert_what_the_store_held_survives()
    # STORE — the ask's substance, in whichever locus the turn wrote it.
    cohort.claim(
        "state: the plan's messages leg — recent garden messages — is named, reply or prompt",
        _the_messages_leg_is_named,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the plan's weather-log leg is named, reply or the job's prompt",
        _the_weather_leg_is_named,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: both of the plan's legs are named, reply or the job's prompt",
        _both_legs_are_named,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: if she configured, the job's existing prompt survives",
        _the_existing_prompt_survives,
        SpecCategory.STORE,
    )
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
