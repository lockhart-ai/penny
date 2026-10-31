"""Announcing a lookup, then replying with no call behind it (#2861).

The contract is `docs/eval-case-design.md`.

**The failure.** In chat, Penny often ends her thinking on the exact tool call she means
to make, then closes the turn with a reply that only PROMISES to look — "let me check the
logs…", "I'm on it, give me a sec!" — and never makes the call. Seen in prod, twice in a
row on the same question about why a scheduled job ran at an unexpected time. Both
completions ended cleanly (``finish_reason: stop``) — the model wrote a preamble and
closed the turn instead of emitting the call it had just named.

**The world**, replaying that shape with invented names and topics: a daily job
(``evening_tech_digest``) that ran on an 8am schedule for days, had that schedule changed
to the evening earlier the same day — through the real store path
(``update_collection_metadata``), so the mutation ledger carries the change and
``memory_get`` shows it under "Recent changes" — and then fired once under the new rule.
The user, having just seen that run, asks why it fired now instead of at 8am.

**The clock problem, and how this world works around it.** Every sample starts on the
suite's own pinned clock (``utils/clock.py``): Wednesday 14 October 2026, 14:00 where the
user is — deliberately neither a morning nor an evening hour, so no case can read "now" as
either without saying so itself. A job that already fired "in the evening, earlier today"
needs "evening" to be a real clock reading, not a label, so this world's invented user
lives in a REAL zone where that pinned instant reads as 7:00 PM local — ``America/Noronha``
(Fernando de Noronha, Brazil; UTC-2 year-round, no DST) — rather than the suite's default
``America/Los_Angeles``, where the same instant reads 2:00 PM and no evening firing could
have already happened today. Every other instant in the world (the job's creation, its
three plain mornings, the schedule change) is pinned the same way, in the same zone, so
nothing in the prompt disagrees with anything else about when things happened.

**The ask.** "awesome, looking good so far! but why did the digest run now? that's supposed
to run daily at 8am?" — after one or two ordinary exchanges that have nothing to do with the
job.

**The claim.** Her reply states the schedule's new hour — matched on a planted fact (the
evening hour), never on wording. The fact is reachable WITHOUT any tool call: the job's own
cadence line in the static Jobs section of the system prompt (``SelfStateHeader._cadence``,
via ``describe_schedule``) renders the schedule in plain words plus its next-run clause —
"daily at 7:00 PM (rule: `...`), next run tomorrow at 7:00 PM" — on every turn, whether or
not she reads anything else. ``memory_get`` on the job would show the same plain-words
schedule line, plus a generic "changed schedule" entry in the mutation ledger (the ledger's
own line names which FIELDS moved, not their new values).  Which of these a sample's turn
actually used, if any, is the route — measured in the tool sequence, never claimed.

**Measured, not claimed**: the tool sequence and the reply spread, same as every other
chat case. The thinking-ends-on-a-named-tool-with-no-call pattern itself isn't a
:class:`Feature` this round (that would be a harness change, out of scope for a
reproduce-first case) — it's read off the sample databases afterward, by hand, per the
run's report.

Every name, place, and topic here is invented; the repo is public.

REPORT-ONLY (``min_pass_rate=None``): this is the issue's first run, to see whether the
failure reproduces at all — the ceiling is the code owner's to set once the numbers are
read.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import update
from sqlmodel import col

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, MemoryRow, MutationEvent, PromptLog
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import _seeded_response, _wire_tool_call
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "plan-then-reply-schedule-change"

# The seeded rows' own model name — kept honest rather than invented, like every other
# world's seeding (see e.g. ``end_to_end.world``).
_SEEDED_MODEL = EVAL_MODELS[0]

# ── The invented world ────────────────────────────────────────────────────────

_USER_NAME = "Dana"
_USER_LOCATION = "Fernando de Noronha, Brazil"
# A REAL zone, UTC-2 with no DST, chosen so the suite's pinned "now" (below) reads as
# 7:00 PM local — see the module docstring's "clock problem" section.
_USER_TIMEZONE = "America/Noronha"
_LOCAL_TZ = ZoneInfo(_USER_TIMEZONE)

# The suite's own pinned starting instant (``utils/clock.py``'s ``SAMPLE_STARTS_AT``),
# read in THIS world's zone — computed rather than hand-copied, so this world never
# drifts from the clock the harness actually runs samples on.
_PINNED_UTC = datetime(2026, 10, 14, 21, 0, tzinfo=ZoneInfo("UTC"))
_NOW_LOCAL = _PINNED_UTC.astimezone(_LOCAL_TZ)
assert _NOW_LOCAL.strftime("%H:%M") == "19:00", (
    f"the pinned instant must read as 7:00 PM in {_USER_TIMEZONE}, it reads {_NOW_LOCAL!r} "
    "— the suite's pinned clock moved; this world's zone choice needs revisiting"
)

_JOB_CONTAINER = "evening_tech_digest"
_JOB_DESCRIPTION = "A daily tech-headlines digest from a couple of invented tech-news sites"
_SITE_A = "https://siliconcourier.test"
_SITE_B = "https://driftwire.test"
_WATCHED_FOR = "the day's top tech headlines"
_JOB_PROGRAM = (
    f"Check {_SITE_A} and {_SITE_B} for the day's top tech headlines. Pick the three "
    "most interesting ones and save them with their titles and a one-line summary."
)

# The job's two schedules — 8am for days, then the evening, changed earlier today. Built
# off the exact instants below rather than hand-typed, so the rule and the "now" the model
# is shown can never quietly disagree.
_CREATED_AT = datetime(2026, 10, 9, 9, 0, tzinfo=_LOCAL_TZ)
_MORNING_RUN_DAYS = (11, 12, 13)  # October mornings this job ran at 8am, before the change
_MORNING_HOUR = (8, 0)
_CHANGE_AT = datetime(2026, 10, 14, 9, 0, tzinfo=_LOCAL_TZ)  # earlier today
_EVENING_RUN_AT = _NOW_LOCAL  # "today", at the new hour — exactly the suite's pinned "now"


def _morning_run_at(day: int) -> datetime:
    hour, minute = _MORNING_HOUR
    return datetime(2026, 10, day, hour, minute, tzinfo=_LOCAL_TZ)


def _rrule_dtstart(when: datetime) -> str:
    """One instant as the ``DTSTART`` a schedule rule anchors on — floating local time,
    the form ``next_occurrence`` reads on the row's own user zone (#1932)."""
    return when.strftime("%Y%m%dT%H%M%S")


_MORNING_SCHEDULE = f"DTSTART:{_rrule_dtstart(_morning_run_at(9))}\nFREQ=DAILY"
_EVENING_SCHEDULE = f"DTSTART:{_rrule_dtstart(_EVENING_RUN_AT)}\nFREQ=DAILY"


def _digest_entry(day: str) -> str:
    """The digest's own content — the same invented headlines every cycle; never
    claimed, just plausible backstory for a job that has actually been running."""
    return (
        f"{day} digest — Silicon Courier: a small compiler project hits 1.0; "
        "Driftwire: a niche keyboard maker sells out its first run."
    )


def _cycle_calls(run_id: str, content: str) -> list[dict]:
    return [
        _wire_tool_call(
            f"{run_id}-0",
            "browse",
            {
                "items": [
                    {"query": _SITE_A, "looking_for": _WATCHED_FOR},
                    {"query": _SITE_B, "looking_for": _WATCHED_FOR},
                ]
            },
        ),
        _wire_tool_call(
            f"{run_id}-1",
            "collection_write",
            {"collection": _JOB_CONTAINER, "entries": [{"content": content}]},
        ),
        _wire_tool_call(f"{run_id}-2", PennyConstants.DONE_TOOL_NAME, {}),
    ]


# ── Seeding, through the stores' own chokepoints, pinned to exact instants ──────


def _pin_run_timestamp(db: Database, *, run_id: str, when: datetime) -> None:
    """Pin one seeded run's prompt rows (and any entry it wrote) to an EXACT instant —
    true of the seeded world, not of anything the model did. The same technique as
    ``end_to_end.world._backdate_run``, but taking an absolute instant rather than a day
    offset: this world needs an exact HOUR (8am vs. 7pm), which a day-preserving offset
    from the pinned clock's own hour cannot give."""
    with db.get_session() as session:
        session.exec(
            update(PromptLog).where(col(PromptLog.run_id) == run_id).values(timestamp=when)
        )
        session.exec(
            update(MemoryEntry)
            .where(col(MemoryEntry.last_written_by_run_id) == run_id)
            .values(created_at=when)
        )
        session.commit()


def _pin_mutation_timestamp(db: Database, *, run_id: str, when: datetime) -> None:
    """Pin the mutation-ledger event one seeded ``run_id`` produced to an exact instant —
    true of the seeded world (the job was created days ago; its schedule changed earlier
    today), not of anything the model did."""
    with db.get_session() as session:
        session.exec(
            update(MutationEvent).where(col(MutationEvent.run_id) == run_id).values(created_at=when)
        )
        session.commit()


def _pin_memory_created_at(db: Database, *, name: str, when: datetime) -> None:
    with db.get_session() as session:
        session.exec(update(MemoryRow).where(col(MemoryRow.name) == name).values(created_at=when))
        session.commit()


def _seed_cycle(
    db: Database, *, run_id: str, when: datetime, entry_key: str, entry_content: str
) -> None:
    """One completed, WORKED collector cycle — browse, write, done — pinned to an exact
    instant."""
    db.messages.log_prompt(
        model=_SEEDED_MODEL,
        messages=[],
        response=_seeded_response(tool_calls=_cycle_calls(run_id, entry_content)),
        agent_name=PennyConstants.MessageAuthor.COLLECTOR,
        run_id=run_id,
        run_target=_JOB_CONTAINER,
    )
    db.messages.set_run_outcome(run_id, RunOutcome.WORKED.value, "done: worked")
    require_memory(db, _JOB_CONTAINER).write_rule_decided(
        [EntryInput(key=entry_key, content=entry_content)],
        author=PennyConstants.MessageAuthor.COLLECTOR,
        run_id=run_id,
    )
    _pin_run_timestamp(db, run_id=run_id, when=when)


# ── The two ordinary exchanges before the ask — nothing to do with the job ──────

_EX0_USER = "morning! coffee's finally kicking in over here"
_EX0_REPLY = "ha, mine too — hope the day treats you kindly! anything on the docket?"
_EX1_USER = "nah just the usual, nothing exciting today"
_EX1_REPLY = "sounds like a good kind of quiet one. let me know if anything comes up!"


def _log_exchange(db: Database, user_text: str, reply: str) -> None:
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, user_text
    )
    assert ask_id is not None, "the seeded ask failed to log"
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        reply,
        parent_id=ask_id,
        recipient=TEST_SENDER,
    )


def _seed_world(db: Database) -> None:
    """Lay the world down: the user (in a zone where the pinned clock reads evening),
    the job created days ago on an 8am schedule, three plain mornings it ran on that
    schedule, the schedule changed to the evening earlier today, tonight's run under the
    new rule, and two ordinary exchanges before the ask."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1991-06-02",
    )
    created_run = seeded_run_id("created-the-digest")
    db.memories.create_collection(
        _JOB_CONTAINER,
        _JOB_DESCRIPTION,
        extraction_prompt=_JOB_PROGRAM,
        schedule=_MORNING_SCHEDULE,
        notify=True,
        created_by_run_id=created_run,
    )
    _pin_memory_created_at(db, name=_JOB_CONTAINER, when=_CREATED_AT)
    _pin_mutation_timestamp(db, run_id=created_run, when=_CREATED_AT)

    for day in _MORNING_RUN_DAYS:
        when = _morning_run_at(day)
        run_id = seeded_run_id(f"digest-morning-{day}")
        _seed_cycle(
            db,
            run_id=run_id,
            when=when,
            entry_key=when.date().isoformat(),
            entry_content=_digest_entry(when.date().isoformat()),
        )

    change_run = seeded_run_id("changed-to-evening")
    db.memories.update_collection_metadata(
        _JOB_CONTAINER, schedule=_EVENING_SCHEDULE, run_id=change_run
    )
    _pin_mutation_timestamp(db, run_id=change_run, when=_CHANGE_AT)

    evening_run_id = seeded_run_id("digest-evening-worked")
    _seed_cycle(
        db,
        run_id=evening_run_id,
        when=_EVENING_RUN_AT,
        entry_key=_EVENING_RUN_AT.date().isoformat(),
        entry_content=_digest_entry(_EVENING_RUN_AT.date().isoformat()),
    )
    db.memories.mark_collected(_JOB_CONTAINER)

    _log_exchange(db, _EX0_USER, _EX0_REPLY)
    _log_exchange(db, _EX1_USER, _EX1_REPLY)


# ── The wordings ──────────────────────────────────────────────────────────────

ASK = (
    "awesome, looking good so far! but why did the digest run now? that's supposed to "
    "run daily at 8am?"
)
ALSO_PHRASED = (
    "nice, that's a solid digest! but wait, why'd it run just now — isn't that supposed "
    "to go out at 8am?",
    "ooh that's great! quick question though, why did the digest just run? i thought it "
    "fired every morning at 8?",
    "love this! but hang on, why is the digest running now instead of 8am like usual?",
    "this is great stuff. one thing though — why did the digest run at this hour? i "
    "thought it was an 8am job?",
)

# The facts every wording carries: the job (by name), and the 8am expectation it's
# questioning. Never the new hour itself, never a tool name — the reply has to state the
# fact, not copy it from the ask.
_JOB_WORDS = ("digest",)
_MORNING_WORDS = ("8am", "8", "morning")

for _wording in (ASK, *ALSO_PHRASED):
    _folded = _wording.casefold()
    assert any(word in _folded for word in _JOB_WORDS), f"no job named: {_wording!r}"
    assert "8" in _folded, f"no 8am expectation named: {_wording!r}"
    for _leak in ("collection_configure", "update_collection_metadata", "evening_tech_digest"):
        assert _leak.casefold() not in _folded, f"a wording names {_leak!r}: {_wording!r}"


# ── The leak guard — enforced at import ──────────────────────────────────────
#
# The new hour must appear nowhere a sample could borrow it from without reading the
# job's own state: not the ask, not the ordinary exchanges, not the job's own
# description/program. The self-state's Jobs section (and ``memory_get``) are the
# legitimate source, and they're outside this guard on purpose — see the module docstring.

_EVENING_HOUR_RE = re.compile(
    r"\b7(?::00)?\s?p\.?m\.?\b|\b19:00\b|\bseven(?:\s+o'?clock)?\s+in\s+the\s+evening\b",
    re.IGNORECASE,
)


def _assert_no_leaked_hour() -> None:
    surfaces = (
        ASK,
        *ALSO_PHRASED,
        _EX0_USER,
        _EX0_REPLY,
        _EX1_USER,
        _EX1_REPLY,
        _JOB_DESCRIPTION,
        _JOB_PROGRAM,
    )
    for text in surfaces:
        assert not _EVENING_HOUR_RE.search(text), f"leaked the evening hour: {text!r}"


_assert_no_leaked_hour()


# ── The claim ─────────────────────────────────────────────────────────────────


def _reply_states_the_evening_hour(sample: SampleObservation, _world: World) -> Answer:
    """BEHAVIOUR: the reply states the job's new hour — the planted fact (#2861) —
    reachable in one look at the job's own Jobs-section line (no tool call required), or
    at ``memory_get``'s schedule line, whichever she reaches for; the route is measured,
    never claimed."""
    found = bool(_EVENING_HOUR_RE.search(sample.reply))
    return found, f"reply: {sample.reply!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_reply_states_the_jobs_new_hour_after_an_evening_firing(
    chat_eval: ChatEval, model: str
) -> None:
    """A job whose schedule moved from 8am to the evening earlier today, then fired once
    under the new rule: asked why it ran now instead of at 8am, the reply should state
    the new (evening) hour the schedule now carries.

    The failure this exists to catch (#2861): a reply that only announces an intent to
    check ("let me check the logs...") with no call behind it, and states no fact at all."""
    cohort = await chat_eval(
        case_id="plan-then-reply-schedule-change",
        behaviour=(
            "In the chat agent, asked why a scheduled job ran at an unexpected hour after "
            "its schedule was changed earlier the same day, Penny's reply states the job's "
            "new hour."
        ),
        area=Area.STANDING_JOBS,
        model=model,
        seed=_seed_world,
        world=World(
            name="plan-then-reply-schedule-change", pages=(), keeps=(), excludes=(), answers=()
        ),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — this is the issue's first reproduction run
        family=_FAMILY,
        timeout=180.0,
    )
    cohort.claim(
        "reply: states the job's new (evening) hour",
        _reply_states_the_evening_hour,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
