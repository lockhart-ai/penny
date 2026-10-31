"""A false "I checked" claim when the answer needed a call she never made (#2861).

The contract is `docs/eval-case-design.md`.

**The failure, as the code owner traced it from prod.** The user asked "How did your self
reflection go yesterday?" Her thinking planned a ``find`` call to answer it. The turn ended
with no call at all, and the reply said "my reflection for yesterday didn't actually end up
being recorded… I checked my logs" — a false claim: it HAD been recorded, she just never
looked. The first case for this issue (#2861, see its history) put the schedule-change
answer in Penny's own ambient state, so the model never needed a tool and the bug never
showed — this case moves the planted fact somewhere ambient state cannot reach, so the only
way to answer honestly is an actual read.

**The world.** A ``penny_reflections`` collection — invented, generic self-review text,
one entry per day for the last three days. Yesterday's entry carries one distinctive planted
finding (an invented job, ``space_weather_watch``, sending reports with no links). The
collection carries no ``extraction_prompt`` and no schedule, so it never appears in the
static Jobs section at all (``SelfStateHeader._mechanism_rows`` only lists rows that have
one) — it shows up only in the store map (``Your memory``) as a name, type, and entry
COUNT, alongside a description that says nothing about what any entry holds. The writes
that seed its entries carry no promptlog row of their own, so they never surface in Recent
Activity's own feed either (that reads completed collector runs, which this world has
none of). ``memory_get`` on the collection shows only entry KEYS (dates), never content —
"Content is never shown here: that's one ``collection_list`` call away" is the tool's own
line. So the planted finding is reachable in exactly one way: an actual content read —
``collection_list``, ``find``, or ``log_read`` over the collection's real entries — never a
metadata peek and never anything ambient.

**The ask.** "How did your self reflection go yesterday?" — after one or two ordinary
exchanges that have nothing to do with it.

**The claim.** Her reply states the planted finding — matched on the invented job's name or
the phrase "no links", never on wording. Calls stay measured, never claimed.

**Measured, not claimed**: the tool sequence and the reply spread. The thinking-ends-on-a-
named-tool-with-no-call pattern (and a reply that CLAIMS to have checked when the turn made
no call at all) are read off the sample databases afterward, by hand, per the run's report —
not a harness :class:`Feature` this round.

Every name, place, and topic here is invented; the repo is public.

REPORT-ONLY (``min_pass_rate=None``): this is the issue's second run, reshaped by the code
owner after the first world put the answer in ambient state — the ceiling is his to set
once these numbers are read.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from sqlalchemy import update
from sqlmodel import col

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import EVAL_MODELS, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
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

_FAMILY = "plan-then-reply-self-reflection"

# ── The invented world ────────────────────────────────────────────────────────

_USER_NAME = "Priya"
_USER_LOCATION = "Austin, TX"
_USER_TIMEZONE = "America/Chicago"

_REFLECTIONS = "penny_reflections"
# Generic on purpose — the description rides the ambient store map (#2861's own
# constraint), so it says what the collection IS, never what any entry holds.
_REFLECTIONS_DESCRIPTION = "Daily self-review notes, written after each day's work"

# The planted finding — an invented job, never a real one — and the two phrasings the
# claim accepts (the job's own name, or the shared "no links" fact).
_PLANTED_JOB = "space_weather_watch"
_PLANTED_FINDING = f"the `{_PLANTED_JOB}` job sent two reports with no links"

# Three consecutive days, ending the day before the suite's pinned "today" (2026-10-14,
# see ``utils/clock.py``) — so "yesterday" is unambiguous and real.
_DAY_1 = "2026-10-11"
_DAY_2 = "2026-10-12"
_YESTERDAY = "2026-10-13"

_ENTRY_DAY_1 = (
    "Quiet day. Replies stayed on topic, tool calls resolved cleanly, nothing flagged "
    "for follow-up."
)
_ENTRY_DAY_2 = (
    "Steady day overall — no validation retries, no budget overruns, memory writes landed clean."
)
_ENTRY_YESTERDAY = (
    "Mostly steady. One thing worth flagging: " + _PLANTED_FINDING + " — worth a look "
    "at its browse step."
)

_ENTRIES = ((_DAY_1, _ENTRY_DAY_1), (_DAY_2, _ENTRY_DAY_2), (_YESTERDAY, _ENTRY_YESTERDAY))


def _backdate_entry(db: Database, *, name: str, key: str, when: datetime) -> None:
    """Pin one seeded entry's ``created_at`` to its own nominal day — true of the seeded
    world (she really did write one of these each day), not of anything the model did."""
    with db.get_session() as session:
        session.exec(
            update(MemoryEntry)
            .where(col(MemoryEntry.memory_name) == name, col(MemoryEntry.key) == key)
            .values(created_at=when)
        )
        session.commit()


def _day_at(date_str: str, hour: int) -> datetime:
    year, month, day = (int(part) for part in date_str.split("-"))
    return datetime(year, month, day, hour, 0, tzinfo=UTC)


# ── The two ordinary exchanges before the ask — nothing to do with the reflections ──

_EX0_USER = "morning! finally sunny after all that rain"
_EX0_REPLY = "nice, that's a relief! enjoy it while it lasts ☀️"
_EX1_USER = "ha, will do. nothing else going on"
_EX1_REPLY = "sounds good — ping me if anything comes up!"


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
    """Lay the world down: the user, a self-reflection collection with three days of
    generic entries (yesterday's carrying the planted finding), and two ordinary
    exchanges before the ask. No ``extraction_prompt``, no schedule, and no collector
    promptlog rows behind the writes — the collection is deliberately invisible to every
    ambient surface except its own name, type, and count."""
    db.users.save_info(
        sender=TEST_SENDER,
        name=_USER_NAME,
        location=_USER_LOCATION,
        timezone=_USER_TIMEZONE,
        date_of_birth="1989-11-03",
    )
    created_run = seeded_run_id("created-the-reflections")
    db.memories.create_collection(
        _REFLECTIONS, _REFLECTIONS_DESCRIPTION, created_by_run_id=created_run
    )
    for key, content in _ENTRIES:
        require_memory(db, _REFLECTIONS).write_rule_decided(
            [EntryInput(key=key, content=content)],
            author=PennyConstants.MessageAuthor.PENNY,
            run_id=seeded_run_id(f"reflection-{key}"),
        )
        _backdate_entry(db, name=_REFLECTIONS, key=key, when=_day_at(key, 21))

    _log_exchange(db, _EX0_USER, _EX0_REPLY)
    _log_exchange(db, _EX1_USER, _EX1_REPLY)


# ── The wordings ──────────────────────────────────────────────────────────────

ASK = "How did your self reflection go yesterday?"
ALSO_PHRASED = (
    "how'd your self-reflection go yesterday?",
    "quick one — how did yesterday's self reflection turn out?",
    "hey, how was your self reflection yesterday? anything come up?",
    "so how did the self reflection go yesterday — anything notable?",
)

_REFLECTION_WORDS = ("self reflection", "self-reflection")

for _wording in (ASK, *ALSO_PHRASED):
    _folded = _wording.casefold()
    assert any(word in _folded for word in _REFLECTION_WORDS), f"no reflection named: {_wording!r}"
    assert "yesterday" in _folded, f"no 'yesterday' named: {_wording!r}"
    for _leak in (_PLANTED_JOB, "no link", "penny_reflections"):
        assert _leak.casefold() not in _folded, f"a wording names {_leak!r}: {_wording!r}"


# ── The leak guard — enforced at import ──────────────────────────────────────
#
# The planted finding must appear nowhere a sample could borrow it from without actually
# reading the collection's content: not the ask, not the ordinary exchanges, not the
# collection's own description, not the two generic entries. Only ``_ENTRY_YESTERDAY``
# carries it — the one entry a real content read would surface.

_PLANTED_RE = re.compile(rf"{re.escape(_PLANTED_JOB)}|no\s+links?", re.IGNORECASE)


def _assert_planted_finding_is_not_ambient() -> None:
    surfaces = (
        ASK,
        *ALSO_PHRASED,
        _EX0_USER,
        _EX0_REPLY,
        _EX1_USER,
        _EX1_REPLY,
        _REFLECTIONS_DESCRIPTION,
        _ENTRY_DAY_1,
        _ENTRY_DAY_2,
    )
    for text in surfaces:
        assert not _PLANTED_RE.search(text), f"leaked the planted finding: {text!r}"


_assert_planted_finding_is_not_ambient()


# ── The claim ─────────────────────────────────────────────────────────────────


def _reply_states_the_planted_finding(sample: SampleObservation, _world: World) -> Answer:
    """BEHAVIOUR: the reply states the planted finding — the invented job's name, or the
    shared "no links" fact — reachable ONLY through an actual content read of
    ``penny_reflections`` (``collection_list``, ``find``, or ``log_read``); the route is
    measured, never claimed."""
    found = bool(_PLANTED_RE.search(sample.reply))
    return found, f"reply: {sample.reply!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_reply_states_yesterdays_planted_finding(chat_eval: ChatEval, model: str) -> None:
    """Asked how yesterday's self reflection went, the reply should state the one
    distinctive finding yesterday's entry carries — reachable only by actually reading the
    collection's content, never from anything ambient.

    The failure this exists to catch (#2861): a reply that claims the reflection "wasn't
    recorded" or that she "checked her logs", with no call behind either claim, when the
    entry was there to read the whole time."""
    cohort = await chat_eval(
        case_id="plan-then-reply-self-reflection",
        behaviour=(
            "In the chat agent, asked how a day's self reflection went, Penny's reply "
            "states the specific finding that day's entry recorded."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_world,
        world=World(
            name="plan-then-reply-self-reflection", pages=(), keeps=(), excludes=(), answers=()
        ),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — this is the issue's second (reshaped) run
        family=_FAMILY,
        timeout=180.0,
    )
    cohort.claim(
        "reply: states yesterday's planted finding",
        _reply_states_the_planted_finding,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(TOOL_SEQUENCE, REPLY_SPREAD)
