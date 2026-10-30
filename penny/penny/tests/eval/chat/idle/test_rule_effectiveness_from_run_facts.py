"""Has a standing rule actually helped? The answer is in the run facts (#2843).

The contract is `docs/eval-case-design.md`.

**ONE case**, ``rule-effectiveness-from-run-facts``.  A standing rule tells Penny never to
hand over a link in a reply or a job update unless she actually opened and read that exact
page THIS run — a concrete behaviour correction, not a parameter she tunes on her own.
Before the world's "now", a job's cycles repeatedly discard a drafted reply for inventing a
link (the ``hallucinated_urls`` validator reject-and-reroll, #2702); after the rule, none
do.  The user asks whether the rule actually helped, and how many times it had to step in
before versus since.  The answer is a FACT, not a memory of having written the rule: the
job's own ``promptlog`` rows carry the discard before the cutoff and none after, reachable
through exactly the surfaces #2843 lands — the ``run-health`` facade (``log_read``) and
``run_list(source=<job>)``/``run_get`` (``render_run_calls``), both newly carrying the
per-run facts clause this ticket adds.

**What this case asserts.**  One fact, SEEDED and never claimed by any turn of this case's
own: exactly five early cycles each carry the discard, and the three recent cycles carry
none — the world's own premise, re-checked before the live turn runs (``_premise``), never
something a sample could make true or false.  The measured BEHAVIOUR is the reply, as TWO
separate claims over a COUNT the seeded facts carry — never a word the claim asserts was
said, which is wording and the eval design forbids it:

- the reply states the standalone count before the rule — "5" or "five" — as its own token
- the reply SEPARATELY states the standalone count since the rule — "0", "none", "zero", or
  "no" — so a sample can state one half right and the other wrong, and each is read apart

Neither the five nor any digit it could be confused for appears ANYWHERE else in the
wordings or the seeded world — the dates are chosen so no date string, and no other world
text, carries the digit 5 (``_assert_no_leaked_five``, enforced at import) — so a reply that
states it can only have counted the seeded facts, never echoed something it was given.

What is NOT claimed: which tool she calls, how many turns she takes, or anything about her
reply beyond the two counts above.  Which read verb she reaches for
(``log_read('run-health')``, ``run_list(source=<job>)``, or the self-state header's own
per-job summary) is a ROUTE, so it is measured in the tool sequence, never claimed.

REPORT-ONLY (``min_pass_rate=None``): #2843 is new capability, not a regression fix, so there
is no prior baseline to beat — the ceiling here is the code owner's to set once the numbers
are read.  The job, its rule, and the page it watches are invented; the repo is public.
"""

from __future__ import annotations

import re
from datetime import timedelta

import pytest
from sqlalchemy import update
from sqlmodel import col, select

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryEntry, PromptLog
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, seeded_run_id
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.end_to_end.world import (
    StandingJob,
    _wire_tool_call,
    seed_job_cycle,
    seed_standing_jobs,
)
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

_FAMILY = "rule-effectiveness-from-run-facts"

RULES = PennyConstants.MEMORY_RULES_COLLECTION

_JOB = StandingJob(
    container="porch-plants",
    url="https://example-garden-supply.test/porch-plants",
    watched_for="restock dates and prices",
    description="watch porch-plant restocks",
    schedule="FREQ=DAILY",
    notify=True,
)

_RULE_KEY = "no_invented_links"
_RULE_CONTENT = (
    "Never hand over a link in a reply or a job update unless you actually opened and read "
    "that exact page this run — never reconstruct or guess one."
)

# Each discarded cycle also closes cleanly (the reroll landed, #2702) — a discard is a
# REJECTED draw along the way, never the cycle's own outcome.  Early = before the rule
# (carries the discard, exactly FIVE cycles); recent = after it (clean, three cycles).
# The pinned clock's "now" is 2026-10-14, so these offsets land on Oct 1, 2, 3, 4, 6 (early),
# Oct 9 (the rule), and Oct 11, 12, 13 (recent) — no calendar date in this world carries the
# digit 5, so a reply stating "5"/"five" can only have come from counting the seeded facts.
_EARLY_DAYS_AGO = (13, 12, 11, 10, 8)
_RECENT_DAYS_AGO = (3, 2, 1)
_RULE_DAYS_AGO = 5


def _cycle_calls(run_id: str) -> list[dict]:
    return [
        _wire_tool_call(f"{run_id}-0", "browse", {"queries": [_JOB.url]}),
        _wire_tool_call(
            f"{run_id}-1",
            "collection_write",
            {"collection": _JOB.container, "entries": [{"content": "restocked — in stock now"}]},
        ),
        _wire_tool_call(f"{run_id}-2", PennyConstants.DONE_TOOL_NAME, {}),
    ]


def _seed_cycle(db: Database, *, run_name: str, days_ago: int, discarded: bool) -> str:
    """One completed cycle, optionally carrying a discarded hallucinated-link draw before the
    kept one landed — a second ``promptlog`` row sharing this cycle's run id, backdated
    alongside it (#2843: the facts clause tallies ``discard_reason`` across every row of a
    run, never just the last)."""
    run_id = seeded_run_id(run_name)
    if discarded:
        discard_row_id = db.messages.log_prompt(
            model="m",
            messages=[],
            response={"choices": []},
            agent_name=PennyConstants.MessageAuthor.COLLECTOR,
            run_id=run_id,
            run_target=_JOB.container,
        )
        assert discard_row_id is not None
        with db.get_session() as session:
            session.exec(
                update(PromptLog)
                .where(col(PromptLog.id) == discard_row_id)
                .values(discard_reason="hallucinated_urls")
            )
            session.commit()
    seed_job_cycle(
        db,
        container=_JOB.container,
        run_name=run_name,
        calls=_cycle_calls(run_id),
        outcome=RunOutcome.WORKED,
        summary="restock checked",
        days_ago=days_ago,
    )
    return run_id


def _backdate_rule(db: Database, *, key: str, days_ago: int) -> None:
    """Push the rule entry's own ``created_at`` back — true of the seeded world, not of
    anything the model did (mirrors ``world._backdate_run``'s technique).  Dates the rule
    between the early and recent cycles, so a model that reads the ``rules`` entry's own
    timestamp sees it land exactly where the discard tally changes."""
    stamp = db.clock.now() - timedelta(days=days_ago)
    with db.get_session() as session:
        session.exec(
            update(MemoryEntry)
            .where(col(MemoryEntry.memory_name) == RULES, col(MemoryEntry.key) == key)
            .values(created_at=stamp)
        )
        session.commit()


def _seed_world(db: Database) -> None:
    """The job standing, its early cycles each discarding a hallucinated link, the rule
    standing (dated between the two halves), and its recent cycles clean."""
    seed_standing_jobs(_JOB)(db)
    require_memory(db, RULES).write_rule_decided(
        [EntryInput(key=_RULE_KEY, content=_RULE_CONTENT)],
        author="chat",
        run_id=seeded_run_id("added-the-rule"),
    )
    _backdate_rule(db, key=_RULE_KEY, days_ago=_RULE_DAYS_AGO)
    for i, days_ago in enumerate(_EARLY_DAYS_AGO):
        _seed_cycle(db, run_name=f"porch-early-{i}", days_ago=days_ago, discarded=True)
    for i, days_ago in enumerate(_RECENT_DAYS_AGO):
        _seed_cycle(db, run_name=f"porch-recent-{i}", days_ago=days_ago, discarded=False)


def _premise() -> Preparer:
    """The seeded split, re-checked before the live turn runs: every one of the five early
    cycles carries the discard, none of the three recent cycles does — the fact the claims
    below measure whether the reply states, never something a sample could make true."""

    def probe(penny: Penny) -> None:
        db = penny.db
        early_ids = [seeded_run_id(f"porch-early-{i}") for i in range(len(_EARLY_DAYS_AGO))]
        recent_ids = [seeded_run_id(f"porch-recent-{i}") for i in range(len(_RECENT_DAYS_AGO))]
        with db.get_session() as session:
            rows = session.exec(
                select(PromptLog).where(col(PromptLog.run_id).in_(early_ids + recent_ids))
            ).all()
            reasons = {row.run_id: row.discard_reason for row in rows if row.discard_reason}
        assert len(early_ids) == 5, "the premise is exactly five early discards"
        assert all(rid in reasons for rid in early_ids), (
            f"every early cycle must carry the discard, reasons seeded: {reasons!r}"
        )
        assert not any(rid in reasons for rid in recent_ids), (
            f"no recent cycle may carry the discard, reasons seeded: {reasons!r}"
        )

    return probe


# ── The leak guard — enforced at import ──────────────────────────────────────

_FIVE_TOKEN_RE = re.compile(r"\b(?:5|five)\b", re.IGNORECASE)
_NONE_TOKEN_RE = re.compile(r"\b(?:0|none|zero|no)\b", re.IGNORECASE)

ASK = (
    "has that rule i added about made-up links actually helped? how many times was it "
    "throwing them out before vs since?"
)
ALSO_PHRASED = (
    "did the no-fake-links rule actually make a difference — how many times did it catch a "
    "made-up link before, compared to since?",
    "i want real numbers: how many times has the made-up-link rule had to step in before "
    "versus after it went into effect?",
    "before that rule went in, how many times did it throw out a fake link, and how many "
    "times has it happened since?",
    "has the fake-link rule been doing anything? give me the count from before it existed "
    "versus after.",
)


def _assert_no_leaked_five() -> None:
    """ENFORCED at import: the digit 5 (or the word "five") appears nowhere in the wordings
    or the world's own visible text — dates, the rule's content, the job's own description —
    so a reply that states "5"/"five" can only have counted the seeded discards, never
    echoed it from somewhere else.  No wording carries any digit at all, since the ask must
    want the number without supplying it."""
    surfaces = (ASK, *ALSO_PHRASED, _RULE_CONTENT, _JOB.description, _JOB.url, _JOB.watched_for)
    for text in surfaces:
        assert not _FIVE_TOKEN_RE.search(text), f"leaked the digit/word five: {text!r}"
    for wording in (ASK, *ALSO_PHRASED):
        assert not re.search(r"\d", wording), f"a wording must carry no digit at all: {wording!r}"
    all_days = (*_EARLY_DAYS_AGO, _RULE_DAYS_AGO, *_RECENT_DAYS_AGO)
    assert len(_EARLY_DAYS_AGO) == 5, "exactly five early cycles is the premise itself"
    # The calendar dates these offsets land on (computed off the pinned clock's 2026-10-14)
    # must carry no "5" digit anywhere, in either the day-of-month or the offset itself,
    # since an offset is never rendered but a mistake here would be a false negative on the
    # guard rather than a real check.
    for offset in all_days:
        day_of_month = 14 - offset
        assert "5" not in str(day_of_month), f"a seeded date lands on the 5th: offset {offset}"


_assert_no_leaked_five()


# ── The claims ────────────────────────────────────────────────────────────────


def _reply_states_five_before(sample: SampleObservation, _world: World) -> Answer:
    """BEHAVIOUR: the reply names the standalone COUNT the seeded facts show for before the
    rule — five early cycles, each carrying one hallucinated-link discard — never a word
    asserted as said, a count read off the facts (the leak guard above is what makes this
    legible: "5"/"five" appears nowhere else a sample could borrow it from)."""
    found = bool(_FIVE_TOKEN_RE.search(sample.reply))
    return found, f"reply: {sample.reply!r}"


def _reply_states_none_since(sample: SampleObservation, _world: World) -> Answer:
    """BEHAVIOUR: the reply SEPARATELY names the standalone count the seeded facts show for
    since the rule — none of the three recent cycles discarded anything.  Read apart from the
    "before" claim so a sample that gets one half right and the other wrong shows both."""
    found = bool(_NONE_TOKEN_RE.search(sample.reply))
    return found, f"reply: {sample.reply!r}"


_MEASURED = (TOOL_SEQUENCE, REPLY_SPREAD)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_reply_counts_the_discards_before_and_since_the_rule(
    chat_eval: ChatEval, model: str
) -> None:
    """A rule against inventing links, a job whose five early cycles each discarded exactly
    that draft and whose three recent cycles don't — the reply should count both halves from
    the seeded facts, each read as its own claim."""
    cohort = await chat_eval(
        case_id="rule-effectiveness-from-run-facts",
        behaviour=(
            "In the chat agent, when the user asks how many times a standing rule had to "
            "catch something before versus since, Penny's reply states both counts the job's "
            "own run facts show."
        ),
        area=Area.STANDING_JOBS,
        model=model,
        seed=_seed_world,
        prepare=_premise(),
        world=World(name="rule-effectiveness-from-run-facts", pages=(), keeps=(), excludes=()),
        ask=ASK,
        also_phrased=ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — #2843 is new capability, no baseline to beat
        family=_FAMILY,
        timeout=120.0,
    )
    cohort.claim(
        "reply: states the standalone count of discards before the rule (5 / five)",
        _reply_states_five_before,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.claim(
        "reply: states the standalone count of discards since the rule (0 / none / zero / no)",
        _reply_states_none_since,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.measure(*_MEASURED)
