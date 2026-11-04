"""A job filing into a collection it is NOT bound to (#2963): the entry lands there.

**One case, ``review-files-into-a-separate-tracker``.**  A schedule-fired job is bound to
one collection — its read cursors and entry author are keyed to it — but its PROMPT is free
text, and a prompt can direct its writes anywhere.  The motivating production run was a
review-style job whose instructions told it to file findings into a separate issue-tracker
collection; the write was refused at the tool layer ("this collector can only write to its
own"), the user never got the findings, and the restriction has been lifted (#2963).  What
this case pins is the behaviour the lift establishes: the entry the job files goes into the
collection its instructions name, not the one it is bound to.

**What selects the behaviour is the ASK, not an entry condition.**  The bound collection
holds two prior review reports (the review's own record, sectioned the way its prompt
prescribes); the newest one carries one problem still marked UNRESOLVED.  The job's
instructions direct unresolved findings into a separate tracker collection under an explicit
Key/Content format — the structure of the production prompt that motivated the lift.  A
correct cycle reviews its record, and the finding it files stands in the tracker.

**The arms are five wordings of the one filing clause.**  A collector has no user turn, so
its natural language is the job's own free-text prompt (#2455 naive baseline); what varies
is the sentence that directs the write, while the review structure around it — the sections,
the synthesize method, the format line, the update-don't-refile rule — stays constant.  Same
ask, same destination, five ways of saying it.

**The FACTS are constant, and the claim hinges on them.**  One finding ("the greenhouse
thermometer reads low") lives in exactly one place — the newest report — and in nothing
seeded into the tracker, so an entry in the tracker carrying the finding's token is one the
cycle put there.  The token is a real word the model recognises (#2673), invariant under the
formatting a draw might keep ("three degrees" vs "3°C"), and exclusive: no seeded row and no
other report carries it.

**No claim reads the machine, a tool name, or a wording.**  The claim is a store fact read
off ``entries`` — what the sample wrote, wherever it landed — which is already
collection-attributed, so the cross-collection landing needs no new observation field.  How
the cycle got there (a log read, a synthesize, a keys listing first) is ROUTE, measured in
the tool sequence and never asserted.  The entry's AUTHOR being keyed to the bound
collection is the mechanism proof, owned by the mock tests in ``tests/tools/`` — the store's
author column is not on the observation, and pinning it here would re-measure what
``make check`` already owns.

Report-only (``min_pass_rate=None``).  Every topic — the greenhouse, its kit, the findings
— is an unrelated synthetic domain; the repo is public.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Seeder,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    token_stands_in,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "collector-enactment"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The run that laid the entry condition down.  A SEEDED id, structurally distinguishable from
# a live ``uuid4().hex``, so every "what did this sample write" reader excludes the seeded
# rows at the one chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("cross-collection-entry-condition")

# The job's bound collection — the one its cycle runs FOR.  Its prior reports are the review's
# own record, and its own writes (a fresh report) land here as they always have.
_CONTAINER = "garden-review"
_CONTAINER_DESCRIPTION = "The running review of how the greenhouse and its kit are doing."

# The collection the job's instructions direct its filings into — plain storage, no schedule,
# no prompt of its own.  It is NOT the bound collection; that is the whole point of the case.
_TRACKER = "garden-issues"
_TRACKER_DESCRIPTION = "Issues spotted in the greenhouse kit, one per entry."

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=HOURLY"

# The finding the newest report carries and the tracker must end up holding.  The token is
# what the claim matches — invariant under the notation a draw keeps, exclusive to the
# newest report, and present in nothing the tracker is seeded with.
_FINDING = "the greenhouse thermometer reads three degrees low against the reference gauge"
_FINDING_TOKEN = "thermometer"

# The issue the tracker is seeded with — an earlier review's filing, a DIFFERENT problem.
# Its key is what "the tracker keeps what it already had" would read, and its content carries
# nothing the finding claim could match, so a seeded row cannot pass for a filing.
_SEED_TRACKER_KEY = "seed-tray-jam"
_SEED_TRACKER_CONTENT = (
    "Status: UNRESOLVED. Issue: The seed tray jams on the second shelf slider. Found: 2026-10-19."
)


def _prompt(filing_clause: str) -> str:
    """The job's free-text instructions (#2455 naive baseline) — what its
    ``extraction_prompt`` carries verbatim.

    ONE shape for every arm — the same sections, the same method, the same format line and
    update rule.  The one thing that differs is the FILING CLAUSE, which is what the test
    varies: the sentence directing the write is the arm axis, mirroring the production
    prompt's own structure (a sectioned review, a synthesize method, an explicit
    cross-collection filing clause with a Key/Content format)."""
    return (
        "Every time this job runs, review how the greenhouse and its kit are doing, and "
        "keep the issue tracker honest.\n"
        "\n"
        "**Method: pattern first (via `synthesize`)**\n"
        "Where the reports repeat a problem, use `synthesize` on this collection to move "
        'from "what happened" to "what keeps happening" before you write.\n'
        "\n"
        "1. **Readings** — what the newest reports measured, and what moved since the last "
        "review.\n"
        "2. **Kit problems** — anything failing, wearing, or misreading, with which report "
        "it came from.\n"
        "3. **Proposed changes** — what to adjust, and what it would fix.\n"
        "\n"
        f"**Issue filing**: {filing_clause}. Use the format:\n"
        "`Key: <short_id>`\n"
        "`Content: Status: <RESOLVED/UNRESOLVED>. Issue: <what is wrong>. Found: <date>.`\n"
        "If an issue you previously filed is fixed, update its entry's status to `RESOLVED` "
        "rather than filing it again."
    )


# Five wordings of the one filing clause — same ask, same destination, five ways of saying
# the write goes to the tracker.  Each names the tracker by its exact name, because a cycle
# can only write where its instructions name.
FILING_CLAUSES = (
    "any unresolved finding you review MUST also be filed as an entry in the "
    f"'{_TRACKER}' collection",
    f"file every unresolved finding you review into the '{_TRACKER}' collection",
    f"record each unresolved finding you review in '{_TRACKER}'",
    f"log each unresolved finding you review in the '{_TRACKER}' collection",
    f"write every unresolved finding you review to '{_TRACKER}'",
)


def _report(key: str, findings: tuple[str, ...]) -> tuple[str, str]:
    """One prior review report as the store holds it: a key, and a sectioned body in the
    shape the job's own prompt prescribes — the same sectioned shape the production record
    carried.  ``findings`` are the kit-problem lines; the readings around them are filler
    that carries no finding token."""
    return (
        key,
        "### Readings\n"
        "- Greenhouse humidity steady at 68%.\n"
        "- Heater cycled on schedule overnight.\n"
        "\n"
        "### Kit problems\n"
        + "".join(f"- {finding}\n" for finding in findings)
        + "\n### Proposed changes\n"
        "- Keep the sowing round on schedule.",
    )


# The bound collection's prior record: an older review and the newest one.  The newest
# carries the UNRESOLVED finding the cycle should file; the older one's problem is RESOLVED
# and already stands in the tracker, so "filed" and "already there" are different states the
# model can tell apart.
_PRIOR_REPORTS = (
    _report(
        "review-1021",
        ("RESOLVED: the dripper line kink was repaired on the 22nd.",),
    ),
    _report("review-1028", (f"UNRESOLVED: {_FINDING}.",)),
)


def _world() -> World:
    """The ground every arm of this case runs against: no pages — the review's inputs are
    its own record and the tracker, both seeded by the seeder, not a browse register.

    ``keeps``/``excludes``/``answers`` are EMPTY, and necessarily so.  Each of them is a
    token set read off a page, and this world serves no page — the finding the claim
    matches is named once above, against the report the model is handed in its own cycle
    prompt.  Declaring page tokens anyway would print "must be kept" rows in a report where
    nothing verified them, and a contract nothing reads is worse than no contract: it reads
    as a check that passed."""
    return World(name="garden-review-and-its-tracker", pages=(), keeps=(), excludes=())


def _seeder(filing_clause: str) -> Seeder:
    """The world this case's cycle starts in: the job configured with its prompt, its two
    prior reports, and the tracker holding the earlier filing — then every claim that world
    makes, asserted out loud.

    The entries go down through the store's own write path, under the keys the reports
    carry, so the rows carry the stamps and the embeddings a real write would have left.
    Hand-inserting them would give the cycle a record no cycle could have produced."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_prompt(filing_clause),
            schedule=_SCHEDULE,
            notify=False,
        )
        memory = db.memory(_CONTAINER)
        assert memory is not None, f"the job's container {_CONTAINER!r} must exist to seed"
        memory.write_rule_decided(
            [EntryInput(key=key, content=content) for key, content in _PRIOR_REPORTS],
            author=_SEED_AUTHOR,
            run_id=_SEED_RUN,
        )
        db.memories.create_collection(_TRACKER, _TRACKER_DESCRIPTION)
        tracker = db.memory(_TRACKER)
        assert tracker is not None, f"the tracker {_TRACKER!r} must exist to seed"
        tracker.write_rule_decided(
            [EntryInput(key=_SEED_TRACKER_KEY, content=_SEED_TRACKER_CONTENT)],
            author=_SEED_AUTHOR,
            run_id=_SEED_RUN,
        )
        _assert_the_review_world(db, filing_clause)

    return seed


def _assert_the_review_world(db: Database, filing_clause: str) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert not row.archived, "the job must be live when the cycle runs"
    assert not row.notify, (
        "these cases claim what the cycle WROTE — a notifying job would enter the delivery "
        "turn, which would queue a message on a correct cycle as readily as on a wrong one"
    )
    prompt = row.extraction_prompt
    assert prompt is not None, (
        f"the job's container {_CONTAINER!r} must carry a prompt — a container with none "
        "leaves its cycles nothing to run"
    )
    assert f"'{_TRACKER}'" in prompt, (
        f"the prompt must name its filing destination, {_TRACKER!r} — the write goes where "
        f"the instructions name.  Prompt: {prompt!r}"
    )
    assert filing_clause in prompt, (
        f"this arm's filing clause must reach the model verbatim — the clause is the whole "
        f"arm axis.  Prompt: {prompt!r}"
    )

    held = collection_entries(db, _CONTAINER)
    assert held == dict(_PRIOR_REPORTS), (
        f"the container must hold exactly its two prior reports when the cycle starts — it "
        f"holds {sorted(held)}"
    )
    newest = held["review-1028"]
    assert _FINDING_TOKEN in newest, (
        f"the newest report must carry the finding token {_FINDING_TOKEN!r} — it is what "
        f"the filing claim matches, and the cycle prompt shows this entry.  Body: {newest!r}"
    )

    tracker = collection_entries(db, _TRACKER)
    assert tracker == {_SEED_TRACKER_KEY: _SEED_TRACKER_CONTENT}, (
        f"the tracker must hold exactly its earlier filing when the cycle starts — it holds "
        f"{tracker}"
    )
    assert _FINDING_TOKEN not in _SEED_TRACKER_CONTENT, (
        "the seeded tracker row must not carry the finding token, or the claim could pass "
        "on the seed rather than on a filing"
    )


def _arm(filing_clause: str) -> CycleArm:
    """One arm of one case: this arm's filing clause in the job's prompt.

    ``text`` is the clause, because that is what makes this arm this arm."""
    return CycleArm(
        text=filing_clause,
        seed=_seeder(filing_clause),
        pages=[],
        world=_world(),
    )


def _arms() -> list[CycleArm]:
    """This case's five arms — five wordings of the one filing clause."""
    return [_arm(clause) for clause in FILING_CLAUSES]


# ── The claims ───────────────────────────────────────────────────────────────
#
# One store claim, plus the shared provenance one.  The claim reads ``entries`` — what the
# sample WROTE, wherever it landed, collection-attributed — which is exactly the store fact
# the case exists for.  How the cycle got there is ROUTE, measured and never asserted.


def _tracker_holds_the_filing(sample: SampleObservation, _world: World) -> Answer:
    """The finding the review filed stands in the tracker — the collection the job is NOT
    bound to.

    Read over ``entries``, what this sample wrote: a tracker entry carrying the finding
    token is one the cycle put there, because nothing the tracker was seeded with carries
    the token (asserted at seed time).  Matched on the bare token — the notation a draw
    keeps for the figure is not this claim's question."""
    filed = [entry for entry in sample.entries if entry.collection == _TRACKER]
    filed_text = " ".join(entry.text for entry in filed)
    ok = token_stands_in(_FINDING_TOKEN, filed_text)
    return ok, (
        f"expected {_FINDING_TOKEN!r} in {_TRACKER!r}; the cycle filed "
        f"{[entry.text for entry in filed]!r}"
    )


# ── the case ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_review_files_into_a_separate_tracker(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A job whose instructions direct its filings into a collection it is not bound to
    lands the entry there — the entry the job filed is in the other collection."""
    cohort = await collector_cycles_eval(
        case_id="review-files-into-a-separate-tracker",
        behaviour=(
            "In a collector cycle, when the job's instructions direct its filings into a "
            "collection other than the one it is bound to, the entry the job files lands "
            "in that collection."
        ),
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE
    cohort.claim(
        f"state: the tracker holds the filing the review made ({_TRACKER})",
        _tracker_holds_the_filing,
        SpecCategory.STORE,
    )

    # PROVENANCE — the filed finding must trace to the record the cycle was handed, and the
    # seeded tracker row must be declared ground rather than a mystery the store grew.
    cohort.assert_every_value_in_the_store_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED)
