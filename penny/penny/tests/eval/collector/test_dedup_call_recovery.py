"""Duplicate-call recovery: the cycle still writes what it owes after a repeat is refused.

Ported to the cohort structure under #2060; the contract is `docs/eval-case-design.md`.

Mid-cycle, the harness replays the model's own first tool call byte-identically, so the
agent loop's seen-calls guard refuses it.  The rejection states the why-now (this exact
call already ran; its result is above) and the legitimate path (reuse that result; this
flags only the identical repeat, not reusing a tool at all).  The live model must MOVE ON
— reuse the earlier read and finish the write the cycle owes — instead of over-generalising
"no repeated calls" and dropping the work.  Production residue this pins (July 2026
tool-failure audit): the terse wording moved the model on ~83% of the time, but the runs
carrying it failed at ~8x the baseline rate.

**Why this is its own case, and not the duplicate-WRITE family's (#2060's first job).**
The refusal here comes from the loop's seen-calls cache, not from the store: nothing was
rejected about the VALUE, so there is no gate outcome, no matched key and no anchor to
copy.  The recovery is "carry on with what you already have", where a write rejection's is
"land it on the entry that exists".  Materially different decisions, so a correct sample
for one says nothing about the other.

  * ONE CYCLE.  The entry condition — an EMPTY digest and a message log carrying work the
    cycle can only discharge by writing — is what selects the behaviour: a no-write is then
    an unambiguous freeze rather than a defensible no-op.
  * THE FAULT IS FIXED, the WORDING varies.  A collector has no user turn, so its natural
    language is the ``SkillSubstitution`` descriptions its rendered program carries, written
    upstream and printed by ``render_skill`` — the same seam an ``extract`` instruction
    reaches the model through.  Five wordings of the one that says what to write; the forced
    repeat and the seeded messages are byte-identical on every arm.
  * THE INJECTION IS HARNESS MACHINERY.  A sample the replay never fired on ran an unbroken
    cycle and exercised no recovery, so it leaves the cohort as a named exclusion rather
    than counting as a behavioural failure.  THE COST, STATED (#2018): ``bail_injected``
    says the repeat was ISSUED, not that the guard refused it — a mutating call landing
    first clears the seen-calls cache (#1673), and such a sample leaves here too.
  * LANDED renders EMPTY, and that is the correct report: a collector moves no conversation
    machine.  The run record's outcome is a RECORD FIELD and is claimed under STORE.
  * the tool calls MEASURED rather than asserted — the route after a refusal is exactly
    where correct runs differ.

**What this world cannot claim, stated rather than papered over.**  A rolling summary is
prose the routine asks the model to compose, so this collection holds no small unique datum
to name: there is no ``449`` here.  The STORE side is therefore a COUNT — the digest holds
the one entry it owes — and the substance is carried by PROVENANCE, which fails a summary
composed out of nothing.  Widening the job so it asks for a nameable value would change the
behaviour being measured, which is the code owner's call and not an in-flight repair.

REPORT-ONLY (``min_pass_rate=None``): the floors and ceilings this run proposes are the
code owner's to accept once the numbers have been read.  All content is synthetic — invented
chat about a work release and a weekend — because the repo is public.

The deterministic mechanism (refuse the repeat in place, do not stop the loop) is pinned in
``tests/agents/test_agentic_loop.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants, RunOutcome
from penny.database import Database
from penny.database.skills import (
    SkillDraft,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    bind_parameters,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.program import program_calls
from penny.prompts import Prompt
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    CollectorCyclesEval,
    CycleArm,
    _InjectDuplicateCall,
    collection_entries,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.fixtures import WEEKLY_DIGEST, WEEKLY_DIGEST_MESSAGES
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_CASE_ID = "duplicate-call-recovery"

# The family every collector recovery case reports under — what a cycle does once one of
# its own calls has already been refused.  A second spelling would split one bucket.
_FAMILY = "collector-recovery"

_INCOMING = PennyConstants.MessageDirection.INCOMING

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The calls the routine makes, in order — what the stored program must read back as under
# the strict rendered dialect, and therefore what this cycle's tool surface is scoped to.
# ``log_read`` opening the program is what the forced repeat lands on, and
# ``collection_write`` closing it is the work the recovery owes.
_PROGRAM_CALLS = ("log_read", "collection_read_latest", "collection_write")

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=MINUTELY;INTERVAL=20"

# The key description, constant across the arms: only ONE instruction varies, and it is the
# one below.
_KEY_HINT = "the key the rolling summary is stored under"


# Five wordings of one instruction — the substitution description that becomes ``{…}`` in
# the rendered program at the write step's content leaf, which is the only natural language
# this cycle is handed, and the row a reader opens when the report says a phrasing diverged.
# Every one says the same thing — fold what is new into the one running summary — and none
# of them names a fact, because the messages are the facts and they are constant.
SUMMARY_HINTS = (
    "one paragraph folding the newest messages into the summary so far",
    "a single paragraph bringing the summary so far up to date with the newest messages",
    "one paragraph covering what the summary already says plus whatever is new",
    "the running summary rewritten as one paragraph, with the latest messages worked in",
    "one paragraph that reads as the whole story so far, newest messages included",
)


# Every arm's ground, and there is ONE of it: this job reads no page.  A world's pages are
# what a browse-driven cycle is answered against; this routine's only source is the message
# log it drains, which reaches the provenance claim through what the model was GIVEN.  So the
# arms genuinely share their ground and the report states it once — a world per arm would
# print five distinct grounds that differ in nothing but a label.  ``keeps``/``excludes`` are
# empty for the reason ``test_watch_cycles.py``'s are: they back a claim this case never
# makes, and declaring tokens nothing reads prints a contract that looks verified.
_WORLD = World(name="weekly-digest", pages=(), keeps=(), excludes=())


# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED name of where the behaviour happens.  The case
# id is a filename; this is the contract, and it renders above every number in the report.
_BEHAVIOUR = (
    "In a rolling-summary collector, when a call she has already made is refused as an "
    "exact repeat, Penny carries on from what that call already returned and still writes "
    "the entry the cycle owes."
)


def _skill(summary_hint: str) -> SkillDraft:
    """The routine the user taught the digest, in the shape run-end extraction leaves behind.

    ONE shape for every arm — the same three steps, the same placeholders, the same
    attachment mark on the destination and on the read of the routine's own collection.  It
    declares NO parameter: the job is pointed at nothing beyond the log it drains and the
    collection it writes.  What differs is the write's CONTENT description, printed by
    ``render_skill``, which is the arm axis."""
    return SkillDraft(
        name="roll_up_recent_messages",
        intent="Keep one running summary of what I've been up to lately, updated as I chat.",
        description="Fold the newest messages into one rolling summary entry.",
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="log_read",
                arguments={"memory": PennyConstants.MEMORY_USER_MESSAGES_LOG},
            ),
            SkillStep(
                ordinal=2,
                source_ordinal=2,
                tool="collection_read_latest",
                arguments={"memory": WEEKLY_DIGEST.name, "k": 1},
                substitutions=[
                    SkillSubstitution(
                        path=["memory"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the collection this is set up on",
                        attachment=True,
                    )
                ],
            ),
            SkillStep(
                ordinal=3,
                source_ordinal=3,
                tool="collection_write",
                arguments={
                    "memory": WEEKLY_DIGEST.name,
                    "entries": [
                        {
                            "key": "summary",
                            "content": (
                                "A quiet stretch: shipped a release at work, started running "
                                "again, and had a low-key weekend."
                            ),
                        }
                    ],
                },
                substitutions=[
                    SkillSubstitution(
                        path=["memory"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the collection this is set up on",
                        attachment=True,
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "key"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description=_KEY_HINT,
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "content"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description=summary_hint,
                    ),
                ],
            ),
        ],
        parameters=[],
        source_run_id=_SEED_AUTHOR,
    )


def _program(summary_hint: str) -> str:
    """The program the apply turn stores, through the production instantiation seam's own
    three steps in its own order: the attachment bound to the container, the runtime join
    (nothing to join — this routine declares no parameter), then the render."""
    skill = _skill(summary_hint)
    attached = retarget_writes(skill.steps, WEEKLY_DIGEST.name)
    joined = bind_parameters(attached, skill.parameters, {})
    return render_skill(joined)


def _summary_slot(summary_hint: str) -> str:
    """How this arm's instruction renders inside the program — the one span that moves."""
    return f"{{{summary_hint}}}"


def _seeder(summary_hint: str):
    """The world this cycle starts in: the routine in the registry, a container configured
    from it, and the ENTRY CONDITION — an empty digest and a log carrying real work — then
    every claim that world makes, asserted out loud.

    The probe is not ceremony.  Two of its claims fail silently and cost a live cycle per
    sample to not notice: a program the strict parser cannot read leaves the cycle with a
    surface of the terminator alone (no ``log_read`` to repeat, no ``collection_write`` to
    owe), and an empty message log leaves the cycle with nothing to summarise — which is the
    same shape as the freeze this case exists to detect."""

    def seed(db: Database) -> None:
        skill = _skill(summary_hint)
        db.skills.upsert(skill, author=_SEED_AUTHOR)
        db.memories.create_collection(
            WEEKLY_DIGEST.name,
            WEEKLY_DIGEST.description,
            extraction_prompt=_program(summary_hint),
            schedule=_SCHEDULE,
            skill_name=slug_skill_name(skill.name),
            skill_params={},
        )
        for message in WEEKLY_DIGEST_MESSAGES:
            db.messages.log_message(_INCOMING, "user", message)
        _assert_the_digest_world(db, summary_hint)

    return seed


# ── The loud seed probe ───────────────────────────────────────────────────────


def _assert_the_digest_world(db: Database, summary_hint: str) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    _assert_the_job_is_configured(db, summary_hint)
    _assert_the_program_parses(db, summary_hint)
    _assert_the_entry_condition(db)


def _assert_the_job_is_configured(db: Database, summary_hint: str) -> None:
    """The routine is registered and the container is configured as an apply turn leaves it:
    the routine stamped, the turn's schedule, live, and NOT notifying."""
    name = slug_skill_name(_skill(summary_hint).name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"
    row = db.memories.get(WEEKLY_DIGEST.name)
    assert row is not None, f"the job's container {WEEKLY_DIGEST.name!r} must exist"
    assert row.skill_name == name, f"the job must run the taught routine, not {row.skill_name!r}"
    assert row.schedule == _SCHEDULE, f"the job must carry its own rule, got {row.schedule!r}"
    assert not row.archived and not row.notify, (
        "the job must be live and SILENT — this case scores what the cycle writes, and a "
        "notifying job would put a second claim in front of the user that nothing here reads"
    )


def _assert_the_program_parses(db: Database, summary_hint: str) -> None:
    """The stored program reads back as the calls it makes under the STRICT dialect (#1911),
    names the container it writes to, carries this arm's own instruction, and stores no
    terminal ``done()``.

    ``log_read`` opening step 1 is what the forced repeat lands on, and ``collection_write``
    closing the program is the work the recovery owes — both have to be on the scoped surface
    for the case to measure anything."""
    row = db.memories.get(WEEKLY_DIGEST.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_PROGRAM_CALLS)))
    assert parsed == _PROGRAM_CALLS, (
        f"the stored program must read back as {list(_PROGRAM_CALLS)} under the rendered "
        f"dialect, got {list(parsed)} — program: {program!r}"
    )
    assert f"'{WEEKLY_DIGEST.name}'" in program, (
        f"the attachment must be bound to {WEEKLY_DIGEST.name!r}.  Program: {program!r}"
    )
    assert _summary_slot(summary_hint) in program, (
        f"this arm's instruction must reach the model as {_summary_slot(summary_hint)!r} — the "
        f"summary description is the whole arm axis.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying one "
        "is a render a chat ledger cannot produce"
    )


def _assert_the_entry_condition(db: Database) -> None:
    """The state the cycle starts from, which is what selects the behaviour being measured.

    An empty digest makes "wrote the entry it owed" an outcome the cycle can only reach by
    doing the work rather than a state the world already satisfied, and the log the program
    drains has to exist and carry the seeded messages — else the cycle has nothing to fold
    in and a no-write is honest rather than a freeze."""
    log = db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG)
    assert log is not None, (
        f"the {PennyConstants.MEMORY_USER_MESSAGES_LOG!r} log the program reads must exist — "
        "its marker row is migration-seeded, and the read dispatches through it"
    )
    assert len(log.read_all()) >= len(WEEKLY_DIGEST_MESSAGES), (
        "the log must carry the seeded messages when the cycle starts — with nothing to fold "
        "in, a cycle that writes nothing is right, and this case would measure that instead"
    )
    held = collection_entries(db, WEEKLY_DIGEST.name)
    assert not held, f"the digest must be empty when the cycle starts, got {held}"


def _arms() -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction over one set of facts.

    ``text`` is the instruction, because that is what makes this arm this arm.  ``pages`` is
    empty: the job browses nothing, so its cycle is handed no register at all."""
    return [
        CycleArm(text=summary_hint, seed=_seeder(summary_hint), pages=[], world=_WORLD)
        for summary_hint in SUMMARY_HINTS
    ]


# ── The claims ───────────────────────────────────────────────────────────────
#
# Every one of them reads END STATE: what the digest holds, and what the run record says.
# Nothing reads a tool name or an ordering — the route a cycle takes after a refusal is
# exactly where correct runs differ, so it is measured in section B and never asserted.


def _one_entry(sample: SampleObservation, _world: World) -> Answer:
    """The digest holds the ONE running summary it owes.

    This is the whole behaviour in one reading, and it fails in both directions.  ZERO is
    the freeze this case is named for: the model reads the refusal as a rule against
    repeating calls and drops the work it still owes.  MORE THAN ONE is a rolling summary
    that has stopped rolling — a cycle appending beside the entry rather than replacing it
    grows without bound and the user is told things they already have."""
    keys = sorted(entry.key or "" for entry in sample.held)
    return len(sample.held) == 1, f"the digest holds {len(sample.held)} entries: {keys}"


def _closed_having_worked(sample: SampleObservation, _world: World) -> Answer:
    """The run's own determination: it completed and it CHANGED something.

    A record field, read literally off ``promptlog.run_outcome`` — ``worked`` is defined as
    "completed and changed something", which is what "the cycle recovered and wrote" means
    without reading a tool name.  ``no_work`` is a clean close that changed nothing and
    ``failed`` is a bail; both are the wrong end state for a cycle that owed a write."""
    closed = sample.run_outcome or "—"
    detail = f" — {sample.run_reason}" if sample.run_reason else ""
    return sample.run_outcome == RunOutcome.WORKED.value, f"the run closed {closed}{detail}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_refused_repeat_still_leaves_the_entry_it_owed(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A byte-identical repeat of the cycle's first call is refused in place; the live model
    carries on from the earlier result and still writes the summary it owes."""
    cohort = await collector_cycles_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        collection=WEEKLY_DIGEST.name,
        arms=_arms(),
        samples_per_phrasing=3,
        wrap_client=lambda real: _InjectDuplicateCall(real),
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.  A collector moves no conversation
    # machine, and the run record's fields are STORE claims below rather than a category
    # borrowed to fill a heading.

    # STORE
    cohort.claim(
        "state: the digest holds the one running summary it owes", _one_entry, SpecCategory.STORE
    )
    cohort.claim(
        "state: the run closed having changed something", _closed_having_worked, SpecCategory.STORE
    )

    # PROVENANCE — the claim with teeth on this world: a summary composed out of nothing,
    # by a cycle that froze on the refusal and then wrote anyway, names things no message
    # said.
    cohort.assert_every_stored_entry_traces_to_the_world()

    # REPLY_SPREAD is not measured: this job is silent, so a correct cohort sends nothing and
    # a reply-spread reading would be blind on every sample by construction.
    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED)
