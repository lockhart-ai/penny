"""Duplicate-CALL recovery contract — when the agent-loop dedup guard rejects a
byte-identical repeat, the reworked message must move the model ON without it
over-generalizing "no repeated calls" and suppressing the work it still owes.

Production failure this pins (July 2026 tool-failure audit): the terse
``"You already made this exact tool call. Try a different query or tool."``
rejection moved the model on ~83% of the time, but the runs containing it failed
at ~8x the baseline rate — traces show the model concluding the policy forbids
repeated calls and then dropping legitimate follow-up work (a verify re-read after
a write) for the rest of the run.  The message now states the why-now (this exact
call already ran; its result is above) AND the legitimate path (reuse that result;
this flags only the identical repeat, not reusing a tool at all) — and since #1673
it leads with the prior call's OUTCOME, so a repeat of a call that FAILED is
answered by "fix the precondition" rather than "reuse its result".

The slip is a model DECISION on a visible tool result, but a natural cycle only
rarely repeats an exact call, so we force ONE byte-identical repeat of the model's
first tool call (``_InjectDuplicateCall``) and let the REAL model drive the recovery
off the production rejection.  The contract is STRUCTURAL, never wording:

  PASS = the cycle RECOVERED — it reused the earlier read and still wrote the
         summary the seeded messages clearly warrant — rather than freezing after
         the rejection (the over-generalization) or spiraling to the step ceiling.

The guard blocks a byte-identical repeat for the whole run, so this measures the
real harm — owed follow-up work being suppressed — via the write completing, not by
forcing a literal re-read (which the unchanged guard would itself refuse).

**The world is the post-#1911 one.**  The program parser is STRICT (a step's call
must OPEN the step, in the rendered ``N. tool(args)`` dialect) and the cycle's tool
surface is SCOPED to that program's own calls, closed over ``Tool.advises`` — so
the collection is what an apply turn leaves behind (migration 0108 pre-seeds
nothing): a taught routine in the registry, and a container configured from it
through the production instantiation seam (retarget → ``bind_parameters`` →
``render_skill``).  The hand-authored prose recipe this module used to seed carried
its own terminal ``done()`` step, which assembly now injects, and its prose step 3
parsed to nothing; the rendered program has neither problem, and the surface it
scopes to carries the ``log_read`` the repeat is forced on and the
``collection_write`` the recovery owes.

Report-only (``min_pass_rate=None``), the canonical convention, pending a joint
read of the re-armed cases.  The deterministic mechanism (reject in place, don't
stop the loop) is pinned in ``tests/agents/test_agentic_loop.py``; this owns the
live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.constants import PennyConstants
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
    Check,
    _InjectDuplicateCall,
    _iter_prompt_messages,
    collection_entries,
    tool_call_sequence,
    tool_was_called,
)
from penny.tests.eval.fixtures import WEEKLY_DIGEST, WEEKLY_DIGEST_MESSAGES

pytestmark = pytest.mark.eval

_INCOMING = PennyConstants.MessageDirection.INCOMING

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The calls the routine makes, in order — what the program must read back as under the
# strict dialect, and therefore what the cycle's surface is scoped to.
WEEKLY_DIGEST_PROGRAM_CALLS = ("log_read", "collection_read_latest", "collection_write")

# The job's cadence — stated because a configured collection has one, though the case
# drives the cycle through ``run_for``, which bypasses readiness.
WEEKLY_DIGEST_SCHEDULE = "FREQ=MINUTELY;INTERVAL=20"

# Substrings of the two #1673 duplicate-call rejection bodies — the outcome-first pair
# (the prior call succeeded, or it failed).  Matched rather than reproduced, so the
# probe reads the SAME text the model read; the full wording is pinned in
# ``tests/agents/test_agentic_loop.py``.
_REFUSAL_MARKERS = (
    "already made this exact tool call earlier in this run",
    "already FAILED earlier this run",
)

# The routine the user taught the digest, in the shape run-end extraction leaves
# behind: every leaf a labelled PLACEHOLDER, the destination and the routine's read of
# its own collection both carrying the attachment mark (#1783).  It declares NO
# parameter — the job is pointed at nothing beyond the log it drains and the collection
# it writes, which the composed prompt states out loud ("The values it is pointed at:
# none — this routine takes none").
WEEKLY_DIGEST_SKILL = SkillDraft(
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
                    description="the key the rolling summary is stored under",
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description=(
                        "one paragraph folding the newest messages into the summary so far"
                    ),
                ),
            ],
        ),
    ],
    parameters=[],
    source_run_id="eval-seed",
)


def rendered_program() -> str:
    """The program the apply turn stores, through the production instantiation seam's
    own three steps in its own order (``render_skill_prompt``): the attachment bound to
    the container, the runtime join (nothing to join — this routine declares no
    parameter), then the render.  Public because the seeder and the probe both read it,
    and a second copy would be free to drift from what the collection stores."""
    attached = retarget_writes(WEEKLY_DIGEST_SKILL.steps, WEEKLY_DIGEST.name)
    joined = bind_parameters(attached, WEEKLY_DIGEST_SKILL.parameters, {})
    return render_skill(joined)


def _seed_digest_with_messages(db: Database) -> None:
    """The world an apply turn leaves — the taught routine in the registry and a container
    configured from it — plus clearly-summarizable seeded messages, then every claim that
    world makes, asserted.

    The messages give the cycle real work, so a recovered run MUST write a summary
    entry: a no-write after the forced duplicate read is a failure to recover (the
    model froze on the rejection instead of reusing the read result)."""
    db.skills.upsert(WEEKLY_DIGEST_SKILL, author=_SEED_AUTHOR)
    db.memories.create_collection(
        WEEKLY_DIGEST.name,
        WEEKLY_DIGEST.description,
        extraction_prompt=rendered_program(),
        schedule=WEEKLY_DIGEST_SCHEDULE,
        skill_name=slug_skill_name(WEEKLY_DIGEST_SKILL.name),
        skill_params={},
    )
    for message in WEEKLY_DIGEST_MESSAGES:
        db.messages.log_message(_INCOMING, "user", message)
    _assert_the_digest_world(db)


# ── The loud seed probe ───────────────────────────────────────────────────────


def _assert_the_digest_world(db: Database) -> None:
    """Everything the seeder is responsible for, asserted out loud.

    Two of its claims fail silently and cost a live cycle per sample to not notice: a
    program the strict parser cannot read leaves the cycle with a surface of the
    terminator ALONE (no ``log_read`` to repeat, no ``collection_write`` to owe), and an
    empty message log leaves the cycle with nothing to summarise, which is the same
    shape as the freeze this case exists to detect."""
    _assert_the_routine_is_registered(db)
    _assert_the_job_is_configured(db)
    _assert_the_program_parses(db)
    _assert_there_is_work_to_do(db)


def _assert_the_routine_is_registered(db: Database) -> None:
    """The routine the container names is one the registry holds — else the composed
    prompt states the routine as gone."""
    name = slug_skill_name(WEEKLY_DIGEST_SKILL.name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"


def _assert_the_job_is_configured(db: Database) -> None:
    """The container the cycle runs is configured as an apply turn leaves it: the routine
    stamped, the turn's schedule, live, and NOT notifying — this case scores what the
    cycle writes, never what it tells the user."""
    row = db.memories.get(WEEKLY_DIGEST.name)
    assert row is not None, f"the job's container {WEEKLY_DIGEST.name!r} must exist"
    assert row.skill_name == slug_skill_name(WEEKLY_DIGEST_SKILL.name), (
        f"the job must run the routine the fixture taught, not {row.skill_name!r}"
    )
    assert row.schedule == WEEKLY_DIGEST_SCHEDULE, (
        f"the job must carry its own rule, got {row.schedule!r}"
    )
    assert not row.archived and not row.notify, (
        "the job must be live and silent — a notifying job would put a second claim in "
        "front of the user that this case does not score"
    )


def _assert_the_program_parses(db: Database) -> None:
    """The stored program reads back as the calls it makes, under the STRICT dialect
    (#1911), names the container it writes to, and stores no terminal ``done()``.

    ``log_read`` opening step 1 is what the forced repeat lands on, and
    ``collection_write`` closing the program is the work the recovery owes — both have
    to be on the scoped surface for the case to measure anything."""
    row = db.memories.get(WEEKLY_DIGEST.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(
        call.tool for call in program_calls(program, frozenset(WEEKLY_DIGEST_PROGRAM_CALLS))
    )
    assert parsed == WEEKLY_DIGEST_PROGRAM_CALLS, (
        f"the stored program must read back as {list(WEEKLY_DIGEST_PROGRAM_CALLS)} under "
        f"the rendered dialect, got {list(parsed)} — program: {program!r}"
    )
    assert f"'{WEEKLY_DIGEST.name}'" in program, (
        f"the attachment must be bound to {WEEKLY_DIGEST.name!r} — a program carrying the "
        f"placeholder does not state where it writes.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying "
        "one is a render a chat ledger cannot produce"
    )


def _assert_there_is_work_to_do(db: Database) -> None:
    """The log the program reads exists and carries the seeded messages, and the
    container is empty — so "wrote the summary" is a real outcome the cycle can only
    reach by doing the work, not a state the world already satisfied."""
    log = db.memory(PennyConstants.MEMORY_USER_MESSAGES_LOG)
    assert log is not None, (
        f"the {PennyConstants.MEMORY_USER_MESSAGES_LOG!r} log the program reads must "
        "exist — its marker row is migration-seeded, and the read dispatches through it"
    )
    held = collection_entries(db, WEEKLY_DIGEST.name)
    assert not held, f"the digest must be empty when the cycle starts, got {held}"


# ── Scoring ───────────────────────────────────────────────────────────────────


def _repeat_was_refused(db: Database) -> bool:
    """Did the loop's dedup guard actually REFUSE the forced repeat, read off the
    persisted tool results — the same text the model read?

    Substrings of the two #1673 rejection bodies rather than a copy of them: the full
    wording is pinned in ``tests/agents/test_agentic_loop.py``, and what this needs to
    tell apart is the guard speaking from the repeat quietly executing."""
    return any(
        message.get("role") == "tool"
        and any(marker in (message.get("content") or "") for marker in _REFUSAL_MARKERS)
        for message in _iter_prompt_messages(db)
    )


def _score_recovered_with_work(db: Database, sent: list[str]) -> list[Check]:
    """The cycle recovered from the forced duplicate call: the guard refused the repeat,
    and the model reused the earlier read and still wrote the summary the seeded messages
    clearly warrant.

    (``guard_recovery_eval`` prepends its own "forced bail fired" guard, which says the
    injector ran; the first check here says the GUARD answered it.)"""
    refused = _repeat_was_refused(db)
    wrote = bool(collection_entries(db, WEEKLY_DIGEST.name))
    calls = tool_call_sequence(db)
    return [
        Check(
            "the byte-identical repeat was refused rather than re-run",
            refused,
            kind="guard",
            rationale=None
            if refused
            else (
                "the loop's dedup rejection never came back, so nothing was measured — "
                f"calls made: {calls or 'none'}"
            ),
        ),
        Check(
            "still wrote the summary it owed after the rejection",
            wrote,
            anchor="collection_write(",
            kind="state",
            rationale=None
            if wrote
            else (
                "nothing was written after the duplicate-call rejection — the harm this "
                "case exists for: the model reads the refusal as a rule against repeating "
                f"calls and drops the work it still owes.  Calls made: {calls or 'none'}"
            ),
        ),
        Check(
            "closed with done() rather than running out of steps",
            tool_was_called(db, "done"),
            anchor="done(",
            scored=False,
            kind="proc",
            rationale=f"calls made: {calls or 'none'}",
        ),
    ]


async def test_duplicate_call_is_rejected_and_recovers(guard_recovery_eval) -> None:
    """A byte-identical repeat of the first tool call is rejected in place; the live
    model reuses the earlier result and still writes the summary it owes."""
    await guard_recovery_eval(
        case_id="duplicate-call-recovery",
        family="collector-guard-recovery",
        collection=WEEKLY_DIGEST.name,
        seed=_seed_digest_with_messages,
        wrap_client=lambda real: _InjectDuplicateCall(real),
        score=_score_recovered_with_work,
        min_pass_rate=None,
    )
