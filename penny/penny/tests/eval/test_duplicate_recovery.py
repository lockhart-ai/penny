"""Duplicate-write recovery contract — when a collector's write is rejected as a
duplicate, the rejection hands back the matched existing key BOUND into the next
call, and the live model must recover instead of key-hunting.

Production failure this pins (July 2026 prompt audit): a duplicate rejection told
the model a similar entry existed but not WHICH one, so it guessed keys, re-read
the collection, or retried variations — burning its step budget (~1,800
wholly-duplicate-rejected writes across ~18% of collector runs in a 4-week window;
the recovery attempts fed the max-steps / died-mid-run failure classes).  Merely
naming the matched key in parentheses left the embedding-match arm recovering at
47%, so ``_format_duplicate`` binds it into a literal
``update_entry(key='<matched>', content=<richer info>)`` per rejected entry.

The slip is a model DECISION on a visible tool result, but a natural cycle only
rarely writes an exact duplicate, so we force ONE duplicate ``collection_write``
(``_InjectDuplicateWrite``) and let the REAL model drive the recovery off the
production rejection message.  The contract is STRUCTURAL, never wording:

  PASS = the rejection BOUND every matched key (the mechanism), the box's keys are
         UNCHANGED (dedup held; no confabulated or proliferated entries), the cycle
         RECOVERED — closed with ``done()`` or refreshed an existing entry — and
         every ``update_entry`` it made targeted a key the box really holds, rather
         than the model's own rejected candidate.

The first of those is new here and is the point of the #1919 pass: the case used to
score only the model's half, so a run where the guard never spoke at all still had
three checks to pass on.  It reads the bound key off the persisted tool result, the
same text the model read.

**The world is the post-#1911 one, and that is what re-arms this case.**  The
program parser is STRICT (a step's call must OPEN the step, in the rendered
``N. tool(args)`` dialect) and the cycle's tool surface is SCOPED to that program's
own calls, closed over ``Tool.advises``.  The hand-authored prose recipe this
module used to seed carried its write mid-sentence, so the parse found nothing,
``collection_write`` was not on the surface at all, and the injected write hit
tool-not-found instead of the dedup guard — the contract silently disarmed, and
nothing in the scoring could tell.  The collection is now what an apply turn leaves
behind (migration 0108 pre-seeds nothing): a taught routine in the registry, and a
container configured from it through the production instantiation seam (retarget →
``bind_parameters`` → ``render_skill``).  ``_assert_the_box_world`` asserts every
link in that chain out loud at seed time, including that ``update_entry`` — the
call the rejection names — is reachable from the program's own write verb through
the declared advice relation.

The routine reads the box and writes what is genuinely new; it browses nothing, so
a recovered cycle has no honest source of a NEW entry and "keys unchanged" stays a
clean read of the recovery rather than of the model's taste in recipes.

Report-only (``min_pass_rate=None``), the canonical convention, pending a joint
read of the re-armed cases.  The deterministic message content is pinned in
``tests/tools/test_memory_tools.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
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
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
    Check,
    _InjectDuplicateWrite,
    _iter_prompt_messages,
    collection_entries,
    tool_call_keys,
    tool_call_sequence,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_DUP_CONTENT,
    RECIPE_BOX_DUP_CONTENT_2,
    RECIPE_BOX_DUP_KEY,
    RECIPE_BOX_DUP_KEY_2,
    RECIPE_BOX_SEED_KEYS,
)
from penny.tools.memory_tools import CollectionWriteTool

pytestmark = pytest.mark.eval

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The calls the routine makes, in order — what the program must read back as under the
# strict dialect, and therefore what the cycle's surface is scoped to.
RECIPE_BOX_PROGRAM_CALLS = ("collection_read_latest", "collection_write")

# The job's cadence — stated because a configured collection has one, though the case
# drives the cycle through ``run_for``, which bypasses readiness.
RECIPE_BOX_SCHEDULE = "FREQ=HOURLY"

# The verb the duplicate rejection binds the matched key into (``_format_duplicate``).
_RECOVERY_VERB = "update_entry"

# The matched key as the rejection renders it: ``call update_entry(key='<matched>',
# content=<richer info>) to refresh it``.  Matched rather than reproduced, so the probe
# reads the SAME text the model read instead of a second copy of the wording.
_BOUND_KEY = re.compile(rf"call {_RECOVERY_VERB}\(key='([^']*)'")

# The routine the user taught the box, in the shape run-end extraction leaves behind:
# every leaf a labelled PLACEHOLDER, the destination and the read of the routine's own
# collection both carrying the attachment mark (#1783).  It declares NO parameter — the
# job is pointed at nothing beyond the box itself, which the composed prompt states out
# loud ("The values it is pointed at: none — this routine takes none").
RECIPE_BOX_SKILL = SkillDraft(
    name="save_new_recipes",
    intent="Keep a box of quick weeknight dinner recipes I can pull from.",
    description="File a quick weeknight recipe that is not already saved.",
    steps=[
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="collection_read_latest",
            arguments={"memory": RECIPE_BOX.name, "k": 20},
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
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": RECIPE_BOX.name,
                "entries": [
                    {
                        "key": "One-pot lemon orzo",
                        "content": "One-pot lemon orzo — orzo, lemon, spinach, parmesan, 20 min.",
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
                    description="the recipe's name",
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the recipe's name, its main ingredients and its cook time",
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
    attached = retarget_writes(RECIPE_BOX_SKILL.steps, RECIPE_BOX.name)
    joined = bind_parameters(attached, RECIPE_BOX_SKILL.parameters, {})
    return render_skill(joined)


def _seed_recipe_box(db: Database) -> None:
    """The world an apply turn leaves: the taught routine in the registry, the box with
    the two recipes it already holds, and the container configured from that routine —
    then every claim that world makes, asserted."""
    db.skills.upsert(RECIPE_BOX_SKILL, author=_SEED_AUTHOR)
    db.memories.create_collection(
        RECIPE_BOX.name,
        RECIPE_BOX.description,
        extraction_prompt=rendered_program(),
        schedule=RECIPE_BOX_SCHEDULE,
        skill_name=slug_skill_name(RECIPE_BOX_SKILL.name),
        skill_params={},
    )
    require_memory(db, RECIPE_BOX.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in RECIPE_BOX.entries],
        author="user",
    )
    _assert_the_box_world(db)


# ── The loud seed probe ───────────────────────────────────────────────────────


def _assert_the_box_world(db: Database) -> None:
    """Everything the seeder is responsible for, asserted out loud.

    The disarm this module was in is the reason: an unreadable program yields an EMPTY
    tool surface, the injected ``collection_write`` then comes back tool-not-found
    instead of duplicate-rejected, and the case scores a run in which the guard under
    test never spoke — silently, one live cycle per sample."""
    _assert_the_routine_is_registered(db)
    _assert_the_job_is_configured(db)
    _assert_the_program_parses(db)
    _assert_the_recovery_verb_is_reachable()
    _assert_the_box_holds_its_recipes(db)


def _assert_the_routine_is_registered(db: Database) -> None:
    """The routine the container names is one the registry holds — else the composed
    prompt states the routine as gone."""
    name = slug_skill_name(RECIPE_BOX_SKILL.name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"


def _assert_the_job_is_configured(db: Database) -> None:
    """The container the cycle runs is configured as an apply turn leaves it: the routine
    stamped, the turn's schedule, live, and NOT notifying — this case scores what the
    cycle writes, never what it tells the user."""
    row = db.memories.get(RECIPE_BOX.name)
    assert row is not None, f"the job's container {RECIPE_BOX.name!r} must exist"
    assert row.skill_name == slug_skill_name(RECIPE_BOX_SKILL.name), (
        f"the job must run the routine the fixture taught, not {row.skill_name!r}"
    )
    assert row.schedule == RECIPE_BOX_SCHEDULE, (
        f"the job must carry its own rule, got {row.schedule!r}"
    )
    assert not row.archived and not row.notify, (
        "the job must be live and silent — a notifying job would put a second claim in "
        "front of the user that this case does not score"
    )


def _assert_the_program_parses(db: Database) -> None:
    """The stored program reads back as the calls it makes, under the STRICT dialect
    (#1911), names the container it writes to, and stores no terminal ``done()``.

    ``collection_write`` being IN that list is what puts it on the cycle's scoped
    surface, which is what makes the injected write reach the dedup guard at all."""
    row = db.memories.get(RECIPE_BOX.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(
        call.tool for call in program_calls(program, frozenset(RECIPE_BOX_PROGRAM_CALLS))
    )
    assert parsed == RECIPE_BOX_PROGRAM_CALLS, (
        f"the stored program must read back as {list(RECIPE_BOX_PROGRAM_CALLS)} under the "
        f"rendered dialect, got {list(parsed)} — program: {program!r}"
    )
    assert f"'{RECIPE_BOX.name}'" in program, (
        f"the attachment must be bound to {RECIPE_BOX.name!r} — a program carrying the "
        f"placeholder does not state where it writes.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying "
        "one is a render a chat ledger cannot produce"
    )


def _assert_the_recovery_verb_is_reachable() -> None:
    """The call the rejection NAMES is on the cycle's surface.

    A scoped surface is the program's own calls closed over ``Tool.advises``, so the
    program naming ``collection_write`` is only half the requirement: ``update_entry``
    reaches the surface because ``collection_write`` DECLARES it as advice.  Read off
    the class attribute production closes over, so an advice relation that was dropped
    fails here rather than as an unexplained recovery collapse."""
    assert _RECOVERY_VERB in CollectionWriteTool.advises, (
        f"{_RECOVERY_VERB!r} must be declared advice of collection_write — the duplicate "
        f"rejection binds a call to it, and a scoped cycle can only make calls its "
        f"program's advice closure carries.  Declared: {CollectionWriteTool.advises}"
    )


def _assert_the_box_holds_its_recipes(db: Database) -> None:
    """The box holds exactly the two seeded recipes — the entries the forced write
    duplicates, and the baseline "keys unchanged" is measured against."""
    held = collection_entries(db, RECIPE_BOX.name)
    assert set(held) == set(RECIPE_BOX_SEED_KEYS), (
        f"the box must hold {sorted(RECIPE_BOX_SEED_KEYS)} when the cycle starts, got "
        f"{sorted(held)}"
    )


# ── Scoring ───────────────────────────────────────────────────────────────────


def _bound_matched_keys(db: Database) -> set[str]:
    """Every existing key the duplicate rejection bound into an ``update_entry`` call,
    read off the persisted tool results — the same text the model read.

    Read from the ledger rather than from a harness flag, and from the RESULT rather
    than from the injector's arguments: what the case pins is that the guard named the
    row that already exists, which is a fact about the message it sent back."""
    bound: set[str] = set()
    for message in _iter_prompt_messages(db):
        if message.get("role") != "tool":
            continue
        bound.update(_BOUND_KEY.findall(message.get("content") or ""))
    return bound


def _score_recovered_from_duplicate(expected_matches: tuple[str, ...]):
    """The case's scorer, bound to the keys the forced batch duplicates.

    ``guard_recovery_eval`` prepends its own "forced bail fired" guard, which says the
    injector ran; these say the GUARD answered it and the model acted on the answer."""

    def score(db: Database, sent: list[str]) -> list[Check]:
        bound = _bound_matched_keys(db)
        keys = set(collection_entries(db, RECIPE_BOX.name))
        keys_unchanged = keys == set(RECIPE_BOX_SEED_KEYS)
        recovered = tool_was_called(db, "done") or tool_was_called(db, _RECOVERY_VERB)
        stray = [key for key in tool_call_keys(db, _RECOVERY_VERB) if key not in keys]
        return [
            _rejection_bound_every_key_check(bound, expected_matches),
            Check(
                "the box still holds exactly the recipes it started with",
                keys_unchanged,
                anchor="collection_write(",
                kind="state",
                rationale=None
                if keys_unchanged
                else (
                    "the box's keys changed on an all-duplicate cycle — the model wrote "
                    f"something it had no source for: {sorted(keys)} against the seeded "
                    f"{sorted(RECIPE_BOX_SEED_KEYS)}"
                ),
            ),
            Check(
                "recovered — closed with done() or refreshed an existing entry",
                recovered,
                kind="proc",
                rationale=None
                if recovered
                else (
                    "the cycle neither closed nor refreshed anything after the duplicate "
                    "rejection — it key-hunted until the step ceiling.  Calls made: "
                    f"{tool_call_sequence(db) or 'none'}"
                ),
            ),
            _no_stray_key_check(stray, keys),
        ]

    return score


def _rejection_bound_every_key_check(bound: set[str], expected: tuple[str, ...]) -> Check:
    """The MECHANISM: the rejection named the row that already exists, once per rejected
    entry.

    The multi-entry batch is why this is a set comparison rather than a "did it say
    anything" probe — the gap #1405 closed was some rejected keys leaving with no match
    named, which one bound key would hide."""
    complete = set(expected) <= bound
    return Check(
        "the duplicate rejection bound every matched key into an update_entry call",
        complete,
        anchor="collection_write(",
        kind="guard",
        rationale=None
        if complete
        else (
            f"the rejection bound {sorted(bound) or 'nothing'} — it must bind "
            f"{sorted(expected)}, one per rejected entry, or the model has to guess which "
            "row it collided with"
        ),
    )


def _no_stray_key_check(stray: list[str], held: set[str]) -> Check:
    """The load-bearing check for #1405: every ``update_entry`` targeted a key the box
    really holds.  The 47%-recovery failure was the model re-using its OWN rejected
    candidate key — a key nothing is filed under — which returns key-not-found and
    starts the ping-pong."""
    return Check(
        "every update_entry targeted a key the box actually holds",
        not stray,
        anchor=f"{_RECOVERY_VERB}(",
        kind="spine",
        rationale=None
        if not stray
        else (
            f"update_entry aimed at {sorted(stray)}, which the box does not hold "
            f"({sorted(held)}) — the rejected candidate key re-used, which is the "
            "key-not-found ping-pong the bound key exists to prevent"
        ),
    )


async def test_duplicate_write_hands_back_key_and_recovers(guard_recovery_eval) -> None:
    """A single duplicate ``collection_write`` is rejected with the matched key BOUND
    into an update_entry call; the live model recovers (done() or update_entry on the
    bound key) without re-using its own rejected key."""
    await guard_recovery_eval(
        case_id="duplicate-write-recovery",
        family="collector-guard-recovery",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectDuplicateWrite(
            real, RECIPE_BOX.name, [(RECIPE_BOX_DUP_KEY, RECIPE_BOX_DUP_CONTENT)]
        ),
        score=_score_recovered_from_duplicate((RECIPE_BOX_SEED_KEYS[0],)),
        min_pass_rate=None,
    )


async def test_multi_entry_duplicate_write_binds_every_key(guard_recovery_eval) -> None:
    """A BATCH of two duplicate writes — each matching a DIFFERENT existing key — is
    rejected with EVERY matched key bound into its own update_entry call; the live
    model recovers without re-using either rejected candidate key.  Guards the
    multi-entry gap (some rejected keys previously left with no match named)."""
    await guard_recovery_eval(
        case_id="duplicate-write-recovery-multi",
        family="collector-guard-recovery",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectDuplicateWrite(
            real,
            RECIPE_BOX.name,
            [
                (RECIPE_BOX_DUP_KEY, RECIPE_BOX_DUP_CONTENT),
                (RECIPE_BOX_DUP_KEY_2, RECIPE_BOX_DUP_CONTENT_2),
            ],
        ),
        score=_score_recovered_from_duplicate(RECIPE_BOX_SEED_KEYS),
        min_pass_rate=None,
    )
