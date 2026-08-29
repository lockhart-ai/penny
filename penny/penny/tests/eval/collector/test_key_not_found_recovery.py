"""Key-not-found write-vs-update recovery contract — after a key-not-found
rejection the model finds the entry under a different key and must land the change
on THAT key, rather than proliferating a fresh one.

Production residue this pins (July 2026 tool-failure audit, item #11): the Jun-13
key-not-found rewrite ("list the keys with collection_keys(...), or search by
content with read_similar(...)") moved recovery 47% → 88%, but the remaining
failures share a shape — the model runs collection_keys, finds the right key, then
picks ``collection_write`` instead of ``update_entry``, because the rejection named
the READ tools but not the write-vs-update decision.  The rejection closes that
gap: it tells the model to refresh the existing entry with
``update_entry(key=<the key you found>, ...)`` and that ``collection_write``
creates NEW keys only.

The slip is a model DECISION on a visible tool result, and a natural cycle only
rarely probes an existing entry with a near-miss key, so we force ONE such
``collection_get`` (``_InjectKeyMiss``) and let the REAL model drive the recovery
off the production rejection message.  The contract is STRUCTURAL, never wording:

  PASS = the box's keys are UNCHANGED (no fresh / duplicate key) AND the existing
         fajitas entry was refreshed with the new detail — the model found the real
         key and landed the change on it, rather than proliferating a key or
         spiraling.  Whether it got there via ``update_entry`` is reported
         ADVISORY, for the reason below.

**The write-vs-update PREFERENCE is no longer a correctness claim.**  Since #1633
the write gate answers an exact-key write carrying a different value with
``KEY_EXISTS_CHANGED`` and refreshes the stored baseline IN PLACE, so a
``collection_write`` on the key the model found lands the change exactly as
``update_entry`` would — it is a correct recovery, not the duplicate-rejected
ping-pong the rejection was written against.  Scoring it as a miss would fail
correct behaviour, so the check is ``scored=False``: it still renders, and which
verb the model reaches for is still worth watching, but the graded claim is the
STATE.  Flagged on the #1919 audit as a candidate retirement.

**The world is the post-#1911 one, and that is what re-arms this case.**  The
program parser is STRICT (a step's call must OPEN the step, in the rendered
``N. tool(args)`` dialect) and the cycle's tool surface is SCOPED to that program's
own calls, closed over ``Tool.advises``.  The hand-authored prose recipe this
module used to seed carried no parseable call at all, so ``collection_get`` was not
on the surface and the injected probe hit tool-not-found instead of the
key-not-found rejection — the contract silently disarmed.  The collection is now
what an apply turn leaves behind (migration 0108 pre-seeds nothing): a taught
routine in the registry, and a container configured from it through the production
instantiation seam (retarget → ``bind_parameters`` → ``render_skill``).

The program names ``collection_get`` — the step the sabotage hijacks, so the probe
reaches the production rejection — and ``collection_write``, the verb the round was
demonstrated with.  Both RECOVERY reads and ``update_entry`` ride in on
``collection_get.advises`` (``collection_keys``, ``read_similar``,
``update_entry``, ``collection_write``), so every arm the rejection points at is
callable.  What the job is pointed at is a VALUE, not prose — the tool-neutral
"record it so the box reflects it" step this fixture used to carry has no rendered
form: the routine declares one parameter (the recipe it keeps current) and the
container binds it to the enriched text, which the composed prompt states as a term
of the job and the program carries in its own write leaf.

Report-only (``min_pass_rate=None``), the canonical convention, pending a joint
read of the re-armed cases.  The deterministic message content is pinned in
``tests/tools/test_memory_tools.py``; this owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.skills import (
    SkillDraft,
    SkillParameter,
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
    _InjectKeyMiss,
    _iter_prompt_messages,
    collection_entries,
    tool_call_sequence,
    tool_was_called,
)
from penny.tests.eval.utils.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_FAJITAS_KEY,
    RECIPE_BOX_FAJITAS_SEED_CONTENT,
    RECIPE_BOX_NEAR_MISS_KEY,
    RECIPE_BOX_SEED_KEYS,
)
from penny.tools.collection_instantiation import skill_params
from penny.tools.memory_tools import CollectionGetTool

pytestmark = pytest.mark.eval

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The calls the routine makes, in order — what the program must read back as under the
# strict dialect, and therefore what the cycle's surface is scoped to.
RECIPE_BOX_PROGRAM_CALLS = ("collection_get", "collection_write")

# The job's cadence — stated because a configured collection has one, though the case
# drives the cycle through ``run_for``, which bypasses readiness.
RECIPE_BOX_SCHEDULE = "FREQ=HOURLY"

# The verb the key-not-found rejection points the model at.
_RECOVERY_VERB = "update_entry"

# The recipe the job is pointed at: the SAME dish the box already holds, plus a detail
# it does not (the marinade).  So keeping the box current means landing this text on the
# entry that exists, and a fresh key would be the proliferation the case fails.  ONE
# constant because the runtime join (#1907) matches the declared parameter's
# demonstrated value against the leaf's — two spellings would join nothing.
RECIPE_BOX_ENRICHED_FAJITAS = (
    "Sheet-pan chicken fajitas — peppers, onion, chicken, 25 min at 425F, "
    "after a 10-minute lime marinade."
)

# The routine the user taught the box, in the shape run-end extraction leaves behind:
# every leaf a labelled PLACEHOLDER except the one the framer's parameter joins, and the
# destination additionally carrying the attachment mark (#1783).
RECIPE_BOX_ENRICH_SKILL = SkillDraft(
    name="keep_a_recipe_current",
    intent="Keep the box's copy of a recipe current when I learn something new about it.",
    description="Check what the box holds for a recipe and record the current version.",
    steps=[
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="collection_get",
            arguments={"memory": RECIPE_BOX.name, "key": RECIPE_BOX_FAJITAS_KEY},
            substitutions=[
                SkillSubstitution(
                    path=["memory"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the collection this is set up on",
                    attachment=True,
                ),
                SkillSubstitution(
                    path=["key"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the key the box already files this recipe under",
                ),
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
                        "key": RECIPE_BOX_FAJITAS_KEY,
                        "content": RECIPE_BOX_ENRICHED_FAJITAS,
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
                    description="the key the box already files this recipe under",
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the recipe as it now stands",
                ),
            ],
        ),
    ],
    parameters=[
        SkillParameter(
            name="recipe",
            description="the recipe to keep current, written out in full",
            value=RECIPE_BOX_ENRICHED_FAJITAS,
        )
    ],
    source_run_id="eval-seed",
)

# What the apply turn bound the routine's one parameter to.
RECIPE_BOX_ENRICH_VALUES = {"recipe": RECIPE_BOX_ENRICHED_FAJITAS}


def rendered_program() -> str:
    """The program the apply turn stores, through the production instantiation seam's
    own three steps in its own order (``render_skill_prompt``): the attachment bound to
    the container, the RUNTIME JOIN (#1907) writing the bound recipe into the leaf the
    demonstration put its own value in, then the render.  Public because the seeder and
    the probe both read it, and a second copy would be free to drift from what the
    collection stores."""
    attached = retarget_writes(RECIPE_BOX_ENRICH_SKILL.steps, RECIPE_BOX.name)
    joined = bind_parameters(attached, RECIPE_BOX_ENRICH_SKILL.parameters, RECIPE_BOX_ENRICH_VALUES)
    return render_skill(joined, RECIPE_BOX_ENRICH_VALUES)


def _seed_recipe_box(db: Database) -> None:
    """The world an apply turn leaves: the taught routine in the registry, the box with
    the two recipes it already holds (the fajitas one still un-enriched), and the
    container configured from that routine — then every claim that world makes,
    asserted."""
    db.skills.upsert(RECIPE_BOX_ENRICH_SKILL, author=_SEED_AUTHOR)
    db.memories.create_collection(
        RECIPE_BOX.name,
        RECIPE_BOX.description,
        extraction_prompt=rendered_program(),
        schedule=RECIPE_BOX_SCHEDULE,
        skill_name=slug_skill_name(RECIPE_BOX_ENRICH_SKILL.name),
        skill_params=RECIPE_BOX_ENRICH_VALUES,
    )
    require_memory(db, RECIPE_BOX.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in RECIPE_BOX.entries],
        author="user",
    )
    _assert_the_enrichment_world(db)


# ── The loud seed probe ───────────────────────────────────────────────────────


def _assert_the_enrichment_world(db: Database) -> None:
    """Everything the seeder is responsible for, asserted out loud.

    The disarm this module was in is the reason: an unreadable program yields an EMPTY
    tool surface, the injected ``collection_get`` then comes back tool-not-found instead
    of key-not-found, and the case scores a run in which the guard under test never
    spoke — silently, one live cycle per sample."""
    _assert_the_routine_is_registered(db)
    _assert_the_job_is_configured(db)
    _assert_the_program_parses(db)
    _assert_both_write_paths_are_reachable()
    _assert_the_box_holds_the_stale_recipe(db)


def _assert_the_routine_is_registered(db: Database) -> None:
    """The routine the container names is one the registry holds — else the composed
    prompt states the routine as gone and the values block falls back to bare names."""
    name = slug_skill_name(RECIPE_BOX_ENRICH_SKILL.name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"


def _assert_the_job_is_configured(db: Database) -> None:
    """The container the cycle runs is configured as an apply turn leaves it: the routine
    and the recipe it is pointed at stamped, the turn's schedule, live, and NOT
    notifying — this case scores what the cycle writes, never what it tells the user."""
    row = db.memories.get(RECIPE_BOX.name)
    assert row is not None, f"the job's container {RECIPE_BOX.name!r} must exist"
    assert row.skill_name == slug_skill_name(RECIPE_BOX_ENRICH_SKILL.name), (
        f"the job must run the routine the fixture taught, not {row.skill_name!r}"
    )
    assert skill_params(row) == RECIPE_BOX_ENRICH_VALUES, (
        f"the job must carry the recipe it is pointed at, got {skill_params(row)}"
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
    (#1911), carries the enriched recipe the runtime join filled, names the container it
    writes to, and stores no terminal ``done()``.

    ``collection_get`` being IN that list is what puts it on the cycle's scoped surface,
    which is what makes the injected near-miss probe reach the key-not-found rejection at
    all."""
    row = db.memories.get(RECIPE_BOX.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(
        call.tool for call in program_calls(program, frozenset(RECIPE_BOX_PROGRAM_CALLS))
    )
    assert parsed == RECIPE_BOX_PROGRAM_CALLS, (
        f"the stored program must read back as {list(RECIPE_BOX_PROGRAM_CALLS)} under the "
        f"rendered dialect, got {list(parsed)} — program: {program!r}"
    )
    assert RECIPE_BOX_ENRICHED_FAJITAS in program, (
        "the runtime join must fill the write leaf with the enriched recipe — the cycle "
        f"has no other source for what to record.  Program: {program!r}"
    )
    assert f"'{RECIPE_BOX.name}'" in program, (
        f"the attachment must be bound to {RECIPE_BOX.name!r} — a program carrying the "
        f"placeholder does not state where it writes.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying "
        "one is a render a chat ledger cannot produce"
    )


def _assert_both_write_paths_are_reachable() -> None:
    """BOTH arms of the write-vs-update decision are callable, so the choice is real.

    A scoped surface is the program's own calls closed over ``Tool.advises``, and the
    program names only ``collection_get`` and ``collection_write`` — ``update_entry``
    and the two recovery READS reach the surface because ``collection_get`` DECLARES
    them.  With ``update_entry`` off the surface the case would measure a decision the
    model could not make; read off the class attribute production closes over, so a
    dropped advice relation fails here instead of as an unexplained collapse."""
    for advised in (_RECOVERY_VERB, "collection_keys", "read_similar"):
        assert advised in CollectionGetTool.advises, (
            f"{advised!r} must be declared advice of collection_get — the key-not-found "
            f"rejection names it, and a scoped cycle can only make calls its program's "
            f"advice closure carries.  Declared: {CollectionGetTool.advises}"
        )


def _assert_the_box_holds_the_stale_recipe(db: Database) -> None:
    """The box holds exactly the two seeded recipes, with the fajitas one still at its
    PRE-enrichment content — the state "was it refreshed?" is measured against, and the
    reason the near-miss key misses in the first place."""
    held = collection_entries(db, RECIPE_BOX.name)
    assert set(held) == set(RECIPE_BOX_SEED_KEYS), (
        f"the box must hold {sorted(RECIPE_BOX_SEED_KEYS)} when the cycle starts, got "
        f"{sorted(held)}"
    )
    assert held[RECIPE_BOX_FAJITAS_KEY] == RECIPE_BOX_FAJITAS_SEED_CONTENT, (
        "the fajitas entry must start un-enriched, else 'was it refreshed?' is already "
        f"true before the cycle runs — got {held[RECIPE_BOX_FAJITAS_KEY]!r}"
    )
    assert RECIPE_BOX_NEAR_MISS_KEY not in held, (
        f"the forced probe's key {RECIPE_BOX_NEAR_MISS_KEY!r} must MISS — a box holding "
        "it would return the entry and the rejection would never fire"
    )


# ── Scoring ───────────────────────────────────────────────────────────────────


def _probe_hit_the_rejection(db: Database) -> bool:
    """Did the forced near-miss probe come back with the production key-not-found
    rejection, read off the persisted tool results — the same text the model read?

    A substring of the rejection, not a copy of it: the full wording is pinned in
    ``tests/tools/test_memory_tools.py``, and what this needs to tell apart is the
    rejection from the tool-not-found error a disarmed surface returns instead."""
    marker = f"Key '{RECIPE_BOX_NEAR_MISS_KEY}' not found in '{RECIPE_BOX.name}'"
    return any(
        message.get("role") == "tool" and marker in (message.get("content") or "")
        for message in _iter_prompt_messages(db)
    )


def _score_landed_on_the_found_key(db: Database, sent: list[str]) -> list[Check]:
    """The model recovered from the key-not-found rejection onto the key the box really
    uses: the box's keys are unchanged (no proliferated or duplicate key) and the
    enrichment landed on the existing fajitas entry.

    Which verb it used is ADVISORY — #1633's write gate refreshes an exact-key write in
    place, so ``collection_write`` on the found key is a correct recovery.
    (``guard_recovery_eval`` prepends its own "forced bail fired" guard, which says the
    injector ran; the first check here says the GUARD answered it.)"""
    entries = collection_entries(db, RECIPE_BOX.name)
    keys = set(entries)
    keys_unchanged = keys == set(RECIPE_BOX_SEED_KEYS)
    reached_update = tool_was_called(db, _RECOVERY_VERB)
    refreshed = entries.get(RECIPE_BOX_FAJITAS_KEY, "") != RECIPE_BOX_FAJITAS_SEED_CONTENT
    hit = _probe_hit_the_rejection(db)
    return [
        Check(
            "the near-miss probe came back with the key-not-found rejection",
            hit,
            anchor="collection_get(",
            kind="guard",
            rationale=None
            if hit
            else (
                "the forced probe never reached the rejection this case is about — on a "
                "scoped surface without collection_get it comes back tool-not-found "
                f"instead.  Calls made: {tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "the box still holds exactly the recipes it started with",
            keys_unchanged,
            anchor="collection_write(",
            kind="state",
            rationale=None
            if keys_unchanged
            else (
                "the box's keys changed — the enrichment was filed under a fresh key "
                f"instead of the one the box uses: {sorted(keys)} against the seeded "
                f"{sorted(RECIPE_BOX_SEED_KEYS)}"
            ),
        ),
        Check(
            "the enrichment landed on the entry the box already had",
            refreshed,
            kind="state",
            rationale=None
            if refreshed
            else (
                f"{RECIPE_BOX_FAJITAS_KEY!r} still holds its pre-enrichment content — the "
                "model found the key or it did not, but nothing was recorded against it"
            ),
        ),
        Check(
            "reached update_entry rather than collection_write",
            reached_update,
            anchor=f"{_RECOVERY_VERB}(",
            scored=False,
            kind="proc",
            rationale=None
            if reached_update
            else (
                "landed the change without update_entry — since #1633 an exact-key "
                "collection_write refreshes the stored baseline in place, so this is a "
                "correct recovery rather than a miss, which is why the row is advisory"
            ),
        ),
    ]


async def test_key_not_found_recovers_to_update_not_write(guard_recovery_eval) -> None:
    """A near-miss ``collection_get`` returns the key-not-found rejection; the live
    model finds the real key and refreshes the existing entry on it rather than
    proliferating a fresh one."""
    await guard_recovery_eval(
        case_id="key-not-found-write-vs-update",
        family="collector-guard-recovery",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectKeyMiss(real, RECIPE_BOX.name, RECIPE_BOX_NEAR_MISS_KEY),
        score=_score_landed_on_the_found_key,
        min_pass_rate=None,
    )
