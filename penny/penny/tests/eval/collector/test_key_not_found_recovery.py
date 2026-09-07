"""Key-not-found recovery: the change lands on the key the collection really uses.

Ported to the cohort structure under #2060; the contract is `docs/eval-case-design.md`.

Mid-cycle, the harness makes the job's own ``collection_get`` step probe a NEAR-MISS key —
a key close to, but not equal to, the one the recipe is actually filed under — so the
production key-not-found rejection comes back.  That rejection names the READ tools and
the write-vs-update decision; it does NOT name the key, because nothing knows it yet.  So
the live model has to go and FIND the key the box uses and record the current version on
it, rather than filing a second entry beside the one that exists.

**Why this is its own case, and not the duplicate-write family's (#2060's first job).**  A
duplicate-write rejection HANDS the model the matched key (``update_entry(key='<matched>',
…)``) — the recovery is a copy of a rendered anchor.  Here the anchor does not exist: the
key is withheld, and the recovery is a lookup followed by a write.  Those are materially
different decisions, so a correct sample for one says nothing about the other.

  * ONE CYCLE.  The entry condition — the box holding the recipe under its real key, still
    un-enriched — is what selects the behaviour.  Preseeded through the store's own write
    path, under the key the program writes to.
  * THE FAULT IS FIXED, the WORDING varies.  A collector has no user turn, so its natural
    language is the ``SkillSubstitution`` descriptions its rendered program carries, written
    upstream and printed by ``render_skill`` — the same seam an ``extract`` instruction
    reaches the model through.  Five wordings of the one that says which key to work on;
    the near-miss probe, the box and the recipe are byte-identical on every arm.
  * THE INJECTION IS HARNESS MACHINERY.  A sample the probe never fired on ran an unbroken
    cycle and exercised no recovery, so it leaves the cohort as a named exclusion rather
    than counting as a behavioural failure.  THE COST, STATED (#2018): the exclusion reads
    ``bail_injected``, which is set when the probe is ISSUED — being issued is not being
    ANSWERED with the key-not-found rejection.  A probe the store answered some other way
    is POOLED AND JUDGED against claims about a recovery it never had to make.  That is the
    residual hole, still open.  This injector hijacks the FIRST model call whatever it is,
    so the exclusion is in practice unreachable here: a cycle that reached the model at all
    issued the probe, and one that did not is already excluded as a dead cycle.
  * LANDED renders EMPTY, and that is the correct report: a collector moves no conversation
    machine, so there is no walk to read a landing off.  The run record's outcome is a
    RECORD FIELD and is claimed under STORE.
  * the tool calls MEASURED rather than asserted — the recovery route is exactly where many
    correct runs differ (``collection_keys`` or ``read_similar``, then ``update_entry`` or
    an exact-key ``collection_write``, which since #1633 refreshes the baseline in place).

REPORT-ONLY (``min_pass_rate=None``): the floors and ceilings this run proposes are the
code owner's to accept once the numbers have been read.  All content is synthetic — a
weeknight recipe box — because the repo is public.

The deterministic rejection wording is pinned in ``tests/tools/test_memory_tools.py``; this
owns the live model-behaviour contract.
"""

from __future__ import annotations

import pytest

from penny.constants import RunOutcome
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
    EVAL_MODELS,
    CollectorCyclesEval,
    CycleArm,
    _InjectKeyMiss,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    StoredEntry,
)
from penny.tests.eval.utils.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_FAJITAS_KEY,
    RECIPE_BOX_FAJITAS_SEED_CONTENT,
    RECIPE_BOX_NEAR_MISS_KEY,
    RECIPE_BOX_SEED_KEYS,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.collection_instantiation import skill_params
from penny.tools.memory_tools import CollectionGetTool

pytestmark = pytest.mark.eval

_CASE_ID = "key-not-found-write-vs-update"

# The family every collector recovery case reports under — what a cycle does once one of
# its own calls has already been refused.  A second spelling would split one bucket.
_FAMILY = "collector-recovery"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The run that laid the entry condition down.  A SEEDED id, structurally distinguishable
# from a live ``uuid4().hex``, so every "what did THIS sample write" reader excludes the
# preseeded rows at the one chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("recipe-box-entry-condition")

# The calls the routine makes, in order — what the stored program must read back as under
# the strict rendered dialect, and therefore what this cycle's tool surface is scoped to.
_PROGRAM_CALLS = ("collection_get", "collection_write")

# The job's cadence.  Stated because a configured collection has one, though the cycle is
# driven through ``run_for``, which bypasses readiness.
_SCHEDULE = "FREQ=HOURLY"

# The verb the key-not-found rejection points the model at.  Named here because the probe
# below asserts it is REACHABLE on a scoped surface, never because a claim reads it.
_RECOVERY_VERB = "update_entry"

# ── The facts, held constant across every arm ────────────────────────────────
#
# The recipe the job is pointed at: the SAME dish the box already holds, plus a detail it
# does not.  So keeping the box current means landing this text on the entry that exists,
# and a fresh key is the proliferation this case is named against.  ONE constant, because
# the runtime join (#1907) matches the declared parameter's demonstrated value against the
# leaf's — two spellings would join nothing.
_ENRICHED_RECIPE = (
    "Sheet-pan chicken fajitas — peppers, onion, chicken, 25 min at 425F, "
    "after a 10-minute lime marinade."
)

# The smallest token that says the enrichment landed.  The identity of this change is the
# step the box did not have; every other word in the enriched text is already in the entry
# the box was seeded with, so a larger expectation would only add words the write was free
# to reword.
#
# SHRUNK from the whole word after the first measured run.  ``marinade`` failed 10 of 15
# gpt-oss samples that had done exactly what the case is named for — found the key the box
# really uses and landed the enrichment on it — because the write RETYPED the value the
# program handed it: ``marinate``, ``marination``, ``marina des``.  The thinking traces
# quote the program's own wording correctly and then re-word it at the write, which is the
# model choosing how to write a value rather than which value to write, and #1994's rule
# excludes exactly that from an assertion.  The other model wrote it verbatim 15 of 15, so
# the whole word was measuring how much like one model the other is.
#
# The stem is where the two shrink questions both answer.  Would a differently-worded
# correct answer fail it?  No — every observed correct variant carries it.  Could a wrong
# value satisfy it?  No — the un-enriched text carries no ``marin`` at all, and neither does
# the one sample that recorded a different step entirely (``lime garnish``), which stays a
# genuine miss.  It appears nowhere else in this world: not in either seeded recipe, not in
# either key, not in the collection's description.
_ENRICHMENT = "marin"


# Five wordings of one instruction — the substitution description that becomes ``{…}`` in
# the rendered program at both key leaves, which is the only natural language this cycle is
# handed, and the row a reader opens when the report says a phrasing diverged.  Every one
# says the same thing — the box already files this recipe somewhere, work on THAT — and none
# of them names the key, because naming it would answer the question the case is asking.
KEY_HINTS = (
    "the key the box already files this recipe under",
    "the key this recipe is saved under in the box",
    "the name the box already keeps this recipe by",
    "the key of the entry the box already holds for this recipe",
    "whatever the box has this recipe filed as",
)


# The one sentence this case exists to check, in the fixed form: "In <the locus>, when <X>,
# Penny <does Y>."  The locus is the SHIPPED name of where the behaviour happens.  The case
# id is a filename; this is the contract, and it renders above every number in the report.
_BEHAVIOUR = (
    "In a recipe-box collector, when the key she looked the recipe up by does not exist, "
    "Penny finds the key the box really files it under and records the current version on "
    "that entry, rather than filing a second one beside it."
)


def _skill(key_hint: str) -> SkillDraft:
    """The routine the user taught the box, in the shape run-end extraction leaves behind.

    ONE shape for every arm — the same two steps, the same placeholders, the same declared
    parameter, the same attachment mark on the destination.  What differs is the KEY
    substitution's description, printed by ``render_skill`` at both key leaves, which is
    the arm axis.  The demonstrated ``arguments`` stay constant across the arms because the
    key leaves never render: the labeller's description replaces them at the seam."""
    return SkillDraft(
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
                        description=key_hint,
                    ),
                ],
            ),
            SkillStep(
                ordinal=2,
                source_ordinal=2,
                tool="collection_write",
                arguments={
                    "memory": RECIPE_BOX.name,
                    "entries": [{"key": RECIPE_BOX_FAJITAS_KEY, "content": _ENRICHED_RECIPE}],
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
                        description=key_hint,
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
                value=_ENRICHED_RECIPE,
            )
        ],
        source_run_id=_SEED_AUTHOR,
    )


_VALUES = {"recipe": _ENRICHED_RECIPE}


def _program(key_hint: str) -> str:
    """The program the apply turn stores, through the production instantiation seam's own
    three steps in its own order: the attachment bound to the container, the runtime join
    writing the bound recipe into the leaf the demonstration put its own value in, then the
    render."""
    skill = _skill(key_hint)
    attached = retarget_writes(skill.steps, RECIPE_BOX.name)
    joined = bind_parameters(attached, skill.parameters, _VALUES)
    return render_skill(joined, _VALUES)


def _key_slot(key_hint: str) -> str:
    """How this arm's instruction renders inside the program — the one span that moves."""
    return f"{{{key_hint}}}"


# Every arm's ground, and there is ONE of it: this job reads no page.  A world's pages are
# what a browse-driven cycle is answered against; this routine's only source is the box
# itself and the recipe its own program carries, both of which reach the provenance claim
# through what the model was GIVEN.  So the arms genuinely share their ground and the report
# states it once — a world per arm would print five distinct grounds that differ in nothing
# but a label.  ``keeps``/``excludes`` are empty for the reason ``test_watch_cycles.py``'s
# are: they back a claim this case never makes, and declaring tokens nothing reads prints a
# contract that looks verified.
_WORLD = World(name="recipe-box", pages=(), keeps=(), excludes=())


def _seeder(key_hint: str):
    """The world this cycle starts in: the routine in the registry, a container configured
    from it, and the ENTRY CONDITION — then every claim that world makes, asserted out loud.

    The entry condition goes down through the store's own write path, under the key the
    program writes to, so the rows carry the stamps and the embeddings a real cycle's write
    would have left.  The probe is not ceremony: a program the strict parser cannot read
    leaves the cycle with a surface of the terminator alone, so the forced probe comes back
    tool-not-found instead of key-not-found and the case measures a guard that never spoke —
    silently, at one live cycle per sample."""

    def seed(db: Database) -> None:
        skill = _skill(key_hint)
        db.skills.upsert(skill, author=_SEED_AUTHOR)
        db.memories.create_collection(
            RECIPE_BOX.name,
            RECIPE_BOX.description,
            extraction_prompt=_program(key_hint),
            schedule=_SCHEDULE,
            skill_name=slug_skill_name(skill.name),
            skill_params=_VALUES,
        )
        require_memory(db, RECIPE_BOX.name).write(
            [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in RECIPE_BOX.entries],
            author=_SEED_AUTHOR,
            run_id=_SEED_RUN,
        )
        _assert_the_enrichment_world(db, key_hint)

    return seed


# ── The loud seed probe ───────────────────────────────────────────────────────


def _assert_the_enrichment_world(db: Database, key_hint: str) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    _assert_the_job_is_configured(db, key_hint)
    _assert_the_program_parses(db, key_hint)
    _assert_both_write_paths_are_reachable()
    _assert_the_entry_condition(db)


def _assert_the_job_is_configured(db: Database, key_hint: str) -> None:
    """The routine is registered and the container is configured as an apply turn leaves it:
    the routine stamped, the recipe it is pointed at bound, live, and NOT notifying."""
    name = slug_skill_name(_skill(key_hint).name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"
    row = db.memories.get(RECIPE_BOX.name)
    assert row is not None, f"the job's container {RECIPE_BOX.name!r} must exist"
    assert row.skill_name == name, f"the job must run the taught routine, not {row.skill_name!r}"
    assert skill_params(row) == _VALUES, (
        f"the job must carry the recipe it is pointed at, got {skill_params(row)}"
    )
    assert not row.archived and not row.notify, (
        "the job must be live and SILENT — this case scores what the cycle writes, and a "
        "notifying job would put a second claim in front of the user that nothing here reads"
    )


def _assert_the_program_parses(db: Database, key_hint: str) -> None:
    """The stored program reads back as the calls it makes under the STRICT dialect (#1911),
    carries the enriched recipe the runtime join filled and this arm's own instruction, names
    the container it writes to, and stores no terminal ``done()``.

    ``collection_get`` being IN that list is what puts it on the cycle's scoped surface,
    which is what makes the forced near-miss probe reach the key-not-found rejection at
    all."""
    row = db.memories.get(RECIPE_BOX.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_PROGRAM_CALLS)))
    assert parsed == _PROGRAM_CALLS, (
        f"the stored program must read back as {list(_PROGRAM_CALLS)} under the rendered "
        f"dialect, got {list(parsed)} — program: {program!r}"
    )
    assert _ENRICHED_RECIPE in program, (
        "the runtime join must fill the write leaf with the enriched recipe — the cycle has "
        f"no other source for what to record.  Program: {program!r}"
    )
    assert f"'{RECIPE_BOX.name}'" in program, (
        f"the attachment must be bound to {RECIPE_BOX.name!r}.  Program: {program!r}"
    )
    assert _key_slot(key_hint) in program, (
        f"this arm's instruction must reach the model as {_key_slot(key_hint)!r} — the key "
        f"description is the whole arm axis.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying one "
        "is a render a chat ledger cannot produce"
    )


def _assert_both_write_paths_are_reachable() -> None:
    """The calls the key-not-found rejection names are on the cycle's surface.

    A scoped surface is the program's own calls closed over ``Tool.advises``, and the
    program names only ``collection_get`` and ``collection_write`` — the two recovery READS
    and ``update_entry`` reach the surface because ``collection_get`` DECLARES them.  With
    them off the surface the case would measure a recovery the model could not make; read
    off the class attribute production closes over, so a dropped advice relation fails here
    rather than as an unexplained collapse."""
    for advised in (_RECOVERY_VERB, "collection_keys", "read_similar"):
        assert advised in CollectionGetTool.advises, (
            f"{advised!r} must be declared advice of collection_get — the key-not-found "
            f"rejection names it, and a scoped cycle can only make calls its program's "
            f"advice closure carries.  Declared: {CollectionGetTool.advises}"
        )


def _assert_the_entry_condition(db: Database) -> None:
    """The state the cycle starts from, which is what selects the behaviour being measured.

    The recipe has to be in the box under its REAL key and still un-enriched — else "was it
    refreshed?" is already true before the cycle runs — and the near-miss key has to MISS,
    else the probe returns the entry and the rejection never fires."""
    held = collection_entries(db, RECIPE_BOX.name)
    assert set(held) == set(RECIPE_BOX_SEED_KEYS), (
        f"the box must hold {sorted(RECIPE_BOX_SEED_KEYS)} when the cycle starts, got "
        f"{sorted(held)}"
    )
    assert held[RECIPE_BOX_FAJITAS_KEY] == RECIPE_BOX_FAJITAS_SEED_CONTENT, (
        f"the recipe must start un-enriched, got {held[RECIPE_BOX_FAJITAS_KEY]!r}"
    )
    assert not any(_ENRICHMENT in f"{key} {content}" for key, content in held.items()), (
        f"{_ENRICHMENT!r} must appear nowhere in the box the cycle starts from — it is the "
        f"token that says the enrichment landed, and uniqueness is a property of the WORLD "
        f"rather than of the token, so a seeded row already carrying it makes the claim "
        f"vacuous.  The box holds {held}"
    )
    assert RECIPE_BOX_NEAR_MISS_KEY not in held, (
        f"the forced probe's key {RECIPE_BOX_NEAR_MISS_KEY!r} must MISS — a box holding it "
        "would return the entry and the rejection would never fire"
    )


def _arms() -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction over one set of facts.

    ``text`` is the instruction, because that is what makes this arm this arm.  ``pages`` is
    empty: the job browses nothing, so its cycle is handed no register at all."""
    return [
        CycleArm(text=key_hint, seed=_seeder(key_hint), pages=[], world=_WORLD)
        for key_hint in KEY_HINTS
    ]


# ── The claims ───────────────────────────────────────────────────────────────
#
# Every one of them reads END STATE: what the box holds, and what the run record says.
# Nothing reads a tool name or an ordering — many routes reach one end state, and here they
# genuinely differ (``collection_keys`` or ``read_similar`` to find the key; ``update_entry``
# or an exact-key ``collection_write``, which since #1633 refreshes the baseline in place),
# so the route is measured in section B and never asserted.


def _held_key(entry: StoredEntry) -> str:
    """One held entry's key.  A collection entry always has one; the empty string is the
    keyless-log shape the type allows and this collection never has."""
    return entry.key or ""


def _still_holds_the_same_recipes(sample: SampleObservation, _world: World) -> Answer:
    """The box holds exactly the recipes it was seeded with — no key more, no key fewer.

    This is the ABSENCE half of the behaviour, and it has a concrete violating shape: a
    THIRD key, the enrichment filed beside the entry it was about instead of on it.  That
    is the residue the case is named for, and it is what a model does when it never finds
    the real key or finds it and writes anyway."""
    keys = sorted(_held_key(entry) for entry in sample.held)
    expected = sorted(RECIPE_BOX_SEED_KEYS)
    return keys == expected, f"the box holds {keys}, against the seeded {expected}"


def _the_recipe_carries_the_enrichment(sample: SampleObservation, _world: World) -> Answer:
    """The entry the box already had now carries the step it did not.

    Read over the WHOLE entry — key and content — because a fact in the key and a blurb in
    the body is a perfectly good way to store it.  Matched on the STEM of the added step,
    for the measured reason recorded above ``_ENRICHMENT``: the identity of this change is
    the step that was added, and which inflection the write happened to type is the model
    choosing how to render a value rather than which value to record."""
    entry = next((e for e in sample.held if _held_key(e) == RECIPE_BOX_FAJITAS_KEY), None)
    if entry is None:
        return False, f"the box no longer holds {RECIPE_BOX_FAJITAS_KEY!r} at all"
    return _ENRICHMENT in entry.text, (
        f"expected {_ENRICHMENT!r}; {RECIPE_BOX_FAJITAS_KEY!r} holds {entry.text!r}"
    )


def _closed_having_worked(sample: SampleObservation, _world: World) -> Answer:
    """The run's own determination: it completed and it CHANGED something.

    A record field, read literally off ``promptlog.run_outcome`` — ``worked`` is defined as
    "completed and changed something", which is what "the recovery landed" means without
    reading a tool name.  ``no_work`` is a clean close that changed nothing and ``failed``
    is a bail; both are the wrong end state for a cycle that owed a write."""
    closed = sample.run_outcome or "—"
    detail = f" — {sample.run_reason}" if sample.run_reason else ""
    return sample.run_outcome == RunOutcome.WORKED.value, f"the run closed {closed}{detail}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_key_not_found_recovers_onto_the_key_the_box_uses(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A near-miss lookup misses; the live model finds the key the box really uses and
    records the current version on that entry rather than proliferating a fresh one."""
    cohort = await collector_cycles_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        collection=RECIPE_BOX.name,
        arms=_arms(),
        samples_per_phrasing=3,
        wrap_client=lambda real: _InjectKeyMiss(real, RECIPE_BOX.name, RECIPE_BOX_NEAR_MISS_KEY),
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.  A collector moves no conversation
    # machine, and the run record's fields are STORE claims below rather than a category
    # borrowed to fill a heading.

    # STORE
    cohort.claim(
        "state: the box still holds exactly the recipes it started with",
        _still_holds_the_same_recipes,
        SpecCategory.STORE,
    )
    cohort.claim(
        f"state: the recipe the box already had now carries the added step ({_ENRICHMENT})",
        _the_recipe_carries_the_enrichment,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the run closed having changed something", _closed_having_worked, SpecCategory.STORE
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()

    # REPLY_SPREAD is not measured: this job is silent, so a correct cohort sends nothing and
    # a reply-spread reading would be blind on every sample by construction.
    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED)
