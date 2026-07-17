"""Skill substrate tests (#1590) — the render, provenance inference, certified-by
-execution, and the seed library, driven through the tool entry points with
deterministic fixtures and fictional content only.
"""

from __future__ import annotations

import json

import pytest

from penny.database import Database
from penny.database.migrate import migrate
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillHole,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    distill_steps,
    render_skill,
    retarget_writes,
    unbound_required_holes,
)
from penny.tools.skill_tools import SkillReadTool

# ── Fixtures: a fictional "watch the elevation of a peak" demonstration ────────
#
# Structural provenance (#1659): the browse query and the extract instruction are
# non-binding string leaves → HOLES; the extracted reading ("1,842 m") flows
# step 1 → step 2 as a BINDING; the write TARGET ("elevations") is owned by
# write-retarget, never a hole.  All fictional.

_UTTERANCE = "Save the Zephyr Ridge elevation to my notes"
_EXTRACTED_VALUE = "1,842 m"

# Every real tool call the framework logs carries the universal ``reasoning``
# think-aloud (``Tool.to_ollama_tool`` injects it) — the model's per-call
# narration.  The fixtures inject it so a demonstration matches a REAL promptlog
# (the #1661 divergence: the old fixtures omitted it, so distill never had to
# strip it and a real run surfaced nonsense ``reasoning`` holes).  Distill drops
# the top-level ``reasoning`` outright — never a hole, never a stored/rendered arg.
_REASONING = "because the user asked me to save it"

_BROWSE_ARGS = {"queries": ["Zephyr Ridge elevation"], "extract": "the elevation above sea level"}
_WRITE_ARGS = {
    "memory": "elevations",
    "entries": [{"key": "Zephyr Ridge elevation", "content": _EXTRACTED_VALUE}],
}

_BROWSE_OK = (
    f"You used `browse` and here's the result: (browse result)\nEXTRACTED: {_EXTRACTED_VALUE}"
)
_WRITE_OK = (
    "You saved an entry to elevations: (collection_write result)\n"
    "Wrote 1 entry to 'elevations': Zephyr Ridge elevation."
)
_BROWSE_FAILED = (
    "You searched for 'Zephyr Ridge elevation' but couldn't read anything "
    "(browse result)\n## browse error: unreachable"
)


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    return db


def _migrated_db(tmp_path) -> Database:
    """A DB built exactly like prod (create_tables then migrate) — what a fresh
    install's skill registry actually contains (migration 0084: table, no rows)."""
    db = Database(str(tmp_path / "seeded.db"))
    db.create_tables()
    migrate(db.db_path)
    return db


def _elevation_steps() -> list[SkillStep]:
    """The distilled steps for the fixture, built directly (independent of the
    inference path) so the render is pinned in isolation."""
    return [
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments=dict(_BROWSE_ARGS),
            substitutions=[
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, hole="queries"),
                SkillSubstitution(path=["extract"], kind=SkillSubKind.HOLE, hole="extract"),
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments=json.loads(json.dumps(_WRITE_ARGS)),
            substitutions=[
                SkillSubstitution(
                    path=["entries", 0, "key"], kind=SkillSubKind.HOLE, hole="queries"
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"], kind=SkillSubKind.BINDING, step=1
                ),
            ],
        ),
    ]


# ── The render: with-holes and the money literal ──────────────────────────────

_WITH_HOLES = (
    "1. browse(queries=[{queries}], extract={extract})\n"
    "2. collection_write(memory='elevations', "
    "entries=[{'key': {queries}, 'content': the value from step 1}])"
)

# THE money literal — steps + bound params → the numbered TEXT extraction_prompt a
# future collection actually runs.  Holes substituted verbatim; the binding reads
# as a legible instruction.
_MONEY_LITERAL = (
    "1. browse(queries=['Cinder Peak elevation'], extract='the elevation above sea level')\n"
    "2. collection_write(memory='elevations', "
    "entries=[{'key': 'Cinder Peak elevation', 'content': the value from step 1}])"
)


def test_render_skill_with_holes_is_the_template():
    """An unbound skill renders holes as ``{name}`` placeholders and the binding as
    a legible instruction — the with-holes recipe the read surface shows."""
    assert render_skill(_elevation_steps()) == _WITH_HOLES


def test_render_skill_bound_is_the_money_literal():
    """steps + bound params → the numbered text prompt a collection will run: every
    hole substituted with the param value verbatim, the binding kept legible."""
    rendered = render_skill(
        _elevation_steps(),
        {"queries": "Cinder Peak elevation", "extract": "the elevation above sea level"},
    )
    assert rendered == _MONEY_LITERAL


# ── Provenance inference: binding / hole / write-target in one run ─────────────


def test_distill_classifies_binding_holes_and_write_target():
    """Structural provenance (#1659): a value that flowed from a prior result is a
    BINDING; every other string leaf is a REQUIRED hole (shared values collapse to
    one); the scoped-write target is NOT a hole — write-retarget owns it."""
    inputs = [
        DistillInput(source_ordinal=1, tool="browse", arguments=_BROWSE_ARGS, result=_BROWSE_OK),
        DistillInput(
            source_ordinal=2, tool="collection_write", arguments=_WRITE_ARGS, result=_WRITE_OK
        ),
    ]
    steps, holes = distill_steps(inputs)

    # Two required holes — the browse query and the extract instruction; the write
    # KEY reuses the query's hole (same value → one shared parameter).
    assert holes == [
        SkillHole(name="queries", required=True),
        SkillHole(name="extract", required=True),
    ]
    # Every hole is required, so an unbound instantiation refuses naming each one;
    # binding them all clears the validation (#1591/#1659, no silent default).
    assert unbound_required_holes(holes, {}) == ["queries", "extract"]
    assert unbound_required_holes(holes, {"queries": "x", "extract": "y"}) == []

    # Step 1: the query and the extract instruction are both HOLES.
    step1 = {tuple(s.path): s for s in steps[0].substitutions}
    assert step1[("queries", 0)].kind == SkillSubKind.HOLE
    assert step1[("queries", 0)].hole == "queries"
    assert step1[("extract",)].kind == SkillSubKind.HOLE

    # Step 2: the key is the SHARED 'queries' hole; the content is a BINDING to step
    # 1's result; the write TARGET ('memory') is a retarget-owned constant — no sub.
    step2 = {tuple(s.path): s for s in steps[1].substitutions}
    assert step2[("entries", 0, "key")].kind == SkillSubKind.HOLE
    assert step2[("entries", 0, "key")].hole == "queries"
    assert step2[("entries", 0, "content")].kind == SkillSubKind.BINDING
    assert step2[("entries", 0, "content")].step == 1
    assert ("memory",) not in step2  # write-target owned by retarget, not parameterized
    assert steps[1].arguments["memory"] == "elevations"  # the constant demo value stays


def test_distill_binds_a_wrapped_prior_result():
    """A binding is structural, not equality: the value binds when it CONTAINS a
    prior result (the model wrapped '$499' into 'Price: $499 today'), #1659."""
    inputs = [
        DistillInput(
            source_ordinal=1, tool="browse", arguments={"queries": ["gadget price"]}, result="$499"
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "prices",
                "entries": [{"key": "gadget", "content": "Price: $499 today"}],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs)
    content = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    assert content.kind == SkillSubKind.BINDING and content.step == 1


def test_distill_does_not_bind_a_trivial_overlap():
    """The binding guard: a sub-``_MIN_BINDING_OVERLAP`` coincidence never binds — a
    1-char prior result contained in a longer arg stays a hole, not a false binding."""
    inputs = [
        DistillInput(source_ordinal=1, tool="browse", arguments={"queries": ["fruit"]}, result="a"),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={"memory": "fruits", "entries": [{"key": "k", "content": "banana"}]},
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs)
    content = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    # 'a' (len 1) is inside 'banana' but too trivial to bind → 'banana' stays a hole.
    assert content.kind == SkillSubKind.HOLE


def test_distill_strips_the_top_level_reasoning_thinkaloud():
    """#1661: the universal top-level ``reasoning`` think-aloud every real call carries
    is stripped at distill — it adds NO hole, never lands in a stored step's
    arguments, and never renders (it is per-run narration; the executing model
    supplies its own reasoning at run time)."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={**_BROWSE_ARGS, "reasoning": _REASONING},
            result=_BROWSE_OK,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={**_WRITE_ARGS, "reasoning": "because it flowed from step 1"},
            result=_WRITE_OK,
        ),
    ]
    steps, holes = distill_steps(inputs)
    # Same two holes as the reasoning-free run (test above) — the think-aloud added none.
    assert holes == [
        SkillHole(name="queries", required=True),
        SkillHole(name="extract", required=True),
    ]
    assert all("reasoning" not in step.arguments for step in steps)
    assert "reasoning=" not in render_skill(steps)


def test_distill_keeps_a_nested_key_named_reasoning():
    """Only the TOP-LEVEL ``reasoning`` is stripped (#1661): a NESTED arg that merely
    shares the name is real routine data — it stays in the stored step (and, as a
    non-binding string leaf, is a hole like any other)."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="collection_write",
            arguments={
                "memory": "notes",
                "reasoning": "top-level narration — stripped",
                "entries": [{"key": "k", "content": "c", "reasoning": "nested — kept"}],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs)
    args = steps[0].arguments
    assert "reasoning" not in args  # the top-level think-aloud is gone
    assert args["entries"][0]["reasoning"] == "nested — kept"  # the nested key is untouched


# ── skill_read: render one / list all ─────────────────────────────────────────


def _seed_skill(db: Database, name: str) -> None:
    """Persist a skill directly (the extractor's output shape) so the read surface
    can be tested independently of the extraction path."""
    steps, holes = distill_steps(
        [
            DistillInput(
                source_ordinal=1, tool="browse", arguments=_BROWSE_ARGS, result=_BROWSE_OK
            ),
            DistillInput(
                source_ordinal=2, tool="collection_write", arguments=_WRITE_ARGS, result=_WRITE_OK
            ),
        ]
    )
    db.skills.upsert(
        SkillDraft(
            name=name,
            intent=_UTTERANCE,
            description=_UTTERANCE,
            steps=steps,
            holes=holes,
            source_run_id="run-A",
        ),
        author="chat",
        description_embedding=None,
    )


@pytest.mark.asyncio
async def test_skill_read_renders_one_and_lists_all(tmp_path):
    """``skill_read(name)`` renders one full recipe; bare ``skill_read()`` lists
    every skill; an unknown name is an actionable miss."""
    db = _make_db(tmp_path)
    _seed_skill(db, "Watch elevation")
    read = SkillReadTool(db)

    one = await read.execute(name="Watch elevation")
    assert one.success and "skill 'Watch elevation'" in one.message and _WITH_HOLES in one.message

    listing = await read.execute()
    assert listing.success and "- Watch elevation:" in listing.message

    missing = await read.execute(name="nope")
    assert not missing.success and "No skill named 'nope'" in missing.message


# ── The empty registry: honest empty state, no seeds ──────────────────────────

# Pinned literal: the honest empty-registry listing.  Migration 0084 ships the
# skill table EMPTY (no seed library — every skill is distilled from a chat run),
# so this is what a fresh install's skill_read() returns.
_EMPTY_LISTING = (
    "No skills yet — teach one by demonstrating a flow here in chat, and I'll learn it "
    "automatically."
)


@pytest.mark.asyncio
async def test_fresh_migrated_registry_is_empty_and_reads_honestly(tmp_path):
    """A prod-identical DB (create_tables + migrate) has the skill table and ZERO
    rows — no seeds — and skill_read() renders the honest empty state verbatim."""
    db = _migrated_db(tmp_path)
    assert db.skills.list_all() == []
    listing = await SkillReadTool(db).execute()
    assert listing.success
    assert listing.message == _EMPTY_LISTING


# ── Write-retarget at apply (#1629, pure) ──────────────────────────────────────


def test_retarget_writes_binds_the_write_memory_to_the_target():
    """The scoped-write step's ``memory`` constant is overwritten with the target
    collection's name, and the render reflects it — a skill demoed against
    'elevations' renders its write to the collection it's applied to."""
    steps = _elevation_steps()  # step 2 writes memory='elevations' (a constant)
    retargeted = retarget_writes(steps, "cinder-elevation")
    rendered = render_skill(
        retargeted, {"queries": "Cinder Peak", "extract": "the elevation above sea level"}
    )
    assert rendered == (
        "1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
        "2. collection_write(memory='cinder-elevation', "
        "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
    )
    # Pure: the source steps are untouched (a skill is target-agnostic at rest).
    assert steps[1].arguments["memory"] == "elevations"


def test_write_target_is_not_a_hole_and_retarget_owns_it():
    """The scoped-write target arg is NOT parameterized (#1659): the demo's
    memory='knowledge' produces no hole, and applying the skill to a collection
    renders the write to that TARGET — write-retarget owns the target structurally."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": ["Zephyr Ridge elevation"], "extract": "the elevation"},
            result=_BROWSE_OK,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "knowledge",
                "entries": [{"key": "Zephyr Ridge elevation", "content": _EXTRACTED_VALUE}],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, holes = distill_steps(inputs)
    assert "memory" not in {hole.name for hole in holes}  # target is not a parameter
    assert all(sub.path != ["memory"] for sub in steps[1].substitutions)
    retargeted = retarget_writes(steps, "peak-notes")
    rendered = render_skill(retargeted, {"queries": "Cinder Peak", "extract": "the elevation"})
    assert "collection_write(memory='peak-notes'" in rendered


def test_retarget_writes_drops_a_hole_on_the_memory_argument():
    """Defensive: distill never makes the write target a hole (#1659), but retarget
    still turns any stray hole on ``memory`` into the target constant — dropping the
    substitution so the render can't put a hole marker back over the fixed target."""
    step = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "{dest}", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[SkillSubstitution(path=["memory"], kind=SkillSubKind.HOLE, hole="dest")],
    )
    retargeted = retarget_writes([step], "target-b")
    assert retargeted[0].arguments["memory"] == "target-b"
    assert all(sub.path != ["memory"] for sub in retargeted[0].substitutions)
    assert "memory='target-b'" in render_skill(retargeted, {})


def test_retarget_writes_leaves_non_write_steps_untouched():
    """A non-scoped-write step (a browse) is passed through unchanged — only the
    scoped-write tools are retargeted."""
    steps = _elevation_steps()
    retargeted = retarget_writes(steps, "target-b")
    assert retargeted[0].arguments == steps[0].arguments  # the browse step is identical
