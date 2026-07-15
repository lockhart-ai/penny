"""Skill substrate tests (#1590) — the render, provenance inference, certified-by
-execution, and the seed library, driven through the tool entry points with
deterministic fixtures and fictional content only.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.migrate import migrate
from penny.database.skill_store import steps_from_json
from penny.database.skills import (
    DistillInput,
    SkillHole,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    distill_steps,
    render_skill,
    retarget_writes,
    unbound_required_holes,
)
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.skill_tools import SkillCreateTool, SkillReadTool

# ── Fixtures: a fictional "watch the elevation of a peak" demonstration ────────
#
# The one utterance phrase the model reused ("Zephyr Ridge elevation") is a HOLE;
# the extracted reading ("1,842 m") flows step 1 → step 2 as a BINDING; the fixed
# instruction and target collection are CONSTANTS.  All fictional.

_UTTERANCE = "Save the Zephyr Ridge elevation to my notes"
_EXTRACTED_VALUE = "1,842 m"

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


def _log_run(
    db: Database,
    run_id: str,
    utterance: str,
    calls: list[tuple[str, dict, str, bool]],
    *,
    stamp_success: bool = True,
) -> None:
    """Log one chat run as a single promptlog row: the triggering user turn, the
    batched tool calls (in order → ordinals), and each call's framed result plus its
    STRUCTURAL success stamp (``tool_success``, #1600 — what the framework writes at
    execution time and skill_create's certification reads).

    ``stamp_success=False`` omits the stamp entirely — a run as logged BEFORE #1600,
    used to exercise the honest absent-stamp refusal."""
    tool_calls = []
    tool_turns = []
    for index, (name, args, result, success) in enumerate(calls, start=1):
        call_id = f"c{index}"
        tool_calls.append(
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}
        )
        turn = {"role": "tool", "tool_call_id": call_id, "content": result}
        if stamp_success:
            turn[PennyConstants.TOOL_RESULT_SUCCESS_KEY] = success
        tool_turns.append(turn)
    user_turn = {
        "role": "user",
        "content": f"live context{PennyConstants.SECTION_SEPARATOR}{utterance}",
    }
    db.messages.log_prompt(
        model="m",
        messages=[user_turn, *tool_turns],
        response={"choices": [{"message": {"tool_calls": tool_calls}}]},
        run_id=run_id,
        agent_name=PennyConstants.CHAT_AGENT_NAME,
    )


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
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, hole="queries")
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
    "1. browse(queries=[{queries}], extract='the elevation above sea level')\n"
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

# The WHOLE skill_create result — what the user sees of what was learned: the
# lead, the identity/intent/holes lines, and the with-holes recipe.
_CREATE_RESULT_LITERAL = (
    "Learned skill 'Watch elevation'.\n"
    "skill 'Watch elevation'\n"
    f"intent: {_UTTERANCE}\n"
    "holes: queries (required)\n"
    "steps:\n"
    f"{_WITH_HOLES}"
)


def test_render_skill_with_holes_is_the_template():
    """An unbound skill renders holes as ``{name}`` placeholders and the binding as
    a legible instruction — the with-holes recipe the read surface shows."""
    assert render_skill(_elevation_steps()) == _WITH_HOLES


def test_render_skill_bound_is_the_money_literal():
    """steps + bound params → the numbered text prompt a collection will run: holes
    substituted with the param value verbatim, the binding kept legible."""
    rendered = render_skill(_elevation_steps(), {"queries": "Cinder Peak elevation"})
    assert rendered == _MONEY_LITERAL


# ── Provenance inference: hole / binding / constant in one run ─────────────────


def test_distill_classifies_all_three_provenance_classes():
    """The fixture exercises hole (utterance), binding (prior result), and constant
    (neither) in one run — the inference is deterministic and tested against it."""
    inputs = [
        DistillInput(source_ordinal=1, tool="browse", arguments=_BROWSE_ARGS, result=_BROWSE_OK),
        DistillInput(
            source_ordinal=2, tool="collection_write", arguments=_WRITE_ARGS, result=_WRITE_OK
        ),
    ]
    steps, holes = distill_steps(inputs, _UTTERANCE)

    # One parameter, deduped across the two places the utterance value appears.
    assert holes == [SkillHole(name="queries", required=True)]
    # The #1591 instantiation rule lives with the holes: an unbound required hole
    # is named; a bound one isn't.
    assert unbound_required_holes(holes, {}) == ["queries"]
    assert unbound_required_holes(holes, {"queries": "Cinder Peak elevation"}) == []

    # Step 1: the query is a HOLE (verbatim in the utterance); the extract
    # instruction is a CONSTANT (never seen, so no substitution).
    step1 = {tuple(s.path): s for s in steps[0].substitutions}
    assert step1[("queries", 0)].kind == SkillSubKind.HOLE
    assert step1[("queries", 0)].hole == "queries"
    assert ("extract",) not in step1  # constant → baked in, not substituted

    # Step 2: the key is the same HOLE; the content is a BINDING to step 1's result;
    # the target collection is a CONSTANT.
    step2 = {tuple(s.path): s for s in steps[1].substitutions}
    assert step2[("entries", 0, "key")].kind == SkillSubKind.HOLE
    assert step2[("entries", 0, "content")].kind == SkillSubKind.BINDING
    assert step2[("entries", 0, "content")].step == 1
    assert ("memory",) not in step2  # constant → baked in


# ── skill_create: end-to-end through the tool ─────────────────────────────────


@pytest.mark.asyncio
async def test_skill_create_end_to_end_renders_the_money_literal(tmp_path):
    """A clean demonstration → skill_create → the stored skill renders (with the
    same param) to the money literal, and the result echoes the learned skill."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
            ("done", {}, "Cycle complete.", True),
        ],
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat")

    result = await tool.execute(name="Watch elevation", from_run="run-A", steps="1-2")

    assert result.success and result.mutated
    assert result.message == _CREATE_RESULT_LITERAL

    stored = db.skills.get("Watch elevation")
    assert stored is not None
    assert stored.source_run_id == "run-A" and stored.author == "chat"
    rendered = render_skill(steps_from_json(stored.steps), {"queries": "Cinder Peak elevation"})
    assert rendered == _MONEY_LITERAL


@pytest.mark.asyncio
async def test_skill_create_excludes_done_and_honours_the_range(tmp_path):
    """``done`` consumes an ordinal but is never a skill step, and a range that
    trims a leading incidental lookup keeps only the selected calls."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-B",
        _UTTERANCE,
        [
            (
                "collection_read_latest",
                {"memory": "elevations"},
                "You looked up your elevations:",
                True,
            ),
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
            ("done", {}, "Cycle complete.", True),
        ],
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat")

    # Trim the leading read (step 1); keep the browse+write (steps 2-3).
    result = await tool.execute(name="Trimmed", from_run="run-B", steps="2-3")
    assert result.success
    trimmed = db.skills.get("Trimmed")
    assert trimmed is not None
    steps = steps_from_json(trimmed.steps)
    assert [s.tool for s in steps] == ["browse", "collection_write"]
    # The binding still points at the skill's own step 1 (renumbered).
    assert steps[1].substitutions[-1].step == 1


@pytest.mark.asyncio
async def test_skill_create_rejects_an_uncertified_step(tmp_path):
    """Certified-by-execution: a selected step that FAILED in the source run is
    rejected with an error naming the failed step — enforced, not documented."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-C",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_FAILED, False),  # total browse failure
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat")

    result = await tool.execute(name="Broken", from_run="run-C", steps="1-2")
    assert not result.success
    assert "step 1 (browse) didn't succeed" in result.message
    assert db.skills.get("Broken") is None  # nothing persisted


@pytest.mark.asyncio
async def test_skill_create_replaces_by_name(tmp_path):
    """No versioning: re-teaching an existing name REPLACES the row and says so."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", _UTTERANCE, [("browse", _BROWSE_ARGS, _BROWSE_OK, True)])
    _log_run(
        db,
        "run-D",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat")

    first = await tool.execute(name="Watch elevation", from_run="run-A", steps="1")
    assert "Learned skill" in first.message

    second = await tool.execute(name="Watch elevation", from_run="run-D", steps="1-2")
    assert "Replaced the previous version of 'Watch elevation'." in second.message
    # One row, now the two-step demonstration.
    assert len(db.skills.list_all()) == 1
    replaced = db.skills.get("Watch elevation")
    assert replaced is not None
    assert len(steps_from_json(replaced.steps)) == 2


@pytest.mark.asyncio
async def test_skill_create_actionable_on_bad_input(tmp_path):
    """A malformed range and an unknown run get actionable, guess-free errors."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", _UTTERANCE, [("browse", _BROWSE_ARGS, _BROWSE_OK, True)])
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat")

    bad_range = await tool.execute(name="X", from_run="run-A", steps="oops")
    assert not bad_range.success and "steps='oops'" in bad_range.message

    unknown = await tool.execute(name="X", from_run="nope", steps="1")
    assert not unknown.success and "No run found with id 'nope'" in unknown.message


# ── skill_read: render one / list all ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_read_renders_one_and_lists_all(tmp_path):
    """``skill_read(name)`` renders one full recipe; bare ``skill_read()`` lists
    every skill; an unknown name is an actionable miss."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
    )
    await SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat").execute(
        name="Watch elevation", from_run="run-A", steps="1-2"
    )
    read = SkillReadTool(db)

    one = await read.execute(name="Watch elevation")
    assert one.success and "skill 'Watch elevation'" in one.message and _WITH_HOLES in one.message

    listing = await read.execute()
    assert listing.success and "- Watch elevation:" in listing.message

    missing = await read.execute(name="nope")
    assert not missing.success and "No skill named 'nope'" in missing.message


# ── The empty registry: honest empty state, no seeds ──────────────────────────

# Pinned literal: the honest empty-registry listing.  Migration 0084 ships the
# skill table EMPTY (no seed library — every skill enters through skill_create),
# so this is what a fresh install's skill_read() returns.
_EMPTY_LISTING = (
    "No skills yet — teach one by demonstrating a flow, then "
    "skill_create(name=<title>, from_run=<run id>, steps=<range>)."
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


# ── Certified-by-execution: absent stamp is an HONEST refusal (#1600) ──────────


@pytest.mark.asyncio
async def test_skill_create_refuses_a_run_with_no_success_stamps(tmp_path):
    """A run logged BEFORE #1600 carries no per-call success stamp.  The
    certification reads the STRUCTURAL bit, not the framed prose — so an absent
    stamp is uncertain, and refuse-to-certify-uncertain beats optimistic-pass
    (visible degradation over silent success): every step refuses, nothing is
    persisted, even though the framed results read like clean successes."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-legacy",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
        stamp_success=False,  # a pre-#1600 run: framed results, no structural stamp
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat")

    result = await tool.execute(name="Legacy", from_run="run-legacy", steps="1-2")
    assert not result.success
    assert "step 1 (browse) didn't succeed" in result.message
    assert db.skills.get("Legacy") is None  # nothing persisted from an uncertain run


# ── Write-retarget at apply (#1629, pure) ──────────────────────────────────────


def test_retarget_writes_binds_the_write_memory_to_the_target():
    """The scoped-write step's ``memory`` constant is overwritten with the target
    collection's name, and the render reflects it — a skill demoed against
    'elevations' renders its write to the collection it's applied to."""
    steps = _elevation_steps()  # step 2 writes memory='elevations' (a constant)
    retargeted = retarget_writes(steps, "cinder-elevation")
    rendered = render_skill(retargeted, {"queries": "Cinder Peak"})
    assert rendered == (
        "1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
        "2. collection_write(memory='cinder-elevation', "
        "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
    )
    # Pure: the source steps are untouched (a skill is target-agnostic at rest).
    assert steps[1].arguments["memory"] == "elevations"


def test_retarget_writes_drops_a_hole_on_the_memory_argument():
    """A write whose ``memory`` was itself a hole is turned into the target constant —
    the substitution addressing that leaf is dropped so the render can't put the hole
    marker back over the fixed target."""
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
