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
from penny.database.skill_store import holes_from_json, steps_from_json
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
from penny.tools.skill_tools import _NOTHING_TO_SAVE, SkillCreateTool, SkillReadTool

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


def _log_run(
    db: Database,
    run_id: str,
    utterance: str,
    calls: list[tuple[str, dict, str, bool]],
    *,
    stamp_success: bool = True,
    fused: bool = True,
) -> None:
    """Log one chat run as a single promptlog row: the triggering user turn, the
    batched tool calls (in order → ordinals), and each call's framed result plus its
    STRUCTURAL success stamp (``tool_success``, #1600 — what the framework writes at
    execution time and skill_create's certification reads).

    Each logged call carries the universal ``reasoning`` think-aloud the framework
    injects on every real tool call (#1661), so the fixture matches a real
    promptlog — distill must strip it, not distill it into a hole.

    ``stamp_success=False`` omits the stamp entirely — a run as logged BEFORE #1600,
    used to exercise the honest absent-stamp refusal.  ``fused=False`` logs the user
    turn as the BARE utterance (a real chat row) instead of the fused
    ``<context>---<utterance>`` form — the origin-extraction fallback (#1661)."""
    tool_calls = []
    tool_turns = []
    for index, (name, args, result, success) in enumerate(calls, start=1):
        call_id = f"c{index}"
        logged_args = {**args, "reasoning": _REASONING}
        tool_calls.append(
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(logged_args)}}
        )
        turn = {"role": "tool", "tool_call_id": call_id, "content": result}
        if stamp_success:
            turn[PennyConstants.TOOL_RESULT_SUCCESS_KEY] = success
        tool_turns.append(turn)
    content = f"live context{PennyConstants.SECTION_SEPARATOR}{utterance}" if fused else utterance
    user_turn = {"role": "user", "content": content}
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

# The WHOLE skill_create result — what the user sees of what was learned: the
# lead, the identity/intent/holes lines, and the with-holes recipe.
_CREATE_RESULT_LITERAL = (
    "Learned skill 'Watch elevation'.\n"
    "skill 'Watch elevation'\n"
    f"intent: {_UTTERANCE}\n"
    "holes: queries (required), extract (required)\n"
    "steps:\n"
    f"{_WITH_HOLES}"
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
    # skill_create runs inside its OWN run (run-B); the preceding run (run-A) is the
    # demonstration it snapshots.  run-A is the most-recent run whose id != run-B.
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    result = await tool.execute(name="Watch elevation")

    assert result.success and result.mutated
    assert result.message == _CREATE_RESULT_LITERAL

    stored = db.skills.get("Watch elevation")
    assert stored is not None
    assert stored.source_run_id == "run-A" and stored.author == "chat"
    stored_steps = steps_from_json(stored.steps)
    rendered = render_skill(
        stored_steps,
        {"queries": "Cinder Peak elevation", "extract": "the elevation above sea level"},
    )
    assert rendered == _MONEY_LITERAL

    # #1661: every demonstration call carried the universal ``reasoning`` think-aloud,
    # but distill strips it — no ``reasoning`` hole, no ``reasoning`` key on any stored
    # step, and the render never prints ``reasoning=`` (not in the money literal above,
    # nor the with-holes form the user was shown).
    assert "reasoning" not in {hole.name for hole in holes_from_json(stored.holes)}
    assert all("reasoning" not in step.arguments for step in stored_steps)
    assert "reasoning=" not in rendered
    assert "reasoning" not in result.message


@pytest.mark.asyncio
async def test_skill_description_from_bare_utterance(tmp_path):
    """#1661: a REAL chat row carries the user's message as the BARE utterance (no
    fused ``---`` Live-context prefix).  The origin-extraction fallback uses the
    whole turn, so the skill's description/intent IS that utterance — the prior
    split-only code left the origin empty, degrading it to a generic ``Skill:
    <name>``."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
        fused=False,  # a real chat row: the bare utterance, no fused Live-context
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    result = await tool.execute(name="Bare origin")
    assert result.success
    stored = db.skills.get("Bare origin")
    assert stored is not None
    # The bare utterance became the skill's description/intent — not "Skill: Bare origin".
    assert stored.intent == _UTTERANCE and stored.description == _UTTERANCE
    assert f"intent: {_UTTERANCE}" in result.message


@pytest.mark.asyncio
async def test_skill_create_captures_the_whole_preceding_run(tmp_path):
    """Name-only capture snapshots EVERY non-``done`` step of the preceding run —
    no range selection.  ``done`` consumes an ordinal but is never a skill step, so
    it's excluded; the leading incidental lookup is kept (nothing is trimmed)."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
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
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    result = await tool.execute(name="Whole run")
    assert result.success
    whole = db.skills.get("Whole run")
    assert whole is not None
    steps = steps_from_json(whole.steps)
    # All three non-``done`` steps captured, in order — the read is NOT trimmed; the
    # ``done`` IS excluded.
    assert [s.tool for s in steps] == ["collection_read_latest", "browse", "collection_write"]
    # The write's content still binds to the browse step (step 2, renumbered).
    assert steps[2].substitutions[-1].step == 2


@pytest.mark.asyncio
async def test_skill_create_filters_failed_steps_and_renumbers_bindings(tmp_path):
    """Filter-not-refuse (#1659): failed steps are DROPPED from the recipe (not a
    whole-save refusal), and a binding renumbers against the SURVIVING steps —
    ``source_ordinal`` keeps the original run position, ``ordinal`` is skill-local."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),  # ok → skill step 1 (source 1)
            ("collection_read_latest", {"memory": "notes"}, "read failed", False),  # dropped
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),  # ok → skill step 2 (source 3)
            ("done", {}, "Cycle complete.", True),
        ],
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    result = await tool.execute(name="Filtered")
    assert result.success
    stored = db.skills.get("Filtered")
    assert stored is not None
    steps = steps_from_json(stored.steps)
    # Only the two SUCCEEDED calls survive; the failed read is left out.
    assert [s.tool for s in steps] == ["browse", "collection_write"]
    # source_ordinal keeps the ORIGINAL run position; ordinal is the skill-local number.
    assert [s.source_ordinal for s in steps] == [1, 3]
    assert [s.ordinal for s in steps] == [1, 2]
    # The content binding renumbers against the SURVIVING steps → skill step 1.
    content = steps[1].substitutions[-1]
    assert content.kind == SkillSubKind.BINDING and content.step == 1


@pytest.mark.asyncio
async def test_skill_create_replaces_by_name(tmp_path):
    """No versioning: re-teaching an existing name REPLACES the row and says so.

    Each save snapshots the PRECEDING run, so the two demonstrations are logged in
    sequence: run-A (a lone browse) is saved first, then run-D (browse + write) is
    logged and saved second — the newer run replaces the older skill."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", _UTTERANCE, [("browse", _BROWSE_ARGS, _BROWSE_OK, True)])
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    # run-A is the only run so far (and != run-B) → it's the preceding run.
    first = await tool.execute(name="Watch elevation")
    assert "Learned skill" in first.message

    # Log a newer demonstration; it becomes the most-recent run != run-B.
    _log_run(
        db,
        "run-D",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
    )
    second = await tool.execute(name="Watch elevation")
    assert "Replaced the previous version of 'Watch elevation'." in second.message
    # One row, now the two-step demonstration.
    assert len(db.skills.list_all()) == 1
    replaced = db.skills.get("Watch elevation")
    assert replaced is not None
    assert len(steps_from_json(replaced.steps)) == 2


@pytest.mark.asyncio
async def test_skill_create_refuses_when_nothing_to_save(tmp_path):
    """The name-only refusals — both resolve to the actionable 'nothing to save'
    message: (1) no run precedes the current one (only run-B exists, and the query
    excludes it), and (2) the preceding run had only a ``done`` (no runnable step)."""
    db = _make_db(tmp_path)

    # (1) No preceding run: the only logged run IS the current one (run-B).
    _log_run(db, "run-B", _UTTERANCE, [("browse", _BROWSE_ARGS, _BROWSE_OK, True)])
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")
    no_preceding = await tool.execute(name="X")
    assert not no_preceding.success
    assert no_preceding.message == _NOTHING_TO_SAVE
    assert db.skills.get("X") is None

    # (2) The preceding run (run-A) ran only ``done`` — no runnable step to capture.
    _log_run(db, "run-A", _UTTERANCE, [("done", {}, "Cycle complete.", True)])
    done_only = await tool.execute(name="X")
    assert not done_only.success
    assert done_only.message == _NOTHING_TO_SAVE
    assert db.skills.get("X") is None


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
    await SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B").execute(
        name="Watch elevation"
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
    "No skills yet — teach one by demonstrating a flow, then skill_create(name=<title>)."
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


# ── Certified-by-execution: nothing survives the filter → nothing to save ─────


@pytest.mark.asyncio
async def test_skill_create_refuses_when_no_step_succeeded(tmp_path):
    """Filter-not-refuse's floor (#1659): when every captured call FAILED, the
    routine didn't actually work — nothing survives the filter, so it resolves to
    the same actionable nothing-to-save refusal; nothing is persisted."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_FAILED, False),
            ("collection_write", _WRITE_ARGS, "write failed", False),
        ],
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    result = await tool.execute(name="AllBroken")
    assert not result.success
    assert result.message == _NOTHING_TO_SAVE
    assert db.skills.get("AllBroken") is None


@pytest.mark.asyncio
async def test_skill_create_refuses_a_run_with_no_success_stamps(tmp_path):
    """A run logged BEFORE #1600 carries no per-call success stamp.  The filter reads
    the STRUCTURAL bit, not the framed prose — an absent stamp is uncertain, so no
    step certifies, none survive, and the nothing-to-save refusal fires (visible
    degradation over silent success) even though the framed results read as clean."""
    db = _make_db(tmp_path)
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [
            ("browse", _BROWSE_ARGS, _BROWSE_OK, True),
            ("collection_write", _WRITE_ARGS, _WRITE_OK, True),
        ],
        stamp_success=False,  # a pre-#1600 run: framed results, no structural stamp
    )
    tool = SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat", run_id="run-B")

    result = await tool.execute(name="Legacy")
    assert not result.success
    assert result.message == _NOTHING_TO_SAVE
    assert db.skills.get("Legacy") is None  # nothing persisted from an uncertain run


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
