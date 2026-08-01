"""Skill substrate tests (#1590) — the render, provenance inference, certified-by
-execution, and the seed library, driven through the tool entry points with
deterministic fixtures and fictional content only.
"""

from __future__ import annotations

import json

import pytest

from penny.database import Database
from penny.database.skills import (
    WRITE_TARGET_DESCRIPTION,
    DistillInput,
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    distill_steps,
    render_skill,
    retarget_writes,
    unbound_required_parameters,
)
from penny.tools.skill_tools import SkillReadTool

# ── Fixtures: a fictional "watch the elevation of a peak" demonstration ────────
#
# Structural provenance (#1659): the browse query and the extract instruction are
# non-binding string leaves → HOLES; the extracted reading ("1,842 m") flows
# step 1 → step 2 as a BINDING.  The leaf holding "elevations" additionally carries
# the ATTACHMENT MARK (#1783) because that value names one of Penny's own
# collections — nothing here looks at which tool the leaf sits on.  All fictional.

_UTTERANCE = "Save the Zephyr Ridge elevation to my notes"
_EXTRACTED_VALUE = "1,842 m"

# The registry the demonstration ran against: the collections a routine could be
# ATTACHED to.  ``distill_steps`` compares demonstrated VALUES against this set — it
# never learns which tools exist, let alone which of them write.
_ATTACHMENT_NAMES = frozenset({"elevations", "knowledge", "prices", "fruits", "notes"})

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
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, parameter="queries"),
                SkillSubstitution(path=["extract"], kind=SkillSubKind.HOLE, parameter="extract"),
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments=json.loads(json.dumps(_WRITE_ARGS)),
            substitutions=[
                SkillSubstitution(
                    path=["memory"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description=WRITE_TARGET_DESCRIPTION,
                    attachment=True,
                ),
                SkillSubstitution(
                    path=["entries", 0, "key"], kind=SkillSubKind.HOLE, parameter="queries"
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
    "2. collection_write(memory={the collection this is set up on}, "
    "entries=[{'key': {queries}, 'content': the value from step 1}])"
)

# THE money literal — steps + bound params → the numbered TEXT recipe.  Holes
# substituted verbatim; the binding reads as a legible instruction; the ATTACHMENT-marked
# leaf stays a placeholder because only ATTACHING the routine to a collection decides
# it — ``params`` can never bind it, so an uninstantiated skill never names a
# collection.  The prompt a collection actually runs adds ``retarget_writes`` (below).
_MONEY_LITERAL = (
    "1. browse(queries=['Cinder Peak elevation'], extract='the elevation above sea level')\n"
    "2. collection_write(memory={the collection this is set up on}, "
    "entries=[{'key': 'Cinder Peak elevation', 'content': the value from step 1}])"
)


def test_render_skill_with_holes_is_the_template():
    """An unbound skill renders holes as ``{name}`` placeholders and the binding as
    a legible instruction — the with-holes recipe the read surface shows."""
    assert render_skill(_elevation_steps()) == _WITH_HOLES


def test_render_skill_bound_is_the_money_literal():
    """steps + bound params → the numbered text recipe: every hole substituted with the
    param value verbatim, the binding kept legible, and the attachment-marked leaf still
    the placeholder — a skill hardcodes nothing from its demonstration (#1777)."""
    rendered = render_skill(
        _elevation_steps(),
        {"queries": "Cinder Peak elevation", "extract": "the elevation above sea level"},
    )
    assert rendered == _MONEY_LITERAL
    assert "elevations" not in rendered  # the demonstrated collection is never named


# ── Provenance inference: binding / candidate / attachment mark in one run ─────


def test_distill_classifies_every_leaf_the_same_way_and_marks_by_value():
    """Structural provenance (#1659): a value that flowed from a prior result is a
    BINDING; every other string leaf is a REQUIRED candidate (shared values collapse to
    one).  #1783: EVERY leaf goes through that one process — the collection name is a
    candidate like any other and reaches the labeller — and it additionally carries the
    ATTACHMENT mark because its VALUE names one of Penny's collections, never because
    of the tool it sits on."""
    inputs = [
        DistillInput(source_ordinal=1, tool="browse", arguments=_BROWSE_ARGS, result=_BROWSE_OK),
        DistillInput(
            source_ordinal=2, tool="collection_write", arguments=_WRITE_ARGS, result=_WRITE_OK
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)

    # Three candidates — the browse query, the extract instruction, and the collection
    # name; the write KEY reuses the query's candidate (same value → one shared).  The
    # collection name is NOT exempt from candidacy: the labeller adjudicates it, and
    # only then does it stop being a parameter (see test_skill_extraction.py).
    assert parameters == [
        SkillParameter(name="queries", required=True),
        SkillParameter(name="extract", required=True),
        SkillParameter(name="memory", required=True),
    ]
    # Every candidate is required, so an unbound instantiation refuses naming each one
    # (the refusal carries the whole SkillParameter — name + description); binding them
    # all clears the validation (#1591/#1659, no silent default).
    assert [p.name for p in unbound_required_parameters(parameters, {})] == [
        "queries",
        "extract",
        "memory",
    ]
    assert unbound_required_parameters(parameters, {"queries": "x", "extract": "y"}) == [
        SkillParameter(name="memory", required=True)
    ]

    # Step 1: the query and the extract instruction are both candidates, neither marked
    # (neither value names a collection).
    step1 = {tuple(s.path): s for s in steps[0].substitutions}
    assert step1[("queries", 0)].kind == SkillSubKind.HOLE
    assert step1[("queries", 0)].parameter == "queries"
    assert step1[("queries", 0)].attachment is False
    assert step1[("extract",)].kind == SkillSubKind.HOLE
    assert step1[("extract",)].attachment is False

    # Step 2: the key is the SHARED 'queries' candidate; the content is a BINDING to
    # step 1's result; the collection name is a candidate carrying the mark.
    step2 = {tuple(s.path): s for s in steps[1].substitutions}
    assert step2[("entries", 0, "key")].kind == SkillSubKind.HOLE
    assert step2[("entries", 0, "key")].parameter == "queries"
    assert step2[("entries", 0, "key")].attachment is False
    assert step2[("entries", 0, "content")].kind == SkillSubKind.BINDING
    assert step2[("entries", 0, "content")].step == 1
    assert step2[("memory",)].kind == SkillSubKind.HOLE
    assert step2[("memory",)].parameter == "memory"
    assert step2[("memory",)].attachment is True
    # The demonstrated collection survives as the verbatim ledger copy (provenance),
    # and never as a rendered literal — pre-labeller it is a {memory} hole, and the
    # instantiation seam binds it to the collection the routine is attached to.
    assert steps[1].arguments["memory"] == "elevations"
    assert "elevations" not in render_skill(steps)
    assert "collection_write(memory={memory}" in render_skill(steps)
    assert "collection_write(memory='peak-notes'" in render_skill(
        retarget_writes(steps, "peak-notes")
    )


def test_distill_marks_by_value_across_arbitrary_tools():
    """#1783's load-bearing case, and the one the old rule could not express: a routine
    whose sequence is a ``log_append`` (a write tool that was never in the whitelist),
    a plugin call (a tool nothing has heard of), and a read of the routine's own
    collection.  The mark follows the VALUE — the collection name wherever it appears,
    including on the read — and every other leaf, log name and calendar id alike, is an
    ordinary candidate the labeller adjudicates."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="collection_read_latest",
            arguments={"memory": "notes", "limit": 5},
            result="You read notes: (collection_read_latest result)\n1 entry",
        ),
        DistillInput(
            source_ordinal=2,
            tool="create_event",
            arguments={"calendar_id": "cal-77", "title": "Ridge survey"},
            result="You created an event: (create_event result)\nCreated.",
        ),
        DistillInput(
            source_ordinal=3,
            tool="log_append",
            arguments={"memory": "survey-log", "content": "Ridge survey booked"},
            result="You appended to survey-log: (log_append result)\nAppended.",
        ),
        DistillInput(
            source_ordinal=4,
            tool="collection_write",
            arguments={"memory": "notes", "entries": [{"key": "k", "content": "c"}]},
            result=_WRITE_OK,
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)
    marked = {
        (step.tool, tuple(sub.path))
        for step in steps
        for sub in step.substitutions
        if sub.attachment
    }
    # 'notes' is a collection, so BOTH the read and the write carry the mark; the log
    # name and the calendar id are values the mark knows nothing about.
    assert marked == {("collection_read_latest", ("memory",)), ("collection_write", ("memory",))}
    assert {p.name for p in parameters} >= {"calendar_id", "title", "memory"}
    # Applying the routine to a collection binds every marked leaf and NOTHING else —
    # the log stays the log, the calendar id stays the calendar id.
    retargeted = retarget_writes(steps, "ridge-notes")
    assert retargeted[0].arguments["memory"] == "ridge-notes"
    assert retargeted[1].arguments == {"calendar_id": "cal-77", "title": "Ridge survey"}
    assert retargeted[2].arguments["memory"] == "survey-log"
    assert retargeted[3].arguments["memory"] == "ridge-notes"


def test_a_bound_leaf_is_never_marked_even_when_its_value_names_a_collection():
    """The mark only reaches an UNEXPLAINED leaf.  A value a prior step produced is
    already explained — it came from that step — so there is nothing for an attachment
    to decide, even when the value happens to match a collection's name.  Marking it
    would let the render seam overwrite a real binding with the collection's own name,
    which is a stale literal written into every cycle, forever."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": ["which list holds the readings"]},
            result="You opened the index (browse result)\nelevations",
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={"memory": "notes", "entries": [{"key": "k", "content": "elevations"}]},
            result=_WRITE_OK,
        ),
    ]
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    subs = {tuple(s.path): s for s in steps[1].substitutions}
    bound = subs[("entries", 0, "content")]
    assert bound.kind == SkillSubKind.BINDING and bound.step == 1
    assert bound.attachment is False  # 'elevations' is a collection, but it BOUND
    # Only the write's own unexplained collection leaf is rebound; the binding survives.
    retargeted = retarget_writes(steps, "ridge-notes")
    assert retargeted[1].arguments["memory"] == "ridge-notes"
    assert retargeted[1].arguments["entries"][0]["content"] == "elevations"
    assert "'content': the value from step 1" in render_skill(retargeted)


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
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
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
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    content = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    # 'a' (len 1) is inside 'banana' but too trivial to bind → 'banana' stays a hole.
    assert content.kind == SkillSubKind.HOLE


# ── #1809: a page is not a returned value — a mid-token match never binds ──────
#
# A browse WITHOUT ``extract`` returns the PAGE, so the payload a leaf is tested
# against is a document, and a document contains most short strings by accident.
# The listing's own URL carries the product's slug, so a write key the assistant
# slugs from the same product is a substring of the payload — while the values the
# round genuinely copied out sit against punctuation ('$499.', '"Aurora Deck 2"').

_FAUX_LISTING_URL = "https://faux-market.example/aurora-deck-2"
_FAUX_LISTING_PAGE = (
    "You opened the Aurora Deck 2 listing (browse result)\n"
    f"## browse {_FAUX_LISTING_URL}:\n"
    '"Aurora Deck 2" — the listed price is $499.\n'
    f"Permalink: {_FAUX_LISTING_URL}\n"
    "In stock."
)

# The recipe the round distils to, whichever label the assistant picked: named slots
# for the two keys IT chose, the fetched values for the two contents it copied.
_LISTING_RECIPE = (
    "1. browse(queries=[{queries}])\n"
    "2. collection_write(memory={memory}, entries=["
    "{'key': {key}, 'content': the value from step 1}, "
    "{'key': {key-2}, 'content': the value from step 1}])"
)


@pytest.mark.parametrize("write_key", ["aurora-deck-2", "aurora-deck-2-price"])
def test_a_write_key_inside_the_pages_url_is_not_a_binding(write_key):
    """A key that appears in the browsed page only as a FRAGMENT of its URL never came
    from the browse — it stays a candidate parameter (#1809).

    Plain containment made the classification turn on an accident of the page: the same
    round, the same leaf, two labels — 'aurora-deck-2' is inside the listing's own URL
    and false-bound to the price, while 'aurora-deck-2-price' is not and classified
    correctly.  The consequence was a routine that wrote each cycle's price under a key
    that was also the price, so nothing ever overwrote anything.  A binding means the
    model COPIED the result, so the value must fill the tokens it lands in; a slice out
    of the middle of one ('…example/aurora-deck-2') is a coincidence of characters.

    Both labels now agree — and the two values the round DID copy still bind even
    though each sits against punctuation in the page ('is $499.', '"Aurora Deck 2"'),
    which is what keeps the narrowing from costing real bindings."""
    inputs = [
        DistillInput(
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [_FAUX_LISTING_URL]},
            result=_FAUX_LISTING_PAGE,
        ),
        DistillInput(
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": "prices",
                "entries": [
                    {"key": write_key, "content": "$499"},
                    {"key": "listing-title", "content": "Aurora Deck 2"},
                ],
            },
            result=_WRITE_OK,
        ),
    ]
    steps, parameters = distill_steps(inputs, _ATTACHMENT_NAMES)
    subs = {tuple(s.path): s for s in steps[1].substitutions}

    assert subs[("entries", 0, "key")].kind == SkillSubKind.HOLE
    assert {parameter.name for parameter in parameters} == {"queries", "memory", "key", "key-2"}
    # Both copied values bind despite their punctuation margins — the price to the
    # figure the page states, the title to the quoted heading.
    for path in (("entries", 0, "content"), ("entries", 1, "content")):
        assert subs[path].kind == SkillSubKind.BINDING
        assert subs[path].step == 1
    # The recipe therefore reads the right way round — a named slot for each key, the
    # fetched value for each content — and never names the demonstrated label.
    assert render_skill(steps) == _LISTING_RECIPE


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
    steps, holes = distill_steps(inputs, _ATTACHMENT_NAMES)
    # The same candidates as the reasoning-free run above — the think-aloud added none.
    assert holes == [
        SkillParameter(name="queries", required=True),
        SkillParameter(name="extract", required=True),
        SkillParameter(name="memory", required=True),
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
    steps, _ = distill_steps(inputs, _ATTACHMENT_NAMES)
    args = steps[0].arguments
    assert "reasoning" not in args  # the top-level think-aloud is gone
    assert args["entries"][0]["reasoning"] == "nested — kept"  # the nested key is untouched


# ── skill_read: render one / list all ─────────────────────────────────────────


def _seed_skill(db: Database, name: str) -> None:
    """Persist a skill directly (the extractor's post-labeller output shape) so the read
    surface can be tested independently of the extraction path."""
    db.skills.upsert(
        SkillDraft(
            name=name,
            intent=_UTTERANCE,
            description=_UTTERANCE,
            steps=_elevation_steps(),
            parameters=[
                SkillParameter(name="queries", required=True),
                SkillParameter(name="extract", required=True),
            ],
            source_run_id="run-A",
        ),
        author="chat",
        description_embedding=None,
    )


@pytest.mark.bare_db
@pytest.mark.asyncio
async def test_skill_read_renders_one_and_lists_all(db):
    """``skill_read(name)`` renders one full recipe; bare ``skill_read()`` lists
    every skill; an unknown name is an actionable miss."""
    _seed_skill(db, "Watch elevation")
    read = SkillReadTool(db)

    one = await read.execute(name="Watch elevation")
    # render_skill_full indents the recipe block two spaces under `steps:` (#1668).
    indented_recipe = "\n".join(f"  {line}" for line in _WITH_HOLES.splitlines())
    assert one.success and "skill 'Watch elevation'" in one.message
    assert indented_recipe in one.message

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
async def test_fresh_migrated_registry_is_empty_and_reads_honestly(db):
    """A prod-identical DB (create_tables + migrate) has the skill table and ZERO
    rows — no seeds — and skill_read() renders the honest empty state verbatim."""
    assert db.skills.list_all() == []
    listing = await SkillReadTool(db).execute()
    assert listing.success
    assert listing.message == _EMPTY_LISTING


# ── Attachment binding at apply (#1629/#1783, pure) ────────────────────────────


def test_retarget_writes_binds_the_write_memory_to_the_target():
    """The attachment-marked ``memory`` placeholder is BOUND to the target
    collection's name, and the render reflects it — a skill demoed against
    'elevations' renders its write to the collection it's applied to."""
    steps = _elevation_steps()  # step 2's memory leaf is the #1777 placeholder
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
    assert steps[1].substitutions[0].kind == SkillSubKind.PLACEHOLDER


def test_retarget_writes_binds_a_migrated_legacy_write_target():
    """A skill taught before #1783 carries no mark of its own — migration 0103 sets one
    on the leaf 0101 gave a placeholder, and from then on it binds like any other marked
    leaf.  So an instantiated collection's program acts on its own collection whatever
    shape the stored skill was taught in."""
    migrated = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "elevations", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[
            SkillSubstitution(
                path=["memory"],
                kind=SkillSubKind.PLACEHOLDER,
                description=WRITE_TARGET_DESCRIPTION,
                attachment=True,
            )
        ],
    )
    retargeted = retarget_writes([migrated], "peak-notes")
    assert render_skill(retargeted) == (
        "1. collection_write(memory='peak-notes', entries=[{'key': 'k', 'content': 'c'}])"
    )


def test_retarget_writes_binds_only_what_carries_the_mark():
    """The deliberate consequence of binding by mark (#1783): a leaf with NO mark is not
    bound, whatever tool it belongs to.  An unmigrated pre-#1783 row would therefore
    keep its demonstrated literal — which is exactly why migration 0103 exists, and why
    nothing in this path may fall back to a tool whitelist."""
    unmarked = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "elevations", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[],
    )
    assert retarget_writes([unmarked], "peak-notes") == [unmarked]


def test_a_user_named_destination_stays_a_parameter():
    """The labeller judged the USER to have named the destination, so the leaf keeps its
    parameter and carries no mark — the attachment must not overwrite a choice the user
    made.  Two such destinations therefore stay distinct, with no mechanism added for
    it: ``params`` binds each one separately (#1783)."""
    step = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "reading-list", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[
            SkillSubstitution(path=["memory"], kind=SkillSubKind.HOLE, parameter="destination")
        ],
    )
    assert retarget_writes([step], "somewhere-else") == [step]
    assert render_skill([step], {"destination": "weekend-reading"}) == (
        "1. collection_write(memory='weekend-reading', entries=[{'key': 'k', 'content': 'c'}])"
    )


def test_retarget_writes_binds_a_marked_hole_and_drops_its_substitution():
    """A marked leaf that is still a HOLE (the labeller left no verdict on it, so the
    render seam is what fills it) is bound to the target and its substitution dropped —
    else the render would put a marker back over the collection's own name."""
    step = SkillStep(
        ordinal=1,
        source_ordinal=1,
        tool="collection_write",
        arguments={"memory": "elevations", "entries": [{"key": "k", "content": "c"}]},
        substitutions=[
            SkillSubstitution(
                path=["memory"], kind=SkillSubKind.HOLE, parameter="memory", attachment=True
            )
        ],
    )
    retargeted = retarget_writes([step], "target-b")
    assert retargeted[0].arguments["memory"] == "target-b"
    assert all(sub.path != ["memory"] for sub in retargeted[0].substitutions)
    assert "memory='target-b'" in render_skill(retargeted, {})


def test_retarget_writes_leaves_unmarked_steps_untouched():
    """A step with no marked leaf (a browse) is passed through unchanged — binding
    reads the mark, never the tool."""
    steps = _elevation_steps()
    retargeted = retarget_writes(steps, "target-b")
    assert retargeted[0].arguments == steps[0].arguments  # the browse step is identical
