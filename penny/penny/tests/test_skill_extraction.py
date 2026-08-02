"""Automatic skill extraction at chat-run end (#1658/#1665/#1668/#1770).

Drives ``SkillExtractor.extract`` over REAL-SHAPED logged runs — every tool call
carries the framework's top-level ``reasoning`` think-aloud, and the user turn is a
bare utterance (no fused ``---`` Live-context prefix), the #1661 shape.  The matrix:
read+write qualifies (correct holes/bindings, reasoning stripped) · pure-read /
pure-write / failed-write-only / bail-nudged / no-calls excluded (each naming its
gate) · failed-step filtering · name slugging · dedup by name and by shape+meaning.
The #1665 additions: orientation verbs (``find`` etc.) are dropped from the recipe
and don't count as the qualifying read (find+write → pure write) · a wrapped write
value binds against a prior result's PAYLOAD (the frame stripped) while a topic-name
key still doesn't · the skill is named GENERICALLY by a micro-context (tagged
NAME:/DESCRIPTION:), falling back to the deterministic slug on any failure · the
run-end narration frame renders the generic name plus the demonstrated-on instance.
The #1770 additions: the labeller ADJUDICATES each candidate parameter — a PARAM
verdict keeps it a required parameter, a PLACEHOLDER verdict drops it and renders
what belongs there instead of freezing the demonstrated value, and NO verdict (or a
malformed one) keeps the arg-derived required parameter · the naming system prompt
as a whole-render literal.  The #1783 additions: the leaf that named a collection is
adjudicated like every other one — the assistant's choice is filled by whatever the
routine is attached to, the user's choice stays a parameter they bind, and two
user-named destinations therefore stay two destinations — as a stored SKILL; a
collector cycle is scoped to ONE collection, so what running such a routine on a
cadence should do is a separate, open question.  Either way the demonstrated
collection survives only as the step's verbatim ledger arguments and never reaches the
rendered recipe.  All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.models import Skill
from penny.database.skill_store import (
    parameters_from_json,
    parameters_to_json,
    steps_from_json,
    steps_to_json,
)
from penny.database.skills import (
    WRITE_TARGET_DESCRIPTION,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    render_skill,
    retarget_writes,
    unbound_required_parameters,
)
from penny.llm.models import LlmMessage, LlmResponse
from penny.prompts import Prompt
from penny.skill_extraction import (
    ExtractionGate,
    NoExtraction,
    SkillExtracted,
    SkillExtractor,
)
from penny.tests.eval.test_state_transitions import learn_to_apply_fixture_skill
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tests.schema_template import migrated_db
from penny.tools.memory_tools import collector_tool_surface
from penny.tools.micro_context import (
    SKILL_NAMING_SYSTEM_PROMPT,
    SKILL_SHAPE_SYSTEM_PROMPT,
    MicroContext,
)
from penny.tools.skill_tools import render_skill_brief, render_skill_full

# ── Real-shaped fixtures: a fictional "watch the aurora deck 2 price" demo ──────

_UTTERANCE = "read the aurora deck 2 listing, find the current price, and remember it"
_PRICE = "$499"
_BROWSE_ARGS = {"queries": ["aurora deck 2 price"], "extract": "the current price"}
_WRITE_ARGS = {
    "memory": "aurora-prices",
    "entries": [{"key": "aurora deck 2 price", "content": _PRICE}],
}
_BROWSE_OK = f"You used `browse` and here's the result: (browse result)\nEXTRACTED: {_PRICE}"
_WRITE_OK = "You saved an entry to aurora-prices: (collection_write result)\nWrote 1 entry."
_READ_OK = "You looked up your notes: (collection_read_latest result)\n(empty)"

_BROWSE = ("browse", _BROWSE_ARGS, _BROWSE_OK, True)
_WRITE = ("collection_write", _WRITE_ARGS, _WRITE_OK, True)


pytestmark = pytest.mark.bare_db


# The collection every demonstration in this module writes into.
_DEMO_COLLECTION = "aurora-prices"


@pytest.fixture(autouse=True)
def _demonstrated_collection(request):
    """The collection the demonstrations write into EXISTS while they run — a write
    that succeeded means its collection was there, and on the chat surface the write
    itself creates it.

    Load-bearing since #1783: distillation marks a leaf whose demonstrated VALUE names
    one of Penny's collections, so a fixture that logged a write against a collection
    the registry never had would exercise a run that cannot happen and would miss the
    mark entirely.  Only for tests that take the ``db`` fixture."""
    if "db" not in request.fixturenames:
        return
    db = request.getfixturevalue("db")
    if db.memories.get(_DEMO_COLLECTION) is None:
        db.memories.create_collection(_DEMO_COLLECTION, "price notes")


def _extractor(
    db: Database,
    mock: MockLlmClient | None = None,
    *,
    model: MockLlmClient | None = None,
) -> SkillExtractor:
    """Build a ``SkillExtractor`` (#1665): ``mock`` is the EMBEDDING client (dedup
    tests set its embed handler); ``model`` is the TEXT client for the naming
    micro-context — a bare mock returns untagged text, so naming falls back to the
    deterministic slug (which keeps the pre-#1665 name/description assertions holding)."""
    client = cast(Any, model or MockLlmClient())
    return SkillExtractor(
        db,
        cast(Any, mock or MockLlmClient()),
        client,
        agent_name="chat",
        # The REAL collector-runnable surface (#1668) — so the lifecycle-filter test
        # exercises the actual masked surface, not a hand-copied set.
        collector_tool_surface=collector_tool_surface(db, client),
    )


def _log_run(
    db: Database,
    run_id: str,
    utterance: str,
    calls: list[tuple[str, dict, str, bool]],
    *,
    stamp_success: bool = True,
    nudges: list[str] | None = None,
) -> None:
    """Log one chat run REAL-SHAPED: the bare utterance turn (no fused Live-context),
    each tool call carrying the universal top-level ``reasoning`` think-aloud (#1661),
    and each call's framed result plus its structural ``tool_success`` stamp (#1600).

    ``nudges`` injects extra user turns (the text-bail nudge markers) so the health
    gate can be exercised; ``stamp_success=False`` omits the stamp (a pre-#1600 run)."""
    tool_calls = []
    tool_turns = []
    for index, (name, args, result, success) in enumerate(calls, start=1):
        call_id = f"c{index}"
        real_args = {**args, "reasoning": f"step {index}: doing {name}"}
        tool_calls.append(
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(real_args)}}
        )
        turn: dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "content": result}
        if stamp_success:
            turn[PennyConstants.TOOL_RESULT_SUCCESS_KEY] = success
        tool_turns.append(turn)
    messages: list[dict] = [{"role": "user", "content": utterance}]
    messages.extend({"role": "user", "content": nudge} for nudge in nudges or [])
    messages.extend(tool_turns)
    db.messages.log_prompt(
        model="m",
        messages=messages,
        response={"choices": [{"message": {"tool_calls": tool_calls}}]},
        run_id=run_id,
        agent_name=PennyConstants.CHAT_AGENT_NAME,
    )


# ── Qualifies: read + write → a skill with the right holes/bindings ────────────


@pytest.mark.asyncio
async def test_read_write_run_qualifies_and_distils_correctly(db):
    """A browse (read) + collection_write (act) run is a routine: it qualifies and a
    skill is extracted with the query/extract as required holes, the write content
    bound to the browse result, and the leaf naming the collection an ATTACHMENT-marked
    placeholder — with a bare model there is no draw to describe it, so it falls back to
    the fixed string (#1777's constant, kept as exactly that fallback by #1783), and the
    demonstrated collection is never rendered.
    The description is the run's bare utterance; the framework ``reasoning`` is gone."""
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted) and not result.replaced
    skill = result.skill
    assert skill.description == _UTTERANCE and skill.intent == _UTTERANCE
    assert skill.author == "chat" and skill.source_run_id == "run-A"
    # Holes: the browse query and the extract instruction; the write KEY reuses the
    # query's hole (same value → one shared parameter).  The write CONTENT is a
    # binding (it flowed from the browse), so it is NOT a hole.
    assert [hole.name for hole in parameters_from_json(skill.parameters)] == ["queries", "extract"]
    steps = steps_from_json(skill.steps)
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    subs = {tuple(s.path): s for s in steps[1].substitutions}
    content_sub = subs[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    # The leaf that named a collection: attachment-marked, and — with no draw to
    # describe it — a placeholder on the fallback string rather than a required
    # parameter nobody could bind.  The demonstrated collection stays in ``arguments``
    # as the ledger copy and never reaches the recipe.
    assert subs[("memory",)].kind == SkillSubKind.PLACEHOLDER
    assert subs[("memory",)].attachment is True
    assert subs[("memory",)].description == WRITE_TARGET_DESCRIPTION
    assert steps[1].arguments["memory"] == _DEMO_COLLECTION
    assert _DEMO_COLLECTION not in render_skill_full(skill)
    # The framework reasoning think-aloud is stripped from every stored step.
    assert all("reasoning" not in step.arguments for step in steps)


@pytest.mark.asyncio
async def test_learning_a_skill_attaches_nothing(db):
    """Extraction LEARNS; it never instantiates (#1706).  The framework used to
    attach the just-learned skill to the collection the demonstrated round
    created, folding teach and instantiate into one turn.  The state machine
    makes them two turns — learn reports what it did and offers, apply binds it
    when the user says so — so the collection is left exactly as the round left
    it: no skill, no rendered prompt, nothing scheduled."""
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    row = db.memories.get("aurora-prices")
    assert row is not None
    assert row.skill_name is None, "learning must not bind the skill to a collection"
    assert row.extraction_prompt is None, "and must not render a program into it"
    assert row.collector_interval_seconds is None, "and must not schedule anything"


# ── Excluded: pure read, pure write, failed-write-only, bail, no-calls ─────────


@pytest.mark.asyncio
async def test_pure_read_run_is_excluded(db):
    """A run that only READ (answering a question) is not a routine → PURE_READ, no
    skill."""
    _log_run(db, "run-A", "what does the aurora deck 2 cost?", [_BROWSE])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_READ)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_pure_write_run_is_excluded(db):
    """A run that only WROTE ('remember this' — the storage atom) is a plain write,
    not a job → PURE_WRITE, no skill."""
    _log_run(db, "run-A", "remember the aurora deck 2 is $499", [_WRITE])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_WRITE)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_failed_write_only_run_is_excluded(db):
    """A run whose only write FAILED does not qualify: the failed call is filtered,
    leaving a pure read → PURE_READ, no skill (visible degradation, not a half-baked
    skill)."""
    failed_write = ("collection_write", _WRITE_ARGS, "write failed", False)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, failed_write])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_READ)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_bail_nudged_run_is_excluded(db):
    """A run poisoned by a text-bail nudge (the model failed to route a call through
    the tool channel) is unhealthy → BAILED, no skill — even though it read+wrote."""
    _log_run(
        db,
        "run-A",
        _UTTERANCE,
        [_BROWSE, _WRITE],
        nudges=[Prompt.CHAT_CALL_AS_TEXT_NUDGE],
    )

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.BAILED)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_run_with_no_tool_calls_is_excluded(db):
    """A pure-conversation turn (no tool calls at all) yields NO_TOOL_CALLS."""
    _log_run(db, "run-A", "hey how's it going", [])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.NO_TOOL_CALLS)


@pytest.mark.asyncio
async def test_run_with_no_certified_steps_is_excluded(db):
    """When a run had calls but NONE succeeded (or a pre-#1600 run has no stamps),
    nothing certifies → NO_CERTIFIED_STEPS, no skill (never an empty skill)."""
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE], stamp_success=False)

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.NO_CERTIFIED_STEPS)


# ── Failed-step filtering: the surviving routine is extracted ──────────────────


@pytest.mark.asyncio
async def test_failed_step_is_filtered_from_the_routine(db):
    """A failed exploratory read is DROPPED (#1659 filter-not-refuse); the surviving
    browse + write still qualify and the extracted skill omits the failed call."""
    failed_read = ("collection_read_latest", {"memory": "notes"}, "read failed", False)
    _log_run(db, "run-A", _UTTERANCE, [failed_read, _BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    # source_ordinal keeps the ORIGINAL run position (the dropped read was ordinal 1).
    assert [step.source_ordinal for step in steps] == [2, 3]


# ── Deterministic naming ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_name_is_a_slug_of_the_utterance_with_urls_stripped(db):
    """The deterministic-slug FALLBACK (#1665): with a bare model (no NAME:/DESCRIPTION:
    naming draw), the name falls back to a slug of the triggering message — the URL is
    removed, lowercased, non-alphanumeric collapsed to hyphens, capped at 6 words — and
    the full message stays the description."""
    utterance = (
        "Read the Aurora Deck 2 listing at https://faux-market.test/aurora-deck-2, "
        "find the current price, and remember it."
    )
    _log_run(db, "run-A", utterance, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"
    assert result.skill.description == utterance  # the full message, untruncated


# ── Dedup: REPLACE by name, and by shape + meaning keeping the existing name ────


@pytest.mark.asyncio
async def test_reteaching_the_same_utterance_replaces_by_name(db):
    """Re-demonstrating a routine whose message slugs to an existing skill name
    REPLACES that skill in place (one row, the newer steps)."""
    extractor = _extractor(db)

    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])
    first = await extractor.extract("run-A")
    assert isinstance(first, SkillExtracted) and not first.replaced

    # A second demonstration of the SAME routine (same utterance → same slug name).
    _log_run(db, "run-B", _UTTERANCE, [_BROWSE, _WRITE])
    second = await extractor.extract("run-B")

    assert isinstance(second, SkillExtracted) and second.replaced
    assert second.skill.name == first.skill.name
    assert second.skill.source_run_id == "run-B"  # the newer demonstration
    assert len(db.skills.list_all()) == 1


@pytest.mark.asyncio
async def test_same_shape_and_meaning_replaces_keeping_existing_name(db):
    """A re-demonstration with a DIFFERENT wording (so a different slug) but the SAME
    tool sequence AND a description embedding within the house dedup threshold
    REPLACES the existing skill, keeping ITS name — the clean/flaky demo collapse."""
    mock = MockLlmClient()

    # Both descriptions embed to the same vector (the aurora topic), so their cosine
    # is 1.0 ≥ MEMORY_DEDUP_CONTENT_SIM_STRICT — a same-meaning match.
    def embed_handler(_model: str, texts: str | list[str]) -> list[list[float]]:
        items = texts if isinstance(texts, list) else [texts]
        return [([1.0, 0.0, 0.0] if "aurora" in t else [0.0, 1.0, 0.0]) for t in items]

    mock.set_embed_handler(embed_handler)
    extractor = _extractor(db, mock)

    _log_run(db, "run-A", "watch the aurora deck 2 price", [_BROWSE, _WRITE])
    first = await extractor.extract("run-A")
    assert isinstance(first, SkillExtracted)
    original_name = first.skill.name

    # Different wording (a different slug), same tool shape, same aurora meaning.
    _log_run(db, "run-B", "keep an eye on the aurora deck 2 price for me", [_BROWSE, _WRITE])
    second = await extractor.extract("run-B")

    assert isinstance(second, SkillExtracted) and second.replaced
    assert second.skill.name == original_name  # kept the existing skill's name
    assert len(db.skills.list_all()) == 1


@pytest.mark.asyncio
async def test_different_meaning_inserts_a_new_skill(db):
    """A same-shape run whose meaning differs (embedding below threshold) is a NEW
    skill, never a false-replace — two skills coexist."""
    mock = MockLlmClient()

    def embed_handler(_model: str, texts: str | list[str]) -> list[list[float]]:
        items = texts if isinstance(texts, list) else [texts]
        return [([1.0, 0.0, 0.0] if "aurora" in t else [0.0, 1.0, 0.0]) for t in items]

    mock.set_embed_handler(embed_handler)
    extractor = _extractor(db, mock)

    _log_run(db, "run-A", "watch the aurora deck 2 price", [_BROWSE, _WRITE])
    await extractor.extract("run-A")
    _log_run(db, "run-B", "watch the harbor weather report", [_BROWSE, _WRITE])
    second = await extractor.extract("run-B")

    assert isinstance(second, SkillExtracted) and not second.replaced
    assert len(db.skills.list_all()) == 2


# ── Non-chat run is excluded ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_collector_run_is_not_extracted(db):
    """A run whose prompts are NOT the chat agent's (a background collector cycle)
    never yields a skill → NOT_CHAT, so extraction is chat-only by construction."""
    db.messages.log_prompt(
        model="m",
        messages=[{"role": "user", "content": ""}],
        response={"choices": [{"message": {"tool_calls": []}}]},
        run_id="run-A",
        agent_name="thoughts",
    )

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.NOT_CHAT)


@pytest.mark.asyncio
async def test_fresh_migrated_registry_stays_empty_without_a_qualifying_run(tmp_path):
    """A prod-identical DB (create_tables + migrate) ships the skill table EMPTY; a
    non-qualifying turn leaves it empty (no seeds, no accidental extraction)."""
    db = migrated_db(str(tmp_path / "seeded.db"))
    _log_run(db, "run-A", "hi there", [])
    result = await _extractor(db).extract("run-A")
    assert isinstance(result, NoExtraction)
    assert db.skills.list_all() == []


# ── #1665 fixtures: orientation, the real framed browse result, a wrapped write ─

_FIND_RESULT = "You searched your memory: (find result)\nNo matching skill — nothing saved yet."
# The REAL browse-with-extract frame shape: a narration line carrying the
# ``(browse result)`` machine tag, then the payload alone (success renders no
# fetch-handle tail — the "saved" phrasing read as the remembering being done).
_REAL_BROWSE_FRAME = "You opened the Aurora Deck 2 listing (browse result)\n$499"
# A write whose content WRAPS the browse payload ('$499') and whose key is a topic
# name (a real parameter that must NOT false-bind to the payload).
_WRAP_WRITE_ARGS = {
    "memory": "aurora-prices",
    "entries": [
        {"key": "aurora deck 2 price", "content": "Current price of Aurora Deck 2 is $499."}
    ],
}
# A one-step recipe for the narration-frame whole-render literal.
_WATCH_STEPS = steps_to_json(
    [
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments={"queries": ["https://shop.test/widget"]},
            substitutions=[
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, parameter="url")
            ],
        )
    ]
)
_WATCH_PARAMS = parameters_to_json([SkillParameter(name="url", required=True)])

# #1770: the motivating write — the round recorded the price AND a second entry it
# composed ITSELF about the page it had just read.  Neither leaf of that second
# entry came from the user, so neither can be a required parameter.
_INVENTED_LABEL = "Page source for Aurora Deck 2"
_INVENTED_WRITE_ARGS = {
    "memory": "aurora-prices",
    "entries": [
        {"key": "aurora deck 2 price", "content": _PRICE},
        {"key": "aurora deck 2 source", "content": _INVENTED_LABEL},
    ],
}
_INVENTED_WRITE = ("collection_write", _INVENTED_WRITE_ARGS, _WRITE_OK, True)


def _naming_model(content: str) -> MockLlmClient:
    """A text model client whose every chat returns ``content`` (the naming draw)."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda _request, _count: LlmResponse(message=LlmMessage(role="assistant", content=content))
    )
    return model


def _two_draw_model(naming: str, shape: str) -> MockLlmClient:
    """A model client answering the two run-end draws SEPARATELY (#1803), dispatched
    on the system prompt each carries — the labeller's provenance verdicts, then the
    shape draw's name + what the routine is about.  Distinct answers are the point:
    the two draws are two questions, and a test that fed both the same text could not
    show which one decided anything."""
    model = MockLlmClient()

    def respond(request: dict, _count: int) -> LlmResponse:
        system = request["messages"][0]["content"]
        drawn = shape if system == SKILL_SHAPE_SYSTEM_PROMPT else naming
        return LlmResponse(message=LlmMessage(role="assistant", content=drawn))

    model.set_response_handler(respond)
    return model


# ── #1665: orientation verbs excluded from steps AND the qualifying read ───────


@pytest.mark.asyncio
async def test_orientation_find_step_is_dropped_from_the_recipe(db):
    """A run that ORIENTS (find) then reads + writes is a routine, but the find call
    is registry-navigation, not routine: it is dropped from the distilled steps, and a
    find result echoing the query never manufactures a false binding (#1665)."""
    find = ("find", {"query": "watch a listing price"}, _FIND_RESULT, True)
    _log_run(db, "run-A", _UTTERANCE, [find, _BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    # find is gone; only the content read + the write survive, in run order.
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    assert [step.source_ordinal for step in steps] == [2, 3]


@pytest.mark.asyncio
async def test_find_plus_write_only_is_a_pure_write_not_a_skill(db):
    """A find + write run has NO content read once orientation is excluded — a find
    does not count as the qualifying read — so it is a pure write (the storage atom),
    not a skill (#1665)."""
    find = ("find", {"query": "aurora prices"}, _FIND_RESULT, True)
    _log_run(db, "run-A", _UTTERANCE, [find, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_WRITE)
    assert db.skills.list_all() == []


# ── #1665: binding compares against the result PAYLOAD, not the frame ──────────


@pytest.mark.asyncio
async def test_wrapped_write_value_binds_against_the_result_payload(db):
    """A write value that WRAPS the browse output ('Current price … is $499.') binds
    to the browse step — the comparison strips the tool-result FRAME to its payload
    ('$499'), so the wraps direction fires (#1665/#1661 item 3).  The 'content' leaf
    is a binding, never a nonsense required parameter; a topic-name KEY still doesn't
    bind (it stays a real parameter)."""
    browse = ("browse", {"queries": ["aurora deck 2 listing"]}, _REAL_BROWSE_FRAME, True)
    write = ("collection_write", _WRAP_WRITE_ARGS, _WRITE_OK, True)
    _log_run(db, "run-A", _UTTERANCE, [browse, write])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    content_sub = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    hole_names = [hole.name for hole in parameters_from_json(result.skill.parameters)]
    assert "content" not in hole_names  # the wrapped value bound; it is NOT a parameter
    assert "key" in hole_names  # the topic-name key did NOT false-bind — it's a parameter


# ── #1665: generic naming via the micro-context, slug fallback ────────────────


@pytest.mark.asyncio
async def test_tagged_naming_micro_context_sets_a_generic_name_and_description(db):
    """A qualifying run's skill is named GENERICALLY by the naming micro-context: a
    tagged NAME:/DESCRIPTION: draw becomes the skill's slugged name + generic
    description (which the description_embedding anchors), NOT the instance
    utterance (#1665).  The demonstrated-on instance rides back for the frame."""
    model = _naming_model(
        "NAME: Watch a listing price\nDESCRIPTION: Look up a price on a listing page and record it."
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    # The instigating ask precedes the demonstration in the conversation — the
    # naming step must SEE it (#1658 intent grounding: the description carries the
    # WHY, so a later re-statement of intent maps to the skill).
    db.messages.log_message(
        direction="incoming",
        sender="user",
        content="can you keep an eye on the zephyr lamp listing for me?",
    )

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "watch-a-listing-price"  # the generic NAME, slugged
    assert result.skill.description == "Look up a price on a listing page and record it."
    assert result.skill.intent == "Look up a price on a listing page and record it."
    assert result.origin_message == _UTTERANCE
    # The naming content leads with the conversation, oldest first — and the
    # DEMONSTRATING message is in it, attributed to the user (#1770).  Presented
    # under its own unattributed heading, the labeller read the conversation as
    # the only record of what the user said, did not find the demonstrated values
    # there, and ruled them assistant-produced — correct reasoning over a
    # presentation that hid the speaker.  Both turns must render as `user:`.
    naming_request = model.requests[-1]
    naming_content = " ".join(m.get("content", "") for m in naming_request["messages"])
    assert "Conversation that led to the construction of this routine" in naming_content
    assert "user: can you keep an eye on the zephyr lamp listing for me?" in naming_content
    assert f"user: {_UTTERANCE}" in naming_content, "the demonstrating turn is the user's"
    assert "First demonstrated by this message" not in naming_content


@pytest.mark.asyncio
async def test_untagged_naming_falls_back_to_the_deterministic_slug(db):
    """When the naming micro-context never produces both tags, extraction does NOT
    block: it falls back to the deterministic slug of the triggering message + that
    message as the description (#1665)."""
    model = _naming_model("I think this is a price-watching routine of some kind.")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # the fallback slug
    assert result.skill.description == _UTTERANCE


# ── #1668: semantic parameter names + descriptions ────────────────────────────


@pytest.mark.asyncio
async def test_tagged_param_labels_become_semantic_names_and_descriptions(db):
    """The naming micro-context relabels each parameter (#1668): tagged PARAM lines
    (keyed by the CURRENT arg-derived name) become the skill's SEMANTIC parameter
    names + descriptions, they render in the parameters block AND as ``{name}``
    placeholders in the steps, and binding is by the semantic name (display form ==
    invocation form) — the binding key at instantiation."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PARAM queries: url — the page to look at\n"
        "PARAM extract: what_to_find — what to pull from it"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description) for p in params] == [
        ("url", "the page to look at"),
        ("what_to_find", "what to pull from it"),
    ]
    # A PARAM verdict says the USER supplied that value (#1770), so it stays a real,
    # REQUIRED parameter — the model binds it per instantiation.
    assert all(p.required for p in params)
    # The full render shows the semantic parameters block AND {semantic} placeholders.
    rendered = render_skill_full(result.skill)
    assert "  - url (required): the page to look at" in rendered
    assert "  - what_to_find (required): what to pull from it" in rendered
    assert "browse(queries=[{url}], extract={what_to_find})" in rendered
    # Binding is by the semantic name — the params binding key at instantiation.
    assert [p.name for p in unbound_required_parameters(params, {})] == ["url", "what_to_find"]
    assert unbound_required_parameters(params, {"url": "u", "what_to_find": "w"}) == []


@pytest.mark.asyncio
async def test_param_labelling_falls_back_per_parameter(db):
    """Per-parameter fallback, not all-or-nothing (#1668/#1770): a candidate the model
    labels gets its semantic name + description; one it omits — or one whose verdict
    line is malformed — keeps its arg-derived name, carries no description, and stays
    a REQUIRED parameter.  Absence is deliberately not a verdict: overloading it to
    mean "drop" would let one flaky draw silently delete a real parameter."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAM queries: url — the page to look at"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description) for p in params] == [
        ("url", "the page to look at"),  # labelled
        ("extract", None),  # unlabelled → arg-derived name, no description
    ]
    assert all(p.required for p in params)
    assert "extract={extract}" in render_skill(steps_from_json(result.skill.steps))

    # A PLACEHOLDER line with no description says nothing about what belongs there,
    # so it is dropped by the parse — the same no-verdict path, NOT a drop.
    malformed = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAM queries: url — the page to look at\n"
        "PLACEHOLDER extract:"
    )
    _log_run(db, "run-B", "check the aurora price again please", [_BROWSE, _WRITE])

    later = await _extractor(db, model=malformed).extract("run-B")

    assert isinstance(later, SkillExtracted)
    kept = parameters_from_json(later.skill.parameters)
    assert [(p.name, p.description, p.required) for p in kept] == [
        ("url", "the page to look at", True),
        ("extract", None, True),
    ]


@pytest.mark.asyncio
async def test_contradictory_verdict_lines_leave_the_parameter_alone(db):
    """A draw that names one candidate TWICE contradicts itself (#1770 — the contract
    asks for exactly one line each), so that candidate gets NO verdict and keeps its
    arg-derived required parameter.  Letting the last line win would let a stray
    trailing PLACEHOLDER delete a real parameter, which is exactly what
    absence-is-never-a-drop exists to prevent."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAM queries: url — the page to look at\n"
        "PARAM extract: what_to_find — what to pull from it\n"
        "PLACEHOLDER extract: something the assistant worked out"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in params] == [
        ("url", "the page to look at", True),
        ("extract", None, True),  # contradicted → no verdict → the default survives
    ]


@pytest.mark.asyncio
async def test_semantic_names_are_hardened_slugged_and_deduped(db):
    """Deterministic hardening of returned names (#1668, load-bearing — the name is
    the binding key): 'Page URL' slugs to 'page_url' (lowercase, spaces→underscores),
    and two parameters that slug to the SAME name are disambiguated with a numeric
    suffix so a binding key can never collide."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAM queries: Page URL — the page\n"
        "PARAM extract: Page URL — the field"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [p.name for p in params] == ["page_url", "page_url_2"]
    # The rename maps through every leaf site — the render substitutes by the slugged name.
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "queries=[{page_url}]" in rendered
    assert "{page_url_2}" in rendered


# ── #1814: the output shape is declared, so tolerance and strictness are one rule ─


@pytest.mark.asyncio
async def test_a_cosmetically_variant_verdict_line_still_carries_its_verdict(db):
    """Everything the DECLARED shape now tolerates, in one draw (#1814).

    The motivating failure: a verdict line arrived with an EN-dash where the parser
    partitioned on an em-dash, so the split found nothing and the entire remainder
    became the parameter's semantic name — a 60-character "name" carrying its own
    description, persisted as the skill's binding key, with no reroll because the
    parse had "succeeded".  Any whitespace-delimited dash variant now separates the
    name from its description (a hyphen INSIDE a name has no spaces around it, so it
    is never mistaken for the separator).  A line may also arrive decorated with a
    list marker or bold, a payload may arrive quoted, and a PARAM line may carry no
    description at all — none of that is the model getting the contract wrong."""
    model = _naming_model(
        "**NAME:** Watch a listing price\n"
        'DESCRIPTION: "Look up a price and record it."\n'
        "- PARAM queries: url – the page to look at\n"
        "* **PARAM extract: what_to_find**"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "watch-a-listing-price"
    assert result.skill.description == "Look up a price and record it."
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in params] == [
        ("url", "the page to look at", True),  # the EN-dash split name from description
        ("what_to_find", None, True),  # no description offered — still a real parameter
    ]


@pytest.mark.asyncio
async def test_a_name_that_swallowed_its_description_is_no_verdict_at_all(db):
    """The backstop for whatever tolerance doesn't reach (#1814): a semantic name is
    a TOKEN, not a sentence, so a line whose "name" swallowed its own description is
    MALFORMED — and a malformed line is no verdict, never good data.  The candidate
    keeps its arg-derived required parameter, which is the flaky-draw-safe direction
    and the whole point: the name is the skill's binding key, and a key nobody could
    ever bind is worse than no verdict at all.  A near-miss tag (``PARAMETRIC``) is
    not a verdict line either, so the write target keeps its unadjudicated fallback."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAM queries: url — the page to look at\n"
        'PARAM extract: what_to_find | the content descriptor to look for (e.g., "price")\n'
        "PARAMETRIC memory: not a verdict line"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in params] == [
        ("url", "the page to look at", True),
        ("extract", None, True),  # malformed line → no verdict → the default survives
    ]
    steps = steps_from_json(result.skill.steps)
    target = {tuple(s.path): s for s in steps[1].substitutions}[("memory",)]
    assert target.kind == SkillSubKind.PLACEHOLDER
    assert target.description == WRITE_TARGET_DESCRIPTION  # PARAMETRIC adjudicated nothing


@pytest.mark.asyncio
async def test_a_draw_missing_a_required_line_is_rerolled_then_falls_back(db):
    """The other half of the declaration (#1814): the REQUIRED lines stay strict.  A
    draw carrying per-candidate verdicts but no ``NAME:`` never parses, so it is
    rerolled once on the unchanged context and then falls back to the deterministic
    slug — the per-item lines being best-effort does NOT make the draw as a whole
    best-effort."""
    model = _naming_model(
        "DESCRIPTION: Look up a price and record it.\nPARAM queries: url — the page to look at"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # the fallback slug
    assert len(model.requests) == 2  # the draw + exactly one reroll, then the fallback
    # No verdict survived the failed draw, so every candidate keeps its arg-derived name.
    assert [p.name for p in parameters_from_json(result.skill.parameters)] == ["queries", "extract"]


# ── #1770: only USER-PROVIDED values are parameters; the rest are placeholders ─


@pytest.mark.asyncio
async def test_placeholder_verdict_drops_the_parameter_and_never_freezes_its_value(db):
    """The motivating case (#1770): the round recorded the price AND a second entry it
    composed itself about the page.  The distiller's 'everything else is a parameter'
    default would make BOTH of that entry's leaves required parameters no user could
    supply, so the labeller adjudicates: a PLACEHOLDER verdict drops the candidate
    from the parameter list and renders the labeller's description in its place.

    Freezing the demonstrated value is the specific failure this prevents — a
    collector re-running the skill would write that stale phrase into the collection
    every cycle, forever — so the attached prompt is asserted WHOLE: the user's two
    values are bound, the assistant's two are placeholders, and neither demonstrated
    phrase appears anywhere.

    The draw here is GROUPED BY VERDICT, the shape the contract asks for (#1807), and
    the same draw INTERLEAVED lands identically: grouping is what the prompt asks of the
    model, never what the parse requires of it, so a draw that ignores the grouping is
    still mapped candidate for candidate rather than partly dropped."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PARAM queries: url — the listing page to read\n"
        "PARAM extract: what_to_find — what to pull from the page\n"
        "PLACEHOLDER key: a short label for the second entry\n"
        "PLACEHOLDER content: a note about the page you just read"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _INVENTED_WRITE])
    extractor = _extractor(db, model=model)

    result = await extractor.extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    # Only the two values the user supplied survive as parameters.
    assert [(p.name, p.description, p.required) for p in params] == [
        ("url", "the listing page to read", True),
        ("what_to_find", "what to pull from the page", True),
    ]
    assert unbound_required_parameters(params, {"url": "u", "what_to_find": "w"}) == []

    # The whole-skill render (what `skill_read` returns — the one surface the recipe
    # is the answer on, #1804): only the real parameters are listed, and every leaf no
    # user can bind — each assistant-produced value AND the write target (#1777) —
    # shows WHAT BELONGS THERE in placeholder syntax, never the demonstrated value.
    assert render_skill_full(result.skill) == (
        "skill 'watch-a-listing-price'\n"
        "what it's for: Look up a price on a listing page and record it.\n"
        "parameters:\n"
        "  - url (required): the listing page to read\n"
        "  - what_to_find (required): what to pull from the page\n"
        "steps:\n"
        "  1. browse(queries=[{url}], extract={what_to_find})\n"
        "  2. collection_write(memory={the collection this is set up on}, entries=["
        "{'key': {url}, 'content': the value from step 1}, "
        "{'key': {a short label for the second entry}, "
        "'content': {a note about the page you just read}}])"
    )

    # The BRIEF render (#1804) — the same skill as the ambient section and the
    # narration frame see it: what it is and what it needs, each parameter named with
    # what to supply for it, and no step, no tool name, no placeholder syntax.
    assert render_skill_brief(result.skill) == (
        "watch-a-listing-price — Look up a price on a listing page and record it. "
        "(needs: url — the listing page to read; what_to_find — what to pull from "
        "the page)"
    )

    # The collection the round wrote into is left exactly as the round left it:
    # learning does not instantiate (#1706 removed the run-end auto-attach), so
    # there is no rendered program for a placeholder to be frozen into.
    row = db.memories.get("aurora-prices")
    assert row is not None
    assert row.skill_name is None and row.extraction_prompt is None

    # The SAME verdicts drawn out of group order (#1807) — the parse reads lines, not
    # blocks, so the ungrouped draw produces the identical skill.
    interleaved = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PLACEHOLDER key: a short label for the second entry\n"
        "PARAM queries: url — the listing page to read\n"
        "PLACEHOLDER content: a note about the page you just read\n"
        "PARAM extract: what_to_find — what to pull from the page"
    )
    _log_run(db, "run-B", _UTTERANCE, [_BROWSE, _INVENTED_WRITE])

    ungrouped = await _extractor(db, model=interleaved).extract("run-B")

    assert isinstance(ungrouped, SkillExtracted)
    assert render_skill_full(ungrouped.skill) == render_skill_full(result.skill)


# ── #1803: what the routine is ABOUT is baked; what it is pointed at is asked ──

# The labelling draw is the same in every case below — both values came from the
# user, which is the whole difficulty: provenance cannot separate them.
_BOTH_USER_SUPPLIED = (
    "NAME: Record a listing price\n"
    "DESCRIPTION: Read a listing and store what it costs.\n"
    "PARAM queries: url — the listing page to check\n"
    "PARAM extract: what_to_find — what to pull off the page"
)


@pytest.mark.asyncio
async def test_the_value_the_routine_is_named_for_becomes_a_constant(db):
    """A skill must not name itself for a value it then asks the user to supply
    (#1803).  Real extractions did: `record-product-price` declaring a required
    `what_to_extract` whose own description offered "price" as the example — so the
    routine could not fire from the natural second ask ("watch the price at <url>"),
    routing to `request` for a value its own name already gave.

    Provenance cannot fix this, and the labeller is right not to try: the user
    supplied BOTH values, and nothing in the demonstrated round says which of them
    will vary next time.  So a second draw decides what the routine IS — and because
    it writes the name and the constants together, the two cannot contradict."""
    model = _two_draw_model(
        _BOTH_USER_SUPPLIED,
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Keep an eye on what a listing costs.\n"
        "CONSTANT what_to_find\n"
        "PARAMETER url",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    # The name and description are the SHAPE draw's — the same decision as the
    # constants, which is what stops a price watcher from asking what to watch.
    assert result.skill.name == "watch-a-listing-price"
    assert result.skill.description == "Keep an eye on what a listing costs."
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description) for p in params] == [("url", "the listing page to check")]
    # And it FIRES from a page alone — the failure that motivated the whole ticket.
    assert unbound_required_parameters(params, {"url": "https://example.test"}) == []

    # The constant renders as its value, not as a blank to fill: a leaf no
    # substitution covers is exactly a baked value, so no new render path exists.
    steps = steps_from_json(result.skill.steps)
    assert steps[0].arguments["extract"] == "the current price"
    assert not [sub for sub in steps[0].substitutions if sub.path == ["extract"]], (
        "a constant leaf carries no substitution"
    )
    rendered = render_skill(steps, {"url": "https://example.test"})
    assert "the current price" in rendered
    assert "{the current price}" not in rendered and "{what_to_find}" not in rendered


@pytest.mark.asyncio
async def test_a_routine_is_never_all_constant_so_it_stays_bindable(db):
    """The over-correction guard, and the floor the shape prompt states (#1803): a
    draw that bakes EVERY value is refused, because a routine with nothing left to
    ask for can only ever repeat the one thing it was demonstrated with.

    That dead end has been reached from the other side already — two runs produced
    `watch-a-price` with every leaf placeholdered, so not even the page could be
    re-bound.  Baking everything arrives at the same place by a new route, so the
    floor is structural: the draw is a contract violation, and the degraded state is
    the honest one — the labeller's name, no constants, every value bindable."""
    model = _two_draw_model(
        _BOTH_USER_SUPPLIED,
        "NAME: Watch the aurora deck 2 price\n"
        "DESCRIPTION: Check that one listing and record its price.\n"
        "CONSTANT what_to_find\n"
        "CONSTANT url",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [p.name for p in params] == ["url", "what_to_find"]
    assert unbound_required_parameters(params, {"url": "u", "what_to_find": "w"}) == []
    # Refused, so the shape draw decided NOTHING — including the name.
    assert result.skill.name == "record-a-listing-price"


@pytest.mark.asyncio
async def test_a_value_named_on_both_lines_stays_a_parameter(db):
    """A draw that names the SAME value on a CONSTANT line AND a PARAMETER line has
    contradicted itself, and the BINDABLE direction wins (#1803) — the same rule the
    labeller's repeated-line drop encodes, for the same reason: a routine that lost a
    parameter to a stray line can never be pointed anywhere new, while one that kept a
    needless parameter merely asks a question it could have answered itself.

    The draw is otherwise VALID — both required lines are there and something is left
    bindable — so the shape's NAME still stands.  That is what makes this a
    contradiction rule rather than a refusal: only the contradicted value is dropped."""
    model = _two_draw_model(
        _BOTH_USER_SUPPLIED,
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Keep an eye on what a listing costs.\n"
        "CONSTANT what_to_find\n"
        "PARAMETER what_to_find\n"
        "PARAMETER url",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [p.name for p in params] == ["url", "what_to_find"], "the contradicted value is kept"
    steps = steps_from_json(result.skill.steps)
    assert [sub.parameter for sub in steps[0].substitutions if sub.path == ["extract"]] == [
        "what_to_find"
    ], "the contradicted leaf is still a bindable hole, not a baked constant"
    # The draw itself was valid, so it named the routine — only the one line was dropped.
    assert result.skill.name == "watch-a-listing-price"


@pytest.mark.asyncio
async def test_an_empty_offered_set_is_not_read_as_everything_being_constant():
    """The bindable floor is about what the DRAW did, so an empty offered set is not a
    draw that baked everything (#1803).  Without that short-circuit the emptiness would
    satisfy "the constants cover every value" and every draw would be refused — a
    contract violation manufactured from having nothing to decide."""
    drawn = "NAME: Watch a listing price\nDESCRIPTION: Keep an eye on what a listing costs.\n"

    shape = await MicroContext(cast(Any, _naming_model(drawn))).shape_skill("content", [])

    assert shape is not None, "an empty offered set is not a refusal"
    assert shape.fixed == frozenset()
    assert shape.name == "Watch a listing price"


@pytest.mark.asyncio
async def test_the_attachment_target_is_never_offered_as_a_constant(db):
    """Where the routine WRITES is decided by what it is applied to, not by what it
    is about (#1783), so an attachment-marked leaf is withheld from the shape draw
    entirely — there is no way to bake it, whatever the draw says.

    Baking it would make a rendered program name the collection the routine was
    demonstrated on rather than the one it runs against, which is the single thing
    the retarget seam exists to prevent.  Keyed to the MARK, not to any tool name:
    a skill is an arbitrary sequence, and a plugin's write is marked the same way."""
    model = _two_draw_model(
        f"{_BOTH_USER_SUPPLIED}\nPARAM memory: destination — the collection to write into",
        # The draw names the destination anyway — and it changes nothing.
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Keep an eye on what a listing costs.\n"
        "CONSTANT destination\n"
        "PARAMETER url\n"
        "PARAMETER what_to_find",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    memory_leaf = [sub for sub in steps[1].substitutions if sub.path == ["memory"]]
    assert [(sub.kind, sub.parameter) for sub in memory_leaf] == [
        (SkillSubKind.HOLE, "destination")
    ], "the destination stays a bindable leaf, never baked"
    assert _DEMO_COLLECTION not in render_skill(steps, {})


# ── #1783: the collection name is adjudicated like every other leaf ────────────


@pytest.mark.asyncio
async def test_assistant_chosen_destination_is_filled_by_the_attachment(db):
    """A PLACEHOLDER verdict on the leaf that named a collection says the assistant
    picked where to put the results.  Nobody can bind that, so the attachment fills it:
    the leaf keeps its mark, renders the LABELLER's wording (not a hardcoded string) in
    the stored recipe, and applying the routine to a collection binds it to that
    collection's own name."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PARAM queries: url — the listing page to read\n"
        "PARAM extract: what_to_find — what to pull from the page\n"
        "PLACEHOLDER memory: wherever this routine keeps its readings"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [p.name for p in params] == ["url", "what_to_find"]
    steps = steps_from_json(result.skill.steps)
    target = {tuple(s.path): s for s in steps[1].substitutions}[("memory",)]
    assert target.kind == SkillSubKind.PLACEHOLDER and target.attachment is True
    # The wording is the labeller's, not the fixed fallback.
    assert target.description == "wherever this routine keeps its readings"
    assert "memory={wherever this routine keeps its readings}" in render_skill(steps)
    assert "memory='aurora-watch'" in render_skill(retarget_writes(steps, "aurora-watch"))


@pytest.mark.asyncio
async def test_two_user_named_destinations_stay_two_parameters(db):
    """The corrected model's load-bearing claim (#1783): a routine writing to two
    places the USER named is TWO parameters, bound separately at instantiation — no new
    mechanism, no privileged argument, nothing user-facing invented.  It falls out of
    the verdicts: the user chose both, so neither is the attachment's to fill.  (Two
    places the ASSISTANT chose are two placeholders and both land on the attached
    collection — also an outcome of the verdicts, not a rule.)"""
    db.memories.create_collection("aurora-archive", "older readings")
    second_write = (
        "collection_write",
        {"memory": "aurora-archive", "entries": [{"key": "aurora deck 2 price", "content": "old"}]},
        "You saved an entry to aurora-archive: (collection_write result)\nWrote 1 entry.",
        True,
    )
    model = _naming_model(
        "NAME: Record a listing price in two places\n"
        "DESCRIPTION: Look up a price and record it in a live list and an archive.\n"
        "PARAM queries: url — the listing page to read\n"
        "PARAM extract: what_to_find — what to pull from the page\n"
        "PARAM memory: live_list — the collection to keep the current reading in\n"
        "PARAM memory-2: archive — the collection to keep past readings in\n"
        "PARAM content: previous_reading — the value to file in the archive"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE, second_write])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert {p.name for p in parameters_from_json(result.skill.parameters)} >= {
        "live_list",
        "archive",
    }
    steps = steps_from_json(result.skill.steps)
    targets = [{tuple(s.path): s for s in step.substitutions}[("memory",)] for step in steps[1:]]
    # The user chose both, so neither carries the mark and the attachment rebinds
    # neither — they render as their own distinct parameters.
    assert [t.parameter for t in targets] == ["live_list", "archive"]
    assert all(t.attachment is False for t in targets)
    assert retarget_writes(steps, "somewhere-else") == steps
    rendered = render_skill(steps, {"live_list": "current-prices", "archive": "price-history"})
    assert "collection_write(memory='current-prices'" in rendered
    assert "collection_write(memory='price-history'" in rendered


def test_shape_system_prompt_whole_render():
    """Whole-render literal of the shape contract (#1803): the framing and its two
    inputs, the three numbered asks — the CORE USER INTENT first, then the name
    written from it, then the per-value ABOUT-vs-POINTED-AT decision as two named
    cases — the coherence rule tying them together, the floor that keeps a routine
    bindable, and the enumerated output shape with one line required per value.

    The worked example is deliberately a FILING routine, not the price watcher these
    tests demonstrate: an example drawn from the case in hand teaches pattern-matching
    on that case, and a skill is an arbitrary tool sequence a plugin may have supplied
    the verbs for.  It is also told as unquoted PROSE: a quoted example value is copied
    verbatim, and the one thing this prompt asks the model to compose is a name."""
    assert SKILL_SHAPE_SYSTEM_PROMPT == (
        "You are deciding what a reusable routine IS. You are given what the user "
        "asked for once, a one-line summary of what was done for them, and the values "
        "used to carry it out that time. Do three things:\n"
        "1. From what the user asked for, extract the CORE USER INTENT — what they "
        "were trying to get done when they asked. Their own words are the evidence, "
        "and the one-line summary of what the round did is a second reading of the "
        "same thing: the values are HOW it was carried out, not what it was FOR.\n"
        "2. Name and describe the ROUTINE by that intent: a short verb-noun name for "
        "the KIND of task, generic — never the specific instance — and one line that "
        "states the intent it serves before any mechanics. A description that falls "
        "back on a specified piece of information where the intent named something "
        "particular has dropped the intent — say what the intent actually was.\n"
        "3. Now picture the user coming back later to set this routine running "
        "again, on a new occasion. What is the MINIMAL information they would have "
        "to give you? Decide every value on that one question:\n"
        "   - PARAMETER. They would have to say it again. The routine works the "
        "same way whatever it is, so it cannot be known until they say — it is "
        "asked for every time the routine is set up.\n"
        "   - CONSTANT. They would NOT have to say it again, because the routine "
        "already IS that. Saying it would be repeating what the name has already "
        "promised. It stays fixed, and asking for it would be asking the user to "
        "tell you what they came to you for.\n"
        "   That the user supplied a value the first time settles nothing here: "
        "they supplied all of them while showing you what to do. The question is "
        "only what they would still have to supply once you already know how.\n"
        "   THE USER'S OWN WORDS DECIDE THIS, not the name you happened to pick. A "
        "thing they NAMED as the point of the task is something they would expect "
        "you to know by now, so it is a CONSTANT — and if the name you wrote in "
        "step 2 leaves it open, the name is what is wrong, not this answer.\n"
        "   For example, after being asked to file the receipts from a particular "
        "sender into a tax folder: they named receipts as the point, so a routine "
        "that files receipts needs only the sender and the folder next time. Had "
        "they asked instead to file whatever they point you at, they named nothing, "
        "and every value would be a PARAMETER. Both are real routines — their ask "
        "is what tells you which one you were taught.\n"
        "   At least one value is always a PARAMETER: a routine that needs nothing "
        "said to it can only ever repeat the one occasion it was shown, which makes "
        "it a record of what happened rather than a routine. And a routine with NO "
        "constant is one that has to be told everything it was already told — if "
        "their ask named what to do, say so.\n"
        "Respond with these tagged lines and nothing else:\n"
        "NAME: <a short generic verb-noun name>\n"
        "DESCRIPTION: <one line: what the routine is for>\n"
        "CONSTANT <value name>   (the routine is about it)\n"
        "PARAMETER <value name>   (the routine is pointed at it)\n"
        "Write ONE line for EVERY value, repeating its name exactly so it maps "
        "back — the name ALONE, with nothing after it.\n"
        "Write nothing else — no preamble, no explanation, no restating the routine."
    )


def test_naming_system_prompt_whole_render():
    """Whole-render literal of the labelling contract (#1665/#1668/#1770/#1807): the
    framing and its inputs, the three numbered asks — intent, then the generic routine
    name, then the per-candidate WHERE-DID-THIS-COME-FROM verdict as two named cases —
    and the enumerated output shape, which asks for the candidates GROUPED BY VERDICT
    (#1807: the verdict was reached correctly and transcribed wrongly, always inside a
    run of the other tag, so the split is what gets written rather than a tag prefixed
    to each drafted line)."""
    assert SKILL_NAMING_SYSTEM_PROMPT == (
        "You are a naming step. You are given the conversation that led to the "
        "construction of a reusable routine, the routine itself — a numbered list of "
        "tool calls with fill-in-the-blank {parameters} — the message that first "
        "demonstrated it, and the routine's candidate parameters (each currently named "
        "after the tool argument it fills, and shown with the value it was demonstrated "
        "with). Do three things:\n"
        "1. From the conversation, extract the CORE USER INTENT — what the user was "
        "trying to get done when they asked for this (e.g. keeping an eye on a "
        "listing's price). The routine exists to serve that intent.\n"
        "2. Name and describe the ROUTINE by that intent: a short verb-noun name for "
        "the KIND of task (e.g. 'watch a listing price for changes'), generic — never "
        "the specific instance — and never mechanics alone ('fetch and store data' "
        "says nothing about when to reach for it).\n"
        "3. Decide, for EVERY candidate parameter, where its demonstrated value came "
        "from. There are two cases:\n"
        "   - THE USER GAVE IT. It came from the user — a page they named, a thing "
        "they asked to be found, a label they chose, a place they said to keep it — "
        "including when the assistant "
        "reworded it ('the current price' for their \"find the price\"). This is a real "
        "parameter: name it by what the value MEANS to the user (e.g. 'url', "
        "'what_to_find', 'label'), NOT the tool argument it happens to fill, and "
        "describe in one line what to supply for it.\n"
        "   - THE ASSISTANT PRODUCED IT. The assistant worked it out from what a step "
        "returned, or wrote it itself while carrying the task out — a summary, a note, "
        "a caption about a page, a place it picked itself to keep the results in. The "
        "user never said it and could not supply it, so it "
        "is NOT a parameter: it is a placeholder, and you describe in one line what "
        "belongs in that spot each time the routine runs.\n"
        "   A parameter filling browse's extract argument is a PLAIN-LANGUAGE "
        "instruction naming what to pull from the page (e.g. 'the current price') — "
        "there is no CSS-selector, XPath, or pattern machinery in this system, so never "
        "name or describe one that way.\n"
        "Respond in exactly this shape and nothing else:\n"
        "NAME: <a short generic verb-noun name>\n"
        "DESCRIPTION: <one line: the user intent it serves, then the mechanics>\n"
        "then the group of candidates THE USER GAVE, one line each:\n"
        "PARAM <current name>: <semantic_name> — <one-line description>\n"
        "then the group of candidates THE ASSISTANT PRODUCED, one line each:\n"
        "PLACEHOLDER <current name>: <one-line description of what belongs there>\n"
        "Sort every candidate into one of those two groups BEFORE you write either "
        "group, then write the groups in that order — every candidate appears in "
        "exactly one group, never both and never neither, repeating its CURRENT name "
        "exactly so it maps back. A group with no candidates in it is left out "
        "entirely. Use a single lowercase word or snake_case for <semantic_name>.\n"
        "Write nothing else — no preamble, no explanation, no restating the routine."
    )


# ── #1668: a skill captures ONLY collector-runnable steps ──────────────────────

_CREATE_OK = "You set up a collection: (collection_set result)\nCreated collection 'widget-prices'."


@pytest.mark.asyncio
async def test_lifecycle_call_is_dropped_from_the_recipe(db):
    """A demo that sets up a container mid-run (collection_set — a lifecycle call
    a collector can never run) has that step DROPPED from the captured skill (#1668):
    a skill renders into a collector prompt, so only collector-runnable steps belong
    in it.  The create's args (name/description) never become nonsense parameters,
    and the create doesn't count toward the read/write taxonomy."""
    create = (
        "collection_set",
        {"name": "widget-prices", "description": "watch the widget price"},
        _CREATE_OK,
        True,
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, create, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    names = [p.name for p in parameters_from_json(result.skill.parameters)]
    assert "name" not in names and "description" not in names and "skill" not in names


# ── #1665/#1804: the run-end narration frame (whole-render literal) ────────────


def test_skill_learned_narration_frame_renders_generic_name_and_demonstrated_on():
    """The narration frame (#1665) renders the GENERIC skill — name · intent · what
    it needs, via render_skill_brief — AND a line naming the INSTANCE it was
    demonstrated on: whole-render literal.

    The frame asks for a description a person can act on, so what it hands the model
    is a description (#1804/#1799).  Nothing in it is shaped like a call, so the
    reciting-the-recipe reading the numbered render invited is not available."""
    skill = Skill(
        name="watch-a-listing-price",
        steps=_WATCH_STEPS,
        parameters=_WATCH_PARAMS,
        intent="Watch a listing page's price and record it.",
        description="Watch a listing page's price and record it.",
        author="chat",
    )
    frame = Prompt.SKILL_LEARNED_NARRATION.format(
        skill=render_skill_brief(skill),
        demonstrated_on="watch the aurora deck 2 price and remember it",
    )
    assert frame == (
        "You just learned a reusable skill from what you did in this conversation — "
        "it's saved automatically, and here is exactly what it captured:\n\n"
        "watch-a-listing-price — Watch a listing page's price and record it. "
        "(needs: url)\n\n"
        "You demonstrated it on: watch the aurora deck 2 price and remember it\n\n"
        "Reply to the user now. FIRST answer what they actually asked: report the "
        "outcome of this round — the value you found and where you stored it — since "
        "this reply is the only one they receive. THEN tell them, in your own words, "
        "that you've learned this routine: name it "
        "by what it does generally (not just this one instance), say plainly what it "
        "does, and name what you'd need from them to run it again. Then offer to set "
        "it running on a schedule if they'd like."
    )
    # The leak #1799 recorded — `browse(queries=[{url}])` read aloud to the user —
    # has no source here: the frame carries no step, no tool name, and no brace
    # placeholder for the model to copy.
    assert "browse" not in frame and "{" not in frame


def test_skill_brief_render_omits_the_needs_tail_when_a_routine_needs_nothing():
    """A skill is an ARBITRARY tool sequence (#1783), so the brief render must hold
    at both extremes: a routine with no parameters carries no ``(needs: …)`` tail at
    all — the line reads as a plain ``name — what it's for``, not an empty "needs:
    none" a reader has to parse before ignoring — and step count never shows,
    however many tools the routine spans."""
    skill = Skill(
        name="tidy-the-receipts-folder",
        steps=_WATCH_STEPS,
        parameters="[]",
        intent="File receipts into the folder they belong in.",
        description="File receipts into the folder they belong in.",
        author="chat",
    )

    assert render_skill_brief(skill) == (
        "tidy-the-receipts-folder — File receipts into the folder they belong in."
    )


def test_learn_to_apply_eval_fixture_is_the_shape_this_pipeline_produces():
    """The learn → apply enactment case (#1706) starts from the world a completed
    teach round leaves behind, and builds its fixture skill by running THIS
    module's verdict application over that round's ledger rather than
    hand-writing the result.  Pin that here, where it costs no GPU: a distiller
    or labeller change that reshapes the skill fails a plain test instead of
    quietly handing the live case an easier — or impossible — starting world.
    (It has already earned this: #1777 made the write target a placeholder, and
    this pin is what reported the reshape rather than the eval discovering it on
    a GPU run.)

    Both placeholder ORIGINS ride in the one fixture — the entry key the
    assistant invented (#1770) and the write target the attachment decides
    (#1777) — so the demonstrated collection and the demonstrated key are BOTH
    absent from the render, which is the property the enactment case leans on.

    Since #1803 the fixture carries ONE parameter, not two: what the routine is
    ABOUT is baked into the step and never asked for again, so only the page is
    left to bind.  That is the shape the measured `elicit → learn` beat now
    produces 8 times out of 8, and this pin is what keeps the enactment case
    starting from it."""
    skill = learn_to_apply_fixture_skill()
    assert sorted(parameter.name for parameter in skill.parameters) == ["url"]
    placeholders = [
        substitution
        for step in skill.steps
        for substitution in step.substitutions
        if substitution.kind == SkillSubKind.PLACEHOLDER
    ]
    assert [substitution.description for substitution in placeholders] == [
        WRITE_TARGET_DESCRIPTION,
        "what to call the entry it saves",
    ]
    # The harm placeholders exist to prevent: a collector re-running this skill
    # must write neither the demonstrated key nor the demonstrated collection
    # back every cycle — and the ambient recipe must promise neither.
    rendered = render_skill(skill.steps, {"url": "https://example.test"})
    assert "aurora deck 2 price" not in rendered
    assert "aurora-deck-2-price" not in rendered
    # The constant renders VERBATIM — a leaf no substitution covers is a baked
    # value, so the routine still states what it pulls off the page.
    assert "the current price" in rendered
    assert "{what to call the entry it saves}" in rendered
    assert f"{{{WRITE_TARGET_DESCRIPTION}}}" in rendered
