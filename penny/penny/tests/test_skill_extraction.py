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
The #1824 rework: the two run-end draws are INDEPENDENT and answer different
questions from disjoint inputs — the LEAF LABELLER names every placeholder from the
routine's calls alone (no verdict exists any more; a leaf cannot become a parameter),
and the SKILL FRAMER writes the name, description and parameters from the user's ask
alone.  So the matrix here is: the framer's signature becomes the skill's, hardened,
with the slug fallback on any failure and a refusal when it asks for nothing · every
leaf renders what belongs there rather than the demonstrated value, per-leaf fallback
to the arg-derived name · both system prompts as whole-render literals.  The #1783
property is unchanged: the leaf that named a collection carries the ATTACHMENT MARK,
so what fills it is decided by what the routine is applied to, and the demonstrated
collection survives only as the step's verbatim ledger arguments.  Binding a framed
parameter to a particular leaf — the run-time join — is explicitly #1824's follow-on.
All content is synthetic (aurora / faux-market).
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
    LEAF_LABELLING_SYSTEM_PROMPT,
    SKILL_FRAMING_SYSTEM_PROMPT,
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
    skill is extracted with every unexplained leaf a PLACEHOLDER, the write content
    bound to the browse result, and the leaf naming the collection ATTACHMENT-marked —
    with a bare model there is no draw to name any of them, so each falls back to its
    arg-derived name and the marked one to the fixed string (#1777's constant, kept as
    exactly that fallback by #1783); the demonstrated collection is never rendered.

    A bare model also means no FRAMING, so the skill declares no parameters at all
    (#1824): what a skill asks for is the framer's answer, and there is nothing to
    infer it from.  The description is the run's bare utterance; the framework
    ``reasoning`` is gone."""
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted) and not result.replaced
    skill = result.skill
    assert skill.description == _UTTERANCE and skill.intent == _UTTERANCE
    assert skill.author == "chat" and skill.source_run_id == "run-A"
    assert parameters_from_json(skill.parameters) == []
    steps = steps_from_json(skill.steps)
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    subs = {tuple(s.path): s for s in steps[1].substitutions}
    content_sub = subs[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    # Every other leaf is a placeholder under its arg-derived name — legible, and never
    # the demonstrated value.  (The write KEY shares the browse query's leaf: same
    # value → one candidate.)
    rendered = render_skill(steps)
    assert "browse(queries=[{queries}], extract={extract})" in rendered
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
    """A text model client whose every chat returns ``content`` — enough when a test
    exercises ONE of the two run-end draws and lets the other fall back."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda _request, _count: LlmResponse(message=LlmMessage(role="assistant", content=content))
    )
    return model


def _two_draw_model(leaves: str, framing: str) -> MockLlmClient:
    """A model client answering the two run-end draws SEPARATELY (#1824), dispatched on
    the system prompt each carries — the leaf labeller's names, then the framer's
    signature.  Distinct answers are the point: the two draws are two questions from
    two inputs, and a test that fed both the same text could not show which one decided
    anything."""
    model = MockLlmClient()

    def respond(request: dict, _count: int) -> LlmResponse:
        system = request["messages"][0]["content"]
        drawn = framing if system == SKILL_FRAMING_SYSTEM_PROMPT else leaves
        return LlmResponse(message=LlmMessage(role="assistant", content=drawn))

    model.set_response_handler(respond)
    return model


def _content_of(model: MockLlmClient, system_prompt: str) -> str:
    """The USER-turn content of the last request carrying ``system_prompt`` — how a
    test reads what one draw was actually shown, which is the half of the two-draw
    separation that only its inputs can prove."""
    for request in reversed(model.requests):
        messages = request["messages"]
        if messages[0]["content"] == system_prompt:
            return " ".join(message.get("content", "") for message in messages[1:])
    raise AssertionError(f"no request carried the {system_prompt[:40]!r} contract")


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
    subs = {tuple(s.path): s for s in steps[1].substitutions}
    content_sub = subs[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    # The topic-name KEY did NOT false-bind — it stayed an unexplained leaf, which
    # since #1824 makes it a placeholder rather than a binding.
    assert subs[("entries", 0, "key")].kind == SkillSubKind.PLACEHOLDER


# ── #1824: the FRAMER names the skill; a failed draw falls back to the slug ────


@pytest.mark.asyncio
async def test_the_framer_names_and_describes_the_skill_from_the_ask(db):
    """A qualifying run's skill is named GENERICALLY by the framer, from the user's own
    messages: a tagged NAME:/DESCRIPTION: draw becomes the skill's slugged name +
    generic description (which the description_embedding anchors), NOT the instance
    utterance (#1665's property, now decided at the interface, #1824).  The
    demonstrated-on instance rides back for the frame.

    The two draws' INPUTS are asserted here too, because their separation is the
    ticket: the framer is handed the user's turns and NOTHING about the routine, and
    the leaf labeller is handed the routine and NOTHING the user said."""
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the page to open\n"
        "PLACEHOLDER extract: what_to_pull — what to read off it",
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PARAMETER url — the page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    # The instigating ask precedes the demonstration in the conversation — the
    # framing step must SEE it (#1658 intent grounding: the description carries the
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
    # The framing content is the ask, oldest first, with the DEMONSTRATING message in
    # it as a user turn — and nothing about the routine.  Presented under its own
    # unattributed heading, an earlier draw read the conversation as the only record of
    # what the user said and ruled the demonstrated values assistant-produced: correct
    # reasoning over a presentation that hid the speaker (#1770).
    framing_content = _content_of(model, SKILL_FRAMING_SYSTEM_PROMPT)
    assert "What the user asked for:" in framing_content
    assert "can you keep an eye on the zephyr lamp listing for me?" in framing_content
    assert _UTTERANCE in framing_content
    assert "browse(" not in framing_content, "the framer never sees the routine's calls"
    # And the leaf labeller's content is the routine, with no conversation at all: the
    # interface question that made the ask relevant is not asked of it any more.
    labelling_content = _content_of(model, LEAF_LABELLING_SYSTEM_PROMPT)
    assert "Routine steps, in the order they ran:" in labelling_content
    assert "browse(queries=[{queries}], extract={extract})" in labelling_content
    assert _UTTERANCE not in labelling_content, "the labeller never sees the ask"
    assert "zephyr" not in labelling_content


@pytest.mark.asyncio
async def test_a_failed_framing_falls_back_to_the_deterministic_slug(db):
    """When the framer never produces both tags, extraction does NOT block: it falls
    back to the deterministic slug of the triggering message + that message as the
    description, and the skill declares NO parameters (#1824).

    Declaring none is the honest degradation: parameters are what the framer decides,
    so inventing them from the leaves would put back exactly the inference this ticket
    removed.  The routine still reads, and re-teaching restores the signature."""
    model = _naming_model("I think this is a price-watching routine of some kind.")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # the fallback slug
    assert result.skill.description == _UTTERANCE
    assert parameters_from_json(result.skill.parameters) == []


@pytest.mark.asyncio
async def test_a_framing_missing_a_required_line_is_rerolled_then_falls_back(db):
    """The REQUIRED lines stay strict (#1814's half of the declaration): a draw
    carrying parameters but no ``NAME:`` never parses, so it is rerolled once on the
    unchanged context and then falls back — the per-item lines being best-effort does
    NOT make the draw as a whole best-effort."""
    model = _naming_model("DESCRIPTION: Look up a price and record it.\nPARAMETER url — the page")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # the fallback slug
    # Two draws each rerolled once (both contracts are violated by this text), then
    # both fell back — the reroll budget is per draw, not per run.
    assert len(model.requests) == 4


@pytest.mark.asyncio
async def test_a_framing_that_asks_for_nothing_is_refused(db):
    """The floor the framing prompt states, enforced (#1824): a draw that names NO
    parameter frames a skill that can only ever repeat the occasion it was taught on,
    which makes it a record of what happened rather than a skill.

    So it is a contract violation like any other — one reroll of the unchanged context,
    then honest degradation to the slug.  (Whether an empty signature should instead be
    legal is a real question; it is refused here because the shape it replaces refused
    the mirror case, and the answer is the code owner's.)"""
    model = _naming_model(
        "NAME: Watch the aurora deck 2 price\nDESCRIPTION: Check that one listing."
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # refused → the slug
    assert parameters_from_json(result.skill.parameters) == []


# ── #1824: the framer's parameters ARE the skill's, hardened ──────────────────


@pytest.mark.asyncio
async def test_framed_parameters_become_the_skills_parameters(db):
    """The parameters a skill declares are the FRAMER's answer and nothing else
    (#1824): each drawn name becomes the binding key at instantiation (display form ==
    invocation form) with its one-line what-to-supply, and they render in the
    parameters block.

    Where a bound parameter meets a leaf of the program is deliberately NOT decided
    here — the run-time join is #1824's follow-on — so this asserts the SIGNATURE,
    which is what instantiation validates against."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PARAMETER url — the page to look at\n"
        "PARAMETER what_to_find — what to pull from it"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description) for p in params] == [
        ("url", "the page to look at"),
        ("what_to_find", "what to pull from it"),
    ]
    assert all(p.required for p in params)
    rendered = render_skill_full(result.skill)
    assert "  - url (required): the page to look at" in rendered
    assert "  - what_to_find (required): what to pull from it" in rendered
    # Binding is by the semantic name — the params binding key at instantiation.
    assert [p.name for p in unbound_required_parameters(params, {})] == ["url", "what_to_find"]
    assert unbound_required_parameters(params, {"url": "u", "what_to_find": "w"}) == []


@pytest.mark.asyncio
async def test_framed_parameter_names_are_hardened_slugged_and_deduped(db):
    """Deterministic hardening of returned names (#1668's rule, unchanged — the name is
    the binding key): 'Page URL' slugs to 'page_url' (lowercase, spaces→underscores),
    two names that slug to the SAME key are disambiguated with a numeric suffix so a
    binding key can never collide, and one that slugs to nothing at all is dropped — a
    parameter nobody can name is not bindable.

    The SAME name written twice is a different case and collapses to one: a binding key
    repeated is one key, so the second line is the same parameter described again, never
    a second parameter."""
    model = _naming_model(
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAMETER Page URL — the page\n"
        "PARAMETER page url — the field\n"
        "PARAMETER Page URL — said twice, one key\n"
        "PARAMETER --- — nothing survives hardening"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description) for p in params] == [
        ("page_url", "the page"),
        ("page_url_2", "the field"),
    ]


# ── #1824: every leaf is a NAMED placeholder — the labeller's whole job ────────


@pytest.mark.asyncio
async def test_every_leaf_becomes_a_named_placeholder(db):
    """Every candidate leaf is a placeholder carrying the labeller's name and its
    one-line what-belongs-here (#1824) — including the ones a user could obviously
    supply, because whether the SKILL asks for something is not a property of a leaf.

    The demonstrated value is never frozen into the recipe: a collector re-running the
    skill would otherwise write that stale phrase into the collection every cycle,
    forever.  So the render is asserted WHOLE — every spot shows what belongs there,
    and neither demonstrated phrase appears anywhere."""
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the listing page to read\n"
        "PLACEHOLDER extract: what_to_pull — what to pull from the page\n"
        "PLACEHOLDER key: entry_key — a short label for the second entry\n"
        "PLACEHOLDER content: note_text — a note about the page you just read",
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price on a listing page and record it.\n"
        "PARAMETER url — the listing page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _INVENTED_WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    # The whole-skill render (what `skill_read` returns — the one surface the recipe is
    # the answer on, #1804): the framer's parameter block, and every leaf showing WHAT
    # BELONGS THERE in placeholder syntax, never the demonstrated value.
    assert render_skill_full(result.skill) == (
        "skill 'watch-a-listing-price'\n"
        "what it's for: Look up a price on a listing page and record it.\n"
        "parameters:\n"
        "  - url (required): the listing page to watch\n"
        "steps:\n"
        "  1. browse(queries=[{the listing page to read}], extract={what to pull from the page})\n"
        "  2. collection_write(memory={the collection this is set up on}, entries=["
        "{'key': {the listing page to read}, 'content': the value from step 1}, "
        "{'key': {a short label for the second entry}, "
        "'content': {a note about the page you just read}}])"
    )
    # Each placeholder also carries the labeller's NAME — the anchor the run-time join
    # binds against.  Nothing renders it yet (#1824's follow-on decides where it
    # surfaces), so it is asserted on the stored substitution.
    steps = steps_from_json(result.skill.steps)
    names = {
        sub.description: sub.name
        for step in steps
        for sub in step.substitutions
        if sub.kind == SkillSubKind.PLACEHOLDER
    }
    assert names["the listing page to read"] == "page_url"
    assert names["a note about the page you just read"] == "note_text"

    # The BRIEF render (#1804) — the same skill as the ambient section and the
    # narration frame see it: what it is and what it needs.
    assert render_skill_brief(result.skill) == (
        "watch-a-listing-price — Look up a price on a listing page and record it. "
        "(needs: url — the listing page to watch)"
    )

    # The collection the round wrote into is left exactly as the round left it:
    # learning does not instantiate (#1706 removed the run-end auto-attach).
    row = db.memories.get("aurora-prices")
    assert row is not None
    assert row.skill_name is None and row.extraction_prompt is None


@pytest.mark.asyncio
async def test_leaf_labelling_falls_back_per_leaf(db):
    """Per-leaf fallback, not all-or-nothing (#1824): a leaf the draw labels gets its
    name + description; one it omits — or one whose line is malformed, or names
    something never offered — keeps its arg-derived name, which renders exactly as the
    recipe rendered before any draw.

    That degradation is legible rather than wrong: ``{extract}`` says the spot is the
    tool's extract argument, which is true and useless, where a guessed description
    would be neither."""
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the page to look at\n"
        "PLACEHOLDER extract:\n"
        "PLACEHOLDER nonesuch: invented — a leaf nobody offered",
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAMETER url — the page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "queries=[{the page to look at}]" in rendered  # labelled
    assert "extract={extract}" in rendered  # blank description → the arg-derived name
    assert "invented" not in rendered  # a leaf nobody offered addresses nothing


@pytest.mark.asyncio
async def test_a_leaf_named_twice_keeps_its_arg_derived_name(db):
    """A draw that names one leaf TWICE contradicts itself (the contract asks for
    exactly one line each), so that leaf gets NO label and keeps its arg-derived name
    (#1824).

    Picking between two names would be arbitrary where the fallback is legible, and
    the same rule ran under the verdict contract for a sharper reason: letting the last
    line win let a stray trailing line decide."""
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the page to look at\n"
        "PLACEHOLDER extract: what_to_pull — what to pull from the page\n"
        "PLACEHOLDER extract: something_else — a second opinion",
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAMETER url — the page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "queries=[{the page to look at}]" in rendered
    assert "extract={extract}" in rendered, "the contradicted leaf keeps its fallback"


# ── #1814: the output shape is declared, so tolerance and strictness are one rule ─


@pytest.mark.asyncio
async def test_a_cosmetically_variant_line_still_carries_its_label(db):
    """Everything the DECLARED shape tolerates, in one draw (#1814).

    The motivating failure: a line arrived with an EN-dash where the parser partitioned
    on an em-dash, so the split found nothing and the entire remainder became the
    semantic name — a 60-character "name" carrying its own description, persisted as a
    binding key, with no reroll because the parse had "succeeded".  Any
    whitespace-delimited dash variant now separates the name from its description (a
    hyphen INSIDE a name has no spaces around it, so it is never mistaken for the
    separator).  A line may also arrive decorated with a list marker or bold, and a
    payload may arrive quoted — none of that is the model getting the contract wrong.

    A ZERO-WIDTH character inside the TAG is the same family (#1824): invisible, decided
    by nothing, and before the tolerance it ate the whole line.  A measured run drew
    ``PLACEHO<U+200B>LDER`` on three of four lines after writing the first correctly,
    costing six of eight checks with the names themselves perfectly good.  Note this is
    tolerance, NOT fuzzy tag matching — the sibling test pins that a genuine misspelling
    still fails its line."""
    model = _two_draw_model(
        "- PLACEHOLDER queries: page_url – the page to look at\n"
        "* **PLACEHO\u200bLDER extract: what_to_pull — what to pull from the page**",
        "**NAME:** Watch a listing price\n"
        'DESCRIPTION: "Look up a price and record it."\n'
        "- PARAMETER url – the page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "watch-a-listing-price"
    assert result.skill.description == "Look up a price and record it."
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description) for p in params] == [("url", "the page to watch")]
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "queries=[{the page to look at}]" in rendered  # the EN-dash split the line
    assert "extract={what to pull from the page}" in rendered  # the zero-width tag survived


@pytest.mark.asyncio
async def test_a_name_that_swallowed_its_description_is_no_label_at_all(db):
    """The backstop for whatever tolerance doesn't reach (#1814): a semantic name is a
    TOKEN, not a sentence, so a line whose "name" swallowed its own description is
    MALFORMED — and a malformed line is no label, never good data.  The leaf keeps its
    arg-derived name, which is the legible direction.  A near-miss tag
    (``PLACEHOLDERS``) is not a label line either.

    Nor is a MISSPELLED one (``PLACEHALDER``, a real draw, #1824): the zero-width
    tolerance strips an invisible transport artifact out of a tag, and stops exactly
    there — a tag the model got WRONG is a different fact from one the transport
    mangled, and fuzzy tag matching is how ``PARAMETRIC`` becomes a verdict."""
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the page to look at\n"
        'PLACEHOLDER extract: what_to_pull | the descriptor to look for (e.g., "price")\n'
        "PLACEHALDER key: entry_key — a misspelled tag is no label\n"
        "PLACEHOLDERS memory: not a label line",
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAMETER url — the page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "extract={extract}" in rendered  # malformed line → no label → the fallback
    steps = steps_from_json(result.skill.steps)
    assert "entry_key" not in [sub.name for step in steps for sub in step.substitutions], (
        "the misspelled tag labelled nothing — no fuzzy matching reached it"
    )
    target = {tuple(s.path): s for s in steps[1].substitutions}[("memory",)]
    assert target.kind == SkillSubKind.PLACEHOLDER
    assert target.description == WRITE_TARGET_DESCRIPTION  # PLACEHOLDERS labelled nothing


# ── #1783/#1824: where the routine writes is still the attachment's to fill ────


@pytest.mark.asyncio
async def test_the_write_target_keeps_its_mark_and_the_labellers_wording(db):
    """The leaf that named a collection is labelled like every other one, and its MARK
    survives (#1783 unchanged by #1824): what fills it is decided by what the routine
    is applied to, so the stored recipe carries the labeller's wording and applying the
    routine to a collection binds it to that collection's own name.

    Keyed to the MARK, not to any tool name: a skill is an arbitrary sequence, and a
    plugin's write is marked the same way."""
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the listing page to read\n"
        "PLACEHOLDER extract: what_to_pull — what to pull from the page\n"
        "PLACEHOLDER memory: destination — wherever this routine keeps its readings",
        "NAME: Watch a listing price\n"
        "DESCRIPTION: Look up a price and record it.\n"
        "PARAMETER url — the page to watch",
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    target = {tuple(s.path): s for s in steps[1].substitutions}[("memory",)]
    assert target.kind == SkillSubKind.PLACEHOLDER and target.attachment is True
    # The wording is the labeller's, not the fixed fallback.
    assert target.description == "wherever this routine keeps its readings"
    assert "memory={wherever this routine keeps its readings}" in render_skill(steps)
    assert "memory='aurora-watch'" in render_skill(retarget_writes(steps, "aurora-watch"))


@pytest.mark.asyncio
async def test_two_destinations_are_two_named_placeholders_and_two_parameters(db):
    """A routine writing to two places (#1783's load-bearing case) under the inverted
    pipeline: the two destinations are two distinctly-NAMED placeholders in the
    program, and the ask that named both is what makes them two PARAMETERS of the
    signature.

    What this branch deliberately does not do is JOIN them — binding a framed parameter
    to a particular leaf is the run-time join, explicitly #1824's follow-on — so both
    marked leaves still fall to the attachment at apply.  Recorded here as the current
    truth rather than asserted as the desired one: the interface half is decided, the
    program half is the next ticket."""
    db.memories.create_collection("aurora-archive", "older readings")
    second_write = (
        "collection_write",
        {"memory": "aurora-archive", "entries": [{"key": "aurora deck 2 price", "content": "old"}]},
        "You saved an entry to aurora-archive: (collection_write result)\nWrote 1 entry.",
        True,
    )
    model = _two_draw_model(
        "PLACEHOLDER queries: page_url — the listing page to read\n"
        "PLACEHOLDER extract: what_to_pull — what to pull from the page\n"
        "PLACEHOLDER memory: live_list — the collection to keep the current reading in\n"
        "PLACEHOLDER memory-2: archive — the collection to keep past readings in\n"
        "PLACEHOLDER content: previous_reading — the value to file in the archive",
        "NAME: Record a listing price in two places\n"
        "DESCRIPTION: Look up a price and record it in a live list and an archive.\n"
        "PARAMETER url — the listing page to read\n"
        "PARAMETER live_list — the collection to keep the current reading in\n"
        "PARAMETER archive — the collection to keep past readings in",
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
    assert [t.name for t in targets] == ["live_list", "archive"]
    assert [t.description for t in targets] == [
        "the collection to keep the current reading in",
        "the collection to keep past readings in",
    ]
    # Neither demonstrated collection reaches the recipe.
    rendered = render_skill(steps)
    assert _DEMO_COLLECTION not in rendered and "aurora-archive" not in rendered


def test_framing_system_prompt_whole_render():
    """Whole-render literal of the framing contract (#1824): the framing and its ONE
    input — what the user asked for — the three numbered asks (the point of the ask,
    then the name written from it, then the re-supply question as two named cases), the
    floor that keeps a skill bindable, and the enumerated output shape.

    **A parameter can only describe information the user DID provide**, and that
    constraint lives INSIDE the PARAMETER case rather than as a trailing imperative —
    because that is where the reasoning it corrects actually walks.  A measured draw
    asked for a `storage_path` reasoning, verbatim, *"They haven't given a file path…
    I'd say need to provide storage_path"*: a need the ask never mentioned, invented at
    the moment the model was deciding what a parameter IS.  So the definition itself
    says a parameter is one of the pieces they provided, and names the two shapes of
    unmentioned need (somewhere to keep it, what to call an entry) as the skill's own
    business.  The paired guard is the two-destinations case, where the user DID name
    the places and they must stay parameters.

    Nothing here mentions a tool call, an argument or a value the routine used: the
    interface is decided from the ask, and the pipeline this replaces failed by asking
    this question of implementation artifacts.  The worked example is deliberately a
    SUMMARISING routine, not the price watcher these tests demonstrate: an example
    drawn from the case in hand teaches pattern-matching on that case, and a skill is
    an arbitrary tool sequence a plugin may have supplied the verbs for.  It is also
    told as unquoted PROSE: a quoted example value is copied verbatim, and the one
    thing this prompt asks the model to compose is a name."""
    assert SKILL_FRAMING_SYSTEM_PROMPT == (
        "You are writing what a reusable skill IS: what it is called, what it is for, "
        "and what someone has to say to set it up. All you are given is what the user "
        "asked for, in their own words. Do three things:\n"
        "1. From their ask, work out what they were trying to get done. Their own "
        "words are the only evidence, and the point of the ask is what the skill is "
        "for.\n"
        "2. Name and describe the SKILL by that: a short verb-noun name for the KIND "
        "of task, generic — never the one occasion — and one line stating what it is "
        "for before any mechanics. A description that falls back on the information "
        "being specified, where the ask named something particular, has dropped the "
        "point of the ask — say what it actually was.\n"
        "3. Now take the pieces of information their ask handed over — every "
        "particular thing they named. For each one, ask: given the skill you just "
        "described, would they have to say it AGAIN to set this skill up on a new "
        "occasion?\n"
        "   - YES → it is a PARAMETER: one of the pieces of information they "
        "THEMSELVES PROVIDED that your framing does not already carry. The skill "
        "works the same way whatever it is, so it cannot be known until they say it. "
        "If they never said it, it cannot be a parameter at all — a need they never "
        "mentioned (somewhere to keep the result, when they never said where; what to "
        "call an entry, when they never named one) is the skill's own business to "
        "settle. Give it a short name (a single lowercase word or snake_case) and one "
        "line saying what to supply for it.\n"
        "   - NO → the name and description you just wrote already carry it, so it is "
        "not a parameter and gets no line at all. Asking for it would be asking them "
        "to tell you what they came to you for.\n"
        "   That they said it once settles nothing: they said all of it once, while "
        "asking. The question is only what is left to say once the skill already "
        "exists.\n"
        "   For example, asked once to summarise a long report: summarising is what "
        "they came for, so a skill that summarises reports needs only the report next "
        "time. Had they asked instead for whatever they say to be done to that "
        "report, they named no task at all, and both pieces would be parameters. "
        "Their ask is what tells you which skill you were asked for.\n"
        "Respond with these tagged lines and nothing else:\n"
        "NAME: <a short generic verb-noun name>\n"
        "DESCRIPTION: <one line: what the skill is for>\n"
        "PARAMETER <parameter_name> — <one line: what to supply for it>\n"
        "Write ONE line per parameter, and none for anything the framing already "
        "carries. At least one piece is always a parameter: a skill with nothing left "
        "to say to it can only ever repeat the one occasion it was asked for.\n"
        "Write nothing else — no preamble, no explanation, no restating the ask."
    )


def test_leaf_labelling_system_prompt_whole_render():
    """Whole-render literal of the leaf-labelling contract (#1824): the framing and its
    ONE input — the routine's calls, in the order they ran — the three numbered asks
    (what the spot holds, a name for the KIND of thing, one line of what belongs there),
    and the enumerated output shape with one line required per placeholder.

    There is no verdict in it, and no mention of the user at all: naming a spot is an
    implementation question, and every wording that asked this draw an interface
    question hit the same ceiling (#1821/#1823).  The one worked example is a MESSAGE
    body — an arbitrary spot from no case in this suite — because an example drawn from
    the case in hand teaches pattern-matching on that case."""
    assert LEAF_LABELLING_SYSTEM_PROMPT == (
        "You are naming the placeholders inside a routine. A routine is a fixed "
        "sequence of tool calls that gets run again on new occasions: the values it "
        "used the first time are gone, and each one leaves a placeholder that has to "
        "be filled in again on every run. You are given the calls IN THE ORDER THEY "
        "RAN, and the placeholders — each currently named after the tool argument it "
        "fills, and shown with the value that sat there the first time. For every "
        "placeholder:\n"
        "1. Work out what that spot HOLDS, from the call it sits in and what the calls "
        "around it do. The value it held once is an EXAMPLE of what belongs there, "
        "never the definition of it.\n"
        "2. Name it for the KIND of thing that belongs there — a single lowercase word "
        "or snake_case, named for the spot and not for the one value it happened to "
        "hold: a spot holding the body of a message is named for being a message body, "
        "never for the one message it carried.\n"
        "3. Describe in one line what belongs in that spot each time the routine "
        "runs.\n"
        "Write ONE line for EVERY placeholder, repeating its CURRENT name exactly so "
        "it maps back:\n"
        "PLACEHOLDER <current name>: <placeholder_name> — <one line: what belongs in "
        "this spot each run>\n"
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
    teach round leaves behind, and builds its fixture skill by running THIS module's
    label application over that round's ledger rather than hand-writing the result.
    Pin that here, where it costs no GPU: a distiller or labeller change that reshapes
    the skill fails a plain test instead of quietly handing the live case an easier —
    or impossible — starting world.  (It has already earned this: #1777 made the write
    target a placeholder, and this pin is what reported the reshape rather than the
    eval discovering it on a GPU run.)

    Since #1824 the fixture is EVERY leaf a named placeholder plus ONE declared
    parameter — the page, framed from the ask.  The signature is unchanged from what
    the measured `elicit → learn` beat produced 8 times out of 8, which is the only
    part of this world the apply turn reads; what changed underneath is that the
    program now names its spots instead of voting on them."""
    skill = learn_to_apply_fixture_skill()
    assert sorted(parameter.name for parameter in skill.parameters) == ["url"]
    placeholders = [
        substitution
        for step in skill.steps
        for substitution in step.substitutions
        if substitution.kind == SkillSubKind.PLACEHOLDER
    ]
    assert [(substitution.name, substitution.description) for substitution in placeholders] == [
        ("page_url", "the listing page to open"),
        ("what_to_pull", "what to read off that page each run"),
        ("destination", WRITE_TARGET_DESCRIPTION),
        ("entry_key", "what to call the entry it saves"),
    ]
    # The harm placeholders exist to prevent: a collector re-running this skill must
    # write neither the demonstrated key nor the demonstrated collection back every
    # cycle — and the ambient recipe must promise neither.
    rendered = render_skill(skill.steps, {"url": "https://example.test"})
    assert "aurora deck 2 price" not in rendered
    assert "aurora-deck-2-price" not in rendered
    assert "{what to call the entry it saves}" in rendered
    assert f"{{{WRITE_TARGET_DESCRIPTION}}}" in rendered
