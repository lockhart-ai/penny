"""Automatic skill extraction at chat-run end (#1658/#1665/#1770/#1828/#1850).

Drives ``SkillExtractor.extract`` over REAL-SHAPED logged runs — every tool call
carries the framework's top-level ``reasoning`` think-aloud, and the user turn is a
bare utterance (no fused ``---`` Live-context prefix), the #1661 shape.  Every run
here is a LEARN turn unless the case is about the gate, because that is the only turn
extraction runs on (#1850): the state the machine landed on is a parameter, so a case
states which turn it is demonstrating rather than inferring it from what the run did.
The matrix: a taught round extracts (correct holes/bindings, reasoning stripped) ·
read-only and write-only taught rounds extract too, since a routine's SHAPE is not a
requisite · an apply-shaped or idle-shaped run of the very same ledger extracts
NOTHING · no-calls / nothing-certified excluded (each naming its gate) · failed-step
filtering · name slugging · dedup by name and by shape+meaning.  The #1665 additions:
orientation verbs (``find`` etc.) are dropped from the recipe · a wrapped write value
binds against a prior result's PAYLOAD (the frame stripped) while a topic-name key
still doesn't · the run-end narration frame renders the name plus the demonstrated-on
instance.

The #1824 inversion, both halves.  The LABELLER names every spot and judges nothing: a
labelled spot becomes a placeholder carrying what belongs there — never the frozen
demonstrated value — while a spot the draw missed, malformed, or named twice keeps its
arg-derived required parameter, and an attachment-marked leaf is a placeholder either
way because a destination is never a parameter (#1827).  The FRAMER writes the
interface from the user's ask alone (#1830) — the name, the description and the
parameters — and its every violation class is pinned against
``MicroContext.frame_skill`` with the reroll count asserted, since "no partial salvage"
is a claim about how many times the model was asked.  Both system prompts ride as
whole-render literals, and the two draws are shown provably different documents.

All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from similarity.dedup import JobSide, is_same_job

from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState
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
    SKILL_FRAME_SYSTEM_PROMPT,
    SKILL_NAMING_SYSTEM_PROMPT,
    MicroContext,
    SkillSignature,
    slug_parameter_name,
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
) -> None:
    """Log one chat run REAL-SHAPED: the bare utterance turn (no fused Live-context),
    each tool call carrying the universal top-level ``reasoning`` think-aloud (#1661),
    and each call's framed result plus its structural ``tool_success`` stamp (#1600).

    ``stamp_success=False`` omits the stamp (a pre-#1600 run)."""
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

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

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

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    row = db.memories.get("aurora-prices")
    assert row is not None
    assert row.skill_name is None, "learning must not bind the skill to a collection"
    assert row.extraction_prompt is None, "and must not render a program into it"
    assert row.schedule is None, "and must not schedule anything"


# ── The gate is the STATE: only a learn turn extracts (#1850) ──────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [ConversationState.APPLY, ConversationState.IDLE, None],
    ids=["apply-turn", "idle-turn", "no-machine-history"],
)
async def test_only_a_learn_turn_extracts(db, state):
    """The SAME ledger the learn cases extract from yields NOTHING on any other turn
    (#1850) — the gate is what the turn WAS, never what it did.

    The measured escape (PR #1849): an apply turn enacted a skill, and the enactment's
    own browse-and-write made the run look like a routine to a shape test, so the tail
    minted a brand-new skill framed from a round that taught nothing — registry
    pollution beside the skill it had just applied.  An idle one-off that browses and
    writes reached the same place.  Absence of machine history is idle too (no rows =
    the cold start), so a deployment whose machine has never moved never learns by
    accident.

    The refusal is NAMED, not silent: the outcome carries ``NOT_LEARN``, so the run
    record says which gate declined and why."""
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A", state=state)

    assert result == NoExtraction(gate=ExtractionGate.NOT_LEARN)
    assert db.skills.list_all() == []


# ── No shape requisites: a taught round extracts whatever shape it had ─────────


@pytest.mark.asyncio
async def test_a_learn_run_that_only_read_extracts(db):
    """A taught routine that only READS is a routine (#1850): the read+write taxonomy
    is gone, so a round demonstrating "check this page for me each morning" is learned
    exactly as a round that also files the answer somewhere is.

    It used to be refused as PURE_READ, on the theory that a routine must sense AND act
    — a judgment about shape that the framework has no standing to make, and that the
    state gate makes unnecessary: the user was teaching, so what they taught is the
    skill."""
    assert not hasattr(ExtractionGate, "PURE_READ")
    _log_run(db, "run-A", "check the aurora deck 2 listing for me like this", [_BROWSE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert [step.tool for step in steps_from_json(result.skill.steps)] == ["browse"]


@pytest.mark.asyncio
async def test_a_learn_run_that_only_wrote_extracts(db):
    """The other half of the same rule (#1850): a taught round that only WRITES is a
    routine too.

    It used to be refused as PURE_WRITE — 'the storage atom, not a job'.  A plain
    'remember this' still mints nothing, because a plain 'remember this' is an IDLE
    turn and never reaches here; what does reach here is a user demonstrating a filing
    routine, which is a routine."""
    assert not hasattr(ExtractionGate, "PURE_WRITE")
    _log_run(db, "run-A", "here's how to file an aurora reading", [_WRITE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert [step.tool for step in steps_from_json(result.skill.steps)] == ["collection_write"]


@pytest.mark.asyncio
async def test_a_failed_write_is_filtered_and_the_read_still_extracts(db):
    """A taught round whose write FAILED keeps the half that worked: the failed call is
    filtered (#1659 filter-not-refuse), and the surviving read is the routine.

    Certified-by-execution is unchanged — nothing that failed enters a skill — but with
    no taxonomy left there is nothing for the survivor to be too small for."""
    failed_write = ("collection_write", _WRITE_ARGS, "write failed", False)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, failed_write])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert [step.tool for step in steps_from_json(result.skill.steps)] == ["browse"]


@pytest.mark.asyncio
async def test_the_health_gate_is_retired(db):
    """There is no BAILED gate any more (#1839).  The call-shaped-text bails it keyed
    on are discarded and re-rolled by the agent loop, so their markers can no longer
    appear in a completed run — a run that recovered is indistinguishable from one that
    never slipped, and it qualifies on its work like any other."""
    assert not hasattr(ExtractionGate, "BAILED")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)


# ── Nothing captured: the learn turn that FAILED (#1839's terminal check) ──────


@pytest.mark.asyncio
async def test_run_with_no_tool_calls_is_excluded(db):
    """A learn turn that made no tool call at all captured nothing → NO_TOOL_CALLS.

    This is the quantity floor that survives #1850, and it is what the learn terminal
    reads: a learn turn's one valid end is a skill in the registry, so a round with
    nothing in it fails the turn honestly rather than storing an empty routine."""
    _log_run(db, "run-A", "hey how's it going", [])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert result == NoExtraction(gate=ExtractionGate.NO_TOOL_CALLS)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_run_with_no_certified_steps_is_excluded(db):
    """The same floor one step in: a learn turn whose calls NONE succeeded (or a
    pre-#1600 run with no stamps) certifies nothing → NO_CERTIFIED_STEPS, no skill
    (never an empty skill)."""
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE], stamp_success=False)

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert result == NoExtraction(gate=ExtractionGate.NO_CERTIFIED_STEPS)
    assert db.skills.list_all() == []


# ── Failed-step filtering: the surviving routine is extracted ──────────────────


@pytest.mark.asyncio
async def test_failed_step_is_filtered_from_the_routine(db):
    """A failed exploratory read is DROPPED (#1659 filter-not-refuse); the surviving
    browse + write still qualify and the extracted skill omits the failed call."""
    failed_read = ("collection_read_latest", {"memory": "notes"}, "read failed", False)
    _log_run(db, "run-A", _UTTERANCE, [failed_read, _BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

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

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

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
    first = await extractor.extract("run-A", state=ConversationState.LEARN)
    assert isinstance(first, SkillExtracted) and not first.replaced

    # A second demonstration of the SAME routine (same utterance → same slug name).
    _log_run(db, "run-B", _UTTERANCE, [_BROWSE, _WRITE])
    second = await extractor.extract("run-B", state=ConversationState.LEARN)

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
    first = await extractor.extract("run-A", state=ConversationState.LEARN)
    assert isinstance(first, SkillExtracted)
    original_name = first.skill.name

    # Different wording (a different slug), same tool shape, same aurora meaning.
    _log_run(db, "run-B", "keep an eye on the aurora deck 2 price for me", [_BROWSE, _WRITE])
    second = await extractor.extract("run-B", state=ConversationState.LEARN)

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
    await extractor.extract("run-A", state=ConversationState.LEARN)
    _log_run(db, "run-B", "watch the harbor weather report", [_BROWSE, _WRITE])
    second = await extractor.extract("run-B", state=ConversationState.LEARN)

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

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert result == NoExtraction(gate=ExtractionGate.NOT_CHAT)


@pytest.mark.asyncio
async def test_fresh_migrated_registry_stays_empty_without_a_qualifying_run(tmp_path):
    """A prod-identical DB (create_tables + migrate) ships the skill table EMPTY; a
    non-qualifying turn leaves it empty (no seeds, no accidental extraction)."""
    db = migrated_db(str(tmp_path / "seeded.db"))
    _log_run(db, "run-A", "hi there", [])
    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)
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


def _run_end_model(*, labels: str = "", framing: str = "") -> MockLlmClient:
    """A text model client answering each RUN-END draw with its own canned output.

    The labeller and the framer share a client and differ only in their system prompt,
    so dispatching on it is what keeps a case's labelling fixture out of the framer's
    answer and vice versa — which is the point of the split: neither draw sees the
    other's evidence (#1824).  A draw with nothing canned for it gets an empty output,
    which violates its contract and takes the honest fallback — exactly the state a case
    that only cares about the other half wants."""
    model = MockLlmClient()

    def respond(request: dict, _count: int) -> LlmResponse:
        drawn = framing if _system(request) == SKILL_FRAME_SYSTEM_PROMPT else labels
        return LlmResponse(message=LlmMessage(role="assistant", content=drawn))

    model.set_response_handler(respond)
    return model


def _system(request: dict) -> str:
    """One logged request's system prompt — which run-end customer made it."""
    return next(
        (m.get("content", "") for m in request["messages"] if m.get("role") == "system"), ""
    )


def _draws(model: MockLlmClient, system_prompt: str) -> int:
    """How many times ONE run-end customer drew.  Counted by its system prompt because
    the two share a client: "this draw took N of the budget" is a claim about one
    customer, and a total would move whenever the other one's luck changed."""
    return sum(1 for request in model.requests if _system(request) == system_prompt)


def _last_request(model: MockLlmClient, system_prompt: str) -> dict:
    """The most recent request ONE run-end customer made."""
    return next(r for r in reversed(model.requests) if _system(r) == system_prompt)


def _user_turn(request: dict) -> str:
    """One logged request's user turn — the whole document a micro-context read."""
    return next((m.get("content", "") for m in request["messages"] if m.get("role") == "user"), "")


# ── #1665: orientation verbs are excluded from the captured steps ──────────────


@pytest.mark.asyncio
async def test_orientation_find_step_is_dropped_from_the_recipe(db):
    """A run that ORIENTS (find) then reads + writes is a routine, but the find call
    is registry-navigation, not routine: it is dropped from the distilled steps, and a
    find result echoing the query never manufactures a false binding (#1665)."""
    find = ("find", {"query": "watch a listing price"}, _FIND_RESULT, True)
    _log_run(db, "run-A", _UTTERANCE, [find, _BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    # find is gone; only the content read + the write survive, in run order.
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    assert [step.source_ordinal for step in steps] == [2, 3]


@pytest.mark.asyncio
async def test_find_plus_write_captures_the_write_alone(db):
    """A taught round that orients then writes captures the WRITE alone — the find is
    still dropped (#1665: a re-run re-orients itself, and a find result echoing its
    query would manufacture a false binding), and what is left is the routine.

    Under the retired taxonomy this run was refused outright, because a find did not
    count as the qualifying read and a write on its own was 'the storage atom'.  The
    exclusion was always about the RECIPE, never about whether the round was worth
    learning (#1850)."""
    find = ("find", {"query": "aurora prices"}, _FIND_RESULT, True)
    _log_run(db, "run-A", _UTTERANCE, [find, _WRITE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    assert [step.tool for step in steps] == ["collection_write"]
    assert [step.source_ordinal for step in steps] == [2]


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

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    content_sub = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    hole_names = [hole.name for hole in parameters_from_json(result.skill.parameters)]
    assert "content" not in hole_names  # the wrapped value bound; it is NOT a parameter
    assert "key" in hole_names  # the topic-name key did NOT false-bind — it's a parameter


# ── #1824: the two run-end draws share no evidence ────────────────────────────


@pytest.mark.asyncio
async def test_the_two_run_end_draws_are_shown_different_evidence(db):
    """Implementation and interface are decided apart, so the two draws are handed
    different documents (#1824) — and this is where that separation is real rather than
    declared.

    The LABELLER is shown the conversation that led to the routine with each turn
    attributed to its speaker (#1770/#1828) — the elicit round as ``penny:``, the
    demonstrating message as the last ``user:`` turn — plus the numbered steps and every
    spot in them.  Presented under its own unattributed heading, the draw once read the
    conversation as the only record of what the user said and reasoned correctly to the
    wrong answer; attribution is what fixed it.

    The FRAMER is shown the user's own turns and NOTHING else (#1830): no assistant
    turns, no steps, no spots.  It never learns which tools ran, and the labeller never
    learns what the user wanted, which is what stops the two contradicting each other.
    Both user turns are the whole ask, and neither turn is repeated."""
    labels = (
        "LABEL queries: listing_page — the page this routine reads\n"
        "LABEL extract: value_to_find — what to pull off the page each run\n"
        "LABEL memory: storage_collection — the collection this is set up on"
    )
    model = _run_end_model(labels=labels)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    # The instigating ask precedes the demonstration in the conversation, and Penny's
    # elicit round sits between them — the labelling step must SEE both (#1658: what a
    # spot IS is only legible against why the routine exists).
    ask = "can you keep an eye on the zephyr lamp listing for me?"
    elicit = "i don't have a routine for that — walk me through it once?"
    db.messages.log_message(direction="incoming", sender="user", content=ask)
    db.messages.log_message(direction="outgoing", sender="penny", content=elicit)

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert result.origin_message == _UTTERANCE
    naming_content = _user_turn(_last_request(model, SKILL_NAMING_SYSTEM_PROMPT))
    assert "Conversation that led to the construction of this routine" in naming_content
    assert f"user: {ask}" in naming_content
    assert f"penny: {elicit}" in naming_content
    assert f"user: {_UTTERANCE}" in naming_content, "the demonstrating turn is the user's"

    # The framer's whole document: the user's turns, one per line, and nothing else.
    assert _user_turn(_last_request(model, SKILL_FRAME_SYSTEM_PROMPT)) == f"{ask}\n{_UTTERANCE}"

    # Both user turns are the rendered document ALONE — the ask lives in the system
    # prompt, so neither draw carries an instruction wrapper.
    assert "Instruction:" not in naming_content
    for request in model.requests:
        assert not _user_turn(request).startswith("Instruction:")


@pytest.mark.asyncio
async def test_an_unlabelled_draw_leaves_every_spot_with_its_arg_derived_name(db):
    """When the draw comes back with no usable line at all, extraction does NOT block
    (#1828): every spot keeps its arg-derived required parameter and the skill takes the
    deterministic slug of the triggering message.

    A draw that labelled NOTHING is a contract violation like any other — re-drawn on
    the unchanged context, then the honest fallback — while a draw that labelled SOME
    spots is best-effort and keeps whatever it landed."""
    model = _run_end_model(labels="I think this is a price-watching routine of some kind.")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # the fallback slug
    assert result.skill.description == _UTTERANCE
    assert (
        _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    )  # the whole reroll budget, then the fallback
    assert [p.name for p in parameters_from_json(result.skill.parameters)] == ["queries", "extract"]


# ── #1824/#1828/#1830: the two halves land on one skill ───────────────────────

_LABELLED_ROUND = (
    "LABEL queries: listing_page — the page whose price this routine reads\n"
    "LABEL extract: value_to_find — what to pull off the page each run\n"
    "LABEL memory: storage_collection — the collection this is set up on\n"
    "LABEL key: note_key — what to call the note it saves\n"
    "LABEL content: note_text — the note it writes about the page"
)
_FRAMED_ROUND = (
    "NAME: price-watcher\n"
    "DESCRIPTION: keep a listing's current price up to date\n"
    "PARAMETER listing_url — the listing whose price to read"
)


@pytest.mark.asyncio
async def test_the_labeller_writes_the_program_and_the_framer_writes_the_interface(db):
    """The inversion's whole behaviour in one run (#1824/#1828/#1830): the round recorded
    the price AND a second entry it composed itself about the page, and the two run-end
    draws answer the two halves of what that leaves behind.

    The LABELLER names EVERY spot — the page, what to find, the labels, the destination —
    and each becomes a placeholder carrying what belongs there.  Freezing a demonstrated
    value is the specific failure that prevents (a collector re-running the skill would
    write that stale phrase into the collection every cycle, forever), so the recipe is
    asserted WHOLE: not one demonstrated value survives in it.

    The FRAMER writes the interface from the ask alone — the name the registry keys on,
    the one line the ambient row states, and the parameter the user re-supplies.

    **The declared interim of this beat**: nothing joins that parameter to a leaf yet, so
    it lives at SKILL level over an all-placeholder recipe — visible in both renders and
    enforceable at instantiation, while the program still reads in the labeller's
    wording.  The runtime join is the next beat."""
    model = _run_end_model(labels=_LABELLED_ROUND, framing=_FRAMED_ROUND)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _INVENTED_WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in params] == [
        ("listing_url", "the listing whose price to read", True)
    ]
    assert render_skill_full(result.skill) == (
        "skill 'price-watcher'\n"
        "what it's for: keep a listing's current price up to date\n"
        "parameters:\n"
        "  - listing_url (required): the listing whose price to read\n"
        "steps:\n"
        "  1. browse(queries=[{the page whose price this routine reads}], "
        "extract={what to pull off the page each run})\n"
        "  2. collection_write(memory={the collection this is set up on}, entries=["
        "{'key': {the page whose price this routine reads}, "
        "'content': the value from step 1}, "
        "{'key': {what to call the note it saves}, "
        "'content': {the note it writes about the page}}])"
    )

    # The BRIEF render (#1804) — the same skill as the ambient section and the
    # narration frame see it.  The framed parameter is what makes the needs tail say
    # something a user could act on, which is what instantiation reads.
    assert render_skill_brief(result.skill) == (
        "price-watcher — keep a listing's current price up to date "
        "(needs: listing_url — the listing whose price to read)"
    )

    # And job identity is decidable again (#1775 tier 1): the same skill bound to the
    # same params is the same job whatever the two collections are called — which needs
    # a skill to HAVE params, so it went dormant for the parameter-less interim.
    bound = {params[0].name: "https://faux-market.example/aurora-deck-2"}
    assert is_same_job(
        JobSide(name="deck-prices", skill=result.skill.name, params=bound),
        JobSide(name="listing-price-monitor", skill=result.skill.name, params=bound),
    )

    # The collection the round wrote into is left exactly as the round left it:
    # learning does not instantiate (#1706 removed the run-end auto-attach), so
    # there is no rendered program for a placeholder to be frozen into.
    row = db.memories.get("aurora-prices")
    assert row is not None
    assert row.skill_name is None and row.extraction_prompt is None


@pytest.mark.asyncio
async def test_a_failed_framing_falls_back_to_the_slug_with_nothing_to_bind(db):
    """The framer's failure rung (#1830): the routine takes the deterministic slug of the
    triggering message, that message as its description, and NO parameters.

    That is exactly the interim #1829 shipped as the DEFAULT, now reached only when the
    interface draw actually failed.  It costs the labelling nothing — the recipe is still
    every spot named — so a failed framing degrades the interface alone rather than the
    whole extraction, and a re-demonstration replaces it through the same path."""
    model = _run_end_model(labels=_LABELLED_ROUND, framing="a price watcher, I think")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _INVENTED_WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert (
        _draws(model, SKILL_FRAME_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    )  # the whole reroll budget
    assert result.skill.name == "read-the-aurora-deck-2-listing"
    assert result.skill.description == _UTTERANCE
    assert parameters_from_json(result.skill.parameters) == []
    assert render_skill_brief(result.skill) == f"read-the-aurora-deck-2-listing — {_UTTERANCE}"
    # The labelling half is untouched — the two draws fail alone.
    assert "{the page whose price this routine reads}" in render_skill(
        steps_from_json(result.skill.steps)
    )


@pytest.mark.asyncio
async def test_a_draw_that_misses_any_spot_fails_whole(db):
    """COVERAGE is checked, not tolerated (#1828, the code owner's ruling): an accepted
    draw may never contain an invalid line, and the caller KNOWS the offered set, so a
    draw that leaves any spot unnamed is a contract violation — re-drawn on the
    unchanged context, then an honest WHOLE-draw failure.

    The observed failure this closes: the tag itself decays mid-draw (``LABLE``),
    the parse rightly refuses the line, and the validator used to accept around it —
    costing that one spot its label silently.  Correctness of accepted results over
    salvage: no partial rescue, every spot keeps its arg-derived name."""
    model = _run_end_model(labels="LABEL queries: listing_page — the page this routine reads")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert (
        _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    )  # the whole reroll budget, then the fallback
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in params] == [
        ("queries", None, True),
        ("extract", None, True),
    ]
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "queries=[{queries}]" in rendered, "the covered spot is not salvaged either"
    assert "extract={extract}" in rendered

    # A DECAYED TAG is that same miss, arriving as the shape the run reported: the line
    # is unreadable, so the spot it meant to name is uncovered.
    decayed = _run_end_model(
        labels="LABLE queries: listing_page — the page this routine reads\n"
        "LABEL extract: value_to_find — what to pull off the page\n"
        "LABEL memory: storage_collection — where the reading is kept"
    )
    _log_run(db, "run-B", "check the aurora price again please", [_BROWSE, _WRITE])

    later = await _extractor(db, model=decayed).extract("run-B", state=ConversationState.LEARN)

    assert isinstance(later, SkillExtracted)
    assert _draws(decayed, SKILL_NAMING_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert [p.name for p in parameters_from_json(later.skill.parameters)] == ["queries", "extract"]


@pytest.mark.asyncio
async def test_a_spot_named_twice_fails_the_whole_draw(db):
    """A draw that names one spot TWICE contradicts itself (#1828 — the contract asks
    for exactly one line each), and there is no line to prefer: taking either would let
    a stray trailing line quietly rename a spot.  So it is the same contract violation
    as a missing line — the whole budget, then the draw fails and every spot keeps its
    arg-derived name."""
    model = _run_end_model(
        labels="LABEL queries: listing_page — the page this routine reads\n"
        "LABEL extract: value_to_find — what to pull off the page\n"
        "LABEL extract: detail_to_check — something else entirely\n"
        "LABEL memory: storage_collection — where the reading is kept"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert [p.name for p in parameters_from_json(result.skill.parameters)] == ["queries", "extract"]


@pytest.mark.asyncio
async def test_a_line_for_a_spot_nobody_offered_fails_the_draw(db):
    """A line naming something the content never listed is a spot INVENTED rather than
    named, so it fails the same coverage constraint (#1828) — the whole budget, then the
    draw fails.

    The prompt asks for one line per listed placeholder and none for anything else, so
    the validator and the contract say one thing.  It also keeps the shared-spot claim
    enforceable: a value filling two argument sites is ONE spot, and a draw that splits
    it keys its second line to a name nobody offered — which is precisely this."""
    model = _run_end_model(
        labels="LABEL queries: listing_page — the page this routine reads\n"
        "LABEL extract: value_to_find — what to pull off the page\n"
        "LABEL memory: storage_collection — where the reading is kept\n"
        "LABEL nowhere: invented_spot — a spot nobody offered"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert (
        _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    )  # the whole reroll budget, then the honest failure
    assert [p.name for p in parameters_from_json(result.skill.parameters)] == ["queries", "extract"]
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "invented_spot" not in rendered and "a spot nobody offered" not in rendered


@pytest.mark.asyncio
async def test_a_line_that_stops_after_the_name_says_nothing_belongs_there(db):
    """A line that stops after the semantic name is WELL-FORMED — the grammar's one
    optional field — so it covers its spot and the draw stands (the labelling eval
    scores the missing description as its own miss).

    The CONSUMER is where it is caught: the description is what the leaf RENDERS as, so
    a blank one would put an empty ``{}`` where the recipe should say what belongs
    there — a spot that stopped being bindable and says nothing, strictly worse than
    the arg-derived name it replaced.  So the extractor reads it as no label."""
    model = _run_end_model(
        labels="LABEL queries: listing_page\n"
        "LABEL extract: value_to_find — what to pull off the page\n"
        "LABEL memory: storage_collection — where the reading is kept"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == 1  # the line is well-formed: no reroll
    kept = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in kept] == [("queries", None, True)]
    rendered = render_skill(steps_from_json(result.skill.steps))
    assert "queries=[{queries}]" in rendered, "a blank description never renders an empty slot"
    assert "{}" not in rendered
    assert "extract={what to pull off the page}" in rendered


def test_the_binding_key_hardener_is_the_one_shipped_rule():
    """The hardening a drawn name must survive to be a binding key (#1668): lowercase,
    whitespace → underscores, nothing outside ``[a-z0-9_]``, no stray underscores — and
    empty when nothing survives, which is a name that could never be bound.

    ONE rule, at every site that needs it: the framer runs its minted parameter names
    through it before they leave the draw (#1830), and the labelling eval scores "did
    the draw produce a usable binding key" through THIS function rather than a copy."""
    assert slug_parameter_name("Page URL") == "page_url"
    assert slug_parameter_name("first news site") == "first_news_site"
    assert slug_parameter_name("queries-2") == "queries2"
    assert slug_parameter_name("  ticker_symbol  ") == "ticker_symbol"
    assert slug_parameter_name("!!") == ""


# ── #1814: the output shape is declared, so tolerance and strictness are one rule ─


@pytest.mark.asyncio
async def test_a_cosmetically_variant_label_line_still_carries_its_label(db):
    """Everything the DECLARED shape tolerates, in one draw (#1814).

    The motivating failure: a line arrived with an EN-dash where the parser partitioned
    on an em-dash, so the split found nothing and the entire remainder became the
    semantic name — a 60-character "name" carrying its own description, persisted as a
    skill's binding key, with no reroll because the parse had "succeeded".  Any
    whitespace-delimited dash variant now separates the name from its description (a
    hyphen INSIDE a name has no spaces around it, so it is never mistaken for the
    separator).  A line may also arrive decorated with a list marker or bold, or with a
    quoted payload — none of that is the model getting the contract wrong, so none of it
    costs a reroll under the coverage rule (#1828): a tolerated line is a well-formed
    line, and it covers its spot like any other."""
    model = _run_end_model(
        labels="- LABEL queries: listing_page – the page this routine reads\n"
        "* **LABEL extract: value_to_find — what to pull off the page**\n"
        'LABEL memory: "storage_collection" — where the reading is kept'
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == 1, (
        "tolerated decoration is not a contract violation"
    )
    assert parameters_from_json(result.skill.parameters) == []
    rendered = render_skill(steps_from_json(result.skill.steps))
    # The EN-dash split the name from its description, and the decoration came off.
    assert "queries=[{the page this routine reads}]" in rendered
    assert "extract={what to pull off the page}" in rendered
    assert "memory={where the reading is kept}" in rendered


@pytest.mark.asyncio
async def test_a_name_that_swallowed_its_description_costs_the_whole_draw(db):
    """The backstop for whatever tolerance doesn't reach (#1814): a semantic name is a
    TOKEN, not a sentence, so a line whose "name" swallowed its own description is
    MALFORMED — never good data.  A near-miss tag (``LABELS``) is not a line
    either.

    Under the coverage rule (#1828) each of those leaves its spot uncovered, so the DRAW
    fails rather than the spot quietly going unnamed — the whole budget, then every spot keeps
    its arg-derived name, and the attachment-marked write target falls back to the fixed
    wording (the attachment fills it, and no user ever could)."""
    model = _run_end_model(
        labels="LABEL queries: listing_page — the page this routine reads\n"
        'LABEL extract: what_to_find | the content descriptor to look for (e.g., "price")\n'
        "LABELS memory: storage_collection — not a label line"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert (
        _draws(model, SKILL_NAMING_SYSTEM_PROMPT) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    )  # the whole reroll budget, then the honest failure
    params = parameters_from_json(result.skill.parameters)
    assert [(p.name, p.description, p.required) for p in params] == [
        ("queries", None, True),
        ("extract", None, True),
    ]
    steps = steps_from_json(result.skill.steps)
    target = {tuple(s.path): s for s in steps[1].substitutions}[("memory",)]
    assert target.kind == SkillSubKind.PLACEHOLDER
    assert target.description == WRITE_TARGET_DESCRIPTION  # nothing labelled it
    assert target.attachment is True  # and the mark stands, for the render seam


# ── #1830: the framer's own contract — every violation class, pinned ──────────

# The framer draws over a document, not an offered set, so the constraint it is accepted
# against is entirely about the draw itself: it must mint at least one well-formed,
# distinctly-named parameter and carry no broken line.  Each violation is the same
# answer — re-drawn on the unchanged context for the whole budget, then an honest
# WHOLE-draw failure — and
# each is pinned WITH its reroll count, because "no partial salvage" is a claim about
# how many times the model was asked, not only about what came back.


async def _framed(drawn: str) -> tuple[SkillSignature | None, int]:
    """One framing draw answered by a mock model with ``drawn`` — the signature (or
    ``None``), and how many times the draw was made."""
    model = _run_end_model(framing=drawn)
    signature = await MicroContext(cast(Any, model)).frame_skill("the user's turns")
    return signature, len(model.requests)


@pytest.mark.asyncio
async def test_the_framer_writes_a_name_a_description_and_the_minted_parameters():
    """A well-formed draw IS the routine's interface (#1830): the generic name, the one
    line it is for, and one parameter per piece the user would have to say again — in
    draw order, because two parameters are told apart by which is which.

    The names come back HARDENED, unlike a leaf label's: this one is the binding key
    instantiation uses (``params={'listing_url': …}``), so the rule that makes it usable
    runs inside the draw rather than somewhere downstream.  And the tolerance the
    declared shape carries applies here like anywhere else — an EN-dash separates a name
    from its description without costing a reroll."""
    signature, draws = await _framed(
        "NAME: listing-price-comparer\n"
        "DESCRIPTION: compare what two marketplace listings currently cost\n"
        "PARAMETER Listing URL – the first listing to read\n"
        "PARAMETER rival_listing — the listing to compare it against"
    )

    assert draws == 1, "tolerated decoration is not a contract violation"
    assert signature is not None
    assert signature.name == "listing-price-comparer"
    assert signature.description == "compare what two marketplace listings currently cost"
    assert [(p.name, p.description) for p in signature.parameters] == [
        ("listing_url", "the first listing to read"),
        ("rival_listing", "the listing to compare it against"),
    ]


@pytest.mark.asyncio
async def test_a_draw_that_asks_for_nothing_fails_whole():
    """The floor the framing prompt states, enforced (#1830): a routine that needs
    nothing said to it can only ever repeat the one occasion it was shown.

    That dead end has been reached from the other side already — two runs produced
    `watch-a-price` with every leaf placeholdered, so not even the page could be re-bound.
    A signature with no parameters arrives at the same place by a new route, so the floor
    is structural: the draw is discarded and re-drawn for the whole budget, then an honest
    refusal rather than a routine nobody can point anywhere."""
    signature, draws = await _framed(
        "NAME: aurora-price-watcher\nDESCRIPTION: keep the aurora deck 2 price up to date"
    )

    assert signature is None
    assert draws == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_malformed_parameter_line_fails_the_whole_draw():
    """An accepted draw never contains an invalid line (#1828's ruling, carried here).

    With nothing offered, a broken line leaves no gap to notice — the parse simply drops
    it and the remaining lines look like a complete answer — so the framer refuses a draw
    the parse dropped anything from.  Both shapes of broken are the same violation: a
    "name" that swallowed its own description (a name is a TOKEN, not a sentence) and a
    line that never reached its description at all.  No partial salvage: the good line on
    either draw is discarded with the bad one."""
    swallowed, draws = await _framed(
        "NAME: listing-price-watcher\n"
        "DESCRIPTION: keep a marketplace listing's current price up to date\n"
        "PARAMETER listing_url — the listing whose price to read\n"
        'PARAMETER what_to_pull | the detail to read off it (e.g., "the price")'
    )

    assert swallowed is None
    assert draws == PennyConstants.DEGENERATE_REROLL_ATTEMPTS

    undescribed, draws = await _framed(
        "NAME: listing-price-watcher\n"
        "DESCRIPTION: keep a marketplace listing's current price up to date\n"
        "PARAMETER listing_url — the listing whose price to read\n"
        "PARAMETER how_often"
    )

    assert undescribed is None, "a parameter nobody can be told what to supply is not an interface"
    assert draws == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_parameter_named_twice_fails_the_whole_draw():
    """A parameter's name is its binding key, so two lines under one key would mean one
    of them silently disappearing at instantiation — the caller cannot prefer either, and
    taking the last would let a stray trailing line quietly redefine the interface.  The
    same contract violation as a broken line: re-drawn for the whole budget, then the draw
    fails.

    Names are compared HARDENED, because that is what they become: `Listing URL` and
    `listing_url` are one key, not two."""
    signature, draws = await _framed(
        "NAME: listing-price-comparer\n"
        "DESCRIPTION: compare what two marketplace listings currently cost\n"
        "PARAMETER listing_url — the first listing to read\n"
        "PARAMETER Listing URL — the listing to compare it against"
    )

    assert signature is None
    assert draws == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_signature_missing_its_name_or_description_fails_the_draw():
    """The NAME and DESCRIPTION lines are REQUIRED in the declared shape, so a draw
    missing either never parses (#1814) — the same reroll-then-fail, because half an
    interface is not an interface: the name is what the registry keys on and the
    description is what every judging surface reads."""
    unnamed, draws = await _framed(
        "DESCRIPTION: keep a marketplace listing's current price up to date\n"
        "PARAMETER listing_url — the listing whose price to read"
    )

    assert unnamed is None
    assert draws == PennyConstants.DEGENERATE_REROLL_ATTEMPTS

    undescribed, draws = await _framed(
        "NAME: listing-price-watcher\nPARAMETER listing_url — the listing whose price to read"
    )

    assert undescribed is None
    assert draws == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


# ── #1783/#1828: where a routine writes is decided by what it is applied to ────


@pytest.mark.asyncio
async def test_the_destination_is_filled_by_the_attachment_under_its_drawn_name(db):
    """The leaf that named a collection is named like every other spot, and what fills
    it is decided by where the routine is APPLIED (#1783/#1827 principle 4).  Nobody
    binds it: the leaf keeps its mark, renders the LABELLER's wording (not a hardcoded
    string) in the stored recipe, and applying the routine to a collection binds it to
    that collection's own name."""
    model = _run_end_model(
        labels="LABEL queries: listing_page — the page this routine reads\n"
        "LABEL extract: value_to_find — what to pull off the page\n"
        "LABEL memory: storage_collection — wherever this routine keeps its readings"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert parameters_from_json(result.skill.parameters) == []
    steps = steps_from_json(result.skill.steps)
    target = {tuple(s.path): s for s in steps[1].substitutions}[("memory",)]
    assert target.kind == SkillSubKind.PLACEHOLDER and target.attachment is True
    # The wording is the labeller's, not the fixed fallback.
    assert target.description == "wherever this routine keeps its readings"
    assert "memory={wherever this routine keeps its readings}" in render_skill(steps)
    assert "memory='aurora-watch'" in render_skill(retarget_writes(steps, "aurora-watch"))


@pytest.mark.asyncio
async def test_two_destinations_both_land_on_the_collection_the_routine_is_applied_to(db):
    """A routine writing to two places Penny named keeps BOTH marks, and applying it
    somewhere binds both to that collection (#1828).

    This supersedes #1783's other direction.  There, a user-supplied VERDICT cleared the
    mark, so two destinations the USER named stayed two separately-bound parameters —
    an outcome of the verdicts, never a rule.  With no verdict left to clear anything, a
    destination is never a parameter (#1827 principle 4) and the attachment fills every
    marked leaf.  If the framer beat needs a user-named destination back as a parameter,
    it decides that from the ask — which is exactly where that decision belongs."""
    db.memories.create_collection("aurora-archive", "older readings")
    second_write = (
        "collection_write",
        {"memory": "aurora-archive", "entries": [{"key": "aurora deck 2 price", "content": "old"}]},
        "You saved an entry to aurora-archive: (collection_write result)\nWrote 1 entry.",
        True,
    )
    model = _run_end_model(
        labels="LABEL queries: listing_page — the page this routine reads\n"
        "LABEL extract: value_to_find — what to pull off the page\n"
        "LABEL memory: live_list — where the current reading is kept\n"
        "LABEL memory-2: archive — where past readings are kept\n"
        "LABEL content: previous_reading — the value filed in the archive"
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE, second_write])

    result = await _extractor(db, model=model).extract("run-A", state=ConversationState.LEARN)

    assert isinstance(result, SkillExtracted)
    assert parameters_from_json(result.skill.parameters) == []
    steps = steps_from_json(result.skill.steps)
    targets = [{tuple(s.path): s for s in step.substitutions}[("memory",)] for step in steps[1:]]
    assert [t.description for t in targets] == [
        "where the current reading is kept",
        "where past readings are kept",
    ]
    assert all(t.attachment is True for t in targets)
    # Neither demonstrated collection reaches the recipe, and applying the routine
    # somewhere binds both leaves to that collection.
    rendered = render_skill(steps)
    assert _DEMO_COLLECTION not in rendered and "aurora-archive" not in rendered
    bound = render_skill(retarget_writes(steps, "current-prices"))
    assert bound.count("collection_write(memory='current-prices'") == 2


def test_framing_system_prompt_whole_render():
    """Whole-render literal of the framing contract (#1830) — AUTHORED AND APPROVED BY
    THE CODE OWNER in review, and swapped in wholesale.

    Five rounds of measured fixes each landed as another clause on the previous
    wording, and the result was an obscure private dialect ("the value from this
    occasion", "a piece taken out of what they gave you") that read as accretion
    because it was.  The guide's rewrite-not-accrete rule applies to the WHOLE prompt,
    not sentence by sentence: what every round learned is kept, but restated once, in
    plain terms, in the order the work is done — read the ask, name the routine, decide
    the parameters, with the PARAMETER line's own schema inline where that line is
    described.

    What each round contributed, all of it still here: the minimal-information question
    and the not-a-parameter list (what the framing carries · storage · cadence ·
    notification) · one value per parameter, never a list (run 1) · a generic name and a
    what-to-supply description, never this occasion's value (run 2) · distinct names for
    two things of the same kind (run 3) · the value's own KIND, never a piece decomposed
    out of it (run 4) · no examples in the description (run 5).

    Round 7 added the one clause the rewrite still lacked, and the failing sample named
    the gap in its own thinking: *"separate parameters for each but that's not
    scalable... they'd need to set up per site?"*.  The rule that two things are two
    parameters was already there and understood — what was missing was PERMISSION that
    several parameters is a normal shape for a routine, not a design smell to be talked
    out of.  So the bullet grants it outright ("it's okay to have several when they
    provided several") before restating the naming half.  A model reasoning its way to a
    worse answer out of unstated architectural anxiety is a presentation defect like any
    other.

    Round 8 (code-owner ruling) changed the step's ROOT.  Every earlier version asked the
    minimal-information question hypothetically — *picture the user coming back; what
    would they have to give?* — which is a question about an imagined future occasion, and
    an imagined occasion admits imagined needs: the measured class was parameters INVENTED
    whole (a `search_term` / `search_keyword` promoted for a routine whose ask named a page
    and no search).  Nothing in the wording forbade it, because the question never said
    where a parameter may come FROM.  So the step is now **enumerate, then filter**: list
    the pieces the user actually said — in reasoning, before deciding anything — and keep
    only those they would have to say again.  A parameter can only be one of those pieces,
    which makes an invention structurally unavailable rather than merely discouraged.  The
    filter clauses are unchanged; what changed is that they now filter a set that exists.

    Round 9 (code-owner authored) put two prohibitions on the NAME-AND-DESCRIBE step,
    where the not-a-parameter list had only ever spoken to step 3.  The description was
    baking the job's terms and the demonstration's specifics into what the routine IS,
    and a description is read by everything that later asks whether a routine COVERS an
    ask — so a routine described by the occasion it was first taught on stops matching
    the next one.  Both classes were measured on the idle → apply cases: "checks a
    webpage daily and notifies when a newer item appears" made three samples doubt a
    two-hourly ask was the same job and park in elicit, and "monitor a ferry timetable
    for updates to the late sailing entry" — where the late sailing IS what
    `search_phrase` binds — made an ask about a different sailing draw request.  So the
    step now says it twice, plainly: no timing, scheduling or notification, and no
    parameter's value.

    Two deliberate absences the code owner ruled on directly.  There is **no worked
    example of a filled line** in this version — its return, if the value-echo class
    comes back, is a measurement rather than an assumption.  And the type nouns 'url'
    and 'topic' ARE the contract being taught, not fixture leakage: teaching the
    canonical name for a kind of value is the point of the naming clause.  The
    'first_plot'/'second_plot' pair keeps the allotment register, far from every case.

    The three tagged lines still RENDER from the declared shape (`render_line`), so the
    tags and separators the model is told to write remain literally the ones the parse
    splits on."""
    assert SKILL_FRAME_SYSTEM_PROMPT == (
        "You are writing the public interface of a reusable routine. You are given what "
        "the user asked for, in their own words. Do three things:\n"
        "\n"
        "1. From what they asked for, extract the CORE USER INTENT — what they were trying "
        "to get done when they asked. Their own words are the evidence.\n"
        "\n"
        "2. Name and describe the ROUTINE by that intent: a short generic verb-noun name "
        "for the KIND of task — never the specific instance — and one line stating what "
        "the routine is for. Do not include any information about timing, scheduling, or "
        "notifications. Do not include any parameter's value in the name or description.\n"
        "\n"
        "3. Decide the PARAMETERS, starting from what the user actually provided. First, "
        "in your reasoning, list the pieces of information the user gave you — the things "
        "they said, not things they might have said. A parameter can only be one of these "
        "pieces; never something they didn't provide. Then keep only the pieces they would "
        "have to provide again to run this routine on a new occasion:\n"
        "   - A piece the name and description already carry is not a parameter — asking "
        "for it would be asking the user what they came to you for.\n"
        "   - Where results are kept, how often it runs, and whether to notify are never "
        "parameters — those are settled when the routine is set running.\n"
        "   - There is always at least one parameter.\n"
        "   Each parameter is one line: PARAMETER <name> — <description>\n"
        "   - name: the piece the user provided and how the routine uses it — if they "
        "pointed you at a website, 'url'; if they named a topic, 'topic'. Generic "
        "snake_case, never the particular site or thing's own name.\n"
        "   - A parameter holds ONE value, of the same kind the user gave it — a url stays "
        "a url, never a city pulled out of one, and never a list. One parameter for each "
        "piece they provided that survives; it's okay to have several when they provided "
        "several. Two of the same kind get names that tell them apart: 'first_plot', "
        "'second_plot'.\n"
        "   - description: one line saying what to supply. Do not include examples.\n"
        "\n"
        "Respond with these tagged lines and nothing else:\n"
        "NAME: <a short generic verb-noun name>\n"
        "DESCRIPTION: <one line: what the routine is for>\n"
        "PARAMETER <parameter_name> — <one line: what the user supplies for it>\n"
        "Write nothing else — no preamble, no explanation, no restating the ask."
    )


def test_labelling_system_prompt_whole_render():
    """Whole-render literal of the labelling contract (#1828): the framing, the three
    inputs it is given, the three numbered asks — what the spot IS, its name, what
    belongs there — the plain-language guard, and the ONE enumerated output line,
    rendered from the declared shape so the tags and separators the model is told to
    write are literally the ones the parse splits on.

    That guard names the STATE it applies to — a spot holding an instruction rather
    than a value — with the browse.extract case as its worked example, rather than
    keying on that argument: a skill is an arbitrary tool sequence, and a plugin verb
    taking a free-text instruction is the same spot under a name nobody enumerated.

    Two absences are the ticket: it never asks where a value came from, and it never
    asks what the routine is called.  Both were this draw's once; the first is gone (a
    spot is a placeholder unconditionally) and the second is the framer's, decided from
    the user's ask alone (#1824).  The opening states that as the job rather than
    forbidding the alternatives one by one — an instruction the model has to be ordered
    out of is a presentation that has failed.

    The worked examples are deliberately unquoted PROSE: a quoted example is copied
    verbatim ~82% of the time, and the one thing this prompt asks the model to compose
    is a name.

    The wire tag is ``LABEL`` — short, common, non-compound — because the tag before it
    DECAYED at measurable rates (#1842, the #1826 long-literal decay class).  Eleven
    characters of ``PLACEHOLDER`` came back as ``PLACEBLODER``, ``PLACEHOLER`` (three
    times in a single draw), ``PLACEHOlder``, and once carrying a zero-width character;
    the parse matches tags exactly, so every one of those read as no line and the
    coverage rule correctly failed the whole draw — discarding judgments that were
    themselves perfect.  The ruling was to change the WORD, not to loosen the match: a
    literal the model has to spell is a literal it can misspell."""
    assert SKILL_NAMING_SYSTEM_PROMPT == (
        "You are a naming step. A routine has just been demonstrated once, and every "
        "spot in it that gets filled in again each time it runs has been pulled out for "
        "you to name. Naming those spots is your whole job: they are all placeholders "
        "already, so nothing here asks where a value came from or what the routine as a "
        "whole should be called.\n"
        "You are given:\n"
        "- The conversation that led to the routine — the last user turn is the one "
        "that demonstrated it\n"
        "- The routine's numbered steps, each spot shown as {its current name}\n"
        "- The placeholders — every spot, with the argument site(s) it fills and the "
        "value it was demonstrated with\n"
        "Do this for EVERY placeholder you are given:\n"
        "1. Work out what that spot IS — the conversation says what the routine is for, "
        "its step says what the value is used to do, and the demonstrated value says "
        "what kind of thing goes there.\n"
        "2. Name it for what it is in this routine (e.g. listing_page, entry_key), NOT "
        "for the tool argument it happens to fill and NOT for the one value it was "
        "demonstrated with — a new value goes there every run.\n"
        "3. Describe in one line what belongs in that spot each time the routine runs.\n"
        "A spot that holds an INSTRUCTION rather than a value — what to look for "
        "wherever the routine reads, e.g. the spot filling browse.extract with the "
        "current price — is PLAIN LANGUAGE: there is no CSS-selector, XPath, or pattern "
        "machinery in this system, so NEVER name or describe one that way.\n"
        "Respond with one LABEL line per placeholder and nothing else:\n"
        "LABEL <current name>: <semantic_name> — "
        "<one-line description of what belongs there each run>\n"
        "Write ONE LABEL line for EVERY placeholder you were given, and none for "
        "anything else, repeating its CURRENT name exactly so it maps back. Two spots "
        "are never the same spot: give each its own name. Use a single lowercase word or "
        "snake_case for <semantic_name>.\n"
        "IMPORTANT: write nothing else — no preamble, no explanation, no restating the "
        "routine."
    )


# ── #1668: a skill captures ONLY collector-runnable steps ──────────────────────

_CREATE_OK = "You set up a collection: (collection_set result)\nCreated collection 'widget-prices'."


@pytest.mark.asyncio
async def test_lifecycle_call_is_dropped_from_the_recipe(db):
    """A demo that sets up a container mid-run (collection_set — a lifecycle call
    a collector can never run) has that step DROPPED from the captured skill (#1668):
    a skill renders into a collector prompt, so only collector-runnable steps belong
    in it.  The create's args (name/description) never become nonsense parameters."""
    create = (
        "collection_set",
        {"name": "widget-prices", "description": "watch the widget price"},
        _CREATE_OK,
        True,
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, create, _WRITE])

    result = await _extractor(db).extract("run-A", state=ConversationState.LEARN)

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
    """The learn → apply enactment cases (#1706) start from the world a completed
    teach round leaves behind, and build their fixture skills by running THIS
    module's own draw-application over that round's ledger rather than
    hand-writing the result — the labeller's spots through ``_apply_leaf_labels``,
    the framer's signature through ``_naming`` + ``_interface_parameters``.  Pin
    the auction script's own (one of five, the case that keeps the original id)
    here, where it costs no GPU: a distiller, labeller or framer change that
    reshapes the skill fails a plain test instead of quietly handing the live case
    an easier — or impossible — starting world.  (It has already earned this:
    #1777 made the write target a placeholder, and this pin is what reported the
    reshape rather than the eval discovering it on a GPU run.)

    Both draws are transcribed from the preceding beat's measured run (#1846), so
    the descriptions below are a labeller's real wording rather than a convenient
    invention — including the write target's, which a landed draw names like any
    other spot (``WRITE_TARGET_DESCRIPTION`` is the fallback for a spot no line
    covered, which is not this fixture's case).

    The shape is the framer's declared interim (#1830): the recipe is ALL
    placeholders — the labeller covers every spot or its draw fails whole, so a
    partly-named routine is not a state extraction can reach — and the interface
    is ONE skill-level parameter, the page, joined to no leaf yet.  Nothing the
    round demonstrated survives into the render: not the collection, not the key,
    not the page, not what it pulled off it.  That is the harm placeholders exist
    to prevent — a collector re-running this routine writing the demonstration's
    own values back every cycle — and it is the property the enactment case leans
    on when the apply turn has to supply the page itself."""
    skill = learn_to_apply_fixture_skill()
    assert sorted(parameter.name for parameter in skill.parameters) == ["url"]
    placeholders = [
        substitution
        for step in skill.steps
        for substitution in step.substitutions
        if substitution.kind == SkillSubKind.PLACEHOLDER
    ]
    assert [substitution.description for substitution in placeholders] == [
        "the url of the page to browse",
        "a plain text description of what information to retrieve from the page",
        "the identifier for the storage area where scraped data will be saved",
        "the key under which the extracted value is stored within that collection",
    ]
    # The whole recipe, verbatim — every spot says what belongs there and NOTHING the
    # round demonstrated survives into it: not the collection, not the key, not the
    # page, not what it pulled off the page.  Binding the framer's parameter changes
    # nothing here, because nothing joins it to a leaf yet (#1830's declared interim).
    assert render_skill(skill.steps, {"url": "https://example.test"}) == (
        "1. browse(queries=[{the url of the page to browse}], "
        "extract={a plain text description of what information to retrieve from the page})\n"
        "2. collection_write(memory={the identifier for the storage area where scraped "
        "data will be saved}, entries=[{'key': {the key under which the extracted value "
        "is stored within that collection}, 'content': the value from step 1}])"
    )
