"""Automatic skill extraction at chat-run end (#1658/#1665).

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
All content is synthetic (aurora / faux-market).
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.migrate import migrate
from penny.database.models import Skill
from penny.database.skill_store import (
    holes_from_json,
    holes_to_json,
    steps_from_json,
    steps_to_json,
)
from penny.database.skills import SkillHole, SkillStep, SkillSubKind, SkillSubstitution
from penny.llm.models import LlmMessage, LlmResponse
from penny.prompts import Prompt
from penny.skill_extraction import (
    ExtractionGate,
    NoExtraction,
    SkillExtracted,
    SkillExtractor,
)
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.skill_tools import render_skill_full

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


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "test.db"))
    db.create_tables()
    return db


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
    return SkillExtractor(
        db,
        cast(Any, mock or MockLlmClient()),
        cast(Any, model or MockLlmClient()),
        agent_name="chat",
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
async def test_read_write_run_qualifies_and_distils_correctly(tmp_path):
    """A browse (read) + collection_write (act) run is a routine: it qualifies and a
    skill is extracted with the query/extract as required holes, the write content
    bound to the browse result, and the write target NOT a hole (retarget owns it).
    The description is the run's bare utterance; the framework ``reasoning`` is gone."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted) and not result.replaced
    skill = result.skill
    assert skill.description == _UTTERANCE and skill.intent == _UTTERANCE
    assert skill.author == "chat" and skill.source_run_id == "run-A"
    # Holes: the browse query and the extract instruction; the write KEY reuses the
    # query's hole (same value → one shared parameter).  The write CONTENT is a
    # binding (it flowed from the browse), so it is NOT a hole.
    assert [hole.name for hole in holes_from_json(skill.holes)] == ["queries", "extract"]
    steps = steps_from_json(skill.steps)
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    content_sub = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    # The framework reasoning think-aloud is stripped from every stored step.
    assert all("reasoning" not in step.arguments for step in steps)


# ── Excluded: pure read, pure write, failed-write-only, bail, no-calls ─────────


@pytest.mark.asyncio
async def test_pure_read_run_is_excluded(tmp_path):
    """A run that only READ (answering a question) is not a routine → PURE_READ, no
    skill."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", "what does the aurora deck 2 cost?", [_BROWSE])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_READ)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_pure_write_run_is_excluded(tmp_path):
    """A run that only WROTE ('remember this' — the storage atom) is a plain write,
    not a job → PURE_WRITE, no skill."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", "remember the aurora deck 2 is $499", [_WRITE])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_WRITE)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_failed_write_only_run_is_excluded(tmp_path):
    """A run whose only write FAILED does not qualify: the failed call is filtered,
    leaving a pure read → PURE_READ, no skill (visible degradation, not a half-baked
    skill)."""
    db = _make_db(tmp_path)
    failed_write = ("collection_write", _WRITE_ARGS, "write failed", False)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, failed_write])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_READ)
    assert db.skills.list_all() == []


@pytest.mark.asyncio
async def test_bail_nudged_run_is_excluded(tmp_path):
    """A run poisoned by a text-bail nudge (the model failed to route a call through
    the tool channel) is unhealthy → BAILED, no skill — even though it read+wrote."""
    db = _make_db(tmp_path)
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
async def test_run_with_no_tool_calls_is_excluded(tmp_path):
    """A pure-conversation turn (no tool calls at all) yields NO_TOOL_CALLS."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", "hey how's it going", [])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.NO_TOOL_CALLS)


@pytest.mark.asyncio
async def test_run_with_no_certified_steps_is_excluded(tmp_path):
    """When a run had calls but NONE succeeded (or a pre-#1600 run has no stamps),
    nothing certifies → NO_CERTIFIED_STEPS, no skill (never an empty skill)."""
    db = _make_db(tmp_path)
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE], stamp_success=False)

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.NO_CERTIFIED_STEPS)


# ── Failed-step filtering: the surviving routine is extracted ──────────────────


@pytest.mark.asyncio
async def test_failed_step_is_filtered_from_the_routine(tmp_path):
    """A failed exploratory read is DROPPED (#1659 filter-not-refuse); the surviving
    browse + write still qualify and the extracted skill omits the failed call."""
    db = _make_db(tmp_path)
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
async def test_name_is_a_slug_of_the_utterance_with_urls_stripped(tmp_path):
    """The deterministic-slug FALLBACK (#1665): with a bare model (no NAME:/DESCRIPTION:
    naming draw), the name falls back to a slug of the triggering message — the URL is
    removed, lowercased, non-alphanumeric collapsed to hyphens, capped at 6 words — and
    the full message stays the description."""
    db = _make_db(tmp_path)
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
async def test_reteaching_the_same_utterance_replaces_by_name(tmp_path):
    """Re-demonstrating a routine whose message slugs to an existing skill name
    REPLACES that skill in place (one row, the newer steps)."""
    db = _make_db(tmp_path)
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
async def test_same_shape_and_meaning_replaces_keeping_existing_name(tmp_path):
    """A re-demonstration with a DIFFERENT wording (so a different slug) but the SAME
    tool sequence AND a description embedding within the house dedup threshold
    REPLACES the existing skill, keeping ITS name — the clean/flaky demo collapse."""
    db = _make_db(tmp_path)
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
async def test_different_meaning_inserts_a_new_skill(tmp_path):
    """A same-shape run whose meaning differs (embedding below threshold) is a NEW
    skill, never a false-replace — two skills coexist."""
    db = _make_db(tmp_path)
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
async def test_collector_run_is_not_extracted(tmp_path):
    """A run whose prompts are NOT the chat agent's (a background collector cycle)
    never yields a skill → NOT_CHAT, so extraction is chat-only by construction."""
    db = _make_db(tmp_path)
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
    db = Database(str(tmp_path / "seeded.db"))
    db.create_tables()
    migrate(db.db_path)
    _log_run(db, "run-A", "hi there", [])
    result = await _extractor(db).extract("run-A")
    assert isinstance(result, NoExtraction)
    assert db.skills.list_all() == []


# ── #1665 fixtures: orientation, the real framed browse result, a wrapped write ─

_FIND_RESULT = "You searched your memory: (find result)\nNo matching skill — nothing saved yet."
# The REAL browse-with-extract frame shape: a narration line carrying the
# ``(browse result)`` machine tag, the payload, then the fetch-handle tail.
_REAL_BROWSE_FRAME = (
    "You opened the Aurora Deck 2 listing (browse result)\n"
    "$499\n\n"
    "Full page content saved to browse-results#1 — read it there for anything more."
)
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
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, hole="url")
            ],
        )
    ]
)
_WATCH_PARAMS = holes_to_json([SkillHole(name="url", required=True)])


def _naming_model(content: str) -> MockLlmClient:
    """A text model client whose every chat returns ``content`` (the naming draw)."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda _request, _count: LlmResponse(message=LlmMessage(role="assistant", content=content))
    )
    return model


# ── #1665: orientation verbs excluded from steps AND the qualifying read ───────


@pytest.mark.asyncio
async def test_orientation_find_step_is_dropped_from_the_recipe(tmp_path):
    """A run that ORIENTS (find) then reads + writes is a routine, but the find call
    is registry-navigation, not routine: it is dropped from the distilled steps, and a
    find result echoing the query never manufactures a false binding (#1665)."""
    db = _make_db(tmp_path)
    find = ("find", {"query": "watch a listing price"}, _FIND_RESULT, True)
    _log_run(db, "run-A", _UTTERANCE, [find, _BROWSE, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    # find is gone; only the content read + the write survive, in run order.
    assert [step.tool for step in steps] == ["browse", "collection_write"]
    assert [step.source_ordinal for step in steps] == [2, 3]


@pytest.mark.asyncio
async def test_find_plus_write_only_is_a_pure_write_not_a_skill(tmp_path):
    """A find + write run has NO content read once orientation is excluded — a find
    does not count as the qualifying read — so it is a pure write (the storage atom),
    not a skill (#1665)."""
    db = _make_db(tmp_path)
    find = ("find", {"query": "aurora prices"}, _FIND_RESULT, True)
    _log_run(db, "run-A", _UTTERANCE, [find, _WRITE])

    result = await _extractor(db).extract("run-A")

    assert result == NoExtraction(gate=ExtractionGate.PURE_WRITE)
    assert db.skills.list_all() == []


# ── #1665: binding compares against the result PAYLOAD, not the frame ──────────


@pytest.mark.asyncio
async def test_wrapped_write_value_binds_against_the_result_payload(tmp_path):
    """A write value that WRAPS the browse output ('Current price … is $499.') binds
    to the browse step — the comparison strips the tool-result FRAME to its payload
    ('$499'), so the wraps direction fires (#1665/#1661 item 3).  The 'content' leaf
    is a binding, never a nonsense required parameter; a topic-name KEY still doesn't
    bind (it stays a real parameter)."""
    db = _make_db(tmp_path)
    browse = ("browse", {"queries": ["aurora deck 2 listing"]}, _REAL_BROWSE_FRAME, True)
    write = ("collection_write", _WRAP_WRITE_ARGS, _WRITE_OK, True)
    _log_run(db, "run-A", _UTTERANCE, [browse, write])

    result = await _extractor(db).extract("run-A")

    assert isinstance(result, SkillExtracted)
    steps = steps_from_json(result.skill.steps)
    content_sub = {tuple(s.path): s for s in steps[1].substitutions}[("entries", 0, "content")]
    assert content_sub.kind.value == "binding" and content_sub.step == 1
    hole_names = [hole.name for hole in holes_from_json(result.skill.holes)]
    assert "content" not in hole_names  # the wrapped value bound; it is NOT a parameter
    assert "key" in hole_names  # the topic-name key did NOT false-bind — it's a parameter


# ── #1665: generic naming via the micro-context, slug fallback ────────────────


@pytest.mark.asyncio
async def test_tagged_naming_micro_context_sets_a_generic_name_and_description(tmp_path):
    """A qualifying run's skill is named GENERICALLY by the naming micro-context: a
    tagged NAME:/DESCRIPTION: draw becomes the skill's slugged name + generic
    description (which the description_embedding anchors), NOT the instance
    utterance (#1665).  The demonstrated-on instance rides back for the frame."""
    db = _make_db(tmp_path)
    model = _naming_model(
        "NAME: Watch a listing price\nDESCRIPTION: Look up a price on a listing page and record it."
    )
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "watch-a-listing-price"  # the generic NAME, slugged
    assert result.skill.description == "Look up a price on a listing page and record it."
    assert result.skill.intent == "Look up a price on a listing page and record it."
    assert result.origin_message == _UTTERANCE


@pytest.mark.asyncio
async def test_untagged_naming_falls_back_to_the_deterministic_slug(tmp_path):
    """When the naming micro-context never produces both tags, extraction does NOT
    block: it falls back to the deterministic slug of the triggering message + that
    message as the description (#1665)."""
    db = _make_db(tmp_path)
    model = _naming_model("I think this is a price-watching routine of some kind.")
    _log_run(db, "run-A", _UTTERANCE, [_BROWSE, _WRITE])

    result = await _extractor(db, model=model).extract("run-A")

    assert isinstance(result, SkillExtracted)
    assert result.skill.name == "read-the-aurora-deck-2-listing"  # the fallback slug
    assert result.skill.description == _UTTERANCE


# ── #1665: the run-end narration frame (whole-render literal) ──────────────────


def test_skill_learned_narration_frame_renders_generic_name_and_demonstrated_on():
    """The narration frame (#1665) renders the GENERIC skill (name · intent ·
    parameters · steps, via the shared render_skill_full) AND a line naming the
    INSTANCE it was demonstrated on — whole-render literal, model-facing 'parameters'
    vocabulary."""
    skill = Skill(
        name="watch-a-listing-price",
        steps=_WATCH_STEPS,
        holes=_WATCH_PARAMS,
        intent="Watch a listing page's price and record it.",
        description="Watch a listing page's price and record it.",
        author="chat",
    )
    frame = Prompt.SKILL_LEARNED_NARRATION.format(
        skill=render_skill_full(skill),
        demonstrated_on="watch the aurora deck 2 price and remember it",
    )
    assert frame == (
        "You just learned a reusable skill from what you did in this conversation — "
        "it's saved automatically, and here is exactly what it captured:\n\n"
        "skill 'watch-a-listing-price'\n"
        "intent: Watch a listing page's price and record it.\n"
        "parameters: url (required)\n"
        "steps:\n"
        "1. browse(queries=[{url}])\n\n"
        "You demonstrated it on: watch the aurora deck 2 price and remember it\n\n"
        "Tell the user, in your own words, that you've learned this routine: name it "
        "by what it does generally (not just this one instance), say plainly what it "
        "does (the steps), and name what you'd need from them to run it again (its "
        "required parameters). Then offer to set it running on a schedule if they'd like."
    )
