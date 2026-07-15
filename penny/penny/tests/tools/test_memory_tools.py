"""Tests for memory tools.

Each tool is exercised through its ``execute`` coroutine end-to-end against a
real Database. The embedding path uses the existing ``mock_llm`` fixture so
similarity reads and dedup have something to work with.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import (
    WriteGateOutcome,
    WriteResult,
    render_run_calls,
)
from penny.database.models import MemoryEntry, MemoryRow, MessageLog, Skill
from penny.database.mutation_store import mutation_change_summary
from penny.database.skill_store import steps_from_json
from penny.database.skills import (
    SkillDraft,
    SkillHole,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    render_skill,
    retarget_writes,
)
from penny.datetime_utils import format_log_timestamp
from penny.llm.client import LlmClient
from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmConnectionError
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.collection_instantiation import render_unbound_holes
from penny.tools.memory_tools import (
    _INERT_JOB_ARGS_REFUSAL,
    _NO_TRIGGER_NOTE,
    _REBIND_NO_SKILL,
    _REINSTANTIATE_CONFLICT,
    _SKILL_GONE,
    CollectionArchiveTool,
    CollectionCatalogTool,
    CollectionCreateTool,
    CollectionDeleteEntryTool,
    CollectionGetTool,
    CollectionKeysTool,
    CollectionMergeTool,
    CollectionReadLatestTool,
    CollectionReadRandomTool,
    CollectionUnarchiveTool,
    CollectionUpdateTool,
    CollectionWriteTool,
    DoneTool,
    ExistsTool,
    FindMineTool,
    GetEventTool,
    LogAppendTool,
    LogCreateTool,
    LogReadTool,
    MemoryMetadataTool,
    ReadRunCallsTool,
    ReadSimilarTool,
    TestExtractionPromptTool,
    UpdateEntryTool,
    _format_duplicate,
    build_memory_tools,
)
from penny.tools.skill_tools import SkillCreateTool


def _make_db(tmp_path) -> Database:
    """Empty test DB with schema only — no migrations.

    Migration 0026 seeds three system log memories; these tool tests
    exercise the tool surface in isolation and declare exactly the
    memories they need.
    """
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_tables()
    return db


def _make_llm_client(mock_llm) -> LlmClient:
    """Build an LlmClient whose default embed handler returns distinct vectors
    per input text, so identical inputs collide and distinct inputs don't."""
    mock_llm.set_embed_handler(_hash_embed)
    return LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        max_retries=1,
        retry_delay=0.0,
    )


def _hash_embed(model: str, text: str | list[str]) -> list[list[float]]:
    """Deterministic embedding: text → unit vector where one axis is 1.0.

    Identical strings map to identical vectors; distinct strings map to
    different axes (cosine = 0), so dedup and similarity behave sensibly in
    tests without depending on a real embedding model.
    """
    inputs = text if isinstance(text, list) else [text]
    return [_single_hash_vec(t) for t in inputs]


def _single_hash_vec(text: str, dim: int = 4096) -> list[float]:
    """Bag-of-words deterministic embedding.  Each word picks an axis via
    SHA-256 → modulo ``dim``; the vector is L2-normalised so cosine is
    comparable across strings.  Identical strings map to identical
    vectors; strings sharing words have meaningful cosine > 0;
    fully-distinct strings map to cosine = 0."""
    vec = [0.0] * dim
    words = text.lower().split() or [text]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:8], "big") % dim
        vec[axis] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class _FailingEmbedClient:
    """An embedding client whose every embed call fails transiently.

    ``embed_text`` catches ``LlmError`` and returns ``None``, so the write path
    hits the fail-hard branch and refuses to persist a vectorless entry (#1412).
    """

    model = "test-model"

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        raise LlmConnectionError("embedding backend unavailable")


class _KeyOnlyFailingEmbedClient:
    """Fails only the embed of a specific string (the key), returning a real
    vector for everything else (the content).

    ``CollectionWriteTool._build_entry`` embeds key then content in two calls, so
    this reproduces a transient miss that lands on the key alone — the entry would
    be stored missing its key vector, which the write path refuses (an entry must
    carry all its vectors).  The startup backfill is the safety net for a
    key-null row that reaches the corpus by another path (migration-seeded content,
    #1468), but the write path still refuses at write time rather than persist a
    vectorless, recall-invisible row that only a later restart would repair.
    """

    def __init__(self, key: str) -> None:
        self._key = key
        self.model = "test-model"

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        inputs = text if isinstance(text, list) else [text]
        if self._key in inputs:
            raise LlmConnectionError("key embed failed")
        return [_single_hash_vec(t) for t in inputs]


# ── Seeding helpers ───────────────────────────────────────────────────────────
#
# ``collection_create`` is now the skill-instantiation front door (#1591): it no
# longer takes an ``extraction_prompt`` and refuses a near-duplicate.  Tests that
# only need a collection to EXIST (to exercise writes/reads/mutations/etc.) seed it
# directly through the store (``_seed_collection``) — the honest, idempotency-free
# way to stand one up — with a deterministic description anchor so dedup/similarity
# behave.  ``_seed_watch_skill`` upserts the fictional "watch a peak's elevation"
# skill the create-flow tests instantiate.

_SKILL_NAME = "Watch elevation"
_SKILL_HOLE = "peak"


def _seed_collection(
    db,
    *,
    name: str,
    description: str = "x",
    extraction_prompt: str = "test fixture extraction prompt",
    collector_interval_seconds: int = 3600,
    intent: str = "test intent",
    notify: bool = False,
    archived: bool = False,
) -> MemoryRow:
    """Stand up a collection through the store (no tool, no idempotency check) so a
    test that just needs one to exist doesn't drive the whole create front door."""
    return db.memories.create_collection(
        name,
        description,
        archived=archived,
        extraction_prompt=extraction_prompt,
        collector_interval_seconds=collector_interval_seconds,
        description_embedding=_single_hash_vec(description),
        intent=intent,
        notify=notify,
    )


def _watch_skill_steps() -> list[SkillStep]:
    """The fictional demonstration's steps: a {peak} hole reused in the browse query
    and the write key, and step 1's reading flowing into step 2 as a binding."""
    return [
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [_SKILL_HOLE], "extract": "the elevation above sea level"},
            substitutions=[
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, hole=_SKILL_HOLE)
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments={"memory": "elevations", "entries": [{"key": _SKILL_HOLE, "content": "x"}]},
            substitutions=[
                SkillSubstitution(
                    path=["entries", 0, "key"], kind=SkillSubKind.HOLE, hole=_SKILL_HOLE
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"], kind=SkillSubKind.BINDING, step=1
                ),
            ],
        ),
    ]


def _seed_watch_skill(
    db,
    *,
    name: str = _SKILL_NAME,
    intent: str = "watch a peak's elevation and save it",
    description: str = "watch a peak's elevation and save it",
    holes: list[SkillHole] | None = None,
    steps: list[SkillStep] | None = None,
    embed: bool = True,
) -> str:
    """Upsert the fictional watch-a-peak skill the create-flow tests instantiate;
    returns its name.  ``embed=False`` seeds it without a description anchor —
    for tests that resolve it by exact name and must keep it OUT of the
    resolve-by-meaning candidate pool (an unembedded skill is silently absent)."""
    draft = SkillDraft(
        name=name,
        intent=intent,
        description=description,
        steps=steps if steps is not None else _watch_skill_steps(),
        holes=holes if holes is not None else [SkillHole(name=_SKILL_HOLE, required=True)],
        source_run_id="run-teach",
    )
    embedding = _single_hash_vec(description) if embed else None
    db.skills.upsert(draft, author="chat", description_embedding=embedding)
    return name


# THE money literal — a skill + params flowing through the real front door into the
# collection's stored ``extraction_prompt``: the {peak} hole bound verbatim in both
# the browse query and the write key, the binding kept legible, and the write
# RETARGETED to the collection's own name (#1629 — the demo wrote to 'elevations').
_MONEY_LITERAL = (
    "1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
    "2. collection_write(memory='cinder-elevation', "
    "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
)

# The on_advance creation echo — the trigger line reads back the source-driven
# form (#1604), the write retargeted to this collection (#1629), everything else
# identical to the recurring echo shape.
_ON_ADVANCE_ECHO_LITERAL = (
    "Created collection 'chained-watch' from skill 'Watch elevation':\n"
    "  intent: digest events as they land\n"
    "  skill: Watch elevation\n"
    "  params: peak=Cinder Peak\n"
    "  trigger: on advance of events-log\n"
    "  notify: False\n"
    "  expires: never\n"
    "  extraction_prompt: |\n"
    "    1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
    "    2. collection_write(memory='chained-watch', "
    "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
)

# The whole creation echo — skill · bound params · trigger · notify · expiry · the
# rendered routine (the money literal, indented) — confirmed back to the user.
_CREATE_ECHO_LITERAL = (
    "Created collection 'cinder-elevation' from skill 'Watch elevation':\n"
    "  intent: watch Cinder Peak's elevation\n"
    "  skill: Watch elevation\n"
    "  params: peak=Cinder Peak\n"
    "  trigger: every 1h\n"
    "  notify: True\n"
    "  expires: never\n"
    "  extraction_prompt: |\n"
    "    1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
    "    2. collection_write(memory='cinder-elevation', "
    "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
)


class TestCollectionCreateFrontDoor:
    """The skill-instantiation front door (#1591): resolve a skill by name/meaning,
    bind its holes, render its steps into the stored prompt, and refuse a
    near-duplicate (#1567).  Results are model-facing text, asserted as whole
    renders."""

    @pytest.mark.asyncio
    async def test_instantiates_skill_and_stores_the_rendered_prompt(self, tmp_path):
        """A clean name match binds the params, renders the skill's steps into the
        collection's extraction_prompt (the money literal), and echoes skill /
        params / trigger / notify / expiry."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation",
            intent="watch Cinder Peak's elevation",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
            notify=True,
        )
        assert result.success and result.mutated
        # The whole echo confirms exactly what landed, without confabulation.
        assert result.message == _CREATE_ECHO_LITERAL
        # The money literal is the stored prompt — a skill rendered through the door.
        stored = db.memories.get("cinder-elevation")
        assert stored.extraction_prompt == _MONEY_LITERAL
        assert stored.intent == "watch Cinder Peak's elevation"
        # notify persists — the sole emission flag now (#1557 retired ``published``).
        assert stored.notify is True

    @pytest.mark.asyncio
    async def test_unbound_required_hole_is_refused_naming_it(self, tmp_path):
        """A skill instantiated without binding a required hole is refused, naming
        the missing parameter and the params shape to supply — nothing created."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="no-peak",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={},
            interval=3600,
        )
        assert result.success is False
        assert result.message == (
            "Can't instantiate 'Watch elevation': the required parameter(s) peak aren't "
            "bound. Pass them in params (e.g. params={'peak': <value>}), then call "
            "collection_create again."
        )
        assert db.memories.get("no-peak") is None

    @pytest.mark.asyncio
    async def test_no_skill_found_elicits_teaching(self, tmp_path):
        """A skill query matching nothing returns the reshaped #1471/#1629 elicitation —
        it NARRATES the two-step teach bootstrap (set up the container → walk me through
        once → skill_create → attach via collection_update), asserted whole."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)  # exists, but shares no words with the query
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="mystery",
            intent="do the mystery thing",
            skill="xyzzy flibbertigibbet quux",
            interval=3600,
        )
        assert result.success is False
        assert result.message == (
            "I don't know how to \"xyzzy flibbertigibbet quux\" yet — there's no skill for "
            "it, so there's nothing to instantiate. Here's how we teach one:\n"
            "1. Set up the container first: collection_create(name=<slug>, "
            'intent="xyzzy flibbertigibbet quux") with NO skill — a storage-only collection '
            "nothing runs against yet.\n"
            "2. Walk me through getting the data ONCE, here in chat, so I actually do it "
            "(browse, extract, and collection_write the result into that collection).\n"
            "3. Save that run as a skill: skill_create(name=<title>, from_run=<that run's "
            "id>, steps=<range>).\n"
            "4. Attach it to make the collection do the job: collection_update(name=<slug>, "
            "skill=<title>, params={…}, interval=<seconds>, notify=<true/false>)."
        )
        assert db.memories.get("mystery") is None

    @pytest.mark.asyncio
    async def test_ambiguous_meaning_returns_candidates_never_picks(self, tmp_path):
        """A paraphrase (not an exact name) that matches a skill by meaning returns
        the ranked candidate(s) + how to narrow — never a silent pick."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)  # description "watch a peak's elevation and save it"
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="some-watch",
            intent="watch a thing",
            skill="watch elevation save",  # shares words → fuzzy match, not exact name
            interval=3600,
        )
        assert result.success is False
        assert result.message == (
            'I know a few skills close to "watch elevation save" — I won\'t guess which '
            "you mean:\n"
            "1. Watch elevation — watch a peak's elevation and save it\n"
            "To use one, call collection_create again with skill='<its exact name>'. If "
            "none of these is the process you mean, walk me through it once and I'll learn "
            "it as a new skill."
        )
        assert db.memories.get("some-watch") is None

    @pytest.mark.asyncio
    async def test_active_near_duplicate_is_refused_naming_reuse(self, tmp_path):
        """Instantiating a collection whose intent semantically duplicates an active
        one creates nothing and points at reuse + the deliberate override (#1567)."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        _seed_collection(
            db,
            name="jacket-price",
            description="watch the blue jacket price",
            intent="watch the blue jacket price",
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="jacket-monitor",
            intent="watch the blue jacket price",  # same purpose → near-duplicate
            skill=_SKILL_NAME,
            params={"peak": "jacket"},
            interval=3600,
        )
        assert result.success is False
        assert result.message == (
            "Already have a collection for this: 'jacket-price' (active) — it covers the "
            "same thing, so I didn't create a second one. Reuse it: read it with "
            "collection_read_latest('jacket-price'), or adjust it with "
            "collection_update(name='jacket-price', ...). If this really is a distinct "
            "task, create it deliberately with collection_create(..., create_anyway=true)."
        )
        assert db.memories.get("jacket-monitor") is None

    @pytest.mark.asyncio
    async def test_tombstone_near_duplicate_surfaces_the_archived_row(self, tmp_path):
        """A near-duplicate of an ARCHIVED collection surfaces the tombstone + its
        archive time and offers unarchive or a deliberate override — never a silent
        proceed (#1567)."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        _seed_collection(
            db,
            name="jacket-price",
            description="watch the blue jacket price",
            intent="watch the blue jacket price",
            archived=True,
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="jacket-monitor",
            intent="watch the blue jacket price",
            skill=_SKILL_NAME,
            params={"peak": "jacket"},
            interval=3600,
        )
        assert result.success is False
        assert (
            "There's an archived collection for this: 'jacket-price' (archived " in result.message
        )
        assert "collection_unarchive('jacket-price')" in result.message
        assert "create_anyway=true" in result.message
        assert db.memories.get("jacket-monitor") is None

    @pytest.mark.asyncio
    async def test_create_anyway_overrides_the_duplicate_check(self, tmp_path):
        """The deliberate override creates the near-duplicate the check would refuse —
        a distinct, explicit act, never a default."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        _seed_collection(
            db,
            name="jacket-price",
            description="watch the blue jacket price",
            intent="watch the blue jacket price",
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="jacket-monitor",
            intent="watch the blue jacket price",
            skill=_SKILL_NAME,
            params={"peak": "jacket"},
            interval=3600,
            create_anyway=True,
        )
        assert result.success and result.mutated
        assert db.memories.get("jacket-monitor") is not None

    @pytest.mark.asyncio
    async def test_one_shot_run_at_trigger_persists(self, tmp_path):
        """The once-shaped trigger (run_at + max_runs) persists the schedule; the
        echo reads it back as a one-time run."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="one-shot",
            intent="check the peak once tomorrow",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            run_at="2026-12-25T09:00:00Z",
            max_runs=1,
        )
        assert result.success
        assert "trigger: runs at 2026-12-25 09:00 UTC, once" in result.message
        row = db.memories.get("one-shot")
        assert row.max_runs == 1
        assert row.run_at is not None

    @pytest.mark.asyncio
    async def test_no_trigger_is_refused(self, tmp_path):
        """A collection with no trigger would never run (silent degradation) — it's
        refused up front, nothing created."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="no-trigger",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
        )
        assert result.success is False
        assert "no trigger" in result.message
        assert db.memories.get("no-trigger") is None

    @pytest.mark.asyncio
    async def test_both_trigger_forms_are_refused(self, tmp_path):
        """Setting both a recurring interval and a run_at schedule is refused — the
        trigger union is exclusive (the schedule is checked before any skill work)."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="both-forms",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
            run_at="2026-12-25T09:00:00Z",
            max_runs=1,
        )
        assert result.success is False
        assert "Pick one trigger" in result.message
        assert db.memories.get("both-forms") is None

    @pytest.mark.asyncio
    async def test_run_at_without_max_runs_is_refused(self, tmp_path):
        """A run_at schedule needs a max_runs bound (else it never retires) — refused
        naming the missing bound."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="unbounded",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            run_at="2026-12-25T09:00:00Z",
        )
        assert result.success is False
        assert "needs max_runs" in result.message
        assert db.memories.get("unbounded") is None

    @pytest.mark.asyncio
    async def test_bad_expires_at_is_actionable(self, tmp_path):
        """A malformed end-condition datetime is refused with the accepted shape, not
        a raw parse error — nothing created."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="bad-expiry",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
            expires_at="not-a-real-date",
        )
        assert result.success is False
        assert "Couldn't read expires_at" in result.message
        assert "ISO-8601" in result.message
        assert db.memories.get("bad-expiry") is None

    @pytest.mark.asyncio
    async def test_transient_skill_resolve_embed_failure_is_actionable(self, tmp_path):
        """A fuzzy skill query whose embed fails transiently is refused with a retry —
        never a silent slide into NO_SKILL_FOUND (which would elicit teaching for a
        skill that might already exist)."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="fuzzy",
            intent="watch a peak",
            skill="something not an exact skill name",
            interval=3600,
        )
        assert result.success is False
        assert "Couldn't resolve the skill" in result.message
        assert "Retry" in result.message
        assert db.memories.get("fuzzy") is None

    @pytest.mark.asyncio
    async def test_on_advance_trigger_persists_source_log(self, tmp_path):
        """The on_advance trigger (#1604) names a source LOG; it persists on the row,
        the collection is paced at the tick (no cadence arg), and the echo reads the
        trigger back as ``on advance of <log>``."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        db.memories.create_log("events-log", "an event stream")
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="chained-watch",
            intent="digest events as they land",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            on_advance="events-log",
        )
        assert result.success and result.mutated
        assert result.message == _ON_ADVANCE_ECHO_LITERAL
        row = db.memories.get("chained-watch")
        assert row.source_log == "events-log"
        assert row.run_at is None and row.max_runs is None
        # Paced at the dispatcher tick when no min_interval is given.
        assert row.collector_interval_seconds == int(RuntimeParams().COLLECTOR_TICK_INTERVAL)

    @pytest.mark.asyncio
    async def test_on_advance_min_interval_sets_the_floor(self, tmp_path):
        """``min_interval`` becomes the collection's cadence floor (the throttle for a
        chatty source), while the trigger stays the source advance."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        db.memories.create_log("events-log", "an event stream")
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="throttled-watch",
            intent="digest events at most hourly",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            on_advance="events-log",
            min_interval=3600,
        )
        assert result.success
        row = db.memories.get("throttled-watch")
        assert row.source_log == "events-log"
        assert row.collector_interval_seconds == 3600

    @pytest.mark.asyncio
    async def test_on_advance_source_must_exist(self, tmp_path):
        """A source name that isn't a memory is refused with the fix (copy an exact log
        name), nothing created — a missing source would never advance."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="dangling",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            on_advance="no-such-log",
        )
        assert result.success is False
        assert "on_advance source 'no-such-log' isn't a memory" in result.message
        assert db.memories.get("dangling") is None

    @pytest.mark.asyncio
    async def test_on_advance_source_must_be_a_log_not_a_collection(self, tmp_path):
        """The frontier trigger fires on a LOG advancing; naming a collection is refused
        naming the shape mismatch — nothing created."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        _seed_collection(db, name="elevations")  # a collection, not a log
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="wrong-shape",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            on_advance="elevations",
        )
        assert result.success is False
        assert "is a collection, not a log" in result.message
        assert db.memories.get("wrong-shape") is None

    @pytest.mark.asyncio
    async def test_on_advance_with_interval_is_refused(self, tmp_path):
        """The trigger union is exclusive — on_advance alongside a recurring interval is
        refused, nothing created."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        db.memories.create_log("events-log", "an event stream")
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="two-forms",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
            on_advance="events-log",
        )
        assert result.success is False
        assert "Pick one trigger" in result.message
        assert db.memories.get("two-forms") is None

    @pytest.mark.asyncio
    async def test_min_interval_without_on_advance_is_refused(self, tmp_path):
        """``min_interval`` is only meaningful as an on_advance floor — passing it with
        a recurring interval is refused with the fix, nothing created."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="misused",
            intent="watch a peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
            min_interval=60,
        )
        assert result.success is False
        assert "min_interval only applies to an on_advance trigger" in result.message
        assert db.memories.get("misused") is None


class TestCreateAndList:
    @pytest.mark.asyncio
    async def test_create_log_persists(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="user-messages", description="inbound"
        )
        memories = {m.name: m for m in db.memories.list_all()}
        assert memories["user-messages"].type == "log"

    @pytest.mark.asyncio
    async def test_create_log_duplicate_returns_user_friendly_message(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="first"
        )
        result = await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="events", description="second"
        )
        assert "already exists" in result.message
        assert "events" in result.message

    @pytest.mark.asyncio
    async def test_update_rejects_short_extraction_prompt(self, tmp_path):
        db = _make_db(tmp_path)
        original_prompt = "test fixture extraction prompt"
        _seed_collection(db, name="notes", extraction_prompt=original_prompt)
        # The optional extraction_prompt rule on CollectionUpdateArgs validates
        # only when present, via the pre-execute Tool.run gate.
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).run(
            name="notes", extraction_prompt="yes"
        )
        assert result.success is False
        assert "extraction_prompt" in result.message
        assert "too short" in result.message
        # Update rejected — original prompt preserved unchanged
        assert db.memories.get("notes").extraction_prompt == original_prompt

    @pytest.mark.asyncio
    async def test_update_rejects_fictitious_tool_call(self, tmp_path):
        db = _make_db(tmp_path)
        original_prompt = (
            'Collect notes.\n1. browse(["x"])\n'
            '2. collection_write("notes", entries=[{key: "k", content: "c"}])\n3. done()'
        )
        _seed_collection(
            db,
            name="notes",
            description="x",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        # A rewrite that introduces a fictitious tool is rejected via the pre-execute
        # Tool.run gate, and the stored prompt is left untouched.
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).run(
            name="notes",
            extraction_prompt=(
                'Collect notes.\n1. browse(["x"])\n2. extract_text(page)\n'
                '3. collection_write("notes", entries=[{key: "k", content: "c"}])\n4. done()'
            ),
        )
        assert result.success is False
        assert "extract_text" in result.message
        assert db.memories.get("notes").extraction_prompt == original_prompt

    @pytest.mark.asyncio
    async def test_update_treats_blank_fields_as_omitted(self, tmp_path, mock_llm):
        # Models emit "" for an optional field they mean to leave alone (gpt-oss
        # was observed passing extraction_prompt="" alongside a recall change).
        # A blank must be skipped, not written through: the recall change lands
        # while the existing prompt/description survive untouched.
        db = _make_db(tmp_path)
        original_prompt = "test fixture extraction prompt that is long enough"
        _seed_collection(
            db,
            name="notes",
            description="real description",
            extraction_prompt=original_prompt,
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            extraction_prompt="",
            description="   ",
            notify=True,
        )
        assert "Updated" in result.message
        updated = db.memories.get("notes")
        assert updated.extraction_prompt == original_prompt  # blank skipped, not blanked
        assert updated.description == "real description"  # blank skipped, not blanked
        # notify flips on the update path (created silent by default → notify-on-new).
        assert updated.notify is True

    @pytest.mark.asyncio
    async def test_update_accepts_but_ignores_intent(self, tmp_path, mock_llm):
        # `intent` is serialized in the metadata the model reads, so it passes it back on an
        # edit.  Rather than reject the whole call over the immutable field (the model then
        # gave up), accept it, leave intent unchanged, and SAY SO in the result.
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="notes",
            description="real description",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="the original goal, set at creation",
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes",
            intent="a rewritten goal the model tried to set",
        )
        assert result.success  # accepted, not rejected over the immutable field
        assert "`intent` was not changed" in result.message  # visible + actionable
        updated = db.memories.get("notes")
        assert updated.intent == "the original goal, set at creation"  # intent untouched

    @pytest.mark.asyncio
    async def test_create_surfaces_description_embed_degradation(self, tmp_path):
        """A transient intent-embed failure still creates the collection, but the
        result NAMES the degraded routing anchor and leaves it NULL for the startup
        backfill to re-heal (#1468) — a visible degradation, not a silent success.
        (An exact-name skill match needs no resolution embed, so only the anchor
        embed fails.)"""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        result = await CollectionCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="notes",
            intent="a running list of notes",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
        )
        assert "Created" in result.message
        assert result.mutated is True
        assert "transient embedding error" in result.message
        assert "self-heal" in result.message
        # Row exists; anchor left NULL for the backfill to re-heal.
        row = db.memories.get("notes")
        assert row is not None
        assert row.description_embedding is None

    @pytest.mark.asyncio
    async def test_log_create_surfaces_description_embed_degradation(self, tmp_path):
        db = _make_db(tmp_path)
        result = await LogCreateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="events", description="event stream"
        )
        assert "Created log" in result.message
        assert "transient embedding error" in result.message
        assert db.memories.get("events").description_embedding is None

    @pytest.mark.asyncio
    async def test_update_failed_description_embed_clears_stale_anchor(self, tmp_path):
        """Changing a description whose embed fails clears the anchor to NULL — it does
        NOT leave the old, now-mismatched vector in place (a stale anchor the NULL-only
        description backfill could never detect, #1468).  The new text lands, the anchor
        is left for the backfill to re-heal, and the degradation surfaces."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="notes",
            description="old subject",
            extraction_prompt="test fixture extraction prompt that is long enough",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert db.memories.get("notes").description_embedding is not None  # a good anchor first

        result = await CollectionUpdateTool(db, cast(Any, _FailingEmbedClient())).execute(
            name="notes", description="a completely different subject"
        )
        assert "Updated" in result.message
        assert "transient embedding error" in result.message
        row = db.memories.get("notes")
        assert row.description == "a completely different subject"  # new text landed
        assert row.description_embedding is None  # stale anchor cleared, not kept


# ── Re-render fixtures (#1620) ────────────────────────────────────────────────
#
# The watch skill re-taught (its extract instruction changed) — a refresh renders a
# DIFFERENT prompt than the original, which is what the byte-identity acceptance
# check keys on.  ``_RIVER_SKILL`` is a distinct skill with a different hole, for the
# swap case.
_RIVER_SKILL = "Track river flow"
_RIVER_HOLE = "river"


def _watch_skill_steps_reteught() -> list[SkillStep]:
    """The watch skill re-taught: the same {peak} hole + step-1→2 binding, but the
    extract instruction is reworded, so a refresh re-renders to different text."""
    steps = _watch_skill_steps()
    steps[0].arguments["extract"] = "the summit elevation in metres"
    return steps


def _river_skill_steps() -> list[SkillStep]:
    """A distinct skill: a {river} hole in the browse query and write key, step 1's
    reading flowing into step 2 — the swap target."""
    return [
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [_RIVER_HOLE], "extract": "the current flow rate"},
            substitutions=[
                SkillSubstitution(path=["queries", 0], kind=SkillSubKind.HOLE, hole=_RIVER_HOLE)
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments={"memory": "flows", "entries": [{"key": _RIVER_HOLE, "content": "x"}]},
            substitutions=[
                SkillSubstitution(
                    path=["entries", 0, "key"], kind=SkillSubKind.HOLE, hole=_RIVER_HOLE
                ),
                SkillSubstitution(
                    path=["entries", 0, "content"], kind=SkillSubKind.BINDING, step=1
                ),
            ],
        ),
    ]


async def _create_watch_collection(db, *, name: str = "cinder-elevation") -> None:
    """Instantiate a collection from the watch skill through the real front door —
    the honest starting point for the refresh/rebind/swap cases (skill provenance
    stamped)."""
    _seed_watch_skill(db)
    result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
        name=name,
        intent="watch Cinder Peak's elevation",
        skill=_SKILL_NAME,
        params={"peak": "Cinder Peak"},
        interval=3600,
    )
    assert result.success


# The refresh echo — the SAME skill re-taught, re-rendered from its CURRENT steps with
# the CURRENT bindings; render-at-update mirrors the creation echo.
_REFRESH_ECHO_LITERAL = (
    "Re-rendered collection 'cinder-elevation' from skill 'Watch elevation':\n"
    "  intent: watch Cinder Peak's elevation\n"
    "  skill: Watch elevation\n"
    "  params: peak=Cinder Peak\n"
    "  trigger: every 1h\n"
    "  notify: False\n"
    "  expires: never\n"
    "  extraction_prompt: |\n"
    "    1. browse(queries=['Cinder Peak'], extract='the summit elevation in metres')\n"
    "    2. collection_write(memory='cinder-elevation', "
    "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
)

# The rebind echo — SAME skill (original steps), NEW params bound and re-rendered.
_REBIND_ECHO_LITERAL = (
    "Re-rendered collection 'cinder-elevation' from skill 'Watch elevation':\n"
    "  intent: watch Cinder Peak's elevation\n"
    "  skill: Watch elevation\n"
    "  params: peak=Ashfall Ridge\n"
    "  trigger: every 1h\n"
    "  notify: False\n"
    "  expires: never\n"
    "  extraction_prompt: |\n"
    "    1. browse(queries=['Ashfall Ridge'], extract='the elevation above sea level')\n"
    "    2. collection_write(memory='cinder-elevation', "
    "entries=[{'key': 'Ashfall Ridge', 'content': the value from step 1}])"
)

# The swap echo — a DIFFERENT skill rendered into the same collection, its write
# retargeted to that collection (#1629 — the river skill demoed against 'flows').
_SWAP_ECHO_LITERAL = (
    "Re-rendered collection 'cinder-elevation' from skill 'Track river flow':\n"
    "  intent: watch Cinder Peak's elevation\n"
    "  skill: Track river flow\n"
    "  params: river=Silt River\n"
    "  trigger: every 1h\n"
    "  notify: False\n"
    "  expires: never\n"
    "  extraction_prompt: |\n"
    "    1. browse(queries=['Silt River'], extract='the current flow rate')\n"
    "    2. collection_write(memory='cinder-elevation', "
    "entries=[{'key': 'Silt River', 'content': the value from step 1}])"
)

# The adopt echo — a legacy skill=NULL collection given a skill for the first time; its
# hand-authored text is replaced by the render, writes retargeted to it (#1629).
_ADOPT_ECHO_LITERAL = (
    "Re-rendered collection 'legacy-notes' from skill 'Watch elevation':\n"
    "  intent: a running list the user asked me to keep\n"
    "  skill: Watch elevation\n"
    "  params: peak=Cinder Peak\n"
    "  trigger: every 1h\n"
    "  notify: False\n"
    "  expires: never\n"
    "  extraction_prompt: |\n"
    "    1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
    "    2. collection_write(memory='legacy-notes', "
    "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
)


class TestCollectionUpdateReinstantiation:
    """Re-render a collection from a new/updated skill (#1620): refresh · rebind ·
    swap · adopt, plus the untouched-prompt invariant.  Every re-render re-stamps the
    skill provenance, records a mutation event with the run id, and echoes the newly
    rendered program (render-at-update mirrors render-at-creation)."""

    @pytest.mark.asyncio
    async def test_refresh_rerenders_from_the_reteught_skill_byte_identical(self, tmp_path):
        """Acceptance (#1620): create → re-teach (upsert REPLACES) → refresh → the
        stored prompt equals render(current skill, current params) byte-for-byte, the
        provenance is re-stamped, and the mutation event names the run + fields."""
        db = _make_db(tmp_path)
        await _create_watch_collection(db)
        # Re-teach the SAME skill with reworded steps (upsert replaces the row).
        _seed_watch_skill(db, steps=_watch_skill_steps_reteught())
        result = await CollectionUpdateTool(
            db, cast(Any, MockLlmClient()), run_id="run-refresh"
        ).execute(name="cinder-elevation", skill=_SKILL_NAME)
        assert result.success and result.mutated
        assert result.message == _REFRESH_ECHO_LITERAL
        # Byte-identity: the stored prompt IS the fresh render of the current skill,
        # with its writes retargeted to this collection's own name (#1629).
        current = db.skills.get(_SKILL_NAME)
        expected = render_skill(
            retarget_writes(steps_from_json(current.steps), "cinder-elevation"),
            {"peak": "Cinder Peak"},
        )
        stored = db.memories.get("cinder-elevation")
        assert stored.extraction_prompt == expected
        # Provenance re-stamped (same skill, same bindings).
        assert stored.skill_name == _SKILL_NAME
        assert stored.skill_params == json.dumps({"peak": "Cinder Peak"})
        # The re-render is recorded as a mutation with the run id + the changed fields.
        rerender = next(
            e for e in db.mutations.history("cinder-elevation", 5) if e.run_id == "run-refresh"
        )
        summary = mutation_change_summary(rerender)
        assert "skill" in summary and "extraction_prompt" in summary

    @pytest.mark.asyncio
    async def test_rebind_rerenders_with_new_params_same_skill(self, tmp_path):
        """New params, same skill: the prompt re-renders with the new bindings and the
        stored params advance, while the skill name stays put."""
        db = _make_db(tmp_path)
        await _create_watch_collection(db)
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation", params={"peak": "Ashfall Ridge"}
        )
        assert result.success and result.mutated
        assert result.message == _REBIND_ECHO_LITERAL
        stored = db.memories.get("cinder-elevation")
        assert stored.skill_name == _SKILL_NAME  # same skill
        assert stored.skill_params == json.dumps({"peak": "Ashfall Ridge"})  # rebound

    @pytest.mark.asyncio
    async def test_swap_rerenders_from_a_different_skill(self, tmp_path):
        """A different skill name renders that skill's steps into the collection and
        re-homes its provenance onto the new skill."""
        db = _make_db(tmp_path)
        await _create_watch_collection(db)
        _seed_watch_skill(
            db,
            name=_RIVER_SKILL,
            intent="track a river's flow",
            description="track a river's flow rate over time",
            steps=_river_skill_steps(),
            holes=[SkillHole(name=_RIVER_HOLE, required=True)],
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation", skill=_RIVER_SKILL, params={"river": "Silt River"}
        )
        assert result.success and result.mutated
        assert result.message == _SWAP_ECHO_LITERAL
        stored = db.memories.get("cinder-elevation")
        assert stored.skill_name == _RIVER_SKILL
        assert stored.skill_params == json.dumps({"river": "Silt River"})

    @pytest.mark.asyncio
    async def test_swap_without_binding_the_new_hole_is_refused_unchanged(self, tmp_path):
        """Swapping to a skill whose required hole the (reused) params don't bind is
        refused naming the hole — and nothing is mutated (prompt + provenance intact)."""
        db = _make_db(tmp_path)
        await _create_watch_collection(db)
        _seed_watch_skill(
            db,
            name=_RIVER_SKILL,
            intent="track a river's flow",
            description="track a river's flow rate over time",
            steps=_river_skill_steps(),
            holes=[SkillHole(name=_RIVER_HOLE, required=True)],
        )
        before = db.memories.get("cinder-elevation").extraction_prompt
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation",
            skill=_RIVER_SKILL,  # no params → 'river' unbound
        )
        assert result.success is False
        # The whole refusal names the missing hole + the params shape to supply.
        assert result.message == render_unbound_holes(_RIVER_SKILL, [_RIVER_HOLE])
        stored = db.memories.get("cinder-elevation")
        assert stored.extraction_prompt == before  # nothing rendered
        assert stored.skill_name == _SKILL_NAME  # provenance unchanged

    @pytest.mark.asyncio
    async def test_rebind_when_pinned_skill_is_gone_is_refused_unchanged(self, tmp_path):
        """A params-only rebind whose pinned skill has since been deleted can't
        re-render — refused actionably (re-teach it, or point at another skill via
        skill=), nothing mutated.  (A skill= arg on a missing skill routes to the
        NO_SKILL_FOUND elicitation instead; this is the current-skill branch.)"""
        db = _make_db(tmp_path)
        await _create_watch_collection(db)
        with Session(db.engine) as session:  # the pinned skill vanishes under it
            session.delete(session.get(Skill, _SKILL_NAME))
            session.commit()
        before = db.memories.get("cinder-elevation").extraction_prompt
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation",
            params={"peak": "Ashfall Ridge"},  # no skill= → reuse current
        )
        assert result.success is False
        assert result.message == _SKILL_GONE.format(skill=_SKILL_NAME)  # whole refusal
        stored = db.memories.get("cinder-elevation")
        assert stored.extraction_prompt == before  # unchanged
        assert stored.skill_name == _SKILL_NAME  # provenance intact

    @pytest.mark.asyncio
    async def test_adopt_replaces_hand_authored_text_and_stamps_provenance(self, tmp_path):
        """A legacy skill=NULL collection given a skill for the first time: its
        hand-authored prompt is replaced by the render and its provenance is stamped."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        hand_authored = "1. hand-written prose the user typed themselves, long enough to pass"
        _seed_collection(
            db,
            name="legacy-notes",
            extraction_prompt=hand_authored,
            intent="a running list the user asked me to keep",
            collector_interval_seconds=3600,
        )
        assert db.memories.get("legacy-notes").skill_name is None  # legacy: no skill
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="legacy-notes", skill=_SKILL_NAME, params={"peak": "Cinder Peak"}
        )
        assert result.success and result.mutated
        assert result.message == _ADOPT_ECHO_LITERAL
        stored = db.memories.get("legacy-notes")
        assert stored.extraction_prompt != hand_authored  # text replaced by the render
        assert stored.skill_name == _SKILL_NAME  # provenance stamped
        assert stored.skill_params == json.dumps({"peak": "Cinder Peak"})

    @pytest.mark.asyncio
    async def test_plain_update_never_touches_the_prompt_or_provenance(self, tmp_path):
        """The pinned invariant: a plain collection_update (no skill/params/prompt)
        changes only the metadata it names — the routine and skill provenance are
        untouched."""
        db = _make_db(tmp_path)
        original = "test fixture extraction prompt that is long enough"
        _seed_collection(db, name="notes", extraction_prompt=original, notify=False)
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="notes", notify=True
        )
        assert result.success
        stored = db.memories.get("notes")
        assert stored.extraction_prompt == original  # untouched
        assert stored.skill_name is None  # untouched
        assert stored.skill_params is None  # untouched
        assert stored.notify is True  # the named change landed

    @pytest.mark.asyncio
    async def test_extraction_prompt_alongside_skill_is_a_conflict(self, tmp_path):
        """A raw extraction_prompt AND skill/params in one call is refused — the render
        owns the prompt, so the two can't both win; nothing changes."""
        db = _make_db(tmp_path)
        await _create_watch_collection(db)
        before = db.memories.get("cinder-elevation").extraction_prompt
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation",
            skill=_SKILL_NAME,
            extraction_prompt="a full replacement body long enough to pass the length gate",
        )
        assert result.success is False
        assert result.message == _REINSTANTIATE_CONFLICT  # whole refusal
        assert db.memories.get("cinder-elevation").extraction_prompt == before

    @pytest.mark.asyncio
    async def test_rebind_on_a_skill_less_collection_is_refused(self, tmp_path):
        """params-only on a hand-authored (skill=NULL) collection has no holes to bind —
        refused, pointing at skill= to adopt one; the prompt is untouched."""
        db = _make_db(tmp_path)
        original = "test fixture extraction prompt that is long enough"
        _seed_collection(db, name="legacy", extraction_prompt=original)
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="legacy", params={"peak": "Cinder Peak"}
        )
        assert result.success is False
        assert result.message == _REBIND_NO_SKILL.format(name="legacy")  # whole refusal
        assert db.memories.get("legacy").extraction_prompt == original


# ── Inert collections + the two-step teach bootstrap (#1629) ──────────────────

# The skill-less creation echo — storage only, no job, and the honest next steps.
_INERT_ECHO_LITERAL = (
    "Set up collection 'deals-watch' — storage only, no job yet:\n"
    "  intent: track the trail-runner shoe deals\n"
    "  status: inert (no skill attached)\n"
    "It'll hold whatever gets written to it, but nothing runs against it until you give it "
    "a skill. Teach me the routine once, save it with skill_create, then attach it with "
    "collection_update(name='deals-watch', skill=<title>, interval=<seconds>) to make it do "
    "something."
)

# The demo run's utterance — the {peak} value ('Meridian Trail 3') appears verbatim, so it
# distills into a hole; the write target ('deals-watch') does not, so it stays a constant
# the render then RETARGETS to whatever collection the skill is applied to.
_BOOTSTRAP_UTTERANCE = "watch the Meridian Trail 3 shoe and save its price"
_BOOTSTRAP_PRICE = "$149"


def _log_demo_run(db, run_id: str, *, write_target: str) -> None:
    """Log one clean chat run (browse → collection_write into ``write_target``) as a
    single promptlog row skill_create can distill: the triggering user turn, the two
    batched tool calls in order (→ ordinals), each call's framed result, and its
    STRUCTURAL success stamp (certified-by-execution)."""
    browse_result = (
        f"You used `browse` and here's the result: (browse result)\nEXTRACTED: {_BOOTSTRAP_PRICE}"
    )
    calls = [
        (
            "browse",
            {"queries": ["Meridian Trail 3"], "extract": "the current price"},
            browse_result,
            True,
        ),
        (
            "collection_write",
            {
                "memory": write_target,
                "entries": [{"key": "Meridian Trail 3", "content": _BOOTSTRAP_PRICE}],
            },
            f"You saved an entry to {write_target}: (collection_write result)\n"
            f"Wrote 1 entry to '{write_target}': Meridian Trail 3.",
            True,
        ),
    ]
    tool_calls = []
    tool_turns = []
    for index, (name, args, result, success) in enumerate(calls, start=1):
        call_id = f"c{index}"
        tool_calls.append(
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}
        )
        tool_turns.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
                PennyConstants.TOOL_RESULT_SUCCESS_KEY: success,
            }
        )
    user_turn = {
        "role": "user",
        "content": f"live context{PennyConstants.SECTION_SEPARATOR}{_BOOTSTRAP_UTTERANCE}",
    }
    db.messages.log_prompt(
        model="m",
        messages=[user_turn, *tool_turns],
        response={"choices": [{"message": {"tool_calls": tool_calls}}]},
        run_id=run_id,
        agent_name=PennyConstants.CHAT_AGENT_NAME,
    )


class TestInertCollections:
    """Skill-less create yields an INERT storage collection (#1629): no
    extraction_prompt / cadence / notify, never dispatches, catalog-visible,
    idempotency still applies — and a job-shaped arg alongside is refused."""

    @pytest.mark.asyncio
    async def test_skill_less_create_is_inert_storage_only(self, tmp_path):
        """A create with no skill lands exactly one storage-only row — no
        extraction_prompt, cadence, notify, or skill provenance — and the echo is
        honest about being inert (whole render)."""
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", intent="track the trail-runner shoe deals"
        )
        assert result.success and result.mutated
        assert result.message == _INERT_ECHO_LITERAL
        row = db.memories.get("deals-watch")
        assert row is not None
        assert row.extraction_prompt is None  # no job
        assert row.collector_interval_seconds is None  # no cadence
        assert row.notify is False  # silent
        assert row.skill_name is None  # no skill attached
        assert row.archived is False  # a live, usable container

    @pytest.mark.asyncio
    async def test_inert_collection_is_catalog_visible(self, tmp_path):
        """An inert collection enumerates in the catalog, marked as storage with no
        routine — not hidden for lacking a prompt (#1629).  Whole render, so the inert
        recipe marker's position and the unchanged rest are both pinned."""
        db = _make_db(tmp_path)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", intent="track the trail-runner shoe deals"
        )
        row = db.memories.get("deals-watch")
        catalog = await CollectionCatalogTool(db).execute()
        assert catalog.message == (
            "## deals-watch\n"
            "status: active\n"
            "expires: never\n"
            f"created: {format_log_timestamp(row.created_at)}\n"
            "description: track the trail-runner shoe deals\n"
            "intent: track the trail-runner shoe deals\n"
            "notify: False\n"
            "extraction_prompt: (none — inert storage, no skill attached yet)"
        )

    @pytest.mark.asyncio
    async def test_job_arg_on_skill_less_create_is_refused(self, tmp_path):
        """A trigger / notify / expiry on a skill-less create has no job to attach to —
        refused naming the two-step fix (whole render), nothing created."""
        db = _make_db(tmp_path)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", intent="track the trail-runner shoe deals", interval=3600
        )
        assert result.success is False
        assert result.message == _INERT_JOB_ARGS_REFUSAL.format(name="deals-watch")
        assert db.memories.get("deals-watch") is None

    @pytest.mark.asyncio
    async def test_inert_create_still_respects_idempotency(self, tmp_path):
        """Idempotency-at-birth (#1567) still applies to an inert create — a
        near-duplicate of an existing collection is refused, nothing created."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="deals",
            description="track the trail-runner shoe deals",
            intent="track shoe deals",
        )
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", intent="track the trail-runner shoe deals"
        )
        assert result.success is False
        assert "Already have a collection for this" in result.message
        assert db.memories.get("deals-watch") is None


class TestWriteRetargetAtApply:
    """WRITE-RETARGET (#1629): applying a skill to a collection binds every
    scoped-write step's ``memory`` argument to that collection's own name — the
    demo target the skill baked in is overwritten at the render seam, so the
    rendered program never lies about where it writes."""

    @pytest.mark.asyncio
    async def test_skill_demoed_against_a_renders_writes_to_b_on_create(self, tmp_path):
        """A skill whose demo wrote into collection A ('elevations'), instantiated into
        collection B, renders its write to B — byte-pinned."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)  # its write step targets 'elevations' (the demo constant)
        result = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="target-b",
            intent="watch a different peak",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
        )
        assert result.success
        stored = db.memories.get("target-b")
        assert stored.extraction_prompt == (
            "1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
            "2. collection_write(memory='target-b', "
            "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
        )

    @pytest.mark.asyncio
    async def test_skill_demoed_against_a_renders_writes_to_b_on_adopt(self, tmp_path):
        """Adopting a skill (demoed against A) onto a legacy collection B retargets its
        write to B — byte-pinned, proving both apply paths retarget."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)  # write step targets 'elevations'
        _seed_collection(
            db,
            name="target-b",
            extraction_prompt="hand-authored prose long enough to pass the gate",
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="target-b", skill=_SKILL_NAME, params={"peak": "Cinder Peak"}
        )
        assert result.success
        stored = db.memories.get("target-b")
        assert stored.extraction_prompt == (
            "1. browse(queries=['Cinder Peak'], extract='the elevation above sea level')\n"
            "2. collection_write(memory='target-b', "
            "entries=[{'key': 'Cinder Peak', 'content': the value from step 1}])"
        )


class TestTwoStepTeachBootstrap:
    """The whole #1629 bootstrap end-to-end through real tool calls (mocked LLM):
    create inert → demonstrate a write into it → skill_create over that run → adopt
    the skill with a trigger + notify → the collection runs the rendered routine."""

    @pytest.mark.asyncio
    async def test_create_inert_teach_adopt_makes_it_run(self, tmp_path):
        db = _make_db(tmp_path)
        # 1. Set up the inert container (real tool call).
        created = await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", intent="track the trail-runner shoe deals"
        )
        assert created.success
        assert db.memories.get("deals-watch").extraction_prompt is None

        # 2. Demonstrate the routine once, writing INTO the inert collection (a logged run).
        _log_demo_run(db, "run-demo", write_target="deals-watch")

        # 3. Promote that run into a skill (real skill_create call over the demo run).
        taught = await SkillCreateTool(db, cast(Any, MockLlmClient()), author="chat").execute(
            name="Watch a shoe price", from_run="run-demo", steps="1-2"
        )
        assert taught.success
        assert db.skills.get("Watch a shoe price") is not None

        # 4. Adopt the skill onto the inert collection with a trigger + notify (real update).
        adopted = await CollectionUpdateTool(
            db, cast(Any, MockLlmClient()), run_id="run-adopt"
        ).execute(
            name="deals-watch",
            skill="Watch a shoe price",
            params={"queries": "Meridian Trail 3"},
            interval=3600,
            notify=True,
        )
        assert adopted.success and adopted.mutated

        # The stored prompt IS the retargeted render of the taught skill; the collection
        # now has a routine + cadence + notify, so it will dispatch (was inert before).
        skill = db.skills.get("Watch a shoe price")
        expected = render_skill(
            retarget_writes(steps_from_json(skill.steps), "deals-watch"),
            {"queries": "Meridian Trail 3"},
        )
        stored = db.memories.get("deals-watch")
        # Byte-identity: the stored prompt IS the retargeted render (write → deals-watch).
        assert stored.extraction_prompt == expected
        assert stored.skill_name == "Watch a shoe price"
        assert stored.collector_interval_seconds == 3600
        assert stored.notify is True


class TestCollectionUpdateTriggerAtApply:
    """Trigger + notify are apply-time properties on collection_update (#1629, the full
    union: interval | run_at+max_runs | on_advance + expires_at + notify) — recorded in
    the mutation event's changed fields."""

    @pytest.mark.asyncio
    async def test_adopt_applies_interval_notify_recorded_in_mutation(self, tmp_path):
        """Adopting a skill with interval + notify sets both and records them in the
        mutation event's changed fields alongside the re-render."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        _seed_collection(
            db,
            name="legacy",
            extraction_prompt="hand-authored prose long enough to pass",
            notify=False,
        )
        result = await CollectionUpdateTool(
            db, cast(Any, MockLlmClient()), run_id="run-adopt"
        ).execute(
            name="legacy",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=7200,
            notify=True,
        )
        assert result.success
        stored = db.memories.get("legacy")
        assert stored.collector_interval_seconds == 7200
        assert stored.notify is True
        event = next(e for e in db.mutations.history("legacy", 5) if e.run_id == "run-adopt")
        summary = mutation_change_summary(event)
        assert "trigger" in summary and "notify" in summary and "skill" in summary

    @pytest.mark.asyncio
    async def test_trigger_replaces_whole_schedule(self, tmp_path):
        """Setting a run_at+max_runs trigger on a recurring collection replaces the whole
        schedule (interval → dispatcher tick, run_at/max_runs set); a later interval
        trigger clears the once-shaped overlay (#1629)."""
        db = _make_db(tmp_path)
        _seed_collection(db, name="watch", collector_interval_seconds=3600)
        # Switch to a one-shot schedule.
        once = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="watch", run_at="2026-12-25T09:00:00Z", max_runs=1
        )
        assert once.success
        row = db.memories.get("watch")
        assert row.run_at is not None and row.max_runs == 1
        # Switch back to a recurring interval — the once overlay must clear.
        back = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="watch", interval=1800
        )
        assert back.success
        row = db.memories.get("watch")
        assert row.collector_interval_seconds == 1800
        assert row.run_at is None and row.max_runs is None

    @pytest.mark.asyncio
    async def test_on_advance_trigger_at_apply_validates_source(self, tmp_path):
        """An on_advance trigger at apply time sets the source_log; a non-existent
        source is refused (the shared validator), nothing changed."""
        db = _make_db(tmp_path)
        db.memories.create_log("events-log", "an event stream")
        _seed_collection(db, name="watch", collector_interval_seconds=3600)
        ok = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="watch", on_advance="events-log"
        )
        assert ok.success
        assert db.memories.get("watch").source_log == "events-log"
        # A missing source log is refused actionably.
        bad = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="watch", on_advance="no-such-log"
        )
        assert bad.success is False
        assert "isn't a memory I have" in bad.message

    @pytest.mark.asyncio
    async def test_adopt_without_trigger_warns_it_wont_run(self, tmp_path):
        """Adopting a skill onto an inert collection with NO trigger leaves it without a
        cadence — the echo carries a visible no-trigger note (#1629), not a silent
        won't-run."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", intent="track the trail-runner shoe deals"
        )
        result = await CollectionUpdateTool(db, cast(Any, MockLlmClient())).execute(
            name="deals-watch", skill=_SKILL_NAME, params={"peak": "Cinder Peak"}
        )
        assert result.success
        # The whole no-trigger note is appended verbatim at the tail of the echo.
        assert result.message.endswith(_NO_TRIGGER_NOTE.format(name="deals-watch"))
        assert db.memories.get("deals-watch").collector_interval_seconds is None


class TestCollectionWritesAndReads:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast", "content": "loves dark roast"},
                {"key": "cold brew", "content": "enjoys cold brew"},
            ],
        )
        assert "Wrote 2 entries to 'likes'" in result.message
        assert result.mutated is True
        latest = await CollectionReadLatestTool(db).execute(memory="likes")
        assert "dark roast" in latest.message
        assert "cold brew" in latest.message
        # Each rendered entry carries an absolute UTC timestamp so the model can
        # place it in time — read-tool output was previously timeless.
        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]", latest.message)

    @pytest.mark.asyncio
    async def test_write_empty_entries_is_actionable_not_bare_pydantic(self, tmp_path, mock_llm):
        """An empty ``entries`` batch gets a named, actionable rejection through the
        arg-validation envelope — not Pydantic's bare "List should have at least 1
        item" (the house wording-unification pass)."""
        db = _make_db(tmp_path)
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.run(memory="likes", entries=[])
        assert result.success is False
        assert "at least one entry" in result.message
        assert "List should have at least 1 item" not in result.message

    @pytest.mark.asyncio
    async def test_write_unknown_key_in_entry_names_nested_path(self, tmp_path, mock_llm):
        """A misspelled/extraneous key INSIDE a batch entry surfaces the nested loc
        path and suggests the valid sibling, rather than being silently dropped or
        mis-rendered as the whole ``entries`` field (#1416)."""
        db = _make_db(tmp_path)
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        result = await write.run(
            memory="likes",
            entries=[{"key": "k", "content": "v", "contnt": "typo"}],
        )
        assert result.success is False
        assert "entries.0.contnt" in result.message
        assert "did you mean 'content'" in result.message

    def test_format_duplicate_binds_key_when_present_and_is_honest_when_keyless(self):
        """The keyed arm binds an ``update_entry(key=...)`` call to the matched key;
        the keyless arm (no key to update) is honest about it and points at the real
        move (skip / write distinct content) — not a dangling refresh imperative."""
        keyed = _format_duplicate(
            WriteResult(
                key="cold brew", outcome=WriteGateOutcome.DUPLICATE, matched_key="cold brew"
            )
        )
        assert "update_entry(key='cold brew'" in keyed
        keyless = _format_duplicate(
            WriteResult(key="cold brew", outcome=WriteGateOutcome.DUPLICATE, matched_key=None)
        )
        assert "no key to update" in keyless
        assert "update_entry(" not in keyless

    @pytest.mark.asyncio
    async def test_write_reports_duplicate_via_tcr(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        result = await write.execute(
            memory="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        assert "Rejected as duplicates" in result.message
        # A fully duplicate-rejected batch wrote nothing — it must read as a
        # no-op so the collector's work/no-work split (and auto-throttle) sees
        # the truth rather than counting the rejected write as "work".
        assert result.mutated is False
        # The candidate's own key is named, the existing key it collided with is
        # named, *and* the matched key is BOUND straight into the update_entry call
        # (not a <existing key> placeholder) — so the model refreshes 'dark roast'
        # rather than re-using its own rejected 'dark roast coffee' key and
        # ping-ponging on key-not-found (#1405).
        assert "dark roast coffee" in result.message
        assert "duplicates existing 'dark roast'" in result.message
        assert "update_entry(key='dark roast', content=<richer info>)" in result.message
        # Whole batch was duplicates → the "nothing new" hint fires.  This is a
        # chat-scope write (scope=None), which has no ``done`` tool, so the hint
        # must NOT name it — chat and collector share this tool surface.
        assert "Nothing new to add this time" in result.message
        assert "done()" not in result.message

    @pytest.mark.asyncio
    async def test_write_all_duplicates_collector_scope_hints_done(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        # A collector binds its writes to one collection via ``scope``.
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test", scope="likes")
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        all_duplicates = await write.execute(
            memory="likes",
            entries=[{"key": "dark roast coffee", "content": "different body entirely"}],
        )
        # Whole batch was duplicates and this is a collector, so the close names
        # ``done()`` — the model can close the cycle instead of key-hunting — while
        # the per-entry rejection still binds the matched key into update_entry.
        assert "Nothing new to add" in all_duplicates.message
        assert "done()" in all_duplicates.message
        assert "update_entry(key='dark roast', content=<richer info>)" in all_duplicates.message
        # A re-write under the SAME key with the SAME value is the change-gate's
        # UNCHANGED outcome (#1587) — the watch's "no change" signal, reported as
        # such (not a generic "duplicate") AND carrying a STOP the collector loop
        # honors (this is a collector-scoped write).
        same_key = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        assert "Unchanged: 'dark roast' already holds the same value" in same_key.message
        assert same_key.mutated is False
        assert same_key.stop == WriteGateOutcome.KEY_EXISTS_UNCHANGED
        # A batch with a genuinely new entry alongside a duplicate gets the
        # per-entry bound refresh + the refresh-or-skip close, never "nothing new".
        partial = await write.execute(
            memory="likes",
            entries=[
                {"key": "cold brew", "content": "a brand new distinct entry"},
                {"key": "dark roast blend", "content": "first body"},
            ],
        )
        assert "Wrote 1 entry" in partial.message
        assert "Rejected as duplicates" in partial.message
        assert "update_entry(key='dark roast', content=<richer info>)" in partial.message
        assert "or skip these" in partial.message
        assert "Nothing new to add" not in partial.message
        assert partial.mutated is True
        # A batch whose entries each duplicate a DIFFERENT existing key must bind
        # EVERY matched key into its own update_entry call — not just the first
        # (#1405: resolve a match for every rejected key in the batch).  'cold brew'
        # now exists (written above), so both entries collide on distinct keys.
        multi = await write.execute(
            memory="likes",
            entries=[
                {"key": "dark roast blend", "content": "first body"},
                {"key": "cold brew coffee", "content": "a brand new distinct entry"},
            ],
        )
        assert "update_entry(key='dark roast', content=<richer info>)" in multi.message
        assert "update_entry(key='cold brew', content=<richer info>)" in multi.message
        assert multi.mutated is False

    @pytest.mark.asyncio
    async def test_chat_scope_unchanged_write_has_no_stop(self, tmp_path, mock_llm):
        """The chat surface (scope=None) gets the SAME enumerated UNCHANGED text but
        NEVER a loop-stop — STOP applies to must-act cadence contexts only (#1587).
        Contrast the collector-scope write above, which sets ``stop``."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        # No scope → chat surface.
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        unchanged = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "first body"}]
        )
        assert "Unchanged: 'dark roast' already holds the same value" in unchanged.message
        assert unchanged.stop is None

    @pytest.mark.asyncio
    async def test_change_gate_changed_auto_refreshes_baseline(self, tmp_path, mock_llm):
        """The CHANGED auto-refresh result text (#1633): re-writing an EXACT key with a
        DIFFERENT value refreshes the stored baseline IN PLACE through the shared write
        gate — the result reports the refresh with NO dangling ``update_entry``
        instruction (the wasted call that would teach the model to redo what already
        happened), ``mutated=True`` (durable state changed), and CHANGED is never
        STOP-worthy so no loop-stop even for a collector-scoped write.  The gate is
        shared, so a chat-surface write (``scope=None``) gets the same auto-refresh and
        stays non-STOP with honest text."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="watch",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="watch a page for a change",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test", scope="watch")
        await write.execute(memory="watch", entries=[{"key": "price", "content": "$42"}])
        changed = await write.execute(memory="watch", entries=[{"key": "price", "content": "$40"}])
        assert changed.message == (
            "Changed: 'price' — the stored baseline was refreshed to the new value (entry)."
        )
        assert changed.mutated is True
        assert changed.stop is None
        # The baseline was refreshed in place — one row, now the new value.
        refreshed = db.memory("watch").get("price")
        assert len(refreshed) == 1
        assert refreshed[0].content == "$40"
        # The gate is shared: a chat-surface write auto-refreshes identically, never STOPs.
        chat_write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="chat")
        await chat_write.execute(memory="watch", entries=[{"key": "note", "content": "v1"}])
        chat_changed = await chat_write.execute(
            memory="watch", entries=[{"key": "note", "content": "v2"}]
        )
        assert chat_changed.message == (
            "Changed: 'note' — the stored baseline was refreshed to the new value (entry)."
        )
        assert chat_changed.mutated is True
        assert chat_changed.stop is None
        assert db.memory("watch").get("note")[0].content == "v2"

    @pytest.mark.asyncio
    async def test_get_returns_entry_or_not_found(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "hello"}]
        )
        assert "hello" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message
        missing = await CollectionGetTool(db).execute(memory="likes", key="absent")
        assert "not found" in missing.message
        # The proven-win read guidance (collection_keys / read_similar; 47%→88%
        # recovery) is intact, AND the rejection now closes the residual write-vs-
        # update decision: once the model finds the entry under a different key it
        # must UPDATE that entry, not collection_write it (which the dedup rejects
        # as a duplicate — the ~1-call ping-pong this guidance removes).
        assert "collection_keys('likes')" in missing.message
        assert "read_similar(memory='likes', anchor=<what you're looking for>)" in missing.message
        assert "update_entry(key=<the key you found>, content=<the new content>)" in missing.message
        assert "creates NEW keys only" in missing.message
        # A bracket-wrapped key (the model's ingrained habit from the old `[key]`
        # display form) is never silently resolved — it's rejected with a teaching
        # error that names the mistake, the current key='...' render, and the bare
        # key ready to reuse.
        bracketed = await CollectionGetTool(db).execute(memory="likes", key="[k]")
        assert bracketed.success is False
        assert bracketed.message == (
            "Key '[k]' not found in 'likes'. The enclosing [brackets] are not part "
            "of the key — entry listings show keys as key='...' and the key is passed "
            "bare, without brackets. This entry's key is 'k'. Retry with key='k'."
        )
        # A bracket-wrapped key whose bare form doesn't exist either gets the
        # ordinary not-found error, not the bracket teaching rejection.
        double_miss = await CollectionGetTool(db).execute(memory="likes", key="[absent]")
        assert "Key '[absent]' not found" in double_miss.message
        assert "Retry with" not in double_miss.message
        # A key that genuinely contains brackets exact-matches with no rejection.
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "[lit]", "content": "bracket literal body"}]
        )
        literal = await CollectionGetTool(db).execute(memory="likes", key="[lit]")
        assert literal.success is True
        assert "bracket literal body" in literal.message

    @pytest.mark.asyncio
    async def test_keys_lists_unique_keys_in_order(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "first", "content": "1"}])
        await write.execute(memory="likes", entries=[{"key": "second", "content": "2"}])
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing.message == "- first\n- second"

    @pytest.mark.asyncio
    async def test_keys_empty_collection_names_source_not_bare_sentinel(self, tmp_path):
        """An empty collection's keys read names the source and marks absence (not an
        error), rather than the bare "(no keys)" sentinel (house wording pass)."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        listing = await CollectionKeysTool(db).execute(memory="likes")
        assert listing.message == "No keys in `likes` — the collection is empty (not an error)."

    @pytest.mark.asyncio
    async def test_read_random_returns_all_when_few(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="likes", entries=[{"key": "a", "content": "1"}])
        rendered = await CollectionReadRandomTool(db).execute(memory="likes", k=5)
        assert "key='a' 1" in rendered.message

    @pytest.mark.asyncio
    async def test_read_similar_uses_embedding(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "coffee", "content": "loves coffee"}]
        )
        # Anchor shares the "coffee" word with the entry — the bag-of-words
        # mock embedding gives meaningful cosine.
        rendered = await ReadSimilarTool(db, client).execute(memory="likes", anchor="coffee please")
        assert "coffee" in rendered.message

    @pytest.mark.asyncio
    async def test_read_similar_returns_populated_homogeneous_collection(self, tmp_path, mock_llm):
        """A populated but homogeneous collection (recipe-shaped entries that all
        cluster together, like the real ``skills`` collection) must return its
        entries for a fuzzy anchor — not "No entries" (#1565).  The old ambient
        cluster/centrality gate on the explicit search suppressed exactly this
        case, removing the model's fuzzy-recovery path when guessing a key."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="playbooks",
            description="reusable how-to recipes",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a set of reusable how-to recipes",
        )
        client = _make_llm_client(mock_llm)
        # Distinct keys + shared "recipe workflow step" stem: the entries cluster
        # tightly (high centrality) yet stay well under the dedup threshold.
        await CollectionWriteTool(db, client, author="test").execute(
            memory="playbooks",
            entries=[
                {"key": "morning-briefing", "content": "recipe workflow step sunrise breakfast"},
                {"key": "evening-recap", "content": "recipe workflow step sunset supper"},
                {"key": "weekly-digest", "content": "recipe workflow step calendar planner"},
                {"key": "topic-tracker", "content": "recipe workflow step magnet compass"},
            ],
        )
        # A vague anchor ("recipe reminder") that only weakly matches — the shape
        # of the model guessing at a recipe's identity.
        rendered = await ReadSimilarTool(db, client).execute(
            memory="playbooks", anchor="recipe reminder"
        )
        assert "No entries" not in rendered.message
        assert "morning-briefing" in rendered.message


class TestEmbedFailureRefusesWrite:
    """A transient embed failure at write time REFUSES the write — no vectorless
    (recall-invisible, dedup-weakening) entry is ever persisted (#1412).  The
    prior behaviour stored the entry without a vector and returned an optimistic
    success; now the tool fails with an actionable retry and nothing lands."""

    @staticmethod
    async def _make_relevant_collection(db) -> None:
        # A fully recall-eligible collection: proves the refusal is about the
        # missing vector, not the memory being excluded from recall.
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

    @pytest.mark.asyncio
    async def test_collection_write_refuses_when_embedding_fails(self, tmp_path):
        db = _make_db(tmp_path)
        await self._make_relevant_collection(db)
        write = CollectionWriteTool(db, cast(Any, _FailingEmbedClient()), author="test")
        result = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "loves dark roast"}]
        )
        # Fail-hard: actionable failure naming the transient cause and binding the
        # retry, and no work reported (so the collector throttle sees no-op).
        assert result.success is False
        assert result.mutated is False
        assert "transient embedding error" in result.message
        assert "'dark roast'" in result.message
        assert "collection_write(memory='likes'" in result.message
        # Nothing persisted — the invariant "every stored entry has a vector" holds.
        with Session(db.engine) as session:
            rows = session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == "likes")).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_collection_write_refuses_on_key_only_embed_failure(self, tmp_path):
        # Even when only the key embed fails (content vector fine), storing the
        # entry would leave it missing a vector — so the write is still refused
        # atomically and nothing lands.  (The backfill now also repairs a
        # key-null row, #1468, but the write path won't persist one to begin with.)
        db = _make_db(tmp_path)
        await self._make_relevant_collection(db)
        write = CollectionWriteTool(
            db, cast(Any, _KeyOnlyFailingEmbedClient("dark roast")), author="test"
        )
        result = await write.execute(
            memory="likes", entries=[{"key": "dark roast", "content": "loves dark roast"}]
        )
        assert result.success is False
        assert result.mutated is False
        assert "'dark roast'" in result.message
        with Session(db.engine) as session:
            rows = session.exec(select(MemoryEntry).where(MemoryEntry.memory_name == "likes")).all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_log_append_refuses_when_embedding_fails(self, tmp_path):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, cast(Any, _FailingEmbedClient()), author="test")
        result = await append.execute(memory="events", content="something happened")
        assert result.success is False
        assert result.mutated is False
        assert "transient embedding error" in result.message
        assert "log_append(memory='events'" in result.message
        with Session(db.engine) as session:
            rows = session.exec(
                select(MemoryEntry).where(MemoryEntry.memory_name == "events")
            ).all()
        assert rows == []


class TestCollectionMutations:
    @pytest.mark.asyncio
    async def test_update_replaces_content(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="likes", entries=[{"key": "k", "content": "old"}]
        )
        result = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "Updated 'k' in 'likes'" in result.message
        fetched = await CollectionGetTool(db).execute(memory="likes", key="k")
        assert "new" in fetched.message
        # A bracket-wrapped key (display form copied from an entry list) is
        # rejected with a teaching error naming the bare key — never absorbed;
        # the entry is untouched.
        bracketed = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="[k]", content="newer"
        )
        assert bracketed.success is False
        assert "Retry with key='k'" in bracketed.message
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message
        # Same teaching rejection on delete: nothing removed, bare key named.
        rejected_delete = await CollectionDeleteEntryTool(db).execute(memory="likes", key="[k]")
        assert rejected_delete.success is False
        assert "Retry with key='k'" in rejected_delete.message
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message
        # A blank replacement is refused (same content bar as collection_write),
        # leaving the existing content untouched rather than blanking the entry.
        # The degenerate-content rule now lives on UpdateEntryArgs.content, so the
        # refusal is produced by the pre-execute Tool.run gate.
        blank = await UpdateEntryTool(db, author="test").run(memory="likes", key="k", content="   ")
        assert blank.success is False
        assert "no word tokens" in blank.message  # what went wrong
        assert "collection_delete_entry" in blank.message  # how to correct it
        assert "new" in (await CollectionGetTool(db).execute(memory="likes", key="k")).message

    @pytest.mark.asyncio
    async def test_update_missing_reports_not_found(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="k", content="new"
        )
        assert "not found" in result.message
        # A bracket-wrapped key whose bare form doesn't exist either gets the
        # ordinary not-found error, not the bracket teaching rejection.
        bracketed = await UpdateEntryTool(db, author="test").execute(
            memory="likes", key="[k]", content="new"
        )
        assert "Key '[k]' not found" in bracketed.message
        assert "Retry with" not in bracketed.message

    @pytest.mark.asyncio
    async def test_archive_and_unarchive(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        assert (
            "Archived 'likes'" in (await CollectionArchiveTool(db).execute(memory="likes")).message
        )
        assert (
            "Unarchived 'likes'"
            in (await CollectionUnarchiveTool(db).execute(memory="likes")).message
        )


class TestLogTools:
    @pytest.mark.asyncio
    async def test_collection_read_latest_refuses_a_log(self, tmp_path, mock_llm):
        """Collection reads error on a log instead of silently bypassing the
        cursored log_read/log_get interface (the read_latest-on-a-log footgun)."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="first"
        )
        rendered = await CollectionReadLatestTool(db).execute(memory="events")
        assert "Refused" in rendered.message
        assert "log_read" in rendered.message

    @pytest.mark.asyncio
    async def test_read_latest_rejects_zero_count(self, tmp_path, mock_llm):
        """``k=0`` (a model guessing zero means "unlimited") reads no entries, so
        the tool would look empty — the arg model refuses it before execute with
        an actionable message (omit k for all), via the Tool.run gate."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="notes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="notes", entries=[{"key": "a", "content": "first"}]
        )
        rejected = await CollectionReadLatestTool(db).run(memory="notes", k=0)
        assert rejected.success is False
        assert "at least 1" in rejected.message
        assert "Omit k" in rejected.message
        # A valid read still returns the entry (the guard only rejects k < 1).
        ok = await CollectionReadLatestTool(db).run(memory="notes")
        assert "first" in ok.message

    @pytest.mark.asyncio
    async def test_log_read_window_mode(self, tmp_path, mock_llm):
        """A non-collector caller (scope=None) gets window-mode log_read: recent
        entries within the fixed look-back window — no cursor, no count arg."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="hello")
        rendered = await LogReadTool(db, "chat", scope=None).execute(memory="events")
        assert "hello" in rendered.message
        # A blank append is refused (blank-only — a bare URL is still a valid log
        # entry), so nothing degenerate joins the stream.  The non-blank rule lives
        # on LogAppendArgs.content, so the refusal comes from the Tool.run gate.
        blank = await append.run(memory="events", content="   ")
        assert blank.success is False
        assert "blank" in blank.message

    @pytest.mark.asyncio
    async def test_collector_runs_log_renders_runs_from_promptlog(self, tmp_path):
        """collector-runs is a read facade over promptlog: log_read renders each
        worked run as a record (``[target] summary`` + its tool trace) — no
        stored entries, no keys, no get.  This is the quality collector's review."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="collector-runs", description="audit"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c0",
                                "type": "function",
                                "function": {
                                    "name": "send_message",
                                    "arguments": '{"content": "Found a new grinder, $300."}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=response,
            agent_name="collector",
            run_id="run-42",
            run_target="espresso-gear",
        )
        db.messages.set_run_outcome("run-42", "worked", "sent an update about a grinder")

        rendered = await LogReadTool(db, "quality", scope="quality").execute(
            memory="collector-runs"
        )

        # collector-runs reads through the uniform log formatter now (it's a log
        # facade like any other) — framed as a fetched batch, runs as records.
        assert "from `collector-runs`" in rendered.message
        assert "[espresso-gear] sent an update about a grinder" in rendered.message
        assert "Found a new grinder, $300." in rendered.message  # the exact message, untruncated

    @pytest.mark.asyncio
    async def test_read_run_calls_renders_by_target(self, tmp_path):
        """read_run_calls is the SEQUENCE lens over runs, orthogonal to target: a
        collector's name renders its runs as ``[target] -> tools -> done``; ``chat``
        renders conversations as ``user -> tools -> penny``.  The valid targets are
        discovered from the DB into its description."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="espresso-gear",
            description="x",
            extraction_prompt="1. browse for new espresso gear. 2. write it. 3. done().",
            collector_interval_seconds=3600,
            intent="track espresso gear",
        )
        # One completed collector run for espresso-gear.
        coll_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "collection_write",
                                    "arguments": '{"memory": "espresso-gear", '
                                    '"entries": [{"content": "Niche grinder"}]}',
                                }
                            },
                            # A worked run closes with the argless done() (#1569) —
                            # without it the run would (correctly) read as a
                            # write-gate STOP (#1587).
                            {
                                "function": {
                                    "name": "done",
                                    "arguments": "{}",
                                }
                            },
                        ]
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=coll_resp,
            agent_name="collector",
            run_id="coll-1",
            run_target="espresso-gear",
        )
        db.messages.set_run_outcome("coll-1", "worked", "wrote a new grinder")
        # One chat run: user message + a tool call, then Penny's reply.
        user_turn = f"live{PennyConstants.SECTION_SEPARATOR}find me a grinder"
        db.messages.log_prompt(
            model="m",
            messages=[{"role": "user", "content": user_turn}],
            response={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "browse",
                                        "arguments": '{"queries": ["espresso grinder"]}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            agent_name=PennyConstants.CHAT_AGENT_NAME,
            run_id="chat-1",
        )
        db.messages.log_prompt(
            model="m",
            messages=[],
            response={"choices": [{"message": {"content": "Here's a good grinder."}}]},
            agent_name=PennyConstants.CHAT_AGENT_NAME,
            run_id="chat-1",
        )

        tool = ReadRunCallsTool(db, "quality")
        # Targets discovered from the DB (chat + the collector) are in the description.
        assert "espresso-gear" in tool.description
        assert "chat" in tool.description

        collector = await tool.run(target="espresso-gear")
        assert "[espresso-gear]" in collector.message
        assert "collection_write(memory='espresso-gear'" in collector.message
        # Argless done() → the conclusion is the run's STRUCTURAL outcome (#1569),
        # never a model summary.
        assert "done: worked" in collector.message

        chat = await tool.run(target="chat")
        assert "user: find me a grinder" in chat.message
        assert "browse(['espresso grinder'])" in chat.message
        assert "penny: Here's a good grinder." in chat.message
        # Each rendered run names its own id, so the surface is an anchor: a reader
        # can reference the run it's inspecting rather than guess it (#1560).
        assert "run chat-1" in chat.message

        # An unknown/typo'd target resolves to a failed, actionable refusal that
        # names the offending value — not a silent empty batch that reads as "this
        # collector has no runs".
        unknown = await tool.run(target="esspreso-gear")
        assert unknown.success is False
        assert "esspreso-gear" in unknown.message

    @staticmethod
    def _log_run(db, *, run_id: str, target: str, summary: str, write_key: str) -> None:
        """Persist one completed collector run for ``target`` (a write + its
        ``done`` summary) — the promptlog rows ``get_event`` / ``read_run_calls``
        render.  ``response`` is a dict (``log_prompt`` serializes it); the inner
        tool-call ``arguments`` is itself a JSON string, as the model emits it."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "w0",
                                "type": "function",
                                "function": {
                                    "name": "collection_write",
                                    "arguments": json.dumps(
                                        {
                                            "memory": target,
                                            "entries": [{"key": write_key, "content": "v"}],
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        db.messages.log_prompt(
            model="m",
            messages=[],
            response=response,
            agent_name="collector",
            run_id=run_id,
            run_target=target,
        )
        db.messages.set_run_outcome(run_id, "worked", summary)

    @staticmethod
    async def _create_collection(db, name: str) -> None:
        _seed_collection(
            db,
            name=name,
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

    @pytest.mark.asyncio
    async def test_get_event_resolves_a_run_id_to_its_canonical_projection(self, tmp_path):
        """``get_event(event_id='run <id>')`` consumes the header's typed run anchor
        VERBATIM and returns exactly the run's canonical tool-call projection — the
        same ``render_run_calls`` view ``read_run_calls`` renders, but for the single
        run the id names (#1580, the run-id ↔ target anchor unification).  The result
        leads with the run's own id, so the rendered anchor and the argument are one."""
        db = _make_db(tmp_path)
        await self._create_collection(db, "ai-news")
        self._log_run(db, run_id="news-1", target="ai-news", summary="wrote 1", write_key="mixtral")

        result = await GetEventTool(db).run(event_id="run news-1")

        # Whole-render: the tool returns the canonical projection byte-for-byte.
        expected = render_run_calls(db.messages.get_run_prompts("news-1"))
        assert result.message == expected
        assert result.message.startswith("run news-1")
        assert "collection_write(memory='ai-news'" in result.message

    @pytest.mark.asyncio
    async def test_get_event_tolerates_the_paren_framed_mutation_anchor(self, tmp_path):
        """The mutation activity line renders its causing run as ``(run <id>)``; the
        parse strips the framing so BOTH rendered forms resolve to the same run — the
        rendered token is consumable verbatim whichever line it came from (#1580)."""
        db = _make_db(tmp_path)
        await self._create_collection(db, "ai-news")
        self._log_run(db, run_id="news-1", target="ai-news", summary="wrote 1", write_key="mixtral")

        bare = await GetEventTool(db).run(event_id="run news-1")
        framed = await GetEventTool(db).run(event_id="(run news-1)")
        assert framed.message == bare.message
        assert framed.success is True

    @pytest.mark.asyncio
    async def test_get_event_untyped_id_is_actionable(self, tmp_path):
        """An id with no recognised type tag is refused, not silently emptied — the
        message names what IS addressable and the guess-free fallbacks (find_mine +
        the activity block), never a bare 'not found' (#1580)."""
        db = _make_db(tmp_path)
        result = await GetEventTool(db).run(event_id="news-1")
        assert result.success is False
        assert "news-1" in result.message
        assert "find_mine" in result.message

    @pytest.mark.asyncio
    async def test_get_event_unknown_run_is_actionable(self, tmp_path):
        """A well-formed run token that matched no recorded run gets a failed,
        actionable refusal naming the id + where valid ids are listed — never an
        empty read that reads as a clean, call-less run (#1580)."""
        db = _make_db(tmp_path)
        result = await GetEventTool(db).run(event_id="run does-not-exist")
        assert result.success is False
        assert "does-not-exist" in result.message
        assert "read_run_calls" in result.message

    @pytest.mark.asyncio
    async def test_append_to_system_log_is_refused(self, tmp_path, mock_llm):
        """Invariant #1: the four framework-managed system logs are written
        only by Python side-effects.  ``log_append`` from any agent gets a
        readable refusal and writes nothing — guarding the conversation-turn
        reconstruction and the run audit trail from model-authored entries."""
        db = _make_db(tmp_path)
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        # The reserved-target check is a pure constant lookup, so it lives on
        # LogAppendArgs.memory and the refusal comes from the Tool.run gate.
        for system_log in PennyConstants.SYSTEM_LOGS:
            result = await append.run(memory=system_log, content="forged turn")
            assert result.success is False
            assert system_log in result.message
            assert "system log" in result.message  # what went wrong + how to fix
        # Nothing was created/written — the refusal short-circuits before the store.
        assert db.memories.get(PennyConstants.MEMORY_PENNY_MESSAGES_LOG) is None

    @pytest.mark.asyncio
    async def test_log_similar_with_client(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        client = _make_llm_client(mock_llm)
        await LogAppendTool(db, client, author="test").execute(
            memory="events", content="coffee is great"
        )
        # Anchor shares words with the entry so the bag-of-words mock
        # embedding gives meaningful cosine and the entry ranks in ``read_similar``.
        rendered = await ReadSimilarTool(db, client).execute(
            memory="events", anchor="coffee morning"
        )
        assert "coffee is great" in rendered.message

    @pytest.mark.asyncio
    async def test_read_next_returns_all_entries_when_no_cursor(self, tmp_path, mock_llm):
        """Without a stored cursor, read_next returns every entry in the log."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await read_next.execute(memory="events")

        assert "first" in rendered.message
        assert "second" in rendered.message

    @pytest.mark.asyncio
    async def test_commit_pending_advances_cursor_to_max_seen(self, tmp_path, mock_llm):
        """commit_pending writes the highest timestamp seen during the run."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")
        await append.execute(memory="events", content="second")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # A new instance after commit should see no entries (cursor caught up).
        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(memory="events")
        # Empty read names the source and marks it as absence, not an error.
        assert (
            rendered.message
            == "No entries in `events` — it's empty or nothing matched (not an error)."
        )

    @pytest.mark.asyncio
    async def test_discard_pending_leaves_cursor_unchanged(self, tmp_path, mock_llm):
        """discard_pending drops the in-memory state without touching the DB cursor."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        await append.execute(memory="events", content="first")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(memory="events")
        read_next.discard_pending()

        # Cursor still at None; a new read sees the same entries.
        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(memory="events")
        assert "first" in rendered.message

    @pytest.mark.asyncio
    async def test_first_cycle_bounded_to_latest_n_entries(self, tmp_path, mock_llm):
        """A brand-new collector (no cursor yet) reading a busy log gets the
        most-recent N entries, not every entry since the dawn of time.

        Without this bound, a fresh collector reading ``user-messages`` (which
        has months of chat history in production) would dump the entire log
        into the first cycle's context.
        """
        from penny.constants import PennyConstants

        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        # Append more entries than the bound to confirm trimming
        n_entries = PennyConstants.LOG_READ_LIMIT + 5
        for i in range(n_entries):
            await append.execute(memory="events", content=f"entry-{i:02d}")

        read_next = LogReadTool(db, agent_name="brand-new-collector", scope="brand-new-collector")
        rendered = await read_next.execute(memory="events")

        # Exactly the latest N entries — entry-(n-N) through entry-(n-1)
        # should appear; older entries should not.
        for i in range(n_entries - PennyConstants.LOG_READ_LIMIT, n_entries):
            assert f"entry-{i:02d}" in rendered.message
        # The first 5 entries must be excluded
        assert "entry-00" not in rendered.message
        assert "entry-04" not in rendered.message

    @pytest.mark.asyncio
    async def test_first_cycle_advances_cursor_so_next_cycle_sees_only_new(
        self, tmp_path, mock_llm
    ):
        """After a bounded first cycle commits, the next cycle picks up
        incrementally — even entries that the first cycle's bound excluded
        stay excluded (since they're older than the cursor)."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")
        for i in range(15):
            await append.execute(memory="events", content=f"old-{i:02d}")

        read_next = LogReadTool(db, agent_name="extractor", scope="extractor")
        await read_next.execute(memory="events")
        read_next.commit_pending()

        # New entries arrive
        await append.execute(memory="events", content="new-after-cursor")

        fresh = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered = await fresh.execute(memory="events")
        assert "new-after-cursor" in rendered.message
        # Old entries excluded by the bound stay excluded
        assert "old-00" not in rendered.message

    @pytest.mark.asyncio
    async def test_cursor_read_is_capped_and_advances_by_batch(self, tmp_path, mock_llm):
        """With a cursor established, a backlog larger than the batch bound is
        returned in bounded chunks — read N, cursor advances by N, the next read
        picks up the next N.  The caller never reasons about a count."""
        from penny.constants import PennyConstants

        limit = PennyConstants.LOG_READ_LIMIT
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        append = LogAppendTool(db, _make_llm_client(mock_llm), author="test")

        # Establish a cursor, then pile up a backlog bigger than one batch.
        await append.execute(memory="events", content="seed")
        seed_read = LogReadTool(db, agent_name="extractor", scope="extractor")
        await seed_read.execute(memory="events")
        seed_read.commit_pending()

        backlog = limit + 3
        for i in range(backlog):
            await append.execute(memory="events", content=f"backlog-{i:02d}")

        first = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered_first = await first.execute(memory="events")
        first.commit_pending()
        # Exactly one batch — the oldest N of the backlog, not all of it.
        assert rendered_first.message.count("backlog-") == limit
        assert "backlog-00" in rendered_first.message
        assert f"backlog-{backlog - 1:02d}" not in rendered_first.message

        # The next read picks up the remainder since the advanced cursor.
        second = LogReadTool(db, agent_name="extractor", scope="extractor")
        rendered_second = await second.execute(memory="events")
        assert f"backlog-{backlog - 1:02d}" in rendered_second.message

    @pytest.mark.asyncio
    async def test_per_agent_cursors_are_independent(self, tmp_path, mock_llm):
        """Two agents reading the same log have independent cursor state."""
        db = _make_db(tmp_path)
        await LogCreateTool(db, cast(Any, MockLlmClient())).execute(name="events", description="x")
        await LogAppendTool(db, _make_llm_client(mock_llm), author="test").execute(
            memory="events", content="hello"
        )

        agent_a = LogReadTool(db, agent_name="a", scope="a")
        await agent_a.execute(memory="events")
        agent_a.commit_pending()

        # Agent B has its own cursor and still sees the entry.
        agent_b = LogReadTool(db, agent_name="b", scope="b")
        rendered = await agent_b.execute(memory="events")
        assert "hello" in rendered.message


class TestEmissionProvenanceRender:
    """The messagelog facade renders emission provenance inline (#1568).

    An outgoing row stamped with the mechanism that sent it reads
    ``(sent by <mechanism>) <content>`` between the timestamp and content on
    every read path (all facade reads synthesize entries through ``_to_entry``)
    — so pulling a message by recency or relevance IS resolving its source.  A
    NULL-mechanism row (a direct reply) renders byte-identical to the
    pre-provenance shape — the unmarked case stays the quiet default."""

    def _seed(self, db: Database) -> None:
        db.memories.create_log(
            PennyConstants.MEMORY_PENNY_MESSAGES_LOG,
            "outbound messages",
        )
        with Session(db.engine) as session:
            session.add(
                MessageLog(
                    direction=PennyConstants.MessageDirection.OUTGOING,
                    sender="penny",
                    content="sure, happy to help!",
                    embedding=serialize_embedding([0.0, 1.0, 0.0]),
                    timestamp=datetime(2026, 7, 2, 9, 10, tzinfo=UTC),
                )
            )
            session.add(
                MessageLog(
                    direction=PennyConstants.MessageDirection.OUTGOING,
                    sender="penny",
                    content="Heads up: the price dropped to $42!",
                    mechanism="price-watch",
                    embedding=serialize_embedding([1.0, 0.0, 0.0]),
                    timestamp=datetime(2026, 7, 2, 9, 14, tzinfo=UTC),
                )
            )
            session.commit()

    @pytest.mark.asyncio
    async def test_log_read_renders_provenance_whole(self, tmp_path):
        """The whole log_read render, both shapes pinned: the direct reply line is
        byte-identical to the pre-provenance shape (``N. [stamp] content``); the
        mechanism-stamped line carries the inline marker."""
        db = _make_db(tmp_path)
        self._seed(db)
        result = await LogReadTool(db, agent_name="reader", scope="reader").execute(
            memory=PennyConstants.MEMORY_PENNY_MESSAGES_LOG
        )
        assert result.message == (
            "2 entries from `penny-messages` (oldest first):\n"
            "1. [2026-07-02 09:10 UTC] sure, happy to help!\n"
            "2. [2026-07-02 09:14 UTC] (sent by price-watch) Heads up: the price dropped to $42!"
        )

    def test_read_similar_carries_provenance_too(self, tmp_path):
        """The similarity path returns the same synthesized entries, so a
        relevance hit on an autonomous send carries its mechanism inline — one
        call resolves both the message and its source."""
        db = _make_db(tmp_path)
        self._seed(db)
        facade = db.memory(PennyConstants.MEMORY_PENNY_MESSAGES_LOG)
        assert facade is not None
        hits = facade.read_similar([1.0, 0.0, 0.0], k=1)
        assert [hit.content for hit in hits] == [
            "(sent by price-watch) Heads up: the price dropped to $42!"
        ]


class TestExistsAndDone:
    @pytest.mark.asyncio
    async def test_exists_yes_via_exact_key(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)
        await CollectionWriteTool(db, client, author="test").execute(
            memory="likes", entries=[{"key": "dark roast", "content": "body"}]
        )
        result = await ExistsTool(db, client).execute(
            memories=["likes"], key="dark roast", content="body"
        )
        assert result.message == "yes"

    @pytest.mark.asyncio
    async def test_exists_no(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await ExistsTool(db, _make_llm_client(mock_llm)).execute(
            memories=["likes"], key="not there", content="nothing"
        )
        assert result.message == "no"

    @pytest.mark.asyncio
    async def test_exists_unknown_memory_name_is_not_found(self, tmp_path, mock_llm):
        """A misspelled memory name must not read as an empty (always-"no")
        memory — that green-lights the write the model was probing for.  The
        probe fails with the actionable not-found refusal naming the bad value."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        result = await ExistsTool(db, _make_llm_client(mock_llm)).execute(
            memories=["lieks"], content="dark roast"
        )
        assert result.success is False
        assert "lieks" in result.message
        assert "not found" in result.message
        # The wrong-name miss names find_mine as the guess-free recovery (#1558).
        assert "find_mine(query=" in result.message

    @pytest.mark.asyncio
    async def test_exists_empty_memories_is_actionable_not_bare_pydantic(self, tmp_path, mock_llm):
        """An empty ``memories`` list gets a named, actionable rejection through the
        arg-validation envelope — not Pydantic's bare "List should have at least 1
        item" (the house wording-unification pass)."""
        db = _make_db(tmp_path)
        result = await ExistsTool(db, _make_llm_client(mock_llm)).run(memories=[], content="x")
        assert result.success is False
        assert "at least one collection name" in result.message
        assert "List should have at least 1 item" not in result.message

    @pytest.mark.asyncio
    async def test_exists_embed_failure_is_inconclusive_not_no(self, tmp_path, mock_llm):
        """When the embed service is down the similarity dedup is skipped, so a
        "no" would be a silent degradation that could green-light a near-duplicate
        write.  The probe surfaces the inconclusive state instead (visible
        degradation)."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)

        def _fail_embed(model: str, text: str | list[str]) -> list[list[float]]:
            raise LlmConnectionError("embedding service unavailable")

        mock_llm.set_embed_handler(_fail_embed)
        result = await ExistsTool(db, client).execute(memories=["likes"], content="nothing")

        assert result.message != "no"
        assert "inconclusive" in result.message

    @pytest.mark.asyncio
    async def test_unicode_hyphen_in_memory_name_normalized(self, tmp_path, mock_llm):
        """Regression: gpt-oss occasionally emits Unicode dashes (U+2010,
        U+2011, …) where ASCII hyphen-minus is expected, breaking string
        comparison in tool args.  Memory-name fields normalise on the way
        in so the rest of the stack sees the canonical form."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="board-games",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        # Non-breaking hyphen U+2011 in the memory name — model output
        # observed in the wild.
        result = await write.execute(
            memory="board‑games",
            entries=[{"key": "k", "content": "v"}],
        )
        assert "Wrote 1 entry to 'board-games'" in result.message

    @pytest.mark.asyncio
    async def test_exists_content_only_uses_content_as_key_probe(self, tmp_path, mock_llm):
        """Regression: ``exists(content="Catan")`` must catch an
        existing entry with ``key="Catan"``, even when the
        existing row's *content* is a long description that doesn't
        cosine-match the short candidate.  The tool now copies content
        into the key slot when the model omits it, letting key-TCR fire
        in the dedup rule."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="board-games",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        client = _make_llm_client(mock_llm)
        # Existing entry: short key, long descriptive content.
        await CollectionWriteTool(db, client, author="test").execute(
            memory="board-games",
            entries=[
                {
                    "key": "Catan",
                    "content": (
                        "Catan – A gateway strategy board game of trading and "
                        "settlement, designed by Klaus Teuber, first published "
                        "1995, widely credited with popularising modern hobby "
                        "board gaming."
                    ),
                }
            ],
        )
        # Probe with content only — what the collector usually does when
        # checking a candidate name before writing.
        result = await ExistsTool(db, client).execute(memories=["board-games"], content="Catan")
        assert result.message == "yes"

    @pytest.mark.asyncio
    async def test_done_is_argless_sentinel(self):
        """done() is argless (#1569): it just marks the cycle finished and returns a
        fixed marker.  The run record is generated from the ledger, so there is no
        model-authored success/summary to report."""
        result = await DoneTool().execute()
        assert result.message == "Cycle complete."
        assert result.success is True
        assert DoneTool.args_model.__name__ == "NoArgs"
        assert DoneTool.parameters == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_done_rejects_arguments(self):
        """Argless via NoArgs (extra='forbid'): any argument passed through the
        validation gate (``Tool.run``) is rejected as an actionable error, not
        silently dropped."""
        result = await DoneTool().run(success=True, summary="x")
        assert result.success is False


class TestAuthorAttribution:
    @pytest.mark.asyncio
    async def test_writes_stamp_constructor_author(self, tmp_path, mock_llm):
        """Author is bound at tool construction (not pulled from ambient state)."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        await CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="preference-extractor"
        ).execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        rows = db.memory("likes").get("k")
        assert rows[0].author == "preference-extractor"


class TestCollectionMerge:
    @pytest.mark.asyncio
    async def test_merge_moves_entries_and_archives_source(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="src",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        _seed_collection(
            db,
            name="dst",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="src", entries=[{"key": "a", "content": "alpha"}])
        await write.execute(memory="src", entries=[{"key": "b", "content": "beta"}])

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "2 moved" in result.message
        assert "archived" in result.message
        assert db.memories.get("src").archived is True
        assert len(db.memory("dst").read_all()) == 2
        assert len(db.memory("src").read_all()) == 0

    @pytest.mark.asyncio
    async def test_merge_drops_colliding_keys(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="src",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        _seed_collection(
            db,
            name="dst",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        write = CollectionWriteTool(db, _make_llm_client(mock_llm), author="test")
        await write.execute(memory="src", entries=[{"key": "shared", "content": "from src"}])
        await write.execute(memory="src", entries=[{"key": "unique", "content": "only in src"}])
        await write.execute(memory="dst", entries=[{"key": "shared", "content": "already in dst"}])

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "1 moved" in result.message
        assert "1 dropped" in result.message
        # The dropped collision keys are named, not just counted.
        assert "'shared'" in result.message
        dst_entries = db.memory("dst").read_all()
        assert len(dst_entries) == 2
        contents = {e.key: e.content for e in dst_entries}
        assert contents["shared"] == "already in dst"  # destination wins
        assert contents["unique"] == "only in src"

    @pytest.mark.asyncio
    async def test_merge_empty_source_archives_it(self, tmp_path):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="src",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        _seed_collection(
            db,
            name="dst",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

        result = await CollectionMergeTool(db, "test").execute(from_memory="src", to_memory="dst")

        assert "archived" in result.message
        assert db.memories.get("src").archived is True


class TestTestExtractionPromptTool:
    """TestExtractionPromptTool delegates to Collector.run_for — test the formatting."""

    class _MockCollector:
        """Duck-typed stub: records the call and returns a configured result."""

        def __init__(self, result: tuple[bool, str]) -> None:
            self._result = result
            self.called_with: str | None = None

        async def run_for(self, collection_name: str) -> tuple[bool, str]:
            self.called_with = collection_name
            return self._result

    @pytest.mark.asyncio
    async def test_success_returns_checkmark_and_summary(self):
        collector = self._MockCollector((True, "Collector cycle complete. wrote 3 entries"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="board-games")
        assert collector.called_with == "board-games"
        assert result.message.startswith("✅")
        assert "wrote 3 entries" in result.message
        assert result.success is True

    @pytest.mark.asyncio
    async def test_failure_returns_x_and_summary(self):
        collector = self._MockCollector((False, "Collector cycle complete. max steps exceeded"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="likes")
        assert result.message.startswith("❌")
        assert "max steps exceeded" in result.message
        # the failure must reach structural accounting, not live only in the ❌ text
        assert result.success is False

    @pytest.mark.asyncio
    async def test_validation_error_returns_x_and_error_message(self):
        collector = self._MockCollector((False, "Collection 'missing' not found."))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        result = await tool.execute(memory="missing")
        assert result.message.startswith("❌")
        assert "not found" in result.message
        assert result.success is False

    @pytest.mark.asyncio
    async def test_unicode_dash_in_memory_name_normalized(self):
        """MemoryNameArgs normalises Unicode dashes before passing to run_for."""
        collector = self._MockCollector((True, "Collector cycle complete. wrote 1 entry"))
        tool = TestExtractionPromptTool(collector)  # ty: ignore[invalid-argument-type]
        await tool.execute(memory="board‑games")  # U+2011 non-breaking hyphen
        assert collector.called_with == "board-games"


class TestFactory:
    """One uniform surface for every agent — reads + lifecycle (shape) + entry
    mutations (contents).  Capability is no longer curated by omission; the
    only per-agent difference is ``scope``, which drives the collector-binding
    *invariant* (see TestScopedFactory), not which tools are present.
    """

    _FULL_SURFACE = {
        # Reads
        "collection_get",
        "collection_read_latest",
        "collection_read_random",
        "collection_keys",
        "memory_metadata",
        "collection_catalog",
        "log_read",
        "read_run_calls",
        "get_event",
        "read_similar",
        "exists",
        "find_mine",
        # Lifecycle (shape)
        "collection_create",
        "collection_update",
        "collection_merge",
        "collection_archive",
        "collection_unarchive",
        "log_create",
        "skill_create",
        "skill_read",
        # Entry mutations (contents)
        "collection_write",
        "update_entry",
        "collection_delete_entry",
        "log_append",
    }

    def test_chat_surface_is_the_full_set(self, tmp_path, mock_llm):
        """Chat (scope=None) gets every memory tool — entry mutations included,
        unrestricted, since edits are user-directed."""
        db = _make_db(tmp_path)
        tools = build_memory_tools(db, _make_llm_client(mock_llm), agent_name="chat")
        assert {tool.name for tool in tools} == self._FULL_SURFACE

    def test_collector_surface_is_the_same_full_set(self, tmp_path, mock_llm):
        """A bound collector (scope=X) gets the identical surface — scope binds
        its entry mutations to X but does not strip lifecycle/other tools."""
        db = _make_db(tmp_path)
        tools = build_memory_tools(
            db, _make_llm_client(mock_llm), agent_name="collector", scope="likes"
        )
        assert {tool.name for tool in tools} == self._FULL_SURFACE


class TestScopedFactory:
    """Scope binds a collector to one collection.  Writes to other collections
    get a clean refusal at the tool layer, so a confused or jailbroken
    collector can't trash unrelated memories.
    """

    @pytest.mark.asyncio
    async def test_scoped_write_rejects_other_collection(self, tmp_path, mock_llm):
        """A scoped collector that tries to write to a different collection
        gets a clean refusal rather than silently corrupting unrelated data."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )
        _seed_collection(
            db,
            name="dislikes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="dislikes", entries=[{"key": "k", "content": "v"}])

        assert (
            result.message == "Refused: this collector can only write to 'likes', not 'dislikes'."
        )
        # And nothing was actually written
        assert db.memory("dislikes").get("k") == []

    @pytest.mark.asyncio
    async def test_scoped_write_allows_target_collection(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="likes",
            description="x",
            extraction_prompt="test fixture extraction prompt",
            collector_interval_seconds=3600,
            intent="a running list the user asked me to keep",
        )

        write = CollectionWriteTool(
            db, _make_llm_client(mock_llm), author="collector:likes", scope="likes"
        )
        result = await write.execute(memory="likes", entries=[{"key": "k", "content": "v"}])

        assert "Wrote 1 entry" in result.message
        assert db.memory("likes").get("k")[0].content == "v"

    @pytest.mark.asyncio
    async def test_scoped_update_entry_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        update = UpdateEntryTool(db, author="collector:likes", scope="likes")
        result = await update.execute(memory="dislikes", key="k", content="v")
        assert "Refused" in result.message

    @pytest.mark.asyncio
    async def test_scoped_delete_rejects_other_collection(self, tmp_path):
        db = _make_db(tmp_path)
        delete = CollectionDeleteEntryTool(db, scope="likes")
        result = await delete.execute(memory="dislikes", key="k")
        assert "Refused" in result.message


class TestRegistryProvenanceAndLifecycle:
    """Operational registry (#1566): ``collection_create`` stamps the spawning
    message + creating run; ``collection_catalog`` (archived-inclusive) and
    ``memory_metadata`` render a status / expires / created-from-message block —
    so any mechanism can state who asked for it, what created it, whether it's
    live, and when it ends, and an archived one stays enumerable and inspectable.
    """

    async def _create(self, db, name, *, created_by_run_id=None, notify=False):
        # Instantiate through the real front door: a hole-less "gather" skill renders
        # into the collection's prompt, and the provenance (creating run) is stamped.
        _seed_watch_skill(
            db,
            name="gather-items",
            intent="gather fresh items on a topic",
            description="gather fresh items on a topic",
            holes=[],
            steps=[
                SkillStep(
                    ordinal=1,
                    source_ordinal=1,
                    tool="browse",
                    arguments={"queries": ["fresh items"], "extract": "the newest items"},
                    substitutions=[],
                )
            ],
        )
        return await CollectionCreateTool(
            db, cast(Any, MockLlmClient()), created_by_run_id=created_by_run_id
        ).execute(
            name=name,
            intent=f"the user's goal for {name}",
            skill="gather-items",
            interval=3600,
            notify=notify,
        )

    def _spawn(self, db, run_id, content):
        """Log an incoming user message and link it to ``run_id`` — mirroring the
        channel's post-run provenance link (the message id isn't known until the
        run returns)."""
        message_id = db.messages.log_message(
            PennyConstants.MessageDirection.INCOMING, "user", content
        )
        db.memories.link_source_message(run_id, message_id)
        return message_id

    @pytest.mark.asyncio
    async def test_create_stamps_run_then_channel_links_message(self, tmp_path):
        db = _make_db(tmp_path)
        # collection_create stamps only the creating run — the spawning message
        # isn't known until the run returns, and there's no end condition.
        await self._create(db, "espresso-reviews", created_by_run_id="run-espresso-01")
        row = db.memories.get("espresso-reviews")
        assert row.created_by_run_id == "run-espresso-01"
        assert row.source_message_id is None
        assert row.expires_at is None
        # The channel then links the spawning message by run id.
        message_id = self._spawn(
            db, "run-espresso-01", "can you keep an eye on new espresso machine reviews?"
        )
        assert db.memories.get("espresso-reviews").source_message_id == message_id

    @pytest.mark.asyncio
    async def test_metadata_renders_full_lifecycle(self, tmp_path):
        db = _make_db(tmp_path)
        await self._create(db, "audiobooks", created_by_run_id="run-audiobooks-01")
        message_id = self._spawn(
            db, "run-audiobooks-01", "keep a running list of good sci-fi audiobooks"
        )
        result = await MemoryMetadataTool(db).execute(memory="audiobooks")
        # One exact-name call answers the whole lifecycle question.
        assert "status: active" in result.message
        assert "expires: never" in result.message
        assert "by run run-audiobooks-01" in result.message
        assert f'from message {message_id} ("keep a running list of good sci-fi audiobooks")' in (
            result.message
        )

    @pytest.mark.asyncio
    async def test_catalog_is_archived_inclusive_and_marks_status(self, tmp_path):
        db = _make_db(tmp_path)
        await self._create(db, "kickstarters", created_by_run_id="run-ks-01")
        message_id = self._spawn(db, "run-ks-01", "watch for new board game kickstarters")
        await self._create(db, "trail-conditions")  # no run id (seeded-style)
        # Archiving must change status, never visibility.
        await CollectionArchiveTool(db).execute(memory="kickstarters")
        result = await CollectionCatalogTool(db).execute()

        # Both collections still render — the just-archived one clearly marked.
        assert "## kickstarters" in result.message
        assert "## trail-conditions" in result.message
        assert "status: archived" in result.message
        assert "status: active" in result.message
        # A count over the inventory surface is correct wrt the DB (2 collections,
        # one of them archived) — the archived row is not hidden.
        assert result.message.count("## ") == 2
        # Provenance (message + ask excerpt) renders on the created line.
        assert (
            f'from message {message_id} ("watch for new board game kickstarters")' in result.message
        )

    @pytest.mark.asyncio
    async def test_long_ask_is_excerpted(self, tmp_path):
        db = _make_db(tmp_path)
        long_ask = (
            "please keep a really thorough running list of every single new mechanical "
            "keyboard group buy you can find anywhere on the internet, forever"
        )
        await self._create(db, "keyboards", created_by_run_id="run-kb-01")
        self._spawn(db, "run-kb-01", long_ask)
        result = await MemoryMetadataTool(db).execute(memory="keyboards")
        # The ask is truncated to the first ~80 chars with an ellipsis marker, so
        # a verbose request doesn't blow up the rendered line.
        assert long_ask[:40] in result.message
        assert long_ask not in result.message
        assert "…" in result.message

    @pytest.mark.asyncio
    async def test_expires_at_renders_end_condition(self, tmp_path):
        db = _make_db(tmp_path)
        expiry = datetime(2026, 12, 25, 9, 0, tzinfo=UTC)
        db.memories.create_collection(
            "holiday-watch",
            "seasonal watch subject matter",
            extraction_prompt=("1. gather holiday deals.\n2. done()."),
            collector_interval_seconds=3600,
            expires_at=expiry,
        )
        result = await MemoryMetadataTool(db).execute(memory="holiday-watch")
        # A set end condition renders as its UTC datetime, not "never".
        assert "expires: 2026-12-25 09:00 UTC" in result.message

    @pytest.mark.asyncio
    async def test_on_advance_trigger_renders_in_metadata(self, tmp_path):
        """An on_advance collection's metadata carries the source-driven trigger line
        (#1604); an interval collection carries none (byte-identical to before)."""
        db = _make_db(tmp_path)
        db.memories.create_log("events-log", "an event stream")
        db.memories.create_collection(
            "chained-watch",
            "digest events subject matter",
            extraction_prompt=('1. log_read("events-log").\n2. digest.'),
            collector_interval_seconds=30,
            source_log="events-log",
        )
        db.memories.create_collection(
            "plain-watch",
            "recurring watch subject matter",
            extraction_prompt=("1. gather deals.\n2. save."),
            collector_interval_seconds=3600,
        )
        on_advance = await MemoryMetadataTool(db).execute(memory="chained-watch")
        assert "trigger: on advance of events-log" in on_advance.message
        # The plain recurring collection renders no trigger line — unchanged shape.
        plain = await MemoryMetadataTool(db).execute(memory="plain-watch")
        assert "trigger:" not in plain.message


class TestCollectionSkillProvenanceRender:
    """Collection→skill provenance (#1603): a skill-instantiated collection records
    the skill it was rendered from + the params bound into its render, and the
    catalog / metadata surfaces name them — ``from skill: <name> (<param>=<value>)``.
    The skill name is a live anchor (one ``skill_read`` hop reaches its steps/holes),
    and the bound params are the reachable input a rebind/re-render consumes.  A
    hand-authored / seeded collection (``skill_name`` NULL) renders EXACTLY as before
    — the unmarked case is the quiet default, pinned byte-for-byte."""

    @pytest.mark.asyncio
    async def test_catalog_names_instantiating_skill_and_params_whole(self, tmp_path):
        """The catalog render names the instantiating skill + its bound params on a
        ``from skill:`` line right before the recipe it produced — asserted as a whole
        render so the line's position and the unchanged rest are both pinned."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        create = await CollectionCreateTool(
            db, cast(Any, MockLlmClient()), created_by_run_id="run-cinder-01"
        ).execute(
            name="cinder-elevation",
            intent="watch Cinder Peak's elevation",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
        )
        assert create.success
        row = db.memories.get("cinder-elevation")
        result = await CollectionCatalogTool(db).execute()
        expected = (
            "## cinder-elevation\n"
            "status: active\n"
            "expires: never\n"
            f"created: {format_log_timestamp(row.created_at)} by run run-cinder-01\n"
            "description: watch Cinder Peak's elevation\n"
            "intent: watch Cinder Peak's elevation\n"
            "notify: False\n"
            "from skill: Watch elevation (peak=Cinder Peak)\n"
            f"extraction_prompt:\n{_MONEY_LITERAL}"
        )
        assert result.message == expected

    @pytest.mark.asyncio
    async def test_metadata_names_instantiating_skill_and_params(self, tmp_path):
        """``memory_metadata`` names the skill + bound params, positioned ahead of the
        extraction prompt it produced — so "which skill made this, and with what?" is
        the same call that reads the collection."""
        db = _make_db(tmp_path)
        _seed_watch_skill(db)
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="cinder-elevation",
            intent="watch Cinder Peak's elevation",
            skill=_SKILL_NAME,
            params={"peak": "Cinder Peak"},
            interval=3600,
        )
        result = await MemoryMetadataTool(db).execute(memory="cinder-elevation")
        assert "from skill: Watch elevation (peak=Cinder Peak)" in result.message
        # The provenance names the recipe's origin, ahead of the recipe itself.
        assert result.message.index("from skill:") < result.message.index("extraction prompt:")

    @pytest.mark.asyncio
    async def test_hand_authored_collection_renders_byte_identical(self, tmp_path):
        """A collection with no skill origin (``skill_name`` NULL — the seeded /
        hand-authored case) renders with NO ``from skill:`` line, byte-for-byte the
        unmarked pre-provenance shape, on both surfaces."""
        db = _make_db(tmp_path)
        _seed_collection(
            db,
            name="plain-watch",
            description="plain subject",
            intent="plain intent",
            extraction_prompt="1. browse(queries=['peaks']).\n2. collection_write(memory='peaks').",
        )
        row = db.memories.get("plain-watch")
        assert row.skill_name is None and row.skill_params is None
        catalog = await CollectionCatalogTool(db).execute()
        expected = (
            "## plain-watch\n"
            "status: active\n"
            "expires: never\n"
            f"created: {format_log_timestamp(row.created_at)}\n"
            "description: plain subject\n"
            "intent: plain intent\n"
            "notify: False\n"
            "extraction_prompt:\n1. browse(queries=['peaks']).\n"
            "2. collection_write(memory='peaks')."
        )
        assert catalog.message == expected
        assert "from skill:" not in catalog.message
        metadata = await MemoryMetadataTool(db).execute(memory="plain-watch")
        assert "from skill:" not in metadata.message

    @pytest.mark.asyncio
    async def test_holeless_skill_renders_skill_name_only(self, tmp_path):
        """A skill with no holes binds no params, so the render names just the skill —
        ``from skill: <name>`` with no parenthesised params."""
        db = _make_db(tmp_path)
        _seed_watch_skill(
            db,
            name="daily-digest",
            intent="gather the day's fresh items",
            description="gather the day's fresh items",
            holes=[],
            steps=[
                SkillStep(
                    ordinal=1,
                    source_ordinal=1,
                    tool="browse",
                    arguments={"queries": ["fresh items"], "extract": "the newest items"},
                    substitutions=[],
                )
            ],
        )
        await CollectionCreateTool(db, cast(Any, MockLlmClient())).execute(
            name="morning-digest",
            intent="the day's fresh items",
            skill="daily-digest",
            interval=3600,
        )
        result = await CollectionCatalogTool(db).execute()
        assert "from skill: daily-digest\n" in result.message
        # No holes bound → no parenthesised params.
        assert "from skill: daily-digest (" not in result.message


# ── find_mine: resolve-by-meaning, identity fused with affordances (#1558) ────

_FIND_MINE_VOCAB = {
    "aurora": 0,
    "beacon": 1,
    "cascade": 2,
    "gamma": 3,
    "delta": 4,
    "echo": 5,
    "foxtrot": 6,
    "reusable": 7,
    "recipes": 8,
    "orbit": 9,
    "nebula": 10,
}
_FIND_MINE_DIM = 16


def _axis_vec(text: str) -> list[float]:
    """A collision-free, L2-normalised vector: each known word owns a fixed axis
    (unknown words ignored), so cosine between two texts is exactly
    (shared words) / (sqrt(len_a) * sqrt(len_b)) — deterministic scores for an
    exact whole-render literal."""
    vec = [0.0] * _FIND_MINE_DIM
    for word in text.lower().split():
        axis = _FIND_MINE_VOCAB.get(word)
        if axis is not None:
            vec[axis] += 1.0
    norm = sum(value * value for value in vec) ** 0.5
    return [value / norm for value in vec] if norm else vec


def _axis_embed(model: str, text: str | list[str]) -> list[list[float]]:
    inputs = text if isinstance(text, list) else [text]
    return [_axis_vec(one) for one in inputs]


def _axis_client(mock_llm) -> LlmClient:
    """A client whose embeddings are the fixed-axis vectors above — so every
    stored description/content anchor and the query share one exact geometry."""
    mock_llm.set_embed_handler(_axis_embed)
    return LlmClient(
        api_url="http://localhost:11434", model="test-model", max_retries=1, retry_delay=0.0
    )


async def _create_collection(db, client: LlmClient, name: str, description: str) -> None:
    """Instantiate a collection whose intent/description anchor is ``description``
    (what ``find_mine`` resolves over).  A hole-less skill supplies the rendered
    prompt; ``create_anyway`` skips the idempotency check so these tests can stand
    up several deliberately-similar collections.  The helper skill is seeded
    UNEMBEDDED (``embed=False``) — it's resolved by exact name, and an anchor in
    the hash geometry would collide with the axis geometry these tests pin."""
    _seed_watch_skill(
        db,
        name="find-skill",
        intent="find skill",
        description="find skill",
        holes=[],
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="browse",
                arguments={"queries": ["items"], "extract": "the items"},
                substitutions=[],
            )
        ],
        embed=False,
    )
    await CollectionCreateTool(db, client).execute(
        name=name, intent=description, skill="find-skill", interval=3600, create_anyway=True
    )


class TestFindMine:
    """Resolve-by-meaning over the whole registry + taught skills (the ``skill``
    table, the sole skills store — #1624), fusing exact identity with how to
    address it (#1558).  The result is model-facing text, so each mode is
    asserted as a whole render."""

    _KITCHEN_SINK = (
        'Found 4 things matching "aurora beacon cascade", best first:\n'
        "1. aurora-watch — active collection: aurora beacon cascade\n"
        "   how to use it: read it with collection_read_latest('aurora-watch'), "
        "reconfigure it with collection_update(name='aurora-watch', ...), archive it "
        "with collection_archive('aurora-watch')\n"
        "2. escalate-aurora — live taught skill: aurora beacon cascade gamma\n"
        "   how to use it: read it with skill_read('escalate-aurora'); to change it, "
        "re-teach it with skill_create — the same name replaces it\n"
        "3. aurora-archive — archived collection: aurora beacon delta\n"
        "   how to use it: restore it with collection_unarchive('aurora-archive'); its "
        "entries stay readable with collection_read_latest('aurora-archive')\n"
        "4. aurora-log — active log: aurora echo foxtrot\n"
        "   how to use it: read it with log_read('aurora-log')\n"
        "Ranked by closeness — if one is what you meant, use its addressing above; "
        "otherwise narrow by its exact name, or pass type=<collection|log|skill>."
    )

    @staticmethod
    async def _seed_world(db, client: LlmClient) -> None:
        """One object of every renderable family, all sharing the query's meaning:
        an active collection, an archived collection, a log, and a taught skill
        (seeded straight into the ``skill`` table with an axis-geometry anchor)."""
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        await _create_collection(db, client, "aurora-archive", "aurora beacon delta")
        await CollectionArchiveTool(db).execute(memory="aurora-archive")
        await LogCreateTool(db, client).execute(
            name="aurora-log",
            description="aurora echo foxtrot",
        )
        db.skills.upsert(
            SkillDraft(
                name="escalate-aurora",
                intent="aurora beacon cascade gamma",
                description="aurora beacon cascade gamma",
                steps=[],
                holes=[],
                source_run_id="run-teach",
            ),
            author="chat",
            description_embedding=_axis_vec("aurora beacon cascade gamma"),
        )

    @pytest.mark.asyncio
    async def test_kitchen_sink_fuses_identity_and_affordances(self, tmp_path, mock_llm):
        """A query matching a mixed set returns each hit's exact identity, family,
        live/archived state, AND the deterministic addressing — best-first."""
        db = _make_db(tmp_path)
        client = _axis_client(mock_llm)
        await self._seed_world(db, client)
        result = await FindMineTool(db, client).execute(query="aurora beacon cascade")
        assert result.success
        assert result.message == self._KITCHEN_SINK

    @pytest.mark.asyncio
    async def test_type_filter_narrows_to_skills(self, tmp_path, mock_llm):
        """``type=skill`` narrows the same world to the taught skill alone, with
        the skill-specific addressing (the skill-vs-collection footgun answered in
        the result)."""
        db = _make_db(tmp_path)
        client = _axis_client(mock_llm)
        await self._seed_world(db, client)
        result = await FindMineTool(db, client).execute(query="aurora beacon cascade", type="skill")
        assert result.message == (
            'Found 1 thing matching "aurora beacon cascade":\n'
            "1. escalate-aurora — live taught skill: aurora beacon cascade gamma\n"
            "   how to use it: read it with skill_read('escalate-aurora'); to change "
            "it, re-teach it with skill_create — the same name replaces it"
        )

    @pytest.mark.asyncio
    async def test_single_confident_match(self, tmp_path, mock_llm):
        db = _make_db(tmp_path)
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "solo-watch", "aurora beacon cascade")
        result = await FindMineTool(db, client).execute(query="aurora beacon cascade")
        assert result.message == (
            'Found 1 thing matching "aurora beacon cascade":\n'
            "1. solo-watch — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_read_latest('solo-watch'), "
            "reconfigure it with collection_update(name='solo-watch', ...), archive it "
            "with collection_archive('solo-watch')"
        )

    @pytest.mark.asyncio
    async def test_ambiguous_returns_all_candidates_ranked(self, tmp_path, mock_llm):
        """Several matches come back ranked with how to narrow — never one silently
        chosen."""
        db = _make_db(tmp_path)
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "watch-primary", "aurora beacon cascade")
        await _create_collection(db, client, "watch-secondary", "aurora beacon delta")
        result = await FindMineTool(db, client).execute(query="aurora beacon cascade")
        assert result.message == (
            'Found 2 things matching "aurora beacon cascade", best first:\n'
            "1. watch-primary — active collection: aurora beacon cascade\n"
            "   how to use it: read it with collection_read_latest('watch-primary'), "
            "reconfigure it with collection_update(name='watch-primary', ...), archive it "
            "with collection_archive('watch-primary')\n"
            "2. watch-secondary — active collection: aurora beacon delta\n"
            "   how to use it: read it with collection_read_latest('watch-secondary'), "
            "reconfigure it with collection_update(name='watch-secondary', ...), archive it "
            "with collection_archive('watch-secondary')\n"
            "Ranked by closeness — if one is what you meant, use its addressing above; "
            "otherwise narrow by its exact name, or pass type=<collection|log|skill>."
        )

    @pytest.mark.asyncio
    async def test_zero_matches_is_honest_empty(self, tmp_path, mock_llm):
        """A query unrelated to everything returns an honest empty naming the wider
        nets (catalog + self-state header) — not an error, no dead end."""
        db = _make_db(tmp_path)
        client = _axis_client(mock_llm)
        await _create_collection(db, client, "aurora-watch", "aurora beacon cascade")
        result = await FindMineTool(db, client).execute(query="orbit nebula")
        assert result.success
        assert result.message == (
            'Nothing of yours matched "orbit nebula". Widen the net: collection_catalog() '
            "lists every collection (archived included), and your current-state header "
            "names your active mechanisms, logs, and recent activity."
        )

    @pytest.mark.asyncio
    async def test_transient_embed_failure_is_actionable(self, tmp_path):
        """A transient query-embed failure returns an actionable retry, not a silent
        empty — the miss is named, the fix bound."""
        db = _make_db(tmp_path)
        result = await FindMineTool(db, cast(Any, _FailingEmbedClient())).execute(query="anything")
        assert result.success is False
        assert "Couldn't embed your query" in result.message
        assert "find_mine(query=" in result.message
