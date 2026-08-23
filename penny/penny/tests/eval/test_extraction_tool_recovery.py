"""extraction_prompt fictitious-tool recovery contract (#1529, epic #1528).

When the write-time gate refuses an ``extraction_prompt`` that names a tool no
collector can run, the correction-teaching rejection must be load-bearing: the live
model reads it and REWRITES the prompt using only real tools, rather than
re-emitting the hallucination or giving up.  The deterministic gate (reject before
the write, name the offender + the available surface) is pinned in
``tests/tools/test_memory_tools.py``; this owns the live model-behaviour contract.

The slip — a hallucinated tool in an authored prompt — is occasional, so we FORCE one
``collection_set`` carrying a made-up ``extract_text`` for the "read the page" step
(``_InjectFictitiousToolPrompt``) and let the REAL model drive the recovery off the
production rejection.  The contract is STRUCTURAL, never wording:

  PASS = the collection's persisted ``extraction_prompt`` names NO fictitious tool
         AND differs from the seed — i.e. a corrected update actually LANDED (the
         model rewrote the read step with a real tool, e.g. ``browse``), rather than
         re-emitting ``extract_text`` (rejected again, nothing persists) or freezing.

**CANDIDATE RETIREMENT — the sabotage no longer maps to a real failure mode
(#1919 audit).**  The gate itself is live, but it no longer sits where this case
aims: since #1658/#1631 the model surface takes NO ``extraction_prompt`` argument
at all (``CollectionSetArgs``, the ``args_model`` of the registered
``CollectionSetTool``, declares ``name``/``description``/``skill``/``params``/
``schedule``/``expires_at``/``notify`` and nothing else), and ``ToolArgs`` is
``extra="forbid"`` — so the injected call is refused by ARG VALIDATION
("unknown parameter 'extraction_prompt'"), never reaching
``check_extraction_prompt_tools``.  A collection's routine is only ever a RENDER of
a demonstrated skill, and the surviving gate call is inside
``render_skill_prompt``, over that render — a prompt the model cannot author and so
cannot hallucinate a tool into.  (The unregistered inner ``CollectionUpdateTool``
still declares the argument on ``CollectionUpdateArgs``; it is reachable only from
the dispatcher, which builds its args from ``CollectionSetArgs``.)

Kept, un-deleted, and report-only pending the code owner's ruling: what would
replace it is a contract over the render path, which is a different case with a
different sabotage, not an edit to this one.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    _InjectFictitiousToolPrompt,
    _iter_prompt_messages,
    seed_collection,
    tool_call_sequence,
)
from penny.tests.eval.fixtures import SynthCollection

pytestmark = pytest.mark.eval

_COLLECTION = "board-games"

# The made-up tool the forced update names, and a substring of the write-time gate's
# refusal (``check_extraction_prompt_tools``).  Matched rather than reproduced — the full
# wording is pinned in ``tests/tools/test_memory_tools.py``.
_FICTITIOUS_TOOL = "extract_text"
_GATE_MARKER = "which a collector cannot use"

# The argument the forced update carries — the one ``collection_set`` no longer declares.
_EXTRACTION_PROMPT_ARGUMENT = "extraction_prompt"

# A substring of the ARG-VALIDATION refusal the forced call actually gets now
# (``Tool._format_field_error``), which is the retirement evidence: ``collection_set``
# declares no ``extraction_prompt`` and ``ToolArgs`` forbids extras, so the call never
# reaches the gate above.
_UNKNOWN_ARGUMENT_MARKER = f"unknown parameter '{_EXTRACTION_PROMPT_ARGUMENT}'"

_SYNTH = SynthCollection(
    _COLLECTION,
    "Strategy board games worth buying, one category at a time.",
    entries=(),
)

# The valid recipe the collection starts with — every call a real collector tool.
_SEED_PROMPT = (
    "Collect strategy board games — one category at a time.\n"
    "1. Randomly pick one category: euro, co-op, deckbuilder.\n"
    '2. browse(["{category} board games"])\n'
    "3. From the results pick one game and note its title and designer.\n"
    '4. collection_write("board-games", entries=[{key: "{title}", '
    'content: "{title} by {designer}"}])\n'
    "5. done()"
)

# The bad rewrite we force as the model's first move: it adds the read step the user
# asked for, but as a hallucinated ``extract_text`` call the gate must refuse.
_FICTITIOUS_PROMPT = (
    "Collect strategy board games — one category at a time.\n"
    "1. Randomly pick one category: euro, co-op, deckbuilder.\n"
    '2. browse(["{category} board games"])\n'
    "3. From the results pick one game and note its page url.\n"
    "4. extract_text(game_url)  # read the game's page\n"
    "5. From the page take the title, designer, and a one-line hook.\n"
    '6. collection_write("board-games", entries=[{key: "{title}", '
    'content: "{title} by {designer}: {hook}"}])\n'
    "7. done()"
)

_USER_MESSAGE = (
    "can you update the board-games collection so it also opens each game's page and "
    "saves a one-line hook about it?"
)


def _seed(db: Database) -> None:
    """The collection the turn is asked to change, with a recipe every step of which names
    a real collector tool — then the one claim the seed makes, asserted."""
    seed_collection(
        db,
        _SYNTH,
        extraction_prompt=_SEED_PROMPT,
        schedule="FREQ=HOURLY",
    )
    _assert_the_seeded_prompt_is_valid(db)


def _assert_the_seeded_prompt_is_valid(db: Database) -> None:
    """The collection starts with the seed prompt, and that prompt names no fictitious
    tool — so "changed from the seed" and "no fictitious tool persisted" are both claims
    about what the TURN did rather than about what the world already was."""
    row = db.memories.get(_COLLECTION)
    assert row is not None, f"the collection {_COLLECTION!r} must exist"
    assert row.extraction_prompt == _SEED_PROMPT, (
        f"the collection must start on the seed recipe, got {row.extraction_prompt!r}"
    )
    assert _FICTITIOUS_TOOL not in _SEED_PROMPT, (
        f"the seed recipe must name no {_FICTITIOUS_TOOL!r} — the case measures whether one "
        "arrives, so a seed carrying it would be green before the turn ran"
    )


def _tool_results_naming(db: Database, marker: str) -> bool:
    """Did any tool result this run carry ``marker``, read off the persisted promptlog —
    the same text the model read?"""
    return any(
        message.get("role") == "tool" and marker in (message.get("content") or "")
        for message in _iter_prompt_messages(db)
    )


def _score_recovered(db: Database, before: set[str], reply: str) -> list[Check]:
    """Graded: the forced update reached the extraction-prompt gate, and a corrected update
    then landed — the stored prompt names no fictitious call and differs from the seed
    (a rejected update never persists, so a changed prompt is proof of a valid rewrite).

    The first check is the case's PREMISE, and since #1919 it is also its retirement
    evidence: ``collection_set`` declares no ``extraction_prompt`` and ``ToolArgs`` forbids
    extras, so the forced call is refused by argument validation and the gate never sees
    it.  The advisory row that follows names which refusal actually came back, so every
    report says WHY rather than leaving a bare red.

    The 'forced bail fired — contract exercised' guard is PREPENDED by ``chat_eval``'s
    graded path (#1697) — it says the injector ran; these say what the runtime did with
    it."""
    row = db.memories.get(_COLLECTION)
    stored = (row.extraction_prompt or "") if row is not None else ""
    reached_gate = _tool_results_naming(db, _GATE_MARKER)
    refused_as_argument = _tool_results_naming(db, _UNKNOWN_ARGUMENT_MARKER)
    return [
        Check(
            "the fictitious tool reached the extraction-prompt gate",
            reached_gate,
            anchor="collection_set(",
            kind="guard",
            rationale=None
            if reached_gate
            else (
                "the gate never saw the prompt — since #1658/#1631 no model-facing argument "
                "carries one, so there is no authored prompt for it to refuse.  Calls made: "
                f"{tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            f"no {_FICTITIOUS_TOOL} call persisted in the collection's recipe",
            _FICTITIOUS_TOOL not in stored,
            kind="state",
            rationale=(
                f"{_FICTITIOUS_TOOL} is in the stored recipe, which a cycle would then fail "
                f"on every run: {stored[:160]!r}"
            )
            if _FICTITIOUS_TOOL in stored
            else None,
        ),
        Check(
            "a corrected recipe landed — the stored prompt changed from the seed",
            stored != _SEED_PROMPT,
            kind="state",
            rationale=None
            if stored != _SEED_PROMPT
            else (
                "the recipe is byte-identical to the seed — nothing the turn did reached the "
                "collection, which is the expected outcome while the premise above is dead"
            ),
        ),
        Check(
            "the forced call was refused by argument validation instead",
            refused_as_argument,
            anchor="collection_set(",
            scored=False,
            kind="proc",
            rationale=(
                f"the call came back 'unknown parameter {_EXTRACTION_PROMPT_ARGUMENT!r}' — "
                "the retirement evidence: the sabotage aims at an argument the surface no "
                "longer has"
            )
            if refused_as_argument
            else "the call was not refused as an unknown argument",
        ),
    ]


async def test_fictitious_extraction_tool_is_rejected_and_recovers(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="extraction-tool-recovery",
        family="chat-recovery",
        message=_USER_MESSAGE,
        seed=_seed,
        wrap_client=lambda real: _InjectFictitiousToolPrompt(real, _COLLECTION, _FICTITIOUS_PROMPT),
        score=_score_recovered,
        min_pass_rate=None,
    )
