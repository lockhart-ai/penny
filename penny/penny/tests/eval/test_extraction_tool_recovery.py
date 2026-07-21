"""extraction_prompt fictitious-tool recovery contract (#1529, epic #1528).

When ``collection_set`` / ``collection_set`` refuses an ``extraction_prompt``
that names a tool no collector can run, the correction-teaching rejection must be
load-bearing: the live model reads it and REWRITES the prompt using only real tools,
rather than re-emitting the hallucination or giving up.  The deterministic gate
(reject before the write, name the offender + the available surface) is pinned in
``tests/tools/test_memory_tools.py``; this owns the live model-behaviour contract.

The slip — a hallucinated tool in an authored prompt — is occasional, so we FORCE one
``collection_set`` carrying a made-up ``extract_text`` for the "read the page" step
(``_InjectFictitiousToolPrompt``) and let the REAL model drive the recovery off the
production rejection.  The contract is STRUCTURAL, never wording:

  PASS = the collection's persisted ``extraction_prompt`` names NO fictitious tool
         AND differs from the seed — i.e. a corrected update actually LANDED (the
         model rewrote the read step with a real tool, e.g. ``browse``), rather than
         re-emitting ``extract_text`` (rejected again, nothing persists) or freezing.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    _InjectFictitiousToolPrompt,
    seed_collection,
)
from penny.tests.eval.fixtures import SynthCollection

pytestmark = pytest.mark.eval

_COLLECTION = "board-games"

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
    seed_collection(
        db,
        _SYNTH,
        extraction_prompt=_SEED_PROMPT,
        interval=3600,
    )


def _score_recovered(db: Database, before: set[str], reply: str) -> list[Check]:
    """Graded: a corrected update landed — the stored prompt has no fictitious call and differs
    from the seed (the rejected ``extract_text`` update never persists, so a changed prompt is
    proof the model recovered with a valid rewrite).

    The 'forced bail fired — contract exercised' guard is PREPENDED by ``chat_eval``'s graded path
    (#1697) — the forced fictitious-tool update is refused by the extraction-prompt gate, so a run
    that never triggered it can't pass — and this scorer owns only the recovery outcome."""
    row = db.memories.get(_COLLECTION)
    stored = row.extraction_prompt if row is not None else ""
    has_fictitious = stored is not None and "extract_text" in stored
    changed = stored != _SEED_PROMPT
    return [
        Check(
            "no fictitious tool persisted in the prompt",
            not has_fictitious,
            rationale=f"extract_text persisted in the prompt: {(stored or '')[:160]!r}"
            if has_fictitious
            else None,
        ),
        Check(
            "a corrected update landed (prompt changed from the seed)",
            changed,
            rationale=None
            if changed
            else (
                "prompt unchanged from the seed — the model gave up or kept re-emitting "
                "the fictitious call"
            ),
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
        min_pass_rate=0.75,
    )
