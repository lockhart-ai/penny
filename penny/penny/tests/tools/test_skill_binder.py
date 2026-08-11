"""The skill BINDER micro-context (#1867) — grammar, validation, and both outcomes.

The binder fills an EXISTING routine's declared parameters from the user's own words.
Everything it can get wrong is decidable in Python — the parameter set is an input, and a
value is only a value if the user typed it — so this file pins the mechanism against
deterministic mock draws; the live-model contract is
``tests/eval/test_skill_binding.py``.

Two enumerated answers and one escape: every declared parameter filled
(:class:`BoundValues`), one or more of them named as unsupplied
(:class:`MissingParameters`, an outcome and never a failure), or ``None`` when no usable
draw came back after the reroll budget.

All content is fictional.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from penny.constants import PennyConstants
from penny.database.skills import SkillParameter, build_binding_content, render_spoken_turns
from penny.llm.models import LlmMessage, LlmResponse
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.micro_context import (
    BIND_SKILL_SYSTEM_PROMPT,
    BoundValues,
    MicroContext,
    MissingParameters,
    SkillBinding,
)

# ── The fixture round: a routine that already exists, asked for again ─────────

_URL = "https://saltmarsh.example/board"
_KEYWORD = "dawn crossing"
_SPOKEN = render_spoken_turns(
    (
        f"can you keep an eye on the sailing board at {_URL} every morning?",
        "tell me when the dawn crossing turns up on it",
    )
)
_PARAMETERS = (
    SkillParameter(name="url", description="the URL of the board to read each run"),
    SkillParameter(name="keyword", description="which entry to look for on it"),
)
_DECLARED = [parameter.name for parameter in _PARAMETERS]
_CONTENT = build_binding_content(
    _SPOKEN,
    "check_sailing_board",
    "read a sailing board and report the status of one entry",
    _PARAMETERS,
)

_CLEAN_DRAW = f"VALUE url: {_URL}\nVALUE keyword: {_KEYWORD}"


def _responds(content: str) -> MockLlmClient:
    """A mock model client whose every chat returns ``content``."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(message=LlmMessage(role="assistant", content=content))
    )
    return model


async def _bind(model: MockLlmClient) -> SkillBinding | None:
    """The binder over the fixture round, driven by ``model``."""
    return await MicroContext(cast(Any, model)).bind_skill(_CONTENT, _DECLARED, _SPOKEN)


# ── The two enumerated answers ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_clean_draw_fills_every_declared_parameter():
    """The complete answer: one VALUE line per declared parameter, keyed by the name the
    SIGNATURE declares (the binding key everything downstream uses) and carrying the
    user's own text verbatim.  One draw, and the call is attributed to its own ledger
    identity so a run trace shows the filling as its own question."""
    model = _responds(_CLEAN_DRAW)

    binding = await _bind(model)

    assert binding == BoundValues(values={"url": _URL, "keyword": _KEYWORD})
    assert len(model.requests) == 1
    assert model.requests[0]["agent_name"] == PennyConstants.SKILL_BIND_AGENT_NAME
    assert model.requests[0]["prompt_type"] == PennyConstants.SKILL_BIND_PROMPT_TYPE
    # The draw reads the rendered document alone — no Instruction:/Content: wrapper.
    assert model.requests[0]["messages"][1]["content"] == _CONTENT
    assert model.requests[0]["messages"][0]["content"] == BIND_SKILL_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_an_unsupplied_parameter_is_an_outcome_that_names_it():
    """The SHORTFALL direction, which is part of the contract rather than a failure: a
    parameter the words supply nothing for gets its own MISSING line, and the answer names
    it.

    The parameter the words DID supply travels with it — throwing away a correct binding
    on the way to reporting a missing one would make the consumer ask for both again."""
    model = _responds(f"VALUE url: {_URL}\nMISSING keyword")

    binding = await _bind(model)

    assert binding == MissingParameters(names=("keyword",), values={"url": _URL})
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_a_round_that_supplies_nothing_reports_every_parameter_missing():
    """Nothing bound at all is still an answer, not a failure — the machine has something
    to act on (these are the things to ask for) rather than an error to render."""
    model = _responds("MISSING url\nMISSING keyword")

    binding = await _bind(model)

    assert binding == MissingParameters(names=("url", "keyword"), values={})


@pytest.mark.asyncio
async def test_the_answer_is_ordered_and_keyed_by_the_declared_names():
    """A draw that writes a name in its own casing still maps home: names are compared
    HARDENED, and what comes back is keyed by the SIGNATURE's spelling in the SIGNATURE's
    order — which is what the derived collection name is then built from."""
    model = _responds(f"VALUE Keyword: {_KEYWORD}\nVALUE URL: {_URL}")

    binding = await _bind(model)

    assert isinstance(binding, BoundValues)
    assert list(binding.values) == ["url", "keyword"]


# ── Validation: everything an accepted draw can never contain ─────────────────


@pytest.mark.asyncio
async def test_a_value_the_user_never_typed_is_rerolled_then_refused():
    """The check the whole customer exists for.  A value that is not a literal span of
    what the user said is a contract violation, not a binding — re-drawn on the unchanged
    context for the whole budget, then an honest refusal.

    The invented value here is the measured failure class in miniature: a url that is
    nearly the one in the message, carrying a separator that appears nowhere in it."""
    invented = "https://saltmarsh.example/sailing_board"
    model = _responds(f"VALUE url: {invented}\nVALUE keyword: {_KEYWORD}")

    binding = await _bind(model)

    assert binding is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_value_lifted_from_the_signature_is_not_evidence():
    """The document renders the signature AND the user's words, and only the second half
    is evidence: a value copied out of a parameter's own description is refused exactly
    like an invented one.

    This is why the span check is given the user's turns rather than the whole document —
    a routine pointed at its own description is a confabulation wearing the document's
    clothes."""
    model = _responds(f"VALUE url: {_URL}\nVALUE keyword: which entry to look for on it")

    assert await _bind(model) is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_parameter_nobody_declared_is_refused():
    """The parameter set is an INPUT.  A line for something the signature does not declare
    is a parameter MINTED by a draw that has no business minting one — the same violation
    as a state outside the offered union."""
    model = _responds(f"{_CLEAN_DRAW}\nVALUE schedule: every morning")

    assert await _bind(model) is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_draw_that_answers_for_only_some_parameters_fails_whole():
    """COVERAGE (#1828's rule, applied to a customer that knows its declared set exactly):
    a parameter with neither a value nor a missing line went unanswered, and silence is
    not the shortfall answer — the shortfall answer is a MISSING line.  So the draw fails
    whole rather than half-binding a routine."""
    model = _responds(f"VALUE url: {_URL}")

    assert await _bind(model) is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_the_same_parameter_answered_twice_is_refused():
    """A contradictory draw.  Taking either line would let a stray trailing one decide
    what the routine is pointed at, and the two lines here disagree about the two things
    that matter — which parameter, and whether it was supplied at all."""
    model = _responds(f"VALUE url: {_URL}\nVALUE url: {_URL}\nMISSING keyword")

    assert await _bind(model) is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_malformed_line_fails_the_whole_draw():
    """The framer's rule (#1830) on a customer whose lines are all PER_ITEM: the grammar
    DROPS a tag-carrying line it cannot carve, best-effort, so the counted drop is the only
    way this customer can see it at all.

    The decay here is the separator one the grammar was built around — a dash where the
    colon belongs, which makes the whole payload read as a "name" the plausibility gate
    refuses.  Coverage is deliberately satisfied by the two lines that DID carve, so the
    dropped line is the only thing wrong with the draw."""
    model = _responds(f"VALUE url — {_URL}\nVALUE keyword: {_KEYWORD}\nMISSING url")

    assert await _bind(model) is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_poison_is_discarded_and_rerolled_then_refused():
    """The binder rides the same poison screen as every other customer: a degeneration
    collapse is discarded and re-drawn on the unchanged context, never appended, and the
    escape after the budget is the same honest ``None``."""
    model = _responds("...???...")

    assert await _bind(model) is None
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


# ── The span tolerance, exactly as wide as it is declared ─────────────────────


@pytest.mark.asyncio
async def test_the_span_check_folds_case_and_whitespace_and_nothing_else():
    """The declared tolerance, both directions.

    A value survives a capital the user did not type and a run of whitespace the render
    introduced — a value written across a line break is the same value.  Nothing else
    moves: a value differing by one character of punctuation is a different value, because
    that is precisely the class the check exists to catch."""
    folded = _responds(f"VALUE url: {_URL.upper()}\nVALUE keyword: Dawn   Crossing")

    binding = await _bind(folded)

    assert isinstance(binding, BoundValues)
    assert binding.values["keyword"] == "Dawn   Crossing"

    punctuated = _responds(f"VALUE url: {_URL}.\nVALUE keyword: {_KEYWORD}")
    assert await _bind(punctuated) is None


@pytest.mark.asyncio
async def test_a_value_spanning_several_of_the_users_words_binds():
    """A value is a SPAN, not a token: the user's whole phrase, taken as they wrote it,
    including the words around the thing itself when that is what they said."""
    model = _responds(f"VALUE url: {_URL}\nVALUE keyword: the dawn crossing turns up")

    binding = await _bind(model)

    assert isinstance(binding, BoundValues)
    assert binding.values["keyword"] == "the dawn crossing turns up"


# ── The contract block the model is shown ─────────────────────────────────────


def test_binding_system_prompt_whole_render():
    """Whole-render literal of the binding contract (#1867): the framing, the three things
    it is given, the three numbered asks — take the list as it stands, copy the user's own
    text, say when there is nothing to copy — the terms guard, and the TWO enumerated
    output lines, rendered from the declared shape so the tags and separators the model is
    told to write are literally the ones the parse splits on.

    Register: plain words, short sentences, and the shortfall stated as PERMISSION ("that
    is a real answer") rather than only as a prohibition — a draw that has to be ordered
    out of guessing is a draw that was never told guessing was optional.

    The terms guard names the STATE it applies to — how often it runs, when it stops,
    whether to tell the user, all settled where the job is set running — rather than
    keying on the words a particular ask happens to use.

    Both tags are short, common, non-compound words (#1842, the long-literal decay class):
    a literal the model has to spell is a literal it can misspell, and the parse matches
    tags exactly."""
    assert BIND_SKILL_SYSTEM_PROMPT == (
        "You are a filling-in step. A routine already exists, and someone has just asked "
        "for it to be run on a new occasion. You are given the routine — what it is "
        "called, what it is for, and each thing it needs — and the user's own words, "
        "exactly as they wrote them.\n"
        "\n"
        "Fill in each thing the routine needs, from those words:\n"
        "1. Take the list of things it needs as it stands. Never add anything to that "
        "list, and never leave anything out.\n"
        "2. For each one, find the part of the user's words that supplies it, and copy "
        "that part EXACTLY as they wrote it — same characters, same spelling. Do not tidy "
        "it up, shorten it, expand it, or complete a piece they left half-said.\n"
        "3. When their words supply nothing for one of them, say it is missing. That is a "
        "real answer, and it is the right one whenever the alternative is a guess.\n"
        "\n"
        "How often the routine runs, when it should stop, and whether to tell the user "
        "are settled when it is set running. They are never things the routine needs, so "
        "they are never a value here.\n"
        "\n"
        "Write one line for each thing the routine needs, and nothing else:\n"
        "VALUE <parameter_name>: <the value, in the user's own words>\n"
        "MISSING <parameter_name>\n"
        "Write the name exactly as the routine lists it. Write nothing else — no "
        "preamble, no explanation, no restating the ask."
    )
