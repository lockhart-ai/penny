"""The notify-composing micro-context — customer #6 (#1911).

The one thing a model still does when a collector cycle finishes: turn the
framework-assembled document into the sentence the user reads.  It has no tool
channel at all, which is the whole point — the four-step notify tail it replaces was
where 42 of 49 measured cycle deaths landed, on a decayed tool-call envelope.

Two contracts here: the system prompt, pinned whole (it is the code owner's wording to
review, and a drift changes what every notify collection says), and the draw's own
acceptance rule — a message that is not worth delivering is re-drawn on the unchanged
context and then fails honestly, rather than being queued and refused downstream.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from penny.constants import PennyConstants
from penny.tools.micro_context import (
    MESSAGE_TAG,
    NOTIFY_SYSTEM_PROMPT,
    MicroContext,
)

_DOCUMENT = "The `tide-times` routine just ran on its own.\n\n## What it wrote\n- 06:12"


class _Model:
    """A model client that replies with each canned draw in turn, recording the calls."""

    def __init__(self, *draws: str) -> None:
        self._draws = list(draws)
        self.calls: list[dict] = []

    async def chat(self, messages: list[dict], **kwargs: Any):
        self.calls.append({"messages": messages, **kwargs})
        text = self._draws[min(len(self.calls) - 1, len(self._draws) - 1)]
        return type("Response", (), {"content": text})()


async def _compose(*draws: str) -> tuple[str | None, _Model]:
    model = _Model(*draws)
    drawn = await MicroContext(cast(Any, model)).compose_notification(
        _DOCUMENT, run_target="tide-times"
    )
    return drawn, model


def test_notify_system_prompt_is_pinned_whole():
    """WORDING FOR CODE-OWNER REVIEW (#1911).

    The retired ``Prompt.COLLECTOR_NOTIFY_STEPS`` distilled to a single-purpose framing.
    Everything it asked for in a message survives — a greeting, what was found in plain
    words, the source link, and a callback ONLY when a past message is genuinely about
    it — while its two ``read_similar`` steps are gone (their results are handed in
    already) and its ``send_message`` step is gone (the framework sends).

    Register: plain words, short sentences, and the no-callback case stated as
    PERMISSION ("that is the ordinary case and it is fine") rather than only as a
    prohibition — a draw that has to be ordered out of inventing a callback is a draw
    that was never told saying nothing was allowed.

    Nothing here mentions a tool, and nothing asks whether to notify: both were settled
    before this draw was made."""
    assert NOTIFY_SYSTEM_PROMPT == (
        "You are writing one message to the user. A routine of theirs just ran on its "
        "own and found something, and you are given everything about that run: what it "
        "did, what it wrote down, and the closest things the two of you have said "
        "before.\n"
        "\n"
        "Write the message they will actually receive:\n"
        "1. Open with a quick greeting.\n"
        "2. Say what the run found, in plain words — the detail that matters, not a "
        "description of the routine.\n"
        "3. Include the source link when the run has one.\n"
        "4. Add one line calling back to an earlier message ONLY when one of them is "
        "genuinely about this. When none of them is, say nothing about them — that is "
        "the ordinary case and it is fine.\n"
        "\n"
        "Keep it short and friendly, the way a person messages a friend. Write only "
        "what the run actually found: never a detail that is not in front of you, and "
        "never a claim about what happens next.\n"
        "\n"
        "Respond with this line and nothing else:\n"
        "MESSAGE: <the message — it may begin on this same line>\n"
        "Everything after MESSAGE: is the message itself, so write no preamble, no "
        "explanation, and no restating of these instructions."
    )


@pytest.mark.asyncio
async def test_the_document_is_the_whole_user_turn_and_the_draw_is_attributed():
    """The assembled document IS the ask (the bare-content turn the other structured
    customers use), and the draw carries its own ledger identity so a run trace shows
    the message-writing call apart from the cycle's own."""
    _, model = await _compose(f"{MESSAGE_TAG} Hey! The dawn sailing is 06:12 now.")

    assert len(model.calls) == 1
    messages = model.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": NOTIFY_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": _DOCUMENT}
    assert model.calls[0]["agent_name"] == PennyConstants.NOTIFY_COMPOSE_AGENT_NAME
    assert model.calls[0]["prompt_type"] == PennyConstants.NOTIFY_COMPOSE_PROMPT_TYPE
    assert model.calls[0]["run_target"] == "tide-times"


@pytest.mark.asyncio
async def test_the_message_is_everything_after_the_tag():
    """The tag says where the message BEGINS: a chatty draw's preamble is left behind
    rather than delivered, and the value spans the REMAINDER so a message that runs to
    several lines survives whole."""
    drawn, _ = await _compose(
        "Sure, here you go.\n"
        f"{MESSAGE_TAG} Hey! The dawn sailing moved to 06:12.\n"
        "Source: https://ex.example/t"
    )
    assert drawn == "Hey! The dawn sailing moved to 06:12.\nSource: https://ex.example/t"


@pytest.mark.asyncio
async def test_an_untagged_draw_is_rerolled_then_fails_honestly():
    """A draw carrying no MESSAGE line at all is a contract violation like any other:
    re-drawn on the unchanged context for the whole budget, then ``None`` — which the
    caller records on the run and sends nothing for."""
    drawn, model = await _compose("I could not think of anything to say.")

    assert drawn is None
    assert len(model.calls) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_a_message_not_worth_sending_is_rerolled_not_queued():
    """The runtime constraint (``accepts``): the drawn text has to be a message a user
    should actually receive, read through the SAME rule the send path validates with.

    A trailing-off body and a model refusal are each re-drawn rather than handed on —
    the first draw here is unusable, the second is fine, and the second is what comes
    back."""
    good = "Hey! The dawn sailing is 06:12 now."
    for bad in ("Hi there! ......???", "I'm sorry, I can't help with that."):
        drawn, model = await _compose(f"{MESSAGE_TAG} {bad}", f"{MESSAGE_TAG} {good}")
        assert drawn == good
        assert len(model.calls) == 2
