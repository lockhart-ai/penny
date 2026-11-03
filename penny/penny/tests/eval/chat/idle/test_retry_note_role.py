"""Stress case: the validator's retry note reaches a redraw as a ``user``-role
message (baseline for #2871, filed by #2902).

**The world.** The same reproducible trigger ``test_stale_call_as_text_retry.py``
proved for #2790: a prior Penny reply is seeded directly into history, written
exactly the way a stale pre-#2788 reply would have been — one line of prose, then
the ``generate_image`` call written out as text instead of made.  A fresh ask to
try again reads that shape in its own history and often imitates it on its OWN
first draw, so ``Agent._invoke_nondegenerate`` discards the draft and appends
``Prompt.CALL_AS_TEXT_RETRY_NOTE`` — today, a plain ``role=user`` message
(``Agent._append_retry_note``) — for the live, same-turn redraw.  No mock, no
forced tool call: the trigger is a seeded PRECONDITION (the stale history), and
what is measured is the real model's own live response to the real note.

**What is claimed.** The retry note is Penny's own harness feedback and names no
subject at all — so a drawn image naming what the USER's own retry ask was about
is attributable only to the user's message, never to the note.  The claim is the
same shape #2790's own case uses: the picture, read off the media store, plus
every value in the reply must trace to something the round was given.

**Real proper nouns** (house rule): the subjects are a real landmark and a real
object, picked to be unrelated to anything else in the suite — never invented
placeholders.

REPORT-ONLY (``min_pass_rate=None``): this baselines #2871 on ``main``; the
ceilings are the code owner's to accept once the numbers are read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple
from unittest.mock import AsyncMock

import pytest

from penny.constants import PennyConstants
from penny.database import Database
from penny.penny import Penny
from penny.tests.conftest import ONE_PX_PNG_B64, TEST_SENDER
from penny.tests.eval.conftest import EVAL_MODELS, Preparer, stored_images
from penny.tests.eval.drivers.chat import ChatEval
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.dispatch_world import assert_dispatch_world
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "retry-note-role"
_GENERATE_IMAGE = "generate_image"
_CASE_ID = "retry-note-user-role"

# What the picture must show — the two concrete things the ORIGINAL ask put in it, a
# real landmark and a real everyday object, unrelated to anything else the suite uses.
SUBJECT = ("Niagara Falls", "a kite")

_PRIOR_ASK = "could you draw a kite flying over Niagara Falls?"
_PRIOR_CALL_AS_TEXT_REPLY = (
    "On it!\n"
    '{"action": "generate_image", '
    '"action_input": "a kite flying over Niagara Falls", '
    '"reasoning": "the user wants a picture of this"}'
)


def install_image_client(penny: Penny) -> None:
    """Put a canned image backend in front of the real ``generate_image`` tool — the
    stub answers the one network call; storing, describing and delivering stay
    production's. Installing a client is also what REGISTERS the tool."""
    client = AsyncMock()
    client.generate_image.return_value = ONE_PX_PNG_B64
    penny.chat_agent._image_client = client


def _seed_stale_call_as_text_reply(db: Database) -> None:
    """The prior turn whose stored Penny reply wrote the call out as text — read back
    by ``_build_conversation`` as ordinary history, exactly as a surviving row from
    before #2788 would."""
    ask_id = db.messages.log_message(
        PennyConstants.MessageDirection.INCOMING, TEST_SENDER, _PRIOR_ASK
    )
    db.messages.log_message(
        PennyConstants.MessageDirection.OUTGOING,
        "penny",
        _PRIOR_CALL_AS_TEXT_REPLY,
        parent_id=ask_id,
        recipient=TEST_SENDER,
    )


class RetryCase(NamedTuple):
    """One case: the follow-up ask in five wordings, and the fact every wording must
    leave."""

    case_id: str
    behaviour: str
    ask: str
    also_phrased: tuple[str, ...]


_RETRY = RetryCase(
    case_id=_CASE_ID,
    behaviour=(
        "In the chat agent, when a rejected draft's retry note reaches a redraw as a "
        "user-role message, the final reply still answers what the user's own "
        "message asked for, not the note."
    ),
    ask="can you try drawing that again?",
    also_phrased=(
        "that didn't work — can you actually draw it this time?",
        "please try making that picture again",
        "could you draw the one you were just describing?",
        "try again with that picture please",
    ),
)


def _assert_world(penny: Penny, case_id: str) -> None:
    """``generate_image`` on the surface, no collection in the registry, no image in
    the media store, and the stale call-as-text reply already sitting in history —
    so every image after the turn is the turn's, and the poisoned prior turn is the
    world's premise, not an accident of timing."""
    assert_dispatch_world(penny, case_id, [_GENERATE_IMAGE])
    held = stored_images(penny.db)
    assert not held, f"{case_id}: the media store must start empty, it holds {held}"


def _prepare(case: RetryCase) -> Preparer:
    """Install the canned image backend, seed the stale call-as-text reply, then
    assert the world it stood up."""

    def prepare(penny: Penny) -> None:
        install_image_client(penny)
        _seed_stale_call_as_text_reply(penny.db)
        _assert_world(penny, case.case_id)

    return prepare


async def _drive(chat_eval: ChatEval, model: str, case: RetryCase) -> Cohort:
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        area=Area.CHAT_TOOLS,
        model=model,
        prepare=_prepare(case),
        world=World(name=case.case_id, pages=(), keeps=(), excludes=()),
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only — baselines #2871, read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


def drew(subject: tuple[str, ...]) -> Callable[[SampleObservation, World], Answer]:
    """The media store holds an image drawn this turn whose description names every
    part of ``subject`` — any one drawn image answers it, so a turn that drew twice
    is measured rather than penalised.  ``fold_typography`` casefolds (among other
    things), so ``subject`` — real proper nouns, capitalised — is folded on both
    sides of the check, not only the haystack."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = [image.description for image in sample.images if image.drawn]
        shows = any(
            all(fold_typography(part) in fold_typography(text) for part in subject)
            for text in drawn
        )
        return shows, f"drew {drawn or 'nothing'}"

    return answer


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_redraw_answers_the_users_ask_not_the_retry_note(
    chat_eval: ChatEval, model: str
) -> None:
    """History already carries a reply that wrote the call as text; the retry note
    the live redraw sees names no subject at all, so the picture that survives
    naming the user's own subject is attributable only to her own message."""
    cohort = await _drive(chat_eval, model, _RETRY)
    # STORE — the picture the user's own retry ask named, read off the media store.
    cohort.claim(
        "state: the media store holds an image drawn this turn of a kite over Niagara Falls",
        drew(SUBJECT),
        SpecCategory.STORE,
    )

    # PROVENANCE — nothing in the reply traces to anything but the user's own turns
    # and the tool results; the retry note is harness feedback, not a source.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
