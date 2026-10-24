"""A stale call-as-text reply in history, then the same ask again (#2790).

**The prod shape.** Before #2788, a reply that wrote a tool call out as text — one line
of prose, then the call object — was delivered to the user verbatim and stayed in chat
history exactly as sent. #2788 stopped that reply from ever shipping, but a history that
already carries one (seeded here directly, the way a surviving row from before that fix
would read) still poisons the NEXT draw: a fresh model reading its own prior turn as a
written-out call imitates the shape, and — before #2790 — every reroll on it was blind,
discarding draw after draw on the UNCHANGED context until the budget ran out and the run
aborted with no picture ever made (the exact prod incident #2790 describes, three drafts
in a row imitating the one in history).

**What the case claims.** Not the tool call, not the reply's wording — the picture. The
media store holds no image at the start of the sample, and the ask is for one; the fact
that survives the turn is a drawn image naming what was asked for. #2790's fix is not
claimed directly (there is no "the model saw a note" fact to assert — behaviour is
measured, never a forced path, per `docs/principles.md` §4.11); it is PROVED by this
case moving from fail to pass, which `docs/eval-working-guide.md` §1 asks of a
failure-driven case: run at the issue's head to show it reproduces, then at the fix's
head to show it resolves.

**The image backend is a canned boundary**, the same stub `test_image_dispatch.py`
uses: `generate_image` is registered only when an image client is present, and the
stub answers the one network call with a one-pixel PNG so everything past it —
storing, describing, delivering — stays production's.

Every name and place here is invented; the repo is public.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code
owner's to accept once the numbers have been read.
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

_FAMILY = "nl-dispatch"
_GENERATE_IMAGE = "generate_image"
_CASE_ID = "stale-call-as-text-history-retry"

# What the picture must show — the two concrete things the ORIGINAL ask put in it, so
# the claim reads the same subject whichever draw finally makes the call.
SUBJECT = ("Sydney Opera House", "paper boat")

# The earlier exchange, seeded directly into history — a prior ask, and a Penny reply
# that wrote the generate_image call out as text instead of making it (one line of
# prose, then the call object, #2787's own observed shape) rather than a real picture.
_PRIOR_ASK = "could you draw a paper boat sailing past the Sydney Opera House?"
_PRIOR_CALL_AS_TEXT_REPLY = (
    "On it!\n"
    '{"action": "generate_image", '
    '"action_input": "a paper boat sailing past the Sydney Opera House", '
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
        "In the chat agent, when conversation history holds an earlier reply that "
        "wrote a tool call as text, Penny still makes the call the next ask wants."
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
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


def drew(subject: tuple[str, ...]) -> Callable[[SampleObservation, World], Answer]:
    """The media store holds an image drawn this turn whose description names every
    part of ``subject`` — any one drawn image answers it, so a turn that drew twice
    is measured rather than penalised."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = [image.description for image in sample.images if image.drawn]
        shows = any(all(part in fold_typography(text) for part in subject) for text in drawn)
        return shows, f"drew {drawn or 'nothing'}"

    return answer


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_retry_after_a_stale_call_as_text_reply_draws_the_picture(
    chat_eval: ChatEval, model: str
) -> None:
    """History already carries a reply that wrote the call as text; the user asks
    again, and the picture still gets made."""
    cohort = await _drive(chat_eval, model, _RETRY)
    # STORE — the picture the retry asked for, read off the media store.
    cohort.claim(
        "state: the media store holds an image drawn this turn of a paper boat and "
        "the Sydney Opera House",
        drew(SUBJECT),
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply half: nothing invented beyond what the ask and the
    # earlier turn actually said.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
