"""Image dispatch: three cases (#2208, #3033 — the chat idle group).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**Three cases, because one setup cannot produce every direction.**  A case is one setup, one run
and one set of assertions (#2100), and the three messages ask different things of the same world:

| direction | case | what the run must leave |
|---|---|---|
| asked for | ``image-request-draws`` | a picture of what was asked for, in the media store |
| the topic alone | ``image-no-fire`` | a reply that invents nothing |
| handed over | ``image-request-hands-over-token`` | a reply carrying the image token once |

**"Draw me X" and "make a picture of X" are one behaviour in two wordings**, so they are two of
the request case's five wordings rather than two cases.  The other three use verbs the tool's
own description does not name ("an image of", "sketch", "an illustration of"), so the cohort is
not measuring how closely a user happens to echo that description.  The subject is the same in
every wording — a teal origami dragon perched on a lighthouse — because the claim names it.

**The image backend is a canned boundary.**  ``generate_image`` is registered only when an image
client is present, and the eval world has no image model to call, so ``install_image_client``
puts a stub client in front of the real tool: it answers every request with the same one-pixel
PNG.  Everything past the network call is production's — the tool stores the picture in the
media store under the description the model wrote, with no source page, and the channel
delivers it by id with the reply.  The model's decision is never touched: it reads the tool's
own description and chooses.

**What the request case claims is the picture, not the call.**  Whether ``generate_image`` fired,
and how many times, is a route measured in the tool sequence.  The fact is what the turn left in
the media store: an image drawn this turn whose description names both things the ask put in
the picture, ``dragon`` and ``lighthouse``.  Both, because a picture of a dragon with no
lighthouse is not what was asked for; and only those two, because ``teal`` and ``origami`` are
words the model may fairly render another way ("turquoise", "folded paper") where the two
nouns have no other rendering.  The rest of the description is prose the tool asks the model to
elaborate — style, light, setting — so no provenance claim reads it.  That the drawn image
reached the user is not claimed: delivery by id is production's own deterministic plumbing, and
a claim over it would run 15/15 by construction.

**What the reply says about the picture is not claimed either.**  The ask asks for a picture,
not a description of one, so "here you go!" is a complete answer and no reply token is owed.
The reply is held only to provenance: every specific value in it traces to what she was given.

**The handover case claims what the record holds, not the wire.**  The ``generate_image`` result
hands her an ``[image <id>]`` token to put in her reply (#3033); egress strips the token from
the wire text — the image attaches by id regardless — and the stored reply keeps it as she
wrote it (#2722).  So this case's reply claims read ``stored_reply`` (the conversation record,
"as she wrote it"), never ``reply`` (the wire, "as the user saw it"): the token is egress
machinery, deterministic and mock-test-owned, and what the model decides is whether she put the
token in her reply at all.  The claims: the media store holds exactly ONE image drawn this turn
for the one request (what survives — a second draw is the defect #3033 fixes, measured here as
the store fact the ruling chose over a restraint claim on the tool sequence); the stored reply
carries the token form exactly once; no other image placeholder stands in it (the excluded
forms are the ones she invented when no token was handed to her — ``{image}``, a bare
``[image]``, and the markdown image shape — each a form the result rules out by naming the one
legitimate token); and the token's id traces to the tool result she was given (PROVENANCE: the
result names the id, so an invented id fails).  The subject is a fresh invented one — a red
paper crane on a stack of books — so this case's fixture never lends its wording to the sibling
cases' claims.

**The no-fire case asserts what survives, never that she refrained** (`docs/principles.md`
§4.3).  A message about a painting someone else made asks for nothing, so whether she draws
anyway is measured in the tool sequence rather than claimed.  The image store is append-only
and no chat tool alters a picture already in it, so a preservation claim over it could not
fail; the world therefore holds nothing to preserve, and the case claims where the machine
lands and that the reply invents nothing.

**The world is asserted out loud before the turn**: the chat surface carries ``generate_image``
(the stub that answers the network call is also what registers the tool, so a stub that failed
to install would read as a model that declined), the registry holds no collection — the
production cold start — and the media store holds no image, so every image after the turn is
the turn's.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.  Every subject and place is invented, because the repo is
public.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple
from unittest.mock import AsyncMock

import pytest

from penny.penny import Penny
from penny.tests.conftest import ONE_PX_PNG_B64
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    Preparer,
    stored_images,
)
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

# Family tag (explicit, meaningful grouping) — shared with the sibling dispatch stories
# (email, notifications, choose) so the report's families rollup reads chat-surface tool
# dispatch as one group.
_FAMILY = "nl-dispatch"

_GENERATE_IMAGE = "generate_image"

# The image token the ``generate_image`` result hands her (#3033) — the bracket
# form naming the stored media row's id, spelled here rather than imported: the
# case is the contract's wording of the form, and it must run on a tree that
# predates the fix's own constants.
_IMAGE_TOKEN_RE = re.compile(r"\[image (\d+)\]")


def install_image_client(penny: Penny) -> None:
    """Put a canned image backend in front of the real ``generate_image`` tool.

    The stub answers the one network call the tool makes with a one-pixel PNG; storing it,
    describing it and delivering it stay production's.  Installing a client is also what
    REGISTERS the tool — ``ChatAgent`` builds it only when an image client is present — so this
    hook stands the world up rather than merely stubbing a call."""
    client = AsyncMock()
    client.generate_image.return_value = ONE_PX_PNG_B64
    penny.chat_agent._image_client = client


# ── The case shape ───────────────────────────────────────────────────────────


class ImageCase(NamedTuple):
    """One case: the message in five wordings, the facts every wording carries, and the world.

    ``facts`` is what every wording must state, held on the case so the probe that checks the
    wordings and the case cannot disagree about which facts are the constant ones."""

    case_id: str
    behaviour: str
    ask: str
    also_phrased: tuple[str, ...]
    facts: tuple[str, ...]
    world: World


# What the request case's picture must show: the two things the ask puts in it.
SUBJECT = ("dragon", "lighthouse")

_REQUEST = ImageCase(
    case_id="image-request-draws",
    behaviour=(
        "In the chat agent, when the user asks for a picture of something, Penny draws one whose "
        "description names what was asked for."
    ),
    ask="can you draw me a teal origami dragon perched on a lighthouse?",
    also_phrased=(
        "make a picture of a teal origami dragon perched on a lighthouse",
        "i'd love an image of a teal origami dragon perched on a lighthouse",
        "sketch me a teal origami dragon perched on a lighthouse, please",
        "could you whip up an illustration of a teal origami dragon perched on a lighthouse?",
    ),
    facts=("teal origami dragon perched on a lighthouse",),
    # Every field EMPTY, and each is a report rather than an omission: no page answers a
    # request to draw, the turn is asked to keep nothing, and the ask asks for a picture rather
    # than for anything said — so no reply token is owed.
    world=World(name="image-request-draws", pages=(), keeps=(), excludes=()),
)

_NO_FIRE = ImageCase(
    case_id="image-no-fire",
    behaviour=(
        "In the chat agent, when a message talks about a painting without asking for a "
        "picture, Penny says nothing she was not given."
    ),
    ask="i saw a really nice watercolor painting at the gallery today, it was lovely",
    also_phrased=(
        "went to the gallery today and there was this lovely watercolor painting",
        "the watercolor painting i saw at the gallery today was really lovely",
        "spent the afternoon at the gallery, one watercolor painting there was so lovely",
        "there was a lovely watercolor painting at the gallery today, it really stuck with me",
    ),
    facts=("watercolor painting", "gallery"),
    world=World(name="image-no-fire", pages=(), keeps=(), excludes=()),
)

_HANDOVER = ImageCase(
    case_id="image-request-hands-over-token",
    behaviour=(
        "In the chat agent, when the user asks for a picture, Penny's reply carries the "
        "image token the generate_image result named — exactly once, and no other image "
        "placeholder."
    ),
    ask="can you draw a red paper crane sitting on a stack of books?",
    also_phrased=(
        "make a picture of a red paper crane sitting on a stack of books",
        "i'd love an image of a red paper crane sitting on a stack of books",
        "sketch a red paper crane sitting on a stack of books, please",
        "could you draw me a red paper crane sitting on a stack of books?",
    ),
    facts=("red paper crane sitting on a stack of books",),
    world=World(name="image-request-hands-over-token", pages=(), keeps=(), excludes=()),
)

# Every case in this file, in one place — so the deterministic probes in
# ``test_eval_harness.py`` can hold the wordings and the world without a GPU.
IMAGE_CASES = (_REQUEST, _NO_FIRE, _HANDOVER)


# ── The premise, asserted before the turn runs ───────────────────────────────


def assert_image_world(penny: Penny, case_id: str) -> None:
    """The world a case is answered in: ``generate_image`` on the chat surface, no collection
    in the registry, and no image in the media store — so every image after the turn is the
    turn's, and the request case's claim cannot be answered by the seed."""
    assert_dispatch_world(penny, case_id, [_GENERATE_IMAGE])
    held = stored_images(penny.db)
    assert not held, f"{case_id}: the media store must start empty, it holds {held}"


def _prepare(case: ImageCase) -> Preparer:
    """Install the canned image backend, then assert the world it stood up."""

    def prepare(penny: Penny) -> None:
        install_image_client(penny)
        assert_image_world(penny, case.case_id)

    return prepare


async def _drive(chat_eval: ChatEval, model: str, case: ImageCase) -> Cohort:
    """Drive one case: the canned backend installed and probed, the message in five wordings."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        area=Area.CHAT_TOOLS,
        model=model,
        prepare=_prepare(case),
        world=case.world,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


# ── The claim, as a pure function over one sample ────────────────────────────
#
# It reads no tool NAME.  Whether ``generate_image`` fired is a route, and an image-making tool
# a plugin adds tomorrow lands in the same store: what is claimed is the picture the turn left.

_ClaimFn = Callable[[SampleObservation, World], Answer]


def drew(subject: tuple[str, ...]) -> _ClaimFn:
    """The media store holds an image drawn this turn whose description names every part of
    ``subject``.

    Drawn, not captured: a browsed page's image carries its source page and is not a picture
    anyone asked for.  This turn's, because the store starts empty.  Any one drawn image
    answers it, so a turn that drew twice is measured rather than penalised.  Folded through
    ``fold_typography`` — the one definition every probe uses — so case and typography cannot
    hide a subject the description does name."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        drawn = [image.description for image in sample.images if image.drawn]
        shows = any(all(part in fold_typography(text) for part in subject) for text in drawn)
        return shows, f"drew {drawn or 'nothing'}"

    return answer


# ── The handover case's claims (#3033) ───────────────────────────────────────
#
# All three reply claims read ``stored_reply`` — the conversation record, which
# keeps what egress strips from the wire (#2722) — never ``reply``, the wire
# form "as the user saw it".  The strip is deterministic egress machinery the
# mock tests own; what the model decides, and what this case measures, is
# whether she put the token in her reply at all.


def _token_ids(text: str) -> list[str]:
    """The id of every ``[image <id>]`` token in ``text``, in order."""
    return _IMAGE_TOKEN_RE.findall(text)


def _token_once(sample: SampleObservation, _world: World) -> Answer:
    """The stored reply carries the image token form exactly once (#3033).

    Once: the token is how she hands the image over, and a doubled token would
    hand it twice."""
    found = _IMAGE_TOKEN_RE.findall(sample.stored_reply)
    return len(found) == 1, f"tokens {found or 'none'} in the stored reply"


# The placeholder forms she invented when no token was handed to her (#3033's
# why).  The result names the one legitimate token, so any of these standing in
# the reply is a wrong fact rather than a wording choice.  Matched folded —
# ``{image}`` and ``{Image}`` are one form.
_EXCLUDED_PLACEHOLDERS = ("{image}", "[image]", "[image attached]", "[image placeholder]", "![")


def _no_other_placeholder(sample: SampleObservation, _world: World) -> Answer:
    """No other image placeholder stands in the stored reply."""
    text = fold_typography(sample.stored_reply)
    present = [form for form in _EXCLUDED_PLACEHOLDERS if form in text]
    return not present, f"placeholders {present}" if present else "none of the excluded forms"


def _one_drawn_image(sample: SampleObservation, _world: World) -> Answer:
    """The media store holds exactly ONE image drawn this turn for the one request.

    What survives — the store fact the ruling chose over a restraint claim on
    the tool sequence, since a second draw is the defect #3033 fixes.  The world
    starts empty, so every drawn image is this turn's."""
    drawn = [image.description for image in sample.images if image.drawn]
    return len(drawn) == 1, f"drew {drawn or 'nothing'}"


def _token_id_is_sourced(sample: SampleObservation, _world: World) -> Answer:
    """Every id the stored reply's token names traces to what she was given
    (PROVENANCE): the generate_image result names the id, so an invented id
    fails.  A claim over the record, not over ``reply`` — the token is stripped
    from the wire, so the id never appears there."""
    missing = [
        i for i in _IMAGE_TOKEN_RE.findall(sample.stored_reply) if i not in sample.given.content
    ]
    return (
        not missing,
        f"ids {missing} appear nowhere in what she was given" if missing else "in the tool result",
    )


# ═══ asked for ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_request_for_a_picture_draws_it(chat_eval: ChatEval, model: str) -> None:
    """An empty media store, the image tool on the surface, and a request for one picture."""
    cohort = await _drive(chat_eval, model, _REQUEST)
    # STORE — the picture the ask asked for, read off the media store.
    cohort.claim(
        "state: the media store holds an image drawn this turn of a dragon on a lighthouse",
        drew(SUBJECT),
        SpecCategory.STORE,
    )

    # PROVENANCE — the REPLY half.  The STORE half of the collections is absent, and this is a
    # report rather than an omission: the ask keeps nothing, and whether a sample wrote something
    # anyway is measured in ENTRIES_STORED.  The image's description is the open prose the tool
    # asks the model to elaborate, so no provenance claim reads it.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ═══ asked for, handed over ══════════════════════════════════════════════════
#
# The result hands her the token; the claims read what she wrote, not the wire.


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_request_for_a_picture_hands_over_the_token(
    chat_eval: ChatEval, model: str
) -> None:
    """An empty media store, the image tool on the surface, and a request for one
    picture — the result hands her the ``[image <id>]`` token to put in her reply
    (#3033)."""
    cohort = await _drive(chat_eval, model, _HANDOVER)
    # STORE — the one drawn image, the token once, no other placeholder.
    cohort.claim(
        "state: the media store holds exactly one image drawn this turn for the one request",
        _one_drawn_image,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the stored reply carries the image token exactly once",
        _token_once,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: no other image placeholder stands in the stored reply",
        _no_other_placeholder,
        SpecCategory.STORE,
    )
    # PROVENANCE — the token's id traces to the tool result she was given; the
    # visible reply (the wire, token stripped) still invents nothing.
    cohort.claim(
        "provenance: the token's id traces to the generate_image result she was given",
        _token_id_is_sourced,
        SpecCategory.PROVENANCE,
    )
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


# ═══ the topic alone ═════════════════════════════════════════════════════════
#
# Art is the SUBJECT of the message and not the request in it, with the image tool on the
# surface — the temptation is the case.  The harm is a picture nobody asked for, and that is
# the model's call on what it was shown, so it is measured in the tool sequence.


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_remark_about_a_painting_stays_in_idle(chat_eval: ChatEval, model: str) -> None:
    """A painting named, and nothing asked of it.

    Its own setup and its own run, because the request case's whole contract is that a picture
    is drawn — a case is one setup, one run and one set of assertions (#2100)."""
    cohort = await _drive(chat_eval, model, _NO_FIRE)
    # STORE — EMPTY, and a report: the world holds nothing a chat turn could alter (the image
    # store is append-only and the registry starts empty), so there is no prior state for a
    # preservation claim to read.

    # PROVENANCE — the reply half: no painter, gallery or title she was not given.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)
