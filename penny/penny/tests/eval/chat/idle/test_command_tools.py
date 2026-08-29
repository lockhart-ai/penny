"""NL-dispatch story: a request to draw makes a picture, a mention of art does not (#1445).

The retired ``/draw`` command became a tool the model reaches from plain language, so the
story is one sentence with two directions:

  * a request to DRAW something ("can you draw me X", "make a picture of X") dispatches to
    ``generate_image`` with a description carrying the subject the user asked for; and
  * a REMARK about art ("i saw a nice watercolor at the gallery") reaches nothing, and the
    reply does not claim a picture was made.

**Dispatch stands on the tool description ALONE.**  No skill teaches this routing — nothing
is pre-seeded since migration 0108 — so these cases seed none: the world they measure is a
fresh deployment's, where the model reads ``generate_image``'s own description and decides.
The loud probe below asserts that world out loud rather than trusting it, because the same
hook that stubs the image model is what REGISTERS the tool (``ChatAgent`` builds it only
when an image client is present), and a hook that failed to install would score every
sample a dispatch miss the model was never offered.

**The conversation state machine fronts every driven turn** (#1706): it classifies before
the chat agent runs, and a draw request lands in whatever state it lands in.  What is
scored here is the chat turn's DISPATCH, never the state it landed in.

Scoring is STRUCTURAL — the persisted call and its description, and the store read before
and after — with one narrow REPLY floor on the no-fire direction, where saying a picture
was drawn is a claim the record contradicts (the visible-degradation rule applied to what
she tells the user).  How well she answered is read at joint review against each case's
``reference`` reply, which is DATA rather than a comment so the deterministic pin in
``test_eval_harness.py`` can run it through this module's own vocabulary without a GPU.

Report-only (``min_pass_rate=None``): a live-model dispatch rate is a number to read, and
the threshold is the code owner's to set once the numbers are read.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple
from unittest.mock import AsyncMock

import pytest

from penny.penny import Penny
from penny.tests.conftest import ONE_PX_PNG_B64
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Preparer,
    Scorer,
    last_tool_args,
    new_collections,
    routing_clean,
    tool_call_sequence,
    tool_not_called,
)
from penny.tests.eval.utils.dispatch_world import assert_dispatch_world

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module — shared with
# the sibling dispatch stories (email, choose) so the report's families rollup reads
# chat-surface tool dispatch as one group.
_FAMILY = "nl-dispatch"

_GENERATE_IMAGE = "generate_image"

# The tool's one required argument, named once since three checks read it.
_DESCRIPTION = "description"


def install_image_client(penny: Penny) -> None:
    """Wire a mocked image client so ``generate_image`` REGISTERS and its boundary call is
    a no-op returning a canned PNG.

    Installing the client is what puts the tool on the surface at all — ``ChatAgent`` gates
    registration on it — so this hook stands the world up rather than merely stubbing a
    network call."""
    client = AsyncMock()
    client.generate_image.return_value = ONE_PX_PNG_B64
    penny.chat_agent._image_client = client


# ── What a reply CLAIMING a picture was made says ─────────────────────────────
#
# Deliberately NARROW, and the only reply reading in the module: this is the one reply
# failure with a structural answer — no image was generated, so saying one was drawn is a
# claim the record contradicts.  Every entry is a COMPLETED act, in a form an offer cannot
# produce ("want me to draw one?" carries none of them), because the future tense is where
# a wide vocabulary would start failing correct answers: "i'll draw inspiration from that"
# is an ordinary sentence, and catching it would make the check mean nothing.
#
# The tripwire is each case's own ``reference`` reply, run through this set by the pin in
# ``test_eval_harness.py``: a vocabulary that cannot pass the answer the case calls correct
# would score every sample a miss, on a GPU, an hour later.
_CLAIMS_A_PICTURE_WAS_MADE = (
    "here's the picture",
    "here's the image",
    "here's your picture",
    "here's what i drew",
    "here's my take",
    "i drew",
    "i've drawn",
    "i painted",
    "i've painted",
    "i sketched",
    "i've sketched",
    "i made you a picture",
    "i made a picture",
    "i generated",
    "i've generated",
    "i whipped up",
)


class _ImageCase(NamedTuple):
    """One agreed message, and what the turn it opens has to look like.

    ``subject`` is the salient token of the thing the user asked to see — what a faithful
    description must carry — or ``None`` for the no-fire direction, which is the
    declaration that this message asks for no picture at all.

    ``reference`` is how the message would be answered WELL: a review target, read at joint
    review and never matched by the scorer.  It is DATA rather than a comment so the
    deterministic pin can run it through this module's reply vocabulary without a GPU — a
    scorer that cannot pass the answer the case itself calls correct is a broken scorer,
    and that is cheaper to find here than on the queue."""

    case_id: str
    message: str
    subject: str | None
    reference: str


_DRAW_REQUEST = _ImageCase(
    case_id="tool-generate-image-draw",
    message="can you draw me a teal origami dragon perched on a coffee mug?",
    subject="dragon",
    reference="here you go — one teal origami dragon, folded and sitting on a mug.",
)

_MAKE_A_PICTURE = _ImageCase(
    case_id="tool-generate-image-picture",
    message="make a picture of a neon cactus wearing tiny sunglasses",
    subject="cactus",
    reference="done — a neon cactus in tiny shades, glowing away.",
)

_GALLERY_MENTION = _ImageCase(
    case_id="tool-generate-image-nofire",
    message="i saw a really nice watercolor painting at the gallery today, it was lovely",
    subject=None,
    reference="oh nice, what was it of? want me to try something in that style?",
)

IMAGE_CASES = (_DRAW_REQUEST, _MAKE_A_PICTURE, _GALLERY_MENTION)


# ── The loud probe: the image tool really is on the surface ───────────────────


def assert_image_world(penny: Penny, case: _ImageCase) -> None:
    """Everything this case's world is responsible for, asserted out loud: ``generate_image``
    is registered, and the registry holds no collection.

    Both claims are the shared dispatch-world probe (``dispatch_world``) — the registry half
    reads COLLECTION-shaped memories only, since the four migration-0026 system log markers
    are in every database and a probe that counted them could never pass."""
    assert_dispatch_world(penny, case.case_id, [_GENERATE_IMAGE])


def _probe_image_world(case: _ImageCase) -> Preparer:
    """Install the mocked image client, then assert the world it stood up."""

    def prepare(penny: Penny) -> None:
        install_image_client(penny)
        assert_image_world(penny, case)

    return prepare


# ── Checks ────────────────────────────────────────────────────────────────────


def _drew_check(db) -> Check:
    """The headline: the request reached the image tool at all.

    Anchored to the call itself, so the verdict sits on the row that made it and a miss
    falls to the run-close table where a missing action belongs."""
    args = last_tool_args(db, _GENERATE_IMAGE)
    fired = args is not None
    return Check(
        "calls: the request reached generate_image",
        fired,
        anchor=f"{_GENERATE_IMAGE}(",
        rationale=None if fired else f"the turn fired {tool_call_sequence(db) or 'nothing'}",
        kind="spine",
    )


def _description_is_real_check(db) -> Check:
    """The call carries a description at all — the floor under the subject check, and its
    own finding: an empty description is a call that fired and drew nothing anyone asked
    for.  N/A when nothing fired, so one failure is reported once."""
    label = "calls: the call carries a description"
    args = last_tool_args(db, _GENERATE_IMAGE)
    if args is None:
        return Check.na(label, anchor=f"{_GENERATE_IMAGE}(", kind="spine")
    described = str(args.get(_DESCRIPTION) or "").strip()
    return Check(
        label,
        bool(described),
        anchor=f"{_GENERATE_IMAGE}(",
        rationale=None if described else f"it passed {args!r}",
        kind="spine",
    )


def _subject_survived_check(db, case: _ImageCase) -> Check:
    """The description is about the thing the USER asked to see — the half that makes
    dispatch worth anything, since a picture of something else is a call that fired and
    pleased nobody.  N/A when nothing fired."""
    label = f"calls: the description is about the {case.subject!r}"
    args = last_tool_args(db, _GENERATE_IMAGE)
    if args is None:
        return Check.na(label, anchor=f"{_GENERATE_IMAGE}(", kind="spine")
    described = str(args.get(_DESCRIPTION) or "")
    carried = (case.subject or "") in described.lower()
    return Check(
        label,
        carried,
        anchor=f"{_GENERATE_IMAGE}(",
        rationale=None if carried else f"it drew {described!r}",
        kind="spine",
    )


def _no_image_call_check(db) -> Check:
    """The image tool did not fire — the no-fire direction's headline, with the rationale
    naming the turn's calls so a miss reads as what happened rather than as a bare red."""
    quiet = tool_not_called(db, _GENERATE_IMAGE)
    return Check(
        "calls: generate_image did not fire on a mention of art",
        quiet,
        rationale=None if quiet else f"it drew {last_tool_args(db, _GENERATE_IMAGE)!r}",
        kind="spine",
    )


def _store_untouched_check(db, before: set[str]) -> Check:
    """Nothing was created.  A drawn image is delivered by id to its own reply and never
    lands in the registry, and this world holds no collection at all — so a collection
    appearing is the whole "nothing else was touched" claim rather than a sample of it."""
    created = sorted(row.name for row in new_collections(db, before))
    return Check(
        "state: nothing was created",
        not created,
        rationale=f"created {created}" if created else None,
        kind="state",
    )


def _claims_no_picture_check(reply: str) -> Check:
    """The reply does not say a picture was made when nothing made one.

    A FLOOR, deliberately narrow: everything else about the answer — whether it engaged
    with the painting, whether the offer was a good one — is read at joint review against
    the case's reference reply, and an answer that just chats about the gallery is not a
    miss."""
    claimed = [phrase for phrase in _CLAIMS_A_PICTURE_WAS_MADE if phrase in reply.lower()]
    return Check(
        "reply: it claims no picture was made",
        not claimed,
        anchor=REPLY_ANCHOR,
        rationale=f"said {claimed}" if claimed else None,
        kind="reply",
    )


def _dispatch_advisories(db, reply: str) -> list[Check]:
    """What the turn actually did, verbatim and UNSCORED — the calls it made and the answer
    it gave — so a report shows the turn whichever way it went and the wording is read
    where wording is read: at joint review."""
    return [
        Check(f"fired: {tool_call_sequence(db)}", True, kind="proc", scored=False),
        Check(f"answered: {reply!r}", True, kind="reply", scored=False),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


# ── Scorers ───────────────────────────────────────────────────────────────────


def _score_draw_request(db, before: set[str], reply: str, *, case: _ImageCase) -> list[Check]:
    """The request reached the image tool, carrying a real description of what was asked
    for, and the turn touched nothing else.

    No reply check on this direction: the picture is delivered by id to its own reply, so
    what she says about it is read at joint review against the case's reference."""
    return [
        _drew_check(db),
        _description_is_real_check(db),
        _subject_survived_check(db, case),
        _store_untouched_check(db, before),
        *_dispatch_advisories(db, reply),
    ]


def _score_art_mention(db, before: set[str], reply: str) -> list[Check]:
    """The mention reached nothing, nothing was created, and the reply does not claim a
    picture that was never made.

    Every check is a negative, because the failure this direction exists to catch is firing
    anything at all."""
    return [
        _no_image_call_check(db),
        _store_untouched_check(db, before),
        _claims_no_picture_check(reply),
        *_dispatch_advisories(db, reply),
    ]


async def _run_image_case(chat_eval: ChatEval, case: _ImageCase) -> None:
    """Drive one image case: the mocked client installed and probed, the scorer bound to
    the case's own subject.  Report-only — the threshold is the code owner's to set once
    the numbers are read."""
    score: Scorer = (
        _score_art_mention if case.subject is None else partial(_score_draw_request, case=case)
    )
    await chat_eval(
        case_id=case.case_id,
        family=_FAMILY,
        message=case.message,
        prepare=_probe_image_world(case),
        score=score,
        min_pass_rate=None,
    )


async def test_draw_request_dispatches(chat_eval: ChatEval) -> None:
    """ "can you draw me X" — the explicit request, where the subject is several words long
    and a description that keeps only the style would still look like it worked."""
    await _run_image_case(chat_eval, _DRAW_REQUEST)


async def test_make_a_picture_dispatches(chat_eval: ChatEval) -> None:
    """ "make a picture of X" — the same request written as a making rather than a drawing,
    so the dispatch cannot be keyed to one verb."""
    await _run_image_case(chat_eval, _MAKE_A_PICTURE)


async def test_casual_art_mention_does_not_dispatch(chat_eval: ChatEval) -> None:
    """A remark about a painting someone else made asks for no picture — the over-firing
    direction, where the topic is art and the request is not."""
    await _run_image_case(chat_eval, _GALLERY_MENTION)
