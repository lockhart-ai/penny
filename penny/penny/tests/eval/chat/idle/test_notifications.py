"""Muting: one case (#2008, tranche 3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**The behaviour is *Penny mutes or unmutes when asked in the tool's own terms, and nothing
else moves*, and one case carries it.**  Muting and unmuting are the same sentence in two
entry conditions — an unmuted world and a muted one — so the design's rule is that one of
them survives, and the MUTE direction strictly dominates: its end state cannot be answered
by the seed.  *The user is muted afterwards* is false of a fresh database by construction, so
a sample that did nothing fails it; *the user is no longer muted* is TRUE of an unseeded world,
so the unmute case's headline rests entirely on its own seed holding — the failure its source
file guards against in as many words.  A claim that cannot pass without the turn acting beats
one that can, so ``explicit-unmute-request-unmutes`` is QUARANTINED here rather than deleted,
and comes back the day the lift direction is the thing being measured.

**The no-fire direction is STATED, never a case.**  ``notifications-no-fire`` — a casual
mention of a quiet day, which must move nothing — is #2008's ruling as the negative direction
of this one behaviour rather than a behaviour of its own.  It cannot be an ARM either: an arm
is one wording of an ask whose expected end state is the case's, and this one expects the
opposite.  So the sentence states both directions and the case carries the negative as a claim
its own world CAN answer — nothing but the notification switch moved — which is the same harm
(acting where nobody asked) read off the world where an ask does exist.

Recorded as a doc gap, the same one tranche 1 recorded: #2004 says a negative direction with
its own expected outcome is its own case, and #2008 rules the opposite for this pair.  What is
genuinely lost here is narrower than it looks and is worth naming: the harm the no-fire case
catches is *the switch moving when nobody asked*, and in a world where somebody DID ask, the
switch moving is the correct answer — so that reading has no home in this case at all.

**Dispatch stands on the tool descriptions ALONE.**  Migration 0076 seeded a "Mute or unmute
notifications" skill whose numbered steps taught this routing; 0092 deleted every seeded rule
entry and 0097 the collection itself, and 0108 leaves nothing pre-seeded at all — so this case
seeds no skill, the registry the turn runs against is empty, and what it measures is whether
``NotificationsMuteTool`` is reachable from an explicit ask with no recipe pointing at it.  The
seeded world is already the production cold start, which is what makes *nothing was created* a
TOTAL reading of what the turn touched rather than a sample of it.

**The state has to be IN FRONT OF THE MODEL, and that is a PREMISE rather than a claim.**
Whether notifications are muted is rendered ambiently by ``SelfStateHeader`` (#1919), and that
is half of what a muting turn stands on: before it, the tool descriptions were the only carrier
of a state the model could not verify.  The probe renders the header itself, before the turn,
so a header that stopped carrying the line fails naming the line rather than leaving fifteen
samples scoring a turn that never saw the state.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.conversation_machine import ConversationState
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)
from penny.tests.eval.utils.dispatch_world import assert_dispatch_world
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) — shared with the sibling dispatch stories
# (email, generate_image, choose) so the report's families rollup reads chat-surface tool
# dispatch as one group.
_FAMILY = "nl-dispatch"

_MUTE = "notifications_mute"
_UNMUTE = "notifications_unmute"

_CASE_ID = "explicit-mute-request-mutes"

# The world every arm is answered against.  Every field is EMPTY, and each is a report rather
# than an omission: no page can answer an ask about Penny's own notification switch, so a page
# set would only give a sample that went browsing something to talk about; the turn is asked to
# write nothing, so a ``keeps`` set would state a contract the ask never made and an
# ``excludes`` set would assert one reading of an ask that rules nothing out; and the ask is an
# INSTRUCTION, so "done — you won't hear from me until you say otherwise" is a complete answer
# and requiring a token would fail a correct run for something nobody requested.
_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=())

_ASK = "please mute notifications"
_ALSO_PHRASED = (
    "mute notifications for me",
    "can you mute notifications?",
    "go ahead and mute notifications",
    "i'd like notifications muted",
)

_BEHAVIOUR = (
    "In the chat agent, when the user asks in the notification tool's own terms for "
    "notifications to be muted, Penny mutes them and moves nothing else — no collection "
    "created, changed or written to — while a passing remark that merely mentions how quiet "
    "things have been moves nothing at all."
)


def assert_mute_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs.

    Three things, each the precondition of a claim below: the chat surface really carries the
    two notification tools (a scored dispatch miss against a model that was never offered the
    tool is the failure this forecloses); the registry holds no COLLECTION, which is what makes
    *nothing was created* a total reading; and notifications are ON — both in the store, so the
    claim that they are muted afterwards cannot be answered by the seed, and in the header the
    model reads, so the turn is answered by something that says which way the switch is set."""
    assert_dispatch_world(penny, _CASE_ID, [_MUTE, _UNMUTE])
    assert not penny.db.users.is_muted(TEST_SENDER), (
        f"{_CASE_ID}: the user must start unmuted, or the claim that this turn muted them is "
        "answered by the seed"
    )
    rendered = SelfStateHeader(penny.db, TEST_SENDER).render()
    assert SelfStateHeader.NOTIFICATIONS_ON in rendered, (
        f"{_CASE_ID}: the header must state {SelfStateHeader.NOTIFICATIONS_ON!r} — the turn "
        f"would otherwise be answered with nothing saying which way the switch is set:\n"
        f"{rendered}"
    )


# ── The claims, as pure functions over one sample ─────────────────────────────
#
# None of them reads a tool NAME.  Whether ``notifications_mute`` fired is a ROUTE — many
# routes reach one end state, and a plugin could add a third verb over the same row tomorrow —
# so it is measured in the tool sequence and never asserted; what the case claims is the switch
# the turn left behind.


def _notifications_are_muted(sample: SampleObservation, _world: World) -> Answer:
    """Proactive notifications are muted for the user afterwards.

    A violating sample is nameable: one that says it has muted them and leaves the row absent —
    the failure the whole retirement of ``/mute`` rests on, since the reply is the only place
    the user learns anything happened."""
    return sample.muted, "nothing was muted, whatever the reply said"


def _nothing_else_moved(sample: SampleObservation, _world: World) -> Answer:
    """No mechanism was created, retired or edited, and nothing was written anywhere.

    The negative direction, read off the world where an ask DOES exist: the harm a passing
    remark risks is acting on something nobody asked for, and this is that harm asked about
    everything except the one switch the ask named.  A violating sample is nameable: one that
    stands a container up to keep a note about being muted, one that files "muted at 14:02"
    into a list, and one that reaches into a mechanism on the way through."""
    touched = sorted(
        one.name for one in sample.mechanisms if one.born_this_run or one.changed_this_run
    )
    wrote = sorted(f"{entry.collection}:{entry.key}" for entry in sample.entries)
    return not touched and not wrote, f"created or changed {touched}, wrote {wrote}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_explicit_mute_request_mutes_and_moves_nothing_else(
    chat_eval: ChatEval, model: str
) -> None:
    """An unmuted world, a header that says so, and a request in the tool's own terms."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        prepare=assert_mute_world,
        world=_WORLD,
        ask=_ASK,
        also_phrased=_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the switch the ask named, then its negative direction.
    cohort.claim(
        "state: notifications are muted for the user", _notifications_are_muted, SpecCategory.STORE
    )
    cohort.claim(
        "state: nothing but the notification switch moved", _nothing_else_moved, SpecCategory.STORE
    )

    # PROVENANCE — the half the source case had none of.  A turn that filed a fact into a
    # collection nobody's world mentions fails the first; a reply that states a value tracing to
    # nothing the model was given fails the second.
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — the call is a route — and on this case it
    # reads BLIND, which the report renders in red.  The chat observation narrows a sample's
    # calls to ``ENACTING_TOOLS``, a six-name whitelist carrying no notification verb, so every
    # sample reads "no call" whether the mute tool fired or not.  Measured anyway, because a
    # feature that prints a number either way has to say which of the two it is — and because
    # the blindness is a fact about the INSTRUMENT rather than about the cohort.  What separates
    # the directions is read off the modal sample until that whitelist is settled; it is
    # reported as a harness gap rather than worked around here, since a second copy of "which
    # calls count" is a second policy.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
