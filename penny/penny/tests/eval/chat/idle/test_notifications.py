"""Muting: two cases (#2008, tranche 3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**Two cases, because one setup cannot produce both directions.**  A case is one setup, one run
and one set of assertions (#2100), and the harm the negative direction catches is *the switch
moving when nobody asked* — which in a world where somebody DID ask is the correct answer.
There is no run that exhibits both, so the two asks get two setups.

| direction | case | what the run must leave |
|---|---|---|
| asked in the tool's own terms | ``explicit-mute-request-mutes`` | notifications muted |
| the topic alone | ``notifications-no-fire`` | the switch exactly where it was |

**Muting and unmuting are the same sentence in two entry conditions** — an unmuted world and a
muted one — so one of them survives, and the MUTE direction strictly dominates: its end state
cannot be answered by the seed.  *The user is muted afterwards* is false of a fresh database by
construction, so a sample that did nothing fails it; *the user is no longer muted* is TRUE of an
unseeded world, so the unmute case's headline rests entirely on its own seed holding — the
failure its source file guards against in as many words.  A claim that cannot pass without the
turn acting beats one that can, so ``explicit-unmute-request-unmutes`` is QUARANTINED here
rather than deleted, and comes back the day the lift direction is the thing being measured.

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

_MUTES = "explicit-mute-request-mutes"
_NO_FIRE = "notifications-no-fire"

# The world every arm is answered against.  Every field is EMPTY, and each is a report rather
# than an omission: no page can answer an ask about Penny's own notification switch, so a page
# set would only give a sample that went browsing something to talk about; the turn is asked to
# write nothing, so a ``keeps`` set would state a contract the ask never made and an
# ``excludes`` set would assert one reading of an ask that rules nothing out; and the ask is an
# INSTRUCTION, so "done — you won't hear from me until you say otherwise" is a complete answer
# and requiring a token would fail a correct run for something nobody requested.
_WORLD = World(name=_MUTES, pages=(), keeps=(), excludes=())

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
    "created, changed or written to."
)


def assert_mute_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs.

    Three things, each the precondition of a claim below: the chat surface really carries the
    two notification tools (a scored dispatch miss against a model that was never offered the
    tool is the failure this forecloses); the registry holds no COLLECTION, which is what makes
    *nothing was created* a total reading; and notifications are ON — both in the store, so the
    claim that they are muted afterwards cannot be answered by the seed, and in the header the
    model reads, so the turn is answered by something that says which way the switch is set."""
    assert_dispatch_world(penny, _MUTES, [_MUTE, _UNMUTE])
    assert not penny.db.users.is_muted(TEST_SENDER), (
        f"{_MUTES}: the user must start unmuted, or the claim that this turn muted them is "
        "answered by the seed"
    )
    rendered = SelfStateHeader(penny.db, TEST_SENDER).render()
    assert SelfStateHeader.NOTIFICATIONS_ON in rendered, (
        f"{_MUTES}: the header must state {SelfStateHeader.NOTIFICATIONS_ON!r} — the turn "
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

    A muting turn is asked for one switch and nothing else, so everything the registry holds is
    out of scope for it.  A violating sample is nameable: one that stands a container up to keep
    a note about being muted, one that files "muted at 14:02" into a list, and one that reaches
    into a mechanism on the way through."""
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
        case_id=_MUTES,
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

    # STORE — the switch the ask named, then everything it must leave alone.
    cohort.claim(
        "state: notifications are muted for the user", _notifications_are_muted, SpecCategory.STORE
    )
    cohort.claim(
        "state: nothing but the notification switch moved", _nothing_else_moved, SpecCategory.STORE
    )

    # PROVENANCE — the STORE half is deliberately absent, and this is a report rather than an
    # omission: the case already claims the turn writes nothing, so no stored entry exists for
    # ``assert_every_stored_entry_traces_to_the_world`` to trace and it would answer green on
    # every sample by construction — a guaranteed row in the denominator, measuring the claim
    # beside it rather than the turn.  The REPLY half carries the category.
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — the call is a route, and which verb reached
    # the switch is hers.  What a divergence here means is a sample that took a different path to
    # the same end state, which is a sample worth opening rather than a claim worth making.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ═══ the topic alone ═════════════════════════════════════════════════════════
#
# Notifications are the SUBJECT of the message and not the request in it.  The harm this case
# exists to catch is the switch moving when nobody asked, which is the one reading the fire
# direction structurally cannot produce — there, the switch moving is the correct answer.

_NO_FIRE_WORLD = World(name=_NO_FIRE, pages=(), keeps=(), excludes=())

# Five wordings of one remark, and every one of them NAMES notifications: a message that merely
# mentioned a quiet day would leave the tool with nothing to be tempted by, and the temptation
# is the case.  What varies is only how a person says it — the subject, the approval and the
# absence of any request are constant.
_REMARK = "your notifications have been really useful this week, thanks"
_REMARK_ALSO_PHRASED = (
    "the notifications you've been sending are landing at a good time, just so you know",
    "i've been getting a decent number of notifications lately and they're handy",
    "notifications have been pretty on point recently",
    "i like the notifications — they've been useful this week",
)

_NO_FIRE_BEHAVIOUR = (
    "In the chat agent, when a message names notifications as its subject without asking for "
    "them to be changed, Penny leaves the switch exactly where she found it and moves nothing "
    "else — no collection created, changed or written to."
)


def assert_no_fire_world(penny: Penny) -> None:
    """The same world the fire case is answered in, asserted under this case's own id.

    Notifications must be ON before the turn, which is what makes the claim below a statement
    about this turn: on a world that started muted, "still unmuted" would be false for a reason
    that has nothing to do with the remark, and on one where nothing rendered the switch there
    would be no lever for a sample to reach for wrongly — so the temptation this case measures
    would be absent rather than declined."""
    assert_dispatch_world(penny, _NO_FIRE, [_MUTE, _UNMUTE])
    assert not penny.db.users.is_muted(TEST_SENDER), (
        f"{_NO_FIRE}: the user must start unmuted, or the claim that the switch did not move "
        "cannot tell a declined temptation from a world that was already quiet"
    )
    rendered = SelfStateHeader(penny.db, TEST_SENDER).render()
    assert SelfStateHeader.NOTIFICATIONS_ON in rendered, (
        f"{_NO_FIRE}: the header must state {SelfStateHeader.NOTIFICATIONS_ON!r} — a lever the "
        f"turn cannot see is one it cannot wrongly reach for:\n{rendered}"
    )


def _the_switch_did_not_move(sample: SampleObservation, _world: World) -> Answer:
    """Proactive notifications are still ON for the user.

    The discriminating claim, and the one the fire direction cannot make: there, the switch
    moving IS the answer.  A violating sample is nameable: one that reads a remark ABOUT
    notifications as a request to change them and silences everything the user was thanking her
    for."""
    return not sample.muted, "the turn muted notifications off a remark that asked for nothing"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_remark_about_notifications_moves_the_switch_nowhere(
    chat_eval: ChatEval, model: str
) -> None:
    """Notifications named, and nothing asked of them.

    Its own setup and its own run, because the harm is the switch moving when nobody asked and
    the fire direction's whole contract is that it moves — a case is one setup, one run and one
    set of assertions (#2100)."""
    cohort: Cohort = await chat_eval(
        case_id=_NO_FIRE,
        behaviour=_NO_FIRE_BEHAVIOUR,
        model=model,
        prepare=assert_no_fire_world,
        world=_NO_FIRE_WORLD,
        ask=_REMARK,
        also_phrased=_REMARK_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the switch, then everything else the remark must leave alone.
    cohort.claim(
        "state: the notification switch is where the turn found it",
        _the_switch_did_not_move,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: nothing but the notification switch moved", _nothing_else_moved, SpecCategory.STORE
    )

    # PROVENANCE — the reply half only.  The STORE half is absent and this is a report rather
    # than an omission: the case claims the turn writes nothing, so no stored entry exists for
    # ``assert_every_stored_entry_traces_to_the_world`` to trace and it would answer green on
    # every sample by construction.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
