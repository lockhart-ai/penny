"""Muting and unmuting: three cases (#2008, tranche 3; #2215).

The contract is `docs/eval-case-design.md`.

**Three cases, because one setup cannot produce more than one of them.**  A case is one setup,
one run and one set of assertions (#2100).  The harm the negative direction catches is *the
switch moving when nobody asked* — which in a world where somebody DID ask is the correct
answer — and the two directions of an ask start from opposite worlds.

| direction | case | the world it starts in | what the run must leave |
|---|---|---|---|
| asked to mute | ``explicit-mute-request-mutes`` | notifications on | notifications muted |
| asked to unmute | ``explicit-unmute-request-unmutes`` | notifications MUTED | notifications on |
| the topic alone | ``notifications-no-fire`` | notifications on | still on, as it found them |

**The unmute case stands on its SEED, and the premise holds the seed.**  *The user is muted
afterwards* is false of a fresh database, so a mute sample that did nothing fails it with no
help from anyone.  *The user is not muted afterwards* is TRUE of a fresh database, so that
claim says something only on a world that started muted — and only when the model was told so.
Its probe therefore asserts both halves before the turn runs: the mute row is in the store,
and the self-state header states the muted line.  A seed that stopped holding fails there,
naming what it lost, rather than scoring fifteen samples green for a turn that had nothing to
undo.

**Dispatch stands on the tool descriptions ALONE.**  Migration 0076 seeded a "Mute or unmute
notifications" skill whose numbered steps taught this routing; 0092 deleted every seeded rule
entry and 0097 the collection itself, and 0108 leaves nothing pre-seeded at all — so no case
here seeds a skill, the registry the turn runs against is empty, and what the two asks measure
is whether ``NotificationsMuteTool`` and ``NotificationsUnmuteTool`` are reachable from an
explicit ask with no recipe pointing at them.  The mute and no-fire worlds are the production
cold start; the unmute world is that cold start with the one switch thrown.

**No case claims what the turn did NOT do** (``docs/principles.md`` §4.3).  The world holds
no collection, so the one prior state a preservation claim can read is the switch itself —
which is the no-fire case's whole claim.  Whether a turn also wrote or stood something up is
its own call, measured in the entries stored and the tool sequence.

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
from penny.database import Database
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.catalogue import Area
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
_UNMUTES = "explicit-unmute-request-unmutes"
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
    "In the chat agent, when the user asks for notifications to be muted, Penny mutes them."
)


def assert_mute_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs.

    Three things, each the precondition of a claim below: the chat surface really carries the
    two notification tools (a scored dispatch miss against a model that was never offered the
    tool is the failure this forecloses); the registry holds no COLLECTION, the production cold
    start; and notifications are ON — both in the store, so the claim that they are muted
    afterwards cannot be answered by the seed, and in the header the model reads, so the turn is
    answered by something that says which way the switch is set."""
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


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_explicit_mute_request_mutes_notifications(
    chat_eval: ChatEval, model: str
) -> None:
    """An unmuted world, a header that says so, and a request for notifications to be muted."""
    cohort: Cohort = await chat_eval(
        case_id=_MUTES,
        behaviour=_BEHAVIOUR,
        area=Area.NOTIFICATIONS,
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

    # STORE — the switch the ask named.  The world holds no collection, so there is nothing
    # else for a preservation claim to read.
    cohort.claim(
        "state: notifications are muted for the user", _notifications_are_muted, SpecCategory.STORE
    )

    # PROVENANCE — the REPLY half.  The STORE half is absent, and this is a report rather than an
    # omission: the ask keeps nothing, and whether a sample wrote something anyway is measured
    # in ENTRIES_STORED.
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — the call is a route, and which verb reached
    # the switch is hers.  What a divergence here means is a sample that took a different path to
    # the same end state, which is a sample worth opening rather than a claim worth making.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ═══ asked to unmute ═════════════════════════════════════════════════════════
#
# The same ask in the other direction, from the other world: notifications are MUTED when the
# turn begins, and the user asks for them back.

# Empty in every field, for the mute world's reasons: no page answers an ask about her own
# switch, the ask keeps and excludes nothing, and "done — they're back on" is a complete answer.
_UNMUTE_WORLD = World(name=_UNMUTES, pages=(), keeps=(), excludes=())

# The mute case's five wordings with the direction turned round, so the pair differs in the
# entry condition and the direction asked for and in nothing else.
_UNMUTE_ASK = "please unmute notifications"
_UNMUTE_ALSO_PHRASED = (
    "unmute notifications for me",
    "can you unmute notifications?",
    "go ahead and unmute notifications",
    "i'd like notifications unmuted",
)

_UNMUTE_BEHAVIOUR = (
    "In the chat agent, when notifications are muted and the user asks for them to be "
    "unmuted, Penny unmutes them."
)


def seed_muted(db: Database) -> None:
    """The unmute case's entry condition: proactive notifications muted for the user, through
    the store's own write — the row ``notifications_mute`` leaves."""
    db.users.set_muted(TEST_SENDER)


def assert_unmute_world(penny: Penny) -> None:
    """The world the unmute case is answered in, asserted out loud before the turn runs.

    The mute world's three preconditions with the switch the other way: the chat surface
    carries the two notification tools; the registry holds no collection; and notifications
    are MUTED — in the store, because "not muted afterwards" is true of a database nobody
    muted and the claim would be answered by the absence of a seed, and in the header the
    model reads, because a turn that was never told about a mute has nothing to lift."""
    assert_dispatch_world(penny, _UNMUTES, [_MUTE, _UNMUTE])
    assert penny.db.users.is_muted(TEST_SENDER), (
        f"{_UNMUTES}: the user must start MUTED, or the claim that they are not muted "
        "afterwards is true of a world nobody seeded"
    )
    rendered = SelfStateHeader(penny.db, TEST_SENDER).render()
    assert SelfStateHeader.NOTIFICATIONS_MUTED in rendered, (
        f"{_UNMUTES}: the header must state {SelfStateHeader.NOTIFICATIONS_MUTED!r} — the "
        f"turn would otherwise be asked to lift a mute nothing told it about:\n{rendered}"
    )


def _notifications_are_on(sample: SampleObservation, _world: World) -> Answer:
    """Proactive notifications are on for the user afterwards — the mute row is gone.

    A violating sample is nameable: one that says they are back on and leaves the row in
    place, so the jobs stay silent while the user has been told otherwise."""
    return not sample.muted, "notifications are still muted, whatever the reply said"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_explicit_unmute_request_unmutes_notifications(
    chat_eval: ChatEval, model: str
) -> None:
    """A muted world, a header that says so, and a request for notifications to be unmuted."""
    cohort: Cohort = await chat_eval(
        case_id=_UNMUTES,
        behaviour=_UNMUTE_BEHAVIOUR,
        area=Area.NOTIFICATIONS,
        model=model,
        seed=seed_muted,
        prepare=assert_unmute_world,
        world=_UNMUTE_WORLD,
        ask=_UNMUTE_ASK,
        also_phrased=_UNMUTE_ALSO_PHRASED,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the switch the ask named, and the only prior state this world holds.  It is a
    # claim rather than a restatement of the seed because ``assert_unmute_world`` holds the
    # seed: every sample this is answered about began muted and was told so.
    cohort.claim(
        "state: notifications are on for the user", _notifications_are_on, SpecCategory.STORE
    )

    # PROVENANCE — the REPLY half; the STORE half is absent for the mute case's reason.
    cohort.assert_every_value_in_the_reply_is_sourced()

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
    "them to be changed, Penny stays in idle with notifications still on."
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


def _notifications_are_still_on(sample: SampleObservation, _world: World) -> Answer:
    """Proactive notifications are still ON for the user.

    The discriminating claim, and the one the fire direction cannot make: there, the switch
    moving IS the answer.  A violating sample is nameable: one that reads a remark ABOUT
    notifications as a request to change them and silences everything the user was thanking her
    for."""
    return not sample.muted, "the turn muted notifications off a remark that asked for nothing"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_remark_about_notifications_leaves_them_on(chat_eval: ChatEval, model: str) -> None:
    """Notifications named, and nothing asked of them.

    Its own setup and its own run, because the harm is the switch moving when nobody asked and
    the fire direction's whole contract is that it moves — a case is one setup, one run and one
    set of assertions (#2100)."""
    cohort: Cohort = await chat_eval(
        case_id=_NO_FIRE,
        behaviour=_NO_FIRE_BEHAVIOUR,
        area=Area.NOTIFICATIONS,
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

    # STORE — the switch the world started with, still where it was: PRESERVATION of the one
    # prior state this world holds.
    cohort.claim(
        "state: notifications are still on, as the turn found them",
        _notifications_are_still_on,
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply half only; the STORE half is absent for the fire direction's reason.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
