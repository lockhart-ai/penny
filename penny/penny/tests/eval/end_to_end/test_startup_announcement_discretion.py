"""Startup announcement discretion: the delivery turn decides for herself
whether, how, and how many times to reply, over a fact rather than an
instruction (#2974).

**The world.** A synthetic "open tickets" collection — her own backlog of
rough edges the user has reported — holds three still-open entries. One
deploy's own real fix resolves exactly one of them; the other two are
untouched by it. The real startup delivery turn (``ChatAgent.
handle_changelog_report``) runs alone, with no follow-up turn: this case
measures the delivery turn's OWN discretion, not a later turn's recall of it.

**Wording.** There is no user-composed ask to word five ways — the trigger is
the framework's own deterministic render of the deploy's entries, the same
turn-to-turn (#2974's one render function, live and in history) — so this
case drives one arm, ``samples`` times, for variance alone.

**What is claimed.** Nothing invented (the generic value-sourcing claim) and
every OTHER open ticket survives untouched — the one the fix actually
resolves is excluded from that claim on purpose, since touching it is the
desired outcome, not a violation of it.

**What is measured, never claimed.** Two rates, since nothing in the new
trigger tells her to reply or to act on the ticket — both are her call:
whether a reply reaches the user at all, and whether she updates the
resolved ticket or tells the user it's fixed."""

from __future__ import annotations

import pytest

from penny.changelog import ChangelogEntry
from penny.tests.eval.conftest import EVAL_MODELS
from penny.tests.eval.drivers.changelog_announcement import ChangelogEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import SampleObservation, Share, ShareReading, SpecCategory
from penny.tests.eval.utils.fixtures import SynthCollection
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_FAMILY = "startup-announcement-discretion"

_TRACKER_NAME = "open-tickets"
_RESOLVED_KEY = "sync delay after reconnecting"
_RESOLVED_ENTRY = (
    f"{_RESOLVED_KEY} — intermittent few-second lag reconnecting after a dropped "
    "connection; reported twice, still open."
)
_OTHER_ENTRIES = (
    "map tiles blurry at max zoom — tiles don't sharpen past the default zoom level; still open.",
    "export file uses the wrong date order — exported names read day-month "
    "instead of the profile's own format; still open.",
)
_TICKETS = SynthCollection(
    name=_TRACKER_NAME,
    description="open reports from the user about rough edges in the app",
    entries=(_RESOLVED_ENTRY, *_OTHER_ENTRIES),
)

# The deploy's own one real-shaped fix — text phrased the way a merged PR's own
# body reads, naming the same fact the resolved ticket above names, so a
# correct update or acknowledgement traces to this, never a guess.
_FIX_PR = 5190
_ENTRIES = (
    ChangelogEntry(
        pr=_FIX_PR,
        text="Fixes the few-second lag that showed up reconnecting after a dropped connection.",
    ),
)

_FIXED_WORDS = ("fixed", "resolved", "sorted out", "patched", "taken care of")

_WORLD = World(name=_FAMILY, pages=(), keeps=(), excludes=(), stores=(_TICKETS,))


def _other_tickets_survive(sample: SampleObservation, _world: World) -> Answer:
    """Every OTHER open ticket is still there, unchanged — the resolved one is
    excluded on purpose: the turn touching it is the outcome this case measures,
    never a violation of what survives (#2974 — whether to act on it is her
    call, and acting on it is not loss)."""
    held = {(e.collection, e.key, e.content) for e in sample.held}
    lost = [
        e
        for e in sample.held_before
        if (e.collection, e.key) != (_TRACKER_NAME, _RESOLVED_KEY)
        and (e.collection, e.key, e.content) not in held
    ]
    return not lost, f"lost {[f'{e.collection}: {e.key}' for e in lost]}"


def _reply_reached_the_user(sample: SampleObservation) -> ShareReading | None:
    """Whether ANY text reached the user this turn — measured, never required
    (#2974): the new trigger states only what happened, so a turn that reads it
    and calls no tool and sends nothing is not a defect by construction."""
    reached = bool(sample.reply.strip())
    return ShareReading(value=1.0 if reached else 0.0, evidence=[sample.reply] if reached else [])


def _acted_on_the_resolved_ticket(sample: SampleObservation) -> ShareReading | None:
    """Whether she updated the resolved ticket's own entry, or told the user it's
    fixed, or both — measured, never required: the owner's design leaves whether
    and how to act on a resolved ticket entirely to her."""
    updated = any(
        e.collection == _TRACKER_NAME and e.key == _RESOLVED_KEY and e.content != _RESOLVED_ENTRY
        for e in sample.held
    )
    told = any(word in sample.reply.casefold() for word in _FIXED_WORDS)
    evidence = [*(["updated the ticket"] if updated else []), *(["told the user"] if told else [])]
    return ShareReading(value=1.0 if (updated or told) else 0.0, evidence=evidence)


_REPLY_REACHED = Share(
    "reply reached the user",
    _reply_reached_the_user,
    open_above=0.0,
    says="of all samples, the share where any text reached the user this turn",
)
_ACTED_ON_TICKET = Share(
    "acted on the resolved ticket",
    _acted_on_the_resolved_ticket,
    open_above=0.0,
    says=(
        "of all samples, the share where she updated the resolved ticket's own "
        "entry, told the user it's fixed, or both"
    ),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_the_delivery_turn_decides_for_herself_what_to_do_with_a_resolved_ticket(
    changelog_eval: ChangelogEval, model: str
) -> None:
    """The real startup delivery turn, alone (no follow-up ask, #2974), over a
    deploy that resolves one of three open tickets she already holds. Nothing
    invented; the other two tickets survive untouched; whether a reply happens
    at all, and whether she acts on the resolved one, are both measured, never
    required."""
    cohort = await changelog_eval(
        case_id=_FAMILY,
        behaviour=(
            "In the chat agent, when a deploy's own fix resolves one of her "
            "already-open tickets, the delivery turn decides for itself whether, "
            "how, and how many times to reply, over the fact of what happened — "
            "never an instruction to send a message or to act on the ticket."
        ),
        area=Area.NOTIFICATIONS,
        model=model,
        changelog_entries=_ENTRIES,
        changelog_commits=_ENTRIES,
        world=_WORLD,
        samples=15,
        min_pass_rate=None,
        family=_FAMILY,
    )
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.claim(
        "state: every other open ticket survives untouched",
        _other_tickets_survive,
        SpecCategory.STORE,
    )
    cohort.measure(_REPLY_REACHED, _ACTED_ON_TICKET)
