"""NL-dispatch contracts for the notification mute/unmute tools — the chat agent
routing a naive-register utterance to ``notifications_mute`` /
``notifications_unmute``, driven against the REAL model and scored on the
PERSISTED ``MuteState`` row + the tool the model actually called.

This is the retirement contract for the ``/mute`` + ``/unmute`` commands (epic
#1445, issue #1447): the slash commands are gone, so the intent must now dispatch
from natural language.  Every case asserts STRUCTURALLY — which tool fired (from
the persisted promptlog) and whether the mute row is present/absent afterward —
never on the reply's wording, which is stochastic.

  mute      — "stop messaging me for a while", "quiet down" → notifications_mute
              called + MuteState present.
  unmute    — "you can message me again", "turn updates back on" (seeded muted)
              → notifications_unmute called + MuteState absent.
  no-fire   — a casual mention ("it's been a quiet day") must NOT call either
              tool and must leave the mute state untouched.

**Dispatch now stands on the tool descriptions ALONE.**  Migration 0076 seeded a
"Mute or unmute notifications" skill whose numbered steps taught this routing;
0092 deleted every seeded rule entry and 0097 the collection itself, and 0108
leaves nothing pre-seeded at all — so these cases seed no skill, the registry the
turn runs against is empty, and what they measure is whether
``NotificationsMuteTool`` / ``NotificationsUnmuteTool`` (live on the chat surface,
``ChatAgent.get_tools``) are reachable from a naive utterance with no recipe
pointing at them.  The seeded world is already the production cold start.

Report-only (``min_pass_rate=None``), the canonical convention — stated per case
rather than inherited from ``chat_eval``'s 0.75 default, so the threshold is a
thing the module says rather than a thing it happens to get.
"""

from __future__ import annotations

import pytest

from penny.database import Database
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    Preparer,
    collection_names,
    tool_call_sequence,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.dispatch_world import assert_dispatch_world

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module — shared with
# the sibling dispatch stories (email, generate_image, choose) so the report's families
# rollup reads chat-surface tool dispatch as one group.
_FAMILY = "nl-dispatch"

_MUTE = "notifications_mute"
_UNMUTE = "notifications_unmute"


def _probe_dispatch_world(case_id: str) -> Preparer:
    """Assert the world each case is answered in, once the runner has built it.

    Both claims are the shared dispatch-world probe: the chat surface really carries the
    two tools (they are registered unconditionally, but a scored dispatch miss against a
    model that was never offered the tool is the failure this forecloses), and the
    registry holds no COLLECTION — which is what makes the no-fire case's "built nothing"
    row a total reading rather than a sample of one."""

    def prepare(penny: Penny) -> None:
        assert_dispatch_world(penny, case_id, [_MUTE, _UNMUTE])

    return prepare


def _seed_muted(db: Database) -> None:
    """Start the user already muted — the precondition for an unmute case."""
    db.users.set_muted(TEST_SENDER)
    assert db.users.is_muted(TEST_SENDER), (
        "the unmute cases start from a muted user — an unseeded one would make "
        "'MuteState absent' true before the turn ran"
    )


# ── Scorers (read the persisted MuteState row + the promptlog tool calls) ─────


def _landed_state_check(db: Database) -> Check:
    """Advisory: which state the conversation machine put this turn in.

    Since #1706 the machine classifies every message BEFORE the chat agent runs, so a
    naive mute request is also a turn that could land in ``learn`` and mint a routine at
    run end.  None of these cases score that — what they measure is the dispatch — but a
    surprising landing is the first thing worth seeing when one of them moves, so it is
    rendered rather than left to be rediscovered."""
    latest = db.machine.latest_transition()
    landed = latest.to_state if latest is not None else None
    return Check(
        "the machine recorded where the turn landed",
        latest is not None,
        scored=False,
        kind="proc",
        rationale=f"landed in {landed!r}" if latest is not None else "the machine never moved",
    )


def _score_mute(db: Database, before: set[str], reply: str) -> list[Check]:
    """A naive "stop messaging me" utterance routed onto the mute tool, scored on the tool
    that fired and the MuteState row it left — never on the reply's wording."""
    fired = tool_was_called(db, _MUTE)
    unmuted = tool_not_called(db, _UNMUTE)
    muted = db.users.is_muted(TEST_SENDER)
    return [
        Check(
            "routed the request onto notifications_mute",
            fired,
            anchor=f"{_MUTE}(",
            kind="spine",
            rationale=None
            if fired
            else (
                "the mute tool never fired — with nothing in the registry pointing at it, "
                "the dispatch stands on the tool description alone.  Calls made: "
                f"{tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "left notifications_unmute alone",
            unmuted,
            anchor=f"{_UNMUTE}(",
            kind="spine",
            rationale=None if unmuted else "called the opposite tool as well as the right one",
        ),
        Check(
            "the user is muted afterwards (the MuteState row is there)",
            muted,
            kind="state",
            rationale=None
            if muted
            else "no MuteState row — whatever the reply said, nothing was actually muted",
        ),
        _landed_state_check(db),
    ]


def _score_unmute(db: Database, before: set[str], reply: str) -> list[Check]:
    """A naive "you can message me again" utterance routed onto the unmute tool, from a
    world that starts muted — scored on the tool that fired and the row it removed."""
    fired = tool_was_called(db, _UNMUTE)
    left_alone = tool_not_called(db, _MUTE)
    unmuted = not db.users.is_muted(TEST_SENDER)
    return [
        Check(
            "routed the request onto notifications_unmute",
            fired,
            anchor=f"{_UNMUTE}(",
            kind="spine",
            rationale=None
            if fired
            else (
                "the unmute tool never fired — with nothing in the registry pointing at it, "
                "the dispatch stands on the tool description alone.  Calls made: "
                f"{tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "left notifications_mute alone",
            left_alone,
            anchor=f"{_MUTE}(",
            kind="spine",
            rationale=None
            if left_alone
            else "re-muted as well as unmuting — the seeded world was already muted",
        ),
        Check(
            "the user is no longer muted (the MuteState row is gone)",
            unmuted,
            kind="state",
            rationale=None
            if unmuted
            else "the MuteState row is still there — whatever the reply said, nothing changed",
        ),
        _landed_state_check(db),
    ]


def _score_no_fire(db: Database, before: set[str], reply: str) -> list[Check]:
    """A casual mention of a quiet day is NOT a mute request: neither tool may fire and the
    mute state must be exactly where the turn found it.

    The registry row is advisory rather than scored — the machine fronts every turn now, so
    an over-eager landing in ``learn`` can mint a routine off an ordinary remark, which is
    worth SEEING here without widening what this module claims to measure."""
    mute_quiet = tool_not_called(db, _MUTE)
    unmute_quiet = tool_not_called(db, _UNMUTE)
    unchanged = not db.users.is_muted(TEST_SENDER)
    built = sorted(collection_names(db) - before)
    return [
        Check(
            "did not mute on a passing mention",
            mute_quiet,
            anchor=f"{_MUTE}(",
            kind="spine",
            rationale=None
            if mute_quiet
            else "muted the user off a remark about the day being quiet",
        ),
        Check(
            "did not unmute on a passing mention",
            unmute_quiet,
            anchor=f"{_UNMUTE}(",
            kind="spine",
            rationale=None if unmute_quiet else "unmuted off a remark about the day being quiet",
        ),
        Check(
            "the mute state is exactly where the turn found it",
            unchanged,
            kind="state",
            rationale=None if unchanged else "the turn left a MuteState row behind",
        ),
        Check(
            "built nothing off an ordinary remark",
            not built,
            scored=False,
            kind="proc",
            rationale=f"created {built}" if built else "nothing was created",
        ),
        _landed_state_check(db),
    ]


# ── Cases ─────────────────────────────────────────────────────────────────────


async def test_mute_stop_messaging(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="mute-stop-messaging",
        family=_FAMILY,
        prepare=_probe_dispatch_world("mute-stop-messaging"),
        message="hey, can you stop messaging me for a while? need some quiet",
        score=_score_mute,
        min_pass_rate=None,
    )


async def test_mute_quiet_down(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="mute-quiet-down",
        family=_FAMILY,
        prepare=_probe_dispatch_world("mute-quiet-down"),
        message="quiet down please — no proactive updates for now",
        score=_score_mute,
        min_pass_rate=None,
    )


async def test_unmute_message_again(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="unmute-message-again",
        family=_FAMILY,
        prepare=_probe_dispatch_world("unmute-message-again"),
        message="ok, you can start messaging me again",
        seed=_seed_muted,
        score=_score_unmute,
        min_pass_rate=None,
    )


async def test_unmute_turn_back_on(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="unmute-turn-back-on",
        family=_FAMILY,
        prepare=_probe_dispatch_world("unmute-turn-back-on"),
        message="go ahead and turn your updates back on",
        seed=_seed_muted,
        score=_score_unmute,
        min_pass_rate=None,
    )


async def test_no_fire_casual_mention(chat_eval: ChatEval) -> None:
    await chat_eval(
        case_id="notifications-no-fire",
        family=_FAMILY,
        prepare=_probe_dispatch_world("notifications-no-fire"),
        message="it's been a quiet day today, not much going on honestly",
        score=_score_no_fire,
        min_pass_rate=None,
    )
