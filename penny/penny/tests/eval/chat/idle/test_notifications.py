"""The muting contracts — an explicit request to mute or unmute, dispatched onto the
tool that does it, driven against the REAL model and scored on the PERSISTED
``MuteState`` row + the tool the model actually called.

This is the retirement contract for the ``/mute`` + ``/unmute`` commands (epic
#1445, issue #1447): the slash commands are gone, so the intent must dispatch from
natural language.  THREE cases, and each asserts STRUCTURALLY — which tool fired
(from the persisted promptlog) and what the mute row says afterward — never on the
reply's wording, which is stochastic.

  mute     — an unmuted world, "please mute notifications" → notifications_mute
             fired, the MuteState row is there, nothing else touched.
  unmute   — a muted world, "okay you can unmute notifications" →
             notifications_unmute fired, the row is gone, nothing else touched.
  no-fire  — a casual mention ("it's been a quiet day") must call neither tool and
             leave the mute state exactly where it found it.

**Each case also verifies the state was IN FRONT OF THE MODEL.**  Whether
notifications are muted is rendered ambiently by ``SelfStateHeader`` (#1919), and
that is half of what a muting turn stands on: before it, nothing in the context
said which way the switch was set, so the tool descriptions were the only carrier
of a state the model could not verify.  The check reads the sample's own persisted
chat SYSTEM PROMPT for the header's line — asserted per sample rather than assumed
from the deterministic pins, because a header that stopped rendering would leave
every case here scoring a turn that never saw the state.

**Dispatch stands on the tool descriptions ALONE.**  Migration 0076 seeded a
"Mute or unmute notifications" skill whose numbered steps taught this routing;
0092 deleted every seeded rule entry and 0097 the collection itself, and 0108
leaves nothing pre-seeded at all — so these cases seed no skill, the registry the
turn runs against is empty, and what they measure is whether
``NotificationsMuteTool`` / ``NotificationsUnmuteTool`` (live on the chat surface,
``ChatAgent.get_tools``) are reachable from an explicit ask with no recipe pointing
at them.  The seeded world is already the production cold start.

Report-only (``min_pass_rate=None``), the canonical convention — stated per case
rather than inherited from ``chat_eval``'s 0.75 default, so the threshold is a
thing the module says rather than a thing it happens to get.
"""

from __future__ import annotations

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.database import Database
from penny.penny import Penny
from penny.tests.conftest import TEST_SENDER
from penny.tests.eval.conftest import (
    ChatEval,
    Check,
    Preparer,
    _iter_prompt_messages,
    collection_names,
    tool_call_sequence,
    tool_not_called,
    tool_was_called,
)
from penny.tests.eval.utils.dispatch_world import assert_dispatch_world

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


def _state_in_context_check(db: Database, line: str) -> Check:
    """The mute state was IN the chat system prompt this sample ran on.

    Half of what a muting turn stands on, and the half that used to be missing: with
    nothing rendering the switch, the tool descriptions were its only carrier and they
    could only describe a state the model had no way to check.  Read off the sample's
    own persisted prompt rather than trusted from the deterministic render pins, because
    a header that stopped rendering would leave every case here scoring a turn that never
    saw the state — green for the wrong reason.

    The expected text comes from ``SelfStateHeader``'s own constant, so the check cannot
    drift from what the header writes."""
    present = any(
        message.get("role") == "system" and line in (message.get("content") or "")
        for message in _iter_prompt_messages(db)
    )
    return Check(
        "the mute state was in the prompt the model answered on",
        present,
        kind="guard",
        rationale=None
        if present
        else (
            f"no system prompt carried {line!r} — the turn was answered with nothing "
            "saying which way notifications were set"
        ),
    )


def _nothing_else_touched_check(db: Database, sibling: str, before: set[str]) -> Check:
    """The turn did the one thing it was asked and no more: the opposite tool stayed
    quiet and no collection was built off the request."""
    quiet = tool_not_called(db, sibling)
    built = sorted(collection_names(db) - before)
    return Check(
        "nothing else was touched",
        quiet and not built,
        anchor=f"{sibling}(",
        kind="spine",
        rationale=None
        if quiet and not built
        else (
            f"{'called ' + sibling + ' as well; ' if not quiet else ''}"
            f"{'created ' + str(built) if built else ''}".strip("; ")
        ),
    )


def _score_mute(db: Database, before: set[str], reply: str) -> list[Check]:
    """An explicit "mute notifications" request, answered on a prompt that says they are
    ON, routed onto the tool that mutes them."""
    fired = tool_was_called(db, _MUTE)
    muted = db.users.is_muted(TEST_SENDER)
    return [
        _state_in_context_check(db, SelfStateHeader.NOTIFICATIONS_ON),
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
            "the user is muted afterwards (the MuteState row is there)",
            muted,
            kind="state",
            rationale=None
            if muted
            else "no MuteState row — whatever the reply said, nothing was actually muted",
        ),
        _nothing_else_touched_check(db, _UNMUTE, before),
        _landed_state_check(db),
    ]


def _score_unmute(db: Database, before: set[str], reply: str) -> list[Check]:
    """An explicit "unmute notifications" request, answered on a prompt that says they
    are MUTED, routed onto the tool that lifts it."""
    fired = tool_was_called(db, _UNMUTE)
    unmuted = not db.users.is_muted(TEST_SENDER)
    return [
        _state_in_context_check(db, SelfStateHeader.NOTIFICATIONS_MUTED),
        Check(
            "routed the request onto notifications_unmute",
            fired,
            anchor=f"{_UNMUTE}(",
            kind="spine",
            rationale=None
            if fired
            else (
                "the unmute tool never fired — with nothing in the registry pointing at "
                "it, the dispatch stands on the tool description alone.  Calls made: "
                f"{tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "the user is no longer muted (the MuteState row is gone)",
            unmuted,
            kind="state",
            rationale=None
            if unmuted
            else "the MuteState row is still there — whatever the reply said, nothing changed",
        ),
        _nothing_else_touched_check(db, _MUTE, before),
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


async def test_an_explicit_mute_request_mutes(chat_eval: ChatEval) -> None:
    """An unmuted world and a request in the tool's own terms: the turn fires the mute
    tool, the MuteState row is there afterwards, and nothing else moved."""
    await chat_eval(
        case_id="explicit-mute-request-mutes",
        family=_FAMILY,
        prepare=_probe_dispatch_world("explicit-mute-request-mutes"),
        message="please mute notifications",
        score=_score_mute,
        min_pass_rate=None,
    )


async def test_an_explicit_unmute_request_unmutes(chat_eval: ChatEval) -> None:
    """A muted world and a request in the tool's own terms: the turn fires the unmute
    tool, the MuteState row is gone afterwards, and nothing else moved."""
    await chat_eval(
        case_id="explicit-unmute-request-unmutes",
        family=_FAMILY,
        prepare=_probe_dispatch_world("explicit-unmute-request-unmutes"),
        message="okay you can unmute notifications",
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
