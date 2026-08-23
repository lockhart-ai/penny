"""NL-dispatch story: a coin flip is Python's, an opinion is hers (#1679/#1680).

The model is a biased chooser — asked to "pick one at random" it gravitates to the
first or most salient option — so a fair pick lives in Python, and the story is one
sentence with two directions:

  * "choose one of X, Y, Z at random" fires ``choose`` with every option the ask named,
    and the reply reports THE PICK THE TOOL RETURNED.  Said equals did: a reply naming a
    different option is the failure the tool exists to prevent, wearing a green call as a
    disguise; and
  * a JUDGMENT ask over the same options ("which do you think is best?") fires nothing —
    an opinion is hers to give, and a coin flip in its place answers a different question.

**Dispatch stands on the tool description ALONE.**  ``ChooseTool`` is registered on every
agent surface (``Agent.get_tools``) and no skill teaches this routing — nothing is
pre-seeded since migration 0108 — so these cases seed none: the world they measure is a
fresh deployment's.  The loud probe below asserts that world out loud rather than trusting
it.

**The conversation state machine fronts every driven turn** (#1706): it classifies before
the chat agent runs, and a pick-one ask lands in whatever state it lands in.  What is
scored here is the chat turn's DISPATCH, never the state it landed in.

Scoring is STRUCTURAL — the persisted call, its options, the pick the tool's own result
frame reported, and the store read before and after.  The two reply checks read that
structure rather than a vocabulary: the said-equals-did check compares the reply against
the pick the RUN produced, and the opinion floor against the options the ASK named.  How
well she answered is read at joint review against each case's ``reference`` reply, which is
DATA rather than a comment so the deterministic pin in ``test_eval_harness.py`` can run it
through this module's own checks without a GPU.

Report-only (``min_pass_rate=None``): a live-model dispatch rate is a number to read, and
the threshold is the code owner's to set once the numbers are read.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

import pytest

from penny.database import Database
from penny.database.models import PromptLog
from penny.penny import Penny
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Preparer,
    live_prompts,
    new_collections,
    routing_clean,
    tool_call_sequence,
)
from penny.tests.eval.dispatch_world import assert_dispatch_world

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) for every case in this module — shared with
# the sibling dispatch stories (email, generate_image) so the report's families rollup
# reads chat-surface tool dispatch as one group.
_FAMILY = "nl-dispatch"

_CHOOSE_TOOL = "choose"
_OPTIONS_ARG = "options"

# The three woods both asks are about.  One set, so the fire and no-fire directions are the
# same choice put two ways and nothing but the ask changes between them.
_OPTIONS = ("cedar", "maple", "birch")

# The real tool's result body (``CHOSE_MESSAGE``, whose whole render is pinned in the
# tool's own unit tests).  Reading the pick out of the persisted frame is what makes
# said-equals-did a structural comparison rather than a guess about which option she meant.
_PICK_PATTERN = re.compile(r"Chose '([^']+)' at random")


class _ChooseCase(NamedTuple):
    """One agreed ask, and what the turn it opens has to look like.

    ``fires`` says which direction this case is: an ask for a random pick, or a judgment
    ask that must reach nothing.  There is no third shape, so the scorer is chosen from it
    rather than from a separate field.

    ``reference`` is how the ask would be answered WELL: a review target, read at joint
    review and never matched by the scorer.  It is DATA rather than a comment so the
    deterministic pin can run it through this module's reply checks without a GPU — a
    scorer that cannot pass the answer the case itself calls correct is a broken scorer,
    and that is cheaper to find here than on the queue.  The fire case's reference is
    written around ONE of the three options, which is what lets the pin replay it as though
    the tool had returned that option."""

    case_id: str
    ask: str
    fires: bool
    reference: str


_RANDOM_PICK = _ChooseCase(
    case_id="choose-dispatch-fires",
    ask=(
        "choose one of cedar, maple, or birch at random for me, and tell me which one you picked."
    ),
    fires=True,
    reference="flipped for it — maple.",
)

_JUDGMENT = _ChooseCase(
    case_id="choose-dispatch-no-fire",
    ask="between cedar, maple, and birch, which do you think makes the best-sounding guitar top?",
    fires=False,
    reference=(
        "cedar, for me — it's warmer and it opens up faster than birch, though maple keeps "
        "its definition better if you play hard."
    ),
)

CHOOSE_CASES = (_RANDOM_PICK, _JUDGMENT)


# ── Reading what the run did ──────────────────────────────────────────────────


def _row_tool_calls(row: PromptLog) -> list[dict]:
    """One promptlog row's emitted tool calls (empty when the draw carried none)."""
    response = json.loads(row.response) if row.response else {}
    choices = response.get("choices") or []
    if not choices:
        return []
    return choices[0].get("message", {}).get("tool_calls") or []


def _choose_calls(db: Database) -> list[dict]:
    """Every ``choose`` call's parsed arguments, from the persisted promptlog.

    Sourced through ``live_prompts`` — the harness's fetch chokepoint, which drops any
    seeded prior turn — so this reads what THIS sample's model did and can never count a
    seeded round's call as the live turn's.  An undecodable argument blob reads as an
    argument-less call rather than raising: a malformed draw is a thing a live model
    produces, and a scorer that died on one would lose the whole sample instead of scoring
    it."""
    calls: list[dict] = []
    for row in live_prompts(db):
        for call in _row_tool_calls(row):
            function = call.get("function", {})
            if function.get("name") != _CHOOSE_TOOL:
                continue
            try:
                calls.append(json.loads(function.get("arguments") or "{}"))
            except json.JSONDecodeError, TypeError:
                calls.append({})
    return calls


def _tool_picks(db: Database) -> list[str]:
    """Every pick the REAL tool returned this sample, read from its persisted result
    frames — through ``live_prompts``, so a seeded prior turn's pick can never stand in for
    this turn's.

    A tool result is durable as a side effect of being fed back, so a pick rides into the
    NEXT call's ``messages``: a run that ended AT the choose call carries no pick here and
    no reply reporting one either, which is the same verdict either way."""
    picks: list[str] = []
    for row in live_prompts(db):
        if row.messages:
            picks += _PICK_PATTERN.findall(row.messages)
    return picks


# ── The loud probe: the chooser really is on the surface ──────────────────────


def assert_choose_world(penny: Penny, case: _ChooseCase) -> None:
    """Everything this case's world is responsible for, asserted out loud: ``choose`` is
    registered, and the registry holds no collection.

    Both claims are the shared dispatch-world probe (``dispatch_world``) — the registry half
    reads COLLECTION-shaped memories only, since the four migration-0026 system log markers
    are in every database and a probe that counted them could never pass."""
    assert_dispatch_world(penny, case.case_id, [_CHOOSE_TOOL])


def _probe_choose_world(case: _ChooseCase) -> Preparer:
    """Assert the world the case is answered in, once it is whole."""

    def prepare(penny: Penny) -> None:
        assert_choose_world(penny, case)

    return prepare


# ── Checks ────────────────────────────────────────────────────────────────────


def _chose_check(db: Database) -> Check:
    """The headline: the ask reached the chooser at all.

    Anchored to the call itself, so the verdict sits on the row that made it and a miss
    falls to the run-close table where a missing action belongs."""
    calls = _choose_calls(db)
    return Check(
        "calls: the ask reached the choose tool",
        bool(calls),
        anchor=f"{_CHOOSE_TOOL}(",
        rationale=None if calls else f"the turn fired {tool_call_sequence(db) or 'nothing'}",
        kind="spine",
    )


def _every_option_check(db: Database) -> Check:
    """The chooser was handed EVERY option the ask named — the half that makes a fair pick
    fair, since a coin flip between two of the three is a biased pick with a green call in
    front of it.  N/A when nothing fired, so one failure is reported once."""
    label = "calls: it was handed every option the ask named"
    calls = _choose_calls(db)
    if not calls:
        return Check.na(label, anchor=f"{_CHOOSE_TOOL}(", kind="spine")
    handed = [{str(option).lower() for option in call.get(_OPTIONS_ARG) or []} for call in calls]
    complete = any(options == set(_OPTIONS) for options in handed)
    return Check(
        label,
        complete,
        anchor=f"{_CHOOSE_TOOL}(",
        rationale=None if complete else f"it was handed {[sorted(o) for o in handed]}",
        kind="spine",
    )


def _reply_reports(pick: str, reply: str) -> bool:
    """Does the reply name the pick the tool returned?

    Pure and case-folded, so the pin in ``test_eval_harness.py`` can run each case's own
    reference reply through it BOTH ways without a GPU: the agreed answer must pass against
    the option it names, and must fail against the two it does not — a comparison that
    passed everything would make the whole said-equals-did check mean nothing."""
    return pick.lower() in reply.lower()


def _reports_the_tools_pick_check(db: Database, reply: str) -> Check:
    """SAID EQUALS DID on the pick itself: the reply names the option the TOOL returned.

    The one check the whole story turns on — a reply naming a different option means she
    free-chose past a call that looks perfectly green in the trace — and the reason it is
    read off the tool's own persisted result rather than off the options list.  N/A when
    the tool returned no pick: there is nothing for the reply to agree with, and the miss
    is already reported by the call check above."""
    label = "reply: she reports the pick the tool returned"
    picks = _tool_picks(db)
    if not picks:
        return Check.na(label, anchor=REPLY_ANCHOR, kind="reply")
    pick = picks[-1]
    said = _reply_reports(pick, reply)
    return Check(
        label,
        said,
        anchor=REPLY_ANCHOR,
        rationale=None if said else f"the tool chose {pick!r} and she said {reply!r}",
        kind="reply",
    )


def _no_choose_call_check(db: Database) -> Check:
    """The chooser did not fire — the no-fire direction's headline, with the rationale
    naming what it was handed so a miss reads as what happened rather than as a bare red."""
    calls = _choose_calls(db)
    return Check(
        "calls: choose did not fire on a judgment ask",
        not calls,
        rationale=f"it flipped between {calls}" if calls else None,
        kind="spine",
    )


def _gave_an_opinion_check(reply: str) -> Check:
    """The reply names one of the options — the floor under "she answered the question
    asked", since an opinion about which wood is best has to name a wood.

    Read against the options the ASK named rather than a vocabulary, so there is nothing to
    drift: how good the opinion was is read at joint review against the case's reference."""
    named = sorted(option for option in _OPTIONS if option in reply.lower())
    return Check(
        "reply: she gave an opinion, naming at least one of the woods",
        bool(named),
        anchor=REPLY_ANCHOR,
        rationale=None if named else f"named none of {list(_OPTIONS)}: {reply!r}",
        kind="reply",
    )


def _store_untouched_check(db: Database, before: set[str]) -> Check:
    """Nothing was created.  Choosing changes no durable state — the tool reports
    ``mutated=False`` — and this world holds no collection at all, so a collection
    appearing is the whole "nothing else was touched" claim rather than a sample of it."""
    created = sorted(row.name for row in new_collections(db, before))
    return Check(
        "state: nothing was created",
        not created,
        rationale=f"created {created}" if created else None,
        kind="state",
    )


def _dispatch_advisories(db: Database, reply: str) -> list[Check]:
    """What the turn actually did, verbatim and UNSCORED — the calls it made, the picks the
    tool returned, and the answer it gave — so a report shows the turn whichever way it went
    and the wording is read where wording is read: at joint review."""
    return [
        Check(f"fired: {tool_call_sequence(db)}", True, kind="proc", scored=False),
        Check(f"the tool returned: {_tool_picks(db)}", True, kind="proc", scored=False),
        Check(f"answered: {reply!r}", True, kind="reply", scored=False),
        Check(
            "calls: clean routing (no re-rolled draw or continue nudge)",
            routing_clean(db),
            scored=False,
            kind="proc",
        ),
    ]


# ── Scorers ───────────────────────────────────────────────────────────────────


def _score_random_pick(db: Database, before: set[str], reply: str) -> list[Check]:
    """The ask reached the chooser with every option, the reply reports the pick it
    returned, and the turn touched nothing else."""
    return [
        _chose_check(db),
        _every_option_check(db),
        _reports_the_tools_pick_check(db, reply),
        _store_untouched_check(db, before),
        *_dispatch_advisories(db, reply),
    ]


def _score_judgment_ask(db: Database, before: set[str], reply: str) -> list[Check]:
    """The judgment ask reached nothing, she answered it herself, and the turn touched
    nothing else."""
    return [
        _no_choose_call_check(db),
        _gave_an_opinion_check(reply),
        _store_untouched_check(db, before),
        *_dispatch_advisories(db, reply),
    ]


async def _run_choose_case(chat_eval: ChatEval, case: _ChooseCase) -> None:
    """Drive one choose case: the surface probed before the turn, the scorer picked by the
    direction the case declares.  Report-only — the threshold is the code owner's to set
    once the numbers are read."""
    await chat_eval(
        case_id=case.case_id,
        family=_FAMILY,
        message=case.ask,
        prepare=_probe_choose_world(case),
        score=_score_random_pick if case.fires else _score_judgment_ask,
        min_pass_rate=None,
    )


async def test_choose_fires_on_a_random_pick_ask(chat_eval: ChatEval) -> None:
    """ "choose one of X, Y, Z at random" — the fair-pick ask, where the failure worth
    catching is a green call whose result the reply then ignores."""
    await _run_choose_case(chat_eval, _RANDOM_PICK)


async def test_choose_does_not_fire_on_a_judgment_ask(chat_eval: ChatEval) -> None:
    """ "which do you think is best?" over the same three options — the over-firing
    direction, where the options are identical and only the question changed."""
    await _run_choose_case(chat_eval, _JUDGMENT)
