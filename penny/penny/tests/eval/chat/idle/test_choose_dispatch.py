"""Dispatch: one case (#2008, tranche 3).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**The behaviour is *Penny fires the tool when the message is a request, and fires nothing when
the message merely shares its topic*, and one case carries it** — because on the END-STATE axis
the two directions agree.  ``choose`` is read-shaped (``mutated=False``): it creates nothing,
writes nothing, and changes no mechanism, so a fair pick and an opinion leave the store in the
same condition.  What separates them is entirely the ROUTE, and a route is measured rather than
asserted — many routes reach one end state, and a rule keyed to the name ``choose`` would not
fire for a verb nobody enumerated.  So ``choose-dispatch-no-fire`` is folded here as the stated
negative rather than kept as a second case.

What that costs is worth naming rather than glossing, because the instrument cannot currently
pay it: the tool sequence would be where such a divergence shows, and it reads BLIND here — the
chat observation narrows a sample's calls to ``ENACTING_TOOLS``, a six-name whitelist carrying
no ``choose``, so every sample reads "no call" whether the fair pick fired or not.  Reported as
a harness gap.  Until it is closed, the claim below is what catches a free choice: a reply
naming an option no record chose fails it, which is the same finding said from the end state.

What the case DOES assert about the pick is the one thing an end state can carry: **the option
the reply reports is the one the run's own record chose.**  A reply naming a different option
means she free-chose past a call that looks perfectly green in the trace — the failure the tool
exists to prevent, wearing a green call as a disguise — and it is a PROVENANCE claim rather than
a route one, because what it reads is where a specific value in the reply came FROM.  On a
sample where the run produced no pick at all the sentence is false rather than unasked: the
reply reports an option the run never chose.

**Dispatch stands on the tool description ALONE.**  ``ChooseTool`` is registered on every agent
surface and no skill teaches this routing — nothing has been pre-seeded since migration 0108 —
so this case seeds none: the world it measures is a fresh deployment's, and that is what makes
*nothing was created* a total reading of what the turn touched rather than a sample of it.

**The model is a biased chooser** — asked to "pick one at random" it gravitates to the first or
most salient option — which is why a fair pick lives in Python (#1679/#1680) and why the reply
agreeing with the tool is the whole point.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

import re

import pytest

from penny.conversation_machine import ConversationState
from penny.penny import Penny
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
    fold_typography,
)
from penny.tests.eval.utils.dispatch_world import assert_dispatch_world
from penny.tests.eval.utils.worlds import World
from penny.tools.choose import CHOSE_MESSAGE

pytestmark = pytest.mark.eval

# Family tag (explicit, meaningful grouping) — shared with the sibling dispatch stories
# (email, generate_image, the muting contracts) so the report's families rollup reads
# chat-surface tool dispatch as one group.
_FAMILY = "nl-dispatch"

_CHOOSE_TOOL = "choose"
_CASE_ID = "choose-dispatch-fires"

# The three woods the ask is about.  ONE set, held constant across every arm, because the claim
# below names a value: the pick the run recorded has to be one of these for the comparison to
# mean anything.  Mutually exclusive and none a substring of another.
_OPTIONS = ("cedar", "maple", "birch")

# How the fair pick reports itself, read out of the TOOL'S OWN result template rather than
# retyped here: the pick sits between the message's two literal halves.  Reading it from the
# constant is what keeps this a structural comparison — a second copy of the sentence would be a
# second contract, free to drift from the one the tool actually writes.
_PICK_FIELD = "{pick}"
_CHOSE_HEAD, _CHOSE_REST = CHOSE_MESSAGE.split(_PICK_FIELD)
_CHOSE_PATTERN = re.compile(
    re.escape(_CHOSE_HEAD) + "(.+?)" + re.escape(_CHOSE_REST.partition("{")[0])
)

# The world every arm is answered against.  Every field is EMPTY, and each is a report rather
# than an omission: no page decides a coin flip, so a page set would only give a sample that
# went browsing something to talk about; the turn is asked to write nothing, so ``keeps`` would
# state a contract the ask never made and ``excludes`` would assert one reading of an ask that
# rules nothing out; and ``answers`` cannot name a token at all, because WHICH wood a correct
# reply states is chosen at random inside the tool — the claim that reads it is the case's own,
# against the pick the run actually made.
_WORLD = World(name=_CASE_ID, pages=(), keeps=(), excludes=())

_ASK = "choose one of cedar, maple, or birch at random for me, and tell me which one you picked."
_ALSO_PHRASED = (
    "pick one of cedar, maple, or birch at random and tell me which one it was",
    "randomly select one of cedar, maple, or birch, and say which you got",
    "flip a coin between cedar, maple and birch and tell me which one won",
    "surprise me with one of cedar, maple, or birch at random — which is it?",
)

_BEHAVIOUR = (
    "In the chat agent, when the user asks for one of several named options to be picked at "
    "random, Penny reports the option the fair pick actually returned and changes nothing "
    "durable — while the same options put as a question about which she prefers are answered "
    "in her own words, changing nothing either."
)

# How the fire direction would be answered WELL — a review target, and the input the
# deterministic pin in ``test_eval_harness.py`` runs through the claim below in BOTH directions:
# it must agree with the option it names and disagree with the two it does not, since a
# comparison that passed every option would let a free-chosen reply score green behind a real
# call.  DATA rather than a comment, so the pin can read it without a GPU.
REFERENCE_REPLY = "flipped for it — maple."


def assert_choose_world(penny: Penny) -> None:
    """The world this case is answered in, asserted out loud before the turn runs: ``choose``
    is registered on the chat surface, and the registry holds no COLLECTION.

    Both claims are the shared dispatch-world probe.  The surface half forecloses a scored
    dispatch miss against a model that was never offered the tool; the registry half is what
    makes *nothing was created* a total reading — the registry read counts COLLECTION-shaped
    memories only, since the four migration-0026 system log markers are in every database and a
    probe that counted them could never pass."""
    assert_dispatch_world(penny, _CASE_ID, [_CHOOSE_TOOL])


def picks_on_the_record(given: str) -> list[str]:
    """Every option the fair pick RETURNED this sample, oldest first, read off the tool's own
    persisted result frames.

    A tool result is durable as a side effect of being fed back, so a pick rides into the next
    call's messages and lands in what the turn was GIVEN.  A run that ended AT the choose call
    carries no pick here and no reply reporting one either, which is the same verdict both
    ways.  Public so the deterministic pin can replay a reference reply against a pick without
    driving a model."""
    return _CHOSE_PATTERN.findall(given)


def reply_reports(pick: str, reply: str) -> bool:
    """Does the reply name the pick the tool returned?

    Folded through the suite's ONE typography definition on both sides, so a reply that wrote
    the option with different casing or a curly apostrophe is read as naming it.  Pure, so the
    pin can run each direction without a GPU."""
    return fold_typography(pick) in fold_typography(reply)


# ── The claims, as pure functions over one sample ─────────────────────────────


def _nothing_durable_changed(sample: SampleObservation, _world: World) -> Answer:
    """The turn created nothing, changed no mechanism and wrote no entry.

    The end state BOTH directions of this behaviour share — choosing changes no durable state,
    and neither does giving an opinion — which is exactly why the two are one case.  A violating
    sample is nameable: one that stands a container up to keep a record of what it picked, and
    one that files the pick into a list nobody asked for."""
    touched = sorted(
        one.name for one in sample.mechanisms if one.born_this_run or one.changed_this_run
    )
    wrote = sorted(f"{entry.collection}:{entry.key}" for entry in sample.entries)
    return not touched and not wrote, f"created or changed {touched}, wrote {wrote}"


def _the_reply_reports_the_pick_the_run_made(sample: SampleObservation, _world: World) -> Answer:
    """The option the reply states is the one the run's own record chose.

    Answers its own sentence on every sample, including the ones where no pick was made: with
    nothing on the record the reply names an option the run never chose, and the sentence is
    FALSE rather than unasked.  A violating sample is nameable in both directions — one that
    made a real call and then named a different wood (the failure the tool exists to prevent,
    wearing a green call as a disguise), and one that picked with no call at all."""
    picks = picks_on_the_record(sample.given)
    if not picks:
        return False, "nothing on the record chose an option, so the reply reports no pick"
    pick = picks[-1]
    return reply_reports(pick, sample.reply), f"the run chose {pick!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_random_pick_is_reported_as_the_tool_made_it(
    chat_eval: ChatEval, model: str
) -> None:
    """Three named options and an ask for one of them at random.

    What is NOT claimed is that the chooser was handed all three: which options a call carries
    is a property of the CALL, and the call is a route.  A pick between two of the three is a
    biased pick behind a green call, and it shows up where routes show up — as a tool-sequence
    divergence a reader opens the sample for."""
    cohort: Cohort = await chat_eval(
        case_id=_CASE_ID,
        behaviour=_BEHAVIOUR,
        model=model,
        prepare=assert_choose_world,
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

    # STORE — the end state both directions share.
    cohort.claim("state: nothing durable changed", _nothing_durable_changed, SpecCategory.STORE)

    # PROVENANCE — said equals did, plus the general guard against a value tracing to nothing.
    cohort.claim(
        "reply: it reports the option the run's own record chose",
        _the_reply_reports_the_pick_the_run_made,
        SpecCategory.PROVENANCE,
        kind="reply",
    )
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — the call is a route — and on this case it
    # reads BLIND, which the report renders in red.  The chat observation narrows a sample's
    # calls to ``ENACTING_TOOLS``, a six-name whitelist carrying no ``choose``, so every sample
    # reads "no call" whether the fair pick fired or not.  What that costs is precisely the
    # negative direction, and it is why the claim above carries the case rather than the
    # feature: a sample that free-chose is invisible here and fails there, because the option it
    # reports appears on no record.  Measured anyway, because a feature that prints a number
    # either way has to say which of the two it is; the whitelist is reported as a harness gap
    # rather than worked around here.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)
