"""Dispatch: the fair pick, from a collector cycle (#2008, tranche 3; retargeted #2812).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**One case — ``choose-dispatch-fires``.**  A job's own instructions name several options and
ask for one of them at random, and the entry the cycle files reports the option the fair pick
returned.

**Why this runs on a COLLECTOR cycle, not a chat turn.**  ``choose`` came off the chat surface
(#2812): 444 measured prod tool calls never dispatched it from chat, while collectors reached
for it 9 times, so it stays where it is actually used.  The claim this case makes is unchanged
— the fair pick lives in Python, and a reply or an entry that disagrees with the tool's own
record means she free-chose past a call that looks perfectly green in the trace — only the
LOCUS moved, from a user's ask to a job's own free-text instructions (#2455 naive baseline: a
job has no user turn, so its natural language is its own ``extraction_prompt``).

What it asserts about the pick is the one thing an end state can carry: **the option the
written entry names is the one the run's own record chose.**  An entry naming a different
option means she free-chose past a call that looks perfectly green in the trace — the failure
the tool exists to prevent, wearing a green call as a disguise — and it is a PROVENANCE claim
rather than a route one, because what it reads is where a specific value in the entry came
FROM.  On a sample where the run produced no pick at all the sentence is false rather than
unasked: the entry reports an option the run never chose.

**The case does not claim what the cycle did NOT do.**  Whether it also wrote to another
collection or browsed is the model's own call on what it was shown, so it is measured — in the
tool sequence — and never asserted (``docs/principles.md`` §4.1, §4.3).  The container starts
empty, so there is no prior state for a preservation claim to read either.

**Dispatch stands on the tool description ALONE.**  ``ChooseTool`` is registered on every
collector cycle's surface and no skill teaches this routing — nothing has been pre-seeded since
migration 0108 — so the job's own prompt is the only recipe pointing at it.

**The model is a biased chooser** — asked to "pick one at random" it gravitates to the first or
most salient option — which is why a fair pick lives in Python (#1679/#1680) and why the
written entry agreeing with the tool is the whole point.

Three real kinds of wood stand in for the options a job's own prompt names — the repo is
public, so nothing here is drawn from any real person's own job.

REPORT-ONLY (``min_pass_rate=None``): the ceilings this run proposes are the code owner's to
accept once the numbers have been read.
"""

from __future__ import annotations

import re

import pytest

from penny.database import Database
from penny.tests.eval.conftest import EVAL_MODELS, Seeder, collection_entries
from penny.tests.eval.drivers.collector_cycles import CollectorCyclesEval, CycleArm
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.choose import CHOSE_MESSAGE

pytestmark = pytest.mark.eval

_FAMILY = "collector-dispatch"

_FIRES = "choose-dispatch-fires"

# The container every arm's cycle files its pick into.  One name, because one job.
_CONTAINER = "firewood-picks"
_CONTAINER_DESCRIPTION = "Which wood I'm burning this week, picked fresh each cycle."
_SCHEDULE = "FREQ=WEEKLY"

# The three woods the job's own prompt names.  ONE set, held constant across every arm, because
# the claim below names a value: the pick the run recorded has to be one of these for the
# comparison to mean anything.  Mutually exclusive and none a substring of another.
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

# Five wordings of ONE instruction — the job's own free text, naming the three woods and where
# to file the pick.  Same ask, same destination; only the words change.
INSTRUCTIONS = (
    "flip a fair coin among 'cedar', 'maple', and 'birch' and write down which one it landed on",
    "pick one of 'cedar', 'maple', or 'birch' at random and record which one you picked",
    "randomly select one of 'cedar', 'maple', or 'birch' and note which one you got",
    "roll the dice between 'cedar', 'maple', and 'birch' and save which one won",
    "choose one of 'cedar', 'maple', or 'birch' at random and file which one you chose",
)

_BEHAVIOUR = (
    "In a collector cycle, when the job's own instructions ask for one of several named "
    "options to be picked at random, Penny files the option the fair pick actually returned."
)

# How the ask would be answered WELL — a review target, and the input the
# deterministic pin in ``test_eval_harness.py`` runs through the claim below in BOTH directions:
# it must agree with the option it names and disagree with the two it does not, since a
# comparison that passed every option would let a free-chosen entry score green behind a real
# call.  DATA rather than a comment, so the pin can read it without a GPU.
REFERENCE_REPLY = "flipped for it — maple."


def picks_on_the_record(given: str) -> list[str]:
    """Every option the fair pick RETURNED this sample, oldest first, read off the tool's own
    persisted result frames.

    A tool result is durable as a side effect of being fed back, so a pick rides into the next
    call's messages and lands in what the cycle was GIVEN.  A run that ended AT the choose call
    carries no pick here and no entry reporting one either, which is the same verdict both
    ways.  Public so the deterministic pin can replay a reference entry against a pick without
    driving a model."""
    return _CHOSE_PATTERN.findall(given)


def reply_reports(pick: str, text: str) -> bool:
    """Does ``text`` name the pick the tool returned?

    Folded through the suite's ONE typography definition on both sides, so an entry that wrote
    the option with different casing or a curly apostrophe is read as naming it.  Pure, so the
    pin can run each direction without a GPU.  Named for the reply-shaped claim this helper
    first served; the collector claim below passes it the written entry's own text instead."""
    return fold_typography(pick) in fold_typography(text)


def _prompt(instruction: str) -> str:
    """The job's free-text instructions (#2455 naive baseline) — what a job's own
    ``extraction_prompt`` carries verbatim, with no skill/program rendering seam."""
    return (
        f"Each time this runs, {instruction}. Write a new entry into '{_CONTAINER}', keyed "
        f"by today's date, stating which wood you picked."
    )


def _seeder(instruction: str) -> Seeder:
    """The world this case's cycle starts in: a job already configured with its prompt, its
    container EMPTY — then every claim that world makes, asserted out loud."""

    def seed(db: Database) -> None:
        db.memories.create_collection(
            _CONTAINER,
            _CONTAINER_DESCRIPTION,
            extraction_prompt=_prompt(instruction),
            schedule=_SCHEDULE,
            notify=False,
        )
        _assert_the_firewood_world(db, instruction)

    return seed


def _assert_the_firewood_world(db: Database, instruction: str) -> None:
    """Everything the seeder is responsible for, asserted where it fails loudly."""
    row = db.memories.get(_CONTAINER)
    assert row is not None, f"the job's container {_CONTAINER!r} must exist"
    assert not row.archived, "the job must be live when the cycle runs"
    prompt = row.extraction_prompt
    assert prompt is not None, f"the job's container {_CONTAINER!r} must carry a prompt"
    assert instruction in prompt, (
        f"this arm's instruction must reach the model as {instruction!r}. Prompt: {prompt!r}"
    )
    held = collection_entries(db, _CONTAINER)
    assert not held, f"the container must be empty when the cycle starts — it holds {held}"


def _world(instruction: str) -> World:
    """The ground every arm of this case runs against — EMPTY, and each field is a report
    rather than an omission: no page decides a coin flip, so a page set would only give a
    sample that went browsing something to talk about, and the job is asked to write its pick,
    not to keep or exclude anything a page carries."""
    return World(name=f"{_FIRES}-{instruction[:12]}", pages=(), keeps=(), excludes=())


def _arm(instruction: str) -> CycleArm:
    """One arm of this case: this instruction, over an empty container, with no page."""
    return CycleArm(
        text=instruction, seed=_seeder(instruction), pages=[], world=_world(instruction)
    )


def _arms() -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction, over the same empty world."""
    return [_arm(instruction) for instruction in INSTRUCTIONS]


# ── The claim, as a pure function over one sample ─────────────────────────────


def _the_entry_reports_the_pick_the_run_made(sample: SampleObservation, _world: World) -> Answer:
    """The option the written entry states is the one the run's own record chose.

    Answers its own sentence on every sample, including the ones where no pick was made: with
    nothing on the record the entry names an option the run never chose, and the sentence is
    FALSE rather than unasked.  A violating sample is nameable in both directions — one that
    made a real call and then filed a different wood (the failure the tool exists to prevent,
    wearing a green call as a disguise), and one that picked with no call at all."""
    picks = picks_on_the_record(sample.given)
    if not picks:
        return False, "nothing on the record chose an option, so there is no pick to file"
    pick = picks[-1]
    written = " ".join(entry.text for entry in sample.entries if entry.collection == _CONTAINER)
    return reply_reports(pick, written), f"the run chose {pick!r}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_random_pick_is_filed_as_the_tool_made_it(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A job's own instructions name three woods and ask for one of them at random, filed into
    the job's container.

    What is NOT claimed is that the chooser was handed all three: which options a call carries
    is a property of the CALL, and the call is a route.  A pick between two of the three is a
    biased pick behind a green call, and it shows up where routes show up — as a tool-sequence
    divergence a reader opens the sample for."""
    cohort = await collector_cycles_eval(
        case_id=_FIRES,
        behaviour=_BEHAVIOUR,
        area=Area.BACKGROUND_COLLECTORS,
        model=model,
        collection=_CONTAINER,
        arms=_arms(),
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # STORE — PROVENANCE, really: said-equals-did on the one thing the cycle writes.
    cohort.claim(
        "state: the written entry reports the option the run's own record chose",
        _the_entry_reports_the_pick_the_run_made,
        SpecCategory.STORE,
    )
    cohort.assert_every_value_in_the_store_is_sourced()

    # TOOL_SEQUENCE is measured and never asserted — the call is a route.  It shows how a sample
    # got to the pick, not whether the pick is right: what the entry DID with it is the claim
    # above, and a sample that free-chose is caught there rather than here, since it files an
    # option no record holds.
    cohort.measure(TOOL_SEQUENCE, ENTRIES_STORED)
