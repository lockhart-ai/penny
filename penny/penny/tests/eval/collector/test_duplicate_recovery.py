"""The two doors onto a stored entry, and what a collector cycle owes at each (#1919).

Ported to the cohort structure under #2060; the contract is `docs/eval-case-design.md`.

A collector's write can land on an entry that already exists in two quite different
situations, and the write gate answers each with its own outcome:

  same value      — the cycle looked and found what it already had, filed under a key it
                    worded differently this time.  Nothing happened, so the cycle STOPs at
                    the chokepoint and the user hears nothing — the same no-news the exact
                    key reads, reached by the other door (``DUPLICATE_UNCHANGED``, in
                    ``WRITE_GATE_STOP_REASONS``).
  divergent value — the cycle found something DIFFERENT about a thing it already tracks.
                    That is news: the rejection binds the matched key into an
                    ``update_entry`` call, the change lands on the entry that exists, and
                    the user is told once.  It must NEVER stop.

**TWO CASES, because a correct sample for one is exactly wrong for the other.**  One writes
nothing and says nothing; the other rewrites the entry and sends a message.  They also
exercise different members of a closed enum — ``DUPLICATE_UNCHANGED`` against ``DUPLICATE``
— which is what the design's "materially different decision" means where a gate is
involved.  Their worlds differ for the same reason, and deliberately (below).

**Why the split is structural rather than a stronger claim.**  Before it, the same-value
door produced a rejection saying an entry like this already existed and inviting the model
to refresh it or write something else — and on the first measured run all ten samples
"recovered" by generating a DIFFERENT recipe.  Read against the surface they were given
that is correct behaviour: the routine says to file recipes not already saved, the gate
said this one was, and nothing said the cycle was over.  A cycle that found nothing new had
no way to say so, so the fix is to give it one rather than to tell the model harder not to
invent (the rational-actor doctrine — fix the state, not the imperative).

**Each case's ROUTINE is what makes its outcome the routine's own job (#1919).**  The
divergent case first ran on the recipe box, and its thinking showed the model serving that
routine faithfully: *"We must submit a new recipe… not already saved."*  A routine whose job
is to FILE SOMETHING NEW is answered correctly by writing something new, so the case was
measuring a generative mandate rather than the decision it meant to.  It runs on a WATCH
now: a page is read and the current value about ONE tracked subject is recorded, so landing
a changed value on the entry that already holds it is the routine carrying out its own job.
The same-fact case stays on the recipe box, where finding nothing new is exactly the tension
its STOP resolves.

Both cases share the porting shape:

  * ONE CYCLE.  The entry condition — what the collection already holds — is what selects
    the behaviour.  Preseeded through the store's own write path, under the key the job's
    program writes to, so the gate compares against a row of exactly the shape a real cycle
    would have left.  The entries' VECTORS are backfilled by the runner right after the
    seed, which is what makes the same-value signal scorable at all: a stored entry with no
    content vector answers every collision with the divergent rejection.
  * THE FAULT IS FIXED, the WORDING varies.  A collector has no user turn, so its natural
    language is the ``SkillSubstitution`` descriptions its rendered program carries, written
    upstream and printed by ``render_skill``.  Five wordings each; the forced write, the
    keys and the values are byte-identical on every arm.  The watch additionally reads five
    prose variants of one page, around a byte-identical datum line — the page's prose is the
    other half of a browse-driven cycle's natural language.
  * THE INJECTION IS HARNESS MACHINERY.  A sample the forced write never fired on ran an
    unbroken cycle and exercised no recovery, so it leaves the cohort as a named exclusion
    rather than counting as a behavioural failure.  THE COST, STATED (#2018):
    ``bail_injected`` says the write was ISSUED, not that the gate classified it the way the
    case is named for — a cycle whose collision took the exact-key door leaves here too.
  * WHEN the write is forced is part of the design.  An injector's response is synthetic and
    bypasses the persisting client (#1695), so a forced call that ENDS the cycle leaves the
    run with no promptlog row at all — nothing for ``set_run_outcome`` to stamp, an empty
    run record, and every record-field claim reading an absent ledger as though the model
    had done nothing.  That is what happened on the same-fact case's first measured run: the
    STOP fired correctly and five samples reported it as behavioural failure.  So both
    writes are staged AFTER the model's first real step.
  * LANDED renders EMPTY on both, and that is the correct report: a collector moves no
    conversation machine.  The run record's outcome and its stop reason are RECORD FIELDS,
    so they are claimed under STORE.
  * the tool calls MEASURED rather than asserted — which door a write went through and
    which verb landed the change are route, and many routes reach one end state.

REPORT-ONLY (``min_pass_rate=None``): the floors and ceilings these runs propose are the
code owner's to accept once the numbers have been read.  All content is synthetic — a
weeknight recipe box, and an invented marketplace listing at an invented price — because the
repo is public.

The gate's classification, the STOP threading and both renders are pinned deterministically
in ``tests/database/test_memory_store.py`` and ``tests/tools/test_memory_tools.py``; this
owns the live model-behaviour contract.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from penny.constants import WRITE_GATE_STOP_REASONS, RunOutcome, WriteGateOutcome
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.skills import (
    SkillDraft,
    SkillParameter,
    SkillStep,
    SkillSubKind,
    SkillSubstitution,
    bind_parameters,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.llm.models import LlmMessage, LlmResponse, LlmToolCall, LlmToolCallFunction
from penny.program import program_calls
from penny.prompts import Prompt
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    CollectorCyclesEval,
    CycleArm,
    _InjectingClient,
    collection_entries,
    seeded_run_id,
)
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    StoredEntry,
)
from penny.tests.eval.utils.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_DUP_CONTENT,
    RECIPE_BOX_DUP_KEY,
    RECIPE_BOX_FAJITAS_KEY,
    RECIPE_BOX_FAJITAS_SEED_CONTENT,
    RECIPE_BOX_SEED_KEYS,
    CannedPage,
)
from penny.tests.eval.utils.worlds import World
from penny.tools.memory_tools import CollectionWriteTool

pytestmark = pytest.mark.eval

# The family every collector recovery case reports under — what a cycle does once one of
# its own calls has already been refused.  A second spelling would split one bucket.
_FAMILY = "collector-recovery"

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The run that laid each entry condition down.  A SEEDED id, structurally distinguishable
# from a live ``uuid4().hex``, so every "what did THIS sample write" reader excludes the
# preseeded rows at the one chokepoint rather than by remembering to.
_SEED_RUN = seeded_run_id("duplicate-entry-condition")

# The verb the divergent-value rejection binds the matched key into.  Named here because
# the probe below asserts it is REACHABLE on a scoped surface, never because a claim reads
# it — which verb landed the change is route.
_RECOVERY_VERB = "update_entry"

# The STOP a re-observation ends on, as the run record states it — read from the shipped
# table rather than restated, so a reworded reason cannot silently stop matching.
_STOP_REASON = WRITE_GATE_STOP_REASONS[WriteGateOutcome.DUPLICATE_UNCHANGED]


def _held_key(entry: StoredEntry) -> str:
    """One held entry's key.  A collection entry always has one; the empty string is the
    keyless-log shape the type allows and neither of these collections ever has."""
    return entry.key or ""


def _held_text(sample: SampleObservation) -> str:
    """The collection as the cycle left it, key and content together.

    The WHOLE entry, because a fact in the key and a blurb in the body is a perfectly good
    way to store it, and a content-only read once reported a 25/32 model failure that was
    entirely its own bug."""
    return " ".join(entry.text for entry in sorted(sample.held, key=_held_key))


def _closed_having_worked(sample: SampleObservation, _world: World) -> Answer:
    """The run's own determination: it completed and it CHANGED something.

    A record field, read literally off ``promptlog.run_outcome`` — ``worked`` is defined as
    "completed and changed something", which is what "the change landed" means without
    reading a tool name.  ``no_work`` is a clean close that changed nothing and ``failed``
    is a bail; both are the wrong end state for a cycle that found news."""
    closed = sample.run_outcome or "—"
    detail = f" — {sample.run_reason}" if sample.run_reason else ""
    return sample.run_outcome == RunOutcome.WORKED.value, f"the run closed {closed}{detail}"


# ── The staged injector ───────────────────────────────────────────────────────


class _InjectDuplicateWriteAfterAStep(_InjectingClient):
    """A forced ``collection_write`` of entries that collide with what the target collection
    already holds, held back until the model's first real tool call has landed **on the main
    loop**.

    **The trigger skips a call carrying no TOOL SURFACE.**  A micro-context calls the same
    client with no tools at all (``MicroContext._draw_clean``), so on a browse-shaped program
    the first call after the model's browse is the browse's own EXTRACT draw — a shared
    after-a-tool-call trigger fires into it, the main loop never sees the forced write, and
    the cycle re-runs its program untouched while ``bail_injected`` reports the sabotage as
    fired.  Reading the tool surface is the structural way to tell the two apart: the loop
    always passes one, a micro-context never does."""

    def __init__(self, real, memory: str, entries: list[tuple[str, str]]) -> None:
        super().__init__(real)
        self._memory = memory
        self._entries = entries
        self._saw_tool = False

    async def chat(self, messages, tools=None, *args, **kwargs):
        if tools is None:
            # A micro-context (no tool channel by construction) — never the injection site.
            return await self._real.chat(messages, *args, tools=tools, **kwargs)
        if self._saw_tool and not self.bail_injected:
            self.bail_injected = True
            return self._bail_response()
        response = await self._real.chat(messages, *args, tools=tools, **kwargs)
        if response.has_tool_calls:
            self._saw_tool = True
        return response

    def _bail_response(self) -> LlmResponse:
        return LlmResponse(
            message=LlmMessage(
                role="assistant",
                tool_calls=[
                    LlmToolCall(
                        id="bail-dup-write-staged",
                        function=LlmToolCallFunction(
                            name="collection_write",
                            arguments={
                                "memory": self._memory,
                                "entries": [
                                    {"key": key, "content": content}
                                    for key, content in self._entries
                                ],
                            },
                        ),
                    )
                ],
            )
        )


# ══ The same-fact world: a recipe box that already holds what was just found ══

_BOX_CASE_ID = "duplicate-write-same-fact-stops"

# The calls the box's routine makes, in order — what the stored program must read back as
# under the strict rendered dialect, and therefore what the cycle's surface is scoped to.
# ``collection_write`` being in that list is what puts it on the surface, which is what
# makes the forced write reach the write gate at all.
_BOX_PROGRAM_CALLS = ("collection_read_latest", "collection_write")

_BOX_SCHEDULE = "FREQ=HOURLY"

_BOX_BEHAVIOUR = (
    "In a recipe-box collector, when the value she has just written is one the box already "
    "holds under a key worded differently this time, Penny writes nothing and tells the "
    "user nothing."
)


# Five wordings of one instruction — the substitution description that becomes ``{…}`` in
# the rendered program at the write step's content leaf, which is the only natural language
# this cycle is handed, and the row a reader opens when the report says a phrasing diverged.
# Every one says the same thing — what a filed recipe consists of — and none of them names a
# dish, because the dishes are the facts and they are constant.
BOX_RECIPE_HINTS = (
    "the recipe's name, its main ingredients and its cook time",
    "what the recipe is, what goes in it and how long it takes",
    "the dish's name plus its core ingredients and its timing",
    "one short line: the recipe, what it needs and how long it cooks",
    "the recipe's title followed by its ingredients and its time",
)


# Every arm's ground, and there is ONE of it: this job reads no page.  A world's pages are
# what a browse-driven cycle is answered against; the box's only source is itself, which
# reaches the provenance claim through what the model was GIVEN.  So the arms genuinely share
# their ground and the report states it once.  ``keeps``/``excludes`` are empty for the reason
# ``test_watch_cycles.py``'s are: they back a claim this case never makes, and declaring
# tokens nothing reads prints a contract that looks verified.
_BOX_WORLD = World(name="recipe-box", pages=(), keeps=(), excludes=())


def _box_skill(recipe_hint: str) -> SkillDraft:
    """The routine the user taught the box, in the shape run-end extraction leaves behind.

    ONE shape for every arm — the same two steps, the same placeholders, the same attachment
    mark on the destination and on the read of the routine's own collection.  It declares NO
    parameter: the job is pointed at nothing beyond the box itself.  What differs is the
    write's CONTENT description, printed by ``render_skill``, which is the arm axis."""
    return SkillDraft(
        name="save_new_recipes",
        intent="Keep a box of quick weeknight dinner recipes I can pull from.",
        description="File a quick weeknight recipe that is not already saved.",
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="collection_read_latest",
                arguments={"memory": RECIPE_BOX.name, "k": 20},
                substitutions=[
                    SkillSubstitution(
                        path=["memory"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the collection this is set up on",
                        attachment=True,
                    )
                ],
            ),
            SkillStep(
                ordinal=2,
                source_ordinal=2,
                tool="collection_write",
                arguments={
                    "memory": RECIPE_BOX.name,
                    "entries": [
                        {
                            "key": "One-pot lemon orzo",
                            "content": (
                                "One-pot lemon orzo — orzo, lemon, spinach, parmesan, 20 min."
                            ),
                        }
                    ],
                },
                substitutions=[
                    SkillSubstitution(
                        path=["memory"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the collection this is set up on",
                        attachment=True,
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "key"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the recipe's name",
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "content"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description=recipe_hint,
                    ),
                ],
            ),
        ],
        parameters=[],
        source_run_id=_SEED_AUTHOR,
    )


def _box_program(recipe_hint: str) -> str:
    """The program the apply turn stores, through the production instantiation seam's own
    three steps: the attachment bound to the container, the runtime join (nothing to join —
    this routine declares no parameter), then the render."""
    skill = _box_skill(recipe_hint)
    attached = retarget_writes(skill.steps, RECIPE_BOX.name)
    joined = bind_parameters(attached, skill.parameters, {})
    return render_skill(joined)


def _box_slot(recipe_hint: str) -> str:
    """How this arm's instruction renders inside the program — the one span that moves."""
    return f"{{{recipe_hint}}}"


def _box_seeder(recipe_hint: str):
    """The world this cycle starts in: the taught routine, the box with the two recipes it
    already holds, and the container configured from that routine — NOTIFYING, because
    whether the user is told is half of what this case scores."""

    def seed(db: Database) -> None:
        skill = _box_skill(recipe_hint)
        db.skills.upsert(skill, author=_SEED_AUTHOR)
        db.memories.create_collection(
            RECIPE_BOX.name,
            RECIPE_BOX.description,
            extraction_prompt=_box_program(recipe_hint),
            schedule=_BOX_SCHEDULE,
            notify=True,
            skill_name=slug_skill_name(skill.name),
            skill_params={},
        )
        require_memory(db, RECIPE_BOX.name).write(
            [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in RECIPE_BOX.entries],
            author=_SEED_AUTHOR,
            run_id=_SEED_RUN,
        )
        _assert_the_box_world(db, recipe_hint)

    return seed


def _assert_the_box_world(db: Database, recipe_hint: str) -> None:
    """Everything the box seeder is responsible for, asserted where it fails loudly.

    An unreadable program yields an EMPTY tool surface, the forced ``collection_write`` then
    comes back tool-not-found instead of reaching the write gate, and the case scores a run
    in which the mechanism under test never spoke — silently, one live cycle per sample."""
    _assert_the_box_job_is_configured(db, recipe_hint)
    _assert_the_box_program_parses(db, recipe_hint)
    _assert_the_recovery_verb_is_reachable()
    _assert_the_box_entry_condition(db)


def _assert_the_box_job_is_configured(db: Database, recipe_hint: str) -> None:
    """The routine is registered and the container is configured as an apply turn leaves it:
    the routine stamped, the turn's schedule, live, and NOTIFYING."""
    name = slug_skill_name(_box_skill(recipe_hint).name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"
    row = db.memories.get(RECIPE_BOX.name)
    assert row is not None, f"the job's container {RECIPE_BOX.name!r} must exist"
    assert row.skill_name == name, f"the job must run the taught routine, not {row.skill_name!r}"
    assert row.schedule == _BOX_SCHEDULE, f"the job must carry its own rule, got {row.schedule!r}"
    assert row.notify and not row.archived, (
        "the job must be live and NOTIFYING — this case scores what reached the user, and on "
        "a silent collection 'nothing was queued' would be true before the cycle ran"
    )


def _assert_the_box_program_parses(db: Database, recipe_hint: str) -> None:
    """The stored program reads back as the calls it makes under the STRICT dialect (#1911),
    names the container it writes to, carries this arm's own instruction, and stores no
    terminal ``done()``.

    ``collection_write`` being IN that list is what puts it on the cycle's scoped surface,
    which is what makes the forced write reach the write gate at all."""
    row = db.memories.get(RECIPE_BOX.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_BOX_PROGRAM_CALLS)))
    assert parsed == _BOX_PROGRAM_CALLS, (
        f"the stored program must read back as {list(_BOX_PROGRAM_CALLS)} under the rendered "
        f"dialect, got {list(parsed)} — program: {program!r}"
    )
    assert f"'{RECIPE_BOX.name}'" in program, (
        f"the attachment must be bound to {RECIPE_BOX.name!r}.  Program: {program!r}"
    )
    assert _box_slot(recipe_hint) in program, (
        f"this arm's instruction must reach the model as {_box_slot(recipe_hint)!r} — the "
        f"recipe description is the whole arm axis.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying one "
        "is a render a chat ledger cannot produce"
    )


def _assert_the_recovery_verb_is_reachable() -> None:
    """The call the divergent-value rejection NAMES is on the cycle's surface.

    A scoped surface is the program's own calls closed over ``Tool.advises``, so the program
    naming ``collection_write`` is only half the requirement: ``update_entry`` reaches the
    surface because ``collection_write`` DECLARES it as advice.  Read off the class attribute
    production closes over, so an advice relation that was dropped fails here rather than as
    an unexplained recovery collapse.  Asserted on BOTH cases: the same-fact one must be able
    to take the divergent door in order for refusing to be a choice."""
    assert _RECOVERY_VERB in CollectionWriteTool.advises, (
        f"{_RECOVERY_VERB!r} must be declared advice of collection_write — the divergent-"
        f"value rejection binds a call to it, and a scoped cycle can only make calls its "
        f"program's advice closure carries.  Declared: {CollectionWriteTool.advises}"
    )


def _assert_the_box_entry_condition(db: Database) -> None:
    """The state the cycle starts from, which is what selects the behaviour being measured.

    The box has to hold the recipe the forced write repeats, at exactly the value it
    repeats, or the collision is a divergent one and this case measures the other door."""
    held = collection_entries(db, RECIPE_BOX.name)
    assert set(held) == set(RECIPE_BOX_SEED_KEYS), (
        f"the box must hold {sorted(RECIPE_BOX_SEED_KEYS)} when the cycle starts, got "
        f"{sorted(held)}"
    )
    assert held[RECIPE_BOX_FAJITAS_KEY] == RECIPE_BOX_FAJITAS_SEED_CONTENT, (
        f"the recipe must start at its seeded value, got {held[RECIPE_BOX_FAJITAS_KEY]!r}"
    )
    assert RECIPE_BOX_DUP_CONTENT == RECIPE_BOX_FAJITAS_SEED_CONTENT, (
        "the forced write must repeat the stored value VERBATIM — that is what makes it "
        "trivially the same value rather than a judgment call"
    )
    assert RECIPE_BOX_DUP_KEY not in held, (
        f"the forced write's key {RECIPE_BOX_DUP_KEY!r} must be one the box does NOT hold — "
        "an exact-key collision is the other door, and this case is about the reworded one"
    )


def _box_arms() -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction over one set of facts."""
    return [
        CycleArm(text=recipe_hint, seed=_box_seeder(recipe_hint), pages=[], world=_BOX_WORLD)
        for recipe_hint in BOX_RECIPE_HINTS
    ]


# ── The same-fact claims ─────────────────────────────────────────────────────


def _holds_only_the_seeded_recipes(sample: SampleObservation, _world: World) -> Answer:
    """The box holds exactly the recipes it was seeded with — no key more, no key fewer.

    This is the ABSENCE half of the behaviour, and it has a concrete violating shape: a
    THIRD key, a recipe the model invented in order to have something to file after being
    told the one it wrote was already saved.  That is the measured failure the structural
    STOP exists to remove, and it is what this claim goes red on."""
    keys = sorted(_held_key(entry) for entry in sample.held)
    expected = sorted(RECIPE_BOX_SEED_KEYS)
    return keys == expected, f"the box holds {keys}, against the seeded {expected}"


def _stopped_on_the_recorded_value(sample: SampleObservation, _world: World) -> Answer:
    """The run stopped at the write chokepoint, on the reason the shipped table names.

    This is what makes silence STRUCTURAL rather than a judgement the model makes each
    cycle: the gate reads the candidate as a value already recorded and raises its STOP on
    the very call that would otherwise complete the program, so the notification the
    framework would have entered is never entered at all."""
    return sample.run_reason == _STOP_REASON, f"the run closed {sample.run_reason or '—'}"


def _told_nothing(sample: SampleObservation, _world: World) -> Answer:
    """Nothing reached the send queue.  No news is not a message saying there is no news.

    Read over the SEND QUEUE, which is what the user will actually receive — a cycle
    enqueues and the drainer is a separate schedule, so a pending-only read of outgoing
    messages reports a delivered notification as silence."""
    return not sample.notifications, f"sent {sample.notifications}"


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_value_already_recorded_stops_the_cycle(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A forced write repeating a stored value VERBATIM under a reworded key is the same
    no-news the exact key reads: the cycle stops at the chokepoint, nothing is written and
    the user hears nothing."""
    cohort = await collector_cycles_eval(
        case_id=_BOX_CASE_ID,
        behaviour=_BOX_BEHAVIOUR,
        model=model,
        collection=RECIPE_BOX.name,
        arms=_box_arms(),
        samples_per_phrasing=3,
        wrap_client=lambda real: _InjectDuplicateWriteAfterAStep(
            real, RECIPE_BOX.name, [(RECIPE_BOX_DUP_KEY, RECIPE_BOX_DUP_CONTENT)]
        ),
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.

    # STORE
    cohort.claim(
        "state: the box still holds exactly the recipes it started with",
        _holds_only_the_seeded_recipes,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the run stopped on the value already being recorded",
        _stopped_on_the_recorded_value,
        SpecCategory.STORE,
    )
    cohort.claim("state: nothing reached the send queue", _told_nothing, SpecCategory.STORE)

    # PROVENANCE — a correct cycle wrote nothing, so this claim is TRUE of it rather than
    # unasked; what it catches is a quiet cycle that filed a recipe no page and no entry
    # ever mentioned.
    cohort.assert_every_stored_entry_traces_to_the_world()

    # REPLY_SPREAD is not measured, and that is the point of the case: a correct cohort sends
    # nothing, so a reply-spread reading would be blind on every sample by construction.  A
    # sample that DID speak is caught by the send-queue claim.
    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED)


# ══ The divergent world: a watch whose tracked value has moved ════════════════

_WATCH_CASE_ID = "duplicate-write-divergent-value-updates"

# The page the watch is pointed at, and the container it records into.  Invented — no real
# listing, no real price.
_LANTERN_PAGE = "https://faux-market.example/keel-lantern"
_LANTERN_CONTAINER = "keel-lantern-watch"
_LANTERN_DESCRIPTION = "What the Keel Lantern listing is posted at right now."

# The calls the watch makes, in order.
_WATCH_PROGRAM_CALLS = ("browse", "collection_write")

_WATCH_SCHEDULE = "FREQ=HOURLY"

# How ``render_skill`` writes a leaf that takes a prior step's result — the watch's recorded
# value, which is what makes it a watch: it stores what the page said.
_BINDING_RENDER = "the value from step 1"

# ── The facts, held constant across every arm ────────────────────────────────
#
# One listing, one subject, one pair of readings.  Constant is what lets the claims below
# name a value: five listings would force every one of them back to a shape claim, and a
# shape claim cannot tell a watch that recorded the right reading from one that recorded a
# plausible number.
_SUBJECT = "Keel Lantern"

# The two readings, in their two roles.  The AMOUNT is the identity of the reading and what
# every claim matches on; the PRICE is how a listing displays it, and it is what the pages
# carry.  They are MUTUALLY EXCLUSIVE as numerals — neither is a substring of the other —
# because the case asserts one is present and the other gone, and no other figure on any
# page contains either as a substring.
_STORED_AMOUNT = "128"
_MOVED_AMOUNT = "96"
_STORED_VALUE = f"{_SUBJECT} — posted at ${_STORED_AMOUNT}."
_MOVED_VALUE = f"{_SUBJECT} — posted at ${_MOVED_AMOUNT}."

# The key the watch already files its subject under, and the key the forced write arrives
# with.  TOKEN-different, not a case variant: the similarity door folds case when it
# tokenizes, and the exact-key door may resolve the two together — either way the write
# would never reach the divergent rejection this case is about.  An extra token still
# collides on containment (what makes it the same subject) while being a genuinely different
# string (what makes it a new key).
_LANTERN_KEY = _SUBJECT
_LANTERN_REWORDED_KEY = "keel lantern price"

# The watched line itself, byte-identical on all five pages, and labelled as THE current
# posting so the neighbouring items' prices further down each page cannot be read as it.
_DATUM = f"Posted at ${_MOVED_AMOUNT}"

_WATCH_BEHAVIOUR = (
    "In a listing-watch collector, when the changed reading she has just written arrives "
    "under a key worded differently from the one it is already filed under, Penny lands it "
    "on the entry that exists and tells the user once, naming the reading it moved to."
)


class WatchReading(NamedTuple):
    """One arm of the divergent case: how the job's ``extract`` step words what to look for,
    and the page prose that answers it.

    ``name`` is what this arm is called wherever it is identified.  ``extract`` is the
    substitution description that becomes ``extract={…}`` in the rendered program.  ``body``
    is this arm's whole page."""

    name: str
    extract: str
    body: str


_HEAD = f"Title: {_SUBJECT} — a small brass lantern | faux-market\n{_LANTERN_PAGE}\n\n"
_LISTING_LINK = f"[{_SUBJECT} listing]({_LANTERN_PAGE})"
_MAKER_LINK = f"[about the maker]({_LANTERN_PAGE}/maker)"

# Five catalogue-grade pages: same url, same item, same posting, five voices.  Each carries
# far more than the job needs — a maker's blurb, a materials block, neighbouring items with
# their OWN prices, housekeeping notes — because a real page does, and because a page thin
# enough to answer only the asked question cannot tell restraint from luck.  The neighbours
# are the load-bearing part: a cycle that grabs a figure is only demonstrably reading the
# RIGHT one where wrong ones are on the page to grab.  No figure anywhere carries `96` or
# `128` as a substring, which is what keeps the claims' two amounts unique in this world.
# Every markdown link sits at the CENTRE of its block, since a search-shaped read is trimmed
# to ±2 lines around each solo link.
READINGS = (
    WatchReading(
        "posted-at",
        "what it is posted at",
        _HEAD + f"{_SUBJECT}, made and sold by one workshop. Stock is small and moves slowly.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: harbourlight_forge (4.9 stars). The posting is reviewed every Monday.\n"
        "Condition: new, unlit. Dispatched within three working days.\n"
        "\n"
        "Materials\n"
        "Spun brass body with a hand-blown glass chimney\n"
        f"{_MAKER_LINK}\n"
        "Burns about five hours on a full reservoir; the wick is replaceable.\n"
        "Weight 610g · Boxed with a spare wick · One trimming tool included\n"
        "\n"
        "Others from this workshop\n"
        "Keel Lantern, wall bracket — $44\n"
        "[wall bracket](https://faux-market.example/keel-bracket)\n"
        "Spare glass chimney — $22\n"
        "Everything ships from the same fictional workshop.\n"
        "\n"
        "Returns are accepted within thirty days; return postage is the buyer's.\n",
    ),
    WatchReading(
        "asking-now",
        "what the listing is asking right now",
        _HEAD + "A fictional workshop listing. The bench reprices its stock as materials move.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Offered by tidewater_brass (4.8 stars). What it asks moves with what brass costs.\n"
        "Condition: new, unlit. Three working days to dispatch.\n"
        "\n"
        "What you get\n"
        f"{_SUBJECT} · spun brass · hand-blown glass chimney\n"
        f"{_MAKER_LINK}\n"
        "About five hours of burn per fill; wick and chimney are serviceable parts.\n"
        "Weight 610g · Original box · Spare wick in the tin\n"
        "\n"
        "Also on the bench\n"
        "Keel Lantern, table stand — $37\n"
        "[table stand](https://faux-market.example/keel-stand)\n"
        "Wick tin, six pack — $11\n"
        "All of it ships from one fictional bench.\n"
        "\n"
        "Thirty-day returns. The buyer pays return postage.\n",
    ),
    WatchReading(
        "current-figure",
        "the figure the listing is currently at",
        _HEAD + "Listed by a fictional chandlery that revises its board every few days.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: quayside_chandlery (4.7 stars). The board is revised as stock turns over.\n"
        "Condition: new, unlit. Ready to dispatch inside three working days.\n"
        "\n"
        "Made from\n"
        "Spun brass, with a glass chimney blown on the same premises\n"
        f"{_MAKER_LINK}\n"
        "Roughly five hours of light per fill; a workshop can replace the wick.\n"
        "Weight 610g · Boxed as it came · Trimming tool included\n"
        "\n"
        "On the same board\n"
        "Keel Lantern, brass hook — $23\n"
        "[brass hook](https://faux-market.example/keel-hook)\n"
        "Moulded carry case — $51\n"
        "The chandlery posts its board every morning.\n"
        "\n"
        "Returns within thirty days, postage paid by the buyer.\n",
    ),
    WatchReading(
        "listed-today",
        "the amount it is listed at today",
        _HEAD + "A fictional clearance shelf. Today's board, republished each morning.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: pier_clearance (4.6 stars). The board is set each day at opening.\n"
        "Condition: new, unlit. Three working days from order to dispatch.\n"
        "\n"
        "Specification\n"
        "Body spun brass · Chimney hand-blown glass\n"
        f"{_MAKER_LINK}\n"
        "About five hours of burn; the reservoir is a serviceable part.\n"
        "Weight 610g · Original box and spare wick · Trimming tool\n"
        "\n"
        "Today's other lamps\n"
        "Keel Lantern, boxed pair — $175\n"
        "[boxed pair](https://faux-market.example/keel-pair)\n"
        "Glass polish, twin pack — $14\n"
        "The whole shelf ships from one fictional warehouse.\n"
        "\n"
        "Returns accepted for thirty days; the buyer covers return postage.\n",
    ),
    WatchReading(
        "page-shows",
        "the amount this listing shows",
        _HEAD + "A fictional consignment listing. The page shows whatever the owner last agreed.\n"
        f"{_DATUM}\n"
        f"{_LISTING_LINK}\n"
        "Seller: lantern_consign (4.9 stars). The posting is whatever this page shows.\n"
        "Condition: new, unlit. Dispatch within three working days.\n"
        "\n"
        "Materials sheet\n"
        "Spun brass · hand-blown glass chimney · 610g\n"
        f"{_MAKER_LINK}\n"
        "The reservoir burns about five hours and the wick is replaceable.\n"
        "Spare wick, trimming tool and the original box are included.\n"
        "\n"
        "Elsewhere in this consignment\n"
        "Keel Lantern, wall bracket — $47\n"
        "[wall bracket](https://faux-market.example/keel-bracket)\n"
        "Chimney and wick set — $33\n"
        "All consigned stock ships from one fictional store.\n"
        "\n"
        "Thirty days to return; return postage is the buyer's.\n",
    ),
)


def _watch_skill(reading: WatchReading) -> SkillDraft:
    """The watch, in the shape run-end extraction leaves behind: the page a PARAMETER site
    (its demonstrated value is what the runtime join fills), the extract instruction and the
    entry key labelled PLACEHOLDERs, the destination attachment-marked, and the recorded
    value BOUND to the browse's own result — which is what makes this a watch rather than a
    generator: what it writes is what the page said."""
    return SkillDraft(
        name="track_posted_value",
        intent="Keep an eye on what the Keel Lantern listing is posted at.",
        description="Read a listing and record what it is posted at right now.",
        steps=[
            SkillStep(
                ordinal=1,
                source_ordinal=1,
                tool="browse",
                arguments={"queries": [_LANTERN_PAGE], "extract": "what it is posted at"},
                substitutions=[
                    SkillSubstitution(
                        path=["queries", 0],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the listing to read",
                    ),
                    SkillSubstitution(
                        path=["extract"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description=reading.extract,
                    ),
                ],
            ),
            SkillStep(
                ordinal=2,
                source_ordinal=2,
                tool="collection_write",
                arguments={
                    "memory": _LANTERN_CONTAINER,
                    "entries": [{"key": _LANTERN_KEY, "content": _STORED_VALUE}],
                },
                substitutions=[
                    SkillSubstitution(
                        path=["memory"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the collection this is set up on",
                        attachment=True,
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "key"],
                        kind=SkillSubKind.PLACEHOLDER,
                        description="the thing being tracked",
                    ),
                    SkillSubstitution(
                        path=["entries", 0, "content"],
                        kind=SkillSubKind.BINDING,
                        step=1,
                    ),
                ],
            ),
        ],
        parameters=[
            SkillParameter(
                name="listing",
                description="the listing page to read each run",
                value=_LANTERN_PAGE,
            )
        ],
        source_run_id=_SEED_AUTHOR,
    )


_WATCH_VALUES = {"listing": _LANTERN_PAGE}


def _watch_program(reading: WatchReading) -> str:
    """The watch's program, through the production instantiation seam's own three steps."""
    skill = _watch_skill(reading)
    attached = retarget_writes(skill.steps, _LANTERN_CONTAINER)
    joined = bind_parameters(attached, skill.parameters, _WATCH_VALUES)
    return render_skill(joined, _WATCH_VALUES)


def _extract_slot(reading: WatchReading) -> str:
    """How this arm's instruction renders inside the program — the one span that moves."""
    return f"extract={{{reading.extract}}}"


def _watch_page(reading: WatchReading) -> CannedPage:
    """The page THIS arm's cycle reads."""
    return CannedPage(match="keel-lantern", text=reading.body)


def _watch_world(reading: WatchReading) -> World:
    """This arm's ground: the page its single cycle reads.

    ``keeps``/``excludes`` are EMPTY, and deliberately.  Those token sets back
    ``assert_something_from_each_page_was_written``, which this case never calls — and there
    is nothing they could usefully hold: a ``keeps`` token has to appear on ONE page so a
    stored copy says which page it came from, and every arm here reads the same url, the same
    item and the same posting."""
    return World(name=reading.name, pages=(_watch_page(reading),), keeps=(), excludes=())


def _watch_seeder(reading: WatchReading):
    """The world this cycle starts in: the taught watch in the registry, the container
    already holding the reading it recorded last time, and the job configured from that
    routine — NOTIFYING, because being told once is half of what this case scores."""

    def seed(db: Database) -> None:
        skill = _watch_skill(reading)
        db.skills.upsert(skill, author=_SEED_AUTHOR)
        db.memories.create_collection(
            _LANTERN_CONTAINER,
            _LANTERN_DESCRIPTION,
            extraction_prompt=_watch_program(reading),
            schedule=_WATCH_SCHEDULE,
            notify=True,
            skill_name=slug_skill_name(skill.name),
            skill_params=_WATCH_VALUES,
        )
        require_memory(db, _LANTERN_CONTAINER).write(
            [EntryInput(key=_LANTERN_KEY, content=_STORED_VALUE)],
            author=_SEED_AUTHOR,
            run_id=_SEED_RUN,
        )
        _assert_the_watch_world(db, reading)

    return seed


def _assert_the_watch_world(db: Database, reading: WatchReading) -> None:
    """Everything the watch seeder is responsible for, asserted where it fails loudly — the
    same silent-failure surface the box's probe covers, on this world's own claims."""
    _assert_the_watch_job_is_configured(db, reading)
    _assert_the_watch_program_parses(db, reading)
    _assert_the_recovery_verb_is_reachable()
    _assert_the_watch_entry_condition(db, reading)


def _assert_the_watch_job_is_configured(db: Database, reading: WatchReading) -> None:
    """The routine is registered and the container is configured as an apply turn leaves it:
    the routine stamped, live, and NOTIFYING."""
    name = slug_skill_name(_watch_skill(reading).name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"
    row = db.memories.get(_LANTERN_CONTAINER)
    assert row is not None, f"the job's container {_LANTERN_CONTAINER!r} must exist"
    assert row.skill_name == name, f"the job must run the taught routine, not {row.skill_name!r}"
    assert row.notify and not row.archived, (
        "the watch must be live and NOTIFYING — this case scores being told exactly once"
    )


def _assert_the_watch_program_parses(db: Database, reading: WatchReading) -> None:
    """The stored program reads back as the calls it makes under the STRICT dialect (#1911),
    carries the page the runtime join filled and this arm's own instruction, names the
    container it writes to, and stores no terminal ``done()``.

    The recorded value is a BINDING (``the value from step 1``) rather than a placeholder,
    which is the whole difference between a watch and a generator: what this routine writes
    is what the page said, so a changed reading belongs on the entry that holds the last
    one."""
    row = db.memories.get(_LANTERN_CONTAINER)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_WATCH_PROGRAM_CALLS)))
    assert parsed == _WATCH_PROGRAM_CALLS, (
        f"the stored program must read back as {list(_WATCH_PROGRAM_CALLS)} under the "
        f"rendered dialect, got {list(parsed)} — program: {program!r}"
    )
    assert _LANTERN_PAGE in program, (
        f"the runtime join must fill the browse leaf with {_LANTERN_PAGE!r} — a cycle can "
        f"only fetch the page something it reads names.  Program: {program!r}"
    )
    assert f"'{_LANTERN_CONTAINER}'" in program, (
        f"the attachment must be bound to {_LANTERN_CONTAINER!r}.  Program: {program!r}"
    )
    assert _extract_slot(reading) in program, (
        f"this arm's instruction must reach the model as {_extract_slot(reading)!r} — the "
        f"extract description is the whole arm axis.  Program: {program!r}"
    )
    assert _BINDING_RENDER in program, (
        "the recorded value must render as the browse's own result — a watch writes what the "
        f"page said, not a fresh phrase.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying one "
        "is a render a chat ledger cannot produce"
    )


def _assert_the_watch_entry_condition(db: Database, reading: WatchReading) -> None:
    """The state the cycle starts from, and the two amounts the claims name.

    Uniqueness is a property of the WORLD, not of the token: the page must carry the moved
    amount and must NOT carry the stored one, or a correct write quoting the page could hold
    both and the "no longer" half would fail a run that did everything right."""
    held = collection_entries(db, _LANTERN_CONTAINER)
    assert held == {_LANTERN_KEY: _STORED_VALUE}, (
        f"the watch must hold exactly its last reading when the cycle starts, got {held}"
    )
    assert _MOVED_AMOUNT in reading.body and _STORED_AMOUNT not in reading.body, (
        f"this arm's page must carry {_MOVED_AMOUNT!r} and not {_STORED_AMOUNT!r} — the "
        "claims name both amounts, and a page mentioning the old one would let a correct "
        "write carry it honestly"
    )
    assert reading.body.count(_DATUM) == 1, (
        f"the watched line {_DATUM!r} must appear exactly once on this arm's page — a second "
        "occurrence is a figure the case never named"
    )
    assert _MOVED_VALUE != _STORED_VALUE, (
        "the forced write must carry a value the watch does NOT already hold, else it is the "
        "same no-news the other case scores"
    )
    assert _LANTERN_REWORDED_KEY.casefold() != _LANTERN_KEY.casefold(), (
        "the forced write's key must differ by a TOKEN, not by case — the similarity door "
        "folds case when it tokenizes and the exact-key door may resolve the two together, "
        "so a case-only variant never reaches the rejection this case is about"
    )


def _watch_arms() -> list[CycleArm]:
    """This case's five arms — five wordings of one instruction over five prose variants of
    one page, around a byte-identical datum line."""
    return [
        CycleArm(
            text=reading.extract,
            seed=_watch_seeder(reading),
            pages=[_watch_page(reading)],
            world=_watch_world(reading),
        )
        for reading in READINGS
    ]


# ── The divergent claims ─────────────────────────────────────────────────────


def _holds_the_moved_reading_only(sample: SampleObservation, _world: World) -> Answer:
    """The watch holds the reading the page moved TO, and no longer the one it moved FROM.

    Both halves are needed and neither is sufficient — a watch that appended the new reading
    beside the old one holds ``96`` while still telling the user something that is no longer
    true, and a watch that never landed the change holds neither.

    Matched on the bare AMOUNTS.  The identity of a reading is its number — that is what
    moves on the page, and what a watch exists to track — while which currency notation a
    draw happened to keep is not a question this claim answers."""
    held = _held_text(sample)
    ok = _MOVED_AMOUNT in held and _STORED_AMOUNT not in held
    return ok, f"expected {_MOVED_AMOUNT} and not {_STORED_AMOUNT}; the watch holds {held!r}"


def _tracks_one_subject(sample: SampleObservation, _world: World) -> Answer:
    """The watch keeps ONE fact.  A watch that files the new reading beside the old one is
    tracking its subject twice, and the user is told about a value they already have."""
    keys = sorted(_held_key(entry) for entry in sample.held)
    return len(sample.held) == 1, f"the watch holds {len(sample.held)} entries: {keys}"


def _told_once(sample: SampleObservation, _world: World) -> Answer:
    """The user was told, and told once.

    Read over the SEND QUEUE, which is what the user will actually receive — a cycle
    enqueues and the drainer is a separate schedule, so a pending-only read of outgoing
    messages reports a delivered notification as silence."""
    count = len(sample.notifications)
    return count == 1, f"{count} messages reached the send queue: {sample.notifications}"


def _the_message_states_the_new_reading(sample: SampleObservation, _world: World) -> Answer:
    """What reached the user names the reading it moved to.

    The COMPLETENESS half of being told: a message that reaches the queue, is whole, and
    names nothing the user did not already know has not told them their watch moved.  Read
    on the bare amount, which is digits and therefore strictly identifiable; every other word
    in the message is the model's to choose, which is why this claim is the ``reply`` kind and
    is never offered a floor."""
    said = "\n".join(sample.notifications)
    return (
        _MOVED_AMOUNT in said,
        f"the message never states {_MOVED_AMOUNT}: {sample.notifications}",
    )


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_divergent_value_lands_on_the_entry_that_exists(
    collector_cycles_eval: CollectorCyclesEval, model: str
) -> None:
    """A watch reads its page, finds the tracked reading has MOVED, and its write arrives
    under a key worded differently from the one the reading is already filed under: the new
    reading lands on that entry, the watch still tracks one subject, and the user is told
    once, naming what it moved to.

    **What this case is FOR, and how it differs from the moved-reading watch case.**
    ``test_watch_cycles.py``'s moved case drives the same shape — a watch whose page moved,
    one notification owed — and this differs from it by ONE thing: the NAMING SLIP.  The
    write arrives under a key worded differently from the one the reading is filed under, so
    it goes through the dedup disjunction rather than the exact key.

    That one difference is the whole point, because it makes this the COUNTERWEIGHT to
    ``DUPLICATE_UNCHANGED``'s STOP.  The same similarity machinery that lets a re-observation
    be recognised as no-news and silence the cycle is the machinery a CHANGED value now
    passes through — so this is the standing proof that a change can never be silenced by it.
    Concretely it is the regression net for the strict content signal: if a real value change
    ever reads as the same value, the collision classifies as no-news, the cycle STOPs, and
    this case goes red before a user stops being told their watch moved."""
    cohort = await collector_cycles_eval(
        case_id=_WATCH_CASE_ID,
        behaviour=_WATCH_BEHAVIOUR,
        model=model,
        collection=_LANTERN_CONTAINER,
        arms=_watch_arms(),
        samples_per_phrasing=3,
        wrap_client=lambda real: _InjectDuplicateWriteAfterAStep(
            real, _LANTERN_CONTAINER, [(_LANTERN_REWORDED_KEY, _MOVED_VALUE)]
        ),
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
    )
    # LANDED — nothing; see the module docstring.

    # STORE
    cohort.claim(
        f"state: the watch holds {_MOVED_AMOUNT} and no longer {_STORED_AMOUNT}",
        _holds_the_moved_reading_only,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the watch still tracks exactly the one subject it started with",
        _tracks_one_subject,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the run closed having changed something", _closed_having_worked, SpecCategory.STORE
    )
    cohort.claim("state: the user was told once", _told_once, SpecCategory.STORE)
    cohort.claim(
        f"reply: the message states the reading it moved to ({_MOVED_AMOUNT})",
        _the_message_states_the_new_reading,
        SpecCategory.STORE,
        kind="reply",
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()

    cohort.measure(TOOL_SEQUENCE, TRANSITIONS, ENTRIES_STORED, REPLY_SPREAD)
