"""The two doors onto a stored entry, and what a collector cycle owes at each (#1919).

A collector's write can land on an entry that already exists in two quite different
situations, and until now the write gate called both a DUPLICATE and handed both the
same recoverable rejection:

  same value      — the cycle looked and found what it already had, filed under a key it
                    worded differently this time.  Nothing happened, so the cycle STOPs
                    at the chokepoint and the user hears nothing — the same no-news
                    ``KEY_EXISTS_UNCHANGED`` reads on the exact key, reached by the other
                    door (``DUPLICATE_UNCHANGED``, in ``WRITE_GATE_STOP_REASONS``).
  divergent value — the cycle found something DIFFERENT about a thing it already tracks.
                    That is news: the rejection binds the matched key into an
                    ``update_entry`` call, the change lands on the entry that exists, and
                    the user is told once.  It must NEVER stop.

**Why the split is structural rather than a stronger scorer.**  Before it, the same-value
door produced a rejection saying an entry like this already existed and inviting the model
to refresh it or write something else — and on the first measured run all ten samples
"recovered" by generating a DIFFERENT recipe.  Read against the surface they were given
that is correct behaviour: the routine says to file recipes not already saved, the gate
said this one was, and nothing said the cycle was over.  A cycle that found nothing new
had no way to say so, so the fix is to give it one rather than to tell the model harder
not to invent (the rational-actor doctrine — fix the state, not the imperative).

The slip is a model DECISION on a visible tool result, but a natural cycle only rarely
writes either shape on demand, so each case FORCES one ``collection_write`` and lets the
REAL model drive what follows.  Both collections NOTIFY, because "was the user told?" is
half of what the split decides and a silent collection cannot answer it.  The contract is
STRUCTURAL, never wording.

**WHEN the write is forced is part of the design, not a detail.**  An injector's response
is synthetic and bypasses the persisting client (#1695), so a forced call that ENDS the
cycle leaves the run with no promptlog row at all — nothing for ``set_run_outcome`` to
stamp, an empty run record, and every cycle-shaped check reading an absent ledger as
though the model had done nothing.  That is precisely what happened on the same-fact
case's first measured run: the STOP fired correctly on the injected first call, and five
samples reported it as behavioural failure.  So BOTH writes are staged AFTER the model's
first real step (``_InjectDuplicateWriteAfterAStep``) and both scorers open on a RAN-GUARD,
which fails loudly rather than silently when a run leaves nothing behind.

**Each case's ROUTINE is what makes its outcome the routine's own job (#1919).**  The
divergent case first ran on the recipe box, and its thinking showed the model serving that
routine faithfully: *"We must submit a new recipe… not already saved."*  A routine whose
job is to FILE SOMETHING NEW is answered correctly by writing something new, and no
rejection wording outranks the program the cycle was given — so the case was measuring a
generative mandate rather than the decision it meant to.  It runs on a WATCH now: a page is
read and the current value about ONE tracked subject is recorded, so landing a changed value
on the entry that already holds it is not a recovery the model has to be talked into, it is
the routine carrying out its own job.  The same-fact case stays on the recipe box, where
finding nothing new is exactly the tension its STOP resolves.

**The world is the post-#1911 one, and that is what re-arms these cases.**  The program
parser is STRICT (a step's call must OPEN the step, in the rendered ``N. tool(args)``
dialect) and the cycle's tool surface is SCOPED to that program's own calls, closed over
``Tool.advises``.  The hand-authored prose recipe this module used to seed carried its
write mid-sentence, so the parse found nothing, ``collection_write`` was not on the
surface at all, and the injected write hit tool-not-found instead of the write gate — the
contract silently disarmed, and nothing in the scoring could tell.  The collection is now
what an apply turn leaves behind (migration 0108 pre-seeds nothing): a taught routine in
the registry, and a container configured from it through the production instantiation seam
(retarget → ``bind_parameters`` → ``render_skill``).  ``_assert_the_box_world`` asserts
every link in that chain out loud at seed time.

Report-only (``min_pass_rate=None``), the canonical convention, pending a joint read.  The
gate's classification, the STOP threading and both renders are pinned deterministically in
``tests/database/test_memory_store.py`` and ``tests/tools/test_memory_tools.py``; this owns
the live model-behaviour contract.
"""

from __future__ import annotations

import re

import pytest

from penny.constants import WRITE_GATE_STOP_REASONS, WriteGateOutcome
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
    Check,
    _InjectAfterToolCall,
    _iter_prompt_messages,
    collection_entries,
    live_prompts,
    queued_sends,
    tool_call_sequence,
    tool_was_called,
)
from penny.tests.eval.fixtures import (
    RECIPE_BOX,
    RECIPE_BOX_DUP_CONTENT,
    RECIPE_BOX_DUP_KEY,
    RECIPE_BOX_FAJITAS_KEY,
    RECIPE_BOX_FAJITAS_SEED_CONTENT,
    RECIPE_BOX_SEED_KEYS,
    CannedPage,
)
from penny.tools.memory_tools import CollectionWriteTool

pytestmark = pytest.mark.eval

# The author on a seeded registry row — a fixture's own hand, never a real agent's.
_SEED_AUTHOR = "eval-seed"

# The calls the routine makes, in order — what the program must read back as under the
# strict dialect, and therefore what the cycle's surface is scoped to.
RECIPE_BOX_PROGRAM_CALLS = ("collection_read_latest", "collection_write")

# The job's cadence — stated because a configured collection has one, though both cases
# drive the cycle through ``run_for``, which bypasses readiness.
RECIPE_BOX_SCHEDULE = "FREQ=HOURLY"

# The verb the divergent-value rejection binds the matched key into.
_RECOVERY_VERB = "update_entry"

# The matched key as that rejection renders it: ``To land the new value:
# update_entry(key='<matched>', content=<the new value>)``.  Matched rather than
# reproduced, so the probe reads the SAME text the model read.
_BOUND_KEY = re.compile(rf"{_RECOVERY_VERB}\(key='([^']*)'")

# A substring of the SAME-VALUE render (``_format_duplicate_unchanged``) — the no-news
# answer, which names no call at all.  The full wording is pinned in
# ``tests/tools/test_memory_tools.py``.
_ALREADY_RECORDED = "Already recorded"

# How ``render_skill`` writes a leaf that takes a prior step's result — the watch's
# recorded value, which is what makes it a watch: it stores what the page said.
_BINDING_RENDER = "the value from step 1"

# The routine the user taught the box, in the shape run-end extraction leaves behind:
# every leaf a labelled PLACEHOLDER, the destination and the read of the routine's own
# collection both carrying the attachment mark (#1783).  It declares NO parameter — the
# job is pointed at nothing beyond the box itself, which the composed prompt states out
# loud ("The values it is pointed at: none — this routine takes none").
RECIPE_BOX_SKILL = SkillDraft(
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
                        "content": "One-pot lemon orzo — orzo, lemon, spinach, parmesan, 20 min.",
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
                    description="the recipe's name, its main ingredients and its cook time",
                ),
            ],
        ),
    ],
    parameters=[],
    source_run_id="eval-seed",
)


def rendered_program() -> str:
    """The program the apply turn stores, through the production instantiation seam's
    own three steps in its own order (``render_skill_prompt``): the attachment bound to
    the container, the runtime join (nothing to join — this routine declares no
    parameter), then the render.  Public because the seeder and the probe both read it,
    and a second copy would be free to drift from what the collection stores."""
    attached = retarget_writes(RECIPE_BOX_SKILL.steps, RECIPE_BOX.name)
    joined = bind_parameters(attached, RECIPE_BOX_SKILL.parameters, {})
    return render_skill(joined)


def _seed_recipe_box(db: Database) -> None:
    """The world an apply turn leaves: the taught routine in the registry, the box with
    the two recipes it already holds, and the container configured from that routine —
    NOTIFYING, because whether the user is told is half of what each case scores."""
    db.skills.upsert(RECIPE_BOX_SKILL, author=_SEED_AUTHOR)
    db.memories.create_collection(
        RECIPE_BOX.name,
        RECIPE_BOX.description,
        extraction_prompt=rendered_program(),
        schedule=RECIPE_BOX_SCHEDULE,
        notify=True,
        skill_name=slug_skill_name(RECIPE_BOX_SKILL.name),
        skill_params={},
    )
    require_memory(db, RECIPE_BOX.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in RECIPE_BOX.entries],
        author="user",
    )
    _assert_the_box_world(db)


# ── The loud seed probe ───────────────────────────────────────────────────────


def _assert_the_box_world(db: Database) -> None:
    """Everything the seeder is responsible for, asserted out loud.

    The disarm this module was in is the reason: an unreadable program yields an EMPTY
    tool surface, the injected ``collection_write`` then comes back tool-not-found
    instead of reaching the write gate, and the case scores a run in which the mechanism
    under test never spoke — silently, one live cycle per sample."""
    _assert_the_routine_is_registered(db)
    _assert_the_job_is_configured(db)
    _assert_the_program_parses(db)
    _assert_the_recovery_verb_is_reachable()
    _assert_the_box_holds_its_recipes(db)


def _assert_the_routine_is_registered(db: Database) -> None:
    """The routine the container names is one the registry holds — else the composed
    prompt states the routine as gone."""
    name = slug_skill_name(RECIPE_BOX_SKILL.name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"


def _assert_the_job_is_configured(db: Database) -> None:
    """The container the cycle runs is configured as an apply turn leaves it: the routine
    stamped, the turn's schedule, live, and NOTIFYING — a silent collection could not
    answer either case's question about what reached the user."""
    row = db.memories.get(RECIPE_BOX.name)
    assert row is not None, f"the job's container {RECIPE_BOX.name!r} must exist"
    assert row.skill_name == slug_skill_name(RECIPE_BOX_SKILL.name), (
        f"the job must run the routine the fixture taught, not {row.skill_name!r}"
    )
    assert row.schedule == RECIPE_BOX_SCHEDULE, (
        f"the job must carry its own rule, got {row.schedule!r}"
    )
    assert row.notify and not row.archived, (
        "the job must be live and NOTIFYING — both cases score what reached the user, and "
        "on a silent collection 'nothing was queued' would be true before the cycle ran"
    )


def _assert_the_program_parses(db: Database) -> None:
    """The stored program reads back as the calls it makes, under the STRICT dialect
    (#1911), names the container it writes to, and stores no terminal ``done()``.

    ``collection_write`` being IN that list is what puts it on the cycle's scoped
    surface, which is what makes the injected write reach the write gate at all."""
    row = db.memories.get(RECIPE_BOX.name)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(
        call.tool for call in program_calls(program, frozenset(RECIPE_BOX_PROGRAM_CALLS))
    )
    assert parsed == RECIPE_BOX_PROGRAM_CALLS, (
        f"the stored program must read back as {list(RECIPE_BOX_PROGRAM_CALLS)} under the "
        f"rendered dialect, got {list(parsed)} — program: {program!r}"
    )
    assert f"'{RECIPE_BOX.name}'" in program, (
        f"the attachment must be bound to {RECIPE_BOX.name!r} — a program carrying the "
        f"placeholder does not state where it writes.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying "
        "one is a render a chat ledger cannot produce"
    )


def _assert_the_recovery_verb_is_reachable() -> None:
    """The call the divergent-value rejection NAMES is on the cycle's surface.

    A scoped surface is the program's own calls closed over ``Tool.advises``, so the
    program naming ``collection_write`` is only half the requirement: ``update_entry``
    reaches the surface because ``collection_write`` DECLARES it as advice.  Read off
    the class attribute production closes over, so an advice relation that was dropped
    fails here rather than as an unexplained recovery collapse."""
    assert _RECOVERY_VERB in CollectionWriteTool.advises, (
        f"{_RECOVERY_VERB!r} must be declared advice of collection_write — the divergent-"
        f"value rejection binds a call to it, and a scoped cycle can only make calls its "
        f"program's advice closure carries.  Declared: {CollectionWriteTool.advises}"
    )


def _assert_the_box_holds_its_recipes(db: Database) -> None:
    """The box holds exactly the two seeded recipes, with the fajitas one at the content
    both cases measure against — the value the same-fact write repeats verbatim, and the
    one the divergent write changes.

    The entries' VECTORS are not asserted here and cannot be: the runner backfills them
    right after this seed runs, which is what makes the same-value signal scorable at all.
    A world that reached the cycle without them would answer every collision with the
    divergent rejection, and the first scored check names that."""
    held = collection_entries(db, RECIPE_BOX.name)
    assert set(held) == set(RECIPE_BOX_SEED_KEYS), (
        f"the box must hold {sorted(RECIPE_BOX_SEED_KEYS)} when the cycle starts, got "
        f"{sorted(held)}"
    )
    assert held[RECIPE_BOX_FAJITAS_KEY] == RECIPE_BOX_FAJITAS_SEED_CONTENT, (
        f"the fajitas entry must start at its seeded value, got {held[RECIPE_BOX_FAJITAS_KEY]!r}"
    )
    assert RECIPE_BOX_DUP_CONTENT == RECIPE_BOX_FAJITAS_SEED_CONTENT, (
        "the same-fact case's forced write must repeat the stored value VERBATIM — that "
        "is what makes it trivially the same value rather than a judgment call"
    )


# ── The staged injector ───────────────────────────────────────────────────────


class _InjectDuplicateWriteAfterAStep(_InjectAfterToolCall):
    """``_InjectDuplicateWrite``'s staged twin: the same forced duplicate write, but held
    back until the model's first REAL tool call has landed.

    The difference is not about what the write gate does — it is about whether the run
    leaves a LEDGER to read.  Every injector returns a synthetic ``LlmResponse`` that
    bypasses the persisting client (the #1695 design: the raw response is persisted
    inside the real client before the wrapper can touch it, which is why
    ``bail_injected`` is the only proof the sabotage fired).  When the forced call is the
    cycle's FIRST and the write gate STOPs on it, the cycle ends having persisted NOTHING:
    ``set_run_outcome`` has no row to stamp, the run record reads empty, and every
    cycle-shaped check scores against an absent ledger — which is exactly how a working
    STOP measured as five 'behavioural' failures.

    Staging the injection after one real step fixes it at the source: the program opens
    with ``collection_read_latest``, that call and its result persist, the forced write
    then STOPs on the second turn, and the stop's terminal tool result rides
    ``trailing_messages`` (#1778) on the row the read already wrote.  The MECHANISM under
    test is untouched — the same write, the same gate, the same STOP."""

    def __init__(self, real, memory: str, entries: list[tuple[str, str]]) -> None:
        super().__init__(real)
        self._memory = memory
        self._entries = entries

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


# ── Reading what the gate and the cycle did ───────────────────────────────────


def _tool_results_naming(db: Database, marker: str) -> bool:
    """Did any tool result this run carry ``marker``, read off the persisted promptlog —
    the same text the model read?"""
    return any(
        message.get("role") == "tool" and marker in (message.get("content") or "")
        for message in _iter_prompt_messages(db)
    )


def _bound_matched_keys(db: Database) -> set[str]:
    """Every existing key the divergent-value rejection bound into an ``update_entry``
    call, read off the persisted tool results."""
    bound: set[str] = set()
    for message in _iter_prompt_messages(db):
        if message.get("role") != "tool":
            continue
        bound.update(_BOUND_KEY.findall(message.get("content") or ""))
    return bound


def _run_reason(db: Database) -> str:
    """The reason the cycle's run record was stamped with — the structural line the
    ledger generated, which for a write-gate STOP is the declared stop reason."""
    outcomes = db.messages.recent_run_outcomes(RECIPE_BOX.name, 1)
    return outcomes[0][1] if outcomes else ""


def _cycle_ran_check(db: Database) -> Check:
    """The RAN-GUARD: the cycle made at least one model call this run left behind.

    ``guard_recovery_eval``'s own guard says the INJECTOR fired, which is a fact about the
    wrapper and not about the run; this says the run exists at all.  Without it a cycle
    that persisted nothing scores every later check against an empty ledger and reports
    the absence as model behaviour — five samples read that way before this guard, on a
    mechanism that was working perfectly.  The multi-cycle runner carries the same floor
    (``_cycles_ran_check``); a single-cycle recovery case needs it for the same reason."""
    rows = live_prompts(db)
    return Check(
        "the cycle left a run behind to read",
        bool(rows),
        kind="guard",
        rationale=None
        if rows
        else (
            "no model call persisted, so there is no ledger and no run record — every "
            "check below is reading an absent cycle, not a model that did nothing"
        ),
    )


# ── Scoring ───────────────────────────────────────────────────────────────────


def _score_stopped_on_the_same_fact(db: Database, sent: list[str]) -> list[Check]:
    """The cycle re-observed what the box already held and STOPPED at the chokepoint:
    nothing written, nothing said, and a run record that names why.

    (``guard_recovery_eval`` prepends its own "forced bail fired" guard, which says the
    injector ran; the first check here says the GATE answered it.)"""
    reason = _run_reason(db)
    expected_reason = WRITE_GATE_STOP_REASONS[WriteGateOutcome.DUPLICATE_UNCHANGED]
    answered = _tool_results_naming(db, _ALREADY_RECORDED)
    keys = set(collection_entries(db, RECIPE_BOX.name))
    keys_unchanged = keys == set(RECIPE_BOX_SEED_KEYS)
    queued = queued_sends(db, RECIPE_BOX.name)
    return [
        _cycle_ran_check(db),
        Check(
            "the write gate answered that the value was already recorded",
            answered,
            anchor="collection_write(",
            kind="guard",
            rationale=None
            if answered
            else (
                "the gate did not read the re-observation as no-news — with the recoverable "
                "rejection instead, the cycle is invited to try something else.  The one "
                "non-behavioural way to land here is a stored entry with no CONTENT VECTOR: "
                "the signal that decides same-value cannot be scored, so the collision reads "
                f"divergent.  Calls made: {tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "the cycle stopped at the write chokepoint",
            reason == expected_reason,
            kind="proc",
            rationale=None
            if reason == expected_reason
            else (
                f"the run record reads {reason!r}, not the declared stop reason "
                f"{expected_reason!r} — the cycle carried on past the re-observation"
            ),
        ),
        Check(
            "the box still holds exactly the recipes it started with",
            keys_unchanged,
            anchor="collection_write(",
            kind="state",
            rationale=None
            if keys_unchanged
            else (
                "the box's keys changed on a cycle that found nothing new — the model "
                f"invented a recipe to have something to file: {sorted(keys)} against the "
                f"seeded {sorted(RECIPE_BOX_SEED_KEYS)}"
            ),
        ),
        Check(
            "the user was told nothing",
            not queued,
            kind="state",
            rationale=None
            if not queued
            else (
                f"{len(queued)} message(s) queued on a cycle that observed nothing new — "
                f"the collection notifies, so silence has to come from the STOP: {queued}"
            ),
        ),
    ]


async def test_same_fact_write_stops_the_cycle(guard_recovery_eval) -> None:
    """A forced ``collection_write`` repeating a stored value VERBATIM under a reworded
    key is the same no-news the exact key reads: the gate says so, the cycle stops at the
    chokepoint, nothing is written and the user hears nothing.

    The write is injected AFTER the model's first real step (the program's own
    ``collection_read_latest``), so the STOP lands on a cycle that has already persisted a
    row — see ``_InjectDuplicateWriteAfterAStep``.  Injecting it first left the run with no
    ledger at all, and the scorer read that absence as the model failing."""
    await guard_recovery_eval(
        case_id="duplicate-write-same-fact-stops",
        family="collector-guard-recovery",
        collection=RECIPE_BOX.name,
        seed=_seed_recipe_box,
        wrap_client=lambda real: _InjectDuplicateWriteAfterAStep(
            real, RECIPE_BOX.name, [(RECIPE_BOX_DUP_KEY, RECIPE_BOX_DUP_CONTENT)]
        ),
        score=_score_stopped_on_the_same_fact,
        min_pass_rate=None,
    )


# ── The watch world: a page read, one tracked value recorded ──────────────────
#
# The divergent case's own world, and a different ROUTINE on purpose.  A recipe box's
# job is to file something not already saved, so a cycle told its write collided is
# answered correctly by writing a different recipe — which is what the first measured
# run did, faithfully ("We must submit a new recipe… not already saved").  A WATCH's job
# is to record the current value about ONE subject it already tracks, so landing a
# changed value on the entry that holds it is the routine's own job rather than a
# recovery the rejection has to argue the model into.

# The page the watch is pointed at, and the value it tracks.  Invented — no real
# listing, no real price.
LANTERN_PAGE = "https://faux-market.example/keel-lantern"
LANTERN_CONTAINER = "keel-lantern-watch"
LANTERN_DESCRIPTION = "What the Keel Lantern listing is posted at right now."

# The key the watch already files its subject under, and the value it holds.
LANTERN_KEY = "Keel Lantern"
LANTERN_STORED_VALUE = "Keel Lantern — posted at $128."

# The DIVERGENT write: the same subject, filed under a key worded differently this time,
# carrying the value the page has MOVED to — a value swap rather than an appended
# clause, because an append is the same value said at greater length.
LANTERN_REWORDED_KEY = "keel lantern"
LANTERN_CHANGED_VALUE = "Keel Lantern — posted at $96."
# What any honest message about that change carries.  A small set rather than one
# spelling, because the message is model-authored; the CHECK is that the new value
# reached the user at all.
LANTERN_CHANGE_TOKENS = ("96", "$96")

# The calls the watch makes, in order.
LANTERN_PROGRAM_CALLS = ("browse", "collection_write")

LANTERN_SCHEDULE = "FREQ=HOURLY"

# The page the cycle actually reads, carrying the moved value — so the browse the model
# really makes and the write the injector forces tell the same story.
LANTERN_PAGE_CONTENT = CannedPage(
    match="keel-lantern",
    text=(
        "Title: Keel Lantern — faux market listing\n"
        "A small brass lantern, sold by a single maker.\n"
        f"Source: {LANTERN_PAGE}\n"
        "Posted at $96 — reduced from $128 earlier this season.\n"
    ),
)

# The routine, in the shape run-end extraction leaves behind: the page a PARAMETER site
# (its demonstrated value is what the runtime join fills), the extract instruction and
# the entry key labelled PLACEHOLDERs, the destination attachment-marked, and the
# recorded value BOUND to the browse's own result — which is what makes this a watch
# rather than a generator: what it writes is what the page said.
LANTERN_SKILL = SkillDraft(
    name="track_posted_value",
    intent="Keep an eye on what the Keel Lantern listing is posted at.",
    description="Read a listing and record what it is posted at right now.",
    steps=[
        SkillStep(
            ordinal=1,
            source_ordinal=1,
            tool="browse",
            arguments={"queries": [LANTERN_PAGE], "extract": "what it is posted at"},
            substitutions=[
                SkillSubstitution(
                    path=["queries", 0],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the listing to read",
                ),
                SkillSubstitution(
                    path=["extract"],
                    kind=SkillSubKind.PLACEHOLDER,
                    description="the value to read off the page",
                ),
            ],
        ),
        SkillStep(
            ordinal=2,
            source_ordinal=2,
            tool="collection_write",
            arguments={
                "memory": LANTERN_CONTAINER,
                "entries": [{"key": LANTERN_KEY, "content": LANTERN_STORED_VALUE}],
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
            value=LANTERN_PAGE,
        )
    ],
    source_run_id="eval-seed",
)

LANTERN_VALUES = {"listing": LANTERN_PAGE}


def rendered_watch_program() -> str:
    """The watch's program, through the production instantiation seam's own three steps."""
    attached = retarget_writes(LANTERN_SKILL.steps, LANTERN_CONTAINER)
    joined = bind_parameters(attached, LANTERN_SKILL.parameters, LANTERN_VALUES)
    return render_skill(joined, LANTERN_VALUES)


def _seed_lantern_watch(db: Database) -> None:
    """The world an apply turn leaves: the taught watch in the registry, the container
    already holding the value it recorded last time, and the job configured from that
    routine — NOTIFYING, because being told once is half of what this case scores."""
    db.skills.upsert(LANTERN_SKILL, author=_SEED_AUTHOR)
    db.memories.create_collection(
        LANTERN_CONTAINER,
        LANTERN_DESCRIPTION,
        extraction_prompt=rendered_watch_program(),
        schedule=LANTERN_SCHEDULE,
        notify=True,
        skill_name=slug_skill_name(LANTERN_SKILL.name),
        skill_params=LANTERN_VALUES,
    )
    require_memory(db, LANTERN_CONTAINER).write(
        [EntryInput(key=LANTERN_KEY, content=LANTERN_STORED_VALUE)], author="collector"
    )
    _assert_the_watch_world(db)


def _assert_the_watch_world(db: Database) -> None:
    """Everything the watch seeder is responsible for, asserted out loud — the same
    silent-failure surface the box's probe covers, on this world's own claims."""
    name = slug_skill_name(LANTERN_SKILL.name)
    assert db.skills.get(name) is not None, f"the job's routine {name!r} must be registered"
    row = db.memories.get(LANTERN_CONTAINER)
    assert row is not None, f"the job's container {LANTERN_CONTAINER!r} must exist"
    assert row.skill_name == name, (
        f"the job must run the routine the fixture taught, not {row.skill_name!r}"
    )
    assert row.notify and not row.archived, (
        "the watch must be live and NOTIFYING — this case scores being told exactly once"
    )
    _assert_the_watch_program_parses(db)
    _assert_the_recovery_verb_is_reachable()
    held = collection_entries(db, LANTERN_CONTAINER)
    assert held == {LANTERN_KEY: LANTERN_STORED_VALUE}, (
        f"the watch must hold exactly its last reading when the cycle starts, got {held}"
    )
    assert LANTERN_CHANGED_VALUE != LANTERN_STORED_VALUE, (
        "the forced write must carry a value the watch does NOT already hold, else it is "
        "the same no-news the other case scores"
    )
    assert LANTERN_REWORDED_KEY != LANTERN_KEY, (
        "the forced write must arrive under a DIFFERENT key, or the exact-key path answers "
        "it and the dedup disjunction — which this case is about — never runs"
    )


def _assert_the_watch_program_parses(db: Database) -> None:
    """The stored program reads back as the calls it makes, carries the page the runtime
    join filled, names the container it writes to, and stores no terminal ``done()``.

    The recorded value is a BINDING (``the value from step 1``) rather than a placeholder,
    which is the whole difference between a watch and a generator: what this routine
    writes is what the page said, so a changed reading belongs on the entry that holds the
    last one."""
    row = db.memories.get(LANTERN_CONTAINER)
    program = (row.extraction_prompt or "") if row is not None else ""
    parsed = tuple(call.tool for call in program_calls(program, frozenset(LANTERN_PROGRAM_CALLS)))
    assert parsed == LANTERN_PROGRAM_CALLS, (
        f"the stored program must read back as {list(LANTERN_PROGRAM_CALLS)} under the "
        f"rendered dialect, got {list(parsed)} — program: {program!r}"
    )
    assert LANTERN_PAGE in program, (
        f"the runtime join must fill the browse leaf with {LANTERN_PAGE!r} — program: {program!r}"
    )
    assert f"'{LANTERN_CONTAINER}'" in program, (
        f"the attachment must be bound to {LANTERN_CONTAINER!r} — program: {program!r}"
    )
    assert _BINDING_RENDER in program, (
        "the recorded value must render as the browse's own result — a watch writes what "
        f"the page said, not a fresh phrase.  Program: {program!r}"
    )
    assert Prompt.COLLECTOR_DONE_STEP not in program, (
        "the terminal step is assembly's to inject (#1916) — a STORED program carrying "
        "one is a render a chat ledger cannot produce"
    )


def _score_landed_the_change(db: Database, sent: list[str]) -> list[Check]:
    """The cycle read a value DIFFERENT from the one it holds: the rejection bound the
    matched key, the change landed on that entry, the cycle closed clean, and the user was
    told once — which is the watch's own job, not a recovery it had to be argued into."""
    entries = collection_entries(db, LANTERN_CONTAINER)
    keys = set(entries)
    keys_unchanged = keys == {LANTERN_KEY}
    bound = _bound_matched_keys(db)
    landed = entries.get(LANTERN_KEY, "") != LANTERN_STORED_VALUE
    closed = tool_was_called(db, "done")
    queued = queued_sends(db, LANTERN_CONTAINER)
    named = [item for item in queued if any(token in item for token in LANTERN_CHANGE_TOKENS)]
    return [
        _cycle_ran_check(db),
        Check(
            "the rejection bound the matched key into an update_entry call",
            LANTERN_KEY in bound,
            anchor="collection_write(",
            kind="guard",
            rationale=None
            if LANTERN_KEY in bound
            else (
                f"the rejection bound {sorted(bound) or 'nothing'} — a divergent value must "
                f"name {LANTERN_KEY!r}, or the model has to guess which row it collided "
                "with.  The one non-behavioural way to land here is the strict CONTENT "
                "signal reading the moved value as the SAME value, which would classify "
                "this collision as no-news and stop the cycle instead"
            ),
        ),
        Check(
            "the new reading landed on the entry the watch already had",
            landed,
            anchor=f"{_RECOVERY_VERB}(",
            kind="state",
            rationale=None
            if landed
            else (
                f"{LANTERN_KEY!r} still holds its previous reading — the change the cycle "
                f"found was rejected and then dropped.  Calls made: "
                f"{tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "the watch still tracks exactly the one subject it started with",
            keys_unchanged,
            kind="state",
            rationale=None
            if keys_unchanged
            else (
                "the watch's keys changed — the new reading was filed as a second entry "
                f"instead of landing on the one it is about: {sorted(keys)}"
            ),
        ),
        Check(
            "the cycle closed with done() rather than stopping",
            closed,
            anchor="done(",
            kind="proc",
            rationale=None
            if closed
            else (
                "no done() — a changed reading is news and must never stop the cycle at "
                f"the chokepoint.  Calls made: {tool_call_sequence(db) or 'none'}"
            ),
        ),
        Check(
            "the user was told once, and told the new value",
            len(queued) == 1 and len(named) == 1,
            kind="state",
            rationale=None
            if len(queued) == 1 and len(named) == 1
            else (
                f"{len(queued)} message(s) queued, {len(named)} naming the new value "
                f"{LANTERN_CHANGE_TOKENS}: {queued}"
            ),
        ),
    ]


async def test_divergent_value_write_updates_and_notifies(guard_recovery_eval) -> None:
    """A watch reads its page, finds the tracked value has MOVED, and its write arrives
    under a key worded differently from the one the reading is already filed under: the
    rejection binds that key, the live model lands the new reading on it, the cycle closes,
    and the one notification names what it moved to.

    Staged after the model's real browse, so the cycle has read the page before the forced
    write — the watch doing its own job, with only the collision forced."""
    await guard_recovery_eval(
        case_id="duplicate-write-divergent-value-updates",
        family="collector-guard-recovery",
        collection=LANTERN_CONTAINER,
        seed=_seed_lantern_watch,
        browse=[LANTERN_PAGE_CONTENT],
        wrap_client=lambda real: _InjectDuplicateWriteAfterAStep(
            real, LANTERN_CONTAINER, [(LANTERN_REWORDED_KEY, LANTERN_CHANGED_VALUE)]
        ),
        score=_score_landed_the_change,
        min_pass_rate=None,
    )
