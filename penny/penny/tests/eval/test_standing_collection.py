"""Standing-collection operations: the user changes a live job in plain words, and
asks what it does.

Two stories, one seeded world.  A routine was taught once and stood up as a job — a
container the round derived, the program rendered into it, an RRULE schedule, notify on,
the routine and its bound values stamped as provenance — and then the user comes back
later and speaks about it the way people do:

  * **operate it** — turn its notifications off, turn them back on, retire it: the things
    done to a job that is already running, and ones the canonical suites never touch
    (``test_state_transitions.py`` measures the edges that BUILD a job,
    ``test_collector_enactment.py`` measures the cycles it then runs);
  * **read it** — "what does that thing actually do?", answered from the routine rather
    than from what the ambient header happens to say about it (#1804 took the recipe off
    that header, so the recipe is a read now, not a recall).

Broadening a job's SCOPE is deliberately not here: a routine's steps are a render of what
was demonstrated, so widening what it collects is a re-teach of the routine rather than an
edit to the job (code-owner ruling), and the state machine's learn arc is where that
belongs.

**The world is built the way production builds one (#1911/migration 0108: nothing is
pre-seeded).**  Every collection here is one the user built: the container is created
storage-only, then configured through the store's own metadata write with the program
rendered through the production instantiation seam's three steps — the attachment bound
to the container, the runtime join (#1907) writing each bound value into the leaf the
demonstration put it in, then the render.  The routine itself goes into the registry
through ``chat_eval(seed_skills=…)``, embedded like a real one.  A hand-authored prose
prompt would be a config defect the collector cannot read (#1916's strict dialect), so a
world seeded that way would be claiming a job that could never run.

**Report-only** (``min_pass_rate=None``): these cases are being re-baselined under the
conversation machine, and the thresholds are the code owner's to set from a read of the
numbers rather than inherited from the pre-machine gates they carried.

Scoring is structural throughout — the persisted collection row (its terms, its schedule,
its notify flag, its archived flag, what it holds) and the calls the run really made.
Each sample also renders, ADVISORY, the state the conversation machine landed the turn
in: every turn here runs the production path, so the machine classifies first, and a
reader of a surprising sample wants to see where it went before anything else.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import pytest

from penny.agents.self_state import SelfStateHeader
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.database.models import MemoryRow
from penny.database.skills import (
    DistillInput,
    SkillDraft,
    SkillStep,
    bind_parameters,
    derive_collection_name,
    distill_steps,
    render_skill,
    retarget_writes,
    slug_skill_name,
)
from penny.penny import Penny
from penny.program import program_calls
from penny.skill_extraction import _apply_leaf_labels, _interface_parameters
from penny.tests.conftest import TEST_SENDER, require_memory
from penny.tests.eval.conftest import (
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Preparer,
    Seeder,
    collection_entries,
    new_collections,
    seeded_run_id,
    tool_was_called,
)
from penny.tools.micro_context import FramedParameter, LeafLabel, SkillLabels, SkillSignature

pytestmark = pytest.mark.eval

_OPERATIONS_FAMILY = "standing-collection-operations"
_LEGIBILITY_FAMILY = "standing-collection-legibility"

# The one call that reconfigures a standing job — named once, since a check reads it
# and an anchor renders it.
_COLLECTION_SET = "collection_set"

# The run that stood the jobs up — a seeded prior turn, so every reader of "what did the
# model do this sample" excludes it (a job's own creation is history, not this turn's work).
_STOOD_UP_RUN = seeded_run_id("stood-up-the-job")

# The two calls the taught routine makes, in order.  Named once: the seeder holds the
# rendered program against them in the STRICT dialect (#1916), so a fixture that stops
# rendering a runnable program fails at the seed rather than as a puzzling miss.
_BROWSE = "browse"
_WRITE = "collection_write"
_PROGRAM_TOOLS = (_BROWSE, _WRITE)

# The reads that answer "what does this thing do".  ``memory_metadata`` renders the
# collection's own rendered program; ``skill_read`` returns the routine's steps.  Both are
# a READ of the recipe — which is the contract — so the spine check accepts either rather
# than keying on the verb that happened to be used when the case was written.
_METADATA = "memory_metadata"
_SKILL_READ = "skill_read"


# ── The taught routine ────────────────────────────────────────────────────────


class _Demonstration(NamedTuple):
    """The round that taught the routine, as its ledger recorded it: the page it read,
    what it was told to pull off it, the container it wrote into, and the entry that
    landed.  Two steps — read the page, write what it said — the canonical taught round."""

    url: str
    extract: str
    collection: str
    entry_key: str
    entry_value: str


def _demonstrated_ledger(demonstration: _Demonstration) -> list[DistillInput]:
    """The two certified steps distillation reads, with their results carrying the real
    ``(<tool> result)`` frame — so the frame is stripped the way production strips it and
    the write's content binds to the browse's PAYLOAD rather than to its narration."""
    return [
        DistillInput(
            source_ordinal=1,
            tool=_BROWSE,
            arguments={"queries": [demonstration.url], "extract": demonstration.extract},
            result=(f"You opened {demonstration.url} (browse result)\n{demonstration.entry_value}"),
        ),
        DistillInput(
            source_ordinal=2,
            tool=_WRITE,
            arguments={
                "memory": demonstration.collection,
                "entries": [{"key": demonstration.entry_key, "content": demonstration.entry_value}],
            },
            result=(
                f"You saved an entry to {demonstration.collection}: "
                f"({_WRITE} result)\nWrote 1 entry."
            ),
        ),
    ]


def _leaf_at(arguments: dict[str, Any], path: list[str | int]) -> Any:
    """The demonstrated value a substitution addresses — indexed directly, since a path a
    step's own substitution carries either resolves or the step is corrupt."""
    node: Any = arguments
    for part in path:
        node = node[part]
    return node


def _labels_by_value(steps: list[SkillStep], authored: dict[str, LeafLabel]) -> SkillLabels:
    """The labeller's draw, authored against the demonstrated VALUES and mapped onto the
    arg-derived spot names the distiller happened to mint.

    Keyed by value because that is what the fixture MEANS — a naming change inside
    distillation would otherwise leave this silently unmapped.  Every authored label must
    map home: an accepted draw covers every spot (#1828), so a partly-labelled routine is
    a shape run-end extraction cannot produce, and it fails here rather than seeding a
    world nothing makes."""
    labels: dict[str, LeafLabel] = {}
    for step in steps:
        for substitution in step.substitutions:
            if substitution.parameter is None:
                continue
            value = str(_leaf_at(step.arguments, substitution.path))
            if value in authored:
                labels[substitution.parameter] = authored[value]
    assert len(labels) == len(authored), (
        f"every authored label must map home — matched {sorted(labels)} of {sorted(authored)}"
    )
    return SkillLabels(labels=labels)


def _taught_routine(
    demonstration: _Demonstration, signature: SkillSignature, authored: dict[str, LeafLabel]
) -> SkillDraft:
    """The skill a demonstrated round leaves in the registry, built by the production
    pipeline over that round's ledger: ``distill_steps`` for the structure, the labeller's
    draw applied by ``_apply_leaf_labels``, the framer's signature applied by
    ``_interface_parameters``.

    Only the two DRAWS are written by hand, which is what a fixture is for — so the
    starting world is the shape extraction really produces (an all-placeholder recipe over
    the SKILL-level parameters the framer minted, each carrying the value its round
    demonstrated it with) rather than a convenient copy of it."""
    steps, parameters = distill_steps(
        _demonstrated_ledger(demonstration), frozenset({demonstration.collection})
    )
    steps, distilled = _apply_leaf_labels(steps, parameters, _labels_by_value(steps, authored))
    return SkillDraft(
        name=signature.name,
        intent=signature.description,
        description=signature.description,
        steps=steps,
        parameters=_interface_parameters(signature, distilled),
        source_run_id=seeded_run_id("taught-the-routine"),
    )


# The one routine both stories stand on, and the round that taught it.  Synthetic
# throughout (the repo is public): an invented shop on an example domain, an invented
# thing to watch for.
_DEMONSTRATED_PAGE = "https://quillmarket.example.com/typewriters"
_DEMONSTRATED_WATCH = "portable typewriters"
_DEMONSTRATED_ENTRY = "Hermes Baby — 1958 portable, case included, $180."

WATCH_ROUTINE = _taught_routine(
    _Demonstration(
        url=_DEMONSTRATED_PAGE,
        extract=_DEMONSTRATED_WATCH,
        collection="typewriter-finds",
        entry_key="Hermes Baby",
        entry_value=_DEMONSTRATED_ENTRY,
    ),
    SkillSignature(
        name="track_new_listings",
        description="Check a page for newly listed items of a given kind and keep what it finds.",
        parameters=(
            FramedParameter(
                name="page",
                description="the address of the page to check each run",
                value=_DEMONSTRATED_PAGE,
            ),
            FramedParameter(
                name="watched_for",
                description="the kind of thing to look for on that page",
                value=_DEMONSTRATED_WATCH,
            ),
        ),
    ),
    {
        _DEMONSTRATED_PAGE: LeafLabel(
            name="page", description="the address of the page to check each run"
        ),
        _DEMONSTRATED_WATCH: LeafLabel(
            name="watched_for", description="the kind of thing to look for on that page"
        ),
        # The destination is a spot like any other, and additionally carries the attachment
        # mark — applying the routine somewhere is what fills it (#1783/#1828).
        "typewriter-finds": LeafLabel(
            name="destination", description="the collection this is set up on"
        ),
        # The entry's key and content are deliberately NOT here: both came out of the
        # browse, so distillation BINDS them to that step (#1659's structural provenance)
        # and no spot is left for a label to name.  Listing one would be a fixture claiming
        # a draw the labeller was never offered — which is what the map-home assertion in
        # `_labels_by_value` catches.
    },
)


# ── A standing job: the routine, applied ──────────────────────────────────────


class StandingJob(NamedTuple):
    """One collection as an apply turn leaves it: the routine it runs, the values it is
    pointed at, and the TERMS the turn set (#1869).

    ``holdings`` is what the job has already gathered — a standing job that has never run
    is not the world these stories are about, and it is what the archive case reads to say
    a retired job KEPT what it collected."""

    routine: SkillDraft
    values: dict[str, str]
    description: str
    schedule: str
    notify: bool
    holdings: tuple[tuple[str, str], ...] = ()

    @property
    def ordered_values(self) -> list[str]:
        """The bound values in the routine's DECLARED order — what the container's name is
        derived from, and not an order a fixture is free to choose."""
        return [self.values[parameter.name] for parameter in self.routine.parameters]

    @property
    def container(self) -> str:
        """The collection the job runs in, through the SHIPPED derivation — spelling the
        name out would be a second copy of the scheme jobs are identified by."""
        return derive_collection_name(slug_skill_name(self.routine.name), self.ordered_values)

    @property
    def program(self) -> str:
        """The program the apply turn stores, through the production instantiation seam's
        own three steps in its own order: the attachment bound to the container, the
        runtime join (#1907) writing each bound value into the leaf the demonstration put
        it in, then the render.

        Composed here rather than called, because the shipped seam takes the registry ROW
        and the runner lays the registry down after the seed runs — so what a fixture must
        not do is invent a fourth step or reorder these three."""
        attached = retarget_writes(self.routine.steps, self.container)
        joined = bind_parameters(attached, self.routine.parameters, self.values)
        return render_skill(joined, self.values)


def seed_standing_jobs(*jobs: StandingJob) -> Seeder:
    """Stand each job up the way the production path does: the container CREATED storage
    only (the round's find-or-create), then configured — the rendered program, the RRULE
    schedule, notify, and the routine plus its bound values stamped as provenance — then
    whatever it has already gathered.

    Both writes go through the real store methods, so the mutation ledger records them
    citing the seeded run: a later "who changed this, and when" read finds a job with a
    history rather than a row that appeared from nowhere."""

    def seed(db: Database) -> None:
        for job in jobs:
            _stand_up(db, job)

    return seed


def _stand_up(db: Database, job: StandingJob) -> None:
    db.memories.create_collection(job.container, job.description)
    db.memories.update_collection_metadata(
        job.container,
        extraction_prompt=job.program,
        schedule=job.schedule,
        replace_schedule=True,
        notify=job.notify,
        skill_name=slug_skill_name(job.routine.name),
        skill_params=job.values,
        run_id=_STOOD_UP_RUN,
    )
    _assert_the_job_can_run(job)
    if job.holdings:
        require_memory(db, job.container).write(
            [EntryInput(key=key, content=content) for key, content in job.holdings],
            author="collector",
            run_id=_STOOD_UP_RUN,
        )


def _assert_the_job_can_run(job: StandingJob) -> None:
    """The stored program really is a program, and it is pointed at something.

    Two ways a seeded job can be a job in name only, and both are silent: its steps stop
    parsing in the STRICT rendered dialect (#1916), which leaves a cycle with a surface of
    the terminator alone and a run record naming a config defect; or the RUNTIME JOIN
    (#1907) stops filling its leaves, which leaves a program still describing what belongs
    in each spot instead of naming the page it fetches.  Either way every case standing on
    this world would be measuring nothing, so both fail here, out loud."""
    program = job.program
    parsed = tuple(call.tool for call in program_calls(program, frozenset(_PROGRAM_TOOLS)))
    assert parsed == _PROGRAM_TOOLS, (
        f"{job.container}: the stored program must parse as {_PROGRAM_TOOLS}, got {parsed}\n"
        f"{program}"
    )
    unjoined = [value for value in job.values.values() if value not in program]
    assert not unjoined, (
        f"{job.container}: the runtime join must write every bound value into the program — "
        f"{unjoined} missing from\n{program}"
    )


# The standing job both stories operate on: the routine above, pointed at the page it was
# taught on, running each morning and telling the user when it finds something.
_FINDS = StandingJob(
    routine=WATCH_ROUTINE,
    values={"page": _DEMONSTRATED_PAGE, "watched_for": _DEMONSTRATED_WATCH},
    description="Portable typewriters newly listed at the Quill Market shop.",
    schedule="FREQ=DAILY;BYHOUR=7",
    notify=True,
    holdings=(
        ("Hermes Baby", _DEMONSTRATED_ENTRY),
        ("Olivetti Lettera 32", "Olivetti Lettera 32 — 1963 portable, case included, $220."),
    ),
)

# The same job, silent — the world the wake-it case starts from.
_SILENT_FINDS = _FINDS._replace(notify=False)

# Held at IMPORT, not only at seed time: `make check` collects this module without running
# a case, so a fixture that stopped rendering a runnable program is caught by the plain
# suite instead of surviving until someone spends GPU on it.
_assert_the_job_can_run(_FINDS)

# How the user refers to it: their own words for the job, never its derived name.  The
# derived name renders on the ambient mechanisms line, so resolving those words to that
# collection is a read the turn is expected to make — asking with the container's own
# name would hand the model the answer to half the case.
#
# It also lends NO word to any scorer below: the legibility patterns credit a reply for
# describing the routine's two moves, so an ask carrying one of those words would let a
# reply that merely echoes the question score as a faithful description.
_THEIR_WORDS = "the typewriter watch"


# ── Reading a job's row ───────────────────────────────────────────────────────


def switch_is_visible(job: StandingJob) -> Preparer:
    """The job's own self-state row states its notify parameter — asserted BEFORE the turn.

    A flip case measures whether she reaches for the per-collection switch, and the model
    can only reach for a lever the state presents: with nothing on the row, a measured
    sample asked to stop one watch's pings enumerated what it could see — archive the
    collection, or mute notifications globally — read that as no granularity, and retired
    the whole job.  So the case's world has to carry the switch, and a world that stopped
    carrying it fails here, naming the row, rather than after the GPU is spent on it.

    Read through the header's own constants, so the probe and the render cannot drift into
    asserting different words for one state."""

    def probe(penny: Penny) -> None:
        chip = SelfStateHeader.MECHANISM_NOTIFIES if job.notify else SelfStateHeader.MECHANISM_QUIET
        rendered = SelfStateHeader(penny.db, TEST_SENDER).render()
        row = next(
            (line for line in rendered.splitlines() if line.startswith(f"- {job.container} ")),
            None,
        )
        assert row is not None, f"{job.container} must render as a mechanism:\n{rendered}"
        assert chip in row, f"{job.container} must render {chip!r} — it renders {row!r}"

    return probe


def _job(db: Database) -> MemoryRow | None:
    return db.memories.get(_FINDS.container)


def _job_still_there_check(row: MemoryRow | None) -> Check:
    """Every claim below is about THIS job, so a missing row is the one failure that has
    to be reported before any of them."""
    return Check(
        "state: the standing job is still in the registry",
        row is not None,
        rationale=None if row is not None else f"{_FINDS.container!r} is gone",
        kind="state",
    )


def _still_live_checks(row: MemoryRow) -> list[Check]:
    """The job is still RUNNING: not retired, still scheduled, still carrying a program,
    still pointed at the same routine.

    Read after every operation that changes ONE thing, because "she changed the thing
    asked for" and "she left the rest alone" are two claims — a turn that quietly retired
    the job, dropped its schedule or emptied its program while flipping a flag has done
    something else entirely, and the flag check alone would call that a pass.  All three
    are read explicitly rather than through a default: a job with NO program is a config
    defect the collector reports, not an absence to be smoothed over."""
    live = not row.archived and row.schedule is not None and row.extraction_prompt is not None
    same_routine = row.skill_name == slug_skill_name(_FINDS.routine.name)
    return [
        Check(
            "state: the job is still live (active, scheduled, with a program to run)",
            live,
            rationale=None
            if live
            else (
                f"archived={row.archived}, schedule={row.schedule!r}, "
                f"program={row.extraction_prompt!r}"
            ),
            kind="state",
        ),
        Check(
            "state: it still runs the routine it was set up with",
            same_routine,
            rationale=None if same_routine else f"skill_name is now {row.skill_name!r}",
            kind="state",
        ),
    ]


def landed_state_check(db: Database) -> Check:
    """Advisory: where the conversation machine landed the turn.

    Every case in these stories runs the production path, so the machine classifies the
    message before the chat agent sees it, and which state it landed in decides what the
    turn was even instructed to do — a reader of a surprising sample wants that first.
    Rendered rather than demanded: these stories are about what the turn LEAVES BEHIND,
    and pinning the edge would be measuring the machine's own beat here.

    PUBLIC because the log-reads story renders the same row for the same reason; a second
    copy would be a second reading of one fact, free to drift a word at a time."""
    latest = db.machine.latest_transition()
    landed = latest.to_state if latest is not None else ConversationState.IDLE.value
    return Check(f"the machine landed the turn in {landed}", True, scored=False, kind="state")


# ── Story: operating a job that is already running ──────────────────────────


def _score_notify_flip(db: Database, before: set[str], *, expected: bool) -> list[Check]:
    """One flip of ONE job's notify switch, in either direction.

    Four claims, and the last three are why this is not just a flag read: she used the
    per-collection switch (rather than the global mute, which silences everything), the
    flag now says what was asked, the JOB is still running (the measured failure was
    archiving the whole thing to stop its pings), and nothing else in the registry moved."""
    row = _job(db)
    if row is None:
        return [_job_still_there_check(row), landed_state_check(db)]
    flipped = row.notify == expected
    return [
        Check(
            "spine: she reconfigured the collection",
            tool_was_called(db, _COLLECTION_SET),
            anchor=f"{_COLLECTION_SET}(",
            rationale=None
            if tool_was_called(db, _COLLECTION_SET)
            else "collection_set never called",
            kind="spine",
        ),
        Check(
            f"state: the job's notify is {_flag(expected)}",
            flipped,
            rationale=None if flipped else f"notify is still {_flag(row.notify)}",
            kind="state",
        ),
        *_still_live_checks(row),
        *_nothing_else_touched_checks(db, before),
        landed_state_check(db),
    ]


def _flag(on: bool) -> str:
    """A notify state in the words the row itself renders — so a check label and the state
    the model read say the same thing."""
    return SelfStateHeader.MECHANISM_NOTIFIES if on else SelfStateHeader.MECHANISM_QUIET


def _nothing_else_touched_checks(db: Database, before: set[str]) -> list[Check]:
    """The turn changed the one thing it was asked to and nothing else: no collection was
    spawned beside the job, and what the job already gathered is still there."""
    spawned = new_collections(db, before)
    return [
        Check(
            "state: nothing else was created",
            not spawned,
            rationale=None if not spawned else f"also created {[row.name for row in spawned]}",
            kind="state",
        ),
        _holdings_kept_check(db),
    ]


def _holdings_kept_check(db: Database) -> Check:
    """What the job already gathered is still readable.

    Every operation here changes ONE thing about a job, and none of them is a licence to
    clear it out — silencing a watch that quietly emptied its collection, or retiring one
    by deleting it, has taken something the user can still read."""
    held = collection_entries(db, _FINDS.container)
    expected = {key for key, _ in _FINDS.holdings}
    kept = expected <= set(held)
    return Check(
        "state: what it gathered is still there",
        kept,
        rationale=None if kept else f"holds {sorted(held)}, expected to keep {sorted(expected)}",
        kind="state",
    )


def _score_archive(db: Database, _before: set[str], _reply: str) -> list[Check]:
    """Retiring a job archives it — a visible tombstone that keeps what it gathered.

    Archived rather than deleted is the contract everywhere in the registry: a retired
    mechanism stays enumerable (the catalog is archived-inclusive) and the same job asked
    for again unarchives it, so a turn that DELETED the collection would have taken the
    user's entries with it."""
    row = _job(db)
    retired = row is not None and row.archived
    return [
        _job_still_there_check(row),
        Check(
            "state: the job is retired (archived)",
            retired,
            rationale=None if retired else "the collection is still active",
            kind="state",
        ),
        _holdings_kept_check(db),
        landed_state_check(db),
    ]


async def test_turning_notifications_off_silences_the_job_without_retiring_it(
    chat_eval: ChatEval,
) -> None:
    """Report-only.  Asked in as many words to turn one job's notifications off: the
    per-collection switch flips, and the job itself keeps running.

    The ask is explicit on purpose — the vocabulary the mute contracts settled on, where
    what is wanted is said rather than implied — because what this measures is whether the
    per-collection switch is REACHABLE, not whether an oblique phrasing can be decoded."""
    await chat_eval(
        case_id="standing-notify-off",
        message=f"turn off notifications for {_THEIR_WORDS}",
        seed=seed_standing_jobs(_FINDS),
        seed_skills=[WATCH_ROUTINE],
        prepare=switch_is_visible(_FINDS),
        score=lambda db, before, _reply: _score_notify_flip(db, before, expected=False),
        min_pass_rate=None,
        family=_OPERATIONS_FAMILY,
    )


async def test_turning_notifications_on_wakes_a_silent_job(chat_eval: ChatEval) -> None:
    """Report-only.  The same switch, the other way, on a job that is already silent —
    the direction that needs a visible ``notify: off`` to have something to flip."""
    await chat_eval(
        case_id="standing-notify-on",
        message=f"turn on notifications for {_THEIR_WORDS}",
        seed=seed_standing_jobs(_SILENT_FINDS),
        seed_skills=[WATCH_ROUTINE],
        prepare=switch_is_visible(_SILENT_FINDS),
        score=lambda db, before, _reply: _score_notify_flip(db, before, expected=True),
        min_pass_rate=None,
        family=_OPERATIONS_FAMILY,
    )


async def test_retiring_it_archives_the_job_and_keeps_what_it_gathered(
    chat_eval: ChatEval,
) -> None:
    """Report-only.  The user is done with the job: it is retired as a tombstone, with
    everything it collected still readable."""
    await chat_eval(
        case_id="standing-archive",
        message=f"i'm done with {_THEIR_WORDS} — you can retire that one",
        seed=seed_standing_jobs(_FINDS),
        seed_skills=[WATCH_ROUTINE],
        score=_score_archive,
        min_pass_rate=None,
        family=_OPERATIONS_FAMILY,
    )


# ── Story: what does that thing actually do? ──────────────────────────────────
#
# The reply has to describe the routine's two moves — it reads the page, it saves what it
# finds — and the families are described many ways, so each is matched against a broad
# verb set with a noun fallback.  Both patterns were widened against captured samples
# (the earlier verb-only search pattern false-negatived "look on the web" and "pulls in …
# databases", scoring faithful descriptions as misses): a scorer that reads only one
# phrasing measures wording rather than fidelity.
#
# The noun fallback names only places the ASK does not: a noun the question already handed
# the model would credit a reply for repeating it (the ask is checked against these
# patterns by construction — see `_THEIR_WORDS`).

_FETCH = (
    r"\b(search\w*|browse\w*|scours?|scans?|hunts?|crawls?|monitors?|gathers?|pulls?\s+in|"
    r"look\w*\s+(for|up|on|at|across|through)|finds?\s+new|fetch\w*|opens?|reads?|checks?)\b"
    r"|\b(the\s+web|online|the\s+internet|the\s+page)\b"
)
# Direct persist verbs match bare; the ambiguous ones (add/store/keep/log/maintain) must be
# ANCHORED to an entry/list object, so "keep an eye on" (about the typewriters, not the
# write) never reads as the save step.
_SAVE = (
    r"\b(saves?|saving|writes?|writing|records?|recording)\b|collection_write"
    r"|\b(adds?|adding|stores?|storing|keeps?|keeping|logs?|logging|maintains?|"
    r"curates?|compiles?|compiling)\b"
    r"[\w\s,'-]{0,20}\b(entry|entries|list|record|records|collection|them|it)\b"
    r"|\bentr(y|ies)\b[^.]{0,30}\b(added|stored|written|saved|created)\b"
)


def _describes(reply: str, pattern: str) -> bool:
    """Whether the reply describes a family, read through the typography the model
    sprinkles (curly quotes, markdown emphasis) — a false negative from a bold marker
    would be the scorer measuring formatting."""
    normalized = reply.casefold().replace("’", "'").replace("“", '"')
    return re.search(pattern, re.sub(r"[*_`]", "", normalized)) is not None


# The ask, and the rule that it lends the scorer nothing — ENFORCED rather than trusted,
# because the leak is invisible once it exists: a question carrying one of the words above
# would credit a reply for repeating the question back.
_LEGIBILITY_ASK = f"what does {_THEIR_WORDS} actually do? walk me through it."
assert not _describes(_LEGIBILITY_ASK, _FETCH) and not _describes(_LEGIBILITY_ASK, _SAVE), (
    f"the ask must lend no word to the patterns that score the reply: {_LEGIBILITY_ASK!r}"
)


def _describes_checks(reply: str) -> list[Check]:
    """The routine's two moves, as the reply describes them — one check each, so a reply
    that got half of it right reads as half right rather than as a failure."""
    return [
        Check(
            f"reply: it describes {claim}",
            _describes(reply, pattern),
            kind="reply",
            anchor=REPLY_ANCHOR,
            rationale=None if _describes(reply, pattern) else f"no {family} family in the reply",
        )
        for claim, family, pattern in (
            ("the page being read", "fetch/read", _FETCH),
            ("what it finds being saved", "save/write", _SAVE),
        )
    ]


def _score_legibility(db: Database, _before: set[str], reply: str) -> list[Check]:
    """She READ the routine and described what it does.

    The read is the spine: the ambient header carries each routine as one row — its name,
    what it is for, what it needs — and NOT its steps (#1804), so a reply describing the
    steps without a read is describing something it never saw.  Either read answers the
    question (the collection's rendered program, or the routine's own steps), so the check
    is about having read rather than about which verb did it."""
    read = tool_was_called(db, _METADATA) or tool_was_called(db, _SKILL_READ)
    return [
        Check(
            "spine: she read the routine rather than recalling it",
            read,
            kind="spine",
            anchor=f"{_METADATA}(",
            rationale=None if read else f"neither {_METADATA} nor {_SKILL_READ} was called",
        ),
        *_describes_checks(reply),
        landed_state_check(db),
    ]


async def test_she_describes_the_routine_the_job_runs(chat_eval: ChatEval) -> None:
    """Report-only.  "What does that actually do?" — answered from the routine, in plain
    words, without inventing a step it does not have."""
    await chat_eval(
        case_id="standing-describe-routine",
        message=_LEGIBILITY_ASK,
        seed=seed_standing_jobs(_FINDS),
        seed_skills=[WATCH_ROUTINE],
        score=_score_legibility,
        min_pass_rate=None,
        family=_LEGIBILITY_FAMILY,
    )
