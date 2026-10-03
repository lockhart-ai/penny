"""The cohort: one request, K phrasings, pooled — and the line between what a case ASSERTS
and what it MEASURES (#1994/#1995).

**Asserted** is the state the round LEFT BEHIND: where the machine landed, what the store
holds, and that every specific value in the reply traces to something the model was given.
Deterministic reads with a pass-rate floor.

**Measured** is everything the model CHOSE: which tools it called and in what order, the shape
of the routine it recorded, the names it picked, the words it replied with.  Many routes reach
one end state, so a route is never asserted — it is scored as spread across the cohort under a
one-sided ceiling, where only an INCREASE is a regression.

**The cohort is the unit.**  N samples of ONE request expressed as K paraphrases, generated
concurrently and weighed against each other — never against a stored baseline, so there is
nothing to drift and nothing to re-baseline on a model swap.

Two properties of the statistic decide how it may be read, both measured rather than assumed:

* **Normalised entropy is BIASED UPWARD at small N** — the same behaviour reads 0.527 at N=32
  and 0.605 at N=15, because the ``log(N)`` denominator shrinks faster than the observed spread
  does.  A recorded ceiling therefore carries its N, and comparing across sizes is refused.
* **Phrasing contributes almost nothing to the spread** (~0.05 of it; model stochasticity
  carries the rest).  That is what justifies pooling.  But phrasings are a COVERAGE mechanism,
  and the pooled number hides what they are for: measured, four phrasings scored
  H = 0.00, 0.52, 0.00, 0.00 — three stable, one that came apart — which pools to 0.18.  So
  every feature also carries a :class:`PhrasingRow`, reporting the weaker honest signal at
  n=3: a wording that produced a value **no other wording did**.

A cohort's samples are HERMETIC — own database, own conversation, own pages — and every one of
them was driven against the same world, so the spread is measured within the pool.
"""

from __future__ import annotations

import calendar
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import CoreSchema, core_schema
from similarity.embeddings import cosine_similarity, token_containment_ratio

from penny.tests.eval.utils.worlds import World

# How far above the observed spread a proposed ceiling sits.  Measured: subsampling real
# 32-sample cohorts down to 15 puts the sampling noise on normalised entropy at ~±0.11, so a
# ceiling ON the observed value would flap on ordinary re-runs while one this far above it
# still separates a variant cohort from a consistent one (they overlap 0.2% at N=15).
CEILING_MARGIN = 0.10

# Unlike the variance margin this one is NOT measured — a round band, stated as such, because
# a mean of large counts is far steadier than a distribution statistic.
COST_CEILING_MARGIN = 0.10

NO_SPREAD = 0.0


# ── What a round was given ───────────────────────────────────────────────────
class Given(str):
    """Everything a round was GIVEN, as one text — which also knows which part of itself was
    STATED to the round and which part the framework wrapped that in (#2203).

    The text is every turn the round was handed, in order, exactly as before: a name or an
    address is sourced by any of it.  ``content`` is the part somebody or something stated —
    the user's turns, a tool's own payload, the entries a store read returned — and a FIGURE
    is sourced by that alone.  The rest is scaffolding: prompt instructions, list numbers,
    counts, the line a result is narrated with.  MEASURED, before the two were told apart: a
    stored recipe entry with an invented `2 tbsp` and `1 tsp` read as sourced, because a read
    tool's own framing said `2 entries` and `1.` and the prompts number their lists.

    ``moments`` are the timestamps the framework rendered for the round — the date and time it
    was told, and every stamp on an entry, a run or a change.  A reply may say WHEN something
    happened, so a date or a clock time is sourced by them; the digits of a stamp are not a
    quantity, and source nothing else.

    A ``str`` so that every reader of the whole text keeps reading it, and the split travels
    with the value it describes instead of beside it, where a caller could leave it behind.  A
    plain string read as one is all content and has no moments: text somebody assembled by
    hand has no frame."""

    content: str
    moments: tuple[datetime, ...]

    def __new__(
        cls, text: str = "", *, content: str | None = None, moments: Iterable[datetime] = ()
    ) -> Given:
        given = super().__new__(cls, text)
        given.content = text if content is None else content
        given.moments = tuple(moments)
        return given

    @classmethod
    def read(cls, value: object) -> Given:
        """``value`` as a ``Given`` — itself, or a plain string as all content."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        raise TypeError(f"what a round was given is text, not {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> CoreSchema:
        """Validated by ``read`` and kept as it is: the stock string schema hands back a plain
        ``str``, which drops the split."""
        return core_schema.no_info_plain_validator_function(cls.read)


# ── What one sample left behind ──────────────────────────────────────────────
class RoutineRecord(BaseModel):
    """One routine the round minted, as the registry holds it."""

    name: str
    shape: str
    names_a_destination: bool
    # Spots the labelling draw left as leaf parameters.  A named spot stops being a parameter,
    # and the labeller names every spot unconditionally — so a leftover one means the draw FELL
    # BACK and the routine kept its arg-derived names.
    open_parameters: list[str] = Field(default_factory=list)
    # Every string at the leaves of the demonstrated call arguments the routine records — what
    # it will actually fetch and look for each run, as against ``shape``, which is one altitude
    # up and carries no argument values at all.  A FLAT list rather than one keyed by tool or
    # by argument position: a skill is an arbitrary tool sequence, so which call carries the
    # page and which carries what to look for is not something a reader can know, and a reading
    # keyed to either stops firing for a shape nobody enumerated.
    demonstrated_values: list[str] = Field(default_factory=list)


class StoredEntry(BaseModel):
    """One entry the round wrote, WHOLE — key and content.

    Both halves, always: a prototype assertion that read content alone reported a 25/32 model
    failure that was entirely its own bug, because samples put the fact in the KEY and the
    blurb in the body, which is a perfectly good way to store it."""

    collection: str
    key: str | None
    content: str

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.key, self.content) if part)


class StoredImage(BaseModel):
    """One image the media store holds — what the row SAYS it is, and where it came from.

    The image bytes are not carried: what a claim can read about a picture is the row's own
    description, which is the text it was generated from (or a browsed page's title), and
    whether a page supplied it.  ``source_url`` is the store's own mark on that second
    question — a browsed capture carries the page it came from, an image made on request
    carries none — so ``drawn`` is a read of the row rather than of which tool wrote it."""

    description: str
    source_url: str | None = None

    @property
    def drawn(self) -> bool:
        """Whether the image was made rather than captured off a page."""
        return self.source_url is None


class MechanismRecord(BaseModel):
    """One MECHANISM — a collection row — as the sample left it.

    ``StoredEntry``'s sibling one level up: that reads what a container HOLDS, this reads the
    container itself, which is the only place a round's own cleanup is legible.  A bail
    archives the container its round built, and an archived row still holds every entry it
    held, so a claim read off the entries cannot tell a retired job from a live one.

    ``born_this_run`` is a REGISTRY read — the row's name was absent when the sample started.

    ``touched_this_run`` is the mutation ledger's answer about the same row: WHAT this sample's
    runs did to it, named the way the store names it — the fields an update reported changed,
    or the ACTION where the store names one instead (archiving is an action rather than a field
    edit, and carries no changed field of its own).  It is the ledger rather than a
    field-by-field diff for the reason the ledger exists: a comparison keyed to a list of
    fields silently exempts whichever field nobody enumerated, and the row's settable surface is
    wider than any list a case would think to write — a description, an expiry, a run quota, a
    rebind's bound values.  A field the store learns to change tomorrow joins this for free.

    A turn that RESTATES a value it was not asked to touch is reported here as having changed
    it, because the store records what an update STATED rather than what differed.  That makes
    this a reading of the CALLS — which fields a call named — and a claim about what the row
    holds reads ``moved_this_run`` instead.  ``changed_this_run`` is a PROPERTY over this rather
    than a field beside it: two facts that always agree are one fact stored twice.

    ``moved_this_run`` is the END STATE of the same row: every field whose value now differs
    from what it held before this sample's first edit to it, read by comparing the ledger's own
    before-value (the prior each event records, #1946) against the row now, through the one
    reader both sides share.  A restated value, and an edit the turn made and then undid, leave
    it empty; an archive names ``archived``, since the flag records its prior the same way.  It
    is what "only the field the ask named moved" is answered from, because what survives a turn
    is the row, and which fields its calls happened to repeat is a route.

    ``notifies`` / ``schedule`` / ``program`` are the row's own configuration as the sample left
    it, read because claims name those VALUES — which way the switch is set, which hour the rule
    fires at, whether the job still has a program to run — questions the ledger cannot answer,
    since it says what moved and never where it landed.

    ``expires`` is whether the row carries an end condition at all — the third TERM a turn that
    stands a job up commits to, beside the schedule it fires on and whether it tells the user.

    ``expires_at``, ``max_runs`` and ``created_at`` are what a claim about WHEN the job stops
    reads: the two columns an end is stored in, and the creation moment a rule with no start
    of its own is anchored at.  They travel as stored, so the reading — on the user's clock —
    is ``job_end``'s.

    ``bound_values`` is what the job's routine is POINTED AT — each declared parameter's bound
    value, off the row's own provenance column, empty for a row no routine was applied to.  A
    rebind writes it beside the program it re-renders, and the ledger keeps no prior for it, so
    a claim about which page a job checks, or about a value a rebind left alone, reads it here.
    """

    name: str
    archived: bool
    born_this_run: bool
    touched_this_run: list[str]
    moved_this_run: list[str]
    notifies: bool
    schedule: str | None
    program: str | None
    expires: bool
    expires_at: datetime | None = None
    max_runs: int | None = None
    created_at: datetime | None = None
    bound_values: dict[str, str] = Field(default_factory=dict)

    @property
    def changed_this_run(self) -> bool:
        """Whether this sample's runs touched the row at all."""
        return bool(self.touched_this_run)


class Arm(BaseModel):
    """ONE arm of a cohort: the input this arm ran, and the world it ran against.

    An arm is the general unit, and "one world in five wordings" is its special case — five
    arms whose ``world`` happens to be the same object.  Chat is that case; a collector is
    not, because its arms vary the job's own inputs (the bound values and the pages that
    answer them together), so each carries its own world by construction.

    Carrying the world HERE rather than on the cohort is what makes the two expressible by one
    seam.  A per-cohort world cannot be narrowed to an arm afterwards, so a cohort holding one
    would force every non-chat shape into a special case at the point a claim is answered —
    which is the layer that must stay shape-agnostic.

    ``label`` is the anchor: it names the arm in the report's rows and inside every sample's
    own name, so a reader who sees "phrasing 3 diverged" has one thing to look up.  ``text``
    is what that arm actually said, verbatim — a label with no text beside it is a dead
    anchor, unreadable exactly when a reader needs it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    label: str
    text: str
    world: World


def distinct_worlds(arms: Sequence[Arm]) -> list[World]:
    """The distinct worlds a set of arms ran against, in arm order.

    ONE entry is the chat and micro-context shape — every arm answered against the same
    ground, which the report states once.  Several is a cohort whose arms each brought their
    own, which is what varying a job's inputs looks like.  Defined once because three readers
    ask it — the cohort, the world fold, and the fold's own counts — and three spellings of
    "are these the same world" is three things to disagree."""
    seen: list[World] = []
    for arm in arms:
        if arm.world not in seen:
            seen.append(arm.world)
    return seen


# What ``SampleObservation.field`` answers for a field the draw did not return.  A rendered
# word rather than an empty string, because it travels into the variance table as a value and
# a blank cell there reads as a rendering bug rather than as an absence.
FIELD_UNSET = "unset"


class OutputField(BaseModel):
    """One field of the STRUCTURED OUTPUT a draw returned, as a string.

    A single-call context (the classifier, the framer, the labeller, the binder, the browse
    extractor) answers with a typed result and touches no store — so what a case asserts and
    what it measures are the same thing: this result's fields.  They are carried as strings
    for the reason every :class:`Feature` value is a string — what is being compared is
    DISTINCTNESS, and two samples agree when they produced the same value.

    Deliberately NOT folded into :class:`StoredEntry`: three claims already read ``entries``
    as "an entry in one of Penny's collections", and a draw's fields arriving in that list
    would silently put them inside every one of those answers."""

    name: str
    value: str


class SampleObservation(BaseModel):
    """Everything one sample left behind, read while its database was still live.

    The whole of what a case can assert or measure.  Read once, at the only moment it is
    available, so the claims are pure functions over data rather than callbacks racing a
    database that is about to close.

    ``complete`` is the COMPLETENESS gate, read BEFORE anything is pooled: a sample's database
    exists from sample START, so counting files as samples reports dead samples as behavioural
    variance — 17 of 31 in the first prototype run.  ``exclusion`` says why, by name."""

    name: str
    phrasing: str
    # WHICH arm produced this sample, by index into the cohort's own arm list.  A hard link,
    # because the world a claim reads is the arm's: matching on the rendered LABEL instead
    # meant an observer that stamped a label no arm carried answered against an empty world,
    # and the claims that actually read one (``_each_source_kept``, ``_nothing_excluded``)
    # would then pass VACUOUSLY rather than fail.  A vacuous pass is the quieter failure and
    # the harder one to notice, so the link is an index and a mismatch raises.
    arm: int = -1
    complete: bool = True
    exclusion: str | None = None
    landed: str | None = None
    # The routine the move NAMED, off the landed transition's own ``skill_name`` — which
    # routine the decision recognised as covering the ask, before anything was stood up.
    # Beside ``landed`` because it is the same row and the same reading: where the machine
    # went, and what it went there about.  ``None`` where the move named none, which is a
    # real reading (an ordinary chat turn names no routine) and not a missing one.
    decision_skill: str | None = None
    # WHEN the landed move was recorded, and the timezone the user's profile carries — the
    # two facts a claim about a job's end is read against, since an ask states its end on the
    # user's own clock ("tonight", "sunday night") relative to the turn that carried it.
    turn_at: datetime | None = None
    timezone: str | None = None
    # The parameters the round is still WAITING ON, off the landed transition's own
    # ``round_shortfall`` — their declared names, in the routine's declared order.  Empty
    # where the move recorded no shortfall, which is the ordinary reading (only a move landing
    # in request carries one) and not a missing one.
    awaiting: list[str] = Field(default_factory=list)
    # The ordered moves this sample's driver walked, in the VOCABULARY of the observer that
    # read it: a chat sample carries the conversation machine's own walk (``idle→learn,
    # learn→apply``, or ``no move`` when it recorded none), a collector sample the ordered
    # shapes of its cycles (``wrote, quiet, wrote+told``).  One field because both answer
    # "what did this run do, in order", and two features read it — :data:`TRANSITIONS` and
    # :data:`CYCLE_SCRIPT` — because the readings they can produce are not the same set, and
    # a feature's ``absent`` is a claim about the observer that fills it.
    walk: str = ""
    routines: list[RoutineRecord] = Field(default_factory=list)
    entries: list[StoredEntry] = Field(default_factory=list)
    # Every entry the store HOLDS when the sample ends — the same ``StoredEntry`` shape as
    # ``entries``, because it is the same three facts about the same rows and two shapes for
    # one thing drift.  ``entries`` is the subset this sample WROTE, which is the right
    # reading for "did she invent that" and the wrong one for "what does the store hold":
    # a cycle that correctly wrote nothing leaves it empty while the store still holds the
    # value it was seeded with, and in a list of writes an entry never touched and an entry
    # deleted are both simply absent.  Both questions are real and they are not the same
    # question, so the store's own end state is carried beside the sample's writes rather
    # than derived from them.
    #
    # WHICH rows are read is the fixture's, like every other observation: a chat sample
    # reads every collection, a collector cycle reads the one container its job is bound to.
    held: list[StoredEntry] = Field(default_factory=list)
    # Every entry the store held when the sample's measured turn BEGAN — ``held`` read at the
    # other end of the turn, which is what makes "what the store already held survives" a read
    # rather than an inference.  The end state alone cannot answer it: an entry the turn
    # deleted or rewrote leaves no trace in ``held``, only an absence.
    held_before: list[StoredEntry] = Field(default_factory=list)
    # Every MECHANISM the registry holds when the sample ends, archived ones included — what
    # a round's own cleanup and a running job's survival are both read off.  Beside ``held``
    # rather than derived from it, because they answer different questions about the same
    # rows: ``held`` is what a container contains, this is what the container IS.
    mechanisms: list[MechanismRecord] = Field(default_factory=list)
    # Whether PROACTIVE NOTIFICATIONS are muted for the user when the sample ends — the
    # global switch, one level up from a mechanism's own ``notifies``.  It is a row in the
    # store like every other end state (present = muted), and it is the only place two
    # neighbouring behaviours can be told apart: silencing ONE running job and silencing
    # EVERYTHING look identical on a mechanism's row, and the second is what a measured
    # sample reached for when it read the first as unavailable.
    #
    # Read by the CHAT observer, which is the only shape whose turns can move it.  A shape
    # that writes its own observation leaves the default, which is what a database nobody
    # muted holds — and no non-chat case claims it.
    muted: bool = False
    # Every image the MEDIA store holds when the sample ends — a store of its own, beside the
    # collections, which is where a picture made on request lands.  Read by the CHAT observer;
    # a shape that writes its own observation leaves it empty, and no non-chat case claims it.
    images: list[StoredImage] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    reply: str = ""
    reply_embedding: list[float] | None = None
    # Everything the round was given, and which part of it was stated — see ``Given``.
    given: Given = Given()
    # The container the round was FRAMED on, read off the move that settled it — the same anchor
    # the turn's instruction rendered.  A write that landed anywhere else invented a destination
    # over one it was given.
    container: str | None = None
    # Collections this round created that carry a schedule or a notify flag.  Learning must not
    # INSTANTIATE, so this is empty on a correct round.
    scheduled: list[str] = Field(default_factory=list)
    # The fields of the structured output this sample's draw returned — empty for a sample
    # driven through the agent loop, which leaves its trail in the stores instead.
    output: list[OutputField] = Field(default_factory=list)
    # ── What a COLLECTOR cycle left, read where a chat turn has no equivalent ──
    #
    # The run record the cycle closed with — RECORD FIELDS, read literally.  ``run_outcome``
    # is the cycle's own determination (``worked`` changed something, ``no_work`` closed
    # clean and changed nothing); ``run_reason`` carries a write-gate STOP by name.  Neither
    # is a route: nothing here reads a tool name or an ordering.
    run_outcome: str | None = None
    run_reason: str | None = None
    # What reached the SEND QUEUE, one string per message.  Counted rather than joined,
    # because "told once" and "told twice" are different findings and a joined blob cannot
    # tell them apart.  ``reply`` above is the same messages as one text, for reply spread.
    notifications: list[str] = Field(default_factory=list)

    @property
    def stored_text(self) -> str:
        """Every entry this sample wrote, key and content together."""
        return " ".join(entry.text for entry in self.entries)

    def field(self, name: str) -> str:
        """This sample's value for one field of its draw's structured output.

        :data:`FIELD_UNSET` where the draw returned no such field, which is a REAL reading and
        not a missing one: a draw that answered with the wrong shape genuinely has nothing
        there, and a claim about that field is false of the sample rather than unasked."""
        return next((one.value for one in self.output if one.name == name), FIELD_UNSET)

    @property
    def output_text(self) -> str:
        """Every field the draw returned, name and value together — what a provenance claim
        over a structured answer reads."""
        return " ".join(f"{one.name} {one.value}" for one in self.output)


# ── What a case claims ───────────────────────────────────────────────────────
class ClaimOutcome(BaseModel):
    """One sample's answer to one claim."""

    sample: str
    ok: bool
    rationale: str | None = None


class Claim(BaseModel):
    """One named claim, answered by every sample it applied to.

    A claim RECORDS rather than raises.  Whether a rate is a failure is the recorded floor's
    job — and until the code owner accepts a floor there is none, so a ported case reports its
    numbers instead of going red on the first miss."""

    label: str
    category: SpecCategory
    kind: str = "state"
    outcomes: list[ClaimOutcome] = Field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def rationales(self) -> list[str]:
        """The distinct notes from the samples that missed — what a reader reads first."""
        seen: list[str] = []
        for outcome in self.outcomes:
            if not outcome.ok and outcome.rationale and outcome.rationale not in seen:
                seen.append(outcome.rationale)
        return seen


# ── What a case measures ─────────────────────────────────────────────────────
class Consequence(StrEnum):
    """What a divergence on this feature COSTS — declared where the case measures it.

    Two classes, because there are two answers a reader needs and no more.  CONSEQUENTIAL means
    a different value implies a different END STATE, so the sample is worth looking at
    individually.  COSMETIC means it is measured, its entropy reported and its ceiling proposed,
    and it says nothing about any one sample: container naming is unconstrained in BOTH measured
    models at 0.90 entropy, which makes it a system-level finding for the variance table rather
    than fifteen per-sample findings."""

    CONSEQUENTIAL = "consequential"
    COSMETIC = "cosmetic"


@dataclass(frozen=True)
class Feature:
    """One measured axis: a name, how to read one sample's value for it, and what a divergence
    on it costs.

    A string, because what is being measured is DISTINCTNESS — two samples agree when they
    produced the same value, and every feature answers that the same way whatever it is
    made of."""

    name: str
    read: Callable[[SampleObservation], str]
    consequence: Consequence = Consequence.CONSEQUENTIAL
    # The reading that means this feature saw NOTHING — what ``read`` returns for a sample that
    # produced no value at all.  Declared so :func:`feature_variance` can tell "every sample
    # agreed" from "this feature never read anything", which are the same 0.000 and opposite
    # findings.  ``None`` means the feature declares no such reading of its own; the EMPTY
    # STRING is a legitimate declaration, not the absence of one — a distinction the earlier
    # ``str = ""`` default could not express, which is how a field populated only on one
    # outcome pooled to a serene 0.000 and proposed a gate on it.
    #
    # **It is a claim about the OBSERVER, and the observer must be able to produce it (#2061).**
    # A declared value no observation on that path yields makes the blindness guard INERT
    # exactly where it reads as armed — the guard cannot fire, so a cohort that really did read
    # nothing would pool to 0.000 and render as agreement, which is the trap the declaration
    # exists to close.  So the answer is per feature and per path, and where no observation can
    # read nothing the honest declaration is ``None``: the pooler still catches a value that
    # came back BLANK, which is the one "nothing" every feature shares.  Each declaration is
    # pinned against its real observer in ``tests/test_eval_harness.py``.
    absent: str | None = None


TOOL_SEQUENCE = Feature(
    "tool sequence", lambda o: " → ".join(o.tool_sequence) or "no call", absent="no call"
)
ROUTINE_SHAPE = Feature(
    "routine shape",
    lambda o: " | ".join(r.shape for r in o.routines) or "no routine",
    absent="no routine",
)
# What the framer called the routine.  Measured DIRECTLY rather than through the container it
# produces: a container name is `derive_collection_name(skill.name, [parameter values])`, and on
# the reference run the parameter half was byte-identical across all 18 samples — so measuring
# the container measured the routine name through a slug function, under a label that hid what
# it was.  Nothing about the naming MECHANISM is loose: `round_framing.container_name` is fully
# deterministic and public precisely so a fixture cannot grow a second copy of the scheme.  What
# varies is the framer's output, upstream of it.
#
# COSMETIC because the end state is equivalent whichever name is drawn — `watch_price` and
# `monitor_listing_price` leave the same round, the same write and the same container shape
# behind — so its spread belongs in the variance table as the FRAMER's naming spread, never as
# a fact about one sample.
ROUTINE_NAME = Feature(
    "routine name",
    lambda o: ", ".join(sorted({r.name for r in o.routines})) or "none",
    consequence=Consequence.COSMETIC,
    absent="none",
)
ENTRIES_STORED = Feature("entries stored", lambda o: str(len(o.entries)))
# The CONVERSATION MACHINE's walk, as a chat observer reads it: ``no move`` is what it returns
# for a sample whose machine recorded no transition at all, so the blindness guard has a
# reading it can actually fire on.
TRANSITIONS = Feature("transitions", lambda o: o.walk, absent="no move")
# What a COLLECTOR's cycles did, in order — the same observation field read by the observer a
# collector has instead of a machine walk, in that observer's own vocabulary (``wrote`` ·
# ``quiet`` · ``…+told``).  It declares NO absent reading, because none exists among the samples
# it is pooled over: every cycle that ran has one of those shapes, and the one shape that means
# nothing ran (``no run``) EXCLUDES the sample before pooling.  Sharing ``transitions``' own
# ``no move`` here would be a declaration this path can never produce (#2061).
CYCLE_SCRIPT = Feature("cycle script", lambda o: o.walk)

# What ``JOB_TERMS`` reads on a turn that stood no job up — and the words the three terms are
# rendered in.  Named because a feature's value is a diff-join key exactly as a claim's label
# is: one respelling splits a feature's history into two distributions.
NO_JOB = "no job"
_UNSCHEDULED = "unscheduled"
_TELLS = "tells"
_SILENT = "silent"
_ENDS = "ends"
_RUNS_ON = "runs on"
_TERM_SEPARATOR = " · "
_JOB_SEPARATOR = " | "


def _terms_drawn_for(mechanism: MechanismRecord) -> str:
    """One job's terms as the turn committed to them: the rule VERBATIM, whether it says
    anything, and whether it stops."""
    return _TERM_SEPARATOR.join(
        (
            mechanism.schedule or _UNSCHEDULED,
            _TELLS if mechanism.notifies else _SILENT,
            _ENDS if mechanism.expires else _RUNS_ON,
        )
    )


def _job_terms(observation: SampleObservation) -> str:
    """The terms of every job THIS TURN stood up or configured, sorted.

    Born OR changed, because standing a job up looks different from either end of a round: a
    cold ask mints the container and configures it in one turn, while a round that was framed
    already built its container earlier and this turn only settles its terms.  Both are the
    same event — a turn deciding how a job will run — and a reading keyed to only one of them
    would go blind on half the edges that draw terms at all.

    Sorted rather than in registry order so two samples that configured the same jobs agree
    whatever order the rows come back in, and the jobs the world was ALREADY running are
    excluded by construction: a feature reading every row would pool the fixture's five
    schedules on every sample and report the seed as agreement."""
    stood_up = sorted(
        _terms_drawn_for(one)
        for one in observation.mechanisms
        if one.born_this_run or one.changed_this_run
    )
    return _JOB_SEPARATOR.join(stood_up) or NO_JOB


# The TERMS a stand-up turn DREW.  The container's name, the routine it runs and the values it
# is pointed at are all supplied framework-side from the round's framing (#1869), so what is
# left for the model to choose is exactly this: how often it fires, whether it says anything,
# and whether it stops.  Three edges draw them — a cold apply, an accepted offer, and a binding
# completed over two turns — and until now nothing measured them at all.
#
# The rule travels VERBATIM rather than through ``cadence_seconds``, because §5's rule is to
# measure the value at what it VARIES at and the gap is computed downstream of the rule the
# draw actually wrote.  That pairs with the assertion side rather than duplicating it: the
# cases' cadence claim reads the GAP, so a sample that merely spelled one cadence differently
# passes the claim and shows here as a divergence — which is the row a reader should open.
#
# CONSEQUENTIAL: a different cadence, a different hour to fire at, a job that says nothing or
# one that never stops are all a different job, and a sample that drew one is worth reading on
# its own.
JOB_TERMS = Feature("job terms", _job_terms, absent=NO_JOB)

# Reply spread is pairwise rather than per-sample, so it is a marker the pooler recognises
# rather than a value any one sample carries.
REPLY_SPREAD = Feature("reply text", lambda o: o.reply, consequence=Consequence.COSMETIC)


def output_field(
    name: str,
    *,
    consequence: Consequence = Consequence.CONSEQUENTIAL,
    absent: str | None = None,
) -> Feature:
    """One field of a draw's STRUCTURED OUTPUT, as a measured axis.

    A single-call context returns a typed result rather than leaving a trail through the
    stores, so its variance axes are simply its own fields: the same values the case asserts,
    compared across the cohort.  No new concept — this is :class:`Feature` reading
    ``SampleObservation.output`` instead of a chat-shaped attribute.

    ``absent`` says WHETHER THE OBSERVER CAN OMIT THIS FIELD, which is the only thing that
    decides whether :data:`FIELD_UNSET` is a reading this axis can ever take (#2061).  The
    default is that it cannot: an observer that emits the field on every outcome — the
    classifier's three, the extractor's three, the framer's name and description and count —
    never produces :data:`FIELD_UNSET`, so declaring it would arm the blindness guard on a
    value no observation yields.  What such a field CAN come back as is BLANK, and the
    pooler catches an all-blank feature whatever it declares.  Pass ``absent=FIELD_UNSET``
    for a field the observer omits on some outcomes — the binder's value for a parameter it
    reported missing — where the omission IS the reading meaning the draw produced nothing
    here, and a cohort where no sample filled it must read as blind rather than as fifteen
    samples agreeing."""
    return Feature(name, lambda o: o.field(name), consequence=consequence, absent=absent)


# ── A measured SHARE: a fraction per sample, which has no entropy ────────────
class ShareReading(BaseModel):
    """One sample's reading of a measured share: the fraction, and what made its numerator.

    ``evidence`` rides with the number because a share alone sends its reader to the
    transcripts, and the things it counted are the finding."""

    value: float
    evidence: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Share:
    """One measured axis whose reading is a FRACTION of something the sample produced.

    :class:`Feature`'s sibling rather than one of them.  A feature measures DISTINCTNESS: two
    samples agree when they produced the same value, and entropy is taken over those values.
    A fraction is continuous, so fifteen samples give close to fifteen different numbers
    whatever the behaviour, and its entropy would read near 1.0 on a cohort in perfect
    agreement.  So a share is reported as what it is: where the cohort's readings sit (the
    median), how far they reach (the range), and which samples read above ``open_above``.

    ``read`` answers ``None`` for a sample that produced nothing to take the fraction of.
    That is this axis's absent reading, and a share that read ``None`` on every pooled sample
    is BLIND, exactly as a feature that read its absent value on every sample is.

    MEASURED, never gated: a share proposes no ceiling, enters no headline, decides no
    sample's standing and answers no claim.  ``open_above`` says which samples a reader is
    pointed at.  It is not a threshold anything is compared against.

    ``says`` is what the number IS, in one sentence, rendered under its row: a fraction with
    no statement of what it is a fraction of is a number a reader has to look up."""

    name: str
    read: Callable[[SampleObservation], ShareReading | None]
    open_above: float
    says: str


# What an exclusion is called when the observation that carried it named no reason.
UNEXPLAINED_EXCLUSION = "the measured turn never ran"


def exclusion_reason(sample: SampleObservation) -> str:
    """Why this sample could not be pooled, named.

    ONE answer, because two surfaces state it — the pooled variance the case document renders,
    and the per-sample record the artifact writes (#2125) — and a second spelling would let a
    document and a record name the same lost sample two different ways."""
    return sample.exclusion or UNEXPLAINED_EXCLUSION


class ExcludedSample(BaseModel):
    name: str
    reason: str


class PhrasingRow(BaseModel):
    """One phrasing's own view of a feature — DIAGNOSTIC, never locked.

    At 3 samples a per-phrasing entropy would be noise wearing a number's clothes, so none is
    reported.  What is here is what 3 samples can honestly say: how many distinct values this
    wording produced, and whether any appeared under NO other wording."""

    arm: str
    n: int
    distinct: int
    values: list[str]
    only_here: list[str] = Field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.only_here)


class VarianceFeature(BaseModel):
    """One feature's spread across the POOLED cohort, plus its per-phrasing rows.

    ``entropy`` is Shannon entropy over the value distribution normalised by ``log(n)`` — the
    spread a cohort of this size could at most show — so 0.0 is total agreement and 1.0 is
    every sample distinct.  ``n`` rides along because that denominator makes the number
    incomparable across cohort sizes."""

    name: str
    n: int
    distinct: int
    modal: int
    entropy: float
    phrasings: list[PhrasingRow] = Field(default_factory=list)
    # Every sample read this feature's ABSENT value, so it saw nothing at all.  Carried
    # separately from the entropy because the two are the same 0.000 and opposite findings:
    # total agreement is the best result a feature can report, and reading nothing on every
    # sample is a feature that CANNOT report an outlier — which is worse than not measuring
    # at all, since the table shows a number either way.  What this catches concretely: a
    # non-chat fixture whose tool-sequence reader is filtered to the chat agent's rows comes
    # back empty on every sample and pools to a serene 0.000.
    blind: bool = False

    @property
    def modal_share(self) -> float:
        return self.modal / self.n if self.n else NO_SPREAD

    @property
    def saturated(self) -> bool:
        """Whether this feature is too spread for a ceiling to mean anything.

        A ceiling catches a RISE, and normalised entropy is bounded at 1.0 — so a feature already
        near the top of its range gets a ceiling it could never breach, which prints a guard that
        cannot fire.

        The boundary is NO MAJORITY BEHAVIOUR: the modal value is not shared by even half
        the samples (exactly half still counts as a majority).  Chosen over "most values are
        distinct" because that reads the wrong quantity at small N — two distinct values in
        three samples is ordinary spread, not saturation, and it would have silenced ceilings
        on cohorts that plainly deserve one.  Measured on the reference run this separates the
        real cases cleanly: the framer's naming sits at modal 5/15 and proposes nothing, while
        tool sequence at 13/15 and routine shape at 15/15 both propose.

        It needs no new constant — "half" is the same majority notion standing decides on — and
        it is read off the two numbers the table already shows.  A judgement about where to stop
        PROPOSING; nothing is gated on it and nothing fails because of it."""
        return self.n > 0 and self.modal * 2 < self.n


class TextSpread(BaseModel):
    """How far the cohort's REPLIES stand apart, via the shared similarity primitives rather
    than a phrasing list somebody guessed: ``cosine_similarity`` over the embeddings the
    replies already carry (every send is embedded at egress, so this costs no model call) and
    ``token_containment_ratio`` over the words themselves.  Two views because they fail
    differently — an embedding says two replies are ABOUT the same thing, which at fixed topic
    is nearly always true, while containment says how much vocabulary they actually reuse."""

    pairs: int
    cosine_mean: float
    cosine_min: float
    containment_mean: float
    # How many of those pairs the COSINE half could actually be computed on.  Carried
    # separately because a reply with no embedding contributes to containment and not to
    # cosine, and ``_mean([])`` is 0.000 — which reads as "every pair maximally dissimilar"
    # when the truth is "no pair was measurable".  A collector's notification is the standing
    # case: a cycle ENQUEUES and the drainer is a separate schedule, so its text is usually
    # absent from the outgoing messages the embeddings are read off (#2017).
    cosine_pairs: int = 0

    @property
    def cosine_measurable(self) -> bool:
        """Whether the cosine half is a reading at all."""
        return self.cosine_pairs > 0


class SampleShare(BaseModel):
    """One pooled sample's reading of a share, by the sample's own name."""

    sample: str
    value: float
    evidence: list[str] = Field(default_factory=list)


class ShareSpread(BaseModel):
    """One :class:`Share` across the POOLED cohort: where its readings sit and how far they
    reach.

    ``readings`` holds only the samples that HAD a reading, in the order they were driven, and
    ``n`` is every pooled sample, so ``len(readings)`` of ``n`` is how much of the cohort this
    number speaks for.  Everything else is derived from those two, so no summary figure can
    drift from the readings it summarises."""

    name: str
    n: int
    open_above: float
    says: str
    readings: list[SampleShare] = Field(default_factory=list)

    @property
    def blind(self) -> bool:
        """Whether NO pooled sample had anything to take the fraction of.  An empty pool is
        not blind: nothing was pooled, which the excluded-samples section reports."""
        return self.n > 0 and not self.readings

    @property
    def values(self) -> list[float]:
        return [reading.value for reading in self.readings]

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.readings else NO_SPREAD

    @property
    def low(self) -> float:
        return min(self.values, default=NO_SPREAD)

    @property
    def high(self) -> float:
        return max(self.values, default=NO_SPREAD)

    @property
    def above(self) -> list[SampleShare]:
        """The samples a reader is pointed at: every reading strictly above ``open_above``."""
        return [reading for reading in self.readings if reading.value > self.open_above]


class CohortVariance(BaseModel):
    """A case's whole measured half: what was pooled, what was thrown out, and the spread."""

    pooled: int = 0
    driven: int = 0
    excluded: list[ExcludedSample] = Field(default_factory=list)
    features: list[VarianceFeature] = Field(default_factory=list)
    text: TextSpread | None = None
    # The measured SHARES, beside the features and never among them: every reader of
    # ``features`` computes a headline, a ceiling or a glyph from an entropy, and a share has
    # none to give.
    shares: list[ShareSpread] = Field(default_factory=list)

    @property
    def dominant_exclusion(self) -> tuple[str, int] | None:
        """The reason that cost this case the most samples, or ``None`` when it lost none.

        Read off the exclusions this cohort ALREADY named rather than from the run-level fault
        tally: that tally is per PROCESS and cannot say which case a fault landed in, and a
        second accounting of the same samples is a second number to disagree with the first.
        Ties break on the reason so a re-render reads identically."""
        if not self.excluded:
            return None
        counts = Counter(sample.reason for sample in self.excluded)
        reason, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        return (reason, count)


class VarianceHeadline(BaseModel):
    """What varies MOST right now, and how much of the case varies at all.

    Max entropy over EVERY feature — no gateable filter.  Surfacing and gating are different
    jobs and saturation belongs only to the second: a ceiling exists to catch a RISE, so a
    feature with no majority behaviour can carry none, and ``proposed_ceiling`` still refuses
    one.  The headline answers a different question — *which aspect of this case is most
    variant* — and excluding a saturated feature from it hides exactly the answer.  If the
    framer's naming is the most variant thing here, that is true, and it is a finding about the
    system rather than noise.

    ``varying`` is the shape beside the magnitude, because "a bunch at zero and one high" and
    "everything wobbling" are different findings that one maximum cannot tell apart.  Counted
    STRUCTURALLY — more than one distinct value — so no magnitude threshold enters."""

    feature: str | None = None
    entropy: float = NO_SPREAD
    varying: int = 0
    total: int = 0

    @property
    def has_reading(self) -> bool:
        return self.feature is not None


def variance_headline(features: Sequence[VarianceFeature]) -> VarianceHeadline:
    """The most variant feature across ``features``, and how many of them vary at all."""
    top = max(features, key=lambda feature: feature.entropy, default=None)
    return VarianceHeadline(
        feature=top.name if top is not None else None,
        entropy=top.entropy if top is not None else NO_SPREAD,
        varying=sum(1 for feature in features if feature.distinct > 1),
        total=len(features),
    )


class RecordedCeiling(BaseModel):
    """A feature's ceiling as it would be RECORDED — ``(feature, model, N, value)``.

    Neither qualifier is decoration, and a comparison across either is REFUSED:

    * **N** — normalised entropy is biased upward at small N (0.527 at N=32 reads 0.605 at
      N=15 for the same behaviour).
    * **MODEL** — measured, two models differ ~3x on the same features, so one shared ceiling
      would be useless for the consistent model and permanently failing for the variant one."""

    feature: str
    model: str
    n: int
    value: float


class CeilingVerdict(BaseModel):
    """The one-sided regression check's answer.  ``comparable`` False means the question is
    refused rather than answered wrongly."""

    feature: str
    comparable: bool
    regressed: bool = False
    observed: float = NO_SPREAD
    ceiling: float = NO_SPREAD
    note: str = ""


# ── The math ─────────────────────────────────────────────────────────────────
def normalised_entropy(values: Sequence[str]) -> float:
    """Shannon entropy over the distribution of ``values``, normalised by ``log(n)``.

    ``log(n)`` — not ``log(distinct)`` — because the question is how much of the spread this
    cohort COULD have shown it actually did; dividing by the distinct count would score "two
    values, evenly split" the same as "fifteen values, evenly split"."""
    if len(values) < 2:
        return NO_SPREAD
    counts = Counter(values)
    total = len(values)
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    # A single-valued cohort computes to NEGATIVE zero, which renders as ``-0.000`` and reads
    # as a number rather than as the absence of one.
    return max(NO_SPREAD, entropy / math.log(total))


def _values_under_other_arms(by_arm: dict[str, list[str]], arm: str) -> set[str]:
    return {value for other, values in by_arm.items() if other != arm for value in values}


def _phrasing_rows(feature: Feature, samples: Sequence[SampleObservation]) -> list[PhrasingRow]:
    by_arm: dict[str, list[str]] = {}
    for sample in samples:
        by_arm.setdefault(sample.phrasing, []).append(feature.read(sample))
    # "Only under this wording" is a comparison BETWEEN wordings, so one phrasing has nothing
    # to say — every value would be trivially unique to the only arm there is.
    comparable = len(by_arm) > 1
    return [
        PhrasingRow(
            arm=arm,
            n=len(values),
            distinct=len(set(values)),
            values=[value for value, _ in Counter(values).most_common()],
            only_here=(
                sorted(set(values) - _values_under_other_arms(by_arm, arm)) if comparable else []
            ),
        )
        for arm, values in by_arm.items()
    ]


def feature_variance(feature: Feature, samples: Sequence[SampleObservation]) -> VarianceFeature:
    """One feature's pooled spread plus its per-phrasing diagnostic rows."""
    values = [feature.read(sample) for sample in samples]
    counts = Counter(values)
    return VarianceFeature(
        name=feature.name,
        n=len(values),
        distinct=len(counts),
        modal=max(counts.values()) if counts else 0,
        entropy=normalised_entropy(values),
        phrasings=_phrasing_rows(feature, samples),
        blind=_is_blind(feature, values),
    )


def _is_blind(feature: Feature, values: Sequence[str]) -> bool:
    """Whether this feature read NOTHING on every sample it was pooled over.

    Two ways a feature can have read nothing, and both have to hold for a feature nobody has
    written yet:

    * **No value at all.**  The empty string is the one "nothing" every feature shares, whatever
      it calls its own — so an all-empty pooling is blind by construction and needs no
      declaration.  This is what a structured field populated only on one outcome does on a run
      where that outcome never happens, and it goes blind precisely on the runs that look best.
    * **The feature's own declared absent reading** — ``no call``, ``no routine``, ``unset``.
      Compared against ``None`` rather than falsiness, so a feature may legitimately declare the
      empty string; a feature declaring nothing is never blind by this route."""
    if not values:
        return False
    seen = set(values)
    return seen == {""} or (feature.absent is not None and seen == {feature.absent})


def text_spread(samples: Sequence[SampleObservation]) -> TextSpread | None:
    """Pairwise reply spread over the pooled cohort, or ``None`` below two replies.

    A reply carrying no embedding contributes to containment and not to cosine — the two are
    reported over the pairs each could actually be computed on, rather than dropping a reply
    from both because one half of it is missing."""
    replies = [sample for sample in samples if sample.reply.strip()]
    if len(replies) < 2:
        return None
    cosines: list[float] = []
    containments: list[float] = []
    for index, left in enumerate(replies):
        for right in replies[index + 1 :]:
            containments.append(token_containment_ratio(left.reply, right.reply))
            if left.reply_embedding and right.reply_embedding:
                cosines.append(cosine_similarity(left.reply_embedding, right.reply_embedding))
    return TextSpread(
        pairs=len(containments),
        cosine_mean=_mean(cosines),
        cosine_min=min(cosines) if cosines else NO_SPREAD,
        containment_mean=_mean(containments),
        cosine_pairs=len(cosines),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else NO_SPREAD


def share_spread(share: Share, samples: Sequence[SampleObservation]) -> ShareSpread:
    """One share's readings across the samples it is pooled over."""
    read = [(sample.name, share.read(sample)) for sample in samples]
    return ShareSpread(
        name=share.name,
        n=len(samples),
        open_above=share.open_above,
        says=share.says,
        readings=[
            SampleShare(sample=name, value=reading.value, evidence=reading.evidence)
            for name, reading in read
            if reading is not None
        ],
    )


def pool(
    samples: Sequence[SampleObservation],
    features: Sequence[Feature],
    shares: Sequence[Share] = (),
) -> CohortVariance:
    """Gate for completeness, THEN pool — the order is the point.

    Nothing is measured over a sample that did not run, and what is excluded is NAMED rather
    than subtracted, so a run that lost half its cohort reads as one that lost half its cohort
    instead of as a suspiciously tidy one."""
    excluded = [
        ExcludedSample(name=s.name, reason=exclusion_reason(s)) for s in samples if not s.complete
    ]
    kept = [sample for sample in samples if sample.complete]
    structural = [feature for feature in features if feature is not REPLY_SPREAD]
    return CohortVariance(
        pooled=len(kept),
        driven=len(samples),
        excluded=excluded,
        features=[feature_variance(feature, kept) for feature in structural],
        text=text_spread(kept) if REPLY_SPREAD in features else None,
        shares=[share_spread(share, kept) for share in shares],
    )


def proposed_ceiling(
    feature: VarianceFeature, model: str, margin: float = CEILING_MARGIN
) -> RecordedCeiling | None:
    """The ceiling this run PROPOSES — observed plus the sampling margin — or ``None`` where the
    feature is already too spread for one to mean anything.

    Proposed, never locked: a near-ceiling feature is a defect to fix first, not a threshold to
    record, and proposing one anyway prints a guard that could not fire.

    A BLIND feature is refused for the same reason from the other end: it reads its absent
    value on every sample, so its entropy is 0.000 for want of any reading at all and a
    ceiling recorded there would lock in the blindness as the expected behaviour."""
    if feature.saturated or feature.blind:
        return None
    return RecordedCeiling(
        feature=feature.name,
        model=model,
        n=feature.n,
        value=round(min(feature.entropy + margin, 1.0), 3),
    )


def compare_to_ceiling(
    ceiling: RecordedCeiling, observed: VarianceFeature, model: str
) -> CeilingVerdict:
    """The one-sided regression check.  A different MODEL or cohort SIZE is incomparable and
    says so — answering would be answering a question nobody asked."""
    if ceiling.model != model:
        return CeilingVerdict(
            feature=ceiling.feature,
            comparable=False,
            note=(
                f"recorded on {ceiling.model}, observed on {model} — measured, two models "
                f"differ ~3x on the same feature, so a shared ceiling measures neither"
            ),
        )
    if ceiling.n != observed.n:
        return CeilingVerdict(
            feature=ceiling.feature,
            comparable=False,
            note=(
                f"recorded at N={ceiling.n}, observed at N={observed.n} — normalised entropy "
                f"is biased upward at small N, so the two are different statistics"
            ),
        )
    return CeilingVerdict(
        feature=ceiling.feature,
        comparable=True,
        regressed=observed.entropy > ceiling.value,
        observed=observed.entropy,
        ceiling=ceiling.value,
    )


# ── Cost ─────────────────────────────────────────────────────────────────────
class SampleCost(BaseModel):
    """What ONE sample spends.  Per sample, never per run — a total is not comparable across
    cohort sizes, the same trap the entropy denominator is.

    INPUT and OUTPUT are split because they mean different things: input is OURS (prompt and
    context design), so a rise is what a prompt edit regresses; output is the MODEL's, so a
    rise on a fixed prompt is a model or config change."""

    samples: int
    calls: float
    seconds: float
    input_tokens: float
    output_tokens: float
    reasoning_tokens: float

    @property
    def reasoning_share(self) -> float:
        return self.reasoning_tokens / self.output_tokens if self.output_tokens else NO_SPREAD


def per_sample_cost(
    *,
    samples: int,
    calls: int,
    duration_ms: int,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
) -> SampleCost | None:
    """Divide a case's totals by the samples that produced them, or ``None`` for no samples."""
    if samples <= 0:
        return None
    return SampleCost(
        samples=samples,
        calls=calls / samples,
        seconds=duration_ms / samples / 1000,
        input_tokens=input_tokens / samples,
        output_tokens=output_tokens / samples,
        reasoning_tokens=reasoning_tokens / samples,
    )


# ── What a case's three sections are computed FROM ───────────────────────────
#
# The numbers live here; the document that renders them is ``report.py``.  The split is the
# one the fan-out depends on: a case's arithmetic is written once and every future port
# inherits it, while how a reader meets it is free to change without touching a single case.


class SpecCategory(StrEnum):
    """Which of the design's three kinds of deterministic assertion a claim is.

    The list is CLOSED and the field is REQUIRED, which is the whole point: a check that fits no
    category cannot be declared, so the audit is a fact the code states rather than a review
    somebody has to remember to run.  A list kept in prose does not stop anything being written.

    The rules themselves live in #1994 §A and #2011; they are deliberately not restated here,
    because a third copy is a third thing to drift.

    Distinct from ``Claim.kind`` (``state`` / ``reply`` / ``spine`` / ``proc``), which says
    where a claim was READ FROM and so decides how its per-sample check renders and what it
    anchors to, while ``category`` says which part of the design it satisfies.  Neither is
    derivable from the other — PROVENANCE has both a store-side claim and a reply-side one."""

    LANDED = "landed"
    STORE = "store"
    PROVENANCE = "provenance"


class AssertionRow(BaseModel):
    """One claim's aggregate across the cohort — the section-A row.

    Where a claim was read from stays on the claim.  ``Claim.kind`` decides how the PER-SAMPLE
    check renders and what it anchors to, and the aggregate row is neither of those — nothing
    on this side is gated, so there is no decision here for it to make."""

    label: str
    passed: int
    total: int
    category: SpecCategory
    rationales: list[str] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else NO_SPREAD

    @property
    def at_full(self) -> bool:
        return self.total > 0 and self.passed == self.total


class AssertionSummary(BaseModel):
    """Every deterministic check the case made, counted once — the case's assertion number.

    ASSERTIONS ARE NOT GATED.  A deterministic check is a thing expected to be strictly true of
    the run, and we expect them at 100%, so a floor under one adds nothing a reader could act
    on: it would either sit at 1.00 and never fire, or sit below and bless the defect as the
    contract.  What replaces it is this single reading, coloured on the ordinary scale and
    REPORTED — nothing on the assertion side fails a run.

    Counted as TOTAL CHECKS PASSED over TOTAL CHECKS — 9 claims x 15 samples = 135 — rather
    than as a mean of per-claim rates.  While every claim shares a denominator the two are the
    same number; they diverge the moment one does not, and the sum stays the direct reading of
    "how many of the things that had to be true were" where the mean silently reweights a claim
    that fewer samples answered."""

    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else NO_SPREAD

    @property
    def at_full(self) -> bool:
        return self.total > 0 and self.passed == self.total


def assertion_summary(rows: Sequence[AssertionRow]) -> AssertionSummary:
    """The case's one assertion number, over every claim and every sample that answered it."""
    return AssertionSummary(
        passed=sum(row.passed for row in rows), total=sum(row.total for row in rows)
    )


# ── Which samples a reader should open ───────────────────────────────────────
#
# The workflow has a human read ONE sample once the cohort is consistent, and that reading is
# sound precisely BECAUSE the samples agree.  So the document hands them the right one rather
# than making them choose — and when the cohort is still variant, the outliers are where the
# work is.


class Standing(StrEnum):
    """What one sample is FOR, to a reader deciding where to look."""

    MODAL = "modal"
    OUTLIER = "outlier"
    TYPICAL = "typical"
    DEAD = "dead"


class FeatureDivergence(BaseModel):
    """One feature on which a sample did something the representative did not.

    This — not the sample's transcript — is what makes an outlier legible.  A sample is outlying
    on a SPECIFIC feature, so rendering 19,000 characters of prose to say "its routine shape was
    `browse → browse` where the representative's was `browse → log_read → collection_write`" is
    the wrong thing by three orders of magnitude.  Show the divergence and its evidence; the
    whole transcript is in the artifact for a reader who then wants it."""

    feature: str
    value: str
    modal: str
    consequence: Consequence = Consequence.CONSEQUENTIAL


class SampleStanding(BaseModel):
    """One sample's place in the cohort: which wording it ran, and whether to open it."""

    name: str
    phrasing: str
    standing: Standing
    shape: str
    divergences: list[FeatureDivergence] = Field(default_factory=list)

    @property
    def worth_opening(self) -> bool:
        """The one sample a reader is asked to READ.  An outlier is not opened — it is summarised
        by what it did differently, which is a few rows rather than a whole transcript."""
        return self.standing == Standing.MODAL


def telling_features(
    pooled: Sequence[SampleObservation], features: Sequence[Feature]
) -> list[Feature]:
    """The features that can say something about an INDIVIDUAL sample.

    A feature whose every pooled value is distinct carries no information about any one of them:
    if all fifteen samples differ, differing is not a divergence, and "this sample named the
    container differently" is true by construction of all fifteen.  That fact belongs to the
    FEATURE and is already stated once in the variance table as `15 distinct` — restating it
    fifteen times as a per-sample finding is how a section meant to name the samples worth
    looking at came to name every one of them.

    Derived, not tuned: the condition is that no two samples agree, which is exactly the point at
    which agreement stops being able to group anything.  A feature that is merely NEARLY that
    variant still survives here, and the honest fix for that is the naming defect itself rather
    than a threshold picked to hide it."""
    structural = [f for f in features if f is not REPLY_SPREAD]
    # Below three samples "no two agree" is not a degeneracy, it is the ordinary case: with two
    # samples there is ONE pair, and their disagreeing carries exactly as much information as
    # anything else does.  The rule targets "every sample invented its own value", which cannot
    # be told apart from ordinary disagreement until a third sample can join a group.
    if len(pooled) < 3:
        return structural
    return [
        feature
        for feature in structural
        if max(Counter(feature.read(s) for s in pooled).values()) > 1
    ]


def everywhere_distinct(
    samples: Sequence[SampleObservation], features: Sequence[Feature]
) -> list[str]:
    """The features every pooled sample gave a different value — named so the report can say it
    ONCE instead of repeating it under every sample."""
    pooled = [s for s in samples if s.complete]
    telling = {feature.name for feature in telling_features(pooled, features)}
    return [
        feature.name
        for feature in features
        if feature is not REPLY_SPREAD and feature.name not in telling
    ]


def sample_shape(sample: SampleObservation, features: Sequence[Feature]) -> str:
    """One sample's whole measured shape — every feature's value at once.

    The shape, not any single feature, is what makes a sample typical or not: a sample agreeing
    on tool sequence while inventing its own routine shape is an outlier, and reading one feature
    at a time would file it under the majority."""
    return " · ".join(feature.read(sample) for feature in features if feature is not REPLY_SPREAD)


def standings(
    samples: Sequence[SampleObservation], features: Sequence[Feature]
) -> list[SampleStanding]:
    """Each sample's standing, in the order the samples were driven.

    Exactly ONE sample is modal — the first to carry the majority shape — because the workflow
    asks a human to read one, and naming eight equally would put the choice straight back on
    them.  Its shape-mates are typical and fold; everything that did something else is an
    outlier and opens.  A dead sample is neither: it has no shape to be typical of, and is the
    harness section's business rather than a reading recommendation."""
    pooled = [s for s in samples if s.complete]
    # Only the features that can distinguish one sample from another decide standing.  Including a
    # maximally-variant one makes every shape unique, so every sample becomes an outlier and the
    # modal sample is whichever happened to be first — the section names everything and therefore
    # nothing.
    # Two sets, deliberately: SHAPE decides standing and reads only the consequential features,
    # because a cosmetic divergence implies no different end state and must not make an outlier.
    # DIVERGENCES record both, so the cosmetic ones can be counted on one line rather than
    # vanishing — measured but unreported is the same blindness as unmeasured.
    telling = telling_features(pooled, features)
    shape_of = [f for f in telling if f.consequence is Consequence.CONSEQUENTIAL]
    shapes = Counter(sample_shape(s, shape_of) for s in pooled)
    modal_shape = shapes.most_common(1)[0][0] if shapes else ""
    # Divergence is measured against the REPRESENTATIVE sample rather than against each feature's
    # own mode, so the two halves of the report cannot disagree: the sample the reader is sent to
    # read has, by construction, nothing in its own divergence list.
    representative = next((s for s in pooled if sample_shape(s, shape_of) == modal_shape), None)
    seen_modal = False
    out: list[SampleStanding] = []
    for sample in samples:
        if not sample.complete:
            # A dead sample has no shape to be typical of: it never ran its measured turn, and
            # it is the harness section's business rather than a reading recommendation.
            out.append(
                SampleStanding(
                    name=sample.name,
                    phrasing=sample.phrasing,
                    standing=Standing.DEAD,
                    shape="",
                )
            )
            continue
        shape = sample_shape(sample, shape_of)
        if shape != modal_shape:
            standing = Standing.OUTLIER
        elif seen_modal:
            standing = Standing.TYPICAL
        else:
            standing, seen_modal = Standing.MODAL, True
        out.append(
            SampleStanding(
                name=sample.name,
                phrasing=sample.phrasing,
                standing=standing,
                shape=shape,
                divergences=divergences(sample, representative, telling),
            )
        )
    return out


def divergences(
    sample: SampleObservation,
    representative: SampleObservation | None,
    features: Sequence[Feature],
) -> list[FeatureDivergence]:
    """Every feature on which ``sample`` differs from the representative, with both values."""
    if representative is None or sample.name == representative.name:
        return []
    return [
        FeatureDivergence(
            feature=feature.name,
            value=mine,
            modal=theirs,
            consequence=feature.consequence,
        )
        for feature in features
        if feature is not REPLY_SPREAD
        and (mine := feature.read(sample)) != (theirs := feature.read(representative))
    ]


# ── Provenance: does a specific value trace to something the model was given? ──
#
# A SPECIFIC value is one of the classes #1994 names that can be recognised WITHOUT a
# dictionary: a number, a URL, or a capitalised NAME PHRASE of two or more words.
#
# The two-word rule is a correction of a MEASURED scorer bug rather than a preference.
# Counting every capitalised word as a name failed 15 of 18 samples on `URLs`, `English`,
# `I’ve` and `Brandt’s` — ordinary English that happens to carry a capital — which is the "too
# strict" half of the exact defect this design replaces, reintroduced by its own first
# implementation.
#
# THE BLIND SPOTS, STATED: a single-word invention, and a recombination of two real names.
# Nothing else in the suite covers them.  Tightening the rule to catch them is what the measured
# false-positive rate above rules out, so the miss is the bought half of that trade.
_NUMBER = r"\d[\d,.:%$]*"
# Dashes the model draws instead of a hyphen — folded everywhere (below), and admitted INTO the
# URL grammar because that is where one was measured: a sample cited the page it read with
# U+2011 hyphens throughout the address.
_DASHES = "‐‑‒–—−"
# The ones among them that are HYPHENS — a mark inside a word, joining its parts — as against
# the dashes that punctuate a sentence.  With the hyphen key they are the marks a word is ONE
# word across, wherever words are read: a label's or a title's words, and a token matched in a
# reply or an entry.  An en dash, an em dash and a minus sign end the word in both.
_DRAWN_HYPHENS = "\u2010\u2011"
_HYPHENS = f"-{_DRAWN_HYPHENS}"
# A URL is bounded by the characters a URI may CONTAIN, not by "everything up to a space".
# `\S*` ran the address into whatever the prose put beside it — first the full stop closing the
# sentence (`…/aurora-deck-2.`, which matches no world), then the delimiter a reply wrapped it
# in, which matched no world either and reported a correctly cited page as an invention (#2049).
#
# Subtracting the wrappers someone happened to observe repairs those and leaves the next one.
# The set a URI may contain is CLOSED (RFC 3986), so everything outside it is refused by one
# line — the backtick and the angle bracket that were seen, and equally `"`, `«»`, `“”`, `｢｣`,
# `{}`, `|` and every other mark nobody has drawn yet.  Widened only by the drawn dashes above,
# which JOIN rather than wrap; the backtick folds to an apostrophe but is a wrapper, so
# admitting it is this defect.
_URI_CHARS = rf"A-Za-z0-9._~%!$&'()*+,;=:/?#@\[\]{_DASHES}-"
_URL = rf"https?://[{_URI_CHARS}]+"
_CAPITALISED = r"[A-Z][A-Za-z'-]*"
_NAME_PHRASE = rf"{_CAPITALISED}(?:\s+{_CAPITALISED})+"
# A meridiem after a figure is part of the TIME, not a word beside it: `7 AM` is one value.
# Read apart, the figure lost the half of the day it names and the marker glued onto whatever
# capital followed it — MEASURED, `7 AM PDT` reported `AM` as an invented name.
# The gap before it is whatever space the model drew, a narrow no-break one included.
_GAP = r"[^\S\n]"
_MERIDIEM = rf"(?:{_GAP}?[AaPp]\.?[Mm]\b)"
# AN IDENTIFIER is a name, not a figure: a word with a number hyphenated onto it or into it
# (`aurora-deck-2`, `seeded-trail-cycle-2`, `patch-2.3-notes`) names one thing, and its digits
# count nothing.  Read as a figure, its number was sourced by any `2` the round happened to
# carry and unsourced wherever the only `2` was the framework's own — MEASURED, a reply naming
# the run it had looked at, by the id the self-state header rendered for it, read as inventing
# a quantity.  So it is weighed as a name is: the world states that whole run of tokens side by
# side, or it does not.  The word comes FIRST: a figure with something hyphenated after it
# (`10-minute`, `15-20 min`) is still a figure with its unit.
_IDENTIFIER_PART = r"[A-Za-z0-9']+(?:\.\d+)*"
_IDENTIFIER = (
    rf"\b[A-Za-z][A-Za-z']*(?=[A-Za-z0-9'.{re.escape(_HYPHENS)}]*[{re.escape(_HYPHENS)}]\d)"
    rf"(?:[{re.escape(_HYPHENS)}]{_IDENTIFIER_PART})+"
)
_SPECIFIC = re.compile(rf"{_URL}|{_IDENTIFIER}|{_NAME_PHRASE}|\b{_NUMBER}\b{_MERIDIEM}?")
# A percentage in front of a word for the speaker's OWN certainty measures nothing in the
# world: "not 100% sure" states no value, and MEASURED, was reported as `unsourced: ['100']`
# on a reply that named no figure at all.  The class is closed by what it means — how sure
# the speaker is — and a percentage of anything else (`100% cotton`, `15% off`) is still read.
_OWN_CERTAINTY = ("sure", "certain", "positive", "confident", "convinced")
_CERTAINTY_FIGURE = re.compile(
    rf"\b\d[\d.]*{_GAP}?%(?={_GAP}+(?:{'|'.join(_OWN_CERTAINTY)})\b)", re.IGNORECASE
)

# Words that carry a capital everywhere in English and are never part of a name, so a phrase is
# not built across them — otherwise a clause boundary glues two sentences into one "name".
_NEVER_A_NAME = frozenset({"i", "im", "ive", "ill", "id"})
# A FIELD LABEL — a capitalised word at the head of a line, followed by at most three more words
# in any case, immediately before a colon (`Genre:`, `Release date:`, `Key Features:`) — is the
# layout the model chose for its own output, not a value it states.  The value after the colon
# is still read.  MEASURED: an entry laid out under `Genre:` failed as `unsourced: ['Genre']`
# because the label glued across the line break onto the name ending the line above, and
# nothing the round was given happened to say "genre".
#
# Markdown decoration does not change what a line IS, so the label is read through it: emphasis
# around the label, on either side of the colon (`**Team News Update:**`, `**Genre**:`), and a
# heading's marks (`### Genre:`).  A HEADING LINE — heading marks, then a label-sized title and
# nothing else (`### Team News Update`) — is the same layout without the colon.  MEASURED: a
# reply laid out under `**Team News Update:**` failed as `unsourced: ['Update']`, because the
# emphasis in front of the label kept the line from reading as one.
#
# What decides layout is POSITION, never the word (#2190): the same capitalised phrase is a
# label at the head of a line and a value in the middle of a sentence.  So the head of a line is
# read through everything that can stand in front of a label there — a list bullet, a list
# number, heading marks, emphasis — and a label is any of:
#
#   a short phrase before a colon      `1.  The Scout Mission: I head over…`
#   …with an aside in brackets         `Cedar (The Top Contender): a classic choice…`
#   a TITLE: a line that is ONLY a short phrase and is marked as a title, by heading marks or
#   by emphasis                        `**Scout**`  ·  `### Team News Update`
#   a TITLE: a line that is only a short phrase and HEADS A LIST, the next line being a bullet
#   under it                           `1. Mistforge Patch Notes Tracker` / `- What it watches: …`
#
# and a list item's own NUMBER is layout too: `5.` at the head of a line counts the items, and
# MEASURED, was reported as `unsourced: ['5']` on a five-step answer over a four-step world.
# MEASURED on two models' replies describing one routine: `Scout`, `Mission`, `Logbook`,
# `Contender`, `Typical` and a dozen more, every one a title the reply gave its own list.
#
# A label and a title are bounded differently because they are different things (#2219).  A
# label INTRODUCES a value on its own line, so it is the few words a field is called by — a
# capitalised word and at most three more.  A title has its line to itself and NAMES the item
# under it: the item's own name, and the word for what kind of thing it is.  That runs longer
# than a field's name — MEASURED, `Verdant Hollow Trail Conditions Tracker`, a job's name and
# what it is, was read as the invented name `Tracker` — and a title is still a name, never a
# sentence: a headline set apart on its line (`Foxes Sign Casimir Oyelaran To A Deal`, or a
# page's own ten-word headline, MEASURED) says what happened, and its names are values.  So a
# title is a capitalised word and at most five more: room for every title-cased title the saved
# replies gave a list or a section (five words at most, MEASURED across them), and short of the
# shortest headline pinned below (seven).
#
# A word in either is a word however it was hyphenated, the drawn hyphens included: MEASURED,
# `Verdant Hollow Trail\u2011Conditions Tracker`, with U+2011 for the hyphen, broke the title
# in two.  A dash that punctuates (en, em) is not a hyphen and still ends the word, so
# `### Casimir Oyelaran\u2014Signed` is not four title words.
#
# A label that COUNTS its own item is the same layout with the number said aloud: `Step 3 –
# Store:` numbers the third thing in the reply's own list exactly as `3.` does.  The number
# stands straight after the label's first word, with or without a dash set off after it.
# MEASURED (#2203): a reply walking a routine as `Step 1 – … Step 2 – … Step 3 – …` over a
# two-step program read as inventing `3`, once a numbered list in the prompt stopped sourcing it.
#
# THE BLIND SPOT, STATED: a name or a short clause in any of those positions
# (`Casimir Oyelaran: signed`, `- Casimir Oyelaran: signed`, `### Casimir Oyelaran`,
# `**Casimir Oyelaran**` alone on its line) is a label by this definition, so an invented name
# there is not read, and nor is a figure a label counts itself by (`Batch 12: …`).  The same
# name after the colon, in a sentence, in a list item that is not a label, or in a heading
# longer than a title, still is.
_LABEL_WORD = rf"[A-Za-z][A-Za-z'{re.escape(_HYPHENS)}]*"
_LABEL_COUNT = rf"(?:{_GAP}+\d+\b(?:{_GAP}+[{re.escape(_DASHES)}-](?={_GAP}))?)?"
_LABEL_HEAD = rf"[A-Z][A-Za-z'{re.escape(_HYPHENS)}]*{_LABEL_COUNT}"
# Label words stand side by side, or either side of the marks that pair them: `Pros/Cons`.
_LABEL_GAP = r"(?:[ \t]*[/&][ \t]*|[ \t]+)"
_LABEL_MORE_WORDS = 3
_TITLE_MORE_WORDS = 5
_LABEL_WORDS_MORE = rf"(?:{_LABEL_GAP}{_LABEL_WORD}){{0,{_LABEL_MORE_WORDS}}}"
_TITLE_WORDS_MORE = rf"(?:{_LABEL_GAP}{_LABEL_WORD}){{0,{_TITLE_MORE_WORDS}}}"
_LABEL_ASIDE = rf"(?:[ \t]*\({_LABEL_WORD}{_LABEL_WORDS_MORE}\))?"
_LABEL = rf"{_LABEL_HEAD}{_LABEL_WORDS_MORE}{_LABEL_ASIDE}"
_TITLE = rf"{_LABEL_HEAD}{_TITLE_WORDS_MORE}{_LABEL_ASIDE}"
_EMPHASIS = r"[*_]*"
_EMPHASISED = r"[*_]+"
_HEADING_MARKS = r"#{1,6}[ \t]+"
_LIST_NUMBER = r"\d+[.)]"
_LIST_BULLET = r"[-*+•]"
_LIST_MARKER = rf"(?:{_LIST_BULLET}|{_LIST_NUMBER})[ \t]+"
_NUMBERED = rf"(?:{_LIST_NUMBER}[ \t]+)?"
_LINE_HEAD = rf"^[ \t]*(?:{_LIST_MARKER})?(?:{_HEADING_MARKS})?"
_LINE_END = r"[ \t]*$"
_A_BULLET_UNDER_IT = rf"(?=\n(?:[ \t]*\n)*[ \t]*{_LIST_BULLET}[ \t])"
_FIELD_LABEL = re.compile(
    rf"{_LINE_HEAD}{_EMPHASIS}{_NUMBERED}{_LABEL}{_EMPHASIS}(?=:)"
    rf"|^[ \t]*{_HEADING_MARKS}{_EMPHASIS}{_NUMBERED}{_TITLE}{_EMPHASIS}{_LINE_END}"
    rf"|{_LINE_HEAD}{_EMPHASISED}{_NUMBERED}{_TITLE}{_EMPHASISED}{_LINE_END}"
    rf"|^[ \t]*{_NUMBERED}{_TITLE}{_LINE_END}{_A_BULLET_UNDER_IT}"
    rf"|^[ \t]*{_EMPHASIS}{_LIST_NUMBER}(?=[ \t])",
    re.MULTILINE,
)

# ONE folding, used by every probe on both sides of every comparison.  A semantic check defeated
# by cosmetics is a scorer bug, and two spellings of "fold the typography" drift apart: measured,
# a reply citing the URL it was given — with U+2011 non-breaking hyphens for its dashes — read as
# an INVENTION here while the claim next door folded that dash and agreed the value was sourced.
#
# The model emits whichever apostrophe its tokenizer prefers; not folding them reported `I’ve`
# and `Brandt’s` as inventions.
_APOSTROPHES = "’‘`´"
# Spaces that are not the space key, including the zero-width one that glues two words together.
_SPACES = " ​  "
_QUOTES = (("“", '"'), ("”", '"'))
# Markdown emphasis wrapped around a value: `**$499**` is the value `$499`.
_DROPPED = "*"


def fold_typography(text: str) -> str:
    """Fold the typography the model sprinkles into its output so a SEMANTIC probe is not
    defeated by cosmetics.  A 0/N from an un-normalised probe is a scorer bug.

    The ONE definition: every probe on either side of any comparison folds through here, so a
    dash the store claim tolerates cannot be an invention to the provenance claim."""
    for mark in _APOSTROPHES:
        text = text.replace(mark, "'")
    for dash in _DASHES:
        text = text.replace(dash, "-")
    for space in _SPACES:
        text = text.replace(space, " ")
    for source, target in _QUOTES:
        text = text.replace(source, target)
    for mark in _DROPPED:
        text = text.replace(mark, "")
    return text.casefold()


# A word written whole, with a hyphen, with the hyphen the model drew instead of one, or with an
# invisible break inside it is ONE word: `silverleaf`, `silver-leaf`, `silver\u2011leaf` and
# `silver\u00adleaf` (a soft hyphen) are the same token.  MEASURED: a reply recommending
# "silver\u2011leaf moss" failed the claim that it states the answer, `silverleaf`, which it did.
#
# The marks are the hyphens (`_HYPHENS`, the same set a label's words are read across) and the
# invisible ones that sit inside a word: soft hyphen, zero-width space, non-joiner, joiner,
# word joiner, byte-order mark.  Only a join BETWEEN TWO LETTERS is taken out.  An en or em
# dash is punctuation rather than a hyphen, and a hyphen beside a figure carries meaning — a
# range, a date, a code — so `10-12` is not `1012`.  A different word is still a different
# word: `silverleaf` does not stand in `silver leaf` or `silverleaves`.
#
# An ADDRESS is not a word.  `harbor-seals.example` and `harborseals.example` are two sites, so
# nothing inside an address is joined, on either side of the comparison — a token that is an
# address is matched as that address, and a word is not found inside an address by taking the
# address's hyphens out.  An address is what the provenance read calls one: `_URL`.
_INVISIBLE_BREAKS = "\u00ad\u200b\u200c\u200d\u2060\ufeff"
_JOIN_INSIDE_A_WORD = re.compile(
    rf"(?<=[^\W\d_])[{re.escape(_HYPHENS + _INVISIBLE_BREAKS)}]+(?=[^\W\d_])"
)


def fold_for_token_match(text: str) -> str:
    """``text`` with each word's own joins taken out, then folded through ``fold_typography``
    — every word in the form it has however it was hyphenated.  Addresses are left as written.

    The two folds read a ZERO-WIDTH SPACE differently, on purpose.  ``fold_typography`` makes
    it a space, because the provenance read builds names out of words and two words glued by
    one are two words.  Here, between two letters, it is taken out: it draws nothing, so the
    word around it is the word the reader sees.  ``token_stands_in`` asks both."""
    kept: list[str] = []
    cursor = 0
    for address in _URL_PATTERN.finditer(text):
        kept += [_JOIN_INSIDE_A_WORD.sub("", text[cursor : address.start()]), address.group()]
        cursor = address.end()
    kept.append(_JOIN_INSIDE_A_WORD.sub("", text[cursor:]))
    return fold_typography("".join(kept))


def token_stands_in(token: str, text: str) -> bool:
    """Whether a token the world names stands in ``text`` — a reply, or an entry.

    THE ONE token match, for every claim that looks for a world's token.  The token stands
    there as written (typography folded, the comparison this has always been), or as the same
    word hyphenated another way.  Either reading is the token; neither makes a different word
    one."""
    return fold_typography(token) in fold_typography(text) or fold_for_token_match(
        token
    ) in fold_for_token_match(text)


_POSSESSIVE = "'s"
# A plural's possessive is its bare apostrophe: `Seals'`.  MEASURED: `the Harbor Seals' site`
# failed as `unsourced: ["Seals'"]` against a page that said "Seals".
_PLURAL_POSSESSIVE = "'"


def _bare(token: str) -> str:
    """A token without its possessive tail — ``Brandt's`` is the same name as ``Brandt``, and
    ``Seals'`` the same name as ``Seals``."""
    return _without_possessive(fold_typography(token))


def _without_possessive(folded: str) -> str:
    """An already-folded token without its possessive tail."""
    for tail in (_POSSESSIVE, _PLURAL_POSSESSIVE):
        if folded.endswith(tail):
            return folded[: -len(tail)]
    return folded


def _blank_field_labels(text: str) -> str:
    """Blank out each line's field label, each title line and each list number, so none is a
    value nor part of one."""
    return _FIELD_LABEL.sub(_blank, text)


def _blank(match: re.Match[str]) -> str:
    return " " * len(match.group())


def _blank_own_certainty(text: str) -> str:
    """Blank out a percentage that says how sure the speaker is."""
    return _CERTAINTY_FIGURE.sub(_blank, text)


def _values_only(text: str) -> str:
    """``text`` with everything that is layout or manner blanked in place, so what is left to
    read is what it states."""
    return _fold_phrases(_blank_own_certainty(_blank_field_labels(text)))


def _fold_phrases(text: str) -> str:
    """Blank out words a name phrase may not be built across."""
    return re.sub(
        rf"\b{_CAPITALISED}\b",
        lambda m: (
            " " * len(m.group())
            if _bare(m.group()).replace("'", "") in _NEVER_A_NAME
            else m.group()
        ),
        text,
    )


def _is_url(token: str) -> bool:
    """A URL is the one specific that can run into the prose around it: a number's word
    boundary and a name phrase's character class both stop before a mark on their own."""
    return "://" in token


def _is_atomic(token: str) -> bool:
    """Whether a match is one value rather than a phrase of them — a URL or a number."""
    return token[0].isdigit() or _is_url(token)


# Marks that end a SENTENCE rather than an address.  Every one of them is a legal URI character,
# so the grammar cannot rule on them and prose has to: nobody's address ends in a comma.
_SENTENCE_MARKS = ".,;:!?"
# Brackets and the apostrophe are legal in a path, so a trailing one is decided by PAIRING
# rather than by the grammar: `…/Foo_(bar)` closes something the address opened, `(…/page)`
# closes something the sentence opened.  A mark that is its own partner pairs by parity.
_PAIRS = {")": "(", "]": "[", "'": "'"}


def _closes_the_address(url: str, opener: str) -> bool:
    """Whether the URL's last character closes something the URL itself opened.  A mark that is
    its own partner has no opener to count, so it pairs by parity instead."""
    if opener == url[-1]:
        return url.count(opener) % 2 == 0
    return url.count(opener) >= url.count(url[-1])


def _runs_past_the_address(url: str) -> bool:
    """Whether the URL's last character belongs to the prose rather than to the address.

    ``_DROPPED`` is read here as well as in the fold: the fold erases it from BOTH sides of the
    comparison, so leaving it on the token hides the character underneath from these rules —
    `…/page.**` would otherwise keep the full stop the fold goes on to expose."""
    last = url[-1]
    if last in _SENTENCE_MARKS or last in _DROPPED:
        return True
    opener = _PAIRS.get(last)
    return opener is not None and not _closes_the_address(url, opener)


def _url_only(token: str) -> str:
    """The address without the prose it ran into.

    The repair lives HERE, on the extracted token, rather than in the fold, because the
    comparison is containment: extra characters on the haystack side are already harmless, and
    only the token can carry a mark that belongs to nobody.  So both sides still fold through
    the one definition, and neither folds through anything else."""
    while token and _runs_past_the_address(token):
        token = token[:-1]
    return token


# Markdown puts a link's text against its target with `](`, and every character of
# `[https://x/y](https://x/y)` is a legal URI character, so the grammar reads the whole link as
# ONE address: `https://x/y](https://x/y)`, a string no world contains, which reported a
# correctly cited source as an invention (#2152).  So a match is split at that seam before
# anything else is read off it, and a link yields each address it holds — its target, and its
# text when the text is an address too — each weighed against the world on its own, so an
# invented address in either position is still caught.
_LINK_SEAM = "]("


def _urls_in(token: str) -> list[str]:
    """Every address a URL match holds, each without the prose it ran into, without repeats.

    One for a bare address; one per side of a markdown link whose text is itself an address.
    A side that is not an address — a relative target — is not a URL specific, so it is not
    one here either."""
    addresses = [_url_only(part) for part in token.split(_LINK_SEAM) if _is_url(part)]
    return list(dict.fromkeys(addresses))


def specifics(text: str) -> list[str]:
    """Every specific value stated in ``text`` — URLs, numbers, and the WORDS of each
    capitalised name phrase — in the order they are said, without repeats.

    A phrase decides WHAT gets checked; its words are what is checked.  Measured, the whole
    phrase is too brittle to compare directly: a capitalised label sitting against a name
    (``Key⁠Ridgeline Foxes Sign Aurelio Brandt``, glued by a narrow no-break space) is not a
    string the world contains, though every name in it is.  A line's field label and a heading
    line's title are layout, not values, and are never read."""
    found: list[str] = []
    for value in _stated_values(text):
        found += [part for part in value if part not in found]
    return found


def _stated_values(text: str) -> list[list[str]]:
    """Each match of the specific-value grammar as the parts it is checked by: a URL match's
    addresses, a number alone, a name phrase's words — kept together, because whether a word is
    sourced can turn on the words it was said beside."""
    values: list[list[str]] = []
    for match in _SPECIFIC.finditer(_values_only(text)):
        token = match.group().strip()
        if _is_url(token):
            parts = _urls_in(token)
        else:
            parts = [token] if _is_atomic(token) else token.split()
        values.append([part for part in parts if part])
    return values


# ── Sourcing: the same value, as a WHOLE token ──
#
# A value is sourced when the world states THAT value, and a value has edges.  Finding its
# characters somewhere is not that: MEASURED, a bare `2` and a bare `1` were found inside
# `425F`, and a `5` inside `25 min`, so quantities nobody gave read as sourced.  The same
# containment sources `art` by `party`, and an address by any longer address it is a prefix of.
#
# So each kind `specifics` names is compared the way that kind has edges:
#
#   a NUMBER   by the numbers the world states.  A number there is a maximal run of digits and
#              the marks that join digits into one figure (`1,299` · `4.25` · `14:30`), so `2`
#              is no part of `425` or of `4.25`.  What stands beside the figure is not the
#              figure: a currency sign, a percent sign, a degree sign, a unit (`425F`, `5pm`).
#              And one figure written two ways is one figure: thousands separators, a trailing
#              `.00`, a leading zero.  A CLOCK TIME is one figure however it is told: an hour
#              with nothing after it is that hour (`7:00`, `07:00`, `7`), and the afternoon
#              hours have two numbers each (`6 PM`, `18:00`).  MEASURED: a job stored as
#              `BYHOUR=7` is "7:00 AM" in 8 of 15 replies describing it.
#   a NAME     by the world's words, as a whole word — a hyphenated one as that run of whole
#              words side by side.  A world often writes a name with no space in it (a domain,
#              a handle), so the words of one phrase run together are that name too: MEASURED,
#              a page given only as `harborseals.com` is "the Harbor Seals page" in 14 of 15
#              replies, and none of them invented a team.  A date written in numbers names
#              its month as well, so `2026-10-02` is given as "Oct 2".
#   a URL      by the addresses the world states, read off it with the grammar a reply's are
#              read with, as the WHOLE address.  How an address is reached is not what it
#              names, so its scheme, a leading `www.` and a closing slash are not compared.
#
#
# WHICH PART OF THE WORLD states each kind is not the same (#2203):
#
#   a NUMBER   by the round's CONTENT alone — what the user said, what a tool returned, what
#              the store held.  The framework's own numbers are not something anybody stated:
#              a prompt's numbered list, the count a read is headed by, the number an entry is
#              listed under.  MEASURED: an invented `2 tbsp` and `1 tsp` in a stored entry read
#              as sourced by a read result's `2 entries` and `1.`.
#   a MOMENT   a date or a clock time, by the round's timestamps as well: the date and time
#              it was told, the stamp on an entry, a run or a change.  A reply may say WHEN, so
#              `2026-10-02`, `October 2`, `20:11` and `8:11 PM` are each sourced by a stamp
#              saying so, and the stamp's digits source nothing else — the day of a date is not
#              a quantity.  A date or a time no stamp tells is read as the numbers it is
#              written in, against the content, as it always was.
#   a NAME and a URL  by everything the round was given, as before.  A prompt's own words are
#              words the round was handed, and a contract's tag turning up in an answer is a
#              copy (#2078).
#
# THE BLIND SPOTS, STATED.  A digit run inside a word with no hyphen (`a3f2b1`, `utf8`) is a
# number by this definition, so content carrying a hash sources the small numbers in it:
# telling an identifier from a figure with its unit (`425F`, `14T10:00`) needs a list of shapes,
# and a list of shapes is what this comparison is being repaired out of.  Content is taken as
# it comes, so a number a tool words into its own message (`Found 3 email(s)`), a date a page
# or an email carries, and a list the content numbers itself each state their digits.  A figure
# only a system prompt states — a job's terms in the self-state header, the values a collector
# is pointed at — is not sourced; the same terms reach content wherever a document or a read
# returns them.  And which half of the day an hour falls in is not weighed: `7 PM` is sourced
# by a world that says `7`, because a world says "7 in the evening" in more ways than a grammar
# can list.
_FIGURE = re.compile(r"(\d+(?:[.,:]\d+)*)(?:[ \t]?([ap])\.?m\b)?")
_CLOCK_TIME = re.compile(r"(\d{1,2}):(\d{2})")
_ON_THE_HOUR = "00"
_CLOCK_SEPARATOR = ":"
_AFTERNOON = "p"
_HALF_DAY = 12
_MONTHS_IN_A_YEAR = 12
_ISO_DATE_MONTH = re.compile(r"\b\d{4}-(\d{2})-\d{2}\b")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_LIST_SEPARATOR = ","
_DECIMAL_POINT = "."
_PLAIN_DECIMAL = re.compile(r"\d+(?:\.\d+)?")
_ZERO = "0"
# What a word is made of once folded.  The name grammar is ASCII, so the world's words are cut
# on the same alphabet: a reply's `Café` is read as `Caf`, and so is the world's.  A digit run
# is a token of its own so two words either side of a figure are not side by side.
_WORLD_TOKEN = re.compile(r"[a-z']+|\d+")
_QUOTE_MARK = "'"
_TOKEN_GAP = " "
_URL_PATTERN = re.compile(_URL)
_HOW_AN_ADDRESS_IS_REACHED = re.compile(r"^https?://(?:www\.)?")
_CLOSING_SLASH = "/"


def _figure_value(figure: str) -> str:
    """One figure in the form every rendering of it shares: `499.00`, `0499` and `499` are one
    amount.  A fraction that says something is kept as written — `2.10` may be a version, which
    `2.1` is not — and a figure that is not a plain decimal (a time, a dotted version) is its
    own text."""
    if not _PLAIN_DECIMAL.fullmatch(figure):
        return figure
    whole, _, fraction = figure.partition(_DECIMAL_POINT)
    whole = whole.lstrip(_ZERO) or _ZERO
    return f"{whole}{_DECIMAL_POINT}{fraction}" if fraction.strip(_ZERO) else whole


def _figure_readings(folded: str) -> list[frozenset[str]]:
    """Every number ``folded`` states, each as the set of forms its value takes.  A comma
    between digits is a thousands separator when exactly three digits follow it, and otherwise
    parts two numbers."""
    return [
        _readings(figure, meridiem)
        for run, meridiem in _FIGURE.findall(folded)
        for figure in _THOUSANDS_SEPARATOR.sub("", run).split(_LIST_SEPARATOR)
    ]


def _readings(figure: str, meridiem: str) -> frozenset[str]:
    """The forms one figure's value takes: itself, and for a time of day, every way of telling
    that time."""
    clock = _CLOCK_TIME.fullmatch(figure)
    hour, minutes = clock.groups() if clock else (figure, _ON_THE_HOUR)
    if not hour.isdigit() or not (clock or meridiem):
        return frozenset({_figure_value(figure)})
    return frozenset(
        told for hour_told in _hours_told(int(hour), meridiem) for told in _told(hour_told, minutes)
    )


def _hours_told(hour: int, meridiem: str) -> set[int]:
    """An hour as written, and the same hour on the other clock: `6 pm` is 18, `12 am` is 0."""
    if not meridiem or hour > _HALF_DAY:
        return {hour}
    return {hour, hour % _HALF_DAY + (_HALF_DAY if meridiem == _AFTERNOON else 0)}


def _told(hour: int, minutes: str) -> list[str]:
    """One time of day in its written forms — on the hour, the hour alone is one of them."""
    exact = f"{hour}{_CLOCK_SEPARATOR}{minutes}"
    return [exact, str(hour)] if minutes == _ON_THE_HOUR else [exact]


def _figures(folded: str) -> frozenset[str]:
    """Every form of every number ``folded`` states."""
    return frozenset(form for readings in _figure_readings(folded) for form in readings)


def _months_named(folded: str) -> list[str]:
    """The month each numeric date in ``folded`` names, in full and as it is abbreviated."""
    months = [int(month) for month in _ISO_DATE_MONTH.findall(folded)]
    return [
        name.casefold()
        for month in months
        if 1 <= month <= _MONTHS_IN_A_YEAR
        for name in (calendar.month_name[month], calendar.month_abbr[month])
    ]


def _whole_tokens(folded: str) -> list[str]:
    """``folded`` as whole tokens — each word without its possessive or the quotes around it."""
    bare = (_without_possessive(token).strip(_QUOTE_MARK) for token in _WORLD_TOKEN.findall(folded))
    return [token for token in bare if token]


def words_in(text: str) -> list[str]:
    """Every WORD ``text`` is written in, in order — typography and case folded the way every
    claim folds them, each word without its possessive or the quotes around it.  A figure is
    not a word: the numbers a text states are read by ``unsourced_specifics``."""
    return [token for token in _whole_tokens(fold_typography(text)) if not token.isdigit()]


def _side_by_side(tokens: Sequence[str]) -> str:
    """Tokens laid out so that containment of one layout in another is a whole-token match."""
    return f"{_TOKEN_GAP}{_TOKEN_GAP.join(tokens)}{_TOKEN_GAP}"


def _address(url: str) -> str:
    """What an address names, without how it is reached."""
    return _HOW_AN_ADDRESS_IS_REACHED.sub("", fold_typography(url)).rstrip(_CLOSING_SLASH)


# ── Moments: a date or a time the round was told ──
#
# Read off the TEXT with the marks the model draws them in — the drawn dashes of an ISO date,
# the narrow space before a meridiem — and compared with the round's timestamps as the
# datetimes they are.  A month is its name or its usual short form; an ordinal's letters and a
# stop after an abbreviation are how the day and the month were written, not part of either.
_DATE_HYPHEN = rf"[{re.escape(_DASHES)}-]"
_MONTH_NUMBERS = {
    name: number
    for number in range(1, _MONTHS_IN_A_YEAR + 1)
    for name in (calendar.month_name[number], calendar.month_abbr[number])
}
_MONTH_SAID = rf"\b(?P<month>{'|'.join(sorted(_MONTH_NUMBERS, key=len, reverse=True))})\b\.?"
_DAY_SAID = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\b"
_YEAR_SAID = r"(?P<year>\d{4})\b"
_THE_YEAR_AFTER = rf"(?:,?{_GAP}+{_YEAR_SAID})?"
_DATES_SAID = (
    re.compile(rf"\b{_YEAR_SAID}{_DATE_HYPHEN}(?P<month>\d{{2}}){_DATE_HYPHEN}(?P<day>\d{{2}})\b"),
    re.compile(rf"{_MONTH_SAID}{_GAP}+{_DAY_SAID}{_THE_YEAR_AFTER}"),
    re.compile(rf"\b{_DAY_SAID}{_GAP}+(?:of{_GAP}+)?{_MONTH_SAID}{_THE_YEAR_AFTER}"),
    re.compile(rf"{_MONTH_SAID},?{_GAP}+{_YEAR_SAID}"),
)
# A clock time stands on its own: `15:10` inside `15:10:00` is part of a longer figure.
_CLOCK_SAID = re.compile(
    rf"(?<![\d:.])(?P<hour>\d{{1,2}})(?::(?P<minutes>\d{{2}}))?\b(?!:\d)"
    rf"(?:{_GAP}?(?P<meridiem>[AaPp])\.?[Mm]\b)?"
)


def _tells_the_date(said: re.Match[str], moments: Sequence[datetime]) -> bool:
    """Whether a date written in the text is the date of a moment the round was told — on
    every part the text states, so `October 2` is told by any October 2nd."""
    parts = said.groupdict()
    month = parts["month"]
    number = int(month) if month.isdigit() else _MONTH_NUMBERS[month]
    day, year = parts.get("day"), parts.get("year")
    return any(
        moment.month == number
        and (day is None or moment.day == int(day))
        and (year is None or moment.year == int(year))
        for moment in moments
    )


def _tells_the_time(said: re.Match[str], moments: Sequence[datetime]) -> bool:
    """Whether a CLOCK TIME written in the text is the time of a moment the round was told.
    A bare number is not a clock time: it takes minutes or a meridiem to say one."""
    minutes, meridiem = said["minutes"], (said["meridiem"] or "").casefold()
    if minutes is None and not meridiem:
        return False
    figure = said["hour"] if minutes is None else f"{said['hour']}{_CLOCK_SEPARATOR}{minutes}"
    told = {form for moment in moments for form in _clock_forms(moment)}
    return bool(_readings(figure, meridiem) & told)


def _clock_forms(moment: datetime) -> list[str]:
    """One moment's time of day in its written forms, on either clock."""
    minutes = f"{moment.minute:02d}"
    hours = {moment.hour, moment.hour % _HALF_DAY or _HALF_DAY}
    return [form for hour in hours for form in _told(hour, minutes)]


def _inside_any(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in spans)


_Tells = Callable[[re.Match[str], Sequence[datetime]], bool]


def _without_told_moments(text: str, moments: Sequence[datetime]) -> str:
    """``text`` with every date and clock time a moment tells blanked in place, so what is left
    to read is what the content has to answer for.  Nothing inside an address is a date."""
    if not moments:
        return text
    addresses = [match.span() for match in _URL_PATTERN.finditer(text)]

    def blank_if(tells: _Tells) -> Callable[[re.Match[str]], str]:
        def blank(said: re.Match[str]) -> str:
            told = not _inside_any(said.span(), addresses) and tells(said, moments)
            return _blank(said) if told else said.group()

        return blank

    for date in _DATES_SAID:
        text = date.sub(blank_if(_tells_the_date), text)
    return _CLOCK_SAID.sub(blank_if(_tells_the_time), text)


@dataclass(frozen=True)
class _WorldValues:
    """Everything a round was given, read as the values it states."""

    addresses: frozenset[str]
    figures: frozenset[str]
    tokens: str
    moments: tuple[datetime, ...] = ()

    def unsourced(self, value: Sequence[str]) -> list[str]:
        """The parts of one stated value the world does not state."""
        if not value or _is_atomic(value[0]):
            return [part for part in value if not self._states(part)]
        return self._unsourced_words(value)

    def _states(self, atom: str) -> bool:
        if _is_url(atom):
            return _address(atom) in self.addresses
        readings = _figure_readings(fold_typography(atom))
        return all(forms & self.figures for forms in readings)

    def _unsourced_words(self, words: Sequence[str]) -> list[str]:
        """A name phrase's words that are neither a whole word of the world nor part of a run
        of the phrase the world writes as one word."""
        keys = [_whole_tokens(fold_typography(word)) for word in words]
        sourced = {index for index, key in enumerate(keys) if self._holds(key)}
        for first in range(len(keys)):
            for last in range(first + 1, len(keys)):
                run_together = "".join(token for key in keys[first : last + 1] for token in key)
                if self._holds([run_together]):
                    sourced.update(range(first, last + 1))
        return [word for index, word in enumerate(words) if index not in sourced]

    def _holds(self, tokens: Sequence[str]) -> bool:
        return _side_by_side(tokens) in self.tokens


def _world_values(given: str) -> _WorldValues:
    """What ``given`` states, kind by kind — a plain string as all content."""
    world = Given.read(given)
    return _read_world(str(world), world.content, world.moments)


@lru_cache(maxsize=8)
def _read_world(everything: str, content: str, moments: tuple[datetime, ...]) -> _WorldValues:
    """One world read once.  A cohort weighs every entry and reply of a sample against one
    world, so the reading is kept rather than repeated per value.  Keyed on all three parts: a
    ``Given`` compares equal to its own text, which says nothing about what in it is content."""
    folded = fold_typography(everything)
    return _WorldValues(
        addresses=frozenset(
            _address(url)
            for match in _URL_PATTERN.finditer(everything)
            for url in _urls_in(match.group())
        ),
        figures=_figures(fold_typography(content)),
        tokens=_side_by_side([*_whole_tokens(folded), *_months_named(folded)]),
        moments=moments,
    )


def unsourced_specifics(text: str, given: str) -> list[str]:
    """The specific values in ``text`` that ``given`` does not state.

    An empty list is the claim holding.  A value is sourced by the SAME value standing whole in
    the world, never by its characters turning up inside a larger one.  Matching folds
    typography and drops possessives on both sides, because a value is usually said in a
    different shape from the one it arrived in — comparing raw forms reported the model's own
    grammar as an invention.

    ``given`` is a :class:`Given` wherever a round was observed: its figures are weighed
    against the round's content, its dates and times against the round's moments as well, and
    its names and addresses against everything."""
    world = _world_values(given)
    missing: list[str] = []
    for value in _stated_values(_without_told_moments(text, world.moments)):
        missing += [part for part in world.unsourced(value) if part not in missing]
    return missing
