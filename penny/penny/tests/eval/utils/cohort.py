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

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
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
    walk: str = ""
    routines: list[RoutineRecord] = Field(default_factory=list)
    entries: list[StoredEntry] = Field(default_factory=list)
    # Every entry the store HOLDS when the sample ends — the same ``StoredEntry`` shape as
    # ``entries``, because it is the same three facts about the same rows and two shapes for
    # one thing drift.  ``entries`` is the subset this round WROTE; this is everything left
    # standing, and it is the only one that can answer a claim about what a round left
    # ALONE: in a list of writes, an entry never touched and an entry deleted are both
    # simply absent.
    held: list[StoredEntry] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    reply: str = ""
    reply_embedding: list[float] | None = None
    # Every message the sample DELIVERED to the user, oldest first.  ``reply`` is only the
    # last one, and a claim about what reached the user has to read them all: a turn that
    # delivered two messages would otherwise be judged on one of them, which is how a
    # discarded draw arriving first went unseen.
    delivered: list[str] = Field(default_factory=list)
    given: str = ""
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
    # ``entries`` above is what THIS SAMPLE WROTE, which is the right reading for "did she
    # invent that" and the wrong one for "what does the collection hold": a cycle that
    # correctly wrote nothing leaves it empty while the store still holds the value it was
    # seeded with.  Both questions are real and they are not the same question, so the store's
    # own end state is carried beside the sample's writes rather than derived from them.
    held: dict[str, str] = Field(default_factory=dict)
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
TRANSITIONS = Feature("transitions", lambda o: o.walk, absent="no move")

# Reply spread is pairwise rather than per-sample, so it is a marker the pooler recognises
# rather than a value any one sample carries.
REPLY_SPREAD = Feature("reply text", lambda o: o.reply, consequence=Consequence.COSMETIC)


def output_field(name: str, *, consequence: Consequence = Consequence.CONSEQUENTIAL) -> Feature:
    """One field of a draw's STRUCTURED OUTPUT, as a measured axis.

    A single-call context returns a typed result rather than leaving a trail through the
    stores, so its variance axes are simply its own fields: the same values the case asserts,
    compared across the cohort.  No new concept — this is :class:`Feature` reading
    ``SampleObservation.output`` instead of a chat-shaped attribute.

    ``absent`` is :data:`FIELD_UNSET`, so a field the draw never returned reads as blind
    rather than as fifteen samples agreeing."""
    return Feature(name, lambda o: o.field(name), consequence=consequence, absent=FIELD_UNSET)


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


class CohortVariance(BaseModel):
    """A case's whole measured half: what was pooled, what was thrown out, and the spread."""

    pooled: int = 0
    driven: int = 0
    excluded: list[ExcludedSample] = Field(default_factory=list)
    features: list[VarianceFeature] = Field(default_factory=list)
    text: TextSpread | None = None

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


def pool(samples: Sequence[SampleObservation], features: Sequence[Feature]) -> CohortVariance:
    """Gate for completeness, THEN pool — the order is the point.

    Nothing is measured over a sample that did not run, and what is excluded is NAMED rather
    than subtracted, so a run that lost half its cohort reads as one that lost half its cohort
    instead of as a suspiciously tidy one."""
    excluded = [
        ExcludedSample(name=s.name, reason=s.exclusion or "the measured turn never ran")
        for s in samples
        if not s.complete
    ]
    kept = [sample for sample in samples if sample.complete]
    structural = [feature for feature in features if feature is not REPLY_SPREAD]
    return CohortVariance(
        pooled=len(kept),
        driven=len(samples),
        excluded=excluded,
        features=[feature_variance(feature, kept) for feature in structural],
        text=text_spread(kept) if REPLY_SPREAD in features else None,
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

    Distinct from ``kind`` (``state`` / ``reply`` / ``spine`` / ``proc``), which is a
    render-and-gating class: ``kind`` decides how a claim renders and whether it can carry a
    floor, ``category`` says which part of the design it satisfies.  Neither is derivable from
    the other — PROVENANCE has both a gated store-side claim and an ungated reply-side one."""

    LANDED = "landed"
    STORE = "store"
    PROVENANCE = "provenance"


# A claim read out of PROSE THE MODEL WROTE, as against one read out of the machine, the
# registry or the store.  It decides how a claim RENDERS and how its per-sample check is
# anchored; it does not decide anything about gating, because assertions are not gated.
REPLY_KIND = "reply"


class AssertionRow(BaseModel):
    """One claim's aggregate across the cohort — the section-A row.

    ``kind`` rides along because it says where the claim was read FROM, which is what anchors
    its per-sample check; it no longer sorts claims into gated and ungated, because nothing on
    this side is gated."""

    label: str
    passed: int
    total: int
    category: SpecCategory
    kind: str = "state"
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
# A URL runs to the first whitespace, MINUS any sentence mark it ran into: `\S+` alone captured
# the full stop closing the sentence, and `…/aurora-deck-2.` matches no world.  Only `.,;:!?` are
# refused as the LAST character, so a url legitimately ending in a bracket, a slash or a dash —
# `…/Foo_(bar)`, `…/news/` — keeps it.
_URL = r"https?://\S*[^\s.,;:!?]"
_CAPITALISED = r"[A-Z][A-Za-z'-]*"
_NAME_PHRASE = rf"{_CAPITALISED}(?:\s+{_CAPITALISED})+"
_SPECIFIC = re.compile(rf"{_URL}|{_NAME_PHRASE}|\b{_NUMBER}\b")

# Words that carry a capital everywhere in English and are never part of a name, so a phrase is
# not built across them — otherwise a clause boundary glues two sentences into one "name".
_NEVER_A_NAME = frozenset({"i", "im", "ive", "ill", "id"})

# ONE folding, used by every probe on both sides of every comparison.  A semantic check defeated
# by cosmetics is a scorer bug, and two spellings of "fold the typography" drift apart: measured,
# a reply citing the URL it was given — with U+2011 non-breaking hyphens for its dashes — read as
# an INVENTION here while the claim next door folded that dash and agreed the value was sourced.
#
# The model emits whichever apostrophe its tokenizer prefers; not folding them reported `I’ve`
# and `Brandt’s` as inventions.
_APOSTROPHES = "’‘`´"
# Dashes it draws instead of a hyphen — in a URL, in a name, anywhere.
_DASHES = "‐‑‒–—−"
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


def _bare(token: str) -> str:
    """A token without its possessive tail — ``Brandt's`` is the same name as ``Brandt``."""
    folded = fold_typography(token)
    return folded[:-2] if folded.endswith("'s") else folded


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


def _is_atomic(token: str) -> bool:
    """Whether a match is one value rather than a phrase of them — a URL or a number."""
    return token[0].isdigit() or "://" in token


def specifics(text: str) -> list[str]:
    """Every specific value stated in ``text`` — URLs, numbers, and the WORDS of each
    capitalised name phrase — in the order they are said, without repeats.

    A phrase decides WHAT gets checked; its words are what is checked.  Measured, the whole
    phrase is too brittle to compare directly: a capitalised label sitting against a name
    (``Key⁠Ridgeline Foxes Sign Aurelio Brandt``, glued by a narrow no-break space) is not a
    string the world contains, though every name in it is."""
    found: list[str] = []
    for match in _SPECIFIC.finditer(_fold_phrases(text)):
        token = match.group().strip()
        parts = [token] if _is_atomic(token) else token.split()
        found += [part for part in parts if part and part not in found]
    return found


def unsourced_specifics(text: str, given: str) -> list[str]:
    """The specific values in ``text`` that appear NOWHERE in ``given``.

    An empty list is the claim holding.  Matching folds apostrophes and drops possessives on
    both sides, because a value is usually said in a different shape from the one it arrived
    in — comparing raw forms reported the model's own grammar as an invention."""
    haystack = " ".join(_bare(word) for word in fold_typography(given).split())
    return [token for token in specifics(text) if _phrase_key(token) not in haystack]


def _phrase_key(token: str) -> str:
    return " ".join(_bare(word) for word in token.split())
