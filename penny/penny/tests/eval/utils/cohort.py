"""The cohort: one request, K phrasings, pooled — and the line between what a case ASSERTS
and what it MEASURES (#1994/#1995).

**Asserted** is the state the round LEFT BEHIND: where the machine landed, what the store
holds, that every specific value in the reply traces to something the model was given, and
that the reply's facts move when the world moves.  Deterministic reads with a pass-rate floor.

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

A **control** is not a cohort arm.  Phrasings are *same world, different words* and are pooled;
a control is *same words, different world* and is an ASSERTION.  It never enters the cohort
sizing, and the driver keeps it a separate cohort so nothing can quietly average the two.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field
from similarity.embeddings import cosine_similarity, token_containment_ratio

BASE_WORLD = "base"
CONTROL_WORLD = "control"

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
    world: str = BASE_WORLD
    complete: bool = True
    exclusion: str | None = None
    landed: str | None = None
    walk: str = ""
    routines: list[RoutineRecord] = Field(default_factory=list)
    entries: list[StoredEntry] = Field(default_factory=list)
    tool_sequence: list[str] = Field(default_factory=list)
    reply: str = ""
    reply_embedding: list[float] | None = None
    given: str = ""

    @property
    def stored_text(self) -> str:
        """Every entry this sample wrote, key and content together."""
        return " ".join(entry.text for entry in self.entries)


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
@dataclass(frozen=True)
class Feature:
    """One measured axis: a name, and how to read one sample's value for it.

    A string, because what is being measured is DISTINCTNESS — two samples agree when they
    produced the same value, and every feature answers that the same way whatever it is
    made of."""

    name: str
    read: Callable[[SampleObservation], str]


TOOL_SEQUENCE = Feature("tool sequence", lambda o: " → ".join(o.tool_sequence) or "no call")
ROUTINE_SHAPE = Feature(
    "routine shape", lambda o: " | ".join(r.shape for r in o.routines) or "no routine"
)
CONTAINER_NAME = Feature(
    "container name", lambda o: ", ".join(sorted({e.collection for e in o.entries})) or "none"
)
ENTRIES_STORED = Feature("entries stored", lambda o: str(len(o.entries)))
TRANSITIONS = Feature("transitions", lambda o: o.walk)

# Reply spread is pairwise rather than per-sample, so it is a marker the pooler recognises
# rather than a value any one sample carries.
REPLY_SPREAD = Feature("reply text", lambda o: o.reply)


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

    @property
    def modal_share(self) -> float:
        return self.modal / self.n if self.n else NO_SPREAD


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


class CohortVariance(BaseModel):
    """A case's whole measured half: what was pooled, what was thrown out, and the spread."""

    pooled: int = 0
    driven: int = 0
    # Samples this case drove in ANOTHER world — its control.  Counted rather than merely absent,
    # because "15 pooled of 18 driven · 0 excluded" is arithmetic that does not close, and a
    # section whose job is to make a run believable must not be the thing raising the question.
    control: int = 0
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
    )


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
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else NO_SPREAD


def pool(
    samples: Sequence[SampleObservation], features: Sequence[Feature], world: str = BASE_WORLD
) -> CohortVariance:
    """Gate for completeness, THEN pool — the order is the point.

    Nothing is measured over a sample that did not run, and what is excluded is NAMED rather
    than subtracted, so a run that lost half its cohort reads as one that lost half its cohort
    instead of as a suspiciously tidy one.

    Only samples from the case's OWN world are pooled.  A control drives the same ask against
    different facts to serve an ASSERTION, so folding its samples into the spread would report
    a deliberate difference as instability — the exclusion is structural here rather than a
    flag a case has to remember to set."""
    excluded = [
        ExcludedSample(name=s.name, reason=s.exclusion or "the measured turn never ran")
        for s in samples
        if not s.complete and s.world == world
    ]
    kept = [sample for sample in samples if sample.complete and sample.world == world]
    structural = [feature for feature in features if feature is not REPLY_SPREAD]
    return CohortVariance(
        pooled=len(kept),
        driven=len(samples),
        control=sum(1 for s in samples if s.world != world and s.complete),
        excluded=excluded,
        features=[feature_variance(feature, kept) for feature in structural],
        text=text_spread(kept) if REPLY_SPREAD in features else None,
    )


def proposed_ceiling(
    feature: VarianceFeature, model: str, margin: float = CEILING_MARGIN
) -> RecordedCeiling:
    """The ceiling this run PROPOSES — observed plus the sampling margin.  Proposed, never
    locked: a near-ceiling feature is a defect to fix first, not a threshold to record."""
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

# A claim read out of PROSE THE MODEL WROTE, as against one read out of the machine, the
# registry or the store.  The distinction is empirical, not editorial — see ``proposed_floor``.
REPLY_KIND = "reply"


class AssertionRow(BaseModel):
    """One claim's aggregate across the cohort — the section-A row.

    ``kind`` rides along because it decides whether a rate is LOCKABLE at all, and that is a
    property of where the claim is read FROM rather than of how well it did."""

    label: str
    passed: int
    total: int
    kind: str = "state"
    rationales: list[str] = Field(default_factory=list)

    @property
    def reads_model_prose(self) -> bool:
        return self.kind == REPLY_KIND

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else NO_SPREAD

    @property
    def at_full(self) -> bool:
        return self.total > 0 and self.passed == self.total


class ProposedFloor(BaseModel):
    """A claim's floor as it would be RECORDED, and whether it is worth recording.

    A claim that already holds on every sample proposes itself as its own floor.  One that does
    not is NOT proposed: the misses are naming work, and recording a floor underneath them
    would bless the defect as the contract."""

    label: str
    n: int
    value: float
    lockable: bool
    note: str = ""


def _missed_note(row: AssertionRow) -> str:
    return f"{row.total - row.passed} of {row.total} missed — read those first"


# Two runs of IDENTICAL code — same commit, same clean tree, same model, same upstream — moved a
# reply-content pass rate by 3 samples of 18 while every structural claim moved by at most 1.
# So the two halves have different noise floors, and ±3 of 18 is ±17 points: a floor tight
# enough to catch a real regression there would flap on a re-run, and one loose enough not to
# flap would catch nothing.  The rule is empirical and it is why ``kind`` is on the row.
UNLOCKABLE_AT_THIS_N = "reported, not floored at this N — it reads model prose"


def proposed_floor(row: AssertionRow) -> ProposedFloor:
    """The floor this run PROPOSES for one claim, and whether it may be locked at all.

    Two independent reasons a rate is not lockable, and the report must not blur them: the claim
    does not hold on every sample yet (the misses are the work), or it is read out of model prose
    at an N where the noise is wider than any useful floor."""
    if row.reads_model_prose:
        return ProposedFloor(
            label=row.label,
            n=row.total,
            value=round(row.pass_rate, 3),
            lockable=False,
            note=UNLOCKABLE_AT_THIS_N,
        )
    return ProposedFloor(
        label=row.label,
        n=row.total,
        value=round(row.pass_rate, 3),
        lockable=row.at_full,
        note="" if row.at_full else _missed_note(row),
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
    CONTROL = "control"
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
    samples: Sequence[SampleObservation],
    features: Sequence[Feature],
    world: str = BASE_WORLD,
) -> list[str]:
    """The features every pooled sample gave a different value — named so the report can say it
    ONCE instead of repeating it under every sample."""
    pooled = [s for s in samples if s.complete and s.world == world]
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
    samples: Sequence[SampleObservation],
    features: Sequence[Feature],
    world: str = BASE_WORLD,
) -> list[SampleStanding]:
    """Each sample's standing, in the order the samples were driven.

    Exactly ONE sample is modal — the first to carry the majority shape — because the workflow
    asks a human to read one, and naming eight equally would put the choice straight back on
    them.  Its shape-mates are typical and fold; everything that did something else is an
    outlier and opens.  A dead sample is neither: it has no shape to be typical of, and is the
    harness section's business rather than a reading recommendation."""
    pooled = [s for s in samples if s.complete and s.world == world]
    # Only the features that can distinguish one sample from another decide standing.  Including a
    # maximally-variant one makes every shape unique, so every sample becomes an outlier and the
    # modal sample is whichever happened to be first — the section names everything and therefore
    # nothing.
    telling = telling_features(pooled, features)
    shapes = Counter(sample_shape(s, telling) for s in pooled)
    modal_shape = shapes.most_common(1)[0][0] if shapes else ""
    # Divergence is measured against the REPRESENTATIVE sample rather than against each feature's
    # own mode, so the two halves of the report cannot disagree: the sample the reader is sent to
    # read has, by construction, nothing in its own divergence list.
    representative = next((s for s in pooled if sample_shape(s, telling) == modal_shape), None)
    seen_modal = False
    out: list[SampleStanding] = []
    for sample in samples:
        if sample.world != world or not sample.complete:
            out.append(
                SampleStanding(
                    name=sample.name,
                    phrasing=sample.phrasing,
                    # A CONTROL is not a dead sample and must never be labelled one: it ran its
                    # measured turn, and it is out of the cohort because it answers an ASSERTION
                    # against different facts, not because anything broke.  Calling it dead put
                    # the map in contradiction with the harness section on the same page —
                    # three samples "too broken to count" beside "0 excluded".
                    standing=Standing.DEAD if not sample.complete else Standing.CONTROL,
                    shape="",
                )
            )
            continue
        shape = sample_shape(sample, telling)
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
        FeatureDivergence(feature=feature.name, value=mine, modal=theirs)
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
# The cross-world form of each is already an assertion of its own — a reply naming the world it
# was not given fails DIRECTED CHANGE — so what is uncovered is a value belonging to NEITHER
# world, a narrower gap than the false-positive rate it buys off.
_NUMBER = r"\d[\d,.:%$]*"
_URL = r"https?://\S+"
_CAPITALISED = r"[A-Z][A-Za-z'-]*"
_NAME_PHRASE = rf"{_CAPITALISED}(?:\s+{_CAPITALISED})+"
_SPECIFIC = re.compile(rf"{_URL}|{_NAME_PHRASE}|\b{_NUMBER}\b")

# Words that carry a capital everywhere in English and are never part of a name, so a phrase is
# not built across them — otherwise a clause boundary glues two sentences into one "name".
_NEVER_A_NAME = frozenset({"i", "im", "ive", "ill", "id"})

# The model emits whichever apostrophe its tokenizer prefers; not folding them reported `I’ve`
# and `Brandt’s` as inventions.
_APOSTROPHES = "’‘`´"


def _fold(text: str) -> str:
    for mark in _APOSTROPHES:
        text = text.replace(mark, "'")
    return text.casefold()


def _bare(token: str) -> str:
    """A token without its possessive tail — ``Brandt's`` is the same name as ``Brandt``."""
    folded = _fold(token)
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
    haystack = " ".join(_bare(word) for word in _fold(given).split())
    return [token for token in specifics(text) if _phrase_key(token) not in haystack]


def _phrase_key(token: str) -> str:
    return " ".join(_bare(word) for word in token.split())
