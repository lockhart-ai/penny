"""The cohort: one request, K phrasings, pooled — and what is ASSERTED about it versus
what is MEASURED (#1994).

A case makes two different kinds of claim, and conflating them is what made the suite
simultaneously too strict and too loose.  This module is the line between them.

**Asserted** is the state the round LEFT BEHIND: where the machine landed, what the store
holds, that every specific value in the reply traces to something the model was given, and
that the reply's facts move when the world moves.  Those are deterministic reads with a
pass-rate floor, and they live with the case (a ``Check`` each) rather than here.

**Measured** is everything the model CHOSE: which tools it called and in what order, the
shape of the routine it recorded, the names it picked, the words it replied with.  Many
routes reach one end state, so a route is never asserted — it is scored as spread across
the cohort and carries a one-sided ceiling, where only an INCREASE is a regression.

**The cohort is the unit of measurement.**  N samples of ONE request, expressed as K
paraphrases, generated concurrently and weighed against each other — never against a stored
baseline, so there is nothing to drift and nothing to re-baseline on a model swap.  A
different *scenario* is a different case; input variation inside a case is paraphrase of the
same request, so the assertions stay constant and any shape change is a real finding.

Two properties of the statistic decide how it may be read, both measured rather than
assumed:

* **Normalised entropy is BIASED UPWARD at small N** — the same behaviour reads 0.527 at
  N=32 and 0.605 at N=15, because the ``log(N)`` denominator shrinks faster than the
  observed spread does.  So a recorded ceiling is meaningless without the N beside it, and
  :class:`RecordedCeiling` carries one: comparing across cohort sizes is refused rather
  than merely discouraged (:func:`compare_to_ceiling`).
* **Phrasing contributes almost nothing to the spread** (~0.05 of it; model stochasticity
  carries the rest — gpt-oss produced 4.8 distinct routine shapes inside a SINGLE phrasing
  and pooling four only reached 9).  That is what justifies pooling: the pooled number
  estimates the underlying variance rather than blending K different things.

  But phrasings are a **coverage** mechanism, not a variance mechanism, and the pooled
  number hides what they are for.  Measured on a consistent model, four phrasings scored
  H = 0.00, 0.52, 0.00, 0.00 — three perfectly consistent, one that destabilised the case
  completely — and pooled that reads 0.18.  So every feature also carries a
  :class:`PhrasingRow` per phrasing.  At 3 samples each there is no reliable per-phrasing
  entropy and none is reported: what those rows carry is the weaker honest signal — how
  many distinct values this wording produced, and whether it produced one **no other
  wording did** — which is a flag for a human to look at, never a gate.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, Field
from similarity.embeddings import cosine_similarity, token_containment_ratio

# The default world an arm reads its facts from.  A case that perturbs the world names its
# other worlds itself; this is the name of the one it started in.
BASE_WORLD = "base"

# The margin a proposed ceiling sits above the observed value.  Sized from the sampling
# noise measured by subsampling real 32-sample cohorts down to 15: the spread on normalised
# entropy is ~±0.11 there, so a ceiling ON the observed value would flap on ordinary
# re-runs while one this far above it still separates a variant cohort from a consistent
# one (at N=15 those two overlap 0.2%).  It is a property of the COHORT SIZE, which is why
# it is spent here rather than guessed per case.
CEILING_MARGIN = 0.10

# What a cohort whose samples all agree scores.  Named because "0.0 means no spread" is the
# one reading of this statistic that has to survive every refactor of how it is computed.
NO_SPREAD = 0.0


class CohortArm(BaseModel):
    """One way of asking the SAME request, and the world it is asked against.

    A case declares its arms as data — the phrasings that pool into one cohort, plus any
    arm that perturbs the world.  Everything that varies per arm is here, so what a reader
    has to compare when a case is copied is one list of arms rather than a spread of
    conditionals through a scorer.

    ``pooled`` is the line between the two jobs an arm can have.  A PHRASING pools: it is
    another sample of the same request, and the recorded ceiling is measured over exactly
    these.  A perturbed-world arm does NOT pool — it exists to make a DIRECTED-CHANGE
    assertion (move the world, the reply's facts must move with it), and folding its
    samples into the spread would report a deliberate difference as instability.
    """

    label: str
    message: str
    samples: int
    world: str = BASE_WORLD
    pooled: bool = True


def expand_arms(arms: Sequence[CohortArm]) -> list[CohortArm]:
    """The arm each sample index runs, in sample order — one entry per sample.

    Explicit expansion rather than modular arithmetic at the call site: a case declares how
    many samples each arm is worth, and the mapping from sample index to arm is then a list
    lookup that a reader can check by counting."""
    expanded: list[CohortArm] = []
    for arm in arms:
        expanded += [arm] * arm.samples
    return expanded


class SampleFacts(BaseModel):
    """What ONE sample left behind, read at scoring time while its database is live.

    ``features`` is the sample's value for each measured feature, keyed by feature name and
    rendered as a string — the tool sequence it called, the shape of the routine it
    recorded, the name it chose.  Strings because what is being measured is *distinctness*:
    two samples agree when they produced the same value, and every feature answers that the
    same way whatever it is made of.  Insertion order is the order features render in.

    ``complete`` is the COMPLETENESS gate, and it is read BEFORE anything is pooled.  A
    sample's database exists from sample START, not completion, so counting files as samples
    reports dead samples as behavioural variance — 17 of 31 in the first prototype run.
    ``exclusion`` says why in the report, by name.
    """

    name: str
    arm: str
    world: str = BASE_WORLD
    pooled: bool = True
    complete: bool = True
    exclusion: str | None = None
    features: dict[str, str] = Field(default_factory=dict)
    reply: str = ""
    reply_embedding: list[float] | None = None


class ExcludedSample(BaseModel):
    """One sample too broken to count, named — never a silent drop."""

    name: str
    reason: str


class PhrasingRow(BaseModel):
    """One phrasing's own view of a feature — DIAGNOSTIC, and never locked.

    At 3 samples a per-phrasing entropy would be noise wearing a number's clothes, so none
    is reported.  What is here is what 3 samples can honestly say: how many distinct values
    this wording produced, and whether any of them appeared under NO other wording.  That
    second one is the finding the pooled number conceals — a case can be perfectly stable
    for three wordings and come apart for the fourth."""

    arm: str
    n: int
    distinct: int
    values: list[str]
    only_here: list[str] = Field(default_factory=list)

    @property
    def flagged(self) -> bool:
        """Whether this wording is worth a human's attention: it produced a value no other
        wording produced, or it disagreed with itself while others did not."""
        return bool(self.only_here)


class VarianceFeature(BaseModel):
    """One measured feature's spread across the POOLED cohort, plus its per-phrasing rows.

    ``entropy`` is Shannon entropy over the value distribution normalised by ``log(n)`` —
    the spread a cohort of this size could at most show — so 0.0 is total agreement and 1.0
    is every sample distinct.  ``n`` rides along because that denominator makes the number
    incomparable across cohort sizes."""

    name: str
    n: int
    distinct: int
    modal: int
    entropy: float
    phrasings: list[PhrasingRow] = Field(default_factory=list)

    @property
    def modal_share(self) -> float:
        """The fraction of the cohort that agreed on the most common value."""
        return self.modal / self.n if self.n else NO_SPREAD


class TextSpread(BaseModel):
    """How far the cohort's REPLIES stand apart, measured with the shared similarity
    primitives rather than with a phrasing list somebody guessed.

    Pairwise over the pooled cohort: ``cosine_similarity`` over the embeddings the replies
    already carry (every outgoing message is embedded at egress, so this costs no model
    call), and ``token_containment_ratio`` over the words themselves.  Two views because
    they fail differently — an embedding says two replies are ABOUT the same thing, which
    at fixed topic is nearly always true, while containment says how much of one reply's
    vocabulary the other actually reuses."""

    pairs: int
    cosine_mean: float
    cosine_min: float
    containment_mean: float


class CohortVariance(BaseModel):
    """A case's whole measured half: what was pooled, what was thrown out, and the spread.

    The two counts are stated separately and always, because "15 samples" and "15 samples
    that ran" are different claims and a report that shows only the second cannot be
    checked."""

    pooled: int = 0
    driven: int = 0
    excluded: list[ExcludedSample] = Field(default_factory=list)
    features: list[VarianceFeature] = Field(default_factory=list)
    text: TextSpread | None = None


class RecordedCeiling(BaseModel):
    """A feature's ceiling as it would be RECORDED — ``(feature, model, N, value)``.

    Neither qualifier is decoration, and both make a comparison across them REFUSED rather
    than merely discouraged (:func:`compare_to_ceiling`):

    * **N** — normalised entropy is biased upward at small N (0.527 at N=32 reads 0.605 at
      N=15 for the same behaviour), so a ceiling recorded at one cohort size says nothing
      about a run at another.
    * **MODEL** — measured, two models differ ~3x on the same features (routine shape 0.53
      against 0.18, tool sequence 0.69 against 0.19), so one shared ceiling would be useless
      for the consistent model and permanently failing for the variant one."""

    feature: str
    model: str
    n: int
    value: float


class CeilingVerdict(BaseModel):
    """The one-sided regression check's answer.

    One-sided by construction: only an INCREASE is a regression, so a cohort that got more
    consistent never fails.  ``comparable`` is False when the cohorts are different sizes,
    and then ``regressed`` says nothing at all — an incomparable pair is reported as such,
    never silently resolved one way."""

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
    cohort COULD have shown it actually did: n samples all distinct is the maximum, and
    dividing by the observed distinct count would score "two values, evenly split" the same
    as "fifteen values, evenly split".

    Returns 0.0 for a cohort of one, where there is no spread to normalise against."""
    if len(values) < 2:
        return NO_SPREAD
    counts = Counter(values)
    total = len(values)
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
    # A single-valued cohort computes to NEGATIVE zero, which renders as ``-0.000`` and reads
    # as a number rather than as the absence of one.  Total agreement is the reading this
    # statistic has to get right above all others, so it is clamped rather than formatted
    # around at each render.
    return max(NO_SPREAD, entropy / math.log(total))


def _phrasing_rows(name: str, samples: Sequence[SampleFacts]) -> list[PhrasingRow]:
    """One row per phrasing for a feature, in arm order, each flagged against the others."""
    by_arm: dict[str, list[str]] = {}
    for sample in samples:
        by_arm.setdefault(sample.arm, []).append(sample.features.get(name, ""))
    # "Only under this wording" is a comparison BETWEEN wordings, so a cohort carrying one
    # phrasing has nothing to say — every value would be trivially unique to the only arm
    # there is, and flagging it would report the shape of the cohort as a finding.
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


def _values_under_other_arms(by_arm: dict[str, list[str]], arm: str) -> set[str]:
    """Every value any OTHER phrasing produced — what makes a value 'only here'."""
    return {value for other, values in by_arm.items() if other != arm for value in values}


def feature_variance(name: str, samples: Sequence[SampleFacts]) -> VarianceFeature:
    """One feature's pooled spread plus its per-phrasing diagnostic rows."""
    values = [sample.features.get(name, "") for sample in samples]
    counts = Counter(values)
    return VarianceFeature(
        name=name,
        n=len(values),
        distinct=len(counts),
        modal=max(counts.values()) if counts else 0,
        entropy=normalised_entropy(values),
        phrasings=_phrasing_rows(name, samples),
    )


def text_spread(samples: Sequence[SampleFacts]) -> TextSpread | None:
    """Pairwise reply spread over the pooled cohort, or ``None`` below two replies.

    A sample whose reply carries no embedding contributes to containment and not to cosine
    — the two are reported over the pairs each could actually be computed on, rather than
    dropping a reply from both because one half of it is missing."""
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


def feature_names(samples: Sequence[SampleFacts]) -> list[str]:
    """Every feature any sample carries, in first-seen order — the order they render in."""
    names: list[str] = []
    for sample in samples:
        names += [name for name in sample.features if name not in names]
    return names


def pool(samples: Sequence[SampleFacts]) -> CohortVariance:
    """Gate for completeness, THEN pool — the summary method, and the order is the point.

    Nothing is measured over a sample that did not run.  What is excluded is named rather
    than subtracted, so a run that lost half its cohort reads as a run that lost half its
    cohort instead of as a suspiciously tidy one."""
    excluded = [
        ExcludedSample(name=sample.name, reason=sample.exclusion or "the measured turn never ran")
        for sample in samples
        if not sample.complete
    ]
    kept = [sample for sample in samples if sample.complete and sample.pooled]
    return CohortVariance(
        pooled=len(kept),
        driven=len(samples),
        excluded=excluded,
        features=[feature_variance(name, kept) for name in feature_names(kept)],
        text=text_spread(kept),
    )


def proposed_ceiling(
    feature: VarianceFeature, model: str, margin: float = CEILING_MARGIN
) -> RecordedCeiling:
    """The ceiling this run PROPOSES for a feature — observed plus the sampling margin.

    Proposed, never locked: what a case's thresholds are is the code owner's to accept once
    the numbers have been read.  A near-ceiling feature is not worth recording at all — it
    is a defect to fix first — but saying so is a reading of the number, not a rule this
    can apply."""
    return RecordedCeiling(
        feature=feature.name,
        model=model,
        n=feature.n,
        value=round(min(feature.entropy + margin, 1.0), 3),
    )


def compare_to_ceiling(
    ceiling: RecordedCeiling, observed: VarianceFeature, model: str
) -> CeilingVerdict:
    """The one-sided regression check: did this feature's spread rise above its ceiling?

    A different cohort SIZE or a different MODEL is INCOMPARABLE and says so, rather than
    being answered wrongly: the normalisation denominator makes two sizes different
    statistics, and two models differ several-fold on the same feature."""
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


# ── Cost: what one sample of this case spends ────────────────────────────────
#
# The cost/performance metric, and the ported case reports it (#1994 §4a).  Nothing new is
# captured — the harness already records calls, wall time and input/output/reasoning tokens
# per prompt — what was missing is that nobody reported or compared them, so a prompt change
# that doubles the context stayed invisible until someone read the bill.
#
# PER SAMPLE, never per run: a total is not comparable across cohort sizes, exactly the trap
# entropy's ``log(N)`` denominator is.
#
# INPUT and OUTPUT are split because they mean different things.  Input is OURS — prompt and
# context design — so a rise means we made the envelope bigger, and that is what a prompt edit
# regresses.  Output is the MODEL's: on a fixed prompt, a rise is a model or config change.
# Measured on 48-sample cohorts over identical fixtures, input was 39,633 against 39,818 —
# identical, as it should be, since the prompts were the same — while output was 5,234 against
# 13,177, almost all of it reasoning.  That 2.5x is what buys the second model its far better
# consistency, and since Penny's target is local hardware it is the trade that decides a model.

# How far above the observed per-sample cost a proposed ceiling sits.  UNLIKE
# :data:`CEILING_MARGIN` this one is NOT measured — it is a round band, stated as such, chosen
# because a mean of large counts is far steadier than a distribution statistic and because the
# number is PROPOSED for the code owner to accept rather than locked here.
COST_CEILING_MARGIN = 0.10


class SampleCost(BaseModel):
    """What ONE sample of a case spends, averaged over the samples that ran.

    ``reasoning`` is the part of ``output`` the provider attributes to thinking, 0 where a
    backend reports none — carried because two models scoring alike while one generates far
    more thinking tokens is a local-hardware regression that no score would show."""

    samples: int
    calls: float
    seconds: float
    input_tokens: float
    output_tokens: float
    reasoning_tokens: float

    @property
    def reasoning_share(self) -> float:
        """The fraction of output spent thinking — 0.0 when nothing was generated."""
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


# ── The case's own three-section report ──────────────────────────────────────
#
# Rendered here, beside the models it renders, so a case that is ported against this
# pattern has ONE file to read: what an arm is, what is asserted, what is measured, and
# what the reader will see.  It is written into the case's own ``<case_id>.md`` above its
# sample transcripts and passes through the run comment verbatim — one rendering, on disk
# and in the comment, rather than two that can disagree.


class AssertionRow(BaseModel):
    """One deterministic assertion's aggregate across the cohort — the section-A row.

    ``passed``/``total`` are the counts the harness already folds per check label, so this
    is a projection rather than a second tally.  An assertion is about END STATE and about
    nothing else: a route reaching that state is measured, never asserted."""

    label: str
    passed: int
    total: int
    kind: str = ""

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else NO_SPREAD

    @property
    def at_full(self) -> bool:
        """Whether every sample that this assertion applied to met it."""
        return self.total > 0 and self.passed == self.total


class ProposedFloor(BaseModel):
    """An assertion's floor as it would be RECORDED, and whether it is worth recording.

    An assertion that already holds on every sample proposes itself as its own floor.  One
    that does not is NOT proposed: the misses are naming work, and recording a floor
    underneath them would bless the defect as the contract."""

    label: str
    n: int
    value: float
    lockable: bool
    note: str = ""


def _missed_note(row: AssertionRow) -> str:
    """Why an assertion is not proposing a floor — the misses, counted."""
    return f"{row.total - row.passed} of {row.total} missed — read those first"


def proposed_floor(row: AssertionRow) -> ProposedFloor:
    """The floor this run proposes for one assertion."""
    return ProposedFloor(
        label=row.label,
        n=row.total,
        value=round(row.pass_rate, 3),
        lockable=row.at_full,
        note="" if row.at_full else _missed_note(row),
    )


class CaseReport(BaseModel):
    """One case's whole result: what was asserted, what was measured, what was thrown out.

    The three sections are three different KINDS of claim and are never mixed.  A reader
    who wants to know whether Penny is correct reads A; one who wants to know whether she
    is stable reads B; one who wants to know whether the run can be believed at all reads
    C — and C is read FIRST, because a cohort that lost half its samples makes the other
    two sections a description of whatever survived."""

    case_id: str
    model: str = ""
    assertions: list[AssertionRow] = Field(default_factory=list)
    variance: CohortVariance = Field(default_factory=CohortVariance)
    cost: SampleCost | None = None

    def render(self) -> str:
        """The summary method: the three sections, in the order they are read."""
        return "\n\n".join(
            [
                f"#### `{self.case_id}` — end-state assertions, variance, harness",
                self._assertions_section(),
                self._variance_section(),
                self._harness_section(),
            ]
        )

    def _assertions_section(self) -> str:
        """A. what the round LEFT BEHIND — deterministic, with a proposed pass-rate floor."""
        if not self.assertions:
            return f"{_SECTION_A}\n\n_(no assertions)_"
        rows = "\n".join(_assertion_row(row) for row in self.assertions)
        return f"{_SECTION_A}\n\n{_ASSERTION_HEAD}\n{rows}"

    def _variance_section(self) -> str:
        """B. what the model CHOSE — pooled spread, a one-sided ceiling, per-phrasing rows."""
        if not self.variance.features:
            return f"{_SECTION_B}\n\n_(nothing pooled — see the harness section)_"
        rows = "\n".join(_variance_row(feature, self.model) for feature in self.variance.features)
        note = _CEILING_NOTE.format(n=self.variance.pooled)
        parts = [f"{_SECTION_B}\n\n{_VARIANCE_HEAD}\n{rows}", note]
        phrasings = self._phrasing_block()
        if phrasings:
            parts.append(phrasings)
        if self.variance.text is not None:
            parts.append(_text_line(self.variance.text))
        if self.cost is not None:
            parts.append(_cost_block(self.cost, self.model))
        return "\n\n".join(parts)

    def _phrasing_block(self) -> str:
        """The per-phrasing diagnostic rows — unlocked, and flagged where a wording produced
        a value no other wording did.  Only the flagged rows render: an unflagged phrasing
        agreeing with its neighbours is the ordinary case and says nothing a reader needs."""
        flagged = [
            (feature.name, row)
            for feature in self.variance.features
            for row in feature.phrasings
            if row.flagged
        ]
        if not flagged:
            return _NO_PHRASING_OUTLIERS
        rows = "\n".join(
            f"| `{name}` | {row.arm} | {row.distinct}/{row.n} | {_values(row.only_here)} |"
            for name, row in flagged
        )
        return f"{_PHRASING_LEAD}\n\n{_PHRASING_HEAD}\n{rows}"

    def _harness_section(self) -> str:
        """C. what was too broken to count — gated BEFORE anything above was pooled."""
        excluded = self.variance.excluded
        counts = (
            f"{self.variance.pooled} pooled of {self.variance.driven} driven · "
            f"{len(excluded)} excluded"
        )
        if not excluded:
            return f"{_SECTION_C}\n\n{counts} — every sample ran its measured turn."
        named = "\n".join(f"- `{sample.name}` — {sample.reason}" for sample in excluded)
        return f"{_SECTION_C}\n\n{counts}\n\n{named}"


_SECTION_A = "**A. Deterministic assertions — end state only.**"
_SECTION_B = "**B. Variance — model output.**"
_SECTION_C = "**C. Harness — samples too broken to count.**"

_ASSERTION_HEAD = "| assertion | held | rate | proposed floor |\n|---|---|---|---|"
_VARIANCE_HEAD = (
    "| feature | distinct | modal | entropy | proposed ceiling |\n|---|---|---|---|---|"
)
_PHRASING_HEAD = "| feature | phrasing | distinct | only under this wording |\n|---|---|---|---|"
_PHRASING_LEAD = (
    "_Per-phrasing rows below are DIAGNOSTIC and never locked — at 3 samples each there is no "
    "reliable per-phrasing entropy, so what is reported is the honest weaker signal: a wording "
    "that produced a value no other wording did. Phrasings are a coverage mechanism, and the "
    "pooled number above hides exactly what they are for._"
)
_NO_PHRASING_OUTLIERS = "_No phrasing produced a value the others did not._"
_COST_LEAD = "**Cost, per sample.**"
_COST_HEAD = "| tokens | observed | proposed ceiling |\n|---|---|---|"
_COST_NOTE = (
    "_Per SAMPLE, never per run — a total is not comparable across cohort sizes. Input is OURS "
    "(prompt and context design), so a rise is what a prompt edit regresses; output is the "
    "MODEL's, so a rise on a fixed prompt is a model or config change. Both ceilings are "
    "one-sided, per model, and PROPOSED — and unlike the variance margin this band is a round "
    "number rather than a measured one._"
)
_CEILING_NOTE = (
    "_Ceilings are PROPOSED, not locked, and are one-sided — only a rise is a regression. "
    "Each is recorded as `(feature, model, N={n}, value)` and a comparison across either "
    "qualifier is REFUSED: normalised entropy is biased upward at small N (the same behaviour "
    "reads 0.527 at N=32 and 0.605 at N=15), and two models differ ~3x on the same feature "
    "(routine shape 0.53 against 0.18), so a shared ceiling would measure neither._"
)


def _assertion_row(row: AssertionRow) -> str:
    floor = proposed_floor(row)
    proposal = f"`{floor.value:.2f}`" if floor.lockable else f"— {floor.note}"
    kind = f" _[{row.kind}]_" if row.kind else ""
    return f"| {row.label}{kind} | {row.passed}/{row.total} | {row.pass_rate:.2f} | {proposal} |"


def _variance_row(feature: VarianceFeature, model: str) -> str:
    ceiling = proposed_ceiling(feature, model)
    return (
        f"| `{feature.name}` | {feature.distinct} | {feature.modal}/{feature.n} "
        f"({feature.modal_share:.2f}) | {feature.entropy:.3f} | "
        f"`{ceiling.value:.2f}` @ {ceiling.model} N={ceiling.n} |"
    )


def _cost_block(cost: SampleCost, model: str) -> str:
    """What one sample spends, with a proposed one-sided ceiling on each half."""
    rows = "\n".join(
        f"| {label} | {observed:,.0f} | `{observed * (1 + COST_CEILING_MARGIN):,.0f}` @ {model} |"
        for label, observed in (
            ("input tokens (ours — prompt and context)", cost.input_tokens),
            ("output tokens (the model's)", cost.output_tokens),
        )
    )
    tail = (
        f"Also per sample: {cost.calls:,.1f} calls · {cost.seconds:,.0f}s · "
        f"{cost.reasoning_tokens:,.0f} reasoning tokens "
        f"({cost.reasoning_share:.0%} of output)."
    )
    return f"{_COST_LEAD}\n\n{_COST_HEAD}\n{rows}\n\n{tail}\n\n{_COST_NOTE}"


def _text_line(text: TextSpread) -> str:
    return (
        f"Reply text over {text.pairs} pairs — cosine mean {text.cosine_mean:.3f} "
        f"min {text.cosine_min:.3f} · containment mean {text.containment_mean:.3f}"
    )


def _values(values: Sequence[str]) -> str:
    """Render a list of feature values for a table cell — backticked, comma-joined."""
    return ", ".join(f"`{value}`" for value in values) if values else "—"


# ── Provenance: does a specific value trace to something the model was given? ──
#
# The shared half of #1994's provenance assertion.  A case supplies the two texts — what was
# said, and what the round was GIVEN — and this answers which specific values in the first
# appear nowhere in the second.  Pure, so it is plain-testable and one definition serves both
# customers a ported case has: the reply against the world, and a stored entry against the
# pages it claims to come from.

# A SPECIFIC value is one of the classes #1994 names that can be recognised WITHOUT a
# dictionary: a number, a URL, or a capitalised NAME PHRASE of two or more words.
#
# The two-word rule is the load-bearing part, and it is a correction of a measured scorer bug
# rather than a preference.  Counting every capitalised word as a name failed 15 of 18 samples
# on `URLs`, `English`, `I’ve` and `Brandt’s` — ordinary English that happens to carry a
# capital — which is the "too strict" half of exactly the defect this whole design replaces.
# A lone capital is far more often grammar or a common noun than an invention; a fabricated
# entity in this domain is a person, a team or a place, and those are written as several
# capitalised words together.
#
# THE BLIND SPOT, STATED: a single-word invention (a bare invented surname) is not caught
# here.  The cross-world half of that is already an assertion of its own — a reply naming the
# world it was not given fails DIRECTED CHANGE — so what is uncovered is a value belonging to
# NEITHER world, and that is a narrower gap than the false-positive rate it buys off.
_NUMBER = r"\d[\d,.:%$]*"
_URL = r"https?://\S+"
_CAPITALISED = r"[A-Z][A-Za-z'-]*"
_NAME_PHRASE = rf"{_CAPITALISED}(?:\s+{_CAPITALISED})+"
_SPECIFIC = re.compile(rf"{_URL}|{_NAME_PHRASE}|\b{_NUMBER}\b")

# Words that carry a capital everywhere in English and are never part of a name, so a phrase
# is not built across them: the first person, which would otherwise glue two unrelated
# sentences into one "name" at a clause boundary.
_NEVER_A_NAME = frozenset({"i", "im", "ive", "ill", "id"})

# Curly and straight apostrophes are the same character as far as a name is concerned, and the
# model emits whichever its tokenizer prefers — so both sides are folded before comparison.
# Not folding them reported `I’ve` and `Brandt’s` as inventions.
_APOSTROPHES = "’‘`´"


def _fold(text: str) -> str:
    """One spelling of a word, whatever apostrophe it was written with."""
    for mark in _APOSTROPHES:
        text = text.replace(mark, "'")
    return text.casefold()


def _bare(token: str) -> str:
    """A token without its possessive tail — ``Brandt's`` is the same name as ``Brandt``."""
    folded = _fold(token)
    return folded[:-2] if folded.endswith("'s") else folded


def specifics(text: str) -> list[str]:
    """Every specific value stated in ``text`` — URLs, numbers, and the WORDS of each
    capitalised name phrase — in the order they are said, without repeats.

    A phrase decides WHAT gets checked; its words are what is checked.  Measured, the whole
    phrase is too brittle to compare directly: a capitalised label sitting against a name
    (``Key⁠Ridgeline Foxes Sign Aurelio Brandt``, glued by a narrow no-break space) is not a
    string the world contains, though every name in it is.

    THE TRADE, STATED: checking word by word cannot catch a RECOMBINATION of two real names.
    The cross-world form of that is an assertion of its own — a reply carrying the world it was
    not given fails directed change — and a false positive here costs more than that gap, since
    reporting the model's own formatting as a fabrication is the exact defect this replaces."""
    found: list[str] = []
    for match in _SPECIFIC.finditer(_fold_phrases(text)):
        token = match.group().strip()
        parts = [token] if _is_atomic(token) else token.split()
        found += [part for part in parts if part and part not in found]
    return found


def _is_atomic(token: str) -> bool:
    """Whether a match is one value rather than a phrase of them — a URL or a number."""
    return token[0].isdigit() or "://" in token


def _fold_phrases(text: str) -> str:
    """Blank out the words a name phrase may not be built across, so a clause boundary like
    ``… saved. I'll check again`` cannot read as the name ``I'll Check``."""
    return re.sub(
        rf"\b{_CAPITALISED}\b",
        lambda m: (
            " " * len(m.group())
            if _bare(m.group()).replace("'", "") in _NEVER_A_NAME
            else m.group()
        ),
        text,
    )


def unsourced_specifics(text: str, given: str) -> list[str]:
    """The specific values in ``text`` that appear NOWHERE in ``given``.

    An empty list is the assertion holding: everything said traces to something the round was
    handed.  Matching folds apostrophes and drops possessives on both sides, because a value is
    usually said in a different shape from the one it arrived in — and comparing the raw forms
    reported the model's own grammar as an invention."""
    haystack = " ".join(_bare(word) for word in _fold(given).split())
    return [token for token in specifics(text) if _phrase_key(token) not in haystack]


def _phrase_key(token: str) -> str:
    """A phrase as it would appear in the folded haystack — word by word, possessives gone."""
    return " ".join(_bare(word) for word in token.split())
