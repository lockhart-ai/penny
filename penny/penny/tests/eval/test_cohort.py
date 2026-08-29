"""The cohort's pure half (#1994/#1995): what is measured, what is gated out, and how the
case's three sections render.

Plain tests, no model and no database — the whole point of keeping the statistic, the
completeness gate and the rendering in one dependency-light leaf is that they are provable in
``make check`` rather than only observable after a paid run.
"""

from __future__ import annotations

import math

from penny.tests.eval.cohort import (
    REPLY_SPREAD,
    ROUTINE_SHAPE,
    AssertionRow,
    CaseReport,
    RoutineRecord,
    SampleObservation,
    compare_to_ceiling,
    feature_variance,
    normalised_entropy,
    per_sample_cost,
    pool,
    proposed_ceiling,
    proposed_floor,
    specifics,
    unsourced_specifics,
)

_MODEL = "openai/gpt-oss-20b"
_OTHER_MODEL = "google/gemma-4-26b-a4b-it"


def _sample(name: str, arm: str, shape: str, **kwargs) -> SampleObservation:
    """One observation carrying a routine of the given SHAPE — the feature every test here
    measures, since what is under test is the statistic and not the reading of it."""
    routines = [RoutineRecord(name="r", shape=shape, names_a_destination=True)]
    return SampleObservation(name=name, phrasing=arm, routines=routines, **kwargs)


# ── The statistic ────────────────────────────────────────────────────────────
def test_entropy_is_zero_when_every_sample_agrees_and_one_when_none_do():
    assert normalised_entropy(["a"] * 8) == 0.0
    assert normalised_entropy([str(index) for index in range(8)]) == 1.0
    assert normalised_entropy(["only one"]) == 0.0


def test_entropy_normalises_by_cohort_size_which_is_what_makes_it_biased_at_small_n():
    """The ``log(n)`` denominator is the whole reason a ceiling has to carry its N.

    The same 19/13 split reads distinctly higher in a smaller cohort, so a threshold recorded
    at one size says nothing about a run at another — which is why the comparison is refused
    rather than merely discouraged."""
    big = normalised_entropy(["a"] * 19 + ["b"] * 13)
    small = normalised_entropy(["a"] * 6 + ["b"] * 4)
    assert round(big, 2) == 0.19
    assert small > big


def test_a_feature_reports_its_modal_share_and_its_per_phrasing_rows():
    samples = [
        _sample("s1", "p1", "browse → write"),
        _sample("s2", "p1", "browse → write"),
        _sample("s3", "p2", "browse → write"),
        _sample("s4", "p2", "browse → browse → write"),
    ]
    feature = feature_variance(ROUTINE_SHAPE, samples)
    assert (feature.distinct, feature.modal, feature.n) == (2, 3, 4)
    assert round(feature.modal_share, 2) == 0.75
    rows = {row.arm: row for row in feature.phrasings}
    assert rows["p1"].distinct == 1 and rows["p1"].only_here == []
    assert rows["p2"].only_here == ["browse → browse → write"]
    assert rows["p2"].flagged and not rows["p1"].flagged
    assert not rows["p1"].values[1:]  # the shared value, produced under both wordings


def test_a_phrasing_that_agrees_with_its_neighbours_is_not_flagged():
    """The pooled number can only hide a wording that came apart, so what the diagnostic rows
    surface is the value NO other wording produced — not merely a wording that disagreed with
    itself over values everyone else also produced."""
    samples = [
        _sample("s1", "p1", "one"),
        _sample("s2", "p1", "two"),
        _sample("s3", "p2", "one"),
        _sample("s4", "p2", "two"),
    ]
    assert all(not row.flagged for row in feature_variance(ROUTINE_SHAPE, samples).phrasings)


def test_one_phrasing_flags_nothing_because_there_is_no_other_wording_to_differ_from():
    """ "Only under this wording" is a comparison BETWEEN wordings.  With one arm every value
    is trivially unique to the only arm there is, and flagging it would report the shape of the
    cohort as a finding."""
    samples = [_sample(f"s{i}", "phrasing 1", str(i)) for i in range(3)]
    rows = feature_variance(ROUTINE_SHAPE, samples).phrasings
    assert [row.arm for row in rows] == ["phrasing 1"]
    assert rows[0].distinct == 3 and rows[0].only_here == [] and not rows[0].flagged


def test_total_agreement_is_zero_and_never_negative_zero():
    """Total agreement is the one reading of this statistic that has to be unmistakable, and
    the sum computes to NEGATIVE zero — which renders as ``-0.000`` and reads as a number
    rather than as the absence of one."""
    feature = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", "same") for i in range(3)])
    assert feature.entropy == 0.0
    assert f"{feature.entropy:.3f}" == "0.000"


# ── The completeness gate ────────────────────────────────────────────────────
def test_a_dead_sample_is_excluded_by_name_before_anything_is_pooled():
    """A sample's database exists from sample START, so counting files as samples reported 17
    of 31 dead samples as behavioural variance.  The gate runs first and names what it drops."""
    samples = [
        _sample("s1", "p1", "browse → write"),
        _sample("s2", "p1", "browse → write"),
        SampleObservation(
            name="s3", phrasing="p1", complete=False, exclusion="the measured turn never ran"
        ),
    ]
    variance = pool(samples, [ROUTINE_SHAPE, REPLY_SPREAD])
    assert (variance.pooled, variance.driven) == (2, 3)
    assert [(row.name, row.reason) for row in variance.excluded] == [
        ("s3", "the measured turn never ran")
    ]
    assert variance.features[0].n == 2


def test_a_control_is_measured_by_nothing_but_still_counts_as_driven():
    """A control drives the same ask against different facts to serve an ASSERTION, so folding
    its samples into the spread would report a deliberate difference as instability — and the
    harness section still counts it, because a control sample is one the case drove."""
    samples = [
        _sample("s1", "p1", "browse → write"),
        _sample("s2", "the ask", "browse → browse → write", world="control"),
    ]
    variance = pool(samples, [ROUTINE_SHAPE])
    assert (variance.pooled, variance.driven, variance.excluded) == (1, 2, [])
    assert variance.features[0].distinct == 1


def test_reply_spread_reads_the_embeddings_the_replies_already_carry():
    samples = [
        SampleObservation(
            name="s1", phrasing="p1", reply="foxes sign a goalie", reply_embedding=[1.0, 0.0]
        ),
        SampleObservation(
            name="s2", phrasing="p1", reply="the foxes signed a goalie", reply_embedding=[0.0, 1.0]
        ),
        SampleObservation(name="s3", phrasing="p2", reply="a goalie was signed"),
    ]
    spread = pool(samples, [ROUTINE_SHAPE, REPLY_SPREAD]).text
    assert spread is not None
    # Three replies → three text pairs, but only one pair has embeddings on both sides.
    assert spread.pairs == 3
    assert spread.cosine_mean == 0.0
    assert spread.containment_mean > 0.0


# ── Thresholds ───────────────────────────────────────────────────────────────
def test_a_ceiling_sits_a_sampling_margin_above_the_observed_value():
    feature = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", str(i)) for i in range(4)])
    ceiling = proposed_ceiling(feature, _MODEL)
    assert ceiling.value == 1.0  # every sample distinct, and the margin cannot exceed 1.0
    quiet = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", "same") for i in range(4)])
    assert proposed_ceiling(quiet, _MODEL).value == 0.10


def test_only_a_rise_in_variance_is_a_regression():
    observed = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", str(i)) for i in range(4)])
    calm = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", "same") for i in range(4)])
    ceiling = proposed_ceiling(calm, _MODEL)
    assert compare_to_ceiling(ceiling, observed, _MODEL).regressed
    assert not compare_to_ceiling(proposed_ceiling(observed, _MODEL), calm, _MODEL).regressed


def test_a_ceiling_refuses_to_be_compared_across_models_or_cohort_sizes():
    """Both qualifiers make the two numbers different statistics, so the verdict says so rather
    than answering wrongly — a shared ceiling would be useless for the consistent model and
    permanently failing for the variant one."""
    observed = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", str(i)) for i in range(4)])
    other_model = compare_to_ceiling(proposed_ceiling(observed, _MODEL), observed, _OTHER_MODEL)
    assert not other_model.comparable and not other_model.regressed
    assert _OTHER_MODEL in other_model.note

    smaller = feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", str(i)) for i in range(3)])
    other_size = compare_to_ceiling(proposed_ceiling(observed, _MODEL), smaller, _MODEL)
    assert not other_size.comparable and "N=3" in other_size.note


def test_an_assertion_that_did_not_hold_proposes_no_floor():
    """Recording a floor underneath the misses would bless the defect as the contract."""
    assert proposed_floor(AssertionRow(label="held", passed=4, total=4)).lockable
    partial = proposed_floor(AssertionRow(label="missed", passed=3, total=4))
    assert not partial.lockable and partial.note == "1 of 4 missed — read those first"


# ── Provenance ───────────────────────────────────────────────────────────────
def test_an_ordinary_capitalised_word_is_not_a_specific_value():
    """The measured scorer bug this rule is a correction of: counting every capital as a name
    failed 15 of 18 samples on `URLs`, `English`, `I’ve` and `Brandt’s` — ordinary English that
    happens to carry a capital, which is the "too strict" half of the defect being replaced."""
    assert specifics("Here is what I saved.") == []
    assert specifics("I’ve saved the URLs and I’ll check again in English.") == []
    assert specifics("- Saved it.\n* Then again.") == []


def test_a_name_phrase_and_a_number_are_specific_values():
    assert specifics("saved the Ridgeline Foxes headline") == ["Ridgeline", "Foxes"]
    assert specifics("kept 2 entries") == ["2"]
    assert specifics("read https://example.com/news") == ["https://example.com/news"]


def test_a_capital_at_a_clause_boundary_does_not_glue_into_a_name():
    """``… saved. I'll check`` must not read as the name ``I'll Check`` — the first person is
    blanked before phrases are built, so a phrase is never assembled across one."""
    assert specifics("Saved it. I’ll check Ridgeline Foxes again.") == ["Ridgeline", "Foxes"]


def test_a_name_the_round_was_never_given_is_reported_by_name():
    given = "Foxes sign veteran goalie Aurelio Brandt to a two-year deal.\nkeep 2 of them"
    assert unsourced_specifics("Saved the Aurelio Brandt signing — 2 entries.", given) == []
    assert unsourced_specifics("Saved the Casimir Oyelaran signing.", given) == [
        "Casimir",
        "Oyelaran",
    ]


def test_a_value_said_in_a_different_shape_from_the_one_it_arrived_in_still_traces():
    """Apostrophes and possessives are the model's grammar, not an invention — and not folding
    them reported `Brandt’s` as a fabrication."""
    assert unsourced_specifics("the Ridgeline Foxes’ goalie", "Ridgeline Foxes news") == []
    assert unsourced_specifics("Aurelio Brandt’s deal", "signed Aurelio Brandt today") == []


def test_a_capitalised_label_against_a_name_is_not_a_fabrication():
    """The residual false positive the per-word rule exists for: the model wrote
    ``Key⁠Ridgeline Foxes …`` with a narrow no-break space, and comparing the whole phrase
    reported a sourced headline as invented."""
    given = "Ridgeline Foxes sign Aurelio Brandt. entries carry a key and content."
    assert unsourced_specifics("Key\u202fRidgeline Foxes Sign Aurelio Brandt", given) == []


def test_a_single_word_invention_is_the_stated_blind_spot():
    """Stated rather than discovered later: a bare invented surname is NOT caught here.  The
    cross-world half of it is an assertion of its own — a reply naming the world it was not
    given fails directed change — so what is uncovered is a value belonging to NEITHER world."""
    assert unsourced_specifics("the Oyelaran signing", "Aurelio Brandt signed") == []


# ── Cost ─────────────────────────────────────────────────────────────────────
def test_cost_is_reported_per_sample_because_a_total_is_not_comparable():
    """The same trap entropy's ``log(N)`` denominator is: a run total says nothing about a run
    of a different size, so the case divides by the samples that produced it."""
    cost = per_sample_cost(
        samples=4,
        calls=48,
        duration_ms=800_000,
        input_tokens=160_000,
        output_tokens=40_000,
        reasoning_tokens=32_000,
    )
    assert cost is not None
    assert (cost.calls, cost.seconds) == (12.0, 200.0)
    assert (cost.input_tokens, cost.output_tokens) == (40_000, 10_000)
    assert cost.reasoning_share == 0.8
    assert (
        per_sample_cost(
            samples=0, calls=0, duration_ms=0, input_tokens=0, output_tokens=0, reasoning_tokens=0
        )
        is None
    )


# ── The three-section render ─────────────────────────────────────────────────
#
# Whole-render, because what a reader takes a threshold from is the rendered document — and a
# number that renders without the model and the N it was measured at is a number somebody will
# compare across both.

_CEILING_NOTE = (
    "_Ceilings are PROPOSED, not locked, and are one-sided — only a rise is a regression. "
    "Each is recorded as `(feature, model, N=3, value)` and a comparison across either "
    "qualifier is REFUSED: normalised entropy is biased upward at small N (the same behaviour "
    "reads 0.527 at N=32 and 0.605 at N=15), and two models differ ~3x on the same feature "
    "(routine shape 0.53 against 0.18), so a shared ceiling would measure neither._"
)
_PHRASING_LEAD = (
    "_Per-phrasing rows below are DIAGNOSTIC and never locked — at 3 samples each there is no "
    "reliable per-phrasing entropy, so what is reported is the honest weaker signal: a wording "
    "that produced a value no other wording did. Phrasings are a coverage mechanism, and the "
    "pooled number above hides exactly what they are for._"
)
_COST_NOTE = (
    "_Per SAMPLE, never per run — a total is not comparable across cohort sizes. Input is OURS "
    "(prompt and context design), so a rise is what a prompt edit regresses; output is the "
    "MODEL's, so a rise on a fixed prompt is a model or config change. Both ceilings are "
    "one-sided, per model, and PROPOSED — and unlike the variance margin this band is a round "
    "number rather than a measured one._"
)


def test_the_case_report_renders_its_three_sections_whole():
    samples = [
        _sample("case-1 (phrasing 1)", "phrasing 1", "browse → write"),
        _sample("case-2 (phrasing 1)", "phrasing 1", "browse → write"),
        _sample("case-3 (phrasing 2)", "phrasing 2", "browse → browse → write"),
        SampleObservation(
            name="case-4 (phrasing 2)",
            phrasing="phrasing 2",
            complete=False,
            exclusion="the measured turn never ran",
        ),
    ]
    rendered = CaseReport(
        case_id="memory-learn-close-shape",
        model=_MODEL,
        assertions=[
            AssertionRow(label="state: the round taught a routine", passed=3, total=3),
            AssertionRow(label="reply: every specific value in it is sourced", passed=2, total=3),
        ],
        variance=pool(samples, [ROUTINE_SHAPE, REPLY_SPREAD]),
        cost=per_sample_cost(
            samples=4,
            calls=48,
            duration_ms=800_000,
            input_tokens=160_000,
            output_tokens=40_000,
            reasoning_tokens=32_000,
        ),
    ).render()
    assert rendered == "\n\n".join(
        [
            "#### `memory-learn-close-shape` — end-state assertions, variance, harness",
            "**A. Deterministic assertions — end state only.**",
            "\n".join(
                [
                    "| assertion | held | rate | proposed floor |",
                    "|---|---|---|---|",
                    "| state: the round taught a routine | 3/3 | 1.00 | `1.00` |",
                    "| reply: every specific value in it is sourced | 2/3 | 0.67 | "
                    "— 1 of 3 missed — read those first |",
                ]
            ),
            "**B. Variance — model output.**",
            "\n".join(
                [
                    "| feature | distinct | modal | entropy | proposed ceiling |",
                    "|---|---|---|---|---|",
                    "| `routine shape` | 2 | 2/3 (0.67) | 0.579 | "
                    "`0.68` @ openai/gpt-oss-20b N=3 |",
                ]
            ),
            _CEILING_NOTE,
            _PHRASING_LEAD,
            "\n".join(
                [
                    "| feature | phrasing | distinct | only under this wording |",
                    "|---|---|---|---|",
                    # BOTH wordings are flagged, and symmetrically so: each produced a value
                    # the other did not, which is exactly the finding the pooled 0.579 hides.
                    "| `routine shape` | phrasing 1 | 1/2 | `browse → write` |",
                    "| `routine shape` | phrasing 2 | 1/1 | `browse → browse → write` |",
                ]
            ),
            "**Cost, per sample.**",
            "\n".join(
                [
                    "| tokens | observed | proposed ceiling |",
                    "|---|---|---|",
                    "| input tokens (ours — prompt and context) | 40,000 | "
                    "`44,000` @ openai/gpt-oss-20b |",
                    "| output tokens (the model's) | 10,000 | `11,000` @ openai/gpt-oss-20b |",
                ]
            ),
            "Also per sample: 12.0 calls · 200s · 8,000 reasoning tokens (80% of output).",
            _COST_NOTE,
            "**C. Harness — samples too broken to count.**",
            "3 pooled of 4 driven · 1 excluded",
            "- `case-4 (phrasing 2)` — the measured turn never ran",
        ]
    )


def test_a_case_whose_cohort_all_agreed_says_so_rather_than_rendering_an_empty_table():
    rendered = CaseReport(
        case_id="quiet",
        model=_MODEL,
        variance=pool(
            [_sample(f"s{i}", "phrasing 1", "browse → write") for i in range(3)], [ROUTINE_SHAPE]
        ),
    ).render()
    assert "_No phrasing produced a value the others did not._" in rendered
    assert "3 pooled of 3 driven · 0 excluded — every sample ran its measured turn." in rendered
    assert "_(no assertions)_" in rendered


def test_entropy_matches_the_shannon_definition_it_claims_to_be():
    """The one number every threshold is expressed in, checked against its own definition
    rather than against a value copied out of a prior run."""
    values = ["a"] * 3 + ["b"] * 1
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(4)
    assert normalised_entropy(values) == expected
