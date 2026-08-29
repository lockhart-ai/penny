"""The cohort's pure half (#1994/#1995): what is measured, what is gated out, and how the
case's three sections render.

Plain tests, no model and no database — the whole point of keeping the statistic, the
completeness gate and the rendering in one dependency-light leaf is that they are provable in
``make check`` rather than only observable after a paid run.
"""

from __future__ import annotations

import math

from penny.conversation_machine import ConversationState
from penny.tests.eval.conftest import _cohort_checks, _phrasing_label
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    FIELD_UNSET,
    REPLY_SPREAD,
    ROUTINE_SHAPE,
    TOOL_SEQUENCE,
    Arm,
    AssertionRow,
    AssertionSummary,
    OutputField,
    RoutineRecord,
    SampleObservation,
    SpecCategory,
    StoredEntry,
    VarianceFeature,
    assertion_summary,
    compare_to_ceiling,
    feature_variance,
    normalised_entropy,
    output_field,
    per_sample_cost,
    pool,
    proposed_ceiling,
    specifics,
    text_spread,
    unsourced_specifics,
    variance_headline,
)
from penny.tests.eval.utils.worlds import World

_MODEL = "openai/gpt-oss-20b"
_OTHER_MODEL = "google/gemma-4-26b-a4b-it"


def _one_arm(world: World) -> list[Arm]:
    """The chat shape as arms: every sample ran "the ask" against one world.

    The label matches ``_phrasing_label``'s single-wording answer, which is what a sample's own
    ``phrasing`` carries.  The world a claim reads is keyed on the sample's ARM INDEX, though,
    so a hand-built sample here defaults to arm 0 — the only arm there is."""
    return [Arm(label="the ask", text="the ask", world=world)]


def _sample(name: str, arm: str, shape: str, **kwargs) -> SampleObservation:
    """One observation carrying a routine of the given SHAPE — the feature every test here
    measures, since what is under test is the statistic and not the reading of it."""
    routines = [RoutineRecord(name="r", shape=shape, names_a_destination=True)]
    return SampleObservation(name=name, phrasing=arm, arm=0, routines=routines, **kwargs)


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
def _mixed() -> VarianceFeature:
    """A cohort with a real majority — three of one shape, two of another, one of a third.

    Not every-sample-distinct, which is now the SATURATED case and proposes nothing."""
    shapes = ["a", "a", "a", "b", "b", "c"]
    return feature_variance(
        ROUTINE_SHAPE, [_sample(f"s{i}", "p1", shape) for i, shape in enumerate(shapes)]
    )


def _quiet() -> VarianceFeature:
    """The same cohort SIZE as `_mixed`, because a ceiling refuses comparison across N."""
    return feature_variance(ROUTINE_SHAPE, [_sample(f"s{i}", "p1", "same") for i in range(6)])


def test_a_ceiling_sits_a_sampling_margin_above_the_observed_value():
    mixed = _mixed()
    ceiling = proposed_ceiling(mixed, _MODEL)
    assert ceiling is not None
    assert ceiling.value == round(mixed.entropy + 0.10, 3)
    quiet = proposed_ceiling(_quiet(), _MODEL)
    assert quiet is not None and quiet.value == 0.10


def test_a_feature_with_nowhere_left_to_rise_proposes_no_ceiling():
    """A ceiling catches a RISE, and normalised entropy is bounded at 1.0 — so where most samples
    already produced a value no other sample produced, any ceiling above the observed value is a
    guard that can never trip, and printing one implies a gate that does not exist.

    The boundary is NO MAJORITY BEHAVIOUR — the modal value is not shared by even half the
    samples. Chosen over "most values are distinct" because that reads the wrong quantity at
    small N: two distinct values in three samples is ordinary spread, not saturation. It is a
    judgement about where to stop PROPOSING — nothing is gated on it."""
    # The framer's naming shape from the reference run: nine distinct names, the commonest held
    # by four of fifteen — no majority behaviour at all.
    scattered = feature_variance(
        ROUTINE_SHAPE, [_sample(f"s{i}", "p1", str(i // 4)) for i in range(15)]
    )
    assert scattered.modal * 2 < scattered.n
    assert scattered.saturated
    assert proposed_ceiling(scattered, _MODEL) is None

    mixed = _mixed()
    assert not mixed.saturated, "a real majority still proposes"
    assert proposed_ceiling(mixed, _MODEL) is not None


def test_only_a_rise_in_variance_is_a_regression():
    observed, calm = _mixed(), _quiet()
    ceiling = proposed_ceiling(calm, _MODEL)
    assert ceiling is not None
    assert compare_to_ceiling(ceiling, observed, _MODEL).regressed
    higher = proposed_ceiling(observed, _MODEL)
    assert higher is not None
    assert not compare_to_ceiling(higher, calm, _MODEL).regressed


def test_a_ceiling_refuses_to_be_compared_across_models_or_cohort_sizes():
    """Both qualifiers make the two numbers different statistics, so the verdict says so rather
    than answering wrongly — a shared ceiling would be useless for the consistent model and
    permanently failing for the variant one."""
    observed = _mixed()
    ceiling = proposed_ceiling(observed, _MODEL)
    assert ceiling is not None
    other_model = compare_to_ceiling(ceiling, observed, _OTHER_MODEL)
    assert not other_model.comparable and not other_model.regressed
    assert _OTHER_MODEL in other_model.note

    smaller = feature_variance(
        ROUTINE_SHAPE, [_sample(f"s{i}", "p1", shape) for i, shape in enumerate(["a", "a", "b"])]
    )
    other_size = compare_to_ceiling(ceiling, smaller, _MODEL)
    assert not other_size.comparable and "N=3" in other_size.note


def test_the_assertion_number_counts_every_check_not_every_claim():
    """One reading over all deterministic checks, since none of them is gated.

    TOTAL PASSED over TOTAL, not a mean of per-claim rates: while every claim shares a
    denominator the two agree, and they part the moment one does not — a claim fewer samples
    answered would otherwise weigh as heavily as one they all did."""
    rows = [
        AssertionRow(label="a", passed=15, total=15, category=SpecCategory.STORE),
        AssertionRow(label="b", passed=15, total=15, category=SpecCategory.STORE),
        AssertionRow(label="c", passed=10, total=15, category=SpecCategory.PROVENANCE),
    ]
    summary = assertion_summary(rows)
    assert (summary.passed, summary.total) == (40, 45)
    assert summary.rate == 40 / 45
    assert not summary.at_full


def test_a_claim_read_out_of_model_prose_is_counted_like_every_other():
    """The gated/ungated split existed only to decide which claims could carry a floor.

    With no floors it has nothing left to decide, so a reply claim is one more claim: it enters
    the same total and colours on its own rate rather than rendering grey."""
    prose = AssertionRow(
        label="reply: every specific value in it is sourced",
        passed=13,
        total=15,
        kind="reply",
        category=SpecCategory.PROVENANCE,
    )
    structural = AssertionRow(label="s", passed=15, total=15, category=SpecCategory.STORE)
    assert assertion_summary([structural, prose]) == AssertionSummary(passed=28, total=30)


def test_an_empty_case_reads_zero_rather_than_dividing_by_nothing():
    summary = assertion_summary([])
    assert (summary.passed, summary.total, summary.rate) == (0, 0, 0.0)
    assert not summary.at_full, "a case that asserted nothing has not passed everything"


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
    """Stated rather than discovered later: a bare invented surname is NOT caught here, and
    nothing else in the suite covers it.  The strict form of this rule failed 15 of 18 samples
    on ordinary English, so the miss is the honest half of a measured trade."""
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


def test_entropy_matches_the_shannon_definition_it_claims_to_be():
    """The one number every threshold is expressed in, checked against its own definition
    rather than against a value copied out of a prior run."""
    values = ["a"] * 3 + ["b"] * 1
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25)) / math.log(4)
    assert normalised_entropy(values) == expected


# ── Naming a sample ──────────────────────────────────────────────────────────
def test_a_sample_is_named_for_the_wording_it_ran():
    """The report's rows and the sample's own name are keyed on the phrasing LABEL, which is
    read off the sample's position in the drive: three samples per wording, in order."""
    assert _phrasing_label(["a", "b"], 0, 3) == "phrasing 1"
    assert _phrasing_label(["a", "b"], 4, 3) == "phrasing 2"
    assert _phrasing_label(["a"], 0, 3) == "the ask"


def test_a_cases_sample_names_are_distinct():
    """Names are how the cohort's claims are dealt back out to the samples that answered them,
    and the same index keys the sample's database file, which nothing deletes — so a collision
    would silently grade one sample against another's turn."""
    names = [f"case-{i + 1} ({_phrasing_label(['a', 'b'], i, 3)})" for i in range(6)]
    assert len(set(names)) == len(names)
    assert names[:2] == ["case-1 (phrasing 1)", "case-2 (phrasing 1)"]
    assert names[-1] == "case-6 (phrasing 2)"


# ── A claim answers its own sentence ─────────────────────────────────────────
_WROTE_INTO_CONTAINER = "state: the demonstrated write landed in the round's container"
_ONE_SOURCE = World(name="base", pages=(), keeps=(("499",),), excludes=())


def _round(name: str, *, container: str | None, wrote: str | None) -> SampleObservation:
    entries = [StoredEntry(collection="round-box", key=None, content=wrote)] if wrote else []
    return SampleObservation(
        name=name,
        phrasing="the ask",
        arm=0,
        landed=ConversationState.LEARN.value,
        container=container,
        entries=entries,
        given=wrote or "",
    )


def test_a_sample_that_wrote_nothing_fails_the_claim_about_where_its_write_landed():
    """A claim is a statement about END STATE and answers its own sentence on every sample.

    With nothing written, `the demonstrated write landed in the round's container` is FALSE —
    no write landed there. Answering TRUE to avoid reporting one broken sample as several
    contract violations traded the truth of the check for the tidiness of its output, and
    printed a perfect 3/3 for a cohort in which a sample wrote nothing at all.

    A sample that did nothing genuinely fails every claim about what it should have done."""
    cohort = Cohort(
        "case",
        "m",
        [
            _round("wrote-into-it", container="round-box", wrote="499"),
            _round("wrote-elsewhere", container="other-box", wrote="499"),
            _round("wrote-nothing", container="round-box", wrote=None),
        ],
        _one_arm(_ONE_SOURCE),
    )
    cohort.assert_the_write_landed_in_the_round_container()
    claim = cohort.claims[0]

    assert (claim.passed, claim.total) == (1, 3), "the denominator counts every sample"
    verdicts = {outcome.sample: outcome.ok for outcome in claim.outcomes}
    assert verdicts == {"wrote-into-it": True, "wrote-elsewhere": False, "wrote-nothing": False}
    assert "nothing was written" in claim.rationales


def test_an_unframed_round_fails_the_only_claim_that_reads_the_framing():
    """No round framed means no container for a write to land in, so the sentence is false.

    This is also the suite's ONLY reader of the round framing — no other claim mentions the
    container — so an unframed round is invisible unless this claim says so."""
    cohort = Cohort(
        "case", "m", [_round("unframed", container=None, wrote="499")], _one_arm(_ONE_SOURCE)
    )
    cohort.assert_the_write_landed_in_the_round_container()
    claim = cohort.claims[0]

    assert (claim.passed, claim.total) == (0, 1)
    assert claim.rationales == ["no round was framed, so no write could land in its container"]


# ── Typography is folded ONCE, on both sides of every comparison ─────────────
_GIVEN = "browse: https://faux-market.example/aurora-deck-2 says Price: $499"


def test_a_url_written_with_a_non_breaking_hyphen_is_the_url_it_was_given():
    """MEASURED: a sample cited the page it read, drawing its dashes as U+2011, and the probe
    called the URL an invention — while the store claim beside it folded that same dash and
    agreed the value was sourced. One folding now serves both."""
    drawn = "https://faux\u2011market.example/aurora\u2011deck\u20112"
    assert unsourced_specifics(f"saved the price from {drawn}", _GIVEN) == []


def test_a_url_ending_a_sentence_does_not_swallow_the_full_stop():
    """MEASURED on both of one model's misses: `\\S+` ran the sentence mark into the URL, so
    `…/aurora-deck-2.` matched no world and a correctly cited page read as invented."""
    assert unsourced_specifics("i read https://faux-market.example/aurora-deck-2.", _GIVEN) == []
    assert unsourced_specifics("see https://faux-market.example/aurora-deck-2, then", _GIVEN) == []


def test_a_url_that_legitimately_ends_in_punctuation_keeps_it():
    """The paired over-correction guard: only `.,;:!?` are refused as the LAST character, so a
    bracket, a slash or a dash that is genuinely part of the address survives."""
    assert specifics("https://en.wikipedia.org/wiki/Foo_(bar)") == [
        "https://en.wikipedia.org/wiki/Foo_(bar)"
    ]
    assert specifics("https://example.com/news/") == ["https://example.com/news/"]


def test_an_invented_url_is_still_caught():
    """The probe must not be loosened into uselessness by either fix."""
    assert unsourced_specifics("see https://other.example/nope.", _GIVEN) == [
        "https://other.example/nope"
    ]


# ── What reached the user (#2009) ────────────────────────────────────────────
_NO_WORLD = World(name="base", pages=(), keeps=(), excludes=())


def _delivered(*messages: str) -> Cohort:
    """A one-sample cohort that delivered exactly these messages, oldest first."""
    sample = SampleObservation(
        name="s0", phrasing="the ask", landed="idle", reply=messages[-1], delivered=list(messages)
    )
    return Cohort("case", _MODEL, _NO_WORLD, [sample])


def _answered(cohort: Cohort) -> tuple[int, int]:
    claim = cohort.claims[0]
    return claim.passed, claim.total


_CLEAN_REPLY = "Lake Baikal is the deepest, at about 1,642 metres."
_SERIALIZED_CALL = '{"queries": ["deepest lake"], "reasoning": "look it up"}'
_LEAKED_ENVELOPE = (
    "<|start|>assistant<|channel|>analysis to=functions.browse code<|message|><|call|>"
)


def test_an_unusable_draw_is_read_by_productions_rule_not_by_the_injected_shape():
    """The claim reads EVERY shape production refuses, not the one a case's injector forces.

    A recovery case keyed to its own fault would pass while a DIFFERENT unusable draw sailed
    out in the same turn — so the condition set is composed from the chat agent's own
    ``invalid_draw_conditions`` plus the transport artifacts the loop checks on every draw,
    and this pins that both halves answer.  Asserting a leaked envelope here is deliberate:
    it is the shape the call-as-text case's own injector never emits."""
    clean = _delivered(_CLEAN_REPLY)
    clean.assert_no_delivered_message_is_an_unusable_draw()
    assert _answered(clean) == (1, 1)

    for unusable in (_SERIALIZED_CALL, _LEAKED_ENVELOPE, "{}"):
        cohort = _delivered(unusable)
        cohort.assert_no_delivered_message_is_an_unusable_draw()
        assert _answered(cohort) == (0, 1), f"{unusable!r} is a draw the loop refuses"
        assert cohort.claims[0].rationales, "and the rationale names the condition that fired"


def test_every_delivered_message_is_read_and_not_only_the_last():
    """What the contract forbids is the bad draw reaching the user AT ALL.

    A turn that delivers two messages would otherwise be judged on one of them, which is how
    a discarded draw arriving FIRST went unseen — so the bad one leads here, where a
    reply-only read would miss it."""
    cohort = _delivered(_SERIALIZED_CALL, _CLEAN_REPLY)
    cohort.assert_no_delivered_message_is_an_unusable_draw()
    assert _answered(cohort) == (0, 1)


def test_a_delivered_message_is_whole_by_the_rule_the_send_path_refuses_by():
    """Completeness is production's ``half_formed_send_reason``, not a letter count.

    The floor it replaces was fifteen letters — a number somebody picked, which stands for
    the question rather than answering it.  Both directions matter, and the SHORT-BUT-COMPLETE
    case is the one that floor got wrong: a brief finished answer is a message a person should
    receive, and a long one trailing off into an ellipsis is not."""
    whole = _delivered("Baikal.")
    whole.assert_every_delivered_message_is_whole()
    assert _answered(whole) == (1, 1), "a short complete answer is a complete message"

    trailed_off = "the deepest one is …"
    bailed_out = "I don't know"
    bare_url = "https://geo.example.test/lakes"
    for half_formed in (trailed_off, bailed_out, bare_url):
        cohort = _delivered(half_formed)
        cohort.assert_every_delivered_message_is_whole()
        assert _answered(cohort) == (0, 1), f"{half_formed!r} is not a message to deliver"
        assert cohort.claims[0].rationales, "and production's own reason is the rationale"


# ── Every deterministic check is scored, counted and coloured ────────────────
def test_no_claim_can_opt_out_of_being_scored():
    """Deterministic checks are ALWAYS scored — there is no advisory claim on this side.

    The per-sample scorer path has ``scored=False`` for spine and proc flavour; nothing
    equivalent exists here, and this pins that a claim cannot acquire one. A claim that could
    render without counting is a claim that could hold a case's number up while failing."""
    world = World(name="base", pages=(), keeps=(("499",),), excludes=())
    samples = [
        SampleObservation(
            name=f"s{i}", phrasing="the ask", arm=0, landed="learn", reply="x", given="x"
        )
        for i in range(2)
    ]
    cohort = Cohort("case", "m", samples, _one_arm(world))
    cohort.assert_machine_landed(ConversationState.LEARN)
    cohort.assert_every_value_in_the_reply_is_sourced()

    checks = _cohort_checks(cohort)
    assert set(checks) == {"s0", "s1"}
    every = [check for sample_checks in checks.values() for check in sample_checks]
    assert len(every) == 4, "both claims reach both samples"
    assert all(check.scored for check in every), "no cohort claim is advisory"
    assert not any(check.ignored for check in every), "no cohort claim is out of the denominator"


def test_the_summary_counts_a_reply_claim_exactly_like_a_state_one():
    """One reading over every check, with no claim able to sit outside it."""
    rows = [
        AssertionRow(label="state: x", passed=15, total=15, category=SpecCategory.STORE),
        AssertionRow(
            label="reply: y", passed=13, total=15, kind="reply", category=SpecCategory.PROVENANCE
        ),
    ]
    assert assertion_summary(rows) == AssertionSummary(passed=28, total=30)


# ── The headline surfaces what varies most, gating is a separate job ────────
def _feature(name: str, entropy: float, *, modal: int, n: int = 15) -> VarianceFeature:
    return VarianceFeature(name=name, n=n, distinct=n - modal + 1, modal=modal, entropy=entropy)


def test_the_headline_names_the_most_variant_feature_even_when_it_cannot_be_gated():
    """SURFACING and GATING are different jobs, and only the second is saturation's.

    A ceiling exists to catch a RISE, so a feature with no majority behaviour can carry none —
    and `proposed_ceiling` still refuses one. The headline answers a different question: which
    aspect of this case is most variant right now. Filtering the saturated feature out of it
    hides exactly the answer, which is what "if a bunch of measures have 0 variance but some
    measure does, that's what we want to surface" asks for."""
    features = [
        _feature("tool sequence", 0.090, modal=14),
        _feature("routine shape", 0.000, modal=15),
        _feature("routine name", 0.885, modal=3),
    ]
    headline = variance_headline(features)

    assert (headline.feature, headline.entropy) == ("routine name", 0.885)
    assert proposed_ceiling(features[2], _MODEL) is None, "and it still proposes no ceiling"


def test_the_headline_says_how_much_of_the_case_varies_at_all():
    """Magnitude alone cannot tell one aspect spiking from everything wobbling.

    Counted STRUCTURALLY — more than one distinct value — so no magnitude threshold enters."""
    spike = variance_headline(
        [
            _feature("tool sequence", 0.000, modal=15),
            _feature("routine shape", 0.000, modal=15),
            _feature("routine name", 0.885, modal=3),
        ]
    )
    assert (spike.varying, spike.total) == (1, 3), "one aspect moved, two did not"

    wobble = variance_headline(
        [
            _feature("tool sequence", 0.400, modal=10),
            _feature("routine shape", 0.300, modal=11),
            _feature("routine name", 0.885, modal=3),
        ]
    )
    assert (wobble.varying, wobble.total) == (3, 3), "and here everything moved"
    assert spike.entropy == wobble.entropy, "which one maximum alone could not have told apart"


def test_a_feature_destabilising_past_the_leader_raises_the_reading():
    """The direct refutation of the gateable-max this replaces.

    Under that design a feature crossing INTO saturation LEFT the set, taking its spread with it,
    so the headline FELL while behaviour worsened. Counting every feature, the same movement
    RAISES the number and renames the feature — which is the signal a reader needs."""
    before = variance_headline(
        [_feature("tool sequence", 0.450, modal=9), _feature("routine name", 0.700, modal=4)]
    )
    after = variance_headline(
        [_feature("tool sequence", 0.950, modal=3), _feature("routine name", 0.700, modal=4)]
    )

    assert before.feature == "routine name", "naming led while the sequence was stable"
    assert after.feature == "tool sequence", "and the destabilised feature takes the headline"
    assert after.entropy > before.entropy, "the number RISES, where the old design fell"


def test_a_case_that_measured_nothing_has_no_reading():
    """No features at all is the only case with nothing to report — with every feature counted,
    a cohort that measured anything always has a most-variant one."""
    empty = variance_headline([])
    assert not empty.has_reading
    assert (empty.varying, empty.total) == (0, 0)


# ── A feature that read NOTHING is not a feature in agreement (#2017) ────────
#
# The two are the same 0.000 and opposite findings, and telling them apart is the whole of
# the measurement-blindness guard: total agreement is the best result a feature can report,
# while reading nothing on every sample is a feature that CANNOT report an outlier.


def test_a_feature_that_read_nothing_on_every_sample_is_blind_not_agreed():
    """The concrete trap: a non-chat fixture whose tool-sequence reader is filtered to the
    chat agent's rows comes back empty on every sample.  Entropy is 0.000 either way, so
    without this the report files a blindfold beside a cohort in perfect agreement."""
    blind = feature_variance(
        TOOL_SEQUENCE, [_sample(f"s{index}", "p1", "shape") for index in range(6)]
    )
    assert blind.entropy == 0.0
    assert blind.blind, "every sample read TOOL_SEQUENCE's absent value"

    seen = feature_variance(
        TOOL_SEQUENCE,
        [_sample(f"s{index}", "p1", "shape", tool_sequence=["browse"]) for index in range(6)],
    )
    assert seen.entropy == 0.0
    assert not seen.blind, "an agreed READING is agreement, not blindness"


def test_a_blind_feature_proposes_no_ceiling():
    """A ceiling recorded on a feature that read nothing locks the blindness in as the
    expected behaviour, and the guard it prints could never fire."""
    blind = feature_variance(
        TOOL_SEQUENCE, [_sample(f"s{index}", "p1", "shape") for index in range(6)]
    )
    assert proposed_ceiling(blind, _MODEL) is None

    seen = feature_variance(
        TOOL_SEQUENCE,
        [_sample(f"s{index}", "p1", "shape", tool_sequence=["browse"]) for index in range(6)],
    )
    assert proposed_ceiling(seen, _MODEL) is not None, "an agreed reading still proposes"


def test_a_feature_declaring_no_absent_reading_is_never_blind():
    """A feature that declares no saw-nothing reading, and reads a real value, is agreement —
    however uniform it is.  ``entries stored`` reading ``0`` on every sample is a fact about
    the round, not a hole in the measurement."""
    entries = feature_variance(
        ENTRIES_STORED, [_sample(f"s{index}", "p1", "shape") for index in range(6)]
    )
    assert entries.distinct == 1 and not entries.blind


def test_a_field_that_is_empty_on_every_sample_is_blind_whatever_it_declares():
    """The empty string is the one "nothing" EVERY feature shares, so it needs no declaration.

    A structured field populated only on one outcome — the extractor's ``reason``, which
    carries the what-is-missing line on ``NOT_PRESENT`` and nothing on ``EXTRACTED`` — reads
    ``""`` on all fifteen samples of a run where every draw succeeded.  It goes blind
    PRECISELY on the runs that look best, and that is when it would otherwise propose a gate.

    The old rule could not catch it either way: the field's real value is ``""`` rather than
    the declared ``unset``, and a feature declaring ``""`` was skipped by a falsiness guard,
    so an empty saw-nothing reading was unexpressible by construction."""
    empty = [
        SampleObservation(
            name=f"s{index}",
            phrasing="the ask",
            output=[OutputField(name="reason", value="")],
        )
        for index in range(6)
    ]
    blind = feature_variance(output_field("reason"), empty)
    assert blind.entropy == 0.0
    assert blind.blind, "every sample read no value at all"
    assert proposed_ceiling(blind, _MODEL) is None, "a gate on a non-reading can never fire"

    # The same feature on a run where the outcome DID happen is a real measurement.
    populated = [
        SampleObservation(
            name=f"s{index}",
            phrasing="the ask",
            output=[OutputField(name="reason", value=f"no price on the page {index % 2}")],
        )
        for index in range(6)
    ]
    seen = feature_variance(output_field("reason"), populated)
    assert not seen.blind and seen.distinct == 2


def test_a_feature_may_declare_the_empty_string_as_its_absent_reading():
    """``None`` means "declares none"; ``""`` is a legitimate declaration.  Collapsing the two
    into a falsiness check is what made an empty saw-nothing reading unexpressible."""
    from penny.tests.eval.utils.cohort import Feature

    declared = Feature("blank", lambda observation: observation.walk, absent="")
    assert declared.absent == ""
    assert Feature("undeclared", lambda observation: observation.walk).absent is None


# ── A single-call context: structured output, no store, no reply (#2017) ─────
def test_a_draws_structured_output_is_read_by_field_and_measured_by_field():
    """What a micro-context case asserts and what it measures are the same values: the
    fields of the typed result the draw returned."""
    sample = SampleObservation(
        name="s1",
        phrasing="the ask",
        output=[
            OutputField(name="outcome", value="extracted"),
            OutputField(name="value", value="84 zorkmids"),
        ],
    )
    assert sample.field("outcome") == "extracted"
    assert sample.field("value") == "84 zorkmids"
    assert sample.output_text == "outcome extracted value 84 zorkmids"

    feature = output_field("outcome")
    assert feature.read(sample) == "extracted"


def test_a_field_the_draw_never_returned_reads_unset_rather_than_blank():
    """A real reading, not a missing one: a draw that answered with the wrong shape genuinely
    has nothing there, so a claim about that field is FALSE of the sample rather than unasked
    — and the variance table gets a word rather than an empty cell."""
    sample = SampleObservation(name="s1", phrasing="the ask")
    assert sample.field("outcome") == FIELD_UNSET
    assert output_field("outcome").absent == FIELD_UNSET


def test_a_single_call_sample_pools_even_though_it_has_no_reply_and_no_store():
    """The completeness gate is per FIXTURE.  A micro-context sample has no reply at all, and
    chat's own "no reply" condition applied here would void every sample and refuse the run —
    so a sample carrying only its structured answer must pool as the complete sample it is."""
    samples = [
        SampleObservation(
            name=f"s{index}",
            phrasing="the ask",
            output=[OutputField(name="outcome", value="extracted")],
        )
        for index in range(4)
    ]
    variance = pool(samples, [output_field("outcome")])
    assert variance.pooled == 4
    assert variance.excluded == []
    assert variance.features[0].distinct == 1
    assert not variance.features[0].blind


def test_a_reply_spread_with_no_embeddings_says_so_rather_than_reading_zero():
    """``_mean([])`` is 0.000, and in the spread table that reads as "every pair maximally
    dissimilar" — the strongest possible finding — when what happened is that nothing could be
    compared.  A collector's notification is the standing case: a cycle ENQUEUES and the
    drainer is a separate schedule, so its text is usually absent from the outgoing messages
    the embeddings are read off."""
    unembedded = [
        SampleObservation(name=f"s{index}", phrasing="the ask", reply="the price moved")
        for index in range(3)
    ]
    spread = text_spread(unembedded)
    assert spread is not None
    assert spread.pairs == 3, "containment is computable over every pair"
    assert spread.cosine_pairs == 0
    assert not spread.cosine_measurable, "0.000 here is the absence of a reading"

    embedded = [
        SampleObservation(
            name=f"s{index}",
            phrasing="the ask",
            reply="the price moved",
            reply_embedding=[1.0, 0.0],
        )
        for index in range(3)
    ]
    measured = text_spread(embedded)
    assert measured is not None and measured.cosine_measurable
    assert measured.cosine_pairs == 3
