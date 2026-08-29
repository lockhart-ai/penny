"""Whole-render tests for the transcript-integrated report grammar (``report.py``, #1725/#1753).

NOT eval-marked — they drive the PURE renderer over hand-built ``SampleTranscript``s (no DB, no
model, no git), so they run inside ``make check`` and pin every form of the iteration-6 grammar
as a WHOLE-RENDER literal (pr-review-guide §6). Every sample folds whole under its banner
(uniform collapse, #1753) carrying only its OWN sequence (#1997 — a cohort's shared system prompts
are stated once on the case document, which has its own whole-render tests below): the clean pass
with all three named micro-context actors (#1773), the failure with a nudge + run-close + n/a, the
harness-timeout placeholder, the diff-mode
regressed flip with a baseline row, and an advisory check + empty thinking on a fragile pass all
render inside a ``<details>``; plus the deterministic cell hygiene (single-copy collapsed
truncation + escaping, #1759) and the fold/parse seam the assembler's re-normalization rides on
(EVERY sample folds whole — the one and only rendering, no banner-only form).
"""

from __future__ import annotations

from penny.tests.eval.utils import cohort, report, worlds
from penny.tests.eval.utils.fixtures import CannedPage


def test_clean_pass_folds_whole_with_its_own_sequence_and_micro_context() -> None:
    """A clean pass folds into one ``<details>`` carrying ONLY its own sequence — the turns it was
    given, the calls it made, what came back, what it replied (#1997).

    Its system prompts are NOT here: every sample in a cohort is handed the same ones, so the case
    document states each distinct one once rather than eighteen times.  What stays per sample is
    what only that sample did — so EVERY micro-context call renders inline as its own named actor
    (🧩 <context> ← user turn: / →, #1759/#1773) with its own thinking (the state classifier at
    the head of the turn it decided, the browse extraction after the call that spawned it, the
    run-end skill labeller closing the turn), and an action with no captured thinking shows
    ``💭 (empty)``."""
    events = [
        report.Event(report.EventKind.USER, "deepest lake?"),
        report.Event(
            report.EventKind.MICRO_IN, "newest message: deepest lake?", context="state-classifier"
        ),
        report.Event(
            report.EventKind.MICRO_OUT,
            "STATE: idle",
            thinking="a question, no task",
            context="state-classifier",
        ),
        report.Event(
            report.EventKind.CALL,
            'browse({"queries":["x"],"extract":"depth"})',
            thinking="verify with source",
        ),
        report.Event(
            report.EventKind.MICRO_IN,
            "Instruction: depth · Content: 1,642 m",
            context="browse-extract",
        ),
        report.Event(
            report.EventKind.MICRO_OUT,
            "EXTRACTED: 1642",
            thinking="value present",
            context="browse-extract",
        ),
        report.Event(report.EventKind.RESULT, "You opened wiki (browse result) · 1642"),
        report.Event(report.EventKind.REPLY, "Lake Baikal, 1,642 m.", thinking=""),
        report.Event(report.EventKind.MICRO_IN, "steps: browse", context="skill-namer"),
        report.Event(
            report.EventKind.MICRO_OUT,
            "NAME: look-up-a-lake-depth",
            thinking="generic name",
            context="skill-namer",
        ),
    ]
    checks = [
        report.CheckView("C1", "browsed", "spine", True, False, True, anchor_index=3),
        report.CheckView("C2", "reply names the fact", "reply", True, False, True, anchor_index=7),
    ]
    banner = report.render_banner(passed=True, duration_s=45, calls=8)
    sample = report.build_sample(
        number=1,
        banner=banner,
        events=events,
        checks=checks,
        run_close_score="2/2",
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 1 — ✅ pass · 45s · 8 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "deepest lake?" | ✅ |\n'
        "|---|---|---|\n"
        "| expected | C1 [spine]⚖ browsed |  |\n"
        "| expected | C2 [reply]⚖ reply names the fact |  |\n"
        "| actual | 🧩 state-classifier ← user turn: newest message: deepest lake? |  |\n"
        "| 💭 | thinking (state-classifier) — 19 chars: a question, no task |  |\n"
        "| actual | 🧩 state-classifier → STATE: idle |  |\n"
        "| 💭 | verify with source |  |\n"
        '| actual | 🔧 browse({"queries":["x"],"extract":"depth"}) | ✅ C1 |\n'
        "| actual | 🧩 browse-extract ← user turn: Instruction: depth · Content: 1,642 m |  |\n"
        "| 💭 | thinking (browse-extract) — 13 chars: "
        "value present |  |\n"
        "| actual | 🧩 browse-extract → EXTRACTED: 1642 |  |\n"
        "| actual | 📥 You opened wiki (browse result) · 1642 |  |\n"
        "| actual | 🤖 Lake Baikal, 1,642 m. | ✅ C2 |\n"
        "| actual | 🧩 skill-namer ← user turn: steps: browse |  |\n"
        "| 💭 | thinking (skill-namer) — 12 chars: "
        "generic name |  |\n"
        "| actual | 🧩 skill-namer → NAME: look-up-a-lake-depth |  |\n"
        "\n"
        "</details>"
    )


def test_failed_sample_with_nudge_run_close_and_na() -> None:
    """A failure folds whole under its banner too (#1753): a recovery nudge renders ``⚠ recovery
    event`` inside its step, the failed anchor verdict carries its rationale + cause, and whole-run
    + n/a checks fall to the run-close table (a missing-action check as ❌, an n/a as ➖)."""
    events = [
        report.Event(report.EventKind.USER, "drop the read step"),
        report.Event(
            report.EventKind.REPLY, "I'll ditch that. Just to...", thinking="fold in once confirmed"
        ),
        report.Event(report.EventKind.NUDGE, "*(nudge)* Please provide your response."),
        report.Event(report.EventKind.REPLY, "Updated plan.", thinking="restate"),
    ]
    checks = [
        report.CheckView(
            "C7",
            "remove: read gone",
            "state",
            True,
            False,
            False,
            rationale="read still in recipe",
            cause="behavioral",
            anchor_index=3,
        ),
        report.CheckView(
            "C2",
            "applied edits",
            "spine",
            True,
            False,
            False,
            rationale="never called",
            cause="behavioral",
            anchor_index=None,
        ),
        report.CheckView("C3", "no give-up reply", "proc", True, False, True, anchor_index=None),
        report.CheckView(
            "C8",
            "reminder set",
            "state",
            True,
            True,
            True,
            rationale="no cadence in the ask",
            anchor_index=None,
        ),
    ]
    banner = report.render_banner(
        passed=False,
        cause="behavioral",
        duration_s=120,
        calls=13,
    )
    sample = report.build_sample(
        number=3, banner=banner, events=events, checks=checks, run_close_score="1/2"
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 3 — ❌ fail · "
        "behavioral · 120s · 13 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "drop the read step" | ❌ |\n'
        "|---|---|---|\n"
        "| expected | C7 [state]⚖ remove: read gone |  |\n"
        "| 💭 | <details><summary>thinking — 22 chars</summary>"
        "fold in once confirmed</details> |  |\n"
        "| actual | 🤖 I'll ditch that. Just to... |  |\n"
        "| actual | 👤 *(nudge)* Please provide your response. | ⚠ recovery event |\n"
        "| 💭 | restate |  |\n"
        "| actual | 🤖 Updated plan. | ❌ C7 — read still in recipe · behavioral |\n"
        "\n"
        "| run-close | whole-conversation contracts | 1/2 |\n"
        "|---|---|---|\n"
        "| expected | C2 [spine]⚖ applied edits | ❌ C2 — never called · behavioral |\n"
        "| expected | C3 [proc]⚖ no give-up reply | ✅ C3 |\n"
        "| expected | C8 [state] reminder set | ➖ n/a — no cadence in the ask |\n"
        "\n"
        "</details>"
    )


def test_timeout_sample_renders_placeholder() -> None:
    """A harness-timeout sample (no completed turn) folds its banner + the honest placeholder —
    never silently omitted (F2). The banner omits ``k/n`` (the scorer never ran)."""
    banner = report.render_banner(
        passed=False,
        cause="harness",
        duration_s=118,
        calls=13,
    )
    sample = report.build_sample(
        number=3,
        banner=banner,
        events=[],
        checks=[],
        run_close_score="",
        placeholder=report.NO_TURNS_PLACEHOLDER,
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 3 — ❌ fail · harness · 118s · 13 calls</summary>\n"
        "\n"
        "_(no completed turns recorded — the sample produced no finished model call, "
        "e.g. a harness timeout)_\n"
        "\n"
        "</details>"
    )


def test_diff_mode_regressed_flip_with_baseline_row() -> None:
    """Diff mode: the step header shows the ✅→❌ flip, a ``baseline`` row carries the prior run's
    passing anchor, and the actual row's verdict is ``✅→❌ REGRESSED``."""
    events = [
        report.Event(report.EventKind.USER, "stop notifying me"),
        report.Event(report.EventKind.REPLY, "Turning it off", thinking="defer"),
    ]
    checks = [
        report.CheckView(
            "C8",
            "notify off",
            "state",
            True,
            False,
            False,
            rationale="notify still on",
            cause="behavioral",
            anchor_index=1,
            regressed=True,
            baseline_event='🔧 collection_set({"notify":false}) → confirmed',
            baseline_ok=True,
        ),
    ]
    banner = report.render_banner(
        passed=False,
        cause="behavioral",
        duration_s=60,
        calls=5,
    )
    sample = report.build_sample(
        number=1, banner=banner, events=events, checks=checks, run_close_score="3/4"
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 1 — ❌ fail · behavioral · 60s · 5 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "stop notifying me" | ✅→❌ |\n'
        "|---|---|---|\n"
        "| expected | C8 [state]⚖ notify off |  |\n"
        '| baseline | 🔧 collection_set({"notify":false}) → confirmed | ✅ C8 *(prior run)* |\n'
        "| 💭 | defer |  |\n"
        "| actual | 🤖 Turning it off | ✅→❌ **REGRESSED** C8 — notify still on · behavioral |\n"
        "\n"
        "</details>"
    )


def test_advisory_and_empty_thinking_on_a_fragile_pass() -> None:
    """An advisory check renders ``ℹ`` in its expected body (its anchor verdict still counts as a
    render, not a score), an empty thought is ``💭 (empty)``, and the banner carries ``fragile``."""
    events = [
        report.Event(report.EventKind.USER, "add game and remind me friday"),
        report.Event(report.EventKind.CALL, 'collection_write("games")', thinking=""),
    ]
    checks = [
        report.CheckView("C1", "entry written", "state", True, False, True, anchor_index=1),
        report.CheckView(
            "C2", "single-write efficiency", "spine", False, False, True, anchor_index=1
        ),
        report.CheckView(
            "C3",
            "reminder set",
            "state",
            True,
            True,
            True,
            rationale="no cadence in the ask",
            anchor_index=None,
        ),
    ]
    banner = report.render_banner(
        passed=True,
        fragile=True,
        duration_s=30,
        calls=4,
    )
    sample = report.build_sample(
        number=2, banner=banner, events=events, checks=checks, run_close_score="1/1"
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 2 — ✅ pass · fragile · 30s · 4 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "add game and remind me friday" | ✅ |\n'
        "|---|---|---|\n"
        "| expected | C1 [state]⚖ entry written |  |\n"
        "| expected | C2 [spine]ℹ single-write efficiency |  |\n"
        '| actual | 🔧 collection_write("games") | ✅ C1 · ✅ C2 |\n'
        "\n"
        "| run-close | whole-conversation contracts | 1/1 |\n"
        "|---|---|---|\n"
        "| expected | C3 [state] reminder set | ➖ n/a — no cadence in the ask |\n"
        "\n"
        "</details>"
    )


def test_cell_hygiene_escape_and_truncate() -> None:
    """The deterministic cell rules: ``|`` is escaped and newlines become ``<br>``; a cell over the
    limit collapses into a SINGLE ``<details>`` — its first line + ``… (<n> chars)`` in the summary,
    the full escaped text inside, one copy, no visible head (#1759)."""
    assert report.escape_cell("a|b\nc") == "a\\|b<br>c"
    long_cell = "A" * 520 + " | pipe and\nnewline"
    assert report.truncate_cell(long_cell) == (
        "<details><summary>"
        + "A" * 520
        + " \\| pipe and … (539 chars)</summary>"
        + "A" * 520
        + " \\| pipe and<br>newline</details>"
    )
    # A short cell is escaped in place with no <details>.
    assert report.truncate_cell("short | cell") == "short \\| cell"


def test_fold_and_parse_round_trip() -> None:
    """The assembler's re-normalization seam (#1753): ``fold_sample`` wraps a body under its banner
    (the one and only rendering — collapsed, full body a click away), and ``parse_sample_block``
    recovers ``(number, banner, body)`` from BOTH the folded form and the legacy ``#### `` heading
    (so a re-assembled prior run's unfolded failures fold uniformly too)."""
    body = '| step 1 · 👤 | "hi" |  |\n|---|---|---|\n| actual | 🤖 hey |  |'
    folded = report.fold_sample(2, "✅ pass · 10s · 2 calls", body)
    assert folded == (
        f"<details><summary>sample 2 — ✅ pass · 10s · 2 calls</summary>\n\n{body}\n\n</details>"
    )
    assert report.parse_sample_block(folded) == (2, "✅ pass · 10s · 2 calls", body)
    heading = f"#### sample 3 — ❌ fail · behavioral · 120s · 5 calls\n\n{body}"
    assert report.parse_sample_block(heading) == (3, "❌ fail · behavioral · 120s · 5 calls", body)


def test_split_sample_blocks_separates_mixed_forms() -> None:
    """``split_sample_blocks`` splits a case transcript into its per-sample blocks in order, across
    a folded block followed by a legacy unfolded ``#### `` block (the re-assembly case)."""
    folded = report.fold_sample(1, "✅ pass · 8s · 2 calls", "| a | b |  |")
    heading = "#### sample 2 — ❌ fail · harness · 120s · 3 calls\n\n_(no completed turns)_"
    transcript = f"{folded}\n\n{heading}\n\n"
    assert report.split_sample_blocks(transcript) == [folded, heading]
    assert report.split_sample_blocks("") == []


def test_a_case_level_preamble_is_split_off_rather_than_parsed_as_a_sample() -> None:
    """A ported case writes its three-section report above its samples (#1995), so a case
    transcript can open on something that is not a sample block.  It comes back as the
    PREAMBLE, verbatim and unparsed — nothing here has to know what a case says about
    itself — and a transcript that opens straight onto sample 1 reports no preamble at all,
    which is what it did before case sections existed."""
    folded = report.fold_sample(1, "✅ pass · 8s · 2 calls", "| a | b |  |")
    sections = "#### `case` — end-state assertions, variance, harness\n\n**A.** …"
    assert report.split_case_transcript(f"{sections}\n\n{folded}\n\n") == (sections, [folded])
    assert report.split_case_transcript(f"{folded}\n\n") == ("", [folded])
    assert report.split_case_transcript("") == ("", [])


# ── The case document: sharing DECLARED, not discovered (#1997) ─────────────


def _prompt(context: str, text: str) -> report.SystemPrompt:
    return report.SystemPrompt(context=context, text=text)


def _pairs(*rows: tuple[str, str, str]) -> list[tuple[str, report.SystemPrompt]]:
    return [(sample, _prompt(context, text)) for sample, context, text in rows]


def test_a_prompt_every_sample_shared_renders_once_naming_them_all():
    """The ordinary case, and the one the whole redesign rests on: a cohort is one world and
    one seed set, so a context's prompt is usually byte-identical across every sample.

    Measured on the reference port's own 18-sample run, four of five contexts are exactly this
    — 125,586 characters of duplicated prompt collapsing to 6,977 — and the summary says `every
    sample` rather than listing eighteen names, because the fact worth reading is that they
    agreed."""
    variants = report.prompt_variants(
        _pairs(
            ("sample 1", "state-classifier", "pick the state"),
            ("sample 2", "state-classifier", "pick the state"),
            ("sample 3", "state-classifier", "pick the state"),
        ),
        total=3,
    )

    assert len(variants) == 1, "one distinct text is one variant, however many samples read it"
    rendered = report.render_prompt_variants(variants)
    assert rendered.count("pick the state") == 1, "the text renders exactly once"
    assert "every sample" in rendered
    assert "sample 1" not in rendered, "naming them adds nothing when they all agree"


def test_prompts_that_differ_each_render_verbatim_naming_their_own_samples():
    """The finding a whole render makes visible.  A `chat` prompt carries the self-state header,
    which feeds each sample its OWN minted collection name back — so on the reference port all 18
    were distinct, and that is the cohort's `container name` variance showing up in the prompt.

    Each distinct text renders WHOLE, so a reader opening a sample's prompt reads what the model
    read — never a block assembled out of what several prompts have in common, which is a text no
    sample was given."""
    variants = report.prompt_variants(
        _pairs(
            ("sample 1", "chat", "you are penny\ncollection: alpha"),
            ("sample 2", "chat", "you are penny\ncollection: beta"),
        ),
        total=2,
    )

    rendered = report.render_prompt_variants(variants)
    assert len(variants) == 2
    assert "collection: alpha" in rendered and "collection: beta" in rendered
    assert "sample 1" in rendered and "sample 2" in rendered
    assert rendered.count("you are penny") == 2, "each prompt is whole, never a diff"


def test_shared_prompts_sort_ahead_of_the_ones_that_differ():
    """A reader meets what every sample had in common before what set them apart, whatever
    order the samples happened to deposit their prompts in."""
    rendered = report.render_prompt_variants(
        report.prompt_variants(
            _pairs(
                ("sample 1", "chat", "own text one"),
                ("sample 2", "chat", "own text two"),
                ("sample 1", "framer", "identical"),
                ("sample 2", "framer", "identical"),
            ),
            total=2,
        )
    )
    assert rendered.index("identical") < rendered.index("own text one")


def test_a_sample_naming_one_context_twice_is_counted_once():
    """A context is an ACTOR, and a sample is handed one prompt per actor — so a repeated row
    for the same text must not make a sample look like two, which would stop a genuinely shared
    prompt from reading as shared."""
    variants = report.prompt_variants(
        _pairs(
            ("sample 1", "chat", "same"),
            ("sample 1", "chat", "same"),
            ("sample 2", "chat", "same"),
        ),
        total=2,
    )
    assert variants[0].samples == ["sample 1", "sample 2"]
    assert variants[0].shared_by_all, "two samples, both accounted for"


def test_the_tail_states_the_inputs_and_what_the_outliers_did():
    """What closes a case: the inputs it was given as TABLES, then the outlier divergences.

    Declarative labels, not prose sentences — a heading names the thing and a reader scanning a
    hundred cases reads labels. The phrasings are a table because the reason they sit together is
    to be read against each other."""
    outlier = cohort.SampleStanding(
        name="c-3",
        phrasing="phrasing 2",
        standing=cohort.Standing.OUTLIER,
        shape="x",
        divergences=[cohort.FeatureDivergence(feature="tool sequence", value="a", modal="b")],
    )
    ground = worlds.World(
        name="one page",
        pages=(CannedPage(match="foxes", text="Title: Foxes\nBrandt signs.\n"),),
        keeps=(("brandt",),),
        excludes=(),
    )
    tail = report.render_case_tail(
        arms=[
            cohort.Arm(label="phrasing 1", text="watch the two pages", world=ground),
            cohort.Arm(label="phrasing 2", text="keep an eye on them", world=ground),
        ],
        outliers=[(3, outlier)],
    )

    assert "#### Test inputs" in tail
    assert "<summary>Phrasings — 2 wordings of one ask</summary>" in tail
    assert "| # | ask |" in tail, "the wordings are a table, not stacked paragraphs"
    assert "| phrasing 1 | watch the two pages |" in tail
    assert "#### Outliers" in tail
    assert "| `tool sequence` | `a` | `b` |" in tail
    assert "Which samples to read" not in tail, "the outlier section already indexes the work"


def test_a_case_declaring_nothing_shared_closes_with_nothing():
    """Every part of the tail is optional, so an unported case — or one driven with a single
    wording against no declared world — closes with nothing at all."""
    assert report.render_case_tail() == ""


# ── The three sections (#1997) ──────────────────────────────────────────────


def _rows(*specs: tuple[str, int, int, str]) -> list[cohort.AssertionRow]:
    return [
        cohort.AssertionRow(
            label=label, passed=passed, total=total, kind=kind, category=cohort.SpecCategory.STORE
        )
        for label, passed, total, kind in specs
    ]


def _observation(name: str, arm: str, tools: list[str], reply: str = "ok"):
    return cohort.SampleObservation(
        name=name, phrasing=arm, landed="learn", tool_sequence=tools, reply=reply
    )


def test_every_claim_renders_the_same_way_and_none_carries_a_floor():
    """Assertions are DETERMINISTIC checks, so none of them is gated and none proposes a floor.

    A reply claim is no longer a special case: the gated/ungated split existed only to decide
    which claims could carry a threshold, so with none to carry it renders and colours exactly
    like a claim read out of the store."""
    rendered = report.CaseSections(
        case_id="c",
        model="m",
        assertions=_rows(
            ("state: the machine landed in learn", 3, 3, "state"),
            ("state: nothing excluded was stored", 2, 3, "state"),
            ("reply: it names what this world says", 3, 3, "reply"),
        ),
    ).render()

    assert "|  | assertion | held | rate |" in rendered, "the floor column is gone"
    assert "proposed floor" not in rendered
    assert "| 🟢 | state: the machine landed in learn | 3/3 | 1.00 |" in rendered
    assert "| 🟡 | state: nothing excluded was stored | 2/3 | 0.67 |" in rendered
    assert "| 🟢 | reply: it names what this world says | 3/3 | 1.00 |" in rendered, (
        "a reply claim colours on its own rate like any other"
    )
    assert "8/9 checks · 89%" in rendered, "one reading over every check the case made"
    assert "nothing on the assertion side fails a run" in rendered, (
        "the report must SAY it is reported rather than enforced"
    )


def test_the_variance_section_reports_the_spread_and_names_the_wording_that_moved_it():
    """The pooled number is the gate; the per-phrasing rows say which wording moved it.  Only a
    FLAGGED row renders — a wording agreeing with its neighbours is the ordinary case."""
    samples = [
        _observation("c-1 (phrasing 1)", "phrasing 1", ["browse"]),
        _observation("c-2 (phrasing 1)", "phrasing 1", ["browse"]),
        _observation("c-3 (phrasing 2)", "phrasing 2", ["browse", "browse"]),
    ]
    rendered = report.CaseSections(
        case_id="c",
        model="gpt",
        variance=cohort.pool(samples, [cohort.TOOL_SEQUENCE]),
    ).render()

    assert "| `tool sequence` | 2 | 2/3 (0.67) |" in rendered
    assert "@ gpt N=3" in rendered, "a ceiling names the model and the N it was measured at"
    assert "`browse → browse`" in rendered, "the wording that produced a value no other did"


def test_a_feature_that_read_nothing_renders_red_and_says_so_rather_than_as_agreement():
    """A blind feature and a cohort in perfect agreement both compute to 0.000, and one is the
    best result a feature can report while the other is the absence of a measurement.

    So it is RED and the ceiling column says what happened — grey would file it beside the
    honest readings, which is exactly the confusion it causes.  The case's own line goes red
    with it: measurement blindness should be visible from the run header, not on the third
    click."""
    samples = [_observation(f"c-{n} (phrasing 1)", "phrasing 1", []) for n in (1, 2, 3)]
    rendered = report.CaseSections(
        case_id="c", model="gpt", variance=cohort.pool(samples, [cohort.TOOL_SEQUENCE])
    ).render()

    assert "| 🔴 | `tool sequence` | 1 | 3/3 (1.00) | 0.000 | " in rendered
    assert "READ NOTHING on every sample; not a reading" in rendered
    assert "@ gpt N=3" not in rendered, "a blind feature proposes no ceiling"


def test_the_harness_section_names_the_dead_samples_and_their_dominant_class():
    """Read FIRST even though it renders last: a cohort that lost half its samples makes the
    other two sections a description of whatever survived.

    The dominant class is computed from THIS cohort's own exclusions — the run-level fault tally
    is per process and cannot say which case a fault landed in, and a second accounting of the
    same samples is a second number to disagree with the first."""
    dead = [
        cohort.SampleObservation(
            name=f"c-{n} (phrasing 1)", phrasing="phrasing 1", complete=False, exclusion="no turn"
        )
        for n in (2, 3)
    ]
    variance = cohort.pool(
        [_observation("c-1 (phrasing 1)", "phrasing 1", ["browse"]), *dead], [cohort.TOOL_SEQUENCE]
    )
    rendered = report.CaseSections(case_id="c", model="m", variance=variance).render()

    assert "1 pooled + 2 excluded = 3 driven" in rendered
    assert "Dominant failure class: **no turn** (2 of 2)." in rendered
    assert "- `c-2 (phrasing 1)` — no turn" in rendered
    assert "- `c-3 (phrasing 1)` — no turn" in rendered


def test_one_modal_sample_is_named_and_every_other_shape_is_an_outlier():
    """Exactly ONE sample is modal, because the workflow asks a human to read one and naming
    eight equally puts the choice straight back on them.

    Only that one is OPENED.  An outlier is communicated by its divergence — the feature, its
    value, the representative's — which is a few rows against a whole transcript; expanding
    fifteen of them is what made one case's report 787,681 characters."""
    samples = [
        _observation("c-1 (a)", "a", ["browse"]),
        _observation("c-2 (a)", "a", ["browse"]),
        _observation("c-3 (b)", "b", ["browse", "write"]),
        cohort.SampleObservation(name="c-4 (b)", phrasing="b", complete=False, exclusion="no turn"),
    ]
    standings = cohort.standings(samples, [cohort.TOOL_SEQUENCE])

    # A DEAD sample has no shape to be typical of — it is the harness section's business rather
    # than a reading recommendation, so it is never offered as one.
    assert [s.standing for s in standings] == [
        cohort.Standing.MODAL,
        cohort.Standing.TYPICAL,
        cohort.Standing.OUTLIER,
        cohort.Standing.DEAD,
    ]
    assert [s.worth_opening for s in standings] == [True, False, False, False]


def test_a_sample_agreeing_on_one_feature_and_not_another_is_an_outlier():
    """Standing reads the sample's WHOLE shape, not any single feature — a sample matching the
    majority tool sequence while minting its own routine shape did something else, and reading
    one feature at a time would file it under the majority and fold it away."""
    samples = [
        cohort.SampleObservation(
            name="c-1 (a)",
            phrasing="a",
            tool_sequence=["browse"],
            routines=[cohort.RoutineRecord(name="r", shape="browse", names_a_destination=True)],
        ),
        cohort.SampleObservation(
            name="c-2 (a)",
            phrasing="a",
            tool_sequence=["browse"],
            routines=[cohort.RoutineRecord(name="r", shape="other", names_a_destination=True)],
        ),
    ]
    features = [cohort.TOOL_SEQUENCE, cohort.ROUTINE_SHAPE]
    assert [s.standing for s in cohort.standings(samples, features)] == [
        cohort.Standing.MODAL,
        cohort.Standing.OUTLIER,
    ]


# ── The three sections, whole ────────────────────────────────────────────────
#
# Whole-render, because what a reader takes a threshold from is the rendered document — and a
# number that renders without the model and the N it was measured at is a number somebody will
# compare across both.

_MODEL = "openai/gpt-oss-20b"
_FLOOR_NOTE = (
    "_These are DETERMINISTIC checks — things expected to be strictly true of the run — so they "
    "carry no floors and none is proposed: a threshold under something expected at 100% either "
    "never fires, or sits below the observed rate and blesses the defect as the contract. The "
    "reading above is REPORTED, not enforced, and nothing on the assertion side fails a run. "
    "VARIANCE is what gates, under one-sided ceilings; a dead cohort still fails on run health._"
)
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


def _shaped(name: str, arm: str, shape: str) -> cohort.SampleObservation:
    return cohort.SampleObservation(
        name=name,
        phrasing=arm,
        landed="learn",
        routines=[cohort.RoutineRecord(name="r", shape=shape, names_a_destination=True)],
        reply=f"done, {shape}",
    )


def test_the_three_sections_render_whole():
    """Every part of the case's score, in the order it is read, as one literal.

    The DEFAULT view is the first line and nothing else — that is what makes ~100 cases a
    document a person pages through — so the three sections are each behind a fold whose summary
    carries that section's worst state."""
    samples = [
        _shaped("case-1 (phrasing 1)", "phrasing 1", "browse → write"),
        _shaped("case-2 (phrasing 1)", "phrasing 1", "browse → write"),
        _shaped("case-3 (phrasing 2)", "phrasing 2", "browse → browse → write"),
        cohort.SampleObservation(
            name="case-4 (phrasing 2)",
            phrasing="phrasing 2",
            complete=False,
            exclusion="the measured turn never ran",
        ),
    ]
    rendered = report.CaseSections(
        case_id="memory-learn-close-shape",
        model=_MODEL,
        assertions=[
            cohort.AssertionRow(
                label="state: the round taught a routine",
                passed=3,
                total=3,
                category=cohort.SpecCategory.STORE,
            ),
            cohort.AssertionRow(
                label="reply: every specific value in it is sourced",
                passed=2,
                total=3,
                kind="reply",
                category=cohort.SpecCategory.PROVENANCE,
            ),
        ],
        variance=cohort.pool(samples, [cohort.ROUTINE_SHAPE, cohort.REPLY_SPREAD]),
        cost=cohort.per_sample_cost(
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
            # The case NAMES the report, and the topline readings are the TABLE under it —
            # no sentence restating what the rows already say. The assertion side is ONE
            # number over every check the case made, 5 of 6, since none of them is gated.
            "### 🔴 `memory-learn-close-shape` — `openai/gpt-oss-20b`",
            "\n".join(
                [
                    "| measure | reading |",
                    "|---|---|",
                    "| **checks** | 🟡 5 / 6 · 83% "
                    "(lowest 🟡 0.67 `reply: every specific value in it is sourced`) |",
                    "| **variance** | ⚪ max H 0.579 `routine shape` · 1 of 1 features vary |",
                    "| **samples** | 3 pooled + 1 excluded = 4 driven |",
                    "| **cost / sample** | 40,000 in · 10,000 out (80% thinking) · "
                    "12.0 calls · 200s |",
                ]
            ),
            report.titled_fold(
                "🟡 Assertions",
                "5/6 checks · 83% · lowest 0.67 `reply: every specific value in it is sourced`",
                "\n\n".join(
                    [
                        # A category nobody wrote a claim for renders as a GAP, not a blank —
                        # this case checks nothing of that kind, which is the finding.
                        "**landed** — _no claim. This case asserts nothing in this category._",
                        "**store**",
                        "\n".join(
                            [
                                "|  | assertion | held | rate |",
                                "|---|---|---|---|",
                                "| 🟢 | state: the round taught a routine | 3/3 | 1.00 |",
                            ]
                        ),
                        "**provenance**",
                        "\n".join(
                            [
                                "|  | assertion | held | rate |",
                                "|---|---|---|---|",
                                # A reply claim is counted and coloured like every other now:
                                # 2 of 3 is amber on the ordinary scale, not grey.
                                "| 🟡 | reply: every specific value in it is sourced | 2/3 | "
                                "0.67 |",
                            ]
                        ),
                        _FLOOR_NOTE,
                    ]
                ),
            ),
            report.titled_fold(
                "⚪ Variance",
                "1 feature · max H 0.579 `routine shape` · 1 of 1 features vary",
                "\n\n".join(
                    [
                        "\n".join(
                            [
                                "|  | feature | distinct | modal | entropy | proposed ceiling |",
                                "|---|---|---|---|---|---|",
                                "| ⚪ | `routine shape` | 2 | 2/3 (0.67) | 0.579 | "
                                "`0.68` @ openai/gpt-oss-20b N=3 |",
                            ]
                        ),
                        _CEILING_NOTE,
                        _PHRASING_LEAD,
                        "\n".join(
                            [
                                "| feature | phrasing | distinct | only under this wording |",
                                "|---|---|---|---|",
                                # BOTH wordings are flagged, and symmetrically so: each produced
                                # a value the other did not — the finding the pooled 0.579 hides.
                                "| `routine shape` | phrasing 1 | 1/2 | `browse → write` |",
                                "| `routine shape` | phrasing 2 | 1/1 | "
                                "`browse → browse → write` |",
                            ]
                        ),
                        "Reply text over 3 pairs — cosine mean 0.000 min 0.000 · "
                        "containment mean 1.000",
                    ]
                ),
            ),
            report.titled_fold(
                "⚪ Cost",
                "40,000 in · 10,000 out (80% thinking) · 12.0 calls · 200s",
                "\n\n".join(
                    [
                        "**Cost, per sample.**",
                        "\n".join(
                            [
                                "| tokens | observed | proposed ceiling |",
                                "|---|---|---|",
                                "| input tokens (ours — prompt and context) | 40,000 | "
                                "`44,000` @ openai/gpt-oss-20b |",
                                "| output tokens (the model's) | 10,000 | "
                                "`11,000` @ openai/gpt-oss-20b |",
                            ]
                        ),
                        "Also per sample: 12.0 calls · 200s · 8,000 reasoning tokens "
                        "(80% of output).",
                        _COST_NOTE,
                    ]
                ),
            ),
            report.titled_fold(
                "🔴 Excluded samples",
                "1 of 4 · dominant: the measured turn never ran",
                "\n\n".join(
                    [
                        "Dominant failure class: **the measured turn never ran** (1 of 1).",
                        "- `case-4 (phrasing 2)` — the measured turn never ran",
                    ]
                ),
            ),
        ]
    )


def test_a_case_whose_cohort_all_agreed_says_so_rather_than_rendering_an_empty_table():
    """The quiet path: nothing excluded, no phrasing outlier, no assertions declared — each
    absence stated in words rather than left as a table with no rows."""
    rendered = report.CaseSections(
        case_id="quiet",
        model=_MODEL,
        variance=cohort.pool(
            [_shaped(f"s{i}", "phrasing 1", "browse → write") for i in range(3)],
            [cohort.ROUTINE_SHAPE],
        ),
    ).render()
    assert "_No phrasing produced a value the others did not._" in rendered
    assert "3 pooled + 0 excluded = 3 driven" in rendered
    assert "_(no assertions)_" in rendered


# ── Internal seams for an oversized sample fold (#1917) ──────────────────────


def _oversized_sample() -> tuple[int, str, str]:
    """A sample whose single fold cannot be posted: many DISTINCT step tables, which is the
    shape that defeats every compression the report has (a degenerate reroll's 25 distinct
    thinking traces, measured at 71,350 chars against a 65,536 cap).  Each block is unique,
    so nothing here could be hoisted or deduped away."""
    steps = [
        report.Step(
            number=index,
            user_message=f"turn {index}",
            verdict="❌",
            rows=[report.Row(f"row {index}", f"unique body {index} " + "x" * 400, [])],
        )
        for index in range(1, 41)
    ]
    body = report.BLOCK_SEPARATOR.join(step.render() for step in steps)
    return 3, "❌ fail · behavioral · 206s · 31 calls", body


def test_an_oversized_sample_renders_as_folds_the_splitter_can_cut_between() -> None:
    """A fold too big to post becomes SEVERAL, each opening on the sample seam — so the
    splitter gains legal cut points inside what used to be one atom, and needs to know
    nothing about parts.  Each part is complete markup on its own."""
    number, banner, body = _oversized_sample()
    budget = 6000
    rendered = report.fold_sample_parts(number, banner, body, budget)
    parts = report.split_sample_blocks(rendered)

    assert len(parts) > 1, "an oversized sample must be given internal seams"
    assert all(len(part) <= budget for part in parts), "every part must fit the budget"
    for index, part in enumerate(parts, start=1):
        recovered_number, recovered_banner, _ = report.parse_sample_block(part)
        assert recovered_number == number
        assert recovered_banner.endswith(
            report.SAMPLE_PART_SUFFIX.format(number=index, total=len(parts))
        )


def test_the_split_parts_reassemble_byte_identical_to_the_unsplit_fold() -> None:
    """THE LOSSLESS CLAIM: concatenating the parts' bodies reproduces the single fold's body
    byte for byte.

    Nothing is truncated, summarised or re-wrapped — the rule the whole report is built on
    (collapsed never means removed), asserted across the seam rather than assumed of it.
    A seam that dropped a separator, reordered a block or ate a blank line would show here
    and nowhere else."""
    number, banner, body = _oversized_sample()
    parts = report.split_sample_blocks(report.fold_sample_parts(number, banner, body, 6000))
    bodies = [report.parse_sample_block(part)[2] for part in parts]
    assert report.BLOCK_SEPARATOR.join(bodies) == body
    assert report.fold_sample(number, banner, report.BLOCK_SEPARATOR.join(bodies)) == (
        report.fold_sample(number, banner, body)
    )


def test_a_sample_that_fits_is_rendered_exactly_as_it_always_was() -> None:
    """The common case is untouched: a fold within budget comes back byte-identical to
    ``fold_sample``, with no part suffix and no extra seam.  Every run that could already
    be posted posts the same way."""
    number, banner, body = _oversized_sample()
    assert report.fold_sample_parts(number, banner, body, 500_000) == report.fold_sample(
        number, banner, body
    )


def test_body_blocks_inverts_the_join_it_is_the_inverse_of() -> None:
    """The seam finder recovers exactly the blocks the body was built from — including a
    system-prompt fold, whose own body carries the blank lines that make a naive split on
    ``\n\n`` wrong."""
    prompt = report.SystemPrompt(context="collector", text="line one\n\nline two")
    step = report.Step(number=1, user_message="hi", verdict="✅", rows=[])
    close = report.RunClose(score="1/1", rows=[])
    blocks = [prompt.render(), step.render(), close.render()]
    body = report.BLOCK_SEPARATOR.join(blocks)
    assert report.body_blocks(body) == blocks


def test_a_single_block_over_budget_keeps_its_own_fold_for_the_guard_to_refuse() -> None:
    """The seam rule outranks the budget, exactly as it does in the splitter: one step too
    big to fit gets its own over-budget fold rather than being cut mid-table.

    That is what leaves the guard something to refuse — a sample whose smallest indivisible
    piece exceeds the cap genuinely cannot be split losslessly, and saying so is the
    contract."""
    huge = report.Step(
        number=1, user_message="x", verdict="", rows=[report.Row("a", "y" * 9000, [])]
    )
    small = report.Step(number=2, user_message="x", verdict="", rows=[report.Row("a", "z", [])])
    body = report.BLOCK_SEPARATOR.join([huge.render(), small.render()])
    parts = report.split_sample_blocks(report.fold_sample_parts(7, "banner", body, 2000))
    assert len(parts) == 2
    assert max(len(part) for part in parts) > 2000
    assert report.BLOCK_SEPARATOR.join(report.parse_sample_block(part)[2] for part in parts) == body


def test_the_harness_counts_add_up():
    """The arithmetic must CLOSE. This is the one surface whose job is to say whether a run can
    be believed, so a pooled count short of the driven one has to name what it lost — a sample it
    cannot account for is how infrastructure failure gets read as behaviour."""
    samples = [
        _observation("c-1 (a)", "a", ["browse"]),
        _observation("c-2 (a)", "a", ["browse"]),
        cohort.SampleObservation(name="c-3 (a)", phrasing="a", complete=False, exclusion="no turn"),
    ]
    variance = cohort.pool(samples, [cohort.TOOL_SEQUENCE])
    assert (variance.pooled, variance.driven) == (2, 3)

    rendered = report.CaseSections(case_id="c", model="m", variance=variance).render()
    assert "2 pooled + 1 excluded = 3 driven" in rendered


def test_the_seeded_world_states_its_own_counts_not_its_renders():
    """The counts a closed fold shows come off the WORLD, so the label and the table it summarises
    cannot disagree — a summary derived from the markdown it summarises is the same mistake as
    diffing rendered prompts, and it silently read zero pages off a two-page world."""
    from penny.tests.eval.utils.worlds import TWO_TEAM_NEWS

    assert TWO_TEAM_NEWS.counts == (2, 6, 2)
    tail = report.render_case_tail(
        arms=[cohort.Arm(label="phrasing 1", text="watch them", world=TWO_TEAM_NEWS)]
    )
    assert (
        "#### Test inputs\n\n"
        "<details><summary>1 phrasing · 2 pages · 6 must-keep, 2 must-not</summary>" in tail
    )
    assert "Seeded pages — 2 pages · 6 must-keep, 2 must-not" in tail
    assert "| 1 | `ridgelinefoxes` |" in tail, "and the table it labels is right there"


def _named(name: str, arm: str, tools: list[str], routine: str):
    return cohort.SampleObservation(
        name=name,
        phrasing=arm,
        tool_sequence=tools,
        routines=[cohort.RoutineRecord(name=routine, shape="s", names_a_destination=True)],
    )


def test_a_feature_every_sample_differs_on_is_not_a_divergence():
    """When 14 of 15 samples are outliers, `outlier` has stopped meaning anything — and the cause
    is a feature at maximum entropy. Every sample's framer draws its own routine name, so every
    diverges on it by construction, so every sample is an outlier and the section names all of
    them and therefore none.

    That fact belongs to the FEATURE and is already stated once in the variance table; it is not
    a finding about any one sample. So it is excluded from standing and from the divergences, and
    a sample whose only difference was the name it drew is not an outlier at all."""
    samples = [_named(f"c-{n}", "a", ["browse"], f"watch_price_{n}") for n in range(4)]
    # …except one, which also took a different route — the only real divergence here.
    samples[3] = _named("c-3", "a", ["browse", "write"], "watch_price_3")
    features = [cohort.ROUTINE_NAME, cohort.TOOL_SEQUENCE]

    assert cohort.everywhere_distinct(samples, features) == ["routine name"]
    standings = cohort.standings(samples, features)
    assert [s.standing for s in standings] == [
        cohort.Standing.MODAL,
        cohort.Standing.TYPICAL,
        cohort.Standing.TYPICAL,
        cohort.Standing.OUTLIER,
    ], "naming the container differently is not what makes a sample worth reading"
    assert [d.feature for d in standings[3].divergences] == ["tool sequence"]


def test_the_outlier_section_states_the_uniform_spread_once():
    """Stated once, above the divergences, rather than repeated under every sample as a finding
    about that sample."""
    standings = [
        (n, cohort.SampleStanding(name=f"c-{n}", phrasing="a", standing=st, shape="x"))
        for n, st in enumerate([cohort.Standing.MODAL, cohort.Standing.TYPICAL], start=1)
    ]
    rendered = report.render_outliers(standings, ["container name"], pooled_count=15)
    assert "All 15 pooled samples chose a distinct `container name`" in rendered
    assert "reported once in Variance" in rendered


def test_the_representative_fold_carries_no_verdict():
    """A sample does not pass or fail — the cohort's assertions hold or they do not, which is the
    whole reason the cohort is the unit of scoring. `✅ pass` was `1/1 (1.00)` respelled."""
    banner = report.render_banner(passed=True, fragile=True, duration_s=17, calls=11)
    summary = report.representative_summary(banner, turns=2)

    assert summary == "2 turns · fragile · 17s · 11 calls"
    assert "pass" not in summary and "fail" not in summary
    assert (
        report.without_verdict("❌ fail · behavioral · 3s · 1 calls") == "behavioral · 3s · 1 calls"
    )


def test_a_thinking_trace_shorter_than_its_own_label_renders_inline():
    """A fold whose contents are shorter than its summary asks for a click to reveal less text
    than the click target."""
    assert report.thinking_row("brief").render() == "| 💭 | brief |  |"
    assert "<details>" in report.thinking_row("x" * 200).render()


def test_every_spec_category_renders_including_the_empty_ones():
    """The design permits exactly THREE kinds of deterministic assertion, and a category nobody
    wrote a claim for is a FINDING rather than a blank: it says this case checks nothing of that
    kind, which is what a reader porting the next case needs to see."""
    rendered = report.CaseSections(
        case_id="c",
        model="m",
        assertions=[
            cohort.AssertionRow(
                label="state: the machine landed in learn",
                passed=3,
                total=3,
                category=cohort.SpecCategory.LANDED,
            )
        ],
    ).render()

    assert "**landed**" in rendered
    for missing in ("store", "provenance"):
        assert f"**{missing}** — _no claim." in rendered, f"{missing} must render as a gap"


def test_a_cosmetic_divergence_never_makes_an_outlier():
    """A feature unconstrained in BOTH measured models is a system-level finding for the
    variance table, not fifteen per-sample findings. Without the split, 8 of 9 outlier rows were
    routine-name-only — reporting "60% of samples are outliers" where the true statement was
    "1 of 15 diverged consequentially"."""
    # Two samples draw the same routine name, so the feature is INFORMATIVE (not all-distinct)
    # and reaches the consequence rule rather than being dropped before it.
    samples = [
        cohort.SampleObservation(
            name=f"c-{n}",
            phrasing="a",
            tool_sequence=["browse"],
            routines=[cohort.RoutineRecord(name=routine, shape="s", names_a_destination=True)],
        )
        for n, routine in enumerate(["watch_price", "watch_price", "monitor_price", "check_price"])
    ]
    features = [cohort.ROUTINE_NAME, cohort.TOOL_SEQUENCE]
    standings = cohort.standings(samples, features)

    assert all(s.standing is not cohort.Standing.OUTLIER for s in standings), (
        "the name the framer drew is cosmetic — it makes no outlier"
    )
    rendered = report.render_outliers(list(enumerate(standings, start=1)))
    assert "No sample diverged consequentially" in rendered
    assert "2 samples differ on `routine name`" in rendered
    assert "entropy reported in Variance" in rendered


# ── Quoted content is EVIDENCE, never structure ──────────────────────────────
_HOSTILE_PROMPT = (
    "## Penny's current state\n"
    "### Active mechanisms\n"
    "**bold** and `code` and - a bullet\n"
    "1. a numbered line"
)


def test_a_quoted_prompt_cannot_inject_a_heading_into_the_document():
    """A prompt is text the model RECEIVED, so rendering it shows the reader something the
    model was never handed — and its headings land in the report's own outline.

    Measured on a real run: `## Penny's current state` from the system prompt OUTRANKED the
    case heading, so the document's structure was written by the content it quotes."""
    rendered = report.SystemPrompt(context="chat", text=_HOSTILE_PROMPT).render()
    # Everything the fence encloses is literal; only what is OUTSIDE it can become structure.
    outside = rendered.split("```")[0] + rendered.split("```")[-1]
    assert not any(line.startswith("#") for line in outside.splitlines())
    for line in _HOSTILE_PROMPT.splitlines():
        assert line in rendered, "and every character survives, byte for byte"


def test_a_prompt_containing_a_code_fence_still_closes_where_it_should():
    """The fence is grown past the longest backtick run inside, so quoted markup that itself
    contains a fence cannot end the block early and spill into the document."""
    fenced = report.fenced("before\n```\ninside\n```\nafter")
    assert fenced.startswith("````") and fenced.endswith("````")
    assert "```\ninside\n```" in fenced


def test_a_rejected_draw_is_quoted_verbatim_rather_than_rendered():
    """A draw is model output the loop threw away. It was previously run through the CELL
    escaper, which both rendered its markdown and flattened its newlines into `<br>` — so the
    one surface kept for diagnosis showed something other than what the model produced."""
    rendered = report.render_rejected_draws(["## a heading\nline two"])
    assert "<br>" not in rendered, "newlines survive"
    assert "\n## a heading\nline two\n" in rendered, "and the text is verbatim"
    outside_fence = rendered.split("```")[0]
    assert "## a heading" not in outside_fence
