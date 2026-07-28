"""Whole-render tests for the transcript-integrated report grammar (``report.py``, #1725/#1753).

NOT eval-marked — they drive the PURE renderer over hand-built ``SampleTranscript``s (no DB, no
model, no git), so they run inside ``make check`` and pin every form of the iteration-6 grammar
as a WHOLE-RENDER literal (pr-review-guide §6). Every sample folds whole under its banner now
(uniform collapse, #1753): the clean pass with per-context system-prompt rows (#1759) + all three
named micro-context actors (#1773), the failure with a nudge + run-close + n/a, the harness-timeout
placeholder, the diff-mode
regressed flip with a baseline row, and an advisory check + empty thinking on a fragile pass all
render inside a ``<details>``; plus the deterministic cell hygiene (single-copy collapsed
truncation + escaping, #1759) and the fold/parse seam the assembler's re-normalization rides on
(EVERY sample folds whole — the one and only rendering, no banner-only form).
"""

from __future__ import annotations

from penny.tests.eval import report


def test_clean_pass_folds_whole_with_system_prompts_and_micro_context() -> None:
    """A clean pass folds into one ``<details>``: its distinct per-context system prompts (#1759)
    render as always-collapsed rows directly under the banner (main agent then each micro-context),
    then EVERY micro-context call renders inline as its own named actor (🧩 <context> ← user turn: /
    →, #1759/#1773) with its own thinking — the state classifier at the head of the turn it decided,
    the browse extraction after the call that spawned it, the run-end skill labeller closing the
    turn — and an action with no captured thinking shows ``💭 (empty)``."""
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
    banner = report.render_banner(
        passed=True, score=1.0, passed_checks=2, total_checks=2, duration_s=45, calls=8
    )
    sample = report.build_sample(
        number=1,
        banner=banner,
        events=events,
        checks=checks,
        run_close_score="2/2",
        system_prompts=[
            report.SystemPrompt("chat", "You are Penny.\nAnswer from sources."),
            report.SystemPrompt("browse-extract", "You extract one value."),
        ],
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 1 — ✅ pass · 2/2 (1.00) · 45s · 8 calls</summary>\n"
        "\n"
        "<details><summary>system prompt — chat (35 chars)</summary>\n"
        "\n"
        "You are Penny.\n"
        "Answer from sources.\n"
        "\n"
        "</details>\n"
        "\n"
        "<details><summary>system prompt — browse-extract (22 chars)</summary>\n"
        "\n"
        "You extract one value.\n"
        "\n"
        "</details>\n"
        "\n"
        '| step 1 · 👤 | "deepest lake?" | ✅ |\n'
        "|---|---|---|\n"
        "| expected | C1 [spine]⚖ browsed |  |\n"
        "| expected | C2 [reply]⚖ reply names the fact |  |\n"
        "| actual | 🧩 state-classifier ← user turn: newest message: deepest lake? |  |\n"
        "| 💭 | <details><summary>thinking (state-classifier)</summary>a question, no task"
        "</details> |  |\n"
        "| actual | 🧩 state-classifier → STATE: idle |  |\n"
        "| 💭 | <details><summary>thinking</summary>verify with source</details> |  |\n"
        '| actual | 🔧 browse({"queries":["x"],"extract":"depth"}) | ✅ C1 |\n'
        "| actual | 🧩 browse-extract ← user turn: Instruction: depth · Content: 1,642 m |  |\n"
        "| 💭 | <details><summary>thinking (browse-extract)</summary>value present</details> |  |\n"
        "| actual | 🧩 browse-extract → EXTRACTED: 1642 |  |\n"
        "| actual | 📥 You opened wiki (browse result) · 1642 |  |\n"
        "| 💭 | 💭 (empty) |  |\n"
        "| actual | 🤖 Lake Baikal, 1,642 m. | ✅ C2 |\n"
        "| actual | 🧩 skill-namer ← user turn: steps: browse |  |\n"
        "| 💭 | <details><summary>thinking (skill-namer)</summary>generic name</details> |  |\n"
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
        score=0.5,
        passed_checks=1,
        total_checks=2,
        cause="behavioral",
        duration_s=120,
        calls=13,
    )
    sample = report.build_sample(
        number=3, banner=banner, events=events, checks=checks, run_close_score="1/2"
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 3 — ❌ fail · 1/2 (0.50) · "
        "behavioral · 120s · 13 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "drop the read step" | ❌ |\n'
        "|---|---|---|\n"
        "| expected | C7 [state]⚖ remove: read gone |  |\n"
        "| 💭 | <details><summary>thinking</summary>fold in once confirmed</details> |  |\n"
        "| actual | 🤖 I'll ditch that. Just to... |  |\n"
        "| actual | 👤 *(nudge)* Please provide your response. | ⚠ recovery event |\n"
        "| 💭 | <details><summary>thinking</summary>restate</details> |  |\n"
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
        score=0.0,
        passed_checks=0,
        total_checks=0,
        cause="harness",
        duration_s=118,
        calls=13,
        checks_evaluated=False,
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
        score=0.75,
        passed_checks=3,
        total_checks=4,
        cause="behavioral",
        duration_s=60,
        calls=5,
    )
    sample = report.build_sample(
        number=1, banner=banner, events=events, checks=checks, run_close_score="3/4"
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 1 — ❌ fail · 3/4 (0.75) · behavioral · 60s · 5 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "stop notifying me" | ✅→❌ |\n'
        "|---|---|---|\n"
        "| expected | C8 [state]⚖ notify off |  |\n"
        '| baseline | 🔧 collection_set({"notify":false}) → confirmed | ✅ C8 *(prior run)* |\n'
        "| 💭 | <details><summary>thinking</summary>defer</details> |  |\n"
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
        score=1.0,
        passed_checks=1,
        total_checks=1,
        fragile=True,
        duration_s=30,
        calls=4,
    )
    sample = report.build_sample(
        number=2, banner=banner, events=events, checks=checks, run_close_score="1/1"
    )
    assert report.render_sample(sample) == (
        "<details><summary>sample 2 — ✅ pass · 1/1 (1.00) · fragile · 30s · 4 calls</summary>\n"
        "\n"
        '| step 1 · 👤 | "add game and remind me friday" | ✅ |\n'
        "|---|---|---|\n"
        "| expected | C1 [state]⚖ entry written |  |\n"
        "| expected | C2 [spine]ℹ single-write efficiency |  |\n"
        "| 💭 | 💭 (empty) |  |\n"
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
    folded = report.fold_sample(2, "✅ pass · 1/1 (1.00) · 10s · 2 calls", body)
    assert folded == (
        "<details><summary>sample 2 — ✅ pass · 1/1 (1.00) · 10s · 2 calls</summary>\n"
        f"\n{body}\n\n"
        "</details>"
    )
    assert report.parse_sample_block(folded) == (2, "✅ pass · 1/1 (1.00) · 10s · 2 calls", body)
    heading = f"#### sample 3 — ❌ fail · behavioral · 120s · 5 calls\n\n{body}"
    assert report.parse_sample_block(heading) == (3, "❌ fail · behavioral · 120s · 5 calls", body)


def test_split_sample_blocks_separates_mixed_forms() -> None:
    """``split_sample_blocks`` splits a case transcript into its per-sample blocks in order, across
    a folded block followed by a legacy unfolded ``#### `` block (the re-assembly case)."""
    folded = report.fold_sample(1, "✅ pass · 1/1 (1.00) · 8s · 2 calls", "| a | b |  |")
    heading = "#### sample 2 — ❌ fail · harness · 120s · 3 calls\n\n_(no completed turns)_"
    transcript = f"{folded}\n\n{heading}\n\n"
    assert report.split_sample_blocks(transcript) == [folded, heading]
    assert report.split_sample_blocks("") == []


# ── Hoisting a case's shared system-prompt block (#1763) ────────────────────


def _prompt_block(context: str, text: str) -> str:
    return report.SystemPrompt(context=context, text=text).render()


_SHARED = "\n".join(f"shared line {n}" for n in range(40))


def _case(name: str, *blocks: str) -> str:
    return f"### `{name}` — fam\n\n" + "\n\nsample\n\n".join(blocks)


def test_the_shared_text_is_hoisted_and_only_the_delta_stays():
    """A case's chat prompts differ at BOTH ends — a timestamp opens them, the
    live self-state closes them — so the shared part is a middle, not a prefix.
    It renders once under the case heading; each sample keeps only the lines that
    are genuinely its own, with a marker standing where the shared text sits, so
    the whole prompt is still reconstructable and the DIFFERENCE is what you
    read (median ~120 bytes on a real run, against a 6.4K wall before)."""
    a = _prompt_block("chat", f"time A\n{_SHARED}\nstate A")
    b = _prompt_block("chat", f"time B\n{_SHARED}\nstate B")
    hoisted = report.hoist_shared_prompt_blocks(_case("beat0", a, b))

    assert report.SHARED_PROMPT_HEADING in hoisted
    assert hoisted.count(_SHARED) == 1, "the shared text renders exactly once"
    assert hoisted.count(report.SHARED_PROMPT_MARKER) == 2, "each sample points at it"
    for unique in ("time A", "time B", "state A", "state B"):
        assert unique in hoisted, f"{unique} — a sample's own text is never dropped"


def test_an_identical_prompt_leaves_a_pure_reference():
    """When the shared block IS the whole prompt — a classifier run, byte-identical
    every sample — the remainder is empty and the row is just the marker.  Same
    mechanism, no special case."""
    block = _prompt_block("state-classifier", _SHARED)
    hoisted = report.hoist_shared_prompt_blocks(_case("beat0", block, block, block))
    assert hoisted.count(_SHARED) == 1
    assert hoisted.count(report.SHARED_PROMPT_MARKER) == 3


def test_hoisting_is_per_case_so_a_case_section_stays_self_contained():
    """Per CASE, not per run: a case's comment has to stand alone once the
    document is split to fit the comment cap, so each case carries its own
    shared block even when two cases happen to share the text."""
    block = _prompt_block("chat", f"t\n{_SHARED}\ns")
    document = _case("beat0-one", block, block) + "\n\n" + _case("beat0-two", block, block)
    hoisted = report.hoist_shared_prompt_blocks(document)
    assert hoisted.count(report.SHARED_PROMPT_HEADING) == 2
    assert hoisted.count(_SHARED) == 2, "one per case — each section stands alone"


def test_shared_lines_are_taken_wherever_they_fall_not_just_the_longest_run():
    """Every shared line hoists, not merely the longest contiguous run of them.
    A sample that gained a line mid-prompt (it created a collection, so its
    self-state grew) splits the shared text in two — taking only the longest run
    left the other half inline on exactly the cases that needed it most."""
    head = "\n".join(f"top {n}" for n in range(20))
    tail = "\n".join(f"bottom {n}" for n in range(20))
    a = _prompt_block("chat", f"{head}\nONLY-A\n{tail}")
    b = _prompt_block("chat", f"{head}\n{tail}")
    hoisted = report.hoist_shared_prompt_blocks(_case("beat0", a, b))

    assert hoisted.count("top 0") == 1, "the run before the gap hoists"
    assert hoisted.count("bottom 0") == 1, "and so does the run after it"
    assert "ONLY-A" in hoisted, "the line that made them differ stays with its sample"


def test_a_single_sample_and_a_tiny_overlap_are_left_alone():
    """Nothing to share (one sample) or nothing worth sharing (a block smaller
    than the markup that would reference it) leaves the section byte-identical —
    the saving has to be real, which is a derived condition, not a tuned number."""
    lone = _case("beat0", _prompt_block("chat", f"only\n{_SHARED}"))
    assert report.hoist_shared_prompt_blocks(lone) == lone

    tiny = _case("beat0", _prompt_block("chat", "a\nx"), _prompt_block("chat", "b\nx"))
    assert report.hoist_shared_prompt_blocks(tiny) == tiny


def test_a_single_case_run_still_hoists():
    """A single-case report renders no `### case` heading — the assembler omits
    it — but its samples repeat their prompt just as hard, so the whole document
    is one section and hoisting still applies."""
    block = _prompt_block("chat", f"t\n{_SHARED}\ns")
    document = f"run header line\n\n{block}\n\nsample one\n\n{block}\n\nsample two"
    hoisted = report.hoist_shared_prompt_blocks(document)
    assert report.SHARED_PROMPT_HEADING in hoisted
    assert hoisted.count(_SHARED) == 1
    assert hoisted.startswith("run header line"), "the header still opens the document"
