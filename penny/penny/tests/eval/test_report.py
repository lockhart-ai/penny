"""Whole-render tests for the transcript-integrated report grammar (``report.py``, #1725/#1753).

NOT eval-marked — they drive the PURE renderer over hand-built ``SampleTranscript``s (no DB, no
model, no git), so they run inside ``make check`` and pin every form of the iteration-6 grammar
as a WHOLE-RENDER literal (pr-review-guide §6). Every sample folds whole under its banner now
(uniform collapse, #1753): the clean pass with per-context system-prompt rows (#1759) + micro-
context, the failure with a nudge + run-close + n/a, the harness-timeout placeholder, the diff-mode
regressed flip with a baseline row, and an advisory check + empty thinking on a fragile pass all
render inside a ``<details>``; plus the deterministic cell hygiene (single-copy collapsed
truncation + escaping, #1759) and the fold/parse seam the assembler's re-normalization rides on
(EVERY sample folds whole — the one and only rendering, no banner-only form).
"""

from __future__ import annotations

from penny.tests.eval import report


def test_clean_pass_folds_whole_with_system_prompts_and_micro_context() -> None:
    """A clean pass folds into one ``<details>``: its distinct per-context system prompts (#1759)
    render as always-collapsed rows directly under the banner (main agent then the browse-extract
    micro-context), then a browse call's micro-context (🧩 ← user turn: / →, #1759) renders inline
    with its own thinking, and an action with no captured thinking shows ``💭 (empty)``."""
    events = [
        report.Event(report.EventKind.USER, "deepest lake?"),
        report.Event(
            report.EventKind.CALL,
            'browse({"queries":["x"],"extract":"depth"})',
            thinking="verify with source",
        ),
        report.Event(report.EventKind.MICRO_IN, "Instruction: depth · Content: 1,642 m"),
        report.Event(report.EventKind.MICRO_OUT, "EXTRACTED: 1642", thinking="value present"),
        report.Event(report.EventKind.RESULT, "You opened wiki (browse result) · 1642"),
        report.Event(report.EventKind.REPLY, "Lake Baikal, 1,642 m.", thinking=""),
    ]
    checks = [
        report.CheckView("C1", "browsed", "spine", True, False, True, anchor_index=1),
        report.CheckView("C2", "reply names the fact", "reply", True, False, True, anchor_index=5),
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
        "| 💭 | <details><summary>thinking</summary>verify with source</details> |  |\n"
        '| actual | 🔧 browse({"queries":["x"],"extract":"depth"}) | ✅ C1 |\n'
        "| actual | 🧩 micro-context ← user turn: Instruction: depth · Content: 1,642 m |  |\n"
        "| 💭 | <details><summary>thinking (micro-context)</summary>value present</details> |  |\n"
        "| actual | 🧩 micro-context → EXTRACTED: 1642 |  |\n"
        "| actual | 📥 You opened wiki (browse result) · 1642 |  |\n"
        "| 💭 | 💭 (empty) |  |\n"
        "| actual | 🤖 Lake Baikal, 1,642 m. | ✅ C2 |\n"
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
