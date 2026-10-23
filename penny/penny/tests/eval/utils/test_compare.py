"""Plain (non-eval) tests for the ``BASELINE=<run>`` comparison builder (``compare.py``,
#2734): a branch run's deterministic checks against a main run's, case by case.

Drives the pure comparison + render helpers over hand-built ``CaseArtifact``/``RunManifest``
records (no model, no git, no container), so they run inside ``make check``. The whole
rendered comment is pinned as a whole-render literal (pr-review-guide §6).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from penny.tests.eval.utils.artifacts import (
    CaseArtifact,
    CaseTimings,
    CauseCounts,
    CheckOutcome,
    RunManifest,
    build_manifest,
)
from penny.tests.eval.utils.compare import (
    HIGHLIGHT_POINTS,
    ProviderMismatchError,
    build_case_comparisons,
    render_comparison,
    render_comparison_table,
    totals_row,
)
from penny.tests.eval.utils.pr_comments import COMPARE_MARKER

_TIMINGS = CaseTimings(calls=1, duration_ms=1000, input_tokens=100, output_tokens=10)


def _artifact(case_id: str, passed: int, total: int) -> CaseArtifact:
    """One case's artifact with a single scored check — the only field ``compare.py`` reads."""
    return CaseArtifact(
        run_id="run-x",
        case_id=case_id,
        family="family",
        mean=0.0,
        all_pass_rate=0.0,
        pathology_excluded_mean=0.0,
        samples=1,
        sample_scores=[0.0],
        sample_causes=[None],
        cause_counts=CauseCounts(),
        checks=[CheckOutcome(label="check", passed=passed, total=total, scored=True)],
        timings=_TIMINGS,
    )


def _manifest(*, commit: str, provider: str | None, run_id: str | None = None) -> RunManifest:
    return build_manifest(
        commit=commit,
        dirty_diff="",
        model="google/gemma-4-26b-a4b-it",
        embedding_model="embeddinggemma",
        samples=5,
        lever="test",
        now=datetime(2026, 10, 23, 23, 0, 0, tzinfo=UTC),
        provider=provider,
        run_id=run_id,
    )


# ── build_case_comparisons: matching cases, a case on one side only ──────────
def test_matching_cases_compute_both_sides_rates() -> None:
    comparisons = build_case_comparisons(
        branch=[_artifact("a", 9, 10)], main=[_artifact("a", 10, 10)]
    )
    assert len(comparisons) == 1
    row = comparisons[0]
    assert row.case_id == "a"
    assert row.branch_rate == pytest.approx(0.9)
    assert row.main_rate == pytest.approx(1.0)
    assert row.moved_points == pytest.approx(-10.0)


def test_a_case_present_on_only_one_side_still_gets_a_row() -> None:
    """A case the branch drove that main never ran (or vice versa) still gets a row, with the
    other side's cell left absent — never silently dropped."""
    branch_only = build_case_comparisons(branch=[_artifact("only-branch", 3, 5)], main=[])
    assert len(branch_only) == 1
    assert branch_only[0].main is None
    assert branch_only[0].main_rate is None
    assert branch_only[0].moved_points is None
    assert branch_only[0].highlighted is False

    main_only = build_case_comparisons(branch=[], main=[_artifact("only-main", 4, 5)])
    assert len(main_only) == 1
    assert main_only[0].branch is None
    assert main_only[0].case_id == "only-main"


def test_row_order_is_branch_first_then_main_only_cases() -> None:
    comparisons = build_case_comparisons(
        branch=[_artifact("a", 1, 1), _artifact("b", 1, 1)],
        main=[_artifact("b", 1, 1), _artifact("c", 1, 1)],
    )
    assert [row.case_id for row in comparisons] == ["a", "b", "c"]


# ── the highlight threshold ───────────────────────────────────────────────────
def test_highlight_threshold_is_inclusive_at_five_points() -> None:
    just_under = build_case_comparisons(
        branch=[_artifact("a", 54, 100)], main=[_artifact("a", 50, 100)]
    )[0]
    assert just_under.moved_points == pytest.approx(4.0)
    assert just_under.highlighted is False

    at_threshold = build_case_comparisons(
        branch=[_artifact("a", 55, 100)], main=[_artifact("a", 50, 100)]
    )[0]
    assert at_threshold.moved_points == pytest.approx(5.0)
    assert at_threshold.highlighted is True


def test_highlight_fires_on_a_drop_as_well_as_a_rise() -> None:
    dropped = build_case_comparisons(
        branch=[_artifact("a", 40, 100)], main=[_artifact("a", 50, 100)]
    )[0]
    moved = dropped.moved_points
    assert moved is not None
    assert moved == pytest.approx(-10.0)
    assert dropped.highlighted is True
    assert dropped.highlighted == (abs(moved) >= HIGHLIGHT_POINTS)


# ── totals ────────────────────────────────────────────────────────────────────
def test_totals_sum_only_the_cases_each_side_actually_reported() -> None:
    """A case absent from one side must not count as a zero against it — the totals sum each
    side over the cases THAT side drove, exactly like the per-case rows above them."""
    comparisons = build_case_comparisons(
        branch=[_artifact("a", 9, 10), _artifact("b", 5, 10), _artifact("c", 3, 5)],
        main=[_artifact("a", 10, 10), _artifact("b", 5, 10), _artifact("d", 4, 5)],
    )
    totals = totals_row(comparisons)
    assert totals.branch == (9 + 5 + 3, 10 + 10 + 5)
    assert totals.main == (10 + 5 + 4, 10 + 10 + 5)


def test_totals_is_absent_when_a_side_reported_nothing() -> None:
    totals = totals_row(build_case_comparisons(branch=[_artifact("a", 1, 1)], main=[]))
    assert totals.branch == (1, 1)
    assert totals.main is None


# ── render_comparison_table (whole-render) ────────────────────────────────────
def test_render_comparison_table_whole_render() -> None:
    comparisons = build_case_comparisons(
        branch=[_artifact("a", 9, 10), _artifact("c", 3, 5)],
        main=[_artifact("a", 10, 10), _artifact("d", 4, 5)],
    )
    assert render_comparison_table(comparisons) == (
        "| case | branch | main | moved |\n"
        "|---|---|---|---|\n"
        "| `a` | 9/10 · 90% | 10/10 · 100% | -10 pts ⚠ |\n"
        "| `c` | 3/5 · 60% | — | — |\n"
        "| `d` | — | 4/5 · 80% | — |\n"
        "| **total** | 12/15 · 80% | 14/15 · 93% | -13 pts ⚠ |"
    )


# ── render_comparison: providers, identity, the merge-base warning ──────────
def test_render_comparison_refuses_across_providers() -> None:
    branch_manifest = _manifest(commit="a" * 40, provider="Groq")
    main_manifest = _manifest(commit="b" * 40, provider="Cloudflare")
    with pytest.raises(ProviderMismatchError, match="Groq.*Cloudflare"):
        render_comparison(branch_manifest, [], main_manifest, [])


def test_render_comparison_whole_render_with_merge_base_ok() -> None:
    branch_manifest = _manifest(commit="a" * 40, provider="Makora", run_id="run-branch-abc")
    main_manifest = _manifest(commit="b" * 40, provider="Makora", run_id="run-main-def")
    document = render_comparison(
        branch_manifest,
        [_artifact("a", 9, 10)],
        main_manifest,
        [_artifact("a", 10, 10)],
        merge_base_ok=True,
    )
    assert document == (
        "## Branch vs main\n"
        "\n"
        "|  | run | commit | provider |\n"
        "|---|---|---|---|\n"
        "| branch | `run-branch-abc` | `aaaaaaaa` | Makora |\n"
        "| main | `run-main-def` | `bbbbbbbb` | Makora |\n"
        "\n"
        "| case | branch | main | moved |\n"
        "|---|---|---|---|\n"
        "| `a` | 9/10 · 90% | 10/10 · 100% | -10 pts ⚠ |\n"
        "| **total** | 9/10 · 90% | 10/10 · 100% | -10 pts ⚠ |"
    )
    assert "merge base" not in document


def test_render_comparison_warns_when_merge_base_is_not_the_baseline_commit() -> None:
    branch_manifest = _manifest(commit="a" * 40, provider="Makora", run_id="run-branch")
    main_manifest = _manifest(commit="b" * 40, provider="Makora", run_id="run-main")
    document = render_comparison(branch_manifest, [], main_manifest, [], merge_base_ok=False)
    assert "warning: the branch's merge base" in document
    assert "`bbbbbbbb`" in document.splitlines()[-1]


def test_render_comparison_is_silent_when_the_merge_base_could_not_be_checked() -> None:
    """``merge_base_ok=None`` — the commit wasn't fetched locally — renders no warning at
    all, the same as a confirmed-ancestor run: an unverifiable check is not a finding."""
    branch_manifest = _manifest(commit="a" * 40, provider="Makora")
    main_manifest = _manifest(commit="b" * 40, provider="Makora")
    document = render_comparison(branch_manifest, [], main_manifest, [], merge_base_ok=None)
    assert "warning" not in document


def test_the_cli_marker_rides_on_the_posted_document() -> None:
    """Confirms the two modules' seam: ``COMPARE_MARKER`` is what the CLI appends after
    rendering, never something ``render_comparison`` itself has to know about."""
    assert COMPARE_MARKER not in render_comparison(
        _manifest(commit="a" * 40, provider="Makora"),
        [],
        _manifest(commit="b" * 40, provider="Makora"),
        [],
    )
