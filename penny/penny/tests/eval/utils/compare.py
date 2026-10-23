"""The ``BASELINE=<run>`` comparison table (#2734): one branch run's deterministic checks
against a main run's, case by case, with totals and a highlight on any case whose passing
rate moved :data:`HIGHLIGHT_POINTS` or more.

Pure artifact consumption over two already-completed runs' ``manifest.json`` +
``results.jsonl`` (``artifacts.py`` via ``assemble.py``'s loaders) — no model, no git, no
network, so the comparison builder is exercised by plain (non-eval) whole-render tests.

The one check this module cannot make itself is whether the branch's merge base is the
baseline's own commit — that needs ``git``, which the eval container never runs against a
worktree's ``.git`` (see the Makefile's ``HEAD_COMMIT_CMD`` note), so the recipe computes it
HOST-side and passes the one fact this module needs in: ``merge_base_ok`` — ``True`` when the
baseline commit is an ancestor of the branch's, ``False`` when it is not, ``None`` when it
could not be checked at all (the commit isn't fetched locally).

CLI (invoked in-container by the ``eval-report`` recipe): ``python -m
penny.tests.eval.utils.compare <out_dir> <branch_report_dir> <baseline_report_dir>
[merge_base_ok]`` — writes the comparison's postable parts into ``out_dir`` (split + marked
exactly like a run-comment part) and prints their file names in post order.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from penny.tests.eval.utils import comment_split
from penny.tests.eval.utils.artifacts import CaseArtifact, RunManifest
from penny.tests.eval.utils.assemble import load_case_artifacts, load_manifest
from penny.tests.eval.utils.pr_comments import COMPARE_MARKER

# The move (in percentage points of the passing rate) that gets flagged — a case moving less
# than this is read as noise, not a finding.
HIGHLIGHT_POINTS = 5.0
HIGHLIGHT_MARK = " ⚠"
MISSING_CELL = "—"
_TOTALS_CASE = "**total**"

COMPARE_HEADING = "## Branch vs main"
_IDENTITY_HEAD = "|  | run | commit | provider |\n|---|---|---|---|"
_IDENTITY_ROW = "| {label} | `{run_id}` | `{commit}` | {provider} |"
_BRANCH_LABEL = "branch"
_MAIN_LABEL = "main"

_TABLE_HEAD = "| case | branch | main | moved |\n|---|---|---|---|"
_ROW = "| {case} | {branch} | {main} | {moved} |"
_CELL = "{passed}/{total} · {rate:.0%}"
_MOVED_CELL = "{points:+.0f} pts{flag}"

_MERGE_BASE_WARNING = (
    "_warning: the branch's merge base with the baseline run's commit is not that commit "
    "(`{commit}`) — this comparison may not be against the branch's own ancestor._"
)

USAGE = (
    "usage: python -m penny.tests.eval.utils.compare <out_dir> <branch_report_dir> "
    "<baseline_report_dir> [merge_base_ok]"
)


class ProviderMismatchError(ValueError):
    """Raised when the branch and baseline runs were pinned to different upstream
    providers — a comparison across them would measure neither one's behaviour."""


# ── The per-case comparison (the pure decision this module is tested on) ─────
@dataclass(frozen=True)
class CaseComparison:
    """One case's reading on each side — ``None`` when the case is absent from that run."""

    case_id: str
    branch: tuple[int, int] | None
    main: tuple[int, int] | None

    @property
    def branch_rate(self) -> float | None:
        return _rate(self.branch)

    @property
    def main_rate(self) -> float | None:
        return _rate(self.main)

    @property
    def moved_points(self) -> float | None:
        """The branch's rate minus main's, in percentage points — ``None`` when either side
        has no reading (absent from that run, or a case with no scored checks)."""
        if self.branch_rate is None or self.main_rate is None:
            return None
        return (self.branch_rate - self.main_rate) * 100

    @property
    def highlighted(self) -> bool:
        moved = self.moved_points
        return moved is not None and abs(moved) >= HIGHLIGHT_POINTS


def _rate(cell: tuple[int, int] | None) -> float | None:
    if cell is None or not cell[1]:
        return None
    return cell[0] / cell[1]


def _scored_totals(artifact: CaseArtifact) -> tuple[int, int]:
    """One case's deterministic total — its scored checks, passed of present (advisory
    checks excluded, the same reading every other case-level total in the harness uses)."""
    scored = [check for check in artifact.checks if check.scored]
    return sum(check.passed for check in scored), sum(check.total for check in scored)


def build_case_comparisons(
    branch: Sequence[CaseArtifact], main: Sequence[CaseArtifact]
) -> list[CaseComparison]:
    """One row per case id appearing on EITHER side — branch order first, then any case main
    drove that the branch did not — so a case present on only one run still gets a row, with
    the other side's cell left absent rather than silently dropped."""
    by_branch = {artifact.case_id: _scored_totals(artifact) for artifact in branch}
    by_main = {artifact.case_id: _scored_totals(artifact) for artifact in main}
    ids = [*by_branch, *(case_id for case_id in by_main if case_id not in by_branch)]
    return [
        CaseComparison(case_id=case_id, branch=by_branch.get(case_id), main=by_main.get(case_id))
        for case_id in ids
    ]


def totals_row(comparisons: Sequence[CaseComparison]) -> CaseComparison:
    """The run-wide totals — each side summed over the cases THAT side actually reported, so
    a case absent from one run does not count as a zero against it."""
    return CaseComparison(
        case_id=_TOTALS_CASE,
        branch=_sum_cells(c.branch for c in comparisons),
        main=_sum_cells(c.main for c in comparisons),
    )


def _sum_cells(cells: Iterable[tuple[int, int] | None]) -> tuple[int, int] | None:
    present = [cell for cell in cells if cell is not None]
    if not present:
        return None
    return sum(passed for passed, _ in present), sum(total for _, total in present)


# ── Rendering ──────────────────────────────────────────────────────────────
def _cell(cell: tuple[int, int] | None) -> str:
    if cell is None:
        return MISSING_CELL
    passed, total = cell
    return _CELL.format(passed=passed, total=total, rate=passed / total if total else 0.0)


def _moved_cell(comparison: CaseComparison) -> str:
    moved = comparison.moved_points
    if moved is None:
        return MISSING_CELL
    flag = HIGHLIGHT_MARK if comparison.highlighted else ""
    return _MOVED_CELL.format(points=moved, flag=flag)


def render_comparison_table(comparisons: Sequence[CaseComparison]) -> str:
    """The table — every case's row, then the totals row last."""
    rows = [*comparisons, totals_row(comparisons)]
    body = "\n".join(
        _ROW.format(
            case=row.case_id if row.case_id == _TOTALS_CASE else f"`{row.case_id}`",
            branch=_cell(row.branch),
            main=_cell(row.main),
            moved=_moved_cell(row),
        )
        for row in rows
    )
    return f"{_TABLE_HEAD}\n{body}"


def _identity_row(label: str, manifest: RunManifest) -> str:
    return _IDENTITY_ROW.format(
        label=label,
        run_id=manifest.run_id,
        commit=manifest.commit[:8],
        provider=manifest.provider or MISSING_CELL,
    )


def render_comparison(
    branch_manifest: RunManifest,
    branch_artifacts: Sequence[CaseArtifact],
    main_manifest: RunManifest,
    main_artifacts: Sequence[CaseArtifact],
    merge_base_ok: bool | None = None,
) -> str:
    """The whole comparison comment (the summary method): refuses across providers, then the
    two runs' identity, the per-case table, and — only when the check ran and failed — the
    merge-base warning."""
    if branch_manifest.provider != main_manifest.provider:
        raise ProviderMismatchError(
            f"refusing to compare — branch ran on {branch_manifest.provider!r}, baseline ran "
            f"on {main_manifest.provider!r}; a comparison across providers measures neither"
        )
    comparisons = build_case_comparisons(branch_artifacts, main_artifacts)
    lines = [
        COMPARE_HEADING,
        "",
        _IDENTITY_HEAD,
        _identity_row(_BRANCH_LABEL, branch_manifest),
        _identity_row(_MAIN_LABEL, main_manifest),
        "",
        render_comparison_table(comparisons),
    ]
    if merge_base_ok is False:
        lines += ["", _MERGE_BASE_WARNING.format(commit=main_manifest.commit[:8])]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_merge_base_ok(argv: Sequence[str]) -> bool | None:
    if len(argv) < 4 or not argv[3]:
        return None
    return argv[3] == "1"


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(USAGE, file=sys.stderr)
        return 2
    out_dir, branch_dir, baseline_dir = Path(argv[0]), Path(argv[1]), Path(argv[2])
    try:
        branch_manifest = load_manifest(branch_dir)
        branch_artifacts = load_case_artifacts(branch_dir)
        baseline_manifest = load_manifest(baseline_dir)
        baseline_artifacts = load_case_artifacts(baseline_dir)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1
    try:
        document = render_comparison(
            branch_manifest,
            branch_artifacts,
            baseline_manifest,
            baseline_artifacts,
            merge_base_ok=_parse_merge_base_ok(argv),
        )
    except ProviderMismatchError as error:
        print(f"compare: {error}", file=sys.stderr)
        return 1
    document = f"{document}\n\n{COMPARE_MARKER}\n"
    parts = comment_split.split_run_summary(document)
    comment_split.enforce_part_cap(parts)
    for name in comment_split.write_parts(parts, out_dir):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
