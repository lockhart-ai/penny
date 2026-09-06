"""Plain (non-eval) tests for the eval-run checkpoint markers (``checkpoint.py``, #1757).

They drive the pure filesystem helpers + the CLI over a SYNTHETIC artifact home (``tmp_path`` — a
throwaway fake home, never the real ``data/eval-artifacts/``), so they run inside ``make check``:
no model, no git, no container. The banner and both resolution refusals are asserted as WHOLE-RENDER
literals (pr-review-guide §6); the marker semantics (a completed run = a ``manifest.json``; posted =
a ``.posted`` marker) are exercised directly. The bare-``RUN`` resolution (#2098) is driven over a
home holding runs from SEVERAL trees — the shape the shared artifact home actually has under a
fleet — so "returns a sibling's run" is a failure the tests can see.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from penny.tests.eval.utils.checkpoint import (
    POSTED_MARKER,
    USAGE,
    is_posted,
    main,
    measured_commit,
    posted_runs_at_commit,
    render_ambiguous,
    render_banner,
    render_no_match,
    run_dirs,
    runs_at_commit,
    unreviewed_runs,
)

# Two 40-char SHAs standing in for two worktrees' HEADs, plus a third nothing was measured at.
COMMIT_HERE = "a" * 40
COMMIT_SIBLING = "b" * 40
COMMIT_UNMEASURED = "c" * 40


def _make_run(
    home: Path,
    name: str,
    *,
    manifest: bool = True,
    posted: bool = False,
    commit: str = COMMIT_HERE,
) -> Path:
    """Materialise a run dir under ``home``: a completed run carries a ``manifest.json`` recording
    the commit it measured; a posted run additionally carries a ``.posted`` marker. Returns the
    dir."""
    run = home / name
    run.mkdir(parents=True)
    if manifest:
        (run / "manifest.json").write_text(f"{json.dumps({'commit': commit})}\n")
    if posted:
        (run / POSTED_MARKER).write_text("https://github.com/o/r/pull/1#issuecomment-1\n")
    return run


def test_run_dirs_requires_a_manifest(tmp_path: Path) -> None:
    """A run dir is a subdir holding ``manifest.json`` — a lever-less dir (no manifest) and a stray
    file are not run dirs; results are name-sorted."""
    _make_run(tmp_path, "run-b")
    _make_run(tmp_path, "run-a")
    _make_run(tmp_path, "run-inflight", manifest=False)  # mid-eval, no manifest yet
    (tmp_path / "loose.txt").write_text("x")
    assert [run.name for run in run_dirs(tmp_path)] == ["run-a", "run-b"]


def test_run_dirs_empty_when_home_absent(tmp_path: Path) -> None:
    """A missing home is not an error — no run dirs."""
    assert run_dirs(tmp_path / "nope") == []


def test_runs_at_commit_is_this_trees_unposted_runs_only(tmp_path: Path) -> None:
    """The candidate set is keyed to the COMMIT, not to recency: a sibling tree's run, this tree's
    already-posted run, and a run whose manifest is unreadable all drop out. The sibling's run is
    deliberately the newest and lexically last, so a recency fall back would surface it."""
    _make_run(tmp_path, "run-mine", commit=COMMIT_HERE)
    _make_run(tmp_path, "run-mine-posted", commit=COMMIT_HERE, posted=True)
    half_written = _make_run(tmp_path, "run-half", commit=COMMIT_HERE)
    (half_written / "manifest.json").write_text('{"commit": "aaaa')  # a concurrent run mid-write
    _make_run(tmp_path, "run-sibling", commit=COMMIT_SIBLING)  # created LAST, sorts LAST
    assert measured_commit(half_written) is None
    assert [run.name for run in runs_at_commit(tmp_path, COMMIT_HERE)] == ["run-mine"]
    assert [run.name for run in posted_runs_at_commit(tmp_path, COMMIT_HERE)] == ["run-mine-posted"]
    assert runs_at_commit(tmp_path, COMMIT_UNMEASURED) == []


def test_unreviewed_runs_excludes_posted(tmp_path: Path) -> None:
    """Only completed runs without a ``.posted`` marker are unreviewed; a posted run drops out."""
    _make_run(tmp_path, "run-1")
    posted = _make_run(tmp_path, "run-2", posted=True)
    _make_run(tmp_path, "run-3")
    assert is_posted(posted)
    assert [run.name for run in unreviewed_runs(tmp_path)] == ["run-1", "run-3"]


def test_render_banner_whole_render(tmp_path: Path) -> None:
    """The loud multi-line banner names every unreviewed run + the post command (exact literal)."""
    _make_run(tmp_path, "run-20990101T000001Z")
    _make_run(tmp_path, "run-20990101T000002Z")
    rule = "=" * 72
    assert render_banner(unreviewed_runs(tmp_path)) == (
        f"{rule}\n"
        "⚠  2 unreviewed eval run(s) — post and review before running again:\n"
        "     run-20990101T000001Z\n"
        "     run-20990101T000002Z\n"
        "→ post each: make eval-report PR=<n> [RUN=<run-dir-name>]\n"
        f"{rule}"
    )


def test_render_banner_empty_when_nothing_unreviewed(tmp_path: Path) -> None:
    """No unreviewed runs → empty string (the recipe prints nothing)."""
    _make_run(tmp_path, "run-1", posted=True)
    assert render_banner(unreviewed_runs(tmp_path)) == ""


def test_render_no_match_whole_render(tmp_path: Path) -> None:
    """The zero-candidate refusal names the home and the commit, says the home is shared, and names
    the runs already posted at that commit when there are any (exact literal, both ways)."""
    posted = _make_run(tmp_path, "run-mine-posted", commit=COMMIT_HERE, posted=True)
    shared = (
        "     the artifact home is SHARED by every worktree (#1734) — a run in it belongs to "
        "whichever tree measured it"
    )
    action = "     run `make eval` on this tree, or post a named run with RUN=<run-dir>"
    assert render_no_match(tmp_path, COMMIT_HERE, [posted]) == (
        f"eval-report: no unposted completed run under {tmp_path} was measured at {COMMIT_HERE}\n"
        f"{shared}\n"
        "     already posted at this commit: run-mine-posted — re-post one with "
        "RUN=<run-dir> FORCE=1\n"
        f"{action}"
    )
    assert render_no_match(tmp_path, COMMIT_HERE, []) == (
        f"eval-report: no unposted completed run under {tmp_path} was measured at {COMMIT_HERE}\n"
        f"{shared}\n"
        f"{action}"
    )


def test_render_ambiguous_whole_render(tmp_path: Path) -> None:
    """The several-candidate refusal LISTS them and asks for RUN= — it never picks one."""
    runs = [_make_run(tmp_path, "run-a"), _make_run(tmp_path, "run-b")]
    assert render_ambiguous(tmp_path, COMMIT_HERE, runs) == (
        f"eval-report: 2 unposted runs under {tmp_path} were measured at {COMMIT_HERE}:\n"
        "     run-a\n"
        "     run-b\n"
        "     the artifact home is SHARED by every worktree (#1734) — a run in it belongs to "
        "whichever tree measured it\n"
        "     name the one to post: make eval-report PR=<n> RUN=<run-dir>"
    )


def test_cli_resolve_three_outcomes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``resolve`` has exactly three outcomes: ONE candidate prints its name (exit 0); NONE and
    SEVERAL each print their refusal on stderr (exit 1) and print NOTHING on stdout — a caller
    capturing stdout gets an empty string, never a sibling's run. A bad arg count is usage (2)."""
    _make_run(tmp_path, "run-mine", commit=COMMIT_HERE)
    _make_run(tmp_path, "run-sibling", commit=COMMIT_SIBLING)

    assert main(["resolve", str(tmp_path), COMMIT_HERE]) == 0
    assert capsys.readouterr().out.strip() == "run-mine"

    assert main(["resolve", str(tmp_path), COMMIT_UNMEASURED]) == 1
    none_out, none_err = capsys.readouterr()
    assert none_out == ""
    assert none_err == f"{render_no_match(tmp_path, COMMIT_UNMEASURED, [])}\n"

    second = _make_run(tmp_path, "run-mine-2", commit=COMMIT_HERE)
    assert main(["resolve", str(tmp_path), COMMIT_HERE]) == 1
    many_out, many_err = capsys.readouterr()
    assert many_out == ""
    assert many_err == (
        f"{render_ambiguous(tmp_path, COMMIT_HERE, [tmp_path / 'run-mine', second])}\n"
    )

    assert main(["resolve", str(tmp_path)]) == 2
    assert capsys.readouterr().err.strip() == USAGE


def test_cli_banner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``banner`` prints the banner when runs are unreviewed (exit 0) and NOTHING when none —
    warn, never block; an unknown verb is usage (exit 2)."""
    _make_run(tmp_path, "run-1")
    expected = render_banner(unreviewed_runs(tmp_path))
    assert main(["banner", str(tmp_path)]) == 0
    assert capsys.readouterr().out == f"{expected}\n"
    (tmp_path / "run-1" / POSTED_MARKER).write_text("url\n")
    assert main(["banner", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""
    assert main(["bogus", str(tmp_path)]) == 2
    assert capsys.readouterr().err.strip() == USAGE
