"""Plain (non-eval) tests for the eval-run checkpoint markers (``checkpoint.py``, #1757).

They drive the pure filesystem helpers + the CLI over a SYNTHETIC artifact home (``tmp_path`` — a
throwaway fake home, never the real ``data/eval-artifacts/``), so they run inside ``make check``:
no model, no git, no container. The banner is asserted as a WHOLE-RENDER literal (pr-review-guide
§6); the marker semantics (a completed run = a ``manifest.json``; posted = a ``.posted`` marker)
are exercised directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from penny.tests.eval.checkpoint import (
    POSTED_MARKER,
    USAGE,
    is_posted,
    latest_run_dir,
    main,
    render_banner,
    run_dirs,
    unreviewed_runs,
)


def _make_run(home: Path, name: str, *, manifest: bool = True, posted: bool = False) -> Path:
    """Materialise a run dir under ``home``: a completed run carries ``manifest.json``; a posted
    run additionally carries a ``.posted`` marker. Returns the dir."""
    run = home / name
    run.mkdir(parents=True)
    if manifest:
        (run / "manifest.json").write_text("{}\n")
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


def test_latest_run_dir_picks_newest_by_mtime(tmp_path: Path) -> None:
    """``latest_run_dir`` returns the most-recently-modified completed run — not the lexically-last
    (the name order and mtime order are deliberately opposed here)."""
    old = _make_run(tmp_path, "run-z")
    new = _make_run(tmp_path, "run-a")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert latest_run_dir(tmp_path) == new
    assert latest_run_dir(tmp_path / "empty") is None


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


def test_cli_latest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``latest`` prints the newest run's NAME (exit 0); an empty home is an actionable stderr
    message (exit 1); a bad arg count is usage (exit 2)."""
    old = _make_run(tmp_path, "run-old")
    new = _make_run(tmp_path, "run-new")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert main(["latest", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "run-new"
    assert main(["latest", str(tmp_path / "empty")]) == 1
    assert "no completed run dirs" in capsys.readouterr().err
    assert main(["latest"]) == 2
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
