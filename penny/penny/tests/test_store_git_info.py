"""Tests for `scripts/store_git_info.sh` (#2891): the host-side script
`up`/`prod`/`prod-ios` call before building, which writes Penny's changelog
commit range into the build-context file. Run the same way the Makefile
recipes run it — against a real, throwaway git repo — not a description of
it (the same approach `test_prune.py`/`test_provider_lock.py` take for
their own shell helpers).

Needs `git` on PATH to build that temp repo and exercise the real `git log`/
`git merge-base` calls the script makes — the Dockerfile installs it for
exactly this. A missing `git` is a hard, reported FAILURE here, never a
silent skip: a skipped test never verified the script that ran."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

_SCRIPT = pathlib.Path("scripts") / "store_git_info.sh"

# The script's own default (#2891, the code owner's ruling) — exercised at a
# smaller override here so a test repo needs only a handful of commits, never
# restated as a hidden assumption the script could drift away from unnoticed.
_DEFAULT_FALLBACK = 10


def _script_path() -> str:
    """The script file itself, found above this test the way the Makefile
    finds it — on the host, or at the mounted `/penny/scripts/...` path
    `docker-compose.override.yml` gives it inside the container."""
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / _SCRIPT
        if candidate.is_file():
            return str(candidate)
    raise AssertionError(f"no {_SCRIPT} above this test")


def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.fail(
            "git is not installed in this environment — scripts/store_git_info.sh "
            "needs it to run, and this test cannot verify the script without it "
            "(the Dockerfile installs it for exactly this; LOCAL=1 runs need it "
            "on the host too)"
        )


def _git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _commit(repo: pathlib.Path, message: str) -> str:
    """One commit carrying ``message`` as its title; returns its hash."""
    (repo / "file.txt").write_text(message)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _titles(commits_file: pathlib.Path) -> list[str]:
    """Each written line's subject half — the hash half is the script's own
    concern, not this test's (``parse_commit_log`` is what reads it for real,
    and has its own unit tests)."""
    if not commits_file.read_text():
        return []
    return [line.split("\x1f")[1] for line in commits_file.read_text().splitlines()]


def _run_script(
    repo: pathlib.Path,
    mode: str,
    *,
    cursor_file: pathlib.Path,
    commits_file: pathlib.Path,
    fallback: int = _DEFAULT_FALLBACK,
) -> None:
    env = {
        **os.environ,
        "CHANGELOG_CURSOR_FILE": str(cursor_file),
        "CHANGELOG_COMMITS_FILE": str(commits_file),
        "CHANGELOG_FALLBACK_COMMITS": str(fallback),
    }
    result = subprocess.run(
        ["bash", _script_path(), mode],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_no_cursor_falls_back_to_the_newest_n_commits(tmp_path: pathlib.Path) -> None:
    """An unset cursor (a fresh install, or dev with no prior deploy) falls
    back to the newest N raw commits, oldest first — never the whole repo
    history."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "feat(penny): first (#1)")
    _commit(repo, "docs: not penny-scope, still counted as one raw commit (#2)")
    _commit(repo, "fix(penny): second (#3)")
    cursor_file = tmp_path / "cursor"
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, "dev", cursor_file=cursor_file, commits_file=commits_file, fallback=2)

    assert not cursor_file.exists(), "dev mode never writes a cursor it didn't read"
    assert _titles(commits_file) == [
        "docs: not penny-scope, still counted as one raw commit (#2)",
        "fix(penny): second (#3)",
    ]


def test_a_valid_cursor_returns_only_the_commits_after_it(tmp_path: pathlib.Path) -> None:
    """A cursor that IS an ancestor of HEAD gets the range since it — not the
    fallback window — and `advance` mode still never writes the cursor
    itself (the Makefile does that, after a successful run)."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    first = _commit(repo, "feat(penny): first (#1)")
    _commit(repo, "fix(penny): second (#2)")
    _commit(repo, "fix(penny): third (#3)")
    cursor_file = tmp_path / "cursor"
    cursor_file.write_text(first)
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, "advance", cursor_file=cursor_file, commits_file=commits_file)

    assert _titles(commits_file) == [
        "fix(penny): second (#2)",
        "fix(penny): third (#3)",
    ]
    assert cursor_file.read_text() == first, "the script only READS the cursor"


def test_a_cursor_not_an_ancestor_falls_back_to_the_newest_n(tmp_path: pathlib.Path) -> None:
    """A cursor history has rewritten past (not an ancestor of HEAD) is
    treated exactly like an unset one: the newest N raw commits, never an
    error and never the whole history."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "feat(penny): first (#1)")
    _commit(repo, "fix(penny): second (#2)")
    cursor_file = tmp_path / "cursor"
    cursor_file.write_text("deadbeef" * 5)  # well-formed, but not an object in this repo
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, "dev", cursor_file=cursor_file, commits_file=commits_file, fallback=1)

    assert _titles(commits_file) == ["fix(penny): second (#2)"]
