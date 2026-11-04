"""Tests for `scripts/store_git_info.sh` (#2891, reworked by #2973): the
host-side script `up`/`prod`/`prod-ios` call before building, which snapshots
Penny's changelog commit window into the build-context file. Run the same way
the Makefile recipes run it — against a real, throwaway git repo — not a
description of it (the same approach `test_prune.py`/`test_provider_lock.py`
take for their own shell helpers).

There is no cursor any more (#2973): the script always writes the newest N raw
commits, oldest first, and WHICH of those are hers to record is decided by
Penny herself at startup (`select_new_entries` — the log is the cursor).

Needs `git` on PATH to build that temp repo and exercise the real `git log`
call the script makes — the Dockerfile installs it for exactly this. A missing
`git` is a hard, reported FAILURE here, never a silent skip: a skipped test
never verified the script that ran."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

_SCRIPT = pathlib.Path("scripts") / "store_git_info.sh"

# Exercised at a smaller override here so a test repo needs only a handful of
# commits, never restated as a hidden assumption the script could drift away
# from unnoticed.
_DEFAULT_SNAPSHOT = 200


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


def _commit(repo: pathlib.Path, subject: str, body: str = "") -> str:
    """One commit carrying ``subject`` as its title and, if given, ``body``
    (which may itself be multi-line) as its message body; returns its hash."""
    (repo / "file.txt").write_text(subject + body)
    _git(repo, "add", "-A")
    args = ["commit", "-q", "-m", subject]
    if body:
        args += ["-m", body]
    _git(repo, *args)
    return _git(repo, "rev-parse", "HEAD")


def _records(commits_file: pathlib.Path) -> list[tuple[str, str]]:
    """Each record's ``(subject, body)`` pair, read the same way
    ``parse_commit_log`` splits them (``\x1e`` between records, ``\x1f``
    between fields) — the hash itself isn't this test's concern
    (``parse_commit_log`` reads it for real, with its own unit tests)."""
    content = commits_file.read_text()
    if not content:
        return []
    pairs: list[tuple[str, str]] = []
    for record in content.split("\x1e"):
        record = record.lstrip("\n")  # git's own between-commit separator only
        if not record:
            continue
        fields = record.split("\x1f", 2)
        subject = fields[1] if len(fields) > 1 else ""
        body = fields[2] if len(fields) > 2 else ""
        pairs.append((subject, body))
    return pairs


def _titles(commits_file: pathlib.Path) -> list[str]:
    """Each record's subject half, in order."""
    return [subject for subject, _body in _records(commits_file)]


def _run_script(
    repo: pathlib.Path,
    *,
    commits_file: pathlib.Path,
    snapshot: int = _DEFAULT_SNAPSHOT,
) -> None:
    env = {
        **os.environ,
        "CHANGELOG_COMMITS_FILE": str(commits_file),
        "CHANGELOG_SNAPSHOT_COMMITS": str(snapshot),
    }
    result = subprocess.run(
        ["bash", _script_path()],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_snapshot_is_the_newest_n_commits_oldest_first(tmp_path: pathlib.Path) -> None:
    """The window is always the newest N raw commits, oldest first — never the
    whole repo history. There is no cursor and no mode: every build gets the
    same fixed window, and Penny's own startup selection cuts it against the
    log."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "feat(penny): first (#1)")
    _commit(repo, "docs: not penny-scope, still counted as one raw commit (#2)")
    _commit(repo, "fix(penny): second (#3)")
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, commits_file=commits_file, snapshot=2)

    assert _titles(commits_file) == [
        "docs: not penny-scope, still counted as one raw commit (#2)",
        "fix(penny): second (#3)",
    ]


def test_no_flag_bounds_the_window_when_the_repo_is_bigger(tmp_path: pathlib.Path) -> None:
    """A repo with more commits than the window keeps only the newest ones —
    the window is a BOUND, not a guarantee to include everything."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    for number in range(1, 6):
        _commit(repo, f"fix(penny): change {number} (#{number})")
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, commits_file=commits_file, snapshot=3)

    assert _titles(commits_file) == [
        "fix(penny): change 3 (#3)",
        "fix(penny): change 4 (#4)",
        "fix(penny): change 5 (#5)",
    ]


def test_multiline_bodies_are_captured_intact(tmp_path: pathlib.Path) -> None:
    """A commit's BODY is captured too, not just its title (#2891, round 4):
    the repo's squash setting makes a squashed commit's body the merged PR's
    own description, which is routinely several lines. The record separator
    is `\x1e`, not a newline, precisely so an embedded newline in a real
    body survives — git's own trailing newline on `%b` is left as is; the
    Python side (`first_body_sentence`) is what trims it."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    body = "First line of the real summary.\nSecond line continues the same paragraph."
    _commit(repo, "fix(penny): second (#2)", body)
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, commits_file=commits_file)

    # git appends its own single trailing newline to a commit's stored body.
    assert _records(commits_file) == [("fix(penny): second (#2)", body + "\n")]


def test_multiple_commits_each_keep_their_own_multiline_body(tmp_path: pathlib.Path) -> None:
    """Two commits, each with its own multi-line body, come back as two
    separate records — one body never bleeds into the next."""
    _require_git()
    repo = tmp_path / "repo"
    _init_repo(repo)
    first_body = "Paragraph one of the first PR.\nStill paragraph one."
    second_body = "Paragraph one of the second PR.\n\nA second paragraph, after a blank line."
    _commit(repo, "feat(penny): first (#1)", first_body)
    _commit(repo, "fix(penny): second (#2)", second_body)
    commits_file = tmp_path / "commits.txt"

    _run_script(repo, commits_file=commits_file, snapshot=5)

    assert _records(commits_file) == [
        ("feat(penny): first (#1)", first_body + "\n"),
        ("fix(penny): second (#2)", second_body + "\n"),
    ]
