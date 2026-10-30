"""Pruning old eval run directories (#2851): the shell helper `make eval-remote` sources
before it does anything else — including before the provider lock — and `make eval-prune`
runs on demand.

The prune logic is a shell file the Makefile recipes source (`scripts/eval-prune.sh`), so
what is tested here is that file, run the same way the recipes run it — not a description
of it (the same approach `test_provider_lock.py` takes for its own shell helper). Every run
dir in these tests is a fixture built directly under `tmp_path`; nothing here ever reads or
writes the real `data/eval-artifacts`, which may hold runs genuinely in progress.
"""

from __future__ import annotations

import pathlib
import subprocess
from datetime import UTC, datetime, timedelta

_PRUNE_SCRIPT = pathlib.Path("scripts") / "eval-prune.sh"
_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_MANIFEST = b"{}"

# The recipe's own call shape: source the file, then call the one function it needs.
_CALL = '. "$1"\nprune_run "$2" "$3" "$4"\n'


def _prune_script() -> str:
    """The script file itself, found above this test the way the Makefile finds it."""
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / _PRUNE_SCRIPT).is_file():
            return str(parent / _PRUNE_SCRIPT)
    raise AssertionError(f"no {_PRUNE_SCRIPT} above this test")


def _stamp(days_ago: float) -> str:
    """A run dir's stamp for a moment `days_ago` days before now, in the exact shape the
    `eval-remote` recipe stamps a run dir with (`date -u +%Y%m%dT%H%M%SZ`)."""
    when = datetime.now(UTC) - timedelta(days=days_ago)
    return when.strftime(_STAMP_FORMAT)


def _run_dir(
    home: pathlib.Path, *, days_ago: float, pid: int, manifest: bool = True, extra_bytes: int = 0
) -> pathlib.Path:
    """One fake run dir under `home`, named exactly as the recipe names one
    (`run-<stamp>-<pid>-<commit8>`, pid zero-padded to 5 digits), optionally carrying a
    `manifest.json` — the one thing that marks a run COMPLETED — and `extra_bytes` of filler
    content, for tests that check the reported freed-space total."""
    name = f"run-{_stamp(days_ago)}-{pid:05d}-deadbeef"
    run_dir = home / name
    run_dir.mkdir(parents=True)
    if manifest:
        (run_dir / "manifest.json").write_bytes(_MANIFEST)
    if extra_bytes:
        (run_dir / "data.bin").write_bytes(b"x" * extra_bytes)
    return run_dir


def _prune(home: pathlib.Path, keep_days: int, lock_dir: pathlib.Path) -> str:
    """Run `prune_run <home> <keep_days> <lock_dir>` and return everything it printed."""
    result = subprocess.run(
        ["sh", "-c", _CALL, "sh", _prune_script(), str(home), str(keep_days), str(lock_dir)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_an_old_completed_run_is_removed(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    old = _run_dir(home, days_ago=10, pid=11111, extra_bytes=50)
    expected_bytes = len(_MANIFEST) + 50

    output = _prune(home, 7, tmp_path / "locks")

    assert not old.exists(), "a completed run past the window is removed"
    assert f"removed {old.name} ({expected_bytes} bytes)" in output
    assert f"removed 1 run(s), freed {expected_bytes} bytes" in output


def test_a_recent_completed_run_is_kept(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    recent = _run_dir(home, days_ago=1, pid=22222)

    output = _prune(home, 7, tmp_path / "locks")

    assert recent.exists(), "a run inside the window stays, however complete"
    assert "nothing older than 7 day(s)" in output


def test_an_old_run_without_a_manifest_is_kept(tmp_path: pathlib.Path) -> None:
    """A run with no `manifest.json` is still writing, or died before finishing — never
    pruned on age alone, whatever its stamp says."""
    home = tmp_path / "home"
    home.mkdir()
    unfinished = _run_dir(home, days_ago=30, pid=33333, manifest=False)

    output = _prune(home, 7, tmp_path / "locks")

    assert unfinished.exists()
    assert "nothing older than 7 day(s)" in output


def test_non_run_entries_are_left_alone(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "not-a-run-dir").mkdir()
    (home / "README.md").write_text("notes")

    output = _prune(home, 7, tmp_path / "locks")

    assert (home / "not-a-run-dir").is_dir()
    assert (home / "README.md").is_file()
    assert "nothing older than 7 day(s)" in output


def test_keep_days_zero_disables_pruning(tmp_path: pathlib.Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    old = _run_dir(home, days_ago=365, pid=44444, extra_bytes=10)

    output = _prune(home, 0, tmp_path / "locks")

    assert old.exists(), "EVAL_ARTIFACT_KEEP_DAYS=0 removes nothing, however old"
    assert "KEEP_DAYS=0 — pruning disabled" in output


def test_a_run_whose_pid_holds_a_provider_lock_is_kept(tmp_path: pathlib.Path) -> None:
    """A lock directory holds one file per holding pid, named for the pid (unpadded) —
    scripts/eval-provider-lock.sh's own layout. A run whose embedded pid still names one is
    left alone, whether that is a run genuinely still in flight or a reused pid colliding by
    coincidence — pruning never guesses which."""
    home = tmp_path / "home"
    home.mkdir()
    locked = _run_dir(home, days_ago=10, pid=55555)
    lock_dir = tmp_path / "locks"
    (lock_dir / "SomeProvider").mkdir(parents=True)
    (lock_dir / "SomeProvider" / "55555").touch()

    output = _prune(home, 7, lock_dir)

    assert locked.exists(), "the locked run's pid is still on a lock, so it is kept"
    assert "nothing older than 7 day(s)" in output


def test_an_old_run_is_removed_once_its_lock_clears(tmp_path: pathlib.Path) -> None:
    """The same run as above, but the lock is for a DIFFERENT pid — pruning is keyed to the
    exact pid the run dir's name carries, never to "a lock exists somewhere"."""
    home = tmp_path / "home"
    home.mkdir()
    old = _run_dir(home, days_ago=10, pid=66666)
    lock_dir = tmp_path / "locks"
    (lock_dir / "SomeProvider").mkdir(parents=True)
    (lock_dir / "SomeProvider" / "99999").touch()

    _prune(home, 7, lock_dir)

    assert not old.exists()
