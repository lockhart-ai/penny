"""Tests for `scripts/deploy-loop.sh` (#2955): the host-side loop that watches
`origin/main`, fast-forwards the deploy checkout, and redeploys through
`make prod-build`/`prod-run`. Run the same way `test_store_git_info.py` runs its
own shell script — against real, throwaway git repos, driving the real `bash`
process — not a description of it.

The loop is long-running, so tests drive it as a subprocess with STUB
`DEPLOY_BUILD_CMD`/`DEPLOY_RUN_CMD` (the loop's own commands are overridable for
exactly this) and force a fetch-check with `SIGUSR1` rather than waiting out
`DEPLOY_INTERVAL` — the loop treats that signal identically to the timer
elapsing, so a test never needs a real-timer sleep to see one fire. Every
assertion that depends on a SECOND tick's outcome (the "not retried" case) waits
for that tick's own distinguishing log line, never a fixed sleep guessed to be
"long enough" — `_wait_until` polls a condition with a generous timeout instead,
same shape as `conftest.py`'s own `wait_until`.

Needs `git` on PATH, same as `test_store_git_info.py` — a missing `git` is a
hard, reported FAILURE here, never a silent skip."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import signal
import subprocess
import time

import pytest

_SCRIPT = pathlib.Path("scripts") / "deploy-loop.sh"


def _script_path() -> str:
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / _SCRIPT
        if candidate.is_file():
            return str(candidate)
    raise AssertionError(f"no {_SCRIPT} above this test")


def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.fail(
            "git is not installed in this environment — scripts/deploy-loop.sh "
            "needs it to run, and this test cannot verify the script without it"
        )


def _git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _commit(repo: pathlib.Path, message: str, content: str) -> str:
    (repo / "file.txt").write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _clone(origin: pathlib.Path, clone_dir: pathlib.Path) -> None:
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone_dir)],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    _git(clone_dir, "config", "user.email", "test@example.com")
    _git(clone_dir, "config", "user.name", "Test")


def _wait_until(condition, timeout: float = 5.0, interval: float = 0.05) -> None:
    """Polls `condition` (a zero-arg callable) until it returns True, or raises
    `TimeoutError` — a bounded, condition-based wait, never a fixed sleep assumed
    to be long enough. `interval` is a short poll gap, not a wait of its own —
    the loop re-checks `condition` every tick and returns the moment it holds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError(f"condition not met within {timeout}s")


def _write_executable(path: pathlib.Path, body: str) -> None:
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(0o755)


def _child_is_stopped(pid: int) -> bool:
    """Reads the kernel's own record that a process is in the stopped ('T') state
    — never a fixed sleep guessing when a SIGSTOP landed. Linux (the gate's
    container, which has no `ps`) exposes it in /proc/<pid>/stat; macOS has no
    /proc and falls back to `ps`, which is present there."""
    proc_stat = pathlib.Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        # Field 3 is the state letter; comm (field 2) may itself contain spaces
        # and parens, so read past the final ')' to the state field.
        return proc_stat.read_text().rsplit(")", 1)[1].split()[0] == "T"
    if shutil.which("ps") is None:
        return False
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and result.stdout.strip().startswith("T")


def _read_or_empty(path: pathlib.Path) -> str:
    """`path`'s text, or "" before it exists — a stub's log file is created by
    its own first write, so a `_wait_until` condition polling it must not treat
    "not written yet" as an error."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _run_stub(
    tmp_path: pathlib.Path, run_log: pathlib.Path, pid_file: pathlib.Path
) -> pathlib.Path:
    """A long-lived "server": records one line per start, writes its own pid
    (stable across the `exec` below), then runs forever — like `make prod-run`'s
    own foreground `docker compose up`. Untrapped, so the loop's plain
    `kill -TERM` on its process group stops it immediately and cleanly, with no
    stop-flag polling needed."""
    stub = tmp_path / "run-stub.sh"
    _write_executable(
        stub,
        f'echo "run $$" >> "{run_log}"\necho "$$" > "{pid_file}"\nexec sleep 999\n',
    )
    return stub


def _crash_stub(tmp_path: pathlib.Path, run_log: pathlib.Path) -> pathlib.Path:
    """A "server" that exits the instant it starts — simulates a crash so the
    loop's own crash-detection and backoff-restart path can be exercised without
    ever sending it a signal."""
    stub = tmp_path / "crash-stub.sh"
    _write_executable(stub, f'echo "run $$" >> "{run_log}"\nexit 1\n')
    return stub


def _build_stub(
    tmp_path: pathlib.Path,
    build_log: pathlib.Path,
    cursor_file: pathlib.Path,
    fail_flag: pathlib.Path,
) -> pathlib.Path:
    """Mirrors `prod-build`'s own shape: a build step, then — only once it has
    actually succeeded — a cursor write. `fail_flag`'s presence fails the build
    step before the cursor line ever runs, exactly like a failing
    `docker compose build` aborting the real recipe before its own cursor
    write."""
    stub = tmp_path / "build-stub.sh"
    _write_executable(
        stub,
        "set -e\n"
        f'head="$(git rev-parse HEAD)"\n'
        f'echo "build $head" >> "{build_log}"\n'
        f'if [ -f "{fail_flag}" ]; then exit 1; fi\n'
        f'echo "$head" > "{cursor_file}"\n'
        f'echo "built $head" >> "{build_log}"\n',
    )
    return stub


def _start_loop(clone_dir: pathlib.Path, env_overrides: dict[str, str]) -> subprocess.Popen:
    env = {
        **os.environ,
        "DEPLOY_INTERVAL": "3600",  # never fires on its own during a test
        "DEPLOY_CRASH_BACKOFF": "0.2",
        **env_overrides,
    }
    return subprocess.Popen(
        ["bash", _script_path()],
        cwd=clone_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _tick(proc: subprocess.Popen) -> None:
    """Forces an immediate fetch-check, standing in for DEPLOY_INTERVAL elapsing."""
    proc.send_signal(signal.SIGUSR1)


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@dataclasses.dataclass
class _Deployment:
    """Everything a test needs to drive and inspect one loop run — the paths
    its stub build/run commands write to, and the started process."""

    proc: subprocess.Popen
    run_log: pathlib.Path
    pid_file: pathlib.Path
    build_log: pathlib.Path
    cursor_file: pathlib.Path
    fail_flag: pathlib.Path
    deploy_log: pathlib.Path
    deploy_run_log: pathlib.Path


def _start_deployment(
    tmp_path: pathlib.Path,
    clone_dir: pathlib.Path,
    *,
    run_cmd: pathlib.Path | None = None,
) -> _Deployment:
    """Starts the loop against `clone_dir` with the standard build/run stubs
    (unless `run_cmd` names a different one, e.g. a crash stub), waits for the
    first child to report in, and hands back every path a test asserts on.

    `cursor_file` doubles as DEPLOY_BUILD_CURSOR_FILE: the startup-build check
    reads it, and a successful build stub writes it — the same file prod-build
    and the loop agree on in production. Pre-written here with the clone's
    CURRENT head when a test hasn't already seeded it with something else, so
    an ordinary test sees no startup build; a test of that startup path writes
    an OLDER commit into it before calling this, and that value survives."""
    run_log = tmp_path / "run.log"
    pid_file = tmp_path / "run.pid"
    build_log = tmp_path / "build.log"
    cursor_file = tmp_path / "cursor"
    fail_flag = tmp_path / "fail-flag"
    deploy_log = tmp_path / "deploy.log"
    deploy_run_log = tmp_path / "deploy-run.log"
    run_stub = run_cmd or _run_stub(tmp_path, run_log, pid_file)
    build_stub = _build_stub(tmp_path, build_log, cursor_file, fail_flag)
    if not cursor_file.exists():
        cursor_file.write_text(_git(clone_dir, "rev-parse", "HEAD"))

    proc = _start_loop(
        clone_dir,
        {
            "DEPLOY_RUN_CMD": str(run_stub),
            "DEPLOY_BUILD_CMD": str(build_stub),
            "DEPLOY_LOG": str(deploy_log),
            "DEPLOY_RUN_LOG": str(deploy_run_log),
            "DEPLOY_BUILD_CURSOR_FILE": str(cursor_file),
        },
    )
    _wait_until(lambda: _read_or_empty(run_log).count("\n") >= 1)
    return _Deployment(
        proc, run_log, pid_file, build_log, cursor_file, fail_flag, deploy_log, deploy_run_log
    )


@pytest.fixture
def repos(tmp_path: pathlib.Path):
    _require_git()
    origin = tmp_path / "origin"
    _init_repo(origin)
    first = _commit(origin, "feat(penny): first (#1)", "A")
    clone_dir = tmp_path / "clone"
    _clone(origin, clone_dir)
    return origin, clone_dir, first


def test_startup_builds_first_when_head_is_already_ahead_of_the_last_build(tmp_path, repos):
    """HEAD can already be ahead of the recorded build at STARTUP — the owner ran
    `git pull` by hand before `make prod-watch`, or a build failed and the loop
    was restarted later. Neither looks "new" to the fetch/ahead check (HEAD
    already equals origin/main), so nothing would ever trigger a build for a
    commit that's already checked out unless the loop builds it itself before
    starting the child."""
    origin, clone_dir, first = repos
    cursor_file = tmp_path / "cursor"
    cursor_file.write_text(first)  # the last successful build was the FIRST commit...
    second = _commit(
        clone_dir, "chore: already checked out, never built", "B"
    )  # ...HEAD moved past it

    dep = _start_deployment(tmp_path, clone_dir)
    try:
        log_text = _read_or_empty(dep.deploy_log)
        assert "building before starting" in log_text
        assert log_text.index("building before starting") < log_text.index("started child"), (
            "the build must run before the child starts"
        )
        assert f"built {second}" in _read_or_empty(dep.build_log)
        assert cursor_file.read_text().strip() == second, (
            "the startup build's own success still advances the cursor"
        )
    finally:
        _stop(dep.proc)


def test_ahead_clean_ff_able_rebuilds_and_restarts(tmp_path, repos):
    """The main path: a new commit on origin/main, a clean checkout on main that
    can fast-forward — the loop merges, builds, stops the old child, and starts
    the new one."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        second = _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert _git(clone_dir, "rev-parse", "HEAD") == second
        assert dep.cursor_file.read_text().strip() == second
        assert f"built {second}" in _read_or_empty(dep.build_log)
    finally:
        _stop(dep.proc)


def test_dirty_tracked_tree_blocks_the_deploy(tmp_path, repos):
    """A tracked, uncommitted change blocks the merge entirely — the old child
    keeps running and nothing is built."""
    origin, clone_dir, _first = repos
    (clone_dir / "file.txt").write_text("locally modified, never committed")
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: "tracked changes" in _read_or_empty(dep.deploy_log))
        assert not dep.build_log.exists()
        assert _read_or_empty(dep.run_log).count("\n") == 1, "old child was never restarted"
    finally:
        _stop(dep.proc)


def test_checkout_on_another_branch_blocks_the_deploy(tmp_path, repos):
    """Checked out on a branch other than DEPLOY_BRANCH never acts, however
    clean or fast-forwardable the tree is."""
    origin, clone_dir, _first = repos
    _git(clone_dir, "checkout", "-q", "-b", "other")
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(
            lambda: (
                "not main" in _read_or_empty(dep.deploy_log)
                or "on other" in _read_or_empty(dep.deploy_log)
            )
        )
        assert _git(clone_dir, "rev-parse", "--abbrev-ref", "HEAD") == "other"
        assert _read_or_empty(dep.run_log).count("\n") == 1
    finally:
        _stop(dep.proc)


def test_diverged_history_blocks_the_deploy(tmp_path, repos):
    """A local commit origin/main never got (so HEAD is no longer an ancestor of
    it) is a non-fast-forward — never resolved by resetting or checking out."""
    origin, clone_dir, _first = repos
    _commit(clone_dir, "chore: local-only commit", "local change")
    local_head = _git(clone_dir, "rev-parse", "HEAD")
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: "not a fast-forward" in _read_or_empty(dep.deploy_log))
        assert _git(clone_dir, "rev-parse", "HEAD") == local_head, "never reset or checked out"
        assert _read_or_empty(dep.run_log).count("\n") == 1
    finally:
        _stop(dep.proc)


def test_unreachable_origin_logs_the_reason_and_leaves_the_child_running(tmp_path, repos):
    """A `git fetch` that fails outright (origin's own path is gone) is logged
    with its reason and never touches the running child — just another skip,
    retried next cycle like any other."""
    _origin, clone_dir, _first = repos
    _git(clone_dir, "remote", "set-url", "origin", str(tmp_path / "no-such-origin"))
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        _tick(dep.proc)

        _wait_until(lambda: "fetch origin main failed" in _read_or_empty(dep.deploy_log))
        assert not dep.build_log.exists()
        assert _read_or_empty(dep.run_log).count("\n") == 1
    finally:
        _stop(dep.proc)


def test_failed_build_leaves_old_child_running_and_never_advances_the_cursor(tmp_path, repos):
    """A build failure is logged and the old Penny keeps running — and because
    the build stub mirrors prod-build's own shape (build step, THEN a cursor
    write), the changelog cursor is never written either: nothing a failed
    build produced is baked into any image, so there is nothing for a
    crash-restarted old container to announce that never actually deployed.
    The ff-only merge still advances HEAD, which is also why the SAME commit's
    build is never retried on the next tick — only a genuinely new commit is."""
    origin, clone_dir, first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    dep.fail_flag.write_text("fail this build")
    try:
        second = _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: "build FAILED" in _read_or_empty(dep.deploy_log))
        assert _git(clone_dir, "rev-parse", "HEAD") == second, (
            "still fast-forwards despite the failure"
        )
        # The cursor already exists (pre-seeded with the startup build check's own
        # commit, same as HEAD at the time) — what must NOT happen is it moving to
        # the failed commit.
        assert dep.cursor_file.read_text().strip() == first, (
            "a failed build must never advance the changelog cursor"
        )
        assert _read_or_empty(dep.run_log).count("\n") == 1, "the old child was never touched"
        assert _read_or_empty(dep.build_log).count("build ") == 1

        # A second tick: HEAD already equals origin/main, so nothing is "ahead" and the
        # build is not retried. Wait for THIS tick's own "up to date" line (logged only
        # on the no-op path) rather than a fixed sleep, so the assertion below can never
        # race the loop's own processing of the signal.
        _tick(dep.proc)
        _wait_until(lambda: "up to date" in _read_or_empty(dep.deploy_log))
        assert _read_or_empty(dep.build_log).count("build ") == 1, (
            "the same failed commit is never retried"
        )
    finally:
        _stop(dep.proc)


def test_crashed_child_is_restarted_after_backoff(tmp_path, repos):
    """When the run command exits on its own (a crash), the loop notices within
    its poll granularity and restarts it after DEPLOY_CRASH_BACKOFF — with no
    git activity and no signal needed to trigger it."""
    _origin, clone_dir, _first = repos
    dep = _start_deployment(
        tmp_path, clone_dir, run_cmd=_crash_stub(tmp_path, tmp_path / "run.log")
    )
    try:
        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2, timeout=5.0)
        assert "exited on its own" in _read_or_empty(dep.deploy_log)
    finally:
        _stop(dep.proc)


def test_stopping_the_loop_stops_the_child_cleanly(tmp_path, repos):
    """Sending the loop a stop signal stops the running child too — Penny never
    keeps running headless after the loop that supervises it has gone away."""
    _origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        child_pid = int(dep.pid_file.read_text().strip())

        dep.proc.send_signal(signal.SIGTERM)
        dep.proc.wait(timeout=5)

        def _child_gone() -> bool:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return True
            return False

        _wait_until(_child_gone)
    finally:
        _stop(dep.proc)


def test_child_runs_detached_from_the_terminal(tmp_path, repos):
    """Under `set -m` a background child that touches the controlling tty is
    stopped by the kernel (SIGTTOU/SIGTTIN) and never resumes (#2971) — so the
    child must read no terminal (stdin from /dev/null) and write to none
    (stdout and stderr appended to DEPLOY_RUN_LOG). The stub records whether
    its own stdin is a terminal and emits on both streams; the old stubs never
    touched a terminal, which is exactly why this went unseen."""
    _origin, clone_dir, _first = repos
    run_log = tmp_path / "run.log"
    pid_file = tmp_path / "run.pid"
    tty_marker = tmp_path / "tty.marker"
    stub = tmp_path / "run-stub.sh"
    _write_executable(
        stub,
        f'echo "run $$" >> "{run_log}"\n'
        f'if [ -t 0 ]; then echo "tty" >> "{tty_marker}"; else echo "notty" >> "{tty_marker}"; fi\n'
        'echo "child stdout line"\n'
        'echo "child stderr line" >&2\n'
        f'echo "$$" > "{pid_file}"\n'
        "exec sleep 999\n",
    )
    dep = _start_deployment(tmp_path, clone_dir, run_cmd=stub)
    try:
        _wait_until(lambda: "notty" in _read_or_empty(tty_marker))
        assert _read_or_empty(tty_marker).strip() == "notty", "stdin is not a terminal"

        def _run_log_has_both_streams() -> bool:
            text = _read_or_empty(dep.deploy_run_log)
            return "child stdout line" in text and "child stderr line" in text

        _wait_until(_run_log_has_both_streams)
        run_log_text = _read_or_empty(dep.deploy_run_log)
        assert "child stdout line" in run_log_text
        assert "child stderr line" in run_log_text, "stderr lands in the run log too"
    finally:
        _stop(dep.proc)


def test_stopped_child_is_still_stopped_by_the_loop(tmp_path, repos):
    """A child the kernel has stopped (SIGSTOP — what an attached `docker compose
    up` suffers under `set -m`, #2971) can't act on the loop's TERM until it is
    continued: stop_child sends SIGCONT along with TERM, so the shutdown still
    lands — the loop exits and the stopped child is gone, instead of hanging."""
    _origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        child_pid = int(dep.pid_file.read_text().strip())
        os.kill(child_pid, signal.SIGSTOP)

        _wait_until(lambda: _child_is_stopped(child_pid))

        dep.proc.send_signal(signal.SIGTERM)
        dep.proc.wait(timeout=5)

        def _child_gone() -> bool:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return True
            return False

        _wait_until(_child_gone)
    finally:
        _stop(dep.proc)
