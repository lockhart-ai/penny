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
import re
import shutil
import signal
import subprocess
import time

import pytest

_SCRIPT = pathlib.Path("scripts") / "deploy-loop.sh"
# The shared runtime-path rule (#3063): one git pathspec per line in
# scripts/runtime-paths — the file both the deploy loop (its default
# DEPLOY_RUNTIME_PATHS) and scripts/store_git_info.sh read. Pinned to the file
# itself by test_runtime_paths_rule_is_the_one_shared_file below; spelled
# joined-for-env where a test passes it as the override variable.
_RUNTIME_PATHS_RULE = (
    "penny",
    ":(exclude)penny/penny/tests/",
    ":(exclude)*.md",
    "similarity",
    "github_api",
    "docker-compose.yml",
)
_RUNTIME_PATHS_JOINED = " ".join(_RUNTIME_PATHS_RULE)
_SCRIPT_PATHS_FILE = pathlib.Path("scripts") / "runtime-paths"
_IDLE_SCRIPT = pathlib.Path("scripts") / "penny-idle.sh"
_MAKEFILE = pathlib.Path("Makefile")
_COMPOSE_FILE = pathlib.Path("docker-compose.yml")


def _repo_file(relpath: pathlib.Path) -> str:
    """A repo-root file's absolute path, found by walking up — the same walk
    `_script_path` does, so the tests stay runnable from any working directory
    and from inside the gate's container, where the repo-root files a test needs
    are mounted read-only one by one (see docker-compose.override.yml)."""
    for parent in pathlib.Path(__file__).resolve().parents:
        candidate = parent / relpath
        if candidate.is_file():
            return str(candidate)
    raise AssertionError(f"no {relpath} above this test")


def _script_path() -> str:
    return _repo_file(_SCRIPT)


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


def _commit_path(repo: pathlib.Path, relpath: str, content: str, message: str) -> str:
    """Commits a change to one specific path — the runtime-path tests (#3049)
    need commits whose ONLY change is a given file, not the fixture's file.txt."""
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
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
    built_file: pathlib.Path,
    fail_flag: pathlib.Path,
) -> pathlib.Path:
    """Mirrors `prod-build`'s own shape: a build step, then — only once it has
    actually succeeded — a write of the built commit to `built_file`, standing
    in for the image's own `git_commit` LABEL the real build retags (#2973).
    `fail_flag`'s presence fails the build step before that line ever runs,
    exactly like a failing `docker compose build` aborting the real recipe
    before its own retag."""
    stub = tmp_path / "build-stub.sh"
    _write_executable(
        stub,
        "set -e\n"
        # The SHORT hash — the same form the GIT_COMMIT build arg (and so the
        # image's git_commit label) carries in production, and the form the
        # loop's startup check compares.
        f'head="$(git rev-parse --short HEAD)"\n'
        f'echo "build $head" >> "{build_log}"\n'
        f'if [ -f "{fail_flag}" ]; then exit 1; fi\n'
        f'echo "$head" > "{built_file}"\n'
        f'echo "built $head" >> "{build_log}"\n',
    )
    return stub


def _stop_stub(
    tmp_path: pathlib.Path, stop_log: pathlib.Path, pid_file: pathlib.Path
) -> pathlib.Path:
    """Stands in for `make prod-stop` (#3015): records that it ran, and whether
    the child was already gone when it did — the loop's shutdown must run the
    stop command only AFTER stop_child has taken the child down, never while it
    is still alive. The child's pid comes from the same `pid_file` the run stub
    wrote, so the liveness check is the kernel's own record, never a sleep."""
    stub = tmp_path / "stop-stub.sh"
    _write_executable(
        stub,
        f'child_pid="$(cat "{pid_file}")"\n'
        f'if kill -0 "$child_pid" 2>/dev/null; then\n'
        f'  echo "stop while child alive" >> "{stop_log}"\n'
        "else\n"
        f'  echo "stop after child gone" >> "{stop_log}"\n'
        "fi\n"
        f'echo "stop" >> "{stop_log}"\n',
    )
    return stub


def _idle_stub(
    tmp_path: pathlib.Path, control_file: pathlib.Path, call_log: pathlib.Path
) -> pathlib.Path:
    """Stands in for `DEPLOY_IDLE_CMD` (#3026): exits with the code
    `control_file` holds (0 idle / 1 busy / 2 unreadable), prints the fixed
    "last activity" timestamp the loop's waiting line names, and records each
    call — a test bounds the polling by the call count, never a fixed sleep."""
    stub = tmp_path / "idle-stub.sh"
    _write_executable(
        stub,
        f'echo call >> "{call_log}"\n'
        f'echo "2026-11-05 04:05:21"\n'
        f'rc="$(cat "{control_file}")"\n'
        f'if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then\n'
        f'  echo "penny-idle: cannot read the database" >&2\n'
        "fi\n"
        f'exit "$rc"\n',
    )
    return stub


def _start_loop(
    clone_dir: pathlib.Path,
    env_overrides: dict[str, str],
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.Popen:
    env = {
        **os.environ,
        "DEPLOY_INTERVAL": "3600",  # never fires on its own during a test
        "DEPLOY_CRASH_BACKOFF": "0.2",
        "DEPLOY_IDLE_POLL": "0.2",  # the idle wait polls fast under test (#3026)
        **env_overrides,
    }
    return subprocess.Popen(
        ["bash", _script_path()],
        cwd=clone_dir,
        env=env,
        stdout=stdout if stdout is not None else subprocess.DEVNULL,
        stderr=stderr if stderr is not None else subprocess.DEVNULL,
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
    its stub build/run/stop commands write to, and the started process.
    `loop_stdout` is what the loop itself wrote to its stdout — what a terminal
    (or the launchd plist's StandardOutPath) would have shown (#2980);
    `loop_stderr` is what the loop wrote to its stderr (#2991)."""

    proc: subprocess.Popen
    run_log: pathlib.Path
    pid_file: pathlib.Path
    build_log: pathlib.Path
    built_file: pathlib.Path
    fail_flag: pathlib.Path
    stop_log: pathlib.Path
    deploy_log: pathlib.Path
    deploy_run_log: pathlib.Path
    loop_stdout: pathlib.Path
    loop_stderr: pathlib.Path
    idle_control: pathlib.Path
    idle_call_log: pathlib.Path


def _start_deployment(
    tmp_path: pathlib.Path,
    clone_dir: pathlib.Path,
    *,
    run_cmd: pathlib.Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> _Deployment:
    """Starts the loop against `clone_dir` with the standard build/run stubs
    (unless `run_cmd` names a different one, e.g. a crash stub), waits for the
    first child to report in, and hands back every path a test asserts on.

    `built_file` stands in for the image's own `git_commit` LABEL (#2973):
    the loop reads the built commit from it via the stubbed
    DEPLOY_IMAGE_COMMIT_CMD, and a successful build stub writes it — the same
    contract prod-build and the loop agree on in production. Pre-written here
    with the clone's CURRENT head when a test hasn't already seeded it with
    something else, so an ordinary test sees no startup build; a test of that
    startup path writes an OLDER commit into it before calling this, and that
    value survives."""
    run_log = tmp_path / "run.log"
    pid_file = tmp_path / "run.pid"
    build_log = tmp_path / "build.log"
    built_file = tmp_path / "built"
    fail_flag = tmp_path / "fail-flag"
    stop_log = tmp_path / "stop.log"
    deploy_log = tmp_path / "deploy.log"
    deploy_run_log = tmp_path / "deploy-run.log"
    loop_stdout = tmp_path / "loop-stdout.log"
    loop_stderr = tmp_path / "loop-stderr.log"
    idle_control = tmp_path / "idle-control"
    idle_call_log = tmp_path / "idle-calls.log"
    run_stub = run_cmd or _run_stub(tmp_path, run_log, pid_file)
    build_stub = _build_stub(tmp_path, build_log, built_file, fail_flag)
    stop_stub = _stop_stub(tmp_path, stop_log, pid_file)
    idle_stub = _idle_stub(tmp_path, idle_control, idle_call_log)
    # The idle check answers "idle" by default (#3026), so an ordinary test's
    # redeploy path is unchanged; an idle-wait test flips `idle_control` to
    # "1" (busy) or "2" (unreadable) before its tick.
    idle_control.write_text("0")
    if not built_file.exists():
        built_file.write_text(_git(clone_dir, "rev-parse", "--short", "HEAD"))

    with loop_stdout.open("w") as stdout_file, loop_stderr.open("w") as stderr_file:
        proc = _start_loop(
            clone_dir,
            {
                "DEPLOY_RUN_CMD": str(run_stub),
                "DEPLOY_BUILD_CMD": str(build_stub),
                "DEPLOY_STOP_CMD": str(stop_stub),
                "DEPLOY_IDLE_CMD": str(idle_stub),
                "DEPLOY_LOG": str(deploy_log),
                "DEPLOY_RUN_LOG": str(deploy_run_log),
                # No quotes around the path on purpose (see the variable's own
                # comment in the script): the loop expands the command unquoted,
                # where quote characters would arrive inside the word.
                "DEPLOY_IMAGE_COMMIT_CMD": f"cat {built_file}",
                # The runtime pathspec (#3049) narrowed to the fixture's own
                # commit shape: the repos below commit `file.txt`, so with this
                # a `file.txt` change is a "runtime" commit and anything else
                # (e.g. a docs file) is a skip. The tests that exercise the
                # script's real default pass their own value here.
                "DEPLOY_RUNTIME_PATHS": "file.txt",
                **(env_overrides or {}),
            },
            stdout=stdout_file.fileno(),
            stderr=stderr_file.fileno(),
        )
    _wait_until(lambda: _read_or_empty(run_log).count("\n") >= 1)
    return _Deployment(
        proc,
        run_log,
        pid_file,
        build_log,
        built_file,
        fail_flag,
        stop_log,
        deploy_log,
        deploy_run_log,
        loop_stdout,
        loop_stderr,
        idle_control,
        idle_call_log,
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
    built_file = tmp_path / "built"
    built_file.write_text(_git(clone_dir, "rev-parse", "--short", first))  # the last
    # successful build was the FIRST commit...
    _commit(clone_dir, "chore: already checked out, never built", "B")  # ...HEAD moved past it
    second_short = _git(clone_dir, "rev-parse", "--short", "HEAD")

    dep = _start_deployment(tmp_path, clone_dir)
    try:
        log_text = _read_or_empty(dep.deploy_log)
        assert "building before starting" in log_text
        assert log_text.index("building before starting") < log_text.index("started child"), (
            "the build must run before the child starts"
        )
        assert f"built {second_short}" in _read_or_empty(dep.build_log)
        assert built_file.read_text().strip() == second_short, (
            "the startup build's own success still retags the image"
        )
    finally:
        _stop(dep.proc)


def test_startup_fetches_origin_before_the_startup_build(tmp_path, repos):
    """origin/main moves ahead while the loop is down (#3020): on start, the
    loop's one startup fetch+fast-forward advances HEAD BEFORE the startup build
    check runs — so the first build and child start happen at the fetched commit,
    not the stale local one, with no interval wait and no tick (the startup fetch
    runs unconditionally, before anything else)."""
    origin, clone_dir, _first = repos
    second = _commit(origin, "fix(penny): second (#2)", "B")
    second_short = second[:7]

    dep = _start_deployment(tmp_path, clone_dir)
    try:
        _wait_until(
            lambda: (
                "building before starting" in _read_or_empty(dep.deploy_log)
                and f"built {second_short}" in _read_or_empty(dep.build_log)
            )
        )
        assert _git(clone_dir, "rev-parse", "HEAD") == second, (
            "the startup fetch fast-forwarded HEAD before anything was built"
        )
        log_text = _read_or_empty(dep.deploy_log)
        assert "fast-forwarding" in log_text
        assert (
            log_text.index("fast-forwarding")
            < log_text.index("building before starting")
            < (log_text.index("started child"))
        ), "the fetch, then the build, then the child start — in that order"
        assert f"built {dep.built_file.read_text().strip()}" in _read_or_empty(dep.build_log)
        assert dep.built_file.read_text().strip() == _git(
            clone_dir, "rev-parse", "--short", "HEAD"
        ), "the startup build's own success retags the image at the fetched commit"
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
        second_short = _git(clone_dir, "rev-parse", "--short", "HEAD")
        assert dep.built_file.read_text().strip() == second_short
        assert f"built {second_short}" in _read_or_empty(dep.build_log)
        # A redeploy stops the old child and starts a new one — prod-run brings the
        # old containers down itself, so the stop command is shutdown-only (#3015).
        # Asserted here, BEFORE the finally's TERM (which legitimately runs it).
        assert not dep.stop_log.exists(), (
            "a redeploy must not run the stop command — only the shutdown does"
        )

        # A follow-up cycle with HEAD already at origin/main is a no-op: it logs its
        # own "up to date" and builds and redeploys nothing (#3021 — only a cycle
        # that ADVANCED HEAD may reach the build). The startup fetch already logged
        # one "up to date", so count occurrences, never test presence.
        ups_before = _read_or_empty(dep.deploy_log).count("up to date")
        builds_before = _read_or_empty(dep.build_log).count("build ")
        _tick(dep.proc)
        _wait_until(lambda: _read_or_empty(dep.deploy_log).count("up to date") > ups_before)
        assert _read_or_empty(dep.build_log).count("build ") == builds_before, (
            "an up-to-date cycle runs no build"
        )
        assert _read_or_empty(dep.run_log).count("\n") == 2, "an up-to-date cycle never redeploys"
    finally:
        _stop(dep.proc)


def test_non_runtime_commits_advance_head_without_building(tmp_path, repos):
    """(#3049) With the shared runtime-path rule as the override, commits
    touching only docs or only penny's own test suite change nothing Penny
    runs: HEAD still fast-forwards, but nothing is built and the running child
    is never restarted — the whole burst of merges is judged by one diff from
    the deployed commit to the new HEAD, and that diff is quiet."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(
        tmp_path, clone_dir, env_overrides={"DEPLOY_RUNTIME_PATHS": _RUNTIME_PATHS_JOINED}
    )
    try:
        _commit_path(origin, "docs/notes.md", "docs", "docs: a note")
        tests_sha = _commit_path(origin, "penny/penny/tests/test_new_case.py", "t", "test: a case")

        _tick(dep.proc)

        _wait_until(lambda: _git(clone_dir, "rev-parse", "HEAD") == tests_sha)
        _wait_until(lambda: "no Penny server changes" in _read_or_empty(dep.deploy_log))
        assert not dep.build_log.exists(), "a non-runtime commit runs no build"
        assert _read_or_empty(dep.run_log).count("\n") == 1, "the child was never restarted"
    finally:
        _stop(dep.proc)


def test_runtime_paths_rule_is_the_one_shared_file():
    """(#3063) The rule lives in scripts/runtime-paths alone: the file's
    non-comment lines are exactly the runtime pathspec list, the deploy loop
    reads it as its default (no inline copy of the list in the script), and
    the Makefile no longer carries its own copy to pass through. A static
    read — the loop's subprocess tests above cover the behaviour, with the
    rule itself stubbed by the override variable."""
    file_lines = [
        stripped
        for line in pathlib.Path(_repo_file(_SCRIPT_PATHS_FILE)).read_text().splitlines()
        if (stripped := line.split("#", 1)[0].strip())
    ]
    assert file_lines == list(_RUNTIME_PATHS_RULE), (
        "scripts/runtime-paths is the one definition of Penny's runtime paths"
    )
    script_text = pathlib.Path(_script_path()).read_text()
    assert "runtime-paths" in script_text, (
        "the deploy loop reads the shared file for its default DEPLOY_RUNTIME_PATHS"
    )
    assert _RUNTIME_PATHS_JOINED not in script_text, (
        "the script carries no inline copy of the list to drift away from the file"
    )
    makefile_text = pathlib.Path(_repo_file(_MAKEFILE)).read_text()
    assert "DEPLOY_RUNTIME_PATHS" not in makefile_text, (
        "the Makefile carries no copy of the list either — the script's file read"
        " is the default, and an environment override still reaches it"
    )


def test_default_runtime_paths_come_from_the_shared_file(tmp_path, repos):
    """(#3063) With no DEPLOY_RUNTIME_PATHS override (the variable passed
    empty, exactly what the script sees unset), the loop's default is read off
    scripts/runtime-paths — so a commit touching a path the rule names builds
    and restarts, and one touching none of them is skipped, with no list
    configured anywhere in the environment. The clone carries the rule file
    itself: in production the deploy checkout always does, since it is
    committed."""
    origin, clone_dir, _first = repos
    (clone_dir / "scripts").mkdir()
    (clone_dir / "scripts" / "runtime-paths").write_text(
        pathlib.Path(_repo_file(_SCRIPT_PATHS_FILE)).read_text()
    )
    dep = _start_deployment(tmp_path, clone_dir, env_overrides={"DEPLOY_RUNTIME_PATHS": ""})
    try:
        docs_sha = _commit_path(origin, "docs/notes.md", "docs", "docs: a note")

        _tick(dep.proc)

        _wait_until(lambda: _git(clone_dir, "rev-parse", "HEAD") == docs_sha)
        _wait_until(lambda: "no Penny server changes" in _read_or_empty(dep.deploy_log))
        assert not dep.build_log.exists(), (
            "a non-runtime commit runs no build under the file default"
        )

        second = _commit_path(
            origin, "penny/penny/tools/new_tool.py", "x", "feat(penny): a tool (#2)"
        )
        _tick(dep.proc)
        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert f"built {second[:7]}" in _read_or_empty(dep.build_log), (
            "a runtime commit deploys under the file default"
        )
    finally:
        _stop(dep.proc)


def test_runtime_commit_builds_and_restarts(tmp_path, repos):
    """(#3049) A commit touching penny/penny/ outside its tests — the paths the
    default runtime pathspec exists for — builds and restarts exactly as every
    commit always did; the skip logic changes nothing on the runtime path."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(
        tmp_path, clone_dir, env_overrides={"DEPLOY_RUNTIME_PATHS": _RUNTIME_PATHS_JOINED}
    )
    try:
        second = _commit_path(
            origin, "penny/penny/tools/new_tool.py", "x", "feat(penny): a tool (#2)"
        )

        _tick(dep.proc)

        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert _git(clone_dir, "rev-parse", "HEAD") == second
        assert f"built {second[:7]}" in _read_or_empty(dep.build_log)
        assert dep.built_file.read_text().strip() == second[:7]
    finally:
        _stop(dep.proc)


def test_skipped_commit_then_runtime_commit_deploys_against_the_deployed_commit(tmp_path, repos):
    """(#3049) A docs-only commit is skipped; the next runtime commit deploys
    even though a skip sits between it and the last deploy — its diff is taken
    against the DEPLOYED commit, never a skipped HEAD, so nothing the skips sat
    on top of is missed. And once it deploys, the deployed record moves to the
    commit actually running: a further docs-only commit is judged against IT."""
    origin, clone_dir, first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        first_short = _git(clone_dir, "rev-parse", "--short", "HEAD")
        docs_sha = _commit_path(origin, "docs/notes.md", "docs", "docs: a note")

        _tick(dep.proc)

        _wait_until(
            lambda: (
                f"skip {docs_sha}: no Penny server changes since {first_short}"
                in _read_or_empty(dep.deploy_log)
            )
        )
        assert not dep.build_log.exists(), "the docs-only commit built nothing"
        assert _read_or_empty(dep.run_log).count("\n") == 1

        second = _commit(origin, "fix(penny): second (#2)", "B")
        _tick(dep.proc)
        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert f"built {second[:7]}" in _read_or_empty(dep.build_log), (
            "the runtime commit deployed despite the skip between it and the deploy"
        )
        assert dep.built_file.read_text().strip() == second[:7]

        # The deployed record advanced with the redeploy: the next docs-only
        # commit is skipped against the commit now running, not the old build.
        third_docs_sha = _commit_path(origin, "docs/more.md", "docs", "docs: another note")
        _tick(dep.proc)
        _wait_until(
            lambda: (
                f"skip {third_docs_sha}: no Penny server changes since {second[:7]}"
                in _read_or_empty(dep.deploy_log)
            )
        )
        assert _read_or_empty(dep.build_log).count("build ") == 1, "no further build ran"
        assert _read_or_empty(dep.run_log).count("\n") == 2, "and no further restart"
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

        # The startup fetch logs the same "not a fast-forward" line (the local
        # commit predates the loop), so count occurrences, never test presence.
        skips_before = _read_or_empty(dep.deploy_log).count("not a fast-forward")
        _tick(dep.proc)
        _wait_until(
            lambda: _read_or_empty(dep.deploy_log).count("not a fast-forward") > skips_before
        )
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
        # The startup fetch fails the same way before the loop even settles in,
        # so count occurrences of this tick's line, never test presence.
        fails_before = _read_or_empty(dep.deploy_log).count("fetch origin main failed")
        _tick(dep.proc)

        _wait_until(
            lambda: _read_or_empty(dep.deploy_log).count("fetch origin main failed") > fails_before
        )
        assert not dep.build_log.exists()
        assert _read_or_empty(dep.run_log).count("\n") == 1
    finally:
        _stop(dep.proc)


def test_failed_build_leaves_old_child_running_and_never_retags_the_image(tmp_path, repos):
    """A build failure is logged and the old Penny keeps running — and because
    the build stub mirrors prod-build's own shape (build step, THEN a retag),
    the image is never retagged to the failed commit: nothing a failed build
    produced is baked into any image, so there is nothing for a
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
        # The built-commit record already exists (pre-seeded with the startup build
        # check's own commit, same as HEAD at the time) — what must NOT happen is it
        # moving to the failed commit.
        assert dep.built_file.read_text().strip() == _git(
            clone_dir, "rev-parse", "--short", first
        ), "a failed build must never retag the image"
        assert _read_or_empty(dep.run_log).count("\n") == 1, "the old child was never touched"
        assert _read_or_empty(dep.build_log).count("build ") == 1

        # A second tick: HEAD already equals origin/main, so nothing is "ahead" and the
        # build is not retried. Wait for THIS tick's own "up to date" line — counted
        # past the one the startup fetch already logged, so the wait can never be
        # satisfied before the tick's cycle has run, and the assertion below can
        # never race the loop's own processing of the signal.
        ups_before = _read_or_empty(dep.deploy_log).count("up to date")
        _tick(dep.proc)
        _wait_until(lambda: _read_or_empty(dep.deploy_log).count("up to date") > ups_before)
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
    keeps running headless after the loop that supervises it has gone away. And
    since #2992 the group kill no longer reaches the child's `docker compose up`,
    the shutdown also runs DEPLOY_STOP_CMD once, after the child is down (#3015):
    the same `prod-stop` target prod-run opens with, so stopping the watcher
    stops Penny's containers too."""
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

        _wait_until(lambda: _read_or_empty(dep.stop_log).count("stop\n") >= 1)
        stop_text = _read_or_empty(dep.stop_log)
        assert stop_text.count("stop\n") == 1, "the stop command runs exactly once"
        assert "stop after child gone" in stop_text, (
            "the stop command runs only after the child is stopped"
        )
        assert "stop while child alive" not in stop_text
        assert "stopping penny:" in _read_or_empty(dep.deploy_log), (
            "the shutdown names the stop command it ran in the loop's own log"
        )
    finally:
        _stop(dep.proc)


def test_child_runs_detached_from_the_terminal(tmp_path, repos):
    """Under `set -m` a background child that touches the controlling tty is
    stopped by the kernel (SIGTTOU/SIGTTIN) and never resumes (#2971) — so the
    child must read no terminal (stdin from /dev/null) and write to none
    (stdout and stderr through a pipe into DEPLOY_RUN_LOG). Its output still
    reaches the loop's stdout (#2980) — tee'd, so a working deploy is visible
    in the terminal that ran `make prod-watch` — but that is a pipe, never the
    tty itself. The stub records whether its own stdin and stdout are
    terminals and emits on both streams; the old stubs never touched a
    terminal, which is exactly why this went unseen."""
    _origin, clone_dir, _first = repos
    run_log = tmp_path / "run.log"
    pid_file = tmp_path / "run.pid"
    tty_marker = tmp_path / "tty.marker"
    stub = tmp_path / "run-stub.sh"
    _write_executable(
        stub,
        f'echo "run $$" >> "{run_log}"\n'
        "for fd in 0 1; do\n"
        '  if [ -t "$fd" ]; then state=tty; else state=notty; fi\n'
        f'  echo "fd$fd-$state" >> "{tty_marker}"\n'
        "done\n"
        'echo "child stdout line"\n'
        'echo "child stderr line" >&2\n'
        f'echo "$$" > "{pid_file}"\n'
        "exec sleep 999\n",
    )
    dep = _start_deployment(tmp_path, clone_dir, run_cmd=stub)
    try:
        _wait_until(lambda: "fd0-notty" in _read_or_empty(tty_marker))
        marker_text = _read_or_empty(tty_marker)
        assert "fd0-notty" in marker_text, "the child's stdin is not a terminal"
        assert "fd1-notty" in marker_text, (
            "the child's stdout is not a terminal either — it is a pipe"
        )
        assert "tty" not in marker_text.replace("-notty", ""), (
            "no stream of the child's is a terminal"
        )

        def _run_log_has_both_streams() -> bool:
            text = _read_or_empty(dep.deploy_run_log)
            return "child stdout line" in text and "child stderr line" in text

        _wait_until(_run_log_has_both_streams)
        run_log_text = _read_or_empty(dep.deploy_run_log)
        assert "child stdout line" in run_log_text
        assert "child stderr line" in run_log_text, "stderr lands in the run log too"

        def _loop_stdout_has_both_streams() -> bool:
            text = _read_or_empty(dep.loop_stdout)
            return "child stdout line" in text and "child stderr line" in text

        _wait_until(_loop_stdout_has_both_streams)
        stdout_text = _read_or_empty(dep.loop_stdout)
        assert "child stdout line" in stdout_text
        assert "child stderr line" in stdout_text, (
            "the child's output streams to the loop's stdout too (#2980)"
        )
    finally:
        _stop(dep.proc)


def test_loop_lines_appear_on_stdout(tmp_path, repos):
    """The loop's own log lines stream to its stdout as well as the log file
    (#2980) — the terminal that ran `make prod-watch` shows the loop working
    instead of looking hung; under launchd that stdout is the plist's
    StandardOutPath, so nothing changes there. Across several forced cycles the
    loop's stderr stays free of `setpgid` text (#2991): under `set -m` the old
    `printf | tee` pipeline put printf in its own process group, and it could
    exit before the shell's setpgid call landed, printing
    `child setpgid (…): Operation not permitted` on every log line — with no
    pipeline there is nothing for bash to setpgid."""
    _origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        _wait_until(lambda: "started child" in _read_or_empty(dep.loop_stdout))
        stdout_text = _read_or_empty(dep.loop_stdout)
        assert stdout_text.startswith("[deploy-loop "), (
            "the loop's own lines carry their log prefix on stdout too"
        )
        assert "started child" in _read_or_empty(dep.deploy_log), (
            "the line still lands in the log file as well"
        )

        # Several forced cycles — force each fetch-check with USR1 rather than
        # waiting out DEPLOY_INTERVAL, waiting for each one's "up to date" line
        # so the ticks can't coalesce into a single cycle.
        for _ in range(4):
            before = _read_or_empty(dep.loop_stdout).count("up to date")
            _tick(dep.proc)
            _wait_until(
                lambda seen=before: _read_or_empty(dep.loop_stdout).count("up to date") > seen
            )
        stdout_text = _read_or_empty(dep.loop_stdout)
        assert stdout_text.count("up to date") >= 4, (
            "the forced cycles each logged their own line on stdout"
        )

        stderr_text = _read_or_empty(dep.loop_stderr)
        assert "setpgid" not in stderr_text, (
            "the loop's stderr carries no setpgid warnings across the ticks (#2991)"
        )
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


def test_busy_penny_builds_but_does_not_restart(tmp_path, repos):
    """A successful build restarts Penny only once she is idle (#3026): while
    the idle check answers busy, the new commit is still fast-forwarded and
    built, the wait is logged once (not once per poll), and the old Penny keeps
    serving — never restarted mid-turn."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    dep.idle_control.write_text("1")
    try:
        second = _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: "waiting for Penny to be idle" in _read_or_empty(dep.deploy_log))
        # The wait keeps polling while she stays busy — bounded by the check's
        # own call count, never a fixed sleep guessed to be long enough.
        _wait_until(lambda: _read_or_empty(dep.idle_call_log).count("\n") >= 3)
        # The per-poll waiting line (distinct from the "build succeeded …
        # waiting for Penny to be idle" line, which shares its tail) once, with
        # the check's last-activity output.
        assert (
            _read_or_empty(dep.deploy_log).count("waiting for Penny to be idle (last activity") == 1
        ), "the wait is logged once, not once per poll"
        assert "last activity 2026-11-05 04:05:21" in _read_or_empty(dep.deploy_log)
        assert _read_or_empty(dep.run_log).count("\n") == 1, "busy Penny was never restarted"
        assert _git(clone_dir, "rev-parse", "HEAD") == second, (
            "the new commit still fast-forwards and builds while Penny is busy"
        )
        assert dep.built_file.read_text().strip() == _git(clone_dir, "rev-parse", "--short", "HEAD")
    finally:
        _stop(dep.proc)


def test_idle_penny_is_redeployed_without_waiting(tmp_path, repos):
    """The idle path: the check answers idle (the default control value), so
    the redeploy happens at once — no waiting line in the log at all."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    try:
        second = _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        # The check ran exactly once, answered idle, and the redeploy happened
        # — no per-poll waiting line in the log at all (that line shares its
        # tail with the "build succeeded …" line, so match the whole shape).
        assert "waiting for Penny to be idle (last activity" not in _read_or_empty(
            dep.deploy_log
        ), "an idle Penny is redeployed at once, with no wait logged"
        assert _read_or_empty(dep.idle_call_log).count("\n") == 1, (
            "the idle check ran once and answered idle"
        )
        assert _git(clone_dir, "rev-parse", "HEAD") == second
    finally:
        _stop(dep.proc)


def test_failed_idle_check_logs_and_deploys_anyway(tmp_path, repos):
    """An unreadable database (exit 2) must never block the deploy forever: the
    failure is logged (the check's own error output with it) and Penny is
    redeployed anyway."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    dep.idle_control.write_text("2")
    try:
        second = _commit(origin, "fix(penny): second (#2)", "B")

        _tick(dep.proc)

        _wait_until(lambda: "idle check FAILED" in _read_or_empty(dep.deploy_log))
        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert _git(clone_dir, "rev-parse", "HEAD") == second, "redeployed despite the failed check"
        assert "cannot read the database" in _read_or_empty(dep.deploy_log), (
            "the check's own error output is in the loop's log"
        )
    finally:
        _stop(dep.proc)


def test_stop_while_waiting_for_idle_stops_cleanly_without_redeploy(tmp_path, repos):
    """A TERM/INT while the loop waits for idle stops the loop through the
    existing shutdown path — Penny is stopped with it, and nothing is
    redeployed."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    dep.idle_control.write_text("1")
    try:
        _commit(origin, "fix(penny): second (#2)", "B")
        _tick(dep.proc)
        _wait_until(lambda: "waiting for Penny to be idle" in _read_or_empty(dep.deploy_log))

        dep.proc.send_signal(signal.SIGTERM)
        dep.proc.wait(timeout=5)

        assert _read_or_empty(dep.run_log).count("\n") == 1, "the stop never redeployed"
        assert "stop arrived while waiting for Penny to be idle" in _read_or_empty(
            dep.deploy_log
        ), "the wait says why it gave up, then leaves the deploy to the shutdown"
        assert "stop" in _read_or_empty(dep.stop_log), "the existing shutdown path ran"
    finally:
        _stop(dep.proc)


def test_newer_commit_landing_while_waiting_is_built_and_deploys_once_idle(tmp_path, repos):
    """While the loop waits for idle, origin keeps moving: a newer commit is
    fast-forwarded and built right away (the same fetch-and-ff path), and it is
    THAT newest build that deploys once Penny finally goes idle."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    dep.idle_control.write_text("1")
    try:
        _commit(origin, "fix(penny): second (#2)", "B")
        _tick(dep.proc)
        _wait_until(lambda: "waiting for Penny to be idle" in _read_or_empty(dep.deploy_log))

        third = _commit(origin, "fix(penny): third (#3)", "C")
        third_short = third[:7]
        _wait_until(lambda: f"built {third_short}" in _read_or_empty(dep.build_log))
        assert _git(clone_dir, "rev-parse", "HEAD") == third, (
            "the newer commit was fast-forwarded and built while waiting"
        )
        assert _read_or_empty(dep.run_log).count("\n") == 1, "still busy, still not restarted"

        dep.idle_control.write_text("0")
        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert dep.built_file.read_text().strip() == third_short, (
            "the build that deployed is the newest one"
        )
    finally:
        _stop(dep.proc)


def test_child_dying_while_waiting_redeploys_without_waiting_for_idle(tmp_path, repos):
    """If the child dies while the loop waits for idle, there is no activity
    left to protect — waiting out the whole idle threshold would leave Penny
    down for it — so the redeploy happens at once, busy check or not."""
    origin, clone_dir, _first = repos
    dep = _start_deployment(tmp_path, clone_dir)
    dep.idle_control.write_text("1")
    try:
        _commit(origin, "fix(penny): second (#2)", "B")
        _tick(dep.proc)
        _wait_until(lambda: "waiting for Penny to be idle" in _read_or_empty(dep.deploy_log))

        child_pid = int(dep.pid_file.read_text().strip())
        os.kill(child_pid, signal.SIGKILL)

        _wait_until(lambda: "nothing left to protect" in _read_or_empty(dep.deploy_log))
        _wait_until(lambda: _read_or_empty(dep.run_log).count("\n") >= 2)
        assert dep.idle_control.read_text().strip() == "1", (
            "redeployed even though the check still answered busy"
        )
    finally:
        _stop(dep.proc)


def _sqlite(db: pathlib.Path, sql: str) -> None:
    result = subprocess.run(["sqlite3", str(db), sql], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_penny_idle_script_reports_busy_idle_and_read_errors(tmp_path):
    """`scripts/penny-idle.sh` against a real, throwaway SQLite database: a
    recent model call makes it busy (exit 1), timestamps older than
    DEPLOY_IDLE_SECONDS make it idle (exit 0, printing the newest timestamp for
    the loop's waiting line), an empty table counts as idle (nothing to
    protect), and an unreadable database is a read error (exit 2), never a
    silent 'idle'. Needs the sqlite3 CLI on PATH — a missing one is a hard,
    reported FAILURE here, never a silent skip."""
    if shutil.which("sqlite3") is None:
        pytest.fail(
            "sqlite3 is not installed in this environment — scripts/penny-idle.sh "
            "reads the newest promptlog/messagelog timestamps through it, and this "
            "test cannot verify the script without it"
        )
    script = _repo_file(_IDLE_SCRIPT)
    db = tmp_path / "penny.db"
    _sqlite(
        db, "CREATE TABLE promptlog (timestamp TEXT); CREATE TABLE messagelog (timestamp TEXT);"
    )

    # A model call moments ago: busy, whatever the message log says.
    _sqlite(
        db,
        "INSERT INTO promptlog VALUES (strftime('%Y-%m-%d %H:%M:%f','now'));"
        "INSERT INTO messagelog VALUES (strftime('%Y-%m-%d %H:%M:%f','now','-2 hours'));",
    )
    busy = subprocess.run(["bash", script, str(db)], capture_output=True, text=True, timeout=10)
    assert busy.returncode == 1, busy.stderr

    # Both logs older than the threshold: idle, with the newest timestamp named.
    _sqlite(
        db,
        "UPDATE promptlog SET timestamp = strftime('%Y-%m-%d %H:%M:%f','now','-2 hours');",
    )
    idle = subprocess.run(["bash", script, str(db)], capture_output=True, text=True, timeout=10)
    assert idle.returncode == 0, idle.stderr
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", idle.stdout.strip()), (
        "the newest timestamp is printed for the waiting line"
    )

    # An empty table has a NULL newest timestamp — no activity to protect.
    _sqlite(db, "DELETE FROM promptlog; DELETE FROM messagelog;")
    empty = subprocess.run(["bash", script, str(db)], capture_output=True, text=True, timeout=10)
    assert empty.returncode == 0, empty.stderr

    # A missing database is a read error, not a silent "idle".
    missing = subprocess.run(
        ["bash", script, str(tmp_path / "no-such.db")],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert missing.returncode == 2
    assert "cannot read" in missing.stderr


def _makefile_target_recipe(makefile_text: str, target: str) -> str:
    """A Makefile target's recipe body: every line between `<target>:` and the
    next line starting at column 0 (a new target, variable or section). Recipe
    lines are tab-indented, so the column-0 line ends the body."""
    lines = makefile_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{target}:"))
    body = []
    for line in lines[start + 1 :]:
        if line and not line.startswith("\t"):
            break
        body.append(line)
    return "\n".join(body)


def test_prod_targets_build_and_run_their_own_penny_prod_tag():
    """Production must build and run its own image tag (#2977): every worktree's
    `make check`/`make up` rebuilds and retags the shared `penny` tag, so when
    prod ran that tag too, a crash-restart (`prod-run` alone, no rebuild) could
    bring up a kitten's dev build in production. The prod path (`prod-build`,
    `prod-run`, `prod-ios`) sets PENNY_IMAGE=penny:prod, which docker-compose.yml
    reads for the penny service's `image:`; the shared paths (`up`, and the eval
    recipes' `docker compose run`) set nothing and keep the shared tag. The
    deploy loop's startup build check inspects that same production tag.

    This is a static read of the three files (no docker is invoked): there is no
    existing harness that executes a Makefile or compose recipe, and running one
    would build images and touch containers — the loop's own subprocess tests
    above cover the script's behaviour with the command stubbed."""
    makefile_text = pathlib.Path(_repo_file(_MAKEFILE)).read_text()
    compose_text = pathlib.Path(_repo_file(_COMPOSE_FILE)).read_text()
    script_text = pathlib.Path(_script_path()).read_text()

    assert "image: ${PENNY_IMAGE:-penny}" in compose_text, (
        "docker-compose.yml reads the overridable image, defaulting to the shared tag"
    )

    for target in ("prod-build", "prod-run", "prod-ios"):
        recipe = _makefile_target_recipe(makefile_text, target)
        assert "PENNY_IMAGE=penny:prod" in recipe, f"{target} must set PENNY_IMAGE=penny:prod"

    up_recipe = _makefile_target_recipe(makefile_text, "up")
    assert "PENNY_IMAGE" not in up_recipe, (
        "`make up` keeps the shared `penny` tag — only the prod path is pinned"
    )

    assert (
        "docker image inspect penny:prod --format {{.Config.Labels.git_commit}}" in script_text
    ), "the deploy loop's startup check inspects the production tag, not the shared one"


# A `*_KEY=`/`*_TOKEN=` value as an argument of `env` — the in-container command
# `env` is docker's argv too — or of a valued `-e/--env` flag, puts the secret on
# a command line any local process can read through `ps`. A chain of earlier
# ordinary `NAME=value` assignments may sit between `env` and the key (the eval
# recipes' argument lists), so the match spans them.
_CREDENTIAL_ON_ENV_ARGV = re.compile(
    r"\benv\s+(?:[A-Z][A-Z0-9_]*=\S+\s+)*[A-Z][A-Z0-9_]*(?:KEY|TOKEN)="
)
_CREDENTIAL_ON_RUN_FLAG = re.compile(r"(?:^|\s)(?:--env|-e)\s+[A-Z][A-Z0-9_]*(?:KEY|TOKEN)=")


def _makefile_targets(makefile_text: str) -> list[str]:
    """Every target name defined in the Makefile: column-0 lines with a colon
    that are not variable definitions (the text right after the colon starts
    with `=` for `:=`/`+=`/`?=`)."""
    names = []
    for line in makefile_text.splitlines():
        if line.startswith(("\t", " ", "#")) or ":" not in line:
            continue
        head, _, tail = line.partition(":")
        if tail.startswith("=") or "=" in head:
            continue
        names.extend(head.split())
    return names


def _recipe_logical_lines(recipe: str) -> list[str]:
    """A recipe's logical shell lines: physical tab-indented lines joined on
    trailing backslashes, since that is how the shell receives a recipe —
    `env NAME=... \\` split across physical lines is one argv, not two."""
    logical: list[str] = []
    pending = ""
    for raw in recipe.splitlines():
        chunk = raw.strip()
        pending = f"{pending} {chunk}" if pending else chunk
        if pending.endswith("\\"):
            pending = pending[:-1]
        else:
            logical.append(pending)
            pending = ""
    if pending:
        logical.append(pending)
    return logical


def test_no_recipe_passes_a_credential_on_a_command_line():
    """Eval secrets reach the model through the environment, never argv (#2986):
    `make eval-remote` and `make replay` used to pass `LLM_API_KEY="..."` as an
    argument of the `docker compose run` command line, where any local process
    could read it through `ps`. The recipes export the key in their own shell and
    hand docker a bare `-e NAME` passthrough instead. This is a static read of
    the Makefile (no docker is invoked): no recipe line may carry a
    `*_KEY=`/`*_TOKEN=` value as an argument of `env` or of a valued `-e/--env`
    flag. Shell-side assignments and `export NAME=...` statements are the
    mechanism, not the violation — a lowercase name never matches, and an
    `export`-ed name is not an argument of anything."""
    makefile_text = pathlib.Path(_repo_file(_MAKEFILE)).read_text()
    offenders = []
    for target in _makefile_targets(makefile_text):
        recipe = _makefile_target_recipe(makefile_text, target)
        for line in _recipe_logical_lines(recipe):
            for pattern in (_CREDENTIAL_ON_ENV_ARGV, _CREDENTIAL_ON_RUN_FLAG):
                if pattern.search(line):
                    offenders.append(f"{target}: {line.strip()}")
    assert not offenders, (
        "these recipes put a credential on a command line — export it in the "
        "recipe's shell and pass docker a bare `-e NAME` instead:\n" + "\n".join(offenders)
    )
