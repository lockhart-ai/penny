"""The provider lock ``make eval-remote`` takes, driven from real shells (#2219, #2821).

The lock is a shell file the recipe sources (``scripts/eval-provider-lock.sh``), so what is
tested is that file, in the shell the recipe runs it in — not a description of it.  Each test
stands up the recipe's own lock block in one or more ``sh`` processes against a lock directory
of its own and reads what they print and what they leave on disk.

No test waits on a clock: the script's wait between two looks at the lock is its own function,
and the driver replaces it — with nothing, so a bounded wait runs out at once, or with a read
of the driver's stdin, so a wait ends exactly when the test says it does.

A pid file holds its holder's own process start time alongside its pid (#2821), so a lock's
holder being gone is one of three ways a name on it goes stale — the others are the pid being
alive but belonging to a DIFFERENT process now (the OS reused it) and the pid being the
LOOKING run's own (it can never legitimately be its own lock's holder).  All three are covered
below; only the first is reachable by an ordinary kill.

An EMPTY pid file — every one written before this change, or one a `ps`-less take writes —
records no start time to be wrong, so it is never enough on its own to call a live holder
stale: that holder is judged by aliveness alone, same as before this change.  Covered below
too, since a run already holding a lock under the old format is exactly what a rollout mixes
with the new one.
"""

from __future__ import annotations

import pathlib
import queue
import signal
import subprocess
import threading
import time
import weakref

_LOCK_SCRIPT = pathlib.Path("scripts") / "eval-provider-lock.sh"

# The recipe's lock block: source the file, take the lock or stop, release it however the
# shell ends, then run — here, until stdin closes.
_RUN = """
. "$1"
{clock}
provider_lock_acquire "$2" "$3" "$4" 1 || exit 1
trap provider_lock_release EXIT
trap 'exit 143' TERM
echo "running $$"
read -r _ || exit 0
"""
# The wait passes at once.
_NO_CLOCK = "provider_lock_sleep() { :; }"
# The wait lasts until the test writes a line.
_CLOCK_ON_STDIN = "provider_lock_sleep() { read -r _; }"
# A shell that never took the lock calling release on it.
_RELEASE_SOMEONE_ELSES = '. "$1"; PROVIDER_LOCK="$2"; provider_lock_release'
# A run that plants a lock naming ITS OWN pid as the holder before it ever tries to take
# one — standing in for the OS handing a run the exact pid a stale lock already names, with
# no need to wait for a real pid to cycle all the way back around.
_PLANT_OWN_PID_THEN_RUN = """
. "$1"
{clock}
PROVIDER_LOCK="$(provider_lock_path "$2" "$3")"
mkdir -p "$PROVIDER_LOCK"
provider_lock_start_time "$$" >"$PROVIDER_LOCK/$$"
PROVIDER_LOCK=""
provider_lock_acquire "$2" "$3" "$4" 1 || exit 1
trap provider_lock_release EXIT
trap 'exit 143' TERM
echo "running $$"
read -r _ || exit 0
"""

_SIGTERM_EXIT = 128 + signal.SIGTERM


def _lock_script() -> str:
    """The lock file itself, found above this test the way the Makefile is."""
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / _LOCK_SCRIPT).is_file():
            return str(parent / _LOCK_SCRIPT)
    raise AssertionError(f"no {_LOCK_SCRIPT} above this test")


def _start(
    locks: pathlib.Path, provider: str, *, wait: int = 60, clock: str = _NO_CLOCK
) -> subprocess.Popen[str]:
    """One run against ``provider``, left running with its stdin open."""
    return subprocess.Popen(
        [
            "sh",
            "-c",
            _RUN.format(clock=clock),
            "sh",
            _lock_script(),
            str(locks),
            provider,
            f"{wait}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


# One background reader per run, so two calls for the same run (this file makes some) share
# one drain and never lose a line the first call read off the pipe before the second call's
# own opening line had even arrived.  A plain `select` on the pipe's fd was tried and
# discarded: once ``readline()`` has pulled a buffered chunk holding MORE than one line into
# Python, the fd itself goes quiet and `select` reports not-ready even though the next line is
# already sitting there ready to read — it waited out the whole timeout on a run that was not
# hung at all.  Only the QUEUE is shared across calls — each call to ``_lines_until`` starts
# its own fresh ``printed``, same as the unbounded version this replaced, so a second call for
# a run `_lines_until` already read from does not re-hand-back the first call's lines.
_READERS: weakref.WeakKeyDictionary[subprocess.Popen[str], queue.Queue[str | None]] = (
    weakref.WeakKeyDictionary()
)


def _reader(run: subprocess.Popen[str]) -> queue.Queue[str | None]:
    """The queue a background thread is draining ``run``'s stdout into."""
    cached = _READERS.get(run)
    if cached is not None:
        return cached
    assert run.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def _drain_stdout() -> None:
        assert run.stdout is not None
        for line in run.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=_drain_stdout, daemon=True).start()
    _READERS[run] = lines
    return lines


def _lines_until(run: subprocess.Popen[str], opening: str, *, timeout: float = 2.0) -> list[str]:
    """What a run printed up to and including its first line opening with ``opening`` —
    bounded: every legitimate case here resolves almost at once (the clock is always
    faked; the margin is just process-scheduling noise), so a run that is actually hung
    (a bug, not a feature — #2821 wedged the whole suite for 19+ minutes this way) fails
    this assertion within ``timeout`` instead of wedging the whole suite."""
    lines = _reader(run)
    printed: list[str] = []
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"no line opening with {opening!r} within {timeout}s — likely hung: {printed}"
            )
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            raise AssertionError(
                f"no line opening with {opening!r} within {timeout}s — likely hung: {printed}"
            ) from None
        if line is None:
            raise AssertionError(f"the run ended without printing {opening!r}: {printed}")
        printed.append(line.rstrip("\n"))
        if printed[-1].startswith(opening):
            return printed


def _finish(run: subprocess.Popen[str]) -> tuple[int, list[str]]:
    """Close a run's stdin, which ends it, and return its exit status and what it printed."""
    printed, _ = run.communicate(input="", timeout=30)
    return run.returncode, printed.splitlines()


def _holders(lock: pathlib.Path) -> list[str]:
    """The pid files a lock holds."""
    return sorted(entry.name for entry in lock.iterdir())


def _a_pid_that_is_gone() -> str:
    """The pid of a shell that has already exited."""
    return subprocess.run(
        ["sh", "-c", "echo $$"], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_a_second_run_on_a_held_provider_waits_and_gives_up_when_the_wait_runs_out(
    tmp_path: pathlib.Path,
) -> None:
    """Two runs pinned to one provider take turns, and the wait for a turn is bounded.

    The second run says whose run it is waiting on at every look, and when the bound passes it
    stops with the lock, its holder and what to do — it does not start anyway.  A run pinned to
    a DIFFERENT provider is not held up at all.  And nothing but the holder gives the lock up:
    not the run that gave up, and not another shell calling release on it."""
    lock = tmp_path / "Groq.lock"
    holder = _start(tmp_path, "Groq")
    assert _lines_until(holder, "running") == [
        f"eval: holding provider Groq ({lock})",
        f"running {holder.pid}",
    ]
    assert _holders(lock) == [f"{holder.pid}"]

    elsewhere = tmp_path / "Some_Cloud_EU.lock"
    status, printed = _finish(_start(tmp_path, "Some Cloud/EU"))
    assert (status, printed[0]) == (0, f"eval: holding provider Some Cloud/EU ({elsewhere})")
    assert not elsewhere.exists(), "and it gave its own lock back when it ended"

    status, printed = _finish(_start(tmp_path, "Groq", wait=2))
    held = f"{lock} is held by pid {holder.pid}"
    assert status == 1
    assert printed == [
        f"eval: provider Groq is busy — {held}; waiting (0s of 2s)",
        f"eval: provider Groq is busy — {held}; waiting (1s of 2s)",
        f"eval: gave up after 2s waiting for provider Groq — {held}. Wait for that run and "
        f"start again, raise EVAL_PROVIDER_LOCK_WAIT, or remove {lock} if that pid is not an "
        "eval run.",
    ]

    not_the_holder = ["sh", "-c", _RELEASE_SOMEONE_ELSES, "sh", _lock_script(), f"{lock}"]
    subprocess.run(not_the_holder, check=True)
    assert _holders(lock) == [f"{holder.pid}"], "the lock is still its holder's"

    assert _finish(holder) == (0, [])
    assert not lock.exists(), "and the holder released it by ending"


def test_a_waiting_run_takes_the_provider_when_its_holder_finishes(tmp_path: pathlib.Path) -> None:
    """The wait ends in a turn: once the holder is done the waiting run holds the provider.

    And it gives the lock back however it ends — here it is killed, which is how a run an agent
    abandons ends."""
    lock = tmp_path / "Groq.lock"
    holder = _start(tmp_path, "Groq")
    _lines_until(holder, "running")

    waiter = _start(tmp_path, "Groq", clock=_CLOCK_ON_STDIN)
    assert _lines_until(waiter, "eval: provider Groq is busy") == [
        f"eval: provider Groq is busy — {lock} is held by pid {holder.pid}; waiting (0s of 60s)"
    ]
    _finish(holder)
    assert waiter.stdin is not None
    waiter.stdin.write("\n")
    waiter.stdin.flush()
    assert _lines_until(waiter, "running") == [
        f"eval: holding provider Groq ({lock})",
        f"running {waiter.pid}",
    ]
    assert _holders(lock) == [f"{waiter.pid}"]

    waiter.terminate()
    assert waiter.wait(timeout=30) == _SIGTERM_EXIT
    assert not lock.exists(), "a run that was killed released the provider on its way out"
    waiter.stdin.close()
    assert waiter.stdout is not None
    waiter.stdout.close()


def test_a_lock_nobody_holds_is_cleared_and_taken(tmp_path: pathlib.Path) -> None:
    """A lock is only as good as its holder being alive.

    A run killed outright leaves its pid file behind, and the next run reads that the pid is
    gone and clears it.  A run killed between making the lock and putting its name in leaves
    an EMPTY lock: the next run gives it one wait — a run may be taking it right now — and
    clears it when it is still empty after."""
    lock = tmp_path / "Groq.lock"
    gone = _a_pid_that_is_gone()
    lock.mkdir()
    (lock / gone).touch()
    status, printed = _finish(_start(tmp_path, "Groq"))
    assert status == 0
    assert printed[:2] == [
        f"eval: provider Groq had a stale lock — its holder, pid {gone}, is gone; cleared {lock}",
        f"eval: holding provider Groq ({lock})",
    ]
    assert not lock.exists()

    lock.mkdir()
    status, printed = _finish(_start(tmp_path, "Groq"))
    assert status == 0
    assert printed[:3] == [
        f"eval: provider Groq is busy — {lock} is held by no pid yet (a run is taking it, or "
        "died doing so); waiting (0s of 60s)",
        f"eval: provider Groq had an abandoned, empty lock; cleared {lock}",
        f"eval: holding provider Groq ({lock})",
    ]
    assert not lock.exists()


def test_a_live_holder_with_an_empty_pid_file_is_waited_on_not_cleared(
    tmp_path: pathlib.Path,
) -> None:
    """Rollout compatibility: a run already holding a lock under the OLD format (every pid
    file before this change) — or one a `ps`-less take wrote — has no start time in its pid
    file for a later look to compare against.  That is not evidence of anything: the holder
    is judged by aliveness alone, same as before #2821, so a live old-format holder is waited
    on and given up on at the bound, same as any other live holder — never torn out from
    under it just because its file records nothing to check."""
    lock = tmp_path / "Groq.lock"
    other = subprocess.Popen(["sleep", "30"])
    try:
        lock.mkdir()
        (lock / f"{other.pid}").touch()
        status, printed = _finish(_start(tmp_path, "Groq", wait=2))
        held = f"{lock} is held by pid {other.pid}"
        assert status == 1
        assert printed == [
            f"eval: provider Groq is busy — {held}; waiting (0s of 2s)",
            f"eval: provider Groq is busy — {held}; waiting (1s of 2s)",
            f"eval: gave up after 2s waiting for provider Groq — {held}. Wait for that run and "
            f"start again, raise EVAL_PROVIDER_LOCK_WAIT, or remove {lock} if that pid is not "
            "an eval run.",
        ]
        assert _holders(lock) == [f"{other.pid}"], "the lock is still its old-format holder's"
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_a_lock_naming_a_reused_pid_is_cleared_and_taken(tmp_path: pathlib.Path) -> None:
    """A dead pid is not the only way a name on the lock goes bad: the OS can hand a brand
    new process the exact pid an old, finished run's lock still names — a run killed by
    something no trap can catch (SIGKILL) leaves exactly this behind once that pid cycles
    back around.  The recorded holder is alive, but its start time no longer matches the one
    the lock was written with, so the next run to look tells it apart from the real holder
    and clears it rather than waiting on a stranger forever."""
    lock = tmp_path / "Groq.lock"
    other = subprocess.Popen(["sleep", "30"])
    try:
        lock.mkdir()
        (lock / f"{other.pid}").write_text("not this process's real start time")
        status, printed = _finish(_start(tmp_path, "Groq"))
        assert status == 0
        assert printed[:2] == [
            f"eval: provider Groq had a stale lock — its holder, pid {other.pid}, is a "
            f"different process now; cleared {lock}",
            f"eval: holding provider Groq ({lock})",
        ]
        assert not lock.exists()
    finally:
        other.terminate()
        other.wait(timeout=10)


def test_a_run_given_its_own_pid_as_the_lock_s_holder_treats_it_as_stale(
    tmp_path: pathlib.Path,
) -> None:
    """Pid reuse's sharpest edge: the OS hands a WAITING run the exact pid a lock already
    names as its holder.  That run can never legitimately be its own lock's holder — it is
    the one still looking — so it recognizes its own pid on the lock as stale on sight,
    clears it, and takes the lock cleanly rather than waiting on itself forever."""
    lock = tmp_path / "Groq.lock"
    run = subprocess.Popen(
        [
            "sh",
            "-c",
            _PLANT_OWN_PID_THEN_RUN.format(clock=_NO_CLOCK),
            "sh",
            _lock_script(),
            str(tmp_path),
            "Groq",
            "60",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert _lines_until(run, "running") == [
        f"eval: provider Groq had a stale lock — its holder, pid {run.pid}, is this very "
        f"shell, waiting on it; cleared {lock}",
        f"eval: holding provider Groq ({lock})",
        f"running {run.pid}",
    ]
    assert _holders(lock) == [f"{run.pid}"]
    assert _finish(run) == (0, [])
    assert not lock.exists()


def test_a_path_that_is_not_this_recipes_lock_is_refused_and_left_alone(
    tmp_path: pathlib.Path,
) -> None:
    """The lock directory is a variable, so the path a lock would take may already be
    something else: a file, a link, a directory with someone's files in it.  None of them is
    waited on, taken or cleared — the run stops and says which path."""
    lock = tmp_path / "Groq.lock"
    refusal = [f"eval: {lock} is not a lock this recipe made — move it away and run again."]

    lock.write_text("keep")
    assert _finish(_start(tmp_path, "Groq")) == (1, refusal)
    assert lock.read_text() == "keep"
    lock.unlink()

    lock.mkdir()
    (lock / "notes.txt").write_text("keep")
    assert _finish(_start(tmp_path, "Groq")) == (1, refusal)
    assert (lock / "notes.txt").read_text() == "keep"
    (lock / "notes.txt").unlink()
    lock.rmdir()

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    lock.symlink_to(elsewhere)
    assert _finish(_start(tmp_path, "Groq")) == (1, refusal)
    assert lock.is_symlink() and elsewhere.is_dir()
