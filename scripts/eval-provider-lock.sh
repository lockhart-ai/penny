# One eval run at a time against one pinned provider. SOURCED by the Makefile's `eval`
# recipe (never run on its own), so the lock's holder is the recipe's own shell.
#
# A pin gets that one provider's capacity and no other, so two runs against the same pin at
# once draw rate limits (HTTP 429) and lose samples to them. Runs against DIFFERENT providers
# share nothing, so each provider has its own lock and they never wait on each other.
#
# The lock is a DIRECTORY in the lock directory, named for the provider, holding one file
# named for the holding shell's pid, holding that process's own start time (`ps -o lstart=`):
#
#   taking it     `mkdir` makes the directory in one step and fails when it exists; the run
#                 then writes its pid file (named for its pid, holding its own start time)
#                 and checks that file is the only one there.
#   releasing it  the holder removes its own pid file, then the directory (the recipe's EXIT
#                 trap, so however the run ends — except a kill no trap can catch, SIGKILL,
#                 which is exactly what the next bullet is for).
#   a STALE lock  one whose holder's name on it is no good: the pid is gone (a run killed
#                 outright, a reboot); the pid is alive, its pid file RECORDS a start time, and
#                 that start time no longer matches the pid's real one, because the OS has
#                 since reused that pid for an unrelated process (an EMPTY pid file — every
#                 one before this change, or one a `ps`-less take wrote — records no start
#                 time to be wrong, so a live pid behind one is judged by aliveness alone, same
#                 as before); or the pid is this very shell's own — the one doing the looking,
#                 so it is never legitimately the lock's holder. The next run to find one of
#                 these says so and clears it: the stale pid's file, then the directory.
#
# Nothing ever removes a pid file but the run it names or a run that found that pid dead, and
# `rmdir` only removes an EMPTY directory — so no step here can take a live run's lock away,
# however many runs are waiting, clearing and taking at once. There is no second lock to guard
# the clearing and so none to be left behind.
#
# The one state with nobody's name on it is an EMPTY lock directory: a run between its `mkdir`
# and its pid file, or one killed there (or between removing its pid file and the directory).
# A waiting run that finds it empty on two polls in a row clears it; a run whose directory
# was cleared from under it either fails to write its pid file or finds another run's beside
# it, and in both cases starts again.
#
# Shell-portable on purpose (macOS bash-as-sh, Linux dash): no `local`, no arrays, and every
# variable prefixed `provider_lock_` so nothing in the recipe is overwritten.

PROVIDER_LOCK=""

# The lock for a provider: the provider's name with anything but [A-Za-z0-9._-] made `_`.
provider_lock_path() { # <dir> <provider>
    printf '%s/%s.lock' "$1" "$(printf '%s' "$2" | tr -c 'A-Za-z0-9._-' '_')"
}

# Whether a pid names a live process. `kill -0` alone reads a live process another user owns
# as dead, so `ps` gets the last word where it exists.
provider_lock_alive() { # <pid>
    kill -0 "$1" 2>/dev/null || ps -p "$1" >/dev/null 2>&1
}

# A pid's own process start time, exactly how the pid file records it and how a later look
# reads it again — the one fact a reused pid cannot share with the run that had it before.
provider_lock_start_time() { # <pid>
    ps -o lstart= -p "$1" 2>/dev/null
}

# Why pid $1's name on the lock is no good right now; empty when it still is. Checked in this
# order: this very shell never legitimately holds a lock it is itself looking at (it is the
# one waiting); a pid that is gone cannot hold anything; and — ONLY when the pid file actually
# recorded a start time — a pid that is alive but whose start time no longer matches what was
# recorded is a different process the OS gave the same number, not the run that took the lock.
# An EMPTY pid file (every pid file before this change, or one a `ps`-less take wrote) records
# no start time to compare against, so a live pid behind one is judged by aliveness alone, same
# as before this change — never called stale just for recording nothing, which would steal the
# lock out from under an old-format run still genuinely holding it.
provider_lock_stale_reason() { # <pid>
    if [ "$1" = "$$" ]; then
        printf '%s' 'is this very shell, waiting on it'
    elif ! provider_lock_alive "$1"; then
        printf '%s' 'is gone'
    else
        provider_lock_recorded="$(cat "$PROVIDER_LOCK/$1" 2>/dev/null)"
        if [ -n "$provider_lock_recorded" ] \
            && [ "$(provider_lock_start_time "$1")" != "$provider_lock_recorded" ]; then
            printf '%s' 'is a different process now'
        fi
    fi
}

# What the wait between two looks at the lock is. Its own function so a test can stand in for
# the clock.
provider_lock_sleep() { # <seconds>
    sleep "$1"
}

# Everything in the lock directory, space-separated: the holder's pid, or nothing.
provider_lock_holders() {
    ls "$PROVIDER_LOCK" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
}

# One attempt at the lock. Succeeds only when this shell's pid file is the only one in it.
# Writing the start time is best-effort, not a reason to fail the take: an environment with
# no `ps` (recording nothing) must still take the lock, not spin forever re-clearing its own
# just-written file as though it named someone else (see provider_lock_stale_reason).
provider_lock_take() {
    mkdir "$PROVIDER_LOCK" 2>/dev/null || return 1
    provider_lock_start_time "$$" >"$PROVIDER_LOCK/$$" 2>/dev/null
    [ "$(provider_lock_holders)" = "$$" ] && return 0
    rm -f "$PROVIDER_LOCK/$$"
    rmdir "$PROVIDER_LOCK" 2>/dev/null
    return 1
}

# Clear a lock whose holder's claim on it is no good (see provider_lock_stale_reason for
# <why>). Only the run whose `rm` removed the stale pid's file goes on to remove the
# directory, so two runs clearing at once cannot both do it. Fails only when the stale pid's
# file is still there afterwards — a file this user may not remove.
provider_lock_clear_stale() { # <provider> <the stale holder> <why>
    if rm "$PROVIDER_LOCK/$2" 2>/dev/null; then
        rmdir "$PROVIDER_LOCK" 2>/dev/null
        echo "eval: provider $1 had a stale lock — its holder, pid $2, $3; cleared $PROVIDER_LOCK"
    fi
    [ ! -e "$PROVIDER_LOCK/$2" ]
}

# Why the lock cannot be taken right now, in provider_lock_blocker; or, for a state this run
# can put right itself, put it right and fail so the caller tries again at once.
provider_lock_look() { # <provider>
    provider_lock_held="$(provider_lock_holders)"
    case "$provider_lock_held" in
    "")
        [ -d "$PROVIDER_LOCK" ] || return 1
        if [ "$provider_lock_empty" = 1 ]; then
            rmdir "$PROVIDER_LOCK" 2>/dev/null
            echo "eval: provider $1 had an abandoned, empty lock; cleared $PROVIDER_LOCK"
            provider_lock_empty=0
            return 1
        fi
        provider_lock_empty=1
        provider_lock_blocker="held by no pid yet (a run is taking it, or died doing so)"
        ;;
    *[!0-9]*)
        provider_lock_empty=0
        provider_lock_blocker="held by pids $provider_lock_held (two runs settling who has it)"
        ;;
    *)
        provider_lock_empty=0
        provider_lock_blocker="held by pid $provider_lock_held"
        provider_lock_reason="$(provider_lock_stale_reason "$provider_lock_held")"
        if [ -n "$provider_lock_reason" ]; then
            provider_lock_clear_stale "$1" "$provider_lock_held" "$provider_lock_reason" && return 1
            provider_lock_blocker="held by pid $provider_lock_held, which $provider_lock_reason, and could not be cleared"
        fi
        ;;
    esac
}

# Whether the lock path is something this recipe could have made: nothing, or a directory
# holding nothing but pid files.
provider_lock_is_ours() {
    if [ -L "$PROVIDER_LOCK" ]; then return 1; fi
    if [ ! -e "$PROVIDER_LOCK" ]; then return 0; fi
    if [ ! -d "$PROVIDER_LOCK" ]; then return 1; fi
    case "$(provider_lock_holders)" in
    *[!0-9\ ]*) return 1 ;;
    esac
}

# Take the lock, waiting up to <wait> seconds for a live holder to finish. Sets PROVIDER_LOCK.
provider_lock_acquire() { # <dir> <provider> <wait seconds> [poll seconds]
    PROVIDER_LOCK="$(provider_lock_path "$1" "$2")"
    provider_lock_waited=0
    provider_lock_poll="${4:-15}"
    provider_lock_empty=0
    mkdir -p "$1" || return 1
    while :; do
        if ! provider_lock_is_ours; then
            echo "eval: $PROVIDER_LOCK is not a lock this recipe made — move it away and run again." >&2
            PROVIDER_LOCK=""
            return 1
        fi
        provider_lock_take && break
        provider_lock_look "$2" || continue
        if [ "$provider_lock_waited" -ge "$3" ]; then
            echo "eval: gave up after ${provider_lock_waited}s waiting for provider $2 — $PROVIDER_LOCK is $provider_lock_blocker. Wait for that run and start again, raise EVAL_PROVIDER_LOCK_WAIT, or remove $PROVIDER_LOCK if that pid is not an eval run." >&2
            PROVIDER_LOCK=""
            return 1
        fi
        echo "eval: provider $2 is busy — $PROVIDER_LOCK is $provider_lock_blocker; waiting (${provider_lock_waited}s of ${3}s)"
        provider_lock_sleep "$provider_lock_poll"
        provider_lock_waited=$((provider_lock_waited + provider_lock_poll))
    done
    echo "eval: holding provider $2 ($PROVIDER_LOCK)"
}

# Release the lock this shell holds. Safe to call when nothing was taken: it only ever removes
# this shell's own pid file, and the directory only once that leaves it empty.
provider_lock_release() {
    if [ -n "$PROVIDER_LOCK" ]; then
        rm -f "$PROVIDER_LOCK/$$"
        rmdir "$PROVIDER_LOCK" 2>/dev/null
    fi
    PROVIDER_LOCK=""
}
