# One eval run at a time against one pinned provider. SOURCED by the Makefile's `eval`
# recipe (never run on its own), so the lock's holder is the recipe's own shell.
#
# A pin gets that one provider's capacity and no other, so two runs against the same pin at
# once draw rate limits (HTTP 429) and lose samples to them. Runs against DIFFERENT providers
# share nothing, so each provider has its own lock and they never wait on each other.
#
# The lock is a DIRECTORY in the lock directory, named for the provider, holding one empty
# file named for the holding shell's pid:
#
#   taking it     `mkdir` makes the directory in one step and fails when it exists; the run
#                 then puts its pid file in and checks that file is the only one there.
#   releasing it  the holder removes its own pid file, then the directory (the recipe's EXIT
#                 trap, so however the run ends).
#   a STALE lock  one whose holder is gone (a run killed outright, a reboot). The next run to
#                 find it says so and clears it: the dead pid's file, then the directory.
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
provider_lock_take() {
    mkdir "$PROVIDER_LOCK" 2>/dev/null || return 1
    touch "$PROVIDER_LOCK/$$" 2>/dev/null || return 1
    [ "$(provider_lock_holders)" = "$$" ] && return 0
    rm -f "$PROVIDER_LOCK/$$"
    rmdir "$PROVIDER_LOCK" 2>/dev/null
    return 1
}

# Clear a lock whose holder is gone. Only the run whose `rm` removed the dead pid's file goes
# on to remove the directory, so two runs clearing at once cannot both do it. Fails only when
# the dead pid's file is still there afterwards — a file this user may not remove.
provider_lock_clear_stale() { # <provider> <the dead holder>
    if rm "$PROVIDER_LOCK/$2" 2>/dev/null; then
        rmdir "$PROVIDER_LOCK" 2>/dev/null
        echo "eval: provider $1 had a stale lock — its holder, pid $2, is gone; cleared $PROVIDER_LOCK"
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
        if ! provider_lock_alive "$provider_lock_held"; then
            provider_lock_clear_stale "$1" "$provider_lock_held" && return 1
            provider_lock_blocker="held by pid $provider_lock_held, which is gone, and could not be cleared"
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
