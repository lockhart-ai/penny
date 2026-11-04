#!/usr/bin/env bash
# Watches origin/<branch> for new commits and redeploys through the Makefile's own
# `prod-build`/`prod-run` targets — never a second build path (#2955). Run by
# `make prod-watch`, or under launchd via the rendered
# scripts/com.penny.deployloop.plist.template.
#
# Deliberately NOT `set -e`: this is a long-running supervisor, not a one-shot
# script — a `git merge-base`/`git status` call returning non-zero is routine
# signal, not a crash, and the loop must survive every command it runs failing.
set -uo pipefail
set -m # job control: each backgrounded command gets its own process group, so
       # stopping the child can take its whole group down (docker compose and
       # whatever IT spawns), not just the one pid bash itself tracked.

DEPLOY_INTERVAL="${DEPLOY_INTERVAL:-300}"
DEPLOY_BUILD_CMD="${DEPLOY_BUILD_CMD:-make prod-build}"
DEPLOY_RUN_CMD="${DEPLOY_RUN_CMD:-make prod-run}"
DEPLOY_LOG="${DEPLOY_LOG:-data/penny/logs/deploy-loop.log}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-main}"
DEPLOY_CRASH_BACKOFF="${DEPLOY_CRASH_BACKOFF:-30}"

mkdir -p "$(dirname "$DEPLOY_LOG")"

log() {
    printf '[deploy-loop %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >>"$DEPLOY_LOG"
}

child_pid=""
stopping=""
immediate=""

# Starts $DEPLOY_RUN_CMD in a subshell that `exec`s into it, so $child_pid names the
# real command (no extra bash layer in between) while still getting its own process
# group from `set -m` + `&`.
start_child() {
    (exec $DEPLOY_RUN_CMD) &
    child_pid=$!
    log "started child (pid $child_pid): $DEPLOY_RUN_CMD"
}

# Stops the current child's whole process group and waits for it to actually exit.
# Never resets or checks out the working tree — this only ever touches the running
# processes, not the git state (#2955).
stop_child() {
    [ -n "$child_pid" ] || return 0
    kill -TERM "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null
    wait "$child_pid" 2>/dev/null
    child_pid=""
}

on_term() {
    stopping=1
}

on_usr1() {
    # Test/operator hook: forces the next loop iteration to run a fetch-check
    # immediately, without waiting out DEPLOY_INTERVAL.
    immediate=1
}

trap on_term TERM INT
trap on_usr1 USR1

# Fetches origin/$DEPLOY_BRANCH and, if it is ahead of HEAD, deploys it — but only
# when every one of these holds; anything else is logged with its reason and the
# current version keeps running, retried next cycle:
#   - checked out on $DEPLOY_BRANCH (never acts on a detached HEAD or another branch)
#   - no tracked changes (untracked files are fine — git status --untracked-files=no)
#   - HEAD is an ancestor of origin/$DEPLOY_BRANCH (a true fast-forward is possible)
#
# The `git merge --ff-only` runs, and HEAD advances, BEFORE the build — while the
# OLD Penny keeps serving (prod-build never touches the running container; only
# prod-run, called below, does). That ordering is also the failed-build memory:
# whether or not the build that follows succeeds, HEAD now IS the commit that was
# attempted, so the next cycle's "is origin/main ahead of HEAD?" check is false
# until a genuinely NEW commit lands — the same commit is never retried, with no
# separate state file needed.
fetch_and_maybe_deploy() {
    if ! git fetch origin "$DEPLOY_BRANCH" --quiet 2>>"$DEPLOY_LOG"; then
        log "git fetch origin $DEPLOY_BRANCH failed"
        return
    fi

    local_head="$(git rev-parse HEAD)"
    remote_head="$(git rev-parse "origin/$DEPLOY_BRANCH")"
    if [ "$local_head" = "$remote_head" ]; then
        log "up to date at $local_head"
        return
    fi

    branch="$(git rev-parse --abbrev-ref HEAD)"
    if [ "$branch" != "$DEPLOY_BRANCH" ]; then
        log "skip $remote_head: on $branch, not $DEPLOY_BRANCH"
        return
    fi
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        log "skip $remote_head: tracked changes present"
        return
    fi
    if ! git merge-base --is-ancestor "$local_head" "origin/$DEPLOY_BRANCH"; then
        log "skip $remote_head: $local_head is not an ancestor — not a fast-forward"
        return
    fi

    log "fast-forwarding $local_head -> $remote_head"
    if ! git merge --ff-only "origin/$DEPLOY_BRANCH" >>"$DEPLOY_LOG" 2>&1; then
        log "ff-only merge to $remote_head failed unexpectedly — leaving old version running"
        return
    fi

    log "building $remote_head: $DEPLOY_BUILD_CMD"
    if ! $DEPLOY_BUILD_CMD >>"$DEPLOY_LOG" 2>&1; then
        log "build FAILED at $remote_head — old Penny keeps running; this commit will not be retried, only the next new one"
        return
    fi

    log "build succeeded at $remote_head — redeploying: $DEPLOY_RUN_CMD"
    stop_child
    start_child
    log "redeployed $local_head -> $remote_head"
}

# Interruptible wait: a trapped signal (TERM/INT/USR1) ends it immediately instead
# of blocking the full duration — this is what makes both the crash-backoff and the
# per-second poll below responsive to a stop request or a forced check.
interruptible_sleep() {
    sleep "$1" &
    wait $! 2>/dev/null
}

start_child
last_check="$(date +%s)"
while [ -z "$stopping" ]; do
    if ! kill -0 "$child_pid" 2>/dev/null; then
        log "child (pid $child_pid) exited on its own — restarting after ${DEPLOY_CRASH_BACKOFF}s"
        interruptible_sleep "$DEPLOY_CRASH_BACKOFF"
        if [ -z "$stopping" ]; then
            start_child
        fi
        last_check="$(date +%s)"
        continue
    fi

    now="$(date +%s)"
    if [ -n "$immediate" ] || [ $((now - last_check)) -ge "$DEPLOY_INTERVAL" ]; then
        immediate=""
        fetch_and_maybe_deploy
        last_check="$(date +%s)"
    fi

    interruptible_sleep 1
done

log "stopping"
stop_child
log "stopped"
