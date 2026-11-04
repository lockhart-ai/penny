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
# Where the child's own stdout/stderr is appended (#2971). The child must never
# hold the terminal — an attached `docker compose up` has the kernel
# SIGTTOU/SIGTTIN the whole background group dead the moment it touches the
# controlling tty — so its output never goes to the tty directly: it reaches the
# loop's stdout through a pipe (tee, #2980) and its stdin comes from /dev/null
# instead.
DEPLOY_RUN_LOG="${DEPLOY_RUN_LOG:-data/penny/logs/deploy-run.log}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-main}"
DEPLOY_CRASH_BACKOFF="${DEPLOY_CRASH_BACKOFF:-30}"
# Which commit the image tagged `penny:prod` was built from (#2973): the image's own
# `git_commit` LABEL, set from the GIT_COMMIT build arg inside the Dockerfile on
# every build path. A successful prod-build retags that tag; a failed one leaves
# the old image standing — so this replaces the old host-side `.changelog_cursor`
# (which a built-but-never-started image advanced for it) as the "last successful
# build" record the startup check compares HEAD against. Production runs its own
# `penny:prod` tag (#2977), which no worktree's `make check` ever retags — that is
# the tag whose build record this reads. Overridable so a test can
# stub it (`cat <file>` in front of a file the build stub writes) and so the script
# runs against a scratch image name; empty output (no image, no label) means "not
# built", exactly what a missing cursor used to mean. Two shell traps shape how
# the default is written: the go-template braces CANNOT sit inside a
# `${VAR:-default}` expansion (bash's brace matching corrupts the word — verified:
# `${X:-a {{b}}}` resolves to garbage), and the value must carry NO quotes and NO
# redirects anyway, because the call site expands it unquoted, where quote
# removal never happens — quotes would arrive inside the words, and a redirect
# would be an argument, not a redirect. So: default assigned on its own line,
# single-quoted there (quote removal happens AT the assignment), landing as a
# quote-free value.
DEPLOY_IMAGE_COMMIT_CMD="${DEPLOY_IMAGE_COMMIT_CMD:-}"
if [ -z "$DEPLOY_IMAGE_COMMIT_CMD" ]; then
    DEPLOY_IMAGE_COMMIT_CMD='docker image inspect penny:prod --format {{.Config.Labels.git_commit}}'
fi

mkdir -p "$(dirname "$DEPLOY_LOG")" "$(dirname "$DEPLOY_RUN_LOG")"

# One line into $DEPLOY_LOG AND onto the loop's own stdout (#2980) — in a
# terminal that's the tty `make prod-watch` was started in, under launchd it's
# the plist's StandardOutPath — so a working deploy is visible where the loop
# was started instead of looking like a hang.
#
# Built once, written twice, NO pipeline (#2991): under `set -m` bash puts
# every pipeline in its own process group, and the old `printf | tee -a`
# pipeline's printf could exit before the shell's setpgid call for it landed,
# printing `child setpgid (…): Operation not permitted` on every log line.
# Plain redirections are not jobs — they never get a process group, so there
# is nothing for bash to setpgid. (The child's own `| tee` pipeline stays: it
# runs once per child start inside the backgrounded subshell and is what
# streams Penny's output.)
log() {
    local line
    line="$(printf '[deploy-loop %s] %s' "$(date '+%Y-%m-%d %H:%M:%S')" "$1")"
    printf '%s\n' "$line" >>"$DEPLOY_LOG"
    printf '%s\n' "$line"
}

child_pid=""
stopping=""
immediate=""

# Starts $DEPLOY_RUN_CMD in a subshell wrapping a pipeline: the child execs with
# stdin from /dev/null and both streams fed to `tee`, which appends them to
# $DEPLOY_RUN_LOG AND streams them to the loop's stdout (#2980) — so Penny's
# output is visible in the terminal `make prod-watch` runs in (the plist's
# StandardOutPath under launchd), while the child itself never touches the tty:
# its stdout is a pipe, not the terminal, so the kernel never stops the group
# with SIGTTOU/SIGTTIN (#2971). Penny's own logs still go to penny.log inside
# the container.
#
# The whole pipeline sits inside one explicit subshell, so $! — and thus
# $child_pid — names that subshell, not `tee`: the subshell is the process
# whose exit means "the child crashed" (when the child dies, its pipe closes,
# tee sees EOF and exits, and the subshell exits with the pipeline's
# pipefail'd status). `set -m` gives the backgrounded subshell its own process
# group and the pipeline's two members are forked inside it, so stop_child's
# group kill still takes down the child and tee together.
#
# `trap : TERM` keeps that shutdown clean: bash defers a trapped signal until
# the command it is waiting on completes, so when the group TERM lands the
# subshell stays alive while its two members die, collects both their
# statuses as the pipeline ends, and only then exits — neither is ever left
# an unreaped orphan. The trap must be a handler (:), never '' — an ignored
# signal is inherited across exec, and tee and the child would then ignore
# TERM too, hanging the shutdown.
start_child() {
    (trap : TERM; exec $DEPLOY_RUN_CMD < /dev/null 2>&1 | tee -a "$DEPLOY_RUN_LOG") &
    child_pid=$!
    log "started child (pid $child_pid): $DEPLOY_RUN_CMD"
}

# Stops the current child's whole process group and waits for it to actually exit.
# Never resets or checks out the working tree — this only ever touches the running
# processes, not the git state (#2955).
stop_child() {
    [ -n "$child_pid" ] || return 0
    # SIGCONT along with the TERM (#2971): a stopped child can't act on TERM until
    # it's continued, so the shutdown would hang on it — continue the group first,
    # then take it down.
    kill -CONT "-$child_pid" 2>/dev/null || kill -CONT "$child_pid" 2>/dev/null
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

# HEAD can already be ahead of the last successful build at STARTUP — the owner ran
# `git pull` by hand before `make prod-watch`, or a build failed and the loop was
# restarted later. Neither looks "new" to fetch_and_maybe_deploy, which only ever
# compares HEAD against origin: once they're equal, nothing will ever trigger a
# build for a commit that's already checked out. So build once here if the commit
# the image tagged `penny:prod` was built from (its own `git_commit` LABEL, #2973)
# doesn't match HEAD, or the label can't be read at all. Short hashes on both
# sides: GIT_COMMIT rides every build path as `git rev-parse --short HEAD`, and
# the loop compares the same form. A failure is logged and the child still starts
# on whatever image already exists — never silently stuck with no Penny at all.
startup_build_if_needed() {
    startup_head="$(git rev-parse --short HEAD)"
    startup_built="$($DEPLOY_IMAGE_COMMIT_CMD 2>/dev/null)"
    if [ "$startup_head" = "$startup_built" ]; then
        return
    fi

    log "HEAD ($startup_head) is ahead of the last successful build (${startup_built:-none}) — building before starting"
    if ! $DEPLOY_BUILD_CMD >>"$DEPLOY_LOG" 2>&1; then
        log "startup build FAILED at $startup_head — starting on the existing image anyway"
    fi
}

startup_build_if_needed
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
