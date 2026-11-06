#!/usr/bin/env bash
# Watches origin/<branch> for new commits and redeploys through the Makefile's own
# `prod-build`/`prod-run` targets — never a second build path (#2955). A successful
# build restarts Penny only once she has been IDLE for DEPLOY_IDLE_SECONDS (#3026):
# a deploy must not cut off a turn or a job run, so the old Penny keeps serving
# while the wait polls DEPLOY_IDLE_CMD (default scripts/penny-idle.sh) and builds
# any newer commit that lands meanwhile. The startup build path never waits —
# Penny isn't running yet. A new commit restarts her only when it changes the
# Penny server (#3049): the loop diffs the DEPLOYED commit (the build the running
# Penny comes from, not just HEAD) against the new HEAD over DEPLOY_RUNTIME_PATHS
# — a commit touching none of them just advances the checkout, logged as a skip.
# Startup is unchanged. Stopping the
# loop (TERM/INT, e.g. Ctrl+C on `make prod-watch`) also stops Penny: the shutdown
# runs `DEPLOY_STOP_CMD` (default `make prod-stop`) after stopping the child (#3015).
# Run by `make prod-watch`, or under launchd via the rendered
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
# The shutdown path runs this after stop_child: since #2992 the child's subshell
# swallows the group TERM, so the group kill above can't reach its `docker compose
# up` — stopping the loop left Penny's containers running (#3015).
DEPLOY_STOP_CMD="${DEPLOY_STOP_CMD:-make prod-stop}"
# How long Penny must have been idle before a successful build restarts her (#3026),
# how often the wait polls, and the check itself — mirroring the other DEPLOY_*_CMD
# variables: exit 0 means idle, 1 busy, anything else a read error (deployed anyway,
# never blocking forever; see wait_for_idle). Default on its own line, single-quoted
# there — the call site expands it unquoted (see DEPLOY_IMAGE_COMMIT_CMD's comment
# below), so the value must carry no quotes and no redirects.
DEPLOY_IDLE_SECONDS="${DEPLOY_IDLE_SECONDS:-600}"
DEPLOY_IDLE_POLL="${DEPLOY_IDLE_POLL:-30}"
DEPLOY_IDLE_CMD="${DEPLOY_IDLE_CMD:-}"
if [ -z "$DEPLOY_IDLE_CMD" ]; then
    DEPLOY_IDLE_CMD='scripts/penny-idle.sh data/penny/penny.db'
fi
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
# The git pathspecs a new commit must touch to be worth building and
# redeploying (#3049): the default is read off the shared runtime-path rule
# (scripts/runtime-paths, #3063) — the same file scripts/store_git_info.sh
# logs the changelog snapshot from, so a commit is announced in her changelog
# exactly when it changes the runtime that would be restarted. The file sits
# next to this script (resolved off $0), so the launchd plist's absolute
# invocation finds it as well as `make prod-watch` does. DEPLOY_RUNTIME_PATHS
# overrides the file (a test narrows the rule to its own fixture's paths).
# The value word-splits UNSQUOTED at the diff call site (runtime_paths_changed),
# so it must carry no quotes of its own — the same constraint DEPLOY_IDLE_CMD's
# and DEPLOY_IMAGE_COMMIT_CMD's defaults carry; the `*.md` glob stays literal
# there because the call site turns globbing off.
DEPLOY_RUNTIME_PATHS="${DEPLOY_RUNTIME_PATHS:-}"
if [ -z "$DEPLOY_RUNTIME_PATHS" ]; then
    paths_file="$(dirname "$0")/runtime-paths"
    if [ ! -f "$paths_file" ]; then
        echo "deploy-loop: cannot read the shared runtime-path rule ($paths_file)" >&2
        exit 1
    fi
    # One pathspec per line, '#' comments and blank lines ignored, joined into
    # the one word-split string the diff call site consumes.
    DEPLOY_RUNTIME_PATHS="$(awk '{sub(/#.*$/, ""); gsub(/[ \t]/, ""); if (NF) print}' "$paths_file" | paste -s -d' ' -)"
    if [ -z "$DEPLOY_RUNTIME_PATHS" ]; then
        echo "deploy-loop: $paths_file lists no paths" >&2
        exit 1
    fi
fi

# fetch_and_fast_forward's return codes: it ADVANCED HEAD to origin (the only
# outcome a deploy may act on), HEAD was ALREADY at origin (a no-op cycle), or
# it skipped/failed (already logged; leave everything alone).
readonly FF_ADVANCED=0 FF_UP_TO_DATE=1 FF_SKIPPED=2

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

# Fetches origin/$DEPLOY_BRANCH and fast-forwards HEAD to it — the loop's ONE git
# path, shared by the per-cycle check and the startup check (#3020), so a start
# never builds the stale local HEAD before it has even looked at origin. It
# advances HEAD only when every one of these holds; anything else is logged with
# its reason and HEAD stays where it is:
#   - the fetch itself succeeds
#   - checked out on $DEPLOY_BRANCH (never acts on a detached HEAD or another branch)
#   - no tracked changes (untracked files are fine — git status --untracked-files=no)
#   - HEAD is an ancestor of origin/$DEPLOY_BRANCH (a true fast-forward is possible)
#
# Returns FF_ADVANCED when it fast-forwarded HEAD to origin/$DEPLOY_BRANCH,
# FF_UP_TO_DATE when HEAD was already there (a no-op cycle — the caller must
# build nothing), FF_SKIPPED on any skip or failure — never a reset or checkout.
fetch_and_fast_forward() {
    if ! git fetch origin "$DEPLOY_BRANCH" --quiet 2>>"$DEPLOY_LOG"; then
        log "git fetch origin $DEPLOY_BRANCH failed"
        return "$FF_SKIPPED"
    fi

    local_head="$(git rev-parse HEAD)"
    remote_head="$(git rev-parse "origin/$DEPLOY_BRANCH")"
    if [ "$local_head" = "$remote_head" ]; then
        log "up to date at $local_head"
        return "$FF_UP_TO_DATE"
    fi

    branch="$(git rev-parse --abbrev-ref HEAD)"
    if [ "$branch" != "$DEPLOY_BRANCH" ]; then
        log "skip $remote_head: on $branch, not $DEPLOY_BRANCH"
        return "$FF_SKIPPED"
    fi
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        log "skip $remote_head: tracked changes present"
        return "$FF_SKIPPED"
    fi
    if ! git merge-base --is-ancestor "$local_head" "origin/$DEPLOY_BRANCH"; then
        log "skip $remote_head: $local_head is not an ancestor — not a fast-forward"
        return "$FF_SKIPPED"
    fi

    log "fast-forwarding $local_head -> $remote_head"
    if ! git merge --ff-only "origin/$DEPLOY_BRANCH" >>"$DEPLOY_LOG" 2>&1; then
        log "ff-only merge to $remote_head failed unexpectedly — leaving old version running"
        return "$FF_SKIPPED"
    fi
    return "$FF_ADVANCED"
}

# Deploys origin/$DEPLOY_BRANCH when it is ahead of HEAD — but only after the
# shared fetch-and-fast-forward above has moved HEAD there.
#
# The `git merge --ff-only` runs, and HEAD advances, BEFORE the build — while the
# OLD Penny keeps serving (prod-build never touches the running container; only
# prod-run, called below, does). That ordering is also the failed-build memory:
# whether or not the build that follows succeeds, HEAD now IS the commit that was
# attempted, so the next cycle's "is origin/main ahead of HEAD?" check is false
# until a genuinely NEW commit lands — the same commit is never retried, with no
# separate state file needed. Only a cycle that ADVANCED HEAD reaches the build —
# an up-to-date cycle is a no-op (an ordinary idle cycle, every DEPLOY_INTERVAL).
# Anything the fetch-and-ff skipped or found already present is already logged
# with its reason; the current version keeps running, retried next cycle.
#
# Before building, the runtime-path check (#3049): the diff from $deployed — the
# commit of the build the running Penny actually comes from, not the previous
# HEAD — to the new HEAD decides whether she restarts at all. A commit touching
# no runtime path just advances the checkout (logged as a skip); the NEXT commit
# that does touch one is diffed against the same deployed commit, so nothing a
# run of skips sits on top of is ever missed. An indeterminate diff (git itself
# failed) builds anyway — a broken check must degrade to the old always-build
# behaviour, never to silently never building.
fetch_and_maybe_deploy() {
    fetch_and_fast_forward
    if [ "$?" -ne "$FF_ADVANCED" ]; then
        return
    fi

    runtime_paths_changed "$deployed" "$remote_head"
    changed=$?
    if [ "$changed" -eq 1 ]; then
        log "skip $remote_head: no Penny server changes since $deployed"
        return
    fi
    if [ "$changed" -eq 2 ]; then
        log "runtime-path diff FAILED ($deployed..$remote_head) — building anyway"
    fi

    log "building $remote_head: $DEPLOY_BUILD_CMD"
    if ! $DEPLOY_BUILD_CMD >>"$DEPLOY_LOG" 2>&1; then
        log "build FAILED at $remote_head — old Penny keeps running; this commit will not be retried, only the next new one"
        return
    fi

    log "build succeeded at $remote_head — waiting for Penny to be idle"
    if ! wait_for_idle; then
        return
    fi

    log "redeploying: $DEPLOY_RUN_CMD"
    stop_child
    start_child
    log "redeployed $local_head -> $remote_head"
    # The running Penny is now the image just built, so record ITS commit as the
    # deployed one (#3049) — read off the same git_commit label the startup check
    # reads, because the newest build that succeeded inside wait_for_idle can be
    # newer than this cycle's own remote_head. Falling back to the cycle's head
    # keeps the tracking alive if the label can't be read.
    deployed="$($DEPLOY_IMAGE_COMMIT_CMD 2>/dev/null)"
    deployed="${deployed:-$remote_head}"
}

# Whether the diff from $1 (the deployed commit) to $2 touches any runtime path
# (#3049): exit 0 = it does, 1 = it doesn't (a docs/tests-only commit — skip the
# build and restart), 2 = the diff itself failed (indeterminate — the caller
# builds anyway rather than silently skipping on a broken check). `git diff
# --quiet` answers the opposite way — 0 on no differences, 1 on differences —
# so the statuses are inverted here, once, at the boundary. The pathspecs
# word-split unquoted off $DEPLOY_RUNTIME_PATHS; `set -f` keeps the `*.md` in
# :(exclude)*.md from globbing against the checkout — nothing matches a literal
# :(exclude)… prefix, but the decision must not depend on that accident.
runtime_paths_changed() {
    set -f
    git diff --quiet "$1" "$2" -- $DEPLOY_RUNTIME_PATHS 2>>"$DEPLOY_LOG"
    local rc=$?
    set +f
    if [ "$rc" -eq 1 ]; then
        return 0
    fi
    if [ "$rc" -eq 0 ]; then
        return 1
    fi
    return 2
}

# Waits for Penny to be idle before a successful build may restart her (#3026):
# a deploy must not cut off a turn or a job run, so the old Penny keeps serving
# until the newest promptlog/messagelog activity in the prod database is older
# than DEPLOY_IDLE_SECONDS — read by polling $DEPLOY_IDLE_CMD every
# $DEPLOY_IDLE_POLL seconds, the wait itself logged once, naming the check's
# "last activity" output. While waiting, origin keeps moving: a newer commit is
# fast-forwarded and built right away (the same fetch-and-ff path as the
# per-cycle check), so the build that deploys once idle is the newest one.
#
# Returns 0 = deploy now: the check says idle, or it FAILED (any exit but 0/1 —
# logged and deployed anyway; a deploy blocked forever on a bad check is worse
# than an eager one), or the child has died while waiting (nothing left to
# protect — the new build is what replaces it, and the wait would otherwise
# leave Penny down for a whole idle threshold).
# Returns 1 = a stop (TERM/INT) arrived while waiting: deploy nothing — the
# main loop's shutdown path takes over, and nothing else.
wait_for_idle() {
    local wait_logged="" out rc
    while :; do
        if [ -n "$stopping" ]; then
            log "stop arrived while waiting for Penny to be idle — not redeploying"
            return 1
        fi
        out="$($DEPLOY_IDLE_CMD 2>>"$DEPLOY_LOG")"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            return 0
        fi
        if [ "$rc" -ne 1 ]; then
            log "idle check FAILED (exit $rc) — deploying anyway: ${out:-no output}"
            return 0
        fi
        if [ -z "$wait_logged" ]; then
            log "waiting for Penny to be idle (last activity ${out:-unknown})"
            wait_logged=1
        fi
        if [ -n "$child_pid" ] && ! kill -0 "$child_pid" 2>/dev/null; then
            log "child exited while waiting for idle — nothing left to protect; redeploying now"
            return 0
        fi
        fetch_and_fast_forward
        if [ "$?" -eq "$FF_ADVANCED" ]; then
            # The same runtime-path check as the per-cycle deploy (#3049): a
            # commit landing mid-wait that touches no runtime path adds nothing
            # to the image being waited for — skip its build, keep waiting.
            runtime_paths_changed "$deployed" "$remote_head"
            if [ "$?" -eq 1 ]; then
                log "skip $remote_head: no Penny server changes since $deployed"
            else
                log "newer commit $remote_head landed while waiting — building it: $DEPLOY_BUILD_CMD"
                if ! $DEPLOY_BUILD_CMD >>"$DEPLOY_LOG" 2>&1; then
                    log "build FAILED at $remote_head while waiting — old Penny keeps running; keep waiting for idle"
                fi
            fi
        fi
        interruptible_sleep "$DEPLOY_IDLE_POLL"
    done
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

# One fetch + fast-forward BEFORE the startup build check (#3020): the loop used
# to build and start the local HEAD before it ever fetched, so a start after
# merges brought up the old build for a whole interval. The same shared
# fetch-and-ff path the per-cycle check runs; a failed fetch or a non-fast-
# forward is logged there and HEAD stays put — the build below then runs on
# whatever HEAD is, exactly as before.
fetch_and_fast_forward || true
startup_build_if_needed
# From here on the loop tracks what's DEPLOYED (#3049): the commit of the build
# the child below actually starts on — the image's own git_commit label, read
# after the startup build so it names whatever that build produced (or the
# previous image's commit if it failed, which is what would really be running).
# If the label can't be read at all, HEAD is the honest fallback. An empty label
# with no image means the diff decides against nothing sensible either way —
# the startup build path above has already handled the "never built" case.
deployed="$($DEPLOY_IMAGE_COMMIT_CMD 2>/dev/null)"
deployed="${deployed:-$(git rev-parse --short HEAD)}"
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
# #3015: stop_child's group kill can't reach the child's `docker compose up` (see
# DEPLOY_STOP_CMD above), so stopping prod is explicit here — the same `prod-stop`
# target prod-run opens with. The redeploy path (stop_child/start_child in
# fetch_and_maybe_deploy) deliberately does NOT run it: prod-run brings the old
# containers down itself.
log "stopping penny: $DEPLOY_STOP_CMD"
if ! $DEPLOY_STOP_CMD >>"$DEPLOY_LOG" 2>&1; then
    log "stopping penny FAILED — Penny's containers may still be running"
fi
log "stopped"
