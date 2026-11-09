#!/usr/bin/env bash
# Waits for Penny to go idle before a simulator build or test run starts
# (#3171): Xcode and the simulator compete with her local model server for
# the GPU. Measured while a `make client-ui-tests` run was going: Penny's
# generation speed fell from about 44 to 7-15 tokens per second, a chat
# reply took 37s instead of about 12s, and load average hit about 350 with
# 15 MTLCompilerService processes.
#
# Called at the top of client-check.sh, client-ui-tests.sh and
# client-stories.sh — BEFORE the simulator lock or any xcodebuild, so a
# waiting run holds nothing.
#
# Polls scripts/penny-idle.sh against the PRIMARY checkout's
# data/penny/penny.db, found from any worktree via `git rev-parse
# --git-common-dir` (a worktree has no data/ of its own, so
# penny-idle.sh's own worktree-relative default would always read as "no
# database"). On CI, and on any machine with no Penny database, that read
# fails and penny-idle.sh exits 2 — this script proceeds at once and says
# so, so CI is unaffected.
#
# Usage: wait-for-penny-idle.sh
# Env:
#   CLIENT_IDLE_SECONDS      how long Penny must have been idle, in seconds
#                            (default 300)
#   CLIENT_IDLE_WAIT         0 skips the wait entirely, for an explicitly
#                            urgent run (default 1 — wait)
#   CLIENT_IDLE_MAX_SECONDS  gives up and exits nonzero after waiting this
#                            many seconds total, instead of polling forever
#                            (default 3600)
#   CLIENT_IDLE_DB           overrides the database path checked — for
#                            testing this script against a temp database;
#                            leave unset in normal use, where the path is
#                            always derived from the primary checkout
set -u

POLL_SECONDS=30

validate_seconds() { # <name> <value>
    case "$2" in
        ''|*[!0-9]*)
            echo "wait-for-penny-idle: $1 must be a whole number of seconds, got '$2'" >&2
            exit 2
            ;;
    esac
}

CLIENT_IDLE_SECONDS="${CLIENT_IDLE_SECONDS:-300}"
CLIENT_IDLE_WAIT="${CLIENT_IDLE_WAIT:-1}"
CLIENT_IDLE_MAX_SECONDS="${CLIENT_IDLE_MAX_SECONDS:-3600}"

validate_seconds CLIENT_IDLE_SECONDS "$CLIENT_IDLE_SECONDS"
validate_seconds CLIENT_IDLE_MAX_SECONDS "$CLIENT_IDLE_MAX_SECONDS"

if [ "$CLIENT_IDLE_WAIT" = "0" ]; then
    echo "wait-for-penny-idle: CLIENT_IDLE_WAIT=0 — skipping the idle wait"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${CLIENT_IDLE_DB:-}" ]; then
    DB_PATH="$CLIENT_IDLE_DB"
else
    PRIMARY_CHECKOUT="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/.."
    DB_PATH="$PRIMARY_CHECKOUT/data/penny/penny.db"
fi

waited=0
while :; do
    newest="$(DEPLOY_IDLE_SECONDS="$CLIENT_IDLE_SECONDS" "$SCRIPT_DIR/penny-idle.sh" "$DB_PATH")"
    status=$?

    if [ "$status" -eq 2 ]; then
        echo "wait-for-penny-idle: no database at $DB_PATH — proceeding at once"
        exit 0
    fi

    if [ "$status" -eq 0 ]; then
        echo "wait-for-penny-idle: Penny idle since $newest — proceeding"
        exit 0
    fi

    if [ "$waited" -ge "$CLIENT_IDLE_MAX_SECONDS" ]; then
        echo "wait-for-penny-idle: gave up after ${waited}s waiting for Penny to go idle (last activity $newest) — raise CLIENT_IDLE_MAX_SECONDS, or set CLIENT_IDLE_WAIT=0 to skip this wait" >&2
        exit 1
    fi

    echo "wait-for-penny-idle: Penny is busy (last activity $newest) — waiting (${waited}s of ${CLIENT_IDLE_MAX_SECONDS}s)"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done
