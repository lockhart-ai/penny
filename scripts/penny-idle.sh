#!/usr/bin/env bash
# Penny's idle check for the deploy loop (#3026): exits 0 when Penny has been
# idle for DEPLOY_IDLE_SECONDS — the newest promptlog AND messagelog timestamps
# in the database are both at least that old — 1 while she is busy (a turn or
# a job step keeps writing to either log), 2 when the database can't be read
# at all (missing file, missing table, bad threshold — the caller decides what
# that means; the deploy loop logs it and deploys anyway rather than block).
#
# Usage: penny-idle.sh [db-path]      (default: data/penny/penny.db)
# Env:   DEPLOY_IDLE_SECONDS          (default: 600)
# On stdout it prints the newest of the two timestamps — the "last activity"
# the deploy loop's waiting line names — whenever the read succeeded.
#
# Timestamps are stored UTC ("2026-11-05 04:05:21.724038"); strftime('%s', …)
# parses that shape and strftime('%s','now') is UTC, so the whole comparison
# runs inside SQLite — nothing date-shaped for the shell to parse differently
# per platform or locale — and `-readonly` opens the production database with
# no chance of taking a write lock or creating a file. An empty table's newest
# timestamp is NULL: no activity to protect, so an empty table counts as idle.
set -u

IDLE_SECONDS="${DEPLOY_IDLE_SECONDS:-600}"
DB_PATH="${1:-data/penny/penny.db}"

case "$IDLE_SECONDS" in
    ''|*[!0-9]*)
        echo "penny-idle: DEPLOY_IDLE_SECONDS must be a whole number of seconds, got '$IDLE_SECONDS'" >&2
        exit 2
        ;;
esac

# One read, two answers: the two tables' ages against now (as "prompt|message",
# -1 marking an empty table), and the newest of the two timestamps for the log.
out="$(sqlite3 -bail -readonly "$DB_PATH" "
SELECT COALESCE(strftime('%s','now') - strftime('%s', (SELECT MAX(timestamp) FROM promptlog)), -1)
       || '|' ||
       COALESCE(strftime('%s','now') - strftime('%s', (SELECT MAX(timestamp) FROM messagelog)), -1);
SELECT COALESCE(MAX(t), '') FROM (
    SELECT MAX(timestamp) AS t FROM promptlog
    UNION ALL
    SELECT MAX(timestamp) FROM messagelog
);" 2>/dev/null)"
if [ $? -ne 0 ]; then
    echo "penny-idle: cannot read $DB_PATH" >&2
    exit 2
fi

ages="${out%%$'\n'*}"
newest="${out#*$'\n'}"
prompt_age="${ages%%|*}"
message_age="${ages##*|}"

busy=0
for age in "$prompt_age" "$message_age"; do
    # -1 marks an empty table (NULL newest) — no activity to protect.
    [ "$age" -lt 0 ] && continue
    [ "$age" -lt "$IDLE_SECONDS" ] && busy=1
done

printf '%s\n' "$newest"
[ "$busy" -eq 0 ] && exit 0
exit 1