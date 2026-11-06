#!/usr/bin/env bash
set -euo pipefail

# Writes Penny's changelog source into the build context, on the HOST, before
# the image builds (#2891) — the mozilla/experimenter `scripts/store_git_info.sh`
# pattern: a few plain `git log` lines, run before `docker build`, since the
# image itself has no git.
#
# Writes the newest $CHANGELOG_SNAPSHOT_COMMITS commits that touch the shared
# runtime-path rule (scripts/runtime-paths, #3063 — one git pathspec per line,
# '#' comments and blank lines ignored; the same file the deploy loop
# scripts/deploy-loop.sh reads as its default DEPLOY_RUNTIME_PATHS), oldest
# first, into the build-context file ($CHANGELOG_COMMITS_FILE, which the image
# COPYs in). That one rule is what makes the changelog and the deploy watcher
# agree: a commit is announced exactly when it changes the runtime that would
# be restarted. Which of the snapshot's commits are hers to RECORD is decided
# at startup by Penny herself — the `changelog` log is the cursor (#2973).
# The newest PR already recorded there cuts the window; a fresh log (none of
# the snapshot's PRs recorded yet) is capped at 10 entries. An empty window is
# fine too — a deploy with no commit history captured simply records and
# announces nothing.
#
# Each record carries the commit's BODY too (`%b`), not just its title: the
# repo's squash setting (`squash_merge_commit_message: PR_BODY`) makes a
# squash commit's body the merged PR's own description, whose first sentence
# is — by the PR-body convention — a plain, cold-readable summary of the
# whole change. A title alone can mislead (a `refactor(penny):` commit read
# by its own real meaning, "same behaviour", even though #2820's refactor
# DID change what Penny's tool calls carry). `%x1e` (record separator) joins
# records instead of a newline, since a body is itself multi-line.

commits_file="${CHANGELOG_COMMITS_FILE:-penny/penny/changelog_commits.txt}"
snapshot_commits="${CHANGELOG_SNAPSHOT_COMMITS:-200}"
pretty_format='format:%H%x1f%s%x1f%b%x1e'

paths_file="$(dirname "$0")/runtime-paths"
if [ ! -f "$paths_file" ]; then
    echo "store_git_info: cannot read the shared runtime-path rule ($paths_file)" >&2
    exit 1
fi
# One pathspec per line, '#' comments and blank lines ignored, joined into the
# one word-split string the git log call consumes.
runtime_paths="$(awk '{sub(/#.*$/, ""); gsub(/[ \t]/, ""); if (NF) print}' "$paths_file" | paste -s -d' ' -)"
if [ -z "$runtime_paths" ]; then
    echo "store_git_info: $paths_file lists no paths" >&2
    exit 1
fi

echo "store_git_info: newest $snapshot_commits commits touching the runtime paths" >&2
# The `*.md` in :(exclude)*.md must not glob against the checkout — nothing
# matches a literal :(exclude)… prefix, but the result must not depend on that
# accident (the same set -f the deploy loop's diff call site uses).
set -f
git log --no-merges -n "$snapshot_commits" --reverse --pretty="$pretty_format" -- $runtime_paths > "$commits_file"
set +f
