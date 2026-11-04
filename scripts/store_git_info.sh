#!/usr/bin/env bash
set -euo pipefail

# Writes Penny's changelog source into the build context, on the HOST, before
# the image builds (#2891) — the mozilla/experimenter `scripts/store_git_info.sh`
# pattern: a few plain `git log` lines, run before `docker build`, since the
# image itself has no git.
#
# Writes the newest $CHANGELOG_SNAPSHOT_COMMITS raw commits (oldest first) into
# the build-context file ($CHANGELOG_COMMITS_FILE, which the image COPYs in).
# There is no cursor here any more (#2973): the snapshot is a fixed bounded
# window, and WHICH of those commits are hers to record is decided at startup
# by Penny herself — the `changelog` log is the cursor. The newest PR already
# recorded there cuts the window; a fresh log (none of the snapshot's PRs
# recorded yet) is capped at 10 entries. An empty window is fine too — a deploy
# with no commit history captured simply records and announces nothing.
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

echo "store_git_info: newest $snapshot_commits raw commits" >&2
git log --no-merges -n "$snapshot_commits" --reverse --pretty="$pretty_format" > "$commits_file"