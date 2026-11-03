#!/usr/bin/env bash
set -euo pipefail

# Writes Penny's changelog source into the build context, on the HOST, before
# the image builds (#2891) — the mozilla/experimenter `scripts/store_git_info.sh`
# pattern: a few plain `git log` lines, run before `docker build`, since the
# image itself has no git.
#
# Usage: scripts/store_git_info.sh <advance|dev>
#   advance — a real deploy (prod/prod-ios): the Makefile writes the new
#             cursor itself, after a successful run — this script only READS
#             the cursor, never writes it.
#   dev     — a dev run (`make up`): the same commit range, cursor untouched.
#
# Reads the cursor dotfile ($CHANGELOG_CURSOR_FILE — the hash of the last
# commit a real deploy announced, kept under data/ so it survives every image
# rebuild) and writes the penny-scope commit range since it into the
# build-context file ($CHANGELOG_COMMITS_FILE, which the image COPYs in). An
# unset cursor, or one history has rewritten past (not an ancestor of HEAD),
# falls back to the newest $CHANGELOG_FALLBACK_COMMITS raw commits (default
# 10, the code owner's own ruling) — scoping to the penny feat/fix/perf/
# refactor types happens in Python at runtime, so this count is of raw
# commits, not penny-scope ones.
#
# Each record carries the commit's BODY too (`%b`), not just its title: the
# repo's squash setting (`squash_merge_commit_message: PR_BODY`) makes a
# squash commit's body the merged PR's own description, whose first sentence
# is — by the PR-body convention — a plain, cold-readable summary of the
# whole change. A title alone can mislead (a `refactor(penny):` commit read
# by its own real meaning, "same behaviour", even though #2820's refactor
# DID change what Penny's tool calls carry). `%x1e` (record separator) joins
# records instead of a newline, since a body is itself multi-line.

mode="${1:?usage: $0 <advance|dev>}"
cursor_file="${CHANGELOG_CURSOR_FILE:-data/penny/.changelog_cursor}"
commits_file="${CHANGELOG_COMMITS_FILE:-penny/penny/changelog_commits.txt}"
fallback_commits="${CHANGELOG_FALLBACK_COMMITS:-10}"
pretty_format='format:%H%x1f%s%x1f%b%x1e'

mkdir -p "$(dirname "$cursor_file")"

cursor=""
if [ -f "$cursor_file" ]; then
  cursor="$(cat "$cursor_file")"
fi

if [ -n "$cursor" ] && git merge-base --is-ancestor "$cursor" HEAD 2>/dev/null; then
  echo "store_git_info ($mode): penny-scope commits since $cursor" >&2
  git log --no-merges --reverse --pretty="$pretty_format" "$cursor"..HEAD > "$commits_file"
else
  echo "store_git_info ($mode): no usable cursor — newest $fallback_commits raw commits" >&2
  git log --no-merges -n "$fallback_commits" --reverse --pretty="$pretty_format" > "$commits_file"
fi
