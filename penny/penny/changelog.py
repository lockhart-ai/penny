"""Penny's changelog (#2845, reworked by #2891) — the DB-facing half:
recording a deploy's commit list into the `changelog` system log, and
rendering the startup delivery turn's input from it.

The pure parsing (`ChangelogEntry`, `parse_commit_log`, `load_commits`,
`COMMITS_PATH`) lives in `penny.changelog_commits` — no DB or framework
imports there, so `penny.config` can import it at module level with no
cycle back through `penny.database`/`penny.validation`/`penny.agents`
(which this module's own `penny.database.memory` import pulls in). Those
names are re-exported here too, so every existing `from penny.changelog
import ChangelogEntry, ...` call site keeps working unchanged.

At startup (`Penny._record_changelog_entries`), every commit this call's
list carries that isn't already recorded is appended to the `changelog`
system log (``PennyConstants.MEMORY_CHANGELOG_LOG`` — a plain append-only
``Log``, read like any other log via ``log_read``), stamped with that
deploy's own time rather than the commit's merge time: the date that
matters is when the change reached her, not when it merged. The commit list
itself is a PARAMETER (``Config.changelog_commits``) — never read from this
module's own file inside ``Penny``, so a test or an eval case supplies its
own list and never touches the real build artifact ("Pass parameters, don't
swap state", root `CLAUDE.md`).
"""

from __future__ import annotations

import re

from penny.changelog_commits import (
    COMMITS_PATH,
    ChangelogEntry,
    load_commits,
    parse_commit_log,
)
from penny.constants import PennyConstants
from penny.database.memory import LogEntryInput, Memory

__all__ = [
    "COMMITS_PATH",
    "ChangelogEntry",
    "load_commits",
    "parse_commit_log",
    "recorded_prs",
    "record_new_entries",
    "render_changelog_announcement",
]

# The leading token every recorded row's content carries (``ChangelogEntry.render``).
# A plain ``Log`` has no second column to key idempotency on, so "already
# recorded?" reads this prefix back off the stored content instead — the render
# and the parse share one format, so they can't drift apart.
_PR_PREFIX_RE = re.compile(r"^PR #(\d+): ")


def recorded_prs(log: Memory) -> set[int]:
    """The PR numbers already recorded in the ``changelog`` log, read off each
    entry's own ``PR #<n>: `` prefix — no second tracking column needed."""
    recorded: set[int] = set()
    for entry in log.read_all():
        match = _PR_PREFIX_RE.match(entry.content)
        if match:
            recorded.add(int(match.group(1)))
    return recorded


def record_new_entries(log: Memory, entries: list[ChangelogEntry]) -> list[ChangelogEntry]:
    """Append every entry in ``entries`` not already recorded in ``log``,
    each stamped with the log's own clock (the deploy time). Returns the
    entries that were actually appended — which is exactly what this
    deploy announces too (#2891): there is no separate selection step, so
    nothing is ever recorded that isn't announced.

    Idempotent: an entry recorded on one call is never appended again on a
    later one — the only guard against a container restart (no rebuild, so
    the same commit list persists) re-announcing what a previous boot
    already told the user. There is no OTHER position or dedupe logic here
    — the host's own cursor (``scripts/store_git_info.sh``) is what keeps
    ``entries`` itself a small, fresh window; this function just never
    double-writes a row within that window."""
    already = recorded_prs(log)
    missing = [entry for entry in entries if entry.pr not in already]
    if not missing:
        return []
    log.append(
        [LogEntryInput(content=entry.render()) for entry in missing],
        author=PennyConstants.CHANGELOG_LOG_AUTHOR,
    )
    return missing


# ── The startup delivery turn's own render (#2846) ───────────────────────────
#
# The framework renders this deploy's new entries verbatim — never model-written
# — as the input of ONE chat-agent turn (``ChatAgent.handle_changelog_report``):
# her persona, standing rules, history and full tool surface, over a fixed list
# of facts.  The PR-number prefix each entry already carries (``render()``) is
# kept here too: it is the same anchor a later ``log_read('changelog')`` shows,
# so a reply that cites a change and a reader checking the log against it are
# looking at the same text.

_CHANGELOG_LEAD = "These changes to your tools and behaviour reached you with this deploy:"
_CHANGELOG_ASK = "Let the user know what changed, in your own words."
_CHANGELOG_NONE_LEAD = (
    "You've just restarted. This deploy changed nothing in your tools or behaviour."
)
_CHANGELOG_NONE_ASK = "Send the user a short message letting them know you're back."


def render_changelog_announcement(entries: list[ChangelogEntry]) -> str:
    """The whole delivery turn's input, rendered from framework-assembled facts
    alone — pure and deterministic, so the exact text the chat agent reads is
    pinned by a whole-render test rather than by whatever a live run produced.

    ``entries`` may be empty — a restart, or a deploy that brought only
    harness/CI/doc changes, still runs this turn (Penny announces on every
    startup); the render just states that nothing changed and asks for a
    short "I'm back" message instead of listing entries."""
    if not entries:
        return "\n".join([_CHANGELOG_NONE_LEAD, "", _CHANGELOG_NONE_ASK])
    lines = [_CHANGELOG_LEAD, ""]
    lines.extend(f"- {entry.render()}" for entry in entries)
    lines.extend(["", _CHANGELOG_ASK])
    return "\n".join(lines)
