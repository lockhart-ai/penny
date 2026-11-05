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

At startup (`Penny._record_changelog_entries`), the entries this call's
list carries that the `changelog` log itself says are new (``record_new_entries`` —
the log is the cursor, #2973) are appended to the `changelog`
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
    "select_new_entries",
    "record_new_entries",
    "render_changelog_announcement",
]

# The fresh-log cap (#2973): a snapshot whose PRs are ALL unrecorded — a fresh
# database — records only this many of its newest entries, the code owner's
# earlier ruling for a fresh install.
_FRESH_LOG_CAP = 10

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


def select_new_entries(
    entries: list[ChangelogEntry], recorded: set[int], fresh_cap: int = _FRESH_LOG_CAP
) -> list[ChangelogEntry]:
    """Which of a deploy snapshot's entries are hers to record, decided by the
    log itself (#2973): the snapshot (``scripts/store_git_info.sh``) is a fixed
    bounded window of recent commits — it carries no cursor — so this finds the
    NEWEST entry whose PR is already recorded in the ``changelog`` log and
    returns every entry after it, oldest first. Everything the log says she has
    seen stops at, everything it doesn't say she has seen is recorded.

    If NONE of the snapshot's PRs is in the log (a fresh database, or one so
    long unbooted that its last recorded PR has fallen out of the snapshot
    window), the newest ``fresh_cap`` entries are returned — the code owner's
    earlier ruling for a fresh install. That cap is also the one honest bound
    of this scheme: a gap of more than one snapshot window (~2–3 weeks of
    commits at this repo's rate) between boots loses the middle commits,
    the same trade the fresh-install cap always accepted.

    Pure — no DB, no file, no clock — so the rule is unit-testable on its own;
    ``record_new_entries`` is its only runtime caller."""
    for index in range(len(entries) - 1, -1, -1):
        if entries[index].pr in recorded:
            return entries[index + 1 :]
    return entries[-fresh_cap:]


def record_new_entries(log: Memory, entries: list[ChangelogEntry]) -> list[ChangelogEntry]:
    """Select (``select_new_entries``) and append the entries of this deploy's
    snapshot that the log itself says are new, each stamped with the log's own
    clock (the deploy time). Returns the entries that were actually appended —
    which is exactly what this deploy announces too (#2891): there is no
    separate announce-selection step, so nothing is ever recorded that isn't
    announced.

    Idempotent: the selection cuts at the newest PR already recorded, so a
    bare restart on an unchanged snapshot appends nothing — the only guard
    against a container restart (no rebuild, so the same snapshot persists)
    re-announcing what a previous boot already told the user."""
    already = recorded_prs(log)
    missing = select_new_entries(entries, already)
    if not missing:
        return []
    log.append(
        [LogEntryInput(content=entry.render()) for entry in missing],
        author=PennyConstants.CHANGELOG_LOG_AUTHOR,
    )
    return missing


# ── The startup delivery turn's own render (#2846, trigger reworked #2974) ───
#
# The framework renders this deploy's new entries verbatim — never model-written
# — as the live turn's own USER message prefixed with ``PennyConstants.
# SYSTEM_NOTIFICATION_PREFIX`` (``ChatAgent.handle_changelog_report``): her
# persona, standing rules, history and full tool surface, over a fixed list
# of facts, with no instruction attached — she decides for herself whether, how,
# and how many times to reply, the same as on any chat turn.  The PR-number
# prefix each entry already carries (``render()``) is kept here too: it is the
# same anchor a later ``log_read('changelog')`` shows, so a reply that cites a
# change and a reader checking the log against it are looking at the same text.
# This is the ONE render used both live and, byte-identical, in every later
# turn's history (``Agent._run_trigger_message`` reads it back off the run's
# own promptlog row — never stored a second time).

_CHANGELOG_RESTART_LEAD = "You've just restarted."
_CHANGELOG_LEAD = "These changes to your tools and behaviour reached you with this deploy:"
_CHANGELOG_NONE_LEAD = "This deploy changed nothing in your tools or behaviour."


def render_changelog_announcement(entries: list[ChangelogEntry]) -> str:
    """The whole delivery turn's input, rendered from framework-assembled facts
    alone — pure and deterministic, so the exact text the chat agent reads is
    pinned by a whole-render test rather than by whatever a live run produced.

    The notice announces only the changes that deployed (#3003): the startup
    preflight's soft warnings ride nothing — she can't act on them, and the
    preflight's own startup log already records them.

    Both forms open on the same restart fact (#2974 — so a later turn reading
    this back off history always sees a restart happened, whether or not
    anything changed with it).  ``entries`` may be empty — a restart, or a
    deploy that brought only harness/CI/doc changes, still runs this turn
    (Penny announces on every startup); the render then just states that
    nothing changed, with nothing else to list."""
    if not entries:
        return " ".join([_CHANGELOG_RESTART_LEAD, _CHANGELOG_NONE_LEAD])
    lines = [f"{_CHANGELOG_RESTART_LEAD} {_CHANGELOG_LEAD}", ""]
    lines.extend(f"- {entry.render()}" for entry in entries)
    return "\n".join(lines)
