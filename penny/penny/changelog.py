"""Penny's changelog (#2845, reworked by #2891) — entries that tell her about
changes to her tools, prompts and behaviour, so her self-reflection can tell
"we changed this" apart from "my own rule did this".

The source is no longer hand-written: it is the title of every merged PR that
touched her (`feat`/`fix`/`perf`/`refactor` in the `penny` scope — #2891),
read straight off `git log`. Every merged PR already carries a conventional
title (`type(scope): title`, squashed onto one commit with `(#<pr>)` on the
end — root `CLAUDE.md` → Git Workflow), so no PR needs its own hand-authored
entry any more.

The commit list is captured at image BUILD time (`make changelog-commits`,
wired into `up`/`prod`/`prod-ios`/`build`) as a plain ``git log`` dump —
`<hash>\x1f<subject>` lines, oldest first — written to the gitignored
`changelog_commits.txt`, co-located with this module so it rides the same
`COPY penny/ ./penny/` Dockerfile step as everything else here. The host's
own Python can't be trusted to import this package (an agent container's
host Python may be 3.9 with no pydantic installed), so the Makefile target
shells out to plain ``git``/text tools only; this module does the parsing,
entirely at RUNTIME, over whatever that file holds.

At startup (`Penny._record_changelog_entries`), every commit not yet recorded
is appended to the `changelog` system log (``PennyConstants.MEMORY_CHANGELOG_LOG``
— a plain append-only ``Log``, read like any other log via ``log_read``),
stamped with that deploy's own time rather than the commit's merge time: the
date that matters is when the change reached her, not when it merged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from penny.constants import PennyConstants
from penny.database.memory import LogEntryInput, Memory

# Co-located with the source list it loads (#2845) — resolved relative to this
# module's own file, so it is correct whether this runs from the repo tree or
# the Docker image's copy, never against the process's working directory.
COMMITS_PATH = Path(__file__).with_name("changelog_commits.txt")

# The leading token every recorded row's content carries (``ChangelogEntry.render``).
# A plain ``Log`` has no second column to key idempotency on, so "already
# recorded?" reads this prefix back off the stored content instead — the render
# and the parse share one format, so they can't drift apart.
_PR_PREFIX_RE = re.compile(r"^PR #(\d+): ")

# One line of the generated commit dump: ``<hash>\x1f<subject>``, where
# ``<subject>`` is a squash-merged PR's own conventional title with its
# trailing ``(#<pr>)`` GitHub appends. Kept types: `feat`/`fix`/`perf` are
# user-visible by definition; `refactor` is kept too (#2891) — in this repo a
# "same behaviour, different shape" change can still change what SHE sees
# (#2820 dropped a tool-call argument without changing what it did for the
# user). Every other type (`test`/`chore`/`docs`) and every other scope
# (`eval`/`harness`/`ci`/`client`/`browser`) is dropped: not hers to announce.
_COMMIT_LINE_RE = re.compile(
    r"^[0-9a-f]+\x1f(?P<kind>feat|fix|perf|refactor)\(penny\): (?P<rest>.+) \(#(?P<pr>\d+)\)$"
)


class ChangelogEntry(BaseModel):
    """One changelog item — a PR number and the text recorded for it (now a
    commit's own conventional title, never hand-written)."""

    model_config = ConfigDict(frozen=True)

    pr: int
    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("a changelog entry's text must not be blank")
        return stripped

    def render(self) -> str:
        """The exact row content a recorded entry carries — ``recorded_prs``
        reads the leading ``PR #<n>: `` back off this same format."""
        return f"PR #{self.pr}: {self.text}"


def parse_commit_log(lines: Iterable[str]) -> list[ChangelogEntry]:
    """Parse ``<hash>\x1f<subject>`` lines (``git log``'s own
    ``--pretty=format:%H%x1f%s``, oldest-first via ``--reverse``) into
    changelog entries. Pure and unit-tested on its own — the only git/file
    plumbing around it (the Makefile recipe, ``load_commits`` below) stays
    thin enough to need no test of its own.

    A line that doesn't match (a merge commit, a hand-written commit with no
    PR suffix, any scope or type this changelog doesn't carry) is silently
    skipped — not every commit is Penny's to announce."""
    entries: list[ChangelogEntry] = []
    for line in lines:
        match = _COMMIT_LINE_RE.match(line)
        if match is None:
            continue
        entries.append(
            ChangelogEntry(pr=int(match["pr"]), text=f"{match['kind']}(penny): {match['rest']}")
        )
    return entries


def load_commits(path: Path = COMMITS_PATH) -> list[ChangelogEntry]:
    """Every penny-scope commit the generated window carries, oldest first —
    empty when the file was never generated: a dev run via ``LOCAL=1 make
    check``, a bare ``pytest`` outside any make target, or a fresh worktree
    that hasn't run ``make up``/``prod``/``build`` yet. A missing file is not
    an error — a deploy with no commit history captured simply records and
    announces nothing."""
    if not path.is_file():
        return []
    return parse_commit_log(path.read_text().splitlines())


def recorded_prs(log: Memory) -> set[int]:
    """The PR numbers already recorded in the ``changelog`` log, read off each
    entry's own ``PR #<n>: `` prefix — no second tracking column needed."""
    recorded: set[int] = set()
    for entry in log.read_all():
        match = _PR_PREFIX_RE.match(entry.content)
        if match:
            recorded.add(int(match.group(1)))
    return recorded


def select_entries_to_announce(
    entries: list[ChangelogEntry], already: set[int]
) -> list[ChangelogEntry]:
    """Which of ``entries`` (oldest first) THIS deploy should announce (#2891).

    Never "every entry this call newly records": on the first deploy after
    switching from the hand-written yaml to commit-derived entries, the
    generated window can hold dozens of real PRs the yaml era never got
    around to recording — flooding one announcement with that backlog isn't
    the point. Instead, find the LAST entry already in the log; everything
    strictly after it, oldest first, is what's new enough to tell the user
    about. Anything before it that's also unrecorded (an old gap the yaml era
    missed) still gets recorded by the caller, just never announced.

    Nothing in ``entries`` is recorded yet — a fresh install, or a window
    that doesn't reach back far enough to overlap the log — records the
    whole list silently and announces nothing: there is no "newest recorded"
    position in this list to anchor off. The same falls out naturally on an
    ordinary restart with no new commits: every entry is already recorded,
    so the last match is the list's own last entry, and nothing follows it."""
    last_recorded_index = None
    for index, entry in enumerate(entries):
        if entry.pr in already:
            last_recorded_index = index
    if last_recorded_index is None:
        return []
    return entries[last_recorded_index + 1 :]


def record_new_entries(log: Memory, entries: list[ChangelogEntry]) -> list[ChangelogEntry]:
    """Append every entry not already recorded in ``log``, each stamped with
    the log's own clock (the deploy time). Idempotent: an entry recorded on
    one call is never appended again on a later one. Returns the entries that
    were actually appended — every commit new to the log, which is NOT the
    same set a deploy announces (``select_entries_to_announce`` decides that,
    from the same ``entries`` list, before this call mutates the log).
    """
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
