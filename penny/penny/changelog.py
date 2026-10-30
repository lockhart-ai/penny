"""Penny's changelog (#2845) — entries written FOR her about changes to her
tools, prompts and behaviour, so her self-reflection can tell "we changed
this" apart from "my own rule did this".

`penny/penny/changelog.yaml` is the single source list — one entry per PR
that changed what she does, in second person, authored and reviewed like any
other line of that PR. The model never writes an entry; this module only
loads the file and records it deterministically.

At startup (`Penny._record_changelog_entries`), every entry not yet recorded
is appended to the `changelog` system log (``PennyConstants.MEMORY_CHANGELOG_LOG``
— a plain append-only ``Log``, read like any other log via ``log_read``),
stamped with that deploy's own time rather than the PR's merge time: the date
that matters is when the change reached her, not when it merged.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from penny.constants import PennyConstants
from penny.database.memory import LogEntryInput, Memory

# Co-located with the source list it loads (#2845) — resolved relative to this
# module's own file, so it is correct whether this runs from the repo tree or
# the Docker image's copy, never against the process's working directory.
CHANGELOG_PATH = Path(__file__).with_name("changelog.yaml")

# The leading token every recorded row's content carries (``ChangelogEntry.render``).
# A plain ``Log`` has no second column to key idempotency on, so "already
# recorded?" reads this prefix back off the stored content instead — the render
# and the parse share one format, so they can't drift apart.
_PR_PREFIX_RE = re.compile(r"^PR #(\d+): ")


class ChangelogEntry(BaseModel):
    """One changelog item — a PR number and the sentence(s) written for Penny."""

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


def load_entries(path: Path = CHANGELOG_PATH) -> list[ChangelogEntry]:
    """Every entry in ``changelog.yaml``, in file order."""
    raw = yaml.safe_load(path.read_text())
    if not raw:
        return []
    return [ChangelogEntry.model_validate(item) for item in raw]


def recorded_prs(log: Memory) -> set[int]:
    """The PR numbers already recorded in the ``changelog`` log, read off each
    entry's own ``PR #<n>: `` prefix — no second tracking column needed."""
    recorded: set[int] = set()
    for entry in log.read_all():
        match = _PR_PREFIX_RE.match(entry.content)
        if match:
            recorded.add(int(match.group(1)))
    return recorded


def record_new_entries(log: Memory, entries: list[ChangelogEntry]) -> int:
    """Append every entry not already recorded in ``log``, each stamped with
    the log's own clock (the deploy time). Idempotent: an entry recorded on
    one call is never appended again on a later one. Returns how many landed.
    """
    already = recorded_prs(log)
    missing = [entry for entry in entries if entry.pr not in already]
    if not missing:
        return 0
    log.append(
        [LogEntryInput(content=entry.render()) for entry in missing],
        author=PennyConstants.CHANGELOG_LOG_AUTHOR,
    )
    return len(missing)
