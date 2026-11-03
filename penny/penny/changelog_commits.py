"""Pure parsing for Penny's changelog source (#2891) — `ChangelogEntry`,
`parse_commit_log`, `load_commits`. No DB or framework imports: this module
exists so `penny.config` can import it at MODULE level with no cycle back
through `penny.database`/`penny.validation`/`penny.agents` (which all end up
importing `penny.config` themselves). The DB-facing half — recording an
entry into the `changelog` log, rendering the delivery turn's input — lives
in `penny.changelog`, which re-exports everything here for the existing
`from penny.changelog import ChangelogEntry, ...` call sites.

The commit list is captured on the HOST, before the image builds
(`scripts/store_git_info.sh`, called by the `up`/`prod`/`prod-ios` Makefile
recipes — the mozilla/experimenter `store_git_info.sh` pattern): a plain
``git log`` dump — `<hash>\x1f<subject>` lines, oldest first — written to
the gitignored `changelog_commits.txt`, co-located with this module so it
rides the same `COPY penny/ ./penny/` Dockerfile step as everything else
here. The image has no git, and the host's own Python can't be trusted to
import this package (an agent container's host Python may be 3.9 with no
pydantic installed) — so the host writes the RAW log only, with plain shell
tools, and this module does the parsing entirely at RUNTIME, over whatever
that file holds.

The script reads a cursor dotfile the host keeps under `data/` (gitignored,
a bind mount — survives every image rebuild): the hash of the last commit a
real deploy announced. It writes the range since that cursor, or — unset,
or history-rewritten past it — the newest few raw commits. `prod`/`prod-ios`
advance the cursor themselves, after a successful run; `up` (dev) never
does. No position or dedupe logic lives in Python: whatever survives
``parse_commit_log``'s scope filter on that file IS this deploy's new
changes — ``penny.changelog.record_new_entries`` records and announces it
as one set, by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

# Co-located with the source list it loads (#2845) — resolved relative to this
# module's own file, so it is correct whether this runs from the repo tree or
# the Docker image's copy, never against the process's working directory.
COMMITS_PATH = Path(__file__).with_name("changelog_commits.txt")

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
        (in ``penny.changelog``) reads the leading ``PR #<n>: `` back off
        this same format."""
        return f"PR #{self.pr}: {self.text}"


def parse_commit_log(lines: Iterable[str]) -> list[ChangelogEntry]:
    """Parse ``<hash>\x1f<subject>`` lines (``git log``'s own
    ``--pretty=format:%H%x1f%s``, oldest-first via ``--reverse``) into
    changelog entries. Pure and unit-tested on its own — the only git/file
    plumbing around it (``scripts/store_git_info.sh``, ``load_commits``
    below) stays thin enough to need no test of its own.

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
    that hasn't run ``make up``/``prod``/``prod-ios`` yet. A missing file is
    not an error — a deploy with no commit history captured simply records
    and announces nothing."""
    if not path.is_file():
        return []
    return parse_commit_log(path.read_text().splitlines())
