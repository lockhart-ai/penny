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
``git log`` dump, oldest first, written to the gitignored
`changelog_commits.txt`, co-located with this module so it rides the same
`COPY penny/ ./penny/` Dockerfile step as everything else here. The image
has no git, and the host's own Python can't be trusted to import this
package (an agent container's host Python may be 3.9 with no pydantic
installed) — so the host writes the RAW log only, with plain shell tools,
and this module does the parsing entirely at RUNTIME, over whatever that
file holds.

The script writes a fixed bounded window — the newest few hundred commits
touching the runtime paths, no cursor (#2973). No position or dedupe logic
lives in Python here: whatever survives ``parse_commit_log`` on that file IS
this deploy's candidate set — ``penny.changelog.record_new_entries`` decides
by the ``changelog`` log itself (the log is the cursor) which of those to
record, and records and announces that set as one, by construction.

Each dumped record carries the commit's BODY too, not just its title
(#2891, round 4): the repo's squash setting
(``squash_merge_commit_message: PR_BODY``) makes a squash commit's body the
merged PR's own description, whose first sentence is — by the PR-body
convention — a plain, cold-readable summary of the whole change. The entry
text is that first sentence when one is usable; otherwise (an older commit,
predating the setting, or a body with nothing but commit-title echoes) it
falls back to the title itself, with its `type(scope):` prefix and trailing
`(#N)` stripped (#2891, round 3 — a title's TYPE is developer metadata that
can mislead: a live sample read a `refactor(penny):` commit by `refactor`'s
own real meaning, "same behaviour", and rationally concluded a tool-call
argument removal had changed nothing for her).

Since #3063 there is no title-scope filter here: WHICH commits reach this
parser is decided by the shared runtime-path rule (`scripts/runtime-paths`
— the snapshot logs only commits touching those paths, the same rule the
deploy watcher restarts on), so every record in the dump is a runtime change
whatever its `type(scope):`. What the title is still read for is the PR
number — the `changelog` log's cursor is keyed by PR, so a commit whose
title carries no trailing `(#N)` suffix (a merge commit, a hand-pushed
commit) has nothing to be recorded under and is skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

# Co-located with the source list it loads (#2845) — resolved relative to this
# module's own file, so it is correct whether this runs from the repo tree or
# the Docker image's copy, never against the process's working directory.
COMMITS_PATH = Path(__file__).with_name("changelog_commits.txt")

# One record of the generated commit dump: ``<hash>\x1f<subject>\x1f<body>``,
# records joined by ``\x1e`` (a body is itself multi-line, so a plain newline
# can't separate records — ``scripts/store_git_info.sh``'s own
# ``git log --pretty=format:'%H%x1f%s%x1f%b%x1e'``). ``<subject>`` is a
# squash-merged PR's own conventional title with its trailing ``(#<pr>)``
# GitHub appends. There is no title-scope filter any more (#3063): the
# snapshot itself holds only commits touching the shared runtime-path rule,
# so a record of any type or scope — or with no conventional prefix at all —
# is hers to announce. The title is read only for the PR number (the
# changelog log's cursor is keyed by PR — no suffix, no entry) and for the
# title-fallback text.
_RECORD_SEP = "\x1e"
_FIELD_SEP = "\x1f"
_PR_SUFFIX_RE = re.compile(r"^(?P<title>.*) \(#(?P<pr>\d+)\)$")

# A leading conventional-commit prefix on the title fallback —
# ``feat(penny): ``, ``chore(ci): ``, a bare ``fix: `` — whatever the type or
# scope (#3063): developer metadata the entry text is better without (#2891).
_CONVENTIONAL_PREFIX_RE = re.compile(r"^[a-z]+(?:\([^)]*\))?: (?P<rest>.+)$")

# A leading line of the body that's just an echo of ONE of the squashed
# commits that made up the PR — the shape GitHub generated automatically
# before this repo's squash setting became ``PR_BODY`` (#2891, round 4) —
# or a bare "Because:" label. Neither is a usable first sentence; skip past
# both (and any blank line between them) to find the real one.
_BULLET_ECHO_RE = re.compile(r"^\*\s*(?:feat|fix|perf|refactor|test|chore|docs)\([\w.-]+\):\s")
_BECAUSE_LABEL_RE = re.compile(r"^Because:", re.IGNORECASE)

# Where the first sentence ends: a `.`/`!`/`?` immediately followed by
# whitespace or the string's end, OR a blank line (a paragraph break) —
# whichever comes first. `re.search` on this alternation finds whichever
# starts earliest, which is exactly "whichever comes first".
_SENTENCE_CUT_RE = re.compile(r"[.!?](?=\s|$)|\n[ \t]*\n")

# A body sentence longer than this is clipped at the last word boundary at
# or before the limit — never mid-word.
_MAX_ENTRY_TEXT_LENGTH = 300


class ChangelogEntry(BaseModel):
    """One changelog item — a PR number and the text recorded for it: the
    merged PR's own body's first sentence, or (no usable one) its title
    with the `type(scope):` prefix stripped (#2891) — never hand-written."""

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


def _skip_label_lines(lines: list[str]) -> list[str]:
    """Drop LEADING lines that are a squashed-commit bullet echo or a bare
    `Because:` label (and any blank line among them) — the real first
    sentence starts at the first line that is neither."""
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or _BULLET_ECHO_RE.match(line) or _BECAUSE_LABEL_RE.match(line):
            index += 1
            continue
        break
    return lines[index:]


def _clip_at_word_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    cut = truncated.rfind(" ")
    if cut <= 0:
        return truncated
    return truncated[:cut].rstrip()


def first_body_sentence(body: str) -> str | None:
    """The PR body's own first sentence, or ``None`` when the body has no
    usable one (empty, or nothing past its bullet-echo/`Because:` preamble)
    — the caller falls back to the stripped title in that case.

    Collapses internal line breaks within the sentence to single spaces (a
    sentence a PR author wrapped across lines reads as one line here), and
    clips to ``_MAX_ENTRY_TEXT_LENGTH`` characters at a word boundary."""
    content_lines = _skip_label_lines(body.splitlines())
    text = "\n".join(content_lines).strip()
    if not text:
        return None
    match = _SENTENCE_CUT_RE.search(text)
    if match is None:
        sentence = text
    elif text[match.start()] in ".!?":
        sentence = text[: match.end()]
    else:
        sentence = text[: match.start()]
    sentence = " ".join(sentence.split()).strip()
    if not sentence:
        return None
    return _clip_at_word_boundary(sentence, _MAX_ENTRY_TEXT_LENGTH)


def parse_commit_log(text: str) -> list[ChangelogEntry]:
    """Parse the whole generated dump (``<hash>\x1f<subject>\x1f<body>``
    records, ``\x1e``-separated — ``scripts/store_git_info.sh``'s own
    ``git log --pretty=format:'%H%x1f%s%x1f%b%x1e'``, oldest-first via
    ``--reverse``) into changelog entries. Pure and unit-tested on its own
    — the only git/file plumbing around it (the script, ``load_commits``
    below) stays thin enough to need no test of its own.

    Every record in the dump is a runtime commit — the snapshot is scoped by
    the shared runtime-path rule (#3063) — so there is no title filter: a
    record of any type or scope, or with no conventional prefix at all,
    becomes an entry. A record with no trailing ``(#N)`` suffix is skipped:
    the changelog log's cursor is keyed by PR number, so it has nothing to be
    recorded under. Each kept record's text is its body's first sentence
    (``first_body_sentence``) when it has one, else its title with any
    `type(scope):` prefix and the trailing `(#N)` stripped."""
    entries: list[ChangelogEntry] = []
    for record in text.split(_RECORD_SEP):
        record = record.lstrip("\n")  # git's own between-commit separator only
        if not record:
            continue
        fields = record.split(_FIELD_SEP, 2)
        if len(fields) < 2:
            continue
        _commit_hash, subject, body = (fields + [""])[:3]
        match = _PR_SUFFIX_RE.match(subject)
        if match is None:
            continue
        title = match["title"]
        prefix = _CONVENTIONAL_PREFIX_RE.match(title)
        entry_text = first_body_sentence(body) or (prefix["rest"] if prefix else title)
        if not entry_text.strip():
            continue  # a subject that is nothing but its (#N) suffix
        entries.append(ChangelogEntry(pr=int(match["pr"]), text=entry_text))
    return entries


def load_commits(path: Path = COMMITS_PATH) -> list[ChangelogEntry]:
    """Every runtime commit the generated window carries, oldest first —
    empty when the file was never generated: a dev run via ``LOCAL=1 make
    check``, a bare ``pytest`` outside any make target, or a fresh worktree
    that hasn't run ``make up``/``prod``/``prod-ios`` yet. A missing file is
    not an error — a deploy with no commit history captured simply records
    and announces nothing."""
    if not path.is_file():
        return []
    return parse_commit_log(path.read_text())
