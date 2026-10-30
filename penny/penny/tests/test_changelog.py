"""Tests for ``penny.changelog`` — the loader and the idempotent recorder
behind #2845's startup step (``Penny._record_changelog_entries``)."""

from __future__ import annotations

from penny.changelog import CHANGELOG_PATH, ChangelogEntry, load_entries, record_new_entries
from penny.constants import PennyConstants
from penny.tests.conftest import require_memory


def test_changelog_yaml_parses():
    """The shipped ``changelog.yaml`` parses into entries, each with a
    positive PR number and non-blank text, every PR number unique."""
    entries = load_entries()
    assert entries
    prs = [entry.pr for entry in entries]
    assert len(prs) == len(set(prs)), "every changelog entry should name a distinct PR"
    for entry in entries:
        assert entry.pr > 0
        assert entry.text


def test_changelog_path_points_at_the_shipped_file():
    """Co-located with the module that loads it (#2845) — present in the repo
    tree, and (by the same ``COPY penny/ ./penny/`` Dockerfile step that ships
    every other module here) in the built image too."""
    assert CHANGELOG_PATH.name == "changelog.yaml"
    assert CHANGELOG_PATH.is_file()


def test_record_new_entries_is_idempotent(db):
    """A fresh ``changelog`` log (its marker row seeded by migration 0131)
    records every entry once; recording the identical entries again adds
    nothing — the idempotency a redeploy's startup relies on."""
    changelog = require_memory(db, PennyConstants.MEMORY_CHANGELOG_LOG)
    entries = [
        ChangelogEntry(pr=9101, text="We added a new way to save a note."),
        ChangelogEntry(pr=9102, text="We renamed the lookup tool."),
    ]

    added = record_new_entries(changelog, entries)
    assert added == 2
    assert [row.content for row in changelog.read_all()] == [e.render() for e in entries]

    again = record_new_entries(changelog, entries)
    assert again == 0
    assert len(changelog.read_all()) == 2


def test_record_new_entries_only_appends_the_missing_ones(db):
    """A later startup that ships one new entry alongside ones already
    recorded appends only the new one — no duplicate, no re-ordering of the
    old rows."""
    changelog = require_memory(db, PennyConstants.MEMORY_CHANGELOG_LOG)
    record_new_entries(changelog, [ChangelogEntry(pr=9201, text="first change")])

    added = record_new_entries(
        changelog,
        [
            ChangelogEntry(pr=9201, text="first change"),
            ChangelogEntry(pr=9202, text="second change"),
        ],
    )
    assert added == 1
    assert [row.content for row in changelog.read_all()] == [
        "PR #9201: first change",
        "PR #9202: second change",
    ]
