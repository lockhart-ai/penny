"""Tests for ``penny.changelog`` — the commit-log parser, the idempotent
recorder, and the announce-selection rule behind #2845/#2891's startup step
(``Penny._record_changelog_entries``)."""

from __future__ import annotations

from penny.changelog import (
    COMMITS_PATH,
    ChangelogEntry,
    load_commits,
    parse_commit_log,
    record_new_entries,
    render_changelog_announcement,
    select_entries_to_announce,
)
from penny.constants import PennyConstants
from penny.tests.conftest import require_memory


def test_parse_commit_log_keeps_feat_fix_perf_refactor_in_penny_scope():
    """The four kept types, in the `penny` scope, each become an entry whose
    PR number and text come straight off the commit line — never the hash."""
    lines = [
        "aaa1111\x1ffeat(penny): add a new way to save a note (#101)",
        "bbb2222\x1ffix(penny): stop dropping a tool result (#102)",
        "ccc3333\x1fperf(penny): batch the embedding backfill (#103)",
        "ddd4444\x1frefactor(penny): drop the per-tool reasoning argument (#104)",
    ]
    entries = parse_commit_log(lines)
    assert entries == [
        ChangelogEntry(pr=101, text="feat(penny): add a new way to save a note"),
        ChangelogEntry(pr=102, text="fix(penny): stop dropping a tool result"),
        ChangelogEntry(pr=103, text="perf(penny): batch the embedding backfill"),
        ChangelogEntry(pr=104, text="refactor(penny): drop the per-tool reasoning argument"),
    ]


def test_parse_commit_log_drops_other_types_and_scopes():
    """`test`/`chore`/`docs` and every non-`penny` scope are dropped, along
    with a merge commit or any line with no `(#N)` suffix to key on."""
    lines = [
        "aaa1111\x1ftest(eval): port a chat case (#201)",
        "bbb2222\x1fchore(ci): bump a pinned action (#202)",
        "ccc3333\x1fdocs(penny): note a design decision (#203)",
        "ddd4444\x1ffix(harness): resolve a bare RUN on eval-report (#204)",
        "eee5555\x1ffix(client): memory view crash (#205)",
        "fff6666\x1fMerge branch 'main' into feature",
        "ggg7777\x1ffix(penny): a commit with no PR suffix at all",
    ]
    assert parse_commit_log(lines) == []


def test_load_commits_returns_empty_when_the_file_is_missing(tmp_path):
    """A dev run via `LOCAL=1 make check`, a bare `pytest`, or a fresh
    worktree that hasn't run `make up`/`prod`/`build` yet never generated
    this file — that's not an error, it's nothing to announce."""
    assert load_commits(tmp_path / "never-generated.txt") == []


def test_load_commits_path_is_colocated_with_the_module():
    """Co-located with the module that parses it (#2891) — the same
    `COPY penny/ ./penny/` Dockerfile step that ships every other module
    here ships this generated file too, with no Dockerfile change."""
    assert COMMITS_PATH.name == "changelog_commits.txt"
    assert COMMITS_PATH.parent.name == "penny"


def test_select_entries_to_announce_fresh_install_announces_nothing():
    """Nothing in the list is recorded yet (a fresh install, or a window
    that doesn't reach back far enough to overlap the log) — there is no
    "newest recorded" position to anchor off, so the whole list is left for
    the caller to record silently, and none of it is announced."""
    entries = [
        ChangelogEntry(pr=101, text="first change"),
        ChangelogEntry(pr=102, text="second change"),
    ]
    assert select_entries_to_announce(entries, already=set()) == []


def test_select_entries_to_announce_restart_with_nothing_new_announces_nothing():
    """Every entry in the list is already recorded (an ordinary restart with
    no new commits merged) — the last match is the list's own last entry, so
    nothing follows it."""
    entries = [
        ChangelogEntry(pr=101, text="first change"),
        ChangelogEntry(pr=102, text="second change"),
    ]
    assert select_entries_to_announce(entries, already={101, 102}) == []


def test_select_entries_to_announce_only_the_tail_after_the_last_recorded_one():
    """The real case this rule exists for (#2891): a generated window that
    holds old PRs the yaml era recorded out of order, scattered gaps the
    yaml era never recorded (never announced either), and a genuinely new
    tail. Only the entries strictly after the LAST already-recorded one,
    oldest first, are announced — not every entry this call newly records."""
    entries = [
        ChangelogEntry(pr=101, text="an old recorded PR"),
        ChangelogEntry(pr=102, text="an old gap the yaml era never recorded"),
        ChangelogEntry(pr=103, text="the newest already-recorded PR"),
        ChangelogEntry(pr=104, text="a genuinely new PR"),
        ChangelogEntry(pr=105, text="another genuinely new PR"),
    ]
    announced = select_entries_to_announce(entries, already={101, 103})
    assert announced == [
        ChangelogEntry(pr=104, text="a genuinely new PR"),
        ChangelogEntry(pr=105, text="another genuinely new PR"),
    ]


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
    assert added == entries
    assert [row.content for row in changelog.read_all()] == [e.render() for e in entries]

    again = record_new_entries(changelog, entries)
    assert again == []
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
    assert added == [ChangelogEntry(pr=9202, text="second change")]
    assert [row.content for row in changelog.read_all()] == [
        "PR #9201: first change",
        "PR #9202: second change",
    ]


def test_render_changelog_announcement_is_whole_literal():
    """The startup delivery turn's input (#2846) is framework-rendered verbatim —
    never model-written — so the exact text the chat agent reads is pinned here
    rather than left to whatever a live run happens to produce."""
    entries = [
        ChangelogEntry(pr=9101, text="We added a new way to save a note."),
        ChangelogEntry(pr=9102, text="We renamed the lookup tool."),
    ]
    assert render_changelog_announcement(entries) == (
        "These changes to your tools and behaviour reached you with this deploy:\n"
        "\n"
        "- PR #9101: We added a new way to save a note.\n"
        "- PR #9102: We renamed the lookup tool.\n"
        "\n"
        "Let the user know what changed, in your own words."
    )


def test_render_changelog_announcement_with_no_entries_is_whole_literal():
    """Penny announces on every startup (#2890), including a deploy that
    brought no new entries (harness-only, CI-only, or a restart on an
    already-recorded commit) — the render says so and asks for a short
    "I'm back" message instead of listing entries."""
    assert render_changelog_announcement([]) == (
        "You've just restarted. This deploy changed nothing in your tools or behaviour.\n"
        "\n"
        "Send the user a short message letting them know you're back."
    )
