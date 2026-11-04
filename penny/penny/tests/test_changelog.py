"""Tests for ``penny.changelog`` — the commit-log parser and the idempotent
recorder behind #2845/#2891's startup step (``Penny._record_changelog_entries``).
The snapshot is a fixed bounded window with no cursor (#2973), so which of its
entries are hers to record is decided by the log itself (``select_new_entries``
— the log is the cursor), and the selected set is recorded and announced as
one — see ``test_changelog_announcement.py``/``test_chat_agent.py`` for the
startup path with an empty and a non-empty commit list."""

from __future__ import annotations

from penny.changelog import (
    COMMITS_PATH,
    ChangelogEntry,
    load_commits,
    parse_commit_log,
    record_new_entries,
    render_changelog_announcement,
    select_new_entries,
)
from penny.changelog_commits import first_body_sentence
from penny.constants import PennyConstants
from penny.tests.conftest import require_memory


def _record(commit_hash: str, subject: str, body: str = "") -> str:
    """One ``<hash>\x1f<subject>\x1f<body>`` record, the shape
    ``scripts/store_git_info.sh``'s own ``git log --pretty=format:
    '%H%x1f%s%x1f%b%x1e'`` writes for a single commit."""
    return f"{commit_hash}\x1f{subject}\x1f{body}"


def _log(*records: str) -> str:
    """The whole generated dump: records joined by ``\x1e``, exactly as git
    writes it (a body is itself multi-line, so a plain newline can't
    separate records)."""
    return "\x1e".join(records)


def test_parse_commit_log_keeps_feat_fix_perf_refactor_in_penny_scope():
    """The four kept types, in the `penny` scope, each become an entry whose
    PR number and text come straight off the commit title — never the hash,
    and never the leading `type(scope):` prefix either (#2891) — when none
    of them has a body to supply a first sentence instead."""
    log = _log(
        _record("aaa1111", "feat(penny): add a new way to save a note (#101)"),
        _record("bbb2222", "fix(penny): stop dropping a tool result (#102)"),
        _record("ccc3333", "perf(penny): batch the embedding backfill (#103)"),
        _record("ddd4444", "refactor(penny): drop the per-tool reasoning argument (#104)"),
    )
    entries = parse_commit_log(log)
    assert entries == [
        ChangelogEntry(pr=101, text="add a new way to save a note"),
        ChangelogEntry(pr=102, text="stop dropping a tool result"),
        ChangelogEntry(pr=103, text="batch the embedding backfill"),
        ChangelogEntry(pr=104, text="drop the per-tool reasoning argument"),
    ]


def test_parse_commit_log_prefers_the_bodys_first_sentence_over_the_title():
    """The real case this fix is for (#2891, round 4): the repo's squash
    setting makes a commit's body the merged PR's own description, whose
    first sentence is a plain, cold-readable summary — a live sample read
    the BARE TITLE `refactor(penny): drop the per-tool reasoning argument`
    and, from `refactor`'s own real meaning ("same behaviour"), rationally
    concluded nothing had changed for her. The body's own first sentence
    doesn't carry that misleading word."""
    log = _log(
        _record(
            "75edd400788b3612932a0573557f61666fc57581",
            "refactor(penny): drop the per-tool reasoning argument (#2820)",
            "Removes the per-tool `reasoning` schema field the framework used to "
            "inject into every tool call. It cost tokens for no reasoning the loop "
            "ever used.",
        )
    )
    entries = parse_commit_log(log)
    assert entries == [
        ChangelogEntry(
            pr=2820,
            text="Removes the per-tool `reasoning` schema field the framework used "
            "to inject into every tool call.",
        )
    ]
    assert entries[0].render() == (
        "PR #2820: Removes the per-tool `reasoning` schema field the framework "
        "used to inject into every tool call."
    )


def test_parse_commit_log_falls_back_to_the_stripped_title_with_no_usable_body():
    """An older commit (predating the squash-setting change) with no body at
    all still gets an entry — the title alone, prefix and suffix stripped."""
    log = _log(
        _record(
            "75edd400788b3612932a0573557f61666fc57581",
            "refactor(penny): drop the per-tool reasoning argument (#2820)",
        )
    )
    entries = parse_commit_log(log)
    assert entries == [ChangelogEntry(pr=2820, text="drop the per-tool reasoning argument")]


def test_parse_commit_log_drops_other_types_and_scopes():
    """`test`/`chore`/`docs` and every non-`penny` scope are dropped, along
    with a merge commit or any record with no `(#N)` suffix to key on."""
    log = _log(
        _record("aaa1111", "test(eval): port a chat case (#201)"),
        _record("bbb2222", "chore(ci): bump a pinned action (#202)"),
        _record("ccc3333", "docs(penny): note a design decision (#203)"),
        _record("ddd4444", "fix(harness): resolve a bare RUN on eval-report (#204)"),
        _record("eee5555", "fix(client): memory view crash (#205)"),
        _record("fff6666", "Merge branch 'main' into feature"),
        _record("ggg7777", "fix(penny): a commit with no PR suffix at all"),
    )
    assert parse_commit_log(log) == []


def test_first_body_sentence_takes_the_first_sentence_of_a_normal_body():
    """A real, unprefixed PR body: cut at the first `.`/`!`/`?` followed by
    whitespace, and a wrapped line collapses to one."""
    body = (
        "Penny now renders gemma's own chat template herself and sends the\n"
        "result as a raw completions prompt, so prod and every eval provider\n"
        "see byte-identical input.\n"
        "\n"
        "#2828 found the eval suite's pinned provider never opens a new thought."
    )
    assert first_body_sentence(body) == (
        "Penny now renders gemma's own chat template herself and sends the "
        "result as a raw completions prompt, so prod and every eval provider "
        "see byte-identical input."
    )


def test_first_body_sentence_skips_a_squashed_commit_bullet_echo():
    """A leading `* type(scope): ...` line is GitHub's own old
    squash-generated echo of one of the commits inside the PR (#2891's
    squash setting predates this repo's current ``PR_BODY`` one) — not a
    sentence Penny should read as the change's own summary."""
    body = (
        "* test(eval): add rule-effectiveness-from-run-facts case\n"
        "\n"
        "* fix(penny): let the model query per-run health facts\n"
        "\n"
        "The user asks whether a standing rule has helped a job. This is the "
        "real summary."
    )
    assert first_body_sentence(body) == "The user asks whether a standing rule has helped a job."


def test_first_body_sentence_skips_a_bare_because_label():
    """A `Because:` line with nothing of its own worth reading is a LABEL
    for the reason that follows on later lines, skipped whole — a `Because`
    clause explains WHY the change was made, never WHAT changed."""
    body = "Because:\n\nThe chat agent was dropping a reply whenever a tool call rode alongside it."
    assert first_body_sentence(body) == (
        "The chat agent was dropping a reply whenever a tool call rode alongside it."
    )


def test_first_body_sentence_skips_a_because_line_with_inline_content_too():
    """A `Because:` line is skipped WHOLE even when it carries its reason
    inline on the same line — still a cause, never the change's own
    summary — moving on to the real sentence that follows it."""
    body = (
        "Because: production's turn searched five more times and drafted raw "
        "URLs after a stored link was right in front of it.\n"
        "\n"
        "This commit makes the stored link's own address visible beside its number."
    )
    assert first_body_sentence(body) == (
        "This commit makes the stored link's own address visible beside its number."
    )


def test_first_body_sentence_returns_none_for_an_empty_body():
    """No body at all (an older, pre-squash-setting commit) — the caller
    falls back to the stripped title."""
    assert first_body_sentence("") is None
    assert first_body_sentence("   \n\n  ") is None


def test_first_body_sentence_returns_none_when_only_label_lines_remain():
    """A body that's nothing but bullet echoes has no real sentence to
    extract — the caller falls back to the stripped title, same as an
    empty body."""
    body = "* test(eval): add a case\n\n* fix(penny): the actual fix"
    assert first_body_sentence(body) is None


def test_first_body_sentence_clips_an_over_long_sentence_at_a_word_boundary():
    """A sentence past the 300-character cap is clipped at the last word
    boundary at or before the limit — never mid-word."""
    long_sentence = (
        "This fix changes the way Penny " + "really truly honestly " * 20 + "handles it."
    )
    body = long_sentence
    result = first_body_sentence(body)
    assert result is not None
    assert len(result) <= 300
    assert long_sentence.startswith(result)
    assert not result.endswith(" ")
    # Clipped mid-sentence, never mid-word: the next character after the cut
    # (in the original) is a word boundary (space), not a continuation of
    # the last word kept.
    assert long_sentence[len(result)] == " "


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


def test_select_new_entries_cuts_at_the_newest_recorded_pr():
    """The log is the cursor (#2973): the newest snapshot entry whose PR is
    already recorded marks where she last saw changes — everything after it,
    oldest first, is hers to record. Earlier recorded PRs in the snapshot
    don't matter; the NEWEST one is the cut."""
    entries = [ChangelogEntry(pr=9300 + n, text=f"change {n}") for n in range(4)]
    selected = select_new_entries(entries, recorded={9301})
    assert selected == [entries[2], entries[3]]


def test_select_new_entries_fresh_log_caps_at_the_newest_10():
    """A snapshot none of whose PRs is recorded (a fresh database) records only
    the newest 10 entries — the code owner's earlier ruling. Penny-scope
    entries, not raw commits: the cap applies after the parse filter."""
    entries = [ChangelogEntry(pr=9400 + n, text=f"change {n}") for n in range(12)]
    selected = select_new_entries(entries, recorded=set())
    assert selected == entries[2:]  # the newest 10 of the 12


def test_select_new_entries_on_an_unchanged_snapshot_returns_nothing():
    """A re-boot on the same snapshot: its newest PR is already recorded, so
    the cut is at the very end and nothing is selected."""
    entries = [ChangelogEntry(pr=9500 + n, text=f"change {n}") for n in range(3)]
    assert select_new_entries(entries, recorded={9502}) == []


def test_record_new_entries_fresh_log_records_only_the_newest_10(db):
    """Through the recorder itself: a fresh ``changelog`` log and a 12-entry
    snapshot appends only the newest 10 — the fresh-install cap (#2973) — and
    a second call on the same snapshot (the re-boot) appends nothing."""
    changelog = require_memory(db, PennyConstants.MEMORY_CHANGELOG_LOG)
    entries = [ChangelogEntry(pr=9600 + n, text=f"change {n}") for n in range(12)]

    added = record_new_entries(changelog, entries)
    assert added == entries[2:]
    assert [row.content for row in changelog.read_all()] == [e.render() for e in entries[2:]]

    again = record_new_entries(changelog, entries)
    assert again == []
    assert len(changelog.read_all()) == 10


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
