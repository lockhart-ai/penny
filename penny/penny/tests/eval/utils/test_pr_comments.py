"""Plain (non-eval) tests for mark-and-replace's decisions (``pr_comments.py``, #2734):
which of a PR's existing comments ``eval-report`` may delete before posting a fresh one —
by marker, and by the shape a PRE-#2734 legacy report carried before the marker existed.

No network, no `gh` — the comment listing is a plain Python list built by hand, standing in
for `gh api`'s JSON. The CLI is driven over stdin/stdout like the rest of the harness's
small CLIs (``checkpoint.py``, ``comment_split.py``).
"""

from __future__ import annotations

import io
import json

import pytest

from penny.tests.eval.utils.pr_comments import (
    COMPARE_MARKER,
    INCLUDE_LEGACY_FLAG,
    REPORT_MARKER,
    USAGE,
    ExistingComment,
    is_legacy_report,
    main,
    select_comments_to_delete,
    select_legacy_report_comments,
    select_marked_comments,
)

APP_LOGIN = "penny-team[bot]"
OTHER_BOT = "dependabot[bot]"

# A real legacy per-case report's opening (PR #2770, before #2734): the heading, a measures
# table whose first row is the checks row, then the rest of the report.
LEGACY_CASE_BODY = (
    "### ⚪ `digest-delivered-whole` 1/20 — `google/gemma-4-26b-a4b-it`\n"
    "\n"
    "In a headline-digest collector, when the job writes one entry.\n"
    "\n"
    "| measure | reading |\n"
    "|---|---|\n"
    "| **checks** | 🟢 105 / 105 · 100% |\n"
    "| **variance** | ⚪ max H 0.000 `tool sequence` |\n"
)
# A hand-written comparison a kitten posted by hand before `compare.py` existed.
LEGACY_COMPARISON_BODY = (
    "## Case-by-case: branch vs main, same base (4716375), on Makora\n"
    "\n"
    "Both runs scope the 20 cases, model `google/gemma-4-26b-a4b-it`.\n"
)
LEGACY_CASE_ONLY_BODY = (
    "## `digest-set-up-in-chat-carries-every-item` only — 2 more runs per side, Makora\n"
    "\n"
    "Per request: two more scoped runs on the branch.\n"
)
# A real coordinator's note on the same PR — quotes numbers in a table, but has neither a
# case heading nor a checks row nor either comparison heading. Never a legacy report.
COORDINATOR_NOTE_BODY = (
    "Coordinator's read of the one case that dropped, `digest-set-up-in-chat-carries-"
    "every-item` (84/120 on the branch against 102/120 on main):\n"
    "\n"
    "| | main | branch |\n"
    "|---|---|---|\n"
    "| checks | 102/120 | 84/120 |\n"
)


def _comment(id: int, author: str, body: str, html_url: str = "") -> ExistingComment:
    return ExistingComment(id=id, author_login=author, body=body, html_url=html_url)


# ── select_marked_comments (marker-based, #2734's own reports) ───────────────
def test_only_marked_app_authored_comments_are_selected() -> None:
    """The four combinations: marked+App (selected), marked+someone else (not), unmarked+App
    (not), and marked with the OTHER kind's marker (not) — only the first qualifies."""
    comments = [
        _comment(1, APP_LOGIN, f"### `case-a`\n\n{REPORT_MARKER}"),
        _comment(2, OTHER_BOT, f"### `case-b`\n\n{REPORT_MARKER}"),
        _comment(3, APP_LOGIN, "a plain reviewer-style note with no marker"),
        _comment(4, APP_LOGIN, f"## Branch vs main\n\n{COMPARE_MARKER}"),
    ]
    selected = select_marked_comments(comments, REPORT_MARKER, APP_LOGIN)
    assert [c.id for c in selected] == [1]


def test_the_compare_marker_selects_independently_of_the_report_marker() -> None:
    """The comparison comment (marked ``COMPARE_MARKER``) is a different kind from a report —
    selecting on one marker never catches the other, so replacing a report never touches the
    standing comparison comment and vice versa."""
    comments = [
        _comment(1, APP_LOGIN, f"### `case-a`\n\n{REPORT_MARKER}"),
        _comment(2, APP_LOGIN, f"## Branch vs main\n\n{COMPARE_MARKER}"),
    ]
    assert [c.id for c in select_marked_comments(comments, REPORT_MARKER, APP_LOGIN)] == [1]
    assert [c.id for c in select_marked_comments(comments, COMPARE_MARKER, APP_LOGIN)] == [2]


def test_no_marked_comments_selects_nothing() -> None:
    comments = [_comment(1, APP_LOGIN, "an ordinary human comment")]
    assert select_marked_comments(comments, REPORT_MARKER, APP_LOGIN) == []
    assert select_marked_comments([], REPORT_MARKER, APP_LOGIN) == []


# ── is_legacy_report / select_legacy_report_comments (pre-#2734 clutter) ─────
def test_a_legacy_case_report_matches() -> None:
    """The real shape a pre-#2734 per-case report carried: the heading line together with
    its checks row — both real text pulled off PR #2770's existing comments."""
    assert is_legacy_report(LEGACY_CASE_BODY) is True


def test_a_hand_written_comparison_heading_matches_either_form() -> None:
    assert is_legacy_report(LEGACY_COMPARISON_BODY) is True
    assert is_legacy_report(LEGACY_CASE_ONLY_BODY) is True


def test_a_human_comment_never_matches() -> None:
    """A human reviewer's comment never matches, however it's shaped — it never carries the
    App's own report structure, and the predicate doesn't need authorship to say so."""
    assert is_legacy_report("Looks good to me, approving.") is False
    assert is_legacy_report("### `case-a` — not actually a report heading") is False


def test_an_app_comment_that_is_not_a_report_does_not_match() -> None:
    """A coordinator's own note, posted by the App token like everything else, quotes
    numbers in a table but carries neither a case heading + checks row nor either
    comparison heading — it is not a report, and must not be swept as one."""
    assert is_legacy_report(COORDINATOR_NOTE_BODY) is False


def test_select_legacy_report_comments_requires_both_shape_and_app_authorship() -> None:
    comments = [
        _comment(1, APP_LOGIN, LEGACY_CASE_BODY),
        _comment(2, OTHER_BOT, LEGACY_CASE_BODY),  # right shape, wrong author
        _comment(3, APP_LOGIN, COORDINATOR_NOTE_BODY),  # right author, wrong shape
        _comment(4, APP_LOGIN, LEGACY_COMPARISON_BODY),
    ]
    assert [c.id for c in select_legacy_report_comments(comments, APP_LOGIN)] == [1, 4]


# ── select_comments_to_delete (the one combined decision eval-report calls) ──
def test_include_legacy_unions_marked_and_legacy_shaped_comments() -> None:
    comments = [
        _comment(1, APP_LOGIN, f"### `case-a`\n\n{REPORT_MARKER}"),  # marked
        _comment(2, APP_LOGIN, LEGACY_CASE_BODY),  # legacy, unmarked
        _comment(3, APP_LOGIN, COORDINATOR_NOTE_BODY),  # neither
        _comment(4, OTHER_BOT, LEGACY_CASE_BODY),  # legacy shape, wrong author
    ]
    without_legacy = select_comments_to_delete(comments, REPORT_MARKER, APP_LOGIN)
    assert [c.id for c in without_legacy] == [1]

    with_legacy = select_comments_to_delete(comments, REPORT_MARKER, APP_LOGIN, include_legacy=True)
    assert [c.id for c in with_legacy] == [1, 2]


def test_include_legacy_never_double_counts_a_comment_matching_both() -> None:
    """A comment marked AND legacy-shaped (shouldn't happen once this ships, but the
    dedup is what keeps a future overlap from posting the same id to `gh api -X DELETE`
    twice)."""
    both = f"{LEGACY_CASE_BODY}\n{REPORT_MARKER}"
    comments = [_comment(1, APP_LOGIN, both)]
    selected = select_comments_to_delete(comments, REPORT_MARKER, APP_LOGIN, include_legacy=True)
    assert [c.id for c in selected] == [1]


# ── CLI ────────────────────────────────────────────────────────────────────
def test_cli_reads_gh_apis_json_and_prints_id_and_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI's whole contract: `gh api`'s comment-listing JSON on stdin in, one
    ``id<TAB>html_url`` line per qualifying comment on stdout, in listing order."""
    raw = json.dumps(
        [
            {
                "id": 11,
                "user": {"login": APP_LOGIN},
                "body": f"x {REPORT_MARKER}",
                "html_url": "https://x/11",
            },
            {
                "id": 12,
                "user": {"login": "someone-else"},
                "body": f"y {REPORT_MARKER}",
                "html_url": "https://x/12",
            },
        ]
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    assert main([REPORT_MARKER, APP_LOGIN]) == 0
    assert capsys.readouterr().out == "11\thttps://x/11\n"


def test_cli_include_legacy_flag_sweeps_unmarked_comments_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = json.dumps(
        [{"id": 21, "user": {"login": APP_LOGIN}, "body": LEGACY_CASE_BODY, "html_url": "u21"}]
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    assert main([REPORT_MARKER, APP_LOGIN, INCLUDE_LEGACY_FLAG]) == 0
    assert capsys.readouterr().out == "21\tu21\n"


def test_cli_with_no_comments_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main([REPORT_MARKER, APP_LOGIN]) == 0
    assert capsys.readouterr().out == ""


def test_cli_bad_args_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([REPORT_MARKER]) == 2
    assert capsys.readouterr().err.strip() == USAGE
    assert main([REPORT_MARKER, APP_LOGIN, "--bogus"]) == 2
