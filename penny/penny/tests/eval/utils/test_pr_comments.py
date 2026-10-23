"""Plain (non-eval) tests for mark-and-replace's one decision (``pr_comments.py``, #2734):
which of a PR's existing comments ``eval-report`` may delete before posting a fresh one.

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
    REPORT_MARKER,
    USAGE,
    ExistingComment,
    main,
    select_marked_comments,
)

APP_LOGIN = "penny-team[bot]"
OTHER_BOT = "dependabot[bot]"


def _comment(id: int, author: str, body: str, html_url: str = "") -> ExistingComment:
    return ExistingComment(id=id, author_login=author, body=body, html_url=html_url)


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


def test_cli_with_no_comments_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main([REPORT_MARKER, APP_LOGIN]) == 0
    assert capsys.readouterr().out == ""


def test_cli_bad_args_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([REPORT_MARKER]) == 2
    assert capsys.readouterr().err.strip() == USAGE
