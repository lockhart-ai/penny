"""Mark-and-replace (#2734): which of a PR's EXISTING comments ``eval-report`` may delete
before it posts a fresh one.

Every comment ``eval-report`` posts carries a hidden HTML-comment marker naming its kind —
the per-case/summary report (:data:`REPORT_MARKER`), or the branch-vs-main comparison table
(:data:`COMPARE_MARKER`) — so a re-run never piles reports up: before posting, the recipe
deletes the PR's existing MARKED comments of that one kind, and only ones the App itself
authored. A reviewer's own comment, a comment with no marker, or one carrying the OTHER
marker, is never a candidate.

This module holds the one PURE decision — which comment ids qualify — so it is unit-testable
with no network; the ``eval-report`` recipe does the GitHub listing (``gh api
repos/{owner}/{repo}/issues/<pr>/comments``) and deleting (``gh api -X DELETE
.../issues/comments/<id>``) over this module's CLI, which reads that listing's JSON from
stdin and prints the ids (+ URLs, for a human to read on a dry run) to delete.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

# The two marker kinds every posted comment carries, each its own HTML comment — so deleting
# one kind never touches the other, and a plain substring check can't cross them (neither
# string is a substring of the other). Must match the Makefile's `EVAL_REPORT_MARKER` /
# `EVAL_COMPARE_MARKER` — the recipe passes the literal text here, it never greps for it.
REPORT_MARKER = "<!-- penny-eval-report -->"
COMPARE_MARKER = "<!-- penny-eval-compare -->"

USAGE = (
    "usage: gh api repos/{owner}/{repo}/issues/<pr>/comments --paginate | "
    "python -m penny.tests.eval.utils.pr_comments <marker> <app_login>"
)


@dataclass(frozen=True)
class ExistingComment:
    """One comment off `gh api`'s PR-comments listing — the three fields this module's one
    decision needs, read out of the GitHub REST shape (``id``, ``user.login``, ``body``,
    ``html_url``)."""

    id: int
    author_login: str
    body: str
    html_url: str = ""


def select_marked_comments(
    comments: Sequence[ExistingComment], marker: str, app_login: str
) -> list[ExistingComment]:
    """The comments ``eval-report`` may delete: carries ``marker`` AND authored by
    ``app_login`` — both, so a reviewer's own comment and a differently-marked report both
    survive untouched, in the order the listing returned them."""
    return [c for c in comments if c.author_login == app_login and marker in c.body]


def _parse_comments(raw: str) -> list[ExistingComment]:
    """``gh api``'s JSON array of comment objects → the fields this module reads."""
    items = json.loads(raw) if raw.strip() else []
    return [
        ExistingComment(
            id=item["id"],
            author_login=item["user"]["login"],
            body=item.get("body", ""),
            html_url=item.get("html_url", ""),
        )
        for item in items
    ]


# ── CLI: read the listing JSON from stdin, print `id<TAB>html_url` per qualifying comment ──
def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    marker, app_login = argv
    comments = _parse_comments(sys.stdin.read())
    for comment in select_marked_comments(comments, marker, app_login):
        print(f"{comment.id}\t{comment.html_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
