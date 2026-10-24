"""Mark-and-replace (#2734): which of a PR's EXISTING comments ``eval-report`` may delete
before it posts a fresh one.

Every comment ``eval-report`` posts carries a hidden HTML-comment marker naming its kind —
the per-case/summary report (:data:`REPORT_MARKER`), or the branch-vs-main comparison table
(:data:`COMPARE_MARKER`) — so a re-run never piles reports up: before posting, the recipe
deletes the PR's existing MARKED comments of that one kind, and only ones the App itself
authored. A reviewer's own comment, a comment with no marker, or one carrying the OTHER
marker, is never a candidate.

A PR can also carry LEGACY report comments from before this feature existed — unmarked,
because the marker didn't exist yet — and those are expected to vanish too, so a PR shows
only the latest run at all times rather than the latest run plus whatever predates #2734.
:func:`is_legacy_report` recognises them by SHAPE: either the per-case report heading
(``### <glyph> \\`<case>\\` <i>/<n> — \\`<model>\\```) together with its ``| **checks** |``
row, or one of the hand-written comparison headings kittens posted by hand before this
module existed (``## Case-by-case: …`` / ``## \\`<case>\\` only — …``). Both are AND'd with
the App's own authorship, same as a marked comment — a human's comment that happens to
quote a heading is never a candidate.

This module holds the PURE decisions — which comment ids qualify, by marker or by legacy
shape — so they are unit-testable with no network; the ``eval-report`` recipe does the
GitHub listing (``gh api repos/{owner}/{repo}/issues/<pr>/comments``) and deleting (``gh api
-X DELETE .../issues/comments/<id>``) over this module's CLI, which reads that listing's
JSON from stdin and prints the ids (+ URLs, for a human to read on a dry run) to delete.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

# The two marker kinds every posted comment carries, each its own HTML comment — so deleting
# one kind never touches the other, and a plain substring check can't cross them (neither
# string is a substring of the other). Must match the Makefile's `EVAL_REPORT_MARKER` /
# `EVAL_COMPARE_MARKER` — the recipe passes the literal text here, it never greps for it.
REPORT_MARKER = "<!-- penny-eval-report -->"
COMPARE_MARKER = "<!-- penny-eval-compare -->"

# A legacy per-case report's own heading line (`report.case_heading`, pre-#2734 and
# unchanged by it): a coloured glyph, the backtick-quoted case id, an optional `N/M` run
# position, an em dash, and the backtick-quoted model — e.g.
# "### ⚪ `digest-delivered-whole` 1/20 — `google/gemma-4-26b-a4b-it`". A single-case run's
# heading carries no position, so it is optional.
LEGACY_CASE_HEADING = re.compile(r"^### [🟢🟡🔴⚪] `[^`]+`(?: \d+/\d+)? — `[^`]+`$", re.MULTILINE)
# The measures-table row every such heading's document carries right under it
# (`report._MEASURE_ROW` filled with `measure="checks"`) — paired with the heading so a
# comment that merely QUOTES a case heading (a coordinator's note, say) is never mistaken
# for the report that heading came from.
LEGACY_CHECKS_ROW = "| **checks** |"
# The hand-written branch-vs-main comparisons kittens posted by hand before `compare.py`
# existed — free-text headings, not a generated shape, so matched by their own two observed
# openings rather than by any table structure underneath.
LEGACY_COMPARISON_HEADING = re.compile(r"^## (?:Case-by-case:|`[^`]+` only —)", re.MULTILINE)

USAGE = (
    "usage: gh api repos/{owner}/{repo}/issues/<pr>/comments --paginate | "
    "python -m penny.tests.eval.utils.pr_comments <marker> <app_login> [--include-legacy]"
)
# The CLI flag that extends a marker selection with the legacy-shaped, unmarked comments a
# marker alone cannot see (there is nothing to match on a pre-#2734 comment).
INCLUDE_LEGACY_FLAG = "--include-legacy"


@dataclass(frozen=True)
class ExistingComment:
    """One comment off `gh api`'s PR-comments listing — the four fields this module's
    decisions need, read out of the GitHub REST shape (``id``, ``user.login``, ``body``,
    ``html_url``)."""

    id: int
    author_login: str
    body: str
    html_url: str = ""


def is_legacy_report(body: str) -> bool:
    """Whether ``body`` has the SHAPE of a report `eval-report` posted before #2734 — no
    marker to check, since the marker didn't exist yet. Either a case heading with its
    checks row, or one of the hand-written comparison headings; nothing else qualifies, so
    a plain markdown table or a coordinator's narrative note (quoting numbers, not a report)
    never matches."""
    if LEGACY_COMPARISON_HEADING.search(body):
        return True
    return bool(LEGACY_CASE_HEADING.search(body)) and LEGACY_CHECKS_ROW in body


def select_marked_comments(
    comments: Sequence[ExistingComment], marker: str, app_login: str
) -> list[ExistingComment]:
    """The comments ``eval-report`` may delete: carries ``marker`` AND authored by
    ``app_login`` — both, so a reviewer's own comment and a differently-marked report both
    survive untouched, in the order the listing returned them."""
    return [c for c in comments if c.author_login == app_login and marker in c.body]


def select_legacy_report_comments(
    comments: Sequence[ExistingComment], app_login: str
) -> list[ExistingComment]:
    """The App's own PRE-#2734 report comments — no marker to check, selected by
    :func:`is_legacy_report`'s shape instead. Same authorship guard as a marked selection."""
    return [c for c in comments if c.author_login == app_login and is_legacy_report(c.body)]


def select_comments_to_delete(
    comments: Sequence[ExistingComment],
    marker: str,
    app_login: str,
    *,
    include_legacy: bool = False,
) -> list[ExistingComment]:
    """The whole deletion set for one `eval-report` posting pass: every marker-matched
    comment, plus — when ``include_legacy`` — every legacy-shaped one not already counted,
    so the PR is left showing only the run about to be posted. ``include_legacy`` is the
    report pass's own choice: a comparison re-post has no legacy shape of its own to sweep."""
    marked = select_marked_comments(comments, marker, app_login)
    if not include_legacy:
        return marked
    seen = {comment.id for comment in marked}
    legacy = [c for c in select_legacy_report_comments(comments, app_login) if c.id not in seen]
    return marked + legacy


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
    include_legacy = len(argv) == 3 and argv[2] == INCLUDE_LEGACY_FLAG
    if len(argv) not in (2, 3) or (len(argv) == 3 and not include_legacy):
        print(USAGE, file=sys.stderr)
        return 2
    marker, app_login = argv[0], argv[1]
    comments = _parse_comments(sys.stdin.read())
    selected = select_comments_to_delete(comments, marker, app_login, include_legacy=include_legacy)
    for comment in selected:
        print(f"{comment.id}\t{comment.html_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
