"""What is TRUE while a case's ask is answered (#1995).

A **world** is the ground a round reads: the pages its tools return, the facts it is
supposed to keep, and the facts the ask tells it to leave alone.  A case declares one as a
fixture and hands it to the driver; the assertions then read the world rather than a list
of tokens restated at each call site, so "she kept what the page said" is one claim about
two objects instead of a comparison somebody has to keep in sync by hand.

The second world a case declares is its **control** — the same ask against different facts.
That is what makes "she read the page" decidable instead of assumed, and it is a different
mechanism from paraphrasing the ask: if Penny were pattern-completing from the shape of the
request, every phrasing would name the same player and every phrasing would be right.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from penny.tests.eval.utils.fixtures import CannedPage


class World(BaseModel):
    """One world: the pages, what must be kept from each, and what must not be kept.

    ``keeps`` is one token set per SOURCE — the names that appear only on that page's
    keepable line, so a stored copy says which page it came from and an invented one matches
    neither.  ``excludes`` are tokens that appear ONLY on a line the ask rules out, which is
    what makes a stored exclusion a read rather than a matter of taste.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    pages: tuple[CannedPage, ...]
    keeps: tuple[tuple[str, ...], ...]
    excludes: tuple[str, ...]

    @property
    def says(self) -> str:
        """Every page's text — the ground a stored fact is traced to."""
        return "\n".join(page.text for page in self.pages)

    @property
    def names(self) -> tuple[str, ...]:
        """Every keepable name in this world, flattened — what a reply drawing on this world
        will carry, and what a reply drawing on another world must not."""
        return tuple(token for source in self.keeps for token in source)

    def render(self) -> str:
        """This world as a table: one row per page, what must be kept from it, and — once — what
        must not be kept from any of them.

        A table rather than stacked prose because these are the rows an assertion reads: "she
        kept what the page said" is a comparison between a page and a token set, and putting them
        in one row is what lets a reader check it at a glance. The page bodies stay openable
        underneath, since the tokens are a claim ABOUT the text and not a substitute for it."""
        if not self.pages:
            return ""
        rows = "\n".join(
            f"| {index + 1} | `{page.match}` | {_tokens(self._keeps_for(index))} |"
            for index, page in enumerate(self.pages)
        )
        bodies = "\n\n".join(
            f"<details><summary>Page {index + 1} — `{page.match}` · {len(page.text):,} chars · "
            f"keeps {_tokens(self._keeps_for(index)) or '—'}</summary>"
            f"\n\n```\n{page.text}\n```\n\n</details>"
            for index, page in enumerate(self.pages)
        )
        parts = [f"{_PAGE_HEAD}\n{rows}"]
        if self.excludes:
            parts.append(f"**Must not be kept, from any page** — {_tokens(self.excludes)}")
        parts.append(bodies)
        return "\n\n".join(parts)

    @property
    def counts(self) -> tuple[int, int, int]:
        """``(pages, must-keep tokens, must-not tokens)`` — what a closed fold states about this
        world, read off the world itself rather than counted back out of its rendered table."""
        return len(self.pages), len(self.names), len(self.excludes)

    def _keeps_for(self, index: int) -> tuple[str, ...]:
        """The tokens this page contributes, or empty where the case named none for it."""
        return self.keeps[index] if index < len(self.keeps) else ()


_PAGE_HEAD = "| # | page | must be kept |\n|---|---|---|"


def _tokens(tokens: tuple[str, ...]) -> str:
    return ", ".join(f"`{token}`" for token in tokens)


# ── The two-team news world, and its control ─────────────────────────────────
#
# Declared here rather than in the case that first needed it: tokens, excluded content and
# pages describe the WORLD, not the case, so a second case reading these pages inherits them
# instead of restating them and drifting.

FOXES_URL = "https://www.ridgelinefoxes.com/news"
SEALS_URL = "https://www.harborseals.com/news"

FOXES_NEWS = CannedPage(
    match="ridgelinefoxes",
    text=(
        "Title: Ridgeline Foxes | Official Site — Team News\n"
        f"{FOXES_URL}\n\n"
        "Foxes sign veteran goalie Aurelio Brandt to a two-year deal — the club "
        "confirmed the signing Thursday morning.\n"
        "Final score: Foxes 3, Rovers 2 (overtime).\n"
        "Training camp opens next month at Ridgeline Arena.\n"
    ),
)

SEALS_NEWS = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor Seals | Official Site — Team News\n"
        f"{SEALS_URL}\n\n"
        "Seals name Petra Volk head of player development after a lengthy search.\n"
        "Final score: Seals 1, Gulls 4.\n"
        "Season ticket renewals open Friday.\n"
    ),
)

FOXES_NEWS_CONTROL = CannedPage(
    match="ridgelinefoxes",
    text=(
        "Title: Ridgeline Foxes | Official Site — Team News\n"
        f"{FOXES_URL}\n\n"
        "Foxes trade defenceman Wilhelmina Roux to the Rovers for a third-round "
        "pick — the deal was filed Tuesday afternoon.\n"
        "Final score: Foxes 1, Rovers 5.\n"
        "Training camp opens next month at Ridgeline Arena.\n"
    ),
)

SEALS_NEWS_CONTROL = CannedPage(
    match="harborseals",
    text=(
        "Title: Harbor Seals | Official Site — Team News\n"
        f"{SEALS_URL}\n\n"
        "Seals sign winger Casimir Oyelaran to a one-year contract ahead of the "
        "autumn window.\n"
        "Final score: Seals 6, Gulls 2.\n"
        "Season ticket renewals open Friday.\n"
    ),
)

# One trade-or-signing per page, among distractors the ask excludes in as many words (a final
# score) and ones it merely does not ask for (a training camp date, ticket renewals).  Only the
# score is an EXCLUSION: whether a training-camp date is notable is a judgement, and asserting
# it would assert one reading of the ask.
TWO_TEAM_NEWS = World(
    name="base",
    pages=(FOXES_NEWS, SEALS_NEWS),
    keeps=(("brandt", "aurelio"), ("volk", "petra")),
    excludes=("rovers 2", "gulls 4"),
)

# The same world with every proper noun and every fact replaced and nothing else changed.  The
# excluded tokens differ too, so a stored score is still decidable in either world.
TWO_TEAM_NEWS_CONTROL = World(
    name="control",
    pages=(FOXES_NEWS_CONTROL, SEALS_NEWS_CONTROL),
    keeps=(("roux", "wilhelmina"), ("oyelaran", "casimir")),
    excludes=("rovers 5", "gulls 2"),
)
