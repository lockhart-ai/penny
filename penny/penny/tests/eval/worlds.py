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

from penny.tests.eval.fixtures import CannedPage


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
