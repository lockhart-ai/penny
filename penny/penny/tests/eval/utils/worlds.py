"""What is TRUE while a case's ask is answered (#1995).

A **world** is the ground a round reads: the pages its tools return, the facts it is
supposed to keep, and the facts the ask tells it to leave alone.  A case declares one as a
fixture and hands it to the driver; the assertions then read the world rather than a list
of tokens restated at each call site, so "she kept what the page said" is one claim about
two objects instead of a comparison somebody has to keep in sync by hand.

A case declares ONE world.  Its samples are hermetic — own database, own conversation, own
pages — and every claim the case makes reads that world.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from penny.tests.eval.utils.fixtures import (
    AURORA_LISTING_499,
    LISTING_URL,
    TOPIC_PAGES,
    CannedPage,
)


class World(BaseModel):
    """One world: the pages, what must be kept from each, and what must not be kept.

    ``keeps`` is one token set per SOURCE — tokens that appear ONLY on that page, so a stored
    copy says which page it came from and an invented one matches neither.  They identify the
    SOURCE; they are not a list of what the ask puts in scope.  That distinction is the whole
    difference between "something from the seals page was written down" and "the seals page's
    player was written down": the seals page's only item is an executive appointment, and a round
    told to collect trades and signings can read the page, correctly find nothing in scope, and
    still be right.  Requiring `volk`/`petra` failed such a round; the sibling case has always
    asked the first question and passes 4/4.

    ``excludes`` are tokens that appear ONLY on a line the ask rules out, which is what makes a
    stored exclusion a read rather than a matter of taste.

    ``answers`` is ``keeps``' REPLY-SIDE counterpart, and it asks the other question.  ``keeps``
    is soundness about the store — did anything from this page get written.  ``answers`` is
    COMPLETENESS about the reply — is the thing the ask asked for actually in it.  Nothing else
    in the design asks that: a reply carrying no values at all passes every provenance claim
    vacuously, because there is nothing in it to be unsourced.

    ALL of them must appear, where ``keeps`` needs any one token per source — the two are
    different quantifiers because they answer different questions.  Tokens are chosen to be
    invariant under the model's own formatting, so the claim reads the VALUE and not its
    rendering; where a figure is grouped differently by different models, the token is the part
    they share.  An empty tuple makes no claim, which is right for an ask that has no answer to
    state — a correction, say, rather than a question.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    pages: tuple[CannedPage, ...]
    keeps: tuple[tuple[str, ...], ...]
    excludes: tuple[str, ...]
    answers: tuple[str, ...] = ()

    @property
    def says(self) -> str:
        """Every page's text — the ground a stored fact is traced to."""
        return "\n".join(page.text for page in self.pages)

    @property
    def names(self) -> tuple[str, ...]:
        """Every keepable token in this world, flattened — what the closed fold counts."""
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


# ── The two-team news world ──────────────────────────────────────────────────
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

# One trade-or-signing per page, among distractors the ask excludes in as many words (a final
# score) and ones it merely does not ask for (a training camp date, ticket renewals).  Only the
# score is an EXCLUSION: whether a training-camp date is notable is a judgement, and asserting
# it would assert one reading of the ask.
TWO_TEAM_NEWS = World(
    name="base",
    pages=(FOXES_NEWS, SEALS_NEWS),
    keeps=(("brandt", "aurelio", "goalie"), ("volk", "petra", "player development")),
    excludes=("rovers 2", "gulls 4"),
)

# ── The listing world ────────────────────────────────────────────────────────
#
# ONE source, because that is what the learn state's canonical case watches and what every
# consistently-passing learn case in the suite uses.  The page has exactly one controllable
# field, its price, so "she kept what the page said" is decidable from a single token.

AURORA_LISTING = World(
    name="base",
    pages=(AURORA_LISTING_499,),
    keeps=(("499",),),
    excludes=(),
)


# ── The canonical learn round, taken verbatim from `transition-elicit-to-learn` ───────────
#
# That case scores mean 1.0 — every scored check 3/3 — with 24/30 modal on tool sequence, and
# it is what weeks of work on the learn state produced.  The ported case takes its turns rather
# than deriving new ones, because a case that already lands where we want to land is the
# specification.
#
# PROSE, not a numbered procedure.  The numbered form appears once in the suite, in a case that
# exists to prove a round survives being written as one; every learn case that passes
# consistently uses this shape.

LISTING_SETUP_ASK = (
    f"can you watch this listing for me daily and let me know when the price changes? {LISTING_URL}"
)

# Penny's offer, which the demonstration answers turn for turn: what to read, what to look for,
# what to remember.
LISTING_TEACH_QUESTION = (
    "i don't have a routine for that yet — can you walk me through it once? "
    "what should i read, what am i looking for, what should i remember?"
)

# The demonstration itself — three discrete actions against one page.
LISTING_DEMO = f"yeah — go to {LISTING_URL}, find the current price, and remember it"

# Four more wordings of that same demonstration.  What varies is only how a person says three
# things in a sentence: which verb opens it, "current price" or "what the price is right now",
# "remember" or "keep" or "save".  What does NOT vary is the prose register, the single source,
# or the three actions — those are what the case measures enactment against.
LISTING_DEMO_PHRASINGS = (
    f"sure — open {LISTING_URL}, get the current price, and keep it",
    f"ok, head to {LISTING_URL}, check what the price is right now, and save it",
    f"yep — read {LISTING_URL}, pull the current price off it, and remember that",
    f"just visit {LISTING_URL}, note the price it's at now, and hang on to it",
)


# ── The lookup world (the recovery cases' ground) ────────────────────────────
#
# ONE page, carrying one fact with a number in it, because what a recovery case asks is
# whether the answer that finally reaches the user came off the page — and a number is a
# specific value a provenance claim can trace.  Nothing is meant to be KEPT: the turn
# answers a question, so a `keeps` token set would state a contract the ask never made.

DEEPEST_LAKE_ASK = "what's the deepest lake in the world?"

# Four more wordings of that same question.  What varies is only how a person asks it —
# which noun opens it, "deepest" or "greatest depth", whether it is put as a plain
# question or as a request to look something up.  What does NOT vary is the fact being
# asked for, the page that carries it, or the state the turn ends in.
DEEPEST_LAKE_PHRASINGS = (
    "which lake is the deepest on earth?",
    "hey, do you know what the world's deepest lake is?",
    "can you look up which lake is the deepest anywhere in the world?",
    "i'm curious — what lake has the greatest depth of any lake?",
)

# `keeps` is empty because the turn answers a question and stores nothing, so a keeps set
# would state a contract the ask never made.  `answers` is the question the REPLY has to
# answer, and it is not that contract read twice: the page's own figure is what says the
# answer came off the page rather than out of the model, which is the whole behaviour these
# cases are named for.  `642` rather than `1,642` because the models group the digits three
# different ways in observed replies — `1,642`, `1642` and `1 642` — and the bare group is
# the part all three share, so the claim reads the value and not the formatting.
DEEPEST_LAKE = World(
    name="base",
    pages=TOPIC_PAGES,
    keeps=(),
    excludes=(),
    answers=("baikal", "642"),
)
