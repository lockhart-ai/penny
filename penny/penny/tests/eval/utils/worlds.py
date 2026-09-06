"""What is TRUE while a case's ask is answered (#1995).

A **world** is the ground a round reads: the pages its tools return, the entries already
sitting in its store, the facts it is supposed to keep, and the facts the ask tells it to
leave alone.  A case declares one as a fixture and hands it to the driver; the assertions
then read the world rather than a list of tokens restated at each call site, so "she kept
what the page said" is one claim about two objects instead of a comparison somebody has to
keep in sync by hand.

A case declares ONE world.  Its samples are hermetic — own database, own conversation, own
pages — and every claim the case makes reads that world.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from penny.tests.eval.utils.fixtures import (
    AURORA_LISTING_499,
    LISTING_URL,
    TOPIC_PAGES,
    CannedPage,
    SynthCollection,
)


class SourceKind(StrEnum):
    """What one of a world's sources IS — the only thing a substrate changes about it.

    A world stands on pages a tool returns, on entries seeded into the store, or on both,
    and every reader asks the same question of all of them: what does it say, what must be
    kept from it, what does a reader open to check that.  So the substrate is a field on the
    source rather than a second list each reader has to remember to look in — which is what
    it was, and why a store-backed world rendered no ground at all (#2108)."""

    PAGE = "page"
    COLLECTION = "collection"


class WorldSource(BaseModel):
    """ONE thing a world is made of: what it is called, what kind of thing it is, what it says.

    THE definition of a world's ground.  ``World.says`` — the text a stored fact or a reply's
    value is traced back to — and ``World.render`` — the table a reader checks that trace
    against — both walk this one list, so a world cannot state one ground to a claim and a
    different one to the report."""

    model_config = ConfigDict(frozen=True)

    label: str
    kind: SourceKind
    text: str


class WorldFacts(BaseModel):
    """What a closed ground fold states about a world: how many sources of each kind, and how
    many tokens each way.

    Read off the world itself rather than counted back out of the rendered table — deriving a
    summary from the markdown it summarises is the same mistake as diffing rendered prompts."""

    model_config = ConfigDict(frozen=True)

    pages: int = 0
    collections: int = 0
    keeps: int = 0
    excludes: int = 0


class World(BaseModel):
    """One world: the sources, what must be kept from each, and what must not be kept.

    ``pages`` are the sources a tool returns and ``stores`` are the ones already in the store
    when the turn begins — the entries a case seeds into the user's own collections.  Both are
    ground the sample is GIVEN, so the driver lays both down and every reader walks them
    together as ``sources``; a world whose ground is entries rather than pages is not a world
    with no ground.

    ``keeps`` is one token set per SOURCE, in that same order — tokens that appear ONLY on that
    source, so a stored copy says which one it came from and an invented one matches neither.
    They identify the SOURCE; they are not a list of what the ask puts in scope.  That
    distinction is the whole difference between "something from the seals page was written down"
    and "the seals page's player was written down": the seals page's only item is an executive
    appointment, and a round told to collect trades and signings can read the page, correctly
    find nothing in scope, and still be right.  Requiring `volk`/`petra` failed such a round; the
    sibling case has always asked the first question and passes 4/4.

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
    stores: tuple[SynthCollection, ...] = ()

    @property
    def sources(self) -> tuple[WorldSource, ...]:
        """Everything this world is made of, pages first — the ONE list every reader walks."""
        return tuple(
            WorldSource(label=page.match, kind=SourceKind.PAGE, text=page.text)
            for page in self.pages
        ) + tuple(
            WorldSource(label=held.name, kind=SourceKind.COLLECTION, text=_holdings(held))
            for held in self.stores
        )

    @property
    def says(self) -> str:
        """Every source's text — the ground a stored fact or a reply's value is traced to."""
        return "\n".join(source.text for source in self.sources)

    @property
    def names(self) -> tuple[str, ...]:
        """Every keepable token in this world, flattened — what the closed fold counts."""
        return tuple(token for source in self.keeps for token in source)

    def render(self) -> str:
        """This world as a table: one row per SOURCE, what must be kept from it, and — once —
        what must not be kept from any of them and what the reply owes.

        A table rather than stacked prose because these are the rows an assertion reads: "she
        kept what the source said" is a comparison between a source and a token set, and putting
        them in one row is what lets a reader check it at a glance. The bodies stay openable
        underneath, since the tokens are a claim ABOUT the text and not a substitute for it.

        A page and a seeded collection render identically, in one table, in that one place — so
        a report reads the same whichever substrate the world stands on, and a reader of a
        store-backed case can see what the sample was answering against (#2108)."""
        sources = self.sources
        if not sources:
            return ""
        rows = "\n".join(
            f"| {index + 1} | {source.kind.value} `{source.label}` | "
            f"{_tokens(self._keeps_for(index))} |"
            for index, source in enumerate(sources)
        )
        bodies = "\n\n".join(self._body(index, source) for index, source in enumerate(sources))
        parts = [f"{_SOURCE_HEAD}\n{rows}"]
        if self.excludes:
            parts.append(f"{_EXCLUDES_LEAD} — {_tokens(self.excludes)}")
        if self.answers:
            parts.append(f"{_ANSWERS_LEAD} — {_tokens(self.answers)}")
        parts.append(bodies)
        return "\n\n".join(parts)

    @property
    def counts(self) -> WorldFacts:
        """What a closed fold states about this world, read off the world itself rather than
        counted back out of its rendered table."""
        return WorldFacts(
            pages=len(self.pages),
            collections=len(self.stores),
            keeps=len(self.names),
            excludes=len(self.excludes),
        )

    def _body(self, index: int, source: WorldSource) -> str:
        """One source's own text, openable under the row that makes a claim about it."""
        return (
            f"<details><summary>{source.kind.value.title()} {index + 1} — `{source.label}` · "
            f"{len(source.text):,} chars · keeps {_tokens(self._keeps_for(index)) or '—'}"
            f"</summary>\n\n```\n{source.text}\n```\n\n</details>"
        )

    def _keeps_for(self, index: int) -> tuple[str, ...]:
        """The tokens this source contributes, or empty where the case named none for it."""
        return self.keeps[index] if index < len(self.keeps) else ()


_SOURCE_HEAD = "| # | source | must be kept |\n|---|---|---|"
_EXCLUDES_LEAD = "**Must not be kept, from any source**"
# The reply-side contract, rendered beside the store-side one because it is the other half of
# what the world claims and it had no home at all before: a case whose whole point is one anchor
# token showed the ask, the claims and the numbers, and nowhere the token itself.
_ANSWERS_LEAD = "**Must be stated by the reply**"


def _holdings(held: SynthCollection) -> str:
    """One seeded collection as the store will hold it: what it is FOR, then each entry under
    the key the seeder derives for it.

    Key and content both, because an assertion about the store reads the WHOLE entry — a fact
    in the key and a blurb in the body is a perfectly good way to store it. The description is
    here because it is ground too: the ambient store map renders it on every turn, so a reply
    could state it with no call at all, and a case whose anchor leaked into a description is
    measuring nothing."""
    return "\n".join(
        [
            f"{held.name} — {held.description}",
            *(f"{key}: {content}" for key, content in held.keyed),
        ]
    )


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

# The ask names BOTH halves — which lake, and how deep — because `answers` states what the
# ask asked for, and an assertion may only require what a correct reply owes.  Asked for the
# lake alone, "It's Lake Baikal, in Siberia." is a complete answer, and requiring the depth
# of it would fail a correct run for something nobody requested.  The depth is what the case
# is FOR (it is the page's own figure, so a reply carrying it read the page rather than its
# own memory), which makes asking for it the fixture's job rather than the claim's.
DEEPEST_LAKE_ASK = "what's the deepest lake in the world, and how deep is it?"

# Four more wordings of that same question.  What varies is only how a person asks it —
# which noun opens it, "deepest" or "greatest depth", whether it is put as a plain
# question or as a request to look something up.  What does NOT vary is the pair of facts
# being asked for, the page that carries them, or the state the turn ends in.
DEEPEST_LAKE_PHRASINGS = (
    "which lake is the deepest on earth, and what depth does it reach?",
    "hey, do you know what the world's deepest lake is and how deep it goes?",
    "can you look up which lake is the deepest anywhere in the world, and how deep?",
    "i'm curious — what lake has the greatest depth of any lake, and what is that depth?",
)

# `keeps` is empty because the turn answers a question and stores nothing, so a keeps set
# would state a contract the ask never made.  `answers` is what the REPLY has to state, and
# it is not that contract read twice: both tokens are things the ask asked for, and the
# page's own figure is what says the answer came off the page rather than out of the model,
# which is the whole behaviour these cases are named for.  `baikal` is a proper noun and
# `642` is digits — both strictly identifiable, neither a phrasing.  `642` rather than
# `1,642` because the models group the digits three different ways in observed replies —
# `1,642`, `1642` and `1 642` — and the bare group is the part all three share, so the claim
# reads the value and not the formatting.  It appears nowhere else in this world, so nothing
# but the depth can satisfy it.
DEEPEST_LAKE = World(
    name="base",
    pages=TOPIC_PAGES,
    keeps=(),
    excludes=(),
    answers=("baikal", "642"),
)
