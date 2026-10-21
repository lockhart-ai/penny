"""Tests for the shared similarity and dedup module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from similarity.dedup import DedupStrategy, JobSide, is_embedding_duplicate, is_same_job
from similarity.identity import (
    KeyRule,
    ShortlistSide,
    key_parts,
    normalise_key,
    number_tokens,
    pair_rule_verdict,
    shortlist,
)

from penny.llm.embeddings import serialize_embedding
from penny.llm.models import LlmResponseError
from penny.llm.similarity import embed_text

# ── embed_text ────────────────────────────────────────────────────────────────


class TestEmbedText:
    @pytest.mark.asyncio
    async def test_returns_vector_on_success(self) -> None:
        client = AsyncMock()
        client.embed.return_value = [[1.0, 2.0, 3.0]]
        result = await embed_text(client, "hello")
        assert result == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self) -> None:
        client = AsyncMock()
        client.embed.side_effect = LlmResponseError("boom")
        result = await embed_text(client, "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_llm_exception_propagates(self) -> None:
        """Bugs in the embed pipeline must surface, not be swallowed as None."""
        client = AsyncMock()
        client.embed.side_effect = RuntimeError("unexpected bug")
        with pytest.raises(RuntimeError, match="unexpected bug"):
            await embed_text(client, "hello")


# ── is_embedding_duplicate ────────────────────────────────────────────────────


def _make_item(name: str, vec: list[float] | None) -> tuple[str, bytes | None]:
    """Helper to build (name, serialized_embedding) tuple."""
    return (name, serialize_embedding(vec) if vec else None)


class TestIsEmbeddingDuplicate:
    def test_none_candidate_vec_returns_none(self) -> None:
        items = [_make_item("foo", [1.0, 0.0])]
        result = is_embedding_duplicate("foo", None, items, DedupStrategy.EMBEDDING_ONLY, 0.8)
        assert result is None

    def test_embedding_only_match(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("different name", vec)]
        result = is_embedding_duplicate("candidate", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9)
        assert result == 0

    def test_embedding_only_no_match(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        items = [_make_item("other", b)]
        result = is_embedding_duplicate("candidate", a, items, DedupStrategy.EMBEDDING_ONLY, 0.5)
        assert result is None

    def test_tcr_and_embedding_both_pass(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("star trek voyager", vec)]
        result = is_embedding_duplicate(
            "star trek", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_and_embedding_tcr_fails(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("completely different", vec)]
        result = is_embedding_duplicate(
            "star trek", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result is None

    def test_tcr_and_embedding_single_token_bypass(self) -> None:
        """Single-token names skip TCR requirement (e.g. acronyms)."""
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("clps", vec)]
        result = is_embedding_duplicate(
            "foo", vec, items, DedupStrategy.TCR_AND_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_tcr_only(self) -> None:
        """TCR passes but no embedding available — still a match in OR mode."""
        items: list[tuple[str, bytes | None]] = [("star trek voyager", None)]
        result = is_embedding_duplicate(
            "star trek", [1.0, 0.0], items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_no_candidate_vec_still_matches_tcr(self) -> None:
        """TCR passes with None candidate_vec — still a match in OR mode."""
        items: list[tuple[str, bytes | None]] = [("star trek voyager", None)]
        result = is_embedding_duplicate(
            "star trek", None, items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_tcr_or_embedding_embedding_only(self) -> None:
        """TCR fails but embedding passes — still a match in OR mode."""
        vec = [1.0, 0.0, 0.0]
        items = [_make_item("completely different", vec)]
        result = is_embedding_duplicate(
            "something else", vec, items, DedupStrategy.TCR_OR_EMBEDDING, 0.9, 0.6
        )
        assert result == 0

    def test_returns_first_match_index(self) -> None:
        vec = [1.0, 0.0, 0.0]
        items = [
            _make_item("no match", [0.0, 1.0, 0.0]),
            _make_item("match", vec),
        ]
        result = is_embedding_duplicate("candidate", vec, items, DedupStrategy.EMBEDDING_ONLY, 0.9)
        assert result == 1


# ── is_same_job — the labelled corpus (#1775) ────────────────────────────────
#
# The deterministic fixture the #1775 strategy was picked against, transposed onto
# the synthetic ``aurora-*`` / ``harbor-*`` families: every pair the corpus labels a
# DUPLICATE (the same one thing named two ways) and every pair it labels DISTINCT
# (the adversarial negatives — same subject, genuinely different thing).  Both
# directions are pinned, because a rule that only proves duplicates are caught is
# half a rule: the failure this fixture exists to prevent is a merge, not a miss.

# Labelled DUPLICATES — one job, two names.  Each line is the axis it forks on.
_DUPLICATE_PAIRS = [
    ("aurora-deck-2-price", "aurora-deck-2-prices"),  # plural
    ("aurora-deck-2-price", "aurora_deck_2_price"),  # underscore separators
    ("aurora-deck-2-price", "Aurora Deck 2 Price"),  # case + whitespace
    ("aurora-deck-2-price", "aurora‑deck‑2‑price"),  # unicode hyphens
    ("aurora-deck-2-price", "aurora-deck-2"),  # dropped qualifier
    ("aurora-deck-2-price", "aurora-deck-2-monitor"),  # role affix
    ("aurora-deck-2-price", "aurora-deck-2-price-watcher"),  # role affix
    ("aurora-deck-2-price", "aurora-deck-2-price-tracker"),  # role affix
    ("aurora-deck-2-price", "aurora-deck-2-price-alerts"),  # role affix, plural
    ("aurora-deck-2-price", "aurora-deck-2-price-latest"),  # role affix
    ("aurora-deck-2-price", "aurora-prices"),  # catch-all vs. the one scoped from it
    ("aurora-deck-2-price", "aurora-price"),
    ("aurora-deck-2-price", "aurora-deck-price"),
    ("aurora-deck-2-price", "aurora-deck-prices"),
    ("aurora-deck-2-price", "aurora-decks"),
    ("aurora-deck-2-price", "aurora-decks-prices"),
    ("harbor-ferry-times", "harbor-ferry-time-monitor"),  # role affix + plural
]

# Labelled DISTINCT — the adversarial negatives.  Every one is a SUBSTITUTION: each
# side contributes a token the other lacks, and that token is the one carrying the
# meaning, which is exactly why no name-similarity threshold can separate them from
# a real duplicate and why containment (a set relation) can.
_DISTINCT_PAIRS = [
    ("aurora-deck-2-price", "aurora-deck-3-price"),  # different listing
    ("aurora-deck-2-price", "aurora-deck-2-condition"),  # different fact
    ("aurora-deck-2-price", "aurora-deck-2-seller-rating"),  # different fact
    ("aurora-deck-2-price", "boreal-deck-2-price"),  # different subject
    ("harbor-ferry-times", "harbor-ferry-fares"),  # different fact
    ("harbor-ferry-times", "harbor-bus-times"),  # different subject
    ("harbor-news-northwind", "harbor-news-southgale"),  # different source
    ("aurora-bids-notified", "aurora-bids-unnotified"),  # opposite halves of a pair
    ("aurora-deck-2-price", "harbor-ferry-times"),  # unrelated
    ("aurora-deck-2-price", "monitor"),  # a bare role word names no job
]

# Labelled duplicates the containment tier does NOT catch: one side says ``auction``
# where the other says ``price``, so it reads as a substitution.  Tier 3
# (near-containment + same-day + param overlap → ASK) is what reaches these, and it
# is deliberately out of scope for #1775 — pinned here so the gap is a recorded
# measurement rather than a surprise.
_KNOWN_MISSES = [
    ("aurora-deck-2-price", "aurora-deck-2-auction-watch"),
    ("aurora-deck-2-price", "aurora-deck-2-auction-watcher"),
]


def _same(name_a: str, name_b: str) -> bool:
    """``is_same_job`` on two bare names."""
    return is_same_job(JobSide(name=name_a), JobSide(name=name_b))


class TestIsSameJob:
    """The job-identity rule (name containment) scored against the labelled corpus."""

    @pytest.mark.parametrize(("name_a", "name_b"), _DUPLICATE_PAIRS)
    def test_labelled_duplicates_are_caught(self, name_a: str, name_b: str) -> None:
        assert _same(name_a, name_b)
        assert _same(name_b, name_a)  # symmetric — order of arrival can't change it

    @pytest.mark.parametrize(("name_a", "name_b"), _DISTINCT_PAIRS)
    def test_labelled_negatives_are_never_merged(self, name_a: str, name_b: str) -> None:
        assert not _same(name_a, name_b)
        assert not _same(name_b, name_a)

    @pytest.mark.parametrize(("name_a", "name_b"), _KNOWN_MISSES)
    def test_substitution_duplicates_are_the_known_gap(self, name_a: str, name_b: str) -> None:
        """Recorded, not asserted-as-correct: these ARE duplicates, and tiers 1-2
        miss them.  Catching them is tier 3's job (ask, never auto-merge)."""
        assert not _same(name_a, name_b)

    def test_corpus_score(self) -> None:
        """The headline measurement: every negative held, and the misses are exactly
        the two substitution-shaped duplicates — so a regression in either direction
        shows up as a number, not as one parametrised line going red."""
        caught = sum(1 for pair in _DUPLICATE_PAIRS + _KNOWN_MISSES if _same(*pair))
        false_merges = sum(1 for pair in _DISTINCT_PAIRS if _same(*pair))
        assert (caught, false_merges) == (17, 0)


# ── Entry-identity rules (#2715) ─────────────────────────────────────────────
#
# The deterministic steps of the identity ladder, over the 26 hand-labelled
# contrived pairs the prototype measured: every pair the rules call SAME is one
# thing named two ways, every pair they call DIFFERENT differs in a number
# token, and everything else is left undecided for the shortlist + decision
# model.  All fixtures are invented scenarios with real proper nouns.

# Pairs whose KEYS normalise to the same entry — the rules must say SAME
# whatever the content says (the value question is the caller's).
_RULES_SAME_KEYS = [
    ("Pink Floyd (band)", "pink floyd"),
    ("The Beatles", "beatles"),
    ("Twin Peaks (1990 series)", "twin peaks"),
    ("Zoë Kravitz", "zoe kravitz"),
    ("philip k. dick", "Philip K. Dick"),
    ("Into the Breach (video game)", "into the breach"),
    ("analogue pocket price", "analogue_pocket_price"),
    ("sea kayaking", "sea kayaking"),
]

# Pairs whose keys differ in a NUMBER token — the rules must say DIFFERENT even
# when the content is identical: a date key is what makes two digest entries
# different, and VI is not VII.
_RULES_DIFFERENT_KEYS = [
    ("2026-10-19", "2026-10-20"),
    ("2026-10-21", "2026-10-20"),
    ("final fantasy vi", "final fantasy vii"),
    ("blade runner", "blade runner 2049"),
    ("solaris", "solaris 2002"),
    ("the big o", "the big o season 2"),
    ("The Big O II (season 2)", "the big o"),
]

# Pairs the rules cannot settle — same neither way — left for the shortlist
# and the decision model.
_RULES_UNSETTLED_KEYS = [
    ("genesis", "genesis p-orridge"),
    ("pink floyd", "pink"),
    ("portishead", "portugal. the man"),
    ("the shining", "the shining girls"),
    ("the hobbit", "the lord of the rings"),
    ("instant coffee", "decaf coffee"),
    ("boston celtics", "boston red sox"),
    ("godspeed you! black emperor", "gybe"),
    ("sea kayaking", "kayaking on the ocean"),
    ("nyc", "New York City"),
    ("dune", "dune messiah"),
    ("olympia sm3", "olympia sm3 typewriter"),
    ("nyc", "new york city metro area"),
]


class TestIdentityRules:
    @pytest.mark.parametrize(("candidate", "existing"), _RULES_SAME_KEYS)
    def test_keys_equal_after_normalising_are_the_same_entry(
        self, candidate: str, existing: str
    ) -> None:
        assert pair_rule_verdict(candidate, existing, "x", "y") is KeyRule.SAME

    @pytest.mark.parametrize(("candidate", "existing"), _RULES_DIFFERENT_KEYS)
    def test_keys_differing_in_a_number_token_are_different_entries(
        self, candidate: str, existing: str
    ) -> None:
        # Decisive even when the CONTENT is identical: two digest entries for
        # consecutive days are different entries, not a re-observation.
        assert pair_rule_verdict(candidate, existing, "same body", "same body") is (
            KeyRule.DIFFERENT
        )

    @pytest.mark.parametrize(("candidate", "existing"), _RULES_UNSETTLED_KEYS)
    def test_look_alike_pairs_are_left_to_the_decision_model(
        self, candidate: str, existing: str
    ) -> None:
        assert pair_rule_verdict(candidate, existing, "x", "y") is None

    @pytest.mark.parametrize(
        ("candidate", "existing"),
        [
            ("Solaris (1972)", "Solaris (2002)"),
            ("Dune (1984)", "Dune (2021)"),
            ("Ghost in the Shell (1995)", "Ghost in the Shell (2007)"),
        ],
    )
    def test_parentheticals_that_both_carry_numbers_and_differ_are_different_entries(
        self, candidate: str, existing: str
    ) -> None:
        # The dangerous direction the parenthetical split exists for: the OLD
        # number rule ran after the parenthetical was stripped, so two films
        # keyed by year normalised to the same entry and MERGED.
        assert pair_rule_verdict(candidate, existing, "x", "y") is KeyRule.DIFFERENT

    def test_a_number_in_only_one_parenthetical_does_not_decide(self) -> None:
        # A qualifier on one side is not an identity: the normalised comparison
        # still runs and the pair is the same entry.
        assert pair_rule_verdict("Twin Peaks", "Twin Peaks (1990 series)", "x", "y") is (
            KeyRule.SAME
        )
        assert pair_rule_verdict("Solaris", "Solaris (1972)", "x", "y") is (KeyRule.SAME)

    def test_vitamin_b6_normalises_to_the_same_key(self) -> None:
        # A digit inside a word is a main-part token, and equal on both sides.
        assert pair_rule_verdict("vitamin-B6", "Vitamin B6", "x", "y") is KeyRule.SAME

    def test_identical_content_settles_nothing(self) -> None:
        # The repeat rule retired: identical content under different keys is the
        # decision model's question, which sees both keys — a deterministic
        # repeat would block two products that happen to share a price.
        assert pair_rule_verdict("switch lite", "analogue pocket", "$219", "$219") is None

    def test_the_number_token_rule_beats_identical_content(self) -> None:
        # The production regression: a daily-digest job could not write
        # 2026-10-21 because the store said it was the same entry as 2026-10-20.
        assert (
            pair_rule_verdict("2026-10-21", "2026-10-20", "same headlines", "same headlines")
            is KeyRule.DIFFERENT
        )

    def test_normalise_key_strips_framing(self) -> None:
        assert normalise_key("Pink Floyd (band)") == "pink floyd"
        assert normalise_key("[dune]") == "dune"
        assert normalise_key("(dune)") == "dune"  # a whole-paren key unwraps
        assert normalise_key("The_Big-O") == "big o"
        assert normalise_key("portugal. the man!") == "portugal the man"

    def test_number_tokens_read_dates_and_roman_numerals(self) -> None:
        assert number_tokens("2026-10-21") == ["10", "2026", "21"]
        assert number_tokens("final fantasy vii") == ["vii"]
        assert number_tokens("dune messiah") == []
        # A token that merely looks Roman is not read as a number — the round
        # trip back to the canonical form rejects it.
        assert number_tokens("vim editor") == []
        assert number_tokens("im planning") == []
        # The main part and the parenthetical are read separately.
        main, paren = key_parts("The Big O II (season 2)")
        assert number_tokens(main) == ["ii"]
        assert paren is not None and number_tokens(paren) == ["2"]

    def test_key_parts_split_the_trailing_parenthetical(self) -> None:
        assert key_parts("Solaris (1972)") == ("solaris", "1972")
        assert key_parts("Dune") == ("dune", None)
        assert key_parts("[dune (2021)]") == ("dune", "2021")

    def test_shortlist_ranks_by_the_best_surface_and_takes_the_top_k(self) -> None:
        existing = [
            ShortlistSide(key_vec=[1.0, 0.0], content_vec=[0.0, 1.0]),
            ShortlistSide(key_vec=None, content_vec=[1.0, 0.0]),
            ShortlistSide(key_vec=[0.0, 1.0], content_vec=None),
            ShortlistSide(key_vec=None, content_vec=None),
        ]
        # The candidate's key matches side 0's key exactly; its content matches
        # side 1's content — the best surface wins, ranking only.
        ranked = shortlist([1.0, 0.0], [1.0, 0.0], existing, k=3)
        assert ranked[0] == 0
        # Side 3 has no vectors to rank on, so it never shortlists — and the
        # shortlist is the top k of those that can rank, no cutoff.
        assert len(ranked) == 3
        assert 3 not in ranked

    def test_shortlist_of_a_vectorless_candidate_is_empty(self) -> None:
        existing = [ShortlistSide(key_vec=[1.0, 0.0], content_vec=None)]
        assert shortlist(None, None, existing, k=3) == []
