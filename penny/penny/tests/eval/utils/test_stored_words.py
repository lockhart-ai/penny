"""The share of stored words that stand nowhere in what the round was given (#2199).

Deterministic and model-free: the reading over hand-built entries and worlds, what it is
blind to, and that measuring it moves no claim, no standing, no ceiling and no case colour."""

from __future__ import annotations

import pytest

from penny.tests.eval.conftest import _cohort_checks, _share_records
from penny.tests.eval.utils import cohort, report
from penny.tests.eval.utils.artifacts import CaseTimings, build_case_artifact
from penny.tests.eval.utils.assertions import Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    TOOL_SEQUENCE,
    Given,
    SampleObservation,
    StoredEntry,
)
from penny.tests.eval.utils.given import read_given
from penny.tests.eval.utils.stored_words import (
    OPEN_ABOVE,
    STOP_WORDS,
    STORED_WORDS_FOUND_NOWHERE,
    content_words,
    same_word,
    stored_words_reading,
)

_MODEL = "openai/gpt-oss-20b"
_NO_TIMINGS = CaseTimings(calls=0, duration_ms=0, input_tokens=0, output_tokens=0)
_KEY = "Lemon barley soup"
_SEEDED = "Lemon barley soup — barley, lemon, kale, 30 min simmer."
_ASK = "add a 5-minute toasted seed topping to my soup recipe"

# The round as the prompt log records it: the user's ask, a read that laid the store out, and
# the write's own result.  Read through the shipped reader, so the world these tests weigh an
# entry against is the content ``given.py`` draws and not a string assembled beside it.
_ROUND = read_given(
    [
        {"role": "system", "content": "You are Penny. Garnish every answer with paprika."},
        {"role": "user", "content": _ASK},
        {
            "role": "tool",
            "content": f"1 entry from `recipes` (newest first):\n"
            f"1. [2026-10-14 18:00 UTC] key='{_KEY}' {_SEEDED}",
        },
        {"role": "tool", "content": f"Updated '{_KEY}' in 'recipes'."},
    ]
)

_RESTATED = (
    "Lemon barley soup — barley, lemon, kale, 30 min simmer, "
    "5\u2011minute toasted\u2011seed topping."
)
_INVENTED = (
    "Lemon barley soup — barley, lemon, kale, 30 min simmer. Topping: toast 2 tbsp pumpkin "
    "seeds with 1 tsp smoked paprika, a pinch of salt, drizzle olive oil."
)


def _entry(content: str, key: str | None = _KEY) -> StoredEntry:
    return StoredEntry(collection="recipes", key=key, content=content)


def _sample(name: str, *contents: str, complete: bool = True) -> SampleObservation:
    return SampleObservation(
        name=name,
        phrasing="phrasing 1",
        complete=complete,
        landed="idle",
        tool_sequence=["update_entry"] if contents else [],
        entries=[_entry(content) for content in contents],
        given=_ROUND,
    )


def test_an_entry_restating_the_seed_and_the_users_words_reads_zero() -> None:
    """Every content word of the restated entry is a word the round was given — the seeded
    entry's, or the user's — so nothing in it stands nowhere.  Written with the drawn hyphen
    the model prefers, which the fold reads as the hyphen it is."""
    reading = stored_words_reading([_entry(_RESTATED)], _ROUND)

    assert reading is not None
    assert (reading.value, reading.evidence) == (0.0, [])


def test_an_invented_ingredient_list_reads_clearly_higher_and_names_its_words() -> None:
    """The same edit with a list of ingredients nobody gave.  The reading is the fraction, and
    the evidence is the words, in the order they were written, each once.

    ``paprika`` stands in the SYSTEM prompt and nowhere in the round's content, so it is found
    nowhere: a prompt's own words are scaffolding, exactly as they are for a figure."""
    reading = stored_words_reading([_entry(_INVENTED)], _ROUND)

    assert reading is not None
    assert reading.evidence == [
        "tbsp",
        "pumpkin",
        "tsp",
        "smoked",
        "paprika",
        "pinch",
        "salt",
        "drizzle",
        "olive",
        "oil",
    ]
    # Ten of the nineteen distinct content words the entry is written in.  `toast` and `seeds`
    # are not among them: the user said `toasted` and `seed`.
    assert reading.value == pytest.approx(10 / 19)
    assert reading.value > OPEN_ABOVE > 0.0


def test_a_sample_that_wrote_nothing_has_no_reading() -> None:
    """No entry is nothing to take a fraction of, which is ``None`` and never ``0.0``: zero is
    the best reading this axis has.  The same holds for an entry with no content word in it —
    function words and a figure are not a denominator."""
    assert stored_words_reading([], _ROUND) is None
    assert STORED_WORDS_FOUND_NOWHERE.read(_sample("wrote-nothing")) is None
    assert stored_words_reading([_entry("It is at 30.", key=None)], _ROUND) is None


def test_function_words_figures_and_single_letters_are_in_neither_half() -> None:
    """The closed stop list, a contraction, a figure and a unit's lone letter are left out of
    the numerator AND the denominator, so wrapping given words in grammar moves nothing."""
    assert content_words("It's the soup that we don't simmer for 30 min at 425F.") == [
        "soup",
        "simmer",
        "min",
    ]
    assert {"the", "and", "of", "not", "would", "their"} <= STOP_WORDS
    wrapped = "And then it is the kale that would be in the soup, for 30 min."
    assert stored_words_reading([_entry(wrapped, key=None)], _ROUND) == cohort.ShareReading(
        value=0.0
    )


def test_each_word_counts_once_however_often_it_was_written() -> None:
    """DISTINCT words: an entry repeating one given word reads no lower, and one repeating an
    invented word reads no higher."""
    once = stored_words_reading([_entry("kale saffron", key=None)], _ROUND)
    padded = stored_words_reading([_entry("kale kale kale saffron saffron", key=None)], _ROUND)

    assert once == padded == cohort.ShareReading(value=0.5, evidence=["saffron"])


def test_a_word_is_read_through_the_folding_the_claims_use() -> None:
    """Case, typography, a possessive and a hyphen do not make a given word a new one — in
    either direction a hyphen can differ."""
    world = Given("", content="Silverleaf moss from the Harbor Seals page. A slow-cooker stew.")
    folded = "SILVER‑LEAF moss, the Seals’ page, a slowcooker stew"

    assert stored_words_reading([_entry(folded, key=None)], world) == cohort.ShareReading(value=0.0)
    # A different word is still a different word.
    other = stored_words_reading([_entry("silver leaves", key=None)], world)
    assert other is not None and other.evidence == ["silver", "leaves"]


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("pepper", "peppers"),
        ("tomato", "tomatoes"),
        ("marinate", "marinated"),
        ("marinate", "marinating"),
        ("store", "storing"),
        ("cook", "cooking"),
        # Two forms of one word, neither of them the bare one.
        ("signing", "signed"),
        ("marinated", "marinating"),
    ],
)
def test_a_word_with_another_ending_is_the_same_word(one: str, other: str) -> None:
    assert same_word(one, other) and same_word(other, one)


@pytest.mark.parametrize(
    ("one", "other"),
    [
        # Two words that merely start alike.
        ("chicken", "chickpea"),
        ("garlic", "garland"),
        ("sign", "signal"),
        ("lime", "limit"),
        # A short word is itself or nothing: under four letters there is no stem to share.
        ("min", "mins"),
        ("new", "news"),
        ("add", "added"),
        # THE STATED MISSES — a different word for a given thing, and a form the ending rule
        # does not reach.  Each reads as found nowhere.
        ("marinate", "marinade"),
        ("bake", "baking"),
        ("chop", "chopped"),
        ("berry", "berries"),
    ],
)
def test_a_different_word_is_not_found_by_starting_alike(one: str, other: str) -> None:
    assert not same_word(one, other) and not same_word(other, one)


def test_what_the_reading_cannot_see_is_stated_and_pinned() -> None:
    """The blind spots, each as the reading it produces — all of them LOW.

    A note saying something false in words the round was given reads zero: this measures the
    words, never what they say.  A key the turn invented is named by its own write result,
    which is content.  And a different word that is a given one plus an ending is found."""
    false_note = "No barley, lemon or kale in my soup recipe."
    assert stored_words_reading([_entry(false_note, key=None)], _ROUND) == cohort.ShareReading(
        value=0.0
    )

    echoed = read_given(
        [
            {"role": "user", "content": "save my soup"},
            {"role": "tool", "content": "Wrote 1 entry to 'recipes': saffron-broth."},
        ]
    )
    invented_key = stored_words_reading([_entry("soup", key="saffron-broth")], echoed)
    assert invented_key == cohort.ShareReading(value=0.0)

    assert same_word("plan", "plane") and same_word("even", "evening")


# ── Across a cohort ──────────────────────────────────────────────────────────


def _cohort(*samples: SampleObservation) -> Cohort:
    made = Cohort("recipe-edit", _MODEL, list(samples))
    made.assert_the_store_holds_an_entry()
    made.assert_every_value_in_the_store_is_sourced()
    return made


def test_a_cohort_reports_the_distribution_over_the_samples_that_wrote() -> None:
    """Median, range and the samples above the stated share — over the samples that HAD a
    reading.  A sample that wrote nothing and a sample that never ran are in neither."""
    samples = [
        _sample("recipe-edit-1", _RESTATED),
        _sample("recipe-edit-2", _INVENTED),
        _sample("recipe-edit-3", _RESTATED),
        _sample("recipe-edit-4"),
        _sample("recipe-edit-5", _INVENTED, complete=False),
    ]
    spread = cohort.pool(samples, [], [STORED_WORDS_FOUND_NOWHERE]).shares[0]

    assert (spread.n, len(spread.readings), spread.blind) == (4, 3, False)
    assert (spread.median, spread.low, spread.high) == (0.0, 0.0, pytest.approx(10 / 19))
    assert [reading.sample for reading in spread.above] == ["recipe-edit-2"]

    # The record carries one value per sample the case drove, in the order it drove them.
    record = _share_records(samples, [STORED_WORDS_FOUND_NOWHERE])[0]
    assert record.name == "stored words found nowhere"
    assert record.values == [0.0, pytest.approx(10 / 19), 0.0, None, None]
    assert (record.open_above, record.blind) == (OPEN_ABOVE, False)


def test_a_cohort_in_which_nobody_wrote_is_blind_not_at_zero() -> None:
    """Every pooled sample read nothing, so there is no distribution: it is blind, and it
    renders red saying so, never as a row of zeros that would read as a clean cohort.  An
    empty pool is not blind — nothing was pooled, and the excluded samples say why."""
    quiet = [_sample(f"recipe-edit-{n}") for n in (1, 2, 3)]
    spread = cohort.pool(quiet, [], [STORED_WORDS_FOUND_NOWHERE]).shares[0]

    assert spread.blind and spread.readings == []
    assert _share_records(quiet, [STORED_WORDS_FOUND_NOWHERE])[0].blind

    dead = [_sample("recipe-edit-1", _INVENTED, complete=False)]
    assert not cohort.pool(dead, [], [STORED_WORDS_FOUND_NOWHERE]).shares[0].blind


def test_measuring_the_share_moves_no_claim_no_standing_no_ceiling_and_no_colour() -> None:
    """MEASURED, never gated.  A cohort that measures the share answers every claim, pools
    every feature, proposes every ceiling, ranks every sample and colours its case exactly as
    the same cohort does without it — including when the share itself is at its worst."""
    samples = [
        _sample("recipe-edit-1", _INVENTED),
        _sample("recipe-edit-2", _INVENTED),
        _sample("recipe-edit-3", _RESTATED),
    ]
    plain, measured = _cohort(*samples), _cohort(*samples)
    plain.measure(TOOL_SEQUENCE, ENTRIES_STORED)
    measured.measure(TOOL_SEQUENCE, ENTRIES_STORED, STORED_WORDS_FOUND_NOWHERE)

    assert measured.features == plain.features, "a share is never one of the features"
    assert measured.shares == [STORED_WORDS_FOUND_NOWHERE]
    assert measured.claims == plain.claims
    assert _cohort_checks(measured) == _cohort_checks(plain)

    without = cohort.pool(samples, plain.features, plain.shares)
    with_share = cohort.pool(samples, measured.features, measured.shares)
    assert with_share.features == without.features
    assert cohort.variance_headline(with_share.features) == cohort.variance_headline(
        without.features
    )
    assert cohort.standings(samples, measured.features) == cohort.standings(samples, plain.features)
    assert with_share.shares[0].high > OPEN_ABOVE, "the share is high, and nothing moved"

    sections = [
        report.CaseSections(case_id="recipe-edit", model=_MODEL, variance=variance)
        for variance in (without, with_share)
    ]
    assert sections[0].glyph() == sections[1].glyph()
    assert sections[0].measures() == sections[1].measures()

    # And the record: the same scores, the same gate, with the share riding beside them.
    artifacts = [
        build_case_artifact(
            run_id="run",
            case_id="recipe-edit",
            family="memory",
            results=[],
            timings=_NO_TIMINGS,
            shares=_share_records(samples, shares),
        )
        for shares in (plain.shares, measured.shares)
    ]
    assert artifacts[0].model_dump(exclude={"shares"}) == artifacts[1].model_dump(
        exclude={"shares"}
    )
    assert [share.name for share in artifacts[1].shares] == ["stored words found nowhere"]


def test_a_blind_share_is_loud_in_its_row_and_absent_from_the_case_colour() -> None:
    """An instrument that read nothing is a fact about the instrument: its own row is red, and
    the case's colour is what its checks and its features make it."""
    quiet = [_sample(f"recipe-edit-{n}") for n in (1, 2, 3)]
    sections = report.CaseSections(
        case_id="recipe-edit",
        model=_MODEL,
        variance=cohort.pool(quiet, [ENTRIES_STORED], [STORED_WORDS_FOUND_NOWHERE]),
    )
    rendered = sections.render()

    assert (
        "| 🔴 | `stored words found nowhere` | 0/3 | — | — | "
        "— READ NOTHING on every sample; not a reading |"
    ) in rendered
    unmeasured = report.CaseSections(
        case_id="recipe-edit", model=_MODEL, variance=cohort.pool(quiet, [ENTRIES_STORED])
    )
    assert sections.glyph() == unmeasured.glyph(), "the blind share repaints nothing"
    assert "the samples to open" not in rendered, "a share with no reading points at nothing"
