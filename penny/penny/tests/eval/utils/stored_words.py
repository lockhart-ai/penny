"""Of the words a turn STORED, the share that stand nowhere in what the round was given
(#2199).

A provenance claim reads the values that have a fixed written form: a number, an address, a
capitalised name.  An entry that gains invented plain prose passes every one of them.
MEASURED: a recipe edit asked to add a marinade stored `mix 2 tbsp lime juice, 1 tsp garlic
powder, 1 tsp cumin` on about half of one model's samples, and a save with one source down
stored a note about a page that was never read.

Whether prose FOLLOWS from a source cannot be answered true or false without a judge, so it is
not a claim.  It is model behaviour, and behaviour is measured.  This is the measurement:

  WHAT IS READ   the entries THIS turn wrote, key and content, all of them together
  A WORD         a run of letters, read through the one folding every claim uses
                 (``cohort.words_in``): typography and case folded, the possessive dropped.  A
                 hyphen parts two words, and a hyphenated word the world writes as one is
                 found as that one word.  A figure is not a word (the provenance claim reads
                 it), and nor is a single letter.
  CONTENT WORDS  every word that is not a FUNCTION word.  The function words are the closed
                 list below, plus any contraction (`don't`, `you're`), which is function words
                 run together.  They are left out of both halves of the fraction.
  DISTINCT       each word counts once per sample, however often it was written, so a long
                 entry repeating one given word reads no lower than a short one.
  THE WORLD      the round's CONTENT as ``given.py`` draws it: the user's turns, each tool's
                 own payload, the entries a read laid out, the document a micro-context was
                 handed.  Never a system prompt, and never Penny's own turns.
  FOUND          the world writes the same word, or the same word with another ENDING: the
                 two share a stem of at least four letters, and what follows the stem in each
                 is one of the closed list of endings below — the regular endings an English
                 noun or verb takes, and the silent `e` one of them replaces.
                 `pepper`/`peppers`, `signed`/`signing`, `marinate`/`marinating`,
                 `store`/`storing`.  No stemmer: the rule is that list and that length.

  THE READING    words found nowhere / content words, per sample.  ``None`` for a sample that
                 wrote no entry (or none with a content word in it), which is this axis's
                 absent reading.

WHAT IT CANNOT SEE, STATED.

It reads WORDS, never what they say: `the page had no news`, built from words the round was
given, reads zero.  MEASURED on saved samples: a note that a page could not be read, and
seven notes that a page did not carry the date asked for, read 0.00 to 0.08.

Read LOW: a different word that is a given one plus a listed ending is found (`plane` beside
`plan`, `evening` beside `even`).  And content is taken as it comes: a write's own result
names the key it wrote, so a key's words are found in that echo; an entry read back after it
was written is found in that read; and the document a micro-context was handed is content, so
its vocabulary is too.

Read HIGH: a different word for a given thing is a word found nowhere (`marinate` beside the
user's `marinade`), and so is a form the ending rule does not reach — a stem under four
letters (`add`/`added`, `bake`/`baking`), a doubled consonant (`chop`/`chopped`), a `y` that
became `i` (`berry`/`berries`).  So an entry that only rewords reads a little above zero, one
word in ten or so on the saved samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from penny.tests.eval.utils.cohort import (
    Given,
    SampleObservation,
    Share,
    ShareReading,
    StoredEntry,
    fold_for_token_match,
    words_in,
)

# The FUNCTION words — the grammar a sentence is held together with, which carries nothing a
# source could have stated.  Closed, and stated once.  Grouped by what each group does, so a
# reader checking a word looks in one place.
_ARTICLES = "a an the"
_CONJUNCTIONS = (
    "and or but nor so yet if then than as because while although though unless until whether"
)
_PREPOSITIONS = (
    "of in on at to from by with without for about into onto over under between among through "
    "during before after above below up down out off near against across around along per via"
)
_PRONOUNS = (
    "i me my mine we us our ours you your yours he him his she her hers it its they them their "
    "theirs this that these those there here who whom whose which what when where why how"
)
_AUXILIARIES = (
    "is am are was were be been being have has had having do does did doing done will would "
    "shall should can could may might must"
)
_DETERMINERS = (
    "all any both each every few many more most much no not some such other another either "
    "neither own same only just also too very now"
)
STOP_WORDS = frozenset(
    " ".join(
        (_ARTICLES, _CONJUNCTIONS, _PREPOSITIONS, _PRONOUNS, _AUXILIARIES, _DETERMINERS)
    ).split()
)

# A letter alone is not a word: a unit's letter beside its figure (`425F`), a list's `a`.
_WORD_LETTERS = 2
# A contraction keeps its apostrophe through the fold, where a possessive loses it.
_CONTRACTION_MARK = "'"

# The ENDINGS one word may differ from another by and still be that word: none, the plural and
# third-person `s`/`es`, the past `d`/`ed`, the progressive `ing`, and the silent `e` those
# replace (`store`/`storing`).  Closed, and stated once.
_ENDINGS = frozenset({"", "e", "s", "es", "d", "ed", "ing"})
# Four letters of shared stem is what keeps two short words that merely start alike apart:
# `new` is not `news`, and `lime` is not `limit`.
_STEM_LETTERS = 4

# The share a reader is pointed at a sample above: more than a quarter of what it stored
# stands nowhere in what it was given.  A round figure, stated as one and not measured.
# Nothing is compared against it and nothing fails by it.
OPEN_ABOVE = 0.25

FEATURE_NAME = "stored words found nowhere"
FEATURE_SAYS = (
    "of the distinct content words in the entries the sample wrote (keys and contents, "
    "function words and figures left out), the fraction that stand nowhere in what the round "
    "was given, allowing a word another ending"
)

_ENTRY_PART_GAP = "\n"


def content_words(text: str) -> list[str]:
    """The DISTINCT content words ``text`` is written in, in the order first written."""
    return list(dict.fromkeys(word for word in words_in(text) if _is_content(word)))


def _is_content(word: str) -> bool:
    return len(word) >= _WORD_LETTERS and _CONTRACTION_MARK not in word and word not in STOP_WORDS


def same_word(one: str, other: str) -> bool:
    """Whether two words are one word, allowing each an ending."""
    if one == other:
        return True
    shared = _shared_stem(one, other)
    return any(
        one[stem:] in _ENDINGS and other[stem:] in _ENDINGS
        for stem in range(_STEM_LETTERS, shared + 1)
    )


def _shared_stem(one: str, other: str) -> int:
    """How many letters two words open with in common."""
    shared = 0
    for mine, theirs in zip(one, other, strict=False):
        if mine != theirs:
            break
        shared += 1
    return shared


class _WorldWords:
    """The words a round's content is written in, read once and asked many times."""

    def __init__(self, content: str) -> None:
        # Both readings of a hyphenated word, the way ``token_stands_in`` asks both: its
        # parts, and the one word they make.
        self._words = frozenset(words_in(content)) | frozenset(_whole_words(content))
        self._by_stem: dict[str, list[str]] = {}
        for word in self._words:
            if len(word) >= _STEM_LETTERS:
                self._by_stem.setdefault(word[:_STEM_LETTERS], []).append(word)

    def states(self, word: str) -> bool:
        if word in self._words:
            return True
        return any(same_word(word, other) for other in self._by_stem.get(word[:_STEM_LETTERS], ()))


def _whole_words(text: str) -> list[str]:
    """``text``'s words with each hyphenated word read as the ONE word it makes."""
    return words_in(fold_for_token_match(text))


@lru_cache(maxsize=8)
def _world_words(content: str) -> _WorldWords:
    """One round's content read once: every sample is weighed against its own."""
    return _WorldWords(content)


def stored_words_reading(entries: Sequence[StoredEntry], given: Given) -> ShareReading | None:
    """What share of the content words in ``entries`` stand nowhere in ``given``'s content, and
    which words those are.  ``None`` when there is no content word to take a share of."""
    text = _ENTRY_PART_GAP.join(_entry_text(entry) for entry in entries)
    written = content_words(text)
    if not written:
        return None
    world = _world_words(given.content)
    as_one_word = _hyphenated_words_the_world_states(text, world)
    nowhere = [
        word
        for word in written
        if not world.states(word) and not any(word in whole for whole in as_one_word)
    ]
    return ShareReading(value=len(nowhere) / len(written), evidence=nowhere)


def _entry_text(entry: StoredEntry) -> str:
    """One entry's key and content, each on its own line."""
    return _ENTRY_PART_GAP.join(part for part in (entry.key, entry.content) if part)


def _hyphenated_words_the_world_states(text: str, world: _WorldWords) -> list[str]:
    """The hyphenated words in ``text`` that the world writes as ONE word (`silver-leaf`
    beside the world's `silverleaf`), each in that one-word form — so its parts are found
    with it rather than counted as two words nobody gave."""
    parts = set(words_in(text))
    return [word for word in _whole_words(text) if word not in parts and world.states(word)]


def _read(sample: SampleObservation) -> ShareReading | None:
    return stored_words_reading(sample.entries, sample.given)


# MEASURED on the cases that write entries.  A share, so it carries no entropy and proposes
# no ceiling: the report states the median and the range across the samples that wrote, and
# names the samples above ``OPEN_ABOVE`` with the words that put them there.
STORED_WORDS_FOUND_NOWHERE = Share(FEATURE_NAME, _read, open_above=OPEN_ABOVE, says=FEATURE_SAYS)
