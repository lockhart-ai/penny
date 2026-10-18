"""Six lookups: a routine's steps when it is not running, something the user said
weeks ago, every list she keeps, one she archived, a search inside one long list,
and a standing job's latest finding and last check (#2293).

Written to the cohort structure; the contract is `docs/eval-case-design.md`.  The
six are built to run unmodified on either side of a change to the tool surface, so
the SAME case measures which tools she reaches for before and after rather than
being re-tuned per commit.  Nothing here asserts a route or a tool name:
`TOOL_SEQUENCE` is measured, never claimed.

**Every anchor is ONE distinctive word**, chosen so an honest wording of the real
answer carries it regardless of how the model phrases the rest — never a hyphenated
slug or a phrase the model is free to reword.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code
owner's to accept once the numbers have been read.  Every page, list and job is
invented, because the repo is public.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from penny.penny import Penny
from penny.tests.eval.conftest import EVAL_MODELS, ChatEval
from penny.tests.eval.utils.assertions import Answer
from penny.tests.eval.utils.catalogue import Area
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    SampleObservation,
    SpecCategory,
    fold_typography,
)
from penny.tests.eval.utils.lookup_world import (
    FAMILY,
    HARBOUR_PAGE_WORD,
    HARBOUR_WATCH_WORD,
    LATEST_FIND_WORD,
    LIME_WORDS,
    LIST_WORDS,
    MANY_LISTS,
    PRESERVE_BOX,
    PRESERVE_WORDS,
    SISTER_FLIGHT_WORD,
    SISTER_MENTION,
    SISTER_REPLY,
    TYPEWRITER_WATCH,
    _seed_harbour_job,
    archive_after_seeding,
    assert_seeded_turns_embedded,
    seed_job_with_run_history,
    seed_long_recipe_list,
    seed_mention_out_of_window,
)
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, REPLY_SPREAD)


def _held_text(sample: SampleObservation, collection: str) -> str:
    """Everything one collection holds as the sample left it, key and content,
    normalized — the whole entry, since a fact in the key and a blurb in the body
    is a perfectly good way to store it."""
    return fold_typography(
        " ".join(entry.text for entry in sample.held if entry.collection == collection)
    )


def _holds(collection: str, token: str) -> Callable[[SampleObservation, World], Answer]:
    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = _held_text(sample, collection)
        return token in held, f"{collection} holds {held!r}"

    return answer


def _the_list_is_archived(collection: str) -> Callable[[SampleObservation, World], Answer]:
    def answer(sample: SampleObservation, _world: World) -> Answer:
        row = next((one for one in sample.mechanisms if one.name == collection), None)
        if row is None:
            return False, f"{collection} was removed from the registry rather than archived"
        return row.archived, f"{collection} is still live"

    return answer


# ═══ 1 · a routine's steps, when it is not running as a job ═════════════════
#
# #2519's naive baseline: there is no skill registry left to hold a routine unattached
# from any job, so "a routine's steps" is read off the running OR ARCHIVED job's own
# prompt instead.  Seeded as a job Penny already knows (``seed_known_routine`` — the
# job itself, plus the earlier conversation that set it up, in her own logs), then
# archived — which is what keeps "not running as any job" literally true.  The two
# anchors live in the job's own plain-text prompt (``f"Check {url} for {extract} and
# save it."``), which a ``find``/``memory_get`` read carries wherever it renders.

_HARBOUR_ASK = "remind me how you check the harbour flag"
_HARBOUR_PHRASINGS = (
    "what's the routine for checking the harbour flag again?",
    "walk me through how you check on the harbour flag",
    "what do you actually do to check the harbour flag?",
    "can you remind me what your harbour-flag check involves?",
)
_HARBOUR_WORLD = World(
    name="unattached routine",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(HARBOUR_PAGE_WORD, HARBOUR_WATCH_WORD),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_routine_not_running_is_recited(chat_eval: ChatEval, model: str) -> None:
    """A taught routine sits in the registry with no job running it — the ask is
    about the procedure itself, not a run of it."""
    cohort = await chat_eval(
        case_id="lookup-routine-steps-unattached",
        behaviour=(
            "In the chat agent, when the user asks what a taught routine does while "
            "it is not running as any job, Penny reads the routine's steps and "
            "states them."
        ),
        area=Area.MEMORY,
        model=model,
        seed=_seed_harbour_job,
        world=_HARBOUR_WORLD,
        ask=_HARBOUR_ASK,
        also_phrased=_HARBOUR_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=FAMILY,
        timeout=240.0,
    )
    # STORE — empty, and that is a report: the ask reads the job's own record, writes
    # nothing, and the registry holds exactly the one job this world seeded either way.
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(*_MEASURED)


# ═══ 2 · something the user said weeks ago, never saved ═════════════════════
#
# A passing mention, not a request to remember it — Penny's own seeded reply beside
# it is plain conversation and names no save, so the world is coherent: nothing
# holds the flight number anywhere, and the only route back to it is reading what
# the user actually said.

_SISTER_ASK = "what flight number did I give you for my sister?"
_SISTER_PHRASINGS = (
    "remind me what flight number I told you for my sister",
    "what was the flight number I mentioned for my sister's trip?",
    "did I ever tell you my sister's flight number — what was it?",
    "go back and check — what flight did I say my sister was on?",
)
_SISTER_WORLD = World(
    name="out-of-window mention",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(SISTER_FLIGHT_WORD,),
)


def _sister_premise(penny: Penny) -> None:
    assert_seeded_turns_embedded(penny.db)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_old_mention_comes_back_out_of_what_was_said(
    chat_eval: ChatEval, model: str
) -> None:
    """A detail the user mentioned in passing, long enough ago to be out of the
    conversation window, with no save anywhere to answer it instead."""
    cohort = await chat_eval(
        case_id="lookup-user-said-out-of-window",
        behaviour=(
            "In the chat agent, when the user asks what they themselves said "
            "earlier and it was never saved anywhere, Penny finds it in what they "
            "said and states it."
        ),
        area=Area.MEMORY,
        model=model,
        seed=seed_mention_out_of_window(SISTER_MENTION, SISTER_REPLY),
        prepare=_sister_premise,
        world=_SISTER_WORLD,
        ask=_SISTER_ASK,
        also_phrased=_SISTER_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=FAMILY,
        timeout=240.0,
    )
    # STORE — empty, and that is a report: the ask keeps nothing and the fact lives
    # only in the conversation.
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(*_MEASURED)


# ═══ 3 · every list she keeps, in a world with thirty ═══════════════════════
#
# Thirty is more than a reply lists comfortably, so "names every one" is a claim a
# reply can miss: each list's own token has to appear, wherever she read the names.

_LISTS_ASK = "what lists do you keep for me?"
_LISTS_PHRASINGS = (
    "can you run me through every list you've got going?",
    "what are all the lists you're keeping track of?",
    "remind me of every list you keep",
    "what have you got lists for right now?",
)
_LISTS_WORLD = World(
    name="thirty lists",
    pages=(),
    keeps=(),
    excludes=(),
    answers=LIST_WORDS,
    stores=MANY_LISTS,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_every_list_is_named_among_thirty(chat_eval: ChatEval, model: str) -> None:
    """Thirty lists — the ask is for all of them, by name."""
    cohort = await chat_eval(
        case_id="lookup-list-of-lists",
        behaviour=(
            "In the chat agent, when the user asks what lists she keeps for them, "
            "Penny names every one."
        ),
        area=Area.MEMORY,
        model=model,
        world=_LISTS_WORLD,
        ask=_LISTS_ASK,
        also_phrased=_LISTS_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=FAMILY,
        timeout=240.0,
    )
    # STORE
    cohort.assert_the_running_mechanisms_survive()
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(*_MEASURED)


# ═══ 4 · what was on a list she has since archived ═══════════════════════════
#
# `find`'s stored-entry search excludes archived collections, so the model has to
# resolve the LIST itself first (by name or by meaning, both archived-inclusive)
# before it can read its entries back.

_PRESERVE_ASK = "what was on that preserve box list I got rid of?"
_PRESERVE_PHRASINGS = (
    "before I archived my preserve box list, what was on it?",
    "remind me what used to be in my preserve box list",
    "what did my old preserve box list have on it?",
    "I deleted my preserve box list a while back — what was in it?",
)
_PRESERVE_WORLD = World(
    name="an archived list",
    pages=(),
    keeps=(),
    excludes=(),
    answers=PRESERVE_WORDS,
    stores=(PRESERVE_BOX,),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_an_archived_list_is_reached_and_read(chat_eval: ChatEval, model: str) -> None:
    """One list, archived with its entries still in it — the ask reaches for it by
    name rather than by its contents."""
    cohort = await chat_eval(
        case_id="lookup-archived-list-contents",
        behaviour=(
            "In the chat agent, when the user asks what was on a list she has "
            "since gotten rid of, Penny reaches the archived list and reads it "
            "back."
        ),
        area=Area.MEMORY,
        model=model,
        seed=archive_after_seeding(PRESERVE_BOX),
        world=_PRESERVE_WORLD,
        ask=_PRESERVE_ASK,
        also_phrased=_PRESERVE_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=FAMILY,
        timeout=240.0,
    )
    # STORE — the list stays archived, with everything it held.
    cohort.claim(
        "state: the list is still archived",
        _the_list_is_archived(PRESERVE_BOX.name),
        SpecCategory.STORE,
    )
    for word in PRESERVE_WORDS:
        cohort.claim(
            f"state: the list still holds the {word!r} entry",
            _holds(PRESERVE_BOX.name, word),
            SpecCategory.STORE,
        )
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(*_MEASURED)


# ═══ 5 · which entries of one long list match something ═════════════════════
#
# ~40 entries, two of them carrying "lime" — written FIRST, so they are the oldest
# in the list rather than among the newest.  `collection_list` returns every
# entry when `k` is omitted, so the honest wrong answer is a model that leans on a
# ranked similarity search, or a habitual small `k`, instead of reading the list
# whole.

_RECIPES_ASK = "which of my recipes use lime?"
_RECIPES_PHRASINGS = (
    "do any of my saved recipes call for lime?",
    "what recipes have I saved that use lime in them?",
    "which recipes on my list use lime?",
    "can you check which of my recipes have lime as an ingredient?",
)
_RECIPES_WORLD = World(
    name="one long list",
    pages=(),
    keeps=(),
    excludes=(),
    answers=LIME_WORDS,
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_long_list_is_searched_not_skimmed(chat_eval: ChatEval, model: str) -> None:
    """One ~40-entry list, with its two matching recipes among the oldest entries —
    recency alone cannot answer this."""
    cohort = await chat_eval(
        case_id="lookup-search-within-a-long-list",
        behaviour=(
            "In the chat agent, when the user asks which entries of one named list "
            "match something, Penny searches the whole list and names them."
        ),
        area=Area.MEMORY,
        model=model,
        seed=seed_long_recipe_list,
        world=_RECIPES_WORLD,
        ask=_RECIPES_ASK,
        also_phrased=_RECIPES_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=FAMILY,
        timeout=240.0,
    )
    # STORE — a read-only ask: the list is exactly as it was.
    cohort.assert_what_the_store_held_survives()
    # PROVENANCE
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(*_MEASURED)


# ═══ 6 · a standing job's latest finding and last check ═════════════════════
#
# The same typewriter watch the house already has, with two completed cycles
# behind it — an earlier find and a later one.  "When did it last check" is
# ambient on the self-state header; "what did it last find" is not, and needs a
# read.  The completeness claim names the LATER find only: the earlier one is a
# correct thing to also mention, never a required one.

_TYPEWRITER_ASK = "what did the typewriter watch turn up last time, and when did it last check?"
_TYPEWRITER_PHRASINGS = (
    "what's the latest the typewriter watch found, and when did it last run?",
    "what did the typewriter watch find most recently, and when did it last check?",
    "any new finds from the typewriter watch, and when was its last check?",
    "give me the typewriter watch's latest find and when it last ran",
)
_TYPEWRITER_WORLD = World(
    name="a standing job's history",
    pages=(),
    keeps=(),
    excludes=(),
    answers=(LATEST_FIND_WORD,),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_running_jobs_last_finding_and_check_are_read(
    chat_eval: ChatEval, model: str
) -> None:
    """A standing job with two completed cycles behind it — the latest finding and
    the last check, both read off its own record."""
    cohort = await chat_eval(
        case_id="lookup-job-last-finding-and-check",
        behaviour=(
            "In the chat agent, when the user asks what a running job last found "
            "and when it last checked, Penny reads its latest entries and its "
            "last run."
        ),
        area=Area.STANDING_JOBS,
        model=model,
        seed=seed_job_with_run_history,
        world=_TYPEWRITER_WORLD,
        ask=_TYPEWRITER_ASK,
        also_phrased=_TYPEWRITER_PHRASINGS,
        samples_per_phrasing=3,
        min_pass_rate=None,
        family=FAMILY,
        timeout=240.0,
    )
    # STORE — the job is still running, as it was, holding both finds.
    cohort.assert_the_running_mechanisms_survive()
    for word in (LATEST_FIND_WORD, "ribbon"):
        cohort.claim(
            f"state: the job still holds the {word!r} entry",
            _holds(TYPEWRITER_WATCH.container, word),
            SpecCategory.STORE,
        )
    # PROVENANCE — covers whatever time the reply states too: a stamp is a "moment"
    # the self-state header rendered, which the design's provenance rule allows.
    cohort.assert_every_value_in_the_reply_is_sourced()
    cohort.assert_the_reply_answers_the_ask()
    cohort.measure(*_MEASURED)
