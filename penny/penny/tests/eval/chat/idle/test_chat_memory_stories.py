"""The memory verbs: save · recall · forget · update · fan-out · no-fire (#2008, tranche 1).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**Seven cases.**  What a user asks Penny's memory to do splits into verbs that are genuinely
different claims rather than scenarios standing in for one — saving, recalling, forgetting and
updating are four contracts, and a sample that is correct for one is wrong for another.  Only
*within* a verb is there paraphrase-collapsing to do, and each case here is one ask in five
wordings against one world, with its facts held constant across the arms.

| behaviour | case | why this survivor |
|---|---|---|
| save | ``memory-look-up-and-save`` | one fetch, one write, one named place |
| recall, from the store | ``memory-cold-recall`` | the fact exists ONLY in the store |
| recall, across the store | ``memory-recall-across-the-store`` | a different sentence — see below |
| forget | ``memory-forget-then-list`` | the only ask that owes the report half |
| update | ``memory-change-lands-on-the-entry-that-exists`` | the edit lands on the entry |
| fan-out | ``memory-a-like-and-a-dislike`` | the slot's only candidate |
| no-fire | ``memory-no-fire-wistful`` | the mention with a matching collection sitting there |

No per-variant pass rate exists for any of these slots — the file they come from scored each
case as one mean over a scorer callback, never per candidate — so every survivor above is
chosen by DOMINANCE, and the sentence beside it is the reason.

**RECALL SPLIT IN TWO, and that is deliberate.**  Its four candidates spanned three starting
states, and the design's test — *would a correct sample for one be wrong for the other?* —
answers differently for each pair:

* *the same conversation* and *a previous session* differ only in where the fact came from,
  and a reply that reads the store answers both.  One behaviour.
* *across the whole store* asks something else: the failure it exists to catch is PARTIAL
  recall — three collections asked about, two answered from — and a reply naming one stored
  fact is correct for the cold case and wrong for this one.  It needs its own sentence
  (*"when one message asks about several of her collections at once"*), and #2004's rule is
  that a candidate needing a different sentence is a different behaviour.

So recall is two cases, not one with three worlds.  Splitting a behaviour in two is a smaller
mistake than collapsing two into one; the pair is named together here.

**UPDATE IS ONE CASE.**  ``memory-change-lands-on-the-entry-that-exists`` is an edit to an
entry the user names differently from the key it is stored under: the recipe is kept as
``Sheet-pan chicken fajitas`` and asked about as *my fajitas recipe*.  The change is made ON
that entry rather than beside it as a second fajitas recipe, and the edit adds to the recipe,
so what the entry already held is still there afterwards.

**THE NO-FIRE DIRECTION IS A CASE OF ITS OWN** (#2100).  The ruling: one case is one
setup, one run, one set of assertions, and where a behaviour's positive and negative
directions cannot both be produced from that one setup and run, they are two cases.  A
no-fire ask expects the opposite end state from save's, so it cannot be an arm of it.

``memory-no-fire-wistful`` is that case.  A collection about the message's own subject sits in
the store, and that temptation is the entire thing it measures: a mention that changes nothing
with a matching list right there is the harder form of the same behaviour, so it also stands for
a mention with nothing in the store to match it.

**QUARANTINED, not deleted** — every candidate that did not survive, with the reason it can
come back deliberately:

* ``memory-remember-and-recall`` — TWO user turns, so two enactments: a save and a recall
  folded into one case.  The count rule is the count wherever it is expressed, and its save
  half is the save behaviour already.
* ``memory-activity-window-recall`` — the same claim as the cold recall (the reply states the
  price, nothing is written) against a world where the anchor is MORE reachable, since the
  self-state activity line renders the key and the collection.  Same sentence, so it is the
  same behaviour; it is the reachability probe worth restoring the day n≤1 is measured
  directly.
* ``memory-look-up-two-hops`` — its distinguishing claim is that the second page was opened,
  which is *browse-and-answer*'s (``chat-answer-one-link-deep``, tranche 2), not save's.
* ``memory-check-then-fill`` — save with a condition on the front that the store map already
  satisfies ambiently, so the condition is not exercised.
* ``memory-forget-one-entry`` — its ask ("what am I into these days? actually drop chess from
  that") leaves the ORDER free, so a reply listing all three and then dropping one is correct;
  that makes the *says what is left* half of the sentence unassertable on it.
* ``memory-look-up-and-update`` — folds save's lookup into update, so a sample can fail it for
  a reason that has nothing to do with editing in place.

The wistful no-fire pool came OUT of quarantine under #2100 and is ported below.

``memory-writes-landed-source-down`` is ported to the cohort structure as its own case under
#2149 and is not in this file.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every collection, game, price and note is invented,
because the repo is public.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest

from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.penny import Penny
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    ChatEval,
    Preparer,
    collection_entries,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
    StoredEntry,
    fold_typography,
)

# The pages the lookup story reads, and the collection shape every seed is written as, read
# from the suite's shared fixtures rather than restated: two copies of a page a case is
# measured against are two contracts free to drift.
from penny.tests.eval.utils.fixtures import (
    MULTIHOP_PAGES,
    SynthCollection,
)
from penny.tests.eval.utils.memory_world import _FAMILY
from penny.tests.eval.utils.worlds import World

pytestmark = pytest.mark.eval

# ── The collections the verbs act on (all user-built) ────────────────────────

_GAMES = SynthCollection(
    "games",
    "Video games the user is tracking or interested in: titles, release dates, and notes.",
    entries=(),
)

# The one game the lookup story is about — invented, so it exists on no real site and cannot be
# answered from what the model already knows.  Its key is the text before the em dash, the way
# every seeded entry in this suite is written.
_MISTFORGE_ENTRY = "Mistforge Tactics — a turn-based strategy game."

_GAMES_WITH_MISTFORGE = SynthCollection(
    _GAMES.name, _GAMES.description, entries=(_MISTFORGE_ENTRY,)
)

_INTO = SynthCollection(
    "things-im-into",
    "Things the user is into: hobbies, music, and the ways they like to spend their time.",
    entries=(
        "chess — enjoys playing chess",
        "hiking — loves weekend hikes",
        "jazz — a big fan of jazz records",
    ),
)

_AVOID = SynthCollection(
    "things-i-avoid",
    "Things the user would rather avoid: places, sounds, and food they dislike.",
    entries=("loud offices — can't focus in them",),
)

# The recipe box the update edits.  Its first entry is keyed
# ``Sheet-pan chicken fajitas`` — the text before the em dash — and no wording of that case says
# the key verbatim.
_RECIPE_BOX = SynthCollection(
    "recipe-box",
    "Saved quick weeknight dinner recipes: short ingredient lists and fast cook times.",
    entries=(
        "Sheet-pan chicken fajitas — peppers, onion, chicken, 25 min at 425F.",
        "One-pot lemon orzo — orzo, lemon, spinach, parmesan, 20 min.",
    ),
)
_FAJITAS_KEY = _RECIPE_BOX.keyed[0][0]

_GEAR_PRICE_KEY = "aurora deck 2 price"
_GEAR_PRICE = "$499"

_GEAR_NOTES = SynthCollection(
    "gear-notes",
    "Notes about gear the user tracks: what a thing is, and what it was listed at.",
    entries=(f"{_GEAR_PRICE_KEY} — {_GEAR_PRICE}",),
)


# ── The smallest datum unique in each world ──────────────────────────────────
#
# Named once each, because two readers have to agree on every one of them: the PREMISE probe,
# which asserts the token is not in the store before the turn, and the CLAIM, which asserts it
# is after — and a token spelled twice would let the premise stop guarding the claim it exists
# for.  Each is shrunk to the part with no alternative rendering and no other bearer in its
# world: ``499`` and not ``$499`` (a draw that wrote the figure without the currency symbol
# read the same entry), ``mistforge`` and not ``Mistforge Tactics`` (a draw that put the title
# in the key and the blurb in the body read the same page), ``office`` and not ``loud offices``
# (a measured reply answered out of all three collections and wrote "loud office spaces", which
# the plural failed — the singular is the part every correct rendering shares).

_SUBJECT = "mistforge"  # the game the lookup story is about
_PRICE = "499"  # the figure the cold recall has to state
_AVOIDED = "office"  # the one thing the avoid list already holds
_INTO_ANCHOR = "chess"  # the note the into-list holds in every world that seeds it
_KEPT_NOTE = "hiking"  # a note the forget story leaves in place
_DROPPED = "jazz"  # the note the forget story names
_LIKE = "bouldering"  # the enthusiasm the fan-out story files
_DISLIKE = "coffee"  # the complaint it files somewhere else
_MARINADE = "lime"  # what the update adds to the recipe
_RECIPE_KEPT = "425"  # what that recipe already said, which adding to it must not lose

# What the forget must leave exactly as it was — the seeded notes it did not name.  Derived from
# the tokens above rather than listed again, so the case can never claim a note is untouched
# while naming that same note as the one to drop.
_FORGET_KEPT = (_INTO_ANCHOR, _KEPT_NOTE)

# What a correct REPLY must name, which is not the same token set the store claims read.  The
# store holds the seeded text verbatim, so a claim about it can name the whole word; the reply
# is the model's own prose, where an inflectable noun has no single correct rendering — a reply
# saying "weekend hikes" answered the ask exactly as well as one saying "hiking", and requiring
# the participle would fail it for a cosmetic reason.  So the reply-side token is the stem, and
# it is still the SMALLEST unique one: no other seeded name, description or entry in any of
# these worlds contains ``hik``.
_FORGET_ANSWERS = (_INTO_ANCHOR, "hik")

# The recipes the update does NOT name, with what each was seeded with —
# derived from the box rather than listed again, so the case can never claim a recipe is
# untouched while naming that same recipe as the one to change.
_RECIPES_NOT_NAMED = tuple(
    (key, content) for key, content in _RECIPE_BOX.keyed if key != _FAJITAS_KEY
)


# ── The case shape ───────────────────────────────────────────────────────────


class _VerbCase(NamedTuple):
    """One memory verb: the world it acts on, the ask in five wordings, and the premise its
    claims stand on.

    ``ask`` and ``also_phrased`` are five wordings of ONE message against one world — the
    cohort's arms.  What varies is only how a person says it; the collections, the subject, the
    value and the end state expected of the turn are constant, which is what makes the fifteen
    samples one number and what lets a claim name a value at all.

    ``holds`` / ``empty`` / ``withholds`` are the seeded world's own PREMISE, asserted loudly
    before the turn runs: what each collection must already carry, which must start empty, and
    which tokens must appear nowhere in the store.  Every one of them is the precondition of a
    claim below — ``withholds`` most of all, because a token already in the store would make
    the claim that names it pass without the turn doing anything.

    ``unspoken`` is a premise about the WORDINGS rather than the store: each is a key the store
    holds that no wording may say verbatim.  A case about finding an entry under a key the user
    never says measures nothing the day one of its wordings says it."""

    case_id: str
    behaviour: str
    world: World
    ask: str
    also_phrased: tuple[str, ...]
    holds: tuple[tuple[str, tuple[str, ...]], ...] = ()
    empty: tuple[str, ...] = ()
    withholds: tuple[str, ...] = ()
    unspoken: tuple[tuple[str, str], ...] = ()


def _collection_text(db: Database, name: str) -> str:
    """Every key and content one collection currently holds, normalized and joined."""
    entries = collection_entries(db, name)
    return fold_typography(" ".join([*entries.keys(), *entries.values()]))


def _store_text(db: Database) -> str:
    """Every key and content the WHOLE store currently holds — what a "nowhere yet" premise is
    read against, since a token sitting in a neighbouring collection would satisfy a claim just
    as well as one the turn wrote."""
    return fold_typography(
        " ".join(_collection_text(db, row.name) for row in db.memories.list_all())
    )


def probe_seeded_world(db: Database, case: _VerbCase) -> None:
    """The case's premise, re-read against a live database — what must be there, what must be
    empty, and what must be nowhere.

    Separate from the ``Preparer`` that wraps it so the same assertions run without a model:
    a premise that quietly stopped holding turns a scored claim into a claim about the
    fixture, and the cheapest place to catch that is ``make check``."""
    for name, tokens in case.holds:
        text = _collection_text(db, name)
        for token in tokens:
            assert token in text, (
                f"{case.case_id}: {name!r} must already hold {token!r} — it holds {text!r}"
            )
    for name in case.empty:
        entries = collection_entries(db, name)
        assert not entries, (
            f"{case.case_id}: {name!r} must start empty, or the claim that it gained an entry "
            f"is answered by the seed — it holds {entries}"
        )
    everywhere = _store_text(db)
    for token in case.withholds:
        assert token not in everywhere, (
            f"{case.case_id}: nothing may hold {token!r} before the turn, or the claim that "
            f"names it passes without the turn acting — the store holds {everywhere!r}"
        )
    _probe_unspoken(db, case)


def _probe_unspoken(db: Database, case: _VerbCase) -> None:
    """Each unspoken key is one the store really holds, and one no wording says verbatim."""
    wordings = [fold_typography(wording) for wording in (case.ask, *case.also_phrased)]
    for name, key in case.unspoken:
        keys = sorted(collection_entries(db, name))
        assert key in keys, f"{case.case_id}: {name!r} must hold the key {key!r} — it holds {keys}"
        spoken = [wording for wording in wordings if fold_typography(key) in wording]
        assert not spoken, (
            f"{case.case_id}: no wording may say the stored key {key!r} verbatim — {spoken}"
        )


def _probe(case: _VerbCase) -> Preparer:
    """The prepare hook: the case's premise, asserted inside the sample it belongs to."""

    def probe(penny: Penny) -> None:
        probe_seeded_world(penny.db, case)

    return probe


async def _drive(chat_eval: ChatEval, model: str, case: _VerbCase) -> Cohort:
    """Drive one verb case: the collections an earlier session left, the pages the ask may
    reach for, and the premise re-asserted before the turn."""
    return await chat_eval(
        case_id=case.case_id,
        behaviour=case.behaviour,
        model=model,
        prepare=_probe(case),
        world=case.world,
        ask=case.ask,
        also_phrased=case.also_phrased,
        samples_per_phrasing=3,
        min_pass_rate=None,  # report-only until the numbers are read with the code owner
        family=_FAMILY,
        timeout=240.0,
    )


# What every case measures, and why the two registry features are absent.  ``ROUTINE_SHAPE``
# and ``ROUTINE_NAME`` read the routines in the registry, and none of these turns teaches one —
# so on a correct cohort they read the same empty registry on every sample and pool to a serene
# 0.000 that is neither agreement nor blindness but a reading of the FIXTURE.  A sample that DID
# mint a routine is caught where it matters: it left idle, which the landing claim reads.
#
# ``TOOL_SEQUENCE`` reads EVERY call the turn made, keyed to no name list (#2112), so it is a
# live measurement on every case here.  On the two recall cases a correct sample reads the
# store, and how it reaches the answer — one aimed read, a sweep of three, a ``find`` — is the
# spread those cases exist to watch.  On the NO-FIRE case it is where ACTING shows up, and
# it reads live there too: answering a message about the store is itself done by reading the
# store, so a cohort that answered and moved on agrees on the calls it took to do that, and a
# sample that went further — writing, minting a container, opening a page — is the spread beside
# it.  ``ENTRIES_STORED`` carries the other half one level down, at the store rather than at the
# call: ``0`` on every sample that wrote nothing, and the sample that wrote something is the
# divergence.  Both are measured rather than claimed, because whether a passing mention is worth
# keeping is hers to decide, and a spread is the honest place a decision that varies is reported.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ── The claims, as pure functions over one sample ────────────────────────────
#
# Every one is a read of what the turn LEFT BEHIND — the collections, the entries and their run
# stamps, the registry rows, the message it sent.  None reads a tool NAME: an entry cannot leave
# a collection by accident, so the end state already says whether the edit happened, and a rule
# keyed to ``collection_delete_entry`` would simply not fire for a plugin verb nobody enumerated.
#
# They stay LOCAL rather than graduating into ``assertions.py``.  A claim graduates at the second
# CUSTOMER, and the cases below are one behaviour family in one file: a second FILE asking
# one of these questions is what would make it shared, and no other slot has asked yet.

_ClaimFn = Callable[[SampleObservation, World], Answer]


def _held_text(sample: SampleObservation, collection: str) -> str:
    """Everything one collection HOLDS as the sample left it, key and content, normalized.

    The whole entry, always: a fact in the key and a blurb in the body is a perfectly good way
    to store it, and a content-only read once reported a 25/32 model failure that was entirely
    its own bug."""
    return fold_typography(
        " ".join(entry.text for entry in sample.held if entry.collection == collection)
    )


def _wrote_into(collection: str) -> _ClaimFn:
    """This turn put an entry in the collection the ask NAMED.

    Read off the run stamp rather than off a count, so a seeded entry can never answer it: what
    is being claimed is that this turn wrote there, not that something is there.  The world's
    own stores are laid down citing a SEEDED run (#2129), so every one of them reads as not this
    run's work and the distinction holds by construction."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        landed = sorted({entry.collection for entry in sample.entries})
        return collection in landed, f"this turn wrote into {landed}"

    return answer


def _stored_the_subject(token: str) -> _ClaimFn:
    """What this turn stored carries the subject — the SMALLEST datum that is unique in the
    world, matched as a substring of the whole entry.

    ``mistforge`` rather than ``Mistforge Tactics``: a draw that stored the title in the key and
    the blurb in the body read the same page as one that wrote the full name into a sentence,
    and the part they share is the fact."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        wrote = [fold_typography(entry.text) for entry in sample.entries]
        return any(token in text for text in wrote), f"this turn stored {wrote}"

    return answer


def _holds(collection: str, token: str) -> _ClaimFn:
    """The collection still holds a fact, whoever put it there — the read that answers *what
    did she leave alone*, which a list of this turn's writes cannot: there, an entry never
    touched and an entry deleted are both simply absent."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = _held_text(sample, collection)
        return token in held, f"{collection} holds {held!r}"

    return answer


def _no_longer_holds(collection: str, token: str) -> _ClaimFn:
    """The one thing the ask named is gone from the collection.

    A violating sample is nameable: one that left it in place, one that dropped a neighbour
    instead, one that "removed" it by rewriting its content while the key still carries it —
    which is why the read is over the WHOLE entry."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = _held_text(sample, collection)
        return token not in held, f"{collection} still holds {held!r}"

    return answer


def _held_keys(sample: SampleObservation, collection: str) -> list[str]:
    """Every key one collection holds as the sample left it, sorted."""
    return sorted(entry.key or "" for entry in sample.held if entry.collection == collection)


def _entry_under(sample: SampleObservation, collection: str, key: str) -> StoredEntry | None:
    """The entry the collection holds under exactly ``key``, as the sample left it."""
    return next(
        (entry for entry in sample.held if entry.collection == collection and entry.key == key),
        None,
    )


def _the_entry_carries(collection: str, key: str, token: str) -> _ClaimFn:
    """The entry that was already there, under the key it was stored under, carries a fact —
    read over the WHOLE entry, key and content.

    Keyed to the stored key rather than to "some entry mentions it", because the claim is about
    WHICH entry the change landed on: the same fact written into a second entry beside it
    leaves this one without it.  A violating sample is nameable — the entry gone, or still
    carrying only what it carried before."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        entry = _entry_under(sample, collection, key)
        if entry is None:
            held = _held_keys(sample, collection)
            return False, f"no entry under {key!r}; {collection} holds {held}"
        text = fold_typography(entry.text)
        return token in text, f"{key!r} holds {text!r}"

    return answer


def _holds_the_keys(collection: str, keys: tuple[str, ...]) -> _ClaimFn:
    """The collection holds exactly the keys it started with — a change landed ON an entry,
    not beside it.

    Set equality rather than a count, so a sample that dropped one recipe and filed the change
    as a new one is named for both halves of what it did."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        held = _held_keys(sample, collection)
        return held == sorted(keys), f"{collection} holds {held}"

    return answer


def _the_entry_is_unchanged(collection: str, key: str, content: str) -> _ClaimFn:
    """An entry the ask did not name holds exactly what it was seeded with."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        entry = _entry_under(sample, collection, key)
        held = None if entry is None else entry.content
        return held == content, f"{key!r} holds {held!r}"

    return answer


def _the_list_itself_survived(collection: str) -> _ClaimFn:
    """The list the ask named is still the live container it was — the read that answers *what
    did she leave standing*, one level up from what it holds.

    ``_holds`` reads a container's CONTENTS; this reads the container.  An archived list still
    holds every entry it held, so an edit that retired the whole list rather than changing the
    one note it was told to change passes every claim about the notes and fails this one.

    Read off the MUTATION LEDGER rather than off a field-by-field diff (``MechanismRecord``):
    what an edit to one entry must not do is touch the container itself, and a comparison keyed
    to a list of fields silently exempts whichever field nobody enumerated.  A violating sample
    is nameable: one that archives the whole list instead of dropping the note it was told to
    drop, one that rewrites the collection's description while it is in there."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        rows = [one for one in sample.mechanisms if one.name == collection]
        state = (
            ", ".join(f"archived={one.archived} changed={one.changed_this_run}" for one in rows)
            or "no longer in the registry"
        )
        standing = any(not one.archived and not one.changed_this_run for one in rows)
        return standing, f"{collection} is {state}"

    return answer


def _landed_in(sample: SampleObservation, collection: str, token: str) -> bool:
    """Whether an entry THIS TURN wrote into ``collection`` carries ``token``."""
    return any(
        entry.collection == collection and token in fold_typography(entry.text)
        for entry in sample.entries
    )


def _filed_in(collection: str, token: str) -> _ClaimFn:
    """One of the message's two facts landed in the collection that fits it."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        landed = sorted(
            {entry.collection for entry in sample.entries if token in fold_typography(entry.text)}
        )
        return _landed_in(sample, collection, token), f"{token} landed in {landed}"

    return answer


def _not_filed_in(collection: str, token: str) -> _ClaimFn:
    """Neither fact was ALSO filed in the other one's collection.

    Not entailed by the pair of destination claims above: a turn that opened one list and wrote
    both facts into it, then opened the other and wrote both again, satisfies both of them while
    filing nothing anywhere in particular — which is the failure this behaviour exists to catch."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        return not _landed_in(sample, collection, token), f"{token} also landed in {collection}"

    return answer


# ═══ save ════════════════════════════════════════════════════════════════════
#
# The value comes off a PAGE rather than out of the user's mouth, and the ask names exactly one
# place for it to go.  The subject is the invented game *Mistforge Tactics*, which exists on no
# real site, so the model has to read rather than answer from what it already knows.

_SAVE = _VerbCase(
    case_id="memory-look-up-and-save",
    behaviour=(
        "In the chat agent, when the user asks her to look a subject up and put it in a list "
        "they name, Penny reads about it and writes it into that list, and the subject is "
        "still there when the turn ends."
    ),
    # ``keeps`` is EMPTY and that is a REPORT.  A keeps token set identifies which SOURCE an
    # entry came from, and this world's two pages are a search hit and the detail page it links
    # to, which share the subject's own name — so no token identifies one page rather than the
    # other, and the detail page is not required reading for this ask anyway.  What the ask
    # requires kept is the subject, which this case's own STORE claim names directly.
    #
    # ``excludes`` is EMPTY for the same kind of reason: the ask rules nothing out in as many
    # words, and a token nobody excluded would assert one reading of the pages rather than a
    # read of the ask.
    #
    # ``answers`` is EMPTY because the ask is an instruction and owes the reply no value: "saved
    # it to your games list" is a complete answer, so requiring a token would fail a correct run
    # for something nobody requested.  Widening an ask so it requests a value is a code owner's
    # call rather than an in-flight repair.
    world=World(name="mistforge", pages=MULTIHOP_PAGES, keeps=(), excludes=(), stores=(_GAMES,)),
    ask="can you look up Mistforge Tactics, read up on it, and save it to my games list?",
    also_phrased=(
        "could you find out about Mistforge Tactics and put it on my games list?",
        "have a look at Mistforge Tactics, read up on it, and add it to my games list",
        "go read about Mistforge Tactics and save what you find to my games list",
        "look Mistforge Tactics up for me and stick it on my games list",
    ),
    empty=(_GAMES.name,),
    withholds=(_SUBJECT,),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_lookup_is_saved_where_it_was_asked_for(chat_eval: ChatEval, model: str) -> None:
    """The ask names a subject to read about and one list to put it in, and the list is empty
    when the turn starts — so everything in it afterwards is this turn's work."""
    cohort = await _drive(chat_eval, model, _SAVE)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the sentence's two halves: the list the ask named gained an entry, and what
    # landed in it is the subject she was sent to read about.
    cohort.claim(
        "state: the list the ask named holds an entry this turn wrote",
        _wrote_into(_GAMES.name),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: what this turn stored names the subject",
        _stored_the_subject(_SUBJECT),
        SpecCategory.STORE,
    )

    # PROVENANCE — the half the source case had none of.  The subject exists on no real site, so
    # a stored fact naming something neither page mentions was invented, and once it is in a
    # collection a collector re-reads it for ever.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ recall, from the store ══════════════════════════════════════════════════
#
# The fact was stored in a PREVIOUS session, so no conversation carries it and an echo is
# impossible: the answer exists only in the store.  No page is installed either, so a browse
# comes back empty — a reply carrying the price read the entry or invented it, and nothing else.

_COLD_RECALL = _VerbCase(
    case_id="memory-cold-recall",
    behaviour=(
        "In the chat agent, when the user asks for a fact they had her remember in an earlier "
        "session and nothing in the conversation carries it, Penny brings it back out of the "
        "store and states it, leaving the store exactly as she found it."
    ),
    # No pages: the fact is in the store and nowhere else, which is the whole point.  ``keeps``
    # and ``excludes`` are empty because there is no source to keep anything from.  ``answers``
    # is the ask's own request — the price — as the SMALLEST unique datum: ``499`` and not
    # ``$499``, because a reply that wrote the figure without the currency symbol read the same
    # entry, and the symbol is the model choosing how to render a number.
    world=World(
        name="cold store",
        pages=(),
        keeps=(),
        excludes=(),
        answers=(_PRICE,),
        stores=(_GEAR_NOTES,),
    ),
    ask=(
        "hey — a while back I asked you to remember what the aurora deck 2 "
        "was listed at. what was the price?"
    ),
    also_phrased=(
        "a while back i had you note down the aurora deck 2's listed price — what was it?",
        "you saved the aurora deck 2's price for me a while ago. what did it say?",
        "what was the aurora deck 2 listed at? i asked you to remember it a while back",
        "remind me of the aurora deck 2 price i had you keep from a while back",
    ),
    holds=((_GEAR_NOTES.name, (_PRICE,)),),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_fact_from_a_previous_session_comes_back(chat_eval: ChatEval, model: str) -> None:
    """A fact stored in a previous session, retrieved with zero conversational trace — the
    absolute test of one-call reachability, since there is nothing else it could come from."""
    cohort = await _drive(chat_eval, model, _COLD_RECALL)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the reply states the value the world carries (the *nothing omitted* direction: a
    # reply that answers nothing at all satisfies every other claim here vacuously), and the
    # note it was read out of SURVIVES.  Whether the turn also wrote something is hers to
    # decide, so it is measured by ``ENTRIES_STORED`` rather than claimed.
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: the note she read is still there",
        _holds(_GEAR_NOTES.name, _PRICE),
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply half.  It is the load-bearing one here: the price is a specific
    # value, and a reply
    # carrying one that traces to nothing the model was GIVEN is the invention this case is
    # named against.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ recall, across the store ════════════════════════════════════════════════
#
# One ask spanning THREE collections.  The failure it exists to catch is PARTIAL recall — a
# reminder that reaches two of the three and reads as complete — which is why the claim is the
# reply's COMPLETENESS and not how many reads it took.

# One token per collection, the same three the world's ``answers`` names, so what the reply is
# read for and what the store is read for cannot drift apart.
_SWEPT = (
    (_INTO.name, _INTO_ANCHOR),
    (_AVOID.name, _AVOIDED),
    (_GAMES.name, _SUBJECT),
)


def _sweep_left_everything(sample: SampleObservation, _world: World) -> Answer:
    """Every collection the ask named still holds what it held.

    A read of what SURVIVES rather than of what this turn wrote: a delete leaves no write
    behind, so a turn that tidied a list on its way through is only visible here."""
    missing = [
        f"{collection}:{token}"
        for collection, token in _SWEPT
        if token not in _held_text(sample, collection)
    ]
    return not missing, f"no longer holds {missing}"


_SWEEP = _VerbCase(
    case_id="memory-recall-across-the-store",
    behaviour=(
        "In the chat agent, when one message asks about several of her collections at once, "
        "Penny answers out of every one of them rather than the first she opens, and changes "
        "nothing while doing it."
    ),
    # ``answers`` is ONE token per collection, and that is the claim: the reminder reached each
    # of the three.  Not every entry of every list — the ask does not quantify a list, so
    # requiring three tokens out of ``things-im-into`` would assert one reading of "what i'm
    # into" rather than reading the ask.  Each token is the smallest unique one in the world:
    # ``offices`` rather than ``loud offices``, since a reply calling them noisy read the same
    # entry.
    world=World(
        name="three lists",
        pages=(),
        keeps=(),
        excludes=(),
        answers=tuple(token for _, token in _SWEPT),
        stores=(_INTO, _AVOID, _GAMES_WITH_MISTFORGE),
    ),
    ask="remind me what i'm into, what i'd rather avoid, and what's on my games list",
    also_phrased=(
        "run me through what i'm into, what i'd rather avoid, and what's on my games list",
        "what am i into, what would i rather avoid, and what's on my games list?",
        "give me a rundown of the things i'm into, the things i'd rather avoid, and my games list",
        "can you remind me what i'm into, what i'd rather avoid, and what's on my games list?",
    ),
    holds=tuple((collection, (token,)) for collection, token in _SWEPT),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_one_ask_recalls_across_the_whole_store(chat_eval: ChatEval, model: str) -> None:
    """One message asks for three collections at once, and the reminder names something out of
    each.  How many reads it took is measured, never asserted — a subset of reads answers the
    ask just as well, and many routes reach one end state."""
    cohort = await _drive(chat_eval, model, _SWEEP)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: each list still holds what it held",
        _sweep_left_everything,
        SpecCategory.STORE,
    )

    # PROVENANCE — the reply half, which is where this case's invention would land: three
    # collections asked about is three chances to name a fourth thing nobody stored.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ forget ══════════════════════════════════════════════════════════════════
#
# The user edits what is already stored, and the whole contract is EXACTNESS: the note they
# named goes, every other one is exactly where it was, and the reply reports what is left.  The
# collection is matched by MEANING rather than by a name repeated back — "my list of things i'm
# into" never says ``things-im-into``.

_FORGET = _VerbCase(
    case_id="memory-forget-then-list",
    behaviour=(
        "In the chat agent, when the user names one note to drop from a list, Penny drops that "
        "note and leaves every other one exactly as it was, and tells them what is still on the "
        "list."
    ),
    # ``answers`` is the two notes that SURVIVE, because the ask asks for them in as many words
    # ("tell me what else is on my list").  The dropped note is deliberately NOT claimed absent
    # from the reply: "dropped jazz — you've still got chess and hiking" is a correct answer that
    # names it, so an absence claim there would fail a correct run.
    world=World(
        name="one list",
        pages=(),
        keeps=(),
        excludes=(),
        answers=_FORGET_ANSWERS,
        stores=(_INTO,),
    ),
    ask="remove jazz from my list of things i'm into, then tell me what else is on it",
    also_phrased=(
        "drop jazz from my list of things i'm into, then tell me what's left on it",
        "take jazz off my list of things i'm into and tell me what else is on there",
        "get rid of jazz on my list of things i'm into, then say what remains",
        "remove jazz from the things i'm into, and let me know what's still on the list",
    ),
    holds=((_INTO.name, (*_FORGET_KEPT, _DROPPED)),),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_forgetting_then_reporting_names_what_is_left(
    chat_eval: ChatEval, model: str
) -> None:
    """One note named for removal, two left standing, and a report of what remains."""
    cohort = await _drive(chat_eval, model, _FORGET)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the named note gone, every other one still there, and the list itself still
    # standing.  The last is one level up, about the container rather than its contents: a turn
    # that archived the whole list instead of dropping one note leaves every remaining note
    # exactly where it was, so only a read of the container itself catches it.  How the
    # survivors got to be there — left alone, or deleted and typed back — is the route, and
    # ``ENTRIES_STORED`` measures it.
    cohort.claim(
        "state: the note it was told to drop is gone",
        _no_longer_holds(_INTO.name, _DROPPED),
        SpecCategory.STORE,
    )
    for token in _FORGET_KEPT:
        cohort.claim(
            f"state: the {token!r} note is still there",
            _holds(_INTO.name, token),
            SpecCategory.STORE,
        )
    cohort.claim(
        "state: the list itself is still the one she was given",
        _the_list_itself_survived(_INTO.name),
        SpecCategory.STORE,
    )
    cohort.assert_the_reply_answers_the_ask()

    # PROVENANCE — the reply half, since a report of what is left is prose that can invent.
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ update ══════════════════════════════════════════════════════════════════
#
# The recipe is stored under ``Sheet-pan chicken fajitas`` and the user calls it their fajitas
# recipe, so the key the change belongs under is one the user never says.  Nothing is forced:
# how the model finds the entry and which verb lands the change are the route, measured by
# ``TOOL_SEQUENCE``.  What is claimed is where the change ended up, and that what the recipe
# already held is still on it.

_UPDATE = _VerbCase(
    case_id="memory-change-lands-on-the-entry-that-exists",
    behaviour=(
        "In the chat agent, when the user asks her to change a note she keeps under a key "
        "worded differently from how the user names it, Penny finds the entry that exists and "
        "records the change on it, and the list holds the same entries it started with."
    ),
    # ``answers`` is EMPTY: the ask is an instruction, so a bare "done" answers it, and
    # requiring a token would fail a correct run for something nobody requested.  No pages: the
    # change comes out of the user's own message.
    world=World(name="recipe box", pages=(), keeps=(), excludes=(), stores=(_RECIPE_BOX,)),
    ask="add a 10-minute lime marinade to my fajitas recipe",
    also_phrased=(
        "can you update my fajitas recipe so the chicken gets a 10-minute lime marinade first?",
        "my fajita recipe needs a 10-minute lime marinade added to it",
        "put a 10-minute lime marinade step into the fajitas recipe i saved",
        "for the fajitas recipe, add that it marinates in lime for 10 minutes before cooking",
    ),
    holds=((_RECIPE_BOX.name, ("fajitas", "orzo", _RECIPE_KEPT)),),
    withholds=(_MARINADE,),
    unspoken=((_RECIPE_BOX.name, _FAJITAS_KEY),),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_change_lands_on_the_entry_that_exists(chat_eval: ChatEval, model: str) -> None:
    """The user names a recipe by a shorter name than the key it is stored under, and the
    change lands on that recipe rather than beside it."""
    cohort = await _drive(chat_eval, model, _UPDATE)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the change is on the entry that was already there, that entry still carries the
    # temperature it held, the box holds the recipes it started with, and the recipe nobody
    # named is exactly as it was.
    cohort.claim(
        "state: the recipe the box already had now carries the marinade",
        _the_entry_carries(_RECIPE_BOX.name, _FAJITAS_KEY, _MARINADE),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: that recipe still carries the 425F it held",
        _the_entry_carries(_RECIPE_BOX.name, _FAJITAS_KEY, _RECIPE_KEPT),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the box holds exactly the recipes it started with (changed on, not beside)",
        _holds_the_keys(_RECIPE_BOX.name, tuple(key for key, _ in _RECIPE_BOX.keyed)),
        SpecCategory.STORE,
    )
    for key, content in _RECIPES_NOT_NAMED:
        cohort.claim(
            f"state: the {key!r} recipe is unchanged",
            _the_entry_is_unchanged(_RECIPE_BOX.name, key, content),
            SpecCategory.STORE,
        )
    cohort.claim(
        "state: the box itself is still the one she was given",
        _the_list_itself_survived(_RECIPE_BOX.name),
        SpecCategory.STORE,
    )

    # PROVENANCE — the marinade is the user's own detail, so a recipe that came back carrying a
    # time, a temperature or an ingredient nobody said was invented into the store for good.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ fan-out ═════════════════════════════════════════════════════════════════
#
# One message carrying TWO facts of opposite sign.  This is the verb where the DESTINATION is
# the claim: two fitting collections already exist, so a correct turn fans the facts out — the
# enthusiasm into the one for things she is into, the complaint into the one for things she
# avoids — rather than folding both into whichever it opened first.

_FAN_OUT = _VerbCase(
    case_id="memory-a-like-and-a-dislike",
    behaviour=(
        "In the chat agent, when one message carries two facts of different kinds and a fitting "
        "list already exists for each, Penny files each fact in the list that fits it and "
        "neither in the other."
    ),
    # ``answers`` is EMPTY for the reason the update case's is: the ask is an instruction, and a
    # confirmation that names neither fact is still a correct answer to "jot these down".
    world=World(name="two lists", pages=(), keeps=(), excludes=(), stores=(_INTO, _AVOID)),
    ask="jot down that I'm into bouldering, and that I can't stand instant coffee",
    also_phrased=(
        "note down that i'm into bouldering and that i can't stand instant coffee",
        "remember that i'm into bouldering, and that i can't stand instant coffee",
        "keep a note that i'm into bouldering, and that instant coffee is something i can't stand",
        "write down that bouldering is something i'm into and that i can't stand instant coffee",
    ),
    holds=((_INTO.name, (_INTO_ANCHOR,)), (_AVOID.name, (_AVOIDED,))),
    withholds=(_LIKE, _DISLIKE),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_like_and_a_dislike_fan_out(chat_eval: ChatEval, model: str) -> None:
    """Two facts in one message, and each belongs somewhere different."""
    cohort = await _drive(chat_eval, model, _FAN_OUT)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(
        "state: the enthusiasm landed in the list for things she is into",
        _filed_in(_INTO.name, _LIKE),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the complaint landed in the list for things she avoids",
        _filed_in(_AVOID.name, _DISLIKE),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the enthusiasm was not ALSO filed with the things she avoids",
        _not_filed_in(_AVOID.name, _LIKE),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the complaint was not ALSO filed with the things she is into",
        _not_filed_in(_INTO.name, _DISLIKE),
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ no-fire ═════════════════════════════════════════════════════════════════
#
# The negative direction, in its own case (#2100).  A message that mentions a subject and asks
# for nothing is answered in conversation, and the round ends where it began: the machine back
# in idle, and whatever the store already held still there.  Those two — the landing, and what
# survives — are what this case asserts.
#
# What it does NOT assert is that she refrained.  Whether a passing mention is worth keeping is
# hers to decide — "I made a good recipe today" is a reasonable thing to remember and a
# reasonable thing to let go — so a claim forbidding a write would be scoring discretion, and
# discretion varies.  It is MEASURED instead: ``ENTRIES_STORED`` and ``TOOL_SEQUENCE`` read
# every sample.  ``TOOL_SEQUENCE`` reads every call the turn made, so it is LIVE on this case
# rather than blind — a sample that answers a message about the store reads the store to do it,
# and the measured runs show exactly that: samples reading while writing nothing.  A cohort that
# answered and moved on agrees on the calls that took, and the sample that went further is the
# spread beside it; ``ENTRIES_STORED`` says the same thing at the store, ``0`` where nothing was
# written and the divergence where something was.
#
# The same reading covers the page: whether a browse happened is a ROUTE, so the tool sequence
# measures it and no assertion names it.

_NO_FIRE_WISTFUL = _VerbCase(
    case_id="memory-no-fire-wistful",
    behaviour=(
        "In the chat agent, when the user muses about something a list she already keeps is "
        "about, Penny answers in conversation and the list she keeps about it is left as it "
        "was."
    ),
    world=World(
        name="a tempting list", pages=(), keeps=(), excludes=(), stores=(_GAMES_WITH_MISTFORGE,)
    ),
    # The temptation IS the setup: a games list holding a strategy game sits in the store while
    # the user muses about finishing a strategy game campaign.  Without it this case measures
    # nothing.  The ask names no game, so what it offers is a topical match and never a
    # reference.
    ask="I finally wrapped up that long strategy game campaign last night, felt so satisfying",
    also_phrased=(
        "finally finished that long strategy game campaign last night — so satisfying",
        "i wrapped up that big strategy game campaign last night, felt really good",
        "last night i finally got to the end of that long strategy game campaign, so satisfying",
        "that long strategy game campaign is finally done as of last night, felt great",
    ),
    holds=((_GAMES.name, (_SUBJECT,)),),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_a_wistful_aside_fires_nothing(chat_eval: ChatEval, model: str) -> None:
    """A games list is sitting right there and the user muses about finishing a game.  A topical
    match is not a request, so the round ends where it began and the list is left as it was."""
    cohort = await _drive(chat_eval, model, _NO_FIRE_WISTFUL)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — what SURVIVES, which is the whole of what a tempting list lets this case assert.
    # Adding to the list is hers; taking something off it is not, and a turn that "tidied" the
    # list on its way past — dropping the entry it read as finished — took something the user
    # kept.  A delete leaves no write behind, so only a read of what the list still HOLDS
    # catches it.
    cohort.claim(
        "state: the list it was tempted by still holds what it held",
        _holds(_GAMES.name, _SUBJECT),
        SpecCategory.STORE,
    )

    # PROVENANCE — both halves.  A sample that did write something into the list may only have
    # written what the user said: a campaign, a title or a date the message never carried is an
    # invention, and it is in the collection for good.
    cohort.assert_every_value_in_the_store_is_sourced()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# Every ported case, in one place — so the deterministic probe in ``test_eval_harness.py`` can
# drive each one's seeder and premise without a GPU.
VERB_CASES = (
    _SAVE,
    _COLD_RECALL,
    _SWEEP,
    _FORGET,
    _UPDATE,
    _FAN_OUT,
    _NO_FIRE_WISTFUL,
)
