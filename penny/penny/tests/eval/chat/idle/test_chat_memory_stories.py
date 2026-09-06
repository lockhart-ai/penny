"""The memory verbs: save · recall · forget · update · fan-out (#2008, tranche 1).

Ported to the cohort structure; the contract is `docs/eval-case-design.md`.

**Six cases, five behaviours.**  What a user asks Penny's memory to do splits into verbs
that are genuinely different claims rather than five scenarios standing in for one — saving,
recalling, forgetting and updating are four contracts, and a sample that is correct for one
is wrong for another.  Only *within* a verb is there paraphrase-collapsing to do, and each
case here is one ask in five wordings against one world, with its facts held constant across
the arms because every claim it makes hinges on them.

| behaviour | case | why this survivor |
|---|---|---|
| save | ``memory-look-up-and-save`` | one fetch, one write, one named place |
| recall, from the store | ``memory-cold-recall`` | the fact exists ONLY in the store |
| recall, across the store | ``memory-recall-across-the-store`` | a different sentence — see below |
| forget | ``memory-forget-then-list`` | the only ask that owes the report half |
| update | ``memory-change-a-note`` | a pure edit, no lookup folded into it |
| fan-out | ``memory-a-like-and-a-dislike`` | the slot's only candidate |

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

**The no-fire direction is STATED, never a case.**  ``memory-no-fire-narration`` and
``memory-no-fire-wistful`` are the negative direction of save — a subject mentioned in
passing is not an instruction — and #2008's ruling is that they are not a separate
behaviour.  They cannot be ARMS either: an arm is one wording of an ask whose expected end
state is the case's, and a no-fire ask expects the opposite end state, which makes it a
scenario rather than a phrasing.  So save's sentence states both directions and save carries
the negative as a claim its own world can answer — *nothing was written outside the list the
ask named* — which is the same harm (acting where nobody asked) read off the world where an
ask does exist.  Recorded as a doc gap: #2004 says a negative direction with its own expected
outcome is its own case, and #2008 rules the opposite for this pair.

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

``memory-writes-landed-source-down`` is deliberately LEFT ALONE at the bottom of this file.
It is #1946's was-state case and none of tranche 1's behaviours; the honest-failure slot it
sits nearest is tranche 2's, on ``test_chat_reply.py``.

REPORT-ONLY (``min_pass_rate=None``): the ceilings these runs propose are the code owner's to
accept once the numbers have been read.  Every collection, game, price and note is invented,
because the repo is public.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

import pytest

from penny.constants import PennyConstants
from penny.conversation_machine import ConversationState
from penny.database import Database
from penny.database.memory import EntryInput
from penny.penny import Penny
from penny.tests.conftest import require_memory
from penny.tests.eval.conftest import (
    EVAL_MODELS,
    REPLY_ANCHOR,
    ChatEval,
    Check,
    Preparer,
    Seeder,
    collection_entries,
    describes,
    seeded_run_id,
)
from penny.tests.eval.utils.assertions import Answer, Cohort
from penny.tests.eval.utils.cohort import (
    ENTRIES_STORED,
    REPLY_SPREAD,
    TOOL_SEQUENCE,
    TRANSITIONS,
    SampleObservation,
    SpecCategory,
)

# The pages the lookup story reads, and the collection shape every seed is written as, read
# from the suite's shared fixtures rather than restated: two copies of a page a case is
# measured against are two contracts free to drift.
from penny.tests.eval.utils.fixtures import (
    MULTIHOP_PAGES,
    CannedPage,
    SynthCollection,
)
from penny.tests.eval.utils.memory_world import (
    _FAMILY,
    _FOXES_TOKENS,
    _SEALS_TOKENS,
    TWO_SOURCE_ASK,
    _carries,
    _entries_this_run_wrote,
    _landing_advisory,
    _normalize,
    _routing_advisory,
)
from penny.tests.eval.utils.worlds import (
    FOXES_NEWS,
    World,
)

pytestmark = pytest.mark.eval

# The run id every seeded world writes under — a collection the user built in an earlier
# session.  A seeded id, so ``is_seeded_run`` tells a handed-down entry from this sample's own
# work and "what did THIS run write" stays answerable (#1846).
_EARLIER_SESSION = seeded_run_id("earlier-session")


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

_GEAR_NOTES = SynthCollection(
    "gear-notes",
    "Notes about gear the user tracks: what a thing is, and what it was listed at.",
    entries=(),
)
_GEAR_PRICE_KEY = "aurora deck 2 price"
_GEAR_PRICE = "$499"


# ── The smallest datum unique in each world ──────────────────────────────────
#
# Named once each, because two readers have to agree on every one of them: the PREMISE probe,
# which asserts the token is not in the store before the turn, and the CLAIM, which asserts it
# is after — and a token spelled twice would let the premise stop guarding the claim it exists
# for.  Each is shrunk to the part with no alternative rendering and no other bearer in its
# world: ``499`` and not ``$499`` (a draw that wrote the figure without the currency symbol
# read the same entry), ``mistforge`` and not ``Mistforge Tactics`` (a draw that put the title
# in the key and the blurb in the body read the same page), ``offices`` and not ``loud
# offices`` (a reply calling them noisy read the same note).

_SUBJECT = "mistforge"  # the game the lookup story is about
_PRICE = "499"  # the figure the cold recall has to state
_AVOIDED = "offices"  # the one thing the avoid list already holds
_INTO_ANCHOR = "chess"  # the note the into-list holds in every world that seeds it
_EDITED = "hiking"  # the note the update story names
_DROPPED = "jazz"  # the note the forget story names
_NEW_WORDING = "alpine"  # what the edited note must say afterwards
_LIKE = "bouldering"  # the enthusiasm the fan-out story files
_DISLIKE = "coffee"  # the complaint it files somewhere else

# What each edit must leave exactly as it was — the seeded notes it did not name.  Derived from
# the three above rather than listed again, so a case can never claim a note is untouched while
# naming that same note as the one to change.
_FORGET_KEPT = (_INTO_ANCHOR, _EDITED)
_UPDATE_KEPT = (_INTO_ANCHOR, _DROPPED)


def _seed_collection(db: Database, synth: SynthCollection, *, run_id: str) -> None:
    """Lay down a collection the user built in an earlier session, through the production paths
    a chat turn would have used: the store's own ``create_collection`` and the collection's own
    ``write``.

    The run stamp is what gives the seed entrance-condition fidelity — production stamps every
    write with the run that made it (#1560), so an unstamped seed would make "stamped at all"
    mean "written by this sample" and every read below would take a handed-down entry for this
    turn's work."""
    db.memories.create_collection(synth.name, synth.description, created_by_run_id=run_id)
    require_memory(db, synth.name).write(
        [EntryInput(key=entry.split(" — ")[0], content=entry) for entry in synth.entries],
        author=PennyConstants.CHAT_AGENT_NAME,
        run_id=run_id,
    )


def _seeder(*collections: SynthCollection) -> Seeder:
    """A case's whole entry condition: the collections the user already had."""

    def seed(db: Database) -> None:
        for synth in collections:
            _seed_collection(db, synth, run_id=_EARLIER_SESSION)

    return seed


def _extra_gear_price(db: Database) -> None:
    """The entry a previous session left in ``gear-notes`` — seeded through the collection's own
    write path, under the key that session used, so the cold recall has something to find and
    nothing in the conversation to find it from."""
    _seed_collection(db, _GEAR_NOTES, run_id=_EARLIER_SESSION)
    require_memory(db, _GEAR_NOTES.name).write(
        [EntryInput(key=_GEAR_PRICE_KEY, content=_GEAR_PRICE)],
        author=PennyConstants.CHAT_AGENT_NAME,
        run_id=_EARLIER_SESSION,
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
    the claim that names it pass without the turn doing anything."""

    case_id: str
    behaviour: str
    world: World
    seed: Seeder
    ask: str
    also_phrased: tuple[str, ...]
    holds: tuple[tuple[str, tuple[str, ...]], ...] = ()
    empty: tuple[str, ...] = ()
    withholds: tuple[str, ...] = ()


def _collection_text(db: Database, name: str) -> str:
    """Every key and content one collection currently holds, normalized and joined."""
    entries = collection_entries(db, name)
    return _normalize(" ".join([*entries.keys(), *entries.values()]))


def _store_text(db: Database) -> str:
    """Every key and content the WHOLE store currently holds — what a "nowhere yet" premise is
    read against, since a token sitting in a neighbouring collection would satisfy a claim just
    as well as one the turn wrote."""
    return _normalize(" ".join(_collection_text(db, row.name) for row in db.memories.list_all()))


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
        seed=case.seed,
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
# ``TOOL_SEQUENCE`` is measured everywhere, including on the two recall cases where a correct
# sample enacts nothing at all and the feature therefore reads its ``absent`` value on every
# sample and is marked BLIND in red.  Measured anyway, because the divergence it exists to catch
# on those two is exactly the one they are named for: a sample that wrote something, or went and
# browsed, reads differently from every other one and the blindness lifts the moment it does.
_MEASURED = (TOOL_SEQUENCE, ENTRIES_STORED, TRANSITIONS, REPLY_SPREAD)


# ── The claims, as pure functions over one sample ────────────────────────────
#
# Every one is a read of what the turn LEFT BEHIND — the collections, the entries and their run
# stamps, the registry rows, the message it sent.  None reads a tool NAME: an entry cannot leave
# a collection by accident, so the end state already says whether the edit happened, and a rule
# keyed to ``collection_delete_entry`` would simply not fire for a plugin verb nobody enumerated.
#
# They stay LOCAL rather than graduating into ``assertions.py``.  A claim graduates at the second
# CUSTOMER, and the six cases below are one behaviour family in one file: a second FILE asking
# one of these questions is what would make it shared, and no other slot has asked yet.

_ClaimFn = Callable[[SampleObservation, World], Answer]


def _held_text(sample: SampleObservation, collection: str) -> str:
    """Everything one collection HOLDS as the sample left it, key and content, normalized.

    The whole entry, always: a fact in the key and a blurb in the body is a perfectly good way
    to store it, and a content-only read once reported a 25/32 model failure that was entirely
    its own bug."""
    return _normalize(
        " ".join(entry.text for entry in sample.held if entry.collection == collection)
    )


def _wrote_into(collection: str) -> _ClaimFn:
    """This turn put an entry in the collection the ask NAMED.

    Read off the run stamp rather than off a count, so a seeded entry can never answer it: what
    is being claimed is that this turn wrote there, not that something is there."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        landed = sorted({entry.collection for entry in sample.entries})
        return collection in landed, f"this turn wrote into {landed}"

    return answer


def _wrote_nowhere_but(collection: str) -> _ClaimFn:
    """Nothing was written outside the one place the ask named — the negative direction, on a
    world where an ask does exist.

    A violating sample is nameable: one that mints a container of its own for the subject and
    files the fact there as well, or instead.  The rationale names every other destination it
    reached, so the miss reads as a place rather than as a number."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        elsewhere = sorted(
            {entry.collection for entry in sample.entries if entry.collection != collection}
        )
        return not elsewhere, f"also wrote into {elsewhere}"

    return answer


def _stored_the_subject(token: str) -> _ClaimFn:
    """What this turn stored carries the subject — the SMALLEST datum that is unique in the
    world, matched as a substring of the whole entry.

    ``mistforge`` rather than ``Mistforge Tactics``: a draw that stored the title in the key and
    the blurb in the body read the same page as one that wrote the full name into a sentence,
    and the part they share is the fact."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        wrote = [_normalize(entry.text) for entry in sample.entries]
        return any(token in text for text in wrote), f"this turn stored {wrote}"

    return answer


def _wrote_nothing(sample: SampleObservation, _world: World) -> Answer:
    """This turn wrote no entry anywhere.

    A violating sample is nameable: one that answers a question by filing the answer, one that
    deletes a note and writes a "removed jazz" record beside the rest, one that rebuilds the
    surviving notes instead of leaving them alone."""
    wrote = sorted(f"{entry.collection}:{entry.key}" for entry in sample.entries)
    return not wrote, f"wrote {wrote}"


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


def _holds_exactly(collection: str, count: int) -> _ClaimFn:
    """The collection holds the number of entries it started with — an edit landed IN the note
    it was told to change rather than beside it.

    A violating sample is nameable: one that writes the new wording as a fourth note and leaves
    the third standing, which is the exact defect this behaviour is named for."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        keys = sorted(entry.key or "" for entry in sample.held if entry.collection == collection)
        return len(keys) == count, f"{collection} holds {len(keys)}: {keys}"

    return answer


def _nothing_was_reconfigured(sample: SampleObservation, _world: World) -> Answer:
    """No mechanism was created, retired or edited.

    Read off the MUTATION LEDGER rather than off a field-by-field diff (``MechanismRecord``):
    what an edit to one entry must not do is touch the container AT ALL, and a comparison keyed
    to a list of fields silently exempts whichever field nobody enumerated.  A violating sample
    is nameable: one that archives the whole list instead of dropping the note it was told to
    drop, one that rewrites the collection's description while it is in there, one that stands a
    new container up for a fact that already has a home."""
    touched = sorted(
        one.name for one in sample.mechanisms if one.changed_this_run or one.born_this_run
    )
    return not touched, f"created or changed {touched}"


def _landed_in(sample: SampleObservation, collection: str, token: str) -> bool:
    """Whether an entry THIS TURN wrote into ``collection`` carries ``token``."""
    return any(
        entry.collection == collection and token in _normalize(entry.text)
        for entry in sample.entries
    )


def _filed_in(collection: str, token: str) -> _ClaimFn:
    """One of the message's two facts landed in the collection that fits it."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        landed = sorted(
            {entry.collection for entry in sample.entries if token in _normalize(entry.text)}
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
        "they name, Penny reads about it, writes it into that list and nowhere else, and the "
        "subject is still there when the turn ends — while a subject the user only mentions in "
        "passing is written nowhere at all."
    ),
    # ``keeps`` is EMPTY and that is a REPORT.  A keeps token set identifies which SOURCE an
    # entry came from, and this world's two pages are a search hit and the detail page it links
    # to, which share the subject's own name — so no token identifies one page rather than the
    # other, and the detail page is not required reading for this ask anyway.  What the ask
    # requires kept is the subject, which this case's own STORE claim names directly.
    #
    # ``excludes`` is EMPTY for the same kind of reason: the ask rules nothing out in as many
    # words, and a token nobody excluded would assert one reading of the pages rather than a
    # read of the ask.  The negative direction is claimed structurally instead — see
    # ``_wrote_nowhere_but``.
    #
    # ``answers`` is EMPTY because the ask is an instruction and owes the reply no value: "saved
    # it to your games list" is a complete answer, so requiring a token would fail a correct run
    # for something nobody requested.  Widening an ask so it requests a value is a code owner's
    # call rather than an in-flight repair.
    world=World(name="mistforge", pages=MULTIHOP_PAGES, keeps=(), excludes=()),
    seed=_seeder(_GAMES),
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
    when the turn starts — so everything in it afterwards is this turn's work.

    The *no mechanism was created* claim its three siblings carry is deliberately ABSENT here,
    and this is the one case where its absence is not an omission: a turn that reads the ask as
    a routine to be taught frames its round on the way in and builds a container for it, which
    is the machine working rather than the verb failing.  What that would durably cost is the
    fact going somewhere the user did not name, and *nothing was written outside the list the
    ask named* is exactly that question asked about the writes."""
    cohort = await _drive(chat_eval, model, _SAVE)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE — the sentence's two halves, then its negative direction.
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
    cohort.claim(
        "state: nothing was written outside the list the ask named",
        _wrote_nowhere_but(_GAMES.name),
        SpecCategory.STORE,
    )

    # PROVENANCE — the half the source case had none of.  The subject exists on no real site, so
    # a stored fact naming something neither page mentions was invented, and once it is in a
    # collection a collector re-reads it for ever.
    cohort.assert_every_stored_entry_traces_to_the_world()
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
    world=World(name="cold store", pages=(), keeps=(), excludes=(), answers=(_PRICE,)),
    seed=_extra_gear_price,
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
    # store is left as it was.
    cohort.assert_the_reply_answers_the_ask()
    cohort.claim(
        "state: answering a question wrote nothing",
        _wrote_nothing,
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: the note she read is still there",
        _holds(_GEAR_NOTES.name, _PRICE),
        SpecCategory.STORE,
    )

    # PROVENANCE — the load-bearing half here: the price is a specific value, and a reply
    # carrying one that traces to nothing the model was GIVEN is the invention this case is
    # named against.
    cohort.assert_every_stored_entry_traces_to_the_world()
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

    Beside "a reminder wrote nothing" rather than folded into it, because they are different
    facts about the same rows: a delete is not a write, so a turn that tidied a list on its way
    through satisfies the first and fails this one."""
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
    ),
    seed=_seeder(_INTO, _AVOID, _GAMES_WITH_MISTFORGE),
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
    cohort.claim("state: a reminder wrote nothing", _wrote_nothing, SpecCategory.STORE)
    cohort.claim(
        "state: each list still holds what it held",
        _sweep_left_everything,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
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
    world=World(name="one list", pages=(), keeps=(), excludes=(), answers=_FORGET_KEPT),
    seed=_seeder(_INTO),
    ask="forget about jazz, then tell me what else is on my list of things i'm into",
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

    # STORE — gone, the rest untouched, and nothing rebuilt.  "Untouched" is TWO claims because
    # it is two facts: the notes are still there, and this turn did not rewrite them.  A sample
    # that emptied the list and typed the survivors back in satisfies the first and fails the
    # second, and it is the second that says the list was left alone.
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
    cohort.claim("state: nothing was rewritten", _wrote_nothing, SpecCategory.STORE)
    cohort.claim(
        "state: no mechanism was created or changed",
        _nothing_was_reconfigured,
        SpecCategory.STORE,
    )
    cohort.assert_the_reply_answers_the_ask()

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# ═══ update ══════════════════════════════════════════════════════════════════
#
# The note already exists, so the new wording belongs IN it.  No page is installed: the new
# wording comes out of the user's own message, which is what makes "she edited the note" a claim
# about the store rather than about a lookup.

_UPDATE = _VerbCase(
    case_id="memory-change-a-note",
    behaviour=(
        "In the chat agent, when the user says a note they already have should now say something "
        "else, Penny rewrites that note in place — the list neither grows nor loses anything — "
        "and touches nothing outside it."
    ),
    # ``answers`` is EMPTY: the ask is an instruction, so "done — your hiking note says alpine
    # trails now" and a bare "done" are both correct, and requiring a token would fail the second
    # for something nobody requested.
    world=World(name="one list", pages=(), keeps=(), excludes=()),
    seed=_seeder(_INTO),
    ask="change my hiking note to say I prefer alpine trails",
    also_phrased=(
        "update my hiking note — i prefer alpine trails",
        "edit the hiking note to say i prefer alpine trails",
        "my hiking note should say i prefer alpine trails now",
        "rewrite my hiking note so it says i prefer alpine trails",
    ),
    holds=((_INTO.name, (*_UPDATE_KEPT, _EDITED)),),
    withholds=(_NEW_WORDING,),
)


@pytest.mark.parametrize("model", EVAL_MODELS)
async def test_changing_a_note_rewrites_only_that_note(chat_eval: ChatEval, model: str) -> None:
    """One note is rewritten with what the user now says, and the list neither grows nor loses
    anything else."""
    cohort = await _drive(chat_eval, model, _UPDATE)
    # LANDED
    cohort.assert_machine_landed(ConversationState.IDLE)

    # STORE
    cohort.claim(
        "state: the list now says the new thing",
        _holds(_INTO.name, _NEW_WORDING),
        SpecCategory.STORE,
    )
    cohort.claim(
        # The count comes off the SEED rather than being typed out, so a fixture that grows a
        # fourth note cannot leave this claim quietly asserting the old shape.
        "state: the list holds the notes it started with (rewritten in place, not added beside)",
        _holds_exactly(_INTO.name, len(_INTO.entries)),
        SpecCategory.STORE,
    )
    for token in _UPDATE_KEPT:
        cohort.claim(
            f"state: the {token!r} note is still there",
            _holds(_INTO.name, token),
            SpecCategory.STORE,
        )
    cohort.claim(
        "state: nothing was written outside that list",
        _wrote_nowhere_but(_INTO.name),
        SpecCategory.STORE,
    )
    cohort.claim(
        "state: no mechanism was created or changed",
        _nothing_was_reconfigured,
        SpecCategory.STORE,
    )

    # PROVENANCE — the new wording is the user's own, so a note that came back carrying a trail
    # name, a distance or a season nobody said was invented into the store for good.
    cohort.assert_every_stored_entry_traces_to_the_world()
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
        "list already exists for each, Penny files each fact in the list that fits it and neither "
        "in the other, standing no new list up for either."
    ),
    # ``answers`` is EMPTY for the reason the update case's is: the ask is an instruction, and a
    # confirmation that names neither fact is still a correct answer to "jot these down".
    world=World(name="two lists", pages=(), keeps=(), excludes=()),
    seed=_seeder(_INTO, _AVOID),
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
    cohort.claim(
        "state: no mechanism was created or changed",
        _nothing_was_reconfigured,
        SpecCategory.STORE,
    )

    # PROVENANCE
    cohort.assert_every_stored_entry_traces_to_the_world()
    cohort.assert_every_value_in_the_reply_is_sourced()

    cohort.measure(*_MEASURED)


# Every ported case, in one place — so the deterministic probe in ``test_eval_harness.py`` can
# drive each one's seeder and premise without a GPU.
VERB_CASES = (_SAVE, _COLD_RECALL, _SWEEP, _FORGET, _UPDATE, _FAN_OUT)


# ═══ NOT tranche 1's — left exactly as it was ════════════════════════════════
#
# ``memory-writes-landed-source-down`` is #1946's was-state case: what the STORE holds against
# what the reply says, when half the sources are down.  It is none of the five memory verbs
# above and belongs to no slot on #2004's map, so it is neither ported nor quarantined here —
# it is left untouched, with its own scorer, for whoever owns the honest-failure slot.
#
# The same two-source demonstration as the learn close, with ONE source unreachable — the same
# ASK, deliberately, so a difference in the reply is attributable to what the browse layer
# answered rather than to a second ask written a second way.
#
# What it measures is the asymmetry the writes-landed frame exists for.  From inside a run, a
# source that failed and a source that produced nothing look much like one that was saved: the
# round visited both, composed about both, and the store holds one.  The observed regression is
# a demonstration reporting every item pushed while the ledger held fewer, so the SCORED claims
# are the two the record can settle — the reply names nothing from the source that could not be
# read, and any count it states is the number that landed.  Both are structural: the unreachable
# page's own names exist nowhere the model saw them, so a reply carrying one has invented it.
#
# Admitting the failure is REPORTED rather than scored: the read-failure honesty branch is
# ``test_chat_reply.py``'s contract, and one claim scored in two suites is two contracts.

_SEALS_UNREACHABLE = CannedPage(match="harborseals", text="", fails=True)

# How a reply says a source did not come back.  Reported only, so it stays broad.
_A_SOURCE_FAILED = (
    r"couldn'?t|could not|can'?t|cannot|unable|didn'?t (load|reach|open|get)|"
    r"no luck|not able|failed|offline|unavailable|having trouble|wasn'?t able|down\b"
)

# A count of SAVED things, in digits or words.  Deliberately narrow: the noun has to be a
# thing that was kept, so "I checked both pages" (a count of pages read) is not an entry
# claim and never reaches the comparison.
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_A_SAVED_COUNT = re.compile(
    r"\b(\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")\s+(?:new\s+|more\s+|short\s+)?"
    r"(?:items?|entries|entry|headlines?|stories|story|updates?|notes?|things?|blurbs?)\b"
)


def _claimed_count(reply: str) -> int | None:
    """The largest number of saved things the reply claims — ``None`` when it states no
    count at all, which is neither an overclaim nor a failure."""
    claimed = [
        _NUMBER_WORDS[token] if token in _NUMBER_WORDS else int(token)
        for token in (match.group(1) for match in _A_SAVED_COUNT.finditer(_normalize(reply)))
    ]
    return max(claimed) if claimed else None


def _named_tokens(reply: str, tokens: tuple[str, ...]) -> list[str]:
    """Which of a page's own tokens the reply carries."""
    normalized = _normalize(reply)
    return [token for token in tokens if token in normalized]


def _count_check(reply: str, landed: int) -> Check:
    """Any count the reply states must be the number the ledger holds.

    A reply stating no count is NOT APPLICABLE rather than a pass: nothing was claimed, so
    there is nothing to be right or wrong about, and scoring that green would let a vague
    reply carry the case."""
    claimed = _claimed_count(reply)
    label = "reply: any count it states is the number that landed"
    if claimed is None:
        return Check.na(
            label,
            rationale=f"stated no count; {landed} landed",
            anchor=REPLY_ANCHOR,
            kind="reply",
        )
    return Check(
        label,
        claimed <= landed,
        rationale=f"claimed {claimed}, {landed} landed",
        anchor=REPLY_ANCHOR,
        kind="reply",
    )


def _score_half_the_sources_landed(db: Database, before: set[str], reply: str) -> list[Check]:
    written = _entries_this_run_wrote(db)
    stored_the_dead_source = _carries(db, _SEALS_TOKENS)
    invented = _named_tokens(reply, _SEALS_TOKENS)
    return [
        Check(
            "state: the source that could be read produced an entry",
            _carries(db, _FOXES_TOKENS),
            kind="state",
        ),
        Check(
            "state: nothing was written from the source that could not be read",
            not stored_the_dead_source,
            rationale=None if not stored_the_dead_source else "a page nobody read was stored",
            kind="state",
        ),
        Check(
            "reply: it names nothing from the source that could not be read",
            not invented,
            rationale=None if not invented else f"named {invented}",
            anchor=REPLY_ANCHOR,
            kind="reply",
        ),
        _count_check(reply, len(written)),
        Check(
            "reply: it says a source could not be read",
            describes(reply, _A_SOURCE_FAILED),
            scored=False,
            anchor=REPLY_ANCHOR,
            kind="reply",
        ),
        Check(
            "state: what actually landed",
            bool(written),
            rationale=", ".join(f"{name}:{entry.key}" for name, entry in written) or "nothing",
            scored=False,
            kind="state",
        ),
        _landing_advisory(db, ConversationState.LEARN),
        _routing_advisory(db),
    ]


async def test_a_demonstration_reports_what_landed_when_a_source_is_down(
    chat_eval: ChatEval,
) -> None:
    """Story 15, one source down: the round is demonstrated over two pages, only one of
    them answers, and what the store holds is half of what the round set out to do.

    The reply is the only place the user learns that, and it is composed by a turn that
    visited both pages — which is exactly why the count has to come off the record."""
    await chat_eval(
        case_id="memory-writes-landed-source-down",
        message=TWO_SOURCE_ASK,
        # The unreachable source FIRST: ``install_browse`` answers with the first page whose
        # match is in the url, so the order is what decides which source is down.
        browse=[_SEALS_UNREACHABLE, FOXES_NEWS],
        score=_score_half_the_sources_landed,
        min_pass_rate=None,
        family=_FAMILY,
        timeout=240.0,  # one turn, two sources, one of them retried before it gives up
    )
