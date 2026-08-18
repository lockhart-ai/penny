"""Framing a round at its ENTRY, and the container it runs into (#1868/#1870, epic #1866).

The framer used to run at the END of a learn turn, over the same user turns it reads
here — which meant the routine's identity was settled only after the round had already
happened, so the round itself had nowhere to put what it found and the model named a
collection by judgment.  That judgment is what this module deletes: identity is the skill
plus the values the user said, both of them literal spans of the user's own words, and a
name derived from them (``derive_collection_name``) is the same name every time the same
job is asked for.

A round reaches that identity by one of TWO draws, and which one is decided by what the
round already has.  A round being TAUGHT has no routine yet, so the framer MINTS one from
the ask (:meth:`RoundFramer.frame_entry`).  A round that asks for a routine Penny ALREADY
KNOWS has one, so nothing is minted and the only open question is which part of the user's
words fills each thing that routine already needs — the BINDER's question
(:meth:`RoundFramer.bind_entry`, #1870).  Both end at the same place: a signature whose
parameters carry values, a name derived from it, and a container found or created under
that name — so what happens after them cannot tell which one ran.

So the draw moves to the moment the machine LANDS in the state — the ask and the
demonstration both exist there, which is the framer's whole input contract — and Python
does the rest deterministically:

* **derive** the container's name from the framed skill and its demonstrated values;
* **find-or-create** it, INERT — storage only, no schedule, no notify, no program — so
  nothing runs against it and the round has somewhere to write;
* **hand it back** as :class:`~penny.conversation_machine.RoundFraming`, which the machine
  records on the move as the round's own state.

**Certified-by-execution is untouched.**  Nothing here writes a skill row: the registry
still only gains a skill at run end, from the ledger of what actually ran.  What the
entry settles is the round's framing and its container, which is why a round that fails
(#1839's honest learn-failure) takes its empty container with it — see
:func:`discard_round_container`.

**The container's lifecycle is find-or-create, never mint-per-round.**  Re-asking for the
same job derives the same name and finds the container that exists — tier-1 dedup by
construction (#1775).

**A round's identity is settled ONCE, at its first entry** (:meth:`RoundFramer.carry_entry`,
#1902).  A correction re-enters learn to refine the PROGRAM of the round's one job, so the
re-entry makes no draw at all and the framing the round already holds carries: the same
routine, the same container, the corrected write landing in place.  The re-draw this
replaces derived a fresh name from a corrected ask, found no container under it, minted a
sibling, and left run-end extraction filing a second routine beside the first.

**A round that ENDS in idle takes its container AND what it wrote to the registry with it**
(:func:`abandon_round_container` / :func:`abandon_round_skill`, #1896/#1902).  A bail
preserves nothing — once the machine is idle the round is over and any next task opens a
flow of its own — so the one container retirement in this module that is NOT guarded on
emptiness is this one: what the demonstration wrote is the round's own intermediate state
rather than an exception to it.  The container is archived (a tombstone, like every other
retirement here).  The routine is resolved by the round's PROVENANCE, recorded when the
round's routine was MINTED (:func:`snapshot_replaced_skill`): a name the round minted over
nothing is deleted, a routine the round was RE-TEACHING is RESTORED to the version the
round found — because the round's own extraction already replaced what it does, and
un-deleting is not the same as putting it back — and a round that minted nothing at all,
having only bound a routine the user already had, leaves the registry alone.
"""

from __future__ import annotations

import logging
from base64 import b64decode, b64encode
from typing import TYPE_CHECKING

from penny.constants import MutationAction, MutationActor, MutationEntityType, PennyConstants
from penny.conversation_machine import (
    CandidateParameter,
    ReplacedSkill,
    RoundEntry,
    RoundFraming,
    RoundProvenance,
    RoundShortfall,
)
from penny.database import Database
from penny.database.models import Skill
from penny.database.mutation_store import MutationDetail
from penny.database.skill_store import parameters_from_json
from penny.database.skills import (
    SkillParameter,
    build_binding_content,
    derive_collection_name,
    render_spoken_turns,
)
from penny.llm.similarity import embed_text
from penny.skill_extraction import build_framing_content
from penny.tools.micro_context import (
    BoundValues,
    FramedParameter,
    MicroContext,
    MissingParameters,
    SkillBinding,
    SkillSignature,
)

if TYPE_CHECKING:
    from penny.llm.client import LlmClient

logger = logging.getLogger(__name__)

# What the mutation ledger records about a round's container — the cause a ``memory`` row
# cannot state for itself, in the three shapes it happens in.  Private: every caller of
# these is in this file, and what a retirement MEANS is this module's own vocabulary.
_ARCHIVED_ROUND_FAILED = "the round that created this container taught nothing"
_REVIVED_SAME_JOB = "the same job is being taught again, into the container it already had"
_ABANDONED_ROUND = "the round that created this container was called off"

# And what it records about a round's ROUTINE, in the two shapes a bail takes it back in.
_ABANDONED_DRAFT = "the round that taught this routine was called off"
_RESTORED_PRE_ROUND = "the round re-teaching this routine was called off, so it is back as it was"


class RoundFramer:
    """Frames a round at its entry and settles the container it runs into.

    Two entries, two draws, one outcome: :meth:`frame_entry` MINTS a routine for a round
    being taught, :meth:`bind_entry` FILLS one the registry already holds (#1870), and
    both hand back the same :class:`~penny.conversation_machine.RoundFraming` over a
    container settled by the same find-or-create rule.  The binder has one further answer
    the framer cannot have (#1885): the routine covers the ask and the user's words are
    SHORT of something it needs, which comes back as a
    :class:`~penny.conversation_machine.RoundShortfall` and routes the turn into request.

    :meth:`carry_entry` is the THIRD entry and the one that draws nothing (#1902): a round
    re-entering learn already settled what it is about, so its framing carries unchanged
    and only its container is re-settled.

    :meth:`bind_entry` is the ONE door for every entry against a known routine (#1894) —
    an apply the words fall short of and a request the classifier drew directly both come
    through it, so both present the same partial binding rather than one of them getting
    specifics and the other a generic ask.  A round that comes back for its missing detail
    hands over what it already settled, and only the still-open parameters are drawn.

    One instance per deployment, holding its clients and database (threaded, never
    ambient).  Injected into :class:`~penny.conversation_machine.ConversationMachine`,
    which owns WHEN each of these runs; what a framed round then IS lives here.
    """

    def __init__(
        self,
        db: Database,
        model_client: LlmClient,
        embedding_client: LlmClient,
        *,
        run_target: str = PennyConstants.CHAT_AGENT_NAME,
    ) -> None:
        self._db = db
        self._micro_context = MicroContext(model_client)
        self._embedding = embedding_client
        # Whose runs these draws belong to, for ledger attribution — the chat agent, since
        # the draw decides how that agent's very next turn is instructed.
        self._run_target = run_target

    async def frame_entry(
        self, *, ask: str | None, message: str, run_id: str | None
    ) -> RoundFraming | None:
        """Frame the round the machine is entering FOR THE FIRST TIME, and settle its
        container — the summary method.

        Draws the interface from the round's user turns, derives the container's name from
        it, finds-or-creates that container, and returns the framing for the machine to
        record.  A round that already carries a framing never reaches here (#1902): it is
        :meth:`carry_entry`'s, because a round's identity is settled once.

        ``None`` when the draw failed: NO container is built, the gate is logged, and
        everything downstream degrades to what it did before this existed — the round runs
        unframed, and run-end extraction frames it the way it always has."""
        signature = await self._draw(ask, message)
        if signature is None:
            logger.warning(
                "The round entering learn could not be framed — no container was built, "
                "so this round runs unframed and is framed at run end instead"
            )
            return None
        framing = RoundFraming(signature=signature, container=container_name(signature))
        await self._settle_container(framing, run_id=run_id)
        return framing

    async def carry_entry(self, framing: RoundFraming, *, run_id: str | None) -> None:
        """The round's framing CARRIED into a re-entry, with no draw at all (#1902).

        Returns nothing, because nothing is settled that the machine does not already
        hold: the framing on the round's newest move IS the answer, and handing back a
        copy would make a carried move indistinguishable from one that settled something —
        which is exactly the distinction the round's provenance is read against.

        A correction re-enters learn to refine the PROGRAM of the round's one job, never to
        decide what that job is: the round settled that at its first entry, the turn that
        demonstrated it was instructed under that name, and the container derived from it
        is where the round has been writing.  Asking again is what forked a round in two —
        a corrected ask read as a different subject, a fresh draw derived a fresh name,
        find-or-create minted a sibling under it, and run-end extraction registered a
        second routine beside the first.

        The CONTAINER is still settled, because the round may have lost it since: a learn
        turn that taught nothing takes it with the failure
        (:func:`discard_round_container`), and the correction that follows is exactly the
        round coming back for it.  Find-or-create is the same rule both draws end on, so
        the round always has somewhere to write."""
        await self._settle_container(framing, run_id=run_id)

    async def _draw(self, ask: str | None, message: str) -> SkillSignature | None:
        """One framing draw over the round's USER turns — the ask it is anchored to and
        the message that just arrived, rendered by the SHIPPED renderer so the document
        this reads is the one every other framing customer reads.

        An unanchored entry (teaching that arrives unprompted, straight from idle) has
        only the one turn, which is the floor the framer already handles."""
        conversation = [(PennyConstants.MessageDirection.INCOMING, ask)] if ask else []
        content = build_framing_content(message, conversation)
        return await self._micro_context.frame_skill(content, run_target=self._run_target)

    async def bind_entry(
        self,
        *,
        skill: str,
        ask: str | None,
        message: str,
        run_id: str | None,
        settled: dict[str, str] | None = None,
    ) -> RoundEntry | None:
        """Frame the round an entry is entering against a routine that ALREADY EXISTS
        (#1870/#1894) — the summary method of the binder half, and the ONE door both a
        cold apply and a request entry come through.

        The classifier bound ``skill`` when it decided, so nothing here decides what the
        round is about: the routine is settled, and the only open question left is which
        part of the user's words fills each thing that routine already declares.  The
        binder answers exactly that, the container's name is derived from the answer, and
        find-or-create does the rest — which IS tier-1 dedup by construction (#1775): the
        same job asked for a second time derives the name it derived the first time and
        runs into the container that already exists, while a different place mints its own.

        ``settled`` is what a PARKED round already bound (#1894) — read off its recorded
        shortfall rather than re-derived — so a round coming back for its missing detail
        draws only the parameters still open and keeps the values the earlier turn read out
        of the earlier words.  Empty for a cold entry, which is every parameter open.

        A :class:`~penny.conversation_machine.RoundShortfall` when the words fell SHORT of
        something the routine needs (#1885) — the routine covers the ask, so the round is
        not failed, it is turned into the ask for the rest.  Nothing is built for it: the
        container's name is derived from every value, so a job missing one has no name yet.

        ``None`` when the round cannot be settled at all — the routine is not in the
        registry, or no usable draw came back.  No container is built either way, so
        nothing is left behind by a round that never started, and the apply turn that
        follows has nothing to configure — which its own state fails honestly rather than
        inventing one (#1875)."""
        routine = self._db.skills.get(skill)
        if routine is None:
            logger.warning(
                "The round bound %r, which the registry does not hold — no container was "
                "built, so this round enters its turn unframed",
                skill,
            )
            return None
        declared = parameters_from_json(routine.parameters)
        already = _already_settled(declared, settled)
        return await self._settle_binding(routine, declared, already, ask, message, run_id=run_id)

    async def _settle_binding(
        self,
        routine: Skill,
        declared: list[SkillParameter],
        already: dict[str, str],
        ask: str | None,
        message: str,
        *,
        run_id: str | None,
    ) -> RoundEntry | None:
        """Fill what is still OPEN and settle the round on the result (#1894).

        A round whose every parameter is already settled needs no draw at all — its values
        are read, not re-decided — so the framing is built straight from them; asking a
        model to re-answer a question Python already holds the answer to is the model-space
        detour this whole seam exists to remove."""
        open_parameters = [parameter for parameter in declared if parameter.name not in already]
        if not open_parameters:
            return await self._framed(routine, declared, already, run_id=run_id)
        bound = await self._bind(routine, open_parameters, ask, message)
        if not isinstance(bound, BoundValues):
            return _shortfall(routine, declared, already, bound)
        return await self._framed(routine, declared, already | bound.values, run_id=run_id)

    async def _framed(
        self,
        routine: Skill,
        declared: list[SkillParameter],
        values: dict[str, str],
        *,
        run_id: str | None,
    ) -> RoundFraming:
        """The round settled: ``routine``'s own interface carrying ``values``, and the
        container derived from it, found or created."""
        signature = _filled_signature(routine, declared, values)
        framing = RoundFraming(signature=signature, container=container_name(signature))
        await self._settle_container(framing, run_id=run_id)
        return framing

    async def _bind(
        self, routine: Skill, open_parameters: list[SkillParameter], ask: str | None, message: str
    ) -> SkillBinding | None:
        """One BINDING draw over the round's user turns, in the binder's own typed union —
        which of its two directions came back is the caller's to read, because the two
        settle the round differently.

        The draw is handed the turns twice over — once inside the rendered document, which
        also renders the signature, and once on their own — because a value is only
        evidence when the USER said it, and the second argument is the text the span check
        actually tests against (#1867).

        Only the OPEN parameters are rendered and offered (#1894): the draw's contract is
        that it answers for exactly the parameters it was handed, so a parameter a parked
        round already settled is not a question to ask again — it is an answer to carry."""
        spoken = render_spoken_turns(_spoken_turns(ask, message))
        content = build_binding_content(spoken, routine.name, routine.description, open_parameters)
        return await self._micro_context.bind_skill(
            content,
            [parameter.name for parameter in open_parameters],
            spoken,
            run_target=self._run_target,
        )

    async def _settle_container(self, framing: RoundFraming, *, run_id: str | None) -> None:
        """Find-or-create the round's container — the one rule every entry ends on.

        The same job asked for again derives the same name and finds the container that
        already exists: nothing is created, and the round continues into what it was
        already writing (tier-1 dedup by construction, #1775).

        Find-or-create is also what makes a container STRANDED by a failure harmless: if
        the turn dies between this step and the machine recording the move, what is left
        behind is an inert, empty collection carrying this job's own name — which the next
        attempt at the same job finds and continues into, rather than an orphan nothing can
        reach."""
        existing = self._db.memories.get(framing.container)
        if existing is None:
            await self._create(framing, run_id=run_id)
        elif existing.archived:
            # The same job, taught again after an earlier round was discarded: its
            # container is the one that already carries this job's name, so it comes back
            # rather than being shadowed by a second row nobody can reach — with its cause
            # on the ledger, like every other move this module makes.
            self._db.memories.unarchive(
                framing.container,
                actor=MutationActor.SYSTEM,
                run_id=run_id,
                note=_REVIVED_SAME_JOB,
            )

    async def _create(self, framing: RoundFraming, *, run_id: str | None) -> None:
        """Create the round's container INERT, through the store's own create chokepoint —
        storage with no job attached: no program, no schedule, no notify, so the dispatcher
        (which selects on a rendered program) never runs anything against it (#1629).

        Its description is the framer's own one line of what the routine is for, which is
        also the registry's meaning anchor — so what the container is FOR and what the
        routine is for cannot say two different things.  The creating run is stamped, and
        the channel links the provoking message to it afterwards by that same id (#1566)."""
        description = framing.signature.description
        embedding = await embed_text(self._embedding, description)
        self._db.memories.create_collection(
            framing.container,
            description,
            description_embedding=embedding,
            created_by_run_id=run_id,
        )
        logger.info(
            "Framed the round as %r and built its container %r",
            framing.signature.name,
            framing.container,
        )


def container_name(signature: SkillSignature) -> str:
    """The container's derived name: the skill plus its demonstrated values, in declared
    parameter order — the one place this module says how a job is identified.

    PUBLIC because a seeded world has to build the container a real round would have
    built: the transition suites lay down rounds that already happened, and a fixture
    deriving the name its own way would be a second copy of the scheme, free to drift from
    the one production actually names jobs with."""
    return derive_collection_name(
        signature.name, [parameter.value for parameter in signature.parameters]
    )


def _spoken_turns(ask: str | None, message: str) -> list[str]:
    """The round's USER turns, in the order they were said — the ask it is anchored to and
    the message that just arrived.

    A cold ask straight from idle is anchored to nothing, so it is the one turn on its own,
    which is the floor both draws already handle.  Deduped for the same reason
    ``build_framing_content`` dedupes: an anchor that IS this message would otherwise be
    handed over twice, and every value would then be a span of a document that says
    everything twice."""
    turns = [ask] if ask else []
    if message and message not in turns:
        turns.append(message)
    return turns


def _already_settled(
    declared: list[SkillParameter], settled: dict[str, str] | None
) -> dict[str, str]:
    """What a parked round already bound, narrowed to what this routine still DECLARES
    (#1894) — in declared order, so everything downstream reads the routine's own order.

    The narrowing is what makes the carry safe across a re-taught routine: a skill is
    REPLACE-able by name, so a value bound to a parameter that no longer exists is a value
    with nowhere to go, and carrying it would put it in a signature the registry does not
    describe."""
    if not settled:
        return {}
    return {
        parameter.name: settled[parameter.name]
        for parameter in declared
        if parameter.name in settled
    }


def _shortfall(
    routine: Skill,
    declared: list[SkillParameter],
    already: dict[str, str],
    bound: MissingParameters | None,
) -> RoundShortfall | None:
    """The two empty-handed directions, told apart and typed for the caller (#1885).

    A ``MissingParameters`` is an ENUMERATED OUTCOME: the draw read the words correctly and
    they named no value for something the routine needs, which is a fact about the ASK
    rather than about the draw.  So it becomes the state a request turn is instructed from
    — the routine, what it is for, what the words DID settle, and each parameter that got
    nothing, carrying the registry's own line of what to supply.  Keeping the values the
    words did settle is what stops the turn asking for them a second time.

    A draw that produced nothing usable names nothing at all, so there is nothing to ask
    for and nothing to state: it stays the honest ``None`` the apply turn fails on.

    ``already`` is what the round had settled before this draw (#1894), and it joins what
    this one settled: the round is one negotiation, so what it holds is everything the user
    has said across it, not only what the newest turn's draw was asked about."""
    _log_shortfall(routine, bound)
    if bound is None:
        return None
    values = already | bound.values
    return RoundShortfall(
        skill=routine.name,
        description=routine.description,
        # Both lists are built in DECLARED order rather than in the order the draw
        # happened to answer in, so the rendered state reads the way the routine is
        # written wherever else it renders.
        bound={
            parameter.name: values[parameter.name]
            for parameter in declared
            if parameter.name in values
        },
        missing=tuple(
            CandidateParameter(name=parameter.name, description=parameter.description)
            for parameter in declared
            if parameter.name in bound.names
        ),
    )


def _filled_signature(
    routine: Skill, declared: list[SkillParameter], values: dict[str, str]
) -> SkillSignature:
    """``routine``'s own interface with THIS round's values in it.

    The name and the description are the REGISTRY's, never redrawn — the routine already
    has an identity, and a second draw's preferred wording would derive a container name
    for a job filed under a different one.  Only the values are this round's, and each is
    looked up by the DECLARED name because that is the key the binder answers under.

    ``values`` is TOTAL over ``declared`` by construction: the caller reaches this only
    when what a parked round already settled plus what the draw filled covers every
    parameter, which is exactly the two halves the round is made of (#1894)."""
    return SkillSignature(
        name=routine.name,
        description=routine.description,
        parameters=tuple(
            FramedParameter(
                name=parameter.name,
                description=parameter.description,
                value=values[parameter.name],
            )
            for parameter in declared
        ),
    )


def _log_shortfall(routine: Skill, bound: MissingParameters | None) -> None:
    """Say which way the binding fell short, in the round's own terms.

    The two are logged apart because they are different facts, and since #1885 they end
    differently too.  A draw that read the words correctly and found them short of
    something NAMES what is missing — the structural ``request`` signal — so the round is
    not failed: the turn lands in request and asks for the rest, at INFO, because a covered
    ask waiting on one detail is an ordinary turn rather than a degradation.  A draw that
    produced nothing usable names nothing at all, and that one is still the honest failure
    an apply turn ends on."""
    if bound is None:
        logger.warning(
            "The round asking for %r could not be bound — no container was built, "
            "so this round enters its turn unframed",
            routine.name,
        )
        return
    logger.info(
        "The round asking for %r supplied no value for %s — no container was built, "
        "so this round enters request and asks for it",
        routine.name,
        ", ".join(bound.names),
    )


def discard_round_container(
    db: Database, framing: RoundFraming | None, *, run_id: str | None
) -> None:
    """Retire the container a FAILED round built (#1839's honest learn-failure path).

    A learn turn that taught nothing leaves a container describing a routine that does not
    exist — minutes old, empty, and named for a job nobody can run — so the failure takes
    it with it rather than leaving it in the store map for the user to wonder about.

    A module function rather than a method because the caller is the run-end learn-terminal
    check, which holds the round's framing and the database and has no business holding a
    micro-context to make draws with.  Guarded on emptiness — unlike
    :func:`abandon_round_container` below, which is the round the USER called off rather
    than the one that failed: a round that failed after writing something leaves that
    something reachable, because the user never said they were done with it."""
    if framing is None:
        return
    archive_round_container(db, framing.container, run_id=run_id, note=_ARCHIVED_ROUND_FAILED)


def abandon_round_container(
    db: Database, framing: RoundFraming | None, *, run_id: str | None
) -> None:
    """Retire the container of a round that ENDED IN IDLE (#1896) — the bail's durable half.

    An idle landing ends the round, and a bail preserves nothing: any next task opens a
    flow of its own, so the container the round built stops being somewhere anything writes
    and becomes a collection named for a job nobody is doing.  Nothing retired it before
    this, so a round walked away from left one behind for good.

    Unlike the retirements above this one is NOT guarded on emptiness, and that is the
    difference between the two reasons a container is retired.  Those retire LITTER — a
    container a mechanism built and then had no use for — where something the user can
    still read is not litter.  This one retires the round's own intermediate state, which
    is exactly what the code owner's bail ruling discards, and what a demonstration wrote
    into it is part of that state rather than an exception to it.  Archiving keeps it
    readable either way: an archived collection is a tombstone, not a deletion, so a bail
    drawn off a flaky classification stays recoverable and the same job taught again
    revives the very container this retires (:meth:`RoundFramer._settle_container`).

    A module function beside :func:`discard_round_container`, and for the same reason: the
    caller holds the round's framing and the database and has no business holding a
    micro-context to make draws with."""
    if framing is None:
        return
    row = db.memories.get(framing.container)
    if row is None or row.archived:
        return
    db.memories.archive(row.name, actor=MutationActor.SYSTEM, run_id=run_id, note=_ABANDONED_ROUND)
    logger.info("Archived the abandoned round's container %r (%s)", row.name, _ABANDONED_ROUND)


def snapshot_replaced_skill(db: Database, framing: RoundFraming) -> ReplacedSkill | None:
    """What the registry ALREADY holds under ``framing``'s pinned name — the thing the
    round's own write is about to replace (#1902).

    ``None`` says the round is minting over nothing, so the routine it registers will be
    the round's own.  A row means it is RE-TEACHING something the user already had, and
    the WHOLE row is carried, because that is what putting it back requires.

    WHEN this is truthful is the caller's (:meth:`ConversationMachine._next_provenance`);
    all this does is read.  The embedding is base64-encoded rather than deserialized to
    floats: round state has to be JSON and this row rides every later move of the round, so
    the vector — which is the bulk of it — travels in the compact form and goes back into
    the column byte-for-byte."""
    row = db.skills.get(framing.skill)
    if row is None:
        return None
    return ReplacedSkill(
        name=row.name,
        steps=row.steps,
        parameters=row.parameters,
        intent=row.intent,
        description=row.description,
        description_embedding=_encode_embedding(row.description_embedding),
        source_run_id=row.source_run_id,
        author=row.author,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def abandon_round_skill(
    db: Database,
    framing: RoundFraming | None,
    provenance: RoundProvenance | None,
    *,
    run_id: str | None,
) -> None:
    """Take back what a round that ENDED IN IDLE wrote to the registry (#1902) — the
    bail's other durable half, beside :func:`abandon_round_container`.

    A round that TAUGHT something registers its routine at run end under the name its
    framing pinned, and every later turn of the same round REPLACES that row rather than
    adding beside it — so the registry entry standing mid-round is this round's own work.
    A bail preserves nothing, and that entry is the round's intermediate state exactly as
    the container is: the user called the job off, and a routine nobody asked for is worse
    standing than absent, because the registry is AMBIENT — every later turn reads it.

    WHICH way it is taken back is READ from the round's provenance, never re-decided, and
    the three answers are the three shapes that state comes in:

    * **no provenance** — the round minted nothing.  A skill-gated round binds a routine
      the registry already holds and teaches nothing, so it wrote nothing and the registry
      is left alone.  This is the case that must never fall through to a delete: the
      routine such a round names is the USER's, not the round's;
    * **provenance carrying nothing** — the round minted its name over an empty slot, so
      the routine standing there is the round's own and it is DELETED;
    * **provenance carrying a row** — the round was re-teaching, so the pre-round version
      is RESTORED.  Skipping the delete would not do: by now the round's own extraction has
      replaced what that routine DOES, so leaving it standing would leave an abandoned,
      half-corrected program live under a name existing jobs still run.  Preserving nothing
      means the registry ends where the round found it, not merely un-deleted.

    There is no draft flag to read and none to clear.  Promotion is implicit SURVIVAL: a
    round that ends any other way simply leaves its routine standing, and it stops being a
    draft because only the round holding that framing could have replaced it.

    Deletion rather than archival is the one place this parts company with the container:
    the skill table is versionless and carries no archived flag, so a name is either in the
    registry or it is not — which is also why the restore path carries a whole row.  Either
    way the change is recorded on the mutation ledger under the SYSTEM actor, like the
    container archive beside it: the registry renders ambiently, so a routine that vanishes
    or reverts between turns is a configuration change the recent-changes block has to
    show."""
    if framing is None or provenance is None:
        return
    if provenance.replaced is not None:
        _restore_round_skill(db, provenance.replaced, run_id=run_id)
        return
    if db.skills.delete(framing.skill):
        _record_skill_mutation(db, framing.skill, MutationAction.DELETED, run_id, _ABANDONED_DRAFT)
        logger.info("Discarded the abandoned round's routine %r", framing.skill)


def _restore_round_skill(db: Database, replaced: ReplacedSkill, *, run_id: str | None) -> None:
    """Put the pre-round routine back, exactly as the round found it."""
    db.skills.restore(
        Skill(
            name=replaced.name,
            steps=replaced.steps,
            parameters=replaced.parameters,
            intent=replaced.intent,
            description=replaced.description,
            description_embedding=_decode_embedding(replaced.description_embedding),
            source_run_id=replaced.source_run_id,
            author=replaced.author,
            created_at=replaced.created_at,
            updated_at=replaced.updated_at,
        )
    )
    _record_skill_mutation(db, replaced.name, MutationAction.UPDATED, run_id, _RESTORED_PRE_ROUND)
    logger.info("Restored the re-taught routine %r to its pre-round version", replaced.name)


def _encode_embedding(blob: bytes | None) -> str | None:
    """The stored description vector as base64 — the JSON-safe form round state carries."""
    return b64encode(blob).decode("ascii") if blob is not None else None


def _decode_embedding(encoded: str | None) -> bytes | None:
    """The carried base64 back as the blob the column stores, byte-for-byte — so a restored
    routine is still resolvable by meaning."""
    return b64decode(encoded) if encoded is not None else None


def _record_skill_mutation(
    db: Database, name: str, action: MutationAction, run_id: str | None, note: str
) -> None:
    """Record one registry change on the mutation ledger, SYSTEM actor — no model asked for
    this, exactly as with the container archive beside it."""
    db.mutations.record(
        entity_type=MutationEntityType.SKILL,
        entity_name=name,
        action=action,
        actor=MutationActor.SYSTEM,
        run_id=run_id,
        detail=MutationDetail(note=note),
    )


def archive_round_container(db: Database, container: str, *, run_id: str | None, note: str) -> None:
    """Archive ``container`` when it exists and holds no entries, recording ``note`` as the
    cause on the mutation ledger — the SYSTEM actor, since no model asked for this.

    Emptiness is the whole guard, and it is a READ rather than a judgment about which
    round put what there: litter is a container minutes old with nothing in it, while one
    holding entries is something the user can still read, whatever the round did next.

    Archive rather than delete: a retired mechanism stays a visible tombstone in the
    archived-inclusive catalog (the 0086/0089 pattern), so a container that came and went
    is answerable rather than absent."""
    row = db.memories.get(container)
    if row is None or row.archived:
        return
    memory = db.memory(row.name)
    if memory is not None and memory.read_all():
        logger.info("Kept the round's container %r — it holds entries", row.name)
        return
    db.memories.archive(row.name, actor=MutationActor.SYSTEM, run_id=run_id, note=note)
    logger.info("Archived the round's container %r (%s)", row.name, note)
