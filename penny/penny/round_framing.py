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
construction (#1775) — while a correction that SHIFTS the job's identity archives the
near-empty container it is replacing and builds the one it now needs.  Archiving is
guarded on emptiness: litter is a container minutes old with nothing in it, and a
container holding entries is not litter whatever the round did next.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from penny.constants import MutationActor, PennyConstants
from penny.conversation_machine import RoundFraming
from penny.database import Database
from penny.database.models import Skill
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
_ARCHIVED_IDENTITY_SHIFTED = "the corrected round is a different job, with its own container"
_REVIVED_SAME_JOB = "the same job is being taught again, into the container it already had"


class RoundFramer:
    """Frames a round at its entry and settles the container it runs into.

    Two entries, two draws, one outcome: :meth:`frame_entry` MINTS a routine for a round
    being taught, :meth:`bind_entry` FILLS one the registry already holds (#1870), and
    both hand back the same :class:`~penny.conversation_machine.RoundFraming` over a
    container settled by the same find-or-create rule.

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
        self,
        *,
        ask: str | None,
        message: str,
        run_id: str | None,
        previous: RoundFraming | None,
    ) -> RoundFraming | None:
        """Frame the round the machine is entering, and settle its container — the
        summary method.

        Draws the interface from the round's user turns, derives the container's name from
        it, reconciles that against whatever the round had before, and returns the framing
        for the machine to record.

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
        await self._settle_container(framing, previous, run_id=run_id)
        return framing

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
        self, *, skill: str, ask: str | None, message: str, run_id: str | None
    ) -> RoundFraming | None:
        """Frame the round an APPLY entry is entering against a routine that ALREADY
        EXISTS (#1870) — the summary method of the binder half.

        The classifier bound ``skill`` when it decided apply, so nothing here decides what
        the round is about: the routine is settled, and the only open question left is
        which part of the user's words fills each thing that routine already declares.  The
        binder answers exactly that, the container's name is derived from the answer, and
        find-or-create does the rest — which IS tier-1 dedup by construction (#1775): the
        same job asked for a second time derives the name it derived the first time and
        runs into the container that already exists, while a different place mints its own.

        ``None`` when the round cannot be settled — the routine is not in the registry, the
        words fell short of something it needs, or no usable draw came back.  No container
        is built in any of those cases, so nothing is left behind by a round that never
        started, and the apply turn that follows has nothing to configure — which its own
        state fails honestly rather than inventing one (#1875)."""
        routine = self._db.skills.get(skill)
        if routine is None:
            logger.warning(
                "The apply round bound %r, which the registry does not hold — no container "
                "was built, so this round enters apply unframed",
                skill,
            )
            return None
        signature = await self._fill(routine, ask, message)
        if signature is None:
            return None
        framing = RoundFraming(signature=signature, container=container_name(signature))
        await self._settle_container(framing, previous=None, run_id=run_id)
        return framing

    async def _fill(self, routine: Skill, ask: str | None, message: str) -> SkillSignature | None:
        """One BINDING draw over the round's user turns, typed back into the signature the
        rest of this module deals in.

        The draw is handed the turns twice over — once inside the rendered document, which
        also renders the signature, and once on their own — because a value is only
        evidence when the USER said it, and the second argument is the text the span check
        actually tests against (#1867)."""
        declared = parameters_from_json(routine.parameters)
        spoken = render_spoken_turns(_spoken_turns(ask, message))
        content = build_binding_content(spoken, routine.name, routine.description, declared)
        bound = await self._micro_context.bind_skill(
            content,
            [parameter.name for parameter in declared],
            spoken,
            run_target=self._run_target,
        )
        return _filled_signature(routine, declared, bound)

    async def _settle_container(
        self, framing: RoundFraming, previous: RoundFraming | None, *, run_id: str | None
    ) -> None:
        """Find-or-create the round's container, retiring the one it replaces.

        A correction that keeps the job's identity derives the same name and finds the
        same container — nothing is created, nothing is retired, and the round continues
        into what it was already writing.  A correction that SHIFTS it leaves the old
        container behind with nothing in it, so that one is archived (guarded on
        emptiness) before the new one is built.

        Find-or-create is also what makes a container STRANDED by a failure harmless: if
        the turn dies between this step and the machine recording the move, what is left
        behind is an inert, empty collection carrying this job's own name — which the next
        attempt at the same job finds and continues into, rather than an orphan nothing can
        reach."""
        if previous is not None and previous.container != framing.container:
            archive_round_container(
                self._db, previous.container, run_id=run_id, note=_ARCHIVED_IDENTITY_SHIFTED
            )
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


def _filled_signature(
    routine: Skill, declared: list[SkillParameter], bound: SkillBinding | None
) -> SkillSignature | None:
    """``routine``'s own interface with THIS round's values in it, or ``None`` on either
    shortfall.

    The name and the description are the REGISTRY's, never redrawn — the routine already
    has an identity, and a second draw's preferred wording would derive a container name
    for a job filed under a different one.  Only the values are this round's, and each is
    looked up by the DECLARED name because that is the key the binder answers under, total
    by construction for a :class:`BoundValues`."""
    if not isinstance(bound, BoundValues):
        _log_shortfall(routine, bound)
        return None
    return SkillSignature(
        name=routine.name,
        description=routine.description,
        parameters=tuple(
            FramedParameter(
                name=parameter.name,
                description=parameter.description,
                value=bound.values[parameter.name],
            )
            for parameter in declared
        ),
    )


def _log_shortfall(routine: Skill, bound: MissingParameters | None) -> None:
    """Say which way the binding fell short, in the round's own terms.

    The two are logged apart because they are different facts.  A draw that read the words
    correctly and found them short of something NAMES what is missing — the structural
    ``request`` signal, which #1866 declares as the follow-on and this beat does not wire,
    so for now it is a logged outcome and an honest failure rather than a state the machine
    parks in.  A draw that produced nothing usable names nothing at all."""
    unframed = "no container was built, so this round enters apply unframed"
    if bound is None:
        logger.warning(
            "The apply round asking for %r could not be bound — %s", routine.name, unframed
        )
        return
    logger.warning(
        "The apply round asking for %r supplied no value for %s — %s",
        routine.name,
        ", ".join(bound.names),
        unframed,
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
    micro-context to make draws with.  Guarded on emptiness like every other retirement
    here: a round that failed after writing something leaves that something reachable."""
    if framing is None:
        return
    archive_round_container(db, framing.container, run_id=run_id, note=_ARCHIVED_ROUND_FAILED)


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
