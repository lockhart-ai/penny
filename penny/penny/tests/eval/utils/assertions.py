"""The named claims a ported case makes, and the :class:`Cohort` it makes them against
(#1994/#1995).

A case reads as `<priors> / <trigger the action> / <assertions>`.  This module is the third
part: every claim is a sentence, and the sentence is the label the report is read by.

**Two rules govern what lives here.**

*Only END STATE is asserted.*  Where the machine landed, what the store holds, whether the
reply is grounded in the world.  A ROUTE is never asserted — many routes reach one end state
(#1993: three different tools all correctly reached the run record and the check had pinned
one), so a route is measured instead.

*A claim RECORDS; it does not raise.*  ``assert_*`` states what the case claims and answers it
for every sample; whether a rate is a failure is the recorded floor's job.  Until the code
owner accepts a floor there is none, so a ported case reports its numbers rather than going red
on the first miss — which is what makes "run it, read it, then lock it" possible at all.

**A claim only one case makes stays INLINE in that case**, as a small local function.  It
graduates here at the SECOND customer, not the first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from penny.agents.chat import ChatAgent
from penny.conversation_machine import ConversationState
from penny.tests.eval.utils.cohort import (
    Arm,
    AssertionRow,
    Claim,
    ClaimOutcome,
    Feature,
    SampleObservation,
    SpecCategory,
    distinct_worlds,
    fold_typography,
    unsourced_specifics,
)
from penny.tests.eval.utils.worlds import World
from penny.text_validity import (
    half_formed_send_reason,
    has_leaked_harmony_envelope,
    is_degenerate_run,
)
from penny.validation.conditions import ConditionKey

# The STORE label several cases claim under, named once because a label is a DIFF-JOIN KEY: a
# copy per case is a chance for a typo to split one claim's history into two.  Deliberately
# case-NEUTRAL, so one wording reads the same whether the round was abandoned, never started,
# or is waiting on a value.
_NOTHING_CREATED = "state: no mechanism was created"

# The ground a claim is answered against by a cohort that declared no arms at all — the
# unported path, whose cohort is empty and answers nothing.  Matches nothing, so a claim made
# against it is vacuous rather than answered on pages the sample never saw.
_NO_GROUND = World(name="no arms", pages=(), keeps=(), excludes=())

# How one sample answers one claim: ``(ok, rationale)``.
Answer = tuple[bool, str | None]
# How one sample answers one claim, judged against the world ITS OWN ARM ran against.  The
# world is a parameter rather than a closure so a claim stays a pure function of the sample and
# the ground it was given — testable without building a cohort, and correct for a cohort whose
# arms do not share one.
WorldClaim = Callable[[SampleObservation, World], Answer]


class Cohort:
    """One drive of one request: its samples, the claims made about them, and what is measured.

    Returned by the driver rather than fed a scorer callback, because the analysis a cohort
    needs is over the COMPLETE set — the variance statistics are cohort-level by definition,
    and a per-sample callback cannot see the cohort it belongs to.

    A cohort is one behaviour at one model, driven across its ARMS.  An arm carries the input
    it ran AND the world it ran against, so a claim reads the world of the arm that produced
    the sample it is answering about — never one world shared by the whole cohort.

    Chat is the special case where every arm's world is the same object (one page set, five
    wordings of one ask).  A collector is not: its arms vary the job's own inputs, so each
    carries its own pages.  Both are the same seam because the world sits on the arm.
    """

    def __init__(
        self,
        case_id: str,
        model: str,
        samples: list[SampleObservation],
        arms: Sequence[Arm] = (),
    ) -> None:
        self.case_id = case_id
        self.model = model
        self.samples = samples
        # The arms this cohort was driven across.  A sample carries only its arm's LABEL —
        # what the report's rows and its own name are keyed on — so the texts and the worlds
        # live here, listed once, rather than reprinted under every sample.
        self.arms = list(arms)
        self.features: list[Feature] = []
        # A claim is DECLARED here and answered at report time, over every sample the case
        # drove.  Declaring rather than answering immediately is what keeps the case body
        # readable as `<priors> / <trigger> / <assertions>`: the claims are written where
        # they belong even though the numbers only exist once every sample has run.
        self._declared: list[tuple[str, str, SpecCategory, WorldClaim]] = []

    @property
    def claims(self) -> list[Claim]:
        """Every declared claim, answered over every sample the cohort drove."""
        return [
            Claim(label=label, kind=kind, category=category, outcomes=self._answer(answer))
            for label, kind, category, answer in self._declared
        ]

    @property
    def worlds(self) -> list[World]:
        """The DISTINCT worlds this cohort's arms ran against, in arm order."""
        return distinct_worlds(self.arms)

    def _world_for(self, sample: SampleObservation) -> World:
        """The world the arm that produced this sample ran against.

        Read off the sample's own arm INDEX rather than matched on its rendered label, and a
        sample that cannot be resolved RAISES.  Both halves matter and the second is the point:
        the claims that read a world (``_each_source_kept``, ``_nothing_excluded``) are
        satisfied by an empty one, so a sample answered against a fallback would pass
        VACUOUSLY — a green check for a question nobody asked, on the exact claims most likely
        to be wrong.  An unresolvable arm is a harness defect, and a harness defect that
        reports passes is worse than one that stops."""
        if not self.arms:
            return _NO_GROUND
        if not 0 <= sample.arm < len(self.arms):
            raise ValueError(
                f"{self.case_id}: sample {sample.name!r} carries arm {sample.arm}, and this "
                f"cohort drove {len(self.arms)} — the observer must stamp the arm it drove, "
                "or every claim that reads a world answers against ground the sample never saw"
            )
        return self.arms[sample.arm].world

    def _answer(self, answer: WorldClaim) -> list[ClaimOutcome]:
        outcomes = []
        for sample in self.samples:
            # An incomplete sample answers nothing: it never ran its measured turn, so grading
            # it would report a failed contract for a contract nobody exercised.
            if not sample.complete:
                continue
            ok, rationale = answer(sample, self._world_for(sample))
            outcomes.append(
                ClaimOutcome(sample=sample.name, ok=ok, rationale=None if ok else rationale)
            )
        return outcomes

    # ── the claims ───────────────────────────────────────────────────────────
    def assert_machine_landed(self, state: ConversationState) -> None:
        """The round ended in the state this story is about.

        Scored rather than reported, for the one case whose contract only EXISTS in the state
        it names: a story about the reply that closes a learn round has no subject at all
        outside that state."""
        self.claim(
            f"state: the machine landed in {state.value}",
            lambda s, _world: (s.landed == state.value, f"walked {s.walk}"),
            SpecCategory.LANDED,
        )

    def assert_a_routine_reached_the_registry(self) -> None:
        self.claim(
            "state: the round taught a routine",
            lambda s, _world: (bool(s.routines), "nothing reached the registry"),
            SpecCategory.STORE,
        )

    def assert_the_routine_names_a_destination(self) -> None:
        """A PROPERTY of the routine, never its shape.

        Read off the ATTACHMENT MARK, which distillation sets on any leaf whose demonstrated
        value named one of Penny's own collections — registry-derived, so it is true of a
        write, of a log append, and of a plugin verb nobody has heard of, and false of a
        routine that only browses.  Keyed to the mark and never to a tool NAME, because a skill
        is an arbitrary tool sequence and a name-keyed rule would simply not fire for a shape
        nobody enumerated.

        The defect it catches: a routine that browses and never persists runs every cycle for
        ever and keeps nothing, and no check in the suite could see it."""
        self.claim(
            "state: the routine it recorded names somewhere to act",
            _names_a_destination,
            SpecCategory.STORE,
        )

    def assert_the_store_holds_an_entry(self) -> None:
        self.claim(
            "state: the store holds at least one entry",
            lambda s, _world: (bool(s.entries), "nothing was written"),
            SpecCategory.STORE,
        )

    def assert_something_from_each_page_was_written(self) -> None:
        """One claim per SOURCE: an ask that says "from each" is not met by keeping one.

        SOMETHING from the page, not the page's named player.  The sibling two-source case has
        always asked it this way and passes 4/4 where this asked for specific names and passed
        7/18 — because the seals page's only item is an executive appointment, so a round told to
        collect trades and signings reads the page, correctly finds nothing in scope, and was
        failed for it.  The token sets identify which page an entry came from; they are not a
        list of what the ask puts in scope.

        Reads the WHOLE entry — key and content — because a fact in the key and a blurb in the
        body is a perfectly good way to store it, and a content-only read reported a 25/32
        model failure that was entirely its own bug."""
        self.claim(
            "state: something from each page was written down",
            _each_source_kept,
            SpecCategory.STORE,
        )

    def assert_the_write_landed_in_the_round_container(self) -> None:
        """The demonstrated write landed where the turn was TOLD to put it.

        The instruction renders the framed container's name verbatim, so the destination is a
        COPY of a rendered anchor and a write that went elsewhere invented one over what it was
        given. This replaces every judgement about what a collection ought to be called."""
        self.claim(
            "state: the demonstrated write landed in the round's container",
            _wrote_into_container,
            SpecCategory.STORE,
        )

    def assert_nothing_was_scheduled(self) -> None:
        """Learning must not INSTANTIATE: teaching a round does not set it running."""
        self.claim(
            "state: nothing it created was scheduled", _nothing_scheduled, SpecCategory.STORE
        )

    def assert_every_spot_is_a_placeholder(self) -> None:
        """Every spot in the routine was named by the labeller.

        A named spot stops being a leaf parameter, and the labeller names every spot
        unconditionally — so a leftover one means the labelling draw fell back as a whole and the
        routine kept its arg-derived names."""
        self.claim(
            "state: every spot in the routine is a placeholder",
            _placeholders_only,
            SpecCategory.STORE,
        )

    def assert_nothing_excluded_was_stored(self) -> None:
        """The exclusion the round was told in as many words.  A read rather than a taste: the
        compared tokens appear ONLY on the excluded line."""
        self.claim(
            "state: nothing the ask excluded was stored", _nothing_excluded, SpecCategory.STORE
        )

    def assert_nothing_was_written(self) -> None:
        """This round wrote no entry anywhere.

        The end-state form of "she did not go and do it": a turn that asks to be taught, asks
        for a missing value, or stands a job up to run LATER has read nothing worth keeping and
        kept nothing.  It reads the entries the sample WROTE rather than what the store holds,
        so a seeded world's own contents can never answer it."""
        self.claim(
            "state: nothing was written to any collection", _nothing_written, SpecCategory.STORE
        )

    def assert_no_mechanism_was_created(self) -> None:
        """No mechanism was created — not an inert container, not a configured job, none.

        The registry read is a list of rows, so "nothing" is a COUNT and not an inference."""
        self.claim(_NOTHING_CREATED, _nothing_was_born, SpecCategory.STORE)

    def assert_the_move_named_the_routine(self, routine: str) -> None:
        """The move the turn recorded NAMED this routine — the decision half of picking one
        out of a registry of real routines of the same kind.

        A registry key, which is strictly identifiable, where "she recognised the right
        routine" said in a sentence is not.  It reads the landed transition's own
        ``skill_name``, so it is answered whether or not the turn went on to build anything —
        which is the distinction it exists for: picking the wrong routine and picking the right
        one and then doing nothing are different failures."""
        self.claim(
            "state: the move named the routine that covers the ask",
            _named_the_routine(routine),
            SpecCategory.LANDED,
        )

    def assert_no_running_mechanism_was_changed(self) -> None:
        """Nothing that was ALREADY running was touched — the only mechanism a turn may change
        is one it created itself.

        Read off the mutation LEDGER rather than a field-by-field diff, so a rebind, a schedule
        change, a description edit and an archive all answer it the same way and the field
        nobody enumerated is caught too.  The born-this-run exemption is what lets one sentence
        serve a turn that builds nothing and a turn that stands a job up: what it forbids is
        reaching into the jobs the world was already running, which is none of any turn's
        business."""
        self.claim(
            "state: no mechanism that was already running was changed",
            _running_mechanisms_untouched,
            SpecCategory.STORE,
        )

    def assert_no_delivered_message_is_an_unusable_draw(self) -> None:
        """Nothing that reached the user is a draw the loop was supposed to throw away.

        Read through PRODUCTION'S OWN declaration of what an unusable chat draw is — the
        chat agent's ``invalid_draw_conditions`` plus the two transport artifacts the loop
        checks on every draw — never through the one fault a case's injector happens to
        force.  A recovery case keyed to its own injected shape would pass while a
        DIFFERENT unusable draw sailed out in the same turn, and a condition added to
        production would arrive here with nobody remembering to copy it.

        Every DELIVERED message, not the last reply: what the contract forbids is the bad
        draw reaching the user at all, and a turn that delivered two messages would
        otherwise be judged on one of them."""
        self.claim(
            "state: nothing delivered to the user was an unusable draw",
            _nothing_unusable_delivered,
            SpecCategory.STORE,
        )

    def assert_every_delivered_message_is_whole(self) -> None:
        """Every message the user received is a complete message.

        Judged by ``half_formed_send_reason`` — the SHARED rule the send path refuses a
        message by and the run-health classifier flags one by — so what Penny will not
        send and what this calls incomplete are one definition.  A chat reply is delivered
        inline by a text turn and never passes that gate, which is what leaves room for a
        turn to finalise a fragment.

        It replaces a letter-count floor: a threshold somebody picked stands for the
        question rather than answering it, and production already answers it."""
        self.claim(
            "state: every message delivered to the user is a complete message",
            _delivered_messages_are_whole,
            SpecCategory.STORE,
        )

    def assert_the_reply_answers_the_ask(self) -> None:
        """The reply states what the ask asked for, in the world's own terms.

        The only COMPLETENESS claim in the set.  Everything else here is soundness — where
        the machine landed, what the store holds, that nothing was invented — and a reply
        that answers nothing at all satisfies all of it: it lands in the right state, it
        delivers a complete message, and it carries no unsourced value because it carries
        no value.  Measured: a sample whose extractor had returned the answer replied
        "Sounds like you're surprised! Bath! What else do you want to hear about Lake
        Baikal?" and passed 5 of 5 claims as the cohort's REPRESENTATIVE sample.

        Read off the world, never off a guessed phrasing: ``World.answers`` holds tokens
        taken from the page the ask is answered against, so this is the same containment
        read ``keeps`` makes about the store, pointed at the reply.  It is a STORE claim
        because the delivered message is a record the sample's database holds, and it is
        the reply KIND because it reads prose — which carries a wider noise floor than a
        structural claim and must never be offered a floor as if it did not."""
        self.claim(
            "reply: it states the answer the world carries",
            _reply_answers_the_ask,
            SpecCategory.STORE,
            kind="reply",
        )

    def assert_every_stored_entry_traces_to_the_world(self) -> None:
        """An entry naming something nobody's page mentions was invented — and once it is in a
        collection, a collector re-reads it for ever."""
        self.claim(
            "state: every stored entry traces to what the round was given",
            _store_is_sourced,
            SpecCategory.PROVENANCE,
        )

    def assert_every_value_in_the_reply_is_sourced(self) -> None:
        """Every specific value in the reply traces to something the model was GIVEN — the
        user's turns and the tool results, never Penny's own turns, or a value she invents
        early in a turn rides into the message history and sources itself from her own account
        of it."""
        self.claim(
            "reply: every specific value in it is sourced",
            _reply_is_sourced,
            SpecCategory.PROVENANCE,
            kind="reply",
        )

    # ── what is measured ─────────────────────────────────────────────────────
    def measure(self, *features: Feature) -> None:
        """Declare the axes this case MEASURES — never asserted, one-sided ceiling."""
        self.features += [feature for feature in features if feature not in self.features]

    # ── a claim only this case makes ─────────────────────────────────────────
    def claim(
        self,
        label: str,
        answer: WorldClaim,
        category: SpecCategory,
        kind: str = "state",
    ) -> None:
        """Declare a claim this case makes and no other, as a local function in the case.

        The design doc's rule needs a door: a claim graduates into this module at the SECOND
        customer, which means the first customer has to be able to state it where it lives.
        Without this every one-off claim would either be pushed in here early — growing a
        shared vocabulary out of single cases — or written as a bare ``assert``, which raises
        instead of recording and takes the run down on one sample.

        Answered exactly like a named claim: over every complete sample, counted, never
        gated."""
        self._claim(label, answer, category, kind=kind)

    # ── internals ────────────────────────────────────────────────────────────
    def _claim(
        self,
        label: str,
        answer: WorldClaim,
        category: SpecCategory,
        kind: str = "state",
    ) -> None:
        """Declare one claim.  It is answered at report time over every sample."""
        flavour = "reply" if label.startswith("reply:") else kind
        self._declared.append((label, flavour, category, answer))


# ── The claims themselves, as pure functions over one sample ─────────────────
def _names_a_destination(sample: SampleObservation, _world: World) -> Answer:
    missing = [r.name for r in sample.routines if not r.names_a_destination]
    return bool(sample.routines) and not missing, f"no destination in {missing}"


def _each_source_kept(sample: SampleObservation, world: World) -> Answer:
    stored = _normalise(sample.stored_text)
    missed = [source[0] for source in world.keeps if not any(t in stored for t in source)]
    return bool(sample.entries) and not missed, f"nothing stored from {missed}"


def _wrote_into_container(sample: SampleObservation, _world: World) -> Answer:
    """Answers its own sentence — *the demonstrated write landed in the round's container* —
    on every sample, including the ones where nothing happened.

    With nothing written, or no round framed, no write landed there and the sentence is FALSE.
    Not unasked: false.  A claim is a statement about end state and nothing else, so a sample
    that did nothing genuinely fails every claim about what it should have done — five unmet
    contracts, not one failure counted five times.  Answering TRUE to keep the failure report
    tidy traded the truth of the check for the shape of its output, and printed 15/15 for a
    cohort in which one sample wrote nothing at all.

    The framing arm is also the suite's only reader of the round framing: no other claim
    mentions the container, so an unframed round is invisible unless this one says so."""
    if sample.container is None:
        return False, "no round was framed, so no write could land in its container"
    if not sample.entries:
        return False, "nothing was written"
    elsewhere = sorted({e.collection for e in sample.entries if e.collection != sample.container})
    landed = any(e.collection == sample.container for e in sample.entries)
    return landed, f"wrote into {elsewhere} instead of {sample.container!r}"


def _nothing_scheduled(sample: SampleObservation, _world: World) -> Answer:
    return not sample.scheduled, f"scheduled {sample.scheduled}"


def _placeholders_only(sample: SampleObservation, _world: World) -> Answer:
    asking = sorted({p for routine in sample.routines for p in routine.open_parameters})
    # Vacuously true over an empty registry, which would render a round that produced nothing as
    # a pass — so the claim only speaks where a routine exists.
    return bool(sample.routines) and not asking, f"still a leaf parameter: {asking}"


def _nothing_written(sample: SampleObservation, _world: World) -> Answer:
    return not sample.entries, f"wrote {sorted({e.collection for e in sample.entries})}"


def _nothing_was_born(sample: SampleObservation, _world: World) -> Answer:
    born = sorted(one.name for one in sample.mechanisms if one.born_this_run)
    return not born, f"created {born}"


def _named_the_routine(routine: str) -> WorldClaim:
    """The claim that the landed move named ``routine``, bound to one case's covering one.

    The rationale names what it bound INSTEAD, because every wrong pick here is a real routine
    that would go on watching the wrong kind of thing."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        return (
            sample.decision_skill == routine,
            f"the move bound {sample.decision_skill!r}, the ask needs {routine!r}",
        )

    return answer


def _running_mechanisms_untouched(sample: SampleObservation, _world: World) -> Answer:
    touched = sorted(
        one.name for one in sample.mechanisms if one.changed_this_run and not one.born_this_run
    )
    return not touched, f"changed {touched}"


def _nothing_excluded(sample: SampleObservation, world: World) -> Answer:
    stored = _normalise(sample.stored_text)
    landed = [token for token in world.excludes if token in stored]
    return not landed, f"stored the excluded {landed}"


# What makes a chat draw unusable, COMPOSED from what production declares rather than
# re-listed: the two transport artifacts ``Agent._unusable_output_condition`` checks on
# every draw whatever its shape, then the chat agent's OWN ``invalid_draw_conditions`` —
# the shapes that are not a reply.  Composed, because a recovery case keyed to the one
# fault its injector forces would pass while a different unusable draw sailed out in the
# same turn, and a condition added to production would arrive here with nobody remembering
# to copy it.
_UNUSABLE_DRAW: tuple[tuple[str, Callable[[str], bool]], ...] = (
    (ConditionKey.TOOL_CALL_LEAK.value, has_leaked_harmony_envelope),
    (ConditionKey.DEGENERATE_OUTPUT.value, is_degenerate_run),
    *((key.value, predicate) for key, predicate in ChatAgent.invalid_draw_conditions),
)


def _unusable_draw_condition(message: str) -> str | None:
    """The name of the condition that makes ``message`` an unusable draw, or ``None``."""
    return next((name for name, is_invalid in _UNUSABLE_DRAW if is_invalid(message)), None)


def _nothing_unusable_delivered(sample: SampleObservation, _world: World) -> Answer:
    """The rationale names the CONDITIONS that fired and how many of the turn's messages
    they landed on, never the messages themselves — the sample's own fold carries those
    verbatim, and a claim that quotes a slice of one would be inventing where to cut."""
    unusable = [
        condition
        for message in sample.delivered
        if (condition := _unusable_draw_condition(message)) is not None
    ]
    return not unusable, f"{len(unusable)} of {len(sample.delivered)} delivered — {unusable}"


def _delivered_messages_are_whole(sample: SampleObservation, _world: World) -> Answer:
    """Production's own refusal reason is the rationale — it already names the defect and
    the next move, so there is nothing for this to add and no message to quote."""
    half_formed = [
        reason
        for message in sample.delivered
        if (reason := half_formed_send_reason(message)) is not None
    ]
    return not half_formed, f"{len(half_formed)} of {len(sample.delivered)} — {half_formed}"


def _reply_answers_the_ask(sample: SampleObservation, world: World) -> Answer:
    """Every token the world says an answer carries is in the reply, folded through the ONE
    typography definition so a no-break space or a curly dash cannot fail a correct answer.

    A world naming no answer tokens makes no claim and is true — that is an ask with nothing
    to state, not a claim that went unasked."""
    said = fold_typography(sample.reply)
    missing = [token for token in world.answers if fold_typography(token) not in said]
    return not missing, f"the reply never states {missing}"


def _store_is_sourced(sample: SampleObservation, _world: World) -> Answer:
    invented = sorted(
        {
            token
            for entry in sample.entries
            for token in unsourced_specifics(entry.text, sample.given)
        }
    )
    return not invented, f"unsourced in the store: {invented}"


def _reply_is_sourced(sample: SampleObservation, _world: World) -> Answer:
    invented = unsourced_specifics(sample.reply, sample.given)
    return not invented, f"unsourced: {invented}"


# The state claims fold through the SAME definition the provenance ones do.  Two spellings of
# one intention drift, and this pair already had: the dash folding here was missing there, so a
# URL written with a non-breaking hyphen was tolerated by one claim and called an invention by
# the other on the same reply.
_normalise = fold_typography


def assertion_rows(claims: Sequence[Claim]) -> list[AssertionRow]:
    """Project claims onto the report's section-A rows.

    ``kind`` travels with the row because it decides whether the rate is LOCKABLE at all — a
    claim read out of model prose has a noise floor several times wider than a structural one —
    and dropping it here is what would silently offer a floor nothing could hold to."""
    return [
        AssertionRow(
            label=claim.label,
            passed=claim.passed,
            total=claim.total,
            category=claim.category,
            kind=claim.kind,
            rationales=claim.rationales,
        )
        for claim in claims
    ]
