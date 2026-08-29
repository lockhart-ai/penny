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

from penny.conversation_machine import ConversationState
from penny.tests.eval.utils.cohort import (
    AssertionRow,
    Claim,
    ClaimOutcome,
    Feature,
    SampleObservation,
    SpecCategory,
    fold_typography,
    unsourced_specifics,
)
from penny.tests.eval.utils.worlds import World

# How one sample answers one claim: ``(ok, rationale)``.
Answer = tuple[bool, str | None]
# How one sample answers one claim, judged against the world the cohort was driven against.
# The world is a parameter rather than a closure so a claim stays a pure function of the sample
# and the ground it was given — testable without building a cohort.
WorldClaim = Callable[[SampleObservation, World], Answer]


class Cohort:
    """One drive of one request: its samples, the claims made about them, and what is measured.

    Returned by the driver rather than fed a scorer callback, because the analysis a cohort
    needs is over the COMPLETE set — the variance statistics are cohort-level by definition,
    and a per-sample callback cannot see the cohort it belongs to.

    A cohort is one request at one model, against ONE world.  Its samples are hermetic — own
    database, own conversation, own pages — and a claim reads that one world.
    """

    def __init__(
        self,
        case_id: str,
        model: str,
        world: World,
        samples: list[SampleObservation],
        phrasings: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.case_id = case_id
        self.model = model
        self.world = world
        self.samples = samples
        # The wordings this cohort was driven with, as ``(label, text)``.  A sample carries only
        # its phrasing's LABEL, which is what the report's rows and its own name are keyed on —
        # so the texts live here, listed once, rather than reprinted under every sample.
        self.phrasings = list(phrasings)
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

    def _answer(self, answer: WorldClaim) -> list[ClaimOutcome]:
        outcomes = []
        for sample in self.samples:
            # An incomplete sample answers nothing: it never ran its measured turn, so grading
            # it would report a failed contract for a contract nobody exercised.
            if not sample.complete:
                continue
            ok, rationale = answer(sample, self.world)
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
        self._claim(
            f"state: the machine landed in {state.value}",
            lambda s, _world: (s.landed == state.value, f"walked {s.walk}"),
            SpecCategory.LANDED,
        )

    def assert_a_routine_reached_the_registry(self) -> None:
        self._claim(
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
        self._claim(
            "state: the routine it recorded names somewhere to act",
            _names_a_destination,
            SpecCategory.STORE,
        )

    def assert_the_store_holds_an_entry(self) -> None:
        self._claim(
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
        self._claim(
            "state: something from each page was written down",
            _each_source_kept,
            SpecCategory.STORE,
        )

    def assert_the_write_landed_in_the_round_container(self) -> None:
        """The demonstrated write landed where the turn was TOLD to put it.

        The instruction renders the framed container's name verbatim, so the destination is a
        COPY of a rendered anchor and a write that went elsewhere invented one over what it was
        given. This replaces every judgement about what a collection ought to be called."""
        self._claim(
            "state: the demonstrated write landed in the round's container",
            _wrote_into_container,
            SpecCategory.STORE,
        )

    def assert_nothing_was_scheduled(self) -> None:
        """Learning must not INSTANTIATE: teaching a round does not set it running."""
        self._claim(
            "state: nothing it created was scheduled", _nothing_scheduled, SpecCategory.STORE
        )

    def assert_every_spot_is_a_placeholder(self) -> None:
        """Every spot in the routine was named by the labeller.

        A named spot stops being a leaf parameter, and the labeller names every spot
        unconditionally — so a leftover one means the labelling draw fell back as a whole and the
        routine kept its arg-derived names."""
        self._claim(
            "state: every spot in the routine is a placeholder",
            _placeholders_only,
            SpecCategory.STORE,
        )

    def assert_nothing_excluded_was_stored(self) -> None:
        """The exclusion the round was told in as many words.  A read rather than a taste: the
        compared tokens appear ONLY on the excluded line."""
        self._claim(
            "state: nothing the ask excluded was stored", _nothing_excluded, SpecCategory.STORE
        )

    def assert_every_stored_entry_traces_to_the_world(self) -> None:
        """An entry naming something nobody's page mentions was invented — and once it is in a
        collection, a collector re-reads it for ever."""
        self._claim(
            "state: every stored entry traces to what the round was given",
            _store_is_sourced,
            SpecCategory.PROVENANCE,
        )

    def assert_every_value_in_the_reply_is_sourced(self) -> None:
        """Every specific value in the reply traces to something the model was GIVEN — the
        user's turns and the tool results, never Penny's own turns, or a value she invents
        early in a turn rides into the message history and sources itself from her own account
        of it."""
        self._claim(
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


def _nothing_excluded(sample: SampleObservation, world: World) -> Answer:
    stored = _normalise(sample.stored_text)
    landed = [token for token in world.excludes if token in stored]
    return not landed, f"stored the excluded {landed}"


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
