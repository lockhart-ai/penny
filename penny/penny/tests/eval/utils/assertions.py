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
    unsourced_specifics,
)
from penny.tests.eval.utils.worlds import World

# How one sample answers one claim: ``(ok, rationale)``.
Answer = tuple[bool, str | None]
# How one sample answers one claim, judged against the world THAT SAMPLE was given — never a
# world closed over at declaration time, or a control's samples would be graded against the
# cohort's facts.
WorldClaim = Callable[[SampleObservation, World], Answer]


class Cohort:
    """One drive of one request: its samples, the claims made about them, and what is measured.

    Returned by the driver rather than fed a scorer callback, because the analysis a cohort
    needs is over the COMPLETE set — the variance statistics are cohort-level by definition,
    and a per-sample callback cannot see the cohort it belongs to.

    A cohort is one request at one model.  A CONTROL is a separate cohort, never an arm of this
    one: phrasings are *same world, different words* and pool into this cohort's score, while a
    control is *same words, different world* and serves an assertion.  Keeping them separate
    objects is what stops the control quietly entering the cohort sizing.
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
        # drove.  Declaring rather than answering immediately is what lets a control adopted
        # AFTER the claims were made still answer them: the case body reads claims-then-control,
        # and a claim evaluated on the spot would have covered only the samples driven so far.
        self._declared: list[tuple[str, str, SpecCategory, WorldClaim]] = []
        self._adopted: list[SampleObservation] = []
        self._worlds: dict[str, World] = {world.name: world}

    @property
    def covered(self) -> list[SampleObservation]:
        """Every sample this case's claims are answered over — its own and any control's.

        A control is a real drive of the same ask: it lands a machine state, mints a routine
        and writes entries exactly as the cohort does, so its end state is as assertable.  What
        differs is the WORLD each sample is judged against, which is why a claim resolves the
        sample's own world rather than closing over one."""
        return [*self.samples, *self._adopted]

    @property
    def claims(self) -> list[Claim]:
        """Every declared claim, answered over every covered sample against its own world."""
        return [
            Claim(label=label, kind=kind, category=category, outcomes=self._answer(answer))
            for label, kind, category, answer in self._declared
        ]

    def _answer(self, answer: WorldClaim) -> list[ClaimOutcome]:
        outcomes = []
        for sample in self.covered:
            # An incomplete sample answers nothing: it never ran its measured turn, so grading
            # it would report a failed contract for a contract nobody exercised.
            if not sample.complete:
                continue
            ok, rationale = answer(sample, self._worlds[sample.world])
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

    def assert_facts_moved_with_the_world(self, control: Cohort) -> None:
        """DIRECTED CHANGE, both directions, against a control drive of the same ask.

        Perturb the world and the reply's facts must move with it: a reply that says the same
        thing over a different world was never reading the world.  This is the claim wording
        variation cannot make — if Penny were pattern-completing from the shape of the request,
        every phrasing would name the same player and every phrasing would be right.

        The control is a cohort the CASE drove and passed in, not a hidden extra drive: an
        assertion that quietly makes three more model calls is a nasty surprise."""
        assert control.world.name != self.world.name, (
            "a control must be a DIFFERENT world from the one it controls"
        )
        # The control's samples come under this case's claims — every one of them, not just
        # these two: a control is a real drive of the same ask, and its end state is as
        # assertable as the cohort's.
        self._adopt(control)
        # The STORE-side half, and the only one of the two that can carry a floor.  It is
        # decidable on every sample — measured, the base world's samples stored $499 and the
        # control's stored $549 — and it is squarely end state, so directed change gets a gated
        # form without asserting anything about prose.  The reply-side half below stays REPORTED:
        # it reads model text, whose rate moves +/-3 of 18 between two runs of identical code.
        self._claim(_STORE_MOVED, _store_reads, SpecCategory.DIRECTED_CHANGE)
        self._claim(_READS_ITS_WORLD, _reply_reads, SpecCategory.DIRECTED_CHANGE, kind="reply")
        self._claim(
            _AVOIDS_THE_OTHER,
            _reply_avoids(self._other_worlds),
            SpecCategory.DIRECTED_CHANGE,
            kind="reply",
        )

    def _other_worlds(self, sample: SampleObservation) -> list[World]:
        """Every world this case drove EXCEPT the one this sample was given."""
        return [world for name, world in self._worlds.items() if name != sample.world]

    # ── what is measured ─────────────────────────────────────────────────────
    def measure(self, *features: Feature) -> None:
        """Declare the axes this case MEASURES — never asserted, one-sided ceiling."""
        self.features += [feature for feature in features if feature not in self.features]

    # ── internals ────────────────────────────────────────────────────────────
    def _claim(
        self,
        label: str,
        answer: WorldClaim,
        category: SpecCategory,
        kind: str = "state",
    ) -> None:
        """Declare one claim.  It is answered at report time over every covered sample."""
        flavour = "reply" if label.startswith("reply:") else kind
        self._declared.append((label, flavour, category, answer))

    def _adopt(self, control: Cohort) -> None:
        """Take a control's samples under this case's claims.

        Every claim the case makes — declared before this call or after it — is then answered
        over the control's samples too, each against the control's own world.  That is what the
        per-sample scorer it replaces did, and dropping it silently shrank every denominator
        from 18 to 15."""
        self._worlds[control.world.name] = control.world
        self._adopted += control.samples
        control._declared.clear()


# The two directed-change labels, named once because each is written twice — once for the
# cohort and once for its control.  A label is what a report is read by and what a baseline
# diff keys on, so the two sides must be the same string rather than two spellings of one
# intention.
_READS_ITS_WORLD = "reply: it names what this world says"
_AVOIDS_THE_OTHER = "reply: it names nothing from the world it was not given"
_STORE_MOVED = "state: what it stored moved with the world"


# ── The claims themselves, as pure functions over one sample ─────────────────
def _names_a_destination(sample: SampleObservation, _world: World) -> Answer:
    missing = [r.name for r in sample.routines if not r.names_a_destination]
    return bool(sample.routines) and not missing, f"no destination in {missing}"


def _each_source_kept(sample: SampleObservation, world: World) -> Answer:
    stored = _normalise(sample.stored_text)
    missed = [source[0] for source in world.keeps if not any(t in stored for t in source)]
    return bool(sample.entries) and not missed, f"nothing stored from {missed}"


def _wrote_into_container(sample: SampleObservation, _world: World) -> Answer:
    if sample.container is None or not sample.entries:
        # Nothing framed the round, or it wrote nothing at all — the second is already the
        # durable-write claim's own miss, and grading it twice reports one failure as two.
        return True, None
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


def _store_reads(sample: SampleObservation, world: World) -> Answer:
    """Directed change read off the STORE: what this sample wrote carries THIS world's fact.

    The half of directed change that is always decidable — a value is in the store or it is
    not — which is what makes it the half that can carry a floor."""
    stored = _normalise(sample.stored_text)
    missing = [name for name in world.names if name.lower() not in stored]
    if not sample.entries:
        return False, "nothing was stored"
    return not missing, f"stored nothing this world says ({missing})"


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


def _reply_reads(sample: SampleObservation, world: World) -> Answer:
    """Half of DIRECTED CHANGE: the reply carries the facts of the world THIS sample was given."""
    named = [token for token in world.names if token in _normalise(sample.reply)]
    return bool(named), "named none of this world's facts"


def _reply_avoids(others: Callable[[SampleObservation], list[World]]) -> WorldClaim:
    """The other half: nothing from the world this sample was NOT given.

    The other worlds are resolved per sample rather than closed over, so the cohort and its
    control are judged by one claim in both directions instead of two spellings of it."""

    def answer(sample: SampleObservation, _world: World) -> Answer:
        reply = _normalise(sample.reply)
        leaked = [token for other in others(sample) for token in other.names if token in reply]
        return not leaked, f"leaked {leaked}"

    return answer


def _normalise(text: str) -> str:
    """Fold the typography the model sprinkles into its output so a SEMANTIC probe is not
    defeated by cosmetics.  A 0/N from an un-normalised probe is a scorer bug."""
    folded = text.lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        folded = folded.replace(dash, "-")
    for space in ("\xa0", "​", " ", " "):
        folded = folded.replace(space, " ")
    for source, target in (("’", "'"), ("“", '"'), ("”", '"'), ("*", "")):
        folded = folded.replace(source, target)
    return folded


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
