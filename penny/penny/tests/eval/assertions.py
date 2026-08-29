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
from penny.tests.eval.cohort import (
    AssertionRow,
    Claim,
    ClaimOutcome,
    Feature,
    SampleObservation,
    unsourced_specifics,
)
from penny.tests.eval.worlds import World

# How one sample answers one claim: ``(ok, rationale)``.
Answer = tuple[bool, str | None]
SampleClaim = Callable[[SampleObservation], Answer]


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
        self, case_id: str, model: str, world: World, samples: list[SampleObservation]
    ) -> None:
        self.case_id = case_id
        self.model = model
        self.world = world
        self.samples = samples
        self.claims: list[Claim] = []
        self.features: list[Feature] = []

    # ── the claims ───────────────────────────────────────────────────────────
    def assert_machine_landed(self, state: ConversationState) -> None:
        """The round ended in the state this story is about.

        Scored rather than reported, for the one case whose contract only EXISTS in the state
        it names: a story about the reply that closes a learn round has no subject at all
        outside that state."""
        self._claim(
            f"state: the machine landed in {state.value}",
            lambda s: (s.landed == state.value, f"walked {s.walk}"),
        )

    def assert_a_routine_reached_the_registry(self) -> None:
        self._claim(
            "state: the round taught a routine",
            lambda s: (bool(s.routines), "nothing reached the registry"),
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
        self._claim("state: the routine it recorded names somewhere to act", _names_a_destination)

    def assert_the_store_holds_an_entry(self) -> None:
        self._claim(
            "state: the store holds at least one entry",
            lambda s: (bool(s.entries), "nothing was written"),
        )

    def assert_each_source_was_kept(self) -> None:
        """One claim per SOURCE: an ask that says "from each" is not met by keeping one.

        Reads the WHOLE entry — key and content — because a fact in the key and a blurb in the
        body is a perfectly good way to store it, and a content-only read reported a 25/32
        model failure that was entirely its own bug."""
        self._claim("state: what each page said was kept", _each_source_kept(self.world))

    def assert_nothing_excluded_was_stored(self) -> None:
        """The exclusion the round was told in as many words.  A read rather than a taste: the
        compared tokens appear ONLY on the excluded line."""
        self._claim("state: nothing the ask excluded was stored", _nothing_excluded(self.world))

    def assert_every_stored_entry_traces_to_the_world(self) -> None:
        """An entry naming something nobody's page mentions was invented — and once it is in a
        collection, a collector re-reads it for ever."""
        self._claim(
            "state: every stored entry traces to what the round was given", _store_is_sourced
        )

    def assert_every_value_in_the_reply_is_sourced(self) -> None:
        """Every specific value in the reply traces to something the model was GIVEN — the
        user's turns and the tool results, never Penny's own turns, or a value she invents
        early in a turn rides into the message history and sources itself from her own account
        of it."""
        self._claim("reply: every specific value in it is sourced", _reply_is_sourced)

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
        reads, avoids = _READS_ITS_WORLD, _AVOIDS_THE_OTHER
        self._claim(reads, _reply_reads(self.world), kind="reply")
        self._claim(avoids, _reply_avoids(control.world), kind="reply")
        control._claim(reads, _reply_reads(control.world), kind="reply")
        control._claim(avoids, _reply_avoids(self.world), kind="reply")
        # The control's answers belong to THIS case's score: they are the same claims about the
        # same round, and reporting them separately would hide half of a two-directional test.
        self._absorb(control)

    # ── what is measured ─────────────────────────────────────────────────────
    def measure(self, *features: Feature) -> None:
        """Declare the axes this case MEASURES — never asserted, one-sided ceiling."""
        self.features += [feature for feature in features if feature not in self.features]

    # ── internals ────────────────────────────────────────────────────────────
    def _claim(self, label: str, answer: SampleClaim, kind: str = "state") -> None:
        """Answer one claim for every COMPLETE sample.

        An incomplete sample answers nothing: it never ran its measured turn, so grading it
        would report a failed contract for a contract nobody exercised."""
        outcomes = []
        for sample in self.samples:
            if not sample.complete:
                continue
            ok, rationale = answer(sample)
            outcomes.append(
                ClaimOutcome(sample=sample.name, ok=ok, rationale=None if ok else rationale)
            )
        flavour = "reply" if label.startswith("reply:") else kind
        self.claims.append(Claim(label=label, kind=flavour, outcomes=outcomes))

    def _absorb(self, other: Cohort) -> None:
        """Fold another cohort's answers into this case's claims, matching on label."""
        by_label = {claim.label: claim for claim in self.claims}
        for claim in other.claims:
            target = by_label.get(claim.label)
            if target is not None:
                target.outcomes += claim.outcomes
        other.claims.clear()


# The two directed-change labels, named once because each is written twice — once for the
# cohort and once for its control.  A label is what a report is read by and what a baseline
# diff keys on, so the two sides must be the same string rather than two spellings of one
# intention.
_READS_ITS_WORLD = "reply: it names what this world says"
_AVOIDS_THE_OTHER = "reply: it names nothing from the world it was not given"


# ── The claims themselves, as pure functions over one sample ─────────────────
def _names_a_destination(sample: SampleObservation) -> Answer:
    missing = [r.name for r in sample.routines if not r.names_a_destination]
    return bool(sample.routines) and not missing, f"no destination in {missing}"


def _each_source_kept(world: World) -> SampleClaim:
    def answer(sample: SampleObservation) -> Answer:
        stored = _normalise(sample.stored_text)
        missed = [source[0] for source in world.keeps if not any(t in stored for t in source)]
        return bool(sample.entries) and not missed, f"nothing stored from {missed}"

    return answer


def _nothing_excluded(world: World) -> SampleClaim:
    def answer(sample: SampleObservation) -> Answer:
        stored = _normalise(sample.stored_text)
        landed = [token for token in world.excludes if token in stored]
        return not landed, f"stored the excluded {landed}"

    return answer


def _store_is_sourced(sample: SampleObservation) -> Answer:
    invented = sorted(
        {
            token
            for entry in sample.entries
            for token in unsourced_specifics(entry.text, sample.given)
        }
    )
    return not invented, f"unsourced in the store: {invented}"


def _reply_is_sourced(sample: SampleObservation) -> Answer:
    invented = unsourced_specifics(sample.reply, sample.given)
    return not invented, f"unsourced: {invented}"


def _reply_reads(world: World) -> SampleClaim:
    def answer(sample: SampleObservation) -> Answer:
        named = [token for token in world.names if token in _normalise(sample.reply)]
        return bool(named), "named none of this world's facts"

    return answer


def _reply_avoids(world: World) -> SampleClaim:
    def answer(sample: SampleObservation) -> Answer:
        leaked = [token for token in world.names if token in _normalise(sample.reply)]
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
    """Project claims onto the report's section-A rows."""
    return [
        AssertionRow(
            label=claim.label,
            passed=claim.passed,
            total=claim.total,
            rationales=claim.rationales,
        )
        for claim in claims
    ]
