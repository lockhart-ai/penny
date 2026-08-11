"""The conversation state machine (#1706) — the model fills states; it never
walks them.

The teach loop is a state machine, and asking the chat model to enact it —
self-locate in a multi-state instruction block, pick the transition, sequence
the round — failed on exactly the transitions (the #1687 evidence base) while
every within-state task succeeded once isolated.  So the machine is harness
plumbing: the harness holds the state, the chat context gets only the current
state's instruction, and transitions are decided by a scoped single-shot
classifier micro-context (:meth:`MicroContext.classify_state`, customer #3 of
the poison/reroll machinery) over the CURRENT state's out-edges only — a 2–4
member union per call, never the global set.

The v1 states: **idle** (ordinary conversation) · **elicit** (a routine was
asked for that no skill covers — Penny asks to be taught) · **learn** (the
steps arrived — do them now, once; the run's framework tail auto-extracts the
skill) · **request** (a skill covers the ask but something it needs is missing
— Penny names the skill and asks for it) · **apply** (the request matches a
known skill and everything it needs was supplied — enact its recipe).

Structural invariants, held here as data and pure functions, never as prompt
prose:

- **The edge table is data** (:data:`OUT_EDGES`): every non-idle state that
  classifies carries the break-out edge → idle (topic changed / called off);
  ``learn`` never returns to elicit — elicit exists to GET instructions, so
  once they have been given there is no going back to it (code-owner ruling,
  beat 4) — but it does reach ``apply``: the demonstrated round ENDS by
  offering to set the routine running, so the machine is parked in learn at the
  exact moment the user accepts, and the acceptance has to be answerable from
  there (before the run-end auto-attach was removed in #1768, learning
  instantiated implicitly and the edge had no work to do).  What supplies the
  skill's values is the round that just ran, so accepting never has to restate
  them.  ``learn`` is reachable from ``idle`` directly — teaching can arrive
  UNPROMPTED ("lemme teach you how to X: do A, B, C"), skipping the teach
  question entirely; entering learn means ONE thing from every source, so the
  condition text is identical on every edge that enters it;
  ``apply`` has NO out-edges — its reset to idle is a post-turn structural
  fact, never a classifier call (there is no message to classify at end of
  run, and completion self-report is the one judgment the machine never asks
  the model for).
- **Fail → stay** (:func:`next_state`): a classifier contract failure — an
  untagged draw, a state outside the union, exhausted poison rerolls — is a
  NON-decision: the machine holds its state.  Distinct from a *classified*
  bail, which is the explicit break-out edge.
- **Apply is offered only when skills exist** (:func:`presented_edges`): with
  no ranked skill candidates in the snapshot, the ``apply`` edge is withheld
  structurally — an empty registry never invites a false apply.

Scope: the classifier machinery plus its DURABLE half.  :class:`ConversationMachine`
holds the state across turns (``db.machine`` — the ``conversation_machine`` row)
and records every move in the ``state_transition`` ledger, so a parked state is
READ by the next message's classification instead of evaporating with the turn
that set it.  The classifier call is itself ledger-visible (its own
``agent_name``/``prompt_type`` promptlog rows), so a transition row joins to the
draw that produced it and per-edge accuracy is scorable over production history.
Still unwired to chat: no per-state chat prompt reads this yet, and the pure
pieces (:func:`presented_edges`, :func:`render_classifier_content`,
:func:`next_state`) stay callable without a database for the eval harness.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import TransitionCause
from penny.prompts import Prompt
from penny.tools.micro_context import SKILL_TAG, MicroContext, StateDraw, StateDrawOutcome

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)


class ConversationState(StrEnum):
    """The machine's closed state set — the union the classifier draws from is
    always a per-state SLICE of this (:data:`OUT_EDGES`), never the whole."""

    IDLE = "idle"
    ELICIT = "elicit"
    LEARN = "learn"
    REQUEST = "request"
    APPLY = "apply"


# The edge table — data, not prose.  Keyed by the CURRENT state; the value is
# the full candidate union a classifier call may choose from (before the
# structural apply-narrowing in ``presented_edges``).  Order is render order.
OUT_EDGES: dict[ConversationState, tuple[ConversationState, ...]] = {
    ConversationState.IDLE: (
        ConversationState.APPLY,
        ConversationState.REQUEST,
        ConversationState.LEARN,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    ),
    ConversationState.ELICIT: (
        ConversationState.LEARN,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    ),
    ConversationState.LEARN: (
        ConversationState.APPLY,
        ConversationState.LEARN,
        ConversationState.IDLE,
    ),
    ConversationState.REQUEST: (
        ConversationState.APPLY,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    ),
    ConversationState.APPLY: (),
}

# One CANONICAL definition per STATE — stable everywhere it renders (the
# code-owner correction on beat 4: states have fixed semantics; only which
# TRANSITIONS are available varies by current state, and that part is already
# structural — OUT_EDGES + the union narrowing.  The prior per-edge meanings
# conflated transition conditions with state identity and drifted per beat;
# four stable strings replace nine drifting ones).  Load-bearing qualifiers
# live in the definitions themselves: learn exists ONLY when the message
# carries instructions; idle owns everything deferred.
STATE_DEFINITIONS: dict[ConversationState, str] = {
    ConversationState.IDLE: (
        "ordinary conversation — chat, questions, passing mentions, or "
        "anything put off for later; no task is being given or taught right now"
    ),
    ConversationState.ELICIT: (
        "the user wants a task done that no known skill covers, and the "
        "assistant is asking to be taught the steps"
    ),
    ConversationState.LEARN: (
        "the user's message gives instructions to follow — what to read, look "
        "for, or remember; a plain command counts, and a message without "
        "instructions is never learn"
    ),
    ConversationState.REQUEST: (
        "a known skill looks like it covers what the user wants, and the "
        "assistant has named that skill and asked for the details it needs "
        "before running it"
    ),
    ConversationState.APPLY: (
        "a known skill does what the user is asking for and everything that "
        "skill needs has been supplied, so it runs"
    ),
}

# The TRANSITION FUNCTION — the condition that selects each move, keyed
# (current, target).  This is the half that legitimately varies by where the
# machine stands, and rendering it AS conditions is what keeps the state
# DEFINITIONS above stable: a state means one thing everywhere; what changes is
# what moves you out of the state you are in.  Conditions come from #1706's
# edge table (steps arrived / still clarifying / broke out; correcting /
# working it out / called off) — the design, not the eval pools.
#
# IDLE is the DECLARED DEFAULT: it carries no condition of its own and always
# renders last as "in all other cases", so a message meeting none of the real
# conditions has an unambiguous home instead of being forced into the nearest
# positive clause.
#
# Each condition states its OWN shape and nothing else — a CHOICE MENU, where the
# reader compares the message against every option rather than being talked out of
# one option inside another.  No edge describes a sibling's case or argues against
# it: the sibling is right there in the same list saying what it is, and a condition
# that also has to describe its neighbour is the neighbour's own wording having
# failed.  Naming your OWN near-misses is not that — the parked-learn correction edge
# says which messages carry no correction, which is still a statement about what IT
# is, and it is there because a nearby shape was measured landing here wrongly.
TRANSITIONS: dict[tuple[ConversationState, ConversationState], str] = {
    (ConversationState.IDLE, ConversationState.APPLY): (
        "one of the known skills does what they are asking for, and their message "
        "contains all the information for the skill's parameters. A skill does the "
        "task once. The schedule and notifications are added when it is set up, so a "
        "skill that does the task once covers an ask to do it repeatedly. Add a second "
        f"line naming that skill: {SKILL_TAG} <its name, exactly as quoted in Known skills>"
    ),
    (ConversationState.IDLE, ConversationState.REQUEST): (
        "a known skill looks like it covers what they are asking for, but "
        "something that skill needs is missing from their message — add a "
        f"second line naming that skill: {SKILL_TAG} <its name, exactly as quoted "
        "in Known skills>"
    ),
    (ConversationState.IDLE, ConversationState.ELICIT): (
        "they are asking to set up an ongoing task or routine and no known skill covers it"
    ),
    (ConversationState.IDLE, ConversationState.LEARN): (
        "the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, "
        "including corrections to previous steps"
    ),
    (ConversationState.REQUEST, ConversationState.APPLY): (
        "their message supplies the details the assistant asked for, or "
        "confirms the skill it named — add a second line naming that skill: "
        f"{SKILL_TAG} <its name, exactly as quoted in Known skills>"
    ),
    (ConversationState.REQUEST, ConversationState.ELICIT): (
        "they say that skill is not what they meant, and still want the task done"
    ),
    (ConversationState.ELICIT, ConversationState.LEARN): (
        "the user's message is a set of instructions to follow for the task "
        "being worked on — what to read, what to look for, what to remember, "
        "including corrections to previous steps"
    ),
    (ConversationState.ELICIT, ConversationState.ELICIT): (
        "they are still working the task out with the assistant — a question "
        "back, or a clarification about the task itself"
    ),
    (ConversationState.LEARN, ConversationState.APPLY): (
        "the user signals positively — accepting what was just demonstrated: a yes, a "
        "great, a go-ahead. They often add how the job should run — its timing, how "
        "long it keeps going, or whether to tell them — but a plain acceptance is "
        "enough. Add a second line naming that skill: "
        f"{SKILL_TAG} <its name, exactly as quoted in Known skills>"
    ),
    (ConversationState.LEARN, ConversationState.LEARN): (
        "the user is correcting their previous instructions — this message itself "
        "restates the steps with changes: what to read, what to look for, what to "
        "remember, where to save it — or asks to simply run it again after a hiccup. "
        "A message that only says the round was wrong, or promises new instructions "
        "later, carries no correction."
    ),
}

# The default transition's rendered condition — IDLE's, everywhere.
DEFAULT_TRANSITION = "in all other cases"

# The states whose draw must also bind a skill (their conditions direct the
# SKILL: line): apply enacts one, request negotiates one.
SKILL_GATED_STATES = (
    ConversationState.APPLY.value,
    ConversationState.REQUEST.value,
)

# The conversation-slice section headers — fixed strings, whole-render pinned.
# Markdown headers, not label lines: the parked, populated contexts this slice
# grows into (many candidates, quoted turns that carry their own lists and
# colons) need STRUCTURAL section boundaries the model can navigate, not
# typographic ones a long context swallows.
_LAST_TURN_HEADER = "## The assistant's last message"
_TASK_HEADER = "## The task being worked on"
_SKILLS_HEADER = "## Known skills"
_MESSAGE_HEADER = "## The user's newest message"
_CURRENT_STATE_HEADER = "## Current state"
_TRANSITIONS_HEADER = "## Transitions"
_NONE_PLACEHOLDER = "(none)"


class CandidateParameter(BaseModel):
    """One declared input of a candidate skill — the light, cycle-free
    projection of the store's ``SkillParameter`` (this module stays a leaf:
    importing the database package from here re-enters it partially
    initialized on a direct import, the documented validation/__init__ cycle
    shape).  ``build_snapshot`` maps the store rows in."""

    name: str
    description: str | None = None


class SkillCandidate(BaseModel):
    """One ranked skill from the registry's structural pre-pass — the ``name``
    is the exact token a gated apply draw must copy back (display form ==
    invocation form), the ``description`` the meaning the render shows beside
    it, and ``parameters`` the skill's declared inputs.  The render shows ALL
    of it — coverage is reasoned from the skill's full metadata (what it does
    AND what it needs to do it), so a skill for a different kind of value
    reads as non-coverage without any imperative saying so."""

    name: str
    description: str
    parameters: list[CandidateParameter] = []

    def render(self) -> str:
        """The one-line candidate render: ``"name" — description`` plus a
        ``(needs: …)`` tail naming each declared parameter with its
        what-to-supply — absent (byte-identical) for a parameterless skill.

        The name is QUOTED because a skill-gated draw has to copy it back
        verbatim, and unquoted it had no visible end: the line joins name to
        description with an em-dash and then uses em-dashes again inside the
        ``needs`` tail, so "copy the name exactly" addressed a boundary the
        render never drew.  A draw pasted the entire line — name, description
        and needs — failed membership validation, rerolled, and did it again,
        losing a turn whose state it had reasoned out correctly twice.  Quotes
        make the token the instruction refers to the token the eye can see."""
        line = f'"{self.name}" — {self.description}'
        if not self.parameters:
            return line
        needs = "; ".join(
            f"{parameter.name} — {parameter.description}"
            if parameter.description
            else parameter.name
            for parameter in self.parameters
        )
        return f"{line} (needs: {needs})"


class MachineSnapshot(BaseModel):
    """The classifier's input — the machine's situation at the moment a message
    arrives, constructed by the caller (the eval harness in v1; chat wiring
    later).  Deliberately narrow: the slice is scoped by the machine's own
    facts, never a raw conversation-recency window.

    ``penny_last_turn`` is what the assistant just said — the newest message is
    a REPLY, and replies are only classifiable against what they answer ("just
    the headline" is steps-arrived only against "what should I look for?").
    ``task_anchor`` is the instigating ask, present when the machine is parked
    in a non-idle state.  ``skill_candidates`` are the registry's ranked
    resolution for this message (the structural pre-pass, built by
    :func:`build_snapshot` — the classifier picks among evidence, it does not
    retrieve); empty means the ``apply`` edge is withheld entirely."""

    state: ConversationState
    penny_last_turn: str | None = None
    task_anchor: str | None = None
    skill_candidates: list[SkillCandidate] = []


class StateDecision(BaseModel):
    """One classification, typed for the machine: the draw outcome plus the
    decided state (``None`` on any non-decision — the fail → stay input) and,
    for an apply decision, the covering skill's name (validated a member of the
    offered candidates by the draw contract — never ``None`` on apply)."""

    outcome: StateDrawOutcome
    state: ConversationState | None = None
    skill: str | None = None


class StateClassifier:
    """Decides one transition per incoming message, in a scoped micro-context."""

    def __init__(self, model_client: LlmClient) -> None:
        self._micro_context = MicroContext(model_client)

    async def classify(
        self, snapshot: MachineSnapshot, message: str, *, run_target: str | None = None
    ) -> StateDecision:
        """One tagged draw over the current state's out-edges: narrow the union
        structurally, render the scoped slice, draw once (poison-screened,
        membership-validated, re-rolled while violated), and type the result."""
        edges = presented_edges(snapshot)
        if not edges:
            raise ValueError(
                f"State '{snapshot.state}' has no out-edges — its transitions "
                "are structural, never classified"
            )
        content = render_classifier_content(snapshot, message)
        draw = await self._micro_context.classify_state(
            content,
            [edge.value for edge in edges],
            skill_gated_states=SKILL_GATED_STATES,
            skills=[candidate.name for candidate in snapshot.skill_candidates],
            run_target=run_target,
        )
        return self._decision(draw)

    @staticmethod
    def _decision(draw: StateDraw) -> StateDecision:
        """The machine-typed decision: a DECIDED draw carries a name guaranteed
        to be a union member (and, for a skill-gated state, a skill guaranteed
        to be a candidate),
        so the enum conversion cannot fail; every other outcome carries no state
        (the non-decision the machine holds on)."""
        if draw.outcome is StateDrawOutcome.DECIDED:
            return StateDecision(
                outcome=draw.outcome,
                state=ConversationState(draw.name),
                skill=draw.skill or None,
            )
        return StateDecision(outcome=draw.outcome)


def presented_edges(snapshot: MachineSnapshot) -> tuple[ConversationState, ...]:
    """The union actually offered to the classifier: the current state's
    out-edges, minus every SKILL-GATED state when the snapshot carries no skill
    candidates — a structural narrowing, so an empty registry never renders an
    option whose contract demands naming a skill there is none of (the
    false-apply invitation, and the same for request: you cannot ask
    for a skill's missing details when no skill is on offer)."""
    edges = OUT_EDGES[snapshot.state]
    if not snapshot.skill_candidates:
        edges = tuple(edge for edge in edges if edge.value not in SKILL_GATED_STATES)
    return edges


def render_classifier_content(snapshot: MachineSnapshot, message: str) -> str:
    """The classifier's whole world, rendered as markdown SECTIONS: WHERE the
    machine stands and WHAT MOVES IT (the current state with its canonical
    definition, then one line per available transition carrying the condition
    that selects it — idle last, the declared default), over the scoped
    conversation slice
    (assistant's last turn, the parked task anchor when one exists, the known
    skills, the newest message), then the offered states with their per-edge
    meanings.

    The skills section ALWAYS renders — ``(none)`` for an empty registry —
    because an edge meaning references it ("no known skill covers it"): the
    no-coverage fact must be a READ off the rendered state, never an inference
    from a missing section (the rational-actor doctrine).  The task anchor, by
    contrast, renders only when parked: no meaning references an absent task."""
    sections = [f"{_LAST_TURN_HEADER}\n{snapshot.penny_last_turn or _NONE_PLACEHOLDER}"]
    if snapshot.task_anchor is not None:
        sections.append(f"{_TASK_HEADER}\n{snapshot.task_anchor}")
    if snapshot.skill_candidates:
        listing = "\n".join(f"- {candidate.render()}" for candidate in snapshot.skill_candidates)
        sections.append(f"{_SKILLS_HEADER}\n{listing}")
    else:
        sections.append(f"{_SKILLS_HEADER}\n{_NONE_PLACEHOLDER}")
    sections.append(f"{_MESSAGE_HEADER}\n{message}")
    current = f"{snapshot.state.value} — {STATE_DEFINITIONS[snapshot.state]}"
    sections.append(f"{_CURRENT_STATE_HEADER}\n{current}")
    transitions = "\n".join(
        f"- {target.value} — {_transition_condition(snapshot.state, target)}"
        for target in presented_edges(snapshot)
    )
    sections.append(f"{_TRANSITIONS_HEADER}\n{transitions}")
    return "\n\n".join(sections)


def _transition_condition(current: ConversationState, target: ConversationState) -> str:
    """The condition selecting one move — IDLE is the declared default (it
    carries no condition of its own), every other target names its own."""
    if target is ConversationState.IDLE:
        return DEFAULT_TRANSITION
    return TRANSITIONS[(current, target)]


def next_state(current: ConversationState, decision: StateDecision) -> ConversationState:
    """Fail → stay: only a DECIDED draw moves the machine.  A contract failure
    (untagged, out-of-union, poison-exhausted) is a NON-decision — the machine
    holds its state, so a flaky draw can never eject a parked teach loop.  A
    *classified* bail is different: that is the explicit break-out edge, and it
    arrives here as a DECIDED transition to idle."""
    if decision.outcome is StateDrawOutcome.DECIDED and decision.state is not None:
        return decision.state
    return current


def build_snapshot(
    db: Database,
    *,
    state: ConversationState,
    message: str,
    penny_last_turn: str | None = None,
    task_anchor: str | None = None,
) -> MachineSnapshot:
    """The production snapshot builder — the machine's situation as the
    classifier's input.

    EVERY skill is offered, as name + description + parameters.  No ranking, no
    cap, no embedding (code-owner ruling on #1706): a relevance gate here would
    contradict #1471, which deliberately renders the whole registry to chat
    "wholesale, no relevance gate, no cap" because gates were what HID skills —
    and this list is strictly smaller than the full recipes chat already carries
    every turn, so shortening it buys nothing and can only hide the covering
    skill.  Offering everything also means the classifier and the chat turn
    reason over the SAME set, so a disagreement can never come from them having
    been shown different evidence."""
    # Function-local import: the database package must never be imported at
    # this module's import time (leaf discipline — see CandidateParameter).
    from penny.database.skill_store import parameters_from_json

    return MachineSnapshot(
        state=state,
        penny_last_turn=penny_last_turn,
        task_anchor=task_anchor,
        skill_candidates=[
            SkillCandidate(
                name=skill.name,
                description=skill.description,
                parameters=[
                    CandidateParameter(name=parameter.name, description=parameter.description)
                    for parameter in parameters_from_json(skill.parameters)
                ],
            )
            for skill in db.skills.list_all()
        ],
    )


# The ONE instruction per state the chat prompt carries.  TOTAL over the state
# set by construction (pinned by a test): the machine always has a state, so a
# turn always has exactly one instruction and there is no default to fall back
# on.  A fallback would mean the state failed to determine the prompt, which is
# the whole thing the machine exists to fix.
STATE_INSTRUCTIONS: dict[ConversationState, str] = {
    ConversationState.IDLE: Prompt.IDLE_INSTRUCTION,
    ConversationState.ELICIT: Prompt.ELICIT_INSTRUCTION,
    ConversationState.LEARN: Prompt.LEARN_INSTRUCTION,
    ConversationState.REQUEST: Prompt.REQUEST_INSTRUCTION,
    ConversationState.APPLY: Prompt.APPLY_INSTRUCTION,
}


def conversation_prompt(state: ConversationState) -> str:
    """The chat system prompt for a state: the invariant physics core with THIS
    state's instruction between head and tail.

    The state's name never renders, and neither does any other state's
    instruction — by the time chat reads this the state is already decided, so
    what it needs is what to do, not where it is.  Indexes ``STATE_INSTRUCTIONS``
    directly: a missing state is a programming error and should raise, never
    quietly compose some other state's prompt."""
    return Prompt.CONVERSATION_HEAD + STATE_INSTRUCTIONS[state] + Prompt.CONVERSATION_TAIL


class ConversationMachine:
    """The machine itself: state held across turns, moved by classification or
    structure, every move recorded — as ONE write.

    :class:`StateClassifier` decides one transition and returns it; this is what
    makes those decisions a machine.  It owns the three things a decision alone
    cannot: reading where the machine stands (the newest logged move), the
    ANCHOR lifecycle (set on entry off idle, carried through the parked round,
    cleared on break-out — so a reply arriving three messages later is still
    classified against the ask it answers), and the structural moves no model is
    asked to make (the post-apply reset).

    There is no materialized state to keep in step with the log: appending the
    move IS moving the machine (``db.machine``), so a failed write leaves the
    machine exactly where it was rather than moving it silently off-ledger.

    Not wired to chat yet — the caller supplies the message and its id.
    """

    def __init__(self, db: Database, classifier: StateClassifier) -> None:
        self._db = db
        self._classifier = classifier

    async def advance(
        self,
        message: str,
        *,
        message_id: int | None = None,
        penny_last_turn: str | None = None,
        run_id: str | None = None,
        run_target: str | None = None,
    ) -> StateDecision:
        """One incoming message, start to finish: settle any structural move
        first, classify from where that leaves the machine, then record the
        result (which is what applies it).  Returns the decision the caller
        acts on."""
        state = self._settle_structural(message_id=message_id)
        snapshot = self._snapshot(state, message, penny_last_turn=penny_last_turn)
        decision = await self._classifier.classify(snapshot, message, run_target=run_target)
        self._record_decision(state, decision, message_id=message_id, run_id=run_id)
        return decision

    def state(self) -> ConversationState:
        """Where the machine stands — the newest move's destination.  No history
        at all is the cold start, and idle is what that means."""
        latest = self._db.machine.latest_transition()
        return ConversationState(latest.to_state) if latest else ConversationState.IDLE

    def link_message(self, run_id: str, message_id: int) -> None:
        """Attach the incoming message to the moves it caused, once it has an id.

        The channel cannot supply the id up front (it logs the message after the
        turn so it never doubles into that turn's own recall), so ``advance``
        runs without it and this closes the loop.  A round that OPENED this turn
        — the machine left idle and parked itself — takes that same message as
        its anchor: the ask a parked round is anchored to is the one that opened
        it, which is precisely the move recorded here."""
        latest = self._db.machine.latest_transition()
        opened_round = (
            latest is not None
            and latest.run_id == run_id
            and latest.from_state == ConversationState.IDLE.value
            and latest.to_state != ConversationState.IDLE.value
        )
        self._db.machine.link_message(run_id, message_id, anchor=opened_round)

    def _settle_structural(self, *, message_id: int | None) -> ConversationState:
        """Apply any move the edge table makes WITHOUT a model: a state with no
        out-edges resets to idle (``apply`` — completion is a structural fact,
        never a self-report the machine asks the model for).  Appended like any
        other move, so the log shows the reset that preceded the draw rather
        than an unexplained jump."""
        state = self.state()
        if OUT_EDGES[state]:
            return state
        self._db.machine.record_transition(
            from_state=state.value,
            to_state=ConversationState.IDLE.value,
            cause=TransitionCause.STRUCTURAL,
            anchor_message_id=None,
            message_id=message_id,
        )
        return ConversationState.IDLE

    def _snapshot(
        self, state: ConversationState, message: str, *, penny_last_turn: str | None
    ) -> MachineSnapshot:
        """The classifier's input, with the anchor resolved from the machine's
        own log — the parked ask is READ from the conversation it points at,
        never a copy this layer keeps."""
        return build_snapshot(
            self._db,
            state=state,
            message=message,
            penny_last_turn=penny_last_turn,
            task_anchor=self._anchor_text(),
        )

    def _anchor_text(self) -> str | None:
        """The instigating ask's text, or ``None`` when the machine is unparked
        (or its anchor message has since been deleted — an absent anchor is the
        idle shape, never an error)."""
        anchor_id = self._anchor_message_id()
        if anchor_id is None:
            return None
        message = self._db.messages.get_by_id(anchor_id)
        return message.content if message is not None else None

    def _anchor_message_id(self) -> int | None:
        """The anchor in effect right now — carried on the newest move."""
        latest = self._db.machine.latest_transition()
        return latest.anchor_message_id if latest else None

    def _record_decision(
        self,
        current: ConversationState,
        decision: StateDecision,
        *,
        message_id: int | None,
        run_id: str | None,
    ) -> None:
        """Append the move ``next_state`` allows — the write that applies it.

        EVERY draw is appended, including one that moves nothing: without the
        held draws the log reports a perfect classifier by construction, and
        per-edge accuracy is exactly what it exists to make scorable."""
        target = next_state(current, decision)
        self._db.machine.record_transition(
            from_state=current.value,
            to_state=target.value,
            cause=TransitionCause.CLASSIFIER,
            anchor_message_id=self._next_anchor(current, target, message_id),
            outcome=decision.outcome.value,
            message_id=message_id,
            run_id=run_id,
            skill_name=decision.skill,
        )

    def _next_anchor(
        self, current: ConversationState, target: ConversationState, message_id: int | None
    ) -> int | None:
        """The anchor lifecycle in one place: idle clears it, entering a parked
        state FROM idle sets it to the instigating message, and a machine
        already parked KEEPS the ask it was parked on — the anchor is what
        started the round, not the newest thing said during it."""
        if target is ConversationState.IDLE:
            return None
        if current is ConversationState.IDLE:
            return message_id
        return self._anchor_message_id()
