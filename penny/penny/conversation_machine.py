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
skill) · **apply** (the request matches a known skill — enact its recipe).

Structural invariants, held here as data and pure functions, never as prompt
prose:

- **The edge table is data** (:data:`OUT_EDGES`): every non-idle state that
  classifies carries the break-out edge → idle (topic changed / called off);
  ``learn`` is unreachable from ``idle`` (steps can only arrive after an ask);
  ``apply`` has NO out-edges — its reset to idle is a post-turn structural
  fact, never a classifier call (there is no message to classify at end of
  run, and completion self-report is the one judgment the machine never asks
  the model for).  A completed learn round resets structurally the same way;
  only a FAILED round leaves the machine parked in ``learn``.
- **Fail → stay** (:func:`next_state`): a classifier contract failure — an
  untagged draw, a state outside the union, exhausted poison rerolls — is a
  NON-decision: the machine holds its state.  Distinct from a *classified*
  bail, which is the explicit break-out edge.
- **Apply is offered only when skills exist** (:func:`presented_edges`): with
  no ranked skill candidates in the snapshot, the ``apply`` edge is withheld
  structurally — an empty registry never invites a false apply.

v1 scope (the classifier machinery alone): the snapshot is constructed by the
caller — the eval harness today, chat wiring later — and nothing here persists
state or touches the DB.  The classifier call itself is ledger-visible (its
own ``agent_name``/``prompt_type`` promptlog rows), so every decision is
attributable and replayable from production history.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import PennyConstants
from penny.database.skill_store import parameters_from_json
from penny.database.skills import SkillParameter
from penny.llm.similarity import embed_text
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
    APPLY = "apply"


# The edge table — data, not prose.  Keyed by the CURRENT state; the value is
# the full candidate union a classifier call may choose from (before the
# structural apply-narrowing in ``presented_edges``).  Order is render order.
OUT_EDGES: dict[ConversationState, tuple[ConversationState, ...]] = {
    ConversationState.IDLE: (
        ConversationState.IDLE,
        ConversationState.APPLY,
        ConversationState.ELICIT,
    ),
    ConversationState.ELICIT: (
        ConversationState.LEARN,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    ),
    ConversationState.LEARN: (
        ConversationState.LEARN,
        ConversationState.ELICIT,
        ConversationState.IDLE,
    ),
    ConversationState.APPLY: (),
}

# One model-facing meaning per EDGE — keyed (current, target), because the same
# target state means something different depending on where the machine stands
# (idle → idle is ordinary chat; learn → learn is a correction retry).  These
# lines are the classifier's whole doctrine: tuned per-edge through the eval
# beats, whole-render pinned by tests.
EDGE_MEANINGS: dict[tuple[ConversationState, ConversationState], str] = {
    (ConversationState.IDLE, ConversationState.IDLE): (
        "ordinary conversation — chat, a passing mention, or a question or "
        "one-off ask the assistant can answer right away; nothing ongoing is "
        "being set up"
    ),
    (ConversationState.IDLE, ConversationState.APPLY): (
        "one of the known skills does what they are asking for — mere "
        "resemblance to a skill is not coverage, and a needed input missing "
        "from their message (like a url) is gathered later, never a reason to "
        f"refuse — add a second line naming that skill: {SKILL_TAG} <its "
        "name, copied exactly from Known skills>"
    ),
    (ConversationState.IDLE, ConversationState.ELICIT): (
        "they are asking to set up an ongoing task or routine and no known "
        "skill covers it — the assistant would need to be taught how"
    ),
    (ConversationState.ELICIT, ConversationState.LEARN): (
        "their message tells the assistant what to do — what to read, what to "
        "look for, or what to remember; a plain instruction or command IS the "
        "teaching, however brief"
    ),
    (ConversationState.ELICIT, ConversationState.ELICIT): (
        "still working out the task — the assistant's question is not answered yet"
    ),
    (ConversationState.ELICIT, ConversationState.IDLE): (
        "they called the task off, changed the topic, or put it off for later"
    ),
    (ConversationState.LEARN, ConversationState.LEARN): (
        "they are correcting or retrying the task just attempted"
    ),
    (ConversationState.LEARN, ConversationState.ELICIT): (
        "back to working out the task — a question or doubt about how, with no "
        "new instructions to act on yet"
    ),
    (ConversationState.LEARN, ConversationState.IDLE): (
        "they called the task off, changed the topic, or put it off for later"
    ),
}

# The conversation-slice section headers — fixed strings, whole-render pinned.
# Markdown headers, not label lines: the parked, populated contexts this slice
# grows into (many candidates, quoted turns that carry their own lists and
# colons) need STRUCTURAL section boundaries the model can navigate, not
# typographic ones a long context swallows.
_LAST_TURN_HEADER = "## The assistant's last message"
_TASK_HEADER = "## The task being worked on"
_SKILLS_HEADER = "## Known skills"
_MESSAGE_HEADER = "## The user's newest message"
_STATES_HEADER = "## States"
_NONE_PLACEHOLDER = "(none)"


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
    parameters: list[SkillParameter] = []

    def render(self) -> str:
        """The one-line candidate render: ``name — description`` plus a
        ``(needs: …)`` tail naming each declared parameter with its
        what-to-supply — absent (byte-identical) for a parameterless skill."""
        line = f"{self.name} — {self.description}"
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
        membership-validated, one reroll), and type the result for the machine."""
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
            skill_gated_state=ConversationState.APPLY.value,
            skills=[candidate.name for candidate in snapshot.skill_candidates],
            run_target=run_target,
        )
        return self._decision(draw)

    @staticmethod
    def _decision(draw: StateDraw) -> StateDecision:
        """The machine-typed decision: a DECIDED draw carries a name guaranteed
        to be a union member (and, when apply, a skill guaranteed a candidate),
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
    out-edges, minus ``apply`` when the snapshot carries no skill candidates —
    a structural narrowing, so an empty registry never renders an apply option
    with nothing under it (the false-apply invitation)."""
    edges = OUT_EDGES[snapshot.state]
    if not snapshot.skill_candidates:
        edges = tuple(edge for edge in edges if edge is not ConversationState.APPLY)
    return edges


def render_classifier_content(snapshot: MachineSnapshot, message: str) -> str:
    """The classifier's whole world, rendered as markdown SECTIONS: the scoped
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
    states = "\n".join(
        f"- {target.value}: {EDGE_MEANINGS[(snapshot.state, target)]}"
        for target in presented_edges(snapshot)
    )
    sections.append(f"{_STATES_HEADER}\n{states}")
    return "\n\n".join(sections)


def next_state(current: ConversationState, decision: StateDecision) -> ConversationState:
    """Fail → stay: only a DECIDED draw moves the machine.  A contract failure
    (untagged, out-of-union, poison-exhausted) is a NON-decision — the machine
    holds its state, so a flaky draw can never eject a parked teach loop.  A
    *classified* bail is different: that is the explicit break-out edge, and it
    arrives here as a DECIDED transition to idle."""
    if decision.outcome is StateDrawOutcome.DECIDED and decision.state is not None:
        return decision.state
    return current


async def build_snapshot(
    db: Database,
    embedding_client: LlmClient,
    *,
    state: ConversationState,
    message: str,
    penny_last_turn: str | None = None,
    task_anchor: str | None = None,
) -> MachineSnapshot:
    """The production snapshot builder — the structural pre-pass that turns the
    machine's situation into the classifier's input.  Embeds the incoming
    message and ranks the skill registry by description-anchor cosine
    (``resolve_by_meaning``, capped at ``FIND_MATCH_LIMIT`` — the same
    resolution surface ``find`` uses, no new threshold), so the classifier
    picks among presented evidence and never retrieves.

    A transient embed failure degrades to NO candidates — logged, and safe by
    construction: with no candidates the ``apply`` edge is structurally
    withheld (``presented_edges``), the perception twin of fail → stay."""
    candidates: list[SkillCandidate] = []
    vector = await embed_text(embedding_client, message)
    if vector is None:
        logger.warning("Snapshot builder: message embed failed — no skill candidates offered")
    else:
        candidates = [
            SkillCandidate(
                name=skill.name,
                description=skill.description,
                parameters=parameters_from_json(skill.parameters),
            )
            for skill in db.skills.resolve_by_meaning(vector, PennyConstants.FIND_MATCH_LIMIT)
        ]
    return MachineSnapshot(
        state=state,
        penny_last_turn=penny_last_turn,
        task_anchor=task_anchor,
        skill_candidates=candidates,
    )
