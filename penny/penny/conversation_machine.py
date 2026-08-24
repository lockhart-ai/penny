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
- **Entering learn FRAMES the round** (:class:`RoundFraming`, #1868): a move that
  lands in learn draws the routine's interface from the user's own turns and
  builds the container its results are kept in, before the turn runs — so the
  demonstrated write copies a rendered anchor instead of inventing a name.  The
  framing is round state with the anchor's own lifecycle (set on entry, carried
  while parked, cleared at idle), and it is the framer collaborator's whole
  visible surface here: with none injected, every move is unframed.
- **Entering apply RETRIES the framing when the round has none** (#1875): apply
  configures the round's own container, so a framing is the one thing that turn
  cannot do without — a move landing there without one draws it here, once, and a
  round already carrying one is answered by what it already has.  A retry that
  fails leaves the move unframed, which is the state the turn fails honestly on.
- **A cold apply the words fall SHORT of lands in request** (:class:`RoundShortfall`,
  #1885): the binder's ``MissingParameters`` is an enumerated outcome, not a failure —
  the routine really does cover the ask — so it ROUTES the move (:func:`_landing`)
  instead of failing the turn.  No classifier condition is added or changed for it: the
  idle → request edge already exists and still fires on its own, while this is a SECOND
  door no edge declares, because only the binder can tell a covered-and-bound ask from a
  covered-but-short one — it is the one that reads the words against what the routine
  declares.  Nothing is built for it: the container's derived name needs every value.
- **Entering request BINDS the round too, from either door** (#1894): a request the
  classifier drew directly runs the SAME binder against the skill it named, so both doors
  produce one :class:`RoundShortfall` and the turn is instructed from a partial binding
  rather than from a generic ask.  That binding is ROUND STATE — recorded on the move
  (``state_transition.round_shortfall``) with the shape ``skill_frame`` has, carried by
  the state that reads it (only a move landing in request keeps one; a round that gets its
  details bound carries a framing instead, and idle clears both).  So a later turn READS
  what the round is waiting on: the classifier is shown the settled values and the named
  gap, and the apply draw completes the binding from those values plus the arriving
  message instead of re-deriving the whole thing from raw conversation.
- **An idle landing ENDS the round, and the container goes with it**
  (:meth:`ConversationMachine._end_round`, #1896): a bail back to idle preserves nothing —
  the anchor, the framing and the binding are all dropped from the move, and the container
  the framing pointed at is ARCHIVED, since clearing the row alone would leave an inert
  collection named for a job nobody is doing and nothing able to reach it.  Archive, never
  delete: the same job taught again revives it.

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
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import TransitionCause
from penny.prompts import Prompt
from penny.tools.micro_context import (
    SKILL_TAG,
    MicroContext,
    SkillSignature,
    StateDraw,
    StateDrawOutcome,
)

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm import LlmClient
    from penny.round_framing import RoundFramer

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
#
# IDLE's definition carries the TASK-LIFETIME boundary too, which is why it says
# what it says.  Each door out of idle states the boundary in its own terms — the
# three below are keyed to STARTING a task, and each names changing-what-exists as
# not that — but a condition is only read while its own bullet is being weighed,
# and a draw that ruled a door out early never reaches the rest.  On the DEFINITION
# it is read once, on every from-idle draw, before any edge is considered: what the
# state IS rather than the same clause repeated at each door.
STATE_DEFINITIONS: dict[ConversationState, str] = {
    ConversationState.IDLE: (
        "ordinary conversation: chat, questions, passing mentions, anything put off for "
        "later, and every request handled right now in the conversation — including "
        "commands about things that already exist. A message with no standing or "
        "scheduling component stays here, whatever it resembles: skills exist only for "
        "tasks that keep running on their own"
    ),
    ConversationState.ELICIT: (
        "the user wants something set up to keep running on its own, no known skill "
        "covers it, and the assistant is asking to be taught the steps"
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
#
# The two SKILL-GATED conditions are keyed to STARTING a task (#1927, code-owner ruling:
# "request is only about starting a new task; changes to existing collections don't
# count").  Both ask whether the message asks to SET A KNOWN SKILL RUNNING — apply when it
# also supplies everything that skill needs, request when something is missing — and both
# carry the same boundary sentence, word for word: changing how something ALREADY SET UP
# behaves is not asking to start it.  Carrying it twice is not one edge describing
# another; it is one fact about what these two doors are FOR, and a reader who rules apply
# out for missing information reads request next.
#
# THIS IS THE BEST MEASURED WORDING, AND IT IS NOT A FIX.  Three rounds tried to close the
# gap it leaves and every one cost more than it bought, so the text stands where round 1
# left it and the residual is recorded rather than papered over:
#
#   R1  this text          target 0.60 · every guard 1.00
#   R2  + "starting to hear about a job is not starting the job"
#                          target 0.40 · idle → elicit cross-domain 1.00 → 0.60
#   R3  wholesale restatement, + "jobs already running are not listed here"
#                          target 0.40 · idle → elicit uncovered 1.00 → 0.60
#   R4  reverted to R1
#
# The target is an ask to turn one RUNNING job's notifications on, which this wording
# holds 3 times in 5; its off-direction holds 5 of 5, and that asymmetry is the finding.
# Only the ON direction can be phrased as starting something, and the two leaks do not
# argue with the boundary — they never reach it.  Both stop at a gap this text cannot
# fill: whether the job the message names EXISTS.  Verbatim, from the draw that leaked:
# "they say 'the modular listing watch' probably already set up?  But there's no known
# skill currently running." — the reader looked, and the slice renders the skill REGISTRY
# and no standing jobs, so absence was the only evidence available and it read as decisive.
# THAT is why no condition text closes this: the reader must VERIFY a job exists, and a
# sentence cannot substitute for the lookup.  The recorded next step is a snapshot that
# renders the standing jobs, which turns the read into exactly that lookup (#1927; the
# code owner rules on it separately).
#
# WATCHED DELETIONS — three formulations, each with what it measured:
#
# (1) The once-covers-repeatedly sentences ("A skill does the task once.  The schedule and
#     notifications are added when it is set up…").  They earned their place under the old
#     COVERAGE keying, making a one-shot routine description comparable to an ongoing ask,
#     but under the START keying their second sentence asserts the thing the boundary must
#     deny — that notifications are what setting a job up settles.  Cost measured: none.
#     Every guard case held at 1.00 without them.
#
# (2) "That holds even when the change itself is phrased as starting something: starting to
#     hear about a job is not starting the job."  It treated a start-VERB as the thing to
#     correct, when the leaks' actual gap was not knowing the job existed.  Cost: the target
#     fell 0.60 → 0.40 AND cross-domain fell 1.00 → 0.60 — two cross-domain draws ruled out
#     apply and request and jumped straight to the default ("this is a request to perform
#     some action that does not match a skill, so should default to idle"), never reaching
#     the elicit bullet, whose own wording never changed.  These two render FIRST, so a
#     clause added here is paid for by an edge further down the list.
#
# (3) "Jobs already running are not listed here, so believe them when they speak of one as
#     going."  The schema statement — true, and aimed at the real gap.  Cost: the target did
#     not move (0.40) and it broke the OTHER neighbour, idle → elicit uncovered 1.00 → 0.60,
#     which is its own inverse: an uncovered SETUP ask read as a reference to a job already
#     running.  Licensing the assumption in prose cannot distinguish the two, because the
#     distinction is a fact about the world and not about the words.
#
# What all three guarded is the idle → apply / idle → request / idle → elicit cases' to
# gate, and they are its gate now.
#
# The IDLE → LEARN condition is the ONE edge whose wording differs from its elicit-parked
# twin, and the one that names a sibling outright (code-owner authored, after the #1898
# first pass).  Both are deliberate.  The wording: "instructions to follow for the task
# being worked on" is accurate from a parked round and FALSE at cold idle, where no task
# is in flight — the same message read as disqualifying because there was nothing it could
# be instructions FOR.  So this edge says what a teach IS from a standing start: the user
# says they are teaching, and the steps are in the message.  The sibling clause: with a
# registry of same-kind routines, 8 of 25 measured draws read a taught subject as covered
# and went to apply, which is the skill-gated conditions being read FIRST and answered
# honestly — a teach is not something they can say anything about, so the fact that a
# user's own steps outrank a close-looking routine has nowhere else to live.  The
# ELICIT → LEARN and parked-LEARN conditions are untouched: from inside a round there IS
# a task being worked on, and the correction shape is what that edge is about.
TRANSITIONS: dict[tuple[ConversationState, ConversationState], str] = {
    (ConversationState.IDLE, ConversationState.APPLY): (
        "their message asks to set one of the known skills running, and supplies "
        "everything that skill needs. Changing how something already set up behaves — "
        "its notifications, when it runs, what it covers — is not asking to start it. "
        f"Add a second line naming that skill: {SKILL_TAG} <its name, exactly as quoted "
        "in Known skills>"
    ),
    (ConversationState.IDLE, ConversationState.REQUEST): (
        "their message asks to set one of the known skills running, but something that "
        "skill needs is missing from their message. Changing how something already set "
        "up behaves — its notifications, when it runs, what it covers — is not asking "
        f"to start it. Add a second line naming that skill: {SKILL_TAG} <its name, "
        "exactly as quoted in Known skills>"
    ),
    (ConversationState.IDLE, ConversationState.ELICIT): (
        "they are asking to set up something that keeps running on its own after this "
        "conversation — a job that watches, repeats, or fires again later without being "
        "asked — and no known skill covers it. A request to do something once, right "
        "now, is not this, however many steps it takes — and saving or remembering the "
        "result does not make it ongoing"
    ),
    (ConversationState.IDLE, ConversationState.LEARN): (
        "the user is teaching a new routine: they say so ('let me teach you', "
        "'here's how', 'new job for you') and their message carries the steps — "
        "what to read, what to look for, what to remember. When they are "
        "teaching, choose learn even if a known skill looks close — their steps "
        "are the new way to do it."
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

# The states whose ENTRY the binder settles (#1894) — the same two, read as states rather
# than as the draw's wire values, because a routine was named either way and what the round
# needs to know is which of its parameters the words fill.  They are one tuple rather than
# two branches so a third skill-gated landing cannot be added to the draw contract and
# silently miss the binder.
_BINDING_STATES = (ConversationState.APPLY, ConversationState.REQUEST)

# The conversation-slice section headers — fixed strings, whole-render pinned.
# Markdown headers, not label lines: the parked, populated contexts this slice
# grows into (many candidates, quoted turns that carry their own lists and
# colons) need STRUCTURAL section boundaries the model can navigate, not
# typographic ones a long context swallows.
_LAST_TURN_HEADER = "## The assistant's last message"
_TASK_HEADER = "## The task being worked on"
_WAITING_HEADER = "## The details this task is waiting on"
_SKILLS_HEADER = "## Known skills"
_JOBS_HEADER = "## Jobs already running"
_MESSAGE_HEADER = "## The user's newest message"
_CURRENT_STATE_HEADER = "## Current state"
_TRANSITIONS_HEADER = "## Transitions"
_NONE_PLACEHOLDER = "(none)"

# The parked round's partial binding, as the classifier reads it (#1894).  Its item shapes
# deliberately match the ones the request turn's own instruction renders — same facts, same
# words, so the two surfaces cannot disagree from different evidence — without being single
# sourced with them, the same call ``SkillCandidate.render`` makes against
# ``render_skill_brief`` (this module is a leaf, and a chat-prompt edit is not a classifier
# edit).  The skill is QUOTED for the reason it is quoted in the candidate list: the move
# out of request has to copy that name back on its ``SKILL:`` line.
_WAITING_SKILL = 'skill: "{skill}"'
_WAITING_GIVEN_HEADER = "already given:"
_WAITING_GIVEN_ITEM = "- {name}: {value}"
_WAITING_NOTHING_GIVEN = "- nothing yet"
_WAITING_MISSING_HEADER = "still needed:"
_WAITING_MISSING_ITEM = "- {name} — {description}"
_WAITING_MISSING_NAME_ONLY = "- {name}"

# One standing job's line (#1927).  Every clause renders in BOTH directions — a job with
# no routine says so, notifications state which way they are set, an unscheduled job says
# that — for the reason the skills section renders "(none)": this section exists so an
# absence is READ rather than inferred, and a clause that vanishes when it is false makes
# the reader infer again.  The name is QUOTED like a skill candidate's, since it is the
# anchor a message's words have to be resolved against.
_JOB_LINE = '- "{name}" — {routine} · {notify} · {schedule}'
_JOB_ROUTINE = 'runs "{skill}"'
_JOB_NO_ROUTINE = "no routine attached"
_JOB_NOTIFY_ON = "notifications on"
_JOB_NOTIFY_OFF = "notifications off"
_JOB_NO_SCHEDULE = "no schedule"


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


class StandingJob(BaseModel):
    """One job the user ALREADY has running — the light, cycle-free projection of a
    CONFIGURED collection's row (``SkillCandidate``'s sibling, and a leaf for the same
    reason; ``build_snapshot`` maps the store rows in).

    It exists because the classifier was asked a question its slice could not answer
    (#1927).  A message that says "turn notifications on for the camera kit price watch"
    refers to something the registry of SKILLS says nothing about — a skill is a routine
    that COULD run, a job is one that IS running — and the reader, finding no such thing
    in the only registry rendered, concluded the message must be asking to start one.
    Verbatim, from the draw that leaked: "they say 'the modular listing watch' probably
    already set up?  But there's no known skill currently running."  It looked, and the
    slice had nothing to look at.

    What each field is FOR is the question it lets a message be judged against: ``name``
    is the anchor the user's own words resolve to, ``skill_name`` says which routine it
    runs (so a job and the skill it came from are one thing, not two), and ``notify`` +
    ``schedule`` are two of the three axes the doors' boundary names — so an ask to change
    one of them can be read as a change to something rendered right here."""

    name: str
    skill_name: str | None = None
    notify: bool = False
    schedule: str | None = None

    def render(self) -> str:
        """The one-line job render: the quoted name, the routine it runs, which way its
        notifications are set, and its schedule verbatim.

        The schedule renders VERBATIM (display form == invocation form, the rule the
        stored RRULE already keeps everywhere else it is shown), and the notify clause
        states BOTH directions — the state this section exists to make readable is
        exactly the one an "adjust it" message is about."""
        routine = (
            _JOB_ROUTINE.format(skill=self.skill_name)
            if self.skill_name is not None
            else _JOB_NO_ROUTINE
        )
        return _JOB_LINE.format(
            name=self.name,
            routine=routine,
            notify=_JOB_NOTIFY_ON if self.notify else _JOB_NOTIFY_OFF,
            schedule=self.schedule if self.schedule is not None else _JOB_NO_SCHEDULE,
        )


class RoundFraming(BaseModel):
    """What a round is FOR, settled when the machine entered learn (#1868) — the framer's
    signature and the name of the container Python built from it.

    It is a fact about the ROUND, so it lives where the round's other facts live: on the
    transition row, beside the anchor and the bound skill.  A draw varies; a re-draw is
    not a re-read — so the entry decision is recorded once and every later reader (the
    turn's own instruction, run-end extraction, a correction re-entering learn) reads THAT
    rather than asking the model again and getting a different answer.

    The skill row itself still enters the registry only at run end, from the ledger
    (certified-by-execution): this is the round's framing, never a stub skill.

    ``container`` is a by-name reference to ``memory.name``, the plain-column form the
    registry already uses for a re-creatable thing (``memory.skill_name``,
    ``messagelog.mechanism``).  It is derivable from the signature
    (``derive_collection_name``), and it is recorded anyway because it is what was
    actually built: the container a later turn archives or writes into is the one that
    exists, never one re-derived from a scheme that may have moved."""

    signature: SkillSignature
    container: str

    @property
    def skill(self) -> str:
        """What the round's routine is CALLED — the name run-end extraction files the
        skill under, so it is also the name a later turn configures the container with."""
        return self.signature.name

    def bound_values(self) -> dict[str, str]:
        """What the round POINTED the routine at: one value per declared parameter, each
        a literal span of the user's own words (the framer's structural guarantee).

        The apply turn binds exactly this (#1869) — the round already settled it, so the
        values are READ off the round rather than re-supplied by a model that would be
        guessing at spans it can no longer see."""
        return {parameter.name: parameter.value for parameter in self.signature.parameters}


class RoundShortfall(BaseModel):
    """What a round's entry FELL SHORT of (#1885/#1894) — the round's OTHER entry answer,
    and the state a request turn is negotiated from.

    A routine was named — by an apply draw the binder then found the words short of, or by
    a request draw that said so outright — and the binder read the user's words against
    what that routine declares.  That is an enumerated outcome rather than a failure: the
    routine really does cover the ask, so the turn is not failed, it is turned into the ask
    for what is missing.  No container is built: the container's name is derived from the
    routine plus ALL its values, so a job missing one of them has no name yet and there is
    nothing to create.

    It carries everything that ask has to be written from, and nothing else: the routine's
    registry ``skill`` name and ``description`` (what it is for, in words the user can be
    answered in), the values the words DID settle (``bound``, keyed and ordered by the
    routine's declared parameters, each a literal span of what the user said), and the
    parameters that got none (``missing``, in declared order, each with the registry's own
    one line of what to supply).  ``CandidateParameter`` is reused for those because it is
    exactly that pair — a declared input's name and what to supply for it.

    It is ROUND STATE (#1894), recorded on the move that settled it and read back on every
    later turn of the round — the same treatment ``RoundFraming`` gets, for the same
    reason: a draw varies, so what the round is waiting on is READ rather than re-decided
    each turn against the whole conversation.  Its lifecycle is the state that can read it:
    a move landing in request carries one, a move that binds the round carries a framing
    instead, and idle clears it."""

    skill: str
    description: str
    bound: dict[str, str] = {}
    missing: tuple[CandidateParameter, ...] = ()


class ReplacedSkill(BaseModel):
    """The routine that stood under a round's pinned name BEFORE the round opened
    (#1902) — the whole of what a bail has to put back.

    It is the row's whole content rather than a marker, because skipping the delete is
    not enough to undo a re-teach: by the time the user bails, the round's own
    extraction has already replaced what the canonical routine DOES, so leaving the row
    standing would leave an abandoned, half-corrected program live under a name existing
    jobs still run.  Restoring it is what makes "a bail preserves nothing" true on this
    side too — the registry goes back to exactly the state the round found it in,
    timestamps included.

    ``description_embedding`` rides base64-encoded rather than as the stored blob or a
    float list, because round state has to be JSON and this row is re-serialized onto
    every move of the round: the vector is the bulk of it, and base64 of the blob is a
    quarter the size of the same floats spelled out."""

    name: str
    steps: str
    parameters: str
    intent: str
    description: str
    description_embedding: str | None = None
    source_run_id: str | None = None
    author: str
    created_at: datetime
    updated_at: datetime


class RoundProvenance(BaseModel):
    """What a round's own registry write REPLACED, settled when the round's routine was
    minted (#1902) — the round's third piece of entry state, beside the framing and the
    shortfall.

    Run-end extraction writes under the name the round's framing pinned, so a round
    TEACHING a new job creates that row while a round RE-TEACHING one the registry already
    holds overwrites it — and from the row alone those two writes are identical.  Which one
    happened is a fact about the ROUND, so it is settled at the round's entry and carried
    on the transition row, and a bail READS it rather than re-deciding it.

    Three states, each a different answer to "what does calling this round off owe the
    registry", and the type is what tells them apart:

    * **absent** (no ``RoundProvenance`` on the move) — the round MINTED nothing.  A
      skill-gated round binds a routine the registry already holds and teaches nothing, so
      it replaced nothing and a bail leaves the registry alone.  This is the quiet default:
      only the framer's minting entry records provenance at all.
    * **present, ``replaced`` empty** — the round minted its name over nothing, so the
      routine standing there is the round's own and a bail DELETES it.
    * **present, ``replaced`` set** — the round is re-teaching a routine the user already
      had, so a bail RESTORES that version.

    There is no draft flag anywhere in this: promotion is implicit SURVIVAL, and this type
    exists only for the one landing that does not survive."""

    replaced: ReplacedSkill | None = None


# What settling a round's ENTRY produces — the two enumerated answers, as two types rather
# than one carrying an emptiable field, so a caller holding a framing can never be holding
# an incomplete one (the same carve ``SkillBinding`` makes one level down).
RoundEntry = RoundFraming | RoundShortfall


def framing_of(entry: RoundEntry | None) -> RoundFraming | None:
    """The entry's FRAMING half, or ``None`` when it settled the other way.

    Named once and read everywhere rather than re-tested at each site: which half an entry
    is decides three separate things (where the move lands, what is recorded on it, what
    the turn is instructed with), and three independent readings of one union are three
    places for it to be read differently."""
    return entry if isinstance(entry, RoundFraming) else None


def shortfall_of(entry: RoundEntry | None) -> RoundShortfall | None:
    """The entry's SHORTFALL half, or ``None`` when it settled the other way."""
    return entry if isinstance(entry, RoundShortfall) else None


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
    retrieve); empty means the ``apply`` edge is withheld entirely.

    ``round_binding`` is what a round parked in request is WAITING ON (#1894) — the
    routine it named, the values the earlier words settled, and the parameters that got
    none — read off the machine rather than re-derived, so the move out of request is
    judged against a named gap instead of a fresh completeness audit of the conversation.
    Present only while a round is parked that way, which is the only state that carries
    one.

    ``standing_jobs`` are the jobs the user ALREADY has running (#1927) — the configured
    collections, whole, the same no-ranking-no-cap rule ``skill_candidates`` keeps.  They
    are what makes "is this ask about something that already exists?" a READ; unlike the
    candidates, an empty list withholds no edge, because a message can refer to a job
    whether or not any exists and the answer to that is what the section carries."""

    state: ConversationState
    penny_last_turn: str | None = None
    task_anchor: str | None = None
    skill_candidates: list[SkillCandidate] = []
    standing_jobs: list[StandingJob] = []
    round_binding: RoundShortfall | None = None


class StateDecision(BaseModel):
    """One classification, typed for the machine: the draw outcome plus the
    decided state (``None`` on any non-decision — the fail → stay input) and,
    for an apply decision, the covering skill's name (validated a member of the
    offered candidates by the draw contract — never ``None`` on apply)."""

    outcome: StateDrawOutcome
    state: ConversationState | None = None
    skill: str | None = None


class TurnEntry(BaseModel):
    """What the machine settled for the message that just arrived — what the caller needs
    to enter the turn with.

    ``state`` is where the machine now stands, ``decision`` the classifier's own typed
    answer, and ``shortfall`` what the round is waiting on (#1885/#1894) — whether this
    turn's binder drew it or an earlier turn of the same round did, and absent whenever the
    round holds none (a routine the registry does not hold, a draw that came back unusable,
    or a landing that is not request at all).  State and shortfall are both READS of the log
    taken after the move was recorded (#1894), never predictions and never copies: the row
    is the machine, so what the turn is entered with is what the machine says it is."""

    state: ConversationState
    decision: StateDecision
    shortfall: RoundShortfall | None = None


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
    contrast, renders only when parked: no meaning references an absent task.

    The JOBS ALREADY RUNNING section always renders for the identical reason, and it is
    the reason it was added (#1927): both skill-gated conditions turn on whether the ask
    STARTS something, and that cannot be decided without knowing what is already going.
    Reading nothing there was measured as reading "nothing is running" — which was true of
    the section that did not exist, not of the world.  It renders BELOW the skills, since
    a job is what a skill becomes once it is set up, and ABOVE the newest message, so the
    words that name a job are read after the jobs are.

    The parked round's BINDING renders beside the anchor for the same reason and on the
    same rule (#1894): it is what the round is waiting on, so it belongs with the task it
    belongs to, and it is there only while a round is parked waiting for it."""
    sections = [f"{_LAST_TURN_HEADER}\n{snapshot.penny_last_turn or _NONE_PLACEHOLDER}"]
    if snapshot.task_anchor is not None:
        sections.append(f"{_TASK_HEADER}\n{snapshot.task_anchor}")
    if snapshot.round_binding is not None:
        sections.append(_waiting_section(snapshot.round_binding))
    if snapshot.skill_candidates:
        listing = "\n".join(f"- {candidate.render()}" for candidate in snapshot.skill_candidates)
        sections.append(f"{_SKILLS_HEADER}\n{listing}")
    else:
        sections.append(f"{_SKILLS_HEADER}\n{_NONE_PLACEHOLDER}")
    sections.append(_jobs_section(snapshot.standing_jobs))
    sections.append(f"{_MESSAGE_HEADER}\n{message}")
    current = f"{snapshot.state.value} — {STATE_DEFINITIONS[snapshot.state]}"
    sections.append(f"{_CURRENT_STATE_HEADER}\n{current}")
    transitions = "\n".join(
        f"- {target.value} — {_transition_condition(snapshot.state, target)}"
        for target in presented_edges(snapshot)
    )
    sections.append(f"{_TRANSITIONS_HEADER}\n{transitions}")
    return "\n\n".join(sections)


def _jobs_section(jobs: list[StandingJob]) -> str:
    """What is already running, as its own section — ``(none)`` when nothing is (#1927).

    The empty case is the one that had to be stated: a reader asked whether a message
    refers to an existing job answers from this section either way, and the whole finding
    behind the section is that a missing answer gets supplied by inference."""
    if not jobs:
        return f"{_JOBS_HEADER}\n{_NONE_PLACEHOLDER}"
    listing = "\n".join(job.render() for job in jobs)
    return f"{_JOBS_HEADER}\n{listing}"


def _waiting_section(binding: RoundShortfall) -> str:
    """What the parked round has and what it lacks, as its own section (#1894).

    Both halves render because the move out of request is judged against both: whether the
    newest message supplies what is still needed, and — since a user often restates what
    they already said — whether it adds anything at all.  The values are the user's own
    words, so the section is a READ, and the skill's name is right there to copy back."""
    lines = [_WAITING_HEADER, _WAITING_SKILL.format(skill=binding.skill)]
    lines.append(_WAITING_GIVEN_HEADER)
    lines.extend(_waiting_given_lines(binding))
    lines.append(_WAITING_MISSING_HEADER)
    lines.extend(_waiting_missing_lines(binding))
    return "\n".join(lines)


def _waiting_given_lines(binding: RoundShortfall) -> list[str]:
    """One line per value the round already settled, VERBATIM — or the stated
    nothing-yet line, because an empty list under a header reads as a rendering fault
    rather than as a fact."""
    if not binding.bound:
        return [_WAITING_NOTHING_GIVEN]
    return [
        _WAITING_GIVEN_ITEM.format(name=name, value=value) for name, value in binding.bound.items()
    ]


def _waiting_missing_lines(binding: RoundShortfall) -> list[str]:
    """One line per detail still missing, with the registry's own what-to-supply — the
    plain-words form of the gap the newest message either fills or does not.  A parameter
    with no description falls back to its bare name (a description is optional on the
    row)."""
    return [
        _WAITING_MISSING_ITEM.format(name=parameter.name, description=parameter.description)
        if parameter.description
        else _WAITING_MISSING_NAME_ONLY.format(name=parameter.name)
        for parameter in binding.missing
    ]


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


def _opens_the_teaching(current: ConversationState, decision: StateDecision) -> bool:
    """Whether this move is the round's FIRST entry into learn (#1902) — the one moment a
    round's provenance is readable, and the only move that mints a routine.

    Two facts, both structural.  Only a LEARN turn registers a routine, so a move arriving
    from anywhere else is a move made before the round has written anything: what stands in
    the registry then is what came before the round.  And a decision that lands in learn is
    the FRAMER's — a skill-gated landing binds a routine the registry already holds, which
    is a round that teaches nothing and therefore replaces nothing."""
    return decision.state is ConversationState.LEARN and current is not ConversationState.LEARN


def _landing(
    current: ConversationState, decision: StateDecision, entry: RoundEntry | None
) -> ConversationState:
    """Where the message actually LEAVES the machine: what the draw decided, redirected to
    request when the round's entry came back a shortfall (#1885).

    This is the one place the binder's enumerated outcome is a routing signal.  The draw
    itself is unchanged and needs no new condition — a covering routine whose values the
    words fall short of is exactly what the classifier means by apply, and only the binder
    can tell the two apart, because only the binder reads the words against what that
    routine declares.  So the DRAW settles which routine, and the BINDING settles whether
    the turn can stand it up or has to ask for the rest.

    It is a SECOND door into request, not a new edge.  ``OUT_EDGES`` already offers request
    from idle and that condition is untouched — a classifier that can see the ask is short
    still parks there directly, and since #1894 that turn is bound at entry too, so both
    doors arrive carrying the same partial binding.  This one is the door no edge declares,
    which is why ``OUT_EDGES`` gains nothing: it opens on a fact only the binder holds, and
    it is also the only way a machine already sitting in request lands back in request (the
    move request has no offered self-edge for).

    The move is still recorded as a classifier move, because a model WAS in the loop and
    its draw is what bound the routine; the raw ``STATE: apply`` it drew stays readable on
    its own promptlog row."""
    if shortfall_of(entry) is not None:
        return ConversationState.REQUEST
    return next_state(current, decision)


def build_snapshot(
    db: Database,
    *,
    state: ConversationState,
    message: str,
    penny_last_turn: str | None = None,
    task_anchor: str | None = None,
    round_binding: RoundShortfall | None = None,
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
        round_binding=round_binding,
        standing_jobs=_standing_jobs(db),
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


def _standing_jobs(db: Database) -> list[StandingJob]:
    """The jobs the user has running, as the classifier reads them (#1927).

    A job is a CONFIGURED collection: it carries an ``extraction_prompt``, which is
    exactly the condition the dispatcher selects on (``Collector._is_ready``) — so what
    this section calls running and what actually runs are one definition rather than two
    that can drift.  An INERT collection is storage the user built and no job at all
    (#1629), and an ARCHIVED one is a retired tombstone; neither is something an ask can
    be about changing, so neither renders.

    Every one is offered, the ``skill_candidates`` rule applied to the other registry: a
    relevance gate here could hide the very job a message names, which is the failure the
    section exists to fix."""
    from penny.database.memory.types import MemoryType

    return [
        StandingJob(
            name=row.name,
            skill_name=row.skill_name,
            notify=row.notify,
            schedule=row.schedule,
        )
        for row in db.memories.list_all()
        if row.type == MemoryType.COLLECTION
        and not row.archived
        and row.extraction_prompt is not None
    ]


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


# What each state RENDERS about the round it is in as its own closing paragraph (#1868) —
# data, keyed by state, so a state absent from this map composes the byte-identical prompt
# it always composed.  Learn is told the container is already there, so the write its
# demonstration makes copies an anchor rather than inventing a destination.  APPLY is not
# here: its instruction NAMES the round inside itself (#1875), because by then the round is
# not one more thing to know about the turn — it is what the turn is configuring.
ROUND_LINES: dict[ConversationState, str] = {
    ConversationState.LEARN: Prompt.ROUND_FRAMING_LINE,
}


# What an apply turn has to have, said where it is missed: apply configures the round's own
# container, so a turn reaching this function without a framing has nothing to configure and
# no instruction that could be composed for it.  It is a programming error rather than a
# case to absorb — the caller fails the turn honestly BEFORE composing a prompt (#1875).
_UNFRAMED_APPLY = (
    "An apply turn has no unframed form — it configures the round's own container, "
    "so the framing has to be settled before its instruction can be composed"
)


def conversation_prompt(
    state: ConversationState,
    framing: RoundFraming | None = None,
    shortfall: RoundShortfall | None = None,
) -> str:
    """The chat system prompt for a state: the invariant physics core with THIS
    state's instruction between head and tail.

    The state's name never renders, and neither does any other state's
    instruction — by the time chat reads this the state is already decided, so
    what it needs is what to do, not where it is.  Indexes ``STATE_INSTRUCTIONS``
    directly: a missing state is a programming error and should raise, never
    quietly compose some other state's prompt.

    ``framing`` is the round's own state (#1868/#1869): the routine the round is about and
    the container its results are kept in, both named VERBATIM, so the write a learn turn
    demonstrates — and the collection an apply turn configures — copy an anchor rather than
    inventing a destination.  Learn renders it as a closing paragraph (:data:`ROUND_LINES`)
    and composes the unchanged prompt without one; apply renders it INSIDE its instruction
    and has no form without one at all.

    ``shortfall`` is the round entry's OTHER answer (#1885), and it is request's closing
    paragraph for the same reason: the routine, what it is for, what the words already
    settled and what they did not are all rendered verbatim, so the ask is a copy rather
    than a guess.  Since #1894 both doors bind at entry and a parked round's binding is read
    back on the turns that follow, so an ordinary request turn is entered with one.  What is
    left formless is the round that has no shortfall to state — the binder could not settle
    it at all, or it settled COMPLETELY on a turn the classifier still parked in request —
    and that composes the byte-identical prompt it always did."""
    return (
        Prompt.CONVERSATION_HEAD
        + _instruction(state, framing, shortfall)
        + Prompt.CONVERSATION_TAIL
    )


def _instruction(
    state: ConversationState, framing: RoundFraming | None, shortfall: RoundShortfall | None
) -> str:
    """The state's instruction as this turn reads it — the round rendered where that state
    says it belongs.

    Apply is the state that STANDS THE ROUND UP, so the round is its subject rather than a
    note appended to it: the container and the routine render inside its own sentences, and
    with no framing there is no turn to instruct.  Every other state's instruction is fixed
    text, with the round's closing line after it when that state has one to say."""
    if state is ConversationState.APPLY:
        if framing is None:
            raise ValueError(_UNFRAMED_APPLY)
        return STATE_INSTRUCTIONS[state].format(skill=framing.skill, container=framing.container)
    return STATE_INSTRUCTIONS[state] + _round_line(state, framing, shortfall)


def _round_line(
    state: ConversationState, framing: RoundFraming | None, shortfall: RoundShortfall | None
) -> str:
    """The round's own closing paragraph — what THIS state has to say about the round it is
    in, read from whichever half of the round's entry settled it.

    Keyed to the state because the two halves answer different questions and only one state
    reads each: learn reads the FRAMING (the routine being taught and the container that
    already exists), request reads the SHORTFALL (the routine that covers the ask and the
    detail still missing).  Elicit reads neither — it is still asking what the task IS — and
    idle carries neither at all."""
    if state is ConversationState.REQUEST:
        return _shortfall_line(shortfall)
    return _framing_line(state, framing)


def _shortfall_line(shortfall: RoundShortfall | None) -> str:
    """The round's shortfall as request's closing paragraph (#1885), or nothing at all.

    Nothing when the round settled none — a routine the registry does not hold, or a draw
    that came back unusable (#1894 binds at both doors, so an ordinary request turn always
    has one): it composes the prompt it always composed, so the rendered state stays
    additive rather than a shape every request turn has to have."""
    if shortfall is None:
        return ""
    return Prompt.ROUND_SHORTFALL_LINE.format(
        skill=shortfall.skill,
        description=shortfall.description,
        bound=_bound_lines(shortfall),
        missing=_missing_lines(shortfall),
    )


def _bound_lines(shortfall: RoundShortfall) -> str:
    """Every value the user's words already settled, one per line and VERBATIM — the half
    of the render that stops the turn asking again for something they have already said.

    Its empty case is a stated line rather than an absent section: "you have none of it
    yet" is a fact the ask is written against, and leaving it to be inferred from a missing
    section is exactly the read a rendered state exists to remove."""
    if not shortfall.bound:
        return Prompt.REQUEST_NOTHING_BOUND
    return "\n".join(
        Prompt.REQUEST_BOUND_ITEM.format(name=name, value=value)
        for name, value in shortfall.bound.items()
    )


def _missing_lines(shortfall: RoundShortfall) -> str:
    """Every detail the routine still needs, each with the registry's own one line of what
    to supply — which is the plain-words form of the same thing, so the ask can be written
    from it without quoting a binding key at the user.

    A parameter with no description falls back to its bare name: a description is optional
    on the row, and rendering an empty clause after a dash would read as a description
    nobody wrote."""
    return "\n".join(
        Prompt.REQUEST_MISSING_ITEM.format(name=parameter.name, description=parameter.description)
        if parameter.description
        else Prompt.REQUEST_MISSING_NAME_ONLY.format(name=parameter.name)
        for parameter in shortfall.missing
    )


def _framing_line(state: ConversationState, framing: RoundFraming | None) -> str:
    """The round's framing as the state's closing paragraph, or nothing at all.

    Keyed to the state because a state with nothing to say about the round says nothing:
    elicit is still asking what the task IS, so it has nothing to say about a container
    that may not exist yet, and idle never carries a framing at all.  Request has its own
    paragraph rather than this one — it reads the shortfall, which is the entry answer it
    actually has.

    The skill and the container render VERBATIM: the whole point of the framing is that a
    name the model would otherwise invent is a name it copies."""
    template = ROUND_LINES.get(state)
    if framing is None or template is None:
        return ""
    return template.format(skill=framing.skill, container=framing.container)


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

    def __init__(
        self, db: Database, classifier: StateClassifier, framer: RoundFramer | None = None
    ) -> None:
        self._db = db
        self._classifier = classifier
        # The learn-ENTRY framer (#1868), injected rather than built here: framing a round
        # means creating its container, which is database + embedding work this module
        # must not import at import time (the leaf discipline — see CandidateParameter).
        # ``None`` leaves every move unframed, which is the pre-#1868 behaviour exactly.
        self._framer = framer

    async def advance(
        self,
        message: str,
        *,
        message_id: int | None = None,
        penny_last_turn: str | None = None,
        run_id: str | None = None,
        run_target: str | None = None,
    ) -> TurnEntry:
        """One incoming message, start to finish: settle any structural move
        first, classify from where that leaves the machine, settle the round's entry when
        the move lands in learn or against a known routine, then record the result (which
        is what applies it).  Returns what the caller enters the turn with.

        Both halves of the entry are recorded, so what comes back is a READ of where that
        left the machine (#1894) rather than a hand-off of something nothing holds: the
        shortfall a turn is entered with is the round's, whether this turn's binder drew it
        or an earlier turn did."""
        state = self._settle_structural(message_id=message_id)
        snapshot = self._snapshot(state, message, penny_last_turn=penny_last_turn)
        decision = await self._classifier.classify(snapshot, message, run_target=run_target)
        entry = await self._frame_round(state, decision, message, run_id=run_id)
        self._record_decision(state, decision, message_id=message_id, run_id=run_id, entry=entry)
        return TurnEntry(state=self.state(), decision=decision, shortfall=self.shortfall())

    def state(self) -> ConversationState:
        """Where the machine stands — the newest move's destination.  No history
        at all is the cold start, and idle is what that means."""
        latest = self._db.machine.latest_transition()
        return ConversationState(latest.to_state) if latest else ConversationState.IDLE

    def framing(self) -> RoundFraming | None:
        """The round's framing (#1868), carried on the newest move — ``None`` when the
        machine is idle, when the round could not be framed, or when nothing frames.

        Read off the log for the same reason the state and the anchor are: the newest row
        IS the machine, so there is nothing else that could disagree with it."""
        latest = self._db.machine.latest_transition()
        if latest is None or latest.skill_frame is None:
            return None
        return RoundFraming.model_validate_json(latest.skill_frame)

    def shortfall(self) -> RoundShortfall | None:
        """The round's partial binding (#1894), carried on the newest move — the routine a
        parked request round named, what the words have settled so far, and what they have
        not.  ``None`` unless a round is parked in request holding one.

        Read off the log exactly like the framing, and for the same reason: the newest row
        IS the machine, so the turn's ask, the classifier's view of the gap, and the
        completion of the binding are all the same fact rather than three derivations of
        it."""
        latest = self._db.machine.latest_transition()
        if latest is None or latest.round_shortfall is None:
            return None
        return RoundShortfall.model_validate_json(latest.round_shortfall)

    def provenance(self) -> RoundProvenance | None:
        """The round's PROVENANCE (#1902), carried on the newest move — what the round's
        own registry write replaced, and ``None`` on every move whose round minted nothing.

        Read off the log exactly like the framing and the binding, and for the same
        reason: whether this round minted its routine, drafted over one the user already
        had, or taught nothing at all is settled once, when the routine is minted.
        Re-deciding it later — after the round's own extraction has written that name —
        would read the round's own work as what the round replaced."""
        latest = self._db.machine.latest_transition()
        if latest is None or latest.round_provenance is None:
            return None
        return RoundProvenance.model_validate_json(latest.round_provenance)

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
        """The classifier's input, with the anchor and the parked round's binding resolved
        from the machine's own log — the ask is READ from the conversation it points at and
        the binding from the move that settled it, never a copy this layer keeps."""
        return build_snapshot(
            self._db,
            state=state,
            message=message,
            penny_last_turn=penny_last_turn,
            task_anchor=self._anchor_text(),
            round_binding=self.shortfall(),
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

    async def _frame_round(
        self,
        current: ConversationState,
        decision: StateDecision,
        message: str,
        *,
        run_id: str | None,
    ) -> RoundEntry | None:
        """The round-framing hook (#1868/#1875/#1870/#1894): the moves that settle the
        round's entry — its framing and the container its results are kept in, or the
        partial binding it is still waiting on — and WHICH of the two draws settles it.

        Its input is the round's user turns and nothing else, which at this moment is the
        ask the round is anchored to plus the message that just arrived — both draws' own
        contract, and both of those exist here, which is why the draw moved to this seam at
        all.

        Only a DECIDED move settles anything, never a fail → stay that merely LEAVES the
        machine where it was: the machine's own rule is that a contract failure moves
        nothing, and building a container (or archiving the one it replaces) off a draw the
        machine refused to act on would be that rule holding for the state and not for the
        registry.  ``decision.state`` says exactly that — it is ``None`` on every
        non-decision.

        ``None`` when the draw failed or nothing frames.  Landing in learn that way runs
        the round unframed, the way it did before this hook existed; landing in APPLY that
        way leaves a turn with nothing to configure, which its own state fails honestly.  A
        SHORTFALL is neither of those (#1885): the routine covers the ask and the words are
        short of something it needs, so the turn asks for the rest — and the move lands in
        request whether the draw said request outright or apply (:func:`_landing`).

        BOTH skill-gated draws bind (:data:`_BINDING_STATES`, #1894), which is what makes
        the two doors into request one door: a request the classifier drew directly is a
        routine it named and details it says are missing, which is the binder's question
        exactly, and answering it here is what lets that turn ask for something specific."""
        framer = self._framer
        if framer is None or decision.state is None:
            return None
        if decision.state is ConversationState.LEARN:
            return await self._frame_learn_entry(framer, current, message, run_id=run_id)
        if decision.state in _BINDING_STATES and self.framing() is None:
            return await self._bind_round_entry(framer, decision, message, run_id=run_id)
        return None

    async def _frame_learn_entry(
        self,
        framer: RoundFramer,
        current: ConversationState,
        message: str,
        *,
        run_id: str | None,
    ) -> RoundFraming | None:
        """A round being TAUGHT has no routine yet, so the framer MINTS one (#1868) — on
        the round's FIRST entry into learn, and only that one (#1902).

        A move from learn back into learn is a correction, and a correction refines the
        PROGRAM of the round's one job rather than deciding what that job is: the round
        settled its identity when it entered, the turn that demonstrated it was instructed
        under that name, and the container derived from it is where the round has been
        writing.  So the framing carries and only the container is re-settled — the same
        routine, the same container, the corrected write landing in place.

        Re-drawing is what forked a round in two: a corrected ask read as a different
        subject, the fresh draw derived a fresh name, find-or-create minted a sibling
        container under it, and run-end extraction registered a second routine beside the
        first.

        Nothing is returned on the carry: the framing is round state the machine already
        holds, so the move settles nothing NEW and every lifecycle below reads it that way
        — which is also what keeps the round's provenance from being re-taken over the
        round's own draft."""
        carried = self.framing()
        if current is ConversationState.LEARN and carried is not None:
            await framer.carry_entry(carried, run_id=run_id)
            return None
        return await framer.frame_entry(ask=self._anchor_text(), message=message, run_id=run_id)

    async def _bind_round_entry(
        self, framer: RoundFramer, decision: StateDecision, message: str, *, run_id: str | None
    ) -> RoundEntry | None:
        """A round asking for a routine Penny ALREADY KNOWS has one, so the BINDER fills it
        (#1870/#1894) — the entry draw of every skill-gated landing, apply and request
        alike, and the frameless-apply seam #1875 opened.

        Apply configures the round's own container, so a framing is the one thing that turn
        cannot do without; a round already carrying one is answered by what it already has
        (the caller's guard), since a re-draw is not a re-read and would file the job under
        a name the container it was built for no longer matches.  Arriving WITHOUT one is
        the cold ask: the routine is whatever the skill-gated decision bound, and the
        values are whatever the round's words supply for what that routine declares.

        What the round ALREADY settled is handed over rather than re-derived (#1894), so a
        message arriving on a parked request completes a binding instead of starting one —
        and only the parameters still open are drawn, which is why the value the user gave
        two turns ago cannot come back different this turn.

        A decided skill-gated draw always names a skill (the draw contract validates it
        against the offered candidates), so a missing one is a broken contract rather than a
        case to absorb — it is logged and the round stays unbound, never bound to a guess."""
        if decision.skill is None:
            logger.warning(
                "A %s decision arrived naming no skill — the round cannot be bound, "
                "so it enters its turn unframed",
                decision.state,
            )
            return None
        return await framer.bind_entry(
            skill=decision.skill,
            ask=self._anchor_text(),
            message=message,
            run_id=run_id,
            settled=self._settled_values(decision.skill),
        )

    def _settled_values(self, skill: str) -> dict[str, str]:
        """What a parked round has already bound for ``skill`` (#1894) — read off the
        recorded shortfall, empty for a cold entry.

        Gated on the ROUTINE matching, because a round that turns out to be about a
        different skill is a different job: values bound against one routine's parameters
        say nothing about another's, whatever the names happen to be."""
        waiting = self.shortfall()
        if waiting is None or waiting.skill != skill:
            return {}
        return waiting.bound

    def _record_decision(
        self,
        current: ConversationState,
        decision: StateDecision,
        *,
        message_id: int | None,
        run_id: str | None,
        entry: RoundEntry | None = None,
    ) -> None:
        """Append the move ``_landing`` settles — the write that applies it.

        EVERY draw is appended, including one that moves nothing: without the
        held draws the log reports a perfect classifier by construction, and
        per-edge accuracy is exactly what it exists to make scorable.

        A move that landed in request through a shortfall carries no framing (#1885): the
        container's name needs every value, so a job missing one has nothing built for it
        and nothing to record.  It carries the SHORTFALL instead (#1894) — the two halves
        of the entry, each recorded on the move that settled it.

        The move that MINTS a round's routine also records what stood under that name
        before it (:meth:`_next_provenance`, #1902) — the round's provenance, settled once
        and carried the same way as the other two.

        A landing in IDLE ends the round, so the move drops all three AND what the round
        left in the store is resolved (:meth:`_end_round`, #1896/#1902) — the row's
        clearing and the store's, which are one fact stated in two places."""
        target = _landing(current, decision, entry)
        drawn = framing_of(entry)
        carried = self._next_framing(target, drawn)
        waiting = self._next_shortfall(target, shortfall_of(entry))
        provenance = self._next_provenance(current, target, decision, drawn)
        self._end_round(target, run_id=run_id)
        self._db.machine.record_transition(
            from_state=current.value,
            to_state=target.value,
            cause=TransitionCause.CLASSIFIER,
            anchor_message_id=self._next_anchor(current, target, message_id),
            outcome=decision.outcome.value,
            message_id=message_id,
            run_id=run_id,
            skill_name=decision.skill,
            skill_frame=carried.model_dump_json() if carried is not None else None,
            round_shortfall=waiting.model_dump_json() if waiting is not None else None,
            round_provenance=provenance.model_dump_json() if provenance is not None else None,
        )

    def _end_round(self, target: ConversationState, *, run_id: str | None) -> None:
        """An idle landing ENDS the round, so BOTH things the round left behind go with it
        (#1896/#1902) — the DURABLE half of the clearing the three lifecycles below do on
        the row: the container it built, and the registry entry it wrote.

        ``_next_framing`` drops the framing from the move and this retires what that
        framing pointed at, because dropping it alone leaves an inert collection named for
        a job nobody is doing, and a routine ambient in every later prompt for a job the
        user just called off.  A bail preserves nothing: once the machine is idle the round
        is over, and any next task opens a flow of its own.

        WHAT the registry side does is read from the round's PROVENANCE, never re-decided:
        a round that minted its name loses the routine outright, a round that was
        re-teaching one the user already had puts the pre-round version BACK — because by
        now the round's own extraction has replaced what that routine does, and preserving
        nothing means the registry ends where the round found it — and a round that minted
        nothing at all leaves the registry untouched, since it never wrote to it.

        Read off the state the round is CARRYING, which is why it runs BEFORE the move is
        recorded — and that order is also the safe one: a write that then fails leaves the
        machine parked where it was over an archived container, which the next move back
        into learn revives by name (the framer's find-or-create), while the other order
        would leave an idle machine with nothing pointing at either any more.

        Only a round that was framed has a container or a pinned routine at all, so every
        other landing here is a no-op — and the post-apply reset is not this path (it is
        its own structural row, appended before the draw, and it drops the framing first),
        so a job the user has just set running is never what a later bail retires."""
        if target is not ConversationState.IDLE:
            return
        # Function-local: the framer's module imports this one, and the leaf discipline
        # keeps every database-touching import off this module's import time.
        from penny.round_framing import abandon_round_container, abandon_round_skill

        framing = self.framing()
        abandon_round_container(self._db, framing, run_id=run_id)
        abandon_round_skill(self._db, framing, self.provenance(), run_id=run_id)

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

    def _next_framing(
        self, target: ConversationState, drawn: RoundFraming | None
    ) -> RoundFraming | None:
        """The framing lifecycle, shaped exactly like the anchor's (#1868): idle clears
        it, a fresh entry draw replaces it, and every other move CARRIES it — the round's
        framing belongs to the round, so the turn that accepts what was demonstrated reads
        the same container the turn that demonstrated it wrote into.

        A move that settled nothing carries rather than clears, which is the same rule read
        the other way: a held draw that merely leaves the machine in learn keeps the
        framing the round already has, instead of losing it to a flaky draw.  Since #1902 a
        correction is that case by construction — it settles the framing the round was
        already carrying, so the replacement and the carry are the same framing."""
        if target is ConversationState.IDLE:
            return None
        return drawn if drawn is not None else self.framing()

    def _next_shortfall(
        self, target: ConversationState, drawn: RoundShortfall | None
    ) -> RoundShortfall | None:
        """The partial binding's lifecycle (#1894), one rule: it belongs to a round parked
        in REQUEST, so a move landing there carries one — a freshly drawn binding replacing
        whatever the round held, and anything else keeping what the round already had (the
        same fail-→-stay reading the framing gets) — while every move that leaves clears it.

        Leaving covers the whole of the rest: a round whose details arrived carries a
        FRAMING instead and has nothing left to wait on; one that turns out to be about a
        different task is waiting on nothing that was ever true of it; a draw that could not
        be bound at all lands in apply, which fails honestly and resets; and idle ends the
        round.  A binding kept past any of those would be state describing a negotiation
        that is over — the exact thing the model would then reason from."""
        if target is not ConversationState.REQUEST:
            return None
        return drawn if drawn is not None else self.shortfall()

    def _next_provenance(
        self,
        current: ConversationState,
        target: ConversationState,
        decision: StateDecision,
        drawn: RoundFraming | None,
    ) -> RoundProvenance | None:
        """The round's PROVENANCE lifecycle (#1902), the framing's own shape: idle clears
        it, the move that MINTS the round's routine takes it, every other move carries it.

        Taken on the move that opens the round's teaching and nowhere else, because that is
        the one moment it reads truthfully.  Only a learn turn registers a routine, so
        before the round's first one the registry still holds what came BEFORE the round;
        from inside learn it does not, and the same read would take the round's own work
        for the thing it replaced.  And only the FRAMER mints — a skill-gated entry binds a
        routine the registry already holds and teaches nothing, so such a round replaced
        nothing and records no provenance at all, which is what keeps a bail from request
        away from a routine the round never wrote."""
        if target is ConversationState.IDLE:
            return None
        if _opens_the_teaching(current, decision) and drawn is not None:
            return self._mint_provenance(drawn)
        return self.provenance()

    def _mint_provenance(self, framing: RoundFraming) -> RoundProvenance:
        """The round's provenance at the moment its routine is minted — carrying whatever
        the registry already holds under that pinned name, and carrying nothing when it
        holds none."""
        # Function-local, like ``_end_round``'s: the leaf discipline keeps every
        # database-touching import off this module's import time.
        from penny.round_framing import snapshot_replaced_skill

        return RoundProvenance(replaced=snapshot_replaced_skill(self._db, framing))
