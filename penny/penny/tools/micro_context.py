"""Single-shot micro-context extraction for content tools.

A content tool (``browse``) that carries a micro-instruction runs the fetched
page content through a FRESH, scoped single-shot model call — content +
instruction, no tools — and returns a small typed result to the main loop.  The
bulk page body never enters the parent run's context: only the extracted value
(or an honest enumerated failure) plus the fetch handle to the stored full
content come back (the anchor discipline).  A micro-context is structurally
incapable of confabulating a stored value it has never seen.

The output contract is ENUMERATED on both sides of the interface, and since
#1814 it is **DECLARED AS DATA**: each customer names a
:class:`~penny.tools.micro_context_shape.MicroContextShape` — its tags, which
lines are required, which repeat per item, how each line's payload carves into
named fields, and which of those fields may be absent.  The prompt's contract
block is RENDERED from that declaration (``render_line``), so the tags and
separators the model is told to write are literally the ones the parser splits
on; :func:`~penny.tools.micro_context_shape.parse_draw` is the ONE validate step
every customer rides, so no customer partitions a string itself.  What that
bought is the failure it replaced: a ``PARAM`` line whose separator came back as
an en-dash where the parser wanted an em-dash used to partition to nothing and
hand the entire remainder over as a parameter's semantic name — a 60-character
"name" carrying its own description, persisted as a skill's binding key, with no
reroll because the parse had "succeeded".  Tolerance is now declared once and
"parsed to something implausible" is a contract violation, not a value.

A violation of a REQUIRED part of the shape is rerolled on the unchanged context
and then fails honestly, per customer (``EXTRACTION_FAILED`` / ``INVALID`` /
the slug fallback).  A violation of an OPTIONAL or PER_ITEM line is simply an
ABSENT line — the leaf labeller's per-placeholder lines are best-effort by design,
so a bad line costs that one placeholder its name and nothing else, and the
arg-derived name it falls back to is a legible degradation (#1824).

The single call is screened by the same degeneracy / leaked-Harmony-envelope
detectors the agent-loop reroll guard uses (:mod:`penny.text_validity`): poison
is discarded and re-drawn on the *unchanged* context up to
``DEGENERATE_REROLL_ATTEMPTS``, never appended (appending a collapse feeds it
back in).  An unextractable result is an honest enumerated outcome, never a
silent empty.

It is itself a ledger-visible model call — its own ``agent_name`` /
``prompt_type`` so run traces attribute it — but it does NOT inflate the parent
run's context: the parent only ever sees the returned :class:`MicroContextResult`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import PennyConstants
from penny.text_validity import has_leaked_harmony_envelope, is_degenerate_run
from penny.tools.micro_context_shape import (
    FieldShape,
    FieldSpec,
    LineAnchor,
    LineRole,
    LineSpec,
    MicroContextShape,
    ParsedDraw,
    ParsedLine,
    PayloadSpan,
    Separator,
    parse_draw,
    render_line,
)

if TYPE_CHECKING:
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)


class DrawField(StrEnum):
    """The named fields the declared shapes carve out of a tagged line.

    A customer reads a draw by these names — ``drawn.field(STATE_TAG,
    DrawField.NAME)`` — never by position and never by partitioning a string, so
    a grammar change lands in the declaration rather than in four parsers."""

    VALUE = "value"
    REASON = "reason"
    NAME = "name"
    DESCRIPTION = "description"
    CURRENT = "current"
    SEMANTIC = "semantic"
    SKILL = "skill"


# The two output tags — the enumerated contract, present on BOTH sides of the
# interface: the prompt names them and the classifier parses them.  The label is
# the interface between model-space and Python-space (the enumerated-cases
# doctrine, #1554).  Without it the not-present case comes back as arbitrary
# prose, which a blank-check classifier reads as an extracted value — a
# confabulation-shaped leak through the exact surface whose design guarantee is
# "cannot confabulate stored values".
EXTRACTED_TAG = "EXTRACTED:"
NOT_PRESENT_TAG = "NOT_PRESENT:"

# The extraction shape (#1814): two ALTERNATIVE tags, exactly one of which must
# OPEN the draw — which is also what makes them mutually exclusive without a rule
# saying so.  ``EXTRACTED:`` spans the REMAINDER (the value is everything that
# follows — a digest or an item-per-line list is served whole, #1682) while the
# ``NOT_PRESENT:`` reason stays a single LINE, so a not-present apology can never
# be multi-line-promoted into a value.
_EXTRACTED_LINE = LineSpec(
    tag=EXTRACTED_TAG,
    role=LineRole.ALTERNATIVE,
    anchor=LineAnchor.OPENS_DRAW,
    span=PayloadSpan.REMAINDER,
    fields=(
        FieldSpec(name=DrawField.VALUE, placeholder="<the value — it may begin on this same line>"),
    ),
)
_NOT_PRESENT_LINE = LineSpec(
    tag=NOT_PRESENT_TAG,
    role=LineRole.ALTERNATIVE,
    anchor=LineAnchor.OPENS_DRAW,
    fields=(
        FieldSpec(name=DrawField.REASON, placeholder="<one short line naming what is missing>"),
    ),
)
EXTRACT_SHAPE = MicroContextShape(lines=(_EXTRACTED_LINE, _NOT_PRESENT_LINE))

# The extraction framing — one legible, single-purpose instruction.  It asks a
# world-question ("what's on the page?"), never a machine-question, forbids
# inventing a value not in the content, and enumerates the closed set of output
# forms so classification downstream is a deterministic tag parse, never a
# judgment over free prose.  The value may be as long as the instruction requires
# (a digest, a list) — the TAG must open the output; only its shape is fixed, not
# its length.  The two tagged lines are RENDERED FROM THE SHAPE, so the prompt
# cannot state a contract the parse doesn't read.
MICRO_CONTEXT_SYSTEM_PROMPT = (
    "You are an extraction step. You are given the full text of one or more web "
    "pages and a single instruction naming exactly what to pull out of them. "
    "The FIRST LINE of your output must open with one of these two tags:\n"
    f"{render_line(_EXTRACTED_LINE)}\n"
    f"{render_line(_NOT_PRESENT_LINE)}\n"
    f"After {EXTRACTED_TAG}, the extracted value is EVERYTHING that follows — as "
    "long as the instruction requires: a single value, one or more paragraphs, or "
    "a list (put one item per line). Use "
    f"{NOT_PRESENT_TAG}, on a single line, when the requested information is not in "
    "the content. Never invent a value that is not in the content, and write "
    "nothing outside the value itself — no preamble, no explanation, no restating "
    "the instruction."
)

_USER_TEMPLATE = "Instruction: {instruction}\n\nContent:\n{content}"

# How many draws a CONTRACT-VIOLATING (but poison-free) output gets: the first
# draw plus one reroll of the unchanged context.  A draw that misses a REQUIRED
# part of the declared shape is a contract violation, not a world-fact — it is
# never promoted to a value; after the reroll the customer fails honestly.
_INVALID_DRAW_BUDGET = 2

# ── Second customer: the LEAF LABELLER — the routine's IMPLEMENTATION (#1824) ───
# The leaf labeller is handed the routine's tool calls — nothing else — and NAMES
# every placeholder in them: a short semantic name plus one line saying what belongs
# in that spot each time the routine runs.  That is its entire job.  Its declared
# shape rides the SAME poison-screen + reroll machinery (``_valid_draw``) as every
# other customer, and each line is keyed by the placeholder's CURRENT (arg-derived)
# name so the system owns an unambiguous mapping back.
#
# Why there is no verdict any more (#1824, superseding #1770/#1807): the old contract
# asked this same draw, per candidate, whether the USER supplied that value — an
# INTERFACE question asked of IMPLEMENTATION artifacts, and the category error is what
# put a ceiling on it.  Measured across three independent wordings (#1821/#1823), the
# per-leaf user-supplied verdict pinned at ~0.7-0.8 on the floor case and would not
# move: both a reworded extract instruction and a storage key slugged out of the user's
# own URL are literally the user's words re-worded by the assistant, so no ordering of
# the two cases separates them.  What separates them is whether the THING the value is
# for was asked for — a question about the ROUND, asked ONCE, which is the skill
# framer's (below).  So every leaf is a placeholder here, unconditionally, and this
# draw only ever describes.  Nothing it writes decides anything about the skill's
# interface, which is why its content carries no conversation at all.
PLACEHOLDER_TAG = "PLACEHOLDER"

# One line per offered placeholder — PER_ITEM, so a malformed line costs that one
# placeholder its name and nothing else (the caller then falls back to the arg-derived
# name; absence never blocks extraction).  The semantic name declares
# ``FieldShape.NAME`` because it is an anchor the runtime join reads, so a "name" that
# is really a name plus its own description is a malformed line, not a value.
_LEAF_LINE = LineSpec(
    tag=PLACEHOLDER_TAG,
    role=LineRole.PER_ITEM,
    fields=(
        FieldSpec(name=DrawField.CURRENT, placeholder="<current name>", separator=Separator.COLON),
        FieldSpec(
            name=DrawField.SEMANTIC,
            placeholder="<placeholder_name>",
            shape=FieldShape.NAME,
            separator=Separator.DASH,
        ),
        FieldSpec(
            name=DrawField.DESCRIPTION,
            placeholder="<one line: what belongs in this spot each run>",
        ),
    ),
)
LEAF_LABELLING_SHAPE = MicroContextShape(lines=(_LEAF_LINE,))

LEAF_LABELLING_SYSTEM_PROMPT = (
    "You are naming the placeholders inside a routine. A routine is a fixed sequence "
    "of tool calls that gets run again on new occasions: the values it used the first "
    "time are gone, and each one leaves a placeholder that has to be filled in again "
    "on every run. You are given the calls IN THE ORDER THEY RAN, and the "
    "placeholders — each currently named after the tool argument it fills, and shown "
    "with the value that sat there the first time. For every placeholder:\n"
    "1. Work out what that spot HOLDS, from the call it sits in and what the calls "
    "around it do. The value it held once is an EXAMPLE of what belongs there, never "
    "the definition of it.\n"
    "2. Name it for the KIND of thing that belongs there — a single lowercase word or "
    "snake_case, named for the spot and not for the one value it happened to hold: a "
    "spot holding the body of a message is named for being a message body, never for "
    "the one message it carried.\n"
    "3. Describe in one line what belongs in that spot each time the routine runs.\n"
    "Write ONE line for EVERY placeholder, repeating its CURRENT name exactly so it "
    "maps back:\n"
    f"{render_line(_LEAF_LINE)}\n"
    "Write nothing else — no preamble, no explanation, no restating the routine."
)

# The single per-call ask; the routine's calls + its placeholders are the content.
_LEAF_LABELLING_INSTRUCTION = (
    "Name every placeholder in this routine's tool calls, and say in one line what "
    "belongs in that spot each time the routine runs."
)

# ── Fourth customer: the SKILL FRAMER — the routine's INTERFACE (#1824) ────────
# The framer writes the skill's public signature: what it is called, what it is for,
# and what someone has to say to set it up again.  Its whole evidence is the user's
# own messages — never the tool calls, never the values they carried, never the
# labeller's output — because the interface is decided by what was ASKED FOR, and the
# calls are how that ask was carried out this once.
#
# The deciding question is asked ONCE, positively: given what this skill IS (the name
# and description this same draw writes), which of the pieces the user handed over
# must they say again?  A piece the framing already carries is not a parameter.  That
# is the question the old per-leaf provenance verdict was standing in for — and the
# reason it could not work is that it was asked once PER VALUE, of implementation
# artifacts, when it only has an answer at the level of the round (#1821/#1823's
# measured ceiling).  Name, description and parameters come out of ONE decision, which
# is what stops a skill from calling itself a price watcher and then asking what to
# watch (#1803's defect, now closed by construction rather than by a second opinion).
PARAMETER_TAG = "PARAMETER"

NAME_TAG = "NAME:"
DESCRIPTION_TAG = "DESCRIPTION:"

_FRAMING_NAME_LINE = LineSpec(
    tag=NAME_TAG,
    fields=(FieldSpec(name=DrawField.NAME, placeholder="<a short generic verb-noun name>"),),
)
_FRAMING_DESCRIPTION_LINE = LineSpec(
    tag=DESCRIPTION_TAG,
    fields=(
        FieldSpec(name=DrawField.DESCRIPTION, placeholder="<one line: what the skill is for>"),
    ),
)
# The parameter lines are PER_ITEM: a malformed one costs that parameter and nothing
# else.  The name declares ``FieldShape.NAME`` because it is the skill's BINDING KEY at
# instantiation, so a "name" carrying its own description is a malformed line rather
# than a 60-character key (#1814's motivating failure).
_FRAMED_PARAMETER_LINE = LineSpec(
    tag=PARAMETER_TAG,
    role=LineRole.PER_ITEM,
    fields=(
        FieldSpec(
            name=DrawField.NAME,
            placeholder="<parameter_name>",
            shape=FieldShape.NAME,
            separator=Separator.DASH,
        ),
        FieldSpec(name=DrawField.DESCRIPTION, placeholder="<one line: what to supply for it>"),
    ),
)
SKILL_FRAMING_SHAPE = MicroContextShape(
    lines=(_FRAMING_NAME_LINE, _FRAMING_DESCRIPTION_LINE, _FRAMED_PARAMETER_LINE)
)

SKILL_FRAMING_SYSTEM_PROMPT = (
    "You are writing what a reusable skill IS: what it is called, what it is for, and "
    "what someone has to say to set it up. All you are given is what the user asked "
    "for, in their own words. Do three things:\n"
    "1. From their ask, work out what they were trying to get done. Their own words "
    "are the only evidence, and the point of the ask is what the skill is for.\n"
    "2. Name and describe the SKILL by that: a short verb-noun name for the KIND of "
    "task, generic — never the one occasion — and one line stating what it is for "
    "before any mechanics. A description that falls back on the information being "
    "specified, where the ask named something particular, has dropped the point of the "
    "ask — say what it actually was.\n"
    "3. Now take the pieces of information their ask handed over — every particular "
    "thing they named. For each one, ask: given the skill you just described, would "
    "they have to say it AGAIN to set this skill up on a new occasion?\n"
    "   - YES → it is a PARAMETER: one of the pieces of information they THEMSELVES "
    "PROVIDED that your framing does not already carry. The skill works the same way "
    "whatever it is, so it cannot be known until they say it. If they never said it, "
    "it cannot be a parameter at all — a need they never mentioned (somewhere to keep "
    "the result, when they never said where; what to call an entry, when they never "
    "named one) is the skill's own business to settle. Give it a short name (a single "
    "lowercase word or snake_case) and one line saying what to supply for it.\n"
    "   - NO → the name and description you just wrote already carry it, so it is not "
    "a parameter and gets no line at all. Asking for it would be asking them to tell "
    "you what they came to you for.\n"
    "   That they said it once settles nothing: they said all of it once, while "
    "asking. The question is only what is left to say once the skill already exists.\n"
    "   For example, asked once to summarise a long report: summarising is what they "
    "came for, so a skill that summarises reports needs only the report next time. "
    "Had they asked instead for whatever they say to be done to that report, they "
    "named no task at all, and both pieces would be parameters. Their ask is what "
    "tells you which skill you were asked for.\n"
    "Respond with these tagged lines and nothing else:\n"
    f"{render_line(_FRAMING_NAME_LINE)}\n"
    f"{render_line(_FRAMING_DESCRIPTION_LINE)}\n"
    f"{render_line(_FRAMED_PARAMETER_LINE)}\n"
    "Write ONE line per parameter, and none for anything the framing already carries. "
    "At least one piece is always a parameter: a skill with nothing left to say to it "
    "can only ever repeat the one occasion it was asked for.\n"
    "Write nothing else — no preamble, no explanation, no restating the ask."
)

# The single per-call ask; the user's own messages are the content.
_SKILL_FRAMING_INSTRUCTION = (
    "Name this skill by the kind of task it does, describe it in one line, and list "
    "the parameters someone would have to supply to set it up on a new occasion."
)


# ── Third customer: conversation-state classification (#1706) ──────────────────
# The classifier contract is a THIRD declared shape riding the SAME poison-screen +
# reroll machinery (``_valid_draw``): given a small conversation slice and a closed
# list of candidate states (the machine's CURRENT out-edges, rendered by
# :mod:`penny.conversation_machine` — never the global state set), emit ONE tagged
# line naming the state the newest message puts the conversation in.  MEMBERSHIP is
# the runtime constraint a static shape can't carry, so it rides in as the
# ``accepts`` predicate: a drawn state outside the offered union is a contract
# violation exactly like an untagged draw — rerolled once on the unchanged context,
# then an honest failure the machine treats as no-transition (fail → stay; the
# caller's rule, encoded in ``conversation_machine.next_state``).
#
# Some states are SKILL-GATED (#1706 beats 2/5 — apply and request):
# their option lines direct the model to add a second ``SKILL:`` line naming
# which of the listed skills is meant.  Drawing a gated state WITHOUT a valid
# in-set skill line is the same contract violation — reroll, then INVALID — so
# such a decision always carries an actionable skill, never a dangling "use a
# skill" with nothing bound.  The line itself is declared OPTIONAL because it is
# only *conditionally* required: a stray one on an ungated draw binds nothing and
# must not cost the decision a reroll.
STATE_TAG = "STATE:"
SKILL_TAG = "SKILL:"

_STATE_LINE = LineSpec(
    tag=STATE_TAG,
    fields=(FieldSpec(name=DrawField.NAME, placeholder="<name>"),),
)
_SKILL_LINE = LineSpec(
    tag=SKILL_TAG,
    role=LineRole.OPTIONAL,
    fields=(
        FieldSpec(
            name=DrawField.SKILL,
            placeholder="<the skill's name, exactly as quoted in Known skills>",
        ),
    ),
)
STATE_CLASSIFIER_SHAPE = MicroContextShape(lines=(_STATE_LINE, _SKILL_LINE))

STATE_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a dispatch step for a conversation between a user and their "
    "assistant. The assistant has real tools (reading pages, saving values), "
    "and a separate context carries out whatever you decide — NEVER judge "
    "whether an action is possible; your only job is the state.\n"
    "\n"
    "You are given:\n"
    "- The assistant's last message\n"
    "- The task being worked on (when there is one)\n"
    "- Known skills — the assistant's existing routines ((none) when it has none)\n"
    "- The user's newest message\n"
    "- Current state — where the conversation stands right now\n"
    "- Transitions — the states you may move to, each with the condition that "
    "selects it; the last one is the default\n"
    "\n"
    "Do this:\n"
    "1. In your reasoning, note what the user's newest message is doing in the "
    "conversation, judging only from what the messages say.\n"
    "2. Pick the ONE transition whose condition the newest message meets. When "
    "none of the conditions is met, pick the default.\n"
    "3. Check whether the chosen transition directs you to add a "
    f"{SKILL_TAG} line.\n"
    "\n"
    "Respond with exactly one line:\n"
    f"{render_line(_STATE_LINE)}\n"
    "The name must be one of the listed transitions, copied EXACTLY. When the "
    f"chosen transition directs it, add exactly one more line — {render_line(_SKILL_LINE)}"
    " — and nothing more.\n"
    "IMPORTANT: write nothing else — no preamble, no explanation, no restating "
    "the messages."
)

# The classifier's user turn is the rendered situation ALONE — no
# ``Instruction:``/``Content:`` wrapper.  That frame is the extraction
# customer's (natural for "here's a page, pull X out"); the classifier's ask
# lives entirely in its system prompt, so wrapping the slice would just repeat
# the instruction and label a structured situation as bulk content.
_STATE_USER_TEMPLATE = "{content}"


class MicroExtractOutcome(StrEnum):
    """The enumerated outcome of a micro-context extraction — a closed set the
    caller renders one way each (never a silent empty).

    ``NOT_PRESENT`` is distinct from ``EXTRACTION_FAILED`` by design: not-present
    is a *successful read of an absent fact* (the page was read; the fact isn't
    there — rendered honestly, no infrastructure failure implied), while
    extraction-failed is the escape for a model that never produced a usable
    tagged line.
    """

    EXTRACTED = "extracted"
    NOT_PRESENT = "not_present"
    EXTRACTION_FAILED = "extraction_failed"
    POISON_REROLL_FAILED = "poison_reroll_failed"


class MicroContextResult(BaseModel):
    """The small typed result the main loop receives from a micro-context.

    ``value`` carries the extracted text on :attr:`MicroExtractOutcome.EXTRACTED`;
    ``reason`` carries the model's one-line what-is-missing on
    :attr:`MicroExtractOutcome.NOT_PRESENT`.  Both are empty on the failure
    outcomes — the caller renders those from the outcome alone.  The populated
    field is what flows to the main loop verbatim; the parent model never
    re-transcribes it.
    """

    outcome: MicroExtractOutcome
    value: str = ""
    reason: str = ""


class LeafPlaceholder(BaseModel):
    """One placeholder's label from the leaf labeller (#1824): its semantic ``name``
    (what the spot holds — the anchor the runtime join reads) and the one-line
    ``description`` of what belongs there each time the routine runs.

    Both are non-blank by construction: the declared line requires them, so a line
    missing either is malformed and simply absent — the caller then falls back to the
    arg-derived name, per placeholder, never all-or-nothing."""

    name: str
    description: str


class LeafLabels(BaseModel):
    """The leaf labeller's typed result (#1824): every offered placeholder's label,
    keyed by its CURRENT (arg-derived) name so the mapping home is unambiguous.

    There is no skill name, description or verdict here — naming the SKILL and deciding
    its parameters is the framer's decision, taken from the user's ask alone, and this
    draw never sees that ask.  A placeholder with no line keeps its arg-derived name,
    which is a legible degradation rather than a wrong answer."""

    placeholders: dict[str, LeafPlaceholder] = {}


class FramedParameter(BaseModel):
    """One parameter of a skill's public signature (#1824): the semantic ``name`` (the
    binding key at instantiation) and the one-line ``description`` of what to supply."""

    name: str
    description: str


class SkillFraming(BaseModel):
    """The skill framer's typed result (#1824): the skill's INTERFACE — a GENERIC
    verb-noun ``name``, a one-line ``description``, and the ``parameters`` someone must
    supply to set it up on a new occasion.

    ``name``/``description`` are non-blank by construction (REQUIRED lines of the
    declared shape, so a draw missing either never parses and the caller falls back to
    the deterministic slug — framing never blocks extraction).  ``parameters`` is
    non-empty by the same contract: a skill with nothing left to say to it can only
    repeat the occasion it was taught on, so an empty draw is a violation, rerolled
    once and then degraded honestly."""

    name: str
    description: str
    parameters: list[FramedParameter] = []


class StateDrawOutcome(StrEnum):
    """The enumerated outcome of a state-classification draw (#1706) — a closed
    set the machine maps one way each.

    ``INVALID`` covers both contract violations — a draw that misses the declared
    shape AND a drawn state outside the offered union (the persisted promptlog row
    holds which) — because the machine treats them identically: no transition
    (fail → stay).  ``POISON_REROLL_FAILED`` is the transport-artifact escape, same
    as extract."""

    DECIDED = "decided"
    INVALID = "invalid"
    POISON_REROLL_FAILED = "poison_reroll_failed"


class StateDraw(BaseModel):
    """The state-classification micro-context's typed result (#1706): the drawn
    state ``name`` — guaranteed a member of the offered union — on
    :attr:`StateDrawOutcome.DECIDED`, empty on the failure outcomes.  ``skill``
    carries the drawn ``SKILL:`` payload when the decided state was skill-gated —
    guaranteed a member of the offered skills — and is empty otherwise.
    String-typed on purpose: this module knows candidate names, never the
    machine's state enum (the machine imports this module, not the reverse)."""

    outcome: StateDrawOutcome
    name: str = ""
    skill: str = ""


class DrawFailure(StrEnum):
    """Why no usable draw came back once the budget was spent — the two ways a
    micro-context comes up empty, which each customer maps onto its own enumerated
    outcome rather than sharing one.

    ``POISON`` is a transport artifact every time (a degeneration collapse or a
    leaked Harmony envelope).  ``INVALID`` is a clean draw that never matched its
    DECLARED shape — an untagged output, a blank payload, or a runtime constraint
    (membership) the shape can't carry."""

    POISON = "poison"
    INVALID = "invalid"


def _accept_any(_drawn: ParsedDraw) -> bool:
    """The default runtime constraint: none.  For a customer with no membership
    sets to check, the declared shape IS the whole contract."""
    return True


_EXTRACT_FAILURES: dict[DrawFailure, MicroExtractOutcome] = {
    DrawFailure.POISON: MicroExtractOutcome.POISON_REROLL_FAILED,
    DrawFailure.INVALID: MicroExtractOutcome.EXTRACTION_FAILED,
}
_STATE_FAILURES: dict[DrawFailure, StateDrawOutcome] = {
    DrawFailure.POISON: StateDrawOutcome.POISON_REROLL_FAILED,
    DrawFailure.INVALID: StateDrawOutcome.INVALID,
}


class MicroContext:
    """Runs a single-shot extraction over bulk content via the shared model client."""

    def __init__(
        self,
        model_client: LlmClient,
        *,
        reroll_attempts: int = PennyConstants.DEGENERATE_REROLL_ATTEMPTS,
    ) -> None:
        self._model_client = model_client
        self._reroll_attempts = reroll_attempts

    async def extract(
        self, content: str, instruction: str, *, run_target: str | None = None
    ) -> MicroContextResult:
        """Extract ``instruction`` from ``content`` in one scoped model call.

        The draw is poison-screened (collapse / leaked envelope → discard and
        re-roll on the unchanged context) and read against :data:`EXTRACT_SHAPE` —
        ``EXTRACTED:`` → the value (everything after the tag, so a multi-line digest
        or an item-per-line list survives whole), ``NOT_PRESENT:`` → the enumerated
        not-present outcome carrying the reason's first line only.  A clean draw
        matching neither is a contract violation, never a value: one reroll of the
        unchanged context, then the extraction fails honestly.  A blank draw takes
        the same path (a blank payload is not a payload).
        """
        drawn = await self._valid_draw(content, instruction, run_target, shape=EXTRACT_SHAPE)
        if isinstance(drawn, DrawFailure):
            return MicroContextResult(outcome=_EXTRACT_FAILURES[drawn])
        return _extraction_result(drawn)

    async def label_leaves(
        self, content: str, leaves: Sequence[str], *, run_target: str | None = None
    ) -> LeafLabels | None:
        """Name every placeholder in a distilled routine's tool calls (#1824) — the
        second customer of this machinery.  Rides the SAME poison-screen + reroll draw
        loop as ``extract``, with the leaf-labelling system prompt, its own ledger
        attribution and its own declared shape (:data:`LEAF_LABELLING_SHAPE`), plus the
        runtime constraint a static shape cannot carry: the draw must name at least one
        of the OFFERED placeholders.  A shape made only of PER_ITEM lines parses an
        empty draw happily, so without that floor a page of prose would come back as a
        successful label carrying nothing.

        ``None`` on any failure — the caller keeps every placeholder's arg-derived
        name, so extraction never blocks on the naming.  Per-placeholder lines are
        best-effort by declaration (``LineRole.PER_ITEM``): one absent or malformed
        line costs that placeholder its name and nothing else."""
        drawn = await self._valid_draw(
            content,
            _LEAF_LABELLING_INSTRUCTION,
            run_target,
            shape=LEAF_LABELLING_SHAPE,
            accepts=lambda parsed: _names_an_offered_leaf(parsed, leaves),
            system_prompt=LEAF_LABELLING_SYSTEM_PROMPT,
            agent_name=PennyConstants.LEAF_LABELLING_AGENT_NAME,
            prompt_type=PennyConstants.LEAF_LABELLING_PROMPT_TYPE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return LeafLabels(placeholders=_leaf_placeholders(drawn.items, leaves))

    async def frame_skill(
        self, content: str, *, run_target: str | None = None
    ) -> SkillFraming | None:
        """Write a skill's public signature from the user's ask (#1824) — its GENERIC
        name, its one-line description, and the parameters someone must supply to set
        it up again — the FOURTH customer of this machinery.  Rides the SAME
        ``_valid_draw`` step as the other three against its own declared shape
        (:data:`SKILL_FRAMING_SHAPE`), plus the runtime constraint a static shape
        cannot carry: a draw that named NO parameter frames a skill that can only
        repeat the occasion it was taught on.

        Both refusals are one reroll of the unchanged context and then ``None``, which
        the caller degrades to the deterministic slug of the triggering message with no
        parameters at all."""
        drawn = await self._valid_draw(
            content,
            _SKILL_FRAMING_INSTRUCTION,
            run_target,
            shape=SKILL_FRAMING_SHAPE,
            accepts=_frames_a_parameter,
            system_prompt=SKILL_FRAMING_SYSTEM_PROMPT,
            agent_name=PennyConstants.SKILL_FRAMING_AGENT_NAME,
            prompt_type=PennyConstants.SKILL_FRAMING_PROMPT_TYPE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return _skill_framing(drawn)

    async def classify_state(
        self,
        content: str,
        allowed: Sequence[str],
        *,
        skill_gated_states: Sequence[str] = (),
        skills: Sequence[str] = (),
        run_target: str | None = None,
    ) -> StateDraw:
        """Pick one state from ``allowed`` for a rendered conversation slice
        (#1706) — the third customer of this machinery.  Rides the SAME
        poison-screen + reroll draw loop as ``extract``, with the dispatch system
        prompt, its own ledger attribution and its own declared shape
        (:data:`STATE_CLASSIFIER_SHAPE`), plus the MEMBERSHIP constraint a static
        shape can't carry: a drawn state outside ``allowed`` is a contract violation
        exactly like an untagged draw — one reroll of the unchanged context, then an
        honest ``INVALID`` the machine reads as no-transition.

        ``skill_gated_states`` names the states whose draw must ALSO carry a
        ``SKILL:`` line naming a member of ``skills`` — drawing one with a
        missing or out-of-set skill is the same contract violation, so a gated
        decision always binds an actionable skill.  A stray ``SKILL:`` line on an
        ungated draw is ignored (the decision stands; the line binds nothing)."""
        drawn = await self._valid_draw(
            content,
            "",
            run_target,
            shape=STATE_CLASSIFIER_SHAPE,
            accepts=partial(
                _state_is_bound,
                allowed=allowed,
                skill_gated_states=skill_gated_states,
                skills=skills,
            ),
            system_prompt=STATE_CLASSIFIER_SYSTEM_PROMPT,
            agent_name=PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
            prompt_type=PennyConstants.STATE_CLASSIFIER_PROMPT_TYPE,
            user_template=_STATE_USER_TEMPLATE,
        )
        if isinstance(drawn, DrawFailure):
            return StateDraw(outcome=_STATE_FAILURES[drawn])
        return _state_draw(drawn, skill_gated_states)

    async def _valid_draw(
        self,
        content: str,
        instruction: str,
        run_target: str | None,
        *,
        shape: MicroContextShape,
        accepts: Callable[[ParsedDraw], bool] = _accept_any,
        system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT,
        agent_name: str = PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        prompt_type: str = PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
        user_template: str = _USER_TEMPLATE,
    ) -> ParsedDraw | DrawFailure:
        """The ONE validate step every customer rides (#1814): a poison-screened
        draw, read against its DECLARED ``shape`` and checked against ``accepts``
        (the runtime constraint — membership sets a static shape can't carry),
        re-rolled on the unchanged context while the contract is violated, then an
        honest typed failure the customer maps onto its own outcome.

        Only the shape (and the prompt + attribution) differs per customer, so no
        customer owns a reroll loop or partitions a string of its own."""
        for _ in range(_INVALID_DRAW_BUDGET):
            draw = await self._draw_clean(
                content,
                instruction,
                run_target,
                system_prompt=system_prompt,
                agent_name=agent_name,
                prompt_type=prompt_type,
                user_template=user_template,
            )
            if draw is None:
                return DrawFailure.POISON
            parsed = parse_draw(draw, shape)
            if parsed is not None and accepts(parsed):
                return parsed
            logger.warning("%s output violated its declared shape — one reroll", agent_name)
        logger.warning("%s output violated its declared shape after reroll", agent_name)
        return DrawFailure.INVALID

    async def _draw_clean(
        self,
        content: str,
        instruction: str,
        run_target: str | None,
        *,
        system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT,
        agent_name: str = PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        prompt_type: str = PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
        user_template: str = _USER_TEMPLATE,
    ) -> str | None:
        """The raw extraction text, re-rolling on poison; ``None`` if every draw
        is unusable.  Mirrors the agent-loop reroll guard — discard poison, never
        append it, re-draw on the same context, abort after the attempt budget.

        The ``system_prompt`` + ledger attribution are parameters (defaulting to the
        browse-extract contract) so a second output contract — run-end skill naming
        (#1665) — rides the SAME poison/reroll loop without duplicating it."""
        messages = self._messages(content, instruction, system_prompt, user_template)
        run_id = uuid.uuid4().hex
        for attempt in range(self._reroll_attempts):
            response = await self._model_client.chat(
                messages=messages,
                agent_name=agent_name,
                prompt_type=prompt_type,
                run_id=run_id,
                run_target=run_target,
            )
            text = response.content or ""
            if not self._is_poison(text):
                return text
            logger.warning(
                "Micro-context output unusable — discarding and re-rolling %d/%d",
                attempt + 1,
                self._reroll_attempts,
            )
        logger.error(
            "Micro-context output still unusable after %d re-rolls — extraction aborted",
            self._reroll_attempts,
        )
        return None

    @staticmethod
    def _is_poison(text: str) -> bool:
        """A degeneration collapse or a leaked Harmony envelope — the same
        transport artifacts the agent-loop reroll guard discards."""
        return has_leaked_harmony_envelope(text) or is_degenerate_run(text)

    @staticmethod
    def _messages(
        content: str,
        instruction: str,
        system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT,
        user_template: str = _USER_TEMPLATE,
    ) -> list[dict]:
        """The scoped two-message context: the contract framing (``system_prompt``,
        default the browse-extract contract), then the user turn shaped by the
        customer's ``user_template`` (default: the instruction paired with bulk
        content; the classifier passes the bare-situation template)."""
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_template.format(instruction=instruction, content=content),
            },
        ]


def _extraction_result(drawn: ParsedDraw) -> MicroContextResult:
    """The typed result for a valid extraction draw, read off the parsed draw by
    field name — the ``EXTRACTED:`` value (everything after the tag) or the
    ``NOT_PRESENT:`` reason (its first line only, so a not-present apology can never
    be multi-line-promoted into a value)."""
    value = drawn.field(EXTRACTED_TAG, DrawField.VALUE)
    if value is not None:
        return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=value)
    reason = drawn.field(NOT_PRESENT_TAG, DrawField.REASON)
    if reason is None:
        # Unreachable while the two tags are declared ALTERNATIVE (a parsed draw
        # carries one of them) — honest rather than silent if that ever changes.
        return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTION_FAILED)
    return MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason=reason)


def _leaf_placeholders(
    items: Sequence[ParsedLine], leaves: Sequence[str]
) -> dict[str, LeafPlaceholder]:
    """Every labelled placeholder as a ``{current_name: LeafPlaceholder}`` map (#1824).

    The grammar already carved and shape-checked each line; what is left is the two
    rules it can't express.  MEMBERSHIP: a line naming something never offered
    addresses nothing, so it is dropped rather than invented into a placeholder.  And a
    placeholder named on MORE THAN ONE line is a contradictory draw (the contract asks
    for exactly one line each), so it is dropped too — the arg-derived name it falls
    back to is legible, where an arbitrary pick between two names is not."""
    offered = set(leaves)
    labels: dict[str, LeafPlaceholder] = {}
    repeated: set[str] = set()
    for item in items:
        current = item.fields[DrawField.CURRENT]
        if current not in offered:
            continue
        if current in labels:
            repeated.add(current)
        labels[current] = LeafPlaceholder(
            name=item.fields[DrawField.SEMANTIC], description=item.fields[DrawField.DESCRIPTION]
        )
    return {name: label for name, label in labels.items() if name not in repeated}


def _names_an_offered_leaf(drawn: ParsedDraw, leaves: Sequence[str]) -> bool:
    """The leaf labeller's floor (#1824): the draw named at least one of the OFFERED
    placeholders.  Its shape is PER_ITEM lines only, so an empty draw parses cleanly —
    without this, prose that named nothing would come back as a successful label.  With
    nothing offered there is nothing to violate."""
    return not leaves or bool(_leaf_placeholders(drawn.items, leaves))


def _frames_a_parameter(drawn: ParsedDraw) -> bool:
    """The framer's floor, as its prompt states it (#1824): a skill has at least one
    parameter.  One that needs nothing said to it can only repeat the occasion it was
    taught on, which makes it a record of what happened rather than a skill — so an
    empty draw is a contract violation like any other: one reroll, then honest
    degradation."""
    return bool(_framed_parameters(drawn.items))


def _framed_parameters(items: Sequence[ParsedLine]) -> list[FramedParameter]:
    """The drawn parameters in draw order, first line per NAME winning — a name is a
    binding key, so a repeat is the same parameter described twice, never two."""
    parameters: list[FramedParameter] = []
    seen: set[str] = set()
    for item in items:
        name = item.fields[DrawField.NAME]
        if name in seen:
            continue
        seen.add(name)
        parameters.append(
            FramedParameter(name=name, description=item.fields[DrawField.DESCRIPTION])
        )
    return parameters


def _skill_framing(drawn: ParsedDraw) -> SkillFraming | None:
    """The framing draw read by FIELD NAME — what the skill is called, what it is for,
    and what has to be supplied to set it up.  ``None`` only if a required line somehow
    went missing: both are REQUIRED in the declared shape, so a draw missing either
    never parses, and the caller degrades to the deterministic slug rather than shipping
    a skill slugged from an empty string."""
    name = drawn.field(NAME_TAG, DrawField.NAME)
    description = drawn.field(DESCRIPTION_TAG, DrawField.DESCRIPTION)
    if name is None or description is None:
        return None
    return SkillFraming(
        name=name, description=description, parameters=_framed_parameters(drawn.items)
    )


def _state_is_bound(
    drawn: ParsedDraw,
    *,
    allowed: Sequence[str],
    skill_gated_states: Sequence[str],
    skills: Sequence[str],
) -> bool:
    """MEMBERSHIP — the runtime constraint the static shape can't carry, applied as
    part of the one validate step so a miss rerolls exactly like a shape violation.

    The drawn state must be one of the offered transitions, and a SKILL-GATED state
    must ALSO carry a ``SKILL:`` line naming one of the offered skills.  Exact match,
    no normalization beyond the shape's own: the prompt says copied exactly, and
    every member was shown verbatim."""
    name = drawn.field(STATE_TAG, DrawField.NAME)
    if name is None or name not in allowed:
        return False
    if name not in skill_gated_states:
        return True
    skill = drawn.field(SKILL_TAG, DrawField.SKILL)
    return skill is not None and skill in skills


def _state_draw(drawn: ParsedDraw, skill_gated_states: Sequence[str]) -> StateDraw:
    """The typed decision for an accepted classifier draw — the state, plus the skill
    a GATED state bound.  A stray ``SKILL:`` line on an ungated draw binds nothing."""
    name = drawn.field(STATE_TAG, DrawField.NAME)
    if name is None:
        return StateDraw(outcome=StateDrawOutcome.INVALID)
    if name not in skill_gated_states:
        return StateDraw(outcome=StateDrawOutcome.DECIDED, name=name)
    skill = drawn.field(SKILL_TAG, DrawField.SKILL)
    if skill is None:
        return StateDraw(outcome=StateDrawOutcome.INVALID)
    return StateDraw(outcome=StateDrawOutcome.DECIDED, name=name, skill=skill)
