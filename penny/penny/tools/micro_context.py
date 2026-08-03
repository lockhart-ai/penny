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
the naming fallback).  A violation of an OPTIONAL or PER_ITEM line is simply an
ABSENT line at the GRAMMAR level — but whether absence is acceptable is the
CUSTOMER's to declare, as a runtime constraint (``accepts``): the leaf labeller
requires a well-formed line for every spot it offered, so a decayed tag is a
whole-draw failure rather than one spot quietly unnamed (#1828), while a customer
with no checkable coverage keeps the best-effort read.  An accepted draw never
contains an invalid line.

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

# ── Second customer: run-end LEAF LABELLING (#1824/#1827/#1828) ────────────────
# The labelling contract is a DIFFERENT declared shape riding the SAME poison-screen
# + reroll machinery (``_valid_draw``): given a demonstrated routine and every SPOT in
# it that gets filled each run, write a semantic NAME + a one-line description for
# EVERY spot — and nothing else.  One line per spot, keyed by the spot's CURRENT
# (arg-derived) name, so the system owns an unambiguous mapping back; the model writes
# LABELS only.
#
# It judges NOTHING (#1824).  The pipeline it replaces asked this same draw where each
# value came from and then, in a second draw, what the routine was about — and the
# per-candidate verdict pinned at ~0.7-0.8 across three independent wording
# interventions (#1821), with the error compounding geometrically as a routine gains
# leaves.  The inversion: every leaf is a placeholder unconditionally, so there is no
# origin question left to get wrong, and the routine's INTERFACE — its name,
# description and parameters — is decided from the user's ask alone by a separate
# customer (the framer).  Neither draw sees the other's evidence, which is what stops
# them contradicting each other.
#
# So the ``PARAM`` line, the ``NAME:``/``DESCRIPTION:`` lines, the verdict union and
# the grouped-by-verdict response structure (#1807) are all gone from this customer.
#
# And COVERAGE IS CHECKED, not tolerated (#1828, the code owner's ruling): an accepted
# draw may never contain an invalid line.  The PER_ITEM "a malformed line is simply
# dropped" tolerance is a #1770-era decision, made for the VERDICT labeller where
# absence had a safe meaning (keep the arg-derived required parameter).  For the leaf
# labeller it is wrong, because the caller knows the exact offered-leaf set, so the
# question "did every spot get a well-formed line?" is ANSWERABLE — and the observed
# failure was the tag itself decaying mid-draw (``PLACEHOlDER``, ``PLACEHOLE``,
# ``PLACEHALER``), which the parse rightly refused and the validator then accepted
# around, silently costing that spot its label.  A draw that misses any offered spot is
# now a contract violation exactly like the classifier drawing an out-of-set state: one
# reroll on the unchanged context, then an honest WHOLE-draw failure.  Correctness of
# accepted results over salvage — a draw that decays twice fails whole, and every spot
# keeps its arg-derived name.
PLACEHOLDER_TAG = "PLACEHOLDER"

# The one line this customer emits, once per offered spot.  The semantic name declares
# ``FieldShape.NAME`` because it is the spot's identity downstream (the binding key the
# runtime join will bind against), so a "name" that is really a name plus its own
# description is a malformed line rather than a value.  Its description is OPTIONAL in
# the grammar — a line that stops after the name still names the spot — which is what
# lets the labelling eval score "did it say what belongs there" as its own miss instead
# of losing the whole line to the parse.  The CONSUMER is where a blank one is caught:
# the description is what the leaf renders as, so ``_apply_leaf_labels`` reads a blank
# one as no label rather than rendering an empty slot.  (That is the one per-spot path
# left, and it is the ticket's own grammar: the line IS well-formed, so coverage holds
# and the draw stands — what is missing is what belongs there, not the line.)
_PLACEHOLDER_LINE = LineSpec(
    tag=PLACEHOLDER_TAG,
    role=LineRole.PER_ITEM,
    fields=(
        FieldSpec(name=DrawField.CURRENT, placeholder="<current name>", separator=Separator.COLON),
        FieldSpec(
            name=DrawField.SEMANTIC,
            placeholder="<semantic_name>",
            shape=FieldShape.NAME,
            separator=Separator.DASH,
        ),
        FieldSpec(
            name=DrawField.DESCRIPTION,
            placeholder="<one-line description of what belongs there each run>",
            required=False,
        ),
    ),
)
SKILL_NAMING_SHAPE = MicroContextShape(lines=(_PLACEHOLDER_LINE,))

SKILL_NAMING_SYSTEM_PROMPT = (
    "You are a naming step. A routine has just been demonstrated once, and every "
    "spot in it that gets filled in again each time it runs has been pulled out for "
    "you to name. Naming those spots is your whole job: they are all placeholders "
    "already, so nothing here asks where a value came from or what the routine as a "
    "whole should be called.\n"
    "You are given:\n"
    "- The conversation that led to the routine — the last user turn is the one that "
    "demonstrated it\n"
    "- The routine's numbered steps, each spot shown as {its current name}\n"
    "- The placeholders — every spot, with the argument site(s) it fills and the "
    "value it was demonstrated with\n"
    "Do this for EVERY placeholder you are given:\n"
    "1. Work out what that spot IS — the conversation says what the routine is for, "
    "its step says what the value is used to do, and the demonstrated value says what "
    "kind of thing goes there.\n"
    "2. Name it for what it is in this routine (e.g. listing_page, entry_key), NOT "
    "for the tool argument it happens to fill and NOT for the one value it was "
    "demonstrated with — a new value goes there every run.\n"
    "3. Describe in one line what belongs in that spot each time the routine runs.\n"
    "A spot that holds an INSTRUCTION rather than a value — what to look for wherever "
    "the routine reads, e.g. the spot filling browse.extract with the current price — "
    "is PLAIN LANGUAGE: there is no CSS-selector, XPath, or pattern machinery in this "
    "system, so NEVER name or describe one that way.\n"
    "Respond with one line per placeholder and nothing else:\n"
    f"{render_line(_PLACEHOLDER_LINE)}\n"
    "Write ONE line for EVERY placeholder you were given, and none for anything else, "
    "repeating its CURRENT name exactly so it maps back. Two spots are never the same "
    "spot: give each its own name. Use a single lowercase word or snake_case for "
    "<semantic_name>.\n"
    "IMPORTANT: write nothing else — no preamble, no explanation, no restating the "
    "routine."
)

# ── Fourth customer: the routine's SHAPE — what it is ABOUT (#1803) ────────────
# The labeller above answers where each value CAME FROM.  It cannot also answer
# what the routine IS, because those are decided from different evidence and the
# second one has no answer inside a single demonstrated round: "keep an eye on the
# aurora deck 2 price" hands over two values, both from the user, and nothing in
# that round says which of them will vary the next time.  Deciding it anyway is
# what produced skills that named themselves for a value and then demanded it —
# `record-product-price` requiring a `what_to_extract` its own name already gave,
# so the routine could not fire from the natural second ask.
#
# So the question moves to its own draw, over a deliberately SMALLER space: what
# the user asked for, and the values the labeller kept.  Naming the routine and
# deciding what it is about are ONE decision here, which is what stops the two
# halves from contradicting — a skill cannot call itself a price watcher and then
# ask what to watch, because the same draw wrote both.
#
# The model does NOT invent the structure (the rejected template attempt): the
# closed set of values is handed to it and it picks a case for each, exactly as
# every other customer of this machinery does.
#
# Since #1828 it no longer fires from the run-end pipeline: it decided over the values
# a per-candidate VERDICT kept, and the labeller draws no verdicts.  Contract, prompt
# and parse are untouched — its own eval drives it, and its refusal rules stay pinned
# deterministically — so the question is intact for whichever beat re-homes it.
NAME_TAG = "NAME:"
DESCRIPTION_TAG = "DESCRIPTION:"
CONSTANT_TAG = "CONSTANT"
PARAMETER_TAG = "PARAMETER"

_SHAPE_NAME_LINE = LineSpec(
    tag=NAME_TAG,
    fields=(FieldSpec(name=DrawField.NAME, placeholder="<a short generic verb-noun name>"),),
)
_SHAPE_DESCRIPTION_LINE = LineSpec(
    tag=DESCRIPTION_TAG,
    fields=(
        FieldSpec(name=DrawField.DESCRIPTION, placeholder="<one line: what the routine is for>"),
    ),
)
# The per-value role lines are PER_ITEM for the same reason the labeller's verdict
# lines are (#1770/#1814): a value with no line simply has no role, which the caller
# reads as PARAMETER — the direction that keeps the skill bindable.  The value name is
# ``FieldShape.NAME`` because it is matched against the offered set, so a name carrying
# a trailing gloss is a malformed line, not a near-miss to be salvaged — which is why
# the contract block below renders from these specs and no longer glosses each tag
# inline: a draw echoed the gloss back, and under a declared shape that is malformed
# rather than tolerated.
_CONSTANT_LINE = LineSpec(
    tag=CONSTANT_TAG,
    role=LineRole.PER_ITEM,
    fields=(FieldSpec(name=DrawField.VALUE, placeholder="<value name>", shape=FieldShape.NAME),),
)
_PARAMETER_LINE = LineSpec(
    tag=PARAMETER_TAG,
    role=LineRole.PER_ITEM,
    fields=(FieldSpec(name=DrawField.VALUE, placeholder="<value name>", shape=FieldShape.NAME),),
)
SKILL_SHAPE_SHAPE = MicroContextShape(
    lines=(_SHAPE_NAME_LINE, _SHAPE_DESCRIPTION_LINE, _CONSTANT_LINE, _PARAMETER_LINE)
)

SKILL_SHAPE_SYSTEM_PROMPT = (
    "You are deciding what a reusable routine IS. You are given what the user "
    "asked for once, a one-line summary of what was done for them, and the values "
    "used to carry it out that time. Do three things:\n"
    "1. From what the user asked for, extract the CORE USER INTENT — what they "
    "were trying to get done when they asked. Their own words are the evidence, "
    "and the one-line summary of what the round did is a second reading of the "
    "same thing: the values are HOW it was carried out, not what it was FOR.\n"
    "2. Name and describe the ROUTINE by that intent: a short verb-noun name for "
    "the KIND of task, generic — never the specific instance — and one line that "
    "states the intent it serves before any mechanics. A description that falls "
    "back on a specified piece of information where the intent named something "
    "particular has dropped the intent — say what the intent actually was.\n"
    "3. Now picture the user coming back later to set this routine running "
    "again, on a new occasion. What is the MINIMAL information they would have "
    "to give you? Decide every value on that one question:\n"
    "   - PARAMETER. They would have to say it again. The routine works the "
    "same way whatever it is, so it cannot be known until they say — it is "
    "asked for every time the routine is set up.\n"
    "   - CONSTANT. They would NOT have to say it again, because the routine "
    "already IS that. Saying it would be repeating what the name has already "
    "promised. It stays fixed, and asking for it would be asking the user to "
    "tell you what they came to you for.\n"
    "   That the user supplied a value the first time settles nothing here: "
    "they supplied all of them while showing you what to do. The question is "
    "only what they would still have to supply once you already know how.\n"
    "   THE USER'S OWN WORDS DECIDE THIS, not the name you happened to pick. A "
    "thing they NAMED as the point of the task is something they would expect "
    "you to know by now, so it is a CONSTANT — and if the name you wrote in "
    "step 2 leaves it open, the name is what is wrong, not this answer.\n"
    "   For example, after being asked to file the receipts from a particular "
    "sender into a tax folder: they named receipts as the point, so a routine "
    "that files receipts needs only the sender and the folder next time. Had "
    "they asked instead to file whatever they point you at, they named nothing, "
    "and every value would be a PARAMETER. Both are real routines — their ask "
    "is what tells you which one you were taught.\n"
    "   At least one value is always a PARAMETER: a routine that needs nothing "
    "said to it can only ever repeat the one occasion it was shown, which makes "
    "it a record of what happened rather than a routine. And a routine with NO "
    "constant is one that has to be told everything it was already told — if "
    "their ask named what to do, say so.\n"
    "Respond with these tagged lines and nothing else:\n"
    f"{render_line(_SHAPE_NAME_LINE)}\n"
    f"{render_line(_SHAPE_DESCRIPTION_LINE)}\n"
    f"{render_line(_CONSTANT_LINE)}   (the routine is about it)\n"
    f"{render_line(_PARAMETER_LINE)}   (the routine is pointed at it)\n"
    "Write ONE line for EVERY value, repeating its name exactly so it maps "
    "back — the name ALONE, with nothing after it.\n"
    "Write nothing else — no preamble, no explanation, no restating the routine."
)

# The single per-call ask; what the user wanted + the values are the content.
_SKILL_SHAPE_INSTRUCTION = (
    "Name this routine by the kind of task it does, describe it in one line, and say "
    "for every value whether the routine is about it or merely pointed at it."
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

# A user turn that is the rendered document ALONE — no ``Instruction:``/``Content:``
# wrapper.  That frame is the extraction customer's (natural for "here's a page, pull X
# out"); a customer whose ask lives entirely in its system prompt would only repeat the
# instruction by wrapping, and would label a structured situation as bulk content.  The
# classifier established it; the leaf labeller reads the same way (#1828), so it is
# named for the SHAPE of the turn rather than for either customer.
_BARE_CONTENT_TEMPLATE = "{content}"


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


class LeafLabel(BaseModel):
    """One spot's label from the leaf labeller (#1828): the ``name`` it drew for what
    that spot IS, and the one-line ``description`` of what belongs there each run.

    ``name`` is carried VERBATIM as the model wrote it — hardening it into a binding
    key is :func:`penny.skill_extraction.slug_parameter_name`'s, applied where the name
    becomes a key rather than here, so what the draw committed to stays readable in the
    ledger and in an eval report.  ``description`` is empty only when the line stopped
    after the name (the grammar's one optional field)."""

    name: str
    description: str = ""


class SkillLabels(BaseModel):
    """The leaf labeller's typed result (#1828): every offered spot's label, keyed by
    the spot's CURRENT (arg-derived) name — the anchor the input document renders
    verbatim, so the map home needs no guess.

    It carries NO routine name, description or parameters: the routine's interface is
    decided from the user's ask alone, by the framer, and this draw never sees that
    question (#1824).  ``labels`` COVERS every offered spot exactly once — a draw that
    missed one, or named one twice, never reaches here (it is rerolled, then fails
    whole), so a caller reading this map never has to wonder which spot went unnamed."""

    labels: dict[str, LeafLabel] = {}


class SkillShape(BaseModel):
    """The shape micro-context's typed result (#1803): what the routine IS — a
    GENERIC verb-noun ``name`` + one-line ``description``, and ``fixed``, the values
    the routine is ABOUT rather than pointed at.

    ``name``/``description`` are non-blank by construction and SUPERSEDE the
    labeller's, because they were written together with ``fixed`` — that is the
    whole point of the second draw, and taking the name from one draw and the
    constants from another would re-open the contradiction (#1803).  ``fixed`` is
    validated for MEMBERSHIP against the offered values and can never cover all of
    them: a routine with nothing left to bind can only repeat its demonstration."""

    name: str
    description: str
    fixed: frozenset[str] = frozenset()


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

    async def label_skill(
        self, content: str, offered: Sequence[str], *, run_target: str | None = None
    ) -> SkillLabels | None:
        """Name EVERY spot in a demonstrated routine (#1828) — the second customer of
        this machinery.  Rides the SAME poison-screen + reroll draw loop as ``extract``,
        with the labelling system prompt, its own ledger attribution, its own declared
        shape (:data:`SKILL_NAMING_SHAPE`) and the bare-content user turn (the rendered
        document IS the whole ask, so there is no instruction to wrap it in), plus the
        runtime constraint a static shape can't carry: the draw must COVER ``offered``,
        the current names of the spots the content listed.

        Returns the labels — one per offered spot, guaranteed — or ``None`` when the
        draw failed (poison exhausted, or coverage still incomplete after the reroll).
        The caller then keeps every spot's arg-derived name, so run-end extraction
        NEVER blocks on the rewrite."""
        drawn = await self._valid_draw(
            content,
            "",
            run_target,
            shape=SKILL_NAMING_SHAPE,
            accepts=partial(_labels_every_spot, offered=offered),
            system_prompt=SKILL_NAMING_SYSTEM_PROMPT,
            agent_name=PennyConstants.SKILL_NAMING_AGENT_NAME,
            prompt_type=PennyConstants.SKILL_NAMING_PROMPT_TYPE,
            user_template=_BARE_CONTENT_TEMPLATE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return _leaf_labels(drawn)

    async def shape_skill(
        self, content: str, values: Sequence[str], *, run_target: str | None = None
    ) -> SkillShape | None:
        """Decide what a distilled routine IS (#1803) — its GENERIC name +
        description and which of ``values`` it is ABOUT — the FOURTH customer of this
        machinery.  Rides the SAME ``_valid_draw`` step as the other three against its
        own declared shape (:data:`SKILL_SHAPE_SHAPE`), plus the runtime constraint a
        static shape cannot carry: the drawn constants must not cover EVERY offered
        value, which would leave a routine nothing to bind and so able only to repeat
        its own demonstration.

        ``None`` on any failure degrades to the labeller's name with no constants at
        all — every value stays a bindable parameter, exactly the pre-#1803 behaviour,
        so the shape draw can only ever ADD the distinction, never cost a skill its
        parameters."""
        drawn = await self._valid_draw(
            content,
            _SKILL_SHAPE_INSTRUCTION,
            run_target,
            shape=SKILL_SHAPE_SHAPE,
            accepts=lambda parsed: _leaves_something_bindable(parsed, values),
            system_prompt=SKILL_SHAPE_SYSTEM_PROMPT,
            agent_name=PennyConstants.SKILL_SHAPE_AGENT_NAME,
            prompt_type=PennyConstants.SKILL_SHAPE_PROMPT_TYPE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return _skill_shape(drawn, values)

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
            user_template=_BARE_CONTENT_TEMPLATE,
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


def _labels_every_spot(drawn: ParsedDraw, offered: Sequence[str]) -> bool:
    """COVERAGE — the runtime constraint the labelling shape can't carry (#1828), and
    the whole of what makes an accepted draw complete: ONE well-formed line per offered
    spot, and no line for anything else.

    Each way of missing that is the same violation, answered the same way — one reroll
    on the unchanged context, then an honest whole-draw failure.  MISSING is the
    observed failure (the tag decays mid-draw, the parse rightly refuses the line, and
    the spot ends up named by nothing).  TWICE is a contradictory draw, and taking
    either line would let a stray trailing one rename a spot.  A line for a spot that
    was never offered is a spot INVENTED rather than named — the shared-spot case's
    exact failure, where a value filling two argument sites is one spot and splitting it
    keys a second line to a name nobody listed.

    The caller knows the offered set, so all three are answerable rather than
    best-effort gaps to absorb; the prompt asks for exactly this ("one line for every
    placeholder you were given, and none for anything else"), so the validator and the
    contract say one thing."""
    named = [item.fields[DrawField.CURRENT] for item in drawn.items]
    return sorted(named) == sorted(set(offered))


def _leaf_labels(drawn: ParsedDraw) -> SkillLabels:
    """Every drawn line as a ``{current_name: LeafLabel}`` map (#1828).

    No filtering and no de-duplication: :func:`_labels_every_spot` accepted this draw
    only because it carries exactly one well-formed line per offered spot and nothing
    else, so the map is complete and unambiguous by construction."""
    return SkillLabels(
        labels={
            item.fields[DrawField.CURRENT]: LeafLabel(
                name=item.fields[DrawField.SEMANTIC],
                description=item.fields.get(DrawField.DESCRIPTION, ""),
            )
            for item in drawn.items
        }
    )


def _drawn_constants(drawn: ParsedDraw, values: Sequence[str]) -> frozenset[str]:
    """The offered values the draw marked CONSTANT (#1803), MEMBERSHIP-filtered.

    A ``PARAMETER`` line for the same value CONTRADICTS the constant and wins — the
    bindable direction is the safe one, the same rule the labeller's repeated-line
    drop encodes.  A line naming something never offered addresses nothing, so it is
    dropped rather than invented into a constant."""
    offered = set(values)
    roles = {
        tag: {
            value
            for item in drawn.items
            if item.tag == tag and (value := item.fields[DrawField.VALUE]) in offered
        }
        for tag in (CONSTANT_TAG, PARAMETER_TAG)
    }
    return frozenset(roles[CONSTANT_TAG] - roles[PARAMETER_TAG])


def _leaves_something_bindable(drawn: ParsedDraw, values: Sequence[str]) -> bool:
    """The floor the shape prompt states, enforced (#1803): a draw may not mark EVERY
    offered value constant.  A routine that needs nothing said to it can only repeat
    the one occasion it was shown, so this is a contract violation like any other —
    one reroll of the unchanged context, then honest degradation."""
    return not values or _drawn_constants(drawn, values) != set(values)


def _skill_shape(drawn: ParsedDraw, values: Sequence[str]) -> SkillShape | None:
    """The shape draw read by FIELD NAME — what the routine is called, what it is for,
    and which values it is ABOUT.  ``None`` only if a required line somehow went missing
    — the same belt-and-braces guard :func:`_skill_label` keeps, since both lines are
    REQUIRED in the declared shape and a draw missing either never parses.

    A blank name reaches nothing: the caller degrades to the labeller's name, which is
    the documented floor, rather than shipping a routine slugged from an empty string
    with an empty description."""
    name = drawn.field(NAME_TAG, DrawField.NAME)
    description = drawn.field(DESCRIPTION_TAG, DrawField.DESCRIPTION)
    if name is None or description is None:
        return None
    return SkillShape(name=name, description=description, fixed=_drawn_constants(drawn, values))


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
