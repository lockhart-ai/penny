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
whole-draw failure rather than one spot quietly unnamed (#1828), and the framer —
which offers nothing and MINTS its lines — refuses a draw whose dropped-line count
is non-zero (#1830), while a customer with no checkable coverage keeps the
best-effort read.  An accepted draw never contains an invalid line.

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
import re
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
# failure was the tag itself decaying mid-draw, which the parse rightly refused and the
# validator then accepted around, silently costing that spot its label.  A draw that
# misses any offered spot is now a contract violation exactly like the classifier
# drawing an out-of-set state: one reroll on the unchanged context, then an honest
# WHOLE-draw failure.  Correctness of accepted results over salvage — a draw that decays
# twice fails whole, and every spot keeps its arg-derived name.
#
# THE TAG IS A SHORT, COMMON, NON-COMPOUND WORD (#1842, the #1826 decay class).  It was
# ``PLACEHOLDER`` — eleven characters, a compound the model could not write reliably: in
# one measured run it labelled every spot correctly in both draws and still failed the
# whole labelling, because the tag came out ``PLACEBLODER``, ``PLACEHOLER`` (three times
# in one draw), ``PLACEHOlder``, and once with a zero-width character inside it.  The
# parse matches tags EXACTLY (no fuzzy matching, by standing ruling), so each decayed
# line read as no line, coverage correctly rejected the draw, and four perfect judgments
# were discarded over spelling.  The fix is the word, not the matching: a literal the
# model has to spell is a literal it can misspell, so the wire tag is one it writes
# without effort.
LABEL_TAG = "LABEL"

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
_LABEL_LINE = LineSpec(
    tag=LABEL_TAG,
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
SKILL_NAMING_SHAPE = MicroContextShape(lines=(_LABEL_LINE,))

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
    "Respond with one LABEL line per placeholder and nothing else:\n"
    f"{render_line(_LABEL_LINE)}\n"
    "Write ONE LABEL line for EVERY placeholder you were given, and none for anything else, "
    "repeating its CURRENT name exactly so it maps back. Two spots are never the same "
    "spot: give each its own name. Use a single lowercase word or snake_case for "
    "<semantic_name>.\n"
    "IMPORTANT: write nothing else — no preamble, no explanation, no restating the "
    "routine."
)

# ── The VALUE line: ONE grammar, two customers (#1867/#1868) ──────────────────
# A parameter's value is written the same way wherever it is written, so the line is
# declared ONCE here, above both customers that deal in values: the FRAMER, which mints a
# parameter and says what the user demonstrated it with, and the BINDER, which fills a
# parameter somebody else already declared.  Two declarations would be two grammars the
# prompt and the parser could drift between — the exact failure the shape-as-data
# discipline exists to prevent — and both customers accept a value on the same terms (a
# literal span of the user's own turns, :func:`_is_a_spoken_span`), so the shared line is a
# shared contract rather than a coincidence of spelling.
#
# The tag is a short, common, non-compound word (#1842, the long-literal decay class).
VALUE_TAG = "VALUE"

# One line per parameter, in the two-field carve the labeller's line has: the parameter's
# name keyed by ``FieldShape.NAME`` (it is the binding key the caller maps home by, so a
# "name" carrying its own value is a malformed line rather than a key), then the value,
# which takes whatever is left of the line.  The COLON separator splits at the FIRST colon
# only, so a url's own ``https:`` travels in the value untouched.
_VALUE_LINE = LineSpec(
    tag=VALUE_TAG,
    role=LineRole.PER_ITEM,
    fields=(
        FieldSpec(
            name=DrawField.NAME,
            placeholder="<parameter_name>",
            shape=FieldShape.NAME,
            separator=Separator.COLON,
        ),
        FieldSpec(name=DrawField.VALUE, placeholder="<the value, in the user's own words>"),
    ),
)


# ── Fourth customer: the routine's INTERFACE — the framer (#1830/#1868) ────────
# The labeller above names every spot in the routine's IMPLEMENTATION.  What the
# routine IS — what to call it, what it is for, and what its user has to say to set
# it running again — is a different question from different evidence, so it is a
# different draw, and the two share no inputs and no outputs (#1824).
#
# This one is asked of the ASK ALONE: the user's own turns of the round, and nothing
# else — no tool calls, no values, no labeller output.  Nothing offers it a set to
# sort; it MINTS the parameters by reading what the user said and deciding which
# pieces of it they would have to say again.
#
# The failure it exists to prevent is a routine that names itself for a value and
# then demands it — `record-product-price` requiring a `what_to_extract` its own name
# already gave, so the routine could not fire from the natural second ask.  Writing
# the name and the parameters in ONE decision is what makes that incoherence
# unavailable: a skill cannot call itself a price watcher and then ask what to watch,
# because the same draw wrote both.
#
# And an accepted draw never contains an invalid line (#1828's ruling, carried to a
# customer with nothing offered): a malformed PARAMETER line or the same parameter
# twice is a contract violation like any other — re-drawn on the unchanged context for
# the whole reroll budget, then an honest WHOLE-draw failure.  With no offered set, the
# missing-line gap the labeller notices is invisible here, so the parse's dropped-line count
# (``ParsedDraw.malformed``) is what makes the same rule checkable.
#
# Since #1868 the draw also says what each minted parameter was DEMONSTRATED WITH — one
# shared ``VALUE`` line per parameter — because the framer now runs at the START of the
# round that teaches the routine, and the container that round's results are kept in is
# named from the skill plus those values (``derive_collection_name``).  The parameter set
# is minted in the same draw, so coverage is checkable exactly as the binder's is: one
# value line per minted parameter, no more and no fewer, and every value a literal span of
# what the user actually typed.  A value nobody said is the confabulation the whole scheme
# exists to make unavailable — it would name a container for a job nobody asked for.
NAME_TAG = "NAME:"
DESCRIPTION_TAG = "DESCRIPTION:"
PARAMETER_TAG = "PARAMETER"

_FRAME_NAME_LINE = LineSpec(
    tag=NAME_TAG,
    fields=(FieldSpec(name=DrawField.NAME, placeholder="<a short generic verb-noun name>"),),
)
_FRAME_DESCRIPTION_LINE = LineSpec(
    tag=DESCRIPTION_TAG,
    fields=(
        FieldSpec(name=DrawField.DESCRIPTION, placeholder="<one line: what the routine is for>"),
    ),
)
# One line per MINTED parameter, carved in two fields exactly like the labeller's
# placeholder line: the name declares ``FieldShape.NAME`` because it IS the binding key
# at instantiation (``params={<name>: …}``), so a "name" carrying its own description is
# a malformed line rather than a 60-character key; the description is REQUIRED because
# it is what the routine's `needs:` row renders — a parameter nobody can be told what to
# supply for is not an interface.
_PARAMETER_LINE = LineSpec(
    tag=PARAMETER_TAG,
    role=LineRole.PER_ITEM,
    fields=(
        FieldSpec(
            name=DrawField.NAME,
            placeholder="<parameter_name>",
            shape=FieldShape.NAME,
            separator=Separator.DASH,
        ),
        FieldSpec(
            name=DrawField.DESCRIPTION,
            placeholder="<one line: what the user supplies for it>",
        ),
    ),
)
SKILL_FRAME_SHAPE = MicroContextShape(
    lines=(_FRAME_NAME_LINE, _FRAME_DESCRIPTION_LINE, _PARAMETER_LINE, _VALUE_LINE)
)

SKILL_FRAME_SYSTEM_PROMPT = (
    "You are writing the public interface of a reusable routine. You are given what "
    "the user asked for, in their own words. Do four things:\n"
    "\n"
    "1. From what they asked for, extract the CORE USER INTENT — what they were trying "
    "to get done when they asked. Their own words are the evidence.\n"
    "\n"
    "2. Name and describe the ROUTINE by that intent: a short generic verb-noun name "
    "for the KIND of task — never the specific instance — and one line stating what "
    "the routine is for. Do not include any information about timing, scheduling, or "
    "notifications. Do not include any parameter's value in the name or description.\n"
    "\n"
    "3. Decide the PARAMETERS, starting from what the user actually provided. First, "
    "in your reasoning, list the pieces of information the user gave you — the things "
    "they said, not things they might have said. A parameter can only be one of these "
    "pieces; never something they didn't provide. Then keep only the pieces they would "
    "have to provide again to run this routine on a new occasion:\n"
    "   - A piece the name and description already carry is not a parameter — asking "
    "for it would be asking the user what they came to you for.\n"
    "   - Where results are kept, how often it runs, and whether to notify are never "
    "parameters — those are settled when the routine is set running.\n"
    "   - There is always at least one parameter.\n"
    "   Each parameter is one line: PARAMETER <name> — <description>\n"
    "   - name: the piece the user provided and how the routine uses it — if they "
    "pointed you at a website, 'url'; if they named a topic, 'topic'. Generic "
    "snake_case, never the particular site or thing's own name.\n"
    "   - A parameter holds ONE value, of the same kind the user gave it — a url stays "
    "a url, never a city pulled out of one, and never a list. One parameter for each "
    "piece they provided that survives; it's okay to have several when they provided "
    "several. Two of the same kind get names that tell them apart: 'first_plot', "
    "'second_plot'.\n"
    "   - description: one line saying what to supply. Do not include examples.\n"
    "\n"
    "4. Give each parameter its VALUE this time — the part of the user's words that "
    "supplies it. Copy that part EXACTLY as they wrote it: same characters, same "
    "spelling. Do not tidy it up, shorten it, expand it, or complete a piece they left "
    "half-said. Every parameter you named gets one value line, and nothing else does.\n"
    "\n"
    "Respond with these tagged lines and nothing else:\n"
    f"{render_line(_FRAME_NAME_LINE)}\n"
    f"{render_line(_FRAME_DESCRIPTION_LINE)}\n"
    f"{render_line(_PARAMETER_LINE)}\n"
    f"{render_line(_VALUE_LINE)}\n"
    "Write all four kinds of line: the NAME line, the DESCRIPTION line, then the "
    "parameter lines, then the value lines. Write nothing else — no preamble, no "
    "explanation, no restating the ask."
)

# The rule a model-written name must survive to be a BINDING KEY (#1668) — lowercase,
# whitespace to underscores, nothing outside ``[a-z0-9_]``, no stray underscores, and
# EMPTY when nothing survives (a name that could never be bound).  Load-bearing: a
# skill's parameter name is what instantiation keys on (``params={'url': …}``), and
# display form == invocation form everywhere it renders.
#
# It lives beside the draws that produce such names rather than in the extraction
# pipeline that used to own it, because the FRAMER mints the key inside its own draw —
# so the rule has to be reachable from here, and this module is the leaf the pipeline
# imports rather than the other way round.  Public: the labelling eval scores "did the
# draw produce a usable binding key" through THIS function, never a copy of it.
_PARAM_WHITESPACE = re.compile(r"\s+")
_PARAM_NON_IDENTIFIER = re.compile(r"[^a-z0-9_]")


def slug_parameter_name(raw: str) -> str:
    """Harden a model-written semantic name into an identifier-safe binding key."""
    lowered = _PARAM_WHITESPACE.sub("_", raw.strip().lower())
    return _PARAM_NON_IDENTIFIER.sub("", lowered).strip("_")


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

# ── Fifth customer: filling an EXISTING signature — the binder (#1867) ─────────
# The framer above writes a routine's interface once, from the round that taught it.
# This one runs every time that routine is asked for again: given the signature as it
# already stands and the user's own words of THIS round, it says what each declared
# parameter's value is.
#
# It MINTS nothing and JUDGES nothing.  The parameter set is fixed before the draw
# begins, so "which parameters does this routine have?" is not a question here — it is
# an input — and the only thing being decided is which part of the user's words fills
# each one.  That is what makes the whole answer checkable in Python: membership against
# the declared set, and every value a literal span of what the user actually said.  A
# binder that returns a value nobody typed has confabulated, and the validator can see
# it without a model in the loop.
#
# The SHORTFALL is part of the contract from the first line, not a failure mode bolted
# on later: a parameter the user's words supply nothing for gets its own ``MISSING``
# line, so the answer "they didn't say" is a POSITIVE statement the draw made rather
# than something inferred from a line that never arrived.  That is #1828's coverage
# ruling applied to a customer that knows its offered set exactly: every declared
# parameter gets exactly one line, of one kind or the other, and nothing else does.
#
# Its VALUE line is the SHARED one declared above (#1868) — the framer writes the same
# line for a parameter it just minted — so the two customers cannot describe a value's
# shape differently.  Its own tag, the shortfall's, is a short, common, non-compound word
# for the same reason (#1842, the long-literal decay class).
MISSING_TAG = "MISSING"

# The shortfall line carries the parameter's name and NOTHING else — what is missing is
# named by the parameter, and what the user would have to say is already written on the
# signature the draw was given, so a model-written reason beside it would be a second
# copy of a sentence the consumer already holds.
_MISSING_LINE = LineSpec(
    tag=MISSING_TAG,
    role=LineRole.PER_ITEM,
    fields=(FieldSpec(name=DrawField.NAME, placeholder="<parameter_name>", shape=FieldShape.NAME),),
)
SKILL_BIND_SHAPE = MicroContextShape(lines=(_VALUE_LINE, _MISSING_LINE))

BIND_SKILL_SYSTEM_PROMPT = (
    "You are a filling-in step. A routine already exists, and someone has just asked "
    "for it to be run on a new occasion. You are given the routine — what it is called, "
    "what it is for, and each thing it needs — and the user's own words, exactly as "
    "they wrote them.\n"
    "\n"
    "Fill in each thing the routine needs, from those words:\n"
    "1. Take the list of things it needs as it stands. Never add anything to that list, "
    "and never leave anything out.\n"
    "2. For each one, find the part of the user's words that supplies it, and copy that "
    "part EXACTLY as they wrote it — same characters, same spelling. Do not tidy it up, "
    "shorten it, expand it, or complete a piece they left half-said.\n"
    "3. When their words supply nothing for one of them, say it is missing. That is a "
    "real answer, and it is the right one whenever the alternative is a guess.\n"
    "\n"
    "How often the routine runs, when it should stop, and whether to tell the user are "
    "settled when it is set running. They are never things the routine needs, so they "
    "are never a value here.\n"
    "\n"
    "Write one line for each thing the routine needs, and nothing else:\n"
    f"{render_line(_VALUE_LINE)}\n"
    f"{render_line(_MISSING_LINE)}\n"
    "Write the name exactly as the routine lists it. Write nothing else — no preamble, "
    "no explanation, no restating the ask."
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
    key is :func:`slug_parameter_name`'s, applied where the name becomes a key rather
    than here, so what the draw committed to stays readable in the ledger and in an
    eval report (a spot's identity is its CURRENT name; this one is a label on it).
    ``description`` is empty only when the line stopped after the name (the grammar's
    one optional field)."""

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


class FramedParameter(BaseModel):
    """One parameter the framer MINTED (#1830): the ``name`` a user's binding will be
    keyed by (``params={<name>: …}``), the one line of what they supply for it, and the
    ``value`` this round demonstrated it with (#1868).

    Unlike a :class:`LeafLabel`'s name, this one is HARDENED here rather than
    downstream — it is not a label on something that already has an identity, it IS
    the identity, so the draw's own result carries the key instantiation will use.

    The ``value`` is carried VERBATIM, because it is a literal span of the user's own
    words and the whole point is that it is theirs — it is what the round's container is
    named from (``derive_collection_name``), so tidying it here would rename a job.

    ``description`` is ``None`` for an UNLABELLED parameter — which a framer draw never
    produces (its ``PARAMETER`` line carries both halves or it is malformed), but a
    parameter read back off the REGISTRY does (#1870: ``SkillParameter.description`` is
    itself optional).  Carried as ``None`` rather than flattened to ``""`` so unlabelled
    and labelled-with-nothing stay the two different facts they are."""

    name: str
    description: str | None = None
    value: str


class SkillSignature(BaseModel):
    """The framer's typed result (#1830): a routine's whole public interface, written
    from the user's ask alone — a GENERIC verb-noun ``name``, a one-line
    ``description``, and the ordered ``parameters`` the user would have to say again,
    each carrying the value this round demonstrated it with (#1868).

    All three come out of ONE decision, which is what stops them contradicting: a
    routine cannot name itself for something it then asks for, because the same draw
    wrote both.  ``parameters`` is never empty and never repeats a name — a draw that
    asked for nothing, or asked for one thing twice, is refused before it reaches
    here — and every one of them carries a value that is a literal span of the user's
    turns, so ``[parameter.value for parameter in parameters]`` is a total, in-declared-
    order read that the container's derived name is built from."""

    name: str
    description: str
    parameters: tuple[FramedParameter, ...] = ()


class BoundValues(BaseModel):
    """The binder's COMPLETE answer (#1867): every declared parameter, keyed by the
    name the SIGNATURE declares it under — never the spelling the draw happened to use
    — and filled with a literal span of the user's own words.

    Keyed by the declared name because that is the binding key everything downstream
    uses (``params={<name>: …}``), and ordered by the declared order because that is
    the order the derived collection name is built in.  A ``BoundValues`` is total by
    construction: a parameter the words supplied nothing for makes the answer a
    :class:`MissingParameters` instead, so a caller reading this one never has to check
    for a hole."""

    values: dict[str, str]


class MissingParameters(BaseModel):
    """The binder's SHORTFALL answer (#1867): the declared parameters the round's turns
    supply no value for, ``names`` in declared order — the structural ``request``
    signal, which since #1885 routes the turn into that state rather than failing it.

    It is an ENUMERATED OUTCOME, not a failure: the draw read the words correctly and
    the words are short of something, which is a different fact from a draw that never
    produced a usable line (that one escapes as ``None``).  ``values`` carries whatever
    the words DID supply, because throwing away a correct binding on the way to
    reporting a missing one would make the consumer ask for both again."""

    names: tuple[str, ...]
    values: dict[str, str] = {}


# What the binder answers with — the two enumerated directions, as two types rather than
# one type carrying an emptiable field, so a consumer that matched ``BoundValues`` can
# never be holding an incomplete one.
SkillBinding = BoundValues | MissingParameters


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
        matching neither is a contract violation, never a value: re-drawn on the
        unchanged context for the whole budget, then the extraction fails honestly.  A
        blank draw takes the same path (a blank payload is not a payload).
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

    async def frame_skill(
        self, content: str, *, run_target: str | None = None
    ) -> SkillSignature | None:
        """Write a routine's public INTERFACE from the user's ask alone (#1830) — the
        FOURTH customer of this machinery.  Rides the SAME ``_valid_draw`` step as the
        other three against its own declared shape (:data:`SKILL_FRAME_SHAPE`) and the
        bare-content user turn (the rendered ask IS the whole document), plus the
        runtime constraint a static shape cannot carry: the draw must mint at least one
        well-formed parameter, carry no broken PARAMETER line, never name the same
        parameter twice, and — since #1868 — give each minted parameter exactly one
        VALUE that is a literal span of what the user said.

        It takes NO offered set — that is the point.  ``content`` is the user's own
        turns and nothing else, so the parameters are minted from the ask rather than
        sorted out of a list somebody else produced (#1824).  That is also why the span
        check reads ``content`` itself rather than a second argument: for this customer
        the document IS the user's words, so there is nothing else in it a value could be
        copied out of (the binder needs its ``spoken`` separately because its document
        renders a signature too).

        ``None`` on any failure — at learn entry that means no container is built and the
        round runs unframed, and at run end the caller falls back to the deterministic
        slug with no parameters.  Honest degradation either way, rather than a
        half-written interface."""
        drawn = await self._valid_draw(
            content,
            "",
            run_target,
            shape=SKILL_FRAME_SHAPE,
            accepts=partial(_mints_a_usable_signature, spoken=content),
            system_prompt=SKILL_FRAME_SYSTEM_PROMPT,
            agent_name=PennyConstants.SKILL_FRAME_AGENT_NAME,
            prompt_type=PennyConstants.SKILL_FRAME_PROMPT_TYPE,
            user_template=_BARE_CONTENT_TEMPLATE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return _skill_signature(drawn)

    async def bind_skill(
        self,
        content: str,
        declared: Sequence[str],
        spoken: str,
        *,
        run_target: str | None = None,
    ) -> SkillBinding | None:
        """Fill an EXISTING routine's declared parameters from the user's own words
        (#1867) — the FIFTH customer of this machinery.  Rides the SAME ``_valid_draw``
        step as the other four against its own declared shape
        (:data:`SKILL_BIND_SHAPE`) and the bare-content user turn (the rendered document
        IS the whole ask), plus the runtime constraints a static shape cannot carry.

        ``content`` is the rendered document — the signature AND the user's turns.
        ``spoken`` is those turns ALONE, and it is a separate argument on purpose: a
        value is only evidence when the USER said it, so a phrase copied out of a
        parameter's own description — which the same document renders — is exactly the
        confabulation the span check exists to catch.  ``declared`` names the parameters
        the signature declares, in declared order; string-typed like ``classify_state``'s
        candidates, so this module knows parameter names and never the skill model.

        Returns :class:`BoundValues` when every declared parameter was filled,
        :class:`MissingParameters` when the words supply nothing for one or more of them
        (an enumerated outcome, not a failure), or ``None`` when no usable draw came back
        — the honest escape the other run-end customers keep."""
        drawn = await self._valid_draw(
            content,
            "",
            run_target,
            shape=SKILL_BIND_SHAPE,
            accepts=partial(_fills_the_declared_signature, declared=declared, spoken=spoken),
            system_prompt=BIND_SKILL_SYSTEM_PROMPT,
            agent_name=PennyConstants.SKILL_BIND_AGENT_NAME,
            prompt_type=PennyConstants.SKILL_BIND_PROMPT_TYPE,
            user_template=_BARE_CONTENT_TEMPLATE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return _skill_binding(drawn, declared)

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
        exactly like an untagged draw — re-drawn on the unchanged context for the
        whole budget, then an honest ``INVALID`` the machine reads as no-transition.

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

        A contract violation gets the SAME patience as poison — one budget
        (``self._reroll_attempts``, defaulting to ``DEGENERATE_REROLL_ATTEMPTS``),
        not a second number beside it (code-owner ruling, from two samples where a
        labelling draw came back with one line for four offered spots, twice each,
        and fell back after two draws).  We know deterministically what a valid draw
        looks like — the offered spots, the membership set, the shape — so a
        violation is DETECTED rather than judged, and a detected-invalid draw is
        cheap to throw away and redraw.  The fallback stays the honest end after the
        budget, not a thing to reach one draw sooner.

        Only the shape (and the prompt + attribution) differs per customer, so no
        customer owns a reroll loop or partitions a string of its own."""
        for _ in range(self._reroll_attempts):
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
            logger.warning(
                "%s output violated its declared shape — discarding and re-rolling", agent_name
            )
        logger.warning(
            "%s output still violated its declared shape after %d draws",
            agent_name,
            self._reroll_attempts,
        )
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

    Each way of missing that is the same violation, answered the same way — re-drawn
    on the unchanged context for the whole budget, then an honest whole-draw failure.
    MISSING is the observed failure (the tag decays mid-draw, the parse refuses it, and
    the spot ends up named by nothing).  TWICE is a contradictory draw, and taking
    either line would let a stray trailing one rename a spot.  A line for a spot that
    was never offered is a spot INVENTED rather than named — the shared-spot case's
    exact failure, where a value filling two argument sites is one spot and splitting it
    keys a second line to a name nobody listed.

    The caller knows the offered set, so all three are answerable rather than
    best-effort gaps to absorb; the prompt asks for exactly this ("one LABEL line for
    every placeholder you were given, and none for anything else"), so the validator and
    the contract say one thing."""
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


def _mints_a_usable_signature(drawn: ParsedDraw, *, spoken: str) -> bool:
    """The runtime constraint the framing shape can't carry (#1830/#1868) — and the whole
    of what makes an accepted signature usable, with nothing offered to check it against.

    The PARAMETER half has three ways to be violated, each answered the same way (re-drawn
    on the unchanged context for the whole budget, then an honest whole-draw failure).  NO
    parameter is the floor the prompt states: a routine that needs nothing said to it can
    only repeat the one occasion it was shown.  A MALFORMED line is a line the parse
    dropped — the grammar's best-effort default is wrong for a customer with no offered
    set, because there is no gap for it to show up as, so the drop is counted and refused
    instead.  A REPEATED name is a contradictory draw, and a parameter's name is its
    binding key: two lines under one key means one of them silently disappears at
    instantiation.

    Names are compared HARDENED, because that is what they become — ``Page URL`` and
    ``page_url`` are one key, not two."""
    if drawn.malformed:
        return False
    minted = _drawn_names(drawn, PARAMETER_TAG)
    if not (minted and all(minted) and len(set(minted)) == len(minted)):
        return False
    return _values_were_demonstrated(drawn, minted, spoken)


def _values_were_demonstrated(drawn: ParsedDraw, minted: Sequence[str], spoken: str) -> bool:
    """The VALUE half of the framer's contract (#1868): every parameter this same draw
    minted was demonstrated with something the user actually said.

    COVERAGE and MEMBERSHIP are ONE comparison against the minted set, exactly as the
    binder compares against the DECLARED set (#1828's rule for a customer that knows its
    offered set — here it knows it because it just wrote it): a parameter with no value, a
    value given twice, and a value for a parameter no line minted are the same violation.

    And every value must be a literal span of what the user said.  The framer's whole
    document IS the user's turns, so ``spoken`` is that document — unlike the binder,
    whose document also renders a signature a value could be copied out of.  The check is
    load-bearing rather than decorative: the container the round writes into is named from
    these values, so a value nobody typed would mint a container for a job nobody asked
    for, and no later step could tell."""
    values = [item for item in drawn.items if item.tag == VALUE_TAG]
    if sorted(_drawn_names(drawn, VALUE_TAG)) != sorted(minted):
        return False
    return all(_is_a_spoken_span(item.fields[DrawField.VALUE], spoken) for item in values)


def _drawn_names(drawn: ParsedDraw, tag: str) -> list[str]:
    """The HARDENED parameter names the draw's ``tag`` lines carry, in draw order — the
    keys everything downstream maps by, read the same way for both of the framer's
    per-item lines so a name written one way on each still matches itself."""
    return [
        slug_parameter_name(item.fields[DrawField.NAME]) for item in drawn.items if item.tag == tag
    ]


def _skill_signature(drawn: ParsedDraw) -> SkillSignature | None:
    """The framing draw read by FIELD NAME — what the routine is called, what it is
    for, what its user has to supply, and what this round supplied (#1868).  ``None`` only
    if a REQUIRED line somehow went missing, which the declared shape already refuses;
    kept as the belt-and-braces guard so a blank interface can never be persisted.

    Parameter names are hardened through :func:`slug_parameter_name` HERE, unlike a
    leaf label's: this name is not a description of something that already has an
    identity, it is the binding key itself — which is also how a VALUE line finds the
    PARAMETER line it belongs to, whichever spelling each of them used.  Every lookup
    resolves: the draw was accepted only because the two sets match exactly."""
    name = drawn.field(NAME_TAG, DrawField.NAME)
    description = drawn.field(DESCRIPTION_TAG, DrawField.DESCRIPTION)
    if name is None or description is None:
        return None
    values = {
        slug_parameter_name(item.fields[DrawField.NAME]): item.fields[DrawField.VALUE]
        for item in drawn.items
        if item.tag == VALUE_TAG
    }
    return SkillSignature(
        name=name,
        description=description,
        parameters=tuple(
            _framed_parameter(item, values) for item in drawn.items if item.tag == PARAMETER_TAG
        ),
    )


def _framed_parameter(item: ParsedLine, values: dict[str, str]) -> FramedParameter:
    """One minted parameter joined to the value line that answered it, by hardened name."""
    key = slug_parameter_name(item.fields[DrawField.NAME])
    return FramedParameter(
        name=key, description=item.fields[DrawField.DESCRIPTION], value=values[key]
    )


# Whitespace tolerance for the literal-span check, declared ONCE beside the check that
# applies it (the grammar's own "tolerance is declared once, deliberately" discipline).
_SPOKEN_WHITESPACE = re.compile(r"\s+")


def spoken_form(text: str) -> str:
    """``text`` in the form the span check compares — whitespace runs collapsed to a
    single space, trimmed, case-folded.

    That is the WHOLE tolerance, and it is deliberately small: a value survives a line
    break the render introduced and a capital the user did not type, and nothing else.
    Punctuation, spelling and word order compare exactly, because that is where the
    failure class lives — the measured defect (#1866) was a bound url carrying an
    underscore that appears nowhere in the user's message, and every looser comparison
    accepts it.

    Public: the binding eval reads a drawn value through THIS function, never through a
    copy of it, so what a case calls a match and what production calls a span are one
    definition."""
    return _SPOKEN_WHITESPACE.sub(" ", text).strip().casefold()


def _is_a_spoken_span(value: str, spoken: str) -> bool:
    """Whether ``value`` is a literal span of what the user said — plain containment
    over :func:`spoken_form`, with no fuzzy matching and no threshold.

    Plain containment rather than the whole-token test structural provenance uses
    (#1809): that one guards a value against turning up by accident inside an entire
    fetched PAGE, while this haystack is the handful of sentences the user just typed,
    where an accidental containment is not a thing that happens — and demanding whole
    tokens would refuse a phrase written hard against a comma."""
    return spoken_form(value) in spoken_form(spoken)


def _fills_the_declared_signature(
    drawn: ParsedDraw, *, declared: Sequence[str], spoken: str
) -> bool:
    """The runtime constraints the binding shape cannot carry (#1867) — and the whole of
    what makes an accepted binding trustworthy.

    COVERAGE and MEMBERSHIP are ONE comparison: the drawn names, hardened, must equal
    the declared names as a multiset.  That is #1828's rule for a customer that knows
    its offered set exactly — a parameter left unanswered, one answered twice, and one
    nobody declared are the same violation, answered the same way (re-drawn on the
    unchanged context for the whole budget, then an honest whole-draw failure).  A
    MALFORMED line is refused for the framer's reason (#1830): the grammar drops it
    best-effort, so the counted drop is the only way this customer sees it at all.

    And every VALUE must be a literal span of what the USER said — the check the whole
    customer exists for.  A routine pointed at a url nobody typed is worse than a routine
    nobody could point anywhere, so an invented value is a contract violation and never a
    binding."""
    if drawn.malformed:
        return False
    drawn_names = sorted(slug_parameter_name(item.fields[DrawField.NAME]) for item in drawn.items)
    if drawn_names != sorted(slug_parameter_name(name) for name in declared):
        return False
    return all(
        _is_a_spoken_span(item.fields[DrawField.VALUE], spoken)
        for item in drawn.items
        if item.tag == VALUE_TAG
    )


def _skill_binding(drawn: ParsedDraw, declared: Sequence[str]) -> SkillBinding:
    """An accepted draw as the enumerated answer it is (#1867), read in DECLARED order
    and keyed by the DECLARED name — the binding key everything downstream uses, so a
    draw that wrote ``Page URL`` where the signature says ``page_url`` still maps home.

    Total by construction: the draw was accepted only because it carries exactly one
    line per declared parameter, so every lookup here resolves."""
    lines = {slug_parameter_name(item.fields[DrawField.NAME]): item for item in drawn.items}
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in declared:
        line = lines[slug_parameter_name(name)]
        if line.tag == VALUE_TAG:
            values[name] = line.fields[DrawField.VALUE]
        else:
            missing.append(name)
    if missing:
        return MissingParameters(names=tuple(missing), values=values)
    return BoundValues(values=values)


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
