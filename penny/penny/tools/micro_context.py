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
and then fails honestly, per customer (``EXTRACTION_FAILED`` / ``INVALID``).  A
violation of an OPTIONAL or PER_ITEM line is simply an ABSENT line at the
GRAMMAR level — but whether absence is acceptable is the CUSTOMER's to declare,
as a runtime constraint (``accepts``): extraction's stated-shortfall line is
OPTIONAL and best-effort (no offered set to check it against), while the
notifier refuses a draw whose text is not worth delivering (:func:`_is_sendable`).
An accepted draw never contains an invalid line.

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
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.config_params import RuntimeParams, token_budget_for
from penny.constants import PageReading, PennyConstants
from penny.llm.call_link import CallLink
from penny.llm.models import LlmMessage
from penny.llm.refusal import is_refusal
from penny.text_validity import (
    half_formed_send_reason,
    has_leaked_harmony_envelope,
    is_degenerate_run,
)
from penny.tools.micro_context_shape import (
    FieldSpec,
    LineAnchor,
    LineRole,
    LineSpec,
    MicroContextShape,
    ParsedDraw,
    PayloadSpan,
    parse_draw,
    render_line,
)

if TYPE_CHECKING:
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)


class DrawField(StrEnum):
    """The named fields the declared shapes carve out of a tagged line.

    A customer reads a draw by these names — ``drawn.field(EXTRACTED_TAG,
    DrawField.VALUE)`` — never by position and never by partitioning a string, so
    a grammar change lands in the declaration rather than in a bespoke parser."""

    VALUE = "value"
    REASON = "reason"
    MESSAGE = "message"
    MISSING = "missing"


# The output tags — the enumerated contract, present on BOTH sides of the
# interface: the prompt names them and the parse reads them.  The label is
# the interface between model-space and Python-space (the enumerated-cases
# doctrine, #1554).  Without it the not-present case comes back as arbitrary
# prose, which a blank-check classifier reads as an extracted value — a
# confabulation-shaped leak through the exact surface whose design guarantee is
# "cannot confabulate stored values".
EXTRACTED_TAG = "EXTRACTED:"
NOT_PRESENT_TAG = "NOT_PRESENT:"
# The stated shortfall (#2360): what the instruction named that the page does
# not carry at all, read off the thinking the real-size front-page incident
# produced.  The model kept wanting to "note the missing summaries" with
# nowhere in the contract to put that note, re-litigating on every draw whether
# a headline without its description was a partial unit or a missing one.  This
# gives it the place: one line, before the value, naming what is not there —
# EXTRACTED still answers with what IS there, and nothing about the decision
# changes, so the loop has an exit that costs nothing else.
#
# Named ``EXTRACT_MISSING_TAG`` rather than bare ``MISSING_TAG`` so this
# customer's own name reads as its own — a WIRE tag is read against its own
# declared shape, and a model only ever sees one system prompt per call, but the
# PYTHON name is module-level and must not collide with another customer's.
EXTRACT_MISSING_TAG = "MISSING:"

# The extraction shape (#1814/#2360): ``EXTRACTED:`` and ``NOT_PRESENT:`` are
# still the two ALTERNATIVE tags, exactly one of which answers the draw — but
# since #2360 an OPTIONAL ``MISSING:`` line may come first, naming what the
# instruction asked for and the page does not carry at all.  ``EXTRACTED:``
# spans the REMAINDER (the value is everything that follows — a digest or an
# item-per-line list is served whole, #1682), so nothing may follow its own
# line without being swallowed into the value: that is why ``MISSING:`` is
# declared to come BEFORE it, never after.  ``_EXTRACTED_LINE``'s anchor is
# ANYWHERE rather than OPENS_DRAW for exactly this reason — it must still be
# found at line 0 when there is no ``MISSING:`` line, and at line 1 when there
# is.  The accepted trade-off: a draw that puts unrelated junk before
# ``EXTRACTED:`` is no longer rejected by position alone the way it used to
# be — the system prompt's "write nothing outside the value itself" and the
# "nothing invented" claim still catch that, just not via the tag's position.
# ``NOT_PRESENT:`` keeps OPENS_DRAW: a ``MISSING:`` line pairs with an
# ``EXTRACTED:`` read (something is there to answer with) and never with
# "the page carries none of it", so its exclusivity-by-position is unchanged.
# The ``NOT_PRESENT:`` reason itself stays a single LINE, so a not-present
# apology can never be multi-line-promoted into a value.
_EXTRACT_MISSING_LINE = LineSpec(
    tag=EXTRACT_MISSING_TAG,
    role=LineRole.OPTIONAL,
    anchor=LineAnchor.OPENS_DRAW,
    fields=(
        FieldSpec(
            name=DrawField.MISSING,
            placeholder="<one short line naming what was asked for and is not on the page>",
        ),
    ),
)
_EXTRACTED_LINE = LineSpec(
    tag=EXTRACTED_TAG,
    role=LineRole.ALTERNATIVE,
    anchor=LineAnchor.ANYWHERE,
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
EXTRACT_SHAPE = MicroContextShape(lines=(_EXTRACT_MISSING_LINE, _EXTRACTED_LINE, _NOT_PRESENT_LINE))

# The extraction framing — one legible, single-purpose instruction.  It asks a
# world-question ("what's on the page?"), never a machine-question, forbids
# inventing a value not in the content, and enumerates the closed set of output
# forms so classification downstream is a deterministic tag parse, never a
# judgment over free prose.  The value may be as long as the instruction requires
# (a digest, a list) — the TAG must open the output; only its shape is fixed, not
# its length.  The tagged lines are RENDERED FROM THE SHAPE, so the prompt
# cannot state a contract the parse doesn't read.
#
# WHERE THE BOUNDARY BETWEEN THE TAGS FALLS IS STATED (#1942).  An instruction
# routinely names several things at once — what each item is called, where it
# points, what it says — and a real page carries some of them and not others.
# The contract never said which tag that page takes, so it was decided per draw:
# one draw over a page holding titles and links but no summaries answered
# NOT_PRESENT (losing the titles and links with it), and a later draw over the
# SAME content extracted.  The rule is stated as the STATE the content is in —
# carries some of it / carries none of it — rather than as a rule about
# instructions that hedge, because what a page holds is not a property of how the
# ask was worded, and a rule keyed to the wording would simply not fire for the
# ask nobody had in front of them.
#
# PLAIN WORDS, WITH PERMISSION AS WELL AS PROHIBITION (#2360).  The two sentences
# that used to carry this alone — "give whatever the content has ... and leave
# out the ones it has nothing for" and "never invent a value that is not in the
# content" — told the model what NOT to do with a missing part and gave it
# nowhere to put the observation it kept trying to make anyway.  Below, a part
# that is only partly on the page is told PLAINLY what to do with it (give the
# part that is there) and WHERE the rest of the thought goes (one MISSING: line,
# named together, before the value) — permission for the thing it was already
# reaching for, not just another way of forbidding invention.
#
# ROUND 2 (#2360): the first round's own "before" run showed the MISSING: line
# working — the undecidable partial/missing debate shrank — but left a page
# asking for an unbounded list (every headline on a ~90-link front page, not a
# fixed "top 3") still mostly timing out.  The new thinking traces showed a
# DIFFERENT loop past that point: having drafted its full answer, the model
# re-scanned the content to double-check it had not missed an item ("Wait,
# I'll check if there's any other story. No.") — completeness anxiety, not the
# partial/missing question #2360 opened on.  Round 2's one-pass sentence ("Read
# the content once...") did not help in round 2's own run, so it is gone.
#
# ROUND 3 (#2360): round 2's run showed the timeout loop move past the re-scan
# into arguing about the OUTPUT FORMAT of one item — whether headline, link and
# summary go on one line or two — because the prompt said "a list (put one item
# per line)" while an item has several parts.  The prompt now says what one
# item's line IS (its parts, in the order the instruction names them, " — "
# separated; a part the page does not give is left off the line) instead of
# leaving the shape of an item for the model to argue about.
# ROUND 4 (#2360): replaced the "AT ALL" sentence after a replay of the slowest
# draws took "the missing line names the descriptions" from 7/18 to 17/18 on the
# top-three ask.
MICRO_CONTEXT_SYSTEM_PROMPT = (
    "You are an extraction step. You are given the full text of one or more web "
    "pages and a single instruction naming exactly what to pull out of them. "
    f"An optional {EXTRACT_MISSING_TAG} line may open your output (see below); whether or "
    "not it is there, the next line — or the first, if it is not — opens with "
    "one of these two tags:\n"
    f"{render_line(_EXTRACTED_LINE)}\n"
    f"{render_line(_NOT_PRESENT_LINE)}\n"
    f"After {EXTRACTED_TAG}, the extracted value is EVERYTHING that follows — as "
    "long as the instruction requires: a single value, one or more paragraphs, or "
    "a list. In a list, each item is one line holding that item's parts in the "
    'order the instruction names them, separated by " — "; a part the page does '
    "not give for that item is simply left off that item's line.\n"
    "An instruction often names several things at once, and a page rarely carries "
    f"all of them. When the page carries at least one, write {EXTRACTED_TAG} and "
    "give exactly what it has for each thing named — a thing that is only partly "
    "on the page still gets a line for the part that is there. Whatever the "
    "instruction named that is not in your answer because the page does not give "
    "it — for a list, a part that none of your items has — name it, all of it "
    f"together, in one short line, before {EXTRACTED_TAG}\n"
    f"{render_line(_EXTRACT_MISSING_LINE)}\n"
    f"Use {NOT_PRESENT_TAG}, on a single line, when the page carries NONE of what "
    "was asked for. Never invent a value that is not in the content — when "
    f"something named is not there, say so on {EXTRACT_MISSING_TAG} (or {NOT_PRESENT_TAG} "
    "when nothing is) instead of guessing it. Write nothing outside the value "
    "itself — no preamble, no explanation, no restating the instruction."
)

# The structural reading-mode note (#2388).  The extension already knows whether it
# read the page as its article body or as its own link list (``isLinkIndex``/``richer``
# in ``browser/src/content/page_text.ts``) — a fact only it can see, since it is the
# one place that reads the live DOM.  On a link-list read the extractor is otherwise
# left to work out, from the text's shape alone, which lines are navigation, where a
# link's title ends and its description begins, and what "the top N" means — and a
# real-size front page is exactly where that reasoning ran long (#2360).  Told plainly
# instead: two replays of the slowest draws on the top-three ask, 18 per variant, cut
# median reasoning tokens a further 18-23% on top of #2360's own prompt, with
# correctness unchanged (18/18 either way).
#
# On an ARTICLE read the note would be FALSE, so it is simply absent — never
# rendered, never hedged.  That absence is what keeps every existing customer (every
# browse read before this ticket) byte-identical: ``reading_note`` defaults to "",
# and the template below with an empty note is character-for-character the one line
# above it replaces.
_LINK_LIST_READING_NOTE = (
    "About the content: this page was read as its list of links — one line per link, "
    "in the order the links appear on the page; that order is the page's own ranking, "
    "so the first links of a kind are its top ones. Each line is a link's own label "
    "and address. A label can run past its title into a further sentence that "
    "describes what it links to.\n\n"
)

_USER_TEMPLATE = "Instruction: {instruction}\n\n{reading_note}Content:\n{content}"

# ── Second customer: telling the user about a cycle — the notifier (#1911) ─────
# The customer above (extraction) runs around a CHAT turn.  This one closes a COLLECTOR
# cycle: once the program's calls are covered and the collection says notify, the
# framework assembles what the cycle did, what it wrote, and the past messages nearest
# to it — all in Python — and this draw turns that into the one message the user gets.
#
# It exists because the same job used to be four numbered steps APPENDED to the
# collector's own program, carried out by the same loop, on the same context: two
# ``read_similar`` calls, "compose one short friendly message", and a ``send_message``.
# Measured, that tail was where the cycles died — 42 of 49 reroll-exhaustion deaths in
# one instrumented run landed in it, immediately after a ``read_similar``, because by
# then the context had grown long and chat-flavoured and the model's tool-call envelope
# decayed.  The tail was 86% of the deaths and almost none of the value.
#
# So the two reads move to Python (a store query needs no model), the compose stays a
# model job (it is prose about a particular find), and the send moves to Python (the
# queue was always deterministic).  What is left for the model is the ONE thing only a
# model can do, on a FRESH short context with NO tool channel at all — which is why a
# decayed tool-call envelope cannot end this cycle: there is no call to decay into.
MESSAGE_TAG = "MESSAGE:"

# One line spanning the REMAINDER — the ``EXTRACTED:`` shape (#1682), since a message
# is prose that may run to several lines.  The tag carries its own colon so it is
# self-delimiting, like the extraction contract's.
#
# It is anchored ANYWHERE rather than at the head of the draw, which is the one place
# this differs from extraction, and deliberately: the tag's whole job here is to say
# where the message the user receives BEGINS, so a chatty draw's "Sure, here you go."
# is something to leave behind rather than something to fail the cycle's notification
# over.  Anything before the tag is dropped; everything after it is the message.
_MESSAGE_LINE = LineSpec(
    tag=MESSAGE_TAG,
    span=PayloadSpan.REMAINDER,
    fields=(
        FieldSpec(
            name=DrawField.MESSAGE,
            placeholder="<the message — it may begin on this same line>",
        ),
    ),
)
NOTIFY_SHAPE = MicroContextShape(lines=(_MESSAGE_LINE,))

# WORDING FOR CODE-OWNER REVIEW (#1911).  A minimal distillation of the retired
# ``Prompt.COLLECTOR_NOTIFY_STEPS`` into a single-purpose framing: the same message
# shape it asked for (a quick greeting · what was found in plain words · the source url
# when there is one · a callback ONLY when a past message is genuinely related), with
# the two ``read_similar`` steps gone because their results are handed in already, and
# the ``send_message`` step gone because the framework sends.  Nothing about tools, and
# nothing about deciding WHETHER to write — both were settled before this draw.
#
# The opening sentence NAMES THE DOCUMENT'S SECTIONS IN THE ORDER THEY RENDER (#1934),
# so the one sentence framing the document cannot describe a different document from
# the one under it: what the cycle wrote leads, the calls follow, and the earlier
# messages are named as BACKGROUND — the only part of the document the reported run did
# not produce, and the part a measured composition took as its template.
NOTIFY_SYSTEM_PROMPT = (
    "You are writing one message to the user. A job of theirs just ran on its own "
    "and found something, and you are given everything about that run: what it wrote "
    "down, how it got there, and — as background — the closest things the two of you "
    "have said before.\n"
    "\n"
    "Write the message they will actually receive:\n"
    "1. Open with a quick greeting.\n"
    "2. Name every item under what it wrote this cycle, in plain words — not a "
    "description of the skill, and not just the ones that look most relevant: the job "
    "already decided what counted as a find, so every one of them gets named.\n"
    "3. Include the source link when the run has one.\n"
    "4. Add one line calling back to an earlier message ONLY when one of them is "
    "genuinely about this. When none of them is, say nothing about them — that is the "
    "ordinary case and it is fine.\n"
    "\n"
    "Keep it short and friendly, the way a person messages a friend. Write only what "
    "the run actually found: never a detail that is not in front of you, and never a "
    "claim about what happens next.\n"
    "\n"
    "Respond with this line and nothing else:\n"
    f"{render_line(_MESSAGE_LINE)}\n"
    f"Everything after {MESSAGE_TAG} is the message itself, so write no preamble, no "
    "explanation, and no restating of these instructions."
)

# A user turn that is the rendered document ALONE — no ``Instruction:``/``Content:``
# wrapper.  That frame is the extraction customer's (natural for "here's a page, pull X
# out"); a customer whose ask lives entirely in its system prompt would only repeat the
# instruction by wrapping, and would label a structured situation as bulk content.  The
# notifier reads this way too (#1911), so it is named for the SHAPE of the turn rather
# than for either customer.
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

    ``missing`` (#2360) carries the draw's OPTIONAL stated shortfall — what the
    instruction named that the page does not carry at all — and is only ever
    populated alongside ``EXTRACTED``: an outcome that already answers nothing
    (``NOT_PRESENT``) has no shortfall to add beside its ``reason``.  Empty when
    the draw named nothing missing, which is the honest "it had everything
    named" read rather than an absent field.
    """

    outcome: MicroExtractOutcome
    value: str = ""
    reason: str = ""
    missing: str = ""


class DrawFailure(StrEnum):
    """Why no usable draw came back once the budget was spent — the two ways a
    micro-context comes up empty, which each customer maps onto its own enumerated
    outcome rather than sharing one.

    ``POISON`` is a transport artifact every time (a degeneration collapse or a
    leaked Harmony envelope).  ``INVALID`` is a clean draw that never matched its
    DECLARED shape — an untagged output, a blank payload, or a runtime constraint
    the shape can't carry (e.g. the notifier's sendability check, :func:`_is_sendable`)."""

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
        self,
        content: str,
        instruction: str,
        *,
        call_link: CallLink,
        run_target: str | None = None,
        reading: PageReading = PageReading.ARTICLE,
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

        ``call_link`` (#2458) is this draw's own linkage — REQUIRED, no default.
        Since this is the one customer invoked from INSIDE a dispatched tool call,
        it carries that call's id; ``_draw_clean`` derives this draw's OWN child
        link from it (:meth:`~penny.llm.call_link.CallLink.child`), so the run this
        extraction makes is attributed back to the enclosing run and tool call with
        no ambient read anywhere.

        ``reading`` (#2388) is the structural mark the extension set when it read the
        page — on :attr:`PageReading.LINK_LIST` the user turn carries one note, BEFORE
        ``Content:``, saying so; on the default :attr:`PageReading.ARTICLE` the note is
        absent and the rendered turn is byte-identical to before this parameter
        existed.
        """
        reading_note = _LINK_LIST_READING_NOTE if reading == PageReading.LINK_LIST else ""
        drawn = await self._valid_draw(
            content,
            instruction,
            run_target,
            call_link=call_link,
            shape=EXTRACT_SHAPE,
            reading_note=reading_note,
        )
        if isinstance(drawn, DrawFailure):
            return MicroContextResult(outcome=_EXTRACT_FAILURES[drawn])
        return _extraction_result(drawn)

    async def compose_notification(
        self,
        content: str,
        *,
        call_link: CallLink,
        run_target: str | None = None,
    ) -> str | None:
        """Write the one message a finished collector cycle sends the user (#1911) — the
        SECOND customer of this machinery, and the first that closes a background cycle
        rather than a chat turn.

        ``content`` is the framework-assembled document (``penny.notification``): what
        the cycle did, what it wrote, and the past messages Python already looked up.
        Rides the SAME ``_valid_draw`` step as extraction, against its own declared
        shape (:data:`NOTIFY_SHAPE`) and the bare-content user turn (the assembled
        document IS the whole ask), plus the runtime constraint a static shape cannot
        carry: the drawn text must be a message worth delivering
        (:func:`_is_sendable`).

        ``None`` when no usable draw came back — the honest escape every run-end
        customer keeps.  The caller records that on the run and sends nothing, which is
        the whole degradation: a cycle that found something and could not phrase it
        still leaves its find written down and its run record saying so."""
        drawn = await self._valid_draw(
            content,
            "",
            run_target,
            call_link=call_link,
            shape=NOTIFY_SHAPE,
            accepts=_is_sendable,
            system_prompt=NOTIFY_SYSTEM_PROMPT,
            agent_name=PennyConstants.NOTIFY_COMPOSE_AGENT_NAME,
            prompt_type=PennyConstants.NOTIFY_COMPOSE_PROMPT_TYPE,
            user_template=_BARE_CONTENT_TEMPLATE,
        )
        if isinstance(drawn, DrawFailure):
            return None
        return drawn.field(MESSAGE_TAG, DrawField.MESSAGE)

    async def _valid_draw(
        self,
        content: str,
        instruction: str,
        run_target: str | None,
        *,
        call_link: CallLink,
        shape: MicroContextShape,
        accepts: Callable[[ParsedDraw], bool] = _accept_any,
        system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT,
        agent_name: str = PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        prompt_type: str = PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
        user_template: str = _USER_TEMPLATE,
        reading_note: str = "",
    ) -> ParsedDraw | DrawFailure:
        """The ONE validate step every customer rides (#1814): a poison-screened
        draw, read against its DECLARED ``shape`` and checked against ``accepts``
        (the runtime constraint a static shape can't carry — e.g. the notifier's
        sendability check), re-rolled on the unchanged context while the contract
        is violated, then an honest typed failure the customer maps onto its own
        outcome.

        A contract violation gets the SAME patience as poison — one budget
        (``self._reroll_attempts``, defaulting to ``DEGENERATE_REROLL_ATTEMPTS``),
        not a second number beside it (code-owner ruling).  We know deterministically
        what a valid draw looks like — the shape, and the customer's own runtime
        constraint — so a violation is DETECTED rather than judged, and a
        detected-invalid draw is cheap to throw away and redraw.  The fallback stays
        the honest end after the budget, not a thing to reach one draw sooner.

        Only the shape (and the prompt + attribution) differs per customer, so no
        customer owns a reroll loop or partitions a string of its own.  ``call_link``
        (#2458) is passed straight through to ``_draw_clean``, which derives ONE
        child link for the whole draw before the reroll loop starts — every attempt
        of one draw still names the same run (and, for a tool-nested draw, the same
        call) it was made for, never re-derived per attempt.  ``reading_note``
        (#2388) rides the same way — only ``extract`` ever supplies a non-empty one;
        every other customer's ``_BARE_CONTENT_TEMPLATE`` ignores it."""
        for _ in range(self._reroll_attempts):
            draw = await self._draw_clean(
                content,
                instruction,
                run_target,
                call_link=call_link,
                system_prompt=system_prompt,
                agent_name=agent_name,
                prompt_type=prompt_type,
                user_template=user_template,
                reading_note=reading_note,
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
        call_link: CallLink,
        system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT,
        agent_name: str = PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        prompt_type: str = PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
        user_template: str = _USER_TEMPLATE,
        reading_note: str = "",
    ) -> str | None:
        """The raw extraction text, re-rolling on poison; ``None`` if every draw
        is unusable.  Mirrors the agent-loop reroll guard — discard poison, never
        append it, re-draw on the same context, abort after the attempt budget.

        The ``system_prompt`` + ledger attribution are parameters (defaulting to the
        browse-extract contract) so a second output contract — notify composition
        (#1911) — rides the SAME poison/reroll loop without duplicating it.

        Derives its OWN draw link via ``call_link.child()`` (#2458) ONCE, before the
        reroll loop — the whole point being that a micro-context's call is its own
        ledger-visible run, never folded into the run it was made for, while every
        reroll attempt of this ONE draw still logs under the SAME child run id.
        ``child()`` is what keeps nesting correct at any depth: it converts
        ``call_link``'s own POSITION (the tool call it is presently inside, if any)
        into this draw's logged ``parent_tool_call_id``, so a draw made from inside
        a dispatched tool call links to that call, while a draw made from inside
        ANOTHER draw (never itself positioned inside a tool call) correctly gets
        ``parent_tool_call_id=None``.

        Every call carries its kind's own token budget (#2476), resolved off the
        shared client's ``runtime`` by ``agent_name`` — a client built with no
        runtime at all (a test, a one-off script) falls back to a default-only
        lookup, the same chain every other runtime param reads.  A draw the cap cut
        (``finish_reason="length"``) gets ONE more call on the same messages with
        thinking disabled — the budget spent its reasoning tokens, so the last
        draw is asked to answer with what it has — and that response takes the cut
        one's place; the poison screen and the parse then read it like any other
        draw.  At most once per draw: a retry that is itself cut is used as-is, an
        honest escape rather than an unbounded loop."""
        messages = self._messages(content, instruction, system_prompt, user_template, reading_note)
        draw_link = call_link.child()
        runtime = self._model_client.runtime
        token_budget = token_budget_for(
            runtime if runtime is not None else RuntimeParams(), agent_name
        )
        retried_cut = False
        for attempt in range(self._reroll_attempts):
            response = await self._model_client.chat(
                messages=messages,
                agent_name=agent_name,
                prompt_type=prompt_type,
                call_link=draw_link,
                run_target=run_target,
                token_budget=token_budget,
            )
            if response.finish_reason == PennyConstants.FINISH_REASON_LENGTH and not retried_cut:
                retried_cut = True
                logger.warning(
                    "%s draw was cut at its token budget — one more draw with thinking off",
                    agent_name,
                )
                response = await self._model_client.chat(
                    messages=messages,
                    agent_name=agent_name,
                    prompt_type=prompt_type,
                    call_link=draw_link,
                    run_target=run_target,
                    token_budget=token_budget,
                    thinking_enabled=False,
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
        reading_note: str = "",
    ) -> list[LlmMessage]:
        """The scoped two-message context: the contract framing (``system_prompt``,
        default the browse-extract contract), then the user turn shaped by the
        customer's ``user_template`` (default: the instruction paired with bulk
        content; the notifier passes the bare-content template).

        ``reading_note`` (#2388) only ever renders on ``_USER_TEMPLATE`` — the
        notifier's ``_BARE_CONTENT_TEMPLATE`` ("{content}") has no ``{reading_note}``
        slot to fill, so ``str.format`` simply ignores the supplied kwarg for it."""
        return [
            LlmMessage(role="system", content=system_prompt),
            LlmMessage(
                role="user",
                content=user_template.format(
                    instruction=instruction, content=content, reading_note=reading_note
                ),
            ),
        ]


def _extraction_result(drawn: ParsedDraw) -> MicroContextResult:
    """The typed result for a valid extraction draw, read off the parsed draw by
    field name — the ``EXTRACTED:`` value (everything after the tag) or the
    ``NOT_PRESENT:`` reason (its first line only, so a not-present apology can never
    be multi-line-promoted into a value).

    ``MISSING:`` (#2360) is read only on the ``EXTRACTED`` branch — it is an
    OPTIONAL line, so an absent or malformed one is simply no shortfall, never a
    violation."""
    value = drawn.field(EXTRACTED_TAG, DrawField.VALUE)
    if value is not None:
        missing = drawn.field(EXTRACT_MISSING_TAG, DrawField.MISSING) or ""
        return MicroContextResult(
            outcome=MicroExtractOutcome.EXTRACTED, value=value, missing=missing
        )
    reason = drawn.field(NOT_PRESENT_TAG, DrawField.REASON)
    if reason is None:
        # Unreachable while the two tags are declared ALTERNATIVE (a parsed draw
        # carries one of them) — honest rather than silent if that ever changes.
        return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTION_FAILED)
    return MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason=reason)


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


def _is_sendable(drawn: ParsedDraw) -> bool:
    """The runtime constraint the notify shape cannot carry (#1911): the drawn text has
    to be a message a user should actually receive.

    Read through the SHARED ``half_formed_send_reason`` rule — the same definition the
    send path validates with and the run-health classifier flags ``⚠ HALF-FORMED SEND``
    on — so what the framework draws and what it is willing to deliver are one
    definition, and a blank / bare-url / trailing-off draw is a contract violation
    re-drawn on the unchanged context rather than something the queue refuses later.  A
    model REFUSAL is refused here for the same reason: it is not a message about the
    run, and nothing downstream would improve it."""
    message = drawn.field(MESSAGE_TAG, DrawField.MESSAGE)
    if message is None:
        return False
    return half_formed_send_reason(message) is None and not is_refusal(message)
