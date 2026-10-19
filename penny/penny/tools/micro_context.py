"""Single-shot micro-context extraction for content tools.

A content tool (``browse``) that carries a micro-instruction runs the fetched
page content through a FRESH, scoped single-shot model call — content +
instruction, no tools — and returns a small typed result to the main loop.  The
bulk page body never enters the parent run's context: only the extracted value
(or an honest enumerated failure) plus the fetch handle to the stored full
content come back (the anchor discipline).  A micro-context is structurally
incapable of confabulating a stored value it has never seen.

A customer that declares a tagged shape does so as DATA, not a bespoke parser:
since #1814 each one names a
:class:`~penny.tools.micro_context_shape.MicroContextShape` — its tags, which
lines are required, which repeat per item, how each line's payload carves into
named fields, and which of those fields may be absent.  The prompt's contract
block is RENDERED from that declaration (``render_line``), so the tags and
separators the model is told to write are literally the ones the parser splits
on; :func:`~penny.tools.micro_context_shape.parse_draw` is the ONE validate step
a shaped customer rides, so it never partitions a string itself.  What that
bought is the failure it replaced: a ``PARAM`` line whose separator came back as
an en-dash where the parser wanted an em-dash used to partition to nothing and
hand the entire remainder over as a parameter's semantic name — a 60-character
"name" carrying its own description, persisted as a skill's binding key, with no
reroll because the parse had "succeeded".  Tolerance is now declared once and
"parsed to something implausible" is a contract violation, not a value.

**Browse-extract is the one customer that no longer declares a shape (#2615):**
reading a page is not a reasoning task, so its draw runs with thinking off and
its plain text IS the value, verbatim — see :meth:`MicroContext.extract` and
:data:`MICRO_CONTEXT_SYSTEM_PROMPT` below.  The notifier still declares one
(:data:`NOTIFY_SHAPE`), so the shape machinery above serves it: a violation of a
REQUIRED part of ITS shape is rerolled on the unchanged context and then fails
honestly; a violation of an OPTIONAL or PER_ITEM line is simply an ABSENT line
at the GRAMMAR level — but whether absence is acceptable is the CUSTOMER's to
declare, as a runtime constraint (``accepts``): the notifier refuses a draw
whose text is not worth delivering (:func:`_is_sendable`).  An accepted draw
never contains an invalid line.

Every customer's single call — shaped or not — is screened by the same
degeneracy / leaked-Harmony-envelope / empty-draw detectors the agent-loop
reroll guard uses (:mod:`penny.text_validity`): poison is discarded and
re-drawn on the *unchanged* context up to ``DEGENERATE_REROLL_ATTEMPTS``, never
appended (appending a collapse feeds it back in).  An unusable result is an
honest enumerated outcome, never a silent empty.

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
    is_empty_draw,
)
from penny.tools.micro_context_shape import (
    FieldSpec,
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
    """The named field a declared shape's line carves out.

    A customer reads a draw by this name — ``drawn.field(MESSAGE_TAG,
    DrawField.MESSAGE)`` — never by position and never by partitioning a string, so
    a grammar change lands in the declaration rather than in a bespoke parser.
    Browse-extract retired its own three fields with its shape (#2615): it reads
    a draw's plain text directly, so this enum now serves the notifier alone."""

    MESSAGE = "message"


# Browse-extract retired its tagged contract (#2615): reading a page is not a
# reasoning task.  Production measured the tagged shape costing a correct read
# its whole thinking budget (~5 minutes) arguing over an undecidable
# partial/missing distinction, and — with thinking off (#2609) — a correct
# plain-text answer discarded and re-rolled for carrying no parseable tag.  The
# code owner's ruling: drop the shape.  The draw always runs with thinking off
# and returns plain text; nothing parses it.  ``EXTRACTED_TAG`` / ``NOT_PRESENT_TAG``
# / ``EXTRACT_MISSING_TAG`` and the declared ``EXTRACT_SHAPE`` are gone with it —
# see :meth:`MicroContext.extract`, which no longer rides the shape/reroll loop
# below at all.
#
# The three things the shape used to carry STRUCTURALLY are said in PLAIN WORDS
# instead, carried over from the shape's own history (#1942/#2360): the
# boundary between "some of it" and "none of it" is a property of what the
# PAGE carries, never of how the ask was worded; a part that is only partly on
# the page gets the part that is there, named alongside what is missing,
# rather than a forbidding sentence with nowhere to put the missing thought;
# and a list ask gets one item per line, each holding only the parts the page
# gives, in the order the instruction names them.
#
# The address is part of the value (#2648).  The draw's content is headed by the
# page's own ``## browse: <url>`` line, but that header reads as scaffolding, not
# as part of the answer: measured on a source-URL standing rule, every extraction
# came back a bare value with no URL, and 8 of 9 page-reading replies carried no
# source at all.  So the value itself now leads with the page's address, whatever
# the instruction asked for — the person reading the answer (the enclosing chat
# turn, the collector writing it into the store) can cite where it came from
# without re-deriving which page it was.
MICRO_CONTEXT_SYSTEM_PROMPT = (
    "You are an extraction step. You are given the full text of one or more web "
    "pages and a single instruction naming exactly what to pull out of them. "
    "Answer in plain text from the page alone, as a list with one item per "
    "line when the instruction asks for several, and never a value the page "
    "doesn't state.\n"
    "\n"
    "Each page's content begins with that page's own address — the URL the page "
    "was read at — and the address is part of the value: begin every answer with "
    "the URL of the page it came from, so the person reading the answer knows "
    "where it came from and can go back to it, whatever the instruction asked "
    "for.\n"
    "\n"
    "An instruction often names several things at once, and a page rarely "
    "carries all of them. Give what the page has for each thing named, in the "
    "order the instruction names them. The page's gaps are part of the answer: "
    "the person reading your answer cannot see the page, so when a thing the "
    "instruction names is not on the page — on one item or on any of them — "
    "say so where that thing's value would have gone, in a few words. A short "
    "note on the item's own line does this (for example, \"— no description on "
    'the page"), and so does one short sentence after the answer naming the '
    "thing the page does not have.\n"
    "\n"
    "If the page carries none of what was asked for, say so plainly in one "
    "sentence and stop there.\n"
    "\n"
    "Write only the answer — no preamble, no restating the instruction, and "
    "nothing about how you are answering. The address the answer begins with is "
    "part of the answer, not a preamble."
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
    "2. Tell them what the job found in flowing, conversational sentences — the way "
    "you'd tell a friend what you turned up, not a form to fill in. Name every item "
    "under what it wrote this cycle — not just the ones that look most relevant: the "
    "job already decided what counted as a find, so every one of them gets named — "
    "and give each one the facts that make it worth telling, read straight from the "
    "record: where it's listed, its price, and what changed about it (a price drop, "
    "a new listing, something hard to find). The items' storage labels (like "
    "`listing-01`) are bookkeeping — talk about the things themselves.\n"
    "3. When the record says more entries exist than are listed, mention that too, "
    "in the same breath, so the count is never a surprise.\n"
    "4. Include the source link when the run has one.\n"
    "5. Add one line calling back to an earlier message ONLY when one of them is "
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

    Two members (#2615): with the tagged shape gone, there is no more
    structural difference between "found some of it" and "found none of it"
    — the plain-text draw says which, and that reads as variance, not as a
    closed field.  ``EXTRACTION_FAILED`` is the one escape left: no usable
    draw came back after the whole reroll budget (degenerate, a leaked
    envelope, or blank) — this merges what used to be a separate
    ``POISON_REROLL_FAILED``, since there is no second failure KIND (a shape
    violation) left to tell it apart from any more.
    """

    EXTRACTED = "extracted"
    EXTRACTION_FAILED = "extraction_failed"


class MicroContextResult(BaseModel):
    """The small typed result the main loop receives from a micro-context.

    ``value`` is the draw's whole plain-text answer on
    :attr:`MicroExtractOutcome.EXTRACTED` (#2615) — what the page has for each
    thing the instruction named, any shortfall, and any "none of it" said
    inside that same text; nothing parses it into separate fields any more.
    Empty on :attr:`MicroExtractOutcome.EXTRACTION_FAILED` — the caller
    renders that from the outcome alone.  The populated field is what flows to
    the main loop verbatim; the parent model never re-transcribes it.
    """

    outcome: MicroExtractOutcome
    value: str = ""


class DrawFailure(StrEnum):
    """Why no usable draw came back once the budget was spent, for a customer
    still riding :meth:`MicroContext._valid_draw`'s shape/accepts loop
    (#2615: browse-extract no longer does — see :meth:`MicroContext.extract`).

    ``POISON`` is a transport artifact every time (a degeneration collapse, a
    leaked Harmony envelope, or a blank draw).  ``INVALID`` is a clean draw
    that never matched its DECLARED shape or its runtime constraint (e.g. the
    notifier's sendability check, :func:`_is_sendable`)."""

    POISON = "poison"
    INVALID = "invalid"


def _accept_any(_drawn: ParsedDraw) -> bool:
    """The default runtime constraint: none.  For a customer with no membership
    sets to check, the declared shape IS the whole contract."""
    return True


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

        Reading a page is not a reasoning task (#2615): the draw runs with
        thinking OFF and its plain text IS the value, verbatim — nothing
        parses it any more.  The poison screen (collapse / leaked envelope /
        blank draw → discard and re-roll on the unchanged context,
        :meth:`_draw_clean`) is the whole validity check left; there is no
        shape to violate.  No usable draw after the whole reroll budget is
        :attr:`MicroExtractOutcome.EXTRACTION_FAILED` — the honest escape,
        never a silent empty.

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
        text = await self._draw_clean(
            content,
            instruction,
            run_target,
            call_link=call_link,
            reading_note=reading_note,
            thinking_enabled=False,
        )
        if text is None:
            return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTION_FAILED)
        return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=text.strip())

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
        Rides ``_valid_draw``'s shape/accepts loop (#2615: browse-extract no longer
        does), against its own declared shape (:data:`NOTIFY_SHAPE`) and the
        bare-content user turn (the assembled document IS the whole ask), plus the
        runtime constraint a static shape cannot carry: the drawn text must be a
        message worth delivering (:func:`_is_sendable`).

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
            thinking_enabled=False,
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
        thinking_enabled: bool = True,
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
                thinking_enabled=thinking_enabled,
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
        thinking_enabled: bool = True,
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
        lookup, the same chain every other runtime param reads.

        ``thinking_enabled`` (#2615) is the per-customer reasoning switch —
        browse-extract always draws with it ``False`` (reading a page is not a
        reasoning task), every other customer keeps the default ``True``.  A
        draw the cap cut (``finish_reason="length"``) gets ONE more call on the
        same messages with thinking forced off — the budget spent its
        reasoning tokens, so the last draw is asked to answer with what it has
        — and that response takes the cut one's place; the poison screen then
        reads it like any other draw.  That retry only fires when the cut draw
        itself drew WITH thinking on: a customer already drawing thinking-off
        would get an identical second call for nothing, so the guard is
        ``not retried_cut and thinking_enabled``.  At most once per draw either
        way: a retry that is itself cut is used as-is, an honest escape rather
        than an unbounded loop."""
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
                thinking_enabled=thinking_enabled,
            )
            if (
                response.finish_reason == PennyConstants.FINISH_REASON_LENGTH
                and not retried_cut
                and thinking_enabled
            ):
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
        """A degeneration collapse, a leaked Harmony envelope, or a blank draw
        (#2615: browse-extract has no shape left to catch an empty payload, so
        it is screened here instead) — the same transport artifacts the
        agent-loop reroll guard discards, plus the empty-draw check it also
        uses."""
        return has_leaked_harmony_envelope(text) or is_degenerate_run(text) or is_empty_draw(text)

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
