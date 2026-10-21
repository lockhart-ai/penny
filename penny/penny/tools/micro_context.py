"""Single-shot micro-context extraction for content tools.

A content tool (``browse``) that carries a micro-instruction runs the fetched
page content through a FRESH, scoped single-shot model call — content +
instruction, no tools — and returns a small typed result to the main loop.  The
bulk page body never enters the parent run's context: only the extracted value
(or an honest enumerated failure) plus the fetch handle to the stored full
content come back (the anchor discipline).  A micro-context is structurally
incapable of confabulating a stored value it has never seen.

**Browse-extract is the one customer** (#2615): reading a page is not a reasoning
task, so its draw runs with thinking off and its plain text IS the value,
verbatim — nothing parses it.  The tagged-line shape machinery that used to serve
a second customer (the notify composer, retired with it in #2656) is gone with
that customer; what remains is the poison screen every draw still rides.

Every draw is screened by the same degeneracy / leaked-Harmony-envelope /
empty-draw detectors the agent-loop reroll guard uses (:mod:`penny.text_validity`):
poison is discarded and re-drawn on the *unchanged* context up to
``DEGENERATE_REROLL_ATTEMPTS``, never appended (appending a collapse feeds it back
in).  An unusable result is an honest enumerated outcome, never a silent empty.

It is itself a ledger-visible model call — its own ``agent_name`` /
``prompt_type`` so run traces attribute it — but it does NOT inflate the parent
run's context: the parent only ever sees the returned :class:`MicroContextResult`.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.config_params import RuntimeParams, token_budget_for
from penny.constants import PageReading, PennyConstants
from penny.llm.call_link import CallLink
from penny.llm.models import LlmMessage, LlmResponse
from penny.text_validity import (
    has_leaked_harmony_envelope,
    is_degenerate_run,
    is_empty_draw,
)
from penny.validation.conditions import ConditionKey

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)

# Browse-extract retired its tagged contract (#2615): reading a page is not a
# reasoning task.  Production measured the tagged shape costing a correct read
# its whole thinking budget (~5 minutes) arguing over an undecidable
# partial/missing distinction, and — with thinking off (#2609) — a correct
# plain-text answer discarded and re-rolled for carrying no parseable tag.  The
# code owner's ruling: drop the shape.  The draw always runs with thinking off
# and returns plain text; nothing parses it.  ``EXTRACTED_TAG`` / ``NOT_PRESENT_TAG``
# / ``EXTRACT_MISSING_TAG`` and the declared shape are gone with it —
# see :meth:`MicroContext.extract`, whose draw never rides a shape loop at all.
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
    "Links travel as numbers: every link in the content — each page's own "
    "address included — is written as its number in brackets, like [7]. The "
    "address is part of the value: begin every answer with the number of the "
    "page it came from, so the person reading the answer knows where it came "
    "from and can go back to it, whatever the instruction asked for. When a "
    "value names a link, write its number in brackets instead of the address.\n"
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


class MicroExtractOutcome(StrEnum):
    """The enumerated outcome of a micro-context extraction — a closed set the
    caller renders one way each (never a silent empty).

    Two members (#2615): with the tagged shape gone, there is no more
    structural difference between "found some of it" and "found none of it"
    — the plain-text draw says which, and that reads as variance, not as a
    closed field.  ``EXTRACTION_FAILED`` is the one escape left: no usable
    draw came back after the whole reroll budget (degenerate, a leaked
    envelope, or blank).
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


class MicroContext:
    """Runs a single-shot extraction over bulk content via the shared model client."""

    def __init__(
        self,
        model_client: LlmClient,
        *,
        reroll_attempts: int = PennyConstants.DEGENERATE_REROLL_ATTEMPTS,
        db: Database,
    ) -> None:
        self._model_client = model_client
        self._reroll_attempts = reroll_attempts
        # The store the discarded draws are marked on (#2702) — REQUIRED, no
        # default and no ``None`` (a construction without one would silently
        # stop marking its rerolls): the enclosing tool's own db (browse), the
        # running Penny's db (the eval driver), the test's ``db`` fixture.
        self._db = db

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

    async def _draw_clean(
        self,
        content: str,
        instruction: str,
        run_target: str | None,
        *,
        call_link: CallLink,
        reading_note: str = "",
        thinking_enabled: bool = True,
    ) -> str | None:
        """The raw extraction text, re-rolling on poison; ``None`` if every draw
        is unusable.  Mirrors the agent-loop reroll guard — discard poison, never
        append it, re-draw on the same context, abort after the attempt budget.

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
        reasoning task).  A draw the cap cut (``finish_reason="length"``) gets ONE
        more call on the same messages with thinking forced off — the budget spent
        its reasoning tokens, so the last draw is asked to answer with what it has
        — and that response takes the cut one's place; the poison screen then
        reads it like any other draw.  That retry only fires when the cut draw
        itself drew WITH thinking on: a customer already drawing thinking-off
        would get an identical second call for nothing, so the guard is
        ``not retried_cut and thinking_enabled``.  At most once per draw either
        way: a retry that is itself cut is used as-is, an honest escape rather
        than an unbounded loop."""
        messages = self._messages(content, instruction, reading_note)
        draw_link = call_link.child()
        runtime = self._model_client.runtime
        token_budget = token_budget_for(
            runtime if runtime is not None else RuntimeParams(),
            PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        )
        retried_cut = False
        for attempt in range(self._reroll_attempts):
            response = await self._model_client.chat(
                messages=messages,
                agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
                prompt_type=PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
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
                    PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
                )
                # The cut draw's own row says why it was set aside (#2702);
                # the thinking-off draw that takes its place follows unmarked.
                self._mark_discarded(response, ConditionKey.BUDGET_CUT)
                response = await self._model_client.chat(
                    messages=messages,
                    agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
                    prompt_type=PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
                    call_link=draw_link,
                    run_target=run_target,
                    token_budget=token_budget,
                    thinking_enabled=False,
                )
            text = response.content or ""
            condition = self._poison_condition(text)
            if condition is None:
                return text
            self._mark_discarded(response, condition)
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
    def _poison_condition(text: str) -> ConditionKey | None:
        """The specific condition making this draw poison, or ``None`` if it is
        clean: a leaked Harmony envelope, a degeneration collapse, or a blank
        draw (#2615: browse-extract has no shape left to catch an empty payload,
        so it is screened here instead) — the same transport artifacts the
        agent-loop reroll guard discards, plus the empty-draw check it also
        uses.  The named condition is what a discarded draw's own ``promptlog``
        row carries (#2702)."""
        if has_leaked_harmony_envelope(text):
            return ConditionKey.TOOL_CALL_LEAK
        if is_degenerate_run(text):
            return ConditionKey.DEGENERATE_OUTPUT
        if is_empty_draw(text):
            return ConditionKey.EMPTY
        return None

    def _mark_discarded(self, response: LlmResponse, condition: ConditionKey) -> None:
        """Stamp on the draw's own ``promptlog`` row that its output was not
        used (#2702) — the same post-hoc mark the agent loop applies, by the
        row id the response carries.

        The one no-op: ``response.prompt_log_id`` is ``None``.  The client
        stamps it only when it writes the row, so it is ``None`` exactly when
        no row exists to mark — a client built without a ``db`` (a test, a
        one-off script) or a call that never finished (those rows already say
        so via ``call_outcome``/``call_error_*``)."""
        if response.prompt_log_id is None:
            return
        self._db.messages.mark_prompt_discarded(response.prompt_log_id, condition.value)

    @staticmethod
    def _messages(
        content: str,
        instruction: str,
        reading_note: str = "",
    ) -> list[LlmMessage]:
        """The scoped two-message context: the extraction contract framing, then
        the user turn pairing the instruction with the bulk content (and, on a
        link-list read, the note saying how the page was read)."""
        return [
            LlmMessage(role="system", content=MICRO_CONTEXT_SYSTEM_PROMPT),
            LlmMessage(
                role="user",
                content=_USER_TEMPLATE.format(
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
