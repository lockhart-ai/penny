"""Single-shot micro-context extraction for content tools.

A content tool (``browse``) that carries a micro-instruction runs the fetched
page content through a FRESH, scoped single-shot model call — content +
instruction, no tools — and returns a small typed result to the main loop.  The
bulk page body never enters the parent run's context: only the one-line
extracted value (or an honest enumerated failure) plus the fetch handle to the
stored full content come back (the anchor discipline).  A micro-context is
structurally incapable of confabulating a stored value it has never seen.

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
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from penny.constants import PennyConstants
from penny.text_validity import has_leaked_harmony_envelope, is_blank, is_degenerate_run

if TYPE_CHECKING:
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)

# The extraction framing — one legible, single-purpose instruction.  It asks a
# world-question ("what's on the page?"), never a machine-question, and forbids
# inventing a value not in the content (the confabulation the micro-context
# structurally prevents).
MICRO_CONTEXT_SYSTEM_PROMPT = (
    "You are an extraction step. You are given the full text of one or more web "
    "pages and a single instruction naming exactly what to pull out of them. "
    "Return only the extracted value, as briefly as the instruction allows — no "
    "preamble, no explanation, no restating the instruction. If the requested "
    "information is not present in the content, say so plainly in one short line. "
    "Never invent a value that is not in the content."
)

_USER_TEMPLATE = "Instruction: {instruction}\n\nContent:\n{content}"


class MicroExtractOutcome(StrEnum):
    """The enumerated outcome of a micro-context extraction — a closed set the
    caller renders one of three ways (never a silent empty)."""

    EXTRACTED = "extracted"
    EXTRACTION_FAILED = "extraction_failed"
    POISON_REROLL_FAILED = "poison_reroll_failed"


class MicroContextResult(BaseModel):
    """The small typed result the main loop receives from a micro-context.

    ``value`` carries the extracted text on :attr:`MicroExtractOutcome.EXTRACTED`
    and is empty on either failure — the caller renders the failure from the
    outcome.  This value is what flows to the main loop verbatim; the parent
    model never re-transcribes it.
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
    ) -> None:
        self._model_client = model_client
        self._reroll_attempts = reroll_attempts

    async def extract(
        self, content: str, instruction: str, *, run_target: str | None = None
    ) -> MicroContextResult:
        """Extract ``instruction`` from ``content`` in one scoped model call.

        Draws a poison-free output (re-rolling on a collapse / leaked envelope),
        then classifies: a clean non-blank draw is the extracted value; a clean
        but blank draw is an honest extraction failure; no clean draw after the
        re-rolls is a poison-reroll failure.
        """
        draw = await self._draw_clean(content, instruction, run_target)
        if draw is None:
            return MicroContextResult(outcome=MicroExtractOutcome.POISON_REROLL_FAILED)
        if is_blank(draw):
            return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTION_FAILED)
        return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=draw.strip())

    async def _draw_clean(
        self, content: str, instruction: str, run_target: str | None
    ) -> str | None:
        """The raw extraction text, re-rolling on poison; ``None`` if every draw
        is unusable.  Mirrors the agent-loop reroll guard — discard poison, never
        append it, re-draw on the same context, abort after the attempt budget."""
        messages = self._messages(content, instruction)
        run_id = uuid.uuid4().hex
        for attempt in range(self._reroll_attempts):
            response = await self._model_client.chat(
                messages=messages,
                agent_name=PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
                prompt_type=PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
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
    def _messages(content: str, instruction: str) -> list[dict]:
        """The scoped two-message context: the extraction framing, then the
        instruction paired with the bulk content."""
        return [
            {"role": "system", "content": MICRO_CONTEXT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(instruction=instruction, content=content),
            },
        ]
