"""Single-shot micro-context extraction for content tools.

A content tool (``browse``) that carries a micro-instruction runs the fetched
page content through a FRESH, scoped single-shot model call — content +
instruction, no tools — and returns a small typed result to the main loop.  The
bulk page body never enters the parent run's context: only the one-line
extracted value (or an honest enumerated failure) plus the fetch handle to the
stored full content come back (the anchor discipline).  A micro-context is
structurally incapable of confabulating a stored value it has never seen.

The output contract is ENUMERATED on both sides of the interface: the prompt
names the two tagged forms (``EXTRACTED: <value>`` / ``NOT_PRESENT: <reason>``)
and classification is a deterministic tag parse — the label is the interface
between model-space and Python-space, so a not-present apology can never be
promoted to an extracted value.  Untagged output is a contract violation: one
reroll of the unchanged context, then an honest ``EXTRACTION_FAILED``.

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

# The two output tags — the enumerated contract, present on BOTH sides of the
# interface: the prompt names them and the classifier parses them.  The label is
# the interface between model-space and Python-space (the enumerated-cases
# doctrine, #1554).  Without it the not-present case comes back as arbitrary
# prose, which a blank-check classifier reads as an extracted value — a
# confabulation-shaped leak through the exact surface whose design guarantee is
# "cannot confabulate stored values".
EXTRACTED_TAG = "EXTRACTED:"
NOT_PRESENT_TAG = "NOT_PRESENT:"

# The extraction framing — one legible, single-purpose instruction.  It asks a
# world-question ("what's on the page?"), never a machine-question, forbids
# inventing a value not in the content, and enumerates the closed set of output
# forms so classification downstream is a deterministic tag parse, never a
# judgment over free prose.
MICRO_CONTEXT_SYSTEM_PROMPT = (
    "You are an extraction step. You are given the full text of one or more web "
    "pages and a single instruction naming exactly what to pull out of them. "
    "Respond with exactly one line, in one of these two forms:\n"
    f"{EXTRACTED_TAG} <the extracted value, as briefly as the instruction allows>\n"
    f"{NOT_PRESENT_TAG} <one short line naming what is missing>\n"
    "Use NOT_PRESENT when the requested information is not in the content. "
    "Never invent a value that is not in the content, and write nothing else — "
    "no preamble, no explanation, no restating the instruction."
)

_USER_TEMPLATE = "Instruction: {instruction}\n\nContent:\n{content}"

# How many draws an UNTAGGED (but poison-free) output gets: the first draw plus
# one reroll of the unchanged context.  Untagged output is a contract violation,
# not a world-fact — it is never promoted to a value; after the reroll the
# extraction fails honestly.
_UNTAGGED_DRAW_BUDGET = 2

# ── Second customer: run-end skill naming (#1665) ──────────────────────────────
# The naming contract is a DIFFERENT enumerated output shape riding the SAME
# poison-screen + reroll machinery (``_draw_clean``): given a distilled routine,
# write a GENERIC verb-noun name + a one-line generic description.  Its two tags
# are enumerated on both sides of the interface, exactly like EXTRACTED:/NOT_PRESENT:
# — the system prompt names them and ``_parse_label`` parses them deterministically.
NAME_TAG = "NAME:"
DESCRIPTION_TAG = "DESCRIPTION:"

SKILL_NAMING_SYSTEM_PROMPT = (
    "You are a naming step. You are given a reusable routine — a numbered list of "
    "tool calls with fill-in-the-blank {variables} — plus the message that first "
    "demonstrated it. Name the routine GENERICALLY: what KIND of task it is, as a "
    "short verb-noun label (e.g. 'watch a page price for changes', 'summarize a "
    "subscription feed'), never the specific thing this one instance happened to "
    "use. Treat every {variable} as a user-supplied input, not part of the name.\n"
    "Respond with exactly two lines, in these forms:\n"
    f"{NAME_TAG} <a short generic verb-noun name>\n"
    f"{DESCRIPTION_TAG} <one line saying what the routine does, generically>\n"
    "Write nothing else — no preamble, no explanation, no restating the routine."
)

# The single per-call ask; the routine itself is the content.  Fixed, so the
# caller only supplies the content (the naming contract is a property of this
# customer, not a per-call parameter).
_SKILL_NAMING_INSTRUCTION = "Name this routine generically and describe in one line what it does."


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


class SkillLabel(BaseModel):
    """The run-end naming micro-context's typed result (#1665): a GENERIC verb-noun
    ``name`` and a one-line generic ``description`` for a distilled routine.  Both
    are non-blank by construction (``_parse_label`` returns ``None`` otherwise, so
    the caller falls back to the deterministic slug — naming never blocks extraction)."""

    name: str
    description: str


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

        Each draw is poison-screened (collapse / leaked envelope → discard and
        re-roll on the unchanged context), then classified by a **deterministic
        tag parse** — ``EXTRACTED:`` → the value, ``NOT_PRESENT:`` → the
        enumerated not-present outcome carrying the reason.  An untagged (but
        clean) draw is a contract violation, never a value: it gets exactly one
        reroll of the unchanged context, then the extraction fails honestly.
        ``is_blank`` is subsumed by the parse (a blank draw carries no tag).
        """
        for _ in range(_UNTAGGED_DRAW_BUDGET):
            draw = await self._draw_clean(content, instruction, run_target)
            if draw is None:
                return MicroContextResult(outcome=MicroExtractOutcome.POISON_REROLL_FAILED)
            result = self._parse_tagged(draw)
            if result is not None:
                return result
            logger.warning("Micro-context output untagged — one reroll of the unchanged context")
        logger.error("Micro-context output untagged after reroll — extraction failed")
        return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTION_FAILED)

    @staticmethod
    def _parse_tagged(draw: str) -> MicroContextResult | None:
        """Deterministic classification of one clean draw by its output tag.

        ``EXTRACTED:`` with a non-blank payload → the value; ``NOT_PRESENT:``
        with a non-blank payload → the not-present outcome carrying the reason.
        Anything else — no tag, or a tag with a blank payload — is ``None``
        (invalid), which the caller rerolls once and then fails honestly.
        """
        text = draw.strip()
        if text.startswith(EXTRACTED_TAG):
            value = text[len(EXTRACTED_TAG) :].strip()
            if not is_blank(value):
                return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=value)
        if text.startswith(NOT_PRESENT_TAG):
            reason = text[len(NOT_PRESENT_TAG) :].strip()
            if not is_blank(reason):
                return MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason=reason)
        return None

    async def label_skill(
        self, content: str, *, run_target: str | None = None
    ) -> SkillLabel | None:
        """Write a GENERIC name + description for a distilled routine (#1665) — the
        second customer of this machinery.  Rides the SAME poison-screen + reroll
        draw loop as ``extract``, with the naming system prompt and its own ledger
        attribution, then a deterministic two-tag parse (``NAME:`` / ``DESCRIPTION:``).

        Returns the label, or ``None`` on ANY failure (poison exhausted, or the
        model never produced both tagged lines) — the caller falls back to the
        deterministic slug, so run-end skill extraction NEVER blocks on the rewrite."""
        for _ in range(_UNTAGGED_DRAW_BUDGET):
            draw = await self._draw_clean(
                content,
                _SKILL_NAMING_INSTRUCTION,
                run_target,
                system_prompt=SKILL_NAMING_SYSTEM_PROMPT,
                agent_name=PennyConstants.SKILL_NAMING_AGENT_NAME,
                prompt_type=PennyConstants.SKILL_NAMING_PROMPT_TYPE,
            )
            if draw is None:
                return None
            label = self._parse_label(draw)
            if label is not None:
                return label
            logger.warning("Skill-naming output untagged — one reroll of the unchanged context")
        logger.warning("Skill-naming output untagged after reroll — falling back to the slug")
        return None

    @staticmethod
    def _parse_label(draw: str) -> SkillLabel | None:
        """Deterministic parse of the two-line naming contract — a ``NAME:`` line
        and a ``DESCRIPTION:`` line, each with a non-blank payload.  Missing either
        (or a blank payload) is a contract violation → ``None`` (the caller rerolls
        once and then falls back), never a partial label."""
        name = _tagged_payload(draw, NAME_TAG)
        description = _tagged_payload(draw, DESCRIPTION_TAG)
        if name is None or description is None:
            return None
        return SkillLabel(name=name, description=description)

    async def _draw_clean(
        self,
        content: str,
        instruction: str,
        run_target: str | None,
        *,
        system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT,
        agent_name: str = PennyConstants.BROWSE_EXTRACT_AGENT_NAME,
        prompt_type: str = PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE,
    ) -> str | None:
        """The raw extraction text, re-rolling on poison; ``None`` if every draw
        is unusable.  Mirrors the agent-loop reroll guard — discard poison, never
        append it, re-draw on the same context, abort after the attempt budget.

        The ``system_prompt`` + ledger attribution are parameters (defaulting to the
        browse-extract contract) so a second output contract — run-end skill naming
        (#1665) — rides the SAME poison/reroll loop without duplicating it."""
        messages = self._messages(content, instruction, system_prompt)
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
        content: str, instruction: str, system_prompt: str = MICRO_CONTEXT_SYSTEM_PROMPT
    ) -> list[dict]:
        """The scoped two-message context: the contract framing (``system_prompt``,
        default the browse-extract contract), then the instruction paired with the
        bulk content."""
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(instruction=instruction, content=content),
            },
        ]


def _tagged_payload(draw: str, tag: str) -> str | None:
    """The stripped payload of the first line of ``draw`` beginning with ``tag``,
    or ``None`` when no such line exists or its payload is blank — the deterministic
    per-tag parse the naming contract (#1665) is classified by."""
    for line in draw.splitlines():
        stripped = line.strip()
        if stripped.startswith(tag):
            payload = stripped[len(tag) :].strip()
            if not is_blank(payload):
                return payload
    return None
