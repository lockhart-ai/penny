"""Single-shot micro-context extraction for content tools.

A content tool (``browse``) that carries a micro-instruction runs the fetched
page content through a FRESH, scoped single-shot model call — content +
instruction, no tools — and returns a small typed result to the main loop.  The
bulk page body never enters the parent run's context: only the extracted value
(or an honest enumerated failure) plus the fetch handle to the stored full
content come back (the anchor discipline).  A micro-context is structurally
incapable of confabulating a stored value it has never seen.

The output contract is ENUMERATED on both sides of the interface: the prompt
names the two tagged forms (``EXTRACTED: <value>`` / ``NOT_PRESENT: <reason>``)
and classification is a deterministic tag parse — the label is the interface
between model-space and Python-space, so a not-present apology can never be
promoted to an extracted value.  The tag must OPEN the output (its first line);
after ``EXTRACTED:`` the value is EVERYTHING that follows — as long as the
instruction requires (a single value, a paragraph, or an item-per-line list), so
a digest-shaped instruction is served whole — while the ``NOT_PRESENT:`` reason
stays a single line (it can never be multi-line-promoted into a value).  Untagged
output is a contract violation: one reroll of the unchanged context, then an
honest ``EXTRACTION_FAILED``.

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
from collections.abc import Sequence
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
# judgment over free prose.  The value may be as long as the instruction requires
# (a digest, a list) — the TAG must open the output; only its shape is fixed, not
# its length.
MICRO_CONTEXT_SYSTEM_PROMPT = (
    "You are an extraction step. You are given the full text of one or more web "
    "pages and a single instruction naming exactly what to pull out of them. "
    "The FIRST LINE of your output must open with one of these two tags:\n"
    f"{EXTRACTED_TAG} <the value — it may begin on this same line>\n"
    f"{NOT_PRESENT_TAG} <one short line naming what is missing>\n"
    f"After {EXTRACTED_TAG}, the extracted value is EVERYTHING that follows — as "
    "long as the instruction requires: a single value, one or more paragraphs, or "
    "a list (put one item per line). Use "
    f"{NOT_PRESENT_TAG}, on a single line, when the requested information is not in "
    "the content. Never invent a value that is not in the content, and write "
    "nothing outside the value itself — no preamble, no explanation, no restating "
    "the instruction."
)

_USER_TEMPLATE = "Instruction: {instruction}\n\nContent:\n{content}"

# How many draws an UNTAGGED (but poison-free) output gets: the first draw plus
# one reroll of the unchanged context.  Untagged output is a contract violation,
# not a world-fact — it is never promoted to a value; after the reroll the
# extraction fails honestly.
_UNTAGGED_DRAW_BUDGET = 2

# ── Second customer: run-end skill naming (#1665/#1668/#1770) ──────────────────
# The naming contract is a DIFFERENT enumerated output shape riding the SAME
# poison-screen + reroll machinery (``_draw_clean``): given a distilled routine
# AND its candidate parameters, write a GENERIC verb-noun name + a one-line generic
# description AND — per candidate — a VERDICT: the user supplied this value (a real
# parameter, named semantically) or they did not (a placeholder).  Every tag is
# enumerated on both sides of the interface, exactly like EXTRACTED:/NOT_PRESENT: —
# the system prompt names them and ``_parse_label`` parses them deterministically.
# Each line is keyed by the candidate's CURRENT (arg-derived) name, so the system
# owns an unambiguous mapping back; the model writes LABELS and VERDICTS only.
#
# Why a verdict at all (#1770): the distiller classifies an unexplained string leaf
# as a required parameter BY DEFAULT — a default that is wrong for a value the model
# derived-and-transformed from a tool result or invented outright, since neither can
# ever be supplied by a user.  Those two origins share no literal span with what
# produced them, so no substring test can reach them (and #1659 already ruled prose
# matching out); the question "did the USER provide this?" has an answer in every
# case and is a judgment, which is why the labeller is the right instrument.
#
# Why the response is GROUPED BY VERDICT (#1807): the judgment was right and the
# TRANSCRIPTION was wrong — in every observed failure the thinking concluded "treat as
# PARAM" and the line came out ``PLACEHOLDER``, always sitting inside a run of genuine
# placeholder lines.  Asked for one line per candidate, the model drafts the line and
# writes the tag last as a prefix, so a run of placeholders sweeps the odd parameter in
# with it.  Asked for the two GROUPS, the split itself is the thing being written and a
# candidate's group is settled before any of its line is.  The parse is unchanged and
# stays ORDER-AGNOSTIC: grouping is what the contract asks for, never what it requires,
# so an interleaved draw still maps every candidate exactly as before.
NAME_TAG = "NAME:"
DESCRIPTION_TAG = "DESCRIPTION:"
PARAM_TAG = "PARAM"
PLACEHOLDER_TAG = "PLACEHOLDER"
# The em-dash separating a parameter's semantic name from its description on a
# ``PARAM <current>: <semantic> — <description>`` line.
_PARAM_DESC_SEPARATOR = "—"

SKILL_NAMING_SYSTEM_PROMPT = (
    "You are a naming step. You are given the conversation that led to the "
    "construction of a reusable routine, the routine itself — a numbered list of "
    "tool calls with fill-in-the-blank {parameters} — the message that first "
    "demonstrated it, and the routine's candidate parameters (each currently named "
    "after the tool argument it fills, and shown with the value it was demonstrated "
    "with). Do three things:\n"
    "1. From the conversation, extract the CORE USER INTENT — what the user was "
    "trying to get done when they asked for this (e.g. keeping an eye on a "
    "listing's price). The routine exists to serve that intent.\n"
    "2. Name and describe the ROUTINE by that intent: a short verb-noun name for "
    "the KIND of task (e.g. 'watch a listing price for changes'), generic — never "
    "the specific instance — and never mechanics alone ('fetch and store data' "
    "says nothing about when to reach for it).\n"
    "3. Decide, for EVERY candidate parameter, where its demonstrated value came "
    "from. There are two cases:\n"
    "   - THE USER GAVE IT. It came from the user — a page they named, a thing "
    "they asked to be found, a label they chose, a place they said to keep it — "
    "including when the assistant "
    "reworded it ('the current price' for their \"find the price\"). This is a real "
    "parameter: name it by what the value MEANS to the user (e.g. 'url', "
    "'what_to_find', 'label'), NOT the tool argument it happens to fill, and "
    "describe in one line what to supply for it.\n"
    "   - THE ASSISTANT PRODUCED IT. The assistant worked it out from what a step "
    "returned, or wrote it itself while carrying the task out — a summary, a note, "
    "a caption about a page, a place it picked itself to keep the results in. The "
    "user never said it and could not supply it, so it "
    "is NOT a parameter: it is a placeholder, and you describe in one line what "
    "belongs in that spot each time the routine runs.\n"
    "   A parameter filling browse's extract argument is a PLAIN-LANGUAGE "
    "instruction naming what to pull from the page (e.g. 'the current price') — "
    "there is no CSS-selector, XPath, or pattern machinery in this system, so never "
    "name or describe one that way.\n"
    "Respond in exactly this shape and nothing else:\n"
    f"{NAME_TAG} <a short generic verb-noun name>\n"
    f"{DESCRIPTION_TAG} <one line: the user intent it serves, then the mechanics>\n"
    "then the group of candidates THE USER GAVE, one line each:\n"
    f"{PARAM_TAG} <current name>: <semantic_name> {_PARAM_DESC_SEPARATOR} <one-line "
    "description>\n"
    "then the group of candidates THE ASSISTANT PRODUCED, one line each:\n"
    f"{PLACEHOLDER_TAG} <current name>: <one-line description of what belongs "
    "there>\n"
    "Sort every candidate into one of those two groups BEFORE you write either group, "
    "then write the groups in that order — every candidate appears in exactly one "
    "group, never both and never neither, repeating its CURRENT name exactly so it "
    "maps back. A group with no candidates in it is left out entirely. Use a single "
    "lowercase word or snake_case for <semantic_name>.\n"
    "Write nothing else — no preamble, no explanation, no restating the routine."
)

# The single per-call ask; the routine + its candidate parameters are the content.
# Fixed, so the caller only supplies the content (the naming contract is a property
# of this customer, not a per-call parameter).
_SKILL_NAMING_INSTRUCTION = (
    "Extract the user's core intent from the conversation, name this routine by that "
    "intent, describe it in one line (intent first, mechanics second), and for every "
    "candidate parameter say whether the user gave that value — naming and describing "
    "it if they did, describing what belongs there if they did not."
)

# ── Third customer: conversation-state classification (#1706) ──────────────────
# The classifier contract is a THIRD enumerated output shape riding the SAME
# poison-screen + reroll machinery (``_draw_clean``): given a small conversation
# slice and a closed list of candidate states (the machine's CURRENT out-edges,
# rendered by :mod:`penny.conversation_machine` — never the global state set),
# emit ONE tagged line naming the state the newest message puts the conversation
# in.  The parse validates MEMBERSHIP, not just shape: a drawn state outside the
# offered union is a contract violation exactly like an untagged draw — rerolled
# once on the unchanged context, then an honest failure the machine treats as
# no-transition (fail → stay; the caller's rule, encoded in
# ``conversation_machine.next_state``).
#
# Some states are SKILL-GATED (#1706 beats 2/5 — apply and request):
# their option lines direct the model to add a second ``SKILL:`` line naming
# which of the listed skills is meant.  Drawing a gated state WITHOUT a valid
# in-set skill line is the same contract violation — reroll, then INVALID — so
# such a decision always carries an actionable skill, never a dangling "use a
# skill" with nothing bound.
STATE_TAG = "STATE:"
SKILL_TAG = "SKILL:"

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
    f"{STATE_TAG} <name>\n"
    "The name must be one of the listed transitions, copied EXACTLY. When the "
    f"chosen transition directs it, add exactly one more line — {SKILL_TAG} <the "
    "skill's name, copied exactly from Known skills> — and nothing more.\n"
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


class ParameterVerdict(StrEnum):
    """Where a candidate parameter's demonstrated value came from (#1770) — the
    closed union the labeller decides per candidate, and the only thing that makes
    an unexplained leaf a parameter rather than a default.

    ``PARAMETER`` = the USER supplied it (verbatim or reworded), so it is a real
    skill input the model rebinds per instantiation.  ``PLACEHOLDER`` = the
    assistant derived it from a step's result or invented it while working, so no
    user could ever supply it — it renders as a placeholder carrying the labeller's
    description, never as the frozen demonstrated value."""

    PARAMETER = "parameter"
    PLACEHOLDER = "placeholder"


class ParameterLabel(BaseModel):
    """One candidate parameter's verdict + label from the naming micro-context
    (#1668/#1770).  ``verdict`` is the decision; ``name`` carries the generic
    semantic name (what the value means, not the tool arg it fills) on
    :attr:`ParameterVerdict.PARAMETER` and is EMPTY on
    :attr:`ParameterVerdict.PLACEHOLDER` (a placeholder is not a parameter, so it
    has no binding key); ``description`` carries the one-line what-to-supply on
    either.  Keyed back to the CURRENT arg-derived name by the parse, so the
    caller's rename/conversion is unambiguous."""

    verdict: ParameterVerdict
    name: str = ""
    description: str = ""


class SkillLabel(BaseModel):
    """The run-end naming micro-context's typed result (#1665/#1668/#1770): a GENERIC
    verb-noun ``name`` + one-line ``description`` for the distilled routine, plus a
    per-candidate verdict+label keyed by the candidate's CURRENT (arg-derived)
    name.  ``name``/``description`` are non-blank by construction (``_parse_label``
    returns ``None`` otherwise, so the caller falls back to the deterministic slug —
    naming never blocks extraction); ``parameters`` may be empty or partial.

    Absence is deliberately NOT a verdict: a candidate with no line keeps its
    current meaning (the arg-derived required parameter), because overloading
    absence to mean "drop it" would let one flaky draw silently delete a real
    parameter — and extraction must never block or degrade on the rewrite."""

    name: str
    description: str
    parameters: dict[str, ParameterLabel] = {}


class StateDrawOutcome(StrEnum):
    """The enumerated outcome of a state-classification draw (#1706) — a closed
    set the machine maps one way each.

    ``INVALID`` covers both contract violations — an untagged draw AND a drawn
    state outside the offered union (the persisted promptlog row holds which) —
    because the machine treats them identically: no transition (fail → stay).
    ``POISON_REROLL_FAILED`` is the transport-artifact escape, same as extract."""

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
        """Deterministic classification of one clean draw by its OPENING tag.

        The tag must open the stripped output (its first line).  ``EXTRACTED:`` →
        the value is EVERYTHING after the tag (the whole remainder, trimmed) — a
        multi-line digest, an item-per-line list, or a single value.
        ``NOT_PRESENT:`` → the reason is the FIRST LINE only, so a not-present
        apology can never be multi-line-promoted into an extracted value.  Anything
        else — no opening tag, or a tag with a blank payload — is ``None``
        (invalid), which the caller rerolls once and then fails honestly.
        """
        text = draw.strip()
        if text.startswith(EXTRACTED_TAG):
            value = text[len(EXTRACTED_TAG) :].strip()
            if not is_blank(value):
                return MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=value)
        if text.startswith(NOT_PRESENT_TAG):
            reason = text[len(NOT_PRESENT_TAG) :].split("\n", 1)[0].strip()
            if not is_blank(reason):
                return MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason=reason)
        return None

    async def label_skill(
        self, content: str, *, run_target: str | None = None
    ) -> SkillLabel | None:
        """Write a GENERIC name + description for a distilled routine AND, per
        candidate parameter, a VERDICT (#1665/#1668/#1770) — the second customer of
        this machinery.  Rides the SAME poison-screen + reroll draw loop as
        ``extract``, with the naming system prompt and its own ledger attribution,
        then a deterministic tag parse (``NAME:`` / ``DESCRIPTION:`` / one ``PARAM``
        or ``PLACEHOLDER`` line per candidate).

        Returns the label, or ``None`` on ANY failure (poison exhausted, or the
        model never produced both the name and description tags) — the caller falls
        back to the deterministic slug, so run-end skill extraction NEVER blocks on
        the rewrite.  Per-candidate verdicts are best-effort: a candidate without a
        valid line is simply absent, which the caller reads as "keep the arg-derived
        required parameter" — absence is never a drop."""
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
        """Deterministic parse of the naming contract — a ``NAME:`` line, a
        ``DESCRIPTION:`` line (each with a non-blank payload), and zero or more
        per-candidate verdict lines (``PARAM``/``PLACEHOLDER``).  Missing the name or
        description (or a blank payload) is a contract violation → ``None`` (the
        caller rerolls once and then falls back), never a partial label.  Verdict
        lines are best-effort — a malformed one is dropped, not fatal."""
        name = _tagged_payload(draw, NAME_TAG)
        description = _tagged_payload(draw, DESCRIPTION_TAG)
        if name is None or description is None:
            return None
        return SkillLabel(name=name, description=description, parameters=_parse_param_labels(draw))

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
        prompt and its own ledger attribution, then a deterministic tag parse
        validated for MEMBERSHIP: a drawn state outside ``allowed`` is a contract
        violation exactly like an untagged draw — one reroll of the unchanged
        context, then an honest ``INVALID`` the machine reads as no-transition.

        ``skill_gated_states`` names the states whose draw must ALSO carry a
        ``SKILL:`` line naming a member of ``skills`` — drawing one with a
        missing or out-of-set skill is the same contract violation, so a gated
        decision always binds an actionable skill.  A stray ``SKILL:`` line on an
        ungated draw is ignored (the decision stands; the line binds nothing)."""
        for _ in range(_UNTAGGED_DRAW_BUDGET):
            draw = await self._draw_clean(
                content,
                "",
                run_target,
                system_prompt=STATE_CLASSIFIER_SYSTEM_PROMPT,
                agent_name=PennyConstants.STATE_CLASSIFIER_AGENT_NAME,
                prompt_type=PennyConstants.STATE_CLASSIFIER_PROMPT_TYPE,
                user_template=_STATE_USER_TEMPLATE,
            )
            if draw is None:
                return StateDraw(outcome=StateDrawOutcome.POISON_REROLL_FAILED)
            decided = self._parse_state_draw(draw, allowed, skill_gated_states, skills)
            if decided is not None:
                return decided
            logger.warning("State-classifier output invalid — one reroll of the unchanged context")
        logger.warning("State-classifier output invalid after reroll — no transition")
        return StateDraw(outcome=StateDrawOutcome.INVALID)

    @staticmethod
    def _parse_state_draw(
        draw: str,
        allowed: Sequence[str],
        skill_gated_states: Sequence[str],
        skills: Sequence[str],
    ) -> StateDraw | None:
        """The tagged state (+ its gated skill), but ONLY when every member is in
        its offered set — no tag, a blank payload, an out-of-union state, or a
        gated state whose ``SKILL:`` line is missing or out-of-set is ``None``
        (invalid), which the caller rerolls once and then fails honestly.  Exact
        match, no normalization: the prompt says copied exactly, and every member
        was shown verbatim."""
        name = _tagged_payload(draw, STATE_TAG)
        if name is None or name not in allowed:
            return None
        if name not in skill_gated_states:
            return StateDraw(outcome=StateDrawOutcome.DECIDED, name=name)
        skill = _tagged_payload(draw, SKILL_TAG)
        if skill is None or skill not in skills:
            return None
        return StateDraw(outcome=StateDrawOutcome.DECIDED, name=name, skill=skill)

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


def _parse_param_labels(draw: str) -> dict[str, ParameterLabel]:
    """Every verdict line parsed into a ``{current_name: ParameterLabel}`` map
    (#1668/#1770) — ``PARAM <current>: <semantic> — <description>`` (the user gave
    it) and ``PLACEHOLDER <current>: <description>`` (the assistant produced it).
    Each line is keyed by the candidate's CURRENT (arg-derived) name so the mapping
    back is unambiguous.

    A malformed line is DROPPED, never coerced: a ``PARAM`` line missing its current
    or semantic name, or a ``PLACEHOLDER`` line missing its current name or its
    description (a placeholder with nothing to say is what the render would have to
    show, so a blank one is no verdict at all).  A candidate named on MORE THAN ONE
    line is a contradictory draw (the prompt asks for exactly one line each) and is
    dropped too — letting the last line win would let a stray trailing
    ``PLACEHOLDER`` delete a real parameter, the exact failure absence-is-not-a-drop
    exists to prevent.  A dropped line is simply an absent verdict, which the caller
    reads as "keep the arg-derived required parameter" — the flaky-draw-safe
    direction."""
    labels: dict[str, ParameterLabel] = {}
    repeated: set[str] = set()
    for line in draw.splitlines():
        parsed = _parse_verdict_line(line.strip())
        if parsed is None:
            continue
        current, label = parsed
        if current in labels:
            repeated.add(current)
        labels[current] = label
    return {name: label for name, label in labels.items() if name not in repeated}


def _parse_verdict_line(line: str) -> tuple[str, ParameterLabel] | None:
    """One candidate's verdict line — a ``PARAM`` line (the user gave it) or a
    ``PLACEHOLDER`` line (the assistant produced it) — or ``None`` for anything else."""
    return _parse_param_line(line) or _parse_placeholder_line(line)


def _parse_param_line(line: str) -> tuple[str, ParameterLabel] | None:
    """``PARAM <current>: <semantic> — <description>`` → the candidate's current name
    and its PARAMETER verdict (the semantic name and description split on the
    em-dash; the description is optional).  ``None`` when the line isn't a ``PARAM``
    line or lacks a current or semantic name."""
    if not line.startswith(f"{PARAM_TAG} "):
        return None
    current, sep, rest = line[len(PARAM_TAG) :].strip().partition(":")
    if not sep:
        return None
    semantic, _, description = rest.partition(_PARAM_DESC_SEPARATOR)
    current, semantic = current.strip(), semantic.strip()
    if is_blank(current) or is_blank(semantic):
        return None
    label = ParameterLabel(
        verdict=ParameterVerdict.PARAMETER, name=semantic, description=description.strip()
    )
    return current, label


def _parse_placeholder_line(line: str) -> tuple[str, ParameterLabel] | None:
    """``PLACEHOLDER <current>: <description>`` → the candidate's current name and its
    PLACEHOLDER verdict.  The description is REQUIRED (it is what the render shows in
    the leaf's place), so a blank one drops the line; ``None`` when the line isn't a
    ``PLACEHOLDER`` line or lacks a current name."""
    if not line.startswith(f"{PLACEHOLDER_TAG} "):
        return None
    current, sep, description = line[len(PLACEHOLDER_TAG) :].strip().partition(":")
    if not sep:
        return None
    current, description = current.strip(), description.strip()
    if is_blank(current) or is_blank(description):
        return None
    return current, ParameterLabel(verdict=ParameterVerdict.PLACEHOLDER, description=description)
