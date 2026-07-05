"""The concrete ``ResponseValidator`` chain for the agentic loop.

Each class here owns exactly one condition from the behaviour taxonomy
(``penny.validation.conditions``) and returns a disposition from
``penny.validation.outcomes`` — the live half of model-I/O validation.  A new
guard is a new validator added to an agent's chain (see ``Agent.response_validators``
/ ``BackgroundAgent.response_validators``), never a new branch in the loop.

Validators are PURE: they read ``(response, ctx)`` and return a disposition,
mutating nothing and reaching into no agent state.  The detection helpers they
need (XML-tag / think-tag / malformed-URL / truncated-URL predicates and the
strong-nudge builder) live here as module functions so the chain has no
dependency back on ``penny.agents`` — keeping this a leaf the loop imports, not
the other way round.

Mapping from the old inline ``_check_response`` branches:

  XML branch              → ``XmlTagValidator``        (Retry, no extra nudge)
  empty branch            → ``EmptyResponseValidator`` (Retry, continue/strong nudge)
  refusal branch          → ``RefusalValidator``       (Retry, no extra nudge)
  hallucinated-URL branch → ``HallucinatedUrlValidator`` (Retry, no extra nudge)
  strip-tool-calls-no-tools → ``HallucinatedToolCallRepair`` (Repair)
  ``handle_text_step``    → ``TextInsteadOfToolValidator`` (NudgeContinue)
  ``handle_premature_terminator`` → ``PrematureDoneValidator`` (RejectToolCall)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from typing import Any

from penny.agents.models import MessageRole
from penny.constants import PennyConstants
from penny.llm.models import LlmResponse
from penny.llm.refusal import is_refusal
from penny.prompts import Prompt
from penny.tools.memory_args import DoneArgs
from penny.tools.memory_tools import DoneTool
from penny.validation.conditions import ConditionKey
from penny.validation.outcomes import (
    LoopContext,
    NudgeContinue,
    Proceed,
    RejectToolCall,
    Repair,
    Retry,
    ValidationOutcome,
)

logger = logging.getLogger(__name__)


# ── Pure text-detection helpers (relocated from agents.base) ─────────────────

# Matches paired XML-like tags in content, e.g. <function=search>...</function>
# or <tools><search>...</search></tools>
_XML_TAG_PATTERN = re.compile(r"<[a-zA-Z]\w*[\s=>].*</[a-zA-Z]\w*>", re.DOTALL)

# Matches <think>...</think> blocks emitted inline by some models (e.g. DeepSeek-R1, Qwen3)
_THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Matches markdown links [text](url) and bare URLs for validation
_MARKDOWN_LINK_URL_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")
_BARE_URL_PATTERN = re.compile(r"(?<!\()(https?://\S+)")


def has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


def strip_think_tags(content: str) -> tuple[str, str | None]:
    """Strip <think>...</think> blocks from content.

    Returns (cleaned_content, extracted_thinking) where extracted_thinking
    contains the concatenated text from all stripped blocks.
    """
    thinking_parts: list[str] = []

    def _collect(match: re.Match) -> str:
        thinking_parts.append(match.group(1).strip())
        return ""

    cleaned = _THINK_TAG_PATTERN.sub(_collect, content).strip()
    extracted = "\n\n".join(thinking_parts) if thinking_parts else None
    return cleaned, extracted


def is_url_truncated(url: str) -> bool:
    """Return True if url appears truncated or malformed.

    Checks for missing host and trailing hyphen (the most common sign of a cut-off path).
    Strips trailing prose punctuation before validation so sentence-ending periods
    don't cause false positives.
    """
    cleaned = url.rstrip(".,;:!?\"')>}]")
    try:
        parsed = urllib.parse.urlparse(cleaned)
    except ValueError:
        return True
    if not parsed.netloc or "." not in parsed.netloc:
        return True
    return cleaned.endswith("-")


def clean_malformed_urls(content: str) -> str:
    """Remove truncated or malformed URLs from model-generated content.

    For markdown links [text](bad_url), the link text is preserved.
    For bare malformed URLs, the URL token is removed entirely.
    Valid URLs are left unchanged.
    """

    def fix_md_link(match: re.Match) -> str:
        text, url = match.group(1), match.group(2)
        if is_url_truncated(url):
            logger.warning("Stripped malformed URL from markdown link: %.120s", url)
            return text
        return match.group(0)

    def fix_bare_url(match: re.Match) -> str:
        url = match.group(1)
        if is_url_truncated(url):
            logger.warning("Stripped malformed bare URL: %.120s", url)
            return ""
        return match.group(0)

    content = _MARKDOWN_LINK_URL_PATTERN.sub(fix_md_link, content)
    content = _BARE_URL_PATTERN.sub(fix_bare_url, content)
    return content


def _extract_urls(text: str) -> list[str]:
    """Extract all URLs from text (both markdown links and bare URLs)."""
    md_urls = [m.group(2) for m in _MARKDOWN_LINK_URL_PATTERN.finditer(text)]
    bare_urls = [m.group(1) for m in _BARE_URL_PATTERN.finditer(text)]
    seen: set[str] = set()
    urls: list[str] = []
    for url in md_urls + bare_urls:
        cleaned = url.rstrip(".,;:!?\"')>}]")
        if cleaned not in seen:
            seen.add(cleaned)
            urls.append(cleaned)
    return urls


def find_hallucinated_urls(text: str, source_text: str) -> list[str]:
    """Return URLs in text that don't appear verbatim in the source text."""
    urls = _extract_urls(text)
    if not urls:
        return []
    return [url for url in urls if url not in source_text]


def build_strong_nudge(messages: list[dict]) -> str:
    """Build a context-aware nudge that includes the original user question.

    Called when many preceding tool calls may have saturated the model's context
    and the model returned empty on the final step (tools stripped).  Using
    forceful language plus the original question breaks the model out of
    search-fixation loops and gives it a clear target after heavy tool use.
    """
    user_messages = [
        m["content"]
        for m in messages
        if m.get("role") == MessageRole.USER and not m["content"].startswith("STOP")
    ]
    original_question = user_messages[-1]
    return Prompt.FINAL_STEP_NUDGE.format(original_question=original_question)


# ── Response-level validators (chat + collector) ─────────────────────────────


class XmlTagValidator:
    """The model wrapped its reply in XML/markup instead of plain prose — retry
    once, re-appending the bad response with no extra nudge (the model usually
    drops the markup on the second pass)."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.XML in ctx.retried:
            return Proceed(response=response)
        if has_xml_tags(response.content.strip()):
            return Retry(condition=ConditionKey.XML, nudge="")
        return Proceed(response=response)


class EmptyResponseValidator:
    """The response carries no substantive content (blank, separators-only, or a
    bare ``<think>`` block with no body) — retry once.

    The nudge depends on whether tools are still available: mid-loop (tools live)
    the ``continue_nudge`` this validator was composed with; on the final step
    (tools stripped) the loop swaps in the forceful ``build_strong_nudge`` carrying
    the original question.  The empty-string sentinel signals "loop, build the
    strong nudge from messages" — the strong builder needs the message history a
    pure validator can't hold.

    ``continue_nudge`` is the mid-loop nudge and is composed per-agent: chat/base
    keep the default ``CONTINUE_NUDGE`` ("Please provide your response."), while
    the collector chain passes ``COLLECTOR_CONTINUE_NUDGE`` — a collector acts only
    through tool calls, so "provide your response" would invite an unparseable prose
    reply; it must be told to make a tool call instead."""

    def __init__(self, continue_nudge: str = Prompt.CONTINUE_NUDGE) -> None:
        self._continue_nudge = continue_nudge

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.EMPTY in ctx.retried:
            return Proceed(response=response)
        effective_content, _ = strip_think_tags(response.content.strip())
        letter_count = sum(1 for character in effective_content if character.isalpha())
        if letter_count >= PennyConstants.MIN_RESPONSE_LETTERS:
            return Proceed(response=response)
        nudge = self._continue_nudge if ctx.tools_available else ""
        return Retry(condition=ConditionKey.EMPTY, nudge=nudge)


class RefusalValidator:
    """The response is a model refusal ("I'm sorry, I can't…") rather than a real
    answer — retry once, re-appending the response with no extra nudge."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.REFUSAL in ctx.retried:
            return Proceed(response=response)
        effective_content, _ = strip_think_tags(response.content.strip())
        if effective_content and is_refusal(effective_content):
            return Retry(condition=ConditionKey.REFUSAL, nudge="")
        return Proceed(response=response)


class HallucinatedUrlValidator:
    """The response cites a URL that never appeared in the source material
    (``ctx.source_text``: tool results + system prompt + history) — retry once so
    the model answers from real sources.  No source text → nothing to check."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.HALLUCINATED_URLS in ctx.retried:
            return Proceed(response=response)
        effective_content, _ = strip_think_tags(response.content.strip())
        if not (ctx.source_text and effective_content):
            return Proceed(response=response)
        bad_urls = find_hallucinated_urls(effective_content, ctx.source_text)
        if bad_urls:
            logger.warning(
                "Hallucinated URL(s): %s",
                ", ".join(url[:80] for url in bad_urls),
            )
            return Retry(condition=ConditionKey.HALLUCINATED_URLS, nudge="")
        return Proceed(response=response)


class HallucinatedToolCallRepair:
    """The model emitted tool calls when no tools are available (final step,
    tools stripped) — strip them in place and let content fall through to the
    rest of the chain, which triggers the empty-content retry/nudge.  A silent
    ``Repair``, never a re-call."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ctx.tools_available or not response.has_tool_calls:
            return Proceed(response=response)
        logger.warning("Model hallucinated tool calls without tools — stripping")
        repaired = response.model_copy(deep=True)
        repaired.message.tool_calls = None
        return Repair(response=repaired)


# ── Collector-only run-shape validators ──────────────────────────────────────

# The ``done()`` argument schema — the exact key set a bare-JSON text bail must
# carry to be an unambiguous ``done`` call.  Derived from the model so it tracks
# the schema (``{success, summary}``); ``reasoning`` is the one extra key gpt-oss's
# native bail shape adds (it is NOT a ``DoneArgs`` field, but it never disambiguates
# away from ``done``, so it is tolerated for the match).
_DONE_REQUIRED_KEYS = frozenset(DoneArgs.model_fields)
_DONE_TOLERATED_KEYS = _DONE_REQUIRED_KEYS | {"reasoning"}
# The full-envelope variant the model also emits: ``{"name": "done", "arguments": {…}}``.
_ENVELOPE_KEYS = frozenset({"name", "arguments"})


def _done_args_from(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """The ``{success, summary}`` payload from a candidate args dict, or ``None``
    unless it carries exactly the ``done`` schema — ``success`` (bool) + ``summary``
    (str), plus at most a tolerated ``reasoning``.  Any other key makes it
    ambiguous, so it is left for the generic text-bail nudge."""
    keys = set(candidate)
    if not _DONE_REQUIRED_KEYS <= keys <= _DONE_TOLERATED_KEYS:
        return None
    if not (isinstance(candidate["success"], bool) and isinstance(candidate["summary"], str)):
        return None
    return {"success": candidate["success"], "summary": candidate["summary"]}


def parse_done_json_bail(content: str) -> dict[str, Any] | None:
    """Parse a plain-text response that is really a ``done()`` call the model failed
    to route through the tool channel, returning the ``done`` arguments it carried
    (``{success, summary}``) — or ``None`` when it is not an unambiguous done bail.

    Two shapes, both convergent in production on Harmony backends (the function
    name rides a header that gets lost):

      - bare args     ``{"success": true, "summary": "…", "reasoning": "…"}``
      - full envelope ``{"name": "done", "arguments": {<bare args>}}``

    ``reasoning`` (not a ``DoneArgs`` field) is tolerated; ANY other extra key, a
    non-``done`` envelope name, or non-JSON text yields ``None``."""
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if set(parsed) == _ENVELOPE_KEYS:
        if parsed.get("name") != DoneTool.name or not isinstance(parsed.get("arguments"), dict):
            return None
        parsed = parsed["arguments"]
    return _done_args_from(parsed)


class DoneJsonBailValidator:
    """A collector emitted the ``done()`` terminator's *arguments* as a plain JSON
    text object instead of a tool call — gpt-oss's native fallback shape on Harmony
    backends (the dominant call-shaped text bail in production).  Reject and TEACH:
    append the stray text plus the shape-specific ``COLLECTOR_DONE_JSON_NUDGE`` —
    naming exactly what the model did (wrote ``done``'s arguments as text) and the
    exact next move (make the real ``done(success=…, summary=…)`` tool call) — and
    continue the loop so the model itself re-emits the call.

    Deliberately NOT a ``Repair``: fabricating a tool call the model never made
    would coerce a malformed emission into a healthy one.  Repair is reserved for
    well-formed calls that transport/parsing mangled (e.g. the Harmony token
    strip); anything the *model* got wrong gets a teaching response it can learn
    from within the run.  The value over the generic text-bail nudge is
    specificity — the model is told precisely which tool to call and why its
    output didn't count, rather than "make a tool call".

    Collector-only by composition (``BackgroundAgent.run_shape_validators`` — chat
    has no ``done`` tool), ordered BEFORE ``TextInsteadOfToolValidator`` so the
    specific teaching outranks the generic nudge.  Unambiguous by construction:
    only the ``{success, summary}`` key set (optionally a tolerated ``reasoning``)
    — unique to ``done`` among the collector tools — or the
    ``{"name": "done", "arguments": {…}}`` envelope; any other shape falls through
    to the generic nudge.  Surfaced via a WARNING naming ``DONE_JSON_BAIL``.
    Bounded by ``max_steps`` exactly like the generic text bail: on the final step
    there's no retry room, so the cycle ends without a ``done()`` and re-runs next
    tick."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ctx.is_final_step or response.has_tool_calls:
            return Proceed(response=response)
        if parse_done_json_bail(response.content) is None:
            return Proceed(response=response)
        logger.warning(
            "done() arguments emitted as JSON text (%s) — teaching the real tool call",
            ConditionKey.DONE_JSON_BAIL,
        )
        return NudgeContinue(message=Prompt.COLLECTOR_DONE_JSON_NUDGE)


class TextInsteadOfToolValidator:
    """A collector narrated prose where a tool call was required.

    A collector acts only through tool calls (``done()`` to finish, otherwise the
    next work tool); a text-only response is a bail that would otherwise be read
    as the final answer, leaving the cycle with no ``done`` record (marked failed,
    cursor uncommitted, re-run next tick).  Since the slip is stochastic, append
    the stray text plus ``COLLECTOR_TOOL_CALL_NUDGE`` and keep the loop going so
    the model recovers with a real tool call.  Bounded by ``max_steps``: on the
    final step there's no room to retry, so the cycle ends without a ``done()``
    and re-runs next tick — a clean reject, not a salvage of the malformed
    output (the cursor stays uncommitted until a real ``done()`` lands)."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ctx.is_final_step or response.has_tool_calls:
            return Proceed(response=response)
        return NudgeContinue(message=Prompt.COLLECTOR_TOOL_CALL_NUDGE)


class PrematureDoneValidator:
    """A collector whose very first tool call is ``done()`` — with no prior read /
    write / browse — is the ``⚠ NO WORK DONE`` bail: the model declared the cycle
    finished without even checking its inputs.

    The model made a *coherent* tool call, so the correction is an error tool
    response (not a text-step nudge): the loop appends a failed tool-result for
    the ``done`` call(s) and continues — a failed ``done`` doesn't stop the loop,
    so the model sees the error and retries with a real tool call first.  Premature
    only when (a) this response's calls are all ``done`` and (b) no non-``done``
    call has run yet this cycle (``ctx.records``).  A batched ``[log_read, done]``
    or a ``done`` after any read is honoured.  Bounded by ``max_steps``: on the
    final step the done closes the cycle."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ctx.is_final_step or not response.has_tool_calls:
            return Proceed(response=response)
        calls = response.message.tool_calls or []
        if any(call.function.name != DoneTool.name for call in calls):
            return Proceed(response=response)
        if any(record.tool != DoneTool.name for record in ctx.records):
            return Proceed(response=response)
        return RejectToolCall(message=Prompt.COLLECTOR_PREMATURE_DONE_REJECTION)
