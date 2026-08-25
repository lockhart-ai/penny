"""The concrete ``ResponseValidator`` chain for the agentic loop.

Each class here owns exactly one condition from the behaviour taxonomy
(``penny.validation.conditions``) and returns a disposition from
``penny.validation.outcomes`` — the live half of model-I/O validation.  A new
guard is a new validator added to an agent's chain (see ``Agent.response_validators``
/ ``BackgroundAgent.response_validators``), never a new branch in the loop.

Validators are PURE: they read ``(response, ctx)`` and return a disposition,
mutating nothing and reaching into no agent state.  The detection helpers they
need (XML-tag / malformed-URL / truncated-URL predicates) live here as module
functions so the chain has no dependency back on ``penny.agents`` — keeping this a
leaf the loop imports, not the other way round.

Mapping from the old inline ``_check_response`` branches:

  XML branch              → ``XmlTagValidator``        (Retry)
  refusal branch          → ``RefusalValidator``       (Retry)
  hallucinated-URL branch → ``HallucinatedUrlValidator`` (Retry)
  strip-tool-calls-no-tools → ``HallucinatedToolCallRepair`` (Repair)

**The call-shaped-text family is NOT here (#1839), and neither is the EMPTY draw
(#1937).** A draw that was meant to be a tool call and is not one — a collector's
prose or done-as-JSON-text, a chat reply that is a serialized call — and a draw that
says nothing at all are INVALID DRAWS, not recoverable turns: the agent loop discards
them and re-rolls the unchanged context (``Agent._unusable_output_condition`` +
``invalid_draw_conditions``), so they never enter the conversation at all.  The
validators that used to append those outputs plus a teaching nudge
(``TextInsteadOfToolValidator`` / ``DoneJsonBailValidator`` / ``CallAsTextValidator`` /
``EmptyResponseValidator``) are gone with their nudges — and with them the whole
user-turn append, since no condition left carries one; their detectors live in
``penny.text_validity`` beside the other invalid-draw predicates.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from abc import ABC, abstractmethod

from penny.llm.models import LlmResponse
from penny.llm.refusal import is_refusal
from penny.text_validity import strip_think_tags
from penny.validation.conditions import ConditionKey
from penny.validation.outcomes import (
    LoopContext,
    NudgeContinue,
    Proceed,
    Repair,
    Retry,
    ValidationOutcome,
)

logger = logging.getLogger(__name__)


# ── Pure text-detection helpers (relocated from agents.base) ─────────────────

# Matches paired XML-like tags in content, e.g. <function=search>...</function>
# or <tools><search>...</search></tools>
_XML_TAG_PATTERN = re.compile(r"<[a-zA-Z]\w*[\s=>].*</[a-zA-Z]\w*>", re.DOTALL)

# Matches markdown links [text](url) and bare URLs for validation
_MARKDOWN_LINK_URL_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")
_BARE_URL_PATTERN = re.compile(r"(?<!\()(https?://\S+)")


def has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


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


# ── Response-level validators (chat + collector) ─────────────────────────────


class XmlTagValidator:
    """The model wrapped its reply in XML/markup instead of plain prose — retry
    once, re-appending the bad response (the model usually drops the markup on the
    second pass)."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.XML in ctx.retried:
            return Proceed(response=response)
        if has_xml_tags(response.content.strip()):
            return Retry(condition=ConditionKey.XML)
        return Proceed(response=response)


class RefusalValidator:
    """The response is a model refusal ("I'm sorry, I can't…") rather than a real
    answer — retry once, re-appending the response."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.REFUSAL in ctx.retried:
            return Proceed(response=response)
        effective_content, _ = strip_think_tags(response.content.strip())
        if effective_content and is_refusal(effective_content):
            return Retry(condition=ConditionKey.REFUSAL)
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
            return Retry(condition=ConditionKey.HALLUCINATED_URLS)
        return Proceed(response=response)


class HallucinatedToolCallRepair:
    """The model emitted tool calls when no tools are available (final step,
    tools stripped) — strip them in place and let content fall through to the
    rest of the chain.  A silent ``Repair``, never a re-call.

    What is left after the strip is usually nothing, and since #1937 nothing in this
    chain answers that: the loop's honest empty-content close
    (``_empty_content_fallback``) states that the run said nothing, rather than a nudge
    turn asking it to try again."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ctx.tools_available or not response.has_tool_calls:
            return Proceed(response=response)
        logger.warning("Model hallucinated tool calls without tools — stripping")
        repaired = response.model_copy(deep=True)
        repaired.message.tool_calls = None
        return Repair(response=repaired)


# ── Chat-only run-shape validators ───────────────────────────────────────────


class _RecordNarrationValidator(ABC):
    """The shared shape of the narrate-from-the-RECORD guards: a chat run that changed
    something durable tells the user about it from what the framework stamped, never from
    memory (SAID==DID).

    The framework does the work deterministically at the text-branch prep
    (``ChatAgent._prepare_text_shape``) and stamps a rendered frame on the ctx; a
    subclass names which frame it reads.  Turning that into a ``NudgeContinue`` — a
    validator in the chat chain, not a branch in the loop — makes the model re-reply
    against the render.  A frame is stamped at most once per run (the prep runs once per
    run id), so the re-reply falls through to the real final answer instead of narrating
    twice.

    On the final step there is no room to continue (tools stripped, the loop would
    exhaust), so it Proceeds — what happened has already happened durably, and it surfaces
    ambiently in the next turn's self-state header."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ctx.is_final_step or response.has_tool_calls:
            return Proceed(response=response)
        frame = self._frame(ctx)
        if frame:
            logger.info("Narrating %s this turn", self._narrating())
            return NudgeContinue(message=frame)
        return Proceed(response=response)

    @abstractmethod
    def _frame(self, ctx: LoopContext) -> str | None:
        """The rendered record this guard narrates, or ``None`` when there is none."""

    @abstractmethod
    def _narrating(self) -> str:
        """What the log line says this run is narrating."""


class SkillNarrationValidator(_RecordNarrationValidator):
    """A chat run that just AUTO-EXTRACTED a skill narrates it in the same turn
    (#1658, SAID==DID) — from the render (its name, what it's for, what it needs)
    rather than from memory."""

    def _frame(self, ctx: LoopContext) -> str | None:
        return ctx.learned_skill_frame

    def _narrating(self) -> str:
        return "an auto-extracted skill"


class AppliedConfigurationValidator(_RecordNarrationValidator):
    """A chat run that just CONFIGURED the round's routine narrates it in the same turn
    (#1869) — the applied-configuration sibling of the skill-learned frame above.

    It exists because that turn no longer supplies the routine or the values it is
    pointed at: the round settled both and the framework supplied them at the call, so
    what is running is a record to READ.  The frame carries that record — cadence, end
    condition, notify, what it watches — and a turn narrating from anything else would be
    stating a configuration it did not make."""

    def _frame(self, ctx: LoopContext) -> str | None:
        return ctx.applied_configuration_frame

    def _narrating(self) -> str:
        return "the configuration this turn applied"


# ── Collector-only run-shape validator ───────────────────────────────────────
