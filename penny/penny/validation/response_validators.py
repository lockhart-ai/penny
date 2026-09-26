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
import string
import unicodedata
import urllib.parse
from abc import ABC, abstractmethod
from typing import NamedTuple

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

# Matches a markdown link [text](url), so a malformed target can be dropped with its text kept
_MARKDOWN_LINK_URL_PATTERN = re.compile(r"\[([^\]]*)\]\((https?://[^)]*)\)")

# ── Reading a URL as the ADDRESS alone ──
#
# A URL in a reply is read as the address and nothing beside it, then compared against the
# source text.  A reply wraps the URLs it cites — gpt-oss's citation bracket
# (`【browse: https://x】`), markdown bold (`**https://x**`), a link whose text is its own URL
# (`[https://x](https://x)`) — and a wrapper read as part of the URL makes a URL the turn really
# fetched look invented (#2159).
#
# So an address is bounded by the characters a URL may CONTAIN, not by a list of wrappers:
# RFC 3986's closed ASCII set, plus the letters, digits and combining marks of a non-ASCII path
# (`/wiki/Straße`).  Every other character ends the address — `】` `」` `）` `。`, a quote, an
# angle bracket, a space, and every mark nobody has drawn yet.
_URL_SCHEME_PATTERN = re.compile(r"https?://")
_URI_CHARS = frozenset(f"{string.ascii_letters}{string.digits}-._~%!$&'()*+,;=:/?#@[]")
_IRI_CATEGORIES = frozenset("LNM")
# Markdown puts a link's text against its target with `](`, and every character of that seam is
# a legal URI character, so a run is split there and each side is read as its own URL.
_LINK_SEAM = "]("
# Marks that are legal in a URI but, at its END, belong to the prose: nobody's address ends in a
# comma, and a trailing `**` / `__` closes the emphasis the reply wrapped it in.
_PROSE_TAIL_MARKS = ".,;:!?*_"
# Closers that are legal in a path, so a trailing one is decided by PAIRING: `…/Foo_(bar)` closes
# what the address opened, `(…/page)` closes what the sentence opened.  A mark that is its own
# partner pairs by parity.
_PAIRS = {")": "(", "]": "[", "'": "'"}


class _Address(NamedTuple):
    """Where one URL's address sits in a text: ``text[start:end]``."""

    start: int
    end: int


def _is_url_char(char: str) -> bool:
    """Whether ``char`` can be part of a URL's address."""
    if char.isascii():
        return char in _URI_CHARS
    return unicodedata.category(char)[0] in _IRI_CATEGORIES


def _run_end(text: str, start: int) -> int:
    """Where the run of URL characters beginning at ``start`` stops."""
    end = start
    while end < len(text) and _is_url_char(text[end]):
        end += 1
    return end


def _closes_the_address(url: str, opener: str) -> bool:
    """Whether the URL's last character closes something the URL itself opened."""
    if opener == url[-1]:
        return url.count(opener) % 2 == 0
    return url.count(opener) >= url.count(url[-1])


def _runs_past_the_address(url: str) -> bool:
    """Whether the URL's last character belongs to the prose rather than to the address."""
    last = url[-1]
    if last in _PROSE_TAIL_MARKS:
        return True
    opener = _PAIRS.get(last)
    return opener is not None and not _closes_the_address(url, opener)


def _url_only(token: str) -> str:
    """The address without the prose marks it ran into at its end."""
    while token and _runs_past_the_address(token):
        token = token[:-1]
    return token


def _is_address(url: str) -> bool:
    """Whether a link side is a URL with something after its scheme — a relative target
    (`/page`) or a bare `https://` said in prose is not an address to check."""
    scheme = _URL_SCHEME_PATTERN.match(url)
    return scheme is not None and scheme.end() < len(url)


def _run_addresses(text: str, start: int, end: int) -> list[_Address]:
    """The addresses in one run of URL characters: one for a bare URL, one per side of a
    markdown link whose sides are URLs."""
    addresses: list[_Address] = []
    side_start = start
    for side in text[start:end].split(_LINK_SEAM):
        url = _url_only(side)
        if _is_address(url):
            addresses.append(_Address(side_start, side_start + len(url)))
        side_start += len(side) + len(_LINK_SEAM)
    return addresses


def _addresses(text: str) -> list[_Address]:
    """Every URL in ``text``, each as the span of its address alone, in order.  A scheme inside
    an address already read (a redirect's `?u=https://…`, a link's second side) is part of it."""
    addresses: list[_Address] = []
    for scheme in _URL_SCHEME_PATTERN.finditer(text):
        if addresses and scheme.start() < addresses[-1].end:
            continue
        addresses += _run_addresses(text, scheme.start(), _run_end(text, scheme.start()))
    return addresses


def _address(url: str) -> str:
    """The first address in ``url`` — the URL alone, however it was wrapped."""
    addresses = _addresses(url)
    if not addresses:
        return url
    return url[addresses[0].start : addresses[0].end]


def has_xml_tags(content: str) -> bool:
    """Return True if content contains XML-like tag pairs."""
    return bool(_XML_TAG_PATTERN.search(content))


def is_url_truncated(url: str) -> bool:
    """Return True if url appears truncated or malformed.

    Checks for missing host and trailing hyphen (the most common sign of a cut-off path),
    on the address alone — so a sentence's full stop or a wrapper around it doesn't hide
    a truncated URL or fake one.
    """
    cleaned = _address(url)
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
    For bare malformed URLs, the address is removed and whatever wrapped it stays.
    Valid URLs are left unchanged.
    """

    def fix_md_link(match: re.Match) -> str:
        text, url = match.group(1), match.group(2)
        if is_url_truncated(url):
            logger.warning("Stripped malformed URL from markdown link: %.120s", url)
            return text
        return match.group(0)

    content = _MARKDOWN_LINK_URL_PATTERN.sub(fix_md_link, content)
    return _strip_truncated_addresses(content)


def _strip_truncated_addresses(content: str) -> str:
    """Remove every bare address that reads as truncated, last first so spans stay valid."""
    for address in reversed(_addresses(content)):
        url = content[address.start : address.end]
        if is_url_truncated(url):
            logger.warning("Stripped malformed bare URL: %.120s", url)
            content = f"{content[: address.start]}{content[address.end :]}"
    return content


def _extract_urls(text: str) -> list[str]:
    """Every URL in text, each read as its address alone, without repeats."""
    urls = [text[address.start : address.end] for address in _addresses(text)]
    return list(dict.fromkeys(urls))


def find_hallucinated_urls(text: str, source_text: str) -> list[str]:
    """Return URLs in text that don't appear verbatim in the source text."""
    urls = _extract_urls(text)
    if not urls:
        return []
    return [url for url in urls if url not in source_text]


# ── Response-level validators (chat + collector) ─────────────────────────────


class XmlTagValidator:
    """The model wrapped its reply in XML/markup instead of plain prose — retry
    once, re-drawing from the unchanged conversation (a fresh draw usually comes
    back without the markup)."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        if ConditionKey.XML in ctx.retried:
            return Proceed(response=response)
        if has_xml_tags(response.content.strip()):
            return Retry(condition=ConditionKey.XML)
        return Proceed(response=response)


class RefusalValidator:
    """The response is a model refusal ("I'm sorry, I can't…") rather than a real
    answer — retry once, re-drawing from the unchanged conversation."""

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
    against the render.  The prep stamps ONE frame per text draw and never the same one
    twice (a run's records are computed once and handed out in a declared order), so a
    turn that has several records to narrate narrates each of them exactly once and the
    draw after the last falls through to the real final answer.

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


class WritesLandedValidator(_RecordNarrationValidator):
    """A chat run that WROTE entries narrates what actually landed (#1946) — the third
    sibling of the two frames above, and the plainest of them.

    A turn's own account of its writes is a count of what it ATTEMPTED: a draw the reroll
    guard discarded never happened, a write the change-gate refused never landed, and
    both look like writes from inside the run.  The ledger's entry stamps say which ones
    survived, so the frame carries that and the reply states it rather than adding up its
    own intentions."""

    def _frame(self, ctx: LoopContext) -> str | None:
        return ctx.writes_landed_frame

    def _narrating(self) -> str:
        return "what this turn wrote"


# ── Collector-only run-shape validator ───────────────────────────────────────
