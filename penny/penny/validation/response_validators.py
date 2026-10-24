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
  hallucinated-URL branch → ``HallucinatedUrlValidator`` (Retry once, then Repair)
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
from typing import NamedTuple

from penny.llm.models import LlmResponse
from penny.llm.refusal import is_refusal
from penny.responses import PennyResponse
from penny.text_validity import strip_think_tags
from penny.tools.link_handles import HandleTable
from penny.validation.conditions import ConditionKey
from penny.validation.outcomes import (
    LoopContext,
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


def _drop_unsourced_urls(content: str, bad_urls: list[str]) -> str:
    """``content`` with each of ``bad_urls`` removed — a markdown link's target
    is dropped with its text kept; a bare address drops outright, the sentence
    around it left as it was.  Same shape as :func:`clean_malformed_urls`, keyed
    to "still unsourced after the retry" rather than "truncated" (#2721)."""
    targets = set(bad_urls)

    def fix_md_link(match: re.Match) -> str:
        text, url = match.group(1), match.group(2)
        return text if _address(url) in targets else match.group(0)

    content = _MARKDOWN_LINK_URL_PATTERN.sub(fix_md_link, content)
    for address in reversed(_addresses(content)):
        url = content[address.start : address.end]
        if url in targets:
            content = f"{content[: address.start]}{content[address.end :]}"
    return content


def _drop_unsourced_link(
    response: LlmResponse, handles: HandleTable, dangling: list[int], bad_urls: list[str]
) -> LlmResponse:
    """The repair once the retry is spent and the second draft still carries the
    defect (#2721): every still-bad link removed — a dangling handle citation
    AND a raw unsourced URL can both be present, so both are dropped, never
    just the first one found — and a plain line telling the user one was left
    out, because it never reaches them silently (visible degradation over
    silent success, docs/principles.md).  The citation grammar is
    ``HandleTable``'s own (``drop_citations``, beside its ``dangling()``); only
    the raw-URL removal is this module's, since a plain unsourced address is
    not a handle concern.  Logged the same way the chain's other repair
    (``HallucinatedToolCallRepair``) records what it did."""
    content = response.message.content
    if dangling:
        content = handles.drop_citations(content, dangling)
        logger.warning("Dropped dangling link handle(s) after retry: %s", dangling)
    if bad_urls:
        content = _drop_unsourced_urls(content, bad_urls)
        logger.warning(
            "Dropped unsourced URL(s) after retry: %s",
            ", ".join(url[:80] for url in bad_urls),
        )
    repaired = response.model_copy(deep=True)
    repaired.message.content = f"{content}\n\n{PennyResponse.LINK_DROPPED_NOTICE}"
    return repaired


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
    the model answers from real sources.  No source text → nothing to check.

    The handle branch beside it (#2717): a link citation ``[n]`` the run's table
    does not hold is rejected with a note naming which links DO exist — the model
    cannot usually see a miscounted handle in its own draft (the number reads as
    real to it), so the re-draw is shown the draft plus the note.  Links that came
    through handles never reach the raw-URL check: the model cites ``[n]``, not an
    address, so the raw-URL branch only fires for a URL still typed outright.

    The retry is spent after one attempt (#2721): when the SECOND draft still
    carries a dangling handle or an unsourced URL, it no longer reaches the user
    silently — the still-bad link is dropped (a ``Repair``, not another re-draw)
    and the message says plainly that one was left out."""

    def check(self, response: LlmResponse, ctx: LoopContext) -> ValidationOutcome:
        effective_content, _ = strip_think_tags(response.content.strip())
        if not (ctx.source_text and effective_content):
            return Proceed(response=response)
        # Both are computed every time — a draft can carry a dangling handle AND a
        # separately typed raw URL at once, and skipping one check because the other
        # already fired would let that second defect through unflagged (#2721 review).
        dangling = ctx.handles.dangling(effective_content) if ctx.handles else []
        bad_urls = find_hallucinated_urls(effective_content, ctx.source_text)
        if not dangling and not bad_urls:
            return Proceed(response=response)
        if ConditionKey.HALLUCINATED_URLS not in ctx.retried:
            if dangling:
                note = ctx.handles.dangling_note(dangling)
                logger.warning("Dangling link handle(s): %s", note)
                return Retry(condition=ConditionKey.HALLUCINATED_URLS, note=note)
            logger.warning(
                "Hallucinated URL(s): %s",
                ", ".join(url[:80] for url in bad_urls),
            )
            return Retry(condition=ConditionKey.HALLUCINATED_URLS)
        return Repair(response=_drop_unsourced_link(response, ctx.handles, dangling, bad_urls))


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
#
# The narrate-from-the-record guard that used to live here (`_RecordNarrationValidator`
# and its two subclasses, `WritesLandedValidator` / `SkillNarrationValidator`) is gone
# (#2455 naive baseline / #2473): what a chat run wrote is now delivered straight onto
# the tool result the moment its call lands (see `ChatAgent.after_tool_step`), and
# there is no skill machinery left to narrate — so `ChatAgent.run_shape_validators` is
# empty and nothing here needs a text-branch guard.


# ── Collector-only run-shape validator ───────────────────────────────────────
