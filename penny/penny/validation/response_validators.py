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
``EmptyResponseValidator``) are gone with their nudges; their detectors live in
``penny.text_validity`` beside the other invalid-draw predicates.  The one append that
remains rides ``Retry``'s ``note`` (#2698): the hallucinated-URL retry shows the model
its draft plus a note naming the URL(s) with no source.
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
from penny.text_validity import strip_think_tags
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


# The note the hallucinated-URL retry shows the model (#2698): its draft plus the URL(s)
# with no source, named, so the redraw can rewrite without them or read a page that has
# them instead of re-drawing blind.  When a flagged URL reads as a MISCOPY of one the
# source text really holds (row 138: a path word dropped out of a CBC livelstory URL),
# the note hands her the exact source URL instead — use it verbatim or drop the link.
_URL_RETRY_NOTE_ONE = (
    "Your reply links {urls}, which isn't from any page you read or anything said in "
    "this conversation. Rewrite the reply without links you can't source, or read a "
    "page that has it."
)
_URL_RETRY_NOTE_MANY = (
    "Your reply links {urls}, which aren't from any page you read or anything said in "
    "this conversation. Rewrite the reply without links you can't source, or read "
    "pages that have them."
)
_URL_RETRY_NOTE_NEAR_MATCH = (
    "Your reply links {bad}, which isn't in this conversation. The closest link here "
    "is {source}: use it exactly, or drop the link."
)
# A source path is a miscopy's match when their edit distance is under this share of
# the longer path — row 138's dropped word was ~15%.
_NEAR_MATCH_PATH_SHARE = 0.2


def _edit_distance(a: str, b: str) -> int:
    """The Levenshtein distance of two strings — URL paths are short, so the plain
    two-row DP table is fine."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _nearest_source_url(bad: str, source_text: str) -> str | None:
    """The source URL the flagged address reads as a miscopy of, or None.

    A match is on the same host, with a path that shares the bad path's last segment
    or sits within ``_NEAR_MATCH_PATH_SHARE`` edit distance of it; the closest wins,
    a tie going to the first in source order.  Never keyed to a host or a path shape —
    whatever URL a draft miscopies is matched the same way."""
    bad_parsed = urllib.parse.urlparse(bad)
    bad_path = bad_parsed.path
    best: tuple[int, str] | None = None
    for source in _extract_urls(source_text):
        parsed = urllib.parse.urlparse(source)
        if parsed.netloc != bad_parsed.netloc:
            continue
        path = parsed.path
        if path.rsplit("/", 1)[-1] == bad_path.rsplit("/", 1)[-1]:
            distance = 0
        else:
            distance = _edit_distance(bad_path, path)
        if distance < _NEAR_MATCH_PATH_SHARE * max(len(bad_path), len(path), 1) and (
            best is None or distance < best[0]
        ):
            best = (distance, source)
    return best[1] if best else None


def url_retry_note(bad_urls: list[str], source_text: str = "") -> str:
    """The retry note for one or more URLs found in a draft with no source.

    Without a near match the wording names the defect alone; when the source text
    holds a same-host URL the flagged address reads as a miscopy of, that URL's
    sentence hands her the exact link to use instead.  Mixed flagged sets get one
    sentence per URL, so any mix reads cleanly."""
    if not any(_nearest_source_url(url, source_text) for url in bad_urls):
        urls = " and ".join(bad_urls)
        template = _URL_RETRY_NOTE_ONE if len(bad_urls) == 1 else _URL_RETRY_NOTE_MANY
        return template.format(urls=urls)
    sentences = []
    for url in bad_urls:
        match = _nearest_source_url(url, source_text)
        if match:
            sentences.append(_URL_RETRY_NOTE_NEAR_MATCH.format(bad=url, source=match))
        else:
            sentences.append(_URL_RETRY_NOTE_ONE.format(urls=url))
    return " ".join(sentences)


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

    Unlike markup and a refusal, the model cannot usually see the defect in its own
    draft — the URL reads as real to it — so the retry is not blind: it carries
    ``Retry.note``, and the loop shows the draft plus a note naming the URL(s) with
    no source (#2698).  From the note the model may rewrite without them or read a
    page that has them.  When a flagged URL is a close miscopy of a URL the source
    text really holds — a dropped path word (#2698 row 138) — the note hands her the
    exact source URL to use verbatim instead.  The note also names the URL inside
    the conversation, so the run's later draws read it as sourced — accepted: the
    note is the true record of what was flagged."""

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
            return Retry(
                condition=ConditionKey.HALLUCINATED_URLS,
                note=url_retry_note(bad_urls, ctx.source_text),
            )
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
#
# The narrate-from-the-record guard that used to live here (`_RecordNarrationValidator`
# and its two subclasses, `WritesLandedValidator` / `SkillNarrationValidator`) is gone
# (#2455 naive baseline / #2473): what a chat run wrote is now delivered straight onto
# the tool result the moment its call lands (see `ChatAgent.after_tool_step`), and
# there is no skill machinery left to narrate — so `ChatAgent.run_shape_validators` is
# empty and nothing here needs a text-branch guard.


# ── Collector-only run-shape validator ───────────────────────────────────────
