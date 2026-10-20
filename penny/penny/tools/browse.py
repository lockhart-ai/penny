"""BrowseTool — searches and reads web pages via the browser extension.

The model packs everything into a single queries array; the tool detects URLs
and reads them directly, while plain text is converted to search URLs.
Queries run in batches of ``max_calls`` fetched in parallel, one batch after
another, up to the per-call total ceiling.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from penny.constants import PageReading, PennyConstants, ProgressEmoji
from penny.database.memory import LogEntryInput
from penny.llm.call_link import CallLink
from penny.llm.embeddings import serialize_embedding
from penny.llm.similarity import embed_text
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.content_cleaning import clean_browser_content
from penny.tools.micro_context import MicroContext, MicroContextResult, MicroExtractOutcome
from penny.tools.models import BrowseArgs, BrowsePage, ToolResult

if TYPE_CHECKING:
    from penny.channels.permission_manager import PermissionManager
    from penny.database import Database
    from penny.database.models import MemoryEntry
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)


class BrowseChannelUnavailableError(Exception):
    """The whole browse *channel* is down for this cycle — no browser is connected
    (or it disconnected and stayed down through every retry) — so every read and
    search in the batch is doomed and retrying URL variants can't help.

    Distinct from a per-page read failure (a single bad/blocked/slow URL), which
    stays a plain exception the model should recover from by trying another source.
    Deliberately NOT a ``ConnectionError`` subclass: ``_read_page``'s per-request
    retry loop catches ``ConnectionError`` to ride out a *transient* disconnect, so
    the terminal, retries-exhausted outage must be a different type — both to skip
    that loop and to let ``_assemble_sections`` tell a channel outage apart from a
    page failure by ``isinstance``, not by string-matching the message.
    """


_URL_PATTERN = re.compile(r"^https?://")

_LINK_RE = re.compile(r"^\s*\[([^\]]*)\]\((https?://(?:[^)\\]|\\.)*)\)\s*$")

# First-person narration of a browse call — the #1480 per-tool override of the
# seam's generic RESULT_NARRATION_*.  The seam prepends ONE line for the WHOLE call
# and appends the `(browse result)` tag itself; the per-page `## browse ...:` /
# `## browse search:` / `## browse error:` section headers stay in the body and
# disambiguate each page.  So these summarise the batch as a single natural clause
# reflecting search-vs-read (a URL query is a direct read; plain text is a search)
# and the outcome (`result.success` is False only when EVERY query errored).
# Composed by ``_result_narration`` from the queries TO narrate — for a result the
# tool itself returned that is the DISPATCHED list (``execute`` sets the composed
# line on ``ToolResult.narration``, #2661), never the requested list, which past
# the per-call cap names pages that were never fetched.
_NARRATION_SEARCHED = "You searched for {queries}"
_NARRATION_OPENED = "You opened {urls}"
_NARRATION_ALSO_OPENED = "and opened {urls}"
_NARRATION_LOOKED_UP = "You looked things up"
# A total failure states the failure AND its CONSEQUENCE.  "but couldn't read anything"
# says what happened to the call and stops there, leaving what follows from it — that
# there is now no source for the value the call went after — to be worked out; measured,
# 3 of 5 replies to an all-sources-failed lookup volunteered an invented figure anyway.
# This is presentation of the state the run is actually in, not an instruction: nothing
# here tells the model what to say, it says what is now true.
NARRATION_FAILURE_SUFFIX = (
    "but couldn't read anything — no page was read, so there's nothing current to quote"
)

# ── Micro-context (extract) render forms ──────────────────────────────────────
# When a browse call carries an ``extract`` instruction, the page body never
# enters the main loop: only the typed result (or an honest failure) plus the
# fetch handle to the full content stored in browse-results.  Since #2615 the
# draw returns plain text with no tagged shape — there is one success render
# and one failure render, not one per enumerated outcome.
_EXTRACT_HANDLE_CLAUSE = "Full page content saved as {handles}."
_EXTRACT_NO_HANDLE_CLAUSE = "The full page content was not separately stored."
# A SUCCESSFUL extraction renders the value WITH THE QUESTION IT ANSWERS, and no
# handle clause.
#
# The pairing is the code owner's ruling (#1918): "extract 'the price of blah' →
# response 'the price of blah is $123'" — unaffected by #2615, which retired the
# tagged shape underneath the value, not this pairing: a turn that makes several
# browse calls still needs to know which ask each answer is for.  Composed HERE,
# mechanically, from the tool's own ``extract`` argument and the value the
# micro-context returned — python-space, not model-space.  The label is the
# instruction VERBATIM, because a rephrasing would be this render inventing a
# question the caller did not ask.
#
# What it fixes: the extract is handed the instruction and returns the value WITHOUT it,
# so a bare answer arrived with nothing saying what it answered.  Measured — a watch
# whose page read "The dawn sailing: scheduled 05:20." got back "05:20", and the cycle
# recorded the previous value instead, its own reasoning saying "the extracted value for
# 'dawn sailing' was empty".  A value that happens to describe itself ("not scheduled")
# survived; a bare number did not, and which is which was luck.
#
# The handle clause stays off a success.  The "saved to browse-results#N" phrasing, read
# at the moment a chat teach round holds the extracted value, answered the wrong
# question: it claimed a save had happened (true of the raw page) in words the model read
# as the remembering being done (false — the value is ephemeral), and the planned
# collection_write never fired (2026-07-19 beat-1b transcripts).  The failure render
# below KEEPS the clause: there the handle is the remedy.
_EXTRACT_SUCCESS = "{instruction}: {value}"
# One failure render (#2615): with no tagged shape, there is no longer a second
# failure KIND (a clean-but-untagged draw) distinct from poison — every draw
# that never came back usable, garbled or blank alike, reads the same way.
_EXTRACT_FAILED = (
    "Couldn't extract {instruction!r} from the page — the extractor never produced a "
    "usable answer after {attempts} attempts. {handle_clause}"
)
_EXTRACT_UNAVAILABLE = (
    "Couldn't extract {instruction!r} — no extraction model is configured for this browse. "
    "{handle_clause}"
)
# A draw that RAISED (a transport failure, say) instead of returning a typed outcome
# (#2367) — rendered like the other failure outcomes, under the page's own header, so
# one page's draw error no longer discards every OTHER page's already-finished result.
_EXTRACT_DRAW_ERROR = (
    "Couldn't extract {instruction!r} from the page — the extraction call errored: "
    "{error}. {handle_clause}"
)
# A FAILED extraction hands back the page's own links (#2141).  The page body never
# enters the conversation when ``extract`` is set, so a failure that says only that it
# failed leaves the model with no anchor for its next call — it re-searches, lands on
# the same page, and loops.  The links are the one part of the page the model can act
# on, so they come back with the failure sentence.
_EXTRACT_LINKS_HEADER = "Links on the page:"

# ── Timeout report (#2367) ──────────────────────────────────────────────────────
# What a browse call reports when the executor's timeout cancels it, built ONLY
# from state written before the cancellation landed (self._read_log /
# self._extract_log) — never claiming more than those logs actually show.
_TIMEOUT_PARTIAL_PROGRESS = "It timed out after {timeout}s, partway through. {read_clause}"
_TIMEOUT_READ_CLAUSE = "Read: {listed}."
_TIMEOUT_NOT_STORED = "not stored"
_TIMEOUT_EXTRACT_FINISHED = "Extraction finished for these pages:\n{rendered}"
_TIMEOUT_EXTRACT_WAITING = "Still waiting on: {labels}."

# Type alias for the browser request function
RequestFn = Callable[[str, dict], Awaitable[tuple[str, str | None, PageReading]]]


def _trim_search_result(text: str, query: str, search_host: str | None) -> str:
    """Parse a search result page into a numbered list of titles and URLs (#2647).

    A result's summary snippet often carries the answer itself, and a reply built
    from snippets is built from pages never opened — so the render is a clean list
    with nothing else from the page: each result is its title (the heading line
    above its link) and the link's own target, and nothing below.  Real results
    carry a heading (``### Title``); a standalone link under anything else — a
    knowledge-panel paragraph, an infobox row — is not a result and is dropped
    (#2697), so its text never renders as a title.  The search engine's own
    affordance links (its summariser, its maps) point back at the search host and
    are not results, so they are dropped the same way the snippet lines are.

    A page with no standalone links at all passes through unchanged — the
    "nothing found" path keeps its own shape.  A page whose standalone links all
    sit under non-heading lines (a search engine that dropped its headings) still
    lists those links under the same instruction header, each titled by its own
    link text (#2697): the raw page — knowledge-panel prose, snippets and all —
    must never enter the context under no header.
    """
    lines = text.split("\n")

    results: list[tuple[str, str]] = []
    unheaded: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        match = _LINK_RE.match(line)
        if not match:
            continue
        url = match.group(2)
        if search_host and _url_host(url) == search_host:
            continue
        above = i - 1
        while above >= 0 and not lines[above].strip():
            above -= 1
        # Only a heading line above the link names a result (#2697): a knowledge
        # panel's paragraph or an infobox row rendered as a "title" under the
        # titles-and-URLs-only header, and the model answered from it.
        if above >= 0 and lines[above].lstrip().startswith("#"):
            results.append((lines[above].lstrip().lstrip("#").strip(), url))
        else:
            unheaded.append((match.group(1), url))

    if not results:
        results = unheaded
    results = results[: PennyConstants.MAX_SEARCH_LINKS]

    if not results:
        return text

    listed = "\n".join(
        f"{number}. {title} ({url})" for number, (title, url) in enumerate(results, 1)
    )
    instruction = Prompt.SEARCH_RESULT_INSTRUCTION.format(query=query)
    return f"{instruction}\n\n{listed}"


def _url_host(url: str) -> str:
    """The host of a link target, lowercased — the one part the engine-affordance
    filter reads."""
    return urllib.parse.urlparse(url).hostname or ""


def _links_clause(text: str) -> str:
    """The standalone markdown links in ``text``, capped at MAX_SEARCH_LINKS, as the
    tail a failed extraction appends — empty when the page carries none.

    Same solo-link detection the search trim uses (``_LINK_RE``): a line that is
    nothing but a link is a link the model can open, and prose merely containing one
    is not.
    """
    links = [line.strip() for line in text.split("\n") if _LINK_RE.match(line)]
    if not links:
        return ""
    listed = "\n".join(links[: PennyConstants.MAX_SEARCH_LINKS])
    return f"\n{_EXTRACT_LINKS_HEADER}\n{listed}"


class BrowseTool(Tool):
    """Search the web and read pages via the browser extension.

    The model emits one tool call with a queries array:
      {"queries": ["topic", "https://example.com", "another topic"]}
    URLs are read directly; plain text is converted to a search URL.
    Every query up to the per-call total ceiling runs in this call, fetched in
    parallel batches of ``max_calls`` — the browser-tab limit — one batch after
    another.  Queries past the total come back named in a dropped section.
    """

    name = "browse"
    args_model = BrowseArgs
    # A backstop, not the bound on extraction: each extraction draw stops at its token
    # budget (8,000 for browse-extract) with one thinking-off retry, so the worst
    # legitimate call on the slowest measured hardware (about 26 tokens/s) is one page
    # fetch exhausting its retries (~165 s) plus one extraction and its retry (16,000
    # tokens, ~615 s). 900 s covers that with margin; when it fires, timeout_report
    # says what finished.
    timeout = 900.0

    def __init__(
        self,
        max_calls: int,
        search_url: str = "https://duckduckgo.com/?q=",
        db: Database | None = None,
        author: str = "unknown",
        *,
        embedding_client: LlmClient,
        model_client: LlmClient | None = None,
        channel_outage_recovery: str = Prompt.BROWSE_OUTAGE_RECOVERY_CHAT,
        max_queries_per_call: int | None = None,
    ):
        self._max_calls = max_calls
        # The per-call TOTAL ceiling (#2662): every requested query up to this runs
        # in this call, in parallel batches of ``max_calls``.  Unwired constructions
        # (older tests, tools built before the setting exists) fall back to the
        # batch size itself, so a tool without a ceiling behaves exactly as the
        # pre-batching tool did.
        self._max_queries_per_call = (
            max_calls if max_queries_per_call is None else (max_queries_per_call)
        )
        self._search_url = search_url
        self._search_host = _url_host(search_url)
        self._db = db
        self._embedding_client = embedding_client
        self._author = author
        # The shared model client powers the ``extract`` micro-context (a
        # single-shot extraction call).  Optional: chat's browse never sets
        # ``extract``, and tests that don't exercise it leave this None; an
        # ``extract`` requested without a client wired degrades visibly rather
        # than silently returning the page body.
        self._micro_context = (
            MicroContext(model_client, db=db) if model_client is not None else None
        )
        # The terminal move bound into a whole-channel outage error.  Defaults to
        # the chat clause; a collector passes its done()-binding clause so the
        # outage names the recovery its agent can actually perform.
        self._channel_outage_recovery = channel_outage_recovery
        self._browse_provider: Callable[[], tuple[RequestFn, PermissionManager] | None] | None = (
            None
        )
        # Progress toward THIS call's timeout report (#2367) — reset at the top of
        # every ``execute()``, written to as the call proceeds so ``timeout_report``
        # can read back whatever had finished when the executor's timeout cancelled
        # us (the cancelled coroutine's own locals are gone by then; only state on
        # ``self`` survives).  Per-instance, not per-call: two concurrent ``browse``
        # calls on the same instance in one step would share this record — accepted
        # for now (the model is steered toward one call per turn via the queries
        # array), not guarded.
        self._read_log: list[tuple[str, MemoryEntry | None]] = []
        self._extract_log: list[tuple[str, str]] = []
        self._extracting = False

    @property
    def description(self) -> str:  # type: ignore[override]
        """Dynamic description reflecting the per-call total ceiling."""
        n = self._max_queries_per_call
        items = "query and/or URL" if n == 1 else "queries and/or URLs"
        return f"Look things up. Pass up to {n} {items}."

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        """Dynamic parameters reflecting the per-call total ceiling."""
        n = self._max_queries_per_call
        return {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Think out loud about what you're looking up and why.",
                },
                "queries": {
                    "type": "array",
                    "description": f"Search queries and/or URLs to look up (max {n})",
                    "items": {"type": "string"},
                    "maxItems": n,
                },
                "extract": {
                    "type": "string",
                    "description": (
                        "Optional. One instruction naming what to pull out of the fetched "
                        'pages (e.g. "the current bid amount"). Use the task\'s own words '
                        'for it, and a broad ask like "what the notice says" is fine. An '
                        "instruction naming several things comes back with whatever each "
                        "page has for each of them, so a detail a page lacks costs you "
                        "only that detail. When set, the full page content is read in a "
                        "separate scoped context and only the extracted value is returned "
                        "here — the page body never enters this conversation. A search "
                        "query comes back as its list of links whatever you set here, so "
                        "pick one and read it in a follow-up call. Omit to receive the "
                        "page content itself."
                    ),
                },
            },
            "required": ["queries", "extract"],
        }

    def set_browse_provider(
        self,
        provider: Callable[[], tuple[RequestFn, PermissionManager] | None],
    ) -> None:
        """Set a provider that returns (request_fn, permission_manager) or None."""
        self._browse_provider = provider

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        """Format lookups into a readable status string."""
        parts: list[str] = []
        for q in arguments.get("queries", []):
            if _URL_PATTERN.match(q):
                short = q.replace("https://", "").replace("http://", "")
                parts.append(f"Reading {short}")
            else:
                parts.append(f'Searching "{q}"')
        return "<br>".join(parts) if parts else "Looking up..."

    @classmethod
    def to_progress_emoji(cls, arguments: dict) -> ProgressEmoji:
        """Pick 📖 if any query is a URL (reading), 🔍 otherwise (searching)."""
        for q in arguments.get("queries", []):
            if _URL_PATTERN.match(q):
                return ProgressEmoji.READING
        return ProgressEmoji.SEARCHING

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """One first-person line summarising the WHOLE browse call.

        The seam (``format_result``) prepends this single line to the joined
        ``result.message`` and appends the ``(browse result)`` tag; the per-page
        ``## browse ...:`` section headers already disambiguate each page in the
        body, so this narrates the *batch* — what was searched and/or opened, and
        whether anything was readable — as one natural clause, not per section.

        A result ``execute`` itself returned carries the composed line on
        ``ToolResult.narration`` (built from the queries actually DISPATCHED,
        #2661), so the seam never reaches here for one — this override answers
        only for a result built OUTSIDE the tool (a synthetic seeded result,
        which has no cap to have dropped anything past) and narrates the
        requested queries.  Branches on ``result.success``, which the tool
        reports False only when every dispatched query errored (a total
        failure), so a partial success still narrates the action it took.
        """
        return cls._result_narration(arguments.get("queries", []), result.success)

    @classmethod
    def _result_narration(cls, queries: list[str], success: bool) -> str:
        """The narration line for ``queries`` — the ONE place the action clause and
        the failure suffix meet.  Callers pass the list to NARRATE: the dispatched
        queries for a result the tool built, the requested ones for a synthetic
        result with no dispatched list behind it."""
        searches = [q for q in queries if not _URL_PATTERN.match(q)]
        reads = [q for q in queries if _URL_PATTERN.match(q)]
        action = cls._browse_action_clause(searches, reads)
        if success:
            return action
        return f"{action} {NARRATION_FAILURE_SUFFIX}"

    @staticmethod
    def _browse_action_clause(searches: list[str], reads: list[str]) -> str:
        """Compose the "You searched for … and opened …" action clause from the
        split queries — searches quoted, direct-read URLs bare."""
        parts: list[str] = []
        if searches:
            quoted = ", ".join(f'"{q}"' for q in searches)
            parts.append(_NARRATION_SEARCHED.format(queries=quoted))
        if reads:
            template = _NARRATION_ALSO_OPENED if searches else _NARRATION_OPENED
            parts.append(template.format(urls=", ".join(reads)))
        return " ".join(parts) if parts else _NARRATION_LOOKED_UP

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Dispatch every requested lookup (up to the per-call total) via the browser
        extension, in parallel batches of ``max_calls``.

        With an ``extract`` micro-instruction (and at least one readable page),
        the fetched content goes to a fresh micro-context and only the typed
        result returns; otherwise the assembled sections return directly.

        ``call_link`` (#2458) is THIS dispatched call's own linkage — the tool
        dispatcher built it via ``CallLink.for_tool_call(tool_call_id)`` before
        ``run`` even validated args. Threaded through to every extraction draw this
        call makes, so each one links back to the run AND to this dispatched call.

        Every result returned here carries its own ``narration`` (#2661), composed
        from ``to_run`` — the queries actually dispatched — so the first-person line
        never names a page past the per-call cap that was never fetched; the
        ``## browse dropped:`` section stays the only place the rest are named."""
        args = BrowseArgs(**kwargs)
        self._read_log, self._extract_log, self._extracting = [], [], False
        total = self._max_queries_per_call
        to_run, dropped = args.queries[:total], args.queries[total:]

        sections: list[str] = []
        page_sections: list[str] = []
        page_readings: list[PageReading] = []
        captures: list[BrowsePage] = []
        channel_outage = False
        results: list[Any] = []
        # Batching (#2662): the parallel cap bounds how many fetches are in flight,
        # not how many the call runs — the requested queries up to the TOTAL ceiling
        # go through in batches of ``max_calls``, one batch after another, so a
        # four-site job pays no extra model round trip.  Order is preserved: each
        # batch's sections append in request order, and the batches run in order.
        for start in range(0, len(to_run), self._max_calls):
            batch = to_run[start : start + self._max_calls]
            tasks = self._build_tasks(batch)
            batch_results = await asyncio.gather(
                *[coro for _, _, coro in tasks], return_exceptions=True
            )
            results.extend(batch_results)
            (
                batch_sections,
                batch_pages,
                batch_readings,
                batch_captures,
                batch_outage,
            ) = self._assemble_sections(tasks, batch_results)
            sections.extend(batch_sections)
            page_sections.extend(batch_pages)
            page_readings.extend(batch_readings)
            captures.extend(batch_captures)
            channel_outage = channel_outage or batch_outage
        if channel_outage:
            sections.append(self._channel_outage_section())

        stored = await self._append_pages_to_browse_results(page_sections)
        await self._store_media(captures)
        self._record_read_log(page_sections, stored)

        if dropped:
            sections.append(self._dropped_section(total, dropped))
        # Browse is a *read* for work-accounting (its browse-results log write is
        # incidental), so ``mutated`` stays False.  But when EVERY dispatched query
        # errored the call did nothing, so it reports ``success=False`` — making the
        # failure visible to structural accounting (``record.failed`` → the run's
        # ``tool_failures`` count → run-health), not just to the error text.  A
        # partial failure keeps ``success=True``: the model works from what succeeded.
        all_failed = bool(results) and all(isinstance(r, BaseException) for r in results)
        if args.extract is not None and not all_failed:
            self._extracting = True
            return await self._extract_result(
                sections, stored, page_readings, args.extract, call_link, ran=to_run
            )
        return ToolResult(
            message=PennyConstants.SECTION_SEPARATOR.join(sections),
            success=not all_failed,
            source_urls=self._read_urls(sections),
            narration=self._result_narration(to_run, not all_failed),
        )

    @staticmethod
    def _read_urls(sections: list[str]) -> list[str]:
        """The URL of every page THIS call actually read successfully, in read
        order — taken off each page section's own ``## browse: <url>`` header
        value, the one place the tool already records it.  A failed read
        (``## browse error:``) read nothing and a search section (``## browse
        search:``) is links, not a page read, so neither contributes (#2648):
        these ride ``ToolResult.source_urls`` to the reply-side fallback, which
        appends the first one when the composed reply carries no URL."""
        return [
            section[len(PennyConstants.BROWSE_PAGE_HEADER) :].partition("\n")[0]
            for section in sections
            if section.startswith(PennyConstants.BROWSE_PAGE_HEADER)
        ]

    def timeout_report(self, timeout: float) -> str:
        """Override of ``Tool.timeout_report`` (#2367): what THIS call had
        finished when the executor's timeout cancelled it, read off
        ``self._read_log`` / ``self._extract_log`` — written to as the call
        progressed, since the cancelled coroutine's own locals are gone by the
        time this runs.  Nothing here claims more than those logs actually show:
        no page is named as read, stored, or extracted unless it is IN one of
        them.  Falls back to the generic default when nothing was read yet (a
        cancellation before the first page came back has nothing to report)."""
        if not self._read_log:
            return super().timeout_report(timeout)
        parts = [_TIMEOUT_PARTIAL_PROGRESS.format(timeout=timeout, read_clause=self._read_clause())]
        if self._extracting:
            parts.append(self._extract_progress_clause())
        return "\n\n".join(parts)

    def _read_clause(self) -> str:
        """The pages fetched before the timeout, each with its own stored handle
        (or a note that it wasn't stored)."""
        listed = "; ".join(
            f"{label} ({self._entry_handle(entry) if entry else _TIMEOUT_NOT_STORED})"
            for label, entry in self._read_log
        )
        return _TIMEOUT_READ_CLAUSE.format(listed=listed)

    def _extract_progress_clause(self) -> str:
        """Which pages' extractions finished before the timeout — rendered with
        their FULL content (the same section the model would have read without a
        timeout), not just a label — and which pages were read but never got an
        extraction result back."""
        finished_labels = {label for label, _ in self._extract_log}
        waiting = [label for label, _ in self._read_log if label not in finished_labels]
        parts = []
        if self._extract_log:
            rendered = PennyConstants.SECTION_SEPARATOR.join(r for _, r in self._extract_log)
            parts.append(_TIMEOUT_EXTRACT_FINISHED.format(rendered=rendered))
        if waiting:
            parts.append(_TIMEOUT_EXTRACT_WAITING.format(labels="; ".join(waiting)))
        return "\n\n".join(parts)

    @staticmethod
    def _page_label(section: str) -> str:
        """A page section's own header line — the one identifier used for BOTH the
        read log and the extract log, so a page is named the same way in either
        (#2367) without re-deriving it at each call site."""
        return section.partition("\n")[0]

    def _record_read_log(self, page_sections: list[str], stored: list[MemoryEntry]) -> None:
        """Record which pages were fetched and where each is stored, toward this
        call's timeout report (#2367) — BEFORE the slow extract stage starts, so a
        cancellation during extraction still leaves this much behind.  ``stored`` is
        empty when nothing was stored (no db, or nothing to store) even though pages
        were fetched; pad with ``None`` rather than mis-pairing a page with the
        wrong handle."""
        handles: list[MemoryEntry | None] = (
            list(stored) if len(stored) == len(page_sections) else [None] * len(page_sections)
        )
        labels = [self._page_label(section) for section in page_sections]
        self._read_log = list(zip(labels, handles, strict=True))

    # ── Micro-context (extract) path ──────────────────────────────────────────

    async def _extract_result(
        self,
        sections: list[str],
        stored: list[MemoryEntry],
        page_readings: list[PageReading],
        instruction: str,
        call_link: CallLink,
        ran: list[str],
    ) -> ToolResult:
        """One micro-context call PER successfully-fetched page (#1682) — each result
        rendered UNDER that page's own ``## browse …:`` section header, so a
        multi-source digest keeps items attributed to their source and a page that
        came back empty renders its own honest text in its own section without
        masking another page's extracted value.  Signal sections (read errors,
        dropped-query notes, channel
        outage) stay verbatim in place; a failed fetch keeps its ``## browse error:``
        section with no micro-context.  **A SEARCH section stays verbatim too (#2141)**
        — titles and links only, by its own header's words, so there is nothing on it
        to extract and it comes back as the list of links it is.  Each stored page-read
        carries its OWN fetch handle (a search section carries none — it was never
        stored).

        The page batch is drawn CONCURRENTLY (#1941).  The per-page contexts are
        independent by construction — a fresh single-shot context, its own document,
        its own ledger row, its own poison/reroll budget — so no draw reads anything
        the draw before it produced, and running them one after another spent the
        whole batch's latency serially (a three-page extract, ~45-60s).

        Attribution survives because the batch is PLANNED before it runs: each page's
        slot in the rendered output and its own fetch handle are bound in section
        order up front (``_extract_batch``), and every result is written back into
        that slot (``_merge_extracted``) — so the sections render in page order
        whatever order the draws finish in.  Nothing else moves: same calls, same
        renders, same per-draw reroll machinery.

        ``ran`` is the queries this call actually dispatched (``execute``'s
        ``to_run``) — the narration names exactly those, never the requested
        list past the cap (#2661)."""
        micro_context = self._micro_context
        if micro_context is None:
            return self._extract_unavailable_result(
                sections, stored, page_readings, instruction, ran
            )
        batch = self._extract_batch(sections, stored, page_readings)
        drawn = await asyncio.gather(
            *[
                self._extract_one_page(
                    micro_context, section, instruction, entry, reading, call_link
                )
                for _, section, entry, reading in batch
            ],
            return_exceptions=True,
        )
        extracted = self._drawn_pages(drawn)
        success = self._read_anything(sections, extracted)
        return ToolResult(
            message=PennyConstants.SECTION_SEPARATOR.join(
                self._merge_extracted(sections, batch, extracted)
            ),
            success=success,
            source_urls=self._read_urls(sections),
            narration=self._result_narration(ran, success),
        )

    @staticmethod
    def _extract_batch(
        sections: list[str], stored: list[MemoryEntry], page_readings: list[PageReading]
    ) -> list[tuple[int, str, MemoryEntry | None, PageReading]]:
        """Plan the concurrent batch — one ``(slot, section, handle, reading)`` per
        PAGE-READ section, in page order.

        Only a page read is extracted from (#2141): a search section is titles and
        links, so it is left where it stands and the merge renders it verbatim, beside
        the error / dropped / outage signals.

        The fetch handles AND the reading marks (#2388) are consumed HERE,
        sequentially, aligned 1:1 with the page-read sections in order, so which page
        owns which handle (or mark) is settled before a single draw is issued: a search
        section (readable, never stored) and a read failure (no content section at
        all) cannot slide a page's handle or reading onto its neighbour, whatever order
        the concurrent draws complete in."""
        handles = iter(stored)
        readings = iter(page_readings)
        batch: list[tuple[int, str, MemoryEntry | None, PageReading]] = []
        for slot, section in enumerate(sections):
            if not section.startswith(PennyConstants.BROWSE_PAGE_HEADER):
                continue  # search / error / dropped / outage — kept verbatim by the merge
            batch.append((slot, section, next(handles, None), next(readings, PageReading.ARTICLE)))
        return batch

    @staticmethod
    def _read_anything(sections: list[str], extracted: list[tuple[str, bool]]) -> bool:
        """Whether the call came back with something the model can work from: a page
        whose extraction succeeded (or honestly reported the fact absent), or a search
        section — which is returned verbatim, so no draw speaks for it (#2141)."""
        if any(section.startswith(PennyConstants.BROWSE_SEARCH_HEADER) for section in sections):
            return True
        return any(succeeded for _, succeeded in extracted)

    @staticmethod
    def _drawn_pages(drawn: list[tuple[str, bool] | BaseException]) -> list[tuple[str, bool]]:
        """Every page the batch drew.  ``_extract_one_page`` already catches its
        own draw's failures and renders them as that page's own result (#2367), so
        nothing here should ever be a ``BaseException`` — one reaching this loop is
        a bug elsewhere, and raising it is what surfaces that bug instead of
        hiding it behind a silent, headerless page failure."""
        pages: list[tuple[str, bool]] = []
        for outcome in drawn:
            if isinstance(outcome, BaseException):
                raise outcome
            pages.append(outcome)
        return pages

    @staticmethod
    def _merge_extracted(
        sections: list[str],
        batch: list[tuple[int, str, MemoryEntry | None, PageReading]],
        extracted: list[tuple[str, bool]],
    ) -> list[str]:
        """Write each drawn page back into ITS OWN slot, leaving every signal section
        exactly where it stood — the rendered order is the page order the batch was
        planned in, never the order the concurrent draws happened to finish in."""
        rendered = list(sections)
        for (slot, _, _, _), (page_section, _succeeded) in zip(batch, extracted, strict=True):
            rendered[slot] = page_section
        return rendered

    async def _extract_one_page(
        self,
        micro_context: MicroContext,
        section: str,
        instruction: str,
        entry: MemoryEntry | None,
        reading: PageReading,
        call_link: CallLink,
    ) -> tuple[str, bool]:
        """Extract ONE fetched page's content and render the typed result under that
        page's own section header — the plain-text answer (#2615) or an honest
        failure — plus this page's fetch handle.  Returns the rendered section and
        whether it was a successful read: only a failed draw (no usable text after
        the whole reroll budget, or a draw error) reports False — a plain-text answer
        that says the page carries none of what was asked is still a successful read
        of that absence, same as every other answer.

        A draw that RAISES (a transport failure, say) is caught HERE, per page
        (#2367), rather than left to propagate out of the whole batch — deliberately
        ``Exception``, never ``BaseException``, so a ``CancelledError`` (this call's
        own 300s timeout firing, or an outer preemption) still propagates untouched.
        Either way, the finished render is recorded to ``self._extract_log`` toward
        this call's timeout report, BEFORE returning — so a page that finishes
        before a later sibling's draw stalls out the whole tool's timeout has
        already been captured."""
        stored = [entry] if entry is not None else []
        links = _links_clause(section)
        try:
            micro = await micro_context.extract(
                section,
                instruction,
                call_link=call_link,
                run_target=self._author,
                reading=reading,
            )
        except Exception as exc:
            body = self._render_draw_error(exc, instruction, stored, links)
            succeeded = False
        else:
            body = self._render_micro_result(micro, instruction, stored, links)
            succeeded = micro.outcome == MicroExtractOutcome.EXTRACTED
        rendered = self._under_page_header(section, body)
        self._extract_log.append((self._page_label(section), rendered))
        return rendered, succeeded

    def _render_draw_error(
        self, error: Exception, instruction: str, stored: list[MemoryEntry], links: str
    ) -> str:
        """One page's own failure when its draw raised instead of returning a typed
        outcome — rendered like the other failure outcomes: the page's own links as
        the only anchor the next call has, plus the fetch handle (#2367).  This is
        what stops a single raised draw from discarding every OTHER page's result,
        which an un-caught raise out of this method used to do."""
        handle_clause = self._handle_clause(stored)
        failure = _EXTRACT_DRAW_ERROR.format(
            instruction=instruction, error=error, handle_clause=handle_clause
        )
        return f"{failure}{links}"

    def _render_micro_result(
        self,
        micro: MicroContextResult,
        instruction: str,
        stored: list[MemoryEntry],
        links: str,
    ) -> str:
        """The main-loop body for one micro-context outcome — on success the extracted
        text LABELLED with the instruction it answers (#1918; no handle clause — see
        _EXTRACT_SUCCESS), or an honest failure carrying the fetch handle to the stored
        full content as its remedy, plus the page's own links (#2141) — the body never
        entered the conversation, so they are the only anchor the next call has.

        Since #2615 the value itself is whatever plain text the draw wrote — what the
        page has for each thing asked, any shortfall, any "none of it" — so there is
        nothing left for this method to compose beyond the instruction label."""
        if micro.outcome == MicroExtractOutcome.EXTRACTED:
            return _EXTRACT_SUCCESS.format(instruction=instruction, value=micro.value)
        handle_clause = self._handle_clause(stored)
        failure = _EXTRACT_FAILED.format(
            instruction=instruction,
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS,
            handle_clause=handle_clause,
        )
        return f"{failure}{links}"

    def _extract_unavailable_result(
        self,
        sections: list[str],
        stored: list[MemoryEntry],
        page_readings: list[PageReading],
        instruction: str,
        ran: list[str],
    ) -> ToolResult:
        """Honest degradation when ``extract`` is requested but no model client is
        wired for the micro-context — each page read renders the named failure under
        its own section header, with its fetch handle to the stored content and the
        page's own links, so the request fails visibly and still leaves an anchor
        rather than dumping the body.  Search and signal sections stay verbatim, as
        they do on every other extract path — so a browse that only searched is served
        whole by a tool with no extraction model, and says nothing about one.  No draw
        happens on this path, so the reading mark is carried only for ``_extract_batch``'s
        shared signature — never read here.  ``ran`` is the queries this call actually
        dispatched, for the narration (#2661)."""
        batch = self._extract_batch(sections, stored, page_readings)
        if batch:
            logger.error(
                "browse(extract=...) requested but no model client is wired for extraction"
            )
        unavailable = [
            (self._unavailable_section(section, instruction, entry), False)
            for _, section, entry, _reading in batch
        ]
        success = self._read_anything(sections, unavailable)
        return ToolResult(
            message=PennyConstants.SECTION_SEPARATOR.join(
                self._merge_extracted(sections, batch, unavailable)
            ),
            success=success,
            source_urls=self._read_urls(sections),
            narration=self._result_narration(ran, success),
        )

    def _unavailable_section(
        self, section: str, instruction: str, entry: MemoryEntry | None
    ) -> str:
        """One page read's no-model degradation, under that page's own header."""
        stored = [entry] if entry is not None else []
        body = _EXTRACT_UNAVAILABLE.format(
            instruction=instruction, handle_clause=self._handle_clause(stored)
        )
        return self._under_page_header(section, f"{body}{_links_clause(section)}")

    @staticmethod
    def _under_page_header(section: str, body: str) -> str:
        """Render a micro-context body under the page's OWN ``## browse …:`` header,
        so every result stays attributed to the page it came from."""
        header_line = section.partition("\n")[0]
        return f"{header_line}\n{body}"

    def _handle_clause(self, stored: list[MemoryEntry]) -> str:
        """The fetch-handle clause — the log and id of each page stored this
        call, or a note when nothing was stored (a pure search has readable
        content but no stored page)."""
        if not stored:
            return _EXTRACT_NO_HANDLE_CLAUSE
        handles = ", ".join(self._entry_handle(entry) for entry in stored)
        return _EXTRACT_HANDLE_CLAUSE.format(handles=handles)

    @staticmethod
    def _entry_handle(entry: MemoryEntry) -> str:
        """One stored page as the two arguments that read it back: ``log='browse-results'
        id=<id>``.  It names no tool, so the clause is true on a surface that carries
        the read and on one that does not (a collector cycle scoped to its program)."""
        return f"log={PennyConstants.MEMORY_BROWSE_RESULTS_LOG!r} id={entry.id}"

    def _build_tasks(self, queries: list[str]) -> list[tuple[str, str, Any]]:
        """One ``(header, value, coroutine)`` per query — URLs read directly, plain
        text routed through the configured search URL."""
        tasks: list[tuple[str, str, Any]] = []
        for q in queries:
            if _URL_PATTERN.match(q):
                tasks.append((PennyConstants.BROWSE_PAGE_HEADER, q, self._read_page(q)))
            else:
                search_url = f"{self._search_url}{urllib.parse.quote(q)}"
                tasks.append((PennyConstants.BROWSE_SEARCH_HEADER, q, self._read_page(search_url)))
        return tasks

    def _assemble_sections(
        self, tasks: list[tuple[str, str, Any]], results: list[Any]
    ) -> tuple[list[str], list[str], list[PageReading], list[BrowsePage], bool]:
        """Fold gathered results into rendered sections, the page-only subset (for
        the browse-results log), each page-only section's own reading mark (#2388,
        aligned 1:1 with ``page_sections`` the same way ``stored`` is), the
        captured page images, and whether a channel outage hit — the outage
        SECTION itself is appended by ``execute`` ONCE per call, not once per
        batch (#2662: a whole-channel outage dooms every batch identically, and
        one rendered outage names it)."""
        sections: list[str] = []
        page_sections: list[str] = []
        page_readings: list[PageReading] = []
        captures: list[BrowsePage] = []
        channel_outage = False
        for (header, value, _), result in zip(tasks, results, strict=True):
            if isinstance(result, BrowseChannelUnavailableError):
                logger.warning("Browse channel outage (%s%s): %s", header, value, result)
                channel_outage = True
                continue
            if isinstance(result, BaseException):
                sections.append(self._error_section(header, value, result))
                continue
            section = self._page_section(header, value, result, self._search_host)
            sections.append(section)
            if header == PennyConstants.BROWSE_PAGE_HEADER:
                page_sections.append(section)
                page_readings.append(result.reading)
                if result.image:
                    captures.append(result)
        return sections, page_sections, page_readings, captures, channel_outage

    @staticmethod
    def _error_section(header: str, value: str, error: BaseException) -> str:
        """Render one failed sub-call — a page-level failure the model recovers from
        by trying another source."""
        logger.warning("Browse sub-call failed (%s%s): %s", header, value, error)
        error_label = f"{PennyConstants.BROWSE_ERROR_HEADER}{value}"
        return (
            f"{error_label}\nCould not read this page: {error}. "
            f"Try a different source or a reworded query; if other queries in this "
            f"batch succeeded, work from those instead of retrying this one."
        )

    def _channel_outage_section(self) -> str:
        """Render the whole-channel outage ONCE, binding the terminal move.

        A disconnected browser dooms every read and search this cycle, so the
        per-URL "try a different source" guidance — repeated once per query —
        misreads a single outage as N independent page failures and invites the
        URL-variant retries (http/https, trailing slash, mirror) that flood the
        production traces.  Naming the outage once and binding the recovery its
        agent can actually perform stops the flailing (visible-degradation)."""
        return (
            f"{PennyConstants.BROWSE_ERROR_HEADER}browser disconnected\n"
            "Browsing is unavailable this cycle: no browser is connected, so every "
            "search and page read fails and retrying other URLs or query variants "
            f"won't help. {self._channel_outage_recovery}"
        )

    @staticmethod
    def _page_section(header: str, value: str, result: BrowsePage, search_host: str) -> str:
        """Render one successful sub-call; a search page parses to its titles and
        URLs (#2647)."""
        text = result.text
        if header == PennyConstants.BROWSE_SEARCH_HEADER:
            text = _trim_search_result(text, value, search_host)
        return f"{header}{value}\n{text}"

    @staticmethod
    def _dropped_section(total: int, dropped: list[str]) -> str:
        """Name the queries dropped past the per-call TOTAL so the omission is visible
        to the model, with the exact recovery: rerun the rest in a follow-up call."""
        ran = "query was" if total == 1 else f"{total} queries were"
        count = len(dropped)
        dropped_list = ", ".join(repr(q) for q in dropped)
        return (
            f"{PennyConstants.BROWSE_DROPPED_HEADER}only the first {ran} run; "
            f"{count} beyond the {total}-per-call total were not run: {dropped_list}. "
            f"Call browse(queries=[{dropped_list}]) again if you still need the results."
        )

    async def _append_pages_to_browse_results(self, page_sections: list[str]) -> list[MemoryEntry]:
        """Side-effect-write each successful page as its own log entry, returning
        the created entries (their ids are the fetch handles a micro-context
        anchors its extracted value to).

        Search-result and error sections are skipped — only full page
        reads carry knowledge worth indexing.  Embeds each entry at
        write time so similarity recall (and the knowledge extractor)
        can address pages individually.
        """
        if self._db is None or not page_sections:
            return []
        entries: list[LogEntryInput] = []
        for section in page_sections:
            vec = await embed_text(self._embedding_client, section)
            entries.append(LogEntryInput(content=section, content_embedding=vec))
        browse_log = self._db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
        if browse_log is None:
            return []
        return browse_log.append(entries, author=self._author)

    async def _read_page(self, url: str) -> BrowsePage:
        """Read a single URL via the browser extension, retrying with backoff on disconnect.

        Raises ``BrowseChannelUnavailableError`` when no browser is reachable after all
        retries (a whole-channel outage — every query this cycle is doomed), a plain
        ``ConnectionError`` for a per-page timeout (that URL is slow/blocking; another
        source may work), and propagates any RuntimeError raised by the browser
        extension itself (a structured page failure: extraction failed, page never
        became ready, host permission denied, etc.).
        """
        for attempt in range(1 + PennyConstants.BROWSE_RETRIES):
            delay = PennyConstants.BROWSE_RETRY_DELAY * (2**attempt)
            connection = self._browse_provider() if self._browse_provider else None
            if not connection:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "No browser connection, retrying in %.0fs (%s)",
                        delay,
                        url,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise BrowseChannelUnavailableError(
                    "no browser is connected after retries — the browser extension isn't "
                    "running, so every web search and page read this cycle is unavailable"
                )

            request_fn, permission_manager = connection
            await permission_manager.check_domain(url)

            try:
                text, image_url, reading = await asyncio.wait_for(
                    request_fn("browse_url", {"url": url}),
                    timeout=PennyConstants.BROWSE_REQUEST_TIMEOUT,
                )
            except TimeoutError:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "Browser request timed out, retrying in %.0fs (%s)",
                        delay,
                        url,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise ConnectionError(
                    f"browser request timed out after {PennyConstants.BROWSE_REQUEST_TIMEOUT}s — "
                    f"the page may be slow or blocking automated reads; try a different source"
                ) from None
            except ConnectionError as exc:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "Browser disconnected, retrying in %.0fs (%s)",
                        delay,
                        url,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise BrowseChannelUnavailableError(
                    "browser disconnected and stayed down through every retry — the "
                    "channel is offline, so every read this cycle is doomed"
                ) from exc

            title = self._parse_title(text)
            text = clean_browser_content(text)
            return BrowsePage(text=text, image=image_url, title=title, url=url, reading=reading)

        raise BrowseChannelUnavailableError("no browser connected")

    @staticmethod
    def _parse_title(raw_text: str) -> str | None:
        """Pull the page title from the extension's ``Title: ...`` prefix line."""
        first_line = raw_text.split("\n", 1)[0]
        prefix = PennyConstants.BROWSE_TITLE_PREFIX
        if first_line.startswith(prefix):
            return first_line[len(prefix) :].strip() or None
        return None

    async def _store_media(self, pages: list[BrowsePage]) -> None:
        """Store each captured page image with its title+URL metadata embedding.

        Best-effort: a page whose image isn't a decodable base64 data URI is
        skipped rather than failing the browse.  The embedding lets channel
        egress match an outgoing message back to the most relevant image.
        """
        if self._db is None:
            return
        for page in pages:
            decoded = self._decode_data_uri(page.image)
            if decoded is None:
                continue
            data, mime_type = decoded
            metadata = self._media_metadata(page)
            vec = await embed_text(self._embedding_client, metadata)
            embedding = serialize_embedding(vec) if vec else None
            self._db.media.put(
                data=data,
                mime_type=mime_type,
                source_url=page.url,
                title=page.title,
                embedding=embedding,
            )

    @staticmethod
    def _media_metadata(page: BrowsePage) -> str:
        """Text embedded for egress matching — the page title plus its URL."""
        return "\n".join(part for part in (page.title, page.url) if part)

    @staticmethod
    def _decode_data_uri(image: str | None) -> tuple[bytes, str] | None:
        """Decode a ``data:<mime>;base64,<data>`` URI into (bytes, mime_type)."""
        if not image or not image.startswith("data:") or ";base64," not in image:
            logger.warning("Browse image is not a base64 data URI; skipping media store")
            return None
        header, encoded = image.split(";base64,", 1)
        mime_type = header[len("data:") :]
        try:
            return base64.b64decode(encoded), mime_type
        except binascii.Error:
            logger.warning("Browse image base64 failed to decode; skipping media store")
            return None
