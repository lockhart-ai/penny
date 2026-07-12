"""BrowseTool — searches and reads web pages via the browser extension.

The model packs everything into a single queries array; the tool detects URLs
and reads them directly, while plain text is converted to search URLs.
Queries are dispatched in parallel.
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

from penny.constants import PennyConstants, ProgressEmoji
from penny.database.memory import LogEntryInput
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

_LINK_RE = re.compile(r"^\s*\[([^\]]*)\]\(https?://(?:[^)\\]|\\.)*\)\s*$")

# First-person narration of a browse call — the #1480 per-tool override of the
# seam's generic RESULT_NARRATION_*.  The seam prepends ONE line for the WHOLE call
# and appends the `(browse result)` tag itself; the per-page `## browse ...:` /
# `## browse search:` / `## browse error:` section headers stay in the body and
# disambiguate each page.  So these summarise the batch as a single natural clause
# reflecting search-vs-read (a URL query is a direct read; plain text is a search)
# and the outcome (`result.success` is False only when EVERY query errored).
_NARRATION_SEARCHED = "You searched for {queries}"
_NARRATION_OPENED = "You opened {urls}"
_NARRATION_ALSO_OPENED = "and opened {urls}"
_NARRATION_LOOKED_UP = "You looked things up"
_NARRATION_FAILURE_SUFFIX = "but couldn't read anything"

# ── Micro-context (extract) render forms ──────────────────────────────────────
# When a browse call carries an ``extract`` instruction, the page body never
# enters the main loop: only the typed result (or an honest enumerated failure)
# plus the fetch handle to the full content stored in browse-results.  One
# render per ``MicroExtractOutcome``, plus the no-model-wired degradation.
# ``NOT_PRESENT`` is a successful read of an absent fact — rendered honestly,
# with no infrastructure-failure framing — distinct from ``EXTRACTION_FAILED``
# (the extractor never produced a usable tagged line).
_EXTRACT_HANDLE_CLAUSE = "Full page content saved to {handles} — read it there for anything more."
_EXTRACT_NO_HANDLE_CLAUSE = "The full page content was not separately stored."
_EXTRACT_SUCCESS = "{value}\n\n{handle_clause}"
_EXTRACT_NOT_PRESENT = "The page doesn't contain {instruction!r} — {reason} {handle_clause}"
_EXTRACT_FAILED = (
    "Couldn't extract {instruction!r} from the page — the extractor returned nothing "
    "usable. {handle_clause}"
)
_EXTRACT_POISON = (
    "Couldn't extract {instruction!r} from the page — the extractor output was unusable "
    "after {attempts} attempts. {handle_clause}"
)
_EXTRACT_UNAVAILABLE = (
    "Couldn't extract {instruction!r} — no extraction model is configured for this browse. "
    "{handle_clause}"
)

# Type alias for the browser request function
RequestFn = Callable[[str, dict], Awaitable[tuple[str, str | None]]]


def _trim_search_result(text: str, context_lines: int = 2) -> str:
    """Trim search result page to lines near standalone markdown links.

    Pipeline: filter to solo-link lines (drops knowledge panel prose),
    cap at MAX_SEARCH_LINKS, then keep context lines around each.
    """
    lines = text.split("\n")

    link_lines: list[int] = []
    for i, line in enumerate(lines):
        if _LINK_RE.match(line):
            link_lines.append(i)

    if not link_lines:
        return text

    keep: set[int] = set()
    for line_number in link_lines[: PennyConstants.MAX_SEARCH_LINKS]:
        for offset in range(-context_lines, context_lines + 1):
            idx = line_number + offset
            if 0 <= idx < len(lines):
                keep.add(idx)

    trimmed = "\n".join(lines[i] for i in sorted(keep))
    return f"{Prompt.SEARCH_RESULT_HEADER}\n\n{trimmed}"


class BrowseTool(Tool):
    """Search the web and read pages via the browser extension.

    The model emits one tool call with a queries array:
      {"queries": ["topic", "https://example.com", "another topic"]}
    URLs are read directly; plain text is converted to a search URL.
    All queries are dispatched in parallel.
    """

    name = "browse"
    args_model = BrowseArgs
    # Whole-tool executor budget.  Must comfortably exceed the per-URL
    # BROWSE_REQUEST_TIMEOUT across all retries so a slow/hung URL fires its own
    # per-attempt timeout — captured by ``asyncio.gather(return_exceptions=True)``
    # as a graceful error section — before the outer executor cancels the whole
    # browse call (which would surface as a blunt "Tool execution timeout").
    timeout = 300.0

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
    ):
        self._max_calls = max_calls
        self._search_url = search_url
        self._db = db
        self._embedding_client = embedding_client
        self._author = author
        # The shared model client powers the ``extract`` micro-context (a
        # single-shot extraction call).  Optional: chat's browse never sets
        # ``extract``, and tests that don't exercise it leave this None; an
        # ``extract`` requested without a client wired degrades visibly rather
        # than silently returning the page body.
        self._micro_context = MicroContext(model_client) if model_client is not None else None
        # The terminal move bound into a whole-channel outage error.  Defaults to
        # the chat clause; a collector passes its done()-binding clause so the
        # outage names the recovery its agent can actually perform.
        self._channel_outage_recovery = channel_outage_recovery
        self._browse_provider: Callable[[], tuple[RequestFn, PermissionManager] | None] | None = (
            None
        )

    @property
    def description(self) -> str:  # type: ignore[override]
        """Dynamic description reflecting current max_calls."""
        n = self._max_calls
        items = "query and/or URL" if n == 1 else "queries and/or URLs"
        return f"Look things up. Pass up to {n} {items}."

    @property
    def parameters(self) -> dict[str, Any]:  # type: ignore[override]
        """Dynamic parameters reflecting current max_calls."""
        n = self._max_calls
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
                        "Optional. One instruction naming exactly what to pull out of the "
                        'fetched pages (e.g. "the current bid amount"). When set, the full '
                        "page content is read in a separate scoped context and only the "
                        "extracted value is returned here — the page body never enters this "
                        "conversation. Omit to receive the page content itself."
                    ),
                },
            },
            "required": ["queries"],
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
                parts.append(f"Reading {short[:50]}")
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
        Branches on ``result.success``, which the tool reports False only when
        every dispatched query errored (a total failure), so a partial success
        still narrates the action it took.
        """
        queries = arguments.get("queries", [])
        searches = [q for q in queries if not _URL_PATTERN.match(q)]
        reads = [q for q in queries if _URL_PATTERN.match(q)]
        action = cls._browse_action_clause(searches, reads)
        if result.success:
            return action
        return f"{action} {_NARRATION_FAILURE_SUFFIX}"

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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch all lookups in parallel via the browser extension.

        With an ``extract`` micro-instruction (and at least one readable page),
        the fetched content goes to a fresh micro-context and only the typed
        result returns; otherwise the assembled sections return directly.
        """
        args = BrowseArgs(**kwargs)
        cap = self._max_calls
        to_run, dropped = args.queries[:cap], args.queries[cap:]

        tasks = self._build_tasks(to_run)
        results = await asyncio.gather(*[coro for _, _, coro in tasks], return_exceptions=True)
        sections, page_sections, captures = self._assemble_sections(tasks, results)

        stored = await self._append_pages_to_browse_results(page_sections)
        await self._store_media(captures)

        if dropped:
            sections.append(self._dropped_section(cap, dropped))
        # Browse is a *read* for work-accounting (its browse-results log write is
        # incidental), so ``mutated`` stays False.  But when EVERY dispatched query
        # errored the call did nothing, so it reports ``success=False`` — making the
        # failure visible to structural accounting (``record.failed`` → the run's
        # ``tool_failures`` count → run-health), not just to the error text.  A
        # partial failure keeps ``success=True``: the model works from what succeeded.
        all_failed = bool(results) and all(isinstance(r, BaseException) for r in results)
        if args.extract is not None and not all_failed:
            return await self._extract_result(sections, args.extract, stored)
        return ToolResult(
            message=PennyConstants.SECTION_SEPARATOR.join(sections),
            success=not all_failed,
        )

    # ── Micro-context (extract) path ──────────────────────────────────────────

    async def _extract_result(
        self, sections: list[str], instruction: str, stored: list[MemoryEntry]
    ) -> ToolResult:
        """Run the fetched content through the micro-context, returning the typed
        value + fetch handle — never the page body — to the main loop.

        The bulk content (page + search sections) feeds the micro-context; the
        short signal sections (read errors, dropped-query notes) stay visible to
        the main loop after the extracted value, so a read failure never hides
        behind a clean extraction (visible degradation)."""
        if self._micro_context is None:
            return self._extract_unavailable_result(instruction, stored)
        content = PennyConstants.SECTION_SEPARATOR.join(
            s for s in sections if self._is_content_section(s)
        )
        signals = [s for s in sections if not self._is_content_section(s)]
        micro = await self._micro_context.extract(content, instruction, run_target=self._author)
        body = self._render_micro_result(micro, instruction, stored)
        # NOT_PRESENT is a *successful read of an absent fact* — the page was
        # fetched and read; the fact isn't there.  Only the failure outcomes
        # (no usable tagged output / poison) report success=False.
        succeeded = micro.outcome in (
            MicroExtractOutcome.EXTRACTED,
            MicroExtractOutcome.NOT_PRESENT,
        )
        message = PennyConstants.SECTION_SEPARATOR.join([body, *signals])
        return ToolResult(message=message, success=succeeded)

    def _render_micro_result(
        self, micro: MicroContextResult, instruction: str, stored: list[MemoryEntry]
    ) -> str:
        """The main-loop body for one micro-context outcome — the extracted value
        or not-present reason (each byte-identical to the micro-context's return)
        or an honest enumerated failure, all carrying the fetch handle to the
        stored full content."""
        handle_clause = self._handle_clause(stored)
        if micro.outcome == MicroExtractOutcome.EXTRACTED:
            return _EXTRACT_SUCCESS.format(value=micro.value, handle_clause=handle_clause)
        if micro.outcome == MicroExtractOutcome.NOT_PRESENT:
            return _EXTRACT_NOT_PRESENT.format(
                instruction=instruction, reason=micro.reason, handle_clause=handle_clause
            )
        if micro.outcome == MicroExtractOutcome.EXTRACTION_FAILED:
            return _EXTRACT_FAILED.format(instruction=instruction, handle_clause=handle_clause)
        return _EXTRACT_POISON.format(
            instruction=instruction,
            attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS,
            handle_clause=handle_clause,
        )

    def _extract_unavailable_result(
        self, instruction: str, stored: list[MemoryEntry]
    ) -> ToolResult:
        """Honest degradation when ``extract`` is requested but no model client is
        wired for the micro-context — the page content is still stored and named
        by its handle, so the request fails visibly rather than dumping the body."""
        logger.error("browse(extract=...) requested but no model client is wired for extraction")
        message = _EXTRACT_UNAVAILABLE.format(
            instruction=instruction, handle_clause=self._handle_clause(stored)
        )
        return ToolResult(message=message, success=False)

    def _handle_clause(self, stored: list[MemoryEntry]) -> str:
        """The fetch-handle clause — the typed ``browse-results#<id>`` anchors for
        the pages stored this call, or a note when nothing was stored (a pure
        search has readable content but no stored page)."""
        if not stored:
            return _EXTRACT_NO_HANDLE_CLAUSE
        handles = ", ".join(self._entry_handle(entry) for entry in stored)
        return _EXTRACT_HANDLE_CLAUSE.format(handles=handles)

    @staticmethod
    def _entry_handle(entry: MemoryEntry) -> str:
        """One typed entry handle: ``browse-results#<id>``."""
        return (
            f"{PennyConstants.MEMORY_BROWSE_RESULTS_LOG}"
            f"{PennyConstants.MEMORY_HANDLE_SEPARATOR}{entry.id}"
        )

    @staticmethod
    def _is_content_section(section: str) -> bool:
        """A readable page/search section (bulk content for the micro-context), vs.
        a short signal section (error / dropped-query note) the main loop keeps."""
        return section.startswith(
            (PennyConstants.BROWSE_PAGE_HEADER, PennyConstants.BROWSE_SEARCH_HEADER)
        )

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
    ) -> tuple[list[str], list[str], list[BrowsePage]]:
        """Fold gathered results into rendered sections, the page-only subset (for
        the browse-results log), and the captured page images."""
        sections: list[str] = []
        page_sections: list[str] = []
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
            section = self._page_section(header, value, result)
            sections.append(section)
            if header == PennyConstants.BROWSE_PAGE_HEADER:
                page_sections.append(section)
                if result.image:
                    captures.append(result)
        if channel_outage:
            sections.append(self._channel_outage_section())
        return sections, page_sections, captures

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
    def _page_section(header: str, value: str, result: BrowsePage) -> str:
        """Render one successful sub-call; search pages are trimmed to their links."""
        text = result.text
        if header == PennyConstants.BROWSE_SEARCH_HEADER:
            text = _trim_search_result(text)
        return f"{header}{value}\n{text}"

    @staticmethod
    def _dropped_section(cap: int, dropped: list[str]) -> str:
        """Name the queries dropped past the per-call cap so the omission is visible
        to the model, with the exact recovery: rerun the rest in a follow-up call."""
        ran = "query was" if cap == 1 else f"{cap} queries were"
        count = len(dropped)
        dropped_list = ", ".join(repr(q) for q in dropped)
        return (
            f"{PennyConstants.BROWSE_DROPPED_HEADER}only the first {ran} run; "
            f"{count} beyond the {cap}-per-call limit were not run: {dropped_list}. "
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
                text, image_url = await asyncio.wait_for(
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
            return BrowsePage(text=text, image=image_url, title=title, url=url)

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
