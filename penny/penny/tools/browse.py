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
from penny.tools.models import BrowseArgs, BrowsePage, ToolResult

if TYPE_CHECKING:
    from penny.channels.permission_manager import PermissionManager
    from penny.database import Database
    from penny.llm import LlmClient

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"^https?://")

_LINK_RE = re.compile(r"^\s*\[([^\]]*)\]\(https?://(?:[^)\\]|\\.)*\)\s*$")

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
        embedding_client: LlmClient | None = None,
        author: str = "unknown",
    ):
        self._max_calls = max_calls
        self._search_url = search_url
        self._db = db
        self._embedding_client = embedding_client
        self._author = author
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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch all lookups in parallel via the browser extension."""
        args = BrowseArgs(**kwargs)

        cap = self._max_calls
        tasks: list[tuple[str, str, Any]] = []
        for q in args.queries[:cap]:
            if _URL_PATTERN.match(q):
                tasks.append((PennyConstants.BROWSE_PAGE_HEADER, q, self._read_page(q)))
            else:
                search_url = self._search_url + urllib.parse.quote(q)
                tasks.append((PennyConstants.BROWSE_SEARCH_HEADER, q, self._read_page(search_url)))

        results = await asyncio.gather(*[coro for _, _, coro in tasks], return_exceptions=True)

        sections: list[str] = []
        page_sections: list[str] = []
        captures: list[BrowsePage] = []
        for (header, value, _), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("Browse sub-call failed (%s%s): %s", header, value, result)
                error_label = f"{PennyConstants.BROWSE_ERROR_HEADER}{value}"
                sections.append(
                    f"{error_label}\nCould not read this page: {result}. "
                    f"Try a different source or a reworded query; if other queries in this "
                    f"batch succeeded, work from those instead of retrying this one."
                )
                continue
            label = f"{header}{value}"
            text = result.text
            if header == PennyConstants.BROWSE_SEARCH_HEADER:
                text = _trim_search_result(text)
            section = f"{label}\n{text}"
            sections.append(section)
            if header == PennyConstants.BROWSE_PAGE_HEADER:
                page_sections.append(section)
                if result.image:
                    captures.append(result)

        await self._append_pages_to_browse_results(page_sections)
        await self._store_media(captures)
        # Browse is a *read* for work-accounting (its browse-results log write is
        # incidental), so ``mutated`` stays False — a collector cycle that only
        # browsed found nothing to record.
        return ToolResult(message=PennyConstants.SECTION_SEPARATOR.join(sections))

    async def _append_pages_to_browse_results(self, page_sections: list[str]) -> None:
        """Side-effect-write each successful page as its own log entry.

        Search-result and error sections are skipped — only full page
        reads carry knowledge worth indexing.  Embeds each entry at
        write time so similarity recall (and the knowledge extractor)
        can address pages individually.
        """
        if self._db is None or not page_sections:
            return
        entries: list[LogEntryInput] = []
        for section in page_sections:
            vec = await embed_text(self._embedding_client, section)
            entries.append(LogEntryInput(content=section, content_embedding=vec))
        browse_log = self._db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
        if browse_log is not None:
            browse_log.append(entries, author=self._author)

    async def _read_page(self, url: str) -> BrowsePage:
        """Read a single URL via the browser extension, retrying with backoff on disconnect.

        Raises ConnectionError if no browser is reachable after all retries, and
        propagates any RuntimeError raised by the browser extension itself (which
        signals a structured failure: extraction failed, page never became ready,
        host permission denied, etc.).
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
                raise ConnectionError(
                    "no browser is connected — the browser extension isn't running, so web "
                    "search and page reads are unavailable; answer from what you already know "
                    "or tell the user the browser is offline"
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
            except ConnectionError:
                if attempt < PennyConstants.BROWSE_RETRIES:
                    logger.info(
                        "Browser disconnected, retrying in %.0fs (%s)",
                        delay,
                        url,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            title = self._parse_title(text)
            text = clean_browser_content(text)
            return BrowsePage(text=text, image=image_url, title=title, url=url)

        raise ConnectionError("no browser connected")

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
