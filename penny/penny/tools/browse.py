"""BrowseTool — searches, picks a result with the decision model, and reads it,
in one call (#2718).

The model packs everything into a single ``items`` array; each item is a search,
a URL, or a link citation, and carries its own "looking for".  A URL is read
directly; a search runs the search, ranks the results behind the tool boundary
(:mod:`penny.tools.browse_picker`), reads the top-ranked result, and falls
through to the next one while the read finds nothing it was asked for — so the
model always gets what the pages say and never a list of titles to choose from.
Items run in batches of ``max_calls``, one batch after another, up to the
per-call total ceiling.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import urllib.parse
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from penny.constants import PageReading, PennyConstants, ProgressEmoji
from penny.database.memory import LogEntryInput
from penny.llm.call_link import CallLink
from penny.llm.embeddings import serialize_embedding
from penny.llm.similarity import embed_text
from penny.prompts import Prompt
from penny.tools.base import Tool
from penny.tools.browse_picker import PickOutcome, ResultPicker
from penny.tools.content_cleaning import clean_browser_content
from penny.tools.link_handles import PAGE_CITATION_LINE, HandleTable
from penny.tools.micro_context import MicroContext, MicroExtractOutcome
from penny.tools.models import BrowseArgs, BrowseItem, BrowsePage, ToolResult

if TYPE_CHECKING:
    from penny.channels.permission_manager import PermissionManager
    from penny.database import Database
    from penny.database.models import MemoryEntry
    from penny.llm import LlmClient
    from penny.llm.decision_client import DecisionClient

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
    that loop and to let the item pipeline tell a channel outage apart from a
    page failure by ``isinstance``, not by string-matching the message.
    """


_URL_PATTERN = re.compile(r"^https?://")

# A query that is nothing but a link citation (#2717) — read as the URL its
# handle stands for, never searched for literally.
_HANDLE_QUERY_RE = re.compile(r"^\[(\d+)\]$")


def _is_read_query(query: str) -> bool:
    """Whether a query names a page to read: a URL, or a link citation."""
    return bool(_URL_PATTERN.match(query) or _HANDLE_QUERY_RE.match(query))


def _arg_items(arguments: dict) -> list[dict]:
    """The ``items`` of a raw call's arguments, as dicts — the class-level
    narrators (status text, emoji, synthetic-result narration) read arguments
    before validation normalised them, so both shapes arrive here."""
    raw = arguments.get("items", [])
    return [item if isinstance(item, dict) else item.model_dump() for item in raw]


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

# ── Search-item render forms (#2718) ─────────────────────────────────────────
# A search item's own section: what the item was looking for, then what the call
# did with it — the pages it read (each page's own section follows, carrying its
# value), or the honest dead ends.
# A call recorded in the DEAD dialect (``queries`` + one shared ``extract``,
# pre-#2718) is rejected at the arg gate — and the rejection carries the exact
# ``items`` call that would succeed, built from its own arguments: the model
# copies the fix instead of guessing at the shape.  History recorded in the old
# dialect ages out; the rejection teaches the new one in one round trip.
_DEAD_DIALECT_TRANSLATION = (
    "browse takes items now, one per lookup, each with what it is looking for. "
    "That call, translated, is: browse(items={items}) — pass it like that."
)
_SEARCH_LOOKING_LINE = "Looking for: {looking_for}."
_SEARCH_NONE_BODY = "None of the results looked likely to carry it, so nothing was read."
# The picker's honest escape (no usable answer) renders the trimmed results list
# under the search section's own header — the one shape where the model still
# sees titles and link handles, to pick from itself in a follow-up call.

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
# The extract output cap (#2718): a value longer than ``BROWSE_EXTRACT_MAX_CHARS``
# is cut at the last line boundary under the cap, and the render says how much
# was left out, with the fetch handle to the stored full page — degradation
# says what it left out and where the rest is (§2.8).
_EXTRACT_CUT_LINE = (
    "(…cut — {omitted} more characters of the page's answer were left out. {handle_clause})"
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


class _ItemOutcome(NamedTuple):
    """One item's finished work, gathered back in item order — the sections it
    rendered (its search section first, then each page it read), the pages it
    actually read, named as the model reads them (for the narration), the
    captured page images, whether a
    whole-channel outage ended its reads, and whether its primary fetch errored
    (the call's ``success=False`` counts only total failure)."""

    sections: list[str] = []
    opened: list[str] = []
    captures: list[BrowsePage] = []
    outage: bool = False
    failed: bool = False


def _parse_search_results(text: str, search_host: str | None) -> list[tuple[str, str]]:
    """Parse a search result page into its ``(title, url)`` results, in page order.

    Each result is its title (the heading line above its link) and the link's own
    target, and nothing below — a result's summary snippet often carries the
    answer itself, and a pick made from snippets is a pick from pages never
    opened.  Real results carry a heading (``### Title``); a standalone link
    under anything else — a knowledge-panel paragraph, an infobox row — is not a
    result and is dropped (#2697).  The search engine's own affordance links (its
    summariser, its maps) point back at the search host and are not results, so
    they are dropped the same way the snippet lines are.

    A page whose standalone links all sit under non-heading lines (a search
    engine that dropped its headings) yields those links, each titled by its own
    link text (#2697) — the picker ranks titles, never knowledge-panel prose.  A
    page with no standalone links at all yields an empty list — the "nothing
    found" path.

    The URLs here are for the PICKER (behind the tool boundary) and the reads —
    they are never rendered to the main loop except as handles (#2717), which
    :meth:`_render_search_list` does on the escape path alone."""
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
    return results[: PennyConstants.MAX_SEARCH_LINKS]


def _render_search_list(query: str, results: list[tuple[str, str]], handles: HandleTable) -> str:
    """The trimmed results list — the picker's honest escape render (#2718), the
    one shape where the model still sees titles and links.  Each result's link
    renders as its handle (#2717): the model opens one by citing the number back
    in a follow-up query, never by retyping the URL."""
    listed = "\n".join(
        f"{number}. {title} [{handles.register(url)}]"
        for number, (title, url) in enumerate(results, 1)
    )
    instruction = Prompt.SEARCH_RESULT_INSTRUCTION.format(query=query)
    return f"{instruction}\n\n{listed}"


def _url_host(url: str) -> str:
    """The host of a link target, lowercased — the one part the engine-affordance
    filter reads."""
    return urllib.parse.urlparse(url).hostname or ""


def _links_clause(text: str, handles: HandleTable) -> str:
    """The standalone markdown links in ``text``, capped at MAX_SEARCH_LINKS, as the
    tail a failed extraction appends — empty when the page carries none.

    Same solo-link detection the search trim uses (``_LINK_RE``): a line that is
    nothing but a link is a link the model can open, and prose merely containing one
    is not.  Each link renders as its handle (#2717) — the only form the model can
    cite it in.
    """
    links = [line.strip() for line in text.split("\n") if _LINK_RE.match(line)]
    if not links:
        return ""
    listed = "\n".join(handles.convert(link) for link in links[: PennyConstants.MAX_SEARCH_LINKS])
    return f"\n{_EXTRACT_LINKS_HEADER}\n{listed}"


class BrowseTool(Tool):
    """Search the web, pick a result with the decision model, and read it — in
    one call (#2718).

    The model emits one tool call with an items array; each item is a search, a
    URL, or a link citation, and carries its own "looking for":
      {"items": [{"query": "topic", "looking_for": "the opening hours"},
                 {"query": "https://example.com", "looking_for": "the price"}]}
    A URL is read directly; a search runs, the picker ranks the results behind
    the tool boundary, and the top-ranked result is read and extracted — falling
    through to the next ranked result while the read finds nothing it was asked
    for.  Every item up to the per-call total ceiling runs in this call, in
    parallel batches of ``max_calls`` — the browser-tab limit — one batch after
    another.  Items past the total come back named in a dropped section.
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
        db: Database,
        search_url: str = "https://duckduckgo.com/?q=",
        author: str = "unknown",
        *,
        embedding_client: LlmClient,
        model_client: LlmClient | None = None,
        decision_client: DecisionClient | None = None,
        channel_outage_recovery: str = Prompt.BROWSE_OUTAGE_RECOVERY_CHAT,
        max_queries_per_call: int | None = None,
        handles: HandleTable,
    ):
        # The run's numbered link table (#2717) — REQUIRED, no default: every URL
        # this tool renders to the model is registered here as a handle, and the
        # queries the model cites back are resolved through it.  One table per
        # run, owned by the run starter and passed explicitly — never ambient
        # state — so a handle the model read from one tool result is the same
        # number it cites in its reply.
        self._handles = handles
        self._max_calls = max_calls
        # The per-call TOTAL ceiling (#2662): every requested item up to this runs
        # in this call, in parallel batches of ``max_calls``.  Unwired constructions
        # (older tests, tools built before the setting exists) fall back to the
        # batch size itself, so a tool without a ceiling behaves exactly as the
        # pre-batching tool did.
        self._max_queries_per_call = (
            max_calls if max_queries_per_call is None else (max_queries_per_call)
        )
        self._search_url = search_url
        self._search_host = _url_host(search_url)
        # REQUIRED — no default, no ``None`` (#2702): the store browse results log
        # to, and the one the micro-context's discarded draws are marked on.
        self._db = db
        self._embedding_client = embedding_client
        self._author = author
        # The shared model client powers the ``extract`` micro-context (a
        # single-shot extraction call).  Optional: tests that don't exercise it
        # leave this None; an extraction without a client wired degrades visibly
        # rather than silently returning the page body.
        self._micro_context = (
            MicroContext(model_client, db=db) if model_client is not None else None
        )
        # The result picker (#2718): which search result to read, decided behind
        # the tool boundary.  ``decision_client`` is the entry-identity ladder's
        # decision client — unset runs the picker's named degraded mode (the
        # shared text model picks with thinking off), named in the preflight.
        self._picker = ResultPicker(decision_client, model_client)
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
        # for now (the model is steered toward one call per turn via the items
        # array), not guarded.
        self._read_log: list[tuple[str, MemoryEntry | None]] = []
        self._extract_log: list[tuple[str, str]] = []
        self._extracting = False

    @property
    def description(self) -> str:  # type: ignore[override]
        """Dynamic description reflecting the per-call total ceiling."""
        n = self._max_queries_per_call
        noun = "item" if n == 1 else "items"
        return (
            f"Look things up. Pass up to {n} {noun}; each is a search, a link "
            "number like [4], or a URL, with what it is looking for. A search is "
            "picked and read in this same call."
        )

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
                "items": {
                    "type": "array",
                    "description": (
                        f"Lookups to run (max {n}). Each is a search, a link number "
                        "like [4], or a URL, with what it is looking for."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A search, a link number like [4], or a URL.",
                            },
                            "looking_for": {
                                "type": "string",
                                "description": (
                                    "What the page must carry, in plain language. Steers "
                                    "which search result is picked and what is pulled out "
                                    'of the page — "the opening hours" or "anything the '
                                    'page says about refunds" are complete asks.'
                                ),
                            },
                        },
                        "required": ["query", "looking_for"],
                    },
                    "maxItems": n,
                },
            },
            "required": ["items"],
        }

    async def run(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Override of ``Tool.run``: a call in the dead dialect (``queries`` +
        one shared ``extract``, the pre-#2718 shape history still carries) is
        rejected with the exact ``items`` call built from its own arguments —
        before the generic validation error, which would name unknown fields
        but not show the call that would succeed (actionable-failure: the fix
        is a copy, not another guess)."""
        if isinstance(kwargs.get("queries"), list) and "items" not in kwargs:
            items = [
                {"query": query, "looking_for": kwargs.get("extract", "")}
                for query in kwargs["queries"]
                if isinstance(query, str)
            ]
            return ToolResult(
                success=False,
                message=_DEAD_DIALECT_TRANSLATION.format(items=json.dumps(items)),
            )
        return await super().run(call_link, **kwargs)

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
        for item in _arg_items(arguments):
            q = item.get("query", "")
            if _is_read_query(q):
                short = q.replace("https://", "").replace("http://", "")
                parts.append(f"Reading {short}")
            else:
                parts.append(f'Searching "{q}"')
        return "<br>".join(parts) if parts else "Looking up..."

    @classmethod
    def to_progress_emoji(cls, arguments: dict) -> ProgressEmoji:
        """Pick 📖 if any item is a URL or a link citation (reading), 🔍 otherwise."""
        for item in _arg_items(arguments):
            if _is_read_query(item.get("query", "")):
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
        ``ToolResult.narration`` (built from the items actually DISPATCHED and
        the pages actually opened, #2661), so the seam never reaches here for one
        — this override answers only for a result built OUTSIDE the tool (a
        synthetic seeded result, which has no cap to have dropped anything past)
        and narrates the requested items.  Branches on ``result.success``, which
        the tool reports False only when every dispatched item errored (a total
        failure), so a partial success still narrates the action it took.
        """
        return cls._result_narration(
            [item.get("query", "") for item in _arg_items(arguments)], result.success
        )

    @classmethod
    def _result_narration(cls, queries: list[str], success: bool) -> str:
        """The narration line for ``queries`` — the ONE place the action clause and
        the failure suffix meet.  Callers pass the list to NARRATE: the requested
        queries for a synthetic result with no dispatched list behind it.  A link
        citation (#2717) narrates as the read it was dispatched as, in the handle
        form the model reads.  A search item's opened pages are NOT named here —
        a synthetic result has none behind it; the tool-composed narration names
        them."""
        searches = [q for q in queries if not _is_read_query(q)]
        reads = [q for q in queries if _is_read_query(q)]
        action = cls._browse_action_clause(searches, reads)
        if success:
            return action
        return f"{action} {NARRATION_FAILURE_SUFFIX}"

    @staticmethod
    def _browse_action_clause(searches: list[str], reads: list[str]) -> str:
        """Compose the "You searched for … and opened …" action clause from the
        split lookups — searches quoted, opened pages in their handle form (the
        model-facing form everywhere else in the result)."""
        parts: list[str] = []
        if searches:
            quoted = ", ".join(f'"{q}"' for q in searches)
            parts.append(_NARRATION_SEARCHED.format(queries=quoted))
        if reads:
            template = _NARRATION_ALSO_OPENED if searches else _NARRATION_OPENED
            parts.append(template.format(urls=", ".join(reads)))
        return " ".join(parts) if parts else _NARRATION_LOOKED_UP

    async def execute(self, call_link: CallLink, **kwargs: Any) -> ToolResult:
        """Run every requested lookup (up to the per-call total), in parallel
        batches of ``max_calls``.

        Each item runs whole (#2718): a URL is read and extracted; a search is
        run, the picker ranks its results behind the tool boundary, and the
        ranked results are read and extracted — falling through while the read
        finds nothing it was asked for.  An item's sections render in its own
        order (the search section, then the pages it read); the call's sections
        render in item order.

        ``call_link`` (#2458) is THIS dispatched call's own linkage — the tool
        dispatcher built it via ``CallLink.for_tool_call(tool_call_id)`` before
        ``run`` even validated args. Threaded through to every picker question
        and extraction draw this call makes, so each one links back to the run
        AND to this dispatched call.

        Every result returned here carries its own ``narration`` (#2661), composed
        from the items actually dispatched and the pages actually opened — so the
        first-person line never names a page past the per-call cap that was never
        fetched; the ``## browse dropped:`` section stays the only place the rest
        are named."""
        args = BrowseArgs(**kwargs)
        self._read_log, self._extract_log, self._extracting = [], [], True
        total = self._max_queries_per_call
        requested, dropped = args.items[:total], args.items[total:]
        # Handles resolve here (#2717): a query that is a bare citation is read as
        # the URL its handle stands for, and a handle the table does not hold is
        # refused honestly instead of searched for literally.
        resolved, dangling = self._resolve_items(requested)

        outcomes: list[_ItemOutcome] = []
        channel_outage = False
        # Batching (#2662): the parallel cap bounds how many ITEMS are in flight,
        # not how many the call runs — the requested items up to the TOTAL ceiling
        # go through in batches of ``max_calls``, one batch after another, so a
        # four-site job pays no extra model round trip.  Order is preserved: each
        # item's sections keep their own order, and the batches run in order.
        for start in range(0, len(resolved), self._max_calls):
            batch = resolved[start : start + self._max_calls]
            gathered = await asyncio.gather(
                *[self._run_item(item, dispatch, call_link) for item, dispatch in batch],
                return_exceptions=True,
            )
            for outcome in gathered:
                if isinstance(outcome, BaseException):
                    raise outcome
            batch_outcomes = cast("list[_ItemOutcome]", gathered)
            outcomes.extend(batch_outcomes)
            channel_outage = channel_outage or any(outcome.outage for outcome in batch_outcomes)

        sections: list[str] = []
        opened: list[str] = []
        captures: list[BrowsePage] = []
        failed = 0
        for outcome in outcomes:
            sections.extend(outcome.sections)
            opened.extend(outcome.opened)
            captures.extend(outcome.captures)
            failed += 1 if outcome.failed else 0
        if channel_outage:
            sections.append(self._channel_outage_section())
        await self._store_media(captures)

        if dropped:
            sections.append(self._dropped_section(total, dropped))
        for item in dangling:
            sections.append(self._dangling_query_section(item.query))
        # Browse is a *read* for work-accounting (its browse-results log write is
        # incidental), so ``mutated`` stays False.  But when EVERY dispatched item's
        # fetch errored the call did nothing, so it reports ``success=False`` — making
        # the failure visible to structural accounting (``record.failed`` → the run's
        # ``tool_failures`` count → run-health), not just to the error text.  A
        # partial failure keeps ``success=True``: the model works from what succeeded.
        all_failed = bool(outcomes) and failed == len(outcomes)
        searches = [item.query for item, _ in resolved if not _is_read_query(item.query)]
        return ToolResult(
            message=self._model_message(sections),
            success=not all_failed,
            source_urls=self._read_urls(sections),
            narration=self._narration(searches, opened, not all_failed),
        )

    def _narration(self, searches: list[str], opened: list[str], success: bool) -> str:
        """The narration line for a result the tool itself built (#2661): the
        searches it dispatched (quoted) and the pages it OPENED — URL reads and
        picked search reads alike, in their handle form (#2717) — never a page
        past the cap or a result the picker never read."""
        action = self._browse_action_clause(searches, opened)
        if success:
            return action
        return f"{action} {NARRATION_FAILURE_SUFFIX}"

    @staticmethod
    def _read_urls(sections: list[str]) -> list[str]:
        """The URL of every page THIS call actually read successfully, in read
        order — taken off each page section's own ``## browse: <url>`` header
        value, the one place the tool already records it.  A failed read
        (``## browse error:``) read nothing and a search section (``## browse
        search:``) is links, not a page read, so neither contributes (#2648):
        these ride ``ToolResult.source_urls`` to the reply-side fallback, which
        appends the first one when the composed reply carries no URL.

        Read off the REAL sections — the sections this call renders to the model
        carry handles (#2717), so the URL lives here and in the table, never in
        the model-facing text."""
        return [
            section[len(PennyConstants.BROWSE_PAGE_HEADER) :].partition("\n")[0]
            for section in sections
            if section.startswith(PennyConstants.BROWSE_PAGE_HEADER)
        ]

    def _resolve_items(
        self, items: list[BrowseItem]
    ) -> tuple[list[tuple[BrowseItem, str]], list[BrowseItem]]:
        """Split the requested items into what dispatches and what dangles.

        A bare-citation query (``"[12]"``) dispatches as the URL its handle
        stands for; anything else dispatches as itself.  A citation the run's
        table does not hold returns in the dangling list, to be refused with the
        honest note naming the links that DO exist — searched for literally, a
        ``"[12]"`` would be a garbage query about the number 12."""
        resolved: list[tuple[BrowseItem, str]] = []
        dangling: list[BrowseItem] = []
        for item in items:
            match = _HANDLE_QUERY_RE.match(item.query)
            if match is None:
                resolved.append((item, item.query))
                continue
            url = self._handles.url_for(int(match.group(1)))
            if url is None:
                dangling.append(item)
            else:
                resolved.append((item, url))
        return resolved, dangling

    def _dangling_query_section(self, query: str) -> str:
        """The honest refusal of a citation this run's table does not hold —
        which numbers don't exist and which do, so the re-draw cites a real one."""
        match = _HANDLE_QUERY_RE.match(query)
        if match is None:  # only reachable from _resolve_queries' own match
            return ""
        number = int(match.group(1))
        return (
            f"{PennyConstants.BROWSE_ERROR_HEADER}{query}\n{self._handles.dangling_note([number])}"
        )

    def _model_message(self, sections: list[str]) -> str:
        """The model-facing join of this call's sections — every URL arrives as
        its handle (#2717).  The real sections (whose headers and bodies carry
        the URLs the table, the browse-results log and ``source_urls`` read)
        convert at this one boundary; conversion is idempotent, so an
        already-converted extract render passes through unchanged.  The one
        plain line telling the model how to cite rides once, at the end, when
        this call rendered any link at all."""
        converted = [self._handles.convert_browse_section(section) for section in sections]
        body = PennyConstants.SECTION_SEPARATOR.join(converted)
        if self._handles:
            body = f"{body}\n\n{PAGE_CITATION_LINE}"
        return body

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
        (or a note that it wasn't stored) — labels converted to their handles,
        the same model-facing form the live sections render (#2717)."""
        listed = "; ".join(
            f"{self._handles.convert_browse_section(label)} "
            f"({self._entry_handle(entry) if entry else _TIMEOUT_NOT_STORED})"
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
            rendered = PennyConstants.SECTION_SEPARATOR.join(
                self._handles.convert_browse_section(r) for _, r in self._extract_log
            )
            parts.append(_TIMEOUT_EXTRACT_FINISHED.format(rendered=rendered))
        if waiting:
            waiting_labels = "; ".join(
                self._handles.convert_browse_section(label) for label in waiting
            )
            parts.append(_TIMEOUT_EXTRACT_WAITING.format(labels=waiting_labels))
        return "\n\n".join(parts)

    @staticmethod
    def _page_label(section: str) -> str:
        """A page section's own header line — the one identifier used for BOTH the
        read log and the extract log, so a page is named the same way in either
        (#2367) without re-deriving it at each call site."""
        return section.partition("\n")[0]

    # ── The per-item pipeline (#2718) ─────────────────────────────────────────

    async def _run_item(self, item: BrowseItem, dispatch: str, call_link: CallLink) -> _ItemOutcome:
        """Run ONE item whole: a URL (or link citation) is read and extracted; a
        search is run, its results ranked behind the tool boundary, and the
        ranked results read and extracted — falling through while the read finds
        nothing it was asked for.  Each read's section renders under the page's
        OWN header, so a multi-page account keeps every value attributed to its
        page whatever order the concurrent items finish in."""
        if _is_read_query(item.query):
            return await self._run_read_item(item, dispatch, call_link)
        return await self._run_search_item(item, call_link)

    async def _run_read_item(self, item: BrowseItem, url: str, call_link: CallLink) -> _ItemOutcome:
        """One URL (or resolved citation) item: fetch, extract, render under the
        page's own header.  A failed fetch renders its own error section."""
        try:
            page = await self._read_page(url)
        except BrowseChannelUnavailableError:
            return _ItemOutcome(outage=True, failed=True)
        except Exception as exc:
            return _ItemOutcome(
                sections=[self._error_section(PennyConstants.BROWSE_PAGE_HEADER, url, exc)],
                failed=True,
            )
        rendered, text = await self._extract_and_render(page, item.looking_for, call_link)
        captures = [page] if page.image else []
        # The page was fetched, but a read the extraction never answered is a read
        # that returned nothing usable — a total failure for this item (the fetch
        # succeeded, so this is the extraction half of ``_read_anything``'s old
        # "did anything readable come back", #2141).  The narration names the page
        # AS THE MODEL GAVE IT (#2735) — its URL, or the citation it wrote.
        return _ItemOutcome(
            sections=[rendered],
            captures=captures,
            opened=[item.query],
            failed=not text,
        )

    async def _run_search_item(self, item: BrowseItem, call_link: CallLink) -> _ItemOutcome:
        """One search item: run the search, rank its results behind the tool
        boundary, and read the ranked results — falling through while a read
        finds nothing it was asked for, up to ``BROWSE_MAX_PICKED_READS``.  The
        item's section comes first (what it was looking for, what the call did);
        each read's page section follows it, carrying what that page held.  A
        picker that answered nothing usable renders the results list as the
        escape — the one shape where the model still sees titles and handles."""
        search_url = f"{self._search_url}{urllib.parse.quote(item.query)}"
        try:
            page = await self._read_page(search_url)
        except BrowseChannelUnavailableError:
            return _ItemOutcome(outage=True, failed=True)
        except Exception as exc:
            return _ItemOutcome(
                sections=[
                    self._error_section(PennyConstants.BROWSE_SEARCH_HEADER, item.query, exc)
                ],
                failed=True,
            )
        looking_line = _SEARCH_LOOKING_LINE.format(looking_for=item.looking_for)
        header = f"{PennyConstants.BROWSE_SEARCH_HEADER}{item.query}"
        results = _parse_search_results(page.text, self._search_host)
        if not results:
            # The "nothing found" page keeps its own shape, under the search
            # header — same as before the picker existed.
            return _ItemOutcome(sections=[f"{header}\n{looking_line}\n{page.text}"])
        pick = await self._picker.rank(
            query=item.query,
            looking_for=item.looking_for,
            results=[(n, title, url) for n, (title, url) in enumerate(results, 1)],
            call_link=call_link,
        )
        if pick.outcome is PickOutcome.FAILED:
            body = f"{looking_line}\n{_render_search_list(item.query, results, self._handles)}"
            return _ItemOutcome(sections=[f"{header}\n{body}"])
        if pick.outcome is PickOutcome.NONE:
            # "None of these" reads nothing — but the model is still left the
            # results list: the picker's none is one model's judgement, and the
            # measured alternative was a follow-up search reworded until the
            # same page came back with no anchor to act on (n≤1).  The list is
            # the honest state: nothing was read, and here is what exists.
            body = (
                f"{looking_line}\n{_SEARCH_NONE_BODY}\n"
                f"{_render_search_list(item.query, results, self._handles)}"
            )
            return _ItemOutcome(sections=[f"{header}\n{body}"])
        sections, opened, outage = await self._read_ranked(item, results, pick.ranked, call_link)
        # No "Read:" line here: each page tried renders its own section directly
        # below, and its header names the page by its address (#2735 — a page's
        # own address takes no handle number, so a bare address line here would
        # be handle-converted into exactly the number the page must not take).
        # ``opened`` feeds the narration, which names pages un-converted.
        return _ItemOutcome(
            sections=[f"{header}\n{looking_line}", *sections], opened=opened, outage=outage
        )

    async def _read_ranked(
        self,
        item: BrowseItem,
        results: list[tuple[str, str]],
        ranked: list[int],
        call_link: CallLink,
    ) -> tuple[list[str], list[str], bool]:
        """Read the ranked results, falling through while a read finds nothing it
        was asked for, up to ``BROWSE_MAX_PICKED_READS``.  Returns the sections
        (one per read, in read order), the URLs of the pages that were actually
        read, and whether a whole-channel outage ended the reads.  A read failure
        is a candidate that yielded nothing — its own error section, then the next
        ranked result.  An extraction that never came back usable falls through
        the same way; an extraction that came back is judged by the picker's
        carry question — a "no" (an unreadable answer counts as yes) falls
        through too, and every page tried still reports what it held."""
        sections: list[str] = []
        opened: list[str] = []
        outage = False
        for number in ranked[: PennyConstants.BROWSE_MAX_PICKED_READS]:
            url = results[number - 1][1]
            # The run itself cites the page it is about to open: the handle is
            # assigned at pick time, before the fetch, so the page's section
            # header carries the number the table already holds (#2735's
            # "as it was cited" arm — the run is the one doing the citing), and
            # the model reads a stable anchor for the page its value came from.
            # The page's number precedes its content links, so none of theirs is
            # displaced; the echo inside the value stays gone (#2740).
            opened.append(f"[{self._handles.register(url)}]")
            try:
                page = await self._read_page(url)
            except BrowseChannelUnavailableError:
                outage = True
                break
            except Exception as exc:
                sections.append(self._error_section(PennyConstants.BROWSE_ERROR_HEADER, url, exc))
                continue
            rendered, text = await self._extract_and_render(page, item.looking_for, call_link)
            sections.append(rendered)
            if text and await self._picker.carries_any(
                looking_for=item.looking_for, extract_text=text, call_link=call_link
            ):
                break
        return sections, opened, outage

    async def _extract_and_render(
        self, page: BrowsePage, instruction: str, call_link: CallLink
    ) -> tuple[str, str]:
        """Extract ONE fetched page's content and render the typed result under
        that page's own ``## browse …:`` header (#1682) — the plain-text answer
        (#2615) or an honest failure.  Returns the rendered section and the
        extracted text (empty on any failure): the text is what the search-item
        fall-through reads, through the picker's carry question.

        The page is stored WHOLE to browse-results first (its entry id is the
        fetch handle the render anchors to), and both progress logs are written
        before the draw, toward this call's timeout report (#2367)."""
        section = f"{PennyConstants.BROWSE_PAGE_HEADER}{page.url}\n{page.text}"
        converted = self._handles.convert_browse_section(section)
        # The failure renders' links clause reads the RAW section — its markdown
        # links are still ``[text](url)`` there; the converted section's are
        # already handles and would yield nothing.
        links = _links_clause(section, self._handles)
        entry = await self._store_one_page(section)
        stored = [entry] if entry is not None else []
        label = self._page_label(section)
        self._read_log.append((label, entry))
        body, text = await self._extract_body(
            converted, page.reading, instruction, stored, call_link, links
        )
        rendered = self._under_page_header(section, body)
        self._extract_log.append((label, rendered))
        return rendered, text

    async def _extract_body(
        self,
        converted: str,
        reading: PageReading,
        instruction: str,
        stored: list[MemoryEntry],
        call_link: CallLink,
        links: str,
    ) -> tuple[str, str]:
        """One page's extract BODY and its usable text.  A draw that RAISES (a
        transport failure, say) renders like the other failure outcomes, under
        the page's own header (#2367) — deliberately ``Exception``, never
        ``BaseException``, so a ``CancelledError`` still propagates untouched.
        The no-model degradation renders its own named failure."""
        handle_clause = self._handle_clause(stored)
        if self._micro_context is None:
            named = _EXTRACT_UNAVAILABLE.format(
                instruction=instruction, handle_clause=handle_clause
            )
            return f"{named}{links}", ""
        try:
            micro = await self._micro_context.extract(
                converted,
                instruction,
                call_link=call_link,
                run_target=self._author,
                reading=reading,
            )
        except Exception as exc:
            named = _EXTRACT_DRAW_ERROR.format(
                instruction=instruction, error=exc, handle_clause=handle_clause
            )
            return f"{named}{links}", ""
        if micro.outcome is not MicroExtractOutcome.EXTRACTED:
            failure = _EXTRACT_FAILED.format(
                instruction=instruction,
                attempts=PennyConstants.DEGENERATE_REROLL_ATTEMPTS,
                handle_clause=handle_clause,
            )
            return f"{failure}{links}", ""
        text = self._cap_extract(micro.value, handle_clause)
        return _EXTRACT_SUCCESS.format(instruction=instruction, value=text), text

    @staticmethod
    def _cap_extract(value: str, handle_clause: str) -> str:
        """The extract output cap (#2718): a value longer than
        ``BROWSE_EXTRACT_MAX_CHARS`` is cut at the last line boundary under the
        cap, and the render says how much was left out, with the fetch handle to
        the stored full page — degradation names what it left out (§2.8)."""
        cap = PennyConstants.BROWSE_EXTRACT_MAX_CHARS
        if len(value) <= cap:
            return value
        cut = value[:cap]
        newline = cut.rfind("\n")
        if newline > 0:
            cut = cut[:newline]
        cut_line = _EXTRACT_CUT_LINE.format(
            omitted=len(value) - len(cut), handle_clause=handle_clause
        )
        return f"{cut}\n{cut_line}"

    async def _store_one_page(self, section: str) -> MemoryEntry | None:
        """Side-effect-write ONE successful page as its own log entry, returning
        it (its id is the fetch handle a render anchors its extracted value to).
        Embeds at write time so similarity recall can address the page."""
        vec = await embed_text(self._embedding_client, section)
        browse_log = self._db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
        if browse_log is None:
            return None
        entries = browse_log.append(
            [LogEntryInput(content=section, content_embedding=vec)], author=self._author
        )
        return entries[0] if entries else None

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
    def _dropped_section(total: int, dropped: list[BrowseItem]) -> str:
        """Name the items dropped past the per-call TOTAL so the omission is visible
        to the model, with the exact recovery: rerun the rest in a follow-up call."""
        ran = "item was" if total == 1 else f"{total} items were"
        count = len(dropped)
        dropped_list = ", ".join(repr(item.query) for item in dropped)
        return (
            f"{PennyConstants.BROWSE_DROPPED_HEADER}only the first {ran} run; "
            f"{count} beyond the {total}-per-call total were not run: {dropped_list}. "
            f"Call browse(items=[{dropped_list}]) again if you still need the results."
        )

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
