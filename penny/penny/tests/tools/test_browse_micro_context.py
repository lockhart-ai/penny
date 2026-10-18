"""Browse micro-contexts (#1588): an ``extract`` browse pulls bulk page content
out of the main run context.

When a browse carries an ``extract`` micro-instruction, the fetched page body
goes into a FRESH single-shot micro-context (content + instruction, no tools) and
only the typed result + a fetch handle return to the main loop — the page body
never enters the run context.  The full content is stored whole in browse-results
and is retrievable by its handle (the anchor discipline).  Chat's browse (no
``extract``) is unchanged.

Browse-extract draws with thinking OFF and its plain text IS the value, verbatim
(#2615) — reading a page is not a reasoning task.  There is no declared output
shape any more (PR #1594's enumerated tagged contract retired with it): the poison
screen (a degeneration collapse, a leaked Harmony envelope, or a blank draw) is the
whole validity check, re-rolled on the unchanged context for the whole budget, then
an honest ``EXTRACTION_FAILED``, never a silent empty.

A multi-page batch draws its pages CONCURRENTLY (#1941) while keeping every #1682
guarantee: per-page attribution, stable page order, one ledger row per context,
and a poison draw re-rolling only its OWN context.  Overlap is asserted
STRUCTURALLY — a probe client records how many draws are in flight at once and
forces the batch to complete in reverse order — never by wall clock.

Fictional pages + deterministic mock model responses throughout.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
from collections import Counter
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.config_params import RuntimeParams
from penny.constants import PageReading, PennyConstants
from penny.database import Database
from penny.database.memory.objects import render_tool_call
from penny.database.models import PromptLog
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmResponse, LlmTimeoutError
from penny.prompts import Prompt
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tests.schema_template import migrated_db
from penny.tools import ToolCall, ToolExecutor, ToolRegistry
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.micro_context import (
    MICRO_CONTEXT_SYSTEM_PROMPT,
    MicroContext,
    MicroContextResult,
    MicroExtractOutcome,
)

# ── Fictional page + extraction fixtures ──────────────────────────────────────

_PAGE_URL = "https://auctions.example/lot/42"
_PAGE_BODY = "Lot 42 — Antique Zephyr Compass. Current bid: 275 zorkmids. Closes Friday at noon."
_PAGE_TEXT = f"Title: Lot 42\n{_PAGE_BODY}"
_BODY_PHRASE = "Antique Zephyr Compass"  # a distinctive body phrase that must NOT leak
_INSTRUCTION = "the current bid amount"
# Since #2615 the draw's plain text IS the value, verbatim — no tag to parse off it.
_EXTRACTED_VALUE = "The current bid is 275 zorkmids."
_NOT_PRESENT_ANSWER = "The page doesn't list a bid amount."
# The stated shortfall (#2360), now said inside the plain text itself rather than on a
# separate ``MISSING:`` line (#2615).
_VALUE_WITH_SHORTFALL = f"{_EXTRACTED_VALUE} The page does not give the seller's name."
# A blank draw — the empty-draw half of the poison screen (#2615: there is no tagged
# shape left to catch this, so ``_is_poison`` screens it directly).
_BLANK_DRAW = "   "
_LOG_GET_RE = re.compile(r"log='browse-results' id=(\d+)")

# A second fictional page for multi-page (per-page) extract batches (#1682).  Its
# distinctive body phrase is the marker the mock model routes THIS page's reply on.
_PAGE_URL_2 = "https://auctions.example/lot/99"
_PAGE_BODY_2 = "Lot 99 — Brass Orrery. Current bid: 512 zorkmids. Closes Monday at noon."
_PAGE_TEXT_2 = f"Title: Lot 99\n{_PAGE_BODY_2}"
_EXTRACTED_VALUE_2 = "The current bid is 512 zorkmids."
_MARKER_1 = _BODY_PHRASE  # "Antique Zephyr Compass" — page 1's body phrase, survives cleaning
_MARKER_2 = "Brass Orrery"  # page 2's body phrase

# A third fictional page — the concurrency batch (#1941) is measured at three pages,
# the size the ticket's serial cost was observed at.
_PAGE_URL_3 = "https://auctions.example/lot/7"
_PAGE_BODY_3 = "Lot 7 — Copper Sundial. Current bid: 3 zorkmids. Closes Wednesday at noon."
_PAGE_TEXT_3 = f"Title: Lot 7\n{_PAGE_BODY_3}"
_EXTRACTED_VALUE_3 = "The current bid is 3 zorkmids."
_MARKER_3 = "Copper Sundial"  # page 3's body phrase

# A SEARCH query (#2141).  Plain text is routed through the configured search URL, and
# what comes back is rendered by ``_trim_search_result`` — the "titles and links only"
# header plus the lines around each solo link.  It is never extracted from, so its
# whole render is the literal the main loop reads.
_SEARCH_QUERY = "antique compass auctions"
_SEARCH_PAGE_URL = f"https://duckduckgo.com/?q={urllib.parse.quote(_SEARCH_QUERY)}"
_SEARCH_RESULT_LINK = "[Lot 42 — the compass lot](https://auctions.example/lot/42)"
_SEARCH_PAGE_TEXT = (
    "Title: antique compass auctions — results\n"
    "Zephyr compasses are eighteenth-century navigation instruments.\n"
    f"{_SEARCH_RESULT_LINK}\n"
    "Bidding closes Friday."
)
_TRIMMED_SEARCH_RESULT = f"{Prompt.SEARCH_RESULT_HEADER}\n\n{_SEARCH_PAGE_TEXT}"

# A PAGE READ carrying standalone markdown links — the anchor a FAILED extraction hands
# back (#2141), since the body itself never enters the conversation.
_CATALOGUE_URL = "https://auctions.example/catalogue"
_CATALOGUE_BODY = "Bidding opens Friday."
_CATALOGUE_LINK_1 = "[Lot 42 — the compass lot](https://auctions.example/lot/42)"
_CATALOGUE_LINK_2 = "[Lot 99 — the orrery lot](https://auctions.example/lot/99)"
_CATALOGUE_TEXT = (
    f"Title: Spring catalogue\n{_CATALOGUE_BODY}\n{_CATALOGUE_LINK_1}\n{_CATALOGUE_LINK_2}"
)
_LINKS_CLAUSE = f"Links on the page:\n{_CATALOGUE_LINK_1}\n{_CATALOGUE_LINK_2}"

# The grace window a probe draw waits on its gate before giving up.  It is only ever
# SPENT on the failure path — in a concurrent batch every gate is already set and
# nothing waits — so a sequential regression fails on the recorded peak (a readable
# assertion naming what it saw) instead of hanging the suite.
_GATE_GRACE_SECONDS = 0.5


def _make_db(tmp_path, name: str = "test") -> Database:
    """A migrated test DB (so the browse-results log exists)."""
    db = migrated_db(str(tmp_path / f"{name}.db"))
    return db


def _provider(page_text: str, reading: PageReading = PageReading.ARTICLE):
    """A browse provider returning ``page_text`` for any URL, no image."""

    async def request_fn(method: str, params: dict) -> tuple[str, str | None, PageReading]:
        return (page_text, None, reading)

    def provider():
        return (request_fn, MagicMock(check_domain=AsyncMock()))

    return provider


def _responds(content: str) -> MockLlmClient:
    """A mock model client whose every chat returns ``content``."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(message=LlmMessage(role="assistant", content=content))
    )
    return model


def _extract_tool(
    db: Database,
    model: MockLlmClient | LlmClient,
    *,
    reading: PageReading = PageReading.ARTICLE,
) -> BrowseTool:
    tool = BrowseTool(
        max_calls=3,
        db=db,
        embedding_client=cast(Any, MockLlmClient()),
        model_client=cast(Any, model),
        author="widget-watch",
    )
    tool.set_browse_provider(_provider(_PAGE_TEXT, reading))
    return tool


def _provider_by_url(pages: dict[str, str], failing: frozenset[str] = frozenset()):
    """A browse provider returning each URL's mapped text; a URL in ``failing``
    raises (a per-page read failure → a ``## browse error:`` section)."""

    async def request_fn(method: str, params: dict) -> tuple[str, str | None, PageReading]:
        url = params["url"]
        if url in failing:
            raise RuntimeError("page unavailable")
        return (pages[url], None, PageReading.ARTICLE)

    def provider():
        return (request_fn, MagicMock(check_domain=AsyncMock()))

    return provider


def _page_router(routes: dict[str, str]):
    """A response handler picking its reply by the first page-marker present in the
    micro-context's content — so each page's own micro-call gets its own reply."""

    def handler(request: dict, count: int) -> LlmResponse:
        content = request["messages"][-1]["content"]
        for marker, reply in routes.items():
            if marker in content:
                return LlmResponse(message=LlmMessage(role="assistant", content=reply))
        raise AssertionError(f"no route for micro-context content: {content!r}")

    return handler


def _responds_routed(routes: dict[str, str]) -> MockLlmClient:
    """A mock model whose reply is chosen per page-marker (the multi-page twin of
    ``_responds``, which returns one fixed reply for every call)."""
    model = MockLlmClient()
    model.set_response_handler(_page_router(routes))
    return model


def _multi_extract_tool(db: Database, model: MockLlmClient | LlmClient, provider) -> BrowseTool:
    """An extract-capable browse tool over a caller-supplied multi-page provider."""
    tool = BrowseTool(
        max_calls=3,
        db=db,
        embedding_client=cast(Any, MockLlmClient()),
        model_client=cast(Any, model),
        author="widget-watch",
    )
    tool.set_browse_provider(provider)
    return tool


def _responds_in_sequence(routes: dict[str, list[str]]) -> tuple[MockLlmClient, Counter[str]]:
    """A mock model whose reply is chosen by page marker AND by how many draws that
    page has already had (the last reply repeating once its list runs out) — the
    re-roll-aware twin of ``_responds_routed``.

    It returns the per-page draw counter beside the client, which is what makes "a
    poisoned draw re-rolls only ITS context" a countable claim rather than an
    inference from the rendered output."""
    draws: Counter[str] = Counter()

    def handler(request: dict, count: int) -> LlmResponse:
        content = request["messages"][-1]["content"]
        for marker, replies in routes.items():
            if marker in content:
                reply = replies[min(draws[marker], len(replies) - 1)]
                draws[marker] += 1
                return LlmResponse(message=LlmMessage(role="assistant", content=reply))
        raise AssertionError(f"no route for micro-context content: {content!r}")

    model = MockLlmClient()
    model.set_response_handler(handler)
    return model, draws


class _OverlapProbe:
    """A model client that records how many micro-context draws are IN FLIGHT at once
    and forces the batch to complete in REVERSE issue order (#1941).

    Both halves are structural, so neither reads a clock.  Each draw parks until
    every expected draw has arrived — a concurrent batch therefore records a peak of
    ``len(replies)`` while a sequential caller records 1 — and then waits for its
    SUCCESSOR to finish before returning, so the LAST page issued is the FIRST to
    complete and a renderer that appended results as they arrived would emit the
    sections backwards.  Every wait carries ``_GATE_GRACE_SECONDS`` so a sequential
    regression fails on the recorded peak instead of deadlocking.
    """

    def __init__(self, replies: dict[str, str]) -> None:
        self.runtime = None  # mirrors LlmClient.runtime — no runtime, no token budget
        self._replies = replies  # page marker → its tagged reply, in issue order
        self._markers = list(replies)
        self._all_in_flight = asyncio.Event()
        self._finished = {marker: asyncio.Event() for marker in replies}
        self.in_flight = 0
        self.peak_in_flight = 0
        self.completion_order: list[str] = []
        self.draws: Counter[str] = Counter()

    async def chat(self, messages: list[LlmMessage], **_kwargs: Any) -> LlmResponse:
        """One micro-context draw: identify whose page it is, join the batch, hold
        until the whole batch is in flight, then complete in reverse issue order."""
        marker = self._marker_for(messages)
        self.draws[marker] += 1
        self._arrive()
        with suppress(TimeoutError):
            await asyncio.wait_for(self._all_in_flight.wait(), _GATE_GRACE_SECONDS)
        await self._wait_for_successor(marker)
        self.in_flight -= 1
        self.completion_order.append(marker)
        self._finished[marker].set()
        return LlmResponse(message=LlmMessage(role="assistant", content=self._replies[marker]))

    def _arrive(self) -> None:
        """Join the batch, recording the peak; the last arrival opens the gate."""
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        if self.in_flight >= len(self._markers):
            self._all_in_flight.set()

    async def _wait_for_successor(self, marker: str) -> None:
        """Park until the NEXT page issued has finished — the reverse-completion
        forcing.  The last page issued has no successor, so it returns first."""
        successor = self._markers.index(marker) + 1
        if successor >= len(self._markers):
            return
        with suppress(TimeoutError):
            await asyncio.wait_for(
                self._finished[self._markers[successor]].wait(), _GATE_GRACE_SECONDS
            )

    def _marker_for(self, messages: list[LlmMessage]) -> str:
        """Whose page this draw is reading, by the first page marker in its content."""
        content = messages[-1].content
        for marker in self._markers:
            if marker in content:
                return marker
        raise AssertionError(f"no route for micro-context content: {content!r}")


# ── Integration: bulk content stays out of the main context ───────────────────


@pytest.mark.asyncio
async def test_extract_keeps_page_body_out_of_main_context(tmp_path, mock_llm):
    """The collector-path browse: an ``extract`` instruction sends the page body to
    a micro-context and only the typed value + handle reach the main loop.  The
    value is byte-identical to the micro-context's return, the full content is
    retrievable by its handle, and the micro-context is a ledger-visible model call
    attributed to its own agent/prompt type."""
    db = _make_db(tmp_path)
    # A real LlmClient (patched by mock_llm) so the micro-context call logs a
    # promptlog row — proving ledger visibility, not just what we passed it.
    client = LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.0,
    )
    mock_llm.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(role="assistant", content=_EXTRACTED_VALUE)
        )
    )
    tool = _extract_tool(db, client)

    # The dispatched call's own link (#2458) a real dispatch would already have
    # built — constructed here directly since this test drives the tool without
    # the agent loop around it.
    dispatched_link = CallLink.top("chat-run-1").for_tool_call("call-browse-1")
    result = await tool.run(dispatched_link, queries=[_PAGE_URL], extract=_INSTRUCTION)

    # The tool result carries the typed value alone — never the page body, and
    # no fetch-handle tail (success renders the value only; the "saved" clause
    # read as the remembering being done).
    assert f"{_INSTRUCTION}: {_EXTRACTED_VALUE}" in result.message
    assert _BODY_PHRASE not in result.message
    assert "Closes Friday" not in result.message
    assert _LOG_GET_RE.search(result.message) is None
    assert result.success is True
    # Byte-identical: the extracted value renders verbatim UNDER this page's own
    # section header (per-page extraction, #1682) — no re-transcription by the parent.
    assert result.message.startswith(
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}"
    )

    # The *built context* — what the agent loop appends as the model-facing tool
    # message — also carries the value, not the body.
    framed = Tool.format_result("browse", {"queries": [_PAGE_URL], "extract": _INSTRUCTION}, result)
    assert _EXTRACTED_VALUE in framed
    assert _BODY_PHRASE not in framed

    # The full page content is still stored whole in browse-results (the render
    # dropped the handle line, not the storage).
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    stored_entries = browse_log.read_recent(window_seconds=3600, cap=None)
    assert any(_BODY_PHRASE in entry.content for entry in stored_entries)

    # The micro-context is a ledger-visible model call with an honest attribution.
    with Session(db.engine) as session:
        rows = session.exec(
            select(PromptLog).where(
                PromptLog.agent_name == PennyConstants.BROWSE_EXTRACT_AGENT_NAME
            )
        ).all()
    assert len(rows) == 1
    assert rows[0].prompt_type == PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE
    assert rows[0].run_target == "widget-watch"
    # And (#2355 part A) it names the TURN and the dispatched CALL it was made for
    # — the link a run's own record reads to list the draws made on its behalf.
    assert rows[0].parent_run_id == "chat-run-1"
    assert rows[0].parent_tool_call_id == "call-browse-1"
    assert rows[0].call_outcome == "completed"
    # The default reading is ARTICLE (#2388) end to end — BrowseTool never
    # fabricates a link-list note for a provider that never marked one.
    sent_messages = json.loads(rows[0].messages)
    assert "read as its list of links" not in sent_messages[1]["content"]


@pytest.mark.asyncio
async def test_extract_link_list_reading_reaches_the_model_end_to_end(tmp_path, mock_llm):
    """#2388: the browser's own reading mark travels through ``BrowsePage`` →
    ``_assemble_sections`` → ``_extract_batch`` → ``_extract_one_page`` into the
    REAL micro-context draw — not just the ``MicroContext.extract`` unit above —
    so the note reaches the model on a provider that marked its page a link list."""
    db = _make_db(tmp_path)
    client = LlmClient(
        api_url="http://localhost:11434", model="test-model", db=db, max_retries=1, retry_delay=0.0
    )
    mock_llm.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(role="assistant", content=_EXTRACTED_VALUE)
        )
    )
    tool = _extract_tool(db, client, reading=PageReading.LINK_LIST)

    result = await tool.run(CallLink.new_top(), queries=[_PAGE_URL], extract=_INSTRUCTION)

    assert f"{_INSTRUCTION}: {_EXTRACTED_VALUE}" in result.message
    with Session(db.engine) as session:
        rows = session.exec(
            select(PromptLog).where(
                PromptLog.agent_name == PennyConstants.BROWSE_EXTRACT_AGENT_NAME
            )
        ).all()
    assert len(rows) == 1
    sent_messages = json.loads(rows[0].messages)
    turn = sent_messages[1]["content"]
    assert "read as its list of links" in turn
    assert turn.index("read as its list of links") < turn.index("Content:")


@pytest.mark.asyncio
async def test_browse_without_extract_is_rejected_at_the_gate(tmp_path):
    """``extract`` is REQUIRED (#1570 — every browse routes page content through a
    micro-context; the page never enters the main context whole).  A missing
    extract is an arg-gate rejection naming the fix, and no fetch happens."""
    db = _make_db(tmp_path)
    model = MockLlmClient()
    tool = _extract_tool(db, model)

    result = await tool.run(CallLink.new_top(), queries=[_PAGE_URL])

    assert result.success is False
    assert "extract" in result.message
    assert model.requests == []  # nothing ran


@pytest.mark.asyncio
async def test_extract_without_model_client_degrades_visibly(tmp_path):
    """An ``extract`` requested with no model client wired fails visibly (named, not a
    silent body dump) under the page's OWN section header, while still storing the
    content by handle — and it hands back the page's links like every other extraction
    failure (#2141), because a model-less browse is no less of a dead end."""
    db = _make_db(tmp_path)
    tool = BrowseTool(
        max_calls=3,
        db=db,
        embedding_client=cast(Any, MockLlmClient()),
        author="widget-watch",
    )
    tool.set_browse_provider(_provider(_CATALOGUE_TEXT))

    result = await tool.execute(CallLink.new_top(), queries=[_CATALOGUE_URL], extract=_INSTRUCTION)

    stored_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert stored_log is not None
    stored = stored_log.read_all()  # content still stored
    assert result.message == (
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_CATALOGUE_URL}\n"
        "Couldn't extract 'the current bid amount' — no extraction model is configured "
        f"for this browse. Full page content saved as "
        f"log='browse-results' id={stored[-1].id}.\n{_LINKS_CLAUSE}"
    )
    assert result.success is False
    assert _CATALOGUE_BODY not in result.message


# ── Whole-render literals: the enumerated micro-result forms (via execute) ─────


async def _execute_extract(tmp_path, name: str, model_content: str):
    db = _make_db(tmp_path, name)
    tool = _extract_tool(db, _responds(model_content))
    result = await tool.execute(CallLink.new_top(), queries=[_PAGE_URL], extract=_INSTRUCTION)
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    stored = browse_log.read_all()
    return result, stored[-1].id


@pytest.mark.asyncio
async def test_micro_result_render_forms(tmp_path):
    """The main-loop render forms, whole-string, as the model reads them — each
    UNDER its page's ``## browse:`` section header (per-page, #1682): a plain
    extracted answer, an answer stating its own shortfall, an answer saying the
    page carries none of it, and failed-after-reroll (#2615: one failure render
    now, whatever kind of draw never came back usable)."""
    header = f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n"

    # Success renders the VALUE ALONE — no fetch-handle tail (the "saved"
    # clause read as the remembering being done on the chat teach round; the
    # failure render below keeps the log_get read as its remedy).
    result, _handle_id = await _execute_extract(tmp_path, "ok", _EXTRACTED_VALUE)
    assert result.message == f"{header}{_INSTRUCTION}: {_EXTRACTED_VALUE}"
    assert result.success is True

    # A stated shortfall (#2360) is now said INSIDE the plain-text value itself
    # (#2615) — nothing composes it separately any more, so the render is the
    # same ``{instruction}: {value}`` form as the plain success above.
    result, _handle_id = await _execute_extract(
        tmp_path, "ok-with-shortfall", _VALUE_WITH_SHORTFALL
    )
    assert result.message == f"{header}{_INSTRUCTION}: {_VALUE_WITH_SHORTFALL}"
    assert result.success is True

    # A page carrying none of what was asked is still a SUCCESSFUL draw (#2615):
    # the plain text says so, and the render is the same success form — no
    # handle clause, because the page was read and the answer stands on its own.
    result, _handle_id = await _execute_extract(tmp_path, "absent", _NOT_PRESENT_ANSWER)
    assert result.message == f"{header}{_INSTRUCTION}: {_NOT_PRESENT_ANSWER}"
    assert result.success is True

    # Every unusable draw — garbled or blank alike — reads the same failure form
    # (#2615: there is no second failure KIND left to render differently).
    result, handle_id = await _execute_extract(tmp_path, "poison", "...???...")
    assert result.message == (
        f"{header}"
        "Couldn't extract 'the current bid amount' from the page — the extractor never "
        "produced a usable answer after 3 attempts. Full page content saved as "
        f"log='browse-results' id={handle_id}."
    )
    assert result.success is False

    result, handle_id = await _execute_extract(tmp_path, "blank", _BLANK_DRAW)
    assert result.message == (
        f"{header}"
        "Couldn't extract 'the current bid amount' from the page — the extractor never "
        "produced a usable answer after 3 attempts. Full page content saved as "
        f"log='browse-results' id={handle_id}."
    )
    assert result.success is False


# ── Per-page extraction in a batched browse (#1682) ───────────────────────────


@pytest.mark.asyncio
async def test_two_page_batch_yields_two_attributed_sections(tmp_path, mock_llm):
    """A two-page batched extract runs ONE micro-context PER page (#1682): two
    sections, each under its OWN ``## browse:`` header, each carrying its OWN value
    and OWN fetch handle (no cross-source contamination) — and the batch logs TWO
    attributed ledger rows (one micro-call per page)."""
    db = _make_db(tmp_path)
    client = LlmClient(
        api_url="http://localhost:11434",
        model="test-model",
        db=db,
        max_retries=1,
        retry_delay=0.0,
    )
    mock_llm.set_response_handler(
        _page_router({_MARKER_1: _EXTRACTED_VALUE, _MARKER_2: _EXTRACTED_VALUE_2})
    )
    tool = _multi_extract_tool(
        db, client, _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2})
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION
    )

    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert len(sections) == 2
    # Each page renders under its OWN header with its OWN value...
    assert sections[0].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n")
    assert _EXTRACTED_VALUE in sections[0]
    assert sections[1].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n")
    assert _EXTRACTED_VALUE_2 in sections[1]
    # ...and no value bleeds across sources; success sections carry the value
    # alone (no fetch-handle tail).
    assert _EXTRACTED_VALUE_2 not in sections[0]
    assert _EXTRACTED_VALUE not in sections[1]
    assert _LOG_GET_RE.search(sections[0]) is None
    assert _LOG_GET_RE.search(sections[1]) is None
    assert result.success is True

    # N pages → N attributed ledger rows (one micro-call per page).
    with Session(db.engine) as session:
        rows = session.exec(
            select(PromptLog).where(
                PromptLog.agent_name == PennyConstants.BROWSE_EXTRACT_AGENT_NAME
            )
        ).all()
    assert len(rows) == 2
    assert {row.run_target for row in rows} == {"widget-watch"}


@pytest.mark.asyncio
async def test_a_none_of_it_page_does_not_mask_extracted_page(tmp_path):
    """A page answering "none of it" on page 1 + an ordinary extracted answer on
    page 2: each renders honestly in its OWN section — the "none of it" page never
    masks the other page's extracted value, and the extracted value never leaks
    into the other section (#1682).  Both are the SAME outcome since #2615 (there
    is no second success outcome left to collapse into one render) — the thing
    this test still holds is per-page attribution, not which outcome each lands
    on."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_1: _NOT_PRESENT_ANSWER, _MARKER_2: _EXTRACTED_VALUE_2})
    tool = _multi_extract_tool(
        db, model, _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2})
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION
    )

    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert len(sections) == 2
    # Page 1 renders its own honest "none of it" text under its own header.
    assert sections[0].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n")
    assert _NOT_PRESENT_ANSWER in sections[0]
    # Page 2's extracted value renders under ITS header — not masked, not leaked.
    assert sections[1].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n")
    assert _EXTRACTED_VALUE_2 in sections[1]
    assert _EXTRACTED_VALUE_2 not in sections[0]
    # Both are successful reads (an extracted answer + an honest "none of it" one).
    assert result.success is True


@pytest.mark.asyncio
async def test_failed_fetch_section_coexists_with_extracted_section(tmp_path):
    """A failed fetch keeps its ``## browse error:`` section (no micro-context) right
    alongside a successfully-extracted page's section (#1682) — the read failure
    stays visible, and its slot never consumes another page's handle."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_2: _EXTRACTED_VALUE_2})
    tool = _multi_extract_tool(
        db,
        model,
        _provider_by_url(
            {_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2}, failing=frozenset({_PAGE_URL})
        ),
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION
    )

    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert len(sections) == 2
    # The failed fetch keeps its verbatim error section — no extraction ran on it.
    assert sections[0].startswith(f"{PennyConstants.BROWSE_ERROR_HEADER}{_PAGE_URL}")
    assert "Could not read this page" in sections[0]
    assert _LOG_GET_RE.search(sections[0]) is None
    # The successful page still extracts, under its own header — value alone.
    assert sections[1].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n")
    assert _EXTRACTED_VALUE_2 in sections[1]
    assert _LOG_GET_RE.search(sections[1]) is None
    assert result.success is True


@pytest.mark.asyncio
async def test_failed_fetch_in_the_middle_keeps_its_slot(tmp_path):
    """A read failure sits in ITS OWN slot and consumes no page's handle — the half of
    the #1682 contract the concurrent batch (#1941) has to keep structurally.

    The failure is in the MIDDLE of three queries, so the claim is testable directly:
    the error section renders between its siblings (page order, not completion
    order).  Which STORED entry belongs to which page (#2615: no longer visible in a
    success render — there is no handle clause on one any more, see
    ``_render_micro_result``) is verified against the store itself."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_1: _EXTRACTED_VALUE, _MARKER_3: _EXTRACTED_VALUE_3})
    tool = _multi_extract_tool(
        db,
        model,
        _provider_by_url(
            {_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2, _PAGE_URL_3: _PAGE_TEXT_3},
            failing=frozenset({_PAGE_URL_2}),
        ),
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2, _PAGE_URL_3], extract=_INSTRUCTION
    )

    # Only the two READ pages were stored; the failed fetch stored nothing.  Each
    # stored entry is attributed to its own page by content, not by a render clause.
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    stored = browse_log.read_all()
    assert len(stored) == 2
    assert any(_MARKER_1 in entry.content for entry in stored)
    assert any(_MARKER_3 in entry.content for entry in stored)
    assert result.message.split(PennyConstants.SECTION_SEPARATOR) == [
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}",
        f"{PennyConstants.BROWSE_ERROR_HEADER}{_PAGE_URL_2}\n"
        "Could not read this page: page unavailable. Try a different source or a "
        "reworded query; if other queries in this batch succeeded, work from those "
        "instead of retrying this one.",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_3}\n{_INSTRUCTION}: {_EXTRACTED_VALUE_3}",
    ]
    assert result.success is True


# ── A search is links; a failed page extraction keeps the page's (#2141) ──────


@pytest.mark.asyncio
async def test_search_section_is_returned_as_links_and_never_extracted(tmp_path):
    """A SEARCH query under ``extract`` comes back as its trimmed list of links, and no
    micro-context is drawn for it (#2141).

    ``_page_section`` prepends ``Prompt.SEARCH_RESULT_HEADER`` — an instruction written
    for the CHAT model, telling it to pick a URL and read it.  Handing that to the
    extractor had the extractor obey it: its output was a bare URL, which violates the
    declared ``EXTRACTED:``/``NOT_PRESENT:`` shape, was re-rolled the whole budget, and
    rendered as "the extractor returned nothing usable" over a page of perfectly good
    links.  So a search is left exactly as the header says it is — titles and links, to
    be picked from — and the call still reports success, since something readable came
    back even though no draw speaks for it."""
    db = _make_db(tmp_path)
    model = MockLlmClient()
    tool = _multi_extract_tool(db, model, _provider(_SEARCH_PAGE_TEXT))

    result = await tool.execute(CallLink.new_top(), queries=[_SEARCH_QUERY], extract=_INSTRUCTION)

    assert result.message == (
        f"{PennyConstants.BROWSE_SEARCH_HEADER}{_SEARCH_QUERY}\n{_TRIMMED_SEARCH_RESULT}"
    )
    assert model.requests == []  # the extractor was never asked to read a link list
    assert result.success is True


@pytest.mark.asyncio
async def test_search_beside_a_page_read_extracts_only_the_page(tmp_path):
    """A batch of a search AND a page read draws ONE micro-context — the page's (#2141).

    The search keeps its slot verbatim beside the extracted page, and the page's own
    fetch handle is unaffected: only the page read was stored, so the handle iterator
    the batch consumes has exactly one entry to give it.  The routed mock raises on any
    draw whose content isn't page 1's, so a draw over the search results would fail the
    test rather than pass unnoticed."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_1: _EXTRACTED_VALUE})
    tool = _multi_extract_tool(
        db,
        model,
        _provider_by_url({_SEARCH_PAGE_URL: _SEARCH_PAGE_TEXT, _PAGE_URL: _PAGE_TEXT}),
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_SEARCH_QUERY, _PAGE_URL], extract=_INSTRUCTION
    )

    assert result.message.split(PennyConstants.SECTION_SEPARATOR) == [
        f"{PennyConstants.BROWSE_SEARCH_HEADER}{_SEARCH_QUERY}\n{_TRIMMED_SEARCH_RESULT}",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}",
    ]
    assert len(model.requests) == 1  # one page read, one draw
    # Only the page read was stored, so the search consumed no handle.
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    assert [entry.content.partition("\n")[0] for entry in browse_log.read_all()] == [
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}"
    ]
    assert result.success is True


@pytest.mark.asyncio
async def test_failed_extraction_hands_back_the_pages_links(tmp_path):
    """A page read whose extractor never produces a usable answer renders the
    failure sentence AND the page's own standalone links (#2141).

    With ``extract`` set the body never enters the conversation, so a failure saying
    only that it failed left the chat model nothing to open — it re-searched, landed on
    the same page, and looped until the pathology gate stopped it.  The links are the
    one part of the page it can act on, so they come back with the failure: the fetch
    handle remedies the body, the links remedy the next call.

    Since #2615 there is no tagged line to fail to produce — a degenerate draw is the
    one way left to land here, so that is what this test drives."""
    db = _make_db(tmp_path)
    model = _responds("...???...")
    tool = _multi_extract_tool(db, model, _provider(_CATALOGUE_TEXT))

    result = await tool.execute(CallLink.new_top(), queries=[_CATALOGUE_URL], extract=_INSTRUCTION)

    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    handle_id = browse_log.read_all()[-1].id
    assert result.message == (
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_CATALOGUE_URL}\n"
        "Couldn't extract 'the current bid amount' from the page — the extractor never "
        "produced a usable answer after 3 attempts. Full page content saved as "
        f"log='browse-results' id={handle_id}.\n{_LINKS_CLAUSE}"
    )
    assert result.success is False
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    # The links are the anchor; the page's prose body still never enters the context.
    assert _CATALOGUE_BODY not in result.message


# ── Concurrent page batch (#1941) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_page_batch_draws_concurrently_and_renders_in_page_order(tmp_path):
    """A three-page extract issues all three micro-context draws CONCURRENTLY and
    still renders them in PAGE order (#1941).

    Both claims are structural, neither is a wall-clock measurement.  Overlap: the
    probe holds every draw until all three are in flight, so the recorded peak is 3 —
    a sequential caller would record 1.  Order: the probe then releases them in
    REVERSE, so the batch COMPLETES backwards and a renderer appending results as
    they arrived would emit the sections backwards.  N pages still cost exactly N
    draws — the batch is a fan-out, not a retry."""
    db = _make_db(tmp_path)
    probe = _OverlapProbe(
        {_MARKER_1: _EXTRACTED_VALUE, _MARKER_2: _EXTRACTED_VALUE_2, _MARKER_3: _EXTRACTED_VALUE_3}
    )
    tool = _multi_extract_tool(
        db,
        cast(Any, probe),
        _provider_by_url(
            {_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2, _PAGE_URL_3: _PAGE_TEXT_3}
        ),
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2, _PAGE_URL_3], extract=_INSTRUCTION
    )

    assert probe.draws == Counter({_MARKER_1: 1, _MARKER_2: 1, _MARKER_3: 1})
    assert probe.peak_in_flight == 3
    assert probe.completion_order == [_MARKER_3, _MARKER_2, _MARKER_1]
    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert sections == [
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n{_INSTRUCTION}: {_EXTRACTED_VALUE_2}",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_3}\n{_INSTRUCTION}: {_EXTRACTED_VALUE_3}",
    ]
    assert result.success is True


@pytest.mark.asyncio
async def test_single_page_batch_is_one_draw_rendered_unchanged(tmp_path):
    """A batch of ONE fans out to nothing and renders byte-identically to the
    pre-#1941 sequential form — the gather is over the planned batch, so the
    single-page path costs exactly one draw with nothing in flight beside it."""
    db = _make_db(tmp_path)
    probe = _OverlapProbe({_MARKER_1: _EXTRACTED_VALUE})
    tool = _multi_extract_tool(db, cast(Any, probe), _provider_by_url({_PAGE_URL: _PAGE_TEXT}))

    result = await tool.execute(CallLink.new_top(), queries=[_PAGE_URL], extract=_INSTRUCTION)

    assert probe.draws == Counter({_MARKER_1: 1})
    assert probe.peak_in_flight == 1
    assert result.message == (
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}"
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_poisoned_draw_rerolls_only_its_own_context(tmp_path):
    """Poison in ONE page's context re-rolls THAT context alone (#1941).

    Each page draws in its own single-shot context with its own reroll budget, so
    page 1's two discarded draws cost page 1 three draws and its concurrent sibling
    exactly one — and both pages still render their own value, in page order."""
    db = _make_db(tmp_path)
    poison = "...???..."
    model, draws = _responds_in_sequence(
        {_MARKER_1: [poison, poison, _EXTRACTED_VALUE], _MARKER_2: [_EXTRACTED_VALUE_2]}
    )
    tool = _multi_extract_tool(
        db, model, _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2})
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION
    )

    assert draws[_MARKER_1] == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert draws[_MARKER_2] == 1  # the clean sibling was never re-drawn
    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert sections == [
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n{_INSTRUCTION}: {_EXTRACTED_VALUE_2}",
    ]
    assert result.success is True


@pytest.mark.asyncio
async def test_a_raising_draw_is_rendered_as_that_pages_own_failure(tmp_path):
    """A transport failure inside ONE page's draw renders as THAT page's own
    failure line, under its own header — and leaves every OTHER page's already-
    drawn result intact (#2367).

    The old behaviour re-raised the earliest failing page and discarded the whole
    call, including a sibling page that had already succeeded.  ``_extract_one_page``
    now catches the draw's own ``Exception`` per page (never ``BaseException``, so a
    genuine cancellation still propagates), so one transport failure costs only its
    own page."""
    db = _make_db(tmp_path)
    finished: list[str] = []

    class _RaisesOnTheFirstPage:
        runtime = None  # mirrors LlmClient.runtime — no runtime, no token budget

        async def chat(self, messages: list[LlmMessage], **_kwargs: Any) -> LlmResponse:
            content = messages[-1].content
            if _MARKER_1 in content:
                raise LlmTimeoutError("model call timed out")
            await asyncio.sleep(0)
            finished.append(_MARKER_2)
            return LlmResponse(message=LlmMessage(role="assistant", content=_EXTRACTED_VALUE_2))

    tool = _multi_extract_tool(
        db,
        cast(Any, _RaisesOnTheFirstPage()),
        _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2}),
    )

    result = await tool.execute(
        CallLink.new_top(), queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION
    )

    assert finished == [_MARKER_2]  # the sibling drew undisturbed
    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert len(sections) == 2
    assert sections[0].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n")
    assert "the extraction call errored: model call timed out" in sections[0]
    assert sections[1].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n")
    assert _EXTRACTED_VALUE_2 in sections[1]
    # Page 2 still succeeded, so the call as a whole did too.
    assert result.success is True


# ── Timeout report (#2367) ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_report_names_the_finished_page_and_the_one_still_waiting(tmp_path):
    """When the executor's timeout cancels an in-flight browse, ``timeout_report``
    names the page whose extraction finished first — with the SAME content the
    model would have gotten without a timeout — and the page still waiting, by its
    own stored handle (#2367).  Nothing claims more than actually happened: page 2's
    draw never returns, so it is never named as extracted."""
    db = _make_db(tmp_path)

    class _StallsOnTheSecondPage:
        runtime = None  # mirrors LlmClient.runtime — no runtime, no token budget

        async def chat(self, messages: list[LlmMessage], **_kwargs: Any) -> LlmResponse:
            content = messages[-1].content
            if _MARKER_1 in content:
                return LlmResponse(message=LlmMessage(role="assistant", content=_EXTRACTED_VALUE))
            await asyncio.sleep(10)  # cancelled by the tool's own timeout, well short of 10s
            raise AssertionError("page 2's draw should have been cancelled")

    tool = _multi_extract_tool(
        db,
        cast(Any, _StallsOnTheSecondPage()),
        _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2}),
    )
    tool.timeout = 0.2  # type: ignore[assignment]  # BrowseTool.timeout defaults to 900.0

    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry, timeout=60.0)  # unused: the tool's own timeout wins

    result = await executor.execute(
        ToolCall(
            tool="browse", arguments={"queries": [_PAGE_URL, _PAGE_URL_2], "extract": _INSTRUCTION}
        ),
        CallLink.new_top(),
    )

    assert result.success is False
    assert result.narration == "You tried to use `browse` but it timed out."
    # Page 1's finished extraction carries its full content, not just a label.
    assert _PAGE_URL in result.message
    assert _EXTRACTED_VALUE in result.message
    # Page 2 is named only as still waiting — never as extracted or not-present.
    assert _PAGE_URL_2 in result.message
    assert "still waiting" in result.message.lower()
    assert _EXTRACTED_VALUE_2 not in result.message


def test_render_tool_call_names_the_micro_context_browse_step():
    """The run-trace line for a micro-context browse step names the extract
    instruction; a plain browse is unchanged."""
    assert (
        render_tool_call("browse", {"queries": [_PAGE_URL], "extract": _INSTRUCTION})
        == "browse(queries=['https://auctions.example/lot/42'], extract='the current bid amount')"
    )
    assert render_tool_call("browse", {"queries": [_PAGE_URL]}) == (
        "browse(['https://auctions.example/lot/42'])"
    )


# ── MicroContext unit: plain-text draw, byte-identity, poison reroll (#2615) ──


@pytest.mark.asyncio
async def test_micro_context_returns_byte_identical_value_with_attribution():
    """A draw's plain text is the extracted value — stripped once, verbatim
    otherwise (#2615: nothing parses it) — and the call carries the ledger
    attribution, drawn with thinking OFF (reading a page is not a reasoning
    task)."""
    model = _responds(f"  {_EXTRACTED_VALUE}  ")
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top(), run_target="widget-watch"
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == _EXTRACTED_VALUE
    assert model.requests[0]["agent_name"] == PennyConstants.BROWSE_EXTRACT_AGENT_NAME
    assert model.requests[0]["prompt_type"] == PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE
    assert model.requests[0]["run_target"] == "widget-watch"
    assert model.requests[0]["thinking_enabled"] is False
    # The default reading is ARTICLE (#2388) — the rendered user turn is
    # byte-identical to what it was before ``reading`` existed: no note, nothing
    # before ``Content:`` beyond the instruction.
    assert (
        model.requests[0]["messages"][1]["content"]
        == f"Instruction: {_INSTRUCTION}\n\nContent:\n{_PAGE_BODY}"
    )


@pytest.mark.asyncio
async def test_micro_context_link_list_reading_adds_one_note_before_content():
    """#2388: a link-list read (the extension's own structural mark, never guessed
    from the text's shape) gets one note, placed BEFORE ``Content:``, saying the
    page was read as its list of links — the byte-identical test above pins its
    absence on every other reading."""
    model = _responds(_EXTRACTED_VALUE)
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top(), reading=PageReading.LINK_LIST
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    turn = model.requests[0]["messages"][1]["content"]
    assert "read as its list of links" in turn
    assert turn.index("read as its list of links") < turn.index("Content:")


@pytest.mark.asyncio
async def test_micro_context_plain_text_answer_needs_no_tag():
    """A page the extractor finds nothing on is still an ``EXTRACTED`` draw whose
    plain text says so (#2615) — there is no second outcome, and no tag, for it to
    land on instead."""
    model = _responds(_NOT_PRESENT_ANSWER)
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == _NOT_PRESENT_ANSWER
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_micro_context_extracts_multi_line_value_whole():
    """A multi-line value (a bulleted digest) round-trips WHOLE (#1682/#2615): the
    plain-text draw IS the value — stripped once, nothing else touched — so an
    item-per-line list survives into ``MicroContextResult`` byte-identical."""
    digest = "Notable lots:\n- Lot 42: 275 zorkmids\n- Lot 99: 512 zorkmids\n- Lot 7: 3 zorkmids"
    model = _responds(digest)
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == digest  # every line preserved, verbatim
    assert result.value.count("\n") == 3  # the four-line digest is intact


def test_micro_context_system_prompt_is_plain_text_with_no_tags():
    """Whole-render literal of the extraction system prompt (#2615): the tagged
    contract (#1682/#1942/#2360) retired entirely — reading a page is not a
    reasoning task, so the draw answers in plain text, as a list when several
    things are asked for, naming a shortfall or a total absence in a plain
    sentence, never a separate tagged line."""
    assert MICRO_CONTEXT_SYSTEM_PROMPT == (
        "You are an extraction step. You are given the full text of one or more web "
        "pages and a single instruction naming exactly what to pull out of them. "
        "Answer in plain text from the page alone, as a list with one item per "
        "line when the instruction asks for several, and never a value the page "
        "doesn't state.\n"
        "\n"
        "An instruction often names several things at once, and a page rarely "
        "carries all of them. Give what the page has for each thing named, in the "
        "order the instruction names them. The page's gaps are part of the answer: "
        "the person reading your answer cannot see the page, so when a thing the "
        "instruction names is not on the page — on one item or on any of them — "
        "say so where that thing's value would have gone, in a few words. A short "
        "note on the item's own line does this (for example, \"— no description on "
        'the page"), and so does one short sentence after the answer naming the '
        "thing the page does not have.\n"
        "\n"
        "If the page carries none of what was asked for, say so plainly in one "
        "sentence and stop there.\n"
        "\n"
        "Write only the answer — no preamble, no restating the instruction, and "
        "nothing about how you are answering."
    )


def test_extract_parameter_description_whole_render(db: Database):
    """Whole-render literal of the ``extract`` parameter's description — the ONE line
    of guidance the calling model reads before it writes an instruction (#1838/#1942).

    It teaches where the instruction comes FROM: the task's own words, broad allowed.
    "Naming exactly what to pull out" read as *be specific* and was measured inflating
    a plain ask into a stricter one than the page could satisfy — the extractor then
    honestly returned NOT_PRESENT and the round died on a fact that was there.  The
    example of a broad ask is deliberately non-fixture phrasing, so no eval pool lends
    its own words to the surface under test.

    What #1942 changed is the CONSEQUENCE it states.  It used to warn that naming an
    extra detail "makes the read come back empty when the page lacks them" — true when a
    partly-answered instruction flipped the whole page to NOT_PRESENT, and false now that
    the read degrades per thing asked for.  A description stating a consequence the
    system no longer has is worse than one stating none.

    #2141 added the SEARCH clause for the same reason: a search is never extracted from,
    so "only the extracted value is returned here" stopped being true of a search query,
    and a description that is false about half its own argument's inputs is worse than
    one that names the exception."""
    tool = _extract_tool(db, _responds(_EXTRACTED_VALUE))
    assert tool.parameters["properties"]["extract"]["description"] == (
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
    )


@pytest.mark.asyncio
async def test_micro_context_blank_draw_is_discarded_and_rerolled():
    """A blank draw (#2615: there is no tag left to catch this, so ``_is_poison``
    screens it directly via ``is_empty_draw``) is discarded and re-drawn on the
    unchanged context for the whole budget, then fails honestly — the same
    patience a garbled collapse gets."""
    model = _responds(_BLANK_DRAW)
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTION_FAILED
    assert result.value == ""
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_micro_context_reroll_can_recover():
    """A reroll re-draws on the unchanged context — a usable draw recovers the
    extraction, and it can arrive anywhere inside the budget: two unusable draws
    still leave a third, and the run costs exactly the draws it took."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="...???..." if count == 1 else _EXTRACTED_VALUE,
            )
        )
    )
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == _EXTRACTED_VALUE
    assert len(model.requests) == 2

    late = MockLlmClient()
    late.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="...???..." if count <= 2 else _EXTRACTED_VALUE,
            )
        )
    )
    recovered = await MicroContext(cast(Any, late)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert recovered.outcome == MicroExtractOutcome.EXTRACTED
    assert recovered.value == _EXTRACTED_VALUE
    assert len(late.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_micro_context_poison_is_discarded_and_rerolled():
    """Poison output (a degeneration collapse) is discarded and re-drawn on the
    unchanged context up to the reroll budget, then fails honestly — the one
    failure outcome left once the tagged shape retired (#2615)."""
    model = _responds("...???...")
    result = await MicroContext(cast(Any, model), reroll_attempts=3).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTION_FAILED
    assert result.value == ""
    assert len(model.requests) == 3


@pytest.mark.asyncio
async def test_extract_always_draws_with_thinking_off():
    """Browse-extract draws with thinking OFF on every attempt (#2615) — reading a
    page is not a reasoning task, and this is the one customer that always says
    so, unconditionally, from the first draw."""
    model = _responds(_EXTRACTED_VALUE)
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, call_link=CallLink.new_top()
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert model.requests[0]["thinking_enabled"] is False


@pytest.mark.asyncio
async def test_draw_clean_cut_draw_gets_one_thinking_off_last_draw():
    """A draw the token budget cut (``finish_reason="length"``, empty content)
    gets ONE more call on the same messages with thinking disabled and the same
    budget (#2476), and that response answers the draw in place of the cut one —
    the poison screen then reads it like any other draw.  Both calls carry the
    call kind's own token budget, resolved off the client's ``runtime`` by
    ``agent_name``.

    Exercised directly against ``_draw_clean`` with ``thinking_enabled=True``
    (#2615): browse-extract's own ``extract()`` always draws thinking OFF from
    the first attempt, so this retry — whose whole point is turning thinking off
    — never fires for it any more (see the guard test below); the mechanic still
    serves a customer that draws with thinking on by default."""
    model = MockLlmClient(
        runtime=RuntimeParams(env_overrides={"TOKEN_BUDGET_BROWSE_EXTRACT": 1234})
    )
    model.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(
                role="assistant",
                content="" if count == 1 else _EXTRACTED_VALUE,
            ),
            finish_reason="length" if count == 1 else None,
        )
    )
    text = await MicroContext(cast(Any, model))._draw_clean(
        _PAGE_BODY, _INSTRUCTION, None, call_link=CallLink.new_top(), thinking_enabled=True
    )
    assert text == _EXTRACTED_VALUE
    assert len(model.requests) == 2
    # Both calls carried the draw's kind budget, resolved from this client's runtime.
    assert model.requests[0]["token_budget"] == 1234
    assert model.requests[1]["token_budget"] == 1234
    # The retry is the thinking-off last draw; the first draw keeps the default.
    assert model.requests[0]["thinking_enabled"] is True
    assert model.requests[1]["thinking_enabled"] is False


@pytest.mark.asyncio
async def test_draw_clean_skips_the_cut_retry_when_already_thinking_off():
    """The cut-draw retry's whole point is turning thinking off (#2476) — so when a
    draw already ran with it off (every browse-extract draw, #2615), a second,
    identical call would buy nothing.  The guard (``not retried_cut and
    thinking_enabled``) skips it: a cut draw already thinking-off costs exactly
    one call, whatever its content."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(role="assistant", content=""),
            finish_reason="length",
        )
    )
    text = await MicroContext(cast(Any, model))._draw_clean(
        _PAGE_BODY, _INSTRUCTION, None, call_link=CallLink.new_top(), thinking_enabled=False
    )
    assert text is None  # every attempt drew the same empty, cut response
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert all(request["thinking_enabled"] is False for request in model.requests)


def test_micro_result_render_by_handle_is_a_typed_id():
    """The fetch handle is a typed ``log='browse-results' id=<id>`` read
    (rendered directly) — on the FAILURE render, where it is the remedy
    (#2615: the one failure outcome left).  A successful extraction renders
    the value alone: the "saved" tail read as the remembering being done at
    exactly the moment a chat teach round held the value (2026-07-19), so
    success carries no handle clause, whatever the plain text says.

    The page's LINKS follow the same split (#2141): handed to every render, and
    only the FAILURE render appends them."""
    tool = BrowseTool(max_calls=1, embedding_client=cast(Any, MockLlmClient()))
    stored = [cast(Any, SimpleNamespace(id=7))]
    links = f"\n{_LINKS_CLAUSE}"
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=_EXTRACTED_VALUE),
        _INSTRUCTION,
        stored,
        links,
    )
    assert body == f"{_INSTRUCTION}: {_EXTRACTED_VALUE}"
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=_NOT_PRESENT_ANSWER),
        _INSTRUCTION,
        stored,
        links,
    )
    assert body == f"{_INSTRUCTION}: {_NOT_PRESENT_ANSWER}"
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTION_FAILED),
        _INSTRUCTION,
        stored,
        links,
    )
    assert body == (
        "Couldn't extract 'the current bid amount' from the page — the extractor never "
        "produced a usable answer after 3 attempts. Full page content saved as "
        f"log='browse-results' id=7.\n{_LINKS_CLAUSE}"
    )
