"""Browse micro-contexts (#1588): an ``extract`` browse pulls bulk page content
out of the main run context.

When a browse carries an ``extract`` micro-instruction, the fetched page body
goes into a FRESH single-shot micro-context (content + instruction, no tools) and
only the typed result + a fetch handle return to the main loop — the page body
never enters the run context.  The full content is stored whole in browse-results
and is retrievable by its handle (the anchor discipline).  Chat's browse (no
``extract``) is unchanged.

The micro-context output contract is ENUMERATED on both sides of the interface
(the review fix on PR #1594): the prompt names the two tagged forms
(``EXTRACTED: <value>`` / ``NOT_PRESENT: <reason>``) and classification is a
deterministic tag parse — untagged output is a contract violation that gets one
reroll and then fails honestly, never a value.

A multi-page batch draws its pages CONCURRENTLY (#1941) while keeping every #1682
guarantee: per-page attribution, stable page order, one ledger row per context,
and a poison draw re-rolling only its OWN context.  Overlap is asserted
STRUCTURALLY — a probe client records how many draws are in flight at once and
forces the batch to complete in reverse order — never by wall clock.

Fictional pages + deterministic mock model responses throughout.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from collections import Counter
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory.objects import render_tool_call
from penny.database.models import PromptLog
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmResponse, LlmTimeoutError
from penny.prompts import Prompt
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tests.schema_template import migrated_db
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.micro_context import (
    EXTRACTED_TAG,
    MICRO_CONTEXT_SYSTEM_PROMPT,
    NOT_PRESENT_TAG,
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
_EXTRACTED_VALUE = "The current bid is 275 zorkmids."
_TAGGED_VALUE = f"{EXTRACTED_TAG} {_EXTRACTED_VALUE}"
_NOT_PRESENT_REASON = "the page lists no bid amount."
_TAGGED_NOT_PRESENT = f"{NOT_PRESENT_TAG} {_NOT_PRESENT_REASON}"
# The confabulation-shaped leak the tag parse exists to stop: non-blank apology
# prose that a blank-check classifier would have promoted to an extracted value.
_UNTAGGED_APOLOGY = "The page doesn't list a price"
_HANDLE_RE = re.compile(r"browse-results#(\d+)")

# A second fictional page for multi-page (per-page) extract batches (#1682).  Its
# distinctive body phrase is the marker the mock model routes THIS page's reply on.
_PAGE_URL_2 = "https://auctions.example/lot/99"
_PAGE_BODY_2 = "Lot 99 — Brass Orrery. Current bid: 512 zorkmids. Closes Monday at noon."
_PAGE_TEXT_2 = f"Title: Lot 99\n{_PAGE_BODY_2}"
_EXTRACTED_VALUE_2 = "The current bid is 512 zorkmids."
_TAGGED_VALUE_2 = f"{EXTRACTED_TAG} {_EXTRACTED_VALUE_2}"
_MARKER_1 = _BODY_PHRASE  # "Antique Zephyr Compass" — page 1's body phrase, survives cleaning
_MARKER_2 = "Brass Orrery"  # page 2's body phrase

# A third fictional page — the concurrency batch (#1941) is measured at three pages,
# the size the ticket's serial cost was observed at.
_PAGE_URL_3 = "https://auctions.example/lot/7"
_PAGE_BODY_3 = "Lot 7 — Copper Sundial. Current bid: 3 zorkmids. Closes Wednesday at noon."
_PAGE_TEXT_3 = f"Title: Lot 7\n{_PAGE_BODY_3}"
_EXTRACTED_VALUE_3 = "The current bid is 3 zorkmids."
_TAGGED_VALUE_3 = f"{EXTRACTED_TAG} {_EXTRACTED_VALUE_3}"
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


def _provider(page_text: str):
    """A browse provider returning ``page_text`` for any URL, no image."""

    async def request_fn(method: str, params: dict) -> tuple[str, str | None]:
        return (page_text, None)

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


def _extract_tool(db: Database, model: MockLlmClient | LlmClient) -> BrowseTool:
    tool = BrowseTool(
        max_calls=3,
        db=db,
        embedding_client=cast(Any, MockLlmClient()),
        model_client=cast(Any, model),
        author="widget-watch",
    )
    tool.set_browse_provider(_provider(_PAGE_TEXT))
    return tool


def _provider_by_url(pages: dict[str, str], failing: frozenset[str] = frozenset()):
    """A browse provider returning each URL's mapped text; a URL in ``failing``
    raises (a per-page read failure → a ``## browse error:`` section)."""

    async def request_fn(method: str, params: dict) -> tuple[str, str | None]:
        url = params["url"]
        if url in failing:
            raise RuntimeError("page unavailable")
        return (pages[url], None)

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
        self._replies = replies  # page marker → its tagged reply, in issue order
        self._markers = list(replies)
        self._all_in_flight = asyncio.Event()
        self._finished = {marker: asyncio.Event() for marker in replies}
        self.in_flight = 0
        self.peak_in_flight = 0
        self.completion_order: list[str] = []
        self.draws: Counter[str] = Counter()

    async def chat(self, messages: list[dict], **_kwargs: Any) -> LlmResponse:
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

    def _marker_for(self, messages: list[dict]) -> str:
        """Whose page this draw is reading, by the first page marker in its content."""
        content = messages[-1].get("content", "")
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
            message=LlmMessage(role="assistant", content=_TAGGED_VALUE)
        )
    )
    tool = _extract_tool(db, client)

    result = await tool.execute(queries=[_PAGE_URL], extract=_INSTRUCTION)

    # The tool result carries the typed value alone — never the page body, and
    # no fetch-handle tail (success renders the value only; the old "saved to
    # browse-results#N" line read as the remembering being done).
    assert f"{_INSTRUCTION}: {_EXTRACTED_VALUE}" in result.message
    assert _BODY_PHRASE not in result.message
    assert "Closes Friday" not in result.message
    assert _HANDLE_RE.search(result.message) is None
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


@pytest.mark.asyncio
async def test_browse_without_extract_is_rejected_at_the_gate(tmp_path):
    """``extract`` is REQUIRED (#1570 — every browse routes page content through a
    micro-context; the page never enters the main context whole).  A missing
    extract is an arg-gate rejection naming the fix, and no fetch happens."""
    db = _make_db(tmp_path)
    model = MockLlmClient()
    tool = _extract_tool(db, model)

    result = await tool.run(queries=[_PAGE_URL])

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

    result = await tool.execute(queries=[_CATALOGUE_URL], extract=_INSTRUCTION)

    stored_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert stored_log is not None
    stored = stored_log.read_all()  # content still stored
    assert result.message == (
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_CATALOGUE_URL}\n"
        "Couldn't extract 'the current bid amount' — no extraction model is configured "
        f"for this browse. Full page content saved to browse-results#{stored[-1].id} — "
        f"read it there for anything more.\n{_LINKS_CLAUSE}"
    )
    assert result.success is False
    assert _CATALOGUE_BODY not in result.message


# ── Whole-render literals: the enumerated micro-result forms (via execute) ─────


async def _execute_extract(tmp_path, name: str, model_content: str):
    db = _make_db(tmp_path, name)
    tool = _extract_tool(db, _responds(model_content))
    result = await tool.execute(queries=[_PAGE_URL], extract=_INSTRUCTION)
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    stored = browse_log.read_all()
    return result, stored[-1].id


@pytest.mark.asyncio
async def test_micro_result_render_forms(tmp_path):
    """The enumerated main-loop render forms, whole-string, as the model reads
    them — each UNDER its page's ``## browse:`` section header (per-page, #1682):
    extracted, not-present, failed-after-reroll (untagged), and poison-then-failed."""
    header = f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n"

    # Success renders the VALUE ALONE — no fetch-handle tail ("saved to
    # browse-results#N" read as the remembering being done on the chat teach
    # round; the failure renders below keep the handle as their remedy).
    result, _handle_id = await _execute_extract(tmp_path, "ok", _TAGGED_VALUE)
    assert result.message == f"{header}{_INSTRUCTION}: {_EXTRACTED_VALUE}"
    assert result.success is True

    result, handle_id = await _execute_extract(tmp_path, "absent", _TAGGED_NOT_PRESENT)
    assert result.message == (
        f"{header}"
        "The page doesn't contain 'the current bid amount' — the page lists no bid amount. "
        f"Full page content saved to browse-results#{handle_id} — read it there for anything more."
    )
    # NOT_PRESENT is a successful read of an absent fact, not a failure.
    assert result.success is True

    result, handle_id = await _execute_extract(tmp_path, "untagged", _UNTAGGED_APOLOGY)
    assert result.message == (
        f"{header}"
        "Couldn't extract 'the current bid amount' from the page — the extractor returned "
        f"nothing usable. Full page content saved to browse-results#{handle_id} — read it "
        "there for anything more."
    )
    assert result.success is False
    # The apology prose was never promoted to a value (the confabulation leak).
    assert _UNTAGGED_APOLOGY not in result.message

    result, handle_id = await _execute_extract(tmp_path, "poison", "...???...")
    assert result.message == (
        f"{header}"
        "Couldn't extract 'the current bid amount' from the page — the extractor output was "
        f"unusable after 3 attempts. Full page content saved to browse-results#{handle_id} — "
        "read it there for anything more."
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
        _page_router({_MARKER_1: _TAGGED_VALUE, _MARKER_2: _TAGGED_VALUE_2})
    )
    tool = _multi_extract_tool(
        db, client, _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2})
    )

    result = await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION)

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
    assert _HANDLE_RE.search(sections[0]) is None
    assert _HANDLE_RE.search(sections[1]) is None
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
async def test_not_present_page_does_not_mask_extracted_page(tmp_path):
    """NOT_PRESENT on page 1 + EXTRACTED on page 2: each renders honestly in its OWN
    section — the not-present page never masks the other page's extracted value, and
    the extracted value never leaks into the not-present section (#1682)."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_1: _TAGGED_NOT_PRESENT, _MARKER_2: _TAGGED_VALUE_2})
    tool = _multi_extract_tool(
        db, model, _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2})
    )

    result = await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION)

    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert len(sections) == 2
    # Page 1 renders the honest not-present form under its own header.
    assert sections[0].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n")
    assert "The page doesn't contain 'the current bid amount'" in sections[0]
    assert _NOT_PRESENT_REASON in sections[0]
    # Page 2's extracted value renders under ITS header — not masked, not leaked.
    assert sections[1].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n")
    assert _EXTRACTED_VALUE_2 in sections[1]
    assert _EXTRACTED_VALUE_2 not in sections[0]
    # Both are successful reads (an extracted value + an honest absence).
    assert result.success is True


@pytest.mark.asyncio
async def test_failed_fetch_section_coexists_with_extracted_section(tmp_path):
    """A failed fetch keeps its ``## browse error:`` section (no micro-context) right
    alongside a successfully-extracted page's section (#1682) — the read failure
    stays visible, and its slot never consumes another page's handle."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_2: _TAGGED_VALUE_2})
    tool = _multi_extract_tool(
        db,
        model,
        _provider_by_url(
            {_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2}, failing=frozenset({_PAGE_URL})
        ),
    )

    result = await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION)

    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert len(sections) == 2
    # The failed fetch keeps its verbatim error section — no extraction ran on it.
    assert sections[0].startswith(f"{PennyConstants.BROWSE_ERROR_HEADER}{_PAGE_URL}")
    assert "Could not read this page" in sections[0]
    assert _HANDLE_RE.search(sections[0]) is None
    # The successful page still extracts, under its own header — value alone.
    assert sections[1].startswith(f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n")
    assert _EXTRACTED_VALUE_2 in sections[1]
    assert _HANDLE_RE.search(sections[1]) is None
    assert result.success is True


@pytest.mark.asyncio
async def test_failed_fetch_in_the_middle_keeps_its_slot_and_its_siblings_handles(tmp_path):
    """A read failure sits in ITS OWN slot and consumes no page's handle — the half of
    the #1682 contract the concurrent batch (#1941) has to keep structurally.

    The failure is in the MIDDLE of three queries, so both claims are testable at
    once: the error section renders between its siblings (page order, not completion
    order), and the last page's NOT_PRESENT render carries the handle of ITS OWN
    stored entry — the second of the two stored, not the first.  The handles are
    bound in section order before any draw is issued, so the un-stored failed fetch
    cannot slide page 3's handle onto page 1."""
    db = _make_db(tmp_path)
    model = _responds_routed({_MARKER_1: _TAGGED_VALUE, _MARKER_3: _TAGGED_NOT_PRESENT})
    tool = _multi_extract_tool(
        db,
        model,
        _provider_by_url(
            {_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2, _PAGE_URL_3: _PAGE_TEXT_3},
            failing=frozenset({_PAGE_URL_2}),
        ),
    )

    result = await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2, _PAGE_URL_3], extract=_INSTRUCTION)

    # Only the two READ pages were stored; the failed fetch stored nothing.
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    stored = browse_log.read_all()
    assert len(stored) == 2
    handles = {
        marker: next(entry.id for entry in stored if marker in entry.content)
        for marker in (_MARKER_1, _MARKER_3)
    }
    assert result.message.split(PennyConstants.SECTION_SEPARATOR) == [
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}",
        f"{PennyConstants.BROWSE_ERROR_HEADER}{_PAGE_URL_2}\n"
        "Could not read this page: page unavailable. Try a different source or a "
        "reworded query; if other queries in this batch succeeded, work from those "
        "instead of retrying this one.",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_3}\n"
        "The page doesn't contain 'the current bid amount' — the page lists no bid "
        f"amount. Full page content saved to browse-results#{handles[_MARKER_3]} — "
        "read it there for anything more.",
    ]
    # Page 3's handle is its OWN entry's (interpolated above); page 1's appears
    # nowhere, so the un-stored failed fetch slid no handle onto its neighbour.
    assert f"browse-results#{handles[_MARKER_1]}" not in result.message
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

    result = await tool.execute(queries=[_SEARCH_QUERY], extract=_INSTRUCTION)

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
    model = _responds_routed({_MARKER_1: _TAGGED_VALUE})
    tool = _multi_extract_tool(
        db,
        model,
        _provider_by_url({_SEARCH_PAGE_URL: _SEARCH_PAGE_TEXT, _PAGE_URL: _PAGE_TEXT}),
    )

    result = await tool.execute(queries=[_SEARCH_QUERY, _PAGE_URL], extract=_INSTRUCTION)

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
    """A page read whose extractor never produces a usable tagged line renders the
    failure sentence AND the page's own standalone links (#2141).

    With ``extract`` set the body never enters the conversation, so a failure saying
    only that it failed left the chat model nothing to open — it re-searched, landed on
    the same page, and looped until the pathology gate stopped it.  The links are the
    one part of the page it can act on, so they come back with the failure: the fetch
    handle remedies the body, the links remedy the next call."""
    db = _make_db(tmp_path)
    model = _responds(_UNTAGGED_APOLOGY)
    tool = _multi_extract_tool(db, model, _provider(_CATALOGUE_TEXT))

    result = await tool.execute(queries=[_CATALOGUE_URL], extract=_INSTRUCTION)

    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    handle_id = browse_log.read_all()[-1].id
    assert result.message == (
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_CATALOGUE_URL}\n"
        "Couldn't extract 'the current bid amount' from the page — the extractor returned "
        f"nothing usable. Full page content saved to browse-results#{handle_id} — read it "
        f"there for anything more.\n{_LINKS_CLAUSE}"
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
        {_MARKER_1: _TAGGED_VALUE, _MARKER_2: _TAGGED_VALUE_2, _MARKER_3: _TAGGED_VALUE_3}
    )
    tool = _multi_extract_tool(
        db,
        cast(Any, probe),
        _provider_by_url(
            {_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2, _PAGE_URL_3: _PAGE_TEXT_3}
        ),
    )

    result = await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2, _PAGE_URL_3], extract=_INSTRUCTION)

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
    probe = _OverlapProbe({_MARKER_1: _TAGGED_VALUE})
    tool = _multi_extract_tool(db, cast(Any, probe), _provider_by_url({_PAGE_URL: _PAGE_TEXT}))

    result = await tool.execute(queries=[_PAGE_URL], extract=_INSTRUCTION)

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
        {_MARKER_1: [poison, poison, _TAGGED_VALUE], _MARKER_2: [_TAGGED_VALUE_2]}
    )
    tool = _multi_extract_tool(
        db, model, _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2})
    )

    result = await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION)

    assert draws[_MARKER_1] == PennyConstants.DEGENERATE_REROLL_ATTEMPTS
    assert draws[_MARKER_2] == 1  # the clean sibling was never re-drawn
    sections = result.message.split(PennyConstants.SECTION_SEPARATOR)
    assert sections == [
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL}\n{_INSTRUCTION}: {_EXTRACTED_VALUE}",
        f"{PennyConstants.BROWSE_PAGE_HEADER}{_PAGE_URL_2}\n{_INSTRUCTION}: {_EXTRACTED_VALUE_2}",
    ]
    assert result.success is True


@pytest.mark.asyncio
async def test_a_raising_draw_ends_the_call_with_every_sibling_awaited(tmp_path):
    """A transport failure inside ONE page's draw ends the whole call with THAT error
    — the behaviour the sequential loop had — and leaves no sibling draw running
    behind it (#1941).

    ``asyncio.gather`` does not cancel siblings when a child raises, so the batch is
    collected with ``return_exceptions=True`` and the earliest failing page re-raised.
    By the time the error leaves the tool every draw has finished, so nothing is still
    spending the backend (and writing its own ledger row) against a call that already
    failed.  The sibling's ``asyncio.sleep(0)`` is a cooperative YIELD, not a wait: it
    gives the loop the chance to resume the failed gather early, which is exactly what
    would leave it orphaned."""
    db = _make_db(tmp_path)
    finished: list[str] = []

    class _RaisesOnTheFirstPage:
        async def chat(self, messages: list[dict], **_kwargs: Any) -> LlmResponse:
            content = messages[-1].get("content", "")
            if _MARKER_1 in content:
                raise LlmTimeoutError("model call timed out")
            await asyncio.sleep(0)
            finished.append(_MARKER_2)
            return LlmResponse(message=LlmMessage(role="assistant", content=_TAGGED_VALUE_2))

    tool = _multi_extract_tool(
        db,
        cast(Any, _RaisesOnTheFirstPage()),
        _provider_by_url({_PAGE_URL: _PAGE_TEXT, _PAGE_URL_2: _PAGE_TEXT_2}),
    )

    with pytest.raises(LlmTimeoutError):
        await tool.execute(queries=[_PAGE_URL, _PAGE_URL_2], extract=_INSTRUCTION)

    assert finished == [_MARKER_2]  # awaited to completion, never orphaned


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


# ── MicroContext unit: tag parse, byte-identity, untagged + poison rerolls ─────


@pytest.mark.asyncio
async def test_micro_context_returns_byte_identical_value_with_attribution():
    """A tagged draw is the extracted value — the payload after the tag, stripped
    once — and the call carries the ledger attribution."""
    model = _responds(f"  {_TAGGED_VALUE}  ")
    result = await MicroContext(cast(Any, model)).extract(
        _PAGE_BODY, _INSTRUCTION, run_target="widget-watch"
    )
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == _EXTRACTED_VALUE
    assert model.requests[0]["agent_name"] == PennyConstants.BROWSE_EXTRACT_AGENT_NAME
    assert model.requests[0]["prompt_type"] == PennyConstants.BROWSE_MICRO_CONTEXT_PROMPT_TYPE
    assert model.requests[0]["run_target"] == "widget-watch"


@pytest.mark.asyncio
async def test_micro_context_not_present_is_enumerated_not_a_value():
    """A ``NOT_PRESENT:`` draw classifies as the enumerated not-present outcome
    carrying the reason — never as an extracted value.  Not-present is a
    successful read of an absent fact, distinct from EXTRACTION_FAILED."""
    model = _responds(_TAGGED_NOT_PRESENT)
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.NOT_PRESENT
    assert result.reason == _NOT_PRESENT_REASON
    assert result.value == ""
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_micro_context_extracts_multi_line_digest_value():
    """A multi-line EXTRACTED value (a bulleted digest, its first item beginning on
    the tag's own line) round-trips WHOLE (#1682): the value is EVERYTHING after the
    tag, so an item-per-line list survives the parse intact into MicroContextResult."""
    digest = "Notable lots:\n- Lot 42: 275 zorkmids\n- Lot 99: 512 zorkmids\n- Lot 7: 3 zorkmids"
    model = _responds(f"{EXTRACTED_TAG} {digest}")
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == digest  # every line preserved, verbatim
    assert result.value.count("\n") == 3  # the four-line digest is intact


@pytest.mark.asyncio
async def test_micro_context_not_present_reason_stays_single_line():
    """A NOT_PRESENT draw takes only its FIRST LINE as the reason (#1682): trailing
    lines (a model that keeps talking) can never be multi-line-promoted into the
    reason, so a not-present apology never smuggles a body value through."""
    model = _responds(f"{NOT_PRESENT_TAG} no bid amount here\nbut lot 42 looks interesting")
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.NOT_PRESENT
    assert result.reason == "no bid amount here"
    assert "lot 42" not in result.reason


@pytest.mark.asyncio
async def test_micro_context_tolerates_a_decorated_tag_line_and_keeps_the_value_verbatim():
    """Tolerance is declared ONCE and applied to the TAG'S OWN LINE only (#1814): a
    draw whose opening tag arrives bolded, list-marked or quoted still parses, while
    the value's later lines are returned untouched — a markdown digest is the value,
    not grammar, so stripping decoration from it would corrupt what was extracted."""
    digest = "**Notable lots:**\n- Lot 42: 275 zorkmids\n- Lot 99: 512 zorkmids"
    model = _responds(f"- **{EXTRACTED_TAG}** {digest}")
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    # Line 1 lost its decoration with the tag; every later line is byte-identical.
    assert result.value == "Notable lots:\n- Lot 42: 275 zorkmids\n- Lot 99: 512 zorkmids"

    quoted = _responds(f'{NOT_PRESENT_TAG} "{_NOT_PRESENT_REASON}"')
    reason = await MicroContext(cast(Any, quoted)).extract(_PAGE_BODY, _INSTRUCTION)
    assert reason.outcome == MicroExtractOutcome.NOT_PRESENT
    assert reason.reason == _NOT_PRESENT_REASON


def test_micro_context_system_prompt_declares_multiline_contract():
    """Whole-render literal of the extraction system prompt (#1682/#1942): the first line
    must OPEN with a tag, an EXTRACTED value may be a multi-line digest / list, a
    NOT_PRESENT reason stays a single line — and the boundary between the two tags is
    STATED: content carrying some of what was asked for is a read, and NOT_PRESENT is for
    content carrying none of it."""
    assert MICRO_CONTEXT_SYSTEM_PROMPT == (
        "You are an extraction step. You are given the full text of one or more web "
        "pages and a single instruction naming exactly what to pull out of them. "
        "The FIRST LINE of your output must open with one of these two tags:\n"
        "EXTRACTED: <the value — it may begin on this same line>\n"
        "NOT_PRESENT: <one short line naming what is missing>\n"
        "After EXTRACTED:, the extracted value is EVERYTHING that follows — as "
        "long as the instruction requires: a single value, one or more paragraphs, or "
        "a list (put one item per line). An instruction often asks for several things "
        "at once. Give whatever the content has for each of them and leave out the "
        "ones it has nothing for: content carrying some of what was asked for is an "
        "EXTRACTED: read. Use "
        "NOT_PRESENT:, on a single line, when the content carries none of what "
        "was asked for. Never invent a value that is not in the content, and write "
        "nothing outside the value itself — no preamble, no explanation, no restating "
        "the instruction."
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
    tool = _extract_tool(db, _responds(_TAGGED_VALUE))
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
async def test_micro_context_untagged_is_rerolled_then_fails():
    """Untagged (but clean) output is a contract violation: discarded and re-drawn on
    the unchanged context for the WHOLE budget — the same patience poison gets, since
    a violation is DETECTED against the declared shape rather than judged — then
    honest EXTRACTION_FAILED.  The apology prose is never promoted to a value, and a
    blank draw takes the same path (no tag to parse)."""
    model = _responds(_UNTAGGED_APOLOGY)
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTION_FAILED
    assert result.value == ""
    assert len(model.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS

    blank = _responds("   ")
    result = await MicroContext(cast(Any, blank)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTION_FAILED
    assert len(blank.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_micro_context_untagged_reroll_can_recover():
    """A reroll re-draws on the unchanged context — a tagged draw recovers the
    extraction, and it can arrive anywhere inside the budget: TWO untagged draws
    still leave a third, and the run costs exactly the draws it took."""
    model = MockLlmClient()
    model.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(
                role="assistant",
                content=_UNTAGGED_APOLOGY if count == 1 else _TAGGED_VALUE,
            )
        )
    )
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTED
    assert result.value == _EXTRACTED_VALUE
    assert len(model.requests) == 2

    late = MockLlmClient()
    late.set_response_handler(
        lambda request, count: LlmResponse(
            message=LlmMessage(
                role="assistant",
                content=_UNTAGGED_APOLOGY if count <= 2 else _TAGGED_VALUE,
            )
        )
    )
    recovered = await MicroContext(cast(Any, late)).extract(_PAGE_BODY, _INSTRUCTION)
    assert recovered.outcome == MicroExtractOutcome.EXTRACTED
    assert recovered.value == _EXTRACTED_VALUE
    assert len(late.requests) == PennyConstants.DEGENERATE_REROLL_ATTEMPTS


@pytest.mark.asyncio
async def test_micro_context_poison_is_discarded_and_rerolled():
    """Poison output (a degeneration collapse) is discarded and re-drawn on the
    unchanged context up to the reroll budget, then fails honestly."""
    model = _responds("...???...")
    result = await MicroContext(cast(Any, model), reroll_attempts=3).extract(
        _PAGE_BODY, _INSTRUCTION
    )
    assert result.outcome == MicroExtractOutcome.POISON_REROLL_FAILED
    assert result.value == ""
    assert len(model.requests) == 3


def test_micro_result_render_by_handle_is_a_typed_id(tmp_path):
    """The fetch handle is a typed ``<memory>#<id>`` anchor (rendered directly) —
    on the FAILURE renders, where it is the remedy.  A successful extraction
    renders the value alone: the old "saved to browse-results#N" tail read as
    the remembering being done at exactly the moment a chat teach round held
    the value (2026-07-19), so success carries no handle clause.

    The page's LINKS follow the same split (#2141): they are handed to every render, and
    only the FAILURE renders append them.  NOT_PRESENT is a successful read of an absent
    fact — the page was fetched and the answer is that the fact isn't there — so there is
    nothing for a link to recover from, and offering one would invite a re-read of a page
    already read."""
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
        MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason="no bid listed."),
        _INSTRUCTION,
        stored,
        links,
    )
    assert body == (
        "The page doesn't contain 'the current bid amount' — no bid listed. "
        "Full page content saved to browse-results#7 — read it there for anything more."
    )
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.POISON_REROLL_FAILED),
        _INSTRUCTION,
        stored,
        links,
    )
    assert body == (
        "Couldn't extract 'the current bid amount' from the page — the extractor output "
        "was unusable after 3 attempts. Full page content saved to browse-results#7 — "
        f"read it there for anything more.\n{_LINKS_CLAUSE}"
    )
