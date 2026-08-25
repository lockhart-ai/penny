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
    """An ``extract`` requested with no model client wired fails visibly (named,
    not a silent body dump) while still storing the content by handle."""
    db = _make_db(tmp_path)
    tool = BrowseTool(
        max_calls=3,
        db=db,
        embedding_client=cast(Any, MockLlmClient()),
        author="widget-watch",
    )
    tool.set_browse_provider(_provider(_PAGE_TEXT))

    result = await tool.execute(queries=[_PAGE_URL], extract=_INSTRUCTION)

    assert result.success is False
    assert "no extraction model is configured" in result.message
    assert _BODY_PHRASE not in result.message
    stored_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert stored_log is not None
    assert stored_log.read_all()  # content still stored


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
    """Whole-render literal of the extraction system prompt (#1682): the first line
    must OPEN with a tag, an EXTRACTED value may be a multi-line digest / list, and a
    NOT_PRESENT reason stays a single line."""
    assert MICRO_CONTEXT_SYSTEM_PROMPT == (
        "You are an extraction step. You are given the full text of one or more web "
        "pages and a single instruction naming exactly what to pull out of them. "
        "The FIRST LINE of your output must open with one of these two tags:\n"
        "EXTRACTED: <the value — it may begin on this same line>\n"
        "NOT_PRESENT: <one short line naming what is missing>\n"
        "After EXTRACTED:, the extracted value is EVERYTHING that follows — as "
        "long as the instruction requires: a single value, one or more paragraphs, or "
        "a list (put one item per line). Use "
        "NOT_PRESENT:, on a single line, when the requested information is not in "
        "the content. Never invent a value that is not in the content, and write "
        "nothing outside the value itself — no preamble, no explanation, no restating "
        "the instruction."
    )


def test_extract_parameter_description_whole_render(db: Database):
    """Whole-render literal of the ``extract`` parameter's description — the ONE line
    of guidance the calling model reads before it writes an instruction (#1838).

    It teaches where the instruction comes FROM: the task's own words, broad allowed.
    "Naming exactly what to pull out" read as *be specific* and was measured inflating
    a plain ask into a stricter one than the page could satisfy — the extractor then
    honestly returned NOT_PRESENT and the round died on a fact that was there.  The
    example of a broad ask is deliberately non-fixture phrasing, so no eval pool lends
    its own words to the surface under test."""
    tool = _extract_tool(db, _responds(_TAGGED_VALUE))
    assert tool.parameters["properties"]["extract"]["description"] == (
        "Optional. One instruction naming what to pull out of the fetched "
        'pages (e.g. "the current bid amount"). Use the task\'s own words '
        "for it — asking for extra details the task never named (an ID, a "
        "date, a second field) makes the read come back empty when the page "
        'lacks them, and a broad ask like "what the notice says" is fine. '
        "When set, the full page content is read in a separate scoped "
        "context and only the extracted value is returned here — the page "
        "body never enters this conversation. Omit to receive the page "
        "content itself."
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
    the value (2026-07-19), so success carries no handle clause."""
    tool = BrowseTool(max_calls=1, embedding_client=cast(Any, MockLlmClient()))
    stored = [cast(Any, SimpleNamespace(id=7))]
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=_EXTRACTED_VALUE),
        _INSTRUCTION,
        stored,
    )
    assert body == f"{_INSTRUCTION}: {_EXTRACTED_VALUE}"
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.NOT_PRESENT, reason="no bid listed."),
        _INSTRUCTION,
        stored,
    )
    assert body == (
        "The page doesn't contain 'the current bid amount' — no bid listed. "
        "Full page content saved to browse-results#7 — read it there for anything more."
    )
