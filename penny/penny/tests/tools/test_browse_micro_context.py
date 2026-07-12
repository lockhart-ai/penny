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

Fictional pages + deterministic mock model responses throughout.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, select

from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory.objects import render_tool_call
from penny.database.migrate import migrate
from penny.database.models import PromptLog
from penny.llm.client import LlmClient
from penny.llm.models import LlmMessage, LlmResponse
from penny.tests.mocks.llm_patches import MockLlmClient
from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.micro_context import (
    EXTRACTED_TAG,
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


def _make_db(tmp_path, name: str = "test") -> Database:
    """A migrated test DB (so the browse-results log exists)."""
    db = Database(str(tmp_path / f"{name}.db"))
    db.create_tables()
    migrate(str(tmp_path / f"{name}.db"))
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

    # The tool result carries the typed value + a handle, never the page body.
    assert _EXTRACTED_VALUE in result.message
    assert _BODY_PHRASE not in result.message
    assert "Closes Friday" not in result.message
    assert _HANDLE_RE.search(result.message) is not None
    assert result.success is True
    # Byte-identical: the extracted value is the leading block, verbatim (no
    # re-transcription by the parent model).
    assert result.message.split("\n\n")[0] == _EXTRACTED_VALUE

    # The *built context* — what the agent loop appends as the model-facing tool
    # message — also carries the value, not the body.
    framed = Tool.format_result("browse", {"queries": [_PAGE_URL], "extract": _INSTRUCTION}, result)
    assert _EXTRACTED_VALUE in framed
    assert _BODY_PHRASE not in framed

    # The full page content is stored whole in browse-results and retrievable by
    # its handle (the anchor discipline).
    handle_id = int(cast("re.Match[str]", _HANDLE_RE.search(result.message)).group(1))
    browse_log = db.memory(PennyConstants.MEMORY_BROWSE_RESULTS_LOG)
    assert browse_log is not None
    fetched = browse_log.entry_by_id(handle_id)
    assert fetched is not None
    assert _BODY_PHRASE in fetched.content

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
async def test_browse_without_extract_is_unchanged(tmp_path):
    """Chat's browse (no ``extract``) is byte-identical to before: the page body
    returns directly and no micro-context model call is made."""
    db = _make_db(tmp_path)
    model = MockLlmClient()
    tool = _extract_tool(db, model)

    result = await tool.execute(queries=[_PAGE_URL])

    assert PennyConstants.BROWSE_PAGE_HEADER in result.message
    assert _BODY_PHRASE in result.message  # the full page body IS returned
    assert result.success is True
    assert model.requests == []  # no extraction call was made


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
    them — extracted, not-present, failed-after-reroll (untagged), and
    poison-rerolled-then-failed."""
    result, handle_id = await _execute_extract(tmp_path, "ok", _TAGGED_VALUE)
    assert result.message == (
        f"{_EXTRACTED_VALUE}\n\n"
        f"Full page content saved to browse-results#{handle_id} — read it there for anything more."
    )
    assert result.success is True

    result, handle_id = await _execute_extract(tmp_path, "absent", _TAGGED_NOT_PRESENT)
    assert result.message == (
        "The page doesn't contain 'the current bid amount' — the page lists no bid amount. "
        f"Full page content saved to browse-results#{handle_id} — read it there for anything more."
    )
    # NOT_PRESENT is a successful read of an absent fact, not a failure.
    assert result.success is True

    result, handle_id = await _execute_extract(tmp_path, "untagged", _UNTAGGED_APOLOGY)
    assert result.message == (
        "Couldn't extract 'the current bid amount' from the page — the extractor returned "
        f"nothing usable. Full page content saved to browse-results#{handle_id} — read it "
        "there for anything more."
    )
    assert result.success is False
    # The apology prose was never promoted to a value (the confabulation leak).
    assert _UNTAGGED_APOLOGY not in result.message

    result, handle_id = await _execute_extract(tmp_path, "poison", "...???...")
    assert result.message == (
        "Couldn't extract 'the current bid amount' from the page — the extractor output was "
        f"unusable after 3 attempts. Full page content saved to browse-results#{handle_id} — "
        "read it there for anything more."
    )
    assert result.success is False


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
async def test_micro_context_untagged_is_rerolled_once_then_fails():
    """Untagged (but clean) output is a contract violation: one reroll of the
    unchanged context, then honest EXTRACTION_FAILED — the apology prose is never
    promoted to a value.  A blank draw takes the same path (no tag to parse)."""
    model = _responds(_UNTAGGED_APOLOGY)
    result = await MicroContext(cast(Any, model)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTION_FAILED
    assert result.value == ""
    assert len(model.requests) == 2  # the draw + exactly one reroll

    blank = _responds("   ")
    result = await MicroContext(cast(Any, blank)).extract(_PAGE_BODY, _INSTRUCTION)
    assert result.outcome == MicroExtractOutcome.EXTRACTION_FAILED
    assert len(blank.requests) == 2


@pytest.mark.asyncio
async def test_micro_context_untagged_reroll_can_recover():
    """The one untagged reroll re-draws on the unchanged context — a tagged
    second draw recovers the extraction."""
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
    """The fetch handle is a typed ``<memory>#<id>`` anchor (rendered directly)."""
    tool = BrowseTool(max_calls=1, embedding_client=cast(Any, MockLlmClient()))
    body = tool._render_micro_result(
        MicroContextResult(outcome=MicroExtractOutcome.EXTRACTED, value=_EXTRACTED_VALUE),
        _INSTRUCTION,
        [cast(Any, SimpleNamespace(id=7))],
    )
    assert body == (
        f"{_EXTRACTED_VALUE}\n\n"
        "Full page content saved to browse-results#7 — read it there for anything more."
    )
