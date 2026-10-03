"""What a round was given, split where its turns still have their structure (#2203).

Every frame below is laid out by PRODUCTION's own render — ``Tool.format_result``,
``format_entries``, ``format_log_timestamp``, the told-date format — so the split is pinned
against what the framework writes and not against a copy of it.  A render that changes shape
fails here, loudly, instead of quietly reading its frame as content.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from penny.constants import PennyConstants
from penny.database.models import MemoryEntry
from penny.datetime_utils import format_log_timestamp
from penny.tests.eval.utils.given import read_given
from penny.tools.base import Tool
from penny.tools.memory_tools import format_entries
from penny.tools.models import ToolResult

_TOLD = datetime(2026, 10, 2, 13, 11, tzinfo=ZoneInfo("America/Los_Angeles"))
_TOLD_LINE = (
    f"{PennyConstants.CURRENT_DATETIME_PREFIX}"
    f"{_TOLD.strftime(PennyConstants.CURRENT_DATETIME_FORMAT)}"
)
_WRITTEN = datetime(2026, 10, 2, 17, 44)
_RAN = datetime(2026, 9, 30, 4, 1)
_ASK = "what is in my recipe box, and when did the watch last run?"


def _entry(key: str | None, content: str, written: datetime = _WRITTEN) -> MemoryEntry:
    return MemoryEntry(
        memory_name="recipe-box", key=key, content=content, author="user", created_at=written
    )


def _call(call_id: str, tool: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [_made(call_id, tool)]}


def _made(call_id: str, tool: str) -> dict:
    return {"id": call_id, "function": {"name": tool, "arguments": "{}"}}


def _result(call_id: str, tool: str, message: str, *, success: bool = True) -> dict:
    framed = Tool.format_result(tool, {"memory": "recipe-box"}, _answer(message, success))
    return {"role": "tool", "tool_call_id": call_id, "content": framed}


def _answer(message: str, success: bool) -> ToolResult:
    return ToolResult(message=message, success=success)


def test_content_is_what_was_stated_and_the_frame_around_it_is_not() -> None:
    """One round, every kind of turn: the prompt is scaffolding apart from the entries it
    lays out, the user's turn is content, and a read's payload is its entries without the
    count line, the list numbers and the stamps the entry render gives them.

    The whole text is untouched — every reader of ``given`` as a string reads what it always
    did — and the stamps come out as the moments they are."""
    recipes = [
        _entry("One-pot orzo", "One-pot orzo — orzo, lemon, spinach, 20 min."),
        _entry("Sheet-pan fajitas", "Sheet-pan fajitas — 25 min at 425F.\nServes 4."),
    ]
    holdings = format_entries([_entry("asking price", "$499", _RAN)])
    prompt = (
        f"{_TOLD_LINE}\n\n1. Read the page.\n2. Then give the answer.\n\n"
        f"## What this collection holds\n{holdings}\n\n- watch — 3 calls"
    )
    read = format_entries(recipes, source="recipe-box", ordering="most recent first")
    one = format_entries(recipes[:1], source="recipe-box")
    log = format_entries([_entry(None, "the first cycle ran clean")], source="runs")
    turns = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": _ASK},
        _call("call-1", "collection_read_latest"),
        _result("call-1", "collection_read_latest", read),
        _call("call-2", "collection_get"),
        _result("call-2", "collection_get", one),
        _call("call-3", "log_read"),
        _result("call-3", "log_read", log),
    ]

    given = read_given(turns)

    assert given == "\n".join(turn["content"] for turn in turns if turn["role"] != "assistant")
    assert given.content == (
        "key='asking price' $499\n"
        f"{_ASK}\n"
        "key='One-pot orzo' One-pot orzo — orzo, lemon, spinach, 20 min.\n"
        "key='Sheet-pan fajitas' Sheet-pan fajitas — 25 min at 425F.\n"
        "Serves 4.\n"
        "key='One-pot orzo' One-pot orzo — orzo, lemon, spinach, 20 min.\n"
        "the first cycle ran clean"
    )
    assert given.moments == (_RAN, _TOLD.replace(tzinfo=None), _WRITTEN)


def test_a_tool_nobody_has_heard_of_is_split_by_the_same_line() -> None:
    """Nothing in the split knows a tool's name.  A result is framed by the one function that
    frames every result, so a plugin's verb is payload after its narration line exactly as a
    built-in is — whether the call worked or failed — and what the tool words into its own
    message, a count included, is its payload taken as it comes.

    A name that leaked Harmony control tokens is paired the way production dispatched it, and
    a turn nobody framed is payload whole."""
    leaked = "ledger_search<|channel|>commentary"
    turns = [
        _call("call-1", "ledger_search"),
        _result("call-1", "ledger_search", "Found 3 invoices:\n- INV-7 for $240"),
        _call("call-2", "ledger_search"),
        _result("call-2", "ledger_search", "No ledger named 'q9'.", success=False),
        _call("call-3", leaked),
        _result("call-3", "ledger_search", ""),
        {"role": "tool", "tool_call_id": "call-9", "content": "an unframed result, 12 lines"},
    ]

    given = read_given(turns)

    assert given.content == (
        "Found 3 invoices:\n- INV-7 for $240\nNo ledger named 'q9'.\n\nan unframed result, 12 lines"
    )
    assert "(ledger_search result)" in given
    assert given.moments == ()


def test_a_stamp_is_a_moment_wherever_the_framework_rendered_it() -> None:
    """The date the round was told and every log stamp are lifted out as moments, so the
    digits of neither are left in the content to be read as a quantity — in a prompt, in a
    document handed to a micro-context, and inside a tool's own payload alike.

    Pinned against production's formats: a told line or a stamp that production would not
    have written is ordinary text and stays where it is."""
    stamp = format_log_timestamp(_WRITTEN)
    document = f"{_TOLD_LINE}\n\nSubject: the quote\nDate: 2026-09-22T15:10:00Z"
    turns = [
        {"role": "user", "content": document},
        _call("call-1", "memory_metadata"),
        _result("call-1", "memory_metadata", f"created: {stamp} by run seeded-setup\nruns: 2"),
        {"role": "user", "content": "it said 2026-13-45 99:99 UTC, which is no time at all"},
    ]

    given = read_given(turns)

    assert given.content == (
        "\nSubject: the quote\nDate: 2026-09-22T15:10:00Z\n"
        "created:  by run seeded-setup\nruns: 2\n"
        "it said 2026-13-45 99:99 UTC, which is no time at all"
    )
    assert given.moments == (_TOLD.replace(tzinfo=None), _WRITTEN)
