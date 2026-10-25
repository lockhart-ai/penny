"""Tests for the raw-template transport (#2829): the render pin and the
parser's pure grammar — both exercised without a network call, since
``penny.llm.raw_template`` is pure functions over message dicts and text.
"""

from __future__ import annotations

import pytest

from penny.llm.models import LlmMessage, LlmToolCall, LlmToolCallFunction, LlmToolParseError
from penny.llm.raw_template import has_vendored_template, parse_completion, render_prompt


class TestHasVendoredTemplate:
    """Structural model-identity dispatch (#2829) — no config knob: a model
    either has a vendored template checked into the repo or it doesn't."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemma-4-26B-A4B-it-QAT-MLX-4bit", True),  # oMLX's own id
            ("google/gemma-4-26b-a4b-it", True),  # OpenRouter's id
            ("gpt-oss:20b", False),
            ("embeddinggemma", False),  # "gemma" alone is not the marker
        ],
    )
    def test_matches_by_model_id_substring(self, model: str, expected: bool) -> None:
        assert has_vendored_template(model) is expected


# A fixed fixture conversation — system + user + an assistant tool call
# carrying passed-back reasoning + its tool result — rendered through the
# REAL vendored template.  Confirmed byte-for-byte against oMLX's own
# ``/v1/chat/completions`` render for an equivalent real request (#2829);
# this pin is what breaks the moment the vendored template file is replaced
# or mis-copied.
_FIXTURE_MESSAGES = [
    LlmMessage(role="system", content="You are Penny."),
    LlmMessage(role="user", content="What's on my calendar?"),
    LlmMessage(
        role="assistant",
        content="",
        thinking="Let me check.",
        tool_calls=[
            LlmToolCall(
                id="call_1",
                function=LlmToolCallFunction(name="get_events", arguments={"day": "today"}),
            )
        ],
    ),
    LlmMessage(role="tool", content="No events today.", tool_call_id="call_1"),
]
_FIXTURE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_events",
            "description": "List events.",
            "parameters": {
                "type": "object",
                "properties": {"day": {"type": "string"}},
                "required": ["day"],
            },
        },
    }
]
_EXPECTED_RENDER = (
    "<bos><|turn>system\n<|think|>\nYou are Penny.<|tool>declaration:get_events"
    '{description:<|"|>List events.<|"|>,parameters:{properties:{day:{type:<|"|>STRING<|"|>}}'
    ',required:[<|"|>day<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>\n<|turn>user\n'
    "What's on my calendar?<turn|>\n<|turn>model\n<|channel>thought\nLet me check.\n<channel|>"
    '<|tool_call>call:get_events{day:<|"|>today<|"|>}<tool_call|><|tool_response>'
    'response:get_events{value:<|"|>No events today.<|"|>}<tool_response|>'
)


class TestRenderPin:
    """The render pin (#2829): byte-for-byte, so a replaced or mis-copied
    vendored template fails here before it ever reaches a real runner."""

    def test_render_matches_byte_for_byte(self) -> None:
        wire = [message.to_log_dict() for message in _FIXTURE_MESSAGES]
        prompt = render_prompt(wire, _FIXTURE_TOOLS, True)
        assert prompt == _EXPECTED_RENDER


class TestParseCompletion:
    """``parse_completion`` reads gemma's own grammar — the thought channel
    (both spellings), its tool-call syntax (including ``<|"|>`` string
    quoting and nested object/array arguments), and a plain reply."""

    def test_a_thought_then_tool_call_parses_both(self) -> None:
        text = (
            "<|channel>thought\nI should look this up.\n<channel|>"
            '<|tool_call>call:collection_list{collection:<|"|>jobs<|"|>,k:5,'
            "active:true}<tool_call|>"
        )
        thinking, calls, content = parse_completion(text)
        assert thinking == "I should look this up."
        assert calls is not None
        assert calls[0].function.name == "collection_list"
        assert calls[0].function.arguments == {"collection": "jobs", "k": 5, "active": True}
        assert content == ""

    def test_omlxs_think_spelling_parses_the_same_as_the_raw_markers(self) -> None:
        """oMLX renders the thought channel as ``<think>...</think>`` text
        (its ``Gemma4OutputParserSession``, selected by model identity and
        applied unconditionally) rather than the raw markers CoreWeave
        returns — the parser accepts both spellings (#2829)."""
        text = "<think>\nJust answer directly.\n</think>\nHere is your answer.<turn|>"
        thinking, calls, content = parse_completion(text)
        assert thinking == "Just answer directly."
        assert calls is None
        assert content == "Here is your answer."

    def test_a_plain_reply_with_no_thought_or_tool_call(self) -> None:
        thinking, calls, content = parse_completion("Sure, here you go.<turn|>")
        assert thinking is None
        assert calls is None
        assert content == "Sure, here you go."

    def test_nested_object_and_array_arguments_and_parallel_calls(self) -> None:
        text = (
            '<|tool_call>call:foo{items:[<|"|>a<|"|>,<|"|>b<|"|>],nested:{x:1,y:3.5}}'
            "<tool_call|><|tool_call>call:bar{n:7}<tool_call|>"
        )
        _thinking, calls, _content = parse_completion(text)
        assert calls is not None
        assert len(calls) == 2
        assert calls[0].function.name == "foo"
        assert calls[0].function.arguments == {"items": ["a", "b"], "nested": {"x": 1, "y": 3.5}}
        assert calls[1].function.name == "bar"
        assert calls[1].function.arguments == {"n": 7}
        # Gemma's own syntax carries no call id at all — Penny synthesises one
        # per call within the response, for the dispatcher's own result-pairing.
        assert calls[0].id != calls[1].id

    def test_a_tool_call_that_opens_but_does_not_parse_raises(self) -> None:
        with pytest.raises(LlmToolParseError):
            parse_completion("<|tool_call>call:foo{bad")

    def test_an_unterminated_string_argument_raises(self) -> None:
        with pytest.raises(LlmToolParseError):
            parse_completion('<|tool_call>call:foo{name:<|"|>oops}<tool_call|>')

    def test_a_tool_call_missing_its_closing_marker_raises(self) -> None:
        with pytest.raises(LlmToolParseError):
            parse_completion("<|tool_call>call:foo{}not the closing marker")
