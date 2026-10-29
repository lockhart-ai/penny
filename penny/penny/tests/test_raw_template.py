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


# A second fixed fixture (#2837): the SAME assistant step carries both prose and a
# tool call — the shape raw_template.py's own parser already keeps (a reply that
# opens with a sentence before calling a tool stays content on that call's own
# message, see TestParseCompletion below) but the template never had to REPLAY
# until the interim-message feature started keeping that prose on the message
# instead of dropping it.  Pinned separately from ``_FIXTURE_MESSAGES`` above —
# that fixture's own tool-call message still carries empty content, and its pin
# must keep passing unchanged.
_COMBINED_FIXTURE_MESSAGES = [
    LlmMessage(role="system", content="You are Penny."),
    LlmMessage(role="user", content="What is my job doing?"),
    LlmMessage(
        role="assistant",
        content="I'll check on that now.",
        tool_calls=[
            LlmToolCall(
                id="call_1",
                function=LlmToolCallFunction(name="run_get", arguments={"run": "run abc"}),
            )
        ],
    ),
    LlmMessage(role="tool", content="the run finished ok", tool_call_id="call_1"),
]
_COMBINED_FIXTURE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_get",
            "description": "Get a run.",
            "parameters": {
                "type": "object",
                "properties": {"run": {"type": "string"}},
                "required": ["run"],
            },
        },
    }
]
_COMBINED_EXPECTED_RENDER = (
    "<bos><|turn>system\nYou are Penny.<|tool>declaration:run_get"
    '{description:<|"|>Get a run.<|"|>,parameters:{properties:{run:{type:<|"|>STRING<|"|>}}'
    ',required:[<|"|>run<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>\n<|turn>user\n'
    "What is my job doing?<turn|>\n<|turn>model\n"
    "I'll check on that now."
    '<|tool_call>call:run_get{run:<|"|>run abc<|"|>}<tool_call|><|tool_response>'
    'response:run_get{value:<|"|>the run finished ok<|"|>}<tool_response|>'
)


class TestCombinedContentAndToolCallRenderPin:
    """The content-beside-a-call render pin (#2837): the prose must replay in
    GENERATION order — content, then the call, then its response — never after
    the response, and the model turn must stay OPEN (no ``<turn|>``, no
    re-opened ``<|turn>model``) so the next step continues the same turn, exactly
    as the content-empty tool-call shape already does."""

    def test_content_before_a_call_replays_in_generation_order_turn_stays_open(self) -> None:
        wire = [message.to_log_dict() for message in _COMBINED_FIXTURE_MESSAGES]
        prompt = render_prompt(wire, _COMBINED_FIXTURE_TOOLS, False)
        assert prompt == _COMBINED_EXPECTED_RENDER

    def test_existing_content_empty_pin_is_unchanged_by_the_combined_shape_branch(self) -> None:
        """The original pin (content empty on the tool-call message) must still
        render byte-identically — the new branch only engages when content is
        non-empty beside a call."""
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

    def test_call_ids_are_unique_across_separate_responses(self) -> None:
        """A per-response counter (``raw_call_1``, ...) would repeat on every
        turn's first call and collide with history/dedup/``CallLink`` tracing
        that keys on the id (#2829) — the id is minted fresh every call, not
        derived from its position."""
        text = "<|tool_call>call:foo{}<tool_call|>"
        first_calls = parse_completion(text)[1]
        second_calls = parse_completion(text)[1]
        assert first_calls is not None and second_calls is not None
        assert first_calls[0].id != second_calls[0].id

    def test_text_before_a_tool_call_is_kept_as_content_not_rerolled(self) -> None:
        """A reply that opens with a sentence before calling a tool keeps that
        sentence as content (#2829) — the chat path's provider-parsed tool
        calls already tolerate this; treating it as unparsed CALL_AS_TEXT
        here would be a regression against that behaviour."""
        text = (
            'Let me check that for you.<|tool_call>call:get_events{day:<|"|>today<|"|>}<tool_call|>'
        )
        thinking, calls, content = parse_completion(text)
        assert thinking is None
        assert calls is not None
        assert calls[0].function.name == "get_events"
        assert content == "Let me check that for you."

    def test_a_trailing_tool_response_marker_is_stripped_not_hallucinated(self) -> None:
        """After a tool call, the model's own generation prompt ends the turn
        at ``<|tool_response>`` (gemma's eos id 50) rather than ``<turn|>``
        (id 106) — a provider that returns this as literal text, instead of
        consuming it as the stop sequence, must not have it leak into
        content (#2829)."""
        text = "<|tool_call>call:foo{}<tool_call|><|tool_response>"
        _thinking, calls, content = parse_completion(text)
        assert calls is not None
        assert content == ""

    def test_a_trailing_eos_marker_is_stripped(self) -> None:
        thinking, calls, content = parse_completion("Sure thing.<eos>")
        assert thinking is None
        assert calls is None
        assert content == "Sure thing."

    def test_a_tool_call_that_opens_but_does_not_parse_raises(self) -> None:
        with pytest.raises(LlmToolParseError):
            parse_completion("<|tool_call>call:foo{bad")

    def test_an_unterminated_string_argument_raises(self) -> None:
        with pytest.raises(LlmToolParseError):
            parse_completion('<|tool_call>call:foo{name:<|"|>oops}<tool_call|>')

    def test_a_tool_call_missing_its_closing_marker_raises(self) -> None:
        with pytest.raises(LlmToolParseError):
            parse_completion("<|tool_call>call:foo{}not the closing marker")


class TestValueShapesTheTemplateCanEmit:
    """``format_argument`` (the vendored template's own macro) has no branch
    for a null/numeric/exponent value — a bare Python value falls to its
    ``else: {{ argument }}`` and Jinja prints ``str()`` of it, which the
    parser must read back exactly (#2829): confirmed empirically by
    rendering each shape through the real template (``str(None) == "None"``,
    ``str(1e21) == "1e+21"``)."""

    def test_null_argument_value(self) -> None:
        _thinking, calls, _content = parse_completion(
            "<|tool_call>call:foo{value:None}<tool_call|>"
        )
        assert calls is not None
        assert calls[0].function.arguments == {"value": None}

    def test_a_very_large_float_renders_in_exponent_form(self) -> None:
        _thinking, calls, _content = parse_completion(
            "<|tool_call>call:foo{value:1e+21}<tool_call|>"
        )
        assert calls is not None
        assert calls[0].function.arguments == {"value": 1e21}

    def test_a_small_negative_exponent_float(self) -> None:
        _thinking, calls, _content = parse_completion(
            "<|tool_call>call:foo{value:1.5e-10}<tool_call|>"
        )
        assert calls is not None
        assert calls[0].function.arguments == {"value": 1.5e-10}

    def test_a_negative_integer(self) -> None:
        _thinking, calls, _content = parse_completion("<|tool_call>call:foo{value:-7}<tool_call|>")
        assert calls is not None
        assert calls[0].function.arguments == {"value": -7}

    def test_a_string_nested_two_levels_deep(self) -> None:
        """A string inside an object inside an array — the same ``<|"|>``
        escaping recurses at every nesting level (the template's
        ``format_argument`` calls itself for both ``mapping`` and
        ``iterable``)."""
        text = '<|tool_call>call:foo{items:[{name:<|"|>first<|"|>}]}<tool_call|>'
        _thinking, calls, _content = parse_completion(text)
        assert calls is not None
        assert calls[0].function.arguments == {"items": [{"name": "first"}]}
