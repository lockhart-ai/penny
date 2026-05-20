"""Validation tests for Pydantic models with required fields.

These models had ``str = ""`` defaults that masked missing-field bugs at
the wire boundary. After tightening the fields to required, the model
must reject incomplete payloads with a ``ValidationError`` instead of
silently producing an instance with empty strings.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from penny.channels.base import PageContext
from penny.channels.browser.models import BrowserIncoming
from penny.channels.discord.models import DiscordUser
from penny.llm.client import LlmClient
from penny.llm.models import LlmToolCall, LlmToolCallFunction
from penny.tools.memory_args import CollectionEntrySpec, CollectionWriteArgs


class TestPageContextRequiresAllFields:
    def test_full_payload_accepted(self) -> None:
        ctx = PageContext(title="Hello", url="https://example.com", text="body")
        assert ctx.title == "Hello"

    @pytest.mark.parametrize("missing", ["title", "url", "text"])
    def test_missing_field_rejected(self, missing: str) -> None:
        payload = {"title": "Hello", "url": "https://example.com", "text": "body"}
        del payload[missing]
        with pytest.raises(ValidationError, match=missing):
            PageContext(**payload)


class TestBrowserIncomingRequiresContentAndSender:
    def test_full_payload_accepted(self) -> None:
        msg = BrowserIncoming(type="message", content="hi", sender="firefox")
        assert msg.content == "hi"

    def test_missing_content_rejected(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            BrowserIncoming(type="message", sender="firefox")  # ty: ignore[missing-argument]

    def test_missing_sender_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sender"):
            BrowserIncoming(type="message", content="hi")  # ty: ignore[missing-argument]


class TestDiscordUserRequiresDiscriminator:
    def test_missing_discriminator_rejected(self) -> None:
        with pytest.raises(ValidationError, match="discriminator"):
            DiscordUser(id="1", username="alice")  # ty: ignore[missing-argument]


class TestLlmToolCallRequiresIdAndFunctionName:
    def test_missing_function_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            LlmToolCallFunction(arguments={})  # ty: ignore[missing-argument]

    def test_missing_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="id"):
            LlmToolCall(function=LlmToolCallFunction(name="search"))  # ty: ignore[missing-argument]

    def test_missing_function_rejected(self) -> None:
        with pytest.raises(ValidationError, match="function"):
            LlmToolCall(id="call_1")  # ty: ignore[missing-argument]


class TestCollectionEntrySpecCoercesStringifiedObjects:
    """CollectionEntrySpec must accept JSON-stringified dicts in addition to plain dicts.

    Some models wrap array entries in outer quotes, producing strings instead of
    objects. The model_validator should parse them back out transparently.
    """

    def test_dict_input_accepted(self) -> None:
        spec = CollectionEntrySpec.model_validate({"key": "foo", "content": "bar"})
        assert spec.key == "foo"
        assert spec.content == "bar"

    def test_json_string_coerced_to_dict(self) -> None:
        entry_json = json.dumps({"key": "tea diplomacy", "content": "https://url"})
        spec = CollectionEntrySpec.model_validate(entry_json)
        assert spec.key == "tea diplomacy"
        assert spec.content == "https://url"

    def test_invalid_string_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            CollectionEntrySpec.model_validate("not a json object")

    def test_collection_write_args_mixed_entries(self) -> None:
        args = CollectionWriteArgs.model_validate(
            {
                "memory": "knowledge",
                "entries": [
                    {"key": "Putin visit", "content": "https://url1"},
                    json.dumps({"key": "Tea diplomacy", "content": "https://url2"}),
                ],
            }
        )
        assert len(args.entries) == 2
        assert args.entries[0].key == "Putin visit"
        assert args.entries[1].key == "Tea diplomacy"
        assert args.entries[1].content == "https://url2"


class TestExtractMalformedArguments:
    """LlmClient._extract_malformed_arguments must recover entries arrays.

    The model sometimes emits ``collection_write`` calls where the first entry
    is a proper object but subsequent ones are unescaped JSON-inside-quotes,
    making the overall JSON unparseable. The fallback extractor should recover
    all key/content pairs via regex.
    """

    def test_queries_array_still_extracted(self) -> None:
        raw = '{"queries": ["what is AI", "machine learning basics"]}'
        result = LlmClient._extract_malformed_arguments(raw)
        assert result == {"queries": ["what is AI", "machine learning basics"]}

    def test_entries_array_with_valid_objects(self) -> None:
        raw = (
            '{"entries": [{"key": "k1", "content": "c1"}, {"key": "k2", "content": "c2"}],'
            ' "memory": "knowledge"}'
        )
        result = LlmClient._extract_malformed_arguments(raw)
        assert result.get("memory") == "knowledge"
        assert result.get("entries") == [
            {"key": "k1", "content": "c1"},
            {"key": "k2", "content": "c2"},
        ]

    def test_entries_array_with_unescaped_stringified_objects(self) -> None:
        # This is the exact malformation from the bug report: entry 2+ are
        # surrounded by outer quotes with unescaped inner quotes, making the
        # raw JSON unparseable by json.loads.
        raw = (
            '{"entries": [{"key": "Putin visit", "content": "https://url1"}, '
            '"{"key": "Tea diplomacy", "content": "https://url2"}", '
            '"{"key": "Samsung news", "content": "https://url3"}"], '
            '"memory": "knowledge"}'
        )
        result = LlmClient._extract_malformed_arguments(raw)
        assert result.get("memory") == "knowledge"
        entries = result.get("entries", [])
        assert len(entries) == 3
        assert entries[0]["key"] == "Putin visit"
        assert entries[1]["key"] == "Tea diplomacy"
        assert entries[2]["key"] == "Samsung news"

    def test_unrecognised_pattern_returns_empty_dict(self) -> None:
        raw = '{"completely": "different", "structure": 42}'
        result = LlmClient._extract_malformed_arguments(raw)
        assert result == {}
