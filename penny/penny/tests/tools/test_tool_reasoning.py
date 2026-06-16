"""Tests for the reasoning field injected into tool call schemas."""

from typing import Any

from penny.tools.base import Tool
from penny.tools.models import ToolOutcome


class _DummyTool(Tool):
    """Minimal tool for testing reasoning injection."""

    name = "dummy"
    description = "A test tool"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "test param"},
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolOutcome:
        return ToolOutcome(message="ok")


class TestToolReasoningSchema:
    """Test that to_ollama_tool() injects a reasoning property."""

    def test_reasoning_property_injected(self):
        """Ollama tool schema includes a reasoning property."""
        tool = _DummyTool()
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert "reasoning" in props
        assert props["reasoning"]["type"] == "string"

    def test_original_properties_preserved(self):
        """Original tool properties are still present alongside reasoning."""
        tool = _DummyTool()
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["description"] == "test param"

    def test_original_parameters_not_mutated(self):
        """Injecting reasoning does not mutate the tool's own parameters dict."""
        tool = _DummyTool()
        tool.to_ollama_tool()
        # The tool's own parameters should NOT have reasoning
        assert "reasoning" not in tool.parameters["properties"]
