"""Tests for the reasoning field injected into tool call schemas.

``to_ollama_tool`` adds a ``reasoning`` string property to every tool's schema
— a structured per-call rationale the run record captures for display and safe
re-exposure in logs the model later reads (raw thinking is never fed back).
One carve-out, pinned here: a tool that declares ``reasoning`` in its OWN
``parameters`` (browse) keeps its hand-written description — injection never
overwrites a declaration.

The one observed misuse — a terminal ``done`` called with ONLY ``reasoning``,
displacing the required ``success``/``summary`` — is taught, not restructured:
``done`` keeps the injected param, its description marks the two fields
REQUIRED and says ``reasoning`` alone is never valid, and the shared
invalid-args envelope already names both missing required fields with their
type + description hints for that exact shape (asserted below).
"""

from typing import Any
from unittest.mock import MagicMock

from penny.tools.base import Tool
from penny.tools.browse import BrowseTool
from penny.tools.memory_tools import DoneTool
from penny.tools.models import ToolResult


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

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(message="ok")


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

    def test_done_injected_like_every_tool(self):
        """The terminal done tool gets the injected reasoning param too — the
        schema is uniform; its misuse is taught (description + envelope), not
        restructured.  ``reasoning`` stays optional: required is unchanged."""
        tool = DoneTool()
        schema = tool.to_ollama_tool()
        params = schema["function"]["parameters"]
        assert "reasoning" in params["properties"]
        assert set(params["required"]) == {"success", "summary"}

    def test_tool_declared_reasoning_not_overwritten(self):
        """A tool that declares reasoning in its OWN parameters (browse) keeps
        its hand-written description — injection never overwrites it."""
        tool = BrowseTool(max_calls=3, embedding_client=MagicMock())
        own_description = tool.parameters["properties"]["reasoning"]["description"]
        schema = tool.to_ollama_tool()
        props = schema["function"]["parameters"]["properties"]
        assert props["reasoning"]["description"] == own_description
        assert "inner monologue" not in props["reasoning"]["description"]


class TestDoneOnlyReasoningEnvelope:
    """The observed displacement failure gets an actionable invalid-args envelope."""

    async def test_done_with_only_reasoning_names_both_required_fields(self):
        """A done call carrying ONLY reasoning (the observed invalid-args
        failure — reasoning is stripped before validation, leaving no required
        args) is refused with the shared envelope naming BOTH missing required
        fields plus their type + description hints, and telling the model to
        call done again."""
        result = await DoneTool().run(reasoning="all sources failed, wrapping up")
        assert result.success is False
        # The first-person frame is carried on ``narration`` (#1482); the per-field
        # remedy stays the body verbatim.
        assert result.narration == "You tried to use `done` but the arguments were wrong:"
        message = result.message
        assert "success" in message and "boolean" in message
        assert "summary" in message and "string" in message
        assert "Call done(<valid arguments>) again." in message
